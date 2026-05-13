from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .. import utils
from ..inference.h2core import (
    H2Prepared,
    fit_h2,
    _has_overlapping_annotations,
    _stack_delete_replicates as _stack_h2,
    _symmetrize_with_design as _sym_h2,
)
from ..inference.jackknife import JackknifeDesign, JackknifeSpec
from ..sumstats.moments import exact_score_z_from_arrays, effective_n_scale
from ..inference.rgcore import (
    RGPrepared,
    InterceptFit,
    RGResultWriter,
    build_manifest_summary_row,
    fit_rg,
    _stack_delete_replicates as _stack_rg,
    _symmetrize_with_design as _sym_rg,
    _make_intercept_keep_mask,
    _build_simple_intercept_weights,
    _compute_intercept_unit_summaries,
    _compute_weighted_intercept_unit_summaries,
    _solve_constrained_intercept_from_sums,
    _full_intercept_denominator,
    _intercept_gamma_total_from_beta,
)
from ..sumstats.sumstats import MatchedSumstats, Sumstats
from ..inference.trace import Trace


@dataclass
class _FastTrait:
    phen: str
    spath: str
    sumstats: Sumstats
    keep: np.ndarray
    z_rg: np.ndarray
    z_filter: np.ndarray
    y_h2: np.ndarray


@dataclass
class _StructUnitStats:
    m: np.ndarray
    Ak: np.ndarray
    Ak2: np.ndarray
    AA: np.ndarray
    AL: np.ndarray


class _FastTraceView:
    def __init__(self, *, nsnps: int, nbins: int, annot_header):
        self._nsnps = int(nsnps)
        self._nbins = int(nbins)
        self.annot_header = annot_header
        self.kmoments = None
        self.kmoments_valid = False

    @property
    def nsnps(self) -> int:
        return self._nsnps

    @property
    def nbins(self) -> int:
        return self._nbins


def _log(log, msg: str):
    if log is not None:
        log._log(msg)


def _write_fast_pair_log(
    pair_prefix: str,
    *,
    phen1: str,
    phen2: str,
    annot_header,
    h2_fit1,
    h2_fit2,
    intercept,
    rg_fit,
    runtime_s: float,
):
    lines: list[str] = []

    def add(msg: str):
        lines.append(msg)

    add(f"[rg:manifest:fast] pair: {phen1} vs {phen2}")

    km_info = getattr(rg_fit, "kmoment_info", None)
    if km_info is not None:
        add(
            "[rg:kmom] "
            f"single-component model-based SE used; "
            f"moment_source={km_info.get('moment_source', 'NA')}, "
            f"alpha_probe={km_info.get('alpha_probe', np.nan):.6g}, "
            f"alpha_probe_err={km_info.get('alpha_probe_err', np.nan):.3e}, "
            f"delta_reff={km_info.get('delta_reff', np.nan):.6g}, "
            f"var_gamma={km_info.get('var_gamma', np.nan):.6g}"
        )

    for name, fit in ((phen1, h2_fit1), (phen2, h2_fit2)):
        if fit.enrich_mode_used:
            add(f"^^^ Phenotype [{name}] enrichment_mode_used: {fit.enrich_mode_used}")

        if len(annot_header) > 1:
            for j, header in enumerate(annot_header):
                line = (
                    f"^^^ Phenotype [{name}] Bin [{header}] "
                    f"sigma_g^2: {fit.sigmas[j, 0]:.6g} (SE: {fit.sigmas[j, 1]:.6g}) "
                    f"h^2_cat: {fit.h2[j, 0]:.6g} (SE: {fit.h2[j, 1]:.6g}) "
                    f"Enrichment: {fit.enrich[j, 0]:.6g} (SE: {fit.enrich[j, 1]:.6g})"
                )
                if fit.enrich_nonoverlap is not None and fit.enrich_overlap is not None:
                    line += (
                        f" Enrichment_nonoverlap: {fit.enrich_nonoverlap[j, 0]:.6g} "
                        f"(SE: {fit.enrich_nonoverlap[j, 1]:.6g})"
                        f" Enrichment_overlap: {fit.enrich_overlap[j, 0]:.6g} "
                        f"(SE: {fit.enrich_overlap[j, 1]:.6g})"
                    )
                if fit.tau is not None and fit.tau_star is not None:
                    line += (
                        f" tau: {fit.tau[j, 0]:.6g} (SE: {fit.tau[j, 1]:.6g})"
                        f" tau_*: {fit.tau_star[j, 0]:.6g} (SE: {fit.tau_star[j, 1]:.6g})"
                    )
                add(line)

        add(
            f"^^^ Phenotype [{name}] Total SNP heritability (h^2): "
            f"{fit.h2[-1, 0]:.6g} SE: {fit.h2[-1, 1]:.6g}"
        )

    add(
        f"^^^ Phenotype [{phen1}] & [{phen2}] "
        f"Intercept (c): {intercept.c[0]:.9g} (SE: {intercept.c[1]:.6g})"
    )

    for j, header in enumerate(annot_header):
        add(
            f"^^^ Phenotype [{phen1}] & [{phen2}] Bin [{header}] "
            f"gamma_g: {rg_fit.gamma[j, 0]:.6g} (SE: {rg_fit.gamma[j, 1]:.6g}) "
            f"rg: {rg_fit.rg[j, 0]:.6g} (SE: {rg_fit.rg[j, 1]:.6g})"
        )

    add(
        f"^^^ Phenotype [{phen1}] & [{phen2}] "
        f"Total genetic covariance (gamma_g): {rg_fit.gamma_total[0]:.6g} "
        f"(SE: {rg_fit.gamma_total[1]:.6g})"
    )
    add(
        f"^^^ Phenotype [{phen1}] & [{phen2}] "
        f"Total genetic correlation (rg): {rg_fit.rg_total[0]:.6g} "
        f"(SE: {rg_fit.rg_total[1]:.6g})"
    )

    end_time = utils._get_time()
    add("Analysis ended at: " + utils._get_timestr(end_time))
    add("run time: " + format(float(runtime_s), ".3f") + " s")
    add("Saved log in " + pair_prefix + ".log")

    with open(pair_prefix + ".log", "w") as fd:
        for line in lines:
            fd.write(line + "\n")


def _full_axis_sumstats_arrays(entry, trace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aligned = entry.aligned
    ss = entry.sumstats
    pos = np.asarray(aligned.pos_on_trace, dtype=np.int64)
    M = int(trace.nsnps)

    beta = np.full(M, np.nan, dtype=np.float64)
    se = np.full(M, np.nan, dtype=np.float64)
    n = np.full(M, np.nan, dtype=np.float64)

    matched = pos >= 0
    if np.any(matched):
        p = pos[matched]
        beta[matched] = ss.beta[p]
        se[matched] = ss.se[p]
        n[matched] = ss.n[p]

    return beta, se, n


def _full_axis_z_array(aligned, trace) -> np.ndarray:
    ss = aligned.sumstats
    pos = np.asarray(aligned.pos_on_trace, dtype=np.int64)
    out = np.full(int(trace.nsnps), np.nan, dtype=np.float64)
    matched = pos >= 0
    if np.any(matched):
        out[matched] = ss.z[pos[matched]]
    return out


def _make_matched_stub(trace, trait: Sumstats, keep: np.ndarray) -> MatchedSumstats:
    # The fast path constructs prepared normal equations directly.  The fitters
    # only need matched metadata, but keep a coherent active SNP stub for writers.
    return MatchedSumstats(
        snps=trace.snps[keep],
        z=np.zeros(int(np.sum(keep)), dtype=np.float64),
        chi2=np.zeros(int(np.sum(keep)), dtype=np.float64),
        beta=np.zeros(int(np.sum(keep)), dtype=np.float64),
        se=np.ones(int(np.sum(keep)), dtype=np.float64),
        n=np.full(int(np.sum(keep)), float(trait.nsamp), dtype=np.float64),
        a1=np.full(int(np.sum(keep)), "", dtype=str),
        a2=np.full(int(np.sum(keep)), "", dtype=str),
        nsamp=float(trait.nsamp),
        n_scale=float(trait.n_scale),
        cov_rank=int(trait.cov_rank),
        cov_rank_source=str(trait.cov_rank_source),
        name=str(trait.name),
        used_summary=None,
        used_top=None,
    )


def _load_fast_trait(
    *,
    spath: str,
    meta: dict,
    shared_trace,
    args,
    cache: dict[str, _FastTrait],
    log,
    verbose_level: int,
) -> _FastTrait:
    cached = cache.get(spath)
    if cached is not None:
        return cached

    cov_rank = meta.get("cov_rank", None)
    phen = meta.get("phen", utils._phen_name_from_path(spath))
    t0 = time.time()

    ss = Sumstats.from_file(
        spath,
        name=phen,
        log=log,
        cov_rank=cov_rank,
        cov_rank_source=("manifest" if cov_rank is not None else None),
        compute_diagnostics=(verbose_level >= 1),
    )
    aligned = ss.align_to_trace(shared_trace)
    keep = np.asarray(
        aligned.keep_mask(chisq_threshold=args.max_chisq, chisq_action=args.chisq_action),
        dtype=bool,
    )

    entry = type("Entry", (), {"aligned": aligned, "sumstats": ss})
    beta, se, n = _full_axis_sumstats_arrays(entry, shared_trace)
    z_filter = _full_axis_z_array(aligned, shared_trace)
    z_h2 = exact_score_z_from_arrays(beta, se, n, nsamp=float(ss.nsamp), cov_rank=0)
    z_rg = exact_score_z_from_arrays(beta, se, n, nsamp=float(ss.nsamp), cov_rank=int(ss.cov_rank))
    y_h2 = z_h2 * z_h2
    y_h2[~np.isfinite(y_h2)] = np.nan

    trait = _FastTrait(
        phen=str(phen),
        spath=str(spath),
        sumstats=ss,
        keep=keep,
        z_rg=z_rg,
        z_filter=z_filter,
        y_h2=y_h2,
    )
    cache[spath] = trait

    if verbose_level >= 1:
        matched_n = int(np.sum(aligned.matched_mask()))
        _log(
            log,
            f"[rg:manifest:fast] cached trait '{phen}' from '{spath}' in {time.time() - t0:.3f}s; "
            f"matched={matched_n}/{shared_trace.nsnps}, kept={int(np.sum(keep))}.",
        )

    return trait


def _manifest_trait_use_counts(manifest_df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest_df.itertuples(index=False):
        paths = (row.sumstats1,) if row.sumstats1 == row.sumstats2 else (row.sumstats1, row.sumstats2)
        for spath in paths:
            counts[spath] = counts.get(spath, 0) + 1
    return counts


def _compute_struct_for_compact_units(
    A: np.ndarray,
    L: np.ndarray,
    active_idx: np.ndarray,
    jk: JackknifeDesign,
    active_mask: np.ndarray,
) -> _StructUnitStats:
    U = int(jk.nunit)
    K = int(A.shape[1])
    active_idx = np.asarray(active_idx, dtype=np.int64)
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_idx.ndim != 1 or active_mask.shape != (active_idx.size,):
        raise ValueError("active_idx/active_mask shape mismatch.")

    out = _StructUnitStats(
        m=np.zeros(U, dtype=np.float64),
        Ak=np.zeros((U, K), dtype=np.float64),
        Ak2=np.zeros((U, K), dtype=np.float64),
        AA=np.zeros((U, K, K), dtype=np.float64),
        AL=np.zeros((U, K, K), dtype=np.float64),
    )
    for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        mu = active_mask[s:e]
        if not np.any(mu):
            continue
        rows = active_idx[s:e] if np.all(mu) else active_idx[s:e][mu]
        Au = A[rows, :]
        Lu = L[rows, :]
        out.m[u] = float(rows.size)
        out.Ak[u] = Au.sum(axis=0, dtype=np.float64)
        out.Ak2[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        out.AA[u] = Au.T @ Au
        out.AL[u] = Au.T @ Lu
    return out


def _compute_ay_for_compact_units(
    A: np.ndarray,
    y: np.ndarray,
    active_idx: np.ndarray,
    jk: JackknifeDesign,
    active_mask: np.ndarray,
) -> np.ndarray:
    U = int(jk.nunit)
    K = int(A.shape[1])
    y = np.asarray(y, dtype=np.float64)
    active_idx = np.asarray(active_idx, dtype=np.int64)
    active_mask = np.asarray(active_mask, dtype=bool)
    if y.shape != (active_idx.size,) or active_mask.shape != (active_idx.size,):
        raise ValueError("y/active_idx/active_mask shape mismatch.")

    out = np.zeros((U, K), dtype=np.float64)
    for u, (s, e) in enumerate(zip(jk.starts, jk.ends)):
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        mu = active_mask[s:e]
        if not np.any(mu):
            continue
        rows = active_idx[s:e] if np.all(mu) else active_idx[s:e][mu]
        yu = y[s:e] if np.all(mu) else y[s:e][mu]
        out[u] = A[rows, :].T @ yu
    return out


def _stack_h2_prepared(
    *,
    fast_tv,
    matched,
    jk,
    active_mask,
    has_overlap,
    struct: _StructUnitStats,
    Ay_unit: np.ndarray,
    n_scale: float,
    summary_y_info: dict,
    y: np.ndarray,
    enrich_mode: str,
    report_tau: bool,
    allow_neg_enr: bool,
    clip_nonfinite_vals: bool,
    jack_mode: str,
) :
    R = int(jk.nrep)
    K = int(fast_tv.nbins)

    M_full = float(struct.m.sum())
    M_rep = _stack_h2(np.array(M_full, dtype=np.float64), struct.m, jk.D).reshape(R + 1)
    Ak_rep = _stack_h2(struct.Ak.sum(axis=0), struct.Ak, jk.D)
    Ay_rep = _stack_h2(Ay_unit.sum(axis=0), Ay_unit, jk.D)
    Ak2_rep = _stack_h2(struct.Ak2.sum(axis=0), struct.Ak2, jk.D)
    AA_rep = _stack_h2(struct.AA.sum(axis=0), struct.AA, jk.D)
    AL_rep = _stack_h2(struct.AL.sum(axis=0), struct.AL, jk.D)

    Ak_full = np.asarray(Ak_rep[-1], dtype=np.float64)
    M_k = Ak_rep[:, :, None]
    M_l = np.broadcast_to(Ak_full[None, :], Ak_rep.shape)[:, None, :]
    trace_KK = utils._calc_trace_from_ld_batch(AL_rep, n_scale, M_k, M_l, delta=None)
    unit_sizes = jk.unit_sizes(active_mask=active_mask, dtype=np.float64)
    trace_KK = _sym_h2(trace_KK, jk, unit_sizes)

    lhs = np.full((R + 1, K + 1, K + 1), n_scale, dtype=np.float64)
    lhs[:, :K, :K] = trace_KK
    lhs[:, K, K] = n_scale

    rhs = np.full((R + 1, K + 1), n_scale, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs[:, :K] = (Ay_rep * n_scale) / Ak_rep
    bad_rhs = (~np.isfinite(rhs[:, :K])) | (~np.isfinite(Ak_rep)) | (Ak_rep <= 0.0)
    rhs[:, :K][bad_rhs] = np.nan

    prepared = H2Prepared(
        trace_view=fast_tv,
        matched=matched,
        jackknife=jk,
        active_mask=active_mask,
        has_overlap=has_overlap,
        unit_sizes=unit_sizes,
        n_scale=float(n_scale),
        summary_y_info=summary_y_info,
        y=y,
        m_unit=struct.m,
        Ak_unit=struct.Ak,
        Ay_unit=Ay_unit,
        Ak2_unit=struct.Ak2,
        AA_unit=struct.AA,
        AL_unit=struct.AL,
        M_rep=M_rep,
        Ak_rep=Ak_rep,
        Ay_rep=Ay_rep,
        Ak2_rep=Ak2_rep,
        AA_rep=AA_rep,
        AL_rep=AL_rep,
        lhs=lhs,
        rhs=rhs,
    )
    return fit_h2(
        prepared,
        enrich_mode=enrich_mode,
        report_tau=report_tau,
        allow_neg_enr=allow_neg_enr,
        clip_nonfinite_vals=clip_nonfinite_vals,
        jack_mode=jack_mode,
        nan_policy=("propagate" if clip_nonfinite_vals else "omit"),
    )


def _stack_rg_prepared(
    *,
    fast_tv,
    matched1,
    matched2,
    jk,
    active_mask,
    struct: _StructUnitStats,
    Ay_unit: np.ndarray,
    n1_scale: float,
    n2_scale: float,
    summary_y_info: dict,
    y: np.ndarray,
):
    R = int(jk.nrep)
    Ak_rep = _stack_rg(struct.Ak.sum(axis=0), struct.Ak, jk.D)
    Ay_rep = _stack_rg(Ay_unit.sum(axis=0), Ay_unit, jk.D)
    AL_rep = _stack_rg(struct.AL.sum(axis=0), struct.AL, jk.D)

    Ak_full = np.asarray(Ak_rep[-1], dtype=np.float64)
    M_k = Ak_rep[:, :, None]
    M_l = np.broadcast_to(Ak_full[None, :], Ak_rep.shape)[:, None, :]
    lhs = utils._calc_rg_trace_from_ld_batch(AL_rep, n1_scale, n2_scale, M_k, M_l)
    unit_sizes = jk.unit_sizes(active_mask=active_mask, dtype=np.float64)
    lhs = _sym_rg(lhs, jk, unit_sizes)

    return RGPrepared(
        trace_view=fast_tv,
        matched1=matched1,
        matched2=matched2,
        jackknife=jk,
        active_mask=active_mask,
        unit_sizes=unit_sizes,
        y=y,
        n1_scale=float(n1_scale),
        n2_scale=float(n2_scale),
        summary_y_info=summary_y_info,
        Ak_unit=struct.Ak,
        Ay_unit=Ay_unit,
        AL_unit=struct.AL,
        Ak_rep=Ak_rep,
        Ay_rep=Ay_rep,
        AL_rep=AL_rep,
        lhs=lhs,
    )


def _fit_score_intercept_scalar_fast(
    *,
    fast_tv,
    matched1,
    matched2,
    jk,
    z1: np.ndarray,
    z2: np.ndarray,
    y: np.ndarray,
    reg_ld: np.ndarray,
    n1_scale: float,
    n2_scale: float,
    h2_fit1,
    h2_fit2,
    intercept_chisq_threshold,
    jack_mode: str,
    nan_policy: str,
    reg_ld_source: str,
    log,
) -> InterceptFit:
    z1 = np.asarray(z1, dtype=np.float64).ravel()
    z2 = np.asarray(z2, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    reg_ld = np.asarray(reg_ld, dtype=np.float64).ravel()
    M = int(fast_tv.nsnps)
    if not (z1.size == z2.size == y.size == reg_ld.size == M):
        raise ValueError("Fast intercept arrays must all live on the pair compact SNP axis.")

    keep, info = _make_intercept_keep_mask(
        z1,
        z2,
        reg_ld,
        y=y,
        weight_ld=reg_ld,
        nsamp_max=max(float(n1_scale), float(n2_scale)),
        chisq_threshold=intercept_chisq_threshold,
        chisq_mode="either",
    )
    if info["n_kept"] <= 1:
        raise RuntimeError("Fast intercept regression has <=1 SNP after filtering.")
    if log is not None and info.get("threshold") is not None:
        tag = " (auto)" if info.get("threshold_mode") == "auto" else ""
        log._log(
            f"[rg:c] intercept chi^2 filter: threshold={info['threshold']:.3f}{tag}, "
            f"mode={info['chisq_mode']}, removed={info['n_removed_chisq']} SNPs, "
            f"kept_after_all={info['n_kept']}."
        )

    R = int(jk.nrep)
    a = np.ones((M, 1), dtype=np.float64)
    x = reg_ld.reshape(-1, 1)
    unit_sizes = jk.unit_sizes(active_mask=keep, dtype=np.float64)
    sqrt_n1n2 = float(np.sqrt(float(n1_scale) * float(n2_scale)))
    if not (np.isfinite(sqrt_n1n2) and sqrt_n1n2 > 0.0):
        raise RuntimeError(f"Invalid n_scale pair for fast intercept: {n1_scale}, {n2_scale}.")

    m_u, t_u, S_u = _compute_intercept_unit_summaries(jk, a, x, y, keep)
    m_full = np.sum(m_u, axis=0, dtype=np.float64)
    t_full = np.sum(t_u, axis=0, dtype=np.float64)
    S_full = np.sum(S_u, axis=0, dtype=np.float64)
    m_rep = _stack_rg(m_full, m_u, jk.D)
    t_rep = _stack_rg(t_full, t_u, jk.D)
    S_rep = _stack_rg(S_full, S_u, jk.D)

    w_score = _build_simple_intercept_weights(reg_ld, keep)
    W_u, XW_u, XXW_u, Sy_u, XWy_u = _compute_weighted_intercept_unit_summaries(jk, x, y, w_score)
    W_rep = _stack_rg(np.sum(W_u, dtype=np.float64), W_u, jk.D)
    XW_rep = _stack_rg(np.sum(XW_u, axis=0, dtype=np.float64), XW_u, jk.D)
    XXW_rep = _stack_rg(np.sum(XXW_u, axis=0, dtype=np.float64), XXW_u, jk.D)
    Sy_rep = _stack_rg(np.sum(Sy_u, dtype=np.float64), Sy_u, jk.D)
    XWy_rep = _stack_rg(np.sum(XWy_u, axis=0, dtype=np.float64), XWy_u, jk.D)

    den0 = _full_intercept_denominator(m_full, S_full, W_rep[-1], XW_rep[-1], XXW_rep[-1])
    denom_floor = 0.0 if not np.isfinite(den0) else max(1e-12 * max(float(den0), 1.0), 0.0)
    c_reps, beta_reps, good = _solve_constrained_intercept_from_sums(
        m_rep,
        S_rep,
        t_rep,
        W_rep,
        XW_rep,
        XXW_rep,
        Sy_rep,
        XWy_rep,
        denom_floor=denom_floor,
    )
    if not bool(good[-1]):
        raise RuntimeError("Fast constrained intercept solve failed on the full sample.")

    c_est, c_se = jk.summarize(
        c_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    c = np.array([float(c_est), float(c_se)], dtype=np.float64)

    b_reps = np.asarray(beta_reps[:, 0], dtype=np.float64)
    b_est, b_se = jk.summarize(
        b_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    gamma_reg_reps = _intercept_gamma_total_from_beta(beta_reps, m_rep, sqrt_n1n2)
    gamma_reg_est, gamma_reg_se = jk.summarize(
        gamma_reg_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )

    h2_tot1 = np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64)
    h2_tot2 = np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rg_reg_reps = gamma_reg_reps / np.sqrt(h2_tot1 * h2_tot2)
    rg_reg_reps[~np.isfinite(rg_reg_reps)] = np.nan
    rg_reg_est, rg_reg_se = jk.summarize(
        rg_reg_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )

    info.update({
        "fixed": False,
        "ld_source": str(reg_ld_source),
        "weight_ld_source": "score-legacy-total-ld",
        "regression_design_mode": str(reg_ld_source),
        "regression_ncoef": 1,
        "weight_mode": "score",
        "summary_y_mode": "beta_se_exact",
        "trait1_n_scale": float(n1_scale),
        "trait2_n_scale": float(n2_scale),
        "irwls_iters": 0,
        "regression_slope_full": float(b_reps[-1]),
        "regression_slope": float(b_est),
        "regression_slope_se": float(b_se),
        "regression_gamma_g_total_full": float(gamma_reg_reps[-1]),
        "regression_gamma_g_total": float(gamma_reg_est),
        "regression_gamma_g_total_se": float(gamma_reg_se),
        "regression_rg_total_full": float(rg_reg_reps[-1]) if np.isfinite(rg_reg_reps[-1]) else np.nan,
        "regression_rg_total": float(rg_reg_est),
        "regression_rg_total_se": float(rg_reg_se),
    })

    if log is not None:
        n_bad = int(np.sum(~np.isfinite(c_reps[:R])))
        log._log(
            f"[rg:c] constrained SCORE-weight intercept (scalar): "
            f"final_c={c[0]:.6g}, bad_reps={n_bad}/{R}"
        )

    return InterceptFit(
        trace_view=fast_tv,
        matched1=matched1,
        matched2=matched2,
        jackknife=jk,
        active_mask=keep,
        unit_sizes=unit_sizes,
        ld=reg_ld,
        y=y,
        c_reps=c_reps,
        c=c,
        info=info,
    )


def dispatch_rg_manifest_fast(args, log, manifest_df, trait_meta, verbose_level: int, *, execution_plan=None):
    if args.align_alleles:
        raise ValueError("--rg-manifest-fast currently does not support --align-alleles.")
    if str(args.rg_se_method).strip().lower() != "jackknife":
        raise ValueError("--rg-manifest-fast currently supports --rg-se-method jackknife only.")
    if args.adjust_delta:
        raise ValueError("--rg-manifest-fast currently does not support --adjust-delta.")
    parsed_write_jack, parsed_write_normeq = utils._parse_verbose_outputs(args.verbose)
    write_jack = bool(args.write_jack) or bool(parsed_write_jack)
    write_normeq_explicit = bool(args.write_normeq) or str(args.verbose).strip().lower() == "normeq"
    if write_normeq_explicit:
        raise ValueError("--rg-manifest-fast currently does not support normeq dumps.")
    if bool(parsed_write_normeq):
        _log(log, "[rg:manifest:fast] verbose requested normal-equation dumps; fast mode ignores that unsupported dump.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jk_spec = JackknifeSpec.parse(args.njack)

    shared_trace = Trace(
        bimpath=args.bim,
        sumpath=None,
        savepath=None,
        log=log,
        ldscores=args.ldscores,
        ldscores_reg=args.ldscores_reg,
        ldscores_reg_w=None,
        annot=args.annot,
        verbose=bool(verbose_level),
        delta=None,
    )
    full_tv = shared_trace.materialize_view()
    A = np.asarray(full_tv.annot, dtype=np.float64, order="C")
    L = np.asarray(full_tv.ldscores, dtype=np.float64, order="C")
    K = int(full_tv.nbins)
    if L.shape != (int(full_tv.nsnps), K):
        raise RuntimeError(f"Unexpected LD-score shape {L.shape}; expected {(int(full_tv.nsnps), K)}.")

    intercept_vals = pd.to_numeric(manifest_df["intercept_rg"], errors="coerce").to_numpy(dtype=np.float64)
    needs_summary_intercept = bool(np.any(~np.isfinite(intercept_vals)))
    reg_ld_full = None
    reg_ld_source = "reg"
    if needs_summary_intercept and K != 1:
        raise RuntimeError(
            "--rg-manifest-fast summary-estimated intercept currently supports only single-component rg. "
            "Use fixed per-row intercept_rg for partitioned rg until the partitioned unconstrained path is derived."
        )
    if needs_summary_intercept and str(args.intercept_weight_mode).strip().lower() != "score":
        raise RuntimeError(
            "--rg-manifest-fast summary-estimated intercept currently requires --intercept-weight-mode score."
        )

    if needs_summary_intercept and getattr(full_tv, "ldscores_reg", None) is not None:
        reg = np.asarray(full_tv.ldscores_reg, dtype=np.float64, order="C")
        if reg.ndim != 2 or reg.shape[0] != int(full_tv.nsnps):
            raise RuntimeError("ldscores_reg must share the primary SNP axis.")
        if reg.shape[1] != 1:
            raise RuntimeError(
                "--rg-manifest-fast summary-estimated intercept currently requires single-component --ldscores-reg."
            )
        reg_ld_full = np.asarray(reg[:, 0], dtype=np.float64)
        reg_ld_source = "reg"
    elif needs_summary_intercept and K == 1:
        reg_ld_full = np.asarray(L[:, 0], dtype=np.float64)
        reg_ld_source = "main"

    full_jk = None
    if jk_spec.mode == "chr":
        full_jk = JackknifeDesign.from_trace_view(full_tv, jk_spec, log=log)

    order = list(range(int(manifest_df.shape[0]))) if execution_plan is None else [int(i) for i in execution_plan]
    trait_cache: dict[str, _FastTrait] = {}
    remaining_uses = _manifest_trait_use_counts(manifest_df)
    results = [None] * int(manifest_df.shape[0])
    t0_all = time.time()
    completed = 0

    for row_pos in order:
        row = manifest_df.iloc[int(row_pos)]
        t0_pair = time.time()
        tr1 = _load_fast_trait(
            spath=row.sumstats1,
            meta=trait_meta[row.sumstats1],
            shared_trace=shared_trace,
            args=args,
            cache=trait_cache,
            log=log,
            verbose_level=verbose_level,
        )
        tr2 = _load_fast_trait(
            spath=row.sumstats2,
            meta=trait_meta[row.sumstats2],
            shared_trace=shared_trace,
            args=args,
            cache=trait_cache,
            log=log,
            verbose_level=verbose_level,
        )

        pair_keep = np.asarray(tr1.keep & tr2.keep, dtype=bool)
        active_idx = np.flatnonzero(pair_keep).astype(np.int64)
        active_n = int(active_idx.size)
        if active_n <= 0:
            raise RuntimeError(f"No SNPs remain for pair {row.phen1} vs {row.phen2}.")

        fast_tv = _FastTraceView(nsnps=active_n, nbins=K, annot_header=full_tv.annot_header)
        if jk_spec.mode == "chr":
            jk = full_jk.subset(pair_keep, log=log)
        else:
            jk = JackknifeDesign.from_trace_view(fast_tv, jk_spec, log=log)

        matched1 = _make_matched_stub(shared_trace, tr1.sumstats, pair_keep)
        matched2 = _make_matched_stub(shared_trace, tr2.sumstats, pair_keep)

        y_h2_1 = np.asarray(tr1.y_h2[active_idx], dtype=np.float64)
        y_h2_2 = np.asarray(tr2.y_h2[active_idx], dtype=np.float64)
        h2_active1 = np.isfinite(y_h2_1)
        h2_active2 = np.isfinite(y_h2_2)
        if not np.any(h2_active1) or not np.any(h2_active2):
            raise RuntimeError(f"No finite h2 summary moments remain for pair {row.phen1} vs {row.phen2}.")

        h2_info1 = {
            "mode": "beta_se_exact",
            "cov_rank": 0,
            "cov_rank_source": "forced0_no_covrank_h2",
            "n_scale": float(effective_n_scale(tr1.sumstats.nsamp, 0)),
            "n_nonfinite": int(active_n - np.sum(h2_active1)),
        }
        h2_info2 = {
            "mode": "beta_se_exact",
            "cov_rank": 0,
            "cov_rank_source": "forced0_no_covrank_h2",
            "n_scale": float(effective_n_scale(tr2.sumstats.nsamp, 0)),
            "n_nonfinite": int(active_n - np.sum(h2_active2)),
        }

        h2_full_mask1 = np.zeros(int(full_tv.nsnps), dtype=bool)
        h2_full_mask2 = np.zeros(int(full_tv.nsnps), dtype=bool)
        h2_full_mask1[active_idx[h2_active1]] = True
        h2_full_mask2[active_idx[h2_active2]] = True
        h2_fit1 = _stack_h2_prepared(
            fast_tv=fast_tv,
            matched=matched1,
            jk=jk,
            active_mask=h2_active1,
            has_overlap=_has_overlapping_annotations(A, active_mask=h2_full_mask1),
            struct=_compute_struct_for_compact_units(A, L, active_idx, jk, h2_active1),
            Ay_unit=_compute_ay_for_compact_units(A, y_h2_1, active_idx, jk, h2_active1),
            n_scale=h2_info1["n_scale"],
            summary_y_info=h2_info1,
            y=np.array([], dtype=np.float64),
            enrich_mode=args.enrich_mode,
            report_tau=True,
            allow_neg_enr=args.allow_neg_enr,
            clip_nonfinite_vals=args.clip_nonfinite_vals,
            jack_mode=args.jack_mode,
        )
        h2_fit2 = _stack_h2_prepared(
            fast_tv=fast_tv,
            matched=matched2,
            jk=jk,
            active_mask=h2_active2,
            has_overlap=_has_overlapping_annotations(A, active_mask=h2_full_mask2),
            struct=_compute_struct_for_compact_units(A, L, active_idx, jk, h2_active2),
            Ay_unit=_compute_ay_for_compact_units(A, y_h2_2, active_idx, jk, h2_active2),
            n_scale=h2_info2["n_scale"],
            summary_y_info=h2_info2,
            y=np.array([], dtype=np.float64),
            enrich_mode=args.enrich_mode,
            report_tau=True,
            allow_neg_enr=args.allow_neg_enr,
            clip_nonfinite_vals=args.clip_nonfinite_vals,
            jack_mode=args.jack_mode,
        )

        z_rg1 = np.asarray(tr1.z_rg[active_idx], dtype=np.float64)
        z_rg2 = np.asarray(tr2.z_rg[active_idx], dtype=np.float64)
        y_rg = z_rg1 * z_rg2
        rg_active = np.isfinite(y_rg)
        if not np.any(rg_active):
            raise RuntimeError(f"No finite rg summary moments remain for pair {row.phen1} vs {row.phen2}.")

        rg_struct = _compute_struct_for_compact_units(A, L, active_idx, jk, rg_active)
        rg_info = {
            "mode": "beta_se_exact",
            "trait1_cov_rank": int(tr1.sumstats.cov_rank),
            "trait1_cov_rank_source": str(tr1.sumstats.cov_rank_source),
            "trait1_n_scale": float(tr1.sumstats.n_scale),
            "trait2_cov_rank": int(tr2.sumstats.cov_rank),
            "trait2_cov_rank_source": str(tr2.sumstats.cov_rank_source),
            "trait2_n_scale": float(tr2.sumstats.n_scale),
            "n_nonfinite": int(active_n - np.sum(rg_active)),
        }
        rg_prepared = _stack_rg_prepared(
            fast_tv=fast_tv,
            matched1=matched1,
            matched2=matched2,
            jk=jk,
            active_mask=rg_active,
            struct=rg_struct,
            Ay_unit=_compute_ay_for_compact_units(A, y_rg, active_idx, jk, rg_active),
            n1_scale=float(tr1.sumstats.n_scale),
            n2_scale=float(tr2.sumstats.n_scale),
            summary_y_info=rg_info,
            y=np.array([], dtype=np.float64),
        )

        row_intercept = float(row.intercept_rg) if pd.notna(row.intercept_rg) else np.nan
        if np.isfinite(row_intercept):
            intercept = InterceptFit(
                trace_view=fast_tv,
                matched1=matched1,
                matched2=matched2,
                jackknife=jk,
                active_mask=rg_active,
                unit_sizes=jk.unit_sizes(active_mask=rg_active, dtype=np.float64),
                ld=np.array([], dtype=np.float64),
                y=y_rg,
                c_reps=np.full(jk.nrep + 1, row_intercept, dtype=np.float64),
                c=np.array([row_intercept, 0.0], dtype=np.float64),
                info={
                    "fixed": True,
                    "source": "manifest",
                    "summary_y_mode": "beta_se_exact",
                    "trait1_n_scale": float(tr1.sumstats.n_scale),
                    "trait2_n_scale": float(tr2.sumstats.n_scale),
                },
            )
        else:
            if reg_ld_full is None:
                raise RuntimeError("No single-component regression LD score is available for intercept estimation.")
            intercept = _fit_score_intercept_scalar_fast(
                fast_tv=fast_tv,
                matched1=matched1,
                matched2=matched2,
                jk=jk,
                z1=np.asarray(tr1.z_filter[active_idx], dtype=np.float64),
                z2=np.asarray(tr2.z_filter[active_idx], dtype=np.float64),
                y=y_rg,
                reg_ld=np.asarray(reg_ld_full[active_idx], dtype=np.float64),
                n1_scale=float(tr1.sumstats.n_scale),
                n2_scale=float(tr2.sumstats.n_scale),
                h2_fit1=h2_fit1,
                h2_fit2=h2_fit2,
                intercept_chisq_threshold=args.intercept_chisq_thr,
                jack_mode=args.jack_mode,
                nan_policy=("propagate" if args.clip_nonfinite_vals else "omit"),
                reg_ld_source=reg_ld_source,
                log=log,
            )

        rg_fit = fit_rg(
            rg_prepared,
            h2_fit1,
            h2_fit2,
            intercept,
            rg_se_method=args.rg_se_method,
            jack_mode=args.jack_mode,
            nan_policy=("propagate" if args.clip_nonfinite_vals else "omit"),
        )

        pair_prefix = str(outdir / row.out_stem)
        _write_fast_pair_log(
            pair_prefix,
            phen1=row.phen1,
            phen2=row.phen2,
            annot_header=fast_tv.annot_header,
            h2_fit1=h2_fit1,
            h2_fit2=h2_fit2,
            intercept=intercept,
            rg_fit=rg_fit,
            runtime_s=(time.time() - t0_pair),
        )
        if write_jack:
            jack_path = pair_prefix + ".rg.jack"
            RGResultWriter.save_jackknife_text(rg_fit, jack_path)
            _log(log, f"[rg:manifest:fast] saved rg jackknife replicate dump to {jack_path}")

        results[int(row_pos)] = build_manifest_summary_row(
            phen1=row.phen1,
            phen2=row.phen2,
            sumstats1=row.sumstats1,
            sumstats2=row.sumstats2,
            cov_rank1=row.cov_rank1,
            cov_rank2=row.cov_rank2,
            intercept_rg_input=row.intercept_rg,
            out_prefix=pair_prefix,
            n_snps=active_n,
            annot_header=fast_tv.annot_header,
            h2_fit1=h2_fit1,
            h2_fit2=h2_fit2,
            intercept=intercept,
            rg_fit=rg_fit,
        )
        completed += 1
        if completed == 1 or completed % 25 == 0 or completed == int(manifest_df.shape[0]):
            _log(
                log,
                f"[rg:manifest:fast] completed {completed}/{manifest_df.shape[0]} pair(s); "
                f"latest {row.phen1} vs {row.phen2}: rg={float(rg_fit.rg_total[0]):.6g} "
                f"(SE {float(rg_fit.rg_total[1]):.6g})."
            )

        paths = (row.sumstats1,) if row.sumstats1 == row.sumstats2 else (row.sumstats1, row.sumstats2)
        for spath in paths:
            remaining_uses[spath] = int(remaining_uses.get(spath, 0)) - 1
            if remaining_uses[spath] <= 0:
                evicted = trait_cache.pop(spath, None)
                if evicted is not None and verbose_level >= 1:
                    _log(log, f"[rg:manifest:fast] evicted trait '{evicted.phen}' from cache after its final pair.")

    out = pd.DataFrame(results)
    summary_path = outdir / "manifest.results.tsv"
    out.to_csv(summary_path, sep="\t", index=False)
    _log(log, f"[rg:manifest:fast] saved batch summary to {summary_path}")
    _log(log, f"[rg:manifest:fast] total fast manifest runtime after Trace load: {time.time() - t0_all:.3f}s.")
