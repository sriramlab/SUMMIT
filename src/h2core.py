from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import utils


@dataclass(frozen=True)
class H2Prepared:
    trace_view: object
    matched: object
    jackknife: object
    active_mask: np.ndarray
    has_overlap: bool
    unit_sizes: np.ndarray
    m_unit: np.ndarray                 # (U,)
    Ak_unit: np.ndarray                # (U,K)
    Az2_unit: np.ndarray               # (U,K)
    Ak2_unit: np.ndarray               # (U,K)
    AA_unit: np.ndarray                # (U,K,K)
    AL_unit: np.ndarray                # (U,K,K)
    M_rep: np.ndarray                  # (R+1,)
    Ak_rep: np.ndarray                 # (R+1,K)
    Az2_rep: np.ndarray                # (R+1,K)
    Ak2_rep: np.ndarray                # (R+1,K)
    AA_rep: np.ndarray                 # (R+1,K,K)
    AL_rep: np.ndarray                 # (R+1,K,K)
    lhs: np.ndarray                    # (R+1,K+1,K+1)
    rhs: np.ndarray                    # (R+1,K+1)


@dataclass(frozen=True)
class H2Fit:
    prepared: H2Prepared
    sigma_reps: np.ndarray                 # (R+1,K+2) = sigma_g bins, sigma_e, sum_g
    h2_reps: np.ndarray                    # (R+1,K+1) = h2_cat bins, total h2
    enrich_reps: np.ndarray                # (R+1,K)
    enrich_overlap_reps: np.ndarray | None
    enrich_nonoverlap_reps: np.ndarray | None
    tau_reps: np.ndarray | None
    tau_star_reps: np.ndarray | None
    sigmas: np.ndarray                     # (K+2,2)
    h2: np.ndarray                         # (K+1,2)
    enrich: np.ndarray                     # (K,2)
    enrich_overlap: np.ndarray | None
    enrich_nonoverlap: np.ndarray | None
    tau: np.ndarray | None
    tau_star: np.ndarray | None
    enrich_mode_requested: str
    enrich_mode_used: str


class H2ResultWriter:
    @staticmethod
    def save_jackknife_text(fit: H2Fit, path: str):
        p = fit.prepared
        R = p.jackknife.nrep
        K = p.trace_view.nbins
        with open(path, "w") as fd:
            fd.write("# replicate\t" + "\t".join([f"sigma_g_{k}" for k in range(K)]) + "\t"
                     + "sigma_e\tsum_g\t"
                     + "\t".join([f"h2_cat_{k}" for k in range(K)]) + "\ttotal_h2\n")
            for r in range(R + 1):
                rep_label = "full" if r == R else str(r)
                vals = np.concatenate([fit.sigma_reps[r], fit.h2_reps[r]])
                fd.write(rep_label + "\t" + "\t".join(f"{x:.10g}" for x in vals) + "\n")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _normalize_enrich_mode(mode: str | None) -> str:
    if mode is None:
        return "auto"
    s = str(mode).strip().lower()
    if s in ("auto", "overlap", "non-overlap", "nonoverlap", "both"):
        return "non-overlap" if s == "nonoverlap" else s
    raise ValueError("enrich_mode must be one of {'auto','overlap','non-overlap','both'}")


def _has_overlapping_annotations(A: np.ndarray) -> bool:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] <= 1:
        return False
    return bool(np.any(np.sum(np.abs(A) > 0.0, axis=1) > 1))


def _solve_linear_batch(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.ndim != 3 or rhs.ndim != 2:
        raise ValueError("lhs must be (R+1,p,p) and rhs must be (R+1,p)")
    if lhs.shape[0] != rhs.shape[0] or lhs.shape[1] != lhs.shape[2] or lhs.shape[1] != rhs.shape[1]:
        raise ValueError("lhs/rhs shape mismatch")

    out = np.full(rhs.shape, np.nan, dtype=np.float64)
    for i in range(lhs.shape[0]):
        A = lhs[i]
        b = rhs[i]
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            continue
        try:
            out[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                out[i] = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
    return out


def _symmetrize_with_design(trace_KK: np.ndarray, jackknife, unit_sizes: np.ndarray) -> np.ndarray:
    """
    Symmetrize replicate trace matrices using the current jackknife design.

    Prefer the newer utils API that takes the explicit delete-incidence matrix and
    per-unit active sizes. Fall back to the older shorthand API for backward
    compatibility.
    """
    unit_sizes = np.asarray(unit_sizes, dtype=np.float64)
    try:
        return utils.symmetrize_trace_with_jackknife(
            trace_KK,
            logger=None,
            verbose=False,
            jk_delete_matrix=np.asarray(jackknife.D, dtype=np.float64, order="C"),
            jk_unit_sizes=unit_sizes,
            jk_delete_d=int(jackknife.delete),
        )
    except TypeError:
        if jackknife.mode == "block":
            return utils.symmetrize_trace_with_jackknife(
                trace_KK,
                logger=None,
                verbose=False,
                jk_block_sizes=unit_sizes,
                jk_delete_d=1,
            )
        return utils.symmetrize_trace_with_jackknife(
            trace_KK,
            logger=None,
            verbose=False,
            jk_n_units=int(jackknife.nunit),
            jk_delete_d=int(jackknife.delete),
        )


def _stack_delete_replicates(full: np.ndarray, unit: np.ndarray, D: np.ndarray) -> np.ndarray:
    full = np.asarray(full, dtype=np.float64)
    unit = np.asarray(unit, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if unit.ndim < 1:
        raise ValueError("unit must have at least one dimension")
    if unit.shape[0] != D.shape[1]:
        raise ValueError("unit first axis must match D.shape[1]")

    tail_shape = unit.shape[1:]
    flat_unit = unit.reshape(unit.shape[0], -1)           # (U,P)
    flat_full = full.reshape(-1)                          # (P,)
    deleted = D @ flat_unit                               # (R,P)
    reps = flat_full[None, :] - deleted                   # (R,P)
    reps = reps.reshape((D.shape[0],) + tail_shape)
    return np.concatenate([reps, full[None, ...]], axis=0)


def _pair_correction_from_deleted_mass(A_keep, A_del, delta):
    A_keep = np.asarray(A_keep, dtype=np.float64)
    A_del = np.asarray(A_del, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    K = int(A_keep.shape[1])

    if d.ndim == 0:
        return (A_keep[:, :, None] * A_del[:, None, :]) * float(d)
    if d.ndim == 1:
        if d.size != K:
            raise ValueError(f"delta vector has length {d.size}, expected K={K}")
        return A_keep[:, :, None] * (A_del * d[None, :])[:, None, :]
    if d.ndim == 2:
        if d.shape != (K, K):
            raise ValueError(f"delta matrix has shape {d.shape}, expected ({K},{K})")
        return (A_keep[:, :, None] * A_del[:, None, :]) * d[None, :, :]
    raise ValueError("delta must be scalar, (K,), or (K,K)")


# -----------------------------------------------------------------------------
# preparation
# -----------------------------------------------------------------------------


def prepare_h2(
    trace_view,
    matched,
    jackknife,
    *,
    active_mask=None,
    ld_kind: str = "main",
    adjust_delta: bool = False,
):
    if trace_view.nsnps != matched.nsnps:
        raise ValueError("TraceView and MatchedSumstats must have the same number of SNPs.")
    if not np.array_equal(trace_view.snps, matched.snps):
        raise ValueError("TraceView and MatchedSumstats SNP order mismatch.")
    if jackknife.nsnps != trace_view.nsnps:
        raise ValueError(
            "JackknifeDesign was not built on this TraceView SNP axis. "
            f"Expected {jackknife.nsnps}, got {trace_view.nsnps}."
        )

    M = trace_view.nsnps
    K = trace_view.nbins
    R = jackknife.nrep
    U = jackknife.nunit

    if active_mask is None:
        active_mask = np.ones(M, dtype=bool)
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.ndim != 1 or active_mask.size != M:
        raise ValueError(f"active_mask must be length {M}; got {active_mask.shape}")

    A = np.asarray(trace_view.annot, dtype=np.float64, order="C")
    if ld_kind == "main":
        L = np.asarray(trace_view.ldscores, dtype=np.float64, order="C")
    elif ld_kind == "reg":
        if trace_view.ldscores_reg is None:
            raise ValueError("ld_kind='reg' requested but TraceView has no ldscores_reg")
        L = np.asarray(trace_view.ldscores_reg, dtype=np.float64, order="C")
        if L.shape[1] != K:
            raise ValueError(
                f"For univariate prepare_h2, regression LD bins must match annotation bins. "
                f"Got L.shape[1]={L.shape[1]}, K={K}."
            )
    else:
        raise ValueError("ld_kind must be 'main' or 'reg'")

    chi2 = np.asarray(matched.chi2, dtype=np.float64)
    if chi2.shape != (M,):
        raise ValueError(f"matched.chi2 must have shape ({M},), got {chi2.shape}")

    m_unit = np.zeros(U, dtype=np.float64)
    Ak_unit = np.zeros((U, K), dtype=np.float64)
    Az2_unit = np.zeros((U, K), dtype=np.float64)
    Ak2_unit = np.zeros((U, K), dtype=np.float64)
    AA_unit = np.zeros((U, K, K), dtype=np.float64)
    AL_unit = np.zeros((U, K, K), dtype=np.float64)

    for u in range(U):
        s = int(jackknife.starts[u])
        e = int(jackknife.ends[u])
        if e <= s:
            continue
        mu = active_mask[s:e]
        if not np.any(mu):
            continue
        Au = A[s:e, :][mu, :]
        Lu = L[s:e, :][mu, :]
        chi2u = chi2[s:e][mu]

        m_unit[u] = float(Au.shape[0])
        Ak_unit[u] = Au.sum(axis=0, dtype=np.float64)
        Az2_unit[u] = Au.T @ chi2u
        Ak2_unit[u] = (Au * Au).sum(axis=0, dtype=np.float64)
        AA_unit[u] = Au.T @ Au
        AL_unit[u] = Au.T @ Lu

    has_overlap = _has_overlapping_annotations(A[active_mask, :])

    M_full = float(m_unit.sum())
    if not (np.isfinite(M_full) and M_full > 0.0):
        raise RuntimeError("No active SNPs remain for H2 preparation.")

    M_rep = _stack_delete_replicates(np.array(M_full, dtype=np.float64), m_unit, jackknife.D).reshape(R + 1)
    Ak_rep = _stack_delete_replicates(Ak_unit.sum(axis=0), Ak_unit, jackknife.D)
    Az2_rep = _stack_delete_replicates(Az2_unit.sum(axis=0), Az2_unit, jackknife.D)
    Ak2_rep = _stack_delete_replicates(Ak2_unit.sum(axis=0), Ak2_unit, jackknife.D)
    AA_rep = _stack_delete_replicates(AA_unit.sum(axis=0), AA_unit, jackknife.D)
    AL_rep = _stack_delete_replicates(AL_unit.sum(axis=0), AL_unit, jackknife.D)

    if jackknife.mode == "chr" and adjust_delta and getattr(trace_view, "delta", None) is not None:
        delta = np.asarray(trace_view.delta, dtype=np.float64)
        del_Ak = jackknife.D @ Ak_unit
        rep_Ak = Ak_rep[:R]
        AL_rep = AL_rep.copy()
        AL_rep[:R] -= _pair_correction_from_deleted_mass(rep_Ak, del_Ak, delta)

    Ak_full = np.asarray(Ak_rep[-1], dtype=np.float64)
    src_mass = np.broadcast_to(Ak_full[None, :], Ak_rep.shape).copy()
    if jackknife.mode == "chr" and adjust_delta and getattr(trace_view, "delta", None) is not None:
        # After explicit deleted-source correction, replicate numerators approximate
        # A_keep^T L_keep, so the source-side normalizer should also be the kept mass.
        src_mass[:R] = Ak_rep[:R]

    M_k = Ak_rep[:, :, None]
    M_l = src_mass[:, None, :]
    delta = np.asarray(trace_view.delta, dtype=np.float64) if (adjust_delta and getattr(trace_view, "delta", None) is not None) else None
    trace_KK = utils._calc_trace_from_ld_batch(AL_rep, matched.nsamp, M_k, M_l, delta=delta)

    trace_KK = _symmetrize_with_design(trace_KK, jackknife, m_unit)

    lhs = np.full((R + 1, K + 1, K + 1), float(matched.nsamp), dtype=np.float64)
    lhs[:, :K, :K] = trace_KK
    lhs[:, K, K] = float(matched.nsamp - 1.0)

    rhs = np.full((R + 1, K + 1), float(matched.nsamp - 1.0), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs[:, :K] = (Az2_rep * float(matched.nsamp)) / Ak_rep
    bad_rhs = (~np.isfinite(rhs[:, :K])) | (~np.isfinite(Ak_rep)) | (Ak_rep <= 0.0)
    rhs[:, :K][bad_rhs] = np.nan

    bad_bins = np.flatnonzero(~np.isfinite(Ak_rep[-1]) | (Ak_rep[-1] <= 0.0))
    if bad_bins.size:
        raise RuntimeError(
            "One or more full-sample bins have non-positive total weight after matching. "
            f"Bad bins: {bad_bins.tolist()}"
        )

    return H2Prepared(
        trace_view=trace_view,
        matched=matched,
        jackknife=jackknife,
        active_mask=active_mask,
        has_overlap=has_overlap,
        unit_sizes=m_unit,
        m_unit=m_unit,
        Ak_unit=Ak_unit,
        Az2_unit=Az2_unit,
        Ak2_unit=Ak2_unit,
        AA_unit=AA_unit,
        AL_unit=AL_unit,
        M_rep=M_rep,
        Ak_rep=Ak_rep,
        Az2_rep=Az2_rep,
        Ak2_rep=Ak2_rep,
        AA_rep=AA_rep,
        AL_rep=AL_rep,
        lhs=lhs,
        rhs=rhs,
    )


# -----------------------------------------------------------------------------
# fitting
# -----------------------------------------------------------------------------


def fit_h2(
    prepared: H2Prepared,
    *,
    enrich_mode: str = "auto",
    report_tau: bool = True,
    allow_neg_enr: bool = False,
    clip_nonfinite_vals: bool = False,
    jack_mode: str = "mean",
    nan_policy: str = "omit",
) -> H2Fit:
    enrich_mode = _normalize_enrich_mode(enrich_mode)

    p = prepared
    R = p.jackknife.nrep
    K = p.trace_view.nbins

    sigma_core = _solve_linear_batch(p.lhs, p.rhs)                 # (R+1, K+1)
    sigma_reps = np.zeros((R + 1, K + 2), dtype=np.float64)
    sigma_reps[:, :K + 1] = sigma_core
    sigma_reps[:, K + 1] = np.nansum(sigma_core[:, :K], axis=1)

    sigma_g = sigma_core[:, :K]
    h2_tot = np.nansum(sigma_g, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = sigma_g / p.Ak_rep
        h2_cat_overlap = np.einsum("bkl,bl->bk", p.AA_rep, w, optimize=True)

    bad_overlap = ~np.isfinite(h2_cat_overlap)
    if clip_nonfinite_vals:
        h2_cat_overlap[bad_overlap] = 0.0
    else:
        h2_cat_overlap[bad_overlap] = np.nan

    h2_reps = np.zeros((R + 1, K + 1), dtype=np.float64)
    h2_reps[:, :K] = h2_cat_overlap
    h2_reps[:, K] = h2_tot

    with np.errstate(divide="ignore", invalid="ignore"):
        prop = p.Ak_rep / p.M_rep[:, None]

    requested = enrich_mode
    if requested == "auto":
        enrich_mode_used = "overlap" if p.has_overlap else "non-overlap"
        modes_to_compute = (enrich_mode_used,)
    elif requested == "both":
        enrich_mode_used = "overlap" if p.has_overlap else "non-overlap"
        modes_to_compute = ("non-overlap", "overlap")
    else:
        enrich_mode_used = requested
        modes_to_compute = (requested,)

    def _compute_enrichment(h2_cat):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (h2_cat / h2_tot[:, None]) / prop
            invalid = (~np.isfinite(out)) | (~np.isfinite(prop)) | (prop <= 0.0)
            if not allow_neg_enr:
                invalid |= (h2_tot[:, None] <= 0.0)
            out[invalid] = np.nan
        return out

    enrich_overlap_reps = None
    enrich_nonoverlap_reps = None
    if "overlap" in modes_to_compute:
        enrich_overlap_reps = _compute_enrichment(h2_cat_overlap)
    if "non-overlap" in modes_to_compute:
        enrich_nonoverlap_reps = _compute_enrichment(sigma_g)

    if requested == "both":
        enrich_reps = enrich_overlap_reps if enrich_mode_used == "overlap" else enrich_nonoverlap_reps
    else:
        enrich_reps = enrich_overlap_reps if enrich_mode_used == "overlap" else enrich_nonoverlap_reps

    tau_reps = None
    tau_star_reps = None
    if report_tau:
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_reps = sigma_g / p.Ak_rep
            meanA = p.Ak_rep / p.M_rep[:, None]
            meanA2 = p.Ak2_rep / p.M_rep[:, None]
            varA = np.maximum(meanA2 - meanA * meanA, 0.0)
            sdA = np.sqrt(varA, dtype=np.float64)
            denom = h2_tot / p.M_rep
            tau_star_reps = tau_reps * (sdA / denom[:, None])

        bad_tau = ~np.isfinite(tau_reps)
        bad_tau_star = ~np.isfinite(tau_star_reps)
        if clip_nonfinite_vals:
            tau_reps[bad_tau] = 0.0
            tau_star_reps[bad_tau_star] = 0.0
        else:
            tau_reps[bad_tau] = np.nan
            tau_star_reps[bad_tau_star] = np.nan

    est, se = p.jackknife.summarize(
        sigma_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    sigmas = np.column_stack([est, se])

    est, se = p.jackknife.summarize(
        h2_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    h2 = np.column_stack([est, se])

    est, se = p.jackknife.summarize(
        enrich_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    enrich = np.column_stack([est, se])

    enrich_overlap = None
    if enrich_overlap_reps is not None:
        est, se = p.jackknife.summarize(
            enrich_overlap_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        enrich_overlap = np.column_stack([est, se])

    enrich_nonoverlap = None
    if enrich_nonoverlap_reps is not None:
        est, se = p.jackknife.summarize(
            enrich_nonoverlap_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        enrich_nonoverlap = np.column_stack([est, se])

    tau = None
    tau_star = None
    if report_tau:
        est, se = p.jackknife.summarize(
            tau_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        tau = np.column_stack([est, se])

        est, se = p.jackknife.summarize(
            tau_star_reps,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        tau_star = np.column_stack([est, se])

    return H2Fit(
        prepared=prepared,
        sigma_reps=sigma_reps,
        h2_reps=h2_reps,
        enrich_reps=enrich_reps,
        enrich_overlap_reps=enrich_overlap_reps,
        enrich_nonoverlap_reps=enrich_nonoverlap_reps,
        tau_reps=tau_reps,
        tau_star_reps=tau_star_reps,
        sigmas=sigmas,
        h2=h2,
        enrich=enrich,
        enrich_overlap=enrich_overlap,
        enrich_nonoverlap=enrich_nonoverlap,
        tau=tau,
        tau_star=tau_star,
        enrich_mode_requested=requested,
        enrich_mode_used=enrich_mode_used,
    )
