from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import json
from .. import utils
from ..sumstats.moments import build_rg_summary_moment

@dataclass(frozen=True)
class RGPrepared:
    trace_view: object
    matched1: object
    matched2: object
    jackknife: object
    active_mask: np.ndarray
    unit_sizes: np.ndarray
    y: np.ndarray                  # (M,)
    n1_scale: float
    n2_scale: float
    summary_y_info: dict | None
    Ak_unit: np.ndarray            # (U,K)
    Ay_unit: np.ndarray            # (U,K)
    AL_unit: np.ndarray            # (U,K,K)
    Ak_rep: np.ndarray             # (R+1,K)
    Ay_rep: np.ndarray             # (R+1,K)
    AL_rep: np.ndarray             # (R+1,K,K)
    lhs: np.ndarray                # (R+1,K,K)


@dataclass(frozen=True)
class InterceptFit:
    trace_view: object
    matched1: object
    matched2: object
    jackknife: object
    active_mask: np.ndarray
    unit_sizes: np.ndarray
    ld: np.ndarray                 # (M,)
    y: np.ndarray                  # (M,)
    c_reps: np.ndarray             # (R+1,)
    c: np.ndarray                  # (2,) [estimate, se]
    info: dict


@dataclass(frozen=True)
class RGFit:
    prepared: RGPrepared
    intercept: InterceptFit
    h2_fit1: object
    h2_fit2: object
    gamma_reps: np.ndarray         # (R+1,K)
    rg_reps: np.ndarray            # (R+1,K)
    gamma: np.ndarray              # (K,2)
    rg: np.ndarray                 # (K,2)
    gamma_total: np.ndarray        # (2,)
    rg_total: np.ndarray           # (2,)
    rg_se_method: str
    kmoment_info: dict | None = None


def _component_rg(gamma, v1, v2):
    gamma_arr, v1_arr, v2_arr = np.broadcast_arrays(
        np.asarray(gamma, dtype=np.float64),
        np.asarray(v1, dtype=np.float64),
        np.asarray(v2, dtype=np.float64),
    )
    out = np.full(gamma_arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(gamma_arr) & np.isfinite(v1_arr) & np.isfinite(v2_arr) & (v1_arr > 0.0) & (v2_arr > 0.0)
    out[valid] = gamma_arr[valid] / np.sqrt(v1_arr[valid] * v2_arr[valid])
    return out


class RGResultWriter:
    @staticmethod
    def save_jackknife_text(fit: RGFit, path: str):
        p = fit.prepared
        R = p.jackknife.nrep
        K = p.trace_view.nbins
        gamma_tot = np.nansum(fit.gamma_reps, axis=1)
        rg_tot = np.full(R + 1, np.nan, dtype=np.float64)
        h1_tot = np.asarray(fit.h2_fit1.h2_reps[:, -1], dtype=np.float64)
        h2_tot = np.asarray(fit.h2_fit2.h2_reps[:, -1], dtype=np.float64)
        rg_tot = _component_rg(gamma_tot, h1_tot, h2_tot)
        with open(path, "w") as fd:
            header = ["replicate", "intercept_c"]
            header += [f"gamma_g_{k}" for k in range(K)]
            header += ["gamma_g_total"]
            header += [f"rg_{k}" for k in range(K)]
            header += ["rg_total"]
            fd.write("# " + "\t".join(header) + "\n")
            for r in range(R + 1):
                label = "full" if r == R else str(r)
                vals = [fit.intercept.c_reps[r]]
                vals += fit.gamma_reps[r].tolist()
                vals += [gamma_tot[r]]
                vals += fit.rg_reps[r].tolist()
                vals += [rg_tot[r]]
                fd.write(label + "\t" + "\t".join(f"{x:.10g}" for x in vals) + "\n")

    @staticmethod
    def _jsonify_numeric(x):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            v = float(arr)
            return v if np.isfinite(v) else None
        return [RGResultWriter._jsonify_numeric(v) for v in arr]

    @staticmethod
    def save_score_normal_equations_json(fit: RGFit, path: str):
        from .h2core import H2ResultWriter

        info = fit.intercept.info if isinstance(fit.intercept.info, dict) else {}
        n_overlap = info.get("n_overlap", None)
        if n_overlap is None:
            raise ValueError(
                "Full SCORE normal-equation dump requires known overlap N; "
                "run rg with --pheno-rg (and optionally --pheno-rg-cov)."
            )

        n_overlap = float(n_overlap)
        if not (np.isfinite(n_overlap) and n_overlap > 0.0):
            raise ValueError(f"Invalid overlap size for SCORE dump: {n_overlap}")

        p = fit.prepared
        K = int(p.trace_view.nbins)
        R = int(p.jackknife.nrep)

        lhs_red = np.asarray(p.lhs, dtype=np.float64)
        Ak = np.asarray(p.Ak_rep, dtype=np.float64)
        Ay = np.asarray(p.Ay_rep, dtype=np.float64)
        c_reps = np.asarray(fit.intercept.c_reps, dtype=np.float64)

        if lhs_red.shape != (R + 1, K, K):
            raise ValueError(f"Unexpected lhs shape: {lhs_red.shape}, expected {(R + 1, K, K)}")
        if Ak.shape != (R + 1, K):
            raise ValueError(f"Unexpected Ak_rep shape: {Ak.shape}, expected {(R + 1, K)}")
        if Ay.shape != (R + 1, K):
            raise ValueError(f"Unexpected Ay_rep shape: {Ay.shape}, expected {(R + 1, K)}")
        if c_reps.shape != (R + 1,):
            raise ValueError(f"Unexpected c_reps shape: {c_reps.shape}, expected {(R + 1,)}")

        n1_raw = float(p.matched1.nsamp)
        n2_raw = float(p.matched2.nsamp)
        n1_scale = float(p.n1_scale)
        n2_scale = float(p.n2_scale)
        sqrt_n1n2 = float(np.sqrt(n1_scale * n2_scale))

        T = np.full((R + 1, K + 1, K + 1), np.nan, dtype=np.float64)
        T[:, :K, :K] = lhs_red + n_overlap
        T[:, :K, K] = n_overlap
        T[:, K, :K] = n_overlap
        T[:, K, K] = n_overlap

        q = np.full((R + 1, K + 1), np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            q[:, :K] = (Ay * sqrt_n1n2) / Ak
        bad_q = (~np.isfinite(Ay)) | (~np.isfinite(Ak)) | (Ak <= 0.0)
        q[:, :K] = np.where(bad_q, np.nan, q[:, :K])

        q[:, K] = c_reps * sqrt_n1n2
        q[:, K] = np.where(np.isfinite(q[:, K]), q[:, K], np.nan)

        sigma = _solve_linear_batch(T, q)

        headers = getattr(p.trace_view, "annot_header", None)
        if headers is None or len(headers) != K:
            headers = [f"bin_{k}" for k in range(K)]
        else:
            headers = [str(h) for h in headers]

        trait1_name = getattr(p.matched1, "name", None)
        if trait1_name is None or str(trait1_name).strip() == "":
            trait1_name = "trait1"
        trait1_name = str(trait1_name)

        trait2_name = getattr(p.matched2, "name", None)
        if trait2_name is None or str(trait2_name).strip() == "":
            trait2_name = "trait2"
        trait2_name = str(trait2_name)

        payload = {
            "system": "cross",
            "meta": {
                "equation_type": "score_full",
                "kernel_name": "score",
                "multi_component": bool(K > 1),
                "partial_overlap": bool(n_overlap < (min(n1_raw, n2_raw) - 0.5)),
                "n_overlap": n_overlap,
                "n1_summary_raw": n1_raw,
                "n2_summary_raw": n2_raw,
                "n1_scale": n1_scale,
                "n2_scale": n2_scale,
                "nrep": R,
                "fixed_intercept_source": str(info.get("source", "")),
                "cov_adjusted": bool(info.get("cov_adjusted", False)),
                "annot_headers": headers,
                "trait1_system": trait1_name,
                "trait2_system": trait2_name,
            },
            "sigma_names": [f"gamma_g_{k}" for k in range(K)] + ["gamma_e"],
            "moment_names": [f"score_row_{k}" for k in range(K)] + ["overlap_row"],
            "full": {
                "replicate": "full",
                "T": RGResultWriter._jsonify_numeric(T[R]),
                "q": RGResultWriter._jsonify_numeric(q[R]),
                "sigma": RGResultWriter._jsonify_numeric(sigma[R]),
            },
            "jackknife": [
                {
                    "replicate": int(r),
                    "T": RGResultWriter._jsonify_numeric(T[r]),
                    "q": RGResultWriter._jsonify_numeric(q[r]),
                    "sigma": RGResultWriter._jsonify_numeric(sigma[r]),
                }
                for r in range(R)
            ],
            "trait1_h2": H2ResultWriter.build_score_normal_equations_payload(
                fit.h2_fit1,
                system=trait1_name,
            ),
            "trait2_h2": H2ResultWriter.build_score_normal_equations_payload(
                fit.h2_fit2,
                system=trait2_name,
            ),
        }

        with open(path, "w") as fd:
            json.dump(payload, fd, indent=2, allow_nan=False)


def build_manifest_summary_row(
    *,
    phen1: str,
    phen2: str,
    sumstats1: str,
    sumstats2: str,
    cov_rank1,
    cov_rank2,
    intercept_rg_input,
    out_prefix: str,
    n_snps: int | None,
    annot_header,
    h2_fit1,
    h2_fit2,
    intercept: InterceptFit,
    rg_fit: RGFit,
) -> dict:
    row = {
        "phen1": phen1,
        "phen2": phen2,
        "sumstats1": sumstats1,
        "sumstats2": sumstats2,
        "cov_rank1": cov_rank1,
        "cov_rank2": cov_rank2,
        "intercept_rg_input": intercept_rg_input,
        "out_prefix": out_prefix,
        "n_snps": (int(n_snps) if n_snps is not None else None),
        "h2_trait1": float(h2_fit1.h2[-1, 0]),
        "h2_trait1_se": float(h2_fit1.h2[-1, 1]),
        "h2_trait2": float(h2_fit2.h2[-1, 0]),
        "h2_trait2_se": float(h2_fit2.h2[-1, 1]),
        "intercept_c": float(intercept.c[0]),
        "intercept_c_se": float(intercept.c[1]),
        "gamma_g_total": float(rg_fit.gamma_total[0]),
        "gamma_g_total_se": float(rg_fit.gamma_total[1]),
        "rg_total": float(rg_fit.rg_total[0]),
        "rg_total_se": float(rg_fit.rg_total[1]),
    }

    for j, header in enumerate(annot_header):
        token = str(header)
        row[f"gamma_g__{token}"] = float(rg_fit.gamma[j, 0])
        row[f"gamma_g__{token}_se"] = float(rg_fit.gamma[j, 1])
        row[f"rg__{token}"] = float(rg_fit.rg[j, 0])
        row[f"rg__{token}_se"] = float(rg_fit.rg[j, 1])

    return row


# -----------------------------------------------------------------------------
# low-level helpers
# -----------------------------------------------------------------------------


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
    flat_unit = unit.reshape(unit.shape[0], -1)
    flat_full = full.reshape(-1)
    deleted = D @ flat_unit
    reps = flat_full[None, :] - deleted
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


def _validate_common_axis(trace_view, matched1, matched2, jackknife):
    M = trace_view.nsnps
    if matched1.nsnps != M or matched2.nsnps != M:
        raise ValueError("TraceView and MatchedSumstats must have the same number of SNPs.")
    if not np.array_equal(trace_view.snps, matched1.snps):
        raise ValueError("TraceView and matched1 SNP order mismatch.")
    if not np.array_equal(trace_view.snps, matched2.snps):
        raise ValueError("TraceView and matched2 SNP order mismatch.")
    if jackknife.nsnps != M:
        raise ValueError(
            "JackknifeDesign was not built on this TraceView SNP axis. "
            f"Expected {jackknife.nsnps}, got {M}."
        )


# -----------------------------------------------------------------------------
# main rg preparation / fit
# -----------------------------------------------------------------------------

def prepare_rg(
    trace_view,
    matched1,
    matched2,
    jackknife,
    *,
    active_mask=None,
    summary_y=None,
    summary_y_info=None,
    adjust_delta: bool = False,
):
    _validate_common_axis(trace_view, matched1, matched2, jackknife)

    M = trace_view.nsnps
    K = trace_view.nbins
    R = jackknife.nrep
    U = jackknife.nunit

    if summary_y is None:
        summary_y, summary_y_info = build_rg_summary_moment(matched1, matched2)

    y = np.asarray(summary_y, dtype=np.float64)
    if y.ndim != 1 or y.size != M:
        raise ValueError(f"summary_y must have shape ({M},), got {y.shape}")

    if summary_y_info is not None:
        n1_scale = float(summary_y_info.get("trait1_n_scale", getattr(matched1, "n_scale", matched1.nsamp)))
        n2_scale = float(summary_y_info.get("trait2_n_scale", getattr(matched2, "n_scale", matched2.nsamp)))
    else:
        n1_scale = float(getattr(matched1, "n_scale", matched1.nsamp))
        n2_scale = float(getattr(matched2, "n_scale", matched2.nsamp))

    if not (np.isfinite(n1_scale) and n1_scale > 0.0 and np.isfinite(n2_scale) and n2_scale > 0.0):
        raise RuntimeError(f"Invalid rg n_scale pair: n1_scale={n1_scale}, n2_scale={n2_scale}")

    if active_mask is None:
        active_mask = np.isfinite(y)
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.ndim != 1 or active_mask.size != M:
        raise ValueError(f"active_mask must be length {M}; got {active_mask.shape}")
    active_mask = active_mask & np.isfinite(y)

    if not np.any(active_mask):
        raise ValueError("No SNPs remain after filtering non-finite rg summary moments.")

    A = np.asarray(trace_view.annot, dtype=np.float64, order="C")
    L = np.asarray(trace_view.ldscores, dtype=np.float64, order="C")

    Ak_unit = np.zeros((U, K), dtype=np.float64)
    Ay_unit = np.zeros((U, K), dtype=np.float64)
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
        yu = y[s:e][mu]
        Ak_unit[u] = Au.sum(axis=0, dtype=np.float64)
        Ay_unit[u] = Au.T @ yu
        AL_unit[u] = Au.T @ Lu

    unit_sizes = jackknife.unit_sizes(active_mask=active_mask, dtype=np.float64)
    Ak_rep = _stack_delete_replicates(Ak_unit.sum(axis=0), Ak_unit, jackknife.D)
    Ay_rep = _stack_delete_replicates(Ay_unit.sum(axis=0), Ay_unit, jackknife.D)
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
        src_mass[:R] = Ak_rep[:R]

    M_k = Ak_rep[:, :, None]
    M_l = src_mass[:, None, :]
    lhs = utils._calc_rg_trace_from_ld_batch(AL_rep, n1_scale, n2_scale, M_k, M_l)

    lhs = _symmetrize_with_design(lhs, jackknife, unit_sizes)

    return RGPrepared(
        trace_view=trace_view,
        matched1=matched1,
        matched2=matched2,
        jackknife=jackknife,
        active_mask=active_mask,
        unit_sizes=unit_sizes,
        y=y,
        n1_scale=n1_scale,
        n2_scale=n2_scale,
        summary_y_info=summary_y_info,
        Ak_unit=Ak_unit,
        Ay_unit=Ay_unit,
        AL_unit=AL_unit,
        Ak_rep=Ak_rep,
        Ay_rep=Ay_rep,
        AL_rep=AL_rep,
        lhs=lhs,
    )


def _estimate_total_rg_delta_se(gamma_tot, h2_tot1, h2_tot2, jackknife, *, unit_sizes, center, nan_policy):
    gamma_tot = np.asarray(gamma_tot, dtype=np.float64)
    h2_tot1 = np.asarray(h2_tot1, dtype=np.float64)
    h2_tot2 = np.asarray(h2_tot2, dtype=np.float64)

    g = float(gamma_tot[-1])
    v1 = float(h2_tot1[-1])
    v2 = float(h2_tot2[-1])
    if not (np.isfinite(g) and np.isfinite(v1) and np.isfinite(v2) and v1 > 0.0 and v2 > 0.0):
        return np.array([np.nan, np.nan], dtype=np.float64)

    rg_full = g / np.sqrt(v1 * v2)
    d_g = 1.0 / np.sqrt(v1 * v2)
    d_v1 = -0.5 * rg_full / v1
    d_v2 = -0.5 * rg_full / v2

    lin_rg = (rg_full + d_g * (gamma_tot - g) + d_v1 * (h2_tot1 - v1) + d_v2 * (h2_tot2 - v2)).astype(np.float64, copy=False)
    lin_rg[-1] = rg_full
    est, se = jackknife.summarize(
        lin_rg,
        unit_sizes=unit_sizes,
        axis=0,
        center=center,
        nan_policy=nan_policy,
    )
    return np.array([float(est), float(se)], dtype=np.float64)


def _combine_se_rss(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)

    out = np.full(a.shape, np.nan, dtype=np.float64)
    ga = np.isfinite(a)
    gb = np.isfinite(b)

    out[ga & ~gb] = np.abs(a[ga & ~gb])
    out[~ga & gb] = np.abs(b[~ga & gb])

    both = ga & gb
    out[both] = np.sqrt(a[both] * a[both] + b[both] * b[both])
    return out


def _estimate_fixedc_cluster_robust_se_single_component(
    prepared: RGPrepared,
    intercept_fit: InterceptFit,
    h2_fit1,
    h2_fit2,
    *,
    gamma_full: float,
    rg_full_bin: float,
    rg_full_total: float,
    robust_kind: str = "cr2",   # one of {'cr0','cr1','cr2'}
    add_external_c_se: bool = False,
    jack_mode: str = "mean",
    nan_policy: str = "omit",
):
    """
    Robust SE for the SINGLE-COMPONENT constrained rg estimator with FIXED external intercept.

    This is the correct sandwich target for the estimator actually reported by
    the constrained pipeline:
        gamma_hat(c0 fixed) solves sum_u m_u(gamma; c0) = 0

    It does NOT profile a free nuisance regression. That earlier approach targets
    a different estimator and tends to reproduce summary-only SEs.

    Parameters
    ----------
    prepared
        RGPrepared from prepare_rg(...), must have K == 1.
    intercept_fit
        InterceptFit. Must correspond to a FIXED external intercept (info['fixed']=True).
    h2_fit1, h2_fit2
        H2 fits, used only to convert robust gamma SE into rg SE via a delta correction
        for denominator uncertainty.
    gamma_full
        Full-sample gamma estimate, i.e. gamma_reps[-1, 0].
    rg_full_bin
        Full-sample per-bin rg estimate, i.e. rg_reps[-1, 0].
    rg_full_total
        Full-sample total rg estimate from gamma_tot_reps[-1] / sqrt(h2_tot1[-1] * h2_tot2[-1]).
    robust_kind
        'cr0' = plain sandwich,
        'cr1' = sandwich * U/(U-1),
        'cr2' = leverage-corrected sandwich with CR1 prefactor.
    add_external_c_se
        If True, and intercept_fit.c[1] > 0, add an independent delta-method variance
        contribution from the external intercept:
            Var_gamma += (d gamma / d c)^2 Var(c)
        This is OFF by default because independence is often not justified.
    """
    p = prepared
    if int(p.trace_view.nbins) != 1:
        raise ValueError("Fixed-c robust SE is implemented only for K == 1.")

    info = intercept_fit.info if isinstance(intercept_fit.info, dict) else {}
    if not bool(info.get("fixed", False)):
        raise ValueError(
            "rg_se_method='robust' currently requires a FIXED external intercept "
            "(intercept_fit.info['fixed'] == True)."
        )

    robust_kind = str(robust_kind).strip().lower()
    if robust_kind not in {"cr0", "cr1", "cr2"}:
        raise ValueError("robust_kind must be one of {'cr0','cr1','cr2'}")

    c0 = float(np.asarray(intercept_fit.c_reps[-1], dtype=np.float64))
    if not np.isfinite(c0):
        raise RuntimeError(f"Non-finite fixed intercept c0={c0}")

    # Block-level quantities from RGPrepared
    a_u = np.asarray(p.Ak_unit[:, 0], dtype=np.float64)       # (U,)
    s_u = np.asarray(p.Ay_unit[:, 0], dtype=np.float64)       # (U,)
    l_u = np.asarray(p.AL_unit[:, 0, 0], dtype=np.float64)    # (U,)

    U_all = int(a_u.size)
    if s_u.size != U_all or l_u.size != U_all:
        raise RuntimeError("Unit-level rg arrays have inconsistent lengths.")

    A_full = float(np.sum(a_u))
    L_full = float(np.sum(l_u))
    T_full = float(np.asarray(p.lhs[-1, 0, 0], dtype=np.float64))
    sqrt_n1n2 = float(np.sqrt(float(p.n1_scale) * float(p.n2_scale)))

    if not (np.isfinite(A_full) and A_full > 0.0):
        raise RuntimeError(f"Invalid full annotation mass A_full={A_full}")
    if not (np.isfinite(L_full) and L_full > 0.0):
        raise RuntimeError(f"Invalid full AL mass L_full={L_full}")
    if not (np.isfinite(T_full) and T_full > 0.0):
        raise RuntimeError(f"Invalid full reduced trace T_full={T_full}")
    if not np.isfinite(gamma_full):
        raise RuntimeError(f"Invalid gamma_full={gamma_full}")

    # Exact blockwise trace contributions for K=1 up to the full-sample scaling.
    # Since T_full is linear in the full AL sum for K=1, apportion by l_u / L_full.
    t_u = T_full * (l_u / L_full)

    # Block score evaluated at the FULL-SAMPLE constrained estimator:
    #   m_u = r_u - t_u * gamma_hat
    # with
    #   r_u = sqrt(n1*n2)/A * (s_u - c0 * a_u)
    r_u = (sqrt_n1n2 / A_full) * (s_u - c0 * a_u)
    psi_u = r_u - t_u * float(gamma_full)

    valid = (
        np.isfinite(a_u) & np.isfinite(s_u) & np.isfinite(l_u) &
        np.isfinite(t_u) & np.isfinite(r_u) & np.isfinite(psi_u) &
        np.isfinite(T_full) & (T_full > 0.0)
    )
    if not np.any(valid):
        raise RuntimeError("No valid jackknife units for fixed-c robust gamma SE.")

    psi_v = psi_u[valid].astype(np.float64, copy=False)
    t_v = t_u[valid].astype(np.float64, copy=False)
    U_eff = int(psi_v.size)
    if U_eff <= 1:
        raise RuntimeError(f"Need at least 2 valid units for robust SE; got U_eff={U_eff}")

    # Optional leverage correction
    if robust_kind == "cr2":
        h_v = np.clip(t_v / T_full, 0.0, 1.0 - 1e-12)
        psi_adj = psi_v / np.sqrt(1.0 - h_v)
        small_sample_factor = float(U_eff) / float(U_eff - 1) if U_eff > 1 else np.nan
    elif robust_kind == "cr1":
        psi_adj = psi_v
        small_sample_factor = float(U_eff) / float(U_eff - 1) if U_eff > 1 else np.nan
    else:  # cr0
        psi_adj = psi_v
        small_sample_factor = 1.0

    meat = float(np.sum(psi_adj * psi_adj))
    var_gamma = small_sample_factor * meat / float(T_full * T_full)

    if add_external_c_se:
        c_se = float(np.asarray(intercept_fit.c[1], dtype=np.float64))
        if np.isfinite(c_se) and c_se > 0.0:
            # d gamma / d c = -sqrt(n1*n2) / T
            var_gamma += (sqrt_n1n2 / T_full) ** 2 * (c_se ** 2)

    if not (np.isfinite(var_gamma) and var_gamma >= 0.0):
        raise RuntimeError(f"Invalid robust gamma variance: {var_gamma}")

    se_gamma = float(np.sqrt(var_gamma))

    # --- Convert robust gamma SE into rg SE ---
    # We keep denominator uncertainty from the existing h2 jackknife path
    # and combine it with the robust numerator SE via delta + RSS.
    # This is most appropriate in the null / near-null regime, which is your use case.

    def _delta_rg_se_from_h2_only(rg_full, v1_full, v2_full, v1_reps, v2_reps):
        rg_full = float(rg_full)
        v1_full = float(v1_full)
        v2_full = float(v2_full)
        v1_reps = np.asarray(v1_reps, dtype=np.float64)
        v2_reps = np.asarray(v2_reps, dtype=np.float64)

        if not (np.isfinite(rg_full) and np.isfinite(v1_full) and np.isfinite(v2_full) and v1_full > 0.0 and v2_full > 0.0):
            return np.nan

        d_v1 = -0.5 * rg_full / v1_full
        d_v2 = -0.5 * rg_full / v2_full
        lin_rg = (rg_full + d_v1 * (v1_reps - v1_full) + d_v2 * (v2_reps - v2_full)).astype(np.float64, copy=False)
        lin_rg[-1] = rg_full

        _, se_den = p.jackknife.summarize(
            lin_rg,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        return float(se_den)

    # Per-bin rg (K=1)
    v1_bin_full = float(np.asarray(h2_fit1.sigma_reps[-1, 0], dtype=np.float64))
    v2_bin_full = float(np.asarray(h2_fit2.sigma_reps[-1, 0], dtype=np.float64))
    d_gamma_bin = np.nan
    if np.isfinite(v1_bin_full) and v1_bin_full > 0.0 and np.isfinite(v2_bin_full) and v2_bin_full > 0.0:
        d_gamma_bin = 1.0 / np.sqrt(v1_bin_full * v2_bin_full)
    se_den_bin = _delta_rg_se_from_h2_only(
        rg_full=float(rg_full_bin),
        v1_full=v1_bin_full,
        v2_full=v2_bin_full,
        v1_reps=np.asarray(h2_fit1.sigma_reps[:, 0], dtype=np.float64),
        v2_reps=np.asarray(h2_fit2.sigma_reps[:, 0], dtype=np.float64),
    )
    if np.isfinite(d_gamma_bin) and np.isfinite(se_den_bin):
        se_rg_bin = float(np.sqrt((d_gamma_bin * se_gamma) ** 2 + se_den_bin ** 2))
    else:
        se_rg_bin = np.nan

    # Total rg
    v1_tot_full = float(np.asarray(h2_fit1.h2_reps[-1, -1], dtype=np.float64))
    v2_tot_full = float(np.asarray(h2_fit2.h2_reps[-1, -1], dtype=np.float64))
    d_gamma_tot = np.nan
    if np.isfinite(v1_tot_full) and v1_tot_full > 0.0 and np.isfinite(v2_tot_full) and v2_tot_full > 0.0:
        d_gamma_tot = 1.0 / np.sqrt(v1_tot_full * v2_tot_full)
    se_den_tot = _delta_rg_se_from_h2_only(
        rg_full=float(rg_full_total),
        v1_full=v1_tot_full,
        v2_full=v2_tot_full,
        v1_reps=np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64),
        v2_reps=np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64),
    )
    if np.isfinite(d_gamma_tot) and np.isfinite(se_den_tot):
        se_rg_total = float(np.sqrt((d_gamma_tot * se_gamma) ** 2 + se_den_tot ** 2))
    else:
        se_rg_total = np.nan

    return {
        "se_gamma": se_gamma,
        "se_rg_bin": se_rg_bin,
        "se_rg_total": se_rg_total,
        "U_eff": int(U_eff),
        "T_full": float(T_full),
        "A_full": float(A_full),
        "L_full": float(L_full),
        "fixed_c": float(c0),
        "robust_kind": robust_kind,
        "small_sample_factor": float(small_sample_factor),
        "max_leverage": float(np.max(np.clip(t_v / T_full, 0.0, 1.0))) if U_eff > 0 else np.nan,
        "score_mean_abs": float(np.mean(np.abs(psi_v))) if U_eff > 0 else np.nan,
    }


def _estimate_kmoment_model_se_single_component(
    prepared: RGPrepared,
    intercept_fit: InterceptFit,
    h2_fit1,
    h2_fit2,
    *,
    gamma_full: float,
    alpha_tol: float = 5e-3,
    drop_tol: float = 0.01,
    sample_warn_tol: float = 0.01,
    sample_hard_tol: float = 0.02,
    jack_mode: str = "mean",
    nan_policy: str = "omit",
):
    """
    Model-based SE for the SINGLE-COMPONENT constrained rg estimator using
    precomputed K-moments.

    Exact target:
      - K == 1
      - fixed external intercept
      - common projected sample space

    Practical relaxed mode:
      - allows small projected-scale mismatch across traits / overlap
      - reuses full-panel moments when SNP loss is small
    """
    p = prepared
    if int(p.trace_view.nbins) != 1:
        raise ValueError("rg_se_method='moments' is implemented only for the single-component case (K == 1).")

    info = intercept_fit.info if isinstance(intercept_fit.info, dict) else {}
    if not bool(info.get("fixed", False)):
        raise ValueError(
            "rg_se_method='moments' currently requires a FIXED external intercept "
            "(--pheno-rg/--pheno-rg-cov or --intercept-rg)."
        )

    tv = p.trace_view
    km = getattr(tv, "kmoments", None)
    if km is None:
        raise ValueError(
            "rg_se_method='moments' requested, but no .gw.kmoments file was found "
            "next to the main LD-score file."
        )

    def _kget(name, default=np.nan):
        try:
            return float(km.get(name, default))
        except Exception:
            return float(default)

    def _rel_gap(a, b):
        a = float(a)
        b = float(b)
        if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
            return np.nan
        return abs(a - b) / max(np.sqrt(a * b), 1.0)

    # ------------------------------------------------------------------
    # 1) Reuse of full-panel moments after SNP filtering: allow only when
    #    the dropped SNP fraction is a small perturbation of the full panel.
    # ------------------------------------------------------------------
    M_full = int(round(_kget("nsnps")))
    if M_full <= 0:
        raise ValueError("kmoments file is missing a valid full-panel SNP count.")

    keep_frac_view = float(tv.nsnps) / float(M_full)
    keep_frac_active = (
        float(np.mean(np.asarray(p.active_mask, dtype=np.float64)))
        if getattr(p, "active_mask", None) is not None and p.active_mask.size
        else 0.0
    )
    keep_frac_total = keep_frac_view * keep_frac_active
    drop_frac_total = 1.0 - keep_frac_total

    if not (drop_frac_total <= float(drop_tol)):
        raise ValueError(
            "rg_se_method='moments' can only reuse full-panel kmoments when SNP loss is small. "
            f"full_panel_M={M_full}, trace_view_M={tv.nsnps}, active_keep={keep_frac_active:.6f}, "
            f"total_drop={drop_frac_total:.3%} > tol={float(drop_tol):.3%}."
        )

    approximate_moments = bool(drop_frac_total > 0.0)

    # ------------------------------------------------------------------
    # 2) Check whether the projected sample spaces are close enough that
    #    the common-space derivation is a small-perturbation approximation.
    # ------------------------------------------------------------------
    syi = p.summary_y_info if isinstance(p.summary_y_info, dict) else {}

    # Same covariate rank is still the safest regime. If ranks differ, the
    # projected spaces are genuinely different, not just slightly different.
    cr1 = syi.get("trait1_cov_rank", None)
    cr2 = syi.get("trait2_cov_rank", None)
    if cr1 is not None and cr2 is not None and int(cr1) != int(cr2):
        raise ValueError(
            "rg_se_method='moments' currently requires the same covariate rank for both traits "
            f"(got trait1_cov_rank={cr1}, trait2_cov_rank={cr2})."
        )

    r1 = float(p.n1_scale)
    r2 = float(p.n2_scale)
    if not (np.isfinite(r1) and np.isfinite(r2) and r1 > 0.0 and r2 > 0.0):
        raise ValueError(
            f"Invalid projected sample scales for rg_se_method='moments': n1_scale={r1}, n2_scale={r2}."
        )

    scale_rel_gap = _rel_gap(r1, r2)

    # Infer the effective overlap scale on the SAME convention as n_scale,
    # without hard-coding whether n_scale = n-q or n-q-1.
    # We do this by estimating the raw->scale offset from the matched sumstats.
    n1_raw = float(getattr(p.matched1, "nsamp", np.nan))
    n2_raw = float(getattr(p.matched2, "nsamp", np.nan))
    n_overlap = info.get("n_overlap", None)

    overlap_check = "unverified"
    overlap_scale = np.nan
    overlap_rel_gap = np.nan

    if np.isfinite(n1_raw) and np.isfinite(n2_raw):
        off1 = n1_raw - r1
        off2 = n2_raw - r2
        off_bar = 0.5 * (off1 + off2)

        if n_overlap is not None:
            n_overlap = float(n_overlap)
            if np.isfinite(n_overlap) and n_overlap > 0.0:
                overlap_scale = n_overlap - off_bar
                if np.isfinite(overlap_scale) and overlap_scale > 0.0:
                    overlap_rel_gap = max(_rel_gap(overlap_scale, r1), _rel_gap(overlap_scale, r2))
                    overlap_check = "effective_overlap_from_raw_n"
                else:
                    overlap_check = "invalid_effective_overlap"
            else:
                overlap_check = "invalid_raw_overlap"
        else:
            overlap_check = "missing_overlap"
    else:
        overlap_check = "missing_raw_nsamp"

    sample_rel_gap = scale_rel_gap
    if np.isfinite(overlap_rel_gap):
        sample_rel_gap = max(sample_rel_gap, overlap_rel_gap)

    if not np.isfinite(sample_rel_gap):
        raise ValueError(
            "Could not construct a valid projected-sample mismatch diagnostic for rg_se_method='moments'."
        )

    if sample_rel_gap > float(sample_hard_tol):
        raise ValueError(
            "rg_se_method='moments' requires near-complete agreement of the projected sample scales. "
            f"scale_rel_gap={scale_rel_gap:.3%}, overlap_rel_gap={overlap_rel_gap:.3%}, "
            f"max_gap={sample_rel_gap:.3%} > hard_tol={float(sample_hard_tol):.3%}."
        )

    approximate_sample_match = bool(sample_rel_gap > 0.0)
    sample_warning = bool(sample_rel_gap > float(sample_warn_tol))

    # ------------------------------------------------------------------
    # 3) Fixed intercept and plug-in point estimates
    # ------------------------------------------------------------------
    c0 = float(np.asarray(intercept_fit.c_reps[-1], dtype=np.float64))
    gamma_full = float(gamma_full)

    if not np.isfinite(c0):
        raise RuntimeError(f"Non-finite fixed intercept in rg_se_method='moments': c={c0}")
    if not np.isfinite(gamma_full):
        raise RuntimeError(f"Non-finite gamma_full in rg_se_method='moments': gamma={gamma_full}")

    # Single-component h2 plug-ins actually used by the variance formula.
    h1 = float(np.clip(np.asarray(h2_fit1.sigma_reps[-1, 0], dtype=np.float64), 0.0, 1.0))
    h2 = float(np.clip(np.asarray(h2_fit2.sigma_reps[-1, 0], dtype=np.float64), 0.0, 1.0))
    e1 = 1.0 - h1
    e2 = 1.0 - h2

    # ------------------------------------------------------------------
    # 4) Choose moment source
    # ------------------------------------------------------------------
    alpha_probe = _kget("alpha_probe")
    alpha_probe_err = abs(alpha_probe - 1.0) if np.isfinite(alpha_probe) else np.nan

    use_probealpha = (
        np.isfinite(alpha_probe_err)
        and alpha_probe_err > float(alpha_tol)
        and np.isfinite(_kget("t0_probealpha"))
        and np.isfinite(_kget("t1_probealpha"))
        and np.isfinite(_kget("t2_probealpha"))
        and (_kget("t0_probealpha") > 0.0)
    )

    if use_probealpha:
        t0 = _kget("t0_probealpha")
        t1 = _kget("t1_probealpha")
        t2 = _kget("t2_probealpha")
        moment_source = "probealpha"
    else:
        t0 = _kget("t0_rank")
        t1 = _kget("t1_rank")
        t2 = _kget("t2_rank")
        moment_source = "rank"

    if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(t2) and t0 > 0.0):
        raise RuntimeError(
            f"Invalid moments selected for model-based SE: t0={t0}, t1={t1}, t2={t2}"
        )

    s4 = t2 - 2.0 * t1 + t0
    delta_reff = (t0 * t0 / s4) if (np.isfinite(s4) and s4 > 0.0) else np.nan

    # Optional Monte Carlo sanity diagnostic
    trace_k2_from_ldscore = _kget("trace_K2_from_ldscore")
    trace_k2_probe = _kget("trace_K2_probe")
    if np.isfinite(trace_k2_from_ldscore) and trace_k2_from_ldscore > 0.0 and np.isfinite(trace_k2_probe):
        k2_probe_relerr = abs(trace_k2_probe - trace_k2_from_ldscore) / trace_k2_from_ldscore
    else:
        k2_probe_relerr = np.nan

    # ------------------------------------------------------------------
    # 5) Model-based variance of gamma
    # ------------------------------------------------------------------
    num = (
        (h1 * h2 + gamma_full * gamma_full) * t2
        + (h1 * e2 + h2 * e1 + 2.0 * gamma_full * c0) * t1
        + (e1 * e2 + c0 * c0) * t0
    )
    var_gamma = num / (t0 * t0)

    tol = 1e-12 * max(1.0, abs(num) / max(t0 * t0, 1.0))
    if np.isfinite(var_gamma) and var_gamma < 0.0 and abs(var_gamma) <= tol:
        var_gamma = 0.0

    if not (np.isfinite(var_gamma) and var_gamma >= 0.0):
        raise RuntimeError(
            f"Invalid model-based gamma variance: var_gamma={var_gamma}, "
            f"num={num}, t0={t0}, t1={t1}, t2={t2}, h1={h1}, h2={h2}, c={c0}, gamma={gamma_full}"
        )

    se_gamma = float(np.sqrt(var_gamma))

    # ------------------------------------------------------------------
    # 6) Convert to rg SE.
    # For K == 1, bin rg and total rg are the same quantity.
    # ------------------------------------------------------------------
    def _delta_rg_se_from_h2_only(rg_full, v1_full, v2_full, v1_reps, v2_reps):
        rg_full = float(rg_full)
        v1_full = float(v1_full)
        v2_full = float(v2_full)
        v1_reps = np.asarray(v1_reps, dtype=np.float64)
        v2_reps = np.asarray(v2_reps, dtype=np.float64)

        if not (
            np.isfinite(rg_full)
            and np.isfinite(v1_full) and v1_full > 0.0
            and np.isfinite(v2_full) and v2_full > 0.0
        ):
            return np.nan

        d_v1 = -0.5 * rg_full / v1_full
        d_v2 = -0.5 * rg_full / v2_full
        lin_rg = (rg_full + d_v1 * (v1_reps - v1_full) + d_v2 * (v2_reps - v2_full)).astype(np.float64, copy=False)
        lin_rg[-1] = rg_full

        _, se_den = p.jackknife.summarize(
            lin_rg,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        return float(se_den)

    h1_reps = np.asarray(h2_fit1.sigma_reps[:, 0], dtype=np.float64)
    h2_reps = np.asarray(h2_fit2.sigma_reps[:, 0], dtype=np.float64)
    h1_full = float(h1_reps[-1])
    h2_full = float(h2_reps[-1])

    if np.isfinite(h1_full) and h1_full > 0.0 and np.isfinite(h2_full) and h2_full > 0.0:
        rg_full = float(gamma_full / np.sqrt(h1_full * h2_full))
        d_gamma = 1.0 / np.sqrt(h1_full * h2_full)
    else:
        rg_full = np.nan
        d_gamma = np.nan

    se_den = _delta_rg_se_from_h2_only(
        rg_full=rg_full,
        v1_full=h1_full,
        v2_full=h2_full,
        v1_reps=h1_reps,
        v2_reps=h2_reps,
    )

    if np.isfinite(d_gamma) and np.isfinite(se_den):
        se_rg = float(np.sqrt((d_gamma * se_gamma) ** 2 + se_den ** 2))
    elif np.isfinite(d_gamma):
        se_rg = float(abs(d_gamma) * se_gamma)
    else:
        se_rg = np.nan

    return {
        "se_gamma": se_gamma,
        "se_rg": se_rg,
        "var_gamma": float(var_gamma),
        "moment_source": moment_source,
        "alpha_probe": alpha_probe,
        "alpha_probe_err": alpha_probe_err,
        "delta_reff": delta_reff,
        "k2_probe_relerr": k2_probe_relerr,
        "approximate_moments": approximate_moments,
        "drop_frac_total": float(drop_frac_total),
        "approximate_sample_match": approximate_sample_match,
        "sample_warning": sample_warning,
        "scale_rel_gap": float(scale_rel_gap),
        "overlap_rel_gap": float(overlap_rel_gap) if np.isfinite(overlap_rel_gap) else np.nan,
        "sample_rel_gap": float(sample_rel_gap),
        "overlap_check": overlap_check,
    }


def fit_rg(
    prepared: RGPrepared,
    h2_fit1,
    h2_fit2,
    intercept_fit: InterceptFit,
    *,
    rg_se_method: str = "jackknife",
    jack_mode: str = "mean",
    nan_policy: str = "omit",
) -> RGFit:
    p = prepared
    K = p.trace_view.nbins
    R = p.jackknife.nrep

    if intercept_fit.c_reps.shape != (R + 1,):
        raise ValueError("intercept_fit.c_reps shape mismatch with RGPrepared replicates.")

    c_reps = np.asarray(intercept_fit.c_reps, dtype=np.float64)
    sqrt_n1n2 = float(np.sqrt(float(p.n1_scale) * float(p.n2_scale)))

    rhs = np.full((R + 1, K), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhs = ((p.Ay_rep - c_reps[:, None] * p.Ak_rep) * sqrt_n1n2) / p.Ak_rep
    bad_rhs = (~np.isfinite(rhs)) | (~np.isfinite(p.Ak_rep)) | (p.Ak_rep <= 0.0)
    rhs[bad_rhs] = np.nan

    gamma_reps = _solve_linear_batch(p.lhs, rhs)
    est, se = p.jackknife.summarize(
        gamma_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    gamma = np.column_stack([est, se])

    v1 = np.asarray(h2_fit1.sigma_reps[:, :K], dtype=np.float64)
    v2 = np.asarray(h2_fit2.sigma_reps[:, :K], dtype=np.float64)
    rg_reps = _component_rg(gamma_reps, v1, v2)

    est, se = p.jackknife.summarize(
        rg_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    rg = np.column_stack([est, se])

    gamma_tot_reps = np.nansum(gamma_reps, axis=1)
    est, se = p.jackknife.summarize(
        gamma_tot_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    gamma_total = np.array([float(est), float(se)], dtype=np.float64)

    h2_tot1 = np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64)
    h2_tot2 = np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64)

    rg_tot_reps = _component_rg(gamma_tot_reps, h2_tot1, h2_tot2)
    est, se = p.jackknife.summarize(
        rg_tot_reps,
        unit_sizes=p.unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    rg_total = np.array([float(est), float(se)], dtype=np.float64)

    rg_se_method = str(rg_se_method).strip().lower()
    if rg_se_method not in {"jackknife", "delta", "robust", "kmoments"}:
        raise ValueError("rg_se_method must be one of {'jackknife','delta','robust','kmoments'}")

    kmoment_info = None

    if rg_se_method == "delta":
        rg_total_delta = _estimate_total_rg_delta_se(
            gamma_tot_reps,
            h2_tot1,
            h2_tot2,
            p.jackknife,
            unit_sizes=p.unit_sizes,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        rg_total = rg_total_delta
        if K == 1 and np.isfinite(rg_total[0]):
            rg[0, 0] = rg_total[0]
            rg[0, 1] = rg_total[1]

    elif rg_se_method == "robust":
        if K != 1:
            raise ValueError(
                "rg_se_method='robust' is currently implemented only for the single-component case (K == 1)."
            )

        gamma_full = float(np.asarray(gamma_reps[-1, 0], dtype=np.float64))
        rg_full_bin = float(np.asarray(rg_reps[-1, 0], dtype=np.float64))
        rg_full_total = float(rg_tot_reps[-1]) if np.isfinite(rg_tot_reps[-1]) else np.nan

        rob = _estimate_fixedc_cluster_robust_se_single_component(
            p,
            intercept_fit,
            h2_fit1,
            h2_fit2,
            gamma_full=gamma_full,
            rg_full_bin=rg_full_bin,
            rg_full_total=rg_full_total,
            robust_kind="cr2",
            add_external_c_se=False,
            jack_mode=jack_mode,
            nan_policy=nan_policy,
        )

        gamma[0, 1] = float(rob["se_gamma"])
        gamma_total[1] = float(rob["se_gamma"])

        if np.isfinite(rob["se_rg_bin"]):
            rg[0, 1] = float(rob["se_rg_bin"])

        if np.isfinite(rob["se_rg_total"]):
            rg_total[1] = float(rob["se_rg_total"])

    elif rg_se_method == "kmoments":
        if K != 1:
            raise ValueError(
                "rg_se_method='kmoments' is currently implemented only for the single-component case (K == 1)."
            )

        gamma_full = float(np.asarray(gamma_reps[-1, 0], dtype=np.float64))
        rg_full_bin = float(np.asarray(rg_reps[-1, 0], dtype=np.float64))
        rg_full_total = float(rg_tot_reps[-1]) if np.isfinite(rg_tot_reps[-1]) else np.nan

        km = _estimate_kmoment_model_se_single_component(
            p,
            intercept_fit,
            h2_fit1,
            h2_fit2,
            gamma_full=gamma_full,
            alpha_tol=5e-3,
            jack_mode=jack_mode,
            nan_policy=nan_policy,
        )

        gamma[0, 1] = float(km["se_gamma"])
        gamma_total[1] = float(km["se_gamma"])

        if np.isfinite(km["se_rg"]):
            rg[0, 1] = float(km["se_rg"])
            rg_total[1] = float(km["se_rg"])

        kmoment_info = km

    return RGFit(
        prepared=prepared,
        intercept=intercept_fit,
        h2_fit1=h2_fit1,
        h2_fit2=h2_fit2,
        gamma_reps=gamma_reps,
        rg_reps=rg_reps,
        gamma=gamma,
        rg=rg,
        gamma_total=gamma_total,
        rg_total=rg_total,
        rg_se_method=rg_se_method,
        kmoment_info=kmoment_info,
    )

# -----------------------------------------------------------------------------
# intercept estimation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InterceptRegressionSystem:
    x: np.ndarray          # (M,P) regression LD design used in the fitted mean
    a: np.ndarray          # (M,P) score-side mass design used in beta(c)=solve(A^T X, A^T(y-c))
    total_ld: np.ndarray   # (M,)
    full_mass: np.ndarray  # (P,) masses on the FULL trace axis for beta -> gamma_total conversion
    source: str
    mode: str


def _as_2d_float_array(x, *, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64, order="C")
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise RuntimeError(f"{name} must be 1D or 2D.")
    return x


def _select_intercept_regression_system(trace_view, *, collapse_reg_ld=False, log=None):
    A_main = _as_2d_float_array(trace_view.annot, name="annotation design")
    L_main = _as_2d_float_array(trace_view.ldscores, name="primary ldscores")
    if A_main.shape[0] != L_main.shape[0]:
        raise RuntimeError("Annotation design and primary ldscores must share the same SNP axis.")

    M = int(A_main.shape[0])

    source = "main"
    X_raw = L_main
    if getattr(trace_view, "ldscores_reg", None) is not None:
        X_raw = _as_2d_float_array(trace_view.ldscores_reg, name="ldscores_reg")
        if X_raw.shape[0] != M:
            raise RuntimeError("ldscores_reg and annotation design must share the same SNP axis.")
        source = "reg"

    P = int(X_raw.shape[1])

    if P == 1:
        if collapse_reg_ld and log is not None:
            log._log(
                "WARNING: --collapse-reg-ld was set, but the intercept regression LD "
                "score is already 1D; no collapse was applied."
            )
        return InterceptRegressionSystem(
            x=np.asarray(X_raw, dtype=np.float64, order="C"),
            a=np.ones((M, 1), dtype=np.float64),
            total_ld=np.asarray(X_raw[:, 0], dtype=np.float64, order="C"),
            full_mass=np.array([float(trace_view.nsnps)], dtype=np.float64),
            source=source,
            mode=source,
        )

    if not collapse_reg_ld:
        if source == "reg":
            raise RuntimeError(
                "--ldscores-reg must contain exactly one LD-score column for unconstrained "
                "rg intercept estimation. If the regression LD scores come from non-overlapping "
                "annotations, pre-collapse them to total LD before passing --ldscores-reg, or pass "
                "--collapse-reg-ld explicitly. Do not collapse overlapping annotations."
            )
        raise RuntimeError(
            "Unconstrained rg intercept estimation requires a 1D regression LD score. "
            "The primary --ldscores file has multiple LD-score columns and no 1D "
            "--ldscores-reg was provided. Provide a pre-collapsed --ldscores-reg file, "
            "or pass --collapse-reg-ld explicitly if the primary LD-score columns are "
            "non-overlapping and can be safely summed."
        )

    if log is not None:
        label = "primary --ldscores" if source == "main" else "--ldscores-reg"
        log._log(
            f"WARNING: --collapse-reg-ld is a compatibility option. Collapsing "
            f"{P}-column {label} to total LD for scalar intercept regression. "
            "This is only valid when the LD-score columns are non-overlapping."
        )
    X = np.sum(X_raw, axis=1, dtype=np.float64, keepdims=True)
    return InterceptRegressionSystem(
        x=np.asarray(X, dtype=np.float64, order="C"),
        a=np.ones((M, 1), dtype=np.float64),
        total_ld=np.asarray(X[:, 0], dtype=np.float64, order="C"),
        full_mass=np.array([float(trace_view.nsnps)], dtype=np.float64),
        source=source,
        mode=f"{source}-collapsed",
    )


def _select_intercept_weight_ld(trace_view, regsys: InterceptRegressionSystem, *, mode: str, weight_ld_override=None, log=None):
    mode = str(mode).strip().lower()
    if mode not in {"ldsc", "score"}:
        raise ValueError("mode must be one of {'ldsc','score'}")

    P = int(regsys.x.shape[1])
    if P != 1:
        raise RuntimeError("Internal error: unconstrained rg intercept regression must use a 1D LD score.")

    if mode == "score":
        if weight_ld_override is not None and log is not None:
            log._log(
                "[rg:c] ignoring ldscores_reg_w in scalar SCORE-weight mode; "
                "the legacy exact single-component estimator uses the regression LD itself."
            )
        return np.asarray(regsys.total_ld, dtype=np.float64, order="C"), "score-legacy-total-ld"

    raw = weight_ld_override
    source = None
    if raw is None and getattr(trace_view, "ldscores_reg_w", None) is not None:
        raw = getattr(trace_view, "ldscores_reg_w")
        source = "trace-view-reg-w"
    elif raw is not None:
        source = "override-reg-w"

    if raw is not None:
        w = np.asarray(raw, dtype=np.float64, order="C")
        if w.ndim == 2:
            if w.shape[1] != 1:
                raise RuntimeError("ldscores_reg_w must be 1D or single-column.")
            w = w[:, 0]
        elif w.ndim != 1:
            raise RuntimeError("ldscores_reg_w must be 1D or single-column.")
        if w.size != int(trace_view.nsnps):
            raise RuntimeError(
                f"ldscores_reg_w length mismatch with TraceView SNP axis: {w.size} vs {trace_view.nsnps}."
            )
        if log is not None:
            log._log("[rg:c] using explicit 1D regression-weight LD (ldscores_reg_w) for intercept weights.")
        return np.asarray(w, dtype=np.float64, order="C"), str(source or "reg-w")

    if log is not None and mode == "ldsc":
        log._log(
            "[rg:c] no ldscores_reg_w provided; scalar LDSC-weighted intercept regression "
            "falls back to the legacy total regression LD weights."
        )
    return np.asarray(regsys.total_ld, dtype=np.float64, order="C"), "fallback-total-ld"


def _make_intercept_keep_mask(
    z1,
    z2,
    total_ld,
    y=None,
    *,
    weight_ld=None,
    nsamp_max,
    chisq_threshold=None,
    chisq_mode="either",
):
    z1 = np.asarray(z1, dtype=np.float64).ravel()
    z2 = np.asarray(z2, dtype=np.float64).ravel()
    total_ld = np.asarray(total_ld, dtype=np.float64).ravel()
    if not (z1.size == z2.size == total_ld.size):
        raise ValueError("z1/z2/total_ld length mismatch in intercept keep-mask construction.")

    if y is None:
        finite_y = np.ones(total_ld.size, dtype=bool)
    else:
        y = np.asarray(y, dtype=np.float64).ravel()
        if y.size != total_ld.size:
            raise ValueError("y length mismatch in intercept keep-mask construction.")
        finite_y = np.isfinite(y)

    if weight_ld is None:
        finite_wld = np.ones(total_ld.size, dtype=bool)
    else:
        weight_ld = np.asarray(weight_ld, dtype=np.float64).ravel()
        if weight_ld.size != total_ld.size:
            raise ValueError("weight_ld length mismatch in intercept keep-mask construction.")
        finite_wld = np.isfinite(weight_ld) & (weight_ld > 0.0)

    finite_z = np.isfinite(z1) & np.isfinite(z2)
    finite_total_ld = np.isfinite(total_ld) & (total_ld > 0.0)
    base = finite_z & finite_total_ld & finite_wld & finite_y

    mode = str(chisq_mode).strip().lower()
    chisq_keep = np.ones(total_ld.size, dtype=bool)
    chisq_thr, chisq_thr_mode = utils._resolve_chisq_threshold(float(nsamp_max), chisq_threshold)
    chisq_thr_used = None

    if chisq_thr is not None:
        chisq_thr = float(chisq_thr)
        if np.isfinite(chisq_thr) and chisq_thr > 0.0:
            c1 = z1 * z1
            c2 = z2 * z2
            c3 = np.abs(z1 * z2)
            if mode in ("either", "max"):
                chisq_keep = (c1 <= chisq_thr) & (c2 <= chisq_thr) & (c3 <= chisq_thr)
            elif mode == "both":
                chisq_keep = ~((c1 > chisq_thr) & (c2 > chisq_thr))
            else:
                raise ValueError("chisq_mode must be one of {'either','both','max'}")
            chisq_thr_used = chisq_thr

    keep = base & chisq_keep

    info = {
        "threshold": chisq_thr_used,
        "threshold_mode": chisq_thr_mode,
        "chisq_mode": mode,
        "n_total": int(total_ld.size),
        "n_base": int(np.sum(base)),
        "n_kept": int(np.sum(keep)),
        "n_removed": int(np.sum(~keep)),
        "n_removed_nonfinite_z": int(np.sum(~finite_z)),
        "n_removed_nonfinite_or_nonpositive_total_ld": int(np.sum(~finite_total_ld)),
        "n_removed_nonfinite_or_nonpositive_weight_ld": int(np.sum(~finite_wld)),
        "n_removed_nonfinite_summary_y": int(np.sum(~finite_y)) if y is not None else 0,
        "n_removed_chisq": int(np.sum(base & (~chisq_keep))),
    }
    return keep, info


def _build_simple_intercept_weights(weight_ld, keep):
    weight_ld = np.asarray(weight_ld, dtype=np.float64).ravel()
    keep = np.asarray(keep, dtype=bool).ravel()
    if weight_ld.size != keep.size:
        raise ValueError("weight_ld/keep length mismatch.")
    if np.any(keep & ((~np.isfinite(weight_ld)) | (weight_ld <= 0.0))):
        raise ValueError("Regression-weight LD must be finite and >0 for kept SNPs in the intercept regression.")
    w = np.zeros(weight_ld.size, dtype=np.float64)
    w[keep] = 1.0 / weight_ld[keep]
    if not np.isfinite(w).all() or np.sum(w) <= 0.0:
        raise ValueError("Invalid intercept regression weights.")
    return w


def _compute_intercept_unit_summaries(jackknife, a, x, y, keep):
    a = np.asarray(a, dtype=np.float64, order="C")
    x = np.asarray(x, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()
    keep = np.asarray(keep, dtype=bool).ravel()

    if a.ndim != 2 or x.ndim != 2:
        raise ValueError("a and x must be 2D arrays.")
    if a.shape != x.shape:
        raise ValueError("a and x must have the same shape.")
    if y.size != a.shape[0] or keep.size != a.shape[0]:
        raise ValueError("Axis length mismatch in intercept unit summaries.")

    U = int(jackknife.nunit)
    P = int(x.shape[1])

    m_u = np.zeros((U, P), dtype=np.float64)
    t_u = np.zeros((U, P), dtype=np.float64)
    S_u = np.zeros((U, P, P), dtype=np.float64)

    for u in range(U):
        s = int(jackknife.starts[u])
        e = int(jackknife.ends[u])
        if e <= s:
            continue
        mu = keep[s:e]
        if not np.any(mu):
            continue
        au = a[s:e, :][mu, :]
        xu = x[s:e, :][mu, :]
        yu = y[s:e][mu]
        m_u[u] = au.sum(axis=0, dtype=np.float64)
        t_u[u] = au.T @ yu
        S_u[u] = au.T @ xu

    return m_u, t_u, S_u


def _compute_weighted_intercept_unit_summaries(jackknife, x, y, w):
    x = np.asarray(x, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if y.size != x.shape[0] or w.size != x.shape[0]:
        raise ValueError("Axis length mismatch in weighted intercept unit summaries.")

    U = int(jackknife.nunit)
    P = int(x.shape[1])

    W_u = np.zeros(U, dtype=np.float64)
    XW_u = np.zeros((U, P), dtype=np.float64)
    XXW_u = np.zeros((U, P, P), dtype=np.float64)
    Sy_u = np.zeros(U, dtype=np.float64)
    XWy_u = np.zeros((U, P), dtype=np.float64)

    for u in range(U):
        s = int(jackknife.starts[u])
        e = int(jackknife.ends[u])
        if e <= s:
            continue
        wu = w[s:e]
        if not np.any(wu != 0.0):
            continue
        xu = x[s:e, :]
        yu = y[s:e]
        wyu = wu * yu
        W_u[u] = float(np.sum(wu))
        XW_u[u] = np.einsum("ni,n->i", xu, wu, optimize=True)
        XXW_u[u] = np.einsum("ni,n,nj->ij", xu, wu, xu, optimize=True)
        Sy_u[u] = float(np.dot(wu, yu))
        XWy_u[u] = np.einsum("ni,n->i", xu, wyu, optimize=True)

    return W_u, XW_u, XXW_u, Sy_u, XWy_u


def _compute_weighted_intercept_summaries(x, y, w):
    x = np.asarray(x, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")
    if y.size != x.shape[0] or w.size != x.shape[0]:
        raise ValueError("Axis length mismatch in weighted intercept summaries.")
    wy = w * y
    W = float(np.sum(w))
    XW = np.einsum("ni,n->i", x, w, optimize=True)
    XXW = np.einsum("ni,n,nj->ij", x, w, x, optimize=True)
    Sy = float(np.dot(w, y))
    XWy = np.einsum("ni,n->i", x, wy, optimize=True)
    return W, XW, XXW, Sy, XWy


def _solve_constrained_intercept_from_sums(
    m_fit,
    S_fit,
    t_fit,
    W,
    XW,
    XXW,
    Sy,
    XWy,
    *,
    denom_floor=0.0,
):
    S_fit = np.asarray(S_fit, dtype=np.float64)
    squeeze = False
    if S_fit.ndim == 2:
        squeeze = True
        S_fit = S_fit[None, :, :]
        m_fit = np.asarray(m_fit, dtype=np.float64).reshape(1, -1)
        t_fit = np.asarray(t_fit, dtype=np.float64).reshape(1, -1)
        W = np.asarray([W], dtype=np.float64)
        XW = np.asarray(XW, dtype=np.float64).reshape(1, -1)
        XXW = np.asarray(XXW, dtype=np.float64).reshape(1, S_fit.shape[1], S_fit.shape[2])
        Sy = np.asarray([Sy], dtype=np.float64)
        XWy = np.asarray(XWy, dtype=np.float64).reshape(1, -1)
    elif S_fit.ndim == 3:
        m_fit = np.asarray(m_fit, dtype=np.float64)
        t_fit = np.asarray(t_fit, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64).ravel()
        XW = np.asarray(XW, dtype=np.float64)
        XXW = np.asarray(XXW, dtype=np.float64)
        Sy = np.asarray(Sy, dtype=np.float64).ravel()
        XWy = np.asarray(XWy, dtype=np.float64)
    else:
        raise ValueError("S_fit must be 2D or 3D.")

    R, P, P2 = S_fit.shape
    if P != P2:
        raise ValueError("S_fit must have square trailing dimensions.")
    if m_fit.shape != (R, P) or t_fit.shape != (R, P):
        raise ValueError("m_fit and t_fit must have shape (R,P).")
    if W.shape != (R,) or Sy.shape != (R,):
        raise ValueError("W and Sy must have shape (R,).")
    if XW.shape != (R, P) or XWy.shape != (R, P):
        raise ValueError("XW and XWy must have shape (R,P).")
    if XXW.shape != (R, P, P):
        raise ValueError("XXW must have shape (R,P,P).")

    alpha = _solve_linear_batch(S_fit, t_fit)
    delta = _solve_linear_batch(S_fit, m_fit)

    c = np.full(R, np.nan, dtype=np.float64)
    beta = np.full((R, P), np.nan, dtype=np.float64)

    good = (
        np.isfinite(W) & (W > 0.0) & np.isfinite(Sy) &
        np.isfinite(S_fit).all(axis=(1, 2)) &
        np.isfinite(m_fit).all(axis=1) &
        np.isfinite(t_fit).all(axis=1) &
        np.isfinite(XW).all(axis=1) &
        np.isfinite(XXW).all(axis=(1, 2)) &
        np.isfinite(XWy).all(axis=1) &
        np.isfinite(alpha).all(axis=1) &
        np.isfinite(delta).all(axis=1)
    )

    if np.any(good):
        num = (
            Sy
            - np.einsum("ri,ri->r", delta, XWy)
            - np.einsum("ri,ri->r", alpha, XW)
            + np.einsum("ri,rij,rj->r", alpha, XXW, delta)
        )
        den = (
            W
            - 2.0 * np.einsum("ri,ri->r", delta, XW)
            + np.einsum("ri,rij,rj->r", delta, XXW, delta)
        )

        if denom_floor is not None:
            denom_floor = float(denom_floor)
            if np.isfinite(denom_floor) and denom_floor > 0.0:
                den = np.where(np.isfinite(den) & (den > 0.0), np.maximum(den, denom_floor), den)

        good = good & np.isfinite(num) & np.isfinite(den) & (den > 0.0)
        if np.any(good):
            c[good] = num[good] / den[good]
            beta[good] = alpha[good] - c[good, None] * delta[good]
            good = good & np.isfinite(c) & np.isfinite(beta).all(axis=1)
            c[~good] = np.nan
            beta[~good] = np.nan

    if squeeze:
        return float(c[0]), beta[0], bool(good[0])
    return c, beta, good


def _full_intercept_denominator(m_fit, S_fit, W, XW, XXW):
    S_fit = np.asarray(S_fit, dtype=np.float64)
    if S_fit.ndim != 2:
        raise ValueError("S_fit must be 2D for the full-sample denominator.")
    delta = _solve_linear_batch(S_fit[None, :, :], np.asarray(m_fit, dtype=np.float64).reshape(1, -1))[0]
    if not np.isfinite(delta).all():
        return np.nan
    den = (
        float(W)
        - 2.0 * float(np.dot(delta, np.asarray(XW, dtype=np.float64)))
        + float(delta @ np.asarray(XXW, dtype=np.float64) @ delta)
    )
    return float(den) if np.isfinite(den) else np.nan


def _intercept_gamma_total_from_beta(beta, mass, sqrt_n1n2):
    beta = np.asarray(beta, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if beta.ndim == 1:
        beta = beta[None, :]
    if beta.ndim != 2:
        raise ValueError("beta must be 1D or 2D.")

    if mass.ndim == 1:
        mass = np.broadcast_to(mass.reshape(1, -1), beta.shape)
    elif mass.ndim == 2:
        if mass.shape != beta.shape:
            raise ValueError("beta/mass dimension mismatch.")
    else:
        raise ValueError("mass must be 1D or 2D.")

    s = float(sqrt_n1n2)
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("sqrt_n1n2 must be positive and finite.")
    return np.einsum("rp,rp->r", beta, mass, optimize=True) / s


def _score_gamma_total_from_c(prepared: RGPrepared, c, *, rep_index: int):
    rep_index = int(rep_index)
    if rep_index < 0:
        rep_index += prepared.jackknife.nrep + 1
    if rep_index < 0 or rep_index > int(prepared.jackknife.nrep):
        raise IndexError(f"rep_index out of range: {rep_index}")

    c = float(c)
    if not np.isfinite(c):
        return np.nan

    Ak = np.asarray(prepared.Ak_rep[rep_index], dtype=np.float64).reshape(1, -1)
    Ay = np.asarray(prepared.Ay_rep[rep_index], dtype=np.float64).reshape(1, -1)
    lhs = np.asarray(prepared.lhs[rep_index], dtype=np.float64).reshape(1, prepared.trace_view.nbins, prepared.trace_view.nbins)
    sqrt_n1n2 = float(np.sqrt(float(prepared.n1_scale) * float(prepared.n2_scale)))

    with np.errstate(divide="ignore", invalid="ignore"):
        rhs = ((Ay - c * Ak) * sqrt_n1n2) / Ak
    bad = (~np.isfinite(rhs)) | (~np.isfinite(Ak)) | (Ak <= 0.0)
    rhs[bad] = np.nan

    gamma = _solve_linear_batch(lhs, rhs)[0]
    if gamma.ndim != 1 or gamma.size != prepared.trace_view.nbins:
        return np.nan
    if not np.isfinite(gamma).any():
        return np.nan
    return float(np.nansum(gamma))


def _replicate_active_mask(jackknife, keep, rep_index: int) -> np.ndarray:
    keep = np.asarray(keep, dtype=bool)
    rep_index = int(rep_index)
    if rep_index == int(jackknife.nrep):
        return keep
    unit_id = np.asarray(jackknife.unit_id, dtype=np.int64)
    D = np.asarray(jackknife.D, dtype=np.float64)
    if D.shape[0] != int(jackknife.nrep) or unit_id.size != keep.size:
        raise ValueError("Jackknife replicate mask shape mismatch.")
    return keep & (D[rep_index, unit_id] < 0.5)


def _ldsc_gencov_weights_1d(
    ld,
    w_ld,
    n1,
    n2,
    m_tot,
    h1,
    h2,
    rho_g,
    *,
    intercept_gencov=0.0,
    intercept_hsq1=1.0,
    intercept_hsq2=1.0,
    intercept_hsq_floor=1e-8,
    weight_floor=1e-12,
):
    ld = np.asarray(ld, dtype=np.float64).ravel()
    w_ld = np.asarray(w_ld, dtype=np.float64).ravel()
    n1 = np.asarray(n1, dtype=np.float64).ravel()
    n2 = np.asarray(n2, dtype=np.float64).ravel()
    if not (ld.size == w_ld.size == n1.size == n2.size):
        raise ValueError("ld, w_ld, n1, n2 must all have the same length.")

    m_tot = float(m_tot)
    if not (np.isfinite(m_tot) and m_tot > 0.0):
        raise ValueError("m_tot must be positive and finite.")

    if intercept_gencov is None:
        intercept_gencov = 0.0
    if intercept_hsq1 is None:
        intercept_hsq1 = 1.0
    if intercept_hsq2 is None:
        intercept_hsq2 = 1.0

    h1 = min(max(float(h1), 0.0), 1.0)
    h2 = min(max(float(h2), 0.0), 1.0)
    rho_g = min(max(float(rho_g), -1.0), 1.0)
    intercept_gencov = float(intercept_gencov)
    intercept_hsq1 = float(intercept_hsq1)
    intercept_hsq2 = float(intercept_hsq2)

    int_floor = float(intercept_hsq_floor)
    if not (np.isfinite(int_floor) and int_floor > 0.0):
        int_floor = 1e-8
    intercept_hsq1 = max(intercept_hsq1, int_floor)
    intercept_hsq2 = max(intercept_hsq2, int_floor)

    ld_eff = np.fmax(ld, 1.0)
    w_ld_eff = np.fmax(w_ld, 1.0)

    a = (n1 * (h1 * ld_eff) / m_tot) + intercept_hsq1
    b = (n2 * (h2 * ld_eff) / m_tot) + intercept_hsq2
    c = (np.sqrt(n1 * n2) * (rho_g * ld_eff) / m_tot) + intercept_gencov

    eps = float(weight_floor)
    if not (np.isfinite(eps) and eps > 0.0):
        eps = 1e-12
    den = a * b + c * c
    den = np.where(np.isfinite(den), den, np.inf)
    den = np.maximum(den, eps)
    w = 1.0 / (w_ld_eff * den)
    if not np.isfinite(w).all():
        raise ValueError("Non-finite LDSC intercept weights encountered.")
    return w


def _make_fixed_intercept_fit(
    trace_view,
    matched1,
    matched2,
    jackknife,
    fixed_c,
    *,
    summary_y=None,
    summary_y_info=None,
    info=None,
    log=None,
):
    fixed_c = float(fixed_c)
    if not np.isfinite(fixed_c):
        raise ValueError("fixed_c must be finite.")

    meta = {} if info is None else dict(info)
    source = str(meta.get("source", "external"))

    c_se = meta.get("external_c_se", 0.0)
    try:
        c_se = float(c_se)
    except Exception:
        c_se = 0.0
    if not np.isfinite(c_se) or c_se < 0.0:
        c_se = 0.0

    if summary_y is None:
        summary_y, summary_y_info = build_rg_summary_moment(matched1, matched2)

    y = np.asarray(summary_y, dtype=np.float64)
    active_mask = np.isfinite(y)
    unit_sizes = jackknife.unit_sizes(active_mask=active_mask, dtype=np.float64)

    c_reps = np.full(jackknife.nrep + 1, fixed_c, dtype=np.float64)
    c = np.array([fixed_c, c_se], dtype=np.float64)

    meta["fixed"] = True
    meta.setdefault("source", source)
    if summary_y_info is not None:
        meta["summary_y_mode"] = str(summary_y_info.get("mode", "unknown"))
        meta["trait1_n_scale"] = float(summary_y_info.get("trait1_n_scale", getattr(matched1, "n_scale", matched1.nsamp)))
        meta["trait2_n_scale"] = float(summary_y_info.get("trait2_n_scale", getattr(matched2, "n_scale", matched2.nsamp)))

    if log is not None:
        if c_se > 0.0:
            log._log(
                f"[rg:c] using fixed external c={fixed_c:.6g} (SE: {c_se:.6g}) "
                f"(source={source}); intercept regression skipped."
            )
        else:
            log._log(
                f"[rg:c] using fixed external c={fixed_c:.6g} "
                f"(source={source}); intercept regression skipped."
            )

    return InterceptFit(
        trace_view=trace_view,
        matched1=matched1,
        matched2=matched2,
        jackknife=jackknife,
        active_mask=active_mask,
        unit_sizes=unit_sizes,
        ld=np.zeros(trace_view.nsnps, dtype=np.float64),
        y=y,
        c_reps=c_reps,
        c=c,
        info=meta,
    )


def fit_intercept(
    trace_view,
    matched1,
    matched2,
    jackknife,
    h2_fit1,
    h2_fit2,
    *,
    summary_y=None,
    summary_y_info=None,
    fixed_c=None,
    fixed_info=None,
    intercept_chisq_threshold=None,
    collapse_reg_ld: bool = False,
    chisq_mode: str = "either",
    intercept_weight_mode: str = "ldsc",
    irwls_iters: int = 2,
    intercept_hsq1: float = 1.0,
    intercept_hsq2: float = 1.0,
    intercept_hsq_floor: float = 1e-8,
    intercept_weight_floor: float = 1e-12,
    denom_floor_rel: float = 1e-12,
    weight_ld_override=None,
    score_prepared: RGPrepared | None = None,
    log=None,
    jack_mode: str = "mean",
    nan_policy: str = "omit",
    **_unused_kwargs,
) -> InterceptFit:
    _validate_common_axis(trace_view, matched1, matched2, jackknife)
    if fixed_c is not None:
        return _make_fixed_intercept_fit(
            trace_view,
            matched1,
            matched2,
            jackknife,
            fixed_c,
            summary_y=summary_y,
            summary_y_info=summary_y_info,
            info=fixed_info,
            log=log,
        )

    mode = str(intercept_weight_mode).strip().lower()
    if mode not in {"ldsc", "score"}:
        raise ValueError("intercept_weight_mode must be one of {'ldsc','score'}")

    if summary_y is None:
        summary_y, summary_y_info = build_rg_summary_moment(matched1, matched2)

    score_prepared_local = score_prepared
    if mode == "ldsc" and score_prepared_local is None:
        score_prepared_local = prepare_rg(
            trace_view,
            matched1,
            matched2,
            jackknife,
            summary_y=summary_y,
            summary_y_info=summary_y_info,
            adjust_delta=False,
        )

    regsys = _select_intercept_regression_system(
        trace_view,
        collapse_reg_ld=collapse_reg_ld,
        log=log,
    )

    x = np.asarray(regsys.x, dtype=np.float64, order="C")
    a = np.asarray(regsys.a, dtype=np.float64, order="C")
    total_ld = np.asarray(regsys.total_ld, dtype=np.float64).ravel()
    full_mass = np.asarray(regsys.full_mass, dtype=np.float64).ravel()

    if x.shape != a.shape:
        raise RuntimeError("Intercept regression design and score-side mass design must have the same shape.")
    if x.shape[0] != trace_view.nsnps or total_ld.size != trace_view.nsnps:
        raise RuntimeError("Intercept regression design was not built on the main SNP axis.")

    P = int(x.shape[1])
    R = int(jackknife.nrep)

    z1 = np.asarray(matched1.z, dtype=np.float64)
    z2 = np.asarray(matched2.z, dtype=np.float64)
    y = np.asarray(summary_y, dtype=np.float64)
    if y.ndim != 1 or y.size != total_ld.size:
        raise ValueError(f"summary_y must have shape ({total_ld.size},), got {y.shape}")

    weight_ld, weight_ld_source = _select_intercept_weight_ld(
        trace_view,
        regsys,
        mode=mode,
        weight_ld_override=weight_ld_override,
        log=log,
    )

    n1_scalar = float(
        summary_y_info.get("trait1_n_scale", getattr(matched1, "n_scale", matched1.nsamp))
        if summary_y_info is not None else getattr(matched1, "n_scale", matched1.nsamp)
    )
    n2_scalar = float(
        summary_y_info.get("trait2_n_scale", getattr(matched2, "n_scale", matched2.nsamp))
        if summary_y_info is not None else getattr(matched2, "n_scale", matched2.nsamp)
    )
    nsamp_max = max(n1_scalar, n2_scalar)

    keep, info = _make_intercept_keep_mask(
        z1,
        z2,
        total_ld,
        y=y,
        weight_ld=weight_ld,
        nsamp_max=nsamp_max,
        chisq_threshold=intercept_chisq_threshold,
        chisq_mode=chisq_mode,
    )

    info["ld_source"] = regsys.source
    info["weight_ld_source"] = weight_ld_source
    info["regression_design_mode"] = regsys.mode
    info["regression_ncoef"] = P
    info["weight_mode"] = mode
    info["summary_y_mode"] = None if summary_y_info is None else str(summary_y_info.get("mode", "unknown"))
    info["trait1_n_scale"] = n1_scalar
    info["trait2_n_scale"] = n2_scalar
    if summary_y_info is not None:
        for key in (
            "trait1_cov_rank",
            "trait1_cov_rank_source",
            "trait2_cov_rank",
            "trait2_cov_rank_source",
        ):
            if key in summary_y_info:
                info[key] = summary_y_info[key]

    if info["n_kept"] <= 1:
        raise RuntimeError("Intercept regression has <=1 SNP after filtering.")

    if log is not None and info.get("threshold") is not None:
        tag = " (auto)" if info.get("threshold_mode") == "auto" else ""
        log._log(
            f"[rg:c] intercept chi^2 filter: threshold={info['threshold']:.3f}{tag}, "
            f"mode={info['chisq_mode']}, removed={info['n_removed_chisq']} SNPs, "
            f"kept_after_all={info['n_kept']}."
        )

    D = np.asarray(jackknife.D, dtype=np.float64, order="C")
    unit_sizes = jackknife.unit_sizes(active_mask=keep, dtype=np.float64)
    sqrt_n1n2 = float(np.sqrt(n1_scalar * n2_scalar))
    if score_prepared_local is not None:
        m_tot_weight = float(np.sum(np.asarray(score_prepared_local.Ak_rep[-1], dtype=np.float64)))
    else:
        m_tot_weight = float(np.sum(full_mass))
    if not (np.isfinite(sqrt_n1n2) and sqrt_n1n2 > 0.0 and np.isfinite(m_tot_weight) and m_tot_weight > 0.0):
        raise RuntimeError(
            f"Invalid scales for intercept regression: sqrt_n1n2={sqrt_n1n2}, m_tot={m_tot_weight}."
        )

    m_u, t_u, S_u = _compute_intercept_unit_summaries(jackknife, a, x, y, keep)
    m_full = np.sum(m_u, axis=0, dtype=np.float64)
    t_full = np.sum(t_u, axis=0, dtype=np.float64)
    S_full = np.sum(S_u, axis=0, dtype=np.float64)

    m_rep = _stack_delete_replicates(m_full, m_u, D)
    t_rep = _stack_delete_replicates(t_full, t_u, D)
    S_rep = _stack_delete_replicates(S_full, S_u, D)

    if not (np.isfinite(m_full).all() and np.isfinite(t_full).all() and np.isfinite(S_full).all()):
        raise RuntimeError("Intercept regression summaries contain non-finite values.")
    if not np.any(m_full > 0.0):
        raise RuntimeError("Intercept regression retained zero score-side mass after filtering.")

    w_score = _build_simple_intercept_weights(weight_ld, keep)
    W_u0, XW_u0, XXW_u0, Sy_u0, XWy_u0 = _compute_weighted_intercept_unit_summaries(
        jackknife,
        x,
        y,
        w_score,
    )
    W_rep0 = _stack_delete_replicates(np.sum(W_u0, dtype=np.float64), W_u0, D)
    XW_rep0 = _stack_delete_replicates(np.sum(XW_u0, axis=0, dtype=np.float64), XW_u0, D)
    XXW_rep0 = _stack_delete_replicates(np.sum(XXW_u0, axis=0, dtype=np.float64), XXW_u0, D)
    Sy_rep0 = _stack_delete_replicates(np.sum(Sy_u0, dtype=np.float64), Sy_u0, D)
    XWy_rep0 = _stack_delete_replicates(np.sum(XWy_u0, axis=0, dtype=np.float64), XWy_u0, D)

    den0 = _full_intercept_denominator(m_full, S_full, W_rep0[-1], XW_rep0[-1], XXW_rep0[-1])
    floor_rel = float(denom_floor_rel)
    if not (np.isfinite(floor_rel) and floor_rel >= 0.0):
        floor_rel = 1e-12
    denom_floor0 = 0.0 if not np.isfinite(den0) else max(floor_rel * max(float(den0), 1.0), 0.0)

    c_reps, beta_reps, good0 = _solve_constrained_intercept_from_sums(
        m_rep,
        S_rep,
        t_rep,
        W_rep0,
        XW_rep0,
        XXW_rep0,
        Sy_rep0,
        XWy_rep0,
        denom_floor=denom_floor0,
    )
    if not bool(good0[-1]):
        raise RuntimeError("Failed to initialize constrained intercept fit.")

    w_final_full = w_score

    if mode == "ldsc":
        n_iter = int(irwls_iters)
        if n_iter < 0:
            raise ValueError("irwls_iters must be >= 0.")
        if not (
            np.isfinite(n1_scalar) and np.isfinite(n2_scalar) and n1_scalar > 0.0 and n2_scalar > 0.0
        ):
            raise RuntimeError(
                f"Invalid n_scale values for intercept IRWLS: n1={n1_scalar}, n2={n2_scalar}."
            )

        h1_tot_reps = np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64)
        h2_tot_reps = np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64)
        if h1_tot_reps.shape != (R + 1,) or h2_tot_reps.shape != (R + 1,):
            raise ValueError("h2 jackknife replicate shape mismatch with rg intercept replicates.")

        n1_vec = np.full(total_ld.size, n1_scalar, dtype=np.float64)
        n2_vec = np.full(total_ld.size, n2_scalar, dtype=np.float64)

        c_full = float(c_reps[-1])
        beta_full = np.asarray(beta_reps[-1], dtype=np.float64).copy()
        h1_plugin = float(np.clip(h1_tot_reps[-1], 0.0, 1.0))
        h2_plugin = float(np.clip(h2_tot_reps[-1], 0.0, 1.0))

        for _ in range(n_iter):
            rho_full = _score_gamma_total_from_c(score_prepared_local, c_full, rep_index=R)
            if not np.isfinite(rho_full):
                raise RuntimeError("Failed to obtain a finite full-sample SCORE plug-in gamma_g for LDSC-IRWLS intercept weighting.")
            w_cur = _ldsc_gencov_weights_1d(
                ld=total_ld,
                w_ld=weight_ld,
                n1=n1_vec,
                n2=n2_vec,
                m_tot=m_tot_weight,
                h1=h1_plugin,
                h2=h2_plugin,
                rho_g=rho_full,
                intercept_gencov=c_full,
                intercept_hsq1=intercept_hsq1,
                intercept_hsq2=intercept_hsq2,
                intercept_hsq_floor=intercept_hsq_floor,
                weight_floor=intercept_weight_floor,
            )
            w_cur = np.where(keep, w_cur, 0.0)
            Wf, XWf, XXWf, Syf, XWyf = _compute_weighted_intercept_summaries(x, y, w_cur)
            c_new, beta_new, ok = _solve_constrained_intercept_from_sums(
                m_full,
                S_full,
                t_full,
                Wf,
                XWf,
                XXWf,
                Syf,
                XWyf,
                denom_floor=0.0,
            )
            if not ok:
                raise RuntimeError("LDSC-IRWLS intercept update failed on the full sample.")
            c_full = float(c_new)
            beta_full = np.asarray(beta_new, dtype=np.float64)
            w_final_full = w_cur

        den_full = _full_intercept_denominator(m_full, S_full, Wf, XWf, XXWf)
        denom_floor_full = 0.0 if not np.isfinite(den_full) else max(floor_rel * max(float(den_full), 1.0), 0.0)
        c_fin, beta_fin, ok = _solve_constrained_intercept_from_sums(
            m_full,
            S_full,
            t_full,
            Wf,
            XWf,
            XXWf,
            Syf,
            XWyf,
            denom_floor=denom_floor_full,
        )
        if not ok:
            raise RuntimeError("Final constrained full-sample intercept solve failed.")
        c_reps[-1] = float(c_fin)
        beta_reps[-1] = np.asarray(beta_fin, dtype=np.float64)

        for r in range(R):
            active = _replicate_active_mask(jackknife, keep, r)
            if int(np.sum(active)) <= 1:
                c_reps[r] = np.nan
                beta_reps[r] = np.nan
                continue

            h1_r = float(np.clip(h1_tot_reps[r], 0.0, 1.0))
            h2_r = float(np.clip(h2_tot_reps[r], 0.0, 1.0))
            if not (np.isfinite(h1_r) and np.isfinite(h2_r)):
                c_reps[r] = np.nan
                beta_reps[r] = np.nan
                continue

            c_r = float(c_reps[r]) if np.isfinite(c_reps[r]) else float(c_reps[-1])
            beta_r = np.asarray(beta_reps[r], dtype=np.float64)
            if beta_r.shape != (P,) or not np.isfinite(beta_r).all():
                beta_r = np.asarray(beta_reps[-1], dtype=np.float64).copy()

            W_last = XW_last = XXW_last = Sy_last = XWy_last = None
            for _ in range(n_iter):
                rho_r = _score_gamma_total_from_c(score_prepared_local, c_r, rep_index=r)
                if not np.isfinite(rho_r):
                    c_r = np.nan
                    beta_r[:] = np.nan
                    W_last = None
                    break
                w_r = _ldsc_gencov_weights_1d(
                    ld=total_ld,
                    w_ld=weight_ld,
                    n1=n1_vec,
                    n2=n2_vec,
                    m_tot=m_tot_weight,
                    h1=h1_r,
                    h2=h2_r,
                    rho_g=rho_r,
                    intercept_gencov=c_r,
                    intercept_hsq1=intercept_hsq1,
                    intercept_hsq2=intercept_hsq2,
                    intercept_hsq_floor=intercept_hsq_floor,
                    weight_floor=intercept_weight_floor,
                )
                w_r = np.where(active, w_r, 0.0)
                W_last, XW_last, XXW_last, Sy_last, XWy_last = _compute_weighted_intercept_summaries(x, y, w_r)
                c_new, beta_new, ok = _solve_constrained_intercept_from_sums(
                    m_rep[r],
                    S_rep[r],
                    t_rep[r],
                    W_last,
                    XW_last,
                    XXW_last,
                    Sy_last,
                    XWy_last,
                    denom_floor=0.0,
                )
                if not ok:
                    c_r = np.nan
                    beta_r[:] = np.nan
                    W_last = None
                    break
                c_r = float(c_new)
                beta_r = np.asarray(beta_new, dtype=np.float64)

            if W_last is not None and np.isfinite(c_r) and np.isfinite(beta_r).all():
                den_r = _full_intercept_denominator(m_rep[r], S_rep[r], W_last, XW_last, XXW_last)
                denom_floor_r = 0.0 if not np.isfinite(den_r) else max(floor_rel * max(float(den_r), 1.0), 0.0)
                c_fin, beta_fin, ok = _solve_constrained_intercept_from_sums(
                    m_rep[r],
                    S_rep[r],
                    t_rep[r],
                    W_last,
                    XW_last,
                    XXW_last,
                    Sy_last,
                    XWy_last,
                    denom_floor=denom_floor_r,
                )
                if ok:
                    c_r = float(c_fin)
                    beta_r = np.asarray(beta_fin, dtype=np.float64)
                else:
                    c_r = np.nan
                    beta_r[:] = np.nan

            c_reps[r] = c_r
            beta_reps[r] = beta_r

        info["h1_plugin"] = h1_plugin
        info["h2_plugin"] = h2_plugin
        info["irwls_iters"] = n_iter
    else:
        info["irwls_iters"] = 0

    est, se = jackknife.summarize(
        c_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    c = np.array([float(est), float(se)], dtype=np.float64)

    beta_est, beta_se = jackknife.summarize(
        beta_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    beta_est = np.asarray(beta_est, dtype=np.float64)
    beta_se = np.asarray(beta_se, dtype=np.float64)

    gamma_reg_reps = _intercept_gamma_total_from_beta(beta_reps, m_rep, sqrt_n1n2)
    gamma_reg_est, gamma_reg_se = jackknife.summarize(
        gamma_reg_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )

    h2_tot1_reps = np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64)
    h2_tot2_reps = np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64)
    rg_reg_reps = _component_rg(gamma_reg_reps, h2_tot1_reps, h2_tot2_reps)

    rg_reg_est, rg_reg_se = jackknife.summarize(
        rg_reg_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )

    info["regression_gamma_g_total_full"] = float(gamma_reg_reps[-1])
    info["regression_gamma_g_total"] = float(gamma_reg_est)
    info["regression_gamma_g_total_se"] = float(gamma_reg_se)
    info["regression_rg_total_full"] = float(rg_reg_reps[-1]) if np.isfinite(rg_reg_reps[-1]) else np.nan
    info["regression_rg_total"] = float(rg_reg_est)
    info["regression_rg_total_se"] = float(rg_reg_se)

    if P == 1:
        b_reps = np.asarray(beta_reps[:, 0], dtype=np.float64)
        b_est, b_se = jackknife.summarize(
            b_reps,
            unit_sizes=unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        info["regression_slope_full"] = float(b_reps[-1])
        info["regression_slope"] = float(b_est)
        info["regression_slope_se"] = float(b_se)
    else:
        info["regression_beta_full"] = [float(v) if np.isfinite(v) else np.nan for v in np.asarray(beta_reps[-1], dtype=np.float64)]
        info["regression_beta"] = [float(v) if np.isfinite(v) else np.nan for v in beta_est]
        info["regression_beta_se"] = [float(v) if np.isfinite(v) else np.nan for v in beta_se]

    if log is not None and np.isfinite(info.get("regression_rg_total", np.nan)):
        if P == 1:
            log._log(
                f"[rg:c] constrained bivariate-regression totals: "
                f"slope={info['regression_slope']:.6g} (SE: {info['regression_slope_se']:.6g}), "
                f"gamma_g_reg={info['regression_gamma_g_total']:.6g} "
                f"(SE: {info['regression_gamma_g_total_se']:.6g}), "
                f"rg_reg={info['regression_rg_total']:.6g} "
                f"(SE: {info['regression_rg_total_se']:.6g})"
            )
        else:
            log._log(
                f"[rg:c] constrained partitioned bivariate-regression totals: "
                f"gamma_g_reg={info['regression_gamma_g_total']:.6g} "
                f"(SE: {info['regression_gamma_g_total_se']:.6g}), "
                f"rg_reg={info['regression_rg_total']:.6g} "
                f"(SE: {info['regression_rg_total_se']:.6g}), "
                f"ncoef={P}"
            )

    if log is not None:
        n_bad = int(np.sum(~np.isfinite(c_reps[:R])))
        if mode == "ldsc":
            tag = "partitioned" if P > 1 else "scalar"
            log._log(
                f"[rg:c] constrained LDSC-IRWLS ({tag}): h1={info['h1_plugin']:.6g}, "
                f"h2={info['h2_plugin']:.6g}, final_c={c[0]:.6g}, "
                f"bad_reps={n_bad}/{R}, iters={info['irwls_iters']}"
            )
        else:
            tag = "partitioned" if P > 1 else "scalar"
            log._log(
                f"[rg:c] constrained SCORE-weight intercept ({tag}): "
                f"final_c={c[0]:.6g}, bad_reps={n_bad}/{R}"
            )

    return InterceptFit(
        trace_view=trace_view,
        matched1=matched1,
        matched2=matched2,
        jackknife=jackknife,
        active_mask=keep,
        unit_sizes=unit_sizes,
        ld=total_ld,
        y=y,
        c_reps=c_reps,
        c=c,
        info=info,
    )



def _external_c_sensitivity_se(prepared: RGPrepared, h2_fit1, h2_fit2, intercept_fit: InterceptFit):
    info = intercept_fit.info if isinstance(intercept_fit.info, dict) else {}
    if str(info.get("source", "")).lower() != "pheno":
        return None

    c_se = info.get("external_c_se", None)
    try:
        c_se = float(c_se)
    except Exception:
        return None

    if not (np.isfinite(c_se) and c_se > 0.0):
        return None

    p = prepared
    K = p.trace_view.nbins
    sqrt_n1n2 = float(np.sqrt(float(p.n1_scale) * float(p.n2_scale)))

    lhs_full = np.asarray(p.lhs[-1], dtype=np.float64)
    rhs_sens = np.full((1, K), sqrt_n1n2, dtype=np.float64)
    sens = _solve_linear_batch(lhs_full[None, :, :], rhs_sens)[0]
    if sens.shape != (K,) or not np.isfinite(sens).all():
        return None

    dc_gamma = -sens
    gamma_se_ext = np.abs(dc_gamma) * c_se
    gamma_total_se_ext = float(np.abs(np.sum(dc_gamma)) * c_se)

    v1_full = np.asarray(h2_fit1.sigma_reps[-1, :K], dtype=np.float64)
    v2_full = np.asarray(h2_fit2.sigma_reps[-1, :K], dtype=np.float64)
    rg_se_ext = np.full(K, np.nan, dtype=np.float64)
    valid_bin = np.isfinite(v1_full) & np.isfinite(v2_full) & (v1_full > 0.0) & (v2_full > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rg_se_ext[valid_bin] = (
            np.abs(dc_gamma[valid_bin]) * c_se / np.sqrt(v1_full[valid_bin] * v2_full[valid_bin])
        )

    h1_tot = float(h2_fit1.h2_reps[-1, -1])
    h2_tot = float(h2_fit2.h2_reps[-1, -1])
    if np.isfinite(h1_tot) and np.isfinite(h2_tot) and h1_tot > 0.0 and h2_tot > 0.0:
        rg_total_se_ext = float(np.abs(np.sum(dc_gamma)) * c_se / np.sqrt(h1_tot * h2_tot))
    else:
        rg_total_se_ext = np.nan

    return gamma_se_ext, rg_se_ext, gamma_total_se_ext, rg_total_se_ext
