from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import utils
import json
from moments import build_rg_summary_moment

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
        with np.errstate(divide="ignore", invalid="ignore"):
            rg_tot = gamma_tot / np.sqrt(h1_tot * h2_tot)
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
        from h2core import H2ResultWriter

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
    with np.errstate(divide="ignore", invalid="ignore"):
        rg_reps = gamma_reps / np.sqrt(v1 * v2)
    rg_reps[~np.isfinite(rg_reps)] = np.nan

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

    with np.errstate(divide="ignore", invalid="ignore"):
        rg_tot_reps = gamma_tot_reps / np.sqrt(h2_tot1 * h2_tot2)
    rg_tot_reps[~np.isfinite(rg_tot_reps)] = np.nan
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


def _select_intercept_regression_ld(trace_view, *, collapse_reg_ld=False, log=None):
    Lreg = trace_view.ldscores_reg
    if Lreg is None:
        Lmain = np.asarray(trace_view.ldscores, dtype=np.float64, order="C")
        if Lmain.ndim == 1:
            return Lmain.reshape(-1, 1), "main"
        if Lmain.ndim != 2:
            raise RuntimeError("Primary ldscores must be 1D or 2D.")
        if Lmain.shape[1] == 1:
            return Lmain, "main"
        if not collapse_reg_ld:
            raise RuntimeError(
                "Intercept regression requires 1D LD by default. Provide --ldscores-reg "
                "or pass --collapse-reg-ld to collapse the primary LD scores."
            )
        if log is not None:
            log._log(
                f"[rg] Collapsing {Lmain.shape[1]}-column primary ldscores to total LD for intercept regression."
            )
        return np.sum(Lmain, axis=1, dtype=np.float64, keepdims=True), "main-collapsed"

    Lreg = np.asarray(Lreg, dtype=np.float64, order="C")
    if Lreg.ndim == 1:
        return Lreg.reshape(-1, 1), "reg"
    if Lreg.ndim != 2:
        raise RuntimeError("ldscores_reg must be 1D or 2D.")
    if Lreg.shape[1] == 1:
        return Lreg, "reg"
    if not collapse_reg_ld:
        raise RuntimeError(
            f"ldscores_reg has {Lreg.shape[1]} columns; provide native 1D regression LD "
            "or pass --collapse-reg-ld."
        )
    if log is not None:
        log._log(
            f"[rg] Collapsing {Lreg.shape[1]}-column ldscores_reg to total LD for intercept regression."
        )
    return np.sum(Lreg, axis=1, dtype=np.float64, keepdims=True), "reg-collapsed"


def _make_intercept_keep_mask(z1, z2, x, *, nsamp_max, threshold=None, chisq_mode="either"):
    z1 = np.asarray(z1, dtype=np.float64).ravel()
    z2 = np.asarray(z2, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if not (z1.size == z2.size == x.size):
        raise ValueError("z1/z2/x length mismatch in intercept keep-mask construction.")

    keep = np.isfinite(z1) & np.isfinite(z2) & np.isfinite(x) & (x > 0.0)
    thr, thr_mode = utils._resolve_chisq_threshold(float(nsamp_max), threshold)
    mode = str(chisq_mode).strip().lower()
    thr_used = None

    if thr is not None:
        thr = float(thr)
        if np.isfinite(thr) and thr > 0.0:
            c1 = z1 * z1
            c2 = z2 * z2
            if mode in ("either", "max"):
                keep &= (c1 <= thr) & (c2 <= thr)
            elif mode == "both":
                keep &= ~((c1 > thr) & (c2 > thr))
            else:
                raise ValueError("chisq_mode must be one of {'either','both','max'}")
            thr_used = thr

    info = {
        "threshold": thr_used,
        "threshold_mode": thr_mode,
        "chisq_mode": mode,
        "n_total": int(z1.size),
        "n_kept": int(np.sum(keep)),
        "n_removed": int(np.sum(~keep)),
        "n_removed_nonfinite_or_nonpositive_ld": int(np.sum((~np.isfinite(x)) | (x <= 0.0))),
    }
    return keep, info


def _build_simple_intercept_weights(x, keep):
    x = np.asarray(x, dtype=np.float64).ravel()
    keep = np.asarray(keep, dtype=bool).ravel()
    if x.size != keep.size:
        raise ValueError("x/keep length mismatch.")
    if np.any(keep & ((~np.isfinite(x)) | (x <= 0.0))):
        raise ValueError("Total LD must be finite and >0 for kept SNPs in the intercept regression.")
    w = np.zeros(x.size, dtype=np.float64)
    w[keep] = 1.0 / x[keep]
    if not np.isfinite(w).all() or np.sum(w) <= 0.0:
        raise ValueError("Invalid intercept regression weights.")
    return w


def _solve_constrained_intercept_scalar_from_sums(
    m_fit,
    l1,
    t1,
    W,
    Sx,
    Sxx,
    Sy,
    Sxy,
    denom_floor=0.0,
):
    m_fit = np.asarray(m_fit, dtype=np.float64)
    l1 = np.asarray(l1, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    Sx = np.asarray(Sx, dtype=np.float64)
    Sxx = np.asarray(Sxx, dtype=np.float64)
    Sy = np.asarray(Sy, dtype=np.float64)
    Sxy = np.asarray(Sxy, dtype=np.float64)

    m_fit, l1, t1, W, Sx, Sxx, Sy, Sxy = np.broadcast_arrays(m_fit, l1, t1, W, Sx, Sxx, Sy, Sxy)
    c = np.full(m_fit.shape, np.nan, dtype=np.float64)
    b = np.full(m_fit.shape, np.nan, dtype=np.float64)

    base_good = (
        np.isfinite(m_fit) & np.isfinite(l1) & np.isfinite(t1) & np.isfinite(W) &
        np.isfinite(Sx) & np.isfinite(Sxx) & np.isfinite(Sy) & np.isfinite(Sxy) &
        (m_fit > 0.0) & (l1 > 0.0) & (W > 0.0)
    )
    if not np.any(base_good):
        return c, b, base_good

    alpha = np.full(m_fit.shape, np.nan, dtype=np.float64)
    beta = np.full(m_fit.shape, np.nan, dtype=np.float64)
    alpha[base_good] = t1[base_good] / l1[base_good]
    beta[base_good] = m_fit[base_good] / l1[base_good]

    num = Sy - alpha * Sx - beta * Sxy + (alpha * beta) * Sxx
    den = W - 2.0 * beta * Sx + (beta * beta) * Sxx

    if denom_floor is not None:
        denom_floor = float(denom_floor)
        if np.isfinite(denom_floor) and denom_floor > 0.0:
            den = np.where(np.isfinite(den) & (den > 0.0), np.maximum(den, denom_floor), den)

    good = base_good & np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    if not np.any(good):
        return c, b, good

    c[good] = num[good] / den[good]
    b[good] = (t1[good] - m_fit[good] * c[good]) / l1[good]

    good = good & np.isfinite(c) & np.isfinite(b)
    c[~good] = np.nan
    b[~good] = np.nan
    return c, b, good


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

    h1 = float(np.clip(h1, 0.0, 1.0))
    h2 = float(np.clip(h2, 0.0, 1.0))
    rho_g = float(np.clip(rho_g, -1.0, 1.0))
    intercept_gencov = 0.0 if intercept_gencov is None else float(intercept_gencov)
    intercept_hsq1 = 1.0 if intercept_hsq1 is None else float(intercept_hsq1)
    intercept_hsq2 = 1.0 if intercept_hsq2 is None else float(intercept_hsq2)

    int_floor = float(intercept_hsq_floor)
    if not (np.isfinite(int_floor) and int_floor > 0.0):
        int_floor = 1e-8
    intercept_hsq1 = max(intercept_hsq1, int_floor)
    intercept_hsq2 = max(intercept_hsq2, int_floor)

    ld_eff = np.fmax(np.where(np.isfinite(ld), ld, 1.0), 1.0)
    w_ld_eff = np.fmax(np.where(np.isfinite(w_ld), w_ld, 1.0), 1.0)

    a = (n1 * (h1 * ld_eff) / m_tot) + intercept_hsq1
    b = (n2 * (h2 * ld_eff) / m_tot) + intercept_hsq2
    c = (np.sqrt(n1 * n2) * (rho_g * ld_eff) / m_tot) + intercept_gencov
    den = a * b + c * c

    eps = float(weight_floor)
    if not (np.isfinite(eps) and eps > 0.0):
        eps = 1e-12
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
    log=None,
    jack_mode: str = "mean",
    nan_policy: str = "omit",
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

    L1, ld_source = _select_intercept_regression_ld(
        trace_view,
        collapse_reg_ld=collapse_reg_ld,
        log=log,
    )
    if L1.shape[1] != 1:
        raise RuntimeError("Intercept regression currently requires 1D LD after any collapsing.")

    x = np.asarray(L1[:, 0], dtype=np.float64)
    z1 = np.asarray(matched1.z, dtype=np.float64)
    z2 = np.asarray(matched2.z, dtype=np.float64)
    y = np.asarray(summary_y, dtype=np.float64)
    if y.ndim != 1 or y.size != x.size:
        raise ValueError(f"summary_y must have shape ({x.size},), got {y.shape}")

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
        x,
        nsamp_max=nsamp_max,
        threshold=intercept_chisq_threshold,
        chisq_mode=chisq_mode,
    )
    keep &= np.isfinite(y)

    info["ld_source"] = ld_source
    info["weight_mode"] = mode
    info["summary_y_mode"] = None if summary_y_info is None else str(summary_y_info.get("mode", "unknown"))
    info["n_removed_nonfinite_summary_y"] = int(np.sum(~np.isfinite(y)))
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
            f"mode={info['chisq_mode']}, removed={info['n_removed']} SNPs, kept={info['n_kept']}."
        )

    U = jackknife.nunit
    starts = np.asarray(jackknife.starts, dtype=np.int64)
    ends = np.asarray(jackknife.ends, dtype=np.int64)
    D = np.asarray(jackknife.D, dtype=np.float64, order="C")

    keep_f = keep.astype(np.float64, copy=False)
    x_fit = np.where(keep, x, 0.0)
    y_fit = np.where(keep, y, 0.0)

    m_u = np.zeros(U, dtype=np.float64)
    l1_u = np.zeros(U, dtype=np.float64)
    t1_u = np.zeros(U, dtype=np.float64)
    for u in range(U):
        s = int(starts[u])
        e = int(ends[u])
        if e <= s:
            continue
        m_u[u] = float(np.sum(keep_f[s:e]))
        l1_u[u] = float(np.sum(x_fit[s:e]))
        t1_u[u] = float(np.sum(y_fit[s:e]))

    m_fit_full = float(np.sum(m_u))
    l1_full = float(np.sum(l1_u))
    t1_full = float(np.sum(t1_u))
    if not (m_fit_full > 1.0 and np.isfinite(l1_full) and l1_full > 0.0 and np.isfinite(t1_full)):
        raise RuntimeError(
            "Intercept regression became ill-posed after filtering: "
            f"M_fit={m_fit_full}, L1={l1_full}, T1={t1_full}."
        )

    def _weighted_scalar_summaries(w: np.ndarray):
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != x.size:
            raise ValueError("Intercept weights length mismatch with LD axis.")
        wx = w * x
        W = float(np.sum(w))
        Sx = float(np.sum(wx))
        Sxx = float(np.sum(wx * x))
        Sy = float(np.sum(w * y))
        Sxy = float(np.sum(wx * y))
        return W, Sx, Sxx, Sy, Sxy

    def _weighted_unit_summaries(w: np.ndarray):
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != x.size:
            raise ValueError("Intercept weights length mismatch with LD axis.")
        wx = w * x
        wxx = wx * x
        wy = w * y
        wxy = wx * y

        W_u = np.zeros(U, dtype=np.float64)
        Sx_u = np.zeros(U, dtype=np.float64)
        Sxx_u = np.zeros(U, dtype=np.float64)
        Sy_u = np.zeros(U, dtype=np.float64)
        Sxy_u = np.zeros(U, dtype=np.float64)
        for u in range(U):
            s = int(starts[u])
            e = int(ends[u])
            if e <= s:
                continue
            W_u[u] = float(np.sum(w[s:e]))
            Sx_u[u] = float(np.sum(wx[s:e]))
            Sxx_u[u] = float(np.sum(wxx[s:e]))
            Sy_u[u] = float(np.sum(wy[s:e]))
            Sxy_u[u] = float(np.sum(wxy[s:e]))
        return W_u, Sx_u, Sxx_u, Sy_u, Sxy_u

    w_score = _build_simple_intercept_weights(x, keep)
    W0, Sx0, Sxx0, Sy0, Sxy0 = _weighted_scalar_summaries(w_score)

    c0, b0, ok0 = _solve_constrained_intercept_scalar_from_sums(
        np.array([m_fit_full], dtype=np.float64),
        np.array([l1_full], dtype=np.float64),
        np.array([t1_full], dtype=np.float64),
        np.array([W0], dtype=np.float64),
        np.array([Sx0], dtype=np.float64),
        np.array([Sxx0], dtype=np.float64),
        np.array([Sy0], dtype=np.float64),
        np.array([Sxy0], dtype=np.float64),
        denom_floor=0.0,
    )
    if not bool(ok0[0]):
        raise RuntimeError("Failed to initialize constrained intercept fit.")

    c_cur = float(c0[0])
    b_cur = float(b0[0])
    w_final = w_score

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

        sqrt_n1n2_scalar = float(np.sqrt(n1_scalar * n2_scalar))
        m_tot_weight = float(trace_view.nsnps)

        h1 = float(h2_fit1.h2[-1, 0]) if np.isfinite(h2_fit1.h2[-1, 0]) else 0.0
        h2 = float(h2_fit2.h2[-1, 0]) if np.isfinite(h2_fit2.h2[-1, 0]) else 0.0
        h1 = float(np.clip(h1, 0.0, 1.0))
        h2 = float(np.clip(h2, 0.0, 1.0))

        for _ in range(n_iter):
            rho_cur = float((m_tot_weight / sqrt_n1n2_scalar) * b_cur)
            w_cur = _ldsc_gencov_weights_1d(
                ld=x,
                w_ld=x,
                n1=np.full(x.size, n1_scalar, dtype=np.float64),
                n2=np.full(x.size, n2_scalar, dtype=np.float64),
                m_tot=m_tot_weight,
                h1=h1,
                h2=h2,
                rho_g=rho_cur,
                intercept_gencov=c_cur,
                intercept_hsq1=intercept_hsq1,
                intercept_hsq2=intercept_hsq2,
                intercept_hsq_floor=intercept_hsq_floor,
                weight_floor=intercept_weight_floor,
            )
            w_cur = np.where(keep, w_cur, 0.0)

            W, Sx, Sxx, Sy, Sxy = _weighted_scalar_summaries(w_cur)
            c_new, b_new, ok = _solve_constrained_intercept_scalar_from_sums(
                np.array([m_fit_full], dtype=np.float64),
                np.array([l1_full], dtype=np.float64),
                np.array([t1_full], dtype=np.float64),
                np.array([W], dtype=np.float64),
                np.array([Sx], dtype=np.float64),
                np.array([Sxx], dtype=np.float64),
                np.array([Sy], dtype=np.float64),
                np.array([Sxy], dtype=np.float64),
                denom_floor=0.0,
            )
            if not bool(ok[0]):
                raise RuntimeError("LDSC-IRWLS intercept update failed on the full sample.")

            c_cur = float(c_new[0])
            b_cur = float(b_new[0])
            w_final = w_cur

        info["h1_plugin"] = h1
        info["h2_plugin"] = h2
        info["irwls_iters"] = n_iter
    else:
        info["irwls_iters"] = 0

    W_u, Sx_u, Sxx_u, Sy_u, Sxy_u = _weighted_unit_summaries(w_final)
    W_full = float(np.sum(W_u))
    Sx_full = float(np.sum(Sx_u))
    Sxx_full = float(np.sum(Sxx_u))
    Sy_full = float(np.sum(Sy_u))
    Sxy_full = float(np.sum(Sxy_u))

    alpha_full = t1_full / l1_full
    beta_full = m_fit_full / l1_full
    den_full = W_full - 2.0 * beta_full * Sx_full + (beta_full * beta_full) * Sxx_full
    floor_rel = float(denom_floor_rel)
    if not (np.isfinite(floor_rel) and floor_rel >= 0.0):
        floor_rel = 1e-12
    denom_floor = max(floor_rel * max(float(den_full), 1.0), 0.0)

    del_m = D @ m_u
    del_l1 = D @ l1_u
    del_t1 = D @ t1_u
    del_W = D @ W_u
    del_Sx = D @ Sx_u
    del_Sxx = D @ Sxx_u
    del_Sy = D @ Sy_u
    del_Sxy = D @ Sxy_u

    m_rep = m_fit_full - del_m
    l1_rep = l1_full - del_l1
    t1_rep = t1_full - del_t1
    W_rep = W_full - del_W
    Sx_rep = Sx_full - del_Sx
    Sxx_rep = Sxx_full - del_Sxx
    Sy_rep = Sy_full - del_Sy
    Sxy_rep = Sxy_full - del_Sxy

    c_rep, _, _ = _solve_constrained_intercept_scalar_from_sums(
        m_rep,
        l1_rep,
        t1_rep,
        W_rep,
        Sx_rep,
        Sxx_rep,
        Sy_rep,
        Sxy_rep,
        denom_floor=denom_floor,
    )
    c_full, _, good_full = _solve_constrained_intercept_scalar_from_sums(
        np.array([m_fit_full], dtype=np.float64),
        np.array([l1_full], dtype=np.float64),
        np.array([t1_full], dtype=np.float64),
        np.array([W_full], dtype=np.float64),
        np.array([Sx_full], dtype=np.float64),
        np.array([Sxx_full], dtype=np.float64),
        np.array([Sy_full], dtype=np.float64),
        np.array([Sxy_full], dtype=np.float64),
        denom_floor=denom_floor,
    )
    if not bool(good_full[0]):
        raise RuntimeError("Final constrained full-sample intercept solve failed.")

    c_reps = np.empty(jackknife.nrep + 1, dtype=np.float64)
    c_reps[:jackknife.nrep] = c_rep
    c_reps[jackknife.nrep] = float(c_full[0])

    unit_sizes = jackknife.unit_sizes(active_mask=keep, dtype=np.float64)
    est, se = jackknife.summarize(
        c_reps,
        unit_sizes=unit_sizes,
        axis=0,
        center=jack_mode,
        nan_policy=nan_policy,
    )
    c = np.array([float(est), float(se)], dtype=np.float64)

    if log is not None:
        n_bad = int(np.sum(~np.isfinite(c_rep)))
        if mode == "ldsc":
            log._log(
                f"[rg:c] constrained LDSC-IRWLS: h1={info['h1_plugin']:.6g}, h2={info['h2_plugin']:.6g}, "
                f"final_c={c[0]:.6g}, bad_reps={n_bad}/{jackknife.nrep}, iters={info['irwls_iters']}"
            )
        else:
            log._log(
                f"[rg:c] constrained SCORE-weight intercept: "
                f"final_c={c[0]:.6g}, bad_reps={n_bad}/{jackknife.nrep}"
            )

    return InterceptFit(
        trace_view=trace_view,
        matched1=matched1,
        matched2=matched2,
        jackknife=jackknife,
        active_mask=keep,
        unit_sizes=unit_sizes,
        ld=x,
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