from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ldsc_h2 import _solve_weighted_design


@dataclass(frozen=True)
class LDSCCovIRWLSFit:
    gamma: np.ndarray
    lhs: np.ndarray
    rhs: np.ndarray
    weights: np.ndarray
    gamma_path: np.ndarray
    n_iter: int
    converged: bool
    final_delta: float
    initialization: str
    rank: int
    condition_number: float


def _require_matching_rg_plugin_axis(
    prepared,
    plugin,
    *,
    label: str,
    require_active_mask: bool = True,
) -> None:
    """Reject h2/intercept plug-ins built on a different SNP/delete design."""
    plugin_prepared = getattr(plugin, "prepared", None)
    if plugin_prepared is None:
        plugin_trace = getattr(plugin, "trace_view", None)
        plugin_jackknife = getattr(plugin, "jackknife", None)
        plugin_active = getattr(plugin, "active_mask", None)
    else:
        plugin_trace = getattr(plugin_prepared, "trace_view", None)
        plugin_jackknife = getattr(plugin_prepared, "jackknife", None)
        plugin_active = getattr(plugin_prepared, "active_mask", None)
    if plugin_trace is None or plugin_jackknife is None or plugin_active is None:
        raise ValueError(
            f"{label} does not expose the Trace SNP axis, active mask, and "
            "jackknife design needed to pair LDSC delete refits."
        )

    ref_trace = prepared.trace_view
    ref_jk = prepared.jackknife
    if not np.array_equal(
        np.asarray(plugin_trace.snps), np.asarray(ref_trace.snps)
    ):
        raise ValueError(f"{label} was fitted on a different Trace SNP axis/order.")
    plugin_active = np.asarray(plugin_active, dtype=bool)
    prepared_active = np.asarray(prepared.active_mask, dtype=bool)
    if plugin_active.shape != prepared_active.shape:
        raise ValueError(f"{label} active SNP mask has the wrong shape.")
    if require_active_mask and not np.array_equal(plugin_active, prepared_active):
        raise ValueError(f"{label} was fitted with a different active SNP mask.")
    if (
        int(plugin_jackknife.nsnps) != int(ref_jk.nsnps)
        or int(plugin_jackknife.nrep) != int(ref_jk.nrep)
        or int(plugin_jackknife.nunit) != int(ref_jk.nunit)
        or not np.array_equal(
            np.asarray(plugin_jackknife.unit_id), np.asarray(ref_jk.unit_id)
        )
        or not np.array_equal(
            np.asarray(plugin_jackknife.D), np.asarray(ref_jk.D)
        )
    ):
        raise ValueError(
            f"{label} was fitted with a different jackknife deletion design/order."
        )


def _cov_ldsc_weights(
    ref_ld_total,
    weight_ld,
    *,
    n1_scale: float,
    n2_scale: float,
    m_total: float,
    h1_total: float,
    h2_total: float,
    gamma_total: float,
    intercept: float,
    ld_floor: float = 1.0,
    variance_floor: float = 1e-12,
) -> np.ndarray:
    """Return score-scale bivariate LDSC inverse-variance weights.

    If ``a_j = E[z1*_j^2]``, ``b_j = E[z2*_j^2]``, and
    ``c_j = E[z1*_j z2*_j]``, the Gaussian working variance of the cross
    product is ``a_j b_j + c_j^2``.  The scalar weight LD supplies the usual
    marker-overcounting factor.  Flooring is restricted to the weight model;
    it never changes the fitted LD-score design.
    """
    ref_ld_total = np.asarray(ref_ld_total, dtype=np.float64).reshape(-1)
    weight_ld = np.asarray(weight_ld, dtype=np.float64).reshape(-1)
    if ref_ld_total.shape != weight_ld.shape:
        raise ValueError("Reference LD and weight LD must share one SNP axis.")
    if not (np.isfinite(ref_ld_total).all() and np.isfinite(weight_ld).all()):
        raise ValueError("Covariance LDSC weight inputs contain non-finite values.")

    n1_scale = float(n1_scale)
    n2_scale = float(n2_scale)
    m_total = float(m_total)
    intercept = float(intercept)
    if not (
        np.isfinite(n1_scale)
        and n1_scale > 0.0
        and np.isfinite(n2_scale)
        and n2_scale > 0.0
        and np.isfinite(m_total)
        and m_total > 0.0
        and np.isfinite(intercept)
    ):
        raise ValueError("Invalid sample-size, reference-mass, or intercept scale.")

    ld_floor = float(ld_floor)
    variance_floor = float(variance_floor)
    if not (np.isfinite(ld_floor) and ld_floor > 0.0):
        raise ValueError("ld_floor must be positive and finite.")
    if not (np.isfinite(variance_floor) and variance_floor > 0.0):
        raise ValueError("variance_floor must be positive and finite.")

    h1 = float(np.clip(h1_total, 0.0, 1.0))
    h2 = float(np.clip(h2_total, 0.0, 1.0))
    # This is a genetic covariance plug-in, not rg.  The [-1, 1] bound is the
    # established LDSC weight stabilization; it affects weights only.
    gamma = float(np.clip(gamma_total, -1.0, 1.0))
    ld_eff = np.maximum(ref_ld_total, ld_floor)
    overcount = np.maximum(weight_ld, ld_floor)

    a = 1.0 + (n1_scale * h1 / m_total) * ld_eff
    b = 1.0 + (n2_scale * h2 / m_total) * ld_eff
    c = intercept + (np.sqrt(n1_scale * n2_scale) * gamma / m_total) * ld_eff
    variance = np.maximum(a * b + c * c, variance_floor)
    weights = 1.0 / (overcount * variance)
    if not (np.isfinite(weights).all() and np.all(weights > 0.0)):
        raise ValueError("Covariance LDSC IRWLS produced invalid weights.")
    return weights


def fit_constrained_cov_ldsc_irwls(
    design,
    response,
    ref_ld_total,
    weight_ld,
    *,
    n1_scale: float,
    n2_scale: float,
    m_annot,
    h1_total: float,
    h2_total: float,
    intercept: float,
    keep=None,
    irwls_iters: int = 3,
    irwls_tol: float = 0.0,
    initial_gamma=None,
    ld_floor: float = 1.0,
    variance_floor: float = 1e-12,
) -> LDSCCovIRWLSFit:
    """Fit fixed-intercept score-scale covariance LDSC by closed-form IRWLS.

    The response is ``z1* z2* - c`` and column ``k`` of the design is
    ``sqrt(n1* n2*) L_k / M_k``.  ``c`` is supplied by SUMMIT's separate
    nuisance-intercept step and is never estimated in this solve.
    """
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    ref_ld_total = np.asarray(ref_ld_total, dtype=np.float64).reshape(-1)
    weight_ld = np.asarray(weight_ld, dtype=np.float64).reshape(-1)
    m_annot = np.asarray(m_annot, dtype=np.float64).reshape(-1)

    if design.ndim != 2:
        raise ValueError(f"design must be 2D; got {design.shape}.")
    m, k = design.shape
    if (
        response.shape != (m,)
        or ref_ld_total.shape != (m,)
        or weight_ld.shape != (m,)
    ):
        raise ValueError(
            "Covariance LDSC design, response, reference LD, and weight LD "
            "must share one SNP axis."
        )
    if m_annot.shape != (k,):
        raise ValueError(f"m_annot must have shape ({k},); got {m_annot.shape}.")
    if not (
        np.isfinite(design).all()
        and np.isfinite(ref_ld_total).all()
        and np.isfinite(weight_ld).all()
        and np.isfinite(m_annot).all()
        and np.all(m_annot > 0.0)
    ):
        raise ValueError("Covariance LDSC inputs contain invalid values.")

    if keep is None:
        keep = np.ones(m, dtype=bool)
    else:
        keep = np.asarray(keep, dtype=bool).reshape(-1)
        if keep.shape != (m,):
            raise ValueError(f"keep must have shape ({m},); got {keep.shape}.")
    if int(np.sum(keep)) < k:
        raise ValueError(
            f"Covariance LDSC fit has fewer retained SNPs ({int(np.sum(keep))}) "
            f"than coefficients ({k})."
        )
    if not np.isfinite(response[keep]).all():
        raise ValueError("The retained covariance LDSC response is non-finite.")

    n1_scale = float(n1_scale)
    n2_scale = float(n2_scale)
    m_total = float(np.sum(m_annot, dtype=np.float64))
    intercept = float(intercept)
    if not (
        np.isfinite(n1_scale)
        and n1_scale > 0.0
        and np.isfinite(n2_scale)
        and n2_scale > 0.0
        and np.isfinite(m_total)
        and m_total > 0.0
        and np.isfinite(h1_total)
        and np.isfinite(h2_total)
        and np.isfinite(intercept)
    ):
        raise ValueError("Invalid covariance LDSC plug-in scale.")

    irwls_iters = int(irwls_iters)
    irwls_tol = float(irwls_tol)
    if irwls_iters < 1:
        raise ValueError("irwls_iters must be at least 1.")
    if not (np.isfinite(irwls_tol) and irwls_tol >= 0.0):
        raise ValueError("irwls_tol must be non-negative and finite.")

    dk = np.asarray(design[keep, :], dtype=np.float64, order="C")
    qk = np.asarray(response[keep], dtype=np.float64)
    if initial_gamma is None:
        if k == 1:
            denom = float(
                np.sqrt(n1_scale * n2_scale)
                * np.sum(ref_ld_total[keep], dtype=np.float64)
            )
            numer = m_total * float(np.sum(qk, dtype=np.float64))
            if np.isfinite(numer) and np.isfinite(denom) and denom != 0.0:
                gamma_current = np.asarray([numer / denom], dtype=np.float64)
                initialization = "aggregate_cov_ldsc"
            else:
                gamma_current, _lhs, _rhs, _rank, _cond = _solve_weighted_design(
                    dk, qk
                )
                initialization = "unweighted_wls"
        else:
            gamma_current, _lhs, _rhs, _rank, _cond = _solve_weighted_design(
                dk, qk
            )
            initialization = "unweighted_wls"
    else:
        gamma_current = np.asarray(initial_gamma, dtype=np.float64).reshape(-1)
        if gamma_current.shape != (k,) or not np.isfinite(gamma_current).all():
            raise ValueError(f"initial_gamma must be finite with shape ({k},).")
        initialization = "provided"

    path = [gamma_current.copy()]
    final_lhs = None
    final_rhs = None
    final_weights = None
    final_rank = 0
    final_condition = np.nan
    final_delta = np.nan
    converged = False
    n_iter = 0

    for iteration in range(irwls_iters):
        weights = _cov_ldsc_weights(
            ref_ld_total[keep],
            weight_ld[keep],
            n1_scale=n1_scale,
            n2_scale=n2_scale,
            m_total=m_total,
            h1_total=h1_total,
            h2_total=h2_total,
            gamma_total=float(np.sum(gamma_current, dtype=np.float64)),
            intercept=intercept,
            ld_floor=ld_floor,
            variance_floor=variance_floor,
        )
        gamma_next, lhs, rhs, rank, condition = _solve_weighted_design(
            dk, qk, weights
        )
        final_lhs = lhs
        final_rhs = rhs
        final_weights = weights
        final_rank = rank
        final_condition = condition
        n_iter = iteration + 1
        final_delta = float(np.max(np.abs(gamma_next - gamma_current)))
        gamma_current = gamma_next
        path.append(gamma_current.copy())
        if irwls_tol > 0.0:
            scale = max(1.0, float(np.max(np.abs(gamma_current))))
            if final_delta <= irwls_tol * scale:
                converged = True
                break

    return LDSCCovIRWLSFit(
        gamma=np.asarray(gamma_current, dtype=np.float64),
        lhs=np.asarray(final_lhs, dtype=np.float64),
        rhs=np.asarray(final_rhs, dtype=np.float64),
        weights=np.asarray(final_weights, dtype=np.float64),
        gamma_path=np.asarray(path, dtype=np.float64),
        n_iter=int(n_iter),
        converged=bool(converged),
        final_delta=float(final_delta),
        initialization=initialization,
        rank=int(final_rank),
        condition_number=float(final_condition),
    )


def fit_rg_ldsc(
    prepared,
    h2_fit1,
    h2_fit2,
    intercept_fit,
    *,
    m_annot,
    jack_mode: str,
    nan_policy: str,
    irwls_iters: int,
    irwls_tol: float,
):
    """Fit covariance and rg while retaining the established ``RGFit`` API."""
    # Lazy import avoids a module cycle with rgcore.fit_rg().
    from .rgcore import RGFit, _component_rg

    p = prepared
    m = int(p.trace_view.nsnps)
    k = int(p.trace_view.nbins)
    R = int(p.jackknife.nrep)
    L = np.asarray(p.trace_view.ldscores, dtype=np.float64)
    y = np.asarray(p.y, dtype=np.float64)
    active = np.asarray(p.active_mask, dtype=bool)
    if L.shape != (m, k) or y.shape != (m,) or active.shape != (m,):
        raise ValueError(
            "LDSC rg requires the regular per-SNP RG preparation; the fast "
            "sufficient-statistic manifest path is not supported."
        )

    m_annot = np.asarray(m_annot, dtype=np.float64).reshape(-1)
    if m_annot.shape != (k,) or not (
        np.isfinite(m_annot).all() and np.all(m_annot > 0.0)
    ):
        raise ValueError("LDSC reference masses do not match the rg LD-score columns.")
    if str(getattr(h2_fit1, "weight_mode", "he")) != "ldsc" or str(
        getattr(h2_fit2, "weight_mode", "he")
    ) != "ldsc":
        raise ValueError(
            "LDSC rg requires LDSC h2 fits so every covariance delete refit uses "
            "the matching h1/h2 plug-ins."
        )
    _require_matching_rg_plugin_axis(p, h2_fit1, label="trait-1 h2 fit")
    _require_matching_rg_plugin_axis(p, h2_fit2, label="trait-2 h2 fit")
    # SUMMIT's intercept fit may intentionally use a stricter, intercept-only
    # SNP filter.  Its c_r values are nevertheless paired to covariance refits
    # by the shared Trace axis and delete-design ordering.
    _require_matching_rg_plugin_axis(
        p,
        intercept_fit,
        label="rg intercept fit",
        require_active_mask=False,
    )

    c_reps = np.asarray(intercept_fit.c_reps, dtype=np.float64)
    h1_total_reps = np.asarray(h2_fit1.h2_reps[:, -1], dtype=np.float64)
    h2_total_reps = np.asarray(h2_fit2.h2_reps[:, -1], dtype=np.float64)
    expected = (R + 1,)
    if (
        c_reps.shape != expected
        or h1_total_reps.shape != expected
        or h2_total_reps.shape != expected
    ):
        raise ValueError("LDSC rg plug-in replicate shapes do not match its jackknife.")

    sqrt_n1n2 = float(np.sqrt(float(p.n1_scale) * float(p.n2_scale)))
    design = (sqrt_n1n2 * L) / m_annot[None, :]
    ref_ld_total = np.sum(L, axis=1, dtype=np.float64)
    reg_w = getattr(p.trace_view, "ldscores_reg_w", None)
    if reg_w is None:
        weight_ld = ref_ld_total.copy()
        weight_ld_source = "total_primary_ld"
    else:
        reg_w = np.asarray(reg_w, dtype=np.float64)
        if reg_w.shape != (m, 1):
            raise ValueError(
                f"LDSC weight-LD input must have shape ({m}, 1); got {reg_w.shape}."
            )
        weight_ld = reg_w[:, 0]
        weight_ld_source = "explicit_ldscores_w"

    valid = (
        active
        & np.isfinite(y)
        & np.isfinite(L).all(axis=1)
        & np.isfinite(weight_ld)
    )
    if int(np.sum(valid)) < k:
        raise ValueError("Too few finite SNPs remain for LDSC-weighted rg.")

    unit_id = np.asarray(p.jackknife.unit_id, dtype=np.int64)
    delete = np.asarray(p.jackknife.D, dtype=np.float64)
    if unit_id.shape != (m,) or delete.shape != (R, int(p.jackknife.nunit)):
        raise ValueError("LDSC rg jackknife design does not match the SNP axis.")

    gamma_reps = np.full((R + 1, k), np.nan, dtype=np.float64)
    failures = []
    full_fit = None
    for r in [R, *range(R)]:
        keep = valid if r == R else (valid & (delete[r, unit_id] < 0.5))
        try:
            fit = fit_constrained_cov_ldsc_irwls(
                design,
                y - float(c_reps[r]),
                ref_ld_total,
                weight_ld,
                n1_scale=float(p.n1_scale),
                n2_scale=float(p.n2_scale),
                m_annot=m_annot,
                h1_total=float(h1_total_reps[r]),
                h2_total=float(h2_total_reps[r]),
                intercept=float(c_reps[r]),
                keep=keep,
                irwls_iters=irwls_iters,
                irwls_tol=irwls_tol,
            )
            gamma_reps[r] = fit.gamma
            if r == R:
                full_fit = fit
        except Exception as exc:
            failures.append(
                {
                    "replicate": "full" if r == R else int(r),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            if r == R:
                break

    if full_fit is None:
        raise RuntimeError(
            f"Full-sample LDSC covariance fit failed: {failures[-1]['error']}"
        )
    if failures:
        first = failures[0]
        raise RuntimeError(
            "One or more exact covariance-LDSC jackknife refits failed; refusing "
            "to report a partial SE. First failure at replicate "
            f"{first['replicate']}: {first['error']}"
        )

    def _summarize(values):
        est, se = p.jackknife.summarize(
            values,
            unit_sizes=p.unit_sizes,
            axis=0,
            center=jack_mode,
            nan_policy=nan_policy,
        )
        return np.asarray(est, dtype=np.float64), np.asarray(se, dtype=np.float64)

    est, se = _summarize(gamma_reps)
    gamma = np.column_stack([est, se])

    h1_components = np.asarray(h2_fit1.sigma_reps[:, :k], dtype=np.float64)
    h2_components = np.asarray(h2_fit2.sigma_reps[:, :k], dtype=np.float64)
    if h1_components.shape != (R + 1, k) or h2_components.shape != (R + 1, k):
        raise ValueError("LDSC h2 component replicates do not match the rg model.")
    rg_reps = _component_rg(gamma_reps, h1_components, h2_components)
    est, se = _summarize(rg_reps)
    rg = np.column_stack([est, se])

    gamma_total_reps = np.sum(gamma_reps, axis=1, dtype=np.float64)
    est, se = _summarize(gamma_total_reps)
    gamma_total = np.asarray([float(est), float(se)], dtype=np.float64)
    rg_total_reps = _component_rg(
        gamma_total_reps, h1_total_reps, h2_total_reps
    )
    est, se = _summarize(rg_total_reps)
    rg_total = np.asarray([float(est), float(se)], dtype=np.float64)

    full = full_fit
    intercept_info = (
        intercept_fit.info if isinstance(intercept_fit.info, dict) else {}
    )
    weight_info = {
        "estimator": "constrained_cov_ldsc_irwls",
        "response": "score_z1_times_z2_minus_summit_intercept",
        "n1_scale": float(p.n1_scale),
        "n2_scale": float(p.n2_scale),
        "m_annot": m_annot.copy(),
        "m_total_for_weights": float(np.sum(m_annot, dtype=np.float64)),
        "weight_ld_source": weight_ld_source,
        "intercept_source": str(intercept_info.get("source", "estimated")),
        "intercept_fixed": bool(intercept_info.get("fixed", False)),
        "intercept_replicates": (
            "fixed_across_replicates"
            if bool(intercept_info.get("fixed", False))
            else "summit_delete_refits"
        ),
        "h2_plugins": "matching_ldsc_delete_refits",
        "irwls_iters": int(full.n_iter),
        "irwls_max_iters": int(irwls_iters),
        "irwls_tol": float(irwls_tol),
        "irwls_converged": bool(full.converged),
        "irwls_final_delta": float(full.final_delta),
        "initialization": full.initialization,
        "weighted_design_rank": int(full.rank),
        "weighted_design_condition_number": float(full.condition_number),
        "gamma_path": full.gamma_path.copy(),
        "weight_min": float(np.min(full.weights)),
        "weight_max": float(np.max(full.weights)),
        "n_failed_replicates": int(len(failures)),
        "replicate_failures": failures[:20],
        "jackknife_weights": "refit_irwls_per_replicate",
    }

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
        rg_se_method="jackknife",
        kmoment_info=None,
        weight_mode="ldsc",
        weight_info=weight_info,
    )
