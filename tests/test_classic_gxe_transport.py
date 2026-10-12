"""Population transport of the classic single-exposure G + GxE + NxE model.

Each cohort centers and per-SD scales its own environment, projects out the
intercept, covariates and environment, and rescales every projected feature
to squared norm ``r``.  The dense oracle below builds those kernels exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from summit.context.oracle import (
    approximate_study_same_person,
    context_fourth_moments,
    context_second_moments,
    dense_genetic_kernels,
    distinct_pair_exposure,
    rank_revealing_projector,
)
from summit.context.spec import ContextComponentIndex, ContextPairIndex
from summit.inference.gxe import (
    GxENormalEquations,
    classic_context_moments,
    classic_different_individual_correction,
    classic_ld_scalars,
    classic_study_same_individual_products,
    solve_normal_equations,
    transfer_reference_normal_equations,
)
from summit.ldscore.gwe_ldscore import classic_environment_moments


def _cohort(g_raw, e_raw, covariates, rows, *, convention="standardized_projected"):
    """Exact classic kernels for one cohort, as the reference/scoring code builds them."""
    g = g_raw[rows]
    g = (g - g.mean(axis=0)) / g.std(axis=0, ddof=1)
    e = e_raw[rows] - e_raw[rows].mean()
    e = e / np.sqrt((e @ e) / (len(e) - 1))
    c = covariates[rows] - covariates[rows].mean(axis=0)
    basis, _ = np.linalg.qr(np.column_stack([c, e]))
    n = len(rows)
    projector = np.eye(n) - 1.0 / n - basis @ basis.T
    r = n - basis.shape[1] - 1
    x = projector @ g
    w = projector @ (g * e[:, None])
    if convention == "standardized_projected":
        x *= np.sqrt(r / np.sum(x * x, axis=0))
        w *= np.sqrt(r / np.sum(w * w, axis=0))
    m = g.shape[1]
    kx, kw = x @ x.T / m, w @ w.T / m
    kn = projector @ (e[:, None] ** 2 * projector)
    genetic = [kx, kw]
    gram = np.array([[np.sum(a * b) for b in genetic] for a in genetic])
    diag = np.stack([np.diag(kx), np.diag(kw)])
    same = diag @ diag.T
    kernels = [kx, kw, kn, projector]
    full = np.array([[np.sum(a * b) for b in kernels] for a in kernels])
    return dict(
        n=n, r=r, e=e, g=g, kx=kx, kw=kw, kn=kn, projector=projector, basis=basis,
        gram=gram, same=same, full=full, traces=np.array([np.trace(k) for k in kernels]),
        moments=classic_environment_moments(e, residual_rank=r),
    )


def _design(seed, n, m, mode):
    rng = np.random.default_rng(seed)
    frequencies = rng.uniform(0.1, 0.5, size=m)
    g_raw = rng.binomial(2, frequencies, size=(n, m)).astype(float)
    if mode == "age":
        e_raw = rng.gamma(2.0, 1.0, size=n)
        keep = 1.0 / (1.0 + np.exp(-(e_raw - 2.0)))
    else:
        e_raw = (rng.random(n) < 0.46).astype(float)
        keep = np.where(e_raw > 0, 0.95, 0.55)
    covariates = rng.normal(size=(n, 2))
    rows = np.flatnonzero(rng.random(n) < keep)
    return rng, g_raw, e_raw, covariates, rows


def _transported(reference, study, *, corrected):
    """Genetic 2x2 block of the transported study Gram (one LD block)."""
    n_r, n_s = reference["n"], study["n"]
    s = n_s * (n_s - 1) / (n_r * (n_r - 1))
    g_r, f_r = classic_context_moments(reference["moments"])
    g_s, f_s = classic_context_moments(study["moments"])
    if not corrected:
        return n_s / n_r * reference["same"] + s * (reference["gram"] - reference["same"])
    same = classic_study_same_individual_products(
        reference["same"], g_r, g_s, reference_n_samples=n_r, study_n_samples=n_s
    )
    scalars = classic_ld_scalars(
        (reference["gram"][:1, :1])[None], np.ones((1, 1)), reference["same"], np.ones(1),
        reference_n_samples=n_r,
    )
    correction = classic_different_individual_correction(
        scalars, f_s - s * f_r, masses=np.ones(1)
    )
    return same + s * (reference["gram"] - reference["same"]) + correction


def test_environment_and_context_moments_follow_their_definitions() -> None:
    rng = np.random.default_rng(1)
    e = rng.gamma(2.0, 1.0, size=40)
    e = (e - e.mean()) / e.std(ddof=1)
    moments = classic_environment_moments(e, residual_rank=35)
    assert moments["n_samples"] == 40 and moments["residual_rank"] == 35
    np.testing.assert_allclose(
        [moments["sum"], moments["sum_squares"], moments["sum_fourth_powers"]],
        [e.sum(), (e**2).sum(), (e**4).sum()], rtol=1e-12, atol=1e-12,
    )
    g, f = classic_context_moments(moments)
    m2, rho = (e**2).mean(), 35 / 40
    u = rho * np.column_stack([np.ones(40), e**2 / m2])
    np.testing.assert_allclose(g, u.T @ u, rtol=1e-12)
    psi_g = np.ones((40, 40)) - np.eye(40)
    psi_e = np.outer(e, e) / m2 - np.diag(e**2 / m2)
    explicit = np.array([[np.sum(psi_g * psi_g), np.sum(psi_g * psi_e)],
                         [np.sum(psi_e * psi_g), np.sum(psi_e * psi_e)]])
    np.testing.assert_allclose(f, explicit, rtol=1e-10, atol=1e-8)
    # The per-variant feature normalization removes the environment's scale.
    scaled = classic_environment_moments(3.0 * e, residual_rank=35)
    for left, right in zip(classic_context_moments(scaled), (g, f)):
        np.testing.assert_allclose(left, right, rtol=1e-12)
    raw_g, raw_f = classic_context_moments(moments, feature_convention="raw_projected")
    np.testing.assert_allclose(raw_g, np.array([[40, (e**2).sum()], [(e**2).sum(), (e**4).sum()]]))
    with pytest.raises(ValueError, match="feature convention"):
        classic_context_moments(moments, feature_convention="genie")


def test_study_equal_to_reference_is_exact() -> None:
    _, g_raw, e_raw, covariates, _ = _design(2, 300, 120, "age")
    reference = _cohort(g_raw, e_raw, covariates, np.arange(300))
    g_r, f_r = classic_context_moments(reference["moments"])
    same = classic_study_same_individual_products(
        reference["same"], g_r, g_r, reference_n_samples=300, study_n_samples=300
    )
    np.testing.assert_array_equal(same, reference["same"])
    correction = classic_different_individual_correction(
        np.ones((3, 1, 1)), f_r - 1.0 * f_r, masses=np.ones(1)
    )
    assert not np.any(correction)
    equations = GxENormalEquations(
        reference["full"], np.zeros(4), reference["traces"], ("G:a", "GxE:a", "NxE", "residual")
    )
    options = dict(
        reference_n_samples=300, study_n_samples=300, reference_residual_rank=reference["r"],
        study_residual_rank=reference["r"], same_individual_products=reference["same"],
        genetic_nxe_traces=reference["full"][:2, 2], q_nxe=1.0, q_residual=float(reference["r"]),
        trace_nxe=reference["full"][2, 3], trace_nxe_sq=reference["full"][2, 2],
    )
    legacy = transfer_reference_normal_equations(equations, **options)
    corrected = transfer_reference_normal_equations(
        equations, **options, study_same_individual_products=same,
        same_person_transport="fourth_moment_difference",
        different_individual_correction=correction,
        different_person_transport="composition_corrected",
    )
    np.testing.assert_array_equal(corrected.matrix, legacy.matrix)
    assert corrected.diagnostics["same_person_transport"] == "fourth_moment_difference"
    assert corrected.diagnostics["different_person_transport"] == "composition_corrected"
    assert legacy.diagnostics["same_person_transport"] == "scaled_legacy"


@pytest.mark.parametrize("mode", ["age", "sex"])
def test_environment_selected_study_gram_is_corrected(mode) -> None:
    _, g_raw, e_raw, covariates, rows = _design(3, 1000, 300, mode)
    reference = _cohort(g_raw, e_raw, covariates, np.arange(1000))
    study = _cohort(g_raw, e_raw, covariates, rows)
    legacy = _transported(reference, study, corrected=False)
    corrected = _transported(reference, study, corrected=True)
    error = {
        name: np.abs(value - study["gram"]) / np.abs(study["gram"])
        for name, value in (("legacy", legacy), ("corrected", corrected))
    }
    # The GxE-GxE entry carries the environment's normalized fourth moment.
    assert error["legacy"][1, 1] > 0.05
    assert error["corrected"][1, 1] < 0.35 * error["legacy"][1, 1]
    assert error["corrected"][0, 0] <= error["legacy"][0, 0] + 1e-3


def test_transfer_composes_the_documented_terms_and_checks_labels() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(6, 6))
    full = base @ base.T + 6 * np.eye(6)
    k = 2
    same = 0.1 * np.eye(2 * k)
    full[:2 * k, :2 * k] += 0.0
    traces = np.full(6, 50.0)
    equations = GxENormalEquations(full, np.zeros(6), traces, tuple("abcdef"))
    options = dict(
        reference_n_samples=60, study_n_samples=40, reference_residual_rank=50,
        study_residual_rank=35, same_individual_products=same,
        genetic_nxe_traces=np.ones(2 * k), q_nxe=1.0, q_residual=35.0,
        trace_nxe=3.0, trace_nxe_sq=4.0,
    )
    study_same = 0.05 * np.eye(2 * k)
    correction = 0.01 * np.ones((2 * k, 2 * k))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = transfer_reference_normal_equations(
            equations, **options, study_same_individual_products=study_same,
            same_person_transport="fourth_moment_difference",
            different_individual_correction=correction,
            different_person_transport="composition_corrected",
        )
    s = 40 * 39 / (60 * 59)
    expected = study_same + s * (full[:2 * k, :2 * k] - same) + correction
    np.testing.assert_allclose(result.matrix[:2 * k, :2 * k], 0.5 * (expected + expected.T))
    with pytest.raises(ValueError, match="fourth_moment_difference"):
        transfer_reference_normal_equations(
            equations, **options, study_same_individual_products=study_same
        )
    with pytest.raises(ValueError, match="composition_corrected"):
        transfer_reference_normal_equations(
            equations, **options, different_individual_correction=correction
        )


def _fit_bias(seed, *, n, m, mode, sigma_gxe, replicates):
    rng, g_raw, e_raw, covariates, rows = _design(seed, n, m, mode)
    reference = _cohort(g_raw, e_raw, covariates, np.arange(n))
    study = _cohort(g_raw, e_raw, covariates, rows)
    names = ("G:a", "GxE:a", "NxE", "residual")
    systems = {}
    exact = study["full"]
    for label, corrected in (("legacy", False), ("corrected", True)):
        matrix = exact.copy()
        matrix[:2, :2] = _transported(reference, study, corrected=corrected)
        systems[label] = matrix
    systems["exact"] = exact
    # Phenotypes from the study's own kernel model: E[y'Ky] = sum theta tr(KK),
    # so the study's own normal equations are unbiased by construction.
    x = study["projector"] @ study["g"]
    w = study["projector"] @ (study["g"] * study["e"][:, None])
    x *= np.sqrt(study["r"] / np.sum(x * x, axis=0))
    w *= np.sqrt(study["r"] / np.sum(w * w, axis=0))
    theta = np.array([0.3, sigma_gxe, 0.05, 0.6])
    kernels = [study["kx"], study["kw"], study["kn"], study["projector"]]
    estimates = {label: [] for label in systems}
    for _ in range(replicates):
        y = (
            np.sqrt(theta[0] / m) * x @ rng.normal(size=m)
            + np.sqrt(sigma_gxe / m) * w @ rng.normal(size=m)
            + np.sqrt(theta[2]) * study["projector"] @ (study["e"] * rng.normal(size=study["n"]))
            + np.sqrt(theta[3]) * study["projector"] @ rng.normal(size=study["n"])
        )
        # The solver works on the normalized phenotype scale, y'Py = r.
        y *= np.sqrt(study["r"] / (y @ y))
        rhs = np.array([y @ kernel @ y for kernel in kernels])
        for label, matrix in systems.items():
            fit = solve_normal_equations(
                GxENormalEquations(matrix, rhs, study["traces"], names),
                allow_ill_conditioned=True, max_condition=1e16,
            )
            estimates[label].append(fit.coefficients)
    estimates = {label: np.asarray(value) for label, value in estimates.items()}
    se = estimates["exact"].std(axis=0, ddof=1)
    bias = {
        label: (estimates[label] - estimates["exact"]).mean(axis=0) / se
        for label in ("legacy", "corrected")
    }
    mc_error = {
        label: (estimates[label] - estimates["exact"]).std(axis=0, ddof=1)
        / se / np.sqrt(replicates)
        for label in ("legacy", "corrected")
    }
    return bias, mc_error, estimates


def test_end_to_end_gxe_variance_bias_disappears_with_the_correction() -> None:
    # Age-like environment, older people more often observed. Biases are in
    # units of the GxE coefficient's s.e. under the study's own Gram.
    bias, mc_error, _ = _fit_bias(11, n=1500, m=250, mode="age", sigma_gxe=0.3, replicates=150)
    assert bias["legacy"][1] < -0.5
    assert abs(bias["corrected"][1]) < 0.2 * abs(bias["legacy"][1])
    assert max(mc_error["legacy"][1], mc_error["corrected"][1]) < 0.05
    # With no GxE only the GxE-GxE entry is wrong, which multiplies a zero
    # coefficient: neither transport biases the null GxE estimate materially.
    null, _, _ = _fit_bias(11, n=1500, m=250, mode="age", sigma_gxe=0.0, replicates=150)
    assert abs(null["legacy"][1]) < 0.1 and abs(null["corrected"][1]) < 0.1


def test_classic_formulas_equal_generalized_ones_for_raw_features() -> None:
    _, g_raw, e_raw, covariates, rows = _design(6, 250, 80, "age")
    for selection in (np.arange(250), rows):
        cohort = _cohort(g_raw, e_raw, covariates, selection, convention="raw_projected")
        phi = np.column_stack([np.ones(cohort["n"]), cohort["e"]])
        projector = rank_revealing_projector(np.column_stack([np.ones(cohort["n"]), cohort["basis"]]))
        features = np.stack([projector.projector @ (phi[:, q, None] * cohort["g"]) for q in range(2)])
        components = ContextComponentIndex(("all",), ContextPairIndex(2))
        kernels = dense_genetic_kernels(features, np.ones((80, 1)), components)
        # Pair order (0,0), (1,1), (0,1): the classic G and GxE kernels are the
        # generalized (0,0) and (1,1) kernels; the classic model drops (0,1).
        np.testing.assert_allclose(kernels[0], cohort["kx"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(kernels[1], cohort["kw"], rtol=1e-10, atol=1e-12)
        g, f = classic_context_moments(cohort["moments"], feature_convention="raw_projected")
        keep = [0, 1]
        np.testing.assert_allclose(g, context_fourth_moments(phi)[np.ix_(keep, keep)], rtol=1e-12)
        np.testing.assert_allclose(
            f,
            distinct_pair_exposure(context_second_moments(phi), context_fourth_moments(phi))[
                np.ix_(keep, keep)],
            rtol=1e-12, atol=1e-6,
        )
    reference = _cohort(g_raw, e_raw, covariates, np.arange(250), convention="raw_projected")
    study = _cohort(g_raw, e_raw, covariates, rows, convention="raw_projected")
    phi_r = np.column_stack([np.ones(250), reference["e"]])
    phi_s = np.column_stack([np.ones(len(rows)), study["e"]])
    g_r, _ = classic_context_moments(reference["moments"], feature_convention="raw_projected")
    g_s, _ = classic_context_moments(study["moments"], feature_convention="raw_projected")
    classic = classic_study_same_individual_products(
        reference["same"], g_r, g_s, reference_n_samples=250, study_n_samples=len(rows))
    # The generalized summary form on the full pair axis, restricted to (0,0),(1,1).
    full_same = np.zeros((3, 3))
    full_same[:2, :2] = reference["same"]
    full_same[2, 2] = 1.0
    generalized = approximate_study_same_person(
        full_same, context_fourth_moments(phi_r), context_fourth_moments(phi_s),
        reference_n=250, study_n=len(rows), annotation_count=1,
    )[:2, :2]
    np.testing.assert_allclose(classic, generalized, rtol=1e-12, atol=1e-10)
    # Same model, same normal equations: the generalized dense system with the
    # (0,0) and (1,1) kernels and residual kernels P diag(e^2) P and P is the
    # classic system, so both solves give the same coefficients.
    from summit.context.oracle import dense_normal_equations, dense_residual_kernels

    phi_s = np.column_stack([np.ones(study["n"]), study["e"]])
    features = np.stack([study["projector"] @ (phi_s[:, q, None] * study["g"]) for q in range(2)])
    kernels = dense_genetic_kernels(
        features, np.ones((80, 1)), ContextComponentIndex(("all",), ContextPairIndex(2)))
    residual = dense_residual_kernels(
        study["projector"], np.column_stack([study["e"] ** 2, np.ones(study["n"])]))
    y = study["projector"] @ np.random.default_rng(7).normal(size=study["n"])
    y *= np.sqrt(study["r"] / (y @ y))
    generalized_system = dense_normal_equations(
        kernels[:2], residual, y, ("G", "GxE"), ("NxE", "residual"))
    np.testing.assert_allclose(generalized_system.matrix, study["full"], rtol=1e-10)
    names = ("G:a", "GxE:a", "NxE", "residual")
    classic_fit = solve_normal_equations(
        GxENormalEquations(study["full"], generalized_system.rhs, study["traces"], names),
        allow_ill_conditioned=True, max_condition=1e16,
    )
    np.testing.assert_allclose(
        classic_fit.coefficients,
        np.linalg.solve(generalized_system.matrix, generalized_system.rhs),
        rtol=1e-8, atol=1e-10,
    )
