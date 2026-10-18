"""Between-block signal covariance of general G×E moments (crossblock_signal)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from summit.context.crossblock_signal import (
    CrossBlockSignalInputs,
    CrossBlockSignalUnavailableWarning,
    block_traces_from_ld_scalars,
    cross_model_crossblock_covariance,
    crossblock_moment_covariance,
    crossblock_signal_covariance,
    derived_variance,
    resolve_standard_errors,
)
from summit.context.oracle import coefficients_to_omegas
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    context_fourth_moments,
    context_second_moments,
)
from summit.ldscore.generalized_gxe_chromosome import (
    chromosome_paired_contrast,
    fit_transferred_chromosome_models,
    prepare_chromosome_transport,
    transferred_chromosome_equations,
)
from summit.ldscore.generalized_gxe_fit_v1 import fit_generalized_gxe_variant_model_v1
from test_different_person_transport import _design
from test_generalized_gxe_fit_v1 import _synthetic_pair


def _basis(rng, n, centered=False):
    c1 = rng.normal(size=n)
    c2 = (rng.random(n) < 0.3).astype(float)
    exposures = np.column_stack([c1, c2])
    if centered:
        exposures = (exposures - exposures.mean(axis=0)) / exposures.std(axis=0)
    return np.column_stack([np.ones(n), exposures])


def _brute_force(index, omegas, traces, phi):
    """C from explicit per-person w vectors and an explicit pair sum over blocks."""
    n, q = phi.shape
    k = len(index.annotation_names)
    pairs = index.pair_index.entries
    sigma = phi.T @ phi / n
    w = np.zeros((k, n, len(pairs)))
    for a in range(k):
        z = phi @ omegas[a] @ sigma
        for pair in pairs:
            if pair.q == pair.r:
                w[a, :, pair.index] = phi[:, pair.q] * z[:, pair.q]
            else:
                w[a, :, pair.index] = phi[:, pair.q] * z[:, pair.r] + phi[:, pair.r] * z[:, pair.q]
    blocks = traces.shape[0]
    g = np.zeros((k, k, k, k))
    for x in range(blocks):
        for y in range(blocks):
            if x != y:
                g += np.einsum("ab,cd->abcd", traces[x], traces[y])
    c = len(index)
    out = np.zeros((c, c))
    for left in index.entries:
        for right in index.entries:
            value = 0.0
            for b in range(k):
                for d in range(k):
                    value += g[left.annotation_index, b, right.annotation_index, d] * float(
                        w[b, :, left.pair_index] @ w[d, :, right.pair_index])
            out[left.index, right.index] = 4.0 * value
    return out


def _inputs(traces, phi):
    return CrossBlockSignalInputs(block_traces=traces, study_n=phi.shape[0],
                                  study_second_moments=context_second_moments(phi),
                                  study_fourth_moments=context_fourth_moments(phi))


def test_moment_covariance_matches_brute_force_with_two_annotations() -> None:
    rng = np.random.default_rng(3)
    phi = _basis(rng, 61)
    index = ContextComponentIndex(("a0", "a1"), ContextPairIndex(3))
    raw = rng.normal(size=(2, 3, 3))
    omegas = 0.5 * (raw + np.swapaxes(raw, 1, 2))
    traces = rng.uniform(0.5, 2.0, size=(5, 2, 2))
    traces = 0.5 * (traces + np.swapaxes(traces, 1, 2))
    observed, diagnostics = crossblock_moment_covariance(index, omegas, _inputs(traces, phi))
    np.testing.assert_allclose(observed, _brute_force(index, omegas, traces, phi), rtol=1e-12, atol=1e-10)
    assert diagnostics["jackknife_blocks"] == 5


def test_null_moment_covariance_reduces_to_exposure_covariance() -> None:
    # Omega = diag(omega_00, 0, ...) with centered contexts: w_0k(i) = omega_00 c_ik, so
    # C[(0,k),(0,l)] = 4 G omega_00^2 sum_i c_ik c_il and the G x E variance entries vanish.
    rng = np.random.default_rng(5)
    phi = _basis(rng, 80, centered=True)
    index = ContextComponentIndex(("all",), ContextPairIndex(3))
    omegas = np.zeros((1, 3, 3)); omegas[0, 0, 0] = 0.4
    traces = rng.uniform(1.0, 3.0, size=(7, 1, 1))
    observed, _ = crossblock_moment_covariance(index, omegas, _inputs(traces, phi))
    g = traces.sum() ** 2 - np.sum(traces ** 2)
    entry = {(e.q, e.r): e.index for e in index.entries}
    for k in (1, 2):
        for l in (1, 2):
            expected = 4.0 * g * 0.4 ** 2 * float(phi[:, k] @ phi[:, l])
            assert observed[entry[(0, k)], entry[(0, l)]] == pytest.approx(expected, rel=1e-12)
        assert abs(observed[entry[(k, k)], entry[(k, k)]]) < 1e-20 * abs(observed[entry[(0, k)], entry[(0, k)]]) + 1e-25


def test_leading_order_matches_exact_profiled_cross_block_covariance() -> None:
    # Exact Gaussian covariance between residual-profiled block moments,
    # 2 tr(H~_u^b V H~_v^b' V) summed over b != b', against 4 G sum_i w w.
    rng = np.random.default_rng(0)
    n, blocks, per, rho = 240, 6, 80, 0.9
    x = np.empty((n, blocks * per))
    for b in range(blocks):
        z = rng.normal(size=(n, per))
        for j in range(1, per):
            z[:, j] = rho * z[:, j - 1] + np.sqrt(1.0 - rho ** 2) * z[:, j]
        x[:, b * per:(b + 1) * per] = z
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    phi = _basis(rng, n, centered=True)
    projection = np.eye(n) - phi @ np.linalg.solve(phi.T @ phi, phi.T)
    kernels = [x[:, b * per:(b + 1) * per] @ x[:, b * per:(b + 1) * per].T / x.shape[1] for b in range(blocks)]
    pairs = ContextPairIndex(3).entries
    omega = np.zeros((3, 3)); omega[0, 0] = 0.5; omega[0, 1] = omega[1, 0] = 0.1
    v = sum(omega[s, t] * (phi[:, s, None] * kern * phi[None, :, t])
            for kern in kernels for s in range(3) for t in range(3)) + 0.5 * np.eye(n)
    v = projection @ v @ projection
    monomials = [np.ones(n), phi[:, 1], phi[:, 2], phi[:, 1] ** 2, phi[:, 1] * phi[:, 2]]
    residual = [projection @ np.diag(m) @ projection for m in monomials]
    t_ee = np.array([[np.sum(a * b) for b in residual] for a in residual])
    profiled = {}
    for b, kern in enumerate(kernels):
        for pair in pairs:
            dq = phi[:, pair.q, None] * kern * phi[None, :, pair.r]
            h = projection @ (dq if pair.q == pair.r else dq + dq.T) @ projection
            kappa = np.linalg.solve(t_ee, np.array([np.sum(h * e) for e in residual]))
            profiled[b, pair.index] = (h - sum(k * e for k, e in zip(kappa, residual))) @ v
    exact = np.zeros((len(pairs), len(pairs)))
    for b in range(blocks):
        for b2 in range(blocks):
            if b != b2:
                for u in range(len(pairs)):
                    for w in range(len(pairs)):
                        exact[u, w] += 2.0 * np.sum(profiled[b, u] * profiled[b2, w].T)
    traces = []
    for kern in kernels:
        projected = projection @ kern @ projection
        traces.append((np.sum(projected ** 2) - np.sum(np.diag(projected) ** 2)) / n)
    index = ContextComponentIndex(("all",), ContextPairIndex(3))
    formula, _ = crossblock_moment_covariance(index, omega[None], _inputs(np.array(traces)[:, None, None], phi))
    entry = {(e.q, e.r): e.index for e in index.entries}
    for key in ((0, 1), (0, 0)):
        ratio = formula[entry[key], entry[key]] / exact[entry[key], entry[key]]
        assert 0.6 < ratio < 1.6, (key, ratio)
    # G x E variance entries carry no leading-order term here; the exact one is small.
    assert abs(exact[entry[(2, 2)], entry[(2, 2)]]) < 0.2 * exact[entry[(0, 1)], entry[(0, 1)]]


def test_block_traces_from_ld_scalars_scale_and_merge_blocks() -> None:
    scalars = np.arange(1.0, 13.0).reshape(3, 2, 2)
    masses = np.array([2.0, 5.0])
    traces = block_traces_from_ld_scalars(scalars, masses, 11)
    np.testing.assert_allclose(traces, 10.0 * scalars / np.outer(masses, masses)[None])
    merged = block_traces_from_ld_scalars(scalars, masses, 11, block_ids=[4, 9, 4])
    np.testing.assert_allclose(merged[0], traces[0] + traces[2])
    np.testing.assert_allclose(merged[1], traces[1])
    with pytest.raises(ValueError):
        resolve_standard_errors("bootstrap")
    assert resolve_standard_errors(None) == "crossblock"


def _with_context_moments(trait, rng):
    q = trait.component_index.pair_index.num_basis
    phi = np.column_stack([np.ones(trait.n_samples), rng.normal(size=(trait.n_samples, q - 1))])
    return replace(trait, context_second_moments=context_second_moments(phi),
                   context_fourth_moments=context_fourth_moments(phi))


def test_variant_fit_adds_crossblock_covariance_by_default(monkeypatch) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    trait = _with_context_moments(trait, np.random.default_rng(11))
    corrected = fit_generalized_gxe_variant_model_v1(reference, trait)
    jackknife = fit_generalized_gxe_variant_model_v1(reference, trait, standard_errors="jackknife")
    assert corrected.manifest["uncertainty"]["crossblock_status"] == "applied"
    assert jackknife.manifest["uncertainty"]["crossblock_status"] == "not_requested"
    np.testing.assert_array_equal(corrected.raw_coefficients, jackknife.raw_coefficients)
    np.testing.assert_array_equal(corrected.raw_jackknife_covariance, jackknife.raw_jackknife_covariance)
    np.testing.assert_array_equal(jackknife.raw_crossblock_covariance, 0.0)
    np.testing.assert_array_equal(jackknife.raw_standard_errors, jackknife.raw_jackknife_standard_errors)
    inputs = CrossBlockSignalInputs(
        block_traces=block_traces_from_ld_scalars(reference.ld_scalars, reference.annotation_masses, trait.n_samples),
        study_n=trait.n_samples, study_second_moments=trait.context_second_moments,
        study_fourth_moments=trait.context_fourth_moments)
    added, _ = crossblock_signal_covariance(corrected.normal_matrix, corrected.raw_coefficients,
                                            reference.component_index, inputs)
    np.testing.assert_allclose(corrected.raw_crossblock_covariance, added, rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(corrected.raw_covariance, corrected.raw_jackknife_covariance + added, rtol=1e-13)
    np.testing.assert_allclose(corrected.raw_standard_errors, np.sqrt(np.diag(corrected.raw_covariance)), rtol=1e-13)


def test_variant_fit_without_context_moments_falls_back_with_warning(monkeypatch) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    with pytest.warns(CrossBlockSignalUnavailableWarning):
        fit = fit_generalized_gxe_variant_model_v1(reference, trait)
    assert fit.manifest["uncertainty"]["crossblock_status"] == "not_applied_missing_study_context_moments"
    np.testing.assert_array_equal(fit.raw_standard_errors, fit.raw_jackknife_standard_errors)


def test_chromosome_models_reproduce_the_manual_jackknife_and_add_the_correction() -> None:
    reference, study, diagonals, phi, rows, moments = _design()
    transport = prepare_chromosome_transport(reference, diagonals, phi)
    total = transferred_chromosome_equations(reference, study, **transport.for_study(rows), **moments).matrix.shape[0]
    c = len(ContextComponentIndex(reference[0].annotation_names, ContextPairIndex(3)))
    restricted = np.array([0, 1, 3] + [6, 7, 9] + list(range(c, total)))
    models = {"full": np.arange(total), "restricted": restricted}
    fits = fit_transferred_chromosome_models(reference, study, models=models, transport=transport,
                                             study_rows=rows, **moments)
    plain = fit_transferred_chromosome_models(reference, study, models=models, transport=transport,
                                              study_rows=rows, standard_errors="jackknife", **moments)
    terms = transport.for_study(rows)
    blocks = np.unique(np.concatenate([x.block_ids for x in reference]))
    full = transferred_chromosome_equations(reference, study, **terms, **moments)
    point = np.linalg.solve(full.matrix, full.rhs)
    loo = np.array([np.linalg.solve(e.matrix, e.rhs) for e in (
        transferred_chromosome_equations(reference, study, **terms, **moments, deleted_blocks=(int(b),))
        for b in blocks)])
    np.testing.assert_allclose(fits["full"].coefficients, point, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fits["full"].loo_coefficients, loo, rtol=1e-10, atol=1e-12)
    centered = loo - loo.mean(axis=0)
    np.testing.assert_allclose(fits["full"].jackknife_covariance,
                               (len(loo) - 1) / len(loo) * centered.T @ centered, rtol=1e-9, atol=1e-14)
    for name in models:
        assert fits[name].uncertainty["crossblock_status"] == "applied"
        assert plain[name].uncertainty["crossblock_status"] == "not_requested"
        np.testing.assert_array_equal(plain[name].crossblock_covariance, 0.0)
        np.testing.assert_array_equal(plain[name].jackknife_covariance, fits[name].jackknife_covariance)
        outside = np.setdiff1d(np.arange(total), fits[name].keep)
        np.testing.assert_array_equal(fits[name].coefficients[outside], 0.0)
    masses = sum(x.block_masses.sum(axis=0) for x in reference)
    scalars = np.concatenate([x.ld_scalars(transport.same_person_intercepts[x.chromosome]) for x in reference])
    inputs = CrossBlockSignalInputs(
        block_traces=block_traces_from_ld_scalars(scalars, masses, len(rows)), study_n=len(rows),
        study_second_moments=context_second_moments(phi, rows), study_fourth_moments=context_fourth_moments(phi, rows))
    added, _ = crossblock_signal_covariance(full.matrix, point, fits["full"].component_index, inputs)
    np.testing.assert_allclose(fits["full"].crossblock_covariance, added, rtol=1e-9, atol=1e-14)
    # Derived quantity and paired contrast: jackknife part plus gradient terms.
    amplification = lambda om: om.sum(axis=0)[0, 1] / om.sum(axis=0)[0, 0]
    derived = fits["full"].derived(amplification)
    assert derived["variance"] == pytest.approx(derived["jackknife_variance"] + derived["crossblock_variance"])
    contrast = chromosome_paired_contrast(fits["full"], amplification, fits["restricted"], amplification)
    cross = cross_model_crossblock_covariance(
        full.matrix, fits["full"].component_index, inputs, omegas=fits["full"].omegas,
        keep_a=fits["full"].keep, keep_b=fits["restricted"].keep)
    assert cross.shape == (total, len(restricted))
    # The restricted model's correction uses the full model's Omega (one shared C per trait).
    assert fits["restricted"].uncertainty["crossblock_diagnostics"]["omega_source"] == "model:full"
    np.testing.assert_array_equal(fits["restricted"].crossblock_moment_covariance,
                                  fits["full"].crossblock_moment_covariance)
    shared, _ = crossblock_signal_covariance(full.matrix, point, fits["full"].component_index, inputs,
                                             keep=restricted, omegas=fits["full"].omegas)
    np.testing.assert_allclose(fits["restricted"].crossblock_covariance[np.ix_(restricted, restricted)], shared,
                               rtol=1e-9, atol=1e-14)
    # Paired contrast = variance of the paired jackknife difference + a_J + a_S - 2 a_JS (delta method).
    from summit.context.crossblock_signal import numerical_gradient
    c_ = len(fits["full"].component_index)
    f = lambda t: amplification(coefficients_to_omegas(t[:c_], fits["full"].component_index))
    ga = numerical_gradient(f, fits["full"].coefficients)
    gb = numerical_gradient(f, fits["restricted"].coefficients)[restricted]
    expected = (ga @ fits["full"].crossblock_covariance @ ga
                + gb @ fits["restricted"].crossblock_covariance[np.ix_(restricted, restricted)] @ gb
                - 2.0 * ga @ cross @ gb)
    assert contrast["crossblock_variance"] == pytest.approx(expected, rel=1e-9)
    diffs = np.array([f(a) - f(b) for a, b in zip(fits["full"].loo_coefficients, fits["restricted"].loo_coefficients)])
    assert contrast["jackknife_variance"] == pytest.approx((len(diffs) - 1) / len(diffs) * np.sum((diffs - diffs.mean()) ** 2))


def test_derived_variance_is_jackknife_without_added_covariance() -> None:
    rng = np.random.default_rng(2)
    theta = rng.normal(size=4) + 2.0
    loo = theta + 0.01 * rng.normal(size=(30, 4))
    f = lambda t: t[1] / t[0]
    out = derived_variance(f, theta, loo, None)
    values = loo[:, 1] / loo[:, 0]
    assert out["jackknife_variance"] == pytest.approx(29 / 30 * np.sum((values - values.mean()) ** 2))
    assert out["crossblock_variance"] == 0.0
    added = np.diag([1e-4, 2e-4, 0.0, 0.0])
    out = derived_variance(f, theta, loo, added)
    gradient = np.array([-theta[1] / theta[0] ** 2, 1 / theta[0], 0.0, 0.0])
    assert out["crossblock_variance"] == pytest.approx(gradient @ added @ gradient, rel=1e-6)
