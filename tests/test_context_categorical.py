from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from summit.context import (
    BoundaryTestResult,
    CategoricalContextPreset,
    ContextComponentIndex,
    ContextPairIndex,
    ContextRankError,
    array_sha256,
    assemble_context_normal_equations,
    binary_basis_transform,
    binary_intercept_to_one_hot_coefficients,
    binary_intercept_to_one_hot_covariance,
    binary_one_hot_to_intercept_coefficients,
    binary_one_hot_to_intercept_covariance,
    build_categorical_preset,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    coefficients_to_omegas,
    combine_genetic_kernels,
    common_scale_features,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    derive_binary_context_fit,
    derive_categorical_context_fit,
    fit_context_model,
    kernel_gram,
    omegas_to_coefficients,
    project_genetic_coefficients_psd,
    project_normalize_phenotype,
    rank_revealing_projector,
    transform_omega,
)
from summit.context import test_binary_boundary as _test_binary_boundary


def _scaled_genotype(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    return genotype


def _binary_fixture(seed: int = 501) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n, m = 36, 30
    context = np.tile(np.array([0, 1]), n // 2)
    preset = build_categorical_preset(
        context, categories=(0, 1), source_name="exposure", binary=True
    )
    continuous = 0.8 * context + rng.normal(size=n)
    fixed = np.column_stack([np.ones(n), preset.fixed_effect_columns, continuous])
    projector = rank_revealing_projector(fixed)
    return {
        "rng": rng,
        "n": n,
        "m": m,
        "context": context,
        "preset": preset,
        "continuous": continuous,
        "fixed": fixed,
        "projector": projector,
        "genotype": _scaled_genotype(rng, n, m),
        "phenotype": rng.normal(size=n),
        "annotations": np.ones((m, 1), dtype=np.float64),
        "loo_groups": tuple(f"group:{index // 5}" for index in range(m)),
        "variant_hash": canonical_sha256({"variants": list(range(m))}),
    }


def _build_objects(
    fixture: dict[str, object],
    basis: np.ndarray,
    residual_basis: np.ndarray,
    *,
    basis_hash: str,
):
    components = ContextComponentIndex(("all",), ContextPairIndex(int(basis.shape[1])))
    residual_names = tuple(
        f"residual:category:{index}" for index in range(residual_basis.shape[1])
    )
    common = {
        "genotype": fixture["genotype"],
        "basis": basis,
        "projector": fixture["projector"],
        "annotations": fixture["annotations"],
        "component_index": components,
        "loo_groups": fixture["loo_groups"],
        "basis_hash": basis_hash,
        "fixed_effect_hash": array_sha256(fixture["fixed"]),
        "variant_hash": fixture["variant_hash"],
        "genotype_scaling": "pre_scaled_input",
    }
    reference = build_context_reference(
        **common, gram_method="exact", same_person_method="exact"
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=fixture["phenotype"],
        residual_basis=residual_basis,
        residual_names=residual_names,
        block_size=5,
    )
    return components, reference, summary


def _binary_objects(seed: int = 501):
    fixture = _binary_fixture(seed)
    preset = fixture["preset"]
    h = np.asarray(preset.basis)
    z = h @ binary_basis_transform().T
    h_objects = _build_objects(
        fixture, h, np.asarray(preset.residual_basis), basis_hash=preset.basis_hash
    )
    z_objects = _build_objects(
        fixture, z, np.asarray(preset.residual_basis), basis_hash=array_sha256(z)
    )
    return fixture, h_objects, z_objects


def _coefficient_transform(
    components: ContextComponentIndex, basis_transform: np.ndarray
) -> np.ndarray:
    result = np.empty((len(components), len(components)), dtype=np.float64)
    for column in range(len(components)):
        unit = np.zeros(len(components), dtype=np.float64)
        unit[column] = 1.0
        omega = coefficients_to_omegas(unit, components)[0]
        transformed = transform_omega(omega, basis_transform)[None, :, :]
        result[:, column] = omegas_to_coefficients(transformed, components)
    return result


def _synthetic_binary_fit(seed: int = 601):
    fixture, (components, reference, summary), _ = _binary_objects(seed)
    fitted = fit_context_model(reference, summary)
    point_genetic = np.array([1.0, 0.64, 0.60])
    point_residual = np.array([0.4, 0.8])
    genetic_deviations = np.array(
        [
            [-0.06, 0.03, -0.02],
            [-0.03, -0.02, 0.01],
            [0.00, 0.01, 0.00],
            [0.02, -0.03, -0.01],
            [0.04, 0.00, 0.01],
            [0.03, 0.01, 0.01],
        ]
    )
    residual_deviations = np.array(
        [
            [-0.02, 0.03],
            [0.01, -0.01],
            [0.00, 0.01],
            [0.02, -0.02],
            [-0.01, 0.00],
            [0.00, -0.01],
        ]
    )
    loo = np.column_stack(
        [
            point_genetic + genetic_deviations,
            point_residual + residual_deviations,
        ]
    )
    centered = loo - loo.mean(axis=0, keepdims=True)
    covariance = (loo.shape[0] - 1.0) / loo.shape[0] * centered.T @ centered
    point = np.concatenate([point_genetic, point_residual])
    synthetic = replace(
        fitted,
        raw_coefficients=point,
        genetic_coefficients=point_genetic,
        residual_coefficients=point_residual,
        raw_omegas=coefficients_to_omegas(point_genetic, components),
        loo_coefficients=loo,
        jackknife_covariance=covariance,
        standard_errors=np.sqrt(np.diag(covariance)),
        psd_projection=None,
    )
    fixture["reference"] = reference
    fixture["summary"] = summary
    return fixture, synthetic


def test_binary_preset_is_exact_hash_stable_one_hot_encoding() -> None:
    context = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    preset = build_categorical_preset(
        context, categories=(0, 1), source_name="exposure", binary=True
    )
    assert isinstance(preset, CategoricalContextPreset)
    assert preset.category_labels == (0, 1)
    np.testing.assert_array_equal(preset.category_counts, [4, 4])
    expected = np.column_stack([context == 0, context == 1]).astype(np.float64)
    np.testing.assert_array_equal(preset.basis, expected)
    np.testing.assert_array_equal(preset.residual_basis, expected)
    np.testing.assert_array_equal(
        preset.basis_spec.evaluate({"exposure": context}), expected
    )
    assert preset.basis_hash == preset.basis_spec.digest
    assert len(preset.category_order_hash) == 64

    fixed_with_intercept = np.column_stack(
        [np.ones(context.size), preset.fixed_effect_columns]
    )
    assert np.linalg.matrix_rank(fixed_with_intercept) == 2
    projected = rank_revealing_projector(fixed_with_intercept).projector @ expected
    np.testing.assert_allclose(projected, 0.0, rtol=0.0, atol=2e-15)

    row_order = np.array([7, 1, 5, 3, 0, 6, 2, 4])
    reordered = build_categorical_preset(
        context[row_order], source_name="exposure", binary=True
    )
    assert reordered.category_labels == preset.category_labels
    assert reordered.category_order_hash == preset.category_order_hash
    assert reordered.basis_hash == preset.basis_hash
    np.testing.assert_array_equal(reordered.basis, expected[row_order])


def test_categorical_preset_preserves_heterogeneous_python_scalar_types() -> None:
    context = [1, "1", 2, "2"]
    preset = build_categorical_preset(context, source_name="typed")
    assert len(preset.category_labels) == 4
    assert {type(value) for value in preset.category_labels} == {int, str}
    np.testing.assert_array_equal(preset.category_counts, np.ones(4, dtype=int))
    np.testing.assert_array_equal(np.sum(preset.basis, axis=1), 1.0)
    np.testing.assert_array_equal(
        preset.basis_spec.evaluate({"typed": context}, n_samples=len(context)),
        preset.basis,
    )

    # Python considers bool/int values equal even though their typed manifest
    # identities differ.  The generic NumPy one-hot evaluator cannot preserve
    # that distinction, so the preset must fail closed instead of overlapping.
    with pytest.raises(ValueError, match="compare equal|one-hot"):
        build_categorical_preset([True, 1, False, 0], source_name="typed")


def test_categorical_preset_keeps_every_category_without_extra_identity() -> None:
    context = np.array(["c", "a", "b", "a", "c", "b", "c"])
    preset = build_categorical_preset(
        context, categories=("a", "b", "c"), source_name="site"
    )
    assert preset.category_labels == ("a", "b", "c")
    np.testing.assert_array_equal(preset.category_counts, [2, 2, 3])
    assert preset.basis.shape == (context.size, 3)
    assert preset.residual_basis.shape == (context.size, 3)
    np.testing.assert_array_equal(np.sum(preset.basis, axis=1), 1.0)
    np.testing.assert_array_equal(preset.residual_basis, preset.basis)
    assert len(preset.basis_spec.columns) == 3
    assert all(column.kind == "one_hot" for column in preset.basis_spec.columns)
    components = ContextComponentIndex(("all",), ContextPairIndex(3))
    assert len(components) == 6


def test_categorical_preset_rejects_incomplete_duplicate_and_nonbinary_orders() -> None:
    context = np.array([0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="categor|observed|order"):
        build_categorical_preset(context, categories=(0, 1))
    with pytest.raises(ValueError, match="duplic|unique"):
        build_categorical_preset(context, categories=(0, 1, 1, 2))
    with pytest.raises(ValueError, match="binary|two"):
        build_categorical_preset(context, categories=(0, 1, 2), binary=True)


def test_binary_basis_and_coefficient_maps_are_exact_and_batched() -> None:
    transform = binary_basis_transform()
    expected_transform = np.array([[1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(transform, expected_transform)
    h = np.eye(2)
    np.testing.assert_array_equal(h @ transform.T, np.array([[1.0, 0.0], [1.0, 1.0]]))

    one_hot = np.array([1.2, 0.7, 0.4])
    expected_intercept = np.array([1.2, 1.1, -0.8])
    observed_intercept = binary_one_hot_to_intercept_coefficients(one_hot)
    np.testing.assert_allclose(observed_intercept, expected_intercept, atol=2e-15)
    np.testing.assert_allclose(
        binary_intercept_to_one_hot_coefficients(observed_intercept),
        one_hot,
        atol=2e-15,
    )
    batch = np.vstack([one_hot, 2.0 * one_hot, -one_hot])
    np.testing.assert_allclose(
        binary_intercept_to_one_hot_coefficients(
            binary_one_hot_to_intercept_coefficients(batch)
        ),
        batch,
        atol=3e-15,
    )


def test_binary_joint_covariance_maps_use_full_off_diagonal_information() -> None:
    covariance = np.array([[0.4, -0.08, 0.05], [-0.08, 0.3, -0.04], [0.05, -0.04, 0.2]])
    one_hot_to_intercept = np.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, -2.0], [-1.0, 0.0, 1.0]]
    )
    expected = one_hot_to_intercept @ covariance @ one_hot_to_intercept.T
    observed = binary_one_hot_to_intercept_covariance(covariance)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        binary_intercept_to_one_hot_covariance(observed),
        covariance,
        rtol=0.0,
        atol=3e-15,
    )
    diagonal_only = one_hot_to_intercept @ np.diag(np.diag(covariance))
    diagonal_only = diagonal_only @ one_hot_to_intercept.T
    assert np.max(np.abs(expected - diagonal_only)) > 0.05


def test_binary_dense_kernels_rhs_and_full_normal_matrix_are_congruent() -> None:
    fixture = _binary_fixture(seed=511)
    preset = fixture["preset"]
    h = np.asarray(preset.basis)
    transform = binary_basis_transform()
    z = h @ transform.T
    projector = fixture["projector"].projector
    features_h = common_scale_features(fixture["genotype"], h, projector)
    features_z = common_scale_features(fixture["genotype"], z, projector)
    np.testing.assert_allclose(
        features_z,
        np.einsum("ab,bnm->anm", transform, features_h),
        rtol=3e-15,
        atol=3e-14,
    )
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    genetic_h = dense_genetic_kernels(features_h, fixture["annotations"], components)
    genetic_z = dense_genetic_kernels(features_z, fixture["annotations"], components)
    intercept_to_one_hot = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
    expected_genetic_z = np.einsum("ab,aij->bij", intercept_to_one_hot, genetic_h)
    np.testing.assert_allclose(genetic_z, expected_genetic_z, rtol=5e-15, atol=5e-13)
    residual = dense_residual_kernels(projector, preset.residual_basis)
    phenotype = project_normalize_phenotype(fixture["phenotype"], fixture["projector"])
    dense_h = dense_normal_equations(
        genetic_h,
        residual,
        phenotype,
        components.names,
        ("residual:0", "residual:1"),
    )
    dense_z = dense_normal_equations(
        genetic_z,
        residual,
        phenotype,
        components.names,
        ("residual:0", "residual:1"),
    )
    coefficient_map = np.block(
        [
            [intercept_to_one_hot, np.zeros((3, 2))],
            [np.zeros((2, 3)), np.eye(2)],
        ]
    )
    np.testing.assert_allclose(
        dense_z.matrix,
        coefficient_map.T @ dense_h.matrix @ coefficient_map,
        rtol=8e-15,
        atol=8e-12,
    )
    np.testing.assert_allclose(
        dense_z.rhs,
        coefficient_map.T @ dense_h.rhs,
        rtol=6e-15,
        atol=6e-13,
    )
    assert abs(dense_h.matrix[3, 4]) > 1.0e-3
    assert np.min(np.abs(dense_h.matrix[:3, 3:])) > 1.0e-3


def test_binary_fits_match_for_every_replicate_covariance_surface_and_kernel() -> None:
    (
        fixture,
        (components, reference_h, summary_h),
        (_, reference_z, summary_z),
    ) = _binary_objects(seed=521)
    preset = fixture["preset"]
    h_grid = np.eye(2)
    z_grid = h_grid @ binary_basis_transform().T
    h_metric = np.asarray(preset.basis).T @ np.asarray(preset.basis) / fixture["n"]
    z_metric = binary_basis_transform() @ h_metric @ binary_basis_transform().T
    fit_h = fit_context_model(
        reference_h,
        summary_h,
        context_grid=h_grid,
        basis_metric=h_metric,
    )
    fit_z = fit_context_model(
        reference_z,
        summary_z,
        context_grid=z_grid,
        basis_metric=z_metric,
    )
    expected_h_genetic = binary_intercept_to_one_hot_coefficients(
        fit_z.genetic_coefficients
    )
    np.testing.assert_allclose(
        fit_h.genetic_coefficients,
        expected_h_genetic,
        rtol=3e-11,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        fit_h.residual_coefficients,
        fit_z.residual_coefficients,
        rtol=3e-11,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        fit_h.loo_coefficients[:, :3],
        binary_intercept_to_one_hot_coefficients(fit_z.loo_coefficients[:, :3]),
        rtol=4e-11,
        atol=4e-10,
    )
    np.testing.assert_allclose(
        fit_h.loo_coefficients[:, 3:],
        fit_z.loo_coefficients[:, 3:],
        rtol=4e-11,
        atol=4e-10,
    )
    intercept_to_one_hot = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
    joint_map = np.block(
        [
            [intercept_to_one_hot, np.zeros((3, 2))],
            [np.zeros((2, 3)), np.eye(2)],
        ]
    )
    np.testing.assert_allclose(
        fit_h.jackknife_covariance,
        joint_map @ fit_z.jackknife_covariance @ joint_map.T,
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        fit_h.context_outputs["annotations"][0]["covariance_surface"],
        fit_z.context_outputs["annotations"][0]["covariance_surface"],
        rtol=4e-11,
        atol=4e-10,
    )
    for h_output, z_output in zip(
        fit_h.jackknife_context_outputs, fit_z.jackknife_context_outputs
    ):
        np.testing.assert_allclose(
            h_output["annotations"][0]["covariance_surface"],
            z_output["annotations"][0]["covariance_surface"],
            rtol=6e-11,
            atol=6e-10,
        )

    projector = fixture["projector"].projector
    features_h = common_scale_features(fixture["genotype"], preset.basis, projector)
    features_z = common_scale_features(
        fixture["genotype"], z_grid[np.asarray(fixture["context"])], projector
    )
    genetic_h = dense_genetic_kernels(features_h, fixture["annotations"], components)
    genetic_z = dense_genetic_kernels(features_z, fixture["annotations"], components)
    residual = dense_residual_kernels(projector, preset.residual_basis)
    covariance_h = combine_genetic_kernels(genetic_h, fit_h.genetic_coefficients)
    covariance_h += combine_genetic_kernels(residual, fit_h.residual_coefficients)
    covariance_z = combine_genetic_kernels(genetic_z, fit_z.genetic_coefficients)
    covariance_z += combine_genetic_kernels(residual, fit_z.residual_coefficients)
    np.testing.assert_allclose(covariance_h, covariance_z, rtol=5e-11, atol=5e-10)


def test_binary_derived_quantities_and_jackknife_uncertainty_are_exact() -> None:
    fixture, fitted = _synthetic_binary_fit()
    summary = derive_binary_context_fit(fitted, fixture["preset"])
    expected_point = {
        "v0": 1.0,
        "v1": 0.64,
        "gamma": 0.60,
        "rho": 0.75,
        "log_sd_ratio": np.log(0.8),
        "tau2_1_given_0": 0.28,
    }
    genetic_loo = fitted.loo_coefficients[:, :3]
    expected_loo = {
        "v0": genetic_loo[:, 0],
        "v1": genetic_loo[:, 1],
        "gamma": genetic_loo[:, 2],
        "rho": genetic_loo[:, 2] / np.sqrt(genetic_loo[:, 0] * genetic_loo[:, 1]),
        "log_sd_ratio": 0.5 * np.log(genetic_loo[:, 1] / genetic_loo[:, 0]),
        "tau2_1_given_0": genetic_loo[:, 1]
        - genetic_loo[:, 2] ** 2 / genetic_loo[:, 0],
    }
    for name, expected in expected_point.items():
        estimate = summary.raw[name]
        assert estimate.estimate == pytest.approx(expected, abs=2e-14)
        np.testing.assert_allclose(
            estimate.loo_values, expected_loo[name], rtol=2e-14, atol=2e-14
        )
        centered = expected_loo[name] - np.mean(expected_loo[name])
        expected_se = np.sqrt(
            (expected_loo[name].size - 1.0)
            / expected_loo[name].size
            * (centered @ centered)
        )
        assert estimate.standard_error == pytest.approx(expected_se, rel=2e-13)
        assert estimate.defined
        assert estimate.status.startswith("defined")
    observed_residual = np.asarray(
        [
            summary.residual_variances[name].estimate
            for name in fixture["preset"].residual_names
        ]
    )
    np.testing.assert_array_equal(observed_residual, fitted.residual_coefficients)
    np.testing.assert_array_equal(summary.raw_omega, [[1.0, 0.60], [0.60, 0.64]])
    np.testing.assert_allclose(
        summary.loo_raw_omegas[:, 0, 0], genetic_loo[:, 0], atol=0.0
    )
    contrast = np.array([-1.0, 1.0, 0.0, 0.0, 0.0])
    expected_contrast_variance = contrast @ fitted.jackknife_covariance @ contrast
    assert summary.equal_variance_contrast.estimate == pytest.approx(-0.36)
    assert summary.equal_variance_contrast.standard_error == pytest.approx(
        np.sqrt(expected_contrast_variance), rel=2e-14
    )
    marginal_only_variance = (
        fitted.jackknife_covariance[0, 0] + fitted.jackknife_covariance[1, 1]
    )
    assert abs(expected_contrast_variance - marginal_only_variance) > 1.0e-4
    assert "including_v0_v1_covariance" in (
        summary.equal_variance_contrast.covariance_method
    )
    assert summary.psd_interpretable is None


def test_binary_trace_proportions_have_deleted_equation_uncertainty() -> None:
    fixture, fitted = _synthetic_binary_fit(seed=606)
    loo_equations = tuple(
        assemble_context_normal_equations(
            fixture["reference"], fixture["summary"], deleted_groups=(group,)
        )
        for group in fitted.jackknife_groups
    )
    summary = derive_binary_context_fit(
        fitted, fixture["preset"], loo_equations=loo_equations
    )
    assert tuple(summary.trace_variance_proportions) == tuple(
        fixture["preset"].residual_names
    )
    for estimate in summary.trace_variance_proportions.values():
        assert estimate.loo_values.shape == (len(fitted.jackknife_groups),)
        assert np.isfinite(estimate.estimate)
        assert np.isfinite(estimate.standard_error)
        assert estimate.status.startswith("defined")
    with pytest.raises(ValueError, match="jackknife group order"):
        derive_binary_context_fit(
            fitted,
            fixture["preset"],
            loo_equations=tuple(reversed(loo_equations)),
        )


def test_categorical_derived_apis_do_not_require_individual_context_rows() -> None:
    fixture, fitted = _synthetic_binary_fit(seed=607)
    descriptor = fixture["preset"].without_individual_data()
    assert not descriptor.has_individual_data
    with pytest.raises(ValueError, match="discarded"):
        descriptor.fixed_effect_design()
    derived = derive_binary_context_fit(fitted, descriptor)
    np.testing.assert_array_equal(derived.raw_omega, fitted.raw_omegas[0])
    boundary = _test_binary_boundary(
        fitted,
        "equal_variances",
        preset=descriptor,
        draws=1,
        seed=1,
    )
    assert np.isfinite(boundary.statistic)


def test_binary_psd_interpretation_is_separate_from_indefinite_raw_output() -> None:
    fixture, fitted = _synthetic_binary_fit(seed=608)
    raw_genetic = np.array([1.0, 0.25, 0.8])
    deviations = fitted.loo_coefficients[:, :3] - np.mean(
        fitted.loo_coefficients[:, :3], axis=0, keepdims=True
    )
    loo = fitted.loo_coefficients.copy()
    loo[:, :3] = raw_genetic + deviations
    centered = loo - loo.mean(axis=0, keepdims=True)
    covariance = (loo.shape[0] - 1.0) / loo.shape[0] * centered.T @ centered
    projection = project_genetic_coefficients_psd(
        raw_genetic,
        covariance[:3, :3],
        fitted.component_index,
        annotations_disjoint=True,
    )
    raw_coefficients = np.concatenate([raw_genetic, fitted.residual_coefficients])
    projected_fit = replace(
        fitted,
        raw_coefficients=raw_coefficients,
        genetic_coefficients=raw_genetic,
        raw_omegas=coefficients_to_omegas(raw_genetic, fitted.component_index),
        loo_coefficients=loo,
        jackknife_covariance=covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        psd_projection=projection,
    )
    derived = derive_binary_context_fit(projected_fit, fixture["preset"])
    np.testing.assert_array_equal(derived.raw_omega, [[1.0, 0.8], [0.8, 0.25]])
    assert np.linalg.eigvalsh(derived.raw_omega)[0] < -0.1
    assert derived.raw["tau2_1_given_0"].estimate < 0.0
    assert derived.psd_interpretable is not None
    assert derived.psd_omega is not None
    assert np.linalg.eigvalsh(derived.psd_omega)[0] > -1.0e-9
    assert derived.psd_interpretable["tau2_1_given_0"].estimate > -1.0e-8
    np.testing.assert_array_equal(projected_fit.genetic_coefficients, raw_genetic)


def test_binary_undefined_raw_domains_are_explicitly_indeterminate() -> None:
    fixture, fitted = _synthetic_binary_fit(seed=611)
    invalid_genetic = np.array([-0.2, 0.5, 0.1])
    invalid_loo = np.tile(invalid_genetic, (fitted.loo_coefficients.shape[0], 1))
    invalid_loo[:, 0] += np.linspace(-0.03, 0.03, invalid_loo.shape[0])
    loo = fitted.loo_coefficients.copy()
    loo[:, :3] = invalid_loo
    centered = loo - loo.mean(axis=0, keepdims=True)
    covariance = (loo.shape[0] - 1.0) / loo.shape[0] * centered.T @ centered
    invalid = replace(
        fitted,
        raw_coefficients=np.concatenate(
            [invalid_genetic, fitted.residual_coefficients]
        ),
        genetic_coefficients=invalid_genetic,
        raw_omegas=coefficients_to_omegas(invalid_genetic, fitted.component_index),
        loo_coefficients=loo,
        jackknife_covariance=covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
    )
    derived = derive_binary_context_fit(invalid, fixture["preset"])
    for name in ("rho", "log_sd_ratio", "tau2_1_given_0"):
        assert not np.isfinite(derived.raw[name].estimate)
        assert derived.raw[name].status != "defined"
    boundary = _test_binary_boundary(
        invalid, "rho=1", preset=fixture["preset"], draws=99, seed=17
    )
    assert boundary.status.startswith("indeterminate")
    assert not np.isfinite(boundary.p_value)


@pytest.mark.parametrize("hypothesis", ["rho=1", "tau2=0", "equal_variances"])
def test_binary_multiplier_boundary_tests_are_seeded_and_joint(
    hypothesis: str,
) -> None:
    fixture, fitted = _synthetic_binary_fit(seed=621)
    first = _test_binary_boundary(
        fitted, hypothesis, preset=fixture["preset"], draws=31, seed=29
    )
    second = _test_binary_boundary(
        fitted, hypothesis, preset=fixture["preset"], draws=31, seed=29
    )
    assert isinstance(first, BoundaryTestResult)
    assert first.hypothesis == hypothesis
    assert first.status == "experimental_multiplier_calibration_defined"
    assert 0.0 <= first.p_value <= 1.0
    assert 0.0 <= first.asymptotic_p_value <= 1.0
    assert first.method.startswith("approximate_loo_multiplier_pseudovalue")
    assert first.bootstrap_statistics.shape == (31,)
    np.testing.assert_array_equal(
        first.bootstrap_statistics, second.bootstrap_statistics
    )
    assert first.p_value == second.p_value
    assert first.covariance_rank <= 3
    if hypothesis == "equal_variances":
        contrast = np.array([-1.0, 1.0, 0.0])
        genetic_covariance = fitted.jackknife_covariance[:3, :3]
        expected_statistic = (contrast @ fitted.genetic_coefficients) ** 2 / (
            contrast @ genetic_covariance @ contrast
        )
        assert first.statistic == pytest.approx(expected_statistic, rel=2e-14)


def test_categorical_derivation_binds_counts_basis_and_residual_columns() -> None:
    fixture, fitted = _synthetic_binary_fit(seed=623)
    wrong_counts = build_categorical_preset(
        np.concatenate([np.zeros(1, dtype=int), np.ones(35, dtype=int)]),
        categories=(0, 1),
        source_name="exposure",
        binary=True,
    )
    with pytest.raises(ValueError, match="basis values|counts|context moments"):
        derive_binary_context_fit(fitted, wrong_counts)

    preset = fixture["preset"]
    _, reference, swapped_summary = _build_objects(
        fixture,
        np.asarray(preset.basis),
        np.asarray(preset.residual_basis)[:, ::-1],
        basis_hash=preset.basis_hash,
    )
    swapped_fit = fit_context_model(reference, swapped_summary)
    with pytest.raises(ValueError, match="residual basis"):
        derive_binary_context_fit(swapped_fit, preset)


def test_binary_boundary_requires_one_hot_preset_coordinates() -> None:
    fixture, _, (_, intercept_reference, intercept_summary) = _binary_objects(seed=521)
    intercept_fit = fit_context_model(intercept_reference, intercept_summary)
    with pytest.raises(ValueError, match="basis hash|basis values"):
        _test_binary_boundary(
            intercept_fit,
            "equal_variances",
            preset=fixture["preset"],
            draws=1,
            seed=1,
        )


def test_binary_tau_boundary_uses_global_rank_one_projection() -> None:
    fixture, fitted = _synthetic_binary_fit(seed=625)
    theta = np.array([0.34839532, 3.06479116, 0.01452528])
    covariance = np.array(
        [
            [2.58216173, 3.77571606, 1.04888921],
            [3.77571606, 7.85711859, 1.51896684],
            [1.04888921, 1.51896684, 0.44357368],
        ]
    )

    # Three orthonormal Helmert contrasts generate six centered LOO
    # deviations whose equal-group jackknife covariance is exactly covariance.
    group_count = fitted.loo_coefficients.shape[0]
    helmert = np.zeros((group_count, 3), dtype=np.float64)
    for column in range(3):
        count = column + 1
        helmert[:count, column] = 1.0 / np.sqrt(count * (count + 1.0))
        helmert[count, column] = -count / np.sqrt(count * (count + 1.0))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    deviations = np.sqrt(group_count / (group_count - 1.0)) * helmert @ covariance_root
    loo = fitted.loo_coefficients.copy()
    loo[:, :3] = theta + deviations
    joint_covariance = fitted.jackknife_covariance.copy()
    joint_covariance[:3, :] = 0.0
    joint_covariance[:, :3] = 0.0
    joint_covariance[:3, :3] = covariance
    adversarial = replace(
        fitted,
        raw_coefficients=np.concatenate([theta, fitted.residual_coefficients]),
        genetic_coefficients=theta,
        raw_omegas=coefficients_to_omegas(theta, fitted.component_index),
        loo_coefficients=loo,
        jackknife_covariance=joint_covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(joint_covariance), 0.0)),
    )

    result = _test_binary_boundary(
        adversarial,
        "tau2=0",
        preset=fixture["preset"],
        draws=1,
        seed=1,
    )
    assert result.statistic == pytest.approx(0.0453860577076, rel=2e-9)
    np.testing.assert_allclose(
        result.null_coefficients,
        [0.00606951, 2.5637113, -0.1247416],
        rtol=2e-6,
        atol=2e-8,
    )
    assert result.status == "experimental_multiplier_calibration_defined"


def test_singleton_category_produces_visible_rank_failure() -> None:
    rng = np.random.default_rng(631)
    n, m = 25, 24
    context = np.array(["rare"] + ["a"] * 12 + ["b"] * 12)
    preset = build_categorical_preset(
        context, categories=("a", "b", "rare"), source_name="site"
    )
    np.testing.assert_array_equal(preset.category_counts, [12, 12, 1])
    continuous = rng.normal(size=n)
    fixed = np.column_stack([np.ones(n), preset.fixed_effect_columns, continuous])
    fixture = {
        "genotype": _scaled_genotype(rng, n, m),
        "basis": preset.basis,
        "projector": rank_revealing_projector(fixed),
        "fixed": fixed,
        "phenotype": rng.normal(size=n),
        "annotations": np.ones((m, 1)),
        "loo_groups": tuple(f"group:{index // 4}" for index in range(m)),
        "variant_hash": canonical_sha256({"variants": list(range(m))}),
    }
    _, reference, trait = _build_objects(
        fixture,
        preset.basis,
        preset.residual_basis,
        basis_hash=preset.basis_hash,
    )
    with pytest.raises(ContextRankError, match="rank|identif"):
        fit_context_model(reference, trait)


def test_three_category_permutation_is_full_fit_and_jackknife_equivariant() -> None:
    rng = np.random.default_rng(641)
    n, m = 45, 30
    context = np.tile(np.array(["a", "b", "c"]), n // 3)
    first = build_categorical_preset(
        context, categories=("a", "b", "c"), source_name="site"
    )
    order = ("c", "a", "b")
    second = build_categorical_preset(context, categories=order, source_name="site")
    assert first.category_order_hash != second.category_order_hash
    continuous = np.choose(
        np.searchsorted(np.array(["a", "b", "c"]), context), [0.0, 0.5, 1.0]
    ) + rng.normal(size=n)
    fixed = np.column_stack([np.ones(n), first.fixed_effect_columns, continuous])
    fixture = {
        "genotype": _scaled_genotype(rng, n, m),
        "projector": rank_revealing_projector(fixed),
        "fixed": fixed,
        "phenotype": rng.normal(size=n),
        "annotations": np.ones((m, 1)),
        "loo_groups": tuple(f"group:{index // 5}" for index in range(m)),
        "variant_hash": canonical_sha256({"variants": list(range(m))}),
    }
    components, reference_first, summary_first = _build_objects(
        fixture,
        first.basis,
        first.residual_basis,
        basis_hash=first.basis_hash,
    )
    _, reference_second, summary_second = _build_objects(
        fixture,
        second.basis,
        second.residual_basis,
        basis_hash=second.basis_hash,
    )
    fit_first = fit_context_model(reference_first, summary_first)
    fit_second = fit_context_model(reference_second, summary_second)

    old_index = {label: index for index, label in enumerate(first.category_labels)}
    permutation = np.zeros((3, 3), dtype=np.float64)
    for new_index, label in enumerate(second.category_labels):
        permutation[new_index, old_index[label]] = 1.0
    genetic_map = _coefficient_transform(components, permutation)
    full_map = np.block(
        [
            [genetic_map, np.zeros((6, 3))],
            [np.zeros((3, 6)), permutation],
        ]
    )
    np.testing.assert_allclose(
        fit_second.raw_coefficients,
        full_map @ fit_first.raw_coefficients,
        rtol=5e-11,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        fit_second.loo_coefficients,
        fit_first.loo_coefficients @ full_map.T,
        rtol=8e-11,
        atol=8e-10,
    )
    np.testing.assert_allclose(
        fit_second.jackknife_covariance,
        full_map @ fit_first.jackknife_covariance @ full_map.T,
        rtol=2e-9,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        fit_second.raw_omegas[0],
        permutation @ fit_first.raw_omegas[0] @ permutation.T,
        rtol=5e-11,
        atol=5e-10,
    )

    derived_first = derive_categorical_context_fit(fit_first, first)
    derived_second = derive_categorical_context_fit(fit_second, second)
    np.testing.assert_allclose(
        derived_second.raw_omegas[0],
        permutation @ derived_first.raw_omegas[0] @ permutation.T,
        rtol=5e-11,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        np.asarray(
            [
                derived_second.residual_variances[name].estimate
                for name in second.residual_names
            ]
        ),
        permutation
        @ np.asarray(
            [
                derived_first.residual_variances[name].estimate
                for name in first.residual_names
            ]
        ),
        rtol=5e-11,
        atol=5e-10,
    )
