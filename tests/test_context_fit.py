from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from summit.context import (
    ContextComponentIndex,
    ContextFitResult,
    ContextNormalEquations,
    ContextPairIndex,
    ContextRankError,
    ContextSolveResult,
    PSDProjectionResult,
    array_sha256,
    assemble_context_normal_equations,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    coefficients_to_omegas,
    common_scale_features,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    derive_context_outputs,
    fit_context_model,
    load_context_fit,
    omegas_to_coefficients,
    project_genetic_coefficients_psd,
    project_normalize_phenotype,
    rank_revealing_projector,
    reference_moments_after_deleting_groups,
    solve_context_normal_equations,
    trait_moments_after_deleting_groups,
    transform_omega,
    validate_fit_compatibility,
    write_context_fit,
)


def _fixture(
    *,
    seed: int = 141,
    n: int = 28,
    m: int = 24,
    q: int = 2,
    k: int = 1,
    residual_mode: str = "homoskedastic",
) -> dict[str, object]:
    if m % 6 != 0:
        raise ValueError("Test fixtures require six equal-size LOO groups.")
    rng = np.random.default_rng(seed)
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    environment = rng.normal(size=n)
    basis = np.ones((n, q), dtype=np.float64)
    if q > 1:
        basis[:, 1] = environment
    if q > 2:
        basis[:, 2:] = rng.normal(size=(n, q - 2))
    fixed = np.column_stack([np.ones(n), basis[:, 1:], rng.normal(size=n)])
    projector = rank_revealing_projector(fixed)
    phenotype = rng.normal(size=n)
    if k == 1:
        annotations = np.ones((m, 1), dtype=np.float64)
        annotation_names = ("all",)
    elif k == 2:
        annotations = np.zeros((m, 2), dtype=np.float64)
        annotations[np.arange(m) % 2 == 0, 0] = 1.0
        annotations[np.arange(m) % 2 == 1, 1] = 1.0
        annotation_names = ("even", "odd")
    else:
        raise ValueError("Test fixture supports K=1 or K=2.")
    if residual_mode == "heteroskedastic":
        residual_basis = np.column_stack([np.ones(n), environment * environment + 0.2])
        residual_names = ("residual:constant", "residual:environment_squared")
    else:
        residual_basis = np.ones((n, 1), dtype=np.float64)
        residual_names = ("residual:constant",)
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q))
    variants = canonical_sha256({"variants": list(range(m))})
    loo_groups = tuple(f"group:{index // (m // 6)}" for index in range(m))
    return {
        "genotype": genotype,
        "basis": basis,
        "fixed": fixed,
        "projector": projector,
        "phenotype": phenotype,
        "annotations": annotations,
        "components": components,
        "residual_basis": residual_basis,
        "residual_names": residual_names,
        "variant_hash": variants,
        "loo_groups": loo_groups,
    }


def _build_objects(fixture: dict[str, object]):
    common = {
        "genotype": fixture["genotype"],
        "basis": fixture["basis"],
        "projector": fixture["projector"],
        "annotations": fixture["annotations"],
        "component_index": fixture["components"],
        "loo_groups": fixture["loo_groups"],
        "basis_hash": array_sha256(fixture["basis"]),
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
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        block_size=7,
    )
    return reference, summary


def _dense_equations(fixture: dict[str, object]):
    y = project_normalize_phenotype(fixture["phenotype"], fixture["projector"])
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], fixture["projector"].projector
    )
    genetic = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    residual = dense_residual_kernels(
        fixture["projector"].projector, fixture["residual_basis"]
    )
    return dense_normal_equations(
        genetic,
        residual,
        y,
        fixture["components"].names,
        fixture["residual_names"],
    )


def _balanced_groups(fixture: dict[str, object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in fixture["loo_groups"]))


def _genetic_coefficient_transform(
    components: ContextComponentIndex, basis_transform: np.ndarray
) -> np.ndarray:
    p_count = len(components)
    result = np.empty((p_count, p_count), dtype=np.float64)
    for column in range(p_count):
        unit = np.zeros(p_count, dtype=np.float64)
        unit[column] = 1.0
        omegas = coefficients_to_omegas(unit, components)
        transformed = np.asarray(
            [transform_omega(omega, basis_transform) for omega in omegas]
        )
        result[:, column] = omegas_to_coefficients(transformed, components)
    return result


def test_matched_exact_summary_assembly_and_coefficients_equal_dense_oracle() -> None:
    fixture = _fixture(seed=141)
    reference, summary = _build_objects(fixture)
    equations = assemble_context_normal_equations(reference, summary)
    assert isinstance(equations, ContextNormalEquations)
    dense = _dense_equations(fixture)
    np.testing.assert_allclose(equations.matrix, dense.matrix, rtol=5e-14, atol=5e-12)
    np.testing.assert_allclose(equations.rhs, dense.rhs, rtol=5e-14, atol=5e-12)
    np.testing.assert_allclose(equations.traces, dense.traces, rtol=5e-14, atol=5e-12)
    assert equations.component_names == dense.component_names
    assert equations.genetic_count == len(fixture["components"])

    solved = solve_context_normal_equations(equations)
    assert isinstance(solved, ContextSolveResult)
    expected = np.linalg.solve(dense.matrix, dense.rhs)
    np.testing.assert_allclose(solved.coefficients, expected, rtol=2e-12, atol=2e-11)
    assert solved.rank == expected.size
    assert solved.relative_residual < 1.0e-11

    fitted = fit_context_model(reference, summary)
    assert isinstance(fitted, ContextFitResult)
    np.testing.assert_allclose(
        fitted.raw_coefficients, expected, rtol=2e-12, atol=2e-11
    )
    np.testing.assert_array_equal(
        fitted.genetic_coefficients, fitted.raw_coefficients[: equations.genetic_count]
    )
    np.testing.assert_array_equal(
        fitted.residual_coefficients, fitted.raw_coefficients[equations.genetic_count :]
    )


def test_deleted_assembly_reconstructs_reference_and_trait_group_numerators() -> None:
    fixture = _fixture(seed=171, residual_mode="heteroskedastic")
    reference, summary = _build_objects(fixture)
    deleted_groups = ("group:1", "group:4")
    observed = assemble_context_normal_equations(
        reference, summary, deleted_groups=deleted_groups
    )
    deleted_reference = reference_moments_after_deleting_groups(
        reference, deleted_groups
    )
    deleted_trait = trait_moments_after_deleting_groups(summary, deleted_groups)
    p_count = len(fixture["components"])
    expected_matrix = np.block(
        [
            [deleted_reference.gram, deleted_trait.genetic_residual],
            [deleted_trait.genetic_residual.T, deleted_trait.residual_gram],
        ]
    )
    expected_rhs = np.concatenate(
        [deleted_trait.genetic_rhs, deleted_trait.residual_rhs]
    )
    expected_traces = np.concatenate(
        [deleted_trait.genetic_traces, deleted_trait.residual_traces]
    )
    np.testing.assert_allclose(observed.matrix, expected_matrix, rtol=5e-14, atol=5e-12)
    np.testing.assert_allclose(observed.rhs, expected_rhs, rtol=5e-14, atol=5e-12)
    np.testing.assert_allclose(observed.traces, expected_traces, rtol=5e-14, atol=5e-12)
    assert observed.genetic_count == p_count
    np.testing.assert_array_equal(deleted_reference.same_person, reference.same_person)


def test_joint_equal_group_jackknife_matches_replicates_and_covariance_formula() -> (
    None
):
    fixture = _fixture(seed=191)
    reference, summary = _build_objects(fixture)
    groups = _balanced_groups(fixture)
    fitted = fit_context_model(reference, summary, loo_groups=groups)
    independently_solved = np.asarray(
        [
            solve_context_normal_equations(
                assemble_context_normal_equations(
                    reference, summary, deleted_groups=(group,)
                )
            ).coefficients
            for group in groups
        ]
    )
    np.testing.assert_allclose(
        fitted.loo_coefficients, independently_solved, rtol=3e-12, atol=3e-11
    )
    centered = independently_solved - np.mean(independently_solved, axis=0)
    expected_covariance = (len(groups) - 1.0) / len(groups) * centered.T @ centered
    np.testing.assert_allclose(
        fitted.jackknife_covariance,
        expected_covariance,
        rtol=4e-13,
        atol=4e-12,
    )
    np.testing.assert_allclose(
        fitted.jackknife_covariance,
        fitted.jackknife_covariance.T,
        rtol=0.0,
        atol=0.0,
    )
    assert np.linalg.eigvalsh(fitted.jackknife_covariance)[0] > -1.0e-10
    off_diagonal = fitted.jackknife_covariance - np.diag(
        np.diag(fitted.jackknife_covariance)
    )
    assert np.max(np.abs(off_diagonal)) > 1.0e-8


def test_structural_rank_failure_reports_null_directions_without_hidden_ridge() -> None:
    fixture = _fixture(seed=211)
    duplicated = np.ones_like(fixture["basis"])
    fixture["basis"] = duplicated
    fixture["fixed"] = np.column_stack(
        [np.ones(duplicated.shape[0]), np.arange(duplicated.shape[0])]
    )
    fixture["projector"] = rank_revealing_projector(fixture["fixed"])
    reference, summary = _build_objects(fixture)
    equations = assemble_context_normal_equations(reference, summary)
    with pytest.raises(ContextRankError, match="rank|identif"):
        solve_context_normal_equations(equations, require_full_rank=True)
    reduced = solve_context_normal_equations(equations, require_full_rank=False)
    assert reduced.rank < equations.matrix.shape[0]
    assert reduced.null_space.shape == (
        equations.matrix.shape[0],
        equations.matrix.shape[0] - reduced.rank,
    )
    np.testing.assert_allclose(
        equations.matrix @ reduced.null_space, 0.0, rtol=0.0, atol=2e-10
    )


@pytest.mark.parametrize(
    "field",
    [
        "basis_hash",
        "variant_hash",
        "annotation_hash",
        "component_index_hash",
        "loo_grouping_hash",
    ],
)
def test_fit_compatibility_rejects_every_manifest_identity_mismatch(field: str) -> None:
    fixture = _fixture(seed=231)
    reference, summary = _build_objects(fixture)
    validate_fit_compatibility(reference, summary)
    manifest = dict(summary.manifest)
    manifest[field] = "0" * 64
    incompatible = replace(summary, manifest=manifest)
    with pytest.raises(ValueError, match="mismatch"):
        validate_fit_compatibility(reference, incompatible)
    with pytest.raises(ValueError, match="mismatch"):
        assemble_context_normal_equations(reference, incompatible)


def test_raw_indefinite_fit_is_preserved_when_psd_interpretation_is_requested() -> None:
    fixture = _fixture(seed=101)
    reference, summary = _build_objects(fixture)
    groups = _balanced_groups(fixture)
    raw = fit_context_model(reference, summary, loo_groups=groups)
    projected = fit_context_model(
        reference,
        summary,
        loo_groups=groups,
        project_psd=True,
        annotations_disjoint=True,
    )
    np.testing.assert_array_equal(projected.raw_coefficients, raw.raw_coefficients)
    raw_omega = coefficients_to_omegas(
        projected.genetic_coefficients, fixture["components"]
    )[0]
    assert np.linalg.eigvalsh(raw_omega)[0] < -1.0e-3
    assert isinstance(projected.psd_projection, PSDProjectionResult)
    assert np.linalg.eigvalsh(projected.psd_projection.projected_omegas[0])[0] > -1.0e-8
    assert projected.psd_projection.distance > 0.0


@pytest.mark.parametrize("q", [2, 3])
def test_covariance_aware_psd_projection_handles_2x2_and_3x3_diagonal_cases(
    q: int,
) -> None:
    components = ContextComponentIndex(("all",), ContextPairIndex(q))
    diagonal = np.linspace(1.0, 0.4, q)
    diagonal[-1] = -0.3
    omega = np.diag(diagonal)
    theta = omegas_to_coefficients(omega[None, :, :], components)
    covariance = np.eye(theta.size)
    theta_before = theta.copy()
    result = project_genetic_coefficients_psd(
        theta, covariance, components, annotations_disjoint=True
    )
    assert isinstance(result, PSDProjectionResult)
    np.testing.assert_array_equal(theta, theta_before)
    expected = np.diag(np.maximum(diagonal, 0.0))
    np.testing.assert_allclose(
        result.projected_omegas[0], expected, rtol=0.0, atol=2.0e-7
    )
    assert np.linalg.eigvalsh(result.projected_omegas[0])[0] > -1.0e-8
    assert result.distance > 0.0
    with pytest.raises(ValueError, match="overlap|disjoint"):
        project_genetic_coefficients_psd(
            theta, covariance, components, annotations_disjoint=False
        )


def test_psd_projection_singular_covariance_uses_declared_tie_break() -> None:
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    theta = np.array([1.0, 1.0, 2.0])
    covariance = np.diag([1.0, 1.0, 0.0])
    result = project_genetic_coefficients_psd(theta, covariance, components)
    assert result.covariance_rank == 2
    assert result.covariance_nullity == 1
    assert result.tie_break_applied is True
    assert result.distance < 2.0e-6
    assert result.euclidean_distance == pytest.approx(1.0, abs=2.0e-6)
    assert np.linalg.eigvalsh(result.projected_omegas[0])[0] > -1.0e-10
    np.testing.assert_array_equal(theta, [1.0, 1.0, 2.0])


def test_psd_projection_rank_and_solution_are_scale_equivariant() -> None:
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    theta = np.array([1.0, 0.25, 0.8])
    covariance = np.array(
        [
            [0.006166666666666667, -0.00125, 0.001166666666666667],
            [-0.00125, 0.002, -0.0003333333333333333],
            [0.001166666666666667, -0.0003333333333333333, 0.0006666666666666666],
        ]
    )
    baseline = project_genetic_coefficients_psd(theta, covariance, components)
    scale = 1.0e-6
    rescaled = project_genetic_coefficients_psd(
        scale * theta, scale * scale * covariance, components
    )
    assert baseline.covariance_rank == rescaled.covariance_rank == 3
    assert baseline.tie_break_applied is False
    assert rescaled.tie_break_applied is False
    np.testing.assert_allclose(
        rescaled.projected_coefficients / scale,
        baseline.projected_coefficients,
        rtol=2.0e-6,
        atol=5.0e-7,
    )


def test_basis_equivariance_includes_loo_covariance_and_context_surfaces() -> None:
    fixture = _fixture(seed=271)
    reference, summary = _build_objects(fixture)
    transform = np.array([[1.0, 0.35], [-0.2, 1.15]])
    transformed_fixture = dict(fixture)
    transformed_fixture["basis"] = np.asarray(fixture["basis"]) @ transform.T
    transformed_reference, transformed_summary = _build_objects(transformed_fixture)
    groups = _balanced_groups(fixture)
    grid = np.array([[1.0, -1.0], [1.0, -0.25], [1.0, 0.75]])
    metric = (
        np.asarray(fixture["basis"]).T
        @ np.asarray(fixture["basis"])
        / np.asarray(fixture["basis"]).shape[0]
    )
    transformed_grid = grid @ transform.T
    transformed_metric = transform @ metric @ transform.T
    fitted = fit_context_model(
        reference,
        summary,
        loo_groups=groups,
        context_grid=grid,
        basis_metric=metric,
    )
    transformed_fitted = fit_context_model(
        transformed_reference,
        transformed_summary,
        loo_groups=groups,
        context_grid=transformed_grid,
        basis_metric=transformed_metric,
    )
    genetic_transform = _genetic_coefficient_transform(fixture["components"], transform)
    full_transform = np.block(
        [
            [genetic_transform, np.zeros((genetic_transform.shape[0], 1))],
            [np.zeros((1, genetic_transform.shape[1])), np.ones((1, 1))],
        ]
    )
    np.testing.assert_allclose(
        transformed_fitted.raw_coefficients,
        full_transform @ fitted.raw_coefficients,
        rtol=3e-11,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        transformed_fitted.loo_coefficients,
        fitted.loo_coefficients @ full_transform.T,
        rtol=5e-11,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        transformed_fitted.jackknife_covariance,
        full_transform @ fitted.jackknife_covariance @ full_transform.T,
        rtol=8e-10,
        atol=8e-10,
    )
    np.testing.assert_allclose(
        transformed_fitted.context_outputs["annotations"][0]["covariance_surface"],
        fitted.context_outputs["annotations"][0]["covariance_surface"],
        rtol=5e-11,
        atol=5e-10,
    )


def test_derived_outputs_distinguish_rank_one_amplification_and_rank_two() -> None:
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    metric = np.eye(2)
    loading = np.array([1.0, 0.5])
    rank_one_omega = np.outer(loading, loading)
    rank_one = derive_context_outputs(
        omegas_to_coefficients(rank_one_omega[None, :, :], components),
        components,
        grid,
        metric,
    )
    expected_surface = grid @ rank_one_omega @ grid.T
    rank_one_annotation = rank_one["annotations"][0]
    np.testing.assert_allclose(
        rank_one_annotation["covariance_surface"],
        expected_surface,
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        rank_one_annotation["variances"], np.diag(expected_surface)
    )
    np.testing.assert_allclose(
        rank_one_annotation["orthogonal_heterogeneity"],
        0.0,
        rtol=0.0,
        atol=2e-14,
    )
    assert rank_one_annotation["spectral_rank_one_fraction"] == pytest.approx(
        1.0, abs=2e-15
    )

    rank_two_omega = np.diag([1.0, 0.5])
    rank_two = derive_context_outputs(
        omegas_to_coefficients(rank_two_omega[None, :, :], components),
        components,
        grid,
        metric,
    )
    rank_two_annotation = rank_two["annotations"][0]
    np.testing.assert_allclose(
        np.sort(rank_two_annotation["basis_metric_operator_eigenvalues"]),
        [0.5, 1.0],
        atol=2e-15,
    )
    assert rank_two_annotation["spectral_rank_one_fraction"] == pytest.approx(2.0 / 3.0)
    assert np.max(rank_two_annotation["orthogonal_heterogeneity"]) > 0.1


def test_residual_only_heteroskedastic_system_recovers_zero_genetic_coefficients() -> (
    None
):
    fixture = _fixture(seed=301, residual_mode="heteroskedastic")
    reference, summary = _build_objects(fixture)
    residual_truth = np.array([0.35, 0.8])
    exact_summary = replace(
        summary,
        genetic_rhs=summary.genetic_residual @ residual_truth,
        residual_rhs=summary.residual_gram @ residual_truth,
    )
    fitted = fit_context_model(reference, exact_summary)
    np.testing.assert_allclose(fitted.genetic_coefficients, 0.0, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(
        fitted.residual_coefficients, residual_truth, rtol=2e-11, atol=2e-11
    )


def test_two_disjoint_annotations_recover_distinct_full_context_covariances() -> None:
    fixture = _fixture(seed=331, n=34, m=30, k=2)
    reference, summary = _build_objects(fixture)
    genetic_truth = omegas_to_coefficients(
        np.array(
            [
                [[0.8, 0.15], [0.15, 0.35]],
                [[0.3, -0.08], [-0.08, 0.65]],
            ]
        ),
        fixture["components"],
    )
    residual_truth = np.array([0.55])
    exact_summary = replace(
        summary,
        genetic_rhs=reference.gram @ genetic_truth
        + summary.genetic_residual @ residual_truth,
        residual_rhs=summary.genetic_residual.T @ genetic_truth
        + summary.residual_gram @ residual_truth,
    )
    fitted = fit_context_model(reference, exact_summary)
    np.testing.assert_allclose(
        fitted.genetic_coefficients, genetic_truth, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        fitted.residual_coefficients, residual_truth, rtol=2e-11, atol=2e-11
    )
    recovered = coefficients_to_omegas(
        fitted.genetic_coefficients, fixture["components"]
    )
    assert not np.allclose(recovered[0], recovered[1])


def test_context_fit_round_trip_is_hash_bound_and_tamper_evident(
    tmp_path: Path,
) -> None:
    fixture = _fixture(seed=351)
    reference, summary = _build_objects(fixture)
    grid = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    fitted = fit_context_model(
        reference,
        summary,
        loo_groups=_balanced_groups(fixture),
        context_grid=grid,
        basis_metric=np.eye(2),
    )
    manifest, arrays = write_context_fit(fitted, tmp_path / "fit")
    loaded = load_context_fit(
        manifest,
        expected={"component_index_hash": fixture["components"].digest},
    )
    assert isinstance(loaded, ContextFitResult)
    np.testing.assert_array_equal(loaded.raw_coefficients, fitted.raw_coefficients)
    np.testing.assert_array_equal(loaded.loo_coefficients, fitted.loo_coefficients)
    np.testing.assert_array_equal(
        loaded.jackknife_covariance, fitted.jackknife_covariance
    )
    np.testing.assert_array_equal(
        loaded.context_outputs["annotations"][0]["covariance_surface"],
        fitted.context_outputs["annotations"][0]["covariance_surface"],
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_context_fit(
            manifest,
            expected={"component_index_hash": canonical_sha256({"wrong": True})},
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_context_fit(manifest)
    assert arrays.stat().st_size > 0


def test_context_fit_round_trip_restores_nan_arrays(tmp_path: Path) -> None:
    fixture = _fixture(seed=353)
    reference, summary = _build_objects(fixture)
    fitted = fit_context_model(reference, summary)
    undefined_surface = np.array([[np.nan, np.nan], [np.nan, 1.0]])
    defined_mask = np.array([[False, False], [False, True]])
    fitted = replace(
        fitted,
        context_outputs={
            "annotations": [
                {
                    "correlation_surface": undefined_surface,
                    "correlation_defined": defined_mask,
                    "rank_one_fraction": float("nan"),
                }
            ]
        },
    )
    manifest, _ = write_context_fit(fitted, tmp_path / "undefined-fit")
    loaded = load_context_fit(manifest)
    restored = loaded.context_outputs["annotations"][0]["correlation_surface"]
    assert isinstance(restored, np.ndarray)
    np.testing.assert_allclose(restored, undefined_surface, equal_nan=True)
    restored_mask = loaded.context_outputs["annotations"][0]["correlation_defined"]
    assert restored_mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(restored_mask, defined_mask)
    assert np.isnan(loaded.context_outputs["annotations"][0]["rank_one_fraction"])
