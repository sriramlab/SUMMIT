from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from typing import Any

import numpy as np
import pytest

from summit.context import (
    ContextRankError,
    DirectionCrossFitArmContractions,
    array_sha256,
    build_balanced_direction_folds,
    build_context_direction_crossfit_contractions,
    build_direction_reference_contractions,
    build_direction_trait_contractions,
    canonical_sha256,
    combine_context_direction_contractions,
    crossfit_context_direction,
    direction_pair_weights,
    evaluate_context_direction,
    exact_same_person_matrix,
    normalize_context_direction,
    optimize_context_direction,
    project_normalize_phenotype,
    rank_revealing_projector,
    validate_direction_crossfit_contractions,
)


def _standardize_genotype(genotype: np.ndarray) -> np.ndarray:
    result = np.asarray(genotype, dtype=np.float64).copy()
    result -= np.mean(result, axis=0, keepdims=True)
    result /= np.std(result, axis=0, ddof=1, keepdims=True)
    return result


def _fixture(
    *,
    l_count: int,
    seed: int,
    n_reference: int = 31,
    n_study: int | None = None,
    n_variants: int = 24,
) -> dict[str, Any]:
    if n_study is None:
        n_study = n_reference
    rng = np.random.default_rng(seed)

    def environments(n_samples: int) -> np.ndarray:
        latent = rng.normal(size=(n_samples, l_count))
        mixing = np.eye(l_count)
        for row in range(1, l_count):
            mixing[row, :row] = np.linspace(0.15, 0.35, row)
        result = latent @ mixing.T
        result -= np.mean(result, axis=0, keepdims=True)
        return result

    reference_environments = environments(n_reference)
    study_environments = (
        reference_environments.copy()
        if n_study == n_reference
        else environments(n_study)
    )
    reference_genotype = _standardize_genotype(
        rng.normal(size=(n_reference, n_variants))
    )
    study_genotype = (
        reference_genotype.copy()
        if n_study == n_reference
        else _standardize_genotype(rng.normal(size=(n_study, n_variants)))
    )
    reference_pc = rng.normal(size=n_reference)
    study_pc = (
        reference_pc.copy() if n_study == n_reference else rng.normal(size=n_study)
    )
    reference_projector = rank_revealing_projector(
        np.column_stack([np.ones(n_reference), reference_environments, reference_pc])
    )
    study_projector = (
        reference_projector
        if n_study == n_reference
        else rank_revealing_projector(
            np.column_stack([np.ones(n_study), study_environments, study_pc])
        )
    )
    context_metric = reference_environments.T @ reference_environments / n_reference
    true_direction = np.linspace(0.9, -0.45, l_count)
    true_direction = true_direction / np.sqrt(
        true_direction @ context_metric @ true_direction
    )
    additive_effect = rng.normal(size=n_variants)
    interaction_effect = rng.normal(size=n_variants)
    environment = study_environments @ true_direction
    phenotype = (
        0.25 * study_genotype @ additive_effect / np.sqrt(n_variants)
        + 0.8
        * environment
        * (study_genotype @ interaction_effect)
        / np.sqrt(n_variants)
        + 0.35 * rng.normal(size=n_study)
    )
    variant_weights = np.linspace(0.55, 1.45, n_variants)
    environment_names = tuple(f"environment_{index}" for index in range(l_count))
    variant_hash = canonical_sha256(
        {"variant_ids": [f"variant_{index}" for index in range(n_variants)]}
    )
    return {
        "reference_genotype": reference_genotype,
        "study_genotype": study_genotype,
        "reference_environments": reference_environments,
        "study_environments": study_environments,
        "reference_projector": reference_projector,
        "study_projector": study_projector,
        "context_metric": context_metric,
        "phenotype": phenotype,
        "variant_weights": variant_weights,
        "environment_names": environment_names,
        "variant_hash": variant_hash,
        "true_direction": true_direction,
    }


def _build_contractions(
    fixture: dict[str, Any], *, variant_mask: np.ndarray | None = None
):
    shared = {
        "context_metric": fixture["context_metric"],
        "environment_names": fixture["environment_names"],
        "variant_hash": fixture["variant_hash"],
        "genotype_scaling": "pre_scaled_input",
        "variant_weights": fixture["variant_weights"],
        "variant_mask": variant_mask,
    }
    reference = build_direction_reference_contractions(
        fixture["reference_genotype"],
        fixture["reference_environments"],
        fixture["reference_projector"],
        **shared,
    )
    trait = build_direction_trait_contractions(
        fixture["study_genotype"],
        fixture["study_environments"],
        fixture["study_projector"],
        fixture["phenotype"],
        **shared,
    )
    return reference, trait, combine_context_direction_contractions(reference, trait)


def _build_crossfit(fixture: dict[str, Any], block_ids: np.ndarray):
    return build_context_direction_crossfit_contractions(
        fixture["reference_genotype"],
        fixture["reference_environments"],
        fixture["reference_projector"],
        fixture["study_genotype"],
        fixture["study_environments"],
        fixture["study_projector"],
        fixture["phenotype"],
        block_ids,
        context_metric=fixture["context_metric"],
        environment_names=fixture["environment_names"],
        variant_hash=fixture["variant_hash"],
        context_spec_hash=canonical_sha256(
            {
                "environment_names": list(fixture["environment_names"]),
                "nuisance": "intercept_context_main_effects_pc",
            }
        ),
        genotype_scaling="pre_scaled_input",
        variant_weights=fixture["variant_weights"],
    )


def _assert_compact_contractions_equal(left, right) -> None:
    assert left.environment_names == right.environment_names
    assert left.pair_index.digest == right.pair_index.digest
    np.testing.assert_array_equal(left.context_metric, right.context_metric)
    np.testing.assert_array_equal(left.reference.base_gram, right.reference.base_gram)
    np.testing.assert_array_equal(
        left.reference.same_person, right.reference.same_person
    )
    np.testing.assert_array_equal(
        left.reference.base_traces, right.reference.base_traces
    )
    np.testing.assert_array_equal(left.summary.base_gram, right.summary.base_gram)
    np.testing.assert_array_equal(left.summary.base_rhs, right.summary.base_rhs)
    np.testing.assert_array_equal(left.summary.base_traces, right.summary.base_traces)
    assert left.reference.annotation_mass == right.reference.annotation_mass
    assert left.summary.annotation_mass == right.summary.annotation_mass
    assert left.reference.selected_variant_hash == right.reference.selected_variant_hash
    assert left.summary.selected_variant_hash == right.summary.selected_variant_hash


def _canonical_direction(raw: np.ndarray, metric: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    value = value / np.sqrt(value @ metric @ value)
    pivot = int(np.argmax(np.abs(value)))
    return value if value[pivot] >= 0.0 else -value


def _dense_kernels(
    fixture: dict[str, Any],
    direction: np.ndarray,
    *,
    reference: bool,
    variant_mask: np.ndarray | None = None,
) -> np.ndarray:
    prefix = "reference" if reference else "study"
    genotype = fixture[f"{prefix}_genotype"]
    environments = fixture[f"{prefix}_environments"]
    projector = fixture[f"{prefix}_projector"].projector
    weights = fixture["variant_weights"]
    if variant_mask is not None:
        genotype = genotype[:, variant_mask]
        weights = weights[variant_mask]
    mass = float(np.sum(weights))
    environment = environments @ direction
    additive_features = projector @ genotype
    interaction_features = projector @ (environment[:, None] * genotype)
    additive = (additive_features * weights[None, :]) @ additive_features.T / mass
    interaction = (
        (interaction_features * weights[None, :]) @ interaction_features.T / mass
    )
    residual = projector
    residual_interaction = projector @ (environment[:, None] ** 2 * projector)
    return np.stack([additive, interaction, residual, residual_interaction])


def _dense_base_kernels(fixture: dict[str, Any], *, reference: bool) -> np.ndarray:
    prefix = "reference" if reference else "study"
    genotype = fixture[f"{prefix}_genotype"]
    context = fixture[f"{prefix}_environments"]
    projector = fixture[f"{prefix}_projector"].projector
    weights = fixture["variant_weights"]
    mass = float(np.sum(weights))
    additive_features = projector @ genotype
    additive = (additive_features * weights[None, :]) @ additive_features.T / mass
    features = np.stack(
        [
            projector @ (context[:, index, None] * genotype)
            for index in range(context.shape[1])
        ]
    )
    pairs = [(index, index) for index in range(context.shape[1])]
    pairs.extend(
        (left, right)
        for left in range(context.shape[1])
        for right in range(left + 1, context.shape[1])
    )
    interaction_kernels: list[np.ndarray] = []
    residual_kernels: list[np.ndarray] = []
    for left, right in pairs:
        if left == right:
            interaction = (features[left] * weights[None, :]) @ features[right].T / mass
            residual_product = context[:, left] * context[:, right]
        else:
            interaction = (
                (features[left] * weights[None, :]) @ features[right].T
                + (features[right] * weights[None, :]) @ features[left].T
            ) / mass
            residual_product = 2.0 * context[:, left] * context[:, right]
        interaction_kernels.append(interaction)
        residual_kernels.append(projector @ (residual_product[:, None] * projector))
    genetic = np.stack([additive, *interaction_kernels])
    if reference:
        return genetic
    return np.concatenate(
        [
            genetic,
            projector[None],
            np.stack(residual_kernels),
        ]
    )


def _dense_evaluation(
    fixture: dict[str, Any],
    direction: np.ndarray,
    *,
    variant_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    reference_kernels = _dense_kernels(
        fixture, direction, reference=True, variant_mask=variant_mask
    )
    study_kernels = _dense_kernels(
        fixture, direction, reference=False, variant_mask=variant_mask
    )
    if reference_kernels.shape[1] != study_kernels.shape[1]:
        raise ValueError("Dense exact oracle requires equal reference/study N.")
    phenotype = project_normalize_phenotype(
        fixture["phenotype"], fixture["study_projector"]
    )
    matrix = np.einsum("aij,bij->ab", study_kernels, study_kernels, optimize=True)
    rhs = np.einsum("i,aij,j->a", phenotype, study_kernels, phenotype)
    traces = np.trace(study_kernels, axis1=1, axis2=2)
    coefficients = np.linalg.solve(matrix, rhs)
    nuisance = np.asarray([0, 2, 3])
    gain = float(
        rhs @ coefficients
        - rhs[nuisance]
        @ np.linalg.solve(matrix[np.ix_(nuisance, nuisance)], rhs[nuisance])
    )
    return {
        "reference_kernels": reference_kernels,
        "study_kernels": study_kernels,
        "matrix": matrix,
        "rhs": rhs,
        "traces": traces,
        "coefficients": coefficients,
        "interaction_coefficient": float(coefficients[1]),
        "interaction_trace_contribution": float(
            coefficients[1] * traces[1] / fixture["study_projector"].residual_rank
        ),
        "he_moment_gain": gain,
    }


def _metric_sphere_grid(metric: np.ndarray, count: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    dimension = metric.shape[0]
    if dimension == 2:
        angles = np.linspace(0.0, np.pi, count, endpoint=False)
        unit = np.column_stack([np.cos(angles), np.sin(angles)])
    elif dimension == 3:
        indices = np.arange(count, dtype=np.float64) + 0.5
        z = 1.0 - 2.0 * indices / count
        radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
        unit = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])
    else:
        raise AssertionError("Grid helper is deliberately limited to L=2,3.")
    return unit @ inverse_sqrt


def _compact_array_shapes(value: Any) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            result.append(item.shape)
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return result


def _isotropic_full_rank_contractions(fixture: dict[str, Any]):
    reference, summary, _ = _build_contractions(fixture)
    metric = np.asarray(fixture["context_metric"])
    metric_pair = np.asarray([metric[0, 0], metric[1, 1], 2.0 * metric[0, 1]])
    pair_outer = np.outer(metric_pair, metric_pair)
    reduced_gram = np.asarray(
        [
            [2.0, 0.18, 0.11, -0.04],
            [0.18, 2.6, 0.09, 0.16],
            [0.11, 0.09, 1.7, -0.08],
            [-0.04, 0.16, -0.08, 2.2],
        ]
    )
    assert np.min(np.linalg.eigvalsh(reduced_gram)) > 1.0
    scalar_or_pair = (
        np.asarray([0]),
        np.arange(1, 4),
        np.asarray([4]),
        np.arange(5, 8),
    )
    trait_gram = np.empty((8, 8), dtype=np.float64)
    for left, left_indices in enumerate(scalar_or_pair):
        for right, right_indices in enumerate(scalar_or_pair):
            if left_indices.size == 1 and right_indices.size == 1:
                block = np.asarray([[reduced_gram[left, right]]])
            elif left_indices.size == 1:
                block = reduced_gram[left, right] * metric_pair[None, :]
            elif right_indices.size == 1:
                block = reduced_gram[left, right] * metric_pair[:, None]
            else:
                block = reduced_gram[left, right] * pair_outer
            trait_gram[np.ix_(left_indices, right_indices)] = block
    reduced_rhs = np.asarray([0.4, 0.7, -0.2, 0.3])
    reduced_traces = np.asarray([8.0, 6.0, 12.0, 4.0])
    trait_rhs = np.concatenate(
        [
            reduced_rhs[:1],
            reduced_rhs[1] * metric_pair,
            reduced_rhs[2:3],
            reduced_rhs[3] * metric_pair,
        ]
    )
    trait_traces = np.concatenate(
        [
            reduced_traces[:1],
            reduced_traces[1] * metric_pair,
            reduced_traces[2:3],
            reduced_traces[3] * metric_pair,
        ]
    )
    reference_gram = np.empty((4, 4), dtype=np.float64)
    reference_gram[0, 0] = reduced_gram[0, 0]
    reference_gram[0, 1:] = reduced_gram[0, 1] * metric_pair
    reference_gram[1:, 0] = reduced_gram[1, 0] * metric_pair
    reference_gram[1:, 1:] = reduced_gram[1, 1] * pair_outer
    reference_traces = np.concatenate(
        [reduced_traces[:1], reduced_traces[1] * metric_pair]
    )

    reference_manifest = copy.deepcopy(reference.manifest)
    reference_manifest["array_hashes"].update(
        {
            "base_gram": array_sha256(reference_gram),
            "same_person": array_sha256(reference_gram),
            "base_traces": array_sha256(reference_traces),
        }
    )
    summary_manifest = copy.deepcopy(summary.manifest)
    summary_manifest["array_hashes"].update(
        {
            "base_gram": array_sha256(trait_gram),
            "base_rhs": array_sha256(trait_rhs),
            "base_traces": array_sha256(trait_traces),
        }
    )
    synthetic_reference = replace(
        reference,
        manifest=reference_manifest,
        base_gram=reference_gram,
        same_person=reference_gram.copy(),
        base_traces=reference_traces,
    )
    synthetic_summary = replace(
        summary,
        manifest=summary_manifest,
        base_gram=trait_gram,
        base_rhs=trait_rhs,
        base_traces=trait_traces,
    )
    return combine_context_direction_contractions(
        synthetic_reference, synthetic_summary
    )


def test_pair_weights_use_diagonal_first_order_and_no_extra_offdiagonal_factor() -> (
    None
):
    direction = np.asarray([0.8, -0.35, 0.45])
    observed = direction_pair_weights(direction)
    expected = np.asarray(
        [
            direction[0] ** 2,
            direction[1] ** 2,
            direction[2] ** 2,
            direction[0] * direction[1],
            direction[0] * direction[2],
            direction[1] * direction[2],
        ]
    )
    np.testing.assert_array_equal(observed, expected)
    assert observed[3] < 0.0


@pytest.mark.parametrize("l_count", [2, 3])
def test_every_stored_base_pair_moment_matches_explicit_dense_kernels(
    l_count: int,
) -> None:
    fixture = _fixture(l_count=l_count, seed=7190 + l_count)
    reference, summary, _ = _build_contractions(fixture)
    reference_kernels = _dense_base_kernels(fixture, reference=True)
    study_kernels = _dense_base_kernels(fixture, reference=False)
    phenotype = project_normalize_phenotype(
        fixture["phenotype"], fixture["study_projector"]
    )
    np.testing.assert_allclose(
        reference.base_gram,
        np.einsum("aij,bij->ab", reference_kernels, reference_kernels),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        reference.same_person,
        exact_same_person_matrix(reference_kernels),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        reference.base_traces,
        np.trace(reference_kernels, axis1=1, axis2=2),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        summary.base_gram,
        np.einsum("aij,bij->ab", study_kernels, study_kernels),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        summary.base_rhs,
        np.einsum("i,aij,j->a", phenotype, study_kernels, phenotype),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        summary.base_traces,
        np.trace(study_kernels, axis1=1, axis2=2),
        rtol=2e-11,
        atol=2e-10,
    )


@pytest.mark.parametrize("l_count", [2, 3])
def test_fixed_direction_contractions_and_all_objectives_match_dense_four_kernels(
    l_count: int,
) -> None:
    fixture = _fixture(l_count=l_count, seed=7200 + l_count)
    _, _, contractions = _build_contractions(fixture)
    raw = np.linspace(0.75, -0.4, l_count)
    direction = _canonical_direction(raw, fixture["context_metric"])
    dense = _dense_evaluation(fixture, direction)
    evaluations = {
        objective: evaluate_context_direction(
            contractions, direction, objective=objective
        )
        for objective in (
            "interaction_coefficient",
            "interaction_trace_contribution",
            "he_moment_gain",
        )
    }
    baseline = evaluations["interaction_coefficient"]
    np.testing.assert_allclose(baseline.matrix, dense["matrix"], rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(baseline.rhs, dense["rhs"], rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(baseline.traces, dense["traces"], rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(
        baseline.coefficients,
        dense["coefficients"],
        rtol=3e-10,
        atol=3e-10,
    )
    assert baseline.rank == 4
    assert np.isfinite(baseline.condition_number)
    for objective, evaluation in evaluations.items():
        np.testing.assert_allclose(evaluation.matrix, baseline.matrix)
        np.testing.assert_allclose(evaluation.rhs, baseline.rhs)
        np.testing.assert_allclose(evaluation.coefficients, baseline.coefficients)
        assert evaluation.objective == objective
        np.testing.assert_allclose(
            evaluation.objective_value,
            dense[objective],
            rtol=3e-10,
            atol=3e-10,
        )


def test_residual_direction_kernel_is_p_diag_e_squared_p_not_a_commuted_shortcut() -> (
    None
):
    fixture = _fixture(l_count=2, seed=7204)
    _, _, contractions = _build_contractions(fixture)
    direction = _canonical_direction(
        np.asarray([0.65, -0.55]), fixture["context_metric"]
    )
    evaluation = evaluate_context_direction(
        contractions, direction, objective="he_moment_gain"
    )
    dense = _dense_evaluation(fixture, direction)
    projector = fixture["study_projector"].projector
    environment = fixture["study_environments"] @ direction
    correct = projector @ (environment[:, None] ** 2 * projector)
    wrong = environment[:, None] ** 2 * projector
    assert np.linalg.norm(correct - wrong) > 1.0e-2
    kernels = dense["study_kernels"]
    np.testing.assert_allclose(kernels[3], correct)
    for left, right in ((0, 3), (1, 3), (2, 3), (3, 3)):
        expected = np.sum(kernels[left] * kernels[right])
        np.testing.assert_allclose(
            evaluation.matrix[left, right], expected, rtol=2e-11, atol=2e-10
        )


def test_objective_ids_remain_distinct_when_coefficient_and_trace_rank_oppositely() -> (
    None
):
    fixture = _fixture(l_count=2, seed=7230)
    _, _, contractions = _build_contractions(fixture)
    first = np.asarray([0.8890861558781508, -0.13149363320964158])
    second = np.asarray([-0.8837586663896023, 0.5238671255059034])
    dense_first = _dense_evaluation(fixture, first)
    dense_second = _dense_evaluation(fixture, second)
    coefficient_difference = float(
        dense_first["interaction_coefficient"] - dense_second["interaction_coefficient"]
    )
    trace_difference = float(
        dense_first["interaction_trace_contribution"]
        - dense_second["interaction_trace_contribution"]
    )
    assert coefficient_difference * trace_difference < 0.0
    for objective in (
        "interaction_coefficient",
        "interaction_trace_contribution",
        "he_moment_gain",
    ):
        for direction, dense in (
            (first, dense_first),
            (second, dense_second),
        ):
            evaluation = evaluate_context_direction(
                contractions, direction, objective=objective
            )
            np.testing.assert_allclose(
                evaluation.objective_value,
                dense[objective],
                rtol=3e-10,
                atol=3e-10,
            )
    with pytest.raises(ValueError, match="(?i)unknown.*objective"):
        evaluate_context_direction(contractions, first, objective="generic_gxe_signal")


def test_independent_reference_genetic_block_uses_exact_same_person_transfer() -> None:
    fixture = _fixture(
        l_count=2,
        seed=7220,
        n_reference=37,
        n_study=29,
        n_variants=25,
    )
    _, _, contractions = _build_contractions(fixture)
    direction = _canonical_direction(np.asarray([0.7, -0.5]), fixture["context_metric"])
    evaluation = evaluate_context_direction(
        contractions, direction, objective="interaction_coefficient"
    )
    reference_kernels = _dense_kernels(fixture, direction, reference=True)[:2]
    reference_gram = np.einsum(
        "aij,bij->ab", reference_kernels, reference_kernels, optimize=True
    )
    same_person = exact_same_person_matrix(reference_kernels)
    reference_n = fixture["reference_genotype"].shape[0]
    study_n = fixture["study_genotype"].shape[0]
    alpha = study_n / reference_n
    beta = study_n * (study_n - 1.0) / (reference_n * (reference_n - 1.0))
    expected = alpha * same_person + beta * (reference_gram - same_person)
    np.testing.assert_allclose(
        evaluation.matrix[:2, :2], expected, rtol=2e-11, atol=2e-10
    )


def test_direction_normalization_evenness_and_deterministic_sign() -> None:
    fixture = _fixture(l_count=3, seed=7205)
    _, _, contractions = _build_contractions(fixture)
    raw = np.asarray([-0.4, 1.1, -0.3])
    positive = normalize_context_direction(raw, fixture["context_metric"])
    negative = normalize_context_direction(-raw, fixture["context_metric"])
    np.testing.assert_array_equal(positive, negative)
    np.testing.assert_allclose(
        positive @ fixture["context_metric"] @ positive, 1.0, atol=3e-15
    )
    assert positive[np.argmax(np.abs(positive))] > 0.0
    left = evaluate_context_direction(
        contractions, positive, objective="interaction_trace_contribution"
    )
    right = evaluate_context_direction(
        contractions, -positive, objective="interaction_trace_contribution"
    )
    np.testing.assert_array_equal(left.direction, right.direction)
    np.testing.assert_allclose(left.matrix, right.matrix, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(left.rhs, right.rhs, rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(left.coefficients, right.coefficients, atol=2e-13)
    assert left.objective_value == pytest.approx(right.objective_value, abs=2e-13)
    with pytest.raises(ValueError, match="(?i)zero|norm"):
        normalize_context_direction(np.zeros(3), fixture["context_metric"])


def test_nonorthogonal_basis_mixing_preserves_fixed_direction_system_and_objective() -> (
    None
):
    fixture = _fixture(l_count=3, seed=7206)
    _, _, original = _build_contractions(fixture)
    transform = np.asarray([[1.35, 0.25, -0.1], [0.0, 0.75, 0.2], [0.15, -0.05, 1.1]])
    mixed = dict(fixture)
    mixed["reference_environments"] = fixture["reference_environments"] @ transform.T
    mixed["study_environments"] = fixture["study_environments"] @ transform.T
    mixed["context_metric"] = transform @ fixture["context_metric"] @ transform.T
    mixed["environment_names"] = tuple(f"mixed_{index}" for index in range(3))
    _, _, transformed = _build_contractions(mixed)
    direction = _canonical_direction(
        np.asarray([0.8, -0.45, 0.3]), fixture["context_metric"]
    )
    transformed_direction = np.linalg.solve(transform.T, direction)
    np.testing.assert_allclose(
        mixed["study_environments"] @ transformed_direction,
        fixture["study_environments"] @ direction,
        rtol=2e-14,
        atol=2e-14,
    )
    left = evaluate_context_direction(original, direction, objective="he_moment_gain")
    right = evaluate_context_direction(
        transformed, transformed_direction, objective="he_moment_gain"
    )
    np.testing.assert_allclose(left.matrix, right.matrix, rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(left.rhs, right.rhs, rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(left.traces, right.traces, rtol=2e-11, atol=2e-10)
    np.testing.assert_allclose(
        left.coefficients, right.coefficients, rtol=3e-10, atol=3e-10
    )
    assert left.objective_value == pytest.approx(
        right.objective_value, rel=3e-10, abs=3e-10
    )


@pytest.mark.parametrize("l_count,grid_count", [(2, 1440), (3, 3500)])
def test_optimizer_matches_independent_dense_metric_sphere_grid(
    l_count: int, grid_count: int
) -> None:
    fixture = _fixture(l_count=l_count, seed=7207 + l_count)
    _, _, contractions = _build_contractions(fixture)
    result = optimize_context_direction(contractions, objective="he_moment_gain")
    grid = _metric_sphere_grid(fixture["context_metric"], grid_count)
    values: list[float] = []
    for direction in grid:
        try:
            dense = _dense_evaluation(fixture, direction)
        except np.linalg.LinAlgError:
            continue
        values.append(float(dense["he_moment_gain"]))
    assert values
    grid_best = max(values)
    assert result.evaluation.objective_value >= grid_best - 2.5e-4 * max(
        abs(grid_best), 1.0e-5
    )
    assert result.evaluation.rank == 4
    assert result.evaluation.relative_residual < 1.0e-9
    np.testing.assert_allclose(
        result.direction @ fixture["context_metric"] @ result.direction,
        1.0,
        atol=3e-12,
    )
    repeated = optimize_context_direction(contractions, objective="he_moment_gain")
    np.testing.assert_array_equal(result.direction, repeated.direction)
    assert result.evaluation.objective_value == repeated.evaluation.objective_value


def test_optimizer_marks_an_exact_continuum_of_full_rank_optima_nonunique() -> None:
    fixture = _fixture(l_count=2, seed=7221)
    contractions = _isotropic_full_rank_contractions(fixture)
    directions = _metric_sphere_grid(fixture["context_metric"], 37)
    values = np.asarray(
        [
            evaluate_context_direction(
                contractions, direction, objective="he_moment_gain"
            ).objective_value
            for direction in directions
        ]
    )
    np.testing.assert_allclose(values, values[0], rtol=0.0, atol=2e-13)
    result = optimize_context_direction(
        contractions,
        objective="he_moment_gain",
        validation_grid_size=73,
    )
    assert "nonunique" in result.status or "unstable" in result.status


def test_evaluation_and_optimization_use_only_compact_contractions(monkeypatch) -> None:
    fixture = _fixture(l_count=2, seed=7211, n_reference=37, n_variants=29)
    _, _, contractions = _build_contractions(fixture)
    direction = _canonical_direction(
        np.asarray([0.7, -0.45]), fixture["context_metric"]
    )
    expected = evaluate_context_direction(
        contractions, direction, objective="interaction_coefficient"
    )
    for name in (
        "reference_genotype",
        "study_genotype",
        "reference_environments",
        "study_environments",
        "phenotype",
    ):
        fixture[name][...] = np.nan
    import summit.context.direction as direction_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Individual-level path revisited after contraction.")

    for name in (
        "common_scale_features",
        "project_normalize_phenotype",
        "build_direction_reference_contractions",
        "build_direction_trait_contractions",
    ):
        if hasattr(direction_module, name):
            monkeypatch.setattr(direction_module, name, forbidden)
    observed = evaluate_context_direction(
        contractions, direction, objective="interaction_coefficient"
    )
    np.testing.assert_array_equal(observed.matrix, expected.matrix)
    np.testing.assert_array_equal(observed.rhs, expected.rhs)
    optimize_context_direction(contractions, objective="interaction_coefficient")
    shapes = _compact_array_shapes(contractions)
    assert shapes
    assert max(max(shape, default=0) for shape in shapes) < 29


def test_singular_metric_and_direction_dependent_four_kernel_alias_fail_closed() -> (
    None
):
    fixture = _fixture(l_count=2, seed=7212)
    singular_metric = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="(?i)metric|rank|positive"):
        build_direction_reference_contractions(
            fixture["reference_genotype"],
            fixture["reference_environments"],
            fixture["reference_projector"],
            context_metric=singular_metric,
            environment_names=fixture["environment_names"],
            variant_hash=fixture["variant_hash"],
            genotype_scaling="pre_scaled_input",
            variant_weights=fixture["variant_weights"],
        )

    n_samples = fixture["study_environments"].shape[0]
    alternating = np.where(np.arange(n_samples) % 2 == 0, -1.0, 1.0)
    second = fixture["study_environments"][:, 1]
    environments = np.column_stack([alternating, second])
    alias_fixture = dict(fixture)
    alias_fixture["reference_environments"] = environments.copy()
    alias_fixture["study_environments"] = environments.copy()
    alias_fixture["context_metric"] = environments.T @ environments / n_samples
    alias_fixture["reference_projector"] = rank_revealing_projector(
        np.column_stack([np.ones(n_samples), environments])
    )
    alias_fixture["study_projector"] = alias_fixture["reference_projector"]
    _, _, contractions = _build_contractions(alias_fixture)
    direction = normalize_context_direction(
        np.asarray([1.0, 0.0]), alias_fixture["context_metric"]
    )
    with pytest.raises(ContextRankError):
        evaluate_context_direction(
            contractions, direction, objective="interaction_coefficient"
        )


def test_context_metric_condition_policy_is_scale_relative_hashed_and_reported() -> (
    None
):
    fixture = _fixture(l_count=2, seed=7222)
    ill_conditioned = np.diag([1.0, 2.0e-11])
    with pytest.raises(ValueError, match="(?i)ill-conditioned|condition"):
        build_direction_reference_contractions(
            fixture["reference_genotype"],
            fixture["reference_environments"],
            fixture["reference_projector"],
            context_metric=ill_conditioned,
            context_metric_source="adversarial_metric",
            environment_names=fixture["environment_names"],
            variant_hash=fixture["variant_hash"],
            genotype_scaling="pre_scaled_input",
            variant_weights=fixture["variant_weights"],
        )

    accepted = np.diag([1.0e-12, 2.0e-22])
    reference = build_direction_reference_contractions(
        fixture["reference_genotype"],
        fixture["reference_environments"],
        fixture["reference_projector"],
        context_metric=accepted,
        context_metric_source="scaled_reference_metric",
        environment_names=fixture["environment_names"],
        variant_hash=fixture["variant_hash"],
        genotype_scaling="pre_scaled_input",
        variant_weights=fixture["variant_weights"],
    )
    shared = reference.manifest["shared_specification"]
    diagnostics = shared["context_metric_diagnostics"]
    assert shared["context_metric_source"] == "scaled_reference_metric"
    assert diagnostics["rank"] == 2
    assert diagnostics["condition_number"] == pytest.approx(5.0e9)
    assert diagnostics["maximum_condition_number"] == pytest.approx(1.0e10)
    assert shared["context_metric_hash"] == array_sha256(accepted)


def test_reference_trait_identity_mismatches_fail_before_evaluation() -> None:
    fixture = _fixture(l_count=2, seed=7213)
    reference, trait, _ = _build_contractions(fixture)
    mismatched_fixture = dict(fixture)
    mismatched_fixture["variant_hash"] = canonical_sha256({"variant_ids": ["wrong"]})
    _, mismatched_trait, _ = _build_contractions(mismatched_fixture)
    with pytest.raises(ValueError, match="(?i)variant|compatib|hash"):
        combine_context_direction_contractions(reference, mismatched_trait)
    changed_metric_trait = replace(
        trait,
        context_metric=np.asarray(trait.context_metric) * 1.01,
    )
    with pytest.raises(ValueError, match="(?i)metric|compatib|hash"):
        combine_context_direction_contractions(reference, changed_metric_trait)


def test_exact_crossfit_fold_tensors_match_fresh_disjoint_subset_rebuilds() -> None:
    fixture = _fixture(l_count=2, seed=7214, n_variants=24)
    block_ids = np.repeat(np.asarray(["b0", "b1", "b2", "b3"]), 6)
    crossfit = _build_crossfit(fixture, block_ids)
    validation = validate_direction_crossfit_contractions(crossfit)
    assert validation["uses_approximate_loo_deletion"] is False
    assert crossfit.manifest["construction"] == "exact_disjoint_variant_subset_rebuild"
    assert crossfit.manifest["uses_approximate_loo_deletion"] is False
    assert sum(crossfit.assignment.fold_variant_counts) == block_ids.size
    assert set(crossfit.assignment.fold_block_identities[0]).isdisjoint(
        crossfit.assignment.fold_block_identities[1]
    )
    assert set(crossfit.assignment.fold_block_identities[0]).union(
        crossfit.assignment.fold_block_identities[1]
    ) == set(crossfit.assignment.block_identities)

    fold_objects = (crossfit.arms[0].train, crossfit.arms[0].heldout)
    for fold, observed in enumerate(fold_objects):
        mask = crossfit.assignment.fold_mask(block_ids, fold)
        _, _, expected = _build_contractions(fixture, variant_mask=mask)
        _assert_compact_contractions_equal(observed, expected)
        dense = _dense_evaluation(
            fixture,
            normalize_context_direction(
                np.asarray([0.7, -0.45]), fixture["context_metric"]
            ),
            variant_mask=mask,
        )
        evaluation = evaluate_context_direction(
            observed,
            np.asarray([0.7, -0.45]),
            objective="interaction_trace_contribution",
        )
        np.testing.assert_allclose(
            evaluation.matrix, dense["matrix"], rtol=2e-11, atol=2e-10
        )
        np.testing.assert_allclose(evaluation.rhs, dense["rhs"], rtol=2e-11, atol=2e-10)

    assert crossfit.arms[0].train_variant_hash == crossfit.arms[1].heldout_variant_hash
    assert crossfit.arms[0].heldout_variant_hash == crossfit.arms[1].train_variant_hash
    assert crossfit.arms[0].train_variant_hash != crossfit.arms[0].heldout_variant_hash

    result = crossfit_context_direction(
        crossfit, objective="he_moment_gain", validation_grid_size=181
    )
    assert len(result.folds) == 2
    np.testing.assert_allclose(
        result.combined_heldout_value,
        np.mean([fold.heldout_objective for fold in result.folds]),
    )
    for arm, fold in zip(crossfit.arms, result.folds):
        assert fold.train_variant_hash == arm.train_variant_hash
        assert fold.heldout_variant_hash == arm.heldout_variant_hash
        np.testing.assert_allclose(
            fold.heldout_evaluation.direction,
            fold.training_optimization.direction,
        )
        assert fold.training_objective == fold.training_optimization.objective_value
        assert fold.heldout_objective == fold.heldout_evaluation.objective_value
    assert result.manifest["in_sample_objective_is_unbiased"] is False


def test_crossfit_training_is_unchanged_when_only_its_heldout_variants_change(
    monkeypatch,
) -> None:
    fixture = _fixture(l_count=2, seed=7215, n_variants=24)
    block_ids = np.repeat(np.asarray([0, 1, 2, 3]), 6)
    original = _build_crossfit(fixture, block_ids)
    arm = original.arms[0]
    heldout_mask = original.assignment.fold_mask(block_ids, 1)
    changed_fixture = dict(fixture)
    rng = np.random.default_rng(7216)
    for cohort in ("reference", "study"):
        key = f"{cohort}_genotype"
        changed = fixture[key].copy()
        changed[:, heldout_mask] = _standardize_genotype(
            rng.normal(size=(changed.shape[0], int(np.sum(heldout_mask))))
        )
        changed_fixture[key] = changed
    changed = _build_crossfit(changed_fixture, block_ids)
    changed_arm = changed.arms[0]
    _assert_compact_contractions_equal(arm.train, changed_arm.train)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            arm.heldout.summary.base_rhs, changed_arm.heldout.summary.base_rhs
        )

    left_training = optimize_context_direction(
        arm.train, objective="he_moment_gain", validation_grid_size=181
    )
    right_training = optimize_context_direction(
        changed_arm.train, objective="he_moment_gain", validation_grid_size=181
    )
    np.testing.assert_array_equal(left_training.direction, right_training.direction)
    assert left_training.objective_value == right_training.objective_value
    left_heldout = evaluate_context_direction(
        arm.heldout, left_training.direction, objective="he_moment_gain"
    )
    right_heldout = evaluate_context_direction(
        changed_arm.heldout, right_training.direction, objective="he_moment_gain"
    )
    assert not np.isclose(
        left_heldout.objective_value,
        right_heldout.objective_value,
        rtol=1e-8,
        atol=1e-10,
    )

    import summit.context.direction as direction_module

    real_optimize = direction_module.optimize_context_direction
    real_evaluate = direction_module.evaluate_context_direction
    paired_heldout = {id(value.train): id(value.heldout) for value in original.arms}
    state = {"forbidden_heldout": None}

    def guarded_optimize(contractions, **kwargs):
        state["forbidden_heldout"] = paired_heldout[id(contractions)]
        try:
            return real_optimize(contractions, **kwargs)
        finally:
            state["forbidden_heldout"] = None

    def guarded_evaluate(contractions, direction, **kwargs):
        if id(contractions) == state["forbidden_heldout"]:
            raise AssertionError("Held-out contractions leaked into optimization.")
        return real_evaluate(contractions, direction, **kwargs)

    monkeypatch.setattr(
        direction_module, "optimize_context_direction", guarded_optimize
    )
    monkeypatch.setattr(
        direction_module, "evaluate_context_direction", guarded_evaluate
    )
    crossfit_context_direction(
        original, objective="he_moment_gain", validation_grid_size=91
    )


def test_crossfit_applies_the_declared_condition_ceiling_to_heldout_folds(
    monkeypatch,
) -> None:
    fixture = _fixture(l_count=2, seed=7223, n_variants=24)
    block_ids = np.repeat(np.asarray(["b0", "b1", "b2", "b3"]), 6)
    contractions = _build_crossfit(fixture, block_ids)
    import summit.context.direction as direction_module

    training = {
        id(arm.train): optimize_context_direction(
            arm.train,
            objective="he_moment_gain",
            validation_grid_size=91,
        )
        for arm in contractions.arms
    }
    real_evaluate = direction_module.evaluate_context_direction

    def fixed_training(train, **kwargs):
        return training[id(train)]

    def ill_conditioned_heldout(heldout, direction, **kwargs):
        evaluation = real_evaluate(heldout, direction, **kwargs)
        return replace(evaluation, condition_number=1.0e12)

    monkeypatch.setattr(direction_module, "optimize_context_direction", fixed_training)
    monkeypatch.setattr(
        direction_module, "evaluate_context_direction", ill_conditioned_heldout
    )
    with pytest.raises(ValueError, match="(?i)held.?out|condition"):
        crossfit_context_direction(
            contractions,
            objective="he_moment_gain",
            validation_grid_size=91,
            max_condition_number=1.0e10,
        )


def test_balanced_fold_assignment_keeps_blocks_whole_and_hashes_partition() -> None:
    block_ids = np.asarray(
        ["a"] * 3 + ["b"] * 5 + ["c"] * 4 + ["d"] * 4,
        dtype=object,
    )
    weights = np.linspace(0.5, 1.5, block_ids.size)
    variant_hash = canonical_sha256(
        {"variant_ids": [f"v{index}" for index in range(block_ids.size)]}
    )
    assignment = build_balanced_direction_folds(
        block_ids,
        variant_hash=variant_hash,
        variant_weights=weights,
    )
    masks = tuple(assignment.fold_mask(block_ids, fold) for fold in (0, 1))
    assert not np.any(masks[0] & masks[1])
    assert np.all(masks[0] | masks[1])
    for block in np.unique(block_ids):
        selected_folds = {
            fold for fold, mask in enumerate(masks) if np.any(mask[block_ids == block])
        }
        assert len(selected_folds) == 1
    assert assignment.fold_variant_counts == tuple(int(np.sum(mask)) for mask in masks)
    np.testing.assert_allclose(
        assignment.fold_weight_masses,
        [np.sum(weights[mask]) for mask in masks],
    )
    with pytest.raises(ValueError, match="(?i)two distinct|two-fold"):
        build_balanced_direction_folds(np.asarray(["one"] * 4))
    with pytest.raises(ValueError, match="(?i)weight"):
        build_balanced_direction_folds(
            block_ids,
            variant_weights=np.where(np.arange(block_ids.size) == 0, -1.0, 1.0),
        )


def test_crossfit_validation_rejects_tampered_arm_and_assignment_hash_claims() -> None:
    fixture = _fixture(l_count=2, seed=7217, n_variants=24)
    block_ids = np.repeat(np.asarray(["b0", "b1", "b2", "b3"]), 6)
    crossfit = _build_crossfit(fixture, block_ids)
    first = crossfit.arms[0]
    tampered_arm = replace(
        first,
        train_variant_hash=first.heldout_variant_hash,
    )
    with pytest.raises(ValueError, match="(?i)variant|disjoint|hash"):
        validate_direction_crossfit_contractions(
            replace(crossfit, arms=(tampered_arm, crossfit.arms[1]))
        )

    tampered_assignment = replace(
        crossfit.assignment,
        fold_variant_counts=(
            crossfit.assignment.fold_variant_counts[0] + 1,
            crossfit.assignment.fold_variant_counts[1] - 1,
        ),
    )
    with pytest.raises(ValueError, match="(?i)assignment|count|hash"):
        validate_direction_crossfit_contractions(
            replace(crossfit, assignment=tampered_assignment)
        )

    overlapping_left_mask = np.zeros(24, dtype=np.bool_)
    overlapping_right_mask = np.zeros(24, dtype=np.bool_)
    overlapping_left_mask[:18] = True
    overlapping_right_mask[6:] = True
    _, _, overlapping_left = _build_contractions(
        fixture, variant_mask=overlapping_left_mask
    )
    _, _, overlapping_right = _build_contractions(
        fixture, variant_mask=overlapping_right_mask
    )
    left_hash = overlapping_left.reference.selected_variant_hash
    right_hash = overlapping_right.reference.selected_variant_hash
    forged_arms = (
        DirectionCrossFitArmContractions(
            fold_id="arm0",
            train_fold=crossfit.assignment.fold_labels[0],
            heldout_fold=crossfit.assignment.fold_labels[1],
            train_variant_hash=left_hash,
            heldout_variant_hash=right_hash,
            train=overlapping_left,
            heldout=overlapping_right,
        ),
        DirectionCrossFitArmContractions(
            fold_id="arm1",
            train_fold=crossfit.assignment.fold_labels[1],
            heldout_fold=crossfit.assignment.fold_labels[0],
            train_variant_hash=right_hash,
            heldout_variant_hash=left_hash,
            train=overlapping_right,
            heldout=overlapping_left,
        ),
    )
    forged_manifest = copy.deepcopy(crossfit.manifest)
    forged_manifest["fold_selected_variant_hashes"] = [left_hash, right_hash]
    with pytest.raises(ValueError, match="(?i)fold|variant|count|mass|assignment|hash"):
        validate_direction_crossfit_contractions(
            replace(crossfit, arms=forged_arms, manifest=forged_manifest)
        )

    fold_one_mask = crossfit.assignment.fold_mask(block_ids, 1)
    shared_spec_hash = canonical_sha256(
        {
            "environment_names": list(fixture["environment_names"]),
            "nuisance": "intercept_context_main_effects_pc",
        }
    )
    altered_metric_common = {
        "context_metric": 2.0 * fixture["context_metric"],
        "environment_names": fixture["environment_names"],
        "variant_hash": fixture["variant_hash"],
        "context_spec_hash": shared_spec_hash,
        "genotype_scaling": "pre_scaled_input",
        "variant_weights": fixture["variant_weights"],
        "variant_mask": fold_one_mask,
        "fold_id": crossfit.assignment.fold_labels[1],
    }
    altered_reference = build_direction_reference_contractions(
        fixture["reference_genotype"],
        fixture["reference_environments"],
        fixture["reference_projector"],
        **altered_metric_common,
    )
    altered_summary = build_direction_trait_contractions(
        fixture["study_genotype"],
        fixture["study_environments"],
        fixture["study_projector"],
        fixture["phenotype"],
        **altered_metric_common,
    )
    altered_fold = combine_context_direction_contractions(
        altered_reference, altered_summary
    )
    valid_fold_zero = crossfit.arms[0].train
    fold_zero_hash = valid_fold_zero.reference.selected_variant_hash
    fold_one_hash = altered_fold.reference.selected_variant_hash
    metric_mismatched_arms = (
        DirectionCrossFitArmContractions(
            "arm0",
            crossfit.assignment.fold_labels[0],
            crossfit.assignment.fold_labels[1],
            fold_zero_hash,
            fold_one_hash,
            valid_fold_zero,
            altered_fold,
        ),
        DirectionCrossFitArmContractions(
            "arm1",
            crossfit.assignment.fold_labels[1],
            crossfit.assignment.fold_labels[0],
            fold_one_hash,
            fold_zero_hash,
            altered_fold,
            valid_fold_zero,
        ),
    )
    with pytest.raises(ValueError, match="(?i)metric|specification|cross.?fold|hash"):
        validate_direction_crossfit_contractions(
            replace(crossfit, arms=metric_mismatched_arms)
        )

    # The two reverse arms duplicate each fold.  Changing only the second
    # copy must also fail closed; checking the first arm's two specifications
    # alone does not establish the identity of the objects actually evaluated
    # in the second arm.
    asymmetric_second_arm = DirectionCrossFitArmContractions(
        "arm1",
        crossfit.assignment.fold_labels[1],
        crossfit.assignment.fold_labels[0],
        fold_one_hash,
        fold_zero_hash,
        altered_fold,
        valid_fold_zero,
    )
    with pytest.raises(ValueError, match="(?i)metric|specification|cross.?fold|hash"):
        validate_direction_crossfit_contractions(
            replace(crossfit, arms=(crossfit.arms[0], asymmetric_second_arm))
        )
