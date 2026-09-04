from __future__ import annotations

import numpy as np
import pytest

from summit.context.fit import ContextNormalEquations, ContextRankError
from summit.context.oracle import transfer_reference_gram
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
)
from summit.ldscore.generalized_gxe_fit_v1 import (
    _trait_moments_after_deleting_blocks,
    assemble_generalized_gxe_normal_equations_v1,
    fit_profiled_response_rank_v1,
    fit_generalized_gxe_variant_model_v1,
    validate_generalized_gxe_trait_compatibility_v1,
)
from summit.ldscore.generalized_gxe_reference_v1 import (
    build_generalized_gxe_variant_reference_v1,
    serialize_generalized_gxe_inference_axes,
)
from summit.ldscore.generalized_gxe_variant import (
    GlobalVariantProbeSpec,
    TwoPassLedger,
)
from summit.ldscore.generalized_gxe_trait_summary import GeneralizedGxETraitSummary
from test_context_stage4_fit_v1 import _artifacts
from test_context_stage4_trait_v1 import _artifact as stable_trait_artifact


def _rank_fit_equations(
    omega: np.ndarray, *, residual_count: int = 3
) -> tuple[ContextNormalEquations, ContextComponentIndex, np.ndarray]:
    components = ContextComponentIndex(("all",), ContextPairIndex(len(omega)))
    genetic = np.asarray(
        [omega[entry.q, entry.r] for entry in components.entries], dtype=float
    )
    residual = np.linspace(0.1, 0.1 * residual_count, residual_count)
    coefficients = np.concatenate((genetic, residual))
    rng = np.random.default_rng(9137)
    design = rng.normal(size=(len(coefficients) + 4, len(coefficients)))
    matrix = design.T @ design + np.eye(len(coefficients))
    rhs = matrix @ coefficients
    equations = ContextNormalEquations(
        matrix=matrix,
        rhs=rhs,
        traces=np.zeros(len(coefficients)),
        component_names=tuple(
            [f"omega_{entry.q}_{entry.r}" for entry in components.entries]
            + [f"residual_{index}" for index in range(residual_count)]
        ),
        genetic_count=len(components),
        annotation_masses=np.ones(1),
        deleted_groups=(),
        reference_genetic_gram=np.eye(len(components)),
        transferred_genetic_gram=np.eye(len(components)),
        reference_n=1000,
        study_n=900,
    )
    return equations, components, coefficients


def test_profiled_rank_one_recovers_exact_rank_one_covariance() -> None:
    baseline = 0.4
    alignment = np.asarray([0.2, -0.1, 0.3])
    response = np.asarray([0.1, 0.25, -0.2])
    omega = np.zeros((4, 4))
    omega[0, 0] = baseline
    omega[0, 1:] = omega[1:, 0] = baseline * alignment
    omega[1:, 1:] = baseline * np.outer(alignment, alignment) + np.outer(
        response, response
    )
    equations, components, expected = _rank_fit_equations(omega)
    result = fit_profiled_response_rank_v1(
        equations, components, response_rank=1
    )
    np.testing.assert_allclose(result.coefficients, expected, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(result.omega, omega, rtol=2e-8, atol=2e-9)
    assert np.linalg.eigvalsh(result.omega)[0] >= -1e-10
    assert np.linalg.matrix_rank(result.conditional_response, tol=1e-9) == 1
    assert result.profiled_quadratic_distance < 1e-8


def test_profiled_rank_models_are_nested_and_profile_residual_terms() -> None:
    baseline = 0.35
    alignment = np.asarray([0.1, 0.2, -0.15])
    response = np.diag([0.08, 0.04, 0.02])
    omega = np.zeros((4, 4))
    omega[0, 0] = baseline
    omega[0, 1:] = omega[1:, 0] = baseline * alignment
    omega[1:, 1:] = baseline * np.outer(alignment, alignment) + response
    equations, components, _ = _rank_fit_equations(omega)
    rank_zero = fit_profiled_response_rank_v1(
        equations, components, response_rank=0
    )
    rank_one = fit_profiled_response_rank_v1(
        equations, components, response_rank=1
    )
    assert rank_one.optimizer_cost <= rank_zero.optimizer_cost + 1e-10
    assert np.linalg.matrix_rank(rank_zero.conditional_response, tol=1e-10) == 0
    assert np.linalg.matrix_rank(rank_one.conditional_response, tol=1e-10) <= 1
    c = len(components)
    residual_normal_equations = (
        equations.matrix[c:, :] @ rank_one.coefficients - equations.rhs[c:]
    )
    np.testing.assert_allclose(residual_normal_equations, 0.0, atol=2e-10)


def test_profiled_rank_fit_rejects_invalid_rank() -> None:
    equations, components, _ = _rank_fit_equations(np.eye(3))
    with pytest.raises(ValueError, match="zero or one"):
        fit_profiled_response_rank_v1(
            equations, components, response_rank=2
        )


def test_profiled_rank_fit_accepts_covariance_metric_for_indefinite_system() -> None:
    baseline = 0.4
    alignment = np.asarray([0.2, -0.1])
    response = np.asarray([0.15, 0.3])
    omega = np.zeros((3, 3))
    omega[0, 0] = baseline
    omega[0, 1:] = omega[1:, 0] = baseline * alignment
    omega[1:, 1:] = baseline * np.outer(alignment, alignment) + np.outer(
        response, response
    )
    equations, components, expected = _rank_fit_equations(omega)
    matrix = np.asarray(equations.matrix).copy()
    matrix[0, 0] = -abs(matrix[0, 0])
    matrix = 0.5 * (matrix + matrix.T)
    indefinite = ContextNormalEquations(
        matrix=matrix,
        rhs=matrix @ expected,
        traces=equations.traces,
        component_names=equations.component_names,
        genetic_count=equations.genetic_count,
        annotation_masses=equations.annotation_masses,
        deleted_groups=equations.deleted_groups,
        reference_genetic_gram=equations.reference_genetic_gram,
        transferred_genetic_gram=equations.transferred_genetic_gram,
        reference_n=equations.reference_n,
        study_n=equations.study_n,
    )
    result = fit_profiled_response_rank_v1(
        indefinite,
        components,
        response_rank=1,
        coefficient_covariance=np.eye(len(components)),
    )
    np.testing.assert_allclose(result.genetic_coefficients, expected[: len(components)], atol=2e-8)
    assert result.objective_metric == "genetic_jackknife_covariance_precision"
    assert result.objective_rank == len(components)


def _completed_ledger(block_ids: np.ndarray) -> dict[str, int]:
    ledger = TwoPassLedger(len(block_ids))
    boundaries = np.flatnonzero(np.diff(block_ids)) + 1
    stops = [*boundaries.tolist(), len(block_ids)]
    for pass_number in (1, 2):
        ledger.begin_pass(pass_number)
        start = 0
        for stop in stops:
            ledger.record_block(start, stop)
            start = stop
        ledger.finish_pass()
    ledger.validate_clean_completion()
    return ledger.to_dict()


def _current_trait(trait) -> GeneralizedGxETraitSummary:
    dimensions = trait.manifest["dimensions"]
    residual_rank = int(
        dimensions.get("residual_rank", dimensions["N_study"] - 2)
    )
    return GeneralizedGxETraitSummary(
        n_samples=trait.n_samples,
        n_variants=trait.n_variants,
        residual_rank=residual_rank,
        component_index=trait.component_index,
        group_ids=trait.group_ids,
        trait_ids=trait.trait_ids,
        residual_names=trait.residual_names,
        genetic_rhs=trait.genetic_rhs,
        genetic_traces=trait.genetic_traces,
        genetic_residual=trait.genetic_residual,
        residual_rhs=trait.residual_rhs,
        residual_traces=trait.residual_traces,
        residual_gram=trait.residual_gram,
        group_rhs_unnormalized_num=trait.group_rhs_unnormalized_num,
        group_trace_unnormalized_num=trait.group_trace_unnormalized_num,
        group_genetic_residual_num=trait.group_genetic_residual_num,
        annotation_masses=trait.annotation_masses,
        group_annotation_masses=trait.group_annotation_masses,
        group_variant_counts=trait.group_variant_counts,
    )


def _matching_reference(
    trait, *, genotype_scale_plan, include_panel: bool = False
):
    counts = np.asarray(trait.group_variant_counts, dtype=np.int64)
    block_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    n_reference = max(10, trait.n_samples + 2)
    fixed_rank = 2
    residual_rank = n_reference - fixed_rank
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=trait.n_variants,
        num_samples=n_reference,
        basis_names=tuple(
            f"basis_{index}"
            for index in range(trait.component_index.pair_index.num_basis)
        ),
        fixed_effect_rank=fixed_rank,
        annotation_names=trait.component_index.annotation_names,
        annotation_masses=trait.annotation_masses,
        variant_block_ids=block_ids,
        block_labels=trait.group_ids,
        residual_component_names=trait.residual_names,
    )
    c = len(trait.component_index)
    masses = np.asarray(trait.annotation_masses, dtype=np.float64)
    component_annotations = np.fromiter(
        (entry.annotation_index for entry in trait.component_index.entries),
        dtype=np.int64,
        count=c,
    )
    component_masses = masses[component_annotations]
    gram = 2.0 * np.eye(c) + 0.05 * np.ones((c, c))
    directed = (
        gram
        * np.outer(component_masses, component_masses)
        / float(residual_rank**2)
    )
    fractions = counts.astype(np.float64) / float(np.sum(counts))
    block_directed = fractions[:, None, None] * directed[None, :, :]
    block_masses = np.asarray(trait.group_annotation_masses, dtype=np.float64)
    deleted = np.empty_like(block_directed)
    for block in range(len(counts)):
        retained_masses = masses - block_masses[block]
        retained_components = retained_masses[component_annotations]
        retained_directed = directed - block_directed[block]
        deleted[block] = (
            float(residual_rank**2)
            * 0.5 * (retained_directed + retained_directed.T)
            / np.outer(retained_components, retained_components)
        )
    arrays = {
        "directed_numerator": directed,
        "symmetric_numerator": directed.copy(),
        "genetic_gram": gram,
        "block_directed_numerator": block_directed,
        "block_annotation_mass": block_masses,
        "deleted_genetic_gram": deleted,
        "same_person": 0.2 * np.eye(c),
    }
    if include_panel:
        p = len(trait.component_index.pair_index)
        arrays["directional_ldscores"] = np.zeros(
            (trait.n_variants, p, c), dtype=np.float64
        )
    phases = {name: 0.0 for name in ("pass1", "barrier", "pass2", "finalize")}
    probe = GlobalVariantProbeSpec(7007, 0, 7)
    diagnostics = {
        "maximum_source_projection_leakage": 0.0,
        "maximum_presymmetry_absolute_error": 0.0,
        "maximum_presymmetry_relative_error": 0.0,
        "block_reconstruction_error": 0.0,
        "same_person_probe_count": probe.probe_count,
        "same_person_cross_tile_finalized": True,
        "minimum_annotation_mass": float(np.min(masses)),
        "minimum_deleted_annotation_mass": float(
            np.min(masses[None, :] - block_masses)
        ),
        "all_values_finite": True,
        "normal_matrix_rank": c,
        "normal_matrix_condition": 1.0,
        "dense_oracle_fixture_version": "stage07_fit_v1",
        "backend_fixed_probe_maximum_error": 0.0,
    }
    return build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe,
        genotype_scale_plan=genotype_scale_plan,
        arrays=arrays,
        pass_ledger=_completed_ledger(block_ids),
        performance_ledger={
            "backend": "fixture",
            "threads": 1,
            "affinity": {},
            "numa_evidence": {},
            "phase_wall_seconds": dict(phases),
            "phase_cpu_seconds": dict(phases),
            "bytes_read": 0,
            "gemm_dimensions": [],
            "peak_rss_bytes": 0,
            "output_bytes": sum(value.nbytes for value in arrays.values()),
        },
        provenance={"fixture": "generalized-fit-current-contract"},
        diagnostics=diagnostics,
    )


def _synthetic_pair(monkeypatch: pytest.MonkeyPatch, *, singular: bool = False):
    _, contextual = _artifacts(monkeypatch, trait_count=1, singular=singular)
    trait = _current_trait(contextual)
    return _matching_reference(
        trait, genotype_scale_plan=contextual.scale_plan
    ), trait


def test_contextual_trait_artifact_is_not_a_generalized_trait_input() -> None:
    contextual = stable_trait_artifact()
    trait = _current_trait(contextual)
    reference = _matching_reference(
        trait, genotype_scale_plan=contextual.scale_plan
    )
    with pytest.raises(ValueError, match="generalized per-variant trait summary"):
        validate_generalized_gxe_trait_compatibility_v1(reference, contextual)


def test_full_fit_matches_direct_dense_symmetric_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    result = fit_generalized_gxe_variant_model_v1(reference, trait)
    expected = np.linalg.solve(
        0.5 * (result.normal_matrix + result.normal_matrix.T),
        result.normal_rhs,
    )
    np.testing.assert_allclose(
        result.raw_coefficients, expected, rtol=2.0e-14, atol=2.0e-14
    )
    assert result.raw_rank == result.normal_matrix.shape[0]
    assert result.manifest["solve"]["ridge"] is False
    assert result.manifest["solve"]["pseudoinverse"] is False


def test_every_deleted_fit_matches_direct_frozen_row_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    result = fit_generalized_gxe_variant_model_v1(reference, trait)
    axes = reference.manifest["axes"]
    component_annotations = np.asarray(
        [entry[0] for entry in axes["components"]["table"]], dtype=np.int64
    )
    full_masses = reference.annotation_masses
    for index, block in enumerate(reference.block_labels):
        retained_masses = full_masses - reference.block_annotation_mass[index]
        directed = (
            reference.directed_numerator
            - reference.block_directed_numerator[index]
        )
        component_masses = retained_masses[component_annotations]
        direct_gram = (
            float(reference.residual_rank**2)
            * 0.5 * (directed + directed.T)
            / np.outer(component_masses, component_masses)
        )
        trait_moments = _trait_moments_after_deleting_blocks(trait, (block,))
        transferred = transfer_reference_gram(
            direct_gram,
            reference.same_person,
            reference_n=reference.n_samples,
            study_n=trait.n_samples,
        )
        c = len(reference.component_index)
        matrix = np.block(
            [
                [transferred, trait_moments.genetic_residual],
                [
                    trait_moments.genetic_residual.T,
                    trait_moments.residual_gram,
                ],
            ]
        )
        rhs = np.concatenate(
            (trait_moments.genetic_rhs[:, 0], trait_moments.residual_rhs[:, 0])
        )
        expected = np.linalg.solve(0.5 * (matrix + matrix.T), rhs)
        np.testing.assert_allclose(
            result.raw_loo_coefficients[index],
            expected,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        equations = assemble_generalized_gxe_normal_equations_v1(
            reference, trait, deleted_blocks=(block,)
        )
        np.testing.assert_array_equal(
            equations.reference_genetic_gram, direct_gram
        )
    assert result.manifest["same_person_jackknife"] == (
        "reuse_full_same_person_v1"
    )


def test_fit_uses_only_compact_summaries_when_panel_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    assert reference.directional_ldscores is None
    result = fit_generalized_gxe_variant_model_v1(reference, trait)
    assert result.manifest["summary_only_fit"] is True
    assert result.manifest["per_variant_panel_accessed"] is False


def test_fit_does_not_invoke_redundant_artifact_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    calls = {"reference": 0, "trait": 0}
    reference_verify = type(reference).verify
    trait_verify = type(trait).verify

    def counted_reference_verify(self) -> None:
        calls["reference"] += 1
        reference_verify(self)

    def counted_trait_verify(self) -> None:
        calls["trait"] += 1
        trait_verify(self)

    monkeypatch.setattr(type(reference), "verify", counted_reference_verify)
    monkeypatch.setattr(type(trait), "verify", counted_trait_verify)
    fit_generalized_gxe_variant_model_v1(reference, trait)
    assert calls == {"reference": 0, "trait": 0}


def test_incompatible_pair_annotation_and_group_axes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(
        trait,
        "group_ids",
        tuple(reversed(trait.group_ids)),
    )
    with pytest.raises(ValueError, match="group_ids"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)

    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(
        trait,
        "component_index",
        ContextComponentIndex(("different",), ContextPairIndex(1)),
    )
    with pytest.raises(ValueError, match="component_order"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)

    reference, trait = _synthetic_pair(monkeypatch)
    object.__setattr__(trait, "annotation_masses", np.asarray([7.0]))
    with pytest.raises(ValueError, match="annotation_masses"):
        validate_generalized_gxe_trait_compatibility_v1(reference, trait)


def test_raw_rank_deficiency_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, trait = _synthetic_pair(monkeypatch, singular=True)
    with pytest.raises(ContextRankError, match="not identifiable"):
        fit_generalized_gxe_variant_model_v1(reference, trait)
