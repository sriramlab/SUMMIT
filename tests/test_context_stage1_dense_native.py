from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pytest

from summit import gxeldcore
from summit.context import (
    AnnotationMode,
    ContextComponentIndex,
    ContextPairIndex,
    GroupedContextReference,
    GroupedContextTraitSummary,
    ProjectorResult,
    annotation_output_contract,
    build_context_reference,
    build_context_trait_summary,
    coefficients_to_omegas,
    context_covariance_surface,
    fit_context_model,
    reference_moments_after_deleting_groups,
    trait_moments_after_deleting_groups,
    transfer_reference_gram,
    transform_omega,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "context_native_stage0"
ATOL = 1.0e-9
RTOL = 1.0e-11
ZERO_HASH = "0" * 64

FIXTURES = (
    "fixture_q1_k2.npz",
    "fixture_q2_k2.npz",
    "fixture_q3_k2.npz",
    "fixture_q4_k3.npz",
)

SEMANTIC_OPERATIONS = (
    "sample_probe_projection_tn",
    "sample_probe_projection_nn",
    "source_tn",
    "full_target_nn",
    "action_projection_tn",
    "action_projection_nn",
    "action_gram_tn",
    "group_target_nn",
    "group_cross_gram_tn",
    "direct_grouped_tn",
    "same_person_target_nn",
    "same_person_projection_tn",
    "same_person_projection_nn",
    "same_person_gram_tn",
    "trait_score_tn",
    "trait_feature_projection_tn",
    "trait_feature_projection_nn",
)

CONTEXTUAL_NATIVE_SOURCE = (
    Path(__file__).parents[1] / "src" / "native" / "contextual_dense_v1.inc"
)

REFERENCE_FIELDS = (
    "gram",
    "same_person",
    "annotation_masses",
    "group_annotation_masses",
    "group_variant_counts",
)

TRAIT_FIELDS = (
    "genetic_rhs",
    "genetic_traces",
    "genetic_residual",
    "residual_rhs",
    "residual_traces",
    "residual_gram",
    "group_rhs_unnormalized_num",
    "group_trace_unnormalized_num",
    "group_genetic_residual_num",
)

SCIENCE_FIELDS = REFERENCE_FIELDS + TRAIT_FIELDS + (
    "raw_gram_numerator",
    "group_gram_numerator",
    "group_gram_numerator_direct_action_scaled",
    "group_gram_numerator_direct_genotype_scaled",
    "group_gram_numerator_restricted",
)


def _assert_close(actual: object, expected: object, label: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=RTOL,
        atol=ATOL,
        err_msg=label,
    )


def _load_fixture(filename: str) -> dict[str, Any]:
    with np.load(FIXTURE_ROOT / filename, allow_pickle=False) as fixture:
        case = {name: np.array(fixture[name], copy=True) for name in fixture.files}
    group_count = int(np.max(case["group_index"], initial=-1)) + 1
    case["annotation_names"] = tuple(
        f"a{index}" for index in range(case["annotations"].shape[1])
    )
    case["group_names"] = tuple(f"g{index}" for index in range(group_count))
    return case


def _projector_from_basis(fixed_basis: np.ndarray) -> ProjectorResult:
    fixed = np.asarray(fixed_basis, dtype=np.float64)
    n_samples, rank = fixed.shape
    projector = np.eye(n_samples, dtype=np.float64) - fixed @ fixed.T
    return ProjectorResult(
        projector=projector,
        fixed_basis=fixed,
        rank=rank,
        residual_rank=n_samples - rank,
        singular_values=np.ones(rank, dtype=np.float64),
        tolerance=0.0,
        maximum_leverage=float(np.max(np.sum(fixed * fixed, axis=1), initial=0.0)),
    )


def _groups(case: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(case["group_names"])
    return tuple(names[int(index)] for index in case["group_index"])


def _components(case: Mapping[str, Any]) -> ContextComponentIndex:
    return ContextComponentIndex(
        tuple(case["annotation_names"]),
        ContextPairIndex(int(np.asarray(case["phi"]).shape[1])),
    )


def _oracle_artifacts(
    case: Mapping[str, Any],
) -> tuple[GroupedContextReference, GroupedContextTraitSummary]:
    components = _components(case)
    projector = _projector_from_basis(np.asarray(case["fixed_basis"]))
    common = {
        "genotype": case["genotype"],
        "basis": case["phi"],
        "projector": projector,
        "annotations": case["annotations"],
        "component_index": components,
        "basis_hash": ZERO_HASH,
        "fixed_effect_hash": ZERO_HASH,
        "variant_hash": ZERO_HASH,
        "loo_groups": _groups(case),
        "genotype_scaling": "pre_scaled_input",
    }
    reference = build_context_reference(
        **common,
        gram_method="hutchinson",
        gram_probes=case["sample_probes"],
        same_person_method="ustat",
        variant_probes=case["variant_probes"],
        probe_tile_size=min(3, np.asarray(case["variant_probes"]).shape[1]),
        contribution_storage="loo_grouped",
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=case["phenotype_raw"],
        residual_basis=case["residual_basis"],
        residual_names=tuple(
            f"residual:{index}"
            for index in range(np.asarray(case["residual_basis"]).shape[1])
        ),
        block_size=min(13, np.asarray(case["genotype"]).shape[1]),
        contribution_storage="loo_grouped",
    )
    assert isinstance(reference, GroupedContextReference)
    assert isinstance(summary, GroupedContextTraitSummary)
    return reference, summary


def _executor(case: Mapping[str, Any], **policies: Any) -> Any:
    return gxeldcore.ContextualBlockExecutorV1(
        np.asfortranarray(case["genotype"], dtype=np.float64),
        np.asfortranarray(case["fixed_basis"], dtype=np.float64),
        np.asfortranarray(case["phi"], dtype=np.float64),
        np.asfortranarray(case["annotations"], dtype=np.float64),
        np.ascontiguousarray(case["group_index"], dtype=np.int64),
        np.asfortranarray(case["sample_probes"], dtype=np.float64),
        np.asfortranarray(case["variant_probes"], dtype=np.float64),
        np.ascontiguousarray(case["phenotype_raw"], dtype=np.float64),
        np.asfortranarray(case["residual_basis"], dtype=np.float64),
        list(case["annotation_names"]),
        list(case["group_names"]),
        **policies,
    )


def _native_artifacts(
    result: Mapping[str, Any],
    oracle_reference: GroupedContextReference,
    oracle_summary: GroupedContextTraitSummary,
) -> tuple[GroupedContextReference, GroupedContextTraitSummary]:
    reference = replace(
        oracle_reference,
        annotation_masses=result["annotation_masses"],
        gram=result["gram"],
        same_person=result["same_person"],
        gram_numerator_contributions=result["group_gram_numerator"],
        group_annotation_masses=result["group_annotation_masses"],
        group_variant_counts=result["group_variant_counts"],
    )
    summary = replace(
        oracle_summary,
        annotation_masses=result["annotation_masses"],
        genetic_rhs=result["genetic_rhs"],
        genetic_traces=result["genetic_traces"],
        genetic_residual=result["genetic_residual"],
        residual_rhs=result["residual_rhs"],
        residual_traces=result["residual_traces"],
        residual_gram=result["residual_gram"],
        rhs_numerator_contributions=result["group_rhs_unnormalized_num"],
        trace_numerator_contributions=result["group_trace_unnormalized_num"],
        genetic_residual_numerator_contributions=(
            result["group_genetic_residual_num"]
        ),
        group_annotation_masses=result["group_annotation_masses"],
        group_variant_counts=result["group_variant_counts"],
    )
    return reference, summary


def _assert_maps(result: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    components = _components(case)
    expected = {
        "pair_q": [entry.q for entry in components.pair_index.entries],
        "pair_r": [entry.r for entry in components.pair_index.entries],
        "pair_eta": [entry.kernel_factor for entry in components.pair_index.entries],
        "component_annotation": [
            entry.annotation_index for entry in components.entries
        ],
        "component_pair": [entry.pair_index for entry in components.entries],
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(result[name], values, err_msg=name)


def _assert_native_matches_oracle(
    result: Mapping[str, Any],
    reference: GroupedContextReference,
    summary: GroupedContextTraitSummary,
) -> None:
    expected = {
        "gram": reference.gram,
        "same_person": reference.same_person,
        "annotation_masses": reference.annotation_masses,
        "group_annotation_masses": reference.group_annotation_masses,
        "group_variant_counts": reference.group_variant_counts,
        "group_gram_numerator": reference.gram_numerator_contributions,
        "genetic_rhs": summary.genetic_rhs,
        "genetic_traces": summary.genetic_traces,
        "genetic_residual": summary.genetic_residual,
        "residual_rhs": summary.residual_rhs,
        "residual_traces": summary.residual_traces,
        "residual_gram": summary.residual_gram,
        "group_rhs_unnormalized_num": summary.rhs_numerator_contributions,
        "group_trace_unnormalized_num": summary.trace_numerator_contributions,
        "group_genetic_residual_num": (
            summary.genetic_residual_numerator_contributions
        ),
    }
    for name, value in expected.items():
        _assert_close(result[name], value, name)

    for name in (
        "group_gram_numerator_direct_action_scaled",
        "group_gram_numerator_direct_genotype_scaled",
        "group_gram_numerator_restricted",
    ):
        _assert_close(result[name], reference.gram_numerator_contributions, name)

    component_annotation = np.asarray(result["component_annotation"], dtype=np.int64)
    component_masses = np.asarray(result["annotation_masses"])[component_annotation]
    expected_raw = np.asarray(result["gram"]) * np.outer(
        component_masses, component_masses
    )
    _assert_close(result["raw_gram_numerator"], expected_raw, "raw Gram numerator")
    reconstructed = np.sum(
        np.asarray(result["group_gram_numerator"]), axis=0, dtype=np.float64
    ) / np.outer(component_masses, component_masses)
    _assert_close(reconstructed, result["gram"], "group reconstruction")


def _assert_native_matches_frozen(
    result: Mapping[str, Any], case: Mapping[str, Any]
) -> None:
    fixture_fields = {
        "gram": "hutch_gram",
        "same_person": "ustat_same_person",
        "annotation_masses": "annotation_masses",
        "group_annotation_masses": "group_annotation_masses",
        "group_variant_counts": "group_variant_counts",
        "genetic_rhs": "trait_genetic_rhs",
        "genetic_traces": "trait_genetic_traces",
        "genetic_residual": "trait_genetic_residual",
        "residual_rhs": "residual_rhs",
        "residual_traces": "residual_traces",
        "residual_gram": "residual_gram",
        "group_rhs_unnormalized_num": "group_rhs_unnormalized_num",
        "group_trace_unnormalized_num": "group_trace_unnormalized_num",
        "group_genetic_residual_num": "group_genetic_residual_num",
    }
    for native_name, fixture_name in fixture_fields.items():
        _assert_close(result[native_name], case[fixture_name], native_name)
    for name in (
        "group_gram_numerator",
        "group_gram_numerator_direct_action_scaled",
        "group_gram_numerator_direct_genotype_scaled",
        "group_gram_numerator_restricted",
    ):
        _assert_close(result[name], case["group_gram_unnormalized_num"], name)
    component_masses = np.asarray(case["component_masses"])
    _assert_close(
        result["raw_gram_numerator"],
        np.asarray(case["hutch_gram"])
        * np.outer(component_masses, component_masses),
        "raw Gram numerator",
    )


def _walk_arrays(value: Any, path: str = "result") -> Iterator[tuple[str, np.ndarray]]:
    if isinstance(value, np.ndarray):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_arrays(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_arrays(item, f"{path}[{index}]")


def _assert_compact(result: Mapping[str, Any], n_samples: int, n_variants: int) -> None:
    for path, value in _walk_arrays(result):
        assert n_samples not in value.shape, f"returned N axis at {path}: {value.shape}"
        assert n_variants not in value.shape, (
            f"returned M axis at {path}: {value.shape}"
        )


@pytest.mark.parametrize("filename", FIXTURES)
def test_fixed_q1_q4_native_moments_deletions_and_final_fits(filename: str) -> None:
    case = _load_fixture(filename)
    oracle_reference, oracle_summary = _oracle_artifacts(case)
    executor = _executor(case)
    preflight = dict(executor.preflight())
    info = dict(executor.info())
    assert info["lifecycle"] == "admitted"
    assert dict(info["admission"]) == preflight

    result = dict(executor.run())
    post_info = dict(executor.info())
    assert post_info["lifecycle"] == "published"
    _assert_maps(result, case)
    _assert_native_matches_oracle(result, oracle_reference, oracle_summary)
    _assert_compact(
        result,
        np.asarray(case["genotype"]).shape[0],
        np.asarray(case["genotype"]).shape[1],
    )

    _assert_native_matches_frozen(result, case)
    assert int(result["reference_n"]) == np.asarray(case["genotype"]).shape[0]
    assert int(result["residual_rank"]) == (
        np.asarray(case["genotype"]).shape[0]
        - np.asarray(case["fixed_basis"]).shape[1]
    )
    assert tuple(result["annotation_names"]) == tuple(case["annotation_names"])
    assert tuple(result["group_names"]) == tuple(case["group_names"])
    assert result["annotation_mode"] == "generic_nonnegative_weights_v1"
    assert result["genotype_scale_policy"] == "pre_scaled_dense_v1"
    assert result["numeric_policy"] == "fp64_v1"
    assert result["deletion_policy"] == "approximate_summary_only_v1"
    assert result["grouped_encoding"] == "grouped_unnormalized_dense_v1"

    native_reference, native_summary = _native_artifacts(
        result, oracle_reference, oracle_summary
    )
    for artifact in (native_reference, native_summary):
        for _, value in vars(artifact).items():
            if isinstance(value, np.ndarray):
                assert value.flags.owndata
                assert not value.flags.writeable
    study_n = int(case["transferred_study_n"])
    _assert_close(
        transfer_reference_gram(
            native_reference.gram,
            native_reference.same_person,
            reference_n=native_reference.n_samples,
            study_n=study_n,
        ),
        case["transferred_gram"],
        "transferred normal system",
    )

    for group_position, group in enumerate(native_reference.loo_group_ids):
        deleted_reference = reference_moments_after_deleting_groups(
            native_reference, (group,)
        )
        deleted_trait = trait_moments_after_deleting_groups(native_summary, (group,))
        _assert_close(
            deleted_reference.gram,
            case["deletion_reference_grams"][group_position],
            f"deletion Gram {group}",
        )
        _assert_close(
            deleted_trait.genetic_rhs,
            case["deletion_trait_rhs"][group_position],
            f"deletion RHS {group}",
        )

    fit = fit_context_model(
        native_reference,
        native_summary,
        project_psd=False,
        context_grid=case["phi"],
    )
    _assert_close(fit.raw_coefficients, case["raw_coefficients"], "raw fit")
    _assert_close(fit.raw_omegas, case["raw_omegas"], "raw Omegas")
    _assert_close(
        fit.loo_coefficients,
        case["deletion_coefficients"],
        "all delete-group fits",
    )
    assert fit.context_outputs is not None
    for annotation_index, output in enumerate(fit.context_outputs["annotations"]):
        expected_surface = context_covariance_surface(
            case["raw_omegas"][annotation_index], case["phi"]
        )
        _assert_close(output["covariance_surface"], expected_surface, "surface")
    for group_position, output in enumerate(fit.jackknife_context_outputs):
        expected_omegas = coefficients_to_omegas(
            case["deletion_coefficients"][
                group_position, : len(native_reference.component_index)
            ],
            native_reference.component_index,
        )
        for annotation_index, annotation_output in enumerate(
            output["annotations"]
        ):
            expected_surface = context_covariance_surface(
                expected_omegas[annotation_index], case["phi"]
            )
            _assert_close(
                annotation_output["covariance_surface"],
                expected_surface,
                f"delete-group surface {group_position}:{annotation_index}",
            )

    units = dict(result["units"])
    assert "raw" in str(units["group_gram_numerator"])
    assert "normalized" in str(units["gram"])
    diagnostics = dict(result["diagnostics"])
    assert int(diagnostics["decode_passes"]) == 1
    assert int(diagnostics["descriptor_decode_passes"]) == 0
    assert int(diagnostics["trait_logical_passes"]) == 1
    assert int(diagnostics["trait_variant_visits"]) == np.asarray(
        case["genotype"]
    ).shape[1]
    assert int(diagnostics["sample_probe_coverage"]) == np.asarray(
        case["sample_probes"]
    ).shape[1]
    assert int(diagnostics["variant_probe_coverage"]) == np.asarray(
        case["variant_probes"]
    ).shape[1]
    assert float(diagnostics["maximum_projection_leakage"]) <= 2.0e-9
    assert float(diagnostics["gram_pre_symmetry_max_abs"]) <= ATOL
    assert float(diagnostics["same_person_pre_symmetry_max_abs"]) <= ATOL
    assert float(diagnostics["residual_gram_pre_symmetry_max_abs"]) <= ATOL
    assert dict(diagnostics["protected_call_counts"]) == dict(
        diagnostics["expected_protected_call_counts"]
    )
    assert bool(diagnostics["group_reconstruction_verified"])
    assert bool(diagnostics["expected_calls_verified"])
    assert bool(diagnostics["scratch_released"])


def _random_case(q_count: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_samples, n_variants, k_count = 32, 40, 2
    genotype = rng.normal(size=(n_samples, n_variants))
    genotype -= genotype.mean(axis=0, keepdims=True)
    genotype /= genotype.std(axis=0, ddof=1, keepdims=True)
    fixed = np.column_stack([np.ones(n_samples), rng.normal(size=n_samples)])
    fixed_basis = np.linalg.qr(fixed, mode="reduced")[0]
    phi = rng.normal(size=(n_samples, q_count))
    if q_count % 2:
        annotations = np.zeros((n_variants, k_count), dtype=np.float64)
        annotations[np.arange(n_variants), np.arange(n_variants) % k_count] = 1.0
        mode = "strict_disjoint_binary_v1"
    else:
        annotations = rng.uniform(0.05, 1.0, size=(n_variants, k_count))
        mode = "generic_nonnegative_weights_v1"
    group_index = np.arange(n_variants, dtype=np.int64) % 4
    sample_probes = (
        2 * rng.integers(0, 2, size=(n_samples, 7)) - 1
    ).astype(np.float64)
    variant_probes = (
        2 * rng.integers(0, 2, size=(n_variants, 7)) - 1
    ).astype(np.float64)
    return {
        "genotype": genotype,
        "fixed_basis": fixed_basis,
        "phi": phi,
        "annotations": annotations,
        "group_index": group_index,
        "sample_probes": sample_probes,
        "variant_probes": variant_probes,
        "phenotype_raw": rng.normal(size=n_samples),
        "residual_basis": np.column_stack(
            [np.ones(n_samples), rng.uniform(0.2, 1.5, size=n_samples)]
        ),
        "annotation_names": ("a0", "a1"),
        "group_names": ("g0", "g1", "g2", "g3"),
        "annotation_mode": mode,
    }


@pytest.mark.parametrize("q_count", [1, 2, 3, 4, 5])
def test_seeded_random_q1_q5_matches_active_python_builders(q_count: int) -> None:
    case = _random_case(q_count, seed=71100 + q_count)
    oracle_reference, oracle_summary = _oracle_artifacts(case)
    result = dict(
        _executor(case, annotation_mode=case["annotation_mode"], variant_block=9).run()
    )
    _assert_maps(result, case)
    _assert_native_matches_oracle(result, oracle_reference, oracle_summary)

    native_reference, native_summary = _native_artifacts(
        result, oracle_reference, oracle_summary
    )
    oracle_fit = fit_context_model(oracle_reference, oracle_summary)
    native_fit = fit_context_model(native_reference, native_summary)
    _assert_close(
        native_fit.raw_coefficients, oracle_fit.raw_coefficients, "random fit"
    )
    _assert_close(native_fit.raw_omegas, oracle_fit.raw_omegas, "random Omegas")
    _assert_close(
        native_fit.loo_coefficients,
        oracle_fit.loo_coefficients,
        "random deletion fits",
    )
    if case["annotation_mode"] == "generic_nonnegative_weights_v1":
        interpretation = annotation_output_contract(
            AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1
        )
        assert not interpretation["standalone_annotation_covariance_allowed"]
        assert interpretation["combined_total_surface_required"]


def test_all_one_and_full_tiles_are_invariant() -> None:
    case = _load_fixture("fixture_q2_k2.npz")
    full = dict(_executor(case).run())
    tiled = dict(
        _executor(
            case,
            variant_block=1,
            sample_probe_tile=1,
            variant_probe_tile=1,
            action_tile=1,
            annotation_tile=1,
            context_tile=1,
            group_tile=1,
            trait_feature_tile=1,
        ).run()
    )
    for name in SCIENCE_FIELDS:
        _assert_close(tiled[name], full[name], f"one/full tile {name}")
    full_counts = dict(full["telemetry"])["protected_call_counts"]
    tiled_counts = dict(tiled["telemetry"])["protected_call_counts"]
    assert dict(full_counts) != dict(tiled_counts)
    assert sum(dict(tiled_counts).values()) > sum(dict(full_counts).values())


@pytest.mark.parametrize(
    "policy",
    [
        "variant_block",
        "sample_probe_tile",
        "variant_probe_tile",
        "action_tile",
        "annotation_tile",
        "context_tile",
        "group_tile",
        "trait_feature_tile",
    ],
)
def test_each_tile_knob_is_independently_invariant(policy: str) -> None:
    case = _load_fixture("fixture_q2_k2.npz")
    full = dict(_executor(case).run())
    executor = _executor(case, **{policy: 1})
    preflight = dict(executor.preflight())
    assert int(preflight[policy]) == 1
    tiled = dict(executor.run())
    for name in SCIENCE_FIELDS:
        _assert_close(tiled[name], full[name], f"{policy}=1 {name}")
    expected = dict(preflight["semantic_call_ledger"])
    observed = dict(dict(tiled["telemetry"])["protected_call_counts"])
    assert observed == expected
    if policy != "group_tile":
        full_calls = dict(dict(full["telemetry"])["protected_call_counts"])
        assert observed != full_calls


def test_strict_disjoint_and_generic_binary_paths_are_identical() -> None:
    case = _load_fixture("fixture_q2_k2.npz")
    generic = dict(
        _executor(case, annotation_mode="generic_nonnegative_weights_v1").run()
    )
    strict = dict(
        _executor(case, annotation_mode="strict_disjoint_binary_v1").run()
    )
    for name in SCIENCE_FIELDS:
        _assert_close(strict[name], generic[name], f"strict/generic {name}")
    assert not bool(dict(generic["diagnostics"])["strict_disjoint_optimized_path"])
    assert bool(dict(strict["diagnostics"])["strict_disjoint_optimized_path"])


def test_each_grouped_algorithm_and_tn_placement_survives_deletion_fits() -> None:
    case = _load_fixture("fixture_q2_k2.npz")
    oracle_reference, oracle_summary = _oracle_artifacts(case)
    result = dict(_executor(case).run())
    _, native_summary = _native_artifacts(result, oracle_reference, oracle_summary)
    for group_field in (
        "group_gram_numerator_direct_action_scaled",
        "group_gram_numerator_direct_genotype_scaled",
        "group_gram_numerator_restricted",
    ):
        reference = replace(
            oracle_reference,
            annotation_masses=result["annotation_masses"],
            gram=result["gram"],
            same_person=result["same_person"],
            gram_numerator_contributions=result[group_field],
            group_annotation_masses=result["group_annotation_masses"],
            group_variant_counts=result["group_variant_counts"],
        )
        fit = fit_context_model(reference, native_summary, project_psd=False)
        _assert_close(fit.raw_coefficients, case["raw_coefficients"], group_field)
        _assert_close(fit.raw_omegas, case["raw_omegas"], group_field)
        _assert_close(
            fit.loo_coefficients,
            case["deletion_coefficients"],
            f"{group_field} deletion fits",
        )


def test_q3_nonorthogonal_reparameterization_is_covariant() -> None:
    case = _random_case(3, seed=81203)
    case["annotation_mode"] = "strict_disjoint_binary_v1"
    transform = np.asarray(
        [[1.0, 0.3, -0.2], [0.2, 1.2, 0.4], [-0.1, 0.25, 0.9]],
        dtype=np.float64,
    )
    assert abs(np.linalg.det(transform)) > 0.5
    transformed = dict(case)
    transformed["phi"] = np.asarray(case["phi"]) @ transform.T

    base_oracle = _oracle_artifacts(case)
    transformed_oracle = _oracle_artifacts(transformed)
    base_result = dict(
        _executor(case, annotation_mode="strict_disjoint_binary_v1").run()
    )
    transformed_result = dict(
        _executor(transformed, annotation_mode="strict_disjoint_binary_v1").run()
    )
    _assert_native_matches_oracle(transformed_result, *transformed_oracle)
    base_artifacts = _native_artifacts(base_result, *base_oracle)
    transformed_artifacts = _native_artifacts(
        transformed_result, *transformed_oracle
    )
    base_fit = fit_context_model(*base_artifacts)
    transformed_fit = fit_context_model(*transformed_artifacts)
    for annotation_index, omega in enumerate(base_fit.raw_omegas):
        transformed_omega = transformed_fit.raw_omegas[annotation_index]
        _assert_close(
            transformed_omega,
            transform_omega(omega, transform),
            "Q3 transformed Omega",
        )
        _assert_close(
            context_covariance_surface(transformed_omega, transformed["phi"]),
            context_covariance_surface(omega, case["phi"]),
            "Q3 invariant covariance surface",
        )


def test_two_variant_probes_preserve_global_cross_tile_ustatistic() -> None:
    case = _load_fixture("fixture_q2_k2.npz")
    case["variant_probes"] = np.asarray(case["variant_probes"])[:, :2].copy()
    oracle_reference, oracle_summary = _oracle_artifacts(case)
    result = dict(_executor(case, variant_probe_tile=1).run())
    _assert_native_matches_oracle(result, oracle_reference, oracle_summary)


@pytest.mark.parametrize("operation", SEMANTIC_OPERATIONS)
def test_targeted_one_shot_corruption_repairs_every_semantic_operation(
    operation: str,
) -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    result = dict(
        _executor(
            case,
            fault_operation=operation,
            fault_mode="one_shot",
            fault_occurrence=1,
        ).run()
    )
    _assert_native_matches_frozen(result, case)
    telemetry = dict(result["telemetry"])
    assert int(telemetry["injection_count"]) == 1
    assert int(telemetry["retry_count"]) >= 1
    assert int(dict(telemetry["protected_call_counts"])[operation]) > 0


@pytest.mark.parametrize(
    "mode", ["repeated", "force_fallback", "nan", "inf", "canary"]
)
def test_repeated_and_detected_fault_modes_use_preallocated_recovery(mode: str) -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    result = dict(
        _executor(
            case,
            fault_operation="source_tn",
            fault_mode=mode,
            fault_occurrence=1,
        ).run()
    )
    _assert_close(result["gram"], case["hutch_gram"], mode)
    telemetry = dict(result["telemetry"])
    assert int(telemetry["injection_count"]) == 1
    expects_fallback = mode in {"repeated", "force_fallback", "canary"}
    assert int(telemetry["fallback_count"]) >= int(expects_fallback)


@pytest.mark.parametrize(
    "mode", ["repair_corruption", "fallback_corruption", "operand_mutation"]
)
def test_terminal_integrity_faults_fail_closed(mode: str) -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    with pytest.raises(RuntimeError, match="ContextualBlockExecutorV1 integrity"):
        _executor(
            case,
            fault_operation="source_tn",
            fault_mode=mode,
            fault_occurrence=1,
        ).run()


def test_unknown_and_unconsumed_fault_targets_fail_closed() -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    with pytest.raises((RuntimeError, ValueError), match="validation:.*fault"):
        _executor(case, fault_operation="not_a_semantic_operation")
    with pytest.raises(RuntimeError, match="integrity:.*fault"):
        _executor(
            case,
            fault_operation="source_tn",
            fault_mode="one_shot",
            fault_occurrence=10**9,
        ).run()


def test_executor_seals_caller_inputs_and_run_is_one_shot() -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    genotype = np.asfortranarray(case["genotype"], dtype=np.float64)
    fixed_basis = np.asfortranarray(case["fixed_basis"], dtype=np.float64)
    phi = np.asfortranarray(case["phi"], dtype=np.float64)
    annotations = np.asfortranarray(case["annotations"], dtype=np.float64)
    group_index = np.ascontiguousarray(case["group_index"], dtype=np.int64)
    sample_probes = np.asfortranarray(case["sample_probes"], dtype=np.float64)
    variant_probes = np.asfortranarray(case["variant_probes"], dtype=np.float64)
    phenotype = np.ascontiguousarray(case["phenotype_raw"], dtype=np.float64)
    residual_basis = np.asfortranarray(case["residual_basis"], dtype=np.float64)
    annotation_names = list(case["annotation_names"])
    group_names = list(case["group_names"])
    executor = gxeldcore.ContextualBlockExecutorV1(
        genotype,
        fixed_basis,
        phi,
        annotations,
        group_index,
        sample_probes,
        variant_probes,
        phenotype,
        residual_basis,
        annotation_names,
        group_names,
    )
    for value in (
        genotype,
        fixed_basis,
        phi,
        annotations,
        group_index,
        sample_probes,
        variant_probes,
        phenotype,
        residual_basis,
    ):
        value[...] = 0
    annotation_names.append("caller mutation")
    group_names.append("caller mutation")

    result = dict(executor.run())
    _assert_native_matches_frozen(result, case)
    assert tuple(result["annotation_names"]) == tuple(case["annotation_names"])
    assert tuple(result["group_names"]) == tuple(case["group_names"])
    with pytest.raises(RuntimeError, match="lifecycle: run is one-shot"):
        executor.run()


def test_exact_and_cap_minus_one_workspace_and_telemetry_admission() -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    preflight = dict(_executor(case).preflight())
    required_workspace = int(preflight["required_workspace_bytes"])
    required_telemetry = int(preflight["required_telemetry_capacity"])
    assert required_workspace > 0
    assert required_telemetry > 0

    exact = _executor(
        case,
        workspace_cap_bytes=required_workspace,
        telemetry_capacity=required_telemetry,
    )
    _assert_close(exact.run()["gram"], case["hutch_gram"], "exact cap")
    with pytest.raises(RuntimeError, match="admission: workspace cap"):
        _executor(case, workspace_cap_bytes=required_workspace - 1)
    with pytest.raises(RuntimeError, match="admission: telemetry capacity"):
        _executor(case, telemetry_capacity=required_telemetry - 1)


def test_recovery_telemetry_uses_its_fault_enabled_exact_reserve() -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    fault = {
        "fault_operation": "source_tn",
        "fault_mode": "repeated",
        "fault_occurrence": 1,
    }
    preflight = dict(_executor(case, **fault).preflight())
    required = int(preflight["required_telemetry_capacity"])
    assert required == 3 * int(preflight["total_protected_calls"]) + 10
    result = dict(_executor(case, telemetry_capacity=required, **fault).run())
    telemetry = dict(result["telemetry"])
    assert int(telemetry["observed_events"]) <= required
    assert int(telemetry["fallback_count"]) == 1
    assert bool(telemetry["complete_without_drop"])
    with pytest.raises(RuntimeError, match="admission: telemetry capacity"):
        _executor(case, telemetry_capacity=required - 1, **fault)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda case: case["annotations"].__setitem__((0, 0), -1.0), "annotation"),
        (lambda case: case["group_index"].__setitem__(0, 99), "group"),
        (lambda case: case["sample_probes"].__setitem__((0, 0), 0.0), "probe"),
        (lambda case: case["fixed_basis"].__setitem__((0, 0), 3.0), "orthonormal"),
        (lambda case: case["annotation_names"].__setitem__(1, "a0"), "annotation"),
    ],
)
def test_validation_rejects_noncanonical_or_invalid_inputs(
    mutation: Any, message: str
) -> None:
    case = _load_fixture("fixture_q1_k2.npz")
    case["annotation_names"] = list(case["annotation_names"])
    mutation(case)
    with pytest.raises((RuntimeError, ValueError), match=f"validation:.*{message}"):
        _executor(case)


def test_strict_mode_rejects_overlapping_generic_weights() -> None:
    case = _random_case(2, seed=99102)
    with pytest.raises((RuntimeError, ValueError), match="validation:.*strict"):
        _executor(case, annotation_mode="strict_disjoint_binary_v1")


def test_checked_layout_arithmetic_rejects_u64_overflow() -> None:
    maximum = np.iinfo(np.uint64).max
    with pytest.raises((OverflowError, RuntimeError), match="overflow"):
        gxeldcore._test_contextual_checked_layout_overflow(maximum, maximum, 8)


def test_contextual_native_source_keeps_the_frozen_reuse_boundary() -> None:
    assert tuple(gxeldcore.contextual_dense_semantic_operations_v1()) == (
        SEMANTIC_OPERATIONS
    )
    source = CONTEXTUAL_NATIVE_SOURCE.read_text(encoding="utf-8")
    for forbidden in (
        "dgemm_nn_raw(",
        "dgemm_tn_raw(",
        "cblas_dgemm(",
        "DirectContext",
        "MultiEnvironmentKernel",
        "MultiEnvironmentDirectContext",
        "scale_x",
        "scale_w",
    ):
        assert forbidden not in source
    for approved in (
        "dgemm_nn_partitioned_rows(",
        "dgemm_tn_partitioned_rows(",
        "dgemm_nn_tiled(",
        "dgemm_tn_tiled(",
    ):
        assert approved in source
