from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from summit import gxeldcore
from summit.context import (
    array_sha256,
    build_context_trait_summary,
    canonical_sha256,
    fit_context_model,
    rank_revealing_projector,
    run_contextual_reference_v1,
)
from summit.context.fit_v1 import fit_contextual_model_v1
from summit.context.trait_v1 import (
    _NATIVE_RESULT_KEYS,
    ContextualTraitPublicationIdentityV1,
    run_contextual_trait_v1,
)

from test_context_stage2_streamed_reference import (
    _assert_close,
    _components,
    _expected_missing_counts,
    _make_case,
    _open_descriptors,
    _run_checkpoint_failure,
    _scale_plan,
)
from test_context_stage3_complete_reference_native import (
    _oracle as _reference_oracle,
    _publication_identity as _reference_publication_identity,
    _stage3_executor,
    _stable_scale_plan,
    _variant_probes,
)


TRAIT_OPERATIONS = (
    "trait_score_tn",
    "trait_feature_projection_tn",
    "trait_feature_projection_nn",
)

ADMISSION_KEYS = {
    "permanent_bytes",
    "residual_phase_bytes",
    "trait_phase_bytes",
    "compact_output_bytes",
    "integrity_reserve_bytes",
    "telemetry_bytes",
    "required_workspace_bytes",
    "required_telemetry_capacity",
    "maximum_protected_output_elements",
    "trait_descriptor_passes",
    "trait_decoded_blocks",
    "total_descriptor_passes",
    "total_decoded_blocks",
    "total_variant_record_visits",
    "total_protected_calls",
    "semantic_call_ledger",
    "selected_tiles",
    "phase_ledger",
    "memory_ledger",
    "memory_lifetimes",
    "memory_accounting_model",
    "os_physical_read_bytes_measured",
    "os_page_faults_measured",
    "physical_io_evidence",
}

SCIENCE_ARRAYS = (
    "genetic_rhs",
    "genetic_traces",
    "genetic_residual",
    "residual_rhs",
    "residual_traces",
    "residual_gram",
    "group_rhs_unnormalized_num",
    "group_trace_unnormalized_num",
    "group_genetic_residual_num",
    "annotation_masses",
    "group_annotation_masses",
    "group_variant_counts",
)


def _trait_inputs(case: Any) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(814_000 + int(case.phi.shape[1]))
    phenotypes = np.asfortranarray(
        rng.normal(size=(case.fixed_basis.shape[0], 3)), dtype=np.float64
    )
    residual_basis = np.asfortranarray(
        np.column_stack(
            [
                np.ones(case.fixed_basis.shape[0], dtype=np.float64),
                np.linspace(-1.0, 1.0, case.fixed_basis.shape[0]),
            ]
        )
    )
    return phenotypes, residual_basis


def _trait_executor(
    case: Any,
    phenotypes: np.ndarray,
    residual_basis: np.ndarray,
    **changes: Any,
) -> Any:
    flags = np.full(
        case.retained_variant_rows.size,
        case.counted_allele_mode == "bim_a1_counted_v1",
        dtype=np.uint8,
    )
    stable_scale = _scale_plan(
        retained_variant_order_sha256=case.retained_variant_order_sha256,
        counted_allele_mode=case.counted_allele_mode,
        centering_source="provided_v1",
        affine_mean_sha256=case.affine_mean_sha256,
        affine_inverse_scale_sha256=case.affine_inverse_scale_sha256,
    )
    options: dict[str, Any] = {
        "annotation_mode": case.annotation_mode,
        "retained_variant_order_sha256": case.retained_variant_order_sha256,
        "affine_mean_sha256": case.affine_mean_sha256,
        "affine_inverse_scale_sha256": case.affine_inverse_scale_sha256,
        "missingness_sha256": case.missingness_sha256,
        "scale_plan_sha256": stable_scale.digest,
        "centering_source": "provided_v1",
    }
    options.update(changes)
    descriptors = _open_descriptors(case.prefix)
    try:
        return gxeldcore.ContextualTraitExecutorV1(
            descriptors[0],
            descriptors[1],
            descriptors[2],
            case.retained_sample_rows,
            case.retained_variant_rows,
            list(case.expected_variant_ids),
            list(case.expected_counted_alleles),
            list(case.expected_other_alleles),
            flags,
            case.affine_mean,
            case.affine_inverse_scale,
            _expected_missing_counts(case),
            case.fixed_basis,
            case.phi,
            case.annotations,
            case.group_index,
            phenotypes,
            residual_basis,
            list(case.annotation_names),
            list(case.group_names),
            [f"trait-{index}" for index in range(phenotypes.shape[1])],
            [f"residual-{index}" for index in range(residual_basis.shape[1])],
            **options,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _oracle_arrays(
    case: Any, phenotypes: np.ndarray, residual_basis: np.ndarray
) -> dict[str, np.ndarray]:
    groups = tuple(case.group_names[int(value)] for value in case.group_index)
    summaries = [
        build_context_trait_summary(
            genotype=case.scaled_genotype,
            basis=case.phi,
            phenotype=phenotypes[:, trait],
            projector=rank_revealing_projector(case.fixed_basis),
            annotations=case.annotations,
            component_index=_components(case),
            residual_basis=residual_basis,
            residual_names=[
                f"residual-{index}" for index in range(residual_basis.shape[1])
            ],
            basis_hash=array_sha256(case.phi),
            fixed_effect_hash=array_sha256(case.fixed_basis),
            variant_hash=canonical_sha256(
                {"logical_variants": list(case.expected_variant_ids)}
            ),
            loo_groups=groups,
            genotype_scaling=_stable_scale_plan(case).digest,
            block_size=4,
            contribution_storage="loo_grouped",
        )
        for trait in range(phenotypes.shape[1])
    ]
    return {
        "genetic_rhs": np.column_stack([summary.genetic_rhs for summary in summaries]),
        "genetic_traces": summaries[0].genetic_traces,
        "genetic_residual": summaries[0].genetic_residual,
        "residual_rhs": np.column_stack(
            [summary.residual_rhs for summary in summaries]
        ),
        "residual_traces": summaries[0].residual_traces,
        "residual_gram": summaries[0].residual_gram,
        "group_rhs_unnormalized_num": np.stack(
            [summary.rhs_numerator_contributions for summary in summaries],
            axis=2,
        ),
        "group_trace_unnormalized_num": (summaries[0].trace_numerator_contributions),
        "group_genetic_residual_num": (
            summaries[0].genetic_residual_numerator_contributions
        ),
        "annotation_masses": summaries[0].annotation_masses,
        "group_annotation_masses": summaries[0].group_annotation_masses,
        "group_variant_counts": summaries[0].group_variant_counts,
    }


def _ceil_div(value: int, width: int) -> int:
    return (value + width - 1) // width


def _assert_one_pass_evidence(
    result: Mapping[str, Any], preflight: Mapping[str, Any], case: Any
) -> None:
    assert set(preflight) == ADMISSION_KEYS
    assert int(preflight["trait_descriptor_passes"]) == 1
    assert int(preflight["total_descriptor_passes"]) == 1
    assert int(preflight["total_variant_record_visits"]) == (
        case.retained_variant_rows.size
    )
    assert int(preflight["trait_decoded_blocks"]) == 3
    assert int(preflight["total_decoded_blocks"]) == 3
    ledger = {
        str(key): int(value)
        for key, value in dict(preflight["semantic_call_ledger"]).items()
    }
    expected_projection_calls = sum(
        case.phi.shape[1] * _ceil_div(width, 5) for width in (4, 4, 3)
    )
    assert ledger["trait_score_tn"] == 3
    assert ledger["trait_feature_projection_tn"] == expected_projection_calls
    assert ledger["trait_feature_projection_nn"] == expected_projection_calls
    assert all(
        value == 0 for name, value in ledger.items() if name not in TRAIT_OPERATIONS
    )
    assert int(preflight["total_protected_calls"]) == sum(ledger.values())

    diagnostics = dict(result["diagnostics"])
    assert int(diagnostics["observed_descriptor_passes"]) == 1
    assert int(diagnostics["observed_decoded_blocks"]) == 3
    assert int(diagnostics["observed_variant_record_visits"]) == (
        case.retained_variant_rows.size
    )
    assert int(diagnostics["observed_phenotype_projections"]) == 3
    assert dict(diagnostics["protected_call_counts"]) == ledger
    for name in (
        "descriptor_accounting_verified",
        "semantic_call_ledger_exact",
        "files_unchanged_at_all_checkpoints",
        "inputs_unchanged_at_all_checkpoints",
        "scratch_released",
        "scratch_released_before_publication",
        "trait_group_reconstruction_verified",
        "phenotype_projection_count_verified",
        "phenotype_normalization_verified",
        "operand_fingerprints_verified",
        "runtime_thread_affinity_fingerprints_verified",
        "independent_scalar_witness_verified",
        "independent_scalar_fallback_available",
    ):
        assert diagnostics[name] is True
    assert float(diagnostics["maximum_projection_leakage"]) <= 2.0e-9
    assert float(diagnostics["phenotype_projection_leakage_max_abs"]) <= 2.0e-9
    assert float(diagnostics["phenotype_normalized_norm_error_max_abs"]) <= 2.0e-9
    assert float(diagnostics["trait_group_reconstruction_max_abs"]) <= 1.0e-9

    telemetry = dict(result["telemetry"])
    assert dict(telemetry["protected_call_counts"]) == ledger
    assert int(telemetry["observed_events"]) == len(telemetry["events"])
    protected = [
        dict(event)
        for event in telemetry["events"]
        if event["event_class"] == "protected_call"
    ]
    assert len(protected) == sum(ledger.values())
    observed = {name: 0 for name in TRAIT_OPERATIONS}
    for event in protected:
        operation = str(event["operation"])
        assert operation in observed
        observed[operation] += 1
        assert int(event["rows"]) > 0
        assert int(event["columns"]) > 0
        assert int(event["reduction"]) > 0
        assert bool(event["transpose_left"]) == operation.endswith("_tn")
        assert str(event["semantic_anchor"]).startswith(operation + "|")
    assert observed == {name: ledger[name] for name in TRAIT_OPERATIONS}


@pytest.mark.parametrize(
    ("q_count", "annotation_mode", "counted_allele_mode"),
    [
        *[
            (q_count, annotation_mode, "bim_a1_counted_v1")
            for annotation_mode in (
                "generic_nonnegative_weights_v1",
                "strict_disjoint_binary_v1",
            )
            for q_count in (1, 2, 3, 4)
        ],
        (4, "generic_nonnegative_weights_v1", "bim_a2_counted_v1"),
    ],
)
def test_native_trait_q1_to_q4_one_pass_matches_python_oracle(
    tmp_path: Path,
    q_count: int,
    annotation_mode: str,
    counted_allele_mode: str,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=q_count,
        annotation_mode=annotation_mode,
        counted_allele_mode=counted_allele_mode,
        name=(f"stage4-trait-q{q_count}-{annotation_mode}-{counted_allele_mode}"),
    )
    phenotypes, residual_basis = _trait_inputs(case)
    executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=4,
        trait_feature_tile=5,
    )
    preflight = dict(executor.preflight())
    result = dict(executor.run())
    expected = _oracle_arrays(case, phenotypes, residual_basis)

    assert set(result) == _NATIVE_RESULT_KEYS
    assert result["complete_trait_artifact"] is False
    assert result["complete_trait_statistics"] is True
    assert result["internal_result_kind"] == "stage4_complete_trait_statistics_v1"
    assert result["state"] == "trait_statistics_complete_ready"
    assert result["lifecycle"] == "trait_statistics_complete_ready"
    for name in SCIENCE_ARRAYS:
        _assert_close(result[name], expected[name], name)
        array = np.asarray(result[name])
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    assert np.asarray(result["genetic_rhs"]).shape == (
        len(_components(case)),
        phenotypes.shape[1],
    )
    assert np.asarray(result["residual_rhs"]).shape == (
        residual_basis.shape[1],
        phenotypes.shape[1],
    )
    assert np.asarray(result["group_rhs_unnormalized_num"]).shape == (
        len(case.group_names),
        len(_components(case)),
        phenotypes.shape[1],
    )
    _assert_one_pass_evidence(result, preflight, case)
    with pytest.raises(RuntimeError, match="one-shot"):
        executor.run()


def test_native_trait_admission_is_exact_and_tiling_is_science_invariant(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=4, name="stage4-trait-admission")
    phenotypes, residual_basis = _trait_inputs(case)
    baseline_executor = _trait_executor(case, phenotypes, residual_basis)
    baseline_preflight = dict(baseline_executor.preflight())
    baseline = dict(baseline_executor.run())
    tiled = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=3,
            trait_feature_tile=2,
        ).run()
    )
    for name in SCIENCE_ARRAYS:
        _assert_close(tiled[name], baseline[name], f"tiled {name}")

    with pytest.raises(RuntimeError, match="workspace cap"):
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            workspace_cap_bytes=int(baseline_preflight["required_workspace_bytes"]) - 1,
        )
    with pytest.raises(RuntimeError, match="telemetry capacity"):
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            telemetry_capacity=int(baseline_preflight["required_telemetry_capacity"])
            - 1,
        )


@pytest.mark.parametrize("operation", TRAIT_OPERATIONS)
@pytest.mark.parametrize("mode", ["one_shot", "repeated"])
def test_native_trait_semantic_fault_recovery_preserves_science(
    tmp_path: Path,
    operation: str,
    mode: str,
) -> None:
    case = _make_case(tmp_path, q_count=3, name=f"stage4-trait-{operation}-{mode}")
    phenotypes, residual_basis = _trait_inputs(case)
    baseline = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=4,
            trait_feature_tile=5,
        ).run()
    )
    anchor_executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        variant_block=4,
        trait_feature_tile=5,
    )
    anchor = next(
        item["semantic_anchor"]
        for item in anchor_executor.semantic_anchors()
        if item["operation"] == operation
    )
    repaired = dict(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=4,
            trait_feature_tile=5,
            fault_operation=operation,
            fault_mode=mode,
            fault_semantic_anchor=anchor,
        ).run()
    )
    for name in SCIENCE_ARRAYS:
        _assert_close(repaired[name], baseline[name], f"repaired {name}")
    telemetry = dict(repaired["telemetry"])
    assert int(telemetry["injection_count"]) == 1
    if mode == "one_shot":
        assert int(telemetry["retry_count"]) == 1
        assert int(telemetry["repair_count"]) == 1
        assert int(telemetry["fallback_count"]) == 0
    else:
        assert int(telemetry["fallback_count"]) == 1


def test_native_trait_post_trait_output_mutation_is_terminal(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, q_count=2, name="stage4-trait-output-mutation")
    phenotypes, residual_basis = _trait_inputs(case)
    executor = _trait_executor(
        case,
        phenotypes,
        residual_basis,
        test_checkpoint="post_trait",
        test_mutation_target="output_buffers",
    )
    error = _run_checkpoint_failure(executor, "post_trait")
    assert "mutation" in str(error).lower()


@pytest.mark.parametrize(
    "annotation_mode",
    ["strict_disjoint_binary_v1", "generic_nonnegative_weights_v1"],
)
def test_native_reference_and_trait_stable_fit_matches_frozen_python_end_to_end(
    tmp_path: Path,
    annotation_mode: str,
) -> None:
    case = _make_case(
        tmp_path,
        q_count=2,
        annotation_mode=annotation_mode,
        name=f"stage4-end-to-end-{annotation_mode}",
    )
    variant_probes = _variant_probes(case)
    reference_publication = _reference_publication_identity(case)
    reference = run_contextual_reference_v1(
        _stage3_executor(case, variant_probes), reference_publication
    )
    phenotypes, residual_basis = _trait_inputs(case)
    trait_publication = ContextualTraitPublicationIdentityV1(
        sample_order_sha256=reference_publication.sample_order_sha256,
        variant_order_allele_sha256=(reference_publication.variant_order_allele_sha256),
        fixed_effect_spec_sha256=(reference_publication.fixed_effect_spec_sha256),
        basis_specification_sha256=(reference_publication.basis_specification_sha256),
        basis_calibration_sha256=(reference_publication.basis_calibration_sha256),
        compatible_reference_identity_sha256=reference.manifest_sha256,
        retained_sample_map_sha256=array_sha256(case.retained_sample_rows),
        retained_variant_order_sha256=case.retained_variant_order_sha256,
        fixed_basis_sha256=array_sha256(case.fixed_basis),
        evaluated_phi_sha256=array_sha256(case.phi),
        genotype_scale_plan_sha256=reference.scale_plan.digest,
        missingness_sha256=case.missingness_sha256,
        annotation_map_sha256=array_sha256(case.annotations),
        annotation_names=case.annotation_names,
        group_map_sha256=array_sha256(case.group_index),
        group_ids=case.group_names,
        phenotype_batch_sha256=array_sha256(phenotypes),
        residual_basis_sha256=array_sha256(residual_basis),
        trait_ids=tuple(f"trait-{index}" for index in range(phenotypes.shape[1])),
        residual_names=tuple(
            f"residual-{index}" for index in range(residual_basis.shape[1])
        ),
    )
    trait = run_contextual_trait_v1(
        _trait_executor(
            case,
            phenotypes,
            residual_basis,
            variant_block=4,
            trait_feature_tile=5,
        ),
        trait_publication,
    )

    groups = tuple(case.group_names[int(value)] for value in case.group_index)
    expected_summary = build_context_trait_summary(
        genotype=case.scaled_genotype,
        basis=case.phi,
        phenotype=phenotypes[:, 0],
        projector=rank_revealing_projector(case.fixed_basis),
        annotations=case.annotations,
        component_index=_components(case),
        residual_basis=residual_basis,
        residual_names=trait.residual_names,
        basis_hash=array_sha256(case.phi),
        fixed_effect_hash=array_sha256(case.fixed_basis),
        variant_hash=canonical_sha256(
            {"logical_variants": list(case.expected_variant_ids)}
        ),
        loo_groups=groups,
        genotype_scaling=_stable_scale_plan(case).digest,
        block_size=4,
        contribution_storage="loo_grouped",
    )
    grid = np.asarray([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    metric = case.phi.T @ case.phi / case.phi.shape[0]
    expected = fit_context_model(
        _reference_oracle(case, variant_probes),
        expected_summary,
        context_grid=grid,
        basis_metric=metric,
        project_psd=False,
    )
    observed = fit_contextual_model_v1(
        reference,
        trait,
        trait_selector="trait-0",
        context_grid=grid,
        basis_metric=metric,
        evaluation_grid_role="non_row_evaluation_grid_v1",
        evaluation_grid_provenance_sha256=canonical_sha256(
            {"fixture": "stage4-native-end-to-end-evaluation-grid-v1"}
        ),
        evaluation_grid_trusted_non_row=True,
        project_psd=False,
    )

    _assert_close(
        observed.raw_coefficients,
        expected.raw_coefficients,
        "stable end-to-end raw coefficients",
    )
    _assert_close(
        observed.raw_loo_coefficients,
        expected.loo_coefficients,
        "stable end-to-end LOO coefficients",
    )
    _assert_close(
        observed.raw_omegas,
        np.asarray(expected.raw_omegas),
        "stable end-to-end raw Omegas",
    )
    full_surfaces = np.stack(
        [
            value["covariance_surface"]
            for value in expected.context_outputs["annotations"]
        ]
    )
    loo_surfaces = np.stack(
        [
            np.stack(
                [value["covariance_surface"] for value in replicate["annotations"]]
            )
            for replicate in expected.jackknife_context_outputs
        ]
    )
    expected_surfaces = np.concatenate((full_surfaces[None, ...], loo_surfaces), axis=0)
    _assert_close(
        observed.raw_covariance_surfaces,
        expected_surfaces,
        "stable end-to-end raw surfaces",
    )
    _assert_close(
        observed.raw_combined_covariance_surfaces,
        np.sum(expected_surfaces, axis=1, dtype=np.float64),
        "stable end-to-end combined surfaces",
    )
