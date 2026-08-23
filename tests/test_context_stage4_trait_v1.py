from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from summit.context.schema import (
    AnnotationMode,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
)
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
)
from summit.context.trait_v1 import (
    CONTEXTUAL_TRAIT_V1_SUFFIX,
    ContextualTraitPublicationIdentityV1,
    adapt_native_contextual_trait_v1,
    load_contextual_trait_v1,
    run_contextual_trait_v1,
    trait_moments_after_deleting_groups_v1,
    write_contextual_trait_v1,
)
from test_context_stage3_reference_v1 import (
    _file_content_identity,
    _native_build_provenance,
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


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _refresh_trait_scientific_digests(result: dict[str, Any]) -> None:
    names = (
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
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
    )
    result["scientific_array_sha256"] = {
        name: array_sha256(result[name]) for name in names
    }


def _event(
    event_class: str,
    *,
    operation: str = "none",
    resolution: str,
    phase: str,
    protected: bool = False,
    witness: bool = False,
) -> dict[str, Any]:
    attempt_event = operation != "none" and event_class in {
        "protected_call",
        "repair",
        "retry",
        "trusted_fallback",
        "semantic_verification",
    }
    semantic_phase = "trait" if operation != "none" else "unspecified"
    is_feature_projection = operation in {
        "trait_feature_projection_tn",
        "trait_feature_projection_nn",
    }
    context_end = int(is_feature_projection)
    variant_end = int(is_feature_projection)
    action_end = int(is_feature_projection)
    canonical_end = int(is_feature_projection)
    anchor = (
        f"{operation}|resident=0:0|probe=0:0|variant=0:{variant_end}"
        "|annotation=0:0"
        f"|context=0:{context_end}|action=0:{action_end}"
        f"|phase={semantic_phase}|role=ordinary|placement=none"
        f"|canonical=0:{canonical_end}"
    )
    return {
        "event_class": event_class,
        "phase": phase,
        "operation": operation,
        "semantic_anchor": anchor,
        "semantic_phase": semantic_phase,
        "semantic_role": "ordinary",
        "semantic_placement": "none",
        "canonical_begin": 0,
        "canonical_end": canonical_end,
        "variant_coordinate_mode": "logical_half_open_range_v1",
        "variant_range_is_exact": True,
        "variant_membership_count": 0,
        "variant_membership_sha256": "",
        "resolution": resolution,
        "attempt": 1 if protected or witness else 0,
        "rows": 1 if attempt_event else 0,
        "columns": 1 if attempt_event else 0,
        "reduction": 1 if attempt_event else 0,
        "left_stride": 1 if attempt_event else 0,
        "right_stride": 1 if attempt_event else 0,
        "output_stride": 1 if attempt_event else 0,
        "transpose_left": attempt_event and operation.endswith("_tn"),
        "sequence": 0,
        "process_id": 1,
        "thread_id": 1,
        "elapsed_ns": 0,
        "operand_fingerprint_fnv64": "1" if attempt_event else "0",
        "witness_fingerprint_fnv64": "2" if attempt_event else "0",
        "accepted_output_fingerprint_fnv64": "3" if attempt_event else "0",
        "runtime_before_fingerprint_fnv64": "4" if attempt_event else "0",
        "runtime_after_fingerprint_fnv64": "4" if attempt_event else "0",
        "prefix_canary_verified": attempt_event,
        "suffix_canary_verified": attempt_event,
        "finiteness_verified": attempt_event,
        "serialized_entry_verified": attempt_event,
        "deterministic_non_vendor_backend": attempt_event,
        "resident_begin": 0,
        "resident_end": 0,
        "probe_begin": 0,
        "probe_end": 0,
        "variant_begin": 0,
        "variant_end": variant_end,
        "annotation_begin": 0,
        "annotation_end": 0,
        "context_begin": 0,
        "context_end": context_end,
        "action_begin": 0,
        "action_end": action_end,
        "group_begin": 0,
        "group_end": 0,
    }


def _execution_evidence(*, m: int, l: int) -> tuple[dict[str, Any], ...]:
    calls = {
        name: int(name in SEMANTIC_OPERATIONS[14:]) for name in SEMANTIC_OPERATIONS
    }
    total_calls = sum(calls.values())
    memory = {
        "permanent_bytes": 100,
        "residual_phase_bytes": 20,
        "trait_phase_bytes": 30,
        "compact_output_bytes": 10,
        "integrity_reserve_bytes": 40,
        "telemetry_bytes": 50,
    }
    phases = {
        "permanent_bytes": ("admission_through_publication", 1),
        "residual_phase_bytes": ("residual_moments_and_trait_preallocated_arena", 1),
        "trait_phase_bytes": ("preallocated_trait_arena", 1),
        "compact_output_bytes": ("publication", 1),
        "integrity_reserve_bytes": ("all_protected_phases", 4),
        "telemetry_bytes": ("admission_through_publication", 1),
    }
    required_capacity = total_calls * 8 + 32
    admission = {
        **memory,
        "required_workspace_bytes": sum(memory.values()),
        "required_telemetry_capacity": required_capacity,
        "maximum_protected_output_elements": 8,
        "trait_descriptor_passes": 1,
        "trait_decoded_blocks": 1,
        "total_descriptor_passes": 1,
        "total_decoded_blocks": 1,
        "total_variant_record_visits": m,
        "total_protected_calls": total_calls,
        "semantic_call_ledger": calls,
        "selected_tiles": {"variant_block": m, "trait_feature_tile": 1},
        "phase_ledger": {
            "residual_moments": {"phenotype_projections": l},
            "trait": {
                "descriptor_passes": 1,
                "decoded_blocks": 1,
                "variant_record_visits": m,
                "logical_bed_record_bytes_touched": 2 * m,
                "access_order": "retained_logical_sequential_v1",
                "logical_duplicate_variant_decodes": 0,
                "phenotypes_in_packed_rhs": l,
            },
        },
        "memory_ledger": memory,
        "memory_lifetimes": {
            name: {
                "byte_count": value,
                "live_phase": phases[name][0],
                "concurrent_copies": phases[name][1],
            }
            for name, value in memory.items()
        },
        "memory_accounting_model": (
            "tracked_vector_payload_bytes_v1_excludes_allocator_metadata_"
            "and_small_strings"
        ),
        "os_physical_read_bytes_measured": False,
        "os_page_faults_measured": False,
        "physical_io_evidence": "logical_mmap_record_touches_only_v1",
    }
    diagnostics = {
        "maximum_projection_leakage": 0.0,
        "phenotype_projection_leakage_max_abs": 0.0,
        "phenotype_normalized_norm_error_max_abs": 0.0,
        "residual_gram_pre_symmetry_max_abs": 0.0,
        "trait_group_reconstruction_max_abs": 0.0,
        "missing_genotype_calls": 0,
        "observed_descriptor_passes": 1,
        "observed_decoded_blocks": 1,
        "observed_variant_record_visits": m,
        "observed_phenotype_projections": l,
        "protected_call_counts": calls,
        "descriptor_accounting_verified": True,
        "semantic_call_ledger_exact": True,
        "files_unchanged_at_all_checkpoints": True,
        "bed_content_evidence": "boundary_sample_only_no_extra_full_BED_traversal",
        "inputs_unchanged_at_all_checkpoints": True,
        "scratch_released": True,
        "scratch_released_before_publication": True,
        "all_large_buffers_preallocated_before_decode": True,
        "runtime_large_allocations_after_decode": 0,
        "tracked_high_water_bytes": sum(memory.values()),
        "tracked_high_water_within_admission": True,
        "strict_disjoint_optimized_path": False,
        "trait_group_reconstruction_verified": True,
        "phenotype_projection_count_verified": True,
        "phenotype_normalization_verified": True,
        "operand_fingerprints_verified": True,
        "runtime_thread_affinity_fingerprints_verified": True,
        "independent_scalar_witness_verified": True,
        "independent_scalar_fallback_available": True,
        "final_output_identity_fnv64": "123",
    }
    events = [
        _event("mutation_checkpoint", resolution="post_seal", phase="admission"),
        _event("mutation_checkpoint", resolution="post_admission", phase="admission"),
        _event(
            "phase_transition",
            resolution="residual_moments_running",
            phase="residual_moments",
        ),
        _event(
            "mutation_checkpoint",
            resolution="post_residual_moments",
            phase="residual_moments",
        ),
        _event("phase_transition", resolution="trait_running", phase="trait"),
        *[
            _event(
                "protected_call",
                operation=operation,
                resolution="primary_deterministic_tiled_fp64",
                phase="trait",
                protected=True,
            )
            for operation in SEMANTIC_OPERATIONS[14:]
        ],
        *[
            _event(
                "semantic_verification",
                operation=operation,
                resolution="scalar_witness_agreement",
                phase="trait",
                witness=True,
            )
            for operation in SEMANTIC_OPERATIONS[14:]
        ],
        _event("mutation_checkpoint", resolution="post_trait", phase="trait"),
        _event(
            "mutation_checkpoint",
            resolution="pre_finalization",
            phase="finalization",
        ),
        _event(
            "scratch_release",
            resolution="execution_arena_released",
            phase="finalization",
        ),
        _event(
            "mutation_checkpoint",
            resolution="pre_publication",
            phase="finalization",
        ),
        _event(
            "publication",
            resolution="compact_trait_statistics_ready",
            phase="publication",
        ),
    ]
    for sequence, recorded in enumerate(events, start=1):
        recorded["sequence"] = sequence
    telemetry = {
        "capacity": required_capacity,
        "required_capacity": required_capacity,
        "observed_events": len(events),
        "injection_count": 0,
        "repair_count": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "complete_without_drop": True,
        "fault_operation": "",
        "fault_mode": "none",
        "fault_semantic_anchor": "",
        "protected_call_counts": calls,
        "events": events,
    }
    return admission, diagnostics, telemetry


def _native_result() -> tuple[dict[str, Any], ContextualTraitPublicationIdentityV1]:
    n, m, q, k, j, h, l = 4, 4, 1, 1, 2, 1, 2
    components = ContextComponentIndex(("all",), ContextPairIndex(q))
    c = len(components)
    retained_order = _sha("retained-order")
    mean_digest = _sha("affine-mean")
    inverse_digest = _sha("affine-inverse")
    scale = GenotypeScalePlanV1(
        policy=GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1,
        retained_variant_order_sha256=retained_order,
        allele_orientation="bim_a1_counted_v1",
        allele_coding="plink_bed_snp_major_diploid_hardcall_v1",
        centering_source="provided_v1",
        centering_formula="provided_variant_affine_mean_v1",
        scaling_formula="dosage_minus_mean_times_inverse_scale_v1",
        missing_imputation="sealed_mean_v1",
        ploidy_policy="diploid_v1",
        affine_mean_sha256=mean_digest,
        affine_inverse_scale_sha256=inverse_digest,
    )
    variant_alleles = _sha("variant-order-alleles")
    publication = ContextualTraitPublicationIdentityV1(
        sample_order_sha256=_sha("sample-order"),
        variant_order_allele_sha256=variant_alleles,
        fixed_effect_spec_sha256=_sha("fixed-effect-spec"),
        basis_specification_sha256=_sha("basis-spec"),
        basis_calibration_sha256=_sha("basis-calibration"),
        compatible_reference_identity_sha256=_sha("reference-identity"),
        retained_sample_map_sha256=_sha("retained-sample-map"),
        retained_variant_order_sha256=retained_order,
        fixed_basis_sha256=_sha("fixed-basis"),
        evaluated_phi_sha256=_sha("evaluated-phi"),
        genotype_scale_plan_sha256=scale.digest,
        missingness_sha256=_sha("missingness"),
        annotation_map_sha256=_sha("annotation-map"),
        annotation_names=("all",),
        group_map_sha256=_sha("group-map"),
        group_ids=("g0", "g1"),
        phenotype_batch_sha256=_sha("phenotype-batch"),
        residual_basis_sha256=_sha("residual-basis"),
        trait_ids=("t0", "t1"),
        residual_names=("environment",),
    )
    group_rhs = np.asarray([[[4.0, 8.0]], [[8.0, 12.0]]])
    group_traces = np.asarray([[4.0], [8.0]])
    group_residual = np.asarray([[[2.0]], [[6.0]]])
    admission, diagnostics, telemetry = _execution_evidence(m=m, l=l)
    result: dict[str, Any] = {
        "complete_trait_artifact": False,
        "complete_trait_statistics": True,
        "internal_result_kind": "stage4_complete_trait_statistics_v1",
        "state": "trait_statistics_complete_ready",
        "lifecycle": "trait_statistics_complete_ready",
        "genetic_rhs": np.sum(group_rhs, axis=0) / 4.0,
        "genetic_traces": np.sum(group_traces, axis=0) / 4.0,
        "genetic_residual": np.sum(group_residual, axis=0) / 4.0,
        "residual_rhs": np.asarray([[1.0, 2.0]]),
        "residual_traces": np.asarray([4.0]),
        "residual_gram": np.asarray([[5.0]]),
        "group_rhs_unnormalized_num": group_rhs,
        "group_trace_unnormalized_num": group_traces,
        "group_genetic_residual_num": group_residual,
        "annotation_masses": np.asarray([4.0]),
        "group_annotation_masses": np.asarray([[2.0], [2.0]]),
        "group_variant_counts": np.asarray([2, 2], dtype=np.int64),
        "study_n": n,
        "n_variants": m,
        "residual_rank": n - 1,
        "q": q,
        "trait_count": l,
        "residual_component_count": h,
        "numeric_policy": "fp64_v1",
        "deletion_semantics": "approximate_summary_only_v1",
        "grouped_encoding": "grouped_unnormalized_dense_v1",
        "group_numerator_unit": "raw_source_annotation_mass_v1",
        "grouped_values_are_unnormalized_numerators": True,
        "residual_only_deletion": "reuse_full_unchanged",
        "phenotype_normalization": "project_then_unit_residual_variance_v1",
        "phenotypes_projected_once": True,
        "feature_mode": "P_diag_phi_G_v1",
        "pair_q": np.asarray(
            [entry.q for entry in components.pair_index.entries], dtype=np.int64
        ),
        "pair_r": np.asarray(
            [entry.r for entry in components.pair_index.entries], dtype=np.int64
        ),
        "pair_eta": np.asarray(
            [entry.kernel_factor for entry in components.pair_index.entries],
            dtype=np.int64,
        ),
        "component_annotation": np.asarray(
            [entry.annotation_index for entry in components.entries], dtype=np.int64
        ),
        "component_pair": np.asarray(
            [entry.pair_index for entry in components.entries], dtype=np.int64
        ),
        "annotation_names": ["all"],
        "group_names": ["g0", "g1"],
        "trait_names": ["t0", "t1"],
        "residual_names": ["environment"],
        "annotation_mode": AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1.value,
        "genotype_scale_policy": scale.policy.value,
        "allele_orientation": scale.allele_orientation,
        "allele_coding": scale.allele_coding,
        "centering_source": scale.centering_source,
        "centering_formula": scale.centering_formula,
        "scaling_formula": scale.scaling_formula,
        "missing_imputation": scale.missing_imputation,
        "ploidy_policy": scale.ploidy_policy,
        "retained_variant_order_sha256": retained_order,
        "affine_mean_sha256": mean_digest,
        "affine_inverse_scale_sha256": inverse_digest,
        "scale_plan_sha256": scale.digest,
        "retained_sample_map_sha256": _sha("retained-sample-map"),
        "variant_order_allele_sha256": variant_alleles,
        "fixed_basis_sha256": _sha("fixed-basis"),
        "evaluated_phi_sha256": _sha("evaluated-phi"),
        "annotation_map_sha256": _sha("annotation-map"),
        "group_map_sha256": _sha("group-map"),
        "phenotype_batch_sha256": _sha("phenotype-batch"),
        "residual_basis_sha256": _sha("residual-basis"),
        "missingness_sha256": _sha("missingness"),
        "sealed_plan_sha256": _sha("sealed-plan"),
        "source_tree_sha256": _sha("source-tree"),
        "execution_plan_sha256": _sha("trait-execution-plan"),
        "phase_evidence_sha256": {
            "residual_derived_state": _sha("residual-derived-state"),
            "post_trait_outputs": _sha("post-trait-outputs"),
        },
        "build_provenance": _native_build_provenance("2" * 40, _sha("source-tree")),
        "contextual_native_api_version": 1,
        "contextual_backend_version": 2,
        "contextual_backend": "plink_bed_descriptor_stream_trait_v1",
        "contextual_execution_backend": "deterministic_tiled_fp64_with_scalar_witness_v1",
        "contextual_build_id": "2" * 40,
        "file_identity_policy": (
            "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2"
        ),
        "file_content_identity": _file_content_identity(m),
        "output_ownership": {"owns_data": True, "read_only": True},
        "numa": {
            "numa_applicable": False,
            "numa_verified": False,
            "policy": "unbound_first_touch_v1",
            "output_numa_node": -1,
            "reason": "unbound standard allocator; no placement or output-node claim",
        },
        "admission": admission,
        "diagnostics": diagnostics,
        "telemetry": telemetry,
    }
    _refresh_trait_scientific_digests(result)
    assert c == k * q * (q + 1) // 2 and j == len(result["group_names"])
    return result, publication


def _artifact():
    result, publication = _native_result()
    return adapt_native_contextual_trait_v1(result, publication)


def test_adapter_publishes_owned_batch_oriented_trait_artifact() -> None:
    native, publication = _native_result()
    artifact = adapt_native_contextual_trait_v1(native, publication)
    assert artifact.genetic_rhs.shape == (1, 2)
    assert artifact.residual_rhs.shape == (1, 2)
    assert artifact.group_rhs_unnormalized_num.shape == (2, 1, 2)
    assert artifact.manifest["logical_schema_version"] == "contextual_trait_v1"
    assert artifact.manifest["compatible_reference_identity_sha256"] == _sha(
        "reference-identity"
    )
    for name in artifact.manifest["arrays"]:
        value = getattr(artifact, name)
        assert not value.flags.writeable
        assert not np.shares_memory(value, native[name])
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError, match="immutable"):
        artifact.manifest["dimensions"]["L"] = 3
    artifact.verify()


def test_summary_only_deletion_subtracts_raw_numerators_then_renormalizes() -> None:
    artifact = _artifact()
    deleted = trait_moments_after_deleting_groups_v1(artifact, ["g0"])
    np.testing.assert_array_equal(deleted.annotation_masses, [2.0])
    np.testing.assert_allclose(deleted.genetic_rhs, [[4.0, 6.0]])
    np.testing.assert_allclose(deleted.genetic_traces, [4.0])
    np.testing.assert_allclose(deleted.genetic_residual, [[3.0]])
    assert deleted.residual_rhs is not artifact.residual_rhs
    np.testing.assert_array_equal(deleted.residual_rhs, artifact.residual_rhs)
    with pytest.raises(ValueError, match="unique"):
        trait_moments_after_deleting_groups_v1(artifact, ["g0", "g0"])
    with pytest.raises(ValueError, match="nonpositive"):
        trait_moments_after_deleting_groups_v1(artifact, ["g0", "g1"])


def test_strict_round_trip_rejects_manifest_and_array_tampering(tmp_path: Path) -> None:
    artifact = _artifact()
    path = write_contextual_trait_v1(artifact, tmp_path / "batch")
    assert path.name.endswith(CONTEXTUAL_TRAIT_V1_SUFFIX)
    loaded = load_contextual_trait_v1(path)
    assert loaded.manifest_sha256 == artifact.manifest_sha256
    np.testing.assert_array_equal(loaded.genetic_rhs, artifact.genetic_rhs)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["genetic_rhs"][0, 0] += 1.0
    tampered = tmp_path / f"tampered{CONTEXTUAL_TRAIT_V1_SUFFIX}"
    np.savez_compressed(tampered, **payload)
    with pytest.raises(ValueError, match="array digest mismatch"):
        load_contextual_trait_v1(tampered)

    payload["genetic_rhs"][0, 0] -= 1.0
    manifest = json.loads(str(payload["manifest_json"].item()))
    manifest["terminal_status"] = "development"
    payload["manifest_json"] = np.asarray(json.dumps(manifest))
    payload["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    np.savez_compressed(tampered, **payload)
    with pytest.raises(ValueError, match="not published"):
        load_contextual_trait_v1(tampered)


def test_adapter_and_runner_fail_closed_on_schema_or_multiple_runs() -> None:
    native, publication = _native_result()
    extra = copy.deepcopy(native)
    extra["future_key"] = True
    with pytest.raises(ValueError, match="schema mismatch"):
        adapt_native_contextual_trait_v1(extra, publication)
    wrong = copy.deepcopy(native)
    wrong["genetic_rhs"] = wrong["genetic_rhs"].T.copy()
    with pytest.raises(ValueError, match="shape|digest"):
        adapt_native_contextual_trait_v1(wrong, publication)

    class Executor:
        def __init__(self) -> None:
            self.calls = 0

        def run(self) -> dict[str, Any]:
            self.calls += 1
            return native

    executor = Executor()
    artifact = run_contextual_trait_v1(executor, publication)
    assert executor.calls == 1
    assert artifact.trait_index("t1") == 1


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("call_ledger", "omitted an active trait operation"),
        ("telemetry_ledger", "telemetry call ledger mismatch"),
        ("lifecycle", "telemetry resolution is invalid"),
    ],
)
def test_adapter_rejects_execution_evidence_tampering(
    tamper: str, message: str
) -> None:
    native, publication = _native_result()
    if tamper == "call_ledger":
        native["admission"]["semantic_call_ledger"]["trait_score_tn"] = 0
    elif tamper == "telemetry_ledger":
        native["telemetry"]["protected_call_counts"] = {
            **native["telemetry"]["protected_call_counts"],
            "trait_score_tn": 0,
        }
    else:
        checkpoint = next(
            event
            for event in native["telemetry"]["events"]
            if event["event_class"] == "mutation_checkpoint"
            and event["resolution"] == "post_trait"
        )
        checkpoint["resolution"] = "post_source"
    with pytest.raises(ValueError, match=message):
        adapt_native_contextual_trait_v1(native, publication)
