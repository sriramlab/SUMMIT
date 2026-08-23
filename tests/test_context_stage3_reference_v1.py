from __future__ import annotations

import copy
import errno
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from summit.context import (
    AnnotationMode,
    ContextComponentIndex,
    ContextPairIndex,
    ContextualReferenceArtifactV1,
    ContextualReferencePublicationIdentityV1,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
    adapt_native_contextual_reference_v1,
    assemble_context_normal_equations,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    canonical_json,
    canonical_sha256,
    contextual_variant_order_allele_sha256_v1,
    load_context_reference,
    load_contextual_reference_v1,
    fit_context_model,
    derive_context_outputs,
    rank_revealing_projector,
    reference_moments_after_deleting_groups_v1,
    run_contextual_reference_v1,
    solve_context_normal_equations,
    write_contextual_reference_v1,
)


def _sha(label: str) -> str:
    return canonical_sha256({"label": label})


def _native_build_provenance(build_id: str, source_tree: str) -> dict[str, Any]:
    return {
        "schema": "contextual_native_build_provenance_v1",
        "source_commit": build_id,
        "source_tree_sha256": source_tree,
        "compiler_id": "GNU",
        "compiler_version": "12.2.0",
        "cxx_standard": 17,
        "build_type": "Release",
        "sanitizer_mode": "none",
        "asan_enabled": False,
        "ubsan_enabled": False,
        "effective_optimization": "-O3",
        "architecture_tuning": "portable",
        "configured_compiler_flags": "-O3",
        "blas_vendor": "synthetic",
        "gemm_integrity_enabled": True,
        "gemm_checksum_enabled": True,
        "private_blas_enabled": False,
        "private_blas_backend": "none",
        "private_blas_sha256": "none",
        "private_blas_source_commit": "none",
        "private_blas_source_tree_sha256": "none",
        "private_blas_config_family": "none",
        "private_blas_header_sha256": "none",
        "private_blas_cblas_header_sha256": "none",
        "private_openblas_enabled": False,
        "private_openblas_sha256": "none",
        "native_arch_optimization_enabled": False,
        "openmp_enabled": False,
        "contextual_dispatch_backend": (
            "deterministic_tiled_fp64_with_scalar_witness_v1"
        ),
        "contextual_dispatch_vendor_calls": False,
    }


def _file_content_identity(m: int) -> dict[str, Any]:
    state = {
        "device": 1,
        "inode": 1,
        "size": 128,
        "link_count": 1,
        "mtime_seconds": 1,
        "mtime_nanoseconds": 0,
        "ctime_seconds": 1,
        "ctime_nanoseconds": 0,
    }
    return {
        "schema": "contextual_file_content_identity_v2",
        "policy": ("sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2"),
        "duplicated_descriptors": True,
        "bim_full_sha256": _sha("bim-content"),
        "fam_full_sha256": _sha("fam-content"),
        "bed_header_sha256": _sha("bed-header"),
        "retained_bed_logical_record_stream_sha256": _sha("bed-records"),
        "retained_record_count": m,
        "bytes_per_bed_record": 2,
        "retained_record_sha256_count": m,
        "bed_stream_accumulation": (
            "authoritative_decode_logical_order_no_extra_bed_traversal_v1"
        ),
        "bim_fam_hash_source": "full_duplicated_descriptor_bytes_v1",
        "bed_descriptor_state": dict(state),
        "bim_descriptor_state": dict(state),
        "fam_descriptor_state": dict(state),
        "boundary_sample_evidence": "first_last_4KiB_per_descriptor_v1",
        "full_bed_file_sha256_claimed": False,
        "absolute_snapshot_or_lease_claimed": False,
        "toctou_closed": False,
        "toctou_nonclaim": (
            "no_absolute_snapshot_or_lease_mutation_after_last_verified_read_"
            "remains_out_of_scope_v1"
        ),
    }


def _refresh_reference_scientific_digests(result: dict[str, Any]) -> None:
    names = (
        "gram",
        "raw_gram_numerator",
        "annotation_masses",
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
        "same_person",
        "group_gram_unnormalized_num",
        "group_annotation_masses",
        "group_variant_counts",
    )
    result["scientific_array_sha256"] = {
        name: array_sha256(result[name]) for name in names
    }


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


def _execution_evidence(
    m: int, bd: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calls = {
        name: (1 if index < 9 or 10 <= index < 14 else 0)
        for index, name in enumerate(SEMANTIC_OPERATIONS)
    }
    total_calls = sum(calls.values())
    memory = {
        "permanent_bytes": 100,
        "source_phase_bytes": 200,
        "action_phase_bytes": 240,
        "group_phase_bytes": 300,
        "same_person_phase_bytes": 400,
        "compact_output_bytes": 50,
        "integrity_reserve_bytes": 80,
        "telemetry_bytes": 100,
    }
    admission = {
        **memory,
        "required_workspace_bytes": 100 + 240 + 300 + 400 + 50 + 80 + 100,
        "required_telemetry_capacity": total_calls * 8 + 32,
        "maximum_protected_output_elements": 12,
        "resident_batches": 1,
        "source_descriptor_passes": 1,
        "action_descriptor_passes": 1,
        "source_decoded_blocks": 1,
        "action_decoded_blocks": 1,
        "group_descriptor_passes": 1,
        "group_decoded_blocks": 1,
        "same_person_descriptor_passes": 1,
        "same_person_decoded_blocks": 1,
        "total_descriptor_passes": 4,
        "total_decoded_blocks": 4,
        "total_variant_record_visits": 4 * m,
        "total_protected_calls": total_calls,
        "semantic_call_ledger": calls,
        "selected_tiles": {
            "variant_block": m,
            "sample_probe_resident": 1,
            "sample_probe_tile": 1,
            "action_tile": 1,
            "annotation_tile": 1,
            "context_tile": 1,
            "variant_probe_tile": 1,
            "group_tile": 1,
        },
        "phase_ledger": {
            "source": {
                "descriptor_passes": 1,
                "decoded_blocks": 1,
                "variant_record_visits": m,
            },
            "action": {
                "descriptor_passes": 1,
                "decoded_blocks": 1,
                "variant_record_visits": m,
            },
            "grouped": {
                "descriptor_passes": 1,
                "decoded_blocks": 1,
                "variant_record_visits": m,
                "execution_order": "sealed_group_contiguous_permutation_v1",
                "selected_algorithm": "group_restricted_action_v1",
            },
            "same_person": {
                "descriptor_passes": 1,
                "decoded_blocks": 1,
                "variant_record_visits": m,
                "global_probe_merge_before_finalization": True,
            },
        },
        "memory_ledger": memory,
        "memory_lifetimes": {
            name: {
                "byte_count": value,
                "live_phase": {
                    "permanent_bytes": "admission_through_publication",
                    "source_phase_bytes": "source_and_action_preallocated_arena",
                    "action_phase_bytes": "source_and_action_preallocated_arena",
                    "group_phase_bytes": "preallocated_group_arena",
                    "same_person_phase_bytes": "preallocated_same_person_arena",
                    "compact_output_bytes": "publication",
                    "integrity_reserve_bytes": "all_protected_phases",
                    "telemetry_bytes": "admission_through_publication",
                }[name],
                "concurrent_copies": 4 if name == "integrity_reserve_bytes" else 1,
            }
            for name, value in memory.items()
        },
        "memory_accounting_model": "tracked_vector_payload_bytes_v1_excludes_allocator_metadata_and_small_strings",
        "target_dense_columns": 1,
        "target_useful_columns": 1,
        "strict_zero_weight_columns_eliminated": 0,
        "os_physical_read_bytes_measured": False,
        "os_page_faults_measured": False,
        "physical_io_evidence": "logical_mmap_record_touches_only_v1",
    }
    for phase in ("source", "action", "same_person"):
        admission["phase_ledger"][phase].update(
            {
                "logical_bed_record_bytes_touched": 2 * m,
                "access_order": "retained_logical_sequential_v1",
                "logical_duplicate_variant_decodes": 0,
            }
        )
    admission["phase_ledger"]["grouped"].update(
        {
            "logical_bed_record_bytes_touched": 2 * m,
            "access_order": "sealed_group_indexed_v1",
            "logical_duplicate_variant_decodes": 0,
            "group_restricted_duplicate_variant_decodes": 0,
            "group_restricted_variant_coverage": m,
        }
    )
    diagnostics = {
        "maximum_projection_leakage": 0.0,
        "gram_pre_symmetry_max_abs": 0.0,
        "missing_genotype_calls": 0,
        "observed_descriptor_passes": 4,
        "observed_decoded_blocks": 4,
        "observed_variant_record_visits": 4 * m,
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
        "tracked_high_water_bytes": admission["required_workspace_bytes"],
        "tracked_high_water_within_admission": True,
        "strict_disjoint_optimized_path": False,
        "same_person_pre_symmetry_max_abs": 0.0,
        "group_reconstruction_max_abs": 0.0,
        "direct_action_restricted_max_abs": 0.0,
        "direct_genotype_restricted_max_abs": 0.0,
        "direct_placement_max_abs": 0.0,
        "group_reconstruction_verified": True,
        "same_person_global_merge_verified": True,
        "same_person_cross_tile_pairs_included": True,
        "variant_probe_coverage": bd,
        "group_execution_variant_visits": m,
        "group_execution_permutation_coverage_verified": True,
        "group_execution_permutation_unique_verified": True,
        "group_execution_contiguous_verified": True,
        "group_tile_batches": 1,
        "logical_order_missing_identity_verified": True,
        "same_person_signed_output_preserved": True,
        "target_dense_columns": 1,
        "target_useful_columns": 1,
        "strict_zero_weight_columns_eliminated": 0,
        "operand_fingerprints_verified": True,
        "runtime_thread_affinity_fingerprints_verified": True,
        "independent_scalar_witness_verified": True,
        "independent_scalar_fallback_available": True,
        "source_identity_fnv64": "1",
        "source_action_identity_fnv64": "2",
        "final_output_identity_fnv64": "3",
    }

    def event(
        event_class: str,
        phase: str,
        operation: str,
        resolution: str,
        attempt: int,
        *,
        protected: bool = False,
        coordinates: bool = False,
    ) -> dict[str, Any]:
        attempt_event = operation != "none" and event_class in {
            "protected_call",
            "repair",
            "retry",
            "trusted_fallback",
            "semantic_verification",
        }
        operation_index = (
            SEMANTIC_OPERATIONS.index(operation) if operation != "none" else -1
        )
        if 0 <= operation_index <= 3:
            semantic_phase = "source"
        elif 4 <= operation_index <= 5:
            semantic_phase = "action"
        elif operation_index == 6:
            semantic_phase = "gram"
        elif 7 <= operation_index <= 9:
            semantic_phase = "group"
        elif 10 <= operation_index <= 13:
            semantic_phase = "same_person"
        elif operation_index >= 14:
            semantic_phase = "trait"
        else:
            semantic_phase = "unspecified"
        if operation != "none":
            phase = "action" if operation == "full_target_nn" else semantic_phase
        semantic_role = (
            "global_merge" if operation == "same_person_gram_tn" else "ordinary"
        )
        semantic_placement = (
            "action_scaled" if operation == "direct_grouped_tn" else "none"
        )
        begins = {
            "resident": 0,
            "probe": 0,
            "variant": 0,
            "annotation": 0,
            "context": 0,
            "action": 0,
            "group": 0,
        }
        ends = {name: int(coordinates) for name in begins}
        if operation == "same_person_gram_tn" and coordinates:
            ends["probe"] = bd
        exact_membership = operation in {
            "group_target_nn",
            "group_cross_gram_tn",
        }
        membership_count = int(coordinates and exact_membership)
        membership_sha256 = (
            _sha(f"variant-membership-{operation}") if exact_membership else ""
        )
        canonical_end = (
            bd
            if coordinates and operation == "same_person_gram_tn"
            else int(
                coordinates
                and operation
                in {
                    "group_target_nn",
                    "group_cross_gram_tn",
                    "direct_grouped_tn",
                }
            )
        )
        anchor = (
            f"{operation}|resident={begins['resident']}:{ends['resident']}"
            f"|probe={begins['probe']}:{ends['probe']}"
            f"|variant={begins['variant']}:{ends['variant']}"
            f"|annotation={begins['annotation']}:{ends['annotation']}"
            f"|context={begins['context']}:{ends['context']}"
            f"|action={begins['action']}:{ends['action']}"
        )
        if operation in SEMANTIC_OPERATIONS[7:10]:
            anchor += f"|group={begins['group']}:{ends['group']}"
        anchor += (
            f"|phase={semantic_phase}|role={semantic_role}"
            f"|placement={semantic_placement}|canonical=0:{canonical_end}"
        )
        if exact_membership:
            anchor += (
                "|variant_mode=exact_membership_sha256_v1"
                f"|variant_count={membership_count}"
                f"|variant_sha256={membership_sha256}"
            )
        return {
            "event_class": event_class,
            "phase": phase,
            "operation": operation,
            "semantic_anchor": anchor,
            "semantic_phase": semantic_phase,
            "semantic_role": semantic_role,
            "semantic_placement": semantic_placement,
            "canonical_begin": 0,
            "canonical_end": canonical_end,
            "variant_coordinate_mode": (
                "exact_membership_sha256_v1"
                if exact_membership
                else "logical_half_open_range_v1"
            ),
            "variant_range_is_exact": True,
            "variant_membership_count": membership_count,
            "variant_membership_sha256": membership_sha256,
            "resolution": resolution,
            "attempt": attempt,
            "rows": int(attempt_event),
            "columns": int(attempt_event),
            "reduction": int(attempt_event),
            "left_stride": int(attempt_event),
            "right_stride": int(attempt_event),
            "output_stride": int(attempt_event),
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
            **{f"{name}_begin": value for name, value in begins.items()},
            **{f"{name}_end": value for name, value in ends.items()},
        }

    events = [
        event("mutation_checkpoint", "admission", "none", "post_seal", 0),
        event("mutation_checkpoint", "admission", "none", "post_admission", 0),
        event("phase_transition", "source", "none", "source_running", 0),
        event("phase_transition", "action", "none", "action_running", 0),
        event("phase_transition", "gram", "none", "gram_running", 0),
        event("phase_transition", "group", "none", "group_running", 0),
        event("phase_transition", "same_person", "none", "same_person_running", 0),
        event("mutation_checkpoint", "source", "none", "post_source", 0),
        event("mutation_checkpoint", "action", "none", "post_action", 0),
        event("mutation_checkpoint", "gram", "none", "post_gram", 0),
        event("mutation_checkpoint", "group", "none", "post_group", 0),
        event("mutation_checkpoint", "same_person", "none", "post_same_person", 0),
    ]
    for operation, count in calls.items():
        for _ in range(count):
            events.append(
                event(
                    "protected_call",
                    "source",
                    operation,
                    "primary_deterministic_tiled_fp64",
                    1,
                    protected=True,
                    coordinates=True,
                )
            )
            events.append(
                event(
                    "semantic_verification",
                    "source",
                    operation,
                    "scalar_witness_agreement",
                    1,
                    coordinates=True,
                )
            )
    events.extend(
        [
            event(
                "semantic_verification",
                "group",
                "none",
                "group_reconstruction_verified",
                0,
            ),
            event(
                "mutation_checkpoint",
                "finalization",
                "none",
                "pre_finalization",
                0,
            ),
            event(
                "scratch_release",
                "finalization",
                "none",
                "execution_arena_released",
                0,
            ),
            event(
                "mutation_checkpoint",
                "finalization",
                "none",
                "pre_publication",
                0,
            ),
            event(
                "publication",
                "publication",
                "none",
                "compact_complete_statistics_ready",
                0,
            ),
        ]
    )
    for sequence, recorded in enumerate(events, start=1):
        recorded["sequence"] = sequence
    telemetry = {
        "capacity": admission["required_telemetry_capacity"],
        "required_capacity": admission["required_telemetry_capacity"],
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


def _native_result() -> tuple[dict[str, Any], ContextualReferencePublicationIdentityV1]:
    q, j, n, m, bt, bd = 2, 3, 8, 6, 4, 3
    components = ContextComponentIndex(("left", "right"), ContextPairIndex(q))
    c = len(components)
    annotation_masses = np.asarray([6.0, 12.0])
    group_masses = np.asarray([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    rng = np.random.default_rng(930_201)
    basis = rng.normal(size=(c, c))
    gram = basis @ basis.T / bt
    component_annotation = np.asarray(
        [entry.annotation_index for entry in components.entries], dtype=np.int64
    )
    component_masses = annotation_masses[component_annotation]
    raw = gram * np.outer(component_masses, component_masses)
    grouped = np.stack([raw / j] * j)
    signed = rng.normal(size=(c, c))
    same_person = 0.5 * (signed + signed.T)
    same_person[0, 0] = -abs(same_person[0, 0]) - 1.0

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
    sample_map_sha256 = _sha("retained-sample-map")
    variant_allele_sha256 = _sha("variant-order-and-alleles")
    publication = ContextualReferencePublicationIdentityV1(
        sample_order_sha256=sample_map_sha256,
        variant_order_allele_sha256=variant_allele_sha256,
        fixed_effect_spec_sha256=_sha("fixed-effect-spec"),
        basis_specification_sha256=_sha("basis-spec"),
        basis_calibration_sha256=_sha("basis-calibration"),
        retained_sample_map_sha256=sample_map_sha256,
        retained_variant_order_sha256=retained_order,
        fixed_basis_sha256=_sha("fixed-basis"),
        evaluated_phi_sha256=_sha("evaluated-phi"),
        genotype_scale_plan_sha256=scale.digest,
        missingness_sha256=_sha("missingness"),
        annotation_map_sha256=_sha("annotation-map"),
        annotation_names=("left", "right"),
        group_map_sha256=_sha("group-map"),
        group_ids=("g0", "g1", "g2"),
        sample_probe_policy="explicit_rademacher_v1",
        sample_probe_identity_sha256=_sha("sample-probes"),
        variant_probe_policy="explicit_variant_rademacher_v1",
        variant_probe_identity_sha256=_sha("variant-probes"),
    )
    pair_entries = components.pair_index.entries
    component_entries = components.entries
    admission, diagnostics, telemetry = _execution_evidence(m, bd)
    result: dict[str, Any] = {
        "complete_reference_artifact": False,
        "file_identity_policy": (
            "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2"
        ),
        "file_content_identity": _file_content_identity(m),
        "complete_reference_statistics": True,
        "internal_result_kind": "stage3_complete_reference_statistics_v1",
        "state": "reference_statistics_complete_ready",
        "lifecycle": "reference_statistics_complete_ready",
        "reference_n": n,
        "n_variants": m,
        "residual_rank": n - 2,
        "q": q,
        "sample_probe_count": bt,
        "variant_probe_count": bd,
        "gram": gram,
        "same_person": same_person,
        "group_gram_unnormalized_num": grouped,
        "annotation_masses": annotation_masses,
        "group_annotation_masses": group_masses,
        "group_variant_counts": np.asarray([2, 2, 2], dtype=np.int64),
        "raw_gram_numerator": raw,
        "numeric_policy": "fp64_v1",
        "deletion_semantics": "approximate_summary_only_v1",
        "same_person_deletion": "reuse_full_unchanged",
        "grouped_encoding": "grouped_unnormalized_dense_v1",
        "group_numerator_unit": "mass_squared_fixed_probe_raw_action_v1",
        "group_gram_raw_unit": "(1/B_T)_sum_raw_cross_products_annotation_mass_unnormalized_v1",
        "grouped_values_are_unnormalized_numerators": True,
        "same_person_signed_preserved": True,
        "selected_grouped_algorithm": "group_restricted_action_v1",
        "grouped_attribution_algorithm": "group_restricted_action_v1",
        "direct_grouped_scaling": "action_scaled_v1",
        "grouped_differential_enabled": False,
        "group_execution_order": "stable_group_contiguous_logical_index_permutation_v1",
        "group_execution_order_sha256": _sha("group-permutation"),
        "group_execution_permutation_sha256": _sha("group-permutation"),
        "sealed_plan_sha256": _sha("sealed-plan"),
        "pair_q": np.asarray([entry.q for entry in pair_entries], dtype=np.int64),
        "pair_r": np.asarray([entry.r for entry in pair_entries], dtype=np.int64),
        "pair_eta": np.asarray(
            [entry.kernel_factor for entry in pair_entries], dtype=np.int64
        ),
        "component_annotation": component_annotation,
        "component_pair": np.asarray(
            [entry.pair_index for entry in component_entries], dtype=np.int64
        ),
        "annotation_names": ["left", "right"],
        "group_names": ["g0", "g1", "g2"],
        "genotype_scale_policy": scale.policy.value,
        "allele_orientation": scale.allele_orientation,
        "allele_coding": scale.allele_coding,
        "centering_source": scale.centering_source,
        "centering_formula": scale.centering_formula,
        "scaling_formula": scale.scaling_formula,
        "missing_imputation": scale.missing_imputation,
        "ploidy_policy": scale.ploidy_policy,
        "affine_mean_sha256": mean_digest,
        "affine_inverse_scale_sha256": inverse_digest,
        "retained_variant_order_sha256": retained_order,
        "scale_plan_sha256": scale.digest,
        "missingness_sha256": _sha("missingness"),
        "retained_sample_map_sha256": sample_map_sha256,
        "variant_order_allele_sha256": variant_allele_sha256,
        "fixed_basis_sha256": _sha("fixed-basis"),
        "evaluated_phi_sha256": _sha("evaluated-phi"),
        "annotation_map_sha256": _sha("annotation-map"),
        "group_map_sha256": _sha("group-map"),
        "probe_policy": "explicit_rademacher_v1",
        "probe_identity_sha256": _sha("sample-probes"),
        "sample_probe_policy": "explicit_rademacher_v1",
        "sample_probe_identity_sha256": _sha("sample-probes"),
        "variant_probe_policy": "explicit_variant_rademacher_v1",
        "variant_probe_identity_sha256": _sha("variant-probes"),
        "annotation_mode": AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1.value,
        "same_person_sha256": array_sha256(same_person),
        "group_gram_unnormalized_num_sha256": array_sha256(grouped),
        "admission": admission,
        "diagnostics": diagnostics,
        "telemetry": telemetry,
        "contextual_native_api_version": 1,
        "contextual_backend_version": 2,
        "contextual_backend": "plink_bed_descriptor_stream_stage2_v1",
        "contextual_execution_backend": "deterministic_tiled_fp64_with_scalar_witness_v1",
        "contextual_build_id": "1" * 40,
        "source_tree_sha256": _sha("source-tree"),
        "execution_plan_sha256": _sha("execution-plan"),
        "phase_evidence_sha256": {"post_gram": _sha("post-gram")},
        "build_provenance": _native_build_provenance("1" * 40, _sha("source-tree")),
        "output_ownership": {"owns_data": True, "read_only": True},
        "numa": {
            "numa_applicable": False,
            "numa_verified": False,
            "policy": "unbound_first_touch_v1",
            "output_numa_node": -1,
            "reason": "unbound standard allocator; no placement or output-node claim",
        },
    }
    _refresh_reference_scientific_digests(result)
    return result, publication


def _artifact() -> ContextualReferenceArtifactV1:
    result, publication = _native_result()
    return adapt_native_contextual_reference_v1(result, publication)


def test_adapter_builds_closed_complete_compact_deeply_immutable_artifact() -> None:
    result, publication = _native_result()
    artifact = adapt_native_contextual_reference_v1(result, publication)
    assert artifact.manifest["artifact_family"] == "contextual_reference"
    assert artifact.manifest["logical_schema_version"] == "contextual_reference_v1"
    assert artifact.manifest["grouped_encoding_version"] == (
        "grouped_unnormalized_dense_v1"
    )
    assert artifact.manifest["terminal_status"] == "published"
    assert artifact.same_person[0, 0] < 0.0
    assert set(artifact.manifest["arrays"]) == {
        "gram",
        "same_person",
        "group_gram_unnormalized_num",
        "annotation_masses",
        "group_annotation_masses",
        "group_variant_counts",
    }
    for name in artifact.manifest["arrays"]:
        value = np.asarray(getattr(artifact, name))
        assert not value.flags.writeable
        assert not np.shares_memory(value, np.asarray(result[name]))
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError, match="immutable"):
        artifact.manifest["dimensions"]["M"] = 99
    artifact.verify()


def test_all_single_and_selected_multiple_deletions_use_raw_units_and_full_d() -> None:
    artifact = _artifact()
    component_annotation = np.asarray(
        [entry.annotation_index for entry in artifact.component_index.entries]
    )
    for requested in (("g0",), ("g1",), ("g2",), ("g0", "g2")):
        moments = reference_moments_after_deleting_groups_v1(artifact, requested)
        deleted = np.asarray([group in requested for group in artifact.group_ids])
        retained_masses = artifact.annotation_masses - np.sum(
            artifact.group_annotation_masses[deleted], axis=0
        )
        masses = retained_masses[component_annotation]
        expected = np.sum(
            artifact.group_gram_unnormalized_num[~deleted], axis=0
        ) / np.outer(masses, masses)
        np.testing.assert_allclose(moments.gram, expected, atol=1e-13, rtol=1e-13)
        np.testing.assert_array_equal(moments.same_person, artifact.same_person)
    with pytest.raises(ValueError, match="unique"):
        reference_moments_after_deleting_groups_v1(artifact, ("g0", "g0"))
    with pytest.raises(ValueError, match="Unknown"):
        reference_moments_after_deleting_groups_v1(artifact, ("unknown",))


def test_single_file_roundtrip_exact_keys_digests_and_schema_isolation(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    path = write_contextual_reference_v1(artifact, tmp_path / "reference")
    assert path.name.endswith(".contextual-reference-v1.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive.files) == len(set(archive.files)) == 8
        assert set(archive.files) == {
            *artifact.manifest["arrays"],
            "manifest_json",
            "manifest_sha256",
        }
    loaded = load_contextual_reference_v1(path)
    assert loaded.manifest_sha256 == artifact.manifest_sha256
    for name in artifact.manifest["arrays"]:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(artifact, name))
    with pytest.raises(ValueError, match="isolated V1 loader"):
        load_context_reference(path)
    with pytest.raises(ValueError, match="V1 suffix"):
        load_contextual_reference_v1(tmp_path / "legacy.context-reference.json")


def test_atomic_writer_refuses_and_preserves_existing_target(tmp_path: Path) -> None:
    artifact = _artifact()
    target = tmp_path / "atomic.contextual-reference-v1.npz"
    target.write_bytes(b"existing")

    with pytest.raises(FileExistsError) as error:
        write_contextual_reference_v1(artifact, target)
    assert error.value.errno == errno.EEXIST
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_run_adapter_calls_native_exactly_once() -> None:
    result, publication = _native_result()

    class Executor:
        calls = 0

        def run(self) -> dict[str, Any]:
            self.calls += 1
            return result

    executor = Executor()
    artifact = run_contextual_reference_v1(executor, publication)
    assert executor.calls == 1
    assert artifact.reference_n == 8


def test_variant_order_allele_hash_matches_native_framing_and_is_sensitive() -> None:
    retained = np.asarray([7, 2], dtype=np.int64)
    variant_ids = ["rs7", "rs2"]
    counted = ["A", "G"]
    other = ["C", "T"]
    orientation = np.asarray([1, 0], dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(b"variant_order_allele_v1")
    for row in range(2):
        digest.update(struct.pack("<q", int(retained[row])))
        for values in (variant_ids, counted, other):
            encoded = values[row].encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        digest.update(bytes((int(orientation[row]),)))
    observed = contextual_variant_order_allele_sha256_v1(
        retained, variant_ids, counted, other, orientation
    )
    assert observed == digest.hexdigest()
    np.testing.assert_array_equal(retained, [7, 2])
    np.testing.assert_array_equal(orientation, [1, 0])
    mutated = retained.copy()
    mutated[0] = 8
    assert (
        contextual_variant_order_allele_sha256_v1(
            mutated, variant_ids, counted, other, orientation
        )
        != observed
    )
    changed_alias = list(counted)
    changed_alias[0] = "C"
    assert (
        contextual_variant_order_allele_sha256_v1(
            retained, variant_ids, changed_alias, other, orientation
        )
        != observed
    )
    with pytest.raises(ValueError):
        contextual_variant_order_allele_sha256_v1(
            retained.astype(np.int32), variant_ids, counted, other, orientation
        )
    bad_orientation = orientation.copy()
    bad_orientation[0] = 2
    with pytest.raises(ValueError):
        contextual_variant_order_allele_sha256_v1(
            retained, variant_ids, counted, other, bad_orientation
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "float_dtype",
        "fractional_count",
        "map_dtype",
        "name_type",
        "event_truncated",
        "event_class",
        "event_coordinate",
        "event_dimensions",
        "memory_ledger",
        "memory_lifetime",
        "numa",
        "file_policy",
        "execution_backend",
    ],
)
def test_adapter_rejects_native_boundary_types_and_nested_evidence(
    mutation: str,
) -> None:
    result, publication = _native_result()
    result = copy.deepcopy(result)
    if mutation == "float_dtype":
        result["gram"] = result["gram"].astype(np.float32)
    elif mutation == "fractional_count":
        result["group_variant_counts"] = np.asarray([2.5, 1.5, 2.0], dtype=np.float64)
    elif mutation == "map_dtype":
        result["pair_q"] = result["pair_q"].astype(np.int32)
    elif mutation == "name_type":
        result["annotation_names"][0] = 7
    elif mutation == "event_truncated":
        index = next(
            index
            for index, event in enumerate(result["telemetry"]["events"])
            if event["event_class"] == "protected_call"
        )
        result["telemetry"]["events"].pop(index)
        result["telemetry"]["observed_events"] -= 1
    elif mutation == "event_class":
        result["telemetry"]["events"][0]["event_class"] = "unknown"
    elif mutation == "event_coordinate":
        event = next(
            event
            for event in result["telemetry"]["events"]
            if event["event_class"] == "protected_call"
        )
        event["variant_end"] = result["n_variants"] + 1
    elif mutation == "event_dimensions":
        event = next(
            event
            for event in result["telemetry"]["events"]
            if event["event_class"] == "protected_call"
        )
        event["rows"] = 0
    elif mutation == "memory_ledger":
        result["admission"]["memory_ledger"]["permanent_bytes"] += 1
    elif mutation == "memory_lifetime":
        result["admission"]["memory_lifetimes"]["integrity_reserve_bytes"][
            "concurrent_copies"
        ] = 3
    elif mutation == "numa":
        result["numa"]["numa_verified"] = True
    elif mutation == "file_policy":
        result["file_identity_policy"] = "unsealed"
    else:
        result["contextual_execution_backend"] = "unknown"
    with pytest.raises(ValueError):
        adapt_native_contextual_reference_v1(result, publication)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_flag",
        "wrong_map",
        "wrong_scale",
        "wrong_group_sum",
        "nonfinite_d",
        "asymmetric_d",
        "group_empties_annotation",
    ],
)
def test_adapter_fail_closed_semantic_matrix(mutation: str) -> None:
    result, publication = _native_result()
    if mutation == "missing_flag":
        result["complete_reference_statistics"] = False
    elif mutation == "wrong_map":
        result["pair_eta"] = np.asarray(result["pair_eta"]).copy()
        result["pair_eta"][0] = 2
    elif mutation == "wrong_scale":
        result["scale_plan_sha256"] = "0" * 64
    elif mutation == "wrong_group_sum":
        result["group_gram_unnormalized_num"] = np.asarray(
            result["group_gram_unnormalized_num"]
        ).copy()
        result["group_gram_unnormalized_num"][0, 0, 0] += 1.0
    elif mutation == "nonfinite_d":
        result["same_person"] = np.asarray(result["same_person"]).copy()
        result["same_person"][0, 0] = np.nan
    elif mutation == "asymmetric_d":
        result["same_person"] = np.asarray(result["same_person"]).copy()
        result["same_person"][0, 1] += 1.0
    elif mutation == "group_empties_annotation":
        result["group_annotation_masses"] = np.asarray(
            result["group_annotation_masses"]
        ).copy()
        result["group_annotation_masses"][0, 0] = result["annotation_masses"][0]
        result["group_annotation_masses"][1:, 0] = 0.0
    with pytest.raises(ValueError):
        adapt_native_contextual_reference_v1(result, publication)


def _rewrite_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **values)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "dtype",
        "payload",
        "manifest",
        "numa",
        "file_policy",
        "execution_backend",
        "event",
    ],
)
def test_loader_fail_closed_tamper_matrix(tmp_path: Path, mutation: str) -> None:
    artifact = _artifact()
    path = write_contextual_reference_v1(artifact, tmp_path / mutation)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if mutation == "extra":
        values["row_axis"] = np.arange(artifact.reference_n)
    elif mutation == "dtype":
        values["gram"] = values["gram"].astype(np.float32)
    elif mutation == "payload":
        values["gram"][0, 0] += 1.0
    elif mutation == "manifest":
        manifest = json.loads(str(values["manifest_json"].item()))
        manifest["logical_schema_version"] = "contextual_trait_v1"
        values["manifest_json"] = np.asarray(canonical_json(manifest))
        values["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    else:
        manifest = json.loads(str(values["manifest_json"].item()))
        if mutation == "numa":
            manifest["execution"]["numa"]["numa_verified"] = True
        elif mutation == "file_policy":
            manifest["execution"]["file_identity_policy"] = "unsealed"
        elif mutation == "execution_backend":
            manifest["execution"]["native_execution_backend"] = "unknown"
        else:
            events = manifest["execution"]["telemetry"]["events"]
            index = next(
                index
                for index, event in enumerate(events)
                if event["event_class"] == "protected_call"
            )
            events.pop(index)
            manifest["execution"]["telemetry"]["observed_events"] -= 1
        values["manifest_json"] = np.asarray(canonical_json(manifest))
        values["manifest_sha256"] = np.asarray(canonical_sha256(manifest))
    _rewrite_npz(path, values)
    with pytest.raises(ValueError):
        load_contextual_reference_v1(path)


def _fixture_native_result(
    oracle: Any,
    data: Any,
    scale: GenotypeScalePlanV1,
) -> tuple[dict[str, Any], ContextualReferencePublicationIdentityV1]:
    components = oracle.component_index
    result, publication = _native_result()
    m = int(data["genotype"].shape[1])
    bd = int(data["variant_probes"].shape[1])
    admission, diagnostics, telemetry = _execution_evidence(m, bd)
    component_annotation = np.asarray(
        [entry.annotation_index for entry in components.entries], dtype=np.int64
    )
    component_masses = oracle.annotation_masses[component_annotation]
    variant_allele_sha256 = _sha("fixture-variant-order-alleles")
    result.update(
        {
            "reference_n": int(data["genotype"].shape[0]),
            "n_variants": m,
            "residual_rank": int(data["residual_rank"]),
            "q": components.pair_index.num_basis,
            "sample_probe_count": int(data["sample_probes"].shape[1]),
            "variant_probe_count": int(data["variant_probes"].shape[1]),
            "gram": oracle.gram,
            "same_person": oracle.same_person,
            "group_gram_unnormalized_num": oracle.gram_numerator_contributions,
            "annotation_masses": oracle.annotation_masses,
            "group_annotation_masses": oracle.group_annotation_masses,
            "group_variant_counts": oracle.group_variant_counts,
            "raw_gram_numerator": oracle.gram
            * np.outer(component_masses, component_masses),
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
            "component_annotation": component_annotation,
            "component_pair": np.asarray(
                [entry.pair_index for entry in components.entries], dtype=np.int64
            ),
            "annotation_names": list(components.annotation_names),
            "group_names": list(oracle.loo_group_ids),
            "genotype_scale_policy": scale.policy.value,
            "allele_orientation": scale.allele_orientation,
            "allele_coding": scale.allele_coding,
            "centering_source": scale.centering_source,
            "centering_formula": scale.centering_formula,
            "scaling_formula": scale.scaling_formula,
            "missing_imputation": scale.missing_imputation,
            "ploidy_policy": scale.ploidy_policy,
            "affine_mean_sha256": scale.affine_mean_sha256,
            "affine_inverse_scale_sha256": scale.affine_inverse_scale_sha256,
            "retained_variant_order_sha256": scale.retained_variant_order_sha256,
            "scale_plan_sha256": scale.digest,
            "probe_identity_sha256": array_sha256(data["sample_probes"]),
            "sample_probe_identity_sha256": array_sha256(data["sample_probes"]),
            "variant_probe_identity_sha256": array_sha256(data["variant_probes"]),
            "annotation_map_sha256": array_sha256(data["annotations"]),
            "group_map_sha256": array_sha256(data["group_index"]),
            "group_execution_order_sha256": _sha("fixture-execution-order"),
            "group_execution_permutation_sha256": _sha("fixture-execution-order"),
            "variant_order_allele_sha256": variant_allele_sha256,
            "same_person_sha256": array_sha256(oracle.same_person),
            "group_gram_unnormalized_num_sha256": array_sha256(
                oracle.gram_numerator_contributions
            ),
            "admission": admission,
            "diagnostics": diagnostics,
            "telemetry": telemetry,
        }
    )
    publication = replace(
        publication,
        variant_order_allele_sha256=variant_allele_sha256,
        retained_variant_order_sha256=scale.retained_variant_order_sha256,
        genotype_scale_plan_sha256=scale.digest,
        annotation_map_sha256=result["annotation_map_sha256"],
        annotation_names=tuple(result["annotation_names"]),
        group_map_sha256=result["group_map_sha256"],
        group_ids=tuple(result["group_names"]),
        sample_probe_identity_sha256=result["sample_probe_identity_sha256"],
        variant_probe_identity_sha256=result["variant_probe_identity_sha256"],
    )
    result["file_content_identity"] = _file_content_identity(m)
    _refresh_reference_scientific_digests(result)
    return result, publication


def test_frozen_fixture_all_deletions_fits_and_surfaces_through_explicit_bridge() -> (
    None
):
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "context_native_stage0"
        / "fixture_q2_k2.npz"
    )
    with np.load(fixture_path, allow_pickle=False) as archive:
        data = {name: np.array(archive[name], copy=True) for name in archive.files}
    genotype = data["genotype"]
    phi = data["phi"]
    annotations = data["annotations"]
    projector = rank_revealing_projector(data["fixed_effects"])
    components = ContextComponentIndex(("a0", "a1"), ContextPairIndex(2))
    group_ids = tuple(f"g{int(index)}" for index in data["group_index"])
    hashes = {
        "basis_hash": array_sha256(phi),
        "fixed_effect_hash": array_sha256(data["fixed_effects"]),
        "variant_hash": _sha("fixture-variants"),
    }
    scale = GenotypeScalePlanV1(
        policy=GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1,
        retained_variant_order_sha256=_sha("fixture-retained-order"),
        allele_orientation="bim_a1_counted_v1",
        allele_coding="plink_bed_snp_major_diploid_hardcall_v1",
        centering_source="provided_v1",
        centering_formula="provided_variant_affine_mean_v1",
        scaling_formula="dosage_minus_mean_times_inverse_scale_v1",
        missing_imputation="sealed_mean_v1",
        ploidy_policy="diploid_v1",
        affine_mean_sha256=array_sha256(data["genotype_mean"]),
        affine_inverse_scale_sha256=array_sha256(data["genotype_scale"]),
    )
    oracle = build_context_reference(
        genotype=genotype,
        basis=phi,
        projector=projector,
        annotations=annotations,
        component_index=components,
        loo_groups=group_ids,
        genotype_scaling=scale.digest,
        gram_method="hutchinson",
        gram_probes=data["sample_probes"],
        same_person_method="ustat",
        variant_probes=data["variant_probes"],
        probe_tile_size=3,
        contribution_storage="loo_grouped",
        **hashes,
    )
    summary = build_context_trait_summary(
        genotype=genotype,
        basis=phi,
        phenotype=data["phenotype_raw"],
        projector=projector,
        annotations=annotations,
        component_index=components,
        residual_basis=data["residual_basis"],
        residual_names=("residual", "context_residual"),
        loo_groups=group_ids,
        genotype_scaling=scale.digest,
        block_size=13,
        contribution_storage="loo_grouped",
        **hashes,
    )
    native, publication = _fixture_native_result(oracle, data, scale)
    artifact = adapt_native_contextual_reference_v1(native, publication)
    np.testing.assert_allclose(artifact.gram, data["hutch_gram"], atol=1e-9, rtol=1e-11)
    np.testing.assert_allclose(
        artifact.same_person, data["ustat_same_person"], atol=1e-9, rtol=1e-11
    )
    np.testing.assert_allclose(
        artifact.group_gram_unnormalized_num,
        data["group_gram_unnormalized_num"],
        atol=1e-9,
        rtol=1e-11,
    )
    bridged = artifact.to_development_grouped_reference(
        basis_hash=hashes["basis_hash"],
        fixed_effect_hash=hashes["fixed_effect_hash"],
        variant_hash=hashes["variant_hash"],
        annotation_hash=oracle.manifest["annotation_hash"],
        loo_grouping_hash=oracle.manifest["loo_grouping_hash"],
    )
    grid = np.asarray([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    metric = phi.T @ phi / phi.shape[0]
    observed_fit = fit_context_model(
        bridged,
        summary,
        context_grid=grid,
        basis_metric=metric,
        project_psd=False,
    )
    expected_fit = fit_context_model(
        oracle,
        summary,
        context_grid=grid,
        basis_metric=metric,
        project_psd=False,
    )
    np.testing.assert_allclose(
        observed_fit.raw_coefficients,
        expected_fit.raw_coefficients,
        atol=1e-9,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        observed_fit.loo_coefficients,
        expected_fit.loo_coefficients,
        atol=1e-9,
        rtol=1e-11,
    )
    for observed, expected in zip(
        observed_fit.context_outputs["annotations"],
        expected_fit.context_outputs["annotations"],
    ):
        np.testing.assert_allclose(
            observed["covariance_surface"],
            expected["covariance_surface"],
            atol=1e-9,
            rtol=1e-11,
        )
    for deleted in (("g0",), ("g1",), ("g2",), ("g3",), ("g0", "g2")):
        observed_equations = assemble_context_normal_equations(
            bridged, summary, deleted
        )
        expected_equations = assemble_context_normal_equations(oracle, summary, deleted)
        observed_solve = solve_context_normal_equations(observed_equations)
        expected_solve = solve_context_normal_equations(expected_equations)
        np.testing.assert_allclose(
            observed_solve.coefficients,
            expected_solve.coefficients,
            atol=1e-9,
            rtol=1e-11,
        )
        observed_surface = derive_context_outputs(
            observed_solve.coefficients[: len(components)],
            components,
            grid,
            metric,
        )
        expected_surface = derive_context_outputs(
            expected_solve.coefficients[: len(components)],
            components,
            grid,
            metric,
        )
        for observed, expected in zip(
            observed_surface["annotations"], expected_surface["annotations"]
        ):
            np.testing.assert_allclose(
                observed["covariance_surface"],
                expected["covariance_surface"],
                atol=1e-9,
                rtol=1e-11,
            )
