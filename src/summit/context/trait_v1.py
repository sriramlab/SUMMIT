"""Stable V1 publication boundary for native contextual trait statistics.

This module is deliberately isolated from :mod:`summit.context.summary`, which
continues to own the private NumPy development formats.  The V1 loader and
writer accept only the single-file contextual-trait V1 family.
"""

from __future__ import annotations

from collections import Counter
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ._artifact_io import (
    StableNpzReader,
    _preflight_stable_npz_members,
    _publish_stable_npz_no_replace,
    _validate_stable_npz_writer_temp,
    fsync_parent_directory,
    python_source_runtime_provenance,
    validate_native_build_provenance_consistency,
    validate_native_fault_selector_targets_event,
    validate_python_source_runtime_provenance,
)
from .schema import (
    AnnotationMode,
    ArtifactFamily,
    ContextSchemaIdentityV1,
    DeletionSemantics,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
    GroupedEncodingVersion,
    LogicalSchemaVersion,
    annotation_output_contract,
)
from .spec import (
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_json,
    canonical_sha256,
    freeze_context_mapping,
)


CONTEXTUAL_TRAIT_V1_MAGIC = "SUMMIT_CONTEXTUAL_TRAIT_V1"
CONTEXTUAL_TRAIT_V1_SUFFIX = ".contextual-trait-v1.npz"
_TRAIT_NATIVE_BACKEND_V1 = "plink_bed_descriptor_stream_trait_v1:2"
_FILE_IDENTITY_POLICY_V1 = (
    "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2"
)
_PYTHON_ADAPTER_PROVENANCE_POLICY = (
    "ordered_relative_module_name_and_content_sha256_sha256_v1"
)
_PYTHON_ADAPTER_MODULES = (
    "_artifact_io.py",
    "annotations.py",
    "schema.py",
    "spec.py",
    "trait_v1.py",
)

_ARRAY_NAMES = (
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
_FILE_KEYS = frozenset((*_ARRAY_NAMES, "manifest_json", "manifest_sha256"))

_LAYOUTS = {
    "genetic_rhs": ["component", "trait"],
    "genetic_traces": ["component"],
    "genetic_residual": ["component", "residual_component"],
    "residual_rhs": ["residual_component", "trait"],
    "residual_traces": ["residual_component"],
    "residual_gram": ["residual_component", "residual_component"],
    "group_rhs_unnormalized_num": ["deletion_group", "component", "trait"],
    "group_trace_unnormalized_num": ["deletion_group", "component"],
    "group_genetic_residual_num": [
        "deletion_group",
        "component",
        "residual_component",
    ],
    "annotation_masses": ["annotation"],
    "group_annotation_masses": ["deletion_group", "annotation"],
    "group_variant_counts": ["deletion_group"],
}

_SCIENTIFIC_POLICY = {
    "feature_order": "P_diag_phi_G_v1",
    "pair_order": "diagonal_then_lexicographic_offdiagonal_v1",
    "component_order": "annotation_major_pair_minor_v1",
    "kernel_mass_normalization": "sum_annotation_weight_v1",
    "omega_packing": "offdiag_stored_once_v1",
    "phenotype_normalization": "project_then_unit_residual_variance_v1",
    "trait_scores": "raw_G_transpose_diag_phi_y_v1",
    "residual_moments": "exact_low_rank_v1",
    "descriptor_traversal": "one_per_admitted_trait_batch_v1",
    "population_transfer": "sample_count_same_distinct_v1",
    "deletion": "approx_group_numerator_full_D_v1",
    "raw_fit": "rank_checked_symmetric_v1",
}

_DELETION_POLICY = {
    "semantics": DeletionSemantics.APPROXIMATE_SUMMARY_ONLY_V1.value,
    "grouped_storage": "unnormalized_numerators",
    "group_numerator_unit": "raw_source_annotation_mass_v1",
    "reference_same_person": "reuse_full_unchanged",
    "residual_only_moments": "reuse_full_unchanged",
    "empty_annotation": "reject",
    "multiple_group_subtraction_supported": True,
    "exact_deleted_trait_kernels": False,
    "claim": "approximate_summary_only",
}

_TRAIT_IDENTITY_KEYS = frozenset(
    {
        "sample_order_sha256",
        "variant_order_allele_sha256",
        "fixed_effect_spec_sha256",
        "basis_specification_sha256",
        "basis_calibration_sha256",
        "retained_sample_map_sha256",
        "retained_variant_order_sha256",
        "fixed_basis_sha256",
        "evaluated_phi_sha256",
        "genotype_scale_plan_sha256",
        "missingness_sha256",
        "annotation_map_sha256",
        "annotation_names_sha256",
        "group_map_sha256",
        "group_names_sha256",
        "phenotype_batch_sha256",
        "residual_basis_sha256",
        "trait_names_sha256",
        "residual_names_sha256",
        "sealed_plan_sha256",
        "source_tree_sha256",
    }
)

_BUILD_PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "source_commit",
        "source_tree_sha256",
        "compiler_id",
        "compiler_version",
        "cxx_standard",
        "build_type",
        "sanitizer_mode",
        "asan_enabled",
        "ubsan_enabled",
        "effective_optimization",
        "architecture_tuning",
        "configured_compiler_flags",
        "blas_vendor",
        "gemm_integrity_enabled",
        "gemm_checksum_enabled",
        "private_blas_enabled",
        "private_blas_backend",
        "private_blas_sha256",
        "private_blas_source_commit",
        "private_blas_source_tree_sha256",
        "private_blas_config_family",
        "private_blas_header_sha256",
        "private_blas_cblas_header_sha256",
        "private_openblas_enabled",
        "private_openblas_sha256",
        "native_arch_optimization_enabled",
        "openmp_enabled",
        "contextual_dispatch_backend",
        "contextual_dispatch_vendor_calls",
    }
)

_TRAIT_SCIENTIFIC_ARRAY_KEYS = frozenset(
    {
        *_ARRAY_NAMES,
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
    }
)

_FILE_CONTENT_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "policy",
        "duplicated_descriptors",
        "bim_full_sha256",
        "fam_full_sha256",
        "bed_header_sha256",
        "retained_bed_logical_record_stream_sha256",
        "retained_record_count",
        "bytes_per_bed_record",
        "retained_record_sha256_count",
        "bed_stream_accumulation",
        "bim_fam_hash_source",
        "bed_descriptor_state",
        "bim_descriptor_state",
        "fam_descriptor_state",
        "boundary_sample_evidence",
        "full_bed_file_sha256_claimed",
        "absolute_snapshot_or_lease_claimed",
        "toctou_closed",
        "toctou_nonclaim",
    }
)

_DESCRIPTOR_STATE_KEYS = frozenset(
    {
        "device",
        "inode",
        "size",
        "link_count",
        "mtime_seconds",
        "mtime_nanoseconds",
        "ctime_seconds",
        "ctime_nanoseconds",
    }
)

_MANIFEST_KEYS = frozenset(
    {
        "magic",
        "artifact_family",
        "logical_schema_version",
        "grouped_encoding_version",
        "native_api_version",
        "native_backend_version",
        "build_id",
        "feature_mode",
        "scientific_policy",
        "dimensions",
        "identity",
        "compatible_reference_identity_sha256",
        "genotype_scale_plan",
        "genotype_scale_plan_sha256",
        "phenotype",
        "execution",
        "deletion",
        "layouts",
        "maps",
        "arrays",
        "terminal_status",
    }
)

_NATIVE_RESULT_KEYS = frozenset(
    {
        "complete_trait_artifact",
        "complete_trait_statistics",
        "internal_result_kind",
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
        "study_n",
        "n_variants",
        "residual_rank",
        "q",
        "trait_count",
        "residual_component_count",
        "numeric_policy",
        "deletion_semantics",
        "grouped_encoding",
        "group_numerator_unit",
        "grouped_values_are_unnormalized_numerators",
        "residual_only_deletion",
        "phenotype_normalization",
        "phenotypes_projected_once",
        "feature_mode",
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
        "annotation_names",
        "group_names",
        "trait_names",
        "residual_names",
        "annotation_mode",
        "genotype_scale_policy",
        "allele_orientation",
        "allele_coding",
        "centering_source",
        "centering_formula",
        "scaling_formula",
        "missing_imputation",
        "ploidy_policy",
        "retained_variant_order_sha256",
        "affine_mean_sha256",
        "affine_inverse_scale_sha256",
        "scale_plan_sha256",
        "retained_sample_map_sha256",
        "variant_order_allele_sha256",
        "fixed_basis_sha256",
        "evaluated_phi_sha256",
        "annotation_map_sha256",
        "group_map_sha256",
        "phenotype_batch_sha256",
        "residual_basis_sha256",
        "missingness_sha256",
        "sealed_plan_sha256",
        "source_tree_sha256",
        "execution_plan_sha256",
        "build_provenance",
        "scientific_array_sha256",
        "phase_evidence_sha256",
        "contextual_native_api_version",
        "contextual_backend_version",
        "contextual_backend",
        "contextual_execution_backend",
        "contextual_build_id",
        "file_identity_policy",
        "file_content_identity",
        "output_ownership",
        "numa",
        "admission",
        "diagnostics",
        "telemetry",
        "state",
        "lifecycle",
    }
)

_SEMANTIC_OPERATIONS = (
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

_ADMISSION_KEYS = frozenset(
    {
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
)

_DIAGNOSTIC_KEYS = frozenset(
    {
        "maximum_projection_leakage",
        "phenotype_projection_leakage_max_abs",
        "phenotype_normalized_norm_error_max_abs",
        "residual_gram_pre_symmetry_max_abs",
        "trait_group_reconstruction_max_abs",
        "missing_genotype_calls",
        "observed_descriptor_passes",
        "observed_decoded_blocks",
        "observed_variant_record_visits",
        "observed_phenotype_projections",
        "protected_call_counts",
        "descriptor_accounting_verified",
        "semantic_call_ledger_exact",
        "files_unchanged_at_all_checkpoints",
        "bed_content_evidence",
        "inputs_unchanged_at_all_checkpoints",
        "scratch_released",
        "scratch_released_before_publication",
        "all_large_buffers_preallocated_before_decode",
        "runtime_large_allocations_after_decode",
        "tracked_high_water_bytes",
        "tracked_high_water_within_admission",
        "strict_disjoint_optimized_path",
        "trait_group_reconstruction_verified",
        "phenotype_projection_count_verified",
        "phenotype_normalization_verified",
        "operand_fingerprints_verified",
        "runtime_thread_affinity_fingerprints_verified",
        "independent_scalar_witness_verified",
        "independent_scalar_fallback_available",
        "final_output_identity_fnv64",
    }
)

_TELEMETRY_KEYS = frozenset(
    {
        "capacity",
        "required_capacity",
        "observed_events",
        "injection_count",
        "repair_count",
        "retry_count",
        "fallback_count",
        "complete_without_drop",
        "fault_operation",
        "fault_mode",
        "fault_semantic_anchor",
        "protected_call_counts",
        "events",
    }
)

_EVENT_KEYS = frozenset(
    {
        "event_class",
        "phase",
        "operation",
        "semantic_anchor",
        "semantic_phase",
        "semantic_role",
        "semantic_placement",
        "canonical_begin",
        "canonical_end",
        "variant_coordinate_mode",
        "variant_range_is_exact",
        "variant_membership_count",
        "variant_membership_sha256",
        "resolution",
        "attempt",
        "rows",
        "columns",
        "reduction",
        "left_stride",
        "right_stride",
        "output_stride",
        "transpose_left",
        "sequence",
        "process_id",
        "thread_id",
        "elapsed_ns",
        "operand_fingerprint_fnv64",
        "witness_fingerprint_fnv64",
        "accepted_output_fingerprint_fnv64",
        "runtime_before_fingerprint_fnv64",
        "runtime_after_fingerprint_fnv64",
        "prefix_canary_verified",
        "suffix_canary_verified",
        "finiteness_verified",
        "serialized_entry_verified",
        "deterministic_non_vendor_backend",
        "resident_begin",
        "resident_end",
        "probe_begin",
        "probe_end",
        "variant_begin",
        "variant_end",
        "annotation_begin",
        "annotation_end",
        "context_begin",
        "context_end",
        "action_begin",
        "action_end",
        "group_begin",
        "group_end",
    }
)


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest.") from exc
    if value != value.lower():
        raise ValueError(f"{name} must use canonical lowercase hexadecimal.")
    return value


def _validate_build_provenance(
    value: Any, *, build_id: str, source_tree_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _BUILD_PROVENANCE_KEYS:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            "Native contextual trait build provenance is invalid; "
            f"missing={sorted(_BUILD_PROVENANCE_KEYS - observed)}, "
            f"extra={sorted(observed - _BUILD_PROVENANCE_KEYS)}."
        )
    provenance = _json_value(value)
    validate_native_build_provenance_consistency(
        provenance,
        family="Native contextual trait",
    )
    if (
        provenance["schema"] != "contextual_native_build_provenance_v1"
        or provenance["source_commit"] != build_id
        or _sha256(
            "build_provenance.source_tree_sha256",
            provenance["source_tree_sha256"],
        )
        != source_tree_sha256
        or provenance["cxx_standard"] != 17
        or provenance["contextual_dispatch_backend"]
        != "deterministic_tiled_fp64_with_scalar_witness_v1"
        or provenance["contextual_dispatch_vendor_calls"] is not False
    ):
        raise ValueError("Native contextual trait build provenance disagrees.")
    return provenance


def _validate_scientific_array_digests(
    value: Any, expected_keys: frozenset[str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("Native contextual trait scientific digest map is invalid.")
    return {
        name: _sha256(f"scientific_array_sha256.{name}", value[name])
        for name in sorted(expected_keys)
    }


def _validate_phase_evidence(value: Any) -> dict[str, str]:
    expected = {"residual_derived_state", "post_trait_outputs"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Native contextual trait phase evidence is invalid.")
    return {
        name: _sha256(f"phase_evidence.{name}", value[name])
        for name in sorted(expected)
    }


def _validate_file_content_identity(value: Any, *, n_variants: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FILE_CONTENT_IDENTITY_KEYS:
        raise ValueError("Native contextual trait file-content identity is invalid.")
    identity = _json_value(value)
    if (
        identity["schema"] != "contextual_file_content_identity_v2"
        or identity["policy"] != _FILE_IDENTITY_POLICY_V1
        or identity["duplicated_descriptors"] is not True
        or identity["bed_stream_accumulation"]
        != "authoritative_decode_logical_order_no_extra_bed_traversal_v1"
        or identity["bim_fam_hash_source"] != "full_duplicated_descriptor_bytes_v1"
        or identity["boundary_sample_evidence"] != "first_last_4KiB_per_descriptor_v1"
        or identity["full_bed_file_sha256_claimed"] is not False
        or identity["absolute_snapshot_or_lease_claimed"] is not False
        or identity["toctou_closed"] is not False
        or identity["toctou_nonclaim"]
        != "no_absolute_snapshot_or_lease_mutation_after_last_verified_read_remains_out_of_scope_v1"
    ):
        raise ValueError("Native contextual trait file-content claims disagree.")
    for name in (
        "bim_full_sha256",
        "fam_full_sha256",
        "bed_header_sha256",
        "retained_bed_logical_record_stream_sha256",
    ):
        _sha256(f"file_content_identity.{name}", identity[name])

    def exact_int(name: str, item: Any) -> int:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Native contextual trait {name} is not an integer.")
        return item

    retained = exact_int("retained_record_count", identity["retained_record_count"])
    record_digests = exact_int(
        "retained_record_sha256_count", identity["retained_record_sha256_count"]
    )
    if retained != n_variants or record_digests != retained:
        raise ValueError("Native contextual trait retained-record evidence disagrees.")
    if exact_int("bytes_per_bed_record", identity["bytes_per_bed_record"]) <= 0:
        raise ValueError("Native contextual trait BED record size is invalid.")
    for descriptor in ("bed", "bim", "fam"):
        state = identity[f"{descriptor}_descriptor_state"]
        if not isinstance(state, Mapping) or set(state) != _DESCRIPTOR_STATE_KEYS:
            raise ValueError("Native contextual trait descriptor state is invalid.")
        for name in _DESCRIPTOR_STATE_KEYS:
            exact_int(f"{descriptor}_descriptor_state.{name}", state[name])
        if state["size"] <= 0 or state["link_count"] <= 0:
            raise ValueError("Native contextual trait descriptor size is invalid.")
        for name in ("mtime_nanoseconds", "ctime_nanoseconds"):
            if not 0 <= state[name] < 1_000_000_000:
                raise ValueError("Native contextual trait timestamp is invalid.")
    return identity


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _json_value(value: Any) -> Any:
    """Own metadata as strict canonical-JSON data."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Contextual trait V1 metadata keys must be strings.")
            if key in result:
                raise ValueError(f"Duplicate contextual trait V1 key {key!r}.")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(
                "Contextual trait V1 metadata cannot contain nonfinite values."
            )
        return value
    raise ValueError(
        f"Contextual trait V1 metadata contains unsupported value "
        f"{type(value).__name__}."
    )


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate contextual trait V1 JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("Contextual trait V1 manifest is not valid JSON.") from exc


def _immutable_array(value: Any, *, dtype: Any) -> np.ndarray:
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    source = np.ascontiguousarray(value, dtype=canonical_dtype)
    storage = bytes(source.tobytes(order="C"))
    result = np.frombuffer(storage, dtype=source.dtype).reshape(source.shape)
    result.setflags(write=False)
    return result


def _array_envelope(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": value.dtype.newbyteorder("<").str,
        "shape": list(value.shape),
        "sha256": array_sha256(value),
    }


def _relative_max(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / scale, initial=0.0))


def _native_array(
    result: Mapping[str, Any], name: str, *, dtype: Any, ndim: int
) -> np.ndarray:
    value = _required_native(result, name)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Native contextual trait {name} is not an ndarray.")
    if value.dtype != np.dtype(dtype) or value.ndim != ndim:
        raise ValueError(f"Native contextual trait {name} dtype/rank mismatch.")
    if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
        raise ValueError(f"Native contextual trait {name} contains nonfinite values.")
    return value


def _required_native(result: Mapping[str, Any], name: str) -> Any:
    if name not in result:
        raise ValueError(f"Native contextual trait result is missing {name!r}.")
    return result[name]


def _native_u64(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Native contextual trait {name} is not an integer.")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError(f"Native contextual trait {name} is outside uint64.")
    return result


def _native_fnv64(name: str, value: Any) -> int:
    if not isinstance(value, str) or not value or not value.isdecimal():
        raise ValueError(f"Native contextual trait {name} is not decimal uint64.")
    result = int(value, 10)
    if result < 0 or result >= 1 << 64 or str(result) != value:
        raise ValueError(f"Native contextual trait {name} is not canonical uint64.")
    return result


def _names(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Native contextual trait {name} must be a nonempty sequence.")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"Native contextual trait {name} contains an invalid name.")
    if len(set(result)) != len(result):
        raise ValueError(f"Native contextual trait {name} must be unique.")
    return result


def _validate_descriptor_scale_plan(scale_plan: GenotypeScalePlanV1) -> None:
    if scale_plan.policy is not GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1:
        raise ValueError("Contextual trait V1 requires sealed affine scaling.")
    if scale_plan.allele_orientation not in {
        "bim_a1_counted_v1",
        "bim_a2_counted_v1",
        "mixed_bim_a1_a2_per_variant_v1",
    }:
        raise ValueError("Contextual trait V1 allele orientation is unsupported.")
    expected = {
        "allele_coding": "plink_bed_snp_major_diploid_hardcall_v1",
        "centering_source": "provided_v1",
        "centering_formula": "provided_variant_affine_mean_v1",
        "scaling_formula": "dosage_minus_mean_times_inverse_scale_v1",
        "missing_imputation": "sealed_mean_v1",
        "ploidy_policy": "diploid_v1",
    }
    for field, value in expected.items():
        if getattr(scale_plan, field) != value:
            raise ValueError(
                f"Contextual trait V1 scale field {field!r} is unsupported."
            )


def _name_map_digest(axis: str, names: Sequence[str]) -> str:
    return canonical_sha256({"axis": axis, "names": list(names)})


def _validate_numa(value: Any) -> dict[str, Any]:
    expected = {
        "numa_applicable",
        "numa_verified",
        "policy",
        "output_numa_node",
        "reason",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Native contextual trait NUMA evidence schema mismatch.")
    if (
        value["numa_applicable"] is not False
        or value["numa_verified"] is not False
        or value["policy"] != "unbound_first_touch_v1"
        or isinstance(value["output_numa_node"], bool)
        or not isinstance(value["output_numa_node"], (int, np.integer))
        or int(value["output_numa_node"]) != -1
        or value["reason"]
        != "unbound standard allocator; no placement or output-node claim"
    ):
        raise ValueError("Native contextual trait NUMA evidence is invalid.")
    return _json_value(value)


def _validate_execution_evidence(
    admission: Any,
    diagnostics: Any,
    telemetry: Any,
    *,
    n: int,
    m: int,
    q: int,
    k: int,
    c: int,
    l: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate the closed Stage-4 execution-evidence vocabulary."""
    if not isinstance(admission, Mapping) or set(admission) != _ADMISSION_KEYS:
        raise ValueError("Native contextual trait admission schema mismatch.")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != _DIAGNOSTIC_KEYS:
        raise ValueError("Native contextual trait diagnostics schema mismatch.")
    if not isinstance(telemetry, Mapping) or set(telemetry) != _TELEMETRY_KEYS:
        raise ValueError("Native contextual trait telemetry schema mismatch.")

    calls = admission["semantic_call_ledger"]
    if not isinstance(calls, Mapping) or set(calls) != set(_SEMANTIC_OPERATIONS):
        raise ValueError("Native contextual trait semantic-call schema mismatch.")
    call_counts = {
        name: _native_u64(f"semantic_call_ledger.{name}", calls[name])
        for name in _SEMANTIC_OPERATIONS
    }
    if any(call_counts[name] != 0 for name in _SEMANTIC_OPERATIONS[:14]):
        raise ValueError("Trait execution reported reference-only protected calls.")
    if any(call_counts[name] == 0 for name in _SEMANTIC_OPERATIONS[14:]):
        raise ValueError("Native contextual trait omitted an active trait operation.")
    total_calls = sum(call_counts.values())
    if (
        _native_u64("total_protected_calls", admission["total_protected_calls"])
        != total_calls
    ):
        raise ValueError("Native contextual trait protected-call total mismatch.")

    metadata_admission = {
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
    admitted = {
        name: _native_u64(f"admission.{name}", admission[name])
        for name in _ADMISSION_KEYS - metadata_admission
    }
    if admitted["trait_descriptor_passes"] != 1:
        raise ValueError("Native contextual trait must make one descriptor pass.")
    if admitted["trait_decoded_blocks"] < 1:
        raise ValueError("Native contextual trait decoded no genotype blocks.")
    if admitted["total_descriptor_passes"] != admitted["trait_descriptor_passes"]:
        raise ValueError("Native contextual trait descriptor-pass total mismatch.")
    if admitted["total_decoded_blocks"] != admitted["trait_decoded_blocks"]:
        raise ValueError("Native contextual trait decoded-block total mismatch.")
    if admitted["total_variant_record_visits"] != m:
        raise ValueError("Native contextual trait variant-visit total mismatch.")
    if admitted["maximum_protected_output_elements"] < 1:
        raise ValueError("Native contextual trait protected-output admission is zero.")
    memory_names = (
        "permanent_bytes",
        "residual_phase_bytes",
        "trait_phase_bytes",
        "compact_output_bytes",
        "integrity_reserve_bytes",
        "telemetry_bytes",
    )
    expected_workspace = sum(admitted[name] for name in memory_names)
    if admitted["required_workspace_bytes"] != expected_workspace:
        raise ValueError("Native contextual trait workspace ledger mismatch.")
    if admitted["required_telemetry_capacity"] != total_calls * 8 + 32:
        raise ValueError("Native contextual trait telemetry admission mismatch.")

    memory_ledger = admission["memory_ledger"]
    if not isinstance(memory_ledger, Mapping) or set(memory_ledger) != set(
        memory_names
    ):
        raise ValueError("Native contextual trait memory-ledger schema mismatch.")
    for name in memory_names:
        if _native_u64(f"memory_ledger.{name}", memory_ledger[name]) != admitted[name]:
            raise ValueError(
                "Native contextual trait memory ledger disagrees with admission."
            )
    lifetime_policy = {
        "permanent_bytes": ("admission_through_publication", 1),
        "residual_phase_bytes": (
            "residual_moments_and_trait_preallocated_arena",
            1,
        ),
        "trait_phase_bytes": ("preallocated_trait_arena", 1),
        "compact_output_bytes": ("publication", 1),
        "integrity_reserve_bytes": ("all_protected_phases", 4),
        "telemetry_bytes": ("admission_through_publication", 1),
    }
    lifetimes = admission["memory_lifetimes"]
    if not isinstance(lifetimes, Mapping) or set(lifetimes) != set(memory_names):
        raise ValueError("Native contextual trait memory-lifetime schema mismatch.")
    for name, (phase, copies) in lifetime_policy.items():
        item = lifetimes[name]
        if not isinstance(item, Mapping) or set(item) != {
            "byte_count",
            "live_phase",
            "concurrent_copies",
        }:
            raise ValueError("Native contextual trait memory lifetime is malformed.")
        if (
            _native_u64(f"memory_lifetimes.{name}.byte_count", item["byte_count"])
            != admitted[name]
            or item["live_phase"] != phase
            or _native_u64(
                f"memory_lifetimes.{name}.concurrent_copies",
                item["concurrent_copies"],
            )
            != copies
        ):
            raise ValueError("Native contextual trait memory lifetime mismatch.")
    if admission["memory_accounting_model"] != (
        "tracked_vector_payload_bytes_v1_excludes_allocator_metadata_and_small_strings"
    ):
        raise ValueError("Native contextual trait memory model mismatch.")

    tiles = admission["selected_tiles"]
    if not isinstance(tiles, Mapping) or set(tiles) != {
        "variant_block",
        "trait_feature_tile",
    }:
        raise ValueError("Native contextual trait selected-tile schema mismatch.")
    if any(
        _native_u64(f"selected_tiles.{name}", value) < 1
        for name, value in tiles.items()
    ):
        raise ValueError("Native contextual trait selected tile is zero.")

    phases = admission["phase_ledger"]
    if not isinstance(phases, Mapping) or set(phases) != {
        "residual_moments",
        "trait",
    }:
        raise ValueError("Native contextual trait phase-ledger schema mismatch.")
    residual_phase = phases["residual_moments"]
    if not isinstance(residual_phase, Mapping) or set(residual_phase) != {
        "phenotype_projections"
    }:
        raise ValueError("Native contextual trait residual phase ledger mismatch.")
    if (
        _native_u64(
            "phase_ledger.residual_moments.phenotype_projections",
            residual_phase["phenotype_projections"],
        )
        != l
    ):
        raise ValueError("Native contextual trait phenotype-projection count mismatch.")
    trait_phase = phases["trait"]
    trait_phase_keys = {
        "descriptor_passes",
        "decoded_blocks",
        "variant_record_visits",
        "logical_bed_record_bytes_touched",
        "access_order",
        "logical_duplicate_variant_decodes",
        "phenotypes_in_packed_rhs",
    }
    if not isinstance(trait_phase, Mapping) or set(trait_phase) != trait_phase_keys:
        raise ValueError("Native contextual trait phase ledger mismatch.")
    phase_values = {
        name: _native_u64(f"phase_ledger.trait.{name}", trait_phase[name])
        for name in trait_phase_keys - {"access_order"}
    }
    if (
        phase_values["descriptor_passes"] != admitted["trait_descriptor_passes"]
        or phase_values["decoded_blocks"] != admitted["trait_decoded_blocks"]
        or phase_values["variant_record_visits"] != m
        or phase_values["logical_duplicate_variant_decodes"] != 0
        or phase_values["phenotypes_in_packed_rhs"] != l
    ):
        raise ValueError("Native contextual trait phase accounting mismatch.")
    logical_bytes = phase_values["logical_bed_record_bytes_touched"]
    if logical_bytes == 0 or logical_bytes % m:
        raise ValueError("Native contextual trait logical-byte evidence is invalid.")
    if trait_phase["access_order"] != "retained_logical_sequential_v1":
        raise ValueError("Native contextual trait access-order evidence mismatch.")
    if (
        admission["os_physical_read_bytes_measured"] is not False
        or admission["os_page_faults_measured"] is not False
        or admission["physical_io_evidence"] != "logical_mmap_record_touches_only_v1"
    ):
        raise ValueError("Native contextual trait physical-I/O claim is invalid.")

    required_true = (
        "descriptor_accounting_verified",
        "semantic_call_ledger_exact",
        "files_unchanged_at_all_checkpoints",
        "inputs_unchanged_at_all_checkpoints",
        "scratch_released",
        "scratch_released_before_publication",
        "all_large_buffers_preallocated_before_decode",
        "tracked_high_water_within_admission",
        "trait_group_reconstruction_verified",
        "phenotype_projection_count_verified",
        "phenotype_normalization_verified",
        "operand_fingerprints_verified",
        "runtime_thread_affinity_fingerprints_verified",
        "independent_scalar_witness_verified",
        "independent_scalar_fallback_available",
    )
    if any(diagnostics[name] is not True for name in required_true):
        raise ValueError("Native contextual trait integrity evidence is incomplete.")
    if not isinstance(diagnostics["strict_disjoint_optimized_path"], bool):
        raise ValueError("Native contextual trait strict-disjoint flag is not boolean.")
    if diagnostics["bed_content_evidence"] != (
        "boundary_sample_only_no_extra_full_BED_traversal"
    ):
        raise ValueError("Native contextual trait BED-content evidence is invalid.")
    if dict(diagnostics["protected_call_counts"]) != call_counts:
        raise ValueError("Native contextual trait diagnostic call ledger mismatch.")
    expected_diagnostic_counts = {
        "observed_descriptor_passes": 1,
        "observed_decoded_blocks": admitted["trait_decoded_blocks"],
        "observed_variant_record_visits": m,
        "observed_phenotype_projections": l,
        "runtime_large_allocations_after_decode": 0,
    }
    for name, expected in expected_diagnostic_counts.items():
        if _native_u64(f"diagnostics.{name}", diagnostics[name]) != expected:
            raise ValueError(f"Native contextual trait diagnostic {name} mismatch.")
    for name in (
        "missing_genotype_calls",
        "tracked_high_water_bytes",
    ):
        _native_u64(f"diagnostics.{name}", diagnostics[name])
    fingerprint = diagnostics["final_output_identity_fnv64"]
    if not isinstance(fingerprint, str) or not fingerprint.isdecimal():
        raise ValueError("Native contextual trait output fingerprint is invalid.")
    if (
        int(diagnostics["tracked_high_water_bytes"])
        > admitted["required_workspace_bytes"]
    ):
        raise ValueError("Native contextual trait high-water mark exceeds admission.")
    for name in (
        "maximum_projection_leakage",
        "phenotype_projection_leakage_max_abs",
        "phenotype_normalized_norm_error_max_abs",
        "residual_gram_pre_symmetry_max_abs",
        "trait_group_reconstruction_max_abs",
    ):
        value = diagnostics[name]
        if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
            raise ValueError(f"Native contextual trait diagnostic {name} is not fp64.")
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"Native contextual trait diagnostic {name} is invalid.")

    if telemetry["complete_without_drop"] is not True:
        raise ValueError("Native contextual trait telemetry was dropped.")
    if dict(telemetry["protected_call_counts"]) != call_counts:
        raise ValueError("Native contextual trait telemetry call ledger mismatch.")
    capacity = _native_u64("telemetry.capacity", telemetry["capacity"])
    required_capacity = _native_u64(
        "telemetry.required_capacity", telemetry["required_capacity"]
    )
    observed_events = _native_u64(
        "telemetry.observed_events", telemetry["observed_events"]
    )
    if (
        required_capacity != admitted["required_telemetry_capacity"]
        or capacity < required_capacity
    ):
        raise ValueError("Native contextual trait telemetry capacity mismatch.")
    events = telemetry["events"]
    if (
        not isinstance(events, list)
        or observed_events != len(events)
        or observed_events > capacity
    ):
        raise ValueError("Native contextual trait telemetry event count mismatch.")
    class_resolutions = {
        "phase_transition": {"residual_moments_running", "trait_running"},
        "mutation_checkpoint": {
            "post_seal",
            "post_admission",
            "post_residual_moments",
            "post_trait",
            "pre_finalization",
            "pre_publication",
        },
        "protected_call": {"primary_deterministic_tiled_fp64"},
        "fault_injection": {
            "one_shot",
            "repeated",
            "repair_corruption",
            "force_fallback",
            "fallback_corruption",
            "fallback_failure",
            "nan",
            "inf",
            "canary",
            "operand_mutation",
            "runtime_mutation",
        },
        "fault_detection": {
            "operand_fingerprint_mismatch",
            "runtime_fingerprint_mismatch",
            "prefix_canary_mismatch",
            "suffix_canary_mismatch",
            "nonfinite_output",
            "scalar_witness_mismatch",
            "forced_fallback_policy",
            "retry_scalar_witness_mismatch",
            "fallback_scalar_witness_mismatch",
            "injected_trusted_fallback_failure",
        },
        "repair": {"verified_retry"},
        "retry": {"deterministic_tiled_retry"},
        "trusted_fallback": {"one_thread_scalar_fp64"},
        "semantic_verification": {"scalar_witness_agreement"},
        "scratch_release": {"execution_arena_released"},
        "publication": {"compact_trait_statistics_ready"},
    }
    protected_events = {name: 0 for name in _SEMANTIC_OPERATIONS}
    class_counts: dict[str, int] = {}
    phase_transitions: list[str] = []
    checkpoints: list[str] = []
    scalar_witness_events = 0
    fault_injection_records: list[dict[str, Any]] = []
    fault_detection_records: list[tuple[str, str]] = []
    recovery_chain_records: list[tuple[str, str]] = []
    protected_anchor_counts: Counter[tuple[str, str]] = Counter()
    verification_anchor_counts: Counter[tuple[str, str]] = Counter()
    event_owner: tuple[int, int] | None = None
    coordinate_limits = {
        "resident": n,
        "probe": l,
        "variant": m,
        "annotation": k,
        "context": q,
        # The native trait projection reuses the historical action coordinate
        # for packed (context, variant) feature columns.
        "action": max(c, l, q * m),
        "group": 0,
    }
    for sequence, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
            raise ValueError("Native contextual trait telemetry event schema mismatch.")
        event_class = event["event_class"]
        if event_class not in class_resolutions:
            raise ValueError(
                "Native contextual trait telemetry event class is unknown."
            )
        if event["resolution"] not in class_resolutions[event_class]:
            raise ValueError("Native contextual trait telemetry resolution is invalid.")
        class_counts[event_class] = class_counts.get(event_class, 0) + 1
        operation = event["operation"]
        if operation not in {*_SEMANTIC_OPERATIONS, "none"}:
            raise ValueError("Native contextual trait telemetry operation is unknown.")
        if not isinstance(event["phase"], str) or event["phase"] not in {
            "admission",
            "residual_moments",
            "trait",
            "finalization",
            "publication",
        }:
            raise ValueError("Native contextual trait telemetry phase is invalid.")
        semantic_phase = event["semantic_phase"]
        semantic_role = event["semantic_role"]
        semantic_placement = event["semantic_placement"]
        if (
            semantic_phase
            not in {
                "unspecified",
                "source",
                "action",
                "gram",
                "group",
                "same_person",
                "residual_moments",
                "trait",
                "finalization",
                "publication",
            }
            or semantic_role
            not in {
                "ordinary",
                "tile",
                "global_merge",
            }
            or semantic_placement
            not in {
                "none",
                "action_scaled",
                "genotype_scaled",
            }
        ):
            raise ValueError(
                "Native contextual trait semantic classification is invalid."
            )
        if operation == "none":
            expected_semantic_phase = "unspecified"
            allowed_semantic_roles = {"ordinary"}
            allowed_semantic_placements = {"none"}
        else:
            operation_index = _SEMANTIC_OPERATIONS.index(operation)
            if operation_index <= 3:
                expected_semantic_phase = "source"
            elif operation_index <= 5:
                expected_semantic_phase = "action"
            elif operation_index == 6:
                expected_semantic_phase = "gram"
            elif operation_index <= 9:
                expected_semantic_phase = "group"
            elif operation_index <= 13:
                expected_semantic_phase = "same_person"
            else:
                expected_semantic_phase = "trait"
            allowed_semantic_roles = (
                {"tile", "global_merge"}
                if operation == "same_person_gram_tn"
                else {"ordinary"}
            )
            allowed_semantic_placements = (
                {"action_scaled", "genotype_scaled"}
                if operation == "direct_grouped_tn"
                else {"none"}
            )
        if (
            semantic_phase != expected_semantic_phase
            or semantic_role not in allowed_semantic_roles
            or semantic_placement not in allowed_semantic_placements
        ):
            raise ValueError(
                "Native contextual trait semantic operation classification is "
                "inconsistent."
            )
        lifecycle_phase_by_resolution = {
            "post_seal": "admission",
            "post_admission": "admission",
            "residual_moments_running": "residual_moments",
            "post_residual_moments": "residual_moments",
            "trait_running": "trait",
            "post_trait": "trait",
            "pre_finalization": "finalization",
            "execution_arena_released": "finalization",
            "pre_publication": "finalization",
            "compact_trait_statistics_ready": "publication",
        }
        expected_event_phase = (
            semantic_phase
            if operation != "none"
            else lifecycle_phase_by_resolution.get(event["resolution"])
        )
        if expected_event_phase is None or event["phase"] != expected_event_phase:
            raise ValueError("Native contextual trait event phase is inconsistent.")
        canonical_begin = _native_u64(
            "telemetry.event.canonical_begin", event["canonical_begin"]
        )
        canonical_end = _native_u64(
            "telemetry.event.canonical_end", event["canonical_end"]
        )
        if canonical_begin > canonical_end:
            raise ValueError("Native contextual trait canonical range is invalid.")
        variant_coordinate_mode = event["variant_coordinate_mode"]
        variant_range_is_exact = event["variant_range_is_exact"]
        variant_membership_count = _native_u64(
            "telemetry.event.variant_membership_count",
            event["variant_membership_count"],
        )
        variant_membership_sha256 = event["variant_membership_sha256"]
        if not isinstance(variant_range_is_exact, bool):
            raise ValueError(
                "Native contextual trait variant exactness claim is invalid."
            )
        if variant_coordinate_mode == "logical_half_open_range_v1":
            if (
                variant_range_is_exact is not True
                or variant_membership_count != 0
                or variant_membership_sha256 != ""
            ):
                raise ValueError(
                    "Native contextual trait logical variant coordinate is invalid."
                )
        elif variant_coordinate_mode == "exact_membership_sha256_v1":
            raise ValueError(
                "Native contextual trait operations must use logical variant ranges."
            )
        else:
            raise ValueError(
                "Native contextual trait variant coordinate mode is unknown."
            )
        process_id = _native_u64("telemetry.event.process_id", event["process_id"])
        thread_id = _native_u64("telemetry.event.thread_id", event["thread_id"])
        observed_owner = (process_id, thread_id)
        if event_owner is None:
            event_owner = observed_owner
        if (
            _native_u64("telemetry.event.sequence", event["sequence"]) != sequence
            or process_id == 0
            or thread_id == 0
            or observed_owner != event_owner
        ):
            raise ValueError(
                "Native contextual trait event sequence/ownership is invalid."
            )
        _native_u64("telemetry.event.elapsed_ns", event["elapsed_ns"])
        fingerprints = {
            name: _native_fnv64(f"telemetry.event.{name}", event[name])
            for name in (
                "operand_fingerprint_fnv64",
                "witness_fingerprint_fnv64",
                "accepted_output_fingerprint_fnv64",
                "runtime_before_fingerprint_fnv64",
                "runtime_after_fingerprint_fnv64",
            )
        }
        integrity_flags = (
            "prefix_canary_verified",
            "suffix_canary_verified",
            "finiteness_verified",
            "serialized_entry_verified",
            "deterministic_non_vendor_backend",
        )
        if any(not isinstance(event[name], bool) for name in integrity_flags):
            raise ValueError("Native contextual trait integrity flags must be boolean.")
        attempt = _native_u64("telemetry.event.attempt", event["attempt"])
        if attempt > 3 or not isinstance(event["transpose_left"], bool):
            raise ValueError(
                "Native contextual trait telemetry attempt/type is invalid."
            )
        numeric_shape = {
            name: _native_u64(f"telemetry.event.{name}", event[name])
            for name in (
                "rows",
                "columns",
                "reduction",
                "left_stride",
                "right_stride",
                "output_stride",
            )
        }
        coordinates: dict[str, tuple[int, int]] = {}
        for axis, limit in coordinate_limits.items():
            begin = _native_u64(f"telemetry.event.{axis}_begin", event[f"{axis}_begin"])
            end = _native_u64(f"telemetry.event.{axis}_end", event[f"{axis}_end"])
            if begin > end or end > limit:
                raise ValueError(
                    "Native contextual trait telemetry coordinate is invalid."
                )
            coordinates[axis] = (begin, end)
        variant_width = coordinates["variant"][1] - coordinates["variant"][0]
        if variant_coordinate_mode == "exact_membership_sha256_v1":
            if (
                variant_membership_count > variant_width
                or variant_range_is_exact
                is not (variant_membership_count == variant_width)
                or (canonical_begin, canonical_end) != coordinates["variant"]
            ):
                raise ValueError(
                    "Native contextual trait exact variant envelope is invalid."
                )
        if operation in {
            "trait_feature_projection_tn",
            "trait_feature_projection_nn",
        }:
            context_begin, context_end = coordinates["context"]
            expected_action = (
                context_begin * m + coordinates["variant"][0],
                context_begin * m + coordinates["variant"][1],
            )
            if (
                context_end != context_begin + 1
                or coordinates["action"] != expected_action
                or (canonical_begin, canonical_end) != expected_action
            ):
                raise ValueError(
                    "Native contextual trait global feature coordinate is invalid."
                )
        expected_anchor = (
            f"{operation}|resident={coordinates['resident'][0]}:"
            f"{coordinates['resident'][1]}"
            f"|probe={coordinates['probe'][0]}:{coordinates['probe'][1]}"
            f"|variant={coordinates['variant'][0]}:{coordinates['variant'][1]}"
            f"|annotation={coordinates['annotation'][0]}:"
            f"{coordinates['annotation'][1]}"
            f"|context={coordinates['context'][0]}:{coordinates['context'][1]}"
            f"|action={coordinates['action'][0]}:{coordinates['action'][1]}"
        )
        expected_anchor += (
            f"|phase={semantic_phase}|role={semantic_role}"
            f"|placement={semantic_placement}"
            f"|canonical={canonical_begin}:{canonical_end}"
        )
        if variant_coordinate_mode == "exact_membership_sha256_v1":
            expected_anchor += (
                f"|variant_mode={variant_coordinate_mode}"
                f"|variant_count={variant_membership_count}"
                f"|variant_sha256={variant_membership_sha256}"
            )
        if event["semantic_anchor"] != expected_anchor:
            raise ValueError("Native contextual trait semantic anchor is invalid.")
        protected_geometry_event = operation != "none" and event_class in {
            "protected_call",
            "fault_injection",
            "fault_detection",
            "repair",
            "retry",
            "trusted_fallback",
            "semantic_verification",
        }
        diagnostic_geometry_event = event_class in {
            "fault_injection",
            "fault_detection",
        }
        if protected_geometry_event:
            if (
                operation not in _SEMANTIC_OPERATIONS[14:]
                or any(
                    numeric_shape[name] == 0
                    for name in ("rows", "columns", "reduction")
                )
                or numeric_shape["output_stride"] < numeric_shape["rows"]
                or numeric_shape["right_stride"] < numeric_shape["reduction"]
                or numeric_shape["left_stride"]
                < (
                    numeric_shape["reduction"]
                    if event["transpose_left"]
                    else numeric_shape["rows"]
                )
                or event["transpose_left"] != operation.endswith("_tn")
            ):
                raise ValueError(
                    "Native contextual trait protected-attempt event is invalid."
                )
            if any(value == 0 for value in fingerprints.values()):
                raise ValueError(
                    "Native contextual trait protected evidence is incomplete."
                )
            if diagnostic_geometry_event:
                if (
                    event["serialized_entry_verified"] is not True
                    or event["deterministic_non_vendor_backend"] is not True
                ):
                    raise ValueError(
                        "Native contextual trait fault evidence is incomplete."
                    )
            elif any(event[name] is not True for name in integrity_flags):
                raise ValueError(
                    "Native contextual trait protected evidence is incomplete."
                )
        if event_class == "protected_call":
            if attempt != 1:
                raise ValueError("Native contextual trait primary attempt is invalid.")
            protected_events[operation] += 1
            protected_anchor_counts[(operation, event["semantic_anchor"])] += 1
        elif event_class in {"retry", "repair"} and attempt != 2:
            raise ValueError("Native contextual trait retry/repair attempt is invalid.")
        elif event_class == "trusted_fallback" and attempt != 3:
            raise ValueError("Native contextual trait fallback attempt is invalid.")
        elif event_class == "fault_injection" and (operation == "none" or attempt != 1):
            raise ValueError("Native contextual trait fault injection is invalid.")
        elif event_class == "fault_injection":
            fault_injection_records.append(dict(event))
        elif event_class == "fault_detection":
            fault_detection_records.append((operation, event["semantic_anchor"]))
            expected_detection_attempt = {
                "retry_scalar_witness_mismatch": 2,
                "fallback_scalar_witness_mismatch": 3,
                "injected_trusted_fallback_failure": 3,
            }.get(event["resolution"], 1)
            if operation == "none" or attempt != expected_detection_attempt:
                raise ValueError("Native contextual trait fault detection is invalid.")
        elif not protected_geometry_event and any(numeric_shape.values()):
            raise ValueError(
                "Native contextual trait lifecycle event carries GEMM dimensions."
            )
        elif (
            not protected_geometry_event
            and event_class != "semantic_verification"
            and attempt != 0
        ):
            raise ValueError("Native contextual trait lifecycle attempt is invalid.")
        if event_class in {"retry", "repair", "trusted_fallback"}:
            recovery_chain_records.append((operation, event["semantic_anchor"]))
        if event_class == "semantic_verification":
            if operation not in _SEMANTIC_OPERATIONS[14:] or attempt not in (1, 2, 3):
                raise ValueError("Native contextual trait scalar witness is invalid.")
            scalar_witness_events += 1
            verification_anchor_counts[(operation, event["semantic_anchor"])] += 1
        elif event_class == "phase_transition":
            phase_transitions.append(event["resolution"])
        elif event_class == "mutation_checkpoint":
            checkpoints.append(event["resolution"])
    if protected_events != call_counts:
        raise ValueError("Native contextual trait protected-event ledger mismatch.")
    if verification_anchor_counts != protected_anchor_counts:
        raise ValueError(
            "Native contextual trait protected-call/scalar-verification attribution "
            "is inconsistent."
        )
    if phase_transitions != ["residual_moments_running", "trait_running"]:
        raise ValueError(
            "Native contextual trait phase-transition sequence is incomplete."
        )
    if checkpoints != [
        "post_seal",
        "post_admission",
        "post_residual_moments",
        "post_trait",
        "pre_finalization",
        "pre_publication",
    ]:
        raise ValueError("Native contextual trait checkpoint sequence is incomplete.")
    injection_count = _native_u64(
        "telemetry.injection_count", telemetry["injection_count"]
    )
    repair_count = _native_u64("telemetry.repair_count", telemetry["repair_count"])
    retry_count = _native_u64("telemetry.retry_count", telemetry["retry_count"])
    fallback_count = _native_u64(
        "telemetry.fallback_count", telemetry["fallback_count"]
    )
    if (
        class_counts.get("scratch_release", 0) != 1
        or class_counts.get("publication", 0) != 1
        or not events
        or events[-1]["event_class"] != "publication"
        or scalar_witness_events != total_calls
        or class_counts.get("fault_injection", 0) != injection_count
        or class_counts.get("repair", 0) != repair_count
        or class_counts.get("retry", 0) != retry_count
        or class_counts.get("trusted_fallback", 0) != fallback_count
    ):
        raise ValueError("Native contextual trait telemetry lifecycle mismatch.")
    if not all(
        isinstance(telemetry[name], str)
        for name in ("fault_operation", "fault_mode", "fault_semantic_anchor")
    ):
        raise ValueError("Native contextual trait fault metadata is invalid.")
    if injection_count == 0:
        if (
            telemetry["fault_operation"]
            or telemetry["fault_mode"] != "none"
            or telemetry["fault_semantic_anchor"]
            or repair_count
            or retry_count
            or fallback_count
            or class_counts.get("fault_detection", 0)
        ):
            raise ValueError(
                "Native contextual trait no-fault telemetry is inconsistent."
            )
    else:
        expected_recovery = {
            "one_shot": (1, 1, 0, 1),
            "nan": (1, 1, 0, 1),
            "inf": (1, 1, 0, 1),
            "canary": (1, 1, 0, 1),
            "repeated": (0, 1, 1, 2),
            "repair_corruption": (0, 1, 1, 2),
            "force_fallback": (0, 0, 1, 1),
        }
        mode = telemetry["fault_mode"]
        if injection_count != 1 or len(fault_injection_records) != 1:
            raise ValueError(
                "Native contextual trait recovered-fault telemetry is inconsistent."
            )
        injected_event = fault_injection_records[0]
        validate_native_fault_selector_targets_event(
            telemetry["fault_semantic_anchor"],
            operation=telemetry["fault_operation"],
            event=injected_event,
            grouped=False,
            identity_required=False,
            family="Native contextual trait",
        )
        if (
            injected_event["operation"] != telemetry["fault_operation"]
            or injected_event["resolution"] != telemetry["fault_mode"]
            or protected_anchor_counts.get(
                (
                    telemetry["fault_operation"],
                    injected_event["semantic_anchor"],
                ),
                0,
            )
            == 0
            or any(
                record
                != (
                    telemetry["fault_operation"],
                    injected_event["semantic_anchor"],
                )
                for record in fault_detection_records
            )
            or any(
                record
                != (
                    telemetry["fault_operation"],
                    injected_event["semantic_anchor"],
                )
                for record in recovery_chain_records
            )
            or telemetry["fault_operation"] not in _SEMANTIC_OPERATIONS[14:]
            or not telemetry["fault_semantic_anchor"]
            or mode not in expected_recovery
            or (
                repair_count,
                retry_count,
                fallback_count,
                class_counts.get("fault_detection", 0),
            )
            != expected_recovery[mode]
        ):
            raise ValueError(
                "Native contextual trait recovered-fault telemetry is inconsistent."
            )
    return _json_value(admission), _json_value(diagnostics), _json_value(telemetry)


@dataclass(frozen=True)
class ContextualTraitPublicationIdentityV1:
    """Python-owned semantic identities not recoverable from compact moments."""

    sample_order_sha256: str
    variant_order_allele_sha256: str
    fixed_effect_spec_sha256: str
    basis_specification_sha256: str
    basis_calibration_sha256: str
    compatible_reference_identity_sha256: str
    retained_sample_map_sha256: str
    retained_variant_order_sha256: str
    fixed_basis_sha256: str
    evaluated_phi_sha256: str
    genotype_scale_plan_sha256: str
    missingness_sha256: str
    annotation_map_sha256: str
    annotation_names: tuple[str, ...]
    group_map_sha256: str
    group_ids: tuple[str, ...]
    phenotype_batch_sha256: str
    residual_basis_sha256: str
    trait_ids: tuple[str, ...]
    residual_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "sample_order_sha256",
            "variant_order_allele_sha256",
            "fixed_effect_spec_sha256",
            "basis_specification_sha256",
            "basis_calibration_sha256",
            "compatible_reference_identity_sha256",
            "retained_sample_map_sha256",
            "retained_variant_order_sha256",
            "fixed_basis_sha256",
            "evaluated_phi_sha256",
            "genotype_scale_plan_sha256",
            "missingness_sha256",
            "annotation_map_sha256",
            "group_map_sha256",
            "phenotype_batch_sha256",
            "residual_basis_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        for name in ("annotation_names", "group_ids", "trait_ids", "residual_names"):
            object.__setattr__(self, name, _names(name, getattr(self, name)))

    def to_dict(self) -> dict[str, str]:
        values = {
            name: getattr(self, name)
            for name in (
                "sample_order_sha256",
                "variant_order_allele_sha256",
                "fixed_effect_spec_sha256",
                "basis_specification_sha256",
                "basis_calibration_sha256",
                "retained_sample_map_sha256",
                "retained_variant_order_sha256",
                "fixed_basis_sha256",
                "evaluated_phi_sha256",
                "genotype_scale_plan_sha256",
                "missingness_sha256",
                "annotation_map_sha256",
                "group_map_sha256",
                "phenotype_batch_sha256",
                "residual_basis_sha256",
            )
        }
        values["annotation_names_sha256"] = _name_map_digest(
            "annotation", self.annotation_names
        )
        values["group_names_sha256"] = _name_map_digest(
            "deletion_group", self.group_ids
        )
        values["trait_names_sha256"] = _name_map_digest("trait", self.trait_ids)
        values["residual_names_sha256"] = _name_map_digest(
            "residual_component", self.residual_names
        )
        return values


@dataclass(frozen=True)
class ContextualTraitMomentsV1:
    """Full-batch trait moments after zero or more approximate deletions."""

    annotation_masses: np.ndarray
    genetic_rhs: np.ndarray
    genetic_traces: np.ndarray
    genetic_residual: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "annotation_masses",
            "genetic_rhs",
            "genetic_traces",
            "genetic_residual",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
        ):
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), dtype=np.float64)
            )


@dataclass(frozen=True)
class ContextualTraitArtifactV1:
    """Complete compact trait artifact with no sample or variant-axis array."""

    manifest: Mapping[str, Any]
    component_index: ContextComponentIndex
    scale_plan: GenotypeScalePlanV1
    group_ids: tuple[str, ...]
    trait_ids: tuple[str, ...]
    residual_names: tuple[str, ...]
    genetic_rhs: np.ndarray
    genetic_traces: np.ndarray
    genetic_residual: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray
    group_rhs_unnormalized_num: np.ndarray
    group_trace_unnormalized_num: np.ndarray
    group_genetic_residual_num: np.ndarray
    annotation_masses: np.ndarray
    group_annotation_masses: np.ndarray
    group_variant_counts: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.component_index, ContextComponentIndex):
            raise ValueError("component_index must be a ContextComponentIndex.")
        if not isinstance(self.scale_plan, GenotypeScalePlanV1):
            raise ValueError("scale_plan must be a GenotypeScalePlanV1.")
        for name in ("group_ids", "trait_ids", "residual_names"):
            values = _names(f"Contextual trait V1 {name}", getattr(self, name))
            object.__setattr__(self, name, values)
        for name in _ARRAY_NAMES:
            dtype = np.int64 if name == "group_variant_counts" else np.float64
            object.__setattr__(
                self, name, _immutable_array(getattr(self, name), dtype=dtype)
            )
        manifest = _json_value(self.manifest)
        _validate_artifact(self, manifest)
        object.__setattr__(self, "manifest", freeze_context_mapping(manifest))

    @property
    def n_samples(self) -> int:
        return int(self.manifest["dimensions"]["N_study"])

    @property
    def n_variants(self) -> int:
        return int(self.manifest["dimensions"]["M"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["dimensions"]["residual_rank"])

    @property
    def n_traits(self) -> int:
        return len(self.trait_ids)

    @property
    def loo_group_ids(self) -> tuple[str, ...]:
        return self.group_ids

    @property
    def rhs_numerator_contributions(self) -> np.ndarray:
        return self.group_rhs_unnormalized_num

    @property
    def trace_numerator_contributions(self) -> np.ndarray:
        return self.group_trace_unnormalized_num

    @property
    def genetic_residual_numerator_contributions(self) -> np.ndarray:
        return self.group_genetic_residual_num

    @property
    def full_moments(self) -> ContextualTraitMomentsV1:
        return ContextualTraitMomentsV1(
            annotation_masses=self.annotation_masses,
            genetic_rhs=self.genetic_rhs,
            genetic_traces=self.genetic_traces,
            genetic_residual=self.genetic_residual,
            residual_rhs=self.residual_rhs,
            residual_traces=self.residual_traces,
            residual_gram=self.residual_gram,
        )

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.manifest)

    def verify(self) -> None:
        _validate_artifact(self, _json_value(self.manifest))

    def trait_index(self, trait: str | int) -> int:
        if isinstance(trait, bool):
            raise ValueError("trait must be a trait ID or integer index.")
        if isinstance(trait, int):
            if trait < 0 or trait >= len(self.trait_ids):
                raise IndexError("Contextual trait V1 trait index is out of range.")
            return trait
        if not isinstance(trait, str):
            raise ValueError("trait must be a trait ID or integer index.")
        try:
            return self.trait_ids.index(trait)
        except ValueError as exc:
            raise ValueError(f"Unknown contextual trait V1 trait {trait!r}.") from exc


def trait_moments_after_deleting_groups_v1(
    artifact: ContextualTraitArtifactV1,
    groups: Sequence[str],
) -> ContextualTraitMomentsV1:
    """Subtract raw grouped numerators and renormalize by retained masses."""
    if not isinstance(artifact, ContextualTraitArtifactV1):
        raise ValueError("artifact must be a ContextualTraitArtifactV1.")
    artifact.verify()
    return _trait_moments_after_deleting_groups_prevalidated_v1(
        artifact, groups
    )


def _trait_moments_after_deleting_groups_prevalidated_v1(
    artifact: ContextualTraitArtifactV1,
    groups: Sequence[str],
) -> ContextualTraitMomentsV1:
    """Subtract grouped rows after an enclosing artifact verification."""
    if isinstance(groups, (str, bytes)):
        raise ValueError("Deleted contextual trait V1 group IDs must be a sequence.")
    requested = tuple(groups)
    if any(not isinstance(value, str) or not value for value in requested):
        raise ValueError(
            "Deleted contextual trait V1 group IDs must be nonempty strings."
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Deleted contextual trait V1 group IDs must be unique.")
    if not requested:
        return artifact.full_moments
    unknown = set(requested) - set(artifact.group_ids)
    if unknown:
        raise ValueError(f"Unknown approximate-LOO groups: {sorted(unknown)}.")
    requested_set = set(requested)
    deleted = np.fromiter(
        (group in requested_set for group in artifact.group_ids),
        dtype=bool,
        count=len(artifact.group_ids),
    )
    retained_masses = artifact.annotation_masses - np.sum(
        artifact.group_annotation_masses[deleted], axis=0, dtype=np.float64
    )
    if np.any(retained_masses <= 0.0):
        raise ValueError(
            "Approximate-LOO deletion leaves a nonpositive annotation mass."
        )
    component_annotation = np.asarray(
        [entry.annotation_index for entry in artifact.component_index.entries],
        dtype=np.int64,
    )
    component_masses = retained_masses[component_annotation]
    return ContextualTraitMomentsV1(
        annotation_masses=retained_masses,
        genetic_rhs=np.sum(
            artifact.group_rhs_unnormalized_num[~deleted],
            axis=0,
            dtype=np.float64,
        )
        / component_masses[:, None],
        genetic_traces=np.sum(
            artifact.group_trace_unnormalized_num[~deleted],
            axis=0,
            dtype=np.float64,
        )
        / component_masses,
        genetic_residual=np.sum(
            artifact.group_genetic_residual_num[~deleted],
            axis=0,
            dtype=np.float64,
        )
        / component_masses[:, None],
        residual_rhs=artifact.residual_rhs,
        residual_traces=artifact.residual_traces,
        residual_gram=artifact.residual_gram,
    )


def _scale_plan_from_manifest(value: Any) -> GenotypeScalePlanV1:
    if not isinstance(value, Mapping):
        raise ValueError("Contextual trait V1 scale plan is invalid.")
    try:
        return GenotypeScalePlanV1(
            policy=GenotypeScalePolicy(value.get("policy")),
            retained_variant_order_sha256=value.get("retained_variant_order_sha256"),
            allele_orientation=value.get("allele_orientation"),
            allele_coding=value.get("allele_coding"),
            centering_source=value.get("centering_source"),
            centering_formula=value.get("centering_formula"),
            scaling_formula=value.get("scaling_formula"),
            missing_imputation=value.get("missing_imputation"),
            ploidy_policy=value.get("ploidy_policy"),
            affine_mean_sha256=value.get("affine_mean_sha256"),
            affine_inverse_scale_sha256=value.get("affine_inverse_scale_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Contextual trait V1 scale plan is invalid.") from exc


def _dimensions(value: Any) -> dict[str, int]:
    names = {
        "N_study",
        "M",
        "Q",
        "K",
        "P_g",
        "C",
        "J",
        "H",
        "L",
        "residual_rank",
    }
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError("Contextual trait V1 dimension schema mismatch.")
    result = {name: _positive_int(name, value[name], minimum=1) for name in names}
    if result["residual_rank"] > result["N_study"]:
        raise ValueError("Contextual trait V1 residual rank exceeds N.")
    return result


def _expected_shapes(dimensions: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    c = dimensions["C"]
    l = dimensions["L"]
    h = dimensions["H"]
    j = dimensions["J"]
    k = dimensions["K"]
    return {
        "genetic_rhs": (c, l),
        "genetic_traces": (c,),
        "genetic_residual": (c, h),
        "residual_rhs": (h, l),
        "residual_traces": (h,),
        "residual_gram": (h, h),
        "group_rhs_unnormalized_num": (j, c, l),
        "group_trace_unnormalized_num": (j, c),
        "group_genetic_residual_num": (j, c, h),
        "annotation_masses": (k,),
        "group_annotation_masses": (j, k),
        "group_variant_counts": (j,),
    }


def _validate_manifest_before_arrays(
    manifest: Any,
) -> tuple[
    ContextComponentIndex,
    GenotypeScalePlanV1,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - set(manifest or {}))
        extra = sorted(set(manifest or {}) - _MANIFEST_KEYS)
        raise ValueError(
            f"Contextual trait V1 manifest key mismatch; missing={missing}, "
            f"extra={extra}."
        )
    if manifest["magic"] != CONTEXTUAL_TRAIT_V1_MAGIC:
        raise ValueError("Contextual trait V1 magic mismatch.")
    try:
        schema = ContextSchemaIdentityV1(
            artifact_family=ArtifactFamily(manifest["artifact_family"]),
            logical_schema_version=LogicalSchemaVersion(
                manifest["logical_schema_version"]
            ),
            grouped_encoding_version=GroupedEncodingVersion(
                manifest["grouped_encoding_version"]
            ),
            native_api_version=manifest["native_api_version"],
            native_backend_version=manifest["native_backend_version"],
            build_id=manifest["build_id"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Contextual trait V1 schema identity is invalid.") from exc
    if schema.artifact_family is not ArtifactFamily.TRAIT:
        raise ValueError("Contextual trait V1 has the wrong artifact family.")
    if schema.native_backend_version != _TRAIT_NATIVE_BACKEND_V1:
        raise ValueError("Contextual trait V1 native backend is unsupported.")
    if manifest["feature_mode"] != RAW_PROJECTED_FEATURE_MODE:
        raise ValueError("Contextual trait V1 feature mode mismatch.")
    if manifest["scientific_policy"] != _SCIENTIFIC_POLICY:
        raise ValueError("Contextual trait V1 scientific policy mismatch.")
    if manifest["deletion"] != _DELETION_POLICY:
        raise ValueError("Contextual trait V1 deletion policy mismatch.")
    if manifest["layouts"] != _LAYOUTS:
        raise ValueError("Contextual trait V1 logical array layouts mismatch.")
    if manifest["terminal_status"] != "published":
        raise ValueError("Contextual trait V1 is not published.")
    _sha256(
        "compatible_reference_identity_sha256",
        manifest["compatible_reference_identity_sha256"],
    )

    dimensions = _dimensions(manifest["dimensions"])
    maps = manifest["maps"]
    map_keys = {
        "pair_map",
        "pair_map_sha256",
        "component_map",
        "component_map_sha256",
        "annotation_names",
        "annotation_map_sha256",
        "annotation_names_sha256",
        "group_ids",
        "group_map_sha256",
        "group_names_sha256",
        "trait_ids",
        "trait_map_sha256",
        "residual_names",
        "residual_map_sha256",
    }
    if not isinstance(maps, Mapping) or set(maps) != map_keys:
        raise ValueError("Contextual trait V1 map schema mismatch.")
    annotation_names = _names("annotation_names", maps["annotation_names"])
    group_ids = _names("group_ids", maps["group_ids"])
    trait_ids = _names("trait_ids", maps["trait_ids"])
    residual_names = _names("residual_names", maps["residual_names"])
    if (
        len(annotation_names) != dimensions["K"]
        or len(group_ids) != dimensions["J"]
        or len(trait_ids) != dimensions["L"]
        or len(residual_names) != dimensions["H"]
    ):
        raise ValueError("Contextual trait V1 named-axis dimensions mismatch.")
    components = ContextComponentIndex(
        annotation_names, ContextPairIndex(dimensions["Q"])
    )
    if (
        len(components.pair_index) != dimensions["P_g"]
        or len(components) != dimensions["C"]
    ):
        raise ValueError("Contextual trait V1 pair/component dimensions mismatch.")
    expected_maps = {
        "pair_map": components.pair_index.to_dict(),
        "pair_map_sha256": components.pair_index.digest,
        "component_map": components.to_dict(),
        "component_map_sha256": components.digest,
        "annotation_names": list(annotation_names),
        "annotation_names_sha256": _name_map_digest("annotation", annotation_names),
        "group_ids": list(group_ids),
        "group_names_sha256": _name_map_digest("deletion_group", group_ids),
        "trait_ids": list(trait_ids),
        "trait_map_sha256": _name_map_digest("trait", trait_ids),
        "residual_names": list(residual_names),
        "residual_map_sha256": _name_map_digest("residual_component", residual_names),
    }
    for name, expected in expected_maps.items():
        if maps[name] != expected:
            raise ValueError(f"Contextual trait V1 {name} mismatch.")
    for name in ("annotation_map_sha256", "group_map_sha256"):
        _sha256(name, maps[name])

    scale_plan = _scale_plan_from_manifest(manifest["genotype_scale_plan"])
    if manifest["genotype_scale_plan_sha256"] != scale_plan.digest:
        raise ValueError("Contextual trait V1 scale-plan digest mismatch.")
    _validate_descriptor_scale_plan(scale_plan)

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != _TRAIT_IDENTITY_KEYS:
        raise ValueError("Contextual trait V1 identity schema mismatch.")
    for name in _TRAIT_IDENTITY_KEYS:
        _sha256(name, identity[name])
    authority = {
        "genotype_scale_plan_sha256": scale_plan.digest,
        "annotation_map_sha256": maps["annotation_map_sha256"],
        "annotation_names_sha256": maps["annotation_names_sha256"],
        "group_map_sha256": maps["group_map_sha256"],
        "group_names_sha256": maps["group_names_sha256"],
        "trait_names_sha256": maps["trait_map_sha256"],
        "residual_names_sha256": maps["residual_map_sha256"],
    }
    for name, expected in authority.items():
        if identity[name] != expected:
            raise ValueError(f"Contextual trait V1 authority {name} mismatch.")

    arrays = manifest["arrays"]
    if not isinstance(arrays, Mapping) or set(arrays) != set(_ARRAY_NAMES):
        raise ValueError("Contextual trait V1 array schema mismatch.")
    for name, shape in _expected_shapes(dimensions).items():
        envelope = arrays[name]
        expected_dtype = "<i8" if name == "group_variant_counts" else "<f8"
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(f"Contextual trait V1 {name} envelope is invalid.")
        if envelope["dtype"] != expected_dtype or envelope["shape"] != list(shape):
            raise ValueError(f"Contextual trait V1 {name} schema is invalid.")
        _sha256(f"arrays.{name}.sha256", envelope["sha256"])
    return components, scale_plan, group_ids, trait_ids, residual_names


def _validate_artifact(
    artifact: ContextualTraitArtifactV1, manifest: Mapping[str, Any]
) -> None:
    (
        components,
        scale_plan,
        groups,
        traits,
        residuals,
    ) = _validate_manifest_before_arrays(manifest)
    if components.digest != artifact.component_index.digest:
        raise ValueError("Contextual trait V1 component map differs from its manifest.")
    if scale_plan.digest != artifact.scale_plan.digest:
        raise ValueError("Contextual trait V1 scale plan differs from its manifest.")
    if groups != artifact.group_ids or traits != artifact.trait_ids:
        raise ValueError("Contextual trait V1 named maps differ from the artifact.")
    if residuals != artifact.residual_names:
        raise ValueError("Contextual trait V1 residual map differs from the artifact.")
    dimensions = _dimensions(manifest["dimensions"])
    for name, shape in _expected_shapes(dimensions).items():
        value = np.asarray(getattr(artifact, name))
        expected_dtype = np.dtype(
            np.int64 if name == "group_variant_counts" else np.float64
        ).newbyteorder("<")
        if value.shape != shape or value.dtype != expected_dtype:
            raise ValueError(
                f"Contextual trait V1 array {name!r} must have shape {shape} "
                f"and dtype {expected_dtype}."
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"Contextual trait V1 array {name!r} is nonfinite.")
        if manifest["arrays"][name] != _array_envelope(value):
            raise ValueError(f"Contextual trait V1 array digest mismatch for {name}.")

    if np.any(artifact.annotation_masses <= 0.0):
        raise ValueError("Contextual trait V1 annotation masses must be positive.")
    if np.any(artifact.group_annotation_masses < 0.0):
        raise ValueError("Contextual trait V1 group masses must be nonnegative.")
    if np.any(artifact.group_variant_counts <= 0):
        raise ValueError("Contextual trait V1 group counts must be positive.")
    if sum(int(value) for value in artifact.group_variant_counts) != dimensions["M"]:
        raise ValueError("Contextual trait V1 group counts do not reconstruct M.")
    if (
        _relative_max(
            np.sum(artifact.group_annotation_masses, axis=0, dtype=np.float64),
            artifact.annotation_masses,
        )
        > 1.0e-12
    ):
        raise ValueError("Contextual trait V1 group masses do not reconstruct totals.")
    if np.any(
        artifact.annotation_masses[None, :] - artifact.group_annotation_masses <= 0.0
    ):
        raise ValueError(
            "Contextual trait V1 contains a single-group deletion that empties an annotation."
        )
    if _relative_max(artifact.residual_gram, artifact.residual_gram.T) > 1.0e-12:
        raise ValueError("Contextual trait V1 residual Gram is not symmetric.")

    component_annotation = np.asarray(
        [entry.annotation_index for entry in artifact.component_index.entries],
        dtype=np.int64,
    )
    masses = artifact.annotation_masses[component_annotation]
    reconstructions = {
        "genetic_rhs": np.sum(
            artifact.group_rhs_unnormalized_num, axis=0, dtype=np.float64
        )
        / masses[:, None],
        "genetic_traces": np.sum(
            artifact.group_trace_unnormalized_num, axis=0, dtype=np.float64
        )
        / masses,
        "genetic_residual": np.sum(
            artifact.group_genetic_residual_num, axis=0, dtype=np.float64
        )
        / masses[:, None],
    }
    for name, reconstructed in reconstructions.items():
        if _relative_max(reconstructed, np.asarray(getattr(artifact, name))) > 1.0e-10:
            raise ValueError(
                f"Contextual trait V1 grouped numerators do not reconstruct {name}."
            )

    identity = manifest["identity"]
    if not isinstance(identity, Mapping) or set(identity) != _TRAIT_IDENTITY_KEYS:
        raise ValueError("Contextual trait V1 identity schema mismatch.")
    for name in _TRAIT_IDENTITY_KEYS:
        _sha256(name, identity[name])
    if (
        identity["retained_variant_order_sha256"]
        != artifact.scale_plan.retained_variant_order_sha256
    ):
        raise ValueError("Contextual trait V1 retained variant identity mismatch.")

    phenotype = manifest["phenotype"]
    if not isinstance(phenotype, Mapping) or set(phenotype) != {
        "normalization_policy",
        "projected_once",
        "score_scaling",
    }:
        raise ValueError("Contextual trait V1 phenotype schema mismatch.")
    if phenotype != {
        "normalization_policy": "project_then_unit_residual_variance_v1",
        "projected_once": True,
        "score_scaling": "raw_scores_no_residual_rank_division_v1",
    }:
        raise ValueError("Contextual trait V1 phenotype policy mismatch.")

    execution = manifest["execution"]
    execution_keys = {
        "annotation_mode",
        "annotation_output_contract",
        "numeric_policy",
        "phenotype_normalization",
        "file_identity_policy",
        "file_content_identity",
        "native_execution_backend",
        "execution_plan_sha256",
        "native_scientific_array_sha256",
        "phase_evidence_sha256",
        "build_provenance",
        "python_adapter_provenance",
        "output_ownership",
        "numa",
        "admission",
        "diagnostics",
        "telemetry",
    }
    if not isinstance(execution, Mapping) or set(execution) != execution_keys:
        raise ValueError("Contextual trait V1 execution schema mismatch.")
    mode = AnnotationMode(execution["annotation_mode"])
    if execution["annotation_output_contract"] != dict(
        annotation_output_contract(mode)
    ):
        raise ValueError("Contextual trait V1 annotation interpretation mismatch.")
    if execution["numeric_policy"] != "fp64_v1":
        raise ValueError("Contextual trait V1 numeric policy mismatch.")
    validate_python_source_runtime_provenance(
        execution["python_adapter_provenance"],
        ordered_modules=_PYTHON_ADAPTER_MODULES,
        policy=_PYTHON_ADAPTER_PROVENANCE_POLICY,
        include_scipy=False,
        family="Contextual trait V1",
    )
    if execution["phenotype_normalization"] != (
        "project_then_unit_residual_variance_v1"
    ):
        raise ValueError("Contextual trait V1 normalization policy mismatch.")
    if execution["file_identity_policy"] != _FILE_IDENTITY_POLICY_V1:
        raise ValueError("Contextual trait V1 file identity policy mismatch.")
    _validate_file_content_identity(
        execution["file_content_identity"], n_variants=dimensions["M"]
    )
    if execution["native_execution_backend"] != (
        "deterministic_tiled_fp64_with_scalar_witness_v1"
    ):
        raise ValueError("Contextual trait V1 native execution backend mismatch.")
    if execution["output_ownership"] != {"owns_data": True, "read_only": True}:
        raise ValueError("Contextual trait V1 output ownership is invalid.")
    _sha256("execution.execution_plan_sha256", execution["execution_plan_sha256"])
    _validate_build_provenance(
        execution["build_provenance"],
        build_id=manifest["build_id"],
        source_tree_sha256=identity["source_tree_sha256"],
    )
    scientific_digests = _validate_scientific_array_digests(
        execution["native_scientific_array_sha256"],
        _TRAIT_SCIENTIFIC_ARRAY_KEYS,
    )
    _validate_phase_evidence(execution["phase_evidence_sha256"])
    pair_entries = artifact.component_index.pair_index.entries
    component_entries = artifact.component_index.entries
    expected_scientific_values = {
        **{name: np.asarray(getattr(artifact, name)) for name in _ARRAY_NAMES},
        "pair_q": np.asarray([entry.q for entry in pair_entries], dtype=np.int64),
        "pair_r": np.asarray([entry.r for entry in pair_entries], dtype=np.int64),
        "pair_eta": np.asarray(
            [entry.kernel_factor for entry in pair_entries], dtype=np.int64
        ),
        "component_annotation": np.asarray(
            [entry.annotation_index for entry in component_entries], dtype=np.int64
        ),
        "component_pair": np.asarray(
            [entry.pair_index for entry in component_entries], dtype=np.int64
        ),
    }
    for name, value in expected_scientific_values.items():
        if scientific_digests[name] != array_sha256(value):
            raise ValueError(f"Contextual trait V1 native digest mismatch for {name}.")
    _validate_numa(execution["numa"])
    _validate_execution_evidence(
        execution["admission"],
        execution["diagnostics"],
        execution["telemetry"],
        n=dimensions["N_study"],
        m=dimensions["M"],
        q=dimensions["Q"],
        k=dimensions["K"],
        c=dimensions["C"],
        l=dimensions["L"],
    )
    canonical_sha256(manifest)


def adapt_native_contextual_trait_v1(
    native_result: Mapping[str, Any],
    publication_identity: ContextualTraitPublicationIdentityV1,
) -> ContextualTraitArtifactV1:
    """Validate and publish one complete native compact trait result."""
    if not isinstance(native_result, Mapping):
        raise ValueError("Native contextual trait result must be a mapping.")
    if set(native_result) != _NATIVE_RESULT_KEYS:
        missing = sorted(_NATIVE_RESULT_KEYS - set(native_result))
        extra = sorted(set(native_result) - _NATIVE_RESULT_KEYS)
        raise ValueError(
            "Native contextual trait result schema mismatch; "
            f"missing={missing}, extra={extra}."
        )
    if not isinstance(publication_identity, ContextualTraitPublicationIdentityV1):
        raise ValueError("publication_identity must use the trait V1 identity type.")
    if _required_native(native_result, "complete_trait_artifact") is not False:
        raise ValueError("Native result must not claim Python artifact publication.")
    if _required_native(native_result, "complete_trait_statistics") is not True:
        raise ValueError("Native result does not contain complete trait statistics.")
    if _required_native(native_result, "internal_result_kind") != (
        "stage4_complete_trait_statistics_v1"
    ):
        raise ValueError("Native contextual trait result kind mismatch.")
    if (
        _required_native(native_result, "state") != "trait_statistics_complete_ready"
        or _required_native(native_result, "lifecycle")
        != "trait_statistics_complete_ready"
    ):
        raise ValueError("Native contextual trait result is not complete.")

    exact_policies = {
        "numeric_policy": "fp64_v1",
        "deletion_semantics": DeletionSemantics.APPROXIMATE_SUMMARY_ONLY_V1.value,
        "grouped_encoding": GroupedEncodingVersion.DENSE_UNNORMALIZED_V1.value,
        "group_numerator_unit": "raw_source_annotation_mass_v1",
        "residual_only_deletion": "reuse_full_unchanged",
        "phenotype_normalization": "project_then_unit_residual_variance_v1",
        "feature_mode": "P_diag_phi_G_v1",
        "file_identity_policy": _FILE_IDENTITY_POLICY_V1,
        "contextual_backend": "plink_bed_descriptor_stream_trait_v1",
        "contextual_execution_backend": (
            "deterministic_tiled_fp64_with_scalar_witness_v1"
        ),
    }
    for name, expected in exact_policies.items():
        if _required_native(native_result, name) != expected:
            raise ValueError(f"Native contextual trait {name} policy mismatch.")
    for name in (
        "grouped_values_are_unnormalized_numerators",
        "phenotypes_projected_once",
    ):
        if _required_native(native_result, name) is not True:
            raise ValueError(f"Native contextual trait {name} evidence is false.")
    ownership = _required_native(native_result, "output_ownership")
    if not isinstance(ownership, Mapping) or dict(ownership) != {
        "owns_data": True,
        "read_only": True,
    }:
        raise ValueError("Native contextual trait output ownership is invalid.")
    numa = _validate_numa(_required_native(native_result, "numa"))

    q = _positive_int("q", _required_native(native_result, "q"))
    annotation_names = _names(
        "annotation_names", _required_native(native_result, "annotation_names")
    )
    group_ids = _names("group_names", _required_native(native_result, "group_names"))
    trait_ids = _names("trait_names", _required_native(native_result, "trait_names"))
    residual_names = _names(
        "residual_names", _required_native(native_result, "residual_names")
    )
    authority_names = {
        "annotation_names": publication_identity.annotation_names,
        "group_names": publication_identity.group_ids,
        "trait_names": publication_identity.trait_ids,
        "residual_names": publication_identity.residual_names,
    }
    observed_names = {
        "annotation_names": annotation_names,
        "group_names": group_ids,
        "trait_names": trait_ids,
        "residual_names": residual_names,
    }
    for name, expected in authority_names.items():
        if observed_names[name] != expected:
            raise ValueError(f"Native and publication {name} differ.")
    if _positive_int("trait_count", native_result["trait_count"]) != len(trait_ids):
        raise ValueError("Native contextual trait trait count mismatch.")
    if _positive_int(
        "residual_component_count", native_result["residual_component_count"]
    ) != len(residual_names):
        raise ValueError("Native contextual trait residual-component count mismatch.")
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q))
    expected_maps = {
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
    }
    for name, expected in expected_maps.items():
        observed = _native_array(native_result, name, dtype=np.int64, ndim=1)
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise ValueError(f"Native contextual trait {name} is not canonical.")

    native_variant_alleles = _sha256(
        "variant_order_allele_sha256",
        _required_native(native_result, "variant_order_allele_sha256"),
    )
    if native_variant_alleles != publication_identity.variant_order_allele_sha256:
        raise ValueError("Native and publication variant/allele identities differ.")
    retained_variant_order_sha256 = _sha256(
        "retained_variant_order_sha256",
        _required_native(native_result, "retained_variant_order_sha256"),
    )
    try:
        scale_plan = GenotypeScalePlanV1(
            policy=GenotypeScalePolicy(
                _required_native(native_result, "genotype_scale_policy")
            ),
            retained_variant_order_sha256=retained_variant_order_sha256,
            allele_orientation=_required_native(native_result, "allele_orientation"),
            allele_coding=_required_native(native_result, "allele_coding"),
            centering_source=_required_native(native_result, "centering_source"),
            centering_formula=_required_native(native_result, "centering_formula"),
            scaling_formula=_required_native(native_result, "scaling_formula"),
            missing_imputation=_required_native(native_result, "missing_imputation"),
            ploidy_policy=_required_native(native_result, "ploidy_policy"),
            affine_mean_sha256=_required_native(native_result, "affine_mean_sha256"),
            affine_inverse_scale_sha256=_required_native(
                native_result, "affine_inverse_scale_sha256"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Native contextual trait scale plan is invalid.") from exc
    if _required_native(native_result, "scale_plan_sha256") != scale_plan.digest:
        raise ValueError("Native contextual trait scale-plan digest mismatch.")
    _validate_descriptor_scale_plan(scale_plan)
    if scale_plan.digest != publication_identity.genotype_scale_plan_sha256:
        raise ValueError("Native and publication genotype scale plans differ.")
    if (
        retained_variant_order_sha256
        != publication_identity.retained_variant_order_sha256
    ):
        raise ValueError("Native and publication retained variant maps differ.")
    authority_fields = {
        "retained_sample_map_sha256": publication_identity.retained_sample_map_sha256,
        "fixed_basis_sha256": publication_identity.fixed_basis_sha256,
        "evaluated_phi_sha256": publication_identity.evaluated_phi_sha256,
        "missingness_sha256": publication_identity.missingness_sha256,
        "annotation_map_sha256": publication_identity.annotation_map_sha256,
        "group_map_sha256": publication_identity.group_map_sha256,
        "phenotype_batch_sha256": publication_identity.phenotype_batch_sha256,
        "residual_basis_sha256": publication_identity.residual_basis_sha256,
    }
    for name, expected in authority_fields.items():
        observed = _sha256(name, _required_native(native_result, name))
        if observed != expected:
            raise ValueError(f"Native and publication {name} identities differ.")
    backend_version = _positive_int(
        "contextual_backend_version",
        _required_native(native_result, "contextual_backend_version"),
    )
    if backend_version != 2:
        raise ValueError("Native contextual trait backend version is unsupported.")
    build_id = _required_native(native_result, "contextual_build_id")
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("Native contextual trait build identity is invalid.")
    source_tree_sha256 = _sha256(
        "source_tree_sha256", _required_native(native_result, "source_tree_sha256")
    )
    execution_plan_sha256 = _sha256(
        "execution_plan_sha256",
        _required_native(native_result, "execution_plan_sha256"),
    )
    build_provenance = _validate_build_provenance(
        _required_native(native_result, "build_provenance"),
        build_id=build_id,
        source_tree_sha256=source_tree_sha256,
    )
    scientific_array_sha256 = _validate_scientific_array_digests(
        _required_native(native_result, "scientific_array_sha256"),
        _TRAIT_SCIENTIFIC_ARRAY_KEYS,
    )
    phase_evidence_sha256 = _validate_phase_evidence(
        _required_native(native_result, "phase_evidence_sha256")
    )

    native_arrays = {
        "genetic_rhs": _native_array(
            native_result, "genetic_rhs", dtype=np.float64, ndim=2
        ),
        "genetic_traces": _native_array(
            native_result, "genetic_traces", dtype=np.float64, ndim=1
        ),
        "genetic_residual": _native_array(
            native_result, "genetic_residual", dtype=np.float64, ndim=2
        ),
        "residual_rhs": _native_array(
            native_result, "residual_rhs", dtype=np.float64, ndim=2
        ),
        "residual_traces": _native_array(
            native_result, "residual_traces", dtype=np.float64, ndim=1
        ),
        "residual_gram": _native_array(
            native_result, "residual_gram", dtype=np.float64, ndim=2
        ),
        "group_rhs_unnormalized_num": _native_array(
            native_result,
            "group_rhs_unnormalized_num",
            dtype=np.float64,
            ndim=3,
        ),
        "group_trace_unnormalized_num": _native_array(
            native_result,
            "group_trace_unnormalized_num",
            dtype=np.float64,
            ndim=2,
        ),
        "group_genetic_residual_num": _native_array(
            native_result,
            "group_genetic_residual_num",
            dtype=np.float64,
            ndim=3,
        ),
        "annotation_masses": _native_array(
            native_result, "annotation_masses", dtype=np.float64, ndim=1
        ),
        "group_annotation_masses": _native_array(
            native_result, "group_annotation_masses", dtype=np.float64, ndim=2
        ),
        "group_variant_counts": _native_array(
            native_result, "group_variant_counts", dtype=np.int64, ndim=1
        ),
    }
    arrays = {
        name: _immutable_array(value, dtype=value.dtype)
        for name, value in native_arrays.items()
    }
    scientific_values = {**native_arrays, **expected_maps}
    for name, value in scientific_values.items():
        if scientific_array_sha256[name] != array_sha256(value):
            raise ValueError(
                f"Native contextual trait scientific digest mismatch for {name}."
            )
    n_samples = _positive_int(
        "study_n", _required_native(native_result, "study_n"), minimum=2
    )
    n_variants = _positive_int(
        "n_variants", _required_native(native_result, "n_variants")
    )
    file_content_identity = _validate_file_content_identity(
        _required_native(native_result, "file_content_identity"),
        n_variants=n_variants,
    )
    residual_rank = _positive_int(
        "residual_rank", _required_native(native_result, "residual_rank")
    )
    if residual_rank > n_samples:
        raise ValueError("Native contextual trait residual rank exceeds N.")
    dimensions = {
        "N_study": n_samples,
        "M": n_variants,
        "Q": q,
        "K": len(annotation_names),
        "P_g": len(components.pair_index),
        "C": len(components),
        "J": len(group_ids),
        "H": len(residual_names),
        "L": len(trait_ids),
        "residual_rank": residual_rank,
    }
    for name, shape in _expected_shapes(dimensions).items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"Native contextual trait {name} has shape {arrays[name].shape}, "
                f"expected {shape}."
            )
    admission, diagnostics, telemetry = _validate_execution_evidence(
        _required_native(native_result, "admission"),
        _required_native(native_result, "diagnostics"),
        _required_native(native_result, "telemetry"),
        n=n_samples,
        m=n_variants,
        q=q,
        k=len(annotation_names),
        c=len(components),
        l=len(trait_ids),
    )

    identity = publication_identity.to_dict()
    identity.update(
        {
            "sealed_plan_sha256": _sha256(
                "sealed_plan_sha256",
                _required_native(native_result, "sealed_plan_sha256"),
            ),
            "source_tree_sha256": source_tree_sha256,
        }
    )
    maps = {
        "pair_map": components.pair_index.to_dict(),
        "pair_map_sha256": components.pair_index.digest,
        "component_map": components.to_dict(),
        "component_map_sha256": components.digest,
        "annotation_names": list(annotation_names),
        "annotation_map_sha256": _sha256(
            "annotation_map_sha256",
            _required_native(native_result, "annotation_map_sha256"),
        ),
        "annotation_names_sha256": _name_map_digest("annotation", annotation_names),
        "group_ids": list(group_ids),
        "group_map_sha256": _sha256(
            "group_map_sha256", _required_native(native_result, "group_map_sha256")
        ),
        "group_names_sha256": _name_map_digest("deletion_group", group_ids),
        "trait_ids": list(trait_ids),
        "trait_map_sha256": _name_map_digest("trait", trait_ids),
        "residual_names": list(residual_names),
        "residual_map_sha256": _name_map_digest("residual_component", residual_names),
    }
    schema = ContextSchemaIdentityV1(
        artifact_family=ArtifactFamily.TRAIT,
        logical_schema_version=LogicalSchemaVersion.TRAIT_V1,
        grouped_encoding_version=GroupedEncodingVersion.DENSE_UNNORMALIZED_V1,
        native_api_version=_positive_int(
            "contextual_native_api_version",
            _required_native(native_result, "contextual_native_api_version"),
        ),
        native_backend_version=(
            f"{_required_native(native_result, 'contextual_backend')}:"
            f"{backend_version}"
        ),
        build_id=build_id,
    )
    manifest: dict[str, Any] = {
        "magic": CONTEXTUAL_TRAIT_V1_MAGIC,
        **schema.to_dict(),
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "scientific_policy": dict(_SCIENTIFIC_POLICY),
        "dimensions": dimensions,
        "identity": identity,
        "compatible_reference_identity_sha256": (
            publication_identity.compatible_reference_identity_sha256
        ),
        "genotype_scale_plan": scale_plan.to_dict(),
        "genotype_scale_plan_sha256": scale_plan.digest,
        "phenotype": {
            "normalization_policy": "project_then_unit_residual_variance_v1",
            "projected_once": True,
            "score_scaling": "raw_scores_no_residual_rank_division_v1",
        },
        "execution": {
            "annotation_mode": str(_required_native(native_result, "annotation_mode")),
            "annotation_output_contract": dict(
                annotation_output_contract(
                    AnnotationMode(_required_native(native_result, "annotation_mode"))
                )
            ),
            "numeric_policy": "fp64_v1",
            "phenotype_normalization": ("project_then_unit_residual_variance_v1"),
            "file_identity_policy": str(
                _required_native(native_result, "file_identity_policy")
            ),
            "file_content_identity": file_content_identity,
            "native_execution_backend": str(
                _required_native(native_result, "contextual_execution_backend")
            ),
            "execution_plan_sha256": execution_plan_sha256,
            "native_scientific_array_sha256": scientific_array_sha256,
            "phase_evidence_sha256": phase_evidence_sha256,
            "build_provenance": build_provenance,
            "python_adapter_provenance": python_source_runtime_provenance(
                _PYTHON_ADAPTER_MODULES,
                policy=_PYTHON_ADAPTER_PROVENANCE_POLICY,
                include_scipy=False,
            ),
            "output_ownership": _json_value(ownership),
            "numa": numa,
            "admission": admission,
            "diagnostics": diagnostics,
            "telemetry": telemetry,
        },
        "deletion": dict(_DELETION_POLICY),
        "layouts": {name: list(layout) for name, layout in _LAYOUTS.items()},
        "maps": maps,
        "arrays": {name: _array_envelope(value) for name, value in arrays.items()},
        "terminal_status": "published",
    }
    return ContextualTraitArtifactV1(
        manifest=manifest,
        component_index=components,
        scale_plan=scale_plan,
        group_ids=group_ids,
        trait_ids=trait_ids,
        residual_names=residual_names,
        **arrays,
    )


def run_contextual_trait_v1(
    executor: Any,
    publication_identity: ContextualTraitPublicationIdentityV1,
) -> ContextualTraitArtifactV1:
    """Make exactly one native ``run`` call and publish its compact result."""
    run = getattr(executor, "run", None)
    if run is None or not callable(run):
        raise ValueError("executor must expose a callable run() method.")
    return adapt_native_contextual_trait_v1(run(), publication_identity)


def write_contextual_trait_v1(
    artifact: ContextualTraitArtifactV1, output: str | Path
) -> Path:
    """Atomically publish a single strict contextual-trait V1 container."""
    if not isinstance(artifact, ContextualTraitArtifactV1):
        raise ValueError("Only ContextualTraitArtifactV1 can use the V1 writer.")
    artifact.verify()
    manifest_json, manifest_sha256, arrays = _preflight_stable_npz_members(
        manifest_json=canonical_json(artifact.manifest),
        manifest_sha256=artifact.manifest_sha256,
        arrays={name: getattr(artifact, name) for name in _ARRAY_NAMES},
        family="Contextual trait V1",
    )
    path = Path(output)
    if path.suffix != ".npz" or not path.name.endswith(CONTEXTUAL_TRAIT_V1_SUFFIX):
        path = Path(str(path) + CONTEXTUAL_TRAIT_V1_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                manifest_json=manifest_json,
                manifest_sha256=manifest_sha256,
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        _validate_stable_npz_writer_temp(
            temporary_name,
            family="Contextual trait V1",
            manifest_json=manifest_json,
            manifest_sha256=manifest_sha256,
            arrays=arrays,
        )
        _publish_stable_npz_no_replace(temporary_name, path)
        fsync_parent_directory(path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_contextual_trait_v1(path: str | Path) -> ContextualTraitArtifactV1:
    """Load only the stable trait V1 family and fail before returning arrays."""
    source = Path(path)
    if not source.name.endswith(CONTEXTUAL_TRAIT_V1_SUFFIX):
        raise ValueError("Contextual trait V1 loader requires the V1 suffix.")
    try:
        with StableNpzReader(
            source,
            family="Contextual trait V1",
            maximum_members=len(_FILE_KEYS),
        ) as archive:
            manifest = _strict_json_loads(
                archive.read_text_scalar(
                    "manifest_json", maximum_bytes=16 * 1024 * 1024
                )
            )
            digest = archive.read_text_scalar("manifest_sha256", maximum_bytes=1024)
            if _sha256("manifest_sha256", digest) != canonical_sha256(manifest):
                raise ValueError("Contextual trait V1 manifest SHA-256 mismatch.")
            (
                components,
                scale_plan,
                group_ids,
                trait_ids,
                residual_names,
            ) = _validate_manifest_before_arrays(manifest)
            shapes = _expected_shapes(_dimensions(manifest["dimensions"]))
            array_specs = {}
            for name in _ARRAY_NAMES:
                expected_dtype = np.dtype(
                    np.int64 if name == "group_variant_counts" else np.float64
                ).newbyteorder("<")
                array_specs[name] = (expected_dtype, shapes[name])
            archive.preflight_arrays(array_specs)
            arrays = {name: archive.load_array(name) for name in _ARRAY_NAMES}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Contextual trait V1"):
            raise
        raise ValueError("Not a valid contextual trait V1 artifact.") from exc
    return ContextualTraitArtifactV1(
        manifest=manifest,
        component_index=components,
        scale_plan=scale_plan,
        group_ids=group_ids,
        trait_ids=trait_ids,
        residual_names=residual_names,
        **arrays,
    )


__all__ = [
    "CONTEXTUAL_TRAIT_V1_MAGIC",
    "CONTEXTUAL_TRAIT_V1_SUFFIX",
    "ContextualTraitArtifactV1",
    "ContextualTraitMomentsV1",
    "ContextualTraitPublicationIdentityV1",
    "adapt_native_contextual_trait_v1",
    "load_contextual_trait_v1",
    "run_contextual_trait_v1",
    "trait_moments_after_deleting_groups_v1",
    "write_contextual_trait_v1",
]
