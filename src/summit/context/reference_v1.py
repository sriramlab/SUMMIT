"""Stable V1 publication boundary for native contextual reference statistics.

Estimator identity: this module implements the sample-axis-probe aggregate
contextual covariance/action estimator. It is not the generalized per-variant
GxE LD-score estimator, whose contract is documented in
``docs/generalized_gxe_variant_ldscore_contract.md``.

This module is intentionally separate from :mod:`summit.context.reference`.
The latter contains the private NumPy development formats; neither loader
accepts artifacts from the other family.
"""

from __future__ import annotations

from collections import Counter
import json
import hashlib
import os
import struct
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
from .oracle import transfer_reference_gram
from .reference import GroupedContextReference, ReferenceMoments
from .schema import (
    AnnotationMode,
    ArtifactFamily,
    ContextSchemaIdentityV1,
    DeletionSemantics,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
    GroupedAttributionAlgorithm,
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
    owned_readonly_array,
)


CONTEXTUAL_REFERENCE_V1_MAGIC = "SUMMIT_CONTEXTUAL_REFERENCE_V1"
CONTEXTUAL_REFERENCE_V1_SUFFIX = ".contextual-reference-v1.npz"
_REFERENCE_NATIVE_BACKEND_V1 = "plink_bed_descriptor_stream_stage2_v1:2"
_FILE_IDENTITY_POLICY_V1 = (
    "sealed_fstat_full_bim_fam_sha256_retained_bed_record_sha256_v2"
)
_PYTHON_ADAPTER_PROVENANCE_POLICY = (
    "ordered_relative_module_name_and_content_sha256_sha256_v1"
)
_PYTHON_ADAPTER_MODULES = (
    "_artifact_io.py",
    "annotations.py",
    "oracle.py",
    "reference.py",
    "reference_v1.py",
    "schema.py",
    "spec.py",
)

_LAYOUTS = {
    "gram": ["component", "component"],
    "same_person": ["component", "component"],
    "group_gram_unnormalized_num": [
        "deletion_group",
        "component",
        "component",
    ],
    "annotation_masses": ["annotation"],
    "group_annotation_masses": ["deletion_group", "annotation"],
    "group_variant_counts": ["deletion_group"],
}

_ARRAY_NAMES = (
    "gram",
    "same_person",
    "group_gram_unnormalized_num",
    "annotation_masses",
    "group_annotation_masses",
    "group_variant_counts",
)
_FILE_KEYS = frozenset((*_ARRAY_NAMES, "manifest_json", "manifest_sha256"))
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
        "genotype_scale_plan",
        "genotype_scale_plan_sha256",
        "probes",
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
        "complete_reference_artifact",
        "file_identity_policy",
        "file_content_identity",
        "internal_result_kind",
        "complete_reference_statistics",
        "gram",
        "raw_gram_numerator",
        "annotation_masses",
        "same_person",
        "group_gram_unnormalized_num",
        "group_annotation_masses",
        "group_variant_counts",
        "reference_n",
        "n_variants",
        "residual_rank",
        "q",
        "numeric_policy",
        "deletion_semantics",
        "same_person_deletion",
        "grouped_encoding",
        "group_numerator_unit",
        "group_gram_raw_unit",
        "grouped_values_are_unnormalized_numerators",
        "same_person_signed_preserved",
        "selected_grouped_algorithm",
        "grouped_attribution_algorithm",
        "direct_grouped_scaling",
        "grouped_differential_enabled",
        "group_execution_order",
        "group_execution_order_sha256",
        "group_execution_permutation_sha256",
        "sealed_plan_sha256",
        "source_tree_sha256",
        "execution_plan_sha256",
        "build_provenance",
        "scientific_array_sha256",
        "phase_evidence_sha256",
        "sample_probe_policy",
        "sample_probe_identity_sha256",
        "sample_probe_count",
        "variant_probe_policy",
        "variant_probe_identity_sha256",
        "variant_probe_count",
        "retained_sample_map_sha256",
        "variant_order_allele_sha256",
        "fixed_basis_sha256",
        "evaluated_phi_sha256",
        "annotation_map_sha256",
        "group_map_sha256",
        "same_person_sha256",
        "group_gram_unnormalized_num_sha256",
        "pair_q",
        "pair_r",
        "pair_eta",
        "component_annotation",
        "component_pair",
        "annotation_names",
        "group_names",
        "annotation_mode",
        "probe_policy",
        "probe_identity_sha256",
        "genotype_scale_policy",
        "allele_orientation",
        "allele_coding",
        "centering_formula",
        "scaling_formula",
        "missing_imputation",
        "ploidy_policy",
        "centering_source",
        "retained_variant_order_sha256",
        "affine_mean_sha256",
        "affine_inverse_scale_sha256",
        "missingness_sha256",
        "scale_plan_sha256",
        "contextual_native_api_version",
        "contextual_backend_version",
        "contextual_backend",
        "contextual_execution_backend",
        "contextual_build_id",
        "output_ownership",
        "numa",
        "diagnostics",
        "telemetry",
        "admission",
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
        "source_phase_bytes",
        "action_phase_bytes",
        "group_phase_bytes",
        "same_person_phase_bytes",
        "compact_output_bytes",
        "integrity_reserve_bytes",
        "telemetry_bytes",
        "required_workspace_bytes",
        "required_telemetry_capacity",
        "maximum_protected_output_elements",
        "resident_batches",
        "source_descriptor_passes",
        "action_descriptor_passes",
        "source_decoded_blocks",
        "action_decoded_blocks",
        "group_descriptor_passes",
        "group_decoded_blocks",
        "same_person_descriptor_passes",
        "same_person_decoded_blocks",
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
        "target_dense_columns",
        "target_useful_columns",
        "strict_zero_weight_columns_eliminated",
        "os_physical_read_bytes_measured",
        "os_page_faults_measured",
        "physical_io_evidence",
    }
)

_DIAGNOSTIC_KEYS = frozenset(
    {
        "maximum_projection_leakage",
        "gram_pre_symmetry_max_abs",
        "missing_genotype_calls",
        "observed_descriptor_passes",
        "observed_decoded_blocks",
        "observed_variant_record_visits",
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
        "same_person_pre_symmetry_max_abs",
        "group_reconstruction_max_abs",
        "direct_action_restricted_max_abs",
        "direct_genotype_restricted_max_abs",
        "direct_placement_max_abs",
        "group_reconstruction_verified",
        "same_person_global_merge_verified",
        "same_person_cross_tile_pairs_included",
        "variant_probe_coverage",
        "group_execution_variant_visits",
        "group_execution_permutation_coverage_verified",
        "group_execution_permutation_unique_verified",
        "group_execution_contiguous_verified",
        "group_tile_batches",
        "logical_order_missing_identity_verified",
        "same_person_signed_output_preserved",
        "target_dense_columns",
        "target_useful_columns",
        "strict_zero_weight_columns_eliminated",
        "operand_fingerprints_verified",
        "runtime_thread_affinity_fingerprints_verified",
        "independent_scalar_witness_verified",
        "independent_scalar_fallback_available",
        "source_identity_fnv64",
        "source_action_identity_fnv64",
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

_SCIENTIFIC_POLICY = {
    "feature_order": "P_diag_phi_G_v1",
    "pair_order": "diagonal_then_lexicographic_offdiagonal_v1",
    "component_order": "annotation_major_pair_minor_v1",
    "kernel_mass_normalization": "sum_annotation_weight_v1",
    "omega_packing": "offdiag_stored_once_v1",
    "same_person": "variant_rademacher_ustat_order2_v1",
    "population_transfer": "sample_count_same_distinct_v1",
    "deletion": "approx_group_numerator_full_D_v1",
    "raw_fit": "rank_checked_symmetric_v1",
}

_DELETION_POLICY = {
    "semantics": DeletionSemantics.APPROXIMATE_SUMMARY_ONLY_V1.value,
    "grouped_storage": "unnormalized_numerators",
    "group_numerator_unit": "mass_squared_fixed_probe_raw_action_v1",
    "same_person": "reuse_full_unchanged",
    "residual_only_moments": "reuse_full_unchanged",
    "empty_annotation": "reject",
    "multiple_group_subtraction_supported": True,
    "exact_deleted_reference_kernels": False,
    "claim": "approximate_summary_only",
}

_REFERENCE_IDENTITY_KEYS = frozenset(
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
        "sample_probe_identity_sha256",
        "variant_probe_identity_sha256",
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

_REFERENCE_SCIENTIFIC_ARRAY_KEYS = frozenset(
    {
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
            "Native contextual reference build provenance is invalid; "
            f"missing={sorted(_BUILD_PROVENANCE_KEYS - observed)}, "
            f"extra={sorted(observed - _BUILD_PROVENANCE_KEYS)}."
        )
    provenance = _json_value(value)
    validate_native_build_provenance_consistency(
        provenance,
        family="Native contextual reference",
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
        raise ValueError("Native contextual reference build provenance disagrees.")
    return provenance


def _validate_scientific_array_digests(
    value: Any, expected_keys: frozenset[str]
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(
            "Native contextual reference scientific digest map is invalid."
        )
    return {
        name: _sha256(f"scientific_array_sha256.{name}", value[name])
        for name in sorted(expected_keys)
    }


def _validate_phase_evidence(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"post_gram"}:
        raise ValueError("Native contextual reference phase evidence is invalid.")
    return {"post_gram": _sha256("phase_evidence.post_gram", value["post_gram"])}


def _validate_file_content_identity(value: Any, *, n_variants: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FILE_CONTENT_IDENTITY_KEYS:
        raise ValueError(
            "Native contextual reference file-content identity is invalid."
        )
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
        raise ValueError("Native contextual reference file-content claims disagree.")
    for name in (
        "bim_full_sha256",
        "fam_full_sha256",
        "bed_header_sha256",
        "retained_bed_logical_record_stream_sha256",
    ):
        _sha256(f"file_content_identity.{name}", identity[name])

    def exact_int(name: str, item: Any) -> int:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"Native contextual reference {name} is not an integer.")
        return item

    retained = exact_int("retained_record_count", identity["retained_record_count"])
    record_digests = exact_int(
        "retained_record_sha256_count", identity["retained_record_sha256_count"]
    )
    if retained != n_variants or record_digests != retained:
        raise ValueError(
            "Native contextual reference retained-record evidence disagrees."
        )
    if exact_int("bytes_per_bed_record", identity["bytes_per_bed_record"]) <= 0:
        raise ValueError("Native contextual reference BED record size is invalid.")
    for descriptor in ("bed", "bim", "fam"):
        state = identity[f"{descriptor}_descriptor_state"]
        if not isinstance(state, Mapping) or set(state) != _DESCRIPTOR_STATE_KEYS:
            raise ValueError("Native contextual reference descriptor state is invalid.")
        for name in _DESCRIPTOR_STATE_KEYS:
            exact_int(f"{descriptor}_descriptor_state.{name}", state[name])
        if state["size"] <= 0 or state["link_count"] <= 0:
            raise ValueError("Native contextual reference descriptor size is invalid.")
        for name in ("mtime_nanoseconds", "ctime_nanoseconds"):
            if not 0 <= state[name] < 1_000_000_000:
                raise ValueError("Native contextual reference timestamp is invalid.")
    return identity


def _exact_names(name: str, values: Any) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{name} must be a nonempty ordered string sequence.")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain only nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _name_map_digest(axis: str, names: Sequence[str]) -> str:
    return canonical_sha256({"axis": axis, "names": list(names)})


def _validate_descriptor_scale_plan(scale_plan: GenotypeScalePlanV1) -> None:
    if scale_plan.policy is not GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1:
        raise ValueError("Contextual reference V1 requires sealed affine scaling.")
    if scale_plan.allele_orientation not in {
        "bim_a1_counted_v1",
        "bim_a2_counted_v1",
        "mixed_bim_a1_a2_per_variant_v1",
    }:
        raise ValueError("Contextual reference V1 allele orientation is unsupported.")
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
                f"Contextual reference V1 scale field {field!r} is unsupported."
            )


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _json_value(value: Any) -> Any:
    """Own a native mapping as strict canonical-JSON data."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Contextual V1 metadata keys must be strings.")
            if key in result:
                raise ValueError(f"Duplicate contextual V1 metadata key {key!r}.")
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
            raise ValueError("Contextual V1 metadata cannot contain NaN or infinity.")
        return value
    raise ValueError(
        f"Contextual V1 metadata contains unsupported value {type(value).__name__}."
    )


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate contextual V1 JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("Contextual reference V1 manifest is not valid JSON.") from exc


def _array_envelope(value: np.ndarray) -> dict[str, Any]:
    dtype = value.dtype.newbyteorder("<").str
    return {
        "dtype": dtype,
        "shape": list(value.shape),
        "sha256": array_sha256(value),
    }


def _immutable_array(value: Any, *, dtype: Any) -> np.ndarray:
    """Return a canonical array backed by private immutable bytes.

    NumPy permits an owner array marked read-only to be made writable again.
    A bytes-backed view instead makes that operation fail at the buffer layer.
    """
    canonical_dtype = np.dtype(dtype).newbyteorder("<")
    source = np.ascontiguousarray(value, dtype=canonical_dtype)
    storage = bytes(source.tobytes(order="C"))
    result = np.frombuffer(storage, dtype=source.dtype).reshape(source.shape)
    result.setflags(write=False)
    return result


def contextual_variant_order_allele_sha256_v1(
    retained_variant_indices: np.ndarray,
    variant_ids: Sequence[str],
    counted_alleles: Sequence[str],
    other_alleles: Sequence[str],
    counted_allele_is_a1: np.ndarray,
) -> str:
    """Hash the logical variant/allele rows with the native V1 framing.

    This is intentionally usable before constructing or running the native
    executor, so the publication identity is an independent assertion rather
    than a value copied from native output.
    """
    if not isinstance(retained_variant_indices, np.ndarray) or (
        retained_variant_indices.dtype != np.dtype(np.int64)
        or retained_variant_indices.ndim != 1
    ):
        raise ValueError(
            "retained_variant_indices must be a one-dimensional int64 array."
        )
    if not isinstance(counted_allele_is_a1, np.ndarray) or (
        counted_allele_is_a1.dtype != np.dtype(np.uint8)
        or counted_allele_is_a1.ndim != 1
    ):
        raise ValueError("counted_allele_is_a1 must be a one-dimensional uint8 array.")
    rows = len(retained_variant_indices)
    fields = (variant_ids, counted_alleles, other_alleles)
    if any(
        not isinstance(values, Sequence) or isinstance(values, (str, bytes))
        for values in fields
    ):
        raise ValueError("Variant and allele fields must be string sequences.")
    if any(len(values) != rows for values in (*fields, counted_allele_is_a1)):
        raise ValueError("Variant/allele identity fields have inconsistent lengths.")
    digest = hashlib.sha256()
    digest.update(b"variant_order_allele_v1")
    for row in range(rows):
        digest.update(struct.pack("<q", int(retained_variant_indices[row])))
        for values in fields:
            value = values[row]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "Variant and allele identity strings must be nonempty."
                )
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        orientation = int(counted_allele_is_a1[row])
        if orientation not in (0, 1):
            raise ValueError("counted_allele_is_a1 values must be zero or one.")
        digest.update(bytes((orientation,)))
    return digest.hexdigest()


def _native_array(
    native_result: Mapping[str, Any],
    name: str,
    *,
    dtype: Any,
    ndim: int,
) -> np.ndarray:
    """Validate a native ndarray without performing a dtype-changing cast."""
    value = _required_native(native_result, name)
    expected_dtype = np.dtype(dtype)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Native contextual reference {name} is not an ndarray.")
    if value.dtype != expected_dtype or value.ndim != ndim:
        raise ValueError(f"Native contextual reference {name} dtype/rank mismatch.")
    if dtype == np.float64 and not np.all(np.isfinite(value)):
        raise ValueError(f"Native contextual reference {name} is nonfinite.")
    return value


def _relative_max(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / scale, initial=0.0))


def _validate_numa_evidence(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "numa_applicable",
        "numa_verified",
        "policy",
        "output_numa_node",
        "reason",
    }:
        raise ValueError("Native contextual reference NUMA evidence is invalid.")
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
        raise ValueError("Native contextual reference NUMA evidence is invalid.")


def _validate_manifest_before_arrays(
    manifest: Any,
) -> tuple[ContextComponentIndex, GenotypeScalePlanV1, tuple[str, ...]]:
    """Reject a wrong family/schema/map before loading scientific payloads."""
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Contextual reference V1 manifest key mismatch.")
    if manifest.get("magic") != CONTEXTUAL_REFERENCE_V1_MAGIC:
        raise ValueError("Contextual reference V1 magic mismatch.")
    identity = ContextSchemaIdentityV1(
        artifact_family=ArtifactFamily(manifest.get("artifact_family")),
        logical_schema_version=LogicalSchemaVersion(
            manifest.get("logical_schema_version")
        ),
        grouped_encoding_version=GroupedEncodingVersion(
            manifest.get("grouped_encoding_version")
        ),
        native_api_version=manifest.get("native_api_version"),
        native_backend_version=manifest.get("native_backend_version"),
        build_id=manifest.get("build_id"),
    )
    if identity.artifact_family is not ArtifactFamily.REFERENCE:
        raise ValueError("Contextual reference V1 has the wrong artifact family.")
    if identity.native_backend_version != _REFERENCE_NATIVE_BACKEND_V1:
        raise ValueError("Contextual reference V1 native backend is unsupported.")
    if manifest.get("feature_mode") != RAW_PROJECTED_FEATURE_MODE:
        raise ValueError("Contextual reference V1 feature mode mismatch.")
    if manifest.get("scientific_policy") != _SCIENTIFIC_POLICY:
        raise ValueError("Contextual reference V1 scientific policy mismatch.")
    if manifest.get("deletion") != _DELETION_POLICY:
        raise ValueError("Contextual reference V1 deletion policy mismatch.")
    if manifest.get("layouts") != _LAYOUTS:
        raise ValueError("Contextual reference V1 logical array layouts mismatch.")
    if manifest.get("terminal_status") != "published":
        raise ValueError("Contextual reference V1 is not published.")

    dimensions = manifest.get("dimensions")
    dimension_keys = {
        "N_reference",
        "M",
        "Q",
        "K",
        "P_g",
        "C",
        "J",
        "B_T",
        "B_D",
        "residual_rank",
    }
    if not isinstance(dimensions, Mapping) or set(dimensions) != dimension_keys:
        raise ValueError("Contextual reference V1 dimension schema mismatch.")
    for name in dimension_keys - {"residual_rank"}:
        minimum = 2 if name in {"B_D", "N_reference"} else 1
        _positive_int(name, dimensions[name], minimum=minimum)
    _positive_int("residual_rank", dimensions["residual_rank"], minimum=1)

    maps = manifest.get("maps")
    map_keys = {
        "pair_map",
        "pair_map_sha256",
        "component_map",
        "component_map_sha256",
        "group_ids",
        "annotation_map_sha256",
        "group_map_sha256",
        "group_execution_permutation_sha256",
        "annotation_names_sha256",
        "group_names_sha256",
    }
    if not isinstance(maps, Mapping) or set(maps) != map_keys:
        raise ValueError("Contextual reference V1 map schema mismatch.")
    component_map = maps.get("component_map")
    pair_map = maps.get("pair_map")
    if not isinstance(component_map, Mapping) or not isinstance(pair_map, Mapping):
        raise ValueError("Contextual reference V1 pair/component map is invalid.")
    annotation_names = component_map.get("annotation_names")
    if not isinstance(annotation_names, list):
        raise ValueError("Contextual reference V1 annotation names are invalid.")
    components = ContextComponentIndex(
        tuple(str(value) for value in annotation_names),
        ContextPairIndex(_positive_int("num_basis", pair_map.get("num_basis"))),
    )
    if (
        pair_map != components.pair_index.to_dict()
        or component_map != components.to_dict()
    ):
        raise ValueError("Contextual reference V1 pair/component map is noncanonical.")
    if maps.get("pair_map_sha256") != components.pair_index.digest:
        raise ValueError("Contextual reference V1 pair-map digest mismatch.")
    if maps.get("component_map_sha256") != components.digest:
        raise ValueError("Contextual reference V1 component-map digest mismatch.")
    for name in (
        "annotation_map_sha256",
        "group_map_sha256",
        "group_execution_permutation_sha256",
    ):
        _sha256(name, maps.get(name))
    group_values = maps.get("group_ids")
    group_ids = _exact_names("Contextual reference V1 group IDs", group_values)
    if maps.get("annotation_names_sha256") != _name_map_digest(
        "annotation", components.annotation_names
    ):
        raise ValueError("Contextual reference V1 annotation-name digest mismatch.")
    if maps.get("group_names_sha256") != _name_map_digest("deletion_group", group_ids):
        raise ValueError("Contextual reference V1 group-name digest mismatch.")

    scale = manifest.get("genotype_scale_plan")
    if not isinstance(scale, Mapping):
        raise ValueError("Contextual reference V1 scale plan is invalid.")
    try:
        scale_plan = GenotypeScalePlanV1(
            policy=GenotypeScalePolicy(scale.get("policy")),
            retained_variant_order_sha256=scale.get("retained_variant_order_sha256"),
            allele_orientation=scale.get("allele_orientation"),
            allele_coding=scale.get("allele_coding"),
            centering_source=scale.get("centering_source"),
            centering_formula=scale.get("centering_formula"),
            scaling_formula=scale.get("scaling_formula"),
            missing_imputation=scale.get("missing_imputation"),
            ploidy_policy=scale.get("ploidy_policy"),
            affine_mean_sha256=scale.get("affine_mean_sha256"),
            affine_inverse_scale_sha256=scale.get("affine_inverse_scale_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Contextual reference V1 scale plan is invalid.") from exc
    if manifest.get("genotype_scale_plan_sha256") != scale_plan.digest:
        raise ValueError("Contextual reference V1 scale-plan digest mismatch.")
    _validate_descriptor_scale_plan(scale_plan)

    identities = manifest.get("identity")
    if (
        not isinstance(identities, Mapping)
        or set(identities) != _REFERENCE_IDENTITY_KEYS
    ):
        raise ValueError("Contextual reference V1 identity fields mismatch.")
    for name in _REFERENCE_IDENTITY_KEYS:
        _sha256(name, identities[name])
    if identities["genotype_scale_plan_sha256"] != scale_plan.digest:
        raise ValueError("Contextual reference V1 authority scale digest mismatch.")
    if identities["annotation_map_sha256"] != maps["annotation_map_sha256"]:
        raise ValueError(
            "Contextual reference V1 authority annotation digest mismatch."
        )
    if identities["annotation_names_sha256"] != maps["annotation_names_sha256"]:
        raise ValueError("Contextual reference V1 authority annotation names mismatch.")
    if identities["group_map_sha256"] != maps["group_map_sha256"]:
        raise ValueError("Contextual reference V1 authority group digest mismatch.")
    if identities["group_names_sha256"] != maps["group_names_sha256"]:
        raise ValueError("Contextual reference V1 authority group names mismatch.")

    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(_ARRAY_NAMES):
        raise ValueError("Contextual reference V1 array schema mismatch.")
    expected_shapes = {
        "gram": [dimensions["C"], dimensions["C"]],
        "same_person": [dimensions["C"], dimensions["C"]],
        "group_gram_unnormalized_num": [
            dimensions["J"],
            dimensions["C"],
            dimensions["C"],
        ],
        "annotation_masses": [dimensions["K"]],
        "group_annotation_masses": [dimensions["J"], dimensions["K"]],
        "group_variant_counts": [dimensions["J"]],
    }
    for name, shape in expected_shapes.items():
        envelope = arrays[name]
        expected_dtype = "<i8" if name == "group_variant_counts" else "<f8"
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(f"Contextual reference V1 {name} envelope is invalid.")
        if envelope.get("dtype") != expected_dtype or envelope.get("shape") != shape:
            raise ValueError(f"Contextual reference V1 {name} schema is invalid.")
        _sha256(f"arrays.{name}.sha256", envelope.get("sha256"))
    return components, scale_plan, group_ids


@dataclass(frozen=True)
class ContextualReferencePublicationIdentityV1:
    """Python-owned semantic identities not derivable from native matrices."""

    sample_order_sha256: str
    variant_order_allele_sha256: str
    fixed_effect_spec_sha256: str
    basis_specification_sha256: str
    basis_calibration_sha256: str
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
    sample_probe_policy: str
    sample_probe_identity_sha256: str
    variant_probe_policy: str
    variant_probe_identity_sha256: str

    def __post_init__(self) -> None:
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
            "sample_probe_identity_sha256",
            "variant_probe_identity_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        annotations = _exact_names("annotation_names", self.annotation_names)
        groups = _exact_names("group_ids", self.group_ids)
        object.__setattr__(self, "annotation_names", annotations)
        object.__setattr__(self, "group_ids", groups)
        if self.sample_probe_policy not in {
            "explicit_rademacher_v1",
            "numpy_philox_per_probe_key_v1",
        }:
            raise ValueError("Unsupported reference sample-probe authority policy.")
        if self.variant_probe_policy not in {
            "explicit_variant_rademacher_v1",
            "numpy_philox_variant_per_probe_key_v1",
        }:
            raise ValueError("Unsupported reference variant-probe authority policy.")

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
                "sample_probe_identity_sha256",
                "variant_probe_identity_sha256",
            )
        }
        values["annotation_names_sha256"] = _name_map_digest(
            "annotation", self.annotation_names
        )
        values["group_names_sha256"] = _name_map_digest(
            "deletion_group", self.group_ids
        )
        return values


@dataclass(frozen=True)
class ContextualReferenceArtifactV1:
    """Complete compact contextual-reference artifact.

    Every array is defensively owned, C-contiguous, and read-only.  The only
    leading contribution axis is the declared deletion-group axis; no array
    retains a reference-sample or retained-variant axis.
    """

    manifest: Mapping[str, Any]
    component_index: ContextComponentIndex
    scale_plan: GenotypeScalePlanV1
    group_ids: tuple[str, ...]
    gram: np.ndarray
    same_person: np.ndarray
    group_gram_unnormalized_num: np.ndarray
    annotation_masses: np.ndarray
    group_annotation_masses: np.ndarray
    group_variant_counts: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.component_index, ContextComponentIndex):
            raise ValueError("component_index must be a ContextComponentIndex.")
        if not isinstance(self.scale_plan, GenotypeScalePlanV1):
            raise ValueError("scale_plan must be a GenotypeScalePlanV1.")
        groups = _exact_names("Contextual V1 group IDs", self.group_ids)
        object.__setattr__(self, "group_ids", groups)
        for name in _ARRAY_NAMES:
            dtype = np.int64 if name == "group_variant_counts" else np.float64
            array = _immutable_array(getattr(self, name), dtype=dtype)
            object.__setattr__(self, name, array)
        manifest = _json_value(self.manifest)
        _validate_artifact(self, manifest)
        object.__setattr__(self, "manifest", freeze_context_mapping(manifest))

    @property
    def n_samples(self) -> int:
        return int(self.manifest["dimensions"]["N_reference"])

    @property
    def reference_n(self) -> int:
        return self.n_samples

    @property
    def n_variants(self) -> int:
        return int(self.manifest["dimensions"]["M"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["dimensions"]["residual_rank"])

    @property
    def loo_group_ids(self) -> tuple[str, ...]:
        return self.group_ids

    @property
    def gram_numerator_contributions(self) -> np.ndarray:
        return self.group_gram_unnormalized_num

    @property
    def full_moments(self) -> ReferenceMoments:
        return ReferenceMoments(
            annotation_masses=self.annotation_masses,
            gram=self.gram,
            same_person=self.same_person,
        )

    def transferred_gram(self, study_n: int) -> np.ndarray:
        return transfer_reference_gram(
            self.gram,
            self.same_person,
            reference_n=self.n_samples,
            study_n=study_n,
        )

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.manifest)

    def verify(self) -> None:
        """Recheck typed array digests and every semantic identity in memory."""
        _validate_artifact(self, _json_value(self.manifest))

    def to_development_grouped_reference(
        self,
        *,
        basis_hash: str,
        fixed_effect_hash: str,
        variant_hash: str,
        annotation_hash: str,
        loo_grouping_hash: str,
        annotation_binding: Mapping[str, str] | None = None,
    ) -> GroupedContextReference:
        """Explicit private bridge to the pre-V1 fitter used before Stage 4.

        This does not write or relabel the stable artifact.  Every supplied
        development identity is a digest and the stable maps/numerators are
        revalidated immediately before construction.
        """
        for name, value in (
            ("basis_hash", basis_hash),
            ("fixed_effect_hash", fixed_effect_hash),
            ("variant_hash", variant_hash),
            ("annotation_hash", annotation_hash),
            ("loo_grouping_hash", loo_grouping_hash),
        ):
            _sha256(name, value)
        # Detect even forced mutation of a read-only array before bridging.
        _validate_artifact(self, _json_value(self.manifest))
        dimensions = self.manifest["dimensions"]
        execution_mode = AnnotationMode(self.manifest["execution"]["annotation_mode"])
        development_annotation: dict[str, str] = {}
        if execution_mode is AnnotationMode.STRICT_DISJOINT_BINARY_V1:
            required = {
                "annotation_definition_hash",
                "annotation_membership_hash",
                "annotation_partition_hash",
            }
            if (
                not isinstance(annotation_binding, Mapping)
                or set(annotation_binding) != required
            ):
                raise ValueError(
                    "The strict-disjoint development bridge requires exact partition "
                    "definition, membership, and identity digests."
                )
            development_annotation["annotation_mode"] = "disjoint_partition"
            for name in required:
                development_annotation[name] = _sha256(name, annotation_binding[name])
        elif annotation_binding is not None:
            raise ValueError(
                "Generic-overlap artifacts cannot carry a strict development partition binding."
            )
        manifest = {
            "kind": "summit.context.reference",
            "schema_version": 1,
            "feature_mode": RAW_PROJECTED_FEATURE_MODE,
            "genotype_scaling": self.scale_plan.digest,
            "basis_hash": basis_hash,
            "fixed_effect_hash": fixed_effect_hash,
            "variant_hash": variant_hash,
            "annotation_hash": annotation_hash,
            "component_index_hash": self.component_index.digest,
            "loo_grouping_hash": loo_grouping_hash,
            "dimensions": {
                "n_samples": self.n_samples,
                "residual_rank": self.residual_rank,
                "n_variants": self.n_variants,
                "q": int(dimensions["Q"]),
                "k": int(dimensions["K"]),
                "p_genetic": int(dimensions["C"]),
                "loo_groups": int(dimensions["J"]),
            },
            "component_order": list(self.component_index.names),
            "annotation_names": list(self.component_index.annotation_names),
            "backend": {
                "name": "explicit_contextual_reference_v1_bridge",
                "source_manifest_sha256": self.manifest_sha256,
            },
            "approximate_loo": {
                "method": "symmetric_grouped_gram_numerator_v1",
                "contribution_storage": "loo_grouped",
                "exact_deleted_kernels": False,
                "same_person_deletion": "reuse_full_reference_D",
                "groups": len(self.group_ids),
                "group_labels": list(self.group_ids),
                "group_variant_counts": self.group_variant_counts.tolist(),
                "group_annotation_masses_hash": array_sha256(
                    self.group_annotation_masses
                ),
            },
            **development_annotation,
        }
        return GroupedContextReference(
            manifest=manifest,
            component_index=self.component_index,
            annotation_masses=self.annotation_masses,
            gram=self.gram,
            same_person=self.same_person,
            gram_numerator_contributions=self.group_gram_unnormalized_num,
            loo_group_ids=self.group_ids,
            group_annotation_masses=self.group_annotation_masses,
            group_variant_counts=self.group_variant_counts,
            phase_times_seconds={},
            peak_rss_bytes=0,
        )


def reference_moments_after_deleting_groups_v1(
    artifact: ContextualReferenceArtifactV1,
    groups: Sequence[str],
) -> ReferenceMoments:
    """Apply the frozen approximate deletion in unnormalized numerator units."""
    if not isinstance(artifact, ContextualReferenceArtifactV1):
        raise ValueError("artifact must be a ContextualReferenceArtifactV1.")
    artifact.verify()
    if isinstance(groups, (str, bytes)):
        raise ValueError("Deleted contextual V1 group IDs must be a sequence.")
    requested = tuple(groups)
    if any(not isinstance(value, str) or not value for value in requested):
        raise ValueError("Deleted contextual V1 group IDs must be nonempty strings.")
    if len(set(requested)) != len(requested):
        raise ValueError("Deleted contextual V1 group IDs must be unique.")
    if not requested:
        return artifact.full_moments
    unknown = set(requested) - set(artifact.group_ids)
    if unknown:
        raise ValueError(f"Unknown approximate-LOO groups: {sorted(unknown)}.")
    deleted = np.fromiter(
        (group in set(requested) for group in artifact.group_ids),
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
    numerator = np.sum(
        artifact.group_gram_unnormalized_num[~deleted],
        axis=0,
        dtype=np.float64,
    )
    gram = numerator / np.outer(component_masses, component_masses)
    return ReferenceMoments(
        annotation_masses=retained_masses,
        gram=0.5 * (gram + gram.T),
        same_person=artifact.same_person,
    )


def _validate_artifact(
    artifact: ContextualReferenceArtifactV1, manifest: Mapping[str, Any]
) -> None:
    if set(manifest) != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - set(manifest))
        extra = sorted(set(manifest) - _MANIFEST_KEYS)
        raise ValueError(
            f"Contextual reference V1 manifest key mismatch; missing={missing}, "
            f"extra={extra}."
        )
    if manifest["magic"] != CONTEXTUAL_REFERENCE_V1_MAGIC:
        raise ValueError("Contextual reference V1 magic mismatch.")
    identity = ContextSchemaIdentityV1(
        artifact_family=ArtifactFamily(manifest["artifact_family"]),
        logical_schema_version=LogicalSchemaVersion(manifest["logical_schema_version"]),
        grouped_encoding_version=GroupedEncodingVersion(
            manifest["grouped_encoding_version"]
        ),
        native_api_version=manifest["native_api_version"],
        native_backend_version=manifest["native_backend_version"],
        build_id=manifest["build_id"],
    )
    if identity.artifact_family is not ArtifactFamily.REFERENCE:
        raise ValueError("Contextual reference V1 has the wrong artifact family.")
    if identity.native_backend_version != _REFERENCE_NATIVE_BACKEND_V1:
        raise ValueError("Contextual reference V1 native backend is unsupported.")
    if manifest["feature_mode"] != RAW_PROJECTED_FEATURE_MODE:
        raise ValueError("Contextual reference V1 feature mode mismatch.")
    if manifest["scientific_policy"] != _SCIENTIFIC_POLICY:
        raise ValueError("Contextual reference V1 scientific policy mismatch.")
    if manifest["deletion"] != _DELETION_POLICY:
        raise ValueError("Contextual reference V1 deletion policy mismatch.")
    if manifest["layouts"] != _LAYOUTS:
        raise ValueError("Contextual reference V1 logical array layouts mismatch.")
    if manifest["terminal_status"] != "published":
        raise ValueError("Contextual reference V1 is not published.")
    if manifest["genotype_scale_plan"] != artifact.scale_plan.to_dict():
        raise ValueError("Contextual reference V1 scale-plan fields mismatch.")
    if manifest["genotype_scale_plan_sha256"] != artifact.scale_plan.digest:
        raise ValueError("Contextual reference V1 scale-plan digest mismatch.")
    _validate_descriptor_scale_plan(artifact.scale_plan)

    dimensions = manifest["dimensions"]
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "N_reference",
        "M",
        "Q",
        "K",
        "P_g",
        "C",
        "J",
        "B_T",
        "B_D",
        "residual_rank",
    }:
        raise ValueError("Contextual reference V1 dimension schema mismatch.")
    n = _positive_int("N_reference", dimensions["N_reference"], minimum=2)
    m = _positive_int("M", dimensions["M"])
    q = _positive_int("Q", dimensions["Q"])
    k = _positive_int("K", dimensions["K"])
    p = _positive_int("P_g", dimensions["P_g"])
    c = _positive_int("C", dimensions["C"])
    j = _positive_int("J", dimensions["J"])
    _positive_int("B_T", dimensions["B_T"])
    _positive_int("B_D", dimensions["B_D"], minimum=2)
    residual_rank = _positive_int(
        "residual_rank", dimensions["residual_rank"], minimum=1
    )
    if residual_rank > n:
        raise ValueError("Contextual reference V1 residual rank exceeds N.")
    if q != artifact.component_index.pair_index.num_basis:
        raise ValueError("Contextual reference V1 Q/map mismatch.")
    if k != len(artifact.component_index.annotation_names):
        raise ValueError("Contextual reference V1 K/map mismatch.")
    if p != len(artifact.component_index.pair_index) or c != len(
        artifact.component_index
    ):
        raise ValueError("Contextual reference V1 pair/component count mismatch.")
    if j != len(artifact.group_ids):
        raise ValueError("Contextual reference V1 J/group-map mismatch.")

    expected_shapes = {
        "gram": (c, c),
        "same_person": (c, c),
        "group_gram_unnormalized_num": (j, c, c),
        "annotation_masses": (k,),
        "group_annotation_masses": (j, k),
        "group_variant_counts": (j,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(artifact, name))
        if value.shape != shape:
            raise ValueError(
                f"Contextual reference V1 array {name!r} has shape "
                f"{value.shape}; expected {shape}."
            )
        expected_dtype = np.dtype(
            np.int64 if name == "group_variant_counts" else np.float64
        ).newbyteorder("<")
        if value.dtype != expected_dtype:
            raise ValueError(
                f"Contextual reference V1 array {name!r} has dtype {value.dtype}; "
                f"expected {expected_dtype}."
            )
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"Contextual reference V1 array {name!r} is non-finite.")

    if np.any(artifact.annotation_masses <= 0.0):
        raise ValueError("Contextual reference V1 annotation masses must be positive.")
    if np.any(artifact.group_annotation_masses < 0.0):
        raise ValueError("Contextual reference V1 group masses must be nonnegative.")
    if np.any(artifact.group_variant_counts <= 0):
        raise ValueError("Contextual reference V1 group counts must be positive.")
    if sum(int(value) for value in artifact.group_variant_counts) != m:
        raise ValueError("Contextual reference V1 group counts do not reconstruct M.")
    if (
        _relative_max(
            np.sum(artifact.group_annotation_masses, axis=0, dtype=np.float64),
            artifact.annotation_masses,
        )
        > 1.0e-12
    ):
        raise ValueError(
            "Contextual reference V1 group masses do not reconstruct totals."
        )
    if np.any(
        artifact.annotation_masses[None, :] - artifact.group_annotation_masses <= 0.0
    ):
        raise ValueError(
            "Contextual reference V1 contains a single-group deletion that empties "
            "an annotation."
        )
    for name in ("gram", "same_person"):
        value = np.asarray(getattr(artifact, name))
        if _relative_max(value, value.T) > 1.0e-12:
            raise ValueError(f"Contextual reference V1 {name} is not symmetric.")
    if (
        _relative_max(
            artifact.group_gram_unnormalized_num,
            np.swapaxes(artifact.group_gram_unnormalized_num, 1, 2),
        )
        > 1.0e-12
    ):
        raise ValueError(
            "Contextual reference V1 grouped numerators are not symmetric."
        )
    component_annotation = np.asarray(
        [entry.annotation_index for entry in artifact.component_index.entries],
        dtype=np.int64,
    )
    component_masses = artifact.annotation_masses[component_annotation]
    reconstructed = np.sum(
        artifact.group_gram_unnormalized_num, axis=0, dtype=np.float64
    ) / np.outer(component_masses, component_masses)
    if _relative_max(reconstructed, artifact.gram) > 1.0e-10:
        raise ValueError(
            "Contextual reference V1 grouped numerators do not reconstruct Gram."
        )

    maps = manifest["maps"]
    expected_maps = {
        "pair_map": artifact.component_index.pair_index.to_dict(),
        "pair_map_sha256": artifact.component_index.pair_index.digest,
        "component_map": artifact.component_index.to_dict(),
        "component_map_sha256": artifact.component_index.digest,
        "group_ids": list(artifact.group_ids),
    }
    if not isinstance(maps, Mapping) or not set(expected_maps).issubset(maps):
        raise ValueError("Contextual reference V1 map schema mismatch.")
    for key, value in expected_maps.items():
        if maps[key] != value:
            raise ValueError(f"Contextual reference V1 {key} mismatch.")
    if set(maps) != {
        *expected_maps,
        "annotation_map_sha256",
        "group_map_sha256",
        "group_execution_permutation_sha256",
        "annotation_names_sha256",
        "group_names_sha256",
    }:
        raise ValueError("Contextual reference V1 map fields mismatch.")
    for name in (
        "annotation_map_sha256",
        "group_map_sha256",
        "group_execution_permutation_sha256",
    ):
        _sha256(name, maps[name])
    if maps["annotation_names_sha256"] != _name_map_digest(
        "annotation", artifact.component_index.annotation_names
    ):
        raise ValueError("Contextual reference V1 annotation-name digest mismatch.")
    if maps["group_names_sha256"] != _name_map_digest(
        "deletion_group", artifact.group_ids
    ):
        raise ValueError("Contextual reference V1 group-name digest mismatch.")

    identities = manifest["identity"]
    if (
        not isinstance(identities, Mapping)
        or set(identities) != _REFERENCE_IDENTITY_KEYS
    ):
        raise ValueError("Contextual reference V1 identity fields mismatch.")
    for name in _REFERENCE_IDENTITY_KEYS:
        _sha256(name, identities[name])
    if (
        identities["retained_variant_order_sha256"]
        != artifact.scale_plan.retained_variant_order_sha256
    ):
        raise ValueError(
            "Contextual reference V1 retained variant order identity mismatch."
        )
    authority_pairs = {
        "genotype_scale_plan_sha256": artifact.scale_plan.digest,
        "annotation_map_sha256": maps["annotation_map_sha256"],
        "annotation_names_sha256": maps["annotation_names_sha256"],
        "group_map_sha256": maps["group_map_sha256"],
        "group_names_sha256": maps["group_names_sha256"],
    }
    for name, expected in authority_pairs.items():
        if identities[name] != expected:
            raise ValueError(f"Contextual reference V1 authority {name} mismatch.")

    arrays = manifest["arrays"]
    if not isinstance(arrays, Mapping) or set(arrays) != set(_ARRAY_NAMES):
        raise ValueError("Contextual reference V1 array manifest mismatch.")
    for name in _ARRAY_NAMES:
        if arrays[name] != _array_envelope(np.asarray(getattr(artifact, name))):
            raise ValueError(
                f"Contextual reference V1 array digest mismatch for {name}."
            )

    probes = manifest["probes"]
    if not isinstance(probes, Mapping) or set(probes) != {"sample", "variant"}:
        raise ValueError("Contextual reference V1 probe schema mismatch.")
    for name, minimum in (("sample", 1), ("variant", 2)):
        item = probes[name]
        if not isinstance(item, Mapping) or set(item) != {
            "policy",
            "count",
            "identity_sha256",
        }:
            raise ValueError(f"Contextual reference V1 {name} probe fields mismatch.")
        if not isinstance(item["policy"], str) or not item["policy"]:
            raise ValueError(f"Contextual reference V1 {name} probe policy is invalid.")
        _positive_int(f"{name} probe count", item["count"], minimum=minimum)
        _sha256(f"{name} probe identity", item["identity_sha256"])
    if int(probes["sample"]["count"]) != int(dimensions["B_T"]):
        raise ValueError("Contextual reference V1 sample-probe count mismatch.")
    if int(probes["variant"]["count"]) != int(dimensions["B_D"]):
        raise ValueError("Contextual reference V1 variant-probe count mismatch.")
    if probes["sample"]["policy"] not in {
        "explicit_rademacher_v1",
        "numpy_philox_per_probe_key_v1",
    }:
        raise ValueError("Contextual reference V1 sample-probe policy is invalid.")
    if probes["variant"]["policy"] not in {
        "explicit_variant_rademacher_v1",
        "numpy_philox_variant_per_probe_key_v1",
    }:
        raise ValueError("Contextual reference V1 variant-probe policy is invalid.")
    for axis in ("sample", "variant"):
        if (
            identities[f"{axis}_probe_identity_sha256"]
            != probes[axis]["identity_sha256"]
        ):
            raise ValueError(
                f"Contextual reference V1 authority {axis}-probe digest mismatch."
            )

    execution = manifest["execution"]
    if not isinstance(execution, Mapping) or set(execution) != {
        "annotation_mode",
        "annotation_output_contract",
        "numeric_policy",
        "grouped_attribution_algorithm",
        "direct_grouped_scaling",
        "grouped_differential_enabled",
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
    }:
        raise ValueError("Contextual reference V1 execution schema mismatch.")
    mode = AnnotationMode(execution["annotation_mode"])
    if execution["annotation_output_contract"] != dict(
        annotation_output_contract(mode)
    ):
        raise ValueError("Contextual reference V1 annotation interpretation mismatch.")
    if execution["numeric_policy"] != "fp64_v1":
        raise ValueError("Contextual reference V1 numeric policy mismatch.")
    validate_python_source_runtime_provenance(
        execution["python_adapter_provenance"],
        ordered_modules=_PYTHON_ADAPTER_MODULES,
        policy=_PYTHON_ADAPTER_PROVENANCE_POLICY,
        include_scipy=False,
        family="Contextual reference V1",
    )
    selected_algorithm = GroupedAttributionAlgorithm(
        execution["grouped_attribution_algorithm"]
    )
    if selected_algorithm is GroupedAttributionAlgorithm.AUTO_QUALIFIED_V1:
        raise ValueError(
            "Contextual reference V1 must record a resolved grouped algorithm."
        )
    if execution["direct_grouped_scaling"] not in {
        "action_scaled_v1",
        "genotype_scaled_v1",
        "both_differential_v1",
    }:
        raise ValueError("Contextual reference V1 direct grouped scaling is invalid.")
    if not isinstance(execution["grouped_differential_enabled"], bool):
        raise ValueError(
            "Contextual reference V1 grouped differential flag is invalid."
        )
    for name in ("admission", "diagnostics", "telemetry"):
        if not isinstance(execution[name], Mapping):
            raise ValueError(f"Contextual reference V1 execution {name} is invalid.")
    _validate_native_execution_evidence(
        execution["admission"],
        execution["diagnostics"],
        execution["telemetry"],
        differential=execution["grouped_differential_enabled"],
        annotation_mode=mode.value,
        n=int(dimensions["N_reference"]),
        m=m,
        q=int(dimensions["Q"]),
        k=int(dimensions["K"]),
        c=int(dimensions["C"]),
        j=int(dimensions["J"]),
        bt=int(dimensions["B_T"]),
        bd=int(dimensions["B_D"]),
    )
    grouped_phase = execution["admission"]["phase_ledger"]["grouped"]
    if grouped_phase["selected_algorithm"] != selected_algorithm.value:
        raise ValueError("Contextual reference V1 grouped phase algorithm mismatch.")
    if grouped_phase["execution_order"] != "sealed_group_contiguous_permutation_v1":
        raise ValueError("Contextual reference V1 grouped execution order mismatch.")
    if execution["file_identity_policy"] != (_FILE_IDENTITY_POLICY_V1):
        raise ValueError("Contextual reference V1 file identity policy is invalid.")
    _validate_file_content_identity(execution["file_content_identity"], n_variants=m)
    if execution["native_execution_backend"] != (
        "deterministic_tiled_fp64_with_scalar_witness_v1"
    ):
        raise ValueError("Contextual reference V1 execution backend is invalid.")
    if execution["output_ownership"] != {"owns_data": True, "read_only": True}:
        raise ValueError("Contextual reference V1 output ownership is invalid.")
    _sha256("execution.execution_plan_sha256", execution["execution_plan_sha256"])
    _validate_build_provenance(
        execution["build_provenance"],
        build_id=manifest["build_id"],
        source_tree_sha256=identities["source_tree_sha256"],
    )
    scientific_digests = _validate_scientific_array_digests(
        execution["native_scientific_array_sha256"],
        _REFERENCE_SCIENTIFIC_ARRAY_KEYS,
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
            raise ValueError(
                f"Contextual reference V1 native digest mismatch for {name}."
            )
    _validate_numa_evidence(execution["numa"])
    if (
        selected_algorithm is GroupedAttributionAlgorithm.DIRECT_GROUPED_TN_V1
        and execution["diagnostics"].get("direct_grouped_plan_qualified") is not True
    ):
        raise ValueError("Direct grouped TN was not qualified for publication.")

    # The container stores this digest separately, avoiding a self-referential
    # manifest while still authenticating the entire canonical object.
    canonical_sha256(manifest)


def _required_native(result: Mapping[str, Any], name: str) -> Any:
    if name not in result:
        raise ValueError(f"Native contextual reference result is missing {name!r}.")
    return result[name]


def _native_u64(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Native contextual reference {name} is not an integer.")
    result = int(value)
    if result < 0 or result >= 1 << 64:
        raise ValueError(f"Native contextual reference {name} is outside uint64.")
    return result


def _native_fnv64(name: str, value: Any) -> int:
    if not isinstance(value, str) or not value or not value.isdecimal():
        raise ValueError(f"Native contextual reference {name} is not decimal uint64.")
    result = int(value, 10)
    if result < 0 or result >= 1 << 64 or str(result) != value:
        raise ValueError(f"Native contextual reference {name} is not canonical uint64.")
    return result


def _validate_native_execution_evidence(
    admission: Any,
    diagnostics: Any,
    telemetry: Any,
    *,
    differential: Any,
    annotation_mode: str,
    n: int,
    m: int,
    q: int,
    k: int,
    c: int,
    j: int,
    bt: int,
    bd: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(admission, Mapping) or set(admission) != _ADMISSION_KEYS:
        raise ValueError("Native contextual reference admission schema mismatch.")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != _DIAGNOSTIC_KEYS:
        raise ValueError("Native contextual reference diagnostics schema mismatch.")
    if not isinstance(telemetry, Mapping) or set(telemetry) != _TELEMETRY_KEYS:
        raise ValueError("Native contextual reference telemetry schema mismatch.")

    calls = admission["semantic_call_ledger"]
    if not isinstance(calls, Mapping) or set(calls) != set(_SEMANTIC_OPERATIONS):
        raise ValueError("Native contextual reference semantic-call schema mismatch.")
    call_counts = {
        name: _native_u64(f"semantic_call_ledger.{name}", calls[name])
        for name in _SEMANTIC_OPERATIONS
    }
    if any(call_counts[name] == 0 for name in _SEMANTIC_OPERATIONS[:9]):
        raise ValueError(
            "Native contextual reference omitted an active reference operation."
        )
    if any(call_counts[name] == 0 for name in _SEMANTIC_OPERATIONS[10:14]):
        raise ValueError("Native contextual reference omitted a same-person operation.")
    if any(call_counts[name] != 0 for name in _SEMANTIC_OPERATIONS[14:]):
        raise ValueError("Reference execution reported trait-only protected calls.")
    if not isinstance(differential, bool):
        raise ValueError("Native grouped differential flag must be boolean.")
    direct_calls = call_counts["direct_grouped_tn"]
    if (differential and direct_calls == 0) or (not differential and direct_calls != 0):
        raise ValueError("Native direct-grouped call ledger disagrees with its plan.")
    total_calls = sum(call_counts.values())
    if (
        _native_u64("total_protected_calls", admission["total_protected_calls"])
        != total_calls
    ):
        raise ValueError("Native contextual reference protected-call total mismatch.")

    numeric_admission = _ADMISSION_KEYS - {
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
        for name in numeric_admission
    }
    expected_passes = sum(
        admitted[name]
        for name in (
            "source_descriptor_passes",
            "action_descriptor_passes",
            "group_descriptor_passes",
            "same_person_descriptor_passes",
        )
    )
    expected_blocks = sum(
        admitted[name]
        for name in (
            "source_decoded_blocks",
            "action_decoded_blocks",
            "group_decoded_blocks",
            "same_person_decoded_blocks",
        )
    )
    if admitted["total_descriptor_passes"] != expected_passes:
        raise ValueError("Native contextual reference descriptor-pass total mismatch.")
    if admitted["total_decoded_blocks"] != expected_blocks:
        raise ValueError("Native contextual reference decoded-block total mismatch.")
    if admitted["total_variant_record_visits"] != expected_passes * m:
        raise ValueError("Native contextual reference variant-visit total mismatch.")
    required_workspace = (
        admitted["permanent_bytes"]
        + max(admitted["source_phase_bytes"], admitted["action_phase_bytes"])
        + admitted["group_phase_bytes"]
        + admitted["same_person_phase_bytes"]
        + admitted["integrity_reserve_bytes"]
        + admitted["compact_output_bytes"]
        + admitted["telemetry_bytes"]
    )
    if admitted["required_workspace_bytes"] != required_workspace:
        raise ValueError("Native contextual reference workspace ledger mismatch.")
    if admitted["required_telemetry_capacity"] != total_calls * 8 + 32:
        raise ValueError("Native contextual reference telemetry admission mismatch.")

    memory_names = (
        "permanent_bytes",
        "source_phase_bytes",
        "action_phase_bytes",
        "group_phase_bytes",
        "same_person_phase_bytes",
        "compact_output_bytes",
        "integrity_reserve_bytes",
        "telemetry_bytes",
    )
    memory_ledger = admission["memory_ledger"]
    if not isinstance(memory_ledger, Mapping) or set(memory_ledger) != set(
        memory_names
    ):
        raise ValueError("Native contextual reference memory-ledger schema mismatch.")
    for name in memory_names:
        if _native_u64(f"memory_ledger.{name}", memory_ledger[name]) != admitted[name]:
            raise ValueError(
                "Native contextual reference memory ledger disagrees with admission."
            )
    lifetime_policy = {
        "permanent_bytes": ("admission_through_publication", 1),
        "source_phase_bytes": ("source_and_action_preallocated_arena", 1),
        "action_phase_bytes": ("source_and_action_preallocated_arena", 1),
        "group_phase_bytes": ("preallocated_group_arena", 1),
        "same_person_phase_bytes": ("preallocated_same_person_arena", 1),
        "compact_output_bytes": ("publication", 1),
        "integrity_reserve_bytes": ("all_protected_phases", 4),
        "telemetry_bytes": ("admission_through_publication", 1),
    }
    memory_lifetimes = admission["memory_lifetimes"]
    if not isinstance(memory_lifetimes, Mapping) or set(memory_lifetimes) != set(
        memory_names
    ):
        raise ValueError("Native contextual reference memory-lifetime schema mismatch.")
    for name, (phase, copies) in lifetime_policy.items():
        item = memory_lifetimes[name]
        if not isinstance(item, Mapping) or set(item) != {
            "byte_count",
            "live_phase",
            "concurrent_copies",
        }:
            raise ValueError(
                "Native contextual reference memory lifetime is malformed."
            )
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
            raise ValueError("Native contextual reference memory lifetime mismatch.")
    if admission["memory_accounting_model"] != (
        "tracked_vector_payload_bytes_v1_excludes_allocator_metadata_and_small_strings"
    ):
        raise ValueError("Native contextual reference memory model mismatch.")

    selected_tiles = admission["selected_tiles"]
    if not isinstance(selected_tiles, Mapping) or set(selected_tiles) != {
        "variant_block",
        "sample_probe_resident",
        "sample_probe_tile",
        "action_tile",
        "annotation_tile",
        "context_tile",
        "variant_probe_tile",
        "group_tile",
    }:
        raise ValueError("Native contextual reference selected-tile schema mismatch.")
    for name, value in selected_tiles.items():
        if _native_u64(f"selected_tiles.{name}", value) < 1:
            raise ValueError("Native contextual reference selected tile is zero.")
    phases = admission["phase_ledger"]
    if not isinstance(phases, Mapping) or set(phases) != {
        "source",
        "action",
        "grouped",
        "same_person",
    }:
        raise ValueError("Native contextual reference phase-ledger schema mismatch.")
    common_phase_keys = {
        "descriptor_passes",
        "decoded_blocks",
        "variant_record_visits",
        "logical_bed_record_bytes_touched",
        "access_order",
        "logical_duplicate_variant_decodes",
    }
    for phase in ("source", "action"):
        item = phases[phase]
        if not isinstance(item, Mapping) or set(item) != common_phase_keys:
            raise ValueError(f"Native contextual reference {phase} ledger mismatch.")
    grouped_phase = phases["grouped"]
    if not isinstance(grouped_phase, Mapping) or set(grouped_phase) != {
        *common_phase_keys,
        "execution_order",
        "selected_algorithm",
        "group_restricted_duplicate_variant_decodes",
        "group_restricted_variant_coverage",
    }:
        raise ValueError("Native contextual reference grouped ledger mismatch.")
    same_phase = phases["same_person"]
    if not isinstance(same_phase, Mapping) or set(same_phase) != {
        *common_phase_keys,
        "global_probe_merge_before_finalization",
    }:
        raise ValueError("Native contextual reference same-person ledger mismatch.")
    if same_phase["global_probe_merge_before_finalization"] is not True:
        raise ValueError(
            "Native same-person probes were finalized before global merge."
        )
    phase_fields = {
        "source": ("source_descriptor_passes", "source_decoded_blocks"),
        "action": ("action_descriptor_passes", "action_decoded_blocks"),
        "grouped": ("group_descriptor_passes", "group_decoded_blocks"),
        "same_person": (
            "same_person_descriptor_passes",
            "same_person_decoded_blocks",
        ),
    }
    observed_bytes_per_record: set[int] = set()
    for phase, (passes_name, blocks_name) in phase_fields.items():
        item = phases[phase]
        passes = _native_u64(
            f"phase_ledger.{phase}.descriptor_passes", item["descriptor_passes"]
        )
        blocks = _native_u64(
            f"phase_ledger.{phase}.decoded_blocks", item["decoded_blocks"]
        )
        visits = _native_u64(
            f"phase_ledger.{phase}.variant_record_visits", item["variant_record_visits"]
        )
        if (
            passes != admitted[passes_name]
            or blocks != admitted[blocks_name]
            or visits != passes * m
        ):
            raise ValueError(
                f"Native contextual reference {phase} accounting mismatch."
            )
        duplicate = _native_u64(
            f"phase_ledger.{phase}.logical_duplicate_variant_decodes",
            item["logical_duplicate_variant_decodes"],
        )
        if duplicate != (passes - 1) * m:
            raise ValueError(
                f"Native contextual reference {phase} duplicate-decode mismatch."
            )
        touched = _native_u64(
            f"phase_ledger.{phase}.logical_bed_record_bytes_touched",
            item["logical_bed_record_bytes_touched"],
        )
        if visits == 0 or touched == 0 or touched % visits:
            raise ValueError(
                f"Native contextual reference {phase} logical-byte evidence is invalid."
            )
        observed_bytes_per_record.add(touched // visits)
    if len(observed_bytes_per_record) != 1:
        raise ValueError(
            "Native contextual reference logical BED record widths disagree."
        )
    if (
        phases["source"]["access_order"] != "retained_logical_sequential_v1"
        or phases["action"]["access_order"] != "retained_logical_sequential_v1"
        or phases["same_person"]["access_order"] != "retained_logical_sequential_v1"
    ):
        raise ValueError(
            "Native contextual reference sequential access evidence mismatch."
        )
    expected_group_access = (
        "sealed_group_indexed_plus_retained_logical_sequential_v1"
        if differential
        else "sealed_group_indexed_v1"
    )
    if grouped_phase["access_order"] != expected_group_access:
        raise ValueError(
            "Native contextual reference grouped access evidence mismatch."
        )
    if (
        _native_u64(
            "group_restricted_duplicate_variant_decodes",
            grouped_phase["group_restricted_duplicate_variant_decodes"],
        )
        != 0
        or _native_u64(
            "group_restricted_variant_coverage",
            grouped_phase["group_restricted_variant_coverage"],
        )
        != m
    ):
        raise ValueError("Native group-restricted coverage evidence mismatch.")
    if (
        admission["os_physical_read_bytes_measured"] is not False
        or admission["os_page_faults_measured"] is not False
        or admission["physical_io_evidence"] != "logical_mmap_record_touches_only_v1"
    ):
        raise ValueError("Native physical-I/O evidence claim is invalid.")

    required_true = (
        "descriptor_accounting_verified",
        "semantic_call_ledger_exact",
        "files_unchanged_at_all_checkpoints",
        "inputs_unchanged_at_all_checkpoints",
        "scratch_released",
        "scratch_released_before_publication",
        "all_large_buffers_preallocated_before_decode",
        "tracked_high_water_within_admission",
        "group_reconstruction_verified",
        "same_person_global_merge_verified",
        "same_person_cross_tile_pairs_included",
        "group_execution_permutation_coverage_verified",
        "group_execution_permutation_unique_verified",
        "group_execution_contiguous_verified",
        "logical_order_missing_identity_verified",
        "same_person_signed_output_preserved",
        "operand_fingerprints_verified",
        "runtime_thread_affinity_fingerprints_verified",
        "independent_scalar_witness_verified",
        "independent_scalar_fallback_available",
    )
    if any(diagnostics[name] is not True for name in required_true):
        raise ValueError(
            "Native contextual reference integrity evidence is incomplete."
        )
    if (
        _native_u64(
            "runtime_large_allocations_after_decode",
            diagnostics["runtime_large_allocations_after_decode"],
        )
        != 0
    ):
        raise ValueError(
            "Native contextual reference allocated large scratch after decode."
        )
    if dict(diagnostics["protected_call_counts"]) != call_counts:
        raise ValueError("Native diagnostic protected-call ledger mismatch.")
    if (
        _native_u64(
            "observed_descriptor_passes", diagnostics["observed_descriptor_passes"]
        )
        != expected_passes
    ):
        raise ValueError("Native observed descriptor passes mismatch admission.")
    if (
        _native_u64("observed_decoded_blocks", diagnostics["observed_decoded_blocks"])
        != expected_blocks
    ):
        raise ValueError("Native observed decoded blocks mismatch admission.")
    if (
        _native_u64(
            "observed_variant_record_visits",
            diagnostics["observed_variant_record_visits"],
        )
        != expected_passes * m
    ):
        raise ValueError("Native observed variant visits mismatch admission.")
    if (
        _native_u64("variant_probe_coverage", diagnostics["variant_probe_coverage"])
        != bd
    ):
        raise ValueError("Native variant-probe coverage is incomplete.")
    if (
        _native_u64(
            "group_execution_variant_visits",
            diagnostics["group_execution_variant_visits"],
        )
        != m
    ):
        raise ValueError(
            "Native group execution does not visit each variant exactly once."
        )

    if telemetry["complete_without_drop"] is not True:
        raise ValueError("Native contextual reference telemetry was dropped.")
    if dict(telemetry["protected_call_counts"]) != call_counts:
        raise ValueError("Native telemetry protected-call ledger mismatch.")
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
        raise ValueError("Native contextual reference telemetry capacity mismatch.")
    events = telemetry["events"]
    if (
        not isinstance(events, list)
        or observed_events != len(events)
        or observed_events > capacity
    ):
        raise ValueError("Native contextual reference telemetry event count mismatch.")
    event_class_counts: dict[str, int] = {}
    protected_event_counts = {name: 0 for name in _SEMANTIC_OPERATIONS}
    phase_transitions: list[str] = []
    checkpoints: list[str] = []
    scalar_witness_events = 0
    group_reconstruction_events = 0
    fault_injection_records: list[dict[str, Any]] = []
    fault_detection_records: list[tuple[str, str]] = []
    recovery_chain_records: list[tuple[str, str]] = []
    protected_anchor_counts: Counter[tuple[str, str]] = Counter()
    verification_anchor_counts: Counter[tuple[str, str]] = Counter()
    event_owner: tuple[int, int] | None = None
    same_person_role_counts = {
        "protected_call": {"tile": 0, "global_merge": 0},
        "semantic_verification": {"tile": 0, "global_merge": 0},
    }
    direct_placement_counts = {
        "protected_call": {"action_scaled": 0, "genotype_scaled": 0},
        "semantic_verification": {"action_scaled": 0, "genotype_scaled": 0},
    }
    class_resolutions = {
        "phase_transition": {
            "source_running",
            "action_running",
            "gram_running",
            "group_running",
            "same_person_running",
        },
        "mutation_checkpoint": {
            "post_seal",
            "post_admission",
            "post_source",
            "post_action",
            "post_gram",
            "post_group",
            "post_same_person",
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
            "semantic_group_reconstruction",
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
            "group_reconstruction_mismatch",
        },
        "repair": {"verified_retry"},
        "retry": {"deterministic_tiled_retry"},
        "trusted_fallback": {"one_thread_scalar_fp64"},
        "semantic_verification": {
            "scalar_witness_agreement",
            "group_reconstruction_verified",
        },
        "scratch_release": {"execution_arena_released"},
        "publication": {"compact_complete_statistics_ready"},
    }
    coordinate_limits = {
        "resident": bt,
        "variant": m,
        "annotation": k,
        "context": q,
        "action": c,
        "group": j,
    }
    for sequence, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
            raise ValueError(
                "Native contextual reference telemetry event schema mismatch."
            )
        event_class = event["event_class"]
        if event_class not in class_resolutions:
            raise ValueError(
                "Native contextual reference telemetry event class is unknown."
            )
        event_class_counts[event_class] = event_class_counts.get(event_class, 0) + 1
        operation = event["operation"]
        if operation not in {*_SEMANTIC_OPERATIONS, "none"}:
            raise ValueError(
                "Native contextual reference telemetry operation is unknown."
            )
        if event["resolution"] not in class_resolutions[event_class]:
            raise ValueError(
                "Native contextual reference telemetry resolution is invalid."
            )
        if not isinstance(event["phase"], str) or event["phase"] not in {
            "admission",
            "source",
            "action",
            "gram",
            "group",
            "same_person",
            "finalization",
            "publication",
        }:
            raise ValueError("Native contextual reference telemetry phase is invalid.")
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
                "Native contextual reference semantic classification is invalid."
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
                "Native contextual reference semantic operation classification "
                "is inconsistent."
            )
        lifecycle_phase_by_resolution = {
            "post_seal": "admission",
            "post_admission": "admission",
            "source_running": "source",
            "post_source": "source",
            "action_running": "action",
            "post_action": "action",
            "gram_running": "gram",
            "post_gram": "gram",
            "group_running": "group",
            "post_group": "group",
            "group_reconstruction_verified": "group",
            "same_person_running": "same_person",
            "post_same_person": "same_person",
            "pre_finalization": "finalization",
            "execution_arena_released": "finalization",
            "pre_publication": "finalization",
            "compact_complete_statistics_ready": "publication",
        }
        expected_event_phase = (
            ("action" if operation == "full_target_nn" else semantic_phase)
            if operation != "none"
            else lifecycle_phase_by_resolution.get(event["resolution"])
        )
        if expected_event_phase is None or event["phase"] != expected_event_phase:
            raise ValueError("Native contextual reference event phase is inconsistent.")
        canonical_begin = _native_u64(
            "telemetry.events.canonical_begin", event["canonical_begin"]
        )
        canonical_end = _native_u64(
            "telemetry.events.canonical_end", event["canonical_end"]
        )
        if canonical_begin > canonical_end:
            raise ValueError("Native contextual reference canonical range is invalid.")
        variant_coordinate_mode = event["variant_coordinate_mode"]
        variant_range_is_exact = event["variant_range_is_exact"]
        variant_membership_count = _native_u64(
            "telemetry.events.variant_membership_count",
            event["variant_membership_count"],
        )
        variant_membership_sha256 = event["variant_membership_sha256"]
        if not isinstance(variant_range_is_exact, bool):
            raise ValueError(
                "Native contextual reference variant exactness claim is invalid."
            )
        if variant_coordinate_mode == "logical_half_open_range_v1":
            if (
                variant_range_is_exact is not True
                or variant_membership_count != 0
                or variant_membership_sha256 != ""
            ):
                raise ValueError(
                    "Native contextual reference logical variant coordinate is invalid."
                )
        elif variant_coordinate_mode == "exact_membership_sha256_v1":
            if variant_membership_count == 0:
                raise ValueError(
                    "Native contextual reference exact variant membership is empty."
                )
            variant_membership_sha256 = _sha256(
                "telemetry.events.variant_membership_sha256",
                variant_membership_sha256,
            )
        else:
            raise ValueError(
                "Native contextual reference variant coordinate mode is unknown."
            )
        exact_variant_operation = operation in {
            "group_target_nn",
            "group_cross_gram_tn",
        } or (
            operation == "full_target_nn"
            and annotation_mode == AnnotationMode.STRICT_DISJOINT_BINARY_V1.value
        )
        expected_variant_coordinate_mode = (
            "exact_membership_sha256_v1"
            if exact_variant_operation
            else "logical_half_open_range_v1"
        )
        if variant_coordinate_mode != expected_variant_coordinate_mode:
            raise ValueError(
                "Native contextual reference operation/variant mode is inconsistent."
            )
        process_id = _native_u64("telemetry.events.process_id", event["process_id"])
        thread_id = _native_u64("telemetry.events.thread_id", event["thread_id"])
        observed_owner = (process_id, thread_id)
        if event_owner is None:
            event_owner = observed_owner
        if (
            _native_u64("telemetry.events.sequence", event["sequence"]) != sequence
            or process_id == 0
            or thread_id == 0
            or observed_owner != event_owner
        ):
            raise ValueError(
                "Native contextual reference event sequence/ownership is invalid."
            )
        _native_u64("telemetry.events.elapsed_ns", event["elapsed_ns"])
        fingerprints = {
            name: _native_fnv64(f"telemetry.events.{name}", event[name])
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
            raise ValueError(
                "Native contextual reference integrity flags must be boolean."
            )
        attempt = _native_u64("telemetry.events.attempt", event["attempt"])
        if attempt > 3 or not isinstance(event["transpose_left"], bool):
            raise ValueError(
                "Native contextual reference telemetry attempt/type is invalid."
            )
        numeric_shape = {
            name: _native_u64(f"telemetry.events.{name}", event[name])
            for name in (
                "rows",
                "columns",
                "reduction",
                "left_stride",
                "right_stride",
                "output_stride",
            )
        }
        probe_limit = bd if operation in _SEMANTIC_OPERATIONS[10:14] else bt
        coordinate_limits_with_probe = {**coordinate_limits, "probe": probe_limit}
        coordinates: dict[str, tuple[int, int]] = {}
        for axis, limit in coordinate_limits_with_probe.items():
            begin = _native_u64(
                f"telemetry.events.{axis}_begin", event[f"{axis}_begin"]
            )
            end = _native_u64(f"telemetry.events.{axis}_end", event[f"{axis}_end"])
            if begin > end or end > limit:
                raise ValueError(
                    "Native contextual reference telemetry range is invalid."
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
                    "Native contextual reference exact variant envelope is invalid."
                )
        if (
            operation == "direct_grouped_tn"
            and (
                canonical_begin,
                canonical_end,
            )
            != coordinates["variant"]
        ):
            raise ValueError(
                "Native contextual reference direct-grouped coordinate is invalid."
            )
        if (
            operation == "same_person_gram_tn"
            and (
                canonical_begin,
                canonical_end,
            )
            != coordinates["probe"]
        ):
            raise ValueError(
                "Native contextual reference same-person Gram coordinate is invalid."
            )
        if operation == "same_person_gram_tn" and (
            (semantic_role == "global_merge" and coordinates["probe"] != (0, bd))
            or coordinates["probe"][0] == coordinates["probe"][1]
        ):
            raise ValueError(
                "Native contextual reference same-person Gram role range is invalid."
            )
        expected_anchor = (
            f"{operation}|resident={coordinates['resident'][0]}:{coordinates['resident'][1]}"
            f"|probe={coordinates['probe'][0]}:{coordinates['probe'][1]}"
            f"|variant={coordinates['variant'][0]}:{coordinates['variant'][1]}"
            f"|annotation={coordinates['annotation'][0]}:{coordinates['annotation'][1]}"
            f"|context={coordinates['context'][0]}:{coordinates['context'][1]}"
            f"|action={coordinates['action'][0]}:{coordinates['action'][1]}"
        )
        if operation in _SEMANTIC_OPERATIONS[7:10]:
            expected_anchor += (
                f"|group={coordinates['group'][0]}:{coordinates['group'][1]}"
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
            raise ValueError("Native contextual reference semantic anchor is invalid.")
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
            if operation not in _SEMANTIC_OPERATIONS[:14] or any(
                numeric_shape[name] == 0 for name in ("rows", "columns", "reduction")
            ):
                raise ValueError(
                    "Native contextual reference protected-attempt event is invalid."
                )
            if (
                numeric_shape["output_stride"] < numeric_shape["rows"]
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
                    "Native contextual reference protected-call strides are invalid."
                )
            if any(value == 0 for value in fingerprints.values()):
                raise ValueError(
                    "Native contextual reference protected evidence is incomplete."
                )
            if diagnostic_geometry_event:
                if (
                    event["serialized_entry_verified"] is not True
                    or event["deterministic_non_vendor_backend"] is not True
                ):
                    raise ValueError(
                        "Native contextual reference fault evidence is incomplete."
                    )
            elif any(event[name] is not True for name in integrity_flags):
                raise ValueError(
                    "Native contextual reference protected evidence is incomplete."
                )
        if event_class == "protected_call":
            if attempt != 1:
                raise ValueError(
                    "Native contextual reference primary attempt is invalid."
                )
            protected_event_counts[operation] += 1
            protected_anchor_counts[(operation, event["semantic_anchor"])] += 1
            if operation == "same_person_gram_tn":
                same_person_role_counts[event_class][semantic_role] += 1
            elif operation == "direct_grouped_tn":
                direct_placement_counts[event_class][semantic_placement] += 1
        elif event_class in {"retry", "repair"} and attempt != 2:
            raise ValueError(
                "Native contextual reference retry/repair attempt is invalid."
            )
        elif event_class == "trusted_fallback" and attempt != 3:
            raise ValueError("Native contextual reference fallback attempt is invalid.")
        elif event_class == "fault_injection":
            fault_injection_records.append(dict(event))
            if event["resolution"] == "semantic_group_reconstruction":
                if operation != "none" or attempt != 0:
                    raise ValueError(
                        "Native contextual reference semantic fault injection is invalid."
                    )
            elif operation == "none" or attempt != 1:
                raise ValueError(
                    "Native contextual reference protected fault injection is invalid."
                )
        elif event_class == "fault_detection":
            fault_detection_records.append((operation, event["semantic_anchor"]))
            expected_detection_attempt = {
                "retry_scalar_witness_mismatch": 2,
                "fallback_scalar_witness_mismatch": 3,
                "injected_trusted_fallback_failure": 3,
                "group_reconstruction_mismatch": 0,
            }.get(event["resolution"], 1)
            if attempt != expected_detection_attempt or (
                event["resolution"] == "group_reconstruction_mismatch"
            ) != (operation == "none"):
                raise ValueError(
                    "Native contextual reference fault detection is invalid."
                )
        elif not protected_geometry_event and any(numeric_shape.values()):
            raise ValueError(
                "Native contextual reference lifecycle event carries GEMM dimensions."
            )
        elif (
            not protected_geometry_event
            and event_class not in {"semantic_verification"}
            and attempt != 0
        ):
            raise ValueError(
                "Native contextual reference lifecycle attempt is invalid."
            )
        if event_class in {"retry", "repair", "trusted_fallback"}:
            recovery_chain_records.append((operation, event["semantic_anchor"]))
        if event_class == "semantic_verification":
            if event["resolution"] == "scalar_witness_agreement":
                if operation == "none" or attempt not in (1, 2, 3):
                    raise ValueError("Native scalar-witness event is invalid.")
                scalar_witness_events += 1
                verification_anchor_counts[(operation, event["semantic_anchor"])] += 1
                if operation == "same_person_gram_tn":
                    same_person_role_counts[event_class][semantic_role] += 1
                elif operation == "direct_grouped_tn":
                    direct_placement_counts[event_class][semantic_placement] += 1
            else:
                if operation != "none" or attempt != 0:
                    raise ValueError("Native group-reconstruction event is invalid.")
                group_reconstruction_events += 1
        if event_class == "phase_transition":
            phase_transitions.append(event["resolution"])
        elif event_class == "mutation_checkpoint":
            checkpoints.append(event["resolution"])
    if protected_event_counts != call_counts:
        raise ValueError(
            "Native protected-call events disagree with the admitted ledger."
        )
    if verification_anchor_counts != protected_anchor_counts:
        raise ValueError(
            "Native protected-call/scalar-verification attribution is inconsistent."
        )
    same_person_calls = call_counts["same_person_gram_tn"]
    expected_same_person_roles = {
        "tile": same_person_calls - 1,
        "global_merge": 1,
    }
    if any(
        counts != expected_same_person_roles
        for counts in same_person_role_counts.values()
    ):
        raise ValueError(
            "Native same-person tile/global-merge event ledger is inconsistent."
        )
    direct_calls = call_counts["direct_grouped_tn"]
    expected_direct_placements = (
        {
            "action_scaled": direct_calls // 2,
            "genotype_scaled": direct_calls // 2,
        }
        if differential
        else {"action_scaled": 0, "genotype_scaled": 0}
    )
    if (differential and direct_calls % 2 != 0) or any(
        counts != expected_direct_placements
        for counts in direct_placement_counts.values()
    ):
        raise ValueError(
            "Native direct-grouped placement event ledger is inconsistent."
        )
    if phase_transitions != [
        "source_running",
        "action_running",
        "gram_running",
        "group_running",
        "same_person_running",
    ]:
        raise ValueError(
            "Native contextual reference phase-transition sequence is incomplete."
        )
    if checkpoints != [
        "post_seal",
        "post_admission",
        "post_source",
        "post_action",
        "post_gram",
        "post_group",
        "post_same_person",
        "pre_finalization",
        "pre_publication",
    ]:
        raise ValueError(
            "Native contextual reference checkpoint sequence is incomplete."
        )
    if (
        event_class_counts.get("scratch_release", 0) != 1
        or event_class_counts.get("publication", 0) != 1
        or events[-1]["event_class"] != "publication"
        or scalar_witness_events != total_calls
        or group_reconstruction_events != 1
        or event_class_counts.get("fault_injection", 0)
        != _native_u64("telemetry.injection_count", telemetry["injection_count"])
        or event_class_counts.get("repair", 0)
        != _native_u64("telemetry.repair_count", telemetry["repair_count"])
        or event_class_counts.get("retry", 0)
        != _native_u64("telemetry.retry_count", telemetry["retry_count"])
        or event_class_counts.get("trusted_fallback", 0)
        != _native_u64("telemetry.fallback_count", telemetry["fallback_count"])
    ):
        raise ValueError("Native contextual reference telemetry lifecycle mismatch.")
    for name in ("fault_operation", "fault_mode", "fault_semantic_anchor"):
        if not isinstance(telemetry[name], str):
            raise ValueError("Native contextual reference fault telemetry is invalid.")
    injection_count = _native_u64(
        "telemetry.injection_count", telemetry["injection_count"]
    )
    repair_count = _native_u64("telemetry.repair_count", telemetry["repair_count"])
    retry_count = _native_u64("telemetry.retry_count", telemetry["retry_count"])
    fallback_count = _native_u64(
        "telemetry.fallback_count", telemetry["fallback_count"]
    )
    if injection_count == 0:
        if (
            telemetry["fault_operation"]
            or telemetry["fault_mode"] != "none"
            or telemetry["fault_semantic_anchor"]
            or repair_count
            or retry_count
            or fallback_count
            or event_class_counts.get("fault_detection", 0)
        ):
            raise ValueError("Native no-fault telemetry is internally inconsistent.")
    else:
        mode = telemetry["fault_mode"]
        expected_recovery = {
            "one_shot": (1, 1, 0, 1),
            "nan": (1, 1, 0, 1),
            "inf": (1, 1, 0, 1),
            "canary": (1, 1, 0, 1),
            "repeated": (0, 1, 1, 2),
            "repair_corruption": (0, 1, 1, 2),
            "force_fallback": (0, 0, 1, 1),
        }
        if injection_count != 1 or len(fault_injection_records) != 1:
            raise ValueError("Native recovered-fault telemetry is inconsistent.")
        injected_event = fault_injection_records[0]
        validate_native_fault_selector_targets_event(
            telemetry["fault_semantic_anchor"],
            operation=telemetry["fault_operation"],
            event=injected_event,
            grouped=telemetry["fault_operation"] in _SEMANTIC_OPERATIONS[7:10],
            identity_required=telemetry["fault_operation"]
            in {"same_person_gram_tn", "direct_grouped_tn"},
            family="Native contextual reference",
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
            or telemetry["fault_operation"] not in _SEMANTIC_OPERATIONS
            or not telemetry["fault_semantic_anchor"]
            or mode not in expected_recovery
            or (
                repair_count,
                retry_count,
                fallback_count,
                event_class_counts.get("fault_detection", 0),
            )
            != expected_recovery[mode]
        ):
            raise ValueError("Native recovered-fault telemetry is inconsistent.")
    return _json_value(admission), _json_value(diagnostics), _json_value(telemetry)


def adapt_native_contextual_reference_v1(
    native_result: Mapping[str, Any],
    publication_identity: ContextualReferencePublicationIdentityV1,
) -> ContextualReferenceArtifactV1:
    """Validate and publish one complete native compact result in memory."""
    if not isinstance(native_result, Mapping):
        raise ValueError("Native contextual reference result must be a mapping.")
    if set(native_result) != _NATIVE_RESULT_KEYS:
        missing = sorted(_NATIVE_RESULT_KEYS - set(native_result))
        extra = sorted(set(native_result) - _NATIVE_RESULT_KEYS)
        raise ValueError(
            f"Native contextual reference result schema mismatch; "
            f"missing={missing}, extra={extra}."
        )
    if not isinstance(publication_identity, ContextualReferencePublicationIdentityV1):
        raise ValueError("publication_identity must use the V1 identity type.")
    if _required_native(native_result, "complete_reference_artifact") is not False:
        raise ValueError("Native result must not claim Python artifact publication.")
    if _required_native(native_result, "complete_reference_statistics") is not True:
        raise ValueError(
            "Native result does not contain complete reference statistics."
        )
    if (
        _required_native(native_result, "internal_result_kind")
        != "stage3_complete_reference_statistics_v1"
    ):
        raise ValueError("Native contextual reference result kind mismatch.")
    if (
        _required_native(native_result, "state")
        != "reference_statistics_complete_ready"
        or _required_native(native_result, "lifecycle")
        != "reference_statistics_complete_ready"
    ):
        raise ValueError("Native contextual reference result is not complete.")
    exact_policies = {
        "numeric_policy": "fp64_v1",
        "deletion_semantics": DeletionSemantics.APPROXIMATE_SUMMARY_ONLY_V1.value,
        "same_person_deletion": "reuse_full_unchanged",
        "grouped_encoding": GroupedEncodingVersion.DENSE_UNNORMALIZED_V1.value,
        "group_numerator_unit": "mass_squared_fixed_probe_raw_action_v1",
        "group_gram_raw_unit": (
            "(1/B_T)_sum_raw_cross_products_annotation_mass_unnormalized_v1"
        ),
        "group_execution_order": (
            "stable_group_contiguous_logical_index_permutation_v1"
        ),
        "file_identity_policy": _FILE_IDENTITY_POLICY_V1,
        "contextual_backend": "plink_bed_descriptor_stream_stage2_v1",
        "contextual_execution_backend": (
            "deterministic_tiled_fp64_with_scalar_witness_v1"
        ),
    }
    for name, expected in exact_policies.items():
        if _required_native(native_result, name) != expected:
            raise ValueError(f"Native contextual reference {name} policy mismatch.")
    for name in (
        "grouped_values_are_unnormalized_numerators",
        "same_person_signed_preserved",
    ):
        if _required_native(native_result, name) is not True:
            raise ValueError(f"Native contextual reference {name} evidence is false.")
    if _required_native(native_result, "sample_probe_policy") != _required_native(
        native_result, "probe_policy"
    ) or _required_native(
        native_result, "sample_probe_identity_sha256"
    ) != _required_native(
        native_result, "probe_identity_sha256"
    ):
        raise ValueError("Native sample-probe aliases disagree.")
    if _required_native(native_result, "selected_grouped_algorithm") != (
        GroupedAttributionAlgorithm.GROUP_RESTRICTED_ACTION_V1.value
    ):
        raise ValueError(
            "Stable V1 publication currently requires group-restricted attribution."
        )
    if _required_native(native_result, "direct_grouped_scaling") not in {
        "action_scaled_v1",
        "genotype_scaled_v1",
        "both_differential_v1",
    }:
        raise ValueError("Native direct grouped scaling policy mismatch.")
    ownership = _required_native(native_result, "output_ownership")
    if not isinstance(ownership, Mapping) or dict(ownership) != {
        "owns_data": True,
        "read_only": True,
    }:
        raise ValueError("Native contextual reference output ownership is invalid.")
    numa = _required_native(native_result, "numa")
    _validate_numa_evidence(numa)

    pair_q_observed = _native_array(native_result, "pair_q", dtype=np.int64, ndim=1)
    pair_r_observed = _native_array(native_result, "pair_r", dtype=np.int64, ndim=1)
    if pair_q_observed.ndim != 1 or pair_r_observed.shape != pair_q_observed.shape:
        raise ValueError("Native contextual reference pair maps are invalid.")
    q = int(
        max(
            np.max(pair_q_observed, initial=-1),
            np.max(pair_r_observed, initial=-1),
        )
        + 1
    )
    _positive_int("q", q)
    native_annotation_names = _required_native(native_result, "annotation_names")
    native_group_names = _required_native(native_result, "group_names")
    if (
        not isinstance(native_annotation_names, (list, tuple))
        or not native_annotation_names
        or any(
            not isinstance(value, str) or not value for value in native_annotation_names
        )
    ):
        raise ValueError("Native contextual reference annotation names are invalid.")
    if (
        not isinstance(native_group_names, (list, tuple))
        or not native_group_names
        or any(not isinstance(value, str) or not value for value in native_group_names)
    ):
        raise ValueError("Native contextual reference group names are invalid.")
    annotation_names = tuple(native_annotation_names)
    group_ids = tuple(native_group_names)
    if annotation_names != publication_identity.annotation_names:
        raise ValueError("Native and publication annotation names differ.")
    if group_ids != publication_identity.group_ids:
        raise ValueError("Native and publication deletion-group labels differ.")
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q))
    expected_maps = {
        "pair_q": np.asarray([entry.q for entry in components.pair_index.entries]),
        "pair_r": np.asarray([entry.r for entry in components.pair_index.entries]),
        "pair_eta": np.asarray(
            [entry.kernel_factor for entry in components.pair_index.entries]
        ),
        "component_annotation": np.asarray(
            [entry.annotation_index for entry in components.entries]
        ),
        "component_pair": np.asarray(
            [entry.pair_index for entry in components.entries]
        ),
    }
    for name, expected in expected_maps.items():
        observed = _native_array(native_result, name, dtype=np.int64, ndim=1)
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise ValueError(f"Native contextual reference {name} is not canonical.")

    observed_variant_alleles = _sha256(
        "variant_order_allele_sha256",
        _required_native(native_result, "variant_order_allele_sha256"),
    )
    if observed_variant_alleles != publication_identity.variant_order_allele_sha256:
        raise ValueError("Native and publication variant/allele identities differ.")
    if _positive_int("q", _required_native(native_result, "q")) != q:
        raise ValueError("Native Q and canonical pair map disagree.")
    if _required_native(native_result, "grouped_attribution_algorithm") != (
        _required_native(native_result, "selected_grouped_algorithm")
    ):
        raise ValueError("Native grouped-algorithm aliases disagree.")
    if _required_native(native_result, "group_execution_permutation_sha256") != (
        _required_native(native_result, "group_execution_order_sha256")
    ):
        raise ValueError("Native group execution-order aliases disagree.")

    retained_variant_order_sha256 = _sha256(
        "retained_variant_order_sha256",
        _required_native(native_result, "retained_variant_order_sha256"),
    )
    scale_string_fields = (
        "allele_orientation",
        "allele_coding",
        "centering_source",
        "centering_formula",
        "scaling_formula",
        "missing_imputation",
        "ploidy_policy",
        "affine_mean_sha256",
        "affine_inverse_scale_sha256",
    )
    if any(
        not isinstance(_required_native(native_result, name), str)
        for name in scale_string_fields
    ):
        raise ValueError("Native contextual reference scale fields must be strings.")
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
    if _required_native(native_result, "scale_plan_sha256") != scale_plan.digest:
        raise ValueError("Native contextual reference scale-plan digest mismatch.")
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
        "sample_probe_identity_sha256": (
            publication_identity.sample_probe_identity_sha256
        ),
        "variant_probe_identity_sha256": (
            publication_identity.variant_probe_identity_sha256
        ),
    }
    for name, expected in authority_fields.items():
        observed = _sha256(name, _required_native(native_result, name))
        if observed != expected:
            raise ValueError(f"Native and publication {name} identities differ.")
    if (
        _required_native(native_result, "sample_probe_policy")
        != publication_identity.sample_probe_policy
        or _required_native(native_result, "variant_probe_policy")
        != publication_identity.variant_probe_policy
    ):
        raise ValueError("Native and publication probe policies differ.")
    backend_version = _positive_int(
        "contextual_backend_version",
        _required_native(native_result, "contextual_backend_version"),
    )
    if backend_version != 2:
        raise ValueError("Native contextual reference backend version is unsupported.")
    build_id = _required_native(native_result, "contextual_build_id")
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("Native contextual reference build identity is invalid.")
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
        _REFERENCE_SCIENTIFIC_ARRAY_KEYS,
    )
    phase_evidence_sha256 = _validate_phase_evidence(
        _required_native(native_result, "phase_evidence_sha256")
    )

    native_arrays = {
        "gram": _native_array(native_result, "gram", dtype=np.float64, ndim=2),
        "same_person": _native_array(
            native_result, "same_person", dtype=np.float64, ndim=2
        ),
        "group_gram_unnormalized_num": _native_array(
            native_result,
            "group_gram_unnormalized_num",
            dtype=np.float64,
            ndim=3,
        ),
        "annotation_masses": _native_array(
            native_result, "annotation_masses", dtype=np.float64, ndim=1
        ),
        "group_annotation_masses": _native_array(
            native_result,
            "group_annotation_masses",
            dtype=np.float64,
            ndim=2,
        ),
        "group_variant_counts": _native_array(
            native_result, "group_variant_counts", dtype=np.int64, ndim=1
        ),
    }
    raw_gram = _native_array(
        native_result, "raw_gram_numerator", dtype=np.float64, ndim=2
    )
    scientific_values = {
        **native_arrays,
        "raw_gram_numerator": raw_gram,
        **expected_maps,
    }
    for name, value in scientific_values.items():
        if scientific_array_sha256[name] != array_sha256(value):
            raise ValueError(
                f"Native contextual reference scientific digest mismatch for {name}."
            )
    arrays = {
        name: owned_readonly_array(value, dtype=value.dtype)
        for name, value in native_arrays.items()
    }
    if _sha256(
        "same_person_sha256", _required_native(native_result, "same_person_sha256")
    ) != array_sha256(arrays["same_person"]):
        raise ValueError("Native same-person typed-array digest mismatch.")
    if _sha256(
        "group_gram_unnormalized_num_sha256",
        _required_native(native_result, "group_gram_unnormalized_num_sha256"),
    ) != array_sha256(arrays["group_gram_unnormalized_num"]):
        raise ValueError("Native grouped-numerator typed-array digest mismatch.")
    component_annotation = np.asarray(
        expected_maps["component_annotation"], dtype=np.int64
    )
    component_masses = arrays["annotation_masses"][component_annotation]
    expected_raw_gram = arrays["gram"] * np.outer(component_masses, component_masses)
    if (
        raw_gram.shape != expected_raw_gram.shape
        or _relative_max(raw_gram, expected_raw_gram) > 1.0e-10
    ):
        raise ValueError("Native raw and normalized Gram units disagree.")

    n_variants = _positive_int(
        "n_variants", _required_native(native_result, "n_variants")
    )
    file_content_identity = _validate_file_content_identity(
        _required_native(native_result, "file_content_identity"),
        n_variants=n_variants,
    )
    n_reference = _positive_int(
        "reference_n", _required_native(native_result, "reference_n"), minimum=2
    )
    sample_probe_count = _positive_int(
        "sample_probe_count", _required_native(native_result, "sample_probe_count")
    )
    variant_probe_count = _positive_int(
        "variant_probe_count",
        _required_native(native_result, "variant_probe_count"),
        minimum=2,
    )
    annotation_mode = AnnotationMode(
        _required_native(native_result, "annotation_mode")
    ).value
    admission, diagnostics, telemetry = _validate_native_execution_evidence(
        _required_native(native_result, "admission"),
        _required_native(native_result, "diagnostics"),
        _required_native(native_result, "telemetry"),
        differential=_required_native(native_result, "grouped_differential_enabled"),
        annotation_mode=annotation_mode,
        n=n_reference,
        m=n_variants,
        q=q,
        k=len(annotation_names),
        c=len(components),
        j=len(group_ids),
        bt=sample_probe_count,
        bd=variant_probe_count,
    )
    identity = publication_identity.to_dict()
    identity.update(
        {
            "source_tree_sha256": source_tree_sha256,
            "sealed_plan_sha256": _sha256(
                "sealed_plan_sha256",
                _required_native(native_result, "sealed_plan_sha256"),
            ),
        }
    )
    maps = {
        "pair_map": components.pair_index.to_dict(),
        "pair_map_sha256": components.pair_index.digest,
        "component_map": components.to_dict(),
        "component_map_sha256": components.digest,
        "group_ids": list(group_ids),
        "annotation_map_sha256": _sha256(
            "annotation_map_sha256",
            _required_native(native_result, "annotation_map_sha256"),
        ),
        "group_map_sha256": _sha256(
            "group_map_sha256", _required_native(native_result, "group_map_sha256")
        ),
        "group_execution_permutation_sha256": _sha256(
            "group_execution_permutation_sha256",
            _required_native(native_result, "group_execution_order_sha256"),
        ),
        "annotation_names_sha256": _name_map_digest("annotation", annotation_names),
        "group_names_sha256": _name_map_digest("deletion_group", group_ids),
    }
    schema = ContextSchemaIdentityV1(
        artifact_family=ArtifactFamily.REFERENCE,
        logical_schema_version=LogicalSchemaVersion.REFERENCE_V1,
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
        "magic": CONTEXTUAL_REFERENCE_V1_MAGIC,
        **schema.to_dict(),
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "scientific_policy": dict(_SCIENTIFIC_POLICY),
        "dimensions": {
            "N_reference": n_reference,
            "M": n_variants,
            "Q": q,
            "K": len(annotation_names),
            "P_g": len(components.pair_index),
            "C": len(components),
            "J": len(group_ids),
            "B_T": sample_probe_count,
            "B_D": variant_probe_count,
            "residual_rank": _positive_int(
                "residual_rank",
                _required_native(native_result, "residual_rank"),
                minimum=1,
            ),
        },
        "identity": identity,
        "genotype_scale_plan": scale_plan.to_dict(),
        "genotype_scale_plan_sha256": scale_plan.digest,
        "probes": {
            "sample": {
                "policy": str(_required_native(native_result, "sample_probe_policy")),
                "count": sample_probe_count,
                "identity_sha256": _sha256(
                    "sample_probe_identity_sha256",
                    _required_native(native_result, "sample_probe_identity_sha256"),
                ),
            },
            "variant": {
                "policy": str(_required_native(native_result, "variant_probe_policy")),
                "count": _positive_int(
                    "variant_probe_count",
                    _required_native(native_result, "variant_probe_count"),
                    minimum=2,
                ),
                "identity_sha256": _sha256(
                    "variant_probe_identity_sha256",
                    _required_native(native_result, "variant_probe_identity_sha256"),
                ),
            },
        },
        "execution": {
            "annotation_mode": annotation_mode,
            "annotation_output_contract": dict(
                annotation_output_contract(AnnotationMode(annotation_mode))
            ),
            "numeric_policy": "fp64_v1",
            "grouped_attribution_algorithm": str(
                _required_native(native_result, "selected_grouped_algorithm")
            ),
            "direct_grouped_scaling": str(
                _required_native(native_result, "direct_grouped_scaling")
            ),
            "grouped_differential_enabled": bool(
                _required_native(native_result, "grouped_differential_enabled")
            ),
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
            "output_ownership": _json_value(
                _required_native(native_result, "output_ownership")
            ),
            "numa": _json_value(_required_native(native_result, "numa")),
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
    return ContextualReferenceArtifactV1(
        manifest=manifest,
        component_index=components,
        scale_plan=scale_plan,
        group_ids=group_ids,
        **arrays,
    )


def run_contextual_reference_v1(
    executor: Any,
    publication_identity: ContextualReferencePublicationIdentityV1,
) -> ContextualReferenceArtifactV1:
    """Make exactly one native ``run`` call and publish its compact result."""
    run = getattr(executor, "run", None)
    if run is None or not callable(run):
        raise ValueError("executor must expose a callable run() method.")
    return adapt_native_contextual_reference_v1(run(), publication_identity)


def write_contextual_reference_v1(
    artifact: ContextualReferenceArtifactV1, output: str | Path
) -> Path:
    """Atomically publish a single strict V1 NPZ container."""
    if not isinstance(artifact, ContextualReferenceArtifactV1):
        raise ValueError("Only ContextualReferenceArtifactV1 can use the V1 writer.")
    _validate_artifact(artifact, _json_value(artifact.manifest))
    manifest_json, manifest_sha256, arrays = _preflight_stable_npz_members(
        manifest_json=canonical_json(artifact.manifest),
        manifest_sha256=artifact.manifest_sha256,
        arrays={name: getattr(artifact, name) for name in _ARRAY_NAMES},
        family="Contextual reference V1",
    )
    path = Path(output)
    if path.suffix != ".npz" or not path.name.endswith(CONTEXTUAL_REFERENCE_V1_SUFFIX):
        path = Path(str(path) + CONTEXTUAL_REFERENCE_V1_SUFFIX)
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
            family="Contextual reference V1",
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


def load_contextual_reference_v1(path: str | Path) -> ContextualReferenceArtifactV1:
    """Load only the stable V1 family and fail before returning any array."""
    source = Path(path)
    if not source.name.endswith(CONTEXTUAL_REFERENCE_V1_SUFFIX):
        raise ValueError("Contextual reference V1 loader requires the V1 suffix.")
    try:
        with StableNpzReader(
            source,
            family="Contextual reference V1",
            maximum_members=len(_FILE_KEYS),
        ) as archive:
            manifest = _strict_json_loads(
                archive.read_text_scalar(
                    "manifest_json", maximum_bytes=16 * 1024 * 1024
                )
            )
            digest = archive.read_text_scalar("manifest_sha256", maximum_bytes=1024)
            if _sha256("manifest_sha256", digest) != canonical_sha256(manifest):
                raise ValueError("Contextual reference V1 manifest SHA-256 mismatch.")
            components, scale_plan, group_ids = _validate_manifest_before_arrays(
                manifest
            )
            dimensions = manifest.get("dimensions")
            expected_shapes = {
                "gram": (dimensions.get("C"), dimensions.get("C")),
                "same_person": (dimensions.get("C"), dimensions.get("C")),
                "group_gram_unnormalized_num": (
                    dimensions.get("J"),
                    dimensions.get("C"),
                    dimensions.get("C"),
                ),
                "annotation_masses": (dimensions.get("K"),),
                "group_annotation_masses": (dimensions.get("J"), dimensions.get("K")),
                "group_variant_counts": (dimensions.get("J"),),
            }
            array_specs = {}
            for name in _ARRAY_NAMES:
                expected_dtype = np.dtype(
                    np.int64 if name == "group_variant_counts" else np.float64
                ).newbyteorder("<")
                array_specs[name] = (expected_dtype, expected_shapes[name])
            archive.preflight_arrays(array_specs)
            arrays = {name: archive.load_array(name) for name in _ARRAY_NAMES}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(
            "Contextual reference V1"
        ):
            raise
        raise ValueError("Not a valid contextual reference V1 artifact.") from exc
    return ContextualReferenceArtifactV1(
        manifest=manifest,
        component_index=components,
        scale_plan=scale_plan,
        group_ids=group_ids,
        **arrays,
    )


__all__ = [
    "CONTEXTUAL_REFERENCE_V1_MAGIC",
    "CONTEXTUAL_REFERENCE_V1_SUFFIX",
    "ContextualReferenceArtifactV1",
    "ContextualReferencePublicationIdentityV1",
    "adapt_native_contextual_reference_v1",
    "contextual_variant_order_allele_sha256_v1",
    "load_contextual_reference_v1",
    "reference_moments_after_deleting_groups_v1",
    "run_contextual_reference_v1",
    "write_contextual_reference_v1",
]
