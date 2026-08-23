"""Strict Stage 6 performance records and a measurement-only plan selector.

This module is intentionally independent of the native executor.  It records
what was measured, provides checked first-order planning estimates, and can
select among fully accepted records.  It does not qualify an implementation,
run benchmarks, or change an execution default.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .spec import canonical_sha256, freeze_context_mapping


CONTEXTUAL_TABLA_DIMENSIONS_V1_SCHEMA = "contextual_tabla_dimensions_v1"
CONTEXTUAL_TABLA_PLAN_V1_SCHEMA = "contextual_tabla_plan_v1"
CONTEXTUAL_PERFORMANCE_EVIDENCE_V1_SCHEMA = "contextual_performance_evidence_v1"
CONTEXTUAL_TABLA_ACCEPTANCE_V1_SCHEMA = "contextual_tabla_acceptance_v1"
CONTEXTUAL_TABLA_BENCHMARK_RECORD_V1_SCHEMA = "contextual_tabla_benchmark_record_v1"
CONTEXTUAL_TABLA_ESTIMATE_V1_SCHEMA = "contextual_tabla_estimate_v1"
CONTEXTUAL_TABLA_SELECTION_V1_SCHEMA = "contextual_tabla_selection_v1"
CONTEXTUAL_TABLA_CACHE_KEY_V1_MAGIC = "SUMMIT_CONTEXTUAL_TABLA_CACHE_KEY_V1"
CONTEXTUAL_TABLA_RECORD_ID_V1_MAGIC = "SUMMIT_CONTEXTUAL_TABLA_RECORD_ID_V1"
CONTEXTUAL_FIXED_PROBE_DISCREPANCY_V1_MAGIC = (
    "SUMMIT_CONTEXTUAL_FIXED_PROBE_DISCREPANCY_V1"
)

MAX_U128 = (1 << 128) - 1
DEFAULT_MATERIAL_SPEEDUP = 1.05

EVIDENCE_KINDS_V1 = frozenset({"measured", "modeled", "unavailable"})
TERMINAL_STATUSES_V1 = frozenset({"accepted", "rejected", "incomplete", "failed"})
GROUPED_ALGORITHMS_V1 = frozenset(
    {"group_restricted_action_v1", "direct_grouped_tn_v1"}
)
DIRECT_GROUPED_SCALING_POLICIES_V1 = frozenset(
    {"not_applicable", "action_scaled_v1", "genotype_scaled_v1"}
)
GROUP_EXECUTION_ORDERS_V1 = frozenset(
    {
        "contiguous_sealed_v1",
        "indexed_group_major_v1",
        "repeated_sequential_scan_v1",
    }
)
ANNOTATION_MODES_V1 = frozenset(
    {"strict_disjoint_binary_v1", "generic_nonnegative_weights_v1"}
)
MULTIPLICATION_BACKENDS_V1 = frozenset({"protected_dense_gemm_v1", "legacy_mailman_v1"})
INTEGRITY_POLICIES_V1 = frozenset({"full_scalar_witness_v1", "algebraic_checksum_v1"})
INTEGRITY_BACKENDS_V1 = frozenset(
    {
        "deterministic_tiled_fp64_with_scalar_witness_v1",
        "deterministic_tiled_fp64_with_algebraic_checksum_v1",
    }
)
NUMA_POLICIES_V1 = frozenset(
    {
        "unbound_first_touch_v1",
        "experimental_single_socket_bound_v1",
        "experimental_two_socket_split_v1",
    }
)
HUGE_PAGE_POLICIES_V1 = frozenset(
    {"disabled_v1", "transparent_host_default_v1", "explicit_huge_pages_v1"}
)
NUMERIC_POLICIES_V1 = frozenset(
    {
        "fp64_v1",
        "fp32_storage_fp64_compute_v1",
        "fp32_sgemm_fp64_reduction_v1",
    }
)
UNBOUND_AFFINITY_IDENTITY_SHA256_V1 = canonical_sha256(
    {
        "numa_policy": "unbound_first_touch_v1",
        "output_numa_node": -1,
        "affinity": "unbound",
    }
)

HOST_IDENTITY_KEYS_V1 = (
    "host_name",
    "machine",
    "kernel_release",
    "cpu_model",
    "physical_core_count",
    "logical_cpu_count",
    "socket_count",
    "numa_node_count",
    "memory_bytes",
)
BACKEND_IDENTITY_KEYS_V1 = (
    "native_backend",
    "native_execution_backend",
    "blas_vendor",
    "blas_version",
    "openmp_runtime",
    "numeric_policy",
)
BUILD_IDENTITY_KEYS_V1 = (
    "source_commit",
    "source_tree_sha256",
    "native_build_provenance_sha256",
    "python_runtime_sha256",
    "dependency_runtime_sha256",
    "compiler_id",
    "compiler_version",
    "build_type",
    "sanitizer_mode",
    "effective_optimization",
    "architecture_tuning",
)

PERFORMANCE_METRIC_KEYS_V1 = (
    "end_to_end_wall_seconds",
    "phase_wall_seconds",
    "phase_cpu_seconds",
    "category_wall_seconds",
    "category_cpu_seconds",
    "logical_descriptor_passes",
    "decoded_blocks",
    "physical_record_visits",
    "physical_read_bytes",
    "random_indexed_visits",
    "duplicate_decodes",
    "variants_per_second",
    "decoded_gb_per_second",
    "protected_call_shapes",
    "protected_call_count",
    "protected_effective_gflops",
    "peak_rss_bytes",
    "admitted_peak_bytes",
    "numa_remote_bytes",
    "integrity_overhead_seconds",
    "integrity_overhead_fraction",
    "artifact_write_load_fit_seconds",
)
PERFORMANCE_METRIC_UNITS_V1 = {
    "end_to_end_wall_seconds": "seconds",
    "phase_wall_seconds": "seconds",
    "phase_cpu_seconds": "seconds",
    "category_wall_seconds": "seconds",
    "category_cpu_seconds": "seconds",
    "logical_descriptor_passes": "count",
    "decoded_blocks": "count",
    "physical_record_visits": "count",
    "physical_read_bytes": "bytes",
    "random_indexed_visits": "count",
    "duplicate_decodes": "count",
    "variants_per_second": "variants/second",
    "decoded_gb_per_second": "GB/second",
    "protected_call_shapes": "elements",
    "protected_call_count": "count",
    "protected_effective_gflops": "GFLOP/second",
    "peak_rss_bytes": "bytes",
    "admitted_peak_bytes": "bytes",
    "numa_remote_bytes": "bytes",
    "integrity_overhead_seconds": "seconds",
    "integrity_overhead_fraction": "fraction",
    "artifact_write_load_fit_seconds": "seconds",
}
REQUIRED_SELECTION_MEASURED_METRICS_V1 = frozenset(
    {
        "end_to_end_wall_seconds",
        "phase_wall_seconds",
        "phase_cpu_seconds",
        "category_wall_seconds",
        "category_cpu_seconds",
        "logical_descriptor_passes",
        "decoded_blocks",
        "physical_record_visits",
        "duplicate_decodes",
        "protected_call_shapes",
        "protected_call_count",
        "peak_rss_bytes",
        "admitted_peak_bytes",
        "integrity_overhead_seconds",
        "integrity_overhead_fraction",
        "artifact_write_load_fit_seconds",
    }
)

PERFORMANCE_PHASE_KEYS_V1 = (
    "source",
    "action",
    "gram",
    "group",
    "same_person",
    "residual_moments",
    "trait",
    "finalization",
    "publication",
    "run_total",
)
PERFORMANCE_CATEGORY_KEYS_V1 = (
    "decode",
    "projection",
    "row_scale",
    "packing",
    "reduction",
)
DESCRIPTOR_PHASE_KEYS_V1 = (
    "source",
    "action",
    "group",
    "same_person",
    "trait",
)
PROTECTED_OPERATION_KEYS_V1 = (
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
PROTECTED_SHAPE_KEYS_V1 = (
    "minimum_rows",
    "maximum_rows",
    "minimum_columns",
    "maximum_columns",
    "minimum_reduction",
    "maximum_reduction",
    "shape_histogram",
)
PROTECTED_SHAPE_HISTOGRAM_ENTRY_KEYS_V1 = (
    "transpose_left",
    "rows",
    "columns",
    "reduction",
    "left_stride",
    "right_stride",
    "output_stride",
    "calls",
)
MMAP_PHYSICAL_READ_BYTES_UNAVAILABLE_SOURCE_V1 = (
    "mmap_physical_read_bytes_unavailable_v1"
)

MEMORY_CATEGORY_KEYS_V1 = (
    "bytes_executor_permanent",
    "bytes_decode",
    "bytes_probe_panel",
    "bytes_source_rhs",
    "bytes_source_scores",
    "bytes_targets",
    "bytes_actions",
    "bytes_action_scratch",
    "bytes_group_targets",
    "bytes_group_action_tile",
    "bytes_group_cross",
    "bytes_direct_group_scaled",
    "bytes_direct_group_output",
    "bytes_group_accum",
    "bytes_same_sketch",
    "bytes_same_g",
    "bytes_same_persistent",
    "bytes_trait_features",
    "bytes_trait_scores",
    "bytes_trait_permanent",
    "bytes_trait_residual",
    "bytes_trait_scratch",
    "bytes_outputs",
    "bytes_projection",
    "bytes_integrity",
    "bytes_telemetry",
    "bytes_headroom",
)
CONSERVATIVE_TELEMETRY_EVENT_BYTES_V1 = 512


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _enum(name: str, value: Any, allowed: frozenset[str]) -> str:
    value = _nonempty(name, value)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)!r}.")
    return value


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _integer(
    name: str,
    value: Any,
    *,
    minimum: int = 0,
    maximum: int = MAX_U128,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(
            f"{name} must be an integer in the closed interval "
            f"[{minimum}, {maximum}]."
        )
    return value


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _git_commit(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a canonical lowercase 40-hex Git commit.")
    return value


def _exact_keys(name: str, value: Mapping[str, Any], expected: Sequence[str]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        raise ValueError(
            f"{name} has missing keys {missing!r} and extra keys {extra!r}."
        )


def checked_u128_sum_v1(*values: int) -> int:
    """Sum nonnegative integers using explicit unsigned-128-bit bounds."""
    total = 0
    for index, value in enumerate(values):
        item = _integer(f"values[{index}]", value)
        if total > MAX_U128 - item:
            raise OverflowError("Unsigned 128-bit addition overflow.")
        total += item
    return total


def checked_u128_product_v1(*values: int) -> int:
    """Multiply nonnegative integers using explicit unsigned-128-bit bounds."""
    product = 1
    for index, value in enumerate(values):
        item = _integer(f"values[{index}]", value)
        if item and product > MAX_U128 // item:
            raise OverflowError("Unsigned 128-bit multiplication overflow.")
        product *= item
    return product


def _ceil_div_u128(name: str, numerator: int, denominator: int) -> int:
    numerator = _integer(f"{name} numerator", numerator)
    denominator = _integer(f"{name} denominator", denominator, minimum=1)
    if numerator == 0:
        return 0
    return 1 + (numerator - 1) // denominator


def _checked_scale_basis_points(value: int, basis_points: int) -> int:
    numerator = checked_u128_product_v1(value, basis_points)
    return _ceil_div_u128("basis-point scale", numerator, 10_000)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class ContextualTablaDimensionsV1:
    """Exact science and benchmark dimensions for one candidate family."""

    n_samples: int
    n_variants: int
    context_count: int
    annotation_count: int
    pair_count: int
    component_count: int
    group_count: int
    sample_probe_count: int
    variant_probe_count: int
    trait_count: int = 0
    residual_basis_count: int = 0
    fixed_rank: int = 1

    def __post_init__(self) -> None:
        for name in (
            "n_samples",
            "n_variants",
            "annotation_count",
            "group_count",
            "sample_probe_count",
            "variant_probe_count",
        ):
            _integer(
                name,
                getattr(self, name),
                minimum=3 if name == "n_samples" else 1,
            )
        _integer("context_count", self.context_count, minimum=1, maximum=4)
        _integer("pair_count", self.pair_count, minimum=1)
        _integer("component_count", self.component_count, minimum=1)
        _integer("trait_count", self.trait_count)
        _integer("residual_basis_count", self.residual_basis_count)
        _integer("fixed_rank", self.fixed_rank)
        expected_pairs = (
            checked_u128_product_v1(self.context_count, self.context_count + 1) // 2
        )
        if self.pair_count != expected_pairs:
            raise ValueError("pair_count must equal Q*(Q+1)/2.")
        if self.component_count != checked_u128_product_v1(
            self.annotation_count, self.pair_count
        ):
            raise ValueError("component_count must equal annotation_count*pair_count.")
        if self.sample_probe_count < 2 or self.variant_probe_count < 2:
            raise ValueError(
                "Both sample and variant probe counts must be at least two."
            )
        if not 1 <= self.fixed_rank < self.n_samples:
            raise ValueError("fixed_rank must be in the native range [1, n_samples).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_TABLA_DIMENSIONS_V1_SCHEMA,
            "n_samples": self.n_samples,
            "n_variants": self.n_variants,
            "context_count": self.context_count,
            "annotation_count": self.annotation_count,
            "pair_count": self.pair_count,
            "component_count": self.component_count,
            "group_count": self.group_count,
            "sample_probe_count": self.sample_probe_count,
            "variant_probe_count": self.variant_probe_count,
            "trait_count": self.trait_count,
            "residual_basis_count": self.residual_basis_count,
            "fixed_rank": self.fixed_rank,
        }


@dataclass(frozen=True, slots=True)
class ContextualTablaPlanV1:
    """One bounded Stage 6 execution-plan candidate."""

    source_variant_block: int = 256
    target_variant_block: int = 256
    grouped_variant_block: int = 256
    same_person_variant_block: int = 256
    trait_variant_block: int = 256
    sample_probe_resident_count: int = 128
    sample_probe_tile: int = 32
    variant_probe_tile: int = 32
    action_tile: int = 8
    annotation_tile: int = 1
    source_coordinate_tile: int = 4
    context_tile: int = 4
    group_tile: int = 1
    resident_source_scores: bool = True
    resident_actions: bool = True
    grouped_attribution_algorithm: str = "group_restricted_action_v1"
    direct_grouped_scaling_policy: str = "not_applicable"
    group_execution_order: str = "contiguous_sealed_v1"
    annotation_mode: str = "strict_disjoint_binary_v1"
    multiplication_backend: str = "protected_dense_gemm_v1"
    integrity_policy: str = "full_scalar_witness_v1"
    integrity_backend: str = "deterministic_tiled_fp64_with_scalar_witness_v1"
    process_count: int = 1
    decode_threads: int = 64
    blas_threads: int = 64
    affinity_cpu_count: int = 64
    affinity_identity_sha256: str = UNBOUND_AFFINITY_IDENTITY_SHA256_V1
    numa_policy: str = "unbound_first_touch_v1"
    output_numa_node: int = -1
    huge_page_policy: str = "disabled_v1"
    allocation_reuse: bool = True
    deterministic_reductions: bool = True
    numeric_policy: str = "fp64_v1"
    reduction_copy_count: int = 1
    integrity_reserve_bytes: int = 0
    telemetry_capacity_bytes: int = 0
    fixed_headroom_bytes: int = 0
    headroom_basis_points: int = 2_000

    def __post_init__(self) -> None:
        for name in (
            "source_variant_block",
            "target_variant_block",
            "grouped_variant_block",
            "same_person_variant_block",
            "trait_variant_block",
            "sample_probe_resident_count",
            "sample_probe_tile",
            "variant_probe_tile",
            "action_tile",
            "annotation_tile",
            "source_coordinate_tile",
            "context_tile",
            "group_tile",
            "process_count",
            "decode_threads",
            "blas_threads",
            "affinity_cpu_count",
            "reduction_copy_count",
        ):
            _integer(name, getattr(self, name), minimum=1)
        for name in (
            "integrity_reserve_bytes",
            "telemetry_capacity_bytes",
            "fixed_headroom_bytes",
        ):
            _integer(name, getattr(self, name))
        _integer("headroom_basis_points", self.headroom_basis_points, maximum=100_000)
        for name in (
            "resident_source_scores",
            "resident_actions",
            "allocation_reuse",
            "deterministic_reductions",
        ):
            _boolean(name, getattr(self, name))
        _enum(
            "grouped_attribution_algorithm",
            self.grouped_attribution_algorithm,
            GROUPED_ALGORITHMS_V1,
        )
        _enum(
            "direct_grouped_scaling_policy",
            self.direct_grouped_scaling_policy,
            DIRECT_GROUPED_SCALING_POLICIES_V1,
        )
        _enum(
            "group_execution_order",
            self.group_execution_order,
            GROUP_EXECUTION_ORDERS_V1,
        )
        _enum("annotation_mode", self.annotation_mode, ANNOTATION_MODES_V1)
        _enum(
            "multiplication_backend",
            self.multiplication_backend,
            MULTIPLICATION_BACKENDS_V1,
        )
        _enum("integrity_policy", self.integrity_policy, INTEGRITY_POLICIES_V1)
        _enum("integrity_backend", self.integrity_backend, INTEGRITY_BACKENDS_V1)
        _sha256("affinity_identity_sha256", self.affinity_identity_sha256)
        _enum("numa_policy", self.numa_policy, NUMA_POLICIES_V1)
        _integer(
            "output_numa_node", self.output_numa_node, minimum=-1, maximum=1_000_000
        )
        _enum("huge_page_policy", self.huge_page_policy, HUGE_PAGE_POLICIES_V1)
        _enum("numeric_policy", self.numeric_policy, NUMERIC_POLICIES_V1)
        direct = self.grouped_attribution_algorithm == "direct_grouped_tn_v1"
        if direct == (self.direct_grouped_scaling_policy == "not_applicable"):
            raise ValueError(
                "direct_grouped_scaling_policy must be applicable exactly for "
                "direct_grouped_tn_v1."
            )
        expected_integrity_backend = {
            "full_scalar_witness_v1": (
                "deterministic_tiled_fp64_with_scalar_witness_v1"
            ),
            "algebraic_checksum_v1": (
                "deterministic_tiled_fp64_with_algebraic_checksum_v1"
            ),
        }[self.integrity_policy]
        if self.integrity_backend != expected_integrity_backend:
            raise ValueError("integrity_policy and integrity_backend disagree.")
        if self.decode_threads > self.affinity_cpu_count:
            raise ValueError("decode_threads cannot exceed affinity_cpu_count.")
        if self.blas_threads > self.affinity_cpu_count:
            raise ValueError("blas_threads cannot exceed affinity_cpu_count.")
        is_unbound = self.numa_policy == "unbound_first_touch_v1"
        if is_unbound != (self.output_numa_node == -1):
            raise ValueError(
                "unbound_first_touch_v1 requires output_numa_node=-1, while an "
                "experimental bound NUMA policy requires an explicit output node."
            )

    @property
    def production_nonqualification_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.multiplication_backend == "legacy_mailman_v1":
            reasons.append("legacy_mailman_not_production_qualified")
        if self.process_count != 1:
            reasons.append("multi_process_not_production_qualified")
        if self.numa_policy != "unbound_first_touch_v1":
            reasons.append("numa_policy_not_production_qualified")
        if self.output_numa_node != -1:
            reasons.append("output_placement_not_production_qualified")
        if self.numeric_policy != "fp64_v1":
            reasons.append("mixed_precision_not_production_qualified")
        if self.huge_page_policy != "disabled_v1":
            reasons.append("huge_page_policy_not_production_qualified")
        return tuple(reasons)

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_TABLA_PLAN_V1_SCHEMA,
            "source_variant_block": self.source_variant_block,
            "target_variant_block": self.target_variant_block,
            "grouped_variant_block": self.grouped_variant_block,
            "same_person_variant_block": self.same_person_variant_block,
            "trait_variant_block": self.trait_variant_block,
            "sample_probe_resident_count": self.sample_probe_resident_count,
            "sample_probe_tile": self.sample_probe_tile,
            "variant_probe_tile": self.variant_probe_tile,
            "action_tile": self.action_tile,
            "annotation_tile": self.annotation_tile,
            "source_coordinate_tile": self.source_coordinate_tile,
            "context_tile": self.context_tile,
            "group_tile": self.group_tile,
            "resident_source_scores": self.resident_source_scores,
            "resident_actions": self.resident_actions,
            "grouped_attribution_algorithm": self.grouped_attribution_algorithm,
            "direct_grouped_scaling_policy": self.direct_grouped_scaling_policy,
            "group_execution_order": self.group_execution_order,
            "annotation_mode": self.annotation_mode,
            "multiplication_backend": self.multiplication_backend,
            "integrity_policy": self.integrity_policy,
            "integrity_backend": self.integrity_backend,
            "process_count": self.process_count,
            "decode_threads": self.decode_threads,
            "blas_threads": self.blas_threads,
            "affinity_cpu_count": self.affinity_cpu_count,
            "affinity_identity_sha256": self.affinity_identity_sha256,
            "numa_policy": self.numa_policy,
            "output_numa_node": self.output_numa_node,
            "huge_page_policy": self.huge_page_policy,
            "allocation_reuse": self.allocation_reuse,
            "deterministic_reductions": self.deterministic_reductions,
            "numeric_policy": self.numeric_policy,
            "reduction_copy_count": self.reduction_copy_count,
            "integrity_reserve_bytes": self.integrity_reserve_bytes,
            "telemetry_capacity_bytes": self.telemetry_capacity_bytes,
            "fixed_headroom_bytes": self.fixed_headroom_bytes,
            "headroom_basis_points": self.headroom_basis_points,
        }


def _validate_evidence_value(name: str, value: Any) -> Any:
    if value is None:
        raise ValueError(f"{name} must contain finite numeric evidence.")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return _integer(name, value)
    if isinstance(value, float):
        return _finite(name, value, minimum=0.0)
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{name} mappings must not be empty.")
        result: dict[str, Any] = {}
        for key, item in value.items():
            key = _nonempty(f"{name} key", key)
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}.")
            result[key] = _validate_evidence_value(f"{name}[{key!r}]", item)
        return freeze_context_mapping(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _validate_evidence_value(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        )
    raise ValueError(f"{name} must contain only finite numeric JSON evidence.")


@dataclass(frozen=True, slots=True)
class ContextualPerformanceEvidenceV1:
    """One explicitly measured, modeled, or unavailable benchmark quantity."""

    evidence_kind: str
    value: Any
    unit: str
    source: str

    def __post_init__(self) -> None:
        _enum("evidence_kind", self.evidence_kind, EVIDENCE_KINDS_V1)
        _nonempty("unit", self.unit)
        _nonempty("source", self.source)
        if self.evidence_kind == "unavailable":
            if self.value is not None:
                raise ValueError("Unavailable evidence must have value=None.")
        else:
            object.__setattr__(
                self, "value", _validate_evidence_value("evidence value", self.value)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_PERFORMANCE_EVIDENCE_V1_SCHEMA,
            "evidence_kind": self.evidence_kind,
            "value": _thaw(self.value),
            "unit": self.unit,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ContextualTablaAcceptanceV1:
    """Exact scientific/integrity evidence required before plan selection."""

    fixed_probe_evidence_kind: str
    fixed_probe_discrepancies: Mapping[str, float]
    every_deletion_science_unchanged: bool
    rank_estimability_identical: bool
    integrity_fault_coverage_identical: bool
    descriptor_passes_explained: bool
    admitted_measured_memory_agrees: bool
    reference_trait_end_to_end_complete: bool
    artifact_round_trip_passed: bool

    def __post_init__(self) -> None:
        _enum(
            "fixed_probe_evidence_kind",
            self.fixed_probe_evidence_kind,
            EVIDENCE_KINDS_V1,
        )
        if not isinstance(self.fixed_probe_discrepancies, Mapping):
            raise ValueError("fixed_probe_discrepancies must be a mapping.")
        discrepancies: dict[str, float] = {}
        for key, value in self.fixed_probe_discrepancies.items():
            key = _nonempty("fixed-probe discrepancy key", key)
            discrepancies[key] = _finite(
                f"fixed_probe_discrepancies[{key!r}]", value, minimum=0.0
            )
        if self.fixed_probe_evidence_kind == "measured" and not discrepancies:
            raise ValueError("Measured fixed-probe discrepancies must not be empty.")
        if self.fixed_probe_evidence_kind != "measured" and discrepancies:
            raise ValueError(
                "Only measured fixed-probe evidence may contain discrepancies."
            )
        object.__setattr__(
            self, "fixed_probe_discrepancies", freeze_context_mapping(discrepancies)
        )
        for name in (
            "every_deletion_science_unchanged",
            "rank_estimability_identical",
            "integrity_fault_coverage_identical",
            "descriptor_passes_explained",
            "admitted_measured_memory_agrees",
            "reference_trait_end_to_end_complete",
            "artifact_round_trip_passed",
        ):
            _boolean(name, getattr(self, name))

    @property
    def fixed_probe_discrepancy_sha256(self) -> str | None:
        if self.fixed_probe_evidence_kind != "measured":
            return None
        return canonical_sha256(
            {
                "magic": CONTEXTUAL_FIXED_PROBE_DISCREPANCY_V1_MAGIC,
                "discrepancies": _thaw(self.fixed_probe_discrepancies),
            }
        )

    @property
    def all_gates_passed(self) -> bool:
        return self.fixed_probe_evidence_kind == "measured" and all(
            (
                self.every_deletion_science_unchanged,
                self.rank_estimability_identical,
                self.integrity_fault_coverage_identical,
                self.descriptor_passes_explained,
                self.admitted_measured_memory_agrees,
                self.reference_trait_end_to_end_complete,
                self.artifact_round_trip_passed,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_TABLA_ACCEPTANCE_V1_SCHEMA,
            "fixed_probe_evidence_kind": self.fixed_probe_evidence_kind,
            "fixed_probe_discrepancies": _thaw(self.fixed_probe_discrepancies),
            "fixed_probe_discrepancy_sha256": self.fixed_probe_discrepancy_sha256,
            "every_deletion_science_unchanged": (self.every_deletion_science_unchanged),
            "rank_estimability_identical": self.rank_estimability_identical,
            "integrity_fault_coverage_identical": (
                self.integrity_fault_coverage_identical
            ),
            "descriptor_passes_explained": self.descriptor_passes_explained,
            "admitted_measured_memory_agrees": (self.admitted_measured_memory_agrees),
            "reference_trait_end_to_end_complete": (
                self.reference_trait_end_to_end_complete
            ),
            "artifact_round_trip_passed": self.artifact_round_trip_passed,
        }


def _strict_host_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys("host_identity", value, HOST_IDENTITY_KEYS_V1)
    result = dict(value)
    for key in HOST_IDENTITY_KEYS_V1[:4]:
        result[key] = _nonempty(f"host_identity[{key!r}]", result[key])
    for key in HOST_IDENTITY_KEYS_V1[4:]:
        result[key] = _integer(f"host_identity[{key!r}]", result[key], minimum=1)
    return freeze_context_mapping(result)


def _strict_backend_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys("backend_identity", value, BACKEND_IDENTITY_KEYS_V1)
    result = dict(value)
    for key in BACKEND_IDENTITY_KEYS_V1:
        result[key] = _nonempty(f"backend_identity[{key!r}]", result[key])
    return freeze_context_mapping(result)


def _strict_build_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys("build_identity", value, BUILD_IDENTITY_KEYS_V1)
    result = dict(value)
    for key in BUILD_IDENTITY_KEYS_V1:
        if key.endswith("_sha256"):
            result[key] = _sha256(f"build_identity[{key!r}]", result[key])
        elif key == "source_commit":
            result[key] = _git_commit(f"build_identity[{key!r}]", result[key])
        else:
            result[key] = _nonempty(f"build_identity[{key!r}]", result[key])
    return freeze_context_mapping(result)


def contextual_tabla_cache_key_v1(
    *,
    host_identity: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    build_identity: Mapping[str, Any],
    dimensions: ContextualTablaDimensionsV1,
) -> str:
    """Return the canonical qualified-plan cache key for all required axes."""
    if not isinstance(dimensions, ContextualTablaDimensionsV1):
        raise ValueError("dimensions must be ContextualTablaDimensionsV1.")
    host = _strict_host_identity(host_identity)
    backend = _strict_backend_identity(backend_identity)
    build = _strict_build_identity(build_identity)
    return canonical_sha256(
        {
            "magic": CONTEXTUAL_TABLA_CACHE_KEY_V1_MAGIC,
            "host_identity": _thaw(host),
            "backend_identity": _thaw(backend),
            "build_identity": _thaw(build),
            "dimensions": dimensions.to_dict(),
        }
    )


def _validate_plan_against_dimensions(
    dimensions: ContextualTablaDimensionsV1, plan: ContextualTablaPlanV1
) -> None:
    limits = (
        (
            "sample_probe_resident_count",
            plan.sample_probe_resident_count,
            dimensions.sample_probe_count,
        ),
        ("sample_probe_tile", plan.sample_probe_tile, plan.sample_probe_resident_count),
        ("variant_probe_tile", plan.variant_probe_tile, dimensions.variant_probe_count),
        ("action_tile", plan.action_tile, dimensions.component_count),
        ("annotation_tile", plan.annotation_tile, dimensions.annotation_count),
        (
            "source_coordinate_tile",
            plan.source_coordinate_tile,
            dimensions.component_count,
        ),
        ("context_tile", plan.context_tile, dimensions.context_count),
        ("group_tile", plan.group_tile, dimensions.group_count),
        ("source_variant_block", plan.source_variant_block, dimensions.n_variants),
        ("target_variant_block", plan.target_variant_block, dimensions.n_variants),
        (
            "grouped_variant_block",
            plan.grouped_variant_block,
            dimensions.n_variants,
        ),
        (
            "same_person_variant_block",
            plan.same_person_variant_block,
            dimensions.n_variants,
        ),
        ("trait_variant_block", plan.trait_variant_block, dimensions.n_variants),
    )
    for name, value, limit in limits:
        if value > limit:
            raise ValueError(f"{name} cannot exceed its corresponding dimension.")


def _validate_metrics(
    metrics: Mapping[str, ContextualPerformanceEvidenceV1],
) -> Mapping[str, ContextualPerformanceEvidenceV1]:
    _exact_keys("metrics", metrics, PERFORMANCE_METRIC_KEYS_V1)
    result: dict[str, ContextualPerformanceEvidenceV1] = {}
    for key in PERFORMANCE_METRIC_KEYS_V1:
        evidence = metrics[key]
        if not isinstance(evidence, ContextualPerformanceEvidenceV1):
            raise ValueError(
                f"metrics[{key!r}] must be ContextualPerformanceEvidenceV1."
            )
        if evidence.unit != PERFORMANCE_METRIC_UNITS_V1[key]:
            raise ValueError(
                f"metrics[{key!r}] unit must be "
                f"{PERFORMANCE_METRIC_UNITS_V1[key]!r}."
            )
        if (
            key == "integrity_overhead_fraction"
            and evidence.evidence_kind != "unavailable"
        ):
            if (
                isinstance(evidence.value, bool)
                or not isinstance(evidence.value, (int, float))
                or float(evidence.value) > 1.0
            ):
                raise ValueError(
                    "integrity_overhead_fraction must be a scalar in [0, 1]."
                )
        if key == "physical_read_bytes" and evidence.evidence_kind == "unavailable":
            if evidence.source != MMAP_PHYSICAL_READ_BYTES_UNAVAILABLE_SOURCE_V1:
                raise ValueError(
                    "Unavailable physical_read_bytes requires the explicit mmap "
                    "capability source."
                )
        result[key] = evidence
    for key in ("phase_wall_seconds", "phase_cpu_seconds"):
        _validate_exact_numeric_mapping(
            f"metrics[{key!r}]", result[key], PERFORMANCE_PHASE_KEYS_V1
        )
    for key in ("category_wall_seconds", "category_cpu_seconds"):
        _validate_exact_numeric_mapping(
            f"metrics[{key!r}]", result[key], PERFORMANCE_CATEGORY_KEYS_V1
        )
    for key in ("logical_descriptor_passes", "decoded_blocks"):
        _validate_exact_numeric_mapping(
            f"metrics[{key!r}]",
            result[key],
            DESCRIPTOR_PHASE_KEYS_V1,
            integer=True,
        )
    _validate_exact_numeric_mapping(
        "metrics['protected_call_count']",
        result["protected_call_count"],
        PROTECTED_OPERATION_KEYS_V1,
        integer=True,
    )
    _validate_exact_numeric_mapping(
        "metrics['protected_effective_gflops']",
        result["protected_effective_gflops"],
        PROTECTED_OPERATION_KEYS_V1,
    )
    _validate_protected_shapes(result["protected_call_shapes"])
    for key in (
        "physical_record_visits",
        "physical_read_bytes",
        "random_indexed_visits",
        "duplicate_decodes",
        "peak_rss_bytes",
        "admitted_peak_bytes",
        "numa_remote_bytes",
    ):
        _validate_scalar_metric(f"metrics[{key!r}]", result[key], integer=True)
    for key in (
        "end_to_end_wall_seconds",
        "variants_per_second",
        "decoded_gb_per_second",
        "integrity_overhead_seconds",
        "integrity_overhead_fraction",
        "artifact_write_load_fit_seconds",
    ):
        _validate_scalar_metric(f"metrics[{key!r}]", result[key])
    counts = result["protected_call_count"]
    shapes = result["protected_call_shapes"]
    throughputs = result["protected_effective_gflops"]
    if counts.evidence_kind != "unavailable" and shapes.evidence_kind != "unavailable":
        for operation in PROTECTED_OPERATION_KEYS_V1:
            count = counts.value[operation]
            shape = shapes.value[operation]
            if count == 0:
                if any(shape[key] != 0 for key in PROTECTED_SHAPE_KEYS_V1[:-1]):
                    raise ValueError(
                        f"Zero-call protected operation {operation!r} must have "
                        "an all-zero shape range."
                    )
                if shape["shape_histogram"]:
                    raise ValueError(
                        f"Zero-call protected operation {operation!r} must have "
                        "an empty shape histogram."
                    )
            else:
                for minimum_key, maximum_key in (
                    ("minimum_rows", "maximum_rows"),
                    ("minimum_columns", "maximum_columns"),
                    ("minimum_reduction", "maximum_reduction"),
                ):
                    if (
                        shape[minimum_key] < 1
                        or shape[maximum_key] < shape[minimum_key]
                    ):
                        raise ValueError(
                            f"Protected operation {operation!r} has an invalid "
                            f"{minimum_key}/{maximum_key} range."
                        )
                histogram = shape["shape_histogram"]
                if not histogram:
                    raise ValueError(
                        f"Protected operation {operation!r} requires a nonempty "
                        "shape histogram."
                    )
                histogram_calls = checked_u128_sum_v1(
                    *(entry["calls"] for entry in histogram)
                )
                if histogram_calls != count:
                    raise ValueError(
                        f"Protected operation {operation!r} shape-histogram calls "
                        "do not match protected_call_count."
                    )
                for axis in ("rows", "columns", "reduction"):
                    observed = [entry[axis] for entry in histogram]
                    if shape[f"minimum_{axis}"] != min(observed) or shape[
                        f"maximum_{axis}"
                    ] != max(observed):
                        raise ValueError(
                            f"Protected operation {operation!r} {axis} range "
                            "does not match its shape histogram."
                        )
    if (
        counts.evidence_kind != "unavailable"
        and throughputs.evidence_kind != "unavailable"
    ):
        for operation in PROTECTED_OPERATION_KEYS_V1:
            if counts.value[operation] == 0 and throughputs.value[operation] != 0:
                raise ValueError(
                    f"Zero-call protected operation {operation!r} must have zero "
                    "effective throughput."
                )
    return freeze_context_mapping(result)


def _validate_scalar_metric(
    name: str, evidence: ContextualPerformanceEvidenceV1, *, integer: bool = False
) -> None:
    if evidence.evidence_kind == "unavailable":
        return
    value = evidence.value
    if integer:
        _integer(name, value)
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be scalar numeric evidence.")


def _validate_exact_numeric_mapping(
    name: str,
    evidence: ContextualPerformanceEvidenceV1,
    keys: Sequence[str],
    *,
    integer: bool = False,
) -> None:
    if evidence.evidence_kind == "unavailable":
        return
    value = evidence.value
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an exact-key mapping.")
    _exact_keys(name, value, keys)
    for key in keys:
        if integer:
            _integer(f"{name}[{key!r}]", value[key])
        elif isinstance(value[key], bool) or not isinstance(value[key], (int, float)):
            raise ValueError(f"{name}[{key!r}] must be scalar numeric evidence.")


def _validate_protected_shapes(evidence: ContextualPerformanceEvidenceV1) -> None:
    if evidence.evidence_kind == "unavailable":
        return
    value = evidence.value
    if not isinstance(value, Mapping):
        raise ValueError("metrics['protected_call_shapes'] must be a mapping.")
    _exact_keys("metrics['protected_call_shapes']", value, PROTECTED_OPERATION_KEYS_V1)
    for operation in PROTECTED_OPERATION_KEYS_V1:
        shape = value[operation]
        if not isinstance(shape, Mapping):
            raise ValueError(
                f"Protected shape {operation!r} must be an exact-key mapping."
            )
        _exact_keys(f"protected shape {operation!r}", shape, PROTECTED_SHAPE_KEYS_V1)
        for key in PROTECTED_SHAPE_KEYS_V1[:-1]:
            _integer(f"protected shape {operation!r}[{key!r}]", shape[key])
        histogram = shape["shape_histogram"]
        if not isinstance(histogram, (list, tuple)):
            raise ValueError(
                f"Protected shape {operation!r} shape_histogram must be a sequence."
            )
        identities: list[tuple[bool, int, int, int, int, int, int]] = []
        for index, entry in enumerate(histogram):
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"Protected shape {operation!r} histogram entry {index} must "
                    "be a mapping."
                )
            _exact_keys(
                f"protected shape {operation!r} histogram entry {index}",
                entry,
                PROTECTED_SHAPE_HISTOGRAM_ENTRY_KEYS_V1,
            )
            _boolean(
                f"protected shape {operation!r} histogram transpose_left",
                entry["transpose_left"],
            )
            if entry["transpose_left"] != operation.endswith("_tn"):
                raise ValueError(
                    f"Protected shape {operation!r} histogram transpose_left "
                    "disagrees with its semantic operation."
                )
            for key in PROTECTED_SHAPE_HISTOGRAM_ENTRY_KEYS_V1[1:]:
                _integer(
                    f"protected shape {operation!r} histogram {key}",
                    entry[key],
                    minimum=1,
                )
            identities.append(
                (
                    entry["transpose_left"],
                    entry["rows"],
                    entry["columns"],
                    entry["reduction"],
                    entry["left_stride"],
                    entry["right_stride"],
                    entry["output_stride"],
                )
            )
        if identities != sorted(set(identities)):
            raise ValueError(
                f"Protected shape {operation!r} histogram must be uniquely "
                "canonical ordered."
            )


def _measured_positive_scalar(
    record: "ContextualTablaBenchmarkRecordV1", metric: str
) -> float | None:
    evidence = record.metrics[metric]
    if evidence.evidence_kind != "measured":
        return None
    if isinstance(evidence.value, bool) or not isinstance(evidence.value, (int, float)):
        return None
    value = float(evidence.value)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


@dataclass(frozen=True, slots=True)
class ContextualTablaBenchmarkRecordV1:
    """Immutable, exact-key machine record for one Stage 6 benchmark run."""

    run_id: str
    host_identity: Mapping[str, Any]
    backend_identity: Mapping[str, Any]
    build_identity: Mapping[str, Any]
    cache_key: str
    dimensions: ContextualTablaDimensionsV1
    plan: ContextualTablaPlanV1
    science_identity_sha256: str
    integrity_identity_sha256: str
    rank_identity_sha256: str
    metrics: Mapping[str, ContextualPerformanceEvidenceV1]
    acceptance: ContextualTablaAcceptanceV1
    terminal_status: str
    production_qualified: bool
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    benchmark_speedup_threshold: float = DEFAULT_MATERIAL_SPEEDUP
    comparison_baseline_run_id: str | None = None
    comparison_baseline_record_sha256: str | None = None
    declared_parent_source_commit: str | None = None

    def __post_init__(self) -> None:
        _nonempty("run_id", self.run_id)
        host = _strict_host_identity(self.host_identity)
        backend = _strict_backend_identity(self.backend_identity)
        build = _strict_build_identity(self.build_identity)
        object.__setattr__(self, "host_identity", host)
        object.__setattr__(self, "backend_identity", backend)
        object.__setattr__(self, "build_identity", build)
        _sha256("cache_key", self.cache_key)
        if not isinstance(self.dimensions, ContextualTablaDimensionsV1):
            raise ValueError("dimensions must be ContextualTablaDimensionsV1.")
        if not isinstance(self.plan, ContextualTablaPlanV1):
            raise ValueError("plan must be ContextualTablaPlanV1.")
        _validate_plan_against_dimensions(self.dimensions, self.plan)
        expected_cache_key = contextual_tabla_cache_key_v1(
            host_identity=host,
            backend_identity=backend,
            build_identity=build,
            dimensions=self.dimensions,
        )
        if self.cache_key != expected_cache_key:
            raise ValueError("cache_key does not match host/backend/build/dimensions.")
        if backend["numeric_policy"] != self.plan.numeric_policy:
            raise ValueError(
                "backend numeric_policy disagrees with plan numeric_policy."
            )
        if backend["native_execution_backend"] != self.plan.integrity_backend:
            raise ValueError(
                "backend native_execution_backend disagrees with plan integrity_backend."
            )
        if self.plan.affinity_cpu_count > host["logical_cpu_count"]:
            raise ValueError(
                "plan affinity_cpu_count cannot exceed host logical_cpu_count."
            )
        if (
            checked_u128_product_v1(
                self.plan.process_count, self.plan.affinity_cpu_count
            )
            > host["logical_cpu_count"]
        ):
            raise ValueError(
                "process_count*affinity_cpu_count cannot exceed host "
                "logical_cpu_count."
            )
        if self.plan.output_numa_node >= host["numa_node_count"]:
            raise ValueError("output_numa_node is outside the host NUMA-node range.")
        _sha256("science_identity_sha256", self.science_identity_sha256)
        _sha256("integrity_identity_sha256", self.integrity_identity_sha256)
        _sha256("rank_identity_sha256", self.rank_identity_sha256)
        object.__setattr__(self, "metrics", _validate_metrics(self.metrics))
        if not isinstance(self.acceptance, ContextualTablaAcceptanceV1):
            raise ValueError("acceptance must be ContextualTablaAcceptanceV1.")
        _enum("terminal_status", self.terminal_status, TERMINAL_STATUSES_V1)
        _boolean("production_qualified", self.production_qualified)
        reasons = tuple(
            _nonempty("rejection reason", reason) for reason in self.rejection_reasons
        )
        if len(reasons) != len(set(reasons)):
            raise ValueError("rejection_reasons must be unique.")
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(
            self,
            "benchmark_speedup_threshold",
            _finite(
                "benchmark_speedup_threshold",
                self.benchmark_speedup_threshold,
                minimum=DEFAULT_MATERIAL_SPEEDUP,
            ),
        )
        relationship = (
            self.comparison_baseline_run_id,
            self.comparison_baseline_record_sha256,
            self.declared_parent_source_commit,
        )
        if any(value is not None for value in relationship):
            if any(value is None for value in relationship):
                raise ValueError(
                    "Cross-build comparison relationship fields must be all present "
                    "or all absent."
                )
            _nonempty("comparison_baseline_run_id", self.comparison_baseline_run_id)
            _sha256(
                "comparison_baseline_record_sha256",
                self.comparison_baseline_record_sha256,
            )
            _nonempty(
                "declared_parent_source_commit", self.declared_parent_source_commit
            )
            _git_commit(
                "declared_parent_source_commit", self.declared_parent_source_commit
            )
        if self.terminal_status == "accepted" and reasons:
            raise ValueError("Accepted records cannot contain rejection reasons.")
        if self.terminal_status != "accepted" and not reasons:
            raise ValueError("Non-accepted records must record rejection reasons.")
        if self.production_qualified:
            qualification_errors = self.selection_ineligibility_reasons()
            if qualification_errors:
                raise ValueError(
                    "A production-qualified record is inconsistent: "
                    + ", ".join(qualification_errors)
                )

    def selection_ineligibility_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.terminal_status != "accepted":
            reasons.append("terminal_status_not_accepted")
            reasons.extend(
                f"record_rejection_reason:{reason}" for reason in self.rejection_reasons
            )
        if not self.production_qualified:
            reasons.append("record_not_production_qualified")
        reasons.extend(self.plan.production_nonqualification_reasons)
        if not self.acceptance.all_gates_passed:
            reasons.append("full_acceptance_evidence_missing")
        for metric in sorted(REQUIRED_SELECTION_MEASURED_METRICS_V1):
            if self.metrics[metric].evidence_kind != "measured":
                reasons.append(f"required_metric_not_measured:{metric}")
        physical_reads = self.metrics["physical_read_bytes"]
        if physical_reads.evidence_kind == "modeled":
            reasons.append(
                "physical_read_bytes_not_measured_or_explicit_mmap_unavailable"
            )
        if _measured_positive_scalar(self, "end_to_end_wall_seconds") is None:
            reasons.append("end_to_end_wall_not_positive_measured_scalar")
        return tuple(dict.fromkeys(reasons))

    @property
    def record_sha256(self) -> str:
        return canonical_sha256(
            {"magic": CONTEXTUAL_TABLA_RECORD_ID_V1_MAGIC, "record": self.to_dict()}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_TABLA_BENCHMARK_RECORD_V1_SCHEMA,
            "run_id": self.run_id,
            "host_identity": _thaw(self.host_identity),
            "backend_identity": _thaw(self.backend_identity),
            "build_identity": _thaw(self.build_identity),
            "cache_key": self.cache_key,
            "dimensions": self.dimensions.to_dict(),
            "plan": self.plan.to_dict(),
            "plan_sha256": self.plan.plan_sha256,
            "science_identity_sha256": self.science_identity_sha256,
            "integrity_identity_sha256": self.integrity_identity_sha256,
            "rank_identity_sha256": self.rank_identity_sha256,
            "metrics": {
                key: self.metrics[key].to_dict() for key in PERFORMANCE_METRIC_KEYS_V1
            },
            "acceptance": self.acceptance.to_dict(),
            "terminal_status": self.terminal_status,
            "production_qualified": self.production_qualified,
            "rejection_reasons": list(self.rejection_reasons),
            "benchmark_speedup_threshold": self.benchmark_speedup_threshold,
            "comparison_baseline_run_id": self.comparison_baseline_run_id,
            "comparison_baseline_record_sha256": (
                self.comparison_baseline_record_sha256
            ),
            "declared_parent_source_commit": self.declared_parent_source_commit,
        }


@dataclass(frozen=True, slots=True)
class ContextualTablaEstimateV1:
    """Checked first-order memory, call, and cost estimate for one plan."""

    dimensions: ContextualTablaDimensionsV1
    plan: ContextualTablaPlanV1
    memory_categories: Mapping[str, int]
    phase_lower_bound_bytes: Mapping[str, int]
    logical_descriptor_passes: Mapping[str, int]
    decoded_blocks: Mapping[str, int]
    protected_call_lower_bounds: Mapping[str, int]
    flop_lower_bounds: Mapping[str, int]
    required_peak_bytes: int
    workspace_cap_bytes: int
    admitted: bool
    evidence_kind: str = "modeled"

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, ContextualTablaDimensionsV1):
            raise ValueError("dimensions must be ContextualTablaDimensionsV1.")
        if not isinstance(self.plan, ContextualTablaPlanV1):
            raise ValueError("plan must be ContextualTablaPlanV1.")
        _exact_keys(
            "memory_categories", self.memory_categories, MEMORY_CATEGORY_KEYS_V1
        )
        for mapping_name in (
            "memory_categories",
            "phase_lower_bound_bytes",
            "logical_descriptor_passes",
            "decoded_blocks",
            "protected_call_lower_bounds",
            "flop_lower_bounds",
        ):
            mapping = getattr(self, mapping_name)
            if not isinstance(mapping, Mapping) or not mapping:
                raise ValueError(f"{mapping_name} must be a nonempty mapping.")
            frozen: dict[str, int] = {}
            for key, value in mapping.items():
                frozen[_nonempty(f"{mapping_name} key", key)] = _integer(
                    f"{mapping_name}[{key!r}]", value
                )
            object.__setattr__(self, mapping_name, freeze_context_mapping(frozen))
        _integer("required_peak_bytes", self.required_peak_bytes)
        _integer("workspace_cap_bytes", self.workspace_cap_bytes)
        _boolean("admitted", self.admitted)
        if self.admitted != (self.required_peak_bytes <= self.workspace_cap_bytes):
            raise ValueError("admitted disagrees with required_peak_bytes and cap.")
        if self.evidence_kind != "modeled":
            raise ValueError("Planning estimates must be labeled modeled.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_TABLA_ESTIMATE_V1_SCHEMA,
            "dimensions": self.dimensions.to_dict(),
            "plan": self.plan.to_dict(),
            "memory_categories": _thaw(self.memory_categories),
            "phase_lower_bound_bytes": _thaw(self.phase_lower_bound_bytes),
            "logical_descriptor_passes": _thaw(self.logical_descriptor_passes),
            "decoded_blocks": _thaw(self.decoded_blocks),
            "protected_call_lower_bounds": _thaw(self.protected_call_lower_bounds),
            "flop_lower_bounds": _thaw(self.flop_lower_bounds),
            "required_peak_bytes": self.required_peak_bytes,
            "workspace_cap_bytes": self.workspace_cap_bytes,
            "admitted": self.admitted,
            "evidence_kind": self.evidence_kind,
            "warning": (
                "First-order modeled lower bound; not measured RSS or production "
                "qualification."
            ),
        }


def _bytes(*factors: int) -> int:
    return checked_u128_product_v1(8, *factors)


def _resident_probe_tiles_v1(total: int, resident: int, tile: int) -> int:
    full_residents, remainder = divmod(total, resident)
    result = checked_u128_product_v1(
        full_residents,
        _ceil_div_u128("resident probe tiles", resident, tile),
    )
    if remainder:
        result = checked_u128_sum_v1(
            result,
            _ceil_div_u128("final resident probe tiles", remainder, tile),
        )
    return result


def _contextual_tabla_memory_v1(
    dimensions: ContextualTablaDimensionsV1,
    plan: ContextualTablaPlanV1,
) -> tuple[dict[str, int], dict[str, int], int]:
    """Model the current native executors' constructor-resident allocations."""
    n = dimensions.n_samples
    m = dimensions.n_variants
    q = dimensions.context_count
    k = dimensions.annotation_count
    pairs = dimensions.pair_count
    c = dimensions.component_count
    groups = dimensions.group_count
    bt = dimensions.sample_probe_count
    bd = dimensions.variant_probe_count
    rank = dimensions.fixed_rank
    traits = dimensions.trait_count
    residuals = dimensions.residual_basis_count
    resident = plan.sample_probe_resident_count
    sample_tile = plan.sample_probe_tile
    variant_tile = plan.variant_probe_tile
    action_tile = plan.action_tile
    annotation_tile = plan.annotation_tile
    context_tile = plan.context_tile
    group_tile = plan.group_tile
    # The native reference executor has one variant-block allocation axis.  A
    # plan record exposes phase-specific candidates, so using their maximum is
    # conservative for every simultaneously resident reference scratch panel.
    reference_block = max(
        plan.source_variant_block,
        plan.target_variant_block,
        plan.grouped_variant_block,
        plan.same_person_variant_block,
    )

    pair_map_elements = checked_u128_sum_v1(
        checked_u128_product_v1(3, pairs),
        checked_u128_product_v1(2, c),
    )
    strict_ids = _bytes(m) if plan.annotation_mode == "strict_disjoint_binary_v1" else 0
    reference_metadata = checked_u128_sum_v1(
        _bytes(n),
        _bytes(m),
        m,
        _bytes(m, 2),
        _bytes(m),
        _bytes(n, rank),
        _bytes(n, q),
        _bytes(m, k),
        _bytes(m),
        _bytes(k),
        _bytes(groups, k),
        _bytes(groups),
        _bytes(pair_map_elements),
        strict_ids,
        _bytes(bt, 2),
        checked_u128_product_v1(m, 64),
        _bytes(bd, 2),
        _bytes(m),
        _bytes(m),
    )
    source_scores = _bytes(m, q, bt)
    raw_actions = _bytes(n, c, bt)
    permanent_outputs = checked_u128_sum_v1(
        _bytes(c, c, 2),
        _bytes(groups, c, c),
        _bytes(c, c),
    )
    compact_outputs = checked_u128_sum_v1(
        _bytes(c, c, 2),
        _bytes(k),
        _bytes(pair_map_elements),
        _bytes(c, c),
        _bytes(groups, c, c),
        _bytes(2, groups, k),
        _bytes(groups),
    )

    max_projection_columns = max(
        sample_tile,
        checked_u128_product_v1(action_tile, sample_tile),
        checked_u128_product_v1(k, q, variant_tile),
    )
    decode = checked_u128_sum_v1(
        _bytes(n, reference_block),
        checked_u128_product_v1(n, reference_block),
        checked_u128_product_v1(4, reference_block),
    )
    probe_panel = _bytes(n, resident, 2)
    projection = checked_u128_sum_v1(
        _bytes(rank, max_projection_columns),
        _bytes(n, max_projection_columns),
    )
    source_rhs = _bytes(n, context_tile, sample_tile)
    targets = _bytes(n, k, q, resident)
    action_scratch = checked_u128_sum_v1(
        _bytes(reference_block, annotation_tile, context_tile, sample_tile),
        _bytes(
            max(
                reference_block,
                checked_u128_product_v1(n, annotation_tile),
            ),
            context_tile,
            sample_tile,
        ),
        _bytes(n, action_tile, sample_tile, 2),
        _bytes(n, c, resident),
        _bytes(c, c),
        _bytes(n, reference_block),
    )
    execution_arena = checked_u128_sum_v1(
        decode,
        probe_panel,
        projection,
        source_rhs,
        targets,
        action_scratch,
    )

    group_targets = _bytes(group_tile, n, k, q, bt)
    group_actions = checked_u128_sum_v1(
        _bytes(n, bt, action_tile),
        _bytes(n, action_tile, sample_tile, 2),
    )
    # Native currently allocates action_tile*C.  Retaining group_tile as an
    # explicit plan axis is conservative and prevents ambiguous tile records.
    group_cross = _bytes(group_tile, action_tile, c)
    direct_scaled = _bytes(n, reference_block)
    direct_output = _bytes(reference_block, action_tile, sample_tile)
    group_accum = _bytes(2, groups, c, c)
    group_arena = checked_u128_sum_v1(
        group_targets,
        group_actions,
        group_cross,
        direct_scaled,
        direct_output,
        group_accum,
    )

    same_sketch = checked_u128_sum_v1(
        _bytes(m, variant_tile),
        _bytes(reference_block, k, variant_tile),
        _bytes(n, k, variant_tile, 2),
        _bytes(n, k, q, variant_tile, 2),
    )
    same_g = _bytes(n, variant_tile, c)
    same_persistent = checked_u128_sum_v1(_bytes(n, c), _bytes(c, c))
    same_arena = checked_u128_sum_v1(same_sketch, same_g, same_persistent)

    reference_max_output = max(
        checked_u128_product_v1(rank, max_projection_columns),
        checked_u128_product_v1(n, max_projection_columns),
        checked_u128_product_v1(reference_block, context_tile, sample_tile),
        checked_u128_product_v1(n, annotation_tile, context_tile, sample_tile),
        checked_u128_product_v1(c, c),
        checked_u128_product_v1(n, k, variant_tile),
        checked_u128_product_v1(n, k, q, variant_tile),
        checked_u128_product_v1(reference_block, action_tile, sample_tile),
        checked_u128_product_v1(action_tile, c),
    )
    derived_reference_integrity = _bytes(
        4, checked_u128_sum_v1(reference_max_output, 16)
    )
    reference_integrity = max(derived_reference_integrity, plan.integrity_reserve_bytes)

    residents = _ceil_div_u128("resident batches", bt, resident)
    probe_tiles = _resident_probe_tiles_v1(bt, resident, sample_tile)
    context_tiles = _ceil_div_u128("context tiles", q, context_tile)
    annotation_tiles = _ceil_div_u128("annotation tiles", k, annotation_tile)
    action_tiles = _ceil_div_u128("action tiles", c, action_tile)
    group_probe_tiles = _ceil_div_u128("group probe tiles", bt, sample_tile)
    source_blocks = _ceil_div_u128(
        "source telemetry blocks", m, plan.source_variant_block
    )
    target_blocks = _ceil_div_u128(
        "target telemetry blocks", m, plan.target_variant_block
    )
    group_blocks = _ceil_div_u128(
        "group telemetry blocks", m, plan.grouped_variant_block
    )
    same_blocks = _ceil_div_u128(
        "same telemetry blocks", m, plan.same_person_variant_block
    )
    variant_tiles = _ceil_div_u128("variant probe tiles", bd, variant_tile)
    reference_calls = checked_u128_sum_v1(
        checked_u128_product_v1(2, probe_tiles),
        checked_u128_product_v1(probe_tiles, source_blocks, context_tiles),
        checked_u128_product_v1(
            probe_tiles, target_blocks, annotation_tiles, context_tiles
        ),
        checked_u128_product_v1(2, probe_tiles, action_tiles),
        residents,
    )
    if plan.grouped_attribution_algorithm == "group_restricted_action_v1":
        restricted_blocks = checked_u128_sum_v1(group_blocks, groups)
        reference_calls = checked_u128_sum_v1(
            reference_calls,
            checked_u128_product_v1(
                restricted_blocks,
                group_probe_tiles,
                annotation_tiles,
                context_tiles,
            ),
            checked_u128_product_v1(groups, action_tiles),
        )
    else:
        reference_calls = checked_u128_sum_v1(
            reference_calls,
            checked_u128_product_v1(group_blocks, q, action_tiles, group_probe_tiles),
        )
    reference_calls = checked_u128_sum_v1(
        reference_calls,
        checked_u128_product_v1(variant_tiles, same_blocks),
        checked_u128_product_v1(3, variant_tiles),
        1,
    )
    reference_event_capacity = checked_u128_sum_v1(
        checked_u128_product_v1(reference_calls, 8), 32
    )
    derived_reference_telemetry = checked_u128_product_v1(
        reference_event_capacity, CONSERVATIVE_TELEMETRY_EVENT_BYTES_V1
    )
    reference_telemetry = max(
        derived_reference_telemetry, plan.telemetry_capacity_bytes
    )

    reference_required = checked_u128_sum_v1(
        reference_metadata,
        source_scores,
        raw_actions,
        permanent_outputs,
        execution_arena,
        group_arena,
        same_arena,
        reference_integrity,
        compact_outputs,
        reference_telemetry,
    )

    trait_permanent = 0
    trait_residual = 0
    trait_features = 0
    trait_scores = 0
    trait_scratch = 0
    trait_integrity = 0
    trait_telemetry = 0
    trait_required = 0
    if traits:
        trait_block = plan.trait_variant_block
        # The Stage 6 harness binds the native trait feature tile to the trait
        # variant block; the plan has no independent trait-feature-tile axis.
        trait_feature_tile = trait_block
        trait_permanent = checked_u128_sum_v1(
            _bytes(n),
            _bytes(m),
            m,
            _bytes(m, 2),
            _bytes(m),
            _bytes(n, rank),
            _bytes(n, q),
            _bytes(m, k),
            _bytes(m),
            _bytes(m),
            checked_u128_product_v1(m, 64),
            strict_ids,
            _bytes(n, traits),
            _bytes(n, residuals),
            _bytes(k),
            _bytes(groups, k),
            _bytes(groups),
            _bytes(pair_map_elements),
            _bytes(c, traits),
            _bytes(c),
            _bytes(c, residuals),
            _bytes(residuals, traits),
            _bytes(residuals),
            _bytes(residuals, residuals),
            _bytes(groups, c, traits),
            _bytes(groups, c),
            _bytes(groups, c, residuals),
            # Compact publication duplicates the result arrays and masses.
            _bytes(c, traits),
            _bytes(c),
            _bytes(c, residuals),
            _bytes(residuals, traits),
            _bytes(residuals),
            _bytes(residuals, residuals),
            _bytes(groups, c, traits),
            _bytes(groups, c),
            _bytes(groups, c, residuals),
            _bytes(k),
            _bytes(groups, k),
            _bytes(groups),
        )
        trait_residual = checked_u128_sum_v1(
            _bytes(n, traits),
            _bytes(rank, traits),
            _bytes(n),
            _bytes(residuals, rank, rank),
        )
        trait_features = checked_u128_sum_v1(
            _bytes(n, q, trait_block, 2),
            _bytes(n, trait_feature_tile),
        )
        trait_scores = checked_u128_sum_v1(
            _bytes(n, q, traits),
            _bytes(trait_block, q, traits),
        )
        trait_scratch = checked_u128_sum_v1(
            _bytes(n, trait_block),
            checked_u128_product_v1(n, trait_block),
            checked_u128_product_v1(4, trait_block),
            _bytes(rank, trait_feature_tile),
            _bytes(n, trait_feature_tile),
        )
        trait_max_output = max(
            checked_u128_product_v1(trait_block, q, traits),
            checked_u128_product_v1(n, trait_feature_tile),
            checked_u128_product_v1(rank, trait_feature_tile),
        )
        trait_integrity = max(
            _bytes(4, checked_u128_sum_v1(trait_max_output, 16)),
            plan.integrity_reserve_bytes,
        )
        trait_blocks = _ceil_div_u128("trait telemetry blocks", m, trait_block)
        full_trait_blocks, final_trait_width = divmod(m, trait_block)
        feature_tiles = checked_u128_product_v1(
            full_trait_blocks,
            _ceil_div_u128("full trait feature tiles", trait_block, trait_feature_tile),
        )
        if final_trait_width:
            feature_tiles = checked_u128_sum_v1(
                feature_tiles,
                _ceil_div_u128(
                    "final trait feature tiles",
                    final_trait_width,
                    trait_feature_tile,
                ),
            )
        trait_calls = checked_u128_sum_v1(
            trait_blocks,
            checked_u128_product_v1(2, q, feature_tiles),
        )
        trait_event_capacity = checked_u128_sum_v1(
            checked_u128_product_v1(trait_calls, 8), 32
        )
        trait_telemetry = max(
            checked_u128_product_v1(
                trait_event_capacity, CONSERVATIVE_TELEMETRY_EVENT_BYTES_V1
            ),
            plan.telemetry_capacity_bytes,
        )
        trait_required = checked_u128_sum_v1(
            trait_permanent,
            trait_residual,
            trait_features,
            trait_scores,
            trait_scratch,
            trait_integrity,
            trait_telemetry,
        )

    categories = {
        "bytes_executor_permanent": reference_metadata,
        "bytes_decode": decode,
        "bytes_probe_panel": probe_panel,
        "bytes_source_rhs": source_rhs,
        "bytes_source_scores": source_scores,
        "bytes_targets": targets,
        "bytes_actions": raw_actions,
        "bytes_action_scratch": action_scratch,
        "bytes_group_targets": group_targets,
        "bytes_group_action_tile": group_actions,
        "bytes_group_cross": group_cross,
        "bytes_direct_group_scaled": direct_scaled,
        "bytes_direct_group_output": direct_output,
        "bytes_group_accum": group_accum,
        "bytes_same_sketch": same_sketch,
        "bytes_same_g": same_g,
        "bytes_same_persistent": same_persistent,
        "bytes_trait_features": trait_features,
        "bytes_trait_scores": trait_scores,
        "bytes_trait_permanent": trait_permanent,
        "bytes_trait_residual": trait_residual,
        "bytes_trait_scratch": trait_scratch,
        "bytes_outputs": checked_u128_sum_v1(permanent_outputs, compact_outputs),
        "bytes_projection": projection,
        "bytes_integrity": max(reference_integrity, trait_integrity),
        "bytes_telemetry": max(reference_telemetry, trait_telemetry),
        "bytes_headroom": 0,
    }
    pre_headroom_peak = max(reference_required, trait_required)
    headroom = checked_u128_sum_v1(
        _checked_scale_basis_points(pre_headroom_peak, plan.headroom_basis_points),
        plan.fixed_headroom_bytes,
    )
    categories["bytes_headroom"] = headroom
    phases = {
        "source": reference_required,
        "action": reference_required,
        "group": reference_required,
        "same_person": reference_required,
        "trait": trait_required,
    }
    if plan.process_count > 1:
        categories = {
            key: checked_u128_product_v1(value, plan.process_count)
            for key, value in categories.items()
        }
        phases = {
            key: checked_u128_product_v1(value, plan.process_count)
            for key, value in phases.items()
        }
    required = checked_u128_product_v1(
        checked_u128_sum_v1(pre_headroom_peak, headroom), plan.process_count
    )
    return categories, phases, required


def _contextual_tabla_protected_call_lower_bounds_v1(
    dimensions: ContextualTablaDimensionsV1,
    plan: ContextualTablaPlanV1,
) -> dict[str, int]:
    """Return operation-summed native call counts or provable minima."""
    m = dimensions.n_variants
    q = dimensions.context_count
    k = dimensions.annotation_count
    c = dimensions.component_count
    groups = dimensions.group_count
    bt = dimensions.sample_probe_count
    bd = dimensions.variant_probe_count
    probe_tiles = _resident_probe_tiles_v1(
        bt, plan.sample_probe_resident_count, plan.sample_probe_tile
    )
    residents = _ceil_div_u128(
        "protected-call resident batches",
        bt,
        plan.sample_probe_resident_count,
    )
    context_tiles = _ceil_div_u128("protected-call context tiles", q, plan.context_tile)
    annotation_tiles = _ceil_div_u128(
        "protected-call annotation tiles", k, plan.annotation_tile
    )
    action_tiles = _ceil_div_u128("protected-call action tiles", c, plan.action_tile)
    group_probe_tiles = _ceil_div_u128(
        "protected-call grouped probe tiles", bt, plan.sample_probe_tile
    )
    variant_probe_tiles = _ceil_div_u128(
        "protected-call variant-probe tiles", bd, plan.variant_probe_tile
    )
    source_blocks = _ceil_div_u128(
        "protected-call source blocks", m, plan.source_variant_block
    )
    target_blocks = _ceil_div_u128(
        "protected-call target blocks", m, plan.target_variant_block
    )
    group_blocks = _ceil_div_u128(
        "protected-call group blocks", m, plan.grouped_variant_block
    )
    same_blocks = _ceil_div_u128(
        "protected-call same-person blocks", m, plan.same_person_variant_block
    )
    source = checked_u128_sum_v1(
        checked_u128_product_v1(2, probe_tiles),
        checked_u128_product_v1(probe_tiles, source_blocks, context_tiles),
    )
    if plan.annotation_mode == "generic_nonnegative_weights_v1":
        full_target = checked_u128_product_v1(
            probe_tiles, target_blocks, annotation_tiles, context_tiles
        )
    else:
        # Every nonempty strict block has at least one active annotation.  The
        # precise active-annotation ledger is data-dependent and can be larger.
        full_target = checked_u128_product_v1(probe_tiles, target_blocks, context_tiles)
    action = checked_u128_sum_v1(
        full_target,
        checked_u128_product_v1(2, probe_tiles, action_tiles),
        residents,
    )
    if plan.grouped_attribution_algorithm == "group_restricted_action_v1":
        if plan.annotation_mode == "generic_nonnegative_weights_v1":
            group_target = checked_u128_product_v1(
                group_blocks,
                group_probe_tiles,
                annotation_tiles,
                context_tiles,
            )
        else:
            group_target = checked_u128_product_v1(
                group_blocks, group_probe_tiles, context_tiles
            )
        group = checked_u128_sum_v1(
            group_target,
            checked_u128_product_v1(groups, action_tiles),
        )
    else:
        group = checked_u128_product_v1(
            group_blocks, q, action_tiles, group_probe_tiles
        )
    same_person = checked_u128_sum_v1(
        checked_u128_product_v1(variant_probe_tiles, same_blocks),
        checked_u128_product_v1(3, variant_probe_tiles),
        1,
    )
    if dimensions.trait_count:
        trait_blocks = _ceil_div_u128(
            "protected-call trait blocks", m, plan.trait_variant_block
        )
        # The executable Stage 6 plan binds trait_feature_tile to trait_block,
        # so each variant block has one feature tile in each context.
        trait = checked_u128_product_v1(
            trait_blocks, checked_u128_sum_v1(1, checked_u128_product_v1(2, q))
        )
    else:
        trait = 0
    per_process = {
        "source": source,
        "action": action,
        "group": group,
        "same_person": same_person,
        "trait": trait,
    }
    return {
        phase: checked_u128_product_v1(count, plan.process_count)
        for phase, count in per_process.items()
    }


def estimate_contextual_tabla_plan_v1(
    dimensions: ContextualTablaDimensionsV1,
    plan: ContextualTablaPlanV1,
    *,
    workspace_cap_bytes: int,
) -> ContextualTablaEstimateV1:
    """Build a conservative, checked first-order Stage 6 planning estimate."""
    if not isinstance(dimensions, ContextualTablaDimensionsV1):
        raise ValueError("dimensions must be ContextualTablaDimensionsV1.")
    if not isinstance(plan, ContextualTablaPlanV1):
        raise ValueError("plan must be ContextualTablaPlanV1.")
    _validate_plan_against_dimensions(dimensions, plan)
    cap = _integer("workspace_cap_bytes", workspace_cap_bytes)
    n = dimensions.n_samples
    m = dimensions.n_variants
    q = dimensions.context_count
    c = dimensions.component_count
    bt = dimensions.sample_probe_count
    bd = dimensions.variant_probe_count
    resident = plan.sample_probe_resident_count
    categories, phases, required_peak = _contextual_tabla_memory_v1(dimensions, plan)

    source_passes = _ceil_div_u128("source passes", bt, resident)
    action_passes = _ceil_div_u128("action passes", bt, resident)
    group_passes = 1
    same_passes = _ceil_div_u128("same-person passes", bd, plan.variant_probe_tile)
    trait_passes = 1 if dimensions.trait_count else 0
    descriptor_passes = {
        "source": source_passes,
        "action": action_passes,
        "group": group_passes,
        "same_person": same_passes,
        "trait": trait_passes,
    }
    variant_blocks = {
        "source": _ceil_div_u128("source blocks", m, plan.source_variant_block),
        "action": _ceil_div_u128("action blocks", m, plan.target_variant_block),
        "group": _ceil_div_u128("grouped blocks", m, plan.grouped_variant_block),
        "same_person": _ceil_div_u128(
            "same-person blocks", m, plan.same_person_variant_block
        ),
        "trait": (
            _ceil_div_u128("trait blocks", m, plan.trait_variant_block)
            if dimensions.trait_count
            else 0
        ),
    }
    decoded_blocks_per_process = {
        phase: checked_u128_product_v1(descriptor_passes[phase], count)
        for phase, count in variant_blocks.items()
    }
    descriptor_passes = {
        phase: checked_u128_product_v1(count, plan.process_count)
        for phase, count in descriptor_passes.items()
    }
    decoded_blocks = {
        phase: checked_u128_product_v1(count, plan.process_count)
        for phase, count in decoded_blocks_per_process.items()
    }
    protected_calls = _contextual_tabla_protected_call_lower_bounds_v1(dimensions, plan)
    if plan.grouped_attribution_algorithm == "group_restricted_action_v1":
        group_flops_per_process = checked_u128_product_v1(2, n, m, q, bt)
    else:
        group_flops_per_process = checked_u128_product_v1(2, n, m, q, c, bt)
    flop_per_process = {
        "source": checked_u128_product_v1(2, n, m, q, bt),
        "action": checked_u128_product_v1(2, n, c, c, bt),
        "group": group_flops_per_process,
        "same_person": checked_u128_product_v1(2, n, c, c, bd),
        "trait": checked_u128_product_v1(2, n, m, q, dimensions.trait_count),
    }
    flops = {
        phase: checked_u128_product_v1(value, plan.process_count)
        for phase, value in flop_per_process.items()
    }
    return ContextualTablaEstimateV1(
        dimensions=dimensions,
        plan=plan,
        memory_categories=categories,
        phase_lower_bound_bytes=phases,
        logical_descriptor_passes=descriptor_passes,
        decoded_blocks=decoded_blocks,
        protected_call_lower_bounds=protected_calls,
        flop_lower_bounds=flops,
        required_peak_bytes=required_peak,
        workspace_cap_bytes=cap,
        admitted=required_peak <= cap,
    )


def tabla_target_first_order_estimate_v1(probe_count: int) -> Mapping[str, Any]:
    """Return the package template's target-workload first-order estimate."""
    probes = _integer("probe_count", probe_count, minimum=2)
    n = 300_000
    m = 1_000_000
    q = 4
    k = 8
    c = 80
    j = 200
    action_tile = 8
    resident = {
        "source_scores": _bytes(m, q, probes),
        "full_raw_actions": _bytes(n, c, probes),
        "full_targets_generic": _bytes(n, k, q, probes),
        "group_action_tile": _bytes(n, action_tile, probes),
        "group_output_fp64": _bytes(j, c, c),
        "group_targets": _bytes(n, k, q, probes),
        "same_person_g_full": _bytes(n, c, probes),
        "same_person_probe_sums_fp64": _bytes(n, c),
        "same_person_sketches_generic": _bytes(n, k, q, probes),
        "sample_probe_context_panel": _bytes(n, q, probes),
    }
    core = {
        "source_core": checked_u128_product_v1(41_600_000, probes),
        "action_core": checked_u128_product_v1(300_800_000, probes),
        "group_restricted_core": checked_u128_sum_v1(
            checked_u128_product_v1(320_000_000, probes), 10_240_000
        ),
        "same_person_core": checked_u128_sum_v1(
            checked_u128_product_v1(268_800_000, probes), 192_000_000
        ),
    }
    columns = {
        "source": checked_u128_product_v1(q, probes),
        "strict_disjoint_useful": checked_u128_product_v1(q, probes),
        "full_target_generic": checked_u128_product_v1(k, q, probes),
        "group_restricted_generic": checked_u128_product_v1(k, q, probes),
        "same_person_generic": checked_u128_product_v1(k, probes),
        "direct_grouped_tn": checked_u128_product_v1(c, q, probes),
    }
    return freeze_context_mapping(
        {
            "evidence_kind": "modeled",
            "dimensions": {
                "n_samples": n,
                "n_variants": m,
                "context_count": q,
                "annotation_count": k,
                "pair_count": 10,
                "component_count": c,
                "group_count": j,
                "sample_probe_count": probes,
                "variant_probe_count": probes,
                "action_tile": action_tile,
                "bytes_per_value": 8,
            },
            "resident_objects_bytes": resident,
            "core_phase_lower_bound_bytes": core,
            "effective_genotype_rhs_columns": columns,
            "warning": (
                "Core lower bounds, not peak admission; add scratch, retry/fallback, "
                "telemetry, allocator, NUMA/process duplication, and headroom."
            ),
        }
    )


TABLA_TARGET_B128_FIRST_ORDER_ESTIMATE_V1 = tabla_target_first_order_estimate_v1(128)
TABLA_TARGET_B1024_FIRST_ORDER_ESTIMATE_V1 = tabla_target_first_order_estimate_v1(1024)


@dataclass(frozen=True, slots=True)
class ContextualTablaCandidateDecisionV1:
    run_id: str
    selected: bool
    speedup: float | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty("run_id", self.run_id)
        _boolean("selected", self.selected)
        if self.speedup is not None:
            object.__setattr__(self, "speedup", _finite("speedup", self.speedup))
        reasons = tuple(
            _nonempty("rejection reason", reason) for reason in self.rejection_reasons
        )
        if len(reasons) != len(set(reasons)):
            raise ValueError("rejection_reasons must be unique.")
        if self.selected == bool(reasons):
            raise ValueError(
                "Exactly selected candidates must have no rejection reasons."
            )
        object.__setattr__(self, "rejection_reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selected": self.selected,
            "speedup": self.speedup,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True, slots=True)
class ContextualTablaSelectionV1:
    baseline_run_id: str
    selected_run_id: str
    selected_plan: ContextualTablaPlanV1
    selected_record_sha256: str
    selected_is_baseline: bool
    selected_speedup: float
    candidate_decisions: tuple[ContextualTablaCandidateDecisionV1, ...]

    def __post_init__(self) -> None:
        _nonempty("baseline_run_id", self.baseline_run_id)
        _nonempty("selected_run_id", self.selected_run_id)
        if not isinstance(self.selected_plan, ContextualTablaPlanV1):
            raise ValueError("selected_plan must be ContextualTablaPlanV1.")
        _sha256("selected_record_sha256", self.selected_record_sha256)
        _boolean("selected_is_baseline", self.selected_is_baseline)
        object.__setattr__(
            self, "selected_speedup", _finite("selected_speedup", self.selected_speedup)
        )
        decisions = tuple(self.candidate_decisions)
        if any(
            not isinstance(item, ContextualTablaCandidateDecisionV1)
            for item in decisions
        ):
            raise ValueError("candidate_decisions contains an invalid item.")
        if tuple(sorted(item.run_id for item in decisions)) != tuple(
            item.run_id for item in decisions
        ):
            raise ValueError("candidate_decisions must be ordered by run_id.")
        selected_decisions = [item for item in decisions if item.selected]
        expected_selected = 0 if self.selected_is_baseline else 1
        if len(selected_decisions) != expected_selected:
            raise ValueError("candidate_decisions selection count is inconsistent.")
        if selected_decisions and selected_decisions[0].run_id != self.selected_run_id:
            raise ValueError("Selected candidate decision has the wrong run_id.")
        object.__setattr__(self, "candidate_decisions", decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXTUAL_TABLA_SELECTION_V1_SCHEMA,
            "baseline_run_id": self.baseline_run_id,
            "selected_run_id": self.selected_run_id,
            "selected_plan": self.selected_plan.to_dict(),
            "selected_record_sha256": self.selected_record_sha256,
            "selected_is_baseline": self.selected_is_baseline,
            "selected_speedup": self.selected_speedup,
            "candidate_decisions": [
                decision.to_dict() for decision in self.candidate_decisions
            ],
        }


def _candidate_reasons(
    baseline: ContextualTablaBenchmarkRecordV1,
    candidate: ContextualTablaBenchmarkRecordV1,
) -> list[str]:
    reasons = list(candidate.selection_ineligibility_reasons())
    if candidate.run_id == baseline.run_id:
        reasons.append("candidate_reuses_baseline_run_id")
    if candidate.host_identity != baseline.host_identity:
        reasons.append("host_identity_mismatch")
    if candidate.build_identity != baseline.build_identity:
        if candidate.comparison_baseline_run_id != baseline.run_id:
            reasons.append("cross_build_baseline_run_id_mismatch")
        if candidate.comparison_baseline_record_sha256 != baseline.record_sha256:
            reasons.append("cross_build_baseline_record_sha256_mismatch")
        if (
            candidate.declared_parent_source_commit
            != baseline.build_identity["source_commit"]
        ):
            reasons.append("cross_build_parent_source_commit_mismatch")
    elif any(
        value is not None
        for value in (
            candidate.comparison_baseline_run_id,
            candidate.comparison_baseline_record_sha256,
            candidate.declared_parent_source_commit,
        )
    ):
        if candidate.comparison_baseline_run_id != baseline.run_id:
            reasons.append("declared_baseline_run_id_mismatch")
        if candidate.comparison_baseline_record_sha256 != baseline.record_sha256:
            reasons.append("declared_baseline_record_sha256_mismatch")
        if (
            candidate.declared_parent_source_commit
            != baseline.build_identity["source_commit"]
        ):
            reasons.append("declared_parent_source_commit_mismatch")
    if candidate.dimensions != baseline.dimensions:
        reasons.append("dimensions_mismatch")
    if candidate.science_identity_sha256 != baseline.science_identity_sha256:
        reasons.append("science_identity_mismatch")
    if candidate.integrity_identity_sha256 != baseline.integrity_identity_sha256:
        reasons.append("integrity_identity_mismatch")
    if candidate.rank_identity_sha256 != baseline.rank_identity_sha256:
        reasons.append("rank_identity_mismatch")
    # The common science identity freezes the acceptance/tolerance contract,
    # while each record's measured acceptance binds its own discrepancies.
    # Requiring the discrepancy vectors themselves to be identical would make
    # a nonzero-but-within-contract candidate impossible to select.
    return list(dict.fromkeys(reasons))


def select_contextual_tabla_plan_v1(
    baseline: ContextualTablaBenchmarkRecordV1,
    candidates: Sequence[ContextualTablaBenchmarkRecordV1],
) -> ContextualTablaSelectionV1:
    """Select the fastest fully accepted material end-to-end improvement."""
    if not isinstance(baseline, ContextualTablaBenchmarkRecordV1):
        raise ValueError("baseline must be ContextualTablaBenchmarkRecordV1.")
    baseline_reasons = baseline.selection_ineligibility_reasons()
    if baseline_reasons:
        raise ValueError(
            "Baseline is not a measured accepted production record: "
            + ", ".join(baseline_reasons)
        )
    baseline_wall = _measured_positive_scalar(baseline, "end_to_end_wall_seconds")
    assert baseline_wall is not None
    records = tuple(candidates)
    if any(not isinstance(item, ContextualTablaBenchmarkRecordV1) for item in records):
        raise ValueError("Every candidate must be ContextualTablaBenchmarkRecordV1.")
    run_ids = [item.run_id for item in records]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Candidate run_id values must be unique.")

    evaluations: dict[
        str, tuple[ContextualTablaBenchmarkRecordV1, float | None, list[str]]
    ] = {}
    eligible: list[tuple[float, str, str, ContextualTablaBenchmarkRecordV1]] = []
    for candidate in records:
        reasons = _candidate_reasons(baseline, candidate)
        candidate_wall = _measured_positive_scalar(candidate, "end_to_end_wall_seconds")
        speedup = None if candidate_wall is None else baseline_wall / candidate_wall
        if not reasons and speedup is not None:
            if speedup < candidate.benchmark_speedup_threshold:
                reasons.append("material_end_to_end_speedup_not_met")
            else:
                eligible.append(
                    (
                        candidate_wall,
                        candidate.plan.plan_sha256,
                        candidate.run_id,
                        candidate,
                    )
                )
        evaluations[candidate.run_id] = (candidate, speedup, reasons)

    selected = min(eligible, default=None, key=lambda item: item[:3])
    selected_record = baseline if selected is None else selected[3]
    decisions: list[ContextualTablaCandidateDecisionV1] = []
    for run_id in sorted(evaluations):
        candidate, speedup, reasons = evaluations[run_id]
        is_selected = selected is not None and candidate is selected_record
        if not is_selected and not reasons:
            assert selected is not None
            if speedup == baseline_wall / selected[0]:
                reasons.append("deterministic_tie_break_lost")
            else:
                reasons.append("not_fastest_qualified_candidate")
        decisions.append(
            ContextualTablaCandidateDecisionV1(
                run_id=run_id,
                selected=is_selected,
                speedup=speedup,
                rejection_reasons=tuple(reasons),
            )
        )
    selected_wall = _measured_positive_scalar(
        selected_record, "end_to_end_wall_seconds"
    )
    assert selected_wall is not None
    return ContextualTablaSelectionV1(
        baseline_run_id=baseline.run_id,
        selected_run_id=selected_record.run_id,
        selected_plan=selected_record.plan,
        selected_record_sha256=selected_record.record_sha256,
        selected_is_baseline=selected is None,
        selected_speedup=baseline_wall / selected_wall,
        candidate_decisions=tuple(decisions),
    )
