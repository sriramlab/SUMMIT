"""Contracts for the generalized per-variant GxE LD-score estimator.

This module owns the estimator identity, global variant/probe counter stream,
work/memory planning, and two-pass accounting. Inference artifacts and their
post-hoc block reductions live in ``generalized_gxe_reference_v1``.
It deliberately contains no genotype traversal or LD-score kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

GENERALIZED_GXE_VARIANT_REFERENCE_KIND = (
    "summit.generalized_gxe.variant_ldscore_reference"
)
GENERALIZED_GXE_VARIANT_SCHEMA_VERSION = 1
GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT = (
    "generalized_gxe_variant_ldscore_v1"
)
GENERALIZED_GXE_VARIANT_ESTIMATOR_FAMILY = (
    "variant_probe_two_pass_per_variant_ldscore"
)
GENERALIZED_GXE_VARIANT_PROBE_ALGORITHM = (
    "counter_global_variant_global_probe_v1"
)
GENERALIZED_GXE_VARIANT_PROBE_NAMESPACE = (
    "summit.generalized_gxe.variant_ldscore.reference"
)
GENERALIZED_GXE_VARIANT_FEATURE_CONVENTION = (
    "raw_projected_common_genotype_scale_v1"
)
GENERALIZED_GXE_VARIANT_NORMAL_ASSEMBLY = (
    "symmetrized_directional_variant_ldscore_v1"
)
_MASK64 = (1 << 64) - 1
_MAX_INT64 = (1 << 63) - 1
_MIX_VARIANT = 0xD2B74407B1CE6E93
_MIX_PROBE = 0xCA5A826395121157
_MIX_ROOT = 0x9E3779B97F4A7C15
def _require_positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if value > _MAX_INT64:
        raise OverflowError(f"{name} exceeds signed 64-bit range")
    return value


def _checked_product(name: str, *values: int) -> int:
    result = 1
    for value in values:
        _require_positive_int(name, value, allow_zero=True)
        result *= value
        if result > _MAX_INT64:
            raise OverflowError(f"{name} exceeds signed 64-bit range")
    return result


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def probe_namespace_key(namespace: str) -> int:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("probe namespace must be a nonempty string")
    # This is only a deterministic PRNG stream selector. FNV-1a keeps the
    # namespace-to-stream mapping stable across processes and languages.
    value = 0xCBF29CE484222325
    for byte in namespace.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & _MASK64
    return value


def global_probe_sign(
    *,
    root_seed: int,
    global_variant_index: int,
    global_probe_index: int,
    namespace_key: int,
) -> float:
    """Return one addressable Rademacher sign using fixed uint64 arithmetic."""
    for name, value in (
        ("root_seed", root_seed),
        ("global_variant_index", global_variant_index),
        ("global_probe_index", global_probe_index),
        ("namespace_key", namespace_key),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0 or value > _MASK64:
            raise ValueError(f"{name} must be in uint64 range")
    state = _splitmix64((root_seed ^ namespace_key ^ _MIX_ROOT) & _MASK64)
    state ^= _splitmix64((global_variant_index + _MIX_VARIANT) & _MASK64)
    state ^= _splitmix64((global_probe_index + _MIX_PROBE) & _MASK64)
    return 1.0 if (_splitmix64(state) >> 63) else -1.0


def generate_global_variant_probes(
    variant_indices: Sequence[int] | np.ndarray,
    probe_indices: Sequence[int] | np.ndarray,
    *,
    root_seed: int,
    namespace: str = GENERALIZED_GXE_VARIANT_PROBE_NAMESPACE,
) -> np.ndarray:
    """Generate a Fortran-order matrix indexed only by global logical axes."""
    variants = np.asarray(variant_indices)
    probes = np.asarray(probe_indices)
    if variants.ndim != 1 or probes.ndim != 1:
        raise ValueError("variant_indices and probe_indices must be one-dimensional")
    if variants.size < 1 or probes.size < 1:
        raise ValueError("at least one variant and one probe are required")
    if variants.dtype.kind not in "iu" or probes.dtype.kind not in "iu":
        raise ValueError("variant_indices and probe_indices must be integers")
    variant_values = [int(value) for value in variants]
    probe_values = [int(value) for value in probes]
    if any(value < 0 or value > _MASK64 for value in variant_values):
        raise ValueError("variant_indices must be in uint64 range")
    if any(value < 0 or value > _MASK64 for value in probe_values):
        raise ValueError("probe_indices must be in uint64 range")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise ValueError("root_seed must be an integer")
    if root_seed < 0 or root_seed > _MASK64:
        raise ValueError("root_seed must be in uint64 range")
    namespace_value = probe_namespace_key(namespace)
    output = np.empty(
        (len(variant_values), len(probe_values)), dtype=np.float64, order="F"
    )
    for column, probe in enumerate(probe_values):
        for row, variant in enumerate(variant_values):
            output[row, column] = global_probe_sign(
                root_seed=root_seed,
                global_variant_index=variant,
                global_probe_index=probe,
                namespace_key=namespace_value,
            )
    return output


@dataclass(frozen=True)
class GlobalVariantProbeSpec:
    root_seed: int
    probe_offset: int
    probe_count: int
    namespace: str = GENERALIZED_GXE_VARIANT_PROBE_NAMESPACE

    def __post_init__(self) -> None:
        for name in ("root_seed", "probe_offset", "probe_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.root_seed < 0 or self.root_seed > _MASK64:
            raise ValueError("root_seed must be in uint64 range")
        if self.probe_offset < 0 or self.probe_count < 1:
            raise ValueError("probe_offset must be nonnegative and probe_count positive")
        if self.probe_offset + self.probe_count - 1 > _MASK64:
            raise OverflowError("probe range exceeds uint64")
        probe_namespace_key(self.namespace)

    @property
    def namespace_key(self) -> int:
        return probe_namespace_key(self.namespace)

    @property
    def probe_indices(self) -> np.ndarray:
        return np.arange(
            self.probe_offset,
            self.probe_offset + self.probe_count,
            dtype=np.uint64,
        )

    def generate(self, variant_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        return generate_global_variant_probes(
            variant_indices,
            self.probe_indices,
            root_seed=self.root_seed,
            namespace=self.namespace,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "distribution": "rademacher",
            "algorithm": GENERALIZED_GXE_VARIANT_PROBE_ALGORITHM,
            "root_seed": self.root_seed,
            "probe_offset": self.probe_offset,
            "probe_count": self.probe_count,
            "variant_index_space": "retained_ordered_variant_axis_v1",
            "tile_invariant": True,
            "shared_with_same_person": True,
            "stream_namespace": self.namespace,
            "stream_namespace_key_uint64": self.namespace_key,
        }


def native_global_variant_probes(
    variant_indices: Sequence[int] | np.ndarray,
    probe_indices: Sequence[int] | np.ndarray,
    *,
    root_seed: int,
    namespace: str = GENERALIZED_GXE_VARIANT_PROBE_NAMESPACE,
    threads: int = 1,
    native_module: Any | None = None,
) -> np.ndarray:
    """Call the native implementation of the same global counter stream."""
    if native_module is None:
        from summit import gxeldcore as native_module

    function = getattr(native_module, "global_variant_rademacher", None)
    if not callable(function):
        raise RuntimeError("The loaded native extension lacks global probe support")
    variants = np.asarray(variant_indices, dtype=np.int64)
    probes = np.asarray(probe_indices, dtype=np.int64)
    return np.asarray(
        function(
            variants,
            probes,
            int(root_seed),
            probe_namespace_key(namespace),
            int(threads),
        ),
        dtype=np.float64,
        order="F",
    )


@dataclass(frozen=True)
class GeneralizedGxEPlanInputs:
    num_samples: int
    num_variants: int
    num_basis: int
    num_annotations: int
    num_probes: int
    memory_limit_bytes: int
    genotype_format: str
    threads: int = 1
    preferred_variant_block_width: int = 4096
    preferred_rhs_tile_columns: int | None = None
    rhs_policy: str = "auto"
    write_directional_panel: bool = True
    output_storage_bytes: int = 8
    headroom_fraction: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "num_samples",
            "num_variants",
            "num_basis",
            "num_annotations",
            "num_probes",
            "memory_limit_bytes",
            "threads",
            "preferred_variant_block_width",
            "output_storage_bytes",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.preferred_rhs_tile_columns is not None:
            _require_positive_int(
                "preferred_rhs_tile_columns", self.preferred_rhs_tile_columns
            )
        if self.genotype_format not in {"bed", "pgen"}:
            raise ValueError("genotype_format must be 'bed' or 'pgen'")
        if self.rhs_policy not in {"auto", "precompute", "tiled"}:
            raise ValueError("rhs_policy must be auto, precompute, or tiled")
        if self.output_storage_bytes not in {4, 8}:
            raise ValueError("output_storage_bytes must be 4 or 8")
        if not isinstance(self.write_directional_panel, bool):
            raise ValueError("write_directional_panel must be boolean")
        if (
            not np.isfinite(self.headroom_fraction)
            or self.headroom_fraction < 0.0
            or self.headroom_fraction > 1.0
        ):
            raise ValueError("headroom_fraction must be between zero and one")


@dataclass(frozen=True)
class GeneralizedGxEWorkPlan:
    dimensions: Mapping[str, int]
    work: Mapping[str, int]
    memory: Mapping[str, int]
    tiling: Mapping[str, Any]
    descriptor: Mapping[str, Any]
    ledger: Mapping[str, int]
    output_size_bytes: int
    peak_resident_bytes: int
    memory_limit_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": dict(self.dimensions),
            "work": dict(self.work),
            "memory": dict(self.memory),
            "tiling": dict(self.tiling),
            "descriptor": dict(self.descriptor),
            "ledger": dict(self.ledger),
            "output_size_bytes": self.output_size_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
        }


def _memory_candidate(
    inputs: GeneralizedGxEPlanInputs,
    *,
    variant_width: int,
    rhs_columns: int,
) -> tuple[dict[str, int], int]:
    n = inputs.num_samples
    m = inputs.num_variants
    q = inputs.num_basis
    k = inputs.num_annotations
    b = inputs.num_probes
    p = q * (q + 1) // 2
    c = k * p
    v = min(variant_width, m)
    output_buffer_rows = v
    memory = {
        "base_sources": _checked_product("base source bytes", 8, n, k, b),
        "contextual_sources": _checked_product(
            "contextual source bytes", 8, n, k, q, b
        ),
        "pass2_rhs": _checked_product("pass-2 RHS bytes", 8, n, rhs_columns),
        "decoded_genotype_block": _checked_product(
            "decoded genotype bytes", 8, n, v
        ),
        "cross_sketch_block": _checked_product(
            "cross-sketch bytes", 8, v, rhs_columns
        ),
        "pair_reduction_scratch": _checked_product(
            "pair reduction bytes", 8, v, p, p
        ),
        "same_person_sample_accumulator": _checked_product(
            "same-person accumulator bytes", 8, c, n
        ),
        "same_person_small_matrices": _checked_product(
            "same-person matrix bytes", 8, 3, c, c
        ),
        "aggregate_matrices": _checked_product(
            "aggregate matrix bytes", 8, 3, c, c
        ),
        "output_buffer": (
            _checked_product(
                "output buffer bytes",
                inputs.output_storage_bytes,
                output_buffer_rows,
                p,
                c,
            )
            if inputs.write_directional_panel
            else 0
        ),
    }
    subtotal = sum(memory.values())
    if subtotal > _MAX_INT64:
        raise OverflowError("resident memory subtotal exceeds signed 64-bit range")
    memory["allocator_headroom"] = math.ceil(
        subtotal * inputs.headroom_fraction
    )
    memory["thread_headroom"] = _checked_product(
        "thread headroom bytes", inputs.threads, 8 * 1024**2
    )
    memory["telemetry_publication_headroom"] = 64 * 1024**2
    peak = sum(memory.values())
    if peak > _MAX_INT64:
        raise OverflowError("peak resident memory exceeds signed 64-bit range")
    return memory, peak


def plan_generalized_gxe_variant_work(
    inputs: GeneralizedGxEPlanInputs,
) -> GeneralizedGxEWorkPlan:
    """Plan bounded memory without ever increasing physical descriptor passes."""
    n = inputs.num_samples
    m = inputs.num_variants
    q = inputs.num_basis
    k = inputs.num_annotations
    b = inputs.num_probes
    p = _checked_product("pair count numerator", q, q + 1) // 2
    c = _checked_product("component count", k, p)
    total_rhs_columns = _checked_product("total RHS columns", k, q, q, b)
    pass1_flops = _checked_product("pass-1 work", 2, n, m, k, b)
    pass2_flops = _checked_product("pass-2 work", 2, n, m, k, q, q, b)
    if pass1_flops + pass2_flops > _MAX_INT64:
        raise OverflowError("total leading work exceeds signed 64-bit range")
    reduction_terms = _checked_product("pair reduction work", m, k, p, p, b)
    output_size = (
        _checked_product(
            "directional output bytes",
            inputs.output_storage_bytes,
            m,
            k,
            p,
            p,
        )
        if inputs.write_directional_panel
        else 0
    )

    preferred_variant_width = min(inputs.preferred_variant_block_width, m)
    selected: tuple[int, int, bool, dict[str, int], int] | None = None
    if inputs.rhs_policy in {"auto", "precompute"}:
        memory, peak = _memory_candidate(
            inputs,
            variant_width=preferred_variant_width,
            rhs_columns=total_rhs_columns,
        )
        if peak <= inputs.memory_limit_bytes:
            selected = (
                preferred_variant_width,
                total_rhs_columns,
                True,
                memory,
                peak,
            )
        elif inputs.rhs_policy == "precompute":
            variant_width = preferred_variant_width
            while variant_width > 1 and selected is None:
                variant_width = max(1, variant_width // 2)
                memory, peak = _memory_candidate(
                    inputs,
                    variant_width=variant_width,
                    rhs_columns=total_rhs_columns,
                )
                if peak <= inputs.memory_limit_bytes:
                    selected = (
                        variant_width,
                        total_rhs_columns,
                        True,
                        memory,
                        peak,
                    )

    if selected is None and inputs.rhs_policy != "precompute":
        minimum_rhs_columns = _checked_product(
            "minimum tiled RHS columns", q, q
        )
        requested_rhs = inputs.preferred_rhs_tile_columns
        if requested_rhs is None:
            requested_rhs = min(total_rhs_columns, max(128, b))
        rhs_width = min(
            total_rhs_columns, max(minimum_rhs_columns, requested_rhs)
        )
        variant_width = preferred_variant_width
        while selected is None and variant_width >= 1:
            memory, peak = _memory_candidate(
                inputs,
                variant_width=variant_width,
                rhs_columns=rhs_width,
            )
            if peak <= inputs.memory_limit_bytes:
                selected = (variant_width, rhs_width, False, memory, peak)
                break
            if variant_width > 1:
                variant_width = max(1, variant_width // 2)
            else:
                break
        while selected is None and rhs_width > minimum_rhs_columns:
            rhs_width = max(minimum_rhs_columns, rhs_width // 2)
            memory, peak = _memory_candidate(
                inputs,
                variant_width=1,
                rhs_columns=rhs_width,
            )
            if peak <= inputs.memory_limit_bytes:
                selected = (1, rhs_width, False, memory, peak)
    if selected is None:
        raise MemoryError(
            "memory limit cannot admit fixed global sources and minimum two-pass tiles"
        )

    variant_width, rhs_width, rhs_precomputed, memory, peak = selected
    decoded_blocks = math.ceil(m / variant_width)
    descriptor_bytes_per_pass = (
        _checked_product("BED bytes", math.ceil(n / 4), m)
        if inputs.genotype_format == "bed"
        else None
    )
    return GeneralizedGxEWorkPlan(
        dimensions={
            "N": n,
            "M": m,
            "Q": q,
            "P": p,
            "K": k,
            "C": c,
            "B": b,
        },
        work={
            "pass1_flops": pass1_flops,
            "pass2_flops": pass2_flops,
            "total_leading_flops": pass1_flops + pass2_flops,
            "pair_reduction_product_terms": reduction_terms,
        },
        memory=memory,
        tiling={
            "variant_block_width": variant_width,
            "total_rhs_columns": total_rhs_columns,
            "rhs_tile_columns": rhs_width,
            "rhs_precomputed": rhs_precomputed,
            "pass1_decoded_blocks": decoded_blocks,
            "pass2_decoded_blocks": decoded_blocks,
        },
        descriptor={
            "format": inputs.genotype_format,
            "planned_complete_passes": 2,
            "planned_variant_record_visits": 2 * m,
            "estimated_bytes_per_pass": descriptor_bytes_per_pass,
        },
        ledger={
            "planned_reference_genotype_passes": 2,
            "observed_reference_genotype_passes": 0,
            "planned_retained_variant_visits": 2 * m,
            "observed_retained_variant_visits": 0,
            "duplicate_retained_variant_visits": 0,
            "pass1_decoded_blocks": 0,
            "pass2_decoded_blocks": 0,
            "retry_count": 0,
            "repair_count": 0,
            "fallback_count": 0,
            "integrity_failure_count": 0,
        },
        output_size_bytes=output_size,
        peak_resident_bytes=peak,
        memory_limit_bytes=inputs.memory_limit_bytes,
    )


@dataclass
class TwoPassLedger:
    num_variants: int
    planned_reference_genotype_passes: int = 2
    observed_reference_genotype_passes: int = 0
    planned_retained_variant_visits: int = field(init=False)
    observed_retained_variant_visits: int = 0
    duplicate_retained_variant_visits: int = 0
    pass1_decoded_blocks: int = 0
    pass2_decoded_blocks: int = 0
    retry_count: int = 0
    repair_count: int = 0
    fallback_count: int = 0
    integrity_failure_count: int = 0
    _active_pass: int | None = field(default=None, init=False, repr=False)
    _next_variant: dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0}, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_positive_int("num_variants", self.num_variants)
        if self.planned_reference_genotype_passes != 2:
            raise ValueError("the generalized estimator always plans exactly two passes")
        self.planned_retained_variant_visits = 2 * self.num_variants

    def begin_pass(self, pass_number: int) -> None:
        if pass_number not in {1, 2}:
            raise ValueError("pass_number must be 1 or 2")
        if self._active_pass is not None:
            raise RuntimeError("a descriptor pass is already active")
        if pass_number != self.observed_reference_genotype_passes + 1:
            raise RuntimeError("descriptor passes must occur once in order")
        self._active_pass = pass_number
        self.observed_reference_genotype_passes += 1

    def record_block(self, row_start: int, row_stop: int) -> None:
        if self._active_pass is None:
            raise RuntimeError("no descriptor pass is active")
        if (
            isinstance(row_start, bool)
            or isinstance(row_stop, bool)
            or not isinstance(row_start, int)
            or not isinstance(row_stop, int)
            or row_start < 0
            or row_stop <= row_start
            or row_stop > self.num_variants
        ):
            raise ValueError("decoded block bounds are invalid")
        pass_number = self._active_pass
        expected = self._next_variant[pass_number]
        if row_start < expected:
            self.duplicate_retained_variant_visits += min(expected, row_stop) - row_start
        elif row_start > expected:
            self.integrity_failure_count += 1
            raise RuntimeError("descriptor pass skipped retained variants")
        self.observed_retained_variant_visits += row_stop - row_start
        self._next_variant[pass_number] = max(expected, row_stop)
        if pass_number == 1:
            self.pass1_decoded_blocks += 1
        else:
            self.pass2_decoded_blocks += 1

    def finish_pass(self) -> None:
        if self._active_pass is None:
            raise RuntimeError("no descriptor pass is active")
        pass_number = self._active_pass
        if self._next_variant[pass_number] != self.num_variants:
            self.integrity_failure_count += 1
            raise RuntimeError("descriptor pass ended before all variants were visited")
        self._active_pass = None

    def record_retry(self) -> None:
        self.retry_count += 1

    def record_repair(self) -> None:
        self.repair_count += 1

    def record_fallback(self) -> None:
        self.fallback_count += 1

    def record_integrity_failure(self) -> None:
        self.integrity_failure_count += 1

    def validate_pass1_barrier(self) -> None:
        """Validate the hard barrier after exactly one complete source pass."""
        if self._active_pass is not None:
            raise RuntimeError("pass 1 remains active at the source barrier")
        expected = {
            "observed_reference_genotype_passes": 1,
            "observed_retained_variant_visits": self.num_variants,
            "duplicate_retained_variant_visits": 0,
            "pass2_decoded_blocks": 0,
            "retry_count": 0,
            "fallback_count": 0,
            "integrity_failure_count": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise RuntimeError(
                    f"clean pass-1 barrier requires {name}={value}"
                )
        if self._next_variant[1] != self.num_variants:
            raise RuntimeError("pass-1 barrier precedes the final retained variant")
        if self._next_variant[2] != 0:
            raise RuntimeError("pass 2 started before the pass-1 barrier")

    def validate_clean_completion(self) -> None:
        if self._active_pass is not None:
            raise RuntimeError("a descriptor pass remains active")
        expected = {
            "observed_reference_genotype_passes": 2,
            "duplicate_retained_variant_visits": 0,
            "observed_retained_variant_visits": 2 * self.num_variants,
            "retry_count": 0,
            "fallback_count": 0,
            "integrity_failure_count": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise RuntimeError(f"clean two-pass ledger requires {name}={value}")

    def to_dict(self) -> dict[str, int]:
        return {
            "planned_reference_genotype_passes": self.planned_reference_genotype_passes,
            "observed_reference_genotype_passes": self.observed_reference_genotype_passes,
            "planned_retained_variant_visits": self.planned_retained_variant_visits,
            "observed_retained_variant_visits": self.observed_retained_variant_visits,
            "duplicate_retained_variant_visits": self.duplicate_retained_variant_visits,
            "pass1_decoded_blocks": self.pass1_decoded_blocks,
            "pass2_decoded_blocks": self.pass2_decoded_blocks,
            "retry_count": self.retry_count,
            "repair_count": self.repair_count,
            "fallback_count": self.fallback_count,
            "integrity_failure_count": self.integrity_failure_count,
        }
