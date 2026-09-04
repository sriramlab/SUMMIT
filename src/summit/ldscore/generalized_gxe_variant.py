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
_COMPONENT_PAIR_BATCH_LIMIT = 8
_COMPONENT_PAIR_BATCH_BYTE_LIMIT = 256 * 1024**2
_SOURCE_ANNOTATION_BATCH_LIMIT = 8
_SOURCE_ANNOTATION_BATCH_BYTE_LIMIT = 2 * 1024**3
_TARGET_ANNOTATION_BATCH_LIMIT = 8


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
            "shared_with_same_person": False,
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
    fixed_effect_rank: int = 0
    threads: int = 1
    preferred_variant_block_width: int = 4096
    preferred_rhs_tile_columns: int | None = None
    preferred_source_probe_tile_width: int | None = None
    preferred_source_annotation_batch_width: int | None = None
    preferred_target_annotation_batch_width: int | None = None
    rhs_policy: str = "auto"
    write_directional_panel: bool = True
    output_storage_bytes: int = 8
    component_diagonal_sample_tile_width: int = 1024
    write_composable_payload: bool = False
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
            "component_diagonal_sample_tile_width",
        ):
            _require_positive_int(name, getattr(self, name))
        _require_positive_int(
            "fixed_effect_rank", self.fixed_effect_rank, allow_zero=True
        )
        if self.fixed_effect_rank >= self.num_samples:
            raise ValueError("fixed_effect_rank must be smaller than num_samples")
        if self.preferred_rhs_tile_columns is not None:
            _require_positive_int(
                "preferred_rhs_tile_columns", self.preferred_rhs_tile_columns
            )
        if self.preferred_source_probe_tile_width is not None:
            _require_positive_int(
                "preferred_source_probe_tile_width",
                self.preferred_source_probe_tile_width,
            )
            if self.preferred_source_probe_tile_width > self.num_probes:
                raise ValueError(
                    "preferred_source_probe_tile_width exceeds num_probes"
                )
        for name in (
            "preferred_source_annotation_batch_width",
            "preferred_target_annotation_batch_width",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_positive_int(name, value)
                if value > self.num_annotations:
                    raise ValueError(f"{name} exceeds num_annotations")
        if self.genotype_format not in {"bed", "pgen"}:
            raise ValueError("genotype_format must be 'bed' or 'pgen'")
        if self.rhs_policy not in {"auto", "precompute", "tiled"}:
            raise ValueError("rhs_policy must be auto, precompute, or tiled")
        if self.output_storage_bytes not in {4, 8}:
            raise ValueError("output_storage_bytes must be 4 or 8")
        if not isinstance(self.write_directional_panel, bool):
            raise ValueError("write_directional_panel must be boolean")
        if not isinstance(self.write_composable_payload, bool):
            raise ValueError("write_composable_payload must be boolean")
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
    rhs_resident_columns: int,
    target_tile_columns: int,
    source_probe_tile_width: int,
    source_annotation_batch_width: int,
    target_annotation_batch_width: int,
) -> tuple[dict[str, int], int]:
    n = inputs.num_samples
    m = inputs.num_variants
    q = inputs.num_basis
    k = inputs.num_annotations
    b = inputs.num_probes
    p = q * (q + 1) // 2
    c = k * p
    v = min(variant_width, m)
    sample_tile = min(inputs.component_diagonal_sample_tile_width, n)
    one_pair_tile = _checked_product(
        "component diagonal pair tile bytes", 8, sample_tile, v
    )
    pair_batch_width = min(
        p,
        _COMPONENT_PAIR_BATCH_LIMIT,
        max(1, _COMPONENT_PAIR_BATCH_BYTE_LIMIT // one_pair_tile),
    )
    memory = {
        "base_sources": _checked_product("base source bytes", 8, n, k, b),
        "contextual_sources": _checked_product(
            "contextual source bytes", 8, n, k, q, b
        ),
        "pass1_probe_signs": _checked_product(
            "pass-1 probe bytes", 8, v, source_probe_tile_width
        ),
        "pass1_source_rhs": _checked_product(
            "pass-1 source RHS bytes",
            8,
            v,
            source_probe_tile_width,
            source_annotation_batch_width,
        ),
        "pass1_source_contribution": _checked_product(
            "pass-1 source contribution bytes",
            8,
            n,
            source_probe_tile_width,
            source_annotation_batch_width,
        ),
        "pass2_rhs": _checked_product(
            "pass-2 RHS bytes", 8, n, rhs_resident_columns
        ),
        "decoded_genotype_block": _checked_product(
            "decoded genotype bytes", 8, n, v
        ),
        # A proven single-nonzero annotation layout may compact one bin's
        # genotype columns before its source GEMM.  The planner does not
        # inspect annotation values, so reserve the conservative all-variants
        # upper bound; overlapping annotations do not materialize this arena.
        "singleton_annotation_source_genotype": (
            _checked_product(
                "singleton annotation source genotype bytes", 8, n, v
            )
            if inputs.genotype_format == "bed" and k > 1
            else 0
        ),
        "cross_sketch_block": _checked_product(
            "cross-sketch bytes",
            8,
            v,
            target_tile_columns,
            target_annotation_batch_width,
        ),
        "pair_reduction_scratch": _checked_product(
            "pair reduction bytes",
            8,
            v,
            p,
            p,
            target_annotation_batch_width,
        ),
        "pair_reduction_tile_scratch": _checked_product(
            "pair reduction tile bytes",
            8,
            v,
            p,
            p,
            target_annotation_batch_width,
        ),
        "target_row_block": _checked_product(
            "target row block bytes", 8, v, p, c
        ),
        "component_kernel_diagonal": _checked_product(
            "component kernel diagonal bytes", 8, c, n
        ),
        "component_diagonal_weighted_fixed_basis": _checked_product(
            "component diagonal weighted fixed basis bytes",
            8,
            n,
            q,
            inputs.fixed_effect_rank,
        ),
        "component_diagonal_projection_coefficients": _checked_product(
            "component diagonal projection bytes",
            8,
            q,
            inputs.fixed_effect_rank,
            v,
        ),
        "component_diagonal_fixed_tile": _checked_product(
            "component diagonal fixed tile bytes",
            8,
            sample_tile,
            inputs.fixed_effect_rank,
        ),
        "component_diagonal_feature_tile": _checked_product(
            "component diagonal feature tile bytes", 8, q, sample_tile, v
        ),
        "component_diagonal_genotype_tile": _checked_product(
            "component diagonal genotype tile bytes", 8, sample_tile, v
        ),
        "component_diagonal_pair_tile": _checked_product(
            "component diagonal pair tile bytes",
            one_pair_tile,
            pair_batch_width,
        ),
        "component_diagonal_annotation_product": _checked_product(
            "component diagonal annotation product bytes",
            8,
            sample_tile,
            k,
            pair_batch_width,
        ),
        "same_person_small_matrices": _checked_product(
            "same-person matrix bytes", 8, 3, c, c
        ),
        "aggregate_matrices": _checked_product(
            "aggregate matrix bytes", 8, 3, c, c
        ),
        "directional_panel": _checked_product(
            "directional panel bytes", 8, m, p, c
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
    reduction_terms = _checked_product(
        "pair reduction product terms", m, k, b, q, q, q, q
    )
    barrier_projection_flops = _checked_product(
        "barrier projection work", 6, n, k, q, b, inputs.fixed_effect_rank
    )
    component_residualization_flops = _checked_product(
        "component residualization work",
        4,
        n,
        m,
        q,
        inputs.fixed_effect_rank,
    )
    component_annotation_flops = _checked_product(
        "component annotation work", 2, n, m, k, p
    )
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
    if inputs.write_composable_payload:
        output_size += _checked_product(
            "composable annotation output bytes", 8, m, k
        )
        output_size += _checked_product(
            "composable component diagonal output bytes", 8, c, n
        )

    preferred_variant_width = min(inputs.preferred_variant_block_width, m)
    requested_rhs_columns = inputs.preferred_rhs_tile_columns
    if requested_rhs_columns is None:
        requested_rhs_columns = min(total_rhs_columns, max(128, b))
    preferred_probe_tile_width = min(
        b, max(1, requested_rhs_columns // (q * q))
    )
    preferred_source_probe_tile_width = (
        preferred_probe_tile_width
        if inputs.preferred_source_probe_tile_width is None
        else inputs.preferred_source_probe_tile_width
    )

    def source_batch_limit(probe_tile_width: int) -> int:
        one_annotation_output = _checked_product(
            "one-annotation source output bytes", 8, n, probe_tile_width
        )
        return min(
            k,
            (
                _SOURCE_ANNOTATION_BATCH_LIMIT
                if inputs.preferred_source_annotation_batch_width is None
                else inputs.preferred_source_annotation_batch_width
            ),
            max(1, _SOURCE_ANNOTATION_BATCH_BYTE_LIMIT // one_annotation_output),
        )

    def candidate_at(
        *,
        variant_width: int,
        probe_tile_width: int,
        source_probe_tile_width: int,
        rhs_precomputed: bool,
    ) -> tuple[int, int, int, int, int, bool, dict[str, int], int] | None:
        target_tile_columns = _checked_product(
            "target tile columns", q, q, probe_tile_width
        )
        maximum_target_batch_width = min(
            k,
            (
                _TARGET_ANNOTATION_BATCH_LIMIT
                if inputs.preferred_target_annotation_batch_width is None
                else inputs.preferred_target_annotation_batch_width
            ),
        )
        for target_batch_width in range(maximum_target_batch_width, 0, -1):
            rhs_resident_columns = (
                total_rhs_columns
                if rhs_precomputed
                else target_tile_columns * target_batch_width
            )
            for source_batch_width in range(
                source_batch_limit(source_probe_tile_width), 0, -1
            ):
                memory, peak = _memory_candidate(
                    inputs,
                    variant_width=variant_width,
                    rhs_resident_columns=rhs_resident_columns,
                    target_tile_columns=target_tile_columns,
                    source_probe_tile_width=source_probe_tile_width,
                    source_annotation_batch_width=source_batch_width,
                    target_annotation_batch_width=target_batch_width,
                )
                if peak <= inputs.memory_limit_bytes:
                    return (
                        variant_width,
                        probe_tile_width,
                        source_probe_tile_width,
                        source_batch_width,
                        target_batch_width,
                        rhs_precomputed,
                        memory,
                        peak,
                    )
        return None

    selected: tuple[
        int, int, int, int, int, bool, dict[str, int], int
    ] | None = None
    if inputs.rhs_policy in {"auto", "precompute"}:
        selected = candidate_at(
            variant_width=preferred_variant_width,
            probe_tile_width=preferred_probe_tile_width,
            source_probe_tile_width=preferred_source_probe_tile_width,
            rhs_precomputed=True,
        )
        if selected is None and inputs.rhs_policy == "precompute":
            variant_width = preferred_variant_width
            probe_tile_width = preferred_probe_tile_width
            source_probe_tile_width = preferred_source_probe_tile_width
            while selected is None:
                if variant_width > 1:
                    variant_width = max(1, variant_width // 2)
                elif probe_tile_width > 1:
                    probe_tile_width = max(1, probe_tile_width // 2)
                elif source_probe_tile_width > 1:
                    source_probe_tile_width = max(
                        1, source_probe_tile_width // 2
                    )
                else:
                    break
                selected = candidate_at(
                    variant_width=variant_width,
                    probe_tile_width=probe_tile_width,
                    source_probe_tile_width=source_probe_tile_width,
                    rhs_precomputed=True,
                )

    if selected is None and inputs.rhs_policy != "precompute":
        probe_tile_width = preferred_probe_tile_width
        source_probe_tile_width = preferred_source_probe_tile_width
        variant_width = preferred_variant_width
        while selected is None and variant_width >= 1:
            selected = candidate_at(
                variant_width=variant_width,
                probe_tile_width=probe_tile_width,
                source_probe_tile_width=source_probe_tile_width,
                rhs_precomputed=False,
            )
            if selected is not None:
                break
            if variant_width > 1:
                variant_width = max(1, variant_width // 2)
            else:
                break
        while selected is None and probe_tile_width > 1:
            probe_tile_width = max(1, probe_tile_width // 2)
            selected = candidate_at(
                variant_width=1,
                probe_tile_width=probe_tile_width,
                source_probe_tile_width=source_probe_tile_width,
                rhs_precomputed=False,
            )
        while selected is None and source_probe_tile_width > 1:
            source_probe_tile_width = max(1, source_probe_tile_width // 2)
            selected = candidate_at(
                variant_width=1,
                probe_tile_width=1,
                source_probe_tile_width=source_probe_tile_width,
                rhs_precomputed=False,
            )
    if selected is None:
        raise MemoryError(
            "memory limit cannot admit fixed global sources and minimum two-pass tiles"
        )

    (
        variant_width,
        probe_tile_width,
        source_probe_tile_width,
        source_annotation_batch_width,
        target_annotation_batch_width,
        rhs_precomputed,
        memory,
        peak,
    ) = selected
    rhs_width = _checked_product(
        "selected target tile columns", q, q, probe_tile_width
    )
    decoded_blocks = math.ceil(m / variant_width)
    pass1_rhs_products = _checked_product(
        "pass-1 RHS products", m, k, b
    )
    pass2_rhs_products = _checked_product(
        "pass-2 RHS products",
        n,
        k,
        q,
        q,
        b,
        1 if rhs_precomputed else decoded_blocks,
    )
    target_cross_scale_products = _checked_product(
        "target cross scaling products", m, k, q, q, b
    )
    pair_reduction_flops = _checked_product(
        "pair reduction flops", 2, reduction_terms
    )
    directed_aggregation_flops = _checked_product(
        "directed aggregation flops", 2, m, k, k, p, p
    )
    total_estimated_flops = sum(
        (
            pass1_flops,
            pass2_flops,
            barrier_projection_flops,
            component_residualization_flops,
            component_annotation_flops,
            pair_reduction_flops,
            directed_aggregation_flops,
        )
    )
    if total_estimated_flops > _MAX_INT64:
        raise OverflowError("total estimated work exceeds signed 64-bit range")
    selected_sample_tile = min(
        inputs.component_diagonal_sample_tile_width, n
    )
    selected_one_pair_tile = _checked_product(
        "selected component pair tile bytes",
        8,
        selected_sample_tile,
        min(variant_width, m),
    )
    component_pair_batch_width = (
        memory["component_diagonal_pair_tile"] // selected_one_pair_tile
    )
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
            "barrier_projection_flops": barrier_projection_flops,
            "component_residualization_flops": (
                component_residualization_flops
            ),
            "component_annotation_flops": component_annotation_flops,
            "pair_reduction_product_terms": reduction_terms,
            "pair_reduction_flops": pair_reduction_flops,
            "directed_aggregation_flops": directed_aggregation_flops,
            "total_estimated_flops": total_estimated_flops,
            "pass1_rhs_products": pass1_rhs_products,
            "pass2_rhs_products": pass2_rhs_products,
            "target_cross_scale_products": target_cross_scale_products,
        },
        memory=memory,
        tiling={
            "variant_block_width": variant_width,
            "total_rhs_columns": total_rhs_columns,
            "rhs_tile_columns": rhs_width,
            "rhs_precomputed": rhs_precomputed,
            "probe_tile_width": probe_tile_width,
            "source_probe_tile_width": source_probe_tile_width,
            "source_annotation_batch_width": source_annotation_batch_width,
            "target_annotation_batch_width": target_annotation_batch_width,
            "component_pair_batch_width": component_pair_batch_width,
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
