"""Contracts for the generalized per-variant GxE LD-score estimator.

This module owns the estimator identity, V1 artifact validation, global
variant/probe counter stream, work/memory planning, and two-pass accounting.
It deliberately contains no genotype traversal or LD-score kernel.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_json,
)
from summit.context.schema import GenotypeScalePlanV1, GenotypeScalePolicy


GENERALIZED_GXE_VARIANT_REFERENCE_KIND = (
    "summit.generalized_gxe.variant_ldscore_reference"
)
GENERALIZED_GXE_VARIANT_SCHEMA_VERSION = 1
GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT = (
    "generalized_gxe_variant_ldscore_v1"
)
GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD = (
    "frozen_full_genome_variant_ldscore_delete_block_v1"
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
GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE = "reuse_full_same_person_v1"

_SAMPLE_PROBE_CONTEXTUAL_KIND = "summit.context.reference.v1"
_SAMPLE_PROBE_CONTEXTUAL_DEVELOPMENT_KIND = "summit.context.reference"
_LEGACY_GXE_REFERENCE_KIND = "summit.gxe.reference"
_LEGACY_JACKKNIFE_ALIAS = "block_local_ldscore_deletion"
_MASK64 = (1 << 64) - 1
_MAX_INT64 = (1 << 63) - 1
_MIX_VARIANT = 0xD2B74407B1CE6E93
_MIX_PROBE = 0xCA5A826395121157
_MIX_ROOT = 0x9E3779B97F4A7C15
_FINGERPRINT_VARIANTS = (0, 1, 2, 7, 31)
_REQUIRED_ARRAYS = frozenset(
    {
        "directed_numerator",
        "symmetric_numerator",
        "genetic_gram",
        "block_directed_numerator",
        "block_annotation_mass",
        "same_person",
    }
)
_OPTIONAL_ARRAYS = frozenset(
    {"deleted_genetic_gram", "directional_ldscores"}
)
_REQUIRED_DIAGNOSTICS = frozenset(
    {
        "maximum_source_projection_leakage",
        "maximum_presymmetry_absolute_error",
        "maximum_presymmetry_relative_error",
        "block_reconstruction_error",
        "same_person_probe_count",
        "same_person_cross_tile_finalized",
        "minimum_annotation_mass",
        "minimum_deleted_annotation_mass",
        "all_values_finite",
        "normal_matrix_rank",
        "normal_matrix_condition",
        "dense_oracle_fixture_version",
        "backend_fixed_probe_maximum_error",
    }
)


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_positive_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if value > _MAX_INT64:
        raise OverflowError(f"{name} exceeds signed 64-bit range")
    return value


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


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
    payload = (
        b"summit-counter-global-variant-global-probe-v1\0"
        + namespace.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


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

    def fingerprint_record(self) -> dict[str, Any]:
        relative_probes = tuple(
            index for index in (0, 1, 3, 7) if index < self.probe_count
        )
        probe_indices = np.asarray(
            [self.probe_offset + value for value in relative_probes],
            dtype=np.uint64,
        )
        variants = np.asarray(_FINGERPRINT_VARIANTS, dtype=np.uint64)
        signs = self.generate(variants)[:, list(relative_probes)]
        signs_int8 = np.asarray(signs, dtype=np.int8, order="C")
        return {
            "variant_indices": list(_FINGERPRINT_VARIANTS),
            "probe_indices": [int(value) for value in probe_indices],
            "shape": list(signs_int8.shape),
            "dtype": "int8",
            "sha256": array_sha256(signs_int8),
        }

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
            "fingerprint": self.fingerprint_record(),
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


def serialize_generalized_gxe_axes(
    *,
    num_variants: int,
    variant_digest: str,
    retained_variant_digest: str,
    num_samples: int,
    sample_digest: str,
    basis_names: Sequence[str],
    basis_digest: str,
    basis_calibration_digest: str,
    fixed_effect_digest: str,
    fixed_effect_rank: int,
    annotation_names: Sequence[str],
    annotation_digest: str,
    annotation_masses: Sequence[float] | np.ndarray,
    variant_block_ids: Sequence[int] | np.ndarray,
    block_labels: Sequence[str],
    jackknife_block_digest: str,
    residual_component_names: Sequence[str],
) -> dict[str, Any]:
    """Serialize ordered axes using the existing contextual indexes."""
    n_variants = _require_positive_int("num_variants", num_variants)
    n_samples = _require_positive_int("num_samples", num_samples)
    names = tuple(str(name) for name in basis_names)
    annotations = tuple(str(name) for name in annotation_names)
    residual_names = tuple(str(name) for name in residual_component_names)
    if not names or not annotations or not residual_names:
        raise ValueError("basis, annotation, and residual component axes must be nonempty")
    pair_index = ContextPairIndex(num_basis=len(names))
    component_index = ContextComponentIndex(
        annotation_names=annotations,
        pair_index=pair_index,
    )
    masses = np.asarray(annotation_masses, dtype=np.float64)
    if masses.shape != (len(annotations),) or not np.all(np.isfinite(masses)):
        raise ValueError("annotation_masses have the wrong shape or are nonfinite")
    if np.any(masses <= 0.0):
        raise ValueError("annotation_masses must be positive")
    blocks = np.asarray(variant_block_ids)
    labels = tuple(str(label) for label in block_labels)
    if blocks.shape != (n_variants,) or blocks.dtype.kind not in "iu":
        raise ValueError("variant_block_ids must be an integer vector of length M")
    block_values = [int(value) for value in blocks]
    if any(value < 0 for value in block_values):
        raise ValueError("variant_block_ids must be nonnegative")
    block_count = max(block_values) + 1
    if set(block_values) != set(range(block_count)) or len(labels) != block_count:
        raise ValueError("block IDs must be contiguous and match block_labels")
    if len(set(labels)) != len(labels):
        raise ValueError("block_labels must be unique")
    return {
        "variants": {
            "count": n_variants,
            "digest": _require_sha256("variant_digest", variant_digest),
            "retained_order_digest": _require_sha256(
                "retained_variant_digest", retained_variant_digest
            ),
        },
        "samples": {
            "count": n_samples,
            "digest": _require_sha256("sample_digest", sample_digest),
        },
        "basis": {
            "names": list(names),
            "digest": _require_sha256("basis_digest", basis_digest),
            "calibration_digest": _require_sha256(
                "basis_calibration_digest", basis_calibration_digest
            ),
        },
        "fixed_effects": {
            "digest": _require_sha256(
                "fixed_effect_digest", fixed_effect_digest
            ),
            "rank": _require_positive_int(
                "fixed_effect_rank", fixed_effect_rank, allow_zero=True
            ),
            "residual_rank": n_samples - fixed_effect_rank,
        },
        "pairs": {
            "table": [[entry.q, entry.r] for entry in pair_index.entries],
            "digest": pair_index.digest,
            "serialization": pair_index.to_dict(),
        },
        "annotations": {
            "names": list(annotations),
            "digest": _require_sha256("annotation_digest", annotation_digest),
            "masses": [float(value) for value in masses],
        },
        "components": {
            "table": [
                [entry.annotation_index, entry.pair_index]
                for entry in component_index.entries
            ],
            "digest": component_index.digest,
            "serialization": component_index.to_dict(),
        },
        "jackknife_blocks": {
            "variant_block_ids": block_values,
            "block_labels": list(labels),
            "digest": _require_sha256(
                "jackknife_block_digest", jackknife_block_digest
            ),
        },
        "residual_components": {"names": list(residual_names)},
    }


def numeric_array_metadata(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise ValueError("generalized reference numeric arrays must be FP64")
    if not np.all(np.isfinite(array)):
        raise ValueError("generalized reference numeric arrays must be finite")
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": "float64",
        "order": "C",
        "sha256": array_sha256(contiguous),
    }


def build_generalized_gxe_variant_manifest(
    *,
    axes: Mapping[str, Any],
    probe_spec: GlobalVariantProbeSpec,
    arrays: Mapping[str, np.ndarray],
    pass_ledger: Mapping[str, Any],
    genotype_scale_plan: GenotypeScalePlanV1,
    performance_ledger: Mapping[str, Any],
    provenance: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(genotype_scale_plan, GenotypeScalePlanV1):
        raise ValueError("genotype_scale_plan must be a GenotypeScalePlanV1")
    payload = {
        "kind": GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
        "schema_version": GENERALIZED_GXE_VARIANT_SCHEMA_VERSION,
        "scientific_contract": GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
        "estimator_family": GENERALIZED_GXE_VARIANT_ESTIMATOR_FAMILY,
        "probe_axis": "variant",
        "feature_convention": GENERALIZED_GXE_VARIANT_FEATURE_CONVENTION,
        "normal_equation_assembly": GENERALIZED_GXE_VARIANT_NORMAL_ASSEMBLY,
        "jackknife_method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
        "same_person_jackknife": GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE,
        "axes": dict(axes),
        "randomization": probe_spec.to_metadata(),
        "genotype_scale_plan": genotype_scale_plan.to_dict(),
        "genotype_scale_plan_sha256": genotype_scale_plan.digest,
        "jackknife": {
            "method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
            "num_blocks": len(axes["jackknife_blocks"]["block_labels"]),
            "block_labels": list(axes["jackknife_blocks"]["block_labels"]),
            "block_axis": "ordered_retained_variants",
            "source_scores_recomputed": False,
            "retained_ldscores_frozen": True,
            "local_context_weighted_ld_assumption": True,
            "same_person_deletion": GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE,
        },
        "numeric_arrays": {
            name: numeric_array_metadata(value) for name, value in arrays.items()
        },
        "pass_ledger": dict(pass_ledger),
        "performance_ledger": dict(performance_ledger),
        "provenance": dict(provenance),
        "diagnostics": dict(diagnostics),
        "per_variant_panel": (
            {
                "storage": "inline_npz",
                "logical_layout": (
                    "variant_target_pair_source_component_c"
                ),
                "logical_compute_dtype": "float64",
                "array": "directional_ldscores",
                **numeric_array_metadata(arrays["directional_ldscores"]),
            }
            if "directional_ldscores" in arrays
            else {
                "storage": "omitted",
                "logical_layout": (
                    "variant_target_pair_source_component_c"
                ),
                "logical_compute_dtype": "float64",
            }
        ),
        "terminal_status": "complete",
    }
    return validate_generalized_gxe_variant_manifest(payload, arrays=arrays)


def _scale_plan_from_manifest(value: Any) -> GenotypeScalePlanV1:
    record = dict(_require_mapping("genotype_scale_plan", value))
    required = {
        "policy",
        "retained_variant_order_sha256",
        "allele_orientation",
        "allele_coding",
        "centering_source",
        "centering_formula",
        "scaling_formula",
        "missing_imputation",
        "ploidy_policy",
        "affine_mean_sha256",
        "affine_inverse_scale_sha256",
    }
    if set(record) != required:
        raise ValueError("genotype scale plan has a noncanonical field set")
    try:
        policy = GenotypeScalePolicy(record.pop("policy"))
    except (TypeError, ValueError) as exc:
        raise ValueError("genotype scale plan policy is unsupported") from exc
    return GenotypeScalePlanV1(policy=policy, **record)


def _validate_performance_ledger(value: Any) -> dict[str, Any]:
    ledger = dict(_require_mapping("performance_ledger", value))
    required = {
        "backend",
        "threads",
        "affinity",
        "numa_evidence",
        "phase_wall_seconds",
        "phase_cpu_seconds",
        "bytes_read",
        "gemm_dimensions",
        "peak_rss_bytes",
        "output_bytes",
    }
    if set(ledger) != required:
        raise ValueError("performance ledger has a noncanonical field set")
    if not isinstance(ledger["backend"], str) or not ledger["backend"]:
        raise ValueError("performance backend must be nonempty text")
    _require_positive_int("performance threads", ledger["threads"])
    phases = {"pass1", "barrier", "pass2", "finalize"}
    for field in ("phase_wall_seconds", "phase_cpu_seconds"):
        values = _require_mapping(f"performance_ledger.{field}", ledger[field])
        if set(values) != phases:
            raise ValueError(f"performance {field} has the wrong phase set")
        for phase, duration in values.items():
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or float(duration) < 0.0
            ):
                raise ValueError(f"performance {field}.{phase} is invalid")
    for field in ("bytes_read", "peak_rss_bytes", "output_bytes"):
        _require_positive_int(field, ledger[field], allow_zero=True)
    if not isinstance(ledger["gemm_dimensions"], list):
        raise ValueError("performance gemm_dimensions must be an array")
    _require_mapping("performance affinity", ledger["affinity"])
    _require_mapping("performance NUMA evidence", ledger["numa_evidence"])
    canonical_json(ledger)
    return ledger


def _validate_axes(axes_value: Any) -> dict[str, Any]:
    axes = dict(_require_mapping("axes", axes_value))
    variants = _require_mapping("axes.variants", axes.get("variants"))
    samples = _require_mapping("axes.samples", axes.get("samples"))
    basis = _require_mapping("axes.basis", axes.get("basis"))
    fixed = _require_mapping("axes.fixed_effects", axes.get("fixed_effects"))
    pairs = _require_mapping("axes.pairs", axes.get("pairs"))
    annotations = _require_mapping("axes.annotations", axes.get("annotations"))
    components = _require_mapping("axes.components", axes.get("components"))
    blocks = _require_mapping("axes.jackknife_blocks", axes.get("jackknife_blocks"))
    residual = _require_mapping(
        "axes.residual_components", axes.get("residual_components")
    )
    n_variants = _require_positive_int("variant count", variants.get("count"))
    n_samples = _require_positive_int("sample count", samples.get("count"))
    _require_sha256("variant digest", variants.get("digest"))
    _require_sha256(
        "retained variant digest", variants.get("retained_order_digest")
    )
    _require_sha256("sample digest", samples.get("digest"))
    basis_names = basis.get("names")
    if not isinstance(basis_names, list) or not basis_names:
        raise ValueError("basis names must be a nonempty ordered array")
    _require_sha256("basis digest", basis.get("digest"))
    _require_sha256("basis calibration digest", basis.get("calibration_digest"))
    fixed_rank = _require_positive_int(
        "fixed-effect rank", fixed.get("rank"), allow_zero=True
    )
    if fixed_rank >= n_samples:
        raise ValueError("fixed-effect rank must be smaller than sample count")
    if fixed.get("residual_rank") != n_samples - fixed_rank:
        raise ValueError("residual rank contradicts sample count and fixed rank")
    _require_sha256("fixed-effect digest", fixed.get("digest"))
    pair_index = ContextPairIndex(num_basis=len(basis_names))
    expected_pair_table = [[entry.q, entry.r] for entry in pair_index.entries]
    if pairs.get("table") != expected_pair_table:
        raise ValueError("pair table does not match diagonal-first schema order")
    if pairs.get("serialization") != pair_index.to_dict():
        raise ValueError("pair serialization does not match the contextual index")
    if pairs.get("digest") != pair_index.digest:
        raise ValueError("pair digest does not match the serialized pair index")
    annotation_names = annotations.get("names")
    masses = np.asarray(annotations.get("masses"), dtype=np.float64)
    if not isinstance(annotation_names, list) or not annotation_names:
        raise ValueError("annotation names must be a nonempty ordered array")
    if masses.shape != (len(annotation_names),) or not np.all(np.isfinite(masses)):
        raise ValueError("annotation masses are missing, nonfinite, or misaligned")
    if np.any(masses <= 0.0):
        raise ValueError("annotation masses must be positive")
    _require_sha256("annotation digest", annotations.get("digest"))
    component_index = ContextComponentIndex(
        annotation_names=tuple(annotation_names),
        pair_index=pair_index,
    )
    expected_component_table = [
        [entry.annotation_index, entry.pair_index]
        for entry in component_index.entries
    ]
    if components.get("table") != expected_component_table:
        raise ValueError("component table does not match annotation-major schema order")
    if components.get("serialization") != component_index.to_dict():
        raise ValueError("component serialization does not match the contextual index")
    if components.get("digest") != component_index.digest:
        raise ValueError("component digest does not match its serialization")
    block_ids = blocks.get("variant_block_ids")
    block_labels = blocks.get("block_labels")
    if not isinstance(block_ids, list) or len(block_ids) != n_variants:
        raise ValueError("variant block IDs must align to the ordered variant axis")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in block_ids
    ):
        raise ValueError("variant block IDs must be nonnegative integers")
    block_count = max(block_ids) + 1
    if set(block_ids) != set(range(block_count)):
        raise ValueError("variant block IDs must be contiguous")
    if (
        not isinstance(block_labels, list)
        or len(block_labels) != block_count
        or len(set(block_labels)) != block_count
    ):
        raise ValueError("block labels must be unique and aligned to block IDs")
    _require_sha256("jackknife block digest", blocks.get("digest"))
    residual_names = residual.get("names")
    if not isinstance(residual_names, list) or not residual_names:
        raise ValueError("residual component order must be recorded")
    return axes


def _validate_randomization(value: Any) -> dict[str, Any]:
    randomization = dict(_require_mapping("randomization", value))
    expected = {
        "distribution": "rademacher",
        "algorithm": GENERALIZED_GXE_VARIANT_PROBE_ALGORITHM,
        "variant_index_space": "retained_ordered_variant_axis_v1",
        "tile_invariant": True,
        "shared_with_same_person": True,
    }
    for key, expected_value in expected.items():
        if randomization.get(key) != expected_value:
            raise ValueError(f"randomization field {key!r} contradicts V1")
    spec = GlobalVariantProbeSpec(
        root_seed=randomization.get("root_seed"),
        probe_offset=randomization.get("probe_offset"),
        probe_count=randomization.get("probe_count"),
        namespace=randomization.get("stream_namespace"),
    )
    if randomization.get("stream_namespace_key_uint64") != spec.namespace_key:
        raise ValueError("probe namespace key does not match the namespace")
    if randomization.get("fingerprint") != spec.fingerprint_record():
        raise ValueError("probe fingerprint does not match the global counter stream")
    return randomization


def _validate_pass_ledger(value: Any, num_variants: int) -> dict[str, Any]:
    ledger = dict(_require_mapping("pass_ledger", value))
    exact_values = {
        "planned_reference_genotype_passes": 2,
        "observed_reference_genotype_passes": 2,
        "planned_retained_variant_visits": 2 * num_variants,
        "observed_retained_variant_visits": 2 * num_variants,
        "duplicate_retained_variant_visits": 0,
        "retry_count": 0,
        "fallback_count": 0,
        "integrity_failure_count": 0,
    }
    for key, expected in exact_values.items():
        if ledger.get(key) != expected:
            raise ValueError(f"published pass ledger field {key!r} must equal {expected}")
    for key in ("pass1_decoded_blocks", "pass2_decoded_blocks", "repair_count"):
        _require_positive_int(key, ledger.get(key), allow_zero=True)
    if ledger["pass1_decoded_blocks"] < 1 or ledger["pass2_decoded_blocks"] < 1:
        raise ValueError("each physical pass must decode at least one block")
    return ledger


def validate_generalized_gxe_variant_manifest(
    payload: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Validate the complete V1 reference manifest and loaded numeric arrays."""
    if not isinstance(payload, Mapping):
        raise ValueError("generalized GxE manifest must be an object")
    result = dict(payload)
    kind = result.get("kind")
    if kind in {
        _SAMPLE_PROBE_CONTEXTUAL_KIND,
        _SAMPLE_PROBE_CONTEXTUAL_DEVELOPMENT_KIND,
    }:
        raise ValueError("sample-probe contextual artifacts are not variant LD scores")
    if kind == _LEGACY_GXE_REFERENCE_KIND:
        raise ValueError("legacy non-general GxE references are not generalized artifacts")
    identity = {
        "kind": GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
        "schema_version": GENERALIZED_GXE_VARIANT_SCHEMA_VERSION,
        "scientific_contract": GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
        "estimator_family": GENERALIZED_GXE_VARIANT_ESTIMATOR_FAMILY,
        "probe_axis": "variant",
        "feature_convention": GENERALIZED_GXE_VARIANT_FEATURE_CONVENTION,
        "normal_equation_assembly": GENERALIZED_GXE_VARIANT_NORMAL_ASSEMBLY,
        "jackknife_method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
        "same_person_jackknife": GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE,
    }
    for key, expected in identity.items():
        if result.get(key) != expected:
            raise ValueError(f"manifest identity field {key!r} contradicts V1")
    axes = _validate_axes(result.get("axes"))
    _validate_randomization(result.get("randomization"))
    scale_plan = _scale_plan_from_manifest(result.get("genotype_scale_plan"))
    if result.get("genotype_scale_plan_sha256") != scale_plan.digest:
        raise ValueError("genotype scale plan digest does not match its record")
    if (
        scale_plan.retained_variant_order_sha256
        != axes["variants"]["retained_order_digest"]
    ):
        raise ValueError("genotype scale plan does not bind the variant axis")
    _validate_performance_ledger(result.get("performance_ledger"))
    if result.get("terminal_status") != "complete":
        raise ValueError("generalized reference terminal status is not complete")
    jackknife = _require_mapping("jackknife", result.get("jackknife"))
    jackknife_expected = {
        "method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
        "block_axis": "ordered_retained_variants",
        "source_scores_recomputed": False,
        "retained_ldscores_frozen": True,
        "local_context_weighted_ld_assumption": True,
        "same_person_deletion": GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE,
    }
    for key, expected in jackknife_expected.items():
        if jackknife.get(key) != expected:
            raise ValueError(f"jackknife field {key!r} contradicts V1")
    block_labels = axes["jackknife_blocks"]["block_labels"]
    if jackknife.get("num_blocks") != len(block_labels):
        raise ValueError("jackknife block count contradicts the ordered block axis")
    if jackknife.get("block_labels") != block_labels:
        raise ValueError("jackknife labels contradict the ordered block axis")
    n_variants = axes["variants"]["count"]
    n_samples = axes["samples"]["count"]
    pair_count = len(axes["pairs"]["table"])
    annotation_count = len(axes["annotations"]["names"])
    component_count = len(axes["components"]["table"])
    block_count = len(block_labels)
    residual_rank = axes["fixed_effects"]["residual_rank"]
    masses = np.asarray(axes["annotations"]["masses"], dtype=np.float64)
    component_annotations = np.asarray(
        [entry[0] for entry in axes["components"]["table"]],
        dtype=np.int64,
    )
    metadata = _require_mapping("numeric_arrays", result.get("numeric_arrays"))
    array_names = set(arrays)
    if not _REQUIRED_ARRAYS <= array_names or not array_names <= (
        _REQUIRED_ARRAYS | _OPTIONAL_ARRAYS
    ):
        raise ValueError("numeric array set is missing required or contains unknown arrays")
    if set(metadata) != array_names:
        raise ValueError("numeric array metadata does not match loaded arrays")
    expected_shapes = {
        "directed_numerator": (component_count, component_count),
        "symmetric_numerator": (component_count, component_count),
        "genetic_gram": (component_count, component_count),
        "block_directed_numerator": (block_count, component_count, component_count),
        "block_annotation_mass": (block_count, annotation_count),
        "same_person": (component_count, component_count),
        "deleted_genetic_gram": (block_count, component_count, component_count),
        "directional_ldscores": (n_variants, pair_count, component_count),
    }
    owned_arrays: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float64) or array.shape != expected_shapes[name]:
            raise ValueError(f"numeric array {name!r} has the wrong shape or dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"numeric array {name!r} is nonfinite")
        expected_metadata = numeric_array_metadata(array)
        if metadata[name] != expected_metadata:
            raise ValueError(f"numeric array {name!r} metadata/hash mismatch")
        owned_arrays[name] = array
    panel = _require_mapping("per_variant_panel", result.get("per_variant_panel"))
    common_panel = {
        "logical_layout": "variant_target_pair_source_component_c",
        "logical_compute_dtype": "float64",
    }
    for key, expected in common_panel.items():
        if panel.get(key) != expected:
            raise ValueError(f"per-variant panel field {key!r} contradicts V1")
    if "directional_ldscores" in owned_arrays:
        expected_panel = {
            "storage": "inline_npz",
            **common_panel,
            "array": "directional_ldscores",
            **numeric_array_metadata(owned_arrays["directional_ldscores"]),
        }
    else:
        expected_panel = {"storage": "omitted", **common_panel}
    if dict(panel) != expected_panel:
        raise ValueError("per-variant panel declaration does not match storage")
    directed = owned_arrays["directed_numerator"]
    symmetric = owned_arrays["symmetric_numerator"]
    if not np.allclose(symmetric, 0.5 * (directed + directed.T), rtol=0.0, atol=1.0e-12):
        raise ValueError("symmetric numerator does not symmetrize the directed numerator")
    denominator = (
        masses[component_annotations, None]
        * masses[component_annotations][None, :]
    )
    expected_gram = float(residual_rank**2) * symmetric / denominator
    if not np.allclose(
        owned_arrays["genetic_gram"],
        expected_gram,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("genetic Gram contradicts numerator/rank/mass normalization")
    block_directed = owned_arrays["block_directed_numerator"]
    if not np.allclose(
        np.sum(block_directed, axis=0),
        directed,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("block directed numerators do not reconstruct the full numerator")
    block_masses = owned_arrays["block_annotation_mass"]
    if not np.allclose(
        np.sum(block_masses, axis=0),
        masses,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("block annotation masses do not reconstruct full masses")
    if not np.allclose(
        owned_arrays["same_person"],
        owned_arrays["same_person"].T,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("same-person matrix must be symmetric")
    if "deleted_genetic_gram" in owned_arrays:
        deleted = owned_arrays["deleted_genetic_gram"]
        for block in range(block_count):
            retained_mass = masses - block_masses[block]
            if np.any(retained_mass <= 0.0):
                raise ValueError("a deletion empties an annotation")
            retained_directed = directed - block_directed[block]
            retained_symmetric = 0.5 * (
                retained_directed + retained_directed.T
            )
            retained_denominator = (
                retained_mass[component_annotations, None]
                * retained_mass[component_annotations][None, :]
            )
            expected_deleted = (
                float(residual_rank**2)
                * retained_symmetric
                / retained_denominator
            )
            if not np.allclose(deleted[block], expected_deleted, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError("cached deleted Gram contradicts frozen row deletion")
    _validate_pass_ledger(result.get("pass_ledger"), n_variants)
    provenance = _require_mapping("provenance", result.get("provenance"))
    source_commit = provenance.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("provenance source_commit must be 40 lowercase hex")
    _require_sha256("source_tree_sha256", provenance.get("source_tree_sha256"))
    _require_sha256("native_binary_sha256", provenance.get("native_binary_sha256"))
    diagnostics = _require_mapping("diagnostics", result.get("diagnostics"))
    if not _REQUIRED_DIAGNOSTICS <= set(diagnostics):
        raise ValueError("scientific diagnostics are incomplete")
    if diagnostics.get("all_values_finite") is not True:
        raise ValueError("scientific diagnostics do not attest finite values")
    if diagnostics.get("same_person_cross_tile_finalized") is not True:
        raise ValueError("same-person cross-tile finalization is not attested")
    if diagnostics.get("same_person_probe_count") != result["randomization"]["probe_count"]:
        raise ValueError("same-person probe count contradicts randomization")
    canonical_json(result)
    return result


@dataclass(frozen=True)
class GeneralizedGxEPlanInputs:
    num_samples: int
    num_variants: int
    num_basis: int
    num_annotations: int
    num_probes: int
    num_jackknife_blocks: int
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
            "num_jackknife_blocks",
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
    j = inputs.num_jackknife_blocks
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
        "block_directed_numerator": _checked_product(
            "block numerator bytes", 8, j, c, c
        ),
        "block_annotation_mass": _checked_product(
            "block mass bytes", 8, j, k
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
            "J": inputs.num_jackknife_blocks,
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
