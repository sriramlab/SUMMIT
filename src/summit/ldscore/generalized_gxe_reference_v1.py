"""Closed V1 artifact for generalized per-variant GxE LD-score references."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from summit.context.reference import ReferenceMoments
from summit.context.schema import GenotypeScalePlanV1
from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    canonical_json,
    freeze_context_mapping,
)
from summit.ldscore.generalized_gxe_variant import (
    GENERALIZED_GXE_VARIANT_ESTIMATOR_FAMILY,
    GENERALIZED_GXE_VARIANT_FEATURE_CONVENTION,
    GENERALIZED_GXE_VARIANT_NORMAL_ASSEMBLY,
    GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
    GENERALIZED_GXE_VARIANT_SCHEMA_VERSION,
    GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
    GlobalVariantProbeSpec,
)


GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD = (
    "frozen_full_genome_variant_ldscore_delete_block_v1"
)
GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE = "reuse_full_same_person_v1"
GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX = (
    ".generalized-gxe-variant-ldscore-v1.npz"
)
_FAMILY = "Generalized GxE variant-LD-score reference V1"
_REQUIRED_ARRAY_NAMES = (
    "directed_numerator",
    "symmetric_numerator",
    "genetic_gram",
    "block_directed_numerator",
    "block_annotation_mass",
    "same_person",
)
_OPTIONAL_ARRAY_NAMES = (
    "deleted_genetic_gram",
    "directional_ldscores",
    "affine_mean",
    "affine_inverse_scale",
)


def _owned_readonly(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r}")

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=invalid_constant,
    )


def _artifact_arrays(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
) -> dict[str, np.ndarray]:
    result = {
        name: getattr(artifact, name) for name in _REQUIRED_ARRAY_NAMES
    }
    for name in _OPTIONAL_ARRAY_NAMES:
        value = getattr(artifact, name)
        if value is not None:
            result[name] = value
    return result


def serialize_generalized_gxe_inference_axes(
    *,
    num_variants: int,
    num_samples: int,
    basis_names: Sequence[str],
    fixed_effect_rank: int,
    annotation_names: Sequence[str],
    annotation_masses: Sequence[float] | np.ndarray,
    variant_block_ids: Sequence[int] | np.ndarray,
    block_labels: Sequence[str],
    residual_component_names: Sequence[str],
) -> dict[str, Any]:
    """Serialize concrete model axes and downstream inference blocks."""
    if (
        isinstance(num_variants, bool)
        or not isinstance(num_variants, int)
        or num_variants < 1
        or isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or num_samples < 2
    ):
        raise ValueError("sample and variant counts are invalid")
    if (
        isinstance(fixed_effect_rank, bool)
        or not isinstance(fixed_effect_rank, int)
        or fixed_effect_rank < 0
        or fixed_effect_rank >= num_samples
    ):
        raise ValueError("fixed-effect rank is invalid")
    basis_labels = tuple(str(value) for value in basis_names)
    annotation_labels = tuple(str(value) for value in annotation_names)
    residual_labels = tuple(str(value) for value in residual_component_names)
    labels = tuple(str(value) for value in block_labels)
    if any(
        not values or len(set(values)) != len(values)
        for values in (basis_labels, annotation_labels, residual_labels, labels)
    ):
        raise ValueError("axis labels must be nonempty and unique")
    masses = np.asarray(annotation_masses, dtype=np.float64)
    if (
        masses.shape != (len(annotation_labels),)
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("annotation masses are invalid")
    blocks = np.asarray(variant_block_ids)
    if blocks.shape != (num_variants,) or blocks.dtype.kind not in "iu":
        raise ValueError("variant block IDs must be an integer vector of length M")
    block_values = blocks.astype(np.int64, copy=False)
    if (
        np.any(block_values < 0)
        or np.any(block_values >= len(labels))
        or set(block_values.tolist()) != set(range(len(labels)))
    ):
        raise ValueError("variant block IDs do not match the block labels")
    pairs = ContextPairIndex(len(basis_labels))
    components = ContextComponentIndex(annotation_labels, pairs)
    return {
        "variants": {"count": num_variants},
        "samples": {"count": num_samples},
        "basis": {"names": list(basis_labels)},
        "fixed_effects": {
            "rank": fixed_effect_rank,
            "residual_rank": num_samples - fixed_effect_rank,
        },
        "pairs": {"table": [[entry.q, entry.r] for entry in pairs.entries]},
        "annotations": {
            "names": list(annotation_labels),
            "masses": [float(value) for value in masses],
        },
        "components": {
            "table": [
                [entry.annotation_index, entry.pair_index]
                for entry in components.entries
            ]
        },
        "jackknife_blocks": {
            "variant_block_ids": [int(value) for value in block_values],
            "block_labels": list(labels),
        },
        "residual_components": {"names": list(residual_labels)},
    }


def _numeric_metadata(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype != np.float64 or not np.all(np.isfinite(array)):
        raise ValueError("generalized reference arrays must be finite FP64")
    return {"shape": list(array.shape), "dtype": "float64", "order": "C"}


def reduce_generalized_gxe_reference_for_inference(
    *,
    directional_ldscores: np.ndarray,
    annotations: np.ndarray,
    variant_block_ids: Sequence[int] | np.ndarray,
    block_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reduce fixed per-SNP LD scores into delete-block normal-equation terms.

    This function is deliberately downstream of reference-score estimation:
    it neither reads genotypes nor changes any retained SNP's LD score.
    """
    directional = np.asarray(directional_ldscores, dtype=np.float64)
    weights = np.asarray(annotations, dtype=np.float64)
    blocks = np.asarray(variant_block_ids)
    labels = tuple(str(label) for label in block_labels)
    if directional.ndim != 3 or weights.ndim != 2:
        raise ValueError("directional LD scores and annotations must be arrays")
    m, pair_count, component_count = directional.shape
    if weights.shape[0] != m or component_count != weights.shape[1] * pair_count:
        raise ValueError("per-SNP reference axes are inconsistent")
    if blocks.shape != (m,) or blocks.dtype.kind not in "iu":
        raise ValueError("inference block IDs must be an integer vector of length M")
    block_ids = blocks.astype(np.int64, copy=False)
    if (
        len(labels) < 2
        or len(set(labels)) != len(labels)
        or np.any(block_ids < 0)
        or np.any(block_ids >= len(labels))
        or set(block_ids.tolist()) != set(range(len(labels)))
    ):
        raise ValueError("inference blocks must be nonempty, contiguous, and labeled")
    if (
        not np.all(np.isfinite(directional))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("per-SNP reference values must be finite and nonnegative in weight")

    block_count = len(labels)
    annotation_count = weights.shape[1]
    block_masses = np.empty((block_count, annotation_count), dtype=np.float64)
    block_directed = np.empty(
        (block_count, component_count, component_count), dtype=np.float64
    )
    for annotation in range(annotation_count):
        annotation_weight = weights[:, annotation]
        block_masses[:, annotation] = np.bincount(
            block_ids, weights=annotation_weight, minlength=block_count
        )
        for target_pair in range(pair_count):
            left = annotation * pair_count + target_pair
            for right in range(component_count):
                block_directed[:, left, right] = np.bincount(
                    block_ids,
                    weights=(
                        annotation_weight
                        * directional[:, target_pair, right]
                    ),
                    minlength=block_count,
                )
    reconstructed = np.sum(block_directed, axis=0, dtype=np.float64)
    full = np.einsum(
        "mk,mpr->kpr", weights, directional, optimize=True
    ).reshape(component_count, component_count)
    reconstruction_error = float(
        np.max(np.abs(reconstructed - full), initial=0.0)
    )
    return block_directed, block_masses, reconstruction_error


def _validate_reference_structure(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    """Validate the concrete axes and numeric arrays directly."""
    if manifest.get("kind") != GENERALIZED_GXE_VARIANT_REFERENCE_KIND:
        raise ValueError("not a generalized variant-LD-score reference")
    try:
        axes = manifest["axes"]
        n = int(axes["samples"]["count"])
        m = int(axes["variants"]["count"])
        basis_names = tuple(axes["basis"]["names"])
        annotation_names = tuple(axes["annotations"]["names"])
        annotation_masses = np.asarray(
            axes["annotations"]["masses"], dtype=np.float64
        )
        block_labels = tuple(axes["jackknife_blocks"]["block_labels"])
        block_ids = np.asarray(
            axes["jackknife_blocks"]["variant_block_ids"], dtype=np.int64
        )
        residual_names = tuple(axes["residual_components"]["names"])
        fixed_rank = int(axes["fixed_effects"]["rank"])
        residual_rank = int(axes["fixed_effects"]["residual_rank"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("generalized reference axes are malformed") from exc
    if n < 2 or m < 1 or not basis_names or not annotation_names:
        raise ValueError("generalized reference axes are empty")
    if (
        len(set(basis_names)) != len(basis_names)
        or len(set(annotation_names)) != len(annotation_names)
        or len(set(block_labels)) != len(block_labels)
        or len(set(residual_names)) != len(residual_names)
    ):
        raise ValueError("generalized reference axis names must be unique")
    if fixed_rank < 0 or residual_rank != n - fixed_rank or residual_rank < 1:
        raise ValueError("generalized reference fixed-effect dimensions are invalid")
    if (
        annotation_masses.shape != (len(annotation_names),)
        or not np.all(np.isfinite(annotation_masses))
        or np.any(annotation_masses <= 0.0)
    ):
        raise ValueError("generalized reference annotation masses are invalid")
    if (
        block_ids.shape != (m,)
        or len(block_labels) < 2
        or np.any(block_ids < 0)
        or np.any(block_ids >= len(block_labels))
        or set(block_ids.tolist()) != set(range(len(block_labels)))
    ):
        raise ValueError("generalized reference block axis is invalid")

    pairs = ContextPairIndex(len(basis_names))
    components = ContextComponentIndex(annotation_names, pairs)
    c_count = len(components)
    j_count = len(block_labels)
    k_count = len(annotation_names)
    expected = {
        "directed_numerator": (c_count, c_count),
        "symmetric_numerator": (c_count, c_count),
        "genetic_gram": (c_count, c_count),
        "block_directed_numerator": (j_count, c_count, c_count),
        "block_annotation_mass": (j_count, k_count),
        "same_person": (c_count, c_count),
    }
    optional = {
        "deleted_genetic_gram": (j_count, c_count, c_count),
        "directional_ldscores": (m, len(pairs), c_count),
        "affine_mean": (m,),
        "affine_inverse_scale": (m,),
    }
    for name, shape in expected.items():
        if name not in arrays or np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"generalized reference array {name!r} has the wrong shape")
    for name, shape in optional.items():
        if name in arrays and np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"generalized reference array {name!r} has the wrong shape")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("generalized reference contains nonfinite numeric values")
    if ("affine_mean" in arrays) != ("affine_inverse_scale" in arrays):
        raise ValueError("genotype affine vectors must be present together")
    if "affine_inverse_scale" in arrays and np.any(
        arrays["affine_inverse_scale"] <= 0.0
    ):
        raise ValueError("genotype affine inverse scales must be positive")
    if not np.allclose(
        np.sum(arrays["block_annotation_mass"], axis=0),
        annotation_masses,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("block annotation masses do not reconstruct the total")
    if not np.allclose(
        np.sum(arrays["block_directed_numerator"], axis=0),
        arrays["directed_numerator"],
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError("block directed numerators do not reconstruct the total")


@dataclass(frozen=True)
class GeneralizedGxEVariantReferenceArtifactV1:
    """Owned immutable aggregate reference plus optional inline SNP panel."""

    manifest: Mapping[str, Any]
    directed_numerator: np.ndarray
    symmetric_numerator: np.ndarray
    genetic_gram: np.ndarray
    block_directed_numerator: np.ndarray
    block_annotation_mass: np.ndarray
    same_person: np.ndarray
    deleted_genetic_gram: np.ndarray | None = None
    directional_ldscores: np.ndarray | None = None
    affine_mean: np.ndarray | None = None
    affine_inverse_scale: np.ndarray | None = None
    component_index: ContextComponentIndex = field(init=False)
    scale_plan: None = field(init=False)
    genotype_scale: Mapping[str, str] = field(init=False)
    block_labels: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        for name in _REQUIRED_ARRAY_NAMES:
            object.__setattr__(self, name, _owned_readonly(getattr(self, name)))
        for name in _OPTIONAL_ARRAY_NAMES:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _owned_readonly(value))
        manifest = _strict_json_loads(canonical_json(self.manifest))
        _validate_reference_structure(manifest, _artifact_arrays(self))
        axes = manifest["axes"]
        pair_index = ContextPairIndex(len(axes["basis"]["names"]))
        component_index = ContextComponentIndex(
            tuple(axes["annotations"]["names"]), pair_index
        )
        object.__setattr__(self, "component_index", component_index)
        scale_source = dict(
            manifest.get("genotype_scale", manifest.get("genotype_scale_plan", {}))
        )
        semantic_fields = {
            "genotype_scale_policy": "genotype_scale_policy",
            "policy": "genotype_scale_policy",
            "allele_orientation": "allele_orientation",
            "allele_coding": "allele_coding",
            "centering_source": "centering_source",
            "centering_formula": "centering_formula",
            "scaling_formula": "scaling_formula",
            "missing_imputation": "missing_imputation",
            "ploidy_policy": "ploidy_policy",
        }
        scale = {
            target: str(scale_source[source])
            for source, target in semantic_fields.items()
            if source in scale_source
        }
        if not scale or any(not key or not value for key, value in scale.items()):
            raise ValueError("generalized reference genotype scale is missing")
        object.__setattr__(self, "scale_plan", None)
        object.__setattr__(self, "genotype_scale", freeze_context_mapping(scale))
        object.__setattr__(
            self,
            "block_labels",
            tuple(axes["jackknife_blocks"]["block_labels"]),
        )
        object.__setattr__(self, "manifest", freeze_context_mapping(manifest))

    @property
    def n_samples(self) -> int:
        return int(self.manifest["axes"]["samples"]["count"])

    @property
    def reference_n(self) -> int:
        return self.n_samples

    @property
    def n_variants(self) -> int:
        return int(self.manifest["axes"]["variants"]["count"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["axes"]["fixed_effects"]["residual_rank"])

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self.block_labels

    @property
    def annotation_masses(self) -> np.ndarray:
        result = np.asarray(
            self.manifest["axes"]["annotations"]["masses"],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @property
    def group_variant_counts(self) -> np.ndarray:
        block_ids = np.asarray(
            self.manifest["axes"]["jackknife_blocks"]["variant_block_ids"],
            dtype=np.int64,
        )
        result = np.bincount(block_ids, minlength=len(self.block_labels))
        result.setflags(write=False)
        return result

    @property
    def full_moments(self) -> ReferenceMoments:
        return ReferenceMoments(
            annotation_masses=self.annotation_masses,
            gram=self.genetic_gram,
            same_person=self.same_person,
        )

    def verify(self) -> None:
        _validate_reference_structure(self.manifest, _artifact_arrays(self))


def build_generalized_gxe_variant_reference_v1(
    *,
    axes: Mapping[str, Any],
    probe_spec: GlobalVariantProbeSpec,
    genotype_scale_plan: GenotypeScalePlanV1 | Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    pass_ledger: Mapping[str, Any],
    performance_ledger: Mapping[str, Any],
    provenance: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> GeneralizedGxEVariantReferenceArtifactV1:
    """Construct a reference validated from its concrete fields and arrays."""
    if not isinstance(probe_spec, GlobalVariantProbeSpec):
        raise TypeError("probe_spec must be a GlobalVariantProbeSpec")
    if isinstance(genotype_scale_plan, GenotypeScalePlanV1):
        scale_record = {
            "genotype_scale_policy": genotype_scale_plan.policy.value,
            "allele_orientation": genotype_scale_plan.allele_orientation,
            "allele_coding": genotype_scale_plan.allele_coding,
            "centering_source": genotype_scale_plan.centering_source,
            "centering_formula": genotype_scale_plan.centering_formula,
            "scaling_formula": genotype_scale_plan.scaling_formula,
            "missing_imputation": genotype_scale_plan.missing_imputation,
            "ploidy_policy": genotype_scale_plan.ploidy_policy,
        }
    elif isinstance(genotype_scale_plan, Mapping):
        scale_record = {
            str(key): str(value) for key, value in genotype_scale_plan.items()
        }
    else:
        raise TypeError("genotype scale metadata must be a mapping")
    if not scale_record or any(not key or not value for key, value in scale_record.items()):
        raise ValueError("genotype scale metadata is invalid")
    owned = {name: _owned_readonly(value) for name, value in arrays.items()}
    randomization = {
        "distribution": "rademacher",
        "algorithm": "counter_global_variant_global_probe_v1",
        "root_seed": probe_spec.root_seed,
        "probe_offset": probe_spec.probe_offset,
        "probe_count": probe_spec.probe_count,
        "variant_index_space": "retained_ordered_variant_axis_v1",
        "tile_invariant": True,
        "shared_with_same_person": True,
        "stream_namespace": probe_spec.namespace,
        "stream_namespace_key_uint64": probe_spec.namespace_key,
    }
    panel = {
        "storage": "omitted",
        "logical_layout": "variant_target_pair_source_component_c",
        "logical_compute_dtype": "float64",
    }
    if "directional_ldscores" in owned:
        panel = {
            **panel,
            "storage": "inline_npz",
            "array": "directional_ldscores",
            **_numeric_metadata(owned["directional_ldscores"]),
        }
    block_labels = list(axes["jackknife_blocks"]["block_labels"])
    manifest = {
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
        "randomization": randomization,
        "genotype_scale": scale_record,
        "jackknife": {
            "method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
            "num_blocks": len(block_labels),
            "block_labels": block_labels,
            "block_axis": "ordered_retained_variants",
            "source_scores_recomputed": False,
            "retained_ldscores_frozen": True,
            "local_context_weighted_ld_assumption": True,
            "same_person_deletion": GENERALIZED_GXE_VARIANT_SAME_PERSON_JACKKNIFE,
        },
        "numeric_arrays": {
            name: _numeric_metadata(value) for name, value in owned.items()
        },
        "pass_ledger": dict(pass_ledger),
        "performance_ledger": dict(performance_ledger),
        "provenance": dict(provenance),
        "diagnostics": dict(diagnostics),
        "per_variant_panel": panel,
        "terminal_status": "complete",
    }
    return GeneralizedGxEVariantReferenceArtifactV1(
        manifest=manifest,
        **{name: owned.get(name) for name in (*_REQUIRED_ARRAY_NAMES, *_OPTIONAL_ARRAY_NAMES)},
    )


def _canonical_native_pass_ledger(
    value: Mapping[str, Any], *, num_variants: int
) -> dict[str, int]:
    ledger = dict(value)
    canonical_names = {
        "planned_reference_genotype_passes",
        "observed_reference_genotype_passes",
        "planned_retained_variant_visits",
        "observed_retained_variant_visits",
        "duplicate_retained_variant_visits",
        "pass1_decoded_blocks",
        "pass2_decoded_blocks",
        "retry_count",
        "repair_count",
        "fallback_count",
        "integrity_failure_count",
    }
    if canonical_names <= set(ledger):
        return {name: int(ledger[name]) for name in canonical_names}
    block_reads = ledger.get("observed_block_reads")
    if (
        isinstance(block_reads, bool)
        or not isinstance(block_reads, int)
        or block_reads < 2
        or block_reads % 2 != 0
    ):
        raise ValueError("native block-read ledger cannot be split across two passes")
    result = {
        "planned_reference_genotype_passes": int(
            ledger.get("planned_reference_genotype_passes", -1)
        ),
        "observed_reference_genotype_passes": int(
            ledger.get("observed_reference_genotype_passes", -1)
        ),
        "planned_retained_variant_visits": int(
            ledger.get("planned_retained_variant_visits", -1)
        ),
        "observed_retained_variant_visits": int(
            ledger.get("observed_retained_variant_visits", -1)
        ),
        "duplicate_retained_variant_visits": int(
            ledger.get("duplicate_variant_visits", -1)
        ),
        "pass1_decoded_blocks": block_reads // 2,
        "pass2_decoded_blocks": block_reads // 2,
        "retry_count": int(ledger.get("retry_count", -1)),
        "repair_count": int(ledger.get("repair_count", -1)),
        "fallback_count": int(ledger.get("fallback_count", -1)),
        "integrity_failure_count": int(ledger.get("integrity_failures", -1)),
    }
    if (
        result["observed_reference_genotype_passes"] != 2
        or result["observed_retained_variant_visits"] != 2 * num_variants
    ):
        raise ValueError("native result is not a complete two-pass execution")
    return result


def build_generalized_gxe_variant_reference_from_native_v1(
    native_result: Any,
    *,
    axes: Mapping[str, Any],
    annotations: np.ndarray,
    probe_spec: GlobalVariantProbeSpec,
    genotype_scale_plan: GenotypeScalePlanV1 | Mapping[str, Any],
    performance_ledger: Mapping[str, Any],
    provenance: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    include_directional_panel: bool = False,
) -> GeneralizedGxEVariantReferenceArtifactV1:
    """Publishable adapter from the Stage 06 native result, without genotypes."""
    if not isinstance(include_directional_panel, bool):
        raise ValueError("native artifact inclusion policy must be boolean")
    native_scale = getattr(native_result, "genotype_scale", None)
    if native_scale is not None and not isinstance(native_scale, Mapping):
        raise ValueError("native genotype scale metadata is invalid")
    required = {
        "directed_numerator": "directed_numerator",
        "symmetric_numerator": "symmetric_numerator",
        "genetic_gram": "genetic_gram",
        "same_person": "same_person",
    }
    arrays: dict[str, np.ndarray] = {}
    for output_name, attribute in required.items():
        if not hasattr(native_result, attribute):
            raise ValueError(f"native result lacks {attribute!r}")
        arrays[output_name] = np.asarray(getattr(native_result, attribute))
    block_directed, block_masses, reconstruction_error = (
        reduce_generalized_gxe_reference_for_inference(
        directional_ldscores=np.asarray(native_result.directional_ldscores),
        annotations=annotations,
        variant_block_ids=axes["jackknife_blocks"]["variant_block_ids"],
        block_labels=axes["jackknife_blocks"]["block_labels"],
        )
    )
    arrays["block_directed_numerator"] = block_directed
    arrays["block_annotation_mass"] = block_masses
    if include_directional_panel:
        arrays["directional_ldscores"] = np.asarray(
            getattr(native_result, "directional_ldscores")
        )
    arrays["affine_mean"] = np.asarray(getattr(native_result, "affine_mean"))
    arrays["affine_inverse_scale"] = np.asarray(
        getattr(native_result, "affine_inverse_scale")
    )
    raw_ledger = getattr(native_result, "ledger", None)
    if not isinstance(raw_ledger, Mapping):
        raise ValueError("native result lacks a pass ledger")
    pass_ledger = _canonical_native_pass_ledger(
        raw_ledger,
        num_variants=int(axes["variants"]["count"]),
    )
    diagnostic_record = dict(diagnostics)
    diagnostic_record["block_reconstruction_error"] = reconstruction_error
    diagnostic_record["minimum_deleted_annotation_mass"] = float(
        np.min(np.asarray(axes["annotations"]["masses"])[None, :] - block_masses)
    )
    return build_generalized_gxe_variant_reference_v1(
        axes=axes,
        probe_spec=probe_spec,
        genotype_scale_plan=(
            native_scale if native_scale is not None else genotype_scale_plan
        ),
        arrays=arrays,
        pass_ledger=pass_ledger,
        performance_ledger=performance_ledger,
        provenance=provenance,
        diagnostics=diagnostic_record,
    )


def reference_moments_after_deleting_variant_blocks_v1(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
    blocks: Sequence[str],
) -> ReferenceMoments:
    """Subtract fixed full-genome target rows without any genotype access."""
    if not isinstance(artifact, GeneralizedGxEVariantReferenceArtifactV1):
        raise ValueError("artifact must be a generalized variant-LD-score reference")
    artifact.verify()
    return _reference_moments_after_deleting_variant_blocks_prevalidated_v1(
        artifact, blocks
    )


def _reference_moments_after_deleting_variant_blocks_prevalidated_v1(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
    blocks: Sequence[str],
) -> ReferenceMoments:
    """Subtract target rows after an enclosing artifact verification."""
    if isinstance(blocks, (str, bytes)):
        raise ValueError("deleted variant blocks must be a sequence")
    requested = tuple(blocks)
    if any(not isinstance(value, str) or not value for value in requested):
        raise ValueError("deleted variant block labels must be nonempty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("deleted variant block labels must be unique")
    unknown = set(requested) - set(artifact.block_labels)
    if unknown:
        raise ValueError(f"unknown variant blocks: {sorted(unknown)}")
    if not requested:
        return artifact.full_moments
    indices = np.fromiter(
        (artifact.block_labels.index(value) for value in requested),
        dtype=np.int64,
        count=len(requested),
    )
    masses = artifact.annotation_masses
    retained_masses = masses - np.sum(
        artifact.block_annotation_mass[indices], axis=0, dtype=np.float64
    )
    if np.any(retained_masses <= 0.0):
        raise ValueError("variant-block deletion empties an annotation")
    directed = artifact.directed_numerator - np.sum(
        artifact.block_directed_numerator[indices], axis=0, dtype=np.float64
    )
    symmetric = 0.5 * (directed + directed.T)
    component_annotations = np.fromiter(
        (entry.annotation_index for entry in artifact.component_index.entries),
        dtype=np.int64,
        count=len(artifact.component_index),
    )
    component_masses = retained_masses[component_annotations]
    gram = (
        float(artifact.residual_rank**2)
        * symmetric
        / np.outer(component_masses, component_masses)
    )
    return ReferenceMoments(
        annotation_masses=retained_masses,
        gram=gram,
        same_person=artifact.same_person,
    )


def write_generalized_gxe_variant_reference_v1(
    artifact: GeneralizedGxEVariantReferenceArtifactV1,
    output: str | Path,
) -> Path:
    """Atomically publish one generalized reference V1 NPZ container."""
    if not isinstance(artifact, GeneralizedGxEVariantReferenceArtifactV1):
        raise ValueError("only the generalized variant-LD-score V1 artifact is accepted")
    artifact.verify()
    arrays = _artifact_arrays(artifact)
    manifest_json = np.asarray(canonical_json(artifact.manifest))
    path = Path(output)
    if not path.name.endswith(GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX):
        path = Path(str(path) + GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                manifest_json=manifest_json,
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_generalized_gxe_variant_reference_v1(
    path: str | Path,
) -> GeneralizedGxEVariantReferenceArtifactV1:
    """Load only the distinct generalized variant-LD-score artifact family."""
    source = Path(path)
    if not source.name.endswith(GENERALIZED_GXE_VARIANT_REFERENCE_V1_SUFFIX):
        raise ValueError("generalized variant-LD-score loader requires its V1 suffix")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "manifest_json" not in archive.files:
                raise ValueError("generalized reference manifest is missing")
            manifest_value = np.asarray(archive["manifest_json"])
            if manifest_value.shape != () or manifest_value.dtype.kind not in "US":
                raise ValueError("generalized reference manifest is invalid")
            manifest_text = str(manifest_value.item())
            if len(manifest_text.encode("utf-8")) > 16 * 1024 * 1024:
                raise ValueError("generalized reference manifest is too large")
            manifest = _strict_json_loads(manifest_text)
            if manifest.get("kind") != GENERALIZED_GXE_VARIANT_REFERENCE_KIND:
                raise ValueError(
                    "other artifact families are not generalized variant LD scores"
                )
            metadata = manifest.get("numeric_arrays")
            if not isinstance(metadata, Mapping):
                raise ValueError("generalized reference numeric metadata is missing")
            names = set(metadata)
            required = set(_REQUIRED_ARRAY_NAMES)
            optional = set(_OPTIONAL_ARRAY_NAMES)
            if not required <= names or not names <= required | optional:
                raise ValueError("generalized reference numeric member set is invalid")
            expected_members = {"manifest_json", *names}
            observed_members = set(archive.files)
            if expected_members != observed_members:
                raise ValueError("generalized reference container key mismatch")
            arrays: dict[str, np.ndarray] = {}
            for name, record in metadata.items():
                if (
                    not isinstance(record, Mapping)
                    or record.get("dtype") != "float64"
                    or record.get("order") != "C"
                    or not isinstance(record.get("shape"), list)
                ):
                    raise ValueError(f"generalized array metadata {name!r} is invalid")
                shape = tuple(record["shape"])
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                ):
                    raise ValueError(f"generalized array shape {name!r} is invalid")
                value = np.asarray(archive[name])
                if value.dtype != np.float64 or value.shape != shape:
                    raise ValueError(
                        f"generalized reference array {name!r} disagrees with metadata"
                    )
                arrays[name] = np.array(value, dtype=np.float64, order="C", copy=True)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and (
            str(exc).startswith(_FAMILY)
            or "generalized" in str(exc)
            or "sample-probe" in str(exc)
        ):
            raise
        raise ValueError("not a valid generalized variant-LD-score artifact") from exc
    return GeneralizedGxEVariantReferenceArtifactV1(
        manifest=manifest,
        **{
            name: arrays.get(name)
            for name in (*_REQUIRED_ARRAY_NAMES, *_OPTIONAL_ARRAY_NAMES)
        },
    )
