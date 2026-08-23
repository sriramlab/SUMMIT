"""Stable summary-only contextual fit V1.

The fitter consumes only immutable contextual reference and trait artifacts.
It performs population transfer, approximate grouped deletion, small-system
assembly, the frozen rank-checked raw solve, the every-group jackknife, and
optional context-surface and PSD interpretations.  It never reads or retains
sample- or variant-axis arrays.
"""

from __future__ import annotations

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
    validate_python_source_runtime_provenance,
)
from .fit import (
    ContextJackknifeError,
    ContextNormalEquations,
    ContextRankError,
    PSDProjectionResult,
    project_genetic_coefficients_psd,
    solve_context_normal_equations,
)
from .oracle import (
    coefficients_to_omegas,
    context_covariance_surface,
    transfer_reference_gram,
)
from .reference import ReferenceMoments
from .reference_v1 import (
    ContextualReferenceArtifactV1,
    reference_moments_after_deleting_groups_v1,
)
from .schema import (
    AnnotationMode,
    ArtifactFamily,
    ContextSchemaIdentityV1,
    GenotypeScalePlanV1,
    GenotypeScalePolicy,
    GroupedEncodingVersion,
    LogicalSchemaVersion,
    annotation_output_contract,
    require_identical_scale_plans,
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
from .trait_v1 import (
    ContextualTraitArtifactV1,
    ContextualTraitMomentsV1,
    trait_moments_after_deleting_groups_v1,
)


CONTEXTUAL_FIT_V1_MAGIC = "SUMMIT_CONTEXTUAL_FIT_V1"
CONTEXTUAL_FIT_V1_SUFFIX = ".contextual-fit-v1.npz"
DEFAULT_FIT_V1_RTOL = 1.0e-10

_REFERENCE_NATIVE_BACKEND_V1 = "plink_bed_descriptor_stream_stage2_v1:2"
_TRAIT_NATIVE_BACKEND_V1 = "plink_bed_descriptor_stream_trait_v1:2"
_FIT_BACKEND_V1 = "python_numpy_scipy_summary_fit_v1"
_BACKEND_PAIR_POLICY_V1 = "exact_reference_stage2_trait_v1_backend_pair_v1"
_EVALUATION_GRID_ROLE_V1 = "non_row_evaluation_grid_v1"
_PYTHON_FIT_PROVENANCE_POLICY = (
    "ordered_relative_module_name_and_content_sha256_sha256_v1"
)
_PYTHON_FIT_MODULES = (
    "_artifact_io.py",
    "annotations.py",
    "fit.py",
    "fit_v1.py",
    "oracle.py",
    "reference.py",
    "reference_v1.py",
    "schema.py",
    "spec.py",
    "trait_v1.py",
)


def _python_fit_implementation_provenance() -> dict[str, object]:
    return python_source_runtime_provenance(
        _PYTHON_FIT_MODULES,
        policy=_PYTHON_FIT_PROVENANCE_POLICY,
        include_scipy=True,
    )


_BASE_ARRAY_NAMES = (
    "normal_matrix",
    "normal_rhs",
    "traces",
    "annotation_masses",
    "reference_genetic_gram",
    "transferred_genetic_gram",
    "raw_coefficients",
    "raw_genetic_coefficients",
    "raw_residual_coefficients",
    "raw_omegas",
    "raw_loo_coefficients",
    "raw_jackknife_covariance",
    "raw_standard_errors",
    "raw_solve_residual",
    "raw_solve_eigenvalues",
    "raw_solve_singular_values",
    "raw_solve_retained_directions",
    "raw_solve_null_space",
)
_PSD_ARRAY_NAMES = (
    "psd_coefficients",
    "psd_omegas",
    "psd_minimum_eigenvalues",
)
_SURFACE_ARRAY_NAMES = (
    "context_grid",
    "basis_metric",
    "raw_covariance_surfaces",
    "raw_combined_covariance_surfaces",
)
_PSD_SURFACE_ARRAY_NAMES = (
    "psd_covariance_surfaces",
    "psd_combined_covariance_surface",
)
_ALL_ARRAY_NAMES = frozenset(
    (
        *_BASE_ARRAY_NAMES,
        *_PSD_ARRAY_NAMES,
        *_SURFACE_ARRAY_NAMES,
        *_PSD_SURFACE_ARRAY_NAMES,
    )
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
        "compatibility",
        "genotype_scale_plan",
        "genotype_scale_plan_sha256",
        "selection",
        "interpretation",
        "solve",
        "jackknife",
        "optional_interpretation",
        "surfaces",
        "maps",
        "layouts",
        "arrays",
        "terminal_status",
    }
)

_SCIENTIFIC_POLICY = {
    "feature_order": "P_diag_phi_G_v1",
    "pair_order": "diagonal_then_lexicographic_offdiagonal_v1",
    "component_order": "annotation_major_pair_minor_v1",
    "population_transfer": "sample_count_same_distinct_v1",
    "deletion": "approx_group_numerator_full_D_v1",
    "assembly": "genetic_then_residual_full_symmetric_v1",
    "raw_fit": "rank_checked_symmetric_v1",
    "jackknife": "every_frozen_group_equal_delete_one_v1",
    "raw_output": "primary_unmodified_v1",
}

_BASE_LAYOUTS = {
    "normal_matrix": ["model_component", "model_component"],
    "normal_rhs": ["model_component"],
    "traces": ["model_component"],
    "annotation_masses": ["annotation"],
    "reference_genetic_gram": ["component", "component"],
    "transferred_genetic_gram": ["component", "component"],
    "raw_coefficients": ["model_component"],
    "raw_genetic_coefficients": ["component"],
    "raw_residual_coefficients": ["residual_component"],
    "raw_omegas": ["annotation", "basis", "basis"],
    "raw_loo_coefficients": ["deletion_group", "model_component"],
    "raw_jackknife_covariance": ["model_component", "model_component"],
    "raw_standard_errors": ["model_component"],
    "raw_solve_residual": ["model_component"],
    "raw_solve_eigenvalues": ["model_component"],
    "raw_solve_singular_values": ["model_component"],
    "raw_solve_retained_directions": ["model_component", "retained_direction"],
    "raw_solve_null_space": ["model_component", "null_direction"],
}
_PSD_LAYOUTS = {
    "psd_coefficients": ["component"],
    "psd_omegas": ["annotation", "basis", "basis"],
    "psd_minimum_eigenvalues": ["annotation"],
}
_SURFACE_LAYOUTS = {
    "context_grid": ["context_point", "basis"],
    "basis_metric": ["basis", "basis"],
    "raw_covariance_surfaces": [
        "fit_replicate",
        "annotation",
        "context_point",
        "context_point",
    ],
    "raw_combined_covariance_surfaces": [
        "fit_replicate",
        "context_point",
        "context_point",
    ],
}
_PSD_SURFACE_LAYOUTS = {
    "psd_covariance_surfaces": [
        "annotation",
        "context_point",
        "context_point",
    ],
    "psd_combined_covariance_surface": ["context_point", "context_point"],
}


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


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key in result:
                raise ValueError(
                    "Contextual fit V1 metadata keys must be unique strings."
                )
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
                "Contextual fit V1 metadata cannot contain non-finite values."
            )
        return value
    raise ValueError(
        f"Contextual fit V1 metadata contains unsupported {type(value).__name__}."
    )


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate contextual fit V1 JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("Contextual fit V1 manifest is not valid JSON.") from exc


def _immutable_array(value: Any) -> np.ndarray:
    source = np.ascontiguousarray(value, dtype=np.dtype(np.float64).newbyteorder("<"))
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


def _scale_plan_from_manifest(value: Any) -> GenotypeScalePlanV1:
    if not isinstance(value, Mapping):
        raise ValueError("Contextual fit V1 scale plan is invalid.")
    try:
        scale_plan = GenotypeScalePlanV1(
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
        raise ValueError("Contextual fit V1 scale plan is invalid.") from exc
    if scale_plan.policy is not GenotypeScalePolicy.SEALED_VARIANT_AFFINE_V1:
        raise ValueError("Contextual fit V1 requires sealed affine scaling.")
    if scale_plan.allele_orientation not in {
        "bim_a1_counted_v1",
        "bim_a2_counted_v1",
        "mixed_bim_a1_a2_per_variant_v1",
    }:
        raise ValueError("Contextual fit V1 allele orientation is unsupported.")
    expected = {
        "allele_coding": "plink_bed_snp_major_diploid_hardcall_v1",
        "centering_source": "provided_v1",
        "centering_formula": "provided_variant_affine_mean_v1",
        "scaling_formula": "dosage_minus_mean_times_inverse_scale_v1",
        "missing_imputation": "sealed_mean_v1",
        "ploidy_policy": "diploid_v1",
    }
    if any(
        getattr(scale_plan, field) != expected_value
        for field, expected_value in expected.items()
    ):
        raise ValueError("Contextual fit V1 descriptor scale fields are unsupported.")
    return scale_plan


def _annotation_mode(artifact: ContextualReferenceArtifactV1) -> AnnotationMode:
    try:
        return AnnotationMode(artifact.manifest["execution"]["annotation_mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reference annotation mode is missing or invalid.") from exc


def _select_trait(
    trait: ContextualTraitArtifactV1, selector: str | int | None
) -> tuple[int, str]:
    if selector is None:
        if trait.n_traits != 1:
            raise ValueError(
                "trait_selector is required when a contextual trait artifact "
                "contains more than one trait."
            )
        return 0, trait.trait_ids[0]
    if isinstance(selector, bool) or not isinstance(selector, (str, int, np.integer)):
        raise ValueError("trait_selector must be an exact trait ID or integer index.")
    if isinstance(selector, str):
        if selector not in trait.trait_ids:
            raise ValueError(f"Unknown contextual trait V1 trait {selector!r}.")
        return trait.trait_ids.index(selector), selector
    index = int(selector)
    if index < 0 or index >= trait.n_traits:
        raise IndexError("Contextual trait V1 trait index is out of range.")
    return index, trait.trait_ids[index]


def validate_contextual_fit_compatibility_v1(
    reference: ContextualReferenceArtifactV1,
    trait: ContextualTraitArtifactV1,
) -> Mapping[str, Any]:
    """Validate every stable cross-artifact identity used by the fitter."""
    if not isinstance(reference, ContextualReferenceArtifactV1):
        raise ValueError("reference must be a ContextualReferenceArtifactV1.")
    if not isinstance(trait, ContextualTraitArtifactV1):
        raise ValueError("trait must be a ContextualTraitArtifactV1.")
    reference.verify()
    trait.verify()
    mismatches: list[str] = []
    for name in (
        "native_api_version",
        "build_id",
        "grouped_encoding_version",
        "feature_mode",
    ):
        if reference.manifest.get(name) != trait.manifest.get(name):
            mismatches.append(name)
    backend_pair = (
        reference.manifest.get("native_backend_version"),
        trait.manifest.get("native_backend_version"),
    )
    if backend_pair != (_REFERENCE_NATIVE_BACKEND_V1, _TRAIT_NATIVE_BACKEND_V1):
        mismatches.append("native_backend_pair")
    try:
        require_identical_scale_plans(reference.scale_plan, trait.scale_plan)
    except ValueError:
        mismatches.append("genotype_scale_plan_sha256")
    if reference.component_index.digest != trait.component_index.digest:
        mismatches.append("component_map_sha256")
    if reference.group_ids != trait.group_ids:
        mismatches.append("group_ids")
    reference_dimensions = reference.manifest.get("dimensions", {})
    trait_dimensions = trait.manifest.get("dimensions", {})
    for name in ("M", "Q", "K", "C", "J"):
        if reference_dimensions.get(name) != trait_dimensions.get(name):
            mismatches.append(f"dimensions.{name}")
    reference_execution = reference.manifest.get("execution", {})
    trait_execution = trait.manifest.get("execution", {})
    for name in ("annotation_mode", "numeric_policy"):
        if reference_execution.get(name) != trait_execution.get(name):
            mismatches.append(f"execution.{name}")
    reference_build_provenance = reference_execution.get("build_provenance")
    trait_build_provenance = trait_execution.get("build_provenance")
    if not isinstance(reference_build_provenance, Mapping) or not isinstance(
        trait_build_provenance, Mapping
    ):
        mismatches.append("execution.build_provenance")
        native_build_provenance_sha256 = ""
    else:
        reference_build_provenance_sha256 = canonical_sha256(
            dict(reference_build_provenance)
        )
        trait_build_provenance_sha256 = canonical_sha256(dict(trait_build_provenance))
        native_build_provenance_sha256 = reference_build_provenance_sha256
        if reference_build_provenance_sha256 != trait_build_provenance_sha256:
            mismatches.append("execution.build_provenance")
    reference_maps = reference.manifest.get("maps", {})
    trait_maps = trait.manifest.get("maps", {})
    for name in ("annotation_map_sha256", "group_map_sha256"):
        if reference_maps.get(name) != trait_maps.get(name):
            mismatches.append(f"maps.{name}")
    reference_identity = reference.manifest.get("identity", {})
    trait_identity = trait.manifest.get("identity", {})
    for name in (
        "variant_order_allele_sha256",
        "retained_variant_order_sha256",
        "basis_specification_sha256",
        "basis_calibration_sha256",
        "source_tree_sha256",
    ):
        if reference_identity.get(name) != trait_identity.get(name):
            mismatches.append(f"identity.{name}")
    reference_deletion = reference.manifest.get("deletion", {})
    trait_deletion = trait.manifest.get("deletion", {})
    for name in (
        "semantics",
        "grouped_storage",
        "empty_annotation",
        "multiple_group_subtraction_supported",
        "claim",
    ):
        left = reference_deletion.get(name)
        right = trait_deletion.get(name)
        if (left is not None or right is not None) and left != right:
            mismatches.append(f"deletion.{name}")
    if (
        trait.manifest.get("compatible_reference_identity_sha256")
        != reference.manifest_sha256
    ):
        mismatches.append("compatible_reference_identity_sha256")
    if not np.array_equal(reference.group_variant_counts, trait.group_variant_counts):
        mismatches.append("group_variant_counts")
    reference_mass_sha256 = array_sha256(reference.annotation_masses)
    trait_mass_sha256 = array_sha256(trait.annotation_masses)
    if reference_mass_sha256 != trait_mass_sha256:
        mismatches.append("annotation_masses")
    reference_group_mass_sha256 = array_sha256(reference.group_annotation_masses)
    trait_group_mass_sha256 = array_sha256(trait.group_annotation_masses)
    if reference_group_mass_sha256 != trait_group_mass_sha256:
        mismatches.append("group_annotation_masses")
    reference_count_sha256 = array_sha256(reference.group_variant_counts)
    trait_count_sha256 = array_sha256(trait.group_variant_counts)
    if reference_count_sha256 != trait_count_sha256:
        mismatches.append("group_variant_counts_sha256")
    for index, group in enumerate(reference.group_ids):
        reference_retained = (
            reference.annotation_masses - reference.group_annotation_masses[index]
        )
        trait_retained = trait.annotation_masses - trait.group_annotation_masses[index]
        if array_sha256(reference_retained) != array_sha256(trait_retained):
            mismatches.append(f"retained_annotation_masses.{group}")
    if mismatches:
        raise ValueError(
            "Contextual reference/trait V1 compatibility mismatch: "
            + ", ".join(sorted(set(mismatches)))
            + "."
        )
    fit_provenance = _python_fit_implementation_provenance()
    return freeze_context_mapping(
        {
            "reference_manifest_sha256": reference.manifest_sha256,
            "trait_manifest_sha256": trait.manifest_sha256,
            "matched_native_api_version": reference.manifest["native_api_version"],
            "reference_native_backend_version": backend_pair[0],
            "trait_native_backend_version": backend_pair[1],
            "backend_pair_policy": _BACKEND_PAIR_POLICY_V1,
            "matched_build_id": reference.manifest["build_id"],
            "matched_native_build_provenance_sha256": (native_build_provenance_sha256),
            "matched_component_map_sha256": reference.component_index.digest,
            "matched_scale_plan_sha256": reference.scale_plan.digest,
            "matched_annotation_mode": reference_execution["annotation_mode"],
            "matched_group_sequence": list(reference.group_ids),
            "matched_source_tree_sha256": reference_identity["source_tree_sha256"],
            "matched_variant_order_allele_sha256": reference_identity[
                "variant_order_allele_sha256"
            ],
            "matched_basis_specification_sha256": reference_identity[
                "basis_specification_sha256"
            ],
            "matched_basis_calibration_sha256": reference_identity[
                "basis_calibration_sha256"
            ],
            "matched_annotation_map_sha256": reference_maps["annotation_map_sha256"],
            "matched_group_map_sha256": reference_maps["group_map_sha256"],
            "matched_annotation_masses_sha256": reference_mass_sha256,
            "matched_group_annotation_masses_sha256": (reference_group_mass_sha256),
            "matched_group_variant_counts_sha256": reference_count_sha256,
            "matched_numeric_policy": reference_execution["numeric_policy"],
            "matched_deletion_policy_sha256": canonical_sha256(
                dict(reference_deletion)
            ),
            "matched_grouped_encoding_version": reference.manifest[
                "grouped_encoding_version"
            ],
            "python_fit_implementation_provenance": fit_provenance,
            "python_fit_implementation_provenance_sha256": canonical_sha256(
                fit_provenance
            ),
        }
    )


def assemble_contextual_normal_equations_from_moments_v1(
    *,
    component_index: ContextComponentIndex,
    reference_n: int,
    trait: ContextualTraitArtifactV1,
    trait_index: int,
    deleted_groups: Sequence[str],
    reference_moments: ReferenceMoments,
    trait_moments: ContextualTraitMomentsV1,
) -> ContextNormalEquations:
    """Assemble the shared population-transfer system from compact moments."""
    requested = tuple(deleted_groups)
    if array_sha256(reference_moments.annotation_masses) != array_sha256(
        trait_moments.annotation_masses
    ):
        raise ValueError("Reference and trait retained annotation masses differ.")
    transferred = transfer_reference_gram(
        reference_moments.gram,
        reference_moments.same_person,
        reference_n=reference_n,
        study_n=trait.n_samples,
    )
    c = len(component_index)
    h = len(trait.residual_names)
    matrix = np.empty((c + h, c + h), dtype=np.float64)
    matrix[:c, :c] = transferred
    matrix[:c, c:] = trait_moments.genetic_residual
    matrix[c:, :c] = trait_moments.genetic_residual.T
    matrix[c:, c:] = trait_moments.residual_gram
    matrix = 0.5 * (matrix + matrix.T)
    rhs = np.concatenate(
        (
            trait_moments.genetic_rhs[:, trait_index],
            trait_moments.residual_rhs[:, trait_index],
        )
    )
    traces = np.concatenate(
        (trait_moments.genetic_traces, trait_moments.residual_traces)
    )
    return ContextNormalEquations(
        matrix=matrix,
        rhs=rhs,
        traces=traces,
        component_names=component_index.names + trait.residual_names,
        genetic_count=c,
        annotation_masses=trait_moments.annotation_masses,
        deleted_groups=requested,
        reference_genetic_gram=reference_moments.gram,
        transferred_genetic_gram=transferred,
        reference_n=reference_n,
        study_n=trait.n_samples,
    )


def assemble_contextual_normal_equations_v1(
    reference: ContextualReferenceArtifactV1,
    trait: ContextualTraitArtifactV1,
    *,
    trait_selector: str | int | None = None,
    deleted_groups: Sequence[str] = (),
) -> ContextNormalEquations:
    """Assemble one selected trait's canonical full small normal system."""
    validate_contextual_fit_compatibility_v1(reference, trait)
    trait_index, _ = _select_trait(trait, trait_selector)
    if isinstance(deleted_groups, (str, bytes)):
        raise ValueError("Deleted contextual fit V1 group IDs must be a sequence.")
    requested = tuple(deleted_groups)
    if any(not isinstance(value, str) or not value for value in requested):
        raise ValueError(
            "Deleted contextual fit V1 group IDs must be nonempty strings."
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Deleted contextual fit V1 group IDs must be unique.")
    unknown = set(requested) - set(reference.group_ids)
    if unknown:
        raise ValueError(f"Unknown approximate-LOO groups: {sorted(unknown)}.")
    reference_moments = reference_moments_after_deleting_groups_v1(
        reference, requested
    )
    trait_moments = trait_moments_after_deleting_groups_v1(trait, requested)
    return assemble_contextual_normal_equations_from_moments_v1(
        component_index=reference.component_index,
        reference_n=reference.n_samples,
        trait=trait,
        trait_index=trait_index,
        deleted_groups=requested,
        reference_moments=reference_moments,
        trait_moments=trait_moments,
    )


@dataclass(frozen=True)
class ContextualFitArtifactV1:
    """One selected trait's stable raw fit and optional interpretations."""

    manifest: Mapping[str, Any]
    component_index: ContextComponentIndex
    scale_plan: GenotypeScalePlanV1
    group_ids: tuple[str, ...]
    trait_ids: tuple[str, ...]
    residual_names: tuple[str, ...]
    selected_trait_id: str
    selected_trait_index: int
    normal_matrix: np.ndarray
    normal_rhs: np.ndarray
    traces: np.ndarray
    annotation_masses: np.ndarray
    reference_genetic_gram: np.ndarray
    transferred_genetic_gram: np.ndarray
    raw_coefficients: np.ndarray
    raw_genetic_coefficients: np.ndarray
    raw_residual_coefficients: np.ndarray
    raw_omegas: np.ndarray
    raw_loo_coefficients: np.ndarray
    raw_jackknife_covariance: np.ndarray
    raw_standard_errors: np.ndarray
    raw_solve_residual: np.ndarray
    raw_solve_eigenvalues: np.ndarray
    raw_solve_singular_values: np.ndarray
    raw_solve_retained_directions: np.ndarray
    raw_solve_null_space: np.ndarray
    psd_coefficients: np.ndarray | None = None
    psd_omegas: np.ndarray | None = None
    psd_minimum_eigenvalues: np.ndarray | None = None
    context_grid: np.ndarray | None = None
    basis_metric: np.ndarray | None = None
    raw_covariance_surfaces: np.ndarray | None = None
    raw_combined_covariance_surfaces: np.ndarray | None = None
    psd_covariance_surfaces: np.ndarray | None = None
    psd_combined_covariance_surface: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.component_index, ContextComponentIndex):
            raise ValueError("component_index must be a ContextComponentIndex.")
        if not isinstance(self.scale_plan, GenotypeScalePlanV1):
            raise ValueError("scale_plan must be a GenotypeScalePlanV1.")
        for name in ("group_ids", "trait_ids", "residual_names"):
            original = getattr(self, name)
            if not isinstance(original, (tuple, list)) or not original:
                raise ValueError(f"Contextual fit V1 {name} must be nonempty.")
            values = tuple(original)
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(
                    f"Contextual fit V1 {name} must contain nonempty strings."
                )
            if len(set(values)) != len(values):
                raise ValueError(f"Contextual fit V1 {name} must be unique.")
            object.__setattr__(self, name, values)
        if (
            isinstance(self.selected_trait_index, bool)
            or not isinstance(self.selected_trait_index, int)
            or self.selected_trait_index < 0
            or self.selected_trait_index >= len(self.trait_ids)
            or self.trait_ids[self.selected_trait_index] != self.selected_trait_id
        ):
            raise ValueError("Contextual fit V1 selected trait identity is invalid.")
        manifest = _json_value(self.manifest)
        present = (
            set(manifest.get("arrays", {})) if isinstance(manifest, Mapping) else set()
        )
        for name in _ALL_ARRAY_NAMES:
            value = getattr(self, name)
            if name in present:
                if value is None:
                    raise ValueError(f"Contextual fit V1 array {name!r} is missing.")
                object.__setattr__(self, name, _immutable_array(value))
            elif value is not None:
                raise ValueError(
                    f"Contextual fit V1 array {name!r} is not declared in the manifest."
                )
        _validate_artifact(self, manifest)
        object.__setattr__(self, "manifest", freeze_context_mapping(manifest))

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.manifest)

    @property
    def raw_rank(self) -> int:
        return int(self.manifest["solve"]["rank"])

    @property
    def n_samples(self) -> int:
        return int(self.manifest["dimensions"]["N_study"])

    def verify(self) -> None:
        _validate_artifact(self, _json_value(self.manifest))


def _surface_panels(
    coefficients: np.ndarray,
    components: ContextComponentIndex,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    replicate_coefficients = np.asarray(coefficients, dtype=np.float64)
    if replicate_coefficients.ndim == 1:
        replicate_coefficients = replicate_coefficients[None, :]
    panels = np.empty(
        (
            replicate_coefficients.shape[0],
            len(components.annotation_names),
            grid.shape[0],
            grid.shape[0],
        ),
        dtype=np.float64,
    )
    for replicate, values in enumerate(replicate_coefficients):
        omegas = coefficients_to_omegas(values, components)
        for annotation, omega in enumerate(omegas):
            panels[replicate, annotation] = context_covariance_surface(omega, grid)
    return panels, np.sum(panels, axis=1, dtype=np.float64)


def _validate_surface_inputs(
    context_grid: object | None,
    basis_metric: object | None,
    evaluation_grid_role: object | None,
    evaluation_grid_provenance_sha256: object | None,
    evaluation_grid_trusted_non_row: object,
    q: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if (context_grid is None) != (basis_metric is None):
        raise ValueError("context_grid and basis_metric must be supplied together.")
    if context_grid is None:
        if (
            evaluation_grid_role is not None
            or evaluation_grid_provenance_sha256 is not None
            or evaluation_grid_trusted_non_row is not False
        ):
            raise ValueError(
                "Evaluation-grid provenance cannot be supplied without surfaces."
            )
        return None, None, None
    if evaluation_grid_role != _EVALUATION_GRID_ROLE_V1:
        raise ValueError(
            "Surface publication requires the exact non-row evaluation-grid role."
        )
    provenance = _sha256(
        "evaluation_grid_provenance_sha256", evaluation_grid_provenance_sha256
    )
    if evaluation_grid_trusted_non_row is not True:
        raise ValueError(
            "Surface publication requires a trusted-caller non-row assertion."
        )
    grid = np.asarray(context_grid, dtype=np.float64)
    metric = np.asarray(basis_metric, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] < 1 or grid.shape[1] != q:
        raise ValueError("context_grid has an incompatible basis dimension.")
    if metric.shape != (q, q):
        raise ValueError("basis_metric has an incompatible shape.")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(metric)):
        raise ValueError("Context surface inputs must be finite.")
    metric_scale = max(float(np.max(np.abs(metric), initial=0.0)), 1.0)
    if np.max(np.abs(metric - metric.T), initial=0.0) > 1.0e-10 * metric_scale:
        raise ValueError("basis_metric must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    if eigenvalues[0] < -1.0e-12 * metric_scale:
        raise ValueError("basis_metric must be positive semidefinite.")
    return (
        np.ascontiguousarray(grid),
        np.ascontiguousarray(0.5 * (metric + metric.T)),
        provenance,
    )


def fit_contextual_model_v1(
    reference: ContextualReferenceArtifactV1,
    trait: ContextualTraitArtifactV1,
    *,
    trait_selector: str | int | None = None,
    rtol: float | None = None,
    context_grid: object | None = None,
    basis_metric: object | None = None,
    evaluation_grid_role: str | None = None,
    evaluation_grid_provenance_sha256: str | None = None,
    evaluation_grid_trusted_non_row: bool = False,
    project_psd: bool = False,
) -> ContextualFitArtifactV1:
    """Fit one trait from compact V1 summaries and publish every-group LOO."""
    if not isinstance(project_psd, bool):
        raise ValueError("project_psd must be boolean.")
    compatibility = validate_contextual_fit_compatibility_v1(reference, trait)
    trait_index, trait_id = _select_trait(trait, trait_selector)
    if isinstance(rtol, bool):
        raise ValueError("rtol must be finite and positive.")
    relative_tolerance = DEFAULT_FIT_V1_RTOL if rtol is None else float(rtol)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("rtol must be finite and positive.")
    mode = _annotation_mode(reference)
    if project_psd and mode is not AnnotationMode.STRICT_DISJOINT_BINARY_V1:
        raise ValueError(
            "PSD interpretation is unavailable for overlapping annotations."
        )
    q = reference.component_index.pair_index.num_basis
    grid, metric, grid_provenance = _validate_surface_inputs(
        context_grid,
        basis_metric,
        evaluation_grid_role,
        evaluation_grid_provenance_sha256,
        evaluation_grid_trusted_non_row,
        q,
    )

    groups = reference.group_ids
    if len(groups) < 2:
        raise ValueError("Every-group jackknife requires at least two groups.")
    if not np.array_equal(reference.group_variant_counts, trait.group_variant_counts):
        raise ValueError("Reference and trait group counts differ.")
    counts = np.asarray(reference.group_variant_counts, dtype=np.int64)
    if np.any(counts <= 0) or int(np.max(counts) - np.min(counts)) > 1:
        raise ValueError(
            "Every-group equal-weight jackknife requires balanced nonempty groups."
        )

    equations = assemble_contextual_normal_equations_v1(
        reference, trait, trait_selector=trait_index
    )
    solve = solve_context_normal_equations(
        equations, rtol=relative_tolerance, require_full_rank=True
    )
    c = len(reference.component_index)
    h = len(trait.residual_names)
    p = c + h
    raw = np.asarray(solve.coefficients, dtype=np.float64)
    genetic = raw[:c].copy()
    residual = raw[c:].copy()
    omegas = coefficients_to_omegas(genetic, reference.component_index)

    loo = np.empty((len(groups), p), dtype=np.float64)
    for index, group in enumerate(groups):
        deleted = assemble_contextual_normal_equations_v1(
            reference,
            trait,
            trait_selector=trait_index,
            deleted_groups=(group,),
        )
        try:
            deleted_solve = solve_context_normal_equations(
                deleted, rtol=relative_tolerance, require_full_rank=True
            )
        except ContextRankError as exc:
            raise ContextJackknifeError(group, exc) from exc
        loo[index] = deleted_solve.coefficients
    centered = loo - np.mean(loo, axis=0, keepdims=True)
    covariance = (len(groups) - 1.0) / len(groups) * (centered.T @ centered)
    covariance = 0.5 * (covariance + covariance.T)
    covariance_scale = max(float(np.max(np.abs(covariance), initial=0.0)), 1.0)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    if covariance_eigenvalues[0] < -1.0e-10 * covariance_scale:
        raise RuntimeError(
            "Computed raw jackknife covariance is materially indefinite."
        )
    diagonal = np.diag(covariance)
    if np.min(diagonal) < -1.0e-10 * covariance_scale:
        raise RuntimeError("Computed raw jackknife covariance has negative variance.")
    standard_errors = np.sqrt(np.maximum(diagonal, 0.0))

    projection: PSDProjectionResult | None = None
    if project_psd:
        projection = project_genetic_coefficients_psd(
            genetic,
            covariance[:c, :c],
            reference.component_index,
            annotations_disjoint=True,
        )

    arrays: dict[str, np.ndarray] = {
        "normal_matrix": np.asarray(equations.matrix),
        "normal_rhs": np.asarray(equations.rhs),
        "traces": np.asarray(equations.traces),
        "annotation_masses": np.asarray(equations.annotation_masses),
        "reference_genetic_gram": np.asarray(equations.reference_genetic_gram),
        "transferred_genetic_gram": np.asarray(equations.transferred_genetic_gram),
        "raw_coefficients": raw,
        "raw_genetic_coefficients": genetic,
        "raw_residual_coefficients": residual,
        "raw_omegas": omegas,
        "raw_loo_coefficients": loo,
        "raw_jackknife_covariance": covariance,
        "raw_standard_errors": standard_errors,
        "raw_solve_residual": np.asarray(solve.solve_residual),
        "raw_solve_eigenvalues": np.asarray(solve.diagnostics.eigenvalues),
        "raw_solve_singular_values": np.asarray(solve.diagnostics.singular_values),
        "raw_solve_retained_directions": np.asarray(solve.retained_directions),
        "raw_solve_null_space": np.asarray(solve.null_space),
    }
    if projection is not None:
        arrays.update(
            {
                "psd_coefficients": projection.projected_coefficients,
                "psd_omegas": projection.projected_omegas,
                "psd_minimum_eigenvalues": projection.minimum_eigenvalues,
            }
        )
    surface_replicates: list[str] = []
    if grid is not None and metric is not None:
        replicate_genetic = np.vstack((genetic, loo[:, :c]))
        raw_surfaces, combined_surfaces = _surface_panels(
            replicate_genetic, reference.component_index, grid
        )
        arrays.update(
            {
                "context_grid": grid,
                "basis_metric": metric,
                "raw_covariance_surfaces": raw_surfaces,
                "raw_combined_covariance_surfaces": combined_surfaces,
            }
        )
        surface_replicates = ["full", *(f"delete:{group}" for group in groups)]
        if projection is not None:
            psd_surfaces, psd_combined = _surface_panels(
                projection.projected_coefficients,
                reference.component_index,
                grid,
            )
            arrays["psd_covariance_surfaces"] = psd_surfaces[0]
            arrays["psd_combined_covariance_surface"] = psd_combined[0]

    immutable_arrays = {name: _immutable_array(value) for name, value in arrays.items()}
    layouts = dict(_BASE_LAYOUTS)
    if projection is not None:
        layouts.update(_PSD_LAYOUTS)
    if grid is not None:
        layouts.update(_SURFACE_LAYOUTS)
        if projection is not None:
            layouts.update(_PSD_SURFACE_LAYOUTS)
    output_contract = dict(annotation_output_contract(mode))
    interpretation = {
        "annotation_mode": mode.value,
        "annotation_output_contract": output_contract,
        "per_annotation_surface_label": output_contract["per_annotation_role"],
        "combined_total_surface_default": (
            mode is AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1
        ),
        "raw_coefficients_are_primary": True,
    }
    condition_number = (
        None if not np.isfinite(solve.condition_number) else solve.condition_number
    )
    projection_manifest: dict[str, Any] = {
        "requested": projection is not None,
        "method": None,
        "raw_output_replaced": False,
        "diagnostics": None,
    }
    if projection is not None:
        projection_manifest = {
            "requested": True,
            "method": "covariance_weighted_psd_projection_v1",
            "raw_output_replaced": False,
            "diagnostics": {
                "distance": projection.distance,
                "euclidean_distance": projection.euclidean_distance,
                "covariance_rank": projection.covariance_rank,
                "covariance_nullity": projection.covariance_nullity,
                "optimizer_success": projection.optimizer_success,
                "optimizer_message": projection.optimizer_message,
                "tie_break_applied": projection.tie_break_applied,
                "cleanup_norm": projection.cleanup_norm,
            },
        }
    schema = ContextSchemaIdentityV1(
        artifact_family=ArtifactFamily.FIT,
        logical_schema_version=LogicalSchemaVersion.FIT_V1,
        grouped_encoding_version=GroupedEncodingVersion(
            reference.manifest["grouped_encoding_version"]
        ),
        native_api_version=reference.manifest["native_api_version"],
        native_backend_version=_FIT_BACKEND_V1,
        build_id=reference.manifest["build_id"],
    )
    manifest: dict[str, Any] = {
        "magic": CONTEXTUAL_FIT_V1_MAGIC,
        **schema.to_dict(),
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "scientific_policy": dict(_SCIENTIFIC_POLICY),
        "dimensions": {
            "N_reference": reference.n_samples,
            "N_study": trait.n_samples,
            "M": reference.n_variants,
            "Q": q,
            "K": len(reference.component_index.annotation_names),
            "P_g": len(reference.component_index.pair_index),
            "C": c,
            "H": h,
            "P_total": p,
            "J": len(groups),
            "L": trait.n_traits,
            "E": 0 if grid is None else grid.shape[0],
        },
        "identity": {
            "reference_manifest_sha256": reference.manifest_sha256,
            "trait_manifest_sha256": trait.manifest_sha256,
            "python_fit_implementation_provenance_sha256": compatibility[
                "python_fit_implementation_provenance_sha256"
            ],
        },
        "compatibility": dict(compatibility),
        "genotype_scale_plan": reference.scale_plan.to_dict(),
        "genotype_scale_plan_sha256": reference.scale_plan.digest,
        "selection": {
            "trait_id": trait_id,
            "trait_index": trait_index,
            "selector_required_for_multi_trait": True,
            "one_selected_trait_per_fit": True,
        },
        "interpretation": interpretation,
        "solve": {
            "method": "full_symmetric_eigendecomposition_rank_checked_v1",
            "require_full_rank": True,
            "relative_tolerance": solve.relative_tolerance,
            "absolute_tolerance": solve.absolute_tolerance,
            "effective_tolerance": solve.diagnostics.tolerance,
            "rank": solve.rank,
            "dimension": p,
            "condition_number": condition_number,
            "relative_residual": solve.relative_residual,
            "minimum_gram_eigenvalue": solve.minimum_gram_eigenvalue,
            "ridge": False,
            "pseudoinverse": False,
        },
        "jackknife": {
            "method": "every_frozen_group_equal_delete_one_v1",
            "groups": list(groups),
            "group_count": len(groups),
            "minimum_variants": int(np.min(counts)),
            "maximum_variants": int(np.max(counts)),
            "covariance": "equal_group_delete_one_joint_raw_coefficients",
            "deletion_semantics": "approximate_summary_only_v1",
            "full_reference_same_person_reused": True,
            "residual_only_moments_reused": True,
        },
        "optional_interpretation": projection_manifest,
        "surfaces": {
            "published": grid is not None,
            "replicates": surface_replicates,
            "raw_all_deletions": grid is not None,
            "combined_total_published": grid is not None,
            "psd_full_published": grid is not None and projection is not None,
            "semantics": "coefficient_covariance_surface_v1",
            "evaluation_grid_role": (
                _EVALUATION_GRID_ROLE_V1 if grid is not None else None
            ),
            "evaluation_grid_provenance_sha256": grid_provenance,
            "trusted_caller_non_row_assertion": grid is not None,
        },
        "maps": {
            "pair_map": reference.component_index.pair_index.to_dict(),
            "pair_map_sha256": reference.component_index.pair_index.digest,
            "component_map": reference.component_index.to_dict(),
            "component_map_sha256": reference.component_index.digest,
            "group_ids": list(groups),
            "trait_ids": list(trait.trait_ids),
            "residual_names": list(trait.residual_names),
        },
        "layouts": layouts,
        "arrays": {
            name: _array_envelope(value) for name, value in immutable_arrays.items()
        },
        "terminal_status": "published",
    }
    return ContextualFitArtifactV1(
        manifest=manifest,
        component_index=reference.component_index,
        scale_plan=reference.scale_plan,
        group_ids=groups,
        trait_ids=trait.trait_ids,
        residual_names=trait.residual_names,
        selected_trait_id=trait_id,
        selected_trait_index=trait_index,
        **{name: immutable_arrays.get(name) for name in _ALL_ARRAY_NAMES},
    )


def _expected_array_shapes(
    dimensions: Mapping[str, Any], arrays: set[str]
) -> dict[str, tuple[int, ...]]:
    c = int(dimensions["C"])
    h = int(dimensions["H"])
    p = int(dimensions["P_total"])
    k = int(dimensions["K"])
    q = int(dimensions["Q"])
    j = int(dimensions["J"])
    e = int(dimensions["E"])
    shapes: dict[str, tuple[int, ...]] = {
        "normal_matrix": (p, p),
        "normal_rhs": (p,),
        "traces": (p,),
        "annotation_masses": (k,),
        "reference_genetic_gram": (c, c),
        "transferred_genetic_gram": (c, c),
        "raw_coefficients": (p,),
        "raw_genetic_coefficients": (c,),
        "raw_residual_coefficients": (h,),
        "raw_omegas": (k, q, q),
        "raw_loo_coefficients": (j, p),
        "raw_jackknife_covariance": (p, p),
        "raw_standard_errors": (p,),
        "raw_solve_residual": (p,),
        "raw_solve_eigenvalues": (p,),
        "raw_solve_singular_values": (p,),
        "raw_solve_retained_directions": (p, p),
        "raw_solve_null_space": (p, 0),
        "psd_coefficients": (c,),
        "psd_omegas": (k, q, q),
        "psd_minimum_eigenvalues": (k,),
        "context_grid": (e, q),
        "basis_metric": (q, q),
        "raw_covariance_surfaces": (j + 1, k, e, e),
        "raw_combined_covariance_surfaces": (j + 1, e, e),
        "psd_covariance_surfaces": (k, e, e),
        "psd_combined_covariance_surface": (e, e),
    }
    return {name: shapes[name] for name in arrays}


def _validate_manifest_before_arrays(
    manifest: Any,
) -> tuple[
    ContextComponentIndex,
    GenotypeScalePlanV1,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
    int,
]:
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Contextual fit V1 manifest key mismatch.")
    if manifest.get("magic") != CONTEXTUAL_FIT_V1_MAGIC:
        raise ValueError("Contextual fit V1 magic mismatch.")
    try:
        schema = ContextSchemaIdentityV1(
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
    except (TypeError, ValueError) as exc:
        raise ValueError("Contextual fit V1 schema identity is invalid.") from exc
    if schema.artifact_family is not ArtifactFamily.FIT:
        raise ValueError("Contextual fit V1 has the wrong artifact family.")
    if manifest.get("native_backend_version") != _FIT_BACKEND_V1:
        raise ValueError("Contextual fit V1 Python backend identity mismatch.")
    if manifest.get("feature_mode") != RAW_PROJECTED_FEATURE_MODE:
        raise ValueError("Contextual fit V1 feature mode mismatch.")
    if manifest.get("scientific_policy") != _SCIENTIFIC_POLICY:
        raise ValueError("Contextual fit V1 scientific policy mismatch.")
    if manifest.get("terminal_status") != "published":
        raise ValueError("Contextual fit V1 is not published.")

    dimensions = manifest.get("dimensions")
    dimension_keys = {
        "N_reference",
        "N_study",
        "M",
        "Q",
        "K",
        "P_g",
        "C",
        "H",
        "P_total",
        "J",
        "L",
        "E",
    }
    if not isinstance(dimensions, Mapping) or set(dimensions) != dimension_keys:
        raise ValueError("Contextual fit V1 dimension schema mismatch.")
    for name in dimension_keys - {"E"}:
        minimum = 2 if name in {"N_reference", "J"} else 1
        _positive_int(name, dimensions[name], minimum=minimum)
    e = _positive_int("E", dimensions["E"], minimum=0)

    maps = manifest.get("maps")
    map_keys = {
        "pair_map",
        "pair_map_sha256",
        "component_map",
        "component_map_sha256",
        "group_ids",
        "trait_ids",
        "residual_names",
    }
    if not isinstance(maps, Mapping) or set(maps) != map_keys:
        raise ValueError("Contextual fit V1 map schema mismatch.")
    pair_map = maps.get("pair_map")
    component_map = maps.get("component_map")
    if not isinstance(pair_map, Mapping) or not isinstance(component_map, Mapping):
        raise ValueError("Contextual fit V1 pair/component map is invalid.")
    annotation_names = component_map.get("annotation_names")
    if not isinstance(annotation_names, list):
        raise ValueError("Contextual fit V1 annotation names are invalid.")
    components = ContextComponentIndex(
        tuple(str(value) for value in annotation_names),
        ContextPairIndex(_positive_int("num_basis", pair_map.get("num_basis"))),
    )
    if (
        pair_map != components.pair_index.to_dict()
        or component_map != components.to_dict()
    ):
        raise ValueError("Contextual fit V1 pair/component map is noncanonical.")
    if maps.get("pair_map_sha256") != components.pair_index.digest:
        raise ValueError("Contextual fit V1 pair-map digest mismatch.")
    if maps.get("component_map_sha256") != components.digest:
        raise ValueError("Contextual fit V1 component-map digest mismatch.")

    def names(axis: str) -> tuple[str, ...]:
        values = maps.get(axis)
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"Contextual fit V1 {axis} are invalid.")
        result = tuple(values)
        if len(set(result)) != len(result):
            raise ValueError(f"Contextual fit V1 {axis} must be unique.")
        return result

    groups = names("group_ids")
    trait_ids = names("trait_ids")
    residual_names = names("residual_names")
    scale_plan = _scale_plan_from_manifest(manifest.get("genotype_scale_plan"))
    if manifest.get("genotype_scale_plan_sha256") != scale_plan.digest:
        raise ValueError("Contextual fit V1 scale-plan digest mismatch.")

    if (
        dimensions["Q"] != components.pair_index.num_basis
        or dimensions["K"] != len(components.annotation_names)
        or dimensions["P_g"] != len(components.pair_index)
        or dimensions["C"] != len(components)
        or dimensions["H"] != len(residual_names)
        or dimensions["P_total"] != dimensions["C"] + dimensions["H"]
        or dimensions["J"] != len(groups)
        or dimensions["L"] != len(trait_ids)
    ):
        raise ValueError("Contextual fit V1 dimensions disagree with maps.")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "trait_id",
        "trait_index",
        "selector_required_for_multi_trait",
        "one_selected_trait_per_fit",
    }:
        raise ValueError("Contextual fit V1 trait selection schema mismatch.")
    selected_index = _positive_int(
        "trait_index", selection.get("trait_index"), minimum=0
    )
    selected_id = selection.get("trait_id")
    if (
        selected_index >= len(trait_ids)
        or not isinstance(selected_id, str)
        or trait_ids[selected_index] != selected_id
        or selection.get("selector_required_for_multi_trait") is not True
        or selection.get("one_selected_trait_per_fit") is not True
    ):
        raise ValueError("Contextual fit V1 trait selection is invalid.")

    interpretation = manifest.get("interpretation")
    if not isinstance(interpretation, Mapping) or set(interpretation) != {
        "annotation_mode",
        "annotation_output_contract",
        "per_annotation_surface_label",
        "combined_total_surface_default",
        "raw_coefficients_are_primary",
    }:
        raise ValueError("Contextual fit V1 interpretation schema mismatch.")
    try:
        mode = AnnotationMode(interpretation["annotation_mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Contextual fit V1 annotation mode is invalid.") from exc
    contract = dict(annotation_output_contract(mode))
    if (
        interpretation["annotation_output_contract"] != contract
        or interpretation["per_annotation_surface_label"]
        != contract["per_annotation_role"]
        or interpretation["combined_total_surface_default"]
        is not (mode is AnnotationMode.GENERIC_NONNEGATIVE_WEIGHTS_V1)
        or interpretation["raw_coefficients_are_primary"] is not True
    ):
        raise ValueError("Contextual fit V1 annotation interpretation mismatch.")

    solve = manifest.get("solve")
    solve_keys = {
        "method",
        "require_full_rank",
        "relative_tolerance",
        "absolute_tolerance",
        "effective_tolerance",
        "rank",
        "dimension",
        "condition_number",
        "relative_residual",
        "minimum_gram_eigenvalue",
        "ridge",
        "pseudoinverse",
    }
    if not isinstance(solve, Mapping) or set(solve) != solve_keys:
        raise ValueError("Contextual fit V1 raw solve schema mismatch.")
    if (
        solve["method"] != "full_symmetric_eigendecomposition_rank_checked_v1"
        or solve["require_full_rank"] is not True
        or solve["ridge"] is not False
        or solve["pseudoinverse"] is not False
        or _positive_int("solve.rank", solve["rank"]) != dimensions["P_total"]
        or _positive_int("solve.dimension", solve["dimension"]) != dimensions["P_total"]
    ):
        raise ValueError("Contextual fit V1 raw solve policy mismatch.")
    for name in (
        "relative_tolerance",
        "absolute_tolerance",
        "effective_tolerance",
    ):
        value = solve[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Contextual fit V1 solve {name} is invalid.")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Contextual fit V1 solve {name} is invalid.")
    for name in ("relative_residual", "minimum_gram_eigenvalue"):
        value = solve[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Contextual fit V1 solve {name} is invalid.")
        if not np.isfinite(value):
            raise ValueError(f"Contextual fit V1 solve {name} is invalid.")
    condition = solve["condition_number"]
    if condition is not None and (
        isinstance(condition, bool)
        or not isinstance(condition, (int, float))
        or not np.isfinite(condition)
        or condition < 1.0
    ):
        raise ValueError("Contextual fit V1 solve condition number is invalid.")

    jackknife = manifest.get("jackknife")
    jackknife_keys = {
        "method",
        "groups",
        "group_count",
        "minimum_variants",
        "maximum_variants",
        "covariance",
        "deletion_semantics",
        "full_reference_same_person_reused",
        "residual_only_moments_reused",
    }
    if not isinstance(jackknife, Mapping) or set(jackknife) != jackknife_keys:
        raise ValueError("Contextual fit V1 jackknife schema mismatch.")
    minimum_variants = _positive_int(
        "jackknife.minimum_variants", jackknife["minimum_variants"]
    )
    maximum_variants = _positive_int(
        "jackknife.maximum_variants", jackknife["maximum_variants"]
    )
    if (
        jackknife["method"] != "every_frozen_group_equal_delete_one_v1"
        or jackknife["groups"] != list(groups)
        or _positive_int("jackknife.group_count", jackknife["group_count"], minimum=2)
        != len(groups)
        or maximum_variants - minimum_variants > 1
        or jackknife["covariance"] != "equal_group_delete_one_joint_raw_coefficients"
        or jackknife["deletion_semantics"] != "approximate_summary_only_v1"
        or jackknife["full_reference_same_person_reused"] is not True
        or jackknife["residual_only_moments_reused"] is not True
    ):
        raise ValueError("Contextual fit V1 jackknife policy mismatch.")

    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("Contextual fit V1 array schema mismatch.")
    array_names = set(arrays)
    if not set(_BASE_ARRAY_NAMES).issubset(array_names) or not array_names.issubset(
        _ALL_ARRAY_NAMES
    ):
        raise ValueError("Contextual fit V1 array fields mismatch.")
    projection = manifest.get("optional_interpretation")
    surfaces = manifest.get("surfaces")
    if not isinstance(projection, Mapping) or not isinstance(surfaces, Mapping):
        raise ValueError("Contextual fit V1 optional output schema mismatch.")
    if set(projection) != {
        "requested",
        "method",
        "raw_output_replaced",
        "diagnostics",
    }:
        raise ValueError("Contextual fit V1 PSD interpretation schema mismatch.")
    if not isinstance(projection["requested"], bool):
        raise ValueError("Contextual fit V1 PSD request flag is invalid.")
    projected = projection.get("requested") is True
    if projection["raw_output_replaced"] is not False:
        raise ValueError("Contextual fit V1 PSD output replaced the raw estimate.")
    if projected:
        diagnostics = projection["diagnostics"]
        diagnostic_keys = {
            "distance",
            "euclidean_distance",
            "covariance_rank",
            "covariance_nullity",
            "optimizer_success",
            "optimizer_message",
            "tie_break_applied",
            "cleanup_norm",
        }
        if (
            projection["method"] != "covariance_weighted_psd_projection_v1"
            or not isinstance(diagnostics, Mapping)
            or set(diagnostics) != diagnostic_keys
            or diagnostics["optimizer_success"] is not True
            or not isinstance(diagnostics["optimizer_message"], str)
            or not isinstance(diagnostics["tie_break_applied"], bool)
        ):
            raise ValueError("Contextual fit V1 PSD interpretation is invalid.")
        covariance_rank = _positive_int(
            "PSD covariance_rank", diagnostics["covariance_rank"], minimum=0
        )
        covariance_nullity = _positive_int(
            "PSD covariance_nullity", diagnostics["covariance_nullity"], minimum=0
        )
        if covariance_rank + covariance_nullity != dimensions["C"]:
            raise ValueError("Contextual fit V1 PSD covariance rank is invalid.")
        for name in ("distance", "euclidean_distance", "cleanup_norm"):
            value = diagnostics[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"Contextual fit V1 PSD diagnostic {name} is invalid.")
    elif projection["method"] is not None or projection["diagnostics"] is not None:
        raise ValueError("Contextual fit V1 unrequested PSD fields are populated.")
    surface_keys = {
        "published",
        "replicates",
        "raw_all_deletions",
        "combined_total_published",
        "psd_full_published",
        "semantics",
        "evaluation_grid_role",
        "evaluation_grid_provenance_sha256",
        "trusted_caller_non_row_assertion",
    }
    if set(surfaces) != surface_keys or not isinstance(surfaces["published"], bool):
        raise ValueError("Contextual fit V1 surface schema mismatch.")
    surface_published = surfaces.get("published") is True
    expected_replicates = ["full", *(f"delete:{group}" for group in groups)]
    if (
        surfaces["replicates"] != (expected_replicates if surface_published else [])
        or surfaces["raw_all_deletions"] is not surface_published
        or surfaces["combined_total_published"] is not surface_published
        or surfaces["psd_full_published"] is not (surface_published and projected)
        or surfaces["semantics"] != "coefficient_covariance_surface_v1"
        or surfaces["evaluation_grid_role"]
        != (_EVALUATION_GRID_ROLE_V1 if surface_published else None)
        or surfaces["trusted_caller_non_row_assertion"] is not surface_published
    ):
        raise ValueError("Contextual fit V1 surface policy mismatch.")
    if surface_published:
        _sha256(
            "surfaces.evaluation_grid_provenance_sha256",
            surfaces["evaluation_grid_provenance_sha256"],
        )
    elif surfaces["evaluation_grid_provenance_sha256"] is not None:
        raise ValueError("Contextual fit V1 unpublished grid has provenance.")
    expected_names = set(_BASE_ARRAY_NAMES)
    if projected:
        expected_names.update(_PSD_ARRAY_NAMES)
    if surface_published:
        expected_names.update(_SURFACE_ARRAY_NAMES)
        if projected:
            expected_names.update(_PSD_SURFACE_ARRAY_NAMES)
    if array_names != expected_names:
        raise ValueError("Contextual fit V1 optional array presence is inconsistent.")
    if surface_published != (e > 0):
        raise ValueError("Contextual fit V1 surface dimensions are inconsistent.")
    expected_shapes = _expected_array_shapes(dimensions, array_names)
    for name, shape in expected_shapes.items():
        envelope = arrays[name]
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "dtype",
            "shape",
            "sha256",
        }:
            raise ValueError(f"Contextual fit V1 {name} envelope is invalid.")
        if envelope.get("dtype") != "<f8" or envelope.get("shape") != list(shape):
            raise ValueError(f"Contextual fit V1 {name} schema is invalid.")
        _sha256(f"arrays.{name}.sha256", envelope.get("sha256"))
    layouts = manifest.get("layouts")
    expected_layouts = {name: _BASE_LAYOUTS[name] for name in _BASE_ARRAY_NAMES}
    if projected:
        expected_layouts.update(_PSD_LAYOUTS)
    if surface_published:
        expected_layouts.update(_SURFACE_LAYOUTS)
        if projected:
            expected_layouts.update(_PSD_SURFACE_LAYOUTS)
    if layouts != expected_layouts:
        raise ValueError("Contextual fit V1 array layouts mismatch.")
    for axis_names in layouts.values():
        if "sample" in axis_names or "variant" in axis_names:
            raise ValueError("Contextual fit V1 contains an individual-level axis.")

    identities = manifest.get("identity")
    compatibility = manifest.get("compatibility")
    if not isinstance(identities, Mapping) or set(identities) != {
        "reference_manifest_sha256",
        "trait_manifest_sha256",
        "python_fit_implementation_provenance_sha256",
    }:
        raise ValueError("Contextual fit V1 source identity schema mismatch.")
    for name, value in identities.items():
        _sha256(name, value)
    compatibility_keys = {
        "reference_manifest_sha256",
        "trait_manifest_sha256",
        "matched_native_api_version",
        "reference_native_backend_version",
        "trait_native_backend_version",
        "backend_pair_policy",
        "matched_build_id",
        "matched_native_build_provenance_sha256",
        "matched_component_map_sha256",
        "matched_scale_plan_sha256",
        "matched_annotation_mode",
        "matched_group_sequence",
        "matched_source_tree_sha256",
        "matched_variant_order_allele_sha256",
        "matched_basis_specification_sha256",
        "matched_basis_calibration_sha256",
        "matched_annotation_map_sha256",
        "matched_group_map_sha256",
        "matched_annotation_masses_sha256",
        "matched_group_annotation_masses_sha256",
        "matched_group_variant_counts_sha256",
        "matched_numeric_policy",
        "matched_deletion_policy_sha256",
        "matched_grouped_encoding_version",
        "python_fit_implementation_provenance",
        "python_fit_implementation_provenance_sha256",
    }
    if (
        not isinstance(compatibility, Mapping)
        or set(compatibility) != compatibility_keys
    ):
        raise ValueError("Contextual fit V1 compatibility evidence is invalid.")
    if (
        compatibility.get("reference_manifest_sha256")
        != identities["reference_manifest_sha256"]
        or compatibility.get("trait_manifest_sha256")
        != identities["trait_manifest_sha256"]
        or compatibility.get("matched_native_api_version")
        != manifest["native_api_version"]
        or compatibility.get("reference_native_backend_version")
        != _REFERENCE_NATIVE_BACKEND_V1
        or compatibility.get("trait_native_backend_version") != _TRAIT_NATIVE_BACKEND_V1
        or compatibility.get("backend_pair_policy") != _BACKEND_PAIR_POLICY_V1
        or compatibility.get("matched_build_id") != manifest["build_id"]
        or compatibility.get("matched_component_map_sha256") != components.digest
        or compatibility.get("matched_scale_plan_sha256") != scale_plan.digest
        or compatibility.get("matched_annotation_mode") != mode.value
        or compatibility.get("matched_group_sequence") != list(groups)
        or compatibility.get("matched_numeric_policy") != "fp64_v1"
        or compatibility.get("matched_grouped_encoding_version")
        != manifest["grouped_encoding_version"]
        or compatibility.get("python_fit_implementation_provenance_sha256")
        != identities["python_fit_implementation_provenance_sha256"]
    ):
        raise ValueError("Contextual fit V1 compatibility identities disagree.")
    provenance = compatibility["python_fit_implementation_provenance"]
    validate_python_source_runtime_provenance(
        provenance,
        ordered_modules=_PYTHON_FIT_MODULES,
        policy=_PYTHON_FIT_PROVENANCE_POLICY,
        include_scipy=True,
        family="Contextual fit V1",
    )
    if (
        canonical_sha256(provenance)
        != compatibility["python_fit_implementation_provenance_sha256"]
    ):
        raise ValueError("Contextual fit V1 Python provenance digest mismatch.")
    for name in (
        "matched_source_tree_sha256",
        "matched_native_build_provenance_sha256",
        "matched_variant_order_allele_sha256",
        "matched_basis_specification_sha256",
        "matched_basis_calibration_sha256",
        "matched_annotation_map_sha256",
        "matched_group_map_sha256",
        "matched_annotation_masses_sha256",
        "matched_group_annotation_masses_sha256",
        "matched_group_variant_counts_sha256",
        "matched_deletion_policy_sha256",
    ):
        _sha256(name, compatibility[name])
    return (
        components,
        scale_plan,
        groups,
        trait_ids,
        residual_names,
        selected_id,
        selected_index,
    )


def _validate_artifact(
    artifact: ContextualFitArtifactV1, manifest: Mapping[str, Any]
) -> None:
    (
        components,
        scale_plan,
        groups,
        trait_ids,
        residual_names,
        selected_id,
        selected_index,
    ) = _validate_manifest_before_arrays(manifest)
    if components.digest != artifact.component_index.digest:
        raise ValueError("Contextual fit V1 component index mismatch.")
    if scale_plan.digest != artifact.scale_plan.digest:
        raise ValueError("Contextual fit V1 scale plan mismatch.")
    if (
        groups != artifact.group_ids
        or trait_ids != artifact.trait_ids
        or residual_names != artifact.residual_names
        or selected_id != artifact.selected_trait_id
        or selected_index != artifact.selected_trait_index
    ):
        raise ValueError("Contextual fit V1 semantic maps mismatch.")
    dimensions = manifest["dimensions"]
    expected_shapes = _expected_array_shapes(dimensions, set(manifest["arrays"]))
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(artifact, name))
        if value.dtype != np.dtype("<f8") or value.shape != shape:
            raise ValueError(
                f"Contextual fit V1 array {name!r} has invalid dtype or shape."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Contextual fit V1 array {name!r} is non-finite.")
        if manifest["arrays"][name] != _array_envelope(value):
            raise ValueError(f"Contextual fit V1 array digest mismatch for {name}.")
    p = int(dimensions["P_total"])
    c = int(dimensions["C"])
    if _relative_max(artifact.normal_matrix, artifact.normal_matrix.T) > 1.0e-12:
        raise ValueError("Contextual fit V1 normal matrix is not symmetric.")
    if np.any(artifact.annotation_masses <= 0.0):
        raise ValueError("Contextual fit V1 annotation masses must be positive.")
    if (
        _relative_max(artifact.normal_matrix[:c, :c], artifact.transferred_genetic_gram)
        > 1.0e-13
    ):
        raise ValueError("Contextual fit V1 transferred Gram assembly mismatch.")
    if (
        _relative_max(
            artifact.raw_jackknife_covariance, artifact.raw_jackknife_covariance.T
        )
        > 1.0e-12
    ):
        raise ValueError("Contextual fit V1 jackknife covariance is not symmetric.")
    if not np.array_equal(
        artifact.raw_coefficients[:c], artifact.raw_genetic_coefficients
    ):
        raise ValueError("Contextual fit V1 genetic coefficient split mismatch.")
    if not np.array_equal(
        artifact.raw_coefficients[c:], artifact.raw_residual_coefficients
    ):
        raise ValueError("Contextual fit V1 residual coefficient split mismatch.")
    expected_omegas = coefficients_to_omegas(
        artifact.raw_genetic_coefficients, artifact.component_index
    )
    if _relative_max(expected_omegas, artifact.raw_omegas) > 1.0e-13:
        raise ValueError("Contextual fit V1 raw Omega packing mismatch.")
    solve = manifest["solve"]
    if not isinstance(solve, Mapping) or solve.get("rank") != p:
        raise ValueError("Contextual fit V1 raw solve is not full rank.")
    if solve.get("ridge") is not False or solve.get("pseudoinverse") is not False:
        raise ValueError("Contextual fit V1 raw solve contains hidden regularization.")
    if artifact.raw_solve_null_space.shape != (p, 0):
        raise ValueError("Contextual fit V1 full-rank solve has a null space.")
    solve_residual = (
        artifact.normal_matrix @ artifact.raw_coefficients - artifact.normal_rhs
    )
    if _relative_max(solve_residual, artifact.raw_solve_residual) > 1.0e-10:
        raise ValueError("Contextual fit V1 solve residual mismatch.")
    relative_residual = float(
        np.linalg.norm(artifact.raw_solve_residual)
        / max(1.0, np.linalg.norm(artifact.normal_rhs))
    )
    if abs(relative_residual - float(solve["relative_residual"])) > 1.0e-12:
        raise ValueError("Contextual fit V1 relative solve residual mismatch.")
    eigenvalues = np.linalg.eigvalsh(artifact.normal_matrix)
    singular_values = np.sort(np.abs(eigenvalues))[::-1]
    if _relative_max(eigenvalues, artifact.raw_solve_eigenvalues) > 1.0e-12:
        raise ValueError("Contextual fit V1 solve eigenvalues mismatch.")
    if _relative_max(singular_values, artifact.raw_solve_singular_values) > 1.0e-12:
        raise ValueError("Contextual fit V1 solve singular values mismatch.")
    directions = artifact.raw_solve_retained_directions
    if _relative_max(directions.T @ directions, np.eye(p)) > 1.0e-12:
        raise ValueError("Contextual fit V1 retained directions are not orthonormal.")
    if (
        _relative_max(
            artifact.normal_matrix @ directions,
            directions * artifact.raw_solve_eigenvalues[None, :],
        )
        > 1.0e-10
    ):
        raise ValueError("Contextual fit V1 retained eigendirections mismatch.")
    if abs(float(solve["minimum_gram_eigenvalue"]) - eigenvalues[0]) > (
        1.0e-12 * max(1.0, abs(eigenvalues[0]))
    ):
        raise ValueError("Contextual fit V1 minimum Gram eigenvalue mismatch.")
    centered = artifact.raw_loo_coefficients - np.mean(
        artifact.raw_loo_coefficients, axis=0, keepdims=True
    )
    expected_covariance = (len(groups) - 1.0) / len(groups) * (centered.T @ centered)
    expected_covariance = 0.5 * (expected_covariance + expected_covariance.T)
    if _relative_max(expected_covariance, artifact.raw_jackknife_covariance) > 1.0e-12:
        raise ValueError("Contextual fit V1 jackknife covariance mismatch.")
    expected_standard_errors = np.sqrt(
        np.maximum(np.diag(artifact.raw_jackknife_covariance), 0.0)
    )
    if _relative_max(expected_standard_errors, artifact.raw_standard_errors) > 1.0e-12:
        raise ValueError("Contextual fit V1 standard errors mismatch.")
    if artifact.context_grid is not None:
        _validate_surface_inputs(
            artifact.context_grid,
            artifact.basis_metric,
            manifest["surfaces"]["evaluation_grid_role"],
            manifest["surfaces"]["evaluation_grid_provenance_sha256"],
            manifest["surfaces"]["trusted_caller_non_row_assertion"],
            artifact.component_index.pair_index.num_basis,
        )
        expected_surfaces, expected_combined = _surface_panels(
            np.vstack(
                (
                    artifact.raw_genetic_coefficients,
                    artifact.raw_loo_coefficients[:, :c],
                )
            ),
            artifact.component_index,
            artifact.context_grid,
        )
        if _relative_max(expected_surfaces, artifact.raw_covariance_surfaces) > 1.0e-12:
            raise ValueError("Contextual fit V1 raw covariance surfaces mismatch.")
        if (
            _relative_max(expected_combined, artifact.raw_combined_covariance_surfaces)
            > 1.0e-12
        ):
            raise ValueError("Contextual fit V1 combined covariance surfaces mismatch.")
    projection = manifest["optional_interpretation"]
    if projection["requested"]:
        if (
            manifest["interpretation"]["annotation_mode"]
            != AnnotationMode.STRICT_DISJOINT_BINARY_V1.value
        ):
            raise ValueError(
                "Contextual fit V1 overlap artifact cannot contain PSD output."
            )
        if np.min(artifact.psd_minimum_eigenvalues) < -1.0e-10:
            raise ValueError("Contextual fit V1 PSD interpretation is infeasible.")
        expected_psd_omegas = coefficients_to_omegas(
            artifact.psd_coefficients, artifact.component_index
        )
        if _relative_max(expected_psd_omegas, artifact.psd_omegas) > 1.0e-12:
            raise ValueError("Contextual fit V1 PSD Omega packing mismatch.")
        expected_minima = np.asarray(
            [np.min(np.linalg.eigvalsh(omega)) for omega in artifact.psd_omegas]
        )
        if _relative_max(expected_minima, artifact.psd_minimum_eigenvalues) > 1.0e-12:
            raise ValueError("Contextual fit V1 PSD eigenvalue diagnostics mismatch.")
        if artifact.context_grid is not None:
            expected_psd_surfaces, expected_psd_combined = _surface_panels(
                artifact.psd_coefficients,
                artifact.component_index,
                artifact.context_grid,
            )
            if (
                _relative_max(
                    expected_psd_surfaces[0], artifact.psd_covariance_surfaces
                )
                > 1.0e-12
                or _relative_max(
                    expected_psd_combined[0],
                    artifact.psd_combined_covariance_surface,
                )
                > 1.0e-12
            ):
                raise ValueError("Contextual fit V1 PSD covariance surfaces mismatch.")
    canonical_sha256(manifest)


def write_contextual_fit_v1(
    artifact: ContextualFitArtifactV1, output: str | Path
) -> Path:
    """Atomically write one strict immutable contextual-fit V1 container."""
    if not isinstance(artifact, ContextualFitArtifactV1):
        raise ValueError("Only ContextualFitArtifactV1 can use the V1 writer.")
    artifact.verify()
    array_names = artifact.manifest["arrays"]
    manifest_json, manifest_sha256, arrays = _preflight_stable_npz_members(
        manifest_json=canonical_json(artifact.manifest),
        manifest_sha256=artifact.manifest_sha256,
        arrays={name: getattr(artifact, name) for name in array_names},
        family="Contextual fit V1",
    )
    path = Path(output)
    if path.suffix != ".npz" or not path.name.endswith(CONTEXTUAL_FIT_V1_SUFFIX):
        path = Path(str(path) + CONTEXTUAL_FIT_V1_SUFFIX)
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
            family="Contextual fit V1",
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


def load_contextual_fit_v1(path: str | Path) -> ContextualFitArtifactV1:
    """Load only the stable fit V1 family, validating metadata before arrays."""
    source = Path(path)
    if not source.name.endswith(CONTEXTUAL_FIT_V1_SUFFIX):
        raise ValueError("Contextual fit V1 loader requires the exact V1 suffix.")
    try:
        with StableNpzReader(
            source,
            family="Contextual fit V1",
            maximum_members=len(_ALL_ARRAY_NAMES) + 2,
        ) as archive:
            manifest = _strict_json_loads(
                archive.read_text_scalar(
                    "manifest_json", maximum_bytes=16 * 1024 * 1024
                )
            )
            digest = archive.read_text_scalar("manifest_sha256", maximum_bytes=1024)
            if manifest.get("magic") != CONTEXTUAL_FIT_V1_MAGIC:
                raise ValueError("Contextual fit V1 magic mismatch.")
            if _sha256("manifest_sha256", digest) != canonical_sha256(manifest):
                raise ValueError("Contextual fit V1 manifest SHA-256 mismatch.")
            (
                components,
                scale_plan,
                groups,
                trait_ids,
                residual_names,
                selected_id,
                selected_index,
            ) = _validate_manifest_before_arrays(manifest)
            expected_shapes = _expected_array_shapes(
                manifest["dimensions"], set(manifest["arrays"])
            )
            archive.preflight_arrays(
                {
                    name: (np.dtype("<f8"), shape)
                    for name, shape in expected_shapes.items()
                }
            )
            arrays = {name: archive.load_array(name) for name in expected_shapes}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Contextual fit V1"):
            raise
        raise ValueError("Not a valid contextual fit V1 artifact.") from exc
    return ContextualFitArtifactV1(
        manifest=manifest,
        component_index=components,
        scale_plan=scale_plan,
        group_ids=groups,
        trait_ids=trait_ids,
        residual_names=residual_names,
        selected_trait_id=selected_id,
        selected_trait_index=selected_index,
        **{name: arrays.get(name) for name in _ALL_ARRAY_NAMES},
    )


__all__ = [
    "CONTEXTUAL_FIT_V1_MAGIC",
    "CONTEXTUAL_FIT_V1_SUFFIX",
    "ContextualFitArtifactV1",
    "assemble_contextual_normal_equations_v1",
    "fit_contextual_model_v1",
    "load_contextual_fit_v1",
    "validate_contextual_fit_compatibility_v1",
    "write_contextual_fit_v1",
]
