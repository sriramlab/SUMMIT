"""Narrow adapter from variant-LD-score references to contextual trait/fit V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from summit.context.fit import (
    ContextJackknifeError,
    ContextNormalEquations,
    ContextRankError,
    solve_context_normal_equations,
)
from summit.context.fit_v1 import (
    DEFAULT_FIT_V1_RTOL,
    assemble_contextual_normal_equations_from_moments_v1,
)
from summit.context.oracle import coefficients_to_omegas
from summit.context.spec import (
    ContextComponentIndex,
    freeze_context_mapping,
    owned_readonly_array,
)
from summit.context.trait_v1 import ContextualTraitMomentsV1
from summit.ldscore.generalized_gxe_reference_v1 import (
    GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
    GeneralizedGxEVariantReferenceArtifactV1,
    _reference_moments_after_deleting_variant_blocks_prevalidated_v1,
)
from summit.ldscore.generalized_gxe_variant import (
    GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
    GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
)
from summit.ldscore.generalized_gxe_trait_summary import (
    GeneralizedGxETraitSummary,
)


GENERALIZED_GXE_VARIANT_FIT_KIND = "summit.generalized_gxe.variant_ldscore_fit"


def _select_trait(
    trait: GeneralizedGxETraitSummary,
    selector: str | int | None,
) -> tuple[int, str]:
    if selector is None:
        if trait.n_traits != 1:
            raise ValueError("trait_selector is required for a multi-trait artifact")
        return 0, trait.trait_ids[0]
    index = trait.trait_index(selector)
    return index, trait.trait_ids[index]


def validate_generalized_gxe_trait_compatibility_v1(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
) -> Mapping[str, Any]:
    """Check concrete axes and dimensions without cryptographic identities."""
    if not isinstance(reference, GeneralizedGxEVariantReferenceArtifactV1):
        raise ValueError("reference must be a generalized variant-LD-score artifact")
    if not isinstance(trait, GeneralizedGxETraitSummary):
        raise ValueError("trait must be a generalized per-variant trait summary")
    mismatches: list[str] = []
    if reference.component_index.entries != trait.component_index.entries:
        mismatches.append("component_order")
    if reference.block_labels != trait.group_ids:
        mismatches.append("group_ids")
    if reference.n_variants != trait.n_variants:
        mismatches.append("variant_count")
    if tuple(reference.manifest["axes"]["residual_components"]["names"]) != trait.residual_names:
        mismatches.append("residual_component_names")
    if not np.array_equal(reference.annotation_masses, trait.annotation_masses):
        mismatches.append("annotation_masses")
    if not np.array_equal(
        reference.block_annotation_mass, trait.group_annotation_masses
    ):
        mismatches.append("group_annotation_masses")
    if not np.array_equal(reference.group_variant_counts, trait.group_variant_counts):
        mismatches.append("group_variant_counts")
    if mismatches:
        raise ValueError(
            "Generalized variant-LD-score reference/trait compatibility mismatch: "
            + ", ".join(sorted(set(mismatches)))
        )
    return freeze_context_mapping(
        {
            "reference_kind": GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
            "variant_count": reference.n_variants,
            "basis_count": reference.component_index.pair_index.num_basis,
            "annotation_count": len(reference.component_index.annotation_names),
            "component_count": len(reference.component_index),
            "block_count": len(reference.block_labels),
            "group_sequence": list(reference.block_labels),
            "compatibility_policy": "direct_axes_dimensions_and_masses_v1",
        }
    )


def _trait_moments_after_deleting_blocks(
    trait: GeneralizedGxETraitSummary,
    blocks: Sequence[str],
) -> ContextualTraitMomentsV1:
    requested = tuple(blocks)
    if len(set(requested)) != len(requested):
        raise ValueError("deleted variant blocks must be unique")
    unknown = set(requested) - set(trait.group_ids)
    if unknown:
        raise ValueError(f"unknown variant blocks: {sorted(unknown)}")
    if not requested:
        return trait.full_moments
    deleted_set = set(requested)
    deleted = np.asarray(
        [label in deleted_set for label in trait.group_ids], dtype=bool
    )
    retained_masses = trait.annotation_masses - np.sum(
        trait.group_annotation_masses[deleted], axis=0, dtype=np.float64
    )
    if np.any(retained_masses <= 0.0):
        raise ValueError("variant-block deletion empties an annotation")
    component_annotations = np.fromiter(
        (entry.annotation_index for entry in trait.component_index.entries),
        dtype=np.int64,
        count=len(trait.component_index),
    )
    component_masses = retained_masses[component_annotations]
    return ContextualTraitMomentsV1(
        annotation_masses=retained_masses,
        genetic_rhs=np.sum(
            trait.group_rhs_unnormalized_num[~deleted], axis=0, dtype=np.float64
        )
        / component_masses[:, None],
        genetic_traces=np.sum(
            trait.group_trace_unnormalized_num[~deleted], axis=0, dtype=np.float64
        )
        / component_masses,
        genetic_residual=np.sum(
            trait.group_genetic_residual_num[~deleted], axis=0, dtype=np.float64
        )
        / component_masses[:, None],
        residual_rhs=trait.residual_rhs,
        residual_traces=trait.residual_traces,
        residual_gram=trait.residual_gram,
    )


def assemble_generalized_gxe_normal_equations_v1(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
    *,
    trait_selector: str | int | None = None,
    deleted_blocks: Sequence[str] = (),
) -> ContextNormalEquations:
    """Use the shared fit assembly from compact reference and trait moments."""
    validate_generalized_gxe_trait_compatibility_v1(reference, trait)
    return _assemble_generalized_gxe_normal_equations_prevalidated_v1(
        reference,
        trait,
        trait_selector=trait_selector,
        deleted_blocks=deleted_blocks,
    )


def _assemble_generalized_gxe_normal_equations_prevalidated_v1(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
    *,
    trait_selector: str | int | None,
    deleted_blocks: Sequence[str],
) -> ContextNormalEquations:
    """Assemble after one enclosing compatibility validation."""
    trait_index, _ = _select_trait(trait, trait_selector)
    if isinstance(deleted_blocks, (str, bytes)):
        raise ValueError("deleted variant blocks must be a sequence")
    requested = tuple(deleted_blocks)
    reference_moments = (
        _reference_moments_after_deleting_variant_blocks_prevalidated_v1(
            reference, requested
        )
    )
    trait_moments = _trait_moments_after_deleting_blocks(trait, requested)
    return assemble_contextual_normal_equations_from_moments_v1(
        component_index=reference.component_index,
        reference_n=reference.n_samples,
        trait=trait,
        trait_index=trait_index,
        deleted_groups=requested,
        reference_moments=reference_moments,
        trait_moments=trait_moments,
    )


def assemble_generalized_gxe_normal_equation_batch_v1(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
    *,
    trait_selector: str | int | None = None,
    deleted_block_sets: Sequence[Sequence[str]],
) -> tuple[ContextNormalEquations, ...]:
    """Validate once and assemble several full/deleted summary systems."""
    validate_generalized_gxe_trait_compatibility_v1(reference, trait)
    if isinstance(deleted_block_sets, (str, bytes)):
        raise ValueError("deleted_block_sets must be a sequence of sequences")
    return tuple(
        _assemble_generalized_gxe_normal_equations_prevalidated_v1(
            reference,
            trait,
            trait_selector=trait_selector,
            deleted_blocks=deleted,
        )
        for deleted in deleted_block_sets
    )


@dataclass(frozen=True)
class GeneralizedGxEVariantFitResultV1:
    manifest: Mapping[str, Any]
    component_index: ContextComponentIndex
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

    def __post_init__(self) -> None:
        if not isinstance(self.component_index, ContextComponentIndex):
            raise ValueError("component_index must be a ContextComponentIndex")
        for name in (
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
        ):
            object.__setattr__(
                self, name, owned_readonly_array(getattr(self, name))
            )
        object.__setattr__(
            self, "manifest", freeze_context_mapping(dict(self.manifest))
        )

    @property
    def raw_rank(self) -> int:
        return int(self.manifest["solve"]["rank"])


def fit_generalized_gxe_variant_model_v1(
    reference: GeneralizedGxEVariantReferenceArtifactV1,
    trait: GeneralizedGxETraitSummary,
    *,
    trait_selector: str | int | None = None,
    rtol: float | None = None,
) -> GeneralizedGxEVariantFitResultV1:
    """Run the existing raw symmetric solve and compact every-block jackknife."""
    compatibility = validate_generalized_gxe_trait_compatibility_v1(
        reference, trait
    )
    trait_index, trait_id = _select_trait(trait, trait_selector)
    if isinstance(rtol, bool):
        raise ValueError("rtol must be finite and positive")
    relative_tolerance = DEFAULT_FIT_V1_RTOL if rtol is None else float(rtol)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("rtol must be finite and positive")
    groups = reference.block_labels
    if len(groups) < 2:
        raise ValueError("every-block jackknife requires at least two blocks")
    counts = reference.group_variant_counts
    if np.any(counts <= 0) or int(np.max(counts) - np.min(counts)) > 1:
        raise ValueError(
            "every-block equal-weight jackknife requires balanced nonempty blocks"
        )
    assembled = assemble_generalized_gxe_normal_equation_batch_v1(
        reference,
        trait,
        trait_selector=trait_index,
        deleted_block_sets=((), *((group,) for group in groups)),
    )
    equations = assembled[0]
    solve = solve_context_normal_equations(
        equations, rtol=relative_tolerance, require_full_rank=True
    )
    c = len(reference.component_index)
    h = len(trait.residual_names)
    raw = np.asarray(solve.coefficients, dtype=np.float64)
    loo = np.empty((len(groups), c + h), dtype=np.float64)
    for index, (group, deleted) in enumerate(
        zip(groups, assembled[1:], strict=True)
    ):
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
    covariance_scale = max(
        float(np.max(np.abs(covariance), initial=0.0)), 1.0
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -1.0e-10 * covariance_scale:
        raise RuntimeError("raw jackknife covariance is materially indefinite")
    diagonal = np.diag(covariance)
    if np.min(diagonal) < -1.0e-10 * covariance_scale:
        raise RuntimeError("raw jackknife covariance has negative variance")
    standard_errors = np.sqrt(np.maximum(diagonal, 0.0))
    manifest = {
        "kind": GENERALIZED_GXE_VARIANT_FIT_KIND,
        "schema_version": 1,
        "scientific_contract": GENERALIZED_GXE_VARIANT_SCIENTIFIC_CONTRACT,
        "source_reference_kind": GENERALIZED_GXE_VARIANT_REFERENCE_KIND,
        "selected_trait_id": trait_id,
        "selected_trait_index": trait_index,
        "jackknife_method": GENERALIZED_GXE_VARIANT_JACKKNIFE_METHOD,
        "same_person_jackknife": "reuse_full_same_person_v1",
        "summary_only_fit": True,
        "per_variant_panel_accessed": False,
        "compatibility": dict(compatibility),
        "solve": {
            "policy": "raw_rank_checked_symmetric_v1",
            "require_full_rank": True,
            "rtol": relative_tolerance,
            "rank": solve.rank,
            "condition_number": solve.condition_number,
            "relative_residual": solve.relative_residual,
            "ridge": False,
            "pseudoinverse": False,
            "coefficient_clipping": False,
            "psd_replacement": False,
        },
    }
    return GeneralizedGxEVariantFitResultV1(
        manifest=manifest,
        component_index=reference.component_index,
        selected_trait_id=trait_id,
        selected_trait_index=trait_index,
        normal_matrix=equations.matrix,
        normal_rhs=equations.rhs,
        traces=equations.traces,
        annotation_masses=equations.annotation_masses,
        reference_genetic_gram=equations.reference_genetic_gram,
        transferred_genetic_gram=equations.transferred_genetic_gram,
        raw_coefficients=raw,
        raw_genetic_coefficients=raw[:c],
        raw_residual_coefficients=raw[c:],
        raw_omegas=coefficients_to_omegas(raw[:c], reference.component_index),
        raw_loo_coefficients=loo,
        raw_jackknife_covariance=covariance,
        raw_standard_errors=standard_errors,
        raw_solve_residual=solve.solve_residual,
        raw_solve_eigenvalues=solve.diagnostics.eigenvalues,
        raw_solve_singular_values=solve.diagnostics.singular_values,
        raw_solve_retained_directions=solve.retained_directions,
        raw_solve_null_space=solve.null_space,
    )
