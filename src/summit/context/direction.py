"""Experimental summary-level learning of one context direction.

The module deliberately implements a narrow reduced model with kernels
``[G, GxE(omega), N, NxE(omega)]``.  Additive--interaction covariance is not
part of this Stage-07B prototype.  Individual-level arrays are accepted only
by the contraction builders; evaluation and optimization consume compact
reference and trait tensors whose dimensions depend on the number of context
coordinates, not on the sample or variant count.

The implementation is a correctness-first dense Python oracle.  Exact
variant-fold objects are rebuilt from disjoint variant subsets.  They are not
obtained through the small-deletion/full-D jackknife approximation used by the
generic Stage-04 fitter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
from scipy.optimize import minimize

from .fit import (
    DEFAULT_SOLVE_RTOL,
    ContextNormalEquations,
    ContextRankError,
    ContextSolveResult,
    solve_context_normal_equations,
)
from .oracle import (
    common_scale_features,
    dense_genetic_kernels,
    dense_residual_kernels,
    exact_same_person_matrix,
    kernel_gram,
    kernel_rhs,
    kernel_traces,
    transfer_reference_gram,
)
from .spec import (
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_json,
    canonical_sha256,
)


DIRECTION_REFERENCE_KIND = "summit.context.direction_reference"
DIRECTION_TRAIT_KIND = "summit.context.direction_trait"
DIRECTION_CROSSFIT_KIND = "summit.context.direction_crossfit"
DIRECTION_SCHEMA_VERSION = 1
DIRECTION_ESTIMAND = "reduced_G_GxE_N_NxE_no_additive_interaction_covariance"
DIRECTION_OBJECTIVES = (
    "interaction_coefficient",
    "interaction_trace_contribution",
    "he_moment_gain",
)
DEFAULT_MAX_ENVIRONMENTS = 3
DEFAULT_MAX_CONTEXT_METRIC_CONDITION = 1.0e10


def _finite(name: str, value: object, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _name(value: Any, label: str) -> str:
    text = str(value)
    if not text or any(character.isspace() for character in text):
        raise ValueError(f"{label} must be a nonempty name without whitespace.")
    return text


def _identity_digest(value: str, label: str) -> str:
    text = _name(value, label)
    if len(text) == 64:
        try:
            int(text, 16)
        except ValueError:
            pass
        else:
            return text.lower()
    return canonical_sha256({"declared_identity": text})


def _environment_names(
    names: Sequence[str] | None, n_environments: int
) -> tuple[str, ...]:
    if names is None:
        result = tuple(f"z{index}" for index in range(n_environments))
    else:
        result = tuple(_name(value, "environment name") for value in names)
    if len(result) != n_environments:
        raise ValueError("Environment names do not match the context dimension.")
    if len(set(result)) != len(result):
        raise ValueError("Environment names must be unique.")
    return result


def _metric_diagnostics(metric: np.ndarray) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(metric)
    leading = float(np.max(np.abs(eigenvalues), initial=0.0))
    tolerance = 1.0e-12 * leading
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.size)
    condition = (
        float(np.max(positive) / np.min(positive)) if positive.size else float("inf")
    )
    return {
        "eigenvalues": [float(value) for value in eigenvalues],
        "rank": rank,
        "tolerance": float(tolerance),
        "condition_number": condition,
        "maximum_condition_number": DEFAULT_MAX_CONTEXT_METRIC_CONDITION,
    }


def _validate_metric(metric: object, n_environments: int) -> np.ndarray:
    array = _finite("context_metric", metric, ndim=2)
    if array.shape != (n_environments, n_environments):
        raise ValueError("Context metric shape does not match the environment count.")
    scale = float(np.max(np.abs(array), initial=0.0))
    asymmetry = float(np.max(np.abs(array - array.T), initial=0.0))
    if asymmetry > 1.0e-12 * max(scale, np.finfo(np.float64).tiny):
        raise ValueError("Context metric is materially asymmetric.")
    symmetric = 0.5 * (array + array.T)
    diagnostics = _metric_diagnostics(symmetric)
    if diagnostics["rank"] != n_environments:
        raise ValueError(
            "Context metric must be positive definite; explicitly prune a "
            "singular context basis before direction learning."
        )
    if diagnostics["condition_number"] > DEFAULT_MAX_CONTEXT_METRIC_CONDITION:
        raise ValueError(
            "Context metric is too ill-conditioned for direction whitening: "
            f"condition {diagnostics['condition_number']:.6g} exceeds the declared "
            f"limit {DEFAULT_MAX_CONTEXT_METRIC_CONDITION:.6g}."
        )
    return np.ascontiguousarray(symmetric)


def _validate_projector(
    projector: object, n_samples: int, label: str
) -> tuple[np.ndarray, int]:
    array = _finite(label, getattr(projector, "projector", projector), ndim=2)
    if array.shape != (n_samples, n_samples):
        raise ValueError(f"{label} shape does not match the sample count.")
    scale = float(np.max(np.abs(array), initial=0.0))
    tolerance = 1.0e-10 * max(scale, np.finfo(np.float64).tiny)
    if np.max(np.abs(array - array.T), initial=0.0) > tolerance:
        raise ValueError(f"{label} is not symmetric.")
    symmetric = 0.5 * (array + array.T)
    if np.max(np.abs(symmetric @ symmetric - symmetric), initial=0.0) > 5.0e-10 * max(
        scale, np.finfo(np.float64).tiny
    ):
        raise ValueError(f"{label} is not idempotent.")
    residual_rank_float = float(np.trace(symmetric))
    residual_rank = int(round(residual_rank_float))
    if (
        residual_rank < 1
        or abs(residual_rank_float - residual_rank) > 1.0e-8 * n_samples
    ):
        raise ValueError(f"{label} has an invalid residual rank.")
    return np.ascontiguousarray(symmetric), residual_rank


def _project_normalize(
    phenotype: object, projector: np.ndarray, residual_rank: int
) -> np.ndarray:
    raw = _finite("phenotype", phenotype, ndim=1)
    if raw.size != projector.shape[0]:
        raise ValueError("Phenotype and study projector use different samples.")
    projected = projector @ raw
    sum_squares = float(projected @ projected)
    minimum = np.finfo(np.float64).eps * float(raw @ raw)
    if not np.isfinite(sum_squares) or sum_squares <= minimum:
        raise ValueError("Phenotype has zero or invalid projected variance.")
    return np.asarray(
        projected * np.sqrt(residual_rank / sum_squares), dtype=np.float64
    )


def _variant_selection(
    n_variants: int,
    variant_weights: object | None,
    variant_mask: object | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if variant_weights is None:
        weights = np.ones(n_variants, dtype=np.float64)
    else:
        weights = _finite("variant_weights", variant_weights, ndim=1)
        if weights.shape != (n_variants,) or np.any(weights < 0.0):
            raise ValueError(
                "variant_weights must be a finite non-negative vector with one "
                "entry per variant."
            )
    if variant_mask is None:
        mask = np.ones(n_variants, dtype=np.bool_)
    else:
        raw_mask = np.asarray(variant_mask)
        if raw_mask.shape != (n_variants,) or raw_mask.dtype != np.bool_:
            raise ValueError(
                "variant_mask must be a Boolean vector with one entry per variant."
            )
        mask = np.ascontiguousarray(raw_mask, dtype=np.bool_)
    selected = np.flatnonzero(mask)
    if selected.size < 1:
        raise ValueError("At least one variant must be selected.")
    selected_weights = np.ascontiguousarray(weights[selected], dtype=np.float64)
    mass = float(np.sum(selected_weights, dtype=np.float64))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Selected variants must have positive total weight.")
    return selected, selected_weights, mass


def _selection_hash(variant_hash: str, n_variants: int, selected: np.ndarray) -> str:
    indicator = np.zeros(n_variants, dtype=np.uint8)
    indicator[selected] = 1
    return canonical_sha256(
        {
            "ordered_variant_hash": variant_hash,
            "n_variants": n_variants,
            "selection_digest": array_sha256(indicator),
            "selected_count": int(selected.size),
        }
    )


def _pair_product_basis(
    context: np.ndarray, pair_index: ContextPairIndex
) -> np.ndarray:
    result = np.empty((context.shape[0], len(pair_index)), dtype=np.float64)
    for pair in pair_index.entries:
        result[:, pair.index] = (
            pair.kernel_factor * context[:, pair.q] * context[:, pair.r]
        )
    return result


def _genetic_base_kernels(
    genotype: np.ndarray,
    context: np.ndarray,
    projector: np.ndarray,
    weights: np.ndarray,
    pair_index: ContextPairIndex,
) -> tuple[np.ndarray, np.ndarray]:
    features_zero = projector @ genotype
    mass = float(np.sum(weights, dtype=np.float64))
    additive = (features_zero * weights[None, :]) @ features_zero.T / mass
    features = common_scale_features(genotype, context, projector)
    components = ContextComponentIndex(("all",), pair_index)
    interactions = dense_genetic_kernels(features, weights[:, None], components)
    kernels = np.concatenate([additive[None, :, :], interactions], axis=0)
    return np.ascontiguousarray(kernels), features


def _shared_specification(
    *,
    environment_names: tuple[str, ...],
    pair_index: ContextPairIndex,
    context_metric: np.ndarray,
    context_metric_source: str,
    context_spec_hash: str,
    variant_hash: str,
    selected_variant_hash: str,
    weight_hash: str,
    genotype_scaling: str,
) -> dict[str, Any]:
    return {
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "estimand": DIRECTION_ESTIMAND,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "environment_names": list(environment_names),
        "pair_index": pair_index.to_dict(),
        "pair_index_hash": pair_index.digest,
        "context_metric_hash": array_sha256(context_metric),
        "context_metric_source": context_metric_source,
        "context_metric_diagnostics": _metric_diagnostics(context_metric),
        "context_spec_hash": context_spec_hash,
        "ordered_variant_hash": variant_hash,
        "selected_variant_hash": selected_variant_hash,
        "selected_weight_hash": weight_hash,
        "genotype_scaling": genotype_scaling,
    }


@dataclass(frozen=True)
class DirectionReferenceContractions:
    """Compact reference moments for ``[G, I_pair_1, ...]``."""

    manifest: dict[str, Any]
    environment_names: tuple[str, ...]
    pair_index: ContextPairIndex
    context_metric: np.ndarray
    base_gram: np.ndarray
    same_person: np.ndarray
    base_traces: np.ndarray
    n_samples: int
    n_variants: int
    annotation_mass: float
    selected_variant_hash: str
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int

    @property
    def compact_dimension(self) -> int:
        return 1 + len(self.pair_index)

    @property
    def genetic_gram(self) -> np.ndarray:
        return self.base_gram

    @property
    def reference_same_person(self) -> np.ndarray:
        return self.same_person


@dataclass(frozen=True)
class DirectionTraitContractions:
    """Compact study moments for ``[G, I_pairs, N, D_pairs]``."""

    manifest: dict[str, Any]
    environment_names: tuple[str, ...]
    pair_index: ContextPairIndex
    context_metric: np.ndarray
    base_gram: np.ndarray
    base_rhs: np.ndarray
    base_traces: np.ndarray
    base_component_names: tuple[str, ...]
    n_samples: int
    residual_rank: int
    n_variants: int
    annotation_mass: float
    selected_variant_hash: str
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int

    @property
    def compact_dimension(self) -> int:
        return 2 + 2 * len(self.pair_index)

    @property
    def trait_gram(self) -> np.ndarray:
        return self.base_gram

    @property
    def rhs(self) -> np.ndarray:
        return self.base_rhs

    @property
    def traces(self) -> np.ndarray:
        return self.base_traces


@dataclass(frozen=True)
class ContextDirectionContractions:
    reference: DirectionReferenceContractions
    summary: DirectionTraitContractions
    manifest: dict[str, Any]

    @property
    def environment_names(self) -> tuple[str, ...]:
        return self.reference.environment_names

    @property
    def pair_index(self) -> ContextPairIndex:
        return self.reference.pair_index

    @property
    def context_metric(self) -> np.ndarray:
        return self.reference.context_metric


@dataclass(frozen=True)
class DirectionEvaluation:
    direction: np.ndarray
    pair_weights: np.ndarray
    omega_outer: np.ndarray
    reduction_matrix: np.ndarray
    equations: ContextNormalEquations
    solve: ContextSolveResult
    matrix: np.ndarray
    rhs: np.ndarray
    traces: np.ndarray
    coefficients: np.ndarray
    trace_contributions: np.ndarray
    objective_name: str
    objective_value: float
    objective_values: dict[str, float]
    full_moment_fit: float
    nuisance_moment_fit: float
    rank: int
    condition_number: float
    relative_residual: float
    status: str

    @property
    def objective(self) -> str:
        return self.objective_name


@dataclass(frozen=True)
class DirectionOptimizationResult:
    direction: np.ndarray
    evaluation: DirectionEvaluation
    objective_name: str
    objective_value: float
    start_directions: np.ndarray
    candidate_objectives: np.ndarray
    grid_best_direction: np.ndarray | None
    grid_best_objective: float | None
    grid_validation_gap: float | None
    distinct_candidate_directions: np.ndarray
    distinct_candidate_objectives: np.ndarray
    near_optimal_directions: np.ndarray
    second_distinct_objective: float | None
    objective_gap: float | None
    nonunique: bool
    evaluations: int
    converged_candidates: int
    failed_candidates: int
    optimizer_messages: tuple[str, ...]
    status: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DirectionFoldAssignment:
    fold_labels: tuple[str, str]
    block_identities: tuple[str, ...]
    block_folds: tuple[int, ...]
    fold_block_identities: tuple[tuple[str, ...], tuple[str, ...]]
    fold_variant_counts: tuple[int, int]
    fold_weight_masses: tuple[float, float]
    fold_selected_variant_hashes: tuple[str, str]
    fold_selected_weight_hashes: tuple[str, str]
    variant_order_hash: str
    block_sequence_hash: str
    assignment_hash: str

    def fold_mask(self, block_ids: object, fold: int) -> np.ndarray:
        if fold not in (0, 1):
            raise ValueError("fold must be 0 or 1.")
        identities = _block_identities(block_ids)
        mapping = dict(zip(self.block_identities, self.block_folds, strict=True))
        if set(identities) != set(mapping):
            raise ValueError("Block identities do not match the frozen assignment.")
        return np.asarray(
            [mapping[value] == fold for value in identities], dtype=np.bool_
        )


@dataclass(frozen=True)
class DirectionCrossFitArmContractions:
    fold_id: str
    train_fold: str
    heldout_fold: str
    train_variant_hash: str
    heldout_variant_hash: str
    train: ContextDirectionContractions
    heldout: ContextDirectionContractions


@dataclass(frozen=True)
class DirectionCrossFitContractions:
    assignment: DirectionFoldAssignment
    arms: tuple[DirectionCrossFitArmContractions, DirectionCrossFitArmContractions]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DirectionCrossFitFoldResult:
    fold_id: str
    train_variant_hash: str
    heldout_variant_hash: str
    training_optimization: DirectionOptimizationResult
    heldout_evaluation: DirectionEvaluation
    training_objective: float
    heldout_objective: float

    @property
    def direction(self) -> np.ndarray:
        return self.training_optimization.direction


@dataclass(frozen=True)
class DirectionCrossFitResult:
    folds: tuple[DirectionCrossFitFoldResult, DirectionCrossFitFoldResult]
    objective_name: str
    combined_heldout_value: float
    fold_direction_alignment: float
    status: str
    manifest: dict[str, Any]


def direction_pair_weights(
    direction: object, pair_index: ContextPairIndex | None = None
) -> np.ndarray:
    """Pack ``omega omega'`` without an extra off-diagonal factor."""
    omega = _finite("direction", direction, ndim=1)
    index = ContextPairIndex(omega.size) if pair_index is None else pair_index
    if index.num_basis != omega.size:
        raise ValueError("Pair index and direction dimensions differ.")
    return np.asarray(
        [omega[pair.q] * omega[pair.r] for pair in index.entries],
        dtype=np.float64,
    )


def normalize_context_direction(
    direction: object, context_metric: object, *, canonical_sign: bool = True
) -> np.ndarray:
    """Metric-normalize a direction and apply a deterministic display sign."""
    omega = _finite("direction", direction, ndim=1)
    metric = _validate_metric(context_metric, omega.size)
    norm_squared = float(omega @ metric @ omega)
    scale = float(np.linalg.norm(omega))
    if (
        not np.isfinite(norm_squared)
        or norm_squared <= np.finfo(np.float64).eps * scale * scale
    ):
        raise ValueError("Direction has zero or invalid context-metric norm.")
    result = np.asarray(omega / np.sqrt(norm_squared), dtype=np.float64)
    if canonical_sign:
        absolute = np.abs(result)
        pivot = int(np.flatnonzero(absolute == np.max(absolute))[0])
        if result[pivot] < 0.0:
            result = -result
    return result


def direction_reduction_matrix(
    direction: object, pair_index: ContextPairIndex
) -> np.ndarray:
    """Map base study components into ``[G,I,N,D]``."""
    omega = _finite("direction", direction, ndim=1)
    if omega.size != pair_index.num_basis:
        raise ValueError("Direction dimension does not match pair index.")
    c = direction_pair_weights(omega, pair_index)
    p_count = len(pair_index)
    reduction = np.zeros((2 + 2 * p_count, 4), dtype=np.float64)
    reduction[0, 0] = 1.0
    reduction[1 : 1 + p_count, 1] = c
    reduction[1 + p_count, 2] = 1.0
    reduction[2 + p_count :, 3] = c
    return reduction


def _base_names(
    environment_names: tuple[str, ...], pair_index: ContextPairIndex
) -> tuple[str, ...]:
    pair_names = tuple(
        f"{environment_names[pair.q]},{environment_names[pair.r]}"
        for pair in pair_index.entries
    )
    return (
        ("G",)
        + tuple(f"I:{name}" for name in pair_names)
        + ("N",)
        + tuple(f"D:{name}" for name in pair_names)
    )


def build_direction_reference_contractions(
    genotype: object,
    context: object,
    projector: object,
    *,
    context_metric: object,
    environment_names: Sequence[str] | None = None,
    variant_hash: str = "development_variant_order",
    context_spec_hash: str = "development_context_specification",
    context_metric_source: str = "reference_fixed_second_moment",
    genotype_scaling: str = "fixed_common_scale",
    fixed_effect_hash: str | None = None,
    variant_weights: object | None = None,
    variant_mask: object | None = None,
    fold_id: str = "all",
) -> DirectionReferenceContractions:
    """Build exact dense reference contractions for the Python prototype."""
    started = time.perf_counter()
    genotype_array = _finite("reference genotype", genotype, ndim=2)
    context_array = _finite("reference context", context, ndim=2)
    n_samples, n_variants = genotype_array.shape
    if context_array.shape[0] != n_samples:
        raise ValueError("Reference genotype and context sample counts differ.")
    n_environments = context_array.shape[1]
    if n_environments < 1 or n_environments > DEFAULT_MAX_ENVIRONMENTS:
        raise ValueError(
            f"Direction learning supports 1..{DEFAULT_MAX_ENVIRONMENTS} environments."
        )
    names = _environment_names(environment_names, n_environments)
    metric = _validate_metric(context_metric, n_environments)
    projector_array, _ = _validate_projector(
        projector, n_samples, "reference projector"
    )
    selected, selected_weights, mass = _variant_selection(
        n_variants, variant_weights, variant_mask
    )
    selected_genotype = np.ascontiguousarray(genotype_array[:, selected])
    pair_index = ContextPairIndex(n_environments)
    kernels, _ = _genetic_base_kernels(
        selected_genotype,
        context_array,
        projector_array,
        selected_weights,
        pair_index,
    )
    moments_started = time.perf_counter()
    gram = kernel_gram(kernels)
    same_person = exact_same_person_matrix(kernels)
    traces = kernel_traces(kernels)
    moments_elapsed = time.perf_counter() - moments_started
    variant_identity = _identity_digest(variant_hash, "variant_hash")
    context_identity = _identity_digest(context_spec_hash, "context_spec_hash")
    selected_hash = _selection_hash(variant_identity, n_variants, selected)
    weight_hash = array_sha256(selected_weights)
    fixed_hash = (
        array_sha256(projector_array)
        if fixed_effect_hash is None
        else _identity_digest(fixed_effect_hash, "fixed_effect_hash")
    )
    scaling = _name(genotype_scaling, "genotype_scaling")
    shared = _shared_specification(
        environment_names=names,
        pair_index=pair_index,
        context_metric=metric,
        context_metric_source=_name(context_metric_source, "context_metric_source"),
        context_spec_hash=context_identity,
        variant_hash=variant_identity,
        selected_variant_hash=selected_hash,
        weight_hash=weight_hash,
        genotype_scaling=scaling,
    )
    arrays = {
        "base_gram": array_sha256(gram),
        "same_person": array_sha256(same_person),
        "base_traces": array_sha256(traces),
        "context_metric": array_sha256(metric),
    }
    manifest = {
        "kind": DIRECTION_REFERENCE_KIND,
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "experimental": True,
        "shared_specification": shared,
        "shared_specification_hash": canonical_sha256(shared),
        "fixed_effect_hash": fixed_hash,
        "fold_id": _name(fold_id, "fold_id"),
        "dimensions": {
            "n_samples": n_samples,
            "n_variants_total": n_variants,
            "n_variants_selected": int(selected.size),
            "n_environments": n_environments,
            "n_pairs": len(pair_index),
            "compact_dimension": kernels.shape[0],
        },
        "annotation_mass": mass,
        "array_hashes": arrays,
        "backend": "python_dense_exact_development",
    }
    elapsed = time.perf_counter() - started
    return DirectionReferenceContractions(
        manifest=manifest,
        environment_names=names,
        pair_index=pair_index,
        context_metric=metric,
        base_gram=np.ascontiguousarray(gram),
        same_person=np.ascontiguousarray(same_person),
        base_traces=np.ascontiguousarray(traces),
        n_samples=n_samples,
        n_variants=int(selected.size),
        annotation_mass=mass,
        selected_variant_hash=selected_hash,
        phase_times_seconds={
            "build_total": elapsed,
            "small_moment_reduction": moments_elapsed,
        },
        peak_rss_bytes=int(psutil.Process().memory_info().rss),
    )


def build_direction_trait_contractions(
    genotype: object,
    context: object,
    projector: object,
    phenotype: object,
    *,
    context_metric: object,
    environment_names: Sequence[str] | None = None,
    variant_hash: str = "development_variant_order",
    context_spec_hash: str = "development_context_specification",
    context_metric_source: str = "reference_fixed_second_moment",
    genotype_scaling: str = "fixed_common_scale",
    fixed_effect_hash: str | None = None,
    variant_weights: object | None = None,
    variant_mask: object | None = None,
    fold_id: str = "all",
) -> DirectionTraitContractions:
    """Build exact dense trait contractions for the Python prototype."""
    started = time.perf_counter()
    genotype_array = _finite("study genotype", genotype, ndim=2)
    context_array = _finite("study context", context, ndim=2)
    n_samples, n_variants = genotype_array.shape
    if context_array.shape[0] != n_samples:
        raise ValueError("Study genotype and context sample counts differ.")
    n_environments = context_array.shape[1]
    if n_environments < 1 or n_environments > DEFAULT_MAX_ENVIRONMENTS:
        raise ValueError(
            f"Direction learning supports 1..{DEFAULT_MAX_ENVIRONMENTS} environments."
        )
    names = _environment_names(environment_names, n_environments)
    metric = _validate_metric(context_metric, n_environments)
    projector_array, residual_rank = _validate_projector(
        projector, n_samples, "study projector"
    )
    y = _project_normalize(phenotype, projector_array, residual_rank)
    selected, selected_weights, mass = _variant_selection(
        n_variants, variant_weights, variant_mask
    )
    selected_genotype = np.ascontiguousarray(genotype_array[:, selected])
    pair_index = ContextPairIndex(n_environments)
    genetic_kernels, _ = _genetic_base_kernels(
        selected_genotype,
        context_array,
        projector_array,
        selected_weights,
        pair_index,
    )
    residual_products = _pair_product_basis(context_array, pair_index)
    direction_residual_kernels = dense_residual_kernels(
        projector_array, residual_products
    )
    kernels = np.concatenate(
        [
            genetic_kernels,
            projector_array[None, :, :],
            direction_residual_kernels,
        ],
        axis=0,
    )
    moments_started = time.perf_counter()
    gram = kernel_gram(kernels)
    rhs = kernel_rhs(kernels, y)
    traces = kernel_traces(kernels)
    moments_elapsed = time.perf_counter() - moments_started
    base_names = _base_names(names, pair_index)
    variant_identity = _identity_digest(variant_hash, "variant_hash")
    context_identity = _identity_digest(context_spec_hash, "context_spec_hash")
    selected_hash = _selection_hash(variant_identity, n_variants, selected)
    weight_hash = array_sha256(selected_weights)
    fixed_hash = (
        array_sha256(projector_array)
        if fixed_effect_hash is None
        else _identity_digest(fixed_effect_hash, "fixed_effect_hash")
    )
    scaling = _name(genotype_scaling, "genotype_scaling")
    shared = _shared_specification(
        environment_names=names,
        pair_index=pair_index,
        context_metric=metric,
        context_metric_source=_name(context_metric_source, "context_metric_source"),
        context_spec_hash=context_identity,
        variant_hash=variant_identity,
        selected_variant_hash=selected_hash,
        weight_hash=weight_hash,
        genotype_scaling=scaling,
    )
    arrays = {
        "base_gram": array_sha256(gram),
        "base_rhs": array_sha256(rhs),
        "base_traces": array_sha256(traces),
        "context_metric": array_sha256(metric),
    }
    manifest = {
        "kind": DIRECTION_TRAIT_KIND,
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "experimental": True,
        "shared_specification": shared,
        "shared_specification_hash": canonical_sha256(shared),
        "fixed_effect_hash": fixed_hash,
        "fold_id": _name(fold_id, "fold_id"),
        "dimensions": {
            "n_samples": n_samples,
            "residual_rank": residual_rank,
            "n_variants_total": n_variants,
            "n_variants_selected": int(selected.size),
            "n_environments": n_environments,
            "n_pairs": len(pair_index),
            "compact_dimension": kernels.shape[0],
        },
        "annotation_mass": mass,
        "base_component_names": list(base_names),
        "array_hashes": arrays,
        "backend": "python_dense_exact_development",
        "phenotype_normalization": "projected_y_squared_norm_equals_residual_rank",
    }
    elapsed = time.perf_counter() - started
    return DirectionTraitContractions(
        manifest=manifest,
        environment_names=names,
        pair_index=pair_index,
        context_metric=metric,
        base_gram=np.ascontiguousarray(gram),
        base_rhs=np.ascontiguousarray(rhs),
        base_traces=np.ascontiguousarray(traces),
        base_component_names=base_names,
        n_samples=n_samples,
        residual_rank=residual_rank,
        n_variants=int(selected.size),
        annotation_mass=mass,
        selected_variant_hash=selected_hash,
        phase_times_seconds={
            "build_total": elapsed,
            "small_moment_reduction": moments_elapsed,
        },
        peak_rss_bytes=int(psutil.Process().memory_info().rss),
    )


def validate_direction_reference(
    reference: DirectionReferenceContractions,
) -> dict[str, Any]:
    if reference.manifest.get("kind") != DIRECTION_REFERENCE_KIND:
        raise ValueError("Invalid direction-reference manifest kind.")
    if reference.manifest.get("schema_version") != DIRECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported direction-reference schema version.")
    l_count = len(reference.environment_names)
    pair_index = ContextPairIndex(l_count)
    if reference.pair_index.digest != pair_index.digest:
        raise ValueError("Direction-reference pair index is not canonical.")
    metric = _validate_metric(reference.context_metric, l_count)
    p_base = 1 + len(pair_index)
    gram = _finite("reference base_gram", reference.base_gram, ndim=2)
    same = _finite("reference same_person", reference.same_person, ndim=2)
    traces = _finite("reference base_traces", reference.base_traces, ndim=1)
    if (
        gram.shape != (p_base, p_base)
        or same.shape != gram.shape
        or traces.shape != (p_base,)
    ):
        raise ValueError("Direction-reference compact tensor shapes are invalid.")
    hashes = reference.manifest.get("array_hashes", {})
    expected = {
        "base_gram": array_sha256(gram),
        "same_person": array_sha256(same),
        "base_traces": array_sha256(traces),
        "context_metric": array_sha256(metric),
    }
    if hashes != expected:
        raise ValueError("Direction-reference compact tensor hash mismatch.")
    shared = reference.manifest.get("shared_specification")
    if not isinstance(shared, Mapping):
        raise ValueError("Direction-reference shared specification is missing.")
    if reference.manifest.get("shared_specification_hash") != canonical_sha256(shared):
        raise ValueError("Direction-reference shared specification hash mismatch.")
    shared_expected = {
        "environment_names": list(reference.environment_names),
        "pair_index_hash": reference.pair_index.digest,
        "context_metric_hash": array_sha256(metric),
        "selected_variant_hash": reference.selected_variant_hash,
        "context_metric_diagnostics": _metric_diagnostics(metric),
    }
    if any(shared.get(field) != value for field, value in shared_expected.items()):
        raise ValueError("Direction-reference shared specification identity mismatch.")
    if not isinstance(shared.get("context_metric_source"), str):
        raise ValueError("Direction-reference context metric source is invalid.")
    _name(shared["context_metric_source"], "context_metric_source")
    dimensions = reference.manifest.get("dimensions", {})
    dimension_expected = {
        "n_samples": reference.n_samples,
        "n_variants_selected": reference.n_variants,
        "n_environments": l_count,
        "n_pairs": len(pair_index),
        "compact_dimension": p_base,
    }
    if any(
        dimensions.get(field) != value for field, value in dimension_expected.items()
    ):
        raise ValueError(
            "Direction-reference manifest dimensions do not match tensors."
        )
    if not np.isclose(
        float(reference.manifest.get("annotation_mass", np.nan)),
        reference.annotation_mass,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Direction-reference annotation mass identity mismatch.")
    scale = max(float(np.max(np.abs(gram), initial=0.0)), 1.0)
    if np.max(np.abs(gram - gram.T), initial=0.0) > 1.0e-11 * scale:
        raise ValueError("Direction-reference Gram matrix is materially asymmetric.")
    if np.max(np.abs(same - same.T), initial=0.0) > 1.0e-11 * scale:
        raise ValueError("Direction-reference same-person matrix is asymmetric.")
    if reference.selected_variant_hash != reference.manifest[
        "shared_specification"
    ].get("selected_variant_hash"):
        raise ValueError("Direction-reference selected-variant identity mismatch.")
    if (
        reference.n_samples < 2
        or reference.n_variants < 1
        or reference.annotation_mass <= 0.0
    ):
        raise ValueError("Direction-reference dimensions or mass are invalid.")
    return {"status": "valid_direction_reference", "compact_dimension": p_base}


def validate_direction_trait(summary: DirectionTraitContractions) -> dict[str, Any]:
    if summary.manifest.get("kind") != DIRECTION_TRAIT_KIND:
        raise ValueError("Invalid direction-trait manifest kind.")
    if summary.manifest.get("schema_version") != DIRECTION_SCHEMA_VERSION:
        raise ValueError("Unsupported direction-trait schema version.")
    l_count = len(summary.environment_names)
    pair_index = ContextPairIndex(l_count)
    if summary.pair_index.digest != pair_index.digest:
        raise ValueError("Direction-trait pair index is not canonical.")
    metric = _validate_metric(summary.context_metric, l_count)
    p_base = 2 + 2 * len(pair_index)
    gram = _finite("trait base_gram", summary.base_gram, ndim=2)
    rhs = _finite("trait base_rhs", summary.base_rhs, ndim=1)
    traces = _finite("trait base_traces", summary.base_traces, ndim=1)
    if (
        gram.shape != (p_base, p_base)
        or rhs.shape != (p_base,)
        or traces.shape != (p_base,)
    ):
        raise ValueError("Direction-trait compact tensor shapes are invalid.")
    if len(summary.base_component_names) != p_base:
        raise ValueError("Direction-trait component names have the wrong length.")
    hashes = summary.manifest.get("array_hashes", {})
    expected = {
        "base_gram": array_sha256(gram),
        "base_rhs": array_sha256(rhs),
        "base_traces": array_sha256(traces),
        "context_metric": array_sha256(metric),
    }
    if hashes != expected:
        raise ValueError("Direction-trait compact tensor hash mismatch.")
    shared = summary.manifest.get("shared_specification")
    if not isinstance(shared, Mapping):
        raise ValueError("Direction-trait shared specification is missing.")
    if summary.manifest.get("shared_specification_hash") != canonical_sha256(shared):
        raise ValueError("Direction-trait shared specification hash mismatch.")
    shared_expected = {
        "environment_names": list(summary.environment_names),
        "pair_index_hash": summary.pair_index.digest,
        "context_metric_hash": array_sha256(metric),
        "selected_variant_hash": summary.selected_variant_hash,
        "context_metric_diagnostics": _metric_diagnostics(metric),
    }
    if any(shared.get(field) != value for field, value in shared_expected.items()):
        raise ValueError("Direction-trait shared specification identity mismatch.")
    if not isinstance(shared.get("context_metric_source"), str):
        raise ValueError("Direction-trait context metric source is invalid.")
    _name(shared["context_metric_source"], "context_metric_source")
    dimensions = summary.manifest.get("dimensions", {})
    dimension_expected = {
        "n_samples": summary.n_samples,
        "residual_rank": summary.residual_rank,
        "n_variants_selected": summary.n_variants,
        "n_environments": l_count,
        "n_pairs": len(pair_index),
        "compact_dimension": p_base,
    }
    if any(
        dimensions.get(field) != value for field, value in dimension_expected.items()
    ):
        raise ValueError("Direction-trait manifest dimensions do not match tensors.")
    if not np.isclose(
        float(summary.manifest.get("annotation_mass", np.nan)),
        summary.annotation_mass,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Direction-trait annotation mass identity mismatch.")
    scale = max(float(np.max(np.abs(gram), initial=0.0)), 1.0)
    if np.max(np.abs(gram - gram.T), initial=0.0) > 1.0e-11 * scale:
        raise ValueError("Direction-trait Gram matrix is materially asymmetric.")
    if summary.selected_variant_hash != summary.manifest["shared_specification"].get(
        "selected_variant_hash"
    ):
        raise ValueError("Direction-trait selected-variant identity mismatch.")
    if (
        summary.n_samples < 1
        or summary.residual_rank < 1
        or summary.residual_rank > summary.n_samples
        or summary.n_variants < 1
        or summary.annotation_mass <= 0.0
    ):
        raise ValueError("Direction-trait dimensions or mass are invalid.")
    return {"status": "valid_direction_trait", "compact_dimension": p_base}


def combine_context_direction_contractions(
    reference: DirectionReferenceContractions,
    summary: DirectionTraitContractions,
) -> ContextDirectionContractions:
    """Validate and bind one compact reference/trait pair."""
    validate_direction_reference(reference)
    validate_direction_trait(summary)
    if reference.manifest.get("shared_specification_hash") != summary.manifest.get(
        "shared_specification_hash"
    ):
        raise ValueError(
            "Direction reference and trait specifications are incompatible."
        )
    if reference.environment_names != summary.environment_names:
        raise ValueError("Direction reference and trait environment order differs.")
    if reference.pair_index.digest != summary.pair_index.digest:
        raise ValueError("Direction reference and trait pair order differs.")
    if reference.selected_variant_hash != summary.selected_variant_hash:
        raise ValueError("Direction reference and trait variant subsets differ.")
    mass_scale = max(abs(reference.annotation_mass), abs(summary.annotation_mass), 1.0)
    if abs(reference.annotation_mass - summary.annotation_mass) > 1.0e-12 * mass_scale:
        raise ValueError("Direction reference and trait annotation masses differ.")
    manifest = {
        "kind": "summit.context.direction_contractions",
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "experimental": True,
        "estimand": DIRECTION_ESTIMAND,
        "shared_specification_hash": reference.manifest["shared_specification_hash"],
        "reference_manifest_hash": canonical_sha256(reference.manifest),
        "trait_manifest_hash": canonical_sha256(summary.manifest),
        "selected_variant_hash": reference.selected_variant_hash,
        "optimizer_reads_individual_data": False,
    }
    return ContextDirectionContractions(
        reference=reference, summary=summary, manifest=manifest
    )


def build_context_direction_contractions(
    reference_genotype: object,
    reference_context: object,
    reference_projector: object,
    study_genotype: object,
    study_context: object,
    study_projector: object,
    phenotype: object,
    **kwargs: Any,
) -> ContextDirectionContractions:
    """Convenience builder for a compatible exact reference/trait pair."""
    reference = build_direction_reference_contractions(
        reference_genotype,
        reference_context,
        reference_projector,
        **kwargs,
    )
    summary = build_direction_trait_contractions(
        study_genotype,
        study_context,
        study_projector,
        phenotype,
        **kwargs,
    )
    return combine_context_direction_contractions(reference, summary)


def validate_context_direction_contractions(
    contractions: ContextDirectionContractions,
) -> dict[str, Any]:
    rebuilt = combine_context_direction_contractions(
        contractions.reference, contractions.summary
    )
    if contractions.manifest != rebuilt.manifest:
        raise ValueError("Combined direction-contraction manifest mismatch.")
    return {
        "status": "valid_context_direction_contractions",
        "n_environments": len(contractions.environment_names),
        "n_pairs": len(contractions.pair_index),
        "contains_individual_arrays": False,
    }


def _contract_normal_equations(
    contractions: ContextDirectionContractions, direction: np.ndarray
) -> tuple[ContextNormalEquations, np.ndarray]:
    reference = contractions.reference
    summary = contractions.summary
    reduction = direction_reduction_matrix(direction, contractions.pair_index)
    p_count = len(contractions.pair_index)
    genetic_reduction = np.zeros((1 + p_count, 2), dtype=np.float64)
    genetic_reduction[0, 0] = 1.0
    genetic_reduction[1:, 1] = direction_pair_weights(
        direction, contractions.pair_index
    )
    reference_gram = genetic_reduction.T @ reference.base_gram @ genetic_reduction
    same_person = genetic_reduction.T @ reference.same_person @ genetic_reduction
    transferred = transfer_reference_gram(
        reference_gram,
        same_person,
        reference_n=reference.n_samples,
        study_n=summary.n_samples,
    )
    matrix = reduction.T @ summary.base_gram @ reduction
    matrix[:2, :2] = transferred
    matrix = 0.5 * (matrix + matrix.T)
    rhs = reduction.T @ summary.base_rhs
    traces = reduction.T @ summary.base_traces
    equations = ContextNormalEquations(
        matrix=np.ascontiguousarray(matrix),
        rhs=np.ascontiguousarray(rhs),
        traces=np.ascontiguousarray(traces),
        component_names=("G", "GxE(direction)", "N", "NxE(direction)"),
        genetic_count=2,
        annotation_masses=np.asarray([summary.annotation_mass], dtype=np.float64),
        deleted_groups=(),
        reference_genetic_gram=np.ascontiguousarray(reference_gram),
        transferred_genetic_gram=np.ascontiguousarray(transferred),
        reference_n=reference.n_samples,
        study_n=summary.n_samples,
    )
    return equations, reduction


def _nuisance_moment_fit(
    equations: ContextNormalEquations, rtol: float | None
) -> tuple[float, ContextSolveResult]:
    keep = np.asarray([0, 2, 3], dtype=np.int64)
    nuisance = ContextNormalEquations(
        matrix=equations.matrix[np.ix_(keep, keep)],
        rhs=equations.rhs[keep],
        traces=equations.traces[keep],
        component_names=tuple(equations.component_names[index] for index in keep),
        genetic_count=1,
        annotation_masses=equations.annotation_masses,
        deleted_groups=equations.deleted_groups,
        reference_genetic_gram=equations.reference_genetic_gram[:1, :1],
        transferred_genetic_gram=equations.transferred_genetic_gram[:1, :1],
        reference_n=equations.reference_n,
        study_n=equations.study_n,
    )
    solve = solve_context_normal_equations(nuisance, rtol=rtol, require_full_rank=True)
    return float(nuisance.rhs @ solve.coefficients), solve


def evaluate_context_direction(
    contractions: ContextDirectionContractions,
    direction: object,
    *,
    objective: str = "he_moment_gain",
    rtol: float | None = None,
) -> DirectionEvaluation:
    """Contract, solve, and score one fixed metric-normalized direction."""
    validate_context_direction_contractions(contractions)
    if objective not in DIRECTION_OBJECTIVES:
        raise ValueError(
            f"Unknown direction objective {objective!r}; expected one of "
            f"{DIRECTION_OBJECTIVES}."
        )
    omega = normalize_context_direction(direction, contractions.context_metric)
    equations, reduction = _contract_normal_equations(contractions, omega)
    solve = solve_context_normal_equations(equations, rtol=rtol, require_full_rank=True)
    full_fit = float(equations.rhs @ solve.coefficients)
    try:
        nuisance_fit, _ = _nuisance_moment_fit(equations, rtol)
    except ContextRankError:
        nuisance_fit = float("nan")
    he_gain = full_fit - nuisance_fit
    objective_values = {
        "interaction_coefficient": float(solve.coefficients[1]),
        "interaction_trace_contribution": float(
            solve.coefficients[1]
            * equations.traces[1]
            / contractions.summary.residual_rank
        ),
        "he_moment_gain": float(he_gain),
    }
    objective_value = objective_values[objective]
    if not np.isfinite(objective_value):
        raise ContextRankError(
            f"Objective {objective!r} is not estimable because its nuisance "
            "normal system is rank deficient.",
            diagnostics=solve.diagnostics,
            component_names=equations.component_names,
        )
    c = direction_pair_weights(omega, contractions.pair_index)
    return DirectionEvaluation(
        direction=omega,
        pair_weights=c,
        omega_outer=np.outer(omega, omega),
        reduction_matrix=reduction,
        equations=equations,
        solve=solve,
        matrix=equations.matrix,
        rhs=equations.rhs,
        traces=equations.traces,
        coefficients=solve.coefficients,
        trace_contributions=solve.coefficients
        * equations.traces
        / contractions.summary.residual_rank,
        objective_name=objective,
        objective_value=objective_value,
        objective_values=objective_values,
        full_moment_fit=full_fit,
        nuisance_moment_fit=nuisance_fit,
        rank=solve.rank,
        condition_number=solve.condition_number,
        relative_residual=solve.relative_residual,
        status="experimental_fixed_direction_evaluation_defined",
    )


def _metric_inverse_sqrt(metric: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def _canonical_unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("Cannot normalize a zero Euclidean direction.")
    result = np.asarray(value / norm, dtype=np.float64)
    absolute = np.abs(result)
    pivot = int(np.flatnonzero(absolute == np.max(absolute))[0])
    if result[pivot] < 0.0:
        result = -result
    return result


def _deterministic_unit_starts(l_count: int) -> np.ndarray:
    starts: list[np.ndarray] = []
    for index in range(l_count):
        value = np.zeros(l_count, dtype=np.float64)
        value[index] = 1.0
        starts.append(value)
    for left in range(l_count):
        for right in range(left + 1, l_count):
            for sign in (1.0, -1.0):
                value = np.zeros(l_count, dtype=np.float64)
                value[left] = 1.0
                value[right] = sign
                starts.append(_canonical_unit(value))
    unique: list[np.ndarray] = []
    for value in starts:
        if not any(
            np.allclose(value, prior, rtol=0.0, atol=1.0e-15) for prior in unique
        ):
            unique.append(value)
    return np.ascontiguousarray(np.stack(unique, axis=0))


def _unit_validation_grid(l_count: int, size: int) -> np.ndarray:
    if size < 1:
        return np.empty((0, l_count), dtype=np.float64)
    if l_count == 1:
        return np.ones((1, 1), dtype=np.float64)
    if l_count == 2:
        angles = np.linspace(0.0, np.pi, size, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    if l_count == 3:
        indices = np.arange(size, dtype=np.float64) + 0.5
        z = 1.0 - 2.0 * indices / size
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        golden = np.pi * (3.0 - np.sqrt(5.0))
        angles = golden * indices
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), z])
    raise ValueError("Validation grids are implemented only for L<=3.")


def optimize_context_direction(
    contractions: ContextDirectionContractions,
    *,
    objective: str = "he_moment_gain",
    rtol: float | None = None,
    validation_grid_size: int | None = None,
    maxiter: int = 500,
    ftol: float = 1.0e-12,
    max_condition_number: float = 1.0e10,
    projective_cluster_tolerance: float = 1.0e-6,
    objective_tie_rtol: float = 1.0e-8,
) -> DirectionOptimizationResult:
    """Maximize an explicit compact objective on the fixed metric sphere."""
    validate_context_direction_contractions(contractions)
    if objective not in DIRECTION_OBJECTIVES:
        raise ValueError(f"Unknown direction objective {objective!r}.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if not np.isfinite(ftol) or ftol <= 0.0:
        raise ValueError("ftol must be finite and positive.")
    if not np.isfinite(max_condition_number) or max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be finite and greater than one.")
    if (
        not np.isfinite(projective_cluster_tolerance)
        or projective_cluster_tolerance <= 0.0
        or projective_cluster_tolerance >= 1.0
    ):
        raise ValueError("projective_cluster_tolerance must lie strictly in (0,1).")
    if not np.isfinite(objective_tie_rtol) or objective_tie_rtol <= 0.0:
        raise ValueError("objective_tie_rtol must be finite and positive.")
    l_count = len(contractions.environment_names)
    inverse_sqrt = _metric_inverse_sqrt(contractions.context_metric)
    starts = _deterministic_unit_starts(l_count)
    evaluations = 0
    failed = 0

    def evaluate_unit(unit: np.ndarray) -> DirectionEvaluation | None:
        nonlocal evaluations, failed
        evaluations += 1
        try:
            evaluation = evaluate_context_direction(
                contractions,
                inverse_sqrt @ _canonical_unit(np.asarray(unit, dtype=np.float64)),
                objective=objective,
                rtol=rtol,
            )
            if evaluation.condition_number > max_condition_number:
                failed += 1
                return None
            return evaluation
        except (ValueError, ContextRankError, FloatingPointError):
            failed += 1
            return None

    def loss(unit: np.ndarray) -> float:
        evaluation = evaluate_unit(unit)
        return 1.0e100 if evaluation is None else -evaluation.objective_value

    messages: list[str] = []
    candidates: list[DirectionEvaluation] = []
    candidate_values: list[float] = []
    for start in starts:
        result = minimize(
            loss,
            start,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda value: float(value @ value - 1.0)},
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        messages.append(str(result.message))
        evaluation = evaluate_unit(result.x) if np.all(np.isfinite(result.x)) else None
        if result.success and evaluation is not None:
            candidates.append(evaluation)
            candidate_values.append(evaluation.objective_value)

    if validation_grid_size is None:
        validation_grid_size = 1 if l_count == 1 else (721 if l_count == 2 else 2048)
    if (
        isinstance(validation_grid_size, bool)
        or not isinstance(validation_grid_size, int)
        or validation_grid_size < 0
    ):
        raise ValueError("validation_grid_size must be a non-negative integer.")
    grid_best: DirectionEvaluation | None = None
    for point in _unit_validation_grid(l_count, validation_grid_size):
        evaluation = evaluate_unit(point)
        if evaluation is not None and (
            grid_best is None or evaluation.objective_value > grid_best.objective_value
        ):
            grid_best = evaluation
    if grid_best is not None:
        candidates.append(grid_best)
        candidate_values.append(grid_best.objective_value)
        grid_start = _canonical_unit(np.linalg.solve(inverse_sqrt, grid_best.direction))
        result = minimize(
            loss,
            grid_start,
            method="SLSQP",
            constraints={"type": "eq", "fun": lambda value: float(value @ value - 1.0)},
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        messages.append(str(result.message))
        refined = evaluate_unit(result.x) if np.all(np.isfinite(result.x)) else None
        if result.success and refined is not None:
            candidates.append(refined)
            candidate_values.append(refined.objective_value)
    if not candidates:
        raise ValueError(
            "No deterministic direction candidate produced an identifiable objective."
        )
    best_index = int(np.argmax(np.asarray(candidate_values)))
    best = candidates[best_index]
    distinct: list[DirectionEvaluation] = []
    for candidate in sorted(
        candidates, key=lambda value: value.objective_value, reverse=True
    ):
        same_projective_direction = any(
            abs(
                float(
                    candidate.direction @ contractions.context_metric @ prior.direction
                )
            )
            >= 1.0 - projective_cluster_tolerance
            for prior in distinct
        )
        if not same_projective_direction:
            distinct.append(candidate)
    distinct_objectives = np.asarray(
        [candidate.objective_value for candidate in distinct], dtype=np.float64
    )
    second_objective = None if len(distinct) < 2 else float(distinct[1].objective_value)
    objective_gap = (
        None
        if second_objective is None
        else float(best.objective_value - second_objective)
    )
    tie_scale = max(
        abs(best.objective_value),
        0.0 if second_objective is None else abs(second_objective),
        np.finfo(np.float64).tiny,
    )
    tie_tolerance = objective_tie_rtol * tie_scale
    nonunique = bool(objective_gap is not None and objective_gap <= tie_tolerance)
    near_optimal = [
        candidate.direction
        for candidate in distinct
        if best.objective_value - candidate.objective_value <= tie_tolerance
    ]
    grid_gap = (
        None
        if grid_best is None
        else float(best.objective_value - grid_best.objective_value)
    )
    manifest = {
        "kind": "summit.context.direction_optimization",
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "experimental": True,
        "estimand": DIRECTION_ESTIMAND,
        "objective": objective,
        "constraint": "omega.T @ fixed_context_metric @ omega == 1",
        "sign_convention": "largest_absolute_coordinate_positive_lowest_index_tie",
        "method": "deterministic_multistart_SLSQP_plus_fixed_sphere_grid",
        "validation_grid_size": validation_grid_size,
        "rtol": DEFAULT_SOLVE_RTOL if rtol is None else float(rtol),
        "maxiter": maxiter,
        "ftol": float(ftol),
        "max_condition_number": float(max_condition_number),
        "projective_cluster_tolerance": float(projective_cluster_tolerance),
        "objective_tie_rtol": float(objective_tie_rtol),
        "objective_tie_tolerance": float(tie_tolerance),
        "contractions_manifest_hash": canonical_sha256(contractions.manifest),
        "optimizer_reads_individual_data": False,
    }
    return DirectionOptimizationResult(
        direction=best.direction,
        evaluation=best,
        objective_name=objective,
        objective_value=best.objective_value,
        start_directions=np.asarray(
            [
                normalize_context_direction(
                    inverse_sqrt @ value, contractions.context_metric
                )
                for value in starts
            ]
        ),
        candidate_objectives=np.asarray(candidate_values, dtype=np.float64),
        grid_best_direction=None if grid_best is None else grid_best.direction,
        grid_best_objective=None if grid_best is None else grid_best.objective_value,
        grid_validation_gap=grid_gap,
        distinct_candidate_directions=np.asarray(
            [candidate.direction for candidate in distinct], dtype=np.float64
        ),
        distinct_candidate_objectives=distinct_objectives,
        near_optimal_directions=np.asarray(near_optimal, dtype=np.float64),
        second_distinct_objective=second_objective,
        objective_gap=objective_gap,
        nonunique=nonunique,
        evaluations=evaluations,
        converged_candidates=len(candidates),
        failed_candidates=failed,
        optimizer_messages=tuple(messages),
        status=(
            "experimental_compact_direction_optimization_nonunique"
            if nonunique
            else "experimental_compact_direction_optimization_defined"
        ),
        manifest=manifest,
    )


def _block_identity(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        payload: Any = {"type": "bool", "value": value}
    elif isinstance(value, int):
        payload = {"type": "int", "value": int(value)}
    elif isinstance(value, float) and np.isfinite(value):
        payload = {"type": "float", "value": float(value)}
    elif isinstance(value, str) and value:
        payload = {"type": "str", "value": value}
    else:
        raise ValueError(
            "Block labels must be finite typed scalars or nonempty strings."
        )
    return canonical_json(payload)


def _block_identities(block_ids: object) -> tuple[str, ...]:
    array = np.asarray(block_ids, dtype=object)
    if array.ndim != 1:
        raise ValueError("block_ids must be one-dimensional.")
    return tuple(_block_identity(value) for value in array)


def _fold_assignment_payload(
    *,
    variant_order_hash: str,
    block_sequence_hash: str,
    fold_labels: tuple[str, str],
    block_identities: tuple[str, ...],
    block_folds: tuple[int, ...],
    fold_variant_counts: tuple[int, int],
    fold_weight_masses: tuple[float, float],
    fold_selected_variant_hashes: tuple[str, str],
    fold_selected_weight_hashes: tuple[str, str],
) -> dict[str, Any]:
    return {
        "variant_order_hash": variant_order_hash,
        "block_sequence_hash": block_sequence_hash,
        "fold_labels": list(fold_labels),
        "block_identities": list(block_identities),
        "block_folds": list(block_folds),
        "fold_variant_counts": list(fold_variant_counts),
        "fold_weight_masses": list(fold_weight_masses),
        "fold_selected_variant_hashes": list(fold_selected_variant_hashes),
        "fold_selected_weight_hashes": list(fold_selected_weight_hashes),
        "method": "largest_block_first_greedy_two_fold_mass_balance",
    }


def build_balanced_direction_folds(
    block_ids: object,
    *,
    variant_hash: str = "development_variant_order",
    variant_weights: object | None = None,
    fold_labels: Sequence[str] = ("fold0", "fold1"),
) -> DirectionFoldAssignment:
    """Assign whole declared blocks to two deterministic mass-balanced folds."""
    identities = _block_identities(block_ids)
    n_variants = len(identities)
    if n_variants < 2:
        raise ValueError(
            "At least two variants are required for two-fold cross-fitting."
        )
    labels = tuple(_name(value, "fold label") for value in fold_labels)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("Exactly two unique fold labels are required.")
    if variant_weights is None:
        weights = np.ones(n_variants, dtype=np.float64)
    else:
        weights = _finite("variant_weights", variant_weights, ndim=1)
        if weights.shape != (n_variants,) or np.any(weights < 0.0):
            raise ValueError("variant_weights are invalid for block assignment.")
    unique = tuple(sorted(set(identities)))
    if len(unique) < 2:
        raise ValueError("At least two distinct variant blocks are required.")
    masses = {
        identity: float(
            np.sum(weights[np.asarray([value == identity for value in identities])])
        )
        for identity in unique
    }
    counts = {identity: identities.count(identity) for identity in unique}
    if any(value <= 0.0 for value in masses.values()):
        raise ValueError("Every declared block must have positive variant weight.")
    ordered = sorted(unique, key=lambda value: (-masses[value], -counts[value], value))
    fold_mass = [0.0, 0.0]
    fold_count = [0, 0]
    fold_blocks: list[list[str]] = [[], []]
    assignment: dict[str, int] = {}
    for identity in ordered:
        target = min(
            range(2), key=lambda fold: (fold_mass[fold], fold_count[fold], fold)
        )
        assignment[identity] = target
        fold_blocks[target].append(identity)
        fold_mass[target] += masses[identity]
        fold_count[target] += counts[identity]
    if any(not blocks for blocks in fold_blocks):
        raise ValueError("Balanced assignment produced an empty fold.")
    variant_identity = _identity_digest(variant_hash, "variant_hash")
    sequence_hash = array_sha256(
        np.asarray(
            [canonical_sha256({"block": value}) for value in identities], dtype="S64"
        )
    )
    block_folds = tuple(assignment[value] for value in unique)
    fold_selected_hashes = tuple(
        _selection_hash(
            variant_identity,
            n_variants,
            np.flatnonzero(
                np.asarray(
                    [assignment[value] == fold for value in identities],
                    dtype=np.bool_,
                )
            ),
        )
        for fold in range(2)
    )
    fold_selected_weight_hashes = tuple(
        array_sha256(
            np.ascontiguousarray(
                weights[
                    np.asarray(
                        [assignment[value] == fold for value in identities],
                        dtype=np.bool_,
                    )
                ],
                dtype=np.float64,
            )
        )
        for fold in range(2)
    )
    payload = _fold_assignment_payload(
        variant_order_hash=variant_identity,
        block_sequence_hash=sequence_hash,
        fold_labels=(labels[0], labels[1]),
        block_identities=unique,
        block_folds=block_folds,
        fold_variant_counts=(fold_count[0], fold_count[1]),
        fold_weight_masses=(fold_mass[0], fold_mass[1]),
        fold_selected_variant_hashes=(
            fold_selected_hashes[0],
            fold_selected_hashes[1],
        ),
        fold_selected_weight_hashes=(
            fold_selected_weight_hashes[0],
            fold_selected_weight_hashes[1],
        ),
    )
    return DirectionFoldAssignment(
        fold_labels=(labels[0], labels[1]),
        block_identities=unique,
        block_folds=block_folds,
        fold_block_identities=(tuple(fold_blocks[0]), tuple(fold_blocks[1])),
        fold_variant_counts=(fold_count[0], fold_count[1]),
        fold_weight_masses=(fold_mass[0], fold_mass[1]),
        fold_selected_variant_hashes=(
            fold_selected_hashes[0],
            fold_selected_hashes[1],
        ),
        fold_selected_weight_hashes=(
            fold_selected_weight_hashes[0],
            fold_selected_weight_hashes[1],
        ),
        variant_order_hash=variant_identity,
        block_sequence_hash=sequence_hash,
        assignment_hash=canonical_sha256(payload),
    )


def _fold_invariant_analysis_specification(
    shared_specification: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "schema_version",
        "estimand",
        "feature_mode",
        "environment_names",
        "pair_index",
        "pair_index_hash",
        "context_metric_hash",
        "context_metric_source",
        "context_metric_diagnostics",
        "context_spec_hash",
        "ordered_variant_hash",
        "genotype_scaling",
    )
    if any(field not in shared_specification for field in fields):
        raise ValueError("Direction fold is missing a shared analysis-spec field.")
    return {field: shared_specification[field] for field in fields}


def build_context_direction_crossfit_contractions(
    reference_genotype: object,
    reference_context: object,
    reference_projector: object,
    study_genotype: object,
    study_context: object,
    study_projector: object,
    phenotype: object,
    block_ids: object,
    *,
    context_metric: object,
    environment_names: Sequence[str] | None = None,
    variant_hash: str = "development_variant_order",
    context_spec_hash: str = "development_context_specification",
    context_metric_source: str = "reference_fixed_second_moment",
    genotype_scaling: str = "fixed_common_scale",
    reference_fixed_effect_hash: str | None = None,
    study_fixed_effect_hash: str | None = None,
    variant_weights: object | None = None,
    fold_labels: Sequence[str] = ("fold0", "fold1"),
) -> DirectionCrossFitContractions:
    """Build exact disjoint two-fold contractions and discard all row arrays."""
    reference_array = _finite("reference genotype", reference_genotype, ndim=2)
    study_array = _finite("study genotype", study_genotype, ndim=2)
    if reference_array.shape[1] != study_array.shape[1]:
        raise ValueError("Reference and study genotype variant counts differ.")
    n_variants = reference_array.shape[1]
    identities = _block_identities(block_ids)
    if len(identities) != n_variants:
        raise ValueError("block_ids must contain one label per ordered variant.")
    assignment = build_balanced_direction_folds(
        block_ids,
        variant_hash=variant_hash,
        variant_weights=variant_weights,
        fold_labels=fold_labels,
    )
    fold_objects: list[ContextDirectionContractions] = []
    for fold in range(2):
        mask = assignment.fold_mask(block_ids, fold)
        reference = build_direction_reference_contractions(
            reference_array,
            reference_context,
            reference_projector,
            context_metric=context_metric,
            environment_names=environment_names,
            variant_hash=variant_hash,
            context_spec_hash=context_spec_hash,
            context_metric_source=context_metric_source,
            genotype_scaling=genotype_scaling,
            fixed_effect_hash=reference_fixed_effect_hash,
            variant_weights=variant_weights,
            variant_mask=mask,
            fold_id=assignment.fold_labels[fold],
        )
        summary = build_direction_trait_contractions(
            study_array,
            study_context,
            study_projector,
            phenotype,
            context_metric=context_metric,
            environment_names=environment_names,
            variant_hash=variant_hash,
            context_spec_hash=context_spec_hash,
            context_metric_source=context_metric_source,
            genotype_scaling=genotype_scaling,
            fixed_effect_hash=study_fixed_effect_hash,
            variant_weights=variant_weights,
            variant_mask=mask,
            fold_id=assignment.fold_labels[fold],
        )
        fold_objects.append(combine_context_direction_contractions(reference, summary))
    analysis_specifications = tuple(
        _fold_invariant_analysis_specification(
            value.reference.manifest["shared_specification"]
        )
        for value in fold_objects
    )
    analysis_hashes = tuple(
        canonical_sha256(value) for value in analysis_specifications
    )
    if analysis_hashes[0] != analysis_hashes[1]:
        raise ValueError(
            "Exact direction folds do not share one analysis specification."
        )
    arms = (
        DirectionCrossFitArmContractions(
            fold_id="arm0",
            train_fold=assignment.fold_labels[0],
            heldout_fold=assignment.fold_labels[1],
            train_variant_hash=fold_objects[0].reference.selected_variant_hash,
            heldout_variant_hash=fold_objects[1].reference.selected_variant_hash,
            train=fold_objects[0],
            heldout=fold_objects[1],
        ),
        DirectionCrossFitArmContractions(
            fold_id="arm1",
            train_fold=assignment.fold_labels[1],
            heldout_fold=assignment.fold_labels[0],
            train_variant_hash=fold_objects[1].reference.selected_variant_hash,
            heldout_variant_hash=fold_objects[0].reference.selected_variant_hash,
            train=fold_objects[1],
            heldout=fold_objects[0],
        ),
    )
    manifest = {
        "kind": DIRECTION_CROSSFIT_KIND,
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "experimental": True,
        "estimand": DIRECTION_ESTIMAND,
        "assignment_hash": assignment.assignment_hash,
        "fold_labels": list(assignment.fold_labels),
        "fold_variant_counts": list(assignment.fold_variant_counts),
        "fold_weight_masses": list(assignment.fold_weight_masses),
        "fold_selected_variant_hashes": [
            value.reference.selected_variant_hash for value in fold_objects
        ],
        "fold_selected_weight_hashes": list(assignment.fold_selected_weight_hashes),
        "analysis_specification_hash": analysis_hashes[0],
        "construction": "exact_disjoint_variant_subset_rebuild",
        "uses_approximate_loo_deletion": False,
        "contains_individual_arrays": False,
    }
    return DirectionCrossFitContractions(
        assignment=assignment, arms=arms, manifest=manifest
    )


def validate_direction_crossfit_contractions(
    contractions: DirectionCrossFitContractions,
) -> dict[str, Any]:
    if contractions.manifest.get("kind") != DIRECTION_CROSSFIT_KIND:
        raise ValueError("Invalid direction-crossfit manifest kind.")
    assignment = contractions.assignment
    if len(assignment.fold_labels) != 2 or len(set(assignment.fold_labels)) != 2:
        raise ValueError("Direction-crossfit fold labels are invalid.")
    if (
        len(assignment.block_identities) != len(assignment.block_folds)
        or len(set(assignment.block_identities)) != len(assignment.block_identities)
        or not assignment.block_identities
        or set(assignment.block_folds) != {0, 1}
    ):
        raise ValueError("Direction-crossfit block assignment is invalid.")
    expected_blocks = tuple(
        tuple(
            identity
            for identity, fold in zip(
                assignment.block_identities, assignment.block_folds, strict=True
            )
            if fold == target
        )
        for target in range(2)
    )
    if tuple(
        tuple(sorted(values)) for values in assignment.fold_block_identities
    ) != tuple(tuple(sorted(values)) for values in expected_blocks):
        raise ValueError(
            "Direction-crossfit per-fold block identities are inconsistent."
        )
    if any(value < 1 for value in assignment.fold_variant_counts) or any(
        not np.isfinite(value) or value <= 0.0
        for value in assignment.fold_weight_masses
    ):
        raise ValueError("Direction-crossfit fold counts or masses are invalid.")
    payload = _fold_assignment_payload(
        variant_order_hash=assignment.variant_order_hash,
        block_sequence_hash=assignment.block_sequence_hash,
        fold_labels=assignment.fold_labels,
        block_identities=assignment.block_identities,
        block_folds=assignment.block_folds,
        fold_variant_counts=assignment.fold_variant_counts,
        fold_weight_masses=assignment.fold_weight_masses,
        fold_selected_variant_hashes=assignment.fold_selected_variant_hashes,
        fold_selected_weight_hashes=assignment.fold_selected_weight_hashes,
    )
    expected_assignment_hash = canonical_sha256(payload)
    if assignment.assignment_hash != expected_assignment_hash:
        raise ValueError(
            "Direction-crossfit assignment identity is internally inconsistent."
        )
    if contractions.manifest.get("assignment_hash") != expected_assignment_hash:
        raise ValueError("Direction-crossfit assignment hash mismatch.")
    manifest_expected = {
        "fold_labels": list(assignment.fold_labels),
        "fold_variant_counts": list(assignment.fold_variant_counts),
        "fold_weight_masses": list(assignment.fold_weight_masses),
        "fold_selected_weight_hashes": list(assignment.fold_selected_weight_hashes),
        "construction": "exact_disjoint_variant_subset_rebuild",
        "uses_approximate_loo_deletion": False,
        "contains_individual_arrays": False,
    }
    if any(
        contractions.manifest.get(field) != value
        for field, value in manifest_expected.items()
    ):
        raise ValueError("Direction-crossfit manifest does not match its assignment.")
    if len(contractions.arms) != 2:
        raise ValueError("Direction cross-fitting requires exactly two reverse arms.")
    canonical_folds = (
        contractions.arms[0].train,
        contractions.arms[0].heldout,
    )
    analysis_hashes = tuple(
        canonical_sha256(
            _fold_invariant_analysis_specification(
                value.reference.manifest["shared_specification"]
            )
        )
        for value in canonical_folds
    )
    if (
        analysis_hashes[0] != analysis_hashes[1]
        or contractions.manifest.get("analysis_specification_hash")
        != analysis_hashes[0]
    ):
        raise ValueError(
            "Direction crossfit folds do not share the frozen analysis specification."
        )
    for arm_index, arm in enumerate(contractions.arms):
        validate_context_direction_contractions(arm.train)
        validate_context_direction_contractions(arm.heldout)
        heldout_index = 1 - arm_index
        expected_train_label = assignment.fold_labels[arm_index]
        expected_heldout_label = assignment.fold_labels[heldout_index]
        expected_train_hash = assignment.fold_selected_variant_hashes[arm_index]
        expected_heldout_hash = assignment.fold_selected_variant_hashes[heldout_index]
        if (
            arm.fold_id != f"arm{arm_index}"
            or arm.train_fold != expected_train_label
            or arm.heldout_fold != expected_heldout_label
        ):
            raise ValueError("Crossfit arm labels do not match the frozen assignment.")
        if (
            arm.train_variant_hash != expected_train_hash
            or arm.heldout_variant_hash != expected_heldout_hash
        ):
            raise ValueError("Crossfit arm variant hashes do not match the assignment.")
        for role, fold_index, value in (
            ("training", arm_index, arm.train),
            ("held-out", heldout_index, arm.heldout),
        ):
            expected_count = assignment.fold_variant_counts[fold_index]
            expected_mass = assignment.fold_weight_masses[fold_index]
            value_analysis_hash = canonical_sha256(
                _fold_invariant_analysis_specification(
                    value.reference.manifest["shared_specification"]
                )
            )
            if value_analysis_hash != analysis_hashes[0]:
                raise ValueError(
                    f"Crossfit {role} fold uses a different analysis specification."
                )
            if (
                value.reference.n_variants != expected_count
                or value.summary.n_variants != expected_count
            ):
                raise ValueError(
                    f"Crossfit {role} variant count does not match the assignment."
                )
            mass_scale = max(abs(expected_mass), 1.0)
            if (
                abs(value.reference.annotation_mass - expected_mass)
                > 1.0e-12 * mass_scale
                or abs(value.summary.annotation_mass - expected_mass)
                > 1.0e-12 * mass_scale
            ):
                raise ValueError(
                    f"Crossfit {role} annotation mass does not match the assignment."
                )
            if (
                value.reference.manifest.get("fold_id")
                != assignment.fold_labels[fold_index]
                or value.summary.manifest.get("fold_id")
                != assignment.fold_labels[fold_index]
            ):
                raise ValueError(
                    f"Crossfit {role} fold identity does not match the assignment."
                )
            shared = value.reference.manifest["shared_specification"]
            if (
                shared.get("selected_weight_hash")
                != assignment.fold_selected_weight_hashes[fold_index]
            ):
                raise ValueError(
                    f"Crossfit {role} selected-weight identity does not match "
                    "the assignment."
                )
        if arm.train_variant_hash == arm.heldout_variant_hash:
            raise ValueError(
                "Training and held-out variant subsets are not disjoint identities."
            )
        if arm.train.reference.selected_variant_hash != arm.train_variant_hash:
            raise ValueError("Crossfit training variant hash mismatch.")
        if arm.heldout.reference.selected_variant_hash != arm.heldout_variant_hash:
            raise ValueError("Crossfit held-out variant hash mismatch.")
    if (
        contractions.arms[0].train_variant_hash
        != contractions.arms[1].heldout_variant_hash
    ):
        raise ValueError("Reverse crossfit arms do not exchange the exact folds.")
    if canonical_sha256(contractions.arms[0].train.manifest) != canonical_sha256(
        contractions.arms[1].heldout.manifest
    ) or canonical_sha256(contractions.arms[0].heldout.manifest) != canonical_sha256(
        contractions.arms[1].train.manifest
    ):
        raise ValueError(
            "Reverse crossfit arms do not reuse the identical frozen fold contractions."
        )
    selected_hashes = list(assignment.fold_selected_variant_hashes)
    if contractions.manifest.get("fold_selected_variant_hashes") != selected_hashes:
        raise ValueError(
            "Direction-crossfit selected-variant manifest is inconsistent."
        )
    if (
        contractions.arms[0].heldout_variant_hash
        != contractions.arms[1].train_variant_hash
    ):
        raise ValueError("Reverse crossfit arms do not exchange the exact folds.")
    return {
        "status": "valid_exact_two_fold_direction_contractions",
        "uses_approximate_loo_deletion": False,
        "contains_individual_arrays": False,
    }


def crossfit_context_direction(
    contractions: DirectionCrossFitContractions,
    *,
    objective: str = "he_moment_gain",
    rtol: float | None = None,
    validation_grid_size: int | None = None,
    maxiter: int = 500,
    ftol: float = 1.0e-12,
    max_condition_number: float = 1.0e10,
    projective_cluster_tolerance: float = 1.0e-6,
    objective_tie_rtol: float = 1.0e-8,
) -> DirectionCrossFitResult:
    """Optimize on each exact training fold and evaluate only its complement."""
    validate_direction_crossfit_contractions(contractions)
    results: list[DirectionCrossFitFoldResult] = []
    for arm in contractions.arms:
        training = optimize_context_direction(
            arm.train,
            objective=objective,
            rtol=rtol,
            validation_grid_size=validation_grid_size,
            maxiter=maxiter,
            ftol=ftol,
            max_condition_number=max_condition_number,
            projective_cluster_tolerance=projective_cluster_tolerance,
            objective_tie_rtol=objective_tie_rtol,
        )
        heldout = evaluate_context_direction(
            arm.heldout,
            training.direction,
            objective=objective,
            rtol=rtol,
        )
        if heldout.condition_number > max_condition_number:
            raise ValueError(
                f"Held-out direction system for {arm.fold_id!r} has condition "
                f"{heldout.condition_number:.6g}, exceeding the declared limit "
                f"{max_condition_number:.6g}; the complete crossfit is indeterminate."
            )
        results.append(
            DirectionCrossFitFoldResult(
                fold_id=arm.fold_id,
                train_variant_hash=arm.train_variant_hash,
                heldout_variant_hash=arm.heldout_variant_hash,
                training_optimization=training,
                heldout_evaluation=heldout,
                training_objective=training.objective_value,
                heldout_objective=heldout.objective_value,
            )
        )
    metric = contractions.arms[0].train.context_metric
    alignment = abs(float(results[0].direction @ metric @ results[1].direction))
    combined = float(np.mean([result.heldout_objective for result in results]))
    manifest = {
        "kind": "summit.context.direction_crossfit_result",
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "experimental": True,
        "objective": objective,
        "assignment_hash": contractions.assignment.assignment_hash,
        "contractions_manifest_hash": canonical_sha256(contractions.manifest),
        "combination": "equal_weight_mean_of_two_fixed_direction_heldout_objectives",
        "in_sample_objective_is_unbiased": False,
        "nonlinear_selection_inference": "not_calibrated",
    }
    nonunique = any(result.training_optimization.nonunique for result in results)
    return DirectionCrossFitResult(
        folds=(results[0], results[1]),
        objective_name=objective,
        combined_heldout_value=combined,
        fold_direction_alignment=alignment,
        status=(
            "experimental_exact_two_fold_direction_crossfit_nonunique_descriptive"
            if nonunique
            else "experimental_exact_two_fold_direction_crossfit_descriptive"
        ),
        manifest=manifest,
    )
