"""Narrow adapter from variant-LD-score references to contextual trait/fit V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

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
from summit.context.oracle import coefficients_to_omegas, omegas_to_coefficients
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


@dataclass(frozen=True)
class ProfiledResponseRankFitResultV1:
    """PSD fit with a fixed rank for the baseline-orthogonal response block."""

    response_rank: int
    coefficients: np.ndarray
    genetic_coefficients: np.ndarray
    residual_coefficients: np.ndarray
    omega: np.ndarray
    alignment: np.ndarray
    conditional_response: np.ndarray
    response_factor: np.ndarray
    objective_metric: str
    objective_rank: int
    profiled_quadratic_distance: float
    optimizer_cost: float
    optimizer_optimality: float
    optimizer_evaluations: int
    optimizer_status: int
    optimizer_message: str
    normal_equation_relative_residual: float

    def __post_init__(self) -> None:
        for name in (
            "coefficients",
            "genetic_coefficients",
            "residual_coefficients",
            "omega",
            "alignment",
            "conditional_response",
            "response_factor",
        ):
            object.__setattr__(
                self, name, owned_readonly_array(getattr(self, name))
            )


def _rank_parameterization(
    parameters: np.ndarray,
    *,
    response_count: int,
    response_rank: int,
    components: ContextComponentIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return genetic coefficients, Jacobian, Omega, alignment, and factor."""
    p = response_count
    r = response_rank
    expected = 1 + p + p * r
    if parameters.shape != (expected,):
        raise ValueError("rank-fit parameter vector has the wrong dimension")
    baseline = float(np.exp(parameters[0]))
    alignment = np.asarray(parameters[1 : 1 + p], dtype=np.float64)
    factor = np.asarray(parameters[1 + p :], dtype=np.float64).reshape(p, r)
    omega = np.zeros((p + 1, p + 1), dtype=np.float64)
    omega[0, 0] = baseline
    omega[0, 1:] = omega[1:, 0] = baseline * alignment
    omega[1:, 1:] = (
        baseline * np.outer(alignment, alignment) + factor @ factor.T
    )
    genetic = omegas_to_coefficients(omega[None, :, :], components)

    derivative_matrices: list[np.ndarray] = []
    baseline_derivative = np.zeros_like(omega)
    baseline_derivative[0, 0] = baseline
    baseline_derivative[0, 1:] = baseline_derivative[1:, 0] = (
        baseline * alignment
    )
    baseline_derivative[1:, 1:] = baseline * np.outer(
        alignment, alignment
    )
    derivative_matrices.append(baseline_derivative)
    for index in range(p):
        derivative = np.zeros_like(omega)
        unit = np.zeros(p, dtype=np.float64)
        unit[index] = 1.0
        derivative[0, 1:] = derivative[1:, 0] = baseline * unit
        derivative[1:, 1:] = baseline * (
            np.outer(unit, alignment) + np.outer(alignment, unit)
        )
        derivative_matrices.append(derivative)
    for column in range(r):
        for row in range(p):
            derivative = np.zeros_like(omega)
            unit = np.zeros(p, dtype=np.float64)
            unit[row] = 1.0
            derivative[1:, 1:] = np.outer(unit, factor[:, column]) + np.outer(
                factor[:, column], unit
            )
            derivative_matrices.append(derivative)
    jacobian = np.column_stack(
        [
            omegas_to_coefficients(value[None, :, :], components)
            for value in derivative_matrices
        ]
    )
    return genetic, jacobian, omega, alignment, factor


def fit_profiled_response_rank_v1(
    equations: ContextNormalEquations,
    components: ContextComponentIndex,
    *,
    response_rank: int,
    initial_genetic_coefficients: object | None = None,
    coefficient_covariance: object | None = None,
    maximum_evaluations: int = 4000,
) -> ProfiledResponseRankFitResultV1:
    """Fit rank 0 or 1 response geometry while profiling residual terms.

    The single-annotation covariance is parameterized as
    ``Omega=[[h, h*a.T], [h*a, h*a*a.T + B*B.T]]``.  This guarantees a PSD
    covariance and ``rank(S_perp) <= response_rank``.  Residual covariance
    components are solved exactly at every genetic parameter value.  When a
    coefficient covariance is supplied, the objective is the covariance-
    weighted distance from the raw genetic estimate; this is the appropriate
    default for method-of-moments systems whose normal matrix can be indefinite.
    """
    if not isinstance(equations, ContextNormalEquations):
        raise ValueError("equations must be ContextNormalEquations")
    if not isinstance(components, ContextComponentIndex):
        raise ValueError("components must be a ContextComponentIndex")
    if len(components.annotation_names) != 1:
        raise ValueError("response-rank fitting currently requires one annotation")
    q = components.pair_index.num_basis
    p = q - 1
    if isinstance(response_rank, bool) or response_rank not in (0, 1):
        raise ValueError("response_rank must be zero or one")
    if response_rank > p:
        raise ValueError("response_rank exceeds the response dimension")
    if (
        isinstance(maximum_evaluations, bool)
        or not isinstance(maximum_evaluations, int)
        or maximum_evaluations < 1
    ):
        raise ValueError("maximum_evaluations must be a positive integer")
    c = len(components)
    if equations.genetic_count != c:
        raise ValueError("normal equation genetic count does not match components")
    matrix = np.asarray(equations.matrix, dtype=np.float64)
    rhs = np.asarray(equations.rhs, dtype=np.float64)
    if matrix.shape != (len(rhs), len(rhs)) or len(rhs) <= c:
        raise ValueError("normal equation has incompatible dimensions")
    matrix = 0.5 * (matrix + matrix.T)
    genetic_matrix = matrix[:c, :c]
    cross_matrix = matrix[:c, c:]
    residual_matrix = matrix[c:, c:]
    genetic_rhs = rhs[:c]
    residual_rhs = rhs[c:]
    residual_cross_solve = np.linalg.solve(residual_matrix, cross_matrix.T)
    residual_rhs_solve = np.linalg.solve(residual_matrix, residual_rhs)
    if coefficient_covariance is None:
        profile_matrix = genetic_matrix - cross_matrix @ residual_cross_solve
        profile_matrix = 0.5 * (profile_matrix + profile_matrix.T)
        profile_rhs = genetic_rhs - cross_matrix @ residual_rhs_solve
        profile_values, profile_vectors = np.linalg.eigh(profile_matrix)
        profile_scale = max(float(profile_values[-1]), 1.0)
        if profile_values[0] <= 1.0e-12 * profile_scale:
            raise ValueError(
                "profiled genetic normal matrix is not positive definite; "
                "supply coefficient_covariance for a covariance-weighted fit"
            )
        unrestricted = np.linalg.solve(profile_matrix, profile_rhs)
        retained_profile = np.ones(c, dtype=bool)
        objective_metric = "profiled_normal_equation"
    else:
        covariance = np.asarray(coefficient_covariance, dtype=np.float64)
        if covariance.shape != (c, c) or np.any(~np.isfinite(covariance)):
            raise ValueError("coefficient_covariance has incompatible dimensions")
        covariance_scale = max(
            float(np.max(np.abs(covariance), initial=0.0)),
            np.finfo(np.float64).tiny,
        )
        asymmetry = float(
            np.max(np.abs(covariance - covariance.T), initial=0.0)
        )
        if asymmetry > 1.0e-10 * covariance_scale:
            raise ValueError("coefficient_covariance is materially asymmetric")
        covariance = 0.5 * (covariance + covariance.T)
        covariance_values, covariance_vectors = np.linalg.eigh(covariance)
        covariance_tolerance = (
            max(100.0 * np.finfo(np.float64).eps, 1.0e-10)
            * max(float(np.max(np.abs(covariance_values))), np.finfo(np.float64).tiny)
        )
        if covariance_values[0] < -covariance_tolerance:
            raise ValueError("coefficient_covariance is not positive semidefinite")
        retained_covariance = covariance_values > covariance_tolerance
        if not np.any(retained_covariance):
            raise ValueError("coefficient_covariance has zero numerical rank")
        covariance_precision = (
            covariance_vectors[:, retained_covariance]
            / covariance_values[retained_covariance]
        ) @ covariance_vectors[:, retained_covariance].T
        profile_matrix = 0.5 * (
            covariance_precision + covariance_precision.T
        )
        profile_values, profile_vectors = np.linalg.eigh(profile_matrix)
        profile_tolerance = 1.0e-10 * max(
            float(np.max(np.abs(profile_values))), np.finfo(np.float64).tiny
        )
        retained_profile = profile_values > profile_tolerance
        unrestricted = np.linalg.solve(matrix, rhs)[:c]
        objective_metric = "genetic_jackknife_covariance_precision"
    profile_factor = (
        np.sqrt(profile_values[retained_profile])[:, None]
        * profile_vectors[:, retained_profile].T
    )

    if initial_genetic_coefficients is None:
        initial_omega = coefficients_to_omegas(unrestricted, components)[0]
        values, vectors = np.linalg.eigh(0.5 * (initial_omega + initial_omega.T))
        initial_omega = (vectors * np.maximum(values, 0.0)) @ vectors.T
    else:
        initial = np.asarray(initial_genetic_coefficients, dtype=np.float64)
        if initial.shape != (c,) or np.any(~np.isfinite(initial)):
            raise ValueError("initial genetic coefficients are invalid")
        initial_omega = coefficients_to_omegas(initial, components)[0]
        if np.linalg.eigvalsh(0.5 * (initial_omega + initial_omega.T))[0] < -1.0e-10:
            raise ValueError("initial genetic covariance must be positive semidefinite")
    omega_scale = max(float(np.max(np.abs(initial_omega), initial=0.0)), 1.0e-12)
    baseline = max(float(initial_omega[0, 0]), 1.0e-10 * omega_scale)
    alignment = initial_omega[0, 1:] / baseline
    conditional = initial_omega[1:, 1:] - baseline * np.outer(
        alignment, alignment
    )
    conditional_values, conditional_vectors = np.linalg.eigh(
        0.5 * (conditional + conditional.T)
    )
    order = np.argsort(conditional_values)[::-1]
    conditional_values = np.maximum(conditional_values[order], 0.0)
    conditional_vectors = conditional_vectors[:, order]

    starts: list[np.ndarray] = []
    base = np.concatenate(([np.log(baseline)], alignment))
    if response_rank == 0:
        starts.append(base)
    else:
        for index in range(p):
            magnitude = np.sqrt(
                max(float(conditional_values[index]), 1.0e-10 * omega_scale)
            )
            starts.append(
                np.concatenate((base, magnitude * conditional_vectors[:, index]))
            )
        starts.append(np.concatenate((base, np.zeros(p, dtype=np.float64))))

    def residual(parameters: np.ndarray) -> np.ndarray:
        genetic, _, _, _, _ = _rank_parameterization(
            parameters,
            response_count=p,
            response_rank=response_rank,
            components=components,
        )
        return profile_factor @ (genetic - unrestricted)

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        _, genetic_jacobian, _, _, _ = _rank_parameterization(
            parameters,
            response_count=p,
            response_rank=response_rank,
            components=components,
        )
        return profile_factor @ genetic_jacobian

    lower = np.full(len(starts[0]), -np.inf, dtype=np.float64)
    upper = np.full(len(starts[0]), np.inf, dtype=np.float64)
    lower[0], upper[0] = -35.0, 35.0
    solutions = [
        least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=maximum_evaluations,
        )
        for start in starts
    ]
    successful = [solution for solution in solutions if solution.success]
    if not successful:
        messages = "; ".join(str(solution.message) for solution in solutions)
        raise RuntimeError(f"response-rank optimization failed: {messages}")
    solution = min(successful, key=lambda value: (value.cost, value.optimality))
    genetic, _, omega, fitted_alignment, response_factor = _rank_parameterization(
        np.asarray(solution.x, dtype=np.float64),
        response_count=p,
        response_rank=response_rank,
        components=components,
    )
    residual_coefficients = np.linalg.solve(
        residual_matrix, residual_rhs - cross_matrix.T @ genetic
    )
    coefficients = np.concatenate((genetic, residual_coefficients))
    normal_residual = matrix @ coefficients - rhs
    relative_residual = float(
        np.linalg.norm(normal_residual)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    )
    fitted_conditional = response_factor @ response_factor.T
    minimum = float(np.linalg.eigvalsh(0.5 * (omega + omega.T))[0])
    if minimum < -1.0e-9 * max(float(np.max(np.abs(omega))), 1.0):
        raise RuntimeError("response-rank parameterization produced non-PSD Omega")
    return ProfiledResponseRankFitResultV1(
        response_rank=response_rank,
        coefficients=coefficients,
        genetic_coefficients=genetic,
        residual_coefficients=residual_coefficients,
        omega=omega,
        alignment=fitted_alignment,
        conditional_response=fitted_conditional,
        response_factor=response_factor,
        objective_metric=objective_metric,
        objective_rank=int(np.sum(retained_profile)),
        profiled_quadratic_distance=float(np.sqrt(2.0 * solution.cost)),
        optimizer_cost=float(solution.cost),
        optimizer_optimality=float(solution.optimality),
        optimizer_evaluations=int(solution.nfev),
        optimizer_status=int(solution.status),
        optimizer_message=str(solution.message),
        normal_equation_relative_residual=relative_residual,
    )


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
