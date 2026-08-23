"""Summary-only fitting for the experimental contextual covariance model.

The fitter consumes only versioned reference and trait-summary objects.  It
assembles the complete small normal system, applies one declared rank policy,
and implements the current SNP-contribution approximate jackknife.  Raw MoM
coefficients are always primary; PSD projection is an optional, separately
labelled interpretation step for disjoint annotations.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
from scipy.optimize import minimize

from .oracle import (
    SymmetricRankDiagnostics,
    _finite_float64,
    coefficients_to_omegas,
    context_covariance_surface,
    omegas_to_coefficients,
    transfer_reference_gram,
)
from .reference import (
    ContextReference,
    GroupedContextReference,
    reference_moments_after_deleting_groups,
)
from .spec import (
    CONTEXT_FIT_KIND,
    CONTEXT_SCHEMA_VERSION,
    CONTEXT_TRAIT_KIND,
    CONTEXT_REFERENCE_KIND,
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
    freeze_context_mapping,
    freeze_context_value,
    owned_readonly_array,
    validate_context_manifest,
)
from .summary import (
    ContextTraitSummary,
    GroupedContextTraitSummary,
    trait_moments_after_deleting_groups,
)


DEFAULT_SOLVE_RTOL = 1.0e-10


@dataclass(frozen=True)
class ContextNormalEquations:
    matrix: np.ndarray
    rhs: np.ndarray
    traces: np.ndarray
    component_names: tuple[str, ...]
    genetic_count: int
    annotation_masses: np.ndarray
    deleted_groups: tuple[str, ...]
    reference_genetic_gram: np.ndarray
    transferred_genetic_gram: np.ndarray
    reference_n: int
    study_n: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_names", tuple(self.component_names))
        object.__setattr__(self, "deleted_groups", tuple(self.deleted_groups))
        for name in (
            "matrix",
            "rhs",
            "traces",
            "annotation_masses",
            "reference_genetic_gram",
            "transferred_genetic_gram",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class ContextSolveResult:
    coefficients: np.ndarray
    diagnostics: SymmetricRankDiagnostics
    rank: int
    condition_number: float
    solve_residual: np.ndarray
    relative_residual: float
    minimum_gram_eigenvalue: float
    retained_directions: np.ndarray
    null_space: np.ndarray
    relative_tolerance: float
    absolute_tolerance: float

    def __post_init__(self) -> None:
        for name in (
            "coefficients",
            "solve_residual",
            "retained_directions",
            "null_space",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class PSDProjectionResult:
    projected_coefficients: np.ndarray
    projected_omegas: np.ndarray
    distance: float
    euclidean_distance: float
    covariance_rank: int
    covariance_nullity: int
    minimum_eigenvalues: np.ndarray
    optimizer_success: bool
    optimizer_message: str
    tie_break_applied: bool
    cleanup_norm: float

    def __post_init__(self) -> None:
        for name in (
            "projected_coefficients",
            "projected_omegas",
            "minimum_eigenvalues",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class ContextFitResult:
    manifest: dict[str, Any]
    component_index: ContextComponentIndex
    residual_names: tuple[str, ...]
    equations: ContextNormalEquations
    solve: ContextSolveResult
    raw_coefficients: np.ndarray
    genetic_coefficients: np.ndarray
    residual_coefficients: np.ndarray
    raw_omegas: np.ndarray
    jackknife_groups: tuple[str, ...]
    loo_coefficients: np.ndarray
    jackknife_covariance: np.ndarray
    standard_errors: np.ndarray
    psd_projection: PSDProjectionResult | None
    context_outputs: dict[str, Any] | None
    jackknife_context_outputs: tuple[dict[str, Any], ...]
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        object.__setattr__(self, "residual_names", tuple(self.residual_names))
        object.__setattr__(self, "jackknife_groups", tuple(self.jackknife_groups))
        object.__setattr__(
            self,
            "phase_times_seconds",
            freeze_context_mapping(self.phase_times_seconds),
        )
        if self.context_outputs is not None:
            object.__setattr__(
                self, "context_outputs", freeze_context_value(self.context_outputs)
            )
        object.__setattr__(
            self,
            "jackknife_context_outputs",
            tuple(freeze_context_value(value) for value in self.jackknife_context_outputs),
        )
        for name in (
            "raw_coefficients",
            "genetic_coefficients",
            "residual_coefficients",
            "raw_omegas",
            "loo_coefficients",
            "jackknife_covariance",
            "standard_errors",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))

    @property
    def raw_genetic_coefficients(self) -> np.ndarray:
        return self.genetic_coefficients

    @property
    def raw_residual_coefficients(self) -> np.ndarray:
        return self.residual_coefficients

    @property
    def jackknife_coefficients(self) -> np.ndarray:
        return self.loo_coefficients


class ContextRankError(ValueError):
    """Raised when a declared contextual normal system is not identifiable."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: SymmetricRankDiagnostics,
        component_names: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.component_names = tuple(component_names)


class ContextJackknifeError(ValueError):
    """Raised when a required approximate-LOO replicate is not estimable."""

    def __init__(self, group: str, cause: BaseException) -> None:
        super().__init__(f"Approximate-LOO group {group!r} is not estimable: {cause}")
        self.group = group
        self.cause = cause


def validate_fit_compatibility(
    reference: ContextReference | GroupedContextReference,
    summary: ContextTraitSummary | GroupedContextTraitSummary,
) -> dict[str, Any]:
    """Validate internal identities and the cross-cohort fitting contract."""
    validate_context_manifest(reference.manifest, expected_kind=CONTEXT_REFERENCE_KIND)
    validate_context_manifest(summary.manifest, expected_kind=CONTEXT_TRAIT_KIND)
    shared_fields = (
        "feature_mode",
        "genotype_scaling",
        "basis_hash",
        "variant_hash",
        "annotation_hash",
        "component_index_hash",
        "loo_grouping_hash",
    )
    mismatches = {
        field: (reference.manifest.get(field), summary.manifest.get(field))
        for field in shared_fields
        if reference.manifest.get(field) != summary.manifest.get(field)
    }
    for field in (
        "annotation_mode",
        "annotation_definition_hash",
        "annotation_membership_hash",
        "annotation_partition_hash",
    ):
        left = reference.manifest.get(field)
        right = summary.manifest.get(field)
        if (left is not None or right is not None) and left != right:
            mismatches[field] = (left, right)
    for dimension in ("n_variants", "q", "k", "p_genetic"):
        left = reference.manifest["dimensions"].get(dimension)
        right = summary.manifest["dimensions"].get(dimension)
        if left != right:
            mismatches[f"dimensions.{dimension}"] = (left, right)
    if reference.component_index.names != summary.component_index.names:
        mismatches["component_order"] = (
            reference.component_index.names,
            summary.component_index.names,
        )
    if reference.loo_group_ids != summary.loo_group_ids:
        mismatches["loo_group_sequence"] = ("reference", "study")
    if mismatches:
        details = ", ".join(sorted(mismatches))
        raise ValueError(f"Reference/trait manifest compatibility mismatch: {details}.")
    if reference.n_samples < 2 or summary.n_samples < 1:
        raise ValueError("Reference and study sample sizes are invalid for transfer.")
    return {
        "matched_fields": list(shared_fields),
        "cohort_specific_fields": ["fixed_effect_hash", "n_samples", "residual_rank"],
        "reference_fixed_effect_hash": reference.manifest["fixed_effect_hash"],
        "study_fixed_effect_hash": summary.manifest["fixed_effect_hash"],
    }


def assemble_context_normal_equations(
    reference: ContextReference | GroupedContextReference,
    summary: ContextTraitSummary | GroupedContextTraitSummary,
    deleted_groups: Sequence[str] = (),
) -> ContextNormalEquations:
    """Assemble the canonical genetic-then-residual normal system."""
    validate_fit_compatibility(reference, summary)
    groups = tuple(str(value) for value in deleted_groups)
    if len(set(groups)) != len(groups):
        raise ValueError("Deleted approximate-LOO groups must be unique.")
    reference_moments = reference_moments_after_deleting_groups(reference, groups)
    trait_moments = trait_moments_after_deleting_groups(summary, groups)
    mass_scale = np.maximum(
        1.0,
        np.maximum(
            np.abs(reference_moments.annotation_masses),
            np.abs(trait_moments.annotation_masses),
        ),
    )
    if (
        np.max(
            np.abs(
                reference_moments.annotation_masses - trait_moments.annotation_masses
            )
            / mass_scale,
            initial=0.0,
        )
        > 1.0e-12
    ):
        raise ValueError("Reference and trait retained annotation masses differ.")
    transferred = transfer_reference_gram(
        reference_moments.gram,
        reference_moments.same_person,
        reference_n=reference.n_samples,
        study_n=summary.n_samples,
    )
    p_genetic = len(summary.component_index)
    h_count = len(summary.residual_names)
    matrix = np.empty((p_genetic + h_count, p_genetic + h_count), dtype=np.float64)
    matrix[:p_genetic, :p_genetic] = transferred
    matrix[:p_genetic, p_genetic:] = trait_moments.genetic_residual
    matrix[p_genetic:, :p_genetic] = trait_moments.genetic_residual.T
    matrix[p_genetic:, p_genetic:] = trait_moments.residual_gram
    matrix = 0.5 * (matrix + matrix.T)
    rhs = np.concatenate([trait_moments.genetic_rhs, trait_moments.residual_rhs])
    traces = np.concatenate(
        [trait_moments.genetic_traces, trait_moments.residual_traces]
    )
    names = summary.component_index.names + summary.residual_names
    return ContextNormalEquations(
        matrix=matrix,
        rhs=rhs,
        traces=traces,
        component_names=names,
        genetic_count=p_genetic,
        annotation_masses=trait_moments.annotation_masses,
        deleted_groups=groups,
        reference_genetic_gram=reference_moments.gram,
        transferred_genetic_gram=transferred,
        reference_n=reference.n_samples,
        study_n=summary.n_samples,
    )


def _rank_diagnostics(
    matrix: np.ndarray, relative_tolerance: float
) -> tuple[SymmetricRankDiagnostics, np.ndarray, float]:
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("rtol must be finite and positive.")
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0)
    asymmetry = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    if asymmetry > 2.0e-10 * scale:
        raise ValueError(f"Normal matrix has material asymmetry ({asymmetry:.6g}).")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    singular_values = np.sort(np.abs(eigenvalues))[::-1]
    leading = float(singular_values[0]) if singular_values.size else 0.0
    absolute_tolerance = 100.0 * np.finfo(np.float64).eps * scale
    tolerance = max(absolute_tolerance, relative_tolerance * leading)
    retained = np.abs(eigenvalues) > tolerance
    rank = int(np.sum(retained))
    condition = (
        float("inf")
        if rank < matrix.shape[0] or rank == 0
        else float(leading / np.min(np.abs(eigenvalues[retained])))
    )
    diagnostics = SymmetricRankDiagnostics(
        symmetry_error=asymmetry,
        eigenvalues=eigenvalues,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition,
        tolerance=float(tolerance),
        null_space=eigenvectors[:, ~retained],
    )
    return diagnostics, eigenvectors[:, retained], absolute_tolerance


def solve_context_normal_equations(
    equations: ContextNormalEquations,
    *,
    rtol: float | None = None,
    require_full_rank: bool = True,
) -> ContextSolveResult:
    """Solve a complete symmetric system without an inverse or regularizer."""
    matrix = _finite_float64("normal matrix", equations.matrix, ndim=2)
    rhs = _finite_float64("normal RHS", equations.rhs, ndim=1)
    if matrix.shape[0] != matrix.shape[1] or rhs.shape != (matrix.shape[0],):
        raise ValueError("Normal-equation shapes are incompatible.")
    relative_tolerance = DEFAULT_SOLVE_RTOL if rtol is None else float(rtol)
    diagnostics, retained_vectors, absolute_tolerance = _rank_diagnostics(
        matrix, relative_tolerance
    )
    if require_full_rank and diagnostics.rank != matrix.shape[0]:
        raise ContextRankError(
            "Contextual normal system is not identifiable: "
            f"rank {diagnostics.rank} of {matrix.shape[0]} at tolerance "
            f"{diagnostics.tolerance:.6g}.",
            diagnostics=diagnostics,
            component_names=equations.component_names,
        )
    symmetric = 0.5 * (matrix + matrix.T)
    if diagnostics.rank == 0:
        coefficients = np.zeros_like(rhs)
    else:
        retained_eigenvalues = np.einsum(
            "ni,nm,mi->i", retained_vectors, symmetric, retained_vectors
        )
        coefficients = retained_vectors @ (
            (retained_vectors.T @ rhs) / retained_eigenvalues
        )
    residual = symmetric @ coefficients - rhs
    relative_residual = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs)))
    return ContextSolveResult(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        diagnostics=diagnostics,
        rank=diagnostics.rank,
        condition_number=diagnostics.condition_number,
        solve_residual=residual,
        relative_residual=relative_residual,
        minimum_gram_eigenvalue=float(diagnostics.eigenvalues[0]),
        retained_directions=retained_vectors,
        null_space=diagnostics.null_space,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def _unweighted_psd_start(
    coefficients: np.ndarray, components: ContextComponentIndex
) -> np.ndarray:
    omegas = coefficients_to_omegas(coefficients, components)
    projected = np.empty_like(omegas)
    for k, omega in enumerate(omegas):
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (omega + omega.T))
        projected[k] = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return omegas_to_coefficients(projected, components)


def _factorized_psd_projection_fallback(
    raw: np.ndarray,
    precision: np.ndarray,
    components: ContextComponentIndex,
) -> tuple[np.ndarray, str]:
    """Solve the PSD projection through full factors and certify its KKT point.

    This deterministic fallback is used only if the direct convex SLSQP
    formulation fails. Full square factors represent the entire PSD cone. We
    use several starts and require primal feasibility, dual feasibility, and
    complementarity before accepting the lowest objective candidate.
    """
    k_count = len(components.annotation_names)
    q_count = components.pair_index.num_basis

    def factors_to_coefficients(flat: np.ndarray) -> np.ndarray:
        factors = np.asarray(flat, dtype=np.float64).reshape(k_count, q_count, q_count)
        omegas = np.einsum("kij,klj->kil", factors, factors, optimize=True)
        return omegas_to_coefficients(omegas, components)

    def coefficient_gradient_matrices(gradient: np.ndarray) -> np.ndarray:
        matrices = np.zeros((k_count, q_count, q_count), dtype=np.float64)
        for component in components.entries:
            value = float(gradient[component.index])
            if component.q == component.r:
                matrices[component.annotation_index, component.q, component.r] += value
            else:
                matrices[component.annotation_index, component.q, component.r] += (
                    0.5 * value
                )
                matrices[component.annotation_index, component.r, component.q] += (
                    0.5 * value
                )
        return matrices

    def objective(flat: np.ndarray) -> float:
        difference = factors_to_coefficients(flat) - raw
        return float(difference @ precision @ difference)

    def gradient(flat: np.ndarray) -> np.ndarray:
        factors = np.asarray(flat, dtype=np.float64).reshape(k_count, q_count, q_count)
        coefficients = factors_to_coefficients(flat)
        coefficient_gradient = 2.0 * precision @ (coefficients - raw)
        matrix_gradient = coefficient_gradient_matrices(coefficient_gradient)
        factor_gradient = 2.0 * np.einsum(
            "kij,kjl->kil", matrix_gradient, factors, optimize=True
        )
        return factor_gradient.ravel()

    raw_omegas = coefficients_to_omegas(raw, components)
    spectral = np.empty((k_count, q_count, q_count), dtype=np.float64)
    coefficient_scale = max(float(np.max(np.abs(raw), initial=0.0)), 1.0e-8)
    amplitude = np.sqrt(coefficient_scale)
    for k, omega in enumerate(raw_omegas):
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (omega + omega.T))
        spectral[k] = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    starts = [
        spectral,
        spectral + np.eye(q_count, dtype=np.float64)[None, :, :] * amplitude * 1.0e-4,
        np.broadcast_to(
            np.eye(q_count, dtype=np.float64) * amplitude,
            (k_count, q_count, q_count),
        ).copy(),
    ]
    rng = np.random.default_rng(0)
    starts.extend(
        rng.normal(scale=amplitude / np.sqrt(q_count), size=spectral.shape)
        for _ in range(3)
    )
    candidates: list[tuple[float, np.ndarray, str]] = []
    for start in starts:
        result = minimize(
            objective,
            np.asarray(start, dtype=np.float64).ravel(),
            jac=gradient,
            method="L-BFGS-B",
            options={
                "ftol": 1.0e-15,
                "gtol": 1.0e-9,
                "maxiter": 4000,
                "maxls": 80,
            },
        )
        value = objective(np.asarray(result.x, dtype=np.float64))
        if np.isfinite(value):
            candidates.append((value, np.asarray(result.x), str(result.message)))
    if not candidates:
        raise RuntimeError("Factorized PSD fallback returned no finite candidate.")
    _, best_factors, message = min(candidates, key=lambda item: item[0])
    candidate = factors_to_coefficients(best_factors)
    omegas = coefficients_to_omegas(candidate, components)
    coefficient_gradient = 2.0 * precision @ (candidate - raw)
    dual_matrices = coefficient_gradient_matrices(coefficient_gradient)
    gradient_scale = max(float(np.max(np.abs(coefficient_gradient), initial=0.0)), 1.0)
    dual_tolerance = 2.0e-7 * gradient_scale
    complementarity_error = 0.0
    for omega, dual in zip(omegas, dual_matrices):
        if np.min(np.linalg.eigvalsh(omega)) < -1.0e-10 * coefficient_scale:
            raise RuntimeError("Factorized PSD fallback failed primal feasibility.")
        if np.min(np.linalg.eigvalsh(dual)) < -dual_tolerance:
            raise RuntimeError("Factorized PSD fallback failed dual feasibility.")
        complementarity_error = max(
            complementarity_error,
            float(np.linalg.norm(dual @ omega, ord="fro")),
        )
    complementarity_scale = max(
        1.0,
        float(np.linalg.norm(dual_matrices)) * float(np.linalg.norm(omegas)),
    )
    if complementarity_error > 2.0e-7 * complementarity_scale:
        raise RuntimeError("Factorized PSD fallback failed complementarity.")
    return candidate, f"factorized fallback certified; {message}"


def project_genetic_coefficients_psd(
    theta: object,
    covariance: object,
    components: ContextComponentIndex,
    *,
    annotations_disjoint: bool = True,
) -> PSDProjectionResult:
    """Covariance-aware PSD projection without replacing the raw estimate."""
    if not annotations_disjoint:
        raise ValueError(
            "Covariance-aware annotation-wise PSD projection is only interpretable "
            "for disjoint annotations."
        )
    reported_raw = _finite_float64("genetic coefficients", theta, ndim=1)
    if reported_raw.shape != (len(components),):
        raise ValueError("Genetic coefficient count does not match component index.")
    covariance_array = _finite_float64("coefficient covariance", covariance, ndim=2)
    if covariance_array.shape != (len(components), len(components)):
        raise ValueError("PSD projection covariance has incompatible dimensions.")
    # Work in coefficient units of order one.  This makes optimizer stopping
    # rules and feasibility cleanup invariant to a simultaneous change of
    # coefficient units (theta -> s theta, V -> s^2 V).
    optimization_scale = float(np.max(np.abs(reported_raw), initial=0.0))
    if optimization_scale > 0.0:
        raw = reported_raw / optimization_scale
        covariance_array = covariance_array / (optimization_scale * optimization_scale)
    else:
        optimization_scale = 1.0
        raw = reported_raw.copy()
    covariance_scale = max(
        float(np.max(np.abs(covariance_array), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    covariance_asymmetry = float(
        np.max(np.abs(covariance_array - covariance_array.T), initial=0.0)
    )
    if covariance_asymmetry > 1.0e-10 * covariance_scale:
        raise ValueError("PSD projection covariance is materially asymmetric.")
    covariance_symmetric = 0.5 * (covariance_array + covariance_array.T)
    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(
        covariance_symmetric
    )
    leading_covariance_eigenvalue = float(
        np.max(np.abs(covariance_eigenvalues), initial=0.0)
    )
    covariance_tolerance = (
        max(100.0 * np.finfo(np.float64).eps, 1.0e-10) * leading_covariance_eigenvalue
    )
    if covariance_eigenvalues[0] < -covariance_tolerance:
        raise ValueError("Jackknife covariance is not positive semidefinite.")
    retained = covariance_eigenvalues > covariance_tolerance
    covariance_rank = int(np.sum(retained))
    if covariance_rank:
        vectors = covariance_eigenvectors[:, retained]
        precision = (vectors / covariance_eigenvalues[retained]) @ vectors.T
    else:
        precision = np.zeros_like(covariance_symmetric)

    def primary(value: np.ndarray) -> float:
        difference = value - raw
        return float(difference @ precision @ difference)

    def primary_gradient(value: np.ndarray) -> np.ndarray:
        return 2.0 * precision @ (value - raw)

    def psd_eigenvalues(value: np.ndarray) -> np.ndarray:
        omegas = coefficients_to_omegas(value, components)
        return np.concatenate([np.linalg.eigvalsh(omega) for omega in omegas])

    raw_minimum = psd_eigenvalues(raw)
    if np.min(raw_minimum) >= -1.0e-12:
        return PSDProjectionResult(
            projected_coefficients=reported_raw.copy(),
            projected_omegas=coefficients_to_omegas(reported_raw, components),
            distance=0.0,
            euclidean_distance=0.0,
            covariance_rank=covariance_rank,
            covariance_nullity=len(components) - covariance_rank,
            minimum_eigenvalues=np.asarray(
                [
                    np.min(np.linalg.eigvalsh(omega))
                    for omega in coefficients_to_omegas(reported_raw, components)
                ]
            ),
            optimizer_success=True,
            optimizer_message="raw estimate already PSD",
            tie_break_applied=False,
            cleanup_norm=0.0,
        )
    start = _unweighted_psd_start(raw, components)
    constraint = {"type": "ineq", "fun": psd_eigenvalues}
    first = minimize(
        primary,
        start,
        jac=primary_gradient,
        constraints=(constraint,),
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 2000, "disp": False},
    )
    if first.success:
        candidate = np.asarray(first.x, dtype=np.float64)
        optimizer_message = str(first.message)
    else:
        candidate, fallback_message = _factorized_psd_projection_fallback(
            raw, precision, components
        )
        optimizer_message = f"direct SLSQP failed ({first.message}); {fallback_message}"
    primary_optimum = primary(candidate)
    tie_break_applied = covariance_rank < len(components)
    if tie_break_applied:
        primary_slack = max(1.0e-14, 1.0e-12 * (1.0 + primary_optimum))

        def euclidean_objective(value: np.ndarray) -> float:
            difference = value - raw
            return float(difference @ difference)

        def euclidean_gradient(value: np.ndarray) -> np.ndarray:
            return 2.0 * (value - raw)

        primary_constraint = {
            "type": "ineq",
            "fun": lambda value: primary_optimum + primary_slack - primary(value),
        }
        second = minimize(
            euclidean_objective,
            candidate,
            jac=euclidean_gradient,
            constraints=(constraint, primary_constraint),
            method="SLSQP",
            options={"ftol": 1.0e-12, "maxiter": 3000, "disp": False},
        )
        if second.success:
            candidate = np.asarray(second.x, dtype=np.float64)
            optimizer_message = f"{optimizer_message}; tie-break: {second.message}"
        else:
            raise RuntimeError(
                "Covariance-aware PSD projection tie-break failed: " f"{second.message}"
            )

    candidate_omegas = coefficients_to_omegas(candidate, components)
    cleaned = np.empty_like(candidate_omegas)
    for k, omega in enumerate(candidate_omegas):
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (omega + omega.T))
        if eigenvalues[0] < -1.0e-7:
            raise RuntimeError(
                "PSD optimizer returned a materially infeasible covariance matrix."
            )
        cleaned[k] = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    cleaned_coefficients = omegas_to_coefficients(cleaned, components)
    reported_coefficients = optimization_scale * cleaned_coefficients
    reported_omegas = optimization_scale * cleaned
    cleanup_norm = float(
        optimization_scale * np.linalg.norm(cleaned_coefficients - candidate)
    )
    difference = reported_coefficients - reported_raw
    minimum_eigenvalues = np.asarray(
        [np.min(np.linalg.eigvalsh(omega)) for omega in reported_omegas],
        dtype=np.float64,
    )
    return PSDProjectionResult(
        projected_coefficients=reported_coefficients,
        projected_omegas=reported_omegas,
        distance=float(np.sqrt(max(primary(cleaned_coefficients), 0.0))),
        euclidean_distance=float(np.linalg.norm(difference)),
        covariance_rank=covariance_rank,
        covariance_nullity=len(components) - covariance_rank,
        minimum_eigenvalues=minimum_eigenvalues,
        optimizer_success=True,
        optimizer_message=optimizer_message,
        tie_break_applied=tie_break_applied,
        cleanup_norm=cleanup_norm,
    )


def derive_context_outputs(
    genetic_coefficients: object,
    components: ContextComponentIndex,
    context_grid: object,
    basis_metric: object,
    genetic_traces: object | None = None,
) -> dict[str, Any]:
    """Compute coefficient surfaces and separately labelled trace summaries."""
    coefficients = _finite_float64("genetic_coefficients", genetic_coefficients, ndim=1)
    if coefficients.shape != (len(components),):
        raise ValueError("Genetic coefficient count does not match component index.")
    grid = _finite_float64("context_grid", context_grid, ndim=2)
    q_count = components.pair_index.num_basis
    if grid.shape[1] != q_count or grid.shape[0] < 1:
        raise ValueError("Context grid has incompatible basis dimension.")
    metric = _finite_float64("basis_metric", basis_metric, ndim=2)
    if metric.shape != (q_count, q_count):
        raise ValueError("Basis metric has incompatible dimensions.")
    metric_asymmetry = np.max(np.abs(metric - metric.T), initial=0.0)
    metric_scale = max(float(np.max(np.abs(metric), initial=0.0)), 1.0)
    if metric_asymmetry > 1.0e-10 * metric_scale:
        raise ValueError("Basis metric must be symmetric.")
    metric_eigenvalues, metric_eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    metric_tolerance = 1.0e-12 * metric_scale
    if metric_eigenvalues[0] < -metric_tolerance:
        raise ValueError("Basis metric must be positive semidefinite.")
    metric_sqrt = (
        metric_eigenvectors * np.sqrt(np.maximum(metric_eigenvalues, 0.0))
    ) @ metric_eigenvectors.T
    traces: np.ndarray | None = None
    if genetic_traces is not None:
        traces = _finite_float64("genetic_traces", genetic_traces, ndim=1)
        if traces.shape != (len(components),):
            raise ValueError("Genetic trace count does not match component index.")
    omegas = coefficients_to_omegas(coefficients, components)
    annotations: list[dict[str, Any]] = []
    for k, annotation_name in enumerate(components.annotation_names):
        omega = omegas[k]
        surface = context_covariance_surface(omega, grid)
        variances = np.diag(surface).copy()
        denominator = np.sqrt(
            np.maximum(variances[:, None], 0.0) * np.maximum(variances[None, :], 0.0)
        )
        correlation_defined = (
            (variances[:, None] > 0.0)
            & (variances[None, :] > 0.0)
            & (denominator > 0.0)
        )
        correlations = np.full_like(surface, np.nan)
        correlations[correlation_defined] = (
            surface[correlation_defined] / denominator[correlation_defined]
        )
        amplification_difference = variances - variances[0]
        log_sd_ratio = np.full_like(variances, np.nan)
        if variances[0] > 0.0:
            positive = variances > 0.0
            log_sd_ratio[positive] = 0.5 * np.log(variances[positive] / variances[0])
        orthogonal = np.full_like(surface, np.nan)
        for i in range(grid.shape[0]):
            if variances[i] > 0.0:
                orthogonal[i] = variances - surface[i] * surface[i] / variances[i]
        operator = metric_sqrt @ omega @ metric_sqrt
        operator_eigenvalues = np.linalg.eigvalsh(0.5 * (operator + operator.T))[::-1]
        operator_scale = max(
            float(np.max(np.abs(operator_eigenvalues), initial=0.0)), 1.0
        )
        if operator_eigenvalues[-1] < -1.0e-10 * operator_scale:
            rank_one_fraction = float("nan")
            rank_one_fraction_status = "undefined_indefinite_raw_operator"
        else:
            positive_sum = float(
                np.sum(np.maximum(operator_eigenvalues, 0.0), dtype=np.float64)
            )
            if positive_sum > 0.0:
                rank_one_fraction = float(
                    max(operator_eigenvalues[0], 0.0) / positive_sum
                )
                rank_one_fraction_status = "defined_psd_operator"
            else:
                rank_one_fraction = float("nan")
                rank_one_fraction_status = "undefined_zero_operator"
        trace_entries: list[dict[str, Any]] = []
        trace_total: float | None = None
        if traces is not None:
            selected = [
                entry for entry in components.entries if entry.annotation_index == k
            ]
            contributions = np.asarray(
                [coefficients[entry.index] * traces[entry.index] for entry in selected]
            )
            trace_entries = [
                {
                    "component": entry.name,
                    "coefficient_times_trace": float(value),
                }
                for entry, value in zip(selected, contributions)
            ]
            trace_total = float(np.sum(contributions, dtype=np.float64))
        annotations.append(
            {
                "annotation": annotation_name,
                "omega": omega,
                "covariance_surface": surface,
                "variances": variances,
                "correlations": correlations,
                "correlation_defined": correlation_defined,
                "amplification_difference_from_first": amplification_difference,
                "log_sd_ratio_from_first": log_sd_ratio,
                "orthogonal_heterogeneity": orthogonal,
                "basis_metric_operator_eigenvalues": operator_eigenvalues,
                "spectral_rank_one_fraction": rank_one_fraction,
                "spectral_rank_one_fraction_status": rank_one_fraction_status,
                "trace_component_contributions": trace_entries,
                "trace_annotation_total": trace_total,
            }
        )
    trace_genetic_total = (
        None if traces is None else float(np.dot(coefficients, traces))
    )
    return {
        "context_grid": grid,
        "basis_metric": metric,
        "annotations": annotations,
        "trace_genetic_total": trace_genetic_total,
        "surface_semantics": "coefficient_covariance_not_observed_sample_trace",
        "trace_semantics": "coefficient_times_observed_kernel_trace",
    }


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _validate_equal_weight_groups(
    all_group_ids: Sequence[str],
    selected_groups: Sequence[str],
    group_variant_counts: Sequence[int] | None = None,
) -> dict[str, Any]:
    if group_variant_counts is None:
        counts = np.asarray(
            [
                sum(value == group for value in all_group_ids)
                for group in selected_groups
            ],
            dtype=np.int64,
        )
    else:
        if len(group_variant_counts) != len(all_group_ids):
            raise ValueError("Grouped approximate-LOO counts have invalid length.")
        lookup = {
            str(label): int(count)
            for label, count in zip(all_group_ids, group_variant_counts)
        }
        counts = np.asarray(
            [lookup[group] for group in selected_groups], dtype=np.int64
        )
    if counts.size < 2 or np.any(counts < 1):
        raise ValueError("Approximate jackknife requires at least two nonempty groups.")
    if int(np.max(counts) - np.min(counts)) > 1:
        raise ValueError(
            "Equal-weight approximate jackknife requires balanced SNP groups "
            "whose sizes differ by at most one."
        )
    return {
        "group_count": int(counts.size),
        "minimum_variants": int(np.min(counts)),
        "maximum_variants": int(np.max(counts)),
        "weighting": "equal_group_delete_one",
    }


def fit_context_model(
    reference: ContextReference | GroupedContextReference,
    summary: ContextTraitSummary | GroupedContextTraitSummary,
    *,
    rtol: float | None = None,
    loo_groups: Sequence[str] | None = None,
    context_grid: object | None = None,
    basis_metric: object | None = None,
    project_psd: bool = False,
    annotations_disjoint: bool | None = None,
) -> ContextFitResult:
    """Fit the unified model using summary objects only."""
    total_start = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    compatibility = validate_fit_compatibility(reference, summary)
    relative_tolerance = DEFAULT_SOLVE_RTOL if rtol is None else float(rtol)
    phase_times: dict[str, float] = {}

    phase_start = time.perf_counter()
    equations = assemble_context_normal_equations(reference, summary)
    solve = solve_context_normal_equations(
        equations, rtol=relative_tolerance, require_full_rank=True
    )
    phase_times["full_assembly_and_solve"] = time.perf_counter() - phase_start
    peak_rss = max(peak_rss, process.memory_info().rss)
    coefficients = solve.coefficients
    p_genetic = equations.genetic_count
    genetic = coefficients[:p_genetic].copy()
    residual = coefficients[p_genetic:].copy()
    raw_omegas = coefficients_to_omegas(genetic, summary.component_index)

    available_groups = _ordered_unique(summary.loo_group_ids)
    selected_groups = (
        available_groups
        if loo_groups is None
        else tuple(str(value) for value in loo_groups)
    )
    if len(set(selected_groups)) != len(selected_groups):
        raise ValueError("Approximate-jackknife group selection contains duplicates.")
    unknown_groups = set(selected_groups) - set(available_groups)
    if unknown_groups:
        raise ValueError(
            f"Unknown approximate-jackknife groups: {sorted(unknown_groups)}."
        )
    if isinstance(summary, GroupedContextTraitSummary) and set(selected_groups) != set(
        available_groups
    ):
        raise ValueError(
            "Grouped equal-weight approximate jackknife requires every frozen LOO "
            "group; a subset is not the declared delete-group covariance."
        )
    balance = _validate_equal_weight_groups(
        summary.loo_group_ids,
        selected_groups,
        (
            summary.group_variant_counts
            if isinstance(summary, GroupedContextTraitSummary)
            else None
        ),
    )
    loo_coefficients = np.empty(
        (len(selected_groups), coefficients.size), dtype=np.float64
    )
    loo_equations: list[ContextNormalEquations] = []
    phase_start = time.perf_counter()
    for index, group in enumerate(selected_groups):
        replicate_equations = assemble_context_normal_equations(
            reference, summary, (group,)
        )
        try:
            replicate_solve = solve_context_normal_equations(
                replicate_equations,
                rtol=relative_tolerance,
                require_full_rank=True,
            )
        except ContextRankError as exc:
            raise ContextJackknifeError(group, exc) from exc
        loo_coefficients[index] = replicate_solve.coefficients
        loo_equations.append(replicate_equations)
    phase_times["approximate_jackknife"] = time.perf_counter() - phase_start
    jackknife_mean = np.mean(loo_coefficients, axis=0)
    centered = loo_coefficients - jackknife_mean[None, :]
    jackknife_covariance = (
        (len(selected_groups) - 1.0) / len(selected_groups) * (centered.T @ centered)
    )
    jackknife_covariance = 0.5 * (jackknife_covariance + jackknife_covariance.T)
    covariance_eigenvalues = np.linalg.eigvalsh(jackknife_covariance)
    covariance_scale = max(
        float(np.max(np.abs(jackknife_covariance), initial=0.0)), 1.0
    )
    if covariance_eigenvalues[0] < -1.0e-10 * covariance_scale:
        raise RuntimeError("Computed jackknife covariance is materially indefinite.")
    diagonal = np.diag(jackknife_covariance)
    if np.min(diagonal) < -1.0e-10 * covariance_scale:
        raise RuntimeError("Computed jackknife covariance has a negative variance.")
    standard_errors = np.sqrt(np.maximum(diagonal, 0.0))
    peak_rss = max(peak_rss, process.memory_info().rss)

    if reference.manifest.get("annotation_mode") == "disjoint_partition":
        inferred_disjoint = True
    elif isinstance(reference, GroupedContextReference):
        inferred_disjoint = False
    else:
        weights = reference.annotation_weights
        inferred_disjoint = bool(
            np.all((weights == 0.0) | (weights == 1.0))
            and np.all(np.sum(weights, axis=1) == 1.0)
        )
    if annotations_disjoint is None:
        annotations_disjoint_value = inferred_disjoint
    else:
        annotations_disjoint_value = bool(annotations_disjoint)
        if annotations_disjoint_value and not inferred_disjoint:
            raise ValueError(
                "annotations_disjoint=True conflicts with overlapping reference weights."
            )
    phase_start = time.perf_counter()
    projection: PSDProjectionResult | None = None
    if project_psd:
        projection = project_genetic_coefficients_psd(
            genetic,
            jackknife_covariance[:p_genetic, :p_genetic],
            summary.component_index,
            annotations_disjoint=annotations_disjoint_value,
        )
    phase_times["optional_psd_projection"] = time.perf_counter() - phase_start

    context_output: dict[str, Any] | None = None
    jackknife_context_outputs: tuple[dict[str, Any], ...] = ()
    if context_grid is not None:
        if basis_metric is None:
            basis_metric_value = np.asarray(
                reference.manifest["context_moments"]["second_moment"],
                dtype=np.float64,
            )
        else:
            basis_metric_value = _finite_float64("basis_metric", basis_metric, ndim=2)
        phase_start = time.perf_counter()
        context_output = derive_context_outputs(
            genetic,
            summary.component_index,
            context_grid,
            basis_metric_value,
            equations.traces[:p_genetic],
        )
        if projection is not None:
            context_output["psd_interpretable"] = derive_context_outputs(
                projection.projected_coefficients,
                summary.component_index,
                context_grid,
                basis_metric_value,
                equations.traces[:p_genetic],
            )
        jackknife_context_outputs = tuple(
            derive_context_outputs(
                loo_coefficients[index, :p_genetic],
                summary.component_index,
                context_grid,
                basis_metric_value,
                loo_equations[index].traces[:p_genetic],
            )
            for index in range(len(selected_groups))
        )
        phase_times["derived_context_outputs"] = time.perf_counter() - phase_start
    phase_times["total"] = time.perf_counter() - total_start
    peak_rss = max(peak_rss, process.memory_info().rss)

    condition_number = (
        None if not np.isfinite(solve.condition_number) else solve.condition_number
    )
    manifest: dict[str, Any] = {
        "kind": CONTEXT_FIT_KIND,
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "genotype_scaling": summary.manifest["genotype_scaling"],
        "basis_hash": summary.manifest["basis_hash"],
        "basis_array_hashes": {
            "study": summary.manifest.get("basis_array_hash"),
            "reference": reference.manifest.get("basis_array_hash"),
        },
        "residual_basis_hash": summary.manifest.get("residual_basis_hash"),
        "fixed_effect_hash": summary.manifest["fixed_effect_hash"],
        "variant_hash": summary.manifest["variant_hash"],
        "annotation_hash": summary.manifest["annotation_hash"],
        "component_index_hash": summary.manifest["component_index_hash"],
        "loo_grouping_hash": summary.manifest["loo_grouping_hash"],
        "dimensions": {
            "n_samples": summary.n_samples,
            "reference_n": reference.n_samples,
            "residual_rank": summary.residual_rank,
            "n_variants": summary.n_variants,
            "q": summary.component_index.pair_index.num_basis,
            "k": len(summary.component_index.annotation_names),
            "h": len(summary.residual_names),
            "p_genetic": p_genetic,
            "p_total": coefficients.size,
            "jackknife_groups": len(selected_groups),
        },
        "component_order": list(summary.component_index.names),
        "annotation_names": list(summary.component_index.annotation_names),
        "residual_order": list(summary.residual_names),
        "context_moments": {
            "study": summary.manifest.get("context_moments"),
            "reference": reference.manifest.get("context_moments"),
        },
        "compatibility": compatibility,
        "assembly": {
            "ordering": "genetic_annotation_major_pair_minor_then_residual",
            "genetic_transfer": "N_and_N_times_N_minus_1",
            "residual_elimination": False,
        },
        "solve": {
            "method": "full_symmetric_eigendecomposition",
            "relative_tolerance": solve.relative_tolerance,
            "absolute_tolerance": solve.absolute_tolerance,
            "effective_tolerance": solve.diagnostics.tolerance,
            "rank": solve.rank,
            "dimension": coefficients.size,
            "condition_number": condition_number,
            "relative_residual": solve.relative_residual,
            "minimum_gram_eigenvalue": solve.minimum_gram_eigenvalue,
        },
        "approximate_loo": {
            "method": (
                "lossless_declared_group_aggregate_delete_v1"
                if isinstance(summary, GroupedContextTraitSummary)
                else "snp_contribution_delete_group_v1"
            ),
            "contribution_storage": summary.manifest.get("approximate_loo", {}).get(
                "contribution_storage", "snp"
            ),
            "full_reference_same_person_reused": True,
            **balance,
            "covariance": "equal_group_delete_one_joint_coefficients",
        },
        "psd_projection": {
            "requested": bool(project_psd),
            "annotations_disjoint": annotations_disjoint_value,
            "raw_output_replaced": False,
        },
        "backend": {
            "name": "python_numpy_scipy_reference",
            "dtype": "float64",
            "reads_individual_level_inputs": False,
        },
    }
    validate_context_manifest(manifest, expected_kind=CONTEXT_FIT_KIND)
    return ContextFitResult(
        manifest=manifest,
        component_index=summary.component_index,
        residual_names=summary.residual_names,
        equations=equations,
        solve=solve,
        raw_coefficients=coefficients.copy(),
        genetic_coefficients=genetic,
        residual_coefficients=residual,
        raw_omegas=raw_omegas,
        jackknife_groups=selected_groups,
        loo_coefficients=loo_coefficients,
        jackknife_covariance=jackknife_covariance,
        standard_errors=standard_errors,
        psd_projection=projection,
        context_outputs=context_output,
        jackknife_context_outputs=jackknife_context_outputs,
        phase_times_seconds=phase_times,
        peak_rss_bytes=int(peak_rss),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__summit_context_ndarray_v1__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": _jsonify(value.tolist()),
        }
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            label = "nan"
        elif value > 0.0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"__summit_context_nonfinite_float_v1__": label}
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _restore_context_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__summit_context_nonfinite_float_v1__"}:
            label = value["__summit_context_nonfinite_float_v1__"]
            if label == "nan":
                return float("nan")
            if label == "positive_infinity":
                return float("inf")
            if label == "negative_infinity":
                return float("-inf")
            raise ValueError("Context-fit non-finite float tag is invalid.")
        if value.get("__summit_context_ndarray_v1__") is True:
            if set(value) != {
                "__summit_context_ndarray_v1__",
                "dtype",
                "shape",
                "values",
            }:
                raise ValueError("Context-fit ndarray tag has unknown fields.")
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "bifu":
                raise ValueError("Context-fit ndarray dtype is not numeric.")
            shape_values = value["shape"]
            if not isinstance(shape_values, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in shape_values
            ):
                raise ValueError("Context-fit ndarray shape is invalid.")
            restored_values = _restore_context_arrays(value["values"])
            try:
                array = np.asarray(restored_values, dtype=dtype)
                return array.reshape(tuple(shape_values))
            except (TypeError, ValueError) as exc:
                raise ValueError("Context-fit ndarray values are invalid.") from exc
        return {key: _restore_context_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return [_restore_context_arrays(item) for item in value]

        # JSON has no NaN representation; `_jsonify` writes non-finite array
        # entries as null.  Restore rectangular numeric/null trees to float64
        # arrays and map null back to NaN so undefined derived surfaces retain
        # their in-memory semantics across a fit round trip.
        def numeric_or_null_tree(item: Any) -> bool:
            if item is None or isinstance(item, (bool, int, float)):
                return True
            return isinstance(item, list) and all(
                numeric_or_null_tree(child) for child in item
            )

        def replace_null(item: Any) -> Any:
            if item is None:
                return float("nan")
            if isinstance(item, list):
                return [replace_null(child) for child in item]
            return item

        def contains_null(item: Any) -> bool:
            if item is None:
                return True
            return isinstance(item, list) and any(
                contains_null(child) for child in item
            )

        if value and contains_null(value) and numeric_or_null_tree(value):
            try:
                return np.asarray(replace_null(value), dtype=np.float64)
            except ValueError:
                pass
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return [_restore_context_arrays(item) for item in value]
        if array.dtype.kind in "bifu" and array.dtype != object:
            return array
        return [_restore_context_arrays(item) for item in value]
    return value


def write_context_fit(
    fit: ContextFitResult, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Atomically write a versioned JSON+NPZ fit artifact."""
    prefix = Path(output_prefix)
    manifest_path = prefix.with_suffix(".context-fit.json")
    arrays_path = prefix.with_suffix(".context-fit.npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{arrays_path.name}.", dir=arrays_path.parent
    )
    projection = fit.psd_projection
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                normal_matrix=fit.equations.matrix,
                normal_rhs=fit.equations.rhs,
                traces=fit.equations.traces,
                annotation_masses=fit.equations.annotation_masses,
                reference_genetic_gram=fit.equations.reference_genetic_gram,
                transferred_genetic_gram=fit.equations.transferred_genetic_gram,
                raw_coefficients=fit.raw_coefficients,
                genetic_coefficients=fit.genetic_coefficients,
                residual_coefficients=fit.residual_coefficients,
                raw_omegas=fit.raw_omegas,
                loo_coefficients=fit.loo_coefficients,
                jackknife_covariance=fit.jackknife_covariance,
                standard_errors=fit.standard_errors,
                solve_residual=fit.solve.solve_residual,
                solve_eigenvalues=fit.solve.diagnostics.eigenvalues,
                solve_singular_values=fit.solve.diagnostics.singular_values,
                solve_retained_directions=fit.solve.retained_directions,
                solve_null_space=fit.solve.null_space,
                projected_coefficients=(
                    np.empty(0, dtype=np.float64)
                    if projection is None
                    else projection.projected_coefficients
                ),
                projected_omegas=(
                    np.empty((0, 0, 0), dtype=np.float64)
                    if projection is None
                    else projection.projected_omegas
                ),
                projection_minimum_eigenvalues=(
                    np.empty(0, dtype=np.float64)
                    if projection is None
                    else projection.minimum_eigenvalues
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, arrays_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    payload = dict(fit.manifest)
    payload["artifact"] = {
        "path": arrays_path.name,
        "sha256": array_sha256(np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)),
        "format": "npz_development_v1",
    }
    payload["performance"] = {
        "phase_times_seconds": fit.phase_times_seconds,
        "peak_rss_bytes": fit.peak_rss_bytes,
    }
    payload["jackknife_groups"] = list(fit.jackknife_groups)
    payload["context_outputs"] = _jsonify(fit.context_outputs)
    payload["jackknife_context_outputs"] = _jsonify(fit.jackknife_context_outputs)
    payload["projection_result"] = (
        None
        if projection is None
        else {
            "distance": projection.distance,
            "euclidean_distance": projection.euclidean_distance,
            "covariance_rank": projection.covariance_rank,
            "covariance_nullity": projection.covariance_nullity,
            "optimizer_success": projection.optimizer_success,
            "optimizer_message": projection.optimizer_message,
            "tie_break_applied": projection.tie_break_applied,
            "cleanup_norm": projection.cleanup_norm,
        }
    )
    _atomic_write_text(
        manifest_path, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    )
    return manifest_path, arrays_path


def load_context_fit(
    manifest_path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> ContextFitResult:
    path = Path(manifest_path)
    if path.name.endswith(".contextual-fit-v1.npz"):
        raise ValueError(
            "Stable contextual fit V1 artifacts require load_contextual_fit_v1()."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_context_manifest(
        payload, expected_kind=CONTEXT_FIT_KIND, expected=expected
    )
    artifact = payload.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("format") != "npz_development_v1"
    ):
        raise ValueError("Context fit has no supported development artifact.")
    arrays_path = path.parent / str(artifact.get("path"))
    observed_hash = array_sha256(
        np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)
    )
    if observed_hash != artifact.get("sha256"):
        raise ValueError("Context-fit artifact SHA-256 mismatch.")
    dimensions = payload["dimensions"]
    q_count = int(dimensions["q"])
    declared_annotation_names = payload.get("annotation_names")
    if declared_annotation_names is None:
        annotation_names = tuple(
            dict.fromkeys(name.split(":", 2)[1] for name in payload["component_order"])
        )
    else:
        annotation_names = tuple(str(value) for value in declared_annotation_names)
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q_count))
    if components.digest != payload["component_index_hash"]:
        raise ValueError("Context-fit component-index digest mismatch.")
    if list(components.names) != payload["component_order"]:
        raise ValueError("Context-fit component order is not canonical.")
    residual_names = tuple(str(value) for value in payload["residual_order"])
    p_genetic = len(components)
    p_total = p_genetic + len(residual_names)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {name: np.array(arrays[name], copy=True) for name in arrays.files}
    required_shapes = {
        "normal_matrix": (p_total, p_total),
        "normal_rhs": (p_total,),
        "traces": (p_total,),
        "annotation_masses": (len(annotation_names),),
        "reference_genetic_gram": (p_genetic, p_genetic),
        "transferred_genetic_gram": (p_genetic, p_genetic),
        "raw_coefficients": (p_total,),
        "genetic_coefficients": (p_genetic,),
        "residual_coefficients": (len(residual_names),),
        "raw_omegas": (len(annotation_names), q_count, q_count),
        "loo_coefficients": (int(dimensions["jackknife_groups"]), p_total),
        "jackknife_covariance": (p_total, p_total),
        "standard_errors": (p_total,),
        "solve_residual": (p_total,),
        "solve_eigenvalues": (p_total,),
        "solve_singular_values": (p_total,),
        "solve_retained_directions": (p_total, int(payload["solve"]["rank"])),
        "solve_null_space": (
            p_total,
            p_total - int(payload["solve"]["rank"]),
        ),
    }
    for name, shape in required_shapes.items():
        if name not in values or values[name].shape != shape:
            raise ValueError(
                f"Context-fit array {name!r} has invalid shape "
                f"{None if name not in values else values[name].shape}; expected {shape}."
            )
        if values[name].dtype != np.dtype(np.float64):
            raise ValueError(f"Context-fit array {name!r} must be float64.")
        if not np.all(np.isfinite(values[name])):
            raise ValueError(f"Context-fit array {name!r} contains non-finite values.")
    diagnostics = SymmetricRankDiagnostics(
        symmetry_error=float(
            np.max(
                np.abs(values["normal_matrix"] - values["normal_matrix"].T),
                initial=0.0,
            )
        ),
        eigenvalues=values["solve_eigenvalues"],
        singular_values=values["solve_singular_values"],
        rank=int(payload["solve"]["rank"]),
        condition_number=(
            float("inf")
            if payload["solve"]["condition_number"] is None
            else float(payload["solve"]["condition_number"])
        ),
        tolerance=float(payload["solve"]["effective_tolerance"]),
        null_space=values["solve_null_space"],
    )
    solve = ContextSolveResult(
        coefficients=values["raw_coefficients"],
        diagnostics=diagnostics,
        rank=diagnostics.rank,
        condition_number=diagnostics.condition_number,
        solve_residual=values["solve_residual"],
        relative_residual=float(payload["solve"]["relative_residual"]),
        minimum_gram_eigenvalue=float(payload["solve"]["minimum_gram_eigenvalue"]),
        retained_directions=values["solve_retained_directions"],
        null_space=values["solve_null_space"],
        relative_tolerance=float(payload["solve"]["relative_tolerance"]),
        absolute_tolerance=float(payload["solve"]["absolute_tolerance"]),
    )
    equations = ContextNormalEquations(
        matrix=values["normal_matrix"],
        rhs=values["normal_rhs"],
        traces=values["traces"],
        component_names=components.names + residual_names,
        genetic_count=p_genetic,
        annotation_masses=values["annotation_masses"],
        deleted_groups=(),
        reference_genetic_gram=values["reference_genetic_gram"],
        transferred_genetic_gram=values["transferred_genetic_gram"],
        reference_n=int(dimensions["reference_n"]),
        study_n=int(dimensions["n_samples"]),
    )
    projection_payload = payload.get("projection_result")
    projection: PSDProjectionResult | None = None
    if projection_payload is not None:
        if not isinstance(projection_payload, Mapping):
            raise ValueError("Context-fit projection result is invalid.")
        projection = PSDProjectionResult(
            projected_coefficients=values["projected_coefficients"],
            projected_omegas=values["projected_omegas"],
            distance=float(projection_payload["distance"]),
            euclidean_distance=float(projection_payload["euclidean_distance"]),
            covariance_rank=int(projection_payload["covariance_rank"]),
            covariance_nullity=int(projection_payload["covariance_nullity"]),
            minimum_eigenvalues=values["projection_minimum_eigenvalues"],
            optimizer_success=bool(projection_payload["optimizer_success"]),
            optimizer_message=str(projection_payload["optimizer_message"]),
            tie_break_applied=bool(projection_payload["tie_break_applied"]),
            cleanup_norm=float(projection_payload["cleanup_norm"]),
        )
    excluded = {
        "artifact",
        "performance",
        "jackknife_groups",
        "context_outputs",
        "jackknife_context_outputs",
        "projection_result",
    }
    performance = payload.get("performance", {})
    restored_context = (
        None
        if payload.get("context_outputs") is None
        else _restore_context_arrays(payload["context_outputs"])
    )
    restored_jackknife_context = tuple(
        _restore_context_arrays(value)
        for value in payload.get("jackknife_context_outputs", [])
    )
    return ContextFitResult(
        manifest={key: value for key, value in payload.items() if key not in excluded},
        component_index=components,
        residual_names=residual_names,
        equations=equations,
        solve=solve,
        raw_coefficients=values["raw_coefficients"],
        genetic_coefficients=values["genetic_coefficients"],
        residual_coefficients=values["residual_coefficients"],
        raw_omegas=values["raw_omegas"],
        jackknife_groups=tuple(str(value) for value in payload["jackknife_groups"]),
        loo_coefficients=values["loo_coefficients"],
        jackknife_covariance=values["jackknife_covariance"],
        standard_errors=values["standard_errors"],
        psd_projection=projection,
        context_outputs=restored_context,
        jackknife_context_outputs=restored_jackknife_context,
        phase_times_seconds=dict(performance.get("phase_times_seconds", {})),
        peak_rss_bytes=int(performance.get("peak_rss_bytes", 0)),
    )
