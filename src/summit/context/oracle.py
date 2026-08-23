"""Dense and small-matrix oracles for unified contextual covariance.

All functions in this module are correctness references.  They intentionally
materialize dense projectors and kernels and must not be used for production
genome-wide workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .spec import ContextComponentIndex, owned_readonly_array


@dataclass(frozen=True)
class ProjectorResult:
    projector: np.ndarray
    fixed_basis: np.ndarray
    rank: int
    residual_rank: int
    singular_values: np.ndarray
    tolerance: float
    maximum_leverage: float

    def __post_init__(self) -> None:
        for name in ("projector", "fixed_basis", "singular_values"):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class ResidualMoments:
    rhs: np.ndarray
    traces: np.ndarray
    gram: np.ndarray

    def __post_init__(self) -> None:
        for name in ("rhs", "traces", "gram"):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class DenseNormalEquations:
    matrix: np.ndarray
    rhs: np.ndarray
    traces: np.ndarray
    component_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_names", tuple(self.component_names))
        for name in ("matrix", "rhs", "traces"):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class SymmetricRankDiagnostics:
    symmetry_error: float
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    tolerance: float
    null_space: np.ndarray

    def __post_init__(self) -> None:
        for name in ("eigenvalues", "singular_values", "null_space"):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


def _finite_float64(name: str, value: object, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def rank_revealing_projector(
    fixed_effects: object, *, rtol: float | None = None
) -> ProjectorResult:
    """Build ``P=I-UU'`` by a deterministic rank-revealing SVD."""
    design = _finite_float64("fixed_effects", fixed_effects, ndim=2)
    n_samples, n_columns = design.shape
    if n_samples < 2:
        raise ValueError("At least two samples are required.")
    if n_columns == 0:
        singular = np.empty(0, dtype=np.float64)
        basis = np.empty((n_samples, 0), dtype=np.float64)
        tolerance = 0.0
    else:
        u, singular, _ = np.linalg.svd(design, full_matrices=False)
        scale = float(singular[0]) if singular.size else 0.0
        if rtol is None:
            tolerance = max(design.shape) * np.finfo(np.float64).eps * scale
        else:
            rtol = float(rtol)
            if not np.isfinite(rtol) or rtol <= 0.0:
                raise ValueError("rtol must be finite and positive.")
            tolerance = rtol * scale
        rank = int(np.sum(singular > tolerance))
        basis = np.ascontiguousarray(u[:, :rank], dtype=np.float64)
    rank = basis.shape[1]
    projector = np.eye(n_samples, dtype=np.float64) - basis @ basis.T
    projector = 0.5 * (projector + projector.T)
    leverage = np.sum(basis * basis, axis=1) if rank else np.zeros(n_samples)
    return ProjectorResult(
        projector=projector,
        fixed_basis=basis,
        rank=rank,
        residual_rank=n_samples - rank,
        singular_values=singular,
        tolerance=float(tolerance),
        maximum_leverage=float(np.max(leverage, initial=0.0)),
    )


def project_normalize_phenotype(
    phenotype: object, projector: ProjectorResult
) -> np.ndarray:
    raw = _finite_float64("phenotype", phenotype, ndim=1)
    if raw.shape[0] != projector.projector.shape[0]:
        raise ValueError("Phenotype and projector sample counts differ.")
    if projector.residual_rank < 1:
        raise ValueError(
            "The fixed-effect design leaves no residual degrees of freedom."
        )
    projected = projector.projector @ raw
    sum_squares = float(projected @ projected)
    # A relative threshold preserves the declared raw-scale estimand under a
    # change of phenotype units.  Zero input still fails because both sides
    # are exactly zero.
    minimum = np.finfo(np.float64).eps * float(raw @ raw)
    if not np.isfinite(sum_squares) or sum_squares <= minimum:
        raise ValueError("Phenotype has zero or invalid residual variance.")
    normalized = projected * np.sqrt(projector.residual_rank / sum_squares)
    return np.asarray(normalized, dtype=np.float64)


def common_scale_features(
    genotype: object, basis: object, projector: object
) -> np.ndarray:
    """Return a ``(Q,N,M)`` array of ``P diag(phi_q) G`` features."""
    genotype_array = _finite_float64("genotype", genotype, ndim=2)
    basis_array = _finite_float64("basis", basis, ndim=2)
    projector_array = _finite_float64("projector", projector, ndim=2)
    n_samples, _ = genotype_array.shape
    if basis_array.shape[0] != n_samples:
        raise ValueError("Basis and genotype sample counts differ.")
    if projector_array.shape != (n_samples, n_samples):
        raise ValueError("Projector shape is incompatible with genotype samples.")
    result = np.empty(
        (basis_array.shape[1], n_samples, genotype_array.shape[1]),
        dtype=np.float64,
    )
    for q in range(basis_array.shape[1]):
        result[q] = projector_array @ (basis_array[:, q, None] * genotype_array)
    return result


def validate_annotations(
    annotations: object, n_variants: int, n_annotations: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    weights = _finite_float64("annotations", annotations, ndim=2)
    if weights.shape[0] != n_variants:
        raise ValueError("Annotation and genotype variant counts differ.")
    if n_annotations is not None and weights.shape[1] != n_annotations:
        raise ValueError("Annotation count does not match the component index.")
    if np.any(weights < 0.0):
        raise ValueError("Annotation weights must be non-negative.")
    masses = np.sum(weights, axis=0, dtype=np.float64)
    if np.any(masses <= 0.0):
        raise ValueError("Every annotation must have positive mass.")
    return np.ascontiguousarray(weights), masses


def dense_genetic_kernels(
    features: object,
    annotations: object,
    components: ContextComponentIndex,
) -> np.ndarray:
    features_array = _finite_float64("features", features, ndim=3)
    q_count, n_samples, n_variants = features_array.shape
    if q_count != components.pair_index.num_basis:
        raise ValueError("Feature basis dimension does not match component index.")
    weights, masses = validate_annotations(
        annotations, n_variants, len(components.annotation_names)
    )
    kernels = np.empty((len(components), n_samples, n_samples), dtype=np.float64)
    for component in components.entries:
        weight = weights[:, component.annotation_index]
        left = features_array[component.q] * weight[None, :]
        if component.q == component.r:
            kernel = left @ features_array[component.q].T
        else:
            kernel = (
                left @ features_array[component.r].T
                + (features_array[component.r] * weight[None, :])
                @ features_array[component.q].T
            )
        kernels[component.index] = kernel / masses[component.annotation_index]
    return kernels


def dense_residual_kernels(projector: object, residual_basis: object) -> np.ndarray:
    projector_array = _finite_float64("projector", projector, ndim=2)
    basis = _finite_float64("residual_basis", residual_basis, ndim=2)
    n_samples = projector_array.shape[0]
    if projector_array.shape[1] != n_samples or basis.shape[0] != n_samples:
        raise ValueError("Residual basis and projector shapes are incompatible.")
    kernels = np.empty((basis.shape[1], n_samples, n_samples), dtype=np.float64)
    for h in range(basis.shape[1]):
        kernels[h] = (projector_array * basis[:, h][None, :]) @ projector_array
    return kernels


def kernel_gram(kernels: object) -> np.ndarray:
    array = _finite_float64("kernels", kernels, ndim=3)
    return np.einsum("aij,bij->ab", array, array, optimize=True)


def kernel_rhs(kernels: object, phenotype: object) -> np.ndarray:
    array = _finite_float64("kernels", kernels, ndim=3)
    y = _finite_float64("phenotype", phenotype, ndim=1)
    if array.shape[1:] != (y.size, y.size):
        raise ValueError("Kernel and phenotype sample dimensions differ.")
    return np.einsum("i,aij,j->a", y, array, y, optimize=True)


def kernel_traces(kernels: object) -> np.ndarray:
    array = _finite_float64("kernels", kernels, ndim=3)
    return np.trace(array, axis1=1, axis2=2)


def dense_normal_equations(
    genetic_kernels: object,
    residual_kernels: object,
    phenotype: object,
    genetic_names: Sequence[str],
    residual_names: Sequence[str],
) -> DenseNormalEquations:
    genetic = _finite_float64("genetic_kernels", genetic_kernels, ndim=3)
    residual = _finite_float64("residual_kernels", residual_kernels, ndim=3)
    if genetic.shape[1:] != residual.shape[1:]:
        raise ValueError("Genetic and residual kernels use different samples.")
    if (
        len(genetic_names) != genetic.shape[0]
        or len(residual_names) != residual.shape[0]
    ):
        raise ValueError("Kernel names do not match kernel counts.")
    all_kernels = np.concatenate([genetic, residual], axis=0)
    return DenseNormalEquations(
        matrix=kernel_gram(all_kernels),
        rhs=kernel_rhs(all_kernels, phenotype),
        traces=kernel_traces(all_kernels),
        component_names=tuple(genetic_names) + tuple(residual_names),
    )


def contextual_scores(
    features: object, phenotype: object, residual_rank: int
) -> np.ndarray:
    feature_array = _finite_float64("features", features, ndim=3)
    y = _finite_float64("phenotype", phenotype, ndim=1)
    if feature_array.shape[1] != y.size:
        raise ValueError("Feature and phenotype sample counts differ.")
    if residual_rank < 1:
        raise ValueError("residual_rank must be positive.")
    return np.einsum("qnm,n->qm", feature_array, y, optimize=True) / np.sqrt(
        residual_rank
    )


def genetic_rhs_from_scores(
    scores: object,
    annotations: object,
    components: ContextComponentIndex,
    residual_rank: int,
) -> np.ndarray:
    score_array = _finite_float64("scores", scores, ndim=2)
    if score_array.shape[0] != components.pair_index.num_basis:
        raise ValueError("Score basis dimension does not match component index.")
    weights, masses = validate_annotations(
        annotations, score_array.shape[1], len(components.annotation_names)
    )
    rhs = np.empty(len(components), dtype=np.float64)
    for component in components.entries:
        products = score_array[component.q] * score_array[component.r]
        numerator = (
            residual_rank
            * component.kernel_factor
            * np.dot(weights[:, component.annotation_index], products)
        )
        rhs[component.index] = numerator / masses[component.annotation_index]
    return rhs


def genetic_trace_from_features(
    features: object, annotations: object, components: ContextComponentIndex
) -> np.ndarray:
    feature_array = _finite_float64("features", features, ndim=3)
    weights, masses = validate_annotations(
        annotations, feature_array.shape[2], len(components.annotation_names)
    )
    result = np.empty(len(components), dtype=np.float64)
    for component in components.entries:
        per_variant = np.einsum(
            "nm,nm->m",
            feature_array[component.q],
            feature_array[component.r],
            optimize=True,
        )
        result[component.index] = (
            component.kernel_factor
            * np.dot(weights[:, component.annotation_index], per_variant)
            / masses[component.annotation_index]
        )
    return result


def genetic_residual_cross_traces(
    features: object,
    residual_basis: object,
    annotations: object,
    components: ContextComponentIndex,
) -> np.ndarray:
    feature_array = _finite_float64("features", features, ndim=3)
    basis = _finite_float64("residual_basis", residual_basis, ndim=2)
    if basis.shape[0] != feature_array.shape[1]:
        raise ValueError("Residual basis and features use different samples.")
    weights, masses = validate_annotations(
        annotations, feature_array.shape[2], len(components.annotation_names)
    )
    result = np.empty((len(components), basis.shape[1]), dtype=np.float64)
    for component in components.entries:
        weight = weights[:, component.annotation_index]
        for h in range(basis.shape[1]):
            per_variant = np.einsum(
                "nm,n,nm->m",
                feature_array[component.q],
                basis[:, h],
                feature_array[component.r],
                optimize=True,
            )
            result[component.index, h] = (
                component.kernel_factor
                * np.dot(weight, per_variant)
                / masses[component.annotation_index]
            )
    return result


def residual_moments_low_rank(
    fixed_basis: object, residual_basis: object, phenotype: object
) -> ResidualMoments:
    basis_u = _finite_float64("fixed_basis", fixed_basis, ndim=2)
    residual = _finite_float64("residual_basis", residual_basis, ndim=2)
    y = _finite_float64("phenotype", phenotype, ndim=1)
    n_samples = y.size
    if basis_u.shape[0] != n_samples or residual.shape[0] != n_samples:
        raise ValueError("Residual-moment inputs use different sample counts.")
    h_count = residual.shape[1]
    rhs = residual.T @ (y * y)
    traces = np.empty(h_count, dtype=np.float64)
    gram = np.empty((h_count, h_count), dtype=np.float64)
    compressed: list[np.ndarray] = []
    for h in range(h_count):
        d_h = residual[:, h]
        udu = basis_u.T @ (d_h[:, None] * basis_u)
        compressed.append(udu)
        traces[h] = np.sum(d_h, dtype=np.float64) - np.trace(udu)
    leverage = np.sum(basis_u * basis_u, axis=1)
    for h in range(h_count):
        for ell in range(h, h_count):
            product = residual[:, h] * residual[:, ell]
            value = (
                np.sum(product, dtype=np.float64)
                - 2.0 * np.dot(product, leverage)
                + np.trace(compressed[h] @ compressed[ell])
            )
            gram[h, ell] = gram[ell, h] = value
    return ResidualMoments(rhs=rhs, traces=traces, gram=gram)


def exact_same_person_matrix(genetic_kernels: object) -> np.ndarray:
    kernels = _finite_float64("genetic_kernels", genetic_kernels, ndim=3)
    diagonals = np.diagonal(kernels, axis1=1, axis2=2)
    return diagonals @ diagonals.T


def transfer_reference_gram(
    reference_gram: object,
    same_person: object,
    *,
    reference_n: int,
    study_n: int,
) -> np.ndarray:
    gram = _finite_float64("reference_gram", reference_gram, ndim=2)
    diagonal = _finite_float64("same_person", same_person, ndim=2)
    if gram.shape[0] != gram.shape[1] or diagonal.shape != gram.shape:
        raise ValueError(
            "Reference Gram and same-person matrices must be equally square."
        )
    if (
        isinstance(reference_n, bool)
        or not isinstance(reference_n, int)
        or reference_n < 2
    ):
        raise ValueError("reference_n must be an integer >= 2.")
    if isinstance(study_n, bool) or not isinstance(study_n, int) or study_n < 1:
        raise ValueError("study_n must be a positive integer.")
    if reference_n == study_n:
        return gram.copy()
    same_scale = study_n / reference_n
    different_scale = study_n * (study_n - 1) / (reference_n * (reference_n - 1))
    result = same_scale * diagonal + different_scale * (gram - diagonal)
    return 0.5 * (result + result.T)


def context_kernel_actions(
    genotype: object,
    basis: object,
    projector: object,
    annotations: object,
    components: ContextComponentIndex,
    probes: object,
) -> np.ndarray:
    """Apply all contextual kernels without materializing them (Route B oracle)."""
    genotype_array = _finite_float64("genotype", genotype, ndim=2)
    basis_array = _finite_float64("basis", basis, ndim=2)
    projector_array = _finite_float64("projector", projector, ndim=2)
    probe_array = _finite_float64("probes", probes, ndim=2)
    n_samples, n_variants = genotype_array.shape
    if basis_array.shape != (n_samples, components.pair_index.num_basis):
        raise ValueError("Basis shape does not match genotype/component dimensions.")
    if (
        projector_array.shape != (n_samples, n_samples)
        or probe_array.shape[0] != n_samples
    ):
        raise ValueError("Projector/probe shapes are incompatible.")
    weights, masses = validate_annotations(
        annotations, n_variants, len(components.annotation_names)
    )
    projected_probes = projector_array @ probe_array
    source = [
        genotype_array.T @ (basis_array[:, q, None] * projected_probes)
        for q in range(basis_array.shape[1])
    ]
    targets: dict[tuple[int, int], np.ndarray] = {}
    for k in range(weights.shape[1]):
        for q in range(basis_array.shape[1]):
            targets[k, q] = genotype_array @ (weights[:, k, None] * source[q])
    actions = np.empty((len(components), n_samples, probe_array.shape[1]))
    for component in components.entries:
        value = (
            basis_array[:, component.q, None]
            * targets[component.annotation_index, component.r]
        )
        if component.q != component.r:
            value = (
                value
                + basis_array[:, component.r, None]
                * targets[component.annotation_index, component.q]
            )
        actions[component.index] = (
            projector_array @ value / masses[component.annotation_index]
        )
    return actions


def hutchinson_gram(actions: object) -> np.ndarray:
    action_array = _finite_float64("actions", actions, ndim=3)
    if action_array.shape[2] < 1:
        raise ValueError("At least one probe is required.")
    gram = np.einsum("anb,cnb->ac", action_array, action_array, optimize=True)
    gram /= action_array.shape[2]
    return 0.5 * (gram + gram.T)


def coefficients_to_omegas(
    coefficients: object, components: ContextComponentIndex
) -> np.ndarray:
    values = _finite_float64("coefficients", coefficients, ndim=1)
    if values.shape != (len(components),):
        raise ValueError("Coefficient count does not match component index.")
    q_count = components.pair_index.num_basis
    omegas = np.zeros((len(components.annotation_names), q_count, q_count))
    for component in components.entries:
        value = values[component.index]
        omegas[component.annotation_index, component.q, component.r] = value
        omegas[component.annotation_index, component.r, component.q] = value
    return omegas


def omegas_to_coefficients(
    omegas: object, components: ContextComponentIndex
) -> np.ndarray:
    array = _finite_float64("omegas", omegas, ndim=3)
    expected = (
        len(components.annotation_names),
        components.pair_index.num_basis,
        components.pair_index.num_basis,
    )
    if array.shape != expected:
        raise ValueError(f"Omega array has shape {array.shape}; expected {expected}.")
    if not np.allclose(array, np.swapaxes(array, 1, 2), rtol=0.0, atol=1e-12):
        raise ValueError("Every Omega matrix must be symmetric.")
    return np.asarray(
        [array[c.annotation_index, c.q, c.r] for c in components.entries],
        dtype=np.float64,
    )


def combine_genetic_kernels(
    genetic_kernels: object, coefficients: object
) -> np.ndarray:
    kernels = _finite_float64("genetic_kernels", genetic_kernels, ndim=3)
    values = _finite_float64("coefficients", coefficients, ndim=1)
    if kernels.shape[0] != values.size:
        raise ValueError("Kernel and coefficient counts differ.")
    return np.einsum("a,aij->ij", values, kernels, optimize=True)


def transform_omega(omega: object, basis_transform: object) -> np.ndarray:
    """Return ``A^-T Omega A^-1`` for ``phi'=A phi``."""
    matrix = _finite_float64("omega", omega, ndim=2)
    transform = _finite_float64("basis_transform", basis_transform, ndim=2)
    if matrix.shape[0] != matrix.shape[1] or transform.shape != matrix.shape:
        raise ValueError("Omega and basis transform must be equally square.")
    inverse = np.linalg.inv(transform)
    result = inverse.T @ matrix @ inverse
    return 0.5 * (result + result.T)


def context_covariance_surface(omega: object, context_values: object) -> np.ndarray:
    matrix = _finite_float64("omega", omega, ndim=2)
    values = _finite_float64("context_values", context_values, ndim=2)
    if matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("Context values and Omega dimensions differ.")
    return values @ matrix @ values.T


def scale_aware_max_discrepancy(observed: object, expected: object) -> float:
    left = _finite_float64("observed", observed)
    right = _finite_float64("expected", expected)
    if left.shape != right.shape:
        raise ValueError("Discrepancy inputs have different shapes.")
    scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left - right) / scale, initial=0.0))


def symmetric_rank_diagnostics(
    matrix: object, *, rtol: float | None = None
) -> SymmetricRankDiagnostics:
    """Diagnose a small symmetric normal matrix without modifying it."""
    array = _finite_float64("matrix", matrix, ndim=2)
    if array.shape[0] != array.shape[1]:
        raise ValueError("Rank diagnostics require a square matrix.")
    scale = max(float(np.max(np.abs(array), initial=0.0)), 1.0)
    symmetry_error = float(np.max(np.abs(array - array.T), initial=0.0))
    if symmetry_error > 2.0e-10 * scale:
        raise ValueError(f"Matrix has material asymmetry ({symmetry_error:.6g}).")
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    singular_values = np.linalg.svd(symmetric, compute_uv=False)
    leading = float(singular_values[0]) if singular_values.size else 0.0
    if rtol is None:
        tolerance = max(array.shape) * np.finfo(np.float64).eps * leading
    else:
        rtol = float(rtol)
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be finite and positive.")
        tolerance = rtol * leading
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float("inf")
        if rank < array.shape[0] or singular_values[-1] <= 0.0
        else float(singular_values[0] / singular_values[-1])
    )
    null_mask = np.abs(eigenvalues) <= tolerance
    return SymmetricRankDiagnostics(
        symmetry_error=symmetry_error,
        eigenvalues=eigenvalues,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition,
        tolerance=float(tolerance),
        null_space=eigenvectors[:, null_mask],
    )
