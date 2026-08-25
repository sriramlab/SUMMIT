"""Per-variant sufficient statistics for generalized GxE inference.

This module expresses the study-side computation in terms of ordinary linear
cross-products.  The dense implementation is a correctness oracle and a
specification for efficient PLINK2 ``--variant-score`` or native streaming
backends; it does not materialize an N-by-N projector.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Sequence

import numpy as np

from summit.context.spec import ContextComponentIndex, ContextPairIndex
from summit.context.trait_v1 import ContextualTraitMomentsV1


def _finite_matrix(name: str, value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _readonly(value: object, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GeneralizedGxEPerVariantTraitStatistics:
    """Per-SNP score, information, and heteroskedastic information rows."""

    scores: np.ndarray
    information: np.ndarray
    residual_information: np.ndarray
    normalized_phenotypes: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray
    residual_rank: int

    def __post_init__(self) -> None:
        for name in (
            "scores",
            "information",
            "residual_information",
            "normalized_phenotypes",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if self.scores.ndim != 3:
            raise ValueError("scores must have shape M by Q by L")
        m, q, _ = self.scores.shape
        p = q * (q + 1) // 2
        if self.information.shape != (m, p):
            raise ValueError("information must have shape M by Q(Q+1)/2")
        if self.residual_information.ndim != 3 or (
            self.residual_information.shape[:2] != (m, p)
        ):
            raise ValueError(
                "residual_information must have shape M by Q(Q+1)/2 by H"
            )
        h = self.residual_information.shape[2]
        n = self.normalized_phenotypes.shape[0]
        if self.normalized_phenotypes.shape[1] != self.scores.shape[2]:
            raise ValueError("normalized phenotype and score trait axes differ")
        if self.residual_rhs.shape != (h, self.scores.shape[2]):
            raise ValueError("residual_rhs has the wrong shape")
        if self.residual_traces.shape != (h,) or self.residual_gram.shape != (h, h):
            raise ValueError("residual trace/Gram arrays have the wrong shape")
        if (
            isinstance(self.residual_rank, bool)
            or not isinstance(self.residual_rank, int)
            or self.residual_rank < 1
            or self.residual_rank > n
        ):
            raise ValueError("residual_rank is invalid")
        for name in (
            "scores",
            "information",
            "residual_information",
            "normalized_phenotypes",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
        ):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains nonfinite values")

    @property
    def n_variants(self) -> int:
        return self.scores.shape[0]

    @property
    def n_basis(self) -> int:
        return self.scores.shape[1]

    @property
    def n_traits(self) -> int:
        return self.scores.shape[2]

    @property
    def n_residual(self) -> int:
        return self.residual_information.shape[2]


@dataclass(frozen=True)
class GeneralizedGxETraitSummary:
    """Compact full/block study moments with directly checked numeric axes."""

    n_samples: int
    n_variants: int
    residual_rank: int
    component_index: ContextComponentIndex
    group_ids: tuple[str, ...]
    trait_ids: tuple[str, ...]
    residual_names: tuple[str, ...]
    genetic_rhs: np.ndarray
    genetic_traces: np.ndarray
    genetic_residual: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray
    group_rhs_unnormalized_num: np.ndarray
    group_trace_unnormalized_num: np.ndarray
    group_genetic_residual_num: np.ndarray
    annotation_masses: np.ndarray
    group_annotation_masses: np.ndarray
    group_variant_counts: np.ndarray
    per_variant: GeneralizedGxEPerVariantTraitStatistics | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "trait_ids", tuple(self.trait_ids))
        object.__setattr__(self, "residual_names", tuple(self.residual_names))
        if not isinstance(self.component_index, ContextComponentIndex):
            raise TypeError("component_index must be a ContextComponentIndex")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("group IDs must be unique")
        if len(set(self.trait_ids)) != len(self.trait_ids):
            raise ValueError("trait IDs must be unique")
        if len(set(self.residual_names)) != len(self.residual_names):
            raise ValueError("residual names must be unique")
        float_arrays = (
            "genetic_rhs",
            "genetic_traces",
            "genetic_residual",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
            "group_rhs_unnormalized_num",
            "group_trace_unnormalized_num",
            "group_genetic_residual_num",
            "annotation_masses",
            "group_annotation_masses",
        )
        for name in float_arrays:
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(
            self,
            "group_variant_counts",
            _readonly(self.group_variant_counts, dtype=np.int64),
        )
        c = len(self.component_index)
        k = len(self.component_index.annotation_names)
        j = len(self.group_ids)
        l = len(self.trait_ids)
        h = len(self.residual_names)
        expected = {
            "genetic_rhs": (c, l),
            "genetic_traces": (c,),
            "genetic_residual": (c, h),
            "residual_rhs": (h, l),
            "residual_traces": (h,),
            "residual_gram": (h, h),
            "group_rhs_unnormalized_num": (j, c, l),
            "group_trace_unnormalized_num": (j, c),
            "group_genetic_residual_num": (j, c, h),
            "annotation_masses": (k,),
            "group_annotation_masses": (j, k),
            "group_variant_counts": (j,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if any(not np.all(np.isfinite(getattr(self, name))) for name in float_arrays):
            raise ValueError("trait summary contains nonfinite values")
        if np.any(self.annotation_masses <= 0.0) or np.any(
            self.group_variant_counts <= 0
        ):
            raise ValueError("annotation masses and group counts must be positive")
        if int(np.sum(self.group_variant_counts)) != self.n_variants:
            raise ValueError("group counts do not reconstruct the variant count")
        if not np.allclose(
            np.sum(self.group_annotation_masses, axis=0),
            self.annotation_masses,
            rtol=1.0e-13,
            atol=1.0e-13,
        ):
            raise ValueError("group masses do not reconstruct annotation masses")
        if self.per_variant is not None and (
            self.per_variant.n_variants != self.n_variants
            or self.per_variant.n_traits != l
            or self.per_variant.n_residual != h
        ):
            raise ValueError("per-variant statistics disagree with compact axes")

    @property
    def n_traits(self) -> int:
        return len(self.trait_ids)

    def trait_index(self, selector: str | int) -> int:
        if isinstance(selector, bool):
            raise ValueError("trait selector must be a name or integer index")
        if isinstance(selector, int):
            if selector < 0 or selector >= self.n_traits:
                raise ValueError("trait index is out of range")
            return selector
        if isinstance(selector, str):
            try:
                return self.trait_ids.index(selector)
            except ValueError as exc:
                raise ValueError(f"unknown trait ID {selector!r}") from exc
        raise ValueError("trait selector must be a name or integer index")

    @property
    def full_moments(self) -> ContextualTraitMomentsV1:
        return ContextualTraitMomentsV1(
            annotation_masses=self.annotation_masses,
            genetic_rhs=self.genetic_rhs,
            genetic_traces=self.genetic_traces,
            genetic_residual=self.genetic_residual,
            residual_rhs=self.residual_rhs,
            residual_traces=self.residual_traces,
            residual_gram=self.residual_gram,
        )

    def verify(self) -> None:
        # Construction performs all direct structural and numerical checks.
        return None


def generalized_gxe_per_variant_trait_statistics(
    *,
    genotype: object,
    basis: object,
    fixed_basis: object,
    phenotypes: object,
    residual_basis: object,
) -> GeneralizedGxEPerVariantTraitStatistics:
    """Compute exact per-SNP statistics through low-rank cross-products.

    This is algebraically identical to explicitly constructing
    ``F_q = (I-UU.T) diag(phi_q) G``, but it never constructs the projector or
    a projected N-by-M feature panel.
    """
    g = _finite_matrix("genotype", genotype)
    phi = _finite_matrix("basis", basis)
    u = _finite_matrix("fixed_basis", fixed_basis)
    y_raw = np.asarray(phenotypes, dtype=np.float64)
    if y_raw.ndim == 1:
        y_raw = y_raw[:, None]
    if y_raw.ndim != 2 or not np.all(np.isfinite(y_raw)):
        raise ValueError("phenotypes must be a finite vector or matrix")
    d = _finite_matrix("residual_basis", residual_basis)
    n, m = g.shape
    if any(value.shape[0] != n for value in (phi, u, y_raw, d)):
        raise ValueError("all inputs must use the same sample axis")
    if u.shape[1] >= n:
        raise ValueError("fixed-effect rank must be smaller than N")
    orthogonality = np.max(
        np.abs(u.T @ u - np.eye(u.shape[1])), initial=0.0
    )
    if orthogonality > 1.0e-10:
        raise ValueError("fixed_basis columns must be orthonormal")

    residual_rank = n - u.shape[1]
    y = y_raw - u @ (u.T @ y_raw)
    sums = np.sum(y * y, axis=0, dtype=np.float64)
    if np.any(sums <= np.finfo(np.float64).eps * n):
        raise ValueError("phenotype has zero residual variance")
    y *= np.sqrt(residual_rank / sums)[None, :]

    q_count = phi.shape[1]
    pair_index = ContextPairIndex(q_count)
    packed_score_weights = (phi[:, :, None] * y[:, None, :]).reshape(
        n, q_count * y.shape[1]
    )
    scores = (g.T @ packed_score_weights).reshape(m, q_count, y.shape[1])

    fixed_cross = [g.T @ (phi[:, q, None] * u) for q in range(q_count)]
    squared = g * g
    information = np.empty((m, len(pair_index)), dtype=np.float64)
    residual_information = np.empty(
        (m, len(pair_index), d.shape[1]), dtype=np.float64
    )
    residual_fixed_cross = [
        [g.T @ ((d[:, h] * phi[:, q])[:, None] * u) for q in range(q_count)]
        for h in range(d.shape[1])
    ]
    compressed_residual = [u.T @ (d[:, h, None] * u) for h in range(d.shape[1])]

    for pair in pair_index.entries:
        q, r = pair.q, pair.r
        information[:, pair.index] = (
            squared.T @ (phi[:, q] * phi[:, r])
            - np.einsum("mp,mp->m", fixed_cross[q], fixed_cross[r])
        )
        for h in range(d.shape[1]):
            left = fixed_cross[q]
            right = fixed_cross[r]
            weighted_left = residual_fixed_cross[h][q]
            weighted_right = residual_fixed_cross[h][r]
            residual_information[:, pair.index, h] = (
                squared.T @ (d[:, h] * phi[:, q] * phi[:, r])
                - np.einsum("mp,mp->m", left, weighted_right)
                - np.einsum("mp,mp->m", weighted_left, right)
                + np.einsum(
                    "mp,pv,mv->m",
                    left,
                    compressed_residual[h],
                    right,
                    optimize=True,
                )
            )

    leverage = np.sum(u * u, axis=1)
    residual_rhs = d.T @ (y * y)
    residual_traces = np.asarray(
        [np.sum(d[:, h]) - np.trace(compressed_residual[h]) for h in range(d.shape[1])]
    )
    residual_gram = np.empty((d.shape[1], d.shape[1]), dtype=np.float64)
    for h in range(d.shape[1]):
        for ell in range(h, d.shape[1]):
            product = d[:, h] * d[:, ell]
            value = (
                np.sum(product)
                - 2.0 * np.dot(product, leverage)
                + np.trace(compressed_residual[h] @ compressed_residual[ell])
            )
            residual_gram[h, ell] = residual_gram[ell, h] = value

    return GeneralizedGxEPerVariantTraitStatistics(
        scores=scores,
        information=information,
        residual_information=residual_information,
        normalized_phenotypes=y,
        residual_rhs=residual_rhs,
        residual_traces=residual_traces,
        residual_gram=residual_gram,
        residual_rank=residual_rank,
    )


def aggregate_generalized_gxe_trait_statistics(
    statistics: GeneralizedGxEPerVariantTraitStatistics,
    *,
    annotations: object,
    annotation_names: Sequence[str],
    variant_group_ids: object,
    group_labels: Sequence[str],
    trait_ids: Sequence[str],
    residual_names: Sequence[str],
    n_samples: int,
    retain_per_variant: bool = True,
) -> GeneralizedGxETraitSummary:
    """Reduce per-SNP rows to full and block numerators in one linear pass."""
    if not isinstance(statistics, GeneralizedGxEPerVariantTraitStatistics):
        raise TypeError("statistics must be generalized per-variant statistics")
    weights = _finite_matrix("annotations", annotations)
    if weights.shape[0] != statistics.n_variants or np.any(weights < 0.0):
        raise ValueError("annotations have the wrong variant axis or negative weights")
    names = tuple(str(value) for value in annotation_names)
    if len(names) != weights.shape[1] or len(set(names)) != len(names):
        raise ValueError("annotation names must uniquely match annotation columns")
    groups = np.asarray(variant_group_ids)
    labels = tuple(str(value) for value in group_labels)
    if groups.shape != (statistics.n_variants,) or groups.dtype.kind not in "iu":
        raise ValueError("variant_group_ids must be an integer vector of length M")
    group_values = groups.astype(np.int64, copy=False)
    if len(labels) < 1 or np.any(group_values < 0) or np.any(group_values >= len(labels)):
        raise ValueError("variant group IDs are outside group_labels")
    if set(group_values.tolist()) != set(range(len(labels))):
        raise ValueError("variant group IDs must be contiguous and nonempty")
    trait_names = tuple(str(value) for value in trait_ids)
    residual_labels = tuple(str(value) for value in residual_names)
    if len(trait_names) != statistics.n_traits:
        raise ValueError("trait IDs do not match score columns")
    if len(residual_labels) != statistics.n_residual:
        raise ValueError("residual names do not match residual-information columns")

    pair_index = ContextPairIndex(statistics.n_basis)
    components = ContextComponentIndex(names, pair_index)
    j_count = len(labels)
    c_count = len(components)
    rhs = np.zeros((j_count, c_count, statistics.n_traits), dtype=np.float64)
    traces = np.zeros((j_count, c_count), dtype=np.float64)
    genetic_residual = np.zeros(
        (j_count, c_count, statistics.n_residual), dtype=np.float64
    )
    group_masses = np.zeros((j_count, weights.shape[1]), dtype=np.float64)
    np.add.at(group_masses, group_values, weights)
    group_counts = np.bincount(group_values, minlength=j_count)

    for component in components.entries:
        annotation = weights[:, component.annotation_index]
        factor = float(component.kernel_factor) * annotation
        score_product = (
            statistics.scores[:, component.q, :]
            * statistics.scores[:, component.r, :]
            * factor[:, None]
        )
        np.add.at(rhs[:, component.index, :], group_values, score_product)
        np.add.at(
            traces[:, component.index],
            group_values,
            factor * statistics.information[:, component.pair_index],
        )
        np.add.at(
            genetic_residual[:, component.index, :],
            group_values,
            factor[:, None]
            * statistics.residual_information[:, component.pair_index, :],
        )

    masses = np.sum(weights, axis=0, dtype=np.float64)
    component_masses = np.asarray(
        [masses[value.annotation_index] for value in components.entries]
    )
    return GeneralizedGxETraitSummary(
        n_samples=int(n_samples),
        n_variants=statistics.n_variants,
        residual_rank=statistics.residual_rank,
        component_index=components,
        group_ids=labels,
        trait_ids=trait_names,
        residual_names=residual_labels,
        genetic_rhs=np.sum(rhs, axis=0) / component_masses[:, None],
        genetic_traces=np.sum(traces, axis=0) / component_masses,
        genetic_residual=(
            np.sum(genetic_residual, axis=0) / component_masses[:, None]
        ),
        residual_rhs=statistics.residual_rhs,
        residual_traces=statistics.residual_traces,
        residual_gram=statistics.residual_gram,
        group_rhs_unnormalized_num=rhs,
        group_trace_unnormalized_num=traces,
        group_genetic_residual_num=genetic_residual,
        annotation_masses=masses,
        group_annotation_masses=group_masses,
        group_variant_counts=group_counts,
        per_variant=statistics if retain_per_variant else None,
    )


def _compressed_column_span(
    values: np.ndarray,
    *,
    relative_tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``basis, coefficients`` with ``values = basis @ coefficients``."""
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("column-span input must be a nonempty matrix")
    left, singular, right = np.linalg.svd(values, full_matrices=False)
    if singular[0] <= 0.0:
        raise ValueError("column-span input has rank zero")
    if relative_tolerance is None:
        tolerance = (
            8.0
            * np.finfo(np.float64).eps
            * max(values.shape)
            * singular[0]
        )
    else:
        tolerance = float(relative_tolerance) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < 1:
        raise ValueError("column-span tolerance removed every direction")
    basis = np.asfortranarray(left[:, :rank])
    coefficients = np.ascontiguousarray(
        singular[:rank, None] * right[:rank, :]
    )
    reconstruction = np.linalg.norm(values - basis @ coefficients)
    scale = max(np.linalg.norm(values), 1.0)
    if reconstruction > 1.0e-10 * scale:
        raise RuntimeError("column-span compression is not numerically exact")
    return basis, coefficients


def stream_generalized_gxe_per_variant_trait_statistics_from_bed(
    *,
    bed_path: str | Path,
    raw_sample_count: int,
    variant_count: int,
    sample_indices: object,
    affine_mean: object,
    affine_inverse_scale: object,
    basis: object,
    fixed_basis: object,
    phenotypes: object,
    residual_basis: object,
    variant_block_width: int = 256,
    span_relative_tolerance: float | None = None,
) -> tuple[GeneralizedGxEPerVariantTraitStatistics, dict[str, object]]:
    """Compute exact study statistics in one direct BED traversal.

    The implementation decodes existing PLINK1 hardcalls with ``pgenlib``;
    PLINK1 BED is a valid PGEN input.  It computes all per-SNP scores and
    projected information terms from linear ``G @ W`` and squared-genotype
    ``(G**2) @ V`` products.  It has no jackknife or annotation input; the
    completed per-SNP rows are reduced for inference afterward.
    """
    try:
        import pgenlib
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("pgenlib is required for the BED streaming backend") from exc

    source = Path(bed_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if (
        isinstance(raw_sample_count, bool)
        or int(raw_sample_count) < 1
        or isinstance(variant_count, bool)
        or int(variant_count) < 1
    ):
        raise ValueError("raw sample and variant counts must be positive")
    raw_n = int(raw_sample_count)
    m_count = int(variant_count)
    if (
        isinstance(variant_block_width, bool)
        or int(variant_block_width) < 1
    ):
        raise ValueError("variant_block_width must be positive")
    block_width = int(variant_block_width)

    rows = np.asarray(sample_indices)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or rows.size < 2:
        raise ValueError("sample_indices must be a nonempty integer vector")
    rows = np.ascontiguousarray(rows, dtype=np.uint32)
    if np.any(rows[1:] <= rows[:-1]) or int(rows[-1]) >= raw_n:
        raise ValueError("sample indices must be strictly increasing and in range")
    n_count = rows.size

    means = np.asarray(affine_mean, dtype=np.float64)
    inverse = np.asarray(affine_inverse_scale, dtype=np.float64)
    if (
        means.shape != (m_count,)
        or inverse.shape != (m_count,)
        or not np.all(np.isfinite(means))
        or not np.all(np.isfinite(inverse))
        or np.any(inverse <= 0.0)
    ):
        raise ValueError("affine genotype vectors are invalid")

    phi = _finite_matrix("basis", basis)
    fixed = _finite_matrix("fixed_basis", fixed_basis)
    residual = _finite_matrix("residual_basis", residual_basis)
    y_raw = np.asarray(phenotypes, dtype=np.float64)
    if y_raw.ndim == 1:
        y_raw = y_raw[:, None]
    if y_raw.ndim != 2 or not np.all(np.isfinite(y_raw)):
        raise ValueError("phenotypes must be a finite vector or matrix")
    if any(value.shape[0] != n_count for value in (phi, fixed, residual, y_raw)):
        raise ValueError("study matrices do not match the selected sample count")
    if fixed.shape[1] >= n_count:
        raise ValueError("fixed-effect rank must be smaller than N")
    if np.max(
        np.abs(fixed.T @ fixed - np.eye(fixed.shape[1])), initial=0.0
    ) > 1.0e-10:
        raise ValueError("fixed_basis columns must be orthonormal")

    q_count = phi.shape[1]
    h_count = residual.shape[1]
    l_count = y_raw.shape[1]
    fixed_rank = fixed.shape[1]
    residual_rank = n_count - fixed_rank
    pair_index = ContextPairIndex(q_count)

    y = y_raw - fixed @ (fixed.T @ y_raw)
    sums = np.sum(y * y, axis=0, dtype=np.float64)
    if np.any(sums <= np.finfo(np.float64).eps * n_count):
        raise ValueError("phenotype has zero residual variance")
    y *= np.sqrt(residual_rank / sums)[None, :]

    # Candidate t is phi_q, followed by d_h * phi_q.  Compress only the
    # multiplier span; every fixed-effect cross-product is reconstructed.
    fixed_multipliers = np.column_stack(
        [
            *[phi[:, q] for q in range(q_count)],
            *[
                residual[:, h] * phi[:, q]
                for h in range(h_count)
                for q in range(q_count)
            ],
        ]
    )
    fixed_multiplier_basis, fixed_multiplier_coefficients = (
        _compressed_column_span(
            fixed_multipliers,
            relative_tolerance=span_relative_tolerance,
        )
    )
    fixed_weights = np.einsum(
        "nr,np->nrp", fixed_multiplier_basis, fixed, optimize=True
    ).reshape(n_count, -1)
    score_weights = (phi[:, :, None] * y[:, None, :]).reshape(
        n_count, q_count * l_count
    )
    linear_weights = np.asfortranarray(
        np.column_stack((score_weights, fixed_weights))
    )

    square_multipliers = np.column_stack(
        [
            *[
                phi[:, pair.q] * phi[:, pair.r]
                for pair in pair_index.entries
            ],
            *[
                residual[:, h] * phi[:, pair.q] * phi[:, pair.r]
                for h in range(h_count)
                for pair in pair_index.entries
            ],
        ]
    )
    square_basis, square_coefficients = _compressed_column_span(
        square_multipliers,
        relative_tolerance=span_relative_tolerance,
    )
    square_basis = np.asfortranarray(square_basis)

    compressed_residual = np.stack(
        [fixed.T @ (residual[:, h, None] * fixed) for h in range(h_count)]
    )
    leverage = np.sum(fixed * fixed, axis=1)
    residual_rhs = residual.T @ (y * y)
    residual_traces = np.asarray(
        [
            np.sum(residual[:, h]) - np.trace(compressed_residual[h])
            for h in range(h_count)
        ]
    )
    residual_gram = np.empty((h_count, h_count), dtype=np.float64)
    for h in range(h_count):
        for ell in range(h, h_count):
            product = residual[:, h] * residual[:, ell]
            value = (
                np.sum(product)
                - 2.0 * np.dot(product, leverage)
                + np.trace(compressed_residual[h] @ compressed_residual[ell])
            )
            residual_gram[h, ell] = residual_gram[ell, h] = value

    scores = np.empty((m_count, q_count, l_count), dtype=np.float64)
    information = np.empty((m_count, len(pair_index)), dtype=np.float64)
    residual_information = np.empty(
        (m_count, len(pair_index), h_count), dtype=np.float64
    )

    genotype_buffer = np.empty((block_width, n_count), dtype=np.int8)
    timings = {
        "decode_seconds": 0.0,
        "standardize_seconds": 0.0,
        "linear_product_seconds": 0.0,
        "squared_product_seconds": 0.0,
        "assembly_seconds": 0.0,
    }
    missing_calls = 0
    started_all = time.perf_counter()
    with pgenlib.PgenReader(
        os.fsencode(source),
        raw_sample_ct=raw_n,
        variant_ct=m_count,
        sample_subset=rows,
    ) as reader:
        if reader.get_raw_sample_ct() != raw_n or reader.get_variant_ct() != m_count:
            raise ValueError("BED header dimensions disagree with declared axes")
        for start in range(0, m_count, block_width):
            stop = min(start + block_width, m_count)
            width = stop - start
            tick = time.perf_counter()
            reader.read_range(start, stop, genotype_buffer[:width], allele_idx=1)
            timings["decode_seconds"] += time.perf_counter() - tick

            tick = time.perf_counter()
            genotype = genotype_buffer[:width].astype(np.float64)
            missing = genotype == -9.0
            missing_calls += int(np.count_nonzero(missing))
            genotype -= means[start:stop, None]
            genotype[missing] = 0.0
            genotype *= inverse[start:stop, None]
            timings["standardize_seconds"] += time.perf_counter() - tick

            tick = time.perf_counter()
            linear = genotype @ linear_weights
            timings["linear_product_seconds"] += time.perf_counter() - tick
            block_scores = linear[:, : q_count * l_count].reshape(
                width, q_count, l_count
            )
            compressed_fixed = linear[:, q_count * l_count :].reshape(
                width, fixed_multiplier_basis.shape[1], fixed_rank
            )
            fixed_cross = np.einsum(
                "brp,rt->btp",
                compressed_fixed,
                fixed_multiplier_coefficients,
                optimize=True,
            )

            tick = time.perf_counter()
            np.square(genotype, out=genotype)
            compressed_square = genotype @ square_basis
            raw_square = compressed_square @ square_coefficients
            timings["squared_product_seconds"] += time.perf_counter() - tick

            tick = time.perf_counter()
            block_information = np.empty((width, len(pair_index)), dtype=np.float64)
            block_residual_information = np.empty(
                (width, len(pair_index), h_count), dtype=np.float64
            )
            for pair in pair_index.entries:
                q, r = pair.q, pair.r
                left = fixed_cross[:, q, :]
                right = fixed_cross[:, r, :]
                block_information[:, pair.index] = (
                    raw_square[:, pair.index]
                    - np.einsum("bp,bp->b", left, right)
                )
                for h in range(h_count):
                    weighted_left = fixed_cross[:, q_count + h * q_count + q, :]
                    weighted_right = fixed_cross[:, q_count + h * q_count + r, :]
                    block_residual_information[:, pair.index, h] = (
                        raw_square[:, len(pair_index) + h * len(pair_index) + pair.index]
                        - np.einsum("bp,bp->b", left, weighted_right)
                        - np.einsum("bp,bp->b", weighted_left, right)
                        + np.einsum(
                            "bp,pv,bv->b",
                            left,
                            compressed_residual[h],
                            right,
                            optimize=True,
                        )
                    )

            scores[start:stop] = block_scores
            information[start:stop] = block_information
            residual_information[start:stop] = block_residual_information
            timings["assembly_seconds"] += time.perf_counter() - tick

    per_variant = GeneralizedGxEPerVariantTraitStatistics(
        scores=scores,
        information=information,
        residual_information=residual_information,
        normalized_phenotypes=y,
        residual_rhs=residual_rhs,
        residual_traces=residual_traces,
        residual_gram=residual_gram,
        residual_rank=residual_rank,
    )
    report: dict[str, object] = {
        "backend": "pgenlib_direct_bed_linear_and_squared_v1",
        "genotype_passes": 1,
        "variant_blocks": (m_count + block_width - 1) // block_width,
        "variant_block_width": block_width,
        "missing_genotype_calls": missing_calls,
        "linear_weight_columns": linear_weights.shape[1],
        "fixed_multiplier_input_columns": fixed_multipliers.shape[1],
        "fixed_multiplier_rank": fixed_multiplier_basis.shape[1],
        "square_multiplier_input_columns": square_multipliers.shape[1],
        "square_multiplier_rank": square_basis.shape[1],
        "wall_seconds": time.perf_counter() - started_all,
        **timings,
    }
    return per_variant, report


GENERALIZED_GXE_TRAIT_SUMMARY_SUFFIX = ".generalized-gxe-trait-summary-v1.npz"


def write_generalized_gxe_trait_summary(
    summary: GeneralizedGxETraitSummary,
    output: str | Path,
) -> Path:
    """Atomically write compact and optional per-SNP statistics."""
    if not isinstance(summary, GeneralizedGxETraitSummary):
        raise TypeError("summary must be a GeneralizedGxETraitSummary")
    path = Path(output)
    if not path.name.endswith(GENERALIZED_GXE_TRAIT_SUMMARY_SUFFIX):
        path = Path(str(path) + GENERALIZED_GXE_TRAIT_SUMMARY_SUFFIX)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "summit.generalized_gxe.trait_summary",
        "schema_version": 1,
        "n_samples": summary.n_samples,
        "n_variants": summary.n_variants,
        "residual_rank": summary.residual_rank,
        "basis_count": summary.component_index.pair_index.num_basis,
        "annotation_names": list(summary.component_index.annotation_names),
        "group_ids": list(summary.group_ids),
        "trait_ids": list(summary.trait_ids),
        "residual_names": list(summary.residual_names),
        "per_variant_included": summary.per_variant is not None,
    }
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
        "genetic_rhs": summary.genetic_rhs,
        "genetic_traces": summary.genetic_traces,
        "genetic_residual": summary.genetic_residual,
        "residual_rhs": summary.residual_rhs,
        "residual_traces": summary.residual_traces,
        "residual_gram": summary.residual_gram,
        "group_rhs_unnormalized_num": summary.group_rhs_unnormalized_num,
        "group_trace_unnormalized_num": summary.group_trace_unnormalized_num,
        "group_genetic_residual_num": summary.group_genetic_residual_num,
        "annotation_masses": summary.annotation_masses,
        "group_annotation_masses": summary.group_annotation_masses,
        "group_variant_counts": summary.group_variant_counts,
    }
    if summary.per_variant is not None:
        arrays.update(
            {
                "variant_scores": summary.per_variant.scores,
                "variant_information": summary.per_variant.information,
                "variant_residual_information": (
                    summary.per_variant.residual_information
                ),
            }
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
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


def load_generalized_gxe_trait_summary(
    source: str | Path,
) -> GeneralizedGxETraitSummary:
    """Load compact generalized trait moments using direct structural checks."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "metadata_json",
        "genetic_rhs",
        "genetic_traces",
        "genetic_residual",
        "residual_rhs",
        "residual_traces",
        "residual_gram",
        "group_rhs_unnormalized_num",
        "group_trace_unnormalized_num",
        "group_genetic_residual_num",
        "annotation_masses",
        "group_annotation_masses",
        "group_variant_counts",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(
                "generalized trait summary is missing arrays: "
                + ", ".join(missing)
            )
        try:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("generalized trait metadata is malformed") from exc
        if (
            metadata.get("kind") != "summit.generalized_gxe.trait_summary"
            or metadata.get("schema_version") != 1
        ):
            raise ValueError("not a generalized GxE trait-summary V1 artifact")
        arrays = {
            name: np.asarray(archive[name])
            for name in required
            if name != "metadata_json"
        }
    try:
        pair_index = ContextPairIndex(int(metadata["basis_count"]))
        component_index = ContextComponentIndex(
            tuple(str(value) for value in metadata["annotation_names"]),
            pair_index,
        )
        summary = GeneralizedGxETraitSummary(
            n_samples=int(metadata["n_samples"]),
            n_variants=int(metadata["n_variants"]),
            residual_rank=int(metadata["residual_rank"]),
            component_index=component_index,
            group_ids=tuple(str(value) for value in metadata["group_ids"]),
            trait_ids=tuple(str(value) for value in metadata["trait_ids"]),
            residual_names=tuple(
                str(value) for value in metadata["residual_names"]
            ),
            per_variant=None,
            **arrays,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("generalized trait summary structure is invalid") from exc
    return summary


__all__ = [
    "GeneralizedGxEPerVariantTraitStatistics",
    "GeneralizedGxETraitSummary",
    "GENERALIZED_GXE_TRAIT_SUMMARY_SUFFIX",
    "aggregate_generalized_gxe_trait_statistics",
    "generalized_gxe_per_variant_trait_statistics",
    "load_generalized_gxe_trait_summary",
    "stream_generalized_gxe_per_variant_trait_statistics_from_bed",
    "write_generalized_gxe_trait_summary",
]
