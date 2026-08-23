"""Correctness-first contextual reference moments.

This module implements the private Python development backend selected in the
Stage-0 design.  It supports exact dense moments and shared-probe estimates,
and it retains symmetric per-SNP Gram numerators for the accepted approximate
summary-only leave-one-out procedure.  It is intentionally not a production
genotype-reader or native-GEMM implementation.
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

from .oracle import (
    ProjectorResult,
    _finite_float64,
    common_scale_features,
    context_kernel_actions,
    dense_genetic_kernels,
    exact_same_person_matrix,
    kernel_gram,
    transfer_reference_gram,
    validate_annotations,
)
from .spec import (
    CONTEXT_REFERENCE_KIND,
    CONTEXT_SCHEMA_VERSION,
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
    freeze_context_mapping,
    owned_readonly_array,
    validate_context_manifest,
)


@dataclass(frozen=True)
class ReferenceMoments:
    """Reference quantities after optional approximate SNP-group deletion."""

    annotation_masses: np.ndarray
    gram: np.ndarray
    same_person: np.ndarray

    def __post_init__(self) -> None:
        for name in ("annotation_masses", "gram", "same_person"):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))

    def transfer(self, study_n: int, *, reference_n: int) -> np.ndarray:
        return transfer_reference_gram(
            self.gram,
            self.same_person,
            reference_n=reference_n,
            study_n=study_n,
        )


@dataclass(frozen=True)
class ContextReference:
    """Versioned standalone reference object for contextual genetic kernels."""

    manifest: dict[str, Any]
    component_index: ContextComponentIndex
    annotation_weights: np.ndarray
    annotation_masses: np.ndarray
    gram: np.ndarray
    same_person: np.ndarray
    gram_numerator_contributions: np.ndarray
    loo_group_ids: tuple[str, ...]
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        object.__setattr__(self, "loo_group_ids", tuple(self.loo_group_ids))
        object.__setattr__(
            self,
            "phase_times_seconds",
            freeze_context_mapping(self.phase_times_seconds),
        )
        for name in (
            "annotation_weights",
            "annotation_masses",
            "gram",
            "same_person",
            "gram_numerator_contributions",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))

    @property
    def n_samples(self) -> int:
        return int(self.manifest["dimensions"]["n_samples"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["dimensions"]["residual_rank"])

    @property
    def n_variants(self) -> int:
        return int(self.manifest["dimensions"]["n_variants"])

    @property
    def full_moments(self) -> ReferenceMoments:
        return ReferenceMoments(
            annotation_masses=self.annotation_masses.copy(),
            gram=self.gram.copy(),
            same_person=self.same_person.copy(),
        )

    def transferred_gram(self, study_n: int) -> np.ndarray:
        return transfer_reference_gram(
            self.gram,
            self.same_person,
            reference_n=self.n_samples,
            study_n=study_n,
        )


@dataclass(frozen=True)
class GroupedContextReference:
    """Reference moments with lossless frozen-LOO group numerators.

    Unlike :class:`ContextReference`, this object has no variant-length arrays.
    Its contribution axis is the ordered set of declared LOO groups.
    """

    manifest: dict[str, Any]
    component_index: ContextComponentIndex
    annotation_masses: np.ndarray
    gram: np.ndarray
    same_person: np.ndarray
    gram_numerator_contributions: np.ndarray
    loo_group_ids: tuple[str, ...]
    group_annotation_masses: np.ndarray
    group_variant_counts: np.ndarray
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        object.__setattr__(self, "loo_group_ids", tuple(self.loo_group_ids))
        object.__setattr__(
            self,
            "phase_times_seconds",
            freeze_context_mapping(self.phase_times_seconds),
        )
        for name in (
            "annotation_masses",
            "gram",
            "same_person",
            "gram_numerator_contributions",
            "group_annotation_masses",
            "group_variant_counts",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))

    @property
    def n_samples(self) -> int:
        return int(self.manifest["dimensions"]["n_samples"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["dimensions"]["residual_rank"])

    @property
    def n_variants(self) -> int:
        return int(self.manifest["dimensions"]["n_variants"])

    @property
    def group_gram_numerator_contributions(self) -> np.ndarray:
        return self.gram_numerator_contributions

    @property
    def full_moments(self) -> ReferenceMoments:
        return ReferenceMoments(
            annotation_masses=self.annotation_masses.copy(),
            gram=self.gram.copy(),
            same_person=self.same_person.copy(),
        )

    def transferred_gram(self, study_n: int) -> np.ndarray:
        return transfer_reference_gram(
            self.gram,
            self.same_person,
            reference_n=self.n_samples,
            study_n=study_n,
        )


def _validate_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest.") from exc
    return value


def _canonical_loo_groups(
    groups: Sequence[Any] | None, n_variants: int
) -> tuple[str, ...]:
    if groups is None:
        return tuple(f"snp:{index}" for index in range(n_variants))
    if len(groups) != n_variants:
        raise ValueError("Approximate-LOO group count must equal the variant count.")
    result = tuple(str(value) for value in groups)
    if any(not value for value in result):
        raise ValueError("Approximate-LOO group labels must be nonempty.")
    if len(set(result)) < 2:
        raise ValueError("Approximate-LOO inference requires at least two groups.")
    return result


def _group_layout(
    groups: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    labels = tuple(dict.fromkeys(str(value) for value in groups))
    lookup = {label: index for index, label in enumerate(labels)}
    indices = np.fromiter(
        (lookup[str(value)] for value in groups), dtype=np.int64, count=len(groups)
    )
    counts = np.bincount(indices, minlength=len(labels)).astype(np.int64, copy=False)
    return labels, indices, counts


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _rademacher(shape: tuple[int, int], seed: int) -> np.ndarray:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Probe seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(0, 2, size=shape) * 2 - 1, dtype=np.float64)


def _component_masses(
    annotation_masses: np.ndarray, components: ContextComponentIndex
) -> np.ndarray:
    return np.asarray(
        [annotation_masses[entry.annotation_index] for entry in components.entries],
        dtype=np.float64,
    )


def _exact_gram_contributions(
    features: np.ndarray,
    annotations: np.ndarray,
    components: ContextComponentIndex,
    raw_kernels: np.ndarray,
    group_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return variant or frozen-group numerators reconstructing the Gram."""
    n_variants = features.shape[2]
    p_count = len(components)
    contribution_count = (
        n_variants
        if group_indices is None
        else int(np.max(group_indices, initial=-1)) + 1
    )
    directional = np.zeros((contribution_count, p_count, p_count), dtype=np.float64)
    raw_targets = np.empty((p_count, features.shape[0], n_variants), dtype=np.float64)
    for b in range(p_count):
        for q in range(features.shape[0]):
            raw_targets[b, q] = np.einsum(
                "nm,nm->m",
                features[q],
                raw_kernels[b] @ features[q],
                optimize=True,
            )
    # The cached q=q contractions above cover diagonal atoms.  Mixed atoms
    # require their declared q,r directional contraction.
    for component_a in components.entries:
        weight = annotations[:, component_a.annotation_index]
        for component_b in components.entries:
            if component_a.q == component_a.r:
                contraction = raw_targets[component_b.index, component_a.q]
            else:
                contraction = np.einsum(
                    "nm,nm->m",
                    features[component_a.q],
                    raw_kernels[component_b.index] @ features[component_a.r],
                    optimize=True,
                )
                contraction *= component_a.kernel_factor
            value = weight * contraction
            if group_indices is None:
                directional[:, component_a.index, component_b.index] = value
            else:
                np.add.at(
                    directional[:, component_a.index, component_b.index],
                    group_indices,
                    value,
                )
    return 0.5 * (directional + np.swapaxes(directional, 1, 2))


def _hutchinson_gram_and_contributions(
    *,
    genotype: np.ndarray,
    basis: np.ndarray,
    projector: np.ndarray,
    features: np.ndarray,
    annotations: np.ndarray,
    annotation_masses: np.ndarray,
    components: ContextComponentIndex,
    probes: np.ndarray,
    probe_tile_size: int,
    group_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shared-probe Gram plus exactly reconstructing grouped numerators."""
    p_count = len(components)
    n_variants = genotype.shape[1]
    b_count = probes.shape[1]
    gram_raw = np.zeros((p_count, p_count), dtype=np.float64)
    contribution_count = (
        n_variants
        if group_indices is None
        else int(np.max(group_indices, initial=-1)) + 1
    )
    directional = np.zeros((contribution_count, p_count, p_count), dtype=np.float64)
    masses = _component_masses(annotation_masses, components)
    for start in range(0, b_count, probe_tile_size):
        stop = min(start + probe_tile_size, b_count)
        z = probes[:, start:stop]
        actions = context_kernel_actions(
            genotype, basis, projector, annotations, components, z
        )
        gram_raw += np.einsum("anv,bnv->ab", actions, actions, optimize=True)
        raw_actions = actions * masses[:, None, None]
        feature_probe = np.einsum("qnm,nv->qmv", features, z, optimize=True)
        feature_action = np.einsum(
            "qnm,bnv->bqmv", features, raw_actions, optimize=True
        )
        for component_a in components.entries:
            weight = annotations[:, component_a.annotation_index]
            q = component_a.q
            r = component_a.r
            for component_b in components.entries:
                value = np.sum(
                    feature_action[component_b.index, q] * feature_probe[r],
                    axis=1,
                    dtype=np.float64,
                )
                if q != r:
                    value += np.sum(
                        feature_action[component_b.index, r] * feature_probe[q],
                        axis=1,
                        dtype=np.float64,
                    )
                contribution = weight * value
                if group_indices is None:
                    directional[:, component_a.index, component_b.index] += contribution
                else:
                    np.add.at(
                        directional[:, component_a.index, component_b.index],
                        group_indices,
                        contribution,
                    )
    gram_raw /= b_count
    pre_symmetry = float(np.max(np.abs(gram_raw - gram_raw.T), initial=0.0))
    gram = 0.5 * (gram_raw + gram_raw.T)
    contributions = 0.5 * (directional + np.swapaxes(directional, 1, 2)) / b_count
    return gram, contributions, pre_symmetry


def _same_person_ustatistic_with_diagnostics(
    features: np.ndarray,
    annotations: np.ndarray,
    components: ContextComponentIndex,
    variant_probes: np.ndarray,
    probe_tile_size: int,
) -> tuple[np.ndarray, float]:
    b_count = variant_probes.shape[1]
    if b_count < 2:
        raise ValueError("Same-person U-statistic requires at least two probes.")
    annotation_masses = np.sum(annotations, axis=0, dtype=np.float64)
    p_count = len(components)
    probe_sums = np.zeros((p_count, features.shape[1]), dtype=np.float64)
    same_probe = np.zeros((p_count, p_count), dtype=np.float64)
    sqrt_weights = np.sqrt(annotations)
    for start in range(0, b_count, probe_tile_size):
        stop = min(start + probe_tile_size, b_count)
        xi = variant_probes[:, start:stop]
        source: dict[tuple[int, int], np.ndarray] = {}
        for k in range(annotations.shape[1]):
            weighted = sqrt_weights[:, k, None] * xi
            for q in range(features.shape[0]):
                source[k, q] = features[q] @ weighted
        g = np.empty((p_count, features.shape[1], stop - start), dtype=np.float64)
        for component in components.entries:
            left = source[component.annotation_index, component.q]
            right = source[component.annotation_index, component.r]
            g[component.index] = (
                component.kernel_factor
                * left
                * right
                / annotation_masses[component.annotation_index]
            )
        probe_sums += np.sum(g, axis=2, dtype=np.float64)
        same_probe += np.einsum("aiv,biv->ab", g, g, optimize=True)
    raw = (probe_sums @ probe_sums.T - same_probe) / (b_count * (b_count - 1))
    pre_symmetry = float(np.max(np.abs(raw - raw.T), initial=0.0))
    return 0.5 * (raw + raw.T), pre_symmetry


def same_person_ustatistic(
    features: object,
    annotations: object,
    components: ContextComponentIndex,
    variant_probes: object,
    *,
    probe_tile_size: int = 32,
) -> np.ndarray:
    """Estimate the signed same-person matrix with a variant-probe U-statistic."""
    feature_array = _finite_float64("features", features, ndim=3)
    if feature_array.shape[0] != components.pair_index.num_basis:
        raise ValueError("Feature basis dimension does not match component index.")
    weights, _ = validate_annotations(
        annotations, feature_array.shape[2], len(components.annotation_names)
    )
    probes = _finite_float64("variant_probes", variant_probes, ndim=2)
    if probes.shape[0] != feature_array.shape[2]:
        raise ValueError("Variant probes and features use different variant counts.")
    tile = _positive_integer("probe_tile_size", probe_tile_size)
    result, _ = _same_person_ustatistic_with_diagnostics(
        feature_array, weights, components, probes, tile
    )
    return result


def build_context_reference(
    *,
    genotype: object,
    basis: object,
    projector: ProjectorResult,
    annotations: object,
    component_index: ContextComponentIndex,
    basis_hash: str,
    fixed_effect_hash: str,
    variant_hash: str,
    loo_groups: Sequence[Any] | None = None,
    genotype_scaling: str = "pre_scaled_input",
    gram_method: str = "exact",
    gram_probes: object | None = None,
    gram_probe_count: int = 64,
    gram_seed: int = 20260819,
    same_person_method: str = "exact",
    variant_probes: object | None = None,
    same_person_probe_count: int = 64,
    same_person_seed: int = 20260820,
    probe_tile_size: int = 32,
    contribution_storage: str = "snp",
) -> ContextReference | GroupedContextReference:
    """Build one standalone contextual reference object on pre-scaled G."""
    total_start = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    genotype_array = _finite_float64("genotype", genotype, ndim=2)
    basis_array = _finite_float64("basis", basis, ndim=2)
    n_samples, n_variants = genotype_array.shape
    if basis_array.shape != (n_samples, component_index.pair_index.num_basis):
        raise ValueError("Basis dimensions do not match genotype/components.")
    if projector.projector.shape != (n_samples, n_samples):
        raise ValueError("Projector and genotype sample counts differ.")
    column_energy = np.sum(genotype_array * genotype_array, axis=0, dtype=np.float64)
    minimum_energy = np.finfo(np.float64).eps * max(float(n_samples), 1.0)
    if np.any(column_energy <= minimum_energy):
        bad = np.flatnonzero(column_energy <= minimum_energy)[:5].tolist()
        raise ValueError(
            "Genotype contains zero or near-zero scale variants after declared QC; "
            f"examples={bad}."
        )
    weights, annotation_masses = validate_annotations(
        annotations, n_variants, len(component_index.annotation_names)
    )
    _validate_sha256("basis_hash", basis_hash)
    _validate_sha256("fixed_effect_hash", fixed_effect_hash)
    _validate_sha256("variant_hash", variant_hash)
    if not genotype_scaling:
        raise ValueError("genotype_scaling must be recorded.")
    groups = _canonical_loo_groups(loo_groups, n_variants)
    if contribution_storage not in {"snp", "loo_grouped"}:
        raise ValueError("contribution_storage must be 'snp' or 'loo_grouped'.")
    group_labels, group_indices, group_variant_counts = _group_layout(groups)
    contribution_group_indices = (
        group_indices if contribution_storage == "loo_grouped" else None
    )
    group_annotation_masses = np.zeros(
        (len(group_labels), weights.shape[1]), dtype=np.float64
    )
    np.add.at(group_annotation_masses, group_indices, weights)
    tile = _positive_integer("probe_tile_size", probe_tile_size)
    if gram_method not in {"exact", "hutchinson"}:
        raise ValueError("gram_method must be 'exact' or 'hutchinson'.")
    if same_person_method not in {"exact", "ustat"}:
        raise ValueError("same_person_method must be 'exact' or 'ustat'.")

    phase_times: dict[str, float] = {}
    phase_start = time.perf_counter()
    features = common_scale_features(genotype_array, basis_array, projector.projector)
    phase_times["feature_construction"] = time.perf_counter() - phase_start
    peak_rss = max(peak_rss, process.memory_info().rss)
    component_masses = _component_masses(annotation_masses, component_index)
    kernels: np.ndarray | None = None

    gram_probe_array: np.ndarray | None = None
    phase_start = time.perf_counter()
    if gram_method == "exact":
        kernels = dense_genetic_kernels(features, weights, component_index)
        gram = kernel_gram(kernels)
        raw_kernels = kernels * component_masses[:, None, None]
        contributions = _exact_gram_contributions(
            features,
            weights,
            component_index,
            raw_kernels,
            contribution_group_indices,
        )
        gram_pre_symmetry = float(np.max(np.abs(gram - gram.T), initial=0.0))
    else:
        if gram_probes is None:
            count = _positive_integer("gram_probe_count", gram_probe_count)
            gram_probe_array = _rademacher((n_samples, count), gram_seed)
        else:
            gram_probe_array = _finite_float64("gram_probes", gram_probes, ndim=2)
            if gram_probe_array.shape[0] != n_samples or gram_probe_array.shape[1] < 1:
                raise ValueError("Gram probes must have shape (N,B) with B >= 1.")
        gram, contributions, gram_pre_symmetry = _hutchinson_gram_and_contributions(
            genotype=genotype_array,
            basis=basis_array,
            projector=projector.projector,
            features=features,
            annotations=weights,
            annotation_masses=annotation_masses,
            components=component_index,
            probes=gram_probe_array,
            probe_tile_size=tile,
            group_indices=contribution_group_indices,
        )
    phase_times["gram_and_snp_contributions"] = time.perf_counter() - phase_start
    peak_rss = max(peak_rss, process.memory_info().rss)

    variant_probe_array: np.ndarray | None = None
    phase_start = time.perf_counter()
    if same_person_method == "exact":
        if kernels is None:
            kernels = dense_genetic_kernels(features, weights, component_index)
        same_person = exact_same_person_matrix(kernels)
        same_person_pre_symmetry = float(
            np.max(np.abs(same_person - same_person.T), initial=0.0)
        )
    else:
        if variant_probes is None:
            count = _positive_integer(
                "same_person_probe_count", same_person_probe_count
            )
            if count < 2:
                raise ValueError(
                    "Same-person U-statistic requires at least two probes."
                )
            variant_probe_array = _rademacher((n_variants, count), same_person_seed)
        else:
            variant_probe_array = _finite_float64(
                "variant_probes", variant_probes, ndim=2
            )
            if variant_probe_array.shape[0] != n_variants:
                raise ValueError("Variant probes and genotype use different variants.")
        (
            same_person,
            same_person_pre_symmetry,
        ) = _same_person_ustatistic_with_diagnostics(
            features, weights, component_index, variant_probe_array, tile
        )
    phase_times["same_person"] = time.perf_counter() - phase_start
    phase_times["total"] = time.perf_counter() - total_start
    peak_rss = max(peak_rss, process.memory_info().rss)

    expected_numerator = gram * np.outer(component_masses, component_masses)
    reconstructed_numerator = np.sum(contributions, axis=0, dtype=np.float64)
    contribution_error = float(
        np.max(
            np.abs(reconstructed_numerator - expected_numerator)
            / np.maximum(
                1.0,
                np.maximum(np.abs(reconstructed_numerator), np.abs(expected_numerator)),
            ),
            initial=0.0,
        )
    )
    annotation_hash = canonical_sha256(
        {
            "annotation_names": list(component_index.annotation_names),
            "weights_sha256": array_sha256(weights),
        }
    )
    loo_hash = canonical_sha256({"groups": list(groups)})
    gram_eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    same_person_eigenvalues = np.linalg.eigvalsh(0.5 * (same_person + same_person.T))
    gram_probe_meta = {
        "method": gram_method,
        "count": 0 if gram_probe_array is None else int(gram_probe_array.shape[1]),
        "seed": gram_seed
        if gram_method == "hutchinson" and gram_probes is None
        else None,
        "digest": None if gram_probe_array is None else array_sha256(gram_probe_array),
        "tile_size": tile,
        "shared_across_components": True,
    }
    same_person_probe_meta = {
        "method": same_person_method,
        "count": (
            0 if variant_probe_array is None else int(variant_probe_array.shape[1])
        ),
        "seed": (
            same_person_seed
            if same_person_method == "ustat" and variant_probes is None
            else None
        ),
        "digest": (
            None if variant_probe_array is None else array_sha256(variant_probe_array)
        ),
        "tile_size": tile,
    }
    manifest: dict[str, Any] = {
        "kind": CONTEXT_REFERENCE_KIND,
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "genotype_scaling": genotype_scaling,
        "basis_hash": basis_hash,
        "basis_array_hash": array_sha256(basis_array),
        "fixed_effect_hash": fixed_effect_hash,
        "variant_hash": variant_hash,
        "annotation_hash": annotation_hash,
        "component_index_hash": component_index.digest,
        "loo_grouping_hash": loo_hash,
        "dimensions": {
            "n_samples": n_samples,
            "residual_rank": projector.residual_rank,
            "n_variants": n_variants,
            "q": basis_array.shape[1],
            "k": weights.shape[1],
            "p_genetic": len(component_index),
            "loo_groups": len(group_labels),
        },
        "component_order": list(component_index.names),
        "annotation_names": list(component_index.annotation_names),
        "context_moments": {
            "mean": np.mean(basis_array, axis=0).tolist(),
            "second_moment": (basis_array.T @ basis_array / n_samples).tolist(),
        },
        "rank_diagnostics": {
            "fixed_effect_rank": projector.rank,
            "residual_rank": projector.residual_rank,
            "maximum_leverage": projector.maximum_leverage,
            "svd_tolerance": projector.tolerance,
        },
        "probes": {
            "gram": gram_probe_meta,
            "same_person": same_person_probe_meta,
        },
        "numerical_diagnostics": {
            "gram_pre_symmetry_max_abs": gram_pre_symmetry,
            "gram_min_eigenvalue": float(gram_eigenvalues[0]),
            "same_person_pre_symmetry_max_abs": same_person_pre_symmetry,
            "same_person_min_eigenvalue": float(same_person_eigenvalues[0]),
            "snp_contribution_reconstruction_relative_max": contribution_error,
        },
        "backend": {
            "name": "python_numpy_reference",
            "dtype": "float64",
            "native_protected_gemm": False,
            "abft": "not_applicable_python_prototype",
        },
        "approximate_loo": {
            "method": (
                "symmetric_grouped_gram_numerator_v1"
                if contribution_storage == "loo_grouped"
                else "symmetric_snp_gram_numerator_v1"
            ),
            "contribution_storage": contribution_storage,
            "exact_deleted_kernels": False,
            "same_person_deletion": "reuse_full_reference_D",
            "groups": len(set(groups)),
            "group_labels": list(group_labels),
            "group_variant_counts": group_variant_counts.tolist(),
            "group_annotation_masses_hash": array_sha256(group_annotation_masses),
        },
    }
    validate_context_manifest(manifest, expected_kind=CONTEXT_REFERENCE_KIND)
    if contribution_storage == "loo_grouped":
        return GroupedContextReference(
            manifest=manifest,
            component_index=component_index,
            annotation_masses=annotation_masses,
            gram=gram,
            same_person=same_person,
            gram_numerator_contributions=contributions,
            loo_group_ids=group_labels,
            group_annotation_masses=group_annotation_masses,
            group_variant_counts=group_variant_counts,
            phase_times_seconds=phase_times,
            peak_rss_bytes=int(peak_rss),
        )
    return ContextReference(
        manifest=manifest,
        component_index=component_index,
        annotation_weights=weights,
        annotation_masses=annotation_masses,
        gram=gram,
        same_person=same_person,
        gram_numerator_contributions=contributions,
        loo_group_ids=groups,
        phase_times_seconds=phase_times,
        peak_rss_bytes=int(peak_rss),
    )


def reference_moments_after_deleting_groups(
    reference: ContextReference | GroupedContextReference, groups: Sequence[str]
) -> ReferenceMoments:
    """Apply the declared block-local approximation and retain full D_R."""
    group_set = {str(value) for value in groups}
    if not group_set:
        return reference.full_moments
    unknown = group_set - set(reference.loo_group_ids)
    if unknown:
        raise ValueError(f"Unknown approximate-LOO groups: {sorted(unknown)}.")
    deleted = np.fromiter(
        (label in group_set for label in reference.loo_group_ids),
        dtype=bool,
        count=len(reference.loo_group_ids),
    )
    if isinstance(reference, GroupedContextReference):
        remaining_masses = reference.annotation_masses - np.sum(
            reference.group_annotation_masses[deleted], axis=0, dtype=np.float64
        )
    else:
        remaining_masses = reference.annotation_masses - np.sum(
            reference.annotation_weights[deleted], axis=0, dtype=np.float64
        )
    if np.any(remaining_masses <= 0.0):
        raise ValueError(
            "Approximate-LOO deletion leaves a nonpositive annotation mass."
        )
    component_masses = _component_masses(remaining_masses, reference.component_index)
    numerator = np.sum(
        reference.gram_numerator_contributions[~deleted], axis=0, dtype=np.float64
    )
    gram = numerator / np.outer(component_masses, component_masses)
    return ReferenceMoments(
        annotation_masses=remaining_masses,
        gram=0.5 * (gram + gram.T),
        same_person=reference.same_person.copy(),
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


def write_context_reference(
    reference: ContextReference | GroupedContextReference, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Atomically write a private JSON+NPZ development reference."""
    prefix = Path(output_prefix)
    manifest_path = prefix.with_suffix(".context-reference.json")
    arrays_path = prefix.with_suffix(".context-reference.npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{arrays_path.name}.", dir=arrays_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if isinstance(reference, GroupedContextReference):
                np.savez_compressed(
                    handle,
                    annotation_masses=reference.annotation_masses,
                    gram=reference.gram,
                    same_person=reference.same_person,
                    gram_numerator_contributions=(
                        reference.gram_numerator_contributions
                    ),
                    loo_group_ids=np.asarray(reference.loo_group_ids, dtype=np.str_),
                    group_annotation_masses=reference.group_annotation_masses,
                    group_variant_counts=reference.group_variant_counts,
                )
            else:
                np.savez_compressed(
                    handle,
                    annotation_weights=reference.annotation_weights,
                    annotation_masses=reference.annotation_masses,
                    gram=reference.gram,
                    same_person=reference.same_person,
                    gram_numerator_contributions=(
                        reference.gram_numerator_contributions
                    ),
                    loo_group_ids=np.asarray(reference.loo_group_ids, dtype=np.str_),
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
    payload = dict(reference.manifest)
    payload["artifact"] = {
        "path": arrays_path.name,
        "sha256": array_sha256(np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)),
        "format": (
            "npz_grouped_development_v2"
            if isinstance(reference, GroupedContextReference)
            else "npz_development_v1"
        ),
    }
    payload["performance"] = {
        "phase_times_seconds": reference.phase_times_seconds,
        "peak_rss_bytes": reference.peak_rss_bytes,
    }
    _atomic_write_text(manifest_path, json.dumps(payload, sort_keys=True, indent=2))
    return manifest_path, arrays_path


def load_context_reference(
    manifest_path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> ContextReference | GroupedContextReference:
    path = Path(manifest_path)
    if path.name.endswith(".contextual-reference-v1.npz"):
        raise ValueError(
            "Stable contextual reference V1 artifacts require the isolated V1 loader."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_context_manifest(
        payload, expected_kind=CONTEXT_REFERENCE_KIND, expected=expected
    )
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("format") not in {
        "npz_development_v1",
        "npz_grouped_development_v2",
    }:
        raise ValueError("Context reference has no supported development artifact.")
    grouped_format = artifact.get("format") == "npz_grouped_development_v2"
    arrays_path = path.parent / str(artifact.get("path"))
    observed_hash = array_sha256(
        np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)
    )
    if observed_hash != artifact.get("sha256"):
        raise ValueError("Context-reference artifact SHA-256 mismatch.")
    dimensions = payload["dimensions"]
    declared_annotation_names = payload.get("annotation_names")
    if declared_annotation_names is None:
        annotation_names = tuple(
            dict.fromkeys(name.split(":", 2)[1] for name in payload["component_order"])
        )
    else:
        annotation_names = tuple(str(value) for value in declared_annotation_names)
    components = ContextComponentIndex(
        annotation_names, ContextPairIndex(int(dimensions["q"]))
    )
    if components.digest != payload["component_index_hash"]:
        raise ValueError("Context-reference component-index digest mismatch.")
    if list(components.names) != payload["component_order"]:
        raise ValueError("Context-reference component order is not canonical.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {name: np.array(arrays[name], copy=True) for name in arrays.files}
    if grouped_format:
        n_groups = int(dimensions.get("loo_groups", 0))
        expected_shapes = {
            "annotation_masses": (int(dimensions["k"]),),
            "gram": (len(components), len(components)),
            "same_person": (len(components), len(components)),
            "gram_numerator_contributions": (
                n_groups,
                len(components),
                len(components),
            ),
            "loo_group_ids": (n_groups,),
            "group_annotation_masses": (n_groups, int(dimensions["k"])),
            "group_variant_counts": (n_groups,),
        }
        for name, shape in expected_shapes.items():
            if name not in values or values[name].shape != shape:
                raise ValueError(
                    f"Grouped context-reference array {name!r} has invalid shape "
                    f"{None if name not in values else values[name].shape}; "
                    f"expected {shape}."
                )
        if set(values) != set(expected_shapes):
            raise ValueError("Grouped context-reference artifact has extra arrays.")
        for name in (
            "annotation_masses",
            "gram",
            "same_person",
            "gram_numerator_contributions",
            "group_annotation_masses",
        ):
            if values[name].dtype != np.dtype(np.float64):
                raise ValueError(
                    f"Grouped context-reference array {name!r} must be float64."
                )
            if not np.all(np.isfinite(values[name])):
                raise ValueError(
                    f"Grouped context-reference array {name!r} contains non-finite values."
                )
        if values["group_variant_counts"].dtype.kind not in {"i", "u"}:
            raise ValueError("Grouped context-reference counts must be integer-valued.")
        if np.any(values["group_variant_counts"] <= 0):
            raise ValueError("Grouped context-reference counts must be positive.")
        if (
            payload["approximate_loo"].get("group_variant_counts")
            != values["group_variant_counts"].tolist()
        ):
            raise ValueError("Grouped context-reference count manifest mismatch.")
        if int(np.sum(values["group_variant_counts"])) != int(dimensions["n_variants"]):
            raise ValueError(
                "Grouped context-reference counts do not reconstruct variant count."
            )
        if np.any(values["group_annotation_masses"] < 0.0):
            raise ValueError("Grouped annotation masses must be nonnegative.")
        reconstructed_masses = np.sum(
            values["group_annotation_masses"], axis=0, dtype=np.float64
        )
        mass_scale = np.maximum(
            1.0,
            np.maximum(
                np.abs(reconstructed_masses), np.abs(values["annotation_masses"])
            ),
        )
        if (
            np.max(
                np.abs(reconstructed_masses - values["annotation_masses"]) / mass_scale,
                initial=0.0,
            )
            > 1.0e-12
        ):
            raise ValueError(
                "Grouped context-reference masses do not reconstruct totals."
            )
        if array_sha256(values["group_annotation_masses"]) != payload[
            "approximate_loo"
        ].get("group_annotation_masses_hash"):
            raise ValueError("Grouped context-reference mass hash mismatch.")
        labels = tuple(str(value) for value in values["loo_group_ids"])
        if len(set(labels)) != len(labels) or any(not value for value in labels):
            raise ValueError(
                "Grouped context-reference labels must be unique and nonempty."
            )
        if list(labels) != payload["approximate_loo"].get("group_labels"):
            raise ValueError("Grouped context-reference label order mismatch.")
        scale = max(
            float(np.max(np.abs(values["gram"]), initial=0.0)),
            float(np.max(np.abs(values["same_person"]), initial=0.0)),
            1.0,
        )
        if (
            np.max(np.abs(values["gram"] - values["gram"].T), initial=0.0)
            > 1.0e-12 * scale
        ):
            raise ValueError("Grouped context-reference Gram is not symmetric.")
        if (
            np.max(
                np.abs(values["same_person"] - values["same_person"].T),
                initial=0.0,
            )
            > 1.0e-12 * scale
        ):
            raise ValueError(
                "Grouped context-reference same-person matrix is not symmetric."
            )
        contribution_scale = max(
            float(np.max(np.abs(values["gram_numerator_contributions"]), initial=0.0)),
            1.0,
        )
        if (
            np.max(
                np.abs(
                    values["gram_numerator_contributions"]
                    - np.swapaxes(values["gram_numerator_contributions"], 1, 2)
                ),
                initial=0.0,
            )
            > 1.0e-12 * contribution_scale
        ):
            raise ValueError(
                "Grouped context-reference contributions are not symmetric."
            )
        component_masses = _component_masses(values["annotation_masses"], components)
        reconstructed = np.sum(
            values["gram_numerator_contributions"], axis=0, dtype=np.float64
        ) / np.outer(component_masses, component_masses)
        reconstruction_scale = np.maximum(
            1.0, np.maximum(np.abs(reconstructed), np.abs(values["gram"]))
        )
        if (
            np.max(
                np.abs(reconstructed - values["gram"]) / reconstruction_scale,
                initial=0.0,
            )
            > 1.0e-10
        ):
            raise ValueError(
                "Grouped context-reference contributions do not reconstruct Gram."
            )
        performance = payload.get("performance", {})
        return GroupedContextReference(
            manifest={
                key: value
                for key, value in payload.items()
                if key not in {"artifact", "performance"}
            },
            component_index=components,
            annotation_masses=values["annotation_masses"],
            gram=values["gram"],
            same_person=values["same_person"],
            gram_numerator_contributions=values["gram_numerator_contributions"],
            loo_group_ids=labels,
            group_annotation_masses=values["group_annotation_masses"],
            group_variant_counts=values["group_variant_counts"].astype(
                np.int64, copy=False
            ),
            phase_times_seconds=dict(performance.get("phase_times_seconds", {})),
            peak_rss_bytes=int(performance.get("peak_rss_bytes", 0)),
        )
    expected_shapes = {
        "annotation_weights": (
            int(dimensions["n_variants"]),
            int(dimensions["k"]),
        ),
        "annotation_masses": (int(dimensions["k"]),),
        "gram": (len(components), len(components)),
        "same_person": (len(components), len(components)),
        "gram_numerator_contributions": (
            int(dimensions["n_variants"]),
            len(components),
            len(components),
        ),
        "loo_group_ids": (int(dimensions["n_variants"]),),
    }
    for name, shape in expected_shapes.items():
        if name not in values or values[name].shape != shape:
            raise ValueError(
                f"Context-reference array {name!r} has invalid shape "
                f"{None if name not in values else values[name].shape}; expected {shape}."
            )
    for name in (
        "annotation_weights",
        "annotation_masses",
        "gram",
        "same_person",
        "gram_numerator_contributions",
    ):
        if values[name].dtype != np.dtype(np.float64):
            raise ValueError(f"Context-reference array {name!r} must be float64.")
        if not np.all(np.isfinite(values[name])):
            raise ValueError(
                f"Context-reference array {name!r} contains non-finite values."
            )
    if np.any(values["annotation_weights"] < 0.0):
        raise ValueError("Context-reference annotations must be nonnegative.")
    recomputed_masses = np.sum(values["annotation_weights"], axis=0, dtype=np.float64)
    mass_scale = np.maximum(
        1.0, np.maximum(np.abs(recomputed_masses), np.abs(values["annotation_masses"]))
    )
    if (
        np.max(np.abs(values["annotation_masses"] - recomputed_masses) / mass_scale)
        > 1e-12
    ):
        raise ValueError("Context-reference annotation masses do not match weights.")
    scale = max(
        float(np.max(np.abs(values["gram"]), initial=0.0)),
        float(np.max(np.abs(values["same_person"]), initial=0.0)),
        1.0,
    )
    if np.max(np.abs(values["gram"] - values["gram"].T), initial=0.0) > 1e-12 * scale:
        raise ValueError("Context-reference Gram is not symmetric.")
    if (
        np.max(np.abs(values["same_person"] - values["same_person"].T), initial=0.0)
        > 1e-12 * scale
    ):
        raise ValueError("Context-reference same-person matrix is not symmetric.")
    contribution_scale = max(
        float(np.max(np.abs(values["gram_numerator_contributions"]), initial=0.0)),
        1.0,
    )
    if (
        np.max(
            np.abs(
                values["gram_numerator_contributions"]
                - np.swapaxes(values["gram_numerator_contributions"], 1, 2)
            ),
            initial=0.0,
        )
        > 1e-12 * contribution_scale
    ):
        raise ValueError("Context-reference SNP contributions are not symmetric.")
    annotation_hash = canonical_sha256(
        {
            "annotation_names": list(components.annotation_names),
            "weights_sha256": array_sha256(values["annotation_weights"]),
        }
    )
    if annotation_hash != payload["annotation_hash"]:
        raise ValueError("Context-reference annotation hash mismatch.")
    if (
        canonical_sha256({"groups": [str(v) for v in values["loo_group_ids"]]})
        != payload["loo_grouping_hash"]
    ):
        raise ValueError("Context-reference LOO grouping hash mismatch.")
    component_masses = _component_masses(values["annotation_masses"], components)
    reconstructed = np.sum(
        values["gram_numerator_contributions"], axis=0, dtype=np.float64
    ) / np.outer(component_masses, component_masses)
    reconstruction_scale = np.maximum(
        1.0, np.maximum(np.abs(reconstructed), np.abs(values["gram"]))
    )
    if np.max(np.abs(reconstructed - values["gram"]) / reconstruction_scale) > 1e-10:
        raise ValueError("Context-reference SNP contributions do not reconstruct Gram.")
    performance = payload.get("performance", {})
    return ContextReference(
        manifest={
            key: value
            for key, value in payload.items()
            if key not in {"artifact", "performance"}
        },
        component_index=components,
        annotation_weights=values["annotation_weights"],
        annotation_masses=values["annotation_masses"],
        gram=values["gram"],
        same_person=values["same_person"],
        gram_numerator_contributions=values["gram_numerator_contributions"],
        loo_group_ids=tuple(str(value) for value in values["loo_group_ids"]),
        phase_times_seconds=dict(performance.get("phase_times_seconds", {})),
        peak_rss_bytes=int(performance.get("peak_rss_bytes", 0)),
    )
