"""Correctness-first contextual trait-summary construction.

The generator streams an in-memory genotype matrix by variant block to model
the eventual decoder contract.  It retains raw per-SNP numerators sufficient
for the accepted summary-only approximate leave-one-out procedure; fitting does
not need the individual-level inputs after this object is written.
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
    project_normalize_phenotype,
    residual_moments_low_rank,
    validate_annotations,
)
from .spec import (
    CONTEXT_SCHEMA_VERSION,
    CONTEXT_TRAIT_KIND,
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_json,
    canonical_sha256,
    freeze_context_mapping,
    owned_readonly_array,
    validate_context_manifest,
)


@dataclass(frozen=True)
class ContextTraitMoments:
    annotation_masses: np.ndarray
    genetic_rhs: np.ndarray
    genetic_traces: np.ndarray
    genetic_residual: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "annotation_masses",
            "genetic_rhs",
            "genetic_traces",
            "genetic_residual",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class ContextTraitSummary:
    manifest: dict[str, Any]
    component_index: ContextComponentIndex
    residual_names: tuple[str, ...]
    annotation_weights: np.ndarray
    annotation_masses: np.ndarray
    genetic_rhs: np.ndarray
    genetic_traces: np.ndarray
    genetic_residual: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray
    rhs_numerator_contributions: np.ndarray
    trace_numerator_contributions: np.ndarray
    genetic_residual_numerator_contributions: np.ndarray
    loo_group_ids: tuple[str, ...]
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int
    decode_passes: int
    decoded_blocks: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        object.__setattr__(self, "residual_names", tuple(self.residual_names))
        object.__setattr__(self, "loo_group_ids", tuple(self.loo_group_ids))
        object.__setattr__(
            self,
            "phase_times_seconds",
            freeze_context_mapping(self.phase_times_seconds),
        )
        for name in (
            "annotation_weights",
            "annotation_masses",
            "genetic_rhs",
            "genetic_traces",
            "genetic_residual",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
            "rhs_numerator_contributions",
            "trace_numerator_contributions",
            "genetic_residual_numerator_contributions",
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
    def full_moments(self) -> ContextTraitMoments:
        return ContextTraitMoments(
            annotation_masses=self.annotation_masses.copy(),
            genetic_rhs=self.genetic_rhs.copy(),
            genetic_traces=self.genetic_traces.copy(),
            genetic_residual=self.genetic_residual.copy(),
            residual_rhs=self.residual_rhs.copy(),
            residual_traces=self.residual_traces.copy(),
            residual_gram=self.residual_gram.copy(),
        )


@dataclass(frozen=True)
class GroupedContextTraitSummary:
    """Trait moments with lossless frozen-LOO group numerators and no M axis."""

    manifest: dict[str, Any]
    component_index: ContextComponentIndex
    residual_names: tuple[str, ...]
    annotation_masses: np.ndarray
    genetic_rhs: np.ndarray
    genetic_traces: np.ndarray
    genetic_residual: np.ndarray
    residual_rhs: np.ndarray
    residual_traces: np.ndarray
    residual_gram: np.ndarray
    rhs_numerator_contributions: np.ndarray
    trace_numerator_contributions: np.ndarray
    genetic_residual_numerator_contributions: np.ndarray
    loo_group_ids: tuple[str, ...]
    group_annotation_masses: np.ndarray
    group_variant_counts: np.ndarray
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int
    decode_passes: int
    decoded_blocks: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        object.__setattr__(self, "residual_names", tuple(self.residual_names))
        object.__setattr__(self, "loo_group_ids", tuple(self.loo_group_ids))
        object.__setattr__(
            self,
            "phase_times_seconds",
            freeze_context_mapping(self.phase_times_seconds),
        )
        for name in (
            "annotation_masses",
            "genetic_rhs",
            "genetic_traces",
            "genetic_residual",
            "residual_rhs",
            "residual_traces",
            "residual_gram",
            "rhs_numerator_contributions",
            "trace_numerator_contributions",
            "genetic_residual_numerator_contributions",
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
    def group_rhs_numerator_contributions(self) -> np.ndarray:
        return self.rhs_numerator_contributions

    @property
    def group_trace_numerator_contributions(self) -> np.ndarray:
        return self.trace_numerator_contributions

    @property
    def group_genetic_residual_numerator_contributions(self) -> np.ndarray:
        return self.genetic_residual_numerator_contributions

    @property
    def full_moments(self) -> ContextTraitMoments:
        return ContextTraitMoments(
            annotation_masses=self.annotation_masses.copy(),
            genetic_rhs=self.genetic_rhs.copy(),
            genetic_traces=self.genetic_traces.copy(),
            genetic_residual=self.genetic_residual.copy(),
            residual_rhs=self.residual_rhs.copy(),
            residual_traces=self.residual_traces.copy(),
            residual_gram=self.residual_gram.copy(),
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


def batched_contextual_scores(
    genotype: object,
    basis: object,
    normalized_phenotypes: object,
    residual_ranks: object,
    *,
    block_size: int = 256,
) -> np.ndarray:
    """Return ``(L,Q,M)`` HE-scale scores in one block-streamed pass."""
    genotype_array = _finite_float64("genotype", genotype, ndim=2)
    basis_array = _finite_float64("basis", basis, ndim=2)
    phenotypes = _finite_float64("normalized_phenotypes", normalized_phenotypes)
    if phenotypes.ndim == 1:
        phenotypes = phenotypes[:, None]
    if phenotypes.ndim != 2:
        raise ValueError("normalized_phenotypes must be one- or two-dimensional.")
    ranks = _finite_float64("residual_ranks", residual_ranks, ndim=1)
    n_samples, n_variants = genotype_array.shape
    if basis_array.shape[0] != n_samples or phenotypes.shape[0] != n_samples:
        raise ValueError("Score inputs use different sample counts.")
    if ranks.shape != (phenotypes.shape[1],) or np.any(ranks < 1.0):
        raise ValueError("One positive residual rank is required per phenotype.")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer.")
    q_count = basis_array.shape[1]
    l_count = phenotypes.shape[1]
    right_hand_sides = np.column_stack(
        [
            basis_array[:, q] * phenotypes[:, ell]
            for ell in range(l_count)
            for q in range(q_count)
        ]
    )
    result = np.empty((l_count, q_count, n_variants), dtype=np.float64)
    for start in range(0, n_variants, block_size):
        stop = min(start + block_size, n_variants)
        products = genotype_array[:, start:stop].T @ right_hand_sides
        result[:, :, start:stop] = (
            products.reshape(stop - start, l_count, q_count).transpose(1, 2, 0)
            / np.sqrt(ranks)[:, None, None]
        )
    return result


def build_context_trait_summary(
    *,
    genotype: object,
    basis: object,
    phenotype: object,
    projector: ProjectorResult,
    annotations: object,
    component_index: ContextComponentIndex,
    residual_basis: object,
    residual_names: Sequence[str],
    basis_hash: str,
    fixed_effect_hash: str,
    variant_hash: str,
    loo_groups: Sequence[Any] | None = None,
    genotype_scaling: str = "pre_scaled_input",
    block_size: int = 256,
    contribution_storage: str = "snp",
) -> ContextTraitSummary | GroupedContextTraitSummary:
    """Construct one complete private contextual trait summary."""
    start_total = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    genotype_array = _finite_float64("genotype", genotype, ndim=2)
    basis_array = _finite_float64("basis", basis, ndim=2)
    residual_array = _finite_float64("residual_basis", residual_basis, ndim=2)
    n_samples, n_variants = genotype_array.shape
    q_count = basis_array.shape[1]
    h_count = residual_array.shape[1]
    if basis_array.shape[0] != n_samples or residual_array.shape[0] != n_samples:
        raise ValueError("Trait-summary inputs use different sample counts.")
    if projector.projector.shape != (n_samples, n_samples):
        raise ValueError("Projector and genotype sample counts differ.")
    if q_count != component_index.pair_index.num_basis:
        raise ValueError("Basis dimension does not match the component index.")
    column_energy = np.sum(genotype_array * genotype_array, axis=0, dtype=np.float64)
    minimum_energy = np.finfo(np.float64).eps * max(float(n_samples), 1.0)
    if np.any(column_energy <= minimum_energy):
        bad = np.flatnonzero(column_energy <= minimum_energy)[:5].tolist()
        raise ValueError(
            "Genotype contains zero or near-zero scale variants after declared QC; "
            f"examples={bad}."
        )
    residual_names_tuple = tuple(str(value) for value in residual_names)
    if (
        len(residual_names_tuple) != h_count
        or len(set(residual_names_tuple)) != h_count
    ):
        raise ValueError(
            "Residual names must be unique and match residual-basis columns."
        )
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer.")
    weights, masses = validate_annotations(
        annotations, n_variants, len(component_index.annotation_names)
    )
    _validate_sha256("basis_hash", basis_hash)
    _validate_sha256("fixed_effect_hash", fixed_effect_hash)
    _validate_sha256("variant_hash", variant_hash)
    if not genotype_scaling:
        raise ValueError("genotype_scaling must be recorded.")
    group_ids = _canonical_loo_groups(loo_groups, n_variants)
    if contribution_storage not in {"snp", "loo_grouped"}:
        raise ValueError("contribution_storage must be 'snp' or 'loo_grouped'.")
    group_labels, group_indices, group_variant_counts = _group_layout(group_ids)
    contribution_count = (
        len(group_labels) if contribution_storage == "loo_grouped" else n_variants
    )
    group_annotation_masses = np.zeros(
        (len(group_labels), weights.shape[1]), dtype=np.float64
    )
    np.add.at(group_annotation_masses, group_indices, weights)

    phase_times: dict[str, float] = {}
    phase_start = time.perf_counter()
    y = project_normalize_phenotype(phenotype, projector)
    residual = residual_moments_low_rank(projector.fixed_basis, residual_array, y)
    phase_times["phenotype_and_residual"] = time.perf_counter() - phase_start

    p_count = len(component_index)
    rhs_contributions = np.zeros((contribution_count, p_count), dtype=np.float64)
    trace_contributions = np.zeros((contribution_count, p_count), dtype=np.float64)
    genetic_residual_contributions = np.zeros(
        (contribution_count, p_count, h_count), dtype=np.float64
    )
    right_hand_sides = basis_array * y[:, None]
    decoded_blocks = 0
    phase_start = time.perf_counter()
    for start in range(0, n_variants, block_size):
        stop = min(start + block_size, n_variants)
        genotype_block = genotype_array[:, start:stop]
        score_products = genotype_block.T @ right_hand_sides
        features = np.empty((q_count, n_samples, stop - start), dtype=np.float64)
        for q in range(q_count):
            features[q] = projector.projector @ (
                basis_array[:, q, None] * genotype_block
            )
        for component in component_index.entries:
            annotation = weights[start:stop, component.annotation_index]
            pair_scale = component.kernel_factor * annotation
            rhs_value = (
                pair_scale
                * score_products[:, component.q]
                * score_products[:, component.r]
            )
            feature_products = np.einsum(
                "nb,nb->b",
                features[component.q],
                features[component.r],
                optimize=True,
            )
            trace_value = pair_scale * feature_products
            if contribution_storage == "loo_grouped":
                local_groups = group_indices[start:stop]
                np.add.at(
                    rhs_contributions[:, component.index], local_groups, rhs_value
                )
                np.add.at(
                    trace_contributions[:, component.index],
                    local_groups,
                    trace_value,
                )
            else:
                rhs_contributions[start:stop, component.index] = rhs_value
                trace_contributions[start:stop, component.index] = trace_value
            for h in range(h_count):
                weighted_products = np.einsum(
                    "nb,n,nb->b",
                    features[component.q],
                    residual_array[:, h],
                    features[component.r],
                    optimize=True,
                )
                value = pair_scale * weighted_products
                if contribution_storage == "loo_grouped":
                    np.add.at(
                        genetic_residual_contributions[:, component.index, h],
                        local_groups,
                        value,
                    )
                else:
                    genetic_residual_contributions[
                        start:stop, component.index, h
                    ] = value
        decoded_blocks += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
    phase_times["genotype_pass"] = time.perf_counter() - phase_start

    component_masses = np.asarray(
        [masses[entry.annotation_index] for entry in component_index.entries]
    )
    genetic_rhs = np.sum(rhs_contributions, axis=0) / component_masses
    genetic_traces = np.sum(trace_contributions, axis=0) / component_masses
    genetic_residual = (
        np.sum(genetic_residual_contributions, axis=0) / component_masses[:, None]
    )
    phase_times["total"] = time.perf_counter() - start_total
    peak_rss = max(peak_rss, process.memory_info().rss)

    annotation_hash = canonical_sha256(
        {
            "annotation_names": list(component_index.annotation_names),
            "weights_sha256": array_sha256(weights),
        }
    )
    loo_grouping_hash = canonical_sha256({"groups": list(group_ids)})
    manifest: dict[str, Any] = {
        "kind": CONTEXT_TRAIT_KIND,
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "genotype_scaling": genotype_scaling,
        "basis_hash": basis_hash,
        "basis_array_hash": array_sha256(basis_array),
        "residual_basis_hash": array_sha256(residual_array),
        "fixed_effect_hash": fixed_effect_hash,
        "variant_hash": variant_hash,
        "annotation_hash": annotation_hash,
        "component_index_hash": component_index.digest,
        "loo_grouping_hash": loo_grouping_hash,
        "dimensions": {
            "n_samples": n_samples,
            "residual_rank": projector.residual_rank,
            "n_variants": n_variants,
            "q": q_count,
            "k": len(component_index.annotation_names),
            "h": h_count,
            "p_genetic": p_count,
            "loo_groups": len(group_labels),
        },
        "component_order": list(component_index.names),
        "annotation_names": list(component_index.annotation_names),
        "residual_order": list(residual_names_tuple),
        "context_moments": {
            "mean": np.mean(basis_array, axis=0).tolist(),
            "second_moment": (basis_array.T @ basis_array / n_samples).tolist(),
        },
        "phenotype_scale": {
            "normalization": "projected_sum_squares_equals_residual_rank",
            "projected_sum_squares": float(y @ y),
        },
        "rank_diagnostics": {
            "fixed_effect_rank": projector.rank,
            "residual_rank": projector.residual_rank,
            "maximum_leverage": projector.maximum_leverage,
            "svd_tolerance": projector.tolerance,
        },
        "backend": {
            "name": "python_numpy_reference",
            "dtype": "float64",
            "decode_passes": 1,
            "block_size": block_size,
        },
        "approximate_loo": {
            "method": (
                "grouped_contribution_summary_v1"
                if contribution_storage == "loo_grouped"
                else "snp_contribution_summary_v1"
            ),
            "contribution_storage": contribution_storage,
            "exact_deleted_kernels": False,
            "groups": len(set(group_ids)),
            "group_labels": list(group_labels),
            "group_variant_counts": group_variant_counts.tolist(),
            "group_annotation_masses_hash": array_sha256(group_annotation_masses),
        },
    }
    validate_context_manifest(manifest, expected_kind=CONTEXT_TRAIT_KIND)
    if contribution_storage == "loo_grouped":
        return GroupedContextTraitSummary(
            manifest=manifest,
            component_index=component_index,
            residual_names=residual_names_tuple,
            annotation_masses=masses,
            genetic_rhs=genetic_rhs,
            genetic_traces=genetic_traces,
            genetic_residual=genetic_residual,
            residual_rhs=residual.rhs,
            residual_traces=residual.traces,
            residual_gram=residual.gram,
            rhs_numerator_contributions=rhs_contributions,
            trace_numerator_contributions=trace_contributions,
            genetic_residual_numerator_contributions=(genetic_residual_contributions),
            loo_group_ids=group_labels,
            group_annotation_masses=group_annotation_masses,
            group_variant_counts=group_variant_counts,
            phase_times_seconds=phase_times,
            peak_rss_bytes=int(peak_rss),
            decode_passes=1,
            decoded_blocks=decoded_blocks,
        )
    return ContextTraitSummary(
        manifest=manifest,
        component_index=component_index,
        residual_names=residual_names_tuple,
        annotation_weights=weights,
        annotation_masses=masses,
        genetic_rhs=genetic_rhs,
        genetic_traces=genetic_traces,
        genetic_residual=genetic_residual,
        residual_rhs=residual.rhs,
        residual_traces=residual.traces,
        residual_gram=residual.gram,
        rhs_numerator_contributions=rhs_contributions,
        trace_numerator_contributions=trace_contributions,
        genetic_residual_numerator_contributions=genetic_residual_contributions,
        loo_group_ids=group_ids,
        phase_times_seconds=phase_times,
        peak_rss_bytes=int(peak_rss),
        decode_passes=1,
        decoded_blocks=decoded_blocks,
    )


def trait_moments_after_deleting_groups(
    summary: ContextTraitSummary | GroupedContextTraitSummary,
    groups: Sequence[str],
) -> ContextTraitMoments:
    group_set = {str(value) for value in groups}
    if not group_set:
        return summary.full_moments
    known = set(summary.loo_group_ids)
    unknown = group_set - known
    if unknown:
        raise ValueError(f"Unknown approximate-LOO groups: {sorted(unknown)}.")
    deleted = np.fromiter(
        (label in group_set for label in summary.loo_group_ids),
        dtype=bool,
        count=len(summary.loo_group_ids),
    )
    if isinstance(summary, GroupedContextTraitSummary):
        remaining_masses = summary.annotation_masses - np.sum(
            summary.group_annotation_masses[deleted], axis=0
        )
    else:
        remaining_masses = summary.annotation_masses - np.sum(
            summary.annotation_weights[deleted], axis=0
        )
    if np.any(remaining_masses <= 0.0):
        raise ValueError(
            "Approximate-LOO deletion leaves a nonpositive annotation mass."
        )
    component_masses = np.asarray(
        [
            remaining_masses[entry.annotation_index]
            for entry in summary.component_index.entries
        ]
    )
    rhs_numerator = np.sum(summary.rhs_numerator_contributions[~deleted], axis=0)
    trace_numerator = np.sum(summary.trace_numerator_contributions[~deleted], axis=0)
    genetic_residual_numerator = np.sum(
        summary.genetic_residual_numerator_contributions[~deleted], axis=0
    )
    return ContextTraitMoments(
        annotation_masses=remaining_masses,
        genetic_rhs=rhs_numerator / component_masses,
        genetic_traces=trace_numerator / component_masses,
        genetic_residual=genetic_residual_numerator / component_masses[:, None],
        residual_rhs=summary.residual_rhs.copy(),
        residual_traces=summary.residual_traces.copy(),
        residual_gram=summary.residual_gram.copy(),
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


def write_context_trait_summary(
    summary: ContextTraitSummary | GroupedContextTraitSummary,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Atomically write a private JSON+NPZ development summary."""
    prefix = Path(output_prefix)
    manifest_path = prefix.with_suffix(".context-trait.json")
    arrays_path = prefix.with_suffix(".context-trait.npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{arrays_path.name}.", dir=arrays_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            common = {
                "annotation_masses": summary.annotation_masses,
                "genetic_rhs": summary.genetic_rhs,
                "genetic_traces": summary.genetic_traces,
                "genetic_residual": summary.genetic_residual,
                "residual_rhs": summary.residual_rhs,
                "residual_traces": summary.residual_traces,
                "residual_gram": summary.residual_gram,
                "rhs_numerator_contributions": (summary.rhs_numerator_contributions),
                "trace_numerator_contributions": (
                    summary.trace_numerator_contributions
                ),
                "genetic_residual_numerator_contributions": (
                    summary.genetic_residual_numerator_contributions
                ),
                "loo_group_ids": np.asarray(summary.loo_group_ids, dtype=np.str_),
            }
            if isinstance(summary, GroupedContextTraitSummary):
                np.savez_compressed(
                    handle,
                    **common,
                    group_annotation_masses=summary.group_annotation_masses,
                    group_variant_counts=summary.group_variant_counts,
                )
            else:
                np.savez_compressed(
                    handle,
                    **common,
                    annotation_weights=summary.annotation_weights,
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
    payload = dict(summary.manifest)
    payload["artifact"] = {
        "path": arrays_path.name,
        "sha256": array_sha256(np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)),
        "format": (
            "npz_grouped_development_v2"
            if isinstance(summary, GroupedContextTraitSummary)
            else "npz_development_v1"
        ),
    }
    payload["performance"] = {
        "phase_times_seconds": summary.phase_times_seconds,
        "peak_rss_bytes": summary.peak_rss_bytes,
        "decoded_blocks": summary.decoded_blocks,
    }
    _atomic_write_text(manifest_path, json.dumps(payload, sort_keys=True, indent=2))
    return manifest_path, arrays_path


def load_context_trait_summary(
    manifest_path: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> ContextTraitSummary | GroupedContextTraitSummary:
    path = Path(manifest_path)
    if path.name.endswith(".contextual-trait-v1.npz"):
        raise ValueError(
            "Stable contextual trait V1 artifacts require "
            "load_contextual_trait_v1()."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_context_manifest(
        payload, expected_kind=CONTEXT_TRAIT_KIND, expected=expected
    )
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("format") not in {
        "npz_development_v1",
        "npz_grouped_development_v2",
    }:
        raise ValueError("Trait summary has no supported development artifact.")
    grouped_format = artifact.get("format") == "npz_grouped_development_v2"
    arrays_path = path.parent / str(artifact.get("path"))
    observed_hash = array_sha256(
        np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)
    )
    if observed_hash != artifact.get("sha256"):
        raise ValueError("Trait-summary artifact SHA-256 mismatch.")
    dimensions = payload["dimensions"]
    declared_annotation_names = payload.get("annotation_names")
    if declared_annotation_names is None:
        annotation_names = tuple(
            name.split(":", 2)[1] for name in payload["component_order"]
        )
        # Preserve compatibility with v1 artifacts lacking explicit names.
        annotation_names = tuple(dict.fromkeys(annotation_names))
    else:
        annotation_names = tuple(str(value) for value in declared_annotation_names)
    component_index = ContextComponentIndex(
        annotation_names, ContextPairIndex(int(dimensions["q"]))
    )
    if component_index.digest != payload["component_index_hash"]:
        raise ValueError("Trait-summary component-index digest mismatch.")
    if list(component_index.names) != payload["component_order"]:
        raise ValueError("Trait-summary component order is not canonical.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {name: np.array(arrays[name], copy=True) for name in arrays.files}
    phase = payload.get("performance", {})
    if grouped_format:
        n_groups = int(dimensions.get("loo_groups", 0))
        p_count = len(component_index)
        h_count = len(payload["residual_order"])
        k_count = int(dimensions["k"])
        expected_shapes = {
            "annotation_masses": (k_count,),
            "genetic_rhs": (p_count,),
            "genetic_traces": (p_count,),
            "genetic_residual": (p_count, h_count),
            "residual_rhs": (h_count,),
            "residual_traces": (h_count,),
            "residual_gram": (h_count, h_count),
            "rhs_numerator_contributions": (n_groups, p_count),
            "trace_numerator_contributions": (n_groups, p_count),
            "genetic_residual_numerator_contributions": (
                n_groups,
                p_count,
                h_count,
            ),
            "loo_group_ids": (n_groups,),
            "group_annotation_masses": (n_groups, k_count),
            "group_variant_counts": (n_groups,),
        }
        for name, shape in expected_shapes.items():
            if name not in values or values[name].shape != shape:
                raise ValueError(
                    f"Grouped trait-summary array {name!r} has invalid shape "
                    f"{None if name not in values else values[name].shape}; "
                    f"expected {shape}."
                )
        if set(values) != set(expected_shapes):
            raise ValueError("Grouped trait-summary artifact has extra arrays.")
        float_names = tuple(
            name
            for name in expected_shapes
            if name not in {"loo_group_ids", "group_variant_counts"}
        )
        for name in float_names:
            if values[name].dtype != np.dtype(np.float64):
                raise ValueError(
                    f"Grouped trait-summary array {name!r} must be float64."
                )
            if not np.all(np.isfinite(values[name])):
                raise ValueError(
                    f"Grouped trait-summary array {name!r} contains non-finite values."
                )
        if values["group_variant_counts"].dtype.kind not in {"i", "u"}:
            raise ValueError("Grouped trait-summary counts must be integer-valued.")
        if np.any(values["group_variant_counts"] <= 0):
            raise ValueError("Grouped trait-summary counts must be positive.")
        if (
            payload["approximate_loo"].get("group_variant_counts")
            != values["group_variant_counts"].tolist()
        ):
            raise ValueError("Grouped trait-summary count manifest mismatch.")
        if int(np.sum(values["group_variant_counts"])) != int(dimensions["n_variants"]):
            raise ValueError(
                "Grouped trait-summary counts do not reconstruct variant count."
            )
        if np.any(values["group_annotation_masses"] < 0.0):
            raise ValueError(
                "Grouped trait-summary annotation masses must be nonnegative."
            )
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
            raise ValueError("Grouped trait-summary masses do not reconstruct totals.")
        if array_sha256(values["group_annotation_masses"]) != payload[
            "approximate_loo"
        ].get("group_annotation_masses_hash"):
            raise ValueError("Grouped trait-summary mass hash mismatch.")
        labels = tuple(str(value) for value in values["loo_group_ids"])
        if len(set(labels)) != len(labels) or any(not value for value in labels):
            raise ValueError(
                "Grouped trait-summary labels must be unique and nonempty."
            )
        if list(labels) != payload["approximate_loo"].get("group_labels"):
            raise ValueError("Grouped trait-summary label order mismatch.")
        residual_scale = max(
            float(np.max(np.abs(values["residual_gram"]), initial=0.0)), 1.0
        )
        if (
            np.max(
                np.abs(values["residual_gram"] - values["residual_gram"].T),
                initial=0.0,
            )
            > 1.0e-12 * residual_scale
        ):
            raise ValueError("Grouped trait-summary residual Gram is not symmetric.")
        component_masses = np.asarray(
            [
                values["annotation_masses"][entry.annotation_index]
                for entry in component_index.entries
            ],
            dtype=np.float64,
        )
        reconstructions = {
            "genetic_rhs": np.sum(
                values["rhs_numerator_contributions"], axis=0, dtype=np.float64
            )
            / component_masses,
            "genetic_traces": np.sum(
                values["trace_numerator_contributions"], axis=0, dtype=np.float64
            )
            / component_masses,
            "genetic_residual": np.sum(
                values["genetic_residual_numerator_contributions"],
                axis=0,
                dtype=np.float64,
            )
            / component_masses[:, None],
        }
        for name, reconstructed in reconstructions.items():
            scale = np.maximum(
                1.0, np.maximum(np.abs(reconstructed), np.abs(values[name]))
            )
            if (
                np.max(
                    np.abs(reconstructed - values[name]) / scale,
                    initial=0.0,
                )
                > 1.0e-10
            ):
                raise ValueError(
                    f"Grouped trait-summary contributions do not reconstruct {name}."
                )
        return GroupedContextTraitSummary(
            manifest={
                key: value
                for key, value in payload.items()
                if key not in {"artifact", "performance"}
            },
            component_index=component_index,
            residual_names=tuple(payload["residual_order"]),
            annotation_masses=values["annotation_masses"],
            genetic_rhs=values["genetic_rhs"],
            genetic_traces=values["genetic_traces"],
            genetic_residual=values["genetic_residual"],
            residual_rhs=values["residual_rhs"],
            residual_traces=values["residual_traces"],
            residual_gram=values["residual_gram"],
            rhs_numerator_contributions=values["rhs_numerator_contributions"],
            trace_numerator_contributions=values["trace_numerator_contributions"],
            genetic_residual_numerator_contributions=values[
                "genetic_residual_numerator_contributions"
            ],
            loo_group_ids=labels,
            group_annotation_masses=values["group_annotation_masses"],
            group_variant_counts=values["group_variant_counts"].astype(
                np.int64, copy=False
            ),
            phase_times_seconds=dict(phase.get("phase_times_seconds", {})),
            peak_rss_bytes=int(phase.get("peak_rss_bytes", 0)),
            decode_passes=int(payload["backend"]["decode_passes"]),
            decoded_blocks=int(phase.get("decoded_blocks", 0)),
        )
    return ContextTraitSummary(
        manifest={
            key: value
            for key, value in payload.items()
            if key not in {"artifact", "performance"}
        },
        component_index=component_index,
        residual_names=tuple(payload["residual_order"]),
        annotation_weights=values["annotation_weights"],
        annotation_masses=values["annotation_masses"],
        genetic_rhs=values["genetic_rhs"],
        genetic_traces=values["genetic_traces"],
        genetic_residual=values["genetic_residual"],
        residual_rhs=values["residual_rhs"],
        residual_traces=values["residual_traces"],
        residual_gram=values["residual_gram"],
        rhs_numerator_contributions=values["rhs_numerator_contributions"],
        trace_numerator_contributions=values["trace_numerator_contributions"],
        genetic_residual_numerator_contributions=values[
            "genetic_residual_numerator_contributions"
        ],
        loo_group_ids=tuple(str(value) for value in values["loo_group_ids"]),
        phase_times_seconds=dict(phase.get("phase_times_seconds", {})),
        peak_rss_bytes=int(phase.get("peak_rss_bytes", 0)),
        decode_passes=int(payload["backend"]["decode_passes"]),
        decoded_blocks=int(phase.get("decoded_blocks", 0)),
    )
