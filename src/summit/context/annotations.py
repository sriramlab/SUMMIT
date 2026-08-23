"""Strict disjoint-annotation layer for contextual covariance.

The first public annotation mode is deliberately narrow: every retained
variant belongs to exactly one ordered binary partition.  Moment artifacts use
losslessly grouped approximate-LOO numerators and retain no variant-length
scientific arrays.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fit import (
    ContextFitResult,
    assemble_context_normal_equations,
    fit_context_model,
    validate_fit_compatibility,
)
from .oracle import coefficients_to_omegas, context_covariance_surface
from .reference import (
    ContextReference,
    GroupedContextReference,
    build_context_reference,
)
from .spec import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
    freeze_context_mapping,
    freeze_context_value,
    owned_readonly_array,
)
from .summary import (
    ContextTraitSummary,
    GroupedContextTraitSummary,
    build_context_trait_summary,
)


ANNOTATION_PARTITION_KIND = "summit.context.annotation_partition"
ANNOTATION_PARTITION_SCHEMA_VERSION = 1
DISJOINT_ANNOTATION_MODE = "disjoint_partition"


def _sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest.") from exc
    return value


def _names(values: Sequence[Any]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError("Annotation names must be nonempty strings.")
    if len(set(result)) != len(result):
        raise ValueError("Annotation names must be unique.")
    return result


def _groups(values: Sequence[Any] | None, n_variants: int) -> tuple[str, ...]:
    if values is None:
        raise ValueError(
            "Disjoint annotation artifacts require explicitly frozen LOO groups."
        )
    if len(values) != n_variants:
        raise ValueError("LOO group count must equal the retained variant count.")
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError("LOO group labels must be nonempty.")
    if len(set(result)) < 2:
        raise ValueError("At least two frozen LOO groups are required.")
    return result


def _group_layout(
    group_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    labels = tuple(dict.fromkeys(group_ids))
    lookup = {label: index for index, label in enumerate(labels)}
    indices = np.fromiter(
        (lookup[label] for label in group_ids),
        dtype=np.int64,
        count=len(group_ids),
    )
    counts = np.bincount(indices, minlength=len(labels)).astype(np.int64, copy=False)
    return labels, indices, counts


@dataclass(frozen=True)
class AnnotationGroupBalance:
    group_labels: tuple[str, ...]
    group_variant_counts: np.ndarray
    group_annotation_masses: np.ndarray
    minimum_group_variants: int
    maximum_group_variants: int
    total_groups_balanced: bool
    minimum_retained_annotation_mass: np.ndarray
    maximum_deleted_mass_share: np.ndarray
    effective_group_count: np.ndarray
    zero_support_group_counts: np.ndarray
    every_deletion_retains_each_annotation: bool
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_labels", tuple(self.group_labels))
        for name in (
            "group_variant_counts",
            "group_annotation_masses",
            "minimum_retained_annotation_mass",
            "maximum_deleted_mass_share",
            "effective_group_count",
            "zero_support_group_counts",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))


@dataclass(frozen=True)
class DisjointAnnotationPartition:
    manifest: dict[str, Any]
    annotation_names: tuple[str, ...]
    definitions: tuple[dict[str, Any], ...]
    weights: np.ndarray
    masses: np.ndarray
    loo_group_ids: tuple[str, ...]
    group_labels: tuple[str, ...]
    group_indices: np.ndarray
    group_annotation_masses: np.ndarray
    group_variant_counts: np.ndarray
    balance: AnnotationGroupBalance

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", freeze_context_mapping(self.manifest))
        object.__setattr__(self, "annotation_names", tuple(self.annotation_names))
        object.__setattr__(
            self,
            "definitions",
            tuple(freeze_context_value(value) for value in self.definitions),
        )
        object.__setattr__(self, "loo_group_ids", tuple(self.loo_group_ids))
        object.__setattr__(self, "group_labels", tuple(self.group_labels))
        for name in (
            "weights",
            "masses",
            "group_indices",
            "group_annotation_masses",
            "group_variant_counts",
        ):
            object.__setattr__(self, name, owned_readonly_array(getattr(self, name)))

    @property
    def n_variants(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_annotations(self) -> int:
        return int(self.weights.shape[1])

    @property
    def digest(self) -> str:
        return str(self.manifest["annotation_partition_hash"])

    @property
    def annotations(self) -> np.ndarray:
        return self.weights

    @property
    def annotation_masses(self) -> np.ndarray:
        return self.masses

    @property
    def group_balance(self) -> AnnotationGroupBalance:
        return self.balance

    def component_index(self, q: int) -> ContextComponentIndex:
        return ContextComponentIndex(self.annotation_names, ContextPairIndex(q))

    def without_variant_data(self) -> dict[str, Any]:
        """Return the aggregate descriptor safe to retain beside summaries."""
        return {
            "manifest": dict(self.manifest),
            "annotation_names": list(self.annotation_names),
            "masses": self.masses.tolist(),
            "group_labels": list(self.group_labels),
            "group_annotation_masses": self.group_annotation_masses.tolist(),
            "group_variant_counts": self.group_variant_counts.tolist(),
        }


def _balance(
    labels: tuple[str, ...],
    counts: np.ndarray,
    group_masses: np.ndarray,
    masses: np.ndarray,
) -> AnnotationGroupBalance:
    retained = masses[None, :] - group_masses
    minimum_retained = np.min(retained, axis=0)
    shares = group_masses / masses[None, :]
    effective = 1.0 / np.sum(shares * shares, axis=0)
    zero_support = np.sum(group_masses == 0.0, axis=0, dtype=np.int64)
    balanced = int(np.max(counts) - np.min(counts)) <= 1
    positive = bool(np.all(minimum_retained > 0.0))
    status = (
        "balanced_all_deletions_estimable"
        if balanced and positive
        else "unbalanced_or_annotation_deletion_failure"
    )
    return AnnotationGroupBalance(
        group_labels=labels,
        group_variant_counts=counts,
        group_annotation_masses=group_masses,
        minimum_group_variants=int(np.min(counts)),
        maximum_group_variants=int(np.max(counts)),
        total_groups_balanced=balanced,
        minimum_retained_annotation_mass=minimum_retained,
        maximum_deleted_mass_share=np.max(shares, axis=0),
        effective_group_count=effective,
        zero_support_group_counts=zero_support,
        every_deletion_retains_each_annotation=positive,
        status=status,
    )


def _definition_payload(
    annotation_names: tuple[str, ...], definitions: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    return {
        "mode": DISJOINT_ANNOTATION_MODE,
        "ordered_bins": [
            {"name": name, "definition": definition}
            for name, definition in zip(annotation_names, definitions)
        ],
        "boundary_convention": "lower_inclusive_upper_exclusive_final_upper_closed",
        "unassigned_policy": "error",
    }


def build_disjoint_annotation_partition(
    annotations: object,
    annotation_names: Sequence[Any],
    *,
    definitions: Sequence[Mapping[str, Any]] | None = None,
    variant_hash: str,
    loo_groups: Sequence[Any] | None,
    source: str,
    source_digest: str | None = None,
    require_balanced: bool = True,
) -> DisjointAnnotationPartition:
    """Validate and freeze an ordered exactly-one binary partition."""
    names = _names(annotation_names)
    weights = np.asarray(annotations, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(names) or weights.shape[0] < 1:
        raise ValueError("Annotation matrix shape does not match ordered names.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Annotation partition contains non-finite values.")
    if not np.all((weights == 0.0) | (weights == 1.0)):
        raise ValueError("Disjoint annotation weights must be exactly binary 0/1.")
    support = np.sum(weights, axis=1)
    unassigned = int(np.sum(support == 0.0))
    multiple = int(np.sum(support > 1.0))
    if unassigned or multiple or not np.all(support == 1.0):
        raise ValueError(
            "Every retained variant must belong to exactly one annotation; "
            f"unassigned={unassigned}, multiply_assigned={multiple}."
        )
    masses = np.sum(weights, axis=0, dtype=np.float64)
    if np.any(masses <= 0.0):
        raise ValueError("Every disjoint annotation must have positive mass.")
    groups = _groups(loo_groups, weights.shape[0])
    labels, indices, counts = _group_layout(groups)
    group_masses = np.zeros((len(labels), len(names)), dtype=np.float64)
    np.add.at(group_masses, indices, weights)
    balance = _balance(labels, counts, group_masses, masses)
    if require_balanced and not balance.total_groups_balanced:
        raise ValueError(
            "Equal-weight annotation jackknife requires total group sizes that "
            "differ by at most one."
        )
    if not balance.every_deletion_retains_each_annotation:
        raise ValueError("A frozen LOO deletion empties at least one annotation.")
    _sha256("variant_hash", variant_hash)
    if not isinstance(source, str) or not source:
        raise ValueError("Annotation source must be a nonempty declaration.")
    source_hash = (
        canonical_sha256({"source": source})
        if source_digest is None
        else _sha256("source_digest", source_digest)
    )
    if definitions is None:
        definitions_tuple = tuple({"declared_name": name} for name in names)
    else:
        if len(definitions) != len(names):
            raise ValueError("Annotation definitions must match ordered names.")
        definitions_tuple = tuple(dict(value) for value in definitions)
    definition_payload = _definition_payload(names, definitions_tuple)
    definition_hash = canonical_sha256(definition_payload)
    membership_hash = array_sha256(weights)
    grouping_hash = canonical_sha256({"groups": list(groups)})
    identity = {
        "kind": ANNOTATION_PARTITION_KIND,
        "schema_version": ANNOTATION_PARTITION_SCHEMA_VERSION,
        "method_version": "strict_binary_disjoint_v1",
        "mode": DISJOINT_ANNOTATION_MODE,
        "variant_hash": variant_hash,
        "annotation_definition_hash": definition_hash,
        "annotation_membership_hash": membership_hash,
        "loo_grouping_hash": grouping_hash,
        "source_digest": source_hash,
    }
    partition_hash = canonical_sha256(identity)
    manifest: dict[str, Any] = {
        **identity,
        "annotation_partition_hash": partition_hash,
        "source": source,
        "annotation_names": list(names),
        "ordered_bins": definition_payload["ordered_bins"],
        "boundary_convention": definition_payload["boundary_convention"],
        "unassigned_policy": "error",
        "dimensions": {
            "n_variants": int(weights.shape[0]),
            "k": len(names),
            "loo_groups": len(labels),
        },
        "masses": masses.tolist(),
        "unassigned_variants": 0,
        "multiply_assigned_variants": 0,
        "group_labels": list(labels),
        "group_variant_counts": counts.tolist(),
        "group_annotation_masses_hash": array_sha256(group_masses),
        "balance": {
            "minimum_group_variants": balance.minimum_group_variants,
            "maximum_group_variants": balance.maximum_group_variants,
            "total_groups_balanced": balance.total_groups_balanced,
            "minimum_retained_annotation_mass": (
                balance.minimum_retained_annotation_mass.tolist()
            ),
            "maximum_deleted_mass_share": (balance.maximum_deleted_mass_share.tolist()),
            "effective_group_count": balance.effective_group_count.tolist(),
            "zero_support_group_counts": balance.zero_support_group_counts.tolist(),
            "every_deletion_retains_each_annotation": (
                balance.every_deletion_retains_each_annotation
            ),
            "status": balance.status,
        },
    }
    result = DisjointAnnotationPartition(
        manifest=manifest,
        annotation_names=names,
        definitions=definitions_tuple,
        weights=weights,
        masses=masses,
        loo_group_ids=groups,
        group_labels=labels,
        group_indices=indices,
        group_annotation_masses=group_masses,
        group_variant_counts=counts,
        balance=balance,
    )
    validate_disjoint_annotation_partition(result)
    return result


def _edges(name: str, values: Sequence[float]) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError(f"{name} edges must contain at least two finite values.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"{name} edges must be strictly increasing.")
    return edges


def build_maf_ld_partition(
    maf: object,
    ld_score: object,
    *,
    maf_edges: Sequence[float],
    ld_edges: Sequence[float],
    variant_hash: str,
    loo_groups: Sequence[Any] | None,
    source: str,
    source_digest: str | None = None,
    require_balanced: bool = True,
) -> DisjointAnnotationPartition:
    """Build ordered Cartesian MAF--LD bins from frozen numeric boundaries."""
    maf_values = np.asarray(maf, dtype=np.float64)
    ld_values = np.asarray(ld_score, dtype=np.float64)
    if maf_values.ndim != 1 or ld_values.shape != maf_values.shape:
        raise ValueError("MAF and LD-score arrays must be equally sized vectors.")
    if not np.all(np.isfinite(maf_values)) or not np.all(np.isfinite(ld_values)):
        raise ValueError("MAF and LD-score inputs must be finite.")
    if np.any((maf_values < 0.0) | (maf_values > 0.5)):
        raise ValueError("MAF values must lie in the range [0,0.5].")
    maf_breaks = _edges("MAF", maf_edges)
    ld_breaks = _edges("LD", ld_edges)
    if np.any((maf_values < maf_breaks[0]) | (maf_values > maf_breaks[-1])):
        raise ValueError("A MAF value lies outside the frozen boundaries.")
    if np.any((ld_values < ld_breaks[0]) | (ld_values > ld_breaks[-1])):
        raise ValueError("An LD-score value lies outside the frozen boundaries.")
    maf_bin = np.searchsorted(maf_breaks, maf_values, side="right") - 1
    ld_bin = np.searchsorted(ld_breaks, ld_values, side="right") - 1
    maf_bin = np.minimum(maf_bin, maf_breaks.size - 2)
    ld_bin = np.minimum(ld_bin, ld_breaks.size - 2)
    names: list[str] = []
    definitions: list[dict[str, Any]] = []
    columns: list[np.ndarray] = []
    for maf_index in range(maf_breaks.size - 1):
        for ld_index in range(ld_breaks.size - 1):
            names.append(f"maf{maf_index}__ld{ld_index}")
            definitions.append(
                {
                    "maf": {
                        "lower": float(maf_breaks[maf_index]),
                        "upper": float(maf_breaks[maf_index + 1]),
                        "upper_closed": maf_index == maf_breaks.size - 2,
                    },
                    "ld_score": {
                        "lower": float(ld_breaks[ld_index]),
                        "upper": float(ld_breaks[ld_index + 1]),
                        "upper_closed": ld_index == ld_breaks.size - 2,
                    },
                }
            )
            columns.append((maf_bin == maf_index) & (ld_bin == ld_index))
    weights = np.column_stack(columns).astype(np.float64, copy=False)
    return build_disjoint_annotation_partition(
        weights,
        names,
        definitions=definitions,
        variant_hash=variant_hash,
        loo_groups=loo_groups,
        source=source,
        source_digest=source_digest,
        require_balanced=require_balanced,
    )


def validate_disjoint_annotation_partition(
    partition: DisjointAnnotationPartition,
) -> dict[str, Any]:
    if not isinstance(partition, DisjointAnnotationPartition):
        raise TypeError("Expected a DisjointAnnotationPartition.")
    manifest = partition.manifest
    if manifest.get("kind") != ANNOTATION_PARTITION_KIND:
        raise ValueError("Invalid annotation-partition kind.")
    if manifest.get("schema_version") != ANNOTATION_PARTITION_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation-partition schema version.")
    if manifest.get("method_version") != "strict_binary_disjoint_v1":
        raise ValueError("Unsupported annotation-partition method version.")
    if manifest.get("mode") != DISJOINT_ANNOTATION_MODE:
        raise ValueError("Only strict disjoint annotation mode is supported.")
    weights = np.asarray(partition.weights)
    if weights.dtype != np.dtype(np.float64) or weights.ndim != 2:
        raise ValueError("Partition weights must be a float64 matrix.")
    if not np.all((weights == 0.0) | (weights == 1.0)):
        raise ValueError("Partition weights are not exactly binary.")
    if not np.all(np.sum(weights, axis=1) == 1.0):
        raise ValueError("Partition membership is not exactly one per variant.")
    dimensions = manifest.get("dimensions", {})
    if dimensions != {
        "n_variants": weights.shape[0],
        "k": weights.shape[1],
        "loo_groups": len(partition.group_labels),
    }:
        raise ValueError("Partition dimensions are inconsistent.")
    if tuple(manifest.get("annotation_names", ())) != partition.annotation_names:
        raise ValueError("Partition annotation order is inconsistent.")
    recomputed_masses = np.sum(weights, axis=0, dtype=np.float64)
    if not np.array_equal(recomputed_masses, partition.masses):
        raise ValueError("Partition annotation masses are inconsistent.")
    if manifest.get("masses") != recomputed_masses.tolist():
        raise ValueError("Partition manifest masses are inconsistent.")
    if array_sha256(weights) != manifest.get("annotation_membership_hash"):
        raise ValueError("Partition membership hash mismatch.")
    if canonical_sha256({"groups": list(partition.loo_group_ids)}) != manifest.get(
        "loo_grouping_hash"
    ):
        raise ValueError("Partition LOO grouping hash mismatch.")
    if array_sha256(partition.group_annotation_masses) != manifest.get(
        "group_annotation_masses_hash"
    ):
        raise ValueError("Partition group-mass hash mismatch.")
    labels, indices, counts = _group_layout(partition.loo_group_ids)
    if labels != partition.group_labels or not np.array_equal(
        indices, partition.group_indices
    ):
        raise ValueError("Partition group layout is inconsistent.")
    reconstructed = np.zeros_like(partition.group_annotation_masses)
    np.add.at(reconstructed, indices, weights)
    if not np.array_equal(reconstructed, partition.group_annotation_masses):
        raise ValueError("Partition group masses do not reconstruct membership.")
    if not np.array_equal(
        counts,
        partition.group_variant_counts,
    ):
        raise ValueError("Partition group counts do not reconstruct membership.")
    if (
        manifest.get("group_labels") != list(labels)
        or manifest.get("group_variant_counts") != counts.tolist()
    ):
        raise ValueError("Partition manifest group layout is inconsistent.")
    expected_balance = _balance(labels, counts, reconstructed, recomputed_masses)
    observed_balance = partition.balance
    scalar_balance = (
        "minimum_group_variants",
        "maximum_group_variants",
        "total_groups_balanced",
        "every_deletion_retains_each_annotation",
        "status",
    )
    if any(
        getattr(expected_balance, name) != getattr(observed_balance, name)
        for name in scalar_balance
    ):
        raise ValueError("Partition balance diagnostics are inconsistent.")
    for name in (
        "group_variant_counts",
        "group_annotation_masses",
        "minimum_retained_annotation_mass",
        "maximum_deleted_mass_share",
        "effective_group_count",
        "zero_support_group_counts",
    ):
        if not np.allclose(
            getattr(expected_balance, name),
            getattr(observed_balance, name),
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError("Partition balance arrays are inconsistent.")
    expected_balance_payload = {
        "minimum_group_variants": expected_balance.minimum_group_variants,
        "maximum_group_variants": expected_balance.maximum_group_variants,
        "total_groups_balanced": expected_balance.total_groups_balanced,
        "minimum_retained_annotation_mass": (
            expected_balance.minimum_retained_annotation_mass.tolist()
        ),
        "maximum_deleted_mass_share": (
            expected_balance.maximum_deleted_mass_share.tolist()
        ),
        "effective_group_count": expected_balance.effective_group_count.tolist(),
        "zero_support_group_counts": (
            expected_balance.zero_support_group_counts.tolist()
        ),
        "every_deletion_retains_each_annotation": (
            expected_balance.every_deletion_retains_each_annotation
        ),
        "status": expected_balance.status,
    }
    if manifest.get("balance") != expected_balance_payload:
        raise ValueError("Partition manifest balance diagnostics are inconsistent.")
    definition_payload = _definition_payload(
        partition.annotation_names, partition.definitions
    )
    if manifest.get("ordered_bins") != definition_payload["ordered_bins"]:
        raise ValueError("Partition ordered-bin definitions are inconsistent.")
    if manifest.get("boundary_convention") != definition_payload["boundary_convention"]:
        raise ValueError("Partition boundary convention is inconsistent.")
    if manifest.get("unassigned_policy") != definition_payload["unassigned_policy"]:
        raise ValueError("Partition unassigned policy is inconsistent.")
    definition_hash = canonical_sha256(definition_payload)
    if definition_hash != manifest.get("annotation_definition_hash"):
        raise ValueError("Partition definition hash mismatch.")
    identity = {
        key: manifest.get(key)
        for key in (
            "kind",
            "schema_version",
            "method_version",
            "mode",
            "variant_hash",
            "annotation_definition_hash",
            "annotation_membership_hash",
            "loo_grouping_hash",
            "source_digest",
        )
    }
    if canonical_sha256(identity) != manifest.get("annotation_partition_hash"):
        raise ValueError("Partition identity hash mismatch.")
    return {
        "status": "valid_strict_disjoint_partition",
        "exactly_one": True,
        "balanced": partition.balance.total_groups_balanced,
        "all_deletions_estimable": (
            partition.balance.every_deletion_retains_each_annotation
        ),
    }


def _bind_partition_manifest(
    manifest: Mapping[str, Any], partition: DisjointAnnotationPartition
) -> dict[str, Any]:
    result = dict(manifest)
    result.update(
        {
            "annotation_mode": DISJOINT_ANNOTATION_MODE,
            "annotation_definition_hash": partition.manifest[
                "annotation_definition_hash"
            ],
            "annotation_membership_hash": partition.manifest[
                "annotation_membership_hash"
            ],
            "annotation_partition_hash": partition.digest,
            "annotation_source_digest": partition.manifest["source_digest"],
            "loo_grouping_hash": partition.manifest["loo_grouping_hash"],
        }
    )
    return result


def _validate_partition_binding(
    manifest: Mapping[str, Any], partition: DisjointAnnotationPartition
) -> None:
    validate_disjoint_annotation_partition(partition)
    expected = {
        "annotation_mode": DISJOINT_ANNOTATION_MODE,
        "annotation_definition_hash": partition.manifest["annotation_definition_hash"],
        "annotation_membership_hash": partition.manifest["annotation_membership_hash"],
        "annotation_partition_hash": partition.digest,
        "loo_grouping_hash": partition.manifest["loo_grouping_hash"],
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("variant_hash") != partition.manifest.get("variant_hash"):
        mismatches.append("variant_hash")
    if mismatches:
        raise ValueError(
            "Grouped artifact and partition identities differ: "
            + ", ".join(sorted(set(mismatches)))
            + "."
        )


def group_context_reference(
    reference: ContextReference | GroupedContextReference,
    partition: DisjointAnnotationPartition | None = None,
) -> GroupedContextReference:
    """Losslessly aggregate a legacy SNP-contribution reference by frozen group."""
    if isinstance(reference, GroupedContextReference):
        if partition is not None:
            _validate_partition_binding(reference.manifest, partition)
        return reference
    groups = reference.loo_group_ids if partition is None else partition.loo_group_ids
    if len(groups) != reference.n_variants:
        raise ValueError("Partition and reference variant counts differ.")
    labels, indices, counts = _group_layout(groups)
    contributions = np.zeros(
        (len(labels),) + reference.gram_numerator_contributions.shape[1:],
        dtype=np.float64,
    )
    np.add.at(contributions, indices, reference.gram_numerator_contributions)
    if partition is None:
        group_masses = np.zeros(
            (len(labels), reference.annotation_masses.size), dtype=np.float64
        )
        np.add.at(group_masses, indices, reference.annotation_weights)
    else:
        validate_disjoint_annotation_partition(partition)
        if reference.manifest.get("variant_hash") != partition.manifest.get(
            "variant_hash"
        ):
            raise ValueError("Partition and reference ordered variants differ.")
        if partition.annotation_names != reference.component_index.annotation_names:
            raise ValueError("Partition and reference annotation orders differ.")
        if array_sha256(reference.annotation_weights) != partition.manifest.get(
            "annotation_membership_hash"
        ):
            raise ValueError("Partition and reference annotation membership differ.")
        group_masses = partition.group_annotation_masses.copy()
    manifest = dict(reference.manifest)
    manifest["dimensions"] = {
        **manifest["dimensions"],
        "loo_groups": len(labels),
    }
    manifest["approximate_loo"] = {
        **manifest["approximate_loo"],
        "method": "symmetric_grouped_gram_numerator_v1",
        "contribution_storage": "loo_grouped",
        "group_labels": list(labels),
        "group_variant_counts": counts.tolist(),
        "group_annotation_masses_hash": array_sha256(group_masses),
    }
    if partition is not None:
        manifest = _bind_partition_manifest(manifest, partition)
    return GroupedContextReference(
        manifest=manifest,
        component_index=reference.component_index,
        annotation_masses=reference.annotation_masses.copy(),
        gram=reference.gram.copy(),
        same_person=reference.same_person.copy(),
        gram_numerator_contributions=contributions,
        loo_group_ids=labels,
        group_annotation_masses=group_masses,
        group_variant_counts=counts,
        phase_times_seconds=dict(reference.phase_times_seconds),
        peak_rss_bytes=reference.peak_rss_bytes,
    )


def group_context_trait_summary(
    summary: ContextTraitSummary | GroupedContextTraitSummary,
    partition: DisjointAnnotationPartition | None = None,
) -> GroupedContextTraitSummary:
    """Losslessly aggregate a legacy SNP-contribution trait summary."""
    if isinstance(summary, GroupedContextTraitSummary):
        if partition is not None:
            _validate_partition_binding(summary.manifest, partition)
        return summary
    groups = summary.loo_group_ids if partition is None else partition.loo_group_ids
    if len(groups) != summary.n_variants:
        raise ValueError("Partition and trait-summary variant counts differ.")
    labels, indices, counts = _group_layout(groups)

    def aggregate(values: np.ndarray) -> np.ndarray:
        result = np.zeros((len(labels),) + values.shape[1:], dtype=np.float64)
        np.add.at(result, indices, values)
        return result

    if partition is None:
        group_masses = np.zeros(
            (len(labels), summary.annotation_masses.size), dtype=np.float64
        )
        np.add.at(group_masses, indices, summary.annotation_weights)
    else:
        validate_disjoint_annotation_partition(partition)
        if summary.manifest.get("variant_hash") != partition.manifest.get(
            "variant_hash"
        ):
            raise ValueError("Partition and trait ordered variants differ.")
        if partition.annotation_names != summary.component_index.annotation_names:
            raise ValueError("Partition and trait annotation orders differ.")
        if array_sha256(summary.annotation_weights) != partition.manifest.get(
            "annotation_membership_hash"
        ):
            raise ValueError("Partition and trait annotation membership differ.")
        group_masses = partition.group_annotation_masses.copy()
    manifest = dict(summary.manifest)
    manifest["dimensions"] = {
        **manifest["dimensions"],
        "loo_groups": len(labels),
    }
    manifest["approximate_loo"] = {
        **manifest["approximate_loo"],
        "method": "grouped_contribution_summary_v1",
        "contribution_storage": "loo_grouped",
        "group_labels": list(labels),
        "group_variant_counts": counts.tolist(),
        "group_annotation_masses_hash": array_sha256(group_masses),
    }
    if partition is not None:
        manifest = _bind_partition_manifest(manifest, partition)
    return GroupedContextTraitSummary(
        manifest=manifest,
        component_index=summary.component_index,
        residual_names=summary.residual_names,
        annotation_masses=summary.annotation_masses.copy(),
        genetic_rhs=summary.genetic_rhs.copy(),
        genetic_traces=summary.genetic_traces.copy(),
        genetic_residual=summary.genetic_residual.copy(),
        residual_rhs=summary.residual_rhs.copy(),
        residual_traces=summary.residual_traces.copy(),
        residual_gram=summary.residual_gram.copy(),
        rhs_numerator_contributions=aggregate(summary.rhs_numerator_contributions),
        trace_numerator_contributions=aggregate(summary.trace_numerator_contributions),
        genetic_residual_numerator_contributions=aggregate(
            summary.genetic_residual_numerator_contributions
        ),
        loo_group_ids=labels,
        group_annotation_masses=group_masses,
        group_variant_counts=counts,
        phase_times_seconds=dict(summary.phase_times_seconds),
        peak_rss_bytes=summary.peak_rss_bytes,
        decode_passes=summary.decode_passes,
        decoded_blocks=summary.decoded_blocks,
    )


def build_grouped_context_reference(
    *, partition: DisjointAnnotationPartition, **kwargs: Any
) -> GroupedContextReference:
    validate_disjoint_annotation_partition(partition)
    if "annotations" in kwargs or "loo_groups" in kwargs:
        raise ValueError("Partition supplies annotations and frozen LOO groups.")
    if kwargs.get("variant_hash") != partition.manifest.get("variant_hash"):
        raise ValueError("Partition and reference ordered variants differ.")
    component_index = kwargs.get("component_index")
    if component_index is None:
        basis = np.asarray(kwargs.get("basis"))
        if basis.ndim != 2:
            raise ValueError("basis is required to infer the component index.")
        component_index = partition.component_index(int(basis.shape[1]))
        kwargs["component_index"] = component_index
    if not isinstance(component_index, ContextComponentIndex):
        raise ValueError("component_index must be a ContextComponentIndex.")
    if component_index.annotation_names != partition.annotation_names:
        raise ValueError("Component and partition annotation orders differ.")
    result = build_context_reference(
        annotations=partition.weights,
        loo_groups=partition.loo_group_ids,
        contribution_storage="loo_grouped",
        **kwargs,
    )
    if not isinstance(result, GroupedContextReference):
        raise RuntimeError("Grouped reference builder returned an invalid object.")
    return replace(
        result, manifest=_bind_partition_manifest(result.manifest, partition)
    )


def build_grouped_context_trait_summary(
    *, partition: DisjointAnnotationPartition, **kwargs: Any
) -> GroupedContextTraitSummary:
    validate_disjoint_annotation_partition(partition)
    if "annotations" in kwargs or "loo_groups" in kwargs:
        raise ValueError("Partition supplies annotations and frozen LOO groups.")
    if kwargs.get("variant_hash") != partition.manifest.get("variant_hash"):
        raise ValueError("Partition and trait ordered variants differ.")
    component_index = kwargs.get("component_index")
    if component_index is None:
        basis = np.asarray(kwargs.get("basis"))
        if basis.ndim != 2:
            raise ValueError("basis is required to infer the component index.")
        component_index = partition.component_index(int(basis.shape[1]))
        kwargs["component_index"] = component_index
    if not isinstance(component_index, ContextComponentIndex):
        raise ValueError("component_index must be a ContextComponentIndex.")
    if component_index.annotation_names != partition.annotation_names:
        raise ValueError("Component and partition annotation orders differ.")
    result = build_context_trait_summary(
        annotations=partition.weights,
        loo_groups=partition.loo_group_ids,
        contribution_storage="loo_grouped",
        **kwargs,
    )
    if not isinstance(result, GroupedContextTraitSummary):
        raise RuntimeError("Grouped trait builder returned an invalid object.")
    return replace(
        result, manifest=_bind_partition_manifest(result.manifest, partition)
    )


def fit_annotation_context_model(
    reference: GroupedContextReference,
    summary: GroupedContextTraitSummary,
    *,
    partition: DisjointAnnotationPartition | None = None,
    **kwargs: Any,
) -> ContextFitResult:
    required_identity = (
        "annotation_mode",
        "annotation_definition_hash",
        "annotation_membership_hash",
        "annotation_partition_hash",
    )
    for artifact in (reference, summary):
        if artifact.manifest.get("annotation_mode") != DISJOINT_ANNOTATION_MODE or any(
            not artifact.manifest.get(key) for key in required_identity[1:]
        ):
            raise ValueError(
                "Annotation fitting requires artifacts already bound to a strict "
                "disjoint partition."
            )
    mismatches = [
        key
        for key in required_identity
        if reference.manifest.get(key) != summary.manifest.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Reference and trait annotation identities differ: "
            + ", ".join(mismatches)
            + "."
        )
    if partition is not None:
        for artifact in (reference, summary):
            _validate_partition_binding(artifact.manifest, partition)
    if set(kwargs.get("loo_groups", reference.loo_group_ids)) != set(
        reference.loo_group_ids
    ):
        raise ValueError("Annotation fit requires every frozen LOO group.")
    if "annotations_disjoint" in kwargs and kwargs["annotations_disjoint"] is not True:
        raise ValueError("Annotation fit is restricted to strict disjoint mode.")
    kwargs["annotations_disjoint"] = True
    result = fit_context_model(reference, summary, **kwargs)
    manifest = {
        **result.manifest,
        "annotation_mode": DISJOINT_ANNOTATION_MODE,
        "annotation_definition_hash": reference.manifest.get(
            "annotation_definition_hash"
        ),
        "annotation_membership_hash": reference.manifest.get(
            "annotation_membership_hash"
        ),
        "annotation_partition_hash": reference.manifest.get(
            "annotation_partition_hash"
        ),
        "annotation_interpretation": {
            "omega": "total_bin_covariance_contribution",
            "per_annotation_mass": "omega_divided_by_annotation_mass",
            "overlapping_annotations": "unsupported",
        },
    }
    return replace(result, manifest=manifest)


def _jackknife_covariance(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        raise ValueError("At least two LOO values are required.")
    centered = values - np.mean(values, axis=0, keepdims=True)
    result = (values.shape[0] - 1.0) / values.shape[0] * (centered.T @ centered)
    return 0.5 * (result + result.T)


def _unpack_pairs(values: np.ndarray, pair_index: ContextPairIndex) -> np.ndarray:
    result = np.zeros((pair_index.num_basis, pair_index.num_basis), dtype=np.float64)
    for pair in pair_index.entries:
        result[pair.q, pair.r] = values[pair.index]
        result[pair.r, pair.q] = values[pair.index]
    return result


@dataclass(frozen=True)
class AnnotationContrastResult:
    left_annotation: str
    right_annotation: str
    scale: str
    coefficient_difference: np.ndarray
    omega_difference: np.ndarray
    loo_coefficient_differences: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    surface_difference: np.ndarray | None
    loo_surface_differences: np.ndarray | None
    trace_difference: float | None
    loo_trace_differences: np.ndarray | None
    trace_standard_error: float | None
    status: str
    manifest: dict[str, Any]

    @property
    def estimate(self) -> np.ndarray:
        return self.coefficient_difference

    @property
    def loo_values(self) -> np.ndarray:
        return self.loo_coefficient_differences

    @property
    def jackknife_covariance(self) -> np.ndarray:
        return self.covariance


@dataclass(frozen=True)
class AnnotationTotalResult:
    scale: str
    coefficient_total: np.ndarray
    omega_total: np.ndarray
    loo_coefficient_totals: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    surface_total: np.ndarray | None
    loo_surface_totals: np.ndarray | None
    trace_total: float | None
    loo_trace_totals: np.ndarray | None
    trace_standard_error: float | None
    status: str
    manifest: dict[str, Any]

    @property
    def estimate(self) -> np.ndarray:
        return self.coefficient_total

    @property
    def loo_values(self) -> np.ndarray:
        return self.loo_coefficient_totals

    @property
    def jackknife_covariance(self) -> np.ndarray:
        return self.covariance


def _annotation_pair_values(
    fit: ContextFitResult,
    annotation_index: int,
    scale: str,
    summary: GroupedContextTraitSummary | None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    pair_count = len(fit.component_index.pair_index)
    start = annotation_index * pair_count
    stop = start + pair_count
    mass = float(fit.equations.annotation_masses[annotation_index])
    if scale == "total_component":
        divisor = 1.0
        loo_divisors = np.ones(len(fit.jackknife_groups), dtype=np.float64)
    else:
        if summary is None:
            raise ValueError(
                "Per-annotation-mass LOO output requires the grouped trait summary."
            )
        group_lookup = {
            label: index for index, label in enumerate(summary.loo_group_ids)
        }
        try:
            deleted = np.asarray(
                [
                    summary.group_annotation_masses[
                        group_lookup[group], annotation_index
                    ]
                    for group in fit.jackknife_groups
                ],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise ValueError("Fit contains an unknown grouped LOO label.") from exc
        divisor = mass
        loo_divisors = mass - deleted
        if np.any(loo_divisors <= 0.0):
            raise ValueError(
                "A LOO replicate has nonpositive retained annotation mass."
            )
    return (
        fit.genetic_coefficients[start:stop] / divisor,
        fit.loo_coefficients[:, start:stop] / loo_divisors[:, None],
        divisor,
        loo_divisors,
    )


def _validate_contrast_scale(scale: str) -> None:
    if scale not in {"total_component", "per_annotation_mass"}:
        raise ValueError(
            "Annotation contrast scale must be 'total_component' or "
            "'per_annotation_mass'."
        )


def _validate_derived_artifact_binding(
    fit: ContextFitResult,
    reference: GroupedContextReference | None,
    summary: GroupedContextTraitSummary | None,
) -> None:
    if reference is not None and summary is None:
        raise ValueError(
            "A grouped reference used for annotation trace derivation requires "
            "its matching grouped trait summary."
        )
    if reference is not None and summary is not None:
        validate_fit_compatibility(reference, summary)
    identity_fields = (
        "annotation_mode",
        "annotation_definition_hash",
        "annotation_membership_hash",
        "annotation_partition_hash",
        "variant_hash",
        "annotation_hash",
        "component_index_hash",
        "loo_grouping_hash",
    )
    for label, artifact in (("reference", reference), ("summary", summary)):
        if artifact is None:
            continue
        mismatches = [
            field
            for field in identity_fields
            if fit.manifest.get(field) != artifact.manifest.get(field)
        ]
        if artifact.component_index.digest != fit.component_index.digest:
            mismatches.append("component_index")
        if mismatches:
            raise ValueError(
                f"Fit and grouped {label} identities differ: "
                + ", ".join(sorted(set(mismatches)))
                + "."
            )
    if summary is not None:
        if summary.residual_names != fit.residual_names:
            raise ValueError("Fit and grouped summary residual orders differ.")
        if set(summary.loo_group_ids) != set(fit.jackknife_groups):
            raise ValueError("Fit and grouped summary LOO group sets differ.")


def _loo_trace_vectors(
    fit: ContextFitResult,
    reference: GroupedContextReference | None,
    summary: GroupedContextTraitSummary | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if reference is None or summary is None:
        return None
    full = fit.equations.traces[: len(fit.component_index)]
    loo = np.vstack(
        [
            assemble_context_normal_equations(reference, summary, (group,)).traces[
                : len(fit.component_index)
            ]
            for group in fit.jackknife_groups
        ]
    )
    return full, loo


def derive_annotation_contrast(
    fit: ContextFitResult,
    left: str,
    right: str,
    *,
    scale: str = "total_component",
    reference: GroupedContextReference | None = None,
    summary: GroupedContextTraitSummary | None = None,
    context_grid: object | None = None,
    basis_metric: object | None = None,
) -> AnnotationContrastResult:
    del basis_metric  # Linear contrasts do not require an operator metric.
    _validate_contrast_scale(scale)
    _validate_derived_artifact_binding(fit, reference, summary)
    names = fit.component_index.annotation_names
    if left == right or left not in names or right not in names:
        raise ValueError("Contrast requires two distinct known annotations.")
    left_index = names.index(left)
    right_index = names.index(right)
    left_full, left_loo, left_divisor, left_loo_divisors = _annotation_pair_values(
        fit, left_index, scale, summary
    )
    right_full, right_loo, right_divisor, right_loo_divisors = _annotation_pair_values(
        fit, right_index, scale, summary
    )
    values = left_full - right_full
    loo = left_loo - right_loo
    covariance = _jackknife_covariance(loo)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    omega = _unpack_pairs(values, fit.component_index.pair_index)
    grid = None if context_grid is None else np.asarray(context_grid, dtype=np.float64)
    surface = None if grid is None else context_covariance_surface(omega, grid)
    loo_surfaces = (
        None
        if grid is None
        else np.stack(
            [
                context_covariance_surface(
                    _unpack_pairs(row, fit.component_index.pair_index), grid
                )
                for row in loo
            ]
        )
    )
    trace_payload = _loo_trace_vectors(fit, reference, summary)
    trace_difference: float | None = None
    loo_trace: np.ndarray | None = None
    trace_se: float | None = None
    trace_status = "unavailable_without_grouped_equations"
    if trace_payload is not None:
        full_traces, loo_traces = trace_payload
        pair_count = len(fit.component_index.pair_index)
        left_slice = slice(left_index * pair_count, (left_index + 1) * pair_count)
        right_slice = slice(right_index * pair_count, (right_index + 1) * pair_count)
        trace_difference = float(
            np.dot(
                fit.genetic_coefficients[left_slice] / left_divisor,
                full_traces[left_slice],
            )
            - np.dot(
                fit.genetic_coefficients[right_slice] / right_divisor,
                full_traces[right_slice],
            )
        )
        loo_trace = np.asarray(
            [
                np.dot(
                    fit.loo_coefficients[index, left_slice] / left_loo_divisors[index],
                    loo_traces[index, left_slice],
                )
                - np.dot(
                    fit.loo_coefficients[index, right_slice]
                    / right_loo_divisors[index],
                    loo_traces[index, right_slice],
                )
                for index in range(len(fit.jackknife_groups))
            ],
            dtype=np.float64,
        )
        trace_se = float(np.sqrt(_jackknife_covariance(loo_trace[:, None])[0, 0]))
        trace_status = "defined_with_deleted_group_traces"
    return AnnotationContrastResult(
        left_annotation=left,
        right_annotation=right,
        scale=scale,
        coefficient_difference=values,
        omega_difference=omega,
        loo_coefficient_differences=loo,
        covariance=covariance,
        standard_errors=standard_errors,
        surface_difference=surface,
        loo_surface_differences=loo_surfaces,
        trace_difference=trace_difference,
        loo_trace_differences=loo_trace,
        trace_standard_error=trace_se,
        status=f"defined_linear_annotation_contrast;trace={trace_status}",
        manifest={
            "kind": "summit.context.annotation_contrast",
            "left": left,
            "right": right,
            "scale": scale,
            "interpretation": (
                "difference_in_total_bin_covariance_contributions"
                if scale == "total_component"
                else "difference_in_covariance_per_unit_annotation_mass"
            ),
            "uses_full_joint_loo_covariance": True,
            "trace_recomputed_per_deleted_group": trace_payload is not None,
            "annotation_partition_hash": fit.manifest.get("annotation_partition_hash"),
        },
    )


def derive_annotation_total(
    fit: ContextFitResult,
    *,
    scale: str = "total_component",
    reference: GroupedContextReference | None = None,
    summary: GroupedContextTraitSummary | None = None,
    context_grid: object | None = None,
    basis_metric: object | None = None,
) -> AnnotationTotalResult:
    del basis_metric
    _validate_contrast_scale(scale)
    _validate_derived_artifact_binding(fit, reference, summary)
    point: list[np.ndarray] = []
    loo_values: list[np.ndarray] = []
    divisors: list[float] = []
    loo_divisors: list[np.ndarray] = []
    for index in range(len(fit.component_index.annotation_names)):
        full, loo, divisor, loo_divisor = _annotation_pair_values(
            fit, index, scale, summary
        )
        point.append(full)
        loo_values.append(loo)
        divisors.append(divisor)
        loo_divisors.append(loo_divisor)
    total = np.sum(point, axis=0)
    loo_total = np.sum(loo_values, axis=0)
    covariance = _jackknife_covariance(loo_total)
    omega = _unpack_pairs(total, fit.component_index.pair_index)
    grid = None if context_grid is None else np.asarray(context_grid, dtype=np.float64)
    surface = None if grid is None else context_covariance_surface(omega, grid)
    loo_surfaces = (
        None
        if grid is None
        else np.stack(
            [
                context_covariance_surface(
                    _unpack_pairs(row, fit.component_index.pair_index), grid
                )
                for row in loo_total
            ]
        )
    )
    trace_payload = _loo_trace_vectors(fit, reference, summary)
    trace_total: float | None = None
    loo_trace: np.ndarray | None = None
    trace_se: float | None = None
    if trace_payload is not None:
        full_traces, loo_traces = trace_payload
        pair_count = len(fit.component_index.pair_index)
        trace_total = float(
            sum(
                np.dot(
                    fit.genetic_coefficients[
                        index * pair_count : (index + 1) * pair_count
                    ]
                    / divisors[index],
                    full_traces[index * pair_count : (index + 1) * pair_count],
                )
                for index in range(len(divisors))
            )
        )
        loo_trace = np.asarray(
            [
                sum(
                    np.dot(
                        fit.loo_coefficients[
                            replicate,
                            index * pair_count : (index + 1) * pair_count,
                        ]
                        / loo_divisors[index][replicate],
                        loo_traces[
                            replicate,
                            index * pair_count : (index + 1) * pair_count,
                        ],
                    )
                    for index in range(len(divisors))
                )
                for replicate in range(len(fit.jackknife_groups))
            ],
            dtype=np.float64,
        )
        trace_se = float(np.sqrt(_jackknife_covariance(loo_trace[:, None])[0, 0]))
    return AnnotationTotalResult(
        scale=scale,
        coefficient_total=total,
        omega_total=omega,
        loo_coefficient_totals=loo_total,
        covariance=covariance,
        standard_errors=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        surface_total=surface,
        loo_surface_totals=loo_surfaces,
        trace_total=trace_total,
        loo_trace_totals=loo_trace,
        trace_standard_error=trace_se,
        status=(
            "defined_total_with_deleted_group_traces"
            if trace_payload is not None
            else "defined_total_trace_unavailable_without_grouped_equations"
        ),
        manifest={
            "kind": "summit.context.annotation_total",
            "scale": scale,
            "interpretation": (
                "sum_of_total_bin_covariance_contributions"
                if scale == "total_component"
                else "sum_of_per_annotation_mass_coefficients_not_a_genome_total"
            ),
            "uses_full_joint_loo_covariance": True,
            "trace_recomputed_per_deleted_group": trace_payload is not None,
            "annotation_partition_hash": fit.manifest.get("annotation_partition_hash"),
        },
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_disjoint_annotation_partition(
    partition: DisjointAnnotationPartition, output_prefix: str | Path
) -> tuple[Path, Path]:
    validate_disjoint_annotation_partition(partition)
    prefix = Path(output_prefix)
    manifest_path = prefix.with_suffix(".context-annotations.json")
    arrays_path = prefix.with_suffix(".context-annotations.npz")
    if manifest_path.exists() or arrays_path.exists():
        raise FileExistsError("Annotation partition output already exists.")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{arrays_path.name}.", dir=arrays_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                weights=partition.weights,
                loo_group_ids=np.asarray(partition.loo_group_ids, dtype=np.str_),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, arrays_path)
        arrays_path.chmod(0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    payload = {
        **partition.manifest,
        "artifact": {
            "path": arrays_path.name,
            "format": "npz_strict_disjoint_v1",
            "sha256": array_sha256(
                np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)
            ),
        },
    }
    _atomic_write(manifest_path, json.dumps(payload, sort_keys=True, indent=2))
    return manifest_path, arrays_path


def load_disjoint_annotation_partition(
    manifest_path: str | Path, *, expected_partition_hash: str | None = None
) -> DisjointAnnotationPartition:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != ANNOTATION_PARTITION_KIND:
        raise ValueError("Invalid annotation-partition kind.")
    if payload.get("schema_version") != ANNOTATION_PARTITION_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation-partition schema version.")
    if payload.get("method_version") != "strict_binary_disjoint_v1":
        raise ValueError("Unsupported annotation-partition method version.")
    if payload.get("mode") != DISJOINT_ANNOTATION_MODE:
        raise ValueError("Unsupported annotation-partition mode.")
    if expected_partition_hash is not None and payload.get(
        "annotation_partition_hash"
    ) != _sha256("expected_partition_hash", expected_partition_hash):
        raise ValueError("Unexpected annotation-partition identity.")
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("format") != (
        "npz_strict_disjoint_v1"
    ):
        raise ValueError("Unsupported annotation-partition artifact.")
    arrays_path = path.parent / str(artifact.get("path"))
    observed = array_sha256(np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8))
    if observed != artifact.get("sha256"):
        raise ValueError("Annotation-partition artifact SHA-256 mismatch.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"weights", "loo_group_ids"}:
            raise ValueError("Annotation-partition artifact arrays are invalid.")
        weights = np.array(arrays["weights"], copy=True)
        groups = tuple(str(value) for value in arrays["loo_group_ids"])
    definitions = tuple(
        dict(value["definition"]) for value in payload.get("ordered_bins", ())
    )
    result = build_disjoint_annotation_partition(
        weights,
        payload.get("annotation_names", ()),
        definitions=definitions,
        variant_hash=str(payload.get("variant_hash")),
        loo_groups=groups,
        source=str(payload.get("source")),
        source_digest=str(payload.get("source_digest")),
        require_balanced=False,
    )
    for key in (
        "annotation_definition_hash",
        "annotation_membership_hash",
        "loo_grouping_hash",
        "annotation_partition_hash",
        "group_annotation_masses_hash",
    ):
        if result.manifest.get(key) != payload.get(key):
            raise ValueError(f"Annotation-partition {key} mismatch.")
    declared_manifest = dict(payload)
    declared_manifest.pop("artifact", None)
    if declared_manifest != result.manifest:
        raise ValueError(
            "Annotation-partition manifest claims do not match the canonical "
            "partition reconstructed from the artifact."
        )
    return result
