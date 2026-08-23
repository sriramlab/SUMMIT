"""Batched transformed-phenotype summaries and scale trajectories.

This is a correctness-first NumPy implementation.  It keeps transformation
validity separate from the common retained-individual mask, visits each
genotype block once, and stores the SNP contributions required by SUMMIT's
summary-only approximate delete-group jackknife.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import psutil

from .fit import (
    DEFAULT_SOLVE_RTOL,
    ContextJackknifeError,
    ContextNormalEquations,
    ContextRankError,
    _rank_diagnostics,
    assemble_context_normal_equations,
    validate_fit_compatibility,
)
from .oracle import (
    ProjectorResult,
    SymmetricRankDiagnostics,
    _finite_float64,
    coefficients_to_omegas,
    project_normalize_phenotype,
    residual_moments_low_rank,
    validate_annotations,
)
from .reference import ContextReference
from .spec import (
    CONTEXT_SCHEMA_VERSION,
    CONTEXT_TRAIT_KIND,
    RAW_PROJECTED_FEATURE_MODE,
    ContextComponentIndex,
    array_sha256,
    canonical_json,
    canonical_sha256,
)
from .summary import ContextTraitSummary


TRANSFORM_SCHEMA_VERSION = 1
TRANSFORM_SPEC_KIND = "summit.context.phenotype_transform"
TRANSFORM_SUMMARY_KIND = "summit.context.transform_summary"
TRANSFORM_FIT_KIND = "summit.context.transform_fit"

_TRANSFORM_KINDS = frozenset({"identity", "log", "box_cox", "user_supplied"})
_TRANSFORM_SUMMARY_ARRAY_NAMES = (
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
)


def _name(label: str, value: object) -> str:
    result = str(value)
    if not result or any(character.isspace() for character in result):
        raise ValueError(f"{label} must be a nonempty token without whitespace.")
    return result


def _sha256(name: str, value: object) -> str:
    result = str(value)
    if len(result) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest.") from exc
    return result


@dataclass(frozen=True)
class PhenotypeTransformSpec:
    """One declared transformation in a versioned scale scan."""

    transform_id: str
    kind: str
    shift: float | None = None
    box_cox_lambda: float | None = None
    source: str | None = None
    schema_version: int = TRANSFORM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _name("transform_id", self.transform_id)
        if self.kind not in _TRANSFORM_KINDS:
            raise ValueError(
                f"Unsupported phenotype transform {self.kind!r}; expected one of "
                f"{sorted(_TRANSFORM_KINDS)}."
            )
        if self.schema_version != TRANSFORM_SCHEMA_VERSION:
            raise ValueError("Unsupported phenotype-transform schema version.")
        if self.kind in {"log", "box_cox"}:
            if self.shift is None or not np.isfinite(float(self.shift)):
                raise ValueError(f"{self.kind} requires an explicit finite shift.")
        elif self.shift is not None:
            raise ValueError(f"{self.kind} does not accept a shift.")
        if self.kind == "box_cox":
            if self.box_cox_lambda is None or not np.isfinite(
                float(self.box_cox_lambda)
            ):
                raise ValueError("box_cox requires an explicit finite lambda.")
        elif self.box_cox_lambda is not None:
            raise ValueError(f"{self.kind} does not accept box_cox_lambda.")
        if self.kind == "user_supplied":
            if self.source is None:
                raise ValueError("user_supplied requires a source column name.")
            _name("user-supplied source", self.source)
        elif self.source is not None:
            raise ValueError(f"{self.kind} does not accept a source column.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": TRANSFORM_SPEC_KIND,
            "schema_version": self.schema_version,
            "transform_id": self.transform_id,
            "transform": self.kind,
        }
        if self.shift is not None:
            payload["shift"] = float(self.shift)
        if self.box_cox_lambda is not None:
            payload["lambda"] = float(self.box_cox_lambda)
        if self.source is not None:
            payload["source"] = self.source
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhenotypeTransformSpec":
        allowed = {
            "kind",
            "schema_version",
            "transform_id",
            "transform",
            "shift",
            "lambda",
            "source",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown phenotype-transform fields: {sorted(unknown)}.")
        if payload.get("kind") != TRANSFORM_SPEC_KIND:
            raise ValueError(f"Transform kind must be {TRANSFORM_SPEC_KIND!r}.")
        return cls(
            transform_id=str(payload.get("transform_id", "")),
            kind=str(payload.get("transform", "")),
            shift=(None if "shift" not in payload else float(payload["shift"])),
            box_cox_lambda=(
                None if "lambda" not in payload else float(payload["lambda"])
            ),
            source=(None if payload.get("source") is None else str(payload["source"])),
            schema_version=payload.get("schema_version"),
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TransformValidity:
    spec: PhenotypeTransformSpec
    valid: bool
    status: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "spec_hash": self.spec.digest,
            "valid": bool(self.valid),
            "status": self.status,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class ContextTransformSummary:
    """Shared moments plus one phenotype-specific RHS for each valid transform."""

    manifest: dict[str, Any]
    transform_records: tuple[TransformValidity, ...]
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
    maximum_rhs_columns: int

    @property
    def transform_ids(self) -> tuple[str, ...]:
        return tuple(
            record.spec.transform_id
            for record in self.transform_records
            if record.valid
        )

    @property
    def invalid_transform_ids(self) -> tuple[str, ...]:
        return tuple(
            record.spec.transform_id
            for record in self.transform_records
            if not record.valid
        )

    @property
    def n_samples(self) -> int:
        return int(self.manifest["dimensions"]["n_samples"])

    @property
    def residual_rank(self) -> int:
        return int(self.manifest["dimensions"]["residual_rank"])

    @property
    def n_variants(self) -> int:
        return int(self.manifest["dimensions"]["n_variants"])

    def to_single_summary(self, transform: str | int) -> ContextTraitSummary:
        """Return a zero-copy-compatible single-transform summary view."""
        if isinstance(transform, bool):
            raise ValueError("transform must be a valid index or transform ID.")
        if isinstance(transform, int):
            index = transform
            if index < 0 or index >= len(self.transform_ids):
                raise IndexError("Transform index is out of range.")
        else:
            try:
                index = self.transform_ids.index(str(transform))
            except ValueError as exc:
                raise ValueError(f"Unknown valid transform {transform!r}.") from exc
        transform_id = self.transform_ids[index]
        record = next(
            item
            for item in self.transform_records
            if item.valid and item.spec.transform_id == transform_id
        )
        manifest = copy.deepcopy(self.manifest["context_trait_template"])
        manifest["kind"] = CONTEXT_TRAIT_KIND
        manifest["transformation"] = record.to_dict()
        manifest["phenotype_scale"] = record.diagnostics["phenotype_scale"]
        return ContextTraitSummary(
            manifest=manifest,
            component_index=self.component_index,
            residual_names=self.residual_names,
            annotation_weights=self.annotation_weights,
            annotation_masses=self.annotation_masses,
            genetic_rhs=self.genetic_rhs[index],
            genetic_traces=self.genetic_traces,
            genetic_residual=self.genetic_residual,
            residual_rhs=self.residual_rhs[index],
            residual_traces=self.residual_traces,
            residual_gram=self.residual_gram,
            rhs_numerator_contributions=self.rhs_numerator_contributions[:, index],
            trace_numerator_contributions=self.trace_numerator_contributions,
            genetic_residual_numerator_contributions=(
                self.genetic_residual_numerator_contributions
            ),
            loo_group_ids=self.loo_group_ids,
            phase_times_seconds=self.phase_times_seconds,
            peak_rss_bytes=self.peak_rss_bytes,
            decode_passes=self.decode_passes,
            decoded_blocks=self.decoded_blocks,
        )

    def rhs_after_deleting_groups(self, groups: Sequence[str]) -> np.ndarray:
        group_set = {str(value) for value in groups}
        unknown = group_set - set(self.loo_group_ids)
        if unknown:
            raise ValueError(f"Unknown approximate-LOO groups: {sorted(unknown)}.")
        deleted = np.fromiter(
            (value in group_set for value in self.loo_group_ids),
            dtype=bool,
            count=self.n_variants,
        )
        remaining_masses = self.annotation_masses - np.sum(
            self.annotation_weights[deleted], axis=0
        )
        if np.any(remaining_masses <= 0.0):
            raise ValueError(
                "Approximate-LOO deletion leaves a nonpositive annotation mass."
            )
        component_masses = np.asarray(
            [
                remaining_masses[entry.annotation_index]
                for entry in self.component_index.entries
            ],
            dtype=np.float64,
        )
        genetic = (
            np.sum(self.rhs_numerator_contributions[~deleted], axis=0)
            / component_masses[None, :]
        )
        return np.concatenate([genetic, self.residual_rhs], axis=1)


@dataclass(frozen=True)
class ContextTransformSolveResult:
    coefficients: np.ndarray
    diagnostics: SymmetricRankDiagnostics
    rank: int
    condition_number: float
    relative_residuals: np.ndarray
    retained_directions: np.ndarray
    null_space: np.ndarray
    relative_tolerance: float
    absolute_tolerance: float


@dataclass(frozen=True)
class ContextTransformFit:
    manifest: dict[str, Any]
    transform_records: tuple[TransformValidity, ...]
    transform_ids: tuple[str, ...]
    component_index: ContextComponentIndex
    residual_names: tuple[str, ...]
    equations: ContextNormalEquations
    solve: ContextTransformSolveResult
    coefficients: np.ndarray
    genetic_coefficients: np.ndarray
    residual_coefficients: np.ndarray
    raw_omegas: np.ndarray
    jackknife_groups: tuple[str, ...]
    loo_coefficients: np.ndarray
    pseudo_values: np.ndarray
    standard_errors: np.ndarray
    phase_times_seconds: dict[str, float]
    peak_rss_bytes: int

    def joint_covariance(self) -> np.ndarray:
        return jackknife_covariance_from_pseudo_values(self.pseudo_values)


@dataclass(frozen=True)
class TransformTrajectory:
    transform_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    estimates: np.ndarray
    loo_estimates: np.ndarray
    pseudo_values: np.ndarray
    standard_errors: np.ndarray
    declared_transform_ids: tuple[str, ...] = ()
    invalid_transform_ids: tuple[str, ...] = ()

    def joint_covariance(self) -> np.ndarray:
        return jackknife_covariance_from_pseudo_values(self.pseudo_values)


@dataclass(frozen=True)
class SimultaneousBandResult:
    transform_ids: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    estimates: np.ndarray
    standard_errors: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    critical_value: float
    confidence_level: float
    multiplier_max_statistics: np.ndarray
    status: str
    declared_transform_ids: tuple[str, ...] = ()
    invalid_transform_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScaleTrajectoryClassification:
    classification: str
    removable_transform_ids: tuple[str, ...]
    robust_transform_ids: tuple[str, ...]
    indeterminate_transform_ids: tuple[str, ...]
    amplification_margin: float
    heterogeneity_margin: float
    rule: str


def validate_context_transform_summary(
    summary: ContextTransformSummary,
) -> dict[str, Any]:
    """Validate the aggregate scan identity and every stored array shape."""
    manifest = summary.manifest
    if manifest.get("kind") != TRANSFORM_SUMMARY_KIND:
        raise ValueError("Not a contextual transformation-summary object.")
    if manifest.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
        raise ValueError("Unsupported transformation-summary schema version.")
    if manifest.get("feature_mode") != RAW_PROJECTED_FEATURE_MODE:
        raise ValueError("Transformation summary does not use raw_projected features.")
    observed_hash = manifest.get("transform_manifest_hash")
    without_hash = {
        key: value
        for key, value in manifest.items()
        if key != "transform_manifest_hash"
    }
    if observed_hash != canonical_sha256(without_hash):
        raise ValueError("Transformation-summary manifest SHA-256 mismatch.")
    valid_ids = tuple(
        record.spec.transform_id for record in summary.transform_records if record.valid
    )
    invalid_ids = tuple(
        record.spec.transform_id
        for record in summary.transform_records
        if not record.valid
    )
    if manifest.get("transformations") != [
        record.to_dict() for record in summary.transform_records
    ]:
        raise ValueError(
            "Transformation records and their manifest-bound specifications differ."
        )
    if valid_ids != tuple(
        manifest.get("valid_transform_ids", ())
    ) or invalid_ids != tuple(manifest.get("invalid_transform_ids", ())):
        raise ValueError("Transformation validity/order does not match the manifest.")
    if len(set(valid_ids + invalid_ids)) != len(summary.transform_records):
        raise ValueError("Transformation IDs are not unique.")
    dimensions = manifest.get("dimensions", {})
    n_samples = int(dimensions.get("n_samples", -1))
    n_variants = int(dimensions.get("n_variants", -1))
    l_count = int(dimensions.get("transformations_valid", -1))
    p_count = len(summary.component_index)
    k_count = len(summary.component_index.annotation_names)
    h_count = len(summary.residual_names)
    expected_shapes = {
        "annotation_weights": (n_variants, k_count),
        "annotation_masses": (k_count,),
        "genetic_rhs": (l_count, p_count),
        "genetic_traces": (p_count,),
        "genetic_residual": (p_count, h_count),
        "residual_rhs": (l_count, h_count),
        "residual_traces": (h_count,),
        "residual_gram": (h_count, h_count),
        "rhs_numerator_contributions": (n_variants, l_count, p_count),
        "trace_numerator_contributions": (n_variants, p_count),
        "genetic_residual_numerator_contributions": (
            n_variants,
            p_count,
            h_count,
        ),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(summary, name))
        if (
            value.shape != shape
            or value.dtype != np.dtype(np.float64)
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                f"Transformation-summary array {name!r} must be finite float "
                f"with shape {shape}; got {value.dtype} {value.shape}."
            )
    validated_weights, observed_masses = validate_annotations(
        summary.annotation_weights, n_variants, k_count
    )
    if not np.array_equal(validated_weights, summary.annotation_weights):
        raise ValueError(
            "Transformation-summary annotation weights changed on validation."
        )
    if not np.allclose(
        observed_masses, summary.annotation_masses, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("Transformation-summary annotation masses are inconsistent.")
    group_ids = _canonical_groups(summary.loo_group_ids, n_variants)
    annotation_hash = canonical_sha256(
        {
            "annotation_names": list(summary.component_index.annotation_names),
            "weights_sha256": array_sha256(summary.annotation_weights),
        }
    )
    loo_grouping_hash = canonical_sha256({"groups": list(group_ids)})
    component_masses = np.asarray(
        [
            summary.annotation_masses[entry.annotation_index]
            for entry in summary.component_index.entries
        ]
    )
    reconstructed_rhs = (
        np.sum(summary.rhs_numerator_contributions, axis=0) / component_masses[None, :]
    )
    reconstructed_traces = (
        np.sum(summary.trace_numerator_contributions, axis=0) / component_masses
    )
    reconstructed_cross = (
        np.sum(summary.genetic_residual_numerator_contributions, axis=0)
        / component_masses[:, None]
    )
    for label, observed, expected in (
        ("genetic RHS", summary.genetic_rhs, reconstructed_rhs),
        ("genetic traces", summary.genetic_traces, reconstructed_traces),
        ("genetic-residual moments", summary.genetic_residual, reconstructed_cross),
    ):
        if not np.allclose(observed, expected, rtol=5e-13, atol=5e-13):
            raise ValueError(f"Transformation-summary {label} do not reconstruct.")
    template = manifest.get("context_trait_template")
    if not isinstance(template, Mapping):
        raise ValueError("Transformation summary lacks a context-trait template.")
    if tuple(template.get("component_order", ())) != summary.component_index.names:
        raise ValueError("Transformation-summary component order is inconsistent.")
    if tuple(template.get("annotation_names", ())) != (
        summary.component_index.annotation_names
    ):
        raise ValueError("Transformation-summary annotation order is inconsistent.")
    if tuple(template.get("residual_order", ())) != summary.residual_names:
        raise ValueError("Transformation-summary residual order is inconsistent.")
    if template.get("basis_hash") != manifest["shared_inputs"].get("basis_hash"):
        raise ValueError("Transformation-summary basis identity is inconsistent.")
    for label, observed, expected in (
        (
            "annotation",
            manifest["shared_inputs"].get("annotation_hash"),
            annotation_hash,
        ),
        ("annotation", template.get("annotation_hash"), annotation_hash),
        (
            "LOO grouping",
            manifest["shared_inputs"].get("loo_grouping_hash"),
            loo_grouping_hash,
        ),
        ("LOO grouping", template.get("loo_grouping_hash"), loo_grouping_hash),
    ):
        if observed != expected:
            raise ValueError(
                f"Transformation-summary {label} identity is inconsistent."
            )
    observed_array_hashes = {
        name: array_sha256(getattr(summary, name))
        for name in _TRANSFORM_SUMMARY_ARRAY_NAMES
    }
    if manifest.get("array_hashes") != observed_array_hashes:
        raise ValueError(
            "Transformation-summary numeric arrays do not match their "
            "manifest-bound content hashes."
        )
    backend = template.get("backend")
    if not isinstance(backend, Mapping):
        raise ValueError("Transformation summary lacks backend evidence.")
    block_size = int(backend.get("block_size", 0))
    tile_size = int(backend.get("transform_tile_size", 0))
    expected_blocks = (
        (n_variants + block_size - 1) // block_size if block_size > 0 else -1
    )
    q_count = summary.component_index.pair_index.num_basis
    expected_rhs_columns = q_count * min(l_count, tile_size) if tile_size > 0 else -1
    if (
        backend.get("decode_passes") != 1
        or summary.decode_passes != 1
        or block_size < 1
        or tile_size < 1
        or summary.decoded_blocks != expected_blocks
        or summary.maximum_rhs_columns != backend.get("maximum_rhs_columns")
        or summary.maximum_rhs_columns != expected_rhs_columns
    ):
        raise ValueError(
            "Transformation-summary persisted decode/RHS evidence is inconsistent."
        )
    if summary.peak_rss_bytes < 0 or any(
        not np.isfinite(float(value)) or float(value) < 0.0
        for value in summary.phase_times_seconds.values()
    ):
        raise ValueError("Transformation-summary performance diagnostics are invalid.")
    if n_samples < 1 or n_variants < 1 or l_count != len(valid_ids):
        raise ValueError("Transformation-summary dimensions are invalid.")
    canonical_json(manifest)
    return {
        "transformations_valid": l_count,
        "transformations_invalid": len(invalid_ids),
        "n_samples": n_samples,
        "n_variants": n_variants,
        "p_genetic": p_count,
    }


def _canonical_groups(groups: Sequence[Any] | None, n_variants: int) -> tuple[str, ...]:
    if groups is None:
        return tuple(f"snp:{index}" for index in range(n_variants))
    if len(groups) != n_variants:
        raise ValueError("Approximate-LOO group count must equal the variant count.")
    result = tuple(str(value) for value in groups)
    if any(not value for value in result) or len(set(result)) < 2:
        raise ValueError("Approximate-LOO groups must contain at least two labels.")
    return result


def _apply_transform(
    spec: PhenotypeTransformSpec,
    original: np.ndarray,
    user_columns: Mapping[str, np.ndarray],
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "retained_count": int(original.size),
        "original_finite_count": int(np.sum(np.isfinite(original))),
    }
    if spec.kind == "user_supplied":
        assert spec.source is not None
        if spec.source not in user_columns:
            return None, "invalid_missing_user_column", diagnostics
        transformed = np.asarray(user_columns[spec.source], dtype=np.float64)
    elif spec.kind == "identity":
        transformed = original.copy()
    else:
        shifted = original + float(spec.shift)
        finite_shifted = shifted[np.isfinite(shifted)]
        diagnostics["minimum_shifted_value"] = (
            None if finite_shifted.size == 0 else float(np.min(finite_shifted))
        )
        diagnostics["nonpositive_shifted_count"] = int(np.sum(shifted <= 0.0))
        if not np.all(np.isfinite(shifted)) or np.any(shifted <= 0.0):
            return None, "invalid_nonpositive_or_nonfinite_shifted_values", diagnostics
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if spec.kind == "log" or abs(float(spec.box_cox_lambda or 0.0)) < 1e-12:
                transformed = np.log(shifted)
            else:
                lam = float(spec.box_cox_lambda)
                transformed = np.expm1(lam * np.log(shifted)) / lam
    diagnostics["transformed_finite_count"] = int(np.sum(np.isfinite(transformed)))
    if transformed.shape != original.shape:
        diagnostics["observed_shape"] = list(transformed.shape)
        return None, "invalid_user_column_shape", diagnostics
    if not np.all(np.isfinite(transformed)):
        return None, "invalid_nonfinite_transformed_values", diagnostics
    transformed = np.asarray(transformed, dtype=np.float64)
    diagnostics["transformed_array_hash"] = array_sha256(transformed)
    return transformed, "valid", diagnostics


def _evaluate_transform_batch(
    *,
    original_phenotype: object,
    specs: Sequence[PhenotypeTransformSpec],
    projector: ProjectorResult,
    retained_mask: object | None,
    user_transforms: Mapping[str, object] | None,
) -> tuple[tuple[TransformValidity, ...], np.ndarray, np.ndarray]:
    raw = np.asarray(original_phenotype)
    if raw.ndim != 1:
        raise ValueError("original_phenotype must be one-dimensional.")
    if retained_mask is None:
        mask = np.ones(raw.size, dtype=bool)
    else:
        mask_array = np.asarray(retained_mask)
        if mask_array.dtype != np.bool_ or mask_array.shape != raw.shape:
            raise ValueError(
                "retained_mask must be a boolean vector matching original_phenotype."
            )
        mask = np.asarray(mask_array, dtype=bool)
    n_samples = projector.projector.shape[0]
    if int(np.sum(mask)) != n_samples:
        raise ValueError(
            "retained_mask count must match the fixed projector/genotype sample count."
        )
    try:
        retained_original = raw[mask].astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "original_phenotype must be numeric on the retained mask."
        ) from exc
    if not specs:
        raise ValueError("At least one phenotype transformation must be declared.")
    ids = tuple(spec.transform_id for spec in specs)
    if len(set(ids)) != len(ids):
        raise ValueError("Transformation IDs must be unique.")
    supplied: dict[str, np.ndarray] = {}
    for source, values in (user_transforms or {}).items():
        source_name = _name("user-supplied source", source)
        column = np.asarray(values)
        if column.ndim != 1 or column.shape != raw.shape:
            supplied[source_name] = column
        else:
            try:
                supplied[source_name] = column[mask].astype(np.float64, copy=False)
            except (TypeError, ValueError):
                supplied[source_name] = np.full(n_samples, np.nan)

    records: list[TransformValidity] = []
    phenotypes: list[np.ndarray] = []
    for spec in specs:
        transformed, status, diagnostics = _apply_transform(
            spec, retained_original, supplied
        )
        if transformed is None:
            records.append(
                TransformValidity(
                    spec=spec,
                    valid=False,
                    status=status,
                    diagnostics=diagnostics,
                )
            )
            continue
        projected = projector.projector @ transformed
        projected_sum_squares = float(projected @ projected)
        diagnostics["phenotype_scale"] = {
            "projection": "fixed_rank_revealing_projector",
            "normalization": "projected_sum_squares_equals_residual_rank",
            "projected_sum_squares_before_normalization": projected_sum_squares,
            "projected_sum_squares": float(projector.residual_rank),
        }
        try:
            normalized = project_normalize_phenotype(transformed, projector)
        except ValueError:
            records.append(
                TransformValidity(
                    spec=spec,
                    valid=False,
                    status="invalid_zero_or_nonfinite_projected_variance",
                    diagnostics=diagnostics,
                )
            )
            continue
        diagnostics["normalized_array_hash"] = array_sha256(normalized)
        records.append(
            TransformValidity(
                spec=spec,
                valid=True,
                status="valid",
                diagnostics=diagnostics,
            )
        )
        phenotypes.append(normalized)
    if not phenotypes:
        raise ValueError("No declared phenotype transformation is valid.")
    return tuple(records), np.column_stack(phenotypes), mask


def build_context_transform_summary(
    *,
    genotype: object,
    basis: object,
    original_phenotype: object,
    transformations: Sequence[PhenotypeTransformSpec],
    projector: ProjectorResult,
    annotations: object,
    component_index: ContextComponentIndex,
    residual_basis: object,
    residual_names: Sequence[str],
    basis_hash: str,
    fixed_effect_hash: str,
    variant_hash: str,
    original_trait: str,
    retained_mask: object | None = None,
    user_transforms: Mapping[str, object] | None = None,
    loo_groups: Sequence[Any] | None = None,
    genotype_scaling: str = "pre_scaled_input",
    block_size: int = 256,
    transform_tile_size: int = 8,
) -> ContextTransformSummary:
    """Build all valid transformed-trait summaries in one genotype pass."""
    total_start = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    genotype_array = _finite_float64("genotype", genotype, ndim=2)
    basis_array = _finite_float64("basis", basis, ndim=2)
    residual_array = _finite_float64("residual_basis", residual_basis, ndim=2)
    n_samples, n_variants = genotype_array.shape
    q_count = basis_array.shape[1]
    h_count = residual_array.shape[1]
    if basis_array.shape[0] != n_samples or residual_array.shape[0] != n_samples:
        raise ValueError("Transform-summary inputs use different sample counts.")
    if projector.projector.shape != (n_samples, n_samples):
        raise ValueError("Projector and genotype sample counts differ.")
    if q_count != component_index.pair_index.num_basis:
        raise ValueError("Basis dimension does not match the component index.")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or isinstance(transform_tile_size, bool)
        or not isinstance(transform_tile_size, int)
        or transform_tile_size < 1
    ):
        raise ValueError(
            "block_size and transform_tile_size must be positive integers."
        )
    residual_names_tuple = tuple(str(value) for value in residual_names)
    if (
        len(residual_names_tuple) != h_count
        or len(set(residual_names_tuple)) != h_count
    ):
        raise ValueError("Residual names must uniquely match residual-basis columns.")
    column_energy = np.sum(genotype_array * genotype_array, axis=0)
    if np.any(column_energy <= np.finfo(np.float64).eps * max(n_samples, 1)):
        raise ValueError("Genotype contains a zero or near-zero scale variant.")
    weights, masses = validate_annotations(
        annotations, n_variants, len(component_index.annotation_names)
    )
    _sha256("basis_hash", basis_hash)
    _sha256("fixed_effect_hash", fixed_effect_hash)
    _sha256("variant_hash", variant_hash)
    if not genotype_scaling:
        raise ValueError("genotype_scaling must be recorded.")
    group_ids = _canonical_groups(loo_groups, n_variants)

    phase_times: dict[str, float] = {}
    phase_start = time.perf_counter()
    records, phenotypes, mask = _evaluate_transform_batch(
        original_phenotype=original_phenotype,
        specs=transformations,
        projector=projector,
        retained_mask=retained_mask,
        user_transforms=user_transforms,
    )
    phase_times["transform_and_normalize"] = time.perf_counter() - phase_start
    l_count = phenotypes.shape[1]
    p_count = len(component_index)

    phase_start = time.perf_counter()
    residual_results = tuple(
        residual_moments_low_rank(
            projector.fixed_basis, residual_array, phenotypes[:, ell]
        )
        for ell in range(l_count)
    )
    residual_rhs = np.stack([item.rhs for item in residual_results])
    residual_traces = residual_results[0].traces
    residual_gram = residual_results[0].gram
    for item in residual_results[1:]:
        if not np.array_equal(item.traces, residual_traces) or not np.array_equal(
            item.gram, residual_gram
        ):
            raise AssertionError("Phenotype-independent residual moments changed.")
    phase_times["residual_moments"] = time.perf_counter() - phase_start

    rhs_contributions = np.zeros((n_variants, l_count, p_count), dtype=np.float64)
    trace_contributions = np.zeros((n_variants, p_count), dtype=np.float64)
    genetic_residual_contributions = np.zeros(
        (n_variants, p_count, h_count), dtype=np.float64
    )
    decoded_blocks = 0
    maximum_rhs_columns = 0
    phase_start = time.perf_counter()
    for start in range(0, n_variants, block_size):
        stop = min(start + block_size, n_variants)
        genotype_block = genotype_array[:, start:stop]
        features = np.empty((q_count, n_samples, stop - start), dtype=np.float64)
        for q in range(q_count):
            features[q] = projector.projector @ (
                basis_array[:, q, None] * genotype_block
            )
        for component in component_index.entries:
            annotation = weights[start:stop, component.annotation_index]
            pair_scale = component.kernel_factor * annotation
            feature_products = np.einsum(
                "nb,nb->b",
                features[component.q],
                features[component.r],
                optimize=True,
            )
            trace_contributions[start:stop, component.index] = (
                pair_scale * feature_products
            )
            for h in range(h_count):
                weighted = np.einsum(
                    "nb,n,nb->b",
                    features[component.q],
                    residual_array[:, h],
                    features[component.r],
                    optimize=True,
                )
                genetic_residual_contributions[start:stop, component.index, h] = (
                    pair_scale * weighted
                )
        for first in range(0, l_count, transform_tile_size):
            last = min(first + transform_tile_size, l_count)
            right_hand_sides = np.column_stack(
                [
                    basis_array[:, q] * phenotypes[:, ell]
                    for q in range(q_count)
                    for ell in range(first, last)
                ]
            )
            maximum_rhs_columns = max(maximum_rhs_columns, right_hand_sides.shape[1])
            products = (
                (genotype_block.T @ right_hand_sides)
                .reshape(stop - start, q_count, last - first)
                .transpose(0, 2, 1)
            )
            for component in component_index.entries:
                pair_scale = (
                    component.kernel_factor
                    * weights[start:stop, component.annotation_index]
                )
                rhs_contributions[start:stop, first:last, component.index] = (
                    pair_scale[:, None]
                    * products[:, :, component.q]
                    * products[:, :, component.r]
                )
        decoded_blocks += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
    phase_times["genotype_pass"] = time.perf_counter() - phase_start

    component_masses = np.asarray(
        [masses[entry.annotation_index] for entry in component_index.entries]
    )
    genetic_rhs = np.sum(rhs_contributions, axis=0) / component_masses[None, :]
    genetic_traces = np.sum(trace_contributions, axis=0) / component_masses
    genetic_residual = (
        np.sum(genetic_residual_contributions, axis=0) / component_masses[:, None]
    )

    annotation_hash = canonical_sha256(
        {
            "annotation_names": list(component_index.annotation_names),
            "weights_sha256": array_sha256(weights),
        }
    )
    loo_grouping_hash = canonical_sha256({"groups": list(group_ids)})
    original_array = np.asarray(original_phenotype)
    shared_template: dict[str, Any] = {
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
        },
        "component_order": list(component_index.names),
        "annotation_names": list(component_index.annotation_names),
        "residual_order": list(residual_names_tuple),
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
        "backend": {
            "name": "python_numpy_transform_reference",
            "dtype": "float64",
            "decode_passes": 1,
            "block_size": block_size,
            "transform_tile_size": transform_tile_size,
            "maximum_rhs_columns": maximum_rhs_columns,
            "rhs_layout": "q_major_transform_minor_within_tile",
        },
        "approximate_loo": {
            "method": "snp_contribution_summary_v1",
            "exact_deleted_kernels": False,
            "groups": len(set(group_ids)),
        },
    }
    valid_ids = [record.spec.transform_id for record in records if record.valid]
    transform_payload = {
        "kind": TRANSFORM_SUMMARY_KIND,
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "original_trait": _name("original_trait", original_trait),
        "retained_mask": {
            "original_count": int(original_array.size),
            "retained_count": int(np.sum(mask)),
            "sha256": array_sha256(mask),
            "policy": "fixed_across_all_transformations",
        },
        "transformations": [record.to_dict() for record in records],
        "valid_transform_ids": valid_ids,
        "invalid_transform_ids": [
            record.spec.transform_id for record in records if not record.valid
        ],
        "array_hashes": {
            "annotation_weights": array_sha256(weights),
            "annotation_masses": array_sha256(masses),
            "genetic_rhs": array_sha256(genetic_rhs),
            "genetic_traces": array_sha256(genetic_traces),
            "genetic_residual": array_sha256(genetic_residual),
            "residual_rhs": array_sha256(residual_rhs),
            "residual_traces": array_sha256(residual_traces),
            "residual_gram": array_sha256(residual_gram),
            "rhs_numerator_contributions": array_sha256(rhs_contributions),
            "trace_numerator_contributions": array_sha256(trace_contributions),
            "genetic_residual_numerator_contributions": array_sha256(
                genetic_residual_contributions
            ),
        },
        "shared_inputs": {
            "basis_hash": basis_hash,
            "fixed_effect_hash": fixed_effect_hash,
            "variant_hash": variant_hash,
            "annotation_hash": annotation_hash,
            "loo_grouping_hash": loo_grouping_hash,
        },
        "dimensions": {
            **shared_template["dimensions"],
            "transformations_declared": len(records),
            "transformations_valid": l_count,
        },
        "context_trait_template": shared_template,
    }
    transform_payload["transform_manifest_hash"] = canonical_sha256(transform_payload)
    canonical_json(transform_payload)
    phase_times["total"] = time.perf_counter() - total_start
    peak_rss = max(peak_rss, process.memory_info().rss)
    result = ContextTransformSummary(
        manifest=transform_payload,
        transform_records=records,
        component_index=component_index,
        residual_names=residual_names_tuple,
        annotation_weights=weights,
        annotation_masses=masses,
        genetic_rhs=genetic_rhs,
        genetic_traces=genetic_traces,
        genetic_residual=genetic_residual,
        residual_rhs=residual_rhs,
        residual_traces=residual_traces,
        residual_gram=residual_gram,
        rhs_numerator_contributions=rhs_contributions,
        trace_numerator_contributions=trace_contributions,
        genetic_residual_numerator_contributions=genetic_residual_contributions,
        loo_group_ids=group_ids,
        phase_times_seconds=phase_times,
        peak_rss_bytes=int(peak_rss),
        decode_passes=1,
        decoded_blocks=decoded_blocks,
        maximum_rhs_columns=maximum_rhs_columns,
    )
    validate_context_transform_summary(result)
    return result


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _balanced_groups(
    all_groups: Sequence[str], selected: Sequence[str]
) -> dict[str, int]:
    counts = np.asarray(
        [sum(item == group for item in all_groups) for group in selected], dtype=int
    )
    if counts.size < 2 or np.any(counts < 1):
        raise ValueError("Approximate jackknife requires at least two nonempty groups.")
    if int(np.max(counts) - np.min(counts)) > 1:
        raise ValueError("Equal-weight approximate jackknife requires balanced groups.")
    return {
        "group_count": int(counts.size),
        "minimum_variants": int(np.min(counts)),
        "maximum_variants": int(np.max(counts)),
    }


def _solve_many(
    equations: ContextNormalEquations,
    rhs_rows: np.ndarray,
    *,
    rtol: float,
) -> ContextTransformSolveResult:
    matrix = _finite_float64("normal matrix", equations.matrix, ndim=2)
    rhs = _finite_float64("batched normal RHS", rhs_rows, ndim=2)
    if rhs.shape[1] != matrix.shape[0]:
        raise ValueError("Batched RHS and normal matrix have incompatible dimensions.")
    diagnostics, retained, absolute_tolerance = _rank_diagnostics(matrix, rtol)
    if diagnostics.rank != matrix.shape[0]:
        raise ContextRankError(
            "Contextual transform normal system is not identifiable: "
            f"rank {diagnostics.rank} of {matrix.shape[0]}.",
            diagnostics=diagnostics,
            component_names=equations.component_names,
        )
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.einsum("ni,nm,mi->i", retained, symmetric, retained)
    coefficients = (retained @ ((retained.T @ rhs.T) / eigenvalues[:, None])).T
    residuals = coefficients @ symmetric - rhs
    relative_residuals = np.linalg.norm(residuals, axis=1) / np.maximum(
        1.0, np.linalg.norm(rhs, axis=1)
    )
    return ContextTransformSolveResult(
        coefficients=coefficients,
        diagnostics=diagnostics,
        rank=diagnostics.rank,
        condition_number=diagnostics.condition_number,
        relative_residuals=relative_residuals,
        retained_directions=retained,
        null_space=diagnostics.null_space,
        relative_tolerance=rtol,
        absolute_tolerance=absolute_tolerance,
    )


def jackknife_covariance_from_pseudo_values(pseudo_values: object) -> np.ndarray:
    values = _finite_float64("pseudo_values", pseudo_values)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Pseudo-values require at least two jackknife groups.")
    flat = values.reshape(values.shape[0], -1)
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    covariance = centered.T @ centered / (values.shape[0] * (values.shape[0] - 1.0))
    return 0.5 * (covariance + covariance.T)


def _standard_errors_from_pseudo_values(pseudo_values: object) -> np.ndarray:
    """Return only the jackknife diagonal without materializing joint covariance."""
    values = _finite_float64("pseudo_values", pseudo_values)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Pseudo-values require at least two jackknife groups.")
    centered = values - np.mean(values, axis=0, keepdims=True)
    variances = np.sum(centered * centered, axis=0) / (
        values.shape[0] * (values.shape[0] - 1.0)
    )
    return np.sqrt(np.maximum(variances, 0.0))


def fit_context_transform_scan(
    reference: ContextReference,
    summary: ContextTransformSummary,
    *,
    rtol: float | None = None,
    loo_groups: Sequence[str] | None = None,
) -> ContextTransformFit:
    """Fit all valid transformations with one factorization per LOO state."""
    total_start = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    validate_context_transform_summary(summary)
    relative_tolerance = DEFAULT_SOLVE_RTOL if rtol is None else float(rtol)
    first_summary = summary.to_single_summary(0)
    compatibility = validate_fit_compatibility(reference, first_summary)
    phase_times: dict[str, float] = {}

    phase_start = time.perf_counter()
    equations = assemble_context_normal_equations(reference, first_summary)
    rhs = np.concatenate([summary.genetic_rhs, summary.residual_rhs], axis=1)
    solve = _solve_many(equations, rhs, rtol=relative_tolerance)
    phase_times["full_assembly_factor_and_solve"] = time.perf_counter() - phase_start
    coefficients = solve.coefficients
    p_genetic = len(summary.component_index)

    available = _ordered_unique(summary.loo_group_ids)
    selected = (
        available if loo_groups is None else tuple(str(value) for value in loo_groups)
    )
    if len(set(selected)) != len(selected):
        raise ValueError("Approximate-jackknife group selection contains duplicates.")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown approximate-jackknife groups: {sorted(unknown)}.")
    if len(selected) != len(available) or set(selected) != set(available):
        raise ValueError(
            "Joint transform-scan jackknife inference requires every declared "
            "approximate-jackknife group; partial replicate sets are not calibrated."
        )
    balance = _balanced_groups(summary.loo_group_ids, selected)
    loo_coefficients = np.empty(
        (len(selected), len(summary.transform_ids), coefficients.shape[1]),
        dtype=np.float64,
    )
    phase_start = time.perf_counter()
    for group_index, group in enumerate(selected):
        replicate_equations = assemble_context_normal_equations(
            reference, first_summary, (group,)
        )
        replicate_rhs = summary.rhs_after_deleting_groups((group,))
        try:
            replicate = _solve_many(
                replicate_equations, replicate_rhs, rtol=relative_tolerance
            )
        except ContextRankError as exc:
            raise ContextJackknifeError(group, exc) from exc
        loo_coefficients[group_index] = replicate.coefficients
    phase_times["joint_approximate_jackknife"] = time.perf_counter() - phase_start
    group_count = len(selected)
    pseudo_values = (
        group_count * coefficients[None, :, :] - (group_count - 1.0) * loo_coefficients
    )
    standard_errors = _standard_errors_from_pseudo_values(pseudo_values)
    raw_omegas = np.stack(
        [
            coefficients_to_omegas(row[:p_genetic], summary.component_index)
            for row in coefficients
        ]
    )
    phase_times["total"] = time.perf_counter() - total_start
    peak_rss = max(peak_rss, process.memory_info().rss)

    manifest = {
        "kind": TRANSFORM_FIT_KIND,
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "transform_manifest_hash": summary.manifest["transform_manifest_hash"],
        "transformations": [record.to_dict() for record in summary.transform_records],
        "declared_transform_ids": [
            record.spec.transform_id for record in summary.transform_records
        ],
        "transform_ids": list(summary.transform_ids),
        "invalid_transform_ids": list(summary.invalid_transform_ids),
        "shared_inputs": summary.manifest["shared_inputs"],
        "dimensions": {
            "transformations": len(summary.transform_ids),
            "p_genetic": p_genetic,
            "p_total": coefficients.shape[1],
            "jackknife_groups": group_count,
        },
        "compatibility": compatibility,
        "solve": {
            "method": "one_symmetric_eigendecomposition_many_rhs",
            "rank": solve.rank,
            "dimension": coefficients.shape[1],
            "relative_tolerance": relative_tolerance,
            "condition_number": (
                None
                if not np.isfinite(solve.condition_number)
                else solve.condition_number
            ),
            "maximum_relative_residual": float(np.max(solve.relative_residuals)),
            "factorizations": 1 + group_count,
        },
        "approximate_loo": {
            "method": "snp_contribution_delete_group_v1",
            "full_reference_same_person_reused": True,
            **balance,
            "joint_storage": "equal_group_pseudo_values_J_by_L_by_P",
        },
        "backend": {
            "name": "python_numpy_transform_reference",
            "dtype": "float64",
            "reads_individual_level_inputs": False,
        },
    }
    canonical_json(manifest)
    return ContextTransformFit(
        manifest=manifest,
        transform_records=summary.transform_records,
        transform_ids=summary.transform_ids,
        component_index=summary.component_index,
        residual_names=summary.residual_names,
        equations=equations,
        solve=solve,
        coefficients=coefficients,
        genetic_coefficients=coefficients[:, :p_genetic],
        residual_coefficients=coefficients[:, p_genetic:],
        raw_omegas=raw_omegas,
        jackknife_groups=selected,
        loo_coefficients=loo_coefficients,
        pseudo_values=pseudo_values,
        standard_errors=standard_errors,
        phase_times_seconds=phase_times,
        peak_rss_bytes=int(peak_rss),
    )


def build_transform_trajectory(
    fit: ContextTransformFit,
    coordinate_function: Callable[[np.ndarray], object] | None = None,
    *,
    coordinate_names: Sequence[str] | None = None,
) -> TransformTrajectory:
    """Build coefficient or user-declared derived trajectories from every LOO."""
    function = (
        (lambda value: value) if coordinate_function is None else coordinate_function
    )
    estimates = np.stack(
        [np.asarray(function(row), dtype=np.float64) for row in fit.coefficients]
    )
    if estimates.ndim == 1:
        estimates = estimates[:, None]
    if estimates.ndim != 2 or not np.all(np.isfinite(estimates)):
        raise ValueError("Derived trajectory must be a finite vector per transform.")
    loo_rows: list[np.ndarray] = []
    for group in range(fit.loo_coefficients.shape[0]):
        transformed = np.stack(
            [
                np.asarray(function(row), dtype=np.float64)
                for row in fit.loo_coefficients[group]
            ]
        )
        if transformed.ndim == 1:
            transformed = transformed[:, None]
        if transformed.shape != estimates.shape or not np.all(np.isfinite(transformed)):
            raise ValueError(
                "Every derived approximate-LOO trajectory must be finite and shape-stable."
            )
        loo_rows.append(transformed)
    loo = np.stack(loo_rows)
    group_count = loo.shape[0]
    pseudo = group_count * estimates[None] - (group_count - 1.0) * loo
    standard_errors = _standard_errors_from_pseudo_values(pseudo)
    if coordinate_names is None:
        names = tuple(f"coordinate:{index}" for index in range(estimates.shape[1]))
    else:
        names = tuple(str(value) for value in coordinate_names)
        if len(names) != estimates.shape[1] or len(set(names)) != len(names):
            raise ValueError(
                "Coordinate names must uniquely match the trajectory width."
            )
    return TransformTrajectory(
        transform_ids=fit.transform_ids,
        coordinate_names=names,
        estimates=estimates,
        loo_estimates=loo,
        pseudo_values=pseudo,
        standard_errors=standard_errors,
        declared_transform_ids=tuple(
            record.spec.transform_id for record in fit.transform_records
        ),
        invalid_transform_ids=tuple(
            record.spec.transform_id
            for record in fit.transform_records
            if not record.valid
        ),
    )


def _validate_transform_trajectory(
    trajectory: TransformTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    estimates = np.asarray(trajectory.estimates, dtype=np.float64)
    loo = np.asarray(trajectory.loo_estimates, dtype=np.float64)
    pseudo = np.asarray(trajectory.pseudo_values, dtype=np.float64)
    standard_errors = np.asarray(trajectory.standard_errors, dtype=np.float64)
    if estimates.ndim != 2:
        raise ValueError("Trajectory estimates must have shape (L,D).")
    l_count, d_count = estimates.shape
    if (
        len(trajectory.transform_ids) != l_count
        or len(set(trajectory.transform_ids)) != l_count
        or len(trajectory.coordinate_names) != d_count
        or len(set(trajectory.coordinate_names)) != d_count
    ):
        raise ValueError("Trajectory IDs/names do not match its (L,D) shape.")
    if pseudo.ndim != 3 or pseudo.shape[0] < 6 or pseudo.shape[1:] != estimates.shape:
        raise ValueError(
            "Trajectory pseudo-values must have shape (J,L,D) with J >= 6."
        )
    if loo.shape != pseudo.shape:
        raise ValueError(
            "Trajectory LOO estimates and pseudo-values must share (J,L,D)."
        )
    if standard_errors.shape != estimates.shape:
        raise ValueError("Trajectory standard errors must have shape (L,D).")
    if not all(
        np.all(np.isfinite(value))
        for value in (estimates, loo, pseudo, standard_errors)
    ) or np.any(standard_errors < 0.0):
        raise ValueError(
            "Trajectory values and nonnegative standard errors must be finite."
        )
    reconstructed = _standard_errors_from_pseudo_values(pseudo)
    scale = max(
        float(np.max(np.abs(standard_errors), initial=0.0)),
        float(np.max(np.abs(reconstructed), initial=0.0)),
    )
    tolerance = (1e-10 + 100.0 * np.finfo(np.float64).eps) * scale
    if np.max(np.abs(standard_errors - reconstructed), initial=0.0) > tolerance:
        raise ValueError(
            "Trajectory standard errors are inconsistent with its pseudo-value "
            "jackknife covariance."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        expected_pseudo = (
            pseudo.shape[0] * estimates[None] - (pseudo.shape[0] - 1.0) * loo
        )
    if not np.all(np.isfinite(expected_pseudo)):
        raise ValueError("Trajectory pseudo-value identity overflowed or is invalid.")
    identity_scale = max(
        float(np.max(np.abs(pseudo), initial=0.0)),
        float(np.max(np.abs(expected_pseudo), initial=0.0)),
    )
    identity_tolerance = (1.0e-10 + 100.0 * np.finfo(np.float64).eps) * identity_scale
    if np.max(np.abs(pseudo - expected_pseudo), initial=0.0) > identity_tolerance:
        raise ValueError(
            "Trajectory pseudo-values are inconsistent with the full and "
            "delete-group estimates."
        )
    _validate_declared_transform_partition(
        trajectory.transform_ids,
        trajectory.declared_transform_ids,
        trajectory.invalid_transform_ids,
    )
    return estimates, loo, pseudo, standard_errors


def _validate_declared_transform_partition(
    valid_ids: Sequence[str],
    declared_ids: Sequence[str],
    invalid_ids: Sequence[str],
) -> tuple[str, ...]:
    """Require declared order to be exactly partitioned into valid and invalid IDs."""
    valid = tuple(str(value) for value in valid_ids)
    invalid = tuple(str(value) for value in invalid_ids)
    declared = tuple(str(value) for value in declared_ids) if declared_ids else valid
    if (
        len(set(valid)) != len(valid)
        or len(set(invalid)) != len(invalid)
        or len(set(declared)) != len(declared)
    ):
        raise ValueError("Transformation identities must be unique.")
    invalid_set = set(invalid)
    if invalid_set.intersection(valid):
        raise ValueError("Valid and invalid transformation identities overlap.")
    expected_valid = tuple(value for value in declared if value not in invalid_set)
    expected_invalid = tuple(value for value in declared if value in invalid_set)
    if expected_valid != valid or expected_invalid != invalid:
        raise ValueError(
            "Declared transformations must be exactly partitioned, in order, into "
            "the analyzed and invalid transformation identities."
        )
    return declared


def simultaneous_trajectory_bands(
    trajectory: TransformTrajectory,
    *,
    confidence_level: float = 0.95,
    draws: int = 4096,
    seed: int = 0,
) -> SimultaneousBandResult:
    """Return studentized equal-group multiplier max-statistic bands."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 31:
        raise ValueError("draws must be an integer of at least 31.")
    estimates, _, pseudo, standard_errors = _validate_transform_trajectory(trajectory)
    estimable = np.isfinite(standard_errors) & (standard_errors > 0.0)
    if not np.any(estimable):
        nan = np.full_like(estimates, np.nan)
        return SimultaneousBandResult(
            transform_ids=trajectory.transform_ids,
            coordinate_names=trajectory.coordinate_names,
            estimates=estimates,
            standard_errors=standard_errors,
            lower=nan,
            upper=nan,
            critical_value=float("nan"),
            confidence_level=confidence_level,
            multiplier_max_statistics=np.empty(0),
            status="indeterminate_no_finite_positive_standard_errors",
            declared_transform_ids=trajectory.declared_transform_ids,
            invalid_transform_ids=trajectory.invalid_transform_ids,
        )
    group_count = pseudo.shape[0]
    centered = pseudo - np.mean(pseudo, axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(draws, group_count), dtype=np.int8)
    signs = signs.astype(np.float64) * 2.0 - 1.0
    multiplier_means = (
        np.einsum("bg,gld->bld", signs, centered, optimize=True) / group_count
    )
    # Recenter every signed pseudo-value sample and recompute its SE.  The
    # identity sum_g (x_g-xbar)^2 = sum_g x_g^2 - J*xbar^2 avoids a
    # draws-by-groups-by-coordinates temporary.  Studentization is essential
    # for small/moderate J; a fixed observed denominator is anti-conservative.
    centered_sum_squares = np.sum(centered * centered, axis=0)
    bootstrap_sum_squares = np.maximum(
        centered_sum_squares[None, :, :]
        - group_count * multiplier_means * multiplier_means,
        0.0,
    )
    bootstrap_standard_errors = np.sqrt(
        bootstrap_sum_squares / (group_count * (group_count - 1.0))
    )
    standardized = np.full_like(multiplier_means, np.inf)
    valid_bootstrap = bootstrap_standard_errors > 0.0
    np.divide(
        np.abs(multiplier_means),
        bootstrap_standard_errors,
        out=standardized,
        where=valid_bootstrap,
    )
    standardized[:, ~estimable] = 0.0
    maxima = np.max(standardized[:, estimable], axis=1)
    critical = float(np.quantile(maxima, confidence_level, method="higher"))
    if not np.isfinite(critical):
        nan = np.full_like(estimates, np.nan)
        return SimultaneousBandResult(
            transform_ids=trajectory.transform_ids,
            coordinate_names=trajectory.coordinate_names,
            estimates=estimates,
            standard_errors=standard_errors,
            lower=nan,
            upper=nan,
            critical_value=critical,
            confidence_level=confidence_level,
            multiplier_max_statistics=maxima,
            status="indeterminate_nonfinite_studentized_multiplier_critical_value",
            declared_transform_ids=trajectory.declared_transform_ids,
            invalid_transform_ids=trajectory.invalid_transform_ids,
        )
    half_width = critical * standard_errors
    lower = estimates - half_width
    upper = estimates + half_width
    lower[~estimable] = np.nan
    upper[~estimable] = np.nan
    return SimultaneousBandResult(
        transform_ids=trajectory.transform_ids,
        coordinate_names=trajectory.coordinate_names,
        estimates=estimates,
        standard_errors=standard_errors,
        lower=lower,
        upper=upper,
        critical_value=critical,
        confidence_level=confidence_level,
        multiplier_max_statistics=maxima,
        status="experimental_equal_group_studentized_multiplier_defined",
        declared_transform_ids=trajectory.declared_transform_ids,
        invalid_transform_ids=trajectory.invalid_transform_ids,
    )


def select_simultaneous_band_coordinate(
    bands: SimultaneousBandResult, coordinate: str | int
) -> SimultaneousBandResult:
    """Select one coordinate without changing the joint max calibration."""
    if isinstance(coordinate, bool):
        raise ValueError("coordinate must be an index or declared coordinate name.")
    if isinstance(coordinate, int):
        index = coordinate
        if index < 0 or index >= len(bands.coordinate_names):
            raise IndexError("Simultaneous-band coordinate is out of range.")
    else:
        try:
            index = bands.coordinate_names.index(str(coordinate))
        except ValueError as exc:
            raise ValueError(
                f"Unknown simultaneous-band coordinate {coordinate!r}."
            ) from exc
    column = slice(index, index + 1)
    return SimultaneousBandResult(
        transform_ids=bands.transform_ids,
        coordinate_names=(bands.coordinate_names[index],),
        estimates=bands.estimates[:, column],
        standard_errors=bands.standard_errors[:, column],
        lower=bands.lower[:, column],
        upper=bands.upper[:, column],
        critical_value=bands.critical_value,
        confidence_level=bands.confidence_level,
        multiplier_max_statistics=bands.multiplier_max_statistics,
        status=bands.status,
        declared_transform_ids=bands.declared_transform_ids,
        invalid_transform_ids=bands.invalid_transform_ids,
    )


def classify_scale_trajectory(
    amplification: SimultaneousBandResult,
    heterogeneity: SimultaneousBandResult,
    *,
    amplification_margin: float,
    heterogeneity_margin: float,
) -> ScaleTrajectoryClassification:
    """Classify a scan using simultaneous equivalence, not non-significance."""
    if amplification.transform_ids != heterogeneity.transform_ids:
        raise ValueError("Mechanism bands use different transformation grids.")
    amplification_declared = _validate_declared_transform_partition(
        amplification.transform_ids,
        amplification.declared_transform_ids,
        amplification.invalid_transform_ids,
    )
    heterogeneity_declared = _validate_declared_transform_partition(
        heterogeneity.transform_ids,
        heterogeneity.declared_transform_ids,
        heterogeneity.invalid_transform_ids,
    )
    if (
        amplification_declared != heterogeneity_declared
        or amplification.invalid_transform_ids != heterogeneity.invalid_transform_ids
    ):
        raise ValueError(
            "Mechanism bands use different declared transformation families."
        )
    critical_values_match = bool(
        amplification.critical_value == heterogeneity.critical_value
        or (
            np.isnan(amplification.critical_value)
            and np.isnan(heterogeneity.critical_value)
        )
    )
    if (
        amplification.confidence_level != heterogeneity.confidence_level
        or not critical_values_match
        or not np.array_equal(
            amplification.multiplier_max_statistics,
            heterogeneity.multiplier_max_statistics,
            equal_nan=True,
        )
        or amplification.status != heterogeneity.status
    ):
        raise ValueError(
            "Amplification and heterogeneity bands must come from one joint "
            "simultaneous calibration."
        )
    if amplification.estimates.shape[1] != 1 or heterogeneity.estimates.shape[1] != 1:
        raise ValueError(
            "Classification requires one amplification and one heterogeneity coordinate."
        )
    for name, margin in (
        ("amplification_margin", amplification_margin),
        ("heterogeneity_margin", heterogeneity_margin),
    ):
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if amplification.status.startswith("indeterminate_"):
        return ScaleTrajectoryClassification(
            classification="indeterminate",
            removable_transform_ids=(),
            robust_transform_ids=(),
            indeterminate_transform_ids=amplification_declared,
            amplification_margin=float(amplification_margin),
            heterogeneity_margin=float(heterogeneity_margin),
            rule=(
                "removable requires both simultaneous bands wholly inside their "
                "declared equivalence margins at one transform; robust requires an "
                "entire band outside a margin at every transform"
            ),
        )
    if (
        amplification.status
        != "experimental_equal_group_studentized_multiplier_defined"
    ):
        raise ValueError("Unrecognized simultaneous-band inference status.")
    if not np.isfinite(amplification.critical_value):
        raise ValueError(
            "A defined simultaneous band must have a finite critical value."
        )
    removable: list[str] = []
    robust: list[str] = []
    indeterminate: list[str] = list(amplification.invalid_transform_ids)
    for index, transform_id in enumerate(amplification.transform_ids):
        bounds = (
            amplification.lower[index, 0],
            amplification.upper[index, 0],
            heterogeneity.lower[index, 0],
            heterogeneity.upper[index, 0],
        )
        if not np.all(np.isfinite(bounds)):
            indeterminate.append(transform_id)
            continue
        amp_equivalent = (
            bounds[0] > -amplification_margin and bounds[1] < amplification_margin
        )
        heterogeneity_point = float(heterogeneity.estimates[index, 0])
        if heterogeneity_point < 0.0 or bounds[3] < 0.0:
            indeterminate.append(transform_id)
            continue
        het_equivalent = bounds[3] < heterogeneity_margin
        if amp_equivalent and het_equivalent:
            removable.append(transform_id)
            continue
        amp_outside = (
            bounds[0] > amplification_margin or bounds[1] < -amplification_margin
        )
        het_outside = bounds[2] > heterogeneity_margin
        if amp_outside or het_outside:
            robust.append(transform_id)
        else:
            indeterminate.append(transform_id)
    if removable:
        classification = "removable_within_declared_family"
    elif not amplification.invalid_transform_ids and len(robust) == len(
        amplification.transform_ids
    ):
        classification = "robust_within_declared_family"
    else:
        classification = "indeterminate"
    return ScaleTrajectoryClassification(
        classification=classification,
        removable_transform_ids=tuple(removable),
        robust_transform_ids=tuple(robust),
        indeterminate_transform_ids=tuple(indeterminate),
        amplification_margin=float(amplification_margin),
        heterogeneity_margin=float(heterogeneity_margin),
        rule=(
            "removable requires both simultaneous bands wholly inside their declared "
            "equivalence margins at one transform; robust requires an entire band "
            "outside a margin at every transform"
        ),
    )


def pointwise_normal_bands(
    trajectory: TransformTrajectory, *, confidence_level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic pointwise bands; not the default scan-level inference."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return (
        trajectory.estimates - critical * trajectory.standard_errors,
        trajectory.estimates + critical * trajectory.standard_errors,
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
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


def write_context_transform_summary(
    summary: ContextTransformSummary, output_prefix: str | Path
) -> tuple[Path, Path]:
    """Write a hash-bound aggregate transform summary; no phenotype rows are stored."""
    validate_context_transform_summary(summary)
    prefix = Path(output_prefix)
    manifest_path = prefix.with_suffix(".context-transform.json")
    arrays_path = prefix.with_suffix(".context-transform.npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{arrays_path.name}.", dir=arrays_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                annotation_weights=summary.annotation_weights,
                annotation_masses=summary.annotation_masses,
                genetic_rhs=summary.genetic_rhs,
                genetic_traces=summary.genetic_traces,
                genetic_residual=summary.genetic_residual,
                residual_rhs=summary.residual_rhs,
                residual_traces=summary.residual_traces,
                residual_gram=summary.residual_gram,
                rhs_numerator_contributions=summary.rhs_numerator_contributions,
                trace_numerator_contributions=summary.trace_numerator_contributions,
                genetic_residual_numerator_contributions=(
                    summary.genetic_residual_numerator_contributions
                ),
                loo_group_ids=np.asarray(summary.loo_group_ids, dtype=np.str_),
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
    payload = copy.deepcopy(summary.manifest)
    payload["artifact"] = {
        "path": arrays_path.name,
        "format": "npz_development_v1",
        "sha256": array_sha256(np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8)),
    }
    payload["performance"] = {
        "phase_times_seconds": summary.phase_times_seconds,
        "peak_rss_bytes": summary.peak_rss_bytes,
        "decode_passes": summary.decode_passes,
        "decoded_blocks": summary.decoded_blocks,
        "maximum_rhs_columns": summary.maximum_rhs_columns,
    }
    _atomic_write_text(manifest_path, json.dumps(payload, sort_keys=True, indent=2))
    return manifest_path, arrays_path


def load_context_transform_summary(
    manifest_path: str | Path,
    *,
    expected_transform_manifest_hash: str | None = None,
) -> ContextTransformSummary:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != TRANSFORM_SUMMARY_KIND:
        raise ValueError("Not a contextual transformation-summary manifest.")
    if payload.get("schema_version") != TRANSFORM_SCHEMA_VERSION:
        raise ValueError("Unsupported transformation-summary schema version.")
    if expected_transform_manifest_hash is not None and payload.get(
        "transform_manifest_hash"
    ) != _sha256("expected_transform_manifest_hash", expected_transform_manifest_hash):
        raise ValueError("Transformation-summary manifest hash mismatch.")
    artifact = payload.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("format") != "npz_development_v1"
    ):
        raise ValueError("Transformation summary has no supported artifact.")
    arrays_path = path.parent / str(artifact.get("path"))
    observed = array_sha256(np.frombuffer(arrays_path.read_bytes(), dtype=np.uint8))
    if observed != artifact.get("sha256"):
        raise ValueError("Transformation-summary artifact SHA-256 mismatch.")
    records = tuple(
        TransformValidity(
            spec=PhenotypeTransformSpec.from_dict(item["spec"]),
            valid=bool(item["valid"]),
            status=str(item["status"]),
            diagnostics=dict(item["diagnostics"]),
        )
        for item in payload["transformations"]
    )
    template = payload["context_trait_template"]
    dimensions = template["dimensions"]
    declared_annotation_names = template.get("annotation_names")
    if declared_annotation_names is None:
        annotation_names = tuple(
            dict.fromkeys(name.split(":", 2)[1] for name in template["component_order"])
        )
    else:
        annotation_names = tuple(str(value) for value in declared_annotation_names)
    from .spec import ContextPairIndex

    component_index = ContextComponentIndex(
        annotation_names, ContextPairIndex(int(dimensions["q"]))
    )
    if list(component_index.names) != template["component_order"]:
        raise ValueError("Transformation-summary component order is not canonical.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {name: np.array(arrays[name], copy=True) for name in arrays.files}
    performance = payload.get("performance", {})
    clean_manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact", "performance"}
    }
    observed_manifest_hash = clean_manifest.get("transform_manifest_hash")
    manifest_without_hash = {
        key: value
        for key, value in clean_manifest.items()
        if key != "transform_manifest_hash"
    }
    if observed_manifest_hash != canonical_sha256(manifest_without_hash):
        raise ValueError("Transformation-summary manifest SHA-256 mismatch.")
    canonical_json(clean_manifest)
    result = ContextTransformSummary(
        manifest=clean_manifest,
        transform_records=records,
        component_index=component_index,
        residual_names=tuple(template["residual_order"]),
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
        phase_times_seconds=dict(performance.get("phase_times_seconds", {})),
        peak_rss_bytes=int(performance.get("peak_rss_bytes", 0)),
        decode_passes=int(performance.get("decode_passes", 1)),
        decoded_blocks=int(performance.get("decoded_blocks", 0)),
        maximum_rhs_columns=int(performance.get("maximum_rhs_columns", 0)),
    )
    validate_context_transform_summary(result)
    return result
