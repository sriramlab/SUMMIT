"""Fixed multi-environment presets and covariance-surface modes.

This module is a correctness-first layer over the generic contextual engine.  It
owns reference calibration, a declared cohort mask, nuisance-design identity,
explicit basis pruning, pair contrasts, and basis-metric covariance modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linear_sum_assignment

from .fit import (
    ContextFitResult,
    fit_context_model,
    project_genetic_coefficients_psd,
)
from .oracle import (
    common_scale_features,
    coefficients_to_omegas,
    rank_revealing_projector,
)
from .spec import (
    CONTEXT_SCHEMA_VERSION,
    RAW_PROJECTED_FEATURE_MODE,
    BasisColumnSpec,
    ContextBasisSpec,
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_json,
    canonical_sha256,
)


MULTIENVIRONMENT_CALIBRATION_KIND = "summit.context.multienvironment_calibration"
MULTIENVIRONMENT_PRESET_KIND = "summit.context.multienvironment_preset"
MULTIENVIRONMENT_SCHEMA_VERSION = 1


def _name(value: Any, label: str) -> str:
    text = str(value)
    if not text or any(character.isspace() for character in text):
        raise ValueError(f"{label} must be a nonempty name without whitespace.")
    return text


def _python_scalar(value: Any) -> bool | int | float | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        value = float(value)
        if np.isfinite(value):
            return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        "Categorical values must be finite scalar numbers, booleans, or "
        "nonempty strings."
    )


def _scalar_identity(value: Any) -> str:
    scalar = _python_scalar(value)
    if isinstance(scalar, bool):
        kind = "bool"
    elif isinstance(scalar, int):
        kind = "int"
    elif isinstance(scalar, float):
        kind = "float"
    else:
        kind = "str"
    return canonical_json({"type": kind, "value": scalar})


def _mask(value: object, n_samples: int, *, label: str = "mask") -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != n_samples or array.dtype != np.bool_:
        raise ValueError(f"{label} must be a boolean array of shape ({n_samples},).")
    if int(np.sum(array)) < 2:
        raise ValueError(f"{label} must retain at least two samples.")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _source_length(
    sources: Mapping[str, Any], specs: Sequence["MultiEnvironmentSourceSpec"]
) -> int:
    lengths: list[int] = []
    for spec in specs:
        if spec.name not in sources:
            raise ValueError(f"Missing declared context source {spec.name!r}.")
        array = np.asarray(
            sources[spec.name], dtype=object if spec.kind == "categorical" else None
        )
        if array.ndim != 1:
            raise ValueError(f"Context source {spec.name!r} must be one-dimensional.")
        lengths.append(array.shape[0])
    if not lengths or len(set(lengths)) != 1:
        raise ValueError(
            "Every declared context source must have one common row count."
        )
    return lengths[0]


def _relative_rank(
    eigenvalues: np.ndarray, rtol: float = 1.0e-10
) -> tuple[int, float, float]:
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    tolerance = rtol * scale
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(positive.size)
    condition = (
        float(np.max(positive) / np.min(positive)) if positive.size else float("inf")
    )
    return rank, condition, tolerance


@dataclass(frozen=True)
class MultiEnvironmentSourceSpec:
    """One source in a fixed reference-calibrated context basis."""

    name: str
    kind: str
    categories: tuple[bool | int | float | str, ...] = ()
    reference_category: bool | int | float | str | None = None
    include_fixed_effect: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "source name"))
        if self.kind not in {"continuous", "categorical"}:
            raise ValueError("Source kind must be 'continuous' or 'categorical'.")
        if not isinstance(self.include_fixed_effect, bool):
            raise ValueError("include_fixed_effect must be boolean.")
        if self.kind == "continuous":
            if self.categories or self.reference_category is not None:
                raise ValueError("Continuous sources cannot declare categories.")
        else:
            categories = tuple(_python_scalar(value) for value in self.categories)
            identities = tuple(_scalar_identity(value) for value in categories)
            if len(set(identities)) != len(identities):
                raise ValueError("Declared categorical levels must be typed-unique.")
            object.__setattr__(self, "categories", categories)
            if self.reference_category is not None:
                reference = _python_scalar(self.reference_category)
                object.__setattr__(self, "reference_category", reference)
                if categories and _scalar_identity(reference) not in identities:
                    raise ValueError("reference_category is not a declared category.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "categories": list(self.categories),
            "reference_category": self.reference_category,
            "include_fixed_effect": self.include_fixed_effect,
        }


@dataclass(frozen=True)
class FixedEffectInteractionSpec:
    """One explicitly declared basis-by-covariate nuisance interaction."""

    basis_name: str
    covariate_name: str
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis_name", _name(self.basis_name, "basis_name"))
        object.__setattr__(
            self, "covariate_name", _name(self.covariate_name, "covariate_name")
        )
        object.__setattr__(self, "name", _name(self.name, "interaction name"))

    def to_dict(self) -> dict[str, str]:
        return {
            "basis_name": self.basis_name,
            "covariate_name": self.covariate_name,
            "name": self.name,
        }


@dataclass(frozen=True)
class MultiEnvironmentSourceCalibration:
    spec: MultiEnvironmentSourceSpec
    center: float | None
    scale: float | None
    categories: tuple[bool | int | float | str, ...]
    reference_category: bool | int | float | str | None
    category_probabilities: tuple[float, ...]
    emitted_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "center": self.center,
            "scale": self.scale,
            "categories": list(self.categories),
            "reference_category": self.reference_category,
            "category_probabilities": list(self.category_probabilities),
            "emitted_names": list(self.emitted_names),
        }


@dataclass(frozen=True)
class MultiEnvironmentConditioning:
    context_metric_eigenvalues: np.ndarray
    context_rank: int
    context_condition_number: float
    context_rank_tolerance: float
    maximum_absolute_context_correlation: float
    high_correlation_pairs: tuple[tuple[str, str, float], ...]
    fixed_effect_rank: int | None
    fixed_effect_columns: int | None
    maximum_leverage: float | None
    projected_feature_gram: np.ndarray | None
    projected_feature_eigenvalues: np.ndarray | None
    projected_feature_rank: int | None
    projected_feature_condition_number: float | None
    study_reconstruction_error: float | None


@dataclass(frozen=True)
class MultiEnvironmentCalibration:
    basis_spec: ContextBasisSpec
    source_calibrations: tuple[MultiEnvironmentSourceCalibration, ...]
    basis: np.ndarray
    basis_metric: np.ndarray
    conditioning: MultiEnvironmentConditioning
    mask_count: int
    mask_hash: str
    manifest: dict[str, Any]

    @property
    def digest(self) -> str:
        return canonical_sha256(self.manifest)

    @property
    def basis_hash(self) -> str:
        return self.digest

    @property
    def names(self) -> tuple[str, ...]:
        return self.basis_spec.names

    @property
    def has_individual_data(self) -> bool:
        return self.basis.shape[0] > 0

    def without_individual_data(self) -> "MultiEnvironmentCalibration":
        return MultiEnvironmentCalibration(
            basis_spec=self.basis_spec,
            source_calibrations=self.source_calibrations,
            basis=np.empty((0, self.basis.shape[1]), dtype=np.float64),
            basis_metric=self.basis_metric.copy(),
            conditioning=self.conditioning,
            mask_count=self.mask_count,
            mask_hash=self.mask_hash,
            manifest=dict(self.manifest),
        )

    def evaluate(
        self, sources: Mapping[str, Any], *, mask: object | None = None
    ) -> np.ndarray:
        return _evaluate_calibration(self, sources, mask=mask)


@dataclass(frozen=True)
class BasisPruningTransform:
    requested_names: tuple[str, ...]
    retained_names: tuple[str, ...]
    retained_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    transform: np.ndarray
    requested_to_retained: np.ndarray
    reconstruction: np.ndarray
    singular_values: np.ndarray
    rank: int
    tolerance: float
    reference_reconstruction_error: float
    manifest: dict[str, Any]

    @property
    def digest(self) -> str:
        return canonical_sha256(self.manifest)


@dataclass(frozen=True)
class MultiEnvironmentPreset:
    calibration: MultiEnvironmentCalibration
    pruning: BasisPruningTransform
    basis: np.ndarray
    fixed_effect_design: np.ndarray
    fixed_effect_names: tuple[str, ...]
    projector: Any | None
    reference_basis_metric: np.ndarray
    study_basis_metric: np.ndarray
    context_grid: np.ndarray
    component_index: ContextComponentIndex
    residual_basis: np.ndarray
    residual_names: tuple[str, ...]
    conditioning: MultiEnvironmentConditioning
    calibrated_basis_hash: str
    fixed_effect_spec_hash: str
    basis_hash: str
    fixed_effect_hash: str
    mask_count: int
    mask_hash: str
    manifest: dict[str, Any]

    @property
    def basis_metric(self) -> np.ndarray:
        """Reference metric used by default for transport-stable modes."""
        return self.reference_basis_metric

    @property
    def has_individual_data(self) -> bool:
        return self.basis.shape[0] > 0

    def without_individual_data(self) -> "MultiEnvironmentPreset":
        q_count = self.basis.shape[1]
        return MultiEnvironmentPreset(
            calibration=self.calibration.without_individual_data(),
            pruning=self.pruning,
            basis=np.empty((0, q_count), dtype=np.float64),
            fixed_effect_design=np.empty(
                (0, len(self.fixed_effect_names)), dtype=np.float64
            ),
            fixed_effect_names=self.fixed_effect_names,
            projector=None,
            reference_basis_metric=self.reference_basis_metric.copy(),
            study_basis_metric=self.study_basis_metric.copy(),
            context_grid=self.context_grid.copy(),
            component_index=self.component_index,
            residual_basis=np.empty((0, len(self.residual_names)), dtype=np.float64),
            residual_names=self.residual_names,
            conditioning=self.conditioning,
            calibrated_basis_hash=self.calibrated_basis_hash,
            fixed_effect_spec_hash=self.fixed_effect_spec_hash,
            basis_hash=self.basis_hash,
            fixed_effect_hash=self.fixed_effect_hash,
            mask_count=self.mask_count,
            mask_hash=self.mask_hash,
            manifest=dict(self.manifest),
        )

    def evaluate_contexts(self, sources: Mapping[str, Any]) -> np.ndarray:
        requested = self.calibration.evaluate(sources)
        return np.ascontiguousarray(requested @ self.pruning.transform)


def _calibration_conditioning(
    basis: np.ndarray, names: Sequence[str], *, correlation_warning: float = 0.95
) -> MultiEnvironmentConditioning:
    metric = basis.T @ basis / basis.shape[0]
    eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    rank, condition, tolerance = _relative_rank(eigenvalues)
    centered = basis[:, 1:] - np.mean(basis[:, 1:], axis=0, keepdims=True)
    correlations = (
        np.corrcoef(centered, rowvar=False)
        if centered.shape[1] > 1
        else np.eye(centered.shape[1], dtype=np.float64)
    )
    correlations = np.atleast_2d(correlations)
    pairs: list[tuple[str, str, float]] = []
    maximum = 0.0
    for left in range(correlations.shape[0]):
        for right in range(left + 1, correlations.shape[0]):
            value = float(correlations[left, right])
            maximum = max(maximum, abs(value))
            if abs(value) >= correlation_warning:
                pairs.append((str(names[left + 1]), str(names[right + 1]), value))
    return MultiEnvironmentConditioning(
        context_metric_eigenvalues=eigenvalues,
        context_rank=rank,
        context_condition_number=condition,
        context_rank_tolerance=tolerance,
        maximum_absolute_context_correlation=maximum,
        high_correlation_pairs=tuple(pairs),
        fixed_effect_rank=None,
        fixed_effect_columns=None,
        maximum_leverage=None,
        projected_feature_gram=None,
        projected_feature_eigenvalues=None,
        projected_feature_rank=None,
        projected_feature_condition_number=None,
        study_reconstruction_error=None,
    )


def calibrate_multienvironment_basis(
    sources: Mapping[str, Any],
    source_specs: Sequence[MultiEnvironmentSourceSpec],
    *,
    mask: object,
    basis_id: str = "multienvironment",
) -> MultiEnvironmentCalibration:
    """Fit immutable centers, scales, levels, and ``M_phi`` on a reference."""
    specs = tuple(source_specs)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Source specifications must be nonempty and uniquely named.")
    n_samples = _source_length(sources, specs)
    retained = _mask(mask, n_samples)
    calibrations: list[MultiEnvironmentSourceCalibration] = []
    columns: list[np.ndarray] = [np.ones(int(np.sum(retained)), dtype=np.float64)]
    column_specs: list[BasisColumnSpec] = [
        BasisColumnSpec(name="intercept", kind="constant", include_fixed_effect=True)
    ]
    for spec in specs:
        raw = np.asarray(
            sources[spec.name], dtype=object if spec.kind == "categorical" else None
        )[retained]
        if spec.kind == "continuous":
            try:
                values = raw.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Continuous source {spec.name!r} is not numeric on the mask."
                ) from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Continuous source {spec.name!r} is non-finite on the mask."
                )
            center = float(np.mean(values))
            scale = float(np.sqrt(np.mean((values - center) ** 2)))
            minimum = np.finfo(np.float64).eps * max(float(np.max(np.abs(values))), 1.0)
            if not np.isfinite(scale) or scale <= minimum:
                raise ValueError(
                    f"Continuous source {spec.name!r} has zero reference scale."
                )
            columns.append((values - center) / scale)
            emitted = (spec.name,)
            column_specs.append(
                BasisColumnSpec(
                    name=spec.name,
                    kind="precomputed",
                    source=f"calibrated.{spec.name}",
                    include_fixed_effect=spec.include_fixed_effect,
                )
            )
            calibrations.append(
                MultiEnvironmentSourceCalibration(
                    spec=spec,
                    center=center,
                    scale=scale,
                    categories=(),
                    reference_category=None,
                    category_probabilities=(),
                    emitted_names=emitted,
                )
            )
            continue
        values = tuple(_python_scalar(value) for value in raw.tolist())
        observed = tuple(_scalar_identity(value) for value in values)
        if spec.categories:
            categories = spec.categories
            identities = tuple(_scalar_identity(value) for value in categories)
        else:
            by_identity: dict[str, bool | int | float | str] = {}
            for identity, value in zip(observed, values):
                by_identity.setdefault(identity, value)
            identities = tuple(sorted(by_identity))
            categories = tuple(by_identity[identity] for identity in identities)
        unknown = set(observed) - set(identities)
        if unknown:
            raise ValueError(
                f"Categorical source {spec.name!r} contains undeclared levels."
            )
        if len(categories) < 2:
            raise ValueError(
                f"Categorical source {spec.name!r} needs at least two reference levels."
            )
        reference = (
            categories[0]
            if spec.reference_category is None
            else spec.reference_category
        )
        reference_identity = _scalar_identity(reference)
        if reference_identity not in identities:
            raise ValueError("reference_category is absent from the reference levels.")
        counts = np.asarray(
            [observed.count(identity) for identity in identities], dtype=np.int64
        )
        if np.any(counts < 1):
            raise ValueError(
                "Every declared categorical level must occur in the reference mask."
            )
        probabilities = counts.astype(np.float64) / len(observed)
        emitted_names: list[str] = []
        for category_index, (identity, probability) in enumerate(
            zip(identities, probabilities)
        ):
            if identity == reference_identity:
                continue
            emitted_name = f"{spec.name}.level{category_index}"
            emitted_names.append(emitted_name)
            indicator = np.fromiter(
                (value == identity for value in observed),
                dtype=np.float64,
                count=len(observed),
            )
            columns.append(indicator - probability)
            column_specs.append(
                BasisColumnSpec(
                    name=emitted_name,
                    kind="precomputed",
                    source=f"calibrated.{emitted_name}",
                    include_fixed_effect=spec.include_fixed_effect,
                )
            )
        calibrations.append(
            MultiEnvironmentSourceCalibration(
                spec=spec,
                center=None,
                scale=None,
                categories=categories,
                reference_category=reference,
                category_probabilities=tuple(float(value) for value in probabilities),
                emitted_names=tuple(emitted_names),
            )
        )
    basis = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    basis_spec = ContextBasisSpec(basis_id=basis_id, columns=tuple(column_specs))
    metric = np.ascontiguousarray(basis.T @ basis / basis.shape[0], dtype=np.float64)
    conditioning = _calibration_conditioning(basis, basis_spec.names)
    manifest = {
        "kind": MULTIENVIRONMENT_CALIBRATION_KIND,
        "schema_version": MULTIENVIRONMENT_SCHEMA_VERSION,
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "basis_id": basis_spec.basis_id,
        "basis_names": list(basis_spec.names),
        "basis_spec": basis_spec.to_dict(),
        "source_calibrations": [value.to_dict() for value in calibrations],
        "reference_mask": {
            "count": int(np.sum(retained)),
            "sha256": array_sha256(retained),
        },
        "reference_basis_metric": metric.tolist(),
        "reference_basis_hash": array_sha256(basis),
        "metric_semantics": "uncentered_reference_second_moment_E_phi_phi_transpose",
        "q": basis.shape[1],
    }
    return MultiEnvironmentCalibration(
        basis_spec=basis_spec,
        source_calibrations=tuple(calibrations),
        basis=basis,
        basis_metric=metric,
        conditioning=conditioning,
        mask_count=int(np.sum(retained)),
        mask_hash=array_sha256(retained),
        manifest=manifest,
    )


def _evaluate_calibration(
    calibration: MultiEnvironmentCalibration,
    sources: Mapping[str, Any],
    *,
    mask: object | None,
) -> np.ndarray:
    specs = tuple(value.spec for value in calibration.source_calibrations)
    normalized: dict[str, np.ndarray] = {}
    lengths: list[int] = []
    for spec in specs:
        if spec.name not in sources:
            raise ValueError(f"Missing declared context source {spec.name!r}.")
        raw = np.asarray(
            sources[spec.name], dtype=object if spec.kind == "categorical" else None
        )
        if raw.ndim == 0:
            raw = raw.reshape(1)
        if raw.ndim != 1:
            raise ValueError(f"Context source {spec.name!r} must be one-dimensional.")
        normalized[spec.name] = raw
        lengths.append(raw.shape[0])
    if len(set(lengths)) != 1:
        raise ValueError("Every context source must have one common row count.")
    n_samples = lengths[0]
    retained = (
        np.ones(n_samples, dtype=np.bool_) if mask is None else _mask(mask, n_samples)
    )
    columns: list[np.ndarray] = [np.ones(int(np.sum(retained)), dtype=np.float64)]
    for fitted in calibration.source_calibrations:
        raw = normalized[fitted.spec.name][retained]
        if fitted.spec.kind == "continuous":
            try:
                values = raw.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Continuous source {fitted.spec.name!r} is not numeric on the mask."
                ) from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Continuous source {fitted.spec.name!r} is non-finite on the mask."
                )
            assert fitted.center is not None and fitted.scale is not None
            columns.append((values - fitted.center) / fitted.scale)
            continue
        observed = tuple(_scalar_identity(value) for value in raw.tolist())
        identities = tuple(_scalar_identity(value) for value in fitted.categories)
        unknown = set(observed) - set(identities)
        if unknown:
            raise ValueError(
                f"Categorical source {fitted.spec.name!r} contains an unknown "
                "categorical level absent from the fixed reference calibration."
            )
        reference_identity = _scalar_identity(fitted.reference_category)
        for identity, probability in zip(identities, fitted.category_probabilities):
            if identity == reference_identity:
                continue
            indicator = np.fromiter(
                (value == identity for value in observed),
                dtype=np.float64,
                count=len(observed),
            )
            columns.append(indicator - probability)
    result = np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)
    if result.shape[1] != len(calibration.names):
        raise AssertionError("Internal calibrated basis shape mismatch.")
    return result


def fit_multienvironment_pruning(
    calibration: MultiEnvironmentCalibration, *, rtol: float = 1.0e-10
) -> BasisPruningTransform:
    """Fit an explicit pinned-intercept pivoted subset on the reference basis."""
    if not calibration.has_individual_data:
        raise ValueError("Reference rows are required to fit a pruning transform.")
    rtol = float(rtol)
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and positive.")
    basis = np.asarray(calibration.basis, dtype=np.float64)
    q_count = basis.shape[1]
    nonconstant = basis[:, 1:] - np.mean(basis[:, 1:], axis=0, keepdims=True)
    if nonconstant.shape[1]:
        _, triangular, pivots = qr(nonconstant, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(triangular))
        scale = float(np.max(diagonal, initial=0.0))
        tolerance = rtol * scale
        nonconstant_rank = int(np.sum(diagonal > tolerance))
        selected_nonconstant = sorted(
            int(pivots[index]) + 1 for index in range(nonconstant_rank)
        )
    else:
        diagonal = np.empty(0, dtype=np.float64)
        tolerance = 0.0
        selected_nonconstant = []
    retained_indices = (0, *selected_nonconstant)
    dropped_indices = tuple(
        index for index in range(q_count) if index not in retained_indices
    )
    transform = np.ascontiguousarray(np.eye(q_count)[:, retained_indices])
    retained_basis = basis @ transform
    reconstruction = np.linalg.lstsq(retained_basis, basis, rcond=rtol)[0]
    residual = basis - retained_basis @ reconstruction
    denominator = max(float(np.linalg.norm(basis)), np.finfo(np.float64).tiny)
    reconstruction_error = float(np.linalg.norm(residual) / denominator)
    singular_values = np.linalg.svd(basis, compute_uv=False)
    retained_names = tuple(calibration.names[index] for index in retained_indices)
    manifest = {
        "method": "pinned_intercept_pivoted_subset_v1",
        "calibration_hash": calibration.digest,
        "requested_names": list(calibration.names),
        "retained_names": list(retained_names),
        "retained_indices": list(retained_indices),
        "dropped_indices": list(dropped_indices),
        "transform": transform.tolist(),
        "reconstruction": reconstruction.tolist(),
        "rtol": rtol,
        "tolerance": tolerance,
        "reference_reconstruction_error": reconstruction_error,
    }
    return BasisPruningTransform(
        requested_names=calibration.names,
        retained_names=retained_names,
        retained_indices=tuple(retained_indices),
        dropped_indices=dropped_indices,
        transform=transform,
        requested_to_retained=transform.copy(),
        reconstruction=np.ascontiguousarray(reconstruction),
        singular_values=singular_values,
        rank=len(retained_indices),
        tolerance=tolerance,
        reference_reconstruction_error=reconstruction_error,
        manifest=manifest,
    )


def _identity_pruning(
    calibration: MultiEnvironmentCalibration,
) -> BasisPruningTransform:
    q_count = len(calibration.names)
    identity = np.eye(q_count, dtype=np.float64)
    singular_values = (
        np.linalg.svd(calibration.basis, compute_uv=False)
        if calibration.has_individual_data
        else np.sqrt(np.maximum(np.linalg.eigvalsh(calibration.basis_metric), 0.0))[
            ::-1
        ]
    )
    manifest = {
        "method": "explicit_identity_no_pruning_v1",
        "calibration_hash": calibration.digest,
        "requested_names": list(calibration.names),
        "retained_names": list(calibration.names),
        "retained_indices": list(range(q_count)),
        "dropped_indices": [],
        "transform": identity.tolist(),
        "reconstruction": identity.tolist(),
        "reference_reconstruction_error": 0.0,
    }
    return BasisPruningTransform(
        requested_names=calibration.names,
        retained_names=calibration.names,
        retained_indices=tuple(range(q_count)),
        dropped_indices=(),
        transform=identity,
        requested_to_retained=identity.copy(),
        reconstruction=identity.copy(),
        singular_values=singular_values,
        rank=q_count,
        tolerance=0.0,
        reference_reconstruction_error=0.0,
        manifest=manifest,
    )


def _default_grid_sources(
    calibration: MultiEnvironmentCalibration,
) -> dict[str, np.ndarray]:
    levels: list[tuple[Any, ...]] = []
    for fitted in calibration.source_calibrations:
        if fitted.spec.kind == "continuous":
            assert fitted.center is not None and fitted.scale is not None
            levels.append(
                tuple(
                    fitted.center + fitted.scale * value for value in (-1.0, 0.0, 1.0)
                )
            )
        else:
            levels.append(tuple(fitted.categories))
    combinations = tuple(product(*levels))
    return {
        fitted.spec.name: np.asarray(
            [row[index] for row in combinations],
            dtype=object if fitted.spec.kind == "categorical" else np.float64,
        )
        for index, fitted in enumerate(calibration.source_calibrations)
    }


def apply_multienvironment_calibration(
    calibration: MultiEnvironmentCalibration,
    sources: Mapping[str, Any],
    *,
    mask: object,
    covariates: Mapping[str, Any] | None = None,
    interactions: Sequence[FixedEffectInteractionSpec] = (),
    annotation_names: Sequence[str] = ("all",),
    pruning: BasisPruningTransform | None = None,
    context_grid_sources: Mapping[str, Any] | None = None,
    residual_basis: object | None = None,
    residual_names: Sequence[str] | None = None,
    genotype_for_diagnostics: object | None = None,
) -> MultiEnvironmentPreset:
    """Apply one immutable calibration and build a cohort-specific preset."""
    specs = tuple(value.spec for value in calibration.source_calibrations)
    n_original = _source_length(sources, specs)
    retained_mask = _mask(mask, n_original)
    requested_basis = calibration.evaluate(sources, mask=retained_mask)
    pruning_value = _identity_pruning(calibration) if pruning is None else pruning
    if pruning_value.manifest.get("calibration_hash") != calibration.digest:
        raise ValueError("Pruning transform was fitted from a different calibration.")
    if pruning_value.transform.shape[0] != requested_basis.shape[1]:
        raise ValueError("Pruning transform has an incompatible requested basis.")
    basis = np.ascontiguousarray(requested_basis @ pruning_value.transform)
    reference_metric = np.ascontiguousarray(
        pruning_value.transform.T @ calibration.basis_metric @ pruning_value.transform
    )
    study_metric = np.ascontiguousarray(basis.T @ basis / basis.shape[0])
    reconstruction_error = float(
        np.linalg.norm(requested_basis - basis @ pruning_value.reconstruction)
        / max(float(np.linalg.norm(requested_basis)), np.finfo(np.float64).tiny)
    )

    covariate_mapping = {} if covariates is None else dict(covariates)
    covariate_columns: dict[str, np.ndarray] = {}
    for raw_name, value in covariate_mapping.items():
        name = _name(raw_name, "covariate name")
        array = np.asarray(value)
        if array.ndim != 1 or array.shape[0] != n_original:
            raise ValueError(
                f"Covariate {name!r} must have the same original row count as all sources."
            )
        try:
            selected = array[retained_mask].astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Covariate {name!r} must be numeric on the mask."
            ) from exc
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"Covariate {name!r} is non-finite on the mask.")
        covariate_columns[name] = np.asarray(selected, dtype=np.float64)

    fixed_arrays: list[np.ndarray] = [np.ones(basis.shape[0], dtype=np.float64)]
    fixed_names: list[str] = ["intercept"]
    requested_fixed = {
        column.name: column.include_fixed_effect
        for column in calibration.basis_spec.columns
    }
    retained_name_to_index = {
        name: index for index, name in enumerate(pruning_value.retained_names)
    }
    for basis_name in pruning_value.retained_names:
        if basis_name != "intercept" and requested_fixed.get(basis_name, True):
            fixed_arrays.append(basis[:, retained_name_to_index[basis_name]])
            fixed_names.append(f"basis:{basis_name}")
    for name, values in covariate_columns.items():
        fixed_arrays.append(values)
        fixed_names.append(f"covariate:{name}")
    interaction_values = tuple(interactions)
    if len({value.name for value in interaction_values}) != len(interaction_values):
        raise ValueError("Fixed-effect interaction names must be unique.")
    for interaction in interaction_values:
        if interaction.basis_name not in retained_name_to_index:
            raise ValueError(
                f"Interaction {interaction.name!r} uses a dropped or unknown basis column."
            )
        if interaction.basis_name == "intercept":
            raise ValueError(
                "Interactions with the intercept duplicate covariate main effects."
            )
        if interaction.covariate_name not in covariate_columns:
            raise ValueError(
                f"Interaction {interaction.name!r} uses an undeclared covariate."
            )
        fixed_arrays.append(
            basis[:, retained_name_to_index[interaction.basis_name]]
            * covariate_columns[interaction.covariate_name]
        )
        fixed_names.append(f"interaction:{interaction.name}")
    fixed_design = np.ascontiguousarray(np.column_stack(fixed_arrays), dtype=np.float64)
    projector = rank_revealing_projector(fixed_design)

    if residual_basis is None:
        residual = np.ones((basis.shape[0], 1), dtype=np.float64)
        residual_names_value = ("residual:identity",)
        residual_spec = "identity"
    else:
        raw_residual = np.asarray(residual_basis, dtype=np.float64)
        if raw_residual.ndim == 1:
            raw_residual = raw_residual[:, None]
        if raw_residual.ndim != 2 or raw_residual.shape[0] != n_original:
            raise ValueError(
                "A custom residual basis must use the original pre-mask row count."
            )
        residual = np.ascontiguousarray(raw_residual[retained_mask])
        if not np.all(np.isfinite(residual)):
            raise ValueError("Residual basis is non-finite on the mask.")
        if residual_names is None:
            raise ValueError("Custom residual basis requires explicit residual_names.")
        residual_names_value = tuple(str(value) for value in residual_names)
        residual_spec = "user_supplied"
    if residual_names is not None and residual_basis is None:
        residual_names_value = tuple(str(value) for value in residual_names)
    if (
        len(residual_names_value) != residual.shape[1]
        or len(set(residual_names_value)) != residual.shape[1]
    ):
        raise ValueError("Residual names must be unique and match residual columns.")

    grid_sources = (
        _default_grid_sources(calibration)
        if context_grid_sources is None
        else context_grid_sources
    )
    context_grid = np.ascontiguousarray(
        calibration.evaluate(grid_sources) @ pruning_value.transform
    )
    annotation_names_value = tuple(str(value) for value in annotation_names)
    component_index = ContextComponentIndex(
        annotation_names_value, ContextPairIndex(basis.shape[1])
    )
    feature_gram: np.ndarray | None = None
    feature_eigenvalues: np.ndarray | None = None
    feature_rank: int | None = None
    feature_condition: float | None = None
    if genotype_for_diagnostics is not None:
        genotype = np.asarray(genotype_for_diagnostics, dtype=np.float64)
        if genotype.ndim != 2:
            raise ValueError("genotype_for_diagnostics must be two-dimensional.")
        if genotype.shape[0] == n_original:
            genotype = genotype[retained_mask]
        elif genotype.shape[0] != basis.shape[0]:
            raise ValueError("Diagnostic genotype rows do not match the declared mask.")
        features = common_scale_features(genotype, basis, projector.projector)
        feature_gram = np.einsum("qnm,rnm->qr", features, features, optimize=True)
        feature_eigenvalues = np.linalg.eigvalsh(0.5 * (feature_gram + feature_gram.T))
        feature_rank, feature_condition, _ = _relative_rank(feature_eigenvalues)
    base_conditioning = _calibration_conditioning(basis, pruning_value.retained_names)
    conditioning = MultiEnvironmentConditioning(
        context_metric_eigenvalues=np.linalg.eigvalsh(reference_metric),
        context_rank=_relative_rank(np.linalg.eigvalsh(reference_metric))[0],
        context_condition_number=_relative_rank(np.linalg.eigvalsh(reference_metric))[
            1
        ],
        context_rank_tolerance=_relative_rank(np.linalg.eigvalsh(reference_metric))[2],
        maximum_absolute_context_correlation=base_conditioning.maximum_absolute_context_correlation,
        high_correlation_pairs=base_conditioning.high_correlation_pairs,
        fixed_effect_rank=projector.rank,
        fixed_effect_columns=fixed_design.shape[1],
        maximum_leverage=projector.maximum_leverage,
        projected_feature_gram=feature_gram,
        projected_feature_eigenvalues=feature_eigenvalues,
        projected_feature_rank=feature_rank,
        projected_feature_condition_number=feature_condition,
        study_reconstruction_error=reconstruction_error,
    )
    fixed_effect_spec = {
        "ordered_names": fixed_names,
        "basis_main_effects": [
            name.removeprefix("basis:")
            for name in fixed_names
            if name.startswith("basis:")
        ],
        "ordered_covariates": list(covariate_columns),
        "interactions": [value.to_dict() for value in interaction_values],
    }
    fixed_effect_spec_hash = canonical_sha256(fixed_effect_spec)
    calibrated_basis_hash = canonical_sha256(
        {
            "calibration_hash": calibration.digest,
            "pruning_hash": pruning_value.digest,
        }
    )
    # Generic reference/trait artifacts expose one shared basis hash.  Bind the
    # nuisance specification into that identity while retaining the two pieces
    # separately for diagnostics.  Cohort-specific matrix values remain in
    # fixed_effect_hash and are intentionally allowed to differ.
    basis_hash = canonical_sha256(
        {
            "calibrated_basis_hash": calibrated_basis_hash,
            "fixed_effect_spec_hash": fixed_effect_spec_hash,
        }
    )
    fixed_effect_hash = array_sha256(fixed_design)
    manifest = {
        "kind": MULTIENVIRONMENT_PRESET_KIND,
        "schema_version": MULTIENVIRONMENT_SCHEMA_VERSION,
        "calibration_hash": calibration.digest,
        "pruning_hash": pruning_value.digest,
        "calibrated_basis_hash": calibrated_basis_hash,
        "fixed_effect_spec_hash": fixed_effect_spec_hash,
        "basis_hash": basis_hash,
        "fixed_effect_hash": fixed_effect_hash,
        "fixed_effect_array_hash": fixed_effect_hash,
        "basis_names": list(pruning_value.retained_names),
        "mask": {
            "count": int(np.sum(retained_mask)),
            "sha256": array_sha256(retained_mask),
        },
        "basis_array_hash": array_sha256(basis),
        "context_grid_hash": array_sha256(context_grid),
        "reference_basis_metric_hash": array_sha256(reference_metric),
        "study_basis_metric_hash": array_sha256(study_metric),
        "reference_basis_metric": reference_metric.tolist(),
        "study_basis_metric": study_metric.tolist(),
        "metric_default": "reference",
        "metric_semantics": "uncentered_second_moment_E_phi_phi_transpose",
        "fixed_effect_specification": fixed_effect_spec,
        "residual_spec": residual_spec,
        "residual_names": list(residual_names_value),
        "residual_basis_hash": array_sha256(residual),
        "component_index_hash": component_index.digest,
        "annotation_names": list(annotation_names_value),
        "dimensions": {
            "original_n": n_original,
            "retained_n": basis.shape[0],
            "requested_q": requested_basis.shape[1],
            "retained_q": basis.shape[1],
            "fixed_columns": fixed_design.shape[1],
        },
        "conditioning": {
            "context_rank": conditioning.context_rank,
            "context_condition_number": conditioning.context_condition_number,
            "fixed_effect_rank": projector.rank,
            "fixed_effect_columns": fixed_design.shape[1],
            "maximum_leverage": projector.maximum_leverage,
            "maximum_absolute_context_correlation": conditioning.maximum_absolute_context_correlation,
            "study_reconstruction_error": reconstruction_error,
            "projected_feature_rank": feature_rank,
            "projected_feature_condition_number": feature_condition,
        },
    }
    return MultiEnvironmentPreset(
        calibration=calibration,
        pruning=pruning_value,
        basis=basis,
        fixed_effect_design=fixed_design,
        fixed_effect_names=tuple(fixed_names),
        projector=projector,
        reference_basis_metric=reference_metric,
        study_basis_metric=study_metric,
        context_grid=context_grid,
        component_index=component_index,
        residual_basis=residual,
        residual_names=residual_names_value,
        conditioning=conditioning,
        calibrated_basis_hash=calibrated_basis_hash,
        fixed_effect_spec_hash=fixed_effect_spec_hash,
        basis_hash=basis_hash,
        fixed_effect_hash=fixed_effect_hash,
        mask_count=int(np.sum(retained_mask)),
        mask_hash=array_sha256(retained_mask),
        manifest=manifest,
    )


@dataclass(frozen=True)
class ContextPairContrast:
    annotation: str
    kind: str
    left: np.ndarray
    right: np.ndarray
    weights: np.ndarray
    estimate: float
    standard_error: float
    loo_values: np.ndarray
    loo_groups: tuple[str, ...]
    status: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ContextCovarianceModes:
    annotation: str
    metric: np.ndarray
    metric_source: str
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    function_values: np.ndarray
    equation_residuals: np.ndarray
    operator_rank: int
    rank_one_fraction: float
    heterogeneity_fraction: float
    loo_eigenvalues: np.ndarray
    loo_eigenfunctions: np.ndarray
    loo_function_values: np.ndarray
    loo_rank_one_fractions: np.ndarray
    eigenvalue_standard_errors: np.ndarray
    function_standard_errors: np.ndarray
    rank_one_fraction_standard_error: float
    eigengaps: np.ndarray
    unstable_modes: tuple[bool, ...]
    mode_clusters: tuple[tuple[int, ...], ...]
    eigenspace_max_principal_angles: np.ndarray
    interpretation: str
    status: str
    use_psd: bool
    manifest: dict[str, Any]


@dataclass(frozen=True)
class MultiEnvironmentFitDiagnostics:
    normal_equation_rank: int
    normal_equation_dimension: int
    normal_equation_condition_number: float
    normal_equation_nullity: int
    context_rank: int
    projected_feature_rank: int | None
    unstable_modes: tuple[bool, ...]
    status: str


def validate_multienvironment_preset(
    preset: MultiEnvironmentPreset, *, require_individual_data: bool = False
) -> dict[str, Any]:
    """Validate the aggregate identity and, when retained, cohort arrays."""
    manifest = preset.manifest
    if manifest.get("kind") != MULTIENVIRONMENT_PRESET_KIND:
        raise ValueError("Invalid multi-environment preset kind.")
    if manifest.get("schema_version") != MULTIENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported multi-environment preset schema version.")
    expected = {
        "calibration_hash": preset.calibration.digest,
        "pruning_hash": preset.pruning.digest,
        "calibrated_basis_hash": canonical_sha256(
            {
                "calibration_hash": preset.calibration.digest,
                "pruning_hash": preset.pruning.digest,
            }
        ),
        "fixed_effect_spec_hash": canonical_sha256(
            manifest.get("fixed_effect_specification", {})
        ),
        "component_index_hash": preset.component_index.digest,
        "context_grid_hash": array_sha256(preset.context_grid),
        "reference_basis_metric_hash": array_sha256(preset.reference_basis_metric),
        "study_basis_metric_hash": array_sha256(preset.study_basis_metric),
    }
    expected["basis_hash"] = canonical_sha256(
        {
            "calibrated_basis_hash": expected["calibrated_basis_hash"],
            "fixed_effect_spec_hash": expected["fixed_effect_spec_hash"],
        }
    )
    mismatches = {
        field: (manifest.get(field), value)
        for field, value in expected.items()
        if manifest.get(field) != value
    }
    direct = {
        "calibrated_basis_hash": preset.calibrated_basis_hash,
        "fixed_effect_spec_hash": preset.fixed_effect_spec_hash,
        "basis_hash": preset.basis_hash,
    }
    mismatches.update(
        {
            f"field.{field}": (value, expected[field])
            for field, value in direct.items()
            if value != expected[field]
        }
    )
    if preset.has_individual_data:
        cohort_expected = {
            "basis_array_hash": array_sha256(preset.basis),
            "fixed_effect_hash": array_sha256(preset.fixed_effect_design),
            "fixed_effect_array_hash": array_sha256(preset.fixed_effect_design),
            "residual_basis_hash": array_sha256(preset.residual_basis),
        }
        mismatches.update(
            {
                field: (manifest.get(field), value)
                for field, value in cohort_expected.items()
                if manifest.get(field) != value
            }
        )
        if preset.fixed_effect_hash != cohort_expected["fixed_effect_hash"]:
            mismatches["field.fixed_effect_hash"] = (
                preset.fixed_effect_hash,
                cohort_expected["fixed_effect_hash"],
            )
    elif require_individual_data:
        raise ValueError("This operation requires retained cohort arrays.")
    if mismatches:
        raise ValueError(f"Multi-environment preset identity mismatch: {mismatches}.")
    return {
        "status": "valid_multienvironment_preset",
        "has_individual_data": preset.has_individual_data,
        "basis_hash": preset.basis_hash,
        "fixed_effect_spec_hash": preset.fixed_effect_spec_hash,
    }


def _validate_preset_fit(preset: MultiEnvironmentPreset, fit: ContextFitResult) -> None:
    validate_multienvironment_preset(preset)
    if fit.component_index.digest != preset.component_index.digest:
        raise ValueError("Fit and multi-environment preset use different components.")
    fit_basis_hash = fit.manifest.get("basis_hash")
    if fit_basis_hash is not None and fit_basis_hash != preset.basis_hash:
        raise ValueError(
            "Fit and multi-environment preset use different basis identities."
        )
    if tuple(fit.residual_names) != tuple(preset.residual_names):
        raise ValueError(
            "Fit and multi-environment preset use different residual order."
        )


def validate_multienvironment_compatibility(
    reference_preset: MultiEnvironmentPreset,
    study_preset: MultiEnvironmentPreset,
) -> dict[str, Any]:
    """Validate shared analysis specifications while allowing cohort arrays to differ."""
    validate_multienvironment_preset(reference_preset)
    validate_multienvironment_preset(study_preset)
    fields = (
        "calibration_hash",
        "pruning_hash",
        "calibrated_basis_hash",
        "fixed_effect_spec_hash",
        "basis_hash",
        "component_index_hash",
    )
    mismatches = {
        field: (
            reference_preset.manifest.get(field),
            study_preset.manifest.get(field),
        )
        for field in fields
        if reference_preset.manifest.get(field) != study_preset.manifest.get(field)
    }
    if reference_preset.residual_names != study_preset.residual_names:
        mismatches["residual_names"] = (
            reference_preset.residual_names,
            study_preset.residual_names,
        )
    if (
        reference_preset.context_grid.shape != study_preset.context_grid.shape
        or not np.allclose(
            reference_preset.context_grid,
            study_preset.context_grid,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        mismatches["context_grid"] = ("reference", "study")
    if mismatches:
        raise ValueError(
            "Reference/study multi-environment specifications are incompatible: "
            f"{mismatches}."
        )
    return {
        "status": "compatible_fixed_multienvironment_specification",
        "shared_basis_hash": reference_preset.basis_hash,
        "shared_fixed_effect_spec_hash": reference_preset.fixed_effect_spec_hash,
        "reference_fixed_effect_hash": reference_preset.fixed_effect_hash,
        "study_fixed_effect_hash": study_preset.fixed_effect_hash,
        "cohort_specific_fixed_effect_arrays_allowed": True,
        "metric_default": "reference",
    }


def fit_multienvironment_model(
    reference: Any,
    summary: Any,
    *,
    reference_preset: MultiEnvironmentPreset,
    study_preset: MultiEnvironmentPreset,
    rtol: float | None = None,
    loo_groups: Sequence[str] | None = None,
    project_psd: bool = False,
    annotations_disjoint: bool | None = None,
) -> ContextFitResult:
    """Validate two calibrated presets and delegate to the generic fitter."""
    validate_multienvironment_compatibility(reference_preset, study_preset)
    if reference.manifest.get("basis_hash") != reference_preset.basis_hash:
        raise ValueError(
            "Reference artifact is not bound to reference_preset.basis_hash."
        )
    if summary.manifest.get("basis_hash") != study_preset.basis_hash:
        raise ValueError("Trait artifact is not bound to study_preset.basis_hash.")
    if (
        reference.manifest.get("fixed_effect_hash")
        != reference_preset.fixed_effect_hash
    ):
        raise ValueError(
            "Reference artifact fixed-effect array hash does not match its preset."
        )
    if summary.manifest.get("fixed_effect_hash") != study_preset.fixed_effect_hash:
        raise ValueError(
            "Trait artifact fixed-effect array hash does not match its preset."
        )
    if reference.component_index.digest != reference_preset.component_index.digest:
        raise ValueError(
            "Reference artifact component index does not match reference_preset."
        )
    if summary.component_index.digest != study_preset.component_index.digest:
        raise ValueError("Trait artifact component index does not match study_preset.")
    if tuple(summary.residual_names) != tuple(study_preset.residual_names):
        raise ValueError("Trait artifact residual order does not match study_preset.")
    return fit_context_model(
        reference,
        summary,
        rtol=rtol,
        loo_groups=loo_groups,
        context_grid=study_preset.context_grid,
        basis_metric=reference_preset.reference_basis_metric,
        project_psd=project_psd,
        annotations_disjoint=annotations_disjoint,
    )


def _context_vector(
    preset: MultiEnvironmentPreset, value: object, *, label: str
) -> np.ndarray:
    if isinstance(value, Mapping):
        evaluated = preset.evaluate_contexts(value)
        if evaluated.shape[0] != 1:
            raise ValueError(
                f"{label} source mapping must describe exactly one context."
            )
        return evaluated[0]
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != preset.basis.shape[1]:
        raise ValueError(
            f"{label} must be a calibrated basis vector of length {preset.basis.shape[1]}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values.")
    return vector


def derive_context_contrast(
    fit: ContextFitResult,
    preset: MultiEnvironmentPreset,
    left: object,
    right: object,
    *,
    annotation: str = "all",
    kind: str = "variance_difference",
) -> ContextPairContrast:
    """Derive a variance difference or cross-context covariance with joint LOO SE."""
    _validate_preset_fit(preset, fit)
    if annotation not in fit.component_index.annotation_names:
        raise ValueError(f"Unknown annotation {annotation!r}.")
    if kind not in {"variance_difference", "covariance"}:
        raise ValueError("kind must be 'variance_difference' or 'covariance'.")
    left_vector = _context_vector(preset, left, label="left")
    right_vector = _context_vector(preset, right, label="right")
    q_count = fit.component_index.pair_index.num_basis
    pair_weights = np.empty(len(fit.component_index.pair_index), dtype=np.float64)
    for pair in fit.component_index.pair_index.entries:
        if kind == "variance_difference":
            if pair.q == pair.r:
                value = left_vector[pair.q] ** 2 - right_vector[pair.q] ** 2
            else:
                value = 2.0 * (
                    left_vector[pair.q] * left_vector[pair.r]
                    - right_vector[pair.q] * right_vector[pair.r]
                )
        elif pair.q == pair.r:
            value = left_vector[pair.q] * right_vector[pair.q]
        else:
            value = (
                left_vector[pair.q] * right_vector[pair.r]
                + left_vector[pair.r] * right_vector[pair.q]
            )
        pair_weights[pair.index] = value
    weights = np.zeros(len(fit.component_index), dtype=np.float64)
    annotation_index = fit.component_index.annotation_names.index(annotation)
    offset = annotation_index * len(fit.component_index.pair_index)
    weights[offset : offset + pair_weights.size] = pair_weights
    estimate = float(weights @ fit.genetic_coefficients)
    p_genetic = len(fit.component_index)
    loo_values = np.asarray(
        fit.loo_coefficients[:, :p_genetic] @ weights, dtype=np.float64
    )
    if loo_values.size < 2:
        standard_error = float("nan")
        status = "indeterminate_fewer_than_two_loo_groups"
    else:
        centered = loo_values - np.mean(loo_values)
        variance = (
            (loo_values.size - 1.0) / loo_values.size * float(centered @ centered)
        )
        standard_error = float(np.sqrt(max(variance, 0.0)))
        covariance_variance = float(
            weights @ fit.jackknife_covariance[:p_genetic, :p_genetic] @ weights
        )
        scale = max(abs(variance), abs(covariance_variance), 1.0)
        if abs(variance - covariance_variance) > 1.0e-9 * scale:
            raise ValueError(
                "LOO contrast variance disagrees with the joint covariance."
            )
        status = "defined_equal_group_approximate_loo"
    manifest = {
        "kind": "summit.context.multienvironment_contrast",
        "schema_version": MULTIENVIRONMENT_SCHEMA_VERSION,
        "basis_hash": preset.basis_hash,
        "component_index_hash": fit.component_index.digest,
        "annotation": annotation,
        "contrast_kind": kind,
        "pair_order": [
            entry.to_dict() for entry in fit.component_index.pair_index.entries
        ],
        "loo_method": "equal_group_delete_one_joint_coefficients",
    }
    return ContextPairContrast(
        annotation=annotation,
        kind=kind,
        left=left_vector,
        right=right_vector,
        weights=weights,
        estimate=estimate,
        standard_error=standard_error,
        loo_values=loo_values,
        loo_groups=tuple(fit.jackknife_groups),
        status=status,
        manifest=manifest,
    )


def _metric_value(
    preset: MultiEnvironmentPreset, metric: str | object
) -> tuple[np.ndarray, str]:
    if isinstance(metric, str):
        if metric == "reference":
            value = preset.reference_basis_metric
        elif metric == "study":
            value = preset.study_basis_metric
        else:
            raise ValueError(
                "metric must be 'reference', 'study', or a numeric matrix."
            )
        source = metric
    else:
        value = np.asarray(metric, dtype=np.float64)
        source = "user_supplied"
    q_count = preset.component_index.pair_index.num_basis
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (q_count, q_count) or not np.all(np.isfinite(value)):
        raise ValueError("Basis metric has incompatible or non-finite values.")
    scale = float(np.max(np.abs(value), initial=0.0))
    if np.max(np.abs(value - value.T), initial=0.0) > 1.0e-10 * scale:
        raise ValueError("Basis metric must be symmetric.")
    value = 0.5 * (value + value.T)
    eigenvalues = np.linalg.eigvalsh(value)
    tolerance = 1.0e-12 * float(np.max(np.abs(eigenvalues), initial=0.0))
    if eigenvalues[0] <= tolerance:
        raise ValueError(
            "Covariance modes require a positive-definite metric; apply an explicit "
            "reference-fitted pruning transform first."
        )
    return np.ascontiguousarray(value), source


def _mode_decomposition(
    omega: np.ndarray, metric: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metric_eigenvalues, metric_eigenvectors = np.linalg.eigh(metric)
    metric_sqrt = (
        metric_eigenvectors * np.sqrt(metric_eigenvalues)
    ) @ metric_eigenvectors.T
    metric_inverse_sqrt = (
        metric_eigenvectors * (1.0 / np.sqrt(metric_eigenvalues))
    ) @ metric_eigenvectors.T
    operator = metric_sqrt @ omega @ metric_sqrt
    eigenvalues, whitened = np.linalg.eigh(0.5 * (operator + operator.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    whitened = whitened[:, order]
    eigenfunctions = metric_inverse_sqrt @ whitened
    for index in range(eigenfunctions.shape[1]):
        pivot = int(np.argmax(np.abs(eigenfunctions[:, index])))
        if eigenfunctions[pivot, index] < 0.0:
            eigenfunctions[:, index] *= -1.0
            whitened[:, index] *= -1.0
    functions = grid @ eigenfunctions
    residuals = np.empty(eigenvalues.size, dtype=np.float64)
    for index, eigenvalue in enumerate(eigenvalues):
        left = omega @ metric @ eigenfunctions[:, index]
        right = eigenvalue * eigenfunctions[:, index]
        residuals[index] = np.linalg.norm(left - right) / max(
            np.linalg.norm(left), np.linalg.norm(right), np.finfo(np.float64).tiny
        )
    return eigenvalues, whitened, eigenfunctions, functions, residuals


def _rank_one_fraction(eigenvalues: np.ndarray) -> tuple[float, str]:
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    tolerance = 1.0e-10 * scale
    if eigenvalues[-1] < -tolerance:
        return float("nan"), "undefined_indefinite_raw_operator"
    total = float(np.sum(np.maximum(eigenvalues, 0.0)))
    if total <= tolerance:
        return float("nan"), "undefined_zero_operator"
    return float(max(float(eigenvalues[0]), 0.0) / total), "defined_psd_operator"


def _mode_clusters(
    eigenvalues: np.ndarray, eigengap_rtol: float
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, tuple[bool, ...]]:
    q_count = eigenvalues.size
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    tolerance = eigengap_rtol * scale
    gaps = np.full(q_count, np.inf, dtype=np.float64)
    if q_count > 1:
        adjacent = np.abs(np.diff(eigenvalues))
        for index in range(q_count):
            candidates: list[float] = []
            if index:
                candidates.append(float(adjacent[index - 1]))
            if index + 1 < q_count:
                candidates.append(float(adjacent[index]))
            gaps[index] = min(candidates)
    clusters: list[tuple[int, ...]] = []
    start = 0
    for index in range(q_count - 1):
        if abs(float(eigenvalues[index] - eigenvalues[index + 1])) > tolerance:
            clusters.append(tuple(range(start, index + 1)))
            start = index + 1
    clusters.append(tuple(range(start, q_count)))
    unstable = tuple(
        any(index in cluster and len(cluster) > 1 for cluster in clusters)
        for index in range(q_count)
    )
    return tuple(clusters), gaps, unstable


def _align_modes(
    full_whitened: np.ndarray,
    replicate_whitened: np.ndarray,
    clusters: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Eigenvalue sorting alone changes scientific mode identity when two
    # replicate eigenvalues cross.  First solve the maximum-overlap assignment
    # to the full-fit functions, then align signs or tied eigenspaces.
    replicate_rows, full_columns = linear_sum_assignment(
        -np.abs(replicate_whitened.T @ full_whitened)
    )
    permutation = np.empty(full_whitened.shape[1], dtype=np.int64)
    permutation[full_columns] = replicate_rows
    aligned = replicate_whitened[:, permutation].copy()
    angles = np.zeros(len(clusters), dtype=np.float64)
    for cluster_index, cluster in enumerate(clusters):
        indices = np.asarray(cluster, dtype=np.int64)
        left = aligned[:, indices]
        right = full_whitened[:, indices]
        singular = np.linalg.svd(left.T @ right, compute_uv=False)
        angles[cluster_index] = float(
            np.max(np.arccos(np.clip(singular, -1.0, 1.0)), initial=0.0)
        )
        if indices.size == 1:
            if float(left[:, 0] @ right[:, 0]) < 0.0:
                aligned[:, indices[0]] *= -1.0
        else:
            u, _, vt = np.linalg.svd(left.T @ right)
            aligned[:, indices] = left @ (u @ vt)
    return aligned, angles, permutation


def derive_covariance_modes(
    fit: ContextFitResult,
    preset: MultiEnvironmentPreset,
    *,
    annotation: str = "all",
    metric: str | object = "reference",
    eigengap_rtol: float = 1.0e-6,
    use_psd: bool = False,
) -> ContextCovarianceModes:
    """Compute generalized covariance modes and equal-group LOO uncertainty."""
    _validate_preset_fit(preset, fit)
    if annotation not in fit.component_index.annotation_names:
        raise ValueError(f"Unknown annotation {annotation!r}.")
    eigengap_rtol = float(eigengap_rtol)
    if not np.isfinite(eigengap_rtol) or eigengap_rtol <= 0.0:
        raise ValueError("eigengap_rtol must be finite and positive.")
    metric_value, metric_source = _metric_value(preset, metric)
    annotation_index = fit.component_index.annotation_names.index(annotation)
    if use_psd:
        if fit.psd_projection is None:
            raise ValueError(
                "use_psd=True requires a fit with requested PSD projection."
            )
        full_coefficients = fit.psd_projection.projected_coefficients
    else:
        full_coefficients = fit.genetic_coefficients
    full_omega = coefficients_to_omegas(full_coefficients, fit.component_index)[
        annotation_index
    ]
    (
        eigenvalues,
        whitened,
        eigenfunctions,
        functions,
        equation_residuals,
    ) = _mode_decomposition(full_omega, metric_value, preset.context_grid)
    clusters, eigengaps, unstable = _mode_clusters(eigenvalues, eigengap_rtol)
    rank_fraction, rank_fraction_status = _rank_one_fraction(eigenvalues)
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    operator_rank = int(np.sum(np.abs(eigenvalues) > 1.0e-10 * scale))

    p_genetic = len(fit.component_index)
    raw_loo = np.asarray(fit.loo_coefficients[:, :p_genetic], dtype=np.float64)
    loo_coefficients: list[np.ndarray] = []
    projection_failure: str | None = None
    if use_psd:
        covariance = fit.jackknife_covariance[:p_genetic, :p_genetic]
        annotations_disjoint = bool(
            fit.manifest.get("psd_projection", {}).get("annotations_disjoint", False)
        )
        for index, coefficients in enumerate(raw_loo):
            try:
                projected = project_genetic_coefficients_psd(
                    coefficients,
                    covariance,
                    fit.component_index,
                    annotations_disjoint=annotations_disjoint,
                )
            except (RuntimeError, ValueError) as exc:
                projection_failure = f"replicate_{index}:{type(exc).__name__}"
                break
            loo_coefficients.append(projected.projected_coefficients)
    else:
        loo_coefficients = [row for row in raw_loo]

    q_count = eigenvalues.size
    if projection_failure is None:
        j_count = len(loo_coefficients)
        loo_eigenvalues = np.empty((j_count, q_count), dtype=np.float64)
        loo_eigenfunctions = np.empty((j_count, q_count, q_count), dtype=np.float64)
        loo_functions = np.empty(
            (j_count, preset.context_grid.shape[0], q_count), dtype=np.float64
        )
        loo_fractions = np.full(j_count, np.nan, dtype=np.float64)
        angles = np.empty((j_count, len(clusters)), dtype=np.float64)
        metric_eigenvalues, metric_eigenvectors = np.linalg.eigh(metric_value)
        metric_inverse_sqrt = (
            metric_eigenvectors * (1.0 / np.sqrt(metric_eigenvalues))
        ) @ metric_eigenvectors.T
        for index, coefficients in enumerate(loo_coefficients):
            omega = coefficients_to_omegas(coefficients, fit.component_index)[
                annotation_index
            ]
            values, replicate_whitened, _, _, _ = _mode_decomposition(
                omega, metric_value, preset.context_grid
            )
            aligned, angles[index], permutation = _align_modes(
                whitened, replicate_whitened, clusters
            )
            aligned_eigenfunctions = metric_inverse_sqrt @ aligned
            loo_eigenvalues[index] = values[permutation]
            loo_eigenfunctions[index] = aligned_eigenfunctions
            loo_functions[index] = preset.context_grid @ aligned_eigenfunctions
            loo_fractions[index] = _rank_one_fraction(values)[0]
        if j_count >= 2:
            multiplier = (j_count - 1.0) / j_count
            centered_values = loo_eigenvalues - np.mean(loo_eigenvalues, axis=0)
            eigenvalue_se = np.sqrt(
                np.maximum(
                    multiplier * np.sum(centered_values * centered_values, axis=0),
                    0.0,
                )
            )
            centered_functions = loo_functions - np.mean(loo_functions, axis=0)
            function_se = np.sqrt(
                np.maximum(
                    multiplier
                    * np.sum(centered_functions * centered_functions, axis=0),
                    0.0,
                )
            )
            for mode_index, is_unstable in enumerate(unstable):
                if is_unstable:
                    function_se[:, mode_index] = np.nan
            if np.all(np.isfinite(loo_fractions)):
                centered_fraction = loo_fractions - np.mean(loo_fractions)
                fraction_se = float(
                    np.sqrt(
                        max(
                            multiplier * float(centered_fraction @ centered_fraction),
                            0.0,
                        )
                    )
                )
            else:
                fraction_se = float("nan")
        else:
            eigenvalue_se = np.full(q_count, np.nan)
            function_se = np.full_like(functions, np.nan)
            fraction_se = float("nan")
        maximum_angles = (
            np.max(angles, axis=0) if angles.size else np.zeros(len(clusters))
        )
    else:
        loo_eigenvalues = np.empty((0, q_count), dtype=np.float64)
        loo_eigenfunctions = np.empty((0, q_count, q_count), dtype=np.float64)
        loo_functions = np.empty(
            (0, preset.context_grid.shape[0], q_count), dtype=np.float64
        )
        loo_fractions = np.empty(0, dtype=np.float64)
        eigenvalue_se = np.full(q_count, np.nan)
        function_se = np.full_like(functions, np.nan)
        fraction_se = float("nan")
        maximum_angles = np.full(len(clusters), np.nan)

    if projection_failure is not None:
        status = f"indeterminate_psd_loo_projection:{projection_failure}"
    elif rank_fraction_status == "undefined_indefinite_raw_operator":
        status = "algebraic_indefinite_raw_operator_not_covariance_modes"
    elif any(unstable):
        status = "defined_eigenvalues_and_eigenspaces_individual_functions_unstable"
    else:
        status = "defined_covariance_surface_modes"
    interpretation = (
        "psd_projected_covariance_surface_modes"
        if use_psd
        else (
            "raw_signed_algebraic_modes"
            if rank_fraction_status == "undefined_indefinite_raw_operator"
            else "raw_psd_interpretable_covariance_surface_modes"
        )
    )
    manifest = {
        "kind": "summit.context.multienvironment_modes",
        "schema_version": MULTIENVIRONMENT_SCHEMA_VERSION,
        "basis_hash": preset.basis_hash,
        "component_index_hash": fit.component_index.digest,
        "annotation": annotation,
        "metric_source": metric_source,
        "metric_hash": array_sha256(metric_value),
        "metric_semantics": "uncentered_second_moment_E_phi_phi_transpose",
        "generalized_equation": "Omega_M_a_equals_lambda_a",
        "normalization": "a_transpose_M_a_equals_one",
        "use_psd": use_psd,
        "loo_projection": (
            "same_covariance_aware_psd_rule_per_deleted_group"
            if use_psd
            else "raw_deleted_group_coefficients"
        ),
        "rank_one_fraction_status": rank_fraction_status,
        "eigengap_rtol": eigengap_rtol,
        "status": status,
    }
    return ContextCovarianceModes(
        annotation=annotation,
        metric=metric_value,
        metric_source=metric_source,
        eigenvalues=eigenvalues,
        eigenfunctions=eigenfunctions,
        function_values=functions,
        equation_residuals=equation_residuals,
        operator_rank=operator_rank,
        rank_one_fraction=rank_fraction,
        heterogeneity_fraction=(
            float(1.0 - rank_fraction) if np.isfinite(rank_fraction) else float("nan")
        ),
        loo_eigenvalues=loo_eigenvalues,
        loo_eigenfunctions=loo_eigenfunctions,
        loo_function_values=loo_functions,
        loo_rank_one_fractions=loo_fractions,
        eigenvalue_standard_errors=eigenvalue_se,
        function_standard_errors=function_se,
        rank_one_fraction_standard_error=fraction_se,
        eigengaps=eigengaps,
        unstable_modes=unstable,
        mode_clusters=clusters,
        eigenspace_max_principal_angles=maximum_angles,
        interpretation=interpretation,
        status=status,
        use_psd=use_psd,
        manifest=manifest,
    )


def diagnose_multienvironment_fit(
    fit: ContextFitResult,
    preset: MultiEnvironmentPreset,
    modes: ContextCovarianceModes | None = None,
) -> MultiEnvironmentFitDiagnostics:
    _validate_preset_fit(preset, fit)
    dimension = fit.equations.matrix.shape[0]
    unstable = () if modes is None else modes.unstable_modes
    status_parts: list[str] = []
    if preset.conditioning.context_rank < preset.basis.shape[1]:
        status_parts.append("context_metric_rank_deficient")
    if fit.solve.rank < dimension:
        status_parts.append("normal_equations_rank_deficient")
    if any(unstable):
        status_parts.append("individual_mode_functions_unstable")
    return MultiEnvironmentFitDiagnostics(
        normal_equation_rank=fit.solve.rank,
        normal_equation_dimension=dimension,
        normal_equation_condition_number=fit.solve.condition_number,
        normal_equation_nullity=dimension - fit.solve.rank,
        context_rank=preset.conditioning.context_rank,
        projected_feature_rank=preset.conditioning.projected_feature_rank,
        unstable_modes=unstable,
        status="defined_full_rank_stable"
        if not status_parts
        else ":".join(status_parts),
    )
