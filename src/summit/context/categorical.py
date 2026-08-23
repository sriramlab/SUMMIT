"""Binary and categorical presets for contextual covariance models.

The functions in this module are correctness-first conveniences around the
generic context engine.  They do not alter the common-scale feature convention:
all genetic categories are retained, while the nuisance fixed-effect design
uses an intercept plus reference-coded category indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from .fit import (
    ContextFitResult,
    ContextNormalEquations,
    project_genetic_coefficients_psd,
)
from .oracle import coefficients_to_omegas, omegas_to_coefficients
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


CATEGORICAL_PRESET_KIND = "summit.context.categorical_preset"


def _python_category(value: Any) -> bool | int | float | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Context categories cannot be non-finite.")
        return value
    if isinstance(value, str) and value:
        return value
    raise ValueError(
        "Context categories must be nonempty strings or finite scalar numbers."
    )


def _category_identity(value: Any) -> str:
    scalar = _python_category(value)
    if isinstance(scalar, bool):
        kind = "bool"
    elif isinstance(scalar, int):
        kind = "int"
    elif isinstance(scalar, float):
        kind = "float"
    else:
        kind = "str"
    return canonical_json({"type": kind, "value": scalar})


@dataclass(frozen=True)
class CategoricalContextPreset:
    """A deterministic all-category genetic and residual context preset."""

    basis_spec: ContextBasisSpec
    category_labels: tuple[bool | int | float | str, ...]
    category_counts: tuple[int, ...]
    category_order_hash: str
    basis: np.ndarray
    residual_basis: np.ndarray
    fixed_effect_columns: np.ndarray
    context_grid: np.ndarray
    basis_metric: np.ndarray
    component_index: ContextComponentIndex
    residual_names: tuple[str, ...]
    manifest: dict[str, Any]

    @property
    def basis_hash(self) -> str:
        return self.basis_spec.digest

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(self.manifest)

    @property
    def num_categories(self) -> int:
        return len(self.category_labels)

    @property
    def is_binary(self) -> bool:
        return self.num_categories == 2

    @property
    def has_individual_data(self) -> bool:
        return self.basis.shape[0] > 0

    def without_individual_data(self) -> CategoricalContextPreset:
        """Return the aggregate descriptor needed by summary-only consumers."""
        category_count = self.num_categories
        return CategoricalContextPreset(
            basis_spec=self.basis_spec,
            category_labels=self.category_labels,
            category_counts=self.category_counts,
            category_order_hash=self.category_order_hash,
            basis=np.empty((0, category_count), dtype=np.float64),
            residual_basis=np.empty((0, category_count), dtype=np.float64),
            fixed_effect_columns=np.empty(
                (0, max(category_count - 1, 0)), dtype=np.float64
            ),
            context_grid=self.context_grid.copy(),
            basis_metric=self.basis_metric.copy(),
            component_index=self.component_index,
            residual_names=self.residual_names,
            manifest=dict(self.manifest),
        )

    def fixed_effect_design(
        self,
        additional_columns: object | None = None,
        *,
        include_intercept: bool = True,
    ) -> np.ndarray:
        """Return intercept/reference-coded context columns plus covariates."""
        if not self.has_individual_data:
            raise ValueError(
                "Individual-level preset arrays were discarded after summary "
                "construction."
            )
        columns: list[np.ndarray] = []
        n_samples = self.basis.shape[0]
        if include_intercept:
            columns.append(np.ones((n_samples, 1), dtype=np.float64))
        if self.fixed_effect_columns.shape[1]:
            columns.append(self.fixed_effect_columns)
        if additional_columns is not None:
            additional = np.asarray(additional_columns, dtype=np.float64)
            if additional.ndim == 1:
                additional = additional[:, None]
            if additional.ndim != 2 or additional.shape[0] != n_samples:
                raise ValueError(
                    "Additional fixed-effect columns have incompatible dimensions."
                )
            if not np.all(np.isfinite(additional)):
                raise ValueError("Additional fixed-effect columns must be finite.")
            columns.append(additional)
        if not columns:
            return np.empty((n_samples, 0), dtype=np.float64)
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def build_categorical_context_preset(
    context: object,
    *,
    categories: Sequence[Any] | None = None,
    source_name: str = "context",
    binary: bool = False,
    annotation_names: Sequence[str] = ("all",),
    minimum_category_count: int = 1,
) -> CategoricalContextPreset:
    """Build deterministic all-one-hot genetic and residual basis matrices.

    If ``categories`` is omitted, category order is the lexicographic order of
    typed canonical JSON identities.  An explicitly supplied order is retained.
    Every declared category must occur and every observed value must be declared.
    """
    # Object dtype preserves typed scalar identities for heterogeneous Python
    # inputs.  Letting NumPy infer a common dtype can silently turn, for
    # example, ``True`` and ``1`` into the same string-valued category.
    raw = np.asarray(context, dtype=object)
    if raw.ndim != 1 or raw.size < 1:
        raise ValueError(
            "Categorical context must be a nonempty one-dimensional array."
        )
    observed_values = tuple(_python_category(value) for value in raw.tolist())
    observed_identities = tuple(_category_identity(value) for value in observed_values)
    if categories is None:
        first_by_identity: dict[str, bool | int | float | str] = {}
        for identity, value in zip(observed_identities, observed_values):
            first_by_identity.setdefault(identity, value)
        identities = tuple(sorted(first_by_identity))
        labels = tuple(first_by_identity[identity] for identity in identities)
    else:
        labels = tuple(_python_category(value) for value in categories)
        if not labels:
            raise ValueError("At least one category must be declared.")
        identities = tuple(_category_identity(value) for value in labels)
        if len(set(identities)) != len(identities):
            raise ValueError("Declared context categories contain duplicates.")
    if binary and len(labels) != 2:
        raise ValueError(
            f"The binary preset requires exactly two categories; observed {len(labels)}."
        )
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            if bool(labels[left] == labels[right]):
                raise ValueError(
                    "Distinct typed categories cannot compare equal under NumPy "
                    "one-hot evaluation."
                )
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    unknown = sorted(set(observed_identities) - set(identity_to_index))
    if unknown:
        raise ValueError("Observed context values include undeclared categories.")
    basis = np.zeros((raw.size, len(labels)), dtype=np.float64)
    for row, identity in enumerate(observed_identities):
        basis[row, identity_to_index[identity]] = 1.0
    counts_array = np.sum(basis, axis=0, dtype=np.int64)
    if np.any(counts_array == 0):
        absent = [index for index, count in enumerate(counts_array) if count == 0]
        raise ValueError(f"Declared context categories are absent at indices {absent}.")
    if (
        isinstance(minimum_category_count, bool)
        or not isinstance(minimum_category_count, int)
        or minimum_category_count < 1
    ):
        raise ValueError("minimum_category_count must be a positive integer.")
    if np.any(counts_array < minimum_category_count):
        rare = [
            index
            for index, count in enumerate(counts_array)
            if count < minimum_category_count
        ]
        raise ValueError(
            "Context categories fail the declared minimum count at indices " f"{rare}."
        )
    columns = tuple(
        BasisColumnSpec(
            name=f"category_{index}",
            kind="one_hot",
            source=str(source_name),
            include_fixed_effect=index > 0,
            parameters=(("category", label),),
        )
        for index, label in enumerate(labels)
    )
    basis_spec = ContextBasisSpec(
        basis_id=f"{source_name}.categorical.{len(labels)}",
        columns=columns,
    )
    evaluated_basis = basis_spec.evaluate(
        {str(source_name): raw}, n_samples=int(raw.size)
    )
    if not np.array_equal(evaluated_basis, basis):
        raise ValueError(
            "Typed category identities are incompatible with one-hot array "
            "evaluation."
        )
    category_order_payload = {
        "source": str(source_name),
        "typed_categories": [
            {"identity": identity, "value": label}
            for identity, label in zip(identities, labels)
        ],
    }
    category_order_hash = canonical_sha256(category_order_payload)
    annotation_tuple = tuple(str(name) for name in annotation_names)
    components = ContextComponentIndex(annotation_tuple, ContextPairIndex(len(labels)))
    counts = tuple(int(value) for value in counts_array)
    residual_names = tuple(f"residual:category:{index}" for index in range(len(labels)))
    manifest: dict[str, Any] = {
        "kind": CATEGORICAL_PRESET_KIND,
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "feature_mode": RAW_PROJECTED_FEATURE_MODE,
        "basis_hash": basis_spec.digest,
        "basis_array_hash": array_sha256(basis),
        "residual_basis_hash": array_sha256(basis),
        "category_order_hash": category_order_hash,
        "source_name": str(source_name),
        "category_labels": list(labels),
        "category_counts": list(counts),
        "num_categories": len(labels),
        "binary_requested": bool(binary),
        "genetic_basis": "all_category_one_hot",
        "residual_basis": "all_category_indicator_kernels_no_extra_identity",
        "fixed_effect_coding": "intercept_plus_reference_category_indicators",
        "reference_category_index": 0,
        "rare_category_diagnostic": {
            "minimum_count": int(np.min(counts_array)),
            "singleton_present": bool(np.any(counts_array == 1)),
            "declared_minimum_count": minimum_category_count,
        },
        "component_index_hash": components.digest,
    }
    canonical_json(manifest)
    return CategoricalContextPreset(
        basis_spec=basis_spec,
        category_labels=labels,
        category_counts=counts,
        category_order_hash=category_order_hash,
        basis=np.ascontiguousarray(basis),
        residual_basis=np.ascontiguousarray(basis.copy()),
        fixed_effect_columns=np.ascontiguousarray(basis[:, 1:]),
        context_grid=np.eye(len(labels), dtype=np.float64),
        basis_metric=np.diag(counts_array.astype(np.float64) / raw.size),
        component_index=components,
        residual_names=residual_names,
        manifest=manifest,
    )


# Concise public alias retained for scripts and interactive use.
build_categorical_preset = build_categorical_context_preset


def build_binary_context_preset(
    context: object,
    *,
    categories: Sequence[Any] | None = None,
    source_name: str = "context",
    annotation_names: Sequence[str] = ("all",),
    minimum_category_count: int = 1,
) -> CategoricalContextPreset:
    """Build the exact two-category specialization of the categorical preset."""
    return build_categorical_context_preset(
        context,
        categories=categories,
        source_name=source_name,
        binary=True,
        annotation_names=annotation_names,
        minimum_category_count=minimum_category_count,
    )


def binary_basis_transform() -> np.ndarray:
    """Return ``A`` for ``(1,E)^T = A (I[E=0],I[E=1])^T``."""
    return np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)


def binary_one_hot_to_intercept_matrix() -> np.ndarray:
    """Map ``(v0,v1,gamma)`` to ``(a,b,c)`` in canonical pair order."""
    return np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1.0, -2.0], [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def binary_intercept_to_one_hot_matrix() -> np.ndarray:
    """Map ``(a,b,c)`` to ``(v0,v1,gamma)`` in canonical pair order."""
    return np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 2.0], [1.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform_binary_coefficients(values: object, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("Binary coefficients must be finite with final dimension 3.")
    return np.asarray(np.einsum("ij,...j->...i", matrix, array), dtype=np.float64)


def binary_one_hot_to_intercept_coefficients(values: object) -> np.ndarray:
    return _transform_binary_coefficients(values, binary_one_hot_to_intercept_matrix())


def binary_intercept_to_one_hot_coefficients(values: object) -> np.ndarray:
    return _transform_binary_coefficients(values, binary_intercept_to_one_hot_matrix())


def _transform_binary_covariance(value: object, matrix: np.ndarray) -> np.ndarray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("Binary coefficient covariance must be finite and 3 by 3.")
    asymmetry = float(np.max(np.abs(covariance - covariance.T), initial=0.0))
    if asymmetry > 1.0e-10 * max(float(np.max(np.abs(covariance))), 1.0):
        raise ValueError("Binary coefficient covariance must be symmetric.")
    result = matrix @ covariance @ matrix.T
    return 0.5 * (result + result.T)


def binary_one_hot_to_intercept_covariance(value: object) -> np.ndarray:
    return _transform_binary_covariance(value, binary_one_hot_to_intercept_matrix())


def binary_intercept_to_one_hot_covariance(value: object) -> np.ndarray:
    return _transform_binary_covariance(value, binary_intercept_to_one_hot_matrix())


@dataclass(frozen=True)
class JackknifeEstimate:
    estimate: float
    standard_error: float
    loo_values: np.ndarray
    status: str

    @property
    def defined(self) -> bool:
        return bool(np.isfinite(self.estimate))

    @property
    def uncertainty_defined(self) -> bool:
        return bool(np.isfinite(self.standard_error))


@dataclass(frozen=True)
class LinearContrastResult:
    estimate: float
    standard_error: float
    z_statistic: float
    p_value: float
    loo_values: np.ndarray
    status: str
    covariance_method: str


@dataclass(frozen=True)
class CategoricalAnnotationSummary:
    annotation_name: str
    omega: np.ndarray
    loo_omegas: np.ndarray
    variances: tuple[JackknifeEstimate, ...]
    covariances: tuple[tuple[JackknifeEstimate, ...], ...]
    correlations: tuple[tuple[JackknifeEstimate, ...], ...]


@dataclass(frozen=True)
class CategoricalContextFitSummary:
    category_labels: tuple[bool | int | float | str, ...]
    category_counts: tuple[int, ...]
    raw: Mapping[str, CategoricalAnnotationSummary]
    raw_omegas: np.ndarray
    loo_raw_omegas: np.ndarray
    psd_interpretable: Mapping[str, CategoricalAnnotationSummary] | None
    psd_omegas: np.ndarray | None
    loo_psd_omegas: np.ndarray | None
    residual_variances: Mapping[str, JackknifeEstimate]
    trace_variance_proportions: Mapping[str, JackknifeEstimate]
    psd_trace_variance_proportions: Mapping[str, JackknifeEstimate] | None
    prevalence_diagnostics: dict[str, Any]
    psd_uncertainty_semantics: str | None

    @property
    def raw_omega(self) -> np.ndarray:
        """Return the sole annotation matrix, rejecting ambiguous multi-K use."""
        if self.raw_omegas.shape[0] != 1:
            raise ValueError("raw_omega is only defined for a single annotation.")
        return self.raw_omegas[0]


@dataclass(frozen=True)
class BinaryContextFitSummary:
    category_labels: tuple[bool | int | float | str, ...]
    raw: Mapping[str, JackknifeEstimate]
    psd_interpretable: Mapping[str, JackknifeEstimate] | None
    raw_omega: np.ndarray
    loo_raw_omegas: np.ndarray
    psd_omega: np.ndarray | None
    loo_psd_omegas: np.ndarray | None
    residual_variances: Mapping[str, JackknifeEstimate]
    trace_variance_proportions: Mapping[str, JackknifeEstimate]
    psd_trace_variance_proportions: Mapping[str, JackknifeEstimate] | None
    equal_variance_contrast: LinearContrastResult
    categorical: CategoricalContextFitSummary


@dataclass(frozen=True)
class BoundaryTestResult:
    hypothesis: str
    statistic: float
    p_value: float
    asymptotic_p_value: float
    null_coefficients: np.ndarray
    bootstrap_statistics: np.ndarray
    covariance_rank: int
    status: str
    method: str
    pseudo_value_covariance_error: float
    seed: int

    @property
    def multiplier_statistics(self) -> np.ndarray:
        return self.bootstrap_statistics


def _jackknife_estimate(
    point: float,
    loo_values: object,
    *,
    point_status: str = "defined",
    loo_statuses: Sequence[str] | None = None,
) -> JackknifeEstimate:
    loo = np.asarray(loo_values, dtype=np.float64)
    if loo.ndim != 1:
        raise ValueError("Derived approximate-LOO values must be one-dimensional.")
    if not np.isfinite(point):
        return JackknifeEstimate(float("nan"), float("nan"), loo, point_status)
    if loo.size < 2:
        return JackknifeEstimate(
            float(point),
            float("nan"),
            loo,
            f"{point_status};point_only_no_loo_values",
        )
    if not np.all(np.isfinite(loo)):
        reasons = sorted(
            {
                status
                for value, status in zip(loo, loo_statuses or ())
                if not np.isfinite(value)
            }
        )
        suffix = ",".join(reasons) if reasons else "undefined_replicate"
        return JackknifeEstimate(
            float(point),
            float("nan"),
            loo,
            f"{point_status};jackknife_indeterminate:{suffix}",
        )
    centered = loo - np.mean(loo)
    variance = (loo.size - 1.0) / loo.size * float(centered @ centered)
    return JackknifeEstimate(float(point), sqrt(max(variance, 0.0)), loo, point_status)


def _correlation_value(omega: np.ndarray, left: int, right: int) -> tuple[float, str]:
    v_left = float(omega[left, left])
    v_right = float(omega[right, right])
    if v_left <= 0.0 or v_right <= 0.0:
        return float("nan"), "undefined_nonpositive_context_variance"
    denominator = sqrt(v_left * v_right)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan"), "undefined_invalid_variance_product"
    return float(omega[left, right] / denominator), "defined"


def _annotation_summary(
    annotation_name: str,
    omega: np.ndarray,
    loo_omegas: np.ndarray,
    *,
    interpretation: str = "raw",
) -> CategoricalAnnotationSummary:
    c_count = omega.shape[0]
    if interpretation not in {"raw", "psd_interpretable"}:
        raise ValueError("Unknown categorical-summary interpretation.")
    linear_status = (
        "defined_raw_linear" if interpretation == "raw" else "defined_psd_interpretable"
    )
    variances = tuple(
        _jackknife_estimate(
            float(omega[c, c]), loo_omegas[:, c, c], point_status=linear_status
        )
        for c in range(c_count)
    )
    covariance_rows: list[tuple[JackknifeEstimate, ...]] = []
    correlation_rows: list[tuple[JackknifeEstimate, ...]] = []
    for left in range(c_count):
        covariance_row: list[JackknifeEstimate] = []
        correlation_row: list[JackknifeEstimate] = []
        for right in range(c_count):
            covariance_row.append(
                _jackknife_estimate(
                    float(omega[left, right]),
                    loo_omegas[:, left, right],
                    point_status=linear_status,
                )
            )
            point, point_status = _correlation_value(omega, left, right)
            loo_pairs = [_correlation_value(value, left, right) for value in loo_omegas]
            if interpretation == "psd_interpretable":
                if point_status == "defined":
                    point_status = "defined_psd_interpretable"
                loo_pairs = [
                    (
                        value,
                        (
                            "defined_psd_interpretable"
                            if status == "defined"
                            else status
                        ),
                    )
                    for value, status in loo_pairs
                ]
            correlation_row.append(
                _jackknife_estimate(
                    point,
                    [value for value, _ in loo_pairs],
                    point_status=point_status,
                    loo_statuses=[status for _, status in loo_pairs],
                )
            )
        covariance_rows.append(tuple(covariance_row))
        correlation_rows.append(tuple(correlation_row))
    return CategoricalAnnotationSummary(
        annotation_name=annotation_name,
        omega=np.asarray(omega, dtype=np.float64),
        loo_omegas=np.asarray(loo_omegas, dtype=np.float64),
        variances=variances,
        covariances=tuple(covariance_rows),
        correlations=tuple(correlation_rows),
    )


def _validate_categorical_fit(
    fit: ContextFitResult, preset: CategoricalContextPreset
) -> None:
    q_count = fit.component_index.pair_index.num_basis
    if q_count != preset.num_categories:
        raise ValueError("Fit and categorical preset use different basis dimensions.")
    if fit.component_index.names != preset.component_index.names:
        raise ValueError("Fit and categorical preset use different component indices.")
    if fit.residual_names != preset.residual_names:
        raise ValueError("Fit residual order does not match the categorical preset.")
    if fit.manifest.get("basis_hash") != preset.basis_hash:
        raise ValueError("Fit basis hash does not match the categorical preset.")
    basis_hashes = fit.manifest.get("basis_array_hashes")
    if not isinstance(basis_hashes, Mapping) or basis_hashes.get(
        "study"
    ) != preset.manifest.get("basis_array_hash"):
        raise ValueError("Fit basis values do not match the categorical preset.")
    if fit.manifest.get("residual_basis_hash") != preset.manifest.get(
        "residual_basis_hash"
    ):
        raise ValueError("Fit residual basis does not match the categorical preset.")
    context_moments = fit.manifest.get("context_moments")
    if not isinstance(context_moments, Mapping) or not isinstance(
        context_moments.get("study"), Mapping
    ):
        raise ValueError(
            "Fit lacks study context moments required to bind categorical counts."
        )
    study_second_moment = np.asarray(
        context_moments["study"].get("second_moment"), dtype=np.float64
    )
    if study_second_moment.shape != preset.basis_metric.shape or not np.allclose(
        study_second_moment,
        preset.basis_metric,
        rtol=0.0,
        atol=5.0e-14,
    ):
        raise ValueError(
            "Categorical preset counts do not match fitted study context moments."
        )


def _trace_proportion(
    coefficients: np.ndarray,
    equations: ContextNormalEquations,
    category_index: int,
    *,
    interpretation: str = "raw",
) -> tuple[float, str]:
    p_genetic = equations.genetic_count
    row = p_genetic + category_index
    genetic = float(equations.matrix[row, :p_genetic] @ coefficients[:p_genetic])
    total = float(equations.matrix[row] @ coefficients)
    scale = max(
        abs(genetic),
        float(np.max(np.abs(equations.matrix[row]), initial=0.0)),
        1.0,
    )
    if abs(total) <= 100.0 * np.finfo(np.float64).eps * scale:
        return float("nan"), "undefined_zero_trace_total"
    status = (
        "defined_raw_trace_ratio"
        if interpretation == "raw"
        else "defined_psd_interpretable_trace_ratio"
    )
    return genetic / total, status


def _trace_estimates(
    full_coefficients: np.ndarray,
    loo_coefficients: np.ndarray,
    full_equations: ContextNormalEquations,
    loo_equations: Sequence[ContextNormalEquations] | None,
    residual_names: Sequence[str],
    *,
    interpretation: str = "raw",
) -> dict[str, JackknifeEstimate]:
    result: dict[str, JackknifeEstimate] = {}
    for category_index, name in enumerate(residual_names):
        point, point_status = _trace_proportion(
            full_coefficients,
            full_equations,
            category_index,
            interpretation=interpretation,
        )
        if loo_equations is None:
            loo_values = np.empty(0, dtype=np.float64)
            loo_statuses: list[str] = []
        else:
            if len(loo_equations) != loo_coefficients.shape[0]:
                raise ValueError("One deleted normal equation is required per LOO fit.")
            pairs = [
                _trace_proportion(
                    coefficients,
                    equations,
                    category_index,
                    interpretation=interpretation,
                )
                for coefficients, equations in zip(loo_coefficients, loo_equations)
            ]
            loo_values = np.asarray([value for value, _ in pairs], dtype=np.float64)
            loo_statuses = [status for _, status in pairs]
        result[name] = _jackknife_estimate(
            point,
            loo_values,
            point_status=point_status,
            loo_statuses=loo_statuses,
        )
    return result


def _projected_loo_genetic_coefficients(
    fit: ContextFitResult,
) -> tuple[np.ndarray | None, str]:
    if fit.psd_projection is None:
        return None, "not_requested"
    p_genetic = len(fit.component_index)
    covariance = fit.jackknife_covariance[:p_genetic, :p_genetic]
    projected = np.empty((fit.loo_coefficients.shape[0], p_genetic), dtype=np.float64)
    for index, values in enumerate(fit.loo_coefficients[:, :p_genetic]):
        try:
            projected[index] = project_genetic_coefficients_psd(
                values,
                covariance,
                fit.component_index,
                annotations_disjoint=True,
            ).projected_coefficients
        except RuntimeError as exc:
            return (
                None,
                "indeterminate_projected_loo_replicate:"
                f"{index}:{type(exc).__name__}",
            )
    return projected, "defined_exploratory_projected_loo"


def derive_categorical_context_fit(
    fit: ContextFitResult,
    preset: CategoricalContextPreset,
    *,
    loo_equations: Sequence[ContextNormalEquations] | None = None,
) -> CategoricalContextFitSummary:
    """Derive category covariance/correlation and trace summaries from one fit."""
    _validate_categorical_fit(fit, preset)
    if loo_equations is not None:
        if len(loo_equations) != len(fit.jackknife_groups):
            raise ValueError("One deleted normal equation is required per LOO fit.")
        for index, (equations, group) in enumerate(
            zip(loo_equations, fit.jackknife_groups)
        ):
            if equations.deleted_groups != (group,):
                raise ValueError(
                    "Deleted normal equations do not match the fitted jackknife "
                    f"group order at replicate {index}."
                )
    raw_omegas = np.asarray(fit.raw_omegas, dtype=np.float64)
    loo_raw_omegas = np.asarray(
        [
            coefficients_to_omegas(
                values[: len(fit.component_index)], fit.component_index
            )
            for values in fit.loo_coefficients
        ],
        dtype=np.float64,
    )
    raw = {
        name: _annotation_summary(name, raw_omegas[index], loo_raw_omegas[:, index])
        for index, name in enumerate(fit.component_index.annotation_names)
    }
    p_genetic = len(fit.component_index)
    residual_variances = {
        name: _jackknife_estimate(
            fit.residual_coefficients[index],
            fit.loo_coefficients[:, p_genetic + index],
            point_status="defined_raw_linear_may_be_negative",
        )
        for index, name in enumerate(fit.residual_names)
    }
    trace_proportions = _trace_estimates(
        fit.raw_coefficients,
        fit.loo_coefficients,
        fit.equations,
        loo_equations,
        fit.residual_names,
    )
    psd_summaries: dict[str, CategoricalAnnotationSummary] | None = None
    psd_omegas: np.ndarray | None = None
    loo_psd_omegas: np.ndarray | None = None
    psd_trace_proportions: dict[str, JackknifeEstimate] | None = None
    psd_semantics: str | None = None
    if fit.psd_projection is not None:
        psd_omegas = np.asarray(fit.psd_projection.projected_omegas, dtype=np.float64)
        loo_psd_coefficients, projection_status = _projected_loo_genetic_coefficients(
            fit
        )
        if loo_psd_coefficients is None:
            summary_loo_psd_omegas = np.empty(
                (
                    0,
                    len(fit.component_index.annotation_names),
                    preset.num_categories,
                    preset.num_categories,
                ),
                dtype=np.float64,
            )
        else:
            loo_psd_omegas = np.asarray(
                [
                    coefficients_to_omegas(values, fit.component_index)
                    for values in loo_psd_coefficients
                ],
                dtype=np.float64,
            )
            summary_loo_psd_omegas = loo_psd_omegas
        psd_summaries = {
            name: _annotation_summary(
                name,
                psd_omegas[index],
                summary_loo_psd_omegas[:, index],
                interpretation="psd_interpretable",
            )
            for index, name in enumerate(fit.component_index.annotation_names)
        }
        psd_full = np.concatenate(
            [fit.psd_projection.projected_coefficients, fit.residual_coefficients]
        )
        if loo_psd_coefficients is None:
            psd_loo = np.empty((0, fit.raw_coefficients.size), dtype=np.float64)
            psd_loo_equations = None
        else:
            psd_loo = np.column_stack(
                [loo_psd_coefficients, fit.loo_coefficients[:, p_genetic:]]
            )
            psd_loo_equations = loo_equations
        psd_trace_proportions = _trace_estimates(
            psd_full,
            psd_loo,
            fit.equations,
            psd_loo_equations,
            fit.residual_names,
            interpretation="psd_interpretable",
        )
        psd_semantics = (
            "exploratory_projection_of_each_approximate_loo_replicate_"
            f"under_full_fit_covariance_metric:{projection_status}"
        )
    study_proportions = np.asarray(preset.category_counts, dtype=np.float64)
    study_proportions /= np.sum(study_proportions)
    reference_proportions: tuple[float, ...] | None = None
    prevalence_difference: float | None = None
    prevalence_status = "reference_context_moments_not_available"
    reference_moments = fit.manifest["context_moments"].get("reference")
    if isinstance(reference_moments, Mapping):
        metric = np.asarray(reference_moments.get("second_moment"), dtype=np.float64)
        if metric.shape == (preset.num_categories, preset.num_categories):
            off_diagonal = metric - np.diag(np.diag(metric))
            metric_total = float(np.trace(metric))
            if (
                np.max(np.abs(off_diagonal), initial=0.0) <= 1.0e-10
                and metric_total > 0.0
            ):
                metric_array = np.diag(metric) / metric_total
                reference_proportions = tuple(float(value) for value in metric_array)
                prevalence_difference = float(
                    np.max(np.abs(metric_array - study_proportions), initial=0.0)
                )
                prevalence_status = "defined_from_reference_one_hot_second_moment"
    return CategoricalContextFitSummary(
        category_labels=preset.category_labels,
        category_counts=preset.category_counts,
        raw=raw,
        raw_omegas=raw_omegas,
        loo_raw_omegas=loo_raw_omegas,
        psd_interpretable=psd_summaries,
        psd_omegas=psd_omegas,
        loo_psd_omegas=loo_psd_omegas,
        residual_variances=residual_variances,
        trace_variance_proportions=trace_proportions,
        psd_trace_variance_proportions=psd_trace_proportions,
        prevalence_diagnostics={
            "counts": preset.category_counts,
            "study_proportions": tuple(float(value) for value in study_proportions),
            "proportions": tuple(float(value) for value in study_proportions),
            "reference_proportions": reference_proportions,
            "maximum_absolute_prevalence_difference": prevalence_difference,
            "comparison_status": prevalence_status,
            "comparison_source": "fit_manifest_reference_context_moments",
            "minimum_count": min(preset.category_counts),
            "singleton_present": any(count == 1 for count in preset.category_counts),
        },
        psd_uncertainty_semantics=psd_semantics,
    )


def _binary_mechanism_values(
    omega: np.ndarray, *, interpretation: str = "raw"
) -> dict[str, tuple[float, str]]:
    if interpretation not in {"raw", "psd_interpretable"}:
        raise ValueError("Unknown binary-summary interpretation.")
    v0 = float(omega[0, 0])
    v1 = float(omega[1, 1])
    gamma = float(omega[0, 1])
    linear_status = (
        "defined_raw_linear" if interpretation == "raw" else "defined_psd_interpretable"
    )
    values: dict[str, tuple[float, str]] = {
        "v0": (v0, linear_status),
        "v1": (v1, linear_status),
        "gamma": (gamma, linear_status),
    }
    if v0 > 0.0 and v1 > 0.0:
        positive_status = (
            "defined_positive_variances"
            if interpretation == "raw"
            else "defined_psd_interpretable_positive_variances"
        )
        values["rho"] = (gamma / sqrt(v0 * v1), positive_status)
        values["log_sd_ratio"] = (
            0.5 * float(np.log(v1 / v0)),
            positive_status,
        )
    else:
        values["rho"] = (
            float("nan"),
            "undefined_nonpositive_context_variance",
        )
        values["log_sd_ratio"] = (
            float("nan"),
            "undefined_nonpositive_context_variance",
        )
    if v0 > 0.0:
        values["tau2_1_given_0"] = (
            v1 - gamma * gamma / v0,
            (
                "defined_raw_may_be_negative"
                if interpretation == "raw"
                else "defined_psd_interpretable"
            ),
        )
    else:
        values["tau2_1_given_0"] = (
            float("nan"),
            "undefined_nonpositive_v0",
        )
    return values


def _binary_mechanism_estimates(
    omega: np.ndarray,
    loo_omegas: np.ndarray,
    *,
    interpretation: str = "raw",
) -> dict[str, JackknifeEstimate]:
    point = _binary_mechanism_values(omega, interpretation=interpretation)
    replicate = [
        _binary_mechanism_values(value, interpretation=interpretation)
        for value in loo_omegas
    ]
    return {
        name: _jackknife_estimate(
            value,
            [entry[name][0] for entry in replicate],
            point_status=status,
            loo_statuses=[entry[name][1] for entry in replicate],
        )
        for name, (value, status) in point.items()
    }


def _equal_variance_contrast(fit: ContextFitResult) -> LinearContrastResult:
    p_total = fit.raw_coefficients.size
    contrast = np.zeros(p_total, dtype=np.float64)
    contrast[0] = -1.0
    contrast[1] = 1.0
    estimate = float(contrast @ fit.raw_coefficients)
    variance = float(contrast @ fit.jackknife_covariance @ contrast)
    loo_values = fit.loo_coefficients @ contrast
    if variance <= 0.0 or not np.isfinite(variance):
        return LinearContrastResult(
            estimate,
            float("nan"),
            float("nan"),
            float("nan"),
            loo_values,
            "indeterminate_nonpositive_joint_contrast_variance",
            "joint_approximate_loo_covariance",
        )
    standard_error = sqrt(variance)
    z_statistic = estimate / standard_error
    return LinearContrastResult(
        estimate,
        standard_error,
        z_statistic,
        float(2.0 * norm.sf(abs(z_statistic))),
        loo_values,
        "defined_difference_test_not_equivalence_test",
        "joint_approximate_loo_covariance_including_v0_v1_covariance",
    )


def derive_binary_context_fit(
    fit: ContextFitResult,
    preset: CategoricalContextPreset,
    *,
    loo_equations: Sequence[ContextNormalEquations] | None = None,
) -> BinaryContextFitSummary:
    """Report binary stratum mechanisms without treating raw MoM as PSD."""
    if not preset.is_binary:
        raise ValueError("Binary derived quantities require exactly two categories.")
    if len(fit.component_index.annotation_names) != 1:
        raise ValueError(
            "The Stage-05 binary mechanism summary requires one annotation; "
            "annotation combinations remain available from the generic engine."
        )
    categorical = derive_categorical_context_fit(
        fit, preset, loo_equations=loo_equations
    )
    raw_omega = categorical.raw_omegas[0]
    loo_raw_omegas = categorical.loo_raw_omegas[:, 0]
    raw = _binary_mechanism_estimates(raw_omega, loo_raw_omegas)
    psd: dict[str, JackknifeEstimate] | None = None
    psd_omega: np.ndarray | None = None
    loo_psd_omegas: np.ndarray | None = None
    if categorical.psd_omegas is not None:
        psd_omega = categorical.psd_omegas[0]
        if categorical.loo_psd_omegas is None:
            mechanism_loo_psd_omegas = np.empty((0, 2, 2), dtype=np.float64)
        else:
            loo_psd_omegas = categorical.loo_psd_omegas[:, 0]
            mechanism_loo_psd_omegas = loo_psd_omegas
        psd = _binary_mechanism_estimates(
            psd_omega,
            mechanism_loo_psd_omegas,
            interpretation="psd_interpretable",
        )
    return BinaryContextFitSummary(
        category_labels=preset.category_labels,
        raw=raw,
        psd_interpretable=psd,
        raw_omega=raw_omega,
        loo_raw_omegas=loo_raw_omegas,
        psd_omega=psd_omega,
        loo_psd_omegas=loo_psd_omegas,
        residual_variances=categorical.residual_variances,
        trace_variance_proportions=categorical.trace_variance_proportions,
        psd_trace_variance_proportions=(categorical.psd_trace_variance_proportions),
        equal_variance_contrast=_equal_variance_contrast(fit),
        categorical=categorical,
    )


def categorical_context_covariance(
    fit: ContextFitResult,
    preset: CategoricalContextPreset,
    *,
    loo_equations: Sequence[ContextNormalEquations] | None = None,
) -> CategoricalContextFitSummary:
    """Public feature name for categorical covariance summaries."""
    return derive_categorical_context_fit(fit, preset, loo_equations=loo_equations)


def binary_context_covariance(
    fit: ContextFitResult,
    preset: CategoricalContextPreset,
    *,
    loo_equations: Sequence[ContextNormalEquations] | None = None,
) -> BinaryContextFitSummary:
    """Public feature name for binary covariance and mechanism summaries."""
    return derive_binary_context_fit(fit, preset, loo_equations=loo_equations)


def _boundary_null_point(
    theta: np.ndarray,
    covariance: np.ndarray,
    precision: np.ndarray,
    hypothesis: str,
) -> tuple[np.ndarray, float, bool]:
    if hypothesis == "equal_variances":
        contrast = np.asarray([-1.0, 1.0, 0.0])
        variance = float(contrast @ covariance @ contrast)
        if variance <= 0.0:
            return np.full(3, np.nan), float("nan"), False
        null = theta - covariance @ contrast * (contrast @ theta) / variance
        statistic = float((contrast @ theta) ** 2 / variance)
        return null, statistic, True

    # Both nonlinear nulls are subsets of the rank-one PSD cone.  Write a
    # rank-one point as u * (cos(alpha)^2, sin(alpha)^2,
    # cos(alpha)sin(alpha)), u >= 0.  For fixed alpha, the optimal u has a
    # closed form.  This reduces the formerly fragile two-dimensional local
    # optimization to a bounded one-dimensional global search.
    lower = 0.0 if hypothesis == "rho=1" else -0.5 * np.pi
    upper = 0.5 * np.pi

    def candidate_and_value(angle: float) -> tuple[np.ndarray, float]:
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))
        direction = np.asarray(
            [cosine * cosine, sine * sine, cosine * sine], dtype=np.float64
        )
        denominator = float(direction @ precision @ direction)
        precision_scale = max(
            float(np.max(np.abs(precision), initial=0.0)),
            np.finfo(np.float64).tiny,
        )
        if not np.isfinite(denominator) or denominator <= (
            100.0 * np.finfo(np.float64).eps * precision_scale
        ):
            return np.full(3, np.nan), float("inf")
        amplitude = max(float(direction @ precision @ theta) / denominator, 0.0)
        point = amplitude * direction
        difference = point - theta
        return point, float(difference @ precision @ difference)

    def objective(angle: float) -> float:
        return candidate_and_value(angle)[1]

    # A rank-one quadratic form produces only a small number of stationary
    # points in this one-dimensional parameterization.  The grid locates every
    # basin, and each basin is refined independently; grid points and both
    # boundaries remain explicit candidates so boundary infima fail closed.
    grid = np.linspace(lower, upper, 257, dtype=np.float64)
    values = np.asarray([objective(float(angle)) for angle in grid])
    candidate_angles = list(grid)
    for index in range(1, grid.size - 1):
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            result = minimize_scalar(
                objective,
                bounds=(float(grid[index - 1]), float(grid[index + 1])),
                method="bounded",
                options={"xatol": 1.0e-13, "maxiter": 200},
            )
            if np.isfinite(result.fun):
                candidate_angles.append(float(result.x))

    best_angle = float("nan")
    best_value = float("inf")
    best_null = np.full(3, np.nan)
    for angle in candidate_angles:
        point, value = candidate_and_value(angle)
        if np.isfinite(value) and value < best_value:
            best_angle = float(angle)
            best_value = value
            best_null = point

    if not np.isfinite(best_angle) or not np.all(np.isfinite(best_null)):
        return best_null, best_value, False
    null_scale = max(float(np.max(np.abs(best_null), initial=0.0)), 1.0e-300)
    at_boundary = min(abs(best_angle - lower), abs(best_angle - upper)) <= 1.0e-10
    if hypothesis == "rho=1":
        valid = (
            not at_boundary and min(best_null[0], best_null[1]) > 1.0e-12 * null_scale
        )
    else:
        valid = not at_boundary and best_null[0] > 1.0e-12 * null_scale
    return best_null, best_value, bool(valid)


def _boundary_covariance_precision(
    covariance: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    leading = float(np.max(np.abs(eigenvalues), initial=0.0))
    if not np.isfinite(leading) or leading <= 0.0:
        return np.zeros_like(symmetric), 0, False
    tolerance = max(100.0 * np.finfo(np.float64).eps, 1.0e-10) * leading
    if eigenvalues[0] < -tolerance:
        return np.zeros_like(symmetric), 0, False
    retained = eigenvalues > tolerance
    rank = int(np.sum(retained))
    if rank:
        precision = (eigenvectors[:, retained] / eigenvalues[retained]) @ eigenvectors[
            :, retained
        ].T
    else:
        precision = np.zeros_like(symmetric)
    return precision, rank, True


def test_binary_boundary(
    fit: ContextFitResult,
    hypothesis: str,
    *,
    preset: CategoricalContextPreset,
    draws: int = 999,
    seed: int = 0,
) -> BoundaryTestResult:
    """Experimental pseudo-value multiplier calibration for binary hypotheses.

    This is deliberately labelled experimental: the multiplier distribution is
    conditional on the approximate SNP-group jackknife and must pass empirical
    null calibration before being used as a production significance claim.
    """
    _validate_categorical_fit(fit, preset)
    if not preset.is_binary:
        raise ValueError("Binary boundary tests require a binary categorical preset.")
    aliases = {
        "rho=1": "rho=1",
        "rho_equal_one": "rho=1",
        "tau2=0": "tau2=0",
        "tau2_equal_zero": "tau2=0",
        "equal_variances": "equal_variances",
    }
    try:
        canonical_hypothesis = aliases[str(hypothesis)]
    except KeyError as exc:
        raise ValueError(
            "hypothesis must be 'rho=1', 'tau2=0', or 'equal_variances'."
        ) from exc
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer.")
    if fit.component_index.pair_index.num_basis != 2:
        raise ValueError("Binary boundary tests require a two-column context basis.")
    if len(fit.component_index.annotation_names) != 1:
        raise ValueError("Binary boundary tests currently require one annotation.")
    theta = np.asarray(fit.genetic_coefficients, dtype=np.float64)
    loo = np.asarray(fit.loo_coefficients[:, :3], dtype=np.float64)
    covariance = np.asarray(fit.jackknife_covariance[:3, :3], dtype=np.float64)
    precision, covariance_rank, covariance_valid = _boundary_covariance_precision(
        covariance
    )
    empty = np.empty(0, dtype=np.float64)
    method = (
        "approximate_loo_multiplier_pseudovalue_rademacher_studentized_"
        "minimum_distance_v2"
    )
    if not covariance_valid:
        return BoundaryTestResult(
            canonical_hypothesis,
            float("nan"),
            float("nan"),
            float("nan"),
            np.full(3, np.nan),
            empty,
            covariance_rank,
            "indeterminate_invalid_joint_jackknife_covariance",
            method,
            float("nan"),
            int(seed),
        )
    if canonical_hypothesis != "equal_variances" and covariance_rank < 3:
        return BoundaryTestResult(
            canonical_hypothesis,
            float("nan"),
            float("nan"),
            float("nan"),
            np.full(3, np.nan),
            empty,
            covariance_rank,
            "indeterminate_rank_deficient_joint_jackknife_covariance",
            method,
            float("nan"),
            int(seed),
        )
    null, statistic, success = _boundary_null_point(
        theta, covariance, precision, canonical_hypothesis
    )
    if not success or not np.isfinite(statistic):
        return BoundaryTestResult(
            canonical_hypothesis,
            float("nan"),
            float("nan"),
            float("nan"),
            null,
            empty,
            covariance_rank,
            "indeterminate_null_projection_failure",
            method,
            float("nan"),
            int(seed),
        )
    group_count = loo.shape[0]
    if group_count < 6:
        return BoundaryTestResult(
            canonical_hypothesis,
            statistic,
            float("nan"),
            float(chi2.sf(statistic, 1)),
            null,
            empty,
            covariance_rank,
            "indeterminate_fewer_than_six_jackknife_groups",
            method,
            float("nan"),
            int(seed),
        )
    pseudo_values = group_count * theta[None, :] - (group_count - 1.0) * loo
    pseudo_centered = pseudo_values - np.mean(pseudo_values, axis=0, keepdims=True)
    pseudo_covariance = (
        pseudo_centered.T @ pseudo_centered / (group_count * (group_count - 1.0))
    )
    covariance_error = float(np.max(np.abs(pseudo_covariance - covariance)))
    covariance_scale = max(
        float(np.max(np.abs(covariance), initial=0.0)), np.finfo(np.float64).tiny
    )
    if covariance_error > 1.0e-10 * covariance_scale:
        return BoundaryTestResult(
            canonical_hypothesis,
            statistic,
            float("nan"),
            float(chi2.sf(statistic, 1)),
            null,
            empty,
            covariance_rank,
            "indeterminate_pseudovalue_covariance_mismatch",
            method,
            covariance_error,
            int(seed),
        )
    rng = np.random.default_rng(seed)
    multipliers = rng.choice(
        np.asarray([-1.0, 1.0]), size=(draws, group_count), replace=True
    )
    bootstrap = np.empty(draws, dtype=np.float64)
    failed = 0
    pseudo_scale = sqrt(group_count / (group_count - 1.0))
    for index, multiplier in enumerate(multipliers):
        pseudo_star = null[None, :] + (
            pseudo_scale * multiplier[:, None] * pseudo_centered
        )
        bootstrap_theta = np.mean(pseudo_star, axis=0)
        bootstrap_centered = pseudo_star - bootstrap_theta[None, :]
        bootstrap_covariance = (
            bootstrap_centered.T
            @ bootstrap_centered
            / (group_count * (group_count - 1.0))
        )
        (
            bootstrap_precision,
            bootstrap_rank,
            bootstrap_covariance_valid,
        ) = _boundary_covariance_precision(bootstrap_covariance)
        if not bootstrap_covariance_valid or (
            canonical_hypothesis != "equal_variances" and bootstrap_rank < 3
        ):
            bootstrap[index] = np.nan
            failed += 1
            continue
        _, value, replicate_success = _boundary_null_point(
            bootstrap_theta,
            bootstrap_covariance,
            bootstrap_precision,
            canonical_hypothesis,
        )
        if replicate_success and np.isfinite(value):
            bootstrap[index] = value
        else:
            bootstrap[index] = np.nan
            failed += 1
    if failed:
        return BoundaryTestResult(
            canonical_hypothesis,
            statistic,
            float("nan"),
            float(chi2.sf(statistic, 1)),
            null,
            bootstrap,
            covariance_rank,
            f"indeterminate_multiplier_projection_failures:{failed}",
            method,
            covariance_error,
            int(seed),
        )
    p_value = float((1 + np.sum(bootstrap >= statistic)) / (draws + 1))
    return BoundaryTestResult(
        canonical_hypothesis,
        statistic,
        p_value,
        float(chi2.sf(statistic, 1)),
        null,
        bootstrap,
        covariance_rank,
        "experimental_multiplier_calibration_defined",
        method,
        covariance_error,
        int(seed),
    )


# Avoid accidental pytest collection when this public function is imported by a test.
test_binary_boundary.__test__ = False

# Generic public feature alias; the test-prefixed name remains for compatibility.
binary_context_boundary_test = test_binary_boundary
