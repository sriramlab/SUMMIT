#!/usr/bin/env python3
"""Validate binary and categorical contextual-covariance presets.

The script runs small deterministic synthetic mechanisms through dense,
matched-summary, and independent-reference fits.  It also checks binary
one-hot versus intercept encoding, categorical permutation equivariance, every
approximate-LOO replicate, and descriptive Monte Carlo bias/coverage.  Only
aggregate results are written; no sample- or variant-level arrays are emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from summit.context import (
    ContextComponentIndex,
    ContextJackknifeError,
    ContextPairIndex,
    ContextRankError,
    array_sha256,
    binary_basis_transform,
    binary_intercept_to_one_hot_covariance,
    binary_intercept_to_one_hot_coefficients,
    binary_one_hot_to_intercept_covariance,
    binary_one_hot_to_intercept_coefficients,
    build_categorical_context_preset,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    coefficients_to_omegas,
    combine_genetic_kernels,
    common_scale_features,
    dense_genetic_kernels,
    dense_normal_equations,
    dense_residual_kernels,
    derive_binary_context_fit,
    derive_categorical_context_fit,
    fit_context_model,
    omegas_to_coefficients,
    project_normalize_phenotype,
    rank_revealing_projector,
    symmetric_rank_diagnostics,
    test_binary_boundary,
)


OUTPUT_STEM = "05_binary_categorical_validation"
GENOTYPE_SCALING = "synthetic_population_unit_variance"
NORMAL_CRITICAL_VALUE = 1.959963984540054


@dataclass(frozen=True)
class Mechanism:
    name: str
    label: str
    category_probabilities: tuple[float, ...]
    omega: np.ndarray
    residual_variances: np.ndarray
    genotype_pc_loading: float = 0.15
    context_pc_correlation: float = 0.0

    @property
    def categories(self) -> tuple[int, ...]:
        return tuple(range(len(self.category_probabilities)))

    @property
    def binary(self) -> bool:
        return len(self.category_probabilities) == 2


@dataclass(frozen=True)
class Cohort:
    mechanism: Mechanism
    genotype: np.ndarray
    preset: Any
    fixed: np.ndarray
    projector: Any
    annotations: np.ndarray
    components: ContextComponentIndex
    genetic_kernels: np.ndarray
    residual_kernels: np.ndarray
    loo_group_ids: tuple[str, ...]
    hashes: dict[str, str]


@dataclass(frozen=True)
class PhenotypeDraw:
    raw: np.ndarray
    normalized: np.ndarray
    normalization_multiplier: float
    minimum_covariance_eigenvalue: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--study-n", type=int, default=96)
    parser.add_argument("--reference-n", type=int, default=192)
    parser.add_argument("--m", type=int, default=72, help="Synthetic variants.")
    parser.add_argument("--loo-groups", type=int, default=6)
    parser.add_argument("--boundary-draws", type=int, default=512)
    parser.add_argument(
        "--boundary-calibration-draws",
        type=int,
        default=128,
        help=(
            "Multiplier draws per hypothesis and replicate in the descriptive "
            "no-context null experiment."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.replicates < 4:
        parser.error("--replicates must be at least 4")
    if args.study_n < 48:
        parser.error("--study-n must be at least 48")
    if args.reference_n < 48:
        parser.error("--reference-n must be at least 48")
    if args.m < 24:
        parser.error("--m must be at least 24")
    if args.loo_groups < 6 or args.loo_groups > args.m // 2:
        parser.error("--loo-groups must satisfy 6 <= groups <= M/2")
    if args.m % args.loo_groups:
        parser.error("--m must be divisible by --loo-groups")
    if args.boundary_draws < 32:
        parser.error("--boundary-draws must be at least 32")
    if args.boundary_calibration_draws < 32:
        parser.error("--boundary-calibration-draws must be at least 32")
    if args.seed < 0:
        parser.error("--seed must be non-negative")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _timed(function: Callable[[], Any]) -> tuple[Any, float, int]:
    started = time.perf_counter()
    result = function()
    return result, float(time.perf_counter() - started), _peak_rss_bytes()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _estimate_summary(value: Any) -> dict[str, Any]:
    return {
        "estimate": _json_safe(float(value.estimate)),
        "standard_error": _json_safe(float(value.standard_error)),
        "status": str(value.status),
        "loo_replicates": int(np.asarray(value.loo_values).size),
    }


def _categorical_annotation_summary(value: Any) -> dict[str, Any]:
    return {
        "annotation_name": str(value.annotation_name),
        "omega": np.asarray(value.omega).tolist(),
        "variances": [_estimate_summary(item) for item in value.variances],
        "covariances": [
            [_estimate_summary(item) for item in row] for row in value.covariances
        ],
        "correlations": [
            [_estimate_summary(item) for item in row] for row in value.correlations
        ],
        "loo_replicates": int(np.asarray(value.loo_omegas).shape[0]),
    }


def _categorical_derived_summary(value: Any) -> dict[str, Any]:
    return {
        "category_labels": list(value.category_labels),
        "category_counts": list(value.category_counts),
        "raw": {
            str(name): _categorical_annotation_summary(summary)
            for name, summary in value.raw.items()
        },
        "raw_omegas": np.asarray(value.raw_omegas).tolist(),
        "psd_interpretable": (
            None
            if value.psd_interpretable is None
            else {
                str(name): _categorical_annotation_summary(summary)
                for name, summary in value.psd_interpretable.items()
            }
        ),
        "psd_omegas": (
            None if value.psd_omegas is None else np.asarray(value.psd_omegas).tolist()
        ),
        "residual_variances": {
            str(name): _estimate_summary(estimate)
            for name, estimate in value.residual_variances.items()
        },
        "trace_variance_proportions": {
            str(name): _estimate_summary(estimate)
            for name, estimate in value.trace_variance_proportions.items()
        },
        "psd_trace_variance_proportions": (
            None
            if value.psd_trace_variance_proportions is None
            else {
                str(name): _estimate_summary(estimate)
                for name, estimate in value.psd_trace_variance_proportions.items()
            }
        ),
        "prevalence_diagnostics": _json_safe(value.prevalence_diagnostics),
        "psd_uncertainty_semantics": value.psd_uncertainty_semantics,
    }


def _binary_derived_summary(value: Any) -> dict[str, Any]:
    contrast = value.equal_variance_contrast
    return {
        "category_labels": list(value.category_labels),
        "raw": {
            str(name): _estimate_summary(estimate)
            for name, estimate in value.raw.items()
        },
        "psd_interpretable": (
            None
            if value.psd_interpretable is None
            else {
                str(name): _estimate_summary(estimate)
                for name, estimate in value.psd_interpretable.items()
            }
        ),
        "raw_omega": np.asarray(value.raw_omega).tolist(),
        "psd_omega": (
            None if value.psd_omega is None else np.asarray(value.psd_omega).tolist()
        ),
        "residual_variances": {
            str(name): _estimate_summary(estimate)
            for name, estimate in value.residual_variances.items()
        },
        "trace_variance_proportions": {
            str(name): _estimate_summary(estimate)
            for name, estimate in value.trace_variance_proportions.items()
        },
        "equal_variance_contrast": {
            "estimate": _json_safe(float(contrast.estimate)),
            "standard_error": _json_safe(float(contrast.standard_error)),
            "z_statistic": _json_safe(float(contrast.z_statistic)),
            "p_value": _json_safe(float(contrast.p_value)),
            "status": str(contrast.status),
            "covariance_method": str(contrast.covariance_method),
            "loo_replicates": int(np.asarray(contrast.loo_values).size),
        },
        "categorical": _categorical_derived_summary(value.categorical),
    }


def _psd_derived_summary(fit: Any, preset: Any, *, binary: bool) -> dict[str, Any]:
    if binary:
        derived = derive_binary_context_fit(fit, preset)
        semantics = derived.categorical.psd_uncertainty_semantics
        summary = _binary_derived_summary(derived)
    else:
        derived = derive_categorical_context_fit(fit, preset)
        semantics = derived.psd_uncertainty_semantics
        summary = _categorical_derived_summary(derived)
    return {
        "status": (
            "defined_point_psd_indeterminate_projected_loo_uncertainty"
            if semantics is not None and "indeterminate" in semantics
            else "defined"
        ),
        "psd_uncertainty_semantics": semantics,
        "summary": summary,
    }


def _maximum_absolute(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Discrepancy inputs differ in shape: {left_array.shape}, "
            f"{right_array.shape}."
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _mechanisms() -> tuple[Mechanism, ...]:
    constant = 0.28 * np.ones((2, 2), dtype=np.float64)
    amplification = np.outer(
        np.asarray([0.42, 0.70], dtype=np.float64),
        np.asarray([0.42, 0.70], dtype=np.float64),
    )
    imperfect_variance = 0.30
    imperfect_rho = 0.55
    imperfect = np.asarray(
        [
            [imperfect_variance, imperfect_rho * imperfect_variance],
            [imperfect_rho * imperfect_variance, imperfect_variance],
        ],
        dtype=np.float64,
    )
    mixed_v0, mixed_v1, mixed_rho = 0.18, 0.42, 0.65
    mixed = np.asarray(
        [
            [mixed_v0, mixed_rho * np.sqrt(mixed_v0 * mixed_v1)],
            [mixed_rho * np.sqrt(mixed_v0 * mixed_v1), mixed_v1],
        ],
        dtype=np.float64,
    )
    categorical_standard_deviations = np.sqrt(
        np.asarray([0.20, 0.35, 0.28], dtype=np.float64)
    )
    categorical_correlations = np.asarray(
        [[1.0, 0.80, 0.50], [0.80, 1.0, 0.60], [0.50, 0.60, 1.0]],
        dtype=np.float64,
    )
    categorical = (
        categorical_standard_deviations[:, None]
        * categorical_correlations
        * categorical_standard_deviations[None, :]
    )
    return (
        Mechanism(
            "no_context_dependence",
            "No context",
            (0.5, 0.5),
            constant,
            np.asarray([0.72, 0.72]),
        ),
        Mechanism(
            "proportional_rank_one_amplification",
            "Rank-one amp.",
            (0.5, 0.5),
            amplification,
            np.asarray([0.55, 0.55]),
        ),
        Mechanism(
            "imperfect_genetic_correlation",
            "Imperfect corr.",
            (0.5, 0.5),
            imperfect,
            np.asarray([0.70, 0.70]),
        ),
        Mechanism(
            "mixed_amplification_and_heterogeneity",
            "Mixed",
            (0.5, 0.5),
            mixed,
            np.asarray([0.65, 0.65]),
        ),
        Mechanism(
            "residual_heterogeneity_only",
            "Residual het.",
            (0.5, 0.5),
            constant,
            np.asarray([0.45, 0.90]),
        ),
        Mechanism(
            "unbalanced_prevalence",
            "Prevalence 0.2",
            (0.8, 0.2),
            mixed,
            np.asarray([0.65, 0.65]),
        ),
        Mechanism(
            "context_correlated_pc",
            "Context-PC corr.",
            (0.5, 0.5),
            constant,
            np.asarray([0.72, 0.72]),
            genotype_pc_loading=0.35,
            context_pc_correlation=0.65,
        ),
        Mechanism(
            "three_category_context",
            "Three category",
            (0.45, 0.35, 0.20),
            categorical,
            np.asarray([0.65, 0.55, 0.75]),
        ),
    )


def _category_sample(
    rng: np.random.Generator,
    n: int,
    probabilities: Sequence[float],
) -> np.ndarray:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    expected = n * probabilities_array
    counts = np.floor(expected).astype(np.int64)
    order = np.argsort(-(expected - counts), kind="stable")
    for index in order[: n - int(np.sum(counts))]:
        counts[index] += 1
    if np.any(counts < 2):
        raise ValueError("Synthetic categories require at least two observations each.")
    context = np.concatenate(
        [
            np.full(count, category, dtype=np.int64)
            for category, count in enumerate(counts)
        ]
    )
    rng.shuffle(context)
    return context


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise RuntimeError("Synthetic fixed-effect column has invalid scale.")
    return np.asarray(centered / scale, dtype=np.float64)


def _variant_loadings(m: int, maximum: float) -> np.ndarray:
    phase = 2.0 * np.pi * (np.arange(m, dtype=np.float64) + 0.5) / m
    pattern = np.sin(phase) + 0.35 * np.cos(3.0 * phase)
    pattern /= np.max(np.abs(pattern))
    return maximum * pattern


def _build_cohort(
    mechanism: Mechanism,
    *,
    rng: np.random.Generator,
    n: int,
    m: int,
    loo_groups: int,
) -> Cohort:
    context = _category_sample(rng, n, mechanism.category_probabilities)
    preset = build_categorical_context_preset(
        context,
        categories=mechanism.categories,
        source_name=f"{mechanism.name}_context",
        binary=mechanism.binary,
    )
    category_score = np.linspace(-1.0, 1.0, len(mechanism.categories))[context]
    category_score = _standardize(category_score[:, None])[:, 0]
    independent_pc = rng.standard_normal(n)
    loading = mechanism.context_pc_correlation
    pc = loading * category_score + np.sqrt(1.0 - loading * loading) * independent_pc
    pc = _standardize(pc[:, None])[:, 0]
    nuisance = _standardize(rng.standard_normal((n, 1)))
    fixed = preset.fixed_effect_design(np.column_stack([pc, nuisance]))
    projector = rank_revealing_projector(fixed)

    variant_loadings = _variant_loadings(m, mechanism.genotype_pc_loading)
    genotype = (
        pc[:, None] * variant_loadings[None, :]
        + rng.standard_normal((n, m))
        * np.sqrt(1.0 - variant_loadings * variant_loadings)[None, :]
    )
    genotype = np.asarray(genotype, dtype=np.float64)
    annotations = np.ones((m, 1), dtype=np.float64)
    components = preset.component_index
    features = common_scale_features(genotype, preset.basis, projector.projector)
    genetic_kernels = dense_genetic_kernels(features, annotations, components)
    residual_kernels = dense_residual_kernels(
        projector.projector, preset.residual_basis
    )
    group_ids = tuple(f"block:{index % loo_groups}" for index in range(m))
    hashes = {
        "basis_hash": preset.basis_hash,
        "fixed_effect_hash": array_sha256(fixed),
        "variant_hash": canonical_sha256(
            {"ordered_synthetic_variants": list(range(m))}
        ),
    }
    return Cohort(
        mechanism=mechanism,
        genotype=genotype,
        preset=preset,
        fixed=np.asarray(fixed, dtype=np.float64),
        projector=projector,
        annotations=annotations,
        components=components,
        genetic_kernels=genetic_kernels,
        residual_kernels=residual_kernels,
        loo_group_ids=group_ids,
        hashes=hashes,
    )


def _build_reference(cohort: Cohort) -> Any:
    return build_context_reference(
        genotype=cohort.genotype,
        basis=cohort.preset.basis,
        projector=cohort.projector,
        annotations=cohort.annotations,
        component_index=cohort.components,
        loo_groups=cohort.loo_group_ids,
        genotype_scaling=GENOTYPE_SCALING,
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=8,
        **cohort.hashes,
    )


def _build_summary(cohort: Cohort, phenotype: np.ndarray) -> Any:
    return build_context_trait_summary(
        genotype=cohort.genotype,
        basis=cohort.preset.basis,
        phenotype=phenotype,
        projector=cohort.projector,
        annotations=cohort.annotations,
        component_index=cohort.components,
        residual_basis=cohort.preset.residual_basis,
        residual_names=cohort.preset.residual_names,
        loo_groups=cohort.loo_group_ids,
        genotype_scaling=GENOTYPE_SCALING,
        block_size=cohort.genotype.shape[1],
        **cohort.hashes,
    )


def _fit(
    reference: Any, summary: Any, preset: Any, *, project_psd: bool = False
) -> Any:
    return fit_context_model(
        reference,
        summary,
        loo_groups=tuple(dict.fromkeys(summary.loo_group_ids)),
        context_grid=preset.context_grid,
        basis_metric=preset.basis_metric,
        project_psd=project_psd,
        annotations_disjoint=True,
    )


def _sample_phenotype(cohort: Cohort, rng: np.random.Generator) -> PhenotypeDraw:
    genetic_coefficients = omegas_to_coefficients(
        cohort.mechanism.omega[None, :, :], cohort.components
    )
    covariance = np.einsum(
        "a,aij->ij", genetic_coefficients, cohort.genetic_kernels, optimize=True
    ) + np.einsum(
        "a,aij->ij",
        cohort.mechanism.residual_variances,
        cohort.residual_kernels,
        optimize=True,
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-10 * covariance_scale:
        raise RuntimeError("Synthetic phenotype covariance is not PSD.")
    random_part = eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0.0)) * rng.standard_normal(eigenvalues.size)
    )
    fixed_signal = cohort.fixed @ np.linspace(0.03, 0.12, cohort.fixed.shape[1])
    raw = np.asarray(random_part + fixed_signal, dtype=np.float64)
    projected = cohort.projector.projector @ raw
    multiplier = cohort.projector.residual_rank / float(projected @ projected)
    normalized = project_normalize_phenotype(raw, cohort.projector)
    return PhenotypeDraw(
        raw=raw,
        normalized=normalized,
        normalization_multiplier=float(multiplier),
        minimum_covariance_eigenvalue=float(np.min(eigenvalues)),
    )


def _truth_coefficients(cohort: Cohort, multiplier: float) -> np.ndarray:
    return multiplier * np.concatenate(
        [
            omegas_to_coefficients(
                cohort.mechanism.omega[None, :, :], cohort.components
            ),
            cohort.mechanism.residual_variances,
        ]
    )


def _surfaces(
    coefficients: np.ndarray, components: ContextComponentIndex
) -> np.ndarray:
    omegas = coefficients_to_omegas(coefficients, components)
    return np.asarray(omegas, dtype=np.float64)


def _dense_fit(cohort: Cohort, draw: PhenotypeDraw) -> tuple[Any, np.ndarray, Any]:
    equations = dense_normal_equations(
        cohort.genetic_kernels,
        cohort.residual_kernels,
        draw.normalized,
        cohort.components.names,
        cohort.preset.residual_names,
    )
    diagnostics = symmetric_rank_diagnostics(equations.matrix, rtol=1.0e-10)
    if diagnostics.rank != equations.matrix.shape[0]:
        raise RuntimeError(
            f"Dense mechanism {cohort.mechanism.name!r} is rank deficient: "
            f"{diagnostics.rank}/{equations.matrix.shape[0]}."
        )
    coefficients = np.linalg.solve(
        0.5 * (equations.matrix + equations.matrix.T), equations.rhs
    )
    return equations, np.asarray(coefficients, dtype=np.float64), diagnostics


def _jackknife_covariance(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True)
    covariance = (array.shape[0] - 1.0) / array.shape[0] * centered.T @ centered
    return 0.5 * (covariance + covariance.T)


def _binary_quantities(genetic_coefficients: object) -> np.ndarray:
    values = np.asarray(genetic_coefficients, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("Binary genetic coefficients must end in length three.")
    v0 = values[..., 0]
    v1 = values[..., 1]
    gamma = values[..., 2]
    positive = (v0 > 0.0) & (v1 > 0.0)
    rho = np.full_like(v0, np.nan, dtype=np.float64)
    log_amplification = np.full_like(v0, np.nan, dtype=np.float64)
    rho[positive] = gamma[positive] / np.sqrt(v0[positive] * v1[positive])
    log_amplification[positive] = 0.5 * np.log(v1[positive] / v0[positive])
    tau = np.full_like(v0, np.nan, dtype=np.float64)
    nonzero_v0 = v0 > 0.0
    tau[nonzero_v0] = v1[nonzero_v0] - gamma[nonzero_v0] ** 2 / v0[nonzero_v0]
    return np.stack([v0, v1, gamma, rho, log_amplification, tau, v1 - v0], axis=-1)


BINARY_QUANTITY_NAMES = (
    "v0",
    "v1",
    "gamma",
    "rho",
    "log_sd_amplification",
    "tau2_1_given_0",
    "variance_difference_v1_minus_v0",
)


def _derived_point_se(fit: Any) -> tuple[np.ndarray, np.ndarray]:
    point = _binary_quantities(fit.genetic_coefficients)
    loo = _binary_quantities(fit.loo_coefficients[:, :3])
    standard_errors = np.full(point.shape, np.nan, dtype=np.float64)
    for index in range(point.size):
        column = loo[:, index]
        if np.all(np.isfinite(column)):
            centered = column - np.mean(column)
            variance = (column.size - 1.0) / column.size * float(centered @ centered)
            standard_errors[index] = np.sqrt(max(variance, 0.0))
    return point, standard_errors


def _summarize_estimates(
    estimates: np.ndarray,
    truths: np.ndarray,
    standard_errors: np.ndarray,
    names: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        estimate = estimates[:, index]
        truth = truths[:, index]
        standard_error = standard_errors[:, index]
        finite = np.isfinite(estimate) & np.isfinite(truth)
        inference = finite & np.isfinite(standard_error)
        errors = estimate[finite] - truth[finite]
        coverage = (
            np.abs(estimate[inference] - truth[inference])
            <= NORMAL_CRITICAL_VALUE * standard_error[inference]
        )
        result[str(name)] = {
            "defined_replicates": int(np.sum(finite)),
            "inference_replicates": int(np.sum(inference)),
            "mean_truth": (
                None if not np.any(finite) else float(np.mean(truth[finite]))
            ),
            "mean_estimate": (
                None if not np.any(finite) else float(np.mean(estimate[finite]))
            ),
            "bias": None if not errors.size else float(np.mean(errors)),
            "rmse": (
                None if not errors.size else float(np.sqrt(np.mean(errors * errors)))
            ),
            "empirical_standard_deviation": (
                None if np.sum(finite) < 2 else float(np.std(estimate[finite], ddof=1))
            ),
            "mean_jackknife_standard_error": (
                None
                if not np.any(inference)
                else float(np.mean(standard_error[inference]))
            ),
            "naive_95_percent_coverage": (
                None if not coverage.size else float(np.mean(coverage))
            ),
            "coverage_monte_carlo_standard_error": (
                None
                if not coverage.size
                else float(
                    np.sqrt(
                        np.mean(coverage) * (1.0 - np.mean(coverage)) / coverage.size
                    )
                )
            ),
        }
    return result


def _fit_diagnostics(fits: Sequence[Any]) -> dict[str, Any]:
    conditions = np.asarray([fit.solve.condition_number for fit in fits])
    residuals = np.asarray([fit.solve.relative_residual for fit in fits])
    minimum_jackknife = np.asarray(
        [np.min(np.linalg.eigvalsh(fit.jackknife_covariance)) for fit in fits]
    )
    return {
        "ranks": sorted({int(fit.solve.rank) for fit in fits}),
        "maximum_condition_number": float(np.max(conditions)),
        "maximum_relative_solve_residual": float(np.max(residuals)),
        "minimum_jackknife_covariance_eigenvalue": float(np.min(minimum_jackknife)),
    }


def _encoding_equivalence(
    cohort: Cohort,
    draw: PhenotypeDraw,
    one_hot_fit: Any,
) -> dict[str, Any]:
    if not cohort.mechanism.binary:
        raise ValueError("Binary encoding equivalence requires two categories.")
    transform = np.asarray(binary_basis_transform(), dtype=np.float64)
    intercept_basis = np.asarray(cohort.preset.basis @ transform.T, dtype=np.float64)
    intercept_grid = np.asarray(
        cohort.preset.context_grid @ transform.T, dtype=np.float64
    )
    intercept_metric = np.asarray(
        transform @ cohort.preset.basis_metric @ transform.T, dtype=np.float64
    )
    components = ContextComponentIndex(
        cohort.components.annotation_names, ContextPairIndex(2)
    )
    basis_hash = canonical_sha256(
        {
            "kind": "binary_intercept_context_basis",
            "one_hot_category_order_hash": cohort.preset.category_order_hash,
            "transform": transform.tolist(),
        }
    )
    common = {
        "genotype": cohort.genotype,
        "basis": intercept_basis,
        "projector": cohort.projector,
        "annotations": cohort.annotations,
        "component_index": components,
        "loo_groups": cohort.loo_group_ids,
        "basis_hash": basis_hash,
        "fixed_effect_hash": cohort.hashes["fixed_effect_hash"],
        "variant_hash": cohort.hashes["variant_hash"],
        "genotype_scaling": GENOTYPE_SCALING,
    }
    reference = build_context_reference(
        **common,
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=8,
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=draw.raw,
        residual_basis=cohort.preset.residual_basis,
        residual_names=cohort.preset.residual_names,
        block_size=cohort.genotype.shape[1],
    )
    intercept_fit = fit_context_model(
        reference,
        summary,
        loo_groups=tuple(dict.fromkeys(cohort.loo_group_ids)),
        context_grid=intercept_grid,
        basis_metric=intercept_metric,
        project_psd=False,
        annotations_disjoint=True,
    )

    one_hot_genetic = np.asarray(one_hot_fit.genetic_coefficients, dtype=np.float64)
    intercept_genetic = np.asarray(intercept_fit.genetic_coefficients, dtype=np.float64)
    expected_intercept = np.asarray(
        binary_one_hot_to_intercept_coefficients(one_hot_genetic),
        dtype=np.float64,
    )
    recovered_one_hot = np.asarray(
        binary_intercept_to_one_hot_coefficients(intercept_genetic),
        dtype=np.float64,
    )
    expected_intercept_loo = np.asarray(
        binary_one_hot_to_intercept_coefficients(one_hot_fit.loo_coefficients[:, :3]),
        dtype=np.float64,
    )
    recovered_one_hot_loo = np.asarray(
        binary_intercept_to_one_hot_coefficients(intercept_fit.loo_coefficients[:, :3]),
        dtype=np.float64,
    )
    expected_intercept_covariance = np.asarray(
        binary_one_hot_to_intercept_covariance(
            one_hot_fit.jackknife_covariance[:3, :3]
        ),
        dtype=np.float64,
    )
    recovered_one_hot_covariance = np.asarray(
        binary_intercept_to_one_hot_covariance(
            intercept_fit.jackknife_covariance[:3, :3]
        ),
        dtype=np.float64,
    )

    one_hot_surface = np.asarray(one_hot_fit.raw_omegas[0], dtype=np.float64)
    intercept_surface = np.asarray(
        intercept_grid @ intercept_fit.raw_omegas[0] @ intercept_grid.T,
        dtype=np.float64,
    )
    loo_surface_errors = []
    for one_hot_values, intercept_values in zip(
        one_hot_fit.loo_coefficients[:, :3],
        intercept_fit.loo_coefficients[:, :3],
    ):
        one_hot_omega = coefficients_to_omegas(one_hot_values, cohort.components)[0]
        intercept_omega = coefficients_to_omegas(intercept_values, components)[0]
        loo_surface_errors.append(
            _maximum_absolute(
                one_hot_omega, intercept_grid @ intercept_omega @ intercept_grid.T
            )
        )

    intercept_features = common_scale_features(
        cohort.genotype, intercept_basis, cohort.projector.projector
    )
    intercept_kernels = dense_genetic_kernels(
        intercept_features, cohort.annotations, components
    )
    one_hot_fitted_covariance = combine_genetic_kernels(
        cohort.genetic_kernels, one_hot_fit.genetic_coefficients
    ) + combine_genetic_kernels(
        cohort.residual_kernels, one_hot_fit.residual_coefficients
    )
    intercept_fitted_covariance = combine_genetic_kernels(
        intercept_kernels, intercept_fit.genetic_coefficients
    ) + combine_genetic_kernels(
        cohort.residual_kernels, intercept_fit.residual_coefficients
    )

    coefficient_transform = np.asarray(
        binary_one_hot_to_intercept_coefficients(np.eye(3)),
        dtype=np.float64,
    ).T
    full_transform = np.zeros(
        (one_hot_fit.raw_coefficients.size, one_hot_fit.raw_coefficients.size),
        dtype=np.float64,
    )
    full_transform[:3, :3] = coefficient_transform
    full_transform[3:, 3:] = np.eye(one_hot_fit.raw_coefficients.size - 3)
    expected_full_covariance = (
        full_transform @ one_hot_fit.jackknife_covariance @ full_transform.T
    )

    return {
        "basis_transform": transform.tolist(),
        "coefficient_one_hot_to_intercept_max_abs": _maximum_absolute(
            intercept_genetic, expected_intercept
        ),
        "coefficient_intercept_to_one_hot_max_abs": _maximum_absolute(
            one_hot_genetic, recovered_one_hot
        ),
        "residual_coefficients_max_abs": _maximum_absolute(
            one_hot_fit.residual_coefficients,
            intercept_fit.residual_coefficients,
        ),
        "loo_one_hot_to_intercept_max_abs": _maximum_absolute(
            intercept_fit.loo_coefficients[:, :3], expected_intercept_loo
        ),
        "loo_intercept_to_one_hot_max_abs": _maximum_absolute(
            one_hot_fit.loo_coefficients[:, :3], recovered_one_hot_loo
        ),
        "loo_surface_max_abs": max(loo_surface_errors, default=0.0),
        "surface_max_abs": _maximum_absolute(one_hot_surface, intercept_surface),
        "phenotype_covariance_max_abs": _maximum_absolute(
            one_hot_fitted_covariance, intercept_fitted_covariance
        ),
        "genetic_jackknife_covariance_one_hot_to_intercept_max_abs": (
            _maximum_absolute(
                intercept_fit.jackknife_covariance[:3, :3],
                expected_intercept_covariance,
            )
        ),
        "genetic_jackknife_covariance_intercept_to_one_hot_max_abs": (
            _maximum_absolute(
                one_hot_fit.jackknife_covariance[:3, :3],
                recovered_one_hot_covariance,
            )
        ),
        "full_jackknife_covariance_max_abs": _maximum_absolute(
            intercept_fit.jackknife_covariance, expected_full_covariance
        ),
        "one_hot_condition_number": float(one_hot_fit.solve.condition_number),
        "intercept_condition_number": float(intercept_fit.solve.condition_number),
    }


def _block_structure_diagnostics(cohort: Cohort, equations: Any) -> dict[str, Any]:
    category_count = len(cohort.preset.category_labels)
    p_genetic = len(cohort.components)
    genetic_diagonal_cross = np.asarray(
        [
            abs(equations.matrix[left, right])
            for left in range(category_count)
            for right in range(left + 1, category_count)
        ],
        dtype=np.float64,
    )
    residual_cross = np.asarray(
        [
            abs(equations.matrix[p_genetic + left, p_genetic + right])
            for left in range(category_count)
            for right in range(left + 1, category_count)
        ],
        dtype=np.float64,
    )
    off_diagonal_kernel_norms = np.asarray(
        [
            np.linalg.norm(cohort.genetic_kernels[entry.index])
            for entry in cohort.components.entries
            if entry.q != entry.r
        ],
        dtype=np.float64,
    )
    maximum_cross_entry = float(
        np.max(
            np.concatenate([genetic_diagonal_cross, residual_cross]),
            initial=0.0,
        )
    )
    return {
        "genetic_diagonal_component_cross_entries": genetic_diagonal_cross.tolist(),
        "residual_category_cross_entries": residual_cross.tolist(),
        "off_diagonal_genetic_kernel_frobenius_norms": (
            off_diagonal_kernel_norms.tolist()
        ),
        "maximum_normal_matrix_cross_entry": maximum_cross_entry,
        "full_system_not_assumed_block_diagonal": bool(maximum_cross_entry > 1.0e-12),
    }


def _boundary_results(fit: Any, preset: Any, draws: int, seed: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for index, hypothesis in enumerate(("rho=1", "tau2=0", "equal_variances")):
        result = test_binary_boundary(
            fit,
            hypothesis,
            preset=preset,
            draws=draws,
            seed=seed + index,
        )
        bootstrap = np.asarray(result.bootstrap_statistics, dtype=np.float64)
        finite = bootstrap[np.isfinite(bootstrap)]
        results[hypothesis] = {
            "hypothesis": result.hypothesis,
            "statistic": _json_safe(float(result.statistic)),
            "p_value": _json_safe(float(result.p_value)),
            "asymptotic_p_value": _json_safe(float(result.asymptotic_p_value)),
            "null_coefficients": _json_safe(result.null_coefficients),
            "covariance_rank": int(result.covariance_rank),
            "status": result.status,
            "method": result.method,
            "pseudo_value_covariance_error": _json_safe(
                float(result.pseudo_value_covariance_error)
            ),
            "seed": int(result.seed),
            "requested_draws": draws,
            "successful_draws": int(finite.size),
            "bootstrap_statistic_quantiles": (
                None
                if not finite.size
                else np.quantile(finite, [0.05, 0.5, 0.95]).tolist()
            ),
        }
    return results


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials < 1:
        return None
    proportion = successes / trials
    z = NORMAL_CRITICAL_VALUE
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [float(max(0.0, center - half_width)), float(min(1.0, center + half_width))]


def _summarize_boundary_null(
    replicate_results: Sequence[dict[str, Any]],
    *,
    draws: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    hypotheses: dict[str, Any] = {}
    for hypothesis in ("rho=1", "tau2=0", "equal_variances"):
        entries = [result[hypothesis] for result in replicate_results]
        p_values = np.asarray(
            [
                float(entry["p_value"])
                for entry in entries
                if entry["p_value"] is not None and np.isfinite(float(entry["p_value"]))
            ],
            dtype=np.float64,
        )
        rejections = int(np.sum(p_values < alpha))
        trials = int(p_values.size)
        status_counts: dict[str, int] = {}
        for entry in entries:
            status = str(entry["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        successful_draws = np.asarray(
            [entry["successful_draws"] for entry in entries], dtype=np.int64
        )
        hypotheses[hypothesis] = {
            "true_under_generating_model": True,
            "replicates_requested": len(entries),
            "replicates_with_finite_p_value": trials,
            "replicates_with_indeterminate_p_value": len(entries) - trials,
            "finite_p_value_fraction": float(trials / len(entries)),
            "status_counts": dict(sorted(status_counts.items())),
            "rejections_at_alpha": rejections,
            "alpha": alpha,
            "empirical_rejection_fraction": (
                None if trials == 0 else float(rejections / trials)
            ),
            "rejection_fraction_wilson_95_percent_interval": _wilson_interval(
                rejections, trials
            ),
            "rejection_fraction_monte_carlo_standard_error": (
                None
                if trials == 0
                else float(
                    np.sqrt(
                        (rejections / trials) * (1.0 - rejections / trials) / trials
                    )
                )
            ),
            "finite_p_value_quantiles": (
                None
                if trials == 0
                else np.quantile(p_values, [0.05, 0.5, 0.95]).tolist()
            ),
            "successful_multiplier_draws_minimum": int(np.min(successful_draws)),
            "successful_multiplier_draws_maximum": int(np.max(successful_draws)),
        }
    rejection_concern = any(
        summary["rejection_fraction_wilson_95_percent_interval"] is not None
        and summary["rejection_fraction_wilson_95_percent_interval"][0] > alpha
        for summary in hypotheses.values()
    )
    inference_failure_concern = any(
        summary["replicates_with_indeterminate_p_value"] > 0
        for summary in hypotheses.values()
    )
    if rejection_concern and inference_failure_concern:
        status = (
            "descriptive_experimental_inference_failures_and_null_rejection_"
            "elevated_not_production_calibrated"
        )
    elif inference_failure_concern:
        status = (
            "descriptive_experimental_inference_failures_" "not_production_calibrated"
        )
    elif rejection_concern:
        status = (
            "descriptive_experimental_null_rejection_elevated_"
            "not_production_calibrated"
        )
    else:
        status = "descriptive_narrow_experimental_not_production_calibrated"
    return {
        "status": status,
        "empirical_rejection_concern": rejection_concern,
        "inference_failure_concern": inference_failure_concern,
        "interpretation": (
            "All three nulls hold for Omega=sigma^2 11'. The Wilson intervals "
            "quantify Monte Carlo uncertainty in this small synthetic experiment; "
            + (
                "at least one interval excludes the nominal alpha in the elevated "
                "direction, so the result is a calibration warning. "
                if rejection_concern
                else "none excludes the nominal alpha in the elevated direction. "
            )
            + (
                "At least one hypothesis also has indeterminate replicate-level "
                "inference. "
                if inference_failure_concern
                else "All replicate-level p-values are finite. "
            )
            + "This is not a production calibration pass."
        ),
        "replicates": len(replicate_results),
        "multiplier_draws_per_hypothesis_and_replicate": draws,
        "hypotheses": hypotheses,
    }


def _categorical_permutation_equivalence(
    cohort: Cohort,
    draw: PhenotypeDraw,
    original_fit: Any,
) -> dict[str, Any]:
    original_labels = tuple(cohort.preset.category_labels)
    if len(original_labels) < 3:
        raise ValueError("Categorical permutation validation requires >=3 categories.")
    permuted_categories = tuple(
        cohort.mechanism.categories[index]
        for index in (2, 0, 1, *range(3, len(original_labels)))
    )
    context = np.argmax(cohort.preset.basis, axis=1)
    preset = build_categorical_context_preset(
        context,
        categories=permuted_categories,
        source_name=f"{cohort.mechanism.name}_context",
        binary=False,
    )
    components = preset.component_index
    common = {
        "genotype": cohort.genotype,
        "basis": preset.basis,
        "projector": cohort.projector,
        "annotations": cohort.annotations,
        "component_index": components,
        "loo_groups": cohort.loo_group_ids,
        "basis_hash": preset.basis_hash,
        "fixed_effect_hash": cohort.hashes["fixed_effect_hash"],
        "variant_hash": cohort.hashes["variant_hash"],
        "genotype_scaling": GENOTYPE_SCALING,
    }
    reference = build_context_reference(
        **common,
        gram_method="exact",
        same_person_method="exact",
        probe_tile_size=8,
    )
    summary = build_context_trait_summary(
        **common,
        phenotype=draw.raw,
        residual_basis=preset.residual_basis,
        residual_names=preset.residual_names,
        block_size=cohort.genotype.shape[1],
    )
    permuted_fit = _fit(reference, summary, preset)
    permuted_labels = tuple(preset.category_labels)
    positions = np.asarray(
        [permuted_labels.index(label) for label in original_labels], dtype=np.int64
    )

    def map_coefficients(values: np.ndarray) -> np.ndarray:
        genetic_count = len(components)
        omega = coefficients_to_omegas(values[:genetic_count], components)[0]
        mapped_omega = omega[np.ix_(positions, positions)]
        mapped_genetic = omegas_to_coefficients(
            mapped_omega[None, :, :], cohort.components
        )
        mapped_residual = np.asarray(values[genetic_count:])[positions]
        return np.concatenate([mapped_genetic, mapped_residual])

    mapped_full = map_coefficients(permuted_fit.raw_coefficients)
    mapped_loo = np.asarray(
        [map_coefficients(values) for values in permuted_fit.loo_coefficients],
        dtype=np.float64,
    )
    mapped_covariance = _jackknife_covariance(mapped_loo)
    original_covariance = combine_genetic_kernels(
        cohort.genetic_kernels, original_fit.genetic_coefficients
    ) + combine_genetic_kernels(
        cohort.residual_kernels, original_fit.residual_coefficients
    )
    permuted_features = common_scale_features(
        cohort.genotype, preset.basis, cohort.projector.projector
    )
    permuted_genetic_kernels = dense_genetic_kernels(
        permuted_features, cohort.annotations, components
    )
    permuted_residual_kernels = dense_residual_kernels(
        cohort.projector.projector, preset.residual_basis
    )
    permuted_covariance = combine_genetic_kernels(
        permuted_genetic_kernels, permuted_fit.genetic_coefficients
    ) + combine_genetic_kernels(
        permuted_residual_kernels, permuted_fit.residual_coefficients
    )
    original_surface = np.asarray(original_fit.raw_omegas[0], dtype=np.float64)
    permuted_surface = np.asarray(permuted_fit.raw_omegas[0], dtype=np.float64)[
        np.ix_(positions, positions)
    ]
    return {
        "original_category_labels": list(original_labels),
        "permuted_category_labels": list(permuted_labels),
        "original_category_order_hash": cohort.preset.category_order_hash,
        "permuted_category_order_hash": preset.category_order_hash,
        "coefficient_max_abs": _maximum_absolute(
            original_fit.raw_coefficients, mapped_full
        ),
        "loo_coefficient_max_abs": _maximum_absolute(
            original_fit.loo_coefficients, mapped_loo
        ),
        "jackknife_covariance_max_abs": _maximum_absolute(
            original_fit.jackknife_covariance, mapped_covariance
        ),
        "surface_max_abs": _maximum_absolute(original_surface, permuted_surface),
        "phenotype_covariance_max_abs": _maximum_absolute(
            original_covariance, permuted_covariance
        ),
        "public_original": _categorical_derived_summary(
            derive_categorical_context_fit(original_fit, cohort.preset)
        ),
        "public_permuted": _categorical_derived_summary(
            derive_categorical_context_fit(permuted_fit, preset)
        ),
    }


def _rare_category_rank_check(args: argparse.Namespace) -> dict[str, Any]:
    n = max(args.study_n, 48)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 9001]))
    context = np.concatenate(
        [
            np.zeros(n - 3, dtype=np.int64),
            np.ones(2, dtype=np.int64),
            np.full(1, 2, dtype=np.int64),
        ]
    )
    rng.shuffle(context)
    stage = "preset"
    try:
        preset = build_categorical_context_preset(
            context,
            categories=(0, 1, 2),
            source_name="rare_category_context",
            binary=False,
        )
        stage = "artifacts"
        genotype = rng.standard_normal((n, args.m))
        nuisance = _standardize(rng.standard_normal((n, 1)))
        fixed = preset.fixed_effect_design(nuisance)
        projector = rank_revealing_projector(fixed)
        annotations = np.ones((args.m, 1), dtype=np.float64)
        loo = tuple(f"block:{index % args.loo_groups}" for index in range(args.m))
        hashes = {
            "basis_hash": preset.basis_hash,
            "fixed_effect_hash": array_sha256(fixed),
            "variant_hash": canonical_sha256(
                {"ordered_synthetic_variants": list(range(args.m))}
            ),
        }
        common = {
            "genotype": genotype,
            "basis": preset.basis,
            "projector": projector,
            "annotations": annotations,
            "component_index": preset.component_index,
            "loo_groups": loo,
            "genotype_scaling": GENOTYPE_SCALING,
            **hashes,
        }
        reference = build_context_reference(
            **common, gram_method="exact", same_person_method="exact"
        )
        summary = build_context_trait_summary(
            **common,
            phenotype=rng.standard_normal(n),
            residual_basis=preset.residual_basis,
            residual_names=preset.residual_names,
            block_size=args.m,
        )
        stage = "fit"
        fit = _fit(reference, summary, preset)
    except (ContextRankError, ContextJackknifeError, ValueError) as exc:
        return {
            "category_counts": [n - 3, 2, 1],
            "failure_detected": True,
            "failure_stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "category_counts": [n - 3, 2, 1],
        "failure_detected": False,
        "unexpected_rank": int(fit.solve.rank),
        "unexpected_dimension": int(fit.raw_coefficients.size),
    }


def _difference_summary(
    differences: np.ndarray, names: Sequence[str]
) -> dict[str, Any]:
    array = np.asarray(differences, dtype=np.float64)
    return {
        str(name): {
            "bias": float(np.mean(array[:, index])),
            "rmse": float(np.sqrt(np.mean(array[:, index] ** 2))),
            "standard_deviation": (
                float(np.std(array[:, index], ddof=1)) if array.shape[0] > 1 else 0.0
            ),
        }
        for index, name in enumerate(names)
    }


def _run_mechanism(
    mechanism: Mechanism,
    *,
    mechanism_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    streams = np.random.SeedSequence([args.seed, mechanism_index]).spawn(
        args.replicates + 2
    )
    study, study_seconds, study_peak = _timed(
        lambda: _build_cohort(
            mechanism,
            rng=np.random.default_rng(streams[0]),
            n=args.study_n,
            m=args.m,
            loo_groups=args.loo_groups,
        )
    )
    independent, independent_cohort_seconds, independent_cohort_peak = _timed(
        lambda: _build_cohort(
            mechanism,
            rng=np.random.default_rng(streams[1]),
            n=args.reference_n,
            m=args.m,
            loo_groups=args.loo_groups,
        )
    )
    matched_reference, matched_reference_seconds, matched_reference_peak = _timed(
        lambda: _build_reference(study)
    )
    (
        independent_reference,
        independent_reference_seconds,
        independent_reference_peak,
    ) = _timed(lambda: _build_reference(independent))

    matched_coefficients: list[np.ndarray] = []
    independent_coefficients: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    matched_standard_errors: list[np.ndarray] = []
    independent_standard_errors: list[np.ndarray] = []
    matched_fits: list[Any] = []
    independent_fits: list[Any] = []
    binary_points: list[np.ndarray] = []
    binary_truths: list[np.ndarray] = []
    binary_standard_errors: list[np.ndarray] = []
    boundary_null_replicates: list[dict[str, Any]] = []
    minimum_generating_eigenvalues: list[float] = []
    normalization_multipliers: list[float] = []
    dense_matrix_error = 0.0
    dense_rhs_error = 0.0
    dense_coefficient_error = 0.0
    dense_surface_error = 0.0
    summary_seconds = 0.0
    matched_fit_seconds = 0.0
    independent_fit_seconds = 0.0
    dense_seconds = 0.0
    boundary_seconds = 0.0
    peak_rss = max(
        study_peak,
        independent_cohort_peak,
        matched_reference_peak,
        independent_reference_peak,
    )
    encoding_equivalence: dict[str, Any] | None = None
    permutation_equivalence: dict[str, Any] | None = None
    public_derived: dict[str, Any] | None = None
    public_psd_derived: dict[str, Any] | None = None
    boundary: dict[str, Any] | None = None
    block_structure: dict[str, Any] | None = None

    for replicate in range(args.replicates):
        draw = _sample_phenotype(study, np.random.default_rng(streams[replicate + 2]))
        summary, elapsed, observed_peak = _timed(
            lambda: _build_summary(study, draw.raw)
        )
        summary_seconds += elapsed
        peak_rss = max(peak_rss, observed_peak)
        matched, elapsed, observed_peak = _timed(
            lambda: _fit(matched_reference, summary, study.preset)
        )
        matched_fit_seconds += elapsed
        peak_rss = max(peak_rss, observed_peak)
        transferred, elapsed, observed_peak = _timed(
            lambda: _fit(independent_reference, summary, study.preset)
        )
        independent_fit_seconds += elapsed
        peak_rss = max(peak_rss, observed_peak)
        (dense_equations, dense_coefficients, _), elapsed, observed_peak = _timed(
            lambda: _dense_fit(study, draw)
        )
        dense_seconds += elapsed
        peak_rss = max(peak_rss, observed_peak)

        truth = _truth_coefficients(study, draw.normalization_multiplier)
        matched_coefficients.append(np.asarray(matched.raw_coefficients))
        independent_coefficients.append(np.asarray(transferred.raw_coefficients))
        truths.append(truth)
        matched_standard_errors.append(np.asarray(matched.standard_errors))
        independent_standard_errors.append(np.asarray(transferred.standard_errors))
        matched_fits.append(matched)
        independent_fits.append(transferred)
        minimum_generating_eigenvalues.append(draw.minimum_covariance_eigenvalue)
        normalization_multipliers.append(draw.normalization_multiplier)

        dense_matrix_error = max(
            dense_matrix_error,
            _maximum_absolute(matched.equations.matrix, dense_equations.matrix),
        )
        dense_rhs_error = max(
            dense_rhs_error,
            _maximum_absolute(matched.equations.rhs, dense_equations.rhs),
        )
        dense_coefficient_error = max(
            dense_coefficient_error,
            _maximum_absolute(matched.raw_coefficients, dense_coefficients),
        )
        dense_surface_error = max(
            dense_surface_error,
            _maximum_absolute(
                matched.raw_omegas,
                _surfaces(
                    dense_coefficients[: len(study.components)], study.components
                ),
            ),
        )

        if mechanism.binary:
            point, standard_error = _derived_point_se(matched)
            binary_points.append(point)
            binary_standard_errors.append(standard_error)
            binary_truths.append(_binary_quantities(truth[:3]))

        replicate_boundary: dict[str, Any] | None = None
        if mechanism.name == "no_context_dependence":
            replicate_boundary, elapsed, observed_peak = _timed(
                lambda: _boundary_results(
                    matched,
                    study.preset,
                    args.boundary_calibration_draws,
                    args.seed + 100_000 + 10 * replicate,
                )
            )
            boundary_null_replicates.append(replicate_boundary)
            boundary_seconds += elapsed
            peak_rss = max(peak_rss, observed_peak)

        if replicate == 0:
            block_structure = _block_structure_diagnostics(study, dense_equations)
            psd_fit = _fit(matched_reference, summary, study.preset, project_psd=True)
            if mechanism.binary:
                public_derived = _binary_derived_summary(
                    derive_binary_context_fit(matched, study.preset)
                )
                public_psd_derived = _psd_derived_summary(
                    psd_fit, study.preset, binary=True
                )
                if (
                    replicate_boundary is not None
                    and args.boundary_calibration_draws == args.boundary_draws
                ):
                    boundary = replicate_boundary
                else:
                    boundary, elapsed, observed_peak = _timed(
                        lambda: _boundary_results(
                            matched,
                            study.preset,
                            args.boundary_draws,
                            args.seed + 1000 * mechanism_index,
                        )
                    )
                    boundary_seconds += elapsed
                    peak_rss = max(peak_rss, observed_peak)
                encoding_equivalence = _encoding_equivalence(study, draw, matched)
            else:
                public_derived = _categorical_derived_summary(
                    derive_categorical_context_fit(matched, study.preset)
                )
                public_psd_derived = _psd_derived_summary(
                    psd_fit, study.preset, binary=False
                )
                permutation_equivalence = _categorical_permutation_equivalence(
                    study, draw, matched
                )

    matched_array = np.asarray(matched_coefficients, dtype=np.float64)
    independent_array = np.asarray(independent_coefficients, dtype=np.float64)
    truth_array = np.asarray(truths, dtype=np.float64)
    matched_se_array = np.asarray(matched_standard_errors, dtype=np.float64)
    independent_se_array = np.asarray(independent_standard_errors, dtype=np.float64)
    component_names = list(matched_fits[0].equations.component_names)
    calibration: dict[str, Any] = {
        "matched": _summarize_estimates(
            matched_array, truth_array, matched_se_array, component_names
        ),
        "independent_reference": _summarize_estimates(
            independent_array, truth_array, independent_se_array, component_names
        ),
    }
    derived_calibration = None
    if mechanism.binary:
        derived_calibration = _summarize_estimates(
            np.asarray(binary_points, dtype=np.float64),
            np.asarray(binary_truths, dtype=np.float64),
            np.asarray(binary_standard_errors, dtype=np.float64),
            BINARY_QUANTITY_NAMES,
        )
    independent_difference = independent_array - matched_array
    independent_genetic_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    independent_difference[:, : len(study.components)] ** 2,
                    axis=1,
                )
            )
        )
    )
    total_runtime = float(time.perf_counter() - total_started)
    return {
        "name": mechanism.name,
        "label": mechanism.label,
        "binary": mechanism.binary,
        "dimensions": {
            "study_n": args.study_n,
            "reference_n": args.reference_n,
            "m": args.m,
            "q": len(mechanism.categories),
            "p_genetic": len(study.components),
            "p_total": matched_array.shape[1],
        },
        "category_labels": list(study.preset.category_labels),
        "study_category_counts": list(study.preset.category_counts),
        "reference_category_counts": list(independent.preset.category_counts),
        "category_probabilities": list(mechanism.category_probabilities),
        "category_order_hash": study.preset.category_order_hash,
        "basis_hash": study.preset.basis_hash,
        "preset_manifest": _json_safe(study.preset.manifest),
        "generating": {
            "omega_before_phenotype_normalization": mechanism.omega.tolist(),
            "residual_variances_before_phenotype_normalization": (
                mechanism.residual_variances.tolist()
            ),
            "context_pc_correlation": mechanism.context_pc_correlation,
            "genotype_pc_loading": mechanism.genotype_pc_loading,
            "minimum_covariance_eigenvalue": float(
                np.min(minimum_generating_eigenvalues)
            ),
            "mean_phenotype_normalization_multiplier": float(
                np.mean(normalization_multipliers)
            ),
        },
        "dense_matched_equivalence": {
            "normal_matrix_max_abs": dense_matrix_error,
            "rhs_max_abs": dense_rhs_error,
            "coefficient_max_abs": dense_coefficient_error,
            "surface_max_abs": dense_surface_error,
        },
        "encoding_equivalence": encoding_equivalence,
        "category_permutation_equivalence": permutation_equivalence,
        "block_structure": block_structure,
        "coefficient_calibration": calibration,
        "binary_derived_calibration": derived_calibration,
        "independent_reference_substitution": {
            "coefficient_differences": _difference_summary(
                independent_difference, component_names
            ),
            "genetic_vector_rms": independent_genetic_rms,
        },
        "public_derived_first_replicate": public_derived,
        "public_psd_derived_first_replicate": public_psd_derived,
        "boundary_inference_first_replicate": boundary,
        "boundary_null_calibration": (
            _summarize_boundary_null(
                boundary_null_replicates,
                draws=args.boundary_calibration_draws,
            )
            if boundary_null_replicates
            else None
        ),
        "fit_diagnostics": {
            "matched": _fit_diagnostics(matched_fits),
            "independent_reference": _fit_diagnostics(independent_fits),
        },
        "runtime_seconds": {
            "study_cohort_and_dense_kernels": study_seconds,
            "independent_cohort_and_dense_kernels": independent_cohort_seconds,
            "matched_reference": matched_reference_seconds,
            "independent_reference": independent_reference_seconds,
            "trait_summaries_total": summary_seconds,
            "matched_fits_total": matched_fit_seconds,
            "independent_reference_fits_total": independent_fit_seconds,
            "dense_solves_total": dense_seconds,
            "boundary_inference_total": boundary_seconds,
            "total": total_runtime,
        },
        "absolute_peak_rss_bytes": max(peak_rss, _peak_rss_bytes()),
    }


def _equivalence_error(record: dict[str, Any]) -> float:
    values: list[float] = []
    encoding = record.get("encoding_equivalence")
    if isinstance(encoding, dict):
        values.extend(
            float(value)
            for key, value in encoding.items()
            if key.endswith("_max_abs") and value is not None
        )
    permutation = record.get("category_permutation_equivalence")
    if isinstance(permutation, dict):
        values.extend(
            float(value)
            for key, value in permutation.items()
            if key.endswith("_max_abs") and value is not None
        )
    return max(values, default=0.0)


def _plot(payload: dict[str, Any], output_dir: Path) -> None:
    records = payload["mechanisms"]
    binary_records = [record for record in records if record["binary"]]
    colors = ("#1f6f8b", "#d97706", "#4c956c")
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)

    quantity_names = ("v0", "v1", "gamma")
    all_truth: list[float] = []
    all_estimate: list[float] = []
    for quantity_index, (quantity, color) in enumerate(zip(quantity_names, colors)):
        truth = np.asarray(
            [
                record["binary_derived_calibration"][quantity]["mean_truth"]
                for record in binary_records
            ],
            dtype=np.float64,
        )
        estimate = np.asarray(
            [
                record["binary_derived_calibration"][quantity]["mean_estimate"]
                for record in binary_records
            ],
            dtype=np.float64,
        )
        all_truth.extend(truth.tolist())
        all_estimate.extend(estimate.tolist())
        axes[0, 0].scatter(
            truth,
            estimate,
            s=40,
            color=color,
            label=quantity,
            zorder=3 + quantity_index,
        )
    bounds = np.asarray([*all_truth, *all_estimate], dtype=np.float64)
    span = max(float(np.ptp(bounds)), 1.0e-6)
    lower = float(np.min(bounds) - 0.08 * span)
    upper = float(np.max(bounds) + 0.08 * span)
    axes[0, 0].plot([lower, upper], [lower, upper], "--", color="0.3")
    axes[0, 0].set_xlim(lower, upper)
    axes[0, 0].set_ylim(lower, upper)
    axes[0, 0].set_xlabel("Mean normalized generating coefficient")
    axes[0, 0].set_ylabel("Mean matched-summary estimate")
    axes[0, 0].legend(frameon=False, fontsize=8)

    positions = np.arange(len(binary_records), dtype=np.float64)
    for quantity, color, marker in zip(quantity_names, colors, ("o-", "s-", "^-")):
        coverage = [
            record["binary_derived_calibration"][quantity]["naive_95_percent_coverage"]
            for record in binary_records
        ]
        axes[0, 1].plot(
            positions, coverage, marker, color=color, label=quantity, linewidth=1.4
        )
    axes[0, 1].axhline(0.95, color="0.3", linestyle="--", linewidth=1.0)
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylabel("Naive 95% jackknife coverage")
    axes[0, 1].set_xticks(
        positions, [record["label"] for record in binary_records], rotation=30
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    boundary = next(
        record["boundary_null_calibration"]
        for record in records
        if record["name"] == "no_context_dependence"
    )
    hypothesis_names = ("rho=1", "tau2=0", "equal_variances")
    boundary_positions = np.arange(len(hypothesis_names), dtype=np.float64)
    rejection_rates = np.asarray(
        [
            boundary["hypotheses"][name]["empirical_rejection_fraction"]
            for name in hypothesis_names
        ],
        dtype=np.float64,
    )
    intervals = np.asarray(
        [
            boundary["hypotheses"][name][
                "rejection_fraction_wilson_95_percent_interval"
            ]
            for name in hypothesis_names
        ],
        dtype=np.float64,
    )
    axes[0, 2].errorbar(
        boundary_positions,
        rejection_rates,
        yerr=np.vstack(
            [rejection_rates - intervals[:, 0], intervals[:, 1] - rejection_rates]
        ),
        fmt="o",
        capsize=5,
        color="#b23a48",
    )
    axes[0, 2].axhline(0.05, color="0.3", linestyle="--", linewidth=1.0)
    axes[0, 2].set_ylim(0.0, max(0.55, float(np.max(intervals[:, 1]) * 1.08)))
    axes[0, 2].set_ylabel("True-null rejection fraction (95% Wilson interval)")
    axes[0, 2].set_xticks(
        boundary_positions,
        [
            f"{label}\n(n={boundary['hypotheses'][name]['replicates_with_finite_p_value']}"
            f"/{boundary['replicates']})"
            for name, label in zip(
                hypothesis_names, (r"$\rho=1$", r"$\tau^2=0$", r"$v_0=v_1$")
            )
        ],
    )
    axes[0, 2].set_title("Experimental boundary calibration", fontsize=10)

    all_positions = np.arange(len(records), dtype=np.float64)
    dense_errors = np.asarray(
        [
            record["dense_matched_equivalence"]["coefficient_max_abs"]
            for record in records
        ]
    )
    encoding_errors = np.asarray([_equivalence_error(record) for record in records])
    positive = np.concatenate(
        [dense_errors[dense_errors > 0.0], encoding_errors[encoding_errors > 0.0]]
    )
    floor = (
        float(np.min(positive)) * 0.5
        if positive.size
        else float(np.finfo(np.float64).eps)
    )
    axes[1, 0].plot(
        all_positions,
        np.maximum(dense_errors, floor),
        "o-",
        color="#1f6f8b",
        label="dense vs matched",
    )
    axes[1, 0].plot(
        all_positions,
        np.maximum(encoding_errors, floor),
        "s--",
        color="#d97706",
        label="encoding/permutation",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Maximum absolute discrepancy")
    axes[1, 0].set_xticks(
        all_positions, [record["label"] for record in records], rotation=30
    )
    axes[1, 0].legend(frameon=False, fontsize=8)

    independent_rms = np.asarray(
        [
            record["independent_reference_substitution"]["genetic_vector_rms"]
            for record in records
        ],
        dtype=np.float64,
    )
    axes[1, 1].bar(
        all_positions,
        independent_rms,
        width=0.65,
        color="#4c956c",
        alpha=0.85,
    )
    axes[1, 1].set_ylabel("Independent minus matched genetic-vector RMS")
    axes[1, 1].set_xticks(
        all_positions, [record["label"] for record in records], rotation=30
    )

    runtimes = np.asarray(
        [record["runtime_seconds"]["total"] for record in records],
        dtype=np.float64,
    )
    rss = np.asarray(
        [record["absolute_peak_rss_bytes"] / 2**20 for record in records],
        dtype=np.float64,
    )
    axes[1, 2].bar(
        all_positions,
        runtimes,
        width=0.65,
        color="#1f6f8b",
        alpha=0.85,
        label="runtime",
    )
    axes[1, 2].set_ylabel("Wall time per mechanism (s)")
    axes[1, 2].set_xticks(
        all_positions, [record["label"] for record in records], rotation=30
    )
    memory_axis = axes[1, 2].twinx()
    memory_axis.plot(
        all_positions,
        rss,
        "o--",
        color="#d97706",
        linewidth=1.3,
        label="peak RSS",
    )
    memory_axis.set_ylabel("Absolute process peak RSS (MiB)")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Binary and categorical contextual-covariance validation "
        f"(R={payload['configuration']['replicates']})"
    )
    figure.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300)
    figure.savefig(output_dir / f"{OUTPUT_STEM}.pdf")
    plt.close(figure)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    records = [
        _run_mechanism(mechanism, mechanism_index=index, args=args)
        for index, mechanism in enumerate(_mechanisms())
    ]
    rare_category = _rare_category_rank_check(args)
    maximum_dense_coefficient_error = max(
        record["dense_matched_equivalence"]["coefficient_max_abs"] for record in records
    )
    maximum_dense_surface_error = max(
        record["dense_matched_equivalence"]["surface_max_abs"] for record in records
    )
    maximum_encoding_error = max(_equivalence_error(record) for record in records)
    all_non_block_diagonal = all(
        record["block_structure"]["full_system_not_assumed_block_diagonal"]
        for record in records
    )
    no_context_record = next(
        record for record in records if record["name"] == "no_context_dependence"
    )
    three_category_record = next(
        record for record in records if record["name"] == "three_category_context"
    )
    aggregate = {
        "maximum_dense_matched_coefficient_error": maximum_dense_coefficient_error,
        "maximum_dense_matched_surface_error": maximum_dense_surface_error,
        "maximum_encoding_or_permutation_error": maximum_encoding_error,
        "all_full_systems_exercise_cross_structure": all_non_block_diagonal,
        "rare_category_rank_failure_detected": rare_category["failure_detected"],
        "maximum_independent_reference_genetic_vector_rms": max(
            record["independent_reference_substitution"]["genetic_vector_rms"]
            for record in records
        ),
        "maximum_condition_number": max(
            record["fit_diagnostics"]["matched"]["maximum_condition_number"]
            for record in records
        ),
        "minimum_jackknife_covariance_eigenvalue": min(
            record["fit_diagnostics"]["matched"][
                "minimum_jackknife_covariance_eigenvalue"
            ]
            for record in records
        ),
        "boundary_route_implemented": all(
            record["boundary_inference_first_replicate"] is not None
            for record in records
            if record["binary"]
        ),
        "boundary_null_calibration_status": no_context_record[
            "boundary_null_calibration"
        ]["status"],
        "boundary_null_empirical_rejection_concern": no_context_record[
            "boundary_null_calibration"
        ]["empirical_rejection_concern"],
        "boundary_null_inference_failure_concern": no_context_record[
            "boundary_null_calibration"
        ]["inference_failure_concern"],
        "production_boundary_calibration_claimed": False,
        "three_category_point_psd_status": three_category_record[
            "public_psd_derived_first_replicate"
        ]["status"],
        "three_category_psd_uncertainty_semantics": three_category_record[
            "public_psd_derived_first_replicate"
        ]["psd_uncertainty_semantics"],
    }
    aggregate["deterministic_equivalence_passed"] = bool(
        maximum_dense_coefficient_error <= 1.0e-9
        and maximum_dense_surface_error <= 1.0e-9
        and maximum_encoding_error <= 1.0e-8
        and all_non_block_diagonal
        and rare_category["failure_detected"]
    )
    payload = {
        "kind": "summit.context.binary_categorical_validation",
        "schema_version": 1,
        "contains_row_data": False,
        "seed": args.seed,
        "configuration": {
            "replicates": args.replicates,
            "study_n": args.study_n,
            "reference_n": args.reference_n,
            "m": args.m,
            "loo_groups": args.loo_groups,
            "boundary_multiplier_draws": args.boundary_draws,
            "boundary_null_calibration_draws": args.boundary_calibration_draws,
            "genotype_scaling": GENOTYPE_SCALING,
            "coverage_interpretation": (
                "small descriptive Monte Carlo; boundary Wald coverage is not a "
                "calibration claim"
            ),
            "boundary_calibration_interpretation": (
                "experimental route exercised under a true joint null with narrow "
                "Monte Carlo intervals; no production calibration pass is claimed"
            ),
        },
        "aggregate": aggregate,
        "rare_category_rank_check": rare_category,
        "runtime_seconds": float(time.perf_counter() - total_started),
        "absolute_peak_rss_bytes": _peak_rss_bytes(),
        "mechanisms": records,
    }
    output = _json_safe(payload)
    (args.output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(output, args.output_dir)


if __name__ == "__main__":
    main()
