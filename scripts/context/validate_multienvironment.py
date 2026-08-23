#!/usr/bin/env python3
"""Validate fixed multi-environment bases and covariance-operator modes.

The default path is a deterministic synthetic gate for Q=3 and Q=4 context
bases.  It exercises the public Stage-07A calibration, preset, pruning, mode,
and contrast APIs together with the existing contextual reference/summary
fitter.  An opt-in real-data path uses a small protected UKBB subset and emits
aggregate diagnostics only.  No sample or variant rows are written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summit.context import (
    FixedEffectInteractionSpec,
    MultiEnvironmentSourceSpec,
    apply_multienvironment_calibration,
    array_sha256,
    build_context_reference,
    build_context_trait_summary,
    calibrate_multienvironment_basis,
    canonical_sha256,
    coefficients_to_omegas,
    common_scale_features,
    dense_genetic_kernels,
    dense_residual_kernels,
    derive_context_contrast,
    derive_covariance_modes,
    fit_multienvironment_model,
    fit_multienvironment_pruning,
    kernel_rhs,
    omegas_to_coefficients,
    project_normalize_phenotype,
    project_genetic_coefficients_psd,
    scale_aware_max_discrepancy,
)


OUTPUT_STEM = "07a_multienvironment_validation"
GENOTYPE_SCALING = "synthetic_population_unit_variance"
DEFAULT_GENOTYPE_PREFIX = Path(
    "/home/bronsonj/UKBB/ldscores/refsample_h2_sensitivity_20260813/"
    "onekg_matched_unrelated_20260813/eur_matching/"
    "UKB_EUR_300k.seed20260813.n5000.common"
)
DEFAULT_PHENOTYPE_ROOT = Path("/home/bronsonj/UKBB/asha/phens")
PC_COLUMNS = tuple(f"f.22009.0.{index}" for index in range(1, 6))


@dataclass(frozen=True)
class SyntheticDefinition:
    name: str
    label: str
    q: int
    correlation: float
    context_pc_correlation: float
    omega: np.ndarray
    expected_operator_rank: int


@dataclass(frozen=True)
class SyntheticCohort:
    definition: SyntheticDefinition
    sources: dict[str, np.ndarray]
    covariates: dict[str, np.ndarray]
    genotype: np.ndarray
    phenotype: np.ndarray | None
    fixed_signal: np.ndarray


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-n", type=int, default=192)
    parser.add_argument("--reference-n", type=int, default=384)
    parser.add_argument("--m", type=int, default=144)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--eigengap-rtol", type=float, default=0.05)
    parser.add_argument("--benchmark-n", type=int, default=160)
    parser.add_argument("--benchmark-m", type=int, default=192)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument(
        "--full-psd-loo",
        action="store_true",
        help=(
            "Project every Q=4 deleted-group replicate as an expensive diagnostic; "
            "the default runs the full PSD LOO path for Q=3 and point PSD for Q=4."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--real-traits",
        action="store_true",
        help="Run an aggregate-only Q3/Q4 BMI/HDL/CRP sanity check.",
    )
    parser.add_argument("--geno-prefix", type=Path, default=DEFAULT_GENOTYPE_PREFIX)
    parser.add_argument("--phenotype-root", type=Path, default=DEFAULT_PHENOTYPE_ROOT)
    parser.add_argument(
        "--covariate-file",
        type=Path,
        default=DEFAULT_PHENOTYPE_ROOT / "testosterone.covar",
    )
    parser.add_argument("--real-n", type=int, default=256)
    parser.add_argument("--real-m", type=int, default=96)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.study_n < 96 or args.reference_n < 96:
        parser.error("study/reference sample sizes must be at least 96")
    if args.m < 48:
        parser.error("--m must be at least 48")
    if args.loo_groups < 6 or args.m % args.loo_groups:
        parser.error("--loo-groups must be at least 6 and divide --m")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if not 0.0 < args.eigengap_rtol < 1.0:
        parser.error("--eigengap-rtol must lie in (0,1)")
    if args.benchmark_n < 64 or args.benchmark_m < 48:
        parser.error("benchmark N/M are too small")
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.real_traits:
        if args.real_n < 96:
            parser.error("--real-n must be at least 96")
        if args.real_m < 48 or args.real_m % args.loo_groups:
            parser.error("--real-m must be at least 48 and divisible by --loo-groups")


def _output_paths(output_dir: Path, include_real: bool) -> tuple[Path, ...]:
    names = [f"{OUTPUT_STEM}.json"]
    for label in ("modes", "conditioning", "performance"):
        names.extend([f"{OUTPUT_STEM}_{label}.png", f"{OUTPUT_STEM}_{label}.pdf"])
    if include_real:
        names.extend(
            [f"{OUTPUT_STEM}_real_traits.png", f"{OUTPUT_STEM}_real_traits.pdf"]
        )
    return tuple(output_dir / name for name in names)


def _prepare_output(output_dir: Path, include_real: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    existing = [
        path.name for path in _output_paths(output_dir, include_real) if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing validation outputs: " + ", ".join(existing)
        )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _save_figure(figure: Any, output_dir: Path, label: str) -> None:
    for suffix in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}_{label}.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        path.chmod(0o600)
    plt.close(figure)


def _standardize(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("Encountered a degenerate numeric column.")
    return np.asarray(centered / scale, dtype=np.float64)


def _definitions() -> tuple[SyntheticDefinition, ...]:
    rank_one_loading = np.asarray([0.52, 0.24, -0.18], dtype=np.float64)
    rank_two_loadings = np.asarray(
        [[0.46, 0.12, -0.16, 0.20], [0.05, -0.30, 0.27, 0.15]],
        dtype=np.float64,
    )
    correlated_loading = np.asarray(
        [[0.40, 0.24, -0.12, 0.18], [0.08, -0.18, 0.31, -0.11]],
        dtype=np.float64,
    )
    return (
        SyntheticDefinition(
            name="q3_independent_rank_one",
            label="Q=3 independent, rank one",
            q=3,
            correlation=0.0,
            context_pc_correlation=0.0,
            omega=np.outer(rank_one_loading, rank_one_loading),
            expected_operator_rank=1,
        ),
        SyntheticDefinition(
            name="q4_correlated_rank_two",
            label="Q=4 correlated, rank two",
            q=4,
            correlation=0.65,
            context_pc_correlation=0.0,
            omega=rank_two_loadings.T @ rank_two_loadings,
            expected_operator_rank=2,
        ),
        SyntheticDefinition(
            name="q4_context_pc_rank_two",
            label="Q=4 context-PC correlation",
            q=4,
            correlation=0.35,
            context_pc_correlation=0.70,
            omega=correlated_loading.T @ correlated_loading,
            expected_operator_rank=2,
        ),
    )


def _source_specs(q: int) -> tuple[Any, ...]:
    return tuple(
        MultiEnvironmentSourceSpec(f"environment_{index}", "continuous")
        for index in range(1, q)
    )


def _interaction_specs(q: int) -> tuple[Any, ...]:
    result = []
    if q >= 2:
        result.append(
            FixedEffectInteractionSpec("environment_1", "pc1", "environment_1_by_pc1")
        )
    if q >= 4:
        result.append(
            FixedEffectInteractionSpec("environment_3", "pc2", "environment_3_by_pc2")
        )
    return tuple(result)


def _sample_contexts(
    rng: np.random.Generator,
    *,
    n: int,
    q: int,
    correlation: float,
    context_pc_correlation: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    l_count = q - 1
    covariance = np.full((l_count, l_count), correlation, dtype=np.float64)
    np.fill_diagonal(covariance, 1.0)
    raw = rng.multivariate_normal(np.zeros(l_count), covariance, size=n)
    pc_noise = rng.standard_normal((n, 2))
    pc1 = (
        context_pc_correlation * _standardize(raw[:, [0]])[:, 0]
        + math.sqrt(max(1.0 - context_pc_correlation**2, 0.0)) * pc_noise[:, 0]
    )
    pc2 = pc_noise[:, 1]
    sources = {
        f"environment_{index + 1}": np.asarray(raw[:, index], dtype=np.float64)
        for index in range(l_count)
    }
    covariates = {
        "pc1": np.asarray(pc1, dtype=np.float64),
        "pc2": np.asarray(pc2, dtype=np.float64),
        "age_covariate": np.asarray(rng.standard_normal(n), dtype=np.float64),
    }
    return sources, covariates


def _sample_genotype(
    rng: np.random.Generator,
    covariates: Mapping[str, np.ndarray],
    *,
    n: int,
    m: int,
) -> np.ndarray:
    phase = 2.0 * np.pi * (np.arange(m) + 0.5) / m
    pc1_loading = 0.22 * np.sin(phase)
    pc2_loading = 0.16 * np.cos(2.0 * phase)
    genotype = (
        rng.standard_normal((n, m))
        + np.asarray(covariates["pc1"])[:, None] * pc1_loading[None, :]
        + np.asarray(covariates["pc2"])[:, None] * pc2_loading[None, :]
    )
    return _standardize(genotype)


def _make_cohort(
    definition: SyntheticDefinition,
    *,
    n: int,
    m: int,
    seed: np.random.SeedSequence,
) -> SyntheticCohort:
    rng = np.random.default_rng(seed)
    sources, covariates = _sample_contexts(
        rng,
        n=n,
        q=definition.q,
        correlation=definition.correlation,
        context_pc_correlation=definition.context_pc_correlation,
    )
    genotype = _sample_genotype(rng, covariates, n=n, m=m)
    fixed_signal = (
        0.14 * np.asarray(covariates["pc1"])
        - 0.08 * np.asarray(covariates["pc2"])
        + 0.05 * np.asarray(covariates["age_covariate"])
    )
    return SyntheticCohort(
        definition=definition,
        sources=sources,
        covariates=covariates,
        genotype=genotype,
        phenotype=None,
        fixed_signal=np.asarray(fixed_signal),
    )


def _sample_phenotype(
    cohort: SyntheticCohort,
    preset: Any,
    *,
    seed: np.random.SeedSequence,
) -> SyntheticCohort:
    rng = np.random.default_rng(seed)
    components = preset.component_index
    features = common_scale_features(
        cohort.genotype, preset.basis, preset.projector.projector
    )
    annotations = np.ones((cohort.genotype.shape[1], 1), dtype=np.float64)
    genetic_kernels = dense_genetic_kernels(features, annotations, components)
    coefficients = omegas_to_coefficients(
        cohort.definition.omega[None, :, :], components
    )
    residual_kernels = dense_residual_kernels(
        preset.projector.projector, preset.residual_basis
    )
    covariance = (
        np.einsum("a,aij->ij", coefficients, genetic_kernels, optimize=True)
        + 0.48 * residual_kernels[0]
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-10 * scale:
        raise RuntimeError("Synthetic phenotype covariance is indefinite.")
    stochastic = eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0.0)) * rng.standard_normal(eigenvalues.size)
    )
    phenotype = stochastic + cohort.fixed_signal
    return SyntheticCohort(
        definition=cohort.definition,
        sources=cohort.sources,
        covariates=cohort.covariates,
        genotype=cohort.genotype,
        phenotype=np.asarray(phenotype),
        fixed_signal=cohort.fixed_signal,
    )


def _metric_modes(
    omega: object, metric: object, context_grid: object
) -> dict[str, Any]:
    covariance = np.asarray(omega, dtype=np.float64)
    basis_metric = np.asarray(metric, dtype=np.float64)
    grid = np.asarray(context_grid, dtype=np.float64)
    metric_values, metric_vectors = np.linalg.eigh(
        0.5 * (basis_metric + basis_metric.T)
    )
    tolerance = 1.0e-12 * max(float(np.max(np.abs(metric_values), initial=0.0)), 1.0)
    if metric_values[0] <= tolerance:
        raise ValueError("Independent mode oracle requires a positive-definite metric.")
    metric_sqrt = (metric_vectors * np.sqrt(metric_values)) @ metric_vectors.T
    metric_inverse_sqrt = (
        metric_vectors * (1.0 / np.sqrt(metric_values))
    ) @ metric_vectors.T
    operator = metric_sqrt @ covariance @ metric_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (operator + operator.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    whitened = eigenvectors[:, order]
    coefficients = metric_inverse_sqrt @ whitened
    functions = grid @ coefficients
    for index in range(coefficients.shape[1]):
        pivot = int(np.argmax(np.abs(functions[:, index])))
        if functions[pivot, index] < 0.0:
            coefficients[:, index] *= -1.0
            whitened[:, index] *= -1.0
            functions[:, index] *= -1.0
    residuals = np.asarray(
        [
            np.linalg.norm(
                covariance @ basis_metric @ coefficients[:, index]
                - eigenvalues[index] * coefficients[:, index]
            )
            / max(
                1.0,
                np.linalg.norm(covariance @ basis_metric @ coefficients[:, index]),
                abs(float(eigenvalues[index])) * np.linalg.norm(coefficients[:, index]),
            )
            for index in range(coefficients.shape[1])
        ]
    )
    reconstructed = (functions * eigenvalues[None, :]) @ functions.T
    surface = grid @ covariance @ grid.T
    positive = np.maximum(eigenvalues, 0.0)
    rank_one_fraction = (
        float(positive[0] / np.sum(positive))
        if np.sum(positive) > 0.0 and eigenvalues[-1] >= -1.0e-10
        else float("nan")
    )
    return {
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "functions": functions,
        "equation_relative_residuals": residuals,
        "surface_reconstruction_discrepancy": scale_aware_max_discrepancy(
            reconstructed, surface
        ),
        "metric_orthonormality_discrepancy": scale_aware_max_discrepancy(
            coefficients.T @ basis_metric @ coefficients,
            np.eye(coefficients.shape[1]),
        ),
        "rank_one_fraction": rank_one_fraction,
    }


def _public_value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise AttributeError(f"Public result lacks any of the fields {names}.")


def _public_mode_payload(value: Any, preset: Any, omega: object) -> dict[str, Any]:
    eigenvalues = np.asarray(_public_value(value, "eigenvalues"), dtype=np.float64)
    coefficients = np.asarray(_public_value(value, "eigenfunctions"), dtype=np.float64)
    metric = np.asarray(_public_value(value, "metric"), dtype=np.float64)
    grid = np.asarray(preset.context_grid, dtype=np.float64)
    covariance = np.asarray(omega, dtype=np.float64)
    reconstructed_functions = grid @ coefficients
    public_functions = np.asarray(
        _public_value(value, "function_values"), dtype=np.float64
    )
    reconstructed_equation_residuals = np.asarray(
        [
            np.linalg.norm(
                covariance @ metric @ coefficients[:, index]
                - eigenvalues[index] * coefficients[:, index]
            )
            / max(
                1.0,
                np.linalg.norm(covariance @ metric @ coefficients[:, index]),
                abs(float(eigenvalues[index])) * np.linalg.norm(coefficients[:, index]),
            )
            for index in range(coefficients.shape[1])
        ],
        dtype=np.float64,
    )
    public_equation_residuals = np.asarray(
        _public_value(value, "equation_residuals"), dtype=np.float64
    )
    unstable_flags = tuple(
        bool(flag) for flag in _public_value(value, "unstable_modes")
    )
    return {
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "functions": public_functions,
        "rank_one_fraction": float(_public_value(value, "rank_one_fraction")),
        "equation_relative_residuals": public_equation_residuals,
        "scale_aware_equation_residuals": reconstructed_equation_residuals,
        "function_field_discrepancy": float(
            scale_aware_max_discrepancy(public_functions, reconstructed_functions)
        ),
        "equation_residual_field_discrepancy": float(
            scale_aware_max_discrepancy(
                public_equation_residuals,
                reconstructed_equation_residuals,
            )
        ),
        "metric_orthonormality_discrepancy": float(
            scale_aware_max_discrepancy(
                coefficients.T @ metric @ coefficients,
                np.eye(coefficients.shape[1]),
            )
        ),
        "loo_eigenvalues": np.asarray(_public_value(value, "loo_eigenvalues")),
        "eigenvalue_standard_errors": np.asarray(
            _public_value(value, "eigenvalue_standard_errors")
        ),
        "stability": {
            "status": str(_public_value(value, "status")),
            "interpretation": str(_public_value(value, "interpretation")),
            "unstable_flags": list(unstable_flags),
            "unstable_mode_indices": [
                int(index) for index, flag in enumerate(unstable_flags) if flag
            ],
            "mode_clusters": _json_safe(_public_value(value, "mode_clusters")),
            "maximum_principal_angles": _json_safe(
                _public_value(value, "eigenspace_max_principal_angles")
            ),
        },
    }


def _contrast_payload(value: Any) -> dict[str, Any]:
    return {
        "estimate": float(_public_value(value, "estimate")),
        "standard_error": float(_public_value(value, "standard_error")),
        "loo_values": np.asarray(_public_value(value, "loo_values")),
        "loo_groups": [str(group) for group in _public_value(value, "loo_groups")],
        "status": str(getattr(value, "status", "defined_joint_loo_contrast")),
    }


def _aggregate_contrast_payload(value: Any) -> dict[str, Any]:
    payload = _contrast_payload(value)
    return {
        "estimate": payload["estimate"],
        "standard_error": payload["standard_error"],
        "status": payload["status"],
        "loo_replicate_count": int(payload["loo_values"].size),
    }


def _calibrate_reference(cohort: SyntheticCohort) -> Any:
    return calibrate_multienvironment_basis(
        cohort.sources,
        _source_specs(cohort.definition.q),
        mask=np.ones(cohort.genotype.shape[0], dtype=bool),
        basis_id=f"synthetic_{cohort.definition.name}",
    )


def _apply_calibration(calibration: Any, cohort: SyntheticCohort) -> Any:
    return apply_multienvironment_calibration(
        calibration,
        cohort.sources,
        mask=np.ones(cohort.genotype.shape[0], dtype=bool),
        covariates=cohort.covariates,
        interactions=_interaction_specs(cohort.definition.q),
        genotype_for_diagnostics=cohort.genotype,
    )


def _basis_hash(preset: Any) -> str:
    for candidate in (
        getattr(preset, "basis_hash", None),
        getattr(preset, "manifest", {}).get("basis_hash"),
        getattr(preset, "manifest", {}).get("calibration_hash"),
    ):
        if isinstance(candidate, str) and len(candidate) == 64:
            return candidate
    raise RuntimeError("Multi-environment preset does not expose a basis hash.")


def _fixed_effect_array_hash(preset: Any) -> str:
    for candidate in (
        getattr(preset, "fixed_effect_hash", None),
        getattr(preset, "manifest", {}).get("fixed_effect_hash"),
    ):
        if isinstance(candidate, str) and len(candidate) == 64:
            return candidate
    raise RuntimeError(
        "Multi-environment preset does not expose a fixed-effect array hash."
    )


def _attach_optional_psd(fit: Any) -> tuple[Any, dict[str, Any]]:
    p_genetic = len(fit.component_index)
    started = time.perf_counter()
    try:
        projection = project_genetic_coefficients_psd(
            fit.genetic_coefficients,
            fit.jackknife_covariance[:p_genetic, :p_genetic],
            fit.component_index,
            annotations_disjoint=True,
        )
    except (RuntimeError, ValueError) as exc:
        return fit, {
            "status": "indeterminate_optional_projection",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
    manifest = dict(fit.manifest)
    manifest["psd_projection"] = {
        "requested": True,
        "annotations_disjoint": True,
        "status": "defined_separate_from_raw_fit",
    }
    projected_fit = replace(fit, manifest=manifest, psd_projection=projection)
    return projected_fit, {
        "status": "defined_separate_from_raw_fit",
        "distance": float(projection.distance),
        "euclidean_distance": float(projection.euclidean_distance),
        "covariance_rank": int(projection.covariance_rank),
        "covariance_nullity": int(projection.covariance_nullity),
        "minimum_eigenvalues": projection.minimum_eigenvalues.tolist(),
        "optimizer_success": bool(projection.optimizer_success),
        "optimizer_message": str(projection.optimizer_message),
        "tie_break_applied": bool(projection.tie_break_applied),
        "cleanup_norm": float(projection.cleanup_norm),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _build_reference_and_fit(
    reference_cohort: SyntheticCohort,
    study_cohort: SyntheticCohort,
    reference_preset: Any,
    study_preset: Any,
    *,
    args: argparse.Namespace,
    identity: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    m = reference_cohort.genotype.shape[1]
    if study_cohort.genotype.shape[1] != m:
        raise RuntimeError("Synthetic reference/study variant counts differ.")
    annotations = np.ones((m, 1), dtype=np.float64)
    groups = tuple(f"block:{index % args.loo_groups}" for index in range(m))
    variant_hash = canonical_sha256(
        {"synthetic_multienvironment_variants": m, "identity": identity}
    )
    reference = build_context_reference(
        genotype=reference_cohort.genotype,
        basis=reference_preset.basis,
        projector=reference_preset.projector,
        annotations=annotations,
        component_index=reference_preset.component_index,
        basis_hash=_basis_hash(reference_preset),
        fixed_effect_hash=_fixed_effect_array_hash(reference_preset),
        variant_hash=variant_hash,
        loo_groups=groups,
        genotype_scaling=GENOTYPE_SCALING,
        gram_method="exact",
        same_person_method="exact",
    )
    if study_cohort.phenotype is None:
        raise RuntimeError("Study phenotype has not been generated.")
    summary = build_context_trait_summary(
        genotype=study_cohort.genotype,
        basis=study_preset.basis,
        phenotype=study_cohort.phenotype,
        projector=study_preset.projector,
        annotations=annotations,
        component_index=study_preset.component_index,
        residual_basis=study_preset.residual_basis,
        residual_names=study_preset.residual_names,
        basis_hash=_basis_hash(study_preset),
        fixed_effect_hash=_fixed_effect_array_hash(study_preset),
        variant_hash=variant_hash,
        loo_groups=groups,
        genotype_scaling=GENOTYPE_SCALING,
        block_size=args.block_size,
    )
    fit = fit_multienvironment_model(
        reference,
        summary,
        reference_preset=reference_preset,
        study_preset=study_preset,
        project_psd=False,
    )
    fit, psd_diagnostics = _attach_optional_psd(fit)
    return reference, summary, fit, psd_diagnostics


def _context_pair(preset: Any) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(preset.context_grid, dtype=np.float64)
    if grid.shape[0] < 2:
        raise RuntimeError("Preset context grid must contain at least two points.")
    distances = np.linalg.norm(grid[:, None, :] - grid[None, :, :], axis=2)
    left, right = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return np.asarray(grid[left]), np.asarray(grid[right])


def _contrast_oracle(
    omega: np.ndarray, left: np.ndarray, right: np.ndarray, kind: str
) -> float:
    if kind == "variance_difference":
        return float(left @ omega @ left - right @ omega @ right)
    if kind == "covariance":
        return float(left @ omega @ right)
    raise ValueError(f"Unsupported contrast kind {kind!r}.")


def _jackknife_standard_error(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or finite.size < 2 or np.any(~np.isfinite(finite)):
        return float("nan")
    centered = finite - np.mean(finite)
    return float(
        np.sqrt((finite.size - 1.0) / finite.size * np.sum(centered * centered))
    )


def _surface_quantity_values(
    omegas: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, np.ndarray]:
    variance_left = np.einsum("q,jqr,r->j", left, omegas, left, optimize=True)
    variance_right = np.einsum("q,jqr,r->j", right, omegas, right, optimize=True)
    covariance = np.einsum("q,jqr,r->j", left, omegas, right, optimize=True)
    positive_both = (variance_left > 0.0) & (variance_right > 0.0)
    log_ratio = np.full(omegas.shape[0], np.nan, dtype=np.float64)
    correlation = np.full(omegas.shape[0], np.nan, dtype=np.float64)
    log_ratio[positive_both] = np.log(
        variance_left[positive_both] / variance_right[positive_both]
    )
    correlation[positive_both] = covariance[positive_both] / np.sqrt(
        variance_left[positive_both] * variance_right[positive_both]
    )
    conditional = np.full(omegas.shape[0], np.nan, dtype=np.float64)
    positive_left = variance_left > 0.0
    conditional[positive_left] = variance_right[positive_left] - (
        covariance[positive_left] ** 2 / variance_left[positive_left]
    )
    return {
        "variance_difference": variance_left - variance_right,
        "log_variance_ratio": log_ratio,
        "covariance": covariance,
        "correlation": correlation,
        "conditional_variance_right_given_left": conditional,
    }


def _pairwise_surface_payload(
    fit: Any,
    left: np.ndarray,
    right: np.ndarray,
    *,
    include_loo_values: bool,
) -> dict[str, Any]:
    p_genetic = len(fit.component_index)
    point_omega = np.asarray(fit.raw_omegas[0], dtype=np.float64)
    loo_omegas = np.stack(
        [
            coefficients_to_omegas(
                np.asarray(row[:p_genetic], dtype=np.float64), fit.component_index
            )[0]
            for row in np.asarray(fit.loo_coefficients)
        ]
    )
    point = _surface_quantity_values(point_omega[None, :, :], left, right)
    loo = _surface_quantity_values(loo_omegas, left, right)
    result: dict[str, Any] = {}
    for name, point_values in point.items():
        estimate = float(point_values[0])
        replicate_values = np.asarray(loo[name], dtype=np.float64)
        finite_count = int(np.sum(np.isfinite(replicate_values)))
        if not np.isfinite(estimate):
            status = "indeterminate_nonpositive_variance"
        elif name == "correlation" and abs(estimate) >= 1.0 - 1.0e-8:
            status = "descriptive_boundary_no_naive_wald"
        elif name == "conditional_variance_right_given_left" and estimate <= 1.0e-10:
            status = "descriptive_boundary_or_non_psd_no_naive_wald"
        elif finite_count != replicate_values.size:
            status = "indeterminate_loo_domain_failure"
        else:
            status = "defined_equal_group_jackknife"
        entry = {
            "estimate": estimate,
            "standard_error": _jackknife_standard_error(replicate_values),
            "status": status,
            "finite_loo_replicates": finite_count,
            "total_loo_replicates": int(replicate_values.size),
        }
        if include_loo_values:
            entry["loo_values"] = replicate_values
        result[name] = entry
    return result


def _conditioning_payload(preset: Any, fit: Any) -> dict[str, Any]:
    conditioning = getattr(preset, "conditioning", {})
    if is_dataclass(conditioning):
        conditioning = asdict(conditioning)
    metric = np.asarray(preset.basis_metric, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    tolerance = 1.0e-12 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    nonconstant = np.asarray(preset.basis[:, 1:], dtype=np.float64)
    correlations = (
        np.corrcoef(nonconstant, rowvar=False)
        if nonconstant.shape[1] > 1
        else np.ones((nonconstant.shape[1], nonconstant.shape[1]))
    )
    off_diagonal = correlations - np.eye(correlations.shape[0])
    return {
        "public": conditioning,
        "metric_eigenvalues": eigenvalues.tolist(),
        "metric_rank": int(np.sum(eigenvalues > tolerance)),
        "metric_condition_number": float(
            np.max(eigenvalues) / np.min(eigenvalues[eigenvalues > tolerance])
        ),
        "maximum_absolute_context_correlation": float(
            np.max(np.abs(off_diagonal), initial=0.0)
        ),
        "fixed_effect_rank": int(preset.projector.rank),
        "fixed_effect_columns": int(preset.fixed_effect_design.shape[1]),
        "maximum_leverage": float(preset.projector.maximum_leverage),
        "normal_equation_rank": int(fit.solve.rank),
        "normal_equation_dimension": int(fit.raw_coefficients.size),
        "normal_condition_number": float(fit.solve.condition_number),
        "normal_relative_residual": float(fit.solve.relative_residual),
    }


def _basis_invariance(
    omega: np.ndarray, metric: np.ndarray, grid: np.ndarray, *, seed: int
) -> dict[str, Any]:
    q = omega.shape[0]
    rng = np.random.default_rng(seed)
    transform = np.eye(q) + 0.18 * rng.standard_normal((q, q))
    while np.linalg.cond(transform) > 5.0:
        transform = np.eye(q) + 0.18 * rng.standard_normal((q, q))
    inverse = np.linalg.inv(transform)
    transformed_omega = inverse.T @ omega @ inverse
    transformed_metric = transform @ metric @ transform.T
    transformed_grid = grid @ transform.T
    surface = grid @ omega @ grid.T
    transformed_surface = transformed_grid @ transformed_omega @ transformed_grid.T
    modes = _metric_modes(omega, metric, grid)
    transformed_modes = _metric_modes(
        transformed_omega, transformed_metric, transformed_grid
    )
    expected_coefficients = inverse.T @ modes["coefficients"]
    observed_coefficients = transformed_modes["coefficients"]
    eigenvalues = np.asarray(modes["eigenvalues"])
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    groups: list[tuple[int, ...]] = []
    start = 0
    for index in range(1, q):
        if abs(float(eigenvalues[index - 1] - eigenvalues[index])) > 1.0e-9 * scale:
            groups.append(tuple(range(start, index)))
            start = index
    groups.append(tuple(range(start, q)))
    coefficient_errors: list[float] = []
    function_errors: list[float] = []
    subspace_errors: list[float] = []
    function_subspace_errors: list[float] = []
    for group in groups:
        indices = np.asarray(group, dtype=np.int64)
        expected = expected_coefficients[:, indices]
        observed = observed_coefficients[:, indices].copy()
        if indices.size == 1:
            if (expected.T @ transformed_metric @ observed).item() < 0.0:
                observed *= -1.0
            coefficient_errors.append(scale_aware_max_discrepancy(expected, observed))
            function_errors.append(
                scale_aware_max_discrepancy(
                    transformed_grid @ expected, transformed_grid @ observed
                )
            )
            continue
        singular_values = np.linalg.svd(
            expected.T @ transformed_metric @ observed, compute_uv=False
        )
        subspace_errors.append(
            float(np.max(np.abs(singular_values - 1.0), initial=0.0))
        )
        expected_functions = transformed_grid @ expected
        observed_functions = transformed_grid @ observed
        function_subspace_errors.append(
            scale_aware_max_discrepancy(
                expected_functions @ expected_functions.T,
                observed_functions @ observed_functions.T,
            )
        )
    return {
        "transform": transform,
        "eigenvalue_groups": [list(group) for group in groups],
        "tied_groups": [list(group) for group in groups if len(group) > 1],
        "surface_discrepancy": float(
            scale_aware_max_discrepancy(surface, transformed_surface)
        ),
        "operator_eigenvalue_discrepancy": float(
            scale_aware_max_discrepancy(
                modes["eigenvalues"], transformed_modes["eigenvalues"]
            )
        ),
        "mode_coefficient_discrepancy": float(max(coefficient_errors, default=0.0)),
        "mode_function_discrepancy": float(max(function_errors, default=0.0)),
        "tied_mode_subspace_discrepancy": float(max(subspace_errors, default=0.0)),
        "tied_mode_function_projector_discrepancy": float(
            max(function_subspace_errors, default=0.0)
        ),
    }


def _run_synthetic_case(
    definition: SyntheticDefinition,
    *,
    case_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    streams = np.random.SeedSequence([args.seed, case_index]).spawn(3)
    reference_cohort = _make_cohort(
        definition,
        n=args.reference_n,
        m=args.m,
        seed=streams[0],
    )
    study_cohort = _make_cohort(
        definition,
        n=args.study_n,
        m=args.m,
        seed=streams[1],
    )
    calibration = _calibrate_reference(reference_cohort)
    reference_preset = _apply_calibration(calibration, reference_cohort)
    study_preset = _apply_calibration(calibration, study_cohort)
    study_cohort = _sample_phenotype(study_cohort, study_preset, seed=streams[2])
    started = time.perf_counter()
    reference, summary, fit, psd_diagnostics = _build_reference_and_fit(
        reference_cohort,
        study_cohort,
        reference_preset,
        study_preset,
        args=args,
        identity=definition.name,
    )
    public_modes_raw = derive_covariance_modes(
        fit,
        study_preset.without_individual_data(),
        annotation="all",
        eigengap_rtol=args.eigengap_rtol,
        use_psd=False,
    )
    raw_payload = _public_mode_payload(
        public_modes_raw, study_preset, fit.raw_omegas[0]
    )
    oracle_raw = _metric_modes(
        fit.raw_omegas[0], reference_preset.basis_metric, study_preset.context_grid
    )
    psd_payload: dict[str, Any] | None = None
    oracle_psd: dict[str, Any] | None = None
    psd_loo_exercised = False
    if fit.psd_projection is not None:
        psd_mode_fit = fit
        if definition.q == 4 and not args.full_psd_loo:
            psd_mode_fit = replace(
                fit,
                jackknife_groups=(),
                loo_coefficients=np.empty(
                    (0, fit.raw_coefficients.size), dtype=np.float64
                ),
            )
        else:
            psd_loo_exercised = True
        public_modes_psd = derive_covariance_modes(
            psd_mode_fit,
            study_preset.without_individual_data(),
            annotation="all",
            eigengap_rtol=args.eigengap_rtol,
            use_psd=True,
        )
        psd_payload = _public_mode_payload(
            public_modes_psd,
            study_preset,
            fit.psd_projection.projected_omegas[0],
        )
        oracle_psd = _metric_modes(
            fit.psd_projection.projected_omegas[0],
            reference_preset.basis_metric,
            study_preset.context_grid,
        )
    left, right = _context_pair(study_preset)
    pairwise_surface = _pairwise_surface_payload(
        fit, left, right, include_loo_values=True
    )
    contrast_results: dict[str, Any] = {}
    for kind in ("variance_difference", "covariance"):
        observed = derive_context_contrast(
            fit,
            study_preset.without_individual_data(),
            left,
            right,
            annotation="all",
            kind=kind,
        )
        payload = _contrast_payload(observed)
        payload["oracle_raw_estimate"] = _contrast_oracle(
            fit.raw_omegas[0], left, right, kind
        )
        payload["estimate_discrepancy"] = abs(
            payload["estimate"] - payload["oracle_raw_estimate"]
        )
        payload["manual_loo_discrepancy"] = float(
            scale_aware_max_discrepancy(
                payload["loo_values"], pairwise_surface[kind]["loo_values"]
            )
        )
        payload["manual_standard_error_discrepancy"] = abs(
            payload["standard_error"] - pairwise_surface[kind]["standard_error"]
        )
        contrast_results[kind] = payload
    invariance = _basis_invariance(
        definition.omega,
        reference_preset.basis_metric,
        study_preset.context_grid,
        seed=args.seed + 100 + case_index,
    )
    normalized_phenotype = project_normalize_phenotype(
        study_cohort.phenotype, study_preset.projector
    )
    study_features = common_scale_features(
        study_cohort.genotype,
        study_preset.basis,
        study_preset.projector.projector,
    )
    fixed_effect_design = np.asarray(study_preset.fixed_effect_design, dtype=np.float64)
    left_vectors, _, _ = np.linalg.svd(fixed_effect_design, full_matrices=False)
    fixed_rank = int(study_preset.projector.rank)
    explicit_projector = np.eye(args.study_n) - (
        left_vectors[:, :fixed_rank] @ left_vectors[:, :fixed_rank].T
    )
    explicit_features = np.stack(
        [
            explicit_projector
            @ (study_preset.basis[:, index, None] * study_cohort.genotype)
            for index in range(definition.q)
        ]
    )
    projected_genotype = explicit_projector @ study_cohort.genotype
    wrong_order_features = np.stack(
        [
            study_preset.basis[:, index, None] * projected_genotype
            for index in range(definition.q)
        ]
    )
    fixed_effect_oracle = {
        "projector_discrepancy": float(
            scale_aware_max_discrepancy(
                study_preset.projector.projector, explicit_projector
            )
        ),
        "feature_discrepancy": float(
            scale_aware_max_discrepancy(study_features, explicit_features)
        ),
        "projected_fixed_effect_residual": float(
            np.max(np.abs(explicit_projector @ fixed_effect_design), initial=0.0)
        ),
        "wrong_order_separation": float(
            scale_aware_max_discrepancy(study_features, wrong_order_features)
        ),
        "declared_interactions": list(_interaction_specs(definition.q)),
    }
    dense_kernels = dense_genetic_kernels(
        study_features,
        np.ones((args.m, 1), dtype=np.float64),
        study_preset.component_index,
    )
    all_kernel_errors = []
    off_diagonal_errors = []
    for entry in study_preset.component_index.entries:
        if entry.q == entry.r:
            expected = (study_features[entry.q] @ study_features[entry.q].T) / args.m
        else:
            expected = (
                study_features[entry.q] @ study_features[entry.r].T
                + study_features[entry.r] @ study_features[entry.q].T
            ) / args.m
        discrepancy = scale_aware_max_discrepancy(dense_kernels[entry.index], expected)
        all_kernel_errors.append(discrepancy)
        if entry.q != entry.r:
            off_diagonal_errors.append(discrepancy)
    dense_rhs = kernel_rhs(dense_kernels, normalized_phenotype)
    dense_oracle = {
        "rhs_discrepancy": float(
            scale_aware_max_discrepancy(summary.genetic_rhs, dense_rhs)
        ),
        "maximum_offdiagonal_kernel_discrepancy": float(
            max(off_diagonal_errors, default=0.0)
        ),
        "maximum_all_kernel_discrepancy": float(max(all_kernel_errors, default=0.0)),
        "signed_generating_offdiagonal_count": int(
            np.sum(
                np.asarray(
                    [
                        definition.omega[entry.q, entry.r]
                        for entry in study_preset.component_index.entries
                        if entry.q != entry.r
                    ]
                )
                < 0.0
            )
        ),
        "negative_offdiagonal_rhs_count": int(
            np.sum(
                np.asarray(
                    [
                        summary.genetic_rhs[entry.index]
                        for entry in study_preset.component_index.entries
                        if entry.q != entry.r
                    ]
                )
                < 0.0
            )
        ),
    }
    mode_discrepancies: dict[str, float | None] = {
        "raw_eigenvalues": float(
            scale_aware_max_discrepancy(
                raw_payload["eigenvalues"], oracle_raw["eigenvalues"]
            )
        ),
        "raw_functions": float(
            scale_aware_max_discrepancy(
                np.abs(raw_payload["functions"]), np.abs(oracle_raw["functions"])
            )
        ),
        "psd_eigenvalues": (
            None
            if psd_payload is None or oracle_psd is None
            else float(
                scale_aware_max_discrepancy(
                    psd_payload["eigenvalues"], oracle_psd["eigenvalues"]
                )
            )
        ),
        "maximum_public_equation_residual": float(
            np.max(
                np.abs(raw_payload["scale_aware_equation_residuals"]),
                initial=0.0,
            )
        ),
        "maximum_psd_scale_aware_equation_residual": (
            None
            if psd_payload is None
            else float(
                np.max(
                    np.abs(psd_payload["scale_aware_equation_residuals"]),
                    initial=0.0,
                )
            )
        ),
    }
    expected_truth_modes = _metric_modes(
        definition.omega,
        reference_preset.basis_metric,
        study_preset.context_grid,
    )
    result = {
        "name": definition.name,
        "label": definition.label,
        "q": definition.q,
        "study_n": args.study_n,
        "reference_n": args.reference_n,
        "m": args.m,
        "context_correlation": definition.correlation,
        "context_pc_correlation": definition.context_pc_correlation,
        "generating_omega": definition.omega.tolist(),
        "expected_operator_rank": definition.expected_operator_rank,
        "truth_operator_eigenvalues": expected_truth_modes["eigenvalues"].tolist(),
        "raw_fitted_omega": fit.raw_omegas[0].tolist(),
        "psd_fitted_omega": (
            None
            if fit.psd_projection is None
            else fit.psd_projection.projected_omegas[0].tolist()
        ),
        "optional_psd_projection": psd_diagnostics,
        "psd_loo_exercised": psd_loo_exercised,
        "raw_modes": _json_safe(raw_payload),
        "psd_modes": _json_safe(psd_payload),
        "mode_discrepancies": mode_discrepancies,
        "mode_oracle_surface_reconstruction_discrepancy": float(
            oracle_raw["surface_reconstruction_discrepancy"]
        ),
        "mode_oracle_metric_orthonormality_discrepancy": float(
            oracle_raw["metric_orthonormality_discrepancy"]
        ),
        "contrasts": _json_safe(contrast_results),
        "five_pairwise_surface_quantities": _json_safe(pairwise_surface),
        "basis_invariance": _json_safe(invariance),
        "dense_oracle": dense_oracle,
        "fixed_effect_oracle": _json_safe(fixed_effect_oracle),
        "conditioning": _conditioning_payload(study_preset, fit),
        "fit": {
            "rank": int(fit.solve.rank),
            "dimension": int(fit.raw_coefficients.size),
            "condition_number": float(fit.solve.condition_number),
            "relative_residual": float(fit.solve.relative_residual),
            "loo_replicates": int(fit.loo_coefficients.shape[0]),
            "reference_total_seconds": float(reference.phase_times_seconds["total"]),
            "trait_total_seconds": float(summary.phase_times_seconds["total"]),
            "decode_passes": int(summary.decode_passes),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    plot_data = {
        "label": definition.label,
        "grid": np.asarray(study_preset.context_grid),
        "truth_modes": expected_truth_modes,
        "display_modes": psd_payload if psd_payload is not None else raw_payload,
        "display_interpretation": (
            "PSD projected" if psd_payload is not None else "raw algebraic"
        ),
    }
    return result, plot_data


def _missingness_contract(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 4000]))
    n = 48
    sources = {
        "environment_1": rng.normal(size=n),
        "environment_2": rng.normal(size=n),
    }
    mask = np.ones(n, dtype=bool)
    mask[[3, 17, 41]] = False
    sources["environment_1"][[3, 17]] = np.nan
    sources["environment_2"][[17, 41]] = np.nan
    covariates = {
        "pc1": rng.normal(size=n),
        "pc2": rng.normal(size=n),
        "age_covariate": rng.normal(size=n),
    }
    calibration = calibrate_multienvironment_basis(
        sources,
        _source_specs(3),
        mask=mask,
        basis_id="missingness_contract",
    )
    preset = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=mask,
        covariates=covariates,
        interactions=_interaction_specs(3),
    )
    implicit_mask_rejected = False
    try:
        apply_multienvironment_calibration(
            calibration,
            sources,
            covariates=covariates,
        )
    except (TypeError, ValueError):
        implicit_mask_rejected = True
    bad_sources = {name: np.array(value, copy=True) for name, value in sources.items()}
    bad_sources["environment_2"][5] = np.nan
    retained_nonfinite_rejected = False
    try:
        apply_multienvironment_calibration(
            calibration,
            bad_sources,
            mask=mask,
            covariates=covariates,
        )
    except ValueError:
        retained_nonfinite_rejected = True
    return {
        "original_rows": n,
        "declared_retained_rows": int(np.sum(mask)),
        "preset_rows": int(preset.basis.shape[0]),
        "implicit_mask_rejected": implicit_mask_rejected,
        "retained_nonfinite_rejected": retained_nonfinite_rejected,
        "manifest_mask_policy": getattr(preset, "manifest", {}).get(
            "missingness", getattr(preset, "manifest", {}).get("retained_mask")
        ),
        "gate_pass": bool(
            preset.basis.shape[0] == int(np.sum(mask))
            and implicit_mask_rejected
            and retained_nonfinite_rejected
        ),
    }


def _mixed_categorical_contract(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 4500]))
    n = 90
    categories = np.asarray((["never", "former", "current"] * (n // 3)), dtype=object)
    rng.shuffle(categories)
    sources = {
        "age": 55.0 + 8.0 * rng.standard_normal(n),
        "smoking": categories,
    }
    specs = (
        MultiEnvironmentSourceSpec("age", "continuous"),
        MultiEnvironmentSourceSpec(
            "smoking",
            "categorical",
            categories=("never", "former", "current"),
            reference_category="never",
        ),
    )
    mask = np.ones(n, dtype=bool)
    calibration = calibrate_multienvironment_basis(
        sources,
        specs,
        mask=mask,
        basis_id="mixed_continuous_categorical_contract",
    )
    preset = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=mask,
        annotation_names=("all",),
    )
    metric_oracle = preset.basis.T @ preset.basis / n
    unknown_sources = {name: value.copy() for name, value in sources.items()}
    unknown_sources["smoking"][0] = "unknown"
    unknown_rejected = False
    try:
        apply_multienvironment_calibration(
            calibration,
            unknown_sources,
            mask=mask,
        )
    except ValueError:
        unknown_rejected = True
    summary_only = preset.without_individual_data()
    dummy_oracle = (
        np.column_stack([categories == "former", categories == "current"]).astype(
            np.float64
        )
        - np.asarray([1.0 / 3.0, 1.0 / 3.0])[None, :]
    )
    dummy_discrepancy = scale_aware_max_discrepancy(preset.basis[:, 2:], dummy_oracle)
    metric_discrepancy = scale_aware_max_discrepancy(preset.basis_metric, metric_oracle)
    return {
        "n": n,
        "q": int(preset.basis.shape[1]),
        "category_counts": {
            label: int(np.sum(categories == label))
            for label in ("never", "former", "current")
        },
        "basis_names": list(preset.manifest.get("basis_names", ())),
        "dummy_encoding_discrepancy": float(dummy_discrepancy),
        "uncentered_metric_discrepancy": float(metric_discrepancy),
        "unknown_category_rejected": unknown_rejected,
        "summary_only_rows_removed": bool(
            summary_only.basis.shape[0] == 0
            and summary_only.fixed_effect_design.shape[0] == 0
        ),
        "genetic_component_count": int(len(preset.component_index)),
        "gate_pass": bool(
            preset.basis.shape == (n, 4)
            and dummy_discrepancy < 1.0e-14
            and metric_discrepancy < 1.0e-14
            and unknown_rejected
            and summary_only.basis.shape[0] == 0
            and summary_only.fixed_effect_design.shape[0] == 0
            and len(preset.component_index) == 10
        ),
    }


def _redundant_pruning(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 5000]))
    n = 128
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    sources = {
        "environment_1": first,
        "environment_2": second,
        "environment_3": first + second,
    }
    covariates = {
        "pc1": rng.normal(size=n),
        "pc2": rng.normal(size=n),
        "age_covariate": rng.normal(size=n),
    }
    calibration = calibrate_multienvironment_basis(
        sources,
        _source_specs(4),
        mask=np.ones(n, dtype=bool),
        basis_id="redundant_pruning",
    )
    preset = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=np.ones(n, dtype=bool),
        covariates=covariates,
    )
    pruning = fit_multienvironment_pruning(calibration, rtol=1.0e-10)
    pruned_preset = apply_multienvironment_calibration(
        calibration,
        sources,
        mask=np.ones(n, dtype=bool),
        covariates=covariates,
        pruning=pruning,
    )
    requested_rank = int(np.linalg.matrix_rank(np.asarray(preset.basis), tol=1.0e-10))
    retained_rank = int(
        np.linalg.matrix_rank(np.asarray(pruned_preset.basis), tol=1.0e-10)
    )
    return {
        "requested_q": int(preset.basis.shape[1]),
        "requested_rank": requested_rank,
        "retained_q": int(pruned_preset.basis.shape[1]),
        "retained_rank": retained_rank,
        "retained_indices": _json_safe(pruning.retained_indices),
        "dropped_indices": _json_safe(pruning.dropped_indices),
        "reconstruction": _json_safe(pruning.reconstruction),
        "status": str(getattr(pruning, "status", "defined_explicit_pruning")),
        "manifest": _json_safe(getattr(pruning, "manifest", {})),
        "gate_pass": bool(
            requested_rank < preset.basis.shape[1]
            and retained_rank == pruned_preset.basis.shape[1]
            and pruned_preset.basis.shape[1] == requested_rank
        ),
    }


def _performance_validation(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for q in (2, 3, 4):
        definition = SyntheticDefinition(
            name=f"benchmark_q{q}",
            label=f"Benchmark Q={q}",
            q=q,
            correlation=0.25,
            context_pc_correlation=0.30,
            omega=0.16 * np.eye(q),
            expected_operator_rank=q,
        )
        reference_times: list[float] = []
        trait_times: list[float] = []
        feature_times: list[float] = []
        reference_phase: list[float] = []
        trait_phase: list[float] = []
        peak_rss = 0
        decode_counts: list[int] = []
        for repeat in range(args.benchmark_repeats):
            streams = np.random.SeedSequence([args.seed, 6000, q, repeat]).spawn(2)
            reference_cohort = _make_cohort(
                definition,
                n=args.benchmark_n,
                m=args.benchmark_m,
                seed=streams[0],
            )
            calibration = _calibrate_reference(reference_cohort)
            preset = _apply_calibration(calibration, reference_cohort)
            groups = tuple(f"block:{index % 12}" for index in range(args.benchmark_m))
            annotations = np.ones((args.benchmark_m, 1), dtype=np.float64)
            variant_hash = canonical_sha256({"benchmark_q": q, "m": args.benchmark_m})
            started = time.perf_counter()
            common_scale_features(
                reference_cohort.genotype,
                preset.basis,
                preset.projector.projector,
            )
            feature_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            reference = build_context_reference(
                genotype=reference_cohort.genotype,
                basis=preset.basis,
                projector=preset.projector,
                annotations=annotations,
                component_index=preset.component_index,
                basis_hash=_basis_hash(preset),
                fixed_effect_hash=_fixed_effect_array_hash(preset),
                variant_hash=variant_hash,
                loo_groups=groups,
                genotype_scaling=GENOTYPE_SCALING,
                gram_method="exact",
                same_person_method="exact",
            )
            reference_times.append(time.perf_counter() - started)
            reference_phase.append(
                reference.phase_times_seconds["feature_construction"]
            )
            phenotype = np.random.default_rng(streams[1]).normal(size=args.benchmark_n)
            started = time.perf_counter()
            summary = build_context_trait_summary(
                genotype=reference_cohort.genotype,
                basis=preset.basis,
                phenotype=phenotype,
                projector=preset.projector,
                annotations=annotations,
                component_index=preset.component_index,
                residual_basis=preset.residual_basis,
                residual_names=preset.residual_names,
                basis_hash=_basis_hash(preset),
                fixed_effect_hash=_fixed_effect_array_hash(preset),
                variant_hash=variant_hash,
                loo_groups=groups,
                genotype_scaling=GENOTYPE_SCALING,
                block_size=args.block_size,
            )
            trait_times.append(time.perf_counter() - started)
            trait_phase.append(summary.phase_times_seconds["genotype_pass"])
            peak_rss = max(peak_rss, reference.peak_rss_bytes, summary.peak_rss_bytes)
            decode_counts.append(int(summary.decode_passes))
        records.append(
            {
                "q": q,
                "p_genetic": q * (q + 1) // 2,
                "n": args.benchmark_n,
                "m": args.benchmark_m,
                "feature_projection_seconds_median": float(np.median(feature_times)),
                "reference_total_seconds_median": float(np.median(reference_times)),
                "reference_feature_phase_seconds_median": float(
                    np.median(reference_phase)
                ),
                "trait_total_seconds_median": float(np.median(trait_times)),
                "trait_genotype_pass_seconds_median": float(np.median(trait_phase)),
                "absolute_peak_rss_bytes": int(peak_rss),
                "decode_passes_observed": sorted(set(decode_counts)),
                "dominant_gemm_count_relative_to_q1": int(q),
                "note": (
                    "feature projection isolates the Q source GEMMs; exact reference "
                    "total also includes the superlinear P_g by P_g Gram/contributions"
                ),
            }
        )
    baseline = records[0]["feature_projection_seconds_median"] / 2.0
    for record in records:
        record["feature_projection_time_over_q_baseline"] = float(
            record["feature_projection_seconds_median"]
            / max(record["q"] * baseline, np.finfo(float).tiny)
        )
    return records


def _read_real_fixture(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import pandas as pd
        from bed_reader import open_bed
    except ImportError as exc:
        raise RuntimeError("--real-traits requires pandas and bed-reader.") from exc
    traits = ("bmi", "hdl", "c_reactive_prot", "smoking_status")
    required = [
        Path(f"{args.geno_prefix}.bed"),
        Path(f"{args.geno_prefix}.bim"),
        Path(f"{args.geno_prefix}.fam"),
        args.covariate_file,
        *(args.phenotype_root / f"{trait}.pheno" for trait in traits),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required real-data inputs are absent: {missing}")

    def read_table(path: Path, usecols: Sequence[str]) -> Any:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            usecols=list(usecols),
            dtype={"FID": str, "IID": str},
        )
        if frame[["FID", "IID"]].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate identifiers.")
        return frame

    fam = pd.read_csv(
        Path(f"{args.geno_prefix}.fam"),
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype=str,
    )
    if fam.duplicated().any():
        raise ValueError("PLINK FAM contains duplicate identifiers.")
    fam["bed_row"] = np.arange(len(fam), dtype=np.int64)
    covariates = read_table(
        args.covariate_file,
        ("FID", "IID", "sex", "age", *PC_COLUMNS),
    )
    merged = fam.merge(covariates, on=["FID", "IID"], validate="one_to_one")
    for trait in traits:
        values = read_table(
            args.phenotype_root / f"{trait}.pheno",
            ("FID", "IID", "pheno"),
        ).rename(columns={"pheno": trait})
        merged = merged.merge(values, on=["FID", "IID"], validate="one_to_one")
    numeric = ["sex", "age", *PC_COLUMNS, *traits]
    values = merged[numeric].to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(values), axis=1) & np.all(values != -9.0, axis=1)
    intersection = merged.loc[valid].copy()
    if len(intersection) < args.real_n:
        raise ValueError(
            f"Only {len(intersection)} complete cases remain for --real-n={args.real_n}."
        )
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 8000]))
    selected_indices = np.sort(
        rng.choice(len(intersection), size=args.real_n, replace=False)
    )
    selected = (
        intersection.iloc[selected_indices]
        .sort_values("bed_row")
        .reset_index(drop=True)
    )
    bed = open_bed(Path(f"{args.geno_prefix}.bed"))
    candidate_count = min(int(bed.sid_count), 4 * args.real_m)
    candidates = np.unique(
        np.linspace(0, int(bed.sid_count) - 1, candidate_count, dtype=np.int64)
    )
    raw = bed.read(
        index=(selected["bed_row"].to_numpy(dtype=np.int64), candidates),
        dtype="float64",
        order="C",
    )
    missing_rate = np.mean(np.isnan(raw), axis=0)
    means = np.nanmean(raw, axis=0)
    allele_frequency = means / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    variances = np.nanvar(raw, axis=0, ddof=1)
    candidate_valid = (
        np.isfinite(means)
        & np.isfinite(variances)
        & (variances > np.finfo(np.float64).eps)
        & (missing_rate <= 0.05)
        & (maf >= 0.05)
    )
    retained = np.flatnonzero(candidate_valid)[: args.real_m]
    if retained.size != args.real_m:
        raise ValueError(
            f"Only {retained.size} deterministic candidate variants passed QC."
        )
    genotype = raw[:, retained].copy()
    selected_means = means[retained]
    missing_genotypes = np.isnan(genotype)
    if np.any(missing_genotypes):
        genotype[missing_genotypes] = np.broadcast_to(
            selected_means[None, :], genotype.shape
        )[missing_genotypes]
    genotype = _standardize(genotype)
    chosen_variant_indices = candidates[retained]
    sex_values, sex_counts = np.unique(selected["sex"], return_counts=True)
    smoking_values, smoking_counts = np.unique(
        selected["smoking_status"], return_counts=True
    )
    fixture = {
        "selected": selected,
        "genotype": genotype,
        "variant_hash": canonical_sha256(
            {"selected_index_sha256": array_sha256(chosen_variant_indices)}
        ),
    }
    diagnostics = {
        "source_sample_count": int(bed.iid_count),
        "source_variant_count": int(bed.sid_count),
        "complete_case_intersection_count": int(len(intersection)),
        "selected_sample_count": int(args.real_n),
        "selected_variant_count": int(args.real_m),
        "candidate_variant_count": int(candidate_count),
        "sex_counts": {
            str(value): int(count) for value, count in zip(sex_values, sex_counts)
        },
        "smoking_counts": {
            str(value): int(count)
            for value, count in zip(smoking_values, smoking_counts)
        },
        "sample_selection": "seeded_without_replacement_from_complete_cases",
        "variant_selection": "evenly_spaced_candidates_then_missingness_maf_qc",
        "phenotype_scale": "values_used_as_stored_no_raw_scale_or_significance_claim",
    }
    return fixture, diagnostics


def _real_designs(selected: Any) -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "q3_age_sex",
            "source_data": {
                "age": selected["age"].to_numpy(dtype=np.float64),
                "sex": selected["sex"].to_numpy(dtype=object),
            },
            "source_specs": (
                MultiEnvironmentSourceSpec("age", "continuous"),
                MultiEnvironmentSourceSpec(
                    "sex",
                    "categorical",
                    categories=(1.0, 2.0),
                    reference_category=1.0,
                ),
            ),
            "interaction_specs": (
                FixedEffectInteractionSpec("age", "pc1", "age_by_pc1"),
            ),
            "traits": ("bmi", "hdl"),
        },
        {
            "name": "q4_age_smoking",
            "source_data": {
                "age": selected["age"].to_numpy(dtype=np.float64),
                "smoking_status": selected["smoking_status"].to_numpy(dtype=object),
            },
            "source_specs": (
                MultiEnvironmentSourceSpec("age", "continuous"),
                MultiEnvironmentSourceSpec(
                    "smoking_status",
                    "categorical",
                    categories=(0, 1, 2),
                    reference_category=0,
                ),
            ),
            "interaction_specs": (
                FixedEffectInteractionSpec("age", "pc1", "age_by_pc1"),
            ),
            "traits": ("bmi", "c_reactive_prot"),
        },
    )


def _real_trait_sanity(args: argparse.Namespace) -> dict[str, Any]:
    fixture, diagnostics = _read_real_fixture(args)
    selected = fixture["selected"]
    covariates = {
        f"pc{index}": selected[column].to_numpy(dtype=np.float64)
        for index, column in enumerate(PC_COLUMNS, start=1)
    }
    annotations = np.ones((args.real_m, 1), dtype=np.float64)
    groups = tuple(f"block:{index % args.loo_groups}" for index in range(args.real_m))
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for design_index, design in enumerate(_real_designs(selected)):
        calibration = calibrate_multienvironment_basis(
            design["source_data"],
            design["source_specs"],
            mask=np.ones(args.real_n, dtype=bool),
            basis_id=design["name"],
        )
        preset = apply_multienvironment_calibration(
            calibration,
            design["source_data"],
            mask=np.ones(args.real_n, dtype=bool),
            covariates=covariates,
            interactions=design["interaction_specs"],
            genotype_for_diagnostics=fixture["genotype"],
        )
        reference = build_context_reference(
            genotype=fixture["genotype"],
            basis=preset.basis,
            projector=preset.projector,
            annotations=annotations,
            component_index=preset.component_index,
            basis_hash=_basis_hash(preset),
            fixed_effect_hash=_fixed_effect_array_hash(preset),
            variant_hash=fixture["variant_hash"],
            loo_groups=groups,
            genotype_scaling="study_mean_imputed_and_unit_variance",
            gram_method="exact",
            same_person_method="exact",
        )
        for trait_index, trait in enumerate(design["traits"]):
            summary = build_context_trait_summary(
                genotype=fixture["genotype"],
                basis=preset.basis,
                phenotype=selected[trait].to_numpy(dtype=np.float64),
                projector=preset.projector,
                annotations=annotations,
                component_index=preset.component_index,
                residual_basis=preset.residual_basis,
                residual_names=preset.residual_names,
                basis_hash=_basis_hash(preset),
                fixed_effect_hash=_fixed_effect_array_hash(preset),
                variant_hash=fixture["variant_hash"],
                loo_groups=groups,
                genotype_scaling="study_mean_imputed_and_unit_variance",
                block_size=args.block_size,
            )
            fit = fit_multienvironment_model(
                reference,
                summary,
                reference_preset=preset,
                study_preset=preset,
                project_psd=False,
            )
            fit, psd_diagnostics = _attach_optional_psd(fit)
            modes = derive_covariance_modes(
                fit,
                preset.without_individual_data(),
                annotation="all",
                eigengap_rtol=args.eigengap_rtol,
                use_psd=False,
            )
            mode_payload = _public_mode_payload(modes, preset, fit.raw_omegas[0])
            psd_oracle = (
                None
                if fit.psd_projection is None
                else _metric_modes(
                    fit.psd_projection.projected_omegas[0],
                    preset.basis_metric,
                    preset.context_grid,
                )
            )
            left, right = _context_pair(preset)
            pairwise_surface = _pairwise_surface_payload(
                fit, left, right, include_loo_values=False
            )
            contrasts = {
                kind: _aggregate_contrast_payload(
                    derive_context_contrast(
                        fit,
                        preset.without_individual_data(),
                        left,
                        right,
                        annotation="all",
                        kind=kind,
                    )
                )
                for kind in ("variance_difference", "covariance")
            }
            records.append(
                {
                    "design": design["name"],
                    "trait": trait,
                    "q": int(preset.basis.shape[1]),
                    "rank": int(fit.solve.rank),
                    "dimension": int(fit.raw_coefficients.size),
                    "condition_number": float(fit.solve.condition_number),
                    "relative_residual": float(fit.solve.relative_residual),
                    "raw_omega": fit.raw_omegas[0].tolist(),
                    "psd_omega": (
                        None
                        if fit.psd_projection is None
                        else fit.psd_projection.projected_omegas[0].tolist()
                    ),
                    "optional_psd_projection": psd_diagnostics,
                    "raw_operator_eigenvalues": mode_payload["eigenvalues"].tolist(),
                    "psd_operator_eigenvalues": (
                        None
                        if psd_oracle is None
                        else psd_oracle["eigenvalues"].tolist()
                    ),
                    "operator_eigenvalue_standard_errors": mode_payload[
                        "eigenvalue_standard_errors"
                    ].tolist(),
                    "raw_rank_one_fraction": mode_payload["rank_one_fraction"],
                    "psd_rank_one_fraction": (
                        None if psd_oracle is None else psd_oracle["rank_one_fraction"]
                    ),
                    "mode_stability": _json_safe(mode_payload["stability"]),
                    "contrasts": _json_safe(contrasts),
                    "five_pairwise_surface_quantities": _json_safe(pairwise_surface),
                    "conditioning": _conditioning_payload(preset, fit),
                    "decode_passes": int(summary.decode_passes),
                    "loo_replicates": int(fit.loo_coefficients.shape[0]),
                }
            )
    return {
        "purpose": "privacy_preserving_numerical_sanity_not_significance_gate",
        "diagnostics": diagnostics,
        "records": records,
        "elapsed_seconds": float(time.perf_counter() - started),
        "privacy": {
            "identifiers_written": False,
            "sample_rows_written": False,
            "variant_rows_written": False,
            "outputs": "aggregate_counts_fit_diagnostics_modes_and_contrasts_only",
        },
    }


def _plot_modes(plot_data: Sequence[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(plot_data), 2, figsize=(11.5, 3.3 * len(plot_data)), squeeze=False
    )
    for row, item in enumerate(plot_data):
        truth = np.asarray(item["truth_modes"]["eigenvalues"])
        fitted = np.asarray(item["display_modes"]["eigenvalues"])
        x = np.arange(truth.size)
        axes[row, 0].bar(x - 0.18, truth, width=0.36, label="generating")
        axes[row, 0].bar(
            x + 0.18,
            fitted,
            width=0.36,
            label=f"fitted {item['display_interpretation']}",
        )
        axes[row, 0].set_xticks(x, [f"mode {index + 1}" for index in x])
        axes[row, 0].set_ylabel("operator eigenvalue")
        axes[row, 0].set_title(item["label"])
        axes[row, 0].grid(axis="y", alpha=0.2)
        axes[row, 0].legend(frameon=False)
        functions = np.asarray(item["display_modes"]["functions"])
        for mode in range(min(functions.shape[1], 3)):
            axes[row, 1].plot(
                np.arange(functions.shape[0]),
                functions[:, mode],
                marker="o",
                label=f"mode {mode + 1}",
            )
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 1].set_xlabel("declared context-grid point")
        axes[row, 1].set_ylabel("context function")
        axes[row, 1].set_title(
            f"{item['display_interpretation']} covariance-surface modes"
        )
        axes[row, 1].grid(alpha=0.2)
        axes[row, 1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_dir, "modes")


def _plot_conditioning(
    cases: Sequence[dict[str, Any]], pruning: Mapping[str, Any], output_dir: Path
) -> None:
    labels = [case["label"] for case in cases]
    correlations = [
        case["conditioning"]["maximum_absolute_context_correlation"] for case in cases
    ]
    metric_conditions = [
        case["conditioning"]["metric_condition_number"] for case in cases
    ]
    normal_conditions = [
        case["conditioning"]["normal_condition_number"] for case in cases
    ]
    invariance_errors = [
        max(
            case["basis_invariance"]["surface_discrepancy"],
            case["basis_invariance"]["operator_eigenvalue_discrepancy"],
            case["basis_invariance"]["mode_function_discrepancy"],
            case["basis_invariance"]["tied_mode_subspace_discrepancy"],
            case["basis_invariance"]["tied_mode_function_projector_discrepancy"],
        )
        for case in cases
    ]
    x = np.arange(len(cases))
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].bar(x, correlations, color="#4C78A8")
    axes[0].set_ylabel("maximum |context correlation|")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Context dependence")
    axes[1].bar(x - 0.18, metric_conditions, 0.36, label="basis metric")
    axes[1].bar(x + 0.18, normal_conditions, 0.36, label="normal system")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("condition number")
    axes[1].set_title("Conditioning")
    axes[1].legend(frameon=False)
    axes[2].bar(x, np.maximum(invariance_errors, 1.0e-18), color="#59A14F")
    axes[2].set_yscale("log")
    axes[2].axhline(1.0e-10, color="#B24A3A", linestyle="--")
    axes[2].set_ylabel("maximum invariant discrepancy")
    axes[2].set_title(
        f"Basis invariance\npruning Q {pruning['requested_q']}→{pruning['retained_q']}"
    )
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_dir, "conditioning")


def _plot_performance(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    q = np.asarray([record["q"] for record in records])
    feature = np.asarray(
        [record["feature_projection_seconds_median"] for record in records]
    )
    reference = np.asarray(
        [record["reference_total_seconds_median"] for record in records]
    )
    trait = np.asarray([record["trait_total_seconds_median"] for record in records])
    rss = (
        np.asarray([record["absolute_peak_rss_bytes"] for record in records]) / 2**20
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    axes[0].plot(q, feature, "o-", label="Q feature projection")
    axes[0].plot(q, reference, "s-", label="full exact reference")
    axes[0].plot(q, trait, "^-", label="trait summary")
    axes[0].set_xlabel("basis dimension Q")
    axes[0].set_ylabel("median wall time (seconds)")
    axes[0].set_title("Reference and trait scaling")
    axes[0].legend(frameon=False)
    axes[1].plot(q, rss, "o-", color="#F28E2B")
    axes[1].set_xlabel("basis dimension Q")
    axes[1].set_ylabel("absolute process peak RSS (MiB)")
    axes[1].set_title("Memory high-water mark")
    for axis in axes:
        axis.set_xticks(q)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_dir, "performance")


def _plot_real(real: Mapping[str, Any], output_dir: Path) -> None:
    records = real["records"]
    labels = [f"{record['trait']}\n{record['design']}" for record in records]
    spectra = [
        np.asarray(
            record["psd_operator_eigenvalues"]
            if record["psd_operator_eigenvalues"] is not None
            else record["raw_operator_eigenvalues"]
        )
        for record in records
    ]
    maximum_q = max(value.size for value in spectra)
    x = np.arange(len(records))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    width = 0.75 / maximum_q
    for mode in range(maximum_q):
        values = [
            spectrum[mode] if mode < spectrum.size else 0.0 for spectrum in spectra
        ]
        axes[0].bar(
            x + (mode - (maximum_q - 1) / 2.0) * width,
            values,
            width,
            label=f"mode {mode + 1}",
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("operator eigenvalue")
    axes[0].set_title("Aggregate real-trait modes (PSD when available)")
    axes[0].legend(frameon=False)
    axes[1].bar(
        x,
        [record["condition_number"] for record in records],
        color="#76B7B2",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("normal-system condition number")
    axes[1].set_title("Real-trait numerical conditioning")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_dir, "real_traits")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    plot_data: list[dict[str, Any]] = []
    for index, definition in enumerate(_definitions()):
        result, plotting = _run_synthetic_case(definition, case_index=index, args=args)
        cases.append(result)
        plot_data.append(plotting)
    missingness = _missingness_contract(args)
    mixed_categorical = _mixed_categorical_contract(args)
    pruning = _redundant_pruning(args)
    performance = _performance_validation(args)
    real = _real_trait_sanity(args) if args.real_traits else None

    dense_gate = (
        all(
            case["dense_oracle"]["rhs_discrepancy"] < 1.0e-11
            and case["dense_oracle"]["maximum_offdiagonal_kernel_discrepancy"] < 1.0e-11
            and case["dense_oracle"]["maximum_all_kernel_discrepancy"] < 1.0e-11
            and case["dense_oracle"]["signed_generating_offdiagonal_count"] >= 1
            for case in cases
        )
        and sum(
            case["dense_oracle"]["negative_offdiagonal_rhs_count"] for case in cases
        )
        >= 1
    )
    fixed_effect_gate = all(
        case["fixed_effect_oracle"]["projector_discrepancy"] < 1.0e-11
        and case["fixed_effect_oracle"]["feature_discrepancy"] < 1.0e-11
        and case["fixed_effect_oracle"]["projected_fixed_effect_residual"] < 1.0e-10
        and case["fixed_effect_oracle"]["wrong_order_separation"] > 1.0e-6
        for case in cases
    )
    mode_gate = all(
        case["mode_discrepancies"]["raw_eigenvalues"] < 1.0e-10
        and case["mode_discrepancies"]["raw_functions"] < 1.0e-9
        and case["mode_discrepancies"]["maximum_public_equation_residual"] < 1.0e-10
        and case["raw_modes"]["function_field_discrepancy"] < 1.0e-12
        and case["mode_oracle_surface_reconstruction_discrepancy"] < 1.0e-10
        and case["mode_oracle_metric_orthonormality_discrepancy"] < 1.0e-10
        for case in cases
    )
    successful_psd_cases = [case for case in cases if case["psd_modes"] is not None]
    psd_mode_gate = all(
        case["mode_discrepancies"]["psd_eigenvalues"] < 1.0e-10
        and case["mode_discrepancies"]["maximum_psd_scale_aware_equation_residual"]
        < 1.0e-10
        and case["psd_modes"]["function_field_discrepancy"] < 1.0e-12
        for case in successful_psd_cases
    )
    invariance_gate = all(
        case["basis_invariance"][name] < 1.0e-10
        for case in cases
        for name in (
            "surface_discrepancy",
            "operator_eigenvalue_discrepancy",
            "mode_coefficient_discrepancy",
            "mode_function_discrepancy",
            "tied_mode_subspace_discrepancy",
            "tied_mode_function_projector_discrepancy",
        )
    )
    contrast_gate = all(
        contrast["estimate_discrepancy"] < 1.0e-10
        and contrast["manual_loo_discrepancy"] < 1.0e-10
        and contrast["manual_standard_error_discrepancy"] < 1.0e-10
        and len(contrast["loo_values"]) == args.loo_groups
        for case in cases
        for contrast in case["contrasts"].values()
    )
    fit_gate = all(
        case["fit"]["rank"] == case["fit"]["dimension"]
        and case["fit"]["relative_residual"] < 1.0e-10
        and case["fit"]["loo_replicates"] == args.loo_groups
        and case["fit"]["decode_passes"] == 1
        for case in cases
    )
    performance_gate = all(
        record["decode_passes_observed"] == [1] for record in performance
    )
    psd_summary = {
        "defined_projection_cases": int(
            sum(
                case["optional_psd_projection"]["status"]
                == "defined_separate_from_raw_fit"
                for case in cases
            )
        ),
        "indeterminate_projection_cases": int(
            sum(
                case["optional_psd_projection"]["status"]
                != "defined_separate_from_raw_fit"
                for case in cases
            )
        ),
        "full_psd_loo_cases": int(
            sum(bool(case["psd_loo_exercised"]) for case in cases)
        ),
        "policy": (
            "optional_projection_failure_is_reported_and_does_not_invalidate_raw_gates"
        ),
    }
    gates = {
        "dense_signed_offdiagonal_oracle": dense_gate,
        "fixed_effect_projector_and_pdqg_order": fixed_effect_gate,
        "public_modes_match_independent_oracle": mode_gate,
        "successful_optional_psd_modes_match_oracle": psd_mode_gate,
        "basis_surface_operator_invariance": invariance_gate,
        "pairwise_contrast_and_loo": contrast_gate,
        "fit_rank_solve_and_single_decode": fit_gate,
        "declared_missingness_mask": bool(missingness["gate_pass"]),
        "mixed_continuous_categorical_basis": bool(mixed_categorical["gate_pass"]),
        "explicit_redundant_basis_pruning": bool(pruning["gate_pass"]),
        "q2_q3_q4_performance_single_decode": performance_gate,
    }
    payload = {
        "kind": "summit.context.multienvironment_validation",
        "seed": int(args.seed),
        "configuration": {
            "study_n": int(args.study_n),
            "reference_n": int(args.reference_n),
            "m": int(args.m),
            "loo_groups": int(args.loo_groups),
            "block_size": int(args.block_size),
            "eigengap_rtol": float(args.eigengap_rtol),
            "full_psd_loo": bool(args.full_psd_loo),
            "basis_metric": "reference_uncentered_second_moment_phi_transpose_phi_over_n",
            "feature_mode": "raw_projected_common_genotype_scale",
        },
        "synthetic_cases": cases,
        "missingness_contract": missingness,
        "mixed_categorical_contract": mixed_categorical,
        "redundant_basis_pruning": pruning,
        "optional_psd_summary": psd_summary,
        "performance": performance,
        "real_trait_sanity": real,
        "gates": gates,
        "verdict": "pass" if all(gates.values()) else "review",
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_policy": {
            "row_level_data_written": False,
            "individual_identifiers_written": False,
            "variant_rows_written": False,
            "synthetic_loo_arrays_written": True,
            "real_trait_loo_arrays_written": False,
        },
    }
    _plot_modes(plot_data, args.output_dir)
    _plot_conditioning(cases, pruning, args.output_dir)
    _plot_performance(performance, args.output_dir)
    if real is not None:
        _plot_real(real, args.output_dir)
    return payload


def main() -> None:
    os.umask(0o077)
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    _prepare_output(args.output_dir, args.real_traits)
    payload = run_validation(args)
    output = args.output_dir / f"{OUTPUT_STEM}.json"
    _write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": payload["verdict"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "real_traits": bool(args.real_traits),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
