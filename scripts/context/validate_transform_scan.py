#!/usr/bin/env python3
"""Validate phenotype transformations and contextual covariance fits.

The default examples use synthetic data. Optional real-trait checks require
explicit local input paths and write aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    PhenotypeTransformSpec,
    TransformTrajectory,
    array_sha256,
    build_context_reference,
    build_context_transform_summary,
    build_transform_trajectory,
    canonical_sha256,
    classify_scale_trajectory,
    coefficients_to_omegas,
    common_scale_features,
    dense_genetic_kernels,
    dense_residual_kernels,
    fit_context_transform_scan,
    omegas_to_coefficients,
    pointwise_normal_bands,
    project_normalize_phenotype,
    rank_revealing_projector,
    scale_aware_max_discrepancy,
    select_simultaneous_band_coordinate,
    simultaneous_trajectory_bands,
)


OUTPUT_STEM = "06_transform_scan_validation"
GENOTYPE_SCALING = "synthetic_population_unit_variance"
TARGET_TRANSFORM_ID = "box_cox_0p5_target"
USER_TARGET_TRANSFORM_ID = "user_known_target"
COORDINATE_NAMES = ("amplification_cross_term", "second_mode_magnitude")
PC_COLUMNS = tuple(f"f.22009.0.{index}" for index in range(1, 6))


@dataclass(frozen=True)
class MechanismDefinition:
    name: str
    label: str
    omega: np.ndarray
    residual_variance: float
    context_mean_slope: float
    expected_classification: str
    sample_fraction: float = 1.0


@dataclass(frozen=True)
class SyntheticFixture:
    definition: MechanismDefinition
    genotype: np.ndarray
    basis: np.ndarray
    fixed: np.ndarray
    projector: Any
    annotations: np.ndarray
    components: ContextComponentIndex
    loo_groups: tuple[str, ...]
    hashes: dict[str, str]
    reference: Any
    original_phenotype: np.ndarray
    known_target: np.ndarray
    transformations: tuple[PhenotypeTransformSpec, ...]
    user_transforms: dict[str, np.ndarray]
    minimum_generating_covariance_eigenvalue: float
    normalization_multiplier: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--m", type=int, default=192)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--transform-tile-size", type=int, default=3)
    parser.add_argument(
        "--benchmark-l",
        type=int,
        nargs="+",
        default=(1, 4, 8, 12),
        metavar="L",
    )
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--calibration-replicates", type=int, default=300)
    parser.add_argument("--calibration-groups", type=int, default=48)
    parser.add_argument("--multiplier-draws", type=int, default=1024)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--amplification-margin", type=float, default=0.10)
    parser.add_argument("--heterogeneity-margin", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--real-traits",
        action="store_true",
        help="Also run a small aggregate-only BMI/CRP sanity analysis.",
    )
    parser.add_argument("--geno-prefix", type=Path, default=None)
    parser.add_argument("--phenotype-root", type=Path, default=None)
    parser.add_argument(
        "--covariate-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--real-n", type=int, default=256)
    parser.add_argument("--real-m", type=int, default=96)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n < 96:
        parser.error("--n must be at least 96")
    if args.m < 48:
        parser.error("--m must be at least 48")
    if args.loo_groups < 6 or args.m % args.loo_groups:
        parser.error("--loo-groups must be at least 6 and divide --m exactly")
    if args.block_size < 1 or args.transform_tile_size < 1:
        parser.error("block and transform tile sizes must be positive")
    benchmark_l = tuple(sorted({int(value) for value in args.benchmark_l}))
    if not benchmark_l or benchmark_l[0] < 1 or benchmark_l[-1] > 32:
        parser.error("--benchmark-l must contain integers in [1, 32]")
    args.benchmark_l = benchmark_l
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be positive")
    if args.calibration_replicates < 50:
        parser.error("--calibration-replicates must be at least 50")
    if args.calibration_groups < 6:
        parser.error("--calibration-groups must be at least 6")
    if args.multiplier_draws < 128:
        parser.error("--multiplier-draws must be at least 128")
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must lie in (0,1)")
    if args.amplification_margin <= 0.0 or args.heterogeneity_margin <= 0.0:
        parser.error("equivalence margins must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.real_traits:
        if any(value is None for value in (args.geno_prefix, args.phenotype_root, args.covariate_file)):
            parser.error("--real-traits requires --geno-prefix, --phenotype-root, and --covariate-file")
        if args.real_n < 96:
            parser.error("--real-n must be at least 96")
        if args.real_m < 48 or args.real_m % args.loo_groups:
            parser.error("--real-m must be at least 48 and divisible by --loo-groups")


def _output_paths(output_dir: Path, *, include_real: bool) -> tuple[Path, ...]:
    names = [f"{OUTPUT_STEM}.json"]
    for label in ("trajectories", "performance", "inference"):
        names.extend([f"{OUTPUT_STEM}_{label}.png", f"{OUTPUT_STEM}_{label}.pdf"])
    if include_real:
        names.extend(
            [f"{OUTPUT_STEM}_real_traits.png", f"{OUTPUT_STEM}_real_traits.pdf"]
        )
    return tuple(output_dir / name for name in names)


def _prepare_output_directory(output_dir: Path, *, include_real: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    existing = [
        path.name
        for path in _output_paths(output_dir, include_real=include_real)
        if path.exists()
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
    scales = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise RuntimeError("Synthetic standardization encountered a degenerate column.")
    return np.asarray(centered / scales, dtype=np.float64)


def _mechanisms() -> tuple[MechanismDefinition, ...]:
    return (
        MechanismDefinition(
            name="removable_amplification",
            label="Removable amplification",
            omega=np.asarray([[0.28, 0.0], [0.0, 0.0]], dtype=np.float64),
            residual_variance=0.50,
            context_mean_slope=0.80,
            expected_classification="removable_within_declared_family",
        ),
        MechanismDefinition(
            name="robust_rank_two",
            label="Robust rank-two",
            omega=np.asarray([[0.40, 0.0], [0.0, 0.80]], dtype=np.float64),
            residual_variance=0.20,
            context_mean_slope=0.20,
            expected_classification="robust_within_declared_family",
            sample_fraction=2.0,
        ),
        MechanismDefinition(
            name="weak_context_signal",
            label="Weak / uncertain",
            omega=np.asarray([[0.10, 0.015], [0.015, 0.012]], dtype=np.float64),
            residual_variance=0.86,
            context_mean_slope=0.35,
            expected_classification="indeterminate",
            sample_fraction=0.5,
        ),
    )


def _declared_transformations() -> tuple[PhenotypeTransformSpec, ...]:
    return (
        PhenotypeTransformSpec("identity", "identity"),
        PhenotypeTransformSpec("log", "log", shift=1.0),
        PhenotypeTransformSpec(
            "box_cox_0p25", "box_cox", shift=1.0, box_cox_lambda=0.25
        ),
        PhenotypeTransformSpec(
            TARGET_TRANSFORM_ID, "box_cox", shift=1.0, box_cox_lambda=0.5
        ),
        PhenotypeTransformSpec(
            "box_cox_0p75", "box_cox", shift=1.0, box_cox_lambda=0.75
        ),
        PhenotypeTransformSpec("box_cox_1", "box_cox", shift=1.0, box_cox_lambda=1.0),
        PhenotypeTransformSpec(
            USER_TARGET_TRANSFORM_ID,
            "user_supplied",
            source="known_box_cox_target",
        ),
        PhenotypeTransformSpec("invalid_shift", "log", shift=0.0),
        PhenotypeTransformSpec(
            "invalid_extreme",
            "box_cox",
            shift=1.0,
            box_cox_lambda=1000.0,
        ),
    )


def _sample_psd_covariance(
    rng: np.random.Generator, covariance: np.ndarray
) -> tuple[np.ndarray, float]:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues)) < -1.0e-10 * scale:
        raise RuntimeError("The synthetic generating covariance is indefinite.")
    draw = eigenvectors @ (
        np.sqrt(np.maximum(eigenvalues, 0.0)) * rng.standard_normal(eigenvalues.size)
    )
    return np.asarray(draw, dtype=np.float64), float(np.min(eigenvalues))


def _make_fixture(
    definition: MechanismDefinition,
    *,
    mechanism_index: int,
    args: argparse.Namespace,
) -> SyntheticFixture:
    n = max(96, int(round(args.n * definition.sample_fraction)))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, mechanism_index]))
    genotype = _standardize(rng.standard_normal((n, args.m)))
    context = _standardize(rng.uniform(-1.0, 1.0, size=(n, 1)))[:, 0]
    nuisance = _standardize(rng.standard_normal((n, 2)))
    basis = np.column_stack([np.ones(n, dtype=np.float64), context])
    fixed = np.column_stack([np.ones(n, dtype=np.float64), context, nuisance])
    projector = rank_revealing_projector(fixed)
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    annotations = np.ones((args.m, 1), dtype=np.float64)
    groups = tuple(f"block:{index % args.loo_groups}" for index in range(args.m))
    hashes = {
        "basis": canonical_sha256(
            {"basis": ["constant", "bounded_standardized_context"]}
        ),
        "fixed": array_sha256(fixed),
        "variant": canonical_sha256({"synthetic_variants": args.m, "seed": args.seed}),
    }
    reference = build_context_reference(
        genotype=genotype,
        basis=basis,
        projector=projector,
        annotations=annotations,
        component_index=components,
        basis_hash=hashes["basis"],
        fixed_effect_hash=hashes["fixed"],
        variant_hash=hashes["variant"],
        loo_groups=groups,
        genotype_scaling=GENOTYPE_SCALING,
        gram_method="exact",
        same_person_method="exact",
    )
    features = common_scale_features(genotype, basis, projector.projector)
    genetic_kernels = dense_genetic_kernels(features, annotations, components)
    residual_kernels = dense_residual_kernels(
        projector.projector, np.ones((n, 1), dtype=np.float64)
    )
    genetic_coefficients = omegas_to_coefficients(
        definition.omega[None, :, :], components
    )
    covariance = (
        np.einsum("a,aij->ij", genetic_coefficients, genetic_kernels, optimize=True)
        + definition.residual_variance * residual_kernels[0]
    )
    stochastic, minimum_eigenvalue = _sample_psd_covariance(rng, covariance)
    known_target = (
        stochastic + definition.context_mean_slope * context + 0.10 * nuisance[:, 0]
    )
    # A constant is in the fixed-effect space.  Choosing it from the realized
    # minimum guarantees both inverse-transform validity and a nonpositive raw
    # value for the deliberately invalid zero-shift logarithm.
    known_target += -0.8 - float(np.min(known_target))
    positive = np.square(1.0 + 0.5 * known_target)
    original = positive - 1.0
    projected = projector.projector @ known_target
    normalization_multiplier = projector.residual_rank / float(projected @ projected)
    return SyntheticFixture(
        definition=definition,
        genotype=genotype,
        basis=np.asarray(basis),
        fixed=np.asarray(fixed),
        projector=projector,
        annotations=annotations,
        components=components,
        loo_groups=groups,
        hashes=hashes,
        reference=reference,
        original_phenotype=np.asarray(original),
        known_target=np.asarray(known_target),
        transformations=_declared_transformations(),
        user_transforms={"known_box_cox_target": np.asarray(known_target)},
        minimum_generating_covariance_eigenvalue=minimum_eigenvalue,
        normalization_multiplier=float(normalization_multiplier),
    )


def _build_summary(
    fixture: SyntheticFixture,
    transformations: Sequence[PhenotypeTransformSpec],
    args: argparse.Namespace,
) -> Any:
    return build_context_transform_summary(
        genotype=fixture.genotype,
        basis=fixture.basis,
        original_phenotype=fixture.original_phenotype,
        transformations=transformations,
        projector=fixture.projector,
        annotations=fixture.annotations,
        component_index=fixture.components,
        residual_basis=np.ones((fixture.genotype.shape[0], 1), dtype=np.float64),
        residual_names=("identity",),
        basis_hash=fixture.hashes["basis"],
        fixed_effect_hash=fixture.hashes["fixed"],
        variant_hash=fixture.hashes["variant"],
        original_trait=f"synthetic_{fixture.definition.name}",
        user_transforms=fixture.user_transforms,
        loo_groups=fixture.loo_groups,
        genotype_scaling=GENOTYPE_SCALING,
        block_size=args.block_size,
        transform_tile_size=args.transform_tile_size,
    )


def _coordinates(
    coefficients: np.ndarray, components: ContextComponentIndex
) -> np.ndarray:
    omega = coefficients_to_omegas(
        np.asarray(coefficients[: len(components)], dtype=np.float64), components
    )[0]
    eigenvalues = np.linalg.eigvalsh(0.5 * (omega + omega.T))
    return np.asarray([omega[0, 1], abs(eigenvalues[0])], dtype=np.float64)


def _coordinate_trajectory(fit: Any, coordinate: int) -> TransformTrajectory:
    return build_transform_trajectory(
        fit,
        lambda row: np.asarray([_coordinates(row, fit.component_index)[coordinate]]),
        coordinate_names=(COORDINATE_NAMES[coordinate],),
    )


def _band_payload(band: Any) -> dict[str, Any]:
    return {
        "estimates": band.estimates[:, 0].tolist(),
        "standard_errors": band.standard_errors[:, 0].tolist(),
        "lower": band.lower[:, 0].tolist(),
        "upper": band.upper[:, 0].tolist(),
        "critical_value": float(band.critical_value),
        "confidence_level": float(band.confidence_level),
        "status": str(band.status),
    }


def _run_mechanism(
    fixture: SyntheticFixture,
    *,
    mechanism_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any, Any]:
    started = time.perf_counter()
    inference_specs = tuple(
        spec
        for spec in fixture.transformations
        if not spec.transform_id.startswith("invalid_")
    )
    summary = _build_summary(fixture, inference_specs, args)
    fit = fit_context_transform_scan(fixture.reference, summary)
    joint_trajectory = build_transform_trajectory(
        fit,
        lambda row: _coordinates(row, fit.component_index),
        coordinate_names=COORDINATE_NAMES,
    )
    joint_band = simultaneous_trajectory_bands(
        joint_trajectory,
        confidence_level=args.confidence_level,
        draws=args.multiplier_draws,
        seed=args.seed + 1000 * mechanism_index,
    )
    bands = tuple(
        select_simultaneous_band_coordinate(joint_band, index) for index in range(2)
    )
    classification = classify_scale_trajectory(
        bands[0],
        bands[1],
        amplification_margin=args.amplification_margin,
        heterogeneity_margin=args.heterogeneity_margin,
    )
    target_index = summary.transform_ids.index(TARGET_TRANSFORM_ID)
    user_index = summary.transform_ids.index(USER_TARGET_TRANSFORM_ID)
    normalized_target = project_normalize_phenotype(
        fixture.known_target, fixture.projector
    )
    recovered_target = (
        np.sqrt(fixture.projector.residual_rank)
        * (fixture.projector.projector @ fixture.known_target)
        / np.linalg.norm(fixture.projector.projector @ fixture.known_target)
    )
    target_discrepancy = scale_aware_max_discrepancy(
        normalized_target, recovered_target
    )
    target_user_coefficient_discrepancy = scale_aware_max_discrepancy(
        fit.coefficients[target_index], fit.coefficients[user_index]
    )
    target_user_loo_discrepancy = scale_aware_max_discrepancy(
        fit.loo_coefficients[:, target_index],
        fit.loo_coefficients[:, user_index],
    )
    pointwise = tuple(
        pointwise_normal_bands(_slice_trajectory(joint_trajectory, index))
        for index in range(2)
    )
    target_pointwise_contains_zero = [
        bool(lower[target_index, 0] <= 0.0 <= upper[target_index, 0])
        for lower, upper in pointwise
    ]
    records = {
        record.spec.transform_id: {
            "valid": bool(record.valid),
            "status": record.status,
        }
        for record in summary.transform_records
    }
    payload = {
        "name": fixture.definition.name,
        "label": fixture.definition.label,
        "n": int(fixture.genotype.shape[0]),
        "m": int(fixture.genotype.shape[1]),
        "known_box_cox_lambda": 0.5,
        "known_shift": 1.0,
        "generating_omega_before_phenotype_normalization": fixture.definition.omega.tolist(),
        "phenotype_normalization_multiplier": fixture.normalization_multiplier,
        "minimum_generating_covariance_eigenvalue": fixture.minimum_generating_covariance_eigenvalue,
        "transform_records": records,
        "valid_transform_ids": list(summary.transform_ids),
        "invalid_transform_ids": list(summary.invalid_transform_ids),
        "known_target_normalization_discrepancy": float(target_discrepancy),
        "target_vs_user_coefficient_discrepancy": float(
            target_user_coefficient_discrepancy
        ),
        "target_vs_user_loo_discrepancy": float(target_user_loo_discrepancy),
        "fit": {
            "rank": int(fit.solve.rank),
            "dimension": int(fit.coefficients.shape[1]),
            "condition_number": float(fit.solve.condition_number),
            "maximum_relative_residual": float(np.max(fit.solve.relative_residuals)),
            "jackknife_groups": int(len(fit.jackknife_groups)),
            "factorizations": int(fit.manifest["solve"]["factorizations"]),
            "joint_pseudo_value_shape": list(fit.pseudo_values.shape),
            "decode_passes": int(summary.decode_passes),
            "decoded_blocks": int(summary.decoded_blocks),
            "maximum_rhs_columns": int(summary.maximum_rhs_columns),
        },
        "coordinates": {
            COORDINATE_NAMES[index]: _band_payload(bands[index]) for index in range(2)
        },
        "target_pointwise_intervals_contain_zero": target_pointwise_contains_zero,
        "classification": asdict(classification),
        "expected_classification": fixture.definition.expected_classification,
        "classification_matches_expected": (
            classification.classification == fixture.definition.expected_classification
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "simultaneous_calibration": "one_joint_max_statistic_over_both_coordinates",
    }
    return payload, summary, fit


def _parity_validation(
    fixture: SyntheticFixture,
    batch_summary: Any,
    batch_fit: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid_specs = tuple(
        record.spec for record in batch_summary.transform_records if record.valid
    )
    genetic_rhs: list[np.ndarray] = []
    residual_rhs: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    loo_coefficients: list[np.ndarray] = []
    single_statuses: dict[str, str] = {}
    separate_elapsed = 0.0
    separate_peak_rss = 0
    for spec in valid_specs:
        started = time.perf_counter()
        summary = _build_summary(fixture, (spec,), args)
        fit = fit_context_transform_scan(fixture.reference, summary)
        separate_elapsed += time.perf_counter() - started
        separate_peak_rss = max(
            separate_peak_rss, summary.peak_rss_bytes, fit.peak_rss_bytes
        )
        genetic_rhs.append(summary.genetic_rhs[0])
        residual_rhs.append(summary.residual_rhs[0])
        coefficients.append(fit.coefficients[0])
        loo_coefficients.append(fit.loo_coefficients[:, 0, :])
        single_statuses[spec.transform_id] = next(
            record.status for record in summary.transform_records
        )
    invalid_only_raises: dict[str, bool] = {}
    invalid_statuses: dict[str, str] = {}
    identity = next(spec for spec in fixture.transformations if spec.kind == "identity")
    for invalid in (
        record.spec for record in batch_summary.transform_records if not record.valid
    ):
        sentinel = _build_summary(fixture, (identity, invalid), args)
        invalid_statuses[invalid.transform_id] = next(
            record.status
            for record in sentinel.transform_records
            if record.spec.transform_id == invalid.transform_id
        )
        try:
            _build_summary(fixture, (invalid,), args)
        except ValueError as exc:
            invalid_only_raises[
                invalid.transform_id
            ] = "No declared phenotype transformation is valid" in str(exc)
        else:
            invalid_only_raises[invalid.transform_id] = False
    separate_coefficients = np.stack(coefficients)
    separate_loo = np.stack(loo_coefficients, axis=1)
    batch_statuses = {
        record.spec.transform_id: record.status
        for record in batch_summary.transform_records
    }
    valid_status_parity = all(
        batch_statuses[name] == status for name, status in single_statuses.items()
    )
    invalid_status_parity = all(
        batch_statuses[name] == status for name, status in invalid_statuses.items()
    )
    discrepancies = {
        "genetic_rhs": float(
            scale_aware_max_discrepancy(
                batch_summary.genetic_rhs, np.stack(genetic_rhs)
            )
        ),
        "residual_rhs": float(
            scale_aware_max_discrepancy(
                batch_summary.residual_rhs, np.stack(residual_rhs)
            )
        ),
        "coefficients": float(
            scale_aware_max_discrepancy(batch_fit.coefficients, separate_coefficients)
        ),
        "loo_coefficients": float(
            scale_aware_max_discrepancy(batch_fit.loo_coefficients, separate_loo)
        ),
    }
    return {
        "valid_transform_count": len(valid_specs),
        "batched_decode_passes": int(batch_summary.decode_passes),
        "separate_decode_passes": len(valid_specs),
        "batched_factorizations": int(batch_fit.manifest["solve"]["factorizations"]),
        "separate_factorizations": int(
            len(valid_specs) * batch_fit.manifest["solve"]["factorizations"]
        ),
        "maximum_scale_aware_discrepancies": discrepancies,
        "valid_status_parity": valid_status_parity,
        "invalid_status_parity": invalid_status_parity,
        "invalid_statuses": invalid_statuses,
        "invalid_only_jobs_raise": invalid_only_raises,
        "separate_elapsed_seconds": float(separate_elapsed),
        "separate_peak_rss_bytes": int(separate_peak_rss),
        "gate_pass": bool(
            max(discrepancies.values()) < 5.0e-11
            and batch_summary.decode_passes == 1
            and valid_status_parity
            and invalid_status_parity
            and all(invalid_only_raises.values())
        ),
    }


def _summary_payload_bytes(summary: Any, fit: Any) -> int:
    arrays = (
        summary.genetic_rhs,
        summary.genetic_traces,
        summary.genetic_residual,
        summary.residual_rhs,
        summary.residual_traces,
        summary.residual_gram,
        summary.rhs_numerator_contributions,
        summary.trace_numerator_contributions,
        summary.genetic_residual_numerator_contributions,
        fit.coefficients,
        fit.loo_coefficients,
        fit.pseudo_values,
        fit.standard_errors,
    )
    return int(sum(np.asarray(value).nbytes for value in arrays))


def _benchmark_transformations(count: int) -> tuple[PhenotypeTransformSpec, ...]:
    lambdas = np.linspace(-0.75, 1.50, count)
    return tuple(
        PhenotypeTransformSpec(
            f"benchmark_{index:02d}",
            "box_cox",
            shift=1.0,
            box_cox_lambda=float(value),
        )
        for index, value in enumerate(lambdas)
    )


def _performance_validation(
    fixture: SyntheticFixture, args: argparse.Namespace
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for count in args.benchmark_l:
        specs = _benchmark_transformations(count)
        batch_times: list[float] = []
        separate_times: list[float] = []
        batch_peak = 0
        separate_peak = 0
        payload_bytes = 0
        maximum_rhs = 0
        batch_decode_passes = 0
        batch_decoded_blocks = 0
        separate_decode_passes = 0
        separate_decoded_blocks = 0
        for _ in range(args.benchmark_repeats):
            started = time.perf_counter()
            summary = _build_summary(fixture, specs, args)
            fit = fit_context_transform_scan(fixture.reference, summary)
            batch_times.append(time.perf_counter() - started)
            batch_peak = max(batch_peak, summary.peak_rss_bytes, fit.peak_rss_bytes)
            payload_bytes = max(payload_bytes, _summary_payload_bytes(summary, fit))
            maximum_rhs = max(maximum_rhs, summary.maximum_rhs_columns)
            batch_decode_passes = max(batch_decode_passes, summary.decode_passes)
            batch_decoded_blocks = max(batch_decoded_blocks, summary.decoded_blocks)
            del summary, fit
            gc.collect()
            started = time.perf_counter()
            for spec in specs:
                single_summary = _build_summary(fixture, (spec,), args)
                single_fit = fit_context_transform_scan(
                    fixture.reference, single_summary
                )
                separate_peak = max(
                    separate_peak,
                    single_summary.peak_rss_bytes,
                    single_fit.peak_rss_bytes,
                )
                separate_decode_passes += single_summary.decode_passes
                separate_decoded_blocks += single_summary.decoded_blocks
            separate_times.append(time.perf_counter() - started)
            del single_summary, single_fit
            gc.collect()
        batch_median = float(np.median(batch_times))
        separate_median = float(np.median(separate_times))
        records.append(
            {
                "l": int(count),
                "batch_elapsed_seconds_median": batch_median,
                "separate_elapsed_seconds_median": separate_median,
                "speedup_separate_over_batch": float(
                    separate_median / max(batch_median, np.finfo(float).tiny)
                ),
                "batch_peak_rss_bytes_absolute": int(batch_peak),
                "separate_peak_rss_bytes_absolute": int(separate_peak),
                "batch_stored_array_bytes": int(payload_bytes),
                "batch_decode_passes": int(batch_decode_passes),
                "batch_decoded_blocks": int(batch_decoded_blocks),
                "separate_decode_passes": int(
                    separate_decode_passes // args.benchmark_repeats
                ),
                "separate_decoded_blocks": int(
                    separate_decoded_blocks // args.benchmark_repeats
                ),
                "maximum_rhs_columns": int(maximum_rhs),
                "declared_rhs_bound": int(
                    fixture.basis.shape[1] * min(args.transform_tile_size, count)
                ),
            }
        )
    return records


def _trajectory_from_pseudo_values(
    pseudo_values: np.ndarray,
    transform_ids: tuple[str, ...],
    coordinate_names: tuple[str, ...],
) -> TransformTrajectory:
    pseudo = np.asarray(pseudo_values, dtype=np.float64)
    estimate = np.mean(pseudo, axis=0)
    groups = pseudo.shape[0]
    loo = (groups * estimate[None, :, :] - pseudo) / (groups - 1.0)
    centered = pseudo - estimate[None, :, :]
    standard_errors = np.sqrt(
        np.maximum(
            np.sum(centered * centered, axis=0) / (groups * (groups - 1.0)),
            0.0,
        )
    )
    return TransformTrajectory(
        transform_ids=transform_ids,
        coordinate_names=coordinate_names,
        estimates=estimate,
        loo_estimates=loo,
        pseudo_values=pseudo,
        standard_errors=standard_errors,
    )


def _slice_trajectory(
    trajectory: TransformTrajectory, coordinate: int
) -> TransformTrajectory:
    return TransformTrajectory(
        transform_ids=trajectory.transform_ids,
        coordinate_names=(trajectory.coordinate_names[coordinate],),
        estimates=trajectory.estimates[:, coordinate : coordinate + 1],
        loo_estimates=trajectory.loo_estimates[:, :, coordinate : coordinate + 1],
        pseudo_values=trajectory.pseudo_values[:, :, coordinate : coordinate + 1],
        standard_errors=trajectory.standard_errors[:, coordinate : coordinate + 1],
        declared_transform_ids=trajectory.declared_transform_ids,
        invalid_transform_ids=trajectory.invalid_transform_ids,
    )


def _inference_calibration(args: argparse.Namespace) -> dict[str, Any]:
    ids = ("lambda:0", "lambda:0.25", "lambda:0.5", "lambda:0.75", "lambda:1")
    truths = {
        "removable": np.column_stack(
            [
                np.asarray([0.21, 0.12, 0.0, 0.11, 0.20]),
                np.full(5, 0.02),
            ]
        ),
        "robust": np.column_stack(
            [np.asarray([0.04, 0.03, 0.02, 0.03, 0.04]), np.full(5, 0.24)]
        ),
        "weak": np.column_stack([np.full(5, 0.04), np.full(5, 0.03)]),
    }
    expected = {
        "removable": "removable_within_declared_family",
        "robust": "robust_within_declared_family",
        "weak": "indeterminate",
    }
    pseudo_scales = {"removable": 0.075, "robust": 0.075, "weak": 0.24}
    l_count = len(ids)
    transform_correlation = 0.72 ** np.abs(
        np.subtract.outer(np.arange(l_count), np.arange(l_count))
    )
    coordinate_correlation = np.asarray([[1.0, 0.25], [0.25, 1.0]])
    base_correlation = np.kron(transform_correlation, coordinate_correlation)
    results: dict[str, Any] = {}
    for scenario_index, (name, truth) in enumerate(truths.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, 7000, scenario_index])
        )
        covariance = pseudo_scales[name] ** 2 * base_correlation
        simultaneous_coverage = 0
        pointwise_family_coverage = 0
        selected_pointwise_coverage = 0
        naive_nonsignificance_removal = 0
        selected_counts = np.zeros(l_count, dtype=int)
        classifications: dict[str, int] = {}
        critical_values: list[float] = []
        for replicate in range(args.calibration_replicates):
            errors = rng.multivariate_normal(
                np.zeros(2 * l_count),
                covariance,
                size=args.calibration_groups,
            ).reshape(args.calibration_groups, l_count, 2)
            trajectory = _trajectory_from_pseudo_values(
                truth[None, :, :] + errors,
                ids,
                COORDINATE_NAMES,
            )
            joint = simultaneous_trajectory_bands(
                trajectory,
                confidence_level=args.confidence_level,
                draws=args.multiplier_draws,
                seed=args.seed + 100000 * scenario_index + replicate,
            )
            simultaneous_coverage += int(
                np.all((joint.lower <= truth) & (truth <= joint.upper))
            )
            critical_values.append(float(joint.critical_value))
            point_lower, point_upper = pointwise_normal_bands(
                trajectory, confidence_level=args.confidence_level
            )
            pointwise_family_coverage += int(
                np.all((point_lower <= truth) & (truth <= point_upper))
            )
            selected = int(np.argmin(np.linalg.norm(trajectory.estimates, axis=1)))
            selected_counts[selected] += 1
            selected_pointwise_coverage += int(
                np.all(
                    (point_lower[selected] <= truth[selected])
                    & (truth[selected] <= point_upper[selected])
                )
            )
            naive_nonsignificance_removal += int(
                np.any(
                    np.all(
                        (point_lower <= 0.0) & (0.0 <= point_upper),
                        axis=1,
                    )
                )
            )
            classification_joint = simultaneous_trajectory_bands(
                trajectory,
                confidence_level=args.confidence_level,
                draws=args.multiplier_draws,
                seed=args.seed + 200000 * scenario_index + replicate,
            )
            amp_band = select_simultaneous_band_coordinate(classification_joint, 0)
            het_band = select_simultaneous_band_coordinate(classification_joint, 1)
            classification = classify_scale_trajectory(
                amp_band,
                het_band,
                amplification_margin=args.amplification_margin,
                heterogeneity_margin=args.heterogeneity_margin,
            ).classification
            classifications[classification] = classifications.get(classification, 0) + 1
        denominator = float(args.calibration_replicates)
        results[name] = {
            "true_trajectory": truth.tolist(),
            "pseudo_value_standard_deviation": pseudo_scales[name],
            "simultaneous_family_coverage": simultaneous_coverage / denominator,
            "pointwise_family_coverage": pointwise_family_coverage / denominator,
            "selected_scale_pointwise_coordinate_coverage": (
                selected_pointwise_coverage / denominator
            ),
            "naive_exists_nonsignificant_scale_removal_rate": (
                naive_nonsignificance_removal / denominator
            ),
            "selected_transform_frequencies": (selected_counts / denominator).tolist(),
            "classification_rates": {
                key: value / denominator
                for key, value in sorted(classifications.items())
            },
            "expected_classification": expected[name],
            "expected_classification_rate": (
                classifications.get(expected[name], 0) / denominator
            ),
            "median_joint_critical_value": float(np.median(critical_values)),
        }
    return {
        "construction": (
            "independent Gaussian equal-group pseudo-values with AR(1) transform "
            "correlation and correlated mechanism coordinates"
        ),
        "replicates": int(args.calibration_replicates),
        "jackknife_groups": int(args.calibration_groups),
        "multiplier_draws": int(args.multiplier_draws),
        "confidence_level": float(args.confidence_level),
        "transform_ids": list(ids),
        "results": results,
    }


def _read_real_fixture(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        import pandas as pd
        from bed_reader import open_bed
    except ImportError as exc:
        raise RuntimeError("--real-traits requires pandas and bed-reader.") from exc
    required = [
        Path(f"{args.geno_prefix}.bed"),
        Path(f"{args.geno_prefix}.bim"),
        Path(f"{args.geno_prefix}.fam"),
        args.covariate_file,
        args.phenotype_root / "bmi.pheno",
        args.phenotype_root / "c_reactive_prot.pheno",
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
        args.covariate_file, ("FID", "IID", "sex", "age", *PC_COLUMNS)
    )
    merged = fam.merge(covariates, on=["FID", "IID"], validate="one_to_one")
    for trait in ("bmi", "c_reactive_prot"):
        values = read_table(
            args.phenotype_root / f"{trait}.pheno", ("FID", "IID", "pheno")
        ).rename(columns={"pheno": trait})
        merged = merged.merge(values, on=["FID", "IID"], validate="one_to_one")
    numeric = ["sex", "age", *PC_COLUMNS, "bmi", "c_reactive_prot"]
    matrix = merged[numeric].to_numpy(dtype=np.float64)
    keep = np.all(np.isfinite(matrix), axis=1) & np.all(matrix != -9.0, axis=1)
    intersection = merged.loc[keep].copy()
    if len(intersection) < args.real_n:
        raise ValueError(
            f"Only {len(intersection)} complete cases remain for --real-n={args.real_n}."
        )
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 9000]))
    selected_rows = np.sort(
        rng.choice(len(intersection), size=args.real_n, replace=False)
    )
    selected = (
        intersection.iloc[selected_rows].sort_values("bed_row").reset_index(drop=True)
    )
    bed = open_bed(Path(f"{args.geno_prefix}.bed"))
    candidate_count = min(int(bed.sid_count), args.real_m * 4)
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
    valid = (
        np.isfinite(means)
        & np.isfinite(variances)
        & (variances > np.finfo(np.float64).eps)
        & (missing_rate <= 0.05)
        & (maf >= 0.05)
    )
    retained = np.flatnonzero(valid)[: args.real_m]
    if retained.size != args.real_m:
        raise ValueError(
            f"Only {retained.size} deterministic candidate variants passed QC."
        )
    genotype = raw[:, retained].copy()
    chosen_means = means[retained]
    missing = np.isnan(genotype)
    if np.any(missing):
        genotype[missing] = np.broadcast_to(chosen_means[None, :], genotype.shape)[
            missing
        ]
    genotype = _standardize(genotype)
    context = _standardize(selected[["sex"]].to_numpy(dtype=np.float64))[:, 0]
    nuisance = _standardize(selected[["age", *PC_COLUMNS]].to_numpy(dtype=np.float64))
    basis = np.column_stack([np.ones(args.real_n), context])
    fixed = np.column_stack([np.ones(args.real_n), context, nuisance])
    projector = rank_revealing_projector(fixed)
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    annotations = np.ones((args.real_m, 1), dtype=np.float64)
    groups = tuple(f"block:{index % args.loo_groups}" for index in range(args.real_m))
    chosen_variant_indices = candidates[retained]
    hashes = {
        "basis": canonical_sha256({"basis": ["constant", "study_standardized_sex"]}),
        "fixed": array_sha256(fixed),
        "variant": canonical_sha256(
            {"selected_index_sha256": array_sha256(chosen_variant_indices)}
        ),
    }
    reference = build_context_reference(
        genotype=genotype,
        basis=basis,
        projector=projector,
        annotations=annotations,
        component_index=components,
        basis_hash=hashes["basis"],
        fixed_effect_hash=hashes["fixed"],
        variant_hash=hashes["variant"],
        loo_groups=groups,
        genotype_scaling="study_mean_imputed_and_unit_variance",
        gram_method="exact",
        same_person_method="exact",
    )
    sex_values, sex_counts = np.unique(
        selected["sex"].to_numpy(dtype=np.float64), return_counts=True
    )
    fixture = {
        "selected": selected,
        "genotype": genotype,
        "basis": basis,
        "fixed": fixed,
        "projector": projector,
        "components": components,
        "annotations": annotations,
        "groups": groups,
        "hashes": hashes,
        "reference": reference,
    }
    diagnostics = {
        "source_sample_count": int(bed.iid_count),
        "source_variant_count": int(bed.sid_count),
        "complete_case_intersection_count": int(len(intersection)),
        "selected_sample_count": int(args.real_n),
        "selected_variant_count": int(args.real_m),
        "candidate_variant_count": int(candidate_count),
        "context_category_counts": {
            str(value): int(count) for value, count in zip(sex_values, sex_counts)
        },
        "sample_selection": "seeded_without_replacement_from_complete_cases",
        "variant_selection": "evenly_spaced_candidates_then_missingness_maf_qc",
    }
    return fixture, diagnostics


def _real_trait_sanity(args: argparse.Namespace) -> dict[str, Any]:
    fixture, diagnostics = _read_real_fixture(args)
    # The inventoried phenotype columns are centered and can be negative.  A
    # fixed, predeclared shift keeps the small sanity grid positive without
    # encoding a sample minimum in the output.
    real_shift = 6.0
    specs = (
        PhenotypeTransformSpec("identity", "identity"),
        PhenotypeTransformSpec("log", "log", shift=real_shift),
        PhenotypeTransformSpec(
            "box_cox_0p25", "box_cox", shift=real_shift, box_cox_lambda=0.25
        ),
        PhenotypeTransformSpec(
            "box_cox_0p5", "box_cox", shift=real_shift, box_cox_lambda=0.5
        ),
        PhenotypeTransformSpec(
            "box_cox_0p75", "box_cox", shift=real_shift, box_cox_lambda=0.75
        ),
        PhenotypeTransformSpec(
            "box_cox_1", "box_cox", shift=real_shift, box_cox_lambda=1.0
        ),
    )
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    selected = fixture["selected"]
    for trait_index, trait in enumerate(("bmi", "c_reactive_prot")):
        summary = build_context_transform_summary(
            genotype=fixture["genotype"],
            basis=fixture["basis"],
            original_phenotype=selected[trait].to_numpy(dtype=np.float64),
            transformations=specs,
            projector=fixture["projector"],
            annotations=fixture["annotations"],
            component_index=fixture["components"],
            residual_basis=np.ones((args.real_n, 1), dtype=np.float64),
            residual_names=("identity",),
            basis_hash=fixture["hashes"]["basis"],
            fixed_effect_hash=fixture["hashes"]["fixed"],
            variant_hash=fixture["hashes"]["variant"],
            original_trait=trait,
            loo_groups=fixture["groups"],
            genotype_scaling="study_mean_imputed_and_unit_variance",
            block_size=args.block_size,
            transform_tile_size=args.transform_tile_size,
        )
        fit = fit_context_transform_scan(fixture["reference"], summary)
        trajectory = build_transform_trajectory(
            fit,
            lambda row: _coordinates(row, fit.component_index),
            coordinate_names=COORDINATE_NAMES,
        )
        joint_band = simultaneous_trajectory_bands(
            trajectory,
            confidence_level=args.confidence_level,
            draws=args.multiplier_draws,
            seed=args.seed + 10000 + 100 * trait_index,
        )
        bands = tuple(
            select_simultaneous_band_coordinate(joint_band, index) for index in range(2)
        )
        classification = classify_scale_trajectory(
            bands[0],
            bands[1],
            amplification_margin=args.amplification_margin,
            heterogeneity_margin=args.heterogeneity_margin,
        )
        records.append(
            {
                "trait": trait,
                "declared_positive_shift": real_shift,
                "transform_ids": list(summary.transform_ids),
                "invalid_transform_ids": list(summary.invalid_transform_ids),
                "rank": int(fit.solve.rank),
                "dimension": int(fit.coefficients.shape[1]),
                "condition_number": float(fit.solve.condition_number),
                "maximum_relative_residual": float(
                    np.max(fit.solve.relative_residuals)
                ),
                "jackknife_groups": int(len(fit.jackknife_groups)),
                "coordinates": {
                    COORDINATE_NAMES[index]: _band_payload(bands[index])
                    for index in range(2)
                },
                "classification": asdict(classification),
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
            "outputs": "aggregate_counts_fit_diagnostics_and_scale_trajectories_only",
        },
    }


def _plot_trajectories(
    mechanism_results: Sequence[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(
        len(mechanism_results),
        2,
        figsize=(12.0, 3.2 * len(mechanism_results)),
        squeeze=False,
    )
    margins = (args.amplification_margin, args.heterogeneity_margin)
    for row, result in enumerate(mechanism_results):
        labels = result["valid_transform_ids"]
        x = np.arange(len(labels))
        for column, coordinate in enumerate(COORDINATE_NAMES):
            axis = axes[row, column]
            values = result["coordinates"][coordinate]
            estimate = np.asarray(values["estimates"])
            lower = np.asarray(values["lower"])
            upper = np.asarray(values["upper"])
            axis.fill_between(x, lower, upper, color="#4C78A8", alpha=0.22)
            axis.plot(x, estimate, marker="o", color="#1F4E79", linewidth=1.7)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.axhline(
                margins[column], color="#B24A3A", linestyle="--", linewidth=0.9
            )
            axis.axhline(
                -margins[column], color="#B24A3A", linestyle="--", linewidth=0.9
            )
            axis.set_xticks(x, labels, rotation=35, ha="right")
            axis.set_ylabel(coordinate.replace("_", " "))
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_title(
                    f"{result['label']}\n{result['classification']['classification']}"
                )
            else:
                axis.set_title("95% simultaneous trajectory band")
    figure.suptitle("Synthetic transform-scan mechanism trajectories", y=1.01)
    figure.tight_layout()
    _save_figure(figure, output_dir, "trajectories")


def _plot_performance(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    l_values = np.asarray([record["l"] for record in records])
    batch = np.asarray([record["batch_elapsed_seconds_median"] for record in records])
    separate = np.asarray(
        [record["separate_elapsed_seconds_median"] for record in records]
    )
    rss = (
        np.asarray([record["batch_peak_rss_bytes_absolute"] for record in records])
        / 2**20
    )
    payload = (
        np.asarray([record["batch_stored_array_bytes"] for record in records]) / 2**20
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    axes[0].plot(l_values, batch, marker="o", label="batched")
    axes[0].plot(l_values, separate, marker="s", label="separate")
    axes[0].set_xlabel("valid transformations L")
    axes[0].set_ylabel("median wall time (seconds)")
    axes[0].set_title("One-pass batching versus repeated scans")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].plot(l_values, rss, marker="o", label="process peak RSS (absolute)")
    axes[1].plot(l_values, payload, marker="s", label="stored NumPy payload")
    axes[1].set_xlabel("valid transformations L")
    axes[1].set_ylabel("MiB")
    axes[1].set_title("Memory versus transformation count")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_dir, "performance")


def _plot_inference(calibration: Mapping[str, Any], output_dir: Path) -> None:
    names = list(calibration["results"])
    results = calibration["results"]
    x = np.arange(len(names))
    simultaneous = [results[name]["simultaneous_family_coverage"] for name in names]
    pointwise = [results[name]["pointwise_family_coverage"] for name in names]
    expected = [results[name]["expected_classification_rate"] for name in names]
    naive = [
        results[name]["naive_exists_nonsignificant_scale_removal_rate"]
        for name in names
    ]
    width = 0.20
    figure, axis = plt.subplots(figsize=(9.0, 4.5))
    axis.bar(x - 1.5 * width, simultaneous, width, label="simultaneous family coverage")
    axis.bar(x - 0.5 * width, pointwise, width, label="pointwise family coverage")
    axis.bar(x + 0.5 * width, expected, width, label="expected classification")
    axis.bar(x + 1.5 * width, naive, width, label="naive any-nonsignificant removal")
    axis.axhline(
        calibration["confidence_level"], color="black", linestyle="--", linewidth=0.9
    )
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Monte Carlo proportion")
    axis.set_title("Simultaneous coverage and scale-selection behavior")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save_figure(figure, output_dir, "inference")


def _plot_real_traits(real: Mapping[str, Any], output_dir: Path) -> None:
    records = real["records"]
    figure, axes = plt.subplots(
        len(records), 2, figsize=(11.5, 3.4 * len(records)), squeeze=False
    )
    for row, record in enumerate(records):
        labels = record["transform_ids"]
        x = np.arange(len(labels))
        for column, coordinate in enumerate(COORDINATE_NAMES):
            axis = axes[row, column]
            values = record["coordinates"][coordinate]
            estimate = np.asarray(values["estimates"])
            lower = np.asarray(values["lower"])
            upper = np.asarray(values["upper"])
            axis.fill_between(x, lower, upper, color="#72B7B2", alpha=0.25)
            axis.plot(x, estimate, marker="o", color="#157A6E")
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(x, labels, rotation=30, ha="right")
            axis.set_ylabel(coordinate.replace("_", " "))
            axis.grid(alpha=0.2)
            axis.set_title(f"{record['trait']} — aggregate sanity")
    figure.tight_layout()
    _save_figure(figure, output_dir, "real_traits")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    fixtures = tuple(
        _make_fixture(definition, mechanism_index=index, args=args)
        for index, definition in enumerate(_mechanisms())
    )
    mechanism_results: list[dict[str, Any]] = []
    summaries: list[Any] = []
    fits: list[Any] = []
    for index, fixture in enumerate(fixtures):
        result, summary, fit = _run_mechanism(fixture, mechanism_index=index, args=args)
        mechanism_results.append(result)
        summaries.append(summary)
        fits.append(fit)
    parity_summary = _build_summary(fixtures[0], fixtures[0].transformations, args)
    parity_fit = fit_context_transform_scan(fixtures[0].reference, parity_summary)
    parity = _parity_validation(fixtures[0], parity_summary, parity_fit, args)
    performance = _performance_validation(fixtures[0], args)
    inference = _inference_calibration(args)
    real = _real_trait_sanity(args) if args.real_traits else None

    exact_target_gate = all(
        result["known_target_normalization_discrepancy"] < 5.0e-13
        and result["target_vs_user_coefficient_discrepancy"] < 5.0e-11
        and result["target_vs_user_loo_discrepancy"] < 5.0e-11
        for result in mechanism_results
    )
    invalid_gate = bool(
        set(parity["invalid_statuses"]) == {"invalid_shift", "invalid_extreme"}
        and all(parity["invalid_only_jobs_raise"].values())
        and parity["invalid_status_parity"]
    )
    rhs_bound_gate = all(
        record["maximum_rhs_columns"] <= record["declared_rhs_bound"]
        for record in performance
    )
    coverage_values = [
        result["simultaneous_family_coverage"]
        for result in inference["results"].values()
    ]
    # This is a descriptive finite-Monte-Carlo guardrail, not an exact-size
    # acceptance region.  It catches grossly broken multiplier scaling.
    coverage_gate = all(0.82 <= value <= 1.0 for value in coverage_values)
    selection_gate = all(
        result["expected_classification_rate"] >= 0.70
        for result in inference["results"].values()
    )
    mechanism_gate = all(
        result["classification_matches_expected"] for result in mechanism_results
    )
    largest_performance_case = max(performance, key=lambda record: record["l"])
    speedup_gate = largest_performance_case["speedup_separate_over_batch"] > 1.0
    real_gate = real is None or all(
        not record["invalid_transform_ids"]
        and record["rank"] == record["dimension"]
        and np.isfinite(record["condition_number"])
        and record["maximum_relative_residual"] < 1.0e-10
        for record in real["records"]
    )
    gates = {
        "exact_known_scale_and_user_column": exact_target_gate,
        "invalid_transform_handling": invalid_gate,
        "batched_vs_separate_parity": bool(parity["gate_pass"]),
        "bounded_rhs_tiling": rhs_bound_gate,
        "measured_speedup_at_largest_grid": speedup_gate,
        "end_to_end_mechanism_classification": mechanism_gate,
        "simultaneous_coverage_guardrail": coverage_gate,
        "simultaneous_selection_guardrail": selection_gate,
        "requested_real_trait_numerical_sanity": real_gate,
    }
    payload = {
        "kind": "summit.context.transform_scan_validation",
        "seed": int(args.seed),
        "configuration": {
            "n": int(args.n),
            "m": int(args.m),
            "loo_groups": int(args.loo_groups),
            "block_size": int(args.block_size),
            "transform_tile_size": int(args.transform_tile_size),
            "benchmark_l": list(args.benchmark_l),
            "benchmark_repeats": int(args.benchmark_repeats),
            "amplification_margin": float(args.amplification_margin),
            "heterogeneity_margin": float(args.heterogeneity_margin),
        },
        "mechanisms": mechanism_results,
        "batched_vs_separate": parity,
        "performance": performance,
        "simultaneous_inference": inference,
        "real_trait_sanity": real,
        "gates": gates,
        "verdict": "pass" if all(gates.values()) else "review",
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_policy": {
            "row_level_data_written": False,
            "individual_identifiers_written": False,
            "variant_rows_written": False,
            "multiplier_draws_written": False,
        },
    }
    _plot_trajectories(mechanism_results, args.output_dir, args)
    _plot_performance(performance, args.output_dir)
    _plot_inference(inference, args.output_dir)
    if real is not None:
        _plot_real_traits(real, args.output_dir)
    return payload


def main() -> None:
    os.umask(0o077)
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    _prepare_output_directory(args.output_dir, include_real=args.real_traits)
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
    if payload["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
