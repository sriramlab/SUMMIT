#!/usr/bin/env python3
"""Validate compact summary-level learning of one context direction.

The default path uses deterministic dense synthetic fixtures.  It checks the
canonical symmetric-pair contractions against directly materialized kernels,
then exercises optimization and exact disjoint-variant-block two-fold
cross-fitting.  Outputs contain aggregate diagnostics only; no sample or
variant rows are written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summit.context import (
    ContextRankError,
    build_context_direction_crossfit_contractions,
    build_direction_reference_contractions,
    build_direction_trait_contractions,
    canonical_sha256,
    combine_context_direction_contractions,
    crossfit_context_direction,
    direction_pair_weights,
    evaluate_context_direction,
    normalize_context_direction,
    optimize_context_direction,
    project_normalize_phenotype,
    rank_revealing_projector,
)


OUTPUT_STEM = "07b_context_direction_validation"
DEFAULT_GENOTYPE_PREFIX = Path(
    "/home/bronsonj/UKBB/ldscores/refsample_h2_sensitivity_20260813/"
    "onekg_matched_unrelated_20260813/eur_matching/"
    "UKB_EUR_300k.seed20260813.n5000.common"
)
DEFAULT_PHENOTYPE_ROOT = Path("/home/bronsonj/UKBB/asha/phens")
PC_COLUMNS = tuple(f"f.22009.0.{index}" for index in range(1, 6))


@dataclass(frozen=True)
class SyntheticScenario:
    """One declared simulation mechanism."""

    name: str
    label: str
    context_correlation: float
    context_pc_correlation: float
    directions: tuple[tuple[float, ...], ...]
    interaction_sd: float
    mismatch: bool = False


@dataclass(frozen=True)
class SyntheticFixture:
    """Individual-level fixture retained only inside a validation call."""

    scenario: SyntheticScenario
    reference_genotype: np.ndarray
    study_genotype: np.ndarray
    reference_context: np.ndarray
    study_context: np.ndarray
    reference_fixed_effects: np.ndarray
    study_fixed_effects: np.ndarray
    reference_projector: Any
    study_projector: Any
    phenotype: np.ndarray
    block_ids: np.ndarray
    variant_weights: np.ndarray
    variant_hash: str
    environment_names: tuple[str, ...]
    context_metric: np.ndarray
    true_directions: tuple[np.ndarray, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--reference-n", type=int, default=320)
    parser.add_argument("--m", type=int, default=144)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--benchmark-evaluations", type=int, default=10_000)
    parser.add_argument("--null-replicates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--real-traits",
        action="store_true",
        help="Run an optional aggregate-only real-genotype/trait sanity check.",
    )
    parser.add_argument("--geno-prefix", type=Path, default=DEFAULT_GENOTYPE_PREFIX)
    parser.add_argument("--phenotype-root", type=Path, default=DEFAULT_PHENOTYPE_ROOT)
    parser.add_argument(
        "--covariate-file",
        type=Path,
        default=DEFAULT_PHENOTYPE_ROOT / "testosterone.covar",
    )
    parser.add_argument("--real-n", type=int, default=192)
    parser.add_argument("--real-m", type=int, default=96)
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n < 96 or args.reference_n < 96:
        parser.error("--n and --reference-n must be at least 96")
    if args.m < 48:
        parser.error("--m must be at least 48")
    if args.blocks < 12 or args.m % args.blocks:
        parser.error("--blocks must be at least 12 and divide --m exactly")
    if args.blocks % 2:
        parser.error("--blocks must be even for balanced two-fold cross-fitting")
    if args.benchmark_evaluations < 100:
        parser.error("--benchmark-evaluations must be at least 100")
    if args.null_replicates < 20:
        parser.error("--null-replicates must be at least 20")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.real_traits:
        if args.real_n < 96 or args.real_m < 48:
            parser.error("--real-n/--real-m must be at least 96/48")
        if args.real_m % args.blocks:
            parser.error("--real-m must be divisible by --blocks")


def _output_paths(output_dir: Path, *, include_real: bool) -> tuple[Path, ...]:
    labels = ["directions", "crossfit", "performance"]
    if include_real:
        labels.append("real_traits")
    names = [f"{OUTPUT_STEM}.json"]
    for label in labels:
        names.extend([f"{OUTPUT_STEM}_{label}.png", f"{OUTPUT_STEM}_{label}.pdf"])
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


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("Encountered a degenerate simulated column.")
    return np.asarray(centered / scale, dtype=np.float64)


def _projector(fixed_effects: np.ndarray) -> np.ndarray:
    design = np.asarray(fixed_effects, dtype=np.float64)
    return np.eye(design.shape[0], dtype=np.float64) - design @ np.linalg.pinv(design)


def _pair_indices(dimension: int) -> tuple[tuple[int, int], ...]:
    return tuple((index, index) for index in range(dimension)) + tuple(
        (left, right)
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )


def _pair_weights(direction: np.ndarray) -> np.ndarray:
    omega = np.asarray(direction, dtype=np.float64)
    return np.asarray(
        [omega[left] * omega[right] for left, right in _pair_indices(omega.size)],
        dtype=np.float64,
    )


def _normalize_direction(
    direction: np.ndarray, context_metric: np.ndarray
) -> np.ndarray:
    omega = np.asarray(direction, dtype=np.float64)
    metric = np.asarray(context_metric, dtype=np.float64)
    norm_squared = float(omega @ metric @ omega)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("Direction has non-positive norm under the context metric.")
    normalized = omega / math.sqrt(norm_squared)
    pivot = int(np.argmax(np.abs(normalized)))
    if normalized[pivot] < 0.0:
        normalized = -normalized
    return np.asarray(normalized, dtype=np.float64)


def _metric_alignment(
    learned: np.ndarray, truth: np.ndarray, context_metric: np.ndarray
) -> float:
    left = _normalize_direction(learned, context_metric)
    right = _normalize_direction(truth, context_metric)
    return float(abs(left @ context_metric @ right))


def _canonical_dense_kernels(
    genotype: np.ndarray,
    context: np.ndarray,
    projector: Any,
    variant_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray, tuple[np.ndarray, ...]]:
    """Materialize G, symmetric interaction pairs, P, and residual pairs."""

    genotype_array = np.asarray(genotype, dtype=np.float64)
    context_array = np.asarray(context, dtype=np.float64)
    projection = np.asarray(
        getattr(projector, "projector", projector), dtype=np.float64
    )
    n, m = genotype_array.shape
    if context_array.shape[0] != n or projection.shape != (n, n):
        raise ValueError("Incompatible genotype, context, and projector shapes.")

    weights = (
        np.ones(m, dtype=np.float64)
        if variant_weights is None
        else np.asarray(variant_weights, dtype=np.float64)
    )
    if weights.shape != (m,) or np.any(weights < 0.0):
        raise ValueError("variant_weights must be non-negative with shape (M,).")
    variant_mass = float(np.sum(weights))
    if variant_mass <= 0.0:
        raise ValueError("variant_weights must have positive total mass.")
    additive_features = projection @ genotype_array
    environment_features = tuple(
        projection @ (context_array[:, index, None] * genotype_array)
        for index in range(context_array.shape[1])
    )
    additive = (
        (additive_features * weights[None, :]) @ additive_features.T / variant_mass
    )
    interaction_pairs: list[np.ndarray] = []
    residual_pairs: list[np.ndarray] = []
    for left, right in _pair_indices(context_array.shape[1]):
        if left == right:
            interaction = (
                (environment_features[left] * weights[None, :])
                @ environment_features[right].T
            ) / variant_mass
            residual_diagonal = context_array[:, left] * context_array[:, right]
        else:
            interaction = (
                (environment_features[left] * weights[None, :])
                @ environment_features[right].T
                + (environment_features[right] * weights[None, :])
                @ environment_features[left].T
            ) / variant_mass
            residual_diagonal = 2.0 * context_array[:, left] * context_array[:, right]
        interaction_pairs.append(np.asarray(interaction, dtype=np.float64))
        residual_pairs.append(
            np.asarray(
                projection @ np.diag(residual_diagonal) @ projection,
                dtype=np.float64,
            )
        )
    return (
        np.asarray(additive, dtype=np.float64),
        tuple(interaction_pairs),
        projection,
        tuple(residual_pairs),
    )


def _direct_reduced_kernels(
    genotype: np.ndarray,
    context: np.ndarray,
    projector: Any,
    direction: np.ndarray,
    variant_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Construct [G, I(omega), P, D(omega)] without pair contractions."""

    genotype_array = np.asarray(genotype, dtype=np.float64)
    context_array = np.asarray(context, dtype=np.float64)
    projection = np.asarray(
        getattr(projector, "projector", projector), dtype=np.float64
    )
    omega = np.asarray(direction, dtype=np.float64)
    m = genotype_array.shape[1]
    weights = (
        np.ones(m, dtype=np.float64)
        if variant_weights is None
        else np.asarray(variant_weights, dtype=np.float64)
    )
    variant_mass = float(np.sum(weights))
    if weights.shape != (m,) or np.any(weights < 0.0) or variant_mass <= 0.0:
        raise ValueError("Invalid variant weights.")
    additive_features = projection @ genotype_array
    environment = context_array @ omega
    interaction_features = projection @ (environment[:, None] * genotype_array)
    return (
        (additive_features * weights[None, :]) @ additive_features.T / variant_mass,
        (interaction_features * weights[None, :])
        @ interaction_features.T
        / variant_mass,
        projection,
        projection @ np.diag(environment * environment) @ projection,
    )


def _kernel_equations(
    kernels: Sequence[np.ndarray], phenotype: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = tuple(np.asarray(kernel, dtype=np.float64) for kernel in kernels)
    y = np.asarray(phenotype, dtype=np.float64)
    gram = np.asarray(
        [[np.sum(left * right) for right in matrices] for left in matrices],
        dtype=np.float64,
    )
    rhs = np.asarray([y @ kernel @ y for kernel in matrices], dtype=np.float64)
    traces = np.asarray([np.trace(kernel) for kernel in matrices], dtype=np.float64)
    return gram, rhs, traces


def _scenarios() -> tuple[SyntheticScenario, ...]:
    return (
        SyntheticScenario(
            name="one_true_direction",
            label="one true direction",
            context_correlation=0.0,
            context_pc_correlation=0.0,
            directions=((1.0, 0.35),),
            interaction_sd=0.55,
        ),
        SyntheticScenario(
            name="competing_directions",
            label="two competing directions",
            context_correlation=0.0,
            context_pc_correlation=0.0,
            directions=((1.0, 0.20), (-0.15, 1.0)),
            interaction_sd=0.48,
        ),
        SyntheticScenario(
            name="correlated_context",
            label="correlated Z",
            context_correlation=0.72,
            context_pc_correlation=0.35,
            directions=((0.35, 1.0),),
            interaction_sd=0.55,
        ),
        SyntheticScenario(
            name="weak_signal",
            label="weak signal",
            context_correlation=0.35,
            context_pc_correlation=0.15,
            directions=((1.0, -0.40),),
            interaction_sd=0.10,
        ),
        SyntheticScenario(
            name="null_signal",
            label="null GxE",
            context_correlation=0.35,
            context_pc_correlation=0.15,
            directions=((1.0, 0.0),),
            interaction_sd=0.0,
        ),
        SyntheticScenario(
            name="reference_mismatch",
            label="reference mismatch",
            context_correlation=0.65,
            context_pc_correlation=0.30,
            directions=((1.0, -0.25),),
            interaction_sd=0.55,
            mismatch=True,
        ),
    )


def _simulate_context(
    rng: np.random.Generator,
    n_samples: int,
    *,
    correlation: float,
    pc_correlation: float,
    skew: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    latent = rng.normal(size=(n_samples, 3))
    pc = _standardize(latent[:, 2])
    first = latent[:, 0]
    second = (
        correlation * latent[:, 0]
        + math.sqrt(max(1.0 - correlation * correlation, 0.0)) * latent[:, 1]
    )
    if skew:
        first = np.sign(first) * np.abs(first) ** 1.45 + 0.35 * (latent[:, 1] ** 2)
        second = np.tanh(1.25 * second) + 0.20 * latent[:, 0] ** 2
    mixing_scale = math.sqrt(max(1.0 - pc_correlation * pc_correlation, 0.0))
    first = mixing_scale * first + pc_correlation * pc
    second = mixing_scale * second - 0.65 * pc_correlation * pc
    context = _standardize(np.column_stack([first, second]))
    return np.asarray(context, dtype=np.float64), np.asarray(pc, dtype=np.float64)


def _simulate_genotype(
    rng: np.random.Generator,
    n_samples: int,
    n_variants: int,
    pc: np.ndarray,
    context: np.ndarray,
    *,
    ld_correlation: float,
    population_loading: float,
) -> np.ndarray:
    innovations = rng.normal(size=(n_samples, n_variants))
    genotype = np.empty_like(innovations)
    genotype[:, 0] = innovations[:, 0]
    innovation_scale = math.sqrt(max(1.0 - ld_correlation**2, 0.0))
    for index in range(1, n_variants):
        genotype[:, index] = (
            ld_correlation * genotype[:, index - 1]
            + innovation_scale * innovations[:, index]
        )
    loadings = rng.normal(size=n_variants)
    context_loadings = rng.normal(size=n_variants)
    genotype += population_loading * pc[:, None] * loadings[None, :]
    genotype += 0.35 * population_loading * context[:, [0]] * context_loadings[None, :]
    return _standardize(genotype)


def _fixed_effect_design(context: np.ndarray, pc: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(context.shape[0], dtype=np.float64),
            context,
            pc,
            pc[:, None] * context,
        ]
    )


def _simulate_fixture(
    scenario: SyntheticScenario,
    *,
    n_study: int,
    n_reference: int,
    n_variants: int,
    n_blocks: int,
    seed: int,
    same_person_reference: bool = False,
) -> SyntheticFixture:
    # Each mechanism has its own RNG stream, making scenario order irrelevant.
    rng = np.random.default_rng(seed)
    study_context, study_pc = _simulate_context(
        rng,
        n_study,
        correlation=scenario.context_correlation,
        pc_correlation=scenario.context_pc_correlation,
    )
    study_genotype = _simulate_genotype(
        rng,
        n_study,
        n_variants,
        study_pc,
        study_context,
        ld_correlation=0.30,
        population_loading=0.20,
    )
    if same_person_reference:
        if n_reference != n_study:
            raise ValueError("Same-person oracle requires equal reference/study N.")
        reference_context = study_context.copy()
        reference_pc = study_pc.copy()
        reference_genotype = study_genotype.copy()
    else:
        mismatch = scenario.mismatch
        reference_context, reference_pc = _simulate_context(
            rng,
            n_reference,
            correlation=(-0.20 if mismatch else scenario.context_correlation),
            pc_correlation=(-0.10 if mismatch else scenario.context_pc_correlation),
            skew=mismatch,
        )
        reference_genotype = _simulate_genotype(
            rng,
            n_reference,
            n_variants,
            reference_pc,
            reference_context,
            ld_correlation=(0.68 if mismatch else 0.30),
            population_loading=(0.48 if mismatch else 0.20),
        )

    study_fixed = _fixed_effect_design(study_context, study_pc)
    reference_fixed = _fixed_effect_design(reference_context, reference_pc)
    study_projection = _projector(study_fixed)
    reference_projection = _projector(reference_fixed)
    context_metric = (
        reference_context.T @ reference_context / float(reference_context.shape[0])
    )
    true_directions = tuple(
        _normalize_direction(np.asarray(direction), context_metric)
        for direction in scenario.directions
    )

    additive_effect = rng.normal(size=n_variants)
    additive_signal = (
        0.22 * study_genotype @ additive_effect / math.sqrt(float(n_variants))
    )
    interaction_signal = np.zeros(n_study, dtype=np.float64)
    variant_indices = np.arange(n_variants)
    for direction_index, direction in enumerate(true_directions):
        selected = variant_indices % len(true_directions) == direction_index
        interaction_effect = rng.normal(size=int(np.sum(selected)))
        score = (
            study_genotype[:, selected]
            @ interaction_effect
            / math.sqrt(float(np.sum(selected)))
        )
        interaction_signal += (
            scenario.interaction_sd
            * (study_context @ direction)
            * score
            / math.sqrt(float(len(true_directions)))
        )
    residual_scale = np.exp(0.08 * study_context[:, 0])
    phenotype = (
        additive_signal
        + interaction_signal
        + 0.42 * residual_scale * rng.normal(size=n_study)
    )
    variant_weights = np.linspace(0.70, 1.30, n_variants, dtype=np.float64)
    block_ids = np.repeat(np.arange(n_blocks, dtype=np.int64), n_variants // n_blocks)

    # Public builders accept the richer ProjectorResult, when available.  The
    # ndarray fallback keeps the independent dense oracle decoupled from it.
    reference_projector: Any = rank_revealing_projector(reference_fixed)
    study_projector: Any = rank_revealing_projector(study_fixed)
    variant_hash = canonical_sha256(
        {
            "fixture": "07b_synthetic",
            "scenario": scenario.name,
            "seed": seed,
            "n_variants": n_variants,
        }
    )
    return SyntheticFixture(
        scenario=scenario,
        reference_genotype=reference_genotype,
        study_genotype=study_genotype,
        reference_context=reference_context,
        study_context=study_context,
        reference_fixed_effects=reference_fixed,
        study_fixed_effects=study_fixed,
        reference_projector=reference_projector,
        study_projector=study_projector,
        phenotype=np.asarray(phenotype, dtype=np.float64),
        block_ids=block_ids,
        variant_weights=variant_weights,
        variant_hash=variant_hash,
        environment_names=("environment_1", "environment_2"),
        context_metric=np.asarray(context_metric, dtype=np.float64),
        true_directions=true_directions,
    )


def _compact_array_summary(value: Any) -> dict[str, Any]:
    arrays: list[np.ndarray] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        identifier = id(item)
        if identifier in seen:
            return
        seen.add(identifier)
        if isinstance(item, np.ndarray):
            arrays.append(item)
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    shapes = [list(array.shape) for array in arrays]
    return {
        "array_count": len(arrays),
        "total_array_bytes": int(sum(array.nbytes for array in arrays)),
        "maximum_axis": int(
            max((max(array.shape, default=0) for array in arrays), default=0)
        ),
        "shapes": shapes,
    }


def _relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    scale = max(float(np.max(np.abs(right))), 1.0)
    return float(np.max(np.abs(left - right)) / scale)


def _build_public_contractions(
    fixture: SyntheticFixture,
    *,
    variant_mask: np.ndarray | None = None,
) -> tuple[Any, Any, Any]:
    shared = {
        "context_metric": fixture.context_metric,
        "environment_names": fixture.environment_names,
        "variant_hash": fixture.variant_hash,
        "genotype_scaling": "pre_scaled_input",
        "variant_weights": fixture.variant_weights,
        "variant_mask": variant_mask,
    }
    reference = build_direction_reference_contractions(
        fixture.reference_genotype,
        fixture.reference_context,
        fixture.reference_projector,
        **shared,
    )
    trait = build_direction_trait_contractions(
        fixture.study_genotype,
        fixture.study_context,
        fixture.study_projector,
        fixture.phenotype,
        **shared,
    )
    return (
        reference,
        trait,
        combine_context_direction_contractions(reference, trait),
    )


def _dense_parity_validation(
    *, n_samples: int, n_variants: int, n_blocks: int, seed: int
) -> dict[str, Any]:
    scenario = _scenarios()[0]
    fixture = _simulate_fixture(
        scenario,
        n_study=n_samples,
        n_reference=n_samples,
        n_variants=n_variants,
        n_blocks=n_blocks,
        seed=seed,
        same_person_reference=True,
    )
    _, _, contractions = _build_public_contractions(fixture)
    direction = normalize_context_direction(
        np.asarray([0.63, -0.41], dtype=np.float64), fixture.context_metric
    )
    pair_weights_public = direction_pair_weights(direction)
    pair_weights_oracle = _pair_weights(direction)

    additive, interaction_pairs, residual, residual_pairs = _canonical_dense_kernels(
        fixture.study_genotype,
        fixture.study_context,
        fixture.study_projector,
        fixture.variant_weights,
    )
    contracted = (
        additive,
        sum(
            weight * kernel
            for weight, kernel in zip(
                pair_weights_oracle, interaction_pairs, strict=True
            )
        ),
        residual,
        sum(
            weight * kernel
            for weight, kernel in zip(pair_weights_oracle, residual_pairs, strict=True)
        ),
    )
    direct = _direct_reduced_kernels(
        fixture.study_genotype,
        fixture.study_context,
        fixture.study_projector,
        direction,
        fixture.variant_weights,
    )
    phenotype = project_normalize_phenotype(fixture.phenotype, fixture.study_projector)
    dense_matrix, dense_rhs, dense_traces = _kernel_equations(direct, phenotype)
    dense_coefficients = np.linalg.solve(dense_matrix, dense_rhs)
    nuisance = np.asarray([0, 2, 3], dtype=np.int64)
    dense_objectives = {
        "interaction_coefficient": float(dense_coefficients[1]),
        "interaction_trace_contribution": float(
            dense_coefficients[1]
            * dense_traces[1]
            / fixture.study_projector.residual_rank
        ),
        "he_moment_gain": float(
            dense_rhs @ dense_coefficients
            - dense_rhs[nuisance]
            @ np.linalg.solve(
                dense_matrix[np.ix_(nuisance, nuisance)], dense_rhs[nuisance]
            )
        ),
    }
    public_evaluations = {
        objective: evaluate_context_direction(
            contractions, direction, objective=objective
        )
        for objective in dense_objectives
    }
    baseline = public_evaluations["he_moment_gain"]
    direct_grid = _metric_circle(fixture.context_metric, 181)
    direct_grid_values = np.full(direct_grid.shape[0], np.nan, dtype=np.float64)
    for index, candidate in enumerate(direct_grid):
        candidate_kernels = _direct_reduced_kernels(
            fixture.study_genotype,
            fixture.study_context,
            fixture.study_projector,
            candidate,
            fixture.variant_weights,
        )
        matrix, rhs, _ = _kernel_equations(candidate_kernels, phenotype)
        if np.linalg.matrix_rank(matrix) != 4:
            continue
        coefficients = np.linalg.solve(matrix, rhs)
        nuisance_matrix = matrix[np.ix_(nuisance, nuisance)]
        if np.linalg.matrix_rank(nuisance_matrix) != 3:
            continue
        direct_grid_values[index] = float(
            rhs @ coefficients
            - rhs[nuisance] @ np.linalg.solve(nuisance_matrix, rhs[nuisance])
        )
    if not np.any(np.isfinite(direct_grid_values)):
        raise RuntimeError("Direct dense direction grid had no identifiable point.")
    dense_best_index = int(np.nanargmax(direct_grid_values))
    public_optimization = optimize_context_direction(
        contractions,
        objective="he_moment_gain",
        validation_grid_size=181,
        maxiter=250,
    )
    direct_grid_best = float(direct_grid_values[dense_best_index])
    optimization_gap = float(public_optimization.objective_value - direct_grid_best)
    optimization_alignment = _metric_alignment(
        public_optimization.direction,
        direct_grid[dense_best_index],
        fixture.context_metric,
    )
    errors = {
        "pair_weights": _relative_error(pair_weights_public, pair_weights_oracle),
        "interaction_kernel_contraction": _relative_error(contracted[1], direct[1]),
        "residual_kernel_contraction": _relative_error(contracted[3], direct[3]),
        "normal_matrix": _relative_error(baseline.matrix, dense_matrix),
        "rhs": _relative_error(baseline.rhs, dense_rhs),
        "traces": _relative_error(baseline.traces, dense_traces),
        "coefficients": _relative_error(baseline.coefficients, dense_coefficients),
    }
    objective_errors = {
        objective: abs(float(public_evaluations[objective].objective_value) - expected)
        / max(abs(expected), 1.0)
        for objective, expected in dense_objectives.items()
    }
    maximum_error = max((*errors.values(), *objective_errors.values()))
    optimization_tolerance = 5.0e-4 * max(abs(direct_grid_best), 1.0)
    return {
        "gate": bool(
            maximum_error < 1.0e-9
            and baseline.rank == 4
            and optimization_gap >= -optimization_tolerance
            and optimization_alignment > 0.995
        ),
        "maximum_relative_error": maximum_error,
        "errors": errors,
        "objective_errors": objective_errors,
        "rank": int(baseline.rank),
        "condition_number": float(baseline.condition_number),
        "direct_individual_level_grid_optimization": {
            "grid_size": int(direct_grid.shape[0]),
            "dense_grid_best_objective": direct_grid_best,
            "compact_optimizer_objective": float(public_optimization.objective_value),
            "compact_minus_dense_grid_gap": optimization_gap,
            "metric_alignment": optimization_alignment,
            "objective_tolerance": optimization_tolerance,
        },
        "dimensions": {
            "n": n_samples,
            "m": n_variants,
            "l": fixture.study_context.shape[1],
            "pair_count": len(interaction_pairs),
            "reduced_kernel_count": 4,
        },
        "pair_convention": (
            "diagonal-first vech(omega omega^T); off-diagonal factors are "
            "carried by the symmetric genetic and residual-product kernels"
        ),
    }


def _metric_circle(context_metric: np.ndarray, count: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(context_metric)
    inverse_sqrt = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.T
    angles = np.linspace(0.0, np.pi, count, endpoint=False)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.asarray(unit @ inverse_sqrt, dtype=np.float64)


def _scenario_validation(
    fixture: SyntheticFixture, *, objective: str = "he_moment_gain"
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    build_start = time.perf_counter()
    _, _, contractions = _build_public_contractions(fixture)
    build_seconds = time.perf_counter() - build_start
    optimize_start = time.perf_counter()
    optimized = optimize_context_direction(
        contractions,
        objective=objective,
        validation_grid_size=181,
        maxiter=250,
    )
    optimization_seconds = time.perf_counter() - optimize_start
    learned = np.asarray(optimized.direction, dtype=np.float64)
    generating_direction_defined = fixture.scenario.interaction_sd > 0.0
    alignments = [
        _metric_alignment(learned, truth, fixture.context_metric)
        for truth in fixture.true_directions
    ]
    truth_evaluations = [
        evaluate_context_direction(contractions, truth, objective=objective)
        for truth in fixture.true_directions
    ]

    grid = _metric_circle(fixture.context_metric, 181)
    grid_values = np.full(grid.shape[0], np.nan, dtype=np.float64)
    for index, direction in enumerate(grid):
        try:
            grid_values[index] = evaluate_context_direction(
                contractions, direction, objective=objective
            ).objective_value
        except ContextRankError:
            continue
    study_metric = (
        fixture.study_context.T
        @ fixture.study_context
        / float(fixture.study_context.shape[0])
    )
    metric_discrepancy = float(
        np.linalg.norm(study_metric - fixture.context_metric, ord="fro")
        / max(np.linalg.norm(fixture.context_metric, ord="fro"), 1.0e-12)
    )
    summary = {
        "name": fixture.scenario.name,
        "label": fixture.scenario.label,
        "objective": objective,
        "training_objective_is_unbiased": False,
        "learned_direction": learned,
        "true_directions": fixture.true_directions,
        "generating_direction_defined": generating_direction_defined,
        "alignment_by_truth": alignments,
        "maximum_alignment": (
            max(alignments) if generating_direction_defined else None
        ),
        "optimized_training_objective": float(optimized.evaluation.objective_value),
        "true_direction_training_objectives": [
            float(evaluation.objective_value) for evaluation in truth_evaluations
        ],
        "interaction_coefficient_at_optimum": float(
            optimized.evaluation.coefficients[1]
        ),
        "rank_at_optimum": int(optimized.evaluation.rank),
        "condition_number_at_optimum": float(optimized.evaluation.condition_number),
        "optimizer_status": str(optimized.status),
        "optimizer_start_count": int(optimized.start_directions.shape[0]),
        "optimizer_converged_candidate_count": int(optimized.converged_candidates),
        "optimizer_failed_evaluation_count": int(optimized.failed_candidates),
        "optimizer_evaluation_count": int(optimized.evaluations),
        "optimizer_grid_validation_gap": optimized.grid_validation_gap,
        "optimizer_nonunique": bool(optimized.nonunique),
        "optimizer_objective_gap_to_second_direction": optimized.objective_gap,
        "optimizer_distinct_candidate_count": int(
            optimized.distinct_candidate_directions.shape[0]
        ),
        "optimizer_near_optimal_direction_count": int(
            optimized.near_optimal_directions.shape[0]
        ),
        "reference_study_metric_relative_discrepancy": metric_discrepancy,
        "context_correlation_study": float(np.corrcoef(fixture.study_context.T)[0, 1]),
        "reference_mismatch_declared": fixture.scenario.mismatch,
        "compact_contractions": _compact_array_summary(contractions),
        "timing_seconds": {
            "build": build_seconds,
            "optimize": optimization_seconds,
        },
    }
    plot_data = {
        "directions": grid,
        "objective_values": grid_values,
        "learned": learned,
    }
    return summary, plot_data


def _build_crossfit_contractions(fixture: SyntheticFixture) -> Any:
    return build_context_direction_crossfit_contractions(
        fixture.reference_genotype,
        fixture.reference_context,
        fixture.reference_projector,
        fixture.study_genotype,
        fixture.study_context,
        fixture.study_projector,
        fixture.phenotype,
        fixture.block_ids,
        context_metric=fixture.context_metric,
        environment_names=fixture.environment_names,
        variant_hash=fixture.variant_hash,
        genotype_scaling="pre_scaled_input",
        variant_weights=fixture.variant_weights,
    )


def _crossfit_validation(
    fixture: SyntheticFixture, *, objective: str = "he_moment_gain"
) -> tuple[dict[str, Any], Any, Any]:
    build_start = time.perf_counter()
    contractions = _build_crossfit_contractions(fixture)
    build_seconds = time.perf_counter() - build_start
    fit_start = time.perf_counter()
    result = crossfit_context_direction(
        contractions,
        objective=objective,
        validation_grid_size=181,
        maxiter=250,
    )
    fit_seconds = time.perf_counter() - fit_start

    fold_masks = tuple(
        contractions.assignment.fold_mask(fixture.block_ids, fold) for fold in range(2)
    )
    disjoint = not bool(np.any(fold_masks[0] & fold_masks[1]))
    exhaustive = bool(np.all(fold_masks[0] | fold_masks[1]))
    fold_summaries: list[dict[str, Any]] = []
    for fold in result.folds:
        direction = np.asarray(fold.direction, dtype=np.float64)
        fold_summaries.append(
            {
                "fold_id": fold.fold_id,
                "training_objective": float(fold.training_objective),
                "heldout_objective": float(fold.heldout_objective),
                "direction": direction,
                "alignment_by_truth": [
                    _metric_alignment(direction, truth, fixture.context_metric)
                    for truth in fixture.true_directions
                ],
                "generating_direction_defined": (fixture.scenario.interaction_sd > 0.0),
                "training_rank": int(fold.training_optimization.evaluation.rank),
                "heldout_rank": int(fold.heldout_evaluation.rank),
                "training_condition_number": float(
                    fold.training_optimization.evaluation.condition_number
                ),
                "heldout_condition_number": float(
                    fold.heldout_evaluation.condition_number
                ),
                "training_optimizer_status": str(fold.training_optimization.status),
                "training_optimizer_nonunique": bool(
                    fold.training_optimization.nonunique
                ),
                "training_optimizer_objective_gap": (
                    fold.training_optimization.objective_gap
                ),
            }
        )
    summary = {
        "name": fixture.scenario.name,
        "objective": objective,
        "status": result.status,
        "combined_heldout_objective": float(result.combined_heldout_value),
        "combined_heldout_objective_is_selection_unbiased_claim": False,
        "nonlinear_selection_inference": result.manifest[
            "nonlinear_selection_inference"
        ],
        "fold_direction_alignment": float(result.fold_direction_alignment),
        "any_training_optimizer_nonunique": any(
            fold.training_optimization.nonunique for fold in result.folds
        ),
        "folds": fold_summaries,
        "assignment": {
            "fold_variant_counts": contractions.assignment.fold_variant_counts,
            "fold_weight_masses": contractions.assignment.fold_weight_masses,
            "fold_block_counts": [
                len(values) for values in contractions.assignment.fold_block_identities
            ],
            "disjoint": disjoint,
            "exhaustive": exhaustive,
            "train_heldout_hashes_distinct": all(
                arm.train_variant_hash != arm.heldout_variant_hash
                for arm in contractions.arms
            ),
            "reverse_arms_exchange_exact_folds": bool(
                contractions.arms[0].train_variant_hash
                == contractions.arms[1].heldout_variant_hash
                and contractions.arms[0].heldout_variant_hash
                == contractions.arms[1].train_variant_hash
            ),
            "construction": contractions.manifest["construction"],
            "uses_approximate_loo_deletion": contractions.manifest[
                "uses_approximate_loo_deletion"
            ],
        },
        "compact_crossfit": _compact_array_summary(contractions),
        "timing_seconds": {"build": build_seconds, "fit": fit_seconds},
    }
    return summary, contractions, result


def _contraction_arrays(contractions: Any) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(contractions.reference.base_gram),
        np.asarray(contractions.reference.same_person),
        np.asarray(contractions.reference.base_traces),
        np.asarray(contractions.summary.base_gram),
        np.asarray(contractions.summary.base_rhs),
        np.asarray(contractions.summary.base_traces),
    )


def _leakage_validation(
    fixture: SyntheticFixture,
    original_crossfit: Any,
    original_result: Any,
    *,
    objective: str = "he_moment_gain",
) -> dict[str, Any]:
    # Arm 0 trains on assignment fold 0 and holds out fold 1.  Permuting sample
    # rows only within held-out columns changes those kernels without touching
    # a single training-fold genotype entry.
    heldout_mask = original_crossfit.assignment.fold_mask(fixture.block_ids, 1)
    reference_mutated = fixture.reference_genotype.copy()
    study_mutated = fixture.study_genotype.copy()
    reference_mutated[:, heldout_mask] = np.roll(
        reference_mutated[:, heldout_mask], shift=17, axis=0
    )
    study_mutated[:, heldout_mask] = np.roll(
        study_mutated[:, heldout_mask], shift=11, axis=0
    )
    mutated_fixture = SyntheticFixture(
        scenario=fixture.scenario,
        reference_genotype=reference_mutated,
        study_genotype=study_mutated,
        reference_context=fixture.reference_context,
        study_context=fixture.study_context,
        reference_fixed_effects=fixture.reference_fixed_effects,
        study_fixed_effects=fixture.study_fixed_effects,
        reference_projector=fixture.reference_projector,
        study_projector=fixture.study_projector,
        phenotype=fixture.phenotype,
        block_ids=fixture.block_ids,
        variant_weights=fixture.variant_weights,
        variant_hash=fixture.variant_hash,
        environment_names=fixture.environment_names,
        context_metric=fixture.context_metric,
        true_directions=fixture.true_directions,
    )
    mutated = _build_crossfit_contractions(mutated_fixture)
    original_train_arrays = _contraction_arrays(original_crossfit.arms[0].train)
    mutated_train_arrays = _contraction_arrays(mutated.arms[0].train)
    training_tensors_identical = all(
        np.array_equal(left, right)
        for left, right in zip(original_train_arrays, mutated_train_arrays, strict=True)
    )
    heldout_tensor_change = max(
        _relative_error(left, right)
        for left, right in zip(
            _contraction_arrays(original_crossfit.arms[0].heldout),
            _contraction_arrays(mutated.arms[0].heldout),
            strict=True,
        )
    )
    original_training = original_result.folds[0].training_optimization
    mutated_training = optimize_context_direction(
        mutated.arms[0].train,
        objective=objective,
        validation_grid_size=181,
        maxiter=250,
    )
    directions_identical = bool(
        np.array_equal(original_training.direction, mutated_training.direction)
    )
    original_heldout = original_result.folds[0].heldout_evaluation
    mutated_heldout = evaluate_context_direction(
        mutated.arms[0].heldout,
        original_training.direction,
        objective=objective,
    )
    heldout_objective_change = float(
        abs(mutated_heldout.objective_value - original_heldout.objective_value)
    )
    gate = bool(
        training_tensors_identical
        and directions_identical
        and heldout_tensor_change > 1.0e-8
        and heldout_objective_change > 1.0e-8
    )
    return {
        "gate": gate,
        "mutation_scope": "heldout_fold_variants_only",
        "training_tensors_bit_identical": training_tensors_identical,
        "training_direction_bit_identical": directions_identical,
        "maximum_heldout_tensor_relative_change": heldout_tensor_change,
        "absolute_heldout_objective_change": heldout_objective_change,
        "optimizer_input": "training_compact_contractions_only",
        "heldout_input": "fixed_training_direction_plus_heldout_compact_contractions",
    }


def _compact_performance(contractions: Any, *, evaluations: int) -> dict[str, Any]:
    grid = _metric_circle(contractions.context_metric, min(evaluations, 997))
    started = time.perf_counter()
    checksum = 0.0
    successful = 0
    for index in range(evaluations):
        try:
            value = evaluate_context_direction(
                contractions,
                grid[index % grid.shape[0]],
                objective="he_moment_gain",
            )
        except ContextRankError:
            continue
        checksum += float(value.objective_value)
        successful += 1
    elapsed = time.perf_counter() - started
    compact = _compact_array_summary(contractions)
    return {
        "requested_evaluations": int(evaluations),
        "successful_evaluations": int(successful),
        "elapsed_seconds": elapsed,
        "evaluations_per_second": float(successful / max(elapsed, 1.0e-12)),
        "objective_checksum": checksum,
        "compact_array_bytes": compact["total_array_bytes"],
        "compact_array_shapes": compact["shapes"],
        "genotypes_revisited_during_compact_evaluation": False,
        "optimization_scaling_note": (
            "all timed evaluations use only L-dependent compact tensors and "
            "4-by-4 normal-equation solves"
        ),
    }


def _null_selection_audit(
    args: argparse.Namespace, *, strong_heldout_benchmark: float
) -> dict[str, Any]:
    null_scenario = next(
        scenario for scenario in _scenarios() if scenario.name == "null_signal"
    )
    n_samples = 96
    n_reference = 128
    n_variants = args.blocks * min(args.m // args.blocks, 4)
    training_values: list[float] = []
    heldout_values: list[float] = []
    fold_alignments: list[float] = []
    nonunique_count = 0
    started = time.perf_counter()
    for replicate in range(args.null_replicates):
        fixture = _simulate_fixture(
            null_scenario,
            n_study=n_samples,
            n_reference=n_reference,
            n_variants=n_variants,
            n_blocks=args.blocks,
            seed=args.seed + 10_000 + replicate,
        )
        contractions = _build_crossfit_contractions(fixture)
        result = crossfit_context_direction(
            contractions,
            objective="he_moment_gain",
            validation_grid_size=61,
            maxiter=150,
        )
        training_values.append(
            float(np.mean([fold.training_objective for fold in result.folds]))
        )
        heldout_values.append(float(result.combined_heldout_value))
        fold_alignments.append(float(result.fold_direction_alignment))
        nonunique_count += int(
            any(fold.training_optimization.nonunique for fold in result.folds)
        )
    training = np.asarray(training_values, dtype=np.float64)
    heldout = np.asarray(heldout_values, dtype=np.float64)
    alignments = np.asarray(fold_alignments, dtype=np.float64)
    optimism = training - heldout
    quantile_probabilities = np.asarray([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    retrospective_threshold = float(np.quantile(heldout, 0.95))
    benchmark_threshold = float(0.10 * strong_heldout_benchmark)

    def distribution(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "quantile_probabilities": quantile_probabilities,
            "quantiles": np.quantile(values, quantile_probabilities),
        }

    return {
        "replicates": int(args.null_replicates),
        "dimensions": {
            "study_n": n_samples,
            "reference_n": n_reference,
            "m": n_variants,
            "physical_blocks": int(args.blocks),
        },
        "training_optimized_gain": distribution(training),
        "fixed_direction_heldout_gain": distribution(heldout),
        "selection_optimism_training_minus_heldout": distribution(optimism),
        "fold_direction_alignment": distribution(alignments),
        "nonunique_training_optimizer_replicates": int(nonunique_count),
        "retrospective_null_95th_percentile": retrospective_threshold,
        "retrospective_threshold_exceedance_fraction": float(
            np.mean(heldout > retrospective_threshold)
        ),
        "strong_benchmark_tenth_threshold": benchmark_threshold,
        "strong_benchmark_threshold_exceedance_fraction": float(
            np.mean(heldout > benchmark_threshold)
        ),
        "threshold_status": (
            "descriptive_only_the_empirical_95th_percentile_is_reused_on_the_same_"
            "null_replicates_and_the_strong_benchmark_threshold_is_not_a_"
            "calibrated_significance_cutoff"
        ),
        "calibrated_false_positive_rate_claimed": False,
        "replicate_rows_written": False,
        "all_values_finite": bool(
            np.all(np.isfinite(training))
            and np.all(np.isfinite(heldout))
            and np.all(np.isfinite(alignments))
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _synthetic_gate_summary(
    dense: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    crossfits: Sequence[Mapping[str, Any]],
    leakage: Mapping[str, Any],
    performance: Mapping[str, Any],
    null_audit: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    by_name = {str(record["name"]): record for record in scenarios}
    cross_by_name = {str(record["name"]): record for record in crossfits}

    def fold_truth_minimum(name: str) -> float:
        return min(
            max(float(value) for value in fold["alignment_by_truth"])
            for fold in cross_by_name[name]["folds"]
        )

    exact_assignments = all(
        record["assignment"]["disjoint"]
        and record["assignment"]["exhaustive"]
        and record["assignment"]["train_heldout_hashes_distinct"]
        and record["assignment"]["reverse_arms_exchange_exact_folds"]
        and not record["assignment"]["uses_approximate_loo_deletion"]
        and min(record["assignment"]["fold_block_counts"]) >= 6
        for record in crossfits
    )
    compact_only = all(
        record["compact_crossfit"]["maximum_axis"]
        < min(record["assignment"]["fold_variant_counts"])
        for record in crossfits
    )
    one = by_name["one_true_direction"]
    correlated = by_name["correlated_context"]
    competing = by_name["competing_directions"]
    weak = cross_by_name["weak_signal"]["combined_heldout_objective"]
    null = cross_by_name["null_signal"]["combined_heldout_objective"]
    one_heldout = cross_by_name["one_true_direction"]["combined_heldout_objective"]
    correlated_heldout = cross_by_name["correlated_context"][
        "combined_heldout_objective"
    ]
    mismatch = by_name["reference_mismatch"]
    matched_metric_discrepancy = max(
        by_name[name]["reference_study_metric_relative_discrepancy"]
        for name in ("one_true_direction", "correlated_context")
    )
    null_ratio = float(null / max(one_heldout, np.finfo(float).tiny))
    weak_ratio = float(weak / max(one_heldout, np.finfo(float).tiny))
    mismatch_objective_ratio = float(
        mismatch["optimized_training_objective"]
        / max(one["optimized_training_objective"], np.finfo(float).tiny)
    )
    gates = {
        "direct_dense_four_kernel_contraction_and_optimization": bool(dense["gate"]),
        "one_true_direction_recovery": bool(
            one["maximum_alignment"] > 0.85
            and fold_truth_minimum("one_true_direction") > 0.80
        ),
        "competing_direction_recovery": bool(
            competing["maximum_alignment"] > 0.70
            and fold_truth_minimum("competing_directions") > 0.65
        ),
        "correlated_context_recovery": bool(
            abs(correlated["context_correlation_study"]) > 0.50
            and correlated["maximum_alignment"] > 0.85
            and fold_truth_minimum("correlated_context") > 0.80
        ),
        "weak_and_null_descriptive_separation": bool(
            null_ratio < 0.10
            and 0.0 <= weak_ratio < 0.25
            and one_heldout > null
            and correlated_heldout > null
        ),
        "reference_mismatch_stress_is_visible": bool(
            mismatch["reference_mismatch_declared"]
            and mismatch["reference_study_metric_relative_discrepancy"]
            > max(0.15, 2.0 * matched_metric_discrepancy)
            and (
                mismatch["maximum_alignment"] < one["maximum_alignment"] - 0.10
                or mismatch_objective_ratio > 5.0
            )
        ),
        "exact_disjoint_twelve_block_two_fold_construction": exact_assignments,
        "train_heldout_leakage_falsification": bool(leakage["gate"]),
        "compact_evaluation_without_row_arrays": bool(
            compact_only
            and performance["successful_evaluations"]
            == performance["requested_evaluations"]
        ),
        "repeated_null_selection_audit": bool(
            null_audit["replicates"] >= 20
            and null_audit["all_values_finite"]
            and not null_audit["calibrated_false_positive_rate_claimed"]
            and null_audit["selection_optimism_training_minus_heldout"]["median"] > 0.0
        ),
    }
    evidence = {
        "null_to_one_direction_heldout_objective_ratio": null_ratio,
        "weak_to_one_direction_heldout_objective_ratio": weak_ratio,
        "null_fold_direction_alignment": cross_by_name["null_signal"][
            "fold_direction_alignment"
        ],
        "false_positive_inference_status": (
            "not_calibrated_no_p_value_or_selection_threshold_is_claimed"
        ),
        "mismatch_to_matched_training_objective_ratio": mismatch_objective_ratio,
        "mismatch_metric_discrepancy": mismatch[
            "reference_study_metric_relative_discrepancy"
        ],
        "largest_matched_metric_discrepancy": matched_metric_discrepancy,
    }
    return gates, evidence


def _metric_angle(direction: np.ndarray, metric: np.ndarray) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    square_root = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
    unit = square_root @ _normalize_direction(direction, metric)
    return float(np.mod(np.arctan2(unit[1], unit[0]), np.pi))


def _plot_directions(
    scenarios: Sequence[Mapping[str, Any]],
    plot_data: Sequence[Mapping[str, np.ndarray]],
    metrics: Sequence[np.ndarray],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), squeeze=False)
    for axis, record, plotting, metric in zip(
        axes.flat, scenarios, plot_data, metrics, strict=True
    ):
        values = np.asarray(plotting["objective_values"], dtype=np.float64)
        x = np.linspace(0.0, 180.0, values.size, endpoint=False)
        finite = values[np.isfinite(values)]
        scale = max(float(np.max(np.abs(finite), initial=0.0)), 1.0e-12)
        axis.plot(x, values / scale, color="#4C78A8", linewidth=1.8)
        learned_angle = math.degrees(
            _metric_angle(np.asarray(record["learned_direction"]), metric)
        )
        axis.axvline(
            learned_angle,
            color="#E45756",
            linewidth=1.5,
            label="learned",
        )
        if record["generating_direction_defined"]:
            for truth_index, truth in enumerate(record["true_directions"]):
                truth_angle = math.degrees(_metric_angle(np.asarray(truth), metric))
                axis.axvline(
                    truth_angle,
                    color="#54A24B",
                    linestyle="--",
                    linewidth=1.2,
                    label="truth" if truth_index == 0 else None,
                )
            alignment_label = f"alignment={record['maximum_alignment']:.2f}"
        else:
            alignment_label = "no generating direction under null"
        axis.set_title(f"{record['label']} ({alignment_label})")
        axis.set_xlabel("metric-sphere direction angle (degrees; sign-identified)")
        axis.set_ylabel("HE gain / scenario max |gain|")
        axis.grid(alpha=0.20)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Compact objective landscapes and learned directions", y=1.01)
    figure.tight_layout()
    _save_figure(figure, output_dir, "directions")


def _plot_crossfit(
    scenarios: Sequence[Mapping[str, Any]],
    crossfits: Sequence[Mapping[str, Any]],
    null_audit: Mapping[str, Any],
    output_dir: Path,
) -> None:
    labels = [str(record["label"]) for record in scenarios]
    train = [
        float(np.mean([fold["training_objective"] for fold in record["folds"]]))
        for record in crossfits
    ]
    heldout = [float(record["combined_heldout_objective"]) for record in crossfits]
    full_alignment = [
        (
            float(record["maximum_alignment"])
            if record["generating_direction_defined"]
            else np.nan
        )
        for record in scenarios
    ]
    fold_truth = [
        (
            float(
                np.mean([max(fold["alignment_by_truth"]) for fold in record["folds"]])
            )
            if record["folds"][0]["generating_direction_defined"]
            else np.nan
        )
        for record in crossfits
    ]
    fold_agreement = [float(record["fold_direction_alignment"]) for record in crossfits]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 4.8))
    axes[0].bar(x - 0.18, train, 0.36, label="training optimized")
    axes[0].bar(x + 0.18, heldout, 0.36, label="fixed-direction held out")
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("HE moment gain (symlog)")
    axes[0].set_title("Selection optimism and exact held-out evaluation")
    axes[0].legend(frameon=False)
    axes[1].plot(x, full_alignment, "o-", label="full-data vs nearest truth")
    axes[1].plot(x, fold_truth, "s-", label="fold vs nearest truth")
    axes[1].plot(x, fold_agreement, "^-", label="between-fold direction")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("absolute metric alignment")
    axes[1].set_title("Direction stability")
    axes[1].legend(frameon=False)
    null_distributions = [
        null_audit["training_optimized_gain"],
        null_audit["fixed_direction_heldout_gain"],
        null_audit["selection_optimism_training_minus_heldout"],
    ]
    null_labels = ["training", "held out", "optimism"]
    for index, distribution in enumerate(null_distributions):
        quantiles = np.asarray(distribution["quantiles"], dtype=np.float64)
        median = quantiles[2]
        axes[2].vlines(index, quantiles[0], quantiles[-1], color="#9C9C9C")
        axes[2].vlines(index, quantiles[1], quantiles[3], color="#4C78A8", linewidth=7)
        axes[2].scatter(index, median, color="#E45756", zorder=3)
    axes[2].set_xticks(np.arange(3), null_labels)
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[2].set_ylabel("HE moment gain (symlog)")
    axes[2].set_title(
        f"Repeated null audit (N={null_audit['replicates']}; min–max/IQR/median)"
    )
    axes[2].grid(axis="y", alpha=0.20)
    for axis in axes[:2]:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.20)
    figure.tight_layout()
    _save_figure(figure, output_dir, "crossfit")


def _plot_performance(
    scenarios: Sequence[Mapping[str, Any]],
    crossfits: Sequence[Mapping[str, Any]],
    performance: Mapping[str, Any],
    output_dir: Path,
) -> None:
    labels = [str(record["name"]).replace("_", "\n") for record in scenarios]
    full_build = [record["timing_seconds"]["build"] for record in scenarios]
    cross_build = [record["timing_seconds"]["build"] for record in crossfits]
    optimization = [
        record["timing_seconds"]["optimize"] + crossfit["timing_seconds"]["fit"]
        for record, crossfit in zip(scenarios, crossfits, strict=True)
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    axes[0].bar(x, full_build, label="full contraction build")
    axes[0].bar(x, cross_build, bottom=full_build, label="two exact fold builds")
    bottom = np.asarray(full_build) + np.asarray(cross_build)
    axes[0].bar(x, optimization, bottom=bottom, label="optimization/evaluation")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("wall seconds")
    axes[0].set_title("Synthetic phase timing")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.20)
    labels_right = ["evaluations / s", "compact KiB"]
    values_right = [
        performance["evaluations_per_second"],
        performance["compact_array_bytes"] / 1024.0,
    ]
    bars = axes[1].bar(labels_right, values_right, color=("#4C78A8", "#F58518"))
    axes[1].bar_label(bars, fmt="%.1f")
    axes[1].set_yscale("log")
    axes[1].set_title(f"{performance['successful_evaluations']:,} compact evaluations")
    axes[1].grid(axis="y", alpha=0.20)
    figure.tight_layout()
    _save_figure(figure, output_dir, "performance")


def _read_real_fixture(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        args.phenotype_root / "smoking_status.pheno",
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
    for name in ("bmi", "c_reactive_prot", "smoking_status"):
        values = read_table(
            args.phenotype_root / f"{name}.pheno",
            ("FID", "IID", "pheno"),
        ).rename(columns={"pheno": name})
        merged = merged.merge(values, on=["FID", "IID"], validate="one_to_one")
    numeric_columns = [
        "sex",
        "age",
        *PC_COLUMNS,
        "bmi",
        "c_reactive_prot",
        "smoking_status",
    ]
    numeric = merged[numeric_columns].to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(numeric), axis=1) & np.all(numeric != -9.0, axis=1)
    intersection = merged.loc[valid].copy()
    if len(intersection) < args.real_n:
        raise ValueError(
            f"Only {len(intersection)} complete cases remain for --real-n={args.real_n}."
        )
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 9700]))
    chosen_rows = np.sort(
        rng.choice(len(intersection), size=args.real_n, replace=False)
    )
    selected = (
        intersection.iloc[chosen_rows].sort_values("bed_row").reset_index(drop=True)
    )
    bed = open_bed(Path(f"{args.geno_prefix}.bed"))
    candidate_count = min(int(bed.sid_count), 5 * args.real_m)
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
    valid_variants = (
        np.isfinite(means)
        & np.isfinite(variances)
        & (variances > np.finfo(np.float64).eps)
        & (missing_rate <= 0.05)
        & (maf >= 0.05)
    )
    retained = np.flatnonzero(valid_variants)[: args.real_m]
    if retained.size != args.real_m:
        raise ValueError(
            f"Only {retained.size} deterministic candidate variants passed QC."
        )
    genotype = raw[:, retained].copy()
    missing_genotype = np.isnan(genotype)
    if np.any(missing_genotype):
        genotype[missing_genotype] = np.broadcast_to(
            means[retained][None, :], genotype.shape
        )[missing_genotype]
    genotype = _standardize(genotype)
    chosen_variant_indices = candidates[retained]

    context = _standardize(
        selected[["age", "sex", "smoking_status"]].to_numpy(dtype=np.float64)
    )
    pcs = _standardize(selected[list(PC_COLUMNS)].to_numpy(dtype=np.float64))
    fixed_effects = np.column_stack(
        [np.ones(args.real_n), context, pcs, pcs[:, [0]] * context]
    )
    projector = rank_revealing_projector(fixed_effects)
    metric = context.T @ context / float(args.real_n)
    block_ids = np.repeat(
        np.arange(args.blocks, dtype=np.int64), args.real_m // args.blocks
    )
    sex_values, sex_counts = np.unique(selected["sex"], return_counts=True)
    smoking_values, smoking_counts = np.unique(
        selected["smoking_status"], return_counts=True
    )
    fixture = {
        "genotype": genotype,
        "context": context,
        "fixed_effects": fixed_effects,
        "projector": projector,
        "metric": metric,
        "block_ids": block_ids,
        "variant_hash": canonical_sha256(
            {"selected_index_sha256": canonical_sha256(chosen_variant_indices.tolist())}
        ),
        "phenotypes": {
            name: selected[name].to_numpy(dtype=np.float64)
            for name in ("bmi", "c_reactive_prot")
        },
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
        "context_coordinates": [
            "standardized_age",
            "standardized_sex_code",
            "standardized_smoking_status_code",
        ],
        "ordered_category_limitation": (
            "sex and smoking are numeric coordinates for this engineering sanity; "
            "no ordinal or biological direction claim is made"
        ),
    }
    return fixture, diagnostics


def _real_trait_sanity(args: argparse.Namespace) -> dict[str, Any]:
    fixture, diagnostics = _read_real_fixture(args)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trait_name, phenotype in fixture["phenotypes"].items():
        synthetic_shell = SyntheticFixture(
            scenario=SyntheticScenario(
                name=f"real_{trait_name}",
                label=trait_name,
                context_correlation=float(np.corrcoef(fixture["context"].T)[0, 1]),
                context_pc_correlation=0.0,
                directions=((1.0, 0.0, 0.0),),
                interaction_sd=0.0,
            ),
            reference_genotype=fixture["genotype"],
            study_genotype=fixture["genotype"],
            reference_context=fixture["context"],
            study_context=fixture["context"],
            reference_fixed_effects=fixture["fixed_effects"],
            study_fixed_effects=fixture["fixed_effects"],
            reference_projector=fixture["projector"],
            study_projector=fixture["projector"],
            phenotype=phenotype,
            block_ids=fixture["block_ids"],
            variant_weights=np.ones(args.real_m, dtype=np.float64),
            variant_hash=fixture["variant_hash"],
            environment_names=("age", "sex_code", "smoking_status_code"),
            context_metric=fixture["metric"],
            true_directions=(np.asarray([1.0, 0.0, 0.0]),),
        )
        _, _, contractions = _build_public_contractions(synthetic_shell)
        optimized = optimize_context_direction(
            contractions,
            objective="he_moment_gain",
            validation_grid_size=384,
            maxiter=250,
        )
        crossfit_contractions = _build_crossfit_contractions(synthetic_shell)
        crossfit = crossfit_context_direction(
            crossfit_contractions,
            objective="he_moment_gain",
            validation_grid_size=384,
            maxiter=250,
        )
        records.append(
            {
                "trait": trait_name,
                "direction_coordinates": list(synthetic_shell.environment_names),
                "learned_direction": optimized.direction,
                "training_objective": float(optimized.objective_value),
                "interaction_coefficient": float(optimized.evaluation.coefficients[1]),
                "interaction_trace_contribution": float(
                    optimized.evaluation.objective_values[
                        "interaction_trace_contribution"
                    ]
                ),
                "rank": int(optimized.evaluation.rank),
                "condition_number": float(optimized.evaluation.condition_number),
                "relative_residual": float(optimized.evaluation.relative_residual),
                "optimizer_status": str(optimized.status),
                "optimizer_nonunique": bool(optimized.nonunique),
                "optimizer_objective_gap": optimized.objective_gap,
                "heldout_objective": float(crossfit.combined_heldout_value),
                "fold_direction_alignment": float(crossfit.fold_direction_alignment),
                "fold_directions": [fold.direction for fold in crossfit.folds],
                "fold_training_objectives": [
                    float(fold.training_objective) for fold in crossfit.folds
                ],
                "fold_heldout_objectives": [
                    float(fold.heldout_objective) for fold in crossfit.folds
                ],
                "fold_optimizer_nonunique": [
                    bool(fold.training_optimization.nonunique)
                    for fold in crossfit.folds
                ],
                "inference_status": (
                    "descriptive_only_no_crossfit_selection_standard_error_or_p_value"
                ),
            }
        )
    return {
        "purpose": "privacy_preserving_engineering_sanity_not_significance_gate",
        "diagnostics": diagnostics,
        "records": records,
        "elapsed_seconds": float(time.perf_counter() - started),
        "privacy": {
            "identifiers_written": False,
            "sample_rows_written": False,
            "variant_rows_written": False,
            "outputs": "aggregate_counts_directions_objectives_and_diagnostics_only",
        },
    }


def _plot_real(real: Mapping[str, Any], output_dir: Path) -> None:
    records = list(real["records"])
    labels = [str(record["trait"]) for record in records]
    coordinate_names = records[0]["direction_coordinates"]
    x = np.arange(len(records))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    width = 0.72 / len(coordinate_names)
    for index, coordinate in enumerate(coordinate_names):
        axes[0].bar(
            x + (index - (len(coordinate_names) - 1) / 2.0) * width,
            [record["learned_direction"][index] for record in records],
            width,
            label=coordinate,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("metric-normalized direction coefficient")
    axes[0].set_title("Real-trait learned direction (coordinate-dependent)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(
        x - 0.18,
        [record["training_objective"] for record in records],
        0.36,
        label="training optimized",
    )
    axes[1].bar(
        x + 0.18,
        [record["heldout_objective"] for record in records],
        0.36,
        label="fixed-direction held out",
    )
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("HE moment gain (symlog)")
    axes[1].set_title("Descriptive exact-fold sanity")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.20)
    figure.tight_layout()
    _save_figure(figure, output_dir, "real_traits")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dense_m = args.blocks * min(args.m // args.blocks, 4)
    dense = _dense_parity_validation(
        n_samples=96,
        n_variants=dense_m,
        n_blocks=args.blocks,
        seed=args.seed + 17,
    )

    scenarios: list[dict[str, Any]] = []
    crossfits: list[dict[str, Any]] = []
    plotting: list[dict[str, np.ndarray]] = []
    metrics: list[np.ndarray] = []
    leakage_fixture: SyntheticFixture | None = None
    leakage_contractions: Any = None
    leakage_result: Any = None
    for index, scenario in enumerate(_scenarios()):
        fixture = _simulate_fixture(
            scenario,
            n_study=args.n,
            n_reference=args.reference_n,
            n_variants=args.m,
            n_blocks=args.blocks,
            seed=args.seed + 100 * index,
        )
        scenario_result, plot_data = _scenario_validation(fixture)
        crossfit_result, compact_crossfit, fitted_crossfit = _crossfit_validation(
            fixture
        )
        scenarios.append(scenario_result)
        crossfits.append(crossfit_result)
        plotting.append(plot_data)
        metrics.append(fixture.context_metric.copy())
        if scenario.name == "one_true_direction":
            leakage_fixture = fixture
            leakage_contractions = compact_crossfit
            leakage_result = fitted_crossfit
    if leakage_fixture is None:
        raise RuntimeError("The one-direction leakage fixture was not constructed.")
    leakage = _leakage_validation(leakage_fixture, leakage_contractions, leakage_result)
    performance = _compact_performance(
        leakage_contractions.arms[0].train,
        evaluations=args.benchmark_evaluations,
    )
    one_heldout_benchmark = next(
        record["combined_heldout_objective"]
        for record in crossfits
        if record["name"] == "one_true_direction"
    )
    null_audit = _null_selection_audit(
        args, strong_heldout_benchmark=float(one_heldout_benchmark)
    )
    gates, comparative_evidence = _synthetic_gate_summary(
        dense, scenarios, crossfits, leakage, performance, null_audit
    )
    synthetic_pass = all(gates.values())
    real: Mapping[str, Any] | None
    if args.real_traits and synthetic_pass:
        real = _real_trait_sanity(args)
    elif args.real_traits:
        real = {
            "status": "skipped_because_synthetic_gate_failed",
            "privacy": {
                "identifiers_written": False,
                "sample_rows_written": False,
                "variant_rows_written": False,
            },
        }
    else:
        real = None

    _plot_directions(scenarios, plotting, metrics, args.output_dir)
    _plot_crossfit(scenarios, crossfits, null_audit, args.output_dir)
    _plot_performance(scenarios, crossfits, performance, args.output_dir)
    if real is not None and "records" in real:
        _plot_real(real, args.output_dir)

    return {
        "kind": "summit.context.context_direction_validation",
        "experimental": True,
        "seed": int(args.seed),
        "configuration": {
            "study_n": int(args.n),
            "reference_n": int(args.reference_n),
            "m": int(args.m),
            "physical_blocks": int(args.blocks),
            "folds": 2,
            "benchmark_evaluations": int(args.benchmark_evaluations),
            "null_replicates": int(args.null_replicates),
            "context_dimension": 2,
            "objective": "he_moment_gain",
            "constraint": "omega.T @ fixed_reference_context_metric @ omega == 1",
            "reduced_kernels": ["G", "I(omega)", "P", "D(omega)"],
        },
        "dense_oracle": dense,
        "synthetic_scenarios": scenarios,
        "crossfit_scenarios": crossfits,
        "leakage_falsification": leakage,
        "comparative_evidence": comparative_evidence,
        "repeated_null_selection_audit": null_audit,
        "performance": performance,
        "real_trait_sanity": real,
        "gates": gates,
        "verdict": "pass" if synthetic_pass else "review",
        "scope_verdict": (
            "narrow_pass_compact_fixed_direction_evaluation_and_strong_signal_"
            "learning_are_supported_but_crossfit_selection_inference_is_not_"
            "calibrated_and_reference_mismatch_is_material"
            if synthetic_pass
            else "review_failed_synthetic_gate_before_method_scope_decision"
        ),
        "interpretation": {
            "training_objective": (
                "optimized in sample and explicitly not treated as unbiased"
            ),
            "heldout_objective": (
                "fixed training-fold direction evaluated on the exact disjoint "
                "variant complement; descriptive, without calibrated selection inference"
            ),
            "covariance_modes": "not used as or equated with learned directions",
            "reference_mismatch": (
                "stress diagnostic, not a transport-robustness guarantee"
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "output_policy": {
            "row_level_data_written": False,
            "individual_identifiers_written": False,
            "variant_rows_written": False,
            "individual_level_arrays_written": False,
            "aggregate_direction_and_objective_curves_written": True,
            "real_trait_inference_claimed": False,
        },
    }


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


if __name__ == "__main__":
    main()
