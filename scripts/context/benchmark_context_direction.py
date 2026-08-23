#!/usr/bin/env python3
"""Benchmark compact Stage-07B direction contractions and optimization.

The deterministic benchmark separates genotype-dependent contraction builders
from post-contraction evaluation and optimization.  Compact evaluation is run
10,000 times by default for L=2 and L=3 over multiple N/M fixtures.  Exact
two-fold cross-fitting is built from whole disjoint variant blocks.  Only
aggregate timings, tensor inventories, hashes, and diagnostics are persisted;
no sample or variant rows are written.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from summit.context import (
    ContextRankError,
    array_sha256,
    build_context_direction_crossfit_contractions,
    build_direction_reference_contractions,
    build_direction_trait_contractions,
    canonical_json,
    canonical_sha256,
    combine_context_direction_contractions,
    crossfit_context_direction,
    evaluate_context_direction,
    normalize_context_direction,
    optimize_context_direction,
    rank_revealing_projector,
    validate_context_direction_contractions,
    validate_direction_crossfit_contractions,
)


OUTPUT_STEM = "07b_context_direction_benchmark"
GENOTYPE_SCALING = "sample_sd_ddof1_pre_scaled_input"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--builder-shapes",
        default="48x64,80x128",
        help="Comma-separated N-by-M builder fixtures, for example 48x64,80x128.",
    )
    parser.add_argument(
        "--l-values",
        default="2,3",
        help="Comma-separated environment dimensions; supported values are 1..3.",
    )
    parser.add_argument("--builder-warmups", type=int, default=1)
    parser.add_argument("--builder-repeats", type=int, default=3)
    parser.add_argument(
        "--evaluation-calls",
        type=int,
        default=10_000,
        help="Compact evaluations in each timing sample.",
    )
    parser.add_argument("--evaluation-repeats", type=int, default=2)
    parser.add_argument("--direction-pool-size", type=int, default=64)
    parser.add_argument("--optimizer-grid-size", type=int, default=128)
    parser.add_argument("--crossfit-grid-size", type=int, default=64)
    parser.add_argument("--optimizer-maxiter", type=int, default=250)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument(
        "--objective",
        choices=(
            "interaction_coefficient",
            "interaction_trace_contribution",
            "he_moment_gain",
        ),
        default="he_moment_gain",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this benchmark's three fixed-name aggregate outputs.",
    )
    return parser


def _integer_list(
    parser: argparse.ArgumentParser, text: str, name: str
) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError:
        parser.error(f"{name} must be a comma-separated list of integers")
    if not values:
        parser.error(f"{name} must contain at least one value")
    if len(set(values)) != len(values):
        parser.error(f"{name} must not contain duplicates")
    return values


def _builder_shapes(
    parser: argparse.ArgumentParser, text: str
) -> tuple[tuple[int, int], ...]:
    values: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        pieces = item.split("x")
        if len(pieces) != 2:
            parser.error("--builder-shapes entries must have form NxM")
        try:
            values.append((int(pieces[0]), int(pieces[1])))
        except ValueError:
            parser.error("--builder-shapes entries must have integer N and M")
    if not values:
        parser.error("--builder-shapes must contain at least one fixture")
    if len(set(values)) != len(values):
        parser.error("--builder-shapes must not contain duplicates")
    return tuple(values)


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    args.l_values = _integer_list(parser, args.l_values, "--l-values")
    args.builder_shapes = _builder_shapes(parser, args.builder_shapes)
    if any(value < 1 or value > 3 for value in args.l_values):
        parser.error("every --l-values entry must lie between 1 and 3")
    if any(
        n_samples < 16 or n_variants < 16
        for n_samples, n_variants in args.builder_shapes
    ):
        parser.error("every builder fixture must have N>=16 and M>=16")
    if args.builder_warmups < 0:
        parser.error("--builder-warmups must be non-negative")
    if args.builder_repeats < 1:
        parser.error("--builder-repeats must be positive")
    if args.evaluation_calls < 1:
        parser.error("--evaluation-calls must be positive")
    if args.evaluation_repeats < 1:
        parser.error("--evaluation-repeats must be positive")
    if args.direction_pool_size < 1:
        parser.error("--direction-pool-size must be positive")
    if args.optimizer_grid_size < 0 or args.crossfit_grid_size < 0:
        parser.error("optimizer grid sizes must be non-negative")
    if args.optimizer_maxiter < 1:
        parser.error("--optimizer-maxiter must be positive")
    if args.blocks < 2:
        parser.error("--blocks must be at least two")
    if any(n_variants % args.blocks for _, n_variants in args.builder_shapes):
        parser.error("--blocks must divide every benchmark M exactly")
    if args.seed < 0:
        parser.error("--seed must be non-negative")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _timed(function: Callable[[], Any]) -> tuple[Any, float, int]:
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    return result, float(elapsed), _peak_rss_bytes()


def _standardize_columns(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True, dtype=np.float64)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("Synthetic fixture generated a degenerate column.")
    return np.ascontiguousarray(centered / scale[None, :], dtype=np.float64)


def _fixed_context_metric(l_count: int) -> np.ndarray:
    metric = np.fromfunction(
        lambda row, column: 0.28 ** np.abs(row - column),
        (l_count, l_count),
        dtype=int,
    )
    return np.asarray(metric, dtype=np.float64)


def _sample_fixture(
    *, l_count: int, n_samples: int, n_variants: int, seed: int, blocks: int
) -> dict[str, Any]:
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, l_count, n_samples, n_variants])
    )
    metric = _fixed_context_metric(l_count)
    metric_sqrt = np.linalg.cholesky(metric)

    def context() -> np.ndarray:
        return np.ascontiguousarray(
            rng.standard_normal((n_samples, l_count)) @ metric_sqrt.T,
            dtype=np.float64,
        )

    reference_context = context()
    study_context = context()
    reference_genotype = _standardize_columns(
        rng.standard_normal((n_samples, n_variants))
    )
    study_genotype = _standardize_columns(rng.standard_normal((n_samples, n_variants)))
    reference_pc = rng.standard_normal(n_samples)
    study_pc = rng.standard_normal(n_samples)
    reference_fixed = np.column_stack(
        [np.ones(n_samples), reference_context, reference_pc]
    )
    study_fixed = np.column_stack([np.ones(n_samples), study_context, study_pc])
    reference_projector = rank_revealing_projector(reference_fixed)
    study_projector = rank_revealing_projector(study_fixed)
    raw_direction = np.linspace(0.9, -0.45, l_count, dtype=np.float64)
    true_direction = normalize_context_direction(raw_direction, metric)
    additive_effect = rng.standard_normal(n_variants)
    interaction_effect = rng.standard_normal(n_variants)
    environment = study_context @ true_direction
    phenotype = (
        0.22 * study_genotype @ additive_effect / math.sqrt(n_variants)
        + 0.72
        * environment
        * (study_genotype @ interaction_effect)
        / math.sqrt(n_variants)
        + 0.42 * rng.standard_normal(n_samples)
    )
    names = tuple(f"environment_{index}" for index in range(l_count))
    context_spec_hash = canonical_sha256(
        {
            "stage": "07b_context_direction_benchmark",
            "environment_names": list(names),
            "context_metric": metric.tolist(),
        }
    )
    variant_hash = canonical_sha256(
        {"ordered_synthetic_variants": list(range(n_variants))}
    )
    variant_weights = np.linspace(0.75, 1.25, n_variants, dtype=np.float64)
    block_ids = np.arange(n_variants, dtype=np.int64) % blocks
    return {
        "reference_genotype": reference_genotype,
        "study_genotype": study_genotype,
        "reference_context": reference_context,
        "study_context": study_context,
        "reference_projector": reference_projector,
        "study_projector": study_projector,
        "phenotype": np.asarray(phenotype, dtype=np.float64),
        "context_metric": metric,
        "environment_names": names,
        "context_spec_hash": context_spec_hash,
        "variant_hash": variant_hash,
        "variant_weights": variant_weights,
        "block_ids": block_ids,
        "true_direction": true_direction,
        "reference_fixed_effect_hash": array_sha256(reference_fixed),
        "study_fixed_effect_hash": array_sha256(study_fixed),
    }


def _json_safe(value: Any) -> Any:
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


def _tensor_inventory(value: Any) -> dict[str, Any]:
    arrays: list[tuple[str, np.ndarray]] = []
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        identifier = id(item)
        if identifier in seen:
            return
        seen.add(identifier)
        if isinstance(item, np.ndarray):
            arrays.append((path, item))
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}")
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "object")
    return {
        "array_count": len(arrays),
        "total_array_bytes": int(sum(array.nbytes for _, array in arrays)),
        "total_array_elements": int(sum(array.size for _, array in arrays)),
        "maximum_axis": int(
            max((max(array.shape, default=0) for _, array in arrays), default=0)
        ),
        "arrays": [
            {
                "path": path,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": int(array.nbytes),
            }
            for path, array in arrays
        ],
    }


def _manifest_bytes(value: Any) -> int:
    manifests: list[Mapping[str, Any]] = []
    seen_objects: set[int] = set()
    seen_manifests: set[int] = set()

    def visit(item: Any) -> None:
        identifier = id(item)
        if identifier in seen_objects:
            return
        seen_objects.add(identifier)
        if is_dataclass(item):
            for field in fields(item):
                child = getattr(item, field.name)
                if field.name == "manifest" and isinstance(child, Mapping):
                    if id(child) not in seen_manifests:
                        manifests.append(child)
                        seen_manifests.add(id(child))
                else:
                    visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return int(
        sum(
            len(canonical_json(_json_safe(manifest)).encode("utf-8"))
            for manifest in manifests
        )
    )


def _expected_compact_tensor_bytes(l_count: int) -> int:
    pair_count = l_count * (l_count + 1) // 2
    reference_dimension = 1 + pair_count
    trait_dimension = 2 + 2 * pair_count
    reference_elements = (
        l_count**2 + 2 * reference_dimension**2 + reference_dimension
    )
    trait_elements = l_count**2 + trait_dimension**2 + 2 * trait_dimension
    return int(8 * (reference_elements + trait_elements))


def _phase_medians(values: Sequence[Any]) -> dict[str, float]:
    names = sorted(
        {str(name) for value in values for name in value.phase_times_seconds}
    )
    return {
        name: float(
            np.median(
                [float(value.phase_times_seconds.get(name, 0.0)) for value in values]
            )
        )
        for name in names
    }


def _repeat_builder(
    function: Callable[[], Any], *, warmups: int, repeats: int
) -> tuple[Any, dict[str, Any]]:
    for _ in range(warmups):
        function()
        gc.collect()
    values: list[Any] = []
    wall_times: list[float] = []
    process_peaks: list[int] = []
    for _ in range(repeats):
        value, seconds, peak = _timed(function)
        values.append(value)
        wall_times.append(seconds)
        process_peaks.append(peak)
        gc.collect()
    return values[-1], {
        "wall_samples": wall_times,
        "wall_median": float(np.median(wall_times)),
        "internal_phase_medians": _phase_medians(values),
        "builder_reported_rss_at_return_samples": [
            int(value.peak_rss_bytes) for value in values
        ],
        "builder_reported_rss_at_return_median": int(
            np.median([value.peak_rss_bytes for value in values])
        ),
        "absolute_process_high_water_samples": process_peaks,
        "absolute_process_high_water_maximum": max(process_peaks),
    }


def _builder_common(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context_metric": fixture["context_metric"],
        "environment_names": fixture["environment_names"],
        "variant_hash": fixture["variant_hash"],
        "context_spec_hash": fixture["context_spec_hash"],
        "genotype_scaling": GENOTYPE_SCALING,
        "variant_weights": fixture["variant_weights"],
    }


def _build_case(
    *, l_count: int, n_samples: int, n_variants: int, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = _sample_fixture(
        l_count=l_count,
        n_samples=n_samples,
        n_variants=n_variants,
        seed=args.seed,
        blocks=args.blocks,
    )
    common = _builder_common(fixture)

    reference, reference_timing = _repeat_builder(
        lambda: build_direction_reference_contractions(
            fixture["reference_genotype"],
            fixture["reference_context"],
            fixture["reference_projector"],
            fixed_effect_hash=fixture["reference_fixed_effect_hash"],
            **common,
        ),
        warmups=args.builder_warmups,
        repeats=args.builder_repeats,
    )
    summary, trait_timing = _repeat_builder(
        lambda: build_direction_trait_contractions(
            fixture["study_genotype"],
            fixture["study_context"],
            fixture["study_projector"],
            fixture["phenotype"],
            fixed_effect_hash=fixture["study_fixed_effect_hash"],
            **common,
        ),
        warmups=args.builder_warmups,
        repeats=args.builder_repeats,
    )
    contractions, combine_seconds, combine_peak = _timed(
        lambda: combine_context_direction_contractions(reference, summary)
    )
    validation = validate_context_direction_contractions(contractions)
    inventory = _tensor_inventory(contractions)
    expected_bytes = _expected_compact_tensor_bytes(l_count)
    manifest_hashes_match = bool(
        contractions.manifest["reference_manifest_hash"]
        == canonical_sha256(reference.manifest)
        and contractions.manifest["trait_manifest_hash"]
        == canonical_sha256(summary.manifest)
        and contractions.manifest["shared_specification_hash"]
        == reference.manifest["shared_specification_hash"]
        == summary.manifest["shared_specification_hash"]
    )
    record = {
        "l": l_count,
        "n": n_samples,
        "m": n_variants,
        "pair_count": len(contractions.pair_index),
        "reference_compact_dimension": reference.compact_dimension,
        "trait_compact_dimension": summary.compact_dimension,
        "builder_workload_proxy_n2m": int(n_samples**2 * n_variants),
        "builder_runtime_seconds": {
            "reference": reference_timing,
            "trait": trait_timing,
            "combine": combine_seconds,
        },
        "absolute_process_high_water_after_combine": combine_peak,
        "compact_payload": {
            "tensor_inventory": inventory,
            "expected_tensor_bytes": expected_bytes,
            "tensor_bytes_match_formula": bool(
                inventory["total_array_bytes"] == expected_bytes
            ),
            "manifest_json_bytes": _manifest_bytes(contractions),
            "estimated_tensor_plus_manifest_bytes": int(
                inventory["total_array_bytes"] + _manifest_bytes(contractions)
            ),
            "maximum_axis_below_n_and_m": bool(
                inventory["maximum_axis"] < min(n_samples, n_variants)
            ),
        },
        "binding_audit": {
            "public_validation": validation,
            "manifest_hashes_match": manifest_hashes_match,
            "same_selected_variant_hash": bool(
                reference.selected_variant_hash == summary.selected_variant_hash
            ),
            "optimizer_reads_individual_data": contractions.manifest.get(
                "optimizer_reads_individual_data"
            ),
        },
    }
    return record, {"fixture": fixture, "contractions": contractions}


def _direction_pool(
    contractions: Any, *, requested: int, objective: str, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < requested and attempts < 20 * requested:
        attempts += 1
        candidate = normalize_context_direction(
            rng.standard_normal(len(contractions.environment_names)),
            contractions.context_metric,
        )
        try:
            evaluate_context_direction(contractions, candidate, objective=objective)
        except ContextRankError:
            continue
        accepted.append(candidate)
    if not accepted:
        raise RuntimeError("No identifiable compact direction was generated.")
    return np.ascontiguousarray(np.stack(accepted, axis=0))


def _benchmark_evaluations(
    contractions: Any,
    *,
    calls: int,
    repeats: int,
    pool_size: int,
    objective: str,
    seed: int,
) -> dict[str, Any]:
    directions = _direction_pool(
        contractions, requested=pool_size, objective=objective, seed=seed
    )
    for index in range(min(32, calls)):
        evaluate_context_direction(
            contractions,
            directions[index % directions.shape[0]],
            objective=objective,
        )
    samples: list[float] = []
    checksums: list[float] = []
    failures: list[int] = []
    high_water: list[int] = []
    for _ in range(repeats):
        checksum = 0.0
        failed = 0
        started = time.perf_counter()
        for index in range(calls):
            try:
                evaluation = evaluate_context_direction(
                    contractions,
                    directions[index % directions.shape[0]],
                    objective=objective,
                )
            except ContextRankError:
                failed += 1
                continue
            checksum += evaluation.objective_value
        samples.append(float(time.perf_counter() - started))
        checksums.append(float(checksum))
        failures.append(failed)
        high_water.append(_peak_rss_bytes())
    median = float(np.median(samples))
    return {
        "calls_per_sample": calls,
        "repeats": repeats,
        "direction_pool_size": int(directions.shape[0]),
        "wall_samples": samples,
        "wall_median": median,
        "microseconds_per_call_median": 1.0e6 * median / calls,
        "calls_per_second_median": calls / median,
        "failed_evaluations": failures,
        "finite_checksums": bool(np.all(np.isfinite(checksums))),
        "checksum_samples": checksums,
        "absolute_process_high_water_samples": high_water,
    }


def _benchmark_optimizer(
    contractions: Any, *, args: argparse.Namespace
) -> dict[str, Any]:
    result, seconds, peak = _timed(
        lambda: optimize_context_direction(
            contractions,
            objective=args.objective,
            validation_grid_size=args.optimizer_grid_size,
            maxiter=args.optimizer_maxiter,
        )
    )
    metric_norm = float(
        result.direction @ contractions.context_metric @ result.direction
    )
    return {
        "wall_seconds": seconds,
        "absolute_process_high_water_bytes": peak,
        "objective": result.objective_name,
        "objective_value": float(result.objective_value),
        "direction": result.direction.tolist(),
        "metric_norm_squared": metric_norm,
        "evaluations": int(result.evaluations),
        "converged_candidates": int(result.converged_candidates),
        "failed_candidates": int(result.failed_candidates),
        "grid_validation_gap": result.grid_validation_gap,
        "status": result.status,
        "manifest_hash_bound": bool(
            result.manifest.get("contractions_manifest_hash")
            == canonical_sha256(contractions.manifest)
        ),
        "optimizer_reads_individual_data": result.manifest.get(
            "optimizer_reads_individual_data"
        ),
    }


def _benchmark_crossfit(
    fixture: Mapping[str, Any], *, args: argparse.Namespace
) -> dict[str, Any]:
    common = _builder_common(fixture)
    contractions, build_seconds, build_peak = _timed(
        lambda: build_context_direction_crossfit_contractions(
            fixture["reference_genotype"],
            fixture["reference_context"],
            fixture["reference_projector"],
            fixture["study_genotype"],
            fixture["study_context"],
            fixture["study_projector"],
            fixture["phenotype"],
            fixture["block_ids"],
            reference_fixed_effect_hash=fixture["reference_fixed_effect_hash"],
            study_fixed_effect_hash=fixture["study_fixed_effect_hash"],
            **common,
        )
    )
    validation = validate_direction_crossfit_contractions(contractions)
    result, fit_seconds, fit_peak = _timed(
        lambda: crossfit_context_direction(
            contractions,
            objective=args.objective,
            validation_grid_size=args.crossfit_grid_size,
            maxiter=args.optimizer_maxiter,
        )
    )
    assignment = contractions.assignment
    left_blocks = set(assignment.fold_block_identities[0])
    right_blocks = set(assignment.fold_block_identities[1])
    arm0, arm1 = contractions.arms
    exact_partition = bool(
        left_blocks.isdisjoint(right_blocks)
        and left_blocks | right_blocks == set(assignment.block_identities)
        and sum(assignment.fold_variant_counts) == fixture["study_genotype"].shape[1]
        and arm0.train_variant_hash == arm1.heldout_variant_hash
        and arm0.heldout_variant_hash == arm1.train_variant_hash
        and arm0.train_variant_hash != arm0.heldout_variant_hash
    )
    direction_drifts = [
        float(np.max(np.abs(fold.direction - fold.heldout_evaluation.direction)))
        for fold in result.folds
    ]
    fixed_heldout = all(drift <= 1.0e-12 for drift in direction_drifts)
    inventory = _tensor_inventory(contractions)
    combined_expected = _expected_compact_tensor_bytes(
        len(arm0.train.environment_names)
    )
    return {
        "build_wall_seconds": build_seconds,
        "fit_wall_seconds": fit_seconds,
        "absolute_process_high_water_bytes": max(build_peak, fit_peak),
        "tensor_inventory": inventory,
        "expected_unique_fold_tensor_bytes": 2 * combined_expected,
        "tensor_bytes_match_two_unique_folds": bool(
            inventory["total_array_bytes"] == 2 * combined_expected
        ),
        "manifest_json_bytes": _manifest_bytes(contractions),
        "validation": validation,
        "fold_variant_counts": list(assignment.fold_variant_counts),
        "fold_weight_masses": list(assignment.fold_weight_masses),
        "assignment_hash_length": len(assignment.assignment_hash),
        "exact_disjoint_complete_reverse_partition": exact_partition,
        "construction": contractions.manifest.get("construction"),
        "uses_approximate_loo_deletion": contractions.manifest.get(
            "uses_approximate_loo_deletion"
        ),
        "contains_individual_arrays_manifest": contractions.manifest.get(
            "contains_individual_arrays"
        ),
        "maximum_axis_below_n_and_m": bool(
            inventory["maximum_axis"]
            < min(
                fixture["study_genotype"].shape[0],
                fixture["study_genotype"].shape[1],
            )
        ),
        "objective": result.objective_name,
        "combined_heldout_value": float(result.combined_heldout_value),
        "fold_direction_alignment": float(result.fold_direction_alignment),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "training_objective": float(fold.training_objective),
                "heldout_objective": float(fold.heldout_objective),
                "direction": fold.direction.tolist(),
                "training_evaluations": int(fold.training_optimization.evaluations),
                "train_and_heldout_hashes_differ": bool(
                    fold.train_variant_hash != fold.heldout_variant_hash
                ),
            }
            for fold in result.folds
        ],
        "heldout_evaluates_fixed_training_direction": fixed_heldout,
        "maximum_heldout_direction_renormalization_drift": max(direction_drifts),
        "heldout_direction_identity_tolerance": 1.0e-12,
        "in_sample_objective_is_unbiased": result.manifest.get(
            "in_sample_objective_is_unbiased"
        ),
        "nonlinear_selection_inference": result.manifest.get(
            "nonlinear_selection_inference"
        ),
        "status": result.status,
    }


def _scaling_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for l_count in sorted({int(record["l"]) for record in records}):
        selected = sorted(
            (record for record in records if record["l"] == l_count),
            key=lambda value: value["builder_workload_proxy_n2m"],
        )
        tensor_bytes = [
            value["compact_payload"]["tensor_inventory"]["total_array_bytes"]
            for value in selected
        ]
        evaluation_times = [
            value["compact_evaluation"]["microseconds_per_call_median"]
            for value in selected
        ]
        result[f"l{l_count}"] = {
            "tensor_bytes": tensor_bytes,
            "tensor_bytes_identical_across_n_m": bool(len(set(tensor_bytes)) == 1),
            "evaluation_microseconds_per_call": evaluation_times,
            "evaluation_time_max_over_min_descriptive": float(
                max(evaluation_times) / min(evaluation_times)
            ),
            "interpretation": (
                "compact tensor dimensions are independent of N/M; timing ratio "
                "is descriptive and not an asymptotic gate"
            ),
        }
    return result


def _plot(
    records: Sequence[Mapping[str, Any]],
    optimizers: Mapping[str, Mapping[str, Any]],
    crossfit: Mapping[str, Any],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), constrained_layout=True)
    colors = {1: "#6b7280", 2: "#1f77b4", 3: "#d97706"}
    for l_count in sorted({int(record["l"]) for record in records}):
        selected = sorted(
            (record for record in records if record["l"] == l_count),
            key=lambda value: value["builder_workload_proxy_n2m"],
        )
        workload = np.asarray(
            [value["builder_workload_proxy_n2m"] for value in selected],
            dtype=np.float64,
        )
        workload /= workload[0]
        reference_time = [
            value["builder_runtime_seconds"]["reference"]["wall_median"]
            for value in selected
        ]
        trait_time = [
            value["builder_runtime_seconds"]["trait"]["wall_median"]
            for value in selected
        ]
        evaluation_time = [
            value["compact_evaluation"]["microseconds_per_call_median"]
            for value in selected
        ]
        tensor_bytes = [
            value["compact_payload"]["tensor_inventory"]["total_array_bytes"]
            for value in selected
        ]
        color = colors[l_count]
        axes[0, 0].plot(
            workload,
            reference_time,
            "o-",
            color=color,
            label=f"L={l_count} reference",
        )
        axes[0, 0].plot(
            workload,
            trait_time,
            "s--",
            color=color,
            label=f"L={l_count} trait",
        )
        axes[0, 1].plot(
            workload,
            evaluation_time,
            "o-",
            color=color,
            label=f"L={l_count}",
        )
        axes[1, 0].plot(
            workload,
            tensor_bytes,
            "o-",
            color=color,
            label=f"L={l_count}",
        )

    optimizer_labels = sorted(optimizers)
    optimizer_times = [optimizers[label]["wall_seconds"] for label in optimizer_labels]
    axes[1, 1].bar(
        np.arange(len(optimizer_labels) + 1),
        optimizer_times + [crossfit["fit_wall_seconds"]],
        color=[colors[int(label.removeprefix("l"))] for label in optimizer_labels]
        + ["#4c956c"],
    )
    axes[1, 1].set_xticks(
        np.arange(len(optimizer_labels) + 1),
        [f"opt {label.upper()}" for label in optimizer_labels] + ["2-fold fit"],
    )

    axes[0, 0].set_ylabel("Median builder wall time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set_ylabel("Compact evaluation (µs/call)")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set_ylabel("Compact tensor bytes")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].set_ylabel("Wall time (s)")
    for axis in axes[:1, :].flat:
        axis.set_xlabel(r"Builder workload proxy $N^2M$ (relative)")
    axes[1, 0].set_xlabel(r"Builder workload proxy $N^2M$ (relative)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle("Compact context-direction evaluation versus raw-data builders")
    for suffix in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        path.chmod(0o600)
    plt.close(figure)


def main() -> None:
    os.umask(0o077)
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    output_paths = tuple(
        args.output_dir / f"{OUTPUT_STEM}.{suffix}" for suffix in ("json", "png", "pdf")
    )
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "benchmark outputs already exist; choose a fresh --output-dir or pass "
            f"--overwrite: {existing}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)

    process = psutil.Process()
    started_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    live: dict[tuple[int, int, int], dict[str, Any]] = {}
    for l_count in args.l_values:
        for n_samples, n_variants in args.builder_shapes:
            record, objects = _build_case(
                l_count=l_count,
                n_samples=n_samples,
                n_variants=n_variants,
                args=args,
            )
            record["compact_evaluation"] = _benchmark_evaluations(
                objects["contractions"],
                calls=args.evaluation_calls,
                repeats=args.evaluation_repeats,
                pool_size=args.direction_pool_size,
                objective=args.objective,
                seed=args.seed + 10_000 * l_count + n_samples + n_variants,
            )
            records.append(record)
            live[(l_count, n_samples, n_variants)] = objects

    largest_n, largest_m = max(
        args.builder_shapes, key=lambda value: value[0] ** 2 * value[1]
    )
    optimizers = {
        f"l{l_count}": _benchmark_optimizer(
            live[(l_count, largest_n, largest_m)]["contractions"], args=args
        )
        for l_count in args.l_values
    }
    crossfit_l = min(args.l_values)
    crossfit = _benchmark_crossfit(
        live[(crossfit_l, largest_n, largest_m)]["fixture"], args=args
    )
    scaling = _scaling_summary(records)

    gate = {
        "all_compact_tensor_bytes_match_formula": all(
            record["compact_payload"]["tensor_bytes_match_formula"]
            for record in records
        ),
        "all_compact_axes_below_n_and_m": all(
            record["compact_payload"]["maximum_axis_below_n_and_m"]
            for record in records
        ),
        "all_manifest_bindings_match": all(
            record["binding_audit"]["manifest_hashes_match"]
            and record["binding_audit"]["same_selected_variant_hash"]
            for record in records
        ),
        "all_evaluation_calls_identifiable": all(
            not any(record["compact_evaluation"]["failed_evaluations"])
            for record in records
        ),
        "compact_tensor_bytes_independent_of_n_m": all(
            value["tensor_bytes_identical_across_n_m"] for value in scaling.values()
        ),
        "all_optimizers_metric_normalized_and_manifest_bound": all(
            abs(value["metric_norm_squared"] - 1.0) <= 1.0e-10
            and value["manifest_hash_bound"]
            and value["optimizer_reads_individual_data"] is False
            for value in optimizers.values()
        ),
        "crossfit_exact_disjoint_complete_reverse_partition": crossfit[
            "exact_disjoint_complete_reverse_partition"
        ],
        "crossfit_tensor_payload_is_two_unique_compact_folds": crossfit[
            "tensor_bytes_match_two_unique_folds"
        ],
        "crossfit_has_no_individual_axes_or_manifest_claim": bool(
            crossfit["maximum_axis_below_n_and_m"]
            and crossfit["contains_individual_arrays_manifest"] is False
        ),
        "crossfit_uses_no_approximate_loo": bool(
            crossfit["uses_approximate_loo_deletion"] is False
        ),
        "crossfit_heldout_uses_fixed_training_directions": crossfit[
            "heldout_evaluates_fixed_training_direction"
        ],
        "crossfit_does_not_label_training_objective_unbiased": bool(
            crossfit["in_sample_objective_is_unbiased"] is False
        ),
    }
    verdict = "pass" if all(gate.values()) else "review"
    payload = {
        "kind": "summit.context.direction_benchmark",
        "schema_version": 1,
        "experimental": True,
        "seed": args.seed,
        "configuration": {
            "builder_shapes": [list(value) for value in args.builder_shapes],
            "l_values": list(args.l_values),
            "builder_warmups": args.builder_warmups,
            "builder_repeats": args.builder_repeats,
            "evaluation_calls_per_sample": args.evaluation_calls,
            "evaluation_repeats": args.evaluation_repeats,
            "direction_pool_size": args.direction_pool_size,
            "optimizer_grid_size": args.optimizer_grid_size,
            "crossfit_grid_size": args.crossfit_grid_size,
            "optimizer_maxiter": args.optimizer_maxiter,
            "blocks": args.blocks,
            "objective": args.objective,
            "genotype_scaling": GENOTYPE_SCALING,
            "blas_threads": "inherited_runtime_configuration",
        },
        "privacy": {
            "fixture": "deterministic_synthetic",
            "individual_or_variant_rows_persisted": False,
            "outputs": "aggregate_metrics_hashes_and_figures_only",
        },
        "measurement_semantics": {
            "compact_evaluation": (
                "evaluate_context_direction only; fixture generation and direction "
                "pool construction excluded"
            ),
            "source_peak_rss_field": (
                "direction builders currently sample process RSS at return; benchmark "
                "also records the absolute OS process high-water mark"
            ),
            "rss": "absolute_process_values_not_incremental_allocations",
            "builder_workload": (
                "N^2*M proxy for genotype-dependent dense development work"
            ),
            "timing_gate": (
                "evaluation timing ratios are descriptive; structural tensor size "
                "and absence of N/M arrays are hard gates"
            ),
        },
        "resource_and_seam_review": {
            "compact_contract": (
                "reference/trait tensors are O(P^2) with P determined only by L"
            ),
            "hash_binding": (
                "array hashes are revalidated and the combined manifest binds both "
                "reference and trait manifests plus the shared specification"
            ),
            "crossfit_contract": (
                "whole blocks are assigned once, exact fold subsets are rebuilt, "
                "reverse arms exchange train/held-out hashes, and held-out evaluation "
                "uses the fixed training direction"
            ),
            "scope": (
                "experimental descriptive optimizer; nonlinear selection inference "
                "is not calibrated and training objectives are not unbiased estimates"
            ),
        },
        "runtime_seconds_before_output": float(time.perf_counter() - started),
        "absolute_process_rss_bytes": {
            "before_cases": started_rss,
            "at_payload": int(process.memory_info().rss),
            "high_water_at_payload": _peak_rss_bytes(),
        },
        "scaling": scaling,
        "optimizers": optimizers,
        "crossfit": crossfit,
        "aggregate_gate": gate,
        "verdict": verdict,
        "records": records,
    }
    output_paths[0].write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    output_paths[0].chmod(0o600)
    _plot(records, optimizers, crossfit, args.output_dir)
    if verdict != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
