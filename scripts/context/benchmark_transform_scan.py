#!/usr/bin/env python3
"""Benchmark batched transformed-trait summaries against separate runs.

The fixture is deterministic and entirely synthetic.  Only aggregate timings,
memory measurements, dimensions, and numerical discrepancies are written; no
phenotype, genotype, context, or variant rows are persisted.
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
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    PhenotypeTransformSpec,
    array_sha256,
    build_context_trait_summary,
    build_context_transform_summary,
    canonical_sha256,
    rank_revealing_projector,
)


OUTPUT_STEM = "06_transform_scan_benchmark"
PARITY_TOLERANCE = 5.0e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=96, help="Synthetic sample count.")
    parser.add_argument("--m", type=int, default=160, help="Synthetic variant count.")
    parser.add_argument(
        "--q-values",
        default="2,3,4",
        help="Comma-separated context-basis dimensions (supported: 1 through 4).",
    )
    parser.add_argument(
        "--l-values",
        default="1,2,4,8",
        help="Comma-separated valid transformation counts (maximum 8).",
    )
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--transform-tile-size", type=int, default=4)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260819)
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
        parser.error(f"{name} must contain at least one integer")
    if len(set(values)) != len(values):
        parser.error(f"{name} must not contain duplicates")
    return values


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    args.q_values = _integer_list(parser, args.q_values, "--q-values")
    args.l_values = _integer_list(parser, args.l_values, "--l-values")
    if args.n < 12:
        parser.error("--n must be at least 12")
    if args.m < 8:
        parser.error("--m must be at least 8")
    if any(value < 1 or value > 4 for value in args.q_values):
        parser.error("every --q-values entry must lie between 1 and 4")
    if any(value < 1 or value > 8 for value in args.l_values):
        parser.error("every --l-values entry must lie between 1 and 8")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.transform_tile_size < 1:
        parser.error("--transform-tile-size must be positive")
    if args.loo_groups < 2 or args.loo_groups > args.m:
        parser.error("--loo-groups must satisfy 2 <= groups <= M")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _timed(function: Callable[[], Any]) -> tuple[Any, float, int]:
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    return result, float(elapsed), _peak_rss_bytes()


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(
        values, axis=0, keepdims=True, dtype=np.float64
    )
    scales = np.std(centered, axis=0, ddof=1)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise RuntimeError("Synthetic fixture generated an invalid column scale.")
    return np.ascontiguousarray(centered / scales[None, :], dtype=np.float64)


def _transformation_pool() -> tuple[PhenotypeTransformSpec, ...]:
    return (
        PhenotypeTransformSpec("identity", "identity"),
        PhenotypeTransformSpec("log", "log", shift=0.0),
        PhenotypeTransformSpec(
            "box_cox_m075", "box_cox", shift=0.0, box_cox_lambda=-0.75
        ),
        PhenotypeTransformSpec(
            "box_cox_m025", "box_cox", shift=0.0, box_cox_lambda=-0.25
        ),
        PhenotypeTransformSpec(
            "box_cox_p025", "box_cox", shift=0.0, box_cox_lambda=0.25
        ),
        PhenotypeTransformSpec(
            "box_cox_p050", "box_cox", shift=0.0, box_cox_lambda=0.50
        ),
        PhenotypeTransformSpec(
            "box_cox_p100", "box_cox", shift=0.0, box_cox_lambda=1.0
        ),
        PhenotypeTransformSpec("user_asinh", "user_supplied", source="user_asinh"),
    )


def _fixture(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    genotype = _standardize_columns(rng.normal(size=(args.n, args.m)))
    context = _standardize_columns(rng.normal(size=(args.n, 3)))
    raw_basis = np.column_stack([np.ones(args.n, dtype=np.float64), context]).astype(
        np.float64, copy=False
    )
    nuisance = _standardize_columns(rng.normal(size=(args.n, 1)))

    latent_trait = (
        0.35 * context[:, 0]
        - 0.15 * context[:, 1]
        + 0.20 * genotype[:, 0]
        + rng.normal(scale=0.55, size=args.n)
    )
    original = np.exp(latent_trait) + 0.25
    user_asinh = np.arcsinh(original)
    annotations = np.ones((args.m, 1), dtype=np.float64)
    groups = tuple(f"group:{index % args.loo_groups}" for index in range(args.m))
    return {
        "genotype": genotype,
        "raw_basis": np.ascontiguousarray(raw_basis),
        "nuisance": nuisance,
        "original": np.asarray(original, dtype=np.float64),
        "user_transforms": {"user_asinh": np.asarray(user_asinh, dtype=np.float64)},
        "annotations": annotations,
        "groups": groups,
        "transformations": _transformation_pool(),
        "variant_hash": canonical_sha256(
            {"ordered_synthetic_variants": list(range(args.m))}
        ),
    }


def _materialize_transform(
    spec: PhenotypeTransformSpec,
    original: np.ndarray,
    user_transforms: dict[str, np.ndarray],
) -> np.ndarray:
    """Independent realization used by the existing single-trait builder."""
    if spec.kind == "identity":
        result = original.copy()
    elif spec.kind == "user_supplied":
        assert spec.source is not None
        result = np.asarray(user_transforms[spec.source], dtype=np.float64)
    else:
        shifted = original + float(spec.shift)
        if spec.kind == "log" or abs(float(spec.box_cox_lambda or 0.0)) < 1.0e-12:
            result = np.log(shifted)
        else:
            transform_lambda = float(spec.box_cox_lambda)
            result = np.expm1(transform_lambda * np.log(shifted)) / transform_lambda
    if result.shape != original.shape or not np.all(np.isfinite(result)):
        raise RuntimeError(f"Independent transform {spec.transform_id!r} is invalid.")
    return np.asarray(result, dtype=np.float64)


def _numeric_payload_bytes(value: Any) -> int:
    names = (
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
    return int(sum(np.asarray(getattr(value, name)).nbytes for name in names))


def _discrepancy(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise RuntimeError(
            f"Parity arrays have different shapes {left_array.shape} and "
            f"{right_array.shape}."
        )
    maximum_absolute = float(np.max(np.abs(left_array - right_array), initial=0.0))
    scale = max(
        float(np.max(np.abs(left_array), initial=0.0)),
        float(np.max(np.abs(right_array), initial=0.0)),
        1.0,
    )
    return {
        "exactly_equal": bool(np.array_equal(left_array, right_array)),
        "maximum_absolute_error": maximum_absolute,
        "scale_aware_error": maximum_absolute / scale,
    }


def _parity(
    batch: Any, separate: Sequence[Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "genetic_rhs": (
            batch.genetic_rhs,
            np.stack([summary.genetic_rhs for summary in separate]),
        ),
        "residual_rhs": (
            batch.residual_rhs,
            np.stack([summary.residual_rhs for summary in separate]),
        ),
        "rhs_numerator_contributions": (
            batch.rhs_numerator_contributions,
            np.stack(
                [summary.rhs_numerator_contributions for summary in separate],
                axis=1,
            ),
        ),
    }
    shared_names = (
        "annotation_weights",
        "annotation_masses",
        "genetic_traces",
        "genetic_residual",
        "residual_traces",
        "residual_gram",
        "trace_numerator_contributions",
        "genetic_residual_numerator_contributions",
    )
    for name in shared_names:
        fields[name] = (
            np.repeat(
                np.asarray(getattr(batch, name))[None, ...], len(separate), axis=0
            ),
            np.stack([np.asarray(getattr(summary, name)) for summary in separate]),
        )
    discrepancies = {
        name: _discrepancy(left, right) for name, (left, right) in fields.items()
    }
    maximum_absolute = max(
        value["maximum_absolute_error"] for value in discrepancies.values()
    )
    maximum_scaled = max(value["scale_aware_error"] for value in discrepancies.values())
    aggregate = {
        "all_fields_exactly_equal": all(
            value["exactly_equal"] for value in discrepancies.values()
        ),
        "exactly_equal_field_count": sum(
            int(value["exactly_equal"]) for value in discrepancies.values()
        ),
        "field_count": len(discrepancies),
        "maximum_absolute_error": float(maximum_absolute),
        "maximum_scale_aware_error": float(maximum_scaled),
        "within_declared_tolerance": bool(maximum_scaled <= PARITY_TOLERANCE),
        "declared_scale_aware_tolerance": PARITY_TOLERANCE,
    }
    return discrepancies, aggregate


def _case(
    *,
    fixture: dict[str, Any],
    q_count: int,
    l_count: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    basis = np.asarray(fixture["raw_basis"][:, :q_count], dtype=np.float64)
    fixed = np.column_stack(
        [np.ones(args.n, dtype=np.float64), basis[:, 1:], fixture["nuisance"]]
    )
    projector = rank_revealing_projector(fixed)
    components = ContextComponentIndex(("all",), ContextPairIndex(q_count))
    residual_basis = np.column_stack(
        [
            np.ones(args.n, dtype=np.float64),
            0.5 + np.square(fixture["raw_basis"][:, 1]),
        ]
    )
    residual_names = ("residual:constant", "residual:context_squared")
    transformations = fixture["transformations"][:l_count]
    common = {
        "genotype": fixture["genotype"],
        "basis": basis,
        "projector": projector,
        "annotations": fixture["annotations"],
        "component_index": components,
        "residual_basis": residual_basis,
        "residual_names": residual_names,
        "basis_hash": array_sha256(basis),
        "fixed_effect_hash": array_sha256(fixed),
        "variant_hash": fixture["variant_hash"],
        "loo_groups": fixture["groups"],
        "genotype_scaling": "sample_sd_ddof1_pre_scaled_input",
        "block_size": args.block_size,
    }

    batch, batch_seconds, batch_process_peak = _timed(
        lambda: build_context_transform_summary(
            **common,
            original_phenotype=fixture["original"],
            transformations=transformations,
            original_trait="synthetic_positive_trait",
            user_transforms=fixture["user_transforms"],
            transform_tile_size=args.transform_tile_size,
        )
    )
    expected_ids = tuple(spec.transform_id for spec in transformations)
    if batch.transform_ids != expected_ids or batch.invalid_transform_ids:
        raise RuntimeError("The benchmark transformation grid was not wholly valid.")

    separate: list[Any] = []
    separate_seconds: list[float] = []
    separate_process_peaks: list[int] = []
    for spec in transformations:

        def build_one(transform: PhenotypeTransformSpec = spec) -> Any:
            phenotype = _materialize_transform(
                transform, fixture["original"], fixture["user_transforms"]
            )
            return build_context_trait_summary(
                **common,
                phenotype=phenotype,
            )

        summary, seconds, process_peak = _timed(build_one)
        separate.append(summary)
        separate_seconds.append(seconds)
        separate_process_peaks.append(process_peak)

    discrepancies, aggregate = _parity(batch, separate)
    expected_blocks = math.ceil(args.m / args.block_size)
    rhs_bound = q_count * min(l_count, args.transform_tile_size)
    if batch.decoded_blocks != expected_blocks:
        raise RuntimeError(
            "Batched transform summary decoded an unexpected block count."
        )
    batch_reported_peak = int(batch.peak_rss_bytes)
    separate_reported_peak = max(int(summary.peak_rss_bytes) for summary in separate)
    separate_total_seconds = float(sum(separate_seconds))
    speedup = separate_total_seconds / batch_seconds

    return {
        "q": q_count,
        "l": l_count,
        "p_genetic": len(components),
        "fixed_effect_rank": projector.rank,
        "residual_rank": projector.residual_rank,
        "transform_ids": list(expected_ids),
        "transform_kinds": [spec.kind for spec in transformations],
        "runtime_seconds": {
            "batched_wall": batch_seconds,
            "separate_wall_sum": separate_total_seconds,
            "separate_wall_each": separate_seconds,
            "speedup_separate_over_batched": speedup,
            "batched_internal_phases": {
                name: float(value) for name, value in batch.phase_times_seconds.items()
            },
            "separate_internal_phase_sums": {
                phase: float(
                    sum(
                        summary.phase_times_seconds.get(phase, 0.0)
                        for summary in separate
                    )
                )
                for phase in (
                    "phenotype_and_residual",
                    "genotype_pass",
                    "total",
                )
            },
        },
        "absolute_peak_rss_bytes": {
            "batched_builder_reported": batch_reported_peak,
            "separate_max_builder_reported": separate_reported_peak,
            "after_batched_process_high_water": batch_process_peak,
            "after_separate_process_high_water": max(separate_process_peaks),
        },
        "numeric_summary_payload_bytes": {
            "batched": _numeric_payload_bytes(batch),
            "separate_sum": int(sum(_numeric_payload_bytes(item) for item in separate)),
        },
        "decode": {
            "blocks_per_pass": expected_blocks,
            "batched_passes": int(batch.decode_passes),
            "batched_blocks": int(batch.decoded_blocks),
            "separate_passes_sum": int(sum(item.decode_passes for item in separate)),
            "separate_blocks_sum": int(sum(item.decoded_blocks for item in separate)),
            "one_batched_decode_per_block": bool(
                batch.decode_passes == 1 and batch.decoded_blocks == expected_blocks
            ),
        },
        "rhs_tiling": {
            "transform_tile_size": args.transform_tile_size,
            "observed_maximum_columns": int(batch.maximum_rhs_columns),
            "declared_column_bound": rhs_bound,
            "bound_holds": bool(batch.maximum_rhs_columns <= rhs_bound),
            "observed_rhs_matrix_bytes": int(8 * args.n * batch.maximum_rhs_columns),
            "declared_rhs_matrix_bound_bytes": int(8 * args.n * rhs_bound),
            "separate_maximum_columns": q_count,
        },
        "parity": aggregate,
        "parity_by_field": discrepancies,
    }


def _plot(
    records: Sequence[dict[str, Any]], output_dir: Path, args: argparse.Namespace
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), constrained_layout=True)
    colors = ("#1f6f8b", "#d97706", "#4c956c", "#8f5aa2")
    for color, q_count in zip(colors, args.q_values):
        selected = sorted(
            (record for record in records if record["q"] == q_count),
            key=lambda value: value["l"],
        )
        l_values = np.asarray([record["l"] for record in selected], dtype=int)
        batch_time = [record["runtime_seconds"]["batched_wall"] for record in selected]
        separate_time = [
            record["runtime_seconds"]["separate_wall_sum"] for record in selected
        ]
        speedup = [
            record["runtime_seconds"]["speedup_separate_over_batched"]
            for record in selected
        ]
        batch_rss = [
            record["absolute_peak_rss_bytes"]["batched_builder_reported"] / 2**20
            for record in selected
        ]
        separate_rss = [
            record["absolute_peak_rss_bytes"]["separate_max_builder_reported"] / 2**20
            for record in selected
        ]
        batch_blocks = [record["decode"]["batched_blocks"] for record in selected]
        separate_blocks = [
            record["decode"]["separate_blocks_sum"] for record in selected
        ]
        parity = [
            max(
                record["parity"]["maximum_scale_aware_error"],
                np.finfo(np.float64).eps,
            )
            for record in selected
        ]
        rhs_columns = [
            record["rhs_tiling"]["observed_maximum_columns"] for record in selected
        ]
        rhs_bounds = [
            record["rhs_tiling"]["declared_column_bound"] for record in selected
        ]

        axes[0, 0].plot(
            l_values, batch_time, "o-", color=color, label=f"Q={q_count} batch"
        )
        axes[0, 0].plot(
            l_values, separate_time, "x--", color=color, label=f"Q={q_count} separate"
        )
        axes[0, 1].plot(l_values, speedup, "o-", color=color, label=f"Q={q_count}")
        axes[0, 2].plot(l_values, batch_rss, "o-", color=color)
        axes[0, 2].plot(l_values, separate_rss, "x--", color=color)
        axes[1, 0].plot(l_values, batch_blocks, "o-", color=color)
        axes[1, 0].plot(l_values, separate_blocks, "x--", color=color)
        axes[1, 1].plot(l_values, parity, "o-", color=color, label=f"Q={q_count}")
        axes[1, 2].plot(l_values, rhs_columns, "o-", color=color)
        axes[1, 2].plot(l_values, rhs_bounds, ":", color=color)

    axes[0, 0].set_ylabel("Wall time (s)")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[0, 1].axhline(1.0, color="0.4", linewidth=0.8, linestyle=":")
    axes[0, 1].set_ylabel("Speedup (separate / batch)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 2].set_ylabel("Absolute reported RSS (MiB)")
    axes[0, 2].text(
        0.02,
        0.98,
        "solid: batch\ndashed: separate",
        transform=axes[0, 2].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1, 0].set_ylabel("Decoded block visits")
    axes[1, 0].text(
        0.02,
        0.98,
        "solid: batch\ndashed: separate",
        transform=axes[1, 0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[1, 1].axhline(PARITY_TOLERANCE, color="0.4", linewidth=0.8, linestyle=":")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Maximum scale-aware error")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 2].set_ylabel("Maximum RHS columns")
    axes[1, 2].text(
        0.02,
        0.98,
        "solid: observed\ndotted: bound",
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=8,
    )
    for axis in axes.flat:
        axis.set_xlabel("Transformations L")
        axis.set_xticks(args.l_values)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Batched contextual transformation summaries "
        f"(N={args.n}, M={args.m}, tile={args.transform_tile_size})"
    )
    figure.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300)
    figure.savefig(output_dir / f"{OUTPUT_STEM}.pdf")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    setup_start_rss = int(process.memory_info().rss)
    fixture, setup_seconds, setup_peak = _timed(lambda: _fixture(args))
    records = [
        _case(
            fixture=fixture,
            q_count=q_count,
            l_count=l_count,
            args=args,
        )
        for q_count in args.q_values
        for l_count in args.l_values
    ]

    maximum_scaled_error = max(
        record["parity"]["maximum_scale_aware_error"] for record in records
    )
    aggregate_gate = {
        "maximum_scale_aware_parity_error": maximum_scaled_error,
        "all_parity_within_tolerance": all(
            record["parity"]["within_declared_tolerance"] for record in records
        ),
        "all_one_batched_decode_per_block": all(
            record["decode"]["one_batched_decode_per_block"] for record in records
        ),
        "all_rhs_bounds_hold": all(
            record["rhs_tiling"]["bound_holds"] for record in records
        ),
    }
    payload = {
        "kind": "summit.context.transform_scan_benchmark",
        "schema_version": 1,
        "seed": args.seed,
        "configuration": {
            "n": args.n,
            "m": args.m,
            "k": 1,
            "q_values": list(args.q_values),
            "l_values": list(args.l_values),
            "block_size": args.block_size,
            "transform_tile_size": args.transform_tile_size,
            "loo_groups": args.loo_groups,
            "blas_threads": "inherited_runtime_configuration",
            "genotype_scaling": "sample_sd_ddof1",
        },
        "privacy": {
            "fixture": "deterministic_synthetic",
            "individual_or_variant_rows_persisted": False,
            "outputs": "aggregate_metrics_and_figures_only",
        },
        "fixture_setup_seconds": setup_seconds,
        "resident_memory_bytes": {
            "before_fixture": setup_start_rss,
            "absolute_process_peak_after_fixture": setup_peak,
            "absolute_process_peak_at_completion": _peak_rss_bytes(),
        },
        "aggregate_gate": aggregate_gate,
        "verdict": (
            "pass"
            if all(
                value
                for key, value in aggregate_gate.items()
                if key != "maximum_scale_aware_parity_error"
            )
            else "review"
        ),
        "records": records,
    }
    output_paths[0].write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(records, args.output_dir, args)

    del records, fixture
    gc.collect()
    if payload["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
