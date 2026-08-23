#!/usr/bin/env python3
"""Benchmark fixed multi-environment reference and trait-summary construction.

The deterministic fixture uses an independently sampled reference and study
cohort but one frozen reference calibration.  For Q=2,3,4 it reports warmup
and repeated timings, absolute process RSS, aggregate payload sizes, and dense
oracle discrepancies.  The operation accounting deliberately separates the
Q-linear genotype projection/score products from the P_g-by-P_g exact moment
reductions in the correctness-first Python backend.

Only aggregate JSON and figures are written; no individual or variant rows are
persisted.
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
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from summit.context import (
    MultiEnvironmentSourceSpec,
    apply_multienvironment_calibration,
    build_context_reference,
    build_context_trait_summary,
    calibrate_multienvironment_basis,
    canonical_sha256,
    common_scale_features,
    dense_genetic_kernels,
    dense_residual_kernels,
    exact_same_person_matrix,
    kernel_gram,
    kernel_rhs,
    kernel_traces,
    project_normalize_phenotype,
)


OUTPUT_STEM = "07a_multienvironment_benchmark"
PARITY_TOLERANCE = 5.0e-11
GENOTYPE_SCALING = "sample_sd_ddof1_pre_scaled_input"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-n", type=int, default=96)
    parser.add_argument("--study-n", type=int, default=96)
    parser.add_argument("--m", type=int, default=120)
    parser.add_argument(
        "--q-values",
        default="2,3,4",
        help="Comma-separated basis dimensions; every value must lie in [2,4].",
    )
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--product-inner-loops",
        type=int,
        default=3,
        help="Loops per isolated Q-linear genotype-product timing sample.",
    )
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
    if args.reference_n < 12 or args.study_n < 12:
        parser.error("reference and study N must each be at least 12")
    if args.m < 8:
        parser.error("--m must be at least 8")
    if any(value < 2 or value > 4 for value in args.q_values):
        parser.error("every --q-values entry must lie between 2 and 4")
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.loo_groups < 2 or args.loo_groups > args.m:
        parser.error("--loo-groups must satisfy 2 <= groups <= M")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.product_inner_loops < 1:
        parser.error("--product-inner-loops must be positive")
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


def _sample_sources(
    rng: np.random.Generator, *, n_samples: int, source_count: int
) -> dict[str, np.ndarray]:
    shared = rng.standard_normal((n_samples, 1))
    independent = rng.standard_normal((n_samples, source_count))
    raw = math.sqrt(0.25) * shared + math.sqrt(0.75) * independent
    return {
        f"environment_{index + 1}": np.asarray(raw[:, index], dtype=np.float64)
        for index in range(source_count)
    }


def _sample_fixture(q_count: int, args: argparse.Namespace) -> dict[str, Any]:
    seed = np.random.SeedSequence([args.seed, q_count])
    ref_seed, study_seed = seed.spawn(2)
    reference_rng = np.random.default_rng(ref_seed)
    study_rng = np.random.default_rng(study_seed)
    reference_sources = _sample_sources(
        reference_rng, n_samples=args.reference_n, source_count=q_count - 1
    )
    study_sources = _sample_sources(
        study_rng, n_samples=args.study_n, source_count=q_count - 1
    )
    reference_genotype = _standardize_columns(
        reference_rng.standard_normal((args.reference_n, args.m))
    )
    study_genotype = _standardize_columns(
        study_rng.standard_normal((args.study_n, args.m))
    )
    context_signal = sum(
        ((-1.0) ** index) * (0.12 / index) * values
        for index, values in enumerate(study_sources.values(), start=1)
    )
    phenotype = (
        context_signal
        + 0.16 * study_genotype[:, 0]
        - 0.09 * study_genotype[:, 1]
        + study_rng.normal(scale=0.75, size=args.study_n)
    )
    annotations = np.ones((args.m, 1), dtype=np.float64)
    loo_groups = tuple(f"group:{index % args.loo_groups}" for index in range(args.m))
    specs = tuple(
        MultiEnvironmentSourceSpec(f"environment_{index}", "continuous")
        for index in range(1, q_count)
    )
    return {
        "reference_sources": reference_sources,
        "study_sources": study_sources,
        "reference_mask": np.ones(args.reference_n, dtype=bool),
        "study_mask": np.ones(args.study_n, dtype=bool),
        "reference_genotype": reference_genotype,
        "study_genotype": study_genotype,
        "phenotype": np.asarray(phenotype, dtype=np.float64),
        "annotations": annotations,
        "loo_groups": loo_groups,
        "source_specs": specs,
        "variant_hash": canonical_sha256(
            {"ordered_synthetic_variants": list(range(args.m))}
        ),
    }


def _numeric_payload_bytes(value: Any, names: Sequence[str]) -> int:
    return int(sum(np.asarray(getattr(value, name)).nbytes for name in names))


def _discrepancy(observed: object, expected: object) -> dict[str, float]:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError(
            f"Dense parity arrays have different shapes {left.shape} and {right.shape}."
        )
    maximum_absolute = float(np.max(np.abs(left - right), initial=0.0))
    scale = max(
        float(np.max(np.abs(left), initial=0.0)),
        float(np.max(np.abs(right), initial=0.0)),
        1.0,
    )
    return {
        "maximum_absolute_error": maximum_absolute,
        "scale_aware_error": maximum_absolute / scale,
    }


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
        value, seconds, process_peak = _timed(function)
        values.append(value)
        wall_times.append(seconds)
        process_peaks.append(process_peak)
        gc.collect()
    result = values[-1]
    telemetry = {
        "wall_samples": wall_times,
        "wall_median": float(np.median(wall_times)),
        "internal_phase_medians": _phase_medians(values),
        "builder_absolute_peak_rss_samples": [
            int(value.peak_rss_bytes) for value in values
        ],
        "builder_absolute_peak_rss_median": int(
            np.median([value.peak_rss_bytes for value in values])
        ),
        "process_absolute_high_water_samples": process_peaks,
        "process_absolute_high_water_maximum": max(process_peaks),
    }
    return result, telemetry


def _repeat_product(
    function: Callable[[], float], *, warmups: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    samples: list[float] = []
    checksum = 0.0
    for _ in range(repeats):
        result, seconds, _ = _timed(function)
        checksum += float(result)
        samples.append(seconds)
    if not np.isfinite(checksum):
        raise RuntimeError("Isolated genotype-product checksum is non-finite.")
    return {
        "wall_samples": samples,
        "wall_median": float(np.median(samples)),
        "finite_checksum": True,
    }


def _validate_fixed_effect_spec(reference_preset: Any, study_preset: Any) -> None:
    reference_names = tuple(str(value) for value in reference_preset.fixed_effect_names)
    study_names = tuple(str(value) for value in study_preset.fixed_effect_names)
    if reference_names != study_names:
        raise RuntimeError("Reference and study fixed-effect specifications differ.")
    if reference_preset.fixed_effect_spec_hash != study_preset.fixed_effect_spec_hash:
        raise RuntimeError(
            "Reference and study fixed-effect specification hashes differ."
        )


def _dense_parity(
    *,
    reference: Any,
    summary: Any,
    reference_preset: Any,
    study_preset: Any,
    fixture: Mapping[str, Any],
) -> tuple[dict[str, Any], float, int]:
    def calculate() -> dict[str, Any]:
        reference_features = common_scale_features(
            fixture["reference_genotype"],
            reference_preset.basis,
            reference_preset.projector.projector,
        )
        reference_kernels = dense_genetic_kernels(
            reference_features,
            fixture["annotations"],
            reference_preset.component_index,
        )
        study_features = common_scale_features(
            fixture["study_genotype"],
            study_preset.basis,
            study_preset.projector.projector,
        )
        study_kernels = dense_genetic_kernels(
            study_features,
            fixture["annotations"],
            study_preset.component_index,
        )
        normalized = project_normalize_phenotype(
            fixture["phenotype"], study_preset.projector
        )
        residual_kernels = dense_residual_kernels(
            study_preset.projector.projector, study_preset.residual_basis
        )
        component_masses = np.asarray(
            [
                reference.annotation_masses[entry.annotation_index]
                for entry in reference.component_index.entries
            ],
            dtype=np.float64,
        )
        summary_component_masses = np.asarray(
            [
                summary.annotation_masses[entry.annotation_index]
                for entry in summary.component_index.entries
            ],
            dtype=np.float64,
        )
        expected = {
            "reference_gram": kernel_gram(reference_kernels),
            "reference_same_person": exact_same_person_matrix(reference_kernels),
            "reference_gram_numerator": (
                reference.gram * np.outer(component_masses, component_masses)
            ),
            "genetic_rhs": kernel_rhs(study_kernels, normalized),
            "genetic_traces": kernel_traces(study_kernels),
            "genetic_residual": np.einsum(
                "aij,hij->ah", study_kernels, residual_kernels, optimize=True
            ),
            "residual_rhs": kernel_rhs(residual_kernels, normalized),
            "residual_traces": kernel_traces(residual_kernels),
            "residual_gram": kernel_gram(residual_kernels),
            "summary_rhs_reconstruction": summary.genetic_rhs,
            "summary_trace_reconstruction": summary.genetic_traces,
            "summary_genetic_residual_reconstruction": summary.genetic_residual,
        }
        observed = {
            "reference_gram": reference.gram,
            "reference_same_person": reference.same_person,
            "reference_gram_numerator": np.sum(
                reference.gram_numerator_contributions, axis=0, dtype=np.float64
            ),
            "genetic_rhs": summary.genetic_rhs,
            "genetic_traces": summary.genetic_traces,
            "genetic_residual": summary.genetic_residual,
            "residual_rhs": summary.residual_rhs,
            "residual_traces": summary.residual_traces,
            "residual_gram": summary.residual_gram,
            "summary_rhs_reconstruction": np.sum(
                summary.rhs_numerator_contributions, axis=0, dtype=np.float64
            )
            / summary_component_masses,
            "summary_trace_reconstruction": np.sum(
                summary.trace_numerator_contributions, axis=0, dtype=np.float64
            )
            / summary_component_masses,
            "summary_genetic_residual_reconstruction": np.sum(
                summary.genetic_residual_numerator_contributions,
                axis=0,
                dtype=np.float64,
            )
            / summary_component_masses[:, None],
        }
        return {name: _discrepancy(observed[name], expected[name]) for name in expected}

    discrepancies, seconds, peak = _timed(calculate)
    maximum_absolute = max(
        value["maximum_absolute_error"] for value in discrepancies.values()
    )
    maximum_scaled = max(value["scale_aware_error"] for value in discrepancies.values())
    return (
        {
            "by_field": discrepancies,
            "maximum_absolute_error": float(maximum_absolute),
            "maximum_scale_aware_error": float(maximum_scaled),
            "declared_scale_aware_tolerance": PARITY_TOLERANCE,
            "within_declared_tolerance": bool(maximum_scaled <= PARITY_TOLERANCE),
        },
        seconds,
        peak,
    )


def _isolated_product_timings(
    *,
    reference_preset: Any,
    study_preset: Any,
    fixture: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    q_count = int(reference_preset.basis.shape[1])
    reference_genotype = np.asarray(fixture["reference_genotype"])
    study_genotype = np.asarray(fixture["study_genotype"])
    normalized = project_normalize_phenotype(
        fixture["phenotype"], study_preset.projector
    )
    right_hand_sides = study_preset.basis * normalized[:, None]

    def reference_products() -> float:
        checksum = 0.0
        for _ in range(args.product_inner_loops):
            for q in range(q_count):
                product = reference_preset.projector.projector @ (
                    reference_preset.basis[:, q, None] * reference_genotype
                )
                checksum += float(product[0, 0])
        return checksum

    def trait_products() -> float:
        checksum = 0.0
        for _ in range(args.product_inner_loops):
            for start in range(0, args.m, args.block_size):
                stop = min(start + args.block_size, args.m)
                block = study_genotype[:, start:stop]
                scores = block.T @ right_hand_sides
                checksum += float(scores[0, 0])
                for q in range(q_count):
                    product = study_preset.projector.projector @ (
                        study_preset.basis[:, q, None] * block
                    )
                    checksum += float(product[0, 0])
        return checksum

    reference = _repeat_product(
        reference_products, warmups=args.warmups, repeats=args.repeats
    )
    trait = _repeat_product(trait_products, warmups=args.warmups, repeats=args.repeats)
    reference["wall_median_per_inner_loop"] = (
        reference["wall_median"] / args.product_inner_loops
    )
    trait["wall_median_per_inner_loop"] = (
        trait["wall_median"] / args.product_inner_loops
    )
    return {"reference": reference, "trait": trait}


def _case(q_count: int, args: argparse.Namespace) -> dict[str, Any]:
    fixture = _sample_fixture(q_count, args)

    calibration, calibration_seconds, calibration_peak = _timed(
        lambda: calibrate_multienvironment_basis(
            fixture["reference_sources"],
            fixture["source_specs"],
            mask=fixture["reference_mask"],
            basis_id=f"benchmark_q{q_count}",
            max_basis=4,
        )
    )
    reference_preset, reference_apply_seconds, reference_apply_peak = _timed(
        lambda: apply_multienvironment_calibration(
            calibration,
            fixture["reference_sources"],
            mask=fixture["reference_mask"],
            annotation_names=("all",),
        )
    )
    study_preset, study_apply_seconds, study_apply_peak = _timed(
        lambda: apply_multienvironment_calibration(
            calibration,
            fixture["study_sources"],
            mask=fixture["study_mask"],
            annotation_names=("all",),
        )
    )
    if reference_preset.basis.shape[1] != q_count:
        raise RuntimeError("Reference calibration produced an unexpected basis size.")
    if study_preset.basis.shape[1] != q_count:
        raise RuntimeError("Study calibration produced an unexpected basis size.")
    if reference_preset.basis_hash != study_preset.basis_hash:
        raise RuntimeError("Frozen calibration did not preserve the basis hash.")
    if reference_preset.component_index.digest != study_preset.component_index.digest:
        raise RuntimeError("Reference and study component orders differ.")

    _validate_fixed_effect_spec(reference_preset, study_preset)
    common = {
        "annotations": fixture["annotations"],
        "basis_hash": reference_preset.basis_hash,
        "variant_hash": fixture["variant_hash"],
        "loo_groups": fixture["loo_groups"],
        "genotype_scaling": GENOTYPE_SCALING,
    }

    def build_reference() -> Any:
        return build_context_reference(
            genotype=fixture["reference_genotype"],
            basis=reference_preset.basis,
            projector=reference_preset.projector,
            component_index=reference_preset.component_index,
            fixed_effect_hash=reference_preset.fixed_effect_hash,
            gram_method="exact",
            same_person_method="exact",
            **common,
        )

    def build_summary() -> Any:
        return build_context_trait_summary(
            genotype=fixture["study_genotype"],
            basis=study_preset.basis,
            phenotype=fixture["phenotype"],
            projector=study_preset.projector,
            component_index=study_preset.component_index,
            fixed_effect_hash=study_preset.fixed_effect_hash,
            residual_basis=study_preset.residual_basis,
            residual_names=study_preset.residual_names,
            block_size=args.block_size,
            **common,
        )

    reference, reference_timing = _repeat_builder(
        build_reference, warmups=args.warmups, repeats=args.repeats
    )
    summary, summary_timing = _repeat_builder(
        build_summary, warmups=args.warmups, repeats=args.repeats
    )
    parity, dense_seconds, dense_peak = _dense_parity(
        reference=reference,
        summary=summary,
        reference_preset=reference_preset,
        study_preset=study_preset,
        fixture=fixture,
    )
    product_timings = _isolated_product_timings(
        reference_preset=reference_preset,
        study_preset=study_preset,
        fixture=fixture,
        args=args,
    )

    p_genetic = len(reference_preset.component_index)
    decoded_blocks = math.ceil(args.m / args.block_size)
    reference_arrays = (
        "annotation_weights",
        "annotation_masses",
        "gram",
        "same_person",
        "gram_numerator_contributions",
    )
    summary_arrays = (
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
    reference_projection_flops = 2 * q_count * args.reference_n**2 * args.m
    trait_projection_flops = 2 * q_count * args.study_n**2 * args.m
    trait_score_flops = 2 * args.study_n * args.m * q_count
    return {
        "q": q_count,
        "p_genetic": p_genetic,
        "dimensions": {
            "reference_n": args.reference_n,
            "study_n": args.study_n,
            "m": args.m,
            "k": 1,
            "fixed_effect_rank_reference": reference_preset.projector.rank,
            "fixed_effect_rank_study": study_preset.projector.rank,
            "residual_rank_reference": reference_preset.projector.residual_rank,
            "residual_rank_study": study_preset.projector.residual_rank,
        },
        "calibration": {
            "seconds": calibration_seconds,
            "reference_apply_seconds": reference_apply_seconds,
            "study_apply_seconds": study_apply_seconds,
            "absolute_process_peak_rss_bytes": max(
                calibration_peak, reference_apply_peak, study_apply_peak
            ),
            "same_basis_hash_across_independent_cohorts": True,
            "same_component_order_across_independent_cohorts": True,
            "reference_metric_shape": list(
                np.asarray(reference_preset.basis_metric).shape
            ),
        },
        "runtime_seconds": {
            "reference": reference_timing,
            "trait": summary_timing,
            "dense_oracle": dense_seconds,
            "isolated_q_linear_products": product_timings,
        },
        "absolute_peak_rss_bytes": {
            "dense_oracle_process_high_water": dense_peak,
            "at_case_completion": _peak_rss_bytes(),
        },
        "numeric_payload_bytes": {
            "reference": _numeric_payload_bytes(reference, reference_arrays),
            "trait": _numeric_payload_bytes(summary, summary_arrays),
        },
        "decode": {
            "reference_decoder_instrumentation": (
                "not_available_for_in_memory_python_reference"
            ),
            "trait_passes": int(summary.decode_passes),
            "trait_blocks": int(summary.decoded_blocks),
            "expected_trait_blocks": decoded_blocks,
            "one_trait_pass": bool(
                summary.decode_passes == 1 and summary.decoded_blocks == decoded_blocks
            ),
        },
        "rhs_bound": {
            "columns": q_count,
            "matrix_bytes": int(8 * args.study_n * q_count),
            "linear_in_q": True,
        },
        "operation_accounting_per_builder_call": {
            "accounting_kind": "source_derived_not_runtime_instrumented",
            "q_linear_genotype_products": {
                "reference_feature_projection_calls": q_count,
                "reference_projection_flops_proxy": reference_projection_flops,
                "trait_feature_projection_calls": q_count * decoded_blocks,
                "trait_score_gemm_calls": decoded_blocks,
                "trait_projection_flops_proxy": trait_projection_flops,
                "trait_score_flops_proxy": trait_score_flops,
                "combined_flops_proxy": (
                    reference_projection_flops
                    + trait_projection_flops
                    + trait_score_flops
                ),
            },
            "exact_python_moment_reductions": {
                "genetic_kernel_components": p_genetic,
                "dense_kernel_assembly_gemm_calls_k1": q_count**2,
                "gram_cells": p_genetic**2,
                "snp_gram_contribution_cells": args.m * p_genetic**2,
                "scaling_warning": (
                    "These exact correctness-first reductions scale with P_g^2; "
                    "they are distinct from the Q-linear genotype projection and "
                    "score products."
                ),
            },
        },
        "dense_parity": parity,
    }


def _empirical_scaling(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q_values = np.asarray([record["q"] for record in records], dtype=np.float64)
    reference_times = np.asarray(
        [
            record["runtime_seconds"]["isolated_q_linear_products"]["reference"][
                "wall_median_per_inner_loop"
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    trait_times = np.asarray(
        [
            record["runtime_seconds"]["isolated_q_linear_products"]["trait"][
                "wall_median_per_inner_loop"
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    def summarize(values: np.ndarray) -> dict[str, Any]:
        normalized = values / q_values
        exponent = None
        if values.size >= 2 and np.all(values > 0.0):
            exponent = float(np.polyfit(np.log(q_values), np.log(values), 1)[0])
        return {
            "median_seconds": values.tolist(),
            "seconds_per_q": normalized.tolist(),
            "log_log_timing_exponent_descriptive": exponent,
            "seconds_per_q_max_over_min": float(
                np.max(normalized) / np.min(normalized)
            ),
            "interpretation": (
                "descriptive_small_numpy_microbenchmark_not_a_native_scaling_gate"
            ),
        }

    first = (
        records[0]["operation_accounting_per_builder_call"][
            "q_linear_genotype_products"
        ]["combined_flops_proxy"]
        / records[0]["q"]
    )
    analytic_linear = all(
        math.isclose(
            record["operation_accounting_per_builder_call"][
                "q_linear_genotype_products"
            ]["combined_flops_proxy"]
            / record["q"],
            first,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for record in records
    )
    return {
        "q_values": q_values.astype(int).tolist(),
        "reference_products": summarize(reference_times),
        "trait_products": summarize(trait_times),
        "source_derived_dominant_flop_proxy_exactly_linear_in_q": analytic_linear,
        "timing_gate_applied": False,
    }


def _plot(records: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    ordered = sorted(records, key=lambda value: value["q"])
    q_values = np.asarray([value["q"] for value in ordered], dtype=int)
    reference_total = np.asarray(
        [value["runtime_seconds"]["reference"]["wall_median"] for value in ordered]
    )
    trait_total = np.asarray(
        [value["runtime_seconds"]["trait"]["wall_median"] for value in ordered]
    )
    reference_products = np.asarray(
        [
            value["runtime_seconds"]["isolated_q_linear_products"]["reference"][
                "wall_median_per_inner_loop"
            ]
            for value in ordered
        ]
    )
    trait_products = np.asarray(
        [
            value["runtime_seconds"]["isolated_q_linear_products"]["trait"][
                "wall_median_per_inner_loop"
            ]
            for value in ordered
        ]
    )
    reference_payload = np.asarray(
        [value["numeric_payload_bytes"]["reference"] / 2**20 for value in ordered]
    )
    trait_payload = np.asarray(
        [value["numeric_payload_bytes"]["trait"] / 2**20 for value in ordered]
    )
    p_squared = np.asarray([value["p_genetic"] ** 2 for value in ordered])
    linear_proxy = q_values / q_values[0]
    reduction_proxy = p_squared / p_squared[0]

    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), constrained_layout=True)
    axes[0, 0].plot(q_values, reference_total, "o-", label="exact reference")
    axes[0, 0].plot(q_values, trait_total, "s-", label="trait summary")
    axes[0, 0].set_ylabel("Median wall time (s)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(q_values, reference_products * 1.0e3, "o-", label="reference")
    axes[0, 1].plot(q_values, trait_products * 1.0e3, "s-", label="trait")
    axes[0, 1].set_ylabel("Q-linear product time (ms)")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(q_values, reference_payload, "o-", label="reference")
    axes[1, 0].plot(q_values, trait_payload, "s-", label="trait")
    axes[1, 0].set_ylabel("Numeric payload (MiB)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(q_values, linear_proxy, "o-", label="Q-linear products")
    axes[1, 1].plot(q_values, reduction_proxy, "s-", label=r"exact $P_g^2$ cells")
    axes[1, 1].set_ylabel("Count relative to smallest Q")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xlabel("Basis dimension Q")
        axis.set_xticks(q_values)
        axis.grid(alpha=0.25)
    figure.suptitle("Fixed multi-environment reference/trait benchmark")
    for suffix in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        path.chmod(0o600)
    plt.close(figure)


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
    records = [_case(q_count, args) for q_count in args.q_values]
    scaling = _empirical_scaling(records)
    maximum_scaled_error = max(
        record["dense_parity"]["maximum_scale_aware_error"] for record in records
    )
    gate = {
        "all_dense_parity_within_tolerance": all(
            record["dense_parity"]["within_declared_tolerance"] for record in records
        ),
        "all_trait_summaries_use_one_decode_pass": all(
            record["decode"]["one_trait_pass"] for record in records
        ),
        "source_derived_dominant_products_exactly_linear_in_q": scaling[
            "source_derived_dominant_flop_proxy_exactly_linear_in_q"
        ],
        "maximum_scale_aware_dense_parity_error": maximum_scaled_error,
    }
    verdict = (
        "pass"
        if all(
            value
            for key, value in gate.items()
            if key != "maximum_scale_aware_dense_parity_error"
        )
        else "review"
    )
    payload = {
        "kind": "summit.context.multienvironment_benchmark",
        "schema_version": 1,
        "seed": args.seed,
        "configuration": {
            "reference_n": args.reference_n,
            "study_n": args.study_n,
            "m": args.m,
            "k": 1,
            "q_values": list(args.q_values),
            "block_size": args.block_size,
            "loo_groups": args.loo_groups,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "product_inner_loops": args.product_inner_loops,
            "genotype_scaling": GENOTYPE_SCALING,
            "blas_threads": "inherited_runtime_configuration",
            "reference_methods": {"gram": "exact", "same_person": "exact"},
        },
        "privacy": {
            "fixture": "deterministic_synthetic_independent_reference_and_study",
            "individual_or_variant_rows_persisted": False,
            "outputs": "aggregate_metrics_and_figures_only",
        },
        "measurement_semantics": {
            "rss": "absolute_process_resident_set_high_water_not_incremental",
            "decode": (
                "trait builder counters are instrumented; reference is an in-memory "
                "Python prototype without decoder instrumentation"
            ),
            "operation_counts": (
                "source-derived analytical counts, not hardware performance counters"
            ),
            "timing_scaling": (
                "small NumPy timing is descriptive; exact analytical operation "
                "accounting is the scaling gate"
            ),
        },
        "runtime_seconds_total": float(time.perf_counter() - started),
        "absolute_process_rss_bytes": {
            "before_cases": started_rss,
            "at_completion": int(process.memory_info().rss),
            "high_water_at_completion": _peak_rss_bytes(),
        },
        "empirical_q_scaling": scaling,
        "aggregate_gate": gate,
        "verdict": verdict,
        "records": records,
    }
    json_path = output_paths[0]
    json_path.write_text(
        json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    json_path.chmod(0o600)
    _plot(records, args.output_dir)
    if verdict != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
