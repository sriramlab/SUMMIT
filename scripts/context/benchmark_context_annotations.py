#!/usr/bin/env python3
"""Benchmark grouped disjoint-annotation contextual covariance.

Small deterministic reference/study fixtures cover K=1,4,8 and Q=2,3,4 by
default.  The benchmark measures grouped reference, one-pass trait summary,
and summary-only grouped approximate-LOO fit phases.  It records exact numeric
payload formulas, decode/GEMM counts, fixed-probe tile parity, selected dense
comparisons, public resource estimates, and production-dimension extrapolations.
Only aggregate diagnostics are persisted.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import tempfile
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from summit.context import (
    AnnotationResourceRequest,
    ContextComponentIndex,
    ContextPairIndex,
    ContextRankError,
    array_sha256,
    assemble_context_normal_equations,
    build_context_reference,
    build_context_trait_summary,
    build_disjoint_annotation_partition,
    build_grouped_context_reference,
    build_grouped_context_trait_summary,
    canonical_json,
    canonical_sha256,
    estimate_annotation_resources,
    fit_annotation_context_model,
    group_context_reference,
    group_context_trait_summary,
    rank_revealing_projector,
    validate_disjoint_annotation_partition,
    write_context_reference,
    write_context_trait_summary,
)


OUTPUT_STEM = "08_context_annotation_benchmark"
GENOTYPE_SCALING = "sample_sd_ddof1_pre_scaled_input"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-reference", type=int, default=48)
    parser.add_argument("--n-study", type=int, default=52)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--q-values", default="2,3,4")
    parser.add_argument("--k-values", default="1,4,8")
    parser.add_argument("--loo-groups", type=int, default=8)
    parser.add_argument("--residual-components", type=int, default=2)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument(
        "--probe-sweep-values",
        default="4,8,16,32",
        help="Fixed nested probe counts for one representative exact comparison.",
    )
    parser.add_argument("--probe-tile-size", type=int, default=4)
    parser.add_argument("--trait-block-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--exact-parity-max-pg",
        type=int,
        default=10,
        help="Run dense exact reference comparisons through this Pg dimension.",
    )
    parser.add_argument("--maximum-action-mib", type=float, default=512.0)
    parser.add_argument("--maximum-p-genetic", type=int, default=100)
    parser.add_argument("--extrapolate-n-reference", type=int, default=300_000)
    parser.add_argument("--extrapolate-n-study", type=int, default=250_000)
    parser.add_argument("--extrapolate-m", type=int, default=1_000_000)
    parser.add_argument("--extrapolate-loo-groups", type=int, default=100)
    parser.add_argument("--extrapolate-probes", type=int, default=128)
    parser.add_argument("--extrapolate-probe-tile", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _integer_list(
    parser: argparse.ArgumentParser, value: str, label: str
) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError:
        parser.error(f"{label} must be a comma-separated integer list")
    if not result or any(item < 1 for item in result):
        parser.error(f"{label} must contain positive integers")
    if len(result) != len(set(result)):
        parser.error(f"{label} must not contain duplicates")
    return result


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n_reference < 12 or args.n_study < 12:
        parser.error("reference and study N must each be at least 12")
    if args.m < 8:
        parser.error("--m must be at least 8")
    if args.loo_groups < 2 or args.loo_groups > args.m:
        parser.error("--loo-groups must satisfy 2 <= J <= M")
    if args.residual_components < 1:
        parser.error("--residual-components must be positive")
    if args.probes < 2:
        parser.error("--probes must be at least 2 for the U-statistic")
    if any(value < 2 for value in args.probe_sweep_values):
        parser.error("--probe-sweep-values must all be at least 2")
    if args.probe_tile_size < 1 or args.trait_block_size < 1:
        parser.error("tile and block sizes must be positive")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats positive")
    if args.exact_parity_max_pg < 0:
        parser.error("--exact-parity-max-pg must be nonnegative")
    if not math.isfinite(args.maximum_action_mib) or args.maximum_action_mib <= 0.0:
        parser.error("--maximum-action-mib must be finite and positive")
    if args.maximum_p_genetic < 1:
        parser.error("--maximum-p-genetic must be positive")
    for k_count in args.k_values:
        if args.m % (k_count * args.loo_groups):
            parser.error(
                "--m must be divisible by K*J for exact annotation/group balance; "
                f"failed K={k_count}, J={args.loo_groups}"
            )
    if max(args.q_values) + 2 >= min(args.n_reference, args.n_study):
        parser.error("sample sizes are too small for the requested fixed-effect ranks")
    positive_extrapolation = (
        args.extrapolate_n_reference,
        args.extrapolate_n_study,
        args.extrapolate_m,
        args.extrapolate_loo_groups,
        args.extrapolate_probes,
        args.extrapolate_probe_tile,
    )
    if any(value < 1 for value in positive_extrapolation):
        parser.error("all extrapolation dimensions must be positive")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _timed(function: Callable[[], Any]) -> tuple[Any, float, int]:
    started = time.perf_counter()
    value = function()
    return value, float(time.perf_counter() - started), _peak_rss_bytes()


def _repeat(
    function: Callable[[], Any], *, warmups: int, repeats: int
) -> tuple[Any, dict[str, Any]]:
    for _ in range(warmups):
        function()
        gc.collect()
    values: list[Any] = []
    seconds: list[float] = []
    high_water: list[int] = []
    for _ in range(repeats):
        value, elapsed, peak = _timed(function)
        values.append(value)
        seconds.append(elapsed)
        high_water.append(peak)
        gc.collect()
    phase_names = sorted(
        {
            str(name)
            for value in values
            for name in getattr(value, "phase_times_seconds", {})
        }
    )
    return values[-1], {
        "wall_seconds_samples": seconds,
        "wall_seconds_median": float(np.median(seconds)),
        "absolute_process_high_water_samples": high_water,
        "absolute_process_high_water_maximum": max(high_water),
        "builder_reported_rss_at_return_samples": [
            int(getattr(value, "peak_rss_bytes", 0)) for value in values
        ],
        "internal_phase_seconds_medians": {
            name: float(
                np.median(
                    [
                        float(value.phase_times_seconds.get(name, 0.0))
                        for value in values
                    ]
                )
            )
            for name in phase_names
        },
    }


def _standardize_columns(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - np.mean(array, axis=0, keepdims=True, dtype=np.float64)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("Synthetic fixture generated a degenerate column.")
    return np.ascontiguousarray(centered / scale[None, :])


def _basis_and_fixed(
    rng: np.random.Generator, n_samples: int, q_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = _standardize_columns(rng.normal(size=(n_samples, max(3, q_count - 1))))
    basis = np.ones((n_samples, q_count), dtype=np.float64)
    if q_count > 1:
        basis[:, 1:] = context[:, : q_count - 1]
    nuisance = _standardize_columns(rng.normal(size=(n_samples, 1)))
    fixed = np.column_stack([np.ones(n_samples), basis[:, 1:], nuisance])
    return basis, fixed, context[:, 0]


def _residual_basis(context: np.ndarray, h_count: int) -> np.ndarray:
    columns = [np.ones(context.size, dtype=np.float64)]
    for power in range(1, h_count):
        value = context ** (power + 1)
        value = value - np.min(value) + 0.25
        columns.append(value)
    return np.ascontiguousarray(np.column_stack(columns))


def _fixture(
    *,
    q_count: int,
    k_count: int,
    args: argparse.Namespace,
    probe_count: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + 10_000 * k_count + 100 * q_count)
    reference_genotype = _standardize_columns(
        rng.normal(size=(args.n_reference, args.m))
    )
    study_genotype = _standardize_columns(rng.normal(size=(args.n_study, args.m)))
    reference_basis, reference_fixed, _ = _basis_and_fixed(
        rng, args.n_reference, q_count
    )
    study_basis, study_fixed, study_context = _basis_and_fixed(
        rng, args.n_study, q_count
    )
    reference_projector = rank_revealing_projector(reference_fixed)
    study_projector = rank_revealing_projector(study_fixed)

    effects = rng.normal(size=args.m) / np.sqrt(args.m)
    contextual_effects = rng.normal(size=args.m) / np.sqrt(args.m)
    phenotype = (
        study_genotype @ effects
        + 0.35 * study_context * (study_genotype @ contextual_effects)
        + rng.normal(scale=1.4, size=args.n_study)
    )
    residual = _residual_basis(study_context, args.residual_components)
    residual_names = tuple(
        "residual:constant" if index == 0 else f"residual:context_power_{index + 1}"
        for index in range(args.residual_components)
    )

    annotations = np.zeros((args.m, k_count), dtype=np.float64)
    annotations[np.arange(args.m), np.arange(args.m) % k_count] = 1.0
    annotation_names = tuple(f"bin:{index}" for index in range(k_count))
    loo_groups = tuple(
        f"group:{(index // k_count) % args.loo_groups}" for index in range(args.m)
    )
    variant_hash = canonical_sha256(
        {"ordered_synthetic_variants": [f"variant:{index}" for index in range(args.m)]}
    )
    definitions = tuple(
        {
            "kind": "synthetic_balanced_disjoint_bin",
            "index": index,
            "label": name,
        }
        for index, name in enumerate(annotation_names)
    )
    partition = build_disjoint_annotation_partition(
        annotations,
        annotation_names,
        definitions=definitions,
        variant_hash=variant_hash,
        loo_groups=loo_groups,
        source="deterministic_synthetic_benchmark",
        source_digest=array_sha256(annotations),
    )
    components = ContextComponentIndex(annotation_names, ContextPairIndex(q_count))
    basis_hash = canonical_sha256(
        {
            "kind": "synthetic_fixed_standard_normal_context_basis",
            "q": q_count,
            "columns": ["constant"]
            + [f"standardized_context_{index}" for index in range(q_count - 1)],
        }
    )
    probe_rng = np.random.default_rng(
        args.seed + 1_000_000 + 10_000 * k_count + q_count
    )
    probes = args.probes if probe_count is None else probe_count
    gram_probes = probe_rng.choice(
        np.asarray([-1.0, 1.0]), size=(args.n_reference, probes)
    )
    variant_probes = probe_rng.choice(np.asarray([-1.0, 1.0]), size=(args.m, probes))
    return {
        "reference_genotype": reference_genotype,
        "study_genotype": study_genotype,
        "reference_basis": reference_basis,
        "study_basis": study_basis,
        "reference_fixed": reference_fixed,
        "study_fixed": study_fixed,
        "reference_projector": reference_projector,
        "study_projector": study_projector,
        "phenotype": np.asarray(phenotype, dtype=np.float64),
        "residual_basis": residual,
        "residual_names": residual_names,
        "annotations": annotations,
        "annotation_names": annotation_names,
        "loo_groups": loo_groups,
        "variant_hash": variant_hash,
        "partition": partition,
        "components": components,
        "basis_hash": basis_hash,
        "gram_probes": np.asarray(gram_probes, dtype=np.float64),
        "variant_probes": np.asarray(variant_probes, dtype=np.float64),
    }


def _reference_kwargs(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "genotype": fixture["reference_genotype"],
        "basis": fixture["reference_basis"],
        "projector": fixture["reference_projector"],
        "basis_hash": fixture["basis_hash"],
        "fixed_effect_hash": array_sha256(fixture["reference_fixed"]),
        "variant_hash": fixture["variant_hash"],
        "genotype_scaling": GENOTYPE_SCALING,
    }


def _trait_kwargs(
    fixture: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "genotype": fixture["study_genotype"],
        "basis": fixture["study_basis"],
        "phenotype": fixture["phenotype"],
        "projector": fixture["study_projector"],
        "residual_basis": fixture["residual_basis"],
        "residual_names": fixture["residual_names"],
        "basis_hash": fixture["basis_hash"],
        "fixed_effect_hash": array_sha256(fixture["study_fixed"]),
        "variant_hash": fixture["variant_hash"],
        "genotype_scaling": GENOTYPE_SCALING,
        "block_size": args.trait_block_size,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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
        "maximum_axis": int(
            max((max(array.shape, default=0) for _, array in arrays), default=0)
        ),
        "m_axis_paths": [
            path for path, array in arrays if value.n_variants in array.shape
        ],
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


def _expected_grouped_reference_bytes(k_count: int, p_genetic: int, groups: int) -> int:
    float_elements = k_count + 2 * p_genetic**2 + groups * (p_genetic**2 + k_count)
    return 8 * float_elements + 8 * groups


def _expected_grouped_trait_bytes(
    k_count: int, p_genetic: int, h_count: int, groups: int
) -> int:
    float_elements = (
        k_count
        + p_genetic * (h_count + 2)
        + 2 * h_count
        + h_count**2
        + groups * (k_count + p_genetic * (h_count + 2))
    )
    return 8 * float_elements + 8 * groups


def _snp_reference_numeric_bytes(k_count: int, p_genetic: int, m_variants: int) -> int:
    return 8 * (k_count + 2 * p_genetic**2 + m_variants * (k_count + p_genetic**2))


def _snp_trait_numeric_bytes(
    k_count: int, p_genetic: int, h_count: int, m_variants: int
) -> int:
    return 8 * (
        k_count
        + p_genetic * (h_count + 2)
        + 2 * h_count
        + h_count**2
        + m_variants * (k_count + p_genetic * (h_count + 2))
    )


def _relative_max(observed: np.ndarray, expected: np.ndarray) -> float:
    scale = np.maximum(1.0, np.maximum(np.abs(observed), np.abs(expected)))
    return float(np.max(np.abs(observed - expected) / scale, initial=0.0))


def _frobenius_relative(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(observed - expected) / denominator)


def _reconstruction(reference: Any, summary: Any) -> dict[str, float]:
    component_masses = np.asarray(
        [
            reference.annotation_masses[entry.annotation_index]
            for entry in reference.component_index.entries
        ],
        dtype=np.float64,
    )
    reference_reconstructed = np.sum(
        reference.group_gram_numerator_contributions, axis=0, dtype=np.float64
    ) / np.outer(component_masses, component_masses)
    rhs = np.sum(summary.rhs_numerator_contributions, axis=0) / component_masses
    traces = np.sum(summary.trace_numerator_contributions, axis=0) / component_masses
    genetic_residual = (
        np.sum(summary.genetic_residual_numerator_contributions, axis=0)
        / component_masses[:, None]
    )
    return {
        "reference_gram_relative_max": _relative_max(
            reference_reconstructed, reference.gram
        ),
        "trait_rhs_relative_max": _relative_max(rhs, summary.genetic_rhs),
        "trait_trace_relative_max": _relative_max(traces, summary.genetic_traces),
        "trait_genetic_residual_relative_max": _relative_max(
            genetic_residual, summary.genetic_residual
        ),
    }


def _resource_request(
    *,
    n_reference: int,
    n_study: int,
    n_variants: int,
    q_count: int,
    k_count: int,
    h_count: int,
    probes: int,
    tile: int,
    groups: int,
    args: argparse.Namespace,
) -> AnnotationResourceRequest:
    return AnnotationResourceRequest(
        n_reference=n_reference,
        n_study=n_study,
        n_variants=n_variants,
        q=q_count,
        k=k_count,
        h=h_count,
        gram_probes=probes,
        same_person_probes=probes,
        probe_tile_size=tile,
        loo_groups=groups,
        memory_cap_bytes=int(args.maximum_action_mib * 2**20),
        reference_method="hutchinson",
        dtype_bytes=8,
    )


def _formula_audit(request: AnnotationResourceRequest, estimate: Any) -> dict[str, Any]:
    pair_count = request.q * (request.q + 1) // 2
    p_genetic = request.k * pair_count
    p_total = p_genetic + request.h
    selected_tile = int(estimate.selected_probe_tile_size)
    probe_tiles = (
        0 if selected_tile == 0 else math.ceil(request.gram_probes / selected_tile)
    )
    reference_core = 8 * (request.k + 2 * p_genetic**2)
    trait_core = 8 * (
        request.k + p_genetic * (request.h + 2) + 2 * request.h + request.h**2
    )
    expected = {
        "p_genetic": p_genetic,
        "p_total": p_total,
        "operator_peak_bytes": 8
        * (
            p_total**2
            + 3 * p_genetic**2
            + request.n_reference * p_genetic
            + request.n_reference * selected_tile * p_genetic
            + request.n_variants * selected_tile * request.q
            + request.n_reference * selected_tile * request.k
        ),
        "current_python_peak_bytes": 8
        * (
            request.q * request.n_reference * request.n_variants
            + p_genetic * request.q * request.n_variants * max(selected_tile, 1)
            + request.n_reference * selected_tile * p_genetic
            + request.n_reference * request.gram_probes
            + request.n_variants * request.same_person_probes
            + request.loo_groups * p_genetic**2
        ),
        "buffer_bytes.normal_matrix": 8 * p_total**2,
        "buffer_bytes.operator_action_tile": 8
        * request.n_reference
        * selected_tile
        * p_genetic,
        "buffer_bytes.operator_source_tile": 8
        * request.n_variants
        * selected_tile
        * request.q,
        "buffer_bytes.operator_annotation_target_tile": 8
        * request.n_reference
        * selected_tile
        * request.k,
        "buffer_bytes.same_person_persistent": 8
        * (request.n_reference * p_genetic + p_genetic**2),
        "storage_bytes.snp_reference_contribution_bytes": 8
        * request.n_variants
        * p_genetic**2,
        "storage_bytes.grouped_reference_contribution_bytes": 8
        * request.loo_groups
        * p_genetic**2,
        "storage_bytes.snp_reference_total_bytes": reference_core
        + 8 * request.n_variants * (request.k + p_genetic**2),
        "storage_bytes.grouped_reference_total_bytes": reference_core
        + 8 * request.loo_groups * (request.k + p_genetic**2 + 1),
        "storage_bytes.snp_trait_contribution_bytes": 8
        * request.n_variants
        * p_genetic
        * (2 + request.h),
        "storage_bytes.grouped_trait_contribution_bytes": 8
        * request.loo_groups
        * p_genetic
        * (2 + request.h),
        "storage_bytes.snp_trait_total_bytes": trait_core
        + 8 * request.n_variants * (request.k + p_genetic * (2 + request.h)),
        "storage_bytes.grouped_trait_total_bytes": trait_core
        + 8 * request.loo_groups * (request.k + p_genetic * (2 + request.h) + 1),
        "operation_counts.source_genotype_products": request.q * probe_tiles,
        "operation_counts.annotation_target_genotype_products": (
            request.k * request.q * probe_tiles
        ),
        "operation_counts.trait_decode_passes": 1,
    }
    observed: dict[str, Any] = {
        "p_genetic": estimate.p_genetic,
        "p_total": estimate.p_total,
        "operator_peak_bytes": estimate.operator_peak_bytes,
        "current_python_peak_bytes": estimate.current_python_peak_bytes,
    }
    for namespace in ("buffer_bytes", "storage_bytes", "operation_counts"):
        values = getattr(estimate, namespace)
        for key, value in values.items():
            observed[f"{namespace}.{key}"] = value
    comparisons = {
        name: {
            "expected": value,
            "observed": observed.get(name),
            "exact": observed.get(name) == value,
        }
        for name, value in expected.items()
    }
    return {
        "pair_count": pair_count,
        "expected": expected,
        "observed": {name: observed.get(name) for name in expected},
        "comparisons": comparisons,
        "mismatches": [
            name for name, comparison in comparisons.items() if not comparison["exact"]
        ],
        "all_exact": all(comparison["exact"] for comparison in comparisons.values()),
    }


def _artifact_sizes(reference: Any, summary: Any) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="summit-stage08-artifact-") as directory:
        root = Path(directory)
        reference_manifest, reference_arrays = write_context_reference(
            reference, root / "reference"
        )
        summary_manifest, summary_arrays = write_context_trait_summary(
            summary, root / "trait"
        )
        return {
            "reference_manifest_bytes": reference_manifest.stat().st_size,
            "reference_npz_bytes": reference_arrays.stat().st_size,
            "trait_manifest_bytes": summary_manifest.stat().st_size,
            "trait_npz_bytes": summary_arrays.stat().st_size,
        }


def _fit_diagnostics(reference: Any, summary: Any, partition: Any) -> dict[str, Any]:
    equations = assemble_context_normal_equations(reference, summary)
    eigenvalues = np.linalg.eigvalsh(0.5 * (equations.matrix + equations.matrix.T))
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    tolerance = 1.0e-10 * scale
    rank = int(np.sum(np.abs(eigenvalues) > tolerance))
    fit, elapsed, peak = _timed(
        lambda: fit_annotation_context_model(reference, summary, partition=partition)
    )
    covariance_eigenvalues = np.linalg.eigvalsh(fit.jackknife_covariance)
    return {
        "equations": equations,
        "fit": fit,
        "record": {
            "wall_seconds": elapsed,
            "absolute_process_high_water_bytes": peak,
            "phase_times_seconds": dict(fit.phase_times_seconds),
            "rank": rank,
            "dimension": int(equations.matrix.shape[0]),
            "condition_number": float(np.linalg.cond(equations.matrix)),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "jackknife_groups": len(fit.jackknife_groups),
            "jackknife_covariance_minimum_eigenvalue": float(covariance_eigenvalues[0]),
            "manifest_partition_mode": fit.manifest.get("annotation_mode"),
            "manifest_loo_storage": fit.manifest.get("approximate_loo", {}).get(
                "method"
            ),
        },
    }


def _case(
    q_count: int, k_count: int, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = _fixture(q_count=q_count, k_count=k_count, args=args)
    partition = fixture["partition"]
    partition_validation = validate_disjoint_annotation_partition(partition)
    reference_kwargs = _reference_kwargs(fixture)
    trait_kwargs = _trait_kwargs(fixture, args)
    reference, reference_timing = _repeat(
        lambda: build_grouped_context_reference(
            partition=partition,
            **reference_kwargs,
            gram_method="hutchinson",
            gram_probes=fixture["gram_probes"],
            same_person_method="ustat",
            variant_probes=fixture["variant_probes"],
            probe_tile_size=args.probe_tile_size,
        ),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    summary, summary_timing = _repeat(
        lambda: build_grouped_context_trait_summary(
            partition=partition, **trait_kwargs
        ),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    fit_info = _fit_diagnostics(reference, summary, partition)
    equations = fit_info.pop("equations")
    fit = fit_info.pop("fit")

    p_genetic = len(fixture["components"])
    reference_inventory = _tensor_inventory(reference)
    summary_inventory = _tensor_inventory(summary)
    expected_reference_bytes = _expected_grouped_reference_bytes(
        k_count, p_genetic, args.loo_groups
    )
    expected_trait_bytes = _expected_grouped_trait_bytes(
        k_count, p_genetic, args.residual_components, args.loo_groups
    )
    request = _resource_request(
        n_reference=args.n_reference,
        n_study=args.n_study,
        n_variants=args.m,
        q_count=q_count,
        k_count=k_count,
        h_count=args.residual_components,
        probes=args.probes,
        tile=args.probe_tile_size,
        groups=args.loo_groups,
        args=args,
    )
    resource_without_pilot = estimate_annotation_resources(request)
    resource_with_pilot = estimate_annotation_resources(
        request, pilot_normal_matrix=equations.matrix
    )
    record: dict[str, Any] = {
        "q": q_count,
        "k": k_count,
        "pair_count": len(fixture["components"].pair_index),
        "p_genetic": p_genetic,
        "p_total": p_genetic + args.residual_components,
        "partition": {
            "validation": partition_validation,
            "digest": partition.digest,
            "annotation_masses": partition.annotation_masses.tolist(),
            "group_variant_counts": partition.group_balance.group_variant_counts.tolist(),
            "group_annotation_masses": (
                partition.group_balance.group_annotation_masses.tolist()
            ),
            "equal_group_delete_one_compatible": bool(
                partition.group_balance.total_groups_balanced
                and partition.group_balance.every_deletion_retains_each_annotation
            ),
        },
        "reference": {
            "timing": reference_timing,
            "phase_times_seconds": dict(reference.phase_times_seconds),
            "tensor_inventory": reference_inventory,
            "expected_grouped_numeric_bytes": expected_reference_bytes,
            "numeric_bytes_match_formula": bool(
                reference_inventory["total_array_bytes"] == expected_reference_bytes
            ),
            "snp_numeric_bytes_counterfactual": _snp_reference_numeric_bytes(
                k_count, p_genetic, args.m
            ),
            "contribution_shape": list(
                reference.group_gram_numerator_contributions.shape
            ),
            "probe_metadata": reference.manifest.get("probes"),
        },
        "trait": {
            "timing": summary_timing,
            "phase_times_seconds": dict(summary.phase_times_seconds),
            "tensor_inventory": summary_inventory,
            "expected_grouped_numeric_bytes": expected_trait_bytes,
            "numeric_bytes_match_formula": bool(
                summary_inventory["total_array_bytes"] == expected_trait_bytes
            ),
            "snp_numeric_bytes_counterfactual": _snp_trait_numeric_bytes(
                k_count, p_genetic, args.residual_components, args.m
            ),
            "decode_passes": summary.decode_passes,
            "decoded_blocks": summary.decoded_blocks,
            "expected_decoded_blocks": math.ceil(args.m / args.trait_block_size),
            "rhs_contribution_shape": list(summary.rhs_numerator_contributions.shape),
            "trace_contribution_shape": list(
                summary.trace_numerator_contributions.shape
            ),
            "genetic_residual_contribution_shape": list(
                summary.genetic_residual_numerator_contributions.shape
            ),
        },
        "fit": fit_info["record"],
        "reconstruction_relative_errors": _reconstruction(reference, summary),
        "resource_estimate_without_pilot": _json_safe(resource_without_pilot),
        "resource_estimate_with_pilot": _json_safe(resource_with_pilot),
        "resource_formula_audit": _formula_audit(request, resource_without_pilot),
        "gemm_and_decode_model": {
            "gram_probe_tiles": math.ceil(args.probes / args.probe_tile_size),
            "source_gemms_per_gram_probe_tile": q_count,
            "target_gemms_per_gram_probe_tile": k_count * q_count,
            "dominant_genotype_gemms_per_gram_probe_tile": q_count * (k_count + 1),
            "same_person_ideal_genotype_gemms_per_probe_tile": k_count,
            "same_person_current_python_feature_gemms_per_probe_tile": (
                k_count * q_count
            ),
            "trait_decode_passes": summary.decode_passes,
            "trait_multi_rhs_score_gemms": summary.decoded_blocks,
            "trait_projected_feature_gemms": summary.decoded_blocks * q_count,
            "small_gram_reduction_multiply_add_pairs": (
                args.n_reference * args.probes * p_genetic**2
            ),
        },
        "artifact_sizes": _artifact_sizes(reference, summary),
        "manifest_binding": {
            "partition_digest_in_reference_manifest": partition.digest
            in canonical_json(reference.manifest),
            "partition_digest_in_trait_manifest": partition.digest
            in canonical_json(summary.manifest),
            "partition_digest_in_fit_manifest": partition.digest
            in canonical_json(fit.manifest),
            "grouping_hash_match": bool(
                reference.manifest.get("loo_grouping_hash")
                == summary.manifest.get("loo_grouping_hash")
            ),
            "annotation_hash_match": bool(
                reference.manifest.get("annotation_hash")
                == summary.manifest.get("annotation_hash")
            ),
        },
    }
    record["dense_exact_comparison"] = None
    if p_genetic <= args.exact_parity_max_pg:
        exact, elapsed, peak = _timed(
            lambda: build_grouped_context_reference(
                partition=partition,
                **reference_kwargs,
                gram_method="exact",
                same_person_method="exact",
                probe_tile_size=args.probe_tile_size,
            )
        )
        record["dense_exact_comparison"] = {
            "wall_seconds": elapsed,
            "absolute_process_high_water_bytes": peak,
            "hutchinson_gram_relative_max": _relative_max(reference.gram, exact.gram),
            "ustat_same_person_relative_max": _relative_max(
                reference.same_person, exact.same_person
            ),
            "exact_grouped_reconstruction_relative_max": _relative_max(
                np.sum(exact.group_gram_numerator_contributions, axis=0)
                / np.outer(
                    np.asarray(
                        [
                            exact.annotation_masses[entry.annotation_index]
                            for entry in exact.component_index.entries
                        ]
                    ),
                    np.asarray(
                        [
                            exact.annotation_masses[entry.annotation_index]
                            for entry in exact.component_index.entries
                        ]
                    ),
                ),
                exact.gram,
            ),
        }
    return record, {
        "fixture": fixture,
        "reference": reference,
        "summary": summary,
    }


def _selected_parity(
    objects: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    fixture = objects["fixture"]
    direct_reference = objects["reference"]
    direct_summary = objects["summary"]
    reference_kwargs = _reference_kwargs(fixture)
    trait_kwargs = _trait_kwargs(fixture, args)
    raw_reference = build_context_reference(
        annotations=fixture["annotations"],
        component_index=fixture["components"],
        loo_groups=fixture["loo_groups"],
        **reference_kwargs,
        gram_method="hutchinson",
        gram_probes=fixture["gram_probes"],
        same_person_method="ustat",
        variant_probes=fixture["variant_probes"],
        probe_tile_size=args.probe_tile_size,
    )
    raw_summary = build_context_trait_summary(
        annotations=fixture["annotations"],
        component_index=fixture["components"],
        loo_groups=fixture["loo_groups"],
        **trait_kwargs,
    )
    adapted_reference = group_context_reference(raw_reference, fixture["partition"])
    adapted_summary = group_context_trait_summary(raw_summary, fixture["partition"])
    tile_one = build_grouped_context_reference(
        partition=fixture["partition"],
        **reference_kwargs,
        gram_method="hutchinson",
        gram_probes=fixture["gram_probes"],
        same_person_method="ustat",
        variant_probes=fixture["variant_probes"],
        probe_tile_size=1,
    )
    return {
        "case": {
            "q": fixture["components"].pair_index.num_basis,
            "k": len(fixture["annotation_names"]),
        },
        "direct_vs_posthoc_grouping_relative_max": {
            "reference_gram": _relative_max(
                direct_reference.gram, adapted_reference.gram
            ),
            "reference_same_person": _relative_max(
                direct_reference.same_person, adapted_reference.same_person
            ),
            "reference_group_contributions": _relative_max(
                direct_reference.group_gram_numerator_contributions,
                adapted_reference.group_gram_numerator_contributions,
            ),
            "trait_rhs": _relative_max(
                direct_summary.genetic_rhs, adapted_summary.genetic_rhs
            ),
            "trait_group_rhs": _relative_max(
                direct_summary.rhs_numerator_contributions,
                adapted_summary.rhs_numerator_contributions,
            ),
            "trait_group_trace": _relative_max(
                direct_summary.trace_numerator_contributions,
                adapted_summary.trace_numerator_contributions,
            ),
            "trait_group_genetic_residual": _relative_max(
                direct_summary.genetic_residual_numerator_contributions,
                adapted_summary.genetic_residual_numerator_contributions,
            ),
        },
        "fixed_probe_tile_one_vs_requested_relative_max": {
            "gram": _relative_max(direct_reference.gram, tile_one.gram),
            "same_person": _relative_max(
                direct_reference.same_person, tile_one.same_person
            ),
            "group_contributions": _relative_max(
                direct_reference.group_gram_numerator_contributions,
                tile_one.group_gram_numerator_contributions,
            ),
        },
    }


def _probe_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Compare nested fixed-probe prefixes with one exact representative case."""
    k_count = 4 if 4 in args.k_values else args.k_values[len(args.k_values) // 2]
    q_count = 3 if 3 in args.q_values else args.q_values[len(args.q_values) // 2]
    maximum_probes = max(args.probe_sweep_values)
    fixture = _fixture(
        q_count=q_count,
        k_count=k_count,
        args=args,
        probe_count=maximum_probes,
    )
    reference_kwargs = _reference_kwargs(fixture)
    exact, exact_seconds, exact_rss = _timed(
        lambda: build_grouped_context_reference(
            partition=fixture["partition"],
            **reference_kwargs,
            gram_method="exact",
            same_person_method="exact",
            probe_tile_size=args.probe_tile_size,
        )
    )
    component_masses = np.asarray(
        [
            exact.annotation_masses[entry.annotation_index]
            for entry in exact.component_index.entries
        ],
        dtype=np.float64,
    )
    records: list[dict[str, Any]] = []
    for probe_count in args.probe_sweep_values:
        approximate, elapsed, peak = _timed(
            lambda probe_count=probe_count: build_grouped_context_reference(
                partition=fixture["partition"],
                **reference_kwargs,
                gram_method="hutchinson",
                gram_probes=fixture["gram_probes"][:, :probe_count],
                same_person_method="ustat",
                variant_probes=fixture["variant_probes"][:, :probe_count],
                probe_tile_size=args.probe_tile_size,
            )
        )
        reconstructed = np.sum(
            approximate.group_gram_numerator_contributions,
            axis=0,
            dtype=np.float64,
        ) / np.outer(component_masses, component_masses)
        request = _resource_request(
            n_reference=args.n_reference,
            n_study=args.n_study,
            n_variants=args.m,
            q_count=q_count,
            k_count=k_count,
            h_count=args.residual_components,
            probes=probe_count,
            tile=args.probe_tile_size,
            groups=args.loo_groups,
            args=args,
        )
        estimate = estimate_annotation_resources(request)
        records.append(
            {
                "probes": probe_count,
                "wall_seconds": elapsed,
                "absolute_process_high_water_bytes": peak,
                "gram_relative_max_vs_exact": _relative_max(
                    approximate.gram, exact.gram
                ),
                "gram_frobenius_relative_vs_exact": _frobenius_relative(
                    approximate.gram, exact.gram
                ),
                "gram_diagonal_relative_max_vs_exact": _relative_max(
                    np.diag(approximate.gram), np.diag(exact.gram)
                ),
                "same_person_relative_max_vs_exact": _relative_max(
                    approximate.same_person, exact.same_person
                ),
                "same_person_frobenius_relative_vs_exact": _frobenius_relative(
                    approximate.same_person, exact.same_person
                ),
                "same_person_diagonal_relative_max_vs_exact": _relative_max(
                    np.diag(approximate.same_person), np.diag(exact.same_person)
                ),
                "grouped_contribution_reconstruction_relative_max": _relative_max(
                    reconstructed, approximate.gram
                ),
                "probe_tiles": math.ceil(probe_count / args.probe_tile_size),
                "source_genotype_gemms": (
                    q_count * math.ceil(probe_count / args.probe_tile_size)
                ),
                "annotation_target_genotype_gemms": (
                    k_count * q_count * math.ceil(probe_count / args.probe_tile_size)
                ),
                "resource_estimate": _json_safe(estimate),
                "resource_formula_audit": _formula_audit(request, estimate),
            }
        )
    return {
        "case": {"k": k_count, "q": q_count, "p_genetic": len(exact.component_index)},
        "probe_construction": "nested_fixed_rademacher_prefixes",
        "exact_baseline": {
            "wall_seconds": exact_seconds,
            "absolute_process_high_water_bytes": exact_rss,
        },
        "records": records,
    }


def _production_extrapolations(args: argparse.Namespace) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for k_count in args.k_values:
        for q_count in args.q_values:
            request = _resource_request(
                n_reference=args.extrapolate_n_reference,
                n_study=args.extrapolate_n_study,
                n_variants=args.extrapolate_m,
                q_count=q_count,
                k_count=k_count,
                h_count=args.residual_components,
                probes=args.extrapolate_probes,
                tile=args.extrapolate_probe_tile,
                groups=args.extrapolate_loo_groups,
                args=args,
            )
            estimate = estimate_annotation_resources(request)
            result.append(
                {
                    "q": q_count,
                    "k": k_count,
                    "measurement": "dimension_formula_extrapolation_not_runtime",
                    "estimate": _json_safe(estimate),
                    "formula_audit": _formula_audit(request, estimate),
                }
            )
    return result


def _plot(
    records: Sequence[Mapping[str, Any]],
    probe_sweep: Mapping[str, Any],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), constrained_layout=True)
    colors = {1: "#1f6f8b", 4: "#d98e04", 8: "#8f3b76"}
    for k_count in sorted({int(record["k"]) for record in records}):
        selected = sorted(
            (record for record in records if record["k"] == k_count),
            key=lambda record: record["q"],
        )
        q_values = [record["q"] for record in selected]
        color = colors.get(k_count)
        axes[0, 0].plot(
            q_values,
            [
                record["reference"]["timing"]["wall_seconds_median"]
                for record in selected
            ],
            "o-",
            label=f"K={k_count}",
            color=color,
        )
        axes[0, 1].plot(
            q_values,
            [record["trait"]["timing"]["wall_seconds_median"] for record in selected],
            "o-",
            label=f"K={k_count}",
            color=color,
        )
        axes[1, 0].plot(
            q_values,
            [record["fit"]["wall_seconds"] for record in selected],
            "o-",
            label=f"K={k_count}",
            color=color,
        )
        axes[1, 1].plot(
            q_values,
            [
                record["reference"]["expected_grouped_numeric_bytes"] / 2**20
                for record in selected
            ],
            "o-",
            label=f"grouped K={k_count}",
            color=color,
        )
        axes[1, 1].plot(
            q_values,
            [
                record["reference"]["snp_numeric_bytes_counterfactual"] / 2**20
                for record in selected
            ],
            "--",
            color=color,
            alpha=0.6,
        )
    axes[0, 0].set_title("Grouped randomized reference")
    axes[0, 1].set_title("One-pass grouped trait summary")
    axes[1, 0].set_title("Grouped approximate-LOO fit")
    axes[1, 1].set_title("Reference numeric storage (solid grouped; dashed SNP)")
    axes[0, 0].set_ylabel("Median wall time (s)")
    axes[0, 1].set_ylabel("Median wall time (s)")
    axes[1, 0].set_ylabel("Wall time (s)")
    axes[1, 1].set_ylabel("MiB")
    axes[1, 1].set_yscale("log")
    sweep_records = probe_sweep["records"]
    sweep_probes = [record["probes"] for record in sweep_records]
    axes[0, 2].plot(
        sweep_probes,
        [record["gram_frobenius_relative_vs_exact"] for record in sweep_records],
        "o-",
        color="#2d6a4f",
    )
    axes[1, 2].plot(
        sweep_probes,
        [record["same_person_frobenius_relative_vs_exact"] for record in sweep_records],
        "o-",
        color="#6a4c93",
    )
    axes[0, 2].set_title("Fixed-probe Gram Frobenius error")
    axes[1, 2].set_title("Fixed-probe same-person Frobenius error")
    for axis in (axes[0, 2], axes[1, 2]):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(sweep_probes, labels=[str(value) for value in sweep_probes])
        axis.set_xlabel("Probe count B")
        axis.set_ylabel("Relative Frobenius error")
        axis.grid(alpha=0.25)
    for axis in axes[:, :2].flat:
        axis.set_xlabel("Context basis dimension Q")
        axis.set_xticks(sorted({int(record["q"]) for record in records}))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Disjoint-annotation contextual covariance scaling")
    for suffix in ("png", "pdf"):
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        path.chmod(0o600)
    plt.close(figure)


def main() -> None:
    os.umask(0o077)
    parser = _parser()
    args = parser.parse_args()
    args.q_values = _integer_list(parser, args.q_values, "--q-values")
    args.k_values = _integer_list(parser, args.k_values, "--k-values")
    args.probe_sweep_values = _integer_list(
        parser, args.probe_sweep_values, "--probe-sweep-values"
    )
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
    started = time.perf_counter()
    rss_before = int(process.memory_info().rss)
    records: list[dict[str, Any]] = []
    selected_objects: dict[str, Any] | None = None
    for k_count in args.k_values:
        for q_count in args.q_values:
            record, objects = _case(q_count, k_count, args)
            records.append(record)
            if selected_objects is None:
                selected_objects = objects
    assert selected_objects is not None
    parity = _selected_parity(selected_objects, args)
    probe_sweep = _probe_sweep(args)
    extrapolations = _production_extrapolations(args)

    reconstruction_max = max(
        value
        for record in records
        for value in record["reconstruction_relative_errors"].values()
    )
    adapter_max = max(
        parity["direct_vs_posthoc_grouping_relative_max"].values(), default=0.0
    )
    tile_max = max(
        parity["fixed_probe_tile_one_vs_requested_relative_max"].values(), default=0.0
    )
    gate = {
        "all_partitions_strict_and_balanced": all(
            record["partition"]["validation"]["status"]
            == "valid_strict_disjoint_partition"
            and record["partition"]["equal_group_delete_one_compatible"]
            for record in records
        ),
        "all_grouped_numeric_payloads_match_exact_formulas": all(
            record["reference"]["numeric_bytes_match_formula"]
            and record["trait"]["numeric_bytes_match_formula"]
            for record in records
        ),
        "no_grouped_artifact_has_variant_axis": all(
            not record["reference"]["tensor_inventory"]["m_axis_paths"]
            and not record["trait"]["tensor_inventory"]["m_axis_paths"]
            for record in records
        ),
        "all_grouped_contributions_reconstruct": reconstruction_max <= 1.0e-10,
        "all_traits_use_one_decode_and_expected_blocks": all(
            record["trait"]["decode_passes"] == 1
            and record["trait"]["decoded_blocks"]
            == record["trait"]["expected_decoded_blocks"]
            for record in records
        ),
        "all_public_resource_formulas_match_independent_arithmetic": all(
            record["resource_formula_audit"]["all_exact"] for record in records
        )
        and all(value["formula_audit"]["all_exact"] for value in extrapolations),
        "all_measured_cases_within_declared_resource_limits": all(
            record["resource_estimate_without_pilot"]["within_memory_cap"]
            and record["resource_estimate_without_pilot"][
                "current_python_within_memory_cap"
            ]
            and record["p_genetic"] <= args.maximum_p_genetic
            for record in records
        ),
        "all_pilot_conditions_are_explicit": all(
            record["resource_estimate_without_pilot"]["condition_status"]
            == "unknown_without_pilot"
            and record["resource_estimate_with_pilot"]["condition_status"]
            in {"pilot_full_rank", "pilot_rank_deficient"}
            for record in records
        ),
        "all_full_and_loo_fits_are_full_rank_with_psd_covariance": all(
            record["fit"]["rank"] == record["fit"]["dimension"]
            and record["fit"]["jackknife_groups"] == args.loo_groups
            and record["fit"]["jackknife_covariance_minimum_eigenvalue"] >= -1.0e-8
            for record in records
        ),
        "partition_hashes_bind_reference_trait_and_fit": all(
            all(record["manifest_binding"].values()) for record in records
        ),
        "direct_and_posthoc_grouping_match": adapter_max <= 1.0e-10,
        "fixed_probe_tiling_matches": tile_max <= 1.0e-10,
        "probe_sweep_grouped_contributions_reconstruct": all(
            record["grouped_contribution_reconstruction_relative_max"] <= 1.0e-10
            for record in probe_sweep["records"]
        ),
        "probe_sweep_public_resource_formulas_match": all(
            record["resource_formula_audit"]["all_exact"]
            for record in probe_sweep["records"]
        ),
    }
    verdict = "pass" if all(gate.values()) else "review"
    payload = {
        "kind": "summit.context.annotation_benchmark",
        "schema_version": 1,
        "experimental": True,
        "seed": args.seed,
        "configuration": {
            "n_reference": args.n_reference,
            "n_study": args.n_study,
            "m_variants": args.m,
            "q_values": list(args.q_values),
            "k_values": list(args.k_values),
            "loo_groups": args.loo_groups,
            "residual_components": args.residual_components,
            "probes": args.probes,
            "probe_sweep_values": list(args.probe_sweep_values),
            "probe_tile_size": args.probe_tile_size,
            "trait_block_size": args.trait_block_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "exact_parity_max_pg": args.exact_parity_max_pg,
            "maximum_action_bytes": int(args.maximum_action_mib * 2**20),
            "maximum_p_genetic": args.maximum_p_genetic,
        },
        "privacy": {
            "fixture": "deterministic_synthetic",
            "individual_or_variant_rows_persisted": False,
            "outputs": "aggregate_metrics_hashes_and_figures_only",
        },
        "measurement_semantics": {
            "measured": "small synthetic wall time, RSS, tensors, artifacts, fits",
            "extrapolated": "public dimension-only resource formulas; no runtime claim",
            "rss": "absolute process values, not incremental allocation",
            "source_peak_rss": "builder samples process RSS during construction",
            "current_python_backend": (
                "correctness-first dense features; grouped storage removes M-axis "
                "retention but does not make every transient production-bounded"
            ),
        },
        "formula_conventions": {
            "p_genetic": "K*Q*(Q+1)/2",
            "grouped_reference_contribution_tensor": "8*J*Pg^2 bytes",
            "grouped_reference_complete_loo_numeric": (
                "8*J*(Pg^2+K+1) bytes; final +1 is int64 group count"
            ),
            "grouped_trait_contribution_tensors": "8*J*Pg*(2+H) bytes",
            "grouped_trait_complete_loo_numeric": (
                "8*J*(Pg*(2+H)+K+1) bytes; final +1 is int64 group count"
            ),
            "action_buffer": "8*N_reference*probe_tile*Pg bytes",
            "source_sketch": "8*M*Q*probe_tile bytes",
            "annotation_target_sketch": (
                "8*N_reference*K*probe_tile bytes when streamed over q"
            ),
            "same_person_persistent": "8*(N_reference*Pg+Pg^2) bytes",
            "gram_genotype_gemms_per_tile": "Q+K*Q",
            "condition_without_pilot": "unknown_without_pilot",
        },
        "runtime_seconds_before_output": float(time.perf_counter() - started),
        "absolute_process_rss_bytes": {
            "before_cases": rss_before,
            "at_payload": int(process.memory_info().rss),
            "high_water_at_payload": _peak_rss_bytes(),
        },
        "maximum_reconstruction_relative_error": reconstruction_max,
        "maximum_adapter_parity_relative_error": adapter_max,
        "maximum_probe_tile_parity_relative_error": tile_max,
        "selected_parity": parity,
        "probe_sweep": probe_sweep,
        "production_dimension_extrapolations": extrapolations,
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
    _plot(records, probe_sweep, args.output_dir)
    if verdict != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
