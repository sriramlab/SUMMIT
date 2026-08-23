#!/usr/bin/env python3
"""Benchmark the correctness-first contextual reference constructor.

The default fixture is deliberately small enough to materialize every dense
kernel.  For Q=1,...,4, it compares the exact reference Gram and same-person
matrices against independent dense-kernel calculations.  By default it also
runs the seeded Hutchinson/variant-probe U-statistic estimators with 32 probes.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    build_context_reference,
    canonical_sha256,
    common_scale_features,
    dense_genetic_kernels,
    exact_same_person_matrix,
    kernel_gram,
    rank_revealing_projector,
)


OUTPUT_STEM = "03_context_reference_benchmark"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=80, help="Synthetic sample count.")
    parser.add_argument("--m", type=int, default=120, help="Synthetic variant count.")
    parser.add_argument("--k", type=int, default=1, help="Disjoint annotation count.")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--randomized-probes",
        type=int,
        default=32,
        help="Hutchinson/U-stat probe count; use 0 to skip randomized estimators.",
    )
    parser.add_argument(
        "--probe-tile-size",
        type=int,
        default=8,
        help="Maximum probe columns processed together by the reference builder.",
    )
    return parser


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.n < 8:
        parser.error("--n must be at least 8")
    if args.m < 4:
        parser.error("--m must be at least 4")
    if args.k < 1 or args.k > args.m:
        parser.error("--k must satisfy 1 <= K <= M")
    if args.randomized_probes < 0:
        parser.error("--randomized-probes must be non-negative")
    if args.randomized_probes == 1:
        parser.error("the same-person U-statistic requires 0 or at least 2 probes")
    if args.probe_tile_size < 1:
        parser.error("--probe-tile-size must be positive")


def _peak_rss_bytes() -> int:
    """Return this process's absolute resident-set high-water mark."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux and the BSDs report KiB; macOS reports bytes.
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
    return np.asarray(centered / scales[None, :], dtype=np.float64)


def _fixture(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    genotype = _standardize_columns(rng.normal(size=(args.n, args.m)))
    context = _standardize_columns(rng.normal(size=(args.n, 3)))
    raw_basis = np.column_stack([np.ones(args.n, dtype=np.float64), context])
    nuisance = _standardize_columns(rng.normal(size=(args.n, 2)))

    annotations = np.zeros((args.m, args.k), dtype=np.float64)
    annotations[np.arange(args.m), np.arange(args.m) % args.k] = 1.0
    annotation_names = tuple(f"annotation_{index}" for index in range(args.k))
    loo_count = min(12, args.m)
    loo_groups = tuple(f"group:{index % loo_count}" for index in range(args.m))

    gram_probes = None
    variant_probes = None
    if args.randomized_probes:
        gram_probes = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(args.n, args.randomized_probes),
        )
        variant_probes = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(args.m, args.randomized_probes),
        )
    return {
        "genotype": genotype,
        "raw_basis": np.asarray(raw_basis, dtype=np.float64),
        "nuisance": nuisance,
        "annotations": annotations,
        "annotation_names": annotation_names,
        "loo_groups": loo_groups,
        "gram_probes": gram_probes,
        "variant_probes": variant_probes,
    }


def _reference_kwargs(
    fixture: dict[str, Any],
    *,
    basis: np.ndarray,
    fixed: np.ndarray,
    projector: Any,
    components: ContextComponentIndex,
    probe_tile_size: int,
) -> dict[str, Any]:
    return {
        "genotype": fixture["genotype"],
        "basis": basis,
        "projector": projector,
        "annotations": fixture["annotations"],
        "component_index": components,
        "loo_groups": fixture["loo_groups"],
        "basis_hash": array_sha256(basis),
        "fixed_effect_hash": array_sha256(fixed),
        "variant_hash": canonical_sha256(
            {"ordered_synthetic_variants": list(range(fixture["genotype"].shape[1]))}
        ),
        "genotype_scaling": "sample_sd_ddof1_pre_scaled_input",
        "probe_tile_size": probe_tile_size,
    }


def _manifest_phase_times(reference: Any) -> dict[str, float]:
    raw = getattr(reference, "phase_times_seconds", None)
    if raw is None:
        manifest = reference.manifest
        raw = manifest.get("phase_times_seconds", manifest.get("phase_times", {}))
    if not isinstance(raw, dict):
        return {}
    result: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool):
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(converted) and converted >= 0.0:
            result[str(name)] = converted
    return result


def _run_case(
    q: int, fixture: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    basis = np.asarray(fixture["raw_basis"][:, :q], dtype=np.float64)
    fixed = np.column_stack(
        [np.ones(args.n, dtype=np.float64), basis[:, 1:], fixture["nuisance"]]
    )
    projector = rank_revealing_projector(fixed)
    components = ContextComponentIndex(fixture["annotation_names"], ContextPairIndex(q))

    def dense_oracle() -> tuple[np.ndarray, np.ndarray]:
        features = common_scale_features(
            fixture["genotype"], basis, projector.projector
        )
        kernels = dense_genetic_kernels(features, fixture["annotations"], components)
        return kernel_gram(kernels), exact_same_person_matrix(kernels)

    (dense_gram, dense_same), dense_seconds, dense_peak = _timed(dense_oracle)
    common = _reference_kwargs(
        fixture,
        basis=basis,
        fixed=fixed,
        projector=projector,
        components=components,
        probe_tile_size=args.probe_tile_size,
    )
    exact, exact_seconds, exact_peak = _timed(
        lambda: build_context_reference(
            **common,
            gram_method="exact",
            same_person_method="exact",
        )
    )
    exact_gram_error = float(np.max(np.abs(exact.gram - dense_gram), initial=0.0))
    exact_same_error = float(
        np.max(np.abs(exact.same_person - dense_same), initial=0.0)
    )

    randomized = None
    randomized_seconds = None
    randomized_peak = None
    randomized_gram_error = None
    randomized_same_error = None
    randomized_phase_times: dict[str, float] = {}
    if args.randomized_probes:
        randomized, randomized_seconds, randomized_peak = _timed(
            lambda: build_context_reference(
                **common,
                gram_method="hutchinson",
                gram_probes=fixture["gram_probes"],
                same_person_method="ustat",
                variant_probes=fixture["variant_probes"],
            )
        )
        randomized_gram_error = float(
            np.max(np.abs(randomized.gram - dense_gram), initial=0.0)
        )
        randomized_same_error = float(
            np.max(np.abs(randomized.same_person - dense_same), initial=0.0)
        )
        randomized_phase_times = _manifest_phase_times(randomized)

    return {
        "q": q,
        "p_genetic": len(components),
        "n": args.n,
        "m": args.m,
        "k": args.k,
        "fixed_effect_rank": projector.rank,
        "residual_rank": projector.residual_rank,
        "maximum_leverage": projector.maximum_leverage,
        "phase_times_seconds": {
            "dense_oracle": dense_seconds,
            "exact_reference_total": exact_seconds,
            "randomized_reference_total": randomized_seconds,
            "exact_reference_internal": _manifest_phase_times(exact),
            "randomized_reference_internal": randomized_phase_times,
        },
        "absolute_peak_rss_bytes": {
            "after_dense_oracle": dense_peak,
            "after_exact_reference": exact_peak,
            "after_randomized_reference": randomized_peak,
        },
        "maximum_absolute_error": {
            "exact_gram_vs_dense": exact_gram_error,
            "exact_same_person_vs_dense": exact_same_error,
            "hutchinson_gram_vs_dense": randomized_gram_error,
            "ustat_same_person_vs_dense": randomized_same_error,
        },
        "reference_array_bytes": {
            "exact": int(
                exact.gram.nbytes
                + exact.same_person.nbytes
                + exact.annotation_masses.nbytes
                + exact.gram_numerator_contributions.nbytes
            ),
            "randomized": (
                None
                if randomized is None
                else int(
                    randomized.gram.nbytes
                    + randomized.same_person.nbytes
                    + randomized.annotation_masses.nbytes
                    + randomized.gram_numerator_contributions.nbytes
                )
            ),
        },
    }


def _plot(
    records: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace
) -> None:
    q_values = np.asarray([record["q"] for record in records], dtype=np.int64)
    exact_times = [
        record["phase_times_seconds"]["exact_reference_total"] for record in records
    ]
    randomized_times = [
        record["phase_times_seconds"]["randomized_reference_total"]
        for record in records
    ]
    peak_rss = [
        max(
            value
            for value in record["absolute_peak_rss_bytes"].values()
            if value is not None
        )
        / 2**20
        for record in records
    ]

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), constrained_layout=True)
    color = "#1f6f8b"
    secondary = "#d97706"
    axes[0].plot(q_values, exact_times, "o-", color=color, label="exact T, exact D")
    if all(value is not None for value in randomized_times):
        axes[0].plot(
            q_values,
            randomized_times,
            "s--",
            color=secondary,
            label=f"Hutch/U-stat B={args.randomized_probes}",
        )
    axes[0].set_ylabel("Wall time (s)")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(q_values, peak_rss, "o-", color=color)
    axes[1].set_ylabel("Absolute process peak RSS (MiB)")

    error_series = {
        "exact T": [
            record["maximum_absolute_error"]["exact_gram_vs_dense"]
            for record in records
        ],
        "exact D": [
            record["maximum_absolute_error"]["exact_same_person_vs_dense"]
            for record in records
        ],
    }
    if args.randomized_probes:
        error_series.update(
            {
                "Hutch T": [
                    record["maximum_absolute_error"]["hutchinson_gram_vs_dense"]
                    for record in records
                ],
                "U-stat D": [
                    record["maximum_absolute_error"]["ustat_same_person_vs_dense"]
                    for record in records
                ],
            }
        )
    markers = ("o-", "s-", "o--", "s--")
    colors = (color, "#4c956c", secondary, "#8f5aa2")
    positive = [
        value
        for values in error_series.values()
        for value in values
        if value is not None and value > 0.0
    ]
    floor = min(positive) * 0.5 if positive else np.finfo(np.float64).eps
    for (label, values), marker, line_color in zip(
        error_series.items(), markers, colors
    ):
        plotted = [floor if value == 0.0 else value for value in values]
        axes[2].plot(q_values, plotted, marker, color=line_color, label=label)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Maximum absolute error")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel("Context basis dimension Q")
        axis.set_xticks(q_values)
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Contextual reference prototype (N={args.n}, M={args.m}, K={args.k})"
    )
    figure.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300)
    figure.savefig(output_dir / f"{OUTPUT_STEM}.pdf")
    plt.close(figure)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture, setup_seconds, setup_peak = _timed(lambda: _fixture(args))
    records = [_run_case(q, fixture, args) for q in range(1, 5)]
    payload = {
        "kind": "summit.context.reference_benchmark",
        "schema_version": 1,
        "seed": args.seed,
        "configuration": {
            "n": args.n,
            "m": args.m,
            "k": args.k,
            "q_values": [1, 2, 3, 4],
            "randomized_probes": args.randomized_probes,
            "probe_tile_size": args.probe_tile_size,
            "genotype_scaling": "sample_sd_ddof1",
        },
        "fixture_setup_seconds": setup_seconds,
        "absolute_peak_rss_after_setup_bytes": setup_peak,
        "absolute_peak_rss_bytes": _peak_rss_bytes(),
        "records": records,
    }
    (args.output_dir / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(records, args.output_dir, args)


if __name__ == "__main__":
    main()
