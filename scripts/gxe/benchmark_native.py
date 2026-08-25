#!/usr/bin/env python3
"""Reproducible native GxE source/target scaling benchmark.

The exact comparison uses missing-free PLINK BED data, K=1, sample-scaled
genotypes, a complete Q=[1,E,C] projection, explicit Python Philox probes,
and float64 accumulation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed
from threadpoolctl import threadpool_info, threadpool_limits

from summit import gxeldcore
from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _build_balanced_vtiles,
    _make_seed,
    _orthonormalize_columns,
)
from summit.logger import Logger


PROBE_COUNTS = (10, 100, 256, 1024)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="Synthetic sample count.")
    parser.add_argument("--m", type=int, default=160, help="Synthetic variant count.")
    parser.add_argument(
        "--probe-counts", default=",".join(str(value) for value in PROBE_COUNTS),
        help="Comma-separated total probe counts.",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=1000)
    parser.add_argument("--target-panel-columns", type=int, default=64)
    parser.add_argument("--workspace-gib", type=float, default=16.0)
    parser.add_argument("--max-abs-error", type=float, default=1e-8)
    parser.add_argument(
        "--stress-repeats",
        type=int,
        default=0,
        help="Repeat native source/target calls against the same dense oracle.",
    )
    parser.add_argument(
        "--full-estimator-probes", type=int, default=0,
        help="Optionally benchmark one complete Python/direct estimator transaction.",
    )
    parser.add_argument(
        "--skip-materialized",
        action="store_true",
        help="Skip redundant Python feature-rematerialization timings.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    return parser


def _timed(function, repeats: int, warmups: int) -> tuple[dict, object]:
    timings: list[float] = []
    result = None
    for _ in range(warmups):
        result = function()
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return {
        "raw": timings,
        "median": float(np.median(timings)),
        "minimum": float(np.min(timings)),
        "maximum": float(np.max(timings)),
    }, result


def _write_plink(directory: Path, raw: np.ndarray) -> Path:
    prefix = directory / "synthetic"
    to_bed(str(prefix) + ".bed", np.asarray(raw, dtype=np.float64))
    return prefix


@contextmanager
def _direct_context(prefix: Path, env: np.ndarray, q: np.ndarray, args):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptors = [
        os.open(str(prefix) + extension, flags)
        for extension in (".bed", ".bim", ".fam")
    ]
    context = None
    try:
        context = gxeldcore.DirectContext(
            bed_descriptor=descriptors[0],
            bim_descriptor=descriptors[1],
            fam_descriptor=descriptors[2],
            row_sel=None,
            ddof=1,
            env=env,
            q_basis=np.asfortranarray(q),
            decode_threads=args.decode_threads,
            max_workspace_bytes=int(args.workspace_gib * 1024**3),
            target_panel_columns=args.target_panel_columns,
        )
        yield context
    finally:
        if context is not None:
            context.close()
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _probes(m: int, count: int, root_seed: int) -> np.ndarray:
    probes = np.empty((m, count), dtype=np.float64, order="F")
    for probe_id in range(count):
        generator = np.random.Generator(
            np.random.Philox(_make_seed(root_seed, 0, probe_id))
        )
        probes[:, probe_id] = (
            2.0 * generator.integers(0, 2, size=m, dtype=np.int8) - 1.0
        )
    return probes


def _dense_design(raw: np.ndarray, env: np.ndarray, q: np.ndarray):
    genotype = np.asarray(raw, dtype=np.float64).copy(order="F")
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    genotype -= q @ (q.T @ genotype)
    interaction = env[:, None] * (
        (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    )
    interaction -= q @ (q.T @ interaction)
    rank = raw.shape[0] - q.shape[1]
    genotype *= np.sqrt(rank / np.sum(genotype * genotype, axis=0))
    interaction *= np.sqrt(rank / np.sum(interaction * interaction, axis=0))
    return np.asfortranarray(genotype), np.asfortranarray(interaction)


def _materialize_scaled_features(
    raw: np.ndarray,
    env: np.ndarray,
    q: np.ndarray,
    scale_x: np.ndarray,
    scale_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    genotype = np.asarray(raw, dtype=np.float64).copy(order="F")
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    additive = genotype - q @ (q.T @ genotype)
    interaction = env[:, None] * genotype
    interaction -= q @ (q.T @ interaction)
    additive *= scale_x.reshape(1, -1)
    interaction *= scale_w.reshape(1, -1)
    return np.asfortranarray(additive), np.asfortranarray(interaction)


def _timing_summary(values: list[float]) -> dict:
    return {
        "raw": values,
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _full_estimator_benchmark(
    directory: Path,
    prefix: Path,
    env_raw: np.ndarray,
    covariate: np.ndarray,
    probes: int,
    args,
) -> dict:
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env_path = directory / "full.env.tsv"
    cov_path = directory / "full.cov.tsv"
    ids.assign(E=env_raw).to_csv(env_path, sep="\t", index=False)
    ids.assign(C=covariate).to_csv(cov_path, sep="\t", index=False)
    outputs = {}
    timings = {}
    for backend in ("python", "direct"):
        raw_timings = []
        for repeat in range(args.warmups + args.repeats):
            estimator = GenomewideEnvLDScore(
                bed_path=str(prefix), env_path=str(env_path), covar_path=str(cov_path),
                annot_path=None,
                out_path=str(directory / f"full-{backend}-{repeat}"),
                log=Logger(suppress=True), rand_dist="rademacher", low_level=None,
                num_vecs=probes, step_size=min(args.step_size, args.m), seed=args.seed,
                dtype="float64", num_threads=args.decode_threads,
                kernel_mode="standardized_projected", genotype_scale="sample",
                impute_method="mean", target_xz_mem=max(0.01, args.workspace_gib),
                native_backend=backend, native_workspace_gib=args.workspace_gib,
                native_target_panel_columns=args.target_panel_columns,
            )
            try:
                with threadpool_limits(limits=args.blas_threads):
                    started = time.perf_counter()
                    estimator._compute_ldscore()
                    elapsed = time.perf_counter() - started
                if repeat >= args.warmups:
                    raw_timings.append(elapsed)
                outputs[backend] = tuple(
                    np.array(getattr(estimator, name), copy=True)
                    for name in ("gxx_ldscore", "gxe_ldscore", "exg_ldscore", "gee_ldscore")
                )
            finally:
                estimator.close()
        timings[backend] = _timing_summary(raw_timings)
    error = max(
        float(np.max(np.abs(native - python)))
        for native, python in zip(outputs["direct"], outputs["python"])
    )
    if error > args.max_abs_error:
        raise RuntimeError(
            f"Full-estimator native/Python error {error} exceeds {args.max_abs_error}."
        )
    return {
        "B": probes,
        "timings_seconds": timings,
        "direct_speedup": timings["python"]["median"] / timings["direct"]["median"],
        "max_abs_artifact_error": error,
    }


def run(args) -> dict:
    if args.n < 8 or args.m < 2 or args.repeats < 1 or args.warmups < 0:
        raise ValueError("Require N>=8, M>=2, repeats>=1, and warmups>=0.")
    counts = tuple(int(value) for value in args.probe_counts.split(",") if value)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("Probe counts must be positive integers.")
    if (
        args.max_abs_error <= 0.0
        or args.stress_repeats < 0
        or args.full_estimator_probes < 0
    ):
        raise ValueError("Scratch-model dimensions and error tolerance must be positive.")
    args.m = int(args.m)
    rng = np.random.default_rng(args.seed)
    allele_frequency = rng.uniform(0.1, 0.45, size=args.m)
    raw = rng.binomial(2, allele_frequency, size=(args.n, args.m)).astype(np.float64)
    for column in np.flatnonzero(raw.std(axis=0, ddof=1) == 0.0):
        raw[0, column] = 0.0
        raw[1, column] = 2.0
    env_raw = rng.normal(size=args.n)
    env = (env_raw - env_raw.mean()) / env_raw.std(ddof=1)
    covariate = rng.normal(size=args.n)
    q = _orthonormalize_columns(np.column_stack([np.ones(args.n), env, covariate]))
    dense_feature_timing, dense_design = _timed(
        lambda: _dense_design(raw, env, q), args.repeats, args.warmups
    )
    x, w = dense_design
    native_build = dict(gxeldcore.build_info())
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    payload = {
        "schema": "summit-native-gxe-benchmark-v2",
        "arguments": {
            "n": args.n, "m": args.m, "probe_counts": list(counts),
            "seed": args.seed, "repeats": args.repeats, "warmups": args.warmups,
            "decode_threads": args.decode_threads, "blas_threads": args.blas_threads,
            "step_size": args.step_size,
            "target_panel_columns": args.target_panel_columns,
            "workspace_gib": args.workspace_gib,
            "max_abs_error": args.max_abs_error,
            "stress_repeats": args.stress_repeats,
            "skip_materialized": args.skip_materialized,
        },
        "provenance": {
            "argv": list(sys.argv),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "affinity": affinity,
            "threadpools": threadpool_info(),
            "native_build": native_build,
            "native_binary": str(Path(gxeldcore.__file__).resolve()),
        },
        "contracts": {
            "source": "explicit Python Philox probes; [Ux,Uw]=2B float64 columns",
            "target_global": "opaque context-bound projected panel, width=2B",
        },
        "dense_feature_timing_seconds": dense_feature_timing,
        "cases": [],
    }

    with tempfile.TemporaryDirectory(prefix="summit-gxe-native-benchmark-") as name:
        directory = Path(name)
        prefix = _write_plink(directory, raw)
        with _direct_context(prefix, env, q, args) as context, threadpool_limits(
            limits=args.blas_threads
        ):
            feature_timing, feature = _timed(
                lambda: context.feature_block(0, args.m, 1e-10, True),
                args.repeats, args.warmups,
            )
            scale_x = np.asarray(feature["scale_x"])
            scale_w = np.asarray(feature["scale_w"])
            for count in counts:
                integrity_before = dict(context.info())
                probes = _probes(args.m, count, args.seed)
                dense_source_timing, dense_pair = _timed(
                    lambda: (x @ probes, w @ probes), args.repeats, args.warmups
                )
                dense_ux, dense_uw = dense_pair
                dense_sources_2b = np.asfortranarray(np.column_stack([dense_ux, dense_uw]))
                dense_target_2b_timing, dense_target_2b = _timed(
                    lambda: (x.T @ dense_sources_2b, w.T @ dense_sources_2b),
                    args.repeats, args.warmups,
                )

                def materialized_source():
                    additive, interaction = _materialize_scaled_features(
                        raw, env, q, scale_x, scale_w
                    )
                    return additive @ probes, interaction @ probes

                python_source_timing = None
                if not args.skip_materialized:
                    python_source_timing, _ = _timed(
                        materialized_source, args.repeats, args.warmups
                    )

                def materialized_target(panel):
                    additive, interaction = _materialize_scaled_features(
                        raw, env, q, scale_x, scale_w
                    )
                    return additive.T @ panel, interaction.T @ panel

                python_target_2b_timing = None
                if not args.skip_materialized:
                    python_target_2b_timing, _ = _timed(
                        lambda: materialized_target(dense_sources_2b),
                        args.repeats, args.warmups,
                    )
                native_source_timing, native_source = _timed(
                    lambda: context.source_block(
                        0, args.m, scale_x, scale_w, np.ones(args.m), probes,
                        True,
                    ),
                    args.repeats, args.warmups,
                )
                native_ux, native_uw, missing = native_source
                if int(missing) != 0:
                    raise RuntimeError("Missing genotypes appeared in a missing-free benchmark.")
                native_sources_2b = np.asfortranarray(
                    np.column_stack([native_ux, native_uw])
                )
                prepare_2b_timing, panel_2b = _timed(
                    lambda: context.prepare_projected_sources(native_sources_2b, 1e-9),
                    args.repeats, args.warmups,
                )
                native_target_2b_timing, native_target_2b = _timed(
                    lambda: context.target_projected_block(
                        0, args.m, scale_x, scale_w, panel_2b, True
                    ),
                    args.repeats, args.warmups,
                )
                native_x_2b, native_w_2b, target_missing_2b, _ = native_target_2b
                if int(target_missing_2b) != 0:
                    raise RuntimeError("Missing genotypes appeared in native target work.")
                source_error = max(
                    float(np.max(np.abs(native_ux - dense_ux))),
                    float(np.max(np.abs(native_uw - dense_uw))),
                )
                target_2b_error = max(
                    float(np.max(np.abs(native_x_2b - dense_target_2b[0]))),
                    float(np.max(np.abs(native_w_2b - dense_target_2b[1]))),
                )
                stress_source_errors: list[float] = []
                stress_source_repeat_deltas: list[float] = []
                stress_2b_errors: list[float] = []
                stress_2b_repeat_deltas: list[float] = []
                for _ in range(args.stress_repeats):
                    stress_ux, stress_uw, stress_source_missing = context.source_block(
                        0, args.m, scale_x, scale_w, np.ones(args.m), probes,
                        True,
                    )
                    stress_x_2b, stress_w_2b, stress_missing_2b, _ = (
                        context.target_projected_block(
                            0, args.m, scale_x, scale_w, panel_2b, True
                        )
                    )
                    if (
                        int(stress_source_missing) != 0
                        or int(stress_missing_2b) != 0
                    ):
                        raise RuntimeError("Missing genotypes appeared in native stress work.")
                    stress_source_errors.append(max(
                        float(np.max(np.abs(stress_ux - dense_ux))),
                        float(np.max(np.abs(stress_uw - dense_uw))),
                    ))
                    stress_source_repeat_deltas.append(max(
                        float(np.max(np.abs(stress_ux - native_ux))),
                        float(np.max(np.abs(stress_uw - native_uw))),
                    ))
                    stress_2b_errors.append(max(
                        float(np.max(np.abs(stress_x_2b - dense_target_2b[0]))),
                        float(np.max(np.abs(stress_w_2b - dense_target_2b[1]))),
                    ))
                    stress_2b_repeat_deltas.append(max(
                        float(np.max(np.abs(stress_x_2b - native_x_2b))),
                        float(np.max(np.abs(stress_w_2b - native_w_2b))),
                    ))
                if stress_2b_errors:
                    source_error = max(source_error, max(stress_source_errors))
                    target_2b_error = max(target_2b_error, max(stress_2b_errors))
                    max_repeat_delta = max(
                        max(stress_source_repeat_deltas),
                        max(stress_2b_repeat_deltas),
                    )
                    if max_repeat_delta > args.max_abs_error:
                        raise RuntimeError(
                            "Native stress output changed across identical calls: "
                            f"max repeat delta={max_repeat_delta}, "
                            f"tolerance={args.max_abs_error}."
                        )
                integrity_after = dict(context.info())
                repaired_columns = int(
                    integrity_after["repaired_gemm_output_columns"]
                ) - int(integrity_before["repaired_gemm_output_columns"])
                retried_inputs = int(
                    integrity_after["retried_gemm_input_mutations"]
                ) - int(integrity_before["retried_gemm_input_mutations"])
                if repaired_columns < 0 or retried_inputs < 0:
                    raise RuntimeError("Native integrity counters decreased during a benchmark case.")
                if max(source_error, target_2b_error) > args.max_abs_error:
                    raise RuntimeError(
                        "Native benchmark equivalence exceeded the configured error tolerance: "
                        f"source={source_error}, target2B={target_2b_error}, "
                        f"tolerance={args.max_abs_error}; "
                        f"ABFT repairs={repaired_columns}, input retries={retried_inputs}."
                    )
                timings = {
                    "native_feature": feature_timing,
                    "dense_gemm_lower_bound_source": dense_source_timing,
                    "dense_gemm_lower_bound_target_2b": dense_target_2b_timing,
                    "native_source": native_source_timing,
                    "native_prepare_projected_2b": prepare_2b_timing,
                    "native_target_2b": native_target_2b_timing,
                }
                speedups = None
                if python_source_timing is not None:
                    assert python_target_2b_timing is not None
                    timings.update({
                        "python_materialized_source": python_source_timing,
                        "python_materialized_target_2b": python_target_2b_timing,
                    })
                    speedups = {
                        "source": python_source_timing["median"] / native_source_timing["median"],
                        "target_2b": python_target_2b_timing["median"] / native_target_2b_timing["median"],
                    }
                case = {
                    "B": count,
                    "actual_target_columns_2b": 2 * count,
                    "timings_seconds": timings,
                    "correctness": {
                        "max_abs_source_error": source_error,
                        "max_abs_target_2b_error": target_2b_error,
                        "max_source_projection_leakage": float(panel_2b.leakage),
                        "abft_repaired_output_columns": repaired_columns,
                        "fresh_decode_input_retries": retried_inputs,
                        "stress_source_errors": stress_source_errors,
                        "stress_source_repeat_deltas": stress_source_repeat_deltas,
                        "stress_target_2b_errors": stress_2b_errors,
                        "stress_target_2b_repeat_deltas": stress_2b_repeat_deltas,
                        "threshold": args.max_abs_error,
                    },
                }
                if speedups is not None:
                    case["speedups_vs_python_materialized"] = speedups
                payload["cases"].append(case)

        if args.full_estimator_probes > 0:
            payload["full_estimator"] = _full_estimator_benchmark(
                directory, prefix, env_raw, covariate,
                int(args.full_estimator_probes), args,
            )
    return payload


def main() -> None:
    args = _parser().parse_args()
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
