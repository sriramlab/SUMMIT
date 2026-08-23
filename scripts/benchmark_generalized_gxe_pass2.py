#!/usr/bin/env python3
"""Moderate two-pass benchmark for generalized per-variant GxE LD scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import tempfile
import time

from bed_reader import to_bed
import numpy as np
import pandas as pd

from summit import gxeldcore
from summit.ldscore.generalized_gxe_pass1 import (
    GeneralizedGxEPass1Executor,
    MatureSequentialGenotypeOperator,
    ProtectedNNOperator,
)
from summit.ldscore.generalized_gxe_pass2 import (
    GeneralizedGxEPass2Executor,
    ProtectedTNOperator,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.logger import Logger


def _orthonormalize(value: np.ndarray) -> np.ndarray:
    left, singular, _right = np.linalg.svd(value, full_matrices=False)
    tolerance = 1.0e-12 * max(1.0, float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    if rank != value.shape[1]:
        raise RuntimeError("benchmark fixed-effect design lost rank")
    return np.asfortranarray(left[:, :rank])


def _inputs(root: Path, args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    frequencies = rng.uniform(0.05, 0.48, size=args.variants)
    raw = rng.binomial(
        2, frequencies, size=(args.samples, args.variants)
    ).astype(np.float64)
    prefix = root / "pass2-benchmark"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    identifiers = {
        "FID": fam[0].astype(str),
        "IID": fam[1].astype(str),
    }
    environment = rng.normal(size=args.samples)
    covariates = rng.normal(size=(args.samples, 2))
    environment_path = root / "environment.tsv"
    pd.DataFrame({**identifiers, "E": environment}).to_csv(
        environment_path, sep="\t", index=False
    )
    covariate_path = root / "covariates.tsv"
    pd.DataFrame(
        {
            **identifiers,
            "C1": covariates[:, 0],
            "C2": covariates[:, 1],
        }
    ).to_csv(covariate_path, sep="\t", index=False)
    basis_columns = [np.ones(args.samples), environment]
    if args.basis >= 3:
        basis_columns.append(environment**2)
    for power in range(3, args.basis):
        basis_columns.append(environment ** (power + 1))
    basis = np.asfortranarray(np.column_stack(basis_columns[: args.basis]))
    fixed = _orthonormalize(
        np.column_stack([np.ones(args.samples), covariates])
    )
    annotations = rng.uniform(
        0.05, 1.0, size=(args.variants, args.annotations)
    )
    return prefix, environment_path, covariate_path, basis, fixed, annotations


def _configured_threads(requested: int) -> int:
    requested = min(int(requested), len(os.sched_getaffinity(0)))
    if requested < 1:
        raise RuntimeError("no CPU is available to the benchmark")
    try:
        return int(gxeldcore.configure_blas_threads(requested))
    except RuntimeError as exc:
        if "different thread count" not in str(exc):
            raise
        configured = int(gxeldcore.build_info()["blas_runtime_threads"])
        if gxeldcore.configure_blas_threads(configured) != configured:
            raise RuntimeError("could not authenticate fixed native thread count")
        return configured


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--variants", type=int, default=4096)
    parser.add_argument("--basis", type=int, default=3)
    parser.add_argument("--annotations", type=int, default=1)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--jackknife-blocks", type=int, default=20)
    parser.add_argument("--variant-block-width", type=int, default=512)
    parser.add_argument("--pass1-probe-width", type=int, default=64)
    parser.add_argument("--pass2-probe-width", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026082205)
    parser.add_argument("--memory-gib", type=float, default=2.0)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    return parser


def _block_ids(num_variants: int, block_count: int) -> np.ndarray:
    return np.repeat(
        np.arange(block_count, dtype=np.int64),
        np.diff(
            np.linspace(0, num_variants, block_count + 1, dtype=np.int64)
        ),
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _mature_tn_calibration(
    *,
    samples: int,
    variant_width: int,
    columns: int,
    threads: int,
    repeats: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    left = np.asfortranarray(rng.normal(size=(samples, variant_width)))
    right = np.asfortranarray(rng.normal(size=(samples, columns)))
    timings = []
    repaired = []
    for _repeat in range(repeats):
        started = time.perf_counter()
        output, repair_count = gxeldcore.protected_matmul_tn(
            left, right, threads
        )
        timings.append(time.perf_counter() - started)
        repaired.append(int(repair_count))
        if np.asarray(output).shape != (variant_width, columns):
            raise RuntimeError("mature TN calibration returned an invalid shape")
        del output
    median_seconds = float(np.median(timings))
    flops = 2 * samples * variant_width * columns
    return {
        "binding": "gxeldcore.protected_matmul_tn",
        "dimensions": [variant_width, samples, columns],
        "repeats": repeats,
        "seconds": timings,
        "median_seconds": median_seconds,
        "leading_flops_per_call": flops,
        "median_gflops_per_second": flops / median_seconds / 1.0e9,
        "repaired_columns": repaired,
    }


def run(args: argparse.Namespace) -> dict:
    for name in (
        "samples",
        "variants",
        "basis",
        "annotations",
        "probes",
        "jackknife_blocks",
        "variant_block_width",
        "pass1_probe_width",
        "pass2_probe_width",
        "threads",
        "calibration_repeats",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.probes < 2:
        raise ValueError("--probes must be at least two")
    if args.jackknife_blocks < 2 or args.jackknife_blocks > args.variants:
        raise ValueError("--jackknife-blocks must be in [2, variants]")
    if args.pass2_probe_width > args.probes:
        raise ValueError("--pass2-probe-width cannot exceed --probes")
    if not np.isfinite(args.memory_gib) or args.memory_gib <= 0.0:
        raise ValueError("--memory-gib must be positive and finite")

    setup_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="summit-generalized-pass2-") as raw_root:
        root = Path(raw_root)
        (
            prefix,
            environment_path,
            covariate_path,
            basis,
            fixed,
            annotations,
        ) = _inputs(root, args)
        threads = _configured_threads(args.threads)
        estimator = GenomewideEnvLDScore(
            bed_path=str(prefix),
            env_path=str(environment_path),
            env_col="E",
            annot_path=None,
            out_path=str(root / "unused-mature-output"),
            log=Logger(suppress=True),
            rand_dist="rademacher",
            low_level=None,
            covar_path=str(covariate_path),
            num_vecs=args.probes,
            step_size=args.variant_block_width,
            seed=args.seed,
            dtype="float64",
            num_threads=threads,
            target_xz_mem=0.5,
            gxe_total_memory_gib=args.memory_gib,
            impute_method="mean",
            kernel_mode="standardized",
            genotype_scale="sample",
            native_backend="python",
        )
        try:
            genotype_operator = (
                MatureSequentialGenotypeOperator.from_genomewide_estimator(
                    estimator
                )
            )
            q_squared = args.basis**2
            plan = plan_generalized_gxe_variant_work(
                GeneralizedGxEPlanInputs(
                    num_samples=args.samples,
                    num_variants=args.variants,
                    num_basis=args.basis,
                    num_annotations=args.annotations,
                    num_probes=args.probes,
                    num_jackknife_blocks=args.jackknife_blocks,
                    memory_limit_bytes=int(args.memory_gib * 1024**3),
                    genotype_format="bed",
                    threads=threads,
                    preferred_variant_block_width=args.variant_block_width,
                    preferred_rhs_tile_columns=(
                        q_squared * args.pass2_probe_width
                    ),
                    rhs_policy="tiled",
                )
            )
            spec = GlobalVariantProbeSpec(
                root_seed=args.seed,
                probe_offset=0,
                probe_count=args.probes,
            )
            names = tuple(
                f"annotation_{index}" for index in range(args.annotations)
            )
            pass1 = GeneralizedGxEPass1Executor(
                genotype_operator=genotype_operator,
                basis=basis,
                fixed_effect_basis=fixed,
                annotations=annotations,
                annotation_names=names,
                annotation_masses=np.sum(annotations, axis=0),
                probe_spec=spec,
                work_plan=plan,
                nn_operator=ProtectedNNOperator(
                    threads=threads, native_module=gxeldcore
                ),
                annotation_tile_width=min(4, args.annotations),
                probe_tile_width=args.pass1_probe_width,
                same_person_sample_tile_width=min(512, args.samples),
                native_probe_module=gxeldcore,
            ).execute()
            result = GeneralizedGxEPass2Executor(
                pass1_result=pass1,
                genotype_operator=genotype_operator,
                basis=basis,
                fixed_effect_basis=fixed,
                annotations=annotations,
                annotation_names=names,
                block_ids=_block_ids(args.variants, args.jackknife_blocks),
                work_plan=plan,
                tn_operator=ProtectedTNOperator(
                    threads=threads, native_module=gxeldcore
                ),
                probe_tile_width=args.pass2_probe_width,
            ).execute()
        finally:
            estimator.close()

    actual_tn = result.telemetry["target_tn"]
    calibration = _mature_tn_calibration(
        samples=args.samples,
        variant_width=args.variant_block_width,
        columns=q_squared * args.pass2_probe_width,
        threads=threads,
        repeats=args.calibration_repeats,
        seed=args.seed + 1,
    )
    actual_gflops = float(actual_tn["gflops_per_second"])
    calibration_gflops = float(calibration["median_gflops_per_second"])
    return {
        "schema": "summit.generalized_gxe.pass2_benchmark.v1",
        "dimensions": {
            "N": args.samples,
            "M": args.variants,
            "Q": args.basis,
            "K": args.annotations,
            "B": args.probes,
            "J": args.jackknife_blocks,
        },
        "threads": threads,
        "variant_block_width": args.variant_block_width,
        "pass2_probe_width": args.pass2_probe_width,
        "phase_seconds": dict(result.telemetry["phase_seconds"]),
        "target_tn": dict(actual_tn),
        "mature_target_kernel_calibration": calibration,
        "executor_to_calibration_gflops_ratio": (
            actual_gflops / calibration_gflops
        ),
        "pass_ledger": result.ledger.to_dict(),
        "pass2_operator_counters": dict(
            result.telemetry["operator_pass2_counters"]
        ),
        "allocation_ledger": dict(result.telemetry["allocation_ledger"]),
        "checks": dict(result.telemetry["checks"]),
        "block_reconstruction_error": result.block_reconstruction_error,
        "presymmetry_absolute_error": result.presymmetry_absolute_error,
        "presymmetry_relative_error": result.presymmetry_relative_error,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "process_wall_seconds_including_setup_and_calibration": (
            time.perf_counter() - setup_started
        ),
        "same_person_reused_for_all_deletions": (
            result.same_person_reused_for_all_deletions
        ),
    }


def main() -> None:
    print(
        json.dumps(
            run(_parser().parse_args()),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
