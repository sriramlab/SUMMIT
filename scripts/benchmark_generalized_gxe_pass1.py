#!/usr/bin/env python3
"""Moderate source-only benchmark for generalized GxE pass 1.

The synthetic BED trio lives in a temporary directory. Genotype blocks are
decoded, mean-imputed, and common-scale standardized only by the mature
``GenomewideEnvLDScore._read_genotype_block`` operation composed through the
Stage 04 adapter. No target score is computed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.logger import Logger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--variants", type=int, default=4096)
    parser.add_argument("--basis", type=int, default=3)
    parser.add_argument("--annotations", type=int, default=3)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--variant-block-width", type=int, default=512)
    parser.add_argument("--probe-tile-width", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026082204)
    parser.add_argument("--memory-gib", type=float, default=2.0)
    return parser


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
    prefix = root / "pass1-benchmark"
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


def run(args: argparse.Namespace) -> dict:
    for name in (
        "samples",
        "variants",
        "basis",
        "annotations",
        "probes",
        "variant_block_width",
        "probe_tile_width",
        "threads",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.probes < 2:
        raise ValueError("--probes must be at least two")
    if not np.isfinite(args.memory_gib) or args.memory_gib <= 0.0:
        raise ValueError("--memory-gib must be positive and finite")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="summit-generalized-pass1-") as raw_root:
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
            plan = plan_generalized_gxe_variant_work(
                GeneralizedGxEPlanInputs(
                    num_samples=args.samples,
                    num_variants=args.variants,
                    num_basis=args.basis,
                    num_annotations=args.annotations,
                    num_probes=args.probes,
                    num_jackknife_blocks=20,
                    memory_limit_bytes=int(args.memory_gib * 1024**3),
                    genotype_format="bed",
                    threads=threads,
                    preferred_variant_block_width=args.variant_block_width,
                    preferred_rhs_tile_columns=args.probe_tile_width,
                    rhs_policy="tiled",
                )
            )
            spec = GlobalVariantProbeSpec(
                root_seed=args.seed,
                probe_offset=0,
                probe_count=args.probes,
            )
            result = GeneralizedGxEPass1Executor(
                genotype_operator=genotype_operator,
                basis=basis,
                fixed_effect_basis=fixed,
                annotations=annotations,
                annotation_names=tuple(
                    f"annotation_{index}" for index in range(args.annotations)
                ),
                annotation_masses=np.sum(annotations, axis=0),
                probe_spec=spec,
                work_plan=plan,
                nn_operator=ProtectedNNOperator(
                    threads=threads, native_module=gxeldcore
                ),
                annotation_tile_width=min(4, args.annotations),
                probe_tile_width=args.probe_tile_width,
                same_person_sample_tile_width=min(512, args.samples),
                native_probe_module=gxeldcore,
            ).execute()
        finally:
            estimator.close()
    telemetry = result.telemetry
    return {
        "schema": "summit.generalized_gxe.pass1_benchmark.v1",
        "scope": "source_only_no_target_scoring",
        "dimensions": {
            "N": args.samples,
            "M": args.variants,
            "Q": args.basis,
            "K": args.annotations,
            "B": args.probes,
        },
        "threads": threads,
        "variant_block_width": args.variant_block_width,
        "probe_tile_width": args.probe_tile_width,
        "phase_seconds": dict(telemetry["phase_seconds"]),
        "source_nn": dict(telemetry["source_nn"]),
        "pass_ledger": result.ledger.to_dict(),
        "operator_counters": dict(telemetry["operator_counters"]),
        "allocation_ledger": dict(telemetry["allocation_ledger"]),
        "projection": {
            "maximum_absolute_leakage": result.maximum_projection_leakage,
            "maximum_relative_leakage": (
                result.maximum_relative_projection_leakage
            ),
        },
        "same_person_presymmetry_error": result.same_person_presymmetry_error,
        "process_wall_seconds_including_synthetic_setup": (
            time.perf_counter() - started
        ),
        "target_scalability_claimed": False,
    }


def main() -> None:
    payload = run(_parser().parse_args())
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
