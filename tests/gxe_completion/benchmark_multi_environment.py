#!/usr/bin/env python3
"""Profile independent and shared GxE reference construction on fixed inputs."""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import resource
import tempfile
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

from summit.ldscore import gxe_multi
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.logger import Logger


def _inputs(root: Path, n: int, m: int, environments: int, seed: int):
    rng = np.random.default_rng(seed)
    frequency = rng.uniform(0.08, 0.48, size=m)
    raw = rng.binomial(2, frequency, size=(n, m)).astype(np.float64)
    prefix = root / "profile"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    base = rng.normal(size=n)
    values = {
        f"environment_{index}": (
            (0.55 ** index) * base
            + math.sqrt(max(0.0, 1.0 - 0.55 ** (2 * index)))
            * rng.normal(size=n)
        )
        for index in range(environments)
    }
    environment = root / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            **values,
        }
    ).to_csv(environment, sep="\t", index=False)
    covariates = root / "covariates.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "covariate_1": rng.normal(size=n),
            "covariate_2": rng.normal(size=n),
        }
    ).to_csv(covariates, sep="\t", index=False)
    return prefix, environment, covariates


def _estimator(
    prefix: Path,
    environment: Path,
    covariates: Path,
    output: Path,
    column: str,
    args,
    *,
    native_backend: str,
) -> GenomewideEnvLDScore:
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(output),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=str(covariates),
        num_vecs=args.probes,
        step_size=args.step_size,
        seed=args.seed,
        dtype="float64",
        num_threads=args.threads,
        target_xz_mem=args.memory_gib,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend=native_backend,
        native_workspace_gib=args.workspace_gib,
        native_target_panel_columns=max(64, 4 * args.probes),
    )


def _gemm_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for record in records:
        key = (
            record["operation"],
            tuple(record["left"]),
            tuple(record["right"]),
            tuple(record["output"]),
        )
        item = grouped.setdefault(
            key,
            {
                "operation": record["operation"],
                "left": record["left"],
                "right": record["right"],
                "output": record["output"],
                "calls": 0,
                "seconds": 0.0,
            },
        )
        item["calls"] += 1
        item["seconds"] += record["seconds"]
    return sorted(grouped.values(), key=lambda value: (
        value["operation"], value["left"], value["right"]
    ))


def _profile_shared(prefix, environment, covariates, root, args):
    estimators = [
        _estimator(
            prefix,
            environment,
            covariates,
            root / f"shared.{index}",
            f"environment_{index}",
            args,
            native_backend="python",
        )
        for index in range(args.environments)
    ]
    decode = {"calls": 0, "seconds": 0.0}
    original_read = estimators[0]._read_genotype_block

    def timed_read(self, start, stop):
        begin = time.perf_counter()
        try:
            return original_read(start, stop)
        finally:
            decode["calls"] += 1
            decode["seconds"] += time.perf_counter() - begin

    estimators[0]._read_genotype_block = types.MethodType(
        timed_read, estimators[0]
    )
    gemm_records: list[dict] = []
    original_nn = gxe_multi._MultiEnvironmentGemm.nn
    original_nn_update = gxe_multi._MultiEnvironmentGemm.nn_update
    original_tn = gxe_multi._MultiEnvironmentGemm.tn
    original_tn_pair = gxe_multi._MultiEnvironmentGemm.tn_pair

    def wrap(operation, function):
        def measured(executor, left, right):
            begin = time.perf_counter()
            result = function(executor, left, right)
            gemm_records.append(
                {
                    "operation": operation,
                    "left": list(left.shape),
                    "right": list(right.shape),
                    "output": list(result.shape),
                    "seconds": time.perf_counter() - begin,
                }
            )
            return result
        return measured

    gxe_multi._MultiEnvironmentGemm.nn = wrap("NN", original_nn)

    def measured_update(executor, left, right, target):
        begin = time.perf_counter()
        original_nn_update(executor, left, right, target)
        gemm_records.append(
            {
                "operation": "NN_UPDATE",
                "left": list(left.shape),
                "right": list(right.shape),
                "output": list(target.shape),
                "seconds": time.perf_counter() - begin,
            }
        )

    gxe_multi._MultiEnvironmentGemm.nn_update = measured_update
    gxe_multi._MultiEnvironmentGemm.tn = wrap("TN", original_tn)

    def measured_pair(executor, left, right_pair):
        begin = time.perf_counter()
        first, second = original_tn_pair(executor, left, right_pair)
        gemm_records.append(
            {
                "operation": "TN",
                "left": list(left.shape),
                "right": [int(right_pair.rows), int(2 * right_pair.columns)],
                "output": [int(first.shape[0]), int(2 * first.shape[1])],
                "seconds": time.perf_counter() - begin,
            }
        )
        return first, second

    gxe_multi._MultiEnvironmentGemm.tn_pair = measured_pair
    profile = None if args.no_profile else cProfile.Profile()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        if profile is not None:
            profile.enable()
        manifest_path = gxe_multi.generate_multi_environment_references(
            estimators,
            batch_manifest=root / "shared.gxe.multi.json",
            requested_backend="direct",
        )
        if profile is not None:
            profile.disable()
    finally:
        gxe_multi._MultiEnvironmentGemm.nn = original_nn
        gxe_multi._MultiEnvironmentGemm.nn_update = original_nn_update
        gxe_multi._MultiEnvironmentGemm.tn = original_tn
        gxe_multi._MultiEnvironmentGemm.tn_pair = original_tn_pair
        for estimator in estimators:
            estimator.close()
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    profile_path = None
    if profile is not None:
        profile_path = args.json.with_suffix(".pstats")
        profile.dump_stats(profile_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "mode": "shared",
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "decode": decode,
        "gemm": _gemm_summary(gemm_records),
        "gemm_total_calls": len(gemm_records),
        "gemm_total_seconds": sum(item["seconds"] for item in gemm_records),
        "shared_genotype_passes": manifest["shared_genotype_passes"],
        "repairs": manifest["repaired_gemm_output_columns"],
        "profile": None if profile_path is None else str(profile_path),
    }


def _profile_independent(prefix, environment, covariates, root, args):
    phases: Counter[str] = Counter()
    decode_calls = 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for index in range(args.environments):
        estimator = _estimator(
            prefix,
            environment,
            covariates,
            root / f"independent.{index}",
            f"environment_{index}",
            args,
            native_backend="direct",
        )
        blocks = len(estimator._make_compute_blocks())
        decode_calls += blocks * 3
        try:
            estimator._compute_ldscore()
            phases.update(estimator.native_phase_timings)
        finally:
            estimator.close()
    return {
        "mode": "independent",
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "decode": {"calls": decode_calls},
        "native_phase_seconds": dict(phases),
        "shared_genotype_passes": 3 * args.environments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("independent", "shared"), required=True)
    parser.add_argument("--environments", type=int, required=True)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--m", type=int, default=800)
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-gib", type=float, default=1.0)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    if args.environments < 1 or (args.mode == "shared" and args.environments < 2):
        raise ValueError("Environment count is invalid for the selected mode.")
    args.json = args.json.resolve()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".gxe-multi-profile-", dir=args.json.parent
    ) as temporary:
        root = Path(temporary)
        prefix, environment, covariates = _inputs(
            root, args.n, args.m, args.environments, args.seed
        )
        if args.mode == "shared":
            result = _profile_shared(
                prefix, environment, covariates, root, args
            )
        else:
            result = _profile_independent(
                prefix, environment, covariates, root, args
            )
    result.update(
        {
            "schema": "summit.gxe.multi_environment_profile.v1",
            "label": args.label,
            "arguments": {
                "environments": args.environments,
                "n": args.n,
                "m": args.m,
                "probes": args.probes,
                "step_size": args.step_size,
                "threads": args.threads,
                "memory_gib": args.memory_gib,
                "workspace_gib": args.workspace_gib,
                "seed": args.seed,
            },
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "affinity": sorted(os.sched_getaffinity(0)),
        }
    )
    args.json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
