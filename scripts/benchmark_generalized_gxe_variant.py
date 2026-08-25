#!/usr/bin/env python3
"""Reproducible planner and BED benchmark for generalized variant LD scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np

from summit.ldscore.generalized_gxe_native import (
    GeneralizedGxENativeBEDExecutor,
    generalized_gxe_performance_ledger_from_native,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _cpu_ids(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError("CPU IDs must be unique nonnegative integers")
    return result


def _plan(args: argparse.Namespace):
    return plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=args.samples,
            num_variants=args.variants,
            num_basis=args.basis,
            num_annotations=args.annotations,
            num_probes=args.probes,
            memory_limit_bytes=args.memory_gib * 1024**3,
            genotype_format="bed",
            threads=args.threads,
            preferred_variant_block_width=args.variant_block_width,
            preferred_rhs_tile_columns=(
                args.basis**2 * args.probe_tile_width
            ),
            rhs_policy="tiled",
        )
    )


def _science_inputs(args: argparse.Namespace):
    n = args.samples
    m = args.variants
    basis_rng = np.random.default_rng(args.seed + 1)
    basis_columns = [np.ones(n)]
    basis_columns.extend(
        basis_rng.normal(size=n) for _ in range(args.basis - 1)
    )
    basis = np.asfortranarray(np.column_stack(basis_columns))
    fixed_rng = np.random.default_rng(args.seed + 2)
    fixed, _ = np.linalg.qr(
        np.column_stack((np.ones(n), fixed_rng.normal(size=n))),
        mode="reduced",
    )
    fixed = np.asfortranarray(fixed)
    if args.annotation_layout == "overlap":
        annotation_rng = np.random.default_rng(args.seed + 3)
        annotations = annotation_rng.uniform(
            0.1, 1.4, size=(m, args.annotations)
        )
    else:
        annotations = np.zeros((m, args.annotations), dtype=np.float64)
        annotations[np.arange(m), np.arange(m) % args.annotations] = 1.0
    annotations = np.ascontiguousarray(annotations, dtype=np.float64)
    return basis, fixed, annotations


def _ensure_bed(args: argparse.Namespace) -> None:
    paths = [Path(str(args.prefix) + suffix) for suffix in (".bed", ".bim", ".fam")]
    present = [path.exists() for path in paths]
    if all(present):
        return
    if any(present):
        raise RuntimeError("benchmark BED prefix is partially present")
    if not args.generate:
        raise RuntimeError("benchmark BED is absent; pass --generate")
    from bed_reader import to_bed

    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    probabilities = rng.uniform(0.08, 0.48, size=args.variants)
    raw = rng.binomial(
        2, probabilities, size=(args.samples, args.variants)
    ).astype(np.float64)
    raw[rng.random(size=raw.shape) < args.missing_fraction] = np.nan
    to_bed(str(args.prefix) + ".bed", raw)


def _summarize_mappings(records: Sequence[dict[str, float]]) -> dict[str, Any]:
    if not records:
        return {}
    result: dict[str, Any] = {}
    for key in records[0]:
        values = [float(record[key]) for record in records]
        result[key] = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def _validate_clean_ledger(
    ledger: Mapping[str, Any], *, variants: int
) -> None:
    if (
        ledger["observed_reference_genotype_passes"] != 2
        or ledger["observed_retained_variant_visits"] != 2 * variants
        or ledger["duplicate_retained_variant_visits"] != 0
        or ledger["retry_count"] != 0
        or ledger["repair_count"] != 0
        or ledger["fallback_count"] != 0
        or ledger["integrity_failures"] != 0
    ):
        raise RuntimeError("benchmark did not produce a clean two-pass ledger")


def _one_execution(
    args: argparse.Namespace,
    plan: Any,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptors = {
        suffix: os.open(str(args.prefix) + suffix, flags)
        for suffix in (".bed", ".bim", ".fam")
    }
    try:
        executor = GeneralizedGxENativeBEDExecutor(
            stable_descriptors=descriptors,
            row_selection=None,
            ddof=1,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=tuple(
                f"annotation_{index}" for index in range(args.annotations)
            ),
            annotation_masses=np.sum(annotations, axis=0, dtype=np.float64),
            probe_spec=GlobalVariantProbeSpec(
                root_seed=args.seed,
                probe_offset=0,
                probe_count=args.probes,
            ),
            work_plan=plan,
            probe_tile_width=args.probe_tile_width,
            same_person_sample_tile_width=args.sample_tile_width,
            threads=args.threads,
            decode_threads=args.threads,
            retain_base_sources=False,
            backend=args.backend,
        )
        begin = time.perf_counter()
        result = executor.execute()
        end_to_end = time.perf_counter() - begin
        return {
            "end_to_end_wall_seconds": end_to_end,
            "ledger": dict(result.ledger),
            "telemetry": dict(result.telemetry),
            "performance_ledger": generalized_gxe_performance_ledger_from_native(
                result
            ),
        }
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from summit import gxeldcore

    _ensure_bed(args)
    basis, fixed, annotations = _science_inputs(args)
    plan = _plan(args)
    placement = None
    if args.cpu_ids is not None:
        if len(args.cpu_ids) != args.threads:
            raise RuntimeError("--cpu-ids count must equal --threads")
        placement = dict(
            gxeldcore.configure_openmp_placement(args.cpu_ids, args.threads)
        )
    gxeldcore.reset_gemm_telemetry()
    runs = [
        _one_execution(args, plan, basis, fixed, annotations)
        for _ in range(args.warmups + args.repeats)
    ]
    measured = runs[args.warmups :]
    for record in measured:
        _validate_clean_ledger(record["ledger"], variants=args.variants)
    wall = [record["end_to_end_wall_seconds"] for record in measured]
    last = measured[-1]
    telemetry = last["telemetry"]
    native_gemm = [dict(value) for value in gxeldcore.consume_gemm_telemetry()]
    return {
        "schema": "summit.generalized_gxe.variant_benchmark.v1",
        "shape": {
            "N": args.samples,
            "M": args.variants,
            "Q": args.basis,
            "K": args.annotations,
            "B": args.probes,
        },
        "configuration": {
            "backend": args.backend,
            "threads": args.threads,
            "cpu_ids": args.cpu_ids,
            "annotation_layout": args.annotation_layout,
            "variant_block_width": plan.tiling["variant_block_width"],
            "probe_tile_width": args.probe_tile_width,
            "rhs_tile_columns": plan.tiling["rhs_tile_columns"],
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "timing": {
            "measured_end_to_end_seconds": wall,
            "median_end_to_end_seconds": statistics.median(wall),
            "range_end_to_end_seconds": [min(wall), max(wall)],
            "phase_wall_seconds": _summarize_mappings(
                [record["telemetry"]["phase_wall_seconds"] for record in measured]
            ),
            "subphase_wall_seconds": _summarize_mappings(
                [
                    record["telemetry"]["subphase_wall_seconds"]
                    for record in measured
                ]
            ),
        },
        "ledger": last["ledger"],
        "performance_ledger": last["performance_ledger"],
        "native_vendor_gemm_records": native_gemm,
        "openmp_placement": placement,
        "build_info": dict(gxeldcore.build_info()),
        "plan": plan.to_dict(),
        "telemetry_policy": {
            "schema": telemetry["schema"],
            "algebraic_checksum_policy": telemetry["algebraic_checksum_policy"],
            "full_serial_output_witness_calls": telemetry[
                "full_serial_output_witness_calls"
            ],
            "numa_evidence": telemetry["numa_evidence"],
        },
    }


def _add_dimensions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=_positive, required=True)
    parser.add_argument("--variants", type=_positive, required=True)
    parser.add_argument("--basis", type=_positive, default=3)
    parser.add_argument("--annotations", type=_positive, default=1)
    parser.add_argument("--probes", type=_positive, default=128)
    parser.add_argument("--threads", type=_positive, default=1)
    parser.add_argument("--variant-block-width", type=_positive, default=4096)
    parser.add_argument("--probe-tile-width", type=_positive, default=16)
    parser.add_argument("--memory-gib", type=_positive, default=128)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or benchmark the generalized variant-probe exactly-two-pass "
            "GxE LD-score executor."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="emit a target-shape dry-run plan")
    _add_dimensions(plan)
    run = commands.add_parser("run", help="run repeated native BED measurements")
    _add_dimensions(run)
    run.add_argument("--prefix", type=Path, required=True)
    run.add_argument("--generate", action="store_true")
    run.add_argument("--missing-fraction", type=float, default=0.005)
    run.add_argument("--seed", type=int, default=80801)
    run.add_argument("--backend", choices=("dense", "packed"), default="dense")
    run.add_argument(
        "--annotation-layout", choices=("overlap", "disjoint"), default="overlap"
    )
    run.add_argument("--sample-tile-width", type=_positive, default=4096)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--repeats", type=_positive, default=3)
    run.add_argument("--cpu-ids", type=_cpu_ids)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.probe_tile_width > args.probes:
        raise SystemExit("--probe-tile-width cannot exceed --probes")
    if args.command == "plan":
        payload = {
            "schema": "summit.generalized_gxe.variant_benchmark_plan.v1",
            "plan": _plan(args).to_dict(),
        }
    else:
        if not 0.0 <= args.missing_fraction < 1.0:
            raise SystemExit("--missing-fraction must be in [0,1)")
        if args.warmups < 0:
            raise SystemExit("--warmups must be nonnegative")
        payload = _run(args)
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
