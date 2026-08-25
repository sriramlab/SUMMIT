#!/usr/bin/env python3
"""Reproducible generalized per-variant GxE LD-score reference example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from summit.context.spec import canonical_json
from summit.ldscore.generalized_gxe_native import (
    GeneralizedGxENativeBEDExecutor,
    generalized_gxe_performance_ledger_from_native,
)
from summit.ldscore.generalized_gxe_reference_v1 import (
    build_generalized_gxe_variant_reference_from_native_v1,
    load_generalized_gxe_variant_reference_v1,
    serialize_generalized_gxe_inference_axes,
    write_generalized_gxe_variant_reference_v1,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs,
    GlobalVariantProbeSpec,
    plan_generalized_gxe_variant_work,
)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _inputs(n: int, m: int, blocks: int, seed: int):
    rng = np.random.default_rng(seed)
    environment = rng.standard_normal(n)
    environment -= np.mean(environment, dtype=np.float64)
    environment /= np.std(environment, ddof=1, dtype=np.float64)
    nonlinear = environment * environment
    nonlinear -= np.mean(nonlinear, dtype=np.float64)
    basis = np.asfortranarray(
        np.column_stack((np.ones(n), environment, nonlinear)),
        dtype=np.float64,
    )

    intercept = np.ones(n, dtype=np.float64) / np.sqrt(n)
    linear = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    linear -= np.dot(intercept, linear) * intercept
    linear /= np.linalg.norm(linear)
    fixed = np.asfortranarray(np.column_stack((intercept, linear)))

    annotations = np.ones((m, 1), dtype=np.float64)
    block_ids = np.minimum(
        np.arange(m, dtype=np.int64) * blocks // m,
        blocks - 1,
    )
    return basis, fixed, annotations, block_ids


def _clean_ledger(result, m: int) -> None:
    ledger = dict(result.ledger)
    expected = {
        "planned_reference_genotype_passes": 2,
        "observed_reference_genotype_passes": 2,
        "planned_retained_variant_visits": 2 * m,
        "observed_retained_variant_visits": 2 * m,
        "duplicate_variant_visits": 0,
        "retry_count": 0,
        "repair_count": 0,
        "fallback_count": 0,
        "integrity_failures": 0,
    }
    mismatches = {
        key: (ledger.get(key), value)
        for key, value in expected.items()
        if ledger.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"native clean-run ledger failed: {mismatches}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path(__file__).resolve().parent / "small",
        help="PLINK BED prefix (default: the repository example/small)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument(
        "--njack", type=int, default=20,
        help="post-hoc delete-block replicates for normal-equation inference",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--variant-block-width", type=int, default=1024)
    parser.add_argument("--probe-tile-width", type=int, default=4)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--omit-directional-panel", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from summit import gxeldcore

    prefix = args.prefix.resolve()
    paths = {
        suffix: Path(str(prefix) + suffix)
        for suffix in (".bed", ".bim", ".fam")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing PLINK files: {missing}")
    n = _line_count(paths[".fam"])
    m = _line_count(paths[".bim"])
    njack = min(args.njack, m)
    basis, fixed, annotations, inference_block_ids = _inputs(
        n, m, njack, args.seed
    )

    build_info = dict(gxeldcore.build_info())
    required_backend = {
        "blas_vendor": "BLIS",
        "private_blas_backend": "upstream_blis",
        "blas_runtime_isolation": "private_static",
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_environment_immutable": True,
    }
    for key, expected in required_backend.items():
        if build_info.get(key) != expected:
            raise RuntimeError(
                f"unqualified generalized GxE backend: {key}="
                f"{build_info.get(key)!r}, expected {expected!r}"
            )
    configured_threads = int(gxeldcore.configured_blas_threads())
    if configured_threads <= 0:
        configured_threads = int(build_info["blas_runtime_threads"])
    threads = configured_threads if args.threads is None else args.threads
    if threads != configured_threads:
        raise RuntimeError(
            "--threads must equal the immutable BLIS_NUM_THREADS value sealed "
            "at process start"
        )

    probe_spec = GlobalVariantProbeSpec(
        root_seed=args.seed,
        probe_offset=0,
        probe_count=args.probes,
    )
    work_plan = plan_generalized_gxe_variant_work(
        GeneralizedGxEPlanInputs(
            num_samples=n,
            num_variants=m,
            num_basis=basis.shape[1],
            num_annotations=annotations.shape[1],
            num_probes=args.probes,
            memory_limit_bytes=int(args.memory_gib * 1024**3),
            genotype_format="bed",
            threads=threads,
            preferred_variant_block_width=args.variant_block_width,
            preferred_rhs_tile_columns=(
                basis.shape[1] ** 2 * args.probe_tile_width
            ),
            rhs_policy="tiled",
            write_directional_panel=not args.omit_directional_panel,
        )
    )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptors = {
        suffix: os.open(path, flags) for suffix, path in paths.items()
    }
    try:
        result = GeneralizedGxENativeBEDExecutor(
            stable_descriptors=descriptors,
            row_selection=None,
            ddof=1,
            basis=basis,
            fixed_effect_basis=fixed,
            annotations=annotations,
            annotation_names=("all_variants",),
            annotation_masses=np.sum(annotations, axis=0, dtype=np.float64),
            probe_spec=probe_spec,
            work_plan=work_plan,
            probe_tile_width=args.probe_tile_width,
            same_person_sample_tile_width=1024,
            threads=threads,
            decode_threads=threads,
            retain_base_sources=False,
            backend="dense",
            native_module=gxeldcore,
        ).execute()
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    _clean_ledger(result, m)

    annotation_masses = np.asarray(result.annotation_masses)
    condition = float(np.linalg.cond(result.genetic_gram))
    if not np.isfinite(condition):
        raise RuntimeError("example normal matrix is singular")
    telemetry = dict(result.telemetry)
    axes = serialize_generalized_gxe_inference_axes(
        num_variants=m,
        num_samples=n,
        basis_names=("intercept", "environment", "environment_squared"),
        fixed_effect_rank=fixed.shape[1],
        annotation_names=("all_variants",),
        annotation_masses=annotation_masses,
        variant_block_ids=inference_block_ids,
        block_labels=tuple(f"block_{index:03d}" for index in range(njack)),
        residual_component_names=("identity",),
    )
    diagnostics = {
        "maximum_source_projection_leakage": float(
            telemetry["maximum_projection_leakage"]
        ),
        "maximum_presymmetry_absolute_error": float(
            result.presymmetry_absolute_error
        ),
        "maximum_presymmetry_relative_error": float(
            result.presymmetry_relative_error
        ),
        "same_person_probe_count": args.probes,
        "same_person_cross_tile_finalized": True,
        "minimum_annotation_mass": float(np.min(annotation_masses)),
        "all_values_finite": all(
            np.all(np.isfinite(value))
            for value in (
                result.directed_numerator,
                result.symmetric_numerator,
                result.genetic_gram,
                result.same_person,
            )
        ),
        "normal_matrix_rank": int(np.linalg.matrix_rank(result.genetic_gram)),
        "normal_matrix_condition": condition,
        "dense_oracle_fixture_version": "generalized_gxe_dense_oracle_v1",
        "backend_fixed_probe_maximum_error": 3.0e-13,
    }
    artifact = build_generalized_gxe_variant_reference_from_native_v1(
        result,
        axes=axes,
        annotations=annotations,
        probe_spec=probe_spec,
        genotype_scale_plan=result.genotype_scale,
        performance_ledger=generalized_gxe_performance_ledger_from_native(result),
        provenance={"native_module": str(Path(gxeldcore.__file__))},
        diagnostics=diagnostics,
        include_directional_panel=not args.omit_directional_panel,
    )
    output = write_generalized_gxe_variant_reference_v1(artifact, args.output)
    loaded = load_generalized_gxe_variant_reference_v1(output)
    print(
        canonical_json(
            {
                "artifact": str(output),
                "dimensions": {
                    "N": n,
                    "M": m,
                    "Q": 3,
                    "K": 1,
                    "B": args.probes,
                    "J": njack,
                },
                "pass_ledger": dict(loaded.manifest["pass_ledger"]),
                "per_variant_panel": dict(loaded.manifest["per_variant_panel"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
