#!/usr/bin/env python3
"""Run the N~10K real-age generalized GxE sanity benchmark."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import time

import numpy as np

from workflow import (
    align_table,
    balanced_inference_block_ids,
    canonical_json,
    fit_record,
    fixed_basis,
    full_fits,
    read_id_table,
    read_plink_axes,
    require_private_blis,
    restricted_diagonal_fit,
    run_reference,
    run_trait_batch,
)


DEFAULT_PILOT = Path("/home/bronsonj/SUMMIT_gxe_pilot_20260808")


def _load_real_inputs(axes, pilot: Path):
    input_dir = pilot / "validation" / "inputs"
    covar_ids, covar_columns = read_id_table(input_dir / "age_dbp_cc.covar")
    env_ids, env_columns = read_id_table(input_dir / "age_dbp_cc.env")
    pheno_ids, pheno_columns = read_id_table(input_dir / "age_dbp_cc.pheno")
    covariates = align_table(axes.sample_ids, covar_ids, covar_columns)
    environment = align_table(axes.sample_ids, env_ids, env_columns)["ENV"]
    dbp = align_table(axes.sample_ids, pheno_ids, pheno_columns)["PHENO"]
    sbp_ids, sbp_columns = read_id_table(
        "/home/bronsonj/UKBB/asha/phens/bp_systolic.pheno"
    )
    sbp = align_table(axes.sample_ids, sbp_ids, sbp_columns)["pheno"]
    for name, value in (("DBP", dbp), ("SBP", sbp)):
        if np.any(value == -9.0):
            raise ValueError(f"{name} has missing sentinel values on the benchmark rows")
    environment = np.asarray(environment, dtype=np.float64)
    # The established complete-case file is already standardized; retain its
    # exact values so this benchmark matches the prior mature analysis.
    design = np.column_stack(
        [
            np.ones(axes.n, dtype=np.float64),
            environment,
            *[covariates[name] for name in covariates],
        ]
    )
    fixed = fixed_basis(design)
    if fixed.shape[1] != 28:
        raise RuntimeError(f"expected fixed-effect rank 28, observed {fixed.shape[1]}")
    basis = np.asfortranarray(np.column_stack([np.ones(axes.n), environment]))
    phenotypes = np.asfortranarray(np.column_stack([dbp, sbp]))
    residual_basis = np.asfortranarray(
        np.column_stack(
            [
                np.ones(axes.n, dtype=np.float64),
                environment * environment,
                2.0 * environment,
            ]
        )
    )
    residual_names = (
        "identity",
        "age_squared_residual",
        "age_linear_residual",
    )
    residual_pairs = ((0, 0), (1, 1), (0, 1))
    return (
        basis,
        fixed,
        phenotypes,
        residual_basis,
        residual_names,
        residual_pairs,
    )


def _parse_genie(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    patterns = {
        "omega_0_0": r"Sigma\^2_g\[0\] : ([^ ]+)  SE : ([^\n]+)",
        "omega_1_1": r"Sigma\^2_gxe\[0\] : ([^ ]+)  SE : ([^\n]+)",
        "age_squared_residual": r"Sigma\^2_nxe\[0\] : ([^ ]+)  SE : ([^\n]+)",
        "identity": r"Sigma\^2_e : ([^ ]+)  SE : ([^\n]+)",
    }
    result = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"could not parse {name} from {path}")
        result[name] = {"estimate": float(match.group(1)), "se": float(match.group(2))}
    return result


def _compare_genie(restricted: dict, genie: dict) -> dict:
    result = {}
    for name, estimate, se in zip(
        restricted["component_names"],
        restricted["coefficients"],
        restricted["standard_errors"],
        strict=True,
    ):
        mature = genie[name]
        combined_se = float(np.hypot(se, mature["se"]))
        difference = float(estimate - mature["estimate"])
        result[name] = {
            "generalized_diagonal_restriction": float(estimate),
            "generalized_jackknife_se": float(se),
            "mature_genie_b10": mature["estimate"],
            "mature_genie_j10_se": mature["se"],
            "difference": difference,
            "difference_over_combined_se": difference / combined_se,
            "within_two_combined_se": abs(difference) <= 2.0 * combined_se,
        }
    return result


def _read_mature_panel(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    values: list[float] = []
    with gzip.open(path, "rt") as handle:
        header = handle.readline().split()
        if len(header) != 4 or header[-1] != "L2_0":
            raise ValueError(f"unexpected mature LD-score header in {path}")
        for line in handle:
            fields = line.split()
            ids.append(fields[1])
            values.append(float(fields[3]))
    return ids, np.asarray(values, dtype=np.float64)


def _panel_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left - right
    centered_left = left - np.mean(left)
    centered_right = right - np.mean(right)
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    return {
        "pearson_correlation": float(centered_left @ centered_right / denominator),
        "mean_left": float(np.mean(left)),
        "mean_right": float(np.mean(right)),
        "mean_difference": float(np.mean(difference)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_frobenius_difference": float(
            np.linalg.norm(difference) / max(np.linalg.norm(right), 1.0)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PILOT / "validation" / "geno" / "age_dbp_cc",
    )
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, nargs="+", default=[128, 1024])
    parser.add_argument(
        "--njack", type=int, default=200,
        help="post-hoc delete-block replicates for normal-equation inference",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--mode",
        choices=("summary", "composable"),
        default="summary",
        help=(
            "composable stores sample-aligned data and should not be used "
            "for public release"
        ),
    )
    parser.add_argument("--memory-gib", type=float, default=64.0)
    parser.add_argument("--variant-block-width", type=int, default=4096)
    parser.add_argument(
        "--probe-tile-width",
        type=int,
        default=None,
        help="pass-2 target tile; defaults to planner-selected",
    )
    parser.add_argument(
        "--source-probe-tile-width",
        type=int,
        default=None,
        help="pass-1 source tile; defaults to each run's full probe count",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {args.output}")
    args.output.mkdir(parents=True, mode=0o700)
    from summit import gxeldcore

    build_info, threads, placement = require_private_blis(gxeldcore)
    axes = read_plink_axes(args.prefix)
    (
        basis,
        fixed,
        phenotypes,
        residual_basis,
        residual_names,
        residual_pairs,
    ) = _load_real_inputs(axes, args.pilot)
    restricted_residual_indices = tuple(
        index for index, (left, right) in enumerate(residual_pairs) if left == right
    )
    annotations = np.ones((axes.m, 1), dtype=np.float64)
    inference_block_ids, inference_block_labels = balanced_inference_block_ids(
        axes.m, args.njack
    )
    memory_bytes = int(args.memory_gib * 1024**3)
    reference_runs = {}
    reference_times = {}
    for probes in args.probes:
        started = time.perf_counter()
        reference_runs[probes] = run_reference(
            axes=axes,
            basis=basis,
            basis_names=("intercept", "age"),
            fixed=fixed,
            annotations=annotations,
            annotation_names=("all_variants",),
            inference_block_ids=inference_block_ids,
            inference_block_labels=inference_block_labels,
            residual_names=residual_names,
            probes=probes,
            seed=args.seed,
            threads=threads,
            memory_bytes=memory_bytes,
            native_module=gxeldcore,
            output=args.output / f"age_b{probes}_reference",
            include_directional_panel=True,
            mode=args.mode,
            variant_block_width=args.variant_block_width,
            probe_tile_width=args.probe_tile_width,
            source_probe_tile_width=(
                args.source_probe_tile_width
                if args.source_probe_tile_width is not None
                else probes
            ),
        )
        reference_times[probes] = time.perf_counter() - started
    trait_started = time.perf_counter()
    trait, trait_path, trait_execution = run_trait_batch(
        axes=axes,
        reference_run=reference_runs[args.probes[0]],
        basis=basis,
        fixed=fixed,
        annotations=annotations,
        annotation_names=("all_variants",),
        inference_block_ids=inference_block_ids,
        inference_block_labels=inference_block_labels,
        phenotypes=phenotypes,
        trait_names=("diastolic_blood_pressure", "systolic_blood_pressure"),
        residual_basis=residual_basis,
        residual_names=residual_names,
        output=args.output / "age_bp_trait_batch",
        variant_block_width=args.variant_block_width,
    )
    trait_seconds = time.perf_counter() - trait_started
    fits = {}
    for probes, reference_run in reference_runs.items():
        full = full_fits(reference_run.artifact, trait)
        restricted = [
            restricted_diagonal_fit(
                reference_run.artifact,
                trait,
                index,
                residual_indices=restricted_residual_indices,
            )
            for index in range(trait.n_traits)
        ]
        fits[str(probes)] = {
            "full_generalized": [fit_record(value) for value in full],
            "diagonal_restriction": restricted,
        }
    genie = _parse_genie(
        args.pilot
        / "validation"
        / "genie_b10_j10_seed20260808"
        / "age_dbp"
    )
    dbp_genie_comparison = {
        str(probes): _compare_genie(fits[str(probes)]["diagonal_restriction"][0], genie)
        for probes in args.probes
    }
    probe_comparison = {}
    if 128 in reference_runs and 1024 in reference_runs:
        panel128 = reference_runs[128].artifact.directional_ldscores
        panel1024 = reference_runs[1024].artifact.directional_ldscores
        for name, target, source in (
            ("XX", 0, 0),
            ("XW", 0, 1),
            ("WX", 1, 0),
            ("WW", 1, 1),
        ):
            probe_comparison[name] = _panel_summary(
                panel128[:, target, source], panel1024[:, target, source]
            )
    mature_directory = args.pilot / "results" / "age_dbp_n10_seed20260808"
    mature_panel_comparison = {}
    panel = reference_runs[max(args.probes)].artifact.directional_ldscores
    for name, filename, target, source in (
        ("XX", "age_dbp.gxx.ldscore.gz", 0, 0),
        ("XW", "age_dbp.gxe.ldscore.gz", 0, 1),
        ("WX", "age_dbp.exg.ldscore.gz", 1, 0),
        ("WW", "age_dbp.gee.ldscore.gz", 1, 1),
    ):
        ids, mature_values = _read_mature_panel(mature_directory / filename)
        if tuple(ids) != axes.variant_ids:
            raise RuntimeError(f"mature {name} variant order differs")
        mature_panel_comparison[name] = _panel_summary(
            panel[:, target, source], mature_values
        )
    result = {
        "schema": "generalized_gxe_real_age_sanity_v1",
        "inputs": {
            "prefix": str(axes.prefix),
            "N": axes.n,
            "M": axes.m,
            "Q": basis.shape[1],
            "fixed_rank": fixed.shape[1],
            "residual_rank": axes.n - fixed.shape[1],
            "traits": ["diastolic_blood_pressure", "systolic_blood_pressure"],
            "reference_probe_counts": args.probes,
            "njack": args.njack,
            "seed": args.seed,
            "residual_names": list(residual_names),
        },
        "backend": {
            "build_info": build_info,
            "openmp_placement": placement,
            "native_binary": str(gxeldcore.__file__),
            "immutable_threads": threads,
        },
        "timing_seconds": {
            "references": {str(key): value for key, value in reference_times.items()},
            "multi_trait_pass": trait_seconds,
        },
        "artifacts": {
            "references": {
                str(key): str(value.artifact_path) for key, value in reference_runs.items()
            },
            "trait": str(trait_path),
        },
        "reference_ledgers": {
            str(key): dict(value.artifact.manifest["pass_ledger"])
            for key, value in reference_runs.items()
        },
        "trait_execution": trait_execution,
        "fits": fits,
        "dbp_mature_genie_comparison": dbp_genie_comparison,
        "b128_vs_b1024_directional_panel": probe_comparison,
        "b1024_vs_prior_mature_b10_directional_panel": mature_panel_comparison,
        "comparison_scope": {
            "exact_shared_model": (
                "diagonal Omega plus identity and age-squared residual"
            ),
            "generalized_extra_components": [
                "omega_0_1",
                "age_linear_residual",
            ],
            "prior_mature_scale": "HWE plus GENIE projected-column scaling",
            "generalized_scale": (
                "one sealed sample-SD affine scale across contexts"
            ),
            "interpretation": (
                "coefficient agreement is judged against combined jackknife "
                "noise; raw panel comparison is diagnostic because scales and "
                "probe streams differ"
            ),
        },
    }
    output = args.output / "real_age_sanity.json"
    output.write_text(canonical_json(result) + "\n")
    print(canonical_json({"result": str(output), "timing_seconds": result["timing_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
