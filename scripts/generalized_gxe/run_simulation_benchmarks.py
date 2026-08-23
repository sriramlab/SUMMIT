#!/usr/bin/env python3
"""Fit generalized GxE simulation batches efficiently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from summit.context.reference_v1 import contextual_variant_order_allele_sha256_v1
from summit.context.spec import array_sha256, canonical_sha256
from summit.context.trait_v1 import load_contextual_trait_v1
from summit.ldscore.generalized_gxe_reference_v1 import (
    load_generalized_gxe_variant_reference_v1,
)

from workflow import (
    ReferenceRun,
    balanced_block_ids,
    canonical_json,
    file_sha256,
    full_fits,
    read_plink_axes,
    require_private_blis,
    restricted_diagonal_fit,
    run_reference,
    run_trait_batch,
    symmetric_context_residual_basis,
    zero_missingness_sha256,
)


def _load_batch(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = directory / "simulation_manifest.json"
    array_path = directory / "simulation_batch.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "summit.generalized_gxe.simulation_batch_v1":
        raise ValueError(f"unsupported simulation manifest {manifest_path}")
    if file_sha256(array_path) != manifest["outputs"]["arrays_sha256"]:
        raise RuntimeError(f"simulation arrays changed: {array_path}")
    with np.load(array_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    if int(manifest["genotype"]["complete_traversals"]) != 1:
        raise ValueError("simulation did not use exactly one genotype traversal")
    if int(manifest["genotype"]["variant_visits"]) != int(manifest["dimensions"]["M"]):
        raise ValueError("simulation genotype-visit ledger is incomplete")
    return manifest, arrays


def _validate_reused_trait(
    *,
    trait,
    axes,
    reference_run: ReferenceRun,
    basis: np.ndarray,
    fixed: np.ndarray,
    annotations: np.ndarray,
    block_ids: np.ndarray,
    block_labels: tuple[str, ...],
    phenotypes: np.ndarray,
    trait_names: tuple[str, ...],
    residual_basis: np.ndarray,
    residual_names: tuple[str, ...],
) -> None:
    """Reject a named trait artifact unless every reusable input is identical."""
    retained_samples = np.arange(axes.n, dtype=np.int64)
    retained_variants = np.arange(axes.m, dtype=np.int64)
    counted_a1 = np.ones(axes.m, dtype=np.uint8)
    identity = trait.manifest["identity"]
    expected_identity = {
        "sample_order_sha256": canonical_sha256(
            {"ordered_iids": list(axes.sample_ids)}
        ),
        "variant_order_allele_sha256": (
            contextual_variant_order_allele_sha256_v1(
                retained_variants,
                axes.variant_ids,
                axes.counted_alleles,
                axes.other_alleles,
                counted_a1,
            )
        ),
        "retained_sample_map_sha256": array_sha256(retained_samples),
        "retained_variant_order_sha256": array_sha256(retained_variants),
        "fixed_effect_spec_sha256": array_sha256(fixed),
        "basis_specification_sha256": array_sha256(basis),
        "basis_calibration_sha256": array_sha256(basis.T @ basis),
        "fixed_basis_sha256": array_sha256(fixed),
        "evaluated_phi_sha256": array_sha256(basis),
        "genotype_scale_plan_sha256": reference_run.artifact.scale_plan.digest,
        "missingness_sha256": zero_missingness_sha256(axes.m, axes.n),
        "annotation_map_sha256": array_sha256(annotations),
        "group_map_sha256": array_sha256(block_ids),
        "phenotype_batch_sha256": array_sha256(phenotypes),
        "residual_basis_sha256": array_sha256(residual_basis),
    }
    mismatches = [
        f"identity.{name}"
        for name, expected in expected_identity.items()
        if identity.get(name) != expected
    ]
    expected_values = {
        "N": (trait.n_samples, axes.n),
        "M": (trait.n_variants, axes.m),
        "trait_ids": (trait.trait_ids, trait_names),
        "residual_names": (trait.residual_names, residual_names),
        "annotation_names": (
            trait.component_index.annotation_names,
            ("all_variants",),
        ),
        "block_labels": (trait.group_ids, block_labels),
        "scale_plan": (
            trait.scale_plan.digest,
            reference_run.artifact.scale_plan.digest,
        ),
    }
    mismatches.extend(
        name
        for name, (observed, expected) in expected_values.items()
        if observed != expected
    )
    if mismatches:
        raise ValueError(
            "reused trait is incompatible: " + ", ".join(mismatches)
        )


def _summarize_batch(
    *,
    label: str,
    manifest: dict,
    arrays: dict[str, np.ndarray],
    reference_run,
    fits,
    restricted,
    trait_path: Path,
    trait_start: int,
    trait_stop: int,
    trait_seconds: float,
    fit_seconds: float,
    residual_pairs: tuple[tuple[int, int], ...],
) -> dict:
    fits = fits[trait_start:trait_stop]
    restricted = restricted[trait_start:trait_stop]
    pairs = reference_run.artifact.component_index.pair_index.entries
    residual_names = tuple(
        reference_run.artifact.manifest["axes"]["residual_components"]["names"]
    )
    component_names = [f"omega_{pair.q}_{pair.r}" for pair in pairs] + list(
        residual_names
    )
    estimates = np.vstack([fit.raw_coefficients for fit in fits])
    standard_errors = np.vstack([fit.raw_standard_errors for fit in fits])
    truth = np.empty_like(estimates)
    normalized_omega = arrays["normalized_omega"]
    for pair_index, pair in enumerate(pairs):
        truth[:, pair_index] = normalized_omega[:, pair.q, pair.r]
    for residual_index, pair in enumerate(residual_pairs):
        truth[:, len(pairs) + residual_index] = (
            arrays["normalized_residual_variance"]
            if pair == (0, 0)
            else 0.0
        )
    z_null = np.divide(
        estimates,
        standard_errors,
        out=np.full_like(estimates, np.nan),
        where=standard_errors > 0.0,
    )
    covered = np.abs(estimates - truth) <= 1.96 * standard_errors
    summaries = {}
    for index, name in enumerate(component_names):
        true_nonzero = np.abs(truth[:, index]) > 1.0e-12
        empirical_sd = float(np.std(estimates[:, index], ddof=1))
        empirical_error_sd = float(
            np.std(estimates[:, index] - truth[:, index], ddof=1)
        )
        mean_jackknife_se = float(np.mean(standard_errors[:, index]))
        summaries[name] = {
            "mean_truth": float(np.mean(truth[:, index])),
            "mean_estimate": float(np.mean(estimates[:, index])),
            "bias": float(np.mean(estimates[:, index] - truth[:, index])),
            "rmse": float(np.sqrt(np.mean((estimates[:, index] - truth[:, index]) ** 2))),
            "empirical_monte_carlo_sd": empirical_sd,
            "empirical_monte_carlo_error_sd": empirical_error_sd,
            "mean_jackknife_se": mean_jackknife_se,
            "empirical_sd_over_mean_jackknife_se": (
                empirical_sd / mean_jackknife_se
            ),
            "empirical_error_sd_over_mean_jackknife_se": (
                empirical_error_sd / mean_jackknife_se
            ),
            "coverage_95": float(np.mean(covered[:, index])),
            "null_rejection_rate_5pct": float(np.mean(np.abs(z_null[:, index]) > 1.96)),
            "true_nonzero": bool(np.all(true_nonzero)),
        }
    restricted_estimates = np.asarray([value["coefficients"] for value in restricted])
    restricted_se = np.asarray([value["standard_errors"] for value in restricted])
    restricted_names = restricted[0]["component_names"]
    truth_by_name = {
        **{
            f"omega_{index}_{index}": normalized_omega[:, index, index]
            for index in range(normalized_omega.shape[1])
        },
        **{
            name: (
                arrays["normalized_residual_variance"]
                if pair == (0, 0)
                else np.zeros(normalized_omega.shape[0], dtype=np.float64)
            )
            for name, pair in zip(residual_names, residual_pairs, strict=True)
        },
    }
    restricted_truth = np.column_stack(
        [truth_by_name[name] for name in restricted_names]
    )
    restricted_summary = {}
    for index, name in enumerate(restricted_names):
        restricted_summary[name] = {
            "mean_truth": float(np.mean(restricted_truth[:, index])),
            "mean_estimate": float(np.mean(restricted_estimates[:, index])),
            "bias": float(
                np.mean(restricted_estimates[:, index] - restricted_truth[:, index])
            ),
            "rmse": float(
                np.sqrt(
                    np.mean(
                        (restricted_estimates[:, index] - restricted_truth[:, index])
                        ** 2
                    )
                )
            ),
            "mean_jackknife_se": float(np.mean(restricted_se[:, index])),
        }
    return {
        "label": label,
        "trait_artifact": str(trait_path),
        "trait_count": trait_stop - trait_start,
        "shared_trait_indices": [trait_start, trait_stop],
        "timing_seconds": {
            "shared_multi_phenotype_trait_pass": trait_seconds,
            "shared_joint_fits": fit_seconds,
        },
        "component_names": component_names,
        "estimates": estimates.tolist(),
        "standard_errors": standard_errors.tolist(),
        "truth": truth.tolist(),
        "summary": summaries,
        "restricted_component_names": restricted_names,
        "restricted_estimates": restricted_estimates.tolist(),
        "restricted_standard_errors": restricted_se.tolist(),
        "restricted_summary": restricted_summary,
        "full_model_condition_numbers": [
            float(fit.manifest["solve"]["condition_number"]) for fit in fits
        ],
        "omega_1_2_available_in_diagonal_restriction": (
            "omega_1_2" in restricted_names
        ),
        "generating_omega": manifest["model"]["omega"],
    }


def _stop_gates(results: dict[str, dict], replicates_per_batch: int) -> dict:
    gates: dict[str, dict] = {}
    false_positive_ceiling = 0.10 if replicates_per_batch >= 100 else 0.24
    diagonal = results.get("diagonal_two_environment")
    if diagonal is not None:
        summary = diagonal["summary"]
        detection = [
            summary[f"omega_{q}_{q}"]["null_rejection_rate_5pct"]
            for q in range(3)
        ]
        null_rates = [
            summary[name]["null_rejection_rate_5pct"]
            for name in ("omega_0_1", "omega_0_2", "omega_1_2")
        ]
        null_se_ratios = [
            summary[name]["empirical_error_sd_over_mean_jackknife_se"]
            for name in ("omega_0_1", "omega_0_2", "omega_1_2")
        ]
        gates["diagonal"] = {
            "requested_replicates": replicates_per_batch,
            "all_requested_fits_completed": (
                diagonal["trait_count"] == replicates_per_batch
            ),
            "minimum_diagonal_detection_rate_at_least_0p60": (
                min(detection) >= 0.60
            ),
            "offdiagonal_false_positive_rate_ceiling": false_positive_ceiling,
            "maximum_offdiagonal_false_positive_rate_within_ceiling": (
                max(null_rates) <= false_positive_ceiling
            ),
            "all_null_empirical_se_ratios_between_0p75_and_1p35": (
                min(null_se_ratios) >= 0.75 and max(null_se_ratios) <= 1.35
            ),
            "detection_rates": detection,
            "false_positive_rates": null_rates,
            "null_empirical_error_sd_over_mean_jackknife_se": null_se_ratios,
        }
        gates["diagonal"]["passed"] = all(
            value for key, value in gates["diagonal"].items() if isinstance(value, bool)
        )
    generalized = results.get("offdiagonal_two_environment")
    if generalized is not None:
        target = generalized["summary"]["omega_1_2"]
        gates["offdiagonal"] = {
            "requested_replicates": replicates_per_batch,
            "all_requested_fits_completed": (
                generalized["trait_count"] == replicates_per_batch
            ),
            "omega_1_2_detection_rate_at_least_0p60": (
                target["null_rejection_rate_5pct"] >= 0.60
            ),
            "omega_1_2_mean_has_correct_sign": (
                target["mean_estimate"] * target["mean_truth"] > 0.0
            ),
            "omega_1_2_abs_bias_at_most_0p10": abs(target["bias"]) <= 0.10,
            "diagonal_restriction_cannot_parameterize_omega_1_2": not generalized[
                "omega_1_2_available_in_diagonal_restriction"
            ],
            "detection_rate": target["null_rejection_rate_5pct"],
        }
        gates["offdiagonal"]["passed"] = all(
            value for key, value in gates["offdiagonal"].items() if isinstance(value, bool)
        )
    gates["all_passed"] = bool(gates) and all(
        value["passed"] for value in gates.values() if isinstance(value, dict)
    )
    return gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--memory-gib", type=float, default=64.0)
    parser.add_argument("--probe-tile-width", type=int, default=4)
    parser.add_argument(
        "--source-probe-tile-width",
        type=int,
        default=None,
        help="pass-1 source tile; defaults to the full probe count",
    )
    parser.add_argument(
        "--reuse-trait",
        type=Path,
        help=(
            "load a sealed compatible multi-phenotype trait artifact without "
            "a study-genotype traversal"
        ),
    )
    parser.add_argument(
        "--reuse-reference",
        type=Path,
        help=(
            "load a sealed compatible generalized reference without another "
            "reference-genotype traversal"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {args.output}")
    batches = [_load_batch(path) for path in args.simulation]
    labels = [manifest["label"] for manifest, _ in batches]
    if len(set(labels)) != len(labels):
        raise ValueError("simulation labels must be unique")
    first_manifest, first_arrays = batches[0]
    replicates_per_batch = int(first_manifest["dimensions"]["replicates"])
    if replicates_per_batch < 2:
        raise ValueError("benchmark requires at least two replicates per batch")
    prefix = Path(first_manifest["genotype"]["prefix"])
    axes = read_plink_axes(prefix)
    source_hashes = first_manifest["genotype"]["source_sha256"]
    for suffix in (".bed", ".bim", ".fam"):
        source_path = Path(str(axes.prefix) + suffix)
        if file_sha256(source_path) != source_hashes[suffix]:
            raise RuntimeError(f"simulation genotype source changed: {source_path}")
    for manifest, arrays in batches:
        if Path(manifest["genotype"]["prefix"]).resolve() != axes.prefix:
            raise ValueError("simulation batches use different genotype sources")
        if manifest["genotype"]["source_sha256"] != source_hashes:
            raise ValueError("simulation batches have different genotype file hashes")
        for name in ("basis", "fixed_basis", "residual_basis", "sample_ids"):
            if not np.array_equal(arrays[name], first_arrays[name]):
                raise ValueError(f"simulation batches differ in shared {name}")
        if int(manifest["dimensions"]["replicates"]) != replicates_per_batch:
            raise ValueError("simulation batches have different replicate counts")
        if arrays["phenotypes"].shape != (axes.n, replicates_per_batch):
            raise ValueError("simulation phenotype dimensions do not match manifest")
        if arrays["trait_names"].shape != (replicates_per_batch,):
            raise ValueError("simulation trait-name count does not match manifest")
        if arrays["normalized_omega"].shape != (replicates_per_batch, 3, 3):
            raise ValueError("simulation normalized Omega dimensions are invalid")
        if arrays["normalized_residual_variance"].shape != (
            replicates_per_batch,
        ):
            raise ValueError("simulation residual-truth dimensions are invalid")
    from summit import gxeldcore

    build_info, threads, placement = require_private_blis(gxeldcore)
    annotations = np.ones((axes.m, 1), dtype=np.float64)
    block_ids, block_labels = balanced_block_ids(axes.m, args.blocks)
    residual_basis, residual_names, residual_pairs = (
        symmetric_context_residual_basis(first_arrays["basis"])
    )
    restricted_residual_indices = tuple(
        index for index, (left, right) in enumerate(residual_pairs) if left == right
    )
    memory_bytes = int(args.memory_gib * 1024**3)
    reference_reused = args.reuse_reference is not None
    if not reference_reused:
        args.output.mkdir(parents=True, mode=0o700)
        reference_started = time.perf_counter()
        reference_run = run_reference(
            axes=axes,
            basis=first_arrays["basis"],
            basis_names=("intercept", "environment_1", "environment_2"),
            fixed=first_arrays["fixed_basis"],
            annotations=annotations,
            annotation_names=("all_variants",),
            block_ids=block_ids,
            block_labels=block_labels,
            residual_names=residual_names,
            probes=args.probes,
            seed=args.seed,
            threads=threads,
            memory_bytes=memory_bytes,
            native_module=gxeldcore,
            output=args.output / f"simulation_reference_b{args.probes}",
            include_directional_panel=False,
            probe_tile_width=args.probe_tile_width,
            source_probe_tile_width=(
                args.source_probe_tile_width
                if args.source_probe_tile_width is not None
                else args.probes
            ),
        )
        reference_seconds = time.perf_counter() - reference_started
        if not np.array_equal(
            reference_run.native_result.affine_mean,
            first_arrays["affine_mean"],
        ):
            raise RuntimeError("simulator/reference affine means differ")
        if not np.array_equal(
            reference_run.native_result.affine_inverse_scale,
            first_arrays["affine_inverse_scale"],
        ):
            maximum = float(
                np.max(
                    np.abs(
                        reference_run.native_result.affine_inverse_scale
                        - first_arrays["affine_inverse_scale"]
                    )
                )
            )
            raise RuntimeError(
                f"simulator/reference inverse scales differ; max={maximum}"
            )
    else:
        reference_path = args.reuse_reference.resolve()
        artifact = load_generalized_gxe_variant_reference_v1(reference_path)
        artifact_axes = artifact.manifest["axes"]
        randomization = artifact.manifest["randomization"]
        provenance = artifact.manifest["provenance"]
        loaded_native_sha256 = file_sha256(Path(gxeldcore.__file__))
        retained = np.arange(axes.m, dtype=np.int64)
        counted_a1 = np.ones(axes.m, dtype=np.uint8)
        variant_digest = contextual_variant_order_allele_sha256_v1(
            retained,
            axes.variant_ids,
            axes.counted_alleles,
            axes.other_alleles,
            counted_a1,
        )
        sample_digest = canonical_sha256(
            {"ordered_iids": list(axes.sample_ids)}
        )
        mismatches = []
        expected = {
            "N": (artifact.n_samples, axes.n),
            "M": (artifact.n_variants, axes.m),
            "sample_order": (
                artifact_axes["samples"]["digest"],
                sample_digest,
            ),
            "variant_order_alleles": (
                artifact_axes["variants"]["digest"],
                variant_digest,
            ),
            "basis": (
                artifact_axes["basis"]["digest"],
                array_sha256(first_arrays["basis"]),
            ),
            "fixed": (
                artifact_axes["fixed_effects"]["digest"],
                array_sha256(first_arrays["fixed_basis"]),
            ),
            "annotations": (
                artifact_axes["annotations"]["digest"],
                array_sha256(annotations),
            ),
            "blocks": (
                artifact_axes["jackknife_blocks"]["digest"],
                array_sha256(block_ids),
            ),
            "block_labels": (
                tuple(artifact_axes["jackknife_blocks"]["block_labels"]),
                tuple(block_labels),
            ),
            "residual_names": (
                tuple(artifact_axes["residual_components"]["names"]),
                residual_names,
            ),
            "probe_count": (int(randomization["probe_count"]), args.probes),
            "probe_seed": (int(randomization["root_seed"]), args.seed),
            "native_binary_sha256": (
                provenance["native_binary_sha256"],
                loaded_native_sha256,
            ),
            "native_source_commit": (
                provenance["source_commit"],
                build_info["source_commit"],
            ),
            "native_source_tree": (
                provenance["source_tree_sha256"],
                build_info["source_tree_sha256"],
            ),
        }
        for name, (observed, requested) in expected.items():
            if observed != requested:
                mismatches.append(name)
        if mismatches:
            raise ValueError(
                "reused reference is incompatible: " + ", ".join(mismatches)
            )
        reference_run = ReferenceRun(
            artifact=artifact,
            native_result=SimpleNamespace(
                ledger={
                    "missing_genotype_calls": int(
                        np.sum(first_arrays["missing_counts"], dtype=np.int64)
                    )
                },
                genotype_scale_plan=artifact.scale_plan,
                affine_mean=np.asarray(first_arrays["affine_mean"]),
                affine_inverse_scale=np.asarray(
                    first_arrays["affine_inverse_scale"]
                ),
            ),
            artifact_path=reference_path,
        )
        reference_seconds = 0.0
    scale_plan = reference_run.artifact.scale_plan
    scale_hashes = {
        "affine_mean": (
            scale_plan.affine_mean_sha256,
            array_sha256(first_arrays["affine_mean"]),
        ),
        "affine_inverse_scale": (
            scale_plan.affine_inverse_scale_sha256,
            array_sha256(first_arrays["affine_inverse_scale"]),
        ),
    }
    bad_scale = [name for name, (left, right) in scale_hashes.items() if left != right]
    if bad_scale:
        raise RuntimeError(
            "simulator/reference sealed scale hashes differ: "
            + ", ".join(bad_scale)
        )
    phenotype_batches = [arrays["phenotypes"] for _, arrays in batches]
    combined_phenotypes = np.asfortranarray(np.column_stack(phenotype_batches))
    combined_trait_names = tuple(
        f"{manifest['label']}__{name}"
        for manifest, arrays in batches
        for name in arrays["trait_names"].tolist()
    )
    if args.reuse_trait is None:
        if reference_reused:
            args.output.mkdir(parents=True, mode=0o700)
        trait_started = time.perf_counter()
        trait, trait_path, trait_execution = run_trait_batch(
            axes=axes,
            reference_run=reference_run,
            basis=first_arrays["basis"],
            fixed=first_arrays["fixed_basis"],
            annotations=annotations,
            annotation_names=("all_variants",),
            block_ids=block_ids,
            block_labels=block_labels,
            phenotypes=combined_phenotypes,
            trait_names=combined_trait_names,
            residual_basis=residual_basis,
            residual_names=residual_names,
            threads=threads,
            memory_bytes=memory_bytes,
            native_module=gxeldcore,
            output=args.output / "all_simulations_trait_batch",
        )
        trait_seconds = time.perf_counter() - trait_started
    else:
        trait_path = args.reuse_trait.resolve()
        trait = load_contextual_trait_v1(trait_path)
        _validate_reused_trait(
            trait=trait,
            axes=axes,
            reference_run=reference_run,
            basis=first_arrays["basis"],
            fixed=first_arrays["fixed_basis"],
            annotations=annotations,
            block_ids=block_ids,
            block_labels=block_labels,
            phenotypes=combined_phenotypes,
            trait_names=combined_trait_names,
            residual_basis=residual_basis,
            residual_names=residual_names,
        )
        if reference_reused:
            args.output.mkdir(parents=True, mode=0o700)
        trait_seconds = 0.0
        trait_execution = {
            "reused_without_genotype_access": True,
            "source_artifact": str(trait_path),
            "source_manifest_sha256": trait.manifest_sha256,
        }
    fit_started = time.perf_counter()
    fits = full_fits(reference_run.artifact, trait)
    restricted = [
        restricted_diagonal_fit(
            reference_run.artifact,
            trait,
            index,
            residual_indices=restricted_residual_indices,
        )
        for index in range(trait.n_traits)
    ]
    fit_seconds = time.perf_counter() - fit_started
    results = {}
    trait_start = 0
    for (manifest, arrays), directory in zip(batches, args.simulation, strict=True):
        label = manifest["label"]
        trait_stop = trait_start + arrays["phenotypes"].shape[1]
        results[label] = _summarize_batch(
            label=label,
            manifest=manifest,
            arrays=arrays,
            reference_run=reference_run,
            fits=fits,
            restricted=restricted,
            trait_path=trait_path,
            trait_start=trait_start,
            trait_stop=trait_stop,
            trait_seconds=trait_seconds,
            fit_seconds=fit_seconds,
            residual_pairs=residual_pairs,
        )
        results[label]["simulation_directory"] = str(directory)
        trait_start = trait_stop
    if trait_start != trait.n_traits:
        raise RuntimeError("simulation trait slices do not cover the trait artifact")
    stop_gates = _stop_gates(results, replicates_per_batch)
    report = {
        "schema": "summit.generalized_gxe.simulation_benchmark_v1",
        "dimensions": {
            "N": axes.n,
            "M": axes.m,
            "Q": 3,
            "K": 1,
            "B": args.probes,
            "J": args.blocks,
            "H": len(residual_names),
            "replicates_per_batch": replicates_per_batch,
        },
        "backend": {
            "build_info": build_info,
            "openmp_placement": placement,
            "native_binary": str(gxeldcore.__file__),
            "native_binary_sha256": file_sha256(Path(gxeldcore.__file__)),
            "immutable_threads": threads,
        },
        "reference": {
            "artifact": str(reference_run.artifact_path),
            "runtime_seconds": reference_seconds,
            "reused_without_genotype_access": reference_reused,
            "pass_ledger": dict(reference_run.artifact.manifest["pass_ledger"]),
            "scale_plan_sha256": reference_run.artifact.scale_plan.digest,
            "simulator_scale_hashes_match_exactly": True,
        },
        "shared_trait_batch": {
            "artifact": str(trait_path),
            "trait_count": trait.n_traits,
            "runtime_seconds": trait_seconds,
            "fit_runtime_seconds": fit_seconds,
            "reused_without_genotype_access": args.reuse_trait is not None,
            "execution": trait_execution,
        },
        "batches": results,
        "stop_gates": stop_gates,
        "design_interpretation": {
            "non_general_restriction": (
                "Omega is diagonal; SNP effects for distinct environment bases "
                "are independent"
            ),
            "generalized_signal": (
                "Omega[1,2] is covariance of the same SNPs' sensitivity effects "
                "for environments 1 and 2"
            ),
            "biological_meaning": (
                "alleles that amplify response to one exposure systematically "
                "amplify or attenuate response to the other exposure"
            ),
            "why_genie_style_cannot_detect_it": (
                "the diagonal restriction has no omega_1_2 parameter or "
                "corresponding symmetrized cross-environment kernel"
            ),
            "residual_model": "full-context",
            "residual_names": list(residual_names),
            "diagonal_restriction_residual_names": [
                residual_names[index] for index in restricted_residual_indices
            ],
        },
    }
    result_path = args.output / "simulation_benchmark.json"
    result_path.write_text(canonical_json(report) + "\n")
    print(canonical_json({"result": str(result_path), "stop_gates": stop_gates}))
    return 0 if stop_gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
