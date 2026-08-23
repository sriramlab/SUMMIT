#!/usr/bin/env python3
"""Privacy-preserving real-genotype sanity checks for categorical contexts.

The script joins the declared PLINK FAM, covariate table, and phenotype tables
in memory, takes deterministic small sample/variant subsets, and writes only
aggregate fit diagnostics and figures.  It is deliberately a numerical sanity
check, not a significance or biological-discovery analysis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from bed_reader import open_bed

from summit.context import (
    ContextComponentIndex,
    array_sha256,
    build_categorical_preset,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    coefficients_to_omegas,
    derive_binary_context_fit,
    derive_categorical_context_fit,
    fit_context_model,
    rank_revealing_projector,
)


DEFAULT_GENOTYPE_PREFIX = Path(
    "/home/bronsonj/UKBB/ldscores/refsample_h2_sensitivity_20260813/"
    "onekg_matched_unrelated_20260813/eur_matching/"
    "UKB_EUR_300k.seed20260813.n5000.common"
)
DEFAULT_PHENOTYPE_ROOT = Path("/home/bronsonj/UKBB/asha/phens")
TRAITS = ("bmi", "hdl", "testosterone", "c_reactive_prot")
SEX_TRAITS = TRAITS
SMOKING_TRAITS = ("bmi", "hdl", "c_reactive_prot")
PC_COLUMNS = tuple(f"f.22009.0.{index}" for index in range(1, 21))
OUTPUT_STEMS = (
    "05_real_traits_sanity.json",
    "05_real_traits_variances.png",
    "05_real_traits_variances.pdf",
    "05_real_traits_surfaces.png",
    "05_real_traits_surfaces.pdf",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geno-prefix", type=Path, default=DEFAULT_GENOTYPE_PREFIX)
    parser.add_argument("--phenotype-root", type=Path, default=DEFAULT_PHENOTYPE_ROOT)
    parser.add_argument(
        "--covariate-file",
        type=Path,
        default=DEFAULT_PHENOTYPE_ROOT / "testosterone.covar",
    )
    parser.add_argument("--max-samples", type=int, default=320)
    parser.add_argument("--variants", type=int, default=96)
    parser.add_argument("--variant-candidate-multiplier", type=int, default=4)
    parser.add_argument("--loo-groups", type=int, default=12)
    parser.add_argument("--minimum-category-count", type=int, default=10)
    parser.add_argument("--maximum-missing-rate", type=float, default=0.05)
    parser.add_argument("--minimum-maf", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.max_samples < 50:
        raise ValueError("--max-samples must be at least 50.")
    if args.variants < 12:
        raise ValueError("--variants must be at least 12.")
    if args.loo_groups < 2 or args.variants % args.loo_groups:
        raise ValueError(
            "--loo-groups must be at least two and divide --variants exactly."
        )
    if args.variant_candidate_multiplier < 2:
        raise ValueError("--variant-candidate-multiplier must be at least two.")
    if args.minimum_category_count < 2:
        raise ValueError("--minimum-category-count must be at least two.")
    if not 0.0 <= args.maximum_missing_rate < 1.0:
        raise ValueError("--maximum-missing-rate must be in [0,1).")
    if not 0.0 < args.minimum_maf < 0.5:
        raise ValueError("--minimum-maf must be in (0,0.5).")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    required = [
        Path(f"{args.geno_prefix}.bed"),
        Path(f"{args.geno_prefix}.bim"),
        Path(f"{args.geno_prefix}.fam"),
        args.covariate_file,
    ]
    required.extend(args.phenotype_root / f"{name}.pheno" for name in TRAITS)
    required.append(args.phenotype_root / "smoking_status.pheno")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required real-data inputs are absent: {missing}.")


def _prepare_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    existing = [name for name in OUTPUT_STEMS if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing real-trait outputs: " + ", ".join(existing)
        )


def _read_unique_table(path: Path, *, usecols: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        usecols=list(usecols),
        dtype={"FID": str, "IID": str},
    )
    if frame[["FID", "IID"]].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate FID/IID pairs.")
    return frame


def _intersection_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    fam = pd.read_csv(
        Path(f"{args.geno_prefix}.fam"),
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype=str,
    )
    if fam.duplicated().any():
        raise ValueError("The PLINK FAM contains duplicate FID/IID pairs.")
    fam["bed_row"] = np.arange(len(fam), dtype=np.int64)
    covariate_columns = ("FID", "IID", "sex", "age", *PC_COLUMNS)
    covariates = _read_unique_table(args.covariate_file, usecols=covariate_columns)
    merged = fam.merge(
        covariates, on=["FID", "IID"], how="inner", validate="one_to_one"
    )
    for trait in (*TRAITS, "smoking_status"):
        phenotype = _read_unique_table(
            args.phenotype_root / f"{trait}.pheno",
            usecols=("FID", "IID", "pheno"),
        ).rename(columns={"pheno": trait})
        merged = merged.merge(
            phenotype, on=["FID", "IID"], how="inner", validate="one_to_one"
        )
    required_numeric = [
        "sex",
        "age",
        *PC_COLUMNS,
        *TRAITS,
        "smoking_status",
    ]
    values = merged[required_numeric].to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(values), axis=1) & np.all(values != -9.0, axis=1)
    intersection = merged.loc[valid].copy()
    intersection_size = len(intersection)
    if intersection_size < args.max_samples:
        raise ValueError(
            f"Only {intersection_size} complete cases remain; requested "
            f"--max-samples={args.max_samples}."
        )
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(
        rng.choice(intersection_size, size=args.max_samples, replace=False)
    )
    selected = intersection.iloc[chosen].sort_values("bed_row").reset_index(drop=True)
    return selected, intersection_size


def _decode_genotypes(
    args: argparse.Namespace, selected: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any], str]:
    bed = open_bed(Path(f"{args.geno_prefix}.bed"))
    candidate_count = min(
        bed.sid_count,
        args.variants * args.variant_candidate_multiplier,
    )
    candidate_indices = np.unique(
        np.linspace(0, bed.sid_count - 1, num=candidate_count, dtype=np.int64)
    )
    raw = bed.read(
        index=(selected["bed_row"].to_numpy(dtype=np.int64), candidate_indices),
        dtype="float64",
        order="C",
    )
    missing_rate = np.mean(np.isnan(raw), axis=0)
    means = np.nanmean(raw, axis=0)
    allele_frequency = means / 2.0
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    centered = raw - means[None, :]
    variances = np.nanvar(centered, axis=0, ddof=1)
    valid = (
        np.isfinite(means)
        & np.isfinite(variances)
        & (variances > np.finfo(np.float64).eps)
        & (missing_rate <= args.maximum_missing_rate)
        & (maf >= args.minimum_maf)
    )
    retained = np.flatnonzero(valid)
    if retained.size < args.variants:
        raise ValueError(
            f"Only {retained.size} of {candidate_indices.size} deterministic "
            "candidate variants passed missingness/MAF/variance QC."
        )
    retained = retained[: args.variants]
    chosen_indices = candidate_indices[retained]
    genotype = raw[:, retained].copy()
    chosen_means = means[retained]
    missing = np.isnan(genotype)
    if np.any(missing):
        genotype[missing] = np.broadcast_to(chosen_means[None, :], genotype.shape)[
            missing
        ]
    genotype -= chosen_means[None, :]
    scales = np.std(genotype, axis=0, ddof=1)
    genotype /= scales[None, :]
    variant_hash = canonical_sha256(
        {
            "source_variant_count": int(bed.sid_count),
            "selected_variant_indices": chosen_indices.tolist(),
            "selection": "evenly_spaced_candidates_then_qc",
        }
    )
    diagnostics = {
        "source_sample_count": int(bed.iid_count),
        "source_variant_count": int(bed.sid_count),
        "candidate_variant_count": int(candidate_indices.size),
        "selected_variant_count": int(args.variants),
        "selected_maf_minimum": float(np.min(maf[retained])),
        "selected_maf_median": float(np.median(maf[retained])),
        "selected_maf_maximum": float(np.max(maf[retained])),
        "selected_missing_rate_maximum": float(np.max(missing_rate[retained])),
        "post_imputation_scale_minimum": float(np.min(scales)),
        "post_imputation_scale_maximum": float(np.max(scales)),
    }
    return np.asarray(genotype, dtype=np.float64), diagnostics, variant_hash


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result -= np.mean(result, axis=0, keepdims=True)
    scales = np.std(result, axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("A requested nuisance covariate has zero/invalid scale.")
    result /= scales[None, :]
    return result


def _preset_hash(preset: Any) -> str:
    hashes = getattr(preset, "hashes", None)
    if isinstance(hashes, Mapping):
        for key in ("basis_hash", "preset_hash", "manifest_hash"):
            value = hashes.get(key)
            if isinstance(value, str) and len(value) == 64:
                return value
    for name in ("basis_hash", "digest"):
        value = getattr(preset, name, None)
        if isinstance(value, str) and len(value) == 64:
            return value
    return array_sha256(np.asarray(preset.basis, dtype=np.float64))


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _category_labels(preset: Any, categories: Sequence[float]) -> list[str]:
    for name in ("category_labels", "labels"):
        values = getattr(preset, name, None)
        if values is not None and len(values) == len(categories):
            return [str(value) for value in values]
    return [f"code_{value:g}" for value in categories]


def _category_counts(values: np.ndarray, categories: Sequence[float]) -> list[int]:
    return [int(np.sum(values == category)) for category in categories]


def _jackknife_standard_error(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=0)
    centered = array - mean
    variance = (
        (array.shape[0] - 1.0) / array.shape[0] * np.sum(centered * centered, axis=0)
    )
    return np.sqrt(np.maximum(variance, 0.0))


def _scalar_jackknife_summary(values: Sequence[float | None]) -> dict[str, Any]:
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    if valid.size != len(values) or valid.size < 2:
        return {
            "standard_error": None,
            "valid_replicates": int(valid.size),
            "status": "indeterminate_in_at_least_one_loo_replicate",
        }
    return {
        "standard_error": float(_jackknife_standard_error(valid)),
        "valid_replicates": int(valid.size),
        "status": "defined",
    }


def _correlation(covariance: float, left: float, right: float) -> float | None:
    if left <= 0.0 or right <= 0.0:
        return None
    return float(covariance / np.sqrt(left * right))


def _binary_mechanism(omega: np.ndarray) -> dict[str, float | None]:
    v0 = float(omega[0, 0])
    v1 = float(omega[1, 1])
    covariance = float(omega[0, 1])
    rho = _correlation(covariance, v0, v1)
    amplification = None
    if v0 > 0.0 and v1 > 0.0:
        amplification = float(0.5 * np.log(v1 / v0))
    orthogonal = None
    if v0 > 0.0:
        orthogonal = float(v1 - covariance * covariance / v0)
    return {
        "v0": v0,
        "v1": v1,
        "covariance": covariance,
        "rho": rho,
        "half_log_variance_ratio": amplification,
        "orthogonal_heterogeneity_1_given_0": orthogonal,
    }


def _fit_report(
    fit: Any,
    component_index: ContextComponentIndex,
    context_grid: np.ndarray,
    labels: Sequence[str],
) -> dict[str, Any]:
    p_genetic = len(component_index)
    raw_omega = np.asarray(fit.raw_omegas[0], dtype=np.float64)
    grid = np.asarray(context_grid, dtype=np.float64)
    surface = grid @ raw_omega @ grid.T
    loo_coefficients = np.asarray(fit.loo_coefficients, dtype=np.float64)
    loo_omegas = np.asarray(
        [
            coefficients_to_omegas(row[:p_genetic], component_index)[0]
            for row in loo_coefficients
        ]
    )
    loo_surfaces = np.einsum("iq,bqr,jr->bij", grid, loo_omegas, grid, optimize=True)
    surface_se = _jackknife_standard_error(loo_surfaces)
    variances = np.diag(surface)
    residual = np.asarray(fit.residual_coefficients, dtype=np.float64)
    residual_se = np.asarray(fit.standard_errors[p_genetic:], dtype=np.float64)
    pairs: list[dict[str, Any]] = []
    for left in range(len(labels)):
        for right in range(left, len(labels)):
            loo_correlations = [
                _correlation(
                    float(item[left, right]),
                    float(item[left, left]),
                    float(item[right, right]),
                )
                for item in loo_surfaces
            ]
            estimate = _correlation(
                float(surface[left, right]),
                float(variances[left]),
                float(variances[right]),
            )
            pairs.append(
                {
                    "left": labels[left],
                    "right": labels[right],
                    "covariance": float(surface[left, right]),
                    "covariance_standard_error": float(surface_se[left, right]),
                    "correlation": estimate,
                    "correlation_uncertainty": _scalar_jackknife_summary(
                        loo_correlations
                    ),
                    "correlation_status": (
                        "defined"
                        if estimate is not None
                        else "indeterminate_raw_variance"
                    ),
                }
            )
    mechanism: dict[str, Any] | None = None
    if len(labels) == 2:
        point = _binary_mechanism(surface)
        loo_mechanisms = [_binary_mechanism(item) for item in loo_surfaces]
        mechanism = {}
        for key, estimate in point.items():
            uncertainty = _scalar_jackknife_summary(
                [record[key] for record in loo_mechanisms]
            )
            mechanism[key] = {
                "estimate": estimate,
                "uncertainty": uncertainty,
                "status": (
                    "defined" if estimate is not None else "indeterminate_raw_variance"
                ),
            }
    trace_total = None
    if fit.context_outputs is not None:
        annotations = fit.context_outputs.get("annotations", [])
        if annotations:
            trace_total = float(annotations[0]["trace_annotation_total"])
    return {
        "coefficient_order": list(fit.equations.component_names),
        "raw_coefficients": np.asarray(fit.raw_coefficients).tolist(),
        "standard_errors": np.asarray(fit.standard_errors).tolist(),
        "jackknife_covariance": np.asarray(fit.jackknife_covariance).tolist(),
        "raw_genetic_omega": raw_omega.tolist(),
        "raw_genetic_covariance_surface": surface.tolist(),
        "surface_standard_errors": surface_se.tolist(),
        "genetic_variances": [
            {
                "category": label,
                "estimate": float(variances[index]),
                "standard_error": float(surface_se[index, index]),
            }
            for index, label in enumerate(labels)
        ],
        "residual_variances": [
            {
                "category": label,
                "estimate": float(residual[index]),
                "standard_error": float(residual_se[index]),
            }
            for index, label in enumerate(labels)
        ],
        "pairwise": pairs,
        "binary_mechanism": mechanism,
        "trace_annotation_total": trace_total,
        "solve": {
            "rank": int(fit.solve.rank),
            "dimension": int(fit.raw_coefficients.size),
            "condition_number": float(fit.solve.condition_number),
            "relative_residual": float(fit.solve.relative_residual),
            "minimum_gram_eigenvalue": float(fit.solve.minimum_gram_eigenvalue),
        },
        "loo_replicates": int(loo_coefficients.shape[0]),
    }


def _context_analysis(
    *,
    name: str,
    categories: Sequence[float],
    traits: Sequence[str],
    selected: pd.DataFrame,
    genotype: np.ndarray,
    annotations: np.ndarray,
    loo_groups: Sequence[str],
    variant_hash: str,
    minimum_category_count: int,
) -> list[dict[str, Any]]:
    context = selected[name].to_numpy(dtype=np.float64)
    preset = build_categorical_preset(
        context,
        categories=categories,
        source_name=name,
        binary=len(categories) == 2,
        annotation_names=("all",),
        minimum_category_count=minimum_category_count,
    )
    labels = _category_labels(preset, categories)
    observed_counts = np.asarray(preset.category_counts, dtype=np.int64)
    if np.any(observed_counts < minimum_category_count):
        raise ValueError(
            f"Context {name!r} has a selected category below the declared "
            f"minimum count {minimum_category_count}: {observed_counts.tolist()}."
        )
    q_count = preset.num_categories
    component_index = preset.component_index
    residual_names = preset.residual_names
    context_grid = preset.context_grid
    basis_metric = preset.basis_metric
    continuous = _standardize_columns(
        selected[["age", *PC_COLUMNS]].to_numpy(dtype=np.float64)
    )
    additional_fixed = [continuous]
    if name == "smoking_status":
        additional_fixed.insert(
            0, (selected["sex"].to_numpy(dtype=np.float64) == 2.0)[:, None]
        )
    fixed = preset.fixed_effect_design(np.column_stack(additional_fixed))
    projector = rank_revealing_projector(fixed)
    basis_hash = _preset_hash(preset)
    fixed_effect_hash = array_sha256(fixed)
    reference = build_context_reference(
        genotype=genotype,
        basis=preset.basis,
        projector=projector,
        annotations=annotations,
        component_index=component_index,
        basis_hash=basis_hash,
        fixed_effect_hash=fixed_effect_hash,
        variant_hash=variant_hash,
        loo_groups=loo_groups,
        genotype_scaling="selected_cohort_sample_centered_unit_variance",
        gram_method="exact",
        same_person_method="exact",
    )
    results: list[dict[str, Any]] = []
    for trait in traits:
        summary = build_context_trait_summary(
            genotype=genotype,
            basis=preset.basis,
            phenotype=selected[trait].to_numpy(dtype=np.float64),
            projector=projector,
            annotations=annotations,
            component_index=component_index,
            residual_basis=preset.residual_basis,
            residual_names=residual_names,
            basis_hash=basis_hash,
            fixed_effect_hash=fixed_effect_hash,
            variant_hash=variant_hash,
            loo_groups=loo_groups,
            genotype_scaling="selected_cohort_sample_centered_unit_variance",
            block_size=genotype.shape[1],
        )
        fit = fit_context_model(
            reference,
            summary,
            context_grid=context_grid,
            basis_metric=basis_metric,
        )
        derived = (
            derive_binary_context_fit(fit, preset)
            if q_count == 2
            else derive_categorical_context_fit(fit, preset)
        )
        derived_omega = (
            np.asarray(derived.raw_omega, dtype=np.float64)
            if q_count == 2
            else np.asarray(derived.raw_omegas[0], dtype=np.float64)
        )
        derived_consistency = float(
            np.max(
                np.abs(derived_omega - np.asarray(fit.raw_omegas[0], dtype=np.float64)),
                initial=0.0,
            )
        )
        results.append(
            {
                "trait": trait,
                "context": name,
                "categories": labels,
                "category_counts": _category_counts(context, categories),
                "n_samples": len(selected),
                "preset": {
                    "basis_hash": preset.basis_hash,
                    "category_order_hash": preset.category_order_hash,
                    "category_labels": _json_safe(preset.category_labels),
                    "category_counts": observed_counts.tolist(),
                    "binary": q_count == 2,
                    "manifest": _json_safe(preset.manifest),
                },
                "fixed_effect_rank": int(projector.rank),
                "residual_rank": int(projector.residual_rank),
                "derived_api_raw_omega_consistency_max_abs": derived_consistency,
                "fit": _fit_report(fit, component_index, context_grid, labels),
            }
        )
    return results


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    _validate_arguments(args)
    started = time.perf_counter()
    selected, intersection_size = _intersection_frame(args)
    genotype, genotype_diagnostics, variant_hash = _decode_genotypes(args, selected)
    annotations = np.ones((args.variants, 1), dtype=np.float64)
    loo_groups = tuple(
        f"group:{index % args.loo_groups}" for index in range(args.variants)
    )
    analyses = _context_analysis(
        name="sex",
        categories=(1.0, 2.0),
        traits=SEX_TRAITS,
        selected=selected,
        genotype=genotype,
        annotations=annotations,
        loo_groups=loo_groups,
        variant_hash=variant_hash,
        minimum_category_count=args.minimum_category_count,
    )
    analyses.extend(
        _context_analysis(
            name="smoking_status",
            categories=(0.0, 1.0, 2.0),
            traits=SMOKING_TRAITS,
            selected=selected,
            genotype=genotype,
            annotations=annotations,
            loo_groups=loo_groups,
            variant_hash=variant_hash,
            minimum_category_count=args.minimum_category_count,
        )
    )
    return {
        "kind": "summit.context.real_trait_categorical_sanity",
        "schema_version": 1,
        "purpose": "numerical_sanity_only_not_a_significance_gate",
        "public_apis_exercised": [
            "build_categorical_preset",
            "build_context_reference",
            "build_context_trait_summary",
            "fit_context_model",
            "derive_binary_context_fit",
            "derive_categorical_context_fit",
        ],
        "seed": args.seed,
        "input": {
            "genotype_prefix": str(args.geno_prefix),
            "source_fam_samples": genotype_diagnostics["source_sample_count"],
            "source_bim_variants": genotype_diagnostics["source_variant_count"],
            "complete_case_intersection_samples": intersection_size,
            "selected_samples": len(selected),
            "traits": list(TRAITS),
            "contexts": ["sex", "smoking_status"],
        },
        "selection": {
            "samples": "seeded_without_replacement_from_all_trait_complete_cases",
            "variants": "evenly_spaced_candidates_then_missingness_maf_variance_qc",
            "selected_variant_identity_sha256": variant_hash,
            "sample_identifiers_or_digest_written": False,
            "variant_rows_written": False,
        },
        "genotype_qc": genotype_diagnostics,
        "approximate_loo": {
            "groups": args.loo_groups,
            "variants_per_group": args.variants // args.loo_groups,
            "balanced": True,
        },
        "analyses": analyses,
        "runtime_seconds": float(time.perf_counter() - started),
        "contains_row_data": False,
        "contains_sample_identifiers": False,
    }


def _write_variance_figure(payload: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    contexts = (("sex", axes[0]), ("smoking_status", axes[1]))
    colors = ("#1f6f8b", "#b05a3c", "#6c7a3d")
    for context, axis in contexts:
        records = [item for item in payload["analyses"] if item["context"] == context]
        x = np.arange(len(records), dtype=np.float64)
        category_count = len(records[0]["categories"])
        offsets = np.linspace(-0.22, 0.22, category_count)
        for category_index in range(category_count):
            estimates = [
                item["fit"]["genetic_variances"][category_index]["estimate"]
                for item in records
            ]
            errors = [
                item["fit"]["genetic_variances"][category_index]["standard_error"]
                for item in records
            ]
            axis.errorbar(
                x + offsets[category_index],
                estimates,
                yerr=errors,
                fmt="o",
                capsize=3,
                color=colors[category_index],
                label=records[0]["categories"][category_index],
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [item["trait"].replace("c_reactive_prot", "CRP") for item in records],
            rotation=25,
            ha="right",
        )
        axis.set_title(context.replace("_", " "))
        axis.set_ylabel("Raw genetic variance coefficient")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Category", frameon=False, fontsize=8)
    figure.suptitle(
        "Small real-data categorical-context sanity check\n"
        "points are raw MoM estimates; bars are approximate SNP-group jackknife SEs",
        fontsize=11,
    )
    for suffix in ("png", "pdf"):
        path = output_dir / f"05_real_traits_variances.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None)
        path.chmod(0o600)
    plt.close(figure)


def _write_surface_figure(payload: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analyses = payload["analyses"]
    surfaces = [
        np.asarray(item["fit"]["raw_genetic_covariance_surface"]) for item in analyses
    ]
    limit = max(float(np.max(np.abs(surface))) for surface in surfaces)
    limit = max(limit, np.finfo(np.float64).eps)
    figure, axes = plt.subplots(2, 4, figsize=(11.2, 5.8), constrained_layout=True)
    image = None
    for axis, item, surface in zip(axes.flat, analyses, surfaces):
        image = axis.imshow(surface, cmap="RdBu_r", vmin=-limit, vmax=limit)
        labels = item["categories"]
        axis.set_xticks(np.arange(len(labels)))
        axis.set_yticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_title(
            f"{item['trait'].replace('c_reactive_prot', 'CRP')} × "
            f"{item['context'].replace('_', ' ')}",
            fontsize=9,
        )
        for row in range(surface.shape[0]):
            for column in range(surface.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{surface[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    for axis in axes.flat[len(analyses) :]:
        axis.axis("off")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.78, label="Raw genetic covariance")
    figure.suptitle(
        "One-hot context covariance surfaces (small real-data sanity subset)",
        fontsize=11,
    )
    for suffix in ("png", "pdf"):
        path = output_dir / f"05_real_traits_surfaces.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None)
        path.chmod(0o600)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    previous_umask = os.umask(0o077)
    try:
        _prepare_output_directory(args.output_dir)
        payload = run_validation(args)
        _write_variance_figure(payload, args.output_dir)
        _write_surface_figure(payload, args.output_dir)
        output = args.output_dir / "05_real_traits_sanity.json"
        output.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output.chmod(0o600)
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
