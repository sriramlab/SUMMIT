#!/usr/bin/env python3
"""Aggregate validation of reference-panel convergence for contextual fits.

Each Monte Carlo replicate keeps the study genotype and phenotype fixed while
replacing the exact in-study genetic Gram with Grams transferred from
independent, matched-population reference cohorts.  A second phenotype draw on
the same study genotype separates phenotype-sampling variation from the
reference-substitution error.  The script writes aggregate statistics only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    build_context_reference,
    build_context_trait_summary,
    canonical_sha256,
    common_scale_features,
    dense_genetic_kernels,
    rank_revealing_projector,
)
from summit.context.fit import fit_context_model


OUTPUT_STEM = "04_fit_reference_convergence"
GENETIC_COEFFICIENTS = np.asarray([0.24, 0.10, 0.04], dtype=np.float64)
RESIDUAL_COEFFICIENT = 0.62
MAXIMUM_GENOTYPE_CONTEXT_LOADING = 0.30


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Write only the aggregate JSON result without importing Matplotlib.",
    )
    parser.add_argument("--replicates", type=int, default=24)
    parser.add_argument("--study-n", type=int, default=96)
    parser.add_argument(
        "--reference-sizes",
        type=int,
        nargs="+",
        default=(48, 192),
        metavar="N",
        help="At least two distinct independent-reference sample sizes.",
    )
    parser.add_argument("--m", type=int, default=24, help="Number of variants.")
    parser.add_argument("--loo-groups", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> tuple[int, ...]:
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2.")
    if args.study_n < 8 or args.study_n % 2:
        raise ValueError("--study-n must be an even integer of at least 8.")
    reference_sizes = tuple(sorted({int(value) for value in args.reference_sizes}))
    if len(reference_sizes) < 2:
        raise ValueError("--reference-sizes must contain at least two distinct values.")
    if any(value < 8 or value % 2 for value in reference_sizes):
        raise ValueError("Every reference size must be an even integer of at least 8.")
    if args.m < 6:
        raise ValueError("--m must be at least 6.")
    if args.loo_groups < 2 or args.loo_groups > args.m:
        raise ValueError("--loo-groups must satisfy 2 <= groups <= M.")
    if args.m % args.loo_groups:
        raise ValueError(
            "--m must be divisible by --loo-groups for balanced LOO groups."
        )
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    return reference_sizes


def _variant_loadings(m: int) -> np.ndarray:
    phase = 2.0 * np.pi * (np.arange(m, dtype=np.float64) + 0.5) / m
    pattern = np.sin(phase) + 0.35 * np.cos(3.0 * phase)
    pattern /= np.max(np.abs(pattern))
    return MAXIMUM_GENOTYPE_CONTEXT_LOADING * pattern


def _sample_matched_population(
    rng: np.random.Generator, *, n: int, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw an exactly balanced binary context and correlated unit-scale G."""
    context = np.concatenate(
        (-np.ones(n // 2, dtype=np.float64), np.ones(n // 2, dtype=np.float64))
    )
    rng.shuffle(context)
    loadings = _variant_loadings(m)
    noise_scale = np.sqrt(1.0 - loadings * loadings)
    genotype = (
        context[:, None] * loadings[None, :]
        + rng.standard_normal((n, m)) * noise_scale[None, :]
    )
    basis = np.column_stack((np.ones(n, dtype=np.float64), context))
    return np.asarray(genotype, dtype=np.float64), basis


def _projector(n: int):
    return rank_revealing_projector(np.empty((n, 0), dtype=np.float64))


def _build_reference(
    genotype: np.ndarray,
    basis: np.ndarray,
    *,
    components: ContextComponentIndex,
    annotations: np.ndarray,
    loo_groups: Sequence[str],
    hashes: dict[str, str],
):
    return build_context_reference(
        genotype=genotype,
        basis=basis,
        projector=_projector(genotype.shape[0]),
        annotations=annotations,
        component_index=components,
        basis_hash=hashes["basis"],
        fixed_effect_hash=hashes["fixed_effects"],
        variant_hash=hashes["variants"],
        loo_groups=loo_groups,
        gram_method="exact",
        same_person_method="exact",
    )


def _build_summary(
    genotype: np.ndarray,
    basis: np.ndarray,
    phenotype: np.ndarray,
    *,
    components: ContextComponentIndex,
    annotations: np.ndarray,
    loo_groups: Sequence[str],
    hashes: dict[str, str],
):
    n = genotype.shape[0]
    return build_context_trait_summary(
        genotype=genotype,
        basis=basis,
        phenotype=phenotype,
        projector=_projector(n),
        annotations=annotations,
        component_index=components,
        residual_basis=np.ones((n, 1), dtype=np.float64),
        residual_names=("identity",),
        basis_hash=hashes["basis"],
        fixed_effect_hash=hashes["fixed_effects"],
        variant_hash=hashes["variants"],
        loo_groups=loo_groups,
        block_size=genotype.shape[1],
    )


def _phenotypes(
    rng_a: np.random.Generator,
    rng_b: np.random.Generator,
    genotype: np.ndarray,
    basis: np.ndarray,
    components: ContextComponentIndex,
    annotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    features = common_scale_features(genotype, basis, np.eye(genotype.shape[0]))
    kernels = dense_genetic_kernels(features, annotations, components)
    covariance = np.einsum("a,aij->ij", GENETIC_COEFFICIENTS, kernels, optimize=True)
    covariance += RESIDUAL_COEFFICIENT * np.eye(genotype.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0.0:
        raise RuntimeError(
            "The declared phenotype covariance is not positive definite."
        )
    factor = np.linalg.cholesky(covariance)
    return (
        factor @ rng_a.standard_normal(genotype.shape[0]),
        factor @ rng_b.standard_normal(genotype.shape[0]),
        float(eigenvalues[0]),
    )


def _fit_record(reference: Any, summary: Any) -> tuple[np.ndarray, dict[str, Any]]:
    result = fit_context_model(reference, summary)
    coefficients = np.asarray(result.raw_coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or not np.all(np.isfinite(coefficients)):
        raise RuntimeError("fit_context_model returned invalid raw coefficients.")
    loo = np.asarray(result.loo_coefficients, dtype=np.float64)
    return coefficients, {
        "rank": int(result.solve.rank),
        "condition_number": float(result.solve.condition_number),
        "relative_residual": float(result.solve.relative_residual),
        "loo_replicates": int(loo.shape[0]),
    }


def _coefficient_summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "standard_deviation": np.std(values, axis=0, ddof=1).tolist(),
    }


def _error_summary(delta: np.ndarray) -> dict[str, Any]:
    coefficient_rms = np.sqrt(np.mean(delta * delta, axis=0))
    genetic_vector_rms = np.sqrt(
        np.mean(np.sum(delta[:, : GENETIC_COEFFICIENTS.size] ** 2, axis=1))
    )
    full_vector_rms = np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
    return {
        "coefficient_bias": np.mean(delta, axis=0).tolist(),
        "coefficient_rms": coefficient_rms.tolist(),
        "genetic_vector_rms": float(genetic_vector_rms),
        "full_vector_rms": float(full_vector_rms),
    }


def _diagnostic_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conditions = np.asarray([record["condition_number"] for record in records])
    residuals = np.asarray([record["relative_residual"] for record in records])
    return {
        "ranks_observed": sorted({int(record["rank"]) for record in records}),
        "loo_replicates_observed": sorted(
            {int(record["loo_replicates"]) for record in records}
        ),
        "maximum_condition_number": float(np.max(conditions)),
        "maximum_relative_solve_residual": float(np.max(residuals)),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    reference_sizes = _validate_arguments(args)
    started = time.perf_counter()
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    annotations = np.ones((args.m, 1), dtype=np.float64)
    loo_groups = tuple(
        f"group:{variant % args.loo_groups}" for variant in range(args.m)
    )
    hashes = {
        "basis": canonical_sha256(
            {"basis": ["constant", "balanced_binary_minus_one_plus_one"]}
        ),
        "fixed_effects": canonical_sha256({"fixed_effects": "none"}),
        "variants": canonical_sha256({"variants": list(range(args.m))}),
    }

    exact_a: list[np.ndarray] = []
    exact_b: list[np.ndarray] = []
    transferred: dict[int, list[np.ndarray]] = {
        reference_n: [] for reference_n in reference_sizes
    }
    exact_diagnostics: list[dict[str, Any]] = []
    transferred_diagnostics: dict[int, list[dict[str, Any]]] = {
        reference_n: [] for reference_n in reference_sizes
    }
    minimum_covariance_eigenvalues: list[float] = []

    for replicate in range(args.replicates):
        streams = np.random.SeedSequence([args.seed, replicate]).spawn(
            3 + len(reference_sizes)
        )
        study_genotype, study_basis = _sample_matched_population(
            np.random.default_rng(streams[0]), n=args.study_n, m=args.m
        )
        phenotype_a, phenotype_b, minimum_eigenvalue = _phenotypes(
            np.random.default_rng(streams[1]),
            np.random.default_rng(streams[2]),
            study_genotype,
            study_basis,
            components,
            annotations,
        )
        minimum_covariance_eigenvalues.append(minimum_eigenvalue)
        study_reference = _build_reference(
            study_genotype,
            study_basis,
            components=components,
            annotations=annotations,
            loo_groups=loo_groups,
            hashes=hashes,
        )
        summary_a = _build_summary(
            study_genotype,
            study_basis,
            phenotype_a,
            components=components,
            annotations=annotations,
            loo_groups=loo_groups,
            hashes=hashes,
        )
        summary_b = _build_summary(
            study_genotype,
            study_basis,
            phenotype_b,
            components=components,
            annotations=annotations,
            loo_groups=loo_groups,
            hashes=hashes,
        )
        coefficient_a, diagnostic_a = _fit_record(study_reference, summary_a)
        coefficient_b, _ = _fit_record(study_reference, summary_b)
        exact_a.append(coefficient_a)
        exact_b.append(coefficient_b)
        exact_diagnostics.append(diagnostic_a)

        for stream, reference_n in zip(streams[3:], reference_sizes):
            reference_genotype, reference_basis = _sample_matched_population(
                np.random.default_rng(stream), n=reference_n, m=args.m
            )
            reference = _build_reference(
                reference_genotype,
                reference_basis,
                components=components,
                annotations=annotations,
                loo_groups=loo_groups,
                hashes=hashes,
            )
            coefficient, diagnostic = _fit_record(reference, summary_a)
            transferred[reference_n].append(coefficient)
            transferred_diagnostics[reference_n].append(diagnostic)

    exact_a_array = np.asarray(exact_a, dtype=np.float64)
    exact_b_array = np.asarray(exact_b, dtype=np.float64)
    phenotype_sampling = _error_summary((exact_a_array - exact_b_array) / np.sqrt(2.0))
    reference_results: list[dict[str, Any]] = []
    for reference_n in reference_sizes:
        values = np.asarray(transferred[reference_n], dtype=np.float64)
        errors = _error_summary(values - exact_a_array)
        errors["coefficient_rms_over_phenotype_sampling_rms"] = (
            np.asarray(errors["coefficient_rms"])
            / np.maximum(
                np.asarray(phenotype_sampling["coefficient_rms"]),
                np.finfo(np.float64).tiny,
            )
        ).tolist()
        errors["genetic_vector_rms_over_phenotype_sampling_rms"] = float(
            errors["genetic_vector_rms"]
            / max(
                float(phenotype_sampling["genetic_vector_rms"]),
                np.finfo(np.float64).tiny,
            )
        )
        reference_results.append(
            {
                "reference_n": reference_n,
                "transferred_fit": _coefficient_summary(values),
                "transferred_minus_exact_same_phenotype": errors,
                "fit_diagnostics": _diagnostic_summary(
                    transferred_diagnostics[reference_n]
                ),
            }
        )

    small_error = reference_results[0]["transferred_minus_exact_same_phenotype"]
    large_error = reference_results[-1]["transferred_minus_exact_same_phenotype"]
    runtime = time.perf_counter() - started
    return {
        "kind": "summit.context.fit_reference_convergence_validation",
        "schema_version": 1,
        "seed": args.seed,
        "replicates": args.replicates,
        "study_n": args.study_n,
        "reference_sizes": list(reference_sizes),
        "n_variants": args.m,
        "q": 2,
        "component_order": list(components.names) + ["residual:identity"],
        "population": {
            "context": "exactly_balanced_binary_minus_one_plus_one",
            "maximum_genotype_context_correlation_loading": (
                MAXIMUM_GENOTYPE_CONTEXT_LOADING
            ),
            "genotype_scale": "fixed_population_mean_zero_variance_one",
            "reference_matching": "same_joint_population_independent_samples",
        },
        "generating_coefficients": {
            "genetic": GENETIC_COEFFICIENTS.tolist(),
            "residual_identity": RESIDUAL_COEFFICIENT,
            "minimum_covariance_eigenvalue_observed": float(
                np.min(minimum_covariance_eigenvalues)
            ),
        },
        "projector": "identity_no_fixed_effects",
        "residual_basis": ["identity"],
        "approximate_loo": {
            "groups": args.loo_groups,
            "variants_per_group": args.m // args.loo_groups,
            "assignment": "variant_index_modulo_group_count",
        },
        "error_source_definitions": {
            "phenotype_sampling": (
                "(exact fit for phenotype A - exact fit for independent phenotype B) "
                "/ sqrt(2), conditional on the same study genotype"
            ),
            "reference_substitution": (
                "transferred independent-reference fit - exact in-study-T fit, "
                "conditional on the same study phenotype"
            ),
        },
        "exact_in_study_t_fit": _coefficient_summary(exact_a_array),
        "phenotype_sampling": phenotype_sampling,
        "reference_results": reference_results,
        "convergence": {
            "largest_over_smallest_reference_genetic_vector_rms_ratio": float(
                large_error["genetic_vector_rms"]
                / max(float(small_error["genetic_vector_rms"]), np.finfo(float).tiny)
            ),
            "largest_reference_has_smaller_genetic_vector_rms": bool(
                large_error["genetic_vector_rms"] < small_error["genetic_vector_rms"]
            ),
        },
        "fit_diagnostics": {"exact_in_study_t": _diagnostic_summary(exact_diagnostics)},
        "runtime_seconds": float(runtime),
        "contains_row_data": False,
    }


def _write_figure(payload: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = [
        name.replace("context:all:", "omega[")
        .replace(",", ",")
        .replace("residual:identity", "residual")
        for name in payload["component_order"]
    ]
    names = [f"{name}]" if name.startswith("omega[") else name for name in names]
    reference_sizes = np.asarray(payload["reference_sizes"], dtype=np.float64)
    coefficient_rms = np.asarray(
        [
            result["transferred_minus_exact_same_phenotype"]["coefficient_rms"]
            for result in payload["reference_results"]
        ]
    )
    ratios = np.asarray(
        [
            result["transferred_minus_exact_same_phenotype"][
                "coefficient_rms_over_phenotype_sampling_rms"
            ]
            for result in payload["reference_results"]
        ]
    )
    colors = ("#1f6f8b", "#b05a3c", "#6c7a3d", "#6f4e7c")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    floor = np.finfo(np.float64).eps
    for index, (name, color) in enumerate(zip(names, colors)):
        axes[0].plot(
            reference_sizes,
            np.maximum(coefficient_rms[:, index], floor),
            "o-",
            color=color,
            linewidth=1.5,
            label=name,
        )
        axes[1].plot(
            reference_sizes,
            np.maximum(ratios[:, index], floor),
            "o-",
            color=color,
            linewidth=1.5,
            label=name,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(reference_sizes)
        axis.set_xticklabels([str(int(value)) for value in reference_sizes])
        axis.set_xlabel("Independent reference sample size")
        axis.grid(color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel("RMS transferred-minus-exact coefficient")
    axes[1].set_ylabel("Reference RMS / phenotype-sampling RMS")
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Contextual-fit convergence with matched independent references "
        f"(study N={payload['study_n']}, M={payload['n_variants']}, "
        f"R={payload['replicates']})",
        fontsize=11.5,
    )
    figure.savefig(output_dir / f"{OUTPUT_STEM}.png", dpi=300)
    figure.savefig(output_dir / f"{OUTPUT_STEM}.pdf")
    plt.close(figure)


def main() -> None:
    args = _arguments()
    payload = run_validation(args)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{OUTPUT_STEM}.json").write_text(serialized, encoding="utf-8")
    if not args.no_figure:
        _write_figure(payload, args.output_dir)


if __name__ == "__main__":
    main()
