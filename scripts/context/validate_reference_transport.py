#!/usr/bin/env python3
"""Monte Carlo validation of unequal-cohort contextual-Gram transport.

The simulation deliberately uses small dense kernels and fixed population
scales.  Reference and study cohorts are independent draws from the same
population, so any difference between the two transfer rules is attributable
to their sample-size scaling rather than population mismatch.  Only aggregate
Monte Carlo summaries are emitted.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    common_scale_features,
    dense_genetic_kernels,
    exact_same_person_matrix,
    kernel_gram,
    transfer_reference_gram,
)


@dataclass(frozen=True)
class PopulationSetting:
    name: str
    context_prevalence: float
    maximum_genotype_context_loading: float


SETTINGS = (
    PopulationSetting("balanced_independent", 0.50, 0.00),
    PopulationSetting("balanced_genotype_context_correlation", 0.50, 0.45),
    PopulationSetting("skewed_context", 0.20, 0.00),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for aggregate JSON and PNG/PDF figures.",
    )
    parser.add_argument("--replicates", type=int, default=80)
    parser.add_argument("--study-n", type=int, default=128)
    parser.add_argument(
        "--reference-sizes",
        type=int,
        nargs=2,
        default=(64, 256),
        metavar=("SMALL_N", "LARGE_N"),
    )
    parser.add_argument("--m", type=int, default=24, help="Number of variants.")
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> tuple[int, int]:
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2.")
    if args.study_n < 2:
        raise ValueError("--study-n must be at least 2.")
    if args.m < 2:
        raise ValueError("--m must be at least 2.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    reference_sizes = tuple(sorted(int(value) for value in args.reference_sizes))
    if reference_sizes[0] < 2:
        raise ValueError("Both --reference-sizes must be at least 2.")
    if reference_sizes[0] == reference_sizes[1]:
        raise ValueError("--reference-sizes must contain two distinct values.")
    if args.study_n in reference_sizes:
        raise ValueError(
            "The study size must differ from both reference sizes so the "
            "validation exercises unequal-cohort transport."
        )
    return reference_sizes


def _variant_loadings(m: int, maximum: float) -> np.ndarray:
    if maximum == 0.0:
        return np.zeros(m, dtype=np.float64)
    phase = 2.0 * np.pi * (np.arange(m, dtype=np.float64) + 0.5) / m
    pattern = np.sin(phase) + 0.35 * np.cos(3.0 * phase)
    pattern /= np.max(np.abs(pattern))
    return maximum * pattern


def _sample_population(
    rng: np.random.Generator,
    *,
    n: int,
    m: int,
    prevalence: float,
    maximum_loading: float,
) -> tuple[np.ndarray, np.ndarray]:
    indicator = rng.binomial(1, prevalence, size=n).astype(np.float64)
    context = (indicator - prevalence) / np.sqrt(prevalence * (1.0 - prevalence))
    loadings = _variant_loadings(m, maximum_loading)
    noise_scale = np.sqrt(1.0 - loadings * loadings)
    genotype = (
        context[:, None] * loadings[None, :]
        + rng.standard_normal((n, m)) * noise_scale[None, :]
    )
    basis = np.column_stack((np.ones(n, dtype=np.float64), context))
    return genotype, basis


def _gram_and_same_person(
    genotype: np.ndarray,
    basis: np.ndarray,
    *,
    components: ContextComponentIndex,
    annotations: np.ndarray,
    identity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = common_scale_features(genotype, basis, identity)
    kernels = dense_genetic_kernels(features, annotations, components)
    return kernel_gram(kernels), exact_same_person_matrix(kernels)


def _frobenius_rows(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("rij,rij->r", values, values, optimize=True))


def _method_metrics(
    estimates: np.ndarray,
    studies: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    mean_estimate = np.mean(estimates, axis=0)
    bias = mean_estimate - target
    target_norm = max(float(np.linalg.norm(target)), np.finfo(np.float64).tiny)
    centered = estimates - mean_estimate[None, :, :]
    target_errors = estimates - target[None, :, :]
    paired_errors = estimates - studies
    study_norms = np.maximum(_frobenius_rows(studies), np.finfo(np.float64).tiny)
    delta_standard_error = np.std(paired_errors, axis=0, ddof=1) / np.sqrt(
        estimates.shape[0]
    )
    return {
        "mean_estimate": mean_estimate.tolist(),
        "bias": bias.tolist(),
        "relative_frobenius_bias": float(np.linalg.norm(bias) / target_norm),
        "relative_estimator_spread": float(
            np.sqrt(np.mean(_frobenius_rows(centered) ** 2)) / target_norm
        ),
        "relative_rmse_to_mean_study_target": float(
            np.sqrt(np.mean(_frobenius_rows(target_errors) ** 2)) / target_norm
        ),
        "relative_rmse_to_independent_study": float(
            np.sqrt(np.mean(_frobenius_rows(paired_errors) ** 2)) / target_norm
        ),
        "mean_per_replicate_relative_error": float(
            np.mean(_frobenius_rows(paired_errors) / study_norms)
        ),
        "paired_difference_monte_carlo_se": delta_standard_error.tolist(),
    }


def _setting_validation(
    setting: PopulationSetting,
    *,
    setting_index: int,
    seed: int,
    replicates: int,
    study_n: int,
    reference_sizes: Sequence[int],
    m: int,
) -> dict[str, Any]:
    components = ContextComponentIndex(("all",), ContextPairIndex(2))
    annotations = np.ones((m, 1), dtype=np.float64)
    identities = {n: np.eye(n, dtype=np.float64) for n in {study_n, *reference_sizes}}
    study_grams: list[np.ndarray] = []
    correct: dict[int, list[np.ndarray]] = {n: [] for n in reference_sizes}
    blind: dict[int, list[np.ndarray]] = {n: [] for n in reference_sizes}
    same_person_fractions: dict[int, list[float]] = {n: [] for n in reference_sizes}
    observed_prevalence: list[float] = []

    for replicate in range(replicates):
        streams = np.random.SeedSequence([seed, setting_index, replicate]).spawn(
            1 + len(reference_sizes)
        )
        study_genotype, study_basis = _sample_population(
            np.random.default_rng(streams[0]),
            n=study_n,
            m=m,
            prevalence=setting.context_prevalence,
            maximum_loading=setting.maximum_genotype_context_loading,
        )
        observed_prevalence.append(float(np.mean(study_basis[:, 1] > 0.0)))
        study_gram, _ = _gram_and_same_person(
            study_genotype,
            study_basis,
            components=components,
            annotations=annotations,
            identity=identities[study_n],
        )
        study_grams.append(study_gram)

        for stream, reference_n in zip(streams[1:], reference_sizes):
            reference_genotype, reference_basis = _sample_population(
                np.random.default_rng(stream),
                n=reference_n,
                m=m,
                prevalence=setting.context_prevalence,
                maximum_loading=setting.maximum_genotype_context_loading,
            )
            reference_gram, same_person = _gram_and_same_person(
                reference_genotype,
                reference_basis,
                components=components,
                annotations=annotations,
                identity=identities[reference_n],
            )
            correct[reference_n].append(
                transfer_reference_gram(
                    reference_gram,
                    same_person,
                    reference_n=reference_n,
                    study_n=study_n,
                )
            )
            blind[reference_n].append((study_n / reference_n) ** 2 * reference_gram)
            same_person_fractions[reference_n].append(
                float(
                    np.linalg.norm(same_person)
                    / max(
                        float(np.linalg.norm(reference_gram)),
                        np.finfo(np.float64).tiny,
                    )
                )
            )

    studies = np.asarray(study_grams, dtype=np.float64)
    target = np.mean(studies, axis=0)
    results: list[dict[str, Any]] = []
    for reference_n in reference_sizes:
        correct_metrics = _method_metrics(
            np.asarray(correct[reference_n]), studies, target
        )
        blind_metrics = _method_metrics(np.asarray(blind[reference_n]), studies, target)
        correct_bias = correct_metrics["relative_frobenius_bias"]
        blind_bias = blind_metrics["relative_frobenius_bias"]
        correct_rmse = correct_metrics["relative_rmse_to_mean_study_target"]
        blind_rmse = blind_metrics["relative_rmse_to_mean_study_target"]
        results.append(
            {
                "reference_n": reference_n,
                "transfer_factors": {
                    "correct_same_person": study_n / reference_n,
                    "correct_different_person": (
                        study_n * (study_n - 1) / (reference_n * (reference_n - 1))
                    ),
                    "blind_all_entries": (study_n / reference_n) ** 2,
                },
                "mean_same_person_frobenius_fraction": float(
                    np.mean(same_person_fractions[reference_n])
                ),
                "correct_two_scale": correct_metrics,
                "blind_squared_n": blind_metrics,
                "blind_over_correct_bias_ratio": float(
                    blind_bias / max(correct_bias, np.finfo(np.float64).tiny)
                ),
                "blind_over_correct_target_rmse_ratio": float(
                    blind_rmse / max(correct_rmse, np.finfo(np.float64).tiny)
                ),
                "correct_has_smaller_bias": bool(correct_bias < blind_bias),
            }
        )

    small_correct = results[0]["correct_two_scale"]
    large_correct = results[-1]["correct_two_scale"]
    return {
        "name": setting.name,
        "context_prevalence": setting.context_prevalence,
        "mean_observed_study_prevalence": float(np.mean(observed_prevalence)),
        "maximum_genotype_context_loading": (setting.maximum_genotype_context_loading),
        "basis": ["constant", "population_standardized_binary_context"],
        "component_order": list(components.names),
        "mean_study_gram": target.tolist(),
        "results": results,
        "large_over_small_correct_target_rmse_ratio": float(
            large_correct["relative_rmse_to_mean_study_target"]
            / small_correct["relative_rmse_to_mean_study_target"]
        ),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    reference_sizes = _validate_arguments(args)
    started = time.perf_counter()
    settings = [
        _setting_validation(
            setting,
            setting_index=index,
            seed=args.seed,
            replicates=args.replicates,
            study_n=args.study_n,
            reference_sizes=reference_sizes,
            m=args.m,
        )
        for index, setting in enumerate(SETTINGS)
    ]
    elapsed = time.perf_counter() - started
    return {
        "kind": "summit.context.reference_transport_validation",
        "schema_version": 1,
        "seed": args.seed,
        "replicates": args.replicates,
        "study_n": args.study_n,
        "reference_sizes": list(reference_sizes),
        "n_variants": args.m,
        "feature_mode": "raw_projected",
        "projector": "identity_to_isolate_transport_scaling",
        "genotype_scale": "fixed_population_mean_zero_variance_one",
        "transfer_rules": {
            "correct_two_scale": "N for same-person plus N(N-1) for different-person",
            "blind_squared_n": "(study_n/reference_n)^2 times reference Gram",
        },
        "settings": settings,
        "runtime_seconds": float(elapsed),
        "contains_row_data": False,
    }


def _write_figure(payload: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    colors = ("#1f6f8b", "#b05a3c", "#6c7a3d")
    metrics = (
        ("relative_frobenius_bias", "Relative Frobenius bias"),
        (
            "relative_rmse_to_mean_study_target",
            "Relative RMSE to mean study target",
        ),
    )
    floor = np.finfo(np.float64).eps
    for axis, (metric, ylabel) in zip(axes, metrics):
        for color, setting in zip(colors, payload["settings"]):
            x = np.asarray(
                [record["reference_n"] for record in setting["results"]],
                dtype=np.float64,
            )
            correct = np.asarray(
                [
                    max(record["correct_two_scale"][metric], floor)
                    for record in setting["results"]
                ]
            )
            blind = np.asarray(
                [
                    max(record["blind_squared_n"][metric], floor)
                    for record in setting["results"]
                ]
            )
            label = setting["name"].replace("_", " ")
            axis.plot(
                x, correct, "o-", color=color, linewidth=1.5, label=f"{label}: correct"
            )
            axis.plot(
                x,
                blind,
                "s--",
                color=color,
                linewidth=1.1,
                alpha=0.78,
                label=f"{label}: blind",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(payload["reference_sizes"])
        axis.set_xticklabels([str(value) for value in payload["reference_sizes"]])
        axis.set_xlabel("Reference sample size")
        axis.set_ylabel(ylabel)
        axis.grid(color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[1].legend(fontsize=7.3, frameon=False, loc="best")
    figure.suptitle(
        "Matched-population contextual reference transport "
        f"(study N={payload['study_n']}, M={payload['n_variants']}, "
        f"R={payload['replicates']})",
        fontsize=12,
    )
    figure.savefig(output_dir / "03_reference_transport_validation.png", dpi=300)
    figure.savefig(output_dir / "03_reference_transport_validation.pdf")
    plt.close(figure)


def main() -> None:
    args = _arguments()
    payload = run_validation(args)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output_dir is None:
        print(serialized, end="")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "03_reference_transport_validation.json").write_text(
        serialized, encoding="utf-8"
    )
    _write_figure(payload, args.output_dir)


if __name__ == "__main__":
    main()
