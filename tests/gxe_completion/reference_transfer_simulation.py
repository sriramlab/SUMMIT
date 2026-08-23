#!/usr/bin/env python3
"""Reproduce the independent-reference Monte Carlo in the GxE report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPONENTS = ("G--G", "G--GxE", "GxE--GxE")


def _features(
    rng: np.random.Generator,
    n_samples: int,
    ar_cholesky: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    environment = rng.normal(size=n_samples)
    environment = (
        (environment - environment.mean()) / environment.std(ddof=1)
    )
    genotype = rng.normal(size=(n_samples, ar_cholesky.shape[0]))
    genotype = genotype @ ar_cholesky.T

    # Standardized E is exactly orthogonal to the intercept up to reduction
    # roundoff. Re-orthogonalize it so the projector construction is explicit.
    intercept = np.ones(n_samples, dtype=np.float64)
    intercept /= np.linalg.norm(intercept)
    environment_basis = environment - intercept * (intercept @ environment)
    environment_basis /= np.linalg.norm(environment_basis)
    basis = np.column_stack([intercept, environment_basis])

    additive = genotype - basis @ (basis.T @ genotype)
    interaction = environment[:, None] * genotype
    interaction -= basis @ (basis.T @ interaction)
    residual_rank = n_samples - basis.shape[1]

    additive *= np.sqrt(
        residual_rank / np.sum(additive * additive, axis=0)
    )
    interaction *= np.sqrt(
        residual_rank / np.sum(interaction * interaction, axis=0)
    )
    return additive, interaction, residual_rank


def _genetic_trace_and_diagonal(
    additive: np.ndarray,
    interaction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = (additive, interaction)
    n_variants = additive.shape[1]
    trace = np.asarray(
        [
            [
                np.sum((left.T @ right) ** 2) / n_variants**2
                for right in features
            ]
            for left in features
        ],
        dtype=np.float64,
    )
    kernel_diagonals = np.column_stack(
        [np.mean(feature * feature, axis=1) for feature in features]
    )
    same_individual = kernel_diagonals.T @ kernel_diagonals
    return trace, same_individual


def run_simulation(
    *,
    seed: int = 20260814,
    reference_samples: int = 2500,
    study_samples: int = 350,
    studies: int = 60,
    variants: int = 60,
    ar1_correlation: float = 0.35,
) -> dict:
    correlation = ar1_correlation ** np.abs(
        np.subtract.outer(np.arange(variants), np.arange(variants))
    )
    ar_cholesky = np.linalg.cholesky(correlation)
    rng = np.random.default_rng(seed)

    reference_additive, reference_interaction, reference_rank = _features(
        rng, reference_samples, ar_cholesky
    )
    reference_trace, reference_diagonal = _genetic_trace_and_diagonal(
        reference_additive, reference_interaction
    )

    transfer_errors = []
    naive_errors = []
    selected_entries = ((0, 0), (0, 1), (1, 1))
    transfer_same_scale = study_samples / reference_samples
    transfer_different_scale = (
        study_samples
        * (study_samples - 1)
        / (reference_samples * (reference_samples - 1))
    )

    for _ in range(studies):
        study_additive, study_interaction, study_rank = _features(
            rng, study_samples, ar_cholesky
        )
        study_trace, _ = _genetic_trace_and_diagonal(
            study_additive, study_interaction
        )
        transferred = (
            transfer_same_scale * reference_diagonal
            + transfer_different_scale
            * (reference_trace - reference_diagonal)
        )
        naive = (study_rank / reference_rank) ** 2 * reference_trace
        transfer_errors.append(
            [
                abs(transferred[index] - study_trace[index])
                / abs(study_trace[index])
                for index in selected_entries
            ]
        )
        naive_errors.append(
            [
                abs(naive[index] - study_trace[index])
                / abs(study_trace[index])
                for index in selected_entries
            ]
        )

    return {
        "schema": "summit.gxe.reference_transfer_simulation.v1",
        "design": {
            "seed": seed,
            "reference_samples": reference_samples,
            "study_samples": study_samples,
            "studies": studies,
            "variants": variants,
            "marker_distribution": "correlated Gaussian",
            "ar1_correlation": ar1_correlation,
            "projected_design": ["intercept", "standardized_environment"],
            "feature_convention": "post-projection per-variant standardization",
            "annotation_count": 1,
        },
        "components": list(COMPONENTS),
        "median_absolute_relative_error_percent": {
            "same_different_person_transfer": (
                100.0 * np.median(np.asarray(transfer_errors), axis=0)
            ).tolist(),
            "naive_squared_residual_rank_scaling": (
                100.0 * np.median(np.asarray(naive_errors), axis=0)
            ).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run_simulation()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
