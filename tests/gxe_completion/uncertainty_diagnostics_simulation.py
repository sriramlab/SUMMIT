#!/usr/bin/env python3
"""Reproducible finite-probe and delete-block diagonal diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _same_person_estimates(sources: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate sum_i E[S_a(i)^2] E[S_b(i)^2] three ways."""
    squares = np.square(sources, dtype=np.float64)
    probes = squares.shape[2]
    sums = squares.sum(axis=2)
    same_probe = np.einsum("aib,cib->ac", squares, squares, optimize=True)
    unbiased = (
        np.einsum("ai,ci->ac", sums, sums, optimize=True) - same_probe
    ) / float(probes * (probes - 1))
    plugin_means = sums / float(probes)
    plugin = np.einsum("ai,ci->ac", plugin_means, plugin_means, optimize=True)

    split = probes // 2
    first = squares[:, :, :split].mean(axis=2)
    second = squares[:, :, split:].mean(axis=2)
    split_probe = 0.5 * (
        np.einsum("ai,ci->ac", first, second, optimize=True)
        + np.einsum("ai,ci->ac", second, first, optimize=True)
    )
    return {"u_statistic": unbiased, "split_probe": split_probe, "plugin": plugin}


def _features(seed: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples, variants = 120, 80
    latent = rng.normal(size=(samples, 8))
    loadings = rng.normal(scale=0.35, size=(8, variants))
    genotype = latent @ loadings + rng.normal(size=(samples, variants))
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    environment = 0.45 * latent[:, 0] + rng.standard_t(df=7, size=samples)
    environment = (environment - environment.mean()) / environment.std(ddof=1)
    interaction = genotype * environment[:, None]
    interaction -= interaction.mean(axis=0)

    annotations = np.ones((2, variants), dtype=np.float64)
    annotations[1] = 0.0
    annotations[1, :16] = 1.0
    annotations[1, 40:48] = 1.0
    masses = annotations.sum(axis=1)
    features = []
    for family in (genotype, interaction):
        for annotation, mass in zip(annotations, masses, strict=True):
            features.append(family * np.sqrt(annotation / mass)[None, :])
    kappas = np.stack([np.square(feature).sum(axis=1) for feature in features])
    truth = np.einsum("ai,ci->ac", kappas, kappas, optimize=True)
    return features, truth, annotations, environment


def _probe_comparison(seed: int, repetitions: int) -> dict:
    features, truth, _, _ = _features(seed)
    truth_norm = float(np.linalg.norm(truth))
    output: dict[str, dict] = {}
    root = np.random.SeedSequence(seed + 1)
    cases = [(distribution, probes) for distribution in ("rademacher", "gaussian") for probes in (2, 4, 16, 64)]
    for case_index, (distribution, probes) in enumerate(cases):
        rng = np.random.default_rng(root.spawn(len(cases))[case_index])
        totals = {name: np.zeros_like(truth) for name in ("u_statistic", "split_probe", "plugin")}
        squared_errors = {name: 0.0 for name in totals}
        non_psd = {name: 0 for name in totals}
        minimum_eigenvalues = {name: [] for name in totals}
        for _ in range(repetitions):
            if distribution == "rademacher":
                random = rng.choice((-1.0, 1.0), size=(80, probes))
            else:
                random = rng.normal(size=(80, probes))
            sources = np.stack([feature @ random for feature in features])
            for name, estimate in _same_person_estimates(sources).items():
                totals[name] += estimate
                squared_errors[name] += float(np.linalg.norm(estimate - truth) ** 2)
                minimum = float(np.linalg.eigvalsh(estimate)[0])
                minimum_eigenvalues[name].append(minimum)
                non_psd[name] += int(minimum < -1.0e-10 * max(1.0, truth_norm))
        case = {}
        for name in totals:
            mean = totals[name] / float(repetitions)
            case[name] = {
                "relative_bias_frobenius": float(np.linalg.norm(mean - truth) / truth_norm),
                "relative_rmse_frobenius": float(
                    np.sqrt(squared_errors[name] / repetitions) / truth_norm
                ),
                "non_psd_fraction": non_psd[name] / float(repetitions),
                "median_minimum_eigenvalue": float(np.median(minimum_eigenvalues[name])),
            }
        output[f"{distribution}_B{probes}"] = case
    return {
        "samples": 120,
        "variants": 80,
        "components": 4,
        "repetitions": repetitions,
        "truth_minimum_eigenvalue": float(np.linalg.eigvalsh(truth)[0]),
        "cases": output,
    }


def _delete_block_comparison(seed: int) -> dict:
    features, _, annotations, _ = _features(seed)
    raw_families = [
        features[0] * np.sqrt(annotations[0].sum()),
        features[2] * np.sqrt(annotations[0].sum()),
    ]
    masses = annotations.sum(axis=1)
    full_rows = []
    for family in raw_families:
        full_rows.extend([np.square(family) @ annotation for annotation in annotations])
    full_rows = np.stack(full_rows)
    component_masses = np.tile(masses, 2)
    full_kappa = full_rows / component_masses[:, None]
    full_diagonal = full_kappa @ full_kappa.T

    relative_errors = []
    block_records = []
    for block, indices in enumerate(np.array_split(np.arange(80), 10)):
        deleted_rows = []
        deleted_masses = []
        for family in raw_families:
            squared = np.square(family[:, indices])
            for annotation in annotations:
                deleted_rows.append(squared @ annotation[indices])
                deleted_masses.append(float(annotation[indices].sum()))
        remaining_masses = component_masses - np.asarray(deleted_masses)
        if np.any(remaining_masses <= 0.0):
            raise RuntimeError("Synthetic deletion unexpectedly removed a full annotation.")
        exact_kappa = (full_rows - np.stack(deleted_rows)) / remaining_masses[:, None]
        exact_diagonal = exact_kappa @ exact_kappa.T
        relative = float(
            np.linalg.norm(full_diagonal - exact_diagonal)
            / max(np.linalg.norm(exact_diagonal), np.finfo(np.float64).tiny)
        )
        relative_errors.append(relative)
        block_records.append(
            {
                "block": block,
                "variants": indices.tolist(),
                "relative_frobenius_error_full_diagonal_reuse": relative,
                "minimum_remaining_annotation_mass": float(remaining_masses.min()),
            }
        )
    return {
        "blocks": 10,
        "median_relative_frobenius_error": float(np.median(relative_errors)),
        "maximum_relative_frobenius_error": float(np.max(relative_errors)),
        "records": block_records,
    }


def _jackknife_calibration(seed: int, repetitions: int) -> dict:
    """Compare exact and full-diagonal-reuse SEs in the production K=1 layout."""
    samples, variants, blocks = 120, 800, 100
    rng = np.random.default_rng(seed + 19)
    latent = rng.normal(size=(samples, 12))
    genotype = (
        latent @ rng.normal(scale=0.25, size=(12, variants))
        + rng.normal(size=(samples, variants))
    )
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    environment = 0.45 * latent[:, 0] + rng.standard_t(df=7, size=samples)
    environment = (
        (environment - environment.mean()) / environment.std(ddof=1)
    )
    interaction = genotype * environment[:, None]
    interaction -= interaction.mean(axis=0)
    families = (genotype, interaction)

    numerators = tuple(family @ family.T for family in families)
    full_kernels = tuple(numerator / variants for numerator in numerators)

    def gram(kernels: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.asarray([
            [np.sum(left * right) for right in kernels]
            for left in kernels
        ])

    def diagonal_gram(kernels: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.asarray([
            [np.dot(np.diag(left), np.diag(right)) for right in kernels]
            for left in kernels
        ])

    full_matrix = gram(full_kernels)
    full_diagonal = diagonal_gram(full_kernels)
    full_traces = np.asarray([np.trace(kernel) for kernel in full_kernels])
    exact_matrices = []
    approximate_matrices = []
    deleted_traces = []
    matrix_errors = []
    for indices in np.array_split(np.arange(variants), blocks):
        remaining = variants - len(indices)
        deleted_kernels = tuple(
            (
                numerator
                - family[:, indices] @ family[:, indices].T
            ) / remaining
            for numerator, family in zip(numerators, families, strict=True)
        )
        exact = gram(deleted_kernels)
        exact_diagonal = diagonal_gram(deleted_kernels)
        approximate = exact + full_diagonal - exact_diagonal
        exact_matrices.append(exact)
        approximate_matrices.append(approximate)
        deleted_traces.append([
            np.trace(kernel) for kernel in deleted_kernels
        ])
        matrix_errors.append(
            float(np.linalg.norm(approximate - exact) / np.linalg.norm(exact))
        )

    exact_matrices_array = np.asarray(exact_matrices)
    approximate_matrices_array = np.asarray(approximate_matrices)
    deleted_traces_array = np.asarray(deleted_traces)
    approximate_transforms = np.linalg.solve(
        approximate_matrices_array, exact_matrices_array
    )
    calibration_repetitions = max(5_000, int(repetitions))

    def scenario(
        name: str,
        coefficients: np.ndarray,
        perturbation_sd: np.ndarray,
        stream: int,
    ) -> dict:
        scenario_rng = np.random.default_rng(seed + stream)
        perturbations = scenario_rng.normal(
            size=(calibration_repetitions, blocks, 2)
        ) * perturbation_sd
        point_coefficients = coefficients + perturbations.mean(axis=1)
        truth = coefficients * full_traces / np.dot(coefficients, full_traces)
        point_proportions = point_coefficients * full_traces[None, :]
        point_proportions /= point_proportions.sum(axis=1)[:, None]

        deleted_coefficients = coefficients + (
            perturbations.sum(axis=1)[:, None, :] - perturbations
        ) / float(blocks - 1)
        exact_proportions = (
            deleted_coefficients * deleted_traces_array[None, :, :]
        )
        exact_proportions /= exact_proportions.sum(axis=2)[:, :, None]
        approximate_coefficients = np.einsum(
            "bij,rbj->rbi",
            approximate_transforms,
            deleted_coefficients,
            optimize=True,
        )
        approximate_proportions = (
            approximate_coefficients * deleted_traces_array[None, :, :]
        )
        approximate_proportions /= approximate_proportions.sum(
            axis=2
        )[:, :, None]

        factor = (blocks - 1.0) / blocks
        exact_se = np.sqrt(
            factor * np.sum(
                np.square(
                    exact_proportions
                    - exact_proportions.mean(axis=1)[:, None, :]
                ),
                axis=1,
            )
        )
        approximate_se = np.sqrt(
            factor * np.sum(
                np.square(
                    approximate_proportions
                    - approximate_proportions.mean(axis=1)[:, None, :]
                ),
                axis=1,
            )
        )
        empirical_sd = point_proportions.std(axis=0, ddof=1)
        exact_coverage = np.mean(
            np.abs(point_proportions - truth[None, :]) <= 1.96 * exact_se,
            axis=0,
        )
        approximate_coverage = np.mean(
            np.abs(point_proportions - truth[None, :])
            <= 1.96 * approximate_se,
            axis=0,
        )
        se_ratio = approximate_se / exact_se
        result = {
            "name": name,
            "true_proportions": truth.tolist(),
            "empirical_sd": empirical_sd.tolist(),
            "exact_mean_reported_se_over_empirical_sd": (
                exact_se.mean(axis=0) / empirical_sd
            ).tolist(),
            "approximate_mean_reported_se_over_empirical_sd": (
                approximate_se.mean(axis=0) / empirical_sd
            ).tolist(),
            "exact_95pct_coverage": exact_coverage.tolist(),
            "approximate_95pct_coverage": approximate_coverage.tolist(),
            "approximate_over_exact_se_ratio_median": np.median(
                se_ratio, axis=0
            ).tolist(),
            "approximate_over_exact_se_ratio_p05": np.quantile(
                se_ratio, 0.05, axis=0
            ).tolist(),
            "approximate_over_exact_se_ratio_p95": np.quantile(
                se_ratio, 0.95, axis=0
            ).tolist(),
        }
        if coefficients[1] == 0.0:
            result["gxe_null_two_sided_type_i_error"] = float(
                1.0 - approximate_coverage[1]
            )
        return result

    alternative = scenario(
        "moderate_gxe",
        np.asarray([0.25, 0.12]),
        np.asarray([0.08, 0.05]),
        33,
    )
    null = scenario(
        "gxe_null",
        np.asarray([0.25, 0.0]),
        np.asarray([0.08, 0.05]),
        34,
    )
    null_type_i = float(null["gxe_null_two_sided_type_i_error"])
    alternative_coverage = np.asarray(
        alternative["approximate_95pct_coverage"]
    )
    alternative_se_ratio = np.asarray(
        alternative["approximate_over_exact_se_ratio_median"]
    )
    passes = bool(
        0.03 <= null_type_i <= 0.07
        and np.all((alternative_coverage >= 0.93) & (alternative_coverage <= 0.99))
        and np.all(alternative_se_ratio <= 1.25)
    )
    return {
        "samples": samples,
        "variants": variants,
        "blocks": blocks,
        "annotations": 1,
        "components": ["G:all", "GxE:all"],
        "repetitions": calibration_repetitions,
        "median_relative_normal_matrix_error": float(np.median(matrix_errors)),
        "maximum_relative_normal_matrix_error": float(np.max(matrix_errors)),
        "moderate_gxe": alternative,
        "gxe_null": null,
        "limited_calibration_gate": {
            "pass": passes,
            "criteria": (
                "GxE-null type-I error in [0.03,0.07], moderate-GxE coverage "
                "in [0.93,0.99], and median approximate/exact SE ratio <=1.25"
            ),
        },
    }


def run(seed: int = 20260815, repetitions: int = 400) -> dict:
    return {
        "schema": "summit-gxe-uncertainty-diagnostics-v2",
        "seed": seed,
        "probe_estimators": _probe_comparison(seed, repetitions),
        "delete_block_same_person_diagonal": _delete_block_comparison(seed),
        "block_local_jackknife_calibration": _jackknife_calibration(
            seed, repetitions
        ),
        "scope": (
            "Finite-probe diagnostics, exact-versus-full-diagonal deletion, and "
            "conditional block-estimating-equation calibration for K=1; not "
            "reference-person, probe-error, or study/reference-overlap uncertainty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--repetitions", type=int, default=400)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed, args.repetitions)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
