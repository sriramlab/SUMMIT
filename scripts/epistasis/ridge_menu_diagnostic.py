"""Training-only ridge tuning and correctly aggregated cross-fit diagnostics.

Bounded fixed-genotype conditional stress, not population qualification. Every
outcome-selected decision is repeated for each regenerated phenotype.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from summit.epistasis.split_reference import ridge_transfer, crossfit_scalar_reference
from summit.epistasis.robust import prepare_robust_scores
from summit.epistasis.cli import _jsonable


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(916387)
    n0, n1, m, b = 384, 640, 256, 4000
    af = rng.uniform(0.15, 0.45, m)
    g = rng.binomial(2, af, (n0 + n1, m)).astype(float)
    z = (g - g[:n0].mean(0)) / g[:n0].std(0)
    x = (g - 2 * af) / np.sqrt(2 * af * (1 - af))
    f = x[:, 0] * x[:, 4]
    c = np.column_stack([np.ones(n0 + n1), x[:, :4], g[:, :4] == 1])
    mean = x @ (rng.normal(size=m) / np.sqrt(m)) + 0.6 * (g[:, 0] == 1)
    variance = 0.4 + 0.6 * x[:, 0] ** 2
    y = mean[:, None] + np.sqrt(variance[:, None]) * rng.normal(size=(n0 + n1, b))
    menu = np.array([0.1, 1.0, 10.0]) * m
    operators = []
    losses = []
    feature_losses = []
    split = 256
    for penalty in menu:
        val = ridge_transfer(z[:split], z[split:n0], c[:split], c[split:n0], penalty)
        losses.append(np.mean((y[split:n0] - val @ y[:split]) ** 2, axis=0))
        feature_losses.append(np.mean((f[split:n0] - val @ f[:split]) ** 2))
        operators.append(ridge_transfer(z[:n0], z[n0:], c[:n0], c[n0:], penalty))
    choice = np.argmin(losses, axis=0)
    feature_choice = int(np.argmin(feature_losses))
    results = []
    for name in (
        "fixed_common_penalty",
        "training_selected_common_penalty",
        "separate_feature_penalty",
        "frozen_prediction_as_covariate",
    ):
        estimates = np.zeros(b)
        errors = np.zeros(b)
        for j, L in enumerate(operators):
            take = (
                np.arange(b)
                if name == "fixed_common_penalty" and j == 1
                else (
                    np.flatnonzero(choice == j)
                    if name != "fixed_common_penalty"
                    else np.array([], int)
                )
            )
            if not len(take):
                continue
            predictions = L @ y[:n0, take]
            if name == "frozen_prediction_as_covariate":
                for k, column in enumerate(take):
                    summary = prepare_robust_scores(
                        f[n0:, None],
                        y[n0:, column],
                        np.column_stack([c[n0:], predictions[:, k]]),
                        feature_names=("f",),
                        trait_names=("y",),
                        metadata={},
                    )
                    estimates[column] = summary.scores[0, 0] / summary.information[0, 0]
                    errors[column] = (
                        np.sqrt(summary.score_covariance[0, 0, 0])
                        / summary.information[0, 0]
                    )
            else:
                F = (
                    operators[feature_choice]
                    if name == "separate_feature_penalty"
                    else L
                )
                residual_f = f[n0:] - F @ f[:n0]
                summary = prepare_robust_scores(
                    residual_f[:, None],
                    y[n0:, take] - predictions,
                    c[n0:],
                    feature_names=("f",),
                    trait_names=tuple(map(str, take)),
                    metadata={},
                )
                estimates[take] = summary.scores[0] / summary.information[0, 0]
                errors[take] = (
                    np.sqrt(summary.score_covariance[:, 0, 0])
                    / summary.information[0, 0]
                )
        results.append(
            dict(
                method=name,
                mean=estimates.mean(),
                empirical_sd=estimates.std(ddof=1),
                mean_se=errors.mean(),
                rejection=float(np.mean(abs(estimates / errors) >= norm.isf(0.025))),
                fits=b,
            )
        )
    folds = np.array_split(np.arange(n0 + n1), 3)
    transfers = []
    for ids in folds:
        train = np.setdiff1d(np.arange(n0 + n1), ids)
        transfers.append(ridge_transfer(z[train], z[ids], c[train], c[ids], m))
    cross = crossfit_scalar_reference(f, c, folds, transfers, variance)
    cross["mean_bias"] = cross["influence"] @ mean
    del cross["influence"]
    with a.out.open("x") as h:
        json.dump(
            _jsonable(
                dict(
                    seed=916387,
                    results=results,
                    crossfit=cross,
                    penalties=menu,
                    outcome_choices=np.bincount(choice, minlength=3),
                    feature_choice=feature_choice,
                    feature_validation_losses=feature_losses,
                    fixed="genotypes, effects, feature, nuisance space, held-out training-validation split and tuning menu",
                    regenerated="all training/test phenotype errors; every outcome-tuning choice; genotype-only feature tuning fixed",
                    inference="adaptive operator has no exact fixed-operator Gaussian variance claim; cross-fit reference includes overlapping training influence",
                    scientific_target="conditional stress against zero generating biological interaction, not population linear-projection calibration",
                )
            ),
            h,
            indent=2,
        )


if __name__ == "__main__":
    main()
