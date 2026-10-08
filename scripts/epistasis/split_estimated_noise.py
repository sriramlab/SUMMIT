"""Estimated-noise split/fold study; fixed operators, finite nuisance span.

Both successful restrictions and failure of ridge shrinkage bias are retained.
No known mean/variance is passed to the estimated-noise calculation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm, t
from summit.epistasis.split_reference import (
    ridge_transfer,
    fixed_split_contrast,
    crossfit_scalar_reference,
    estimated_contrast_noise,
)
from summit.epistasis.cli import _jsonable


def run(out, seed, draws):
    rng = np.random.default_rng(seed)
    n0, n1, m = 300, 500, 60
    g = rng.binomial(2, 0.35, (n0 + n1, m)).astype(float)
    # Cross-locus dependence induces a nonzero feature projection and makes
    # differing residualization operators' normalization consequential.
    g[:, 2] = np.where(rng.random(n0 + n1) < 0.8, g[:, 0], g[:, 2])
    z = (g - g[:n0].mean(0)) / g[:n0].std(0)
    c = np.column_stack([np.ones(len(g)), z[:, :2], g[:, :2] == 1])
    f = z[:, 0] * z[:, 2]
    nuisance = np.column_stack([c, z])
    mean = nuisance @ rng.normal(size=nuisance.shape[1]) * 0.2
    design = block_diag(
        np.column_stack([nuisance[:n0], f[:n0]]),
        np.column_stack([nuisance[n0:], f[n0:]]),
    )
    ols = nuisance[n0:] @ np.linalg.pinv(nuisance[:n0], rcond=1e-11)
    weak = ridge_transfer(z[:n0], z[n0:], c[:n0], c[n0:], m * 0.1)
    strong = ridge_transfer(z[:n0], z[n0:], c[:n0], c[n0:], m * 10)
    records = []
    for noise in ("gaussian_common", "gaussian_surface", "t5_surface"):
        variance = (
            np.ones(len(g)) if noise == "gaussian_common" else 0.4 + 0.6 * z[:, 0] ** 2
        )
        errors = (
            rng.normal(size=(len(g), draws))
            if noise != "t5_surface"
            else rng.standard_t(5, (len(g), draws)) / np.sqrt(5 / 3)
        )
        for effect in (0.0, 0.15):
            mu = mean + effect * f
            y = mu[:, None] + np.sqrt(variance[:, None]) * errors
            for label, ly, lf in (
                ("ridge_common", weak, weak),
                ("ridge_unequal", weak, strong),
                ("ols_restricted", ols, ols),
                ("ols_unequal_feature", ols, strong),
            ):
                ref = fixed_split_contrast(
                    f[:n0],
                    f[n0:],
                    c[n0:],
                    ly,
                    lf,
                )
                response = ref["interaction_response"]
                contrast = (
                    np.r_[-ref["training_contrast"], ref["test_contrast"]] / response
                )
                # Includes the estimated training contribution and analytic response.
                try:
                    fitted = estimated_contrast_noise(
                        contrast,
                        y,
                        design,
                        None
                        if noise == "gaussian_common"
                        else np.column_stack([np.ones(len(g)), z[:, 0] ** 2]),
                        nonnegative_components=True,
                    )
                except ValueError as error:
                    raise ValueError(f"{noise}/{effect}/{label}: {error}") from error
                estimates = contrast @ y
                positive = np.broadcast_to(fitted["positive_variance"], draws)
                se = np.sqrt(np.where(positive, fitted["covariance"][:, 0, 0], np.nan))
                threshold = (
                    t.isf(0.025, fitted["residual_df"])
                    if noise == "gaussian_common"
                    else norm.isf(0.025)
                )
                expectation = contrast @ mu
                records.append(
                    dict(
                        noise=noise,
                        effect=effect,
                        method=label,
                        response=response,
                        bias=float(np.mean(estimates - effect)),
                        empirical_sd=float(estimates.std(ddof=1)),
                        mean_se=float(np.nanmean(se)),
                        known_noise_sd=float(np.sqrt((contrast**2) @ variance)),
                        coverage=float(
                            np.mean(abs(estimates - effect) <= threshold * se)
                        ),
                        centered_coverage=float(
                            np.mean(abs(estimates - expectation) <= threshold * se)
                        ),
                        zero_rejection=float(np.mean(abs(estimates) > threshold * se)),
                        valid_rejection=float(
                            np.mean(abs(estimates[positive]) > threshold * se[positive])
                        ),
                        nonpositive_variance=int((~positive).sum()),
                        scheduled=draws,
                        valid=int(positive.sum()),
                        mean_bias_diagnostic=float(expectation - effect),
                    )
                )
        # Shared-participant cross-fold influence and fold-specific mean residuals.
        folds = np.array_split(np.arange(len(g)), 3)
        transfers = [
            nuisance[ids]
            @ np.linalg.pinv(
                nuisance[np.setdiff1d(np.arange(len(g)), ids)], rcond=1e-11
            )
            for ids in folds
        ]
        cross = crossfit_scalar_reference(f, c, folds, transfers, variance)
        fold_design = block_diag(
            *[np.column_stack([nuisance[ids], f[ids]]) for ids in folds]
        )
        y = mean[:, None] + np.sqrt(variance[:, None]) * errors
        fit = estimated_contrast_noise(
            cross["influence"],
            y,
            fold_design,
            None
            if noise == "gaussian_common"
            else np.column_stack([np.ones(len(g)), z[:, 0] ** 2]),
            nonnegative_components=True,
        )
        positive = np.broadcast_to(fit["positive_variance"], draws)
        se = np.sqrt(np.where(positive, fit["covariance"][:, 0, 0], np.nan))
        estimates = cross["influence"] @ y
        threshold = (
            t.isf(0.025, fit["residual_df"])
            if noise == "gaussian_common"
            else norm.isf(0.025)
        )
        records.append(
            dict(
                noise=noise,
                effect=0.0,
                method="ols_crossfit",
                response=cross["interaction_response"],
                bias=float(estimates.mean()),
                empirical_sd=float(estimates.std(ddof=1)),
                mean_se=float(se.mean()),
                known_noise_sd=float(np.sqrt(cross["variance"])),
                invalid_independent_fold_sd=float(
                    np.sqrt(cross["variance_if_folds_incorrectly_independent"])
                ),
                coverage=float(np.mean(abs(estimates) <= threshold * se)),
                zero_rejection=float(np.mean(abs(estimates) > threshold * se)),
                valid_rejection=float(
                    np.mean(abs(estimates[positive]) > threshold * se[positive])
                ),
                nonpositive_variance=int((~positive).sum()),
                scheduled=draws,
                valid=int(positive.sum()),
            )
        )
    with out.open("x") as handle:
        json.dump(
            _jsonable(
                dict(
                    seed=seed,
                    draws=draws,
                    n_train=n0,
                    n_test=n1,
                    markers=m,
                    records=records,
                    restriction="fixed feature and operators; finite full-rank additive mean below sample size; unknown common noise or prespecified nonnegative two-component variance surface; constrained moments are not claimed unbiased",
                    failure="estimated noise does not remove ridge mean bias; unrestricted p>N nuisance and outcome-adaptive operators remain unresolved",
                )
            ),
            handle,
            indent=2,
        )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=881973)
    p.add_argument("--draws", type=int, default=4000)
    args = p.parse_args()
    run(args.out, args.seed, args.draws)


if __name__ == "__main__":
    main()
