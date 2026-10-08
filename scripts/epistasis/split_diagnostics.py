"""Fixed-design nuisance decomposition with known population projection.

Independent diploid loci have an exactly zero additive projection of a centered
cross-product. Generating means/variances are diagnostic inputs, never a fit mode.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import norm
from summit.epistasis.split_reference import ridge_transfer, split_scalar_reference
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.cli import _jsonable


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=307159)
    p.add_argument("--draws", type=int, default=4000)
    a = p.parse_args()
    start = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    n0, n1, m = 384, 640, 256
    af = rng.uniform(0.15, 0.45, m)
    raw = rng.binomial(2, af, (n0 + n1, m)).astype(float)
    x = (raw - 2 * af) / np.sqrt(2 * af * (1 - af))
    # Native training convention: empirical scale, not HWE scaling for ridge.
    z = (raw - raw[:n0].mean(0)) / raw[:n0].std(0)
    c = np.column_stack([np.ones(len(x)), x[:, :4], raw[:, :4] == 1])
    f = x[:, 0] * x[:, 4]
    effects = rng.normal(size=m) / np.sqrt(m)
    mean = x @ effects + 0.6 * (raw[:, 0] == 1)
    var = 0.4 + 0.6 * x[:, 0] ** 2
    epsilon = np.sqrt(var[:, None]) * rng.normal(size=(len(x), a.draws))
    zero = np.zeros((n1, n0))
    records = []
    for outcome_multiplier, feature_multiplier in [
        (1.0, 1.0),
        (0.1, 0.1),
        (10.0, 10.0),
        (0.1, 10.0),
        (10.0, 0.1),
    ]:
        ly = ridge_transfer(z[:n0], z[n0:], c[:n0], c[n0:], m * outcome_multiplier)
        lf = ridge_transfer(z[:n0], z[n0:], c[:n0], c[n0:], m * feature_multiplier)
        cases = [("both_estimated", ly, lf, False)]
        if outcome_multiplier == feature_multiplier == 1:
            cases += [
                ("oracle_mean_reference_feature", zero, zero, True),
                ("estimated_mean_reference_feature", ly, zero, False),
                ("oracle_mean_estimated_feature", zero, lf, True),
            ]
        for name, L, F, oracle in cases:
            means = np.zeros_like(mean) if oracle else mean
            ref = split_scalar_reference(
                f[:n0],
                f[n0:],
                c[n0:],
                L,
                F,
                train_mean=means[:n0],
                test_mean=means[n0:],
                train_variance=var[:n0],
                test_variance=var[n0:],
            )
            feature = f[n0:] - F @ f[:n0]
            outcomes = (
                means[n0:, None] + epsilon[n0:] - L @ (means[:n0, None] + epsilon[:n0])
            )
            summary = prepare_robust_scores(
                feature[:, None],
                outcomes,
                c[n0:],
                feature_names=("f",),
                trait_names=tuple(map(str, range(a.draws))),
                metadata={},
            )
            beta = summary.scores[0] / summary.information[0, 0]
            se = np.sqrt(summary.score_covariance[:, 0, 0]) / summary.information[0, 0]
            estimate = {k: v for k, v in ref.items() if not k.endswith("contrast")}
            records.append(
                dict(
                    case=name,
                    outcome_penalty=m * outcome_multiplier,
                    feature_penalty=m * feature_multiplier,
                    **estimate,
                    empirical_mean=float(beta.mean()),
                    empirical_sd=float(beta.std(ddof=1)),
                    mean_hc3_se=float(se.mean()),
                    hc3_rejection=float(np.mean(abs(beta / se) > norm.isf(0.025))),
                    exact_centered_gaussian_rejection=float(
                        np.mean(
                            abs(
                                (beta - ref["expectation"])
                                / np.sqrt(ref["total_noise_variance"])
                            )
                            > norm.isf(0.025)
                        )
                    ),
                    training_variance_fraction=ref["training_noise_variance"]
                    / ref["total_noise_variance"],
                    bias_in_noise_sd=ref["expectation"]
                    / np.sqrt(ref["total_noise_variance"]),
                )
            )
    report = dict(
        seed=a.seed,
        draws=a.draws,
        n_train=n0,
        n_test=n1,
        markers=m,
        records=records,
        fixed="independent discrete genotype design, effects, penalties, features, independent population projection m=0",
        regenerated="training AND confirmation heteroskedastic Gaussian errors",
        limitation="fixed-tuning conditional diagnostic; no outcome-selected tuning; not a population calibration experiment",
        seconds=time.perf_counter() - start,
    )
    with a.out.open("x") as h:
        json.dump(_jsonable(report), h, indent=2)


if __name__ == "__main__":
    main()
