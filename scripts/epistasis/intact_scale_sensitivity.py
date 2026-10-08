"""A realized-phenotype transformation with an explicit observed-scale truth.

This quadratic transformation is a diagnostic scale choice, not a recommended
trait normalization. It transforms both the mean and the realized error.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.epistasis.intact_validation import (
    intact_panel,
    coordinates,
    projection_truth,
    reduce_records,
)
from summit.epistasis.robust import prepare_robust_scores


def scale_means(x, e):
    mu = 0.7 * x + 0.7 * e
    # Y=mu+N(0,1), g(Y)=Y+.3Y^2; E[g(Y)|G]=mu+.3(mu^2+1).
    return mu, mu + 0.3 * (mu**2 + 1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--genotypes", required=True)
    p.add_argument("--covariates", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=285791)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    raw, cov, axis, panel = intact_panel(
        a.genotypes, a.covariates, "12:66358347", "5", 73451
    )
    x, d, _, _ = coordinates(raw)
    e = x[:, 16:].sum(1)
    e = (e - e.mean()) / e.std()
    ids = np.random.default_rng(73452).choice(len(x), 6144, replace=False)[2048:]
    target = x[ids, 0]
    e = e[ids]
    c = np.column_stack(
        [np.ones(len(ids)), cov[ids], x[ids, :16], d[ids, :16], e, e**2]
    )
    f = (target * e)[:, None]
    mu, transformed_mean = scale_means(target, e)
    truths = (0.0, 2 * 0.3 * 0.7 * 0.7)
    # Independent diagnostic only: never passed to the production summaries.
    for mean, truth in zip((mu, transformed_mean), truths):
        np.testing.assert_allclose(projection_truth(f, c, mean), truth, atol=1e-11)
    rng = np.random.default_rng(a.seed)
    rows = []
    for start_draw in range(0, a.draws, 128):
        batch = min(128, a.draws - start_draw)
        y = mu[:, None] + rng.normal(size=(len(ids), batch))
        for name, outcomes, truth in zip(
            ("original", "quadratic"), (y, y + 0.3 * y**2), truths
        ):
            summary = prepare_robust_scores(
                f,
                outcomes,
                c,
                feature_names=("direction",),
                trait_names=tuple(map(str, range(batch))),
                metadata={},
            )
            b = summary.scores[0] / summary.information[0, 0]
            se = np.sqrt(summary.score_covariance[:, 0, 0]) / summary.information[0, 0]
            for j in range(batch):
                rows.append(
                    dict(
                        sampling="fixed",
                        setting=name,
                        adjustment="finite",
                        method="burden",
                        replicate=start_draw + j,
                        failed=False,
                        outside_scope=";".join(
                            summary.metadata["outside_confirmation_design"]
                        ),
                        estimate=float(b[j]),
                        se=float(se[j]),
                        truth=truth,
                        p=float(2 * norm.sf(abs(b[j] / se[j]))),
                        coverage=float(abs(b[j] - truth) <= norm.isf(0.025) * se[j]),
                        alignment_squared=1.0,
                    )
                )
    reduce_records(rows, a.out)
    (a.out / "design.json").write_text(
        json.dumps(
            dict(
                panel=panel,
                seed=a.seed,
                draws=a.draws,
                n=len(ids),
                direction="prespecified standardized distal burden",
                original="Y=.7x+.7e+Gaussian(0,1)",
                transformation="g(Y)=Y+.3Y^2",
                transformed_conditional_mean="mu+.3(mu^2+1)",
                nuisance="aligned covariates, 16 local additive/dominance terms, e and e^2",
                truths=dict(original=truths[0], quadratic=truths[1]),
                interpretation="quadratic scale has a nonzero mean interaction, not a biological-null calibration test",
                genotype_passes_per_draw=0,
                seconds=time.perf_counter() - start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            ),
            indent=2,
        )
    )
    print(pd.read_csv(a.out / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
