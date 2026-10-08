"""Independent-seed check of the derived residual-surface extension and pair tails."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from scripts.epistasis.validate import run_setting, interval
from summit.prediction.spec import VariantAxis


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    n, m, replicates = 384, 96, 100
    axis = VariantAxis(tuple(f"v{i}" for i in range(m)), ("1",)*m,
                       tuple(range(1, m+1)), ("A",)*m, ("G",)*m)
    rows, times = [], []
    for i, name in enumerate(("additive_null_p30", "heteroskedastic", "many_weak")):
        raw = np.random.default_rng(12831+i).binomial(2, .3, (n, m)).astype(float)
        for extension in (False, True):
            records, timing = run_setting(name, raw, axis, np.random.default_rng(43712+i),
                                           replicates, "native", residual_extension=extension)
            method = "modifier_square_residual" if extension else "fame_iid_residual"
            for record in records:
                record["method"] = method
            timing["method"] = method
            rows.extend(records)
            times.append(timing)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out/"replicates.csv", index=False)
    summaries = []
    for (setting, method), group in frame.groupby(["setting", "method"], sort=False):
        hits = int((group.p <= .05).sum())
        lo, hi = interval(hits, len(group))
        summaries.append(dict(setting=setting, method=method, count=len(group), valid=int(group.valid.sum()),
            mean_estimate=group.coefficient.mean(), expected=group.expected_moment_coefficient.iloc[0],
            empirical_sd=group.coefficient.std(), mean_se=group.se.mean(), rejection_05=hits/len(group),
            rejection_lo=lo, rejection_hi=hi, coverage_all=group.covered.mean()))
    table = pd.DataFrame(summaries)
    table.to_csv(args.out/"summary.csv", index=False)
    # For a single supplied pair under known Gaussian nuisance covariance,
    # s/sqrt(H) is exactly N(0,1); its square is chi-square(1), including tails.
    tails = [dict(alpha=alpha, chi_square_threshold=float(chi2.isf(alpha, 1)),
                  recovered_tail=float(2*norm.sf(np.sqrt(chi2.isf(alpha, 1)))))
             for alpha in (.05, .01, 5e-8)]
    (args.out/"design.json").write_text(json.dumps(dict(
        independent_genotype_seeds=[12831, 12832, 12833], independent_phenotype_seeds=[43712, 43713, 43714],
        panel="residual_extension_confirmation_v1", n=n, m=m, replicates=replicates,
        paired_same_phenotypes=True, alpha=.05, timings=times, exact_known_covariance_single_pair_tails=tails,
        limitation="Tail identity does not validate FAME plug-in Wald tails or fitted-covariance pair tests."), indent=2)+"\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
