"""Prospective qualification rule and its exact binomial operating characteristics."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import beta, binom


def decision(hits, draws):
    lo = 0 if hits == 0 else float(beta.ppf(0.05, hits, draws - hits + 1))
    hi = 1 if hits == draws else float(beta.ppf(0.95, hits + 1, draws - hits))
    if lo > 0.075:
        status = "clear material inflation"
    elif hi < 0.025:
        status = "clear conservatism"
    elif hi <= 0.075:
        status = "demonstrated control of size above .075"
    else:
        status = "insufficient precision"
    return dict(status=status, lower_one_sided_95=lo, upper_one_sided_95=hi)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    records = []
    for n in (100, 2000):
        labels = np.array([decision(k, n)["status"] for k in range(n + 1)])
        for size in (0.025, 0.05, 0.075, 0.10, 0.15):
            mass = binom.pmf(np.arange(n + 1), n, size)
            records.append(
                dict(
                    draws=n,
                    true_size=size,
                    probabilities={
                        name: float(mass[labels == name].sum())
                        for name in sorted(set(labels))
                    },
                )
            )
    spec = dict(
        base="e828c6743cf7154b7f81d93e8bcd63372688d676",
        procedure="common frozen imputation plus identifiable-span HC3 v2",
        alpha=0.05,
        rule="One-sided 95% exact upper bound <=.075 demonstrates exclusion of >50% relative inflation. Lower bound>.075 is material inflation. Upper bound<.025 is marked conservative. Otherwise inconclusive. No SE or alpha tuning.",
        operating_characteristics=records,
        full_confirmation=dict(
            genotype_seed=913742,
            phenotype_seed=526813,
            training_n=2048,
            test_n=4096,
            m=8192,
            replicates=100,
        ),
        nested=dict(
            genotype_seeds=[738291, 261849, 814763],
            phenotype_seed=682194,
            training_n=2048,
            test_n=4096,
            m=8192,
            draws=2000,
            fixed="genotypes, independently trained direction, true realized causal mean and effect sizes",
            regenerated="confirmation residuals; every mean and HC3 refitted; no new training",
        ),
        tails=dict(
            seed=712431,
            n=512,
            alphas=[0.05, 0.005, 0.0005, 5e-6, 5e-8],
            interpretation="scalar Gaussian fixed-design validators; corresponding 1,10,100,10000,1000000 supplied hypotheses at family .05; no genomewide production claim",
        ),
        failures="retained separately; never counted as successful conservative fits",
        independence="outer training/genotype configurations are independent validation units; nested draws are conditional residual experiments",
    )
    with a.out.open("x") as h:
        json.dump(spec, h, indent=2)


if __name__ == "__main__":
    main()
