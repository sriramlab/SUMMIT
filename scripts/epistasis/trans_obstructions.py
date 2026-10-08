"""Separate biological nulls, phenotype-scale interactions and projection nulls."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from summit.epistasis.robust import prepare_robust_scores, robust_score_tests
from summit.epistasis.trans import stratified_fixed_effects


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(416957)
    records = []
    n = 4096
    for rep in range(2000):
        s = rng.integers(2, size=n)
        af = 0.2 + 0.1 * s
        target = np.zeros(n)
        hidden = np.zeros(n)
        other = np.zeros(n)
        for _ in range(2):
            causal = rng.binomial(1, af)
            hidden += causal
            target += np.where(
                rng.random(n) < (0.85 - 0.5 * s), causal, rng.binomial(1, af)
            )
            other += rng.binomial(1, 0.25 + 0.1 * s)
        y = (
            2 * hidden
            + 0.8 * (hidden == 1)
            + 1.5 * other
            + (0.5 + target) * rng.standard_t(5, size=n) / np.sqrt(5 / 3)
        )
        c = np.column_stack([np.ones(n), s, target, target == 1, other, other == 1])
        cs, _ = stratified_fixed_effects(c, s.astype(str), memory_bytes=2**30)
        for method, C in (
            ("intercepts_only", c),
            ("stratum_specific_main_effects", cs),
        ):
            summary = prepare_robust_scores(
                (target * other)[:, None],
                y,
                C,
                feature_names=("f",),
                trait_names=("y",),
                metadata={},
                sampling_model="iid_population_projection",
            )
            fit = robust_score_tests(summary)
            records.append(
                dict(
                    experiment="stratum_dependent_local_LD",
                    replicate=rep,
                    method=method,
                    beta=fit["beta"][0],
                    se=fit["standard_errors"][0],
                    p=fit["kernel_p"],
                    failed=False,
                    projection_null_justified=(
                        method == "stratum_specific_main_effects"
                    ),
                    biological_null=True,
                )
            )
    for experiment in ("higher_order_dependence", "nonlinear_phenotype_scale"):
        for rep in range(100):
            if experiment == "higher_order_dependence":
                hap = rng.integers(2, size=(n, 2, 2))
                g = hap.sum(1)
                hidden = np.logical_xor(hap[:, :, 0], hap[:, :, 1]).sum(1)
                x = g.astype(float) - 1
                y = hidden + rng.normal(size=n)
                truth = -1.0
                c = np.column_stack([np.ones(n), x, g == 1])
            else:
                g = rng.binomial(2, [0.3, 0.35], size=(n, 2))
                x = (g - 2 * np.array([0.3, 0.35])) / np.sqrt(
                    2 * np.array([0.3, 0.35]) * np.array([0.7, 0.65])
                )
                latent = x.sum(1)
                y = latent + 0.3 * latent**2 + rng.normal(size=n)
                truth = 0.6
                c = np.column_stack([np.ones(n), x, g == 1])
            summary = prepare_robust_scores(
                (x[:, 0] * x[:, 1])[:, None],
                y,
                c,
                feature_names=("f",),
                trait_names=("y",),
                metadata={},
                sampling_model="iid_population_projection",
            )
            fit = robust_score_tests(summary)
            records.append(
                dict(
                    experiment=experiment,
                    replicate=rep,
                    method="population_projection",
                    beta=fit["beta"][0],
                    se=fit["standard_errors"][0],
                    p=fit["kernel_p"],
                    truth=truth,
                    coverage=float(
                        abs(fit["beta"][0] - truth)
                        < 1.95996398454 * fit["standard_errors"][0]
                    ),
                    pairwise_r=float(np.corrcoef(x.T)[0, 1]),
                    failed=False,
                    projection_null_justified=False,
                    biological_null=True,
                )
            )
    pd.DataFrame(records).to_csv(a.out / "replicates.csv", index=False)
    (a.out / "design.json").write_text(
        json.dumps(
            dict(
                seed=416957,
                n=n,
                structure="LD between target and hidden additive cause varies by observed stratum; blocks independent conditional on stratum. Both designs include stratum intercepts; only one includes stratum-specific main slopes.",
                higher_order="Per haplotype, target and partner independent Bernoulli(.5), hidden additive causal allele their XOR. Two independent haplotypes. Pairwise LD is zero for every locus pair, but E[(Ga-1)(Gb-1)(Gc-1)]=-.25 and the population product coefficient is -1.",
                nonlinear="Latent additive x1+x2 transformed to latent+.3*latent^2 before independent Gaussian noise. On the analyzed scale the interaction coefficient is .6; a biological mechanism is not identified by rejecting its zero null.",
                random="IID genotype individuals and residuals regenerated; mechanisms/effects fixed; no training, probes or sketches",
                interpretation="Known nonzero projection coefficients are not false positives for the projection null. Their rejection cannot establish absence of phantom or phenotype-scale interaction.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
