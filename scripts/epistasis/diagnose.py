"""Reproduce saved failures and separate Gaussian shape from covariance misspecification."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.epistasis.validate import SETTINGS, run_setting
from summit.prediction.spec import VariantAxis
from summit.epistasis.summary import fit_epistasis
from summit.epistasis.quadratic import moment_spectrum, quadratic_quantile, quadratic_sf
from summit.epistasis.cli import _jsonable


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = pd.read_csv("benchmarks/epistasis/panel_20261001/replicates.csv")
    n, m = 384, 96
    axis = VariantAxis(tuple(f"v{i}" for i in range(m)), ("1",)*m, tuple(range(1,m+1)), ("A",)*m, ("G",)*m)
    records = []
    for name in ("additive_null_p30", "few_strong", "dominance", "heteroskedastic"):
        rng = np.random.default_rng(7300+SETTINGS.index(name))
        haplotypes = []
        for _ in range(2):
            h = (rng.random((n,m)) < .3).astype(float)
            for j in range(1,8):
                copy = rng.random(n) < .6
                h[copy,j] = h[copy,j-1]
            haplotypes.append(h)
        experiment = run_setting(name, sum(haplotypes), axis, rng, 100, "numpy", return_experiment=True)
        s, k, v = (experiment[key] for key in ("summary","kernels","covariance"))
        fits = [fit_epistasis(s, i) for i in range(100)]
        estimates = np.array([f["coefficients"][1] for f in fits])
        previous = old.loc[old.setting == name, "coefficient"].to_numpy()
        np.testing.assert_allclose(estimates, previous, rtol=1e-10, atol=1e-10)
        spectrum = moment_spectrum(k, s.matrix, v, 1)
        a = spectrum["contrast"]
        inverse = np.linalg.inv(s.matrix)
        contrasts = np.einsum("ab,bij->aij", inverse, k)
        pseudo = experiment["expected"]
        # Exact Isserlis expectation of the FAME plug-in variance, including
        # reuse of y in both theta_hat and phenotype-dependent cubic moments.
        base, coupling = 0., 0.
        for c in range(len(k)):
            b = a@k[c]@a
            base += 2*pseudo[c]*np.trace(b@v)
            coupling += 4*np.trace((contrasts[c]@v)@(b@v))
        quantiles = {str(prob): quadratic_quantile(prob, spectrum["eigenvalues"]) for prob in (.025,.5,.95,.975)}
        null_tail = quadratic_sf(1.6448536269514722*np.sqrt(spectrum["variance"])+spectrum["mean"], spectrum["eigenvalues"])
        record = dict(setting=name, replicates=100, original_coefficient_max_difference=float(max(abs(estimates-previous))),
            biological_epistasis=name=="few_strong", pseudo_true_coefficient=float(pseudo[1]),
            empirical_mean=float(estimates.mean()), empirical_sd=float(estimates.std(ddof=1)),
            exact_gaussian_sd=float(np.sqrt(spectrum["variance"])), quadratic_skewness=spectrum["skewness"],
            mean_fame_variance=float(np.mean([f["covariance"][1,1] for f in fits])),
            expected_plugin_variance=float(base+coupling), covariance_projection_variance=float(base),
            same_phenotype_nuisance_coupling=float(coupling), exact_quantiles=quantiles,
            true_sd_normal_cutoff_actual_upper_tail=null_tail,
            expected_component_variance_contributions=experiment["expected_contributions"],
            realized_component_variance_mean=[float(x.mean()) for x in experiment["realized_contributions"]],
            reference="exact; no randomized reference error", fixed="genotypes, causal variant identities, fixed-effect design",
            regenerated="Gaussian effect sizes and residuals; inference over random effects, not conditional on realized effects")
        records.append(record)
        print(json.dumps(_jsonable(record)), flush=True)
    (args.out/"diagnosis.json").write_text(json.dumps(_jsonable(records),indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
