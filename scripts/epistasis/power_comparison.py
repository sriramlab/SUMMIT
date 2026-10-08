"""Known-target burden/sparse/distributed comparison with known nuisance V."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summit.epistasis.oracle import selected_kernels
from summit.epistasis.pairs import prepare_pair_scores, pair_tests
from summit.epistasis.prepare import SelectedStudy, fit_scale
from summit.epistasis.summary import fit_epistasis
from summit.prediction.genotype import ArrayGenotypeSource
from summit.prediction.spec import VariantAxis
from scripts.epistasis.validate import interval


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    n, m, reps = 256, 64, 100
    rng = np.random.default_rng(5804)
    raw = rng.binomial(2, .3, (n, m)).astype(float)
    axis = VariantAxis(tuple(f"v{i}" for i in range(m)), ("1",)*m, tuple(range(1, m+1)), ("A",)*m, ("G",)*m)
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis)
    scale = fit_scale(source, np.arange(n))
    x = (raw-scale.mean)*scale.inverse_scale
    background = np.arange(8, 16)
    pairs = [("v0", f"v{i}") for i in background]
    fixed = np.column_stack([np.ones(n), x[:, 0], x[:, background]])
    e = np.column_stack([np.ones(n), x[:, 0]])
    weights = np.column_stack([np.ones(m), np.isin(np.arange(m), background)])
    kernels, projection = selected_kernels(x, e, weights, fixed)
    p = projection.projector
    f = p@(x[:, 0, None]*x[:, background])
    v = .3*x@x.T/m + .7*np.eye(n)
    # Same known additive covariance for the score comparisons in every setting.
    vi = np.linalg.inv(v)
    study = SelectedStudy(source, np.arange(n), scale, modifiers=e, weights=weights,
        fixed_effects=fixed, component_names=("additive", "epistasis"), definitions={"pairs": pairs}, backend="numpy")
    ref = study.reference(exact=True)
    records = []
    for setting in ("null", "mixed_sign_distributed", "one_strong_pair", "aligned_burden"):
        y = np.sqrt(.3/m)*x@rng.normal(size=(m, reps)) + np.sqrt(.7)*rng.normal(size=(n, reps))
        if setting == "mixed_sign_distributed":
            signal = f/np.sqrt(len(background))
        elif setting == "one_strong_pair":
            signal = f[:, :1]
        elif setting == "aligned_burden":
            signal = f.sum(axis=1, keepdims=True)/np.sqrt(len(background))
        else:
            signal = None
        if signal is not None:
            signal = signal*np.sqrt(.15*projection.residual_rank/np.sum(signal**2))
            y += signal@rng.normal(size=(signal.shape[1], reps))
        summary, _ = study.summarize(ref, y, trait_names=tuple(f"r{i}" for i in range(reps)))
        signed = prepare_pair_scores(x, axis, pairs, y, fixed_effects=fixed,
            covariance_solve=lambda z: vi@z, covariance_identity="oracle_known_additive_V",
            covariance_known=True, trait_names=summary.trait_names, sample_identity="synthetic_seed5804")
        for i in range(reps):
            fit = fit_epistasis(summary, i)
            tests = pair_tests(signed, trait=i, burden_weights=np.ones(len(pairs)))
            for method, value in (("FAME_plugin_Wald", fit["wald_p_one_sided"][1]),
                                  ("joint_score_known_V", tests["joint_p"]),
                                  ("burden_known_V", tests["burden_p"]),
                                  ("sparse_Bonferroni_known_V", tests["sparse_bonferroni_p"]),
                                  ("adaptive_Bonferroni_known_V", tests["adaptive_bonferroni_p"])):
                records.append(dict(setting=setting, replicate=i, method=method, p=value))
    frame = pd.DataFrame(records)
    frame.to_csv(args.out/"replicates.csv", index=False)
    table = []
    for (setting, method), group in frame.groupby(["setting", "method"], sort=False):
        hits = int((group.p <= .05).sum())
        lo, hi = interval(hits, len(group))
        table.append(dict(setting=setting, method=method, fits=len(group), failed=int(group.p.isna().sum()),
                          rejection_05=hits/len(group), lower=lo, upper=hi))
    pd.DataFrame(table).to_csv(args.out/"summary.csv", index=False)
    (args.out/"design.json").write_text(json.dumps(dict(seed=5804, n=n, m=m, replicates=reps,
        alpha=.05, pairs=8, known_target=True, expected_projected_signal_variance=.15,
        interpretation="Score comparisons know the nuisance V; FAME estimates nuisance variances. Not a discovery comparison."), indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    table = pd.DataFrame(table)
    settings = list(table.setting.unique())
    for j, method in enumerate(table.method.unique()):
        part = table[table.method == method]
        ax.errorbar(np.arange(4)+(j-2)*.12, part.rejection_05,
                    yerr=np.stack([part.rejection_05-part.lower, part.upper-part.rejection_05]),
                    fmt="o", label=method, ms=4)
    ax.axhline(.05, color="grey", lw=1)
    ax.set_xticks(np.arange(4), settings)
    ax.set_ylabel("Rejection / power at 0.05; binomial 95% CI")
    ax.legend(fontsize=8)
    fig.savefig(args.out/"power.png", dpi=180)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
