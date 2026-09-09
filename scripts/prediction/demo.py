"""Reproducible small multi-trait fit and held-out scoring example."""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    from pathlib import Path
    import numpy as np
    from summit.prediction import (ArrayGenotypeSource, VariantAxis, TraitTraining, CandidatePrior,
        ResponseGeometry, SolverSpec, estimate_scale, plan_prediction, fit_prediction, ScoreInput, score_prediction, separate_scales)
    from summit.prediction.features import fit_contexts, evaluate_contexts, evaluate_fixed
    from summit.prediction.artifacts import write_json
    root = Path(args.out)
    root.mkdir(exist_ok=False)
    rng = np.random.default_rng(20260909)
    n, m = 96, 128
    exposure = rng.normal(size=(n, 2))
    calls = rng.binomial(2, .3, size=(n, m)).astype(float)
    phenotype = calls[:, 3]+.4*exposure[:, 0]*calls[:, 7]+rng.normal(size=n)
    calls[rng.random(calls.shape) < .02] = np.nan
    axis = VariantAxis(tuple(f"rs{i}" for i in range(m)), ("1",)*m,
        tuple(range(1, m+1)), ("A",)*m, ("G",)*m, "synthetic")
    source = ArrayGenotypeSource(calls, [("sim", str(i)) for i in range(n)], axis, hard_calls=True)
    traits, score_inputs = [], {}
    omega = np.array([[.3, .04, -.02], [.04, .08, .01], [-.02, .01, .06]])
    heldout = np.arange(72, n)
    for ti in range(2):
        rows = np.flatnonzero((np.arange(n) < 72) & (np.arange(n) % (7+ti) != ti))
        contexts = {"e1": exposure[rows, 0], "e2": exposure[rows, 1]}
        phi, context_spec, metric = fit_contexts(contexts, [dict(name=name, kind="continuous") for name in contexts])
        fixed_spec = dict(kind="summit.prediction.fixed", schema_version=1, terms=[dict(name="intercept", factors=[]),
            *[dict(name=name, factors=[dict(source="context", name=name, power=1)]) for name in contexts]])
        fixed = evaluate_fixed(fixed_spec, {}, phi, context_spec["names"])
        scale = estimate_scale(source, rows, np.arange(m), block_size=32)
        candidates = tuple(CandidatePrior(name, covariance, np.exp(.2*phi[:, 1]),
            {"prior": name, "source": "synthetic", "residual": "exp(0.2*e1)"}) for name, covariance in
            [("full", omega), ("two_scale", separate_scales(omega, .5, .25)), ("amplification", separate_scales(omega, .5, 0))])
        trait_id = f"trait{ti}"
        traits.append(TraitTraining(trait_id, rows, np.arange(m), phenotype[rows]+ti*.1, phi, fixed, scale,
            candidates, context_spec, fixed_spec, {"units": "synthetic phenotype"}, ResponseGeometry(omega, metric[1:, 1:], "discovery")))
        score_phi = evaluate_contexts(context_spec, {"e1": exposure[heldout, 0], "e2": exposure[heldout, 1]})
        score_fixed = evaluate_fixed(fixed_spec, {}, score_phi, context_spec["names"])
        score_inputs[trait_id] = ScoreInput(heldout, score_phi, score_fixed, context_spec, fixed_spec)
    plan = plan_prediction(traits, source, storage="compact", block_size=32, rhs_columns=6)
    models = fit_prediction(traits, source, output=root/"models", plan=plan, solver=SolverSpec(rtol=1e-10))
    result = score_prediction(models, source, score_inputs, block_size=32, rhs_columns=6)
    np.savetxt(root/"heldout_predictions.tsv", np.column_stack([result.prediction[m.key] for m in models]),
        delimiter="\t", header="\t".join("/".join(m.key) for m in models), comments="")
    summary = dict(seed=20260909, traits=2, models=len(models), variants=m, discovery_counts=[len(t.rows) for t in traits],
        heldout_samples=len(heldout), maximum_relative_true_residual=max(m.convergence["relative_true_residual"] for m in models),
        score_ledger=result.report["ledger"])
    write_json(root/"summary.json", summary)
    print(summary)
    return 0
