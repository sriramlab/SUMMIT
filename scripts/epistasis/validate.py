"""Prespecified bounded exploratory panel; no calibration tuning or multipliers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy.stats import beta

from summit.epistasis.cli import _jsonable
from summit.epistasis.oracle import selected_kernels
from summit.epistasis.prepare import SelectedStudy, fit_scale, block_jackknife
from summit.epistasis.summary import fit_epistasis
from summit.prediction.genotype import ArrayGenotypeSource, FileGenotypeSource
from summit.prediction.spec import VariantAxis

SETTINGS = (
    "additive_null_p15", "additive_null_p30", "additive_null_p50",
    "strong_local", "strong_local_adjusted", "imperfect_local_tag", "omitted_causal",
    "dominance", "dominance_adjusted", "heteroskedastic", "many_weak", "few_strong",
    "multiple_targets", "helpful_group", "uninformative_group",
    "structure_adjusted", "structure_misspecified",
)


def real_panel(path, rng, n, m):
    """Bounded reads from a verified local panel; no participant arrays persisted."""
    with FileGenotypeSource(path) as file:
        rows = np.sort(rng.choice(len(file.samples), n, replace=False))
        # A local window plus dispersed variants; read enough candidates to
        # retain common polymorphic sites on this small sample mask.
        selected = np.unique(np.r_[np.arange(256), np.linspace(256, len(file.variants.ids)-1, 256).astype(int)])
        file.prepare(rows, 512, 1)
        raw = file.read(selected).astype(float)
        raw[raw == -127] = np.nan
        af = np.nanmean(raw, axis=0)/2
        keep = np.flatnonzero((af > .15) & (af < .85))
        if len(keep) < m:
            raise ValueError("bounded real panel has too few common variants")
        chosen = keep[np.linspace(0, len(keep)-1, m).astype(int)]
        return raw[:, chosen], file.variants.subset(selected[chosen]), dict(
            source=str(Path(path).resolve()), source_identity=file.identity,
            selected_variant_indices=selected[chosen].tolist(), sample_draw_seed=912,
            sample_count=n, variant_count=m, phenotype="simulated",
        )


def run_setting(name, raw, axis, rng, replicates, backend, *, residual_extension=False,
                return_experiment=False):
    started = time.perf_counter()
    n, m = raw.shape
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis)
    scale = fit_scale(source, np.arange(n), block_size=32)
    x = (raw-scale.mean)*scale.inverse_scale
    x[np.isnan(x)] = 0
    fixed = [np.ones(n), x[:, 0]]
    if name == "strong_local_adjusted":
        fixed.append(x[:, 1])
    if name == "imperfect_local_tag":
        fixed.append(x[:, 2])
    if name.endswith("dominance_adjusted"):
        fixed.append((raw[:, 0] == 1).astype(float))
    if name.startswith("structure"):
        structure = np.r_[np.full(n//2, -1.), np.ones(n-n//2)]
        if name == "structure_adjusted":
            fixed.append(structure)
    e = np.column_stack([np.ones(n), x[:, 0]])
    weights = np.ones((m, 2))
    weights[0, 1] = 0
    names = ["additive", "epistasis"]
    if name == "omitted_causal":
        weights[1] = 0
    if name == "multiple_targets":
        e = np.column_stack([e, x[:, 8]])
        weights = np.column_stack([weights, np.ones(m)])
        weights[8, 2] = 0
        fixed.append(x[:, 8])
        names.append("epistasis_second")
    if name in ("helpful_group", "uninformative_group"):
        weights[:, 1] = 0
        weights[12:20 if name == "helpful_group" else 12, 1] = 1
        if name == "uninformative_group":
            weights[30:38, 1] = 1
    fixed = np.column_stack(fixed)
    kernels, projection = selected_kernels(x, e, weights, fixed)
    p = projection.projector
    factors = [np.sqrt(.3/m)*x]
    signal = False
    if name in ("strong_local", "strong_local_adjusted", "imperfect_local_tag", "omitted_causal"):
        factors.append(np.sqrt(.7)*x[:, 1, None])
    if "dominance" in name:
        h = (raw[:, 0] == 1).astype(float)
        h -= h.mean()
        h /= np.std(h)
        factors.append(np.sqrt(.5)*h[:, None])
    if name in ("many_weak", "real_many_weak", "multiple_targets"):
        factors.append(np.sqrt(.15/(m-1))*x[:, 0, None]*x[:, 1:])
        signal = True
    if name == "multiple_targets":
        factors.append(np.sqrt(.15/(m-1))*x[:, 8, None]*x[:, np.arange(m) != 8])
    if name == "few_strong":
        factors.append(np.sqrt(.15)*x[:, 0, None]*x[:, 12, None])
        signal = True
    if name in ("helpful_group", "uninformative_group"):
        factors.append(np.sqrt(.15/8)*x[:, 0, None]*x[:, 12:20])
        signal = True
    if name.startswith("structure"):
        factors.append(np.sqrt(.5)*structure[:, None])
    noise_var = np.full(n, .7)
    if name == "heteroskedastic":
        noise_var = .7*(.3 + .7*x[:, 0]**2)
    v = np.diag(noise_var)
    y = np.sqrt(noise_var)[:, None]*rng.normal(size=(n, replicates))
    realized_contributions = []
    for f in factors:
        effect = f @ rng.normal(size=(f.shape[1], replicates))
        y += effect
        if return_experiment:
            realized_contributions.append(np.sum((p@effect)**2, axis=0)/projection.residual_rank)
        v += f @ f.T
    v = p @ v @ p
    residual_options = (dict(residual_basis=np.column_stack([e[:, 1]**2, np.ones(n)]),
                             residual_names=("residual_target_square", "residual"))
                        if residual_extension else {})
    study = SelectedStudy(source, np.arange(n), scale, fixed_effects=fixed,
        modifiers=e, weights=weights, component_names=names, definitions={"setting": name},
        block_size=32, backend=backend, memory_bytes=2**30, **residual_options)
    if residual_extension:
        kernels = np.concatenate([kernels[:-1], ((p*e[:, 1]**2)@p)[None], kernels[-1:]])
    ref_start = time.perf_counter()
    reference = study.reference(exact=True)
    ref_seconds = time.perf_counter()-ref_start
    summary, qrows = study.summarize(reference, y, trait_names=tuple(f"rep{i}" for i in range(replicates)),
                                   retain_variant_rows=True)
    expected_q = np.einsum("aij,ji->a", kernels, v)
    expected = np.linalg.solve(reference.matrix, expected_q)
    if return_experiment:
        return dict(study=study, reference=reference, summary=summary, kernels=kernels,
                    covariance=v, phenotype=y, genotype=x, fixed_effects=fixed,
                    expected=expected, realized_contributions=realized_contributions,
                    expected_contributions=[float(np.sum((p@f)**2)/projection.residual_rank) for f in factors])
    records = []
    for i in range(replicates):
        record = dict(setting=name, replicate=i, signal=signal, expected_moment_coefficient=float(expected[1]),
                      coefficient=np.nan, se=np.nan, jackknife_se=np.nan, p=np.nan, covered=False,
                      valid=False, failure=None)
        try:
            fit = fit_epistasis(summary, i)
            jk = block_jackknife(reference, summary, qrows, np.arange(m)//12, trait=i)
            record.update(coefficient=float(fit["coefficients"][1]), se=float(fit["standard_errors"][1]),
                          jackknife_se=float(jk["standard_errors"][1]), p=float(fit["wald_p_one_sided"][1]),
                          covered=bool(abs(fit["coefficients"][1]-expected[1]) <= 1.959963984540054*fit["standard_errors"][1]),
                          valid=bool(fit["covariance_valid"]),
                          failure=None if fit["covariance_valid"] else "invalid_plugin_covariance")
        except (ValueError, np.linalg.LinAlgError) as exc:
            record["failure"] = str(exc)
        records.append(record)
    telemetry = dict(setting=name, n=n, m=m, replicates=replicates,
                     elapsed_seconds=time.perf_counter()-started, reference_seconds=ref_seconds,
                     peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     genotype_passes=study.stream.ledger.traversals,
                     target_counts=[int(np.sum(raw[:, 0] == i)) for i in (0, 1, 2)],
                     max_abs_projected_feature_correlation=float(np.max(np.abs(np.corrcoef(x.T)-np.eye(m)))))
    return records, telemetry


def interval(successes, n):
    return (0. if successes == 0 else float(beta.ppf(.025, successes, n-successes+1)),
            1. if successes == n else float(beta.ppf(.975, successes+1, n-successes)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--real-genotypes", type=Path)
    parser.add_argument("--backend", choices=("numpy", "native"), default="numpy")
    args = parser.parse_args()
    if not 2 <= args.replicates <= 100:
        raise ValueError("initial validation uses 2 to 100 replicates per setting")
    args.out.mkdir(parents=True, exist_ok=False)
    n, m = 384, 96
    axis = VariantAxis(tuple(f"v{i}" for i in range(m)), ("1",)*m, tuple(range(1, m+1)), ("A",)*m, ("G",)*m)
    records, times = [], []
    for setting_index, name in enumerate(SETTINGS):
        rng = np.random.default_rng(7300+setting_index)
        af = np.full((n, m), .3)
        if name.endswith("p15"):
            af[:, 0] = .15
        if name.endswith("p50"):
            af[:, 0] = .5
        if name.startswith("structure"):
            af[:n//2] -= .08
            af[n//2:] += .08
        # Two correlated haplotypes retain discrete genotypes and local LD.
        haplotypes = []
        for _ in range(2):
            h = (rng.random((n, m)) < af).astype(float)
            for j in range(1, 8):
                copy = rng.random(n) < .6
                h[copy, j] = h[copy, j-1]
            haplotypes.append(h)
        raw = sum(haplotypes)
        result, timing = run_setting(name, raw, axis, rng, args.replicates, args.backend)
        records.extend(result)
        times.append(timing)
        print(json.dumps(timing), flush=True)
    real_metadata = None
    if args.real_genotypes:
        rng = np.random.default_rng(912)
        raw, axis, real_metadata = real_panel(args.real_genotypes, rng, n, m)
        for name in ("real_additive_null", "real_many_weak", "real_dominance"):
            result, timing = run_setting(name, raw, axis, rng, args.replicates, args.backend)
            records.extend(result)
            times.append(timing)
            print(json.dumps(timing), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.out/"replicates.csv", index=False)
    aggregate = []
    for name, group in frame.groupby("setting", sort=False):
        valid = group[group.valid]
        count = len(group)
        row = dict(setting=name, fits=count, valid=len(valid), failures=count-len(valid),
                   mean_estimate=group.coefficient.mean(), expected=group.expected_moment_coefficient.iloc[0],
                   bias=group.coefficient.mean()-group.expected_moment_coefficient.iloc[0],
                   empirical_sd=group.coefficient.std(ddof=1), mean_se=valid.se.mean(),
                   mean_jackknife_se=group.jackknife_se.mean(),
                   mean_mcse=group.coefficient.std(ddof=1)/np.sqrt(group.coefficient.notna().sum()),
                   coverage_all=float(group.covered.sum()/count), coverage_valid=valid.covered.mean())
        for alpha in (.05, .01):
            hits = int((group.p <= alpha).sum())
            lo, hi = interval(hits, count)
            row.update({f"rejection_{alpha}": hits/count, f"rejection_{alpha}_lo": lo,
                        f"rejection_{alpha}_hi": hi, f"rejection_{alpha}_valid": (valid.p <= alpha).mean()})
        aggregate.append(row)
    aggregate = pd.DataFrame(aggregate)
    aggregate.to_csv(args.out/"summary.csv", index=False)
    (args.out/"design.json").write_text(json.dumps(_jsonable(dict(
        panel="exploratory_prespecified_v1", seeds="7300 + setting index; real panel 912",
        settings=SETTINGS, replicates=args.replicates, n=n, m=m, backend=args.backend,
        real_panel=real_metadata, timing=times,
        interpretation="Conditional-on-genotypes exploratory validation; exact references; no genome-wide tail calibration; failures remain in denominators",
    )), indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True, layout="constrained")
    index = np.arange(len(aggregate))
    axes[0].errorbar(aggregate.bias, index, xerr=1.96*aggregate.mean_mcse, fmt="o", ms=4)
    axes[0].axvline(0, color="grey", lw=1)
    axes[0].set_xlabel("Bias vs conditional MoM expectation (95% MC interval)")
    axes[1].plot(aggregate.mean_se/aggregate.empirical_sd, index, "o", label="FAME analytic")
    axes[1].plot(aggregate.mean_jackknife_se/aggregate.empirical_sd, index, "x", label="SNP-block comparison")
    axes[1].axvline(1, color="grey", lw=1)
    axes[1].set_xlabel("Mean reported SE / empirical SD")
    axes[1].legend(loc="best")
    rate = aggregate["rejection_0.05"]
    axes[2].errorbar(rate, index, xerr=np.stack([rate-aggregate["rejection_0.05_lo"],
                                               aggregate["rejection_0.05_hi"]-rate]), fmt="o", ms=4)
    axes[2].axvline(.05, color="grey", lw=1)
    axes[2].set_xlabel("Rejection / power at 0.05 (exact binomial 95% CI)")
    axes[0].set_yticks(index, aggregate.setting)
    axes[0].invert_yaxis()
    fig.savefig(args.out/"validation.png", dpi=180)
    fig.savefig(args.out/"validation.pdf")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
