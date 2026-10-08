"""Reduce frozen validation outputs and make publication-quality static figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def reduce(frame, groups):
    records = []
    for keys, part in frame.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = part.loc[~part.failed & np.isfinite(part.p)].copy()
        n = len(valid)
        result = dict(zip(groups, keys))
        result.update(scheduled=len(part), valid=n, failures=len(part) - n)
        if "outside_scope" in part:
            result["outside_scope"] = int(part.outside_scope.fillna("").ne("").sum())
        for alpha, label in [(0.05, "05"), (0.005, "005"), (0.0005, "0005")]:
            k = int((valid.p <= alpha).sum())
            result.update(
                {
                    "hits_" + label: k,
                    "rate_" + label: k / n if n else np.nan,
                    "scheduled_rate_" + label: k / len(part),
                    "lower_" + label: float(beta_dist.ppf(0.025, k, n - k + 1))
                    if k
                    else 0.0,
                    "upper_" + label: float(beta_dist.ppf(0.975, k + 1, n - k))
                    if k < n
                    else 1.0,
                    "one_sided_upper_" + label: float(beta_dist.ppf(0.95, k + 1, n - k))
                    if k < n
                    else 1.0,
                }
            )
        beta_name = "beta0" if "beta0" in valid else "beta"
        se_name = "se0" if "se0" in valid else "se"
        truth_name = "truth0" if "truth0" in valid else "truth"
        truth = valid[truth_name].fillna(0).to_numpy() if truth_name in valid else 0.0
        error = valid[beta_name].to_numpy() - truth
        ses = valid[se_name].to_numpy()
        result.update(
            bias=float(np.mean(error)),
            error_sd=float(np.std(error, ddof=1)),
            coefficient_sd=float(valid[beta_name].std(ddof=1)),
            rmse=float(np.sqrt(np.mean(error**2))),
            mean_se=float(np.mean(ses)),
            coverage=float(np.mean(abs(error) <= 1.95996398454 * ses)),
        )
        records.append(result)
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("benchmarks/epistasis"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--whole-population", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    full = []
    nested = []
    for label in ("development", "confirmation", "omitted_confirmation"):
        base = a.root / f"trans_{label}_20261003"
        f = pd.read_csv(base / "replicates.csv")
        f["panel"] = label
        full.append(f)
        f = pd.read_csv(base / "population_resampling.csv")
        f["panel"] = label
        nested.append(f)
    full = reduce(pd.concat(full), ["panel", "method", "setting"])
    nested = reduce(pd.concat(nested), ["panel", "outer", "noise"])
    full.to_csv(a.out / "full_fits.csv", index=False)
    nested.to_csv(a.out / "frozen_model_population.csv", index=False)
    obstruct = reduce(
        pd.read_csv(a.root / "trans_obstructions_20261003/replicates.csv"),
        ["experiment", "method"],
    )
    obstruct.to_csv(a.out / "obstructions.csv", index=False)
    tails = []
    matched = []
    for label, dirname in [
        ("development", "trans_tails_20261003"),
        ("confirmation", "trans_tails_confirmation_20261003"),
    ]:
        f = pd.DataFrame(json.loads((a.root / dirname / "tails.json").read_text()))
        f["panel"] = label
        tails.append(f)
        design = json.loads((a.root / dirname / "design.json").read_text())
        for c in design["independent_production_comparisons"]:
            if "matched_wild_p" in c:
                normal = np.array(c["matched_normal_p"])
                wild = np.array(c["matched_wild_p"])
                matched.append(
                    dict(
                        panel=label,
                        n=c["n"],
                        design=c["design"],
                        noise=c["noise"],
                        draws=len(normal),
                        normal_hits_05=int(np.sum(normal <= 0.05)),
                        wild_hits_05=int(np.sum(wild <= 0.05)),
                        mean_absolute_p_difference=float(np.mean(abs(normal - wild))),
                        maximum_absolute_p_difference=float(np.max(abs(normal - wild))),
                    )
                )
    tails = pd.concat(tails)
    tails.to_csv(a.out / "tails.csv", index=False)
    pd.DataFrame(matched).to_csv(a.out / "matched_wild.csv", index=False)
    if a.whole_population:
        reduce(
            pd.read_csv(a.whole_population / "replicates.csv"),
            ["noise", "signal_expected_variance"],
        ).to_csv(a.out / "whole_population.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 170,
        }
    )
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    sub = nested[
        nested.panel.isin(["confirmation", "omitted_confirmation"])
    ].reset_index(drop=True)
    x = np.arange(len(sub))
    axs[0].errorbar(
        x,
        sub.rate_05,
        yerr=[sub.rate_05 - sub.lower_05, sub.upper_05 - sub.rate_05],
        fmt="o",
        capsize=3,
    )
    axs[0].axhline(0.05, color="black", lw=1)
    axs[0].axhline(
        0.075, color="grey", ls="--", lw=1, label="Material-inflation screen"
    )
    axs[0].set(
        xticks=x,
        xticklabels=[
            f"{'O' if r.panel.startswith('omitted') else 'C'}{r.outer} {r.noise}"
            for r in sub.itertuples()
        ],
        ylabel="Rejection probability at 0.05",
        title="Fresh genotype draws; each training model frozen",
    )
    axs[0].tick_params(axis="x", labelrotation=65)
    axs[0].legend(fontsize=8)
    settings = [
        "aligned_weak",
        "aligned_moderate",
        "sparse_moderate",
        "distributed_moderate",
    ]
    for j, method in enumerate(["learned", "burden"]):
        v = (
            full[(full.panel == "confirmation") & (full.method == method)]
            .set_index("setting")
            .loc[settings]
        )
        xx = np.arange(4) + (j - 0.5) * 0.18
        axs[1].errorbar(
            xx,
            v.rate_005,
            yerr=[v.rate_005 - v.lower_005, v.upper_005 - v.rate_005],
            fmt="o",
            capsize=3,
            label=method,
        )
    axs[1].set(
        xticks=np.arange(4),
        xticklabels=["Aligned\nweak", "Aligned\nmoderate", "Sparse", "Mixed sign"],
        ylim=(-0.02, 1.05),
        ylabel="Power at 0.005 (family of ten)",
        title="100 independent complete fits; 95% MC intervals",
    )
    axs[1].legend()
    fig.savefig(a.out / "population_calibration_power.png")
    fig.savefig(a.out / "population_calibration_power.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, alpha in zip(axs, [0.005, 0.0005]):
        for j, (design, noise) in enumerate(
            [
                (d, n)
                for d in ["common", "sparse"]
                for n in ["hetero_gaussian", "hetero_t5"]
            ]
        ):
            v = tails[
                (tails.panel == "confirmation")
                & (tails.method == "HC3")
                & (tails.signal == 0)
                & (tails.alpha == alpha)
                & (tails.design == design)
                & (tails.noise == noise)
            ].sort_values("n")
            lo = np.array([v[0] for v in v["interval"]])
            hi = np.array([v[1] for v in v["interval"]])
            ax.errorbar(
                np.log2(v.n) + (j - 1.5) * 0.05,
                v.rate,
                yerr=[v.rate - lo, hi - v.rate],
                fmt="o",
                capsize=2,
                label=design + " " + noise.removeprefix("hetero_"),
            )
        ax.axhline(alpha, color="black", lw=1)
        ax.set(
            xticks=[11, 13, 15],
            xticklabels=["2,048", "8,192", "32,768"],
            xlabel="People; fixed design",
            ylabel="Rejection probability (log scale)",
            title=f"20,000 residual draws; threshold {alpha:g}",
            yscale="log",
        )
        ax.annotate(
            "Sparse N=8,192 undefined",
            xy=(0.49, 0.96),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=8,
        )
    axs[0].legend(fontsize=8)
    fig.savefig(a.out / "tail_calibration_log.png")
    fig.savefig(a.out / "tail_calibration_log.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    v = full[(full.panel == "confirmation") & (full.method == "learned")].reset_index(
        drop=True
    )
    x = np.arange(len(v))
    axs[0].errorbar(
        x, v.bias, yerr=1.96 * v.error_sd / np.sqrt(v.valid), fmt="o", capsize=3
    )
    axs[0].axhline(0, color="black", lw=1)
    axs[0].set(
        xticks=x,
        xticklabels=v.setting,
        ylabel="Mean coefficient error (95% MC interval)",
        title="Replicate-specific learned-direction truth",
    )
    axs[0].tick_params(axis="x", labelrotation=65)
    axs[1].plot(x, v.error_sd, "o-", label="SD(estimate − target)")
    axs[1].plot(x, v.mean_se, "s-", label="Mean reported SE")
    axs[1].set(
        xticks=x,
        xticklabels=v.setting,
        ylabel="Coefficient units",
        title="Uncertainty after complete retraining",
    )
    axs[1].tick_params(axis="x", labelrotation=65)
    axs[1].legend()
    fig.savefig(a.out / "estimation_uncertainty.png")
    fig.savefig(a.out / "estimation_uncertainty.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
