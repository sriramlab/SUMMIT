"""Recompute milestone reductions and figures from preserved replicate records."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cp(k, n):
    return (
        0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1),
        1.0 if k == n else beta.ppf(0.975, k + 1, n - k),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=Path("benchmarks/epistasis"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    frames = []
    summaries = []
    for label in ("a", "b"):
        directory = a.root / f"robust_confirm_scope_{label}_checked_20261002"
        frame = pd.read_csv(directory / "replicates.csv")
        frame["panel"] = label
        frames.append(frame)
        s = pd.read_csv(directory / "summary.csv")
        s["panel"] = label
        summaries.append(s)
    real = pd.concat(frames)
    summary = pd.concat(summaries)
    rows = []
    families = [
        "supplied_pairs",
        "target_region",
        "set_cross",
        "set_overlap",
        "within",
        "target_genome_sketch8",
        "set_remainder_sketch8",
        "weighted_score",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for j, family in enumerate(families):
        part = real[
            (real.method == "HC3_kernel")
            & real.setting.isin(
                [f"real_{family}_null_gaussian", f"real_{family}_null_hetero_dominance"]
            )
        ]
        n = len(part)
        k = int((part.p <= 0.05).sum())
        lo, hi = cp(k, n)
        rate = k / n
        axes[0].errorbar(
            rate,
            j,
            xerr=[[rate - lo], [hi - rate]],
            fmt="o",
            color="#a2552c"
            if part.leverage.max() > 0.1 or part.effective_support.min() < 100
            else "#28688c",
            capsize=3,
        )
        rows.append(
            dict(
                family=family,
                fits=n,
                failures=int(part.failed.sum()),
                rejections=k,
                rate=rate,
                lower=lo,
                upper=hi,
                coverage=part.coverage.mean(),
                maximum_leverage=part.leverage.max(),
                minimum_effective_support=part.effective_support.min(),
            )
        )
    axes[0].set(
        yticks=range(len(families)),
        yticklabels=[v.replace("_", " ") for v in families],
        xlabel="Null rejection at .05 (binomial reference 95% interval)",
        xlim=(0, 0.13),
        title="N=4,096, M=8,192; orange includes out-of-scope designs",
    )
    axes[0].axvline(0.05, ls="--", color="grey")
    axes[0].invert_yaxis()
    selected = summary[
        (summary.method == "HC3_kernel") & summary.setting.str.contains("_signal_")
    ]
    axes[1].scatter(
        selected.empirical_sd,
        selected.mean_se,
        c=selected.panel.map({"a": "#28688c", "b": "#a2552c"}),
        alpha=0.8,
    )
    limit = max(selected.empirical_sd.max(), selected.mean_se.max()) * 1.1
    axes[1].plot([0, limit], [0, limit], "--", color="grey")
    axes[1].set(
        xlabel="Empirical SD of first mean coefficient",
        ylabel="Mean HC3 standard error",
        title="Correct-mean fixed-effect alternatives",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    fig.savefig(a.out / "calibration_and_estimation.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(a.out / "pooled_real_nulls.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for i, label in enumerate(("a", "b")):
        part = summary[(summary.panel == label) & summary.method.eq("HC3_kernel")]
        part = part.set_index("setting").loc[
            [f"real_{f}_signal_mixed01" for f in families]
        ]
        axes[0].plot(
            part.coverage,
            np.arange(len(families)) + (i - 0.5) * 0.15,
            "o",
            label="Panel " + label,
        )
        axes[1].plot(
            part.bias / part.empirical_sd,
            np.arange(len(families)) + (i - 0.5) * 0.15,
            "o",
        )
    axes[0].axvline(0.95, color="grey", ls="--")
    axes[0].legend()
    axes[0].set(
        xlabel="Mean per-coefficient 95% interval coverage",
        xlim=(0.85, 1),
        title="100 residual fits per setting; fixed mean effects",
    )
    axes[1].axvline(0, color="grey", ls="--")
    axes[1].set(
        xlabel="Bias / empirical SD, first coefficient",
        title="Group designs outside scope remain visible",
    )
    for ax in axes:
        ax.set(
            yticks=range(len(families)),
            yticklabels=[f.replace("_", " ") for f in families],
        )
        ax.invert_yaxis()
    fig.savefig(a.out / "coverage_and_bias.png", dpi=180)
    plt.close(fig)

    tails = pd.read_csv(a.root / "robust_tails_scope_20261002/tails.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, method, alpha in zip(axes, ["HC3_single", "HC3_family10"], [0.005, 0.05]):
        part = tails[(tails.method == method) & (tails.alpha == alpha)]
        ax.errorbar(
            np.arange(3),
            part.rate,
            yerr=np.array([part.rate - part.lower, part.upper - part.rate]),
            fmt="o",
            capsize=4,
        )
        ax.axhline(alpha, color="grey", ls="--")
        ax.set(
            xticks=range(3),
            xticklabels=["Gaussian", "Heteroskedastic", "t5"],
            ylabel="Rejection probability",
            title=f"{method}: alpha {alpha}; 20,000 draws each",
        )
    fig.savefig(a.out / "conditional_tails.png", dpi=180)
    plt.close(fig)

    # "null" is a scientific setting label, not a missing-value token.
    direction = pd.read_csv(
        a.root / "direction_confirm_scope_20261002/replicates.csv",
        keep_default_na=False,
    )
    methods = [
        "learned_direction",
        "additive_PGS_direction",
        "regional_kernel",
        "supplied_burden",
    ]
    settings = ["null", "mixed_weak", "mixed_moderate", "aligned_weak"]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    direction_rows = []
    for j, method in enumerate(methods):
        rates = []
        err = []
        for setting in settings:
            part = direction[
                (direction.setting == setting) & (direction.method == method)
            ]
            n = len(part)
            k = int((part.p <= 0.05).sum())
            lo, hi = cp(k, n)
            v = k / n
            rates.append(v)
            err.append([v - lo, hi - v])
            direction_rows.append(
                dict(
                    setting=setting,
                    method=method,
                    n=n,
                    failures=int(part.failed.sum()),
                    rate=v,
                    lower=lo,
                    upper=hi,
                )
            )
        ax.bar(
            np.arange(4) + (j - 1.5) * 0.19,
            rates,
            0.18,
            yerr=np.asarray(err).T,
            capsize=2,
            label=method.replace("_", " "),
        )
    ax.axhline(0.05, ls="--", color="grey")
    ax.set(
        xticks=range(4),
        xticklabels=["Null", "Mixed .002", "Mixed .01", "Aligned .002"],
        ylabel="Rejection at .05",
        ylim=(0, 1),
        title="Independent training: 2,048 training / 8,000 test / 16,384 markers",
    )
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(a.out / "independent_direction_power.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(direction_rows).to_csv(a.out / "direction_counts.csv", index=False)

    mech = pd.read_csv(a.root / "robust_mechanisms_20261002/replicates.csv")
    dense = pd.read_csv(a.root / "dense_followup_both_checked_20261002/replicates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    choices = [
        "global_nonlinearity_signal0.0_linear",
        "global_nonlinearity_signal0.0_frozen_curve",
        "global_nonlinearity_signal0.0_declared_cubic_refitted",
        "misspecified_single_index",
        "scale_raw",
        "scale_log",
        "scale_frozen_rank_normal",
    ]
    labels = [
        "Global nonlinear, linear baseline",
        "Independently trained cubic curve",
        "Declared cubic basis, refitted",
        "Misspecified index",
        "Prespecified raw scale",
        "Prespecified log scale",
        "Frozen external rank transform",
    ]
    for j, name in enumerate(choices):
        part = mech[mech.setting == name]
        n = len(part)
        k = int((part.p <= 0.05).sum())
        lo, hi = cp(k, n)
        v = k / n
        axes[0].errorbar(v, j, xerr=[[v - lo], [hi - v]], fmt="o", capsize=3)
    axes[0].set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="Rejection at .05",
        xlim=(-0.02, 1.02),
        title="Phenotype scale and nonlinear mean",
    )
    axes[0].invert_yaxis()
    axes[0].axvline(0.05, color="grey", ls="--")
    for j, method in enumerate(
        [
            "discovery_array_linear_exact",
            "confirmation_array_linear_exact",
            "confirmation_dense_linear_exact",
        ]
    ):
        values = []
        errs = []
        for center in (10000000, 50000000, 150000000):
            part = dense[(dense.setting == center) & (dense.method == method)]
            n = len(part)
            k = int((part.p <= 0.05).sum())
            lo, hi = cp(k, n)
            v = k / n
            values.append(v)
            errs.append([v - lo, hi - v])
        axes[1].bar(
            np.arange(3) + (j - 1) * 0.24,
            values,
            0.23,
            yerr=np.asarray(errs).T,
            label=method.replace("_linear_exact", "").replace("_", " "),
            capsize=2,
        )
    axes[1].set(
        xticks=range(3),
        xticklabels=["Chr1 10 Mb", "Chr1 50 Mb", "Chr1 150 Mb"],
        ylabel="Rejection at .05",
        ylim=(0, 1.15),
        title="Hidden additive causal locus; Gaussian follow-up",
    )
    axes[1].legend(fontsize=8)
    axes[1].axhline(0.05, color="grey", ls="--")
    axes[1].text(
        0.02,
        0.98,
        "Dense HC3: all 300 fits undefined\n(retained; not counted as null non-rejections)",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    fig.savefig(a.out / "nonlinear_scale_and_dense_followup.png", dpi=180)
    plt.close(fig)

    names = [
        "robust_disk_portable_4096_20261002",
        "robust_disk_portable_16384_20261002",
        "robust_disk_private_16384_20261002",
        "robust_disk_portable_32768_20261002",
    ]
    workloads = [json.loads((a.root / (name + ".json")).read_text()) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    labels = [
        "4k/8k\nportable",
        "16k/32k\nportable",
        "16k/32k\nprivate",
        "32k/64k\nportable",
    ]
    for ax, key in zip(axes, ("seconds", "peak_process_rss_bytes")):
        values = [
            sum(r[key] for r in w["records"])
            if key == "seconds"
            else max(r[key] for r in w["records"]) / 2**30
            for w in workloads
        ]
        ax.bar(range(4), values, color=["#28688c", "#28688c", "#a2552c", "#28688c"])
        ax.set(
            xticks=range(4),
            xticklabels=labels,
            ylabel="Seconds" if key == "seconds" else "GiB",
            title="Complete measured stages, two physical CPUs"
            if key == "seconds"
            else "Process peak RSS, including fixture generation",
        )
    fig.savefig(a.out / "complete_workloads.png", dpi=180)
    plt.close(fig)
    sketch = pd.read_csv(a.root / "robust_native_sketches_checked_20261002/summary.csv")
    sketch["dimensions"] = sketch.setting.str.split("_").str[0]
    sketch["null"] = sketch.setting.str.contains("null")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for j, dim in enumerate(["exact", "8", "16"]):
        for flag, ax in zip([True, False], axes):
            part = sketch[(sketch.dimensions == dim) & (sketch["null"] == flag)]
            valid = part[part.failures == 0]
            ax.scatter(
                j + np.linspace(-0.08, 0.08, len(valid)), valid.rejection, label=dim
            )
            if part.failures.sum():
                ax.text(
                    j,
                    0.01,
                    f"{sum(part.failures>0)} banks undefined",
                    ha="center",
                    color="#a2552c",
                    fontsize=8,
                )
    for ax, title in zip(
        axes,
        [
            "Same 100 null phenotypes across every bank",
            "Same 100 signal phenotypes; variance .005",
        ],
    ):
        ax.set(
            xticks=range(3),
            xticklabels=["Exact 32 pairs", "8 dimensions", "16 dimensions"],
            ylabel="Rejection at .05",
            ylim=(0, 0.4),
            title=title,
        )
    axes[0].axhline(0.05, color="grey", ls="--")
    fig.savefig(a.out / "native_sketch_variation.png", dpi=180)
    plt.close(fig)
    report = dict(
        primary_nulls=rows,
        direction=direction_rows,
        workloads=[
            dict(
                name=name,
                n=w["n"],
                m=w["m"],
                seconds=sum(r["seconds"] for r in w["records"]),
                cpu_seconds=sum(r["cpu_seconds"] for r in w["records"]),
                peak_rss_bytes=max(r["peak_process_rss_bytes"] for r in w["records"]),
            )
            for name, w in zip(names, workloads)
        ],
        inputs="preserved per-replicate files; failures retained, valid-only plot points explicitly labeled; sketch banks share phenotypes",
    )
    (a.out / "numbers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
