"""Reduce frozen continuation results without choosing methods or thresholds."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta
from scripts.epistasis.continuation_design import decision


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    root = Path("benchmarks/epistasis")
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for k, seed in enumerate((738291, 261849, 814763)):
        table = pd.read_csv(root / f"direction_nested_{seed}_20261002/replicates.csv")
        for j, (name, rows) in enumerate(table.groupby("method", sort=False)):
            valid = rows[~rows.failed]
            hits = int((valid.p < 0.05).sum())
            total = len(rows)
            status = decision(hits, len(valid))
            records.append(
                dict(
                    panel=seed,
                    method=name,
                    hits=hits,
                    draws=total,
                    failures=int(rows.failed.sum()),
                    **status,
                )
            )
            lo = 0 if hits == 0 else beta.ppf(0.025, hits, len(valid) - hits + 1)
            hi = beta.ppf(0.975, hits + 1, len(valid) - hits)
            pos = j + (k - 1) * 0.22
            axes[0].errorbar(
                pos,
                hits / len(valid),
                yerr=[[hits / len(valid) - lo], [hi - hits / len(valid)]],
                fmt="o",
                color=f"C{k}",
                label=str(seed) if j == 0 else None,
            )
    axes[0].axhline(0.05, color="black", ls="--")
    axes[0].axhline(0.075, color="red", ls=":")
    axes[0].set_xticks(
        range(4), ["Learned", "Additive PGS", "Kernel", "Burden"], rotation=20
    )
    axes[0].set_ylabel("Conditional null rejection, alpha=.05")
    axes[0].legend(title="Genotype/training configuration")
    full = pd.read_csv(root / "direction_span_confirmation_20261002/summary.csv")
    full["setting"] = full["setting"].fillna("null")
    full.to_csv(a.out / "full_confirmation.csv", index=False)
    for j, name in enumerate(full.method.unique()):
        rows = full[full.method == name]
        axes[1].errorbar(
            np.arange(len(rows)),
            rows.rejection,
            yerr=[rows.rejection - rows.lower, rows.upper - rows.rejection],
            fmt="o-",
            elinewidth=0.7,
            capsize=2,
            label=name,
        )
    axes[1].set_xticks(range(4), full.setting.unique(), rotation=20)
    axes[1].set_ylabel("Rejection / power (100 full retrainings)")
    axes[1].legend(fontsize=8)
    fig.savefig(a.out / "independent_calibration_power.png", dpi=180)
    plt.close(fig)
    tail = json.loads((root / "ratio_tails_checked_20261002.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    tt = pd.DataFrame([r for r in tail["results"] if "nominal_alpha" in r])
    tt.to_csv(a.out / "ratio_tails.csv", index=False)
    for case, rows in tt.groupby("case", sort=False):
        axes[0].loglog(rows.nominal_alpha, rows.p, "o-", label=case)
    axes[0].loglog([5e-8, 0.05], [5e-8, 0.05], "k--")
    axes[0].set_xlabel("Nominal two-sided alpha")
    axes[0].set_ylabel("Actual Gaussian HC3 rejection")
    axes[0].legend(fontsize=8)
    ortho = pd.read_csv(root / "orthogonal_real_confirmation_20261002/summary.csv")
    ortho["setting"] = ortho["setting"].fillna("null")
    ortho.to_csv(a.out / "orthogonal_confirmation.csv", index=False)
    for name, rows in ortho.groupby("method", sort=False):
        axes[1].errorbar(
            np.arange(len(rows)),
            rows.rejection,
            yerr=[rows.rejection - rows.lower, rows.upper - rows.rejection],
            fmt="o-",
            elinewidth=0.7,
            capsize=2,
            label=name,
        )
    axes[1].set_xticks(range(3), ortho.setting.unique())
    axes[1].set_ylabel("Rejection / power")
    axes[1].legend()
    fig.savefig(a.out / "tails_and_orthogonal.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, rows in ortho.groupby("method", sort=False):
        axes[0].errorbar(
            np.arange(len(rows)),
            rows.bias,
            yerr=1.959963984540054 * rows.empirical_sd / np.sqrt(rows.fits),
            fmt="o-",
            capsize=2,
            label=name,
        )
        hits = np.rint(rows.coverage * rows.fits).astype(int)
        lower = beta.ppf(0.025, hits, rows.fits - hits + 1)
        upper = beta.ppf(0.975, hits + 1, rows.fits - hits)
        axes[1].errorbar(
            np.arange(len(rows)),
            rows.coverage,
            yerr=[rows.coverage - lower, upper - rows.coverage],
            fmt="o-",
            capsize=2,
            label=name,
        )
    for ax in axes:
        ax.set_xticks(range(3), ortho.setting.unique())
        ax.legend()
    axes[0].set_ylabel("Coefficient bias")
    axes[1].set_ylabel("95% interval coverage")
    axes[1].axhline(0.95, color="black", ls="--")
    fig.savefig(a.out / "orthogonal_estimation.png", dpi=180)
    plt.close(fig)
    whole = json.loads((root / "whole_marker_panel_20261002.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.barh(
        [r["stage"] for r in whole["records"]], [r["seconds"] for r in whole["records"]]
    )
    ax.set_xlabel("Measured wall seconds")
    ax.set_title(
        f"Real BED N={whole['n']:,}, M={whole['m']:,}; training N={whole['training_n']:,}"
    )
    fig.savefig(a.out / "whole_marker_workload.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(a.out / "conditional_qualification.csv", index=False)
    (a.out / "numbers.json").write_text(
        json.dumps(
            dict(
                conditional=records,
                whole_sizes={
                    k: whole[k] for k in ("n", "m", "training_n", "confirmation_n")
                },
                whole_total_seconds=sum(r["seconds"] for r in whole["records"]),
                whole_peak_rss_bytes=max(
                    r.get("peak_rss_bytes", 0) for r in whole["records"]
                ),
                caveat="Nested draws condition on three distinct realized means and fitted training models; they are not 6000 independent full pipelines.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
