"""Plot missing snps for the SUMMIT manuscript."""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POPS = ["EUR", "SAS", "AFR"]


H2_VALUES = [0.10, 0.25, 0.40]


PCAUSAL_VALUES = [0.01, 0.10, 1.00]


COLORS = {0.01: "#7b3294", 0.10: "#2c7fb8", 1.00: "#1a9850"}


MARKERS = {0.01: "^", 0.10: "s", 1.00: "o"}


def plot(summary: pd.DataFrame, out: Path) -> None:
    mpl.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.1), sharex=True, sharey=True)
    fractions = np.sort(summary["missing_fraction"].unique())
    for row, true_h2 in enumerate(H2_VALUES):
        for col, pop in enumerate(POPS):
            ax = axes[row, col]
            for pcausal in PCAUSAL_VALUES:
                sub = summary[
                    summary["pop"].eq(pop)
                    & summary["true_h2"].eq(true_h2)
                    & summary["pcausal"].eq(pcausal)
                ].sort_values("missing_fraction")
                x = 100 * sub["missing_fraction"].to_numpy(float)
                y = sub["proportion"].to_numpy(float)
                lo = sub["ci95_low"].to_numpy(float)
                hi = sub["ci95_high"].to_numpy(float)
                ax.errorbar(
                    x,
                    y,
                    yerr=np.vstack([y - lo, hi - y]),
                    color=COLORS[pcausal],
                    marker=MARKERS[pcausal],
                    linewidth=1.6,
                    markersize=5.7,
                    capsize=2.5,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=rf"$p_{{\mathrm{{causal}}}}={pcausal:g}$",
                )
            ax.plot(
                100 * fractions,
                1 - fractions,
                color="#555555",
                linestyle=(0, (4, 2)),
                linewidth=1.2,
                label="Retained SNP proportion",
            )
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(100 * fractions)
            ax.set_ylim(0.42, 1.08)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            if row == 0:
                ax.set_title(pop, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    rf"$h_g^2={true_h2:g}$" + "\nEstimate / complete-panel estimate"
                )
            if row == 2:
                ax.set_xlabel("SNPs omitted (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.4, w_pad=1.3)
    for suffix in ["pdf", "png"]:
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(
            out / f"fig_s11_missing_snp_sensitivity.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    plot(pd.read_csv("data/sim_h2/missing_snp_summary.csv"), Path("figs/supplementary"))
