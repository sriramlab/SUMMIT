"""Plot gazal for the SUMMIT manuscript."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import gazal_labels as gazal_plot


def finite_xy(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return df[np.isfinite(df[x]) & np.isfinite(df[y])].copy()


def plot_gazal_callouts(
    merged: pd.DataFrame, out_prefix: Path, *, estimator_label: str
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    dat = merged[
        merged["dataset_kind"].eq("local") & merged["pop"].eq("EUR_300k")
    ].copy()
    dat["is_selected"] = dat["annotation"].isin(gazal_plot.CALLOUT_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.05))
    fig.subplots_adjust(left=0.073, right=0.985, bottom=0.145, top=0.870, wspace=0.260)

    panels = [
        (
            axes[0],
            finite_xy(dat, "gazal_tau_star", "summit_tau_star"),
            "gazal_tau_star",
            "summit_tau_star",
            "gazal_tau_star_se",
            "summit_tau_star_se",
            r"$\tau^*$",
            (-0.60, 0.78),
        ),
        (
            axes[1],
            finite_xy(dat, "gazal_log2_enrichment", "summit_log2_enrichment"),
            "gazal_log2_enrichment",
            "summit_log2_enrichment",
            "gazal_log2_enrichment_se",
            "summit_log2_enrichment_se",
            r"$\log_2$ enrichment",
            (-0.75, 3.95),
        ),
    ]

    for i, (ax, df, x, y, xerr, yerr, title, lim) in enumerate(panels):
        lo, hi = lim
        ax.axhline(0, color="#D0D3D8", lw=0.8, zorder=0)
        ax.axvline(0, color="#D0D3D8", lw=0.8, zorder=0)
        ax.plot([lo, hi], [lo, hi], color="#6F6F6F", lw=1.05, ls="--", zorder=1)

        base = df[~df["is_selected"]]
        ax.errorbar(
            base[x],
            base[y],
            xerr=base[xerr],
            yerr=base[yerr],
            fmt="o",
            ms=3.9,
            color="#8EA9C4",
            ecolor="#D5E0EA",
            elinewidth=0.50,
            capsize=0,
            alpha=0.42,
            zorder=2,
        )
        selected = df[df["is_selected"]].copy()
        ax.errorbar(
            selected[x],
            selected[y],
            xerr=selected[xerr],
            yerr=selected[yerr],
            fmt="o",
            ms=5.7,
            color="#D55E00",
            markeredgecolor="white",
            markeredgewidth=0.65,
            ecolor="#E7A673",
            elinewidth=0.75,
            capsize=0,
            alpha=0.95,
            zorder=3,
        )

        for row in selected.itertuples(index=False):
            if row.annotation not in gazal_plot.CALLOUT_POSITIONS[title]:
                continue
            ax.annotate(
                gazal_plot.CALLOUT_LABELS[row.annotation],
                xy=(getattr(row, x), getattr(row, y)),
                xytext=gazal_plot.CALLOUT_POSITIONS[title][row.annotation],
                textcoords="data",
                ha="left",
                va="center",
                fontsize=7.9,
                color="#2F2F2F",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "lw": 0.55,
                    "color": "#8C8C8C",
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
                zorder=5,
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, pad=8)
        ax.set_xlabel(f"Gazal et al. 2017 {title}")
        ax.set_ylabel(f"SUMMIT EUR (300k) {title}")
        ax.text(
            -0.10 if i == 0 else -0.07,
            1.05,
            chr(ord("A") + i),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        r = np.corrcoef(df[x], df[y])[0, 1] if len(df) > 2 else np.nan
        ax.text(
            0.035,
            0.965,
            f"n={len(df)}, r={r:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
        )
        ax.grid(axis="both", color="#ECECEC", lw=0.5, alpha=0.9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_gazal_callouts(
        pd.read_csv(
            "data/external/gazal_comparison.tsv", sep="\t", float_precision="round_trip"
        ),
        Path("figs/supplementary/fig_s30_baseline_gazal_comparison"),
        estimator_label="fixed IVW point estimates; SNP-block jackknife SE",
    )
