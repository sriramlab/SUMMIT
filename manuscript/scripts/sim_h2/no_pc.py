"""Plot no pc for the SUMMIT manuscript."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
from pc_residualization import (
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    POPS,
    draw_summary,
    finite_values,
    gmean_pos,
)


COMPARISON_ORDER = ["pc", "rhe"]


COMPARISON_LABEL = {
    "pc": "no-PC / PC",
    "rhe": "no-PC / RHE-mc",
}


def ratio_ticks(ylim: tuple[float, float]) -> list[float]:
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]
    return [tick for tick in candidates if ylim[0] <= tick <= ylim[1]]


def style_ratio_axis(ax, ylim: tuple[float, float]) -> None:
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ticks = ratio_ticks(ylim)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.65)


def make_figure(df: pd.DataFrame, args: argparse.Namespace) -> Path:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(
        2,
        len(args.pops),
        figsize=(3.15 * len(args.pops), 5.25),
        sharey="row",
    )
    if len(args.pops) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    shared_ylim = tuple(args.ratio_ylim)

    for row, comparison in enumerate(COMPARISON_ORDER):
        row_df = df[df["comparison"] == comparison]
        for col, pop in enumerate(args.pops):
            ax = axes[row, col]
            sub = row_df[row_df["pop"] == pop].copy()
            draw_summary(ax, sub, "mse_ratio", "gmean", rng)
            style_ratio_axis(ax, shared_ylim)
            if row == 0:
                ax.set_title(pop)
                ax.tick_params(axis="x", labelbottom=False)
            if col == 0:
                ax.set_ylabel(f"MSE ratio\n{COMPARISON_LABEL[comparison]}")
            else:
                ax.set_ylabel("")

    sns.despine(fig=fig)
    fig.tight_layout(w_pad=1.1, h_pad=0.75)

    out = args.outdir / f"{args.out_prefix}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    make_figure(
        pd.read_csv("data/sim_h2/no_pc_comparison.tsv", sep="\t"),
        argparse.Namespace(
            pops=["EUR", "SAS", "AFR"],
            seed=0,
            ratio_ylim=(0.5, 100.0),
            outdir=Path("figs/supplementary"),
            out_prefix="fig_s09_pc_adjustment",
        ),
    )
