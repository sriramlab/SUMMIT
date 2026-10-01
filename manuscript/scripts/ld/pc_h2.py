#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
POPS = ["EUR_300k", "EUR", "SAS", "AFR"]
POP_COLOR = {
    "EUR": TAB20[0],
    "EUR_300k": TAB20[1],
    "AFR": TAB20[6],
    "SAS": TAB20[4],
}
POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "AFR": "AFR",
    "SAS": "SAS",
}
OFFSET = {
    "EUR_300k": -0.42,
    "EUR": -0.14,
    "SAS": 0.14,
    "AFR": 0.42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA sensitivity total h2 trends."
    )
    parser.add_argument(
        "--csv",
        default="data/ld/pc_sensitivity_h2.csv",
        help="Parsed total-h2 CSV.",
    )
    parser.add_argument(
        "--outfile",
        default="figs/supplementary/fig_s04_pc_sensitivity_h2.pdf",
        help="Primary output figure path.",
    )
    return parser.parse_args()


def resolve_png_path(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(".png")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"No rows found in {args.csv}")

    phen_order = ["HT", "BMI", "EA"]
    for phen_label in df["phen_label"].tolist():
        if phen_label not in phen_order:
            phen_order.append(phen_label)

    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(
        1,
        len(phen_order),
        figsize=(14.5, 4.6),
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    if len(phen_order) == 1:
        axes = [axes]

    x_values = sorted(df["num_pcs"].unique())
    x_min = min(x_values) - 2
    x_max = max(x_values) + 2

    for ax, phen_label in zip(axes, phen_order):
        sub = df[df["phen_label"] == phen_label].copy()
        for pop in POPS:
            pop_df = sub[sub["pop"] == pop].sort_values("num_pcs")
            if pop_df.empty:
                continue
            x = pop_df["num_pcs"].to_numpy(dtype=float) + OFFSET[pop]
            y = pop_df["h2"].to_numpy(dtype=float)
            yerr = pop_df["h2_se"].to_numpy(dtype=float)

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                color=POP_COLOR[pop],
                markersize=5.5,
                linewidth=2.2,
                elinewidth=1.2,
                capsize=2.8,
                capthick=1.2,
                alpha=0.98,
                zorder=3,
            )

        ax.set_title(phen_label, fontsize=15, pad=10)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(x_values)
        ax.set_xlabel("Number of genotype PCs", fontsize=12)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Total $h^2$", fontsize=13)

    handles = [
        Line2D(
            [0],
            [0],
            color=POP_COLOR[pop],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=POP_LABEL[pop],
        )
        for pop in POPS
    ]
    fig.legend(
        handles=handles,
        labels=[POP_LABEL[pop] for pop in POPS],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=2.2,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.3)

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = resolve_png_path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Wrote figure to {out_path}")
    print(f"Wrote figure to {png_path}")


if __name__ == "__main__":
    main()
