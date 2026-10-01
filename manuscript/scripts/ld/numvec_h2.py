#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
POPS = ["EUR_300k", "EUR", "SAS", "AFR"]
DODGE_ORDER = ["EUR_300k", "EUR", "SAS", "AFR"]
POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "SAS": "SAS",
    "AFR": "AFR",
}
POP_COLOR = {
    "EUR_300k": TAB20[1],
    "EUR": TAB20[0],
    "SAS": TAB20[4],
    "AFR": TAB20[6],
}
TARGET_BS = [1000, 2000, 5000, 8000, 10000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot total h2 sensitivity to LD-score random vectors."
    )
    parser.add_argument(
        "--csv",
        default="data/ld/numvec_h2.csv",
        help="Parsed total-h2 CSV.",
    )
    parser.add_argument(
        "--outfile",
        default="figs/supplementary/fig_s22_h2_projection_sensitivity.pdf",
        help="Output PDF path. A PNG with the same basename is also written.",
    )
    parser.add_argument("--xscale", choices=["log", "linear"], default="log")
    parser.add_argument(
        "--dodge-log10",
        type=float,
        default=0.045,
        help=(
            "Half-width of population dodge in log10 units. "
            "Default gives about +/-10%% horizontal offset; use 0 to disable."
        ),
    )
    return parser.parse_args()


def pop_dodge_factors(width_log10: float) -> dict[str, float]:
    if width_log10 <= 0:
        return {pop: 1.0 for pop in POPS}
    offsets = np.linspace(-width_log10, width_log10, len(DODGE_ORDER))
    return {pop: float(10.0**offset) for pop, offset in zip(DODGE_ORDER, offsets)}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"No rows found in {args.csv}")

    phen_order = ["HT", "BMI", "EA"]
    for label in df["phen_label"].dropna().astype(str).unique():
        if label not in phen_order:
            phen_order.append(label)
    phen_order = [
        label for label in phen_order if label in set(df["phen_label"].astype(str))
    ]

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "legend.fontsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(phen_order),
        figsize=(14.7, 4.7),
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    if len(phen_order) == 1:
        axes = [axes]

    dodge_factor = pop_dodge_factors(args.dodge_log10)
    x_min = min(TARGET_BS) * min(dodge_factor.values()) * 0.93
    x_max = max(TARGET_BS) * max(dodge_factor.values()) * 1.08

    for ax, phen_label in zip(axes, phen_order):
        sub = df[df["phen_label"].astype(str) == phen_label].copy()
        for pop in POPS:
            pop_df = sub[sub["pop"] == pop].sort_values("numvec")
            if pop_df.empty:
                continue
            x = pop_df["numvec"].to_numpy(dtype=float) * dodge_factor[pop]
            y = pop_df["h2"].to_numpy(dtype=float)
            yerr = pop_df["h2_se"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                color=POP_COLOR[pop],
                markersize=5.8,
                linewidth=2.2,
                elinewidth=1.05,
                capsize=2.6,
                capthick=1.05,
                alpha=0.98,
                zorder=3,
            )

        ax.set_title(phen_label, pad=9)
        ax.set_xlabel("Random vectors (x1000)")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if args.xscale == "log":
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
        else:
            ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(mticker.FixedLocator(TARGET_BS))
        ax.xaxis.set_major_formatter(
            mticker.FixedFormatter([str(x // 1000) for x in TARGET_BS])
        )
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", which="major", labelsize=10.5, rotation=0)

    axes[0].set_ylabel("Total $h^2$")

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
        bbox_to_anchor=(0.5, 1.035),
        handlelength=2.2,
        columnspacing=1.45,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.25)

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    print(f"Wrote figure to {out_path}")
    print(f"Wrote figure to {png_path}")


if __name__ == "__main__":
    main()
