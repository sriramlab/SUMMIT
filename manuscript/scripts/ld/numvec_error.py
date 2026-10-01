"""Plot numvec error for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


COHORT_LABELS = {
    "AFR": "AFR",
    "SAS": "SAS",
    "EUR": "EUR",
    "EUR_300k": "EUR (300k)",
}


NUMVEC_COLORS = {
    10: "#9e3d22",
    100: "#3f6f9f",
    1000: "#2f7f5f",
    10000: "#2f2f2f",
}


def layout_shape(layout: str, n_panels: int) -> Tuple[int, int, Tuple[float, float]]:
    if layout == "row":
        return 1, n_panels, (7.4, 2.15)
    if layout == "grid":
        return 2, int(math.ceil(n_panels / 2)), (6.9, 5.1)
    raise ValueError(f"Unknown layout: {layout}")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def setup_axes(layout: str, n_panels: int):
    nrows, ncols, figsize = layout_shape(layout, n_panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat


def plot_error_figure(
    metrics: pd.DataFrame,
    pops: Sequence[str],
    numvecs: Sequence[int],
    ref_numvec: int,
    layout: str,
    outdir: Path,
    dpi: int,
) -> None:
    fig, axes = setup_axes(layout, len(pops))
    for idx, (ax, pop) in enumerate(zip(axes, pops)):
        work = metrics.loc[metrics["pop"].eq(pop)].copy()
        positions = list(numvecs)
        data = [
            work.loc[work["numvec"].eq(x), "rel_sum_error_pct"]
            .dropna()
            .to_numpy(dtype=float)
            for x in positions
        ]
        present_positions = [x for x, vals in zip(positions, data) if vals.size]
        present_data = [vals for vals in data if vals.size]
        if present_data:
            widths = [0.18 * x for x in present_positions]
            bp = ax.boxplot(
                present_data,
                positions=present_positions,
                widths=widths,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "black", "linewidth": 0.9},
                boxprops={"linewidth": 0.7},
                whiskerprops={"linewidth": 0.7},
                capprops={"linewidth": 0.7},
            )
            for box, x in zip(bp["boxes"], present_positions):
                box.set(
                    facecolor=NUMVEC_COLORS.get(x, "#777777"),
                    edgecolor="black",
                    alpha=0.38,
                )

            rng = np.random.default_rng(20260509 + idx)
            for x, vals in zip(present_positions, present_data):
                jitter = np.exp(rng.normal(0.0, 0.045, size=vals.size))
                ax.scatter(
                    np.full(vals.size, x, dtype=float) * jitter,
                    vals,
                    s=9,
                    color=NUMVEC_COLORS.get(x, "#333333"),
                    edgecolor="white",
                    linewidth=0.2,
                    alpha=0.86,
                    zorder=3,
                )

        ax.set_xscale("log")
        ax.axhline(
            0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.58, zorder=1
        )
        ax.set_xticks(positions)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{int(x):,}" if x >= 1000 else f"{int(x)}"
            )
        )
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(
            True, which="major", axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.7
        )
        ax.set_title(COHORT_LABELS.get(pop, pop), pad=3)
        ax.text(
            -0.12,
            1.12,
            PANEL_LABELS[idx],
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )
        ax.set_xlabel("Random vectors")
        if idx % (2 if layout == "grid" else len(pops)) == 0:
            ax.set_ylabel("Relative error in mean LD score (%)")

    fig.subplots_adjust(
        wspace=0.32 if layout == "row" else 0.28,
        hspace=0.45 if layout == "grid" else 0.1,
    )
    for ext in ["pdf", "png"]:
        fig.savefig(outdir / f"fig_s20_ld_projection_error.{ext}", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    plot_error_figure(
        pd.read_csv("data/ld/numvec_error.csv"),
        ["EUR_300k", "EUR", "SAS", "AFR"],
        [10, 100, 1000, 10000],
        10000,
        "row",
        Path("figs/supplementary"),
        600,
    )
