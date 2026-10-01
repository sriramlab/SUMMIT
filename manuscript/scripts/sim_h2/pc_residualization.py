"""Plot pc residualization for the SUMMIT manuscript."""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POPS = ["EUR", "SAS", "AFR"]


CATEGORY_ORDER = ["gw", "mafld_gcta", "mafld_ldak"]


CATEGORY_LABEL = {
    "gw": "Single-component",
    "mafld_gcta": "MAF-LD GCTA",
    "mafld_ldak": "MAF-LD LDAK",
}


CATEGORY_TICK = {
    "gw": "Single-\ncomponent",
    "mafld_gcta": "MAF-LD\nGCTA",
    "mafld_ldak": "MAF-LD\nLDAK",
}


TAB20 = list(plt.get_cmap("tab20").colors)


CATEGORY_COLOR = {
    "gw": TAB20[14],
    "mafld_gcta": TAB20[0],
    "mafld_ldak": TAB20[2],
}


def finite_values(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


def gmean_pos(x: Iterable[float]) -> float:
    vals = finite_values(x)
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(vals))))


def draw_summary(
    ax, sub: pd.DataFrame, y_col: str, summary: str, rng: np.random.Generator
):
    x_lookup = {category: i for i, category in enumerate(CATEGORY_ORDER)}
    for category in CATEGORY_ORDER:
        vals = pd.to_numeric(
            sub[sub["category"] == category][y_col], errors="coerce"
        ).to_numpy(float)
        vals = vals[np.isfinite(vals)]
        x = float(x_lookup[category])
        color = CATEGORY_COLOR[category]

        if vals.size:
            jitter = rng.uniform(-0.13, 0.13, size=vals.size)
            ax.scatter(
                np.full(vals.size, x) + jitter,
                vals,
                s=28,
                alpha=0.52,
                color=color,
                edgecolors="none",
                zorder=2,
            )

            if vals.size >= 2:
                q25 = float(np.nanpercentile(vals, 25))
                q75 = float(np.nanpercentile(vals, 75))
                ax.plot([x, x], [q25, q75], color=color, linewidth=2.0, zorder=3)
                ax.plot(
                    [x - 0.07, x + 0.07],
                    [q25, q25],
                    color=color,
                    linewidth=2.0,
                    zorder=3,
                )
                ax.plot(
                    [x - 0.07, x + 0.07],
                    [q75, q75],
                    color=color,
                    linewidth=2.0,
                    zorder=3,
                )

            center = gmean_pos(vals) if summary == "gmean" else float(np.nanmean(vals))
            if np.isfinite(center):
                ax.scatter(
                    [x],
                    [center],
                    marker="D",
                    s=82,
                    color=color,
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=4,
                )

    ax.set_xlim(-0.45, len(CATEGORY_ORDER) - 0.55)
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_TICK[c] for c in CATEGORY_ORDER], fontsize=10)
    ax.grid(axis="x", visible=False)
