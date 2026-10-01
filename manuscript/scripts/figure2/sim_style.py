"""Plot sim style for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl


POPS_DEFAULT = ["EUR", "SAS", "AFR"]


METHODS = ["summit", "covldsc", "sumher_ldak"]


METHOD_LABEL = {
    "summit": "SUMMIT",
    "covldsc": "cov-LDSC",
    "sumher_ldak": "SumHer-LDAK",
}


METHOD_COLOR = {
    "summit": mpl.colormaps["tab20"].colors[4],
    "covldsc": mpl.colormaps["tab20"].colors[2],
    "sumher_ldak": mpl.colormaps["tab20"].colors[8],
}


METHOD_MARKER = {
    "summit": "o",
    "covldsc": "s",
    "sumher_ldak": "D",
}


VISUAL_PROFILES = {
    "standard": {"text": 1.0, "marker": 1.0, "line": 1.0},
    "readable": {"text": 1.25, "marker": 1.45, "line": 1.45},
}


_TEXT_SCALE = 1.0


_MARKER_SCALE = 1.0


_LINE_SCALE = 1.0


def set_visual_profile(profile: str = "standard") -> None:
    if profile not in VISUAL_PROFILES:
        raise ValueError(
            f"Unknown visual profile {profile!r}; expected one of {sorted(VISUAL_PROFILES)}"
        )
    global _TEXT_SCALE, _MARKER_SCALE, _LINE_SCALE
    scales = VISUAL_PROFILES[profile]
    _TEXT_SCALE = scales["text"]
    _MARKER_SCALE = scales["marker"]
    _LINE_SCALE = scales["line"]


def _fs(size: float) -> float:
    return size * _TEXT_SCALE


def _lw(width: float) -> float:
    return width * _LINE_SCALE


def _ms(size: float) -> float:
    return size * _MARKER_SCALE


def _scatter_size(area: float) -> float:
    return area * (_MARKER_SCALE**2)


def set_style(profile: str = "standard"):
    set_visual_profile(profile)
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": _fs(12.0),
            "axes.titlesize": _fs(14.0),
            "axes.labelsize": _fs(13.2),
            "axes.linewidth": _lw(1.15),
            "xtick.labelsize": _fs(12.0),
            "ytick.labelsize": _fs(12.0),
            "xtick.major.width": _lw(1.05),
            "ytick.major.width": _lw(1.05),
            "xtick.major.size": _ms(4.0),
            "ytick.major.size": _ms(4.0),
            "grid.color": "#D9D9D9",
            "grid.linewidth": _lw(0.8),
            "grid.alpha": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": _fs(11.8),
        }
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5A5A5A")
    ax.spines["bottom"].set_color("#5A5A5A")
    ax.tick_params(colors="#333333")


def draw_grouped_points(
    ax,
    df: pd.DataFrame,
    ycol: str,
    ylim: tuple[float, float],
    title: str,
    yref: float | None = None,
    show_values: bool = True,
):
    centers = np.arange(len(POPS_DEFAULT), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(METHODS))

    if yref is not None:
        ax.axhline(
            yref, color="#6E6E6E", linewidth=_lw(1.2), linestyle=(0, (4, 2)), zorder=1
        )

    for boundary in centers[:-1] + 0.5:
        ax.axvline(boundary, color="#EBEBEB", linewidth=_lw(1.0), zorder=0)

    for j, method in enumerate(METHODS):
        method_sub = df[df["method"] == method].copy()
        xs = centers + offsets[j]
        ys = []
        for pop in POPS_DEFAULT:
            row = method_sub[method_sub["pop"] == pop]
            ys.append(float(row[ycol].iloc[0]) if not row.empty else np.nan)
        ax.plot(
            xs, ys, color=METHOD_COLOR[method], linewidth=_lw(1.5), alpha=0.35, zorder=2
        )
        ax.scatter(
            xs,
            ys,
            s=_scatter_size(76),
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            edgecolor="white",
            linewidth=_lw(1.0),
            zorder=3,
        )

        if show_values:
            for xpos, y, pop in zip(xs, ys, POPS_DEFAULT):
                if not np.isfinite(y):
                    continue
                y_text = y + 0.012
                va = "bottom"
                if (ycol, pop, method) in {
                    ("auroc", "EUR", "covldsc"),
                    ("aupr", "EUR", "covldsc"),
                    ("aupr", "SAS", "covldsc"),
                    ("auroc", "SAS", "sumher_ldak"),
                }:
                    y_text = y - 0.014
                    va = "top"
                ax.text(
                    xpos,
                    y_text,
                    f"{y:.3f}",
                    ha="center",
                    va=va,
                    fontsize=_fs(9.4),
                    color="#222222",
                )

    ax.set_xlim(-0.5, len(centers) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(centers)
    ax.set_xticklabels(POPS_DEFAULT)
    ax.set_title(title, pad=6)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    style_axis(ax)
