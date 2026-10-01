"""Plot layout for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import common

OUTDIR = Path("figs/main")

TAB20 = list(plt.get_cmap("tab20").colors)


PANEL_B_METHODS = ["summit", "covldsc", "sumher_ldak"]


STYLE_METHOD = {
    "summit": "summit",
    "covsumrhe": "summit",
    "covldsc": "covldsc",
    "sumher_ldak": "sumher_ldak",
    "sumher_ldak_20000": "sumher_ldak",
}


MAIN_POINT_SIZE = 92


ERROR_BAR_LW = 2.6


ERROR_CAP_HALF_WIDTH = 0.055


METHOD_COLOR = {
    "covsumrhe": TAB20[4],
    "rhe": TAB20[0],
    "covldsc": TAB20[2],
    "ldsc_20000": TAB20[10],
    "sumher_20000": TAB20[6],
    "sumher_ldak_20000": TAB20[8],
    "hdl": TAB20[19],
    "sumrhe": TAB20[5],
    "ldsc": TAB20[10],
    "sumher_gcta": TAB20[6],
    "sumher_ldak": TAB20[8],
}


METHOD_MARKER = {
    "covsumrhe": "o",
    "rhe": "^",
    "covldsc": "s",
    "ldsc_20000": "v",
    "sumher_20000": "P",
    "sumher_ldak_20000": "D",
    "hdl": "X",
    "sumrhe": "o",
    "ldsc": "v",
    "sumher_gcta": "P",
    "sumher_ldak": "D",
}


@dataclass(frozen=True)
class FigureLayout:
    out_stem: str
    panel_c_title: str
    panel_d_title: str
    draw_panel_c: Callable[[plt.Figure, object], list[plt.Axes]]
    draw_panel_d: Callable[[plt.Figure, object], list[plt.Axes]]


def style_method(method: str) -> str:
    return STYLE_METHOD.get(method, method)


def point_color(method: str):
    base = style_method(method)
    if base in common.sim.METHOD_COLOR:
        return common.sim.METHOD_COLOR[base]
    return METHOD_COLOR[method]


def point_marker(method: str) -> str:
    base = style_method(method)
    if base in common.sim.METHOD_MARKER:
        return common.sim.METHOD_MARKER[base]
    return METHOD_MARKER[method]


def scatter_main_point(
    ax: plt.Axes, xpos: float, ypos: float, method: str, zorder: int = 4
) -> None:
    ax.scatter(
        [xpos],
        [ypos],
        s=MAIN_POINT_SIZE,
        marker=point_marker(method),
        color=point_color(method),
        edgecolor="white",
        linewidth=0.9,
        zorder=zorder,
    )


def log_ticks(ylim: tuple[float, float]) -> list[float]:
    if ylim[1] > 1000:
        candidates = [1, 10, 100, 1000, 10000, 100000]
    elif ylim[1] > 100:
        candidates = [0.5, 1, 2, 5, 10, 100, 1000]
    else:
        candidates = [0.5, 1, 2, 5, 10, 50, 100]
    return [tick for tick in candidates if ylim[0] <= tick <= ylim[1]]


def style_log_ratio_axis(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    ax.axhline(1.0, color="#6E6E6E", linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ticks = log_ticks(ylim)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    common.style_axis(ax)


def draw_genomewide_panel_grouped(
    fig: plt.Figure, spec, panels: common.SimPanels
) -> list[plt.Axes]:
    del fig
    ax = plt.subplot(spec)
    centers = np.arange(len(common.POPS), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(PANEL_B_METHODS))
    ylim = common.sim.panel_a_limits(panels.panel_a)

    for j, method in enumerate(PANEL_B_METHODS):
        xs = []
        ys = []
        for i, pop in enumerate(common.POPS):
            row = panels.panel_a[
                (panels.panel_a["pop"] == pop) & (panels.panel_a["method"] == method)
            ]
            if row.empty:
                raise RuntimeError(f"Panel B missing {pop}/{method}")
            row0 = row.iloc[0]
            xpos = centers[i] + offsets[j]
            median = float(row0["median"])
            q25 = float(row0["q25"])
            q75 = float(row0["q75"])
            xs.append(xpos)
            ys.append(median)
            if q25 != q75:
                ax.vlines(
                    xpos,
                    q25,
                    q75,
                    color=point_color(method),
                    linewidth=ERROR_BAR_LW,
                    zorder=3,
                )
                ax.hlines(
                    [q25, q75],
                    xpos - ERROR_CAP_HALF_WIDTH,
                    xpos + ERROR_CAP_HALF_WIDTH,
                    color=point_color(method),
                    linewidth=ERROR_BAR_LW,
                    zorder=3,
                )
            scatter_main_point(ax, xpos, median, method, zorder=4)

        ax.plot(xs, ys, color=point_color(method), linewidth=1.5, alpha=0.35, zorder=2)

    for boundary in centers[:-1] + 0.5:
        ax.axvline(boundary, color="#EBEBEB", linewidth=1.0, zorder=0)

    style_log_ratio_axis(ax, ylim)
    ax.set_xlim(-0.5, len(common.POPS) - 0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels(common.POPS)
    ax.set_ylabel("Relative MSE\nvs SUMMIT")
    return [ax]


def render_figure(layout: FigureLayout) -> tuple[Path, Path]:
    common.set_style(profile="readable")
    panels = common.load_sim_panels(common.PACKAGE_ROOT)
    ld_df = common.load_ld_deficit(common.LD_DEFICIT_TSV)

    fig = plt.figure(figsize=(15.8, 11.2))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.32, 0.68],
        wspace=0.26,
        top=0.89,
        bottom=0.08,
        left=0.075,
        right=0.985,
    )

    axes_ld = common.draw_ld_vertical_stack(
        fig, outer[0, 0], ld_df, connect_within_pop=True
    )
    sim_grid = outer[0, 1].subgridspec(
        3, 6, height_ratios=[1.0, 1.0, 1.0], hspace=0.54, wspace=0.43
    )
    axes_b = draw_genomewide_panel_grouped(fig, sim_grid[0, 0:6], panels)
    axes_c = layout.draw_panel_c(fig, sim_grid[1, 0:6])
    axes_d = layout.draw_panel_d(fig, sim_grid[2, 0:6])

    fig.legend(
        handles=common.ld_legend_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.205, 0.985),
        columnspacing=0.8,
        handletextpad=0.45,
    )
    fig.legend(
        handles=common.method_legend_handles(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.69, 0.985),
        columnspacing=1.35,
        handletextpad=0.5,
    )

    common.add_panel_label(fig, axes_ld, "A", dx=-0.045, dy=0.008)
    common.add_panel_label(fig, axes_b, "B", dx=-0.055, dy=0.008)
    common.add_panel_label(fig, axes_c, "C", dx=-0.055, dy=0.008)
    common.add_panel_label(fig, axes_d, "D", dx=-0.055, dy=0.008)

    outprefix = OUTDIR / layout.out_stem
    common.save_figure(fig, outprefix)
    plt.close(fig)
    return outprefix.with_suffix(".pdf"), outprefix.with_suffix(".png")
