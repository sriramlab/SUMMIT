"""Plot coding for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import common

POPS = ["EUR", "SAS", "AFR"]


METHODS = ["summit", "covldsc", "sumher_ldak"]


ALPHA = 0.05


def draw_enrichment_recovery(fig: plt.Figure, spec, panels: common.SimPanels):
    inner = spec.subgridspec(1, 2, wspace=0.16)
    ax_roc = fig.add_subplot(inner[0, 0])
    ax_pr = fig.add_subplot(inner[0, 1], sharey=ax_roc)
    yvals = panels.panel_b[["auroc", "aupr"]].to_numpy(dtype=float).ravel()
    ylims = (
        max(0.45, float(np.nanmin(yvals)) - 0.04),
        min(0.90, float(np.nanmax(yvals)) + 0.05),
    )
    common.sim.draw_grouped_points(
        ax_roc, panels.panel_b, "auroc", ylims, "AUROC", yref=0.5, show_values=True
    )
    common.sim.draw_grouped_points(
        ax_pr, panels.panel_b, "aupr", ylims, "AUPR", yref=0.5, show_values=True
    )
    ax_roc.set_ylabel("Mean score")
    plt.setp(ax_pr.get_yticklabels(), visible=False)
    ax_pr.spines["left"].set_visible(False)
    return [ax_roc, ax_pr]


def draw_calibration_panel(
    ax,
    df: pd.DataFrame,
    ylim: tuple[float, float],
    ylabel: str,
    show_counts: bool = True,
) -> None:
    centers = np.arange(len(POPS), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(METHODS))

    ax.axhline(
        ALPHA,
        color="#6E6E6E",
        linewidth=common._lw(1.2),
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    for boundary in centers[:-1] + 0.5:
        ax.axvline(boundary, color="#EBEBEB", linewidth=common._lw(1.0), zorder=0)

    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for j, method in enumerate(METHODS):
        sub = df[df["method"].astype(str) == method].copy()
        xs = centers + offsets[j]
        ys = []
        lo = []
        hi = []
        ns = []
        for pop in POPS:
            row = sub[sub["pop"].astype(str) == pop].iloc[0]
            ys.append(float(row["fpr"]))
            lo.append(float(row["fpr_lo"]))
            hi.append(float(row["fpr_hi"]))
            ns.append(int(row["N_valid"]))
        ys_arr = np.asarray(ys, dtype=float)
        yerr = np.vstack([ys_arr - np.asarray(lo), np.asarray(hi) - ys_arr])

        ax.errorbar(
            xs,
            ys_arr,
            yerr=yerr,
            fmt=common.sim.METHOD_MARKER[method],
            markersize=common._ms(8.0),
            linewidth=common._lw(2.0),
            capsize=common._ms(3.8),
            color=common.sim.METHOD_COLOR[method],
            markerfacecolor=common.sim.METHOD_COLOR[method],
            markeredgecolor="white",
            markeredgewidth=common._lw(1.0),
            zorder=3,
        )
        ax.plot(
            xs,
            ys_arr,
            color=common.sim.METHOD_COLOR[method],
            linewidth=common._lw(1.5),
            alpha=0.35,
            zorder=2,
        )

        if show_counts:
            for xpos, n in zip(xs, ns):
                ax.text(
                    xpos,
                    1.035,
                    f"n={n}",
                    transform=trans,
                    ha="center",
                    va="bottom",
                    fontsize=common._fs(9.2),
                    color=common.sim.METHOD_COLOR[method],
                    clip_on=False,
                )

    ax.set_xlim(-0.5, len(centers) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(centers)
    ax.set_xticklabels(POPS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    common.style_axis(ax)
