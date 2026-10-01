"""Plot normalization for the SUMMIT manuscript."""
from __future__ import annotations

from matplotlib.lines import Line2D
import numpy as np
import common
import architecture as v3


def method_handles():
    return [
        Line2D(
            [0],
            [0],
            marker=v3.v2.signed_method_marker(method),
            linestyle="none",
            markersize=common._ms(7.0),
            markerfacecolor=v3.v2.signed_method_color(method),
            markeredgecolor="white",
            label=v3.SIGNED_LABEL[method],
        )
        for method in v3.SIGNED_METHODS
    ]


def panel_setup(ax, pop, records):
    counts = [
        len(records[records["pop"].eq(pop) & records["method"].eq(method)])
        for method in v3.SIGNED_METHODS
    ]
    ax.set_title(
        pop + "\n" + "/".join(map(str, counts)) + " valid replicates",
        fontsize=common._fs(13.0),
        fontweight="bold",
        pad=7,
    )
    ax.grid(True, alpha=0.27)
    common.style_axis(ax)


def scatter_methods(ax, sub, xcol, ycol):
    for method in v3.SIGNED_METHODS:
        values = sub[sub["method"].eq(method)]
        ax.scatter(
            values[xcol],
            values[ycol],
            s=common._scatter_size(24),
            marker=v3.v2.signed_method_marker(method),
            facecolor=v3.v2.signed_method_color(method),
            edgecolor="white",
            linewidth=common._lw(0.35),
            alpha=0.52,
            zorder=3,
        )


def draw_direct(axes, records):
    xlim = (0.0, 1.58)
    ylim = (-0.54, 0.88)
    xline = np.linspace(*xlim, 300)
    for idx, (ax, pop) in enumerate(zip(axes, v3.POPS)):
        sub = records[records["pop"].eq(pop)]
        ax.plot(
            xline,
            v3.TRUE_TOTAL_RG * xline,
            color="#666666",
            linestyle=(0, (4, 2)),
            linewidth=common._lw(1.3),
            zorder=1,
        )
        scatter_methods(ax, sub, "denominator", "gamma_g")
        ax.scatter(
            [np.sqrt(v3.TRUE_TOTAL_H2_TRAIT1 * v3.TRUE_TOTAL_H2_TRAIT2)],
            [v3.TRUE_TOTAL_GENCOV],
            marker="*",
            s=common._scatter_size(120),
            facecolor="#111111",
            edgecolor="white",
            linewidth=common._lw(0.7),
            zorder=5,
        )
        ax.axhline(0.0, color="#B0B0B0", linewidth=common._lw(0.7), zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6])
        ax.set_yticks([-0.4, 0.0, 0.4, 0.8])
        panel_setup(ax, pop, records)
        ax.set_xlabel(r"Heritability scale, $\sqrt{\widehat h_1^2\widehat h_2^2}$")
        if idx == 0:
            ax.set_ylabel(r"Genetic covariance, $\widehat\gamma_g$")
    handles = method_handles() + [
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle=(0, (4, 2)),
            linewidth=common._lw(1.3),
            label=r"Correct $r_g$: $\widehat\gamma_g=0.3\widehat D$",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markersize=common._ms(9),
            markerfacecolor="#111111",
            markeredgecolor="white",
            label="True components",
        ),
    ]
    axes[1].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        fontsize=common._fs(10.2),
    )
