"""Plot total h2 for the SUMMIT manuscript."""
from __future__ import annotations

from matplotlib.ticker import MultipleLocator
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import enrichment as mainfig

FIG_DIR = Path("figs/supplementary")

H2_DIR = Path("data/external")

TAG = "without_mvp_hare"


def load_h2(include_mvp_hare: bool = False) -> pd.DataFrame:
    df = pd.read_csv(H2_DIR / "total_h2.tsv", sep="\t")
    if not include_mvp_hare:
        df = df[~df["source"].eq("MVP_R4_HARE")].copy()
    return df


def draw_h2_block(
    ax_grid, h2_df: pd.DataFrame
) -> tuple[list[plt.Axes], dict[str, Line2D]]:
    cohort_limits = {
        cohort: mainfig.h2_panel_limits(h2_df[h2_df["cohort"].eq(cohort)])
        for cohort in mainfig.COHORTS
    }
    for cohort in ("EUR_300k", "SAS"):
        sub = h2_df[h2_df["cohort"].eq(cohort)]
        vals = pd.to_numeric(
            pd.concat([sub["ukb_h2"], sub["external_h2"]]), errors="coerce"
        )
        if vals.dropna().min() >= 0:
            _lo, hi = cohort_limits[cohort]
            cohort_limits[cohort] = (0.0, hi)
    axes = []
    legend_items: dict[str, Line2D] = {}
    for r, method in enumerate(mainfig.METHODS):
        for c, cohort in enumerate(mainfig.COHORTS):
            ax = ax_grid[r, c]
            sub = h2_df[h2_df["method"].eq(method) & h2_df["cohort"].eq(cohort)]
            lo, hi = cohort_limits[cohort]
            ax.plot(
                [lo, hi],
                [lo, hi],
                color="#5f6368",
                lw=1.05,
                ls="--",
                alpha=0.78,
                zorder=1,
            )
            ax.axhline(0, color="#d1d5db", lw=0.75, zorder=0)
            ax.axvline(0, color="#d1d5db", lw=0.75, zorder=0)
            for source in mainfig.h2plot.SOURCE_ORDER:
                pts = sub[sub["source_label"].astype(str).eq(source)]
                if pts.empty:
                    continue
                label = mainfig.h2plot.SOURCE_LABEL.get(source, source)
                color = mainfig.h2plot.SOURCE_COLOR.get(source, "#666666")
                legend_items.setdefault(
                    label,
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=0.35,
                        label=label,
                        markersize=6.2,
                        linestyle="None",
                        alpha=0.62,
                    ),
                )
                pts_in, pts_off = mainfig.split_visible_h2(pts, lo, hi)
                if not pts_in.empty:
                    ax.errorbar(
                        pts_in["ukb_h2"],
                        pts_in["external_h2"],
                        xerr=pts_in["ukb_h2_se"],
                        yerr=pts_in["external_h2_se"],
                        fmt="o",
                        ms=5.2,
                        lw=0.55,
                        elinewidth=0.7,
                        capsize=1.7,
                        capthick=0.7,
                        color=color,
                        ecolor=color,
                        alpha=0.62,
                        markeredgecolor="white",
                        markeredgewidth=0.35,
                        zorder=3,
                    )
                mainfig.add_offscale_h2(ax, pts_off, lo, hi, color)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MultipleLocator(0.50))
            ax.yaxis.set_major_locator(MultipleLocator(0.50))
            ax.minorticks_off()
            ax.grid(True, color="#9ca3af", alpha=0.22, lw=0.45)
            if r == 0:
                ax.set_title(mainfig.COHORT_LABEL[cohort], fontsize=15, pad=8)
            if c == 0:
                ax.set_ylabel(
                    f"{mainfig.METHOD_LABEL[method]}\nexternal $h^2$", fontsize=12.5
                )
            else:
                ax.set_ylabel("external $h^2$", fontsize=12.5)
            ax.set_xlabel("UKB $h^2$", fontsize=12.5)
            ax.tick_params(labelsize=11)
            axes.append(ax)
    return axes, legend_items


def build_figure(include_mvp_hare: bool = False) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 11.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 9.5,
        }
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    h2_df = load_h2(include_mvp_hare)

    fig, ax_grid = plt.subplots(2, 3, figsize=(11.8, 7.4), sharex=False, sharey=False)
    _, legend_items = draw_h2_block(ax_grid, h2_df)
    fig.legend(
        legend_items.values(),
        legend_items.keys(),
        frameon=False,
        fontsize=9.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 1])

    stem = (
        "external_total_h2_supplement_with_mvp_hare"
        if include_mvp_hare
        else "fig_s43_external_heritability"
    )
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=260)
    plt.close(fig)
    return [pdf, png]


if __name__ == "__main__":
    build_figure(False)
