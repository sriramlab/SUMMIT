"""Plot population ld for the SUMMIT manuscript."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FIXED_GROUP_COLORS = {
    "GW": "#55A868",  # green
    "CHR": "#4C72B0",  # blue
    "20Mb": "#C44E52",  # red
    "2Mb": "#8172B2",  # purple
    "50Mb": "#DD8452",  # orange
}


def set_plot_style() -> None:
    """Shared style settings for both figures."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fixed_group_color_map() -> Dict[str, tuple]:
    """Return stable colors for LD-score estimator/window labels."""
    return dict(FIXED_GROUP_COLORS)


def _complete_panel_df(
    summary_case: pd.DataFrame, pop: str, group_order: List[str]
) -> pd.DataFrame:
    """
    Force identical x categories across panels.
    Missing groups get mean=0 and se=nan, present=False.
    """
    sub = summary_case[summary_case["pop"] == pop][["group", "mean", "se"]].copy()
    sub["group"] = sub["group"].astype(str)

    present = set(sub["group"].tolist())
    rows = []
    for g in group_order:
        if g in present:
            r = sub[sub["group"] == g].iloc[0]
            rows.append(
                dict(group=g, mean=float(r["mean"]), se=float(r["se"]), present=True)
            )
        else:
            rows.append(dict(group=g, mean=0.0, se=float("nan"), present=False))

    out = pd.DataFrame(rows)
    out["group"] = pd.Categorical(out["group"], categories=group_order, ordered=True)
    return out


def _set_xticks_and_labels(ax, labels: List[str]) -> None:
    """Always set both ticks and tick labels to avoid FixedLocator mismatch errors."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, ha="center")


def _grid_group_order(summ_nocov: pd.DataFrame, summ_pc: pd.DataFrame) -> List[str]:
    """Order the plotted groups by no-covariate mean LD score."""
    tmp = summ_nocov.copy()
    tmp["group"] = tmp["group"].astype(str)
    group_order = (
        tmp.groupby("group", sort=False)["mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Ensure PC groups are also included defensively.
    for g in sorted(set(summ_pc["group"].astype(str).unique())):
        if g not in group_order:
            group_order.append(g)

    # Display the selected window sizes.
    return [g for g in group_order if g != "50Mb"]


def plot_grid_2xK(
    summ_nocov: pd.DataFrame,
    summ_pc: pd.DataFrame,
    pops: List[str],
    with_se: bool,
    outpath: Path,
) -> None:
    set_plot_style()

    group_order = _grid_group_order(summ_nocov, summ_pc)
    color_map = fixed_group_color_map()

    # Fallback colors for unexpected groups.
    fallback_palette = sns.color_palette("muted", n_colors=max(1, len(group_order)))
    for i, g in enumerate(group_order):
        color_map.setdefault(g, fallback_palette[i % len(fallback_palette)])

    K = len(pops)
    fig, axes = plt.subplots(
        2,
        K,
        figsize=(3.5 * K, 6.0),
        sharey=False,
        sharex=False,
    )
    if K == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    cases = [("nocov", "No covariates"), ("pc", "PC-adjusted")]

    for r, (case_key, case_label) in enumerate(cases):
        for c, pop in enumerate(pops):
            ax = axes[r, c]

            summary = summ_nocov if case_key == "nocov" else summ_pc
            dfp = _complete_panel_df(
                summary[summary["case"] == case_key], pop, group_order
            )
            x_order = group_order
            bar_width = 0.8

            sns.barplot(
                data=dfp,
                x="group",
                y="mean",
                order=x_order,
                ax=ax,
                errorbar=None,
                palette=color_map,
                edgecolor="0.2",
                linewidth=0.8,
                width=bar_width,
            )

            if with_se:
                lk = {
                    str(rw.group): (float(rw.mean), float(rw.se))
                    for rw in dfp.itertuples(index=False)
                }
                for rect, grp in zip(ax.patches, x_order):
                    m, s = lk.get(grp, (np.nan, np.nan))
                    if not np.isfinite(s) or s <= 0:
                        continue
                    x_center = rect.get_x() + rect.get_width() / 2.0
                    ax.errorbar(
                        x_center,
                        m,
                        yerr=s,
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.05,
                        capsize=3.0,
                        capthick=1.05,
                        zorder=10,
                        clip_on=False,
                    )

            ax.set_title(f"{pop}  |  {case_label}", pad=6.0)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=0)
            _set_xticks_and_labels(ax, x_order)

            ymax = float(dfp["mean"].max()) if len(dfp) else 1.0
            if not np.isfinite(ymax) or ymax <= 0:
                ymax = 1.0
            ax.set_ylim(0.0, ymax * 1.18)

            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            ax.grid(True, axis="y", alpha=0.35)
            ax.grid(False, axis="x")

    fig.text(0.5, 0.05, "Window size", ha="center", va="center", fontsize=16)
    fig.text(
        0.05,
        0.5,
        "Mean LD score (per SNP)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    fig.tight_layout(rect=(0.05, 0.045, 0.995, 0.99))

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    set_plot_style()
    plot_grid_2xK(
        pd.read_csv("data/ld/summary_nocov.csv"),
        pd.read_csv("data/ld/summary_pc.csv"),
        ["EUR", "SAS", "AFR"],
        False,
        Path("figs/supplementary/fig_s02_ld_scores.pdf"),
    )
