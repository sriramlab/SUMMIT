"""Plot enrichment for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cov_style as covplot
import h2_style as h2plot


def load_enrichment(tag):
    return pd.read_csv("data/external/enrichment_values.csv")


def load_enrichment_jackknife(tag):
    return pd.read_csv("data/external/enrichment_jackknife.tsv", sep="\t")


XMAX_ENRICH = 12.0


COHORTS = ["EUR_300k", "SAS", "AFR"]


METHODS = ["summit", "covldsc"]


COHORT_LABEL = {"EUR_300k": "EUR", "SAS": "SAS", "AFR": "AFR"}


METHOD_LABEL = {"summit": "SUMMIT", "covldsc": "cov-LDSC"}


SELECTED_PLUS_QTL = covplot.ANNOTATIONS_SELECTED_PLUS_QTL


def h2_panel_limits(sub: pd.DataFrame) -> tuple[float, float]:
    vals = pd.concat([sub["ukb_h2"], sub["external_h2"]], ignore_index=True)
    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return -0.05, 1.0
    lo = float(vals.min())
    hi = float(vals.max())
    span = max(hi - lo, 0.25)
    lo -= 0.08 * span
    hi += 0.08 * span
    hi = min(hi, h2plot.MAX_AXIS_H2)
    if lo > 0:
        lo = 0.0
    return lo, hi


def split_visible_h2(
    pts: pd.DataFrame, lo: float, hi: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible = pts["ukb_h2"].between(lo, hi) & pts["external_h2"].between(lo, hi)
    return pts[visible].copy(), pts[~visible].copy()


def add_offscale_h2(ax: plt.Axes, pts: pd.DataFrame, lo: float, hi: float, color: str):
    handle = None
    if pts.empty:
        return handle
    span = hi - lo
    edge = 0.025 * span
    for row in pts.itertuples(index=False):
        x = min(max(float(row.ukb_h2), lo + edge), hi - edge)
        y = min(max(float(row.external_h2), lo + edge), hi - edge)
        if float(row.external_h2) > hi:
            marker = "^"
        elif float(row.external_h2) < lo:
            marker = "v"
        elif float(row.ukb_h2) > hi:
            marker = ">"
        else:
            marker = "<"
        handle = ax.scatter(
            [x],
            [y],
            marker=marker,
            s=82,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
    return handle


def draw_enrichment_median(ax: plt.Axes, med: float, y: float, color: str) -> None:
    if not np.isfinite(med):
        return
    xpos = min(max(med, 0.0), XMAX_ENRICH)
    ax.plot(xpos, y, marker="D", ms=6.8, color="white", mec="0.22", mew=0.9, zorder=6)
    ax.plot(xpos, y, marker="D", ms=4.8, color=color, mec="white", mew=0.3, zorder=7)


def draw_enrichment_jackknife_ci(
    ax: plt.Axes, ci_lo: float, ci_hi: float, y: float, color: str
) -> None:
    lo = max(ci_lo, 0.0)
    hi = min(ci_hi, XMAX_ENRICH)
    if np.isfinite(ci_lo) and np.isfinite(ci_hi) and hi >= 0.0 and lo <= XMAX_ENRICH:
        ax.plot([lo, hi], [y, y], color=color, lw=1.25, alpha=0.93, zorder=4)
    if np.isfinite(ci_lo) and ci_lo < 0.0:
        ax.scatter(
            [0.0],
            [y],
            marker="<",
            s=18,
            color=color,
            alpha=0.93,
            linewidths=0,
            zorder=5,
        )
    if np.isfinite(ci_hi) and ci_hi > XMAX_ENRICH:
        ax.scatter(
            [XMAX_ENRICH],
            [y],
            marker=">",
            s=18,
            color=color,
            alpha=0.93,
            linewidths=0,
            zorder=5,
        )
