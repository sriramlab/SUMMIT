#!/usr/bin/env python3
"""
Build a manuscript-style 3-panel total h2 figure.

Panel A:
  Total h2 across 20 representative traits and 4 populations for a reduced
  set of methods (default: SUMMIT, cov-LDSC, SumHer-LDAK).

Panel B:
  Estimated h2 versus WGS h2 for the MAF-LD 24-bin annotation, using all
  traits with WGS reference estimates.

Panel C:
  Pairwise cross-population Wald z-statistics for total h2 differences using the
  full curated pop-difference trait set.
"""

import argparse
import math
import os
from itertools import combinations
from statistics import NormalDist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
PASTEL1 = [to_hex(c) for c in colormaps["Pastel1"].colors]

POPS = ["EUR_300k", "EUR", "SAS", "AFR"]
POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "SAS": "SAS",
    "AFR": "AFR",
}
POP_COLOR = {
    "EUR": TAB20[0],
    "EUR_300k": TAB20[1],
    "SAS": TAB20[4],
    "AFR": TAB20[6],
}

METHOD_LABEL = {
    "sumrhe": "SUMMIT",
    "covldsc": "cov-LDSC",
    "sumher_ldak": "SumHer-LDAK",
}
METHOD_COLOR = {
    "sumrhe": TAB20[4],
    "covldsc": TAB20[2],
    "sumher_ldak": TAB20[8],
}
METHOD_MARKER = {
    "sumrhe": "o",
    "covldsc": "s",
    "sumher_ldak": "D",
}

NORM = NormalDist()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outs-base",
        default="data/real_h2",
        help="Base directory with per-pop MAF-LD outputs.",
    )
    ap.add_argument(
        "--csv-name",
        default="estimates.csv",
        help="Per-pop CSV filename for MAF-LD total h2 outputs.",
    )
    ap.add_argument(
        "--phen-list",
        default="data/traits/population_comparison_traits.txt",
        help="Representative phenotype list used for panel A.",
    )
    ap.add_argument(
        "--popdiff-phen-list",
        default="data/traits/trait_categories.tsv",
        help="Curated phenotype list used for panel C.",
    )
    ap.add_argument(
        "--wgs-phen-list",
        default="data/traits/wgs_heritability.csv",
        help="Phenotype reference file with WGS h2 for panel B.",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=24,
        help="Number of MAF-LD bins to show (default: 24).",
    )
    ap.add_argument(
        "--methods",
        default="sumrhe,covldsc,sumher_ldak",
        help="Comma-separated method subset to show.",
    )
    ap.add_argument(
        "--panel-a-sort",
        choices=["grouped", "global"],
        default="grouped",
        help="Sort panel A by grouped phenotype pairs or globally by SUMMIT EUR(300k) h2.",
    )
    ap.add_argument(
        "--shade-negative",
        action="store_true",
        help="Shade the negative h2 region in panel A behind the bars.",
    )
    ap.add_argument(
        "--show-comments",
        action="store_true",
        help="Show small explanatory comments in the figure.",
    )
    circle_group = ap.add_mutually_exclusive_group()
    circle_group.add_argument(
        "--circle-summit-bonf-hits",
        action="store_true",
        default=True,
        help="Circle Bonferroni-significant SUMMIT hits in panel C (default).",
    )
    circle_group.add_argument(
        "--no-circle-summit-bonf-hits",
        dest="circle_summit_bonf_hits",
        action="store_false",
        help="Do not circle Bonferroni-significant SUMMIT hits in panel C.",
    )
    annotate_group = ap.add_mutually_exclusive_group()
    annotate_group.add_argument(
        "--annotate-summit-bonf-hits",
        action="store_true",
        default=True,
        help="Annotate Bonferroni-significant SUMMIT hits in panel C with phenotype acronyms (default).",
    )
    annotate_group.add_argument(
        "--no-annotate-summit-bonf-hits",
        dest="annotate_summit_bonf_hits",
        action="store_false",
        help="Do not annotate Bonferroni-significant SUMMIT hits in panel C.",
    )
    panel_c_group = ap.add_mutually_exclusive_group()
    panel_c_group.add_argument(
        "--panel-c-summit-only",
        action="store_true",
        default=True,
        help="Restrict panel C to SUMMIT only while leaving panels A/B unchanged (default).",
    )
    panel_c_group.add_argument(
        "--panel-c-all-methods",
        dest="panel_c_summit_only",
        action="store_false",
        help="Show all selected methods in panel C.",
    )
    ap.add_argument(
        "--outfile",
        default="figs/main/fig04_real_trait_heritability.pdf",
        help="Output PDF path.",
    )
    ap.add_argument(
        "--preview-png",
        default="",
        help="Optional PNG path. By default, saves a PNG with the same prefix as --outfile.",
    )
    return ap.parse_args()


def load_phenotypes(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["phen", "code", "abbr", "class"],
    )
    df["phen"] = df["phen"].astype(str)
    df["abbr"] = df["abbr"].astype(str)
    df["class"] = df["class"].astype(str)
    return df


def build_class_palette(classes):
    palette = {}
    uniq = list(dict.fromkeys([str(c) for c in classes]))
    for idx, cls in enumerate(uniq):
        palette[cls] = PASTEL1[idx % len(PASTEL1)]
    return palette


def default_png_path(outfile: str) -> str:
    root, _ext = os.path.splitext(outfile)
    if not root:
        root = outfile
    return f"{root}.png"


def load_mafld_table(outs_base: str, csv_name: str) -> pd.DataFrame:
    dfs = []
    for pop in POPS:
        path = os.path.join(outs_base, pop, csv_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV for {pop}: {path}")
        df = pd.read_csv(path)
        df["pop"] = pop
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    need = ["pop", "phen", "method", "num_bins", "h2", "h2_se", "window"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    df["phen"] = df["phen"].astype(str)
    df["method"] = df["method"].astype(str)
    df["pop"] = df["pop"].astype(str)
    df["num_bins"] = pd.to_numeric(df["num_bins"], errors="coerce")
    df["h2"] = pd.to_numeric(df["h2"], errors="coerce")
    df["h2_se"] = pd.to_numeric(df["h2_se"], errors="coerce")
    df["window_sort"] = pd.to_numeric(df["window"], errors="coerce").fillna(-1)
    return df


def select_largest_window(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    return (
        df.sort_values("window_sort")
        .groupby(group_cols, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def load_wgs_reference(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", comment="#")
    need = ["Phenotype", "Acronym", "h2_wgs", "h2_wgs_se"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"WGS phenotype table missing columns: {missing}")

    out = df[need].copy()
    out["Phenotype"] = out["Phenotype"].astype(str)
    out["Acronym"] = out["Acronym"].astype(str)
    out["h2_wgs"] = pd.to_numeric(out["h2_wgs"], errors="coerce")
    out["h2_wgs_se"] = pd.to_numeric(out["h2_wgs_se"], errors="coerce")
    return out


def prepare_panel_subset(
    df_all: pd.DataFrame, phen_meta: pd.DataFrame, bins_keep: int, methods
):
    keep = phen_meta["phen"].tolist()
    df = df_all[df_all["phen"].isin(keep) & df_all["method"].isin(methods)].copy()
    df = df[df["num_bins"] == int(bins_keep)].copy()
    df = select_largest_window(df, ["pop", "phen", "method", "num_bins"])
    return df.sort_values(["method", "phen", "pop"]).reset_index(drop=True)


def prepare_panel_b(
    df_all: pd.DataFrame, ref_df: pd.DataFrame, bins_keep: int, methods
):
    df = df_all[
        (df_all["num_bins"] == int(bins_keep)) & (df_all["method"].isin(methods))
    ].copy()
    df = select_largest_window(df, ["pop", "phen", "method", "num_bins"])

    ref = ref_df.rename(
        columns={
            "Phenotype": "phen",
            "Acronym": "abbr",
            "h2_wgs": "ref_h2",
            "h2_wgs_se": "ref_h2_se",
        }
    )
    ref["phen"] = ref["phen"].astype(str)
    merged = df.merge(ref, how="inner", on="phen")
    return merged.sort_values(["pop", "method", "phen"]).reset_index(drop=True)


def prepare_panel_c(df_c_source: pd.DataFrame, phen_meta: pd.DataFrame, methods):
    phen_order = phen_meta["phen"].tolist()
    phen2abbr = dict(zip(phen_meta["phen"], phen_meta["abbr"]))

    rows = []
    for pop1, pop2 in combinations(POPS, 2):
        for phen in phen_order:
            for method in methods:
                left = df_c_source[
                    (df_c_source["pop"] == pop1)
                    & (df_c_source["phen"] == phen)
                    & (df_c_source["method"] == method)
                ]
                right = df_c_source[
                    (df_c_source["pop"] == pop2)
                    & (df_c_source["phen"] == phen)
                    & (df_c_source["method"] == method)
                ]
                if left.empty or right.empty:
                    continue

                h1 = float(left["h2"].iloc[0])
                h2 = float(right["h2"].iloc[0])
                s1 = float(left["h2_se"].iloc[0])
                s2 = float(right["h2_se"].iloc[0])
                if not (
                    np.isfinite(h1)
                    and np.isfinite(h2)
                    and np.isfinite(s1)
                    and np.isfinite(s2)
                ):
                    continue
                if s1 <= 0 or s2 <= 0:
                    continue

                diff = h1 - h2
                se_diff = math.sqrt(s1**2 + s2**2)
                t_stat = diff / se_diff
                p_value = 2.0 * (1.0 - NORM.cdf(abs(t_stat)))

                rows.append(
                    {
                        "pop_pair": f"{pop1} vs {pop2}",
                        "method": method,
                        "phen": phen,
                        "phen_abbr": phen2abbr[phen],
                        "t_stat": t_stat,
                        "p_value": p_value,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows available for panel C.")

    df["reject_nominal"] = df["p_value"] < 0.05
    df["reject_bonf"] = False
    for method in methods:
        idx = df["method"] == method
        m = int(idx.sum())
        if m <= 0:
            continue
        df.loc[idx, "reject_bonf"] = df.loc[idx, "p_value"] < (0.05 / m)
    return df


def order_panel_a_phenotypes(
    phen_meta: pd.DataFrame, df_a: pd.DataFrame, sort_mode: str
):
    score_df = df_a[(df_a["pop"] == "EUR_300k") & (df_a["method"] == "sumrhe")][
        ["phen", "h2"]
    ].drop_duplicates("phen")
    score_map = dict(zip(score_df["phen"], score_df["h2"]))

    meta = phen_meta.copy()
    meta["score"] = meta["phen"].map(score_map).fillna(-np.inf)

    if sort_mode == "global":
        meta = meta.sort_values(
            ["score", "phen"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        return meta.drop(columns="score"), False

    blocks = []
    for start in range(0, len(meta), 2):
        block = meta.iloc[start : start + 2].copy()
        block = block.sort_values(
            ["score", "phen"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        block_score = float(block["score"].max()) if not block.empty else -np.inf
        blocks.append((block_score, block))

    ordered_blocks = [
        block for _, block in sorted(blocks, key=lambda x: x[0], reverse=True)
    ]
    ordered = (
        pd.concat(ordered_blocks, ignore_index=True)
        if ordered_blocks
        else meta.iloc[0:0].copy()
    )
    return ordered.drop(columns="score"), True


def pretty_pair_label(pair: str) -> str:
    left, right = [s.strip() for s in pair.split("vs")]
    return f"{POP_LABEL.get(left, left)}\nvs {POP_LABEL.get(right, right)}"


def add_panel_label(fig, ax, label: str, dx: float = -0.03, dy: float = 0.006):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        fontsize=19.8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_vertical_clip_marker(ax, x, y, lo, hi, color):
    span = hi - lo
    if y > hi:
        ax.scatter(
            [x],
            [hi - 0.03 * span],
            marker="^",
            s=44,
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=8,
            clip_on=False,
        )
    elif y < lo:
        ax.scatter(
            [x],
            [lo + 0.03 * span],
            marker="v",
            s=44,
            c=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=8,
            clip_on=False,
        )


def draw_clipped_errorbar(ax, x, y, se, lo, hi):
    if not np.isfinite(y) or not np.isfinite(se) or se <= 0:
        return
    y1 = max(lo, y - se)
    y2 = min(hi, y + se)
    if y2 < lo or y1 > hi:
        return
    ax.plot([x, x], [y1, y2], color="black", linewidth=0.9, zorder=7)
    cap_w = 0.08
    ax.plot([x - cap_w, x + cap_w], [y1, y1], color="black", linewidth=0.9, zorder=7)
    ax.plot([x - cap_w, x + cap_w], [y2, y2], color="black", linewidth=0.9, zorder=7)


def points_to_data_units(ax, *, x_points: float = 0.0, y_points: float = 0.0):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()

    dx = 0.0
    if x_points and bbox.width > 0 and fig_w > 0:
        dx = x_points * (x1 - x0) / (bbox.width * fig_w * 72.0)

    dy = 0.0
    if y_points and bbox.height > 0 and fig_h > 0:
        dy = y_points * (y1 - y0) / (bbox.height * fig_h * 72.0)

    return dx, dy


def spread_label_positions(targets, lo: float, hi: float, min_gap: float):
    vals = np.asarray(targets, dtype=float)
    if vals.size == 0:
        return vals.copy()
    if vals.size == 1:
        return np.clip(vals, lo, hi)

    span = max(0.0, hi - lo)
    gap = min(float(min_gap), span / (vals.size - 1)) if span > 0 else 0.0

    order = np.argsort(vals, kind="stable")
    placed = vals[order].copy()
    placed[0] = float(np.clip(placed[0], lo, hi))
    for idx in range(1, placed.size):
        placed[idx] = max(placed[idx], placed[idx - 1] + gap)

    overflow = placed[-1] - hi
    if overflow > 0:
        placed -= overflow
        for idx in range(placed.size - 2, -1, -1):
            placed[idx] = min(placed[idx], placed[idx + 1] - gap)

    underflow = lo - placed[0]
    if underflow > 0:
        placed += underflow
        for idx in range(1, placed.size):
            placed[idx] = max(placed[idx], placed[idx - 1] + gap)

    out = np.empty_like(placed)
    out[order] = placed
    return out


def add_spaced_hit_annotations(
    ax, point_xs, point_ys, labels, *, anchor_x: float, prefer_left: bool, color: str
):
    if len(labels) == 0:
        return

    point_xs = np.asarray(point_xs, dtype=float)
    point_ys = np.asarray(point_ys, dtype=float)
    labels = [str(label) for label in labels]
    y_lo, y_hi = ax.get_ylim()
    _xpad, ypad = points_to_data_units(ax, y_points=7.0)
    _xgap, min_gap = points_to_data_units(ax, y_points=11.0)

    label_ys = point_ys.copy()
    neg_mask = point_ys < 0
    pos_mask = ~neg_mask
    for mask, sign in ((neg_mask, -1.0), (pos_mask, 1.0)):
        if not np.any(mask):
            continue
        lower = y_lo + ypad
        upper = y_hi - ypad
        if sign < 0:
            upper = min(upper, -0.25 * ypad)
        else:
            lower = max(lower, 0.25 * ypad)
        targets = point_ys[mask] + sign * ypad
        label_ys[mask] = spread_label_positions(targets, lower, upper, min_gap)

    ha = "right" if prefer_left else "left"
    connector_rad = -0.12 if prefer_left else 0.12
    for px, py, ly, label in zip(point_xs, point_ys, label_ys, labels):
        ax.annotate(
            label,
            xy=(float(px), float(py)),
            xytext=(float(anchor_x), float(ly)),
            textcoords="data",
            fontsize=9.0,
            color=color,
            ha=ha,
            va="center",
            zorder=5,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.12),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.65,
                alpha=0.85,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=f"arc3,rad={connector_rad}",
            ),
        )


def plot_panel_a(
    fig,
    gs,
    df_a: pd.DataFrame,
    phen_meta: pd.DataFrame,
    methods,
    class_palette,
    *,
    show_bands: bool,
    shade_negative: bool,
    show_comments: bool,
):
    axes = []
    phen_order = phen_meta["phen"].tolist()
    phen_abbr = phen_meta["abbr"].tolist()

    x = np.arange(len(phen_order))
    n_pops = len(POPS)
    group_width = 0.72
    bar_w = group_width / n_pops
    y_lo, y_hi = -0.25, 1.00
    y_ticks = np.arange(-0.25, 1.01, 0.25)

    for row_idx, method in enumerate(methods):
        ax = fig.add_subplot(gs[row_idx, 0], sharex=axes[0] if axes else None)
        axes.append(ax)

        if show_bands:
            for start in range(0, len(phen_meta), 2):
                cls = phen_meta.iloc[start]["class"]
                ax.axvspan(
                    start - 0.5,
                    min(start + 1.5, len(phen_meta) - 0.5),
                    color=class_palette.get(cls, "#f0f0f0"),
                    alpha=0.12,
                    zorder=0,
                )

        if shade_negative:
            ax.axhspan(y_lo, 0.0, color="0.88", alpha=0.60, zorder=0.5)

        dmethod = df_a[df_a["method"] == method].copy()
        for pop_idx, pop in enumerate(POPS):
            dpop = dmethod[dmethod["pop"] == pop].copy().set_index("phen")
            offset = (pop_idx - (n_pops - 1) / 2.0) * bar_w

            for i, phen in enumerate(phen_order):
                if phen not in dpop.index:
                    continue
                row = dpop.loc[phen]
                h2 = row["h2"]
                se = row["h2_se"]
                if not np.isfinite(h2):
                    continue

                x0 = x[i] + offset
                h_clip = float(np.clip(h2, y_lo, y_hi))
                ax.bar(
                    x0,
                    h_clip,
                    width=bar_w * 0.85,
                    color=POP_COLOR[pop],
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
                draw_clipped_errorbar(ax, x0, float(h2), float(se), y_lo, y_hi)
                draw_vertical_clip_marker(ax, x0, float(h2), y_lo, y_hi, POP_COLOR[pop])

        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8, zorder=2)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(y_ticks)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
        ax.text(
            0.00,
            1.15,
            METHOD_LABEL[method],
            transform=ax.transAxes,
            fontsize=13.8,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if row_idx < len(methods) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(phen_abbr, rotation=0, fontsize=9.9)

    axes[len(axes) // 2].set_ylabel("Total h²", fontsize=13.2)
    add_panel_label(fig, axes[0], "A")

    pop_handles = [
        Patch(
            facecolor=POP_COLOR[p], edgecolor="black", linewidth=0.6, label=POP_LABEL[p]
        )
        for p in POPS
    ]
    axes[0].legend(
        handles=pop_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.28),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=1.1,
        columnspacing=1.1,
    )

    if show_comments and show_bands:
        axes[-1].text(
            1.01,
            -0.33,
            "Pastel bands mark phenotype classes.",
            transform=axes[-1].transAxes,
            ha="right",
            va="top",
            fontsize=9.7,
            color="0.35",
        )


def plot_panel_b(fig, gs, df_b: pd.DataFrame, methods, *, show_comments: bool):
    pop_axes = []
    x_lo, x_hi = -0.25, 1.25
    y_lo, y_hi = -0.25, 1.25
    x_span = x_hi - x_lo
    x_ticks = [0.0, 0.5, 1.0]
    y_ticks = np.arange(-0.25, 1.26, 0.25)
    x_dodge = {
        "covldsc": -0.006 * x_span,
        "sumher_ldak": 0.0,
        "sumrhe": 0.006 * x_span,
    }

    for idx, pop in enumerate(POPS):
        ax = fig.add_subplot(
            gs[idx // 2, idx % 2],
            sharex=pop_axes[0] if pop_axes else None,
            sharey=pop_axes[0] if pop_axes else None,
        )
        pop_axes.append(ax)
        sub = df_b[df_b["pop"] == pop].copy()
        plot_methods = [m for m in methods if m != "sumrhe"]
        if "sumrhe" in methods:
            plot_methods.append("sumrhe")

        ax.plot(
            [x_lo, x_hi],
            [x_lo, x_hi],
            linestyle="--",
            linewidth=1.2,
            color="0.45",
            zorder=1,
        )

        for method in plot_methods:
            dm = sub[sub["method"] == method].copy()
            if dm.empty:
                continue

            dm["x"] = pd.to_numeric(dm["ref_h2"], errors="coerce") + x_dodge.get(
                method, 0.0
            )
            dm["y"] = pd.to_numeric(dm["h2"], errors="coerce")
            dm["xse"] = pd.to_numeric(dm["ref_h2_se"], errors="coerce")
            dm["yse"] = pd.to_numeric(dm["h2_se"], errors="coerce")
            dm = dm.dropna(subset=["x", "y"])

            inside = (dm["y"] >= y_lo) & (dm["y"] <= y_hi)
            shown = dm[inside].copy()
            clipped = dm[~inside].copy()

            if not shown.empty:
                ax.errorbar(
                    shown["x"].to_numpy(),
                    shown["y"].to_numpy(),
                    xerr=shown["xse"].to_numpy(),
                    yerr=shown["yse"].to_numpy(),
                    fmt="none",
                    ecolor=METHOD_COLOR[method],
                    elinewidth=0.60 if method != "sumrhe" else 0.85,
                    capsize=1.8,
                    alpha=0.38 if method != "sumrhe" else 0.78,
                    zorder=2,
                    rasterized=True,
                )

                ax.scatter(
                    shown["x"].to_numpy(),
                    shown["y"].to_numpy(),
                    s=26 if method != "sumrhe" else 48,
                    marker=METHOD_MARKER[method],
                    c=METHOD_COLOR[method],
                    edgecolors="black" if method == "sumrhe" else "none",
                    linewidths=0.8 if method == "sumrhe" else 0.0,
                    alpha=0.86 if method == "sumrhe" else 0.52,
                    zorder=4 if method == "sumrhe" else 3,
                    rasterized=True,
                )

            for _, row in clipped.iterrows():
                draw_vertical_clip_marker(
                    ax,
                    float(row["x"]),
                    float(row["y"]),
                    y_lo,
                    y_hi,
                    METHOD_COLOR[method],
                )

        ax.set_title(POP_LABEL[pop], fontsize=12.7, pad=4)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{tick:.1f}" for tick in x_ticks])
        ax.set_yticks(y_ticks)
        ax.set_aspect("equal", adjustable="box", anchor="W")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx, ax in enumerate(pop_axes):
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=(idx % 2 == 0))
        ax.set_xlabel("WGS common h²", fontsize=12.1, labelpad=4)
    pop_axes[0].set_ylabel("Estimated h²", fontsize=12.1)
    pop_axes[2].set_ylabel("Estimated h²", fontsize=12.1)
    add_panel_label(fig, pop_axes[0], "B")

    if show_comments:
        pop_axes[1].text(
            1.02,
            1.12,
            "MAF-LD (24 bins)",
            transform=pop_axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            color="0.35",
        )


def plot_panel_c(
    fig,
    ax,
    df_c: pd.DataFrame,
    methods,
    *,
    circle_summit_hits: bool,
    annotate_summit_hits: bool,
):
    pop_pairs = [f"{a} vs {b}" for a, b in combinations(POPS, 2)]
    centers = np.arange(len(pop_pairs), dtype=float)
    if len(methods) == 1:
        box_w = 0.36
        offsets = [0.0]
    else:
        group_width = 0.82
        box_w = group_width / len(methods)
        offsets = [(j - (len(methods) - 1) / 2.0) * box_w for j in range(len(methods))]
    rng = np.random.default_rng(0)

    method_tbonf = {}
    for method in methods:
        idx = df_c["method"] == method
        m = int(idx.sum())
        method_tbonf[method] = NORM.inv_cdf(1.0 - (0.05 / max(1, m)) / 2.0)

    for j, method in enumerate(methods):
        positions = centers + offsets[j]
        data = []
        for pair in pop_pairs:
            vals = (
                df_c.loc[
                    (df_c["pop_pair"] == pair) & (df_c["method"] == method),
                    "t_stat",
                ]
                .astype(float)
                .to_numpy()
            )
            vals = vals[np.isfinite(vals)]
            data.append(vals)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_w * 0.88,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
            zorder=2,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(METHOD_COLOR[method])
            patch.set_alpha(0.28)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.95)
        for key in ["whiskers", "caps", "medians"]:
            for item in bp[key]:
                item.set_color("black")
                item.set_linewidth(0.95)

        for i, pair in enumerate(pop_pairs):
            g = (
                df_c.loc[(df_c["pop_pair"] == pair) & (df_c["method"] == method),]
                .copy()
                .reset_index(drop=True)
            )
            g["t_stat"] = pd.to_numeric(g["t_stat"], errors="coerce")
            g = g[np.isfinite(g["t_stat"])].reset_index(drop=True)
            vals = g["t_stat"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            xs = positions[i] + rng.normal(0.0, box_w * 0.12, size=vals.size)
            ax.scatter(
                xs,
                vals,
                s=20,
                c=METHOD_COLOR[method],
                alpha=0.55 if method != "sumrhe" else 0.72,
                edgecolors="black",
                linewidths=0.25,
                zorder=3,
                rasterized=True,
            )
            if method == "sumrhe":
                hit_mask = g["reject_bonf"].to_numpy()
                if circle_summit_hits and np.any(hit_mask):
                    ax.scatter(
                        xs[hit_mask],
                        vals[hit_mask],
                        s=68,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=4.5,
                    )
                if annotate_summit_hits and np.any(hit_mask):
                    hit_rows = g.loc[hit_mask, ["phen_abbr", "t_stat"]].reset_index(
                        drop=True
                    )
                    hit_xs = xs[hit_mask]
                    prefer_left = offsets[j] <= 0
                    if i == 0 and prefer_left:
                        prefer_left = False
                    elif i == len(pop_pairs) - 1 and not prefer_left:
                        prefer_left = True
                    anchor_x = positions[i] + (-1 if prefer_left else 1) * box_w * 0.72
                    add_spaced_hit_annotations(
                        ax,
                        hit_xs,
                        hit_rows["t_stat"].to_numpy(dtype=float),
                        hit_rows["phen_abbr"].tolist(),
                        anchor_x=anchor_x,
                        prefer_left=prefer_left,
                        color=METHOD_COLOR["sumrhe"],
                    )

    t_nom = NORM.inv_cdf(1.0 - 0.05 / 2.0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.75, zorder=1)
    ax.axhline(+t_nom, color="0.4", linestyle="--", linewidth=1.15, alpha=0.9, zorder=1)
    ax.axhline(-t_nom, color="0.4", linestyle="--", linewidth=1.15, alpha=0.9, zorder=1)
    for method in methods:
        t_b = method_tbonf[method]
        ax.axhline(
            +t_b,
            color=METHOD_COLOR[method],
            linestyle="--",
            linewidth=1.05,
            alpha=0.55,
            zorder=1,
        )
        ax.axhline(
            -t_b,
            color=METHOD_COLOR[method],
            linestyle="--",
            linewidth=1.05,
            alpha=0.55,
            zorder=1,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([pretty_pair_label(p) for p in pop_pairs], fontsize=9.9)
    ax.set_ylabel(r"$h^2$ difference Wald z-statistic", fontsize=12.7)
    ax.set_ylim(-8.2, 8.2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(fig, ax, "C")


def build_method_legend(ax, methods):
    ax.axis("off")
    handles = []
    for method in methods:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method],
                markerfacecolor=METHOD_COLOR[method],
                markeredgecolor="black" if method == "sumrhe" else METHOD_COLOR[method],
                markeredgewidth=0.8 if method == "sumrhe" else 0.0,
                markersize=9.4,
                linewidth=0,
                label=METHOD_LABEL[method],
            )
        )

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.45, 1.0),
        ncol=1,
        frameon=True,
        fontsize=9.9,
        title="Method",
        title_fontsize=9.9,
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.7,
        labelspacing=0.6,
    )


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise RuntimeError("No methods provided via --methods.")
    panel_c_methods = ["sumrhe"] if args.panel_c_summit_only else methods
    legend_methods = methods.copy()
    for method in panel_c_methods:
        if method not in legend_methods:
            legend_methods.append(method)

    phen_meta_a = load_phenotypes(args.phen_list)
    phen_meta_c = load_phenotypes(args.popdiff_phen_list)
    df_all = load_mafld_table(args.outs_base, args.csv_name)
    ref_df = load_wgs_reference(args.wgs_phen_list)

    df_a_source = prepare_panel_subset(df_all, phen_meta_a, args.bins, methods)
    phen_meta_a, panel_a_show_bands = order_panel_a_phenotypes(
        phen_meta_a, df_a_source, args.panel_a_sort
    )
    class_palette_a = build_class_palette(phen_meta_a["class"].tolist())
    df_a = prepare_panel_subset(df_all, phen_meta_a, args.bins, methods)
    df_c_source = prepare_panel_subset(df_all, phen_meta_c, args.bins, panel_c_methods)
    df_b = prepare_panel_b(df_all, ref_df, args.bins, methods)
    df_c = prepare_panel_c(df_c_source, phen_meta_c, panel_c_methods)

    sns.set_theme(
        style="white",
        context="talk",
        rc={
            "axes.labelsize": 12.1,
            "axes.titlesize": 13.2,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    fig = plt.figure(figsize=(16.2, 14.3))
    outer = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.08, 1.35],
        width_ratios=[1.15, 0.27, 1.03],
        hspace=0.20,
        wspace=0.12,
    )

    gs_a = outer[0, :].subgridspec(len(methods), 1, hspace=0.24)
    gs_b = outer[1, 0].subgridspec(2, 2, hspace=0.42, wspace=0.13)
    ax_legend = fig.add_subplot(outer[1, 1])
    ax_c = fig.add_subplot(outer[1, 2])

    plot_panel_a(
        fig,
        gs_a,
        df_a,
        phen_meta_a,
        methods,
        class_palette_a,
        show_bands=panel_a_show_bands,
        shade_negative=args.shade_negative,
        show_comments=args.show_comments,
    )
    plot_panel_b(fig, gs_b, df_b, methods, show_comments=args.show_comments)
    plot_panel_c(
        fig,
        ax_c,
        df_c,
        panel_c_methods,
        circle_summit_hits=args.circle_summit_bonf_hits,
        annotate_summit_hits=args.annotate_summit_bonf_hits,
    )
    build_method_legend(ax_legend, legend_methods)

    if args.show_comments:
        fig.text(
            0.985,
            0.985,
            "Off-scale values are marked by triangles at the axis boundary.",
            ha="right",
            va="top",
            fontsize=9.9,
            color="0.35",
        )

    outdir = os.path.dirname(args.outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(args.outfile, bbox_inches="tight")

    png_path = args.preview_png if args.preview_png else default_png_path(args.outfile)
    if os.path.abspath(png_path) != os.path.abspath(args.outfile):
        png_dir = os.path.dirname(png_path)
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
        fig.savefig(png_path, dpi=220, bbox_inches="tight")

    plt.close(fig)
    print(f"[ok] saved figure: {args.outfile}")
    if os.path.abspath(png_path) != os.path.abspath(args.outfile):
        print(f"[ok] saved preview: {png_path}")


if __name__ == "__main__":
    main()
