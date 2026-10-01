"""Plot panels for the SUMMIT manuscript."""
from __future__ import annotations

import argparse
import math
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import baseline_common as base
import annotation_families as families
import style as common

POP_ORDER = ["EUR_300k", "EUR", "SAS", "AFR"]


DEFAULT_MAIN_AB_SIZE = (14.2, 5.25)


DEFAULT_SUPP_ACD_SIZE = (16.2, 8.6)


SPECIFICITY_CONTROL_PAIRS = [
    ("DGF / DHS", "DGF_ENCODE_w_flanking", "DHS_Trynka_w_flanking"),
    (
        "Weak enhancer / enhancer",
        "WeakEnhancer_Hoffman_w_flanking",
        "Enhancer_Hoffman_w_flanking",
    ),
    ("Transcribed / intron", "Transcr_Hoffman_w_flanking", "Intron_UCSC_w_flanking"),
]


RATIO_GROUPS = [
    ("hujoel", base.HUJOEL_PAIRS),
    ("core_regulatory", base.CORE_REGULATORY_PAIRS),
    ("specificity_control", SPECIFICITY_CONTROL_PAIRS),
]


FAMILY_ORDER = [
    "Hujoel",
    "Selection/conservation",
    "Molecular QTL",
    "Coding/gene model",
    "Regulatory",
    "Other",
]


def figure_size(
    args: argparse.Namespace, default: Tuple[float, float]
) -> Tuple[float, float]:
    width = default[0] if args.width is None else args.width
    height = default[1] if args.height is None else args.height
    return width, height


def set_style() -> None:
    common.set_style()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "legend.title_fontsize": 9.2,
        }
    )


def add_panel_letter(
    fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.035, dy: float = 0.018
) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx, bbox.y1 + dy, label, fontsize=16, fontweight="bold", va="top"
    )


def pop_label(pop: str) -> str:
    return common.POP_LABEL.get(str(pop), str(pop))


def pop_color(pop: str) -> str:
    return common.POP_COLOR.get(str(pop), "#666666")


def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(
        textwrap.wrap(
            str(label), width=width, break_long_words=False, break_on_hyphens=False
        )
    )


def panel_a_pair_order(pair_mode: str, reference_pop: str) -> List[str]:
    if pair_mode == "all":
        return common.all_pair_order()
    if pair_mode == "reference":
        return [f"{reference_pop}-{pop}" for pop in POP_ORDER if pop != reference_pop]
    raise ValueError(f"Unknown panel-A pair mode: {pair_mode}")


def panel_a_estimands(estimand_mode: str) -> List[str]:
    if estimand_mode == "both":
        return ["h2", "tau_star"]
    return [estimand_mode]


def ratio_order() -> List[Tuple[str, str]]:
    ordered: List[Tuple[str, str]] = []
    for contrast_set, pairs in RATIO_GROUPS:
        ordered.extend((contrast_set, pair_name) for pair_name, _, _ in pairs)
    return ordered


def plot_enrichment_scatter_panel(
    fig: plt.Figure,
    gs_cell,
    annotation_pairs: pd.DataFrame,
    annotation_summary: pd.DataFrame,
    reference_pop: str,
    enrichment_scale: str,
) -> List[plt.Axes]:
    subgs = gs_cell.subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.40], wspace=0.18)
    axes = [fig.add_subplot(subgs[0, i]) for i in range(3)]
    legend_ax = fig.add_subplot(subgs[0, 3])
    legend_ax.axis("off")

    if enrichment_scale == "raw":
        enrichment_metric = "raw_enrichment"
        axis_label = "raw enrichment"
        baseline = 1.0
        min_pad = 0.25
    elif enrichment_scale == "raw_meta_log2":
        enrichment_metric = "raw_meta_log2_enrichment"
        axis_label = "log2 enrichment (raw meta)"
        baseline = 0.0
        min_pad = 0.18
    elif enrichment_scale == "log2":
        enrichment_metric = "log2_enrichment"
        axis_label = "log2 enrichment"
        baseline = 0.0
        min_pad = 0.18
    else:
        raise ValueError(f"Unknown enrichment scale: {enrichment_scale}")

    enrich_pairs = annotation_pairs[
        annotation_pairs["metric"] == enrichment_metric
    ].copy()
    targets = [
        p
        for p in POP_ORDER
        if p != reference_pop and p in set(enrich_pairs["target_pop"].astype(str))
    ]
    finite_vals = pd.concat(
        [enrich_pairs["reference_estimate"], enrich_pairs["target_estimate"]], axis=0
    ).dropna()
    if len(finite_vals):
        lo = min(float(finite_vals.min()), baseline)
        hi = max(float(finite_vals.max()), baseline)
        pad = max(min_pad, 0.08 * (hi - lo))
        lims = (lo - pad, hi + pad)
    else:
        lims = (-1.0, 4.0) if enrichment_scale == "log2" else (-1.0, 4.0)

    for ax, target in zip(axes, targets):
        sub = enrich_pairs[enrich_pairs["target_pop"] == target].copy()
        for family in FAMILY_ORDER:
            grp = sub[sub["annotation_family"] == family]
            if grp.empty:
                continue
            ax.scatter(
                grp["reference_estimate"],
                grp["target_estimate"],
                s=22,
                alpha=0.82,
                color=families.FAMILY_COLORS.get(
                    family, families.FAMILY_COLORS["Other"]
                ),
                linewidths=0,
            )

        ax.plot(lims, lims, color="#777777", linewidth=0.8, linestyle=":")
        ax.axhline(baseline, color="#D0D0D0", linewidth=0.7)
        ax.axvline(baseline, color="#D0D0D0", linewidth=0.7)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.grid(color="#D9D9D9", linestyle=":", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{pop_label(reference_pop)} vs {pop_label(target)}", pad=5)
        ax.set_xlabel(f"{pop_label(reference_pop)} {axis_label}")
        if ax is axes[0]:
            ax.set_ylabel(f"Target {axis_label}")
        else:
            ax.tick_params(axis="y", labelleft=False)

        smry = annotation_summary[
            (annotation_summary["target_pop"] == target)
            & (annotation_summary["metric"] == enrichment_metric)
        ]
        if not smry.empty:
            row = smry.iloc[0]
            ax.text(
                0.95,
                0.06,
                f"r={row['pearson']:.2f}\n$\\rho$={row['spearman']:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#303030",
                bbox={
                    "boxstyle": "square,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#4D4D4D",
                    "linewidth": 0.7,
                    "alpha": 0.94,
                },
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=families.FAMILY_COLORS.get(
                family, families.FAMILY_COLORS["Other"]
            ),
            markeredgecolor="#303030",
            markeredgewidth=0.55,
            linestyle="",
            markersize=5.5,
            label=family,
        )
        for family in FAMILY_ORDER
        if family in set(enrich_pairs["annotation_family"].astype(str))
    ]
    legend_ax.legend(
        handles=handles,
        title="Annotation family",
        loc="center left",
        frameon=False,
        borderaxespad=0,
        handletextpad=0.5,
        labelspacing=0.8,
    )
    return axes


def plot_panel_a(
    fig: plt.Figure,
    gs_cell,
    pairwise_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[plt.Axes, List[plt.Axes]]:
    compact_main = getattr(args, "figure_kind", "full") == "main_ab"
    estimands = panel_a_estimands(args.panel_a_estimands)
    pair_order = panel_a_pair_order(args.panel_a_pairs, args.reference_pop)
    pair_pretty = [common.pair_label(*p.split("-"), pretty=True) for p in pair_order]
    method_order = common.method_display_order(pairwise_df["method"].unique())

    if len(estimands) == 1:
        axes = [fig.add_subplot(gs_cell)]
    else:
        subgs = gs_cell.subgridspec(len(estimands), 1, hspace=0.24)
        axes = [fig.add_subplot(subgs[i, 0]) for i in range(len(estimands))]

    n_methods = max(1, len(method_order))
    group_spacing = 1.28 if compact_main else 1.0
    x_base = np.arange(len(pair_order), dtype=float) * group_spacing
    group_span = 0.88
    bar_w = group_span / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_w

    work = pairwise_df.copy()
    work["pair"] = pd.Categorical(work["pair"], categories=pair_order, ordered=True)
    work = work[work["pair"].notna()].copy()

    legend_handles = []
    legend_labels = []
    for midx, method in enumerate(method_order):
        color = common.METHOD_COLOR.get(method, "#4C4C4C")
        label = common.METHOD_LABEL.get(method, method)
        legend_handles.append(
            common.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        )
        legend_labels.append(label)

        for estimand, ax in zip(estimands, axes):
            ycol, secol, title, ylabel = common.panel_a_metric_spec(
                estimand, args.panel_a_corr_metric
            )
            sub = work[
                (work["method"] == method) & (work["estimand"] == estimand)
            ].copy()
            sub = sub.set_index("pair").reindex(pair_order)
            ys = sub[ycol].to_numpy(dtype=float)
            es = sub[secol].to_numpy(dtype=float)
            xs = x_base + offsets[midx]
            mask = np.isfinite(ys)
            bars = ax.bar(
                xs[mask],
                ys[mask],
                width=bar_w * 0.92,
                color=color,
                edgecolor="none",
                zorder=3,
            )
            if method == "sumrhe":
                common.add_bar_contour(ax, bars, color="black", lw=1.1, zorder=3.6)
            e_mask = mask & np.isfinite(es)
            if np.any(e_mask):
                ax.errorbar(
                    xs[e_mask],
                    ys[e_mask],
                    yerr=es[e_mask],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.9,
                    capsize=2.5,
                    zorder=4,
                )

    for estimand, ax in zip(estimands, axes):
        _, _, title, ylabel = common.panel_a_metric_spec(
            estimand, args.panel_a_corr_metric
        )
        if compact_main:
            title = ""
            ylabel = (
                "Pearson $r$"
                if args.panel_a_corr_metric == "pearson"
                else "Spearman $\\rho$"
            )
        ax.axhline(0.0, color="#5F5F5F", linewidth=0.8, linestyle="-", zorder=2)
        ax.axhspan(-0.25, 0.0, color="#F2F2F2", zorder=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, color="#D9D9D9")
        ax.set_axisbelow(True)
        ax.set_ylim(-0.25, 1.08 if compact_main else 1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=5)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(pair_pretty, rotation=0, ha="center")
    axes[-1].set_xlabel("Population pair")

    legend_ax = axes[-1]
    if compact_main:
        legend_loc = "lower center"
        legend_anchor = (0.5, 1.005)
        legend_ncol = len(method_order)
    elif len(estimands) == 1:
        legend_loc = "upper right"
        legend_anchor = (1.0, 0.985)
        legend_ncol = 2
    else:
        legend_loc = "center left"
        legend_anchor = (0.76, 0.62)
        legend_ncol = 1
    leg = legend_ax.legend(
        legend_handles,
        legend_labels,
        ncol=legend_ncol,
        frameon=True,
        fancybox=False,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        borderaxespad=0.0,
        columnspacing=0.75,
        handletextpad=0.45,
        borderpad=0.32 if compact_main else 0.4,
        labelspacing=0.28 if compact_main else 0.5,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor("#4D4D4D")
    leg.get_frame().set_linewidth(0.8)

    return axes[0], axes


def plot_ratio_panel(ax: plt.Axes, ratio_meta: pd.DataFrame) -> None:
    order = ratio_order()
    y_lookup = {key: len(order) - 1 - i for i, key in enumerate(order)}
    pops = [p for p in POP_ORDER if p in set(ratio_meta["pop"].astype(str))]
    offsets = np.linspace(0.27, -0.27, num=len(pops))

    for offset, pop in zip(offsets, pops):
        sub = ratio_meta[ratio_meta["pop"] == pop].copy()
        for _, row in sub.iterrows():
            key = (row["contrast_set"], row["pair_name"])
            if key not in y_lookup:
                continue
            est = float(row["estimate"])
            lo = float(row["ci_lo"]) if pd.notna(row["ci_lo"]) else math.nan
            hi = float(row["ci_hi"]) if pd.notna(row["ci_hi"]) else math.nan
            if not np.isfinite(est) or est <= 0:
                continue
            bh = float(row["bh_fdr_one_sided_all_panel_b"])
            bonf = float(row["bonferroni_one_sided_all_panel_b"])
            if np.isfinite(bonf) and bonf < 0.05:
                marker = "*"
                marker_size = 7.5
            elif np.isfinite(bh) and bh < 0.05:
                marker = "D"
                marker_size = 4.9
            else:
                marker = "o"
                marker_size = 4.7
            xerr = None
            if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0:
                xerr = np.array([[est - lo], [hi - est]])
            ax.errorbar(
                est,
                y_lookup[key] + offset,
                xerr=xerr,
                fmt=marker,
                color=pop_color(pop),
                ecolor=pop_color(pop),
                markersize=marker_size,
                elinewidth=1.05,
                capsize=0,
                zorder=3,
            )

    # Visual separators after Hujoel and core-regulatory blocks.
    for sep_after in [3, 7]:
        ax.axhline(len(order) - sep_after - 0.5, color="#CFCFCF", linewidth=0.8)

    ax.set_xscale("log")
    ax.axvline(1.0, color="#5F5F5F", linewidth=0.8, linestyle=":")
    ax.grid(axis="x", color="#D9D9D9", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks([y_lookup[key] for key in order])
    ax.set_yticklabels([wrap_label(pair, width=29) for _, pair in order], fontsize=8.4)
    ax.tick_params(axis="y", pad=3)
    ax.set_xlabel("Cross-trait meta enrichment ratio")
    ax.set_title("Enrichment-ratio contrasts", pad=6)

    finite = ratio_meta[["estimate", "ci_lo", "ci_hi"]].to_numpy(dtype=float)
    vals = finite[np.isfinite(finite) & (finite > 0)]
    if vals.size:
        xmin = max(0.35, float(np.nanmin(vals)) * 0.78)
        xmax = min(12.0, float(np.nanmax(vals)) * 1.20)
    else:
        xmin, xmax = 0.45, 9.5
    ax.set_xlim(xmin, xmax)
    ticks = [0.5, 1.0, 2.0, 4.0, 8.0]
    ax.set_xticks([t for t in ticks if xmin <= t <= xmax])
    ax.set_xticklabels([f"{t:g}" for t in ticks if xmin <= t <= xmax])

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.2,
            color=pop_color(pop),
            label=pop_label(pop),
        )
        for pop in pops
    ]
    leg = ax.legend(
        handles=handles,
        title="Cohort",
        loc="lower right",
        ncol=1,
        frameon=True,
        fancybox=False,
        borderpad=0.35,
        handletextpad=0.45,
        labelspacing=0.32,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.96)
    leg.get_frame().set_edgecolor("#4D4D4D")
    leg.get_frame().set_linewidth(0.8)


def plot_trait_profile_panel(
    ax: plt.Axes, trait_detail: pd.DataFrame, reference_pop: str
) -> None:
    targets = [
        p
        for p in POP_ORDER
        if p != reference_pop and p in set(trait_detail["target_pop"].astype(str))
    ]
    data = [
        trait_detail.loc[trait_detail["target_pop"] == pop, "spearman"]
        .dropna()
        .to_numpy(dtype=float)
        for pop in targets
    ]
    positions = np.arange(len(targets), dtype=float)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.2},
        boxprops={"facecolor": "#F5F5F5", "edgecolor": "#777777", "linewidth": 0.8},
        whiskerprops={"color": "#777777", "linewidth": 0.8},
        capprops={"color": "#777777", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_alpha(1.0)

    rng = np.random.default_rng(20260511)
    for x, pop, vals in zip(positions, targets, data):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=vals.size)
        ax.scatter(
            np.full(vals.size, x) + jitter,
            vals,
            s=13,
            color=pop_color(pop),
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )

    ax.axhline(0, color="#5F5F5F", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.08, 1.02)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{pop_label(pop)}\nn={len(vals)}" for pop, vals in zip(targets, data)]
    )
    ax.set_ylabel("Per-trait Spearman rho")
    ax.set_title("Trait-level enrichment profiles", pad=6)


def build_main_ab_figure(
    panel_a_df: pd.DataFrame,
    panels: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
    out_pdf: str,
    out_png: str,
) -> None:
    fig = plt.figure(figsize=figure_size(args, DEFAULT_MAIN_AB_SIZE))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.16, 1.34],
        left=0.07,
        right=0.985,
        top=0.80,
        bottom=0.155,
        wspace=0.34,
    )

    ax_a, _ = plot_panel_a(
        fig=fig,
        gs_cell=outer[0, 0],
        pairwise_df=panel_a_df,
        args=args,
    )
    ax_ratio = fig.add_subplot(outer[0, 1])
    plot_ratio_panel(ax_ratio, panels["ratio_meta"])
    ax_ratio.set_title("")

    add_panel_letter(fig, ax_a, "A", dx=-0.048, dy=0.075)
    add_panel_letter(fig, ax_ratio, "B", dx=-0.035, dy=0.075)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)


def build_supp_acd_figure(
    panel_a_df: pd.DataFrame,
    panels: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
    out_pdf: str,
    out_png: str,
) -> None:
    fig = plt.figure(figsize=figure_size(args, DEFAULT_SUPP_ACD_SIZE))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.72, 1.0],
        left=0.075,
        right=0.985,
        top=0.94,
        bottom=0.085,
        hspace=0.36,
    )
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.26)

    ax_a, _ = plot_panel_a(
        fig=fig,
        gs_cell=top[0, 0],
        pairwise_df=panel_a_df,
        args=args,
    )
    ax_trait = fig.add_subplot(top[0, 1])
    plot_trait_profile_panel(ax_trait, panels["trait_detail"], args.reference_pop)
    scatter_axes = plot_enrichment_scatter_panel(
        fig=fig,
        gs_cell=outer[1, 0],
        annotation_pairs=panels["annotation_pairs"],
        annotation_summary=panels["annotation_summary"],
        reference_pop=args.reference_pop,
        enrichment_scale=args.enrichment_scale,
    )

    add_panel_letter(fig, ax_a, "A")
    add_panel_letter(fig, ax_trait, "B")
    add_panel_letter(fig, scatter_axes[0], "C")

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)
