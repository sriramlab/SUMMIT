#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import cross_population as scatter
import venn_maintext as venn


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = Path("data/real_rg/estimates.csv")
DEFAULT_OUTDIR = Path("figs/main")
DEFAULT_BASENAME = "genetic_correlation"
DEFAULT_POPS = ["EUR_300k", "AFR", "SAS"]
DEFAULT_SCATTER_POPS = ["EUR", "SAS", "AFR"]
DEFAULT_REF_POP = "EUR_300k"
DEFAULT_METHOD = "summit"
DEFAULT_ANNOT = "mafld_8bins"
DEFAULT_THRESHOLD = "bh"
DEFAULT_ALPHA = 0.05
DEFAULT_BH_Q = 0.05
DEFAULT_FAMILY_SIZE = 780
SCATTER_TICK_STEP = 0.5
SCATTER_TITLE_SIZE = 12.6
SCATTER_AXIS_LABEL_SIZE = 11.4
SCATTER_TICK_LABEL_SIZE = 9.8
SCATTER_PAIR_LABEL_SIZE = 8.8
SCATTER_R_LABEL_SIZE = 11.6
SCATTER_PANEL_LETTER_SIZE = 14.2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Draw genetic-correlation scatter and overlap panels."
    )
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Parsed rg CSV from the supplied aggregate table.",
    )
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory.")
    ap.add_argument(
        "--basename", default=DEFAULT_BASENAME, help="Output basename prefix."
    )
    ap.add_argument("--dpi", type=int, default=450)
    ap.add_argument("--family-size", type=int, default=DEFAULT_FAMILY_SIZE)
    ap.add_argument(
        "--threshold",
        choices=["nominal", "bh", "bonferroni"],
        default=DEFAULT_THRESHOLD,
    )
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--bh-q", type=float, default=DEFAULT_BH_Q)
    ap.add_argument(
        "--circle-size-mode", choices=["fixed", "proportional"], default="fixed"
    )
    ap.add_argument("--proportional-min-radius", type=float, default=0.18)
    return ap.parse_args()


def load_panel_hits(
    df: pd.DataFrame,
    *,
    pops: Sequence[str],
    annot: str,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_size: int,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    family_sizes = {pop: family_size for pop in pops}
    return venn.load_all_panel_hits(
        df,
        pops=pops,
        annot=annot,
        threshold=threshold,
        alpha=alpha,
        bh_q=bh_q,
        family_sizes=family_sizes,
    )


def build_scatter_frames(
    df: pd.DataFrame,
    *,
    scatter_pops: Sequence[str],
    ref_pop: str,
    method: str,
    annot: str,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_size: int,
) -> Tuple[List[pd.DataFrame], List[Dict[str, object]]]:
    frames = [
        scatter.build_comparison_table(
            df,
            ref_pop=ref_pop,
            small_pop=pop,
            method=method,
            annot=annot,
            threshold=threshold,
            alpha=alpha,
            bh_q=bh_q,
            family_size=family_size,
        )
        for pop in scatter_pops
    ]
    metrics: List[Dict[str, object]] = []
    for pop, frame in zip(scatter_pops, frames):
        row: Dict[str, object] = {
            "pop": pop,
            "pop_label": scatter.POP_LABEL.get(pop, pop),
            "ref_pop": ref_pop,
            "ref_pop_label": scatter.POP_LABEL.get(ref_pop, ref_pop),
            "method": method,
            "annot_type": annot,
            "threshold": threshold,
            "alpha": alpha,
            "bh_q": bh_q,
            "family_size": family_size,
            "errorbar_scale": 1.0,
            "label_max": 8,
            "label_outliers": 3,
            "label_min_distance": 0.12,
        }
        row.update(scatter.summarize(frame))
        metrics.append(row)
    return frames, metrics


def draw_scatter_row(
    *,
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    frames: Sequence[pd.DataFrame],
    metrics: Sequence[Dict[str, object]],
    scatter_pops: Sequence[str],
    ref_pop: str,
    method: str,
    annot: str,
    letters: Sequence[str],
) -> List[Tuple[plt.Axes, pd.DataFrame, pd.DataFrame, Tuple[float, float]]]:
    limits = scatter.data_limits(frames, errorbar_scale=1.0)
    label_jobs: List[
        Tuple[plt.Axes, pd.DataFrame, pd.DataFrame, Tuple[float, float]]
    ] = []
    for idx, (ax, pop, frame, metric, letter) in enumerate(
        zip(axes, scatter_pops, frames, metrics, letters)
    ):
        labels = scatter.draw_panel(
            ax,
            tab=frame,
            pop=pop,
            ref_pop=ref_pop,
            method=method,
            annot=annot,
            limits=limits,
            errorbar_scale=1.0,
            label_max=8,
            label_outliers=3,
            label_min_distance=0.12,
            letter=letter,
            metrics=metric,
        )
        format_scatter_axis(ax, letter=letter, show_y_tick_labels=(idx == 0))
        label_jobs.append((ax, frame, labels, limits))
    del fig
    return label_jobs


def format_scatter_axis(ax: plt.Axes, *, letter: str, show_y_tick_labels: bool) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(SCATTER_TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(SCATTER_TICK_STEP))
    ax.tick_params(axis="both", which="major", labelsize=SCATTER_TICK_LABEL_SIZE)
    ax.tick_params(axis="y", which="major", labelleft=show_y_tick_labels)
    ax.set_xlabel(ax.get_xlabel(), fontsize=SCATTER_AXIS_LABEL_SIZE, labelpad=4.0)
    ax.set_ylabel(ax.get_ylabel(), fontsize=SCATTER_AXIS_LABEL_SIZE, labelpad=4.0)
    ax.title.set_fontsize(SCATTER_TITLE_SIZE)

    for text in ax.texts:
        if text.get_text() == letter:
            text.set_fontsize(SCATTER_PANEL_LETTER_SIZE)
        elif text.get_text().startswith("r="):
            text.set_fontsize(SCATTER_R_LABEL_SIZE)


def draw_population_overlap_panel(
    ax: plt.Axes,
    *,
    hits_by_group: Dict[Tuple[str, str], pd.DataFrame],
    method: str,
    pops: Sequence[str],
    letter: str,
    annot: str,
    threshold_label: str,
    circle_size_mode: str,
    proportional_min_radius: float,
) -> None:
    hit_map = {pop: hits_by_group[(pop, method)] for pop in pops}
    set_map = venn.to_set_map(hit_map)
    venn.draw_three_set_panel(
        ax,
        letter=letter,
        title=venn.METHOD_LABEL[method],
        subtitle=f"Population overlap of {threshold_label} trait pairs",
        title_color=venn.METHOD_COLOR[method],
        set_names=list(pops),
        set_labels=[venn.POP_LABEL[pop] for pop in pops],
        set_colors=[venn.POP_COLOR[pop] for pop in pops],
        set_values=[set_map[pop] for pop in pops],
        circle_size_mode=circle_size_mode,
        proportional_min_radius=proportional_min_radius,
    )
    del annot


def draw_method_overlap_panel(
    ax: plt.Axes,
    *,
    hits_by_group: Dict[Tuple[str, str], pd.DataFrame],
    pop: str,
    letter: str,
    circle_size_mode: str,
    proportional_min_radius: float,
) -> None:
    set_names = list(venn.BOTTOM_ROW_METHODS)
    hit_map = {method: hits_by_group[(pop, method)] for method in set_names}
    set_map = venn.to_set_map(hit_map)
    venn.draw_two_set_panel(
        ax,
        letter=letter,
        title=venn.POP_LABEL[pop],
        subtitle="SUMMIT vs cov-LDSC",
        title_color=venn.POP_COLOR[pop],
        set_names=set_names,
        set_labels=[venn.METHOD_LABEL[method] for method in set_names],
        set_colors=[venn.METHOD_COLOR[method] for method in set_names],
        set_values=[set_map[method] for method in set_names],
        circle_size_mode=circle_size_mode,
        proportional_min_radius=proportional_min_radius,
    )


def add_scatter_labels(
    *,
    fig: plt.Figure,
    label_jobs: Sequence[
        Tuple[plt.Axes, pd.DataFrame, pd.DataFrame, Tuple[float, float]]
    ],
) -> None:
    fig.canvas.draw()
    for ax, frame, labels, limits in label_jobs:
        scatter.add_point_labels(
            ax,
            tab=frame,
            labels=labels,
            limits=limits,
            errorbar_scale=1.0,
        )


def save(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_replace_toprow_version(
    *,
    df: pd.DataFrame,
    hits_by_group: Dict[Tuple[str, str], pd.DataFrame],
    args: argparse.Namespace,
    outbase: Path,
) -> None:
    frames, metrics = build_scatter_frames(
        df,
        scatter_pops=DEFAULT_SCATTER_POPS,
        ref_pop=DEFAULT_REF_POP,
        method=DEFAULT_METHOD,
        annot=DEFAULT_ANNOT,
        threshold=args.threshold,
        alpha=args.alpha,
        bh_q=args.bh_q,
        family_size=args.family_size,
    )

    fig = plt.figure(figsize=(12.2, 8.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.08, 0.96], hspace=0.38, wspace=0.30)
    scatter_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]
    venn_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    label_jobs = draw_scatter_row(
        fig=fig,
        axes=scatter_axes,
        frames=frames,
        metrics=metrics,
        scatter_pops=DEFAULT_SCATTER_POPS,
        ref_pop=DEFAULT_REF_POP,
        method=DEFAULT_METHOD,
        annot=DEFAULT_ANNOT,
        letters=["A", "B", "C"],
    )
    for ax, pop, letter in zip(venn_axes, venn.BOTTOM_PANEL_POPS, ["D", "E", "F"]):
        draw_method_overlap_panel(
            ax,
            hits_by_group=hits_by_group,
            pop=pop,
            letter=letter,
            circle_size_mode=args.circle_size_mode,
            proportional_min_radius=args.proportional_min_radius,
        )

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.975)
    add_scatter_labels(fig=fig, label_jobs=label_jobs)
    save(
        fig,
        outbase.with_name(outbase.name + "__v1_scatter_plus_method_venns"),
        dpi=args.dpi,
    )


def build_keep_all_plus_scatter_version(
    *,
    df: pd.DataFrame,
    hits_by_group: Dict[Tuple[str, str], pd.DataFrame],
    args: argparse.Namespace,
    outbase: Path,
) -> None:
    frames, metrics = build_scatter_frames(
        df,
        scatter_pops=DEFAULT_SCATTER_POPS,
        ref_pop=DEFAULT_REF_POP,
        method=DEFAULT_METHOD,
        annot=DEFAULT_ANNOT,
        threshold=args.threshold,
        alpha=args.alpha,
        bh_q=args.bh_q,
        family_size=args.family_size,
    )

    fig = plt.figure(figsize=(12.2, 12.0))
    gs = fig.add_gridspec(
        3, 6, height_ratios=[0.78, 0.92, 1.04], hspace=0.20, wspace=0.24
    )

    top_axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
    middle_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]
    scatter_axes = [
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
    ]

    for ax, method, letter in zip(top_axes, venn.TOP_ROW_METHODS, ["A", "B"]):
        draw_population_overlap_panel(
            ax,
            hits_by_group=hits_by_group,
            method=method,
            pops=DEFAULT_POPS,
            letter=letter,
            annot=DEFAULT_ANNOT,
            threshold_label="FDR-significant",
            circle_size_mode=args.circle_size_mode,
            proportional_min_radius=args.proportional_min_radius,
        )
    for ax, pop, letter in zip(middle_axes, venn.BOTTOM_PANEL_POPS, ["C", "D", "E"]):
        draw_method_overlap_panel(
            ax,
            hits_by_group=hits_by_group,
            pop=pop,
            letter=letter,
            circle_size_mode=args.circle_size_mode,
            proportional_min_radius=args.proportional_min_radius,
        )

    label_jobs = draw_scatter_row(
        fig=fig,
        axes=scatter_axes,
        frames=frames,
        metrics=metrics,
        scatter_pops=DEFAULT_SCATTER_POPS,
        ref_pop=DEFAULT_REF_POP,
        method=DEFAULT_METHOD,
        annot=DEFAULT_ANNOT,
        letters=["F", "G", "H"],
    )

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.980)
    add_scatter_labels(fig=fig, label_jobs=label_jobs)
    save(
        fig,
        outbase.with_name(outbase.name + "__v2_all_venns_plus_scatter"),
        dpi=args.dpi,
    )


def build_method_venns_top_scatter_bottom_version(
    *,
    df: pd.DataFrame,
    hits_by_group: Dict[Tuple[str, str], pd.DataFrame],
    args: argparse.Namespace,
    outbase: Path,
) -> None:
    frames, metrics = build_scatter_frames(
        df,
        scatter_pops=DEFAULT_SCATTER_POPS,
        ref_pop=DEFAULT_REF_POP,
        method=DEFAULT_METHOD,
        annot=DEFAULT_ANNOT,
        threshold=args.threshold,
        alpha=args.alpha,
        bh_q=args.bh_q,
        family_size=args.family_size,
    )

    fig = plt.figure(figsize=(12.2, 8.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.88, 1.08], hspace=0.06, wspace=0.30)
    venn_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]
    scatter_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    for ax, pop, letter in zip(venn_axes, venn.BOTTOM_PANEL_POPS, ["A", "B", "C"]):
        draw_method_overlap_panel(
            ax,
            hits_by_group=hits_by_group,
            pop=pop,
            letter=letter,
            circle_size_mode=args.circle_size_mode,
            proportional_min_radius=args.proportional_min_radius,
        )

    label_jobs = draw_scatter_row(
        fig=fig,
        axes=scatter_axes,
        frames=frames,
        metrics=metrics,
        scatter_pops=DEFAULT_SCATTER_POPS,
        ref_pop=DEFAULT_REF_POP,
        method=DEFAULT_METHOD,
        annot=DEFAULT_ANNOT,
        letters=["D", "E", "F"],
    )

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.070, top=0.975)
    add_scatter_labels(fig=fig, label_jobs=label_jobs)
    save(fig, outbase, dpi=args.dpi)


def main() -> None:
    args = parse_args()
    if args.family_size <= 0:
        raise SystemExit("--family-size must be positive.")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing input CSV: {csv_path}")

    venn.set_style()
    scatter.set_style()
    scatter.LABEL_FONT_SIZE = SCATTER_PAIR_LABEL_SIZE

    df = pd.read_csv(csv_path)
    venn.validate_inputs(
        argparse.Namespace(
            pops=DEFAULT_POPS,
            family_size=args.family_size,
            supp=False,
            annot=DEFAULT_ANNOT,
            circle_size_mode=args.circle_size_mode,
            proportional_min_radius=args.proportional_min_radius,
        ),
        df,
    )
    hits_by_group = load_panel_hits(
        df,
        pops=DEFAULT_POPS,
        annot=DEFAULT_ANNOT,
        threshold=args.threshold,
        alpha=args.alpha,
        bh_q=args.bh_q,
        family_size=args.family_size,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outbase = outdir / args.basename
    build_replace_toprow_version(
        df=df, hits_by_group=hits_by_group, args=args, outbase=outbase
    )
    build_keep_all_plus_scatter_version(
        df=df, hits_by_group=hits_by_group, args=args, outbase=outbase
    )
    build_method_venns_top_scatter_bottom_version(
        df=df, hits_by_group=hits_by_group, args=args, outbase=outbase
    )

    print(f"Wrote {outbase.name}__v1_scatter_plus_method_venns.[png,pdf]")
    print(f"Wrote {outbase.name}__v2_all_venns_plus_scatter.[png,pdf]")
    print(f"Wrote {outbase.name}__v3_method_venns_top_scatter_bottom.[png,pdf]")


if __name__ == "__main__":
    main()
