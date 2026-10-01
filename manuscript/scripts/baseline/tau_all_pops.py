"""Plot tau all pops for the SUMMIT manuscript."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


POP_ORDER = ["EUR_300k", "EUR", "SAS", "AFR"]


POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "SAS": "SAS",
    "AFR": "AFR",
}


TAB20 = [plt.matplotlib.colors.to_hex(c) for c in colormaps["tab20"].colors]


POP_COLOR = {
    "EUR_300k": TAB20[1],
    "EUR": TAB20[0],
    "SAS": TAB20[4],
    "AFR": TAB20[6],
}


FAMILY_ORDER = [
    "Hujoel",
    "Selection/conservation",
    "Molecular QTL",
    "Coding/gene model",
    "Regulatory",
    "Other",
]


def build_annotation_metadata(df: pd.DataFrame) -> pd.DataFrame:
    ann = (
        df[["annotation", "annotation_label", "annotation_family"]]
        .drop_duplicates("annotation")
        .copy()
    )
    ann["_source_order"] = np.arange(len(ann))
    family_rank = {family: idx for idx, family in enumerate(FAMILY_ORDER)}
    ann["_family_rank"] = (
        ann["annotation_family"].map(family_rank).fillna(len(FAMILY_ORDER)).astype(int)
    )
    ann = ann.sort_values(["_family_rank", "_source_order"]).reset_index(drop=True)
    return ann


def load_plot_data(
    meta_csv: Path,
    pops: Sequence[str],
) -> tuple[pd.DataFrame, list[str], Dict[str, str], Dict[str, str], pd.DataFrame]:
    df = pd.read_csv(meta_csv)
    required = {
        "pop",
        "annotation",
        "annotation_label",
        "annotation_family",
        "n_traits",
        "tau_star_meta",
        "tau_star_ci_lo",
        "tau_star_ci_hi",
        "tau_star_se",
    }
    missing_cols = sorted(required.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {meta_csv}: {missing_cols}")

    missing_pops = [pop for pop in pops if pop not in set(df["pop"])]
    if missing_pops:
        raise ValueError(f"Missing selected populations in {meta_csv}: {missing_pops}")

    ann = build_annotation_metadata(df[df["pop"].isin(pops)].copy())
    annotation_order = ann["annotation"].tolist()
    annotation_label = dict(zip(ann["annotation"], ann["annotation_label"]))
    annotation_family = dict(zip(ann["annotation"], ann["annotation_family"]))

    sub = df[df["annotation"].isin(annotation_order) & df["pop"].isin(pops)].copy()
    sub["annotation"] = pd.Categorical(
        sub["annotation"], categories=annotation_order, ordered=True
    )
    sub["pop"] = pd.Categorical(sub["pop"], categories=list(pops), ordered=True)
    sub["annotation_group"] = sub["annotation"].astype(str).map(annotation_family)
    sub["annotation_label_plot"] = (
        sub["annotation"]
        .astype(str)
        .map(annotation_label)
        .fillna(sub["annotation_label"])
    )
    sub = sub.sort_values(["annotation", "pop"]).reset_index(drop=True)

    expected = len(annotation_order) * len(pops)
    if sub.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} annotation-pop rows, found {sub.shape[0]}"
        )
    return sub, annotation_order, annotation_label, annotation_family, ann


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 9.8,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.8,
            "legend.title_fontsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def annotation_panels(
    annotation_order: Sequence[str],
    annotation_family: Dict[str, str],
) -> list[tuple[str, list[str]]]:
    left_families = {
        "Hujoel",
        "Selection/conservation",
        "Molecular QTL",
        "Coding/gene model",
    }
    left = [a for a in annotation_order if annotation_family.get(a) in left_families]
    right = [a for a in annotation_order if a not in set(left)]
    return [
        ("Conservation, QTL and gene model", left),
        ("Regulatory and other", right),
    ]


def add_family_separators(
    ax: plt.Axes,
    annotation_order: Sequence[str],
    annotation_family: Dict[str, str],
    y_lookup: Dict[str, float],
) -> None:
    for idx in range(len(annotation_order) - 1):
        current_family = annotation_family.get(annotation_order[idx])
        next_family = annotation_family.get(annotation_order[idx + 1])
        if current_family != next_family:
            ax.axhline(
                y_lookup[annotation_order[idx]] - 0.5,
                color="#BFBFBF",
                linewidth=0.8,
                zorder=1,
            )


def plot_tau_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    annotation_order: Sequence[str],
    annotation_label: Dict[str, str],
    annotation_family: Dict[str, str],
    title: str,
    x_limit: float,
) -> None:
    order = list(annotation_order)
    y_base = np.arange(len(order))[::-1]
    y_lookup = dict(zip(order, y_base))
    offsets = np.linspace(0.27, -0.27, num=len(POP_ORDER))

    for offset, pop in zip(offsets, POP_ORDER):
        sub = df[
            (df["pop"].astype(str) == pop) & (df["annotation"].astype(str).isin(order))
        ].copy()
        sub["annotation"] = pd.Categorical(
            sub["annotation"].astype(str), categories=order, ordered=True
        )
        sub = sub.sort_values("annotation")
        y = sub["annotation"].astype(str).map(y_lookup).to_numpy(dtype=float) + offset
        x = sub["tau_star_meta"].to_numpy(dtype=float)
        se = sub["tau_star_se"].to_numpy(dtype=float)

        for xi, yi, se_val in zip(x, y, se):
            if not np.isfinite(xi):
                continue
            xerr = None
            if np.isfinite(se_val):
                xerr = np.array([[se_val], [se_val]])
            ax.errorbar(
                xi,
                yi,
                xerr=xerr,
                fmt="o",
                markersize=4.0,
                elinewidth=0.95,
                capsize=0,
                color=POP_COLOR.get(pop, "#666666"),
                ecolor=POP_COLOR.get(pop, "#666666"),
                zorder=3,
            )

    ax.axvline(0.0, color="#5F5F5F", linewidth=0.9, linestyle=":")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.set_yticks(y_base)
    ax.set_yticklabels([annotation_label.get(a, a) for a in order])
    ax.set_title(title, pad=5)
    ax.set_ylim(-1.0, len(order))
    add_family_separators(ax, order, annotation_family, y_lookup)

    finite_bounds = np.column_stack(
        [
            df["tau_star_meta"].to_numpy(dtype=float)
            - df["tau_star_se"].to_numpy(dtype=float),
            df["tau_star_meta"].to_numpy(dtype=float)
            + df["tau_star_se"].to_numpy(dtype=float),
        ]
    )
    finite = finite_bounds[np.isfinite(finite_bounds)]
    if finite.size:
        if x_limit > 0:
            limit = float(x_limit)
        else:
            xmin = float(np.min(finite))
            xmax = float(np.max(finite))
            limit = max(abs(xmin), abs(xmax))
            limit += 0.06 * (2.0 * limit + 1e-8)
        ax.set_xlim(-limit, limit)
        if limit <= 1.25:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))


def plot_tau_all_pops(
    df: pd.DataFrame,
    annotation_order: Sequence[str],
    annotation_label: Dict[str, str],
    annotation_family: Dict[str, str],
    out_png: Path,
    out_pdf: Path,
    width: float,
    height: float,
    dpi: int,
    x_limit: float,
) -> None:
    set_style()
    panels = annotation_panels(annotation_order, annotation_family)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(width, height),
        sharex=True,
        gridspec_kw={"wspace": 0.72},
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, order) in zip(axes, panels):
        plot_tau_panel(
            ax,
            df,
            order,
            annotation_label,
            annotation_family,
            title,
            x_limit,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.2,
            color=POP_COLOR[pop],
            label=POP_LABEL[pop],
        )
        for pop in POP_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        title="Cohort",
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=len(POP_ORDER),
        columnspacing=1.15,
        handletextpad=0.4,
        frameon=True,
        fancybox=False,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("#4D4D4D")
    legend.get_frame().set_linewidth(0.8)

    fig.supxlabel(r"Cross-trait meta $\tau^{*}$", y=0.035, fontsize=10.5)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.855, bottom=0.105, wspace=0.72)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    df, order, labels, families, _ = load_plot_data(
        Path("data/baseline/tau_all_pops.csv"), POP_ORDER
    )
    height = max(
        7.4,
        2.2 + 0.20 * max(len(order) for _, order in annotation_panels(order, families)),
    )
    plot_tau_all_pops(
        df,
        order,
        labels,
        families,
        Path("figs/supplementary/fig_s29_baseline_tau.png"),
        Path("figs/supplementary/fig_s29_baseline_tau.pdf"),
        9.8,
        height,
        450,
        0.8,
    )
