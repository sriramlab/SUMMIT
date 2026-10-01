#!/usr/bin/env python3
from __future__ import annotations

"""
Build a publication-oriented main-text Venn figure for MAF-LD 8-bin rg hits.

The layout is fixed to five panels:
  A. Population overlap of SUMMIT significant hits
  B. Population overlap of cov-LDSC significant hits
  C. SUMMIT vs cov-LDSC overlap in EUR (300k)
  D. SUMMIT vs cov-LDSC overlap in AFR
  E. SUMMIT vs cov-LDSC overlap in SAS

Unlike the generic venn.py helper, this script uses fixed-geometry circles and
manually placed count labels. That gives substantially more control over panel
alignment, typography, and empty-set handling for a manuscript-quality figure.

An optional proportional-circle mode is also available. That mode delegates the
circle geometry to matplotlib-venn so circle areas track hit counts, while the
default fixed mode preserves the cleaner manuscript layout.
"""

import argparse
from math import erf, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import warnings
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

try:
    from matplotlib_venn import venn2, venn3
except Exception:
    venn2 = None
    venn3 = None


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLS = {"pop", "method", "annot_type", "phen1", "phen2", "rg", "rg_se"}

TOP_ROW_METHODS = ["summit", "covldsc"]
BOTTOM_ROW_METHODS = ["summit", "covldsc"]
DEFAULT_POPS = ["EUR_300k", "AFR", "SAS"]
BOTTOM_PANEL_POPS = ["EUR_300k", "SAS", "AFR"]
DEFAULT_ANNOT = "mafld_8bins"
DEFAULT_TEST_FAMILY_SIZE = 780
SUPP_METHODS = ["summit", "covldsc"]
SUPP_TOP_ANNOTS = ["baseline", "mafld_24bins", "single"]
SUPP_BOTTOM_ANNOTS = ["mafld_8bins", "mafld_24bins", "baseline"]
TAB20 = [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]

POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "AFR": "AFR",
    "SAS": "SAS",
}
POP_COLOR = {
    "EUR_300k": TAB20[1],
    "AFR": TAB20[6],
    "SAS": TAB20[4],
}

METHOD_LABEL = {
    "summit": "SUMMIT",
    "covldsc": "cov-LDSC",
}
METHOD_COLOR = {
    "summit": TAB20[4],
    "covldsc": TAB20[2],
}

ANNOT_LABEL = {
    "baseline": "Baseline",
    "mafld_8bins": "MAF-LD 8-bin annotations",
    "mafld_24bins": "MAF-LD 24-bin annotations",
    "single": "Single-component annotation",
}
ANNOT_PANEL_LABEL = {
    "baseline": "Baseline",
    "mafld_8bins": "MAF-LD 8 bins",
    "mafld_24bins": "MAF-LD 24 bins",
    "single": "Single component",
}
ANNOT_SET_LABEL = {
    "baseline": "Baseline",
    "mafld_8bins": "MAF-LD 8",
    "mafld_24bins": "MAF-LD 24",
    "single": "Single",
}
ANNOT_COLOR = {
    "baseline": TAB20[8],
    "mafld_8bins": TAB20[0],
    "mafld_24bins": TAB20[16],
    "single": TAB20[18],
}

TEXT_DARK = "#111827"
TEXT_MUTED = "#4B5563"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a fixed-layout main-text Venn figure for significant rg hits "
            "in a single annotation model."
        )
    )
    ap.add_argument(
        "--csv",
        default="data/real_rg/estimates.csv",
        help="Parsed rg CSV from the supplied aggregate table.",
    )
    ap.add_argument(
        "--annot",
        default=DEFAULT_ANNOT,
        help="Annotation type to plot (default: mafld_8bins).",
    )
    ap.add_argument(
        "--pops",
        nargs="+",
        default=DEFAULT_POPS,
        help="Exactly three populations, in display order (default: EUR_300k AFR SAS).",
    )
    ap.add_argument(
        "--threshold",
        choices=["nominal", "bh", "bonferroni"],
        default="bh",
        help="Significance thresholding rule (default: bh).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for nominal / Bonferroni thresholding.",
    )
    ap.add_argument(
        "--bh-q",
        type=float,
        default=0.05,
        help="FDR q for BH thresholding.",
    )
    ap.add_argument(
        "--family-size",
        type=int,
        default=DEFAULT_TEST_FAMILY_SIZE,
        help=(
            "Number of tests in each thresholding family for BH / Bonferroni "
            f"(default: {DEFAULT_TEST_FAMILY_SIZE})."
        ),
    )
    ap.add_argument(
        "--outdir",
        default="figs/supplementary",
        help="Output directory for figure files and source data.",
    )
    ap.add_argument(
        "--basename",
        default="venn_maintext",
        help="Basename for output files.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Raster DPI for PNG output.",
    )
    ap.add_argument(
        "--width",
        type=float,
        default=12.2,
        help="Figure width in inches.",
    )
    ap.add_argument(
        "--height",
        type=float,
        default=8.9,
        help="Figure height in inches.",
    )
    ap.add_argument(
        "--circle-size-mode",
        choices=["fixed", "proportional"],
        default="fixed",
        help=(
            "Circle geometry mode. 'fixed' preserves the current manuscript layout; "
            "'proportional' uses area-scaled circles within each panel."
        ),
    )
    ap.add_argument(
        "--proportional-min-radius",
        type=float,
        default=0.18,
        help=(
            "Minimum displayed circle radius in proportional mode. "
            "Used to keep very small sets legible."
        ),
    )
    ap.add_argument(
        "--supp",
        action="store_true",
        help=(
            "Build supplementary overlap figures: one per method, with top-row population overlaps "
            "for baseline, MAF-LD 24 bins, and single-component models, and bottom-row annotation "
            "overlaps for each population."
        ),
    )
    return ap.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def normal_two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    az = abs(z)
    phi = 0.5 * (1.0 + erf(az / sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return float(np.clip(p, 0.0, 1.0))


def bh_fdr_reject(
    pvals: np.ndarray, q: float = 0.05, m_total: int | None = None
) -> np.ndarray:
    reject = np.zeros_like(pvals, dtype=bool)
    finite = np.isfinite(pvals)
    pv = pvals[finite]
    if pv.size == 0:
        return reject

    order = np.argsort(pv)
    pv_sorted = pv[order]
    m = pv_sorted.size if m_total is None else int(m_total)
    if m < pv_sorted.size:
        raise ValueError(
            f"BH family size must be >= number of finite p-values (got m_total={m}, finite={pv_sorted.size})."
        )
    thresh = (np.arange(1, m + 1) / m) * q
    ok = pv_sorted <= thresh[: pv_sorted.size]
    if not np.any(ok):
        return reject

    k = int(np.max(np.where(ok)[0]))
    cutoff = pv_sorted[k]
    reject[np.where(finite)[0]] = pv <= cutoff
    return reject


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def pair_key(a: str, b: str) -> str:
    x, y = canonical_pair(a, b)
    return f"{x}||{y}"


def split_pair_key(key: str) -> Tuple[str, str]:
    left, right = key.split("||", 1)
    return left, right


def _dedupe_pairs_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    a = tmp["phen1"].astype(str).to_numpy()
    b = tmp["phen2"].astype(str).to_numpy()
    trait1 = np.empty_like(a, dtype=object)
    trait2 = np.empty_like(b, dtype=object)
    for i in range(len(a)):
        x, y = canonical_pair(a[i], b[i])
        trait1[i] = x
        trait2[i] = y
    tmp["trait1"] = trait1
    tmp["trait2"] = trait2
    tmp = tmp.sort_values(["trait1", "trait2"])
    tmp = tmp.groupby(["trait1", "trait2"], as_index=False).first()
    return tmp


def _compute_pvals(tmp: pd.DataFrame) -> np.ndarray:
    rg = pd.to_numeric(tmp["rg"], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(tmp["rg_se"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(rg) & np.isfinite(se) & (se > 0)
    p = np.full(rg.shape, np.nan, dtype=float)
    z = np.full(rg.shape, np.nan, dtype=float)
    z[valid] = rg[valid] / se[valid]
    for idx in np.where(valid)[0]:
        p[idx] = normal_two_sided_p(z[idx])
    return p


def _threshold_mask(
    p: np.ndarray,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_size: int,
) -> np.ndarray:
    if threshold == "nominal":
        return p < alpha
    if threshold == "bh":
        return bh_fdr_reject(p, q=bh_q, m_total=family_size)
    if threshold == "bonferroni":
        if family_size <= 0:
            return np.zeros_like(p, dtype=bool)
        return p < (alpha / family_size)
    raise ValueError(f"Unknown threshold: {threshold}")


def significant_hits_for_group(
    df: pd.DataFrame,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_size: int,
) -> pd.DataFrame:
    tmp = _dedupe_pairs_keep_first(df)
    p = _compute_pvals(tmp)
    sig = _threshold_mask(
        p, threshold=threshold, alpha=alpha, bh_q=bh_q, family_size=family_size
    )
    if not np.any(sig):
        return pd.DataFrame(
            columns=["trait1", "trait2", "pair_key", "rg", "rg_se", "p_value"]
        )

    out = tmp.loc[sig, ["trait1", "trait2", "rg", "rg_se"]].copy()
    out["pair_key"] = [
        pair_key(a, b)
        for a, b in out[["trait1", "trait2"]].itertuples(index=False, name=None)
    ]
    out["p_value"] = p[sig]
    out = out.sort_values(["p_value", "trait1", "trait2"]).reset_index(drop=True)
    return out[["trait1", "trait2", "pair_key", "rg", "rg_se", "p_value"]]


def load_hits(
    df: pd.DataFrame,
    *,
    pop: str,
    method: str,
    annot: str,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_size: int,
) -> pd.DataFrame:
    sub = df[
        (df["pop"] == pop) & (df["method"] == method) & (df["annot_type"] == annot)
    ].copy()
    hits = significant_hits_for_group(
        sub, threshold=threshold, alpha=alpha, bh_q=bh_q, family_size=family_size
    )
    hits.insert(0, "annot_type", annot)
    hits.insert(0, "method", method)
    hits.insert(0, "pop", pop)
    return hits


def to_set_map(hits: Dict[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    return {name: set(frame["pair_key"].astype(str)) for name, frame in hits.items()}


def population_pair_family_sizes(
    df: pd.DataFrame, pops: Sequence[str]
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pop in pops:
        sub = df[df["pop"] == pop]
        pair_keys = {
            pair_key(str(phen1), str(phen2))
            for phen1, phen2 in sub[["phen1", "phen2"]].itertuples(
                index=False, name=None
            )
        }
        out[pop] = len(pair_keys)
    return out


def mix_with_white(color: str, amount: float) -> Tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    return tuple((1.0 - amount) * rgb + amount * np.ones(3))


def darker(color: str, amount: float = 0.22) -> Tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * (1.0 - amount), 0.0, 1.0))


def text_effects() -> List[pe.AbstractPathEffect]:
    return [pe.withStroke(linewidth=1.8, foreground="white")]


def region_counts_three(sets: Sequence[Set[str]]) -> Dict[str, int]:
    a, b, c = sets
    return {
        "100": len(a - b - c),
        "010": len(b - a - c),
        "001": len(c - a - b),
        "110": len((a & b) - c),
        "101": len((a & c) - b),
        "011": len((b & c) - a),
        "111": len(a & b & c),
    }


def region_counts_two(sets: Sequence[Set[str]]) -> Dict[str, int]:
    a, b = sets
    return {
        "10": len(a - b),
        "11": len(a & b),
        "01": len(b - a),
    }


def nice_annot_label(annot: str) -> str:
    return ANNOT_LABEL.get(annot, annot.replace("_", " "))


def source_region_label(members: Sequence[str]) -> str:
    if len(members) == 1:
        return f"{members[0]} only"
    if len(members) == 2:
        return " & ".join(members)
    if len(members) == 3:
        return "Shared by all three"
    return "Unassigned"


def region_label_from_code(
    code: str,
    set_names: Sequence[str],
    display_map: Dict[str, str],
) -> str:
    members = [display_map[name] for bit, name in zip(code, set_names) if bit == "1"]
    return source_region_label(members)


def membership_table(
    *,
    panel_letter: str,
    panel_title: str,
    comparison_type: str,
    annot: str,
    set_names: Sequence[str],
    display_map: Dict[str, str],
    set_map: Dict[str, Set[str]],
) -> pd.DataFrame:
    universe = sorted(set().union(*set_map.values()))
    rows: List[Dict[str, object]] = []
    for key in universe:
        t1, t2 = split_pair_key(key)
        members = [name for name in set_names if key in set_map[name]]
        row: Dict[str, object] = {
            "panel": panel_letter,
            "panel_title": panel_title,
            "comparison_type": comparison_type,
            "annot_type": annot,
            "trait1": t1,
            "trait2": t2,
            "pair_key": key,
            "region": source_region_label([display_map[name] for name in members]),
        }
        for idx, name in enumerate(set_names, start=1):
            row[f"set_{idx}_key"] = name
            row[f"set_{idx}_name"] = display_map[name]
            row[f"set_{idx}_member"] = key in set_map[name]
        for idx in range(len(set_names) + 1, 4):
            row[f"set_{idx}_key"] = ""
            row[f"set_{idx}_name"] = ""
            row[f"set_{idx}_member"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def region_count_table(
    *,
    panel_letter: str,
    panel_title: str,
    set_names: Sequence[str],
    display_map: Dict[str, str],
    counts: Dict[str, int],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "panel": panel_letter,
                "panel_title": panel_title,
                "region_code": code,
                "region_label": region_label_from_code(code, set_names, display_map),
                "count": value,
            }
            for code, value in counts.items()
        ]
    )


def point_xy(point: object) -> Tuple[float, float]:
    return float(getattr(point, "x")), float(getattr(point, "y"))


def apply_min_radius(radii: Sequence[float], min_radius: float) -> List[float]:
    return [max(float(radius), float(min_radius)) for radius in radii]


def unit_vector(
    dx: float, dy: float, fallback: Tuple[float, float] = (1.0, 0.0)
) -> np.ndarray:
    vec = np.array([dx, dy], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array(fallback, dtype=float)
    return vec / norm


def draw_panel_header(
    ax: plt.Axes, letter: str, title: str, subtitle: str, title_color: str
) -> None:
    ax.text(
        -0.07,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.0,
        fontweight="bold",
        color=TEXT_DARK,
        clip_on=False,
    )
    ax.text(
        0.5,
        1.08,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=title_color,
        clip_on=False,
    )
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.0,
        color=TEXT_MUTED,
        clip_on=False,
    )


def add_circle(
    ax: plt.Axes,
    *,
    center: Tuple[float, float],
    radius: float,
    color: str,
    is_empty: bool = False,
    zorder: float = 1.0,
) -> None:
    edge = darker(color, 0.18)
    face = mpl.colors.to_rgba(color, 0.28 if not is_empty else 0.06)
    circle = Circle(
        center,
        radius=radius,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.9,
        linestyle=(0, (4, 2)) if is_empty else "solid",
        zorder=zorder,
    )
    ax.add_patch(circle)


def add_set_label(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    total: int,
    color: str,
    ha: str = "center",
    background: bool = False,
) -> None:
    ax.text(
        x,
        y,
        f"{label}\n{total:,} hits",
        ha=ha,
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=color,
        linespacing=1.18,
        bbox=(
            {
                "boxstyle": "round,pad=0.04,rounding_size=0.06",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.96,
            }
            if background
            else None
        ),
        zorder=6,
    )


def add_set_label_axes(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    total: int,
    color: str,
    ha: str = "center",
    background: bool = False,
) -> None:
    ax.text(
        x,
        y,
        f"{label}\n{total:,} hits",
        transform=ax.transAxes,
        ha=ha,
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=color,
        linespacing=1.18,
        clip_on=False,
        bbox=(
            {
                "boxstyle": "round,pad=0.04,rounding_size=0.06",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.96,
            }
            if background
            else None
        ),
        zorder=6,
    )


def add_region_count(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    value: int,
    show_zero: bool = False,
) -> None:
    if value == 0 and not show_zero:
        return
    ax.text(
        x,
        y,
        f"{value:,}",
        ha="center",
        va="center",
        fontsize=12.6 if value < 1000 else 11.8,
        fontweight="bold",
        color=TEXT_DARK if value > 0 else TEXT_MUTED,
        path_effects=text_effects(),
        zorder=5,
    )


def add_circle_outline(
    ax: plt.Axes,
    *,
    center: Tuple[float, float],
    radius: float,
    color: str,
    is_empty: bool = False,
    zorder: float = 4.0,
) -> None:
    edge = darker(color, 0.18)
    face = mpl.colors.to_rgba(color, 0.04) if is_empty else (0.0, 0.0, 0.0, 0.0)
    circle = Circle(
        center,
        radius=radius,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.9,
        linestyle=(0, (4, 2)) if is_empty else "solid",
        zorder=zorder,
    )
    ax.add_patch(circle)


def draw_fixed_three_set_panel(
    ax: plt.Axes,
    *,
    letter: str,
    title: str,
    subtitle: str,
    title_color: str,
    set_names: Sequence[str],
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
    label_specs: Sequence[Tuple[float, float, str]] | None = None,
    label_background_mask: Sequence[bool] | None = None,
) -> Dict[str, int]:
    draw_panel_header(ax, letter, title, subtitle, title_color)

    ax.set_aspect("equal")
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-2.35, 2.28)
    ax.axis("off")

    centers = [(-0.96, 0.42), (0.96, 0.42), (0.0, -0.82)]
    radius = 1.42

    for idx in [2, 0, 1]:
        add_circle(
            ax,
            center=centers[idx],
            radius=radius,
            color=set_colors[idx],
            is_empty=len(set_values[idx]) == 0,
            zorder=1.0 + 0.1 * idx,
        )

    specs = (
        list(label_specs)
        if label_specs is not None
        else [
            (-1.18, 2.02, "right"),
            (1.44, 2.02, "left"),
            (0.0, -2.0, "center"),
        ]
    )
    backgrounds = (
        list(label_background_mask)
        if label_background_mask is not None
        else [False] * len(set_labels)
    )
    for (x, y, ha), label, color, values, background in zip(
        specs,
        set_labels,
        set_colors,
        set_values,
        backgrounds,
    ):
        add_set_label(
            ax,
            x=x,
            y=y,
            label=label,
            total=len(values),
            color=color,
            ha=ha,
            background=background,
        )

    counts = region_counts_three(set_values)
    region_pos = {
        "100": (-1.53, 0.70),
        "010": (1.53, 0.70),
        "001": (0.00, -1.44),
        "110": (0.00, 0.98),
        "101": (-0.79, -0.30),
        "011": (0.79, -0.30),
        "111": (0.00, 0.14),
    }
    for code, (x, y) in region_pos.items():
        add_region_count(ax, x=x, y=y, value=counts[code], show_zero=False)
    return counts


def draw_fixed_two_set_panel(
    ax: plt.Axes,
    *,
    letter: str,
    title: str,
    subtitle: str,
    title_color: str,
    set_names: Sequence[str],
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
) -> Dict[str, int]:
    draw_panel_header(ax, letter, title, subtitle, title_color)

    ax.set_aspect("equal")
    ax.set_xlim(-2.52, 2.52)
    ax.set_ylim(-2.05, 2.28)
    ax.axis("off")

    centers = [(-0.82, 0.0), (0.82, 0.0)]
    radius = 1.38

    for idx in [0, 1]:
        add_circle(
            ax,
            center=centers[idx],
            radius=radius,
            color=set_colors[idx],
            is_empty=len(set_values[idx]) == 0,
            zorder=1.0 + 0.1 * idx,
        )

    label_positions = [(-1.56, 1.74), (1.56, 1.74)]
    for (x, y), label, color, values in zip(
        label_positions, set_labels, set_colors, set_values
    ):
        add_set_label(ax, x=x, y=y, label=label, total=len(values), color=color)

    counts = region_counts_two(set_values)
    region_pos = {
        "10": (-1.22, 0.02),
        "11": (0.00, 0.02),
        "01": (1.22, 0.02),
    }
    for code, (x, y) in region_pos.items():
        add_region_count(ax, x=x, y=y, value=counts[code], show_zero=False)
    return counts


def style_venn_region_labels(venn_obj, *, region_ids: Sequence[str]) -> None:
    for region_id in region_ids:
        label = venn_obj.get_label_by_id(region_id)
        if label is None or not label.get_text().strip():
            continue
        label.set_fontsize(13.0 if len(label.get_text()) < 4 else 12.0)
        label.set_fontweight("bold")
        label.set_color(TEXT_DARK if label.get_text().strip() != "0" else TEXT_MUTED)
        label.set_path_effects(text_effects())


def style_venn_patches(venn_obj, *, region_ids: Sequence[str]) -> None:
    for region_id in region_ids:
        patch = venn_obj.get_patch_by_id(region_id)
        if patch is None:
            continue
        patch.set_alpha(0.34)
        patch.set_edgecolor("none")
        patch.set_linewidth(0.0)


def set_axis_limits_from_circles(
    ax: plt.Axes,
    *,
    centers: Sequence[Tuple[float, float]],
    radii: Sequence[float],
    x_pad: float,
    y_top_pad: float,
    y_bottom_pad: float,
) -> None:
    xs = [x for x, _ in centers]
    ys = [y for _, y in centers]
    xmin = min(x - r for x, r in zip(xs, radii))
    xmax = max(x + r for x, r in zip(xs, radii))
    ymin = min(y - r for y, r in zip(ys, radii))
    ymax = max(y + r for y, r in zip(ys, radii))
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_bottom_pad, ymax + y_top_pad)


def add_proportional_three_set_labels(
    ax: plt.Axes,
    *,
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
    positions: Sequence[Tuple[float, float, str]] | None = None,
    background_mask: Sequence[bool] | None = None,
) -> None:
    specs = (
        list(positions)
        if positions is not None
        else [
            (0.34, 0.79, "center"),
            (0.80, 0.67, "left"),
            (0.80, 0.44, "left"),
        ]
    )
    backgrounds = (
        list(background_mask)
        if background_mask is not None
        else [False] * len(set_labels)
    )
    for (x, y, ha), label, color, values, background in zip(
        specs,
        set_labels,
        set_colors,
        set_values,
        backgrounds,
    ):
        add_set_label_axes(
            ax,
            x=x,
            y=y,
            label=label,
            total=len(values),
            color=color,
            ha=ha,
            background=background,
        )


def add_proportional_two_set_labels(
    ax: plt.Axes,
    *,
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
) -> None:
    positions = [
        (0.19, 0.67, "center"),
        (0.82, 0.67, "center"),
    ]
    for (x, y, ha), label, color, values in zip(
        positions, set_labels, set_colors, set_values
    ):
        add_set_label_axes(
            ax, x=x, y=y, label=label, total=len(values), color=color, ha=ha
        )


def add_missing_two_set_labels(
    ax: plt.Axes,
    *,
    venn_obj,
    counts: Dict[str, int],
    centers: Sequence[Tuple[float, float]],
    radii: Sequence[float],
) -> None:
    display_radii = [max(radius, 0.11) for radius in radii]
    x_left, y_left = centers[0]
    x_right, y_right = centers[1]
    positions = {
        "10": (x_left - 0.34 * display_radii[0], y_left),
        "11": ((x_left + x_right) / 2.0, (y_left + y_right) / 2.0),
        "01": (x_right + 0.34 * display_radii[1], y_right),
    }
    for region_id, value in counts.items():
        label = venn_obj.get_label_by_id(region_id)
        if label is not None and label.get_text().strip():
            continue
        x, y = positions[region_id]
        add_region_count(ax, x=x, y=y, value=value, show_zero=False)


def add_proportional_three_region_counts(
    ax: plt.Axes,
    *,
    counts: Dict[str, int],
    centers: Sequence[Tuple[float, float]],
    radii: Sequence[float],
) -> None:
    pts = [np.array(center, dtype=float) for center in centers]
    rs = np.array([max(radius, 0.11) for radius in radii], dtype=float)
    centroid = sum(pts) / 3.0

    def outward_anchor(i: int) -> np.ndarray:
        others = (pts[(i + 1) % 3] + pts[(i + 2) % 3]) / 2.0
        direction = unit_vector(*(pts[i] - others))
        return pts[i] + direction * (0.58 * rs[i])

    def pair_anchor(i: int, j: int, k: int) -> np.ndarray:
        midpoint = (pts[i] + pts[j]) / 2.0
        direction = unit_vector(
            *(midpoint - pts[k]), fallback=tuple(unit_vector(*(midpoint - centroid)))
        )
        return midpoint + direction * (0.24 * min(rs[i], rs[j]))

    anchors = {
        "100": outward_anchor(0),
        "010": outward_anchor(1),
        "001": outward_anchor(2),
        "110": pair_anchor(0, 1, 2),
        "101": pair_anchor(0, 2, 1),
        "011": pair_anchor(1, 2, 0),
        "111": centroid.copy(),
    }
    positions = {code: anchor.copy() for code, anchor in anchors.items()}
    codes = ["100", "010", "001", "110", "101", "011", "111"]
    min_sep = max(0.20, 0.36 * float(np.median(rs)))
    max_disp = max(0.28, 0.78 * float(np.max(rs)))

    for _ in range(60):
        moved = False
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                code_i = codes[i]
                code_j = codes[j]
                delta = positions[code_j] - positions[code_i]
                dist = float(np.linalg.norm(delta))
                if dist >= min_sep:
                    continue
                direction = unit_vector(
                    delta[0],
                    delta[1],
                    fallback=(1.0, 0.25 * (j - i)),
                )
                shift = 0.5 * (min_sep - dist) * direction
                if code_i != "111":
                    positions[code_i] -= shift
                if code_j != "111":
                    positions[code_j] += shift
                moved = True

        for code in codes:
            if code == "111":
                continue
            delta = positions[code] - anchors[code]
            dist = float(np.linalg.norm(delta))
            if dist > max_disp:
                positions[code] = anchors[code] + (delta / dist) * max_disp

        if not moved:
            break

    for code in codes:
        x, y = positions[code]
        add_region_count(
            ax, x=float(x), y=float(y), value=counts[code], show_zero=False
        )


def draw_proportional_three_set_panel(
    ax: plt.Axes,
    *,
    letter: str,
    title: str,
    subtitle: str,
    title_color: str,
    set_names: Sequence[str],
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
    min_radius: float,
    label_positions: Sequence[Tuple[float, float, str]] | None = None,
    label_background_mask: Sequence[bool] | None = None,
) -> Dict[str, int]:
    if venn3 is None:
        raise SystemExit(
            "ERROR: --circle-size-mode proportional requires matplotlib-venn."
        )

    draw_panel_header(ax, letter, title, subtitle, title_color)
    ax.set_aspect("equal")
    ax.axis("off")

    counts = region_counts_three(set_values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Circle .* has zero area.")
        warnings.filterwarnings("ignore", message="Bad circle positioning.")
        warnings.filterwarnings("ignore", message="All circles have zero area.")
        venn_obj = venn3(
            subsets=counts,
            set_labels=None,
            set_colors=tuple(set_colors),
            alpha=0.34,
            ax=ax,
            subset_label_formatter=lambda value: "",
        )

    style_venn_patches(
        venn_obj, region_ids=["100", "010", "110", "001", "101", "011", "111"]
    )

    centers = [point_xy(venn_obj.get_circle_center(i)) for i in range(3)]
    radii = [float(venn_obj.get_circle_radius(i)) for i in range(3)]
    display_radii = apply_min_radius(radii, min_radius=min_radius)

    for center, radius, color, values in zip(
        centers, display_radii, set_colors, set_values
    ):
        add_circle_outline(
            ax, center=center, radius=radius, color=color, is_empty=len(values) == 0
        )

    add_proportional_three_region_counts(
        ax,
        counts=counts,
        centers=centers,
        radii=display_radii,
    )
    add_proportional_three_set_labels(
        ax,
        set_labels=set_labels,
        set_colors=set_colors,
        set_values=set_values,
        positions=label_positions,
        background_mask=label_background_mask,
    )
    set_axis_limits_from_circles(
        ax,
        centers=centers,
        radii=display_radii,
        x_pad=0.32,
        y_top_pad=0.58,
        y_bottom_pad=0.55,
    )
    return counts


def draw_proportional_two_set_panel(
    ax: plt.Axes,
    *,
    letter: str,
    title: str,
    subtitle: str,
    title_color: str,
    set_names: Sequence[str],
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
    min_radius: float,
) -> Dict[str, int]:
    if venn2 is None:
        raise SystemExit(
            "ERROR: --circle-size-mode proportional requires matplotlib-venn."
        )

    draw_panel_header(ax, letter, title, subtitle, title_color)
    ax.set_aspect("equal")
    ax.axis("off")

    counts = region_counts_two(set_values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Circle .* has zero area.")
        warnings.filterwarnings("ignore", message="Bad circle positioning.")
        warnings.filterwarnings("ignore", message="All circles have zero area.")
        venn_obj = venn2(
            subsets=counts,
            set_labels=None,
            set_colors=tuple(set_colors),
            alpha=0.34,
            ax=ax,
            subset_label_formatter=lambda value: f"{int(round(value)):,}"
            if value > 0
            else "",
        )

    style_venn_patches(venn_obj, region_ids=["10", "11", "01"])
    style_venn_region_labels(venn_obj, region_ids=["10", "11", "01"])

    centers = [point_xy(venn_obj.get_circle_center(i)) for i in range(2)]
    radii = [float(venn_obj.get_circle_radius(i)) for i in range(2)]
    display_radii = apply_min_radius(radii, min_radius=min_radius)

    for center, radius, color, values in zip(
        centers, display_radii, set_colors, set_values
    ):
        add_circle_outline(
            ax, center=center, radius=radius, color=color, is_empty=len(values) == 0
        )

    add_missing_two_set_labels(
        ax, venn_obj=venn_obj, counts=counts, centers=centers, radii=display_radii
    )
    add_proportional_two_set_labels(
        ax, set_labels=set_labels, set_colors=set_colors, set_values=set_values
    )
    set_axis_limits_from_circles(
        ax,
        centers=centers,
        radii=display_radii,
        x_pad=0.32,
        y_top_pad=0.62,
        y_bottom_pad=0.28,
    )
    return counts


def draw_three_set_panel(
    ax: plt.Axes,
    *,
    letter: str,
    title: str,
    subtitle: str,
    title_color: str,
    set_names: Sequence[str],
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
    circle_size_mode: str,
    proportional_min_radius: float,
    fixed_label_specs: Sequence[Tuple[float, float, str]] | None = None,
    proportional_label_positions: Sequence[Tuple[float, float, str]] | None = None,
    label_background_mask: Sequence[bool] | None = None,
) -> Dict[str, int]:
    if circle_size_mode == "proportional":
        return draw_proportional_three_set_panel(
            ax,
            letter=letter,
            title=title,
            subtitle=subtitle,
            title_color=title_color,
            set_names=set_names,
            set_labels=set_labels,
            set_colors=set_colors,
            set_values=set_values,
            min_radius=proportional_min_radius,
            label_positions=proportional_label_positions,
            label_background_mask=label_background_mask,
        )
    return draw_fixed_three_set_panel(
        ax,
        letter=letter,
        title=title,
        subtitle=subtitle,
        title_color=title_color,
        set_names=set_names,
        set_labels=set_labels,
        set_colors=set_colors,
        set_values=set_values,
        label_specs=fixed_label_specs,
        label_background_mask=label_background_mask,
    )


def draw_two_set_panel(
    ax: plt.Axes,
    *,
    letter: str,
    title: str,
    subtitle: str,
    title_color: str,
    set_names: Sequence[str],
    set_labels: Sequence[str],
    set_colors: Sequence[str],
    set_values: Sequence[Set[str]],
    circle_size_mode: str,
    proportional_min_radius: float,
) -> Dict[str, int]:
    if circle_size_mode == "proportional":
        return draw_proportional_two_set_panel(
            ax,
            letter=letter,
            title=title,
            subtitle=subtitle,
            title_color=title_color,
            set_names=set_names,
            set_labels=set_labels,
            set_colors=set_colors,
            set_values=set_values,
            min_radius=proportional_min_radius,
        )
    return draw_fixed_two_set_panel(
        ax,
        letter=letter,
        title=title,
        subtitle=subtitle,
        title_color=title_color,
        set_names=set_names,
        set_labels=set_labels,
        set_colors=set_colors,
        set_values=set_values,
    )


def validate_inputs(args: argparse.Namespace, df: pd.DataFrame) -> None:
    if len(args.pops) != 3:
        raise SystemExit(
            f"ERROR: expected exactly 3 populations for this figure, got {len(args.pops)}."
        )

    missing = sorted(REQUIRED_COLS.difference(df.columns))
    if missing:
        raise SystemExit(f"ERROR: CSV missing required columns: {missing}")

    unknown_pops = [pop for pop in args.pops if pop not in POP_LABEL]
    if unknown_pops:
        raise SystemExit(
            f"ERROR: missing pretty labels/colors for populations: {unknown_pops}"
        )

    if args.circle_size_mode == "proportional" and (venn2 is None or venn3 is None):
        raise SystemExit(
            "ERROR: --circle-size-mode proportional requires matplotlib-venn."
        )

    if args.proportional_min_radius <= 0:
        raise SystemExit("ERROR: --proportional-min-radius must be > 0.")

    if args.family_size <= 0:
        raise SystemExit("ERROR: --family-size must be > 0.")

    annots_present = (
        set(df["annot_type"].astype(str).unique())
        if "annot_type" in df.columns
        else set()
    )
    required_annots = (
        set(SUPP_TOP_ANNOTS + SUPP_BOTTOM_ANNOTS) if args.supp else {args.annot}
    )
    missing_annots = sorted(required_annots - annots_present)
    if missing_annots:
        raise SystemExit(
            f"ERROR: required annotations not found in input CSV: {missing_annots}"
        )


def load_all_panel_hits(
    df: pd.DataFrame,
    *,
    pops: Sequence[str],
    annot: str,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_sizes: Dict[str, int],
) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for pop in pops:
        for method in sorted(set(TOP_ROW_METHODS + BOTTOM_ROW_METHODS)):
            out[(pop, method)] = load_hits(
                df,
                pop=pop,
                method=method,
                annot=annot,
                threshold=threshold,
                alpha=alpha,
                bh_q=bh_q,
                family_size=family_sizes[pop],
            )
    return out


def load_hit_cache(
    df: pd.DataFrame,
    *,
    pops: Sequence[str],
    methods: Sequence[str],
    annots: Sequence[str],
    threshold: str,
    alpha: float,
    bh_q: float,
    family_sizes: Dict[str, int],
) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for pop in pops:
        for method in methods:
            for annot in annots:
                out[(pop, method, annot)] = load_hits(
                    df,
                    pop=pop,
                    method=method,
                    annot=annot,
                    threshold=threshold,
                    alpha=alpha,
                    bh_q=bh_q,
                    family_size=family_sizes[pop],
                )
    return out


def build_figure(
    *,
    hits_by_group: Dict[Tuple[str, str], pd.DataFrame],
    pops: Sequence[str],
    annot: str,
    significance_label: str,
    footer_text: str,
    circle_size_mode: str,
    proportional_min_radius: float,
    width: float,
    height: float,
) -> Tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.16, 1.0],
        hspace=0.11,
        wspace=0.08,
    )
    axes = {
        "A": fig.add_subplot(gs[0, 0:3]),
        "B": fig.add_subplot(gs[0, 3:6]),
        "C": fig.add_subplot(gs[1, 0:2]),
        "D": fig.add_subplot(gs[1, 2:4]),
        "E": fig.add_subplot(gs[1, 4:6]),
    }

    source_tables: List[pd.DataFrame] = []
    count_tables: List[pd.DataFrame] = []
    bottom_row_pops = [pop for pop in BOTTOM_PANEL_POPS if pop in pops]

    top_subtitle = f"Population overlap of {significance_label} trait pairs"
    bottom_subtitle = "SUMMIT vs cov-LDSC"

    for panel_letter, method in zip(["A", "B"], TOP_ROW_METHODS):
        title = METHOD_LABEL[method]
        set_names = list(pops)
        display_map = {pop: POP_LABEL[pop] for pop in pops}
        set_labels = [POP_LABEL[pop] for pop in pops]
        set_colors = [POP_COLOR[pop] for pop in pops]
        hit_map = {pop: hits_by_group[(pop, method)] for pop in pops}
        set_map = to_set_map(hit_map)
        set_values = [set_map[name] for name in set_names]
        counts = draw_three_set_panel(
            axes[panel_letter],
            letter=panel_letter,
            title=title,
            subtitle=top_subtitle,
            title_color=METHOD_COLOR[method],
            set_names=set_names,
            set_labels=set_labels,
            set_colors=set_colors,
            set_values=set_values,
            circle_size_mode=circle_size_mode,
            proportional_min_radius=proportional_min_radius,
        )
        source_tables.append(
            membership_table(
                panel_letter=panel_letter,
                panel_title=title,
                comparison_type="population_overlap",
                annot=annot,
                set_names=set_names,
                display_map=display_map,
                set_map=set_map,
            )
        )
        count_tables.append(
            region_count_table(
                panel_letter=panel_letter,
                panel_title=title,
                set_names=set_names,
                display_map=display_map,
                counts=counts,
            )
        )

    for panel_letter, pop in zip(["C", "D", "E"], bottom_row_pops):
        title = POP_LABEL[pop]
        set_names = list(BOTTOM_ROW_METHODS)
        display_map = {method: METHOD_LABEL[method] for method in set_names}
        set_labels = [METHOD_LABEL[method] for method in set_names]
        set_colors = [METHOD_COLOR[method] for method in set_names]
        hit_map = {method: hits_by_group[(pop, method)] for method in set_names}
        set_map = to_set_map(hit_map)
        set_values = [set_map[name] for name in set_names]
        counts = draw_two_set_panel(
            axes[panel_letter],
            letter=panel_letter,
            title=title,
            subtitle=bottom_subtitle,
            title_color=POP_COLOR[pop],
            set_names=set_names,
            set_labels=set_labels,
            set_colors=set_colors,
            set_values=set_values,
            circle_size_mode=circle_size_mode,
            proportional_min_radius=proportional_min_radius,
        )
        source_tables.append(
            membership_table(
                panel_letter=panel_letter,
                panel_title=title,
                comparison_type="method_overlap",
                annot=annot,
                set_names=set_names,
                display_map=display_map,
                set_map=set_map,
            )
        )
        count_tables.append(
            region_count_table(
                panel_letter=panel_letter,
                panel_title=title,
                set_names=set_names,
                display_map=display_map,
                counts=counts,
            )
        )

    if footer_text:
        fig.text(
            0.5,
            0.025,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=TEXT_MUTED,
        )
    return (
        fig,
        pd.concat(source_tables, ignore_index=True),
        pd.concat(count_tables, ignore_index=True),
    )


def add_figure_method_header(fig: plt.Figure, method: str) -> None:
    fig.text(
        0.5,
        0.995,
        METHOD_LABEL[method],
        ha="center",
        va="top",
        fontsize=15.0,
        fontweight="bold",
        color=METHOD_COLOR[method],
    )


def build_supp_figure(
    *,
    hit_cache: Dict[Tuple[str, str, str], pd.DataFrame],
    method: str,
    pops: Sequence[str],
    significance_label: str,
    footer_text: str,
    circle_size_mode: str,
    proportional_min_radius: float,
    width: float,
    height: float,
) -> Tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    fig = plt.figure(figsize=(max(width, 14.8), max(height, 10.1)))
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.08, 1.08],
        hspace=0.08,
        wspace=0.10,
    )
    axes = {
        "A": fig.add_subplot(gs[0, 0:2]),
        "B": fig.add_subplot(gs[0, 2:4]),
        "C": fig.add_subplot(gs[0, 4:6]),
        "D": fig.add_subplot(gs[1, 0:2]),
        "E": fig.add_subplot(gs[1, 2:4]),
        "F": fig.add_subplot(gs[1, 4:6]),
    }

    add_figure_method_header(fig, method)

    source_tables: List[pd.DataFrame] = []
    count_tables: List[pd.DataFrame] = []
    bottom_row_pops = [pop for pop in BOTTOM_PANEL_POPS if pop in pops]

    top_subtitle = f"Population overlap of {significance_label} trait pairs"
    bottom_subtitle = "Overlap across annotations"
    fixed_population_label_specs = [
        (-0.96, 1.34, "center"),
        (0.96, 1.34, "center"),
        (0.0, -1.88, "center"),
    ]
    proportional_population_label_positions = [
        (0.30, 0.58, "center"),
        (0.70, 0.58, "center"),
        (0.50, 0.14, "center"),
    ]
    fixed_annotation_label_specs = [
        (-0.96, 1.34, "center"),
        (0.96, 1.34, "center"),
        (0.0, -1.88, "center"),
    ]
    proportional_annotation_label_positions = [
        (0.30, 0.58, "center"),
        (0.70, 0.58, "center"),
        (0.50, 0.14, "center"),
    ]

    for panel_letter, annot in zip(["A", "B", "C"], SUPP_TOP_ANNOTS):
        title = ANNOT_PANEL_LABEL[annot]
        set_names = list(pops)
        display_map = {pop: POP_LABEL[pop] for pop in pops}
        set_labels = [POP_LABEL[pop] for pop in pops]
        set_colors = [POP_COLOR[pop] for pop in pops]
        hit_map = {pop: hit_cache[(pop, method, annot)] for pop in pops}
        set_map = to_set_map(hit_map)
        set_values = [set_map[name] for name in set_names]
        counts = draw_three_set_panel(
            axes[panel_letter],
            letter=panel_letter,
            title=title,
            subtitle=top_subtitle,
            title_color=ANNOT_COLOR[annot],
            set_names=set_names,
            set_labels=set_labels,
            set_colors=set_colors,
            set_values=set_values,
            circle_size_mode=circle_size_mode,
            proportional_min_radius=proportional_min_radius,
            fixed_label_specs=fixed_population_label_specs,
            proportional_label_positions=proportional_population_label_positions,
        )
        source_tables.append(
            membership_table(
                panel_letter=panel_letter,
                panel_title=title,
                comparison_type="population_overlap",
                annot=annot,
                set_names=set_names,
                display_map=display_map,
                set_map=set_map,
            )
        )
        count_tables.append(
            region_count_table(
                panel_letter=panel_letter,
                panel_title=title,
                set_names=set_names,
                display_map=display_map,
                counts=counts,
            )
        )

    for panel_letter, pop in zip(["D", "E", "F"], bottom_row_pops):
        title = POP_LABEL[pop]
        set_names = list(SUPP_BOTTOM_ANNOTS)
        display_map = {annot: ANNOT_PANEL_LABEL[annot] for annot in set_names}
        set_labels = [ANNOT_SET_LABEL[annot] for annot in set_names]
        set_colors = [ANNOT_COLOR[annot] for annot in set_names]
        hit_map = {annot: hit_cache[(pop, method, annot)] for annot in set_names}
        set_map = to_set_map(hit_map)
        set_values = [set_map[name] for name in set_names]
        counts = draw_three_set_panel(
            axes[panel_letter],
            letter=panel_letter,
            title=title,
            subtitle=bottom_subtitle,
            title_color=POP_COLOR[pop],
            set_names=set_names,
            set_labels=set_labels,
            set_colors=set_colors,
            set_values=set_values,
            circle_size_mode=circle_size_mode,
            proportional_min_radius=proportional_min_radius,
            fixed_label_specs=fixed_annotation_label_specs,
            proportional_label_positions=proportional_annotation_label_positions,
        )
        source_tables.append(
            membership_table(
                panel_letter=panel_letter,
                panel_title=title,
                comparison_type="annotation_overlap",
                annot="__".join(set_names),
                set_names=set_names,
                display_map=display_map,
                set_map=set_map,
            )
        )
        count_tables.append(
            region_count_table(
                panel_letter=panel_letter,
                panel_title=title,
                set_names=set_names,
                display_map=display_map,
                counts=counts,
            )
        )

    if footer_text:
        fig.text(
            0.5,
            0.02,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=TEXT_MUTED,
        )
    return (
        fig,
        pd.concat(source_tables, ignore_index=True),
        pd.concat(count_tables, ignore_index=True),
    )


def save_figure_outputs(
    *,
    fig: plt.Figure,
    source_data: pd.DataFrame,
    region_counts: pd.DataFrame,
    outdir: Path,
    stem: str,
    dpi: int,
) -> None:
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    source_path = outdir / f"{stem}__source_data.csv"
    counts_path = outdir / f"{stem}__region_counts.csv"

    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    source_data.to_csv(source_path, index=False)
    region_counts.to_csv(counts_path, index=False)

    print(f"[write] {png_path}")
    print(f"[write] {pdf_path}")
    print(f"[write] {source_path}")
    print(f"[write] {counts_path}")


def threshold_note(threshold: str, alpha: float, bh_q: float) -> str:
    if threshold == "bh":
        return f"BH FDR q <= {bh_q:g}"
    if threshold == "bonferroni":
        return f"Bonferroni alpha = {alpha:g}"
    return f"Nominal alpha = {alpha:g}"


def significance_label(threshold: str) -> str:
    if threshold == "bh":
        return "FDR-significant"
    if threshold == "bonferroni":
        return "Bonferroni-significant"
    if threshold == "nominal":
        return "nominally significant"
    raise ValueError(f"Unknown threshold: {threshold}")


def main() -> None:
    args = parse_args()
    set_style()

    df = pd.read_csv(args.csv)
    validate_inputs(args, df)

    df["pop"] = df["pop"].astype(str)
    df["method"] = df["method"].astype(str)
    df["annot_type"] = df["annot_type"].astype(str)
    df["phen1"] = df["phen1"].astype(str)
    df["phen2"] = df["phen2"].astype(str)

    observed_family_sizes = population_pair_family_sizes(df, args.pops)
    family_sizes = {pop: args.family_size for pop in args.pops}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for pop in args.pops:
        print(
            f"[family] {POP_LABEL[pop]}: threshold_tests={family_sizes[pop]:,} "
            f"observed_trait_pairs={observed_family_sizes[pop]:,}"
        )
        if observed_family_sizes[pop] != family_sizes[pop]:
            print(
                f"[warning] {POP_LABEL[pop]} observed_trait_pairs differs from "
                f"--family-size; using threshold_tests={family_sizes[pop]:,}."
            )

    if args.supp:
        hit_cache = load_hit_cache(
            df,
            pops=args.pops,
            methods=SUPP_METHODS,
            annots=sorted(set(SUPP_TOP_ANNOTS + SUPP_BOTTOM_ANNOTS)),
            threshold=args.threshold,
            alpha=args.alpha,
            bh_q=args.bh_q,
            family_sizes=family_sizes,
        )

        for method in SUPP_METHODS:
            fig, source_data, region_counts = build_supp_figure(
                hit_cache=hit_cache,
                method=method,
                pops=args.pops,
                significance_label=significance_label(args.threshold),
                footer_text="",
                circle_size_mode=args.circle_size_mode,
                proportional_min_radius=args.proportional_min_radius,
                width=args.width,
                height=args.height,
            )
            source_data.insert(0, "figure_mode", "supp")
            region_counts.insert(0, "figure_mode", "supp")
            source_data.insert(1, "figure_method", method)
            region_counts.insert(1, "figure_method", method)
            source_data.insert(2, "circle_size_mode", args.circle_size_mode)
            region_counts.insert(2, "circle_size_mode", args.circle_size_mode)
            source_data.insert(
                3, "proportional_min_radius", args.proportional_min_radius
            )
            region_counts.insert(
                3, "proportional_min_radius", args.proportional_min_radius
            )
            source_data.insert(4, "threshold_family_size", args.family_size)
            region_counts.insert(4, "threshold_family_size", args.family_size)

            stem = f"{args.basename}__supp__{method}__{args.threshold}"
            if args.circle_size_mode != "fixed":
                stem += f"__{args.circle_size_mode}"
            save_figure_outputs(
                fig=fig,
                source_data=source_data,
                region_counts=region_counts,
                outdir=outdir,
                stem=stem,
                dpi=args.dpi,
            )

            for pop in args.pops:
                top_counts = {
                    annot: len(hit_cache[(pop, method, annot)])
                    for annot in SUPP_BOTTOM_ANNOTS
                }
                print(
                    f"[counts] {METHOD_LABEL[method]} {POP_LABEL[pop]}: "
                    f"MAF-LD 8={top_counts['mafld_8bins']:,} "
                    f"MAF-LD 24={top_counts['mafld_24bins']:,} "
                    f"baseline={top_counts['baseline']:,}"
                )
    else:
        hits_by_group = load_all_panel_hits(
            df,
            pops=args.pops,
            annot=args.annot,
            threshold=args.threshold,
            alpha=args.alpha,
            bh_q=args.bh_q,
            family_sizes=family_sizes,
        )

        fig, source_data, region_counts = build_figure(
            hits_by_group=hits_by_group,
            pops=args.pops,
            annot=args.annot,
            significance_label=significance_label(args.threshold),
            footer_text="",
            circle_size_mode=args.circle_size_mode,
            proportional_min_radius=args.proportional_min_radius,
            width=args.width,
            height=args.height,
        )

        source_data.insert(0, "figure_mode", "main")
        region_counts.insert(0, "figure_mode", "main")
        source_data.insert(1, "circle_size_mode", args.circle_size_mode)
        region_counts.insert(1, "circle_size_mode", args.circle_size_mode)
        source_data.insert(2, "proportional_min_radius", args.proportional_min_radius)
        region_counts.insert(2, "proportional_min_radius", args.proportional_min_radius)
        source_data.insert(3, "threshold_family_size", args.family_size)
        region_counts.insert(3, "threshold_family_size", args.family_size)

        stem = f"{args.basename}__{args.annot}__{args.threshold}"
        if args.circle_size_mode != "fixed":
            stem += f"__{args.circle_size_mode}"
        save_figure_outputs(
            fig=fig,
            source_data=source_data,
            region_counts=region_counts,
            outdir=outdir,
            stem=stem,
            dpi=args.dpi,
        )

        for pop in args.pops:
            summit_n = len(hits_by_group[(pop, "summit")])
            covldsc_n = len(hits_by_group[(pop, "covldsc")])
            overlap_n = len(
                set(hits_by_group[(pop, "summit")]["pair_key"].astype(str))
                & set(hits_by_group[(pop, "covldsc")]["pair_key"].astype(str))
            )
            print(
                f"[counts] {POP_LABEL[pop]}: "
                f"SUMMIT={summit_n:,} cov-LDSC={covldsc_n:,} overlap={overlap_n:,}"
            )


if __name__ == "__main__":
    main()
