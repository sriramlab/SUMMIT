#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from venn_maintext import load_hits, normal_two_sided_p


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = Path("data/real_rg/estimates.csv")
DEFAULT_OUTDIR = Path("figs/main")
DEFAULT_REF_POP = "EUR_300k"
DEFAULT_SMALL_POPS = ["EUR", "SAS", "AFR"]
DEFAULT_METHOD = "summit"
DEFAULT_ANNOT = "mafld_8bins"
DEFAULT_TEST_FAMILY_SIZE = 780

TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "SAS": "SAS",
    "AFR": "AFR",
}
POP_COLOR = {
    "EUR_300k": TAB20[1],
    "EUR": TAB20[0],
    "SAS": TAB20[4],
    "AFR": TAB20[6],
}
METHOD_LABEL = {
    "summit": "SUMMIT",
    "covldsc": "cov-LDSC",
    "summit_unc": "Unconstrained SUMMIT",
}
ANNOT_LABEL = {
    "baseline": "Baseline",
    "mafld_8bins": "MAF-LD 8 bins",
    "mafld_24bins": "MAF-LD 24 bins",
    "single": "Single component",
}
TEXT_DARK = "#111827"
TEXT_MUTED = "#4B5563"
LABEL_FONT_SIZE = 7.2
LABEL_EDGE_PAD = 0.018
LABEL_MARGIN = 0.010
METRIC_BOX_RECT = (0.018, 0.885, 0.225, 0.990)
MIN_LEADER_LEN = 0.025
MANUAL_LABEL_POSITIONS = {
    "EUR": {
        "HbA1c-WHR": (0.43, 0.82, "right"),
        "T2D-WHR": (0.55, 0.90, "right"),
        "HDL-HTN": (0.46, 0.135, "left"),
    },
    "SAS": {
        "WBC-WHR": (0.49, 0.77, "right"),
        "CRT-EA": (0.36, 0.65, "right"),
        "EA-FI": (0.82, 0.60, "left"),
        "ALB-BMI": (0.43, 0.105, "left"),
    },
    "AFR": {
        "ApoB-LDL": (0.82, 0.94, "left"),
        "SBP-DBP": (0.82, 0.72, "left"),
        "ALT-GGT": (0.64, 0.78, "right"),
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Scatter smaller-cohort FDR-significant SUMMIT hits against the matching "
            "EUR_300k SUMMIT estimates."
        )
    )
    ap.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Parsed rg CSV from the supplied aggregate table.",
    )
    ap.add_argument(
        "--ref-pop", default=DEFAULT_REF_POP, help="Reference cohort on the x-axis."
    )
    ap.add_argument(
        "--small-pops",
        nargs="+",
        default=DEFAULT_SMALL_POPS,
        help="Smaller cohorts to plot against --ref-pop.",
    )
    ap.add_argument("--method", default=DEFAULT_METHOD, help="Method to compare.")
    ap.add_argument("--annot", default=DEFAULT_ANNOT, help="Annotation type to use.")
    ap.add_argument(
        "--threshold",
        choices=["nominal", "bh", "bonferroni"],
        default="bh",
        help="Significance rule used to define smaller-cohort hits.",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.05, help="Alpha for nominal / Bonferroni."
    )
    ap.add_argument("--bh-q", type=float, default=0.05, help="BH FDR q threshold.")
    ap.add_argument(
        "--family-size",
        type=int,
        default=DEFAULT_TEST_FAMILY_SIZE,
        help="Number of tests per population/method/annotation family.",
    )
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory.")
    ap.add_argument(
        "--basename",
        default="scatter_eur300k_vs_smaller_summit_hits__mafld_8bins__bh",
        help="Basename for combined figure and tables.",
    )
    ap.add_argument(
        "--errorbar-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to rg_se for x/y error bars. Default is +/-1 SE.",
    )
    ap.add_argument("--dpi", type=int, default=450)
    ap.add_argument(
        "--width", type=float, default=10.8, help="Combined figure width in inches."
    )
    ap.add_argument(
        "--height", type=float, default=3.7, help="Combined figure height in inches."
    )
    ap.add_argument(
        "--panel-specific-limits",
        action="store_true",
        help="Use separate axis limits per panel. Default uses shared limits across panels.",
    )
    ap.add_argument(
        "--label-max",
        type=int,
        default=8,
        help="Label points when a panel has at most this many hits. Use 0 to disable labels.",
    )
    ap.add_argument(
        "--label-outliers",
        type=int,
        default=3,
        help="For larger panels, label this many high-|delta rg| outliers.",
    )
    ap.add_argument(
        "--label-min-distance",
        type=float,
        default=0.12,
        help="Minimum normalized spacing between automatically selected outlier labels.",
    )
    return ap.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def parse_ok_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def canonical_pair_with_names(row: pd.Series) -> Tuple[str, str, str, str]:
    phen1 = str(row["phen1"])
    phen2 = str(row["phen2"])
    name1 = clean_name(row.get("phen1_name", phen1), phen1)
    name2 = clean_name(row.get("phen2_name", phen2), phen2)
    if phen1 <= phen2:
        return phen1, phen2, name1, name2
    return phen2, phen1, name2, name1


def clean_name(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and not np.isfinite(value):
        return fallback
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return fallback
    return s


def pair_key(a: str, b: str) -> str:
    x, y = (a, b) if a <= b else (b, a)
    return f"{x}||{y}"


def result_table(
    df: pd.DataFrame, *, pop: str, method: str, annot: str
) -> pd.DataFrame:
    sub = df[
        (df["pop"].astype(str) == str(pop))
        & (df["method"].astype(str) == str(method))
        & (df["annot_type"].astype(str) == str(annot))
    ].copy()
    if "parse_ok" in sub.columns:
        sub = sub.loc[parse_ok_mask(sub["parse_ok"])].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "trait1",
                "trait2",
                "trait1_name",
                "trait2_name",
                "pair_key",
                "rg",
                "rg_se",
                "p_value",
            ]
        )

    pairs = [canonical_pair_with_names(row) for _, row in sub.iterrows()]
    sub["trait1"] = [x[0] for x in pairs]
    sub["trait2"] = [x[1] for x in pairs]
    sub["trait1_name"] = [x[2] for x in pairs]
    sub["trait2_name"] = [x[3] for x in pairs]
    sub["pair_key"] = [
        pair_key(a, b)
        for a, b in sub[["trait1", "trait2"]].itertuples(index=False, name=None)
    ]
    sub["rg"] = pd.to_numeric(sub["rg"], errors="coerce")
    sub["rg_se"] = pd.to_numeric(sub["rg_se"], errors="coerce")
    sub["p_value"] = [
        normal_two_sided_p(float(rg) / float(se))
        if np.isfinite(rg) and np.isfinite(se) and float(se) > 0
        else np.nan
        for rg, se in sub[["rg", "rg_se"]].itertuples(index=False, name=None)
    ]
    sub = (
        sub.sort_values(["trait1", "trait2"])
        .groupby(["trait1", "trait2"], as_index=False)
        .first()
    )
    return sub[
        [
            "trait1",
            "trait2",
            "trait1_name",
            "trait2_name",
            "pair_key",
            "rg",
            "rg_se",
            "p_value",
        ]
    ].copy()


def build_comparison_table(
    df: pd.DataFrame,
    *,
    ref_pop: str,
    small_pop: str,
    method: str,
    annot: str,
    threshold: str,
    alpha: float,
    bh_q: float,
    family_size: int,
) -> pd.DataFrame:
    small_hits = load_hits(
        df,
        pop=small_pop,
        method=method,
        annot=annot,
        threshold=threshold,
        alpha=alpha,
        bh_q=bh_q,
        family_size=family_size,
    )
    ref_hits = load_hits(
        df,
        pop=ref_pop,
        method=method,
        annot=annot,
        threshold=threshold,
        alpha=alpha,
        bh_q=bh_q,
        family_size=family_size,
    )
    ref_hit_pairs = set(ref_hits["pair_key"].astype(str))

    small = result_table(df, pop=small_pop, method=method, annot=annot)
    ref = result_table(df, pop=ref_pop, method=method, annot=annot)

    hit_keys = small_hits[["trait1", "trait2", "pair_key", "p_value"]].rename(
        columns={"p_value": "small_fdr_input_p_value"}
    )
    merged = hit_keys.merge(
        small,
        on=["trait1", "trait2", "pair_key"],
        how="left",
        validate="one_to_one",
    ).merge(
        ref,
        on=["trait1", "trait2", "pair_key"],
        how="left",
        suffixes=("_small", "_ref"),
        validate="one_to_one",
    )

    merged.insert(0, "pop", small_pop)
    merged.insert(1, "pop_label", POP_LABEL.get(small_pop, small_pop))
    merged.insert(2, "ref_pop", ref_pop)
    merged.insert(3, "ref_pop_label", POP_LABEL.get(ref_pop, ref_pop))
    merged.insert(4, "method", method)
    merged.insert(5, "annot_type", annot)
    merged["pair_label"] = [
        f"{clean_name(a, t1)}-{clean_name(b, t2)}"
        for t1, t2, a, b in merged[
            ["trait1", "trait2", "trait1_name_small", "trait2_name_small"]
        ].itertuples(index=False, name=None)
    ]
    merged["is_ref_fdr_hit"] = merged["pair_key"].astype(str).isin(ref_hit_pairs)
    merged["delta_rg_small_minus_ref"] = merged["rg_small"] - merged["rg_ref"]
    merged = merged.sort_values(
        ["small_fdr_input_p_value", "trait1", "trait2"]
    ).reset_index(drop=True)
    return merged


def data_limits(
    frames: Sequence[pd.DataFrame], *, errorbar_scale: float
) -> Tuple[float, float]:
    vals: List[np.ndarray] = []
    for frame in frames:
        if frame.empty:
            continue
        x = frame["rg_ref"].to_numpy(float)
        y = frame["rg_small"].to_numpy(float)
        xerr = errorbar_scale * frame["rg_se_ref"].to_numpy(float)
        yerr = errorbar_scale * frame["rg_se_small"].to_numpy(float)
        vals.extend([x, y, x - xerr, x + xerr, y - yerr, y + yerr])
    if not vals:
        return -1.0, 1.0
    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if lo == hi:
        pad = max(0.1, abs(lo) * 0.08)
    else:
        pad = max(0.04, 0.075 * (hi - lo))
    return lo - pad, hi + pad


def summarize(tab: pd.DataFrame) -> Dict[str, object]:
    x = tab["rg_ref"].to_numpy(float)
    y = tab["rg_small"].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    d = y - x
    if x.size > 1 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
    else:
        r = np.nan
    return {
        "n_hits": int(tab.shape[0]),
        "n_plotted": int(x.size),
        "n_ref_fdr_hits": int(tab["is_ref_fdr_hit"].sum())
        if "is_ref_fdr_hit" in tab.columns
        else 0,
        "pearson_r": r,
        "rmse_delta_rg": float(math.sqrt(np.mean(d * d))) if d.size else np.nan,
        "median_abs_delta_rg": float(np.median(np.abs(d))) if d.size else np.nan,
        "mean_delta_rg": float(np.mean(d)) if d.size else np.nan,
    }


def select_label_rows(
    tab: pd.DataFrame,
    *,
    label_max: int,
    label_outliers: int,
    label_min_distance: float,
    limits: Tuple[float, float],
) -> pd.DataFrame:
    finite = tab[
        np.isfinite(tab["rg_ref"].to_numpy(float))
        & np.isfinite(tab["rg_small"].to_numpy(float))
    ].copy()
    if finite.empty or label_max <= 0:
        return finite.iloc[0:0].copy()
    if finite.shape[0] <= label_max:
        return finite.sort_values(
            ["rg_small", "rg_ref"], ascending=[False, True]
        ).copy()

    n_labels = max(0, int(label_outliers))
    if n_labels == 0:
        return finite.iloc[0:0].copy()

    finite["abs_delta_rg"] = finite["delta_rg_small_minus_ref"].abs()
    ranked = finite.sort_values(
        ["abs_delta_rg", "small_fdr_input_p_value"], ascending=[False, True]
    )
    lo, hi = limits
    span = max(float(hi - lo), 1e-9)
    selected: List[pd.Series] = []
    for _, row in ranked.iterrows():
        if len(selected) >= n_labels:
            break
        x = float(row["rg_ref"])
        y = float(row["rg_small"])
        too_close = False
        for prev in selected:
            px = float(prev["rg_ref"])
            py = float(prev["rg_small"])
            dist = math.hypot((x - px) / span, (y - py) / span)
            if dist < label_min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(row)

    if len(selected) < n_labels:
        selected_keys = {str(row["pair_key"]) for row in selected}
        for _, row in ranked.iterrows():
            if len(selected) >= n_labels:
                break
            if str(row["pair_key"]) not in selected_keys:
                selected.append(row)
                selected_keys.add(str(row["pair_key"]))

    if not selected:
        return finite.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def manual_label_rows(tab: pd.DataFrame, *, pop: str) -> pd.DataFrame:
    positions = MANUAL_LABEL_POSITIONS.get(pop, {})
    if not positions:
        return tab.iloc[0:0].copy()

    rows: List[pd.Series] = []
    for label, (x, y, ha) in positions.items():
        match = tab[tab["pair_label"].astype(str) == label]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["label_ax_x"] = float(x)
        row["label_ax_y"] = float(y)
        row["label_ha"] = str(ha)
        rows.append(row)

    if not rows:
        return tab.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def point_to_axes_coords(
    row: pd.Series, limits: Tuple[float, float]
) -> Tuple[float, float]:
    lo, hi = limits
    span = max(float(hi - lo), 1e-9)
    return (float(row["rg_ref"]) - lo) / span, (float(row["rg_small"]) - lo) / span


def value_to_axes_coord(value: float, limits: Tuple[float, float]) -> float:
    lo, hi = limits
    span = max(float(hi - lo), 1e-9)
    return (float(value) - lo) / span


def text_size_axes(ax: plt.Axes, label: str) -> Tuple[float, float]:
    fig = ax.figure
    tmp = ax.text(
        0.0,
        0.0,
        label,
        transform=ax.transAxes,
        fontsize=LABEL_FONT_SIZE,
        ha="center",
        va="center",
        alpha=0.0,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = tmp.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
    tmp.remove()
    return float(bbox.width) + 0.018, float(bbox.height) + 0.012


def data_obstacles(
    tab: pd.DataFrame,
    *,
    limits: Tuple[float, float],
    errorbar_scale: float,
) -> Tuple[
    List[Tuple[float, float, float, float]],
    List[Tuple[Tuple[float, float], Tuple[float, float]]],
]:
    rects: List[Tuple[float, float, float, float]] = []
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    lo, hi = limits
    span = max(float(hi - lo), 1e-9)
    zero = value_to_axes_coord(0.0, limits)

    segments.append(((0.0, 0.0), (1.0, 1.0)))
    if 0.0 <= zero <= 1.0:
        segments.append(((zero, 0.0), (zero, 1.0)))
        segments.append(((0.0, zero), (1.0, zero)))

    for row in tab.itertuples(index=False):
        x = value_to_axes_coord(float(row.rg_ref), limits)
        y = value_to_axes_coord(float(row.rg_small), limits)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        marker_pad = 0.014
        rects.append((x - marker_pad, y - marker_pad, x + marker_pad, y + marker_pad))

        xerr = (
            errorbar_scale * float(row.rg_se_ref) / span
            if np.isfinite(row.rg_se_ref)
            else 0.0
        )
        yerr = (
            errorbar_scale * float(row.rg_se_small) / span
            if np.isfinite(row.rg_se_small)
            else 0.0
        )
        if xerr > 0:
            segments.append(((x - xerr, y), (x + xerr, y)))
        if yerr > 0:
            segments.append(((x, y - yerr), (x, y + yerr)))
    return rects, segments


def rect_from_center(
    cx: float, cy: float, width: float, height: float
) -> Tuple[float, float, float, float]:
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def rects_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    margin: float = 0.0,
) -> bool:
    return not (
        a[2] + margin <= b[0]
        or b[2] + margin <= a[0]
        or a[3] + margin <= b[1]
        or b[3] + margin <= a[1]
    )


def point_in_rect(
    point: Tuple[float, float], rect: Tuple[float, float, float, float]
) -> bool:
    x, y = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def closest_rect_point(
    point: Tuple[float, float],
    rect: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    x, y = point
    cx = min(max(x, rect[0]), rect[2])
    cy = min(max(y, rect[1]), rect[3])
    if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
        distances = [
            (abs(x - rect[0]), (rect[0], y)),
            (abs(x - rect[2]), (rect[2], y)),
            (abs(y - rect[1]), (x, rect[1])),
            (abs(y - rect[3]), (x, rect[3])),
        ]
        _, anchor = min(distances, key=lambda z: z[0])
        return anchor
    return cx, cy


def orient(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> bool:
    eps = 1e-10
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
        and abs(orient(a, c, b)) <= eps
    )


def segments_intersect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> bool:
    eps = 1e-10
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


def segment_intersects_rect(
    segment: Tuple[Tuple[float, float], Tuple[float, float]],
    rect: Tuple[float, float, float, float],
) -> bool:
    a, b = segment
    if point_in_rect(a, rect) or point_in_rect(b, rect):
        return True
    left, bottom, right, top = rect
    edges = [
        ((left, bottom), (right, bottom)),
        ((right, bottom), (right, top)),
        ((right, top), (left, top)),
        ((left, top), (left, bottom)),
    ]
    return any(segments_intersect(a, b, c, d) for c, d in edges)


def rect_line_penalty(rect: Tuple[float, float, float, float]) -> float:
    reference_lines = [
        ((0.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (0.0, 1.0)),
    ]
    return 0.20 * sum(segment_intersects_rect(line, rect) for line in reference_lines)


def build_label_candidates(
    ax: plt.Axes,
    row: pd.Series,
    *,
    limits: Tuple[float, float],
    obstacle_rects: Sequence[Tuple[float, float, float, float]],
    obstacle_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> List[Dict[str, object]]:
    point = point_to_axes_coords(row, limits)
    width, height = text_size_axes(ax, str(row["pair_label"]))
    dirs = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.72, 0.72),
        (-0.72, 0.72),
        (0.72, -0.72),
        (-0.72, -0.72),
    ]
    distances = [0.080, 0.105, 0.135, 0.170, 0.215, 0.265, 0.320]
    candidates: List[Dict[str, object]] = []
    for dist in distances:
        for dx, dy in dirs:
            cx = point[0] + dist * dx
            cy = point[1] + dist * dy
            rect = rect_from_center(cx, cy, width, height)
            if (
                rect[0] < LABEL_EDGE_PAD
                or rect[1] < LABEL_EDGE_PAD
                or rect[2] > 1.0 - LABEL_EDGE_PAD
                or rect[3] > 1.0 - LABEL_EDGE_PAD
                or rects_overlap(rect, METRIC_BOX_RECT, margin=LABEL_MARGIN)
                or any(
                    rects_overlap(rect, obst, margin=0.004) for obst in obstacle_rects
                )
                or any(segment_intersects_rect(seg, rect) for seg in obstacle_segments)
                or point_in_rect(point, rect)
            ):
                continue
            anchor = closest_rect_point(point, rect)
            leader = (point, anchor)
            length = math.hypot(anchor[0] - point[0], anchor[1] - point[1])
            if length < MIN_LEADER_LEN:
                continue
            candidates.append(
                {
                    "row": row,
                    "center": (cx, cy),
                    "rect": rect,
                    "leader": leader,
                    "score": length + 0.015 * dist + rect_line_penalty(rect),
                }
            )
    candidates.sort(key=lambda x: float(x["score"]))
    return candidates[:36]


def layout_is_valid(combo: Sequence[Dict[str, object]]) -> bool:
    for i in range(len(combo)):
        rect_i = combo[i]["rect"]
        leader_i = combo[i]["leader"]
        for j in range(i + 1, len(combo)):
            rect_j = combo[j]["rect"]
            leader_j = combo[j]["leader"]
            if rects_overlap(rect_i, rect_j, margin=LABEL_MARGIN):
                return False
            if segments_intersect(leader_i[0], leader_i[1], leader_j[0], leader_j[1]):
                return False
            if segment_intersects_rect(leader_i, rect_j) or segment_intersects_rect(
                leader_j, rect_i
            ):
                return False
    return True


def choose_label_layout(
    ax: plt.Axes,
    labels: pd.DataFrame,
    *,
    limits: Tuple[float, float],
    obstacle_rects: Sequence[Tuple[float, float, float, float]],
    obstacle_segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> List[Dict[str, object]]:
    if labels.empty:
        return []
    rows = [pd.Series(row._asdict()) for row in labels.itertuples(index=False)]
    candidate_sets = [
        build_label_candidates(
            ax,
            row,
            limits=limits,
            obstacle_rects=obstacle_rects,
            obstacle_segments=obstacle_segments,
        )
        for row in rows
    ]
    if any(len(cands) == 0 for cands in candidate_sets):
        return []

    best_combo: Tuple[Dict[str, object], ...] | None = None
    best_score = math.inf
    for combo in itertools.product(*candidate_sets):
        score = sum(float(cand["score"]) for cand in combo)
        if score >= best_score:
            continue
        if not layout_is_valid(combo):
            continue
        best_score = score
        best_combo = combo

    if best_combo is not None:
        return list(best_combo)

    best_penalized: Tuple[Dict[str, object], ...] | None = None
    best_penalty_score = math.inf
    for combo in itertools.product(*candidate_sets):
        penalty = 0.0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if rects_overlap(
                    combo[i]["rect"], combo[j]["rect"], margin=LABEL_MARGIN
                ):
                    penalty += 10.0
                if segments_intersect(
                    combo[i]["leader"][0],
                    combo[i]["leader"][1],
                    combo[j]["leader"][0],
                    combo[j]["leader"][1],
                ):
                    penalty += 10.0
                if segment_intersects_rect(
                    combo[i]["leader"], combo[j]["rect"]
                ) or segment_intersects_rect(combo[j]["leader"], combo[i]["rect"]):
                    penalty += 8.0
        score = penalty + sum(float(cand["score"]) for cand in combo)
        if score < best_penalty_score:
            best_penalty_score = score
            best_penalized = combo
    return list(best_penalized) if best_penalized is not None else []


def add_point_labels(
    ax: plt.Axes,
    *,
    tab: pd.DataFrame,
    labels: pd.DataFrame,
    limits: Tuple[float, float],
    errorbar_scale: float,
) -> None:
    del tab, limits, errorbar_scale
    for row in labels.itertuples(index=False):
        ax.annotate(
            str(row.pair_label),
            xy=(float(row.rg_ref), float(row.rg_small)),
            xycoords="data",
            xytext=(float(row.label_ax_x), float(row.label_ax_y)),
            textcoords=ax.transAxes,
            ha=str(row.label_ha),
            va="center",
            fontsize=LABEL_FONT_SIZE,
            color=TEXT_DARK,
            zorder=7,
            arrowprops=dict(
                arrowstyle="-",
                color="#374151",
                lw=0.65,
                alpha=0.80,
                shrinkA=2.0,
                shrinkB=3.0,
                connectionstyle="arc3,rad=0",
            ),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.45),
            annotation_clip=False,
        )


def draw_panel(
    ax: plt.Axes,
    *,
    tab: pd.DataFrame,
    pop: str,
    ref_pop: str,
    method: str,
    annot: str,
    limits: Tuple[float, float],
    errorbar_scale: float,
    label_max: int,
    label_outliers: int,
    label_min_distance: float,
    letter: str,
    metrics: Dict[str, object],
) -> None:
    color = POP_COLOR.get(pop, TAB20[0])
    lo, hi = limits
    ax.axhline(0.0, color="#D1D5DB", lw=0.8, zorder=0)
    ax.axvline(0.0, color="#D1D5DB", lw=0.8, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="#6B7280", lw=1.0, ls=(0, (4, 3)), zorder=1)

    x = tab["rg_ref"].to_numpy(float)
    y = tab["rg_small"].to_numpy(float)
    xerr = errorbar_scale * tab["rg_se_ref"].to_numpy(float)
    yerr = errorbar_scale * tab["rg_se_small"].to_numpy(float)
    is_ref_hit = tab["is_ref_fdr_hit"].to_numpy(bool)
    finite = np.isfinite(x) & np.isfinite(y)

    for ref_hit_value, marker_face, marker_label in [
        (True, color, f"Also {POP_LABEL.get(ref_pop, ref_pop)} FDR hit"),
        (False, "white", f"Not {POP_LABEL.get(ref_pop, ref_pop)} FDR hit"),
    ]:
        keep = finite & (is_ref_hit == ref_hit_value)
        if not np.any(keep):
            continue
        ax.errorbar(
            x[keep],
            y[keep],
            xerr=xerr[keep],
            yerr=yerr[keep],
            fmt="o",
            ms=4.8,
            mfc=marker_face,
            mec=color,
            mew=1.0,
            ecolor=color,
            elinewidth=0.7,
            capsize=2.0,
            alpha=0.78,
            label=marker_label,
            zorder=3 if ref_hit_value else 2,
        )

    del label_max, label_outliers, label_min_distance
    label_tab = manual_label_rows(tab, pop=pop)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E7EB", lw=0.55, alpha=0.8)
    ax.set_title(
        f"{POP_LABEL.get(pop, pop)} FDR hits (n={int(metrics['n_hits'])})",
        color=color,
        fontweight="bold",
    )
    method_label = METHOD_LABEL.get(method, method)
    ax.set_xlabel(f"{POP_LABEL.get(ref_pop, ref_pop)} {method_label} rg")
    ax.set_ylabel(f"{POP_LABEL.get(pop, pop)} {method_label} rg")

    r = metrics.get("pearson_r", np.nan)
    r_txt = f"{float(r):.2f}" if np.isfinite(r) else "NA"
    txt = f"r={r_txt}"
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=TEXT_MUTED,
        bbox=dict(
            facecolor="white",
            edgecolor="#D1D5DB",
            lw=0.5,
            alpha=0.92,
            boxstyle="round,pad=0.25",
        ),
        zorder=8,
    )
    ax.text(
        -0.10,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=TEXT_DARK,
        clip_on=False,
    )
    return label_tab


def save_figure(fig: plt.Figure, outbase: Path, *, dpi: int) -> None:
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi)
    fig.savefig(outbase.with_suffix(".pdf"))


def make_combined_figure(
    frames: Sequence[pd.DataFrame],
    metrics: Sequence[Dict[str, object]],
    *,
    pops: Sequence[str],
    ref_pop: str,
    method: str,
    annot: str,
    errorbar_scale: float,
    label_max: int,
    label_outliers: int,
    label_min_distance: float,
    width: float,
    height: float,
    shared_limits: bool,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(pops), figsize=(width, height), squeeze=False)
    all_limits = data_limits(frames, errorbar_scale=errorbar_scale)
    label_jobs: List[
        Tuple[plt.Axes, pd.DataFrame, pd.DataFrame, Tuple[float, float]]
    ] = []
    for idx, (pop, frame, metric) in enumerate(zip(pops, frames, metrics)):
        limits = (
            all_limits
            if shared_limits
            else data_limits([frame], errorbar_scale=errorbar_scale)
        )
        labels = draw_panel(
            axes[0][idx],
            tab=frame,
            pop=pop,
            ref_pop=ref_pop,
            method=method,
            annot=annot,
            limits=limits,
            errorbar_scale=errorbar_scale,
            label_max=label_max,
            label_outliers=label_outliers,
            label_min_distance=label_min_distance,
            letter=chr(ord("A") + idx),
            metrics=metric,
        )
        label_jobs.append((axes[0][idx], frame, labels, limits))

    handles, labels = [], []
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        for handle, label in zip(h, lab):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if len(labels) > 1:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            frameon=False,
            fontsize=8,
        )
    fig.tight_layout(pad=0.55, w_pad=1.0)
    fig.canvas.draw()
    for ax, frame, labels, limits in label_jobs:
        add_point_labels(
            ax,
            tab=frame,
            labels=labels,
            limits=limits,
            errorbar_scale=errorbar_scale,
        )
    return fig


def make_single_figure(
    frame: pd.DataFrame,
    metric: Dict[str, object],
    *,
    pop: str,
    ref_pop: str,
    method: str,
    annot: str,
    errorbar_scale: float,
    label_max: int,
    label_outliers: int,
    label_min_distance: float,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.4))
    labels = draw_panel(
        ax,
        tab=frame,
        pop=pop,
        ref_pop=ref_pop,
        method=method,
        annot=annot,
        limits=data_limits([frame], errorbar_scale=errorbar_scale),
        errorbar_scale=errorbar_scale,
        label_max=label_max,
        label_outliers=label_outliers,
        label_min_distance=label_min_distance,
        letter="",
        metrics=metric,
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    if len(legend_labels) > 1:
        ax.legend(handles, legend_labels, loc="best", frameon=True, fontsize=7.2)
    fig.tight_layout(pad=0.55)
    fig.canvas.draw()
    add_point_labels(
        ax,
        tab=frame,
        labels=labels,
        limits=data_limits([frame], errorbar_scale=errorbar_scale),
        errorbar_scale=errorbar_scale,
    )
    return fig


def main() -> None:
    args = parse_args()
    set_style()

    if args.errorbar_scale < 0:
        raise SystemExit("--errorbar-scale must be non-negative.")
    if args.label_max < 0:
        raise SystemExit("--label-max must be non-negative.")
    if args.label_outliers < 0:
        raise SystemExit("--label-outliers must be non-negative.")
    if args.label_min_distance < 0:
        raise SystemExit("--label-min-distance must be non-negative.")
    if args.ref_pop in args.small_pops:
        raise SystemExit("--small-pops must not include --ref-pop.")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing input CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    frames = [
        build_comparison_table(
            df,
            ref_pop=args.ref_pop,
            small_pop=pop,
            method=args.method,
            annot=args.annot,
            threshold=args.threshold,
            alpha=args.alpha,
            bh_q=args.bh_q,
            family_size=args.family_size,
        )
        for pop in args.small_pops
    ]
    metrics: List[Dict[str, object]] = []
    for pop, frame in zip(args.small_pops, frames):
        row = {
            "pop": pop,
            "pop_label": POP_LABEL.get(pop, pop),
            "ref_pop": args.ref_pop,
            "ref_pop_label": POP_LABEL.get(args.ref_pop, args.ref_pop),
            "method": args.method,
            "annot_type": args.annot,
            "threshold": args.threshold,
            "alpha": args.alpha,
            "bh_q": args.bh_q,
            "family_size": args.family_size,
            "errorbar_scale": args.errorbar_scale,
            "label_max": args.label_max,
            "label_outliers": args.label_outliers,
            "label_min_distance": args.label_min_distance,
        }
        row.update(summarize(frame))
        metrics.append(row)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outbase = outdir / args.basename

    combined = make_combined_figure(
        frames,
        metrics,
        pops=args.small_pops,
        ref_pop=args.ref_pop,
        method=args.method,
        annot=args.annot,
        errorbar_scale=args.errorbar_scale,
        label_max=args.label_max,
        label_outliers=args.label_outliers,
        label_min_distance=args.label_min_distance,
        width=args.width,
        height=args.height,
        shared_limits=not args.panel_specific_limits,
    )
    save_figure(combined, outbase, dpi=args.dpi)
    plt.close(combined)

    source = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    source.insert(0, "threshold", args.threshold)
    source.insert(1, "alpha", args.alpha)
    source.insert(2, "bh_q", args.bh_q)
    source.insert(3, "family_size", args.family_size)
    source.insert(4, "errorbar_scale", args.errorbar_scale)
    source.insert(5, "label_max", args.label_max)
    source.insert(6, "label_outliers", args.label_outliers)
    source.insert(7, "label_min_distance", args.label_min_distance)
    source_path = outbase.with_name(outbase.name + "__source_data.csv")
    metrics_path = outbase.with_name(outbase.name + "__metrics.csv")
    source.to_csv(source_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    for pop, frame, metric in zip(args.small_pops, frames, metrics):
        single_base = outdir / f"{args.basename}__{pop}"
        fig = make_single_figure(
            frame,
            metric,
            pop=pop,
            ref_pop=args.ref_pop,
            method=args.method,
            annot=args.annot,
            errorbar_scale=args.errorbar_scale,
            label_max=args.label_max,
            label_outliers=args.label_outliers,
            label_min_distance=args.label_min_distance,
        )
        save_figure(fig, single_base, dpi=args.dpi)
        plt.close(fig)
        frame.to_csv(
            single_base.with_name(single_base.name + "__source_data.csv"), index=False
        )

    print(f"[write] {outbase.with_suffix('.png')}")
    print(f"[write] {outbase.with_suffix('.pdf')}")
    print(f"[write] {source_path}")
    print(f"[write] {metrics_path}")
    for row in metrics:
        print(
            "[counts:{pop}] hits={n_hits} plotted={n_plotted} also_ref_fdr={n_ref_fdr_hits} r={pearson_r:.6g}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
