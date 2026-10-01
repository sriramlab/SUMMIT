#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


TAB20 = [to_hex(c) for c in colormaps["tab20"].colors]
palette = {
    "summit": TAB20[4],
    "covldsc": TAB20[2],
}
method_label_map = {
    "summit": "SUMMIT",
    "covldsc": "cov-LDSC",
}
annot_label_map = {
    "single": "Single",
    "maf_6bins": "MAF (6)",
    "mafld_8bins": "MAF-LD (8)",
    "mafld_24bins": "MAF-LD (24)",
    "baseline": "Baseline",
}
preferred_annot_order = [
    "single",
    "maf_6bins",
    "mafld_8bins",
    "mafld_24bins",
    "baseline",
]


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }
)


# ---------------------------
# Helpers
# ---------------------------


def pretty_annot_name(x: str) -> str:
    return annot_label_map.get(x, x.replace("_", " "))


def pair_id_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pair_id"] = out["phen1"].astype(str) + "||" + out["phen2"].astype(str)
    return out


def concordance_correlation_coefficient(
    x: Sequence[float], y: Sequence[float]
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    n = x.size
    if n == 0:
        return np.nan
    if n == 1:
        return 1.0 if np.isclose(x[0], y[0], equal_nan=False) else np.nan

    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    denom = vx + vy + (mx - my) ** 2
    if np.isclose(denom, 0.0):
        return 1.0 if np.allclose(x, y) else np.nan
    return 2.0 * cov / denom


def choose_annotations(
    available: Iterable[str], requested: Optional[Sequence[str]]
) -> List[str]:
    available = list(dict.fromkeys(available))
    if requested:
        out = [a for a in requested if a in set(available)]
        missing = [a for a in requested if a not in set(available)]
        if missing:
            raise ValueError(
                f"Requested annotation(s) not found in data: {', '.join(missing)}"
            )
        return out

    avail_set = set(available)
    out = [a for a in preferred_annot_order if a in avail_set]
    out.extend([a for a in available if a not in set(out)])
    return out


def build_series_maps(
    df: pd.DataFrame,
    pop: str,
    annotations: Sequence[str],
    methods: Sequence[str],
) -> Tuple[Dict[str, Dict[str, pd.Series]], Dict[str, Dict[str, pd.Series]]]:
    sub = df.loc[df["pop"] == pop].copy()
    if sub.empty:
        raise ValueError(f"No rows found for population '{pop}'.")

    sub = sub.loc[
        sub["method"].isin(methods) & sub["annot_type"].isin(annotations)
    ].copy()
    if sub.empty:
        raise ValueError("No rows left after filtering by methods/annotations.")

    sub = pair_id_frame(sub)
    sub = sub.drop_duplicates(subset=["method", "annot_type", "pair_id"], keep="last")

    rg_map: Dict[str, Dict[str, pd.Series]] = {m: {} for m in methods}
    se_map: Dict[str, Dict[str, pd.Series]] = {m: {} for m in methods}

    for method in methods:
        for annot in annotations:
            cur = sub.loc[
                (sub["method"] == method) & (sub["annot_type"] == annot),
                ["pair_id", "rg", "rg_se"],
            ].copy()
            if cur.empty:
                rg_map[method][annot] = pd.Series(dtype=float)
                se_map[method][annot] = pd.Series(dtype=float)
            else:
                rg_map[method][annot] = cur.set_index("pair_id")["rg"].astype(float)
                se_map[method][annot] = cur.set_index("pair_id")["rg_se"].astype(float)

    return rg_map, se_map


def finite_pair_index(s: pd.Series) -> Set[str]:
    if s.empty:
        return set()
    return set(s.index[np.isfinite(s.to_numpy(dtype=float))])


def valid_pairs_within_method(
    rg_map: Dict[str, Dict[str, pd.Series]],
    annot_x: str,
    annot_y: str,
    method: str,
) -> List[str]:
    sx = rg_map[method][annot_x]
    sy = rg_map[method][annot_y]
    return sorted(finite_pair_index(sx) & finite_pair_index(sy))


def strict_common_pairs(
    rg_map: Dict[str, Dict[str, pd.Series]],
    annot_x: str,
    annot_y: str,
    methods: Sequence[str],
) -> List[str]:
    common: Optional[Set[str]] = None
    for method in methods:
        valid = set(valid_pairs_within_method(rg_map, annot_x, annot_y, method))
        common = valid if common is None else (common & valid)
    return sorted(common) if common is not None else []


def valid_pairs_for_annotation_method(
    se_map: Dict[str, Dict[str, pd.Series]],
    annot: str,
    method: str,
) -> List[str]:
    return sorted(finite_pair_index(se_map[method][annot]))


def common_pairs_for_annotation(
    se_map: Dict[str, Dict[str, pd.Series]],
    annot: str,
    methods: Sequence[str],
) -> List[str]:
    common: Optional[Set[str]] = None
    for method in methods:
        valid = finite_pair_index(se_map[method][annot])
        common = valid if common is None else (common & valid)
    return sorted(common) if common is not None else []


def pooled_limits(
    arrays: Sequence[np.ndarray], pad_frac: float = 0.06
) -> Tuple[float, float]:
    vals = np.concatenate(
        [
            np.asarray(a, dtype=float).ravel()
            for a in arrays
            if a is not None and np.asarray(a).size > 0
        ]
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        delta = 0.1 if np.isclose(lo, 0.0) else 0.1 * abs(lo)
        lo -= delta
        hi += delta
    span = hi - lo
    pad = pad_frac * span
    return lo - pad, hi + pad


def choose_hist_bins(values: np.ndarray) -> np.ndarray | int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 10
    try:
        bins = np.histogram_bin_edges(values, bins="fd")
        if bins.size < 5:
            bins = np.histogram_bin_edges(
                values, bins=min(12, max(5, values.size // 2))
            )
        return bins
    except Exception:
        return min(20, max(8, values.size // 4))


def normalize_overlap_arg(tokens: Sequence[str]) -> Set[str]:
    out = set(tokens)
    if "none" in out:
        return set()
    if "all" in out:
        return {"ccc", "scatter", "hist"}
    return {x for x in out if x in {"ccc", "scatter", "hist"}}


def format_metric(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "nan"


def overlap_mode_label(kind: str, use_overlap: bool) -> str:
    if kind in {"ccc", "scatter"}:
        return "common across methods/annotations" if use_overlap else "method-specific"
    return "common across methods" if use_overlap else "method-specific"


# ---------------------------
# Draw panels
# ---------------------------


def draw_scatter_panel(
    ax: plt.Axes,
    xs: Dict[str, np.ndarray],
    ys: Dict[str, np.ndarray],
    methods: Sequence[str],
    overlap_scatter: bool,
    n_summit_used: int,
    n_covldsc_used: int,
    n_common: int,
):
    pooled = [xs[m] for m in methods] + [ys[m] for m in methods]
    pooled = [a for a in pooled if np.asarray(a).size > 0]
    if not pooled:
        ax.text(
            0.5,
            0.5,
            "No finite pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    lo, hi = pooled_limits(pooled)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0, zorder=1)
    for method in methods[::-1]:
        x = np.asarray(xs[method], dtype=float)
        y = np.asarray(ys[method], dtype=float)
        if x.size == 0:
            continue
        ax.scatter(
            x,
            y,
            s=16,
            alpha=0.20,
            color=palette[method],
            linewidths=0.25,
            edgecolors="white",
            rasterized=True,
            zorder=2,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(length=2.5)

    if overlap_scatter:
        label = f"common n={n_common}"
    else:
        label = f"SUMMIT n={n_summit_used}\ncov-LDSC n={n_covldsc_used}"
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox=dict(
            boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.87
        ),
    )


def draw_heatmap_panel(
    ax: plt.Axes,
    delta_metric: float,
    summit_metric: float,
    covldsc_metric: float,
    n_summit_used: int,
    n_covldsc_used: int,
    n_common: int,
    overlap_ccc: bool,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
):
    if np.isfinite(delta_metric):
        face = cmap(norm(delta_metric))
    else:
        face = (0.92, 0.92, 0.92, 1.0)

    ax.imshow(
        np.array([[0.0]]),
        cmap=mpl.colors.ListedColormap([face]),
        aspect="auto",
        extent=(0, 1, 0, 1),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    if overlap_ccc:
        text = (
            f"SUMMIT: {format_metric(summit_metric)}\n"
            f"cov-LDSC: {format_metric(covldsc_metric)}"
        )
        ax.text(
            0.5, 0.56, text, ha="center", va="center", fontsize=7.8, fontweight="bold"
        )
        ax.text(
            0.5, 0.15, f"common n={n_common}", ha="center", va="center", fontsize=7.0
        )
    else:
        text = (
            f"SUMMIT: {format_metric(summit_metric)} (n={n_summit_used})\n"
            f"cov-LDSC: {format_metric(covldsc_metric)} (n={n_covldsc_used})"
        )
        ax.text(
            0.5, 0.50, text, ha="center", va="center", fontsize=7.8, fontweight="bold"
        )
        ax.text(
            0.5, 0.14, f"common n={n_common}", ha="center", va="center", fontsize=7.0
        )


def draw_hist_panel(
    ax: plt.Axes,
    se_map: Dict[str, Dict[str, pd.Series]],
    annot: str,
    methods: Sequence[str],
    pop: str,
    overlap_hist: bool,
):
    vals: Dict[str, np.ndarray] = {}
    pooled: List[np.ndarray] = []

    if overlap_hist:
        common = common_pairs_for_annotation(se_map, annot, methods)
        for method in methods:
            s = se_map[method][annot].reindex(common)
            v = s[np.isfinite(s.values)].to_numpy(dtype=float)
            vals[method] = v
            if v.size:
                pooled.append(v)
        empty_msg = "No common finite SE"
    else:
        for method in methods:
            s = se_map[method][annot]
            v = s[np.isfinite(s.values)].to_numpy(dtype=float)
            vals[method] = v
            if v.size:
                pooled.append(v)
        empty_msg = "No finite SE"

    if not pooled:
        ax.text(
            0.5,
            0.5,
            empty_msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    bins = choose_hist_bins(np.concatenate(pooled))
    maxval = 0.0
    for method in methods:
        v = vals[method]
        if v.size == 0:
            continue
        ax.hist(
            v,
            bins=bins,
            density=True,
            alpha=0.35,
            color=palette[method],
            edgecolor=palette[method],
            linewidth=0.9,
        )
        maxval = max(float(np.max(v)), maxval)

    max_axis_val = 5.0 if pop == "AFR" else 2.0
    xmax = min(max(maxval * 1.02, 0.05), max_axis_val)
    ax.set_xlim(0, xmax)

    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(length=2.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------
# Metrics
# ---------------------------


def compute_metrics_table(
    rg_map: Dict[str, Dict[str, pd.Series]],
    annotations: Sequence[str],
    methods: Sequence[str],
    min_common_pairs: int,
    overlap_ccc: bool,
) -> pd.DataFrame:
    rows = []
    for i, annot_row in enumerate(annotations):
        for j, annot_col in enumerate(annotations):
            if i == j:
                continue

            common = strict_common_pairs(rg_map, annot_row, annot_col, methods)
            summit_pairs = valid_pairs_within_method(
                rg_map, annot_row, annot_col, "summit"
            )
            covldsc_pairs = valid_pairs_within_method(
                rg_map, annot_row, annot_col, "covldsc"
            )

            summit_x = rg_map["summit"][annot_row]
            summit_y = rg_map["summit"][annot_col]
            covldsc_x = rg_map["covldsc"][annot_row]
            covldsc_y = rg_map["covldsc"][annot_col]

            if overlap_ccc:
                summit_used = common
                covldsc_used = common
            else:
                summit_used = summit_pairs
                covldsc_used = covldsc_pairs

            n_summit_available = len(summit_pairs)
            n_covldsc_available = len(covldsc_pairs)
            n_common = len(common)
            n_summit_used = len(summit_used)
            n_covldsc_used = len(covldsc_used)

            summit_ccc = (
                concordance_correlation_coefficient(
                    summit_x.reindex(summit_used).to_numpy(dtype=float),
                    summit_y.reindex(summit_used).to_numpy(dtype=float),
                )
                if n_summit_used >= min_common_pairs
                else np.nan
            )
            covldsc_ccc = (
                concordance_correlation_coefficient(
                    covldsc_x.reindex(covldsc_used).to_numpy(dtype=float),
                    covldsc_y.reindex(covldsc_used).to_numpy(dtype=float),
                )
                if n_covldsc_used >= min_common_pairs
                else np.nan
            )
            delta = (
                summit_ccc - covldsc_ccc
                if np.isfinite(summit_ccc) and np.isfinite(covldsc_ccc)
                else np.nan
            )

            rows.append(
                {
                    "annot_x": annot_row,
                    "annot_y": annot_col,
                    "annot_x_label": pretty_annot_name(annot_row),
                    "annot_y_label": pretty_annot_name(annot_col),
                    "ccc_overlap": bool(overlap_ccc),
                    "n_summit_available": n_summit_available,
                    "n_covldsc_available": n_covldsc_available,
                    "n_common": n_common,
                    "n_summit_used": n_summit_used,
                    "n_covldsc_used": n_covldsc_used,
                    "n_summit": n_summit_used,
                    "n_covldsc": n_covldsc_used,
                    "ccc_summit": summit_ccc,
                    "ccc_covldsc": covldsc_ccc,
                    "delta_ccc": delta,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------
# Main plotting routine
# ---------------------------


def make_figure(
    df: pd.DataFrame,
    pop: str,
    annotations: Sequence[str],
    methods: Sequence[str],
    scatter_triangle: str,
    min_common_pairs: int,
    out_prefix: Path,
    overlap_modes: Set[str],
) -> None:
    overlap_ccc = "ccc" in overlap_modes
    overlap_scatter = "scatter" in overlap_modes
    overlap_hist = "hist" in overlap_modes

    rg_map, se_map = build_series_maps(
        df, pop=pop, annotations=annotations, methods=methods
    )
    metrics = compute_metrics_table(
        rg_map,
        annotations,
        methods,
        min_common_pairs=min_common_pairs,
        overlap_ccc=overlap_ccc,
    )

    delta_vals = metrics["delta_ccc"].to_numpy(dtype=float)
    finite_delta = delta_vals[np.isfinite(delta_vals)]
    vmax = float(np.max(np.abs(finite_delta))) if finite_delta.size else 0.25
    vmax = max(vmax, 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")

    n = len(annotations)
    cell = 2.35 if n <= 4 else 2.10
    fig, axes = plt.subplots(
        n, n, figsize=(cell * n + 1.8, cell * n + 1.20), squeeze=False
    )

    metric_lookup = {
        (r.annot_x, r.annot_y): (
            r.ccc_summit,
            r.ccc_covldsc,
            r.delta_ccc,
            int(r.n_summit_used),
            int(r.n_covldsc_used),
            int(r.n_common),
        )
        for r in metrics.itertuples(index=False)
    }

    for i, annot_row in enumerate(annotations):
        for j, annot_col in enumerate(annotations):
            ax = axes[i, j]

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            if i == j:
                draw_hist_panel(
                    ax,
                    se_map,
                    annot_row,
                    methods,
                    pop,
                    overlap_hist=overlap_hist,
                )
            else:
                use_scatter = (i < j) if scatter_triangle == "upper" else (i > j)
                (
                    summit_ccc,
                    covldsc_ccc,
                    delta_ccc,
                    n_summit_used,
                    n_covldsc_used,
                    n_common,
                ) = metric_lookup[(annot_row, annot_col)]

                if use_scatter:
                    if overlap_scatter:
                        pair_sets = {
                            m: strict_common_pairs(
                                rg_map, annot_row, annot_col, methods
                            )
                            for m in methods
                        }
                        enough = len(pair_sets[methods[0]]) >= min_common_pairs
                        empty_text = f"< {min_common_pairs} common\npairs"
                        n_summit_plot = len(pair_sets["summit"])
                        n_covldsc_plot = len(pair_sets["covldsc"])
                        n_common_plot = len(pair_sets[methods[0]])
                    else:
                        pair_sets = {
                            m: valid_pairs_within_method(
                                rg_map, annot_row, annot_col, m
                            )
                            for m in methods
                        }
                        enough = (
                            max((len(v) for v in pair_sets.values()), default=0)
                            >= min_common_pairs
                        )
                        empty_text = f"< {min_common_pairs} usable\npairs"
                        n_summit_plot = len(pair_sets["summit"])
                        n_covldsc_plot = len(pair_sets["covldsc"])
                        n_common_plot = len(
                            strict_common_pairs(rg_map, annot_row, annot_col, methods)
                        )

                    if not enough:
                        ax.text(
                            0.5,
                            0.5,
                            empty_text,
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=8.5,
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        xs = {
                            m: rg_map[m][annot_row]
                            .reindex(pair_sets[m])
                            .to_numpy(dtype=float)
                            for m in methods
                        }
                        ys = {
                            m: rg_map[m][annot_col]
                            .reindex(pair_sets[m])
                            .to_numpy(dtype=float)
                            for m in methods
                        }
                        draw_scatter_panel(
                            ax,
                            xs,
                            ys,
                            methods,
                            overlap_scatter=overlap_scatter,
                            n_summit_used=n_summit_plot,
                            n_covldsc_used=n_covldsc_plot,
                            n_common=n_common_plot,
                        )
                else:
                    draw_heatmap_panel(
                        ax,
                        delta_metric=delta_ccc,
                        summit_metric=summit_ccc,
                        covldsc_metric=covldsc_ccc,
                        n_summit_used=n_summit_used,
                        n_covldsc_used=n_covldsc_used,
                        n_common=n_common,
                        overlap_ccc=overlap_ccc,
                        norm=norm,
                        cmap=cmap,
                    )

            if i == 0:
                ax.set_title(pretty_annot_name(annot_col), pad=8)
            if j == 0:
                ax.set_ylabel(pretty_annot_name(annot_row), labelpad=10)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[m],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=6.5,
            label=method_label_map[m],
        )
        for m in methods
    ]
    handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="y = x")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.subplots_adjust(
        left=0.10, right=0.91, top=0.865, bottom=0.09, wspace=0.14, hspace=0.14
    )

    cax = fig.add_axes([0.925, 0.18, 0.018, 0.60])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\Delta$CCC (SUMMIT - cov-LDSC)", rotation=90)

    metrics = metrics.assign(
        pop=pop,
        scatter_triangle=scatter_triangle,
        overlap_scatter=overlap_scatter,
        overlap_hist=overlap_hist,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    metrics_path = out_prefix.with_name(out_prefix.name + ".metrics.csv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    metrics.to_csv(metrics_path, index=False)
    plt.close(fig)

    print(f"[write] {png_path}")
    print(f"[write] {pdf_path}")
    print(f"[write] {metrics_path}")


# ---------------------------
# CLI
# ---------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Scatter/heatmap matrix for total rg agreement across annotation schemes. "
            "Pair selection can be chosen separately for scatter panels, CCC heatmaps, and diagonal SE histograms."
        )
    )
    ap.add_argument(
        "--csv",
        default=None,
        help=(
            "Parsed CSV from the supplied aggregate table. If omitted, the script uses "
            "{out_root}/{pop}/parsed_real_rg.csv."
        ),
    )
    ap.add_argument(
        "--out-root",
        default="data/real_rg",
        help="Root directory used by parse_real.py.",
    )
    ap.add_argument(
        "--pop", required=True, help="Population to plot, e.g. AFR, SAS, EUR, EUR_300k."
    )
    ap.add_argument(
        "--annotations",
        nargs="+",
        # default=None,
        # default=["single", "maf_6bins", "mafld_8bins", "mafld_24bins", "baseline"], #maf6 not ready for cov-ldsc
        default=["single", "mafld_8bins", "mafld_24bins", "baseline"],
        help=(
            "Annotation types to include, in plotting order. "
            "If omitted, a preferred order is used when present."
        ),
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["summit", "covldsc"],
        choices=["summit", "covldsc"],
        help="Methods to compare. The plotting layout expects summit and covldsc.",
    )
    ap.add_argument(
        "--scatter-triangle",
        choices=["upper", "lower"],
        default="upper",
        help=(
            "Which off-diagonal half gets the scatter panels. Default is 'upper' "
            "to match your (1,2) scatter example."
        ),
    )
    ap.add_argument(
        "--overlap",
        nargs="+",
        choices=["ccc", "scatter", "hist", "all", "none"],
        default=["scatter"],
        help=(
            "Which figure elements should use strict overlapping trait pairs. "
            "Default keeps the previous behavior: scatter panels use overlapping pairs, "
            "while CCC and histograms are method-specific. Use '--overlap all' to make all three use overlapping pairs, or '--overlap none' to make all three method-specific."
        ),
    )
    ap.add_argument(
        "--min-common-pairs",
        type=int,
        default=3,
        help="Minimum number of usable trait pairs required for a scatter/metric panel.",
    )
    ap.add_argument(
        "--out-prefix",
        default=None,
        help=(
            "Output prefix without extension. The script writes .png, .pdf, and .metrics.csv. "
            "Default: {out_root}/{pop}/scatter_rg_annot_supp_{pop}."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv is None:
        csv_path = Path(args.out_root) / args.pop / "parsed_real_rg.csv"
    else:
        csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    needed = {"pop", "method", "annot_type", "phen1", "phen2", "rg", "rg_se"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    available_annots = (
        df.loc[df["pop"] == args.pop, "annot_type"].dropna().astype(str).tolist()
    )
    annotations = choose_annotations(available_annots, args.annotations)
    if len(annotations) < 2:
        raise ValueError("Need at least two annotations to make the agreement matrix.")

    methods = list(args.methods)
    if set(methods) != {"summit", "covldsc"}:
        raise ValueError(
            "This plotting layout currently expects exactly: --methods summit covldsc"
        )

    overlap_modes = normalize_overlap_arg(args.overlap)

    if args.out_prefix is None:
        out_prefix = (
            Path(args.out_root) / args.pop / f"scatter_rg_annot_supp_{args.pop}"
        )
    else:
        out_prefix = Path(args.out_prefix)

    make_figure(
        df=df,
        pop=args.pop,
        annotations=annotations,
        methods=methods,
        scatter_triangle=args.scatter_triangle,
        min_common_pairs=args.min_common_pairs,
        out_prefix=out_prefix,
        overlap_modes=overlap_modes,
    )


if __name__ == "__main__":
    main()
