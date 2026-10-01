#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from math import erf, sqrt


# ----------------------------
# Stats helpers
# ----------------------------

pop_map = {"EUR": "EUR", "AFR": "AFR", "EUR_300k": "EUR (300k)", "SAS": "SAS"}
TEST_FAMILY_SIZE = 780


def normal_two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    az = abs(z)
    phi = 0.5 * (1.0 + erf(az / sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return float(np.clip(p, 0.0, 1.0))


def bh_fdr_reject(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    reject = np.zeros_like(pvals, dtype=bool)
    finite = np.isfinite(pvals)
    pv = pvals[finite]
    if pv.size == 0:
        return reject

    m = TEST_FAMILY_SIZE
    if m < pv.size:
        raise ValueError(
            f"FDR family size must be >= number of finite p-values (got m={m}, finite={pv.size})."
        )

    order = np.argsort(pv)
    pv_sorted = pv[order]
    thresh = (np.arange(1, pv_sorted.size + 1) / m) * q
    ok = pv_sorted <= thresh
    if not np.any(ok):
        return reject

    cutoff = pv_sorted[int(np.max(np.where(ok)[0]))]
    reject[np.where(finite)[0]] = pv <= cutoff
    return reject


def compute_sig_masks(
    df: pd.DataFrame,
    traits: List[str],
    alpha: float = 0.05,
    bh_q: float = 0.05,
    threshold: str = "fdr",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (nominal_mask, threshold_mask) as boolean (T,T) matrices.

    FDR/Bonferroni is computed over the full input df using a hard-coded
    780-test family, then mapped onto the displayed trait subset. Masks are
    symmetric; diagonal is False.
    """
    idx = {t: i for i, t in enumerate(traits)}
    T = len(traits)
    nom = np.zeros((T, T), dtype=bool)
    fdr = np.zeros((T, T), dtype=bool)

    if df.empty:
        return nom, fdr

    def key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    tmp = df.copy()
    tmp["a"] = tmp.apply(lambda r: key(str(r["phen1"]), str(r["phen2"]))[0], axis=1)
    tmp["b"] = tmp.apply(lambda r: key(str(r["phen1"]), str(r["phen2"]))[1], axis=1)
    tmp = tmp.sort_values(["a", "b"])
    tmp = tmp.groupby(["a", "b"], as_index=False).first()

    pvals: List[float] = []
    for _, r in tmp.iterrows():
        rg = r.get("rg", np.nan)
        se = r.get("rg_se", np.nan)
        rg = float(rg) if rg is not None else np.nan
        se = float(se) if se is not None else np.nan
        if not (np.isfinite(rg) and np.isfinite(se) and se > 0):
            pvals.append(np.nan)
        else:
            pvals.append(normal_two_sided_p(rg / se))

    pvals_arr = np.asarray(pvals, dtype=float)
    if threshold == "fdr":
        fdr_reject = bh_fdr_reject(pvals_arr, q=bh_q)
    elif threshold == "bonf":
        fdr_reject = pvals_arr < (alpha / TEST_FAMILY_SIZE)
    else:
        raise ValueError(f"Unknown threshold: {threshold}")

    for (_, r), p, is_fdr in zip(tmp.iterrows(), pvals, fdr_reject):
        a, b = str(r["a"]), str(r["b"])
        if a not in idx or b not in idx or a == b:
            continue
        i, j = idx[a], idx[b]
        if np.isfinite(p) and p < alpha:
            nom[i, j] = nom[j, i] = True
        if bool(is_fdr):
            fdr[i, j] = fdr[j, i] = True

    np.fill_diagonal(nom, False)
    np.fill_diagonal(fdr, False)
    return nom, fdr


# ----------------------------
# Matrix construction
# ----------------------------


def build_rg_matrix(
    df: pd.DataFrame, traits: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    idx = {t: i for i, t in enumerate(traits)}
    T = len(traits)
    R = np.full((T, T), np.nan, dtype=float)
    SE = np.full((T, T), np.nan, dtype=float)
    np.fill_diagonal(R, 1.0)
    np.fill_diagonal(SE, 0.0)

    def key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    tmp = df.copy()
    tmp["a"] = tmp.apply(lambda r: key(str(r["phen1"]), str(r["phen2"]))[0], axis=1)
    tmp["b"] = tmp.apply(lambda r: key(str(r["phen1"]), str(r["phen2"]))[1], axis=1)
    tmp = tmp.sort_values(["a", "b"])
    tmp = tmp.groupby(["a", "b"], as_index=False).first()

    for _, r in tmp.iterrows():
        a, b = str(r["a"]), str(r["b"])
        if a not in idx or b not in idx or a == b:
            continue
        i, j = idx[a], idx[b]
        rg = r.get("rg", np.nan)
        se = r.get("rg_se", np.nan)
        rg = float(rg) if rg is not None else np.nan
        se = float(se) if se is not None else np.nan
        R[i, j] = rg
        R[j, i] = rg
        SE[i, j] = se
        SE[j, i] = se

    return R, SE


# ----------------------------
# Phen-list parsing (acronym + group ordering)
# ----------------------------


@dataclass(frozen=True)
class PhenMeta:
    field: str
    acronym: str
    group: str


def load_phen_list_meta(
    phen_list_path: Optional[str],
) -> Tuple[List[str], Dict[str, PhenMeta]]:
    """
    Expect lines like:
      albumin 30600 ALB biochem_liver
      asp_at 30650 AST biochem_liver
      blood_platelet 30080 PLT hematology
    """
    if not phen_list_path:
        return [], {}

    phen2meta: Dict[str, PhenMeta] = {}
    phen_order: List[str] = []

    with open(phen_list_path, "r") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue

            phen = parts[0]
            field = parts[1] if len(parts) >= 2 else ""
            acronym = parts[2] if len(parts) >= 3 else phen
            group = parts[3] if len(parts) >= 4 else "ungrouped"

            if phen not in phen2meta:
                phen2meta[phen] = PhenMeta(field=field, acronym=acronym, group=group)
                phen_order.append(phen)

    return phen_order, phen2meta


def build_grouped_trait_order_and_blocks(
    phen_order: List[str],
    phen2meta: Dict[str, PhenMeta],
    present_traits: set,
) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
    if not phen_order or not phen2meta:
        traits = sorted(list(present_traits))
        blocks = [("all", traits)] if traits else []
        labels = traits[:]
        return traits, blocks, labels

    group_to_traits: "OrderedDict[str, List[str]]" = OrderedDict()
    for phen in phen_order:
        if phen not in present_traits:
            continue
        g = phen2meta.get(
            phen, PhenMeta(field="", acronym=phen, group="ungrouped")
        ).group
        group_to_traits.setdefault(g, []).append(phen)

    traits: List[str] = []
    blocks: List[Tuple[str, List[str]]] = []
    labels: List[str] = []

    for g, ts in group_to_traits.items():
        if not ts:
            continue
        blocks.append((g, ts))
        traits.extend(ts)

    for t in traits:
        labels.append(
            phen2meta.get(t, PhenMeta(field="", acronym=t, group="ungrouped")).acronym
        )

    return traits, blocks, labels


def block_boundaries(blocks: List[Tuple[str, List[str]]]) -> List[int]:
    boundaries = []
    pos = 0
    for _, block_traits in blocks:
        pos += len(block_traits)
        boundaries.append(pos)
    return boundaries


# ----------------------------
# Plotting helpers
# ----------------------------


def get_population_order(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []

    pop_series = df["pop"]
    if isinstance(pop_series.dtype, pd.CategoricalDtype):
        cat = pop_series.dtype
        if getattr(cat, "ordered", False):
            categories = [str(x) for x in list(cat.categories)]
            present = set(pop_series.astype(str).tolist())
            return [p for p in categories if p in present]

    return [str(x) for x in pop_series.astype(str).drop_duplicates().tolist()]


def compute_layout_params(T: int, *, panel: bool = False) -> Dict[str, float]:
    if panel:
        label_fs = float(np.clip(11.5 - 0.13 * T, 5.8, 9.5))
        cbar_fs = float(np.clip(12.0 - 0.10 * T, 7.5, 10.0))
        title_fs = float(np.clip(15.0 - 0.08 * T, 10.0, 13.0))
        return {
            "label_fs": label_fs,
            "cbar_fs": cbar_fs,
            "title_fs": title_fs,
            "fdr_star_s": 90.0,
            "nom_dot_s": 18.0,
            "na_fontsize": 8.0,
            "grid_lw": 0.55,
            "block_lw": 1.05,
            "na_x_linewidth": 0.95,
            "discordant_outline_lw": 1.2,
        }

    label_fs = float(np.clip(18.0 - 0.15 * T, 10.5, 13.5))
    cbar_fs = float(np.clip(15.0 - 0.05 * T, 11.0, 13.5))
    title_fs = float(np.clip(16.0 - 0.06 * T, 12.0, 14.0))
    return {
        "label_fs": label_fs,
        "cbar_fs": cbar_fs,
        "title_fs": title_fs,
        "fdr_star_s": 190.0,
        "nom_dot_s": 35.0,
        "na_fontsize": 10.0,
        "grid_lw": 0.60,
        "block_lw": 1.20,
        "na_x_linewidth": 1.00,
        "discordant_outline_lw": 1.40,
    }


def draw_heatmap_ax(
    ax: plt.Axes,
    R: np.ndarray,
    nom_mask: np.ndarray,
    fdr_mask: np.ndarray,
    outline_mask: np.ndarray,
    labels: List[str],
    blocks: List[Tuple[str, List[str]]],
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    missing_color: str = "#f0f0f0",
    upper_sig_only: bool = True,
    show_na: bool = True,
    na_text: str = "",
    na_fontsize: float = 9.0,
    na_x_linewidth: float = 1.0,
    area_cap: float = 1.0,
    area_max_frac: float = 0.96,
    fdr_star_s: float = 140.0,
    nom_dot_s: float = 24.0,
    discordant_outline_lw: float = 1.4,
    diag_color: str = "#d9d9d9",
    diag_edgecolor: str = "none",
    diag_linewidth: float = 0.0,
    diag_hatch: str = "",
    diag_alpha: float = 1.0,
    label_fs: float = 10.0,
    grid_lw: float = 0.6,
    block_lw: float = 1.2,
    show_xlabels: bool = True,
    show_ylabels: bool = True,
):
    T = R.shape[0]

    cm = mpl.colormaps.get_cmap(cmap).copy()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(T - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for i in range(T):
        for j in range(T):
            val = R[i, j]

            if i == j and str(diag_color).lower() != "cmap":
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=diag_color,
                        edgecolor=diag_edgecolor,
                        linewidth=diag_linewidth,
                        hatch=diag_hatch,
                        alpha=diag_alpha,
                        zorder=1,
                    )
                )
                continue

            base_fc = "white" if np.isfinite(val) else missing_color
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=base_fc,
                    edgecolor="none",
                    zorder=1,
                )
            )

            if not np.isfinite(val) or i == j:
                continue

            if area_cap <= 0:
                raise ValueError("area_cap must be > 0")

            mag = min(float(abs(val)), float(area_cap))
            if mag <= 0.0:
                continue

            side = area_max_frac * sqrt(mag / float(area_cap))
            side = min(side, area_max_frac)

            color_val = float(np.clip(val, vmin, vmax))
            fc = cm(norm(color_val))

            do_outline = bool(outline_mask[i, j])
            ec = "black" if do_outline else "none"
            lw = discordant_outline_lw if do_outline else 0.0

            ax.add_patch(
                Rectangle(
                    (j - side / 2.0, i - side / 2.0),
                    side,
                    side,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=lw,
                    zorder=2,
                )
            )

    ax.set_xticks(np.arange(T))
    ax.set_yticks(np.arange(T))
    ax.set_xticklabels(
        labels if show_xlabels else [],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=label_fs,
    )
    ax.set_yticklabels(labels if show_ylabels else [], fontsize=label_fs)

    ax.tick_params(
        axis="x", which="major", bottom=show_xlabels, labelbottom=show_xlabels, length=0
    )
    ax.tick_params(
        axis="y", which="major", left=show_ylabels, labelleft=show_ylabels, length=0
    )

    ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    for b in block_boundaries(blocks):
        if b <= 0 or b >= T:
            continue
        ax.axhline(b - 0.5, color="black", linewidth=block_lw, zorder=5)
        ax.axvline(b - 0.5, color="black", linewidth=block_lw, zorder=5)

    def _sig_visible(i: int, j: int) -> bool:
        if i == j:
            return False
        if upper_sig_only:
            return i < j
        return True

    if show_na:
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                if not np.isfinite(R[i, j]):
                    if na_text:
                        ax.text(
                            j,
                            i,
                            na_text,
                            ha="center",
                            va="center",
                            fontsize=na_fontsize,
                            color="black",
                            zorder=10,
                        )
                    x0, x1 = j - 0.45, j + 0.45
                    y0, y1 = i - 0.45, i + 0.45
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color="black",
                        linewidth=na_x_linewidth,
                        alpha=0.75,
                        zorder=10,
                    )
                    ax.plot(
                        [x0, x1],
                        [y1, y0],
                        color="black",
                        linewidth=na_x_linewidth,
                        alpha=0.75,
                        zorder=10,
                    )

    for i in range(T):
        for j in range(T):
            if not _sig_visible(i, j):
                continue
            if not np.isfinite(R[i, j]) or i == j:
                continue

            if bool(fdr_mask[i, j]):
                ax.scatter(
                    [j],
                    [i],
                    marker="*",
                    s=fdr_star_s,
                    c="black",
                    linewidths=0.0,
                    zorder=25,
                )
            elif bool(nom_mask[i, j]):
                ax.scatter(
                    [j],
                    [i],
                    marker="o",
                    s=nom_dot_s,
                    c="black",
                    linewidths=0.0,
                    zorder=25,
                )

    return mpl.cm.ScalarMappable(norm=norm, cmap=cm)


def plot_heatmap(
    R: np.ndarray,
    nom_mask: np.ndarray,
    fdr_mask: np.ndarray,
    outline_mask: np.ndarray,
    labels: List[str],
    outpath: Path,
    blocks: List[Tuple[str, List[str]]],
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    missing_color: str = "#f0f0f0",
    upper_sig_only: bool = True,
    dpi: int = 220,
    show_na: bool = True,
    na_text: str = "",
    na_fontsize: int = 9,
    na_x_linewidth: float = 1.0,
    area_cap: float = 1.0,
    area_max_frac: float = 0.96,
    fdr_star_s: float = 140.0,
    nom_dot_s: float = 24.0,
    discordant_outline_lw: float = 1.4,
    diag_color: str = "#d9d9d9",
    diag_edgecolor: str = "none",
    diag_linewidth: float = 0.0,
    diag_hatch: str = "",
    diag_alpha: float = 1.0,
):
    T = R.shape[0]
    layout = compute_layout_params(T, panel=False)

    fig_w = max(10.0, 0.30 * T + 3.2)
    fig_h = max(9.0, 0.28 * T + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sm = draw_heatmap_ax(
        ax=ax,
        R=R,
        nom_mask=nom_mask,
        fdr_mask=fdr_mask,
        outline_mask=outline_mask,
        labels=labels,
        blocks=blocks,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        missing_color=missing_color,
        upper_sig_only=upper_sig_only,
        show_na=show_na,
        na_text=na_text,
        na_fontsize=na_fontsize,
        na_x_linewidth=na_x_linewidth,
        area_cap=area_cap,
        area_max_frac=area_max_frac,
        fdr_star_s=fdr_star_s,
        nom_dot_s=nom_dot_s,
        discordant_outline_lw=discordant_outline_lw,
        diag_color=diag_color,
        diag_edgecolor=diag_edgecolor,
        diag_linewidth=diag_linewidth,
        diag_hatch=diag_hatch,
        diag_alpha=diag_alpha,
        label_fs=layout["label_fs"],
        grid_lw=layout["grid_lw"],
        block_lw=layout["block_lw"],
        show_xlabels=True,
        show_ylabels=True,
    )

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Genetic correlation (rg)", fontsize=layout["cbar_fs"])
    cbar.ax.tick_params(labelsize=layout["cbar_fs"])

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_panel(
    payloads: List[Tuple[str, Dict[str, Any]]],
    labels: List[str],
    outpath: Path,
    blocks: List[Tuple[str, List[str]]],
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    missing_color: str = "#f0f0f0",
    dpi: int = 220,
    show_na: bool = True,
    na_text: str = "",
    area_cap: float = 1.0,
    area_max_frac: float = 0.96,
    diag_color: str = "#d9d9d9",
    diag_edgecolor: str = "none",
    diag_linewidth: float = 0.0,
    diag_hatch: str = "",
    diag_alpha: float = 1.0,
    shared_axes: bool = False,
):
    if len(payloads) != 4:
        raise ValueError(
            f"panel mode expects exactly 4 populations, got {len(payloads)}"
        )

    T = len(labels)
    layout = compute_layout_params(T, panel=True)

    ax_w = max(5.8, 0.19 * T + 2.4)
    ax_h = max(5.6, 0.18 * T + 2.2)
    fig_w = 2.0 * ax_w + 1.4
    fig_h = 2.0 * ax_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.05],
        wspace=0.10 if shared_axes else 0.18,
        hspace=0.12 if shared_axes else 0.20,
    )

    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax = fig.add_subplot(gs[:, 2])

    sm = None
    for idx, (pop, payload) in enumerate(payloads):
        r = idx // 2
        c = idx % 2
        ax = axes[r, c]

        show_xlabels = True
        show_ylabels = True
        if shared_axes:
            show_xlabels = r == 1
            show_ylabels = c == 0

        sm = draw_heatmap_ax(
            ax=ax,
            R=payload["R"],
            nom_mask=payload["nom"],
            fdr_mask=payload["fdr"],
            outline_mask=payload["outline"],
            labels=labels,
            blocks=blocks,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            missing_color=missing_color,
            upper_sig_only=payload["upper_sig_only"],
            show_na=show_na,
            na_text=na_text,
            na_fontsize=layout["na_fontsize"],
            na_x_linewidth=layout["na_x_linewidth"],
            area_cap=area_cap,
            area_max_frac=area_max_frac,
            fdr_star_s=layout["fdr_star_s"],
            nom_dot_s=layout["nom_dot_s"],
            discordant_outline_lw=layout["discordant_outline_lw"],
            diag_color=diag_color,
            diag_edgecolor=diag_edgecolor,
            diag_linewidth=diag_linewidth,
            diag_hatch=diag_hatch,
            diag_alpha=diag_alpha,
            label_fs=layout["label_fs"],
            grid_lw=layout["grid_lw"],
            block_lw=layout["block_lw"],
            show_xlabels=show_xlabels,
            show_ylabels=show_ylabels,
        )
        ax.set_title(pop_map[str(pop)], fontsize=layout["title_fs"], pad=8.0)

    if sm is None:
        raise RuntimeError("No heatmap payloads were rendered.")

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Genetic correlation (rg)", fontsize=layout["cbar_fs"])
    cbar.ax.tick_params(labelsize=layout["cbar_fs"])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Data preparation for plotting
# ----------------------------


def prepare_plot_payload(
    dpop: pd.DataFrame,
    *,
    compare_annots: bool,
    annot_a: str,
    annot_b: Optional[str],
    method: str,
    traits: List[str],
    alpha: float,
    bh_q: float,
    threshold: str,
    upper_stars_only: bool,
) -> Dict[str, Any]:
    T = len(traits)

    if compare_annots:
        dA = dpop[dpop["annot_type"] == annot_a].copy()
        dB = dpop[dpop["annot_type"] == annot_b].copy()

        R_A, SE_A = build_rg_matrix(dA, traits)
        R_B, SE_B = build_rg_matrix(dB, traits)

        nom_A, fdr_A = compute_sig_masks(
            dA, traits, alpha=alpha, bh_q=bh_q, threshold=threshold
        )
        nom_B, fdr_B = compute_sig_masks(
            dB, traits, alpha=alpha, bh_q=bh_q, threshold=threshold
        )

        R = np.full((T, T), np.nan, dtype=float)
        np.fill_diagonal(R, 1.0)
        for i in range(T):
            for j in range(T):
                if i < j:
                    R[i, j] = R_B[i, j]
                elif i > j:
                    R[i, j] = R_A[i, j]

        nom = np.zeros((T, T), dtype=bool)
        fdr = np.zeros((T, T), dtype=bool)
        outline = np.zeros((T, T), dtype=bool)

        discordant = np.logical_xor(fdr_A, fdr_B)

        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                if i < j:
                    nom[i, j] = bool(nom_B[i, j])
                    fdr[i, j] = bool(fdr_B[i, j])
                    outline[i, j] = bool(discordant[i, j] and fdr_B[i, j])
                elif i > j:
                    nom[i, j] = bool(nom_A[i, j])
                    fdr[i, j] = bool(fdr_A[i, j])
                    outline[i, j] = bool(discordant[i, j] and fdr_A[i, j])

        upper_sig_only = False

    else:
        if method == "both":
            dsum = dpop[dpop["method"] == "summit"].copy()
            dcov = dpop[dpop["method"] == "covldsc"].copy()

            R_sum, SE_sum = build_rg_matrix(dsum, traits)
            R_cov, SE_cov = build_rg_matrix(dcov, traits)

            nom_sum, fdr_sum = compute_sig_masks(
                dsum, traits, alpha=alpha, bh_q=bh_q, threshold=threshold
            )
            nom_cov, fdr_cov = compute_sig_masks(
                dcov, traits, alpha=alpha, bh_q=bh_q, threshold=threshold
            )

            R = np.full((T, T), np.nan, dtype=float)
            np.fill_diagonal(R, 1.0)
            for i in range(T):
                for j in range(T):
                    if i < j:
                        R[i, j] = R_cov[i, j]
                    elif i > j:
                        R[i, j] = R_sum[i, j]

            nom = np.zeros((T, T), dtype=bool)
            fdr = np.zeros((T, T), dtype=bool)
            outline = np.zeros((T, T), dtype=bool)

            discordant = np.logical_xor(fdr_sum, fdr_cov)

            for i in range(T):
                for j in range(T):
                    if i == j:
                        continue
                    if i < j:
                        nom[i, j] = bool(nom_cov[i, j])
                        fdr[i, j] = bool(fdr_cov[i, j])
                        outline[i, j] = bool(discordant[i, j] and fdr_cov[i, j])
                    elif i > j:
                        nom[i, j] = bool(nom_sum[i, j])
                        fdr[i, j] = bool(fdr_sum[i, j])
                        outline[i, j] = bool(discordant[i, j] and fdr_sum[i, j])

            upper_sig_only = False

        else:
            R, SE = build_rg_matrix(dpop, traits)
            nom, fdr = compute_sig_masks(
                dpop, traits, alpha=alpha, bh_q=bh_q, threshold=threshold
            )
            outline = np.zeros((T, T), dtype=bool)
            upper_sig_only = bool(upper_stars_only)

    return {
        "R": R,
        "nom": nom,
        "fdr": fdr,
        "outline": outline,
        "upper_sig_only": upper_sig_only,
    }


def make_single_outpath(
    outdir: Path,
    *,
    compare_annots: bool,
    annot_a: str,
    annot_b: Optional[str],
    method: str,
    pop: str,
    threshold: str,
) -> Path:
    if compare_annots:
        return (
            outdir / f"rg_heatmap_{pop}_{method}_{annot_a}_vs_{annot_b}_{threshold}.png"
        )
    if method == "both":
        return outdir / f"rg_heatmap_{pop}_both_{annot_a}_{threshold}.png"
    return outdir / f"rg_heatmap_{pop}_{method}_{annot_a}_{threshold}.png"


def make_panel_outpath(
    outdir: Path,
    *,
    compare_annots: bool,
    annot_a: str,
    annot_b: Optional[str],
    method: str,
    threshold: str,
    shared_axes: bool,
) -> Path:
    suffix = f"_{threshold}" + ("_sharedaxes" if shared_axes else "")
    if compare_annots:
        return outdir / f"rg_heatmap_panel_{method}_{annot_a}_vs_{annot_b}{suffix}.png"
    if method == "both":
        return outdir / f"rg_heatmap_panel_both_{annot_a}{suffix}.png"
    return outdir / f"rg_heatmap_panel_{method}_{annot_a}{suffix}.png"


# ----------------------------
# main
# ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="data/real_rg/estimates.csv",
        help="CSV with required columns: pop, method, annot_type, phen1, phen2, rg, rg_se",
    )
    ap.add_argument(
        "--phen-list",
        default="data/traits/trait_categories.tsv",
        help="Phen list with 4 columns: phen field acronym group (used for ordering, grouping, and labels)",
    )
    ap.add_argument(
        "--method",
        default="both",
        choices=["both", "covldsc", "summit"],
        help=(
            "If --annot is a single annotation: "
            "'both' plots lower=summit and upper=covldsc; otherwise plots a single method.\n"
            "If --annot has TWO annotations (comma-separated): you MUST set --method to 'summit' or 'covldsc' "
            "and the plot will be lower=first annot, upper=second annot for that method."
        ),
    )
    ap.add_argument(
        "--annot",
        default="mafld_8bins",
        help="Either ONE annot (e.g., 'mafld_24bins') or TWO comma-separated annots (e.g., 'single,mafld_8bins').",
    )
    ap.add_argument(
        "--outdir", default="figs/supplementary", help="Output directory for figures"
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="Nominal alpha")
    ap.add_argument(
        "--bh-q",
        type=float,
        default=0.05,
        help=f"BH/FDR q using a fixed {TEST_FAMILY_SIZE}-test family",
    )
    ap.add_argument(
        "--pops",
        nargs="+",
        default=["EUR_300k", "EUR", "SAS", "AFR"],
        help="Population(s) to render, in order.",
    )
    ap.add_argument(
        "--threshold",
        choices=["fdr", "bonf"],
        default="fdr",
        help=f"Star/outline threshold using a fixed {TEST_FAMILY_SIZE}-test family.",
    )
    ap.add_argument(
        "--upper-stars-only",
        type=int,
        default=1,
        help="1: markers only on upper triangle (single-method single-annot). In comparison plots, markers are forced on BOTH triangles.",
    )
    ap.add_argument(
        "--panel",
        action="store_true",
        help="Pack the 4 population heatmaps into a single 2x2 panel with one shared colorbar.",
    )
    ap.add_argument(
        "--shared-axes",
        action="store_true",
        help="With --panel, hide interior tick labels and keep them only on the left column and bottom row.",
    )
    ap.add_argument("--vmin", type=float, default=-1.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument(
        "--area-cap",
        type=float,
        default=1.0,
        help="Cap |rg| used for area encoding (default 1.0).",
    )
    ap.add_argument(
        "--area-max-frac",
        type=float,
        default=0.96,
        help="Max inner square side as fraction of cell when |rg|>=cap.",
    )
    ap.add_argument(
        "--diag-color",
        default="#d9d9d9",
        help='Diagonal fill color (default grey). Use "cmap" to color diagonal using the rg colormap.',
    )
    ap.add_argument(
        "--diag-edgecolor", default="none", help="Diagonal edge color (default none)."
    )
    ap.add_argument(
        "--diag-linewidth", type=float, default=0.0, help="Diagonal edge linewidth."
    )
    ap.add_argument(
        "--diag-hatch",
        default="",
        help="Diagonal hatch pattern (e.g., '/', 'xx', etc.).",
    )
    ap.add_argument(
        "--diag-alpha", type=float, default=1.0, help="Diagonal alpha (0..1)."
    )
    args = ap.parse_args()

    if args.shared_axes and not args.panel:
        raise SystemExit("Error: --shared-axes only applies together with --panel.")

    df = pd.read_csv(args.csv)

    need_cols = {"pop", "method", "annot_type", "phen1", "phen2", "rg", "rg_se"}
    missing = sorted(list(need_cols - set(df.columns)))
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    allowed_annots = {"baseline", "celltype", "mafld_8bins", "mafld_24bins", "single"}

    annots = [a.strip() for a in str(args.annot).split(",") if a.strip()]
    if len(annots) == 0:
        raise SystemExit("Error: --annot is empty.")
    if len(annots) > 2:
        raise SystemExit(
            f"Error: --annot supports at most 2 annotations (got {len(annots)})."
        )

    for a in annots:
        if a not in allowed_annots:
            raise SystemExit(
                f"Error: unsupported annot '{a}'. Allowed: {sorted(allowed_annots)}"
            )

    compare_annots = len(annots) == 2
    annot_a = annots[0]
    annot_b = annots[1] if compare_annots else None

    if compare_annots and args.method == "both":
        raise SystemExit(
            "Error: when passing TWO annotations in --annot, you must set --method to 'summit' or 'covldsc' (not 'both')."
        )

    outdir = Path(args.outdir)

    # ----------------------------
    # Filter DF + determine present traits
    # ----------------------------

    if compare_annots:
        df_f = df[
            (df["method"] == args.method) & (df["annot_type"].isin([annot_a, annot_b]))
        ].copy()
    else:
        df_f = df[df["annot_type"] == annot_a].copy()
        if args.method != "both":
            df_f = df_f[df_f["method"] == args.method].copy()

    pops = [str(pop) for pop in args.pops]
    if not pops:
        raise SystemExit(
            "No rows after filtering (check --annot/--method and that the CSV contains those results)."
        )

    present_traits = set(df_f["phen1"].astype(str)).union(
        set(df_f["phen2"].astype(str))
    )
    if not present_traits:
        raise SystemExit("No traits present after filtering by --annot/--method.")

    phen_order, phen2meta = load_phen_list_meta(args.phen_list)
    traits, blocks, labels = build_grouped_trait_order_and_blocks(
        phen_order, phen2meta, present_traits
    )

    if len(traits) < 2:
        raise SystemExit(
            "Need at least 2 traits after phen-list/presence filtering to plot a heatmap."
        )

    trait_set = set(traits)
    df_plot = df_f[df_f["phen1"].isin(trait_set) & df_f["phen2"].isin(trait_set)].copy()
    if df_plot.empty:
        raise SystemExit(
            "No rows remain after restricting to phen-list traits (check phen codes)."
        )

    if args.panel:
        if len(pops) != 4:
            raise SystemExit(
                f"Error: --panel expects exactly 4 populations after filtering; got {len(pops)} ({pops})."
            )

        panel_payloads: List[Tuple[str, Dict[str, Any]]] = []
        for pop in pops:
            dpop_plot = df_plot[df_plot["pop"].astype(str) == pop].copy()
            dpop = df_f[df_f["pop"].astype(str) == pop].copy()
            if dpop_plot.empty:
                raise SystemExit(
                    f"Error: no rows remain for population '{pop}' after filtering."
                )
            payload = prepare_plot_payload(
                dpop,
                compare_annots=compare_annots,
                annot_a=annot_a,
                annot_b=annot_b,
                method=args.method,
                traits=traits,
                alpha=args.alpha,
                bh_q=args.bh_q,
                threshold=args.threshold,
                upper_stars_only=bool(args.upper_stars_only),
            )
            panel_payloads.append((pop, payload))

        outpath = make_panel_outpath(
            outdir,
            compare_annots=compare_annots,
            annot_a=annot_a,
            annot_b=annot_b,
            method=args.method,
            threshold=args.threshold,
            shared_axes=bool(args.shared_axes),
        )

        plot_heatmap_panel(
            payloads=panel_payloads,
            labels=labels,
            outpath=outpath,
            blocks=blocks,
            vmin=args.vmin,
            vmax=args.vmax,
            area_cap=args.area_cap,
            area_max_frac=args.area_max_frac,
            diag_color=args.diag_color,
            diag_edgecolor=args.diag_edgecolor,
            diag_linewidth=args.diag_linewidth,
            diag_hatch=args.diag_hatch,
            diag_alpha=args.diag_alpha,
            shared_axes=bool(args.shared_axes),
        )
        print(f"[write] {outpath}")
        print("Done.")
        return

    for pop in pops:
        dpop_plot = df_plot[df_plot["pop"].astype(str) == pop].copy()
        dpop = df_f[df_f["pop"].astype(str) == pop].copy()
        if dpop_plot.empty:
            continue

        payload = prepare_plot_payload(
            dpop,
            compare_annots=compare_annots,
            annot_a=annot_a,
            annot_b=annot_b,
            method=args.method,
            traits=traits,
            alpha=args.alpha,
            bh_q=args.bh_q,
            threshold=args.threshold,
            upper_stars_only=bool(args.upper_stars_only),
        )

        outpath = make_single_outpath(
            outdir,
            compare_annots=compare_annots,
            annot_a=annot_a,
            annot_b=annot_b,
            method=args.method,
            pop=pop,
            threshold=args.threshold,
        )

        plot_heatmap(
            R=payload["R"],
            nom_mask=payload["nom"],
            fdr_mask=payload["fdr"],
            outline_mask=payload["outline"],
            labels=labels,
            outpath=outpath,
            blocks=blocks,
            vmin=args.vmin,
            vmax=args.vmax,
            upper_sig_only=payload["upper_sig_only"],
            area_cap=args.area_cap,
            area_max_frac=args.area_max_frac,
            diag_color=args.diag_color,
            diag_edgecolor=args.diag_edgecolor,
            diag_linewidth=args.diag_linewidth,
            diag_hatch=args.diag_hatch,
            diag_alpha=args.diag_alpha,
        )
        print(f"[write] {outpath}")

    print("Done.")


if __name__ == "__main__":
    main()
