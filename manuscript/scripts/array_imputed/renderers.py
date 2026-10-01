"""Plot renderers for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import matplotlib as mpl


OUT_DIR = Path("figs/supplementary")

PANEL = "#222222"


GRID = "#d9d9d9"


ARRAY = "#4d4d4d"


IMPUTED = "#0072b2"


RARE = "#cc6677"


COMMON = "#88ccee"


KEY_TAU_ANNOTATIONS = [
    "Conserved_Primate_phastCons46way_w_flanking",
    "GTEx_eQTL_MaxCPP",
    "Ancient_Sequence_Age_Human_Promoter_w_flanking",
    "GERP.RSsup4",
    "BLUEPRINT_DNA_methylation_MaxCPP",
    "non_synonymous",
    "GERP.NS",
    "Backgrd_Selection_Stat",
    "CpG_Content_50kb",
    "Coding_UCSC_w_flanking",
    "Promoter_UCSC_w_flanking",
    "PromoterFlanking_Hoffman_w_flanking",
    "DHS_Trynka_w_flanking",
]


LABELS = {
    "Ancient_Sequence_Age_Human_Enhancer_w_flanking": "Ancient enhancer age",
    "Ancient_Sequence_Age_Human_Promoter_w_flanking": "Ancient promoter age",
    "Backgrd_Selection_Stat": "Background selection",
    "BivFlnk_w_flanking": "Bivalent flank",
    "BLUEPRINT_DNA_methylation_MaxCPP": "BLUEPRINT DNA methylation",
    "BLUEPRINT_H3K27acQTL_MaxCPP": "BLUEPRINT H3K27ac QTL",
    "BLUEPRINT_H3K4me1QTL_MaxCPP": "BLUEPRINT H3K4me1 QTL",
    "Coding_UCSC_w_flanking": "Coding + flank",
    "Conserved_LindbladToh_w_flanking": "Lindblad-Toh conserved",
    "Conserved_Mammal_phastCons46way_w_flanking": "Mammal phastCons",
    "Conserved_Primate_phastCons46way_w_flanking": "Primate phastCons",
    "Conserved_Vertebrate_phastCons46way_w_flanking": "Vertebrate phastCons",
    "CpG_Content_50kb": "CpG content, 50 kb",
    "DGF_ENCODE_w_flanking": "DGF ENCODE + flank",
    "DHS_peaks_Trynka": "DHS peaks",
    "DHS_Trynka_w_flanking": "DHS + flank",
    "Enhancer_Andersson_w_flanking": "Andersson enhancer",
    "Enhancer_Hoffman_w_flanking": "Hoffman enhancer",
    "FetalDHS_Trynka_w_flanking": "Fetal DHS + flank",
    "GERP.NS": "GERP.NS",
    "GERP.RSsup4": "GERP.RSsup4",
    "GTEx_eQTL_MaxCPP": "GTEx eQTL MaxCPP",
    "H3K27ac_Hnisz_w_flanking": "Hnisz H3K27ac",
    "H3K27ac_PGC2_w_flanking": "PGC2 H3K27ac",
    "H3K4me1_peaks_Trynka": "H3K4me1 peaks",
    "H3K4me1_Trynka_w_flanking": "H3K4me1 + flank",
    "H3K4me3_peaks_Trynka": "H3K4me3 peaks",
    "H3K4me3_Trynka_w_flanking": "H3K4me3 + flank",
    "H3K9ac_peaks_Trynka": "H3K9ac peaks",
    "H3K9ac_Trynka_w_flanking": "H3K9ac + flank",
    "Human_Enhancer_Villar_Species_Enhancer_Count": "Villar enhancer count",
    "Human_Enhancer_Villar_w_flanking": "Villar enhancer",
    "Human_Promoter_Villar_ExAC_w_flanking": "Villar promoter ExAC",
    "Human_Promoter_Villar_w_flanking": "Villar promoter",
    "Intron_UCSC_w_flanking": "Intron + flank",
    "non_synonymous": "Non-synonymous",
    "PromoterFlanking_Hoffman_w_flanking": "Promoter-flanking",
    "Promoter_UCSC_w_flanking": "Promoter + flank",
    "Repressed_Hoffman_w_flanking": "Repressed",
    "SuperEnhancer_Hnisz_w_flanking": "Super-enhancer",
    "synonymous": "Synonymous",
    "TFBS_ENCODE_w_flanking": "TFBS ENCODE + flank",
    "Transcr_Hoffman_w_flanking": "Transcribed",
    "TSS_Hoffman_w_flanking": "TSS",
    "UTR_3_UCSC_w_flanking": "3' UTR + flank",
    "UTR_5_UCSC_w_flanking": "5' UTR + flank",
    "WeakEnhancer_Hoffman_w_flanking": "Weak enhancer",
}


def short_label(annotation: str) -> str:
    return LABELS.get(
        annotation, annotation.replace("_w_flanking", "").replace("_", " ")
    )


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def add_identity(
    ax: mpl.axes.Axes, x: pd.Series, y: pd.Series, equal_aspect: bool = True
) -> None:
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    pad = 0.04 * (hi - lo)
    lo = max(0, lo - pad)
    hi = hi + pad
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def plot_total_h2_scatter(
    ax: mpl.axes.Axes,
    total: pd.DataFrame,
    label_top: bool = False,
    equal_aspect: bool = True,
) -> None:
    ax.errorbar(
        total["array_h2"],
        total["imp_h2"],
        xerr=total["array_h2_se"],
        yerr=total["imp_h2_se"],
        fmt="none",
        ecolor="#b5b5b5",
        elinewidth=1.0,
        capsize=1.8,
        alpha=0.72,
        zorder=1,
    )
    ax.scatter(total["array_h2"], total["imp_h2"], s=34, color=PANEL, alpha=0.76, lw=0)
    add_identity(ax, total["array_h2"], total["imp_h2"], equal_aspect=equal_aspect)
    ax.set_xlabel(r"Array total $h^2$")
    ax.set_ylabel(r"Imputed total $h^2$")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    pearson = total["array_h2"].corr(total["imp_h2"], method="pearson")
    spearman = total["array_h2"].corr(total["imp_h2"], method="spearman")
    ax.text(
        0.97,
        0.05,
        f"Pearson r = {pearson:.3f}\nSpearman $\\rho$ = {spearman:.3f}\n{len(total)} traits",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "lw": 0.6},
    )
    if not label_top:
        return
    offsets = {
        "height": ("HT", 9, -10, "left"),
        "educational_qualification": ("EA", -24, -17, "right"),
        "fluid_intelligence_score": ("FI", -22, 25, "right"),
        "urate": ("UA", -26, -14, "right"),
    }
    for phen, (label, dx, dy, ha) in offsets.items():
        row = total.loc[total["phen"] == phen].iloc[0]
        ax.annotate(
            label,
            xy=(row["array_h2"], row["imp_h2"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=11,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#777777"},
        )


def plot_log2_enrichment_scatter(
    ax: mpl.axes.Axes, meta: pd.DataFrame, equal_aspect: bool = True
) -> None:
    x = meta["array_log2_enrichment_meta"]
    y = meta["imputed_log2_enrichment_meta"]
    ax.errorbar(
        x,
        y,
        xerr=meta["array_log2_enrichment_se_meta"],
        yerr=meta["imputed_log2_enrichment_se_meta"],
        fmt="none",
        ecolor="#b5b5b5",
        elinewidth=1.0,
        capsize=1.6,
        alpha=0.70,
        zorder=1,
    )
    ax.scatter(x, y, s=34, color=PANEL, alpha=0.72, lw=0)
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Array log2 enrichment")
    ax.set_ylabel("Imputed log2 enrichment")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    pearson = x.corr(y, method="pearson")
    spearman = x.corr(y, method="spearman")
    ax.text(
        0.97,
        0.05,
        f"Pearson r = {pearson:.3f}\nSpearman $\\rho$ = {spearman:.3f}\n{len(meta)} annotations",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "lw": 0.6},
    )
    labels = {
        "non_synonymous": ("NS", 0.95, 3.72, "left"),
        "GTEx_eQTL_MaxCPP": ("GTEx", 2.20, 3.65, "left"),
        "Ancient_Sequence_Age_Human_Promoter_w_flanking": (
            "Promoter age",
            2.20,
            3.38,
            "left",
        ),
        "BLUEPRINT_DNA_methylation_MaxCPP": ("DNAm", 0.42, 2.92, "right"),
        "Conserved_Primate_phastCons46way_w_flanking": ("Primate", 0.34, 1.82, "right"),
        "Backgrd_Selection_Stat": ("BGS", 0.70, 0.22, "left"),
    }
    for ann, (txt, xtext, ytext, ha) in labels.items():
        row = meta.loc[meta["annotation"] == ann].iloc[0]
        ax.annotate(
            txt,
            xy=(row["array_log2_enrichment_meta"], row["imputed_log2_enrichment_meta"]),
            xytext=(xtext, ytext),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=10.5,
            arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "#777777"},
        )


def selected_tau_annotations(meta: pd.DataFrame) -> list[str]:
    top_array = meta.nlargest(8, "array_tau_star_meta")["annotation"].tolist()
    top_imp = meta.nlargest(10, "imputed_tau_star_meta")["annotation"].tolist()
    negative = [
        "Coding_UCSC_w_flanking",
        "Promoter_UCSC_w_flanking",
        "PromoterFlanking_Hoffman_w_flanking",
        "DHS_Trynka_w_flanking",
        "non_synonymous",
    ]
    selected = []
    for ann in [*top_imp, *top_array, *negative]:
        if ann not in selected:
            selected.append(ann)
    return selected


def plot_tau_dumbbell(
    ax: mpl.axes.Axes,
    meta: pd.DataFrame,
    annotations: list[str] | None = None,
    max_rows: int | None = 14,
) -> None:
    annotations = annotations or selected_tau_annotations(meta)
    data = meta[meta["annotation"].isin(annotations)].copy()
    data = data.sort_values("imputed_tau_star_meta", ascending=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.tail(max_rows)
    y = np.arange(len(data))
    ax.hlines(
        y,
        data["array_tau_star_meta"],
        data["imputed_tau_star_meta"],
        color="#bdbdbd",
        lw=1.6,
        zorder=1,
    )
    ax.scatter(
        data["array_tau_star_meta"], y, s=27, color=ARRAY, label="Array", zorder=2
    )
    ax.scatter(
        data["imputed_tau_star_meta"], y, s=30, color=IMPUTED, label="Imputed", zorder=3
    )
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(a) for a in data["annotation"]])
    ax.set_xlabel(r"Meta-analyzed $\tau^{*}$")
    ax.grid(True, axis="x", color=GRID, lw=0.5, alpha=0.7)
    ax.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#cccccc",
        loc="lower right",
    )


def plot_maf_composition_aligned_rare_left(
    ax: mpl.axes.Axes,
    variant_summary: pd.DataFrame,
    annotations: list[str],
    counts: pd.DataFrame,
) -> None:
    data = variant_summary.set_index("annotation").loc[annotations].reset_index()
    rare_pct = data["frac_maf_lt_0.01"] * 100
    common_pct = 100 - rare_pct
    y = np.arange(len(data))
    overall = counts.loc[counts["genotype"] == "imputed"].iloc[0]
    overall_rare_pct = overall["n_maf_lt_0.01"] / overall["n_variants"] * 100

    ax.barh(y, rare_pct, color=RARE, label=r"MAF $<$ 0.01")
    ax.barh(y, common_pct, left=rare_pct, color=COMMON, label=r"MAF $\geq$ 0.01")
    ax.axvline(overall_rare_pct, color="#666666", lw=1.1, ls="--")
    ax.text(
        overall_rare_pct + 1.2,
        1.015,
        f"All imputed: {overall_rare_pct:.1f}%",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.5, len(data) - 0.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of imputed variants")
    ax.grid(True, axis="x", color=GRID, lw=0.5, alpha=0.7)
    for i, pct in enumerate(rare_pct):
        ax.text(
            101,
            i,
            f"{pct:.1f}%",
            ha="left",
            va="center",
            fontsize=9.5,
            color="#555555",
        )
    ax.text(
        101,
        len(data) - 0.15,
        r"MAF $<$ 0.01",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(1.02, 0.02), ncol=1)


def figure_summary_2x2_maf_aligned_rare_left(
    total: pd.DataFrame,
    meta: pd.DataFrame,
    variant_summary_tau: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(14.4, 10.6), constrained_layout=False)
    left_x = 0.18
    right_x = 0.62
    top_y = 0.58
    bottom_y = 0.12
    width = 0.32
    height = 0.34
    ax_a = fig.add_axes([left_x, top_y, width, height])
    ax_b = fig.add_axes([right_x, top_y, width, height])
    ax_c = fig.add_axes([left_x, bottom_y, width, height])
    ax_d = fig.add_axes([right_x, bottom_y, width, height])

    plot_total_h2_scatter(ax_a, total, label_top=True, equal_aspect=False)
    plot_log2_enrichment_scatter(ax_b, meta, equal_aspect=False)
    plot_tau_dumbbell(ax_c, meta, annotations=KEY_TAU_ANNOTATIONS, max_rows=None)

    ordered_annotations = (
        meta[meta["annotation"].isin(KEY_TAU_ANNOTATIONS)]
        .sort_values("imputed_tau_star_meta", ascending=True)["annotation"]
        .tolist()
    )
    plot_maf_composition_aligned_rare_left(
        ax_d, variant_summary_tau, ordered_annotations, counts
    )
    ax_c.set_ylim(-0.5, len(ordered_annotations) - 0.5)
    ax_d.set_ylim(-0.5, len(ordered_annotations) - 0.5)

    label_offset_x = 0.035
    label_offset_y = 0.018
    for label, x, y in [
        ("A", left_x - label_offset_x, top_y + height + label_offset_y),
        ("B", right_x - label_offset_x, top_y + height + label_offset_y),
        ("C", left_x - label_offset_x, bottom_y + height + label_offset_y),
        ("D", right_x - label_offset_x, bottom_y + height + label_offset_y),
    ]:
        fig.text(
            x,
            y,
            label,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color=PANEL,
        )
    savefig(fig, "rendition10_summary_2x2_maf_aligned_rare_left")


def figure_tau_star_enrichment_rank_aligned(meta: pd.DataFrame) -> None:
    selected = selected_tau_annotations(meta)
    data = meta[meta["annotation"].isin(selected)].copy()
    data = data.sort_values(
        ["imputed_enrichment_rank", "array_enrichment_rank"], ascending=True
    )
    labels = [short_label(a) for a in data["annotation"]]
    y = np.arange(len(data))

    fig = plt.figure(
        figsize=(12.8, max(7.2, len(data) * 0.43)), constrained_layout=False
    )
    ax0 = fig.add_axes([0.28, 0.12, 0.20, 0.78])
    cax = fig.add_axes([0.495, 0.32, 0.014, 0.36])
    ax1 = fig.add_axes([0.60, 0.12, 0.34, 0.78])

    matrix = data[["array_tau_star_meta", "imputed_tau_star_meta"]].to_numpy()
    norm = TwoSlopeNorm(vmin=-0.55, vcenter=0, vmax=0.80)
    im = ax0.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto", origin="upper")
    ax0.set_ylim(len(data) - 0.5, -0.5)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Array", "Imputed"])
    ax0.set_title(r"Meta-analyzed $\tau^{*}$", pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.35 else "#222222"
            ax0.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\tau^{*}$")

    ax1.hlines(
        y,
        data["array_enrichment_rank"],
        data["imputed_enrichment_rank"],
        color="#bdbdbd",
        lw=1.7,
        zorder=1,
    )
    ax1.scatter(
        data["array_enrichment_rank"], y, color=ARRAY, s=46, label="Array", zorder=3
    )
    ax1.scatter(
        data["imputed_enrichment_rank"],
        y,
        color=IMPUTED,
        s=48,
        label="Imputed",
        zorder=4,
    )
    ax1.set_ylim(len(data) - 0.5, -0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([])
    ax1.set_xlabel("Enrichment rank, 1 = highest")
    ax1.set_title("Annotation enrichment rank", pad=10)
    ax1.set_xlim(48.5, 0.5)
    ax1.set_xticks([40, 30, 20, 10, 1])
    ax1.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.75)
    ax1.legend(
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#cccccc",
        loc="lower right",
    )

    fig.text(
        0.235,
        0.925,
        "A",
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color=PANEL,
    )
    fig.text(
        0.555,
        0.925,
        "B",
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color=PANEL,
    )
    savefig(fig, "rendition8_tau_star_enrichment_rank_aligned")
