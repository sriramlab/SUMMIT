#!/usr/bin/env python3
from __future__ import annotations

"""
Make a publication-style supplementary figure highlighting representative
trait pairs with evidence for MAF-LD rg trends.

Design
------
- One multi-panel figure containing the top-ranked hit pairs.
- Within each pair panel:
  - x-axis is MAF group
  - each MAF group contains four horizontally separated LD-quartile points
  - lines connect the same LD quartile across MAF groups
  - error bars are shown for each bin-level rg estimate
  - pair-level directional and omnibus p-values are displayed in-panel
- No total-rg side panel is shown.

This figure is meant to be clean enough for a manuscript supplement. The
default bin setting is 8, but you can switch with `--numbins`.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gradient_statistics import (
    canonical_pair,
    infer_K_from_columns,
    infer_bin_names,
    parse_bin_name,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MAF_BIN_RE = re.compile(r"^m(?P<lo>\d+)_(?P<hi>\d+)_ld\d+$")

LABEL_OVERRIDE = {
    "bp_systolic": "SBP",
    "diastolic_blood_pressure": "DBP",
    "hypertension_i10": "HTN",
    "type2_diabetes_e11": "T2D",
    "dyslipidaemia_e78": "DISLIP",
    "alanine_aminotransferase": "ALT",
    "asp_at": "AST",
    "alka_phos": "ALP",
    "white_blood_cell_count": "WBC",
    "haemoglobin_concentration": "Hb",
    "heel_bone_mineral_density": "BMD",
    "educational_qualification": "Education",
    "fluid_intelligence_score": "Fluid IQ",
    "chronic_ischaemic_heart_disease_i25": "IHD",
}


def trait_label(x: str) -> str:
    return LABEL_OVERRIDE.get(str(x), str(x).replace("_", " "))


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.12)
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 400,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.3,
            "legend.title_fontsize": 9.3,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", facecolor="white")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight", facecolor="white")


def fmt_p(p: object) -> str:
    try:
        v = float(p)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    if v < 1e-3:
        return f"{v:.1e}"
    return f"{v:.3f}"


def pretty_maf_label(raw: str) -> str:
    m = MAF_BIN_RE.match(str(raw).strip())
    if m is None:
        return str(raw)
    lo = int(m.group("lo")) / 100.0
    hi = int(m.group("hi")) / 100.0
    lo_txt = "0.0" if abs(lo) < 1e-12 else f"{lo:.2f}"
    hi_txt = f"{hi:.2f}"
    return f"{lo_txt}-{hi_txt}"


def pretty_ld_label(raw: str) -> str:
    info = parse_bin_name(0, f"m000_001_{raw}")
    return f"LD Q{info.ld_index + 1}"


def build_bin_layout(
    bin_names: Sequence[str],
) -> Tuple[List[str], List[str], Dict[Tuple[int, int], int]]:
    infos = [parse_bin_name(i, name) for i, name in enumerate(bin_names)]
    maf_labels_raw: List[str] = []
    ld_labels_raw: List[str] = []
    maf_index: Dict[str, int] = {}
    ld_index: Dict[str, int] = {}
    idx_map: Dict[Tuple[int, int], int] = {}

    for info in infos:
        if info.maf_label not in maf_index:
            maf_index[info.maf_label] = len(maf_labels_raw)
            maf_labels_raw.append(info.bin_name)
        if info.ld_label not in ld_index:
            ld_index[info.ld_label] = len(ld_labels_raw)
            ld_labels_raw.append(info.ld_label)
        idx_map[(maf_index[info.maf_label], ld_index[info.ld_label])] = int(
            info.bin_index
        )

    maf_labels = [pretty_maf_label(x) for x in maf_labels_raw]
    ld_labels = [f"LD Q{i + 1}" for i in range(len(ld_labels_raw))]
    return maf_labels, ld_labels, idx_map


def load_pair_row(
    parsed: pd.DataFrame, *, pop: str, method: str, a: str, b: str
) -> pd.Series:
    sub = parsed[
        (parsed["pop"].astype(str) == str(pop))
        & (parsed["method"].astype(str) == str(method))
    ].copy()
    keys = sub.apply(lambda r: canonical_pair(r["phen1"], r["phen2"]), axis=1)
    sub = sub[keys == (str(a), str(b))]
    if sub.empty:
        raise KeyError(
            f"No parsed row found for pop={pop}, method={method}, pair=({a},{b})"
        )
    return sub.iloc[0]


def default_run_dir(numbins: int) -> Path:
    return (
        SCRIPT_DIR
        / "mafld_rg_gradient_pairs"
        / f"mafld_{int(numbins)}bins"
        / "pops_EUR_300k__methods_summit__maf_0_1"
    )


def default_csv_path(numbins: int) -> Path:
    return (
        SCRIPT_DIR.parent
        / "outs_real"
        / f"parsed_real_rg_mafld_{int(numbins)}bins_allpops.csv"
    )


def make_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--numbins",
        type=int,
        choices=[8, 24],
        default=8,
        help="MAFLD bin setting used to choose default run-dir/csv. Default: 8",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing pair_summary.csv and hit tables. Default depends on --numbins.",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Parsed MAFLD rg CSV matching the run. Default depends on --numbins.",
    )
    ap.add_argument(
        "--hit-file",
        default="pair_hits_directional_nominal.csv",
        help="Hit table inside run-dir used to choose candidate pairs.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=6,
        help="Maximum number of top pairs to include.",
    )
    ap.add_argument(
        "--min-frac-positive",
        type=float,
        default=0.75,
        help="Minimum fraction of LD panels with positive oriented effect to keep.",
    )
    ap.add_argument(
        "--min-valid-panels",
        type=int,
        default=3,
        help="Minimum number of valid LD panels required.",
    )
    ap.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of panel columns.",
    )
    ap.add_argument(
        "--error-bars",
        choices=["se", "ci95"],
        default="ci95",
        help="Error bars to plot for each rg estimate. Default: ci95",
    )
    ap.add_argument(
        "--outpath",
        default=None,
        help="Figure PDF path. Defaults to <run-dir>/supp_top_pairs_<hit-stem>.pdf",
    )
    return ap


def main() -> None:
    args = make_arg_parser().parse_args()
    set_plot_style()

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.numbins)
    pair_summary_path = run_dir / "pair_summary.csv"
    hit_path = run_dir / args.hit_file
    parsed_path = Path(args.csv) if args.csv else default_csv_path(args.numbins)

    if not pair_summary_path.exists():
        raise SystemExit(f"Missing {pair_summary_path}")
    if not hit_path.exists():
        raise SystemExit(f"Missing {hit_path}")
    if not parsed_path.exists():
        raise SystemExit(f"Missing {parsed_path}")

    pair_summary = pd.read_csv(pair_summary_path)
    hits = pd.read_csv(hit_path)
    parsed = pd.read_csv(parsed_path)

    hits = hits[
        (
            pd.to_numeric(hits.get("frac_positive_panels", np.nan), errors="coerce")
            >= float(args.min_frac_positive)
        )
        & (
            pd.to_numeric(hits.get("n_valid_panels", np.nan), errors="coerce")
            >= int(args.min_valid_panels)
        )
    ].copy()
    if hits.empty:
        raise SystemExit("No hit pairs remain after the requested filters.")

    hits = hits.sort_values(
        ["pair_gls_p_one_sided", "pair_omnibus_p", "a", "b"], kind="mergesort"
    ).reset_index(drop=True)
    hits = hits.head(int(args.top_n)).copy()
    hits["pair_label"] = hits.apply(
        lambda r: f"{trait_label(str(r['a']))} vs {trait_label(str(r['b']))}", axis=1
    )

    K = infer_K_from_columns(parsed)
    if K <= 0:
        raise SystemExit("Could not infer #bins from parsed CSV.")
    bin_names = infer_bin_names(parsed, K)
    maf_labels, ld_labels, idx_map = build_bin_layout(bin_names)
    n_maf = len(maf_labels)
    n_ld = len(ld_labels)

    group_gap = 2.35 if n_maf <= 2 else 1.55
    x_centers = np.arange(n_maf, dtype=float) * group_gap
    ld_offsets = np.linspace(-0.26, 0.26, n_ld)
    palette = sns.color_palette("colorblind", n_colors=n_ld)
    yerr_multiplier = 1.0 if args.error_bars == "se" else 1.96

    n_pairs = hits.shape[0]
    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(n_pairs / ncols))
    fig_w = 6.0 * ncols
    fig_h = 3.45 * nrows + 0.9
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False
    )
    axes_flat = axes.flatten()

    source_rows: List[Dict[str, object]] = []

    for ax_idx, meta in enumerate(hits.itertuples(index=False)):
        ax = axes_flat[ax_idx]
        row_idx = ax_idx // ncols
        col_idx = ax_idx % ncols
        row = load_pair_row(
            parsed,
            pop=str(meta.pop),
            method=str(meta.method),
            a=str(meta.a),
            b=str(meta.b),
        )
        valid_panel_labels = set(
            x for x in str(getattr(meta, "valid_panel_labels", "")).split(",") if x
        )

        for j in range(n_maf):
            ax.axvspan(
                x_centers[j] - 0.48,
                x_centers[j] + 0.48,
                color=("0.98" if j % 2 == 0 else "0.94"),
                zorder=0,
            )

        y_values: List[float] = []
        for ld_idx in range(n_ld):
            y = []
            se = []
            x = []
            ld_label_raw = f"ld{ld_idx + 1}"
            is_valid_panel = (
                (ld_label_raw in valid_panel_labels) if valid_panel_labels else True
            )
            alpha = 1.0 if is_valid_panel else 0.33
            lw = 2.3 if is_valid_panel else 1.5
            ls = "-" if is_valid_panel else "--"
            for maf_idx in range(n_maf):
                bin_idx = idx_map.get((maf_idx, ld_idx))
                if bin_idx is None:
                    continue
                rg = pd.to_numeric(
                    pd.Series([row.get(f"rg_bin_{bin_idx}", np.nan)]), errors="coerce"
                ).iloc[0]
                rg_se = pd.to_numeric(
                    pd.Series([row.get(f"rg_bin_se_{bin_idx}", np.nan)]),
                    errors="coerce",
                ).iloc[0]
                xpos = x_centers[maf_idx] + ld_offsets[ld_idx]
                x.append(xpos)
                y.append(rg)
                se.append(rg_se)
                if np.isfinite(rg):
                    y_values.append(float(rg))
                    if np.isfinite(rg_se):
                        yerr = yerr_multiplier * float(rg_se)
                        y_values.extend([float(rg) - yerr, float(rg) + yerr])
                source_rows.append(
                    {
                        "a": str(meta.a),
                        "b": str(meta.b),
                        "pair_label": str(meta.pair_label),
                        "ld_quartile": ld_label_raw,
                        "ld_label_plot": ld_labels[ld_idx],
                        "maf_group_index": maf_idx,
                        "maf_group_label": maf_labels[maf_idx],
                        "bin_index": int(bin_idx),
                        "bin_name": str(
                            row.get(f"bin_name_{bin_idx}", f"bin_{bin_idx}")
                        ),
                        "rg_bin": float(rg) if np.isfinite(rg) else np.nan,
                        "rg_bin_se": float(rg_se) if np.isfinite(rg_se) else np.nan,
                        "error_bars": str(args.error_bars),
                        "error_bar_multiplier": float(yerr_multiplier),
                        "rg_bin_yerr": yerr_multiplier * float(rg_se)
                        if np.isfinite(rg_se)
                        else np.nan,
                        "pair_gls_p_one_sided": float(meta.pair_gls_p_one_sided),
                        "pair_omnibus_p": float(meta.pair_omnibus_p),
                        "frac_positive_panels": float(meta.frac_positive_panels),
                    }
                )

            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            se_arr = np.asarray(se, dtype=float)
            ax.errorbar(
                x_arr,
                y_arr,
                yerr=yerr_multiplier * se_arr,
                color=palette[ld_idx],
                marker="o",
                markersize=5.0,
                markeredgewidth=0.0,
                capsize=3.0,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=3 if is_valid_panel else 2,
                label=ld_labels[ld_idx],
            )

        ax.axhline(0.0, color="0.35", lw=1.0, ls=":", zorder=1)
        ax.set_xticks(x_centers)
        if row_idx == nrows - 1:
            ax.set_xticklabels(maf_labels)
            ax.set_xlabel("MAF group")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        ax.set_xlim(x_centers[0] - 0.72, x_centers[-1] + 0.72)
        ax.set_title(
            f"{meta.pair_label} (p={fmt_p(meta.pair_gls_p_one_sided)})",
            loc="left",
            pad=8,
        )
        if y_values:
            lo = min(y_values)
            hi = max(y_values)
            pad = max(0.10 * (hi - lo), 0.055)
            ax.set_ylim(lo - pad, hi + pad)
        if col_idx == 0:
            ax.set_ylabel("rg")
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for k in range(n_pairs, len(axes_flat)):
        axes_flat[k].axis("off")

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=palette[i],
            marker="o",
            lw=2.3,
            markersize=5,
            label=ld_labels[i],
        )
        for i in range(n_ld)
    ]
    fig.legend(
        handles=handles,
        labels=ld_labels,
        loc="upper center",
        ncol=n_ld,
        frameon=False,
        bbox_to_anchor=(0.5, 0.972),
        title="LD strata",
    )

    hit_stem = Path(args.hit_file).stem
    outpath = (
        Path(args.outpath)
        if args.outpath
        else (run_dir / f"supp_top_pairs_{hit_stem}.pdf")
    )
    save_figure(fig, outpath)
    plt.close(fig)

    source_df = pd.DataFrame(source_rows)
    source_path = outpath.with_name(outpath.stem + "_source_data.csv")
    source_df.to_csv(source_path, index=False)

    print(f"[write] {outpath}")
    print(f"[write] {outpath.with_suffix('.png')}")
    print(f"[write] {source_path}")
    print(f"[done] pairs={n_pairs}")


if __name__ == "__main__":
    main()
