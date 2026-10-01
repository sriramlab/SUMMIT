"""Plot figure8 for the SUMMIT manuscript."""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import reference_swap as base
import diagnostic
import enrichment_layout as merged

LD_SUMMARY = Path("data/external/ld_components.tsv")

OUT_DIR = Path("figs/main")

OUT_STEM = "fig08_external_enrichment"


def load_component_means() -> pd.DataFrame:
    """Load the 59 BaselineLD component means used in the diagnostic figure."""
    df = pd.read_csv(LD_SUMMARY, sep="\t")
    out = df[
        (df["panel_id"] == "baselineLDv22_mvp_gia_eur_common_sas")
        & (df["row_type"] == "annotation_component")
    ].copy()
    required = {"metric_name", "eur_mean", "sas_mean", "component_label"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"missing LD-summary columns: {sorted(missing)}")
    out["is_maf"] = out["metric_name"].str.fullmatch(r"MAFbin\d+")
    out = out[
        np.isfinite(out["eur_mean"])
        & np.isfinite(out["sas_mean"])
        & (out["eur_mean"] > 0)
        & (out["sas_mean"] > 0)
    ].copy()
    if len(out) != 59 or int(out["is_maf"].sum()) != 10:
        raise ValueError(
            "expected 59 BaselineLD components including 10 MAF bins; "
            f"observed {len(out)} components including {int(out['is_maf'].sum())} MAF bins"
        )
    return out


def no_intercept_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xx = float(np.dot(x, x))
    yy = float(np.dot(y, y))
    xy = float(np.dot(x, y))
    slope = xy / xx
    r2 = (xy * xy) / (xx * yy)
    return slope, r2


def draw_component_means(
    ax: plt.Axes, df: pd.DataFrame, *, fit_non_maf: bool = False
) -> None:
    """Draw BaselineLD component-mean LD scores for the two references."""
    non_maf = df[~df["is_maf"]]
    maf = df[df["is_maf"]]

    ax.scatter(
        non_maf["eur_mean"],
        non_maf["sas_mean"],
        s=30,
        color=diagnostic.NON_MAF_COLOR,
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        maf["eur_mean"],
        maf["sas_mean"],
        s=48,
        marker="^",
        color=diagnostic.MAF_COLOR,
        alpha=0.82,
        linewidths=0,
        zorder=4,
    )

    fit_df = non_maf if fit_non_maf else df
    x = fit_df["eur_mean"].to_numpy(dtype=float)
    y = fit_df["sas_mean"].to_numpy(dtype=float)
    slope, r2 = no_intercept_stats(x, y)
    all_x = df["eur_mean"].to_numpy(dtype=float)
    all_y = df["sas_mean"].to_numpy(dtype=float)
    low = float(min(all_x.min(), all_y.min()) * 0.80)
    high = float(max(all_x.max(), all_y.max()) * 1.18)
    line = np.geomspace(low, high, 300)
    ax.plot(line, line, ls="--", lw=1.2, color="#333333", zorder=1)
    ax.plot(line, slope * line, lw=1.5, color="#b2182b", zorder=2)

    label_offsets = {
        "base": (-20, 9),
        "MAFbin1": (-17, 10),
        "MAFbin5": (10, -24),
        "MAFbin10": (8, -13),
    }
    for metric_name, offset in label_offsets.items():
        row = df.loc[df["metric_name"] == metric_name]
        if row.empty:
            continue
        label = (
            "Base"
            if metric_name == "base"
            else metric_name.replace("MAFbin", "MAF bin ")
        )
        ax.annotate(
            label,
            (float(row.iloc[0]["eur_mean"]), float(row.iloc[0]["sas_mean"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.7,
            color="#374151",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] > 0 else "top",
            arrowprops={
                "arrowstyle": "-",
                "color": "#6b7280",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("EUR (300k) mean LD score")
    ax.set_ylabel("SAS mean LD score")
    ax.set_title(
        "Mean LD score by BaselineLD component\n"
        + ("Non-MAF fit: " if fit_non_maf else "")
        + (
            rf"$R^2={r2:.5f}$, slope $={slope:.3f}$"
            if fit_non_maf
            else rf"$R^2={r2:.3f}$, slope $={slope:.3f}$"
        ),
        fontsize=11.0,
    )
    ax.grid(True, color="#e1e5ea", linewidth=0.8)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=diagnostic.NON_MAF_COLOR,
            markeredgecolor="none",
            markersize=5.7,
            alpha=0.75,
            label=f"Non-MAF components (n={len(non_maf)})",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=diagnostic.MAF_COLOR,
            markeredgecolor="none",
            markersize=6.5,
            alpha=0.85,
            label=f"MAF bins (n={len(maf)})",
        ),
        Line2D([0], [0], color="#333333", lw=1.2, ls="--", label="Identity line"),
        Line2D(
            [0],
            [0],
            color="#b2182b",
            lw=1.5,
            label=(
                "Non-MAF fit through origin" if fit_non_maf else "Fit through origin"
            ),
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        framealpha=0.94,
        fontsize=8.4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-non-maf", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--out-stem", default=OUT_STEM)
    args = parser.parse_args()

    base.set_style()
    enrich_df, enrich_jack = merged.load_enrichment()
    positive, non_maf, maf = base.load_reference_swap()
    component_means = load_component_means()

    fig = plt.figure(figsize=(12.4, 9.2))
    top = fig.add_gridspec(
        1,
        1,
        left=0.225,
        right=0.985,
        bottom=0.565,
        top=0.950,
    )
    bottom = fig.add_gridspec(
        1,
        2,
        left=0.080,
        right=0.985,
        bottom=0.085,
        top=0.455,
        width_ratios=[1.0, 1.0],
        wspace=0.30,
    )

    enrich_axes = merged.draw_enrichment_block(fig, top[0], enrich_df, enrich_jack)
    merged.add_enrichment_legend(fig, enrich_axes)

    scatter_ax = fig.add_subplot(bottom[0])
    ld_ax = fig.add_subplot(bottom[1])
    base.draw_supplementary_scatter(scatter_ax, positive, non_maf, maf)
    draw_component_means(ld_ax, component_means, fit_non_maf=args.fit_non_maf)
    base.format_diagnostic_axis(scatter_ax)
    base.format_diagnostic_axis(ld_ax)

    base.add_panel_label(fig, enrich_axes[0], "A", dx=-0.155)
    base.add_panel_label(fig, scatter_ax, "B", dx=-0.050)
    base.add_panel_label(fig, ld_ax, "C", dx=-0.050)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / f"{args.out_stem}.pdf"
    png = args.out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=360)
    plt.close(fig)

    fit_components = (
        component_means.loc[~component_means["is_maf"]]
        if args.fit_non_maf
        else component_means
    )
    slope, r2 = no_intercept_stats(
        fit_components["eur_mean"].to_numpy(dtype=float),
        fit_components["sas_mean"].to_numpy(dtype=float),
    )
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(
        "Panel C inputs: "
        f"{len(component_means)} BaselineLD components on 291,793 common SNPs; "
        f"{int(component_means['is_maf'].sum())} MAF bins; "
        f"fit components={len(fit_components)}; "
        f"no-intercept slope={slope:.12g}; R2={r2:.12g}."
    )


if __name__ == "__main__":
    main()
