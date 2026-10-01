"""Rebuild Supplementary Figure S44 from aggregate reference-swap and LD tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
import reference_swap
import ratio_panel


def main():
    reference_swap.set_style()
    _, non_maf, maf = reference_swap.load_reference_swap()
    data = json.loads(Path("data/external/ld_hexbin.json").read_text())
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13.2, 5.1), gridspec_kw={"width_ratios": [1.13, 1.0]}
    )
    ratio_panel.draw_supplementary_ratio(left, non_maf, maf)
    ratio_panel.format_diagnostic_axis(left)
    hb = PolyCollection(
        data["hexagons"],
        array=np.asarray(data["counts"]),
        norm=LogNorm(),
        cmap="viridis",
        edgecolors="face",
        zorder=1,
    )
    right.add_collection(hb)
    right.set_xscale("log")
    right.set_yscale("log")
    bins = pd.DataFrame(data["quantiles"])
    right.errorbar(
        bins.x_median,
        bins.y_median,
        xerr=np.vstack([bins.x_median - bins.x_q25, bins.x_q75 - bins.x_median]),
        yerr=np.vstack([bins.y_median - bins.y_q25, bins.y_q75 - bins.y_median]),
        fmt="o-",
        color="black",
        ecolor="black",
        elinewidth=0.65,
        lw=1,
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.7,
        alpha=0.88,
        zorder=3,
        label="quantile-bin median (IQR)",
    )
    lo, hi = data["limits"]
    slope = data["slope"]
    right.plot([lo, hi], [lo, hi], color="0.35", lw=1, ls="--", zorder=1)
    right.plot([lo, hi], [lo * slope, hi * slope], color="#b2182b", lw=1.1, zorder=1)
    right.set_xlim(lo, hi)
    right.set_ylim(lo, hi)
    right.set_title(
        f"Single-component LD score\npositive SNPs={data['n']:,}, $R^2$={data['r2']:.3f}, slope={slope:.3f}",
        fontsize=11,
    )
    right.set_xlabel("EUR (300k) LD score")
    right.set_ylabel("SAS LD score")
    right.grid(True, color="0.9", lw=0.55, zorder=0)
    right.set_axisbelow(True)
    legend = right.legend(frameon=True, loc="upper left", fontsize=8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.3")
    legend.get_frame().set_alpha(1)
    cb = fig.colorbar(hb, ax=right, fraction=0.046, pad=0.02)
    cb.set_label("SNP count per hexbin")
    for ax, letter in [(left, "A"), (right, "B")]:
        ax.text(
            -0.16,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )
    fig.tight_layout(w_pad=2.4)
    out = Path("figs/supplementary/fig_s44_external_reference_mismatch.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
