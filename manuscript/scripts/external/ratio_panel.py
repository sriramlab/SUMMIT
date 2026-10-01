"""Plot ratio panel for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import diagnostic


def draw_supplementary_ratio(
    ax: plt.Axes,
    non_maf: pd.DataFrame,
    maf: pd.DataFrame,
) -> None:
    """Reproduce panel B of the supplementary diagnostic figure."""
    rng = np.random.default_rng(11)
    quantities = [
        ("$h^2_{total}$", non_maf["h2_ratio_sas_over_eur"].to_numpy(dtype=float)),
        ("$h^2_{annot}$", non_maf["h2bin_ratio_sas_over_eur"].to_numpy(dtype=float)),
        ("enrich.", non_maf["enrichment_ratio_sas_over_eur"].to_numpy(dtype=float)),
    ]
    for i, (_, values) in enumerate(quantities, start=1):
        values = values[np.isfinite(values) & (values > 0)]
        jitter = rng.uniform(-0.20, 0.20, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=11,
            color=diagnostic.NON_MAF_COLOR,
            alpha=0.38,
            linewidths=0,
            zorder=2,
        )
        diagnostic.median_bar(ax, i, values, width=0.26)

    maf_bins = sorted(maf["maf_bin"].unique())
    offset = len(quantities)
    for i, maf_bin in enumerate(maf_bins, start=offset + 1):
        values = maf.loc[
            maf["maf_bin"] == maf_bin,
            "enrichment_ratio_sas_over_eur",
        ].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        jitter = rng.uniform(-0.19, 0.19, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=18,
            marker="^",
            color=diagnostic.MAF_COLOR,
            alpha=0.58,
            linewidths=0,
            zorder=2,
        )
        diagnostic.median_bar(ax, i, values, width=0.25)

    ax.axhline(1.0, color="#333333", ls="--", lw=1.2, zorder=1)
    ax.axvline(offset + 0.5, color="#c7cbd1", lw=1.0, zorder=1)
    ax.set_xticks(range(1, offset + len(maf_bins) + 1))
    ax.set_xticklabels(
        [label for label, _ in quantities] + [f"MAF{maf_bin}" for maf_bin in maf_bins],
        rotation=45,
        ha="right",
    )
    ax.set_xlim(0.45, offset + len(maf_bins) + 0.55)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.23, 600)
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16, 64, 256, 512])
    ax.set_yticklabels(["1/4", "1/2", "1", "2", "4", "8", "16", "64", "256", "512"])
    ax.set_ylabel("SAS LD / EUR LD estimate (log2 scale)")
    ax.text(0.155, 1.035, "Non-MAF", transform=ax.transAxes, ha="center", va="bottom")
    ax.text(
        0.67,
        1.035,
        "MAF-bin enrichment",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )
    ax.grid(True, color="#e1e5ea", linewidth=0.8)


def format_diagnostic_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.1)
