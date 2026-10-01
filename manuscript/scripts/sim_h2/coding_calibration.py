"""Plot coding calibration for the SUMMIT manuscript."""
from __future__ import annotations

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POPS = ["EUR", "SAS", "AFR"]


BOOTSTRAP_CI = 0.95


METHODS_KEEP = [
    "rhe",
    "sumrhe",
    "covsumrhe",
    "ldsc_20000",
    "covldsc",
    "sumher_20000",
    "sumher_ldak_20000",
]


PRETTY_LABEL_OVERRIDES = {
    "sumrhe": "SUMMIT\n(no PC)",
    "covsumrhe": "SUMMIT",
    "ldsc_20000": "LDSC",
    "sumher_20000": "SumHer\n(GCTA)",
    "sumher_ldak_20000": "SumHer\n(LDAK)",
}


def draw_allmethod_figure(
    cal, perpop: dict[str, pd.DataFrame], global_methods: set[str]
) -> plt.Figure:
    sns.set_style("whitegrid")
    methods_ord = [method for method in METHODS_KEEP if method in global_methods]
    pretty_label_map = dict(cal.PRETTY_LABEL)
    pretty_label_map.update(PRETTY_LABEL_OVERRIDES)

    fig_w = 11.8
    fig_h = 4.8
    fig, axes = plt.subplots(1, len(POPS), figsize=(fig_w, fig_h), sharey=True)
    axes = np.atleast_1d(axes)

    for j, pop in enumerate(POPS):
        ax = axes[j]
        cal._plot_lines(
            ax,
            perpop[pop],
            methods_ord,
            title=pop,
            xlabel="Nominal threshold α",
            ylabel="Empirical rejection rate",
            highlight_methods=("covsumrhe",),
            pretty_label_map=pretty_label_map,
            show_ribbon=True,
            ribbon_kind="se",
            ribbon_alpha=0.13,
            ci_level=BOOTSTRAP_CI,
        )
        if j > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    cal._add_shared_legend(fig, axes)
    return fig


if __name__ == "__main__":
    from pathlib import Path
    import calibration_style as cal

    source = pd.read_csv("data/sim_h2/coding_calibration.tsv", sep="\t")
    fig = draw_allmethod_figure(
        cal,
        {pop: source[source["pop"] == pop].copy() for pop in POPS},
        set(source["method_label"]),
    )
    fig.savefig(
        "figs/supplementary/fig_s15_coding_calibration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
