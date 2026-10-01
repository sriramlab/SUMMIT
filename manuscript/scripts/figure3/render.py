from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import layout

root = Path("data/sim_rg/figure3")
fig = layout.build_figure(
    "rg",
    *[
        pd.read_csv(root / (x + ".tsv"), sep="\t")
        for x in ["calibration", "panelB", "panelC", "normalization"]
    ],
)
fig.savefig(
    "figs/main/fig03_genetic_correlation_benchmarks.png", dpi=240, facecolor="white"
)
plt.close(fig)
