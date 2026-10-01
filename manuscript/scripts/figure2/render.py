from pathlib import Path
import pandas as pd
import layout
import common
import coding
import style

style.set_figure2_style(common=common, prior=coding)
root = Path("data/sim_h2/figure2")
layout.plot_power_and_calibration_curves(
    *[
        pd.read_csv(root / (x + ".tsv"), sep="\t", float_precision="round_trip")
        for x in ["discrimination", "calibration", "overlap_curves"]
    ],
    Path("figs/main/fig02_enrichment_benchmarks"),
    common=common,
    panel_source=coding,
)
