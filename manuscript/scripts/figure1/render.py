from pathlib import Path
import pandas as pd
import common
import layout
import mafld
import ld_panel

common._case_ylim = ld_panel.panel_ld_limits
common.draw_ld_vertical_stack = ld_panel.draw_ld_panel
mafld.ARCHITECTURE_TITLES = {"GCTA": "MAF-LD: GCTA", "LDAK": "MAF-LD: LDAK"}
root = Path("data/sim_h2/figure1")
c = layout.FigureLayout(
    out_stem="fig01_ld_and_h2_benchmarks",
    panel_c_title="MAF-LD relative MSE",
    panel_d_title="MAF-LD absolute relative bias",
    draw_panel_c=mafld.draw_panel_c_median(
        pd.read_csv(root / "panel_c_relmse_summary.tsv", sep="\t")
    ),
    draw_panel_d=mafld.draw_panel_d_histogram(
        pd.read_csv(root / "panel_d_bias_source.tsv", sep="\t")
    ),
)
layout.render_figure(c)
