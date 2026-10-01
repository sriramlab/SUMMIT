from pathlib import Path
import argparse
import pandas as pd
import panels

p = argparse.ArgumentParser()
p.add_argument("--figure", choices=["5", "S28"], required=True)
p.add_argument("--outfile", required=True)
a = p.parse_args()
out = Path(a.outfile)
args = argparse.Namespace(
    figure_kind="main_ab" if a.figure == "5" else "supp_acd",
    panel_a_estimands="h2",
    panel_a_pairs="reference",
    reference_pop="EUR_300k",
    panel_a_corr_metric="pearson",
    width=None,
    height=None,
    dpi=450,
    enrichment_scale="log2",
)
panels.set_style()
folder = Path("data/baseline/figure5" if a.figure == "5" else "data/baseline/s28")
panel_a = pd.read_csv(folder / "panelA_pairwise_portability.csv")
if a.figure == "5":
    panels.build_main_ab_figure(
        panel_a,
        {"ratio_meta": pd.read_csv(folder / "ratio_meta.csv")},
        args,
        str(out.with_suffix(".pdf")),
        str(out),
    )
else:
    data = {
        key: pd.read_csv(folder / (key + ".csv"))
        for key in ["trait_detail", "annotation_pairs", "annotation_summary"]
    }
    panels.build_supp_acd_figure(
        panel_a, data, args, str(out.with_suffix(".pdf")), str(out)
    )
