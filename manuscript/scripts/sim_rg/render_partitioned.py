from pathlib import Path
import argparse
import sys
import partitioned

p = argparse.ArgumentParser()
p.add_argument("--figure", choices=["S16", "S17"], required=True)
p.add_argument("--outfile", required=True)
a = p.parse_args()
original = partitioned.save_figure


# The original program loops over simulation architectures; save only flatH2/GCTA.
def save_selected(fig, outpath, *args, **kwargs):
    if str(outpath).endswith("_flatH2_GCTA.pdf"):
        return original(fig, Path(a.outfile), *args, **kwargs)


partitioned.save_figure = save_selected
sys.argv = [
    sys.argv[0],
    "--base",
    "data/sim_rg/partitioned",
    "--true-params",
    "data/sim_rg/partitioned/truth.csv",
    "--outdir",
    "figs/supplementary",
    "--plot-error",
    "--y-scale",
    "independent",
]
if a.figure == "S17":
    sys.argv.append("--plot-all-bins")
partitioned.main()
