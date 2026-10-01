"""Render the retained Bonferroni or SUMMIT FDR overlap figure."""
import argparse
from pathlib import Path
import sys
import venn_maintext as venn

parser = argparse.ArgumentParser()
parser.add_argument("--figure", choices=["S37", "S38"], required=True)
parser.add_argument("--outfile", type=Path, required=True)
args = parser.parse_args()
save = venn.save_figure_outputs


def save_selected(**kwargs):
    if args.figure == "S38" and "__covldsc__" in kwargs["stem"]:
        venn.plt.close(kwargs["fig"])
        return
    kwargs["outdir"] = args.outfile.parent
    kwargs["stem"] = args.outfile.stem
    save(**kwargs)


venn.save_figure_outputs = save_selected
sys.argv = [
    sys.argv[0],
    "--csv",
    "data/real_rg/estimates.csv",
    "--outdir",
    str(args.outfile.parent),
    "--threshold",
    "bonferroni" if args.figure == "S37" else "bh",
]
if args.figure == "S38":
    sys.argv.append("--supp")
venn.main()
