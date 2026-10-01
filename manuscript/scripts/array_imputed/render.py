"""Render the two array/imputed comparisons from their saved aggregate tables."""
import argparse
from pathlib import Path
import pandas as pd
import renderers

parser = argparse.ArgumentParser()
parser.add_argument("--figure", choices=["S31", "S32"], required=True)
parser.add_argument("--outfile", type=Path, required=True)
args = parser.parse_args()
root = Path("data/array_imputed")
meta = pd.read_csv(root / "annotation_comparison.csv")
renderers.setup_style()
save = renderers.savefig
renderers.OUT_DIR = args.outfile.parent
renderers.savefig = lambda fig, stem: save(fig, args.outfile.stem)
if args.figure == "S31":
    total = pd.read_csv(root / "total_heritability.csv")
    maf = pd.read_csv(root / "annotation_variant_counts.csv")
    counts = pd.read_csv(root / "variant_count_summary.csv")
    renderers.figure_summary_2x2_maf_aligned_rare_left(total, meta, maf, counts)
else:
    renderers.figure_tau_star_enrichment_rank_aligned(meta)
