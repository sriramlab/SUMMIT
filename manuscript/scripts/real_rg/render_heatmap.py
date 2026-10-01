"""Render a manuscript heatmap with an explicit output path."""
import argparse
from pathlib import Path
import sys
import heatmap

parser = argparse.ArgumentParser()
parser.add_argument("--outfile", required=True)
args, remaining = parser.parse_known_args()
heatmap.make_panel_outpath = lambda *a, **k: Path(args.outfile)
heatmap.make_single_outpath = lambda *a, **k: Path(args.outfile)
sys.argv = [sys.argv[0], *remaining]
heatmap.main()
