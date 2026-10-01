#!/usr/bin/env python3
"""Plot Figure 7: genetic-correlation agreement and significant-pair overlap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve().parent
SCRIPT_DIR = HERE
sys.path.insert(0, str(SCRIPT_DIR))

import layout as figure8  # noqa: E402
import venn_maintext as venn  # noqa: E402


INPUT = Path("data/real_rg/estimates.csv")
OUTBASE = Path("figs/main/fig07_genetic_correlation_agreement")


def main() -> None:
    frame = pd.read_csv(INPUT)
    args = argparse.Namespace(
        threshold="bh",
        alpha=0.05,
        bh_q=0.05,
        family_size=780,
        circle_size_mode="fixed",
        proportional_min_radius=0.18,
        dpi=450,
    )

    hits = figure8.load_panel_hits(
        frame,
        pops=figure8.DEFAULT_POPS,
        annot=figure8.DEFAULT_ANNOT,
        threshold=args.threshold,
        alpha=args.alpha,
        bh_q=args.bh_q,
        family_size=args.family_size,
    )

    original_draw_panel = figure8.scatter.draw_panel

    def draw_panel_without_afr_r(*draw_args, **draw_kwargs):
        result = original_draw_panel(*draw_args, **draw_kwargs)
        if draw_kwargs.get("pop") == "AFR":
            ax = draw_kwargs.get("ax", draw_args[0])
            for label in ax.texts:
                if label.get_text().startswith("r="):
                    label.set_visible(False)
        return result

    figure8.scatter.draw_panel = draw_panel_without_afr_r
    venn.set_style()
    figure8.scatter.set_style()
    figure8.scatter.LABEL_FONT_SIZE = figure8.SCATTER_PAIR_LABEL_SIZE
    figure8.build_method_venns_top_scatter_bottom_version(
        df=frame,
        hits_by_group=hits,
        args=args,
        outbase=OUTBASE,
    )

    stem = OUTBASE.with_name(OUTBASE.name)
    print(f"Wrote {stem}.png")
    print(f"Wrote {stem}.pdf")


if __name__ == "__main__":
    main()
