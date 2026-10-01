"""Render the ROC and precision–recall curves for Figure S14."""
import coding_curves_balanced as source

assert source.WINDOWS_KEEP_BEST == {20000}
source.TITLE_FONTSIZE = 16
source.AXIS_LABEL_FONTSIZE = 14
source.TICK_LABEL_FONTSIZE = 10
_original_subplots = source.plt.subplots


def _subplots(*args, **kwargs):
    kwargs["figsize"] = (14.5, 7.5)
    return _original_subplots(*args, **kwargs)


source.plt.subplots = _subplots
# Axes and legend positions.
_original_layout = source.plt.Figure.tight_layout


def _layout(self, *args, **kwargs):
    rect = list(kwargs.get("rect", (0, 0, 0.88, 1)))
    rect[2] += 28.8 / (14.5 * 72)
    kwargs["rect"] = rect
    return _original_layout(self, *args, **kwargs)


source.plt.Figure.tight_layout = _layout
_original_legend = source.plt.Figure.legend


def _legend(self, *args, **kwargs):
    kwargs["bbox_to_anchor"] = (0.89 - 1.8 / (14.5 * 72), 0.5)
    return _original_legend(self, *args, **kwargs)


source.plt.Figure.legend = _legend
# Read the curve summaries.
import json
import numpy as np
from pathlib import Path


def _saved_curves(*args, **kwargs):
    groups = json.loads(Path("data/sim_h2/coding_curves.json").read_text())
    return tuple(
        {
            (item["pop"], item["method"]): {
                k: np.asarray(v) if isinstance(v, list) else v
                for k, v in item["values"].items()
            }
            for item in group
        }
        for group in groups
    )


source.build_balanced_curves = _saved_curves
source.main()
