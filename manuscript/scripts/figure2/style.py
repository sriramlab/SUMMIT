"""Plot style for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib as mpl


FIGURE2_VISUAL_SCALE = {
    "text": 1.12,
    "marker": 1.20,
    "line": 1.20,
}


def set_figure2_style(*, common, prior) -> None:
    """Apply one intermediate visual scale to all imported plotting modules."""
    common.set_style(profile="standard")
    prior.common.set_style(profile="standard")
    modules = (common, common.sim, prior.common, prior.common.sim)
    for module in modules:
        module._TEXT_SCALE = FIGURE2_VISUAL_SCALE["text"]
        module._MARKER_SCALE = FIGURE2_VISUAL_SCALE["marker"]
        module._LINE_SCALE = FIGURE2_VISUAL_SCALE["line"]

    mpl.rcParams.update(
        {
            "font.size": common._fs(12.0),
            "axes.titlesize": common._fs(14.0),
            "axes.labelsize": common._fs(13.2),
            "axes.linewidth": common._lw(1.15),
            "axes.titlepad": common._ms(6.0),
            "xtick.labelsize": common._fs(12.0),
            "ytick.labelsize": common._fs(12.0),
            "xtick.major.width": common._lw(1.05),
            "ytick.major.width": common._lw(1.05),
            "xtick.major.size": common._ms(4.0),
            "ytick.major.size": common._ms(4.0),
            "grid.linewidth": common._lw(0.8),
            "legend.fontsize": common._fs(11.8),
        }
    )
