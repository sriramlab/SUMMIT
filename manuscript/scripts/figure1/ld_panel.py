"""Plot ld panel for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import common
import ld_limits as ld_limits


def panel_ld_limits(source: pd.DataFrame, case_label: str) -> tuple[float, float]:
    """Set Panel A limits and check that all data remain visible."""
    if case_label == "PC-adjusted":
        values = pd.to_numeric(source["deficit_pct"], errors="raise").to_numpy(float)
        if values.min() < -0.8 or values.max() > 15.0:
            raise RuntimeError("PC-adjusted Panel A values fall outside (-0.8, 15)")
        return (-0.8, 15.0)
    return ld_limits.ld_axis_limits(source, case_label)


ORIGINAL_DRAW_LD_PANEL = common.draw_ld_vertical_stack


def draw_ld_panel(fig, spec, source, connect_within_pop=True):
    axes = ORIGINAL_DRAW_LD_PANEL(
        fig,
        spec,
        source,
        connect_within_pop=connect_within_pop,
    )
    for axis in axes:
        axis.set_ylabel("LD deficit (%)")
    # Keep the PC-adjusted panel on the manuscript scale.
    axes[1].set_ylim(-0.8, 15.0)
    axes[1].set_yticks(np.arange(0.0, 15.0, 2.0))
    return axes
