"""Plot architecture for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import method_style as v2

POPS = v2.POPS


SIGNED_METHODS = v2.SIGNED_METHODS


SIGNED_LABEL = v2.SIGNED_LABEL


TRUE_TOTAL_H2_TRAIT1 = 0.25


TRUE_TOTAL_H2_TRAIT2 = 0.60


TRUE_TOTAL_RG = 0.30


TRUE_TOTAL_GENCOV = TRUE_TOTAL_RG * np.sqrt(TRUE_TOTAL_H2_TRAIT1 * TRUE_TOTAL_H2_TRAIT2)


SIGNED_PROFILES = ["common_positive", "low_positive"]


BAND_COLOR = "#F4F4F7"


def shade_population_bands(ax: plt.Axes, centers, half_width: float) -> None:
    """Alternate a pale background per population instead of divider rules.

    Vertical divider lines are indistinguishable from a confidence interval
    that runs the full height of the axis, which happens for SumHer-LDAK.
    """
    for idx in range(1, len(centers), 2):
        ax.axvspan(
            centers[idx] - half_width,
            centers[idx] + half_width,
            color=BAND_COLOR,
            linewidth=0.0,
            zorder=0,
        )
