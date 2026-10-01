"""Plot diagnostic for the SUMMIT manuscript."""
from __future__ import annotations

from pathlib import Path
import numpy as np


INPUT = Path("data/external/reference_swap.tsv")

DROP_SOURCES = {"MAGIC2021", "MVP_R4_HARE"}


NON_MAF_COLOR = "#9aa1aa"


MAF_COLOR = "#b2182b"


CONTOUR_COLOR = "#2f7fab"


def is_maf_annotation(annotation: str) -> bool:
    return annotation.startswith("MAFbin")


def source_from_trait(trait: object) -> str:
    return str(trait).split(".", 1)[0]


def add_density_contours(ax, x, y, color="#2f7fab"):
    """Draw simple smoothed-looking contours from a 2D histogram."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 20:
        return

    x_pad = 0.05 * max(np.ptp(x), 1.0)
    y_pad = 0.05 * max(np.ptp(y), 1.0)
    x_edges = np.linspace(x.min() - x_pad, x.max() + x_pad, 56)
    y_edges = np.linspace(y.min() - y_pad, y.max() + y_pad, 56)
    hist, xe, ye = np.histogram2d(x, y, bins=(x_edges, y_edges))

    try:
        from scipy.ndimage import gaussian_filter

        hist = gaussian_filter(hist, sigma=1.2)
    except Exception:
        pass

    z = hist.T
    positive = z[z > 0]
    if positive.size == 0:
        return

    levels = np.quantile(positive, [0.55, 0.70, 0.82, 0.91])
    levels = np.unique(levels)
    if levels.size == 0:
        return

    x_centers = (xe[:-1] + xe[1:]) / 2
    y_centers = (ye[:-1] + ye[1:]) / 2
    ax.contour(x_centers, y_centers, z, levels=levels, colors=color, linewidths=1.35)


def median_bar(ax, x, y, width=0.28, color="black", lw=2.4):
    med = np.nanmedian(y)
    ax.plot(
        [x - width, x + width],
        [med, med],
        color=color,
        lw=lw,
        solid_capstyle="butt",
        zorder=4,
    )
    return med
