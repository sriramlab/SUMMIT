"""Plot sim style for the SUMMIT manuscript."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl


METHODS = ["summit", "covldsc", "sumher_ldak"]


METHOD_LABEL = {
    "summit": "SUMMIT",
    "covldsc": "cov-LDSC",
    "sumher_ldak": "SumHer-LDAK",
}


METHOD_COLOR = {
    "summit": mpl.colormaps["tab20"].colors[4],
    "covldsc": mpl.colormaps["tab20"].colors[2],
    "sumher_ldak": mpl.colormaps["tab20"].colors[8],
}


METHOD_MARKER = {
    "summit": "o",
    "covldsc": "s",
    "sumher_ldak": "D",
}


VISUAL_PROFILES = {
    "standard": {"text": 1.0, "marker": 1.0, "line": 1.0},
    "readable": {"text": 1.25, "marker": 1.45, "line": 1.45},
}


_TEXT_SCALE = 1.0


_MARKER_SCALE = 1.0


_LINE_SCALE = 1.0


def set_visual_profile(profile: str = "standard") -> None:
    if profile not in VISUAL_PROFILES:
        raise ValueError(
            f"Unknown visual profile {profile!r}; expected one of {sorted(VISUAL_PROFILES)}"
        )
    global _TEXT_SCALE, _MARKER_SCALE, _LINE_SCALE
    scales = VISUAL_PROFILES[profile]
    _TEXT_SCALE = scales["text"]
    _MARKER_SCALE = scales["marker"]
    _LINE_SCALE = scales["line"]


def _fs(size: float) -> float:
    return size * _TEXT_SCALE


def _lw(width: float) -> float:
    return width * _LINE_SCALE


def _ms(size: float) -> float:
    return size * _MARKER_SCALE


def set_style(profile: str = "standard"):
    set_visual_profile(profile)
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": _fs(12.0),
            "axes.titlesize": _fs(14.0),
            "axes.labelsize": _fs(13.2),
            "axes.linewidth": _lw(1.15),
            "xtick.labelsize": _fs(12.0),
            "ytick.labelsize": _fs(12.0),
            "xtick.major.width": _lw(1.05),
            "ytick.major.width": _lw(1.05),
            "xtick.major.size": _ms(4.0),
            "ytick.major.size": _ms(4.0),
            "grid.color": "#D9D9D9",
            "grid.linewidth": _lw(0.8),
            "grid.alpha": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": _fs(11.8),
        }
    )


def panel_a_limits(panel_a: pd.DataFrame) -> tuple[float, float]:
    low_vals = panel_a[["q25", "median"]].to_numpy(dtype=float).ravel()
    low_vals = low_vals[np.isfinite(low_vals) & (low_vals > 0)]
    high_vals = panel_a[["q75", "median"]].to_numpy(dtype=float).ravel()
    high_vals = high_vals[np.isfinite(high_vals) & (high_vals > 0)]

    ymin = min(0.8, float(np.nanmin(low_vals)) * 0.95)
    ymax = max(7000.0, float(np.nanmax(high_vals)) * 1.12)
    return ymin, ymax


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5A5A5A")
    ax.spines["bottom"].set_color("#5A5A5A")
    ax.tick_params(colors="#333333")
