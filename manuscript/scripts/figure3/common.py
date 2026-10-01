"""Plot common for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib as mpl
import sim_style as sim

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


def _scatter_size(area: float) -> float:
    return area * (_MARKER_SCALE**2)


def set_style(profile: str = "standard") -> None:
    set_visual_profile(profile)
    sim.set_style(profile=profile)
    mpl.rcParams.update(
        {
            "savefig.dpi": 300,
            "figure.dpi": 140,
            "font.size": _fs(12.0),
            "axes.titlesize": _fs(14.0),
            "axes.labelsize": _fs(13.2),
            "axes.linewidth": _lw(1.15),
            "axes.titlepad": _ms(6.0),
            "xtick.labelsize": _fs(12.0),
            "ytick.labelsize": _fs(12.0),
            "xtick.major.width": _lw(1.05),
            "ytick.major.width": _lw(1.05),
            "xtick.major.size": _ms(4.0),
            "ytick.major.size": _ms(4.0),
            "grid.linewidth": _lw(0.8),
            "legend.fontsize": _fs(11.8),
        }
    )


def style_axis(ax) -> None:
    sim.style_axis(ax)


def add_panel_label(
    fig: plt.Figure, axes, label: str, dx: float = -0.055, dy: float = 0.012
) -> None:
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    fig.text(
        x0 + dx,
        y1 + dy,
        label,
        fontsize=_fs(18),
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def method_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=sim.METHOD_COLOR[method],
            marker=sim.METHOD_MARKER[method],
            linestyle="-",
            linewidth=_lw(1.9),
            markersize=_ms(8),
            markerfacecolor=sim.METHOD_COLOR[method],
            markeredgecolor="white",
            markeredgewidth=_lw(0.9),
            label=sim.METHOD_LABEL[method],
        )
        for method in sim.METHODS
    ]
