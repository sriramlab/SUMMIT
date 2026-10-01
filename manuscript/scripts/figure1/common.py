"""Plot common for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from dataclasses import dataclass
from pathlib import Path
import matplotlib as mpl
import sim_style as sim

LD_DEFICIT_TSV = Path("data/sim_h2/figure1/panel_a_source.tsv")

PACKAGE_ROOT = Path(".")


def load_sim_panels(*args, **kwargs):
    return SimPanels(
        pd.read_csv("data/sim_h2/figure1/panel_b_source.tsv", sep="\t"),
        pd.DataFrame(),
        pd.DataFrame(),
    )


POPS = ["EUR", "SAS", "AFR"]


LD_ESTIMATORS = ["2Mb", "20Mb", "CHR"]


LD_ESTIMATOR_LABEL = {"2Mb": "2Mb", "20Mb": "20Mb", "CHR": "CHR"}


LD_ESTIMATOR_COLOR = {
    "2Mb": "#8172B2",
    "20Mb": "#C44E52",
    "CHR": "#4C72B0",
}


LD_ESTIMATOR_MARKER = {"2Mb": "o", "20Mb": "s", "CHR": "^"}


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


@dataclass
class SimPanels:
    panel_a: pd.DataFrame
    panel_b: pd.DataFrame
    panel_c: pd.DataFrame


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


def load_ld_deficit(path: Path = LD_DEFICIT_TSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing LD deficit summary: {path}")
    df = pd.read_csv(path, sep="\t")
    df = df[df["pop"].isin(POPS) & df["group"].isin(LD_ESTIMATORS)].copy()
    df["pop"] = pd.Categorical(df["pop"], categories=POPS, ordered=True)
    df["group"] = pd.Categorical(df["group"], categories=LD_ESTIMATORS, ordered=True)
    df["case_label"] = pd.Categorical(
        df["case_label"],
        categories=["No covariates", "PC-adjusted"],
        ordered=True,
    )
    return df.sort_values(["case_label", "pop", "group"]).reset_index(drop=True)


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


def ld_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=LD_ESTIMATOR_COLOR[estimator],
            marker=LD_ESTIMATOR_MARKER[estimator],
            linestyle="none",
            markersize=_ms(8),
            markerfacecolor=LD_ESTIMATOR_COLOR[estimator],
            markeredgecolor="#303030",
            markeredgewidth=_lw(0.9),
            label=LD_ESTIMATOR_LABEL[estimator],
        )
        for estimator in LD_ESTIMATORS
    ]


def _case_ylim(sub: pd.DataFrame, case_label: str) -> tuple[float, float]:
    vals = sub["deficit_pct"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 10.0)
    ymax = float(np.nanmax(vals))
    if case_label == "No covariates":
        return (-3.0, max(105.0, np.ceil((ymax + 3.0) / 10.0) * 10.0))
    return (-0.8, max(15.0, np.ceil((ymax + 1.2) / 2.0) * 2.0))


def draw_ld_vertical_stack(
    fig: plt.Figure,
    spec,
    ld_df: pd.DataFrame,
    connect_within_pop: bool = True,
):
    inner = spec.subgridspec(2, 1, hspace=0.28)
    axes = [fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0])]
    x_lookup = {pop: i for i, pop in enumerate(POPS)}
    offsets = {"2Mb": -0.18, "20Mb": 0.0, "CHR": 0.18}

    for ax, case_label in zip(axes, ["No covariates", "PC-adjusted"]):
        sub_case = ld_df[ld_df["case_label"].astype(str) == case_label].copy()
        if connect_within_pop:
            for pop in POPS:
                sub_pop = sub_case[sub_case["pop"].astype(str) == pop]
                points = []
                for estimator in LD_ESTIMATORS:
                    row = sub_pop[sub_pop["group"].astype(str) == estimator]
                    if row.empty:
                        continue
                    points.append(
                        (
                            x_lookup[pop] + offsets[estimator],
                            float(row["deficit_pct"].iloc[0]),
                        )
                    )
                if len(points) > 1:
                    xs, ys = zip(*points)
                    ax.plot(
                        xs, ys, color="#9A9A9A", linewidth=_lw(1.4), alpha=0.9, zorder=1
                    )

        for estimator in LD_ESTIMATORS:
            sub = sub_case[sub_case["group"].astype(str) == estimator]
            if sub.empty:
                continue
            xs = [
                x_lookup[str(pop)] + offsets[estimator]
                for pop in sub["pop"].astype(str)
            ]
            ys = sub["deficit_pct"].to_numpy(dtype=float)
            if not connect_within_pop:
                ax.plot(
                    xs,
                    ys,
                    color=LD_ESTIMATOR_COLOR[estimator],
                    linewidth=_lw(1.25),
                    alpha=0.35,
                    zorder=2,
                )
            ax.scatter(
                xs,
                ys,
                s=_scatter_size(64),
                marker=LD_ESTIMATOR_MARKER[estimator],
                color=LD_ESTIMATOR_COLOR[estimator],
                edgecolor="#303030",
                linewidth=_lw(0.9),
                zorder=3,
                label=LD_ESTIMATOR_LABEL[estimator],
            )

        ax.axhline(
            0.0, color="#777777", linewidth=_lw(1.1), linestyle=(0, (4, 2)), zorder=0
        )
        ax.set_title(
            "No PC adjustment" if case_label == "No covariates" else "PC-adjusted"
        )
        ax.set_ylim(*_case_ylim(sub_case, case_label))
        ax.set_xlim(-0.5, len(POPS) - 0.5)
        ax.set_xticks(range(len(POPS)))
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        style_axis(ax)

    for ax in axes:
        ax.set_xticklabels(POPS)
    axes[0].set_ylabel("LD missed (%)")
    axes[1].set_ylabel("LD missed (%)")
    return axes


def save_figure(fig: plt.Figure, outprefix: Path) -> None:
    outprefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outprefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outprefix.with_suffix(".png"), bbox_inches="tight")
