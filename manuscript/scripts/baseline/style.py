"""Plot style for the SUMMIT manuscript."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Rectangle


TAB20 = [plt.matplotlib.colors.to_hex(c) for c in colormaps["tab20"].colors]


POP_ORDER = ["EUR_300k", "EUR", "SAS", "AFR"]


POP_LABEL = {
    "EUR_300k": "EUR (300k)",
    "EUR": "EUR",
    "SAS": "SAS",
    "AFR": "AFR",
}


POP_COLOR = {
    "EUR": TAB20[0],
    "EUR_300k": TAB20[1],
    "SAS": TAB20[4],
    "AFR": TAB20[6],
}


METHOD_LABEL = {
    "sumrhe": "SUMMIT",
    "sumher": "SumHer-GCTA",
    "sumher_ldak": "SumHer-LDAK",
    "ldsc": "LDSC",
    "covldsc": "cov-LDSC",
}


METHOD_COLOR = {
    "sumrhe": TAB20[4],
    "sumher": TAB20[6],
    "sumher_ldak": TAB20[8],
    "ldsc": TAB20[10],
    "covldsc": TAB20[2],
}


METHOD_ORDER = ["sumrhe", "covldsc", "ldsc", "sumher", "sumher_ldak"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "legend.title_fontsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def pop_rank(pop: str) -> int:
    try:
        return POP_ORDER.index(str(pop))
    except ValueError:
        return 999


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    ra, rb = pop_rank(a), pop_rank(b)
    if ra < rb:
        return a, b
    if rb < ra:
        return b, a
    return (a, b) if str(a) <= str(b) else (b, a)


def pair_label(a: str, b: str, pretty: bool = False) -> str:
    lo, hi = canonical_pair(a, b)
    if pretty:
        return f"{POP_LABEL.get(lo, lo)}-{POP_LABEL.get(hi, hi)}"
    return f"{lo}-{hi}"


def method_display_order(methods_present: Iterable[str]) -> List[str]:
    present = list(dict.fromkeys([str(m) for m in methods_present]))
    out = [m for m in METHOD_ORDER if m in present]
    out.extend([m for m in present if m not in out])
    return out


def add_bar_contour(ax, bars, color="black", lw=1.1, zorder=4.2):
    for patch in bars.patches:
        h = patch.get_height()
        if not np.isfinite(h) or h == 0:
            continue

        x = patch.get_x()
        w = patch.get_width()
        x0 = x
        x1 = x + w

        # bar baseline and outer end
        y_base = patch.get_y()  # usually 0 for ax.bar(...)
        y_top = y_base + h

        # draw only left, right, and top-at-outer-end
        ax.plot(
            [x0, x0],
            [y_base, y_top],
            color=color,
            lw=lw,
            zorder=zorder,
            solid_capstyle="butt",
        )
        ax.plot(
            [x1, x1],
            [y_base, y_top],
            color=color,
            lw=lw,
            zorder=zorder,
            solid_capstyle="butt",
        )
        ax.plot(
            [x0, x1],
            [y_top, y_top],
            color=color,
            lw=lw,
            zorder=zorder,
            solid_capstyle="butt",
        )


def all_pair_order() -> List[str]:
    pairs = []
    for i in range(len(POP_ORDER)):
        for j in range(i + 1, len(POP_ORDER)):
            pairs.append(f"{POP_ORDER[i]}-{POP_ORDER[j]}")
    return pairs


def panel_a_metric_spec(estimand: str, metric: str) -> Tuple[str, str, str, str]:
    estimand_title = {
        "h2": r"Partitioned $h^2$",
        "tau_star": r"Partitioned $\tau^{*}$",
    }.get(estimand, estimand)
    if metric == "pearson":
        return (
            "pearson_r_agg",
            "pearson_r_se",
            estimand_title + r": IVW (Fisher-z) Pearson $r$ across phenotypes",
            r"Aggregated Pearson $r$",
        )
    if metric == "spearman":
        return (
            "spearman_rho_agg",
            "spearman_rho_se",
            estimand_title + r": IVW (Fisher-z) Spearman $\rho$ across phenotypes",
            r"Aggregated Spearman $\rho$",
        )
    raise ValueError(f"Unsupported panel-A correlation metric: {metric}")
