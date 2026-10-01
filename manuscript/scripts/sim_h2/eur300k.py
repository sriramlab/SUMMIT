"""Plot eur300k for the SUMMIT manuscript."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd


TAB20 = list(plt.get_cmap("tab20").colors)


METHODS_ALL = ["covsumrhe", "covldsc", "ldsc_50000"]


METHODS_NO_BASE = ["covldsc", "ldsc_50000"]


BASE_KEY = "covsumrhe"


PRETTY_LABEL = {
    "covsumrhe": "SUMMIT",
    "covldsc": "cov-LDSC",
    "ldsc_50000": "LDSC",
}


METHOD_COLOR = {
    "covsumrhe": TAB20[4],
    "covldsc": TAB20[2],
    "ldsc_50000": TAB20[10],
}


H2_VALS = [0.1, 0.25, 0.4]


TITLE_FONTSIZE = 20


AXIS_LABEL_FONTSIZE = 18


XTICK_LABEL_FONTSIZE = 12


YTICK_LABEL_FONTSIZE = 12


def gmean_pos(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(values))))


def compute_relmse_by_setting(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["sqerr"] = (work["estimate"] - work["h2"]) ** 2
    mse = work.groupby(["method_window", "h2", "pcausal"], as_index=False).agg(
        mse=("sqerr", "mean")
    )
    base = mse.loc[mse["method_window"] == BASE_KEY, ["h2", "pcausal", "mse"]].rename(
        columns={"mse": "mse_base"}
    )
    rel = mse.merge(base, on=["h2", "pcausal"], how="inner")
    rel["rel_mse"] = rel["mse"] / rel["mse_base"]
    return rel[["method_window", "h2", "pcausal", "rel_mse"]]


def summarize_relmse(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_window in METHODS_NO_BASE:
        values = rel.loc[rel["method_window"] == method_window, "rel_mse"].to_numpy(
            float
        )
        values = values[np.isfinite(values)]
        rows.append(
            {
                "method_window": method_window,
                "n": int(values.size),
                "gmean": gmean_pos(values),
                "median": float(np.nanmedian(values)),
                "q25": float(np.nanpercentile(values, 25)),
                "q75": float(np.nanpercentile(values, 75)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(rows)


def set_linear_ylim_with_pad(ax, values: np.ndarray) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.set_ylim(0.5, 2.0)
        return
    low = min(float(np.nanmin(values)), 1.0)
    high = max(float(np.nanmax(values)), 1.0)
    span = high - low
    pad = 0.08 * span if span > 0 else 0.15
    ax.set_ylim(low - pad, high + pad)


def robust_error_ylim_by_group(df: pd.DataFrame) -> tuple[float, float]:
    lows, highs = [], []
    for method_window in METHODS_ALL:
        values = df.loc[df["method_window"] == method_window, "err"].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q1 = float(np.nanpercentile(values, 25))
        q3 = float(np.nanpercentile(values, 75))
        iqr = q3 - q1
        if iqr <= 0 or not np.isfinite(iqr):
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
        else:
            fence_low = q1 - 1.5 * iqr
            fence_high = q3 + 1.5 * iqr
            low = float(np.nanmin(values[values >= fence_low]))
            high = float(np.nanmax(values[values <= fence_high]))
        lows.extend([low, q1, 0.0])
        highs.extend([high, q3, 0.0])

    if not lows or not highs:
        return (-1.0, 1.0)
    low = float(np.nanmin(lows))
    high = float(np.nanmax(highs))
    span = high - low
    pad = 0.22 * span if span > 0 else 0.25
    return low - pad, high + pad


def xtick_labelsize(n_methods: int) -> float:
    return XTICK_LABEL_FONTSIZE if n_methods < 6 else 9.5


def plot_relmse_panel(
    ax,
    rel_sub: pd.DataFrame,
    rel_sum_sub: pd.DataFrame,
    all_rel_values: np.ndarray,
    rng: np.random.Generator,
) -> None:
    x = np.arange(len(METHODS_ALL), dtype=float)

    for idx, method_window in enumerate(METHODS_ALL):
        color = METHOD_COLOR[method_window]
        if method_window == BASE_KEY:
            ax.scatter([x[idx]], [1.0], s=90, marker="D", color=color, zorder=4)
            continue

        values = rel_sub.loc[
            rel_sub["method_window"] == method_window, "rel_mse"
        ].to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size:
            jitter = rng.uniform(-0.18, 0.18, size=values.size)
            ax.scatter(
                np.full(values.size, x[idx]) + jitter,
                values,
                s=22,
                alpha=0.38,
                color=color,
                edgecolors="none",
                zorder=2,
            )

        row = rel_sum_sub.loc[rel_sum_sub["method_window"] == method_window]
        if row.shape[0] == 1:
            q25 = float(row["q25"].iloc[0])
            q75 = float(row["q75"].iloc[0])
            gmean = float(row["gmean"].iloc[0])
            ax.plot([x[idx], x[idx]], [q25, q75], color=color, linewidth=2.2, zorder=3)
            ax.plot(
                [x[idx] - 0.08, x[idx] + 0.08],
                [q25, q25],
                color=color,
                linewidth=2.2,
                zorder=3,
            )
            ax.plot(
                [x[idx] - 0.08, x[idx] + 0.08],
                [q75, q75],
                color=color,
                linewidth=2.2,
                zorder=3,
            )
            ax.scatter([x[idx]], [gmean], s=90, marker="D", color=color, zorder=4)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_LABEL[m] for m in METHODS_ALL], rotation=0, ha="center")
    ax.set_xlim(-0.5, len(METHODS_ALL) - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=xtick_labelsize(len(METHODS_ALL)))
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)
    for tick in ax.get_xticklabels():
        if tick.get_text() == PRETTY_LABEL[BASE_KEY]:
            tick.set_fontweight("bold")
    set_linear_ylim_with_pad(ax, all_rel_values)


def plot_error_panel(ax, df_sub: pd.DataFrame, ylims: tuple[float, float]) -> None:
    plot_df = df_sub.copy()
    plot_df["err"] = plot_df["estimate"] - plot_df["h2"]
    order = [PRETTY_LABEL[m] for m in METHODS_ALL]
    palette = {PRETTY_LABEL[m]: METHOD_COLOR[m] for m in METHODS_ALL}

    sns.boxplot(
        data=plot_df,
        x="pretty",
        y="err",
        order=order,
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1.2,
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.8)
    ax.set_xlabel("")
    ax.set_xticks(np.arange(len(METHODS_ALL), dtype=float))
    ax.set_xticklabels(order, rotation=0, ha="center")
    ax.set_xlim(-0.5, len(METHODS_ALL) - 0.5)
    ax.margins(x=0.0)
    ax.tick_params(axis="x", labelsize=xtick_labelsize(len(METHODS_ALL)))
    ax.tick_params(axis="y", labelsize=YTICK_LABEL_FONTSIZE)
    for tick in ax.get_xticklabels():
        if tick.get_text() == PRETTY_LABEL[BASE_KEY]:
            tick.set_fontweight("bold")
    ax.set_ylim(*ylims)


def make_figure(df: pd.DataFrame, out_pdf: Path) -> None:
    sns.set_style("whitegrid")
    rng = np.random.default_rng(0)
    rel = compute_relmse_by_setting(df)
    all_rel_values = rel.loc[
        rel["method_window"].isin(METHODS_NO_BASE), "rel_mse"
    ].to_numpy(float)

    err_df = df.copy()
    err_df["err"] = err_df["estimate"] - err_df["h2"]
    err_ylims = robust_error_ylim_by_group(err_df)

    fig, axes = plt.subplots(2, len(H2_VALS), figsize=(11.4, 7.4), sharey=False)

    for col, h2 in enumerate(H2_VALS):
        rel_h2 = rel[np.isclose(rel["h2"], h2)].copy()
        rel_sum_h2 = summarize_relmse(
            rel_h2[rel_h2["method_window"].isin(METHODS_NO_BASE)]
        )
        plot_relmse_panel(axes[0, col], rel_h2, rel_sum_h2, all_rel_values, rng)
        axes[0, col].set_title(f"$h^2$ = {h2:g}", fontsize=TITLE_FONTSIZE)
        axes[0, col].set_ylabel(
            "Relative MSE" if col == 0 else "", fontsize=AXIS_LABEL_FONTSIZE
        )

        df_h2 = df[np.isclose(df["h2"], h2)].copy()
        plot_error_panel(axes[1, col], df_h2, err_ylims)
        axes[1, col].set_ylabel(
            "Error (estimate - true $h^2$)" if col == 0 else "",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure(
        pd.read_csv("data/sim_h2/eur300k_h2.tsv", sep="\t"),
        Path("figs/supplementary/fig_s07_h2_simulations_eur300k.pdf"),
    )
