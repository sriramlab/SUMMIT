#!/usr/bin/env python3
"""Render the validated R=100 generalized GxE benchmark figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "omega_0_0": r"$\Omega_{00}$",
    "omega_1_1": r"$\Omega_{11}$",
    "omega_2_2": r"$\Omega_{22}$",
    "omega_0_1": r"$\Omega_{01}$",
    "omega_0_2": r"$\Omega_{02}$",
    "omega_1_2": r"$\Omega_{12}$",
}
BLUE = "#2673b8"
GREEN = "#2a9d6f"
GRAY = "#59636e"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _arrays(batch: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = list(batch["component_names"])
    return (
        names,
        np.asarray(batch["estimates"], dtype=np.float64),
        np.asarray(batch["standard_errors"], dtype=np.float64),
        np.asarray(batch["truth"], dtype=np.float64),
    )


def _component(array: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return array[:, names.index(name)]


def _pair_order(q: int) -> tuple[tuple[int, int], ...]:
    return tuple((index, index) for index in range(q)) + tuple(
        (left, right)
        for left in range(q)
        for right in range(left + 1, q)
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _render(b128: dict, b1024: dict, output: Path) -> None:
    diagonal128 = b128["batches"]["diagonal_two_environment"]
    diagonal1024 = b1024["batches"]["diagonal_two_environment"]
    signal1024 = b1024["batches"]["offdiagonal_two_environment"]
    names128, estimates128, se128, truth128 = _arrays(diagonal128)
    names1024, estimates1024, se1024, truth1024 = _arrays(diagonal1024)
    signal_names, signal_estimates, signal_se, signal_truth = _arrays(signal1024)
    genetic_names = [f"omega_{left}_{right}" for left, right in _pair_order(3)]
    null_names = ("omega_0_1", "omega_0_2", "omega_1_2")
    rng = np.random.default_rng(7100)

    def ratio(
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        truth: np.ndarray,
        names: list[str],
        name: str,
    ) -> float:
        error = _component(estimates, names, name) - _component(truth, names, name)
        return float(
            np.std(error, ddof=1)
            / np.mean(_component(standard_errors, names, name))
        )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    x = np.arange(len(genetic_names))
    width = 0.36

    ax = axes[0, 0]
    ratio128 = [
        ratio(estimates128, se128, truth128, names128, name)
        for name in genetic_names
    ]
    ratio1024 = [
        ratio(estimates1024, se1024, truth1024, names1024, name)
        for name in genetic_names
    ]
    ax.bar(x - width / 2, ratio128, width, color=BLUE, label="B=128")
    ax.bar(x + width / 2, ratio1024, width, color=GREEN, label="B=1024")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(
        x, [DISPLAY[name] for name in genetic_names], rotation=30, ha="right"
    )
    ax.set_ylabel("empirical error SD / mean JK SE")
    ax.set_title("A. Probe-count calibration sensitivity")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    empirical = []
    reported = []
    for name in genetic_names:
        error = _component(estimates1024, names1024, name) - _component(
            truth1024, names1024, name
        )
        empirical.append(np.std(error, ddof=1))
        reported.append(np.mean(_component(se1024, names1024, name)))
    ax.bar(
        x - width / 2,
        empirical,
        width,
        color=BLUE,
        label="empirical MC error SD",
    )
    ax.bar(
        x + width / 2,
        reported,
        width,
        color=GREEN,
        label="mean SNP-JK SE",
    )
    ax.set_xticks(
        x, [DISPLAY[name] for name in genetic_names], rotation=30, ha="right"
    )
    ax.set_ylabel("SE")
    ax.set_title("B. B=1024 absolute SE comparison")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for position, name in enumerate(null_names):
        z = _component(estimates1024, names1024, name) / _component(
            se1024, names1024, name
        )
        jitter = rng.uniform(-0.14, 0.14, size=z.size)
        ax.scatter(position + jitter, z, s=14, alpha=0.60, color=GREEN)
        rejected = int(np.sum(np.abs(z) > 1.96))
        ax.annotate(
            f"{rejected}/{z.size} rejected",
            xy=(position, np.max(z)),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.axhline(1.96, color=GRAY, linestyle="--", linewidth=1.0)
    ax.axhline(-1.96, color=GRAY, linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(null_names)), [DISPLAY[name] for name in null_names])
    ax.set_ylabel("estimate / reported SNP-JK SE")
    ax.set_title("C. All 100 null z-scores at B=1024")

    ax = axes[1, 1]
    null_values = _component(estimates1024, names1024, "omega_1_2")
    signal_values = _component(signal_estimates, signal_names, "omega_1_2")
    parts = ax.violinplot(
        [null_values, signal_values],
        positions=[0, 1],
        widths=0.72,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], (BLUE, GREEN), strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
    for position, values, errors, color in (
        (0, null_values, _component(se1024, names1024, "omega_1_2"), BLUE),
        (
            1,
            signal_values,
            _component(signal_se, signal_names, "omega_1_2"),
            GREEN,
        ),
    ):
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(position + jitter, values, s=13, alpha=0.55, color=color)
        ax.errorbar(
            position,
            np.mean(values),
            yerr=1.96 * np.mean(errors),
            fmt="D",
            color="black",
            markersize=4,
            capsize=3,
        )
    ax.axhline(0.0, color=BLUE, linestyle="--", linewidth=1.0)
    ax.axhline(
        float(np.mean(_component(signal_truth, signal_names, "omega_1_2"))),
        color=GREEN,
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xticks([0, 1], ["null", r"$\Omega_{12}=0.18$"])
    ax.set_ylabel(r"estimated $\Omega_{12}$")
    ax.set_title("D. Null and generalized-signal separation")

    figure.suptitle(
        "Generalized G×E uncertainty benchmark with 100 replicates",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r100-b128", type=Path, required=True)
    parser.add_argument("--r100-b1024", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "r100_calibration.png"
    _style()
    _render(_load(args.r100_b128), _load(args.r100_b1024), output)
    print(json.dumps({"figure": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
