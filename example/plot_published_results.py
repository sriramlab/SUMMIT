"""Draw the GENIE example from published aggregate estimates."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ("Smoking", "Sex", "Age", "Statin use")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "example/out/published-results")
    args = parser.parse_args()
    with (ROOT / "docs/wiki/assets/genie-published-gxe.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    labels, values = {}, {}
    for row in records:
        trait, environment = row["trait"], row["environment"]
        key = (trait, environment)
        if key in values or environment not in ENVIRONMENTS:
            raise ValueError("Duplicate estimate or unknown environment")
        if labels.setdefault(trait, row["label"]) != row["label"]:
            raise ValueError("Inconsistent trait label")
        estimate, se = float(row["estimate"]), float(row["standard_error"])
        if not np.isfinite([estimate, se]).all() or se <= 0:
            raise ValueError("Expected finite estimates and positive standard errors")
        values[key] = (estimate, se)
    expected = {(trait, env) for trait in labels for env in ENVIRONMENTS}
    if len(labels) != 12 or set(values) != expected:
        raise ValueError("Expected all four environments for each of twelve traits")
    outputs = [args.out_dir / f"genie-published-gxe.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Choose a new output directory; figures already exist")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(12, 6.4), sharey=True)
    fig.subplots_adjust(left=.21, right=.985, bottom=.16, top=.79, wspace=.30)
    fig.text(.035, .955, "Gene–environment interaction in UK Biobank",
             fontsize=17, weight="bold")
    fig.text(.035, .905, "Published GENIE results  |  12 selected traits  |  Points ± 2 SE",
             color="#555555")
    colors = ("#247d8c", "#73568a", "#c26b28", "#254d66")
    y = np.arange(len(labels))
    for ax, environment, color in zip(axes, ENVIRONMENTS, colors):
        points = np.array([values[(trait, environment)] for trait in labels])
        ax.errorbar(100 * points[:, 0], y, xerr=200 * points[:, 1],
                    fmt="o", color=color, markersize=4.5, elinewidth=1.2,
                    capsize=2.5)
        ax.axvline(0, color="#9ba3aa", linewidth=.8, linestyle="--")
        ax.set_title(environment, loc="left", pad=12, fontsize=12, weight="bold")
        ax.set_ylim(len(labels) - .4, -.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e5e8eb", linewidth=.7)
        ax.tick_params(axis="y", length=0, pad=9)
        for boundary in (2.5, 4.5, 7.5):
            ax.axhline(boundary, color="#e5e8eb", linewidth=.7)
    axes[0].set_yticks(y, list(labels.values()))
    fig.text(.60, .055, "G×E heritability (%) · horizontal scales differ between panels",
             ha="center", fontsize=10)
    fig.savefig(outputs[0], dpi=180, metadata={"Software": "SUMMIT"})
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Creator": "SUMMIT", "Date": None})
    outputs[1].write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n")
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
