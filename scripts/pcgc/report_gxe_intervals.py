"""Summarize calibrate_gxe_intervals.py outputs as Markdown tables.

Reads every *.json result in the given directories and prints, for each
setting and component, the variance ratio (mean estimated variance over the
empirical variance) and 95% normal-interval coverage, with Monte Carlo SEs,
for the SNP-block jackknife, main's experimental sampling SEs (when run) and
the cross-block calibrated estimator.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def label(spec):
    parts = [("fixed" if spec["design"] == "fixed" else "ascertained P=.5"),
             f"K={spec['K']:g}", f"h={spec['h']:g}", f"Q={spec['q']}", f"N={spec['n']}", f"M={spec['m']}"]
    if spec.get("ld") != "none":
        parts.append("AR(1) LD")
    if spec.get("annotations") != "one":
        parts.append("overlapping annotations")
    if spec.get("risk") != "supplied":
        parts.append("fitted risks")
    if spec.get("method", "pcgc") != "pcgc":
        parts.append(spec["method"])
    if spec.get("risk_slope", .3) != .3:
        parts.append(f"risk slope {spec['risk_slope']:g}")
    return ", ".join(parts)


def component_names(spec):
    from itertools import combinations_with_replacement
    q = spec["q"]
    pairs = [(u, u) for u in range(q)]+[(u, v) for u in range(q) for v in range(u+1, q)]
    k = 1 if spec["annotations"] == "one" else 2
    return [f"{'A'+str(a)+' ' if k > 1 else ''}Omega_{u}{v}" for a in range(k) for u, v in pairs]


def cell(block, j):
    if block is None:
        return "—"
    return (f"{block['variance_ratio'][j]:.2f} ({block['variance_ratio_mcse'][j]:.2f}); "
            f"{100*block['coverage'][j]:.1f} ({100*block['coverage_mcse'][j]:.1f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for directory in args.directories:
        for path in sorted(directory.glob("*.json")):
            try:
                result = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "spec" not in result or "jackknife" not in result:
                continue
            rows.append((path.stem, result))
    print("| Setting | Component | Truth | Rel. bias | Jackknife | Sampling | Sampling + architecture | Cross-block |")
    print("|---|---|---:|---:|---|---|---|---|")
    for name, result in rows:
        spec = result["spec"]
        names = component_names(spec)
        truth = np.asarray(result["truth"])
        mean = np.asarray(result["mean"])
        for j, comp in enumerate(names):
            bias = (mean[j]-truth[j])/truth[j] if truth[j] else np.nan
            print(f"| {name}: {label(spec)} (R={result['replicates']-result['failures']}) | {comp} | {truth[j]:.3g} | "
                  f"{'—' if np.isnan(bias) else f'{100*bias:+.1f}%'} | {cell(result['jackknife'], j)} | "
                  f"{cell(result.get('sampling'), j)} | {cell(result.get('architecture'), j)} | {cell(result['crossblock'], j)} |")


if __name__ == "__main__":
    main()
