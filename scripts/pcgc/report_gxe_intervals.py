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


def decision(rows):
    """Apply the predeclared default rule (decision_rule.txt) to every cell."""
    from scipy import stats
    failures, cells = [], []
    for name, result in rows:
        data = np.load(Path(result["_path"]).with_suffix(".npz"))
        ok = ~data["failed"]
        est, truth = data["estimate"][ok], data["truth"]
        jk = data["jk_var"][ok]
        new = jk+data["between_var"][ok]+data["nuisance_var"][ok]
        n = len(est)
        err = est-truth
        for j in range(est.shape[1]):
            cj = np.abs(err[:, j]) <= 1.959963984540054*np.sqrt(jk[:, j])
            cn = np.abs(err[:, j]) <= 1.959963984540054*np.sqrt(new[:, j])
            diff = cn.astype(float)-cj
            dse = diff.std(ddof=1)/np.sqrt(n)
            mcse = np.sqrt(cn.mean()*(1-cn.mean())/n)
            ratio, ratio_mcse = result["crossblock"]["variance_ratio"][j], result["crossblock"]["variance_ratio_mcse"][j]
            sd = est[:, j].std(ddof=1)
            se = np.sqrt(new[:, j])
            cell = dict(setting=name, component=j, truth=float(truth[j]), coverage=float(cn.mean()), mcse=float(mcse),
                        jk_coverage=float(cj.mean()), ratio=ratio, jk_ratio=result["jackknife"]["variance_ratio"][j],
                        oracle_coverage=float(np.mean(np.abs(err[:, j]) <= 1.959963984540054*sd)),
                        se_estimate_correlation=float(np.corrcoef(se, err[:, j])[0, 1]),
                        relative_sd_of_se=float(se.std()/se.mean()),
                        t_coverage=float(np.mean(np.abs(err[:, j]) <= stats.t.ppf(.975, result["spec"]["blocks"]-1)*se)))
            flags = ""
            if abs(ratio-1) > max(.1, 3*ratio_mcse): flags += "a"
            if cell["coverage"] < .95-2.58*mcse: flags += "b"
            if truth[j] == 0 and 1-cell["coverage"] > .05+2.58*mcse: flags += "c"
            if dse > 0 and diff.mean() < -2.58*dse: flags += "d"
            cell["failed_rules"] = flags
            cells.append(cell)
            if flags:
                failures.append(cell)
    return cells, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--decision", action="store_true", help="apply the predeclared default rule")
    parser.add_argument("--exclude", nargs="*", default=[], help="setting names excluded from the rule")
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
            result["_path"] = str(path)
            rows.append((path.stem, result))
    if args.decision:
        cells, failures = decision([r for r in rows if r[0] not in args.exclude])
        coverage = np.array([c["coverage"] for c in cells])
        print(json.dumps(dict(cells=len(cells), mean_coverage=coverage.mean(),
            minimum_coverage=coverage.min(), failures=[(c["setting"], c["component"], c["failed_rules"],
            round(c["coverage"], 3), round(c["jk_coverage"], 3)) for c in failures],
            ranges={k: [min(c[k] for c in cells), max(c[k] for c in cells)] for k in
                    ("ratio", "jk_ratio", "coverage", "jk_coverage", "oracle_coverage", "t_coverage",
                     "se_estimate_correlation", "relative_sd_of_se")}), indent=1))
        return
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
