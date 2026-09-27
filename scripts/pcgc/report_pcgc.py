#!/usr/bin/env python3
"""Summarize predeclared PCGC accuracy/calibration screens, preserving failures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def wilson(successes, n):
    z = 1.959963984540054
    center = (successes/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(successes/n*(1-successes/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def metrics(estimates, truths, standard_errors, attempted, *, component=False, absolute_margin=None):
    est, truth, se = map(np.asarray, (estimates, truths, standard_errors))
    point = np.isfinite(est) & np.isfinite(truth)
    intervals = point & np.isfinite(se) & (se >= 0)
    n = int(point.sum())
    result = dict(attempted=attempted, finite_estimates=n, finite_intervals=int(intervals.sum()))
    if n < 2:
        return dict(result, qualified_screen=False, reason="fewer than two finite estimates")
    error = (est-truth)[point]
    sd = est[point].std(ddof=1)
    mcse = error.std(ddof=1)/np.sqrt(n)
    ci = error.mean()+np.array([-1, 1])*t.ppf(.975, n-1)*mcse
    margin = max(.01 if component else .02, .1*abs(truth[point].mean())) if absolute_margin is None else absolute_margin
    covered = intervals & (np.abs(est-truth) <= 1.959963984540054*se)
    rejected = intervals & (np.abs(est) > 1.959963984540054*se)
    coverage_ci, rejection_ci = wilson(int(covered.sum()), attempted), wilson(int(rejected.sum()), attempted)
    rms = float(np.sqrt(np.mean(se[intervals]**2))) if intervals.any() else None
    ratio = rms/sd if rms is not None and sd > 0 else None
    ratio_ci = None
    if intervals.sum() >= 3:
        rng = np.random.default_rng(92518)
        valid_est, valid_se = est[intervals], se[intervals]
        indices = rng.integers(0, len(valid_est), (2000, len(valid_est)))
        bootstrap_sd = valid_est[indices].std(axis=1, ddof=1)
        ratios = np.sqrt(np.mean(valid_se[indices]**2, axis=1))/bootstrap_sd
        ratio_ci = np.quantile(ratios[np.isfinite(ratios)], [.025, .975]).tolist()
    null = bool(np.all(truth[point] == 0))
    bias_ok = bool(np.all(np.abs(ci) < margin))
    calibration_ok = (coverage_ci[0] > .85 and coverage_ci[0] <= .95 <= coverage_ci[1] and
                      ratio is not None and .8 <= ratio <= 1.25 and intervals.sum() == attempted)
    if null:
        calibration_ok = calibration_ok and rejection_ci[0] <= .05 <= rejection_ci[1] and rejection_ci[1] < .15
    return dict(result, truth=float(truth[point].mean()), mean=float(est[point].mean()), bias=float(error.mean()),
                bias_mcse=float(mcse), bias_ci=ci.tolist(), bias_margin=margin, bias_equivalent=bias_ok,
                empirical_sd=float(sd), rmse=float(np.sqrt(np.mean(error**2))), rms_standard_error=rms,
                rms_se_over_sd=ratio, rms_se_over_sd_bootstrap_ci=ratio_ci,
                coverage=float(covered.sum()/attempted), coverage_wilson_ci=coverage_ci,
                null_rejection=float(rejected.sum()/attempted) if null else None,
                null_rejection_wilson_ci=rejection_ci if null else None,
                calibration_screen=bool(calibration_ok), qualified_screen=bool(bias_ok and calibration_ok and n == attempted))


def collect(runs):
    records, sources, seen = [], {}, set()
    for path in runs:
        manifest = json.loads((path/"manifest.json").read_text())
        rows = json.loads((path/"replicates.json").read_text())
        generator = manifest["generator"]
        if generator == "conditional_gaussian_liability_v1":
            generator = "gaussian"
        for row in rows:
            key = (generator, row["scenario"], row["seed"], row["risk"], row["method"])
            if key in seen:
                raise ValueError(f"duplicate simulation result: {key}")
            seen.add(key)
            row["generator"] = generator
            row["phase"] = "confirmation" if manifest["arguments"]["seed_offset"] >= 20 else "pilot"
            row["jackknife_blocks"] = manifest["arguments"].get("jackknife")
            records.append(row)
        sources[str(path)] = {"manifest_sha256": hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest(),
                              "results_sha256": hashlib.sha256((path/"replicates.json").read_bytes()).hexdigest()}
    for generator, scenario in {(r["generator"], r["scenario"]) for r in records}:
        if len({r["seed"] for r in records if r["generator"] == generator and r["scenario"] == scenario}) > 100:
            raise ValueError("scenario exceeds the 100-dataset limit")
    return records, sources


def summarize(records):
    groups = {}
    for r in records:
        group = tuple(r[k] for k in ("generator", "phase", "scenario", "risk", "method"))
        groups.setdefault(group, []).append(r)
    summaries = []
    for key, rows in sorted(groups.items()):
        k = len(rows[0]["truth"])
        for scale in ("conditional", "marginal"):
            if key[-2] == "global" and scale == "conditional":
                continue  # A global population-liability conversion is marginal.
            for component in range(k+1):
                label = "total" if component == k else f"component_{component}"
                est, truth, se = [], [], []
                for r in rows:
                    tr = r["truth"] if scale == "conditional" else r["marginal_truth"]
                    truth.append(sum(tr) if component == k else tr[component])
                    factor = 1 if scale == "conditional" else 1+r.get("risk_diagnostics", {}).get("covariate_variance", 0)
                    if "failure" in r:
                        est.append(np.nan)
                        se.append(np.nan)
                    elif component == k:
                        est.append(r[f"{scale}_total"])
                        se.append(r["conditional_total_standard_error"]/factor)
                    else:
                        est.append(r[f"{scale}_components"][component])
                        se.append(r.get("conditional_standard_errors", [np.nan]*k)[component]/factor)
                result = metrics(est, truth, se, len(rows), component=component != k)
                result.update(dict(zip(("generator", "phase", "scenario", "risk", "method"), key)), scale=scale, quantity=label)
                if key[2] == "S6":
                    result.update(qualified_screen=False, scope="deliberately_mismatched_kernel")
                summaries.append(result)
    return summaries


def paired(records):
    baseline = {(r["generator"], r["phase"], r["scenario"], r["risk"], r["seed"]): r
                for r in records if r["method"] == "pcgc" and "failure" not in r}
    groups = {}
    for r in records:
        key = tuple(r[k] for k in ("generator", "phase", "scenario", "risk", "seed"))
        if r["method"] == "pcgc" or "failure" in r or key not in baseline:
            continue
        base = baseline[key]
        truth = sum(r["marginal_truth"])
        differences = (r["marginal_total"]-base["marginal_total"],
                       (r["marginal_total"]-truth)**2-(base["marginal_total"]-truth)**2)
        groups.setdefault((*key[:-1], r["method"]), []).append(differences)
    result = []
    for key, rows in sorted(groups.items()):
        x = np.asarray(rows)
        if len(x) < 2:
            continue
        half = t.ppf(.975, len(x)-1)*x.std(axis=0, ddof=1)/np.sqrt(len(x))
        result.append(dict(zip(("generator", "phase", "scenario", "risk", "method"), key),
            pairs=len(x), estimate_difference=float(x[:, 0].mean()), mse_difference=float(x[:, 1].mean()),
            estimate_difference_ci=[float(x[:, 0].mean()-half[0]), float(x[:, 0].mean()+half[0])],
            mse_difference_ci=[float(x[:, 1].mean()-half[1]), float(x[:, 1].mean()+half[1])]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records, sources = collect(args.runs)
    summaries = summarize(records)
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(sources=sources, screens=summaries, paired_comparisons=paired(records),
                  uncertainty="Frozen target-row jackknife; no risk or prevalence refitting. Screens apply only to the listed scenarios.")
    (args.out/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    fields = ["generator", "phase", "scenario", "risk", "method", "scale", "quantity", "attempted", "bias", "bias_ci",
              "bias_equivalent", "empirical_sd", "rms_se_over_sd", "coverage", "coverage_wilson_ci", "calibration_screen", "qualified_screen"]
    with (args.out/"screens.tsv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
