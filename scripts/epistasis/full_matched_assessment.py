"""Compact assessment with scheduled denominators and Monte Carlo intervals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


def interval(k, n):
    if not n:
        return [None, None]
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
            float(beta.ppf(.975, k+1, n-k)) if k < n else 1.]


def assess(roots):
    summary = []
    resources = []
    for root in roots:
        root = root.resolve()
        design = json.loads((root / "design.json").read_text())
        arguments = design["arguments"]
        if not (root / "resources.json").exists():
            raise ValueError(f"scheduled workload incomplete: {root}")
        records = [json.loads(line) for line in (root / "replicates.jsonl").read_text().splitlines()]
        reference = json.loads((Path(design["reference"]) / "reference.json").read_text())
        settings = arguments["settings"].split(",")
        n = arguments["replicates"]
        methods = arguments.get("methods", "learned,burden,oracle,joint").split(",")
        if len(records) != n*len(settings)*len(methods):
            raise ValueError(f"scheduled denominator disagrees: {root}")
        for setting in settings:
            for method in methods:
                rows = [r for r in records if r["setting"] == setting and r["method"] == method]
                if len(rows) != n or len({r["replicate"] for r in rows}) != n:
                    raise ValueError("missing or duplicated scheduled replicate")
                valid = [r for r in rows if not r["failed"]]
                supported = [r for r in valid if not r["outside_scope"]]
                out = dict(root=str(root), phase=design["phase"], target=reference["target"],
                    setting=setting, method=method, training_n=arguments["training_samples"],
                    confirmation_n=arguments["confirmation_samples"], markers=reference["m"],
                    structure_covariates=arguments.get("structure_covariates", ""),
                    scheduled=n, numerical_failures=n-len(valid), unsupported=len(valid)-len(supported),
                    biological_null=design.get("scenario_definitions", reference["definitions"])[setting]["biological_null"])
                for suffix, selected in [("all_numerical", valid), ("supported", supported)]:
                    for label, alpha in [("05", .05), ("005", .005)]:
                        k = sum(r["p"] < alpha for r in selected)
                        out[f"rejections_{label}_{suffix}"] = k
                        out[f"denominator_{suffix}"] = len(selected)
                        out[f"rate_{label}_{suffix}"] = k/len(selected) if selected else None
                        out[f"mc95_{label}_{suffix}"] = interval(k, len(selected))
                if valid:
                    out["maximum_leverage"] = max(r["max_leverage"] for r in valid)
                    out["mean_remaining_signal_variance"] = np.mean([r["remaining_signal_variance"] for r in valid])
                    alignment = [r["direction_alignment"] for r in valid if r["direction_alignment"] is not None]
                    out["mean_direction_alignment"] = float(np.mean(alignment)) if alignment else None
                    out["reference_signal_variance"] = valid[0]["reference_signal_variance"]
                    out["mean_realized_training_signal_variance"] = np.mean([r["realized_training_signal_variance"] for r in valid])
                    out["mean_realized_confirmation_signal_variance"] = np.mean([r["realized_confirmation_signal_variance"] for r in valid])
                    if method != "joint":
                        error = np.array([r["error"] for r in valid])
                        se = np.array([r["se"] for r in valid])
                        out.update(bias=float(error.mean()), empirical_error_sd=float(error.std(ddof=1)) if len(error)>1 else None,
                            mean_se=float(se.mean()), rms_se=float(np.sqrt(np.mean(se**2))),
                            mean_standardized_error=float(np.mean(error/se)),
                            projection_coverage=float(np.mean([r["coverage"] for r in valid])),
                            projection_coverage_mc95=interval(sum(r["coverage"] for r in valid), len(valid)),
                            mean_leakage_noise_ratio=float(np.mean([r["leakage_noise_ratio"] for r in valid])),
                            rms_leakage_noise_ratio=float(np.sqrt(np.mean([r["leakage_noise_ratio"]**2 for r in valid]))))
                    covered = [r["joint_coverage"] for r in valid if "joint_coverage" in r]
                    out["joint_projection_coverage"] = np.mean(covered) if covered else None
                summary.append(out)
        resources.append(dict(root=str(root), **json.loads((root / "resources.json").read_text())))
    return summary, resources


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summaries, resources = assess(a.inputs)
    a.out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(a.out / "assessment.csv", index=False)
    with (a.out / "assessment.json").open("x") as handle:
        json.dump(dict(summaries=summaries, resources=resources), handle, indent=2)
    columns = ["target", "setting", "method", "scheduled", "numerical_failures", "unsupported",
               "rate_05_all_numerical", "rate_005_all_numerical", "projection_coverage", "mean_direction_alignment"]
    print(pd.DataFrame(summaries).reindex(columns=columns).to_string(index=False))


if __name__ == "__main__":
    main()
