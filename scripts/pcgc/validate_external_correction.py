#!/usr/bin/env python3
"""Re-evaluate finite-pair external transfer on existing confirmation datasets.

Only the risk scalar changes. With zero same-person matrix, every full and
frozen-delete estimate/SE rescales by old_factor/new_factor; covariance by its
square. Regenerate the risks with the original seed, without any Gram products.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_pcgc import generate, SCENARIOS
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk
from summit.pcgc.moments import risk_pair_factor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("validate_pcgc.py"), root/"src/summit/sumstats/binary.py", root/"src/summit/pcgc/moments.py"]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    records, inputs, seen = [], {}, set()
    for directory in args.runs:
        old_manifest = json.loads((directory/"manifest.json").read_text())
        original = json.loads((directory/"replicates.json").read_text())
        settings = old_manifest["arguments"]
        if settings["seed_offset"] != 20 or settings["replicates"] != 80 or settings.get("generator", "gaussian") != "gaussian":
            raise ValueError("this paired correction expects the original Gaussian 80-dataset confirmation")
        if old_manifest["source_hashes"]["src/summit/pcgc/moments.py"] != "71fce39540e0df3d7a97ded3a730c559f76730126e62060819ce0afb03b2c6d4":
            raise ValueError("source run does not use the original mean-square transfer; do not correct it twice")
        inputs[str(directory)] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                  for name in ("manifest.json", "replicates.json")}
        for scenario, seed in sorted({(r["scenario"], r["seed"]) for r in original}):
            if (scenario, seed) in seen:
                raise ValueError("duplicate source dataset")
            seen.add((scenario, seed))
            data = generate(seed, scenario, settings["samples"], settings["variants"], 1, settings["ld_block"])
            risks = {
                "supplied": prepare_binary_risk(data["y"], SCENARIOS[scenario]["K"], population_risk=data["k"], covariate_variance=data["Vc"]),
                "fitted": fit_binary_risk(data["y"], SCENARIOS[scenario]["K"], None if data["gamma"] == 0 else data["cov"][:, None]),
            }
            for old in original:
                if (old["scenario"], old["seed"], old["method"]) != (scenario, seed, "pcgc-ld"):
                    continue
                row = dict(old)
                risk = risks[row["risk"]]
                diagnostics = risk.diagnostics()
                for key in ("population_risk_range", "sensitivity_range", "covariate_variance"):
                    np.testing.assert_allclose(diagnostics[key], old["risk_diagnostics"][key], rtol=1e-10, atol=1e-12)
                factor = np.mean(risk.sensitivity**2)**2/risk_pair_factor(risk)
                for key in ("conditional_components", "marginal_components", "conditional_total", "marginal_total",
                            "conditional_standard_errors", "conditional_total_standard_error"):
                    if key in row:
                        row[key] = (np.asarray(row[key])*factor).tolist()
                if "conditional_jackknife_covariance" in row:
                    row["conditional_jackknife_covariance"] = (np.asarray(row["conditional_jackknife_covariance"])*factor**2).tolist()
                if "minimum_eigenvalue" in row:
                    row["minimum_eigenvalue"] /= factor
                row.update(finite_pair_estimate_multiplier=float(factor), correction="distinct_person_risk_average_v1")
                records.append(row)
        print(json.dumps(dict(source=str(directory), corrected_records=len(records))), flush=True)
    manifest = dict(generator="gaussian", arguments=dict(seed_offset=20, replicates=80, jackknife=50), inputs=inputs,
                    source_hashes=source_hashes,
                    status="paired reevaluation after finite-pair correction; no new datasets or estimator tuning")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.out/"replicates.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
