#!/usr/bin/env python3
"""Compare four/eight-bin sensitivity bases on existing continuous-risk seeds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_pcgc import generate, SCENARIOS
from summit.sumstats.binary import prepare_binary_risk
from summit.pcgc.research import exact_moments
from summit.pcgc.moments import fit_moments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--variants", type=int, default=4000)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 19:
        parser.error("reuse existing pilot indices 1..19")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).resolve().with_name("validate_pcgc.py"),
               root/"src/summit/pcgc/research.py", root/"src/summit/pcgc/moments.py", root/"src/summit/sumstats/binary.py"]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    scope="paired basis-resolution diagnostic with supplied risks; not new confirmation")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    records = []
    for scenario in ("S3", "S4"):
        for index in range(1, args.replicates+1):
            seed = 820000+1000*int(scenario[1:])+index
            data = generate(seed, scenario, args.samples, args.variants, 1, 40)
            risk = prepare_binary_risk(data["y"], SCENARIOS[scenario]["K"], population_risk=data["k"], covariate_variance=data["Vc"])
            exact = exact_moments(data["x"], data["annotations"], risk)
            target = fit_moments(exact)["marginal_total"]
            H, _ = exact.equations()
            for bins in (4, 8):
                labels = np.searchsorted(np.quantile(risk.sensitivity, np.arange(1, bins)/bins), risk.sensitivity)
                averages = {g: risk.sensitivity[labels == g].mean() for g in np.unique(labels)}
                approximated = np.array([averages[g] for g in labels])
                moments = exact_moments(data["x"], data["annotations"], risk, "pcgc-basis", sensitivity=approximated)
                fit = fit_moments(moments)
                records.append(dict(scenario=scenario, seed=seed, bins=bins,
                    relative_weight_error=float(np.linalg.norm(approximated-risk.sensitivity)/np.linalg.norm(risk.sensitivity)),
                    maximum_relative_weight_error=float(np.max(np.abs(approximated/risk.sensitivity-1))),
                    relative_gram_error=float(np.linalg.norm(moments.equations()[0]-H)/np.linalg.norm(H)),
                    exact_marginal_estimate=target, marginal_estimate=fit["marginal_total"], difference=fit["marginal_total"]-target))
            print(json.dumps(dict(scenario=scenario, replicate=index)), flush=True)
    (args.out/"replicates.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
