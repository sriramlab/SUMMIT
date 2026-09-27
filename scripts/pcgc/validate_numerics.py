#!/usr/bin/env python3
"""Probe-error audit against exact PCGC Gram matrices and estimates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from summit.pcgc.moments import fit_moments
from summit.pcgc.reference import prepare_moments
from summit.pcgc.research import exact_moments
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.sumstats.binary import prepare_binary_risk
from validate_pcgc import generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--variants", type=int, default=400)
    p.add_argument("--native", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--probes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--scenario", choices=("S3", "S5"), default="S3")
    args = p.parse_args()
    if not 1 <= args.seeds <= 8:
        p.error("use 1..8 numerical seeds")
    args.out.mkdir(parents=True, exist_ok=False)
    data = generate(810021, args.scenario, args.samples, args.variants, 100, 40)
    x, a = data["x"], data["annotations"]
    risk = prepare_binary_risk(data["y"], .1, population_risk=data["k"], covariate_variance=data["Vc"])
    records = []
    for method in ("pcgc", "pcgc-inverse"):
        exact = exact_moments(x, a, risk, method)
        exact_H, _ = exact.equations()
        blocks = np.arange(len(a))*min(50,len(a))//len(a)
        exact_fit = fit_moments(exact, block_ids=blocks)
        exact_theta = exact_fit["conditional_total"]
        for probes in args.probes:
            for seed in range(args.seeds):
                start = time.perf_counter()
                m, diagnostics = prepare_moments(ArraySequentialGenotypeOperator(x), a, risk, method,
                    probes=probes, seed=seed+9613, native=args.native, threads=args.threads)
                H, _ = m.equations()
                record = dict(method=method, probes=probes, seed=seed,
                    offdiagonal_gram_relative_error=float(np.linalg.norm(H-exact_H)/np.linalg.norm(exact_H)),
                    signed_offdiagonal_gram_relative_error=float((H[0, 0]-exact_H[0, 0])/exact_H[0, 0]),
                    exact_estimate=exact_theta, seconds=time.perf_counter()-start, **diagnostics)
                try:
                    fit = fit_moments(m, block_ids=blocks)
                    record["estimate_error"] = fit["conditional_total"]-exact_theta
                    record["component_errors"] = (np.asarray(fit['conditional_components'])-exact_fit['conditional_components']).tolist()
                    record["component_error_over_se"] = (np.asarray(record['component_errors'])/exact_fit['conditional_standard_errors']).tolist()
                    record["normal_relative_asymmetry"] = fit['normal_relative_asymmetry']
                except ValueError as exc:
                    record["failure"] = str(exc)
                records.append(record)
    (args.out/"numerics.json").write_text(json.dumps(dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, rows=records), indent=2)+"\n")
    for method in ("pcgc", "pcgc-inverse"):
        for probes in args.probes:
            rows = [r for r in records if r["method"] == method and r["probes"] == probes]
            print(json.dumps(dict(method=method, probes=probes,
                rms_relative_gram_error=float(np.sqrt(np.mean([r["offdiagonal_gram_relative_error"]**2 for r in rows]))),
                rms_estimate_error=float(np.sqrt(np.mean([r.get("estimate_error", np.nan)**2 for r in rows]))))), flush=True)


if __name__ == "__main__":
    main()
