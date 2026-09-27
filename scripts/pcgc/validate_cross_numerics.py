#!/usr/bin/env python3
"""Check rectangular cross moments against exact Gram products with overlap."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cross_simulation import generate
from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.pcgc.cross import prepare_pair, fit_pair, SELECTION_CONTRACT
from summit.pcgc.research import exact_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--variants", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 8:
        parser.error("use 1..8 numerical seeds")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).resolve().with_name("cross_simulation.py"),
               *sorted((root/"src/summit/pcgc").glob("*.py"))]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    data = generate(911000, "BB_shared", args.samples, args.variants)
    x, a = data["x"], np.ones((args.variants, 1))
    pair_args = data["left"], data["right"], data["left_rows"], data["right_rows"]
    exact = exact_pair(x, a, *pair_args)
    exact_fit = fit_pair(exact)
    records = []
    for probes in (64, 256):
        for seed in range(args.seeds):
            start = time.perf_counter()
            result = prepare_pair(ArraySequentialGenotypeOperator(x), a, *pair_args, selection_contract=SELECTION_CONTRACT,
                                  probes=probes, seed=9726+seed, native=args.native, threads=args.threads)
            record = dict(probes=probes, seed=seed, seconds=time.perf_counter()-start)
            for name in ("left", "right", "cross"):
                H, _ = getattr(result, name).equations()
                expected, _ = getattr(exact, name).equations()
                record[name+"_relative_gram_error"] = float(np.linalg.norm(H-expected)/np.linalg.norm(expected))
            try:
                fit = fit_pair(result)
                for name in ("conditional_covariance", "genetic_correlation"):
                    record[name+"_error"] = None if fit[name] is None else fit[name]-exact_fit[name]
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                record["failure"] = str(exc)
            records.append(record)
    report = dict(manifest, rows=records)
    (args.out/"numerics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    for probes in (64, 256):
        selected = [r for r in records if r["probes"] == probes]
        print(json.dumps(dict(probes=probes, rms_relative_cross_gram_error=float(np.sqrt(np.mean([r["cross_relative_gram_error"]**2 for r in selected]))),
                              rms_covariance_error=float(np.sqrt(np.mean([r["conditional_covariance_error"]**2 for r in selected]))))), flush=True)


if __name__ == "__main__":
    main()
