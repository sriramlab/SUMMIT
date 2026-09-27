#!/usr/bin/env python3
"""Bounded bivariate PCGC pilot with explicit overlap selection accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cross_simulation import generate
from summit.pcgc.cross import fit_pair
from summit.pcgc.research import exact_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=("BB0", "BB_shared", "BQ"), required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--variants", type=int, default=4000)
    parser.add_argument("--jackknife", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100 or not 0 <= args.seed_offset <= 100-args.replicates:
        parser.error("at most 100 replicates per scenario")
    if args.samples < 40 or args.samples % 4 or args.variants < 40 or args.variants % 40 or not 2 <= args.jackknife <= args.variants:
        parser.error("sample count must be divisible by four, variants by 40, and at least two jackknife blocks are required")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("cross_simulation.py"),
               *sorted((root/"src/summit/pcgc").glob("*.py")), root/"src/summit/sumstats/binary.py"]
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    generator="joint_population_gaussian_with_marginal_sampling_v1", reference="exact_study")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    a = np.ones((args.variants, 1))
    blocks = np.floor(np.arange(args.variants)*args.jackknife/args.variants).astype(int)
    rows = []
    for replicate in range(args.seed_offset, args.seed_offset+args.replicates):
        start = time.perf_counter()
        seed = 910000+1000*("BB0", "BB_shared", "BQ").index(args.scenario)+replicate
        data = generate(seed, args.scenario, args.samples, args.variants)
        for risk in ("supplied", "fitted"):
            left, right = (data["left"], data["right"]) if risk == "supplied" else (data["fitted_left"], data["fitted_right"])
            for method in ("pcgc", "pcgc-inverse"):
                row = dict(seed=seed, scenario=args.scenario, risk=risk, method=method,
                           truth_covariance=data["truth_covariance"], truth_rg=data["truth_rg"], diagnostics=data["diagnostics"])
                try:
                    if isinstance(left, str) or isinstance(right, str):
                        raise ValueError(left if isinstance(left, str) else right)
                    pair = exact_pair(data["x"], a, left, right, data["left_rows"], data["right_rows"], method=method)
                    row.update(fit_pair(pair, block_ids=blocks))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    row["failure"] = str(exc)
                rows.append(row)
        print(json.dumps(dict(scenario=args.scenario, replicate=replicate, seconds=time.perf_counter()-start)), flush=True)
    (args.out/"replicates.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
