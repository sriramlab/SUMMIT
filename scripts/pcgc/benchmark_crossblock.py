"""Wall time and planned workspace of preparation with and without cross-block rows.

Uses SUMMIT's in-memory genotype operator and the production two-pass
preparation; alternates the order of the two variants across repeats.
"""
import argparse
import json
import time
import numpy as np
from scipy.special import ndtr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--variants", type=int, default=4000)
    parser.add_argument("--contexts", type=int, default=3)
    parser.add_argument("--annotations", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=200)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    from summit.pcgc.gxe import prepare_gxe_moments, fit_gxe
    from summit.pcgc.crossblock import contiguous_blocks
    from summit.sumstats.binary import fit_binary_risk
    from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator as Operator
    rng = np.random.default_rng(5)
    n, m, q = args.samples, args.variants, args.contexts
    x = rng.normal(size=(n, m))
    env = rng.uniform(-1, 1, (n, q-1))
    phi = np.column_stack((np.ones(n), env))
    y = (rng.uniform(size=n) < ndtr(-1+.3*env[:, 0])).astype(float)
    risk = fit_binary_risk(y, .1, env)
    a = np.column_stack([np.ones(m)]+[(np.arange(m) % args.annotations == j).astype(float) for j in range(1, args.annotations)])
    common = dict(liability_sd=1., probes=args.probes, seed=1, block_size=256, threads=args.threads, native=True,
                  memory_bytes=8*2**30)
    rows = []
    for repeat in range(args.repeats):
        for variant in (("plain", "crossblock") if repeat % 2 == 0 else ("crossblock", "plain")):
            extra = dict(crossblock_blocks=args.blocks, risk_covariates=env) if variant == "crossblock" else {}
            started = time.perf_counter()
            moments, diagnostics = prepare_gxe_moments(Operator(x), a, risk, phi, **common, **extra)
            prepared = time.perf_counter()-started
            started = time.perf_counter()
            fit = fit_gxe(moments, block_ids=contiguous_blocks(m, args.blocks))
            rows.append(dict(variant=variant, repeat=repeat, preparation_seconds=prepared,
                             fit_seconds=time.perf_counter()-started,
                             planned_bytes=diagnostics["peak_planned_workspace_bytes"],
                             method=fit["uncertainty_method"]))
            print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for variant in ("plain", "crossblock"):
        values = [r for r in rows if r["variant"] == variant]
        summary[variant] = dict(median_preparation_seconds=float(np.median([r["preparation_seconds"] for r in values])),
                                median_fit_seconds=float(np.median([r["fit_seconds"] for r in values])))
    print(json.dumps(dict(args=vars(args), summary=summary)))


if __name__ == "__main__":
    main()
