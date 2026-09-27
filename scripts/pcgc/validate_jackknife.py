#!/usr/bin/env python3
"""Paired audit of old/new SNP deletion equations on existing simulation seeds.

No new architecture grid, genotype dumps, or reference Monte Carlo noise.
The old equations are reproduced only here, never exposed by inference.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from validate_pcgc import generate, SCENARIOS
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk
from summit.pcgc.research import exact_moments
from summit.pcgc.moments import fit_moments


def legacy_fit(moments, diagonal_rows, ids):
    a, n = moments.annotations, moments.n_samples
    full_ld = moments.ldscores + diagonal_rows/n**2
    loo = []
    for block in np.unique(ids):
        keep = ids != block
        mass = a[keep].sum(axis=0)
        directed = a[keep].T @ full_ld[keep]
        H = n*n*(directed+directed.T)/(2*np.outer(mass, mass))-moments.same_person
        loo.append(np.linalg.solve(H, a[keep].T @ moments.rhs_rows[keep]/mass))
    loo = np.asarray(loo).sum(axis=1)
    return float(np.sqrt((len(loo)-1)*np.var(loo)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenarios', nargs='+', default=['S3', 'S4', 'S7'], choices=SCENARIOS)
    parser.add_argument('--replicates', type=int, default=10)
    parser.add_argument('--seed-offset', type=int, default=20)
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--variants', type=int, default=4000)
    parser.add_argument('--blocks', type=int, nargs='+', default=[50, 100])
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100 or not 0 <= args.seed_offset <= 100-args.replicates:
        parser.error('at most 100 original seeds per scenario')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError('process-local THP guard failed')
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name('validate_pcgc.py'),
               root/'src/summit/sumstats/binary.py', *sorted((root/'src/summit/pcgc').glob('*.py'))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    source_hashes=hashes, seed_rule='820000 + 1000*scenario_number + replicate_index',
                    affinity=sorted(os.sched_getaffinity(0)), thp_disabled=True,
                    conditioning='same datasets as original confirmation; exact LD; fitted risks held fixed in SNP deletions')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    records = []
    for scenario in args.scenarios:
        for index in range(args.seed_offset, args.seed_offset+args.replicates):
            start = time.perf_counter()
            seed = 820000+1000*int(scenario[1:])+index
            data = generate(seed, scenario, args.samples, args.variants, 1, 40)
            x, a = data['x'], data['annotations']
            risks = dict(supplied=prepare_binary_risk(data['y'], SCENARIOS[scenario]['K'],
                                                     population_risk=data['k'], covariate_variance=data['Vc']),
                         fitted=fit_binary_risk(data['y'], SCENARIOS[scenario]['K'],
                                                data['cov'][:, None] if data['gamma'] else None))
            for name, risk in risks.items():
                moments = exact_moments(x, a, risk)
                squares = x*x*risk.sensitivity[:, None]**2
                diag = squares.T @ (squares @ a)
                for count in args.blocks:
                    ids = np.arange(len(a))*count//len(a)
                    fit = fit_moments(moments, block_ids=ids)
                    fit.pop('jackknife_replicates')
                    fit.update(scenario=scenario, seed=seed, risk=name, blocks=count,
                               truth=float(data['truth'].sum()/(1+data['Vc'])),
                               old_marginal_se=legacy_fit(moments, diag, ids)/(1+risk.covariate_variance),
                               null_pair_variance_se=float(np.sqrt(2*np.linalg.inv(moments.equations()[0]).sum())/(1+risk.covariate_variance)),
                               covariate_variance=risk.covariate_variance)
                    records.append(fit)
            print(json.dumps(dict(scenario=scenario, index=index, seconds=time.perf_counter()-start)), flush=True)
    # Hashes are captured BEFORE work; do not edit this driver while it runs.
    (args.out/'replicates.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    summary = []
    for scenario in args.scenarios:
        for risk in ('supplied', 'fitted'):
            for blocks in args.blocks:
                rows = [r for r in records if (r['scenario'], r['risk'], r['blocks']) == (scenario,risk,blocks)]
                est = np.array([r['marginal_total'] for r in rows])
                err = est-np.array([r['truth'] for r in rows])
                old = np.array([r['old_marginal_se'] for r in rows])
                new = np.array([r['marginal_total_standard_error'] for r in rows])
                sd = est.std(ddof=1) if len(rows)>1 else None
                summary.append(dict(scenario=scenario, risk=risk, blocks=blocks, replicates=len(rows),
                    mean=float(est.mean()), sd=sd, mean_se_multiplier=float(np.mean(new/old)),
                    old_rms_se_sd=float(np.sqrt(np.mean(old**2))/sd) if sd else None,
                    new_rms_se_sd=float(np.sqrt(np.mean(new**2))/sd) if sd else None,
                    old_coverage=int(np.sum(np.abs(err)<=1.96*old)), new_coverage=int(np.sum(np.abs(err)<=1.96*new))))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
