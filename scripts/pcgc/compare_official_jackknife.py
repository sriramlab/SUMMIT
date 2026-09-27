#!/usr/bin/env python3
"""Paired, bounded audit against pinned upstream S-PCGC jackknife code.

The upstream function is fetched, SHA256 checked, and isolated from its CLI.
Its reference input is adapted to match SUMMIT's FULL normal equations. This
isolates deletion rules; it is not a claim of end-to-end S-PCGC equivalence.
No participant-level data or genotype matrices are written.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pandas as pd

from validate_pcgc import generate, SCENARIOS
from summit.pcgc.moments import fit_moments
from summit.pcgc.research import exact_moments
from summit.sumstats.binary import prepare_binary_risk, fit_binary_risk

UPSTREAM = 'https://raw.githubusercontent.com/omerwe/S-PCGC/5211d173de45c6a88151928892581c058ba593cd/pcgc_main.py'
UPSTREAM_SHA = 'f081d2a3d9f3b65b6a34e0a98be0f6321b83c4a7d0814e3939ce0f453bd08e57'


def upstream_function():
    source = urlopen(UPSTREAM, timeout=30).read()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA:
        raise RuntimeError('upstream code identity changed')
    cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == 'SPCGC')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'compute_taus')
    namespace = dict(np=np, logging=logging)
    exec(compile(ast.Module(body=[method], type_ignores=[]), UPSTREAM, 'exec'), namespace)
    return namespace['compute_taus']


def official_deletions(function, x, a, risk, moments, blocks):
    n, m = x.shape
    masses = a.sum(axis=0)
    names = [str(j) for j in range(a.shape[1])]
    v = risk.sensitivity*risk.z
    scores = x.T @ v
    diagonal = (x*x) @ a * v[:, None]**2 / masses
    q = float(np.mean(risk.sensitivity**2))
    obj = SimpleNamespace(df_Gty=pd.DataFrame(np.sqrt(diagonal), columns=names),
                          df_sumstats=pd.DataFrame({'pcgc_sumstat': scores}),
                          trace_ratios=np.ones(len(names)), N=n, mean_Q=q, deflation_ratio=1.)
    H, _ = moments.equations()
    prodr2 = pd.DataFrame(H*np.outer(masses, masses)/(n*n*q*q), columns=names, index=names)
    # Upstream squares annotation weights in its estimating instruments.
    # Supplying sqrt(A) gives exactly SUMMIT's A-weighted score numerator.
    annot = pd.DataFrame(np.sqrt(a), columns=names)
    sync = pd.DataFrame({'min_annot': np.zeros(len(names))}, index=names)
    coef, deleted, _, _ = function(None, SimpleNamespace(fit_intercept=False, n_blocks=blocks),
                                  obj, obj, annot, prodr2, sync, masses, None, None)
    # Independent translation of upstream's fixed-H/fixed-intercept rule.
    expected = []
    separators = np.floor(np.linspace(0, m, blocks+1)).astype(int)
    total = a.T @ (scores*scores)
    for lo, hi in zip(separators[:-1], separators[1:]):
        retained = masses-a[lo:hi].sum(axis=0)
        b = (total-a[lo:hi].T @ (scores[lo:hi]**2))/retained-diagonal.sum(axis=0)
        expected.append(np.linalg.solve(H, b))
    deleted = deleted*masses
    np.testing.assert_allclose(deleted, expected, rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(coef*masses, fit_moments(moments)['conditional_components'], rtol=2e-10, atol=1e-11)
    return deleted


def null_expected_variance(x, a, risk, blocks):
    """Exact independent-Bernoulli null variance, conditional on X and risks.

    Also compute E[SNP-JK covariance] analytically, with no simulated outcomes.
    This diagnostic deliberately does not assert fixed-case-count calibration.
    """
    n, m = x.shape
    f = x*risk.sensitivity[:, None]
    f2 = f*f
    U = (f.T @ f)**2-f2.T @ f2
    mass = a.sum(axis=0)
    # theta = C @ t, where Cov(t)=2U at the independent Bernoulli null.
    C = np.linalg.solve((a.T @ U @ a)/mass[None, :], a.T)
    true = 2*C @ U @ C.T
    selectors = []
    ids = np.arange(m)*blocks//m
    for label in range(blocks):
        target = a.copy()
        target[ids == label] = 0
        selectors.append(np.linalg.solve((target.T @ U @ a)/mass[None, :], target.T))
    selectors = np.asarray(selectors)
    selectors -= selectors.mean(axis=0)
    expected = sum(2*c @ U @ c.T for c in selectors)*(blocks-1)/blocks
    return dict(true_covariance=true.tolist(), expected_jackknife_covariance=expected.tolist(),
                total_variance_ratio=float(expected.sum()/true.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=['S4', 'S7'])
    parser.add_argument('--replicates', type=int, default=80)
    parser.add_argument('--seed-offset', type=int, default=20)
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--variants', type=int, default=4000)
    parser.add_argument('--blocks', nargs='+', type=int, default=[50, 100])
    parser.add_argument('--null-analytic', action='store_true', help='Small-design conditional check, first supplied-risk S7 dataset only.')
    args = parser.parse_args()
    if not 1 <= args.replicates <= 100 or not 0 <= args.seed_offset <= 100-args.replicates:
        parser.error('at most 100 original seeds per scenario')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError('process-local THP guard failed')
    function = upstream_function()
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name('validate_pcgc.py'),
               root/'src/summit/sumstats/binary.py', root/'src/summit/pcgc/moments.py',
               root/'src/summit/pcgc/research.py']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    source_hashes=hashes, upstream_url=UPSTREAM, upstream_sha256=UPSTREAM_SHA,
                    reference='matched full equations, not upstream reference estimator',
                    affinity=sorted(os.sched_getaffinity(0)), thp_disabled=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows, analytic = [], []
    for scenario in args.scenarios:
        for index in range(args.seed_offset, args.seed_offset+args.replicates):
            start = time.perf_counter()
            seed = 820000+1000*int(scenario[1:])+index
            data = generate(seed, scenario, args.samples, args.variants, 1, 40)
            x, a = data['x'], data['annotations']
            risks = dict(supplied=prepare_binary_risk(data['y'], SCENARIOS[scenario]['K'], population_risk=data['k'], covariate_variance=data['Vc']),
                         fitted=fit_binary_risk(data['y'], SCENARIOS[scenario]['K'], data['cov'][:,None] if data['gamma'] else None))
            for name, risk in risks.items():
                moments = exact_moments(x, a, risk)
                for blocks in args.blocks:
                    fit = fit_moments(moments, block_ids=np.arange(len(a))*blocks//len(a))
                    deleted = official_deletions(function, x, a, risk, moments, blocks)
                    official_se = np.sqrt((blocks-1)*np.var(deleted.sum(axis=1)))/(1+risk.covariate_variance)
                    rows.append(dict(scenario=scenario, seed=seed, risk=name, blocks=blocks,
                                     estimate=fit['marginal_total'], truth=float(data['truth'].sum()/(1+data['Vc'])),
                                     summit_se=fit['marginal_total_standard_error'], official_se=float(official_se),
                                     component_estimates=fit['marginal_components'],
                                     summit_component_se=fit['marginal_standard_errors'],
                                     official_component_se=(np.sqrt((blocks-1)*np.var(deleted,axis=0))/(1+risk.covariate_variance)).tolist()))
                    if args.null_analytic and scenario == 'S7' and index == args.seed_offset and name == 'supplied':
                        analytic.append(dict(seed=seed, blocks=blocks, **null_expected_variance(x,a,risk,blocks)))
            print(json.dumps(dict(scenario=scenario, index=index, seconds=time.perf_counter()-start)), flush=True)
    (args.out/'replicates.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    (args.out/'analytic.json').write_text(json.dumps(analytic, indent=2, allow_nan=False)+'\n')
    summary=[]
    for key in sorted({(r['scenario'],r['risk'],r['blocks']) for r in rows}):
        group=[r for r in rows if (r['scenario'],r['risk'],r['blocks'])==key]
        est=np.array([r['estimate'] for r in group]); error=est-np.array([r['truth'] for r in group])
        item=dict(zip(('scenario','risk','blocks'),key), replicates=len(group), mean=float(est.mean()), sd=float(est.std(ddof=1)) if len(group)>1 else None)
        for method in ('summit','official'):
            se=np.array([r[method+'_se'] for r in group])
            item[method]=dict(rms_se=float(np.sqrt(np.mean(se*se))), rms_se_sd=float(np.sqrt(np.mean(se*se))/est.std(ddof=1)) if len(group)>1 else None,
                              coverage=int(np.sum(np.abs(error)<=1.96*se)))
        summary.append(item)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
