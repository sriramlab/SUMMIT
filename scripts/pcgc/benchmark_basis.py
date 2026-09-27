#!/usr/bin/env python3
"""Measure the exact rank-one contraction against the full shared-basis path."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from summit.ldscore.generalized_gxe_pass1 import ArraySequentialGenotypeOperator
from summit.pcgc.reference import prepare_moments, generalized_reference, contract_reference
from summit.sumstats.binary import prepare_binary_risk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ctypes.CDLL(None).prctl(41,1,0,0,0)
    rng = np.random.default_rng(841923)
    n, m, q = 2000, 1600, 4
    x, a = rng.normal(size=(n,m)), np.ones((m,1))
    group = np.arange(n) % q
    risk = prepare_binary_risk(np.arange(n) % 3 == 0, .1, population_risk=np.array([.02,.05,.1,.2])[group])
    phi, coef = np.eye(q)[group], risk.sensitivity[:q]
    opts = dict(probes=64, seed=4638, block_size=256, native=False)
    results = []
    for repeat in range(3):
        t = time.perf_counter()
        ref, scored, plan = generalized_reference(ArraySequentialGenotypeOperator(x), a, phi,
                                                   responses=(risk.sensitivity*risk.z)[:,None], **opts)
        ld, sp = contract_reference(ref, coef)
        full_seconds = time.perf_counter()-t
        t = time.perf_counter()
        moments, diagnostics = prepare_moments(ArraySequentialGenotypeOperator(x), a, risk, 'pcgc-basis',
                                                basis=phi, coefficients=coef, **opts)
        collapsed_seconds = time.perf_counter()-t
        full_H = n*n*(a.T @ ld)/(m*m)-sp
        error = float(np.linalg.norm(full_H-moments.equations()[0])/np.linalg.norm(full_H))
        if error > 1e-12:
            raise RuntimeError('basis contraction parity failed')
        results.append(dict(full_basis_seconds=full_seconds, collapsed_seconds=collapsed_seconds,
                            relative_matrix_error=error, full_basis_planned_bytes=plan.peak_resident_bytes,
                            collapsed_planned_bytes=diagnostics['peak_planned_total_bytes']))
    root = Path(__file__).resolve().parents[2]
    output = dict(samples=n, variants=m, basis=q, probes=64, backend='NumPy controlled comparison', repeats=results,
                  median_speedup=float(np.median([r['full_basis_seconds'] for r in results])/
                                       np.median([r['collapsed_seconds'] for r in results])),
                  source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__).resolve(), root/'src/summit/pcgc/reference.py']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
