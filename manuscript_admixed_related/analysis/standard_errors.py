"""Factors of the long-range LD correction of jackknife standard errors (Table E, LD-score columns), per cohort, jackknife
design (200 blocks or chromosomes) and heritability, from `summit relatives longrange` sums with 8 MAF-LD bins.

With weights w of the bin estimates in total h2 (from the bin-by-bin LD-score sums), rho the between-group share of the
LD-score mass, rho_s its part from each individual paired with themselves, f_g the groups' shares and n the sample size,
the corrected jackknife variance is V_jk / (1 - rho_s - (rho - rho_s)(1 + 2 h2)) + 4 h2^2 (1 - sum f_g^2) / n; relative to
the leading-order variance 2 w'Lw this gives the factor multiplying the jackknife SE.
usage: python standard_errors.py OUT LONGRANGE_JSON [LONGRANGE_JSON ...]      (writes OUT/se_factors.tsv)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

out, files = sys.argv[1], sys.argv[2:]
rows = []
for f in files:
    lr = json.load(open(f))
    L, T = np.array(lr['L']['self']), np.array(lr['T']['self'])
    w = np.linalg.solve(T.T, np.ones(len(L)))
    v0, n = 2 * w @ L @ w, lr['n']
    for design, label in (('blocks', '200 blocks'), ('chr', 'chromosomes')):
        d = lr['designs'][design]
        rho_s = max(d['rho']['self'] - d['rho']['n'], 0)
        rho_l = max(d['rho']['self'] - rho_s, 0)
        g = np.array(d['group_snps'], float)
        g = g[g > 0] / g.sum()
        row = dict(cohort=os.path.basename(f).split('.')[0], n=n, design=label, rho=d['rho']['self'], rho_self=rho_s)
        for h2 in (0.25, 0.5):
            row[f'factor_h2_{h2}'] = np.sqrt(1 / (1 - rho_s - rho_l * (1 + 2 * h2)) + 4 * h2 ** 2 * (1 - (g ** 2).sum()) / n / v0)
        rows.append(row)
os.makedirs(out, exist_ok=True)
pd.DataFrame(rows).to_csv(f'{out}/se_factors.tsv', sep='\t', index=False, float_format='%.4g')
print(pd.DataFrame(rows).round(3).to_string(index=False))
