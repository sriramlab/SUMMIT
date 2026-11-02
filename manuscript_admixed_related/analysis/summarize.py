"""Summary of the simulations: every TASK/results.tsv of simulate.py and TASK/methods_*.tsv of compare_methods.py.

Per cohort, setting, analysis (arm) and model: mean, bias with its Monte Carlo SE, SD across replicates, mean SE, SE/SD,
coverage of 95% t intervals (199 degrees of freedom for the 200-block jackknife) and the variance of the unrelated-subset
estimate divided by that of the analysis (single-component unrelated subset for REML). For the 'ancestry' designs bias is
also given against the target after adjustment for the environment terms (columns *_adjusted). Genetic-correlation settings
also report rg. Tables A1/A2 pool the main grid by cohort-free averages over its settings without and with shared effects;
their variance ratio is the sum of the unrelated-subset variances over settings divided by the sum for the analysis.
Writes OUT/replicates.tsv, OUT/summary.tsv, OUT/table_grid.tsv.
usage: python summarize.py OUT TASK [TASK ...]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

out, tasks = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
T = stats.t.ppf(0.975, 199)
files = [f for t in tasks for f in [f'{t}/results.tsv'] + sorted(glob.glob(f'{t}/methods_*.tsv')) if os.path.exists(f)]
R = pd.concat([pd.read_csv(f, sep='\t') for f in files], ignore_index=True)
R.to_csv(f'{out}/replicates.tsv', sep='\t', index=False, float_format='%.8g')
key = ['cohort', 'setting', 'design', 'label', 'arm', 'model']
rows = []
for k, d in R.groupby(key, sort=True):
    row = dict(zip(key, k), n=len(d))
    for suffix, truth in (('', d.truth), ('_adjusted', d.get('truth_adjusted'))):
        if truth is None or truth.isna().all():
            continue
        err = d.est - truth
        row.update({f'bias{suffix}': err.mean(), f'mcse{suffix}': err.std(ddof=1) / np.sqrt(len(d)),
                    f'coverage{suffix}': float((err.abs() <= T * d.se).mean())})
    row.update(mean=d.est.mean(), sd=d.est.std(ddof=1), mean_se=d.se.mean(), se_over_sd=d.se.mean() / d.est.std(ddof=1))
    if 'rg' in d and d.rg.notna().any():
        row.update(rg_mean=d.rg.mean(), rg_bias=(d.rg - d.rg_truth).mean(),
                   rg_coverage=float(((d.rg - d.rg_truth).abs() <= T * d.rg_se).mean()))
    rows.append(row)
S = pd.DataFrame(rows)
u = S[S.arm == 'U'].set_index(['cohort', 'setting', 'model']).sd ** 2
ref_model = np.where(S.arm == 'REML', 'single', S.model)
S['var_ratio'] = u.reindex(list(zip(S.cohort, S.setting, ref_model))).to_numpy() / S.sd ** 2
S.to_csv(f'{out}/summary.tsv', sep='\t', index=False, float_format='%.5g')

grid = S[S.design == 'grid'].copy()
grid['shared'] = grid.label.str.endswith('shared')
grid['var_u'] = u.reindex(list(zip(grid.cohort, grid.setting, np.where(grid.arm == 'REML', 'single', grid.model)))).to_numpy()
grid['var_arm'] = grid.sd ** 2
G = grid.groupby(['model', 'arm', 'shared']).agg(settings=('setting', 'size'), bias=('bias', 'mean'),
                                                  coverage=('coverage', 'mean'), var_u=('var_u', 'sum'), var_arm=('var_arm', 'sum'))
G['var_ratio'] = G.var_u / G.var_arm
G.drop(columns=['var_u', 'var_arm']).reset_index().to_csv(f'{out}/table_grid.tsv', sep='\t', index=False, float_format='%.4g')
print(G.drop(columns=['var_u', 'var_arm']).round(3).to_string())
