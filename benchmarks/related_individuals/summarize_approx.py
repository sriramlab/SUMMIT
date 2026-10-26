"""Compare exact and approximate related-pair parts (run from ~/summit_relatives on veena).

For each ancestry group and scenario: mean and SD of h2, mean jackknife SE, SE/SD, coverage of 95% t intervals, and the
replicate-wise ratio of the approximate SE to the exact SE (the estimates themselves should agree to rounding, because both
approximations keep sum_j V_j REL_M_j). Same for the genetic covariance and rg.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

POPS = ['EUR', 'SAS', 'AFR']
T = stats.t.ppf(0.975, 199)
ARMS = ['pairexcl', 'pairexcl_A1', 'pairexcl_A2']


def load_h2():
    rows = []
    for pop in POPS:
        meta = pd.read_csv(f'final/{pop}/meta.tsv', sep='\t')
        meta['phen'] = 'y' + meta['col'].astype(str)
        for arm in ARMS:
            for f in glob.glob(f'final/{pop}/fit/{arm}_200_*.results.tsv'):
                if not re.fullmatch(rf'{arm}_200_\d+\.results\.tsv', os.path.basename(f)):
                    continue
                d = pd.read_csv(f, sep='\t')[['phen', 'h2', 'h2_se']]
                d['phen'] = d['phen'].astype(str).str.replace(r'\.tsv(\.gz)?$', '', regex=True)
                rows.append(d.assign(pop=pop, arm=arm).merge(meta[['phen', 'scenario', 'rep', 'h2g']], on='phen'))
    return pd.concat(rows)


def load_rg():
    pat = re.compile(r'Total genetic correlation \(rg\): (\S+) \(SE: (\S+)\)')
    patg = re.compile(r'Total genetic covariance \(gamma_g\): (\S+) \(SE: (\S+)\)')
    rows = []
    for pop in POPS:
        for arm in ARMS:
            for f in glob.glob(f'final/{pop}/rg/fit/{arm}_200_*.log'):
                m = re.fullmatch(rf'{arm}_200_(\d+)\.log', os.path.basename(f))
                if not m:
                    continue
                txt = open(f).read()
                a, g = pat.search(txt), patg.search(txt)
                if a and g:
                    rows.append(dict(pop=pop, arm=arm, pair=int(m.group(1)), rg=float(a.group(1)), se=float(a.group(2)),
                                     gcov=float(g.group(1)), gse=float(g.group(2))))
    return pd.DataFrame(rows)


def main():
    d = load_h2()
    out = []
    for (pop, sc), g in d.groupby(['pop', 'scenario']):
        w = g.pivot_table(index='rep', columns='arm', values=['h2', 'h2_se'])
        truth = float(g.h2g.iloc[0])
        for arm in ARMS:
            if ('h2', arm) not in w.columns:
                continue
            h, se = w[('h2', arm)], w[('h2_se', arm)]
            out.append(dict(pop=pop, scenario=sc, arm=arm, n=int(h.notna().sum()), bias=h.mean() - truth, sd=h.std(ddof=1),
                            mean_se=se.mean(), se_over_sd=se.mean() / h.std(ddof=1),
                            coverage=float(np.mean(np.abs(h - truth) <= T * se)),
                            max_abs_diff_vs_exact=float(np.max(np.abs(h - w[('h2', 'pairexcl')]))),
                            se_ratio_vs_exact=float(np.mean(se / w[('h2_se', 'pairexcl')]))))
    s = pd.DataFrame(out)
    pd.set_option('display.width', 220)
    print(s.round(4).to_string(index=False))
    s.to_csv('final/approx_h2_summary.tsv', sep='\t', index=False, float_format='%.5f')
    r = load_rg()
    if len(r):
        out = []
        for pop, g in r.groupby('pop'):
            w = g.pivot_table(index='pair', columns='arm', values=['rg', 'se', 'gcov', 'gse'])
            for arm in ARMS:
                if ('gcov', arm) not in w.columns:
                    continue
                out.append(dict(pop=pop, arm=arm, n=int(w[('gcov', arm)].notna().sum()),
                                gcov_bias=w[('gcov', arm)].mean() - 0.125,
                                gcov_coverage=float(np.mean(np.abs(w[('gcov', arm)] - 0.125) <= T * w[('gse', arm)])),
                                gse_ratio_vs_exact=float(np.mean(w[('gse', arm)] / w[('gse', 'pairexcl')])),
                                max_abs_gcov_diff=float(np.max(np.abs(w[('gcov', arm)] - w[('gcov', 'pairexcl')]))),
                                rg_mean=w[('rg', arm)].mean(), rg_sd=w[('rg', arm)].std(ddof=1),
                                rg_coverage=float(np.mean(np.abs(w[('rg', arm)] - 0.5) <= T * w[('se', arm)]))))
        t = pd.DataFrame(out)
        print(t.round(4).to_string(index=False))
        t.to_csv('final/approx_rg_summary.tsv', sep='\t', index=False, float_format='%.5f')


if __name__ == '__main__':
    main()
