"""Heritability of each trait in a cohort against a reference cohort (Fig. 4A): z = (h2 - h2_ref) / sqrt(se^2 + se_ref^2),
two-sided p, Bonferroni and Benjamini-Hochberg adjustment over the traits, for the analysis and model given.
Inputs are real_h2.tsv files of real_traits.py (the reference, e.g. unrelated white British participants, analysis U).
usage: python compare_groups.py COHORT_REAL_H2 REFERENCE_REAL_H2 OUT [--arm XA] [--reference-arm U] [--model 8bin]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from genetic_correlation import bh

ap = argparse.ArgumentParser()
ap.add_argument('cohort')
ap.add_argument('reference')
ap.add_argument('out')
ap.add_argument('--arm', default='XA')
ap.add_argument('--reference-arm', default='U')
ap.add_argument('--model', default='8bin')
a = ap.parse_args()
pick = lambda f, arm: pd.read_csv(f, sep='\t').query('arm == @arm and model == @a.model').set_index('trait')[['h2', 'se']]
d = pick(a.cohort, a.arm).join(pick(a.reference, a.reference_arm), rsuffix='_ref', how='inner')
d['z'] = (d.h2 - d.h2_ref) / np.sqrt(d.se ** 2 + d.se_ref ** 2)
d['p'] = 2 * norm.sf(d.z.abs())
d['bonferroni'] = np.minimum(d.p * len(d), 1)
d['fdr'] = bh(d.p.to_numpy())
d.reset_index().to_csv(a.out, sep='\t', index=False, float_format='%.4g')
print(f'{len(d)} traits; Bonferroni < 0.05: {(d.bonferroni < 0.05).sum()}; FDR < 0.05: {(d.fdr < 0.05).sum()}')
