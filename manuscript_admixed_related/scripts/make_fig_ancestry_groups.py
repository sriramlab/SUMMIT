"""Figure 7 (Appendix G) of "Heritability from summary statistics of admixed and related biobank participants": European and
African-ancestry participants in one analysis. A, genetic correlation between the European group and all participants of
African ancestry for each trait whose heritability exceeds twice its standard error in both groups (95% intervals), with
origin covariates; B, heritability from one kernel for all pairs against the size-weighted mean of the two groups'
estimates; C, genetic correlation in the simulations by scenario (truth as dashed lines).
Reads ../data/ancestry_groups_traits.tsv and ../data/ancestry_groups_simulations.tsv.gz; writes OUT/fig_pairgroups.{pdf,png}
and appends its summaries to OUT/numbers_fig4.txt.
usage: python make_fig_ancestry_groups.py [--out DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(HERE, '..', 'output'))
OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'savefig.dpi': 300, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
LABEL = {'albumin': 'Albumin', 'alka_phos': 'ALP', 'alanine_aminotransferase': 'ALT', 'asp_at': 'AST', 'creatinine': 'Creatinine',
         'cystatin_c': 'Cystatin C', 'ggt': 'GGT', 'glucose': 'Glucose', 'hba1c': 'HbA1c', 'hdl': 'HDL', 'igf_1': 'IGF-1',
         'ldl': 'LDL', 'testosterone': 'Testosterone', 'triglycerides': 'Triglycerides', 'urate': 'Urate',
         'vitamin_d': 'Vitamin D', 'blood_rbc_count': 'RBC count', 'white_blood_cell_count': 'WBC count',
         'blood_platelet': 'Platelets', 'blood_mscv': 'MSCV', 'haemoglobin_concentration': 'Hemoglobin',
         'height': 'Height', 'bmi': 'BMI', 'bp_systolic': 'SBP', 'diastolic_blood_pressure': 'DBP',
         'heel_bone_mineral_density': 'Heel BMD', 'calcium': 'Calcium', 'urea': 'Urea', 'apo_b': 'ApoB',
         'fluid_intelligence_score': 'Fluid intelligence', 'neuroticism_score': 'Neuroticism', 'sleep_duration': 'Sleep duration'}

real = pd.read_csv(f'{DATA}/ancestry_groups_traits.tsv', sep='\t', index_col=0)
allsims = pd.read_csv(f'{DATA}/ancestry_groups_simulations.tsv.gz', sep='\t')
raw = allsims[allsims.scenario == 'rho05_t3_raw']
sims = allsims[allsims.scenario != 'rho05_t3_raw']

fig = plt.figure(figsize=(7.2, 2.9))
axA = fig.add_axes([0.08, 0.22, 0.30, 0.70])
axB = fig.add_axes([0.47, 0.22, 0.19, 0.70])
axC = fig.add_axes([0.75, 0.22, 0.24, 0.70])

ok = (real.h2_EUR > 2 * real.h2_EUR_se) & (real.h2_AFR > 2 * real.h2_AFR_se) & np.isfinite(real.rg)
d = real[ok].sort_values('rg')
y = np.arange(len(d))
axA.errorbar(d.rg, y, xerr=1.96 * d.rg_se, fmt='o', ms=3.2, mfc='tab:blue', mec='k', mew=0.5, ecolor='0.55', elinewidth=0.8,
             capsize=0)
axA.axvline(1, color='0.6', lw=0.6, ls='--'); axA.axvline(np.median(d.rg), color='tab:blue', lw=0.7)
axA.set_yticks(y); axA.set_yticklabels([LABEL.get(t, t) for t in d.index], fontsize=6)
axA.set_xlabel('Genetic correlation')
axA.set_xlim(-0.6, 1.8)

wmean = (real.h2_EUR * real.nE + real.h2_AFR * real.nA) / (real.nE + real.nA)
axB.plot(wmean, real.h2_pooled, 'o', ms=3, mfc='tab:orange', mec='k', mew=0.5)
lim = [0, max(wmean.max(), real.h2_pooled.max()) * 1.05]
axB.plot(lim, lim, color='0.6', lw=0.6, ls='--')
axB.set_xlim(lim); axB.set_ylim(lim)
axB.set_xlabel('Size-weighted mean of\ngroup estimates'); axB.set_ylabel('One kernel for all pairs')

SC = [('rho1', 'rg', 1.0, 'Shared effects'), ('rho1_h2', 'rg', 1.0, 'Shared, unequal $h^2$'),
      ('rho05', 'rg', 0.5, 'Correlation 0.5'), ('rho05_la', 'rg', 0.5, 'Local ancestry'),
      ('rho05_la', 'rg_snponly', 0.5, 'Same, SNPs only'), ('rho05_t3', 'rg', 0.5, 'Heavy tails, rank-normal'),
      ('raw', 'rg', 0.5, 'Heavy tails, untransformed')]
vals, labs, truth = [], [], []
for sc, col, t, lab in SC:
    src = raw if sc == 'raw' else sims[sims.scenario == sc]
    v = src[col].to_numpy(float); v = v[np.isfinite(v)]
    vals.append(v); labs.append(lab); truth.append(t)
bp = axC.boxplot(vals, positions=range(len(vals)), widths=0.55, showfliers=False, patch_artist=True,
                 medianprops=dict(color='k', lw=0.8), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                 capprops=dict(lw=0.6))
for patch in bp['boxes']:
    patch.set_facecolor('tab:blue'); patch.set_alpha(0.55)
for i, t in enumerate(truth):
    axC.plot([i - 0.35, i + 0.35], [t, t], color='tab:red', lw=0.9, ls='--')
axC.set_xticks(range(len(vals))); axC.set_xticklabels(labs, rotation=50, ha='right', fontsize=6, rotation_mode='anchor')
axC.set_ylabel('Genetic correlation')
for ax, s in ((axA, 'A'), (axB, 'B'), (axC, 'C')):
    fig.text(ax.get_position().x0 - (0.07 if ax is axA else 0.055), 0.95, s, fontsize=10, fontweight='bold', va='bottom')
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_pairgroups.{ext}', bbox_inches='tight')

below = (real.h2_pooled < wmean)
with open(f'{OUT}/numbers_fig4.txt', 'w') as f:
    f.write(f'Fig4A {int(ok.sum())} traits, median rg {np.median(d.rg):.3f}, upper 95% limit below 1: '
            f'{int((d.rg + 1.96 * d.rg_se < 1).sum())}\n')
    f.write(f'Fig4B one kernel below the size-weighted mean for {int(below.sum())} of {int(wmean.notna().sum())} traits, '
            f'median difference {np.nanmedian(real.h2_pooled - wmean):+.4f}\n')
    for (sc, col, t, lab), v in zip(SC, vals):
        f.write(f'Fig4C {lab}: mean {v.mean():.3f}, SD {v.std(ddof=1):.3f} (truth {t})\n')
print('written')
