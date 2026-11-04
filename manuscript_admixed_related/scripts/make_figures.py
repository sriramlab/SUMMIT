"""Figures 2, 5 and 6 of "Heritability from summary statistics of admixed and related biobank participants", from the
aggregate results in ../data. Every heritability result uses 8 MAF-LD bins unless a panel says otherwise.

Figure 2   simulations: bias and coverage of the methods that keep relatives, precision against the unrelated subset,
           local ancestry, and precision of the wider African-ancestry group.
Figure 5   genetic correlation in all participants of the European group against its unrelated subset.
Figure 6   heritability due to direct genetic effects (simulations and 40 traits in the siblings).
Figures 1, 3, 4 and 7 are drawn by make_fig1.py, make_fig34.py and make_fig_ancestry_groups.py.
OUT/numbers.txt lists every plotted summary.
usage: python make_figures.py [--out DIR]
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

import argparse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
OUT = ap.parse_args().out
SRC = {k: os.path.join(DATA, v) for k, v in dict(
    grid='simulations.tsv.gz', reml='simulations_reml.tsv.gz', covldsc='simulations_covldsc.tsv.gz',
    la_sims='local_ancestry_simulations.tsv.gz', wide='wider_group_simulations.tsv.gz',
    sib_sim='direct_effects_simulations.tsv.gz', traits40='cohort_traits.tsv', cohort_tests='cohort_tests.tsv',
    la_real='local_ancestry_traits.tsv', sib32='direct_effects_traits.tsv', sib_extra='direct_effects_traits_extra.tsv',
    sib_extra2='direct_effects_traits_extra2.tsv', rg='genetic_correlation_cohorts.tsv.gz',
    rg_ref='genetic_correlation_reference.tsv').items()}
SRC['ukb406'] = [os.path.join(DATA, 'white_british_traits.tsv')]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42})
C = {'U': 'tab:blue', 'K': 'tab:red', 'XA': 'tab:green', 'REML': 'tab:purple', 'covLDSC': 'tab:orange'}
M = {'U': 's', 'K': 'D', 'XA': 'o', 'REML': '^', 'covLDSC': 'v'}
LAB = {'U': 'Unrelated subset', 'K': 'Related pairs kept', 'XA': 'Related pairs excluded (ours)',
       'REML': 'Two-kernel REML', 'covLDSC': 'cov-LDSC'}
T199 = stats.t.ppf(0.975, 199)
NUM = []
NAMES = {'alanine_aminotransferase': 'ALT', 'albumin': 'Albumin', 'alka_phos': 'ALP', 'apo_b': 'ApoB', 'asp_at': 'AST',
         'blood_mscv': 'MSCV', 'blood_platelet': 'Platelets', 'blood_rbc_count': 'RBC', 'bmi': 'BMI', 'bp_systolic': 'SBP',
         'calcium': 'Ca', 'creatinine': 'CRT', 'cystatin_c': 'CysC', 'diastolic_blood_pressure': 'DBP',
         'fluid_intelligence_score': 'FI', 'ggt': 'GGT', 'glucose': 'GLU', 'haemoglobin_concentration': 'Hb', 'hba1c': 'HbA1c',
         'hdl': 'HDL', 'heel_bone_mineral_density': 'BMD', 'height': 'HT', 'igf_1': 'IGF-1', 'ldl': 'LDL',
         'neuroticism_score': 'NEU', 'sleep_duration': 'Sleep', 'testosterone': 'Testo', 'triglycerides': 'TG', 'urate': 'Urate',
         'urea': 'Urea', 'vitamin_d': 'VitD', 'white_blood_cell_count': 'WBC', 'educational_qualification': 'EA',
         'ever_smoked': 'SMK', 'hand_grip_strength': 'Grip', 'whr': 'WHR', 'chronic_ischaemic_heart_disease_i25': 'IHD',
         'dyslipidaemia_e78': 'DYS', 'hypertension_i10': 'HTN', 'type2_diabetes_e11': 'T2D'}


def note(s):
    NUM.append(s)


def first(paths):
    return next(p for p in paths if os.path.exists(p))


def letter(ax, s, dx=-0.16, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, ls=':', lw=0.7, color='0.8'); ax.set_axisbelow(True)


def point(ax, x, y, key, yerr=None, ms=5.5):
    ax.errorbar(x, y, yerr=yerr, fmt=M[key], color=C[key], mfc=C[key], mec='black', mew=0.5, ms=ms, capsize=2, elinewidth=0.9,
                ls='none', zorder=3)


def pooled_bias(d):
    s = d.groupby(['cohort', 'setting']).err.agg(['mean', 'var', 'size'])
    return s['mean'].mean(), 1.96 * np.sqrt((s['var'] / s['size']).sum()) / len(s)


def log_ticks(ax_, ticks):
    ax_.set_major_locator(mticker.FixedLocator(ticks)); ax_.set_minor_locator(mticker.NullLocator())
    ax_.set_major_formatter(mticker.FixedFormatter([f'{t:g}' for t in ticks]))


# ============================================================================================== Figure 2: simulations
a = pd.read_csv(SRC['grid'], sep='\t', low_memory=False)
a = a[a.design.isin(['grid', 'mafld']) & a.arm.isin(['U', 'K', 'XA'])].assign(method=lambda d: d.arm)
b = pd.read_csv(SRC['reml'], sep='\t', low_memory=False)
b = b[b.design.isin(['grid', 'mafld']) & (b.arm == 'REML')].assign(method='REML')
c = pd.read_csv(SRC['covldsc'], sep='\t', low_memory=False)
c = c[c.design.isin(['grid', 'mafld']) & (c.arm == 'covLDSC_gw') & (c.model == '8bin')].assign(method='covLDSC')
S = pd.concat([a[a.model == '8bin'], b, c], ignore_index=True)
S['err'] = S.est - S.truth
S['cov'] = (S.err.abs() <= T199 * S.se)
S['scen'] = np.where(S.design == 'mafld', 'mafld', np.where(S.s2u > 0, 'shared', 'none'))
SCEN = [('none', 'Uniform'), ('shared', 'Shared by\nrelatives'), ('mafld', 'MAF/LD-\ndependent')]
METH = ['U', 'K', 'XA', 'REML', 'covLDSC']

fig = plt.figure(figsize=(7.2, 5.3))
gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.9, left=0.085, right=0.99, top=0.88, bottom=0.12)
axA, axB, axC = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
axD, axE = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
off = np.linspace(-0.28, 0.28, len(METH))
for i, (sc, _) in enumerate(SCEN):
    for k, mth in enumerate(METH):
        d = S[(S.scen == sc) & (S.method == mth)]
        m, h = pooled_bias(d)
        point(axA, i + off[k], m, mth, yerr=h, ms=5)
        cv = d.groupby(['cohort', 'setting'])['cov'].mean().mean()
        point(axB, i + off[k], cv, mth, ms=5)
        note(f'Fig2A/B {sc} {mth}: bias {m:+.4f} +- {h:.4f}, coverage {cv:.3f}')
axA.axhline(0, color='black', lw=0.8)
axA.set_ylabel('Bias of ĥ²'); axA.set_title('Bias')
axB.axhline(0.95, color='0.4', lw=0.8, ls='--')
axB.set_ylabel('Coverage of 95% intervals'); axB.set_title('Coverage'); axB.set_ylim(0.2, 1.02)
for ax in (axA, axB):
    ax.set_xticks(range(len(SCEN))); ax.set_xticklabels([l for _, l in SCEN]); ax.set_xlim(-0.5, len(SCEN) - 0.5)
    for x in (0.5, 1.5):
        ax.axvline(x, color='0.85', lw=0.7)
    grid(ax)
# C: variance of the unrelated-subset estimate over that of each method (same model; one component for REML)
u8 = a[(a.model == '8bin') & (a.arm == 'U') & (a.design == 'grid')].groupby(['cohort', 'setting']).est.var()
u1 = a[(a.model == 'single') & (a.arm == 'U') & (a.design == 'grid')].groupby(['cohort', 'setting']).est.var()
G = [('EUR', 'European'), ('SAS', 'South\nAsian'), ('AFR', 'Admixed')]
for k, (mth, ref) in enumerate([('XA', u8), ('REML', u1), ('covLDSC', u8)]):
    v = S[(S.method == mth) & (S.design == 'grid')].groupby(['cohort', 'setting']).est.var()
    for i, (coh, _) in enumerate(G):
        r = ref.loc[coh].sum() / v.loc[coh].sum()
        point(axC, i + (k - 1) * 0.22, r, mth, ms=5)
        note(f'Fig2C {coh} {mth}: variance ratio {r:.3f}')
axC.axhline(1, color='0.4', lw=0.8, ls='--')
axC.set_yscale('log'); axC.set_ylim(0.04, 3); log_ticks(axC.yaxis, [0.05, 0.1, 0.2, 0.5, 1, 2])
axC.set_xticks(range(3)); axC.set_xticklabels([l for _, l in G]); axC.set_xlim(-0.5, 2.5)
axC.set_ylabel('Relative precision'); axC.set_title('Precision'); grid(axC)
fig.legend([Line2D([], [], marker=M[m], color=C[m], mfc=C[m], mec='black', mew=0.5, ms=5.5, ls='none') for m in METH],
           [LAB[m] for m in METH], loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.53, 1.0), handletextpad=0.3,
           columnspacing=1.4)
# D: local ancestry in the admixed group
la = pd.read_csv(SRC['la_sims'], sep='\t', keep_default_na=False, na_values=['nan', 'NaN', ''])
lorder = [('null', 'None'), ('model', 'Local ancestry\n0.05'), ('untagged', 'Untagged\n0.25'),
          ('untagged2', 'Untagged\n0.5')]
for k, (mod, lab, col, mk) in enumerate([('SNP8c', 'SNPs only', 'tab:red', 'D'), ('LA9c', 'SNPs and local ancestry (ours)', 'tab:green', 'o')]):
    for i, (s, _) in enumerate(lorder):
        e = la[(la.setting == s) & (la.model == mod)]
        err = e.h2_snp - e.truth_snp
        m, h = err.mean(), 1.96 * err.std(ddof=1) / np.sqrt(len(err))
        axD.errorbar(i + (k - 0.5) * 0.24, m, yerr=h, fmt=mk, color=col, mec='black', mew=0.5, ms=5, capsize=2, elinewidth=0.9,
                     label=lab if i == 0 else None)
        note(f'Fig2D {s} {mod}: bias {m:+.4f} +- {h:.4f}' + (f'; LA component {e.h2_la.mean():.4f}' if mod == 'LA9c' else ''))
axD.axhline(0, color='black', lw=0.8)
axD.set_xticks(range(len(lorder))); axD.set_xticklabels([l for _, l in lorder]); axD.set_xlim(-0.5, len(lorder) - 0.5)
axD.set_ylabel('Bias of SNP ĥ²'); axD.set_title('Local ancestry, admixed group (4,324)'); axD.set_ylim(-0.04, 0.27)
axD.legend(loc='upper center', ncol=2); grid(axD)
# E: precision of the wider African-ancestry group against its admixed members
WIDE = pd.read_csv(SRC['wide'], sep='\t', low_memory=False, keep_default_na=False, na_values=['nan', 'NaN', ''])
lab_set = {6: '$h^2$ 0.25', 7: '0.25\nshared', 10: '0.25\nsparse', 12: '0.4', 13: '0.4\nshared', 22: '0.4\nLDAK'}
for k, (coh, lab, col, mk) in enumerate([('AFRWA', 'Admixed members (4,202)', 'tab:blue', 's'),
                                          ('AFRW', 'Wider group (9,647)', 'tab:green', 'o')]):
    for i, s in enumerate(lab_set):
        r = WIDE[WIDE.task == f'{coh}_{s:02d}']
        e = r[(r.arm == 'XA') & (r.model == '8bin')]
        sd = e.est.std(ddof=1)
        axE.plot(i + (k - 0.5) * 0.24, sd, mk, color=col, mec='black', mew=0.5, ms=5.5, label=lab if i == 0 else None)
        note(f'Fig2E {coh} setting {s}: SD {sd:.4f}, bias {(e.est - e.truth).mean():+.4f}')
axE.set_ylim(0, 0.16); axE.set_xticks(range(len(lab_set))); axE.set_xticklabels(list(lab_set.values()))
axE.set_xlim(-0.5, len(lab_set) - 0.5)
axE.set_ylabel('SD of ĥ²'); axE.set_title('Wider African-ancestry group'); axE.legend(loc='lower left'); grid(axE)
for ax, s, dx in ((axA, 'A', -0.36), (axB, 'B', -0.36), (axC, 'C', -0.36), (axD, 'D', -0.15), (axE, 'E', -0.15)):
    letter(ax, s, dx=dx)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig2_simulations.{ext}')
plt.close(fig)

# ============================================================================================== Figure S1: direct effects
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.28, left=0.08, right=0.98, bottom=0.2, top=0.9))
sr = pd.read_csv(SRC['sib_sim'], sep='\t')
order = [('base', 'None'), ('sibenv', 'Sibship\nenvironment'), ('extfam', 'Extended\nfamily'), ('nurture', 'Parental\ngenotypes'),
         ('ancestry', 'Ancestry-\nlinked')]
for k, (est, lab, col, mk) in enumerate([('pop_ext', 'Population', 'tab:blue', 's'), ('dir_ext', 'Within sibships (ours)', 'tab:green', 'o')]):
    for i, (s, _) in enumerate(order):
        e = sr[(sr.setting == s) & (sr.estimator == est)]
        err = e.est - e.truth
        m, h = err.mean(), 1.96 * err.std(ddof=1) / np.sqrt(len(err))
        axA.errorbar(i + (k - 0.5) * 0.24, m, yerr=h, fmt=mk, color=col, mec='black', mew=0.5, ms=5, capsize=2, elinewidth=0.9,
                     label=lab if i == 0 else None)
        note(f'FigS1A {s} {est}: bias {m:+.4f} +- {h:.4f}')
axA.axhline(0, color='black', lw=0.8)
axA.set_xticks(range(len(order))); axA.set_xticklabels([l for _, l in order]); axA.set_xlim(-0.5, len(order) - 0.5)
axA.set_ylabel('Bias'); axA.set_title('Simulations, 35,365 siblings'); axA.legend(loc='upper left'); grid(axA)
s32 = pd.read_csv(SRC['sib32'], sep='\t')[['trait', 'dir_8bin', 'dir_8bin_se_snpjk', 'pop_8bin', 'pop_8bin_se']]
sx = pd.read_csv(SRC['sib_extra'], sep='\t')
sx = sx[sx.trait.isin(['educational_qualification', 'ever_smoked'])][['trait', 'dir_8bin', 'dir_8bin_se_snpjk', 'pop_8bin', 'pop_8bin_se']]
s6 = pd.read_csv(SRC['sib_extra2'], sep='\t').rename(columns={'direct': 'dir_8bin', 'direct_se_snpjk': 'dir_8bin_se_snpjk',
                                                              'population': 'pop_8bin', 'population_se': 'pop_8bin_se'})
D_ = pd.concat([s32, sx, s6[s32.columns]]).drop_duplicates('trait')
axB.errorbar(D_.pop_8bin, D_.dir_8bin, xerr=1.96 * D_.pop_8bin_se, yerr=1.96 * D_.dir_8bin_se_snpjk, fmt='o', color='tab:green',
             mec='black', mew=0.5, ms=4.5, capsize=1.2, elinewidth=0.6, ecolor='0.6')
for t, dx, dy in [('height', -8, 6), ('educational_qualification', 8, -2)]:
    r_ = D_[D_.trait == t].iloc[0]
    axB.annotate(NAMES[t], (r_.pop_8bin, r_.dir_8bin), xytext=(dx, dy), textcoords='offset points', fontsize=7,
                 ha='right' if dx < 0 else 'left')
axB.plot([-0.15, 0.7], [-0.15, 0.7], ls='--', color='0.4', lw=0.8)
axB.set_xlim(-0.02, 0.68); axB.set_ylim(-0.15, 0.68)
axB.set_xlabel('Population ĥ²'); axB.set_ylabel('Direct-effect ĥ²'); axB.set_title('40 traits, 35,365 siblings')
grid(axB, 'both')
note(f'FigS1B {len(D_)} traits')
letter(axA, 'A', dx=-0.12); letter(axB, 'B', dx=-0.13)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/figS1_direct.{ext}')
plt.close(fig)

# ============================================================================================== Figure 5: genetic correlation, European group
R = pd.read_csv(SRC['rg'], sep='\t')
def rga(v, a_, m='8bin'):
    return R[(R.variant == v) & (R.arm == a_) & (R.model == m)].set_index('key')
ref = pd.read_csv(SRC['rg_ref'], sep='\t')
ref['key'] = ['|'.join(sorted(p)) for p in zip(ref.phen1, ref.phen2)]
ref = ref.set_index('key')
fig, axB = plt.subplots(1, 1, figsize=(2.8, 2.6), gridspec_kw=dict(left=0.2, right=0.96, bottom=0.17, top=0.9))
e_x, e_u = rga('EUR', 'XA'), rga('EUR', 'U')
jj = e_x.join(e_u[['rg_total', 'rg_total_se']], rsuffix='_u', how='inner')
axB.plot(jj.rg_total_u, jj.rg_total, '.', color=C['XA'], ms=3, alpha=0.6)
axB.plot([-1.2, 1.2], [-1.2, 1.2], ls='--', color='0.4', lw=0.8); axB.set_xlim(-1.2, 1.2); axB.set_ylim(-1.2, 1.2)
axB.set_xlabel('Unrelated subset (12,148)'); axB.set_ylabel('All participants (14,558)'); axB.set_title('r̂$_g$, European group')
grid(axB, 'both')
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/figS2_rg.{ext}')
plt.close(fig)

open(f'{OUT}/numbers.txt', 'w').write('\n'.join(NUM) + '\n')
print('\n'.join(NUM))
