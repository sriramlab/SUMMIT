"""Figures 3 and 4 of "Heritability from summary statistics of admixed and related biobank participants", from the
aggregate results in ../data.
Figure 3  relatives in two biobanks: h2 of all participants, with relatives' phenotypic resemblance excluded from or kept
          in the estimating equations, against the unrelated subset, in UK Biobank white British participants (A) and in
          the European (B), African (C) and Admixed American (D) groups of All of Us. Panels B-D need the All of Us
          aggregate table (--aou; not included in this folder), and are left empty without it.
Figure 4  the wider group of all participants of African ancestry in the UK Biobank: h2 (A) and rg (B) against white
          British participants, standard errors of h2 (C) and rg (D) against the unrelated admixed subset, local-ancestry
          component (E).
usage: python make_fig34.py [--out DIR] [--aou ALL_OF_US_ARMS_CSV]
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import argparse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
ap.add_argument('--aou', default=None, help='All of Us arms table (group, trait, model, n_U, h2_U, se_U, h2_K, se_K, n_X, '
                                           'h2_X, se_X, ...); panels 3B-D are left empty without it')
ARGS = ap.parse_args()
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42})
C = {'U': 'tab:blue', 'K': 'tab:red', 'XA': 'tab:green'}
NAMES = {'alanine_aminotransferase': 'ALT', 'albumin': 'Albumin', 'alka_phos': 'ALP', 'apo_b': 'ApoB', 'asp_at': 'AST',
         'blood_mscv': 'MSCV', 'blood_platelet': 'Platelets', 'blood_rbc_count': 'RBC', 'bmi': 'BMI', 'bp_systolic': 'SBP',
         'calcium': 'Ca', 'creatinine': 'CRT', 'cystatin_c': 'CysC', 'diastolic_blood_pressure': 'DBP',
         'fluid_intelligence_score': 'FI', 'ggt': 'GGT', 'glucose': 'GLU', 'haemoglobin_concentration': 'Hb', 'hba1c': 'HbA1c',
         'hdl': 'HDL', 'heel_bone_mineral_density': 'BMD', 'height': 'HT', 'igf_1': 'IGF-1', 'ldl': 'LDL',
         'neuroticism_score': 'NEU', 'sleep_duration': 'Sleep', 'testosterone': 'Testo', 'triglycerides': 'TG', 'urate': 'Urate',
         'urea': 'Urea', 'vitamin_d': 'VitD', 'white_blood_cell_count': 'WBC', 'educational_qualification': 'EA',
         'ever_smoked': 'SMK', 'hand_grip_strength': 'Grip', 'whr': 'WHR', 'chronic_ischaemic_heart_disease_i25': 'IHD',
         'dyslipidaemia_e78': 'DYS', 'hypertension_i10': 'HTN', 'type2_diabetes_e11': 'T2D'}
GNAME = {'eur': 'European', 'afr': 'African', 'amr': 'Admixed American'}


def letter(ax, s, dx=-0.16, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, ls=':', lw=0.7, color='0.8'); ax.set_axisbelow(True)


def log_ticks(ax_, ticks):
    ax_.set_major_locator(mticker.FixedLocator(ticks)); ax_.set_minor_locator(mticker.NullLocator())
    ax_.set_major_formatter(mticker.FixedFormatter([f'{t:g}' for t in ticks]))


# --------------------------------------------------------------------------------------------------------------- data
U406 = pd.read_csv(f'{DATA}/white_british_traits.tsv', sep='\t')
U406 = U406[U406.model == '8bin']
if 'h2L_xa' in U406:
    U406 = U406.assign(h2_unrel=U406.h2L_unrel, h2_kept=U406.h2L_kept, h2_xa=U406.h2L_xa)
E = pd.read_csv(f'{DATA}/cohort_traits.tsv', sep='\t')
CT = pd.read_csv(f'{DATA}/cohort_tests.tsv', sep='\t')
LA = pd.read_csv(f'{DATA}/local_ancestry_traits.tsv', sep='\t')
R = pd.read_csv(f'{DATA}/genetic_correlation_cohorts.tsv.gz', sep='\t')
REF = pd.read_csv(f'{DATA}/genetic_correlation_reference.tsv', sep='\t')
REF['key'] = ['|'.join(sorted(p)) for p in zip(REF.phen1, REF.phen2)]
REF = REF.set_index('key')
AO = pd.read_csv(ARGS.aou) if ARGS.aou else None
if AO is not None:
    AO = AO[AO.model == '8bin']


def arm(cohort, a_, s='base', m='8bin'):
    return E[(E.cohort == cohort) & (E.arm == a_) & (E.model == m) & (E.set == s)].set_index('trait')


def rga(v, a_, m='8bin'):
    return R[(R.variant == v) & (R.arm == a_) & (R.model == m)].set_index('key')


# -------------------------------------------------------------------------------------------------------------- panels
def p_wb(ax):
    u = U406
    ax.plot(u.h2_unrel, u.h2_kept, 'D', color=C['K'], mec='black', mew=0.5, ms=4, label='Kept')
    ax.plot(u.h2_unrel, u.h2_xa, 'o', color=C['XA'], mec='black', mew=0.5, ms=4.5, label='Excluded')
    hi = 0.75
    ax.plot([0, hi], [0, hi], ls='--', color='0.4', lw=0.8); ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel('Unrelated subset (338k)'); ax.set_ylabel('All participants (407k)')
    ax.set_title('UK Biobank, white British', fontsize=8)
    ax.legend(loc='lower right', title='Related pairs', fontsize=6.5, title_fontsize=6.5); grid(ax, 'both')


def p_aou(ax, g, title=None, legend=False):
    if AO is None:
        ax.set_title(title or f'All of Us, {GNAME[g]}', fontsize=8); ax.text(0.5, 0.5, 'All of Us table\nnot included',
                                                                         ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return
    s = AO[AO.group == g].set_index('trait')
    ax.errorbar(s.h2_U, s.h2_K, xerr=1.96 * s.se_U, yerr=1.96 * s.se_K, fmt='D', color=C['K'], mec='black', mew=0.5, ms=4,
                capsize=1.2, elinewidth=0.6, label='Kept', zorder=3)
    ax.errorbar(s.h2_U, s.h2_X, xerr=1.96 * s.se_U, yerr=1.96 * s.se_X, fmt='o', color=C['XA'], mec='black', mew=0.5,
                ms=4.5, capsize=1.2, elinewidth=0.6, label='Excluded', zorder=4)
    ax.plot([0, 0.75], [0, 0.75], ls='--', color='0.4', lw=0.8); ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.75)
    ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel(f'Unrelated subset ({s.n_U.median() / 1000:.0f}k)')
    ax.set_ylabel(f'All participants ({s.n_X.median() / 1000:.0f}k)')
    ax.set_title(title or f'All of Us, {GNAME[g]}', fontsize=8)
    if legend:
        ax.legend(loc='lower right', title='Related pairs', fontsize=6.5, title_fontsize=6.5)
    grid(ax, 'both')


def p_se_h2(ax):
    w, r0 = arm('Wider African group', 'XA'), arm('African (standard)', 'U')
    j = w[['se']].join(r0[['se']], rsuffix='_ref')
    ax.plot(j.se_ref, j.se, 'o', color=C['XA'], mec='black', mew=0.5, ms=4.5)
    lo_, hi_ = 0.03, 0.8
    ax.plot([lo_, hi_], [lo_, hi_], ls='--', color='0.4', lw=0.8, label='Equal')
    ax.plot([lo_, hi_], [lo_ / 2, hi_ / 2], ls=':', color='0.4', lw=1.0, label='Half')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    log_ticks(ax.xaxis, [0.05, 0.1, 0.2, 0.5]); log_ticks(ax.yaxis, [0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel('Unrelated admixed\nsubset'); ax.set_ylabel('Wider group (9,647)')
    ax.set_title(f'SE of ĥ² (median ratio {(j.se / j.se_ref).median():.2f})', fontsize=8)
    ax.legend(loc='upper left', fontsize=6.5); grid(ax, 'both')


def p_se_rg(ax):
    w = rga('AFRWX', 'XA')
    a0 = rga('AFR', 'U')
    ss = w.join(a0[['rg_total_se']], rsuffix='_a', how='inner')
    ax.plot(ss.rg_total_se_a, ss.rg_total_se, '.', color=C['XA'], ms=3, alpha=0.6)
    lo_, hi_ = 0.02, 5
    ax.plot([lo_, hi_], [lo_, hi_], ls='--', color='0.4', lw=0.8, label='Equal')
    ax.plot([lo_, hi_], [lo_ / 2, hi_ / 2], ls=':', color='0.4', lw=1.0, label='Half')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    log_ticks(ax.xaxis, [0.05, 0.2, 1]); log_ticks(ax.yaxis, [0.05, 0.2, 1])
    ax.set_xlabel('Unrelated admixed\nsubset'); ax.set_ylabel('Wider group (9,647)')
    ax.set_title(f'SE of r̂$_g$ (median ratio {(ss.rg_total_se / ss.rg_total_se_a).median():.2f})', fontsize=8)
    ax.legend(loc='upper left', fontsize=6.5); grid(ax, 'both')


def p_rg(ax, ylabel='Wider group (9,647)'):
    w = rga('AFRWX', 'XA').join(REF[['rg', 'rg_se']], how='inner')
    sig = w.q < 0.05
    ax.plot(w.rg[~sig], w.rg_total[~sig], '.', color='0.7', ms=3, label='Other pairs')
    ax.errorbar(w.rg[sig], w.rg_total[sig], yerr=1.96 * w.rg_total_se[sig], fmt='o', color=C['XA'], mec='black', mew=0.5,
                ms=4, capsize=1.2, elinewidth=0.7, label='FDR < 0.05')
    ax.plot([-1, 1], [-1, 1], ls='--', color='0.4', lw=0.8)
    ax.set_xlim(-1, 1); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('White British (291,273)'); ax.set_ylabel(ylabel); ax.set_title('r̂$_g$', fontsize=8)
    ax.legend(loc='lower right', fontsize=6.5); grid(ax, 'both')


def p_cmp_wb(ax):
    o = CT[(CT.set == 'origin') & (CT.ref == 'EUR300k') & (CT.cohort == 'Wider African group')].set_index('trait')
    bb = CT[(CT.set == 'base') & (CT.ref == 'EUR300k') & (CT.cohort == 'Wider African group')].set_index('trait')
    ax.errorbar(o.ref_h2, o.h2, yerr=1.96 * o.se, fmt='o', color=C['XA'], mec='black', mew=0.5, ms=4, capsize=1.2,
                elinewidth=0.6)
    fi = 'fluid_intelligence_score'
    ax.plot(bb.ref_h2[fi], bb.h2[fi], 'o', mfc='white', mec='tab:red', mew=1.0, ms=4.5)
    ax.annotate('', xy=(o.ref_h2[fi], o.h2[fi] + 0.03), xytext=(bb.ref_h2[fi], bb.h2[fi] - 0.03),
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=0.9))
    ax.plot([-0.25, 1], [-0.25, 1], ls='--', color='0.4', lw=0.8)
    ax.set_xlim(-0.02, 0.62); ax.set_ylim(-0.25, 1.0)
    ax.set_xlabel('White British (291,273)'); ax.set_ylabel('Wider group (9,647)')
    ax.set_title('ĥ² with origin covariates', fontsize=8); grid(ax, 'both')


def p_la(ax):
    lw_, la_ = LA[LA.group == 'AFRW'].set_index('trait'), LA[LA.group == 'AFRWA'].set_index('trait')
    order = lw_.la.sort_values().index
    x = np.arange(len(order))
    for k, (dd, col, mk, lab) in enumerate([(la_, 'tab:blue', 's', 'Admixed members (4,202)'),
                                           (lw_, 'tab:green', 'o', 'Wider group (9,647)')]):
        dd = dd.reindex(order)
        ax.errorbar(x + (k - 0.5) * 0.3, dd.la, yerr=1.96 * dd.la_se, fmt=mk, color=col, mec='black', mew=0.4, ms=3.6,
                    capsize=1.2, elinewidth=0.7, label=lab)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NAMES.get(t, t) for t in order], rotation=90, fontsize=6.3)
    ax.set_xlim(-0.7, len(order) - 0.3); ax.set_ylim(-0.12, 0.27)
    ax.set_ylabel('Local-ancestry component'); ax.legend(loc='upper left'); grid(ax)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}')
    plt.close(fig)
    print('written', name)


# -------------------------------------------------------------------------------------------------------------- layouts
def figure3():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), gridspec_kw=dict(wspace=0.55, left=0.075, right=0.985, top=0.82,
                                                                        bottom=0.18))
    p_wb(axs[0]); p_aou(axs[1], 'eur'); p_aou(axs[2], 'afr'); p_aou(axs[3], 'amr')
    for ax, t in zip(axs, ['UK Biobank\nwhite British', 'All of Us\nEuropean', 'All of Us\nAfrican', 'All of Us\nAdmixed American']):
        ax.set_title(t, fontsize=8)
    for ax, s in zip(axs, 'ABCD'):
        letter(ax, s, dx=-0.42, dy=1.12)
    save(fig, 'fig3_relatives')


def figure4():
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95], wspace=0.62, hspace=0.62, left=0.075, right=0.99, top=0.9,
                          bottom=0.12)
    axA, axB, axC, axD = [fig.add_subplot(gs[0, k]) for k in range(4)]
    axE = fig.add_subplot(gs[1, :])
    p_cmp_wb(axA); p_rg(axB); p_se_h2(axC); p_se_rg(axD); p_la(axE)
    for ax in (axC, axD):
        ax.set_title(ax.get_title().replace(' (median', '\n(median'), fontsize=8)
    axA.set_title('ĥ² with origin\ncovariates', fontsize=8); axB.set_title('r̂$_g$', fontsize=8)
    for ax, s in zip((axA, axB, axC, axD), 'ABCD'):
        letter(ax, s, dx=-0.45, dy=1.12)
    letter(axE, 'E', dx=-0.06, dy=1.02)
    save(fig, 'fig4_african')


if __name__ == '__main__':
    figure3(); figure4()
