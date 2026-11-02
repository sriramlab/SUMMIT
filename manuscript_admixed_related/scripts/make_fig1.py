"""Figure 1 of "Heritability from summary statistics of admixed and related biobank participants": the pairs of
individuals that each estimator uses, and local ancestry as a set of markers.
usage: python make_fig1.py [--out DIR]
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'mathtext.fontset': 'dejavusans',
                     'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42})
USED, OUTC, SELF, AFR, EUR = 'white', 'tab:orange', '0.25', 'tab:brown', '0.8'


def matrix(ax, kind, fam, sib):
    """kind 'pop': related pairs out, diagonal kept; 'dir': every pair within an extended family out, diagonal included."""
    n = len(fam)
    for i in range(n):
        for j in range(n):
            if i == j:
                c = SELF if kind == 'pop' else OUTC
            elif fam[i] == fam[j] and fam[i] >= 0:
                c = OUTC
            else:
                c = USED
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=c, ec='0.6', lw=0.4))
    if kind == 'dir':                                      # sibships outlined
        for s in sorted(set(sib)):
            ix = [k for k in range(n) if sib[k] == s]
            if len(ix) > 1:
                a, b = min(ix), max(ix)
                ax.add_patch(Rectangle((a, n - 1 - b), b - a + 1, b - a + 1, fill=False, ec='k', lw=1.1))
    ax.set_xlim(-0.2, n + 0.1); ax.set_ylim(-0.2, n + 0.1); ax.set_aspect('equal'); ax.axis('off')


fig = plt.figure(figsize=(7.2, 2.75))
axA = fig.add_axes([0.02, 0.17, 0.27, 0.66])
axB = fig.add_axes([0.35, 0.17, 0.27, 0.66])
axC = fig.add_axes([0.68, 0.17, 0.31, 0.66])
# a: population heritability; one family of three, one of two, unrelated individuals
fam_a = [0, 0, 0, 1, 1, -1, -2, -3, -4, -5]
matrix(axA, 'pop', fam_a, None)
axA.set_title('Population heritability', loc='left')
# b: direct effects; extended family 0 holds two sibships (cousins), family 1 one sibship of three, family 2 one of two
fam_b = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
sib_b = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
matrix(axB, 'dir', fam_b, sib_b)
axB.set_title('Direct effects (sibling deviations)', loc='left')
# c: local-ancestry tracts of four admixed haplotype pairs and the two classes of markers
rng = np.random.default_rng(3)
L = 100.0
for k in range(4):
    for h in range(2):
        y = 3.6 - k * 0.95 - h * 0.34
        pos, anc = 0.0, rng.integers(0, 2)
        while pos < L:
            ln = rng.exponential(28)
            axC.add_patch(Rectangle((pos, y), min(ln, L - pos), 0.26, color=AFR if anc else EUR, lw=0))
            pos += ln; anc = 1 - anc
snp = np.sort(rng.uniform(0, L, 60))
axC.vlines(snp, 4.2, 4.45, color='tab:blue', lw=0.6)
axC.vlines(np.arange(5, L, 10), 4.6, 4.85, color=AFR, lw=1.4)
axC.text(-2, 4.32, 'SNPs', ha='right', va='center', fontsize=7.5)
axC.text(-2, 4.72, 'Local ancestry', ha='right', va='center', fontsize=7.5)
axC.set_xlim(-30, L + 1); axC.set_ylim(0.3, 5.0); axC.axis('off')
axC.set_title('Local ancestry as a set of markers', loc='left')
fig.legend([Patch(fc=USED, ec='0.6', lw=0.4), Patch(color=OUTC), Patch(color=SELF), Patch(fill=False, ec='k', lw=0.9), Patch(color=AFR),
            Patch(color=EUR)],
           ['Pair used', 'Pair left out', 'Individual with themselves', 'Sibship', 'African segment', 'European segment'],
           loc='lower center', ncol=6, frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
for ax, s in zip((axA, axB, axC), 'ABC'):
    pos = ax.get_position()
    fig.text(pos.x0, pos.y1 + 0.085, s, fontsize=11, fontweight='bold', va='bottom')
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig1_schematic.{ext}', bbox_inches='tight')
print('written')
