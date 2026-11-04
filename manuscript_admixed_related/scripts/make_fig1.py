"""Figure 1 of "Heritability from summary statistics of admixed and related biobank participants". A, pairs of relatives
left out of the estimating equations; B, local ancestry of admixed and unadmixed individuals, with the SNPs and the
local-ancestry loci; C, the model for the products of phenotypes of pairs, with SNP and local-ancestry similarity; D, the
summary statistics computed in the study sample (per SNP, per trait and per local-ancestry locus) and the estimates
obtained from them. Matrices are toy values.
Writes OUT/fig1_schematic.{pdf,png}.
usage: python make_fig1.py [--out DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output'))
OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'mathtext.fontset': 'dejavusans', 'savefig.dpi': 300,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
USED, OUTC, SELF, AFR, EUR = 'white', 'tab:orange', '0.25', 'tab:brown', '0.8'


def panel_relatives(ax):
    fam = [0, 0, 0, 1, 1, -1, -2, -3, -4, -5]
    n = len(fam)
    for i in range(n):
        for j in range(n):
            c = SELF if i == j else OUTC if fam[i] == fam[j] and fam[i] >= 0 else USED
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=c, ec='0.6', lw=0.4))
    ax.set_xlim(-0.2, n + 0.1); ax.set_ylim(-0.2, n + 0.1); ax.set_aspect('equal'); ax.axis('off')


def panel_admixed(ax):
    rng = np.random.default_rng(7)
    LEN, NI = 100.0, 8
    admixed = [True] * 6 + [False] * 2
    mean_len = [18, 30, 45, 22, 60, 35]
    for k in range(NI):
        for h in range(2):
            y = NI - 1 - k + (0.38 if h == 0 else 0.0)
            if admixed[k]:
                pos, anc = 0.0, int(rng.integers(0, 2))
                while pos < LEN:
                    end = min(pos + rng.exponential(mean_len[k]), LEN)
                    ax.add_patch(Rectangle((pos, y), end - pos, 0.32, color=AFR if anc else EUR, lw=0))
                    pos, anc = end, 1 - anc
            else:
                ax.add_patch(Rectangle((0, y), LEN, 0.32, color=AFR, lw=0))
    snp = np.sort(rng.uniform(0, LEN, 70))
    ax.vlines(snp, NI + 0.25, NI + 0.6, color='tab:blue', lw=0.6)
    ax.vlines((np.arange(40) + 0.5) * 2.5, NI + 0.85, NI + 1.2, color=AFR, lw=1.1)
    ax.text(-2, NI + 0.42, 'SNPs', ha='right', va='center', fontsize=7)
    ax.text(-2, NI + 1.02, 'Local-ancestry loci', ha='right', va='center', fontsize=7)
    for k in range(NI):
        ax.text(LEN + 2, NI - 1 - k + 0.35, 'admixed' if admixed[k] else 'unadmixed', ha='left', va='center',
                fontsize=6.5, color='0.3')
    ax.set_xlim(-28, LEN + 16); ax.set_ylim(-0.2, NI + 1.4); ax.axis('off')


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=11, fontweight='bold', va='bottom')


def save(fig):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/fig1_schematic.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('written')


CM_SNP = LinearSegmentedColormap.from_list('snp', ['white', 'tab:blue'])
CM_LA = LinearSegmentedColormap.from_list('la', ['white', AFR])
CM_Y = LinearSegmentedColormap.from_list('y', ['white', '0.35'])


def calm_matrix(ax, M, cmap, rel=None, diag_dark=False, show=None):
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            vis = True if show is None else show[i, j]
            fc = cmap(float(np.clip(M[i, j], 0, 1))) if vis else 'white'
            if diag_dark and i == j and vis:
                fc = SELF
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec='0.8', lw=0.3))
    if rel is not None:
        for i, j in zip(*np.nonzero(rel)):
            ax.add_patch(Rectangle((j + 0.08, n - 1 - i + 0.08), 0.84, 0.84, fill=False, ec=OUTC, lw=1.0))
    ax.set_xlim(-0.05, n + 0.05); ax.set_ylim(-0.05, n + 0.05); ax.set_aspect('equal'); ax.axis('off')


def calm_sample(n=8, seed=2):
    rng = np.random.default_rng(seed)
    K = np.abs(rng.normal(0, 0.05, (n, n))); K = (K + K.T) / 2
    fam = [0, 1, 2]
    rel = np.zeros((n, n), bool)
    for i in fam:
        for j in fam:
            if i != j:
                rel[i, j] = True
                K[i, j] = 0.45
    np.fill_diagonal(K, 1.0)
    KL = np.zeros((n, n))
    adm = np.arange(6)
    u = rng.uniform(0.1, 0.6, (6, 6)); u = (u + u.T) / 2
    KL[np.ix_(adm, adm)] = u
    KL[adm, adm] = 0.9
    return K, KL, rel


def row_of_matrices(fig, x0, y0, w, h, mats, ops, labels):
    k = len(mats)
    cw = h * fig.get_figheight() / fig.get_figwidth()            # square cells
    gap = (w - k * cw) / (k - 1)
    for i, (args, lab) in enumerate(zip(mats, labels)):
        ax = fig.add_axes([x0 + i * (cw + gap), y0, cw, h])
        calm_matrix(ax, *args)
        ax.text(4, -0.5, lab, ha='center', va='top', fontsize=6.8)
        if i < k - 1:
            fig.text(x0 + i * (cw + gap) + cw + gap / 2, y0 + h / 2, ops[i], ha='center', va='center', fontsize=9)


W, TOP, DH = 7.2, 3.85, 1.6          # figure width; inches taken by panels A-C; height of panel D
INK, RELC, LAC = '0.1', OUTC, AFR


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, lw=0.9, color='0.35',
                                 shrinkA=0, shrinkB=0))


def plain_table(ax, x, y, cols, colors, nrow, cw=0.34, rh=0.15, hh=0.2):
    """Header row of height hh with the column symbols, nrow empty body rows, a vertical ellipsis in the last row."""
    for k, (c, col) in enumerate(zip(cols, colors)):
        ax.add_patch(Rectangle((x + k * cw, y + nrow * rh), cw, hh, fc='white', ec='0.45', lw=0.5))
        ax.text(x + (k + 0.5) * cw, y + nrow * rh + 0.45 * hh, c, ha='center', va='center', fontsize=7.5, color=col)
        for r in range(nrow):
            ax.add_patch(Rectangle((x + k * cw, y + r * rh), cw, rh, fc='white', ec='0.8', lw=0.4))
        if nrow > 1:
            ax.text(x + (k + 0.5) * cw, y + 0.55 * rh, r'$\vdots$', ha='center', va='center', fontsize=6, color='0.5')
    return x + len(cols) * cw


def panel_d(ax):
    rng = np.random.default_rng(4)
    n, m, cs = 9, 13, 0.09
    x0, y0 = 0.45, 0.3
    ytop = y0 + n * cs
    G = rng.choice([0.0, 0.45, 0.85], size=(n, m), p=[0.45, 0.4, 0.15])
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((x0 + j * cs, y0 + (n - 1 - i) * cs), cs, cs, fc=CM_Y(G[i, j]), ec='0.8', lw=0.3))
        ax.add_patch(Rectangle((x0 + (m + 0.6) * cs, y0 + (n - 1 - i) * cs), cs, cs, fc=CM_Y(rng.uniform(0.1, 0.8)),
                               ec='0.8', lw=0.3))
    ax.text(x0 + m * cs / 2, y0 - 0.07, 'genotypes', ha='center', va='top', fontsize=6.8)
    ax.text(x0 + (m + 1.1) * cs, y0 - 0.07, 'phenotype', ha='center', va='top', fontsize=6.8)
    ax.text(x0 + (m + 1.6) * cs / 2, ytop + 0.12, 'Study sample', ha='center', va='bottom', fontsize=7.5)
    ya = y0 + n * cs / 2
    xa0, xa1 = 2.0, 2.6
    arrow(ax, xa0, ya, xa1, ya)
    ax.text((xa0 + xa1) / 2, ya + 0.08, 'computed in\nthe sample', ha='center', va='bottom', fontsize=6.5, linespacing=1.05)
    tx, cw = 2.75, 0.3
    end = plain_table(ax, tx, y0, [r'$m_j$', r'$\ell_j$', r'$\ell^{\mathcal{R}}_j$', r'$\ell^{\mathrm{LA}}_j$'],
                      [INK, INK, RELC, LAC], 4, cw=cw)
    ax.text((tx + end) / 2, y0 - 0.07, 'per SNP', ha='center', va='top', fontsize=6.5, color='0.3')
    t2 = end + 0.15
    e2 = plain_table(ax, t2, y0, [r'$c^{\mathcal{R}}$'], [RELC], 1, cw=cw)
    ax.text((t2 + e2) / 2, y0 - 0.07, 'per trait', ha='center', va='top', fontsize=6.5, color='0.3')
    t3 = e2 + 0.15
    e3 = plain_table(ax, t3, y0, [r'$m_l$', r'$\ell_l$'], [LAC, LAC], 1, cw=cw)
    ax.text((t3 + e3) / 2, y0 - 0.07, 'per locus', ha='center', va='top', fontsize=6.5, color='0.3')
    ax.text((tx + e3) / 2, ytop + 0.12, 'Summary statistics', ha='center', va='bottom', fontsize=7.5)
    xb0 = e3 + 0.15
    xb1 = xb0 + 0.5
    arrow(ax, xb0, ya, xb1, ya)
    ax.text((xb0 + xb1) / 2, ya + 0.08, 'method of\nmoments', ha='center', va='bottom', fontsize=6.5, linespacing=1.05)
    ex = xb1 + 0.12
    ax.text(ex, ytop + 0.12, 'Estimates', ha='left', va='bottom', fontsize=7.5)
    lines = [(r'SNP heritability $h^2$', INK, 0), (r'genetic correlation $r_g$', INK, 0.2),
             ('local-ancestry', LAC, 0.42), (r'component $\sigma^2_{\mathrm{LA}}$', LAC, 0.57), ('standard errors', INK, 0.8)]
    for t, c, dy in lines:
        ax.text(ex, ytop - 0.1 - dy, t, ha='left', va='center', fontsize=7, color=c)
    ax.set_xlim(0, W); ax.set_ylim(0, DH); ax.axis('off')


def figure1():
    H = TOP + DH
    fig = plt.figure(figsize=(W, H))
    fy = lambda inch_from_top: 1 - inch_from_top / H
    axA = fig.add_axes([0.065, fy(2.0), 0.22, 1.7 / H])
    axB = fig.add_axes([0.445, fy(2.0), 0.42, 1.7 / H])
    panel_relatives(axA); panel_admixed(axB)
    fig.canvas.draw()
    top = max(axA.get_position().y1, axB.get_position().y1)
    letter(fig, 0.02, top + 0.1 / H, 'A'); letter(fig, axB.get_position().x0, top + 0.1 / H, 'B')
    fig.legend([Patch(fc=USED, ec='0.6', lw=0.4), Patch(color=OUTC), Patch(color=SELF), Patch(color=AFR), Patch(color=EUR)],
               ['Pair used', 'Pair of relatives left out', 'Individual with themselves', 'African segment',
                'European segment'], loc='center', ncol=5, frameon=False, fontsize=6.8, handlelength=1.2,
               columnspacing=1.3, bbox_to_anchor=(0.5, fy(2.225)))
    K, KL, rel = calm_sample()
    off = ~np.eye(8, dtype=bool)
    E = 0.5 * K + 0.35 * KL
    E[rel] += 0.3
    np.fill_diagonal(E, 1.0)
    letter(fig, 0.02, fy(2.7), 'C')
    row_of_matrices(fig, 0.08, fy(3.55), 0.86, 0.8 / H,
                    [(E, CM_Y, rel, True), (np.where(off, K, 0), CM_SNP, rel, False, off),
                     (np.where(off, KL, 0), CM_LA, rel, False, off), (np.eye(8), CM_Y, None, True, ~off)],
                    [r'$\approx\;\sigma_g^2\,\times$', r'$+\;\sigma_{\mathrm{LA}}^2\times$', r'$+\;\sigma_e^2\,\times$'],
                    ['phenotype products', 'SNP similarity', 'local-ancestry similarity', 'same individual'])
    axD = fig.add_axes([0, 0, 1, DH / H])
    panel_d(axD)
    letter(fig, 0.02, (DH - 0.18) / H, 'D')
    save(fig)


if __name__ == '__main__':
    figure1()
