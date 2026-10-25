"""Summarize the final related-individuals benchmark and draw the wiki figures (run in ~/summit_relatives on veena).

Inputs: final/<pop>/meta.tsv, final/<pop>/fit/<arm>_<njack>_<chunk>.results.tsv (h2) and
final/<pop>/rg/fit/<arm>_<njack>_<pair>.log (rg). Truth for every arm is the simulated h2 of the unadjusted
genotypes (0.25, or 0.10 in scenario D) and rg 0.5.
Outputs in final/summary/: h2_summary.tsv, rg_summary.tsv, efficiency.tsv, and the figures
related_benchmark_h2.{png,pdf} and related_benchmark_power.{png,pdf} in the style of the SUMMIT paper figures.
usage: python summarize_final.py
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

POPS = ('EUR', 'SAS', 'AFR')
SCEN = list('ABCDE')
ARMS = ('unrel', 'naive', 'pairexcl')
LABEL = {'unrel': 'Unrelated subset', 'naive': 'Everyone, related pairs kept', 'pairexcl': 'Everyone, related pairs excluded',
         'pairexcl_noV': 'Everyone, pairs excluded, no V'}
COLOR = {'unrel': mpl.colormaps['tab20'].colors[0], 'naive': mpl.colormaps['tab20'].colors[2],
         'pairexcl': mpl.colormaps['tab20'].colors[4], 'pairexcl_noV': mpl.colormaps['tab20'].colors[8]}
MARKER = {'unrel': 's', 'naive': 'D', 'pairexcl': 'o', 'pairexcl_noV': '^'}
TCRIT = {'200': stats.t.ppf(0.975, 199), 'chr': stats.t.ppf(0.975, 21)}
OUT = 'final/summary'
os.makedirs(OUT, exist_ok=True)


def set_style():
    mpl.rcParams.update({
        'figure.dpi': 140, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans', 'font.size': 12.0,
        'axes.titlesize': 14.0, 'axes.labelsize': 13.2, 'axes.linewidth': 1.15, 'xtick.labelsize': 12.0,
        'ytick.labelsize': 12.0, 'xtick.major.width': 1.05, 'ytick.major.width': 1.05, 'xtick.major.size': 4.0,
        'ytick.major.size': 4.0, 'grid.color': '#D9D9D9', 'grid.linewidth': 0.8, 'grid.alpha': 0.8,
        'axes.facecolor': 'white', 'figure.facecolor': 'white', 'legend.frameon': False, 'legend.fontsize': 11.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#5A5A5A')
    ax.spines['bottom'].set_color('#5A5A5A')
    ax.tick_params(colors='#333333')
    ax.grid(axis='y')
    ax.set_axisbelow(True)


def load_h2():
    rows = []
    for pop in POPS:
        meta = pd.read_csv(f'final/{pop}/meta.tsv', sep='\t')
        meta['phen'] = 'y' + meta['col'].astype(str)
        for f in glob.glob(f'final/{pop}/fit/*_*_*.results.tsv'):
            arm, nj, _ = re.match(r'(.+)_(200|chr)_(\d+)\.results\.tsv', os.path.basename(f)).groups()
            d = pd.read_csv(f, sep='\t')[['phen', 'h2', 'h2_se']]
            d['phen'] = d['phen'].astype(str).str.replace(r'\.tsv(\.gz)?$', '', regex=True)
            rows.append(d.assign(pop=pop, arm=arm, njack=nj))
    d = pd.concat(rows)
    meta = pd.concat([pd.read_csv(f'final/{p}/meta.tsv', sep='\t').assign(pop=p) for p in POPS])
    meta['phen'] = 'y' + meta['col'].astype(str)
    d = d.merge(meta[['pop', 'phen', 'scenario', 'rep', 'h2g']], on=['pop', 'phen'], how='left')
    assert d.scenario.notna().all()
    w = d.pivot_table(index=['pop', 'arm', 'phen', 'scenario', 'rep', 'h2g'], columns='njack',
                      values=['h2', 'h2_se']).reset_index()
    w.columns = ['_'.join(c).strip('_') for c in w.columns]
    w['h2'] = w['h2_200']
    return w


def summarize_h2(w):
    out = []
    for (pop, arm, sc), g in w.groupby(['pop', 'arm', 'scenario']):
        truth = float(g.h2g.iloc[0]); n = len(g)
        row = dict(pop=pop, arm=arm, scenario=sc, n=n, truth=truth, mean=g.h2.mean(), bias=g.h2.mean() - truth,
                   mcse=g.h2.std(ddof=1) / np.sqrt(n), sd=g.h2.std(ddof=1))
        for nj in ('200', 'chr'):
            se = g[f'h2_se_{nj}']
            row[f'mean_se_{nj}'] = se.mean()
            row[f'se_over_sd_{nj}'] = se.mean() / row['sd']
            row[f'coverage_{nj}'] = float(np.mean(np.abs(g.h2 - truth) <= TCRIT[nj] * se))
            row[f'power_{nj}'] = float(np.mean(g.h2 / se > TCRIT[nj]))
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(f'{OUT}/h2_summary.tsv', sep='\t', index=False, float_format='%.5f')
    eff = []
    for (pop, sc), g in s.groupby(['pop', 'scenario']):
        u = g.set_index('arm')
        if 'unrel' in u.index and 'pairexcl' in u.index:
            eff.append(dict(pop=pop, scenario=sc, var_ratio=(u.loc['unrel', 'sd'] / u.loc['pairexcl', 'sd']) ** 2,
                            se_ratio=u.loc['pairexcl', 'mean_se_200'] / u.loc['unrel', 'mean_se_200']))
    e = pd.DataFrame(eff)
    e.to_csv(f'{OUT}/efficiency.tsv', sep='\t', index=False, float_format='%.4f')
    return s, e


def load_rg():
    rows = []
    pat = re.compile(r'Total genetic correlation \(rg\): (\S+) \(SE: (\S+)\)')
    patg = re.compile(r'Total genetic covariance \(gamma_g\): (\S+) \(SE: (\S+)\)')
    for pop in POPS:
        for f in glob.glob(f'final/{pop}/rg/fit/*_*_*.log'):
            arm, nj, r = re.match(r'(.+)_(200|chr)_(\d+)\.log', os.path.basename(f)).groups()
            txt = open(f).read()
            m, mg = pat.search(txt), patg.search(txt)
            if m and mg:
                rows.append(dict(pop=pop, arm=arm, njack=nj, pair=int(r), rg=float(m.group(1)), se=float(m.group(2)),
                                 gcov=float(mg.group(1)), gse=float(mg.group(2))))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    out = []
    for (pop, arm, nj), g in d.groupby(['pop', 'arm', 'njack']):
        n = len(g)
        out.append(dict(pop=pop, arm=arm, njack=nj, n=n, mean=g.rg.mean(), bias=g.rg.mean() - 0.5,
                        mcse=g.rg.std(ddof=1) / np.sqrt(n), sd=g.rg.std(ddof=1), mean_se=g.se.mean(),
                        se_over_sd=g.se.mean() / g.rg.std(ddof=1),
                        coverage=float(np.mean(np.abs(g.rg - 0.5) <= TCRIT[nj] * g.se)),
                        gcov_bias=g.gcov.mean() - 0.125, gcov_mcse=g.gcov.std(ddof=1) / np.sqrt(n),
                        gcov_coverage=float(np.mean(np.abs(g.gcov - 0.125) <= TCRIT[nj] * g.gse))))
    s = pd.DataFrame(out)
    s.to_csv(f'{OUT}/rg_summary.tsv', sep='\t', index=False, float_format='%.5f')
    return s


def panel_label(fig, ax, letter):
    pos = ax.get_position()
    fig.text(pos.x0 - 0.055, pos.y1 + 0.012, letter, fontsize=18, fontweight='bold', ha='left', va='bottom')


def figure_h2(s, rg):
    """Rows: bias of h2 by scenario, coverage of 95% intervals by scenario; columns EUR, SAS, AFR; plus rg panels."""
    set_style()
    arms = [a for a in ARMS if a in set(s.arm)]
    ncol = 3
    fig, axes = plt.subplots(3 if rg is not None else 2, ncol, figsize=(13.5, 10.8 if rg is not None else 7.6),
                             gridspec_kw=dict(hspace=0.62, wspace=0.18), sharey='row')
    off = np.linspace(-0.22, 0.22, len(arms))
    x = np.arange(len(SCEN))
    for c, pop in enumerate(POPS):
        ax = axes[0, c]
        for a, o in zip(arms, off):
            g = s[(s['pop'] == pop) & (s.arm == a)].set_index('scenario').reindex(SCEN)
            ax.errorbar(x + o, g.bias, yerr=1.96 * g.mcse, fmt=MARKER[a], color=COLOR[a], ms=6.5, capsize=3,
                        lw=1.3, label=LABEL[a])
        ax.axhline(0, color='#888888', ls='--', lw=1)
        ax.set_xticks(x, SCEN); ax.set_title(pop); style_axis(ax)
        if c == 0:
            ax.set_ylabel('Bias in $h^2$')
        ax = axes[1, c]
        for a, o in zip(arms, off):
            g = s[(s['pop'] == pop) & (s.arm == a)].set_index('scenario').reindex(SCEN)
            cov = g['coverage_200']
            err = 1.96 * np.sqrt(cov * (1 - cov) / g.n)
            ax.errorbar(x + o, cov, yerr=err, fmt=MARKER[a], color=COLOR[a], ms=6.5, capsize=3, lw=1.3)
        ax.axhline(0.95, color='#888888', ls='--', lw=1)
        ax.set_xticks(x, SCEN); ax.set_ylim(-0.03, 1.03); style_axis(ax)
        ax.set_xlabel('Scenario')
        if c == 0:
            ax.set_ylabel('Coverage of 95% CI')
    if rg is not None:
        for c, pop in enumerate(POPS):
            ax = axes[2, c]
            g = rg[(rg['pop'] == pop) & (rg.njack == '200')].set_index('arm').reindex(arms)
            xx = np.arange(len(arms))
            for i, a in enumerate(arms):
                ax.errorbar(i, g.loc[a, 'gcov_bias'], yerr=1.96 * g.loc[a, 'gcov_mcse'], fmt=MARKER[a], color=COLOR[a],
                            ms=7, capsize=3, lw=1.3)
                ax.text(i, g.loc[a, 'gcov_bias'] + 1.96 * g.loc[a, 'gcov_mcse'] + 0.004, f"{g.loc[a, 'gcov_coverage']:.2f}",
                        ha='center', va='bottom', fontsize=10.5, color='#333333')
            ax.axhline(0, color='#888888', ls='--', lw=1)
            ax.set_xticks(xx, ['Unrelated', 'Kept', 'Excluded'][:len(arms)]); style_axis(ax)
            ax.set_xlim(-0.6, len(arms) - 0.4)
            if c == 0:
                ax.set_ylabel('Bias in genetic\ncovariance')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(arms), bbox_to_anchor=(0.5, 0.995))
    for r, letter in zip(range(axes.shape[0]), 'ABC'):
        panel_label(fig, axes[r, 0], letter)
    fig.savefig(f'{OUT}/related_benchmark_h2.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/related_benchmark_h2.pdf', bbox_inches='tight')
    plt.close(fig)


def figure_power(s, e):
    """(A) sampling variance of the unrelated-subset estimate relative to pair exclusion; (B) power to detect h2 > 0
    at the 5% level, from each arm's empirical sampling SD pooled over scenarios A-E (500 replicates)."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.4), gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    x = np.arange(len(POPS))
    for i, pop in enumerate(POPS):
        g = e[e['pop'] == pop]
        ax.scatter(np.full(len(g), i) + np.linspace(-0.16, 0.16, len(g)), g.var_ratio, color=COLOR['pairexcl'],
                   marker='o', s=34, alpha=0.55, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [g.var_ratio.mean()] * 2, color=COLOR['pairexcl'], lw=2.4, zorder=3)
    ax.axhline(1, color='#888888', ls='--', lw=1)
    ax.set_xticks(x, POPS); ax.set_ylabel('Variance, unrelated subset /\nrelated pairs excluded'); style_axis(ax)
    ax = axes[1]
    h = np.linspace(0, 0.2, 201)
    ls = {'EUR': '-', 'SAS': '--', 'AFR': ':'}
    for pop in POPS:
        for a in ('unrel', 'pairexcl'):
            sd = np.sqrt((s[(s['pop'] == pop) & (s.arm == a)].sd ** 2).mean())
            ax.plot(h, stats.norm.sf(1.645 - h / sd), color=COLOR[a], ls=ls[pop], lw=2.0)
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=COLOR[a], lw=2, label=LABEL[a]) for a in ('unrel', 'pairexcl')]
    hl += [Line2D([], [], color='#555555', ls=ls[p], lw=2, label=p) for p in POPS]
    ax.legend(handles=hl, loc='center left', bbox_to_anchor=(1.03, 0.5), fontsize=10.5)
    ax.set_xticks(np.arange(0, 0.201, 0.05))
    ax.set_xlabel('True $h^2$'); ax.set_ylabel('Power, one-sided 5% test'); ax.set_ylim(0, 1.02); style_axis(ax)
    panel_label(fig, axes[0], 'A'); panel_label(fig, axes[1], 'B')
    fig.savefig(f'{OUT}/related_benchmark_power.png', bbox_inches='tight')
    fig.savefig(f'{OUT}/related_benchmark_power.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    w = load_h2()
    s, e = summarize_h2(w)
    rg = load_rg()
    pd.set_option('display.width', 220)
    cols = ['pop', 'arm', 'scenario', 'n', 'truth', 'mean', 'bias', 'mcse', 'sd', 'mean_se_200', 'se_over_sd_200',
            'coverage_200', 'mean_se_chr', 'se_over_sd_chr', 'coverage_chr', 'power_200']
    print(s[cols].round(4).to_string(index=False))
    print(e.round(3).to_string(index=False))
    if rg is not None:
        print(rg.round(4).to_string(index=False))
    figure_h2(s[s.arm.isin(ARMS)], None if rg is None else rg[rg.arm.isin(ARMS)])
    figure_power(s, e)
    print('figures written to', OUT)
