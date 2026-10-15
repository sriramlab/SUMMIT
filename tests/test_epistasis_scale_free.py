"""Scale-free focal-context x background U-statistics against dense oracles."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import norm, rankdata

from summit.epistasis.scale_free.design import Context, Outcome, build_plan
from summit.epistasis.scale_free.stats import hoeffding, null_covariance, summarize
from summit.epistasis.scale_free.stream import accumulate, background_axis, checked_gram
from summit.epistasis.scale_free.transforms import blom_rank_normal, residualize_within
from summit.prediction.genotype import ArrayGenotypeSource, estimate_scale
from summit.prediction.spec import VariantAxis

from prediction_helpers import prediction_threads


def _native_available():
    try:
        from summit import gxeldcore  # noqa: F401
        return True
    except ImportError:
        return False


BACKENDS = [False] + ([True] if _native_available() else [])


def _threads(native):
    if not native:
        return 1
    from epistasis_helpers import epistasis_threads
    return epistasis_threads()


def simulated(seed=3, n=900, m=90):
    rng = np.random.default_rng(seed)
    freq = rng.uniform(.1, .5, m)
    raw = rng.binomial(2, freq, (n, m)).astype(np.int8)
    raw[rng.random((n, m)) < .01] = -127
    chrom = tuple(str(c) for c in np.repeat([1, 2, 3], m // 3))
    axis = VariantAxis(tuple(f'v{i}' for i in range(m)), chrom, tuple(range(1, m + 1)), ('A',) * m, ('G',) * m)
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis, hard_calls=True)
    focal = np.minimum(np.where(raw[:, 0] == -127, 0, raw[:, 0]), 2)
    other = rng.integers(0, 3, n)
    x = np.where(raw == -127, np.nan, raw.astype(float))
    x = (x - np.nanmean(x, 0)) / np.nanstd(x, 0)
    x[np.isnan(x)] = 0
    beta, gamma = rng.normal(0, .15, m), rng.normal(0, .15, m)
    beta[:30] = gamma[:30] = 0
    latent = x @ beta + (focal - 1) * (x @ gamma) + .4 * focal + rng.normal(size=n)
    y = np.exp(.5 * latent)
    y2 = rng.normal(size=n) + .2 * other
    y2[rng.random(n) < .05] = np.nan
    cov = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n)])
    ctx = [Context('focal', focal, {0: 0., 1: 1., 2: 2.}, exclude_chromosomes=('1',)),
           Context('other', other, {0: 0., 1: 1., 2: 2.}, low=(0, 1), high=(2,), exclude_chromosomes=('2',))]
    out = [Outcome('y', y, positive=True), Outcome('y2', y2)]
    return source, raw, ctx, out, cov


def subset(rows, ctx, out, cov):
    from dataclasses import replace
    return ([replace(c, labels=c.labels[rows]) for c in ctx], [replace(o, values=o.values[rows]) for o in out],
            cov[rows])


def dense_scores(raw, scale, plan):
    """Independent class-normalized group scores (G, M, q) from the full matrix."""
    x = np.where(raw == -127, np.nan, raw.astype(float))
    x = (x - scale.mean) * scale.inverse_scale
    x[np.isnan(x)] = 0
    x = x[plan.rows]
    y = np.zeros((len(plan.rows), plan.n_columns))
    y[plan.order] = np.vstack(plan.blocks)
    s = np.stack([x[plan.groups == g].T @ y[plan.groups == g] for g in range(plan.n_groups)])
    return s / plan.counts[:, None, :]


def oracle_kernel(s, c1, c2, w):
    return np.einsum('gj,hj,j->gh', s[:, :, c1], s[:, :, c2], w)


@pytest.mark.parametrize('native', BACKENDS)
def test_stream_kernels_match_dense_oracle(native):
    source, raw, ctx, out, cov = simulated()
    rows = np.flatnonzero(np.arange(len(source.samples)) % 7 != 3)
    ctx, out, cov = subset(rows, ctx, out, cov)
    plan = build_plan(rows, ctx, out, cov, scales=('raw', 'within_rank'), pattern_scales=('within_rank',),
                      additive_scales=('raw',), n_groups=6, seed=5)
    annotation = (np.arange(90) % 4 == 0).astype(float)
    bg = background_axis(source, annotations=annotation[:, None], names=('A',), n_blocks=9)
    scale = estimate_scale(source, rows, bg.variants, threads=_threads(native))
    acc = accumulate(source, plan, scale, bg, block_size=17, threads=_threads(native), native=native, batch=3)
    s = dense_scores(np.asarray(source.values), scale, plan)
    chrom = np.array(bg.chromosomes)
    for pair, (k1, k2) in (('gy', ('gy', 'gy')), ('ll', ('lo', 'lo')), ('hh', ('hi', 'hi')), ('lh', ('lo', 'hi')),
                           ('yy', ('y', 'y'))):
        first, second = plan.kinds[k1], plan.kinds[k2]
        for p, (a, _) in enumerate(plan.keys[k1]):
            keep = ~np.isin(chrom, plan.analyses[a]['exclude_chromosomes'])
            for i, w in enumerate((np.ones(90), annotation)):
                expected = oracle_kernel(s, first.start + p, second.start + p, w * keep)
                np.testing.assert_allclose(acc.kernels[pair][i, p], expected, rtol=1e-10, atol=1e-13)
                off = np.array([np.sum(oracle_kernel(s, first.start + p, second.start + p, w * keep * (bg.blocks == b)))
                                - np.trace(oracle_kernel(s, first.start + p, second.start + p,
                                                         w * keep * (bg.blocks == b))) for b in range(bg.n_blocks)])
                np.testing.assert_allclose(acc.between[pair][i, p], off, rtol=1e-9, atol=1e-13)
    assert acc.diagnostics['kernel_checksum_repairs'] == 0


def prototype_hoeffding(t):
    """Reference formula from the frozen prototype (explicit off-diagonal masks)."""
    g = t.shape[1]
    off = ~np.eye(g, dtype=bool)
    flat = t[:, off]
    theta = flat.mean(1)
    e2 = flat @ flat.T / flat.shape[1]
    rows = t.sum(2)
    sq = np.einsum('mgh,ngh->mng', t, t).sum(2)
    e1 = (np.einsum('mg,ng->mn', rows, rows) - sq) / (g * (g - 1) * (g - 2))
    tot = t.sum((1, 2))
    tt = (np.outer(tot, tot) - 4 * np.einsum('mg,ng->mn', rows, rows) + 2 * sq) / (g * (g - 1) * (g - 2) * (g - 3))
    return theta, 4 * (g - 2) / (g * (g - 1)) * (e1 - tt) + 2 / (g * (g - 1)) * (e2 - tt)


def test_hoeffding_matches_prototype_and_simulated_variance():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(3, 12, 12))
    t = (t + t.transpose(0, 2, 1)) / 2
    for k in t:
        np.fill_diagonal(k, 0)
    theta, cov = hoeffding(t)
    ref_theta, ref_cov = prototype_hoeffding(t)
    np.testing.assert_allclose(theta, ref_theta, rtol=1e-13)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-10, atol=1e-15)
    # U-statistic of i.i.d. unit scores: kernel a_g a_h has variance known in closed form
    # (unbiased per replicate, so single estimates may be negative; compare means)
    g, reps, estimates, variances = 30, 3000, [], []
    for _ in range(reps):
        a = rng.normal(.5, 1, g)
        k = np.outer(a, a)
        np.fill_diagonal(k, 0)
        th, cv = hoeffding(k[None])
        estimates.append(th[0])
        variances.append(cv[0, 0])
    assert abs(np.var(estimates) / np.mean(variances) - 1) < .1


def test_statistics_match_direct_definitions():
    source, raw, ctx, out, cov = simulated(seed=11)
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx, out, cov, scales=('within_rank',), additive_scales=('within_rank',), n_groups=8,
                      seed=2)
    bg = background_axis(source, n_blocks=6)
    scale = estimate_scale(source, rows, bg.variants)
    acc = accumulate(source, plan, scale, bg, block_size=32, native=False)
    records = {(r['analysis'], r['statistic']): r for r in summarize(plan, acc)}
    s = dense_scores(np.asarray(source.values), scale, plan)
    chrom = np.array(bg.chromosomes)
    g = plan.n_groups

    def theta(c1, c2, w):
        k = oracle_kernel(s, c1, c2, w) / w.sum()
        k = (k + k.T) / 2
        return (k.sum() - np.trace(k)) / (g * (g - 1))

    for p, (a, _) in enumerate(plan.keys['gy']):
        w = (~np.isin(chrom, plan.analyses[a]['exclude_chromosomes'])).astype(float)
        gy, lo, hi = plan.kinds['gy'].start + p, plan.kinds['lo'].start + p, plan.kinds['hi'].start + p
        assert np.isclose(records[(a, 'dosage')]['estimate'], theta(gy, gy, w), rtol=1e-10)
        yc = plan.kinds['y'].start + p
        assert np.isclose(records[(a, 'additive')]['estimate'], theta(yc, yc, w), rtol=1e-10)
        ty = theta(yc, yc, w)
        assert np.isclose(records[(a, 'relative_dosage')]['estimate'], theta(gy, gy, w) / ty if ty > 0 else np.nan,
                          rtol=1e-8, equal_nan=True)
        ratio = plan.analyses[a]['global_ratio']['within_rank']
        assert np.isclose(records[(a, 'global_excess')]['estimate'], theta(gy, gy, w) - ratio * theta(yc, yc, w),
                          rtol=1e-8, atol=1e-16)
        ll, hh, lh = theta(lo, lo, w), theta(hi, hi, w), theta(lo, hi, w)
        expected_rho = lh / np.sqrt(ll * hh) if ll > 0 and hh > 0 else np.nan
        assert np.isclose(records[(a, 'rho')]['estimate'], expected_rho, rtol=1e-9, equal_nan=True)
        assert np.isclose(records[(a, 'delta2')]['estimate'], ll + hh - 2 * lh, rtol=1e-9, atol=1e-15)
        # SNP-block jackknife recomputed from scratch for the dosage statistic
        drops = []
        for b in range(bg.n_blocks):
            wb = w * (bg.blocks != b)
            if wb.sum() < w.sum():
                drops.append(theta(gy, gy, wb))
        drops = np.array(drops)
        se = np.sqrt((len(drops) - 1) / len(drops) * np.sum((drops - drops.mean()) ** 2))
        assert np.isclose(records[(a, 'dosage')]['se_jackknife'], se, rtol=1e-8)
        # null-restricted SE recomputed from the dense oracle kernel: Var0 = 2 mean_{a != b} T^2 / (G (G - 1))
        r = records[(a, 'dosage')]
        k = oracle_kernel(s, gy, gy, w) / w.sum()
        k = (k + k.T) / 2
        np.fill_diagonal(k, 0)
        se0 = np.sqrt(2 * np.sum(k ** 2) / (g * (g - 1)) ** 2)
        assert np.isclose(r['se_null'], se0, rtol=1e-9)
        assert r['se'] == r['se_null'] and r['se_rule'] == 'null_restricted'
        assert np.isclose(r['z'], r['estimate'] / se0, rtol=1e-9)
        r = records[(a, 'rho')]
        assert r['se_rule'] == 'max'
        assert np.isclose(r['se'], np.nanmax([r['se_hoeffding'], r['se_jackknife']]), equal_nan=True)


def test_within_rank_columns_invariant_to_class_specific_monotone_maps():
    source, raw, ctx, out, cov = simulated(seed=4)
    rng = np.random.default_rng(9)
    u = rng.normal(size=len(source.samples))
    k = ctx[0].labels
    distorted = np.select([k == 0, k == 1, k == 2], [u, 3 + 2 * u, np.exp(1.5 * u)])
    outs = [Outcome('u', u), Outcome('h', distorted)]
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx[:1], outs, np.zeros((len(rows), 0)), scales=('pooled_rank', 'within_rank'),
                      n_groups=6)
    by = {(plan.analyses[c['analysis']]['outcome'], c['scale'], c['kind']): j for j, c in enumerate(plan.columns)}
    y = np.vstack(plan.blocks)
    for kind in ('gy', 'lo', 'hi'):
        np.testing.assert_array_equal(y[:, by[('u', 'within_rank', kind)]], y[:, by[('h', 'within_rank', kind)]])
    assert not np.allclose(y[:, by[('u', 'pooled_rank', 'gy')]], y[:, by[('h', 'pooled_rank', 'gy')]])


def test_transforms():
    y = np.array([3., 1., 2., 2., 5., 4.])
    strata = np.array([0, 0, 0, 1, 1, 1])
    z = blom_rank_normal(y)
    np.testing.assert_allclose(z, norm.ppf((rankdata(y) - .375) / 6.25))
    zs = blom_rank_normal(y, strata)
    for s in (0, 1):
        m = strata == s
        np.testing.assert_allclose(zs[m], norm.ppf((rankdata(y[m]) - .375) / 3.25))
    rng = np.random.default_rng(0)
    c = rng.normal(size=(6, 1))
    r = residualize_within(y, c, strata)
    for s in (0, 1):
        m = strata == s
        a = np.column_stack([np.ones(3), c[m]])
        np.testing.assert_allclose(r[m], y[m] - a @ np.linalg.lstsq(a, y[m], rcond=None)[0], atol=1e-12)


def test_checksum_repairs_and_rejects(monkeypatch):
    rng = np.random.default_rng(0)
    left = rng.normal(size=(4, 9, 50))
    left[:, -1] = left[:, :-1].sum(1)
    right = rng.normal(size=(4, 9, 50))
    right[:, -1] = right[:, :-1].sum(1)
    exact, repaired = checked_gram(left, right)
    assert repaired == 0
    np.testing.assert_allclose(exact, np.matmul(left, right.transpose(0, 2, 1))[:, :-1, :-1])
    real = np.matmul

    def faulty(a, b, *args, **kw):
        r = real(a, b, *args, **kw)
        if r.ndim == 3 and r.shape[1:] == (9, 9):
            r[2, 3, 4] += 1e-3 * abs(r[2, 3, 4]) + 1e-6
        return r
    monkeypatch.setattr(np, 'matmul', faulty)
    fixed, repaired = checked_gram(left, right)
    assert repaired == 1
    np.testing.assert_allclose(fixed, exact, rtol=1e-12)
    monkeypatch.setattr(np, 'dot', lambda a, b: real(a, b) + 1.)
    with pytest.raises(RuntimeError):
        checked_gram(left, right)


def test_group_and_class_checks():
    source, raw, ctx, out, cov = simulated()
    rows = np.arange(len(source.samples))
    with pytest.raises(ValueError, match='groups have fewer'):
        build_plan(rows, ctx, out, cov, n_groups=400)
    with pytest.raises(ValueError, match='strictly increasing'):
        build_plan(rows[::-1], ctx, out, cov)


def test_cli_end_to_end(tmp_path):
    if not _native_available():
        pytest.skip('native extension is not loaded')
    from bed_reader import to_bed
    from epistasis_helpers import entrypoint as main
    source, raw, ctx, out, cov = simulated(seed=21, n=700, m=60)
    n, m = raw.shape
    ids = [str(i) for i in range(n)]
    values = np.where(raw == -127, np.nan, raw.astype(float))
    to_bed(tmp_path / 'input.bed', values, properties=dict(
        fid=ids, iid=ids, sid=[f'v{i}' for i in range(m)], chromosome=list(source.variants.chromosome),
        bp_position=np.arange(1, m + 1), allele_1=['A'] * m, allele_2=['G'] * m))
    (tmp_path / 'samples.tsv').write_text('FID IID\n' + ''.join(f'{i} {i}\n' for i in range(n)))
    y = out[0].values
    (tmp_path / 'pheno.tsv').write_text('FID IID y sex\n' + ''.join(
        f'{i} {i} {y[i]} {int(cov[i, 1])}\n' for i in range(n)))
    (tmp_path / 'cov.tsv').write_text('FID IID c1 batch\n' + ''.join(
        f'{i} {i} {cov[i, 0]} b{i % 3}\n' for i in range(n)))
    (tmp_path / 'ctx.tsv').write_text('FID IID k\n' + ''.join(f'{i} {i} {ctx[1].labels[i]}\n' for i in range(n)))
    (tmp_path / 'ann.tsv').write_text('SNP A\n' + ''.join(f'v{j} {int(j % 3 == 0)}\n' for j in range(m)))
    manifest = dict(kind='summit.epistasis.scale_free', schema_version=1, genotypes={'geno': 'input.bed'},
                    samples='samples.tsv', phenotypes={'file': 'pheno.tsv', 'columns': ['y'], 'positive': ['y']},
                    covariates={'file': 'cov.tsv', 'columns': ['c1'], 'categorical': ['batch']},
                    strata={'file': 'pheno.tsv', 'column': 'sex'},
                    contexts=[{'name': 'v0', 'variant': 'v0', 'min_minor_homozygotes': 10},
                              {'name': 'other', 'file': 'ctx.tsv', 'column': 'k', 'values': {'0': 0, '1': 1, '2': 2},
                               'low': [0, 1], 'high': [2], 'exclude_chromosomes': ['2']}],
                    background={'annotations': {'file': 'ann.tsv', 'columns': ['A']}, 'jackknife_blocks': 6},
                    scales=['pooled_rank', 'within_rank'], additive_scales=['pooled_rank'], groups=5,
                    save_kernels=True)
    (tmp_path / 'm.json').write_text(json.dumps(manifest))
    assert main(['epistasis', 'scale-free', str(tmp_path / 'm.json'), '--out', str(tmp_path / 'out'),
                 '--block-size', '16', '--memory-gib', '1']) == 0
    import pandas as pd
    res = pd.read_csv(tmp_path / 'out' / 'results.tsv', sep='\t')
    assert set(res.statistic) >= {'dosage', 'dosage[A]', 'dosage_diff[A]', 'rho', 'log_amp', 'delta2',
                                  'sel_lam[A]', 'sel_amp[A]'}
    assert len(res) == 2 * (8 + 3 + 5 + 5)
    # the SE rule survives a round trip through pandas (whose default NA tokens include 'null')
    assert res.se_rule.notna().all() and set(res.se_rule) == {'null_restricted', 'max'}
    degenerate = res.statistic.str.split('[').str[0].isin(['dosage', 'delta2'])
    assert (res.se_rule[degenerate] == 'null_restricted').all() and np.allclose(res.se[degenerate],
                                                                                res.se_null[degenerate])
    assert {'additive', 'relative_dosage', 'global_excess', 'additive[A]'} <= set(res.statistic)
    flipped = res.statistic.str.split('[').str[0].isin(['dosage', 'delta2', 'relative_dosage'])
    assert set(res.p_method[flipped]) == {'signflip'} and set(res.p_method[~flipped]) == {'normal'}
    assert np.allclose(res.p[~flipped], res.p_normal[~flipped], equal_nan=True)
    run = json.loads((tmp_path / 'out' / 'run.json').read_text())
    assert run['contexts']['v0']['chromosome'] == '1'
    assert run['covariate_slopes'] == 'common' and run['null_p'] == 'signflip'
    assert (tmp_path / 'out' / 'COMPLETE').exists()
    assert np.load(tmp_path / 'out' / 'kernels.npz')['kernel_gy'].shape == (2, 4, 5, 5)


def _pair_data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    x, z = rng.binomial(2, .35, n), rng.binomial(2, .3, n)
    cov = rng.normal(size=(n, 2))
    latent = .3 * x + .2 * z + .15 * x * z + cov @ [.2, -.1] + rng.normal(size=n)
    return np.exp(.7 * latent), x, z, cov, rng.integers(0, 2, n)


def test_cpm_two_bins_is_logistic_regression():
    from summit.epistasis.scale_free.rank_pairs import cpm_fit
    y, x, z, cov, _ = _pair_data(1)
    X = np.column_stack([x - x.mean(), z - z.mean(), cov])
    c = (y > np.median(y)).astype(np.int64)
    fit = cpm_fit(c, X, 'logit')
    A = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(A.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ b))
        b += np.linalg.solve(A.T @ (A * (p * (1 - p))[:, None]), A.T @ (c - p))
    assert fit['converged']
    np.testing.assert_allclose(fit['beta'], b[1:], rtol=1e-6, atol=1e-8)
    p = 1 / (1 + np.exp(-A @ b))
    np.testing.assert_allclose(fit['cov'], np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]))[1:, 1:], rtol=1e-5)


def test_pair_battery_rank_statistics_invariant_to_monotone_maps():
    from summit.epistasis.scale_free.rank_pairs import pair_battery
    y, x, z, cov, sex = _pair_data(2)
    a = pair_battery(y, x, z, cov, strata=sex, links=('logit', 'cloglog'), boot=20)
    b = pair_battery(np.log(y) ** 3 + 7, x, z, cov, strata=sex, links=('logit', 'cloglog'), boot=20)
    for key in ('cpm_logit_z', 'cpm_cloglog_z', 'strat_byz_trend_z', 'strat_byx_trend_z', 'auc_trend_z',
                'R_star', 'R_star_lower'):
        assert np.isclose(a[key], b[key], rtol=1e-6), key
    assert a['cpm_logit_z'] > 3 and not np.isclose(a['hc3_raw_z'], b['hc3_raw_z'])


def test_global_ratio_from_class_spreads():
    rng = np.random.default_rng(3)
    n = 30000
    k = rng.binomial(2, .3, n)
    u = rng.normal(size=n)
    y = 1 + .4 * k + np.exp(.3 * k) * u
    ctx = [Context('k', k, {0: 0., 1: 1., 2: 2.})]
    plan = build_plan(np.arange(n), ctx, [Outcome('y', y)], np.zeros((n, 0)), scales=('raw',), n_groups=4)
    g = (k - k.mean()) / k.std()
    s = np.array([y[k == c].std() for c in range(3)])
    pi = np.bincount(k) / n
    gv = np.array([g[k == c][0] for c in range(3)])
    expected = (np.sum(pi * gv * s) / np.sum(pi * s)) ** 2
    assert np.isclose(plan.analyses[0]['global_ratio']['raw'], expected, rtol=1e-10)
    assert abs(expected / (np.sum(pi * gv * np.exp(.3 * np.arange(3))) / np.sum(pi * np.exp(.3 * np.arange(3)))) ** 2
               - 1) < .15


def test_hc3_matches_qr_reference():
    from summit.epistasis.scale_free.rank_pairs import hc3_wald
    rng = np.random.default_rng(4)
    n = 3000
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 4)), rng.integers(0, 2, n)])
    y = X @ [1, .2, 0, -.3, .1, .4] + rng.standard_t(4, n) * (1 + .5 * np.abs(X[:, 1]))
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    e = y - X @ beta
    h = np.sum(Q * Q, axis=1)
    rinv = np.linalg.inv(R)
    bread = rinv @ rinv.T
    s = X * (e / (1 - h))[:, None]
    cov = bread @ (s.T @ s) @ bread
    fit = hc3_wald(y, X, [1, 3])
    np.testing.assert_allclose(fit['beta'], beta, rtol=1e-9)
    np.testing.assert_allclose(fit['cov'], cov, rtol=1e-8)


def test_stratified_rank_test_calibrated_for_rare_homozygotes():
    from summit.epistasis.scale_free.rank_pairs import stratified_rank_test
    rng = np.random.default_rng(7)
    n, p = 60000, []
    for _ in range(30):
        x, z = rng.binomial(2, .02, n), rng.binomial(2, .3, n)
        y = np.exp(.3 * x + .2 * z + rng.normal(size=n))
        p.append(stratified_rank_test(y, x, z, strata=rng.integers(0, 2, n))['omnibus_p'])
    assert min(p) > 1e-3 and np.mean(np.array(p) < .05) <= .15


def test_relative_stratified_test_separates_modifier_from_specific():
    from summit.epistasis.scale_free.rank_pairs import relative_stratified_test, stratified_rank_test
    rng = np.random.default_rng(11)
    n = 120000
    x, z = rng.binomial(2, .3, n), rng.binomial(2, .3, n)
    P = rng.normal(size=n)
    e = rng.normal(size=n)
    # z rescales the noise only: the stratified test rejects, the relative test should not
    y = .25 * x + .3 * P + .2 * z + .8 * e * np.exp(.35 * z)
    assert stratified_rank_test(y, x, z)['omnibus_p'] < 1e-6
    assert relative_stratified_test(y, x, z, P)['p'] > 1e-3
    # z modifies x specifically: both reject
    y2 = (.25 + .15 * z) * x + .3 * P + .2 * z + .8 * e
    assert relative_stratified_test(y2, x, z, P)['p'] < 1e-6


def test_pair_battery_handles_collinear_covariates():
    from summit.epistasis.scale_free.rank_pairs import pair_battery
    y, x, z, cov, sex = _pair_data(3, n=6000)
    centre = np.random.default_rng(1).integers(0, 3, len(y))
    dummies = np.column_stack([centre == 1, centre == 2, centre == 0]).astype(float)   # sums to one
    a = pair_battery(y, x, z, np.column_stack([cov, dummies]), strata=sex, links=('logit',), boot=10)
    b = pair_battery(y, x, z, np.column_stack([cov, dummies[:, :2]]), strata=sex, links=('logit',), boot=10)
    assert a['cpm_logit_converged'] and np.isclose(a['cpm_logit_z'], b['cpm_logit_z'], rtol=1e-6)


def test_target_variance_qtl_biases_iid_residual_fit_and_modifier_squares_removes_it():
    """Same-person terms make the FAME-like moment fit respond to a target variance QTL (theory10 Prop. 8)."""
    from summit.epistasis.prepare import SelectedStudy, fit_scale
    from summit.epistasis.summary import fit_epistasis
    rng = np.random.default_rng(5)
    n, m = 1500, 200
    raw = rng.binomial(2, rng.uniform(.1, .5, m), (n, m)).astype(float)
    raw[:, 0] = rng.binomial(2, .25, n)
    axis = VariantAxis(tuple(f'v{i}' for i in range(m)), ('1',) * m, tuple(range(1, m + 1)), ('A',) * m, ('G',) * m)
    make = lambda: ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis)
    scale = fit_scale(make(), np.arange(n), block_size=64, threads=1)
    x = (raw - scale.mean) * scale.inverse_scale
    g = x[:, 0]
    w = np.ones((m, 2))
    w[0] = 0
    noise = rng.normal(size=n) * np.exp(.6 * g)
    y = (x @ (rng.normal(0, np.sqrt(.4 / m), m) * (np.arange(m) > 0)) + .3 * g + noise * np.sqrt(.6) / noise.std())[:, None]
    fits = {}
    for name, extra in (('iid', {}), ('squares', dict(residual_basis=np.column_stack([g ** 2, np.ones(n)]),
                                                      residual_names=('residual_square:epi', 'residual')))):
        study = SelectedStudy(make(), np.arange(n), scale, fixed_effects=np.column_stack([np.ones(n), g]),
                              modifiers=np.column_stack([np.ones(n), g]), weights=w,
                              component_names=('additive', 'epi'), definitions={'target': 'v0'},
                              backend='numpy', block_size=64, threads=1, **extra)
        summary, *_ = study.summarize(study.reference(exact=True), y, trait_names=('y',))
        fit = fit_epistasis(summary)
        k = list(fit['component_names']).index('epi')
        fits[name] = fit['coefficients'][k] / fit['standard_errors'][k]
    assert fits['iid'] > 3 and abs(fits['squares']) < 2.5


def test_linear_relative_form_handles_qualitative_interaction():
    from summit.epistasis.scale_free.rank_pairs import relative_stratified_test
    rng = np.random.default_rng(12)
    n = 120000
    x, z = rng.binomial(2, .3, n), rng.binomial(2, .3, n)
    P, e = rng.normal(size=n), rng.normal(size=n)
    y = .3 * x * (z > 0) + .3 * P + .2 * z + .8 * e           # x acts only when z > 0
    r = relative_stratified_test(y, x, z, P)
    assert r['p_linear'] < 1e-6
    y0 = .25 * x + .3 * P + .2 * z + .8 * e * np.exp(.35 * z)  # noise-only modifier
    assert relative_stratified_test(y0, x, z, P)['p_linear'] > 1e-3


def test_hc3_tolerates_exactly_fitted_rows():
    from summit.epistasis.scale_free.rank_pairs import hc3_wald
    rng = np.random.default_rng(3)
    n = 500
    single = np.zeros(n)
    single[7] = 1                                   # a dummy that isolates one row: leverage 1
    X = np.column_stack([np.ones(n), rng.normal(size=n), single])
    y = X[:, 1] * .3 + rng.normal(size=n)
    fit = hc3_wald(y, X, [1])
    assert np.isfinite(fit['p']) and np.all(np.isfinite(fit['cov']))


def test_null_covariance_matches_definition():
    rng = np.random.default_rng(4)
    g = 9
    t = rng.normal(size=(3, g, g))
    t = (t + t.transpose(0, 2, 1)) / 2
    for k in t:
        np.fill_diagonal(k, 0)
    off = ~np.eye(g, dtype=bool)
    expected = 2 * np.array([[np.mean(a[off] * b[off]) for b in t] for a in t]) / (g * (g - 1))
    np.testing.assert_allclose(null_covariance(t), expected, rtol=1e-12)
    # the contrast kernel ll + hh - 2 lh has the variance of the linear combination
    c = np.array([1., 1., -2.])
    contrast = np.tensordot(c, t, axes=1)
    np.testing.assert_allclose(c @ null_covariance(t) @ c, null_covariance(contrast[None])[0, 0], rtol=1e-12)


def _ld_genotypes(rng, n, m, chromosomes, rho=.8):
    """Two haplotypes per person from AR(1) Gaussian fields thresholded at allele frequencies."""
    per = m // chromosomes
    freq = rng.uniform(.1, .5, m)
    calls = np.zeros((n, m))
    for _ in range(2):
        z = rng.normal(size=(n, m))
        for j in range(1, m):
            if j % per:
                z[:, j] = rho * z[:, j - 1] + np.sqrt(1 - rho ** 2) * z[:, j]
        calls += z < norm.ppf(freq)
    return calls.astype(np.int8)


def test_null_variance_calibrated_under_no_interaction():
    """Dosage and delta2 z scores with the null-restricted SE under two exact nulls.

    raw scale: the focal variant changes only the noise variance (no interaction);
    within-class ranks: the focal variant shifts and stretches the whole liability.
    """
    rng = np.random.default_rng(17)
    n, m, reps = 1500, 90, 200
    raw = _ld_genotypes(rng, n, m, 3)
    chrom = tuple(str(c) for c in np.repeat([1, 2, 3], m // 3))
    axis = VariantAxis(tuple(f'v{i}' for i in range(m)), chrom, tuple(range(1, m + 1)), ('A',) * m, ('G',) * m)
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis, hard_calls=True)
    x = (raw - raw.mean(0)) / raw.std(0)
    age = rng.normal(size=n)
    outcomes, ctx, analyses = [], [], []
    for r in range(reps):
        # a new focal variant per replicate: the target is the variance over sampled people,
        # not the variance conditional on one fixed set of genotypes
        focal = rng.binomial(2, .3, n)
        spread = np.exp(.2 * (focal - focal.mean()))
        beta = np.zeros(m)
        causal = rng.choice(m, 12, replace=False)
        beta[causal] = rng.normal(0, .2, 12)
        beta[causal[0]] = .6
        noise = rng.standard_t(5, n)
        outcomes.append(Outcome(f'v{r}', x @ beta + .3 * age + spread * noise))
        liability = x @ beta + .3 * age + noise
        outcomes.append(Outcome(f'a{r}', np.exp(.4 * (.2 * focal + spread * liability)), positive=True))
        ctx.append(Context(f'f{r}', focal, {0: 0., 1: 1., 2: 2.}, low=(0,), high=(1, 2), exclude_chromosomes=('1',)))
        analyses += [(f'f{r}', f'v{r}'), (f'f{r}', f'a{r}')]
    rows = np.arange(n)
    plan = build_plan(rows, ctx, outcomes, age[:, None], analyses=analyses, scales=('raw', 'within_rank'),
                      pattern_scales=('within_rank',), n_groups=30, seed=3)
    bg = background_axis(source, n_blocks=12)
    scale = estimate_scale(source, rows, bg.variants)
    acc = accumulate(source, plan, scale, bg, block_size=64, native=False)
    records = summarize(plan, acc)
    sets = {('v', 'raw', 'dosage'): [], ('a', 'within_rank', 'dosage'): [], ('a', 'within_rank', 'delta2'): []}
    for rec in records:
        key = (rec['outcome'][0], rec['scale'], rec['statistic'])
        if key in sets:
            sets[key].append(rec)
    for key, recs in sets.items():
        assert len(recs) == reps
        est = np.array([r['estimate'] for r in recs])
        se0 = np.array([r['se_null'] for r in recs])
        z = est / se0
        assert all(r['se_rule'] == 'null_restricted' and r['se'] == r['se_null'] for r in recs)
        # Monte Carlo SD of these summaries is about 0.07 (z mean, z SD) and 0.1 (variance ratio)
        assert abs(z.mean()) < .25, key
        assert .82 < z.std() < 1.18, key
        assert .7 < np.var(est) / np.mean(se0 ** 2) < 1.3, key
        assert np.all(np.isfinite(se0) & (se0 > 0)), key
        # the full Hoeffding estimate is noisier and can even be negative (reported as NaN)
        hoeff = np.array([r['se_hoeffding'] for r in recs])
        finite = np.isfinite(hoeff)
        assert np.std(np.log(se0)) < np.std(np.log(hoeff[finite])), key


def test_records_choose_the_test_standard_error():
    source, raw, ctx, out, cov = simulated(seed=13)
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx, out, cov, scales=('within_rank',), n_groups=8, seed=2)
    annotation = (np.arange(90) % 3 == 0).astype(float)
    bg = background_axis(source, annotations=annotation[:, None], names=('A',), n_blocks=6)
    scale = estimate_scale(source, rows, bg.variants)
    acc = accumulate(source, plan, scale, bg, block_size=32, native=False)
    for r in summarize(plan, acc):
        base = r['statistic'].split('[')[0]
        if base in ('dosage', 'delta2'):
            assert r['se_rule'] == 'null_restricted' and np.isfinite(r['se_null']) and r['se'] == r['se_null']
        else:
            assert r['se_rule'] == 'max' and np.isnan(r['se_null'])
            finite = [v for v in (r['se_hoeffding'], r['se_jackknife']) if np.isfinite(v)]
            assert r['se'] == max(finite) if finite else np.isnan(r['se'])


def _svd_residualize(values, covariates, strata, rcond=1e-10):
    """The SVD residualization used up to d265bf4, kept as the reference."""
    y2 = values[:, None] if values.ndim == 1 else values
    codes = np.unique(strata, return_inverse=True)[1].reshape(-1)
    out = np.empty_like(y2)
    for s in np.unique(codes):
        rows = np.flatnonzero(codes == s)
        design = np.column_stack([np.ones(len(rows)), covariates[rows]])
        u, sv, _ = np.linalg.svd(design, full_matrices=False)
        basis = u[:, sv > rcond * sv[0]]
        out[rows] = y2[rows] - basis @ (basis.T @ y2[rows])
    return out[:, 0] if values.ndim == 1 else out


def test_residualize_within_matches_svd_reference_on_collinear_covariates():
    rng = np.random.default_rng(8)
    n = 6000
    sex = rng.integers(0, 2, n)
    age = rng.uniform(40, 70, n)
    centre = rng.choice(22, n, p=np.r_[np.full(20, .049), .019, .001])
    dummies = (centre[:, None] == np.arange(1, 22)).astype(float)
    pcs = rng.normal(size=(n, 20)) * np.linspace(5, .5, 20)
    cov = np.column_stack([age, age ** 2, sex, age * sex, pcs, dummies, pcs[:, :1]])
    focal = rng.binomial(2, .3, n)
    strata = focal * 2 + sex
    y = np.column_stack([rng.normal(size=n) + .01 * age, rng.standard_t(3, n)])
    new, ref = residualize_within(y, cov, strata), _svd_residualize(y, cov, strata)
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(residualize_within(y[:, 0], cov, strata), ref[:, 0], rtol=0, atol=1e-9)
    for s in np.unique(strata):
        m = strata == s
        np.testing.assert_allclose(new[m].mean(0), 0, atol=1e-10)
        bound = 1e-9 * np.abs(cov[m]).sum(0)[:, None] * np.abs(new[m]).max(0)
        assert np.all(np.abs(cov[m].T @ new[m]) <= bound)
    with pytest.raises(ValueError, match='no residual degrees of freedom'):
        residualize_within(y[:3], cov[:3, :4], None)


def test_kernel_pool_matches_serial_with_segments_inside_blocks():
    source, raw, ctx, out, cov = simulated(seed=5)
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx, out, cov, scales=('pooled_rank', 'within_rank'), pattern_scales=('within_rank',),
                      additive_scales=('pooled_rank',), n_groups=7, seed=4)
    annotation = (np.arange(90) % 5 < 2).astype(float) * .5
    bg = background_axis(source, annotations=annotation[:, None], names=('A',), n_blocks=11)
    scale = estimate_scale(source, rows, bg.variants)
    serial = accumulate(source, plan, scale, bg, block_size=13, threads=1, native=False, batch=1)
    pooled = accumulate(source, plan, scale, bg, block_size=13, threads=4, native=False, batch=1)
    for p in serial.kernels:
        np.testing.assert_allclose(pooled.kernels[p], serial.kernels[p], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(pooled.between[p], serial.between[p], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(pooled.block_weight, serial.block_weight)
    s = dense_scores(np.asarray(source.values), scale, plan)
    chrom = np.array(bg.chromosomes)
    for p, (a, _) in enumerate(plan.keys['gy']):
        keep = ~np.isin(chrom, plan.analyses[a]['exclude_chromosomes'])
        expected = oracle_kernel(s, plan.kinds['gy'].start + p, plan.kinds['gy'].start + p, annotation * keep)
        np.testing.assert_allclose(pooled.kernels['gy'][1, p], expected, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize('slopes', ['class', 'common'])
def test_covariate_slopes_match_direct_least_squares(slopes):
    source, raw, ctx, out, cov = simulated(seed=9)
    n = len(source.samples)
    rows = np.arange(n)
    sex = np.random.default_rng(2).integers(0, 2, n)
    kwargs = {} if slopes == 'common' else dict(covariate_slopes='class')    # common is the default
    plan = build_plan(rows, ctx[:1], out[1:], cov, strata=sex, scales=('raw',), pattern_scales=(), n_groups=6,
                      seed=1, **kwargs)
    y = np.zeros((n, plan.n_columns))
    y[plan.order] = np.vstack(plan.blocks)
    c, o = ctx[0], out[1]
    idx = np.flatnonzero((c.labels >= 0) & np.isfinite(o.values) & np.all(np.isfinite(cov), axis=1))
    lab, yy = c.labels[idx], o.values[idx]
    value = np.array([c.values[int(k)] for k in lab])
    g = (value - value.mean()) / value.std()
    expected = np.empty(len(idx))
    for s in (0, 1):
        if slopes == 'class':
            for k in np.unique(lab):
                m = (sex[idx] == s) & (lab == k)
                d = np.column_stack([np.ones(m.sum()), cov[idx][m]])
                expected[m] = yy[m] - d @ np.linalg.lstsq(d, yy[m], rcond=None)[0]
        else:
            m = sex[idx] == s
            d = np.column_stack([(lab[m][:, None] == np.unique(lab)[None, :]).astype(float), cov[idx][m]])
            expected[m] = yy[m] - d @ np.linalg.lstsq(d, yy[m], rcond=None)[0]
    np.testing.assert_allclose(y[idx, plan.kinds['gy'].start], g * expected, rtol=1e-9, atol=1e-10)
    with pytest.raises(ValueError, match='covariate_slopes'):
        build_plan(rows, ctx[:1], out[1:], cov, scales=('raw',), n_groups=6, covariate_slopes='none')


def test_signflip_p_matches_exact_enumeration():
    """Sign-flip p of dosage and delta2 against all 2^G sign patterns of the oracle kernels."""
    source, raw, ctx, out, cov = simulated(seed=11)
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx, out, cov, scales=('within_rank',), additive_scales=('within_rank',), n_groups=8,
                      seed=2)
    bg = background_axis(source, n_blocks=6)
    scale = estimate_scale(source, rows, bg.variants)
    acc = accumulate(source, plan, scale, bg, block_size=32, native=False)
    records = {(r['analysis'], r['statistic']): r for r in summarize(plan, acc)}
    s = dense_scores(np.asarray(source.values), scale, plan)
    chrom = np.array(bg.chromosomes)
    g = plan.n_groups
    signs = np.array(np.meshgrid(*[[-1., 1.]] * g)).reshape(g, -1).T

    def exact(k):
        k = (k + k.T) / 2
        np.fill_diagonal(k, 0.)
        q = np.einsum('ij,ij->i', signs @ k, signs)
        return np.mean(q >= k.sum() * (1 - 1e-12))

    checked = 0
    for p, (a, _) in enumerate(plan.keys['gy']):
        w = (~np.isin(chrom, plan.analyses[a]['exclude_chromosomes'])).astype(float)
        gy, lo, hi = plan.kinds['gy'].start + p, plan.kinds['lo'].start + p, plan.kinds['hi'].start + p
        kern = {c: oracle_kernel(s, c[0], c[1], w) / w.sum() for c in ((gy, gy), (lo, lo), (hi, hi), (lo, hi))}
        for name, k in (('dosage', kern[(gy, gy)]),
                        ('delta2', kern[(lo, lo)] + kern[(hi, hi)] - 2 * kern[(lo, hi)])):
            r = records[(a, name)]
            assert r['p_method'] == 'signflip' and r['flips'] >= 20_000
            e = exact(k)
            assert abs(r['p'] - e) < 5 * np.sqrt(e * (1 - e) / r['flips']) + 2e-4
            assert np.isclose(r['p_normal'], norm.sf(r['z']))
            checked += 1
        assert records[(a, 'relative_dosage')]['p'] == records[(a, 'dosage')]['p']
        assert records[(a, 'rho')]['p_method'] == 'normal'
    assert checked >= 4


def test_signflip_p_is_calibrated_when_the_null_is_skewed():
    """Two dominant score components: the normal tail over-rejects, the sign flip stays nominal."""
    from summit.epistasis.scale_free.stats import signflip_sf
    rng = np.random.default_rng(8)
    g, reps = 40, 500
    flip, normal = [], []
    for _ in range(reps):
        sc = rng.normal(size=(g, 2)) * [3., 1.]
        t = sc @ sc.T
        np.fill_diagonal(t, 0.)
        p, used, skew = signflip_sf(t, flips=(2000, 2000, 0.), seed=int(rng.integers(2**31)))
        flip.append(p)
        normal.append(norm.sf(t.sum() / np.sqrt(2 * np.sum(t ** 2))))
    flip, normal = np.array(flip), np.array(normal)
    assert skew > 1
    assert .025 <= np.mean(flip < .05) <= .08
    assert np.mean(flip < .01) <= .022
    assert np.mean(normal < .01) > 2 * max(np.mean(flip < .01), 1 / reps)


def test_jackknife_complements_use_leave_one_out_weights():
    """dosage_diff[A] and sel_*[A] jackknife: each delete-one estimate uses its own complement weights."""
    from summit.epistasis.scale_free.stats import dosage_statistics, pattern_statistics
    source, raw, ctx, out, cov = simulated(seed=11, n=3000, m=90)
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx, out, cov, scales=('raw',), n_groups=8, seed=2)
    annotation = np.where(np.arange(90) % 30 < 15, .9, .1)      # dense in some blocks, sparse in others
    bg = background_axis(source, annotations=annotation[:, None], names=('A',), n_blocks=12)
    scale = estimate_scale(source, rows, bg.variants)
    acc = accumulate(source, plan, scale, bg, block_size=32, native=False)
    recs = {(r['analysis'], r['statistic']): r for r in summarize(plan, acc)}
    pairs = plan.n_groups * (plan.n_groups - 1)

    def jackknife(function, betweens, keep):
        w = (acc.block_weight * keep).sum(axis=1)
        values = []
        for b in np.flatnonzero((acc.block_weight * keep).sum(axis=0) > 0):
            wb = w - acc.block_weight[:, b] * keep[b]
            values.append(function([(bt.sum(axis=1) - bt[:, b]) / (wb * pairs) for bt in betweens], wb))
        return {k: np.sqrt((len(values) - 1) / len(values) * np.sum((np.array([v[k] for v in values])
                                                                       - np.mean([v[k] for v in values])) ** 2))
                for k in values[0]}

    checked = 0
    for p, (a, _) in enumerate(plan.keys['gy']):
        se = jackknife(lambda th, wb: dosage_statistics(th[0], ('A',), wb), [acc.between['gy'][:, p]],
                       acc.keep['gy'][p])
        r = recs[(a, 'dosage_diff[A]')]
        assert np.isclose(r['se_jackknife'], se['dosage_diff[A]'], rtol=1e-8)
        checked += 1
    for p, (a, _) in enumerate(plan.keys['lo']):
        def pattern(th, wb):
            return pattern_statistics(np.concatenate([[th[0][i], th[1][i], th[2][i]] for i in range(len(wb))]),
                                      ('A',), wb)
        se = jackknife(pattern, [acc.between[k][:, p] for k in ('ll', 'hh', 'lh')], acc.keep['ll'][p])
        for key in ('sel_lam[A]', 'sel_amp[A]'):
            r = recs[(a, key)]
            if np.isfinite(r['se_jackknife']):
                assert np.isclose(r['se_jackknife'], se[key], rtol=1e-8)
                checked += 1
    assert checked >= 3


def test_effective_counts_remove_degree_of_freedom_bias_in_log_amp():
    """Class-specific slopes with many covariates: scores normalized by sum(1 - h) keep log_amp unbiased."""
    rng = np.random.default_rng(3)
    n, m, p, reps = 3000, 150, 100, 8
    raw = rng.binomial(2, rng.uniform(.1, .5, m), (n, m)).astype(np.int8)
    chrom = tuple(str(c) for c in np.repeat([1, 2, 3], m // 3))
    axis = VariantAxis(tuple(f'v{i}' for i in range(m)), chrom, tuple(range(1, m + 1)), ('A',) * m, ('G',) * m)
    source = ArrayGenotypeSource(raw, [(str(i), str(i)) for i in range(n)], axis, hard_calls=True)
    x = (raw - raw.mean(0)) / raw.std(0)
    rows = np.arange(n)
    bg = background_axis(source, n_blocks=12)
    scale = estimate_scale(source, rows, bg.variants)
    values = []
    for r in range(reps):
        focal = (rng.random(n) < .12).astype(int)
        liability = x @ rng.normal(0, np.sqrt(.7 / m), m) + rng.normal(0, np.sqrt(.3), n)
        y = np.exp(.5 * (.3 * focal + np.exp(.2 * focal) * liability))      # exact affine-modifier null
        plan = build_plan(rows, [Context('f', focal, {0: 0., 1: 1.}, low=(0,), high=(1,), exclude_chromosomes=('1',))],
                          [Outcome('y', y, positive=True)], rng.normal(size=(n, p)), scales=('within_rank',),
                          n_groups=10, seed=r, covariate_slopes='class')
        hi = plan.kinds['hi'].start
        n_hi = plan.analyses[0]['n_high']
        assert np.isclose(plan.counts[:, hi].sum(), n_hi - (1 + p), rtol=1e-8)    # sum of 1 - leverage
        acc = accumulate(source, plan, scale, bg, block_size=64, native=False)
        values += [rec['estimate'] for rec in summarize(plan, acc) if rec['statistic'] == 'log_amp']
    # with raw counts the shift would be about 2 log((1 - 101/360) / (1 - 101/2640)), i.e. -0.58
    assert abs(np.mean(values)) < .25


def test_variant_contexts_reject_fractional_dosages():
    from summit.epistasis.scale_free.workflow import variant_contexts
    n = 8
    dosage = np.array([[0, .4, .9, 1.1, 1.6, 1.95, 2, .2]], dtype=float).T
    axis = VariantAxis(('v0',), ('1',), (1,), ('A',), ('G',))
    source = ArrayGenotypeSource(dosage, [(str(i), str(i)) for i in range(n)], axis, hard_calls=False)
    with pytest.raises(ValueError, match='hard calls'):
        variant_contexts(source, np.arange(n), [{'name': 'v0', 'variant': 'v0', 'min_minor_homozygotes': 1}])


def test_background_blocks_are_contiguous_when_blocks_exceed_variants():
    source, raw, ctx, out, cov = simulated(seed=2, n=200, m=90)
    bg = background_axis(source, n_blocks=200)
    assert bg.n_blocks == len(np.unique(bg.blocks)) == len(bg.block_chromosomes)
    assert np.array_equal(np.unique(bg.blocks), np.arange(bg.n_blocks))


def test_empty_annotation_drops_only_that_annotation():
    source, raw, ctx, out, cov = simulated(seed=6)
    rows = np.arange(len(source.samples))
    plan = build_plan(rows, ctx[:1], out[1:], cov, scales=('within_rank',), n_groups=6, seed=1)
    chrom1 = (np.array(source.variants.chromosome) == '1').astype(float)    # the context's own chromosome
    bg = background_axis(source, annotations=chrom1[:, None], names=('own',), n_blocks=6)
    scale = estimate_scale(source, rows, bg.variants)
    acc = accumulate(source, plan, scale, bg, block_size=32, native=False)
    stats = {r['statistic'] for r in summarize(plan, acc)}
    assert {'dosage', 'rho', 'log_amp', 'delta2'} <= stats
    assert not any('[own]' in s for s in stats)


def test_pair_battery_runs_without_minor_homozygotes():
    from summit.epistasis.scale_free.rank_pairs import pair_battery
    rng = np.random.default_rng(4)
    n = 3000
    x = (rng.random(n) < .1).astype(int)                     # no minor homozygotes
    z = rng.binomial(2, .3, n)
    y = np.exp(.3 * x + .2 * z + .2 * x * z + rng.normal(size=n))
    out = pair_battery(y, x, z, rng.normal(size=(n, 2)), boot=20)
    assert np.isfinite(out['cpm_logit_z']) and np.isfinite(out['ls_probit_z'])


def _write_cli_inputs(tmp_path, seed=21, n=700, m=60):
    from bed_reader import to_bed
    source, raw, ctx, out, cov = simulated(seed=seed, n=n, m=m)
    ids = [str(i) for i in range(n)]
    values = np.where(raw == -127, np.nan, raw.astype(float))
    to_bed(tmp_path / 'input.bed', values, properties=dict(
        fid=ids, iid=ids, sid=[f'v{i}' for i in range(m)], chromosome=list(source.variants.chromosome),
        bp_position=np.arange(1, m + 1), allele_1=['A'] * m, allele_2=['G'] * m))
    (tmp_path / 'samples.tsv').write_text('FID IID\n' + ''.join(f'{i} {i}\n' for i in range(n)))
    y = out[0].values
    sex = [str(int(cov[i, 1])) if i % 50 else 'NA' for i in range(n)]       # some missing strata
    (tmp_path / 'pheno.tsv').write_text('FID IID y sex\n' + ''.join(f'{i} {i} {y[i]} {sex[i]}\n' for i in range(n)))
    (tmp_path / 'cov.tsv').write_text('FID IID c1\n' + ''.join(f'{i} {i} {cov[i, 0]}\n' for i in range(n)))
    (tmp_path / 'ctx.tsv').write_text('FID IID k\n' + ''.join(f'{i} {i} {ctx[1].labels[i]}\n' for i in range(n)))
    return dict(kind='summit.epistasis.scale_free', schema_version=1, genotypes={'geno': 'input.bed'},
                samples='samples.tsv', phenotypes={'file': 'pheno.tsv', 'columns': ['y'], 'positive': ['y']},
                covariates={'file': 'cov.tsv', 'columns': ['c1']}, strata={'file': 'pheno.tsv', 'column': 'sex'},
                contexts=[{'name': 'v0', 'variant': 'v0', 'min_minor_homozygotes': 10},
                          {'name': 'other', 'file': 'ctx.tsv', 'column': 'k', 'values': {'0': 0, '1': 1, '2': 2},
                           'low': [0, 1], 'high': [2], 'exclude_chromosomes': ['2']}],
                background={'jackknife_blocks': 6}, scales=['within_rank'], groups=5)


@pytest.mark.parametrize('change, message', [
    (lambda m: m['contexts'][1].update(exclude_chromosomes=['chr2']), 'exclude_chromosomes'),
    (lambda m: m['contexts'][1].update(low=[0, 1], high=[1, 2]), 'disjoint'),
    (lambda m: m['phenotypes'].update(positive=['y', 'z']), 'unknown columns'),
])
def test_manifest_validation(tmp_path, change, message):
    if not _native_available():
        pytest.skip('native extension is not loaded')
    from summit.epistasis.scale_free.workflow import run
    manifest = _write_cli_inputs(tmp_path)
    change(manifest)
    (tmp_path / 'm.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        run(tmp_path / 'm.json', tmp_path / 'out', threads=prediction_threads())


def test_missing_strata_and_memory_budget(tmp_path):
    if not _native_available():
        pytest.skip('native extension is not loaded')
    import pandas as pd
    from summit.epistasis.scale_free.workflow import run
    manifest = _write_cli_inputs(tmp_path)
    (tmp_path / 'm.json').write_text(json.dumps(manifest))
    with pytest.raises(MemoryError, match='split the analyses'):
        run(tmp_path / 'm.json', tmp_path / 'tiny', threads=prediction_threads(), memory_bytes=1000)
    run(tmp_path / 'm.json', tmp_path / 'out', threads=prediction_threads(), memory_bytes=2**30)
    res = pd.read_csv(tmp_path / 'out' / 'results.tsv', sep='\t')
    run_info = json.loads((tmp_path / 'out' / 'run.json').read_text())
    for a in run_info['analyses']:          # the 14 rows with a missing stratum are dropped, not a third stratum
        assert a['n'] <= 700 - 14
    assert set(res.statistic) >= {'dosage', 'rho'}
