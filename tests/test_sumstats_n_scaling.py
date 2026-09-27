"""Independent checks of variable-N OLS moments and their estimating equations."""
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from summit.inference.h2core import prepare_h2, fit_h2
from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
from summit.inference.rgcore import prepare_rg, fit_rg, fit_intercept
from summit.inference.trace import TraceView
from summit.sumstats.moments import (
    build_h2_summary_moment, build_rg_summary_moment,
    exact_score_z_from_arrays, h2_moment_from_arrays,
)
from summit.sumstats.sumstats import MatchedSumstats


def _matched(trace, n, raw_score, q=0):
    n = np.asarray(n, dtype=float)
    raw_score = np.asarray(raw_score, dtype=float)
    scale = n.max() - q - 1
    # Invert the OLS partial-correlation identity, independently of the helper.
    r = raw_score / np.sqrt(scale)
    se = np.full(n.size, 0.05)
    beta = se * r * np.sqrt((n - q - 2) / (1 - r*r))
    return MatchedSumstats(
        snps=trace.snps, n=n, nsamp=n.max(), n_scale=scale,
        z=beta/se, chi2=(beta/se)**2, beta=beta, se=se,
        a1=np.repeat("A", n.size), a2=np.repeat("C", n.size),
        cov_rank=q, cov_rank_source="test", name="test",
    )


def _fixture(k=2):
    m = 60
    idx = np.arange(m)
    A = np.ones((m, 1)) if k == 1 else np.eye(2)[idx % 2]
    # Each deletion retains a symmetric two-component HE system.
    L = 0.25 + 1.5*A + 0.05*((idx//2) % 5)[:, None]
    trace = TraceView(
        snps=np.asarray([f"rs{i}" for i in idx]), chr=idx//10+1,
        bp=idx*100+1, annot=A, annot_header=np.asarray([f"a{i}" for i in range(k)]),
        ldscores=L,
    )
    jk = JackknifeDesign.from_trace_view(trace, JackknifeSpec.parse(6))
    # Missingness depends strongly on the annotation and on the block.
    n1 = 1200 - 350*(idx % 2) - 23*(idx % 7)
    n2 = 1700 - 500*(idx % 2) - 31*(idx % 5)
    return trace, jk, n1.astype(float), n2.astype(float)


@pytest.mark.parametrize("q", [0, 5])
def test_constant_n_is_bit_identical_and_variable_n_null_is_one(q):
    n = np.array([13., 80., 300., 1000.]) + q
    beta = np.array([0.2, -0.3, 0.5, -0.9])
    se = np.array([0.4, 0.5, 0.6, 0.7])
    constant = np.full(n.size, n.max())
    old = np.sqrt(n.max()-q-1)*beta / np.sqrt(beta*beta + (constant-q-2)*se*se)
    np.testing.assert_array_equal(h2_moment_from_arrays(beta, se, constant, n.max(), q), old*old)
    np.testing.assert_allclose(h2_moment_from_arrays(se, se, n, n.max(), q), 1., atol=2e-14)
    # A response below zero is valid after the null subtraction.
    assert h2_moment_from_arrays(np.zeros(4), se, n, n.max(), q)[0] < 0


def test_exact_permutation_null_with_different_observed_subsets():
    x = np.array([0., 1., 2., 0., 2., 1.])
    ys = np.array(list(permutations([0., 0.5, 1.5, 3., 5., 9.])))
    for keep in [np.arange(6), np.array([0, 1, 3, 4])]:
        xc = x[keep] - x[keep].mean()
        yc = ys[:, keep] - ys[:, keep].mean(axis=1, keepdims=True)
        beta = yc @ xc / (xc @ xc)
        resid = yc - beta[:, None]*xc
        se = np.sqrt(np.sum(resid*resid, axis=1)/(keep.size-2)/(xc @ xc))
        n = np.full(beta.size, keep.size)
        old = exact_score_z_from_arrays(beta, se, n, 6, 0)**2
        corrected = h2_moment_from_arrays(beta, se, n, 6)
        np.testing.assert_allclose(old.mean(), 5/(keep.size-1), atol=2e-14)
        np.testing.assert_allclose(corrected.mean(), 1., atol=2e-14)


@pytest.mark.parametrize("weight", ["he", "ldsc"])
@pytest.mark.parametrize("signal", [0., 0.12])
def test_h2_recovers_null_and_signal_with_annotation_dependent_n(weight, signal):
    tv, jk, n, _ = _fixture()
    h = np.array([signal, signal/2])
    scale = n.max()-1
    expected = 1 + scale * (tv.ldscores / tv.annot.sum(axis=0)) @ h
    noise = scale/(n-1)
    ss = _matched(tv, n, np.sqrt(expected-1+noise))
    y, _ = build_h2_summary_moment(ss)
    np.testing.assert_allclose(y, expected, atol=3e-14)
    p = prepare_h2(tv, ss, jk)
    fit = fit_h2(p, weight_mode=weight, ldsc_m_annot=tv.annot.sum(axis=0),
                 ldsc_overlap_matrix=tv.annot.T@tv.annot, ldsc_source_nsnps=tv.nsnps)
    np.testing.assert_allclose(fit.sigma_reps[:, :2], np.tile(h, (jk.nrep+1, 1)), atol=2e-13)
    np.testing.assert_allclose(fit.h2[-1], [h.sum(), 0.], atol=2e-13)


@pytest.mark.parametrize("weight", ["he", "ldsc"])
@pytest.mark.parametrize("c", [0., 0.3, -0.2])
def test_rg_fixed_overlap_and_delete_refits_match_direct_equations(weight, c):
    tv, jk, n1, n2 = _fixture()
    m = tv.annot.sum(axis=0)
    scale = np.sqrt((n1.max()-1)*(n2.max()-1))
    b = scale/np.sqrt((n1-1)*(n2-1))
    gamma = np.array([0.04, -0.015])
    raw_y = c*b + scale * (tv.ldscores/m) @ gamma
    # Signed products test negative covariance as well as allele direction.
    ss1 = _matched(tv, n1, np.sqrt(np.abs(raw_y)))
    ss2 = _matched(tv, n2, np.sign(raw_y)*np.sqrt(np.abs(raw_y)))
    y, info = build_rg_summary_moment(ss1, ss2)
    np.testing.assert_allclose(y, raw_y, atol=1e-14)
    p = prepare_rg(tv, ss1, ss2, jk)
    hfits = []
    for ss in [ss1, ss2]:
        # Known h2 plug-ins isolate the covariance estimating equations.
        h = np.tile([0.2, 0.1, 0.3], (jk.nrep+1, 1))
        hfits.append(SimpleNamespace(weight_mode=weight, prepared=p,
                                    h2_reps=h, sigma_reps=h))
    intercept = fit_intercept(tv, ss1, ss2, jk, *hfits, fixed_c=c)
    f = fit_rg(p, *hfits, intercept, weight_mode=weight, ldsc_m_annot=m)
    np.testing.assert_allclose(f.gamma_reps, np.tile(gamma, (jk.nrep+1, 1)), atol=2e-13)
    np.testing.assert_allclose(f.gamma[:, 1], 0., atol=2e-13)
    for r in range(jk.nrep+1):
        keep = np.ones(tv.nsnps, bool) if r == jk.nrep else jk.D[r, jk.unit_id] == 0
        np.testing.assert_allclose(p.intercept_mass_rep[r], tv.annot[keep].T@b[keep])


@pytest.mark.parametrize("mode", ["score", "ldsc"])
@pytest.mark.parametrize("q", [0, 4])
def test_estimated_overlap_uses_snp_scaling_in_every_refit(mode, q):
    tv, jk, n1, n2 = _fixture(k=1)
    scale = np.sqrt((n1.max()-q-1)*(n2.max()-q-1))
    b = scale/np.sqrt((n1-q-1)*(n2-q-1))
    c = 0.17
    gamma = 0.06
    raw_y = c*b + scale*tv.ldscores[:, 0]*gamma/tv.nsnps
    ss1 = _matched(tv, n1, np.sqrt(raw_y), q=q)
    ss2 = _matched(tv, n2, np.sqrt(raw_y), q=q)
    p = prepare_rg(tv, ss1, ss2, jk)
    h = SimpleNamespace(h2_reps=np.full((jk.nrep+1, 1), 0.3))
    fit = fit_intercept(tv, ss1, ss2, jk, h, h, intercept_weight_mode=mode,
                        intercept_chisq_threshold=None, score_prepared=p)
    np.testing.assert_allclose(fit.c_reps, c, atol=2e-11)
    np.testing.assert_allclose(fit.info['regression_gamma_g_total_full'], gamma, atol=2e-11)


def test_kmoment_se_rejects_varying_sample_sets():
    tv, jk, n1, n2 = _fixture(k=1)
    ss1 = _matched(tv, n1, np.full(tv.nsnps, 2.))
    ss2 = _matched(tv, n2, np.full(tv.nsnps, 2.))
    p = prepare_rg(tv, ss1, ss2, jk)
    h1 = fit_h2(prepare_h2(tv, ss1, jk))
    h2 = fit_h2(prepare_h2(tv, ss2, jk))
    c = fit_intercept(tv, ss1, ss2, jk, h1, h2, fixed_c=0.)
    with pytest.raises(ValueError, match='constant per-SNP sample sizes'):
        fit_rg(p, h1, h2, c, rg_se_method='kmoments')


def test_ldsc_h2_noisy_refits_equal_regression_on_local_scores():
    tv, jk, n, _ = _fixture()
    scale = n.max()-1
    noise = scale/(n-1)
    mass = tv.annot.sum(axis=0)
    D = scale*tv.ldscores/mass
    q = D @ np.array([0.12, 0.06]) + 0.5*np.sin(np.arange(n.size))
    ss = _matched(tv, n, np.sqrt(noise+q))
    fit = fit_h2(prepare_h2(tv, ss, jk), weight_mode='ldsc', ldsc_m_annot=mass,
                 ldsc_overlap_matrix=tv.annot.T@tv.annot, ldsc_source_nsnps=tv.nsnps)
    expected = []
    for r in range(jk.nrep+1):
        keep = np.ones(tv.nsnps, bool) if r == jk.nrep else jk.D[r, jk.unit_id] == 0
        h = np.linalg.lstsq(D[keep], q[keep], rcond=None)[0]
        # Independent local-N formulation: z_local^2-1 against n*_j L_j/M.
        local_D = D[keep]/noise[keep, None]
        local_q = q[keep]/noise[keep]
        ld = np.maximum(tv.ldscores[keep].sum(axis=1), 1.)
        for _ in range(3):
            mu = 1 + np.clip(h.sum(), 0, 1)*(n[keep]-1)*ld/mass.sum()
            root_w = 1/np.sqrt(2*mu*mu*ld)
            h = np.linalg.lstsq(local_D*root_w[:, None], local_q*root_w, rcond=None)[0]
        expected.append(h)
    np.testing.assert_allclose(fit.sigma_reps[:, :2], expected, atol=3e-14)
    total = np.sum(expected, axis=1)
    se = np.sqrt((jk.nrep-1)/jk.nrep*np.sum((total[:-1]-total[:-1].mean())**2))
    np.testing.assert_allclose(fit.h2[-1, 1], se, atol=3e-14)
    assert se > 0


def test_ldsc_rg_noisy_refits_equal_regression_on_local_scores():
    tv, jk, n1, n2 = _fixture()
    N1, N2 = n1.max()-1, n2.max()-1
    noise1, noise2 = N1/(n1-1), N2/(n2-1)
    b = np.sqrt(noise1*noise2)
    mass = tv.annot.sum(axis=0)
    D = np.sqrt(N1*N2)*tv.ldscores/mass
    c = .2
    y = c*b + D @ np.array([.03, .01]) + .2*np.cos(np.arange(n1.size))
    ss1, ss2 = _matched(tv, n1, np.sqrt(y)), _matched(tv, n2, np.sqrt(y))
    p = prepare_rg(tv, ss1, ss2, jk)
    hs = [SimpleNamespace(weight_mode='ldsc', prepared=p,
                          h2_reps=np.tile([h/2, h/2, h], (jk.nrep+1, 1)),
                          sigma_reps=np.tile([h/2, h/2, h], (jk.nrep+1, 1)))
          for h in [.3, .4]]
    intercept = fit_intercept(tv, ss1, ss2, jk, *hs, fixed_c=c)
    intercept = replace(intercept, c_reps=np.linspace(.1, .2, jk.nrep+1))
    fit = fit_rg(p, *hs, intercept, weight_mode='ldsc', ldsc_m_annot=mass)
    expected = []
    for r in range(jk.nrep+1):
        keep = np.ones(tv.nsnps, bool) if r == jk.nrep else jk.D[r, jk.unit_id] == 0
        cr = intercept.c_reps[r]
        response = y-cr*b
        gamma = np.linalg.lstsq(D[keep], response[keep], rcond=None)[0]
        local_D = D[keep]/b[keep, None]
        local_q = y[keep]/b[keep]-cr
        ld = np.maximum(tv.ldscores[keep].sum(axis=1), 1.)
        for _ in range(3):
            v1 = 1 + (n1[keep]-1)*.3*ld/mass.sum()
            v2 = 1 + (n2[keep]-1)*.4*ld/mass.sum()
            cv = cr + np.sqrt((n1[keep]-1)*(n2[keep]-1))*np.clip(gamma.sum(), -1, 1)*ld/mass.sum()
            rw = 1/np.sqrt((v1*v2+cv*cv)*ld)
            gamma = np.linalg.lstsq(local_D*rw[:, None], local_q*rw, rcond=None)[0]
        expected.append(gamma)
    np.testing.assert_allclose(fit.gamma_reps, expected, atol=3e-14)
    total = np.sum(expected, axis=1)
    se = np.sqrt((jk.nrep-1)/jk.nrep*np.sum((total[:-1]-total[:-1].mean())**2))
    np.testing.assert_allclose(fit.gamma_total[1], se, atol=3e-14)
    assert se > 0


def test_overlap_weighted_sums_ignore_nonfinite_excluded_rows():
    from summit.inference.rgcore import (
        _compute_weighted_intercept_unit_summaries,
        _compute_weighted_intercept_summaries,
    )
    tv, jk, n, _ = _fixture(k=1)
    x = tv.ldscores.copy()
    y = .2*x[:, 0] + .1
    b = (n.max()-1)/(n-1)
    w = np.ones(n.size)
    w[::5] = 0.
    good = w != 0
    expected = _compute_weighted_intercept_summaries(x[good], y[good], w[good], b[good])
    x[~good] = np.nan
    y[~good] = np.nan
    b[~good] = np.nan
    full = _compute_weighted_intercept_summaries(x, y, w, b)
    units = _compute_weighted_intercept_unit_summaries(jk, x, y, w, b)
    for a, u, e in zip(full, units, expected):
        np.testing.assert_allclose(a, e)
        np.testing.assert_allclose(u.sum(axis=0), e)


@pytest.mark.parametrize('method', ['jackknife', 'delta', 'robust'])
def test_self_trait_rg_and_se_with_varying_n(method):
    tv, jk, n, _ = _fixture(k=1)
    scale = n.max()-1
    noise = scale/(n-1)
    signal = scale*tv.ldscores[:, 0]*.15/tv.nsnps
    raw_square = noise + signal + .1*np.sin(np.arange(n.size))
    ss = _matched(tv, n, np.sqrt(raw_square))
    p = prepare_rg(tv, ss, ss, jk)
    h = fit_h2(prepare_h2(tv, ss, jk))
    intercept = fit_intercept(tv, ss, ss, jk, h, h, fixed_c=1.)
    f = fit_rg(p, h, h, intercept, rg_se_method=method)
    np.testing.assert_allclose(f.gamma_reps[:, 0], h.h2_reps[:, -1], atol=3e-14)
    np.testing.assert_allclose(f.rg_reps[:, 0], 1., atol=3e-14)
    assert np.isfinite(f.gamma_total).all()
    assert np.isfinite(f.rg_total).all()
    if method == 'jackknife':
        np.testing.assert_allclose(f.gamma_total[1], h.h2[-1, 1], atol=3e-14)
        np.testing.assert_allclose(f.rg_total[1], 0., atol=3e-14)


def test_external_overlap_se_uses_correct_sensitivity():
    from summit.inference.rgcore import _external_c_sensitivity_se
    tv, jk, n1, n2 = _fixture()
    s1, s2 = _matched(tv, n1, np.full(tv.nsnps, 3.)), _matched(tv, n2, np.full(tv.nsnps, 3.))
    p = prepare_rg(tv, s1, s2, jk)
    h1, h2 = fit_h2(prepare_h2(tv, s1, jk)), fit_h2(prepare_h2(tv, s2, jk))
    c = fit_intercept(tv, s1, s2, jk, h1, h2, fixed_c=.2)
    c = replace(c, info={**c.info, 'source': 'pheno', 'external_c_se': .02})
    sensitivity = _external_c_sensitivity_se(p, h1, h2, c)[0]
    lo = fit_rg(p, h1, h2, c).gamma_reps[-1]
    hi = fit_rg(p, h1, h2, replace(c, c_reps=c.c_reps + .001)).gamma_reps[-1]
    np.testing.assert_allclose(sensitivity, np.abs((hi-lo)/.001)*.02, rtol=1e-11, atol=1e-14)
