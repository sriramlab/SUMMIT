"""The overlapping covariance in HE scale from summary statistics: the fitted line of z1*z2 on LD at each SNP's own
LD term, with LDSC-style weights.

With causal variants denser than the GWAS SNPs, a SNP's own term carries little of the genetic signal, so the line
passes through the overlapping covariance at a SNP that tags no other GWAS SNP; at zero LD it is about one slope lower.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from summit.inference.h2core import prepare_h2, fit_h2
from summit.inference.jackknife import JackknifeDesign, JackknifeSpec
from summit.inference.rgcore import (
    InterceptRegressionSystem, _overlap_own_ld, fit_intercept, fit_rg, prepare_rg,
)
from summit.inference.trace import TraceView
from summit.sumstats.sumstats import MatchedSumstats

M = 240
N = 5000.0

# The overlap supplied on the fixture below (float.hex): gamma and rg replicates, as on origin/main 97307e4.
LEGACY_FIXED_GAMMA = [
    "0x1.7aa3e7f976967p-9",
    "0x1.992ba21dd3af3p-9",
    "0x1.a991665d33de8p-9",
    "0x1.7cccb56af1ce8p-9",
    "0x1.002e9ebd7d66ep-8",
    "0x1.87c8f28ae86a4p-9",
    "0x1.dc86a1b610037p-9",
    "0x1.b7d3d4b8ab177p-9",
    "0x1.b3664be6137a5p-9",
    "0x1.a077c72e7099ep-9",
    "0x1.de4d833d7c041p-9",
    "0x1.b8b480ba80066p-9",
    "0x1.b00702602cdd6p-9",
]
LEGACY_FIXED_RG = [
    "0x1.1f934369e9024p+0",
    "0x1.03c7dd981af95p+0",
    "0x1.2ecaefd6ff87bp+0",
    "0x1.096cd3e0cf684p+0",
    "0x1.4562417e1f1dfp+0",
    "0x1.18587b8d055b3p+0",
    "0x1.1fd1c8798cf13p+0",
    "0x1.067a316c55074p+0",
    "0x1.2eb8cb277b26fp+0",
    "0x1.fac2827c4b1a7p-1",
    "0x1.0f59b7aa51292p+0",
    "0x1.2906b334fa634p+0",
    "0x1.19f19e8285d8ep+0",
]


def _trace(v=None, k=1, ld=None, nblocks=12):
    m = M if ld is None else len(ld)
    idx = np.arange(m)
    if ld is None:
        ld = 1.0 + 9.0 * ((idx * 37) % m) / m + 0.3 * (idx % 3)     # 1..10.6, unordered on the axis
    A = np.ones((m, 1)) if k == 1 else np.eye(k)[idx % k]
    L = np.asarray(ld, dtype=float)[:, None] * A                    # each SNP's LD in its own column
    w = 1.0 if v is None else np.asarray(v)[:, None]
    tv = TraceView(
        snps=np.asarray([f"rs{i}" for i in idx]), chr=idx // 40 + 1, bp=idx * 1000 + 1,
        annot=A * w, annot_header=np.asarray([f"a{j}" for j in range(k)]), ldscores=L,
        annot_mass=None if v is None else A,
    )
    return tv, JackknifeDesign.from_trace_view(tv, JackknifeSpec.parse(nblocks))


def _matched(tv, score):
    """Sumstats whose exact score statistic is `score` at constant N (so the h2 moment is score**2)."""
    m = tv.nsnps
    scale = N - 1.0
    r = np.asarray(score, dtype=float) / np.sqrt(scale)
    se = np.full(m, 0.05)
    beta = se * r * np.sqrt((N - 2.0) / (1.0 - r * r))
    return MatchedSumstats(
        snps=tv.snps, n=np.full(m, N), nsamp=N, n_scale=scale, z=beta / se, chi2=(beta / se) ** 2, beta=beta, se=se,
        a1=np.repeat("A", m), a2=np.repeat("C", m), cov_rank=0, cov_rank_source="test", name="t",
    )


def _h(jk, value=0.3):
    return SimpleNamespace(h2_reps=np.full((jk.nrep + 1, 2), value))


def _fit(tv, jk, ss1, ss2, h=None, **kw):
    h = _h(jk) if h is None else h
    return fit_intercept(tv, ss1, ss2, jk, h, h, **kw)


def _legacy_fixture():
    tv, jk = _trace()
    rng = np.random.default_rng(3)
    l = tv.ldscores[:, 0]
    z1 = rng.standard_normal(M) * np.sqrt(1 + 0.1 * l)
    z2 = 0.4 * z1 + rng.standard_normal(M)
    s1, s2 = _matched(tv, z1), _matched(tv, z2)
    h1, h2 = fit_h2(prepare_h2(tv, s1, jk)), fit_h2(prepare_h2(tv, s2, jk))
    return tv, jk, s1, s2, h1, h2, prepare_rg(tv, s1, s2, jk)


def test_self_recovers_the_overlap_with_dense_causal_variants():
    """In-sample LD of a small genotype panel: causal variants in equicorrelated blocks, one in six typed (the GWAS
    SNPs, plus isolated typed SNPs), expected products 1 + b sum_k r_jk^2 over the causal variants (one trait with
    itself: overlapping covariance 1). At the own term the line is right up to the SNP's own causal share."""
    rng = np.random.default_rng(5)
    n, k = 1200, 6
    cols, typed = [], []
    for m, r in zip([12, 24, 36, 48, 60, 72, 96, 120] * 2, np.linspace(0.25, 0.9, 16)):
        f = rng.standard_normal((n, 1))
        cols.append(np.sqrt(r) * f + np.sqrt(1 - r) * rng.standard_normal((n, m)))
        typed.append(np.r_[np.ones(m // k, bool), np.zeros(m - m // k, bool)])
    cols.append(rng.standard_normal((n, 40)))
    typed.append(np.ones(40, bool))
    X = np.hstack(cols)
    X = (X - X.mean(0)) / X.std(0)
    R = X.T @ X / n
    r2 = R * R - (1 - R * R) / (n - 2)
    J = np.flatnonzero(np.concatenate(typed))
    perm = rng.permutation(J.size)                                     # spread blocks over the jackknife units
    l_causal, l_gwas = r2[J].sum(1)[perm], r2[np.ix_(J, J)].sum(1)[perm]
    tv, jk = _trace(ld=l_gwas, nblocks=8)
    ss = _matched(tv, np.sqrt(1.0 + 0.003 * l_causal))
    for weights in ("score", "ldsc"):
        fit = _fit(tv, jk, ss, ss, intercept_weight_mode=weights)
        assert abs(fit.c_reps[-1] - 1.0) < 0.35 * fit.info["regression_slope_full"]


@pytest.mark.parametrize("weights", ["score", "ldsc"])
def test_self_reads_the_line_at_the_own_term(weights):
    tv, jk = _trace()
    l = tv.ldscores[:, 0]
    ss = _matched(tv, np.sqrt(1.0 + 0.08 * (l - 1.0)))
    np.testing.assert_allclose(_fit(tv, jk, ss, ss, intercept_weight_mode=weights).c_reps, 1.0, atol=1e-10)


def test_supplied_overlap_is_unchanged():
    """The supplied-overlap path is untouched: the replicates equal origin/main 97307e4's (bitwise on the platform that
    recorded them; to 1e-12 here, since BLAS builds differ in the last bits)."""
    tv, jk, s1, s2, h1, h2, p = _legacy_fixture()
    fx = fit_intercept(tv, s1, s2, jk, h1, h2, fixed_c=0.21)
    np.testing.assert_array_equal(fx.c_reps, 0.21)
    f = fit_rg(p, h1, h2, fx)
    np.testing.assert_allclose(f.gamma_reps[:, 0], [float.fromhex(v) for v in LEGACY_FIXED_GAMMA], rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.rg_reps[:, 0], [float.fromhex(v) for v in LEGACY_FIXED_RG], rtol=1e-12, atol=0)


@pytest.mark.parametrize("weights", ["score", "ldsc"])
def test_partitioned_and_variance_weighted_own_terms(weights):
    """Two LD columns with SNP-variance weights: each SNP's own term is V in its own column; the collapsed total's own
    term is V, and the per-column own terms of a two-column design are the weighted annotation."""
    v = 0.75 + 0.25 * ((np.arange(M) * 11) % 17) / 16.0
    tv, jk = _trace(v=v, k=2)
    l = tv.ldscores.sum(axis=1)
    ss = _matched(tv, np.sqrt(1.0 + 0.05 * (l - v)))
    np.testing.assert_allclose(_fit(tv, jk, ss, ss, intercept_weight_mode=weights).c_reps, 1.0, atol=1e-10)
    two = InterceptRegressionSystem(x=tv.ldscores, a=np.ones((M, 2)), total_ld=l, full_mass=np.full(2, M / 2),
                                    source="main", mode="main")
    np.testing.assert_array_equal(_overlap_own_ld(tv, two), tv.annot)
    reg = InterceptRegressionSystem(x=l[:, None], a=np.ones((M, 1)), total_ld=l, full_mass=np.array([M]),
                                    source="reg", mode="reg")
    np.testing.assert_array_equal(_overlap_own_ld(tv, reg), 1.0)


def test_ldsc_weights_fall_back_to_score_weights():
    tv, jk = _trace()
    rng = np.random.default_rng(4)
    l = tv.ldscores[:, 0]
    ss1 = _matched(tv, rng.standard_normal(M) * np.sqrt(1 + 0.1 * l))
    ss2 = _matched(tv, rng.standard_normal(M) * np.sqrt(1 + 0.1 * l))
    score = _fit(tv, jk, ss1, ss2, intercept_weight_mode="score")
    bad = _fit(tv, jk, ss1, ss2, h=_h(jk, np.nan), intercept_weight_mode="ldsc")   # no usable h2 plug-ins
    assert bad.info["weight_mode"] == "score" and bad.info["weight_mode_requested"] == "ldsc"
    assert "ldsc_weight_fallback" in bad.info
    np.testing.assert_array_equal(bad.c_reps, score.c_reps)
    ok = _fit(tv, jk, ss1, ss2, intercept_weight_mode="ldsc")
    assert ok.info["weight_mode"] == "ldsc" and ok.info["irwls_iters"] == 2


def test_nonfinite_own_terms_leave_the_fit():
    v = np.full(M, 0.9)
    v[:5] = np.nan
    tv, jk = _trace(v=v)
    l = tv.ldscores[:, 0]
    ss = _matched(tv, np.sqrt(1.0 + 0.05 * (l - 0.9)))
    fit = _fit(tv, jk, ss, ss, intercept_weight_mode="score")
    assert fit.info["n_removed_nonfinite_own_ld"] == 5
    np.testing.assert_allclose(fit.c_reps, 1.0, atol=1e-10)


def test_cli_default_weights():
    from summit.cli import build_parser
    assert build_parser().parse_args([]).intercept_weight_mode == "ldsc"


def test_replicate_that_cannot_identify_the_overlap_keeps_the_full_sample_value():
    """A delete-block replicate that leaves a single SNP in the overlap regression cannot separate the overlap from the
    slope (its denominator is zero up to rounding). It is held at the full-sample value on every platform, rather than
    taking a quotient of rounding errors."""
    tv, jk = _trace(nblocks=4)
    l = tv.ldscores[:, 0]
    rng = np.random.default_rng(8)
    ss = _matched(tv, np.sqrt(1.0 + 0.05 * (l - 1.0)) * (1.0 + 0.01 * rng.standard_normal(M)))
    unit = np.asarray(jk.unit_id)
    keep_one = np.flatnonzero(unit != unit[0])[:1]                    # outside block 0: a single SNP
    big = np.zeros(M, bool)
    big[unit == unit[0]] = True
    big[keep_one] = True
    z = np.where(big, ss.z, 50.0)                                      # the chi^2 filter drops all other SNPs
    ss = MatchedSumstats(snps=ss.snps, n=ss.n, nsamp=ss.nsamp, n_scale=ss.n_scale, z=z, chi2=z * z, beta=ss.beta,
                         se=ss.se, a1=ss.a1, a2=ss.a2, cov_rank=0, cov_rank_source="test", name="t")
    for weights in ("score", "ldsc"):
        fit = _fit(tv, jk, ss, ss, intercept_weight_mode=weights, intercept_chisq_threshold=30.0)
        assert np.isfinite(fit.c_reps).all()
        assert fit.info["n_reps_overlap_held_at_full_sample"] >= 1
