"""The overlapping covariance in HE scale from summary statistics: the fitted line of z1*z2 on LD at each SNP's own
LD term (ld_point 'self'), with LDSC-style weights.

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

# origin/main 97307e4 on the legacy fixture below (float.hex): c replicates with 1/l weights, the full-sample c with
# LDSC weights, and gamma / rg replicates with the overlap supplied.
LEGACY_C_SCORE = [
    "0x1.e0469fe3b601ep-2",
    "0x1.d4eab2a7635f3p-2",
    "0x1.0d816d4defa45p-1",
    "0x1.3090165b5d327p-1",
    "0x1.08d942367ebacp-1",
    "0x1.415a88824e926p-1",
    "0x1.4563b1c832f4bp-1",
    "0x1.47cfa3ddae1cep-1",
    "0x1.218602d12168bp-1",
    "0x1.2c55b63eba6afp-1",
    "0x1.38d32c900b4dap-1",
    "0x1.64852f53cffcap-1",
    "0x1.2833451dc9354p-1",
]
LEGACY_C_LDSC_FULL = "0x1.4521d8cee454ap-1"
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
    itself: overlapping covariance 1). Reading at zero LD is most of a slope low; at the own term it is right up to
    the SNP's own causal share."""
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
        self_ = _fit(tv, jk, ss, ss, ld_point="self", intercept_weight_mode=weights)
        zero = _fit(tv, jk, ss, ss, ld_point="zero", intercept_weight_mode=weights)
        slope = self_.info["regression_slope_full"]
        assert abs(self_.c_reps[-1] - 1.0) < 0.35 * slope
        assert zero.c_reps[-1] - 1.0 < -0.6 * slope


@pytest.mark.parametrize("weights", ["score", "ldsc"])
def test_self_reads_the_line_at_the_own_term(weights):
    tv, jk = _trace()
    l = tv.ldscores[:, 0]
    ss = _matched(tv, np.sqrt(1.0 + 0.08 * (l - 1.0)))
    np.testing.assert_allclose(_fit(tv, jk, ss, ss, ld_point="self", intercept_weight_mode=weights).c_reps, 1.0,
                               atol=1e-10)
    np.testing.assert_allclose(_fit(tv, jk, ss, ss, ld_point="zero", intercept_weight_mode=weights).c_reps, 0.92,
                               atol=1e-10)


def test_zero_with_score_weights_is_the_legacy_estimate_bitwise():
    tv, jk, s1, s2, h1, h2, p = _legacy_fixture()
    fit = fit_intercept(tv, s1, s2, jk, h1, h2, intercept_weight_mode="score", ld_point="zero", score_prepared=p)
    assert [float(v).hex() for v in fit.c_reps] == LEGACY_C_SCORE
    # LDSC weights: the same full-sample estimate; replicates now reuse the full-sample weights.
    ldsc = fit_intercept(tv, s1, s2, jk, h1, h2, intercept_weight_mode="ldsc", ld_point="zero", score_prepared=p)
    assert ldsc.c_reps[-1] == pytest.approx(float.fromhex(LEGACY_C_LDSC_FULL), rel=1e-10, abs=0.0)


def test_supplied_overlap_is_unchanged_bitwise():
    tv, jk, s1, s2, h1, h2, p = _legacy_fixture()
    for mode in ("self", "zero"):
        fx = fit_intercept(tv, s1, s2, jk, h1, h2, fixed_c=0.21, ld_point=mode)
        np.testing.assert_array_equal(fx.c_reps, 0.21)
        f = fit_rg(p, h1, h2, fx)
        assert [float(v).hex() for v in f.gamma_reps[:, 0]] == LEGACY_FIXED_GAMMA
        assert [float(v).hex() for v in f.rg_reps[:, 0]] == LEGACY_FIXED_RG


@pytest.mark.parametrize("weights", ["score", "ldsc"])
def test_partitioned_and_variance_weighted_own_terms(weights):
    """Two LD columns with SNP-variance weights: each SNP's own term is V in its own column; the collapsed total's own
    term is V, and the per-column own terms of a two-column design are the weighted annotation."""
    v = 0.75 + 0.25 * ((np.arange(M) * 11) % 17) / 16.0
    tv, jk = _trace(v=v, k=2)
    l = tv.ldscores.sum(axis=1)
    ss = _matched(tv, np.sqrt(1.0 + 0.05 * (l - v)))
    np.testing.assert_allclose(_fit(tv, jk, ss, ss, ld_point="self", intercept_weight_mode=weights).c_reps, 1.0,
                               atol=1e-10)
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
    fit = _fit(tv, jk, ss, ss, ld_point="self", intercept_weight_mode="score")
    assert fit.info["n_removed_nonfinite_own_ld"] == 5
    np.testing.assert_allclose(fit.c_reps, 1.0, atol=1e-10)


def test_ld_point_option():
    from summit.cli import build_parser
    args = build_parser().parse_args([])
    assert (args.intercept_ld_point, args.intercept_weight_mode) == ("self", "ldsc")
    assert build_parser().parse_args(["--overlap-covariance-ld-point", "zero"]).intercept_ld_point == "zero"
    tv, jk = _trace()
    ss = _matched(tv, np.full(M, 1.2))
    with pytest.raises(ValueError, match="LD point"):
        _fit(tv, jk, ss, ss, ld_point="auto")
