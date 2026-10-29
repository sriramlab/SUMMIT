import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.relatives.moments import (
    append_rel_m_to_sumstats,
    compute_relative_pair_moments,
    compute_relative_pair_totals,
    extract_rel_m12,
    pairs_within_sample,
    read_relative_pairs,
    relative_pair_ld_rel,
)
from summit.sumstats.moments import build_h2_summary_moment, exact_score_z_from_arrays, h2_moment_from_arrays
from summit.sumstats.sumstats import Sumstats


def _simulate(tmp_path, *, N=240, M=300, npairs=50, ncov=2, seed=0, missing=0.002):
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.1, 0.5, M)
    hap = (rng.random((2 * N, M)) < freq).astype(np.int8)
    # relatives: individual 2k+1 shares one haplotype with 2k (half-sib-like) for k < npairs
    for k in range(npairs):
        hap[2 * (2 * k + 1)] = hap[2 * (2 * k)]
    # a three-person chain 2k, 2k+1 and 2k+3 so that family closure adds a pair
    hap[2 * 3] = hap[2 * 0 + 1]
    G = (hap[0::2] + hap[1::2]).astype(np.float32)
    G[rng.random(G.shape) < missing] = np.nan
    iid = np.array([f"s{i}" for i in range(N)])
    sid = np.array([f"rs{j}" for j in range(M)])
    prefix = tmp_path / "geno"
    bed_reader.to_bed(str(prefix) + ".bed", G, properties={
        "fid": iid, "iid": iid, "sid": sid, "chromosome": np.repeat(["1", "2"], [M // 2, M - M // 2]),
        "bp_position": np.arange(1, M + 1) * 1000, "allele_1": np.full(M, "A"), "allele_2": np.full(M, "G")})
    fam_effect = np.zeros(N)
    for k in range(npairs):
        fam_effect[[2 * k, 2 * k + 1]] = rng.standard_normal()
    covs = rng.standard_normal((N, ncov))
    y = rng.standard_normal(N) + fam_effect + covs @ np.arange(1, ncov + 1) * 0.3
    pd.DataFrame({"FID": iid, "IID": iid, "y": y}).to_csv(tmp_path / "pheno.txt", sep=" ", index=False)
    cov_path = None
    if ncov:
        cdf = pd.DataFrame(covs, columns=[f"c{k}" for k in range(ncov)])
        cdf.insert(0, "IID", iid); cdf.insert(0, "FID", iid)
        cdf.to_csv(tmp_path / "covar.txt", sep=" ", index=False)
        cov_path = str(tmp_path / "covar.txt")
    pairs = pd.DataFrame({"IID1": [f"s{2 * k}" for k in range(npairs)] + ["s1"],
                          "IID2": [f"s{2 * k + 1}" for k in range(npairs)] + ["s3"]})
    pairs.to_csv(tmp_path / "pairs.txt", sep=" ", index=False)
    return str(prefix), G, y, covs, iid, cov_path


def _projected(G, y, covs):
    N = G.shape[0]
    V = np.column_stack([np.ones(N)] + ([covs] if covs is not None and covs.size else []))
    Q, _ = np.linalg.qr(V)
    p = V.shape[1] - 1
    n_eff = N - p - 1
    G = np.asarray(G, dtype=np.float64)
    X = np.where(np.isnan(G), np.nanmean(G, 0)[None, :], G)
    X -= Q @ (Q.T @ X)
    X *= np.sqrt(n_eff / (X ** 2).sum(0))[None, :]
    yy = y - Q @ (Q.T @ y)
    yy *= np.sqrt(n_eff / (yy ** 2).sum())
    return X, yy, p, n_eff


def _family_pairs(path, iid):
    rp = read_relative_pairs(path, pd.DataFrame({"FID": iid, "IID": iid}))
    return rp.i, rp.j


@pytest.mark.parametrize("ncov", [0, 2])
def test_rel_m_matches_brute_force(tmp_path, ncov):
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov)
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path, step_size=97)
    tab = pd.read_csv(out + ".tsv.gz", sep="\t")
    X, yy, p, n_eff = _projected(G, y, covs if ncov else None)
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    N = G.shape[0]
    brute = 2.0 * ((X[pi] * X[pj]) * (yy[pi] * yy[pj])[:, None]).sum(0) / n_eff
    np.testing.assert_allclose(tab["REL_M_y"].to_numpy(), brute, rtol=1e-8, atol=1e-12)


def test_family_closure_adds_chain_pair(tmp_path):
    _, _, _, _, iid, _ = _simulate(tmp_path, ncov=0)
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    pairs = set(zip(pi.tolist(), pj.tolist()))
    assert (0, 1) in pairs and (1, 3) in pairs and (0, 3) in pairs and (2, 3) in pairs
    rp = read_relative_pairs(str(tmp_path / "pairs.txt"), pd.DataFrame({"FID": iid, "IID": iid}), closure=False)
    assert (0, 3) not in set(zip(rp.i.tolist(), rp.j.tolist()))


@pytest.mark.parametrize("ncov", [0, 2])
def test_pair_excluded_quadratic_form_identity(tmp_path, ncov):
    """(n_scale/M) sum_j (m_j - REL_M_j) equals y'K_u y, K_u = GRM with related pairs removed, exactly when the
    h2 moment uses the GWAS covariate rank."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov, missing=0.0)
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path)
    rel = pd.read_csv(out + ".tsv.gz", sep="\t")["REL_M_y"].to_numpy()
    N, M = G.shape
    X, yy, p, n_eff = _projected(G, y, covs if ncov else None)
    # PLINK-style OLS beta/SE with the same covariates
    r = (X.T @ yy) / n_eff
    nu_true = N - p - 2
    resid_var = (1 - r * r) / nu_true
    beta = r            # on the normalized scale; SUMMIT's moment depends on beta/SE only through r
    se = np.sqrt(resid_var)
    m = h2_moment_from_arrays(beta=beta, se=se, n_obs=np.full(M, float(N)), nsamp=float(N), cov_rank=p)
    lhs = n_eff / M * (m - rel).sum()
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    K = X @ X.T / M
    K[pi, pj] = 0.0
    K[pj, pi] = 0.0
    np.testing.assert_allclose(lhs, yy @ K @ yy, rtol=1e-9)


def test_sumstats_rel_m_column_is_subtracted(tmp_path):
    M = 50
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"SNP": [f"rs{j}" for j in range(M)], "A1": "A", "A2": "G", "N": 1000,
                       "BETA": rng.normal(0, 0.03, M), "SE": 0.03, "REL_M": rng.normal(0.1, 0.01, M)})
    df.loc[5, "REL_M"] = np.nan
    path = tmp_path / "ss.txt"
    df.to_csv(path, sep="\t", index=False)
    ss = Sumstats.from_file(str(path), name="t")
    assert ss.rel_m is not None and ss.nsnps == M - 1

    class Matched:
        beta, se, n, nsamp, rel_m = ss.beta, ss.se, ss.n, ss.nsamp, ss.rel_m

    y, info = build_h2_summary_moment(Matched)
    base = h2_moment_from_arrays(ss.beta, ss.se, ss.n, ss.nsamp, 0)
    np.testing.assert_allclose(y, base - ss.rel_m)
    assert info["related_pairs"]["c_rel"] == pytest.approx(float(np.mean(ss.rel_m)))
    df2 = df.drop(columns=["REL_M"])
    df2.to_csv(tmp_path / "ss2.txt", sep="\t", index=False)
    assert Sumstats.from_file(str(tmp_path / "ss2.txt"), name="t2").rel_m is None


def _brute_ld_rel(X, pi, pj, n_eff, A):
    """(1/N_eff^2) sum over ordered related pairs of x_ij x_i'j W_k,ii', W_k = sum_j' a_j'k x_ij' x_i'j'."""
    G = X[pi] * X[pj]
    return 2.0 * (G.T @ (G @ A)) / n_eff ** 2


def _snp_meta(M):
    return {"CHR": np.repeat([1, 2], [M // 2, M - M // 2]), "BP": np.arange(1, M + 1) * 1000,
            "SNP": [f"rs{j}" for j in range(M)]}


def _ld_rel(G, covs, pairs_path, iid, A, step_size=97):
    """LD_REL from relative_pair_ld_rel on the whole sample; the basis includes the intercept, which centered
    genotypes are orthogonal to, so it spans the same projection as SUMMIT's LD-score step."""
    N, M = G.shape
    p = 0 if covs is None or not covs.size else covs.shape[1]
    C = None if p == 0 else np.linalg.qr(np.column_stack([np.ones(N), covs]))[0]
    rp = read_relative_pairs(pairs_path, pd.DataFrame({"FID": iid, "IID": iid}))
    a, b = pairs_within_sample(rp, np.arange(N), N)
    n_eff = N - p - 1
    return relative_pair_ld_rel(lambda s, e: G[:, s:e], M, a, b, A, C, n_eff, step_size=step_size), n_eff


@pytest.mark.parametrize("ncov", [0, 2])
def test_ld_rel_partitioned_matches_brute_force(tmp_path, ncov):
    _, G, y, covs, iid, _ = _simulate(tmp_path, ncov=ncov)
    M = G.shape[1]
    b = (np.random.default_rng(5).random(M) < 0.4).astype(float)
    A = np.column_stack([b, 1 - b]) * np.random.default_rng(6).uniform(0.9, 1.0, M)[:, None]   # V-like weights
    got, n_eff = _ld_rel(G, covs if ncov else None, str(tmp_path / "pairs.txt"), iid, A)
    X, _, p, n_eff2 = _projected(G, y, covs if ncov else None)
    assert n_eff == n_eff2
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    np.testing.assert_allclose(got, _brute_ld_rel(X, pi, pj, n_eff, A), rtol=1e-8, atol=1e-10)


def test_missing_phenotype_rel_m_uses_complete_cases(tmp_path):
    """REL_M uses the phenotype's complete cases (with complete covariates)."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2)
    ph = pd.read_csv(tmp_path / "pheno.txt", sep=" ", dtype={"FID": str, "IID": str})
    ph.loc[::7, "y"] = np.nan
    ph.to_csv(tmp_path / "pheno.txt", sep=" ", index=False, na_rep="NA")
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path, step_size=97)
    tab = pd.read_csv(out + ".tsv.gz", sep="\t")
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    keep = ph["y"].notna().to_numpy()
    pos = -np.ones(keep.size, dtype=int)
    pos[keep] = np.arange(keep.sum())
    use = keep[pi] & keep[pj]
    a, b = pos[pi[use]], pos[pj[use]]
    Xs, ys, p, ne = _projected(G[keep], y[keep], covs[keep])
    n = int(keep.sum())
    brute = 2.0 * ((Xs[a] * Xs[b]) * (ys[a] * ys[b])[:, None]).sum(0) / ne
    np.testing.assert_allclose(tab["REL_M_y"].to_numpy(), brute, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("ncov", [0, 2])
def test_ld_rel_gives_pair_excluded_trace(tmp_path, ncov):
    """With in-sample LD scores l_j = sum_j' r_jj'^2 - M/N_eff (SUMMIT's convention), l - LD_REL reconstructs the
    pair-excluded trace exactly: N_eff + N_eff^2/M^2 sum_j (l_j - LD_REL_j) = tr(K_u^2)."""
    _, G, y, covs, iid, _ = _simulate(tmp_path, ncov=ncov, missing=0.0)
    X, _, p, n_eff = _projected(G, y, covs if ncov else None)
    N, M = X.shape
    ld_exact = ((X.T @ X / n_eff) ** 2).sum(1) - M / n_eff
    rel, _ = _ld_rel(G, covs if ncov else None, str(tmp_path / "pairs.txt"), iid, np.ones((M, 1)))
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    K = X @ X.T / M
    assert n_eff + n_eff ** 2 / M ** 2 * ld_exact.sum() == pytest.approx((K * K).sum(), rel=1e-9)
    K[pi, pj] = 0.0
    K[pj, pi] = 0.0
    assert n_eff + n_eff ** 2 / M ** 2 * (ld_exact - rel[:, 0]).sum() == pytest.approx((K * K).sum(), rel=1e-9)


def _two_phenotypes(tmp_path, y, iid, seed=11):
    rng = np.random.default_rng(seed)
    y2 = 0.5 * y + rng.standard_normal(y.size)
    f2 = rng.standard_normal(50)                          # an extra effect shared within the first 50 pairs
    y2[0:100:2] += f2
    y2[1:100:2] += f2
    pd.DataFrame({"FID": iid, "IID": iid, "y": y, "y2": y2}).to_csv(tmp_path / "pheno2.txt", sep=" ", index=False)
    return y2


def _ols_sumstats(path, X, yy, p, rel_m=None):
    """Write a SUMMIT sumstats file whose beta/SE reproduce the partial correlations of the projected data."""
    N, M = X.shape
    n_eff = N - p - 1
    r = (X.T @ yy) / n_eff
    df = pd.DataFrame({"SNP": [f"rs{j}" for j in range(M)], "A1": "A", "A2": "G", "N": N,
                       "BETA": r, "SE": np.sqrt((1 - r * r) / (N - p - 2))})
    if rel_m is not None:
        df["REL_M"] = rel_m
    df.to_csv(path, sep="\t", index=False, float_format="%.15g")
    return r


@pytest.mark.parametrize("ncov", [0, 2])
def test_rel_m12_matches_brute_force_and_cross_identity(tmp_path, ncov):
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov, missing=0.0)
    y2 = _two_phenotypes(tmp_path, y, iid)
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno2.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path, step_size=97, cross=True)
    tab = pd.read_csv(out + ".tsv.gz", sep="\t")
    X, y1t, p, n_eff = _projected(G, y, covs if ncov else None)
    _, y2t, _, _ = _projected(G, y2, covs if ncov else None)
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    N, M = X.shape
    kappa = 1.0 / n_eff
    brute = kappa * ((X[pi] * X[pj]) * (y1t[pi] * y2t[pj] + y1t[pj] * y2t[pi])[:, None]).sum(0)
    np.testing.assert_allclose(tab["REL_M12_y__y2"].to_numpy(), brute, rtol=1e-8, atol=1e-12)
    # the self pair would equal REL_M: check REL_M from the same run against its own formula
    np.testing.assert_allclose(tab["REL_M_y"].to_numpy(),
                               kappa * 2.0 * ((X[pi] * X[pj]) * (y1t[pi] * y1t[pj])[:, None]).sum(0), rtol=1e-8, atol=1e-12)
    # exact with the GWAS covariate rank: (1/M) sum_j (z1* z2* - REL_M12_j) = y1'K_u y2 / N_eff
    r1, r2 = (X.T @ y1t) / n_eff, (X.T @ y2t) / n_eff
    z1 = exact_score_z_from_arrays(r1, np.sqrt((1 - r1 ** 2) / (N - p - 2)), np.full(M, N), N, p)
    z2 = exact_score_z_from_arrays(r2, np.sqrt((1 - r2 ** 2) / (N - p - 2)), np.full(M, N), N, p)
    K = X @ X.T / M
    K[pi, pj] = 0.0
    K[pj, pi] = 0.0
    assert np.mean(z1 * z2 - tab["REL_M12_y__y2"].to_numpy()) == pytest.approx(y1t @ K @ y2t / n_eff, rel=1e-9)
    c12 = extract_rel_m12(out + ".tsv.gz", "y2", "y", str(tmp_path / "m12.tsv"))
    assert c12 == pytest.approx(float(brute.mean()))


def test_rg_self_pair_with_relatives_equals_h2(tmp_path):
    """SUM-CORE with REL_M in both files and REL_M12 for the pair: a trait paired with itself (c_ov = 1) gives
    gamma = pair-excluded h2, so the cross part is excluded exactly like the h2 parts."""
    from summit.inference.sumcore import Sumcore
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2, missing=0.0)
    pd.DataFrame({"FID": iid, "IID": iid, "y": y, "yb": y}).to_csv(tmp_path / "pheno_self.txt", sep=" ", index=False)
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno_self.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path, cross=True)
    X, yy, p, n_eff = _projected(G, y, covs)
    N, M = X.shape
    _ols_sumstats(tmp_path / "ss.tsv", X, yy, p)
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), out + ".tsv.gz", "y", str(tmp_path / "ssa.tsv"))
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), out + ".tsv.gz", "yb", str(tmp_path / "ssb.tsv"))
    extract_rel_m12(out + ".tsv.gz", "y", "yb", str(tmp_path / "m12.tsv"))
    ld_exact = ((X.T @ X / n_eff) ** 2).sum(1) - M / n_eff
    rel, _ = _ld_rel(G, covs, str(tmp_path / "pairs.txt"), iid, np.ones((M, 1)))
    pd.DataFrame({**_snp_meta(M), "L2_0": ld_exact, "LD_REL_L2_0": rel[:, 0]})[
        ["CHR", "SNP", "BP", "L2_0", "LD_REL_L2_0"]].to_csv(
        tmp_path / "in.gw.ldscore.gz", sep="\t", index=False, float_format="%.6f")
    common = dict(ldscores=str(tmp_path / "in.gw.ldscore.gz"), njack="chr", chisq_threshold=None,
                  intercept_chisq_thr=None, intercept_rg=1.0, align_alleles=False)
    res = Sumcore(rg=f"{tmp_path / 'ssa.tsv'},{tmp_path / 'ssb.tsv'}", rel_m12=str(tmp_path / "m12.tsv"), **common)._run()
    gamma = float(res["rg_fit"].gamma_total[0])
    assert gamma == pytest.approx(float(res["h2_fit1"].h2[-1, 0]), rel=1e-8)
    assert float(res["h2_fit2"].h2[-1, 0]) == pytest.approx(float(res["h2_fit1"].h2[-1, 0]), rel=1e-12)
    # the pair-excluded h2 differs from the naive one (sumstats without REL_M keep LD_REL unused), so the check is
    # not vacuous
    naive = Sumcore(rg=f"{tmp_path / 'ss.tsv'},{tmp_path / 'ss.tsv'}", **common)._run()
    assert abs(float(naive["rg_fit"].gamma_total[0]) - gamma) > 1e-4
    with pytest.raises(ValueError, match="--rel-m12"):
        Sumcore(rg=f"{tmp_path / 'ssa.tsv'},{tmp_path / 'ssb.tsv'}", **common)._run()


# ---------------------------------------------------------------------------------------------- one value per phenotype
def _annot(tmp_path, M, seed=5):
    b = (np.random.default_rng(seed).random(M) < 0.4).astype(int)
    df = pd.DataFrame({**_snp_meta(M), "CM": 0, "lowmaf": b, "highmaf": 1 - b})
    df.to_csv(tmp_path / "annot.txt", sep="\t", index=False)
    return str(tmp_path / "annot.txt"), np.column_stack([b, 1 - b]).astype(float)


def _v_weights(G, covs):
    """1 - adjusted R2 of each SNP on the covariates (the LD-score step's V)."""
    N = G.shape[0]
    X = np.where(np.isnan(G), np.nanmean(G, 0)[None, :], np.asarray(G, dtype=np.float64))
    X = X - X.mean(0)
    p = 0 if covs is None else covs.shape[1]
    Q = np.linalg.qr(np.column_stack([np.ones(N)] + ([covs] if p else [])))[0]
    raw = (X ** 2).sum(0)
    ss = ((X - Q @ (Q.T @ X)) ** 2).sum(0)
    return np.minimum(1.0, ss / raw) * (N - 1) / (N - p - 1)


@pytest.mark.parametrize("ncov", [0, 2])
def test_totals_are_v_weighted_means_of_per_snp_rel_m(tmp_path, ncov):
    """With every phenotype observed in the same sample, the one-pass totals equal sum_j V_j a_jk REL_M_j / sum V_j a_jk
    of the exact per-SNP values, overall and per class, and the same for REL_M12."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov)
    _two_phenotypes(tmp_path, y, iid)
    M = G.shape[1]
    annot, A = _annot(tmp_path, M)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path, cross=True, annot=annot)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "tot"), step_size=61, **kw)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "ex"), step_size=97, **kw)
    t1 = pd.read_csv(tmp_path / "tot.rel_m.tsv", sep="\t").set_index("PHENO")
    t2 = pd.read_csv(tmp_path / "ex.rel_m.tsv", sep="\t").set_index("PHENO")
    cols = ["REL_M", "REL_M_lowmaf", "REL_M_highmaf"]
    np.testing.assert_allclose(t1.loc[["y", "y2"], cols].to_numpy(), t2.loc[["y", "y2"], cols].to_numpy(), rtol=1e-9)
    per_snp = pd.read_csv(tmp_path / "ex.tsv.gz", sep="\t")
    v = _v_weights(G, covs if ncov else None)
    for name in ("y", "y2"):
        r = per_snp[f"REL_M_{name}"].to_numpy()
        want = [v @ r / v.sum()] + [(v * A[:, k]) @ r / (v * A[:, k]).sum() for k in range(2)]
        np.testing.assert_allclose(t2.loc[name, cols].to_numpy(dtype=float), want, rtol=1e-8)
    m1 = pd.read_csv(tmp_path / "tot.rel_m12.tsv", sep="\t")
    m2 = pd.read_csv(tmp_path / "ex.rel_m12.tsv", sep="\t")
    r12 = per_snp["REL_M12_y__y2"].to_numpy()
    assert float(m1.REL_M12[0]) == pytest.approx(float(m2.REL_M12[0]), rel=1e-9)
    assert float(m2.REL_M12[0]) == pytest.approx(v @ r12 / v.sum(), rel=1e-8)
    assert int(t1.loc["y", "PAIRS"]) == int(t2.loc["y", "PAIRS"]) > 0


@pytest.mark.parametrize("missing", [0.0, 0.01])
def test_totals_are_exact_for_phenotypes_measured_in_subsets(tmp_path, missing):
    """Phenotypes observed in different subsets (one where a covariate is constant, so the sample has a lower covariate
    rank) get exactly the per-SNP values of their own samples from the single pass, overall and per class."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, N=500, M=260, npairs=100, ncov=2, missing=missing)
    rng = np.random.default_rng(4)
    cv = pd.read_csv(cov_path, sep=" ", dtype={"FID": str, "IID": str})
    cv["grp"] = (rng.random(len(cv)) < 0.6).astype(float)
    cv.to_csv(tmp_path / "covar3.txt", sep=" ", index=False)
    y2 = _two_phenotypes(tmp_path, y, iid)
    ph = pd.DataFrame({"FID": iid, "IID": iid, "full": y, "ninth": y2, "grp1": y + y2})
    ph.loc[::9, "ninth"] = np.nan
    ph.loc[cv.grp.to_numpy() == 0, "grp1"] = np.nan
    ph.to_csv(tmp_path / "pheno3.txt", sep=" ", index=False, na_rep="NA")
    annot, A = _annot(tmp_path, G.shape[1])
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=str(tmp_path / "covar3.txt"), annot=annot)
    s1 = compute_relative_pair_totals(prefix, str(tmp_path / "pheno3.txt"), str(tmp_path / "tot"), step_size=37, **kw)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno3.txt"), str(tmp_path / "ex"), step_size=97, **kw)
    t1 = pd.read_csv(tmp_path / "tot.rel_m.tsv", sep="\t").set_index("PHENO")
    t2 = pd.read_csv(tmp_path / "ex.rel_m.tsv", sep="\t").set_index("PHENO")
    assert s1["phenotypes"]["grp1"]["n_covariates"] == 2 and s1["phenotypes"]["full"]["n_covariates"] == 3
    for name in ("full", "ninth", "grp1"):
        assert t1.loc[name, ["N", "N_COVARIATES", "PAIRS"]].tolist() == t2.loc[name, ["N", "N_COVARIATES", "PAIRS"]].tolist()
        np.testing.assert_allclose(t1.loc[name, ["REL_M", "REL_M_lowmaf", "REL_M_highmaf"]].to_numpy(dtype=float),
                                   t2.loc[name, ["REL_M", "REL_M_lowmaf", "REL_M_highmaf"]].to_numpy(dtype=float), rtol=1e-9)


def _sample_projection(G, Z, mask):
    """Genotypes and covariates of one complete-case sample, projected and normalized as SUMMIT's GWAS sees them:
    rows outside the sample are zero; returns (X, V, Q, n_eff), V = (1 - R2 on the covariates) (n - 1) / N_eff."""
    N, M = G.shape
    g = np.asarray(G[mask], dtype=np.float64)
    g = np.where(np.isnan(g), np.nanmean(g, 0)[None, :], g)
    g -= g.mean(0)
    n = int(mask.sum())
    U, sv, _ = np.linalg.svd(np.column_stack([np.ones(n), Z[mask]]), full_matrices=False)
    Q = U[:, sv > 1e-10 * sv[0]]
    n_eff = n - Q.shape[1]
    raw = (g ** 2).sum(0)
    g -= Q @ (Q.T @ g)
    ss = (g ** 2).sum(0)
    X = np.zeros((N, M))
    X[mask] = g * np.sqrt(n_eff / ss)[None, :]
    return X, np.minimum(1.0, ss / raw) * (n - 1) / n_eff, Q, n_eff


@pytest.mark.parametrize("missing", [0.0, 0.01])
def test_rel_m12_for_phenotypes_in_different_samples(tmp_path, missing):
    """REL_M12 of phenotypes in different complete-case samples (one with a lower covariate rank) equals
    y_a'(K_ab o R) y_b / sqrt(N_eff,a N_eff,b) / sum_j w_j, with K_ab = sum_j w_j x_a,j x_b,j' between the two samples'
    projected genotypes, R the related pairs of different individuals and w_j = a_jk sqrt(V_a,j V_b,j), overall, per
    class and outside the annotation; it is the related-pair part of SUMMIT's cross moment z*_a z*_b, and pairs in the
    same sample keep the existing values."""
    prefix, G, y, covs, iid, _ = _simulate(tmp_path, N=500, M=260, npairs=100, ncov=2, missing=missing)
    rng = np.random.default_rng(4)
    grp = (rng.random(len(iid)) < 0.6).astype(float)
    cdf = pd.DataFrame({"FID": iid, "IID": iid, "c0": covs[:, 0], "c1": covs[:, 1], "grp": grp})
    cdf.to_csv(tmp_path / "covar3.txt", sep=" ", index=False)
    y2 = _two_phenotypes(tmp_path, y, iid)
    ph = pd.DataFrame({"FID": iid, "IID": iid, "full": y, "ninth": y2, "grp1": y + y2,
                       "full2": y2 + rng.standard_normal(len(iid))})
    ph.loc[::9, "ninth"] = np.nan
    ph.loc[grp == 0, "grp1"] = np.nan
    ph.to_csv(tmp_path / "pheno4.txt", sep=" ", index=False, na_rep="NA")
    M = G.shape[1]
    b = (np.random.default_rng(5).random(M) < 0.4).astype(int)
    A = np.column_stack([b, 1 - b]).astype(float)
    A[-30:] = 0.0                                          # SNPs outside the annotation
    pd.DataFrame({**_snp_meta(M), "CM": 0, "lowmaf": b, "highmaf": 1 - b}).iloc[:-30].to_csv(
        tmp_path / "annot.txt", sep="\t", index=False)
    annot = str(tmp_path / "annot.txt")
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=str(tmp_path / "covar3.txt"), annot=annot, cross=True)
    s1 = compute_relative_pair_totals(prefix, str(tmp_path / "pheno4.txt"), str(tmp_path / "tot"), step_size=37, **kw)
    assert s1["cross_pairs"] == {"same_sample": 1, "different_samples": 5}
    tab = pd.read_csv(tmp_path / "tot.rel_m12.tsv", sep="\t")
    assert list(zip(tab.PHENO1, tab.PHENO2)) == [("full", "ninth"), ("full", "grp1"), ("full", "full2"),
                                                 ("ninth", "grp1"), ("ninth", "full2"), ("grp1", "full2")]
    Z = np.column_stack([covs, grp])
    fam = read_relative_pairs(str(tmp_path / "pairs.txt"), pd.DataFrame({"FID": iid, "IID": iid})).family
    R = (fam[:, None] == fam[None, :]) & ~np.eye(len(iid), dtype=bool)
    proj = {}
    for name in ph.columns[2:]:
        mask = np.isfinite(ph[name].to_numpy())
        X, V, Q, n_eff = _sample_projection(G, Z, mask)
        yy = np.zeros(len(iid))
        v = ph[name].to_numpy()[mask]
        v = v - Q @ (Q.T @ v)
        yy[mask] = v * np.sqrt(n_eff / (v ** 2).sum())
        proj[name] = (X, V, yy, n_eff, mask)
    for _, row in tab.iterrows():
        Xa, Va, ya, na, ma = proj[row.PHENO1]
        Xb, Vb, yb, nb, mb = proj[row.PHENO2]
        assert row.N == int((ma & mb).sum())
        assert row.PAIRS == int((R & (ma[:, None] & mb[None, :] | mb[:, None] & ma[None, :])).sum() // 2)
        for col, a_k in (("REL_M12", np.ones(M)), ("REL_M12_lowmaf", A[:, 0]), ("REL_M12_highmaf", A[:, 1])):
            w = a_k * np.sqrt(Va * Vb)
            K = (Xa * w[None, :]) @ Xb.T
            want = ya @ (K * R) @ yb / np.sqrt(na * nb) / w.sum()
            assert row[col] == pytest.approx(want, rel=1e-9, abs=1e-12), (row.PHENO1, row.PHENO2, col)
        if not np.array_equal(ma, mb):                     # the related-pair part of SUMMIT's cross moment
            zs = []
            for X, yy, ne, m in ((Xa, ya, na, ma), (Xb, yb, nb, mb)):
                r, n = (X.T @ yy) / ne, int(m.sum())
                zs.append(exact_score_z_from_arrays(r, np.sqrt((1 - r * r) / (ne - 1)), np.full(M, n), n, n - ne - 1))
            za, zb = zs
            w = np.sqrt(Va * Vb)
            K = (Xa * w[None, :]) @ Xb.T
            assert w @ (za * zb) - row.REL_M12 * w.sum() == pytest.approx(ya @ (K * ~R) @ yb / np.sqrt(na * nb), rel=1e-9)
    c = extract_rel_m12(str(tmp_path / "tot.rel_m12.tsv"), "grp1", "ninth", str(tmp_path / "m12.tsv"), annot=annot)
    m12 = pd.read_csv(tmp_path / "m12.tsv", sep="\t")
    r = tab[(tab.PHENO1 == "ninth") & (tab.PHENO2 == "grp1")].iloc[0]
    assert c == pytest.approx(r.REL_M12) and list(m12.columns) == ["SNP", "REL_M12"]
    np.testing.assert_allclose(m12.REL_M12.to_numpy(), A[:-30] @ r[["REL_M12_lowmaf", "REL_M12_highmaf"]].to_numpy(float))
    # with --inbreeding the cross-sample pairs carry no REL_S12, and cross-m12 then writes REL_M12 alone
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno4.txt"), str(tmp_path / "inb"), step_size=37,
                                 inbreeding=True, **kw)
    ti = pd.read_csv(tmp_path / "inb.rel_m12.tsv", sep="\t")
    num = [c_ for c_ in tab.columns if c_.startswith("REL_M12")]
    np.testing.assert_allclose(ti[num].to_numpy(float), tab[num].to_numpy(float), rtol=1e-12)
    same = (ti.PHENO1 == "full") & (ti.PHENO2 == "full2")
    assert ti.loc[same, "REL_S12"].notna().all() and ti.loc[~same, ["REL_S12", "REL_S12_lowmaf"]].isna().all().all()
    extract_rel_m12(str(tmp_path / "inb.rel_m12.tsv"), "full", "grp1", str(tmp_path / "m12i.tsv"))
    assert list(pd.read_csv(tmp_path / "m12i.tsv", sep="\t").columns) == ["REL_M12"]


def test_append_writes_constant_or_per_class_rel_m(tmp_path):
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2)
    M = G.shape[1]
    annot, A = _annot(tmp_path, M)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "tot"),
                                 pairs=str(tmp_path / "pairs.txt"), covar=cov_path, annot=annot)
    X, yy, p, n_eff = _projected(G, y, covs)
    _ols_sumstats(tmp_path / "ss.tsv", X, yy, p)
    tab = pd.read_csv(tmp_path / "tot.rel_m.tsv", sep="\t").iloc[0]
    assert append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / "tot.rel_m.tsv"), "y",
                                    str(tmp_path / "a.tsv")) == 0
    a = pd.read_csv(tmp_path / "a.tsv", sep="\t")
    np.testing.assert_allclose(a.REL_M.to_numpy(), tab.REL_M)
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / "tot.rel_m.tsv"), "y", str(tmp_path / "b.tsv"),
                             annot=annot)
    b = pd.read_csv(tmp_path / "b.tsv", sep="\t")
    np.testing.assert_allclose(b.REL_M.to_numpy(), A @ np.array([tab.REL_M_lowmaf, tab.REL_M_highmaf]))
    with pytest.raises(ValueError, match="no single row"):
        append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / "tot.rel_m.tsv"), "zz", str(tmp_path / "c.tsv"))


def test_constant_rel_m_keeps_the_h2_point_estimate(tmp_path):
    """The single-component estimate depends on REL_M only through sum_j V_j REL_M_j, so the per-phenotype value gives
    the exact per-SNP estimate when the LD scores' V weights are those of the GWAS sample; only the SE changes."""
    from summit.inference.sumrhe import Sumrhe
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, N=400, M=400, npairs=80, ncov=2, missing=0.0)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "ex"), **kw)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "tot"), **kw)
    X, yy, p, n_eff = _projected(G, y, covs)
    N, M = X.shape
    _ols_sumstats(tmp_path / "ss.tsv", X, yy, p)
    for f, rel in (("sx.tsv", "ex.tsv.gz"), ("st.tsv", "tot.rel_m.tsv")):
        append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / rel), "y", str(tmp_path / f))
    v = _v_weights(G, covs)
    ld = ((X.T @ X / n_eff) ** 2).sum(1) - M / n_eff
    rel_ld, _ = _ld_rel(G, covs, str(tmp_path / "pairs.txt"), iid, v[:, None])
    pd.DataFrame({**_snp_meta(M), "V": v, "L2_0": ld, "LD_REL_L2_0": rel_ld[:, 0]})[
        ["CHR", "SNP", "BP", "V", "L2_0", "LD_REL_L2_0"]].to_csv(
        tmp_path / "in.gw.ldscore.gz", sep="\t", index=False, float_format="%.10g")
    fits = {}
    for f in ("sx.tsv", "st.tsv"):
        s = Sumrhe(h2_path=str(tmp_path / f), ldscores=str(tmp_path / "in.gw.ldscore.gz"), njack="20", chisq_threshold=None)
        s._run()
        fits[f] = s.results[0].h2[-1]
    assert fits["st.tsv"][0] == pytest.approx(fits["sx.tsv"][0], rel=1e-6)
    assert fits["st.tsv"][1] == pytest.approx(fits["sx.tsv"][1], rel=0.1)


def test_rg_with_one_rel_m12_value_matches_per_snp(tmp_path):
    """rg with the pair's single REL_M12 value (file or number) gives the per-SNP point estimate."""
    from summit.inference.sumcore import Sumcore
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2, missing=0.0)
    y2 = _two_phenotypes(tmp_path, y, iid)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path, cross=True)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "ex"), **kw)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "tot"), **kw)
    X, y1t, p, n_eff = _projected(G, y, covs)
    _, y2t, _, _ = _projected(G, y2, covs)
    N, M = X.shape
    _ols_sumstats(tmp_path / "s1.tsv", X, y1t, p)
    _ols_sumstats(tmp_path / "s2.tsv", X, y2t, p)
    for tag, rel in (("x", "ex.tsv.gz"), ("t", "tot.rel_m.tsv")):
        append_rel_m_to_sumstats(str(tmp_path / "s1.tsv"), str(tmp_path / rel), "y", str(tmp_path / f"a{tag}.tsv"))
        append_rel_m_to_sumstats(str(tmp_path / "s2.tsv"), str(tmp_path / rel), "y2", str(tmp_path / f"b{tag}.tsv"))
    vx = extract_rel_m12(str(tmp_path / "ex.tsv.gz"), "y", "y2", str(tmp_path / "m12x.tsv"))
    vt = extract_rel_m12(str(tmp_path / "tot.rel_m12.tsv"), "y2", "y", str(tmp_path / "m12t.tsv"))
    assert list(pd.read_csv(tmp_path / "m12t.tsv", sep="\t").columns) == ["REL_M12"]
    v = _v_weights(G, covs)
    ld = ((X.T @ X / n_eff) ** 2).sum(1) - M / n_eff
    rel_ld, _ = _ld_rel(G, covs, str(tmp_path / "pairs.txt"), iid, v[:, None])
    pd.DataFrame({**_snp_meta(M), "V": v, "L2_0": ld, "LD_REL_L2_0": rel_ld[:, 0]})[
        ["CHR", "SNP", "BP", "V", "L2_0", "LD_REL_L2_0"]].to_csv(
        tmp_path / "in.gw.ldscore.gz", sep="\t", index=False, float_format="%.10g")
    common = dict(ldscores=str(tmp_path / "in.gw.ldscore.gz"), njack="chr", chisq_threshold=None,
                  intercept_chisq_thr=None, intercept_rg=1.0, align_alleles=False)
    gx = Sumcore(rg=f"{tmp_path / 'ax.tsv'},{tmp_path / 'bx.tsv'}", rel_m12=str(tmp_path / "m12x.tsv"), **common)._run()
    gt = Sumcore(rg=f"{tmp_path / 'at.tsv'},{tmp_path / 'bt.tsv'}", rel_m12=str(tmp_path / "m12t.tsv"), **common)._run()
    gn = Sumcore(rg=f"{tmp_path / 'at.tsv'},{tmp_path / 'bt.tsv'}", rel_m12=repr(vt), **common)._run()
    gam = lambda r: float(r["rg_fit"].gamma_total[0])
    assert gam(gt) == pytest.approx(gam(gx), rel=1e-6)
    assert gam(gn) == pytest.approx(gam(gt), rel=1e-9)
    assert vt == pytest.approx(v @ pd.read_csv(tmp_path / "m12x.tsv", sep="\t").REL_M12.to_numpy() / v.sum(), rel=1e-8)


# ---------------------------------------------------------------------------------------- --inbreeding (diagonal terms)
def _kernels_tilde(X, v, A):
    """SUMMIT's kernels K~_k = sum_j a_jk V_j x_j x_j' / sum_j a_jk V_j (column 0 of the list: all SNPs)."""
    out = []
    for a in [np.ones(X.shape[1])] + [A[:, c] for c in range(A.shape[1])]:
        w = a * v
        out.append((X * w[None, :]) @ X.T / w.sum())
    return out


@pytest.mark.parametrize("ncov", [0, 2])
def test_inbreeding_values_match_brute_force(tmp_path, ncov):
    """REL_S, REL_LAMBDA and REL_DIAG (overall, per class and per pair of classes) equal their definitions."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov, missing=0.0)
    M = G.shape[1]
    annot, A = _annot(tmp_path, M)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "tot"), pairs=str(tmp_path / "pairs.txt"),
                                 covar=cov_path, annot=annot, inbreeding=True, step_size=61)
    tab = pd.read_csv(tmp_path / "tot.rel_m.tsv", sep="\t").iloc[0]
    X, yy, p, n_eff = _projected(G, y, covs if ncov else None)
    N = X.shape[0]
    v = _v_weights(G, covs if ncov else None)
    Q = np.linalg.qr(np.column_stack([np.ones(N)] + ([covs] if ncov else [])))[0]
    H = Q @ Q.T
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    Ks = _kernels_tilde(X, v, A)
    names = ["", "_lowmaf", "_highmaf"]
    for K, nm in zip(Ks, names):
        d = np.diag(K)
        np.testing.assert_allclose(d.sum(), n_eff, rtol=1e-10)
        assert tab[f"REL_S{nm}"] == pytest.approx(float((d - 1.0) @ (yy * yy)) / n_eff, rel=1e-8, abs=1e-12)
        lam = float(d @ np.diag(H)) + 2.0 * float((K[pi, pj] * H[pi, pj]).sum())
        assert tab[f"REL_LAMBDA{nm}"] == pytest.approx(lam, rel=1e-8)
    assert tab["REL_DIAG"] == pytest.approx(float(np.diag(Ks[0]) @ np.diag(Ks[0])) - n_eff, rel=1e-7)
    cls = ["lowmaf", "highmaf"]
    for c, Kc in zip(cls, Ks[1:]):
        for d_, Kd in zip(cls, Ks[1:]):
            assert tab[f"REL_DIAG_{c}__{d_}"] == pytest.approx(float(np.diag(Kc) @ np.diag(Kd)) - n_eff, rel=1e-7)
    # the single-component value is the class-mass-weighted combination of the class values
    w = np.array([(A[:, c] * v).sum() for c in range(2)]); w /= w.sum()
    S = np.array([[tab[f"REL_DIAG_{c}__{d_}"] for d_ in cls] for c in cls])
    assert tab["REL_DIAG"] == pytest.approx(float(w @ S @ w), rel=1e-7)
    for c, row in zip(cls, S):
        assert tab[f"REL_DIAG_{c}"] == pytest.approx(float(row @ w), rel=1e-9)


def _exact_offdiagonal(G, y, covs, pi, pj, A=None, drop_diagonal=True, exact_leverage=True):
    """Exact individual-level method of moments with kernels K_k = P Xs_k Xs_k' P / M_k (Xs standardized genotypes):
    the first equations use pairs outside the related pairs (and, with drop_diagonal, no diagonal terms), with
    coefficient tr(K^S_k P) (or tr(K_k), SUMMIT's convention without the leverage term); returns total h2."""
    G = np.asarray(G, dtype=np.float64)
    N, M = G.shape
    Xs = (G - G.mean(0)) / G.std(0, ddof=1)
    Q = np.linalg.qr(np.column_stack([np.ones(N)] + ([covs] if covs is not None else [])))[0]
    P = np.eye(N) - Q @ Q.T
    PXs = P @ Xs
    yp = P @ y
    cols = [np.ones(M)] if A is None else [A[:, c] for c in range(A.shape[1])]
    Ks = [(PXs * a[None, :]) @ PXs.T / a.sum() for a in cols]
    C = len(Ks)
    L = np.zeros((C + 1, C + 1)); rhs = np.zeros(C + 1)
    for k, Kk in enumerate(Ks):
        S = Kk.copy()
        S[pi, pj] = 0.0; S[pj, pi] = 0.0
        if drop_diagonal:
            np.fill_diagonal(S, 0.0)
        for l, Kl in enumerate(Ks):
            L[k, l] = (S * Kl).sum()
        L[k, C] = np.trace(S @ P) if exact_leverage else np.trace(Kk)
        rhs[k] = yp @ S @ yp
        L[C, k] = np.trace(Kk)
    L[C, C] = np.trace(P); rhs[C] = yp @ yp
    sig = np.linalg.solve(L, rhs)
    return sig[:C].sum() / sig.sum()


def _write_fit_inputs(tmp_path, G, y, covs, iid, A=None, names=None):
    """Sumstats with the covariate rank, and exact V-weighted in-sample LD scores with their related-pair parts."""
    X, yy, p, n_eff = _projected(G, y, covs)
    N, M = X.shape
    r = (X.T @ yy) / n_eff
    pd.DataFrame({"SNP": [f"rs{j}" for j in range(M)], "A1": "A", "A2": "G", "N": N, "BETA": r,
                  "SE": np.sqrt((1 - r * r) / (N - p - 2)), "COV_RANK": p}).to_csv(
        tmp_path / "ss.tsv", sep="\t", index=False, float_format="%.15g")
    v = _v_weights(G, covs)
    R2 = (X.T @ X / n_eff) ** 2
    W = v[:, None] if A is None else v[:, None] * A
    ld = R2 @ W - W.sum(0)[None, :] / n_eff
    rel, _ = _ld_rel(G, covs, str(tmp_path / "pairs.txt"), iid, W)
    df = pd.DataFrame(_snp_meta(M))[["CHR", "SNP", "BP"]]
    df["V"] = v
    for k in range(W.shape[1]):
        df[f"L2_{k}"] = ld[:, k]
    for k in range(W.shape[1]):
        df[f"LD_REL_L2_{k}"] = rel[:, k]
    df.to_csv(tmp_path / "in.gw.ldscore.gz", sep="\t", index=False, float_format="%.15g")
    fit_annot = None
    if A is not None:
        fa = pd.DataFrame(_snp_meta(M))[["CHR", "BP", "SNP"]]
        for k, nm in enumerate(names):
            fa[nm] = A[:, k]
        fa.to_csv(tmp_path / "fit_annot.txt", sep="\t", index=False)
        fit_annot = str(tmp_path / "fit_annot.txt")
    return str(tmp_path / "ss.tsv"), str(tmp_path / "in.gw.ldscore.gz"), fit_annot


@pytest.mark.parametrize("partitioned,njack", [(False, "20"), (True, "20"), (False, "chr"), (True, "chr")])
def test_inbreeding_h2_equals_exact_estimator_without_diagonal_terms(tmp_path, partitioned, njack):
    """moments --inbreeding, append and --h2 reproduce the exact individual-level estimator whose first equations use
    only pairs of different, unrelated individuals (with exact LD scores and complete genotypes); without the option
    the fit is the usual pair-excluded estimator."""
    from summit.inference.sumrhe import Sumrhe
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, N=360, M=300, npairs=70, ncov=2, missing=0.0)
    rng = np.random.default_rng(11)
    y = y + rng.standard_normal(y.size) * np.linspace(0.5, 1.5, y.size)        # residual variance varies between people
    pd.DataFrame({"FID": iid, "IID": iid, "y": y}).to_csv(tmp_path / "pheno.txt", sep=" ", index=False)
    annot, A = _annot(tmp_path, G.shape[1]) if partitioned else (None, None)
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path, annot=annot)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "inb"), inbreeding=True, **kw)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "std"), **kw)
    ss, ld, fit_annot = _write_fit_inputs(tmp_path, G, y, covs, iid, A, ["lowmaf", "highmaf"])
    for tag in ("inb", "std"):
        append_rel_m_to_sumstats(ss, str(tmp_path / f"{tag}.rel_m.tsv"), "y", str(tmp_path / f"{tag}.tsv"), annot=annot)
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    fit = {}
    for tag in ("inb", "std"):
        s = Sumrhe(h2_path=str(tmp_path / f"{tag}.tsv"), ldscores=ld, annot=fit_annot, njack=njack, chisq_threshold=None)
        s._run()
        fit[tag] = float(s.results[0].h2[-1, 0])
        assert np.isfinite(s.results[0].h2[-1, 1])
    exact_inb = _exact_offdiagonal(G, y, covs, pi, pj, A)
    exact_std = _exact_offdiagonal(G, y, covs, pi, pj, A, drop_diagonal=False, exact_leverage=False)
    assert fit["inb"] == pytest.approx(exact_inb, rel=1e-7)
    assert fit["std"] == pytest.approx(exact_std, rel=1e-7)
    assert abs(fit["inb"] - fit["std"]) > 1e-6


def test_inbreeding_columns_are_appended_and_validated(tmp_path):
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2, missing=0.0)
    annot, A = _annot(tmp_path, G.shape[1])
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "tot"), pairs=str(tmp_path / "pairs.txt"),
                                 covar=cov_path, annot=annot, inbreeding=True)
    X, yy, p, n_eff = _projected(G, y, covs)
    _ols_sumstats(tmp_path / "ss.tsv", X, yy, p)
    tab = pd.read_csv(tmp_path / "tot.rel_m.tsv", sep="\t").iloc[0]
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / "tot.rel_m.tsv"), "y", str(tmp_path / "a.tsv"))
    a = pd.read_csv(tmp_path / "a.tsv", sep="\t")
    for col in ("REL_S", "REL_LAMBDA", "REL_DIAG"):
        np.testing.assert_allclose(a[col].to_numpy(), tab[col])
    assert not any(c.startswith("REL_DIAG_") for c in a.columns)
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / "tot.rel_m.tsv"), "y", str(tmp_path / "b.tsv"),
                             annot=annot)
    b = pd.read_csv(tmp_path / "b.tsv", sep="\t")
    cls = ["lowmaf", "highmaf"]
    np.testing.assert_allclose(b.REL_S.to_numpy(), A @ np.array([tab[f"REL_S_{c}"] for c in cls]))
    for d in cls:
        np.testing.assert_allclose(b[f"REL_DIAG_{d}"].to_numpy(), A @ np.array([tab[f"REL_DIAG_{c}__{d}"] for c in cls]))
    ss = Sumstats.from_file(str(tmp_path / "b.tsv"), name="b")
    assert set(ss.rel_diag) == {"REL_S", "REL_LAMBDA", "REL_DIAG", "REL_DIAG_lowmaf", "REL_DIAG_highmaf"}
    b.drop(columns=["REL_LAMBDA"]).to_csv(tmp_path / "c.tsv", sep="\t", index=False)
    with pytest.raises(RuntimeError, match="some but not all"):
        Sumstats.from_file(str(tmp_path / "c.tsv"), name="c")
    b.drop(columns=["REL_M"]).to_csv(tmp_path / "d.tsv", sep="\t", index=False)
    with pytest.raises(RuntimeError, match="no REL_M|need REL_M"):
        Sumstats.from_file(str(tmp_path / "d.tsv"), name="d")


def test_inbreeding_per_snp_path_matches_totals(tmp_path):
    """moments --per-snp --inbreeding writes the same diagonal-term values (and per-class REL_M12, REL_S12) as the
    one-pass totals, and append/cross-m12 take them from the per-SNP run's PREFIX.rel_m(12).tsv."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2, missing=0.004)
    _two_phenotypes(tmp_path, y, iid)
    annot, A = _annot(tmp_path, G.shape[1])
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path, cross=True, annot=annot, inbreeding=True)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "tot"), step_size=61, **kw)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "ex"), step_size=97, **kw)
    t1 = pd.read_csv(tmp_path / "tot.rel_m.tsv", sep="\t").set_index("PHENO")
    t2 = pd.read_csv(tmp_path / "ex.rel_m.tsv", sep="\t").set_index("PHENO")
    assert list(t1.columns) == list(t2.columns) and "REL_DIAG_lowmaf__highmaf" in t1.columns
    np.testing.assert_allclose(t1.loc[["y", "y2"]].to_numpy(float), t2.loc[["y", "y2"]].to_numpy(float), rtol=1e-8, atol=1e-12)
    c1 = pd.read_csv(tmp_path / "tot.rel_m12.tsv", sep="\t")
    c2 = pd.read_csv(tmp_path / "ex.rel_m12.tsv", sep="\t")
    assert {"REL_M12_lowmaf", "REL_S12", "REL_S12_highmaf"} <= set(c1.columns)
    num = [c for c in c1.columns if c.startswith("REL_")]
    np.testing.assert_allclose(c1[num].to_numpy(float), c2[num].to_numpy(float), rtol=1e-8, atol=1e-12)
    X, yy, p, n_eff = _projected(G, y, covs)
    _ols_sumstats(tmp_path / "ss.tsv", X, yy, p)
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), str(tmp_path / "ex.tsv.gz"), "y", str(tmp_path / "a.tsv"), annot=annot)
    a = pd.read_csv(tmp_path / "a.tsv", sep="\t")
    np.testing.assert_allclose(a.REL_S.to_numpy(), A @ t1.loc["y", ["REL_S_lowmaf", "REL_S_highmaf"]].to_numpy(float))
    assert "REL_DIAG_highmaf" in a.columns
    extract_rel_m12(str(tmp_path / "tot.rel_m12.tsv"), "y", "y2", str(tmp_path / "m12.tsv"), annot=annot)
    m12 = pd.read_csv(tmp_path / "m12.tsv", sep="\t")
    row = c1.iloc[0]
    np.testing.assert_allclose(m12.REL_M12.to_numpy(), A @ row[["REL_M12_lowmaf", "REL_M12_highmaf"]].to_numpy(float))
    np.testing.assert_allclose(m12.REL_S12.to_numpy(), A @ row[["REL_S12_lowmaf", "REL_S12_highmaf"]].to_numpy(float))
    extract_rel_m12(str(tmp_path / "ex.tsv.gz"), "y2", "y", str(tmp_path / "m12x.tsv"))
    assert set(pd.read_csv(tmp_path / "m12x.tsv", sep="\t").columns) == {"SNP", "REL_M12", "REL_S12"}


def _exact_sigmas(G, covs, pi, pj, ya, yb, A=None):
    """Exact method-of-moments components (sigma_1..C, sigma_e) without related pairs and diagonal terms, for the
    cross products of ya and yb (ya = yb for h2)."""
    G = np.asarray(G, dtype=np.float64)
    N, M = G.shape
    Xs = (G - G.mean(0)) / G.std(0, ddof=1)
    Q = np.linalg.qr(np.column_stack([np.ones(N)] + ([covs] if covs is not None else [])))[0]
    P = np.eye(N) - Q @ Q.T
    PXs = P @ Xs
    pa, pb = P @ ya, P @ yb
    cols = [np.ones(M)] if A is None else [A[:, c] for c in range(A.shape[1])]
    Ks = [(PXs * a[None, :]) @ PXs.T / a.sum() for a in cols]
    C = len(Ks)
    L = np.zeros((C + 1, C + 1)); rhs = np.zeros(C + 1)
    for k, Kk in enumerate(Ks):
        S = Kk.copy()
        S[pi, pj] = 0.0; S[pj, pi] = 0.0
        np.fill_diagonal(S, 0.0)
        for l, Kl in enumerate(Ks):
            L[k, l] = (S * Kl).sum()
        L[k, C] = np.trace(S @ P)
        rhs[k] = pa @ S @ pb
        L[C, k] = np.trace(Kk)
    L[C, C] = np.trace(P); rhs[C] = pa @ pb
    return np.linalg.solve(L, rhs)


@pytest.mark.parametrize("partitioned", [False, True])
def test_inbreeding_rg_equals_exact_estimator_without_diagonal_terms(tmp_path, partitioned):
    """--rg with the --inbreeding values (both sumstats and REL_S12 from cross-m12) reproduces the exact genetic
    correlation whose h2 and cross-trait equations use only pairs of different, unrelated individuals."""
    from summit.inference.sumcore import Sumcore
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, N=360, M=300, npairs=70, ncov=2, missing=0.0)
    rng = np.random.default_rng(21)
    Gs = np.asarray(G, dtype=np.float64)
    Gs = (Gs - Gs.mean(0)) / Gs.std(0)
    g1 = Gs @ rng.normal(0, np.sqrt(0.6 / G.shape[1]), G.shape[1])
    g2 = 0.6 * g1 + Gs @ rng.normal(0, np.sqrt(0.3 / G.shape[1]), G.shape[1])
    y = g1 + 0.5 * y                                              # keeps the shared family effect and covariates
    y2 = g2 + 0.3 * y + rng.standard_normal(y.size) * np.linspace(0.6, 1.2, y.size)
    pd.DataFrame({"FID": iid, "IID": iid, "y": y, "y2": y2}).to_csv(tmp_path / "pheno2.txt", sep=" ", index=False)
    annot, A = _annot(tmp_path, G.shape[1]) if partitioned else (None, None)
    compute_relative_pair_totals(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / "tot"), pairs=str(tmp_path / "pairs.txt"),
                                 covar=cov_path, annot=annot, cross=True, inbreeding=True)
    s1, ld, fit_annot = _write_fit_inputs(tmp_path, G, y, covs, iid, A, ["lowmaf", "highmaf"])
    X, y2t, p, n_eff = _projected(G, y2, covs)
    N, M = X.shape
    r = (X.T @ y2t) / n_eff
    pd.DataFrame({"SNP": [f"rs{j}" for j in range(M)], "A1": "A", "A2": "G", "N": N, "BETA": r,
                  "SE": np.sqrt((1 - r * r) / (N - p - 2)), "COV_RANK": p}).to_csv(
        tmp_path / "s2.tsv", sep="\t", index=False, float_format="%.15g")
    append_rel_m_to_sumstats(s1, str(tmp_path / "tot.rel_m.tsv"), "y", str(tmp_path / "a.tsv"), annot=annot)
    append_rel_m_to_sumstats(str(tmp_path / "s2.tsv"), str(tmp_path / "tot.rel_m.tsv"), "y2", str(tmp_path / "b.tsv"),
                             annot=annot)
    extract_rel_m12(str(tmp_path / "tot.rel_m12.tsv"), "y", "y2", str(tmp_path / "m12.tsv"), annot=annot)
    _, y1t, _, _ = _projected(G, y, covs)
    c_ov = float(y1t @ y2t / n_eff)
    common = dict(ldscores=ld, annot=fit_annot, njack="chr", chisq_threshold=None, intercept_chisq_thr=None,
                  intercept_rg=c_ov, align_alleles=False)
    res = Sumcore(rg=f"{tmp_path / 'a.tsv'},{tmp_path / 'b.tsv'}", rel_m12=str(tmp_path / "m12.tsv"), **common)._run()
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    s11 = _exact_sigmas(G, covs, pi, pj, y, y, A)
    s22 = _exact_sigmas(G, covs, pi, pj, y2, y2, A)
    s12 = _exact_sigmas(G, covs, pi, pj, y, y2, A)
    rg_exact = s12[:-1].sum() / np.sqrt(s11[:-1].sum() * s22[:-1].sum())
    assert np.isfinite(rg_exact) and abs(rg_exact) > 0.1
    assert float(res["rg_fit"].rg_total[0]) == pytest.approx(rg_exact, rel=1e-6)
    h2_1 = s11[:-1].sum() / s11.sum()
    assert float(res["h2_fit1"].h2[-1, 0]) == pytest.approx(h2_1, rel=1e-6)
    # every input is required, and an estimated intercept is refused
    with pytest.raises(ValueError, match="REL_S12 in --rel-m12 no"):
        Sumcore(rg=f"{tmp_path / 'a.tsv'},{tmp_path / 'b.tsv'}", rel_m12="0.01", **common)._run()
    free = {**common, "intercept_rg": None}
    with pytest.raises(ValueError, match="overlapping covariance"):
        Sumcore(rg=f"{tmp_path / 'a.tsv'},{tmp_path / 'b.tsv'}", rel_m12=str(tmp_path / "m12.tsv"), **free)._run()


def test_pair_products_in_chunks_match_one_chunk(tmp_path, monkeypatch):
    """REL_M (totals and per-SNP, with cross-trait and diagonal terms) and LD_REL do not depend on how the related
    pairs are split into chunks; with a tiny chunk every block is split into many pieces."""
    import summit.relatives.moments as mom
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2, missing=0.004)
    _two_phenotypes(tmp_path, y, iid)
    annot, A = _annot(tmp_path, G.shape[1])
    kw = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path, cross=True, annot=annot, inbreeding=True)

    def run(tag):
        compute_relative_pair_totals(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / f"tot{tag}"), step_size=61, **kw)
        compute_relative_pair_moments(prefix, str(tmp_path / "pheno2.txt"), str(tmp_path / f"ex{tag}"), step_size=97, **kw)
        ld, _ = _ld_rel(G, covs, str(tmp_path / "pairs.txt"), iid, A)
        return ([pd.read_csv(tmp_path / f"{f}{tag}.rel_m.tsv", sep="\t").set_index("PHENO") for f in ("tot", "ex")]
                + [pd.read_csv(tmp_path / f"{f}{tag}.rel_m12.tsv", sep="\t") for f in ("tot", "ex")]
                + [pd.read_csv(tmp_path / f"ex{tag}.tsv.gz", sep="\t")], ld)

    base, ld0 = run("a")
    monkeypatch.setattr(mom, "_PAIR_CHUNK_ELEMS", 7)
    small, ld1 = run("b")
    for t0, t1 in zip(base, small):
        num = [c for c in t0.columns if t0[c].dtype.kind == "f"]
        np.testing.assert_allclose(t0[num].to_numpy(float), t1[num].to_numpy(float), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(ld0, ld1, rtol=1e-10, atol=1e-13)
    a, b = pairs_within_sample(read_relative_pairs(str(tmp_path / "pairs.txt"), pd.DataFrame({"FID": iid, "IID": iid})),
                               np.arange(G.shape[0]), G.shape[0])
    C = np.linalg.qr(np.column_stack([np.ones(G.shape[0]), covs]))[0]
    ld2 = relative_pair_ld_rel(lambda s, e: G[:, s:e], G.shape[1], a, b, A, C, G.shape[0] - 3, step_size=97, pair_chunk=3)
    np.testing.assert_allclose(ld0, ld2, rtol=1e-10, atol=1e-13)
