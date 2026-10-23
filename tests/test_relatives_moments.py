import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.relatives.moments import (
    adjust_ldscores,
    append_rel_m_to_sumstats,
    compute_relative_pair_moments,
    extract_rel_m12,
    read_relative_pairs,
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
    kappa = (N - 1.0) * (N - p - 2.0) / ((N - 2.0) * n_eff ** 2)
    brute = kappa * 2.0 * ((X[pi] * X[pj]) * (yy[pi] * yy[pj])[:, None]).sum(0)
    np.testing.assert_allclose(tab["REL_M_y"].to_numpy(), brute, rtol=1e-8, atol=1e-12)


def test_family_closure_adds_chain_pair(tmp_path):
    _, _, _, _, iid, _ = _simulate(tmp_path, ncov=0)
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    pairs = set(zip(pi.tolist(), pj.tolist()))
    assert (0, 1) in pairs and (1, 3) in pairs and (0, 3) in pairs and (2, 3) in pairs
    rp = read_relative_pairs(str(tmp_path / "pairs.txt"), pd.DataFrame({"FID": iid, "IID": iid}), closure=False)
    assert (0, 3) not in set(zip(rp.i.tolist(), rp.j.tolist()))


@pytest.mark.parametrize("ncov,rtol", [(0, 1e-9), (2, 2e-3)])
def test_pair_excluded_quadratic_form_identity(tmp_path, ncov, rtol):
    """(n_scale/M) sum_j (m_j - REL_M_j) equals y'K_u y, K_u = GRM with related pairs removed.

    Exact without covariates; with p covariates SUMMIT's h2 moment uses the intercept-only (cov_rank 0)
    convention, which matches the projected scale up to O(p^2/N^2)."""
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
    m = h2_moment_from_arrays(beta=beta, se=se, n_obs=np.full(M, float(N)), nsamp=float(N), cov_rank=0)
    lhs = (N - 1.0) / M * (m - rel).sum()
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    K = X @ X.T / M
    K[pi, pj] = 0.0
    K[pj, pi] = 0.0
    rhs = yy @ K @ yy * (N - 1.0) / n_eff
    np.testing.assert_allclose(lhs, rhs, rtol=rtol)


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


@pytest.mark.parametrize("ncov", [0, 2])
def test_ld_rel_partitioned_matches_brute_force(tmp_path, ncov):
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov)
    M = G.shape[1]
    b = (np.random.default_rng(5).random(M) < 0.4).astype(int)
    annot = pd.DataFrame({**_snp_meta(M), "binA": b, "binB": 1 - b})
    annot.to_csv(tmp_path / "annot.txt", sep=" ", index=False)
    common = dict(pairs=str(tmp_path / "pairs.txt"), covar=cov_path, step_size=97, ld_rel=True)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "relp"),
                                  annot=str(tmp_path / "annot.txt"), **common)
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), str(tmp_path / "rel1"), **common)
    X, _, p, n_eff = _projected(G, y, covs if ncov else None)
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    brute = _brute_ld_rel(X, pi, pj, n_eff, annot[["binA", "binB"]].to_numpy(float))
    tab = pd.read_csv(tmp_path / "relp.tsv.gz", sep="\t")
    np.testing.assert_allclose(tab[["LD_REL_binA", "LD_REL_binB"]].to_numpy(), brute, rtol=1e-8, atol=1e-10)
    one = pd.read_csv(tmp_path / "rel1.tsv.gz", sep="\t")["LD_REL"].to_numpy()
    np.testing.assert_allclose(one, brute.sum(1), rtol=1e-8, atol=1e-10)


def test_missing_phenotype_rel_m_and_ld_rel_sample(tmp_path):
    """REL_M uses the phenotype's complete cases; LD_REL uses every sample with complete covariates."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=2)
    ph = pd.read_csv(tmp_path / "pheno.txt", sep=" ", dtype={"FID": str, "IID": str})
    ph.loc[::7, "y"] = np.nan
    ph.to_csv(tmp_path / "pheno.txt", sep=" ", index=False, na_rep="NA")
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path, step_size=97, ld_rel=True)
    tab = pd.read_csv(out + ".tsv.gz", sep="\t")
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    X, _, _, n_eff = _projected(G, y, covs)
    np.testing.assert_allclose(tab["LD_REL"].to_numpy(), _brute_ld_rel(X, pi, pj, n_eff, np.ones((G.shape[1], 1)))[:, 0],
                               rtol=1e-8, atol=1e-10)
    keep = ph["y"].notna().to_numpy()
    pos = -np.ones(keep.size, dtype=int)
    pos[keep] = np.arange(keep.sum())
    use = keep[pi] & keep[pj]
    a, b = pos[pi[use]], pos[pj[use]]
    Xs, ys, p, ne = _projected(G[keep], y[keep], covs[keep])
    n = int(keep.sum())
    kappa = (n - 1.0) * (n - p - 2.0) / ((n - 2.0) * ne ** 2)
    brute = kappa * 2.0 * ((Xs[a] * Xs[b]) * (ys[a] * ys[b])[:, None]).sum(0)
    np.testing.assert_allclose(tab["REL_M_y"].to_numpy(), brute, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("ncov", [0, 2])
def test_adjust_ld_gives_pair_excluded_trace(tmp_path, ncov):
    """With in-sample LD scores l_j = sum_j' r_jj'^2 - M/N_eff (SUMMIT's convention), the adjusted scores
    reconstruct the pair-excluded trace exactly: N_eff + N_eff^2/M^2 sum_j (l_j - LD_REL_j) = tr(K_u^2)."""
    prefix, G, y, covs, iid, cov_path = _simulate(tmp_path, ncov=ncov, missing=0.0)
    out = str(tmp_path / "rel")
    compute_relative_pair_moments(prefix, str(tmp_path / "pheno.txt"), out, pairs=str(tmp_path / "pairs.txt"),
                                  covar=cov_path, ld_rel=True)
    X, _, p, n_eff = _projected(G, y, covs if ncov else None)
    N, M = X.shape
    ld_exact = ((X.T @ X / n_eff) ** 2).sum(1) - M / n_eff
    pd.DataFrame({**_snp_meta(M), "L2_0": ld_exact})[["CHR", "SNP", "BP", "L2_0"]].to_csv(
        tmp_path / "in.gw.ldscore.gz", sep="\t", index=False, float_format="%.6f")
    info = adjust_ldscores(str(tmp_path / "in.gw.ldscore.gz"), out + ".tsv.gz", str(tmp_path / "adj.gw.ldscore.gz"))
    adj = pd.read_csv(tmp_path / "adj.gw.ldscore.gz", sep="\t")
    pi, pj = _family_pairs(str(tmp_path / "pairs.txt"), iid)
    K = X @ X.T / M
    full = n_eff + n_eff ** 2 / M ** 2 * info["columns"]["L2_0"]["sum_ld"]
    assert full == pytest.approx((K * K).sum(), rel=1e-6)
    K[pi, pj] = 0.0
    K[pj, pi] = 0.0
    assert n_eff + n_eff ** 2 / M ** 2 * adj["L2_0"].sum() == pytest.approx((K * K).sum(), rel=1e-6)
    with pytest.raises(ValueError, match="LD_REL_L2_1"):
        pd.DataFrame({**_snp_meta(M), "L2_0": ld_exact, "L2_1": ld_exact})[["CHR", "SNP", "BP", "L2_0", "L2_1"]].to_csv(
            tmp_path / "two.gw.ldscore.gz", sep="\t", index=False)
        adjust_ldscores(str(tmp_path / "two.gw.ldscore.gz"), out + ".tsv.gz", str(tmp_path / "x.gz"))


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
    kappa = (N - 1.0) * (N - p - 2.0) / ((N - 2.0) * n_eff ** 2)
    brute = kappa * ((X[pi] * X[pj]) * (y1t[pi] * y2t[pj] + y1t[pj] * y2t[pi])[:, None]).sum(0)
    np.testing.assert_allclose(tab["REL_M12_y__y2"].to_numpy(), brute, rtol=1e-8, atol=1e-12)
    # the self pair would equal REL_M: check REL_M from the same run against its own formula
    np.testing.assert_allclose(tab["REL_M_y"].to_numpy(),
                               kappa * 2.0 * ((X[pi] * X[pj]) * (y1t[pi] * y1t[pj])[:, None]).sum(0), rtol=1e-8, atol=1e-12)
    if ncov == 0:                                         # exact: (1/M) sum_j (z1* z2* - REL_M12_j) = y1'K_u y2 / N_eff
        r1, r2 = (X.T @ y1t) / n_eff, (X.T @ y2t) / n_eff
        z1 = exact_score_z_from_arrays(r1, np.sqrt((1 - r1 ** 2) / (N - 2)), np.full(M, N), N, 0)
        z2 = exact_score_z_from_arrays(r2, np.sqrt((1 - r2 ** 2) / (N - 2)), np.full(M, N), N, 0)
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
                                  covar=cov_path, cross=True, ld_rel=True)
    X, yy, p, n_eff = _projected(G, y, covs)
    N, M = X.shape
    _ols_sumstats(tmp_path / "ss.tsv", X, yy, p)
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), out + ".tsv.gz", "y", str(tmp_path / "ssa.tsv"))
    append_rel_m_to_sumstats(str(tmp_path / "ss.tsv"), out + ".tsv.gz", "yb", str(tmp_path / "ssb.tsv"))
    extract_rel_m12(out + ".tsv.gz", "y", "yb", str(tmp_path / "m12.tsv"))
    ld_exact = ((X.T @ X / n_eff) ** 2).sum(1) - M / n_eff
    pd.DataFrame({**_snp_meta(M), "L2_0": ld_exact})[["CHR", "SNP", "BP", "L2_0"]].to_csv(
        tmp_path / "in.gw.ldscore.gz", sep="\t", index=False, float_format="%.6f")
    adjust_ldscores(str(tmp_path / "in.gw.ldscore.gz"), out + ".tsv.gz", str(tmp_path / "adj.gw.ldscore.gz"))
    common = dict(ldscores=str(tmp_path / "adj.gw.ldscore.gz"), njack="chr", chisq_threshold=None,
                  intercept_chisq_thr=None, intercept_rg=1.0, align_alleles=False)
    res = Sumcore(rg=f"{tmp_path / 'ssa.tsv'},{tmp_path / 'ssb.tsv'}", rel_m12=str(tmp_path / "m12.tsv"), **common)._run()
    gamma = float(res["rg_fit"].gamma_total[0])
    assert gamma == pytest.approx(float(res["h2_fit1"].h2[-1, 0]), rel=1e-8)
    assert float(res["h2_fit2"].h2[-1, 0]) == pytest.approx(float(res["h2_fit1"].h2[-1, 0]), rel=1e-12)
    # the pair-excluded h2 differs from the naive one, so the check is not vacuous
    naive = Sumcore(rg=f"{tmp_path / 'ss.tsv'},{tmp_path / 'ss.tsv'}",
                    **{**common, "ldscores": str(tmp_path / "in.gw.ldscore.gz")})._run()
    assert abs(float(naive["rg_fit"].gamma_total[0]) - gamma) > 1e-4
    with pytest.raises(ValueError, match="--rel-m12"):
        Sumcore(rg=f"{tmp_path / 'ssa.tsv'},{tmp_path / 'ssb.tsv'}", **common)._run()
