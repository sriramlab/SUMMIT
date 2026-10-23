import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.relatives.moments import compute_relative_pair_moments, read_relative_pairs
from summit.sumstats.moments import build_h2_summary_moment, h2_moment_from_arrays
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
