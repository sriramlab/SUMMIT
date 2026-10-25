"""SNP-variance weights (V column) and related-pair LD_REL columns: LD-score step, Trace reader, h2/rg fits."""
import numpy as np
import pandas as pd
import pytest

bed_reader = pytest.importorskip("bed_reader")

from summit.inference.sumcore import Sumcore
from summit.inference.sumrhe import Sumrhe
from summit.inference.trace import Trace
from summit.ldscore.gw_ldscore import GenomewideLDScore
from summit.relatives.moments import read_relative_pairs


class _Log:
    def __init__(self):
        self.messages = []

    def _log(self, message):
        self.messages.append(str(message))

    def _save_log(self, path):
        self.saved_path = str(path)


def _cohort(tmp_path, *, N=360, M=160, seed=1, missing=0.01, nrel=0):
    """Two populations with different allele frequencies; covariates = population score + noise and a second one."""
    rng = np.random.default_rng(seed)
    pop = (rng.random(N) < 0.5).astype(float)
    f0 = rng.uniform(0.08, 0.5, M)
    f1 = np.clip(f0 + rng.normal(0, 0.15, M), 0.03, 0.97)
    G = rng.binomial(2, np.where(pop[:, None] == 1, f1[None, :], f0[None, :])).astype(np.float64)
    for k in range(nrel):                                  # relatives: 2k+1 shares half of 2k's genotypes
        G[2 * k + 1, ::2] = G[2 * k, ::2]
    G[:, 7] = 0.0                                          # monomorphic
    G[rng.random(G.shape) < missing] = np.nan
    covs = np.column_stack([pop + rng.normal(0, 0.3, N), rng.normal(size=N)])
    iid = np.array([f"s{i}" for i in range(N)])
    prefix = str(tmp_path / "geno")
    bed_reader.to_bed(prefix + ".bed", G, properties={
        "fid": iid, "iid": iid, "sid": [f"rs{j}" for j in range(M)],
        "chromosome": np.repeat(["1", "2"], [M // 2, M - M // 2]), "bp_position": np.arange(1, M + 1) * 1000,
        "allele_1": np.full(M, "A"), "allele_2": np.full(M, "G")})
    cdf = pd.DataFrame(covs, columns=["c0", "c1"])
    cdf.insert(0, "IID", iid)
    cdf.insert(0, "FID", iid)
    cdf.to_csv(tmp_path / "covar.txt", sep=" ", index=False)
    if nrel:
        pd.DataFrame({"IID1": [f"s{2 * k}" for k in range(nrel)],
                      "IID2": [f"s{2 * k + 1}" for k in range(nrel)]}).to_csv(tmp_path / "pairs.txt", sep=" ", index=False)
    return prefix, G, covs, iid


def _standardized_projected(G, covs):
    """Mean-imputed centered genotypes, their projection on [1, covs], V = 1 - adjusted R2 and re-standardized X~."""
    N = G.shape[0]
    p = covs.shape[1]
    X = np.where(np.isnan(G), np.nanmean(G, 0)[None, :], G) - np.nanmean(G, 0)[None, :]
    Q, _ = np.linalg.qr(np.column_stack([np.ones(N), covs]))
    PX = X - Q @ (Q.T @ X)
    tot, res = (X ** 2).sum(0), (PX ** 2).sum(0)
    ok = tot > 1e-12
    v = np.zeros(G.shape[1])
    v[ok] = res[ok] / tot[ok] * (N - 1) / (N - p - 1)
    ne = N - p - 1
    Xt = np.zeros_like(PX)
    Xt[:, ok] = PX[:, ok] * np.sqrt(ne / res[ok])[None, :]
    return X, PX, v, Xt, ne, Q


def _run_ld(prefix, out, covar, *, annot=None, weights=True, pairs=None, nvecs=48, seed=7, fmt="bed"):
    est = GenomewideLDScore(
        bed_path=f"{prefix}.{fmt}", annot_path=annot, out_path=str(out), log=_Log(), rand_dist="rademacher",
        low_level={"numa_mode": None}, covar_path=covar, num_vecs=nvecs, step_size=37, seed=seed, dtype="float64",
        num_threads=2, ddof=1, target_xz_mem=0.05, use_mailman=False, impute_method="mean",
        snp_var_weights=weights, relative_pairs=pairs,
    )
    try:
        est._compute_ldscore()
        return est
    finally:
        est.close()


def _meta(M):
    return pd.DataFrame({"CHR": np.repeat([1, 2], [M // 2, M - M // 2]), "SNP": [f"rs{j}" for j in range(M)],
                         "BP": np.arange(1, M + 1) * 1000})


def test_v_column_is_one_minus_adjusted_r2_and_weights_partner_snps(tmp_path):
    prefix, G, covs, iid = _cohort(tmp_path)
    covar = str(tmp_path / "covar.txt")
    est = _run_ld(prefix, tmp_path / "v", covar)
    df = pd.read_csv(tmp_path / "v.gw.ldscore.gz", sep="\t")
    assert df.columns.tolist() == ["CHR", "SNP", "BP", "V", "L2_0"]
    _, _, v, _, _, _ = _standardized_projected(G, covs)
    # the native residual-variance pass works in single precision
    np.testing.assert_allclose(est.snp_var, v, rtol=1e-6, atol=1e-9)
    assert est.snp_var[7] == 0.0
    np.testing.assert_allclose(df["V"].to_numpy(), v, atol=1e-6)
    assert np.loadtxt(tmp_path / "v.gw.M") == pytest.approx(G.shape[1])

    # the same LD scores as an ordinary run with V as a continuous annotation and the same probes
    ann = _meta(G.shape[1])
    ann["VW"] = est.snp_var
    ann.to_csv(tmp_path / "annV.txt", sep=" ", index=False, float_format="%.17g")
    proto = _run_ld(prefix, tmp_path / "proto", covar, annot=str(tmp_path / "annV.txt"), weights=False)
    np.testing.assert_allclose(est.gwldscore, proto.gwldscore, rtol=1e-6, atol=1e-6)

    plain = _run_ld(prefix, tmp_path / "plain", covar, weights=False)
    assert pd.read_csv(tmp_path / "plain.gw.ldscore.gz", sep="\t").columns.tolist() == ["CHR", "SNP", "BP", "L2_0"]
    assert plain.snp_var is None
    with pytest.raises(ValueError, match="snp-var-weights-off"):
        GenomewideLDScore(
            bed_path=f"{prefix}.bed", annot_path=None, out_path=str(tmp_path / "skew"), log=_Log(),
            rand_dist="rademacher", low_level={"numa_mode": None}, covar_path=covar, num_vecs=4, step_size=37,
            seed=1, dtype="float64", num_threads=1, correct_skew=True)


def test_pgen_v_and_ld_scores_match_bed(tmp_path):
    pgenlib = pytest.importorskip("pgenlib")
    prefix, G, covs, iid = _cohort(tmp_path, missing=0.0)
    N, M = G.shape
    codes = np.ascontiguousarray((2 - G).T.astype(np.int8))           # PGEN dosages count REF = BED allele 2
    with pgenlib.PgenWriter(bytes(f"{prefix}.pgen", "utf8"), sample_ct=N, variant_ct=M, nonref_flags=False) as w:
        w.append_biallelic_batch(codes)
    with open(f"{prefix}.psam", "w") as fh:
        fh.write("#FID\tIID\n" + "".join(f"{i}\t{i}\n" for i in iid))
    meta = _meta(M)
    with open(f"{prefix}.pvar", "w") as fh:
        fh.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for c, s, b in zip(meta.CHR, meta.SNP, meta.BP):
            fh.write(f"{c}\t{b}\t{s}\tG\tA\n")
    covar = str(tmp_path / "covar.txt")
    bed = _run_ld(prefix, tmp_path / "b", covar)
    pg = _run_ld(prefix, tmp_path / "p", covar, fmt="pgen")
    np.testing.assert_allclose(pg.snp_var, bed.snp_var, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(pg.gwldscore, bed.gwldscore, rtol=1e-6, atol=1e-6)


def test_ld_rel_columns_match_brute_force(tmp_path):
    prefix, G, covs, iid = _cohort(tmp_path, missing=0.0, nrel=30)
    M = G.shape[1]
    b = (np.random.default_rng(5).random(M) < 0.4).astype(int)
    ann = _meta(M)
    ann["binA"], ann["binB"] = b, 1 - b
    ann.to_csv(tmp_path / "annot.txt", sep=" ", index=False)
    est = _run_ld(prefix, tmp_path / "rel", str(tmp_path / "covar.txt"), annot=str(tmp_path / "annot.txt"),
                  pairs=str(tmp_path / "pairs.txt"))
    df = pd.read_csv(tmp_path / "rel.gw.ldscore.gz", sep="\t")
    assert df.columns.tolist() == ["CHR", "SNP", "BP", "V", "binA", "binB", "LD_REL_binA", "LD_REL_binB"]
    _, _, v, Xt, ne, _ = _standardized_projected(G, covs)
    rp = read_relative_pairs(str(tmp_path / "pairs.txt"), pd.DataFrame({"FID": iid, "IID": iid}))
    A = np.column_stack([b, 1 - b]) * v[:, None]
    Z = Xt[rp.i] * Xt[rp.j]
    brute = 2.0 * (Z.T @ (Z @ A)) / ne ** 2
    np.testing.assert_allclose(est.ld_rel, brute, rtol=1e-8, atol=1e-10)


def _write_ld(path, meta, cols):
    df = meta.copy()
    for name, values in cols.items():
        df[name] = values
    df.to_csv(path, sep="\t", index=False, float_format="%.12g")
    return str(path)


def test_trace_reads_v_and_ld_rel(tmp_path):
    M = 40
    meta = _meta(M)
    rng = np.random.default_rng(0)
    v, ld, rel = rng.uniform(0.9, 1.0, M), rng.uniform(1, 5, M), rng.uniform(0, 0.1, M)
    path = _write_ld(tmp_path / "a.gw.ldscore.gz", meta, {"V": v, "L2_0": ld, "LD_REL_L2_0": rel})
    tr = Trace(ldscores=path)
    np.testing.assert_allclose(tr.annot[:, 0], v)
    np.testing.assert_allclose(tr.annot_mass[:, 0], 1.0)
    np.testing.assert_allclose(tr.ldscores[:, 0], ld)
    np.testing.assert_allclose(tr.ldscores_rel[:, 0], rel)
    tv = tr.materialize_view(np.arange(M) % 3 > 0)
    np.testing.assert_allclose(tv.pair_excluded().ldscores[:, 0], (ld - rel)[np.arange(M) % 3 > 0])
    with pytest.raises(ValueError, match="snp-var-weights-off"):
        Trace(ldscores=path, snp_var_weights=False)
    plain = Trace(ldscores=_write_ld(tmp_path / "b.gw.ldscore.gz", meta, {"L2_0": ld}))
    assert plain.annot_mass is None and plain.ldscores_rel is None
    with pytest.raises(ValueError, match="directly follow"):
        Trace(ldscores=_write_ld(tmp_path / "c.gw.ldscore.gz", meta, {"L2_0": ld, "V": v}))
    with pytest.raises(ValueError, match="do not match"):
        Trace(ldscores=_write_ld(tmp_path / "d.gw.ldscore.gz", meta, {"L2_0": ld, "LD_REL_x": rel}))
    with pytest.raises(ValueError, match="ordinary LD-score files"):
        Trace(ldscores=_write_ld(tmp_path / "e.gw.ldscore.gz", meta, {"L2_0": ld}), ldscores_reg=path)


def _exact_setting(tmp_path, *, N=500, M=240, seed=4):
    """Exact (deterministic) in-sample LD scores, ordinary and V-weighted, for a structured cohort without missing
    genotypes, and the exact HE kernels they correspond to."""
    _, G, covs, iid = _cohort(tmp_path, N=N, M=M, seed=seed, missing=0.0)
    X, PX, v, Xt, ne, Q = _standardized_projected(G, covs)
    sd = X.std(0, ddof=1)
    ok = sd > 0
    Xs = np.zeros_like(X)
    Xs[:, ok] = X[:, ok] / sd[ok]
    PXs = Xs - Q @ (Q.T @ Xs)
    meta = _meta(M)
    R2 = (Xt.T @ Xt / ne) ** 2
    ld_v = R2 @ v - v.sum() / ne
    ld_1 = R2 @ ok.astype(float) - ok.sum() / ne
    paths = {
        "ord": _write_ld(tmp_path / "ord.gw.ldscore.gz", meta, {"L2_0": ld_1}),
        "v": _write_ld(tmp_path / "v.gw.ldscore.gz", meta, {"V": v, "L2_0": ld_v}),
        "proto": _write_ld(tmp_path / "proto.gw.ldscore.gz", meta, {"L2_0": ld_v}),
    }
    ann = meta.copy()
    ann["VW"] = v
    ann.to_csv(tmp_path / "annV.txt", sep=" ", index=False, float_format="%.17g")
    P = np.eye(N) - Q @ Q.T
    kernels = {"ord": Xt @ Xt.T / ok.sum(), "v": PXs @ PXs.T / ok.sum()}
    return G, covs, Xs, PXs, Q, P, v, ok, paths, kernels


def _sumstats(path, PXs, yp, N, p):
    """PLINK-style beta/SE with the covariate rank, for the polymorphic SNPs (monomorphic SNPs have no GWAS row)."""
    ss = (PXs ** 2).sum(0)
    ok = ss > 1e-10
    r = (PXs[:, ok].T @ yp) / np.sqrt(ss[ok] * (yp @ yp))
    pd.DataFrame({"SNP": [f"rs{j}" for j in np.flatnonzero(ok)], "A1": "A", "A2": "G", "N": N, "BETA": r,
                  "SE": np.sqrt((1 - r ** 2) / (N - p - 2)), "COV_RANK": p}).to_csv(
        path, sep="\t", index=False, float_format="%.15g")
    return str(path)


def _exact_he(K, P, ya, yb):
    L = np.array([[(K * K).sum(), np.trace(K @ P)], [np.trace(K), np.trace(P)]])
    return np.linalg.solve(L, [ya @ K @ yb, ya @ P @ yb])


def test_v_mode_h2_and_rg_match_exact_he_with_the_unadjusted_kernel(tmp_path):
    G, covs, Xs, PXs, Q, P, v, ok, paths, kernels = _exact_setting(tmp_path)
    N, M = G.shape
    p = covs.shape[1]
    rng = np.random.default_rng(9)
    pop = covs[:, 0]
    for rep in range(3):
        g1 = Xs @ rng.normal(0, np.sqrt(0.5 / M), M)
        g2 = 0.7 * g1 + Xs @ rng.normal(0, np.sqrt(0.2 / M), M)
        y1 = g1 + 0.8 * pop + rng.normal(0, np.sqrt(0.5), N)
        y2 = g2 + 0.3 * pop + 0.3 * (y1 - g1) + rng.normal(0, 0.7, N)
        y1p, y2p = y1 - Q @ (Q.T @ y1), y2 - Q @ (Q.T @ y2)
        s1 = _sumstats(tmp_path / f"a{rep}.tsv", PXs, y1p, N, p)
        s2 = _sumstats(tmp_path / f"b{rep}.tsv", PXs, y2p, N, p)
        fit = {}
        for kind in ("ord", "v"):
            h = Sumrhe(h2_path=s1, ldscores=paths[kind], njack="chr", chisq_threshold=None)
            h._run()
            fit[kind] = float(h.results[0].h2[-1, 0])
            assert np.isfinite(h.results[0].h2[-1, 1])
        e = {k: _exact_he(K, P, y1p, y1p) for k, K in kernels.items()}
        exact = {k: s[0] / s.sum() for k, s in e.items()}
        # with the covariate rank, both modes equal exact HE with their kernels
        assert fit["ord"] == pytest.approx(exact["ord"], rel=1e-8)
        assert fit["v"] == pytest.approx(exact["v"], rel=1e-8)

        # plumbing: V mode = the ordinary fit of the V-weighted kernel, converted to the unadjusted scale
        h = Sumrhe(h2_path=s1, ldscores=paths["proto"], annot=str(tmp_path / "annV.txt"), njack="chr",
                   chisq_threshold=None)
        h._run()
        sv = float(h.results[0].h2[-1, 0])
        tm = sv / v.sum() * ok.sum()
        assert fit["v"] == pytest.approx(tm / (tm + 1.0 - sv), rel=1e-10)

        c_ov = float(y1p @ y2p / np.sqrt((y1p @ y1p) * (y2p @ y2p)))
        for kind in ("ord", "v"):
            K = kernels[kind]
            g = _exact_he(K, P, y1p, y2p)[0]
            rg_exact = g / np.sqrt(_exact_he(K, P, y1p, y1p)[0] * _exact_he(K, P, y2p, y2p)[0])
            res = Sumcore(rg=f"{s1},{s2}", ldscores=paths[kind], njack="chr", chisq_threshold=None,
                          intercept_chisq_thr=None, intercept_rg=c_ov, align_alleles=False)._run()
            assert float(res["rg_fit"].rg_total[0]) == pytest.approx(rg_exact, abs=1e-4)


def test_v_mode_refuses_unsupported_options(tmp_path):
    _, _, _, _, _, _, _, _, paths, _ = _exact_setting(tmp_path, N=200, M=60)
    meta = _meta(60)
    ss = tmp_path / "ss.tsv"
    pd.DataFrame({"SNP": meta.SNP, "A1": "A", "A2": "G", "N": 200, "BETA": 0.01, "SE": 0.07}).to_csv(
        ss, sep="\t", index=False)
    with pytest.raises(ValueError, match="snp-var-weights-off"):
        Sumrhe(h2_path=str(ss), ldscores=paths["v"], weight_mode="ldsc", njack="chr")
    with pytest.raises(ValueError, match="snp-var-weights-off"):
        Sumcore(rg=f"{ss},{ss}", ldscores=paths["v"], njack="chr", rg_se_method="robust", intercept_rg=1.0)


def test_ld_rel_columns_are_subtracted_only_for_sumstats_with_rel_m(tmp_path):
    M = 120
    meta = _meta(M)
    rng = np.random.default_rng(2)
    ld = rng.uniform(5, 30, M)
    rel = rng.uniform(0.0, 0.5, M)
    with_rel = _write_ld(tmp_path / "w.gw.ldscore.gz", meta, {"L2_0": ld, "LD_REL_L2_0": rel})
    adjusted = _write_ld(tmp_path / "adj.gw.ldscore.gz", meta, {"L2_0": ld - rel})
    plain = _write_ld(tmp_path / "p.gw.ldscore.gz", meta, {"L2_0": ld})
    base = pd.DataFrame({"SNP": meta.SNP, "A1": "A", "A2": "G", "N": 5000,
                         "BETA": rng.normal(0, 0.02, M), "SE": 0.014})
    base.to_csv(tmp_path / "s.tsv", sep="\t", index=False)
    base.assign(REL_M=rng.normal(0.02, 0.01, M)).to_csv(tmp_path / "sr.tsv", sep="\t", index=False)

    def h2(ss_path, ld_path, njack):
        s = Sumrhe(h2_path=str(ss_path), ldscores=ld_path, njack=njack, chisq_threshold=None, log=_Log())
        s._run()
        return s.results[0].h2[-1]

    for njack in ("chr", "10"):
        np.testing.assert_allclose(h2(tmp_path / "sr.tsv", with_rel, njack), h2(tmp_path / "sr.tsv", adjusted, njack),
                                   rtol=1e-9)
        np.testing.assert_allclose(h2(tmp_path / "s.tsv", with_rel, njack), h2(tmp_path / "s.tsv", plain, njack),
                                   rtol=1e-12)
