"""Tests of cross-population LD scores from `summit admixed-ld compose --cross-source`: the composed cross scores of an
admixed cohort with a European GWAS on other people against the exact ones, and cross-population rg with them. In a
module of their own, so that their larger simulated genome is released before the other tests run (the cross-trait
GxE tests plan their memory from the pytest process's peak)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from admixed_ld_sim import gwas, in_child, simulate_int8, write_bed  # noqa: E402

from summit.admixed_ld.reference import build_reference  # noqa: E402


@pytest.fixture(scope="module")
def simx(tmp_path_factory):
    """A denser genome for the cross-population scores (40 SNPs per cM on two 60-cM chromosomes, founder copying at 1
    per cM), so that partners carry a fifth of each cross score: the admixed cohort with its PCs and GWAS, the African
    panel, and 3,000 Europeans split into a reference panel (600), an independent panel (300) and a disjoint European
    GWAS sample (2,100). Dosages are kept as int8 and the larger products are made chromosome by chromosome, so that
    the pytest process's peak memory stays low (the cross-trait GxE tests plan their memory from it); the simulation
    itself runs in a child process."""
    d = tmp_path_factory.mktemp("admixed_ld_cross")
    S = in_child(simulate_int8, seed=7, chr_cm=60.0, snps_per_chr=2400, copy_rate=1.0, n_panel=3000, n_cohort=2500)
    v, G, GA, GE = S["variants"], S["cohort"], S["panel_A"], S["panel_E"]
    iid = write_bed(str(d / "cohort"), G, v, "c")
    std0 = lambda X: (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    K = sum((lambda X: X @ X.T)(std0(G[:, j])) for j in (np.flatnonzero(v.CHR.to_numpy() == c)
                                                       for c in v.CHR.unique()))
    w, U = np.linalg.eigh(K)                             # the PCs from the people-by-people product, not an SVD
    C = U[:, ::-1][:, :2] * np.sqrt(w[::-1][:2])
    del K, U
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(d / "cohort.cov", sep="\t",
                                                                                   index=False)
    rng = np.random.default_rng(5)
    os.makedirs(d / "ss")
    for t in range(2):
        y = rng.standard_normal(G.shape[0])
        beta, se = _gwas_by_chromosome(G, y, C, v.CHR.to_numpy())
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta,
                      "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(d / "ss" / f"trait{t}.tsv", sep="\t",
                                                                       index=False)
    p = G.mean(0) / 2
    cls = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"SNP": v.SNP, "low": (cls == 0).astype(int), "high": (cls == 1).astype(int)}).to_csv(
        d / "annot.tsv", sep="\t", index=False)
    nE = GE.shape[0]
    E = dict(ref=GE[:int(0.2 * nE)], ind=GE[int(0.2 * nE):int(0.3 * nE)], two=GE[int(0.3 * nE):])
    write_bed(str(d / "pA"), GA, v, "a")
    write_bed(str(d / "pE"), E["ref"], v, "e")
    write_bed(str(d / "pE2"), E["ind"], v, "f")
    build_reference([str(d / "pA"), str(d / "pE"), str(d / "pE2")], ["AFR", "EUR", "EUR2"], str(d / "ref"),
                    window_kb=3000, step=256, log=lambda *a: None)
    from summit.admixed_ld.curve import genotype_curve
    genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "cv"), covar=str(d / "cohort.cov"), cell_cm=0.5,
                   log=lambda *a: None)
    return dict(dir=d, variants=v, G=G, C=C, cls=cls, E=E)


def _gwas_by_chromosome(G, y, C, chrom):
    """The marginal GWAS of each SNP, one chromosome at a time (each SNP's regression is its own)."""
    beta, se = np.empty(G.shape[1]), np.empty(G.shape[1])
    for c in np.unique(chrom):
        j = np.flatnonzero(chrom == c)
        beta[j], se[j] = gwas(G[:, j].astype(np.float64), y, C)
    return beta, se


def _std(G):
    return (G - G.mean(0)) / G.std(0, ddof=1)


def _scaled(G, C):
    """Standardized genotypes adjusted for the covariates C (an intercept is implied) and scaled to unit residual
    variance, so that U'U / nu is the adjusted correlation; also the partner weights V and nu."""
    X, n = _std(G), G.shape[0]
    Q, _ = np.linalg.qr((C - C.mean(0)) / C.std(0, ddof=1)) if C.shape[1] else (np.zeros((n, 0)), None)
    Xr = X - Q @ (Q.T @ X)
    ss, nu = np.einsum("ij,ij->j", Xr, Xr), n - Q.shape[1] - 1
    return Xr / np.sqrt(ss / nu), ss / np.einsum("ij,ij->j", X, X) * (n - 1) / nu, nu


def test_cross_population_ldscores_match_the_exact_cross(simx, tmp_path):
    """Cross-population LD scores of the admixed cohort with a European GWAS on other people: the composed
    sum_k rho_1 rho_2 sqrt(V_1k V_2k) against the exact one from the two samples' genotypes (the cohort's
    covariate-adjusted correlations, the European sample's correlations). With the European panel also one of the
    cohort's sources (its square unbiased) and with an independent European panel: the partners' part of the score
    within 3% of the exact one in mass, SNP by SNP as close as the panel's own noise allows, and the two panels
    agreeing; the own scores unchanged by asking for the cross scores; nothing beyond the window on average."""
    from summit.admixed_ld.compose import compose_ldscores
    d, v, E = simx["dir"], simx["variants"], simx["E"]
    got, par = {}, {}
    for arm, xs in (("shared", "EUR"), ("independent", "EUR2")):
        o = str(tmp_path / f"x_{arm}")
        par[arm] = compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), o, sources=["AFR", "EUR"],
                                    curve=str(d / "cv"), cross_source=xs, log=lambda *a: None)
        got[arm] = pd.read_csv(f"{o}.cross_{xs}.gw.ldscore.gz", sep="\t")
        assert list(got[arm].columns) == ["CHR", "SNP", "BP", "V", "low", "high"]
        assert par[arm]["cross"]["panel_is_a_source"] == (arm == "shared")
    o = str(tmp_path / "own_only")
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), o, sources=["AFR", "EUR"],
                     curve=str(d / "cv"), log=lambda *a: None)
    own = pd.read_csv(f"{o}.gw.ldscore.gz", sep="\t")
    pd.testing.assert_frame_equal(own, pd.read_csv(str(tmp_path / "x_shared") + ".gw.ldscore.gz", sep="\t"))
    # the exact cross score: the cohort adjusted for its PCs, the European sample without covariates (V_2 = 1); the
    # window lies within a chromosome, so each chromosome is its own block
    snps = got["shared"].SNP
    keep = np.flatnonzero(v.SNP.isin(snps).to_numpy())
    ids, chrom, bp, cls = v.SNP.to_numpy()[keep], v.CHR.to_numpy()[keep], v.BP.to_numpy()[keep], simx["cls"][keep]
    M = len(keep)
    w, P_ex, beyond = np.empty(M), np.empty((M, 2)), np.empty(M)
    P_panel = {arm: np.empty((M, 2)) for arm in ("shared", "independent")}
    for j in (np.flatnonzero(chrom == c) for c in np.unique(chrom)):
        U1, V1, nu = _scaled(simx["G"][:, keep[j]], simx["C"])
        R1, wj, cj = U1.T @ U1 / nu, np.sqrt(V1), cls[j]
        del U1
        win = np.abs(bp[j][:, None] - bp[j][None, :]) <= 3_000_000
        np.fill_diagonal(win, False)
        part = lambda R: np.stack([(np.where(win, R1 * R, 0.0) * wj[None, :])[:, cj == k].sum(1)
                                   for k in range(2)], 1)
        corr2 = lambda G: (lambda Y: Y.T @ Y / (G.shape[0] - 1))(_std(G))
        R2 = corr2(E["two"][:, keep[j]])
        w[j], P_ex[j] = wj, part(R2)                                 # partners' part of the exact score
        beyond_win = ~win
        np.fill_diagonal(beyond_win, False)
        beyond[j] = (np.where(beyond_win, R1 * R2, 0.0) * wj[None, :]).sum(1)
        for arm, panel in (("shared", E["ref"]), ("independent", E["ind"])):
            P_panel[arm][j] = part(corr2(panel[:, keep[j]]))      # the cohort's own correlations with this panel
        del R1, R2
    assert abs(float((w * beyond).sum())) < 0.02 * float((w[:, None] * P_ex).sum())
    for arm in ("shared", "independent"):
        g = got[arm].set_index("SNP").reindex(ids)
        np.testing.assert_allclose(g.V.to_numpy(), np.sqrt(own.set_index("SNP").reindex(ids).V.to_numpy()), rtol=1e-5)
        P_c = g[["low", "high"]].to_numpy()
        P_c[np.arange(len(cls)), cls] -= g.V.to_numpy()             # the partners' part: less each SNP with itself
        r = float((w[:, None] * P_c).sum() / (w[:, None] * P_ex).sum())
        assert abs(r - 1) < 0.03, (arm, r)
        for c in range(2):
            assert abs(P_c[:, c].sum() / P_ex[:, c].sum() - 1) < 0.06, (arm, c)
        # SNP by SNP, the panel's sampling noise bounds the agreement: the cohort's own correlations with this panel
        best = np.corrcoef(P_panel[arm].sum(1), P_ex.sum(1))[0, 1]
        rc = np.corrcoef(P_c.sum(1), P_ex.sum(1))[0, 1]
        assert rc > 0.9 * best, (arm, rc, best)
        assert par[arm]["cross"]["ld_mass"] == pytest.approx(float((g.V.to_numpy()[:, None] *
                                                                    g[["low", "high"]].to_numpy()).sum()), rel=1e-4)
    a = got["shared"].set_index("SNP")[["low", "high"]].to_numpy().sum()
    b = got["independent"].set_index("SNP").reindex(got["shared"].SNP)[["low", "high"]].to_numpy().sum()
    assert abs(a / b - 1) < 0.02


def _ld_table(path, snps, chrom, bp, L, V, names=("low", "high")):
    pd.DataFrame({"CHR": chrom, "SNP": snps, "BP": bp, "V": V, names[0]: L[:, 0], names[1]: L[:, 1]}).to_csv(
        f"{path}.gw.ldscore.gz", sep="\t", index=False, float_format="%.8f")
    return f"{path}.gw.ldscore.gz"


def test_cross_population_rg(simx, tmp_path):
    """Cross-population rg (summit --rg with --ldscores-h2-1/-2): traits with a cross-population impact correlation of
    0.6 in the admixed cohort and in the disjoint European sample. Each trait's h2 in cross mode equals its own h2 fit
    with its own LD scores; with the cross LD scores in place of both own ones it is the ordinary rg; the composed
    cross and own scores give the rg and genetic covariance of the exact ones (same GWAS) and both cover 0.6."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.inference.sumcore import Sumcore
    d, v, E = simx["dir"], simx["variants"], simx["E"]
    o = str(tmp_path / "c")
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), o, sources=["AFR", "EUR"],
                     curve=str(d / "cv"), cross_source="EUR", log=lambda *a: None)
    cx = pd.read_csv(f"{o}.cross_EUR.gw.ldscore.gz", sep="\t").set_index("SNP")
    c1 = pd.read_csv(f"{o}.gw.ldscore.gz", sep="\t").set_index("SNP").reindex(cx.index)
    snps = cx.index.to_numpy()
    keep = v.SNP.isin(snps).to_numpy()
    assert (v.SNP[keep].to_numpy() == snps).all()
    chrom, bp, cls = v.CHR.to_numpy()[keep], v.BP.to_numpy()[keep], simx["cls"][keep]
    A = np.zeros((keep.sum(), 2)); A[np.arange(keep.sum()), cls] = 1
    # traits: per-standardized-genotype effects correlated 0.6 across the populations, h2 0.5 in each
    rng = np.random.default_rng(11)
    M = keep.sum()
    b = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=M) * np.sqrt(0.5 / M)
    G1, G2 = simx["G"][:, keep], E["two"][:, keep]
    blocks = [np.flatnonzero(chrom == c) for c in np.unique(chrom)]   # products one chromosome (pair) at a time
    y1 = sum(_std(G1[:, j]) @ b[j, 0] for j in blocks); y1 = y1 + rng.standard_normal(len(y1)) * np.sqrt(1 - y1.var())
    y2 = sum(_std(G2[:, j]) @ b[j, 1] for j in blocks); y2 = y2 + rng.standard_normal(len(y2)) * np.sqrt(1 - y2.var())
    C = simx["C"]
    for name, G, y, Cv, rank in (("t1", G1, y1, C, 2), ("t2", G2, y2, np.zeros((G2.shape[0], 0)), 0)):
        beta, se = _gwas_by_chromosome(G, y, Cv, chrom)
        pd.DataFrame({"SNP": snps, "CHR": chrom, "BP": bp, "A1": v.A1.to_numpy()[keep], "A2": v.A2.to_numpy()[keep],
                      "N": len(y), "BETA": beta, "SE": se, "Z": beta / se, "COV_RANK": rank}).to_csv(
            tmp_path / f"{name}.tsv", sep="\t", index=False)
    pd.DataFrame({"CHR": chrom, "BP": bp, "SNP": snps, "low": A[:, 0].astype(int), "high": A[:, 1].astype(int)}).to_csv(
        tmp_path / "annot.tsv", sep="\t", index=False)                # a fit's annotation: CHR, BP, SNP first
    # exact LD scores: own (in-sample, noise-corrected, over every pair of chromosomes) for each sample, and the cross
    # score of the two (within the window, so within a chromosome)
    U1, V1, U2 = [None] * len(blocks), np.empty(M), [None] * len(blocks)
    for i, j in enumerate(blocks):
        U1[i], V1[j], nu1 = _scaled(G1[:, j], C)
        U2[i], _, nu2 = _scaled(G2[:, j], np.zeros((G2.shape[0], 0)))
    L1 = np.tile(-(A * V1[:, None]).sum(0) / nu1, (M, 1))
    L2 = np.tile(-A.sum(0) / nu2, (M, 1))
    Lx = np.empty((M, 2))
    for i, j in enumerate(blocks):
        for i2, j2 in enumerate(blocks):
            R1, R2 = U1[i].T @ U1[i2] / nu1, U2[i].T @ U2[i2] / nu2
            L1[j] += (R1 * R1) @ (A[j2] * V1[j2, None])
            L2[j] += (R2 * R2) @ A[j2]
            if i == i2:
                win = np.abs(bp[j][:, None] - bp[j][None, :]) <= 3_000_000
                Lx[j] = np.where(win, R1 * R2, 0.0) @ (A[j] * np.sqrt(V1[j])[:, None])
            del R1, R2
    del U1, U2
    f = {"own1": _ld_table(tmp_path / "own1", snps, chrom, bp, L1, V1),
         "own2": _ld_table(tmp_path / "own2", snps, chrom, bp, L2, np.ones(M)),
         "cross": _ld_table(tmp_path / "cross", snps, chrom, bp, Lx, np.sqrt(V1)),
         "own1_c": _ld_table(tmp_path / "own1_c", snps, chrom, bp, c1[["low", "high"]].to_numpy(), c1.V.to_numpy()),
         "cross_c": _ld_table(tmp_path / "cross_c", snps, chrom, bp, cx[["low", "high"]].to_numpy(), cx.V.to_numpy())}
    kw = dict(rg=f"{tmp_path / 't1.tsv'},{tmp_path / 't2.tsv'}", annot=str(tmp_path / "annot.tsv"), njack=50,
              chisq_threshold=None, intercept_chisq_thr=None, align_alleles=True, intercept_rg=0.0,
              drop_ambiguous=False)                              # the simulator's C/G SNPs are strand-ambiguous
    run = lambda ld, h2=None: (lambda sc: (sc._run(), sc.result)[1])(Sumcore(ldscores=ld, ldscores_h2=h2, **kw))
    ex = run(f["cross"], (f["own1"], f["own2"]))
    co = run(f["cross_c"], (f["own1_c"], f["own2"]))
    # plumbing: h2 in cross mode is each trait's own fit; with one trace for all, the ordinary rg
    from summit.inference.sumcore import Sumcore as _S
    for t, own in ((1, f["own1"]), (2, f["own2"])):
        plain = run(own)                                             # ordinary rg with this own trace for everything
        h_cross = ex[f"h2_fit{t}"].h2[-1, 0]
        assert h_cross == pytest.approx(plain[f"h2_fit{t}"].h2[-1, 0], rel=1e-10)
    same = run(f["own1"], (f["own1"], f["own1"]))
    plain1 = run(f["own1"])
    assert same["rg_fit"].rg_total[0] == pytest.approx(plain1["rg_fit"].rg_total[0], rel=1e-10)
    # composed against exact LD on the same GWAS, and both against the truth
    rg_e, se_e = ex["rg_fit"].rg_total
    rg_c, se_c = co["rg_fit"].rg_total
    print(f"\ncross-population rg (truth 0.6): exact LD {rg_e:.4f} (SE {se_e:.4f}), composed {rg_c:.4f} (SE {se_c:.4f}); "
          f"gamma {ex['rg_fit'].gamma_total[0]:.5g} / {co['rg_fit'].gamma_total[0]:.5g}; h2 trait 1 "
          f"{ex['h2_fit1'].h2[-1, 0]:.4f} / {co['h2_fit1'].h2[-1, 0]:.4f}, trait 2 {ex['h2_fit2'].h2[-1, 0]:.4f}")
    assert abs(rg_c - rg_e) < 0.1 * se_e + 0.02, (rg_c, rg_e, se_e)
    assert co["rg_fit"].gamma_total[0] == pytest.approx(ex["rg_fit"].gamma_total[0], rel=0.05)
    assert abs(rg_e - 0.6) < 3 * se_e and abs(rg_c - 0.6) < 3 * se_c, (rg_e, se_e, rg_c, se_c)
    assert ex["intercept"].c[0] == 0.0
