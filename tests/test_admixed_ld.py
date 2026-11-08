"""Tests of `summit admixed-ld`: the reference resource, variant matching, the standard-error regression, the genotype
curve, and composed LD scores against exact in-sample LD scores of a simulated admixed cohort."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from admixed_ld_sim import (gwas, insample_ldscores, principal_components, simulate, simulate_families,  # noqa: E402
                            write_bed)

from summit.admixed_ld._io import match_variants  # noqa: E402
from summit.admixed_ld.model import Kernel, regress_standard_errors  # noqa: E402
from summit.admixed_ld.reference import Reference, _band_structure, build_reference  # noqa: E402


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    d = tmp_path_factory.mktemp("admixed_ld")
    S = simulate(seed=3)
    v = S["variants"]
    write_bed(str(d / "panelA"), S["panel_A"], v, "a")
    vE = v.copy()
    sw = np.arange(len(v)) % 3 == 0                                   # second panel lists some alleles the other way
    vE.loc[sw, ["A1", "A2"]] = vE.loc[sw, ["A2", "A1"]].to_numpy()
    GE = S["panel_E"].copy()
    GE[:, sw] = 2 - GE[:, sw]
    write_bed(str(d / "panelE"), GE, vE, "e")
    iid = write_bed(str(d / "cohort"), S["cohort"], v, "c")
    C = principal_components(S["cohort"], 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(d / "cohort.cov", sep="\t",
                                                                                   index=False)
    rng = np.random.default_rng(5)
    os.makedirs(d / "ss")
    for t in range(3):
        y = rng.standard_normal(S["cohort"].shape[0])
        beta, se = gwas(S["cohort"], y, C)
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta,
                      "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(d / "ss" / f"trait{t}.tsv", sep="\t",
                                                                       index=False)
    p = S["cohort"].mean(0) / 2
    cls = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"SNP": v.SNP, "low": (cls == 0).astype(int), "high": (cls == 1).astype(int)}).to_csv(
        d / "annot.tsv", sep="\t", index=False)
    build_reference([str(d / "panelA"), str(d / "panelE")], ["AFR", "EUR"], str(d / "ref"), window_kb=3000,
                    step=256, log=lambda *a: None)
    return dict(dir=d, S=S, C=C, cls=cls)


def test_band_structure():
    bp = np.array([10, 20, 35, 100, 101, 160])
    hi, indptr, indices, rows = _band_structure(bp, 30)
    pairs = {(int(r), int(k)) for r, k in zip(rows, indices)}
    brute = {(j, k) for j in range(bp.size) for k in range(j + 1, bp.size) if bp[k] - bp[j] <= 30}
    assert pairs == brute
    assert indptr[-1] == len(brute)
    assert np.all(np.diff(indptr) >= 0)


def test_reference_matches_numpy(sim):
    ref = Reference(str(sim["dir"] / "ref"))
    S = sim["S"]
    z = ref.load("1")
    m = z["bp"].size
    for s, G in ((0, S["panel_A"]), (1, S["panel_E"])):
        X = G[:, :m] - G[:, :m].mean(0)
        n = G.shape[0]
        np.testing.assert_allclose(z[f"p_{s}"], G[:, :m].mean(0) / 2, atol=1e-12)
        np.testing.assert_allclose(z[f"v_{s}"], (X * X).sum(0) / (n - 1), rtol=1e-10)
        cnt = np.diff(z["indptr"])
        rows = np.repeat(np.arange(m), cnt)
        k = z["indices"].astype(int)
        full = X.T @ X / (n - 1)
        np.testing.assert_allclose(z[f"cov_{s}"], full[rows, k], rtol=2e-6, atol=2e-7)
        m22 = (X * X).T @ (X * X) / n
        np.testing.assert_allclose(z[f"m22_{s}"], m22[rows, k], rtol=2e-6, atol=2e-7)


def test_match_variants():
    t = pd.DataFrame({"CHR": ["1"] * 5, "BP": [1, 2, 3, 4, 5], "SNP": list("abcde"),
                      "COUNTED": ["A", "A", "A", "A", "A"], "OTHER": ["G", "G", "T", "G", "G"]})
    r = pd.DataFrame({"CHR": ["1"] * 6, "BP": [1, 2, 3, 4, 5, 5], "SNP": list("abcdef"),
                      "COUNTED": ["A", "G", "T", "T", "A", "A"], "OTHER": ["G", "A", "A", "C", "G", "G"]})
    ti, ri, fl = match_variants(t, r)
    # 1 same; 2 swapped; 3 ambiguous A/T listed swapped (no strand resolution); 4 strand flip same; 5 duplicate key
    assert ti.tolist() == [0, 1, 2, 3]
    assert fl.tolist() == [False, True, True, False]


def test_standard_error_regression_recovers_ancestry(sim):
    S, C = sim["S"], sim["C"]
    ref = Reference(str(sim["dir"] / "ref"))
    rv = ref.variants(["AFR", "EUR"])
    ss = pd.read_csv(sim["dir"] / "ss" / "trait0.tsv", sep="\t").set_index("SNP").loc[rv.SNP]
    r = regress_standard_errors(ss.N, ss.SE, ss.BETA, rv.v_AFR, rv.v_EUR, rv.p_AFR - rv.p_EUR)
    assert abs(r["m"] - S["theta"].mean()) < 0.03
    la = S["local_ancestry"]
    Z = np.column_stack([np.ones(la.shape[0]), C])
    Q, _ = np.linalg.qr(Z)
    res = la - Q @ (Q.T @ la)
    kappa_true = float(np.mean(np.einsum("ij,ij->j", res, res) / (la.shape[0] - 3)))
    assert abs(r["kappa0"] / kappa_true - 1) < 0.15


def test_curve_recovers_decay_and_ancestry(sim):
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    curve, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "cohort"), covar=str(d / "cohort.cov"),
                                 cell_cm=0.5, log=lambda *a: None)
    lam = info["fit_2_to_20cm"]["lam"]
    assert abs(lam / (sim["S"]["generations"] / 100) - 1) < 0.35
    assert abs(info["theta"]["mean"] - sim["S"]["theta"].mean()) < 0.03
    assert info["inter_kernel"] < 0.01
    assert os.path.exists(f"{d / 'cohort'}.curve.tsv")


def test_curve_relatedness_is_that_of_adjusted_genotypes(sim, tmp_path):
    """The curve's relatedness of listed pairs equals the GRM of the covariate-adjusted, standardized genotypes."""
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    G, C = sim["S"]["cohort"], sim["C"]
    iid = pd.read_csv(f"{d / 'cohort'}.fam", sep=r"\s+", header=None, dtype=str)[1].to_numpy()
    pairs = [(0, 1), (2, 7), (10, 11), (5, 40), (100, 3)]
    pd.DataFrame({"IID1": [iid[a] for a, _ in pairs], "IID2": [iid[b] for _, b in pairs],
                  "K": [0.5, 0.2, 0.05, 0.3, 0.01]}).to_csv(tmp_path / "pairs.tsv", sep="\t", index=False)
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(tmp_path / "pairs.tsv"), kinship_col="K",
                             kinship_min=0.02, closure=False, log=lambda *a: None)
    X = G - G.mean(0)
    Cs = (C - C.mean(0)) / C.std(0, ddof=1)
    Q, _ = np.linalg.qr(Cs)
    Xr = X - Q @ (Q.T @ X)
    var = np.einsum("ij,ij->j", Xr, Xr) / (G.shape[0] - 1 - C.shape[1])
    ok = var > 1e-8
    Z = Xr[:, ok] / np.sqrt(var[ok])
    g = np.array([Z[a] @ Z[b] / ok.sum() for a, b in pairs[:4]])     # the last pair is below --kinship-min
    r = info["relatedness"]
    assert r["pairs"] == 4 and r["n_snps"] == ok.sum()
    assert r["sum_squared_ordered"] == pytest.approx(2 * np.sum(g ** 2), rel=1e-8)
    # --closure adds the pairs implied within a family: (0, 1) and (1, 2) give (0, 2)
    pd.DataFrame({"IID1": [iid[0], iid[1]], "IID2": [iid[1], iid[2]]}).to_csv(tmp_path / "chain.tsv", sep="\t",
                                                                           index=False)
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c2"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(tmp_path / "chain.tsv"), closure=True,
                             log=lambda *a: None)
    g = np.array([Z[a] @ Z[b] / ok.sum() for a, b in ((0, 1), (0, 2), (1, 2))])
    assert info["relatedness"]["pairs"] == 3
    assert info["relatedness"]["sum_squared_ordered"] == pytest.approx(2 * np.sum(g ** 2), rel=1e-8)


@pytest.mark.parametrize("decay", ["curve", "date"])
def test_composed_ldscores_match_insample(sim, decay):
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    if not os.path.exists(f"{d / 'cohort'}.curve.json"):
        genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "cohort"), covar=str(d / "cohort.cov"),
                       cell_cm=0.5, log=lambda *a: None)
    kw = dict(curve=str(d / "cohort")) if decay == "curve" else dict(generations=sim["S"]["generations"])
    out = str(d / f"composed_{decay}")
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), out, log=lambda *a: None, **kw)
    got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
    assert list(got.columns) == ["CHR", "SNP", "BP", "V", "low", "high"]
    keep = sim["S"]["variants"].SNP.isin(got.SNP).to_numpy()          # SNPs without variance in both panels drop
    assert keep.sum() == len(got) and keep.mean() > 0.99
    M_file = np.loadtxt(out + ".gw.M")
    np.testing.assert_allclose(M_file, np.bincount(sim["cls"][keep], minlength=2))
    L_in, V_in = insample_ldscores(sim["S"]["cohort"][:, keep], sim["C"], sim["cls"][keep], 2)
    L_c = got[["low", "high"]].to_numpy()
    mass_in = float((V_in[:, None] * L_in).sum())
    mass_c = float((got.V.to_numpy()[:, None] * L_c).sum())
    tol = 0.05 if decay == "curve" else 0.10
    assert abs(mass_c / mass_in - 1) < tol, (mass_c, mass_in)
    for c in range(2):
        assert abs(L_c[:, c].sum() / L_in[:, c].sum() - 1) < 2 * tol
    # the date path assumes one exponential plus the genome-wide constant, so it cannot follow the chromosome-level
    # term that a PC takes up in this two-chromosome genome
    assert np.corrcoef(L_c.sum(1), L_in.sum(1))[0, 1] > (0.9 if decay == "curve" else 0.85)
    # V: the model has genome-wide ancestry only; in this two-chromosome genome a PC also takes up chromosome-level
    # ancestry, so compare V on average rather than SNP by SNP.
    dv = got.V.to_numpy() - V_in
    assert abs(np.median(dv)) < 0.01 and np.mean(np.abs(dv)) < 0.03
    params = json.load(open(out + ".admixed_ld.json"))
    assert params["n_snps"] == len(got)


def test_command_line_end_to_end(sim, tmp_path):
    from summit.entrypoint import main
    d = sim["dir"]
    ref = str(tmp_path / "ref")
    assert main(["admixed-ld", "reference", "--panel", str(d / "panelA"), "--panel", str(d / "panelE"),
                 "--names", "AFR,EUR", "--out", ref, "--step", "300"]) == 0
    assert main(["admixed-ld", "curve", "--geno", str(d / "cohort"), "--ref", ref, "--covar", str(d / "cohort.cov"),
                 "--cell-cm", "0.5", "--out", str(tmp_path / "c")]) == 0
    out = str(tmp_path / "ld")
    assert main(["admixed-ld", "compose", "--ref", ref, "--sumstats", str(d / "ss"), "--annot", str(d / "annot.tsv"),
                 "--curve", str(tmp_path / "c"), "--out", out]) == 0
    assert os.path.exists(out + ".gw.ldscore.gz") and os.path.exists(out + ".gw.M")
    assert main(["admixed-ld", "compose", "--ref", ref, "--sumstats", str(d / "ss" / "trait0.tsv"),
                 "--annot", str(d / "annot.tsv"), "--admixture-generations", "6", "--ancestry-sd", "0.1",
                 "--out", out + "_date"]) == 0


def test_unbiased_squared_covariance_is_the_u_statistic():
    """The closed form equals the U-statistic of h(i,j) h(k,l), h(i,j) = (x_i - x_j)(y_i - y_j) / 2, over all ordered
    quadruples of distinct people, which is unbiased for the squared covariance."""
    from itertools import permutations
    from summit.admixed_ld.compose import unbiased_squared_covariance
    rng = np.random.default_rng(7)
    n = 8
    x = rng.binomial(2, 0.3, n).astype(float)
    y = np.clip(x + rng.binomial(1, 0.4, n) - rng.binomial(1, 0.3, n), 0, 2)
    h = lambda i, j: (x[i] - x[j]) * (y[i] - y[j]) / 2.0
    brute = np.mean([h(i, j) * h(k, l) for i, j, k, l in permutations(range(n), 4)])
    a, b = x - x.mean(), y - y.mean()
    c, m22 = (a * b).sum() / (n - 1), (a * a * b * b).mean()
    vv = (a * a).sum() / (n - 1) * (b * b).sum() / (n - 1)
    assert unbiased_squared_covariance(c, m22, vv, n) == pytest.approx(brute, rel=1e-10, abs=1e-12)


def test_kernel_forms():
    k = Kernel.from_decay(0.2, 0.08, 3500.0, source="t")
    assert k(0.0) == pytest.approx(0.2)
    assert k.far == pytest.approx(-2 * 0.2 / (0.08 * 3500))
    assert k(1e6) == pytest.approx(k.far)
    d = np.arange(0.5, 40.5, 0.5)
    kc = Kernel.from_curve(d, 0.18 * np.exp(-0.07 * d) - 0.001, -0.001, 0.2, half_width=0, rel_width=0)
    assert kc(0.0) == pytest.approx(0.2)
    assert kc(10.0) == pytest.approx(0.18 * np.exp(-0.7) - 0.001, rel=1e-6)
    assert kc(100.0) == pytest.approx(-0.001)
    assert kc.lam == pytest.approx(0.07, rel=0.05)


@pytest.fixture(scope="module")
def sim3(tmp_path_factory):
    from admixed_ld_sim import simulate_k
    d = tmp_path_factory.mktemp("admixed_ld_k3")
    S = simulate_k(seed=4)
    v = S["variants"]
    names = ["S0", "S1", "S2"]
    for nm, G in zip(names, S["panels"]):
        write_bed(str(d / f"panel{nm}"), G, v, nm)
    iid = write_bed(str(d / "cohort"), S["cohort"], v, "c")
    # as many PCs as ancestry axes: in a three-chromosome genome further PCs take up chromosome-level ancestry
    C = principal_components(S["cohort"], 2)
    pd.DataFrame({"FID": iid, "IID": iid, **{f"PC{k + 1}": C[:, k] for k in range(2)}}).to_csv(
        d / "cohort.cov", sep="\t", index=False)
    rng = np.random.default_rng(6)
    os.makedirs(d / "ss")
    for t in range(3):
        y = rng.standard_normal(S["cohort"].shape[0])
        beta, se = gwas(S["cohort"], y, C)
        ok = np.isfinite(beta)
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta,
                      "SE": se, "COV_RANK": 2})[ok].to_csv(d / "ss" / f"trait{t}.tsv", sep="\t", index=False)
    p = S["cohort"].mean(0) / 2
    cls = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"SNP": v.SNP, "low": (cls == 0).astype(int), "high": (cls == 1).astype(int)}).to_csv(
        d / "annot.tsv", sep="\t", index=False)
    build_reference([str(d / f"panel{nm}") for nm in names], names, str(d / "ref"), window_kb=3000, step=256,
                    log=lambda *a: None)
    return dict(dir=d, S=S, C=C, cls=cls, names=names)


def test_three_sources_compose_matches_insample(sim3):
    """Three sources: the curve recovers the ancestry vector and the decay, the standard errors the mean ancestry, and
    the composed LD scores match exact in-sample LD scores."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d, S, names = sim3["dir"], sim3["S"], sim3["names"]
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "cohort"), sources=names,
                             covar=str(d / "cohort.cov"), cell_cm=0.5, log=lambda *a: None)
    np.testing.assert_allclose(info["theta"]["mean_vector"], S["theta"].mean(0)[:2], atol=0.03)
    for s in names:
        assert abs(info["fit_per_source"][s]["lam"] / (S["generations"] / 100) - 1) < 0.5
    out = str(d / "composed3")
    params = compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), out, sources=names,
                              curve=str(d / "cohort"), log=lambda *a: None)
    np.testing.assert_allclose(params["mean_ancestry_vector"], S["m"], atol=0.05)
    got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
    keep = S["variants"].SNP.isin(got.SNP).to_numpy()
    L_in, V_in = insample_ldscores(S["cohort"][:, keep], sim3["C"], sim3["cls"][keep], 2)
    L_c = got[["low", "high"]].to_numpy()
    mass_in = float((V_in[:, None] * L_in).sum())
    mass_c = float((got.V.to_numpy()[:, None] * L_c).sum())
    assert abs(mass_c / mass_in - 1) < 0.03, (mass_c, mass_in)
    assert np.corrcoef(L_c.sum(1), L_in.sum(1))[0, 1] > 0.95


def test_derived_source_from_the_cohort(sim3, tmp_path):
    """A third source with only a small panel: derived from the cohort, its frequencies match the source's and the
    composed LD scores match in-sample LD scores about as well as with a full panel of that source."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.derive import derive_source
    d, S, names = sim3["dir"], sim3["S"], sim3["names"]
    v = S["variants"]
    write_bed(str(tmp_path / "small"), S["panels"][2][:40], v, "s")
    build_reference([str(d / "panelS0"), str(d / "panelS1")], names[:2], str(tmp_path / "ref2"), step=256,
                    log=lambda *a: None)
    derive_source(str(d / "cohort"), str(tmp_path / "ref2"), str(tmp_path / "ref3"), name="S2",
                  panel=str(tmp_path / "small"), covar=str(d / "cohort.cov"), step=256, log=lambda *a: None)
    ref3 = Reference(str(tmp_path / "ref3"))
    rv = ref3.variants(["S2"]).set_index("SNP")
    p_true = pd.Series(S["panels"][2].mean(0) / 2, index=v.SNP).reindex(rv.index)
    assert np.corrcoef(rv.p_S2, p_true)[0, 1] > 0.98
    genotype_curve(str(d / "cohort"), str(tmp_path / "ref3"), str(tmp_path / "c3"), sources=names,
                   covar=str(d / "cohort.cov"), cell_cm=0.5, log=lambda *a: None)
    out = str(tmp_path / "derived")
    compose_ldscores(str(tmp_path / "ref3"), [str(d / "ss")], str(d / "annot.tsv"), out, sources=names,
                     curve=str(tmp_path / "c3"), log=lambda *a: None)
    got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
    keep = v.SNP.isin(got.SNP).to_numpy()
    L_in, V_in = insample_ldscores(S["cohort"][:, keep], sim3["C"], sim3["cls"][keep], 2)
    L_c = got[["low", "high"]].to_numpy()
    mass_in = float((V_in[:, None] * L_in).sum())
    mass_c = float((got.V.to_numpy()[:, None] * L_c).sum())
    assert abs(mass_c / mass_in - 1) < 0.05, (mass_c, mass_in)
    assert np.corrcoef(L_c.sum(1), L_in.sum(1))[0, 1] > 0.93


def test_contrast_noise_identity_and_mixing():
    """Panels alone give the independent-noise products; a derived source's errors that carry the panels' errors give
    the covariance of the contrasts' errors (checked by simulation)."""
    from summit.admixed_ld.model import contrast_noise, unbiased_products, unbiased_products_cov
    rng = np.random.default_rng(1)
    M = 40
    V = rng.uniform(0.05, 0.5, (M, 3))
    n = np.array([400.0, 300.0, 40.0])
    D = rng.normal(size=(M, 2))
    S, R = contrast_noise(V, n)
    np.testing.assert_allclose(unbiased_products_cov(D, S),
                               unbiased_products(D, V[:, :2] / (4 * n[:2]), V[:, 2] / (4 * n[2])))
    np.testing.assert_allclose(R, -np.repeat((V[:, 2] / (4 * n[2]))[:, None], 2, axis=1))
    G = np.array([[1, 0, 0], [0, 1, 0], [-0.2, -0.8, 1.0]])
    S2, R2 = contrast_noise(V, n, G)
    eta = rng.standard_normal((400000, 3)) * np.sqrt(V[0] / (4 * n))
    e = eta @ G.T
    dn = e[:, :2] - e[:, 2:3]
    np.testing.assert_allclose(np.cov(dn.T), S2[0], rtol=0.02)
    np.testing.assert_allclose([np.cov(dn[:, a], e[:, 2])[0, 1] for a in range(2)], R2[0], rtol=0.03)


def test_ridge_calibration_recovers_the_family_position():
    """Frequencies of a pseudo-source p' = (pN - b pK) / alpha, noisy and carrying the panels' errors, and a small
    anchor panel of N: the calibration recovers alpha and b, also with errors in every input."""
    from summit.admixed_ld.derive import ridge_calibration
    rng = np.random.default_rng(3)
    M = 60000
    anc = rng.beta(0.5, 0.5, M)
    pA = np.clip(anc + rng.normal(0, 0.15, M), 0.01, 0.99)
    pE = np.clip(anc + rng.normal(0, 0.15, M), 0.01, 0.99)
    pN = np.clip(anc + rng.normal(0, 0.15, M), 0.01, 0.99)
    alpha, b = 1.08, np.array([0.02, -0.10])
    nk, n0, na = np.array([400.0, 400.0]), 1500.0, 40
    eA = rng.normal(0, np.sqrt(pA * (1 - pA) / (2 * nk[0])))
    eE = rng.normal(0, np.sqrt(pE * (1 - pE) / (2 * nk[1])))
    c = np.array([0.2, 0.8])
    pp = (pN - b[0] * pA - b[1] * pE) / alpha
    var0 = np.abs(pp * (1 - pp)) / (2 * n0) + 1e-6
    p_der = pp + rng.normal(0, np.sqrt(var0)) - c[0] * eA - c[1] * eE
    f = rng.binomial(2 * na, pN) / (2 * na)
    Pk = np.column_stack([pA + eA, pE + eE])
    Vk = 2 * Pk * (1 - Pk)
    out = ridge_calibration(f, na, p_der, var0, c, Pk, Vk, nk)
    assert abs(out["alpha"] - alpha) < 4 * out["se_alpha"] + 0.003, out
    np.testing.assert_allclose(out["b"], b, atol=0.01)


def test_derived_source_with_panel_ld(sim3, tmp_path):
    """The derived source with the anchor panel's window LD: frequencies calibrated to the anchors, the manifest
    carries their error structure, and composed LD scores match in-sample LD scores."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.derive import derive_source
    d, S, names = sim3["dir"], sim3["S"], sim3["names"]
    v = S["variants"]
    write_bed(str(tmp_path / "small"), S["panels"][2][:40], v, "s")
    build_reference([str(d / "panelS0"), str(d / "panelS1")], names[:2], str(tmp_path / "ref2"), step=256,
                    log=lambda *a: None)
    man = derive_source(str(d / "cohort"), str(tmp_path / "ref2"), str(tmp_path / "ref3"), name="S2",
                        panel=str(tmp_path / "small"), covar=str(d / "cohort.cov"), ld="panel", step=256,
                        log=lambda *a: None)
    src = man["sources"][2]
    assert src["n"] == 40 and src["ld"] == "panel" and set(src["noise_mix"]) == {"S0", "S1"}
    assert 0 < src["n_freq"] <= src["n_freq_weights"] * 1.05
    ref3 = Reference(str(tmp_path / "ref3"))
    z = ref3.load(ref3.chromosomes[0])
    assert {"cov_2", "m22_2", "vld_2", "fvar_2"} <= set(z.files) and "covvar_2" not in z.files
    assert (z["fvar_2"] > 0).all() and np.isfinite(z["v_2"]).all()
    rv = ref3.variants(["S2"]).set_index("SNP")
    p_true = pd.Series(S["panels"][2].mean(0) / 2, index=v.SNP).reindex(rv.index)
    assert np.corrcoef(rv.p_S2, p_true)[0, 1] > 0.98
    # panel covariances under a manifest entry that looks like an inversion source (m_anchor, no ld key): no mix
    assert ref3.covariance_noise_mix("S2") is None
    look = json.load(open(tmp_path / "ref3" / "manifest.json"))
    look["sources"][2].pop("ld")
    look["sources"][2].update(m_anchor=[0.5, 0.3, 0.2], decay_anchor=[0.1, 0.1, 0.1])
    os.makedirs(tmp_path / "look")
    json.dump(look, open(tmp_path / "look" / "manifest.json", "w"))
    for c in ref3.chromosomes:
        os.symlink(tmp_path / "ref3" / f"chr{c}.npz", tmp_path / "look" / f"chr{c}.npz")
    assert Reference(str(tmp_path / "look")).covariance_noise_mix("S2") is None
    genotype_curve(str(d / "cohort"), str(tmp_path / "ref3"), str(tmp_path / "c3"), sources=names,
                   covar=str(d / "cohort.cov"), cell_cm=0.5, log=lambda *a: None)
    out = str(tmp_path / "derived")
    compose_ldscores(str(tmp_path / "ref3"), [str(d / "ss")], str(d / "annot.tsv"), out, sources=names,
                     curve=str(tmp_path / "c3"), log=lambda *a: None)
    got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
    keep = v.SNP.isin(got.SNP).to_numpy()
    L_in, V_in = insample_ldscores(S["cohort"][:, keep], sim3["C"], sim3["cls"][keep], 2)
    L_c = got[["low", "high"]].to_numpy()
    mass_c = float((got.V.to_numpy()[:, None] * L_c).sum())
    mass_in = float((V_in[:, None] * L_in).sum())
    assert abs(mass_c / mass_in - 1) < 0.05, (mass_c, mass_in)
    _check_matrix_with_derived_source(ref3, tmp_path, out, names)


def test_curve_removes_panel_noise_from_its_denominators(sim, tmp_path):
    """With small panels the raw squared contrasts carry the panels' sampling variance, which shrinks the kernel; the
    curve uses the products without it, so a 12-person reference gives about the kernel of the full panels."""
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    S, v = sim["S"], sim["S"]["variants"]
    write_bed(str(tmp_path / "smallA"), S["panel_A"][:12], v, "sa")
    write_bed(str(tmp_path / "smallE"), S["panel_E"][:12], v, "se")
    build_reference([str(tmp_path / "smallA"), str(tmp_path / "smallE")], ["AFR", "EUR"], str(tmp_path / "ref"),
                    window_kb=3000, step=256, log=lambda *a: None)
    kw = dict(covar=str(d / "cohort.cov"), cell_cm=0.5, cells_cm=0, log=lambda *a: None)
    full, _ = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "full"), **kw)
    fixed, info = genotype_curve(str(d / "cohort"), str(tmp_path / "ref"), str(tmp_path / "fixed"), **kw)
    raw, _ = genotype_curve(str(d / "cohort"), str(tmp_path / "ref"), str(tmp_path / "raw"), noise_correction=False,
                            **kw)
    sel = (full.d_cm >= 1) & (full.d_cm <= 20)
    k_full = float(full.kernel[sel].mean())
    r_fixed = float(fixed.set_index("d_cm").kernel.reindex(full.d_cm[sel]).mean()) / k_full
    r_raw = float(raw.set_index("d_cm").kernel.reindex(full.d_cm[sel]).mean()) / k_full
    print(f"kernel ratio to the full panels: raw {r_raw:.3f}, corrected {r_fixed:.3f}")
    assert info["noise_correction"] and info["panel_n_freq"]["AFR"] == 12
    assert r_raw < 0.9                       # the raw denominators shrink the kernel
    assert abs(r_fixed - 1) < abs(r_raw - 1) and abs(r_fixed - 1) < 0.08


def _dweighted_cov(path, lo=1.0, hi=3.0):
    """sum D_j D_k c_jk / sum (D_j D_k)^2 of the first source's covariances over band pairs lo-hi cM apart."""
    R = Reference(path)
    num = den = 0.0
    for c in R.chromosomes:
        z = R.load(c)
        D = z["p_0"] - z["p_1"]
        jl = np.repeat(np.arange(z["snp"].size), np.diff(z["indptr"]))
        kl = z["indices"].astype(np.int64)
        d = np.abs(z["cm"][kl] - z["cm"][jl])
        s = (d >= lo) & (d <= hi)
        w = D[jl[s]] * D[kl[s]]
        num += float((w * z["cov_0"][s]).sum())
        den += float((w * w).sum())
    return num / den


def test_panel_admixture_removed(tmp_path):
    """A source panel with some admixture of its own carries admixture LD; clean-panels removes it and leaves a pure
    panel's reference as it was."""
    import admixed_ld_sim
    from summit.admixed_ld.panel_admixture import remove_panel_admixture
    rng = np.random.default_rng(11)
    n_chr, spc = 8, 300                                            # enough chromosomes for the jackknife
    chrom = np.repeat(np.arange(1, n_chr + 1), spc)
    cm = np.concatenate([np.sort(rng.uniform(0, 60.0, spc)) for _ in range(n_chr)])
    M = cm.size
    pE = rng.uniform(0.05, 0.95, M)
    pA = np.clip(pE + rng.normal(0, 0.25, M), 0.03, 0.97)
    founders = {s: (rng.random((40, M)) < p[None, :]).astype(np.int8) for s, p in (("A", pA), ("E", pE))}
    hap = admixed_ld_sim._example._haplotypes
    n = 800
    hA, _ = hap(rng, {"A": founders["A"]}, cm, 2 * n, 3.0)
    hE, _ = hap(rng, {"E": founders["E"]}, cm, 2 * n, 3.0)
    k = 0.9 * 0.1 / 0.05 ** 2 - 1                                  # panel of source A with 10% of source E (SD 0.05)
    hX, _ = hap(rng, founders, cm, 2 * n, 3.0, anc=np.repeat(rng.beta(0.9 * k, 0.1 * k, n), 2), break_rate=0.08)
    var = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"sim{j:05d}" for j in range(M)], "CM": cm,
                        "BP": np.rint(cm * 1e6).astype(np.int64) + 1, "A1": "A", "A2": "G"})
    for nm, h in (("A", hA), ("E", hE), ("X", hX)):
        write_bed(tmp_path / nm, (h[0::2] + h[1::2]).astype(np.float64), var, nm)
    quiet = lambda *a, **k: None
    for out, a in (("pure", "A"), ("adm", "X")):
        build_reference([str(tmp_path / a), str(tmp_path / "E")], ["A", "E"], str(tmp_path / out), match="id",
                        log=quiet)
    for r, a in (("pure", "A"), ("adm", "X")):
        remove_panel_admixture(str(tmp_path / r), {"A": str(tmp_path / a), "E": str(tmp_path / "E")},
                               out=str(tmp_path / f"{r}_clean"), match="id", log=quiet)
    base = _dweighted_cov(tmp_path / "pure")
    raw = _dweighted_cov(tmp_path / "adm") - base
    cleaned = _dweighted_cov(tmp_path / "adm_clean") - base
    assert raw > 0.1                                                # the panel's admixture LD is visible
    assert abs(cleaned) < 0.15 * raw                                # and mostly removed
    assert json.load(open(tmp_path / "adm_clean" / "manifest.json"))["sources"][0]["panel_admixture"]["removed"]
    assert abs(_dweighted_cov(tmp_path / "pure_clean") - base) < 0.002   # a pure panel stays as it was
    man = json.load(open(tmp_path / "pure_clean" / "manifest.json"))
    assert all((not s["panel_admixture"]["removed"]) or s["panel_admixture"]["kappa0"][0][0] < 0.005
               for s in man["sources"])
    with pytest.raises(ValueError):                                 # cleaning twice is refused
        remove_panel_admixture(str(tmp_path / "pure_clean"), {"A": str(tmp_path / "A")}, match="id", log=quiet)
    # the same fits applied without the panels' genotypes reproduce the cleaned reference exactly
    from summit.admixed_ld.panel_admixture import apply_panel_admixture
    apply_panel_admixture(str(tmp_path / "adm"), str(tmp_path / "adm_clean"), out=str(tmp_path / "adm_applied"),
                          log=quiet)
    for c in json.load(open(tmp_path / "adm_clean" / "manifest.json"))["chromosomes"]:
        za, zb = np.load(tmp_path / "adm_clean" / f"chr{c}.npz"), np.load(tmp_path / "adm_applied" / f"chr{c}.npz")
        assert set(za.files) == set(zb.files)
        for key in za.files:
            np.testing.assert_array_equal(za[key], zb[key])
    with pytest.raises(ValueError):                                 # applying twice is refused
        apply_panel_admixture(str(tmp_path / "adm_applied"), str(tmp_path / "adm_clean"), log=quiet)


def test_regional_matrix(sim, tmp_path):
    """summit admixed-ld matrix: the composed correlation equals the formula evaluated on the panel genotypes, is
    positive semi-definite, is closer to the cohort's covariate-adjusted in-sample correlation than the matrix without
    the admixture term, and the command line writes the three files."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.matrix import compose_matrix
    from summit.admixed_ld.model import KernelMatrix
    from summit.entrypoint import main
    d, S = sim["dir"], sim["S"]
    if not os.path.exists(f"{d / 'cohort'}.curve.json"):
        genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "cohort"), covar=str(d / "cohort.cov"),
                       cell_cm=0.5, log=lambda *a: None)
    comp = str(d / "composed_curve")
    if not os.path.exists(comp + ".admixed_ld.json"):
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), comp, curve=str(d / "cohort"),
                         log=lambda *a: None)
    v = S["variants"]
    c0 = v.CHR.astype(str).iloc[0]
    on = np.flatnonzero((v.CHR.astype(str) == c0).to_numpy())
    on = on[(v.BP.to_numpy()[on] >= v.BP.to_numpy()[on[100]]) & (v.BP.to_numpy()[on] < v.BP.to_numpy()[on[100]] + 2_900_000)]
    assert on.size >= 20                                              # inside one reference window (3000 kb)
    ids = tmp_path / "snps.txt"
    v.SNP.iloc[on].to_csv(ids, index=False, header=False)
    R, snp, info = compose_matrix(str(d / "ref"), str(tmp_path / "m"), snps=str(ids), params=comp + ".admixed_ld.json",
                                  curve=str(d / "cohort"), taper="none", log=lambda *a: None)
    sel = pd.Index(v.SNP).get_indexer(snp.SNP)
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 1.0) and info["min_eigenvalue"] > -1e-8
    assert info["complete_band"]
    # the formula on the panel genotypes
    GA, GE = S["panel_A"][:, sel].astype(float), S["panel_E"][:, sel].astype(float)
    CA, CE = np.cov(GA, rowvar=False), np.cov(GE, rowvar=False)
    Dv = GA.mean(0) / 2 - GE.mean(0) / 2
    pr = json.load(open(comp + ".admixed_ld.json"))
    m = float(pr["mean_ancestry"])
    ci, cdf = json.load(open(f"{d / 'cohort'}.curve.json")), pd.read_csv(f"{d / 'cohort'}.curve.tsv", sep="\t")
    lam = ci["fit_2_to_20cm"]["lam"]
    kern = KernelMatrix.from_curve(cdf.d_cm.to_numpy(), cdf.kernel.to_numpy(), [[ci["inter_kernel"]]],
                                   [[pr["kappa0"]]], [lam, lam])
    cm = v.CM.to_numpy()[sel]
    dd = np.abs(cm[:, None] - cm[None, :])
    C = m * np.exp(-lam * dd) * CA + (1 - m) * np.exp(-lam * dd) * CE + Dv[:, None] * Dv[None, :] * kern(dd)[..., 0, 0]
    sd = np.sqrt(np.diag(C))
    np.testing.assert_allclose(R, C / sd[:, None] / sd[None, :], atol=1e-6)
    # against the in-sample correlation of the covariate-adjusted cohort genotypes
    X = S["cohort"][:, sel].astype(float)
    X -= X.mean(0)
    Q = np.linalg.qr(np.column_stack([np.ones(len(X)), sim["C"]]))[0]
    X -= Q @ (Q.T @ X)
    Rin = np.corrcoef(X, rowvar=False)
    R0, _, _ = compose_matrix(str(d / "ref"), str(tmp_path / "m0"), snps=str(ids), params=comp + ".admixed_ld.json",
                              curve=str(d / "cohort"), taper="none", admixture=False, log=lambda *a: None)
    iu = np.triu_indices(len(sel), 1)
    err, err0 = np.sqrt(np.mean((R - Rin)[iu] ** 2)), np.sqrt(np.mean((R0 - Rin)[iu] ** 2))
    assert err < 0.8 * err0, (err, err0)          # 0.62 in this small simulation (in-sample noise at n = 3000)
    Rt, _, it = compose_matrix(str(d / "ref"), str(tmp_path / "mt"), snps=str(ids), params=comp + ".admixed_ld.json",
                               curve=str(d / "cohort"), taper="auto", log=lambda *a: None)
    assert all(np.isfinite(x["tau_cm"]) for x in it["taper"].values()) and it["min_eigenvalue"] > -1e-8
    out = str(tmp_path / "cli")
    assert main(["admixed-ld", "matrix", "--ref", str(d / "ref"), "--params", comp + ".admixed_ld.json",
                 "--region", f"{c0}:{int(v.BP.iloc[on[0]])}-{int(v.BP.iloc[on[-1]])}", "--admixture-generations",
                 str(S["generations"]), "--taper", "none", "--out", out]) == 0
    for ext in (".ld", ".ld.npz", ".snps.tsv", ".matrix.json"):
        assert os.path.exists(out + ext)
    assert len(pd.read_csv(out + ".snps.tsv", sep="\t")) == len(sel)


def test_window_noise_counts_derived_panel_noise_once():
    """A source derived by inversion, C^N = [R - sum_t pi^a_t C^t] / pi^a_N with the anchor cohort's pi^a, carries minus
    the known panels' errors. Composed for a target with other proportions and decays (pi != pi^a), the noise variance
    that compose subtracts equals the variance of the composed covariance (simulated), and leaving out the covariance
    with the panels' errors (the error before 4 Nov 2026) removes their noise twice."""
    from summit.admixed_ld.compose import window_noise
    rng = np.random.default_rng(12)
    d = np.array([0.0, 0.02, 0.1, 0.3, 1.0])
    ma, la = np.array([0.10, 0.50, 0.40]), np.array([0.09, 0.12, 0.10])    # anchor cohort: known S0, S1, derived S2
    mt, lt = np.array([0.25, 0.45, 0.30]), np.array([0.15, 0.08, 0.12])    # target
    pa = ma[:, None] * np.exp(-la[:, None] * d[None, :])
    pt = mt[:, None] * np.exp(-lt[:, None] * d[None, :])
    Ck = rng.uniform(0.05, 0.3, (2, d.size))
    CN = rng.uniform(0.05, 0.3, d.size)
    Vk = np.array([2.0, 1.0])[:, None] * rng.uniform(0.5, 1.5, (2, d.size)) * 1e-3     # panels' Var(c_t)
    VR = 2e-4 * np.ones(d.size)                                            # the cohort's Var(c_R)
    B = 400_000
    ek = rng.standard_normal((B, 2, d.size)) * np.sqrt(Vk)[None]
    eR = rng.standard_normal((B, d.size)) * np.sqrt(VR)[None]
    R = (pa[:2] * Ck).sum(0) + pa[2] * CN
    cN = (R + eR - (pa[:2][None] * (Ck[None] + ek)).sum(1)) / pa[2]
    covvar = (VR + (pa[:2] ** 2 * Vk).sum(0)) / pa[2] ** 2
    np.testing.assert_allclose(np.var(cN, axis=0), covvar, rtol=0.02)     # covvar is the derived covariance's own
    comp = (pt[:2][None] * (Ck[None] + ek)).sum(1) + pt[2] * cN
    cross = [(2, t, ma[2], la[2], ma[t], la[t]) for t in range(2)]
    pis, vns = [pt[0], pt[1], pt[2]], [Vk[0], Vk[1], covvar]
    got = window_noise(pis, vns, cross, d)
    np.testing.assert_allclose(got, np.var(comp, axis=0), rtol=0.02)
    old = window_noise(pis, vns, [], d)
    assert np.all(old - np.var(comp, axis=0) > 10 * 0.02 * np.var(comp, axis=0))
    # with the target's pi equal to the anchor's, the cross term is twice the panels' part of covvar
    same = window_noise([pa[0], pa[1], pa[2]], [Vk[0], Vk[1], covvar], cross, d)
    np.testing.assert_allclose(same, VR, rtol=1e-12)


def _check_matrix_with_derived_source(ref, tmp_path, params_prefix, names):
    """The regional matrix (with the automatic taper) works for a reference with a derived source."""
    from summit.admixed_ld.matrix import compose_matrix
    z = ref.load(ref.chromosomes[0])
    ids = tmp_path / "region.txt"
    pd.Series(z["snp"][10:40]).to_csv(ids, index=False, header=False)
    R, snp, info = compose_matrix(ref.path, str(tmp_path / "mat"), snps=str(ids), sources=names,
                                  params=params_prefix + ".admixed_ld.json", curve=str(tmp_path / "c3"),
                                  taper="auto", log=lambda *a: None)
    assert len(snp) == 30 and np.allclose(np.diag(R), 1.0) and info["min_eigenvalue"] > -1e-8
    assert all(np.isfinite(info["taper"][s]["tau_cm"]) for s in names)


def test_inversion_counts_panel_noise_once(sim3, tmp_path):
    """LD by inversion with small known panels: their noise enters the derived covariances with a minus sign and enters
    the composition again, so compose must remove it once. With 200-person panels and a 3,000-person cohort the
    composed LD mass is then within 2.5% of in-sample (0.992 here); counting the noise twice, as before 4 Nov 2026,
    gave 0.919."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.derive import derive_source
    d, S, names = sim3["dir"], sim3["S"], sim3["names"]
    v = S["variants"]
    for i in range(2):
        write_bed(str(tmp_path / f"p{i}"), S["panels"][i][:200], v, f"p{i}")
    write_bed(str(tmp_path / "small"), S["panels"][2][:40], v, "s")
    build_reference([str(tmp_path / "p0"), str(tmp_path / "p1")], names[:2], str(tmp_path / "ref2"), step=256,
                    log=lambda *a: None)
    man = derive_source(str(d / "cohort"), str(tmp_path / "ref2"), str(tmp_path / "ref3"), name="S2",
                        panel=str(tmp_path / "small"), covar=str(d / "cohort.cov"), step=256, log=lambda *a: None)
    cn = man["sources"][2]["covariance_noise"]
    assert set(cn["mix"]) == {"S0", "S1"} and len(cn["pi"]) == 2
    ref3 = Reference(str(tmp_path / "ref3"))
    assert ref3.covariance_noise_mix("S0") is None and ref3.covariance_noise_mix("S2")[1].keys() == {"S0", "S1"}
    old = json.load(open(tmp_path / "ref3" / "manifest.json"))             # the layout before covariance_noise
    del old["sources"][2]["covariance_noise"]
    os.makedirs(tmp_path / "old_layout")
    for c in ref3.chromosomes:
        os.symlink(tmp_path / "ref3" / f"chr{c}.npz", tmp_path / "old_layout" / f"chr{c}.npz")
    json.dump(old, open(tmp_path / "old_layout" / "manifest.json", "w"))
    assert Reference(str(tmp_path / "old_layout")).covariance_noise_mix("S2") == ref3.covariance_noise_mix("S2")
    old["sources"][2]["how"] = "derive, covvar without the doubly counted panel noise (test)"
    json.dump(old, open(tmp_path / "old_layout" / "manifest.json", "w"))
    with pytest.raises(ValueError):                                         # the research fix must not be stacked
        Reference(str(tmp_path / "old_layout")).covariance_noise_mix("S2")
    genotype_curve(str(d / "cohort"), str(tmp_path / "ref3"), str(tmp_path / "c3"), sources=names,
                   covar=str(d / "cohort.cov"), cell_cm=0.5, log=lambda *a: None)
    out = str(tmp_path / "inv")
    params = compose_ldscores(str(tmp_path / "ref3"), [str(d / "ss")], str(d / "annot.tsv"), out, sources=names,
                              curve=str(tmp_path / "c3"), log=lambda *a: None)
    assert {x["panel"] for x in params["derived_covariance_cross_noise"]} == {"S0", "S1"}
    _check_matrix_with_derived_source(ref3, tmp_path, out, names)
    got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
    keep = v.SNP.isin(got.SNP).to_numpy()
    L_in, V_in = insample_ldscores(S["cohort"][:, keep], sim3["C"], sim3["cls"][keep], 2)
    mass_c = float((got.V.to_numpy()[:, None] * got[["low", "high"]].to_numpy()).sum())
    mass_in = float((V_in[:, None] * L_in).sum())
    assert abs(mass_c / mass_in - 1) < 0.025, (mass_c, mass_in)


def test_curve_corrects_frequency_noise_once(sim, tmp_path):
    """The curve's denominators and global ancestry, recomputed here from the genotypes, the covariates and the
    reference: each SNP's squared contrast loses the frequencies' sampling variance exactly once, and the ancestry
    regression's right-hand side loses the covariance of the contrast's error with the baseline's (both wp/derived
    and wp/localanc corrected the curve; the merge keeps one correction)."""
    from summit.admixed_ld.curve import genotype_curve
    d, S, G, C = sim["dir"], sim["S"], sim["S"]["cohort"], sim["C"]
    v = S["variants"]
    n = 15
    write_bed(str(tmp_path / "sA"), S["panel_A"][:n], v, "sa")
    write_bed(str(tmp_path / "sE"), S["panel_E"][:n], v, "se")
    build_reference([str(tmp_path / "sA"), str(tmp_path / "sE")], ["AFR", "EUR"], str(tmp_path / "ref"),
                    step=256, log=lambda *a: None)
    cell = 0.5
    curve, info = genotype_curve(str(d / "cohort"), str(tmp_path / "ref"), str(tmp_path / "c"),
                                 covar=str(d / "cohort.cov"), cell_cm=cell, cells_cm=0, log=lambda *a: None)
    raw, info_raw = genotype_curve(str(d / "cohort"), str(tmp_path / "ref"), str(tmp_path / "r"),
                                   covar=str(d / "cohort.cov"), cell_cm=cell, cells_cm=0, noise_correction=False,
                                   log=lambda *a: None)
    rv = Reference(str(tmp_path / "ref")).variants(["AFR", "EUR"]).set_index("SNP").loc[v.SNP]
    D = (rv.p_AFR - rv.p_EUR).to_numpy()
    S_ = (rv.v_AFR / (4 * n) + rv.v_EUR / (4 * n)).to_numpy()            # Var(e_A - e_E)
    R_ = -(rv.v_EUR / (4 * n)).to_numpy()                                  # Cov(e_A - e_E, e_E)
    X = G - G.mean(0)
    Q, _ = np.linalg.qr((C - C.mean(0)) / C.std(0, ddof=1))
    Xr = X - Q @ (Q.T @ X)
    var = np.einsum("ij,ij->j", Xr, Xr) / (G.shape[0] - 1 - C.shape[1])
    inv = np.divide(1.0, var, out=np.zeros_like(var), where=var > 1e-8)
    chrom, cm = rv.CHR.to_numpy(), rv.CM.to_numpy()

    def den(times):
        out = {}
        for c in pd.unique(chrom):
            ix = np.flatnonzero(chrom == c)
            b = ((cm[ix] - cm[ix].min()) / cell).astype(int)
            q = np.bincount(b, weights=(D[ix] ** 2 - times * S_[ix]) * inv[ix])
            for lag in (2, 6, 20):
                out[lag] = out.get(lag, 0.0) + float(q[:-lag] @ q[lag:])
        return out

    def theta_mean(times):
        inf = np.abs(D) >= 0.2
        num = D[inf] @ (G.mean(0)[inf] / 2 - rv.p_EUR.to_numpy()[inf]) + times * R_[inf].sum()
        return num / float((D[inf] ** 2 - times * S_[inf]).sum())

    got = curve.set_index("d_cm").denominator
    got_raw = raw.set_index("d_cm").denominator
    once, twice, none = den(1), den(2), den(0)
    for lag in once:
        assert got[lag * cell] == pytest.approx(once[lag], rel=1e-6)
        assert got_raw[lag * cell] == pytest.approx(none[lag], rel=1e-6)
        assert abs(twice[lag] / once[lag] - 1) > 0.05 and abs(none[lag] / once[lag] - 1) > 0.05
    assert info["theta"]["mean"] == pytest.approx(theta_mean(1), rel=1e-6)
    assert info_raw["theta"]["mean"] == pytest.approx(theta_mean(0), rel=1e-6)
    assert abs(theta_mean(2) - theta_mean(1)) > 0.01


@pytest.fixture(scope="module")
def fam(tmp_path_factory):
    """An admixed cohort with sibships and first cousins (IBD segments from simulated meioses), its panels, PCs,
    summary statistics, annotation, reference and listed relative pairs."""
    d = tmp_path_factory.mktemp("admixed_ld_fam")
    S = simulate_families(seed=11, n_sibships=150, sibs=3, n_cousin_families=300, n_unrelated=600)
    v = S["variants"]
    write_bed(str(d / "panelA"), S["panel_A"], v, "a")
    write_bed(str(d / "panelE"), S["panel_E"], v, "e")
    iid = write_bed(str(d / "cohort"), S["cohort"], v, "c")
    C = principal_components(S["cohort"], 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(d / "cohort.cov", sep="\t",
                                                                                   index=False)
    P = S["pairs"]
    pd.DataFrame({"IID1": iid[P[:, 0]], "IID2": iid[P[:, 1]]}).to_csv(d / "pairs.tsv", sep="\t", index=False)
    rng = np.random.default_rng(5)
    os.makedirs(d / "ss")
    for t in range(3):
        y = rng.standard_normal(S["cohort"].shape[0])
        beta, se = gwas(S["cohort"], y, C)
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta,
                      "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(d / "ss" / f"trait{t}.tsv", sep="\t",
                                                                       index=False)
    p = S["cohort"].mean(0) / 2
    cls = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"SNP": v.SNP, "low": (cls == 0).astype(int), "high": (cls == 1).astype(int)}).to_csv(
        d / "annot.tsv", sep="\t", index=False)
    build_reference([str(d / "panelA"), str(d / "panelE")], ["AFR", "EUR"], str(d / "ref"), window_kb=3000,
                    step=256, log=lambda *a: None)
    from summit.admixed_ld.curve import genotype_curve
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "c"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(d / "pairs.tsv"), relatedness_cell_cm=0.5,
                             relatedness_memory_gb=0.0005, threads=3, log=lambda *a: None)   # several passes
    # the adjusted, standardized genotypes of the curve
    G = S["cohort"]
    X = G - G.mean(0)
    Q, _ = np.linalg.qr((C - C.mean(0)) / C.std(0, ddof=1))
    Xr = X - Q @ (Q.T @ X)
    var = np.einsum("ij,ij->j", Xr, Xr) / (G.shape[0] - 1 - C.shape[1])
    ok = var > 1e-8
    U = np.zeros_like(Xr)
    U[:, ok] = Xr[:, ok] / np.sqrt(var[ok])
    return dict(dir=d, S=S, C=C, cls=cls, info=info, U=U, ok=ok)


def test_relatives_profile_sums_are_exact(fam):
    """The curve's lag sums of the listed pairs' local relatedness, their SNP-pair counts and the between-chromosome
    sum equal a direct computation from the adjusted genotypes (with the pairs split over several passes)."""
    U, ok, v = fam["U"], fam["ok"], fam["S"]["variants"]
    P = fam["S"]["pairs"]
    Y = U[P[:, 0]] * U[P[:, 1]]                                   # pairs x SNPs
    tab = pd.read_csv(f"{fam['dir'] / 'c'}.relatedness.tsv", sep="\t").set_index("lag")
    chrom, cm = v.CHR.to_numpy(), v.CM.to_numpy()
    lag_sum, w, dsum = {}, {}, {}
    G = []
    for c in pd.unique(chrom):
        ix = np.flatnonzero((chrom == c) & ok)
        x = cm[ix]
        cell = ((x - cm[chrom == c].min()) / 0.5).astype(int)
        R = np.zeros((P.shape[0], cell.max() + 1))
        np.add.at(R.T, cell, Y[:, ix].T)
        jj, kk = np.triu_indices(ix.size, 1)
        lag_jk = np.abs(cell[jj] - cell[kk])
        for L in range(R.shape[1]):
            if L == 0:
                s_ = ((R ** 2).sum() - (Y[:, ix] ** 2).sum()) / 2
            else:
                s_ = (R[:, :-L] * R[:, L:]).sum()
            lag_sum[L] = lag_sum.get(L, 0.0) + 2 * s_
            sel = lag_jk == L
            w[L] = w.get(L, 0) + int(sel.sum())
            dsum[L] = dsum.get(L, 0.0) + float(np.abs(x[jj] - x[kk])[sel].sum())
        G.append(Y[:, ix].sum(axis=1))
    for L in (0, 1, 2, 7, 40, 150):
        assert tab.loc[L, "listed"] == pytest.approx(lag_sum[L], rel=1e-8)
        assert tab.loc[L, "w"] == w[L]
        assert tab.loc[L, "d_cm"] == pytest.approx(dsum[L] / w[L], rel=1e-5)        # the reference stores cM in float32
    G = np.stack(G)
    r = fam["info"]["relatedness"]
    assert r["between"]["listed"] == pytest.approx(2 * float(((G.sum(0)) ** 2 - (G ** 2).sum(0)).sum()), rel=1e-8)
    assert r["pairs"] == P.shape[0] and r["profile"]["n_controls"] == 20000
    g = Y[:, ok].sum(1) / ok.sum()
    assert r["sum_squared_ordered"] == pytest.approx(2 * np.sum(g ** 2), rel=1e-8)


def test_relatives_profile_term_matches_realized_ld(fam):
    """With relatives sharing IBD segments, the composed relatives' term reproduces their realized excess of in-sample
    LD on one chromosome and between chromosomes (the uniform term does not), and the composed LD mass follows."""
    from summit.admixed_ld.compose import RelatednessProfile, compose_ldscores
    d, S = fam["dir"], fam["S"]
    out = {}
    for arm, rel in (("none", None), ("uniform", "uniform"), ("profile", "curve")):
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(d / f"comp_{arm}"),
                         curve=str(d / "c"), relatedness=rel, log=lambda *a: None)
        out[arm] = pd.read_csv(str(d / f"comp_{arm}") + ".gw.ldscore.gz", sep="\t")
    keep = S["variants"].SNP.isin(out["none"].SNP).to_numpy()
    V = out["none"].V.to_numpy()
    n = S["cohort"].shape[0]
    nu = n - 1 - fam["C"].shape[1]
    term = {k: (out[k][["low", "high"]].to_numpy() - out["none"][["low", "high"]].to_numpy()).sum(1)
            for k in ("uniform", "profile")}
    # realized excess of the listed pairs: sum over ordered pairs of y_j y_k, less what random pairs give (rho^2)
    U = fam["U"][:, keep]
    P = S["pairs"]
    Y = U[P[:, 0]] * U[P[:, 1]]
    rho2 = (U.T @ U / nu) ** 2 - 1.0 / nu
    E = 2 * (Y.T @ Y) - 2 * P.shape[0] * rho2
    np.fill_diagonal(E, 0.0)
    chrom = S["variants"].CHR.to_numpy()[keep]
    same = chrom[:, None] == chrom[None, :]
    real_same, real_other = V @ (E * same) @ V / nu ** 2, V @ (E * ~same) @ V / nu ** 2
    # the composed term split the same way (brute force from the profile; compose convolves on a grid)
    info = json.load(open(f"{d / 'c'}.curve.json"))
    prof = RelatednessProfile.from_curve(str(d / "c"), info)
    cm = S["variants"].CM.to_numpy()[keep]
    Dd = np.abs(cm[:, None] - cm[None, :])
    Ks = np.where(same, prof(Dd), 0.0)
    np.fill_diagonal(Ks, 0.0)
    comp_same = V @ Ks @ V / nu ** 2
    comp_other = prof.between * (V @ (~same) @ V) / nu ** 2
    assert V @ term["profile"] == pytest.approx(comp_same + comp_other, rel=1e-3)
    assert comp_same == pytest.approx(real_same, rel=0.03)
    assert comp_other == pytest.approx(real_other, rel=0.03)
    unif = info["relatedness"]["sum_squared_ordered"] * (V @ (~same) @ V) / nu ** 2
    assert unif / real_other - 1 > 0.05                           # g^2 also counts IBD within chromosomes and noise
    from admixed_ld_sim import insample_ldscores
    L_in, V_in = insample_ldscores(S["cohort"][:, keep], fam["C"], fam["cls"][keep], 2)
    mass_in = float((V_in[:, None] * L_in).sum())
    mass = {k: float((v_.V.to_numpy()[:, None] * v_[["low", "high"]].to_numpy()).sum()) for k, v_ in out.items()}
    assert abs(mass["profile"] / mass_in - 1) < 0.01 < abs(mass["none"] / mass_in - 1)


def test_relatives_profile_absent_pairs_change_nothing(fam, tmp_path):
    """No listed pair in the sample: no profile, and compose with --relatedness curve equals compose without it; the
    uniform term from a curve with a profile is the old sum of g^2."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = fam["dir"]
    pd.DataFrame({"IID1": ["nobody1"], "IID2": ["nobody2"]}).to_csv(tmp_path / "none.tsv", sep="\t", index=False)
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c0"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(tmp_path / "none.tsv"), log=lambda *a: None)
    assert info["relatedness"]["pairs"] == 0 and "profile" not in info["relatedness"]
    assert not os.path.exists(f"{tmp_path / 'c0'}.relatedness.tsv")
    kw = dict(curve=str(tmp_path / "c0"), log=lambda *a: None)
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "a"), **kw)
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "b"), relatedness="curve",
                     **kw)
    A = pd.read_csv(f"{tmp_path / 'a'}.gw.ldscore.gz", sep="\t")
    B = pd.read_csv(f"{tmp_path / 'b'}.gw.ldscore.gz", sep="\t")
    pd.testing.assert_frame_equal(A, B)
    assert "relatedness_no_pairs" in {f["flag"] for f in json.load(open(f"{tmp_path / 'b'}.admixed_ld.json"))["flags"]}
    # the uniform term: the curve's sum of g^2, as a number
    s = fam["info"]["relatedness"]["sum_squared_ordered"]
    for name, rel in (("u1", "uniform"), ("u2", s)):
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / name),
                         curve=str(d / "c"), relatedness=rel, log=lambda *a: None)
    U1 = pd.read_csv(f"{tmp_path / 'u1'}.gw.ldscore.gz", sep="\t")
    U2 = pd.read_csv(f"{tmp_path / 'u2'}.gw.ldscore.gz", sep="\t")
    pd.testing.assert_frame_equal(U1, U2)


def test_background_measures_all_pairs(fam, tmp_path):
    """Without a pair list, the cross-person LD among weakly differentiated SNPs measures the relatedness LD of all
    pairs; here every relative is listed, so it matches the listed pairs' profile beyond the window and between
    chromosomes, and compose uses it from --background-min-cm on."""
    from summit.admixed_ld.compose import BackgroundProfile, RelatednessProfile, compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = fam["dir"]
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "cb"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(d / "pairs.tsv"), relatedness_cell_cm=0.5,
                             background=True, background_quantile=0.5, log=lambda *a: None)
    b = pd.read_csv(f"{tmp_path / 'cb'}.background.tsv", sep="\t")
    r = pd.read_csv(f"{tmp_path / 'cb'}.relatedness.tsv", sep="\t")
    m = b.merge(r[["lag", "phi", "w"]], on="lag", suffixes=("", "_l"))
    s = (m.d_cm >= 3) & (m.d_cm < 150)
    bg = float((m.excess * m.w)[s].sum() / m.w[s].sum())
    li = float((m.phi * m.w_l)[s].sum() / m.w_l[s].sum())
    assert bg == pytest.approx(li, rel=0.10)
    assert info["background"]["between"]["value"] == pytest.approx(info["relatedness"]["between"]["value"], rel=0.05)
    assert info["background"]["null"] == pytest.approx(3 * (fam["S"]["cohort"].shape[0] - 3) /
                                                       fam["S"]["cohort"].shape[0])
    prof = RelatednessProfile.from_curve(str(tmp_path / "cb"), info, background_min_cm=3.0, background_window="zero")
    assert isinstance(prof, BackgroundProfile)
    assert prof.between == info["background"]["between"]["value"]
    assert float(prof(1.0)) == pytest.approx(float(prof.listed(1.0)))
    assert float(prof(20.0)) == pytest.approx(float(prof.listed(20.0) + prof.remainder(20.0)))
    assert (prof.amp >= 0).all()
    # every relative is listed: what the listed pairs do not carry is small
    assert abs(float(prof.remainder(20.0))) < 0.1 * float(prof.listed(20.0))
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "ld"),
                     curve=str(tmp_path / "cb"), relatedness="curve", background_window="zero", log=lambda *a: None)
    params = json.load(open(f"{tmp_path / 'ld'}.admixed_ld.json"))
    assert params["relatedness_profile"]["mode"] == "background"
    with pytest.raises(ValueError, match="clean-panels"):          # the default needs the panels' own background
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "ld2"),
                         curve=str(tmp_path / "cb"), relatedness="curve", log=lambda *a: None)


def _panel_with_sibs(rng, founders, chrom, cm, n_unrelated, n_sibships, copy_rate=3.0):
    """Dosages of an unadmixed panel (one source's founders) with n_sibships pairs of full sibs among its members."""
    import admixed_ld_sim
    hap = admixed_ld_sim._example._haplotypes
    h, _ = hap(rng, founders, cm, 2 * (n_unrelated + 2 * n_sibships), copy_rate)
    people = [h[2 * i:2 * i + 2] for i in range(n_unrelated + 2 * n_sibships)]
    out = [p[0] + p[1] for p in people[:n_unrelated]]
    for s in range(n_sibships):
        P, Q = people[n_unrelated + 2 * s], people[n_unrelated + 2 * s + 1]
        for _ in range(2):
            out.append(admixed_ld_sim._gamete(rng, P[0], P[1], chrom, cm) + admixed_ld_sim._gamete(rng, Q[0], Q[1],
                                                                                                   chrom, cm))
    return np.stack(out).astype(np.float64)


def test_panel_background_removed_inside_the_band(tmp_path):
    """A panel with relatives among its members carries their background LD in its unbiased squared covariances
    (E U = c^2 + v_j v_k S(d) / (n (n - 3))). clean-panels measures it with the cohort's background estimator, and
    compose with the cohort's background at every distance removes it inside the band, so that a panel with relatives
    and one without compose nearly the same LD; with the rule of 5 Nov 2026 (no removal) they do not."""
    from summit.admixed_ld.compose import SmoothBackground, compose_ldscores, unbiased_squared_covariance
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.panel_admixture import apply_panel_admixture, remove_panel_admixture
    import admixed_ld_sim
    rng = np.random.default_rng(23)
    n_chr, spc, chr_cm = 4, 500, 60.0
    chrom = np.repeat(np.arange(1, n_chr + 1), spc)
    cm = np.concatenate([np.sort(rng.uniform(0, chr_cm, spc)) for _ in range(n_chr)])
    M = cm.size
    pE = rng.uniform(0.05, 0.95, M)
    pA = np.clip(pE + rng.normal(0, 0.25, M), 0.03, 0.97)
    founders = {s: (rng.random((40, M)) < p[None, :]).astype(np.int8) for s, p in (("A", pA), ("E", pE))}
    var = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"sim{j:05d}" for j in range(M)], "CM": cm,
                        "BP": np.rint(cm * 1e6).astype(np.int64) + 1, "A1": "A", "A2": "G"})
    n = 600
    panels = {"Arel": _panel_with_sibs(rng, {"A": founders["A"]}, chrom, cm, 300, 150),
              "Aunrel": _panel_with_sibs(rng, {"A": founders["A"]}, chrom, cm, n, 0),
              "E": _panel_with_sibs(rng, {"E": founders["E"]}, chrom, cm, n, 0)}
    for nm, G in panels.items():
        write_bed(str(tmp_path / nm), G, var, nm)
    hap = admixed_ld_sim._example._haplotypes
    nc = 1500
    th = rng.beta(0.75 * 20, 0.25 * 20, nc)
    hC, _ = hap(rng, founders, cm, 2 * nc, 3.0, anc=np.repeat(th, 2), break_rate=0.08)
    GC = (hC[0::2] + hC[1::2]).astype(np.float64)
    iid = write_bed(str(tmp_path / "cohort"), GC, var, "c")
    C = principal_components(GC, 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(tmp_path / "cohort.cov", sep="\t",
                                                                                   index=False)
    os.makedirs(tmp_path / "ss")
    for t in range(2):
        beta, se = gwas(GC, rng.standard_normal(nc), C)
        pd.DataFrame({"SNP": var.SNP, "CHR": var.CHR, "BP": var.BP, "A1": var.A1, "A2": var.A2, "N": nc,
                      "BETA": beta, "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(tmp_path / "ss" / f"t{t}.tsv",
                                                                                     sep="\t", index=False)
    pc = GC.mean(0) / 2
    lo = (np.minimum(pc, 1 - pc) < 0.2).astype(int)
    pd.DataFrame({"SNP": var.SNP, "low": lo, "high": 1 - lo}).to_csv(tmp_path / "annot.tsv", sep="\t", index=False)
    quiet = lambda *a, **k: None
    for r in ("rel", "unrel"):
        build_reference([str(tmp_path / f"A{r}"), str(tmp_path / "E")], ["AFR", "EUR"], str(tmp_path / f"ref_{r}"),
                        match="id", log=quiet)
        remove_panel_admixture(str(tmp_path / f"ref_{r}"), {"AFR": str(tmp_path / f"A{r}"), "EUR": str(tmp_path / "E")},
                               out=str(tmp_path / f"ref_{r}_c"), match="id", background_snps=2000,
                               background_quantile=1.0, log=quiet)
    man = json.load(open(tmp_path / "ref_rel_c" / "manifest.json"))
    rec = man["sources"][0]["panel_background"]
    assert rec["n"] == n and set(rec["table"]) == {"lag", "d_cm", "w", "excess"}
    # the unbiased squared covariances of the panel with sibs hold its background: band pairs at 1-3 cM, where the
    # founder-copying model leaves no within-source LD, against the panel's measured background per pair
    ref = Reference(str(tmp_path / "ref_rel"))
    num, cnt = 0.0, 0
    for c in ref.chromosomes:
        z = ref.load(c)
        jl = np.repeat(np.arange(z["bp"].size), np.diff(z["indptr"]))
        kl = z["indices"].astype(np.int64)
        dd = np.abs(z["cm"][kl] - z["cm"][jl])
        sel = (dd >= 1.0) & (dd < 3.0) & (z["v_0"][jl] > 0.05) & (z["v_0"][kl] > 0.05)
        vv = z["v_0"][jl][sel].astype(np.float64) * z["v_0"][kl][sel]
        U = unbiased_squared_covariance(z["cov_0"][sel].astype(np.float64), z["m22_0"][sel].astype(np.float64), vv, n)
        num += float((U / vv).sum())
        cnt += int(sel.sum())
    sm = SmoothBackground.from_record(rec, fit_min_cm=1.5)
    pooled = float(np.mean([sm.per_pair(x) for x in np.linspace(1.0, 3.0, 9)]))
    assert num / cnt == pytest.approx(pooled, rel=0.15)
    assert num / cnt > 1e-4                                       # sibs: about (1/4 + 1/8) x 300 / (n (n - 3))
    # --from-fits carries the record
    apply_panel_admixture(str(tmp_path / "ref_rel"), str(tmp_path / "ref_rel_c"), out=str(tmp_path / "ref_rel_f"),
                          log=quiet)
    assert json.load(open(tmp_path / "ref_rel_f" / "manifest.json"))["sources"][0]["panel_background"]["table"] == \
        rec["table"]
    # compose: the panel's relatives' LD is removed inside the band
    genotype_curve(str(tmp_path / "cohort"), str(tmp_path / "ref_rel"), str(tmp_path / "cv"),
                   covar=str(tmp_path / "cohort.cov"), match="id", background=True, background_quantile=0.5, log=quiet)
    mass = {}
    for r in ("rel", "unrel"):
        for win in ("panels", "zero"):
            o = str(tmp_path / f"ld_{r}_{win}")
            compose_ldscores(str(tmp_path / f"ref_{r}_c"), [str(tmp_path / "ss")], str(tmp_path / "annot.tsv"), o,
                             match="id", curve=str(tmp_path / "cv"), relatedness="curve", background_window=win,
                             background_min_cm=1.5, log=quiet)
            L = pd.read_csv(f"{o}.gw.ldscore.gz", sep="\t")
            mass[r, win] = float((L.V.to_numpy()[:, None] * L[["low", "high"]].to_numpy()).sum())
    gap_zero = mass["rel", "zero"] - mass["unrel", "zero"]
    gap_panels = mass["rel", "panels"] - mass["unrel", "panels"]
    assert gap_zero > 0.001 * mass["unrel", "zero"]                # the panel's sibs inflate the band
    # most of it is removed; what stays is noise and, within a few tenths of a cM, the sibs' sharing of haplotypes in
    # strong LD (the fourth moments of the shared haplotype), which the smooth form fitted beyond 1.5 cM leaves out
    # (here half of the panel's members are sibs; in a biobank panel it is a small part of the band's LD)
    assert abs(gap_panels) < 0.4 * gap_zero
    params = json.load(open(str(tmp_path / "ld_rel_panels") + ".admixed_ld.json"))
    assert set(params["panel_background"]) == {"AFR", "EUR"} and params["relatedness_profile"]["window"] == "panels"


def test_derived_source_anchor_background_removed_inside_the_band(tmp_path):
    """A derived source with a panel's window LD (derive --ld panel) holds its anchors' background in the panel's
    unbiased squared covariances, like any panel. clean-panels given the anchor panel records it, and compose removes it
    inside the band, so anchors with sibs and anchors without compose nearly the same LD; without the record compose
    refuses the cohort's background at every distance (it would count the anchors' background twice)."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.derive import derive_source
    from summit.admixed_ld.panel_admixture import remove_panel_admixture
    import admixed_ld_sim
    rng = np.random.default_rng(29)
    n_chr, spc, chr_cm = 4, 500, 60.0
    chrom = np.repeat(np.arange(1, n_chr + 1), spc)
    cm = np.concatenate([np.sort(rng.uniform(0, chr_cm, spc)) for _ in range(n_chr)])
    M = cm.size
    pE = rng.uniform(0.05, 0.95, M)
    pA = np.clip(pE + rng.normal(0, 0.25, M), 0.03, 0.97)
    founders = {s: (rng.random((40, M)) < p[None, :]).astype(np.int8) for s, p in (("A", pA), ("E", pE))}
    var = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"sim{j:05d}" for j in range(M)], "CM": cm,
                        "BP": np.rint(cm * 1e6).astype(np.int64) + 1, "A1": "A", "A2": "G"})
    n = 600
    panels = {"Arel": _panel_with_sibs(rng, {"A": founders["A"]}, chrom, cm, 300, 150),
              "Aunrel": _panel_with_sibs(rng, {"A": founders["A"]}, chrom, cm, n, 0),
              "E": _panel_with_sibs(rng, {"E": founders["E"]}, chrom, cm, n, 0)}
    for nm, G in panels.items():
        write_bed(str(tmp_path / nm), G, var, nm)
    hap = admixed_ld_sim._example._haplotypes
    nc = 1500
    th = rng.beta(0.75 * 20, 0.25 * 20, nc)
    hC, _ = hap(rng, founders, cm, 2 * nc, 3.0, anc=np.repeat(th, 2), break_rate=0.08)
    GC = (hC[0::2] + hC[1::2]).astype(np.float64)
    iid = write_bed(str(tmp_path / "cohort"), GC, var, "c")
    C = principal_components(GC, 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(tmp_path / "cohort.cov", sep="\t",
                                                                                   index=False)
    os.makedirs(tmp_path / "ss")
    for t in range(2):
        beta, se = gwas(GC, rng.standard_normal(nc), C)
        pd.DataFrame({"SNP": var.SNP, "CHR": var.CHR, "BP": var.BP, "A1": var.A1, "A2": var.A2, "N": nc,
                      "BETA": beta, "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(tmp_path / "ss" / f"t{t}.tsv",
                                                                                     sep="\t", index=False)
    pc = GC.mean(0) / 2
    lo = (np.minimum(pc, 1 - pc) < 0.2).astype(int)
    pd.DataFrame({"SNP": var.SNP, "low": lo, "high": 1 - lo}).to_csv(tmp_path / "annot.tsv", sep="\t", index=False)
    quiet = lambda *a, **k: None
    build_reference([str(tmp_path / "Aunrel"), str(tmp_path / "E")], ["AFR", "EUR"], str(tmp_path / "ref"),
                    match="id", log=quiet)
    cov = str(tmp_path / "cohort.cov")
    for r in ("rel", "unrel"):
        derive_source(str(tmp_path / "cohort"), str(tmp_path / "ref"), str(tmp_path / f"refd_{r}"), name="AFRd",
                      known=["EUR"], panel=str(tmp_path / f"A{r}"), covar=cov, match="id", ld="panel", step=256,
                      log=quiet)
    # the European panel cleaned, the anchors not: compose refuses
    remove_panel_admixture(str(tmp_path / "refd_rel"), {"EUR": str(tmp_path / "E")}, out=str(tmp_path / "refd_rel_e"),
                           match="id", background_snps=2000, background_quantile=1.0, log=quiet)
    genotype_curve(str(tmp_path / "cohort"), str(tmp_path / "refd_rel_e"), str(tmp_path / "cv_e"), covar=cov,
                   match="id", background=True, background_quantile=0.5, log=quiet)
    with pytest.raises(ValueError, match="AFRd"):
        compose_ldscores(str(tmp_path / "refd_rel_e"), [str(tmp_path / "ss")], str(tmp_path / "annot.tsv"),
                         str(tmp_path / "ld_e"), match="id", curve=str(tmp_path / "cv_e"), relatedness="curve",
                         background_window="panels", background_min_cm=1.5, log=quiet)
    mass = {}
    for r in ("rel", "unrel"):
        rc = str(tmp_path / f"refd_{r}_c")
        remove_panel_admixture(str(tmp_path / f"refd_{r}"), {"AFRd": str(tmp_path / f"A{r}"), "EUR": str(tmp_path / "E")},
                               out=rc, match="id", background_snps=2000, background_quantile=1.0, log=quiet)
        man = json.load(open(os.path.join(rc, "manifest.json")))
        rec = {s["name"]: s for s in man["sources"]}["AFRd"]["panel_background"]
        assert rec["n"] == n and set(rec["table"]) == {"lag", "d_cm", "w", "excess"}
        genotype_curve(str(tmp_path / "cohort"), rc, str(tmp_path / f"cv_{r}"), covar=cov, match="id",
                       background=True, background_quantile=0.5, log=quiet)
        for win in ("panels", "zero"):
            o = str(tmp_path / f"ld_{r}_{win}")
            compose_ldscores(rc, [str(tmp_path / "ss")], str(tmp_path / "annot.tsv"), o, match="id",
                             curve=str(tmp_path / f"cv_{r}"), relatedness="curve", background_window=win,
                             background_min_cm=1.5, log=quiet)
            L = pd.read_csv(f"{o}.gw.ldscore.gz", sep="\t")
            mass[r, win] = float((L.V.to_numpy()[:, None] * L[["low", "high"]].to_numpy()).sum())
    gap_zero = mass["rel", "zero"] - mass["unrel", "zero"]
    gap_panels = mass["rel", "panels"] - mass["unrel", "panels"]
    assert gap_zero > 0.001 * mass["unrel", "zero"], mass          # the anchors' sibs inflate the band
    assert abs(gap_panels) < 0.4 * gap_zero, mass
    params = json.load(open(str(tmp_path / "ld_rel_panels") + ".admixed_ld.json"))
    assert set(params["panel_background"]) == {"AFRd", "EUR"}


class _FakeRef:
    """The parts of Reference that panel_variance_scale reads: chromosomes and load()."""

    def __init__(self, arrays):
        self.chromosomes = ["1"]
        self._z = arrays

    def load(self, c):
        class Z(dict):
            files = property(lambda self: list(self.keys()))
        return Z(self._z)


def test_panel_variance_scale_limits():
    """A small panel drawn from the source keeps its covariances (f ~ 1); a large panel of another population has its
    correlations transferred (f ~ sqrt(v / vld))."""
    from summit.admixed_ld.compose import panel_variance_scale
    rng = np.random.default_rng(7)
    m = 40_000
    p = rng.uniform(0.03, 0.97, m)
    v = 2 * p * (1 - p)
    fvar = np.full(m, 1e-6)
    # panel of 34 drawn from the source: sample variances of 34 dosages
    n = 34
    X = rng.binomial(2, p[None, :], size=(n, m)).astype(float)
    vld = X.var(axis=0, ddof=1)
    ok = vld > 0
    z = {"v_0": v[ok], "vld_0": vld[ok], "p_0": p[ok], "fvar_0": fvar[ok]}
    f, info = panel_variance_scale(_FakeRef(z), 0, n)
    assert info["shrinkage_weight_median"] < 0.15, info
    assert abs(np.mean(f["1"] ** 2) - 1) < 0.03, info
    # panel of 2,000 from another population (frequencies drifted, F_ST about 0.02)
    n = 2000
    k = m // 4
    q = np.clip(p[:k] + rng.normal(0, np.sqrt(0.02 * p[:k] * (1 - p[:k]))), 0.005, 0.995)
    X = rng.binomial(2, q[None, :], size=(n, k)).astype(float)
    vld = X.var(axis=0, ddof=1)
    z = {"v_0": v[:k], "vld_0": vld, "p_0": p[:k], "fvar_0": fvar[:k]}
    f, info = panel_variance_scale(_FakeRef(z), 0, n)
    assert info["shrinkage_weight_median"] > 0.9, info
    np.testing.assert_allclose(f["1"], np.sqrt(v[:k] / vld), rtol=0.1)            # an estimate, close to it


def test_matrix_with_derived_source_from_another_population_is_psd(sim3, tmp_path):
    """A derived source whose panel's variances differ from its own (a panel of another population): the regional
    matrix transfers the panel's correlations and stays positive semidefinite."""
    from summit.admixed_ld.derive import derive_source
    from summit.admixed_ld.matrix import compose_matrix
    d, S, names = sim3["dir"], sim3["S"], sim3["names"]
    v = S["variants"]
    write_bed(str(tmp_path / "other"), S["panels"][0][:200], v, "o")       # anchors from source S0, not S2
    build_reference([str(d / "panelS0"), str(d / "panelS1")], names[:2], str(tmp_path / "ref2"), step=256,
                    log=lambda *a: None)
    derive_source(str(d / "cohort"), str(tmp_path / "ref2"), str(tmp_path / "ref3"), name="S2",
                  panel=str(tmp_path / "other"), covar=str(d / "cohort.cov"), ld="panel", step=256, log=lambda *a: None)
    ref3 = Reference(str(tmp_path / "ref3"))
    z = ref3.load(ref3.chromosomes[0])
    assert np.median(np.abs(np.log(z["vld_2"] / z["v_2"]))[z["v_2"] > 0.05]) > 0.02     # the scales differ
    ids = tmp_path / "region.txt"
    pd.Series(z["snp"][:120]).to_csv(ids, index=False, header=False)
    R, snp, info = compose_matrix(ref3.path, str(tmp_path / "mat"), snps=str(ids), sources=names,
                                  mean_ancestry="0.5,0.3,0.2", kappa0="0.1,0.0,0.1", decay_per_cm=0.1,
                                  taper="none", log=lambda *a: None)
    assert info["min_eigenvalue"] > -1e-8, info["min_eigenvalue"]
    assert np.all(np.abs(R) <= 1 + 1e-9)
    # with nearly all ancestry from S2 and no admixture term, the matrix is the anchor panel's correlation matrix
    R1, snp1, _ = compose_matrix(ref3.path, str(tmp_path / "mat1"), snps=str(ids), sources=names,
                                 mean_ancestry="0.0005,0.0005,0.999", kappa0="0.1,0.0,0.1", decay_per_cm=1e-6,
                                 taper="none", admixture=False, log=lambda *a: None)
    col = pd.Index(v.SNP.astype(str)).get_indexer(snp1.SNP.astype(str))
    Xp = S["panels"][0][:200][:, col].astype(float)
    poly = np.flatnonzero(Xp.var(axis=0) > 0)                     # SNPs the panel has variation at
    Rp = np.corrcoef(Xp[:, poly], rowvar=False)
    iu = np.triu_indices(poly.size, 1)
    bp = snp1.BP.to_numpy()[poly]
    inside = np.abs(bp[iu[0]] - bp[iu[1]]) < ref3.manifest["window_kb"] * 1000 * 0.95   # pairs the reference stores
    err = np.max(np.abs(R1[np.ix_(poly, poly)][iu] - Rp[iu])[inside])
    assert poly.size > 60 and inside.sum() > 1000 and err < 5e-3, (poly.size, inside.sum(), err)


def test_compose_fails_closed_and_records_flags(sim, tmp_path):
    """Inputs that disagree raise; allowed departures from the standard recipe are recorded in the params json (7 Nov
    2026). A curve above kappa_res(0) at 2 cM is an error unless allowed (then flagged); a date without a published
    ancestry SD flags the default SD; a panel whose own admixture was never measured is flagged."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    quiet = lambda *a, **k: None
    if not os.path.exists(f"{d / 'cohort'}.curve.json"):
        genotype_curve(str(d / "cohort"), str(d / "ref"), str(d / "cohort"), covar=str(d / "cohort.cov"),
                       cell_cm=0.5, log=quiet)
    cdf = pd.read_csv(f"{d / 'cohort'}.curve.tsv", sep="\t")
    k2 = float(np.interp(2.0, cdf.d_cm.to_numpy(), cdf.kernel.to_numpy()))
    assert k2 > 0
    args = (str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"))

    def flags(out):
        return {f["flag"] for f in json.load(open(out + ".admixed_ld.json"))["flags"]}

    out = str(tmp_path / "curve")
    compose_ldscores(*args, out, curve=str(d / "cohort"), log=quiet)
    assert flags(out) == {"panel_admixture_unmeasured"}            # the test reference was never cleaned
    low = dict(curve=str(d / "cohort"), mean_ancestry="0.6", kappa0=f"{0.5 * k2:.6g}", log=quiet)
    with pytest.raises(ValueError, match="allow-flat-kernel-join"):
        compose_ldscores(*args, str(tmp_path / "flat"), **low)
    assert not os.path.exists(str(tmp_path / "flat") + ".gw.ldscore.gz")
    compose_ldscores(*args, str(tmp_path / "flat"), allow_flat_kernel_join=True, **low)
    assert flags(str(tmp_path / "flat")) == {"kernel_flat_join", "panel_admixture_unmeasured"}
    out = str(tmp_path / "date")
    compose_ldscores(*args, out, generations=sim["S"]["generations"], log=quiet)
    assert flags(out) == {"ancestry_sd_default", "panel_admixture_unmeasured"}
    compose_ldscores(*args, out + "_sd", generations=sim["S"]["generations"], theta_sd=0.1, log=quiet)
    assert flags(out + "_sd") == {"panel_admixture_unmeasured"}
