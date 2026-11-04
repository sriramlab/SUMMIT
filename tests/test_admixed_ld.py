"""Tests of `summit admixed-ld`: the reference resource, variant matching, the standard-error regression, the genotype
curve, and composed LD scores against exact in-sample LD scores of a simulated admixed cohort."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from admixed_ld_sim import gwas, insample_ldscores, principal_components, simulate, write_bed  # noqa: E402

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
