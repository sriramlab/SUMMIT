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
