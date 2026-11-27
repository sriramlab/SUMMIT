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
from summit.admixed_ld.model import KernelMatrix, fit_exponential, regress_standard_errors_k  # noqa: E402
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
    d = (rv.p_AFR - rv.p_EUR).to_numpy()
    r = regress_standard_errors_k(ss.N, ss.SE, ss.BETA, np.column_stack([rv.v_AFR, rv.v_EUR]), (d * d)[:, None, None])
    r = dict(m=r["m"][0], kappa0=r["kappa0"][0][0])
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
    """The curve's relatedness of listed pairs equals the GRM of the covariate-adjusted, standardized genotypes (with
    the 20 or more listed pairs over which the curve records summaries)."""
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    G, C = sim["S"]["cohort"], sim["C"]
    iid = pd.read_csv(f"{d / 'cohort'}.fam", sep=r"\s+", header=None, dtype=str)[1].to_numpy()
    pairs = [(0, 1), (2, 7), (10, 11), (5, 40)] + [(100 + 2 * i, 101 + 2 * i) for i in range(20)] + [(100, 3)]
    kin = [0.5, 0.2, 0.05, 0.3] + [0.1] * 20 + [0.01]
    pd.DataFrame({"IID1": [iid[a] for a, _ in pairs], "IID2": [iid[b] for _, b in pairs],
                  "K": kin}).to_csv(tmp_path / "pairs.tsv", sep="\t", index=False)
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
    g = np.array([Z[a] @ Z[b] / ok.sum() for a, b in pairs[:-1]])    # the last pair is below --kinship-min
    r = info["relatedness"]
    assert r["pairs"] == len(pairs) - 1 and r["n_snps"] == ok.sum()
    assert r["sum_squared_ordered"] == pytest.approx(2 * np.sum(g ** 2), rel=1e-8)
    # --closure adds the pairs implied within a family: (0, 1) and (1, 2) give (0, 2)
    chain = [(0, 1), (1, 2)] + [(100 + 2 * i, 101 + 2 * i) for i in range(20)]
    pd.DataFrame({"IID1": [iid[a] for a, _ in chain], "IID2": [iid[b] for _, b in chain]}).to_csv(
        tmp_path / "chain.tsv", sep="\t", index=False)
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c2"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(tmp_path / "chain.tsv"), closure=True,
                             log=lambda *a: None)
    g = np.array([Z[a] @ Z[b] / ok.sum() for a, b in [(0, 1), (0, 2), (1, 2)] + chain[2:]])
    assert info["relatedness"]["pairs"] == 23
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
                 "--cell-cm", "0.5", "--no-background", "--out", str(tmp_path / "c")]) == 0
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
    k = KernelMatrix.from_decay(0.2, 0.08, 3500.0, 2, source="t")
    assert k(0.0)[0, 0] == pytest.approx(0.2)
    assert k.far[0, 0] == pytest.approx(-2 * 0.2 / (0.08 * 3500))
    assert k(1e6)[0, 0] == pytest.approx(k.far[0, 0])
    d = np.arange(0.5, 40.5, 0.5)
    curve = 0.18 * np.exp(-0.07 * d) - 0.001
    assert fit_exponential(d, curve)["lam"] == pytest.approx(0.07, rel=0.05)
    kc = KernelMatrix.from_curve(d, curve[:, None, None], [[-0.001]], [[0.2]], [0.07, 0.07], half_width=0,
                                 rel_width=0)
    assert kc(0.0)[0, 0] == pytest.approx(0.2)
    assert kc(1.0)[0, 0] == pytest.approx(0.2 * (np.interp(2.0, d, curve) / 0.2) ** 0.5, rel=1e-9)   # the join
    assert kc(10.0)[0, 0] == pytest.approx(0.18 * np.exp(-0.7) - 0.001, rel=1e-6)
    assert kc(100.0)[0, 0] == pytest.approx(-0.001)


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
    from summit.admixed_ld.model import contrast_noise, unbiased_products_cov

    def unbiased_products(D, var_c, var_base):                 # the two-source-panel formula written out
        O = D[:, :, None] * D[:, None, :] - np.asarray(var_base, np.float64)[:, None, None]
        idx = np.arange(D.shape[1])
        O[:, idx, idx] -= var_c
        return O
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


def _copy_curve(src, dst, **changes):
    """A copy of the curve files at prefix src under prefix dst, with keys of the curve's json replaced."""
    import glob
    import shutil
    for f in glob.glob(f"{src}.*"):
        shutil.copy(f, f"{dst}{f[len(str(src)):]}")
    info = json.load(open(f"{src}.curve.json"))
    info.update(changes)
    json.dump(info, open(f"{dst}.curve.json", "w"))


def test_relatives_term_is_put_on_the_gwas_sample(fam, tmp_path):
    """The curve's relatives' sums run over the ordered pairs of its own people: compose multiplies the relatives'
    term (profile and uniform) by N (N - 1) / (n (n - 1)), exactly 1 (bit-identical) when the curve and the GWAS have
    the same people, records the factor, and flags a factor away from 1. Nothing else in the composition changes, and
    a number given with --relatedness is used as given."""
    from summit.admixed_ld.compose import compose_ldscores, relatedness_scale
    d = fam["dir"]
    n = int(fam["info"]["n_samples"])
    N = fam["S"]["cohort"].shape[0]
    assert n == N
    _copy_curve(d / "c", tmp_path / "c2", n_samples=2 * n)        # the same sums said to be over twice the people
    f = N * (N - 1.0) / (2 * n * (2 * n - 1.0))
    assert relatedness_scale(2 * n, N) == f and relatedness_scale(n, N) == 1.0
    out, par = {}, {}
    # the copy's json says 2n people and its statistic has n, so compose would stop on the statistic (a file of other
    # people); the same-person term is random mating in every arm here, the relatives' term being the subject
    for c in ("c", "c2"):
        cpref = str(d / "c") if c == "c" else str(tmp_path / "c2")
        for arm, rel in (("none", None), ("profile", "curve"), ("uniform", "uniform"), ("number", 123.0)):
            o = str(tmp_path / f"{c}_{arm}")
            compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), o, curve=cpref, relatedness=rel,
                             release_sample_factors=False, log=lambda *a: None)
            out[c, arm] = pd.read_csv(o + ".gw.ldscore.gz", sep="\t")
            par[c, arm] = json.load(open(o + ".admixed_ld.json"))
    cols = ["low", "high"]
    pd.testing.assert_frame_equal(out["c", "none"], out["c2", "none"])
    pd.testing.assert_frame_equal(out["c", "number"], out["c2", "number"])
    for arm in ("profile", "uniform"):
        t1 = out["c", arm][cols].to_numpy() - out["c", "none"][cols].to_numpy()
        t2 = out["c2", arm][cols].to_numpy() - out["c2", "none"][cols].to_numpy()
        # exact up to the writer's rounding (%.6f: each value within 5e-7, so a difference within 1e-6 and f times
        # one within (1 + f) 1e-6); the unscaled term of before is off by (1 - f) t1, over a thousand times that
        np.testing.assert_allclose(t2, f * t1, rtol=0, atol=(1 + f) * 1e-6 + 1e-12)
        assert np.abs((1 - f) * t1).max() > 1000 * (1 + f) * 1e-6
        assert par["c", arm]["relatedness_scale"]["factor"] == 1.0
        assert par["c", arm]["relatedness_scale"]["curve_people"] == n
        assert par["c2", arm]["relatedness_scale"]["factor"] == pytest.approx(f, rel=1e-12)
        assert "relatedness_curve_sample_differs" not in {x["flag"] for x in par["c", arm]["flags"]}
        assert "relatedness_curve_sample_differs" in {x["flag"] for x in par["c2", arm]["flags"]}
    for c in ("c", "c2"):
        assert par[c, "none"]["relatedness_scale"] is None and par[c, "number"]["relatedness_scale"] is None
    # the matched case is the composition of before the scale (f = 1 exactly): the profile term equals the brute-force
    # (1 / nu^2) sums of test_relatives_profile_term_matches_realized_ld, which that test checks; here, bit-identity of
    # the curve and its copy with the same n
    _copy_curve(d / "c", tmp_path / "c1")
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "c1_profile"),
                     curve=str(tmp_path / "c1"), relatedness="curve", release_sample_factors=False, log=lambda *a: None)
    with pytest.raises(ValueError, match=f"n_samples {n} in the statistic, {2 * n} in the curve"):
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "c2_sf"),
                         curve=str(tmp_path / "c2"), dry_run=True, log=lambda *a: None)
    pd.testing.assert_frame_equal(out["c", "profile"], pd.read_csv(f"{tmp_path / 'c1_profile'}.gw.ldscore.gz", sep="\t"))
    with pytest.raises(ValueError, match="at least 2 people"):
        relatedness_scale(1, N)


def test_relatives_term_of_a_gwas_on_part_of_the_curve(fam, tmp_path):
    """A GWAS of a random 60% of the curve's people: with the whole cohort's curve, the relatives' term matches the
    realized excess of in-sample LD that the listed pairs inside the GWAS create, and the term composed with a curve
    made on the GWAS's own people; without the scale it would be about 1 / 0.36 times too large."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d, S, C = fam["dir"], fam["S"], fam["C"]
    G = S["cohort"]
    n = G.shape[0]
    rng = np.random.default_rng(3)
    sub = np.sort(rng.choice(n, int(0.6 * n), replace=False))
    N = sub.size
    iid = np.array([f"c{i:06d}" for i in range(n)])
    pd.DataFrame({"FID": iid[sub], "IID": iid[sub]}).to_csv(tmp_path / "sub.keep", sep="\t", header=False,
                                                            index=False)
    v = S["variants"]
    os.makedirs(tmp_path / "ss_sub")
    for t in range(3):
        y = rng.standard_normal(N)
        beta, se = gwas(G[sub], y, C[sub])
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": N, "BETA": beta,
                      "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(tmp_path / "ss_sub" / f"trait{t}.tsv",
                                                                       sep="\t", index=False)
    _, info_s = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "cs"), covar=str(d / "cohort.cov"),
                               keep=str(tmp_path / "sub.keep"), cell_cm=0.5, relative_pairs=str(d / "pairs.tsv"),
                               relatedness_cell_cm=0.5, log=lambda *a: None)
    assert info_s["n_samples"] == N
    out, par = {}, {}
    for arm, cpref, rel in (("none", d / "c", None), ("whole", d / "c", "curve"), ("own", tmp_path / "cs", "curve")):
        o = str(tmp_path / f"sub_{arm}")
        compose_ldscores(str(d / "ref"), [str(tmp_path / "ss_sub")], str(d / "annot.tsv"), o, curve=str(cpref),
                         relatedness=rel, log=lambda *a: None)
        out[arm] = pd.read_csv(o + ".gw.ldscore.gz", sep="\t")
        par[arm] = json.load(open(o + ".admixed_ld.json"))
    f = N * (N - 1.0) / (n * (n - 1.0))
    assert par["whole"]["relatedness_scale"]["factor"] == pytest.approx(f, rel=1e-12)
    assert par["own"]["relatedness_scale"]["factor"] == 1.0
    keep = v.SNP.isin(out["none"].SNP).to_numpy()
    V = out["none"].V.to_numpy()
    nu = N - 1 - C.shape[1]
    mass = lambda X: float(V @ X[["low", "high"]].to_numpy().sum(1))
    term = {a: mass(out[a]) - mass(out["none"]) for a in ("whole", "own")}
    # realized excess of the listed pairs inside the GWAS, in its own adjusted, standardized genotypes
    Gs = G[sub][:, keep]
    X = Gs - Gs.mean(0)
    Q, _ = np.linalg.qr((C[sub] - C[sub].mean(0)) / C[sub].std(0, ddof=1))
    Xr = X - Q @ (Q.T @ X)
    var = np.einsum("ij,ij->j", Xr, Xr) / nu
    U = np.where(var > 1e-8, Xr / np.sqrt(np.maximum(var, 1e-300)), 0.0)
    pos = np.full(n, -1)
    pos[sub] = np.arange(N)
    P = pos[S["pairs"]]
    P = P[(P >= 0).all(1)]
    Y = U[P[:, 0]] * U[P[:, 1]]
    rho2 = (U.T @ U / nu) ** 2 - 1.0 / nu
    E = 2 * (Y.T @ Y) - 2 * P.shape[0] * rho2
    np.fill_diagonal(E, 0.0)
    real = float(V @ E @ V) / nu ** 2
    share = P.shape[0] / S["pairs"].shape[0]                      # the listed pairs the GWAS holds (expected f)
    assert abs(share / f - 1) < 0.25
    # the whole cohort's curve, scaled, against the realized term (corrected for the random share of pairs held)
    assert term["whole"] / (real * f / share) == pytest.approx(1.0, abs=0.05)
    assert term["own"] / real == pytest.approx(1.0, abs=0.05)
    assert term["whole"] / term["own"] == pytest.approx(f / share, rel=0.05)
    assert (term["whole"] / f) / real > 2.0                       # the unscaled term of before
    from admixed_ld_sim import insample_ldscores
    L_in, V_in = insample_ldscores(Gs, C[sub], fam["cls"][keep], 2)
    mass_in = float((V_in[:, None] * L_in).sum())
    print(f"\nsubset {N} of {n}; f {f:.4f}, pairs held {share:.4f}; term whole {term['whole']:.4g}, own "
          f"{term['own']:.4g}, realized {real:.4g}; mass none / whole / own / in-sample {mass(out['none']):.5g} / "
          f"{mass(out['whole']):.5g} / {mass(out['own']):.5g} / {mass_in:.5g}; unscaled whole "
          f"{mass(out['none']) + term['whole'] / f:.5g}")
    assert abs(mass(out["whole"]) / mass_in - 1) < 0.015 < 0.1 < (mass(out["none"]) + term["whole"] / f) / mass_in - 1


def test_background_measures_all_pairs(fam, tmp_path):
    """Without a pair list, the cross-person LD among weakly differentiated SNPs measures the relatedness LD of all
    pairs; here every relative is listed, so it matches the listed pairs' profile beyond the window and between
    chromosomes, and compose uses it from --background-min-cm on."""
    from summit.admixed_ld.compose import BackgroundProfile, RelatednessProfile, compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = fam["dir"]
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "cb"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(d / "pairs.tsv"), relatedness_cell_cm=0.5,
                             background=True, background_quantile=0.5, relatedness_profile=True,
                             log=lambda *a: None)
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
    big = z["v_2"] > 0.05                            # SNPs without variance in the source or its anchors have no ratio
    assert np.median(np.abs(np.log(z["vld_2"][big] / z["v_2"][big]))) > 0.02     # the scales differ
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
    # the flags of the inputs themselves: without the curve's sample statistic (tested on its own below)
    compose_ldscores(*args, out, curve=str(d / "cohort"), release_sample_factors=False, log=quiet)
    assert flags(out) == {"panel_admixture_unmeasured"}            # the test reference was never cleaned
    low = dict(curve=str(d / "cohort"), mean_ancestry="0.6", kappa0=f"{0.5 * k2:.6g}", release_sample_factors=False,
               log=quiet)
    with pytest.raises(ValueError, match="allow-flat-kernel-join"):
        compose_ldscores(*args, str(tmp_path / "flat"), **low)
    assert not os.path.exists(str(tmp_path / "flat") + ".gw.ldscore.gz")
    compose_ldscores(*args, str(tmp_path / "flat"), allow_flat_kernel_join=True, **low)
    assert flags(str(tmp_path / "flat")) == {"kernel_flat_join", "panel_admixture_unmeasured"}
    out = str(tmp_path / "date")
    compose_ldscores(*args, out, generations=sim["S"]["generations"], log=quiet)
    assert flags(out) == {"ancestry_sd_from_identity", "panel_admixture_unmeasured"}
    p = json.load(open(out + ".admixed_ld.json"))
    sd_id = np.sqrt((2 * p["mean_ancestry"] * (1 - p["mean_ancestry"]) - p["kappa0"]) / 4)
    assert p["theta"]["var"] == pytest.approx(sd_id ** 2, rel=1e-6)
    compose_ldscores(*args, out + "_sd", generations=sim["S"]["generations"], theta_sd=0.1, log=quiet)
    assert flags(out + "_sd") == {"panel_admixture_unmeasured"}


# ------------------------------------------------------------------------------------------------- sample factors
@pytest.mark.parametrize("k", [2, 3])
def test_factor_same_person_term_is_the_random_mating_term_under_random_mating(k):
    """With the factors random mating implies (level 1, theta, Lambda = 2 (diag theta - theta theta')), the factor
    same-person term equals compose's random-mating term exactly, for two and three sources."""
    from summit.admixed_ld.model import moments_from_vectors, moments_tensors, same_person_term
    from summit.admixed_ld.sample_factors import factor_loadings, factor_names, factor_same_person_term
    rng = np.random.default_rng(4 + k)
    M, K, k1 = 400, 3, k - 1
    P = rng.uniform(0.05, 0.95, (M, k))                              # baseline last
    V = 2 * P * (1 - P)
    D = P[:, :-1] - P[:, -1:]
    O = D[:, :, None] * D[:, None, :]
    theta = rng.dirichlet(4.0 * np.arange(1, k + 1), 20000)[:, :-1]     # the non-baseline ancestries
    mom = moments_tensors(moments_from_vectors(theta), k1)
    m_vec = np.append(theta.mean(0), 1 - theta.mean(0).sum())
    pbar = P @ m_vec
    beta_ = 2.0 * D * (1.0 - 2.0 * pbar)[:, None]
    s2 = V @ m_vec + rng.uniform(0.01, 0.05, M)
    cres = s2 * rng.uniform(0.8, 0.95, M)
    cls = rng.integers(0, K, M)
    nu = 700.0
    rm = same_person_term(beta_, O, s2, cres, cls, K, mom, nu)
    lam = [2 * ((a == b) * theta[:, a] - theta[:, a] * theta[:, b]) for a in range(k1) for b in range(a, k1)]
    U = np.column_stack([np.ones(len(theta)), theta] + lam)
    assert U.shape[1] == len(factor_names(list("ABC")[:k]))
    W = U - U.mean(0)
    sf = factor_same_person_term(factor_loadings(V, O), s2, cres, cls, K, W.T @ W / len(W), n_sf=nu, nu_sf=nu, nu=nu)
    np.testing.assert_allclose(sf, rm, rtol=1e-9, atol=1e-13)


def _between_chromosome_sp(G, C, chrom):
    """The referee's same-person excess summed over pairs of SNPs on different chromosomes, by brute force."""
    n = G.shape[0]
    X = G - G.mean(0)
    s2 = (X ** 2).sum(0) / (n - 1)
    ok = s2 > 0
    X = X[:, ok] / np.sqrt(s2[ok])
    Q = np.linalg.qr(np.column_stack([np.ones(n), C]))[0]
    nu = n - Q.shape[1]
    Y = (X - Q @ (Q.T @ X)) ** 2
    ch = chrom[ok]
    tot = 0.0
    for c in np.unique(ch):
        Yc, Yo = Y[:, ch == c], Y[:, ch != c]
        tot += (Yc.sum(1) @ Yo.sum(1)) / nu ** 2 - Yc.sum() * Yo.sum() / (n * nu ** 2)
    return tot


def test_sample_factors_give_the_realized_same_person_term(sim, tmp_path):
    """The statistic's realized between-chromosome same-person mass is the referee's (brute force); its factors
    predict it in the full cohort and in 400 people with ten in-sample PCs."""
    from summit.admixed_ld.sample_factors import sample_factors
    d, S = sim["dir"], sim["S"]
    chrom = S["variants"].CHR.to_numpy()
    info = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "full"), covar=str(d / "cohort.cov"),
                          log=lambda *a: None)
    real = _between_chromosome_sp(S["cohort"], sim["C"], chrom)
    np.testing.assert_allclose(info["diagnostics"]["between_chromosome_same_person_mass"], real, rtol=1e-8)
    assert abs(info["diagnostics"]["share_predicted"] - 1) < 0.03
    rng = np.random.default_rng(11)
    idx = np.sort(rng.choice(S["cohort"].shape[0], 400, replace=False))
    G = S["cohort"][idx]
    iid = write_bed(str(tmp_path / "sub"), G, S["variants"], "s")
    C = principal_components(G, 10)
    pd.DataFrame({"FID": iid, "IID": iid, **{f"PC{i + 1}": C[:, i] for i in range(10)}}).to_csv(
        tmp_path / "sub.cov", sep="\t", index=False)
    info = sample_factors(str(tmp_path / "sub"), str(d / "ref"), str(tmp_path / "sub"), covar=str(tmp_path / "sub.cov"),
                          log=lambda *a: None)
    np.testing.assert_allclose(info["diagnostics"]["between_chromosome_same_person_mass"],
                               _between_chromosome_sp(G, C, chrom), rtol=1e-8)
    assert abs(info["diagnostics"]["share_predicted"] - 1) < 0.12
    assert info["n_samples"] == 400 and info["nu"] == 400 - 1 - 10


def test_sample_factors_follow_first_generation_people(tmp_path):
    """A cohort of F1s (one haplotype from each source) and random-mating admixed people: the random-mating model of
    the same-person term misses the spread of local-ancestry variance across people; the factors do not."""
    from summit.admixed_ld.sample_factors import factor_loadings, sample_factors
    from admixed_ld_sim import _haplotypes_k
    rng = np.random.default_rng(21)
    n_chr, per, chr_cm, M = 6, 300, 60.0, 1800
    chrom = np.repeat(np.arange(1, n_chr + 1), per)
    cm = np.concatenate([np.sort(rng.uniform(0, chr_cm, per)) for _ in range(n_chr)])
    base = rng.uniform(0.05, 0.95, M)
    freqs = [np.clip(base + rng.normal(0, 0.25, M), 0.03, 0.97), base]
    founders = [(rng.random((40, M)) < p[None, :]).astype(np.int8) for p in freqs]
    geno = lambda h: (h[0::2] + h[1::2]).astype(np.float64)
    panels = [geno(_haplotypes_k(rng, founders, cm, 2 * 1500, 3.0, source=s)[0]) for s in range(2)]
    n_f1, n_rm = 700, 700
    th_rm = rng.beta(4, 4, n_rm)
    th_h = np.concatenate([np.tile([[1.0, 0.0], [0.0, 1.0]], (n_f1, 1)),
                           np.repeat(np.column_stack([th_rm, 1 - th_rm]), 2, axis=0)])
    G = geno(_haplotypes_k(rng, founders, cm, 2 * (n_f1 + n_rm), 3.0, theta_h=th_h, break_rate=0.06)[0])
    v = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"f{j:05d}" for j in range(M)], "CM": cm,
                      "BP": np.rint(cm * 1e6).astype(np.int64) + 1, "A1": "A", "A2": "G"})
    write_bed(str(tmp_path / "pA"), panels[0], v, "a")
    write_bed(str(tmp_path / "pE"), panels[1], v, "e")
    build_reference([str(tmp_path / "pA"), str(tmp_path / "pE")], ["AFR", "EUR"], str(tmp_path / "ref"),
                    window_kb=3000, step=256, log=lambda *a: None)
    iid = write_bed(str(tmp_path / "co"), G, v, "c")
    C = principal_components(G, 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(tmp_path / "co.cov", sep="\t",
                                                                                 index=False)
    info = sample_factors(str(tmp_path / "co"), str(tmp_path / "ref"), str(tmp_path / "co"),
                          covar=str(tmp_path / "co.cov"), log=lambda *a: None)
    real = _between_chromosome_sp(G, C, chrom)
    np.testing.assert_allclose(info["diagnostics"]["between_chromosome_same_person_mass"], real, rtol=1e-8)
    # the random-mating model, with each person's true expected ancestry (F1s 0.5)
    th = np.concatenate([np.full(n_f1, 0.5), th_rm])
    U = np.column_stack([np.ones(th.size), th, 2 * th * (1 - th)])
    W = U - U.mean(0)
    S_rm = W.T @ W / th.size
    Pp = np.column_stack(freqs)
    Vv = 2 * Pp * (1 - Pp)
    Dd = (Pp[:, 0] - Pp[:, 1])[:, None]
    A = factor_loadings(Vv, Dd[:, :, None] * Dd[:, None, :]) / (G.var(0, ddof=1))[:, None]
    n, nu = G.shape[0], G.shape[0] - 3
    Ac = np.stack([A[chrom == c].sum(0) for c in range(1, n_chr + 1)])
    At = Ac.sum(0)
    pred_rm = n / nu ** 2 * (At @ S_rm @ At - np.einsum("ca,ab,cb->", Ac, S_rm, Ac))
    sf_share, rm_share = info["diagnostics"]["share_predicted"], pred_rm / real   # 0.97 and 0.82 for this seed
    assert abs(sf_share - 1) < 0.06 and rm_share < 0.9 and abs(sf_share - 1) < abs(rm_share - 1) / 3, (sf_share, rm_share)
    lam_sd = np.sqrt(max(np.asarray(info["sigma"])[2, 2], 0))
    assert lam_sd > 0.15                                             # F1s have lambda 0, the others about 0.45


def test_compose_with_sample_factors(sim, tmp_path):
    """In 400 people with ten in-sample PCs, compose with the sample factors is closer to the exact in-sample LD than
    the random-mating term; both terms are recorded, and the inputs are checked."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.sample_factors import sample_factors
    d, S = sim["dir"], sim["S"]
    v = S["variants"]
    rng = np.random.default_rng(11)
    idx = np.sort(rng.choice(S["cohort"].shape[0], 400, replace=False))
    G = S["cohort"][idx]
    iid = write_bed(str(tmp_path / "sub"), G, v, "s")
    C = principal_components(G, 10)
    pd.DataFrame({"FID": iid, "IID": iid, **{f"PC{i + 1}": C[:, i] for i in range(10)}}).to_csv(
        tmp_path / "sub.cov", sep="\t", index=False)
    os.makedirs(tmp_path / "ss")
    for t in range(2):
        beta, se = gwas(G, rng.standard_normal(400), C)
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": 400, "BETA": beta,
                      "SE": se, "Z": beta / se, "COV_RANK": 10}).to_csv(tmp_path / "ss" / f"t{t}.tsv", sep="\t",
                                                                        index=False)
    p = G.mean(0) / 2
    cls = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"SNP": v.SNP, "low": (cls == 0).astype(int), "high": (cls == 1).astype(int)}).to_csv(
        tmp_path / "annot.tsv", sep="\t", index=False)
    sample_factors(str(tmp_path / "sub"), str(d / "ref"), str(tmp_path / "sub"), covar=str(tmp_path / "sub.cov"),
                   log=lambda *a: None)
    sf = str(tmp_path / "sub.sample_factors.json")
    ratio = {}
    for arm, kw in (("rm", {}), ("sf", dict(sample_factors=sf))):
        out = str(tmp_path / arm)
        compose_ldscores(str(d / "ref"), [str(tmp_path / "ss")], str(tmp_path / "annot.tsv"), out,
                         generations=S["generations"], log=lambda *a: None, **kw)
        got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
        keep = v.SNP.isin(got.SNP).to_numpy()
        L_in, V_in = insample_ldscores(G[:, keep], C, cls[keep], 2)
        ok = np.isfinite(L_in).all(1) & np.isfinite(V_in)
        Lc, Vc = got[["low", "high"]].to_numpy()[ok], got.V.to_numpy()[ok]
        ratio[arm] = float((Vc[:, None] * Lc).sum() / (V_in[ok, None] * L_in[ok]).sum())
    assert abs(ratio["sf"] - 1) < abs(ratio["rm"] - 1) and abs(ratio["sf"] - 1) < 0.08, ratio
    pj = json.load(open(str(tmp_path / "sf") + ".admixed_ld.json"))
    sfj = pj["sample_factors"]
    assert sfj["n_samples"] == 400
    assert sfj["same_person_mass"] < sfj["same_person_mass_random_mating"]
    cal = sfj["calibration"]                                         # scaled to the realized between-chromosome mass
    real = json.load(open(sf))["diagnostics"]["between_chromosome_same_person_mass"]
    np.testing.assert_allclose(cal["factor"], real / cal["compose_between_chromosome_mass"], rtol=1e-12)
    np.testing.assert_allclose(sfj["same_person_mass"], cal["factor"] * sfj["same_person_mass_uncalibrated"],
                               rtol=1e-10)
    compose_ldscores(str(d / "ref"), [str(tmp_path / "ss")], str(tmp_path / "annot.tsv"), str(tmp_path / "sfu"),
                     generations=S["generations"], sample_factors=sf, sample_factors_calibrate=False,
                     log=lambda *a: None)
    pu = json.load(open(str(tmp_path / "sfu") + ".admixed_ld.json"))["sample_factors"]
    assert pu["calibration"] is None
    np.testing.assert_allclose(pu["same_person_mass"], sfj["same_person_mass_uncalibrated"], rtol=1e-12)
    bad = json.load(open(sf))
    bad["sources"] = ["EUR", "AFR"]
    json.dump(bad, open(tmp_path / "bad.sample_factors.json", "w"))
    with pytest.raises(ValueError, match="sources"):
        compose_ldscores(str(d / "ref"), [str(tmp_path / "ss")], str(tmp_path / "annot.tsv"), str(tmp_path / "x"),
                         generations=S["generations"], sample_factors=str(tmp_path / "bad.sample_factors.json"),
                         log=lambda *a: None)


def test_compose_writes_one_file_per_gwas_sample_size(fam, tmp_path):
    """Traits with two sample sizes (the whole cohort and a random 60% of it): compose writes one LD-score file per N,
    each equal to a compose of that N's traits alone (the population part is shared; the same-person and relatives'
    terms follow each N), and the main file is the median N's."""
    import shutil
    from summit.admixed_ld.compose import compose_ldscores
    d, S, C = fam["dir"], fam["S"], fam["C"]
    G = S["cohort"]
    n = G.shape[0]
    rng = np.random.default_rng(8)
    sub = np.sort(rng.choice(n, int(0.6 * n), replace=False))
    v = S["variants"]
    for nm in ("mix", "full", "part"):
        os.makedirs(tmp_path / nm)
    for t in range(2):
        shutil.copy(d / "ss" / f"trait{t}.tsv", tmp_path / "mix" / f"full{t}.tsv")
        shutil.copy(d / "ss" / f"trait{t}.tsv", tmp_path / "full" / f"full{t}.tsv")
    for t in range(3):
        beta, se = gwas(G[sub], rng.standard_normal(sub.size), C[sub])
        df = pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": sub.size, "BETA": beta,
                           "SE": se, "Z": beta / se, "COV_RANK": 2})
        df.to_csv(tmp_path / "mix" / f"part{t}.tsv", sep="\t", index=False)
        df.to_csv(tmp_path / "part" / f"part{t}.tsv", sep="\t", index=False)
    first = str(tmp_path / "first")
    compose_ldscores(str(d / "ref"), [str(tmp_path / "full")], str(d / "annot.tsv"), first, curve=str(d / "c"),
                     relatedness="curve", release_sample_factors=False, log=lambda *a: None)
    pj = json.load(open(first + ".admixed_ld.json"))
    kw = dict(curve=str(d / "c"), relatedness="curve", mean_ancestry=str(pj["mean_ancestry"]),
              kappa0=str(pj["kappa0"]), release_sample_factors=False, log=lambda *a: None)
    out = {}
    for nm in ("mix", "full", "part"):
        compose_ldscores(str(d / "ref"), [str(tmp_path / nm)], str(d / "annot.tsv"), str(tmp_path / f"c_{nm}"), **kw)
        out[nm] = pd.read_csv(str(tmp_path / f"c_{nm}") + ".gw.ldscore.gz", sep="\t")
    groups = pd.read_csv(str(tmp_path / "c_mix") + ".n_groups.tsv", sep="\t")
    assert sorted(groups.N.unique()) == [sub.size, n]
    for N_, ref_ in ((n, "full"), (sub.size, "part")):
        f = str(tmp_path / groups[groups.N == N_].ldscores.iloc[0])     # file names, next to the table
        got = pd.read_csv(f, sep="\t")
        np.testing.assert_allclose(got[["V", "low", "high"]].to_numpy(), out[ref_][["V", "low", "high"]].to_numpy(),
                                   rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(out["mix"][["low", "high"]].to_numpy(), out["part"][["low", "high"]].to_numpy(),
                               rtol=1e-8, atol=1e-10)                # 3 of 5 traits have the smaller N (the median)
    pm = json.load(open(str(tmp_path / "c_mix") + ".admixed_ld.json"))
    assert len(pm["n_groups"]) == 2
    assert {g["label"] for g in pm["n_groups"]} == {str(n), str(sub.size)}
    assert not os.path.exists(str(tmp_path / "c_full") + ".n_groups.tsv")   # one sample size: no extra files
    # the params json's plan lists the file each trait was given, as n_groups.tsv does, and each sample size's file
    pf = pd.DataFrame(pm["plan"]["files"])
    pd.testing.assert_frame_equal(pf.sort_values("trait").reset_index(drop=True),
                                  groups.sort_values("trait").reset_index(drop=True), check_dtype=False)
    assert sorted(s["file"] for s in pm["plan"]["sample_sizes"]) == sorted(groups.ldscores.unique())
    pfull = json.load(open(str(tmp_path / "c_full") + ".admixed_ld.json"))["plan"]
    assert {f["ldscores"] for f in pfull["files"]} == {"c_full.gw.ldscore.gz"} and len(pfull["files"]) == 2
    # summit --h2 with the compose output fits each trait with its own group's file (n_groups.tsv), in directory order
    import subprocess
    import sys
    an = pd.read_csv(d / "annot.tsv", sep="\t").merge(v[["CHR", "BP", "SNP"]], on="SNP")
    an[["CHR", "BP", "SNP", "low", "high"]].to_csv(tmp_path / "annot_h2.tsv", sep="\t", index=False)
    run = lambda *a: subprocess.run([sys.executable, "-m", "summit", *a, "--annot", str(tmp_path / "annot_h2.tsv"),
                                     "--njack", "20"], check=True, capture_output=True, text=True)
    run("--h2", str(tmp_path / "mix"), "--ldscores", str(tmp_path / "c_mix") + ".gw.ldscore.gz", "--out",
        str(tmp_path / "h_mix"))
    got = pd.read_csv(str(tmp_path / "h_mix") + ".results.tsv", sep="\t").set_index("phen")
    assert list(got.index) == ["full0", "full1", "part0", "part1", "part2"]
    for N_, nm in ((n, "full"), (sub.size, "part")):
        f = str(tmp_path / groups[groups.N == N_].ldscores.iloc[0])
        run("--h2", str(tmp_path / nm), "--ldscores", f, "--ldscore-groups", "off", "--out", str(tmp_path / f"h_{nm}"))
        ref_ = pd.read_csv(str(tmp_path / f"h_{nm}") + ".results.tsv", sep="\t").set_index("phen")
        np.testing.assert_allclose(got.loc[ref_.index, "h2"], ref_.h2, rtol=1e-10)
        np.testing.assert_allclose(got.loc[ref_.index, "h2_se"], ref_.h2_se, rtol=1e-10)
    used = pd.read_csv(str(tmp_path / "h_mix") + ".ldscore_groups.tsv", sep="\t")
    assert len(used) == 5 and used.ldscores.nunique() == 2
    # the main file alone (old behaviour) differs for the traits of the other sample size
    run("--h2", str(tmp_path / "mix"), "--ldscores", str(tmp_path / "c_mix") + ".gw.ldscore.gz", "--ldscore-groups",
        "off", "--out", str(tmp_path / "h_main"))
    main_ = pd.read_csv(str(tmp_path / "h_main") + ".results.tsv", sep="\t").set_index("phen")
    assert not np.allclose(main_.loc[["full0", "full1"], "h2"], got.loc[["full0", "full1"], "h2"], rtol=1e-8)


@pytest.fixture(scope="module")
def fam_groups(fam, tmp_path_factory):
    """Five traits with two GWAS sample sizes (full0, full1 of the whole cohort; part0-2 of a random 60% of it) composed
    to one prefix with its n_groups.tsv, and the annotation for --h2."""
    import shutil
    from summit.admixed_ld.compose import compose_ldscores
    d, S, C = fam["dir"], fam["S"], fam["C"]
    t = tmp_path_factory.mktemp("fam_groups")
    G, v = S["cohort"], S["variants"]
    rng = np.random.default_rng(8)
    sub = np.sort(rng.choice(G.shape[0], int(0.6 * G.shape[0]), replace=False))
    os.makedirs(t / "mix")
    for i in range(2):
        shutil.copy(d / "ss" / f"trait{i}.tsv", t / "mix" / f"full{i}.tsv")
    for i in range(3):
        beta, se = gwas(G[sub], rng.standard_normal(sub.size), C[sub])
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": sub.size, "BETA": beta,
                      "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(t / "mix" / f"part{i}.tsv", sep="\t", index=False)
    kw = dict(curve=str(d / "c"), relatedness="curve", release_sample_factors=False, log=lambda *a: None)
    compose_ldscores(str(d / "ref"), [str(t / "mix")], str(d / "annot.tsv"), str(t / "c_mix"), **kw)
    an = pd.read_csv(d / "annot.tsv", sep="\t").merge(v[["CHR", "BP", "SNP"]], on="SNP")
    an[["CHR", "BP", "SNP", "low", "high"]].to_csv(t / "annot_h2.tsv", sep="\t", index=False)
    return dict(dir=t, prefix=str(t / "c_mix"), n=G.shape[0], n_sub=sub.size, compose_kw=kw)


def _summit_h2(fg, *a, check=True):
    import subprocess
    return subprocess.run([sys.executable, "-m", "summit", *a, "--annot", str(fg["dir"] / "annot_h2.tsv"),
                           "--njack", "20"], check=check, capture_output=True, text=True)


def test_h2_over_ldscore_groups_rebuilds_its_links_on_a_rerun(fam_groups, tmp_path):
    """A second --h2 to the same --out with other summary statistics of the same names, one trait moved to the other
    group and one trait left out fits exactly the second run's files, each trait once, as a fresh --out does."""
    import shutil
    fg = fam_groups
    tab = pd.read_csv(fg["prefix"] + ".n_groups.tsv", sep="\t")
    tab["ldscores"] = [f if os.path.isabs(f) else str(fg["dir"] / f) for f in tab.ldscores]   # the table moves here
    shutil.copy(fg["prefix"] + ".gw.ldscore.gz", tmp_path / "moved.gw.ldscore.gz")
    full_file = tab.ldscores[tab.trait == "full0.tsv"].iloc[0]
    moved = tab[tab.trait != "part2.tsv"].copy()
    moved.loc[moved.trait == "part0.tsv", "ldscores"] = full_file       # part0 changed group (a recomposition)
    moved.to_csv(tmp_path / "moved.n_groups.tsv", sep="\t", index=False)
    os.makedirs(tmp_path / "mix2")
    for f in sorted(os.listdir(fg["dir"] / "mix")):
        if f == "part2.tsv":                                            # left out of the second run
            continue
        t = pd.read_csv(fg["dir"] / "mix" / f, sep="\t")
        t["BETA"] *= 3.0
        t["Z"] = t.BETA / t.SE
        t.to_csv(tmp_path / "mix2" / f, sep="\t", index=False)
    out = str(tmp_path / "h")
    _summit_h2(fg, "--h2", str(fg["dir"] / "mix"), "--ldscores", fg["prefix"] + ".gw.ldscore.gz", "--out", out)
    first = pd.read_csv(out + ".results.tsv", sep="\t").set_index("phen")
    assert sorted(os.listdir(out + ".ldgroups/1/ss")) == ["part0.tsv", "part1.tsv", "part2.tsv"]
    second = []
    for o in (out, str(tmp_path / "fresh")):                              # the same --out again, and a fresh one
        _summit_h2(fg, "--h2", str(tmp_path / "mix2"), "--ldscores", str(tmp_path / "moved.gw.ldscore.gz"), "--out", o)
        second.append(pd.read_csv(o + ".results.tsv", sep="\t").set_index("phen"))
    rerun, fresh = second
    assert list(rerun.index) == ["full0", "full1", "part0", "part1"]
    pd.testing.assert_frame_equal(rerun, fresh)
    assert not np.allclose(rerun.h2, first.loc[rerun.index, "h2"], rtol=1e-6)
    assert sorted(os.listdir(out + ".ldgroups/0/ss")) == ["full0.tsv", "full1.tsv", "part0.tsv"]
    assert sorted(os.listdir(out + ".ldgroups/1/ss")) == ["part1.tsv"]
    for f in os.listdir(out + ".ldgroups/0/ss"):
        assert os.path.realpath(os.path.join(out + ".ldgroups/0/ss", f)) == str((tmp_path / "mix2" / f).resolve())
    used = pd.read_csv(out + ".ldscore_groups.tsv", sep="\t").set_index("phen").ldscores
    assert len(used) == 4 and used["part0"] == used["full0"] != used["part1"]


def test_h2_over_ldscore_groups_writes_the_single_run_outputs(fam_groups, tmp_path):
    """With LD-score groups, --write-jack's per-trait files are OUT.<phen>.jack, as in a single-file run and equal to
    a fit of the group alone, and OUT.log is this run's whole log (every group's fit; a rerun does not append)."""
    fg = fam_groups
    out = str(tmp_path / "h")
    for _ in range(2):                                                  # the second run to the same --out
        _summit_h2(fg, "--h2", str(fg["dir"] / "mix"), "--ldscores", fg["prefix"] + ".gw.ldscore.gz", "--write-jack",
                   "--out", out)
    phen = ["full0", "full1", "part0", "part1", "part2"]
    assert all(os.path.exists(f"{out}.{p}.jack") for p in phen)
    left = [f for r, _, fs in os.walk(out + ".ldgroups") for f in fs if f.endswith(".jack")]
    assert left == []
    log = open(out + ".log").read()
    assert log.count("Analysis started at") == 2 and log.count("[h2] part0.tsv: LD scores") == 1
    assert "Total SNP heritability" in log and f"Saved h2 results table in {out}.results.tsv" in log
    from summit.cli import _ldscore_groups
    _, tab = _ldscore_groups(fg["prefix"])
    alone = str(tmp_path / "alone")
    _summit_h2(fg, "--h2", str(fg["dir"] / "mix" / "part1.tsv"), "--ldscores",
               tab.ldscores[tab.trait == "part1.tsv"].iloc[0], "--ldscore-groups", "off", "--write-jack", "--out", alone)
    assert open(f"{out}.part1.jack").read() == open(f"{alone}.part1.jack").read()
    # OUT.results.tsv holds the groups' rows as their fits wrote them (numbers and "nan" as in a single-file run)
    merged = open(out + ".results.tsv").read().splitlines()
    rows = {ln.split("\t")[1]: ln.split("\t")[2:] for ln in merged[1:]}
    assert [ln.split("\t")[0] for ln in merged[1:]] == [str(i) for i in range(5)] and list(rows) == phen
    for k in os.listdir(out + ".ldgroups"):
        fit = open(os.path.join(out + ".ldgroups", k, "fit.results.tsv")).read().splitlines()
        assert fit[0] == merged[0]
        for ln in fit[1:]:
            assert rows[ln.split("\t")[1]] == ln.split("\t")[2:]


def test_n_groups_table_on_a_rerun_from_any_directory_and_for_one_trait(fam, fam_groups, tmp_path, monkeypatch):
    """compose's n_groups.tsv names its files relative to its own directory (read from any working directory; older
    tables with compose's --out paths stay readable); a rerun to the same prefix with one sample size removes it; and
    --h2 with one file in auto mode stops for a trait the table does not list."""
    import shutil
    from summit.admixed_ld.compose import compose_ldscores
    from summit.cli import _ldscore_groups
    d, fg = fam["dir"], fam_groups
    out = str(tmp_path / "same")
    compose_ldscores(str(d / "ref"), [str(fg["dir"] / "mix")], str(d / "annot.tsv"), out, **fg["compose_kw"])
    raw = pd.read_csv(out + ".n_groups.tsv", sep="\t")
    assert len(raw) == 5 and raw.ldscores.nunique() == 2 and not any(os.sep in f for f in raw.ldscores)
    os.makedirs(tmp_path / "elsewhere")
    monkeypatch.chdir(tmp_path / "elsewhere")
    _, tab = _ldscore_groups(out + ".gw.ldscore.gz")
    assert all(os.path.isabs(f) and os.path.exists(f) and os.path.dirname(f) == str(tmp_path) for f in tab.ldscores)
    # a table written before the names (compose run in tmp_path with --out res/c): the files next to the table
    os.makedirs(tmp_path / "res")
    old = []
    for t, n_, f in raw[["trait", "N", "ldscores"]].itertuples(index=False):
        name = f.replace("same.", "c.", 1)
        if not os.path.exists(tmp_path / "res" / name):
            os.symlink(tmp_path / f, tmp_path / "res" / name)
        old.append(dict(trait=t, N=n_, ldscores=f"res/{name}"))
    pd.DataFrame(old).to_csv(tmp_path / "res" / "c.n_groups.tsv", sep="\t", index=False)
    for cwd in (tmp_path / "elsewhere", tmp_path):
        monkeypatch.chdir(cwd)
        _, tab_old = _ldscore_groups(str(tmp_path / "res" / "c"))
        assert [os.path.realpath(f) for f in tab_old.ldscores] == [os.path.realpath(f) for f in tab.ldscores]
    # the same prefix again with one sample size: no table is left for --h2 to read
    os.makedirs(tmp_path / "full")
    for i in range(2):
        shutil.copy(fg["dir"] / "mix" / f"full{i}.tsv", tmp_path / "full" / f"full{i}.tsv")
    compose_ldscores(str(d / "ref"), [str(tmp_path / "full")], str(d / "annot.tsv"), out, **fg["compose_kw"])
    assert not os.path.exists(out + ".n_groups.tsv") and _ldscore_groups(out + ".gw.ldscore.gz") is None
    # one --h2 file: its own group's file, or an error for a trait that the table does not list
    _, ftab = _ldscore_groups(fg["prefix"])
    part_file = ftab.ldscores[ftab.trait == "part0.tsv"].iloc[0]
    one = _summit_h2(fg, "--h2", str(fg["dir"] / "mix" / "part0.tsv"), "--ldscores", fg["prefix"] + ".gw.ldscore.gz",
                     "--out", str(tmp_path / "one"))
    assert f"LD scores {part_file}" in one.stdout and ".n_groups.tsv" in one.stdout
    _summit_h2(fg, "--h2", str(fg["dir"] / "mix" / "part0.tsv"), "--ldscores", part_file, "--ldscore-groups", "off",
               "--out", str(tmp_path / "off"))
    a, b = (pd.read_csv(str(tmp_path / x) + ".results.tsv", sep="\t") for x in ("one", "off"))
    np.testing.assert_allclose(a.h2, b.h2, rtol=1e-12)
    shutil.copy(fg["dir"] / "mix" / "part0.tsv", tmp_path / "other.tsv")
    r = _summit_h2(fg, "--h2", str(tmp_path / "other.tsv"), "--ldscores", fg["prefix"] + ".gw.ldscore.gz", "--out",
                   str(tmp_path / "other"), check=False)
    assert r.returncode != 0 and "other.tsv of --h2 is not in" in r.stdout + r.stderr
    assert not os.path.exists(str(tmp_path / "other") + ".results.tsv")


def test_group_file_names_resolve_next_to_their_table_only(fam_groups, tmp_path, monkeypatch):
    """A file name in n_groups.tsv is looked up next to the table only: when it is missing there, --h2 stops with that
    path and never takes a file of the same name from the working directory (another run's); an old entry with a
    directory part keeps its fallbacks."""
    import shutil
    import subprocess
    from summit.cli import _ldscore_groups
    fg = fam_groups
    tab = pd.read_csv(fg["prefix"] + ".n_groups.tsv", sep="\t")
    a, b = tmp_path / "runA", tmp_path / "runB"
    os.makedirs(a)
    os.makedirs(b)
    for f in ["c_mix.gw.ldscore.gz", "c_mix.gw.M", "c_mix.n_groups.tsv"] + sorted(set(tab.ldscores)):
        shutil.copy(fg["dir"] / f, a / f)
    gone = tab.ldscores[tab.trait == "part0.tsv"].iloc[0]
    os.remove(a / gone)
    shutil.copy(fg["dir"] / gone, b / gone)                             # another run's file of the same name
    monkeypatch.chdir(b)
    _, t = _ldscore_groups(str(a / "c_mix"))
    assert t.ldscores[t.trait == "part0.tsv"].iloc[0] == str(a / gone)
    r = subprocess.run([sys.executable, "-m", "summit", "--h2", str(fg["dir"] / "mix"), "--ldscores",
                        str(a / "c_mix.gw.ldscore.gz"), "--annot", str(fg["dir"] / "annot_h2.tsv"), "--njack", "20",
                        "--out", str(tmp_path / "h")], cwd=b, capture_output=True, text=True)
    assert r.returncode != 0 and f"{a / gone} listed in" in r.stdout + r.stderr and "does not exist" in r.stdout + r.stderr
    assert not os.path.exists(str(tmp_path / "h") + ".results.tsv")
    # an entry with a directory part (a table written before the names): the file next to the table, else from here
    old = pd.DataFrame([dict(trait="part0.tsv", N=1, ldscores=f"res/{gone}")])
    old.to_csv(a / "old.n_groups.tsv", sep="\t", index=False)
    os.makedirs(b / "res")
    shutil.copy(fg["dir"] / gone, b / "res" / gone)
    assert _ldscore_groups(str(a / "old"))[1].ldscores.iloc[0] == f"res/{gone}"


def test_curve_writes_the_sample_statistic_and_compose_uses_it(sim, tmp_path):
    """The curve's pass writes the sample statistic of the same people (as sample-factors run on its own), and compose
    with that curve uses it for the same-person term unless told not to (curve mode's default, 22 Nov 2026)."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.sample_factors import sample_factors
    d = sim["dir"]
    quiet = lambda *a, **k: None
    cv = str(tmp_path / "cv")
    _, cinfo = genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5, log=quiet)
    assert cinfo["sample_factors"] == "cv.sample_factors.json" and os.path.exists(cv + ".sample_factors.json")
    alone = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "alone"), covar=str(d / "cohort.cov"),
                           ld_mass=False, log=quiet)
    with_curve = json.load(open(cv + ".sample_factors.json"))          # accumulated in the curve's own pass
    assert with_curve["n_snps"] == alone["n_snps"] and with_curve["n_samples"] == alone["n_samples"]
    for k in ("sigma", "mean"):
        np.testing.assert_allclose(with_curve[k], alone[k], rtol=1e-11, atol=1e-15)
    for k in ("between_chromosome_same_person_mass", "predicted_by_factors"):
        np.testing.assert_allclose(with_curve["diagnostics"][k], alone["diagnostics"][k], rtol=1e-11)
    # the curve itself does not change when the statistic is accumulated in its pass
    cv0 = str(tmp_path / "cv0")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv0, covar=str(d / "cohort.cov"), cell_cm=0.5,
                   sample_factors=False, log=quiet)
    assert not os.path.exists(cv0 + ".sample_factors.json")
    pd.testing.assert_frame_equal(pd.read_csv(cv + ".curve.tsv", sep="\t"), pd.read_csv(cv0 + ".curve.tsv", sep="\t"))
    # the curve's json names its files by their basename (cv.cells.npz), the rest by path
    strip = lambda j: json.loads(json.dumps(j).replace(cv0, "OUT").replace(cv, "OUT").replace('"cv0.', '"OUT.')
                                 .replace('"cv.', '"OUT.'))
    j1, j0 = json.load(open(cv + ".curve.json")), json.load(open(cv0 + ".curve.json"))
    j1.pop("sample_factors")
    assert strip(j1) == strip(j0)
    for k in ("S1_upper", "S2_upper"):
        np.testing.assert_array_equal(np.load(cv + ".cells.npz")[k], np.load(cv0 + ".cells.npz")[k])
    args = (str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"))
    got = {}
    for arm, kw in (("auto", {}), ("explicit", dict(sample_factors=cv + ".sample_factors.json")),
                    ("random_mating", dict(release_sample_factors=False))):
        compose_ldscores(*args, str(tmp_path / arm), curve=cv, log=quiet, **kw)
        got[arm] = pd.read_csv(str(tmp_path / arm) + ".gw.ldscore.gz", sep="\t")[["low", "high"]].to_numpy()
    np.testing.assert_allclose(got["auto"], got["explicit"], rtol=1e-12, atol=1e-14)
    assert not np.allclose(got["auto"], got["random_mating"], rtol=1e-6, atol=0)
    pj = json.load(open(str(tmp_path / "auto") + ".admixed_ld.json"))
    assert pj["sample_factors"]["from_release"] is True
    assert "sample_factors_sample_differs" not in {f["flag"] for f in pj["flags"]}


def test_compose_takes_only_the_statistic_its_curve_lists(sim, tmp_path):
    """compose with a curve takes the statistic the curve's json lists and stops when it is missing or of other people
    or covariates (a file of another run); a curve that lists none (here remade with --no-sample-factors over an older
    statistic) gives the random-mating term with the flag curve_without_sample_factors."""
    import shutil
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.sample_factors import sample_factors
    d = sim["dir"]
    quiet = lambda *a, **k: None
    cv = str(tmp_path / "cv")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5, log=quiet)
    shutil.copy(cv + ".sample_factors.json", tmp_path / "good.sample_factors.json")
    args = (str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"))
    kw = dict(curve=cv, mean_ancestry="0.6", kappa0="0.3", allow_flat_kernel_join=True, log=quiet)
    plan = compose_ldscores(*args, str(tmp_path / "ok"), dry_run=True, **kw)
    assert plan["same_person_term"] == "the sample statistic released with the curve"
    # another run's statistic in its place: no covariates (rank 0 against the curve's 2)
    sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "nocov"), ld_mass=False, log=quiet)
    shutil.copy(tmp_path / "nocov.sample_factors.json", cv + ".sample_factors.json")
    for case in ("other run", "missing"):
        with pytest.raises(ValueError, match="--sample-factors PATH") as e:
            compose_ldscores(*args, str(tmp_path / "x"), dry_run=True, **kw)
        assert "--no-sample-factors" in str(e.value)
        assert ("covariate_rank 0 in the statistic, 2 in the curve" if case == "other run" else "does not exist") \
            in str(e.value), str(e.value)
        if case == "other run":
            os.remove(cv + ".sample_factors.json")
    # the curve remade without the statistic (and without covariates) over the earlier run's file
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, cell_cm=0.5, sample_factors=False, log=quiet)
    shutil.copy(tmp_path / "good.sample_factors.json", cv + ".sample_factors.json")
    lines = []
    compose_ldscores(*args, str(tmp_path / "stale"), **dict(kw, log=lambda *a, **k: lines.append(" ".join(map(str, a)))))
    compose_ldscores(*args, str(tmp_path / "rm"), release_sample_factors=False, **kw)
    pj, rm = (json.load(open(str(tmp_path / x) + ".admixed_ld.json")) for x in ("stale", "rm"))
    assert pj["sample_factors"] is None and pj["plan"]["same_person_term"] == "random mating"
    fl = {f["flag"]: f["detail"] for f in pj["flags"]}
    assert "lists no sample statistic" in fl["curve_without_sample_factors"]
    assert "cv.sample_factors.json is not the curve's and is not used" in fl["curve_without_sample_factors"]
    assert "curve_without_sample_factors" not in {f["flag"] for f in rm["flags"]}
    assert any("WARNING (recorded flag curve_without_sample_factors)" in ln for ln in lines)
    a, b = (pd.read_csv(str(tmp_path / x) + ".gw.ldscore.gz", sep="\t") for x in ("stale", "rm"))
    pd.testing.assert_frame_equal(a, b)


def test_curve_does_not_overwrite_a_statistic_it_did_not_make(sim, tmp_path):
    """The curve tags its statistic (made_by 'curve') and stops before its pass when PREFIX.sample_factors.json is
    another file, such as one of sample-factors (made_by 'sample-factors') or an untagged one; it replaces a curve's."""
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.sample_factors import sample_factors
    d = sim["dir"]
    quiet = lambda *a, **k: None
    cv = str(tmp_path / "cv")
    sf = sample_factors(str(d / "cohort"), str(d / "ref"), cv, ld_mass=False, log=quiet)      # no covariates
    assert sf["made_by"] == "sample-factors"
    before = open(cv + ".sample_factors.json").read()
    kw = dict(covar=str(d / "cohort.cov"), cell_cm=0.5, cells_cm=0, log=quiet)
    with pytest.raises(ValueError, match="not written by a curve .*--no-sample-factors"):
        genotype_curve(str(d / "cohort"), str(d / "ref"), cv, **kw)
    assert open(cv + ".sample_factors.json").read() == before and not os.path.exists(cv + ".curve.json")
    untagged = json.loads(before)
    untagged.pop("made_by")
    json.dump(untagged, open(cv + ".sample_factors.json", "w"))
    with pytest.raises(ValueError, match="made_by None"):
        genotype_curve(str(d / "cohort"), str(d / "ref"), cv, **kw)
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, sample_factors=False, **kw)    # leaves the file alone
    assert json.load(open(cv + ".sample_factors.json")) == untagged
    os.remove(cv + ".sample_factors.json")
    for _ in range(2):                                                  # a curve's own statistic is replaced
        genotype_curve(str(d / "cohort"), str(d / "ref"), cv, **kw)
        j = json.load(open(cv + ".sample_factors.json"))
        assert j["made_by"] == "curve" and j["covariate_rank"] == 2


def test_single_chromosome_curve_makes_no_statistic(sim, tmp_path):
    """A curve on one chromosome has no second half for the statistic: the curve is written without it (sample_factors
    null and the flag sample_factors_not_made, also over an earlier run's statistic at the same prefix), compose with it
    takes the random-mating term, and sample-factors alone and an accumulator with an empty half stop with an error."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.sample_factors import (FactorAccumulator, SampleFactorsUnavailable, factor_names,
                                                  sample_factors)
    d, S = sim["dir"], sim["S"]
    quiet = lambda *a, **k: None
    v = S["variants"]
    c1 = (v.CHR == v.CHR.iloc[0]).to_numpy()
    write_bed(str(tmp_path / "one"), S["cohort"][:, c1], v[c1].reset_index(drop=True), "c")
    cv = str(tmp_path / "cv")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5, cells_cm=0,
                   log=quiet)                                             # an earlier run on every chromosome
    assert json.load(open(cv + ".curve.json"))["sample_factors"] == "cv.sample_factors.json"
    lines = []
    _, info = genotype_curve(str(tmp_path / "one"), str(d / "ref"), cv, cell_cm=0.5, cells_cm=0,
                             log=lambda *a, **k: lines.append(" ".join(map(str, a))))
    cj = json.load(open(cv + ".curve.json"))
    assert cj["sample_factors"] is None and info["sample_factors"] is None and cj["n_snps"] == int(c1.sum())
    detail = {f["flag"]: f["detail"] for f in cj["flags"]}["sample_factors_not_made"]
    assert "1 chromosome(s)" in detail
    assert any("WARNING (recorded flag sample_factors_not_made)" in ln for ln in lines)
    clog = []
    plan = compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "x"), curve=cv,
                            mean_ancestry="0.6", kappa0="0.3", allow_flat_kernel_join=True, dry_run=True,
                            log=lambda *a, **k: clog.append(" ".join(map(str, a))))
    assert plan["same_person_term"] == "random mating" and "curve_without_sample_factors" in plan["flags"]
    assert any("could not make one (the SNPs with finite loadings are on 1 chromosome(s)" in ln for ln in clog)
    with pytest.raises(SampleFactorsUnavailable, match="two or more chromosomes"):
        sample_factors(str(tmp_path / "one"), str(d / "ref"), str(tmp_path / "sf"), ld_mass=False, log=quiet)
    assert not os.path.exists(tmp_path / "sf.sample_factors.json")
    rng = np.random.default_rng(1)                                        # two chromosomes, the second without SNPs
    names = factor_names(["AFR", "EUR"])
    acc = FactorAccumulator(50, 49, 2, len(names), ld_mass=False)
    X = rng.standard_normal((50, 30))
    acc.add(0, X - X.mean(0), np.zeros((50, 0)), np.abs(rng.standard_normal((30, len(names)))))
    with pytest.raises(SampleFactorsUnavailable, match="second half .* 0 SNPs on 0 chromosome"):
        acc.finish(np.zeros((50, 0)), ["AFR", "EUR"], names, log=quiet)


def test_compose_plan_dry_run_and_require(sim, tmp_path):
    """compose writes its plan (decay, same-person term, relatives, cell matrix, sample sizes) before composing:
    --dry-run stops there, --require stops when a component would be missing, and the plan is in the params json."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    quiet = lambda *a, **k: None
    cv = str(tmp_path / "cv")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5, log=quiet)
    args = (str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"))
    out = str(tmp_path / "dry")
    plan = compose_ldscores(*args, out, curve=cv, dry_run=True, log=quiet)
    assert os.path.exists(out + ".plan.json") and not os.path.exists(out + ".gw.ldscore.gz")
    assert plan["mode"] == "curve" and plan["components"]["statistic"] and plan["components"]["curve"]
    assert plan["same_person_term"].startswith("the sample statistic released") and plan["traits"] == 3
    assert json.load(open(out + ".plan.json")) == json.loads(json.dumps(plan))
    with pytest.raises(ValueError, match="--require statistic"):
        compose_ldscores(*args, str(tmp_path / "no_sf"), curve=cv, release_sample_factors=False, require="statistic",
                         dry_run=True, log=quiet)
    with pytest.raises(ValueError, match="unknown component"):
        compose_ldscores(*args, str(tmp_path / "bad"), curve=cv, require="statistics", dry_run=True, log=quiet)
    out = str(tmp_path / "run")
    compose_ldscores(*args, out, curve=cv, require="curve,statistic", log=quiet)
    pj = json.load(open(out + ".admixed_ld.json"))
    assert pj["plan"]["mode"] == "curve" and pj["plan"]["components"] == plan["components"]
    date = compose_ldscores(*args, str(tmp_path / "date"), generations=sim["S"]["generations"], dry_run=True, log=quiet)
    assert date["mode"] == "date" and date["same_person_term"] == "random mating"


def test_compose_overlapping_annotation_columns(sim, tmp_path):
    """Overlapping 0/1 columns (two MAF classes and a functional column across both) are composed on their disjoint
    combinations and summed back: each column's LD scores and masses equal the sums of the combinations' when those
    are given as a partition, and SUMMIT's h2 fit reads the result."""
    import subprocess
    import sys
    from summit.admixed_ld.compose import compose_ldscores
    d, S = sim["dir"], sim["S"]
    quiet = lambda *a, **k: None
    v = S["variants"]
    a = pd.read_csv(d / "annot.tsv", sep="\t").merge(v[["CHR", "BP", "SNP"]], on="SNP")
    func = (np.arange(len(a)) % 3 == 0).astype(int)
    over = pd.DataFrame({"CHR": a.CHR, "BP": a.BP, "SNP": a.SNP, "low": a.low, "high": a.high, "func": func})
    over.to_csv(tmp_path / "over.tsv", sep="\t", index=False)
    atoms = pd.DataFrame({"CHR": a.CHR, "BP": a.BP, "SNP": a.SNP, "low0": a.low * (1 - func), "low1": a.low * func,
                          "high0": a.high * (1 - func), "high1": a.high * func})
    atoms.to_csv(tmp_path / "atoms.tsv", sep="\t", index=False)
    kw = dict(generations=S["generations"], log=quiet)
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(tmp_path / "over.tsv"), str(tmp_path / "o"), **kw)
    compose_ldscores(str(d / "ref"), [str(d / "ss")], str(tmp_path / "atoms.tsv"), str(tmp_path / "a"), **kw)
    lo = pd.read_csv(str(tmp_path / "o") + ".gw.ldscore.gz", sep="\t")
    la = pd.read_csv(str(tmp_path / "a") + ".gw.ldscore.gz", sep="\t")
    assert list(lo.columns[-3:]) == ["low", "high", "func"]
    np.testing.assert_allclose(lo.V, la.V, rtol=1e-6)
    scale = float(np.abs(la[["low0", "low1", "high0", "high1"]].to_numpy()).max())
    for col, parts in (("low", ["low0", "low1"]), ("high", ["high0", "high1"]), ("func", ["low1", "high1"])):
        np.testing.assert_allclose(lo[col], la[parts].sum(axis=1), rtol=1e-6, atol=1e-6 * scale)
    mo = pd.read_csv(str(tmp_path / "o") + ".gw.M", sep="\t", header=None).to_numpy().ravel()
    ma = pd.read_csv(str(tmp_path / "a") + ".gw.M", sep="\t", header=None).to_numpy().ravel()
    np.testing.assert_allclose(mo, [ma[0] + ma[1], ma[2] + ma[3], ma[1] + ma[3]])
    pj = json.load(open(str(tmp_path / "o") + ".admixed_ld.json"))
    assert pj["annotation"] == dict(columns=["low", "high", "func"], overlapping=True, combinations=4)
    np.testing.assert_allclose(pj["ld_mass"], json.load(open(str(tmp_path / "a") + ".admixed_ld.json"))["ld_mass"],
                               rtol=1e-10)
    subprocess.run([sys.executable, "-m", "summit", "--h2", str(d / "ss"), "--ldscores",
                    str(tmp_path / "o") + ".gw.ldscore.gz", "--annot", str(tmp_path / "over.tsv"), "--njack", "20",
                    "--out", str(tmp_path / "h")], check=True, capture_output=True, text=True)
    res = pd.read_csv(str(tmp_path / "h") + ".results.tsv", sep="\t")
    assert len(res) == 3 and (res.num_bins == 3).all() and np.isfinite(res.h2).all()


def test_compose_stops_on_too_many_annotation_combinations(sim, tmp_path):
    """Overlapping columns with more distinct combinations than --max-annotation-atoms (default 64) stop compose
    before any per-combination array, with the memory it would need; a higher limit composes them."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.entrypoint import main
    d, S = sim["dir"], sim["S"]
    v = S["variants"]
    rng = np.random.default_rng(3)
    a = pd.read_csv(d / "annot.tsv", sep="\t")
    many = a[["SNP", "low", "high"]].copy()
    for k in range(6):                                                   # 2 x 2^6 = 128 combinations at most
        many[f"f{k}"] = (rng.random(len(many)) < 0.5).astype(int)
    many.to_csv(tmp_path / "many.tsv", sep="\t", index=False)
    n_comb = len(np.unique(many[["low", "high"] + [f"f{k}" for k in range(6)]].to_numpy(), axis=0))
    assert n_comb > 64
    args = (str(d / "ref"), [str(d / "ss")], str(tmp_path / "many.tsv"), str(tmp_path / "o"))
    with pytest.raises(ValueError, match=f"form {n_comb} distinct combinations .* more than --max-annotation-atoms "
                                         r"64.*GiB"):
        compose_ldscores(*args, generations=S["generations"], dry_run=True, log=lambda *a: None)
    plan = compose_ldscores(*args, generations=S["generations"], dry_run=True, max_annotation_atoms=n_comb,
                            log=lambda *a: None)
    assert plan["mode"] == "date"
    assert main(["admixed-ld", "compose", "--ref", str(d / "ref"), "--sumstats", str(d / "ss"), "--annot",
                 str(tmp_path / "many.tsv"), "--admixture-generations", str(S["generations"]), "--dry-run",
                 "--max-annotation-atoms", str(n_comb), "--out", str(tmp_path / "cli")]) == 0
    assert len(v) == len(many)


def test_release_coverage_flags_traits_measured_in_part_of_the_cohort(sim, tmp_path):
    """A trait whose GWAS N is below 85% of the curve's people is flagged (gwas_sample_smaller_than_release); traits
    of the curve's size are not."""
    import shutil
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d = sim["dir"]
    quiet = lambda *a, **k: None
    cv = str(tmp_path / "cv")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5,
                   sample_factors=False, log=quiet)
    n = json.load(open(cv + ".curve.json"))["n_samples"]
    os.makedirs(tmp_path / "ss")
    for t in range(3):
        shutil.copy(d / "ss" / f"trait{t}.tsv", tmp_path / "ss" / f"trait{t}.tsv")
    flags = lambda out: {f["flag"]: f for f in json.load(open(out + ".admixed_ld.json"))["flags"]}
    out = str(tmp_path / "all")
    compose_ldscores(str(d / "ref"), [str(tmp_path / "ss")], str(d / "annot.tsv"), out, curve=cv, log=quiet)
    assert "gwas_sample_smaller_than_release" not in flags(out)
    t0 = pd.read_csv(tmp_path / "ss" / "trait0.tsv", sep="\t")
    t0["N"] = int(0.5 * n)
    t0.to_csv(tmp_path / "ss" / "trait0.tsv", sep="\t", index=False)
    out = str(tmp_path / "part")
    compose_ldscores(str(d / "ref"), [str(tmp_path / "ss")], str(d / "annot.tsv"), out, curve=cv, log=quiet)
    f = flags(out)["gwas_sample_smaller_than_release"]
    assert "trait0" in f["detail"] and "trait1" not in f["detail"] and f["min_share"] == 0.85


def test_release_share_is_of_the_statistic_used(sim, tmp_path):
    """A trait measured in half of the cohort is flagged against the release's people when composed with the curve's
    statistic, and not when composed with the statistic of its own people (the flag's remedy): the share is that of
    the people of the statistic in use."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    from summit.admixed_ld.sample_factors import sample_factors
    d, S = sim["dir"], sim["S"]
    quiet = lambda *a, **k: None
    v = S["variants"]
    cv = str(tmp_path / "cv")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5, cells_cm=0,
                   log=quiet)
    n = json.load(open(cv + ".curve.json"))["n_samples"]
    rng = np.random.default_rng(21)
    idx = np.sort(rng.choice(n, n // 2, replace=False))
    G = S["cohort"][idx]
    iid = write_bed(str(tmp_path / "half"), G, v, "h")
    C = principal_components(G, 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(tmp_path / "half.cov", sep="\t",
                                                                                   index=False)
    os.makedirs(tmp_path / "ss")
    beta, se = gwas(G, rng.standard_normal(idx.size), C)
    pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": idx.size, "BETA": beta,
                  "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(tmp_path / "ss" / "half.tsv", sep="\t", index=False)
    sample_factors(str(tmp_path / "half"), str(d / "ref"), str(tmp_path / "half"), covar=str(tmp_path / "half.cov"),
                   ld_mass=False, log=quiet)
    flags = lambda p: {f["flag"]: f for f in p["flags"]}
    kw = dict(curve=cv, mean_ancestry="0.6", kappa0="0.3", dry_run=True, log=quiet)
    args = (str(d / "ref"), [str(tmp_path / "ss")], str(d / "annot.tsv"))
    rel = compose_ldscores(*args, str(tmp_path / "rel"), **kw)            # the plan lists the flags' names
    assert "gwas_sample_smaller_than_release" in rel["flags"]
    own = compose_ldscores(*args, str(tmp_path / "own"), sample_factors=str(tmp_path / "half.sample_factors.json"), **kw)
    assert "gwas_sample_smaller_than_release" not in own["flags"]
    assert own["same_person_term"].startswith("the sample statistic") and "released" not in own["same_person_term"]
    # composed in full: the flag records the people it was measured against
    out = str(tmp_path / "rel_full")
    compose_ldscores(*args, out, **dict(kw, dry_run=False))
    g = flags(json.load(open(out + ".admixed_ld.json")))["gwas_sample_smaller_than_release"]
    assert g["against"] == "release" and g["people"] == n and f"0.50 of {n}" in g["detail"]


def test_stochastic_ld_mass_agrees_with_the_exact_one(sim, tmp_path):
    """The realized LD mass from random probes (exact diagonals, Rademacher probes for the off-diagonal parts) is
    within a few of its recorded standard errors of the exact mass, in total and between chromosomes."""
    from summit.admixed_ld.sample_factors import sample_factors
    d = sim["dir"]
    quiet = lambda *a, **k: None
    kw = dict(covar=str(d / "cohort.cov"), log=quiet)
    ex = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "ex"), **kw)["ld_mass"]
    st = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "st"), ld_mass_probes=300, seed=4,
                        **kw)["ld_mass"]
    assert ex["method"] == "exact" and st["method"] == "stochastic" and st["probes"] == 300
    for k, se in (("total", "se_total"), ("between_chromosomes", "se_between_chromosomes")):
        assert 0 < st[se] < 0.05 * abs(ex["total"]), (k, st[se], ex["total"])   # the between part is small here
        assert abs(st[k] - ex[k]) < 4.0 * st[se], (k, st[k], ex[k], st[se])


def test_calibrated_sample_factor_term_scales_with_the_gwas_sample(sim, tmp_path):
    """The cohort's statistic composed for a GWAS of 60% of its people: the calibration factor is the same as for the
    whole sample (it is fixed at the statistic's own nu), and the calibrated same-person term scales as 1 / nu, as the
    uncalibrated term does."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.sample_factors import sample_factors
    d, S = sim["dir"], sim["S"]
    v = S["variants"]
    rng = np.random.default_rng(12)
    idx = np.sort(rng.choice(S["cohort"].shape[0], 400, replace=False))
    G = S["cohort"][idx]
    iid = write_bed(str(tmp_path / "sub"), G, v, "s")
    C = principal_components(G, 10)
    pd.DataFrame({"FID": iid, "IID": iid, **{f"PC{i + 1}": C[:, i] for i in range(10)}}).to_csv(
        tmp_path / "sub.cov", sep="\t", index=False)
    beta, se = gwas(G, rng.standard_normal(400), C)
    for n in (400, 240):
        os.makedirs(tmp_path / f"ss{n}")
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": n, "BETA": beta, "SE": se,
                      "Z": beta / se, "COV_RANK": 10}).to_csv(tmp_path / f"ss{n}" / "t0.tsv", sep="\t", index=False)
    p = G.mean(0) / 2
    cls = (np.minimum(p, 1 - p) >= 0.2).astype(int)
    pd.DataFrame({"SNP": v.SNP, "low": (cls == 0).astype(int), "high": (cls == 1).astype(int)}).to_csv(
        tmp_path / "annot.tsv", sep="\t", index=False)
    sample_factors(str(tmp_path / "sub"), str(d / "ref"), str(tmp_path / "sub"), covar=str(tmp_path / "sub.cov"),
                   log=lambda *a: None)
    got = {}
    for n in (400, 240):
        out = str(tmp_path / f"c{n}")
        compose_ldscores(str(d / "ref"), [str(tmp_path / f"ss{n}")], str(tmp_path / "annot.tsv"), out,
                         generations=S["generations"], mean_ancestry="0.6", kappa0="0.3",
                         sample_factors=str(tmp_path / "sub.sample_factors.json"), log=lambda *a: None)
        pj = json.load(open(out + ".admixed_ld.json"))
        got[n] = dict(pj["sample_factors"], gwas_nu=float(pj["nu"]))
    assert got[400]["calibration"] is not None and got[240]["calibration"] is not None
    f4, f2 = got[400]["calibration"]["factor"], got[240]["calibration"]["factor"]
    assert abs(f2 / f4 - 1) < 1e-9, (f4, f2)
    r_nu = got[400]["gwas_nu"] / got[240]["gwas_nu"]
    for k in ("same_person_mass", "same_person_mass_uncalibrated"):
        assert abs((got[240][k] / got[400][k]) / r_nu - 1) < 1e-9, (k, got[240][k] / got[400][k], r_nu)


def test_sample_factors_ld_mass_and_its_check_in_compose(sim, tmp_path):
    """The statistic's realized in-sample LD mass (total, and over pairs on different chromosomes) equals a brute-force
    computation from the genotypes; compose records its own mass against it and, with calibrate_between_chromosomes,
    matches it by a uniform term on pairs of SNPs on different chromosomes."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.sample_factors import sample_factors
    d, S = sim["dir"], sim["S"]
    G, C = S["cohort"], sim["C"]
    info = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "sf"), covar=str(d / "cohort.cov"),
                          log=lambda *a: None)
    n = G.shape[0]
    X = G - G.mean(0)
    s2 = (X ** 2).sum(0) / (n - 1)
    ok = s2 > 0
    X = X[:, ok] / np.sqrt(s2[ok])
    Q = np.linalg.qr(np.column_stack([np.ones(n), C]))[0]
    nu = n - Q.shape[1]
    X -= Q @ (Q.T @ X)
    R = X.T @ X / nu
    V = np.diag(R).copy()
    P = R ** 2 - np.outer(V, V) / nu
    chrom = S["variants"].CHR.to_numpy()[ok]
    diff = chrom[:, None] != chrom[None, :]
    np.testing.assert_allclose(info["ld_mass"]["total"], P.sum(), rtol=1e-9)
    np.testing.assert_allclose(info["ld_mass"]["between_chromosomes"], P[diff].sum(), rtol=1e-9)
    L_in, V_in = insample_ldscores(G[:, ok], C, sim["cls"][ok], 2)
    np.testing.assert_allclose(info["ld_mass"]["total"], float((V_in[:, None] * L_in).sum()), rtol=1e-6)
    sf = str(tmp_path / "sf.sample_factors.json")
    for arm, cal in (("check", False), ("cal", True)):
        out = str(tmp_path / arm)
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), out, generations=S["generations"],
                         sample_factors=sf, calibrate_between_chromosomes=cal, log=lambda *a: None)
        chk = json.load(open(out + ".admixed_ld.json"))["ld_mass_check"]
        got = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
        mass = float((got.V.to_numpy()[:, None] * got[["low", "high"]].to_numpy()).sum())
        if cal:
            assert abs(mass / chk["realized_total"] - 1) < 1e-6         # GWAS N = the statistic's n here
            assert chk["between_chromosome_shift_per_pair"] == pytest.approx(
                (chk["realized_total"] - chk["composed_total"]) / chk["shift_mass"] * chk["between_chromosome_shift_per_pair"])
        else:
            np.testing.assert_allclose(chk["composed_total"], mass, rtol=1e-6)   # the file holds about 7 digits
            assert 0.8 < chk["ratio"] < 1.25


def test_compose_weighs_a_stochastic_ld_mass_by_its_standard_error(sim, tmp_path):
    """A realized LD mass from probes is recorded in ld_mass_check with its method and SEs; the 5% flag
    ld_mass_differs_from_cohort needs the difference to exceed 2 SE as well; and --calibrate-between-chromosomes is
    skipped (flag between_chromosome_shift_skipped) unless the SE of the total is at most 10% of the realized
    between-chromosome mass, as an exact mass always allows."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.sample_factors import sample_factors
    d, S = sim["dir"], sim["S"]
    quiet = lambda *a, **k: None
    sf_ex = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "ex"), covar=str(d / "cohort.cov"),
                           log=quiet)
    sf_st = sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "st"), covar=str(d / "cohort.cov"),
                           ld_mass_probes=3, seed=2, log=quiet)
    run = lambda sf, out, cal=False: (compose_ldscores(
        str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / out), generations=S["generations"],
        mean_ancestry="0.6", kappa0="0.3", sample_factors=sf, calibrate_between_chromosomes=cal, log=quiet),
        json.load(open(str(tmp_path / out) + ".admixed_ld.json")))[1]
    flags = lambda pj: {f["flag"]: f["detail"] for f in pj["flags"]}
    # 3 probes: the SE of the total is far above 10% of the between-chromosome mass, so no shift
    st = run(str(tmp_path / "st.sample_factors.json"), "st", cal=True)
    chk = st["ld_mass_check"]
    assert chk["method"] == "stochastic" and chk["probes"] == 3
    assert chk["se_realized_total"] > 0.1 * chk["realized_between_chromosomes"]
    assert "between_chromosome_shift_per_pair" not in chk
    assert "stochastic" in flags(st)["between_chromosome_shift_skipped"]
    # the exact mass (and a stochastic one with a small SE) is shifted
    ex = run(str(tmp_path / "ex.sample_factors.json"), "ex", cal=True)
    assert ex["ld_mass_check"]["method"] == "exact" and ex["ld_mass_check"]["se_realized_total"] is None
    assert "between_chromosome_shift_per_pair" in ex["ld_mass_check"]
    assert "between_chromosome_shift_skipped" not in flags(ex)
    btw = sf_ex["ld_mass"]["between_chromosomes"]                     # negative here: the SE is compared with |.|
    small = dict(sf_st, ld_mass=dict(sf_st["ld_mass"], between_chromosomes=btw, se_total=1e-3 * abs(btw)))
    json.dump(small, open(tmp_path / "small.sample_factors.json", "w"))
    sm = run(str(tmp_path / "small.sample_factors.json"), "small", cal=True)
    assert "between_chromosome_shift_per_pair" in sm["ld_mass_check"]
    # the 5% flag: a realized total 8% above the composed one, exact or stochastic with a small SE, is flagged; with
    # an SE of 10% of the total (2 SE = 20%) it is not
    base = ex["ld_mass_check"]
    m2 = base["realized_total"] / sf_ex["ld_mass"]["total"]             # the SNP-count factor compose applies
    tot = 1.08 * base["composed_total"] / m2
    for name, extra, flagged in (("exact8", dict(total=tot), True),
                                 ("stoch8", dict(total=tot, method="stochastic", probes=10, se_total=1e-3 * tot), True),
                                 ("noisy8", dict(total=tot, method="stochastic", probes=10, se_total=0.1 * tot), False)):
        json.dump(dict(sf_ex, ld_mass=dict(sf_ex["ld_mass"], **extra)), open(tmp_path / f"{name}.sample_factors.json", "w"))
        pj = run(str(tmp_path / f"{name}.sample_factors.json"), name)
        assert abs(pj["ld_mass_check"]["ratio"] - 1 / 1.08) < 1e-9
        assert ("ld_mass_differs_from_cohort" in flags(pj)) is flagged, name


def test_between_chromosome_shift_without_a_realized_ld_mass_is_skipped_with_a_flag(sim, tmp_path):
    """--calibrate-between-chromosomes needs the realized LD mass: with the curve's statistic made without it (the
    curve's default) or with no statistic, compose writes the uncalibrated LD scores, warns and records the flag
    between_chromosome_shift_skipped with the reason."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.curve import genotype_curve
    d, S = sim["dir"], sim["S"]
    quiet = lambda *a, **k: None
    cv = str(tmp_path / "cv")
    genotype_curve(str(d / "cohort"), str(d / "ref"), cv, covar=str(d / "cohort.cov"), cell_cm=0.5, cells_cm=0,
                   log=quiet)
    assert json.load(open(cv + ".sample_factors.json"))["ld_mass"] is None
    args = (str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"))
    kw = dict(mean_ancestry="0.6", kappa0="0.3", allow_flat_kernel_join=True)
    for arm, extra, why in (("curve", dict(curve=cv), "records no realized LD mass"),
                            ("date", dict(generations=S["generations"]), "uses no sample statistic")):
        got, lines = {}, []
        for cal in (False, True):
            out = str(tmp_path / f"{arm}_{cal}")
            compose_ldscores(*args, out, calibrate_between_chromosomes=cal, **extra, **kw,
                             log=lambda *a, **k: lines.append(" ".join(map(str, a))))
            got[cal] = pd.read_csv(out + ".gw.ldscore.gz", sep="\t")
        fl = {f["flag"]: f["detail"] for f in json.load(open(str(tmp_path / f"{arm}_True") + ".admixed_ld.json"))["flags"]}
        assert why in fl["between_chromosome_shift_skipped"] and "--sample-factors-ld-mass" in fl[
            "between_chromosome_shift_skipped"]
        assert any("WARNING (recorded flag between_chromosome_shift_skipped)" in ln for ln in lines)
        pd.testing.assert_frame_equal(got[False], got[True])


def test_h2_over_ldscore_groups_checks_each_traits_sample_size(fam_groups, tmp_path):
    """--h2 with compose's n_groups.tsv stops when a trait's summary statistics have another median N than the table
    gives it (beyond compose's 1%): a GWAS rerun on other people under the same file name is not fitted with the
    earlier sample's LD scores. Within the tolerance it runs."""
    import shutil
    fg = fam_groups
    for k, (scale, ok) in enumerate(((0.9, False), (1.004, True))):
        mix = tmp_path / f"mix{k}"
        shutil.copytree(fg["dir"] / "mix", mix)
        t = pd.read_csv(mix / "part0.tsv", sep="\t")
        t["N"] = np.round(t.N * scale).astype(int)
        t.to_csv(mix / "part0.tsv", sep="\t", index=False)
        for h2 in (mix, mix / "part0.tsv"):                             # a directory, and one file
            out = str(tmp_path / f"h{k}_{h2.name}")
            r = _summit_h2(fg, "--h2", str(h2), "--ldscores", fg["prefix"] + ".gw.ldscore.gz", "--out", out,
                           check=False)
            assert (r.returncode == 0) is ok, r.stdout[-2000:] + r.stderr[-2000:]
            if not ok:
                msg = r.stdout + r.stderr
                assert f"median N {int(round(fg['n_sub'] * scale))}" in msg and "part0.tsv the LD scores composed for N" \
                    f" {fg['n_sub']}" in msg, msg[-2000:]
                assert not os.path.exists(out + ".results.tsv")


def test_h2_with_one_file_removes_an_earlier_group_runs_outputs(fam_groups, tmp_path):
    """A --h2 run with one LD-score file to an --out where a run over LD-score groups left OUT.ldscore_groups.tsv and
    OUT.ldgroups/ removes them (logged), so they cannot describe the new results; files that --h2 did not make in
    OUT.ldgroups/ are left in place."""
    fg = fam_groups
    out = str(tmp_path / "h")
    main = fg["prefix"] + ".gw.ldscore.gz"
    for extra in (None, "user.txt"):
        _summit_h2(fg, "--h2", str(fg["dir"] / "mix"), "--ldscores", main, "--out", out)
        assert os.path.exists(out + ".ldscore_groups.tsv") and os.path.isdir(out + ".ldgroups")
        if extra:
            open(os.path.join(out + ".ldgroups", extra), "w").close()
        r = _summit_h2(fg, "--h2", str(fg["dir"] / "mix"), "--ldscores", main, "--ldscore-groups", "off", "--out", out)
        assert not os.path.exists(out + ".ldscore_groups.tsv")
        log = open(out + ".log").read()
        assert f"removed {out}.ldscore_groups.tsv" in log
        if extra:
            assert os.listdir(out + ".ldgroups") == [extra] and "is left in place" in log
        else:
            assert not os.path.exists(out + ".ldgroups") and f"removed {out}.ldgroups/" in log
        assert len(pd.read_csv(out + ".results.tsv", sep="\t")) == 5 and r.returncode == 0


def test_overlapping_annotation_columns_in_curve_mode(fam, fam_groups, tmp_path):
    """Overlapping columns in full curve mode (the sample statistic with its realized LD mass, the cell matrix, the
    relatives' profile, the between-chromosome shift and one file per GWAS sample size) equal the sums of their
    combinations composed as a partition, in the main file and in each sample size's file."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.sample_factors import sample_factors
    d, fg = fam["dir"], fam_groups
    quiet = lambda *a, **k: None
    sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "sf"), covar=str(d / "cohort.cov"), log=quiet)
    a = pd.read_csv(d / "annot.tsv", sep="\t")
    func = (np.arange(len(a)) % 3 == 0).astype(int)
    pd.DataFrame({"SNP": a.SNP, "low": a.low, "high": a.high, "func": func}).to_csv(tmp_path / "over.tsv", sep="\t",
                                                                                   index=False)
    pd.DataFrame({"SNP": a.SNP, "low0": a.low * (1 - func), "low1": a.low * func, "high0": a.high * (1 - func),
                  "high1": a.high * func}).to_csv(tmp_path / "atoms.tsv", sep="\t", index=False)
    kw = dict(curve=str(d / "c"), relatedness="curve", cell_matrix="auto", calibrate_between_chromosomes=True,
              sample_factors=str(tmp_path / "sf.sample_factors.json"), log=quiet)
    pj = {}
    for nm in ("over", "atoms"):
        compose_ldscores(str(d / "ref"), [str(fg["dir"] / "mix")], str(tmp_path / f"{nm}.tsv"), str(tmp_path / nm),
                         **kw)
        pj[nm] = json.load(open(str(tmp_path / nm) + ".admixed_ld.json"))
    p = pj["over"]
    assert p["plan"]["components"] == dict(curve=True, statistic=True, cells=True, relatives=True, background=False)
    assert "between_chromosome_shift_per_pair" in p["ld_mass_check"] and len(p["n_groups"]) == 2
    np.testing.assert_allclose(p["ld_mass"], pj["atoms"]["ld_mass"], rtol=1e-10)
    tab = {nm: pd.read_csv(str(tmp_path / nm) + ".n_groups.tsv", sep="\t") for nm in pj}
    pairs = [("over.gw.ldscore.gz", "atoms.gw.ldscore.gz")]
    for N_ in sorted(tab["over"].N.unique()):
        pairs.append(tuple(tab[nm][tab[nm].N == N_].ldscores.iloc[0] for nm in ("over", "atoms")))
    assert len(pairs) == 3
    for fo, fa in pairs:
        lo, la = pd.read_csv(tmp_path / fo, sep="\t"), pd.read_csv(tmp_path / fa, sep="\t")
        np.testing.assert_array_equal(lo.V, la.V)
        for col, parts in (("low", ["low0", "low1"]), ("high", ["high0", "high1"]), ("func", ["low1", "high1"])):
            # the files hold six decimals: a column and its two rounded parts differ by at most 1.5e-6
            np.testing.assert_allclose(lo[col], la[parts].sum(axis=1), rtol=1e-12, atol=1.6e-6)
        mo = pd.read_csv(tmp_path / fo.replace(".gw.ldscore.gz", ".gw.M"), sep="\t", header=None).to_numpy().ravel()
        ma = pd.read_csv(tmp_path / fa.replace(".gw.ldscore.gz", ".gw.M"), sep="\t", header=None).to_numpy().ravel()
        np.testing.assert_array_equal(mo, [ma[0] + ma[1], ma[2] + ma[3], ma[1] + ma[3]])
