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
