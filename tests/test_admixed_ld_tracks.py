"""Tests of the ancestry-track correlation (`summit admixed-ld tractor-sumstats`, `compose --tracks`, `track-rg`): the
composed track LD scores against the cohort's exact ones, the own scores unchanged by the new options, recovery of a
known correlation of per-allele effects across local-ancestry tracks (Hou et al.'s r_admix) and of each track with a
disjoint European GWAS, sandwich against OLS standard errors, dense partners and partitioned classes. The cohort, its
known local ancestry, the Tractor regressions and the exact scores are made in a child process (the cross-trait GxE
tests plan their memory from the pytest process's peak); the test process composes and fits only."""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from admixed_ld_sim import in_child  # noqa: E402
from admixed_ld_tracks_sim import build_all  # noqa: E402

# per scenario: per-allele effects correlated rho across the tracks; 'dense' puts them on every eligible variant (3 per
# GWAS SNP), 'hetero' lets the residual variance rise with ancestry, 'bins' gives the low-MAF class 3 times the
# per-allele variance; the European GWAS of 'sparse' has effects correlated 0.8 with the E track's
TRAITS = dict(sparse=dict(rho=0.6, reps=8, r_ex=0.8, sA=1.2, sE=0.8),
              dense=dict(rho=0.6, reps=8, causal="all"),
              hetero=dict(rho=0.6, reps=6, eta=0.6, h2=0.3, sA=1.5, sE=0.75),
              bins=dict(rho=0.6, reps=6, causal="all", classes=(3.0, 1.0)))
QUIET = lambda *a: None


@pytest.fixture(scope="module")
def trk(tmp_path_factory):
    """The simulated cohort (2,500 admixed people with known local ancestry, two 20-cM chromosomes, 1,459 eligible
    variants of which a third are the GWAS SNPs), panels of 500 per source and a European panel of other people, and a
    composition cache keyed by its options."""
    d = str(tmp_path_factory.mktemp("admixed_ld_tracks"))
    truth = in_child(build_all, d=d, traits=TRAITS)
    from summit.admixed_ld.tracks import tractor_sumstats
    rec = tractor_sumstats(f"{d}/tractor/sparse.0.tsv", f"{d}/ts0", sources=["A", "E"], n_covariates=1, log=QUIET)
    cache = {}

    def compose(annot="annot1", partners="rows", cross=None, alpha=None, tracks=True):
        from summit.admixed_ld.compose import compose_ldscores
        key = (annot, partners, cross, alpha, tracks)
        if key not in cache:
            o = f"{d}/c_{annot}_{partners}_{cross}_{alpha}_{tracks}"
            p = compose_ldscores(f"{d}/ref", [f"{d}/ts0.A.sumstats"], f"{d}/{annot}.tsv", o, sources=["A", "E"],
                                 generations=truth["generations"], mean_ancestry=rec["mean_ancestry_arg"],
                                 theta_sd=truth["theta_sd"], partners=partners, cross_source=cross, tracks=tracks,
                                 track_alpha=alpha, log=QUIET)
            cache[key] = (o, p)
        return cache[key]

    return dict(dir=d, truth=truth, rec=rec, compose=compose)


def _fits(trk, scenario, comp, annot, se="hc", external=None, cohort_track=None, groups=None, njack=10):
    """track-rg of every trait of a scenario; returns the per-trait summaries."""
    from summit.admixed_ld.tracks import track_rg, tractor_sumstats
    d = trk["dir"]
    out = []
    for r in range(trk["truth"]["scenarios"][scenario]["reps"]):
        pre = f"{d}/ts_{scenario}{r}_{se}"
        if not os.path.exists(f"{pre}.tracks.json"):
            tractor_sumstats(f"{d}/tractor/{scenario}.{r}.tsv", pre, sources=["A", "E"], n_covariates=1, se=se,
                             allow_ols=se == "ols", log=QUIET)
        out.append(track_rg(comp, pre, f"{d}/{annot}.tsv", f"{pre}.{os.path.basename(comp)}", njack=njack,
                            allow_ols=se == "ols", external=None if external is None else
                            f"{d}/eur/{scenario}.{r}.sumstats", cohort_track=cohort_track, groups=groups,
                            keep_ambiguous=True, log=QUIET))
    return out


def _mean(fits, f):
    return float(np.mean([f(x) for x in fits]))


def test_track_scores_match_exact(trk):
    """The composed track LD scores (own A, own E, cross) against the cohort's exact in-sample ones (its own
    within-track allele deviations from the true local ancestry, U-statistics in its own people), with the GWAS SNPs
    as partners and with every eligible variant as partners: mass within 3%, SNP by SNP as close as 500-person panels
    allow, the V columns m v w within 2%; the partners' counts in .gw.M."""
    d = trk["dir"]
    ex = np.load(f"{d}/exact.npz")
    for partners, key in (("rows", "rows"), ("reference", "all")):
        o, p = trk["compose"](partners=partners)
        tb = p["tracks"]
        assert tb["n_rows"] == trk["truth"]["n_rows"]
        assert tb["n_partners"] == (trk["truth"]["n_rows"] if partners == "rows" else trk["truth"]["n_eligible"])
        for name, col, vk in (("track_A", "A", "VA"), ("track_E", "E", "VE"), ("track_A_E", "AE", None)):
            t = pd.read_csv(f"{o}.{name}.gw.ldscore.gz", sep="\t").set_index("SNP").reindex(ex["snps"])
            e = ex[f"{key}_{col}"]
            V = t.V.to_numpy()
            Ve = ex[f"{key}_{vk}"] if vk else np.sqrt(ex[f"{key}_VA"] * ex[f"{key}_VE"])
            assert abs(np.mean(V / Ve) - 1) < 0.02, (partners, name)
            ratio = float((V * t["all"]).sum() / (V * e).sum())
            assert abs(ratio - 1) < 0.03, (partners, name, ratio)
            assert np.corrcoef(t["all"], e)[0, 1] > 0.85, (partners, name)
            with open(f"{o}.{name}.gw.M") as fh:
                assert float(fh.read().split()[0]) == tb["n_partners"]
        assert tb["files"]["track_A_E"]["ld_mass"] > 0


def test_own_scores_unchanged(trk):
    """Asking for the track scores (and the cross-population ones) leaves the ordinary composed LD scores exactly as
    they were; with the partners set to a reference whose annotated variants are the GWAS SNPs, the dense code path
    gives the same scores."""
    o0, p0 = trk["compose"](tracks=False)
    o1, p1 = trk["compose"]()
    o2, p2 = trk["compose"](cross="E")
    read = lambda o: gzip.open(f"{o}.gw.ldscore.gz").read()
    assert read(o0) == read(o1) == read(o2)
    assert p0["ld_mass"] == p1["ld_mass"] == p2["ld_mass"]
    # an annotation listing only the GWAS SNPs: the reference partners are the rows
    d = trk["dir"]
    a = pd.read_csv(f"{d}/annot1.tsv", sep="\t")
    rows = pd.read_csv(f"{d}/ts0.A.sumstats", sep="\t").SNP
    a[a.SNP.isin(rows)].to_csv(f"{d}/annot_rows.tsv", sep="\t", index=False)
    o3, p3 = trk["compose"](annot="annot_rows", partners="reference")
    o4, p4 = trk["compose"](annot="annot_rows")
    for name in ("", ".track_A", ".track_E", ".track_A_E"):
        x, y = (pd.read_csv(f"{o}{name}.gw.ldscore.gz", sep="\t") for o in (o3, o4))
        pd.testing.assert_frame_equal(x, y, rtol=1e-6)
    assert p3["ld_mass"] == pytest.approx(p4["ld_mass"], rel=1e-10)


def test_track_rg_recovers_r_admix(trk):
    """Per-allele effects correlated 0.6 across the tracks (track variances 1.44 and 0.64 per allele), causal at the
    GWAS SNPs: the mean over traits of r covers 0.6; with the GWAS SNPs as partners r is SUMMIT's own cross-mode
    rg_total; two MAF classes give the same r within noise and the group of both classes is r itself; the cross
    intercept is zero."""
    comp1, _ = trk["compose"]()
    f1 = _fits(trk, "sparse", comp1, "annot1")
    r1 = [x["pairs"]["A_E"]["r"][0] for x in f1]
    assert abs(np.mean(r1) - 0.6) < 0.1, r1
    for x in f1:
        pr = x["pairs"]["A_E"]
        assert pr["r"][0] == pytest.approx(pr["summit_rg_total"][0], rel=1e-9)
        assert pr["r"][1] == pytest.approx(pr["summit_rg_total"][1], rel=0.1)
        assert pr["track_h2"]["A"][0] > pr["track_h2"]["E"][0] > 0
    comp2, _ = trk["compose"](annot="annot")
    f2 = _fits(trk, "sparse", comp2, "annot", groups=["both=low,high"])
    r2 = [x["pairs"]["A_E"]["r"][0] for x in f2]
    assert abs(np.mean(r2) - np.mean(r1)) < 0.05
    for x in f2:
        pr = x["pairs"]["A_E"]
        assert pr["groups"]["both"]["r"][0] == pytest.approx(pr["r"][0], rel=1e-12)
        assert pr["r"][0] == pytest.approx(pr["summit_rg_total"][0], rel=1e-9)
        assert set(pr["classes"]) == {"low", "high"}


def test_triangle_with_a_european_gwas(trk):
    """The tracks against a GWAS of other Europeans whose effects correlate 0.8 with the E track's: r(E, EUR) covers
    0.8, r(A, EUR) covers 0.48 = r_admix r(E, EUR), the triangle check is centred on zero, and the European panel shared
    with the E source (its square unbiased) agrees with an independent European panel."""
    comp_e, pe = trk["compose"](cross="E")
    comp_x, px = trk["compose"](cross="X")
    assert {"track_A_vs_E", "track_E_vs_E", "cohort_E"} <= set(pe["tracks"]["files"])
    fe = _fits(trk, "sparse", comp_e, "annot1", external=True)
    fx = _fits(trk, "sparse", comp_x, "annot1", external=True, cohort_track="E")
    for fits in (fe, fx):
        assert abs(_mean(fits, lambda x: x["external"]["tracks"]["E"]["r"][0]) - 0.8) < 0.1
        assert abs(_mean(fits, lambda x: x["external"]["tracks"]["A"]["r"][0]) - 0.48) < 0.1
        assert abs(_mean(fits, lambda x: x["external"]["triangle_check"]["difference"])) < 0.08
        assert all(np.isfinite(x["external"]["triangle_check"]["difference_se"]) for x in fits)
    for s in ("A", "E"):
        a = _mean(fe, lambda x: x["external"]["tracks"][s]["r"][0])
        b = _mean(fx, lambda x: x["external"]["tracks"][s]["r"][0])
        assert abs(a - b) < 0.05, (s, a, b)


def test_sandwich_against_ols_standard_errors(trk):
    """Residual variance that rises with ancestry (eta 0.6): with sandwich SEs r covers the truth; with OLS SEs the
    own-track moments carry the heteroscedasticity (the E track's h2 collapses, r is far off). The reader refuses OLS
    input unless asked and flags it; track-rg refuses OLS sumstats unless asked."""
    from summit.admixed_ld.tracks import track_rg, tractor_sumstats
    d = trk["dir"]
    comp, _ = trk["compose"]()
    hc = _fits(trk, "hetero", comp, "annot1")
    ols = _fits(trk, "hetero", comp, "annot1", se="ols")
    r_hc = np.array([x["pairs"]["A_E"]["r"][0] for x in hc])
    r_ols = np.array([x["pairs"]["A_E"]["r"][0] for x in ols])     # NaN where the E track's h2 falls below 0
    hE = lambda f: _mean(f, lambda x: x["pairs"]["A_E"]["track_h2"]["E"][0])
    print(f"\nvariance rising with ancestry (truth 0.6): sandwich SEs r {np.mean(r_hc):.4f}, OLS SEs r "
          f"{np.nanmean(r_ols):.4f} ({int(np.isnan(r_ols).sum())} undefined); E track h2 {hE(hc):.4f} against "
          f"{hE(ols):.4f}")
    assert np.isfinite(r_hc).all() and abs(np.mean(r_hc) - 0.6) < 0.15, r_hc
    assert np.isnan(r_ols).any() or np.nanmean(r_ols) - 0.6 > 0.2, r_ols
    assert hE(ols) < 0.6 * hE(hc)
    assert any(f["flag"] == "ols_standard_errors" for f in ols[0]["flags"])
    with pytest.raises(ValueError, match="sandwich"):
        tractor_sumstats(f"{d}/tractor/hetero.0.tsv", f"{d}/refused", sources=["A", "E"], se="ols", log=QUIET)
    with pytest.raises(ValueError, match="OLS standard errors"):
        track_rg(comp, f"{d}/ts_hetero0_ols", f"{d}/annot1.tsv", f"{d}/refused", njack=10, log=QUIET)
    t = pd.read_csv(f"{d}/tractor/hetero.0.tsv", sep="\t").drop(columns=["se_hc_anc0", "se_hc_anc1"])
    t.to_csv(f"{d}/no_hc.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="no sandwich SE column"):
        tractor_sumstats(f"{d}/no_hc.tsv", f"{d}/refused", sources=["A", "E"], log=QUIET)


def test_dense_partners(trk):
    """Causal variants on every eligible variant, three per GWAS SNP: with the GWAS SNPs as partners r is attenuated by
    the factor the scores predict, f_X / sqrt(f_A f_E) with f_t = (rows / partners) sum_j V_j l^dense_j / sum_j V_j
    l^rows_j; with the dense partners r covers the truth. Options measured on the GWAS SNPs are refused."""
    from summit.admixed_ld.compose import compose_ldscores
    d = trk["dir"]
    comp_r, pr = trk["compose"]()
    comp_d, pd_ = trk["compose"](partners="reference")
    rr = _mean(_fits(trk, "dense", comp_r, "annot1"), lambda x: x["pairs"]["A_E"]["r"][0])
    rd = _mean(_fits(trk, "dense", comp_d, "annot1"), lambda x: x["pairs"]["A_E"]["r"][0])
    f = {}
    for name in ("track_A", "track_E", "track_A_E"):
        a, b = (pd.read_csv(f"{o}.{name}.gw.ldscore.gz", sep="\t") for o in (comp_d, comp_r))
        f[name] = (pr["tracks"]["n_partners"] / pd_["tracks"]["n_partners"]) * (a.V * a["all"]).sum() / (b.V * b["all"]).sum()
    pred = f["track_A_E"] / np.sqrt(f["track_A"] * f["track_E"])
    print(f"\ndense causal (truth 0.6): GWAS-SNP partners {rr:.4f}, dense partners {rd:.4f}; predicted factor "
          f"{pred:.4f}, observed {rr / rd:.4f}")
    assert pred < 0.97
    assert abs(rd - 0.6) < 0.1, rd
    assert abs(rr / rd - pred) < 0.08, (rr, rd, pred)
    with pytest.raises(ValueError, match="measured on the GWAS SNPs"):
        compose_ldscores(f"{d}/ref", [f"{d}/ts0.A.sumstats"], f"{d}/annot1.tsv", f"{d}/refused", sources=["A", "E"],
                         generations=8, mean_ancestry=trk["rec"]["mean_ancestry_arg"], theta_sd=0.25,
                         partners="reference", cell_matrix=True, log=QUIET)


def test_partitioned_classes(trk):
    """Two MAF classes, the low-MAF class with three times the per-allele variance, causal everywhere, dense partners:
    r is the classes' per-allele variances weighted by the partners' counts (recomputed from the per-class estimates)
    and covers the truth; with different row and partner counts it is not SUMMIT's rg_total, which weights by rows."""
    comp, p = trk["compose"](annot="annot", partners="reference")
    n = np.asarray(p["tracks"]["partner_counts"])
    with open(f"{comp}.gw.M") as fh:
        assert np.allclose([float(x) for x in fh.read().split()], n)
    fits = _fits(trk, "bins", comp, "annot")
    for x in fits:
        pr = x["pairs"]["A_E"]
        t = {k: np.array([pr["classes"][c][k][0] for c in ("low", "high")]) for k in ("tau_1", "tau_2", "tau_12")}
        assert pr["r"][0] == pytest.approx((t["tau_12"] @ n) / np.sqrt((t["tau_1"] @ n) * (t["tau_2"] @ n)), rel=1e-9)
    r = _mean(fits, lambda x: x["pairs"]["A_E"]["r"][0])
    assert abs(r - 0.6) < 0.12, r
    tau_low = _mean(fits, lambda x: x["pairs"]["A_E"]["classes"]["low"]["tau_1"][0])
    tau_high = _mean(fits, lambda x: x["pairs"]["A_E"]["classes"]["high"]["tau_1"][0])
    assert tau_low > 1.5 * tau_high                                  # the per-class variances follow the truth (3x)


def test_track_alpha_weights(trk):
    """Per-allele weights [2 f (1 - f)]^alpha (f at the mean ancestry) give V^s = m_s v^s w; the default is 1 / s^2 of
    the cohort's model, which differs from 1 / (2 f (1 - f)) only by the admixture variance."""
    from summit.admixed_ld.reference import Reference
    d = trk["dir"]
    o, p = trk["compose"](alpha=-0.38)
    o0, p0 = trk["compose"]()
    m = np.asarray(p["tracks"]["mean_ancestry"])
    rv = Reference(f"{d}/ref").variants(["A", "E"]).set_index("SNP")
    tA = pd.read_csv(f"{o}.track_A.gw.ldscore.gz", sep="\t").set_index("SNP")
    tE = pd.read_csv(f"{o}.track_E.gw.ldscore.gz", sep="\t").set_index("SNP")
    r = rv.reindex(tA.index)
    f = m[0] * r.p_A + m[1] * r.p_E
    w = (2 * f * (1 - f)) ** -0.38
    np.testing.assert_allclose(tA.V, m[0] * r.v_A * w, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(tE.V, m[1] * r.v_E * w, rtol=2e-5, atol=2e-6)
    t0 = pd.read_csv(f"{o0}.track_A.gw.ldscore.gz", sep="\t").set_index("SNP").reindex(tA.index)
    w0 = t0.V / (m[0] * r.v_A)                                       # 1 / s^2
    assert np.median(np.abs(w0 * 2 * f * (1 - f) - 1)) < 0.05
    assert p["tracks"]["alpha"] == -0.38 and "2 f_k" in p["tracks"]["weights"]


def test_tractor_reader(trk):
    """The reader: one SUMMIT file per track with A1 = ALT (Tractor's counted allele), the sandwich SEs, TRACK and
    SE_TYPE; OLS SEs recomputed as beta / tval (se is rounded); COV_RANK = covariates + 2 (K - 1); the mean
    local-ancestry share for --mean-ancestry; the noise check (N SE^2 2 LAprop AF (1 - AF)) agrees between tracks with
    OLS SEs; compose refuses track sumstats without --mean-ancestry."""
    from summit.admixed_ld.compose import compose_ldscores
    from summit.admixed_ld.tracks import tractor_sumstats
    d = trk["dir"]
    raw = pd.read_csv(f"{d}/tractor/sparse.0.tsv", sep="\t")
    rec = trk["rec"]
    assert rec["se_type"] == "HC" and rec["sources"] == ["A", "E"]
    assert rec["tracks"]["A"]["mean_local_ancestry"] == pytest.approx(trk["truth"]["m_A"], abs=0.02)
    for i, s in enumerate(("A", "E")):
        t = pd.read_csv(rec["files"][s], sep="\t")
        assert list(t.columns) == ["SNP", "CHR", "BP", "A1", "A2", "N", "BETA", "SE", "Z", "COV_RANK", "AF", "LA_PROP",
                                   "SE_TYPE", "TRACK"]
        assert (t.A1 == raw.ALT).all() and (t.A2 == raw.REF).all() and (t.TRACK == s).all()
        np.testing.assert_allclose(t.SE, raw[f"se_hc_anc{i}"], rtol=1e-9)
        assert (t.COV_RANK == 3).all()
    ols = tractor_sumstats(f"{d}/tractor/sparse.0.tsv", f"{d}/ts0ols", sources=["A", "E"], se="ols", allow_ols=True,
                           n_covariates=1, log=QUIET)
    t = pd.read_csv(ols["files"]["A"], sep="\t")
    np.testing.assert_allclose(t.SE, raw.beta_anc0 / raw.tval_anc0, rtol=1e-9)
    assert not np.allclose(t.SE, raw.se_anc0, rtol=1e-6)               # se_anc0 is rounded to four decimals
    nA, nE = (ols["tracks"][s]["noise_variance_median"] for s in ("A", "E"))
    assert nA == pytest.approx(nE, rel=0.05) and 0.3 < nA < 1.0
    with pytest.raises(ValueError, match="mean-ancestry"):
        compose_ldscores(f"{d}/ref", [f"{d}/ts0.A.sumstats"], f"{d}/annot1.tsv", f"{d}/refused", sources=["A", "E"],
                         generations=8, tracks=True, log=QUIET)


def test_cli(trk, tmp_path):
    """The three commands from the command line, on one trait."""
    from summit.admixed_ld.__main__ import main
    d, truth = trk["dir"], trk["truth"]
    o = str(tmp_path / "x")
    assert main(["tractor-sumstats", "--tractor", f"{d}/tractor/sparse.1.tsv", "--sources", "A,E", "--covariates", "1",
                 "--out", o]) == 0
    with open(f"{o}.tracks.json") as fh:
        m = json.load(fh)["mean_ancestry_arg"]
    assert main(["compose", "--ref", f"{d}/ref", "--sumstats", f"{o}.A.sumstats", "--annot", f"{d}/annot.tsv",
                 "--sources", "A,E", "--admixture-generations", str(truth["generations"]), "--mean-ancestry", m,
                 "--ancestry-sd", str(truth["theta_sd"]), "--tracks", "--cross-source", "E", "--out", o]) == 0
    assert main(["track-rg", "--tracks", o, "--sumstats", o, "--annot", f"{d}/annot.tsv", "--njack", "10",
                 "--external", f"{d}/eur/sparse.1.sumstats", "--keep-ambiguous", "--group", "all=low,high",
                 "--out", o]) == 0
    with open(f"{o}.track_rg.json") as fh:
        res = json.load(fh)
    assert np.isfinite(res["pairs"]["A_E"]["r"][0]) and "triangle_check" in res["external"]
    tsv = pd.read_csv(f"{o}.track_rg.tsv", sep="\t")
    assert {"r", "r[low]", "r[high]", "r[group all]", "track_h2[A]"} <= set(tsv.what)
    with open(f"{o}.admixed_ld.json") as fh:
        flags = [f["flag"] for f in json.load(fh)["flags"]]
    assert "kappa0_from_ancestry_sd" in flags
