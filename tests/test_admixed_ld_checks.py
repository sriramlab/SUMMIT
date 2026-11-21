"""Input checks of `summit admixed-ld` that must fail closed, or record a flag, with a readable message: genome build or
allele coding, strand, genetic-map units, sex chromosomes, monomorphic panel SNPs, few SNPs, missing genotypes, the
curve's covariates and standard errors of another kind. Small simulations (int8 dosages, a few hundred people) keep
the test process's memory low."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from admixed_ld_sim import gwas, principal_components, simulate, write_bed
from summit.admixed_ld._io import (check_map_rates, check_match, genetic_positions, match_variants, strand_check,
                                   strand_check_fails)
from summit.admixed_ld.compose import compose_ldscores, se_regression_outliers
from summit.admixed_ld.curve import genotype_curve
from summit.admixed_ld.reference import build_reference

_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
quiet = lambda *a: None


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("admixed_ld_checks")
    S = simulate(seed=7, n_chr=2, snps_per_chr=400, chr_cm=60.0, n_panel=300, n_cohort=900)
    v = S["variants"]
    write_bed(str(d / "panelA"), S["panel_A"].astype(np.int8), v, "a")
    write_bed(str(d / "panelE"), S["panel_E"].astype(np.int8), v, "e")
    iid = write_bed(str(d / "cohort"), S["cohort"].astype(np.int8), v, "c")
    C = principal_components(S["cohort"], 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(d / "cohort.cov", sep="\t",
                                                                                   index=False)
    y = np.random.default_rng(1).standard_normal(S["cohort"].shape[0])
    beta, se = gwas(S["cohort"], y, C)
    os.makedirs(d / "ss")
    pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta,
                  "SE": se, "Z": beta / se, "COV_RANK": 2}).to_csv(d / "ss" / "trait0.tsv", sep="\t", index=False)
    pd.DataFrame({"SNP": v.SNP, "all": 1}).to_csv(d / "annot.tsv", sep="\t", index=False)
    build_reference([str(d / "panelA"), str(d / "panelE")], ["AFR", "EUR"], str(d / "ref"), step=256, log=quiet)
    return dict(dir=d, S=S, C=C, iid=iid)


def test_match_report_counts_and_drops_ambiguous():
    t = pd.DataFrame({"CHR": "1", "BP": [1, 2, 3, 4], "SNP": list("abcd"), "COUNTED": ["A", "C", "A", "A"],
                      "OTHER": ["G", "G", "T", "C"]})
    r = t.assign(OTHER=["G", "G", "T", "G"])                  # d disagrees in its alleles
    rep = {}
    ti, ri, fl = match_variants(t, r, report=rep)
    assert ti.tolist() == [0, 1, 2] and rep["n_keys"] == 4 and rep["n_allele_mismatch"] == 1
    assert rep["n_ambiguous"] == 2                             # C/G and A/T
    rep2 = {}
    ti2, _, _ = match_variants(t, r, drop_ambiguous=True, report=rep2)
    assert ti2.tolist() == [0] and rep2["n_ambiguous_dropped"] == 2


def test_another_build_fails_closed():
    rep = dict(n_target=5000, n_reference=5000, n_keys=5000, n_matched=2000, n_allele_mismatch=3000,
               n_ambiguous=0, n_ambiguous_dropped=0, how="position")
    with pytest.raises(ValueError, match="genome builds or the allele coding differ"):
        check_match(rep, "curve: the cohort", quiet)
    msgs = []
    check_match(dict(rep, n_keys=2000, n_matched=2000, n_allele_mismatch=0), "compose: the GWAS", msgs.append)
    assert any("only 40.0%" in m for m in msgs)


def test_genetic_map_in_morgans_or_bp_fails_closed(tmp_path):
    v = pd.DataFrame({"CHR": "1", "BP": np.arange(1, 201) * 200_000, "SNP": [f"s{i}" for i in range(200)]})
    m = pd.DataFrame({"chr": "1", "position": v.BP, "COMBINED_rate(cM/Mb)": 1.0, "Genetic_Map(cM)": v.BP / 1e6})
    m.assign(**{"Genetic_Map(cM)": v.BP / 1e8}).to_csv(tmp_path / "morgans.txt", sep=" ", index=False)
    with pytest.raises(ValueError, match="not Morgans"):
        genetic_positions(v, str(tmp_path / "morgans.txt"), log=quiet)
    m.to_csv(tmp_path / "ok.txt", sep=" ", index=False)
    cm, src = genetic_positions(v, str(tmp_path / "ok.txt"), log=quiet)
    assert np.allclose(cm, v.BP / 1e6)
    with pytest.raises(ValueError, match="cM per Mb"):
        check_map_rates(v.CHR, v.BP, v.BP.astype(float), "a CM column with base pairs")


def test_sex_chromosomes_are_refused(small):
    d = small["dir"]
    with pytest.raises(ValueError, match="autosomes only"):
        build_reference([str(d / "panelA"), str(d / "panelE")], ["AFR", "EUR"], str(d / "refX"), chromosomes=["X"],
                        log=quiet)


def test_strand_check_rule():
    rng = np.random.default_rng(0)
    M = 4000
    pB = rng.uniform(0.05, 0.95, M)
    D = np.clip(pB + rng.normal(0, 0.3, M), 0.02, 0.98)[:, None] - pB[:, None]
    f = pB + 0.6 * D[:, 0] + rng.normal(0, 0.01, M)
    amb = rng.random(M) < 0.15
    ok = strand_check(f, pB, D, amb)
    assert ok["ambiguous_discordant"] < 0.02 and not strand_check_fails(ok)
    bad = strand_check(np.where(amb & (rng.random(M) < 0.5), 1 - f, f), pB, D, amb)
    assert bad["ambiguous_discordant"] > 0.2 and strand_check_fails(bad)


def test_cohort_on_the_other_strand_fails_closed(small, tmp_path):
    """The cohort's BIM on the opposite strand: A/G SNPs resolve by their complement, but C/G SNPs then read as an
    allele swap and their dosages would enter the curve with the wrong sign."""
    d = small["dir"]
    for ext in ("bed", "fam"):
        os.symlink(d / f"cohort.{ext}", tmp_path / f"flipped.{ext}")
    bim = pd.read_csv(d / "cohort.bim", sep=r"\s+", header=None, dtype=str)
    bim[4], bim[5] = bim[4].map(_COMP), bim[5].map(_COMP)
    bim.to_csv(tmp_path / "flipped.bim", sep="\t", header=False, index=False)
    with pytest.raises(ValueError, match="different strands"):
        genotype_curve(str(tmp_path / "flipped"), str(d / "ref"), str(tmp_path / "c"), covar=str(d / "cohort.cov"),
                       cell_cm=0.5, log=quiet)
    _, info = genotype_curve(str(tmp_path / "flipped"), str(d / "ref"), str(tmp_path / "c2"),
                             covar=str(d / "cohort.cov"), cell_cm=0.5, drop_ambiguous=True, log=quiet)
    assert info["match"]["n_ambiguous_dropped"] > 0 and info["match"]["drop_ambiguous"]
    assert info["strand_check"]["ambiguous_snps"] == 0


def test_missing_genotypes_are_recorded(small, tmp_path):
    d, S = small["dir"], small["S"]
    G = S["cohort"].copy()
    G[np.random.default_rng(3).random(G.shape) < 0.08] = np.nan
    write_bed(str(tmp_path / "miss"), G, S["variants"], "c")
    _, info = genotype_curve(str(tmp_path / "miss"), str(d / "ref"), str(tmp_path / "c"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, log=quiet)
    assert 0.07 < info["missing_share"] < 0.09
    assert any(f["flag"] == "missing_genotypes" for f in info["flags"])


def test_monomorphic_panel_snps_compose(small, tmp_path):
    d, S = small["dir"], small["S"]
    GA = S["panel_A"].astype(np.int8).copy()
    GA[:, :25] = 0                                            # fixed in the first panel
    write_bed(str(tmp_path / "panelA"), GA, S["variants"], "a")
    build_reference([str(tmp_path / "panelA"), str(d / "panelE")], ["AFR", "EUR"], str(tmp_path / "ref"), step=256,
                    log=quiet)
    compose_ldscores(str(tmp_path / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "ld"),
                     generations=8, log=quiet)
    L = pd.read_csv(tmp_path / "ld.gw.ldscore.gz", sep="\t")
    poly = (GA.var(axis=0) > 0) | (S["panel_E"].var(axis=0) > 0)          # kept unless fixed in every panel
    assert L.SNP.tolist() == S["variants"].SNP[poly].tolist()
    assert np.isfinite(L["all"]).all() and (L["all"] > 0).all()


def test_few_snps_give_a_readable_error(tmp_path):
    S = simulate(seed=2, n_chr=1, snps_per_chr=40, chr_cm=2.5, n_panel=60, n_cohort=200)
    v = S["variants"]
    write_bed(str(tmp_path / "pA"), S["panel_A"].astype(np.int8), v, "a")
    write_bed(str(tmp_path / "pE"), S["panel_E"].astype(np.int8), v, "e")
    write_bed(str(tmp_path / "co"), S["cohort"].astype(np.int8), v, "c")
    build_reference([str(tmp_path / "pA"), str(tmp_path / "pE")], ["AFR", "EUR"], str(tmp_path / "ref"), log=quiet)
    with pytest.raises(ValueError, match="use date mode"):
        genotype_curve(str(tmp_path / "co"), str(tmp_path / "ref"), str(tmp_path / "c"), log=quiet)


def test_curve_with_other_covariates_is_flagged(small, tmp_path):
    d = small["dir"]
    genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c"), covar=str(d / "cohort.cov"), cell_cm=0.5,
                   log=quiet)
    ss = pd.read_csv(d / "ss" / "trait0.tsv", sep="\t").assign(COV_RANK=5)
    os.makedirs(tmp_path / "ss")
    ss.to_csv(tmp_path / "ss" / "trait0.tsv", sep="\t", index=False)
    p = compose_ldscores(str(d / "ref"), [str(tmp_path / "ss")], str(d / "annot.tsv"), str(tmp_path / "ld"),
                         curve=str(tmp_path / "c"), log=quiet)
    assert any(f["flag"] == "covariates_differ_from_curve" for f in p["flags"])
    same = compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "ld2"),
                            curve=str(tmp_path / "c"), log=quiet)
    assert not any(f["flag"] == "covariates_differ_from_curve" for f in same["flags"])
    assert json.load(open(tmp_path / "ld2.admixed_ld.json"))["match"]["n_allele_mismatch"] == 0


def test_standard_errors_of_another_kind_are_flagged():
    t = pd.DataFrame({"trait": [f"t{i}" for i in range(6)], "m": [[0.60, 0.40]] * 5 + [[0.52, 0.48]],
                      "kappa0": [[[0.20]]] * 6, "fit_r2": [0.99] * 6})
    msgs = []
    f = se_regression_outliers(t, msgs.append)
    assert len(f) == 1 and "t5" in f[0]["detail"] and "t0" not in f[0]["detail"] and msgs
    assert se_regression_outliers(t.iloc[:5], quiet) == []


def _tractor_table(path, cols):
    rng = np.random.default_rng(4)
    n = 50
    t = pd.DataFrame({"CHR": 1, "POS": np.arange(1, n + 1) * 1000, "ID": [f"s{i}" for i in range(n)], "REF": "A",
                      "ALT": "G", "N": 4000})
    for s in (0, 1):
        t[f"beta_anc{s}"] = rng.normal(0, 0.02, n)
        t[f"se_anc{s}"] = 0.02
        t[f"AF_anc{s}"] = rng.uniform(0.1, 0.9, n)
        t[f"LAprop_anc{s}"] = 0.5
        for c in cols:
            t[f"{c}_anc{s}"] = 0.021 if c == "se_hcn" else 0.019
    t.to_csv(path, sep="\t", index=False)


def test_tractor_sumstats_prefers_the_covariates_only_sandwich(tmp_path):
    from summit.admixed_ld.tracks import tractor_sumstats
    _tractor_table(tmp_path / "both.tsv", ["se_hc", "se_hcn"])
    rec = tractor_sumstats(str(tmp_path / "both.tsv"), str(tmp_path / "both"), sources=["AFR", "EUR"],
                           n_covariates=10, log=quiet)
    t = pd.read_csv(tmp_path / "both.AFR.sumstats", sep="\t")
    assert np.allclose(t.SE, 0.021) and not any(f["flag"] == "sandwich_from_full_model_residuals" for f in rec["flags"])
    _tractor_table(tmp_path / "hc1.tsv", ["se_hc1"])
    msgs = []
    rec = tractor_sumstats(str(tmp_path / "hc1.tsv"), str(tmp_path / "hc1"), sources=["AFR", "EUR"], n_covariates=10,
                           log=msgs.append)
    assert any(f["flag"] == "sandwich_from_full_model_residuals" for f in rec["flags"])
    assert any("biased low" in m for m in msgs)
    _tractor_table(tmp_path / "ols.tsv", [])
    with pytest.raises(ValueError, match="covariates-only model"):
        tractor_sumstats(str(tmp_path / "ols.tsv"), str(tmp_path / "ols"), sources=["AFR", "EUR"], log=quiet)


# ----------------------------------------------------------------------------------------------- curve-mode defaults
def test_command_line_defaults_are_curve_mode(small, tmp_path):
    from summit.entrypoint import main
    d = small["dir"]
    ref = str(tmp_path / "ref_clean")
    assert main(["admixed-ld", "clean-panels", "--ref", str(d / "ref"), "--panel", str(d / "panelA"), "--panel",
                 str(d / "panelE"), "--names", "AFR,EUR", "--out", ref]) == 0
    assert main(["admixed-ld", "curve", "--geno", str(d / "cohort"), "--ref", ref, "--covar", str(d / "cohort.cov"),
                 "--cell-cm", "0.5", "--out", str(tmp_path / "c")]) == 0
    info = json.load(open(tmp_path / "c.curve.json"))
    assert "background" in info and "cells" in info                  # the background term by default
    out = str(tmp_path / "ld")
    base = ["admixed-ld", "compose", "--ref", ref, "--sumstats", str(d / "ss"), "--annot", str(d / "annot.tsv"),
            "--curve", str(tmp_path / "c")]
    assert main(base + ["--out", out]) == 0
    p = json.load(open(out + ".admixed_ld.json"))
    assert p["relatedness_profile"]["mode"] == "background" and p["cell_matrix"] is not None
    assert main(base + ["--relatedness", "none", "--no-cell-matrix", "--out", out + "_bare"]) == 0
    q = json.load(open(out + "_bare.admixed_ld.json"))
    assert q["relatedness_profile"] is None and q["cell_matrix"] is None
    # the same curve with a reference that was never cleaned: the background term needs the panels' own
    with pytest.raises(ValueError, match="--relatedness none"):
        compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "x"),
                         curve=str(tmp_path / "c"), relatedness="auto", cell_matrix="auto", log=quiet)


def test_listed_pairs_profile_only_without_background(small, tmp_path):
    d, iid = small["dir"], small["iid"]
    pd.DataFrame({"IID1": iid[0:40:2], "IID2": iid[1:40:2]}).to_csv(tmp_path / "pairs.tsv", sep="\t", index=False)
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(tmp_path / "pairs.tsv"), background=True, log=quiet)
    assert info["relatedness"]["pairs"] == 20 and "profile" not in info["relatedness"]
    assert not os.path.exists(tmp_path / "c.relatedness.tsv")
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "c2"), covar=str(d / "cohort.cov"),
                             cell_cm=0.5, relative_pairs=str(tmp_path / "pairs.tsv"), background=False, log=quiet)
    assert "profile" in info["relatedness"] and os.path.exists(tmp_path / "c2.relatedness.tsv")


def test_background_memory_budget_subsamples(small, tmp_path):
    d = small["dir"]
    kw = dict(covar=str(d / "cohort.cov"), cell_cm=0.5, background=True, background_quantile=0.5, log=quiet)
    _, full = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "f"), **kw)
    _, cut = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "s"), background_memory_gb=0.0005, **kw)
    assert not full["background"]["subsampled"] and set(cut["background"]["subsampled"]) == {"1", "2"}
    assert cut["background"]["n_snps"] < full["background"]["n_snps"]
    a, b = (pd.read_csv(tmp_path / f"{x}.background.tsv", sep="\t") for x in ("f", "s"))
    far = lambda t: float((t.excess * t.w)[t.d_cm > 5].sum() / t.w[t.d_cm > 5].sum())
    assert np.isfinite(far(b)) and abs(far(b) - far(a)) < 5 * abs(far(a)) + 1.0     # same quantity, noisier


def test_per_sample_size_files_only_when_they_differ(small, tmp_path):
    d = small["dir"]
    ss = pd.read_csv(d / "ss" / "trait0.tsv", sep="\t")
    for tag, (n1, n2) in {"small": (900, 1800), "large": (100000, 120000)}.items():
        os.makedirs(tmp_path / tag)
        for i, n in enumerate((n1, n2)):                 # N scales SE^2, so m and kappa_res(0) do not move
            ss.assign(N=n, SE=ss.SE * np.sqrt(900.0 / n)).to_csv(tmp_path / tag / f"t{i}.tsv", sep="\t", index=False)
        p = compose_ldscores(str(d / "ref"), [str(tmp_path / tag)], str(d / "annot.tsv"), str(tmp_path / f"ld_{tag}"),
                             generations=8, log=quiet)
        files = sorted(x.name for x in tmp_path.glob(f"ld_{tag}.n*.gw.ldscore.gz"))
        if tag == "small":
            assert len(files) == 2 and os.path.exists(tmp_path / f"ld_{tag}.n_groups.tsv")
        else:
            assert files == [] and all(g.get("merged") for g in p["n_groups"])
