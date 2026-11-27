"""The cohort's release and the analyst's side of `summit admixed-ld`: the MAF/LD bins against a direct computation,
the release folder and compose --release, the 50-person rule of the sample statistic, the covariate rank of summary
statistics without COV_RANK, the summary-statistic files of a directory, LD scores over other partners (refused by
summit --h2/--rg), compose prefixes in --rg, exactly symmetric regional matrices, the annotation of the cross-population
SNPs, and the table of flags in the docs.
Small simulations (int8 dosages, 900 people, two chromosomes) keep the test process's memory low."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admixed_ld_sim import gwas, principal_components, simulate, write_bed
from summit.admixed_ld.bins import BIN_NAMES, local_ld_scores, maf_classes, maf_ld_bins, make_bins
from summit.admixed_ld.compose import compose_ldscores
from summit.admixed_ld.curve import genotype_curve
from summit.admixed_ld.panel_admixture import remove_panel_admixture
from summit.admixed_ld.reference import build_reference
from summit.admixed_ld.release import make_release

REPO = Path(__file__).resolve().parents[1]
quiet = lambda *a: None


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("admixed_ld_release")
    S = simulate(seed=5, n_chr=2, snps_per_chr=400, chr_cm=60.0, n_panel=300, n_cohort=900)
    v = S["variants"]
    write_bed(str(d / "panelA"), S["panel_A"].astype(np.int8), v, "a")
    write_bed(str(d / "panelE"), S["panel_E"].astype(np.int8), v, "e")
    G = S["cohort"].astype(np.float32)
    iid = write_bed(str(d / "cohort"), G, v, "c")
    C = principal_components(S["cohort"], 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(d / "cohort.cov", sep="\t",
                                                                                   index=False)
    rng = np.random.default_rng(2)
    os.makedirs(d / "ss")
    for t in range(2):
        y = rng.standard_normal(G.shape[0])
        beta, se = gwas(S["cohort"], y, C)
        ok = np.isfinite(beta) & np.isfinite(se) & (se > 0)
        pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1": v.A1, "A2": v.A2, "N": len(y), "BETA": beta,
                      "SE": se, "COV_RANK": 2})[ok].to_csv(d / "ss" / f"trait{t}.sumstats", sep="\t", index=False)
    pd.DataFrame({"CHR": v.CHR, "BP": v.BP, "SNP": v.SNP, "all": 1}).to_csv(d / "annot.tsv", sep="\t", index=False)
    build_reference([str(d / "panelA"), str(d / "panelE")], ["AFR", "EUR"], str(d / "ref0"), step=256, log=quiet)
    remove_panel_admixture(str(d / "ref0"), {"AFR": str(d / "panelA"), "EUR": str(d / "panelE")}, out=str(d / "ref"),
                           log=quiet)
    return dict(dir=d, S=S, iid=iid)


def _direct_local_scores(G, bp, chrom, window=1_000_000):
    """r^2 - (1 - r^2) / (n - 2) over the polymorphic SNPs within `window` on the chromosome, from the full
    correlation matrix; also which SNPs are polymorphic."""
    X = np.where(np.isnan(G), np.nanmean(G, axis=0), G)
    n = X.shape[0]
    X = X - X.mean(axis=0)
    sd = np.sqrt((X * X).sum(axis=0) / (n - 1))
    Z = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)
    R = Z.T @ Z / (n - 1)
    r2 = R * R
    near = (np.abs(bp[:, None] - bp[None, :]) <= window) & (chrom[:, None] == chrom[None, :]) & (sd > 0)[None, :]
    return np.where(near, r2 - (1 - r2) / (n - 2), 0.0).sum(axis=1), sd > 0


def test_local_scores_and_bins_match_a_direct_computation(small, tmp_path):
    """The local score of every SNP equals the sum over its partners within 1 Mb from the full correlation matrix, in
    a subset of people and with missing calls (at the SNP's mean); the bins are the MAF bins' quartiles pooled by
    quartile, with the MAF of the called genotypes."""
    S, v = small["S"], small["S"]["variants"]
    G = S["cohort"][:400].astype(np.float64)
    G[np.random.default_rng(3).random(G.shape) < 0.02] = np.nan
    iid = write_bed(str(tmp_path / "miss"), G.astype(np.float32), v, "m")
    keep = tmp_path / "keep.txt"
    pd.DataFrame({"FID": iid[50:], "IID": iid[50:]}).to_csv(keep, sep=" ", header=False, index=False)
    tab, n = local_ld_scores(str(tmp_path / "miss"), keep=str(keep), step=97, log=quiet)
    want, poly = _direct_local_scores(G[50:], v.BP.to_numpy(), v.CHR.to_numpy())
    assert n == 350 and tab.SNP.tolist() == v.SNP[poly].tolist()
    np.testing.assert_allclose(tab.local.to_numpy(), want[poly], rtol=1e-5, atol=1e-5)
    f = np.nanmean(G[50:], axis=0)[poly] / 2
    np.testing.assert_allclose(tab.MAF.to_numpy(), np.minimum(f, 1 - f), atol=1e-12)
    A = maf_ld_bins(tab.MAF.to_numpy(), tab.local.to_numpy())
    assert (A.sum(axis=1) == 1).all()
    mb = np.digitize(tab.MAF.to_numpy(), [0.02, 0.05, 0.1, 0.2, 0.3]) + 1
    lab = np.asarray(BIN_NAMES)[A.argmax(axis=1)]
    assert set(lab[mb <= 2]) <= {f"maf_low_ld{q}" for q in range(1, 5)}
    assert set(lab[mb >= 3]) <= {f"maf_high_ld{q}" for q in range(1, 5)}
    for b in range(1, 7):                                          # quartiles of the score within each MAF bin
        sel = np.flatnonzero(mb == b)
        if sel.size < 8:
            continue
        q = np.array([int(x[-1]) for x in lab[sel]])
        order = np.argsort(tab.local.to_numpy()[sel], kind="stable")
        assert (np.diff(q[order]) >= 0).all()
        assert np.abs(np.bincount(q, minlength=5)[1:] - sel.size / 4).max() <= 1
    rec = make_bins(str(tmp_path / "miss"), str(tmp_path / "b"), keep=str(keep), log=quiet)
    a8 = pd.read_csv(tmp_path / "b.bins8.annot.gz", sep="\t")
    a2 = pd.read_csv(tmp_path / "b.maf2.annot.gz", sep="\t")
    assert list(a8.columns) == ["CHR", "BP", "SNP", "CM"] + BIN_NAMES
    assert list(a2.columns)[4:] == ["maf_low", "maf_high"]
    np.testing.assert_array_equal(a8[BIN_NAMES[:4]].sum(axis=1), a2.maf_low)
    assert rec["n_people"] == 350 and not os.path.exists(tmp_path / "b.local_ld.tsv.gz")
    # the 24 unpooled bins (six MAF bins by four quartiles) pool by quartile into the eight bins
    a24 = pd.read_csv(tmp_path / "b.bins24.annot.gz", sep="\t")
    f24 = [c for c in a24.columns if c not in ("CHR", "BP", "SNP", "CM")]
    assert len(f24) == 24 and (a24[f24].sum(axis=1) == 1).all() and a24.SNP.tolist() == a8.SNP.tolist()
    for c in ("low", "high"):
        for q in range(1, 5):
            members = [x for x in f24 if x.endswith(f"_ld{q}") and (float(x.split("_")[2]) <= 0.05) == (c == "low")]
            np.testing.assert_array_equal(a24[members].sum(axis=1), a8[f"maf_{c}_ld{q}"])


def test_two_maf_classes_from_frequencies(small, tmp_path):
    """The analyst's fallback: the two MAF classes from an allele-frequency column, or from the panels at a mean
    ancestry."""
    d, v = small["dir"], small["S"]["variants"]
    f = np.random.default_rng(4).uniform(0, 1, len(v))
    pd.DataFrame({"SNP": v.SNP, "CHR": v.CHR, "BP": v.BP, "A1_FREQ": f}).to_csv(tmp_path / "ss.tsv", sep="\t",
                                                                               index=False)
    maf_classes(str(tmp_path / "a"), sumstats=str(tmp_path / "ss.tsv"), log=quiet)
    a = pd.read_csv(tmp_path / "a.maf2.annot.gz", sep="\t")
    np.testing.assert_array_equal(a.maf_low.to_numpy(), (np.minimum(f, 1 - f) < 0.05).astype(int))
    rec = maf_classes(str(tmp_path / "r"), ref=str(d / "ref"), mean_ancestry="0.7", log=quiet)
    assert rec["maf_classes"]["maf_low"] + rec["maf_classes"]["maf_high"] == rec["n_snps"]
    with pytest.raises(ValueError, match="no allele-frequency column"):
        maf_classes(str(tmp_path / "x"), sumstats=str(d / "ss" / "trait0.sumstats"), log=quiet)


def test_release_folder_and_compose_from_it(small, tmp_path):
    """release writes the curve with its sample statistic, background term and cell matrix, the bins of the same
    people, release.json and README.txt, and nothing about a person; compose --release gives the LD scores of
    compose --curve on that curve; a folder with other files is refused."""
    d = small["dir"]
    out = tmp_path / "rel"
    rec = make_release(str(d / "cohort"), str(d / "ref"), str(out), covar=str(d / "cohort.cov"), cell_cm=0.5,
                       log=quiet)
    assert rec["people"] == 900 and rec["covariate_rank"] == 2 and rec["sample_statistic"]
    names = sorted(os.listdir(out))
    assert names == sorted(["README.txt", "release.json", "cohort.curve.tsv", "cohort.curve.json",
                            "cohort.background.tsv", "cohort.cells.npz", "cohort.sample_factors.json",
                            "cohort.bins8.annot.gz", "cohort.bins24.annot.gz", "cohort.maf2.annot.gz"])
    assert {f["file"] for f in rec["files"]} | {"README.txt", "release.json"} == set(names)
    text = " ".join((out / "README.txt").read_text().split())
    assert "Never share" in text and "at least 50 people" in text
    sf = json.load(open(out / "cohort.sample_factors.json"))
    assert set(sf["leverage"]) == {"mean", "sd"}                # no maximum or quantile: one person's value
    assert "max_abs" not in json.load(open(out / "cohort.curve.json")).get("relatedness", {})
    ss = [str(d / "ss" / f"trait{t}.sumstats") for t in range(2)]
    cli = dict(relatedness="auto", cell_matrix="auto", log=quiet)      # the command line's defaults
    p_rel = compose_ldscores(str(d / "ref"), ss, str(out / "cohort.bins8.annot.gz"), str(tmp_path / "c_rel"),
                             release=str(out), **cli)
    p_cur = compose_ldscores(str(d / "ref"), ss, str(out / "cohort.bins8.annot.gz"), str(tmp_path / "c_cur"),
                             curve=str(out / "cohort"), **cli)
    a, b = (pd.read_csv(tmp_path / f"{x}.gw.ldscore.gz", sep="\t") for x in ("c_rel", "c_cur"))
    pd.testing.assert_frame_equal(a, b)
    assert p_rel["release"]["people"] == 900 and p_rel["plan"]["components"] == p_cur["plan"]["components"]
    assert p_rel["plan"]["components"]["statistic"] and p_rel["plan"]["components"]["cells"]
    with pytest.raises(ValueError, match="not a release folder"):
        make_release(str(d / "cohort"), str(d / "ref"), str(d), covar=str(d / "cohort.cov"), log=quiet)
    rec2 = make_release(str(d / "cohort"), str(d / "ref"), str(out), covar=str(d / "cohort.cov"), cell_cm=0.5,
                        log=quiet)                                  # a release folder is replaced
    assert rec2["people"] == 900


def test_statistic_needs_fifty_people(small, tmp_path):
    from summit.admixed_ld.sample_factors import sample_factors
    d, iid = small["dir"], small["iid"]
    keep = tmp_path / "few.txt"
    pd.DataFrame({"FID": iid[:40], "IID": iid[:40]}).to_csv(keep, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="needs at least 50"):
        sample_factors(str(d / "cohort"), str(d / "ref"), str(tmp_path / "sf"), keep=str(keep), log=quiet)
    with pytest.raises(ValueError, match="needs at least 50.*--no-sample-factors"):
        genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "cv"), keep=str(keep), log=quiet)


def test_cov_rank_from_the_curve(small, tmp_path):
    """Summary statistics without COV_RANK take the curve's covariate rank (flag cov_rank_from_curve), --cov-rank
    sets it, a COV_RANK that differs from the curve's is flagged even when it is 0, and date mode without either
    records cov_rank_missing."""
    d = small["dir"]
    curve = tmp_path / "cv"
    genotype_curve(str(d / "cohort"), str(d / "ref"), str(curve), covar=str(d / "cohort.cov"), cell_cm=0.5,
                   log=quiet)
    t = pd.read_csv(d / "ss" / "trait0.sumstats", sep="\t")
    t.drop(columns="COV_RANK").to_csv(tmp_path / "norank.tsv", sep="\t", index=False)
    t.assign(COV_RANK=0).to_csv(tmp_path / "rank0.tsv", sep="\t", index=False)
    kw = dict(dry_run=False, log=quiet)
    ref, an = str(d / "ref"), str(d / "annot.tsv")
    p = compose_ldscores(ref, [str(tmp_path / "norank.tsv")], an, str(tmp_path / "a"), curve=str(curve), **kw)
    assert p["covariate_rank"] == 2 and p["nu"] == 900 - 2 - 1
    assert "cov_rank_from_curve" in [f["flag"] for f in p["flags"]]
    p_with = compose_ldscores(ref, [str(d / "ss" / "trait0.sumstats")], an, str(tmp_path / "b"), curve=str(curve),
                              **kw)
    a, b = (pd.read_csv(tmp_path / f"{x}.gw.ldscore.gz", sep="\t") for x in ("a", "b"))
    pd.testing.assert_frame_equal(a, b)                               # the same as with COV_RANK 2
    p3 = compose_ldscores(ref, [str(tmp_path / "norank.tsv")], an, str(tmp_path / "c"), curve=str(curve), cov_rank=5,
                          **kw)
    assert p3["covariate_rank"] == 5 and "covariates_differ_from_curve" in [f["flag"] for f in p3["flags"]]
    p0 = compose_ldscores(ref, [str(tmp_path / "rank0.tsv")], an, str(tmp_path / "z"), curve=str(curve), **kw)
    assert "covariates_differ_from_curve" in [f["flag"] for f in p0["flags"]]
    pd_ = compose_ldscores(ref, [str(tmp_path / "norank.tsv")], an, str(tmp_path / "d"), generations=8, **kw)
    assert pd_["covariate_rank"] == 0 and "cov_rank_missing" in [f["flag"] for f in pd_["flags"]]


def test_sumstats_directory_reads_what_h2_reads(small, tmp_path):
    """A directory of .sumstats files (the example's names) is read whole, as summit --h2 DIR reads it."""
    d = small["dir"]
    p = compose_ldscores(str(d / "ref"), [str(d / "ss")], str(d / "annot.tsv"), str(tmp_path / "dir"),
                         generations=8, log=quiet)
    assert p["plan"]["traits"] == 2
    os.makedirs(tmp_path / "empty")
    with pytest.raises(ValueError, match="No summary-statistic files in the directory"):
        compose_ldscores(str(d / "ref"), [str(tmp_path / "empty")], str(d / "annot.tsv"), str(tmp_path / "e"),
                         generations=8, log=quiet)


def _summit(*args, cwd):
    # the summit that the tests import (an installed or editable build), so compiled modules are found
    return subprocess.run([sys.executable, "-m", "summit", *args], cwd=cwd, capture_output=True, text=True)


def test_partner_ldscores_are_refused_and_rg_takes_a_prefix(small, tmp_path):
    """LD scores composed over other partners than the GWAS SNPs carry a marker that summit --h2 and --rg refuse;
    a later composition over the rows to the same prefix removes it; --rg resolves a compose prefix as --h2 does."""
    d = small["dir"]
    ss = [str(d / "ss" / f"trait{t}.sumstats") for t in range(2)]
    an = str(d / "annot.tsv")
    p = compose_ldscores(str(d / "ref"), ss, an, str(tmp_path / "ld"), generations=8, partners="reference", log=quiet)
    assert "partners_reference" in [f["flag"] for f in p["flags"]]
    assert os.path.exists(tmp_path / "ld.partners.json")
    r = _summit("--h2", ss[0], "--ldscores", str(tmp_path / "ld"), "--annot", an, "--out", str(tmp_path / "h"),
                cwd=tmp_path)
    assert r.returncode != 0 and "--partners reference" in (r.stderr + r.stdout)
    r = _summit("--rg", ",".join(ss), "--ldscores", str(tmp_path / "ld"), "--annot", an, "--out", str(tmp_path / "g"),
                cwd=tmp_path)
    assert r.returncode != 0 and "--partners reference" in (r.stderr + r.stdout)
    compose_ldscores(str(d / "ref"), ss, an, str(tmp_path / "ld"), generations=8, log=quiet)
    assert not os.path.exists(tmp_path / "ld.partners.json")
    r = _summit("--rg", ",".join(ss), "--ldscores", str(tmp_path / "ld"), "--annot", an, "--keep-ambiguous",
                "--out", str(tmp_path / "g2"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Total genetic correlation" in open(tmp_path / "g2.log").read()


def test_regional_matrices_are_exactly_symmetric(small, tmp_path):
    from summit.admixed_ld.matrix import compose_matrix
    d = small["dir"]
    compose_ldscores(str(d / "ref"), [str(d / "ss" / "trait0.sumstats")], str(d / "annot.tsv"), str(tmp_path / "ld"),
                     generations=8, log=quiet)
    compose_matrix(str(d / "ref"), str(tmp_path / "m"), region="1:1-20000000",
                   params=str(tmp_path / "ld.admixed_ld.json"), generations=8,
                   sumstats=str(d / "ss" / "trait0.sumstats"), error_variance=True, log=quiet)
    R, V = np.load(tmp_path / "m.ld.npz")["R"], np.load(tmp_path / "m.ld_var.npz")["V"]
    assert np.array_equal(R, R.T) and np.array_equal(V, V.T)
    Rt, Vt = np.loadtxt(tmp_path / "m.ld"), np.loadtxt(tmp_path / "m.ld_var")
    assert np.array_equal(Rt, Rt.T) and np.array_equal(Vt, Vt.T)


def test_cross_source_writes_the_annotation_of_its_snps(small, tmp_path):
    """compose --cross-source writes the annotation and the IDs of the cross file's SNPs; composing with that annotation
    puts the admixed GWAS's own LD scores on the same SNPs in the same order, as the cross-population fit needs."""
    d = small["dir"]
    ss = [str(d / "ss" / "trait0.sumstats")]
    p = compose_ldscores(str(d / "ref"), ss, str(d / "annot.tsv"), str(tmp_path / "x"), generations=8,
                         cross_source="EUR", log=quiet)
    cross = pd.read_csv(tmp_path / "x.cross_EUR.gw.ldscore.gz", sep="\t")
    an = pd.read_csv(tmp_path / "x.cross_EUR.annot.gz", sep="\t")
    ids = pd.read_csv(tmp_path / "x.cross_EUR.snps", header=None)[0]
    assert list(an.columns) == ["CHR", "BP", "SNP", "CM", "all"] and (an["all"] == 1).all()
    assert an.SNP.tolist() == cross.SNP.tolist() == ids.tolist()
    assert p["cross"]["annotation"].endswith("x.cross_EUR.annot.gz")
    compose_ldscores(str(d / "ref"), ss, str(tmp_path / "x.cross_EUR.annot.gz"), str(tmp_path / "own"),
                     generations=8, log=quiet)
    own = pd.read_csv(tmp_path / "own.gw.ldscore.gz", sep="\t")
    assert own.SNP.tolist() == cross.SNP.tolist()


def _cli(*args, cwd):
    """`summit admixed-ld ...` as a user runs it (the summit that the tests import)."""
    return subprocess.run([sys.executable, "-m", "summit", "admixed-ld", *args], cwd=cwd, capture_output=True,
                          text=True)


def test_release_command_checks_first_and_creates_parent_folders(small, tmp_path):
    """The command line's release checks its inputs before it writes anything (a refused release leaves no folder and
    no log), builds into missing parent folders, and records no pair's relatedness; compose, bins and sample-factors
    create the parent folder of --out; a folder step given a file as --out stops with a message."""
    d, iid = small["dir"], small["iid"]
    few = tmp_path / "few.txt"
    pd.DataFrame({"FID": iid[:40], "IID": iid[:40]}).to_csv(few, sep=" ", header=False, index=False)
    pairs = tmp_path / "pairs.tsv"
    pd.DataFrame({"IID1": iid[[0, 2]], "IID2": iid[[1, 3]]}).to_csv(pairs, sep="\t", index=False)
    before = set(os.listdir(tmp_path))
    r = _cli("release", "--geno", str(d / "cohort"), "--ref", str(d / "ref"), "--keep", str(few), "--covar",
             str(d / "cohort.cov"), "--out", "x/release", cwd=tmp_path)
    assert r.returncode != 0 and "at least 50" in r.stderr
    assert set(os.listdir(tmp_path)) == before                      # nothing written, no log
    r = _cli("release", "--geno", str(d / "cohort"), "--ref", str(d / "ref"), "--covar", str(d / "cohort.cov"),
             "--relative-pairs", str(pairs), "--cell-cm", "0.5", "--out", "x/y/release", cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    rel = tmp_path / "x" / "y" / "release"
    assert {"release.json", "release.log", "README.txt"} <= set(os.listdir(rel))
    assert [f for f in os.listdir(rel.parent) if f != "release"] == []    # no work folder left behind
    cj = json.load(open(rel / "cohort.curve.json"))
    # two listed pairs: no summary over them, a flag and a note; compose (below) works with the background term
    assert cj["relatedness"]["pairs"] == 2 and cj["relatedness"]["summaries"] is None
    assert not {"mean", "sum_squared_ordered", "max_abs", "profile"} & set(cj["relatedness"])
    assert "relative_pair_summaries_omitted" in [f["flag"] for f in cj["flags"]]
    assert "Fewer than 20 related pairs" in " ".join((rel / "README.txt").read_text().split())
    assert cj["background"]["file"] == "cohort.background.tsv" and cj["cells"]["file"] == "cohort.cells.npz"
    assert "largest" not in (rel / "release.log").read_text()
    ss = str(d / "ss" / "trait0.sumstats")
    r = _cli("compose", "--ref", str(d / "ref"), "--release", str(rel), "--sumstats", ss, "--annot",
             str(rel / "cohort.bins8.annot.gz"), "--out", "ld/cohort", cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.exists(tmp_path / "ld" / "cohort.gw.ldscore.gz") and os.path.exists(tmp_path / "ld" /
                                                                                      "cohort.compose.log")
    r = _cli("bins", "--geno", str(d / "cohort"), "--out", "bins/cohort", cwd=tmp_path)
    assert r.returncode == 0 and os.path.exists(tmp_path / "bins" / "cohort.bins8.annot.gz")
    r = _cli("bins", "--geno", str(d / "cohort"), "--keep", str(few), "--out", "bins/few", cwd=tmp_path)
    assert r.returncode != 0 and "at least 50" in r.stderr
    (tmp_path / "afile").write_text("x\n")
    r = _cli("reference", "--panel", str(d / "panelA"), "--panel", str(d / "panelE"), "--names", "AFR,EUR", "--out",
             "afile", cwd=tmp_path)
    assert r.returncode != 0 and "is not a folder" in r.stderr and "FileExistsError" not in r.stderr


def test_release_rerun_keeps_the_earlier_release_until_the_new_one_is_complete(small, tmp_path, monkeypatch):
    """A rerun that fails leaves the earlier release as it was and its partial folder at OUT.failed; a rerun that
    succeeds replaces the release and removes OUT.failed; a folder of other files is refused before anything changes."""
    import summit.admixed_ld.curve as curve_mod
    d = small["dir"]
    out = tmp_path / "release"
    kw = dict(covar=str(d / "cohort.cov"), cell_cm=0.5, log=quiet)
    make_release(str(d / "cohort"), str(d / "ref"), str(out), **kw)
    first = (out / "release.json").read_text()

    def broken(*a, **k):
        raise RuntimeError("the curve stopped")
    monkeypatch.setattr(curve_mod, "genotype_curve", broken)
    with pytest.raises(RuntimeError, match="the curve stopped"):
        make_release(str(d / "cohort"), str(d / "ref"), str(out), **kw)
    assert (out / "release.json").read_text() == first and os.path.exists(out / "cohort.curve.json")
    assert os.path.isdir(tmp_path / "release.failed")
    monkeypatch.undo()
    make_release(str(d / "cohort"), str(d / "ref"), str(out), **kw)
    assert not os.path.exists(tmp_path / "release.failed")
    assert sorted(f for f in os.listdir(tmp_path) if f.startswith(".")) == []
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("x\n")
    with pytest.raises(ValueError, match="not a release folder"):
        make_release(str(d / "cohort"), str(d / "ref"), str(other), **kw)
    assert os.listdir(other) == ["notes.txt"]


def test_release_on_snps_filtered_on_maf(small, tmp_path):
    """A cohort whose SNPs were filtered on MAF (no SNP below 0.05): the release leaves the empty bins and class out,
    says so in its README, and compose and the fit read the annotation."""
    S, v = small["S"], small["S"]["variants"]
    G = S["cohort"]
    f = G.mean(axis=0) / 2
    common = np.minimum(f, 1 - f) >= 0.05
    iid = write_bed(str(tmp_path / "common"), G[:, common].astype(np.int8), v[common].reset_index(drop=True), "c")
    C = principal_components(G, 2)
    pd.DataFrame({"FID": iid, "IID": iid, "PC1": C[:, 0], "PC2": C[:, 1]}).to_csv(tmp_path / "common.cov",
                                                                                   sep="\t", index=False)
    rec = make_release(str(tmp_path / "common"), str(small["dir"] / "ref"), str(tmp_path / "release"),
                       covar=str(tmp_path / "common.cov"), cell_cm=0.5, log=quiet)
    assert rec["bins"]["empty_left_out"] == ([f"maf_low_ld{q}" for q in range(1, 5)]
                                             + [f"maf_{b}_ld{q}" for b in ("0_0.02", "0.02_0.05") for q in range(1, 5)]
                                             + ["maf_low"])
    a8 = pd.read_csv(tmp_path / "release" / "cohort.bins8.annot.gz", sep="\t")
    assert list(a8.columns)[4:] == [f"maf_high_ld{q}" for q in range(1, 5)]
    assert "maf_low" in " ".join((tmp_path / "release" / "README.txt").read_text().split())
    compose_ldscores(str(small["dir"] / "ref"), [str(small["dir"] / "ss" / "trait0.sumstats")],
                     str(tmp_path / "release" / "cohort.bins8.annot.gz"), str(tmp_path / "ld"),
                     release=str(tmp_path / "release"), log=quiet)
    assert list(pd.read_csv(tmp_path / "ld.gw.ldscore.gz", sep="\t", nrows=1).columns)[4:] == list(a8.columns)[4:]


def test_a_moved_release_reads_its_own_files(small, tmp_path, monkeypatch):
    """compose --release other/release, run from a folder that has a release of its own at ./release, reads the
    background term and the cell matrix of other/release."""
    d = small["dir"]
    other = tmp_path / "other"
    make_release(str(d / "cohort"), str(d / "ref"), str(other / "release"), covar=str(d / "cohort.cov"),
                 cell_cm=0.5, log=quiet)
    here = tmp_path / "here"
    make_release(str(d / "cohort"), str(d / "ref"), str(here / "release"), cell_cm=0.5, log=quiet)   # no covariates
    monkeypatch.chdir(here)
    cli = dict(relatedness="auto", cell_matrix="auto", log=quiet)
    ss, an = [str(d / "ss" / "trait0.sumstats")], str(d / "annot.tsv")
    p_moved = compose_ldscores(str(d / "ref"), ss, an, str(tmp_path / "moved"), release="../other/release", **cli)
    monkeypatch.chdir(other)
    p_home = compose_ldscores(str(d / "ref"), ss, an, str(tmp_path / "home"), release="release", **cli)
    assert p_moved["relatedness_profile"]["source"] == "../other/release/cohort.background.tsv"
    assert p_moved["cell_matrix"]["file"] == "../other/release/cohort.cells.npz"
    a, b = (pd.read_csv(tmp_path / f"{x}.gw.ldscore.gz", sep="\t") for x in ("moved", "home"))
    pd.testing.assert_frame_equal(a, b)


def test_cov_rank_option_overrides_the_column(small, tmp_path):
    """--cov-rank overrides a COV_RANK column (logged), as summit --h2 --cov-rank does; a negative value is refused;
    a rank taken from a statistic given to compose is flagged as such; summit --h2 warns when composed LD scores meet
    summary statistics without COV_RANK and no --cov-rank."""
    from summit.admixed_ld.sample_factors import sample_factors
    d = small["dir"]
    ref, an, ss = str(d / "ref"), str(d / "annot.tsv"), str(d / "ss" / "trait0.sumstats")
    msgs = []
    p = compose_ldscores(ref, [ss], an, str(tmp_path / "a"), generations=8, cov_rank=5, log=msgs.append)
    assert p["covariate_rank"] == 5 and any("--cov-rank 5 overrides the COV_RANK 2" in m for m in msgs)
    with pytest.raises(ValueError, match="0 or more"):
        compose_ldscores(ref, [ss], an, str(tmp_path / "b"), generations=8, cov_rank=-1, log=quiet)
    t = pd.read_csv(ss, sep="\t").drop(columns="COV_RANK")
    t.to_csv(tmp_path / "norank.tsv", sep="\t", index=False)
    sample_factors(str(d / "cohort"), ref, str(tmp_path / "sf"), covar=str(d / "cohort.cov"), log=quiet)
    p = compose_ldscores(ref, [str(tmp_path / "norank.tsv")], an, str(tmp_path / "c"), generations=8,
                         sample_factors=str(tmp_path / "sf"), log=quiet)
    assert p["covariate_rank"] == 2 and "cov_rank_from_sample_statistic" in [f["flag"] for f in p["flags"]]
    r = _summit("--h2", str(tmp_path / "norank.tsv"), "--ldscores", str(tmp_path / "c"), "--annot", an, "--out",
                str(tmp_path / "h"), cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--cov-rank 2" in open(tmp_path / "h.log").read()


def _bed(prefix, G, chrom, bp, a1=None):
    n, m = G.shape
    v = pd.DataFrame({"CHR": np.asarray(chrom).astype(str), "SNP": [f"s{j}" for j in range(m)],
                      "CM": np.asarray(bp) / 1e6, "BP": np.asarray(bp), "A1": a1 if a1 is not None else "A", "A2": "G"})
    write_bed(prefix, G.astype(np.float32), v, "p")
    return v


def test_bins_edge_cases(tmp_path):
    """A MAF exactly at a cut falls in the same bin whichever allele is counted; a monomorphic SNP is neither a row
    nor a partner and leaves the others' scores and quartiles unchanged; a partner exactly 1 Mb away is included."""
    rng = np.random.default_rng(9)
    n = 100
    base = rng.binomial(2, 0.3, (n, 6)).astype(float)
    at_cut = np.zeros(n)
    at_cut[:20] = 1.0                                               # 20 copies of 200: MAF exactly 0.1
    G = np.column_stack([base, at_cut, 2.0 - at_cut])               # the same SNP, the other allele counted
    bp = np.array([1, 400_001, 1_000_001, 1_000_002, 2_500_000, 3_000_000, 4_000_000, 6_000_000])
    _bed(str(tmp_path / "e"), G, ["1"] * 8, bp)
    tab, _ = local_ld_scores(str(tmp_path / "e"), log=quiet)
    assert tab.MAF.iloc[6] == tab.MAF.iloc[7] == 0.1
    A = maf_ld_bins(tab.MAF.to_numpy(), tab.local.to_numpy())
    assert np.digitize([0.1], [0.02, 0.05, 0.1, 0.2, 0.3])[0] + 1 == 4
    want, _ = _direct_local_scores(G, bp, np.array(["1"] * 8))
    np.testing.assert_allclose(tab.local.to_numpy(), want, rtol=1e-5, atol=1e-5)   # SNP 0 and 2 are 1 Mb apart
    G2 = np.column_stack([G[:, :3], np.ones(n), G[:, 3:]])           # a monomorphic SNP in the middle
    bp2 = np.r_[bp[:3], 1_000_000, bp[3:]]
    _bed(str(tmp_path / "m"), G2, ["1"] * 9, bp2)
    msgs = []
    tab2, _ = local_ld_scores(str(tmp_path / "m"), log=msgs.append)
    assert len(tab2) == 8 and "s3" not in tab2.SNP.tolist() and any("monomorphic" in m for m in msgs)
    np.testing.assert_allclose(tab2.local.to_numpy(), tab.local.to_numpy(), rtol=1e-6)
    np.testing.assert_array_equal(maf_ld_bins(tab2.MAF.to_numpy(), tab2.local.to_numpy()), A)


def test_few_relative_pairs_record_no_pair_summary(small, tmp_path):
    """With one listed pair the curve records only the count (flag relative_pair_summaries_omitted), also without the
    background term, where it skips the listed pairs' profile; compose with the background term is unchanged by the
    pair, and a relatives' term from the pairs alone is refused with a message."""
    d, iid = small["dir"], small["iid"]
    pairs = tmp_path / "pairs.tsv"
    pd.DataFrame({"IID1": [iid[0]], "IID2": [iid[1]]}).to_csv(pairs, sep="\t", index=False)
    kw = dict(covar=str(d / "cohort.cov"), cell_cm=0.5, background=True, log=quiet)
    _, info = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "p"), relative_pairs=str(pairs), **kw)
    _, info0 = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "n"), **kw)
    assert info["relatedness"] == dict(pairs=1, summaries=None, kinship_col=None, kinship_min=None, closure=False)
    assert "relative_pair_summaries_omitted" in [f["flag"] for f in info["flags"]]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "p.background.tsv", sep="\t"),
                                  pd.read_csv(tmp_path / "n.background.tsv", sep="\t"))
    ss, an = [str(d / "ss" / "trait0.sumstats")], str(d / "annot.tsv")
    cli = dict(relatedness="auto", cell_matrix="auto", log=quiet)
    p = compose_ldscores(str(d / "ref"), ss, an, str(tmp_path / "c"), curve=str(tmp_path / "p"), **cli)
    assert p["plan"]["components"]["background"]
    _, infn = genotype_curve(str(d / "cohort"), str(d / "ref"), str(tmp_path / "nb"), relative_pairs=str(pairs),
                             **dict(kw, background=False))
    assert set(infn["relatedness"]) == {"pairs", "summaries", "kinship_col", "kinship_min", "closure"}
    assert not os.path.exists(tmp_path / "nb.relatedness.tsv")
    with pytest.raises(ValueError, match="no summary of its listed related pairs"):
        compose_ldscores(str(d / "ref"), ss, an, str(tmp_path / "u"), curve=str(tmp_path / "nb"),
                         relatedness="uniform", log=quiet)


def test_custom_bins_and_a_release_without_bins(small, tmp_path):
    """Other MAF cuts, LD quantiles and class cut give files named after them whose fine bins pool into the coarse
    ones; a class cut that is not one of the MAF cuts is refused; a release made with bins=False has no annotation,
    its README asks the analyst for one, and compose reads the release with another annotation."""
    from summit.admixed_ld.bins import bin_scheme
    d = small["dir"]
    sc = bin_scheme("0.01,0.05,0.2", 3, 0.05)
    rec = make_bins(str(d / "cohort"), str(tmp_path / "c"), scheme=sc, log=quiet)
    tag = "mafcuts0.01-0.05-0.2_ld3_class0.05"
    assert rec["files"] == dict(pooled=f"c.{tag}.bins6.annot.gz", fine=f"c.{tag}.bins12.annot.gz",
                                classes=f"c.{tag}.maf2.annot.gz")
    fine = pd.read_csv(tmp_path / rec["files"]["fine"], sep="\t")
    pooled = pd.read_csv(tmp_path / rec["files"]["pooled"], sep="\t")
    tab, _ = local_ld_scores(str(d / "cohort"), log=quiet)
    mb = np.digitize(tab.MAF.to_numpy(), [0.01, 0.05, 0.2])
    names = [f"maf_{lo}_{hi}_ld{q}" for lo, hi in (("0", "0.01"), ("0.01", "0.05"), ("0.05", "0.2"), ("0.2", "0.5"))
             for q in (1, 2, 3)]
    assert [c for c in fine.columns[4:]] == [n for n in names if n in fine.columns]
    for b, (lo, hi) in enumerate((("0", "0.01"), ("0.01", "0.05"), ("0.05", "0.2"), ("0.2", "0.5"))):
        cols = [f"maf_{lo}_{hi}_ld{q}" for q in (1, 2, 3) if f"maf_{lo}_{hi}_ld{q}" in fine.columns]
        sel = mb == b
        assert (fine[cols].sum(axis=1).to_numpy() == sel).all()
        if sel.sum() >= 3:                                       # tertiles of the local score within the MAF bin
            q = fine.loc[sel, cols].to_numpy().argmax(axis=1)
            assert (np.diff(q[np.argsort(tab.local.to_numpy()[sel], kind="stable")]) >= 0).all()
    for q in (1, 2, 3):
        low = [c for c in fine.columns[4:] if c.endswith(f"_ld{q}") and float(c.split("_")[2]) <= 0.05]
        high = [c for c in fine.columns[4:] if c.endswith(f"_ld{q}") and float(c.split("_")[2]) > 0.05]
        np.testing.assert_array_equal(fine[low].sum(axis=1), pooled[f"maf_low_ld{q}"])
        np.testing.assert_array_equal(fine[high].sum(axis=1), pooled[f"maf_high_ld{q}"])
    with pytest.raises(ValueError, match="must be one of the MAF cuts"):
        bin_scheme("0.01,0.2", 4, 0.05)
    out = tmp_path / "release"
    rel = make_release(str(d / "cohort"), str(d / "ref"), str(out), covar=str(d / "cohort.cov"), cell_cm=0.5,
                       bins=False, log=quiet)
    assert rel["bins"] is None and not [f for f in os.listdir(out) if f.endswith(".annot.gz")]
    assert "brings the annotation" in " ".join((out / "README.txt").read_text().split())
    compose_ldscores(str(d / "ref"), [str(d / "ss" / "trait0.sumstats")], str(tmp_path / rec["files"]["classes"]),
                     str(tmp_path / "ld"), release=str(out), log=quiet)
    assert os.path.exists(tmp_path / "ld.gw.ldscore.gz")


def test_every_flag_is_in_the_docs():
    """Every flag that the admixed-ld code records is in the flags table of the SUMMIT-MIX details page."""
    code = "".join(p.read_text() for p in (REPO / "src" / "summit" / "admixed_ld").glob("*.py"))
    flags = set(re.findall(r'flag="([a-z0-9_]+)"', code))
    doc = (REPO / "docs" / "wiki" / "SUMMIT-MIX-details.md").read_text()
    table = set(re.findall(r"^\| `([a-z0-9_]+)` \|", doc, flags=re.M))
    assert flags, "no flags found in the code"
    assert flags <= table, sorted(flags - table)
    assert table <= flags, sorted(table - flags)
