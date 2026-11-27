"""The cohort's release and the analyst's side of `summit admixed-ld`: the MAF/LD bins against a direct computation,
the release folder and compose --release, the 50-person rule of the sample statistic, the covariate rank of summary
statistics without COV_RANK, the summary-statistic files of a directory, LD scores over other partners (refused by
summit --h2/--rg), compose prefixes in --rg, exactly symmetric regional matrices and the annotation of the
cross-population SNPs.
Small simulations (int8 dosages, 900 people, two chromosomes) keep the test process's memory low."""
from __future__ import annotations

import json
import os
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
    """r^2 - (1 - r^2) / (n - 2) over the SNPs within `window` on the chromosome, from the full correlation matrix."""
    X = np.where(np.isnan(G), np.nanmean(G, axis=0), G)
    n = X.shape[0]
    X = X - X.mean(axis=0)
    sd = np.sqrt((X * X).sum(axis=0) / (n - 1))
    Z = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)
    R = Z.T @ Z / (n - 1)
    r2 = R * R
    near = (np.abs(bp[:, None] - bp[None, :]) <= window) & (chrom[:, None] == chrom[None, :])
    return np.where(near, r2 - (1 - r2) / (n - 2), 0.0).sum(axis=1)


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
    assert n == 350 and len(tab) == len(v)
    want = _direct_local_scores(G[50:], v.BP.to_numpy(), v.CHR.to_numpy())
    np.testing.assert_allclose(tab.local.to_numpy(), want, rtol=1e-5, atol=1e-5)
    f = np.nanmean(G[50:], axis=0) / 2
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
                            "cohort.bins8.annot.gz", "cohort.maf2.annot.gz"])
    assert {f["file"] for f in rec["files"]} | {"README.txt", "release.json"} == set(names)
    text = " ".join((out / "README.txt").read_text().split())
    assert "Never share" in text and "at least 50 people" in text
    sf = json.load(open(out / "cohort.sample_factors.json"))
    assert "max" not in sf["leverage"] and len(sf["leverage"]["quantiles_90_99"]) == 2
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
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "summit", *args], cwd=cwd, env=env, capture_output=True, text=True)


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
