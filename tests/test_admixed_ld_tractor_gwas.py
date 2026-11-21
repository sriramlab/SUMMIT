"""Tests of `summit admixed-ld tractor-gwas`, the Tractor-style GWAS from phased haplotypes and local ancestry with the
sandwich standard errors from the covariates-only residuals (se_hcn_anc<s>): betas, OLS, covariates-only and HC1
standard errors against a direct regression of each SNP and against the track simulator's Tractor reference; the same
tables with other block sizes and threads; `tractor-sumstats` reading them without a full-model-sandwich flag; and the
fail-closed cases (unphased or incomplete genotypes, .lanc files that do not match the PGEN, more than two ancestries,
samples missing from the phenotypes or covariates, traits with too few complete cases). The cohort is small (300
people, two chromosomes of 100 SNPs, int8 haplotypes), so the test process's memory stays low."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from admixed_ld_tracks_sim import _admixed, _mosaic, tractor  # noqa: E402

N_PEOPLE, M_CHR = 300, 100
QUIET = lambda *a: None


def _write_pgen(prefix, al, chrom, bp, ids, ref, alt, fids, iids, phased=True, phasepresent=None):
    """Phased PGEN of alleles `al` (variants x 2n int32, haplotype 0 then 1 of each person; -9 missing)."""
    import pgenlib
    with open(f"{prefix}.psam", "w") as fh:
        fh.write("#FID\tIID\n" + "".join(f"{f}\t{i}\n" for f, i in zip(fids, iids)))
    pd.DataFrame({"#CHROM": chrom, "POS": bp, "ID": ids, "REF": ref, "ALT": alt}).to_csv(
        f"{prefix}.pvar", sep="\t", index=False)
    w = pgenlib.PgenWriter(f"{prefix}.pgen".encode(), len(iids), al.shape[0], False,
                           hardcall_phase_present=phased)
    if phasepresent is None:
        w.append_alleles_batch(np.ascontiguousarray(al, np.int32), all_phased=phased)
    else:
        for j in range(al.shape[0]):
            w.append_partially_phased(np.ascontiguousarray(al[j], np.int32), np.ascontiguousarray(phasepresent[j],
                                                                                                    np.uint8))
    w.close()


def _write_lanc(path, la):
    """admix-kit .lanc of haplotype ancestries `la` (2n x M; rows 2i, 2i + 1 are person i's haplotypes 0 and 1)."""
    n2, M = la.shape
    lines = [f"{M} {n2 // 2}"]
    for i in range(n2 // 2):
        c0, c1 = la[2 * i], la[2 * i + 1]
        ch = np.flatnonzero((c0[1:] != c0[:-1]) | (c1[1:] != c1[:-1])) + 1
        lines.append(" ".join(f"{e}:{c0[s]}{c1[s]}" for s, e in zip(np.r_[0, ch], np.r_[ch, M])))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    """300 admixed people (ancestry 1 = source A, about 60%) with exact local ancestry on two chromosomes, phenotypes
    with effects on both tracks and noise that rises with ancestry, covariates with a few missing values, a --keep
    list without 10 people, and the track variables of every SNP for the direct regressions."""
    d = tmp_path_factory.mktemp("tractor_gwas")
    rng = np.random.default_rng(11)
    n = N_PEOPLE
    fids, iids = [f"F{i}" for i in range(n)], [f"I{i}" for i in range(n)]
    theta = rng.beta(3.0, 2.0, n)
    th_h = np.clip(np.repeat(theta, 2) + rng.normal(0, 0.05, 2 * n), 0, 1)
    tracks = {}
    for c in (1, 2):
        cm = np.sort(rng.uniform(0, 10, M_CHR))
        anc = (rng.random((8, M_CHR)) < rng.uniform(0.15, 0.85, M_CHR)[None, :]).astype(np.int8)
        founders = {s: _mosaic(rng, anc ^ (rng.random(anc.shape) < 0.06).astype(np.int8), cm, 30, 1.0)
                    for s in ("A", "E")}
        hap, la = _admixed(rng, founders, cm, th_h, dict(A=1.2, E=0.6), 0.08)
        bp = np.rint(cm * 1e6).astype(np.int64) + 1
        ids = [f"c{c}_{j}" for j in range(M_CHR)]
        _write_pgen(str(d / f"chr{c}"), hap.T.astype(np.int32), [str(c)] * M_CHR, bp, ids, ["A"] * M_CHR,
                    ["G"] * M_CHR, fids, iids)
        _write_lanc(str(d / f"chr{c}.lanc"), la)
        a = la.astype(np.int8)
        g1 = hap[0::2] * a[0::2] + hap[1::2] * a[1::2]
        tracks[c] = dict(l1=(a[0::2] + a[1::2]).astype(np.float64), g1=g1.astype(np.float64),
                         g0=(hap[0::2] + hap[1::2] - g1).astype(np.float64), hap=hap, la=la, bp=bp, ids=ids)
    ga = np.concatenate([tracks[c]["l1"] for c in (1, 2)], axis=1).mean(1) / 2
    C = np.column_stack([ga, rng.standard_normal(n), rng.uniform(20, 70, n)])
    C[rng.choice(n, 12, replace=False), 2] = np.nan                       # age missing for 12 people
    cj = [(1, 5), (1, 40), (2, 17), (2, 70)]
    g = sum(0.25 * tracks[c]["g1"][:, j] + 0.15 * tracks[c]["g0"][:, j] for c, j in cj)
    noise = rng.standard_normal((n, 3)) * np.exp(0.8 * (ga - ga.mean()) / ga.std())[:, None]
    Y = g[:, None] + 0.4 * C[:, [0]] + noise
    Y[rng.random(n) < 0.15, 1] = np.nan                                    # trait 'part' has its own sample
    Y[rng.random(n) < 0.90, 2] = np.nan                                    # trait 'few': about 30 cases
    fmt = lambda x: "NA" if not np.isfinite(x) else f"{x:.10g}"
    pd.DataFrame({"FID": fids, "IID": iids, "full": map(fmt, Y[:, 0]), "part": map(fmt, Y[:, 1])}).to_csv(
        d / "pheno.tsv", sep="\t", index=False)
    pd.DataFrame({"FID": fids, "IID": iids, "few": map(fmt, Y[:, 2])}).to_csv(d / "pheno_few.tsv", sep="\t",
                                                                              index=False)
    pd.DataFrame({"FID": fids, "IID": iids, "ga": map(fmt, C[:, 0]), "pc": map(fmt, C[:, 1]),
                  "age": map(fmt, C[:, 2])}).to_csv(d / "covar.tsv", sep="\t", index=False)
    keep = np.sort(rng.choice(n, n - 10, replace=False))
    pd.DataFrame({"FID": np.array(fids)[keep], "IID": np.array(iids)[keep]}).to_csv(d / "keep.txt", sep=" ",
                                                                                    header=False, index=False)
    kept = np.zeros(n, bool)
    kept[keep] = True
    # read back exactly what the files hold
    Y = pd.read_csv(d / "pheno.tsv", sep="\t")[["full", "part"]].to_numpy(np.float64)
    C = pd.read_csv(d / "covar.tsv", sep="\t")[["ga", "pc", "age"]].to_numpy(np.float64)
    return dict(dir=d, tracks=tracks, Y=Y, C=C, kept=kept, fids=fids, iids=iids)


def _run(argv):
    from summit.admixed_ld.__main__ import main
    return main(["tractor-gwas"] + [str(x) for x in argv])


def _base(d, out, *extra, pgen=None, lanc=None, pheno=None, covar=None):
    return ["--pgen", pgen or f"{d}/chr{{chr}}", "--lanc", lanc or f"{d}/chr{{chr}}.lanc",
            "--pheno", pheno or f"{d}/pheno.tsv", "--covar", covar or f"{d}/covar.tsv", "--out", out, *extra]


@pytest.fixture(scope="module")
def gwas(cohort):
    d = cohort["dir"]
    out = f"{d}/gw"
    assert _run(_base(d, out, "--keep", f"{d}/keep.txt", "--sources", "E,A", "--hc1")) == 0
    return dict(out=out, tables={t: pd.read_csv(f"{out}.{t}.tractor.tsv", sep="\t") for t in ("full", "part")},
                rec=json.load(open(f"{out}.tractor_gwas.json")))


def _direct(y, C, l1, g0, g1):
    """Per-SNP regression of y on [1, C, l, g^0, g^1] with the full design: betas and the OLS, covariates-only
    sandwich and HC1 SEs of g^0 and g^1."""
    n = y.size
    X0 = np.column_stack([np.ones(n), C])
    H0 = X0 @ np.linalg.solve(X0.T @ X0, X0.T)
    e0, h = y - H0 @ y, np.diag(H0)
    X = np.column_stack([X0, l1, g0, g1])
    A = np.linalg.inv(X.T @ X)
    b = A @ (X.T @ y)
    e = y - X @ b
    df = n - X.shape[1]
    V = dict(se=A * (e @ e) / df, se_hcn=A @ ((X.T * (e0 ** 2 / (1 - h))) @ X) @ A,
             se_hc=A @ ((X.T * e ** 2) @ X) @ A * n / df)
    return b[-2:], {k: np.sqrt(np.diag(v)[-2:]) for k, v in V.items()}


def _sample(cohort, t):
    k = {"full": 0, "part": 1}[t]
    return cohort["kept"] & np.isfinite(cohort["C"]).all(1) & np.isfinite(cohort["Y"][:, k]), k


def test_matches_direct_regression(cohort, gwas):
    """Betas, OLS, covariates-only (kind n) and HC1 SEs of a handful of SNPs equal a direct regression with the full
    design to 1e-10, for a trait on the whole sample and one with its own complete cases; N, COV_RANK, AF and LAprop
    are those of the trait's sample; ALT is the counted allele; only SNPs without variation on a track are NA."""
    for t, tab in gwas["tables"].items():
        m, k = _sample(cohort, t)
        assert (tab.N == m.sum()).all() and (tab.COV_RANK == 3 + 2).all()
        assert gwas["rec"]["traits"][t]["N"] == m.sum()
        assert list(tab.ID[:3]) == ["c1_0", "c1_1", "c1_2"] and (tab.ALT == "G").all() and len(tab) == 2 * M_CHR
        y, C = cohort["Y"][m, k], cohort["C"][m]
        checked = 0
        for c in (1, 2):
            tr = cohort["tracks"][c]
            sub = tab[tab.CHR == c].reset_index(drop=True)
            l1, g0, g1 = tr["l1"][m], tr["g0"][m], tr["g1"][m]
            af1, af0 = g1.sum(0) / l1.sum(0), g0.sum(0) / (2 - l1).sum(0)
            assert np.allclose(sub.AF_anc1, af1, rtol=1e-10) and np.allclose(sub.AF_anc0, af0, rtol=1e-10)
            assert np.allclose(sub.LAprop_anc1, l1.mean(0) / 2, rtol=1e-10)
            assert np.allclose(sub.LAprop_anc0, 1 - l1.mean(0) / 2, rtol=1e-10)
            na = sub.beta_anc0.isna() | sub.beta_anc1.isna()
            degenerate = np.isin(af0, (0, 1)) | np.isin(af1, (0, 1))
            assert (na.to_numpy() == degenerate).all()
            for j in np.flatnonzero(~degenerate)[::12]:
                b, se = _direct(y, C, l1[:, j], g0[:, j], g1[:, j])
                for anc in (0, 1):
                    assert sub[f"beta_anc{anc}"][j] == pytest.approx(b[anc], rel=1e-10, abs=1e-13)
                    for col in ("se", "se_hcn", "se_hc"):
                        assert sub[f"{col}_anc{anc}"][j] == pytest.approx(se[col][anc], rel=1e-10)
                    assert sub[f"tval_anc{anc}"][j] == pytest.approx(b[anc] / se["se"][anc], rel=1e-10, abs=1e-13)
                    assert sub[f"tval_hcn_anc{anc}"][j] == pytest.approx(b[anc] / se["se_hcn"][anc], rel=1e-10,
                                                                         abs=1e-13)
                checked += 1
        assert checked >= 12
    # the covariates-only sandwich is not the full-model one (the noise follows ancestry)
    tab = gwas["tables"]["full"].dropna()
    assert not np.allclose(tab.se_hcn_anc1, tab.se_hc_anc1, rtol=1e-3)


def test_matches_the_simulator_reference(cohort, gwas):
    """Every SNP of the whole-sample trait against the track simulator's Tractor (OLS and HC1, anc1 = track A)."""
    m, k = _sample(cohort, "full")
    tab = gwas["tables"]["full"]
    tr = cohort["tracks"]
    cat = lambda key: np.concatenate([tr[c][key][m] for c in (1, 2)], axis=1)
    ok = tab.beta_anc0.notna().to_numpy()
    assert ok.mean() > 0.9
    ref = tractor(cohort["Y"][m, k:k + 1], cat("g1")[:, ok], cat("g0")[:, ok], cat("l1")[:, ok], cohort["C"][m])
    for anc, s in ((1, "A"), (0, "E")):
        assert np.allclose(tab[f"beta_anc{anc}"][ok], ref[f"beta_{s}"][:, 0], rtol=1e-9, atol=1e-12)
        assert np.allclose(tab[f"se_anc{anc}"][ok], ref[f"se_{s}"][:, 0], rtol=1e-9)
        assert np.allclose(tab[f"se_hc_anc{anc}"][ok], ref[f"sehc_{s}"][:, 0], rtol=1e-8)


def test_blocks_and_threads_give_the_same_tables(cohort, gwas):
    """Small blocks fitted on two threads, one chromosome at a time without {chr}: the same numbers."""
    d = cohort["dir"]
    for c in (1, 2):
        out = f"{d}/thr{c}"
        assert _run(_base(d, out, "--keep", f"{d}/keep.txt", "--hc1", "--step", "7", "--threads", "2",
                          pgen=f"{d}/chr{c}", lanc=f"{d}/chr{c}.lanc")) == 0
        for t, ref in gwas["tables"].items():
            got = pd.read_csv(f"{out}.{t}.tractor.tsv", sep="\t")
            exp = ref[ref.CHR == c].reset_index(drop=True)
            assert list(got.columns) == list(exp.columns)
            num = [x for x in got.columns if got[x].dtype.kind == "f"]
            assert got[[x for x in got.columns if x not in num]].equals(exp[[x for x in exp.columns if x not in num]])
            assert np.allclose(got[num], exp[num], rtol=1e-11, atol=1e-14, equal_nan=True)


def test_tractor_sumstats_reads_the_output(cohort, gwas, tmp_path):
    """tractor-sumstats takes se_hcn (also next to se_hc), records no full-model-sandwich or COV_RANK flag, and its SEs
    are the covariates-only ones; A1 is ALT."""
    from summit.admixed_ld.tracks import tractor_sumstats
    tab = gwas["tables"]["full"]
    rec = tractor_sumstats(f"{gwas['out']}.full.tractor.tsv", str(tmp_path / "ts"), sources=["E", "A"], log=QUIET)
    flags = {f["flag"] for f in rec["flags"]}
    assert rec["se_type"] == "HC" and not flags & {"sandwich_from_full_model_residuals", "cov_rank_without_covariates"}
    for anc, s in ((0, "E"), (1, "A")):
        ss = pd.read_csv(tmp_path / f"ts.{s}.sumstats", sep="\t").set_index("SNP")
        ok = tab.set_index("ID")[f"se_hcn_anc{anc}"].dropna()
        assert np.allclose(ss.SE.reindex(ok.index), ok, rtol=1e-9)
        assert (ss.A1 == "G").all() and (ss.COV_RANK == 5).all()
    assert gwas["rec"]["traits"]["full"]["next"].startswith("summit admixed-ld tractor-sumstats --tractor ")
    assert "--sources E,A" in gwas["rec"]["traits"]["full"]["next"]


def _copy_chr1(cohort, tmp, al=None, la=None, phased=True, phasepresent=None, lanc_text=None):
    """chr1 of the cohort, optionally with other alleles, phase or local ancestry, under tmp/x."""
    tr = cohort["tracks"][1]
    p = str(tmp / "x")
    _write_pgen(p, tr["hap"].T.astype(np.int32) if al is None else al, ["1"] * M_CHR, tr["bp"], tr["ids"],
                ["A"] * M_CHR, ["G"] * M_CHR, cohort["fids"], cohort["iids"], phased=phased,
                phasepresent=phasepresent)
    if lanc_text is not None:
        with open(p + ".lanc", "w") as fh:
            fh.write(lanc_text)
    else:
        _write_lanc(p + ".lanc", tr["la"] if la is None else la)
    return p


def _fails(cohort, tmp, match, *extra, **kw):
    d = cohort["dir"]
    p = kw.pop("p", None) or f"{d}/chr1"
    args = _base(d, str(tmp / "f"), *extra, pgen=p, lanc=kw.pop("lanc", None) or p + ".lanc", **kw)
    with pytest.raises(ValueError, match=match):
        _run(args)
    assert not [f for f in os.listdir(tmp) if f.startswith("f.") and f.endswith((".tsv", ".partial"))]


def test_genotypes_that_are_not_fully_phased_fail_closed(cohort, tmp_path):
    tr = cohort["tracks"][1]
    al = tr["hap"].T.astype(np.int32)
    p = _copy_chr1(cohort, tmp_path, phased=False)
    _fails(cohort, tmp_path, "no phased genotypes", p=p)
    used = np.isfinite(cohort["C"]).all(1)                                # people of the analysis
    het = np.flatnonzero((al[60, 0::2] != al[60, 1::2]) & used)[0]
    ph = np.ones((M_CHR, N_PEOPLE), np.uint8)
    ph[60, het] = 0
    p = _copy_chr1(cohort, tmp_path, phasepresent=ph)
    _fails(cohort, tmp_path, "unphased heterozygous calls at c1_60", "--step", "16", p=p)
    i = np.flatnonzero(used)[3]
    al[30, 2 * i:2 * i + 2] = -9
    p = _copy_chr1(cohort, tmp_path, al=al)
    _fails(cohort, tmp_path, "missing genotype calls at c1_30", p=p)


def test_local_ancestry_that_does_not_match_fails_closed(cohort, tmp_path):
    tr = cohort["tracks"][1]
    d = cohort["dir"]
    text = open(f"{d}/chr1.lanc").read()
    _fails(cohort, tmp_path, "has 99 variants but the PGEN has 100",
           p=_copy_chr1(cohort, tmp_path, lanc_text=text.replace(f"{M_CHR} ", "99 ", 1)))
    lines = text.splitlines()
    p = _copy_chr1(cohort, tmp_path, lanc_text="\n".join([f"{M_CHR} {N_PEOPLE - 1}"] + lines[1:-1]) + "\n")
    _fails(cohort, tmp_path, "samples but the PGEN has", p=p)
    la = tr["la"].copy()
    la[1, 50:] = 2                                                          # a third ancestry
    _fails(cohort, tmp_path, "two ancestries", p=_copy_chr1(cohort, tmp_path, la=la))
    _fails(cohort, tmp_path, "two ancestries", "--sources", "AFR,EUR,NAT")
    _fails(cohort, tmp_path, "both with \\{chr\\} .* or both", lanc=f"{d}/chr1.lanc", p=f"{d}/chr{{chr}}")


def test_samples_and_traits_that_do_not_match_fail_closed(cohort, tmp_path):
    d = cohort["dir"]
    ph = pd.read_csv(f"{d}/pheno.tsv", sep="\t", dtype=str, keep_default_na=False)
    ph.iloc[1:].to_csv(tmp_path / "ph_short.tsv", sep="\t", index=False)
    _fails(cohort, tmp_path, "1 of the 300 PGEN samples .* not in the phenotype file", pheno=tmp_path / "ph_short.tsv")
    cv = pd.read_csv(f"{d}/covar.tsv", sep="\t", dtype=str, keep_default_na=False)
    cv.assign(age=cv.age.replace("NA", "")).to_csv(tmp_path / "cv_empty.tsv", sep="\t", index=False)
    _fails(cohort, tmp_path, "fewer fields than the header", covar=tmp_path / "cv_empty.tsv")
    cv.assign(FID="0").to_csv(tmp_path / "cv_fid.tsv", sep="\t", index=False)
    _fails(cohort, tmp_path, "None of the 300 PGEN samples is in the covariate file", covar=tmp_path / "cv_fid.tsv")
    cv.assign(sex=np.where(np.arange(len(cv)) % 2, "male", "female")).to_csv(tmp_path / "cv_str.tsv", sep="\t",
                                                                               index=False)
    _fails(cohort, tmp_path, "covariate sex .* is not numeric", covar=tmp_path / "cv_str.tsv")
    _fails(cohort, tmp_path, "too few usable complete cases.*few \\(\\d+ complete cases \\(fewer than 50\\)\\)",
           pheno=f"{d}/pheno_few.tsv")
    # a lower --min-cases lets the trait through
    assert _run(_base(d, str(tmp_path / "few"), "--min-cases", "20", pgen=f"{d}/chr1", lanc=f"{d}/chr1.lanc",
                      pheno=f"{d}/pheno_few.tsv")) == 0
    assert pd.read_csv(tmp_path / "few.few.tractor.tsv", sep="\t").N.iloc[0] < 50
