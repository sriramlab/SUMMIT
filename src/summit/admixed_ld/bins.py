"""The MAF/LD bins of the fit (`summit admixed-ld bins`), cut where the genotypes are.

The eight bins: six MAF bins (cut at 0.02, 0.05, 0.1, 0.2 and 0.3), the quartiles of a local in-sample LD score within
each, pooled by quartile into two MAF classes (MAF below 0.05, `maf_low`, and at least 0.05, `maf_high`), so that
`maf_low_ld1` holds the lowest LD quartile of the two rare MAF bins. The local score of SNP j is

    l_j = sum over the SNPs k within 1 Mb of j on its chromosome, j included, of r_jk^2 - (1 - r_jk^2) / (n - 2),

with r_jk the correlation of the two SNPs' genotypes in the n people (missing calls at the SNP's mean, no covariates):
the unbiased squared correlation. A score summed over the whole genome would carry the sample's own sampling noise,
mostly its same-person part, and quartiles cut on it select SNPs on that noise; a local score carries almost none.
MAF is the minor allele's count among the called genotypes of the same people over twice their number. SNPs without
variance in these people are neither rows nor partners. The quartiles use every SNP of the genotype file, so the bins
are cut on the GWAS's SNP set; at least 50 people are needed, as for every released aggregate.

Only the cohort can compute this score. The two MAF classes alone need only the MAF, which an analyst without the
cohort's bins can take from the summary statistics' allele frequencies or from the panels at the mean ancestry
(`maf_classes`).

Outputs are SUMMIT annotation files (CHR BP SNP CM and one 0/1 column per class) that compose and `summit --h2/--rg`
read as `--annot`: PREFIX.bins8.annot.gz (eight bins) and PREFIX.maf2.annot.gz (two MAF classes, the eight bins
pooled by MAF class). Only these 0/1 columns are written; the per-SNP scores and frequencies, which are not needed to
fit, are written only with scores=True (PREFIX.local_ld.tsv.gz).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

WINDOW_BP = 1_000_000
MAF_CUTS = (0.02, 0.05, 0.1, 0.2, 0.3)
LOW_MAF_BINS = (1, 2)                    # the MAF bins below 0.05
MAF_CLASS_CUT = 0.05
BIN_NAMES = [f"maf_{c}_ld{q}" for c in ("low", "high") for q in range(1, 5)]
CLASS_NAMES = ["maf_low", "maf_high"]


def _standardize(X):
    """Missing calls (NaN) at the SNP's mean, centred and scaled to unit variance (n - 1), float32; monomorphic SNPs
    become 0. Returns (standardized, MAF, polymorphic). The MAF is the minor allele's count over twice the called
    genotypes, min(c, 2 n - c) / (2 n), the same number whichever allele is counted, so a SNP exactly at a cut falls in
    the same bin either way."""
    called = ~np.isnan(X)
    cnt = called.sum(axis=0)
    c = np.where(called, X, 0.0).sum(axis=0)
    mu = np.where(cnt > 0, c / np.maximum(cnt, 1), np.nan)
    Xc = np.where(called, X - mu, 0.0)
    sd = np.sqrt((Xc * Xc).sum(axis=0) / (X.shape[0] - 1))
    maf = np.where(cnt > 0, np.minimum(c, 2.0 * cnt - c) / np.maximum(2.0 * cnt, 1), np.nan)
    return np.divide(Xc, sd, out=np.zeros_like(Xc), where=sd > 0).astype(np.float32), maf, sd > 0


def _window_scores(read, bp, n, window, step, out):
    """Local scores of one chromosome's SNPs (positions bp, sorted) into out: rows in chunks of `step`, each against
    its polymorphic partners within `window` bp (a partner exactly `window` away included); read(a, b) gives the
    standardized genotypes of SNPs a..b-1 (n x (b - a)) and which of them are polymorphic."""
    m = bp.size
    for s in range(0, m, step):
        e = min(m, s + step)
        lo = int(np.searchsorted(bp, bp[s] - window, "left"))
        hi = int(np.searchsorted(bp, bp[e - 1] + window, "right"))
        X, poly = read(lo, hi)
        R = (X[:, s - lo:e - lo].T @ X) / np.float32(n - 1)
        r2 = R.astype(np.float64) ** 2
        d = np.abs(bp[s:e, None] - bp[None, lo:hi])
        out[s:e] = np.where((d <= window) & poly[None, :], r2 - (1.0 - r2) / (n - 2), 0.0).sum(axis=1)


def local_ld_scores(geno, *, keep=None, rows=None, window_bp=WINDOW_BP, step=None, threads=1, log=print):
    """Each autosomal SNP's local in-sample LD score and MAF in the people of `keep` (all people without it; or the
    sample rows `rows`, indices into the genotype file's samples), from a PLINK 1 BED or PLINK 2 PGEN prefix. Returns
    (table of SNP, CHR, BP, CM, MAF, local in the genotype file's order, the number of people)."""
    from ._io import AUTOSOMES, DosageBlocks
    from .curve import _cohort_samples
    t0 = time.perf_counter()
    from .sample_factors import MIN_PEOPLE
    if rows is None:
        _, rows, _ = _cohort_samples(geno, None, keep, log)
    n = int(rows.size)
    if n < MIN_PEOPLE:
        raise ValueError(f"The bins would describe {n} people; they need at least {MIN_PEOPLE}, as every released "
                         "aggregate of SUMMIT-MIX does.")
    if step is None:                                   # rows x partners float32 and the chunk's products
        step = int(np.clip((1 << 28) // (4 * max(n, 1)), 256, 4096))
    blocks = DosageBlocks(geno, rows=rows, threads=threads, step=step)
    v = blocks.variants
    chrom = v.CHR.astype(str).str.replace(r"^(?i:chr)", "", regex=True).to_numpy()
    bp = v.BP.to_numpy(np.int64)
    auto = np.isin(chrom, AUTOSOMES)
    if not auto.any():
        raise ValueError(f"{geno} has no autosomal SNPs (chromosomes 1-22).")
    if (~auto).any():
        log(f"[admixed-ld] bins: {int((~auto).sum())} SNPs off the autosomes are left out")
    local, maf, poly = np.full(len(v), np.nan), np.full(len(v), np.nan), np.zeros(len(v), dtype=bool)
    try:
        for c in [x for x in AUTOSOMES if x in set(chrom[auto])]:
            ix = np.flatnonzero(chrom == c)
            ix = ix[np.argsort(bp[ix], kind="stable")]
            def read(a, b, ix=ix):
                Xs, f, pl = _standardize(blocks.read(ix[a:b]))
                maf[ix[a:b]], poly[ix[a:b]] = f, pl
                return Xs, pl

            out = np.zeros(ix.size)
            _window_scores(read, bp[ix], n, window_bp, step, out)
            local[ix] = out
            log(f"[admixed-ld] bins chr{c}: {ix.size} SNPs, mean local LD score {out.mean():.3f} "
                f"({time.perf_counter() - t0:.0f} s)")
    finally:
        blocks.close()
    if (auto & ~np.isfinite(maf)).any():
        bad = int((auto & ~np.isfinite(maf)).sum())
        raise ValueError(f"{bad} SNPs of {geno} have no called genotype in these people; filter them out first.")
    mono = auto & ~poly
    if mono.any():
        # no genotype variance: no LD score and no heritability, so neither a row nor a partner of the bins
        log(f"[admixed-ld] bins: {int(mono.sum())} SNPs monomorphic in these people are left out")
    keep_rows = auto & poly & np.isfinite(local)
    df = pd.DataFrame({"SNP": v.SNP.astype(str), "CHR": chrom, "BP": bp, "CM": v.CM.to_numpy(np.float64),
                       "MAF": maf, "local": local})[keep_rows].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{geno} has no polymorphic autosomal SNP in these people.")
    return df, n


def maf_ld_bins(maf, score) -> np.ndarray:
    """The eight MAF/LD bins as an SNPs x 8 0/1 matrix (columns BIN_NAMES): the MAF bins of MAF_CUTS, the quartiles of
    `score` within each (by rank, ties in order), pooled by quartile into MAF below 0.05 and at least 0.05."""
    maf, score = np.asarray(maf, np.float64), np.asarray(score, np.float64)
    if not (np.isfinite(maf).all() and np.isfinite(score).all()):
        raise ValueError("MAF or local LD scores are missing.")
    mb = np.digitize(maf, MAF_CUTS) + 1
    q = np.zeros(maf.size, dtype=np.int64)
    for b in range(1, len(MAF_CUTS) + 2):
        sel = np.flatnonzero(mb == b)
        if sel.size >= 4:
            q[sel] = pd.qcut(pd.Series(score[sel]).rank(method="first"), 4, labels=False).to_numpy() + 1
        elif sel.size:                                 # fewer SNPs than quartiles: one per quartile, lowest first
            q[sel[np.argsort(score[sel], kind="stable")]] = np.arange(1, sel.size + 1)
    low = np.isin(mb, LOW_MAF_BINS)
    A = np.zeros((maf.size, len(BIN_NAMES)), dtype=np.int8)
    for i, cls in enumerate((low, ~low)):
        for qq in range(1, 5):
            A[:, 4 * i + qq - 1] = cls & (q == qq)
    return A


def _write_annot(path, meta, cols, names, log):
    df = meta[["CHR", "BP", "SNP", "CM"]].reset_index(drop=True).copy()
    for i, nm in enumerate(names):
        df[nm] = np.asarray(cols)[:, i].astype(int)
    tmp = f"{path}.{os.getpid()}.tmp.gz"
    df.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, path)
    log(f"[admixed-ld] bins: wrote {path} ({len(df)} SNPs; " + ", ".join(f"{nm} {int(df[nm].sum())}" for nm in names)
        + ")")
    return path


def write_bins(out, table, *, scores=False, log=print) -> dict:
    """PREFIX.bins8.annot.gz and PREFIX.maf2.annot.gz from local_ld_scores' table (and PREFIX.local_ld.tsv.gz with
    scores). An empty bin or class (for example the MAF class below 0.05 of a SNP set filtered on MAF) is left out of
    the file, with a log line and a record; a bin with fewer than 100 SNPs is a warning (its heritability is poorly
    determined). Returns the record of what was written."""
    A = maf_ld_bins(table.MAF.to_numpy(), table.local.to_numpy())
    sizes = A.sum(axis=0)
    two = np.column_stack([A[:, :4].sum(axis=1), A[:, 4:].sum(axis=1)])
    sizes2 = two.sum(axis=0)
    keep8, keep2 = sizes > 0, sizes2 > 0
    dropped = [nm for nm, k in zip(BIN_NAMES + CLASS_NAMES, list(keep8) + list(keep2)) if not k]
    if dropped:
        log(f"[admixed-ld] bins: empty bins and classes left out of the annotations (no SNP has a MAF there): "
            f"{', '.join(dropped)}")
    small = [f"{nm} ({int(s)})" for nm, s in zip(BIN_NAMES, sizes) if 0 < s < 100]
    if small:
        log(f"[admixed-ld] bins: WARNING: bins with fewer than 100 SNPs, whose heritability is poorly determined: "
            f"{', '.join(small)}; the two MAF classes are the more robust fit")
    names8 = [nm for nm, k in zip(BIN_NAMES, keep8) if k]
    names2 = [nm for nm, k in zip(CLASS_NAMES, keep2) if k]
    files = dict(bins8=_write_annot(f"{out}.bins8.annot.gz", table, A[:, keep8], names8, log),
                 maf2=_write_annot(f"{out}.maf2.annot.gz", table, two[:, keep2], names2, log))
    if scores:
        p = f"{out}.local_ld.tsv.gz"
        table[["SNP", "CHR", "BP", "MAF", "local"]].to_csv(f"{p}.{os.getpid()}.tmp.gz", sep="\t", index=False,
                                                           float_format="%.6g", compression="gzip")
        os.replace(f"{p}.{os.getpid()}.tmp.gz", p)
        files["local_ld"] = p
        log(f"[admixed-ld] bins: wrote {p} (per-SNP MAF and local score; not needed for the fit)")
    return dict(files={k: os.path.basename(v) for k, v in files.items()}, n_snps=int(len(table)),
                bins={nm: int(s) for nm, s in zip(BIN_NAMES, sizes) if s > 0},
                maf_classes={nm: int(s) for nm, s in zip(CLASS_NAMES, sizes2) if s > 0}, empty_left_out=dropped,
                window_bp=WINDOW_BP, maf_cuts=list(MAF_CUTS), maf_class_cut=MAF_CLASS_CUT,
                definition="six MAF bins (cuts 0.02, 0.05, 0.1, 0.2, 0.3), quartiles of the local in-sample LD score "
                           "(partners within 1 Mb, r^2 - (1 - r^2) / (n - 2), no covariates) within each, pooled by "
                           "quartile into MAF below 0.05 (maf_low) and at least 0.05 (maf_high)")


def make_bins(geno, out, *, keep=None, rows=None, step=None, threads=1, scores=False, log=print) -> dict:
    """`summit admixed-ld bins --geno`: the local scores of the people of `keep` (or the sample rows `rows`) and the
    two annotation files."""
    t0 = time.perf_counter()
    table, n = local_ld_scores(geno, keep=keep, rows=rows, step=step, threads=threads, log=log)
    rec = write_bins(out, table, scores=scores, log=log)
    rec["n_people"] = int(n)
    log(f"[admixed-ld] bins: {rec['n_snps']} SNPs in {time.perf_counter() - t0:.0f} s")
    return rec


_FREQ_COLUMNS = ("A1_FREQ", "A1FREQ", "EAF", "AF", "AF1", "FREQ", "FREQ1", "FRQ", "MAF", "A1_AF")


def maf_classes(out, *, sumstats=None, freq_col=None, ref=None, mean_ancestry=None, sources=None, log=print) -> dict:
    """The two MAF classes alone (PREFIX.maf2.annot.gz), for an analyst without the cohort's bins: MAF from an allele
    frequency column of the summary statistics (freq_col, or the first of _FREQ_COLUMNS present), or from the
    reference panels at the given mean ancestry (frequency sum_s m_s p_s, for the reference variants)."""
    if (sumstats is None) == (ref is None):
        raise ValueError("Give either --sumstats (with an allele-frequency column) or --ref with --mean-ancestry.")
    if sumstats is not None:
        df = pd.read_csv(sumstats, sep=r"\s+", compression="infer", dtype={"SNP": str, "CHR": str})
        cols = {c.upper().lstrip("#"): c for c in df.columns}
        col = freq_col if freq_col is not None else next((cols[c] for c in _FREQ_COLUMNS if c in cols), None)
        if col is None or col not in df.columns:
            raise ValueError(f"{sumstats} has no allele-frequency column (looked for {', '.join(_FREQ_COLUMNS)}); "
                             "name it with --freq-col, or give --ref with --mean-ancestry.")
        snp = cols.get("SNP", cols.get("ID"))
        if snp is None:
            raise ValueError(f"{sumstats} has no SNP column.")
        f = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        ok = np.isfinite(f) & (f >= 0) & (f <= 1)
        meta = pd.DataFrame({"SNP": df[snp].astype(str),
                             "CHR": df[cols["CHR"]].astype(str) if "CHR" in cols else "0",
                             "BP": pd.to_numeric(df[cols["BP"]], errors="coerce").fillna(0).astype(np.int64)
                             if "BP" in cols else 0, "CM": 0.0})[ok].reset_index(drop=True)
        maf = np.minimum(f[ok], 1.0 - f[ok])
        source = f"the allele frequencies of {sumstats} (column {col})"
        if (~ok).any():
            log(f"[admixed-ld] bins: {int((~ok).sum())} SNPs without a frequency in {sumstats} are left out")
    else:
        from .reference import Reference
        if mean_ancestry is None:
            raise ValueError("--ref needs --mean-ancestry (the GWAS's mean ancestry, for example the mean_ancestry "
                             "of compose's plan).")
        R = Reference(ref)
        src = list(sources or R.names[:2])
        m = [float(x) for x in str(mean_ancestry).split(",") if x.strip()]
        if len(m) == len(src) - 1:
            m = m + [1.0 - sum(m)]
        if len(m) != len(src) or min(m) < 0:
            raise ValueError(f"--mean-ancestry needs {len(src) - 1} or {len(src)} values in [0, 1].")
        m = np.asarray(m) / np.sum(m)
        rv = R.variants(src)
        p = np.column_stack([rv[f"p_{s}"].to_numpy(np.float64) for s in src]) @ m
        ok = np.isfinite(p)
        meta = rv.loc[ok, ["CHR", "BP", "SNP", "CM"]].reset_index(drop=True)
        maf = np.minimum(p[ok], 1.0 - p[ok])
        source = f"the panels of {ref} at mean ancestry " + ", ".join(f"{s} {x:.3f}" for s, x in zip(src, m))
    poly = maf > 0
    if (~poly).any():
        log(f"[admixed-ld] bins: {int((~poly).sum())} SNPs with MAF 0 are left out")
        meta, maf = meta[poly].reset_index(drop=True), maf[poly]
    two = np.column_stack([maf < MAF_CLASS_CUT, maf >= MAF_CLASS_CUT]).astype(np.int8)
    keep2 = two.sum(axis=0) > 0
    dropped = [nm for nm, k in zip(CLASS_NAMES, keep2) if not k]
    if dropped:
        log(f"[admixed-ld] bins: empty classes left out of the annotation: {', '.join(dropped)}")
    path = _write_annot(f"{out}.maf2.annot.gz", meta, two[:, keep2], [nm for nm, k in zip(CLASS_NAMES, keep2) if k],
                        log)
    log(f"[admixed-ld] bins: two MAF classes (cut at {MAF_CLASS_CUT}) from {source}")
    return dict(files=dict(maf2=os.path.basename(path)), n_snps=int(len(meta)), source=source,
                maf_classes={nm: int(s) for nm, s in zip(CLASS_NAMES, two.sum(axis=0)) if s > 0},
                empty_left_out=dropped)


def write_record(path, rec):
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
