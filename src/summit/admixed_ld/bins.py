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
read as `--annot`: PREFIX.bins8.annot.gz (eight bins), PREFIX.bins24.annot.gz (the 24 unpooled bins, six MAF bins by
four LD quartiles, for large samples) and PREFIX.maf2.annot.gz (two MAF classes, the eight bins pooled by MAF class).
Other MAF cuts, numbers of LD quantiles and class cuts (bin_scheme) give files named after them. Only these 0/1 columns are written; the per-SNP scores and frequencies, which are not needed to
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


def bin_scheme(maf_cuts=MAF_CUTS, ld_quantiles=4, class_cut=MAF_CLASS_CUT) -> dict:
    """The MAF/LD bins: MAF bins cut at maf_cuts, ld_quantiles quantiles of the local score within each, and the MAF
    class cut at which they are pooled (one of the MAF cuts). Defaults: the paper's six MAF bins, quartiles, 0.05."""
    cuts = [float(x) for x in (maf_cuts.split(",") if isinstance(maf_cuts, str) else maf_cuts)]
    if not cuts or any(not 0 < c < 0.5 for c in cuts) or any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"--maf-cuts must be increasing MAF values between 0 and 0.5; got {maf_cuts}.")
    nq = int(ld_quantiles)
    if nq < 1:
        raise ValueError(f"--ld-quantiles must be 1 or more; got {ld_quantiles}.")
    cut = float(class_cut)
    if not any(np.isclose(cut, c, rtol=0, atol=1e-12) for c in cuts):
        raise ValueError(f"--maf-class-cut {cut:g} must be one of the MAF cuts ({', '.join(f'{c:g}' for c in cuts)}), "
                         "so that the pooled bins hold whole MAF bins.")
    default = (np.allclose(cuts, MAF_CUTS) if len(cuts) == len(MAF_CUTS) else False) and nq == 4 and \
        np.isclose(cut, MAF_CLASS_CUT)
    edges = [0.0] + cuts + [0.5]
    fine = [f"maf_{edges[b]:g}_{edges[b + 1]:g}_ld{q}" for b in range(len(cuts) + 1) for q in range(1, nq + 1)]
    pooled = [f"maf_{c}_ld{q}" for c in ("low", "high") for q in range(1, nq + 1)]
    # file names: bins<n> for n bins; a scheme other than the default also names its cuts and quantiles
    tag = "" if default else f"mafcuts{'-'.join(f'{c:g}' for c in cuts)}_ld{nq}_class{cut:g}."
    return dict(maf_cuts=cuts, ld_quantiles=nq, class_cut=cut, default=bool(default), fine_names=fine,
                pooled_names=pooled, tag=tag, n_maf_bins=len(cuts) + 1)


def _mafld_index(maf, score, cuts, nq):
    """Each SNP's MAF bin (1 .. len(cuts) + 1) and LD quantile within it (1 .. nq, by rank of the score, ties in
    order; a MAF bin with fewer SNPs than quantiles has one SNP per quantile, lowest first)."""
    maf, score = np.asarray(maf, np.float64), np.asarray(score, np.float64)
    if not (np.isfinite(maf).all() and np.isfinite(score).all()):
        raise ValueError("MAF or local LD scores are missing.")
    mb = np.digitize(maf, cuts) + 1
    q = np.zeros(maf.size, dtype=np.int64)
    for b in range(1, len(cuts) + 2):
        sel = np.flatnonzero(mb == b)
        if sel.size >= nq:
            q[sel] = pd.qcut(pd.Series(score[sel]).rank(method="first"), nq, labels=False).to_numpy() + 1
        elif sel.size:
            q[sel[np.argsort(score[sel], kind="stable")]] = np.arange(1, sel.size + 1)
    return mb, q


def maf_ld_fine_bins(maf, score, scheme=None) -> np.ndarray:
    """The unpooled MAF/LD bins as an SNPs x (MAF bins x quantiles) 0/1 matrix (columns scheme['fine_names']; the
    default scheme gives 24 bins, six MAF bins by four LD quartiles)."""
    sc = scheme or bin_scheme()
    mb, q = _mafld_index(maf, score, sc["maf_cuts"], sc["ld_quantiles"])
    nq = sc["ld_quantiles"]
    A = np.zeros((mb.size, sc["n_maf_bins"] * nq), dtype=np.int8)
    for b in range(1, sc["n_maf_bins"] + 1):
        for qq in range(1, nq + 1):
            A[:, (b - 1) * nq + qq - 1] = (mb == b) & (q == qq)
    return A


def maf_ld_bins(maf, score, scheme=None) -> np.ndarray:
    """The pooled MAF/LD bins as an SNPs x (2 x quantiles) 0/1 matrix (columns scheme['pooled_names']; by default the
    eight bins, columns BIN_NAMES): the fine bins pooled by quantile into MAF below the class cut and at least it."""
    sc = scheme or bin_scheme()
    F = maf_ld_fine_bins(maf, score, sc)
    nq = sc["ld_quantiles"]
    n_low = int(np.flatnonzero(np.isclose(sc["maf_cuts"], sc["class_cut"], rtol=0, atol=1e-12))[0]) + 1
    A = np.zeros((F.shape[0], 2 * nq), dtype=np.int8)
    for qq in range(nq):
        A[:, qq] = F[:, [b * nq + qq for b in range(n_low)]].sum(axis=1)
        A[:, nq + qq] = F[:, [b * nq + qq for b in range(n_low, sc["n_maf_bins"])]].sum(axis=1)
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


def write_bins(out, table, *, scores=False, scheme=None, log=print) -> dict:
    """The pooled MAF/LD bins, the unpooled bins and the two MAF classes from local_ld_scores' table, as annotation
    files: by default PREFIX.bins8.annot.gz, PREFIX.bins24.annot.gz and PREFIX.maf2.annot.gz (another scheme also names
    its cuts and quantiles), and PREFIX.local_ld.tsv.gz with scores. An empty bin or class (for example the MAF class
    below 0.05 of a SNP set filtered on MAF) is left out of its file, with a log line and a record; a pooled bin with
    fewer than 100 SNPs is a warning (its heritability is poorly determined). Returns the record of what was written."""
    sc = scheme or bin_scheme()
    F = maf_ld_fine_bins(table.MAF.to_numpy(), table.local.to_numpy(), sc)
    A = maf_ld_bins(table.MAF.to_numpy(), table.local.to_numpy(), sc)
    nq = sc["ld_quantiles"]
    two = np.column_stack([A[:, :nq].sum(axis=1), A[:, nq:].sum(axis=1)])
    parts = [("pooled", A, sc["pooled_names"]), ("fine", F, sc["fine_names"]), ("classes", two, CLASS_NAMES)]
    if F.shape[1] == A.shape[1]:                     # one MAF cut, the class cut: the fine bins are the pooled ones
        parts = [parts[0], parts[2]]
    sizes = {k: M.sum(axis=0) for k, M, _ in parts}
    dropped = [nm for k, M, names in parts for nm, n_ in zip(names, sizes[k]) if n_ == 0]
    if dropped:
        log(f"[admixed-ld] bins: empty bins and classes left out of the annotations (no SNP has a MAF there): "
            f"{', '.join(dropped)}")
    small = [f"{nm} ({int(n_)})" for nm, n_ in zip(sc["pooled_names"], sizes["pooled"]) if 0 < n_ < 100]
    if small:
        log(f"[admixed-ld] bins: WARNING: bins with fewer than 100 SNPs, whose heritability is poorly determined: "
            f"{', '.join(small)}; the two MAF classes are the more robust fit")
    files = {}
    for k, M, names in parts:
        keep = sizes[k] > 0
        stem = "maf2" if k == "classes" else f"bins{M.shape[1]}"
        files[k] = _write_annot(f"{out}.{sc['tag']}{stem}.annot.gz", table, M[:, keep],
                                [nm for nm, k_ in zip(names, keep) if k_], log)
    if scores:
        p = f"{out}.local_ld.tsv.gz"
        table[["SNP", "CHR", "BP", "MAF", "local"]].to_csv(f"{p}.{os.getpid()}.tmp.gz", sep="\t", index=False,
                                                           float_format="%.6g", compression="gzip")
        os.replace(f"{p}.{os.getpid()}.tmp.gz", p)
        files["local_ld"] = p
        log(f"[admixed-ld] bins: wrote {p} (per-SNP MAF and local score; not needed for the fit)")
    cuts = ", ".join(f"{c:g}" for c in sc["maf_cuts"])
    return dict(files={k: os.path.basename(v) for k, v in files.items()}, n_snps=int(len(table)),
                bins={nm: int(n_) for nm, n_ in zip(sc["pooled_names"], sizes["pooled"]) if n_ > 0},
                fine_bins={nm: int(n_) for nm, n_ in zip(sc["fine_names"], F.sum(axis=0)) if n_ > 0},
                maf_classes={nm: int(n_) for nm, n_ in zip(CLASS_NAMES, sizes["classes"]) if n_ > 0},
                empty_left_out=dropped, window_bp=WINDOW_BP, maf_cuts=list(sc["maf_cuts"]),
                ld_quantiles=nq, maf_class_cut=sc["class_cut"],
                definition=f"{sc['n_maf_bins']} MAF bins (cuts {cuts}) by {nq} quantiles of the local in-sample LD "
                           "score (partners within 1 Mb, r^2 - (1 - r^2) / (n - 2), no covariates) within each "
                           f"(the unpooled bins), pooled by quantile into MAF below {sc['class_cut']:g} (maf_low) and "
                           "at least it (maf_high)")


def make_bins(geno, out, *, keep=None, rows=None, step=None, threads=1, scores=False, scheme=None, log=print) -> dict:
    """`summit admixed-ld bins --geno`: the local scores of the people of `keep` (or the sample rows `rows`) and the
    annotation files."""
    t0 = time.perf_counter()
    sc = scheme or bin_scheme()
    table, n = local_ld_scores(geno, keep=keep, rows=rows, step=step, threads=threads, log=log)
    rec = write_bins(out, table, scores=scores, scheme=sc, log=log)
    rec["n_people"] = int(n)
    log(f"[admixed-ld] bins: {rec['n_snps']} SNPs in {time.perf_counter() - t0:.0f} s")
    return rec


_FREQ_COLUMNS = ("A1_FREQ", "A1FREQ", "EAF", "AF", "AF1", "FREQ", "FREQ1", "FRQ", "MAF", "A1_AF")


def maf_classes(out, *, sumstats=None, freq_col=None, ref=None, mean_ancestry=None, sources=None,
                class_cut=MAF_CLASS_CUT, log=print) -> dict:
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
    cut = float(class_cut)
    if not 0 < cut < 0.5:
        raise ValueError(f"--maf-class-cut must be between 0 and 0.5; got {class_cut}.")
    two = np.column_stack([maf < cut, maf >= cut]).astype(np.int8)
    keep2 = two.sum(axis=0) > 0
    dropped = [nm for nm, k in zip(CLASS_NAMES, keep2) if not k]
    if dropped:
        log(f"[admixed-ld] bins: empty classes left out of the annotation: {', '.join(dropped)}")
    tag = "" if np.isclose(cut, MAF_CLASS_CUT) else f"class{cut:g}."
    path = _write_annot(f"{out}.{tag}maf2.annot.gz", meta, two[:, keep2], [nm for nm, k in zip(CLASS_NAMES, keep2) if k],
                        log)
    log(f"[admixed-ld] bins: two MAF classes (cut at {cut:g}) from {source}")
    return dict(files=dict(maf2=os.path.basename(path)), n_snps=int(len(meta)), source=source,
                maf_classes={nm: int(s) for nm, s in zip(CLASS_NAMES, two.sum(axis=0)) if s > 0},
                empty_left_out=dropped)


def write_record(path, rec):
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1)
