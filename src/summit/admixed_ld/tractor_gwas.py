"""A Tractor-style GWAS of an admixed cohort with the sandwich standard errors of the track moments
(`summit admixed-ld tractor-gwas`).

Per SNP j and trait, on the trait's complete cases (the trait and every covariate), the model is Tractor's (Atkinson et
al. 2021, run_tractor.R, two ancestries):
    y = 1 + covariates + l_j + g^0_j + g^1_j + e,
with l_j the number of the person's haplotypes of ancestry 1 at the SNP (from the admix-kit .lanc) and g^s_j the number
of counted (ALT) alleles on the person's haplotypes of ancestry s (from the phased PGEN), s = 0, 1 the .lanc's codes and
Tractor's anc0, anc1. The fit projects the covariates out of the three regressors (Frisch-Waugh) and solves one 3 x 3
system per SNP, vectorized over blocks of SNPs. With x~_i the person's projected regressors, S = sum_i x~_i x~_i', e the
full model's residuals and k = rank(1, covariates) + 3 parameters:
    OLS (se_anc<s>, Tractor's)        V = s^2 S^-1,  s^2 = e'e / (n - k)
    covariates-only (se_hcn_anc<s>)   V = S^-1 (sum_i x~_i x~_i' e0_i^2 / (1 - h_i)) S^-1
    HC1 (se_hc_anc<s>, with hc1)      V = S^-1 (sum_i x~_i x~_i' e_i^2) S^-1 n / (n - k)
where e0 is the residual of the trait on the covariates only and h_i the person's hat value under the covariates. The
track moments of `tractor-sumstats` and `track-rg` read z^2 - 1 as a moment over distinct people, which holds when SE^2
is the diagonal sum_i c_i^2 y_i^2 of beta = sum_i c_i y_i; the covariates-only sandwich estimates it at any allele
frequency and with noise that follows ancestry. HC1 takes the full model's residuals, which the fit shrinks at alleles
carried by few people on one track, so it is biased low there. AF_anc<s> = sum g^s / sum l^s and LAprop_anc<s> =
mean l^s / 2 (Tractor's columns, with l^0 = 2 - l) are those of the trait's sample. A SNP whose design is singular in
a trait's sample (a track without variation, or one track's alleles equal to its local ancestry) has NA statistics.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

_MISSING = ("", "NA", "NaN", "nan", "na", "N/A", ".", "None", "null")
_FLOAT = "%.12g"
_SITE_ROWS = 2_000_000            # people x SNPs per block by default (about 64 bytes each per thread)


# --------------------------------------------------------------------------------------------------------- inputs
def _chromosome_inputs(pgen, lanc, chromosomes):
    """[(label, PGEN prefix, .lanc path)]: one entry, or one per chromosome when the paths contain {chr}."""
    if ("{chr}" in pgen) != ("{chr}" in lanc):
        raise ValueError("Give --pgen and --lanc both with {chr} (one PGEN and one .lanc per chromosome) or both "
                         "without it.")
    if "{chr}" not in pgen:
        if chromosomes:
            raise ValueError("--chromosomes needs --pgen and --lanc with {chr}.")
        return [(None, pgen, lanc)]
    explicit = bool(chromosomes)
    labels = list(chromosomes) if explicit else [str(c) for c in range(1, 23)]
    out = []
    for c in labels:
        p, la = pgen.replace("{chr}", c), lanc.replace("{chr}", c)
        prefix = p[:-5] if p.lower().endswith(".pgen") else p
        if not os.path.exists(prefix + ".pgen"):
            if explicit:
                raise ValueError(f"Chromosome {c}: {prefix}.pgen not found.")
            continue
        if not os.path.exists(la):
            raise ValueError(f"Chromosome {c}: the PGEN exists but the local ancestry {la} does not.")
        out.append((c, prefix, la))
    if not out:
        raise ValueError(f"No PGEN matches {pgen} for chromosomes {labels[0]}-{labels[-1]}.")
    return out


def _psam_has_fid(path):
    with open(path) as fh:
        for line in fh:
            if line.startswith("##") or not line.strip():
                continue
            return line.split()[0] == "#FID"
    return False


def _keys(df, use_fid):
    return (df["FID"].astype(str) + "\t" + df["IID"].astype(str)) if use_fid else df["IID"].astype(str)


def _read_table(path, what):
    """FID/IID table with one or more value columns, as strings (missing tokens kept for _numeric)."""
    t = pd.read_csv(path, sep=r"\s+", dtype=str, keep_default_na=False)
    short = np.flatnonzero((t.isna() | (t == "")).any(axis=1).to_numpy())    # white space never gives "" otherwise
    if short.size:
        raise ValueError(f"{path}: {short.size} rows have fewer fields than the header (first: line {short[0] + 2}); "
                         "fields are separated by white space, so write missing values as NA, not as empty fields.")
    t.columns = [c.lstrip("#") if c.lstrip("#") in ("FID", "IID") else c for c in map(str, t.columns)]
    if "IID" not in t.columns:
        raise ValueError(f"The {what} file {path} needs a header line with FID and IID columns.")
    cols = [c for c in t.columns if c not in ("FID", "IID")]
    if not cols:
        raise ValueError(f"The {what} file {path} has no columns besides FID and IID.")
    return t, cols


def _numeric(t, cols, path, what, log):
    out = np.empty((len(t), len(cols)))
    for k, c in enumerate(cols):
        s = t[c].astype(str).str.strip()
        miss = s.isin(_MISSING)
        v = pd.to_numeric(s.where(~miss), errors="coerce")
        bad = v.isna().to_numpy() & ~miss.to_numpy()
        if bad.any():
            raise ValueError(f"{what} {c} in {path} is not numeric (e.g. {s[bad].iloc[0]!r}); code categorical "
                             "covariates as 0/1 columns and missing values as NA.")
        x = v.to_numpy(np.float64)
        if np.isinf(x).any():
            raise ValueError(f"{what} {c} in {path} has infinite values.")
        n9 = int(np.sum(x == -9))
        if n9:
            log(f"[admixed-ld] WARNING: {n9} values of -9 in {what} {c} are read as numbers; write missing values "
                "as NA")
        out[:, k] = x
    return out


def _aligned(path, what, keys, use_fid_psam, log):
    """Values of the file's columns for the analysis samples (keys in PGEN order); every sample must be listed."""
    t, cols = _read_table(path, what)
    use_fid = use_fid_psam and "FID" in t.columns
    k = _keys(t, use_fid)
    dup = k[k.duplicated()]
    if len(dup):
        raise ValueError(f"{path} lists {len(dup)} samples more than once (first: {dup.iloc[0].replace(chr(9), ' ')}).")
    want = keys if use_fid else keys.str.split("\t").str[-1]
    if not use_fid and want.duplicated().any():
        raise ValueError(f"The {what} file {path} has no FID column and the PGEN's IIDs are not unique; add FID.")
    pos = pd.Index(k).get_indexer(want)
    absent = np.flatnonzero(pos < 0)
    by = "FID and IID" if use_fid else "IID"
    if absent.size == len(want):
        raise ValueError(f"None of the {len(want)} PGEN samples is in the {what} file {path} (matched by {by}).")
    if absent.size:
        ex = ", ".join(want.iloc[absent[:3]].str.replace("\t", " "))
        raise ValueError(f"{absent.size} of the {len(want)} PGEN samples (after --keep) are not in the {what} file "
                         f"{path} (matched by {by}; first: {ex}). List them with NA values or leave them out with "
                         "--keep.")
    extra = len(t) - len(want)
    if extra:
        log(f"[admixed-ld] tractor-gwas: {extra} samples of {path} are not among the PGEN samples used (ignored)")
    return _numeric(t.iloc[pos].reset_index(drop=True), cols, path, what, log), cols


def _keep_rows(samples, use_fid, keep, log):
    idx = np.arange(len(samples))
    if keep is None:
        return idx
    k = pd.read_csv(keep, sep=r"\s+", header=None, dtype=str, keep_default_na=False)
    if k.shape[1] >= 2:
        want = set((k[0] + "\t" + k[1]) if use_fid else k[1])
    else:
        want = set(k[0])
    key = _keys(samples, use_fid) if (use_fid and k.shape[1] >= 2) else samples["IID"].astype(str)
    idx = idx[key.isin(want).to_numpy()]
    if idx.size == 0:
        raise ValueError(f"No PGEN samples are in {keep}.")
    if len(want) > idx.size:
        log(f"[admixed-ld] tractor-gwas: {len(want) - idx.size} samples of {keep} are not in the PGEN")
    return idx


class _LocalAncestry:
    """admix-kit local ancestry of one chromosome for chosen samples, held as flat segment arrays so that a block of
    SNPs is expanded for every person at once. Codes must be 0 and 1 (two ancestries)."""

    def __init__(self, path, n_sites, n_samples, rows):
        from ..local_ancestry.markers import read_lanc
        with open(path) as fh:
            head = fh.readline().split()
        if len(head) != 2 or not all(x.isdigit() for x in head):
            raise ValueError(f"{path} is not an admix-kit .lanc file (its first line must be 'n_snp n_indiv').")
        if int(head[0]) != n_sites:
            raise ValueError(f"{path} has {head[0]} variants but the PGEN has {n_sites} (.pvar rows); the .lanc must "
                             "be aligned to the PGEN's variants.")
        if int(head[1]) != n_samples:
            raise ValueError(f"{path} has {head[1]} samples but the PGEN has {n_samples} (.psam rows); the .lanc must "
                             "list the PGEN's samples in .psam order.")
        try:
            segs = read_lanc(path, n_sites)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{path} is not a well-formed admix-kit .lanc file ({exc or 'an empty line'}).") from None
        M = int(n_sites)
        ends, starts, codes = [], [], []
        for i, r in enumerate(rows):
            st, an = segs[int(r)]
            st = st.astype(np.int64)
            if st[0] <= 0 or np.any(np.diff(st) <= 0):
                raise ValueError(f"{path}: the segment ends of sample {int(r) + 1} are not increasing.")
            ends.append(i * M + st)
            starts.append(i * M + np.r_[0, st[:-1]])
            codes.append(an)
        self.end = np.concatenate(ends)
        self.start = np.concatenate(starts)
        self.codes = np.ascontiguousarray(np.concatenate(codes), dtype=np.int8)
        bad = np.setdiff1d(np.unique(np.concatenate([an.ravel() for _, an in segs])), [0, 1])
        if bad.size:
            raise ValueError(f"{path} has ancestry code {int(bad[0])}: tractor-gwas fits two ancestries (codes 0 and "
                             "1). Run local ancestry with two sources, or split the cohort.")
        self.n, self.m = len(rows), M

    def block(self, s0, s1):
        """Ancestry codes (n, s1 - s0, 2) int8 of both haplotypes at SNP indices [s0, s1)."""
        base = np.arange(self.n, dtype=np.int64) * self.m
        a = np.searchsorted(self.end, base + s0, side="right")
        b = np.searchsorted(self.end, base + s1 - 1, side="right")
        cnt = b - a + 1
        idx = np.arange(int(cnt.sum())) + np.repeat(a - (np.cumsum(cnt) - cnt), cnt)
        lo = np.maximum(self.start[idx], np.repeat(base + s0, cnt))
        hi = np.minimum(self.end[idx], np.repeat(base + s1, cnt))
        return np.repeat(self.codes[idx], hi - lo, axis=0).reshape(self.n, s1 - s0, 2)


# ------------------------------------------------------------------------------------------------------- the model
def _design(y, C, cov_names, min_cases):
    """One trait's complete cases (y finite; C is complete already), the orthonormal basis Q of [1, covariates] there,
    the covariates-only residuals and the kind-n weights e0^2 / (1 - h)."""
    from ..ldscore.gw_ldscore import _orthonormal_covariate_basis
    mask = np.isfinite(y)
    n = int(mask.sum())
    if n < min_cases:
        return dict(n=n, error=f"{n} complete cases (fewer than {min_cases})")
    yt = y[mask]
    if np.ptp(yt) == 0:
        return dict(n=n, error="the trait is constant in its complete cases")
    Cm = C[mask]
    sd = Cm.std(0)
    used = sd > 0
    Z = np.column_stack([np.ones(n), (Cm[:, used] - Cm[:, used].mean(0)) / sd[used]])
    Q, rank = _orthonormal_covariate_basis(Z)[:2]
    Q = np.ascontiguousarray(Q)
    df = n - rank - 3
    if df < 1:
        return dict(n=n, error=f"{n} complete cases leave no residual degrees of freedom")
    yr = yt - Q @ (Q.T @ yt)
    one_h = 1.0 - np.einsum("ij,ij->i", Q, Q)
    w = np.where(one_h > 1e-10, yr * yr / np.where(one_h > 1e-10, one_h, 1.0), 0.0)
    return dict(n=n, mask=mask, Q=Q, rank=int(rank), df=int(df), yr=yr, w=w, yy=float(yr @ yr), error=None,
                cov_rank=int(rank) - 1 + 2, dropped=[c for c, u in zip(cov_names, used) if not u],
                one_person_covariates=int(np.sum(one_h <= 1e-10)))


def _groups(designs):
    """Traits with the same complete cases share the projected regressors: one group per distinct sample."""
    out = {}
    for t, d in enumerate(designs):
        key = d["mask"].tobytes()
        g = out.get(key)
        if g is None:
            g = out[key] = dict(idx=None if d["mask"].all() else np.flatnonzero(d["mask"]), Q=d["Q"], n=d["n"],
                                df=d["df"], traits=[])
        g["traits"].append(t)
    for g in out.values():
        ds = [designs[t] for t in g["traits"]]
        g["Yr"] = np.ascontiguousarray(np.stack([d["yr"] for d in ds]))
        g["W"] = np.ascontiguousarray(np.stack([d["w"] for d in ds]))
        g["yy"] = np.array([d["yy"] for d in ds])
    return list(out.values())


def _fit_group(X, g, hc1):
    """Statistics of one sample group's traits for one block. X: (3, n, B) float64 regressors [l, g^0, g^1] of the
    group's people (overwritten by their covariate residuals). Returns {trait index: {statistic: (B,) array}}."""
    Q, Yr, W, df, n = g["Q"], g["Yr"], g["W"], g["df"], g["n"]
    T, B = Yr.shape[0], X.shape[2]
    s = X.sum(axis=1)                                                     # raw sums, for AF and LAprop
    ss = np.einsum("aij,aij->aj", X, X) - s * s / n                       # centred sums of squares
    for i in range(3):
        X[i] -= Q @ (Q.T @ X[i])
    S = np.empty((B, 3, 3))
    Mn = np.empty((T, B, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            P = X[i] * X[j]
            S[:, i, j] = S[:, j, i] = P.sum(axis=0)
            Mn[:, :, i, j] = Mn[:, :, j, i] = W @ P
    del P
    xty = np.stack([Yr @ X[i] for i in range(3)], axis=-1)                # T x B x 3
    dg = np.diagonal(S, axis1=1, axis2=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        good = (np.all(ss.T > 0, axis=1) & np.all(dg > 1e-8 * ss.T, axis=1)
                & (np.linalg.det(S) > 1e-10 * np.prod(dg, axis=1)))
    S[~good] = np.eye(3)
    Si = np.linalg.inv(S)
    b = np.einsum("jab,tjb->tja", Si, xty)
    s2 = (g["yy"][:, None] - np.einsum("tja,tja->tj", b, xty)) / df
    dSi = np.diagonal(Si, axis1=1, axis2=2)                               # B x 3
    v_ols = s2[:, :, None] * dSi[None]
    v_n = np.einsum("jab,tjbc,jac->tja", Si, Mn, Si)
    v_hc = None
    if hc1:
        v_hc = np.empty((T, B, 3))
        for t in range(T):
            e = Yr[t][:, None] - X[0] * b[t, :, 0] - X[1] * b[t, :, 1] - X[2] * b[t, :, 2]
            np.multiply(e, e, out=e)
            Mh = np.empty((B, 3, 3))
            for i in range(3):
                for j in range(i, 3):
                    Mh[:, i, j] = Mh[:, j, i] = np.einsum("ij,ij,ij->j", e, X[i], X[j])
            v_hc[t] = np.einsum("jab,jbc,jac->ja", Si, Mh, Si) * (n / df)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = {0: s[1] / (2.0 * n - s[0]), 1: s[2] / s[0]}
        lap = {1: s[0] / (2.0 * n)}
        lap[0] = 1.0 - lap[1]
        out = {}
        for k, t in enumerate(g["traits"]):
            r = {}
            for anc, ix in ((0, 1), (1, 2)):
                beta = np.where(good, b[k, :, ix], np.nan)
                se = np.where(good, np.sqrt(np.maximum(v_ols[k, :, ix], 0.0)), np.nan)
                sen = np.where(good, np.sqrt(np.maximum(v_n[k, :, ix], 0.0)), np.nan)
                r[f"beta_anc{anc}"], r[f"se_anc{anc}"], r[f"tval_anc{anc}"] = beta, se, beta / se
                r[f"se_hcn_anc{anc}"], r[f"tval_hcn_anc{anc}"] = sen, beta / sen
                if v_hc is not None:
                    r[f"se_hc_anc{anc}"] = np.where(good, np.sqrt(np.maximum(v_hc[k, :, ix], 0.0)), np.nan)
                r[f"AF_anc{anc}"], r[f"LAprop_anc{anc}"] = freq[anc], lap[anc]
            r["_good"] = good
            out[t] = r
    return out


def _columns(hc1):
    cols = ["CHR", "POS", "ID", "REF", "ALT", "N", "COV_RANK"]
    for anc in (0, 1):
        cols += [f"beta_anc{anc}", f"se_anc{anc}", f"tval_anc{anc}", f"se_hcn_anc{anc}", f"tval_hcn_anc{anc}"]
        cols += [f"se_hc_anc{anc}"] if hc1 else []
        cols += [f"AF_anc{anc}", f"LAprop_anc{anc}"]
    return cols


# ---------------------------------------------------------------------------------------------------- the command
def tractor_gwas(pgen, lanc, pheno, covar, out, *, keep=None, sources=None, chromosomes=None, hc1=False,
                 min_cases=50, step=None, threads=1, log=print):
    """Tractor-style GWAS of every trait of `pheno` (see the module docstring); writes <out>.<trait>.tractor.tsv per
    trait (CHR POS ID REF ALT N COV_RANK and, per ancestry s, beta_anc<s>, se_anc<s>, tval_anc<s>, se_hcn_anc<s>,
    tval_hcn_anc<s>, [se_hc_anc<s>], AF_anc<s>, LAprop_anc<s>; ALT is the counted allele) and <out>.tractor_gwas.json.

    pgen: phased PLINK 2 prefix, or a path with {chr}; lanc: admix-kit .lanc aligned to the PGEN's variants and samples
    (or with {chr}); pheno: FID IID and one column per trait; covar: FID IID and the covariates (PCs included); keep:
    FID IID or IID per line; sources: names of the .lanc's codes 0 and 1 (Tractor's anc0, anc1), for the record and
    tractor-sumstats; threads: blocks fitted in parallel (each with one BLAS thread). Returns the record."""
    from ..ldscore.genotype_source import read_psam_sample_ids, read_pvar_variants, resolve_genotype_input
    import pgenlib
    from threadpoolctl import threadpool_limits
    t_start = time.time()
    names = list(sources) if sources else ["anc0", "anc1"]
    if len(names) != 2:
        raise ValueError(f"tractor-gwas fits two ancestries (the .lanc's codes 0 and 1); --sources names "
                         f"{len(names)}. Give two names, for codes 0 and 1 in that order.")
    if len(set(names)) != 2:
        raise ValueError("Give two distinct --sources names.")
    if min_cases < 1:
        raise ValueError("min_cases must be positive.")
    chroms = _chromosome_inputs(pgen, lanc, chromosomes)
    gis = []
    for c, prefix, la in chroms:
        gi = resolve_genotype_input(prefix + ".pgen")
        gis.append(gi)
    samples = read_psam_sample_ids(gis[0].sample_path)
    use_fid = _psam_has_fid(gis[0].sample_path)
    for gi in gis[1:]:
        other = read_psam_sample_ids(gi.sample_path)
        if len(other) != len(samples) or not (other.values == samples.values).all():
            raise ValueError(f"{gi.sample_path} does not list the samples of {gis[0].sample_path} in the same order; "
                             "every chromosome's PGEN must hold the same people.")
    use = _keep_rows(samples, use_fid, keep, log)
    keys = _keys(samples.iloc[use].reset_index(drop=True), use_fid)
    Y, traits = _aligned(pheno, "phenotype", keys, use_fid, log)
    C, cov_names = _aligned(covar, "covariate", keys, use_fid, log)
    bad_names = [t for t in traits if os.sep in t or t in (".", "..")]
    if bad_names:
        raise ValueError(f"Trait names become file names; rename {', '.join(bad_names)}.")
    complete = np.isfinite(C).all(axis=1)
    sub = complete & np.isfinite(Y).any(axis=1)
    if not sub.any():
        raise ValueError("No sample has every covariate and a trait value.")
    rows = use[sub].astype(np.uint32)                                     # PGEN sample indices read, in PGEN order
    Y, C = Y[sub], C[sub]
    designs = [_design(Y[:, t], C, cov_names, min_cases) for t in range(len(traits))]
    few = [f"{t} ({d['error']})" for t, d in zip(traits, designs) if d["error"]]
    if few:
        raise ValueError(f"Traits with too few usable complete cases (trait and every covariate): {'; '.join(few)}. "
                         "Leave them out of --pheno or lower --min-cases.")
    for t, d in zip(traits, designs):
        if d["dropped"]:
            log(f"[admixed-ld] tractor-gwas: {t}: covariates constant in its sample, left out: "
                f"{', '.join(d['dropped'])}")
        if d["rank"] - 1 < len(cov_names) - len(d["dropped"]):
            log(f"[admixed-ld] tractor-gwas: {t}: covariates of rank {d['rank'] - 1} (linearly dependent directions "
                "left out)")
        if d["one_person_covariates"]:
            log(f"[admixed-ld] WARNING: {t}: {d['one_person_covariates']} people are fitted exactly by the covariates "
                "(hat value 1); they add nothing to the fit")
    groups = _groups(designs)
    n_read = rows.size
    step = int(step) if step else int(np.clip(_SITE_ROWS // n_read, 16, 2048))
    if step < 1:
        raise ValueError("step must be positive.")
    threads = max(1, int(threads))
    log(f"[admixed-ld] tractor-gwas: {len(samples)} PGEN samples, {use.size} after --keep, {n_read} with every "
        f"covariate and a trait; {len(traits)} traits in {len(groups)} sample groups; {len(cov_names)} covariates; "
        f"anc0 = {names[0]}, anc1 = {names[1]}; {len(chroms)} PGEN file(s); blocks of {step} SNPs, {threads} threads")
    cols = _columns(hc1)
    files = {t: f"{out}.{t}.tractor.tsv" for t in traits}
    tmp = {t: f + ".partial" for t, f in files.items()}
    handles = {}
    n_snps, n_bad = 0, np.zeros(len(traits), np.int64)
    per_chrom = []
    try:
        for t in traits:
            handles[t] = open(tmp[t], "w")
            handles[t].write("\t".join(cols) + "\n")
        for (c, prefix, la_path), gi in zip(chroms, gis):
            t0 = time.time()
            v = read_pvar_variants(gi.variant_path)
            M = len(v)
            try:
                probe = pgenlib.PgenReader(os.fsencode(gi.genotype_path), raw_sample_ct=len(samples), variant_ct=M)
            except Exception as exc:
                raise ValueError(f"{gi.genotype_path} does not open with the counts of its .psam ({len(samples)}) and "
                                 f".pvar ({M}): {exc}") from None
            try:
                if int(probe.get_variant_ct()) != M or int(probe.get_raw_sample_ct()) != len(samples):
                    raise ValueError(f"{gi.genotype_path} does not match its .pvar/.psam counts.")
                if not probe.hardcall_phase_present():
                    raise ValueError(f"{gi.genotype_path} holds no phased genotypes. Tractor needs phased haplotypes: "
                                     "make the PGEN from the phased VCF that local ancestry was called on (plink2 "
                                     "--vcf phased.vcf.gz --make-pgen keeps the phase).")
            finally:
                probe.close()
            lanc_obj = _LocalAncestry(la_path, M, len(samples), rows)
            meta_l = [v[c].tolist() for c in ("CHR", "BP", "SNP", "A2", "A1")]    # CHR POS ID REF ALT
            local, readers = threading.local(), []
            lock = threading.Lock()

            def work(s0, gi=gi, M=M, lanc_obj=lanc_obj, meta_l=meta_l, local=local, readers=readers, lock=lock):
                s1 = min(M, s0 + step)
                r = getattr(local, "reader", None)
                if r is None:
                    r = local.reader = pgenlib.PgenReader(os.fsencode(gi.genotype_path), raw_sample_ct=len(samples),
                                                          variant_ct=M, sample_subset=rows)
                    local.al = np.empty((step, 2 * n_read), np.int32)
                    local.ph = np.empty((step, n_read), np.uint8)
                    with lock:
                        readers.append(r)
                al, ph = local.al[:s1 - s0], local.ph[:s1 - s0]
                r.read_alleles_and_phasepresent_range(s0, s1, al, ph)
                al = al.astype(np.int8)                                   # 0, 1 or -9 (missing)
                miss = np.flatnonzero((al < 0).any(axis=1))
                if miss.size:
                    raise ValueError(f"{gi.genotype_path}: missing genotype calls at {meta_l[2][s0 + miss[0]]} "
                                     f"(and {miss.size - 1} more variants in its block). Tractor needs complete phased "
                                     "calls: impute them or leave those variants out.")
                h0, h1 = al[:, 0::2], al[:, 1::2]
                unph = np.flatnonzero(((h0 != h1) & (ph == 0)).any(axis=1))
                if unph.size:
                    raise ValueError(f"{gi.genotype_path}: unphased heterozygous calls at {meta_l[2][s0 + unph[0]]} "
                                     f"(and {unph.size - 1} more variants in its block). Every call must be phased.")
                h0 = np.ascontiguousarray(h0.T, dtype=np.int8)
                h1 = np.ascontiguousarray(h1.T, dtype=np.int8)
                a = lanc_obj.block(s0, s1)
                a0, a1 = a[:, :, 0], a[:, :, 1]
                g1 = h0 * a0 + h1 * a1                                    # ALT alleles on ancestry-1 haplotypes
                g0 = h0 + h1 - g1
                l1 = a0 + a1
                del h0, h1, a
                res = {}
                for g in groups:
                    X = np.empty((3, g["n"], s1 - s0))
                    for i, z in enumerate((l1, g0, g1)):
                        X[i] = z if g["idx"] is None else z[g["idx"]]
                    res.update(_fit_group(X, g, hc1))
                del X
                lead = [meta_l[k][s0:s1] for k in range(5)]
                text = [_format_rows(lead, designs[k]["n"], designs[k]["cov_rank"],
                                     np.column_stack([res[k][c] for c in cols[7:]])) for k in range(len(traits))]
                return text, [int((~res[k]["_good"]).sum()) for k in range(len(traits))]

            starts = range(0, M, step)
            try:
                if threads > 1:
                    with threadpool_limits(limits=1, user_api="blas"):
                        ex = ThreadPoolExecutor(threads)
                        try:
                            _write(_in_order(ex, work, starts, 2 * threads), traits, handles, n_bad)
                        finally:
                            ex.shutdown(wait=True, cancel_futures=True)
                else:
                    _write(map(work, starts), traits, handles, n_bad)
            finally:
                for r in readers:
                    r.close()
            n_snps += M
            per_chrom.append(dict(chromosome=c, pgen=gi.genotype_path, lanc=os.path.abspath(la_path), snps=M,
                                  seconds=round(time.time() - t0, 1)))
            log(f"[admixed-ld] tractor-gwas: {'chromosome ' + c if c else gi.genotype_path}: {M} SNPs in "
                f"{time.time() - t0:.1f} s")
        for t in traits:
            handles[t].close()
            os.replace(tmp[t], files[t])
    except BaseException:
        for t, fh in handles.items():
            fh.close()
            if os.path.exists(tmp[t]):
                os.remove(tmp[t])
        raise
    src = ",".join(names)
    rec = dict(format="summit-admixed-ld-tractor-gwas", version=1, sources=dict(anc0=names[0], anc1=names[1]),
               model="y ~ 1 + covariates + l + g_anc0 + g_anc1 per SNP on each trait's complete cases (l: ancestry-1 "
                     "haplotypes; g_anc<s>: ALT alleles on ancestry-s haplotypes)",
               se=dict(se_anc="OLS", se_hcn_anc="sandwich with the covariates-only residuals e0^2 / (1 - h)",
                       se_hc_anc="HC1 (full-model residuals)" if hc1 else None),
               pgen=pgen, lanc=lanc, pheno=os.path.abspath(pheno), covar=os.path.abspath(covar),
               keep=None if keep is None else os.path.abspath(keep), covariates=cov_names, samples_pgen=len(samples),
               samples_read=int(n_read), snps=int(n_snps), step=step, threads=threads, chromosomes=per_chrom,
               traits={}, seconds=round(time.time() - t_start, 1))
    for k, (t, d) in enumerate(zip(traits, designs)):
        rec["traits"][t] = dict(file=files[t], N=d["n"], COV_RANK=d["cov_rank"], dropped_covariates=d["dropped"],
                                singular_snps=int(n_bad[k]),
                                next=f"summit admixed-ld tractor-sumstats --tractor {files[t]} --sources {src} "
                                     f"--out {out}.{t}")
        log(f"[admixed-ld] tractor-gwas: {t}: N {d['n']}, COV_RANK {d['cov_rank']}, {n_snps} SNPs ({int(n_bad[k])} "
            f"with a singular design) -> {files[t]}")
    with open(f"{out}.tractor_gwas.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    log(f"[admixed-ld] tractor-gwas: done in {time.time() - t_start:.1f} s; next, per trait: summit admixed-ld "
        f"tractor-sumstats --tractor {out}.<trait>.tractor.tsv --sources {src} --out <prefix>")
    return rec


def _in_order(ex, fn, items, ahead):
    """fn over items on the pool, results in order, with at most `ahead` blocks submitted beyond the one consumed (so
    memory stays bounded and an error stops the run without fitting the remaining blocks)."""
    from collections import deque
    pending = deque()
    try:
        for x in items:
            pending.append(ex.submit(fn, x))
            if len(pending) >= ahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for f in pending:
            f.cancel()


def _format_rows(lead, n, cov_rank, floats):
    """Table rows of one block: the five variant columns (lists), N, COV_RANK and the float statistics (B x k), with
    NA for NaN (Python's % formatting row by row: several times faster than pandas' to_csv with a float format)."""
    k = floats.shape[1]
    fmt = "%s\t%d\t%s\t%s\t%s\t" + f"{n}\t{cov_rank}" + ("\t" + _FLOAT) * k + "\n"
    vals = floats.tolist()
    lines = [fmt % (c, p, i, r, a, *x) for c, p, i, r, a, x in zip(*lead, vals)]
    for j in np.flatnonzero(np.isnan(floats).any(axis=1)):
        x = "\t".join("NA" if v != v else _FLOAT % v for v in vals[j])
        lines[j] = "%s\t%d\t%s\t%s\t%s\t" % tuple(z[j] for z in lead) + f"{n}\t{cov_rank}\t{x}\n"
    return "".join(lines)


def _write(results, traits, handles, n_bad):
    """Append each block's rows, in SNP order, to every trait's table."""
    for text, bad in results:
        for k, t in enumerate(traits):
            handles[t].write(text[k])
            n_bad[k] += bad[k]
