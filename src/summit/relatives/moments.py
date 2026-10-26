"""Related-pair HE moments for SUMMIT heritability with related individuals.

SUMMIT's h2 response for SNP j is the HE-scale moment m_j = z*_j^2, with
y'K y = (n/M) sum_j m_j for the covariate-projected GRM K = X X'/M. Removing the
related pairs P from the HE estimator (keeping every individual) replaces y'K y by
y'K_u y = y'K y - sum_{(i,i') in P} K_ii' y_i y_i', and the second term decomposes
over SNPs:

    sum_{(i,i') in P} K_ii' y_i y_i' = (1/M) sum_j s_j,
    s_j = sum_{(i,i') in P, ordered} x_ij x_i'j y_i y_i'.

With genotypes and phenotype projected on the GWAS covariates and normalized to
N_eff = n - p - 1, SUMMIT's beta/SE moment with the GWAS covariate rank (c = p) is
m_j = (x_j'y)^2 / N_eff exactly, so the per-SNP correction on the m_j scale is

    REL_M_j = s_j / N_eff.

With c = 0 (no covariate rank supplied) the moment's derivative with respect to (x_j'y)^2
differs by a relative p / n^2. SUMMIT subtracts REL_M_j from m_j when a sumstats
file carries a REL_M column, which excludes the related pairs from the numerator
of every SUM-RHE normal equation, for any annotation model or jackknife design.

With LD scores computed in an unrelated sample of the same population no trace
change is needed. If LD scores are computed in the related sample itself, the
related pairs' share of each in-sample LD score is

    LD_REL_j = (M / N_eff^2) sum_{(i,i') in P, ordered} x_ij x_i'j K_ii'.

SUMMIT's in-sample LD scores are l_j = sum_j' r_jj'^2 - M / N_eff, so N_eff + (N_eff^2 / M^2) sum_j (l_j - LD_REL_j)
is exactly the trace of the pair-excluded GRM. For partitioned LD scores l_{j,k} = sum_j' a_j'k r_jj'^2 - M_k / N_eff
the related share is

    LD_REL_{j,k} = (1 / N_eff^2) sum_{(i,i') in P, ordered} x_ij x_i'j W_{k,ii'},   W_{k,ii'} = sum_j' a_j'k x_ij' x_i'j'.

The LD-score step computes these columns (`summit --geno ... --relative-pairs FILE`, via `relative_pair_ld_rel`) on
its own samples, covariates and annotation weights and writes them as LD_REL_<column> next to the LD scores; the fit
subtracts them when the sumstats carry REL_M.

Pairs: a file with IID1/IID2 (optionally FID1/FID2 and a kinship column with a
threshold). By default pairs are closed into families (connected components) and
every within-family pair is excluded.

One number per phenotype (the default). The point estimate needs REL_M only through
sum_j V_j a_jk REL_M_j for each annotation column k; the jackknife needs it block by
block. By default `summit relatives moments` returns, per phenotype, the V-weighted mean
of REL_M over all SNPs (and over each class of a partitioning annotation), and
`summit relatives append` writes that value as a constant REL_M column (class by class
with --annot). The point estimate is unchanged for the single-component model (and for
each class of the annotation used); only the jackknife replicates are approximated, by
spreading each total evenly over its SNPs. In the UK Biobank simulations this changed
jackknife standard errors by less than 0.1%. The totals come from one genotype pass for
every phenotype:

    sum_j V_j a_jk REL_M_j = (2 / N_eff) sum_{(i,i') in P} y_i y_i' W_{k,ii'},   W_{k,ii'} = sum_j V_j a_jk x_ij x_i'j,

with the pair kernels W accumulated in one genotype pass for every complete-case sample,
each with its own covariate basis, mean imputation, projection, N_eff and V weights, so the
values equal those computed from the exact per-SNP REL_M. `--per-snp` also writes the
per-SNP REL_M (one genotype pass per complete-case sample).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from bed_reader import open_bed

from ..ldscore.genotype_source import read_fam_sample_ids, resolve_genotype_input
from ..ldscore.gw_ldscore import read_cov


@dataclass(frozen=True)
class RelativePairs:
    """Unordered within-sample pairs (i < j), as indices into the genotype sample order."""
    i: np.ndarray
    j: np.ndarray
    family: np.ndarray          # family label per sample (singletons have their own label)

    @property
    def npairs(self) -> int:
        return int(self.i.size)


def read_relative_pairs(path: str, sample_ids: pd.DataFrame, *, kinship_col: str | None = None,
                        kinship_min: float | None = None, closure: bool = True) -> RelativePairs:
    df = pd.read_csv(path, sep=r"\s+", dtype=str)
    cols = {c.upper(): c for c in df.columns}
    if "IID1" not in cols or "IID2" not in cols:
        raise ValueError("Relative-pair file needs IID1 and IID2 columns.")
    if kinship_col is not None:
        if kinship_col not in df.columns:
            raise ValueError(f"Kinship column {kinship_col!r} not in {path}.")
        k = pd.to_numeric(df[kinship_col], errors="coerce")
        if kinship_min is not None:
            df = df.loc[k > float(kinship_min)]
    iid = sample_ids["IID"].astype(str).to_numpy()
    if "FID1" in cols and "FID2" in cols:
        key = pd.Index(sample_ids["FID"].astype(str) + "\t" + sample_ids["IID"].astype(str))
        a = key.get_indexer(df[cols["FID1"]].astype(str) + "\t" + df[cols["IID1"]].astype(str))
        b = key.get_indexer(df[cols["FID2"]].astype(str) + "\t" + df[cols["IID2"]].astype(str))
    else:
        if pd.Index(iid).has_duplicates:
            raise ValueError("Duplicate IIDs in the genotype samples; supply FID1/FID2 in the pair file.")
        key = pd.Index(iid)
        a = key.get_indexer(df[cols["IID1"]].astype(str))
        b = key.get_indexer(df[cols["IID2"]].astype(str))
    ok = (a >= 0) & (b >= 0) & (a != b)
    a, b = a[ok], b[ok]
    n = len(iid)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a, b):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    fam = np.array([find(x) for x in range(n)])
    if closure:
        members: dict[int, list[int]] = {}
        for x in range(n):
            members.setdefault(int(fam[x]), []).append(x)
        pi, pj = [], []
        for mem in members.values():
            for u in range(len(mem)):
                for v in range(u + 1, len(mem)):
                    pi.append(mem[u])
                    pj.append(mem[v])
        pi_arr = np.asarray(pi, dtype=np.int64)
        pj_arr = np.asarray(pj, dtype=np.int64)
    else:
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        uniq = np.unique(np.stack([lo, hi], 1), axis=0) if lo.size else np.zeros((0, 2), dtype=np.int64)
        pi_arr, pj_arr = uniq[:, 0].astype(np.int64), uniq[:, 1].astype(np.int64)
    return RelativePairs(i=pi_arr, j=pj_arr, family=fam)


def _project_normalize(X: np.ndarray, C: np.ndarray, n_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute, center, project out the orthonormal covariate basis C, normalize columns to sum of squares n_eff."""
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
    if C is not None and C.shape[1]:
        X = X - C @ (C.T @ X)
    ss = np.einsum("ij,ij->j", X, X)
    ok = np.isfinite(ss) & (ss > 1e-10)
    X[:, ok] *= np.sqrt(n_eff / ss[ok])[None, :]
    X[:, ~ok] = 0.0
    return X, ok


def pairs_within_sample(rp: RelativePairs, idx: np.ndarray, n_all: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions within the sample `idx` (indices into the genotype sample order) of the pairs with both members in it."""
    pos = -np.ones(int(n_all), dtype=np.int64)
    pos[np.asarray(idx, dtype=np.int64)] = np.arange(len(idx))
    a, b = pos[rp.i], pos[rp.j]
    use = (a >= 0) & (b >= 0)
    return a[use], b[use]


def relative_pair_ld_rel(read_block, nsnps: int, a: np.ndarray, b: np.ndarray, A: np.ndarray, C, n_eff: float,
                         step_size: int = 5000) -> np.ndarray:
    """LD_REL_{j,k} (nsnps x K) for unordered pairs (a, b) of one sample, in two genotype passes.

    `read_block(s, e)` returns the sample's genotypes of SNPs s..e-1 (samples x SNPs, NaN or mean-imputed for missing
    calls); C is the orthonormal covariate basis of that sample (or None); A holds the per-SNP weights of the LD-score
    columns (the annotation, times V for SNP-variance-weighted LD scores).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] != int(nsnps):
        raise ValueError(f"LD_REL weights have {A.shape[0]} rows for {nsnps} SNPs.")

    def _blocks():
        for s in range(0, int(nsnps), int(step_size)):
            yield s, min(int(nsnps), s + int(step_size))

    W = np.zeros((a.size, A.shape[1]))                          # first pass: W_k on pairs
    for s, e in _blocks():
        X, _ = _project_normalize(np.array(read_block(s, e), dtype=np.float64), C, n_eff)
        W += (X[a] * X[b]) @ A[s:e]
    out = np.zeros((int(nsnps), A.shape[1]), dtype=np.float64)
    scale = 2.0 / float(n_eff) ** 2                              # unordered pairs counted twice
    for s, e in _blocks():
        X, _ = _project_normalize(np.array(read_block(s, e), dtype=np.float64), C, n_eff)
        out[s:e] = scale * ((X[a] * X[b]).T @ W)
    return out




def _center_impute(X: np.ndarray) -> np.ndarray:
    """Mean-impute missing calls (NaN) and center every column, in place (all-missing columns become 0)."""
    nan = np.isnan(X)
    if nan.any():
        X[nan] = 0.0
        cnt = X.shape[0] - nan.sum(axis=0)
        X -= X.sum(axis=0) / np.maximum(cnt, 1)
        X[nan] = 0.0
    else:
        X -= X.mean(axis=0)
    return X


def _pair_rows_block(X: np.ndarray, C, n_eff: float, rows: np.ndarray):
    """Projected, normalized genotypes of the sample rows `rows`, from centered genotypes X (samples x SNPs).

    Columns are projected on the orthonormal covariate basis C and scaled so that the projected column has sum of
    squares n_eff over the whole sample, as `_project_normalize` does, but only the requested rows are formed:
    ||P x||^2 = ||x||^2 - ||C'x||^2. Returns (rows x SNPs, 1 - R^2 of each SNP on the covariates)."""
    raw = np.einsum("ij,ij->j", X, X)
    Xr = X[rows]
    if C is not None and C.shape[1]:
        B = C.T @ X
        ss = raw - np.einsum("ij,ij->j", B, B)
        Xr -= C[rows] @ B
    else:
        ss = raw.copy()
    ok = np.isfinite(ss) & (ss > 1e-10)
    scale = np.zeros(ss.shape)
    scale[ok] = np.sqrt(n_eff / ss[ok])
    Xr *= scale[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(ok & (raw > 0.0), np.minimum(1.0, ss / raw), 0.0)
    return Xr, r1


def _auto_step(step_size, n: int) -> int:
    """SNPs per genotype block: `step_size`, or about 512 MiB of float64 genotypes (64 to 5000 SNPs)."""
    if step_size is not None:
        return max(1, int(step_size))
    return int(np.clip((1 << 29) // (8 * max(int(n), 1)), 64, 5000))


def read_annotation_classes(annot: str, snps) -> tuple[np.ndarray, list[str]]:
    """0/1 class indicators (SNPs x classes) of a partitioning annotation, aligned to `snps` (absent SNPs: all 0).

    The file has a SNP column (CHR, BP and CM are ignored) and one column per class; every listed SNP must belong to
    exactly one class, as in SUMMIT's MAF/LD bins."""
    df = pd.read_csv(annot, sep=r"\s+", dtype={"SNP": str})
    if "SNP" not in df.columns:
        raise ValueError(f"{annot} has no SNP column.")
    cls = [c for c in df.columns if c not in ("CHR", "BP", "SNP", "CM")]
    if not cls:
        raise ValueError(f"{annot} has no annotation columns.")
    vals = df[cls].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not (np.isin(vals, (0.0, 1.0)).all() and (vals.sum(axis=1) == 1.0).all()):
        raise ValueError(f"{annot} is not a partition (each SNP in exactly one class, 0/1 values); per-class REL_M values "
                         "need one. Leave out --annot to get one REL_M value per phenotype.")
    if df["SNP"].duplicated().any():
        raise ValueError(f"{annot} has duplicate SNP IDs.")
    pos = pd.Index(df["SNP"]).get_indexer(np.asarray(snps).astype(str))
    out = np.zeros((pos.size, len(cls)))
    out[pos >= 0] = vals[pos[pos >= 0]]
    return out, cls


def _read_inputs(geno, pheno, covar, pheno_cols):
    gi = resolve_genotype_input(geno)
    if gi.format != "bed":
        raise ValueError("Relative-pair moments currently read PLINK 1 BED input.")
    sample_ids = read_fam_sample_ids(gi.sample_path)
    bim = pd.read_csv(gi.variant_path, sep=r"\s+", header=None,
                      names=["CHR", "SNP", "CM", "BP", "A1", "A2"], dtype={"SNP": str, "A1": str, "A2": str})
    bed = open_bed(gi.genotype_path)
    n_all, M = bed.shape
    if len(bim) != M:
        raise ValueError("BIM/BED variant count mismatch.")
    ph = pd.read_csv(pheno, sep=r"\s+", dtype={"FID": str, "IID": str})
    if pheno_cols is None:
        pheno_cols = [c for c in ph.columns if c not in ("FID", "IID")]
    merged = sample_ids.merge(ph[["FID", "IID"] + list(pheno_cols)], on=["FID", "IID"], how="left")
    Yall = merged[pheno_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    cov_ok = np.ones(n_all, dtype=bool)
    if covar is not None:
        cv = pd.read_csv(covar, sep=r"\s+", dtype={"FID": str, "IID": str}, keep_default_na=False)
        cm = sample_ids.merge(cv, on=["FID", "IID"], how="left")
        vals = cm.drop(columns=["FID", "IID"]).apply(pd.to_numeric, errors="coerce")
        cov_ok = ~vals.isna().any(axis=1).to_numpy()
    return sample_ids, bim, bed, n_all, M, list(pheno_cols), Yall, cov_ok


def _covariate_basis(covar, sample_ids, idx):
    if covar is None:
        return np.zeros((idx.size, 0))
    C, _, keep_global = read_cov(cov_filename=covar, fam_filename=None, std=True, cov_impute_method="ignore",
                                 logger=None, sample_idx=idx, ddof=1, sample_ids=sample_ids)
    if not np.array_equal(np.asarray(keep_global), idx):
        raise RuntimeError("Covariate sample selection disagrees with the complete-case set.")
    return np.asarray(C, dtype=np.float64)


def _complete_case_groups(Yall, cov_ok):
    groups: dict[bytes, list[int]] = {}
    for k in range(Yall.shape[1]):
        mask = np.isfinite(Yall[:, k]) & cov_ok
        groups.setdefault(mask.tobytes(), []).append(k)
    return groups


def _write_totals(out, rows, cross_rows, classes, summary):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cols = ["PHENO", "N", "N_COVARIATES", "PAIRS", "REL_M"] + [f"REL_M_{c}" for c in classes]
    pd.DataFrame(rows, columns=cols).to_csv(out + ".rel_m.tsv", sep="\t", index=False, float_format="%.10g")
    if cross_rows is not None:
        pd.DataFrame(cross_rows, columns=["PHENO1", "PHENO2", "N", "PAIRS", "REL_M12"]).to_csv(
            out + ".rel_m12.tsv", sep="\t", index=False, float_format="%.10g")
    with open(out + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)


def compute_relative_pair_totals(
    geno: str,
    pheno: str,
    out: str,
    *,
    pairs: str,
    covar: str | None = None,
    pheno_cols: list[str] | None = None,
    kinship_col: str | None = None,
    kinship_min: float | None = None,
    closure: bool = True,
    step_size: int | None = None,
    cross: bool = False,
    annot: str | None = None,
    log=None,
) -> dict:
    """One REL_M value per phenotype (and per class of `annot`): the V-weighted mean sum_j V_j REL_M_j / sum_j V_j.

    Writes `out`.rel_m.tsv (PHENO N N_COVARIATES PAIRS REL_M [REL_M_<class> ...]), with `cross` also `out`.rel_m12.tsv
    (PHENO1 PHENO2 N PAIRS REL_M12, phenotype pairs with identical complete-case samples), and `out`.json.

    One genotype pass serves every phenotype, and the values equal those computed from the exact per-SNP REL_M
    (`compute_relative_pair_moments`): each complete-case sample keeps its own covariate basis (the same as the per-SNP
    path), mean imputation, projection, N_eff and V weights. The sample's means, sums of squares and covariate
    cross-products come from products of the union's genotype block with indicator and basis matrices (missing calls
    through a sparse correction), and projected genotypes are formed only for people in families.
    """
    from scipy import sparse

    def _log(msg):
        if log is not None:
            log._log(msg)

    sample_ids, bim, bed, n_all, M, pheno_cols, Yall, cov_ok = _read_inputs(geno, pheno, covar, pheno_cols)
    groups = _complete_case_groups(Yall, cov_ok)
    union = np.zeros(n_all, dtype=bool)
    for gkey in groups:
        union |= np.frombuffer(gkey, dtype=bool)
    idx0 = np.flatnonzero(union)
    if idx0.size == 0:
        raise ValueError("No sample has a phenotype and complete covariates.")
    n0 = int(idx0.size)
    rp = read_relative_pairs(pairs, sample_ids, kinship_col=kinship_col, kinship_min=kinship_min, closure=closure)
    _log(f"[relatives] {rp.npairs} unordered related pairs ({'family closure' if closure else 'as listed'}).")
    a0, b0 = pairs_within_sample(rp, idx0, n_all)                 # positions within the union
    rows, inv = np.unique(np.concatenate([a0, b0]), return_inverse=True)
    ar, br = inv[:a0.size], inv[a0.size:]
    if annot is not None:
        A, classes = read_annotation_classes(annot, bim["SNP"].to_numpy())
    else:
        A, classes = np.zeros((M, 0)), []
    K = 1 + len(classes)
    G = []                                                        # one entry per complete-case sample
    cols, off = [], 0
    for gkey, ks in groups.items():
        mask = np.frombuffer(gkey, dtype=bool)
        idx = np.flatnonzero(mask)
        C = _covariate_basis(covar, sample_ids, idx)              # the per-SNP path's basis for this sample
        n, p = int(idx.size), int(C.shape[1])
        m0 = mask[idx0]
        Cfull = np.zeros((n0, p))
        Cfull[m0] = C
        cols.append(Cfull)
        inside = m0[a0] & m0[b0]
        G.append(dict(ks=ks, idx=idx, m0=m0, C=C, n=n, p=p, n_eff=float(n - p - 1), adj=float(n - 1) / float(n - p - 1),
                      off=off, inside=inside, pa=ar[inside], pb=br[inside], Cf=Cfull[rows],
                      W=np.zeros((int(inside.sum()), K)), vsum=np.zeros(K)))
        off += p
    Call = np.asfortranarray(np.hstack(cols)) if off else np.zeros((n0, 0))
    del cols
    Mmask = np.column_stack([g["m0"] for g in G]).astype(np.float64)
    ng = np.array([g["n"] for g in G], dtype=np.float64)
    step = _auto_step(step_size, n0)
    _log(f"[relatives] one genotype pass over {M} SNPs in blocks of {step} for {len(pheno_cols)} phenotype(s) in "
         f"{len(G)} complete-case sample(s): union n={n0}, {a0.size} pairs, {rows.size} individuals in families.")
    nblk = -(-M // step)
    for b_, s in enumerate(range(0, M, step)):
        e = min(M, s + step)
        X = np.asarray(bed.read(index=np.s_[idx0, s:e], dtype="float64"))
        nan = np.isnan(X)
        anynan = bool(nan.any())
        if anynan:
            X[nan] = 0.0
            NAN = sparse.csc_matrix(nan, dtype=np.float64)
            nmiss = np.asarray((NAN.T @ Mmask).T)                 # samples x SNPs: missing calls per sample
            CN = np.asarray((NAN.T @ Call).T) if off else np.zeros((0, e - s))
        S1 = Mmask.T @ X
        S2 = Mmask.T @ (X * X)
        S0 = ng[:, None] - (nmiss if anynan else 0.0)
        CX = Call.T @ X if off else np.zeros((0, e - s))
        Xf = X[rows]
        Of = ~nan[rows] if anynan else None
        A_blk = np.column_stack([np.ones(e - s), A[s:e]])
        for k, g in enumerate(G):
            mu = S1[k] / np.maximum(S0[k], 1.0)
            raw = S2[k] - mu * S1[k]                              # sum over called genotypes of (x - mu)^2
            B = CX[g["off"]:g["off"] + g["p"]]
            if anynan:
                B = B + mu[None, :] * CN[g["off"]:g["off"] + g["p"]]   # sum_i C_i (x_i - mu) o_i
            ss = raw - np.einsum("ij,ij->j", B, B)
            ok = np.isfinite(ss) & (ss > 1e-10)
            scale = np.zeros(ss.shape)
            scale[ok] = np.sqrt(g["n_eff"] / ss[ok])
            with np.errstate(divide="ignore", invalid="ignore"):
                r1 = np.where(ok & (raw > 0.0), np.minimum(1.0, ss / raw), 0.0)
            xh = Xf - mu[None, :]
            if anynan:
                xh *= Of
            if g["p"]:
                xh -= g["Cf"] @ B
            xh *= scale[None, :]
            wcols = (r1 * g["adj"])[:, None] * A_blk
            g["W"] += (xh[g["pa"]] * xh[g["pb"]]) @ wcols
            g["vsum"] += wcols.sum(axis=0)
        if (b_ + 1) % max(1, nblk // 10) == 0:
            _log(f"[relatives] {e} of {M} SNPs")
    rows_out, cross_rows = [], ([] if cross else None)
    summary = {"nsnps": int(M), "method": "one REL_M value per phenotype (and class): V-weighted mean, one genotype pass",
               "union_n": n0, "classes": classes, "npairs_listed": int(rp.npairs), "phenotypes": {}}
    for g in G:
        ks, idx, n, p, n_eff = g["ks"], g["idx"], g["n"], g["p"], g["n_eff"]
        Y, _ = _project_normalize(Yall[idx][:, ks].copy(), g["C"], n_eff)
        yfull = np.zeros((n0, len(ks)))
        yfull[g["m0"]] = Y
        ga, gb = a0[g["inside"]], b0[g["inside"]]
        wp = yfull[ga] * yfull[gb]
        tot = (2.0 / n_eff) * (wp.T @ g["W"])                     # phenotypes x (all, classes): sum_j V_j a_jk REL_M_j
        vals = tot / np.where(g["vsum"] > 0, g["vsum"], np.nan)
        npair = int(g["inside"].sum())
        for t, k in enumerate(ks):
            name = pheno_cols[k]
            rows_out.append([name, n, p, npair, *vals[t].tolist()])
            summary["phenotypes"][name] = {"n": n, "n_covariates": p, "pairs_used": npair, "rel_m": float(vals[t, 0]),
                                           "rel_m_class": dict(zip(classes, map(float, vals[t, 1:])))}
        if cross:
            for t1 in range(len(ks)):
                for t2 in range(t1 + 1, len(ks)):
                    w12 = yfull[ga, t1] * yfull[gb, t2] + yfull[gb, t1] * yfull[ga, t2]
                    v12 = float((w12 @ g["W"][:, 0]) / n_eff / g["vsum"][0])
                    cross_rows.append([pheno_cols[ks[t1]], pheno_cols[ks[t2]], n, npair, v12])
        _log(f"[relatives] phenotypes {[pheno_cols[k] for k in ks][:5]}: n={n}, p={p}, pairs used={npair}.")
    if cross:
        skipped = sum(len(g1["ks"]) * len(g2["ks"]) for i1, g1 in enumerate(G) for g2 in G[i1 + 1:])
        if skipped:
            _log(f"[relatives] REL_M12 not computed for {skipped} phenotype pair(s) with different complete-case samples.")
    _write_totals(out, rows_out, cross_rows, classes, summary)
    return summary


def compute_relative_pair_moments(
    geno: str,
    pheno: str,
    out: str,
    *,
    pairs: str,
    covar: str | None = None,
    pheno_cols: list[str] | None = None,
    kinship_col: str | None = None,
    kinship_min: float | None = None,
    closure: bool = True,
    step_size: int | None = None,
    cross: bool = False,
    annot: str | None = None,
    log=None,
) -> dict:
    """Exact per-SNP REL_M (one column per phenotype) in `out`.tsv.gz (`summit relatives moments --per-snp`).

    Phenotypes sharing the same complete-case sample set (phenotype and covariates
    non-missing) are processed together; each set gets its own covariate basis and genotype pass.
    With `cross`, REL_M12_<a>__<b> is also written for every pair of phenotypes with the
    same complete-case sample (the cross-trait moment z*_a z*_b = (x_j'y_a)(x_j'y_b) / N_eff,
    so REL_M12_j = (1 / N_eff) * sum over ordered pairs of x_ij x_i'j y_a,i y_b,i').
    The per-phenotype values of `compute_relative_pair_totals` are also written (`out`.rel_m.tsv), computed exactly
    from the per-SNP values with each sample's own V weights.
    """
    def _log(msg):
        if log is not None:
            log._log(msg)

    sample_ids, bim, bed, n_all, M, pheno_cols, Yall, cov_ok = _read_inputs(geno, pheno, covar, pheno_cols)
    rp = read_relative_pairs(pairs, sample_ids, kinship_col=kinship_col, kinship_min=kinship_min, closure=closure)
    _log(f"[relatives] {rp.npairs} unordered related pairs ({'family closure' if closure else 'as listed'}).")
    groups = _complete_case_groups(Yall, cov_ok)
    if annot is not None:
        A, classes = read_annotation_classes(annot, bim["SNP"].to_numpy())
    else:
        A, classes = np.zeros((M, 0)), []

    rel_m = np.zeros((M, len(pheno_cols)), dtype=np.float64)
    rel_m12: dict[tuple[int, int], np.ndarray] = {}
    summary = {"nsnps": int(M), "phenotypes": {}, "npairs_listed": int(rp.npairs), "classes": classes}
    rows_out, cross_rows = [], ([] if cross else None)
    if cross:
        skipped = [(pheno_cols[k1], pheno_cols[k2]) for g1, ks1 in groups.items() for g2, ks2 in groups.items()
                   if g1 < g2 for k1 in ks1 for k2 in ks2]
        if skipped:
            _log(f"[relatives] REL_M12 not computed for {len(skipped)} phenotype pair(s) with different "
                 f"complete-case samples, e.g. {skipped[:3]}.")
    for gkey, ks in groups.items():
        mask = np.frombuffer(gkey, dtype=bool)
        idx = np.flatnonzero(mask)
        n = int(idx.size)
        C = _covariate_basis(covar, sample_ids, idx)
        p = int(C.shape[1])
        n_eff = float(n - p - 1)
        adj = float(n - 1) / float(n - p - 1)
        kappa = 1.0 / n_eff
        a, b = pairs_within_sample(rp, idx, n_all)
        rows, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
        ar, br = inv[:a.size], inv[a.size:]
        Y, _ = _project_normalize(Yall[idx][:, ks].copy(), C, n_eff)
        wp = Y[a] * Y[b]                                          # pairs x phenotypes
        xpairs = [(t1, t2) for t1 in range(len(ks)) for t2 in range(t1 + 1, len(ks))] if cross else []
        if xpairs:                                                # pairs x phenotype pairs, both orders of (i, i')
            wx = np.stack([Y[a, t1] * Y[b, t2] + Y[b, t1] * Y[a, t2] for t1, t2 in xpairs], axis=1)
            for t1, t2 in xpairs:
                rel_m12[(ks[t1], ks[t2])] = np.zeros(M, dtype=np.float64)
        v = np.zeros(M)
        step = _auto_step(step_size, n)
        nblk = -(-M // step)
        for b, s in enumerate(range(0, M, step)):
            e = min(M, s + step)
            if (b + 1) % max(1, nblk // 10) == 0:
                _log(f"[relatives] phenotypes {[pheno_cols[k] for k in ks][:3]}: {e} of {M} SNPs")
            X = _center_impute(np.asarray(bed.read(index=np.s_[idx, s:e], dtype="float64")))
            Xr, r1 = _pair_rows_block(X, C, n_eff, rows)
            v[s:e] = r1 * adj
            G = Xr[ar] * Xr[br]                                   # pairs x SNPs
            rel_m[s:e, ks] = kappa * 2.0 * (G.T @ wp)
            if xpairs:
                blk = kappa * (G.T @ wx)
                for c_, (t1, t2) in enumerate(xpairs):
                    rel_m12[(ks[t1], ks[t2])][s:e] = blk[:, c_]
        wcols = v[:, None] * np.column_stack([np.ones(M), A])
        vsum = wcols.sum(axis=0)
        for t, k in enumerate(ks):
            vals = (wcols.T @ rel_m[:, k]) / np.where(vsum > 0, vsum, np.nan)
            rows_out.append([pheno_cols[k], n, p, int(a.size), *vals.tolist()])
            summary["phenotypes"][pheno_cols[k]] = {
                "n": n, "n_covariates": p, "pairs_used": int(a.size),
                "c_rel": float(rel_m[:, k].mean()),
                "sum_rel_m": float(rel_m[:, k].sum()),
                "rel_m": float(vals[0]), "rel_m_class": dict(zip(classes, map(float, vals[1:]))),
            }
        for t1, t2 in xpairs:
            cross_rows.append([pheno_cols[ks[t1]], pheno_cols[ks[t2]], n, int(a.size),
                               float(v @ rel_m12[(ks[t1], ks[t2])] / vsum[0])])
        _log(f"[relatives] phenotypes {[pheno_cols[k] for k in ks]}: n={n}, p={p}, pairs used={a.size}.")

    table = bim[["CHR", "SNP", "BP", "A1", "A2"]].copy()
    for k, name in enumerate(pheno_cols):
        table[f"REL_M_{name}"] = rel_m[:, k]
    if rel_m12:
        summary["cross"] = {}
        for (k1, k2), v12 in rel_m12.items():
            name = f"{pheno_cols[k1]}__{pheno_cols[k2]}"
            table[f"REL_M12_{name}"] = v12
            summary["cross"][name] = {"c_rel12": float(v12.mean()), "sum_rel_m12": float(v12.sum())}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    table.to_csv(out + ".tsv.gz", sep="\t", index=False, float_format="%.10g")
    _write_totals(out, rows_out, cross_rows, classes, summary)
    return summary


def _is_totals_table(path: str, key: str) -> bool:
    return key in pd.read_csv(path, sep="\t", nrows=0).columns


def append_rel_m_to_sumstats(sumstats: str, rel_table: str, pheno: str, out: str, annot: str | None = None) -> int:
    """Copy a sumstats file adding REL_M; returns the number of SNPs without a value.

    `rel_table` is either the per-phenotype table of `summit relatives moments` (`.rel_m.tsv`): REL_M is then the
    phenotype's value for every SNP or, with `annot` (the annotation used in `moments`), its class's value (SNPs outside
    the annotation get the all-SNP value); or the per-SNP table of `--per-snp` (`.tsv.gz`), matched by SNP ID (REL_M is
    invariant to allele orientation)."""
    ss = pd.read_csv(sumstats, sep=r"\s+")
    snp_col = next((c for c in ("SNP", "ID", "snp", "rsid") if c in ss.columns), None)
    if snp_col is None:
        raise ValueError("Sumstats need a SNP/ID column.")
    if _is_totals_table(rel_table, "PHENO"):
        tab = pd.read_csv(rel_table, sep="\t", dtype={"PHENO": str})
        row = tab[tab["PHENO"] == str(pheno)]
        if len(row) != 1:
            raise ValueError(f"{rel_table} has no single row for phenotype {pheno!r}.")
        row = row.iloc[0]
        out_df = ss.copy()
        out_df["REL_M"] = float(row["REL_M"])
        if annot is not None:
            A, classes = read_annotation_classes(annot, ss[snp_col].astype(str).to_numpy())
            missing = [c for c in classes if f"REL_M_{c}" not in tab.columns]
            if missing:
                raise ValueError(f"{rel_table} has no per-class values for {missing[:3]}: run `moments` with this --annot.")
            inside = A.sum(axis=1) > 0
            cls_vals = np.array([float(row[f"REL_M_{c}"]) for c in classes])
            out_df.loc[inside, "REL_M"] = A[inside] @ cls_vals
        out_df.to_csv(out, sep="\t", index=False, float_format="%.10g")
        return 0
    rel = pd.read_csv(rel_table, sep="\t", usecols=["SNP", f"REL_M_{pheno}"]).rename(columns={f"REL_M_{pheno}": "REL_M"})
    if rel["SNP"].duplicated().any():
        raise ValueError(f"{rel_table} has duplicate SNP IDs; REL_M cannot be matched unambiguously.")
    merged = ss.merge(rel, left_on=snp_col, right_on="SNP", how="left", suffixes=("", "_rel"))
    if "SNP_rel" in merged.columns:
        merged = merged.drop(columns=["SNP_rel"])
    if snp_col != "SNP" and "SNP" in merged.columns and "SNP" not in ss.columns:
        merged = merged.drop(columns=["SNP"])
    missing = int(merged["REL_M"].isna().sum())
    merged.to_csv(out, sep="\t", index=False, float_format="%.10g")
    return missing


def extract_rel_m12(rel_table: str, pheno1: str, pheno2: str, out: str) -> float:
    """Write REL_M12 for one phenotype pair (either order) for `summit --rg ... --rel-m12`.

    From the per-pair table of `moments --cross` (`.rel_m12.tsv`) the file holds one value (column REL_M12); from the
    per-SNP table of `--per-snp --cross` it holds SNP and REL_M12. Returns the value (or the per-SNP mean)."""
    if _is_totals_table(rel_table, "PHENO1"):
        tab = pd.read_csv(rel_table, sep="\t", dtype={"PHENO1": str, "PHENO2": str})
        hit = tab[((tab.PHENO1 == pheno1) & (tab.PHENO2 == pheno2)) | ((tab.PHENO1 == pheno2) & (tab.PHENO2 == pheno1))]
        if len(hit) != 1:
            raise ValueError(f"{rel_table} has no REL_M12 for {pheno1} and {pheno2}; run moments with --cross "
                             "(both phenotypes need the same complete-case sample).")
        val = float(hit["REL_M12"].iloc[0])
        pd.DataFrame({"REL_M12": [val]}).to_csv(out, sep="\t", index=False, float_format="%.10g")
        return val
    cols = pd.read_csv(rel_table, sep="\t", nrows=0).columns
    name = next((f"REL_M12_{a}__{b}" for a, b in ((pheno1, pheno2), (pheno2, pheno1))
                 if f"REL_M12_{a}__{b}" in cols), None)
    if name is None:
        raise ValueError(f"{rel_table} has no REL_M12 for {pheno1} and {pheno2}; run moments with --cross.")
    t = pd.read_csv(rel_table, sep="\t", usecols=["SNP", name], dtype={"SNP": str}).rename(columns={name: "REL_M12"})
    t.to_csv(out, sep="\t", index=False, float_format="%.10g")
    return float(t["REL_M12"].mean())
