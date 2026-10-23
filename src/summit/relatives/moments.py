"""Related-pair HE moments for SUMMIT heritability with related individuals.

SUMMIT's h2 response for SNP j is the HE-scale moment m_j = z*_j^2, with
y'K y = (n/M) sum_j m_j for the covariate-projected GRM K = X X'/M. Removing the
related pairs P from the HE estimator (keeping every individual) replaces y'K y by
y'K_u y = y'K y - sum_{(i,i') in P} K_ii' y_i y_i', and the second term decomposes
over SNPs:

    sum_{(i,i') in P} K_ii' y_i y_i' = (1/M) sum_j s_j,
    s_j = sum_{(i,i') in P, ordered} x_ij x_i'j y_i y_i'.

With genotypes and phenotype projected on the GWAS covariates and normalized to
N_eff = n - p - 1, the per-SNP correction on SUMMIT's m_j scale is

    REL_M_j = kappa * s_j,   kappa = (n - 1)(n - p - 2) / ((n - 2) N_eff^2),

the derivative of SUMMIT's beta/SE moment (cov_rank 0 convention) with respect to
(x_j'y)^2 at zero association. SUMMIT subtracts REL_M_j from m_j when a sumstats
file carries a REL_M column, which excludes the related pairs from the numerator
of every SUM-RHE normal equation, for any annotation model or jackknife design.

With LD scores computed in an unrelated sample of the same population no trace
change is needed. If LD scores are computed in the related sample itself, the
related pairs' share of each in-sample LD score is

    LD_REL_j = (M / N_eff^2) sum_{(i,i') in P, ordered} x_ij x_i'j K_ii',

which this module can also write.

Pairs: a file with IID1/IID2 (optionally FID1/FID2 and a kinship column with a
threshold). By default pairs are closed into families (connected components) and
every within-family pair is excluded.
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
    step_size: int = 5000,
    ld_rel: bool = False,
    log=None,
) -> dict:
    """Write per-SNP REL_M (one column per phenotype) and optional LD_REL to `out`.tsv.gz.

    Phenotypes sharing the same complete-case sample set (phenotype and covariates
    non-missing) are processed together; each set gets its own covariate basis.
    """
    def _log(msg):
        if log is not None:
            log._log(msg)

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

    rp = read_relative_pairs(pairs, sample_ids, kinship_col=kinship_col, kinship_min=kinship_min, closure=closure)
    _log(f"[relatives] {rp.npairs} unordered related pairs ({'family closure' if closure else 'as listed'}).")

    # group phenotypes by complete-case sample set
    groups: dict[bytes, list[int]] = {}
    for k in range(len(pheno_cols)):
        mask = np.isfinite(Yall[:, k]) & cov_ok
        groups.setdefault(mask.tobytes(), []).append(k)

    rel_m = np.zeros((M, len(pheno_cols)), dtype=np.float64)
    ld_rel_out = np.zeros(M, dtype=np.float64) if ld_rel else None
    summary = {"nsnps": int(M), "phenotypes": {}, "npairs_listed": int(rp.npairs)}
    for gkey, ks in groups.items():
        mask = np.frombuffer(gkey, dtype=bool)
        idx = np.flatnonzero(mask)
        n = int(idx.size)
        if covar is not None:
            C, _, keep_global = read_cov(cov_filename=covar, fam_filename=None, std=True, cov_impute_method="ignore",
                                        logger=None, sample_idx=idx, ddof=1, sample_ids=sample_ids)
            if not np.array_equal(np.asarray(keep_global), idx):
                raise RuntimeError("Covariate sample selection disagrees with the complete-case set.")
            C = np.asarray(C, dtype=np.float64)
        else:
            C = np.zeros((n, 0))
        p = int(C.shape[1])
        n_eff = float(n - p - 1)
        kappa = (n - 1.0) * (n - p - 2.0) / ((n - 2.0) * n_eff * n_eff)
        pos = -np.ones(n_all, dtype=np.int64)
        pos[idx] = np.arange(n)
        a, b = pos[rp.i], pos[rp.j]
        use = (a >= 0) & (b >= 0)
        a, b = a[use], b[use]
        Y, _ = _project_normalize(Yall[idx][:, ks].copy(), C, n_eff)
        wp = Y[a] * Y[b]                                          # pairs x phenotypes
        Kp = np.zeros(a.size) if ld_rel else None
        if ld_rel:                                                 # first pass: K on pairs
            for s in range(0, M, step_size):
                e = min(M, s + step_size)
                X, _ = _project_normalize(bed.read(index=np.s_[idx, s:e], dtype="float64"), C, n_eff)
                Kp += np.einsum("pm,pm->p", X[a], X[b])
            Kp /= M
        for s in range(0, M, step_size):
            e = min(M, s + step_size)
            X, ok = _project_normalize(bed.read(index=np.s_[idx, s:e], dtype="float64"), C, n_eff)
            G = X[a] * X[b]                                       # pairs x SNPs
            rel_m[s:e, ks] = kappa * 2.0 * (G.T @ wp)
            if ld_rel:
                ld_rel_out[s:e] = (M / n_eff ** 2) * 2.0 * (G.T @ Kp)
        for t, k in enumerate(ks):
            summary["phenotypes"][pheno_cols[k]] = {
                "n": n, "n_covariates": p, "pairs_used": int(a.size),
                "c_rel": float(rel_m[:, k].mean()),
                "sum_rel_m": float(rel_m[:, k].sum()),
            }
        _log(f"[relatives] phenotypes {[pheno_cols[k] for k in ks]}: n={n}, p={p}, pairs used={a.size}.")

    table = bim[["CHR", "SNP", "BP", "A1", "A2"]].copy()
    for k, name in enumerate(pheno_cols):
        table[f"REL_M_{name}"] = rel_m[:, k]
    if ld_rel:
        table["LD_REL"] = ld_rel_out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    table.to_csv(out + ".tsv.gz", sep="\t", index=False, float_format="%.10g")
    with open(out + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def append_rel_m_to_sumstats(sumstats: str, rel_table: str, pheno: str, out: str) -> int:
    """Copy a sumstats file adding REL_M (matched by SNP ID; REL_M is invariant to allele orientation)."""
    ss = pd.read_csv(sumstats, sep=r"\s+")
    snp_col = next((c for c in ("SNP", "ID", "snp", "rsid") if c in ss.columns), None)
    if snp_col is None:
        raise ValueError("Sumstats need a SNP/ID column.")
    rel = pd.read_csv(rel_table, sep="\t", usecols=["SNP", f"REL_M_{pheno}"]).rename(columns={f"REL_M_{pheno}": "REL_M"})
    merged = ss.merge(rel, left_on=snp_col, right_on="SNP", how="left", suffixes=("", "_rel"))
    if "SNP_rel" in merged.columns:
        merged = merged.drop(columns=["SNP_rel"])
    if snp_col != "SNP" and "SNP" in merged.columns and "SNP" not in ss.columns:
        merged = merged.drop(columns=["SNP"])
    missing = int(merged["REL_M"].isna().sum())
    merged.to_csv(out, sep="\t", index=False, float_format="%.10g")
    return missing
