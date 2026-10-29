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

Inbreeding (optional, `--inbreeding`). The first estimating equation also contains each individual paired with
themselves, weighted by the diagonal K_ii of the GRM, which is larger for inbred individuals. When the residual variance
grows with inbreeding (untagged causal variants), these diagonal terms bias h2 in related and unrelated samples alike.
With `inbreeding=True` the first equation uses only pairs of different, unrelated individuals. In the units of SUMMIT's
kernels (K~_k = sum_j a_jk V_j x_j x_j' / sum_j a_jk V_j, trace N_eff) its exact form is

    sum_l [tr(K~u_k K~u_l) - D_kl] sigma_l + lambda_k sigma_e = y'K~u_k y - sum_i K~_k,ii y_i^2,

with D_kl = sum_i K~_k,ii K~_l,ii and lambda_k = sum_i K~_k,ii H_ii + sum_{(i,i') in P} K~_k,ii' H_ii' (H the projection on
the covariates, intercept included); the second equation is unchanged. Per phenotype and class this needs
REL_S_k = sum_i (K~_k,ii - 1) y_i^2 / N_eff, and per sample REL_LAMBDA_k = lambda_k and REL_DIAG_kl = D_kl - N_eff,
which the same genotype pass computes from the projected genotypes of everyone in the sample. `append` writes them as
constant columns (REL_S, REL_LAMBDA, REL_DIAG and, with --annot, REL_DIAG_<class>), and `summit --h2` uses them when
present.
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


def _inbreeding_columns(classes):
    """Column names of the --inbreeding values in PREFIX.rel_m.tsv: all SNPs, then per class (REL_DIAG_<c> is the
    class-mass-weighted row sum used by single-component fits of class-wise sumstats), then per pair of classes."""
    cols = ["REL_S", "REL_LAMBDA", "REL_DIAG"]
    for c in classes:
        cols += [f"REL_S_{c}", f"REL_LAMBDA_{c}", f"REL_DIAG_{c}"]
    cols += [f"REL_DIAG_{c}__{d}" for c in classes for d in classes]
    return cols


def _write_totals(out, rows, cross_rows, classes, summary, inbreeding=False):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    cols = ["PHENO", "N", "N_COVARIATES", "PAIRS", "REL_M"] + [f"REL_M_{c}" for c in classes]
    if inbreeding:
        cols += _inbreeding_columns(classes)
    pd.DataFrame(rows, columns=cols).to_csv(out + ".rel_m.tsv", sep="\t", index=False, float_format="%.12g")
    if cross_rows is not None:
        ccols = ["PHENO1", "PHENO2", "N", "PAIRS", "REL_M12"] + [f"REL_M12_{c}" for c in classes]
        if inbreeding:
            ccols += ["REL_S12"] + [f"REL_S12_{c}" for c in classes]
        pd.DataFrame(cross_rows, columns=ccols).to_csv(out + ".rel_m12.tsv", sep="\t", index=False, float_format="%.12g")
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
    inbreeding: bool = False,
    log=None,
) -> dict:
    """One REL_M value per phenotype (and per class of `annot`): the V-weighted mean sum_j V_j REL_M_j / sum_j V_j.

    With `inbreeding`, also the values that leave the diagonal terms out of the first estimating equation (REL_S per
    phenotype, REL_LAMBDA and REL_DIAG per sample; module docstring). These need the projected genotypes of everyone in
    each complete-case sample, so the pass costs about one extra projection of the genotype matrix per sample.

    Writes `out`.rel_m.tsv (PHENO N N_COVARIATES PAIRS REL_M [REL_M_<class> ...]), with `cross` also `out`.rel_m12.tsv
    (PHENO1 PHENO2 N PAIRS REL_M12 [REL_M12_<class> ...] for every phenotype pair; N is the number of individuals in both
    samples and PAIRS the number of related pairs with a member in each), and `out`.json.

    Phenotypes a and b measured in different complete-case samples have the cross-trait moment
    z*_a z*_b = (x_a'y_a)(x_b'y_b) / sqrt(N_eff,a N_eff,b), each factor projected and normalized in its own sample, so
    REL_M12_j = sum over ordered related pairs (i in sample a, i' in sample b) of x_a,ij y_a,i x_b,i'j y_b,i' divided by
    sqrt(N_eff,a N_eff,b), and the per-pair values are its means weighted by sqrt(V_a,j V_b,j) (V_j when the samples are
    the same). The same pass accumulates these sums for every pair of phenotypes as one matrix product per class over the
    related pairs and SNPs; REL_S12 (`inbreeding`) is left empty (NaN) for such pairs.

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
        Y, _ = _project_normalize(Yall[idx][:, ks].copy(), C, float(n - p - 1))
        yfull = np.zeros((n0, len(ks)))
        yfull[m0] = Y
        G.append(dict(ks=ks, idx=idx, m0=m0, C=C, n=n, p=p, n_eff=float(n - p - 1), adj=float(n - 1) / float(n - p - 1),
                      off=off, inside=inside, pa=ar[inside], pb=br[inside], Cf=Cfull[rows], Y=Y, yfull=yfull,
                      W=np.zeros((int(inside.sum()), K)), vsum=np.zeros(K),
                      rows=(np.flatnonzero(m0) if inbreeding else None),
                      D=(np.zeros((n, K)) if inbreeding else None)))
        off += p
    Call = np.asfortranarray(np.hstack(cols)) if off else np.zeros((n0, 0))
    del cols
    Mmask = np.column_stack([g["m0"] for g in G]).astype(np.float64)
    ng = np.array([g["n"] for g in G], dtype=np.float64)
    step = _auto_step(step_size, n0)
    xs = None
    if cross and len(G) > 1:                                      # phenotype pairs in different samples
        xs = _CrossSamplePairs(G, A, a0, b0, ar, br, len(classes))
        if step_size is None:
            step = min(step, xs.max_step)
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
            if xs is not None:
                xs.store(k, xh, r1 * g["adj"])
            if inbreeding:                                        # diagonal sum_j V_j a_jk x_ij^2 for everyone in the sample
                xg = X[g["rows"]] - mu[None, :]
                if anynan:
                    xg *= ~nan[g["rows"]]
                if g["p"]:
                    xg -= g["C"] @ B
                xg *= scale[None, :]
                g["D"] += (xg * xg) @ wcols
        if xs is not None:
            xs.accumulate(s, e)
        if (b_ + 1) % max(1, nblk // 10) == 0:
            _log(f"[relatives] {e} of {M} SNPs")
    rows_out, cross_rows = [], ([] if cross else None)
    summary = {"nsnps": int(M), "method": "one REL_M value per phenotype (and class): V-weighted mean, one genotype pass",
               "union_n": n0, "classes": classes, "npairs_listed": int(rp.npairs), "inbreeding": bool(inbreeding),
               "phenotypes": {}}
    for g in G:
        ks, idx, n, p, n_eff, Y, yfull = g["ks"], g["idx"], g["n"], g["p"], g["n_eff"], g["Y"], g["yfull"]
        ga, gb = a0[g["inside"]], b0[g["inside"]]
        wp = yfull[ga] * yfull[gb]
        tot = (2.0 / n_eff) * (wp.T @ g["W"])                     # phenotypes x (all, classes): sum_j V_j a_jk REL_M_j
        vals = tot / np.where(g["vsum"] > 0, g["vsum"], np.nan)
        npair = int(g["inside"].sum())
        inb = None
        if inbreeding:
            gpos = np.cumsum(g["m0"]) - 1                         # union position -> position in the sample
            inb = _diagonal_terms(g["D"], g["W"], g["vsum"], g["C"], n, n_eff, gpos[ga], gpos[gb], Y, len(classes))
        for t, k in enumerate(ks):
            name = pheno_cols[k]
            extra = [] if inb is None else inb["row"](t)
            rows_out.append([name, n, p, npair, *vals[t].tolist(), *extra])
            summary["phenotypes"][name] = {"n": n, "n_covariates": p, "pairs_used": npair, "rel_m": float(vals[t, 0]),
                                           "rel_m_class": dict(zip(classes, map(float, vals[t, 1:])))}
            if inb is not None:
                summary["phenotypes"][name].update({"rel_s": float(inb["c_s"][0, t]), "rel_lambda": float(inb["lam"][0]),
                                                    "rel_diag": float(inb["s"][0, 0])})
        if inb is not None:
            _log_diagonal_terms(_log, n, inb)
        if cross:
            for t1 in range(len(ks)):
                for t2 in range(t1 + 1, len(ks)):
                    w12 = yfull[ga, t1] * yfull[gb, t2] + yfull[gb, t1] * yfull[ga, t2]
                    v12 = (w12 @ g["W"]) / n_eff / np.where(g["vsum"] > 0, g["vsum"], np.nan)
                    cross_rows.append([pheno_cols[ks[t1]], pheno_cols[ks[t2]], n, npair, *v12.tolist(),
                                       *(inb["cross"](t1, t2) if inb is not None else [])])
        _log(f"[relatives] phenotypes {[pheno_cols[k] for k in ks][:5]}: n={n}, p={p}, pairs used={npair}.")
    if cross:
        nsame = len(cross_rows)
        if xs is not None:
            nan_s12 = [np.nan] * K if inbreeding else []
            for k1, k2, n12, npair, v12 in xs.values():
                cross_rows.append([pheno_cols[k1], pheno_cols[k2], n12, npair, *v12.tolist(), *nan_s12])
        order = {name: k for k, name in enumerate(pheno_cols)}
        cross_rows.sort(key=lambda r: (order[r[0]], order[r[1]]))
        summary["cross_pairs"] = {"same_sample": nsame, "different_samples": len(cross_rows) - nsame}
        _log(f"[relatives] REL_M12 for {len(cross_rows)} phenotype pair(s), {len(cross_rows) - nsame} of them with "
             "different complete-case samples" + (" (REL_S12 is not computed for those)." if inbreeding else "."))
    _write_totals(out, rows_out, cross_rows, classes, summary, inbreeding=inbreeding)
    return summary


class _CrossSamplePairs:
    """REL_M12 sums for phenotypes in different complete-case samples, accumulated in the totals pass.

    For SNP j of class c (the annotation's class, or one extra class for SNPs outside it) and phenotypes t1, t2 with
    samples g, h: num_c[t1, t2] = sum_j sum over unordered related pairs (i, i') of u_t1,ij u_t2,i'j, with
    u_t,ij = y_t,i x_g(t),ij sqrt(V_g(t),j) (genotypes projected and normalized in the phenotype's own sample, y zero
    outside it), and den_c[g, h] = sum_j sqrt(V_gj V_hj). Each block's pair rows are kept per sample (`store`); `accumulate`
    then adds one matrix product of (phenotypes x pairs*SNPs) arrays per class."""

    def __init__(self, G, A, a0, b0, ar, br, nclass):
        self.G, self.ar, self.br, self.nclass = G, ar, br, nclass
        self.cls = (np.where(A.sum(axis=1) > 0, A.argmax(axis=1), nclass) if nclass
                    else np.zeros(A.shape[0], dtype=np.int64))
        self.slot = np.concatenate([g["ks"] for g in G])          # phenotype of each row of ya/yb, grouped by sample
        self.gof = np.repeat(np.arange(len(G)), [len(g["ks"]) for g in G])
        self.ya = np.vstack([g["yfull"][a0].T for g in G])        # phenotypes x pairs: y at the first member
        self.yb = np.vstack([g["yfull"][b0].T for g in G])
        T, P = self.ya.shape
        self.num = np.zeros((nclass + 1, T, T))
        self.den = np.zeros((nclass + 1, len(G), len(G)))
        self.width = int(max(1, (1 << 23) // max(1, T * P)))     # SNPs per product (64 MiB per factor)
        self.max_step = int(max(16, (1 << 26) // max(1, 2 * len(G) * P)))   # block size for the stored pair rows
        self.xa = self.xb = self.sv = None
        self.m0 = [g["m0"] for g in G]
        self.a0, self.b0 = a0, b0

    def store(self, k, xh, v):
        """Projected genotypes xh (family rows x block SNPs) and V weights of sample k for the current block."""
        if self.xa is None or self.xa.shape[2] != xh.shape[1]:
            shape = (len(self.G), self.ar.size, xh.shape[1])
            self.xa, self.xb, self.sv = np.empty(shape), np.empty(shape), np.empty((len(self.G), xh.shape[1]))
        sv = np.sqrt(v)
        np.multiply(xh[self.ar], sv[None, :], out=self.xa[k])
        np.multiply(xh[self.br], sv[None, :], out=self.xb[k])
        self.sv[k] = sv

    def accumulate(self, s, e):
        cls = self.cls[s:e]
        perm = np.argsort(cls, kind="stable")
        cls = cls[perm]
        T, P = self.ya.shape
        cuts = np.flatnonzero(np.diff(cls)) + 1
        buf_a, buf_b = np.empty(T * P * self.width), np.empty(T * P * self.width)
        for r0, r1 in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [cls.size]])):
            c = int(cls[r0])
            for q0 in range(r0, r1, self.width):
                cj = perm[q0:min(r1, q0 + self.width)]
                w = cj.size
                sv = self.sv[:, cj]
                self.den[c] += sv @ sv.T
                if P == 0:
                    continue
                za, zb = buf_a[:T * P * w].reshape(T, P, w), buf_b[:T * P * w].reshape(T, P, w)
                o = 0
                for k, g in enumerate(self.G):
                    t = len(g["ks"])
                    np.multiply(self.ya[o:o + t, :, None], self.xa[k][None, :, cj], out=za[o:o + t])
                    np.multiply(self.yb[o:o + t, :, None], self.xb[k][None, :, cj], out=zb[o:o + t])
                    o += t
                self.num[c] += za.reshape(T, -1) @ zb.reshape(T, -1).T

    def values(self):
        """(phenotype 1, phenotype 2, N in both samples, related pairs with a member in each, REL_M12 (all, classes))
        for every pair of phenotypes in different samples, phenotype 1 first in the phenotype order."""
        num = self.num + self.num.transpose(0, 2, 1)              # both orders of each related pair
        out = []
        for s1 in range(self.slot.size):
            for s2 in range(self.slot.size):
                k1, k2 = int(self.slot[s1]), int(self.slot[s2])
                g, h = int(self.gof[s1]), int(self.gof[s2])
                if g == h or k1 > k2:
                    continue
                norm = np.sqrt(self.G[g]["n_eff"] * self.G[h]["n_eff"])
                nc, dc = num[:, s1, s2] / norm, self.den[:, g, h]
                vals = np.concatenate([[nc.sum() / dc.sum() if dc.sum() > 0 else np.nan],
                                       np.where(dc[:self.nclass] > 0, nc[:self.nclass], np.nan)
                                       / np.where(dc[:self.nclass] > 0, dc[:self.nclass], 1.0)])
                mg, mh = self.m0[g], self.m0[h]
                npair = int(((mg[self.a0] & mh[self.b0]) | (mg[self.b0] & mh[self.a0])).sum())
                out.append((k1, k2, int((mg & mh).sum()), npair, vals))
        return out


def _diagonal_terms(D, W, vsum, C, n, n_eff, pa, pb, Y, nclass):
    """Values that leave the diagonal terms out of the first estimating equation, for one complete-case sample.

    D (n x (1 + C)) holds sum_j a_jk V_j x_ij^2 and W (pairs x (1 + C)) sum_j a_jk V_j x_ij x_i'j for the unordered
    related pairs (pa, pb) (positions in the sample); vsum = sum_j a_jk V_j (column 0: all SNPs). In the units of
    SUMMIT's kernels K~_k = sum_j a_jk V_j x_j x_j' / vsum_k: REL_S_k = sum_i (K~_k,ii - 1) y_i^2 / N_eff per phenotype,
    REL_LAMBDA_k = sum_i K~_k,ii H_ii + sum_{ordered pairs} K~_k,ii' H_ii' and REL_DIAG_kl = sum_i K~_k,ii K~_l,ii - N_eff,
    with H = 11'/n + CC' the projection on the covariates (C orthonormal and centered). Y holds the sample's projected
    phenotypes normalized to sum of squares N_eff. The cross-trait value of two phenotypes is
    REL_S12_k = sum_i (K~_k,ii - 1) y_1i y_2i / N_eff."""
    vs = np.where(vsum > 0, vsum, np.nan)
    Kd = D / vs[None, :]                                          # n x (all, classes): K~_k,ii
    p = C.shape[1]
    Hii = np.full(n, 1.0 / n) + ((C * C).sum(axis=1) if p else 0.0)
    Hab = np.full(pa.size, 1.0 / n) + ((C[pa] * C[pb]).sum(axis=1) if p else 0.0)
    eta_s = Kd.T @ Hii
    eta_r = 2.0 * ((W / vs[None, :]).T @ Hab)                     # ordered pairs: twice the unordered sum
    lam = eta_s + eta_r
    s = Kd.T @ Kd - n_eff                                         # (1 + C) x (1 + C)
    Kc = Kd - 1.0
    c_s = (Kc.T @ (Y * Y)) / n_eff                                # (1 + C) x phenotypes
    w = vsum[1:] / vsum[1:].sum() if nclass else np.zeros(0)
    rowsum = s[1:, 1:] @ w if nclass else np.zeros(0)            # class-mass-weighted row sums (single-component fits)

    def row(t):
        out = [float(c_s[0, t]), float(lam[0]), float(s[0, 0])]
        for c in range(nclass):
            out += [float(c_s[1 + c, t]), float(lam[1 + c]), float(rowsum[c])]
        out += [float(s[1 + c, 1 + d]) for c in range(nclass) for d in range(nclass)]
        return out

    def cross(t1, t2):
        return ((Kc.T @ (Y[:, t1] * Y[:, t2])) / n_eff).tolist()

    return {"c_s": c_s, "lam": lam, "s": s, "eta_s": eta_s, "eta_r": eta_r, "row": row, "cross": cross}


def _log_diagonal_terms(_log, n, inb):
    _log(f"[relatives] diagonal terms (--inbreeding), n={n}: lambda={inb['lam'][0]:.6g} "
         f"(sum_i K_ii H_ii={inb['eta_s'][0]:.6g}, related pairs {inb['eta_r'][0]:.6g}), "
         f"REL_DIAG={inb['s'][0, 0]:.6g}, REL_S of the first phenotype={inb['c_s'][0, 0]:.6g}.")


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
    inbreeding: bool = False,
    log=None,
) -> dict:
    """Exact per-SNP REL_M (one column per phenotype) in `out`.tsv.gz (`summit relatives moments --per-snp`).

    Phenotypes sharing the same complete-case sample set (phenotype and covariates
    non-missing) are processed together; each set gets its own covariate basis and genotype pass.
    With `cross`, REL_M12_<a>__<b> is also written for every pair of phenotypes with the
    same complete-case sample (the cross-trait moment z*_a z*_b = (x_j'y_a)(x_j'y_b) / N_eff,
    so REL_M12_j = (1 / N_eff) * sum over ordered pairs of x_ij x_i'j y_a,i y_b,i').
    The per-phenotype values of `compute_relative_pair_totals` are also written (`out`.rel_m.tsv), computed exactly
    from the per-SNP values with each sample's own V weights, and with `inbreeding` the same diagonal-term values,
    from the projected genotypes of every individual in the same pass.
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
    summary = {"nsnps": int(M), "phenotypes": {}, "npairs_listed": int(rp.npairs), "classes": classes,
               "inbreeding": bool(inbreeding)}
    Aw = np.column_stack([np.ones(M), A])
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
        if inbreeding:
            D = np.zeros((n, Aw.shape[1]))
            W = np.zeros((a.size, Aw.shape[1]))
        step = _auto_step(step_size, n)
        nblk = -(-M // step)
        for ib, s in enumerate(range(0, M, step)):
            e = min(M, s + step)
            if (ib + 1) % max(1, nblk // 10) == 0:
                _log(f"[relatives] phenotypes {[pheno_cols[k] for k in ks][:3]}: {e} of {M} SNPs")
            X = _center_impute(np.asarray(bed.read(index=np.s_[idx, s:e], dtype="float64")))
            Xr, r1 = _pair_rows_block(X, C, n_eff, rows)
            v[s:e] = r1 * adj
            G = Xr[ar] * Xr[br]                                   # pairs x SNPs
            rel_m[s:e, ks] = kappa * 2.0 * (G.T @ wp)
            if inbreeding:                                        # projected genotypes of everyone in the sample
                wblk = (r1 * adj)[:, None] * Aw[s:e]
                W += G @ wblk
                Xp = X - C @ (C.T @ X) if p else X.copy()
                ssq = np.einsum("ij,ij->j", Xp, Xp)
                ok = ssq > 1e-10
                Xp[:, ok] *= np.sqrt(n_eff / ssq[ok])[None, :]
                Xp[:, ~ok] = 0.0
                D += (Xp * Xp) @ wblk
            if xpairs:
                blk = kappa * (G.T @ wx)
                for c_, (t1, t2) in enumerate(xpairs):
                    rel_m12[(ks[t1], ks[t2])][s:e] = blk[:, c_]
        wcols = v[:, None] * Aw
        vsum = wcols.sum(axis=0)
        inb = _diagonal_terms(D, W, vsum, C, n, n_eff, a, b, Y, len(classes)) if inbreeding else None
        if inb is not None:
            _log_diagonal_terms(_log, n, inb)
        for t, k in enumerate(ks):
            vals = (wcols.T @ rel_m[:, k]) / np.where(vsum > 0, vsum, np.nan)
            rows_out.append([pheno_cols[k], n, p, int(a.size), *vals.tolist(), *(inb["row"](t) if inb is not None else [])])
            summary["phenotypes"][pheno_cols[k]] = {
                "n": n, "n_covariates": p, "pairs_used": int(a.size),
                "c_rel": float(rel_m[:, k].mean()),
                "sum_rel_m": float(rel_m[:, k].sum()),
                "rel_m": float(vals[0]), "rel_m_class": dict(zip(classes, map(float, vals[1:]))),
            }
        for t1, t2 in xpairs:
            v12 = (wcols.T @ rel_m12[(ks[t1], ks[t2])]) / np.where(vsum > 0, vsum, np.nan)
            cross_rows.append([pheno_cols[ks[t1]], pheno_cols[ks[t2]], n, int(a.size), *v12.tolist(),
                               *(inb["cross"](t1, t2) if inb is not None else [])])
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
    _write_totals(out, rows_out, cross_rows, classes, summary, inbreeding=inbreeding)
    return summary


def _is_totals_table(path: str, key: str) -> bool:
    return key in pd.read_csv(path, sep="\t", nrows=0).columns


def _sibling_totals(per_snp_table: str, suffix: str) -> str | None:
    """PREFIX.<suffix> next to a per-SNP table PREFIX.tsv.gz (both written by one `moments --per-snp` run), if it exists."""
    base = str(per_snp_table)
    for ext in (".tsv.gz", ".tsv"):
        if base.endswith(ext):
            path = base[: -len(ext)] + suffix
            return path if os.path.exists(path) else None
    return None


def _class_column(row, name: str, classes, A, inside) -> np.ndarray:
    """Per-SNP values of a per-class column: the class's value inside the annotation, the all-SNP value outside."""
    out = np.full(A.shape[0], float(row[name]))
    if classes:
        out[inside] = A[inside] @ np.array([float(row[f"{name}_{c}"]) for c in classes])
    return out


def _annot_or_none(annot, snps, cols, need):
    """Class indicators for `snps`, checking that the per-class columns `need(c)` exist; ([], zeros, all-False) without annot."""
    if annot is None:
        return [], np.zeros((len(snps), 0)), np.zeros(len(snps), dtype=bool)
    A, classes = read_annotation_classes(annot, np.asarray(snps).astype(str))
    missing = [x for c in classes for x in need(c) if x not in cols]
    if missing:
        raise ValueError(f"no per-class values for {missing[:3]}: run `summit relatives moments` with this --annot.")
    return classes, A, A.sum(axis=1) > 0


def _add_diagonal_columns(df: pd.DataFrame, row, cols, snps, annot) -> None:
    """Add the `moments --inbreeding` values (REL_S, REL_LAMBDA, REL_DIAG; with annot their class values and one
    REL_DIAG_<class> column per class, row d of the class-by-class REL_DIAG matrix) to a sumstats frame."""
    classes, A, inside = _annot_or_none(annot, snps, cols, lambda c: (f"REL_S_{c}", f"REL_DIAG_{c}__{c}"))
    for col in ("REL_S", "REL_LAMBDA", "REL_DIAG"):
        df[col] = _class_column(row, col, classes, A, inside)
    for d in classes:
        vals = np.full(len(snps), float(row["REL_DIAG"]))
        vals[inside] = A[inside] @ np.array([float(row[f"REL_DIAG_{c}__{d}"]) for c in classes])
        df[f"REL_DIAG_{d}"] = vals


def append_rel_m_to_sumstats(sumstats: str, rel_table: str, pheno: str, out: str, annot: str | None = None) -> int:
    """Copy a sumstats file adding REL_M; returns the number of SNPs without a value.

    `rel_table` is either the per-phenotype table of `summit relatives moments` (`.rel_m.tsv`): REL_M is then the
    phenotype's value for every SNP or, with `annot` (the annotation used in `moments`), its class's value (SNPs outside
    the annotation get the all-SNP value); or the per-SNP table of `--per-snp` (`.tsv.gz`), matched by SNP ID (REL_M is
    invariant to allele orientation). If `moments` ran with `--inbreeding`, the diagonal-term values are written too
    (REL_S, REL_LAMBDA and REL_DIAG, the class's values with `annot`, and with `annot` one REL_DIAG_<class> column per
    class); for a per-SNP table they come from PREFIX.rel_m.tsv of the same run."""
    ss = pd.read_csv(sumstats, sep=r"\s+")
    snp_col = next((c for c in ("SNP", "ID", "snp", "rsid") if c in ss.columns), None)
    if snp_col is None:
        raise ValueError("Sumstats need a SNP/ID column.")
    per_snp = not _is_totals_table(rel_table, "PHENO")
    totals = _sibling_totals(rel_table, ".rel_m.tsv") if per_snp else rel_table
    row = tcols = None
    if totals is not None:
        tab = pd.read_csv(totals, sep="\t", dtype={"PHENO": str})
        hit = tab[tab["PHENO"] == str(pheno)]
        if len(hit) != 1:
            raise ValueError(f"{totals} has no single row for phenotype {pheno!r}.")
        row, tcols = hit.iloc[0], tab.columns
    if per_snp:
        rel = pd.read_csv(rel_table, sep="\t", usecols=["SNP", f"REL_M_{pheno}"]).rename(columns={f"REL_M_{pheno}": "REL_M"})
        if rel["SNP"].duplicated().any():
            raise ValueError(f"{rel_table} has duplicate SNP IDs; REL_M cannot be matched unambiguously.")
        out_df = ss.merge(rel, left_on=snp_col, right_on="SNP", how="left", suffixes=("", "_rel"))
        if "SNP_rel" in out_df.columns:
            out_df = out_df.drop(columns=["SNP_rel"])
        if snp_col != "SNP" and "SNP" in out_df.columns and "SNP" not in ss.columns:
            out_df = out_df.drop(columns=["SNP"])
        missing = int(out_df["REL_M"].isna().sum())
    else:
        out_df = ss.copy()
        classes, A, inside = _annot_or_none(annot, ss[snp_col], tcols, lambda c: (f"REL_M_{c}",))
        out_df["REL_M"] = _class_column(row, "REL_M", classes, A, inside)
        missing = 0
    if row is not None and "REL_S" in tcols:                     # written by `moments --inbreeding`
        _add_diagonal_columns(out_df, row, tcols, out_df[snp_col], annot)
    out_df.to_csv(out, sep="\t", index=False, float_format="%.12g")
    return missing


def extract_rel_m12(rel_table: str, pheno1: str, pheno2: str, out: str, annot: str | None = None) -> float:
    """Write REL_M12 for one phenotype pair (either order) for `summit --rg ... --rel-m12`.

    From the per-pair table of `moments --cross` (`.rel_m12.tsv`) the file holds one value (column REL_M12) or, with
    `annot`, SNP and the class's REL_M12 for every SNP of the annotation; from the per-SNP table of `--per-snp --cross`
    it holds SNP and REL_M12. If `moments` ran with `--inbreeding`, REL_S12 (the cross-trait diagonal-term value) is
    written next to REL_M12, in the same layout, unless it is empty (phenotypes in different complete-case samples).
    Returns the REL_M12 value (or its mean over SNPs)."""
    per_snp = not _is_totals_table(rel_table, "PHENO1")
    totals = _sibling_totals(rel_table, ".rel_m12.tsv") if per_snp else rel_table
    row = tcols = None
    if totals is not None:
        tab = pd.read_csv(totals, sep="\t", dtype={"PHENO1": str, "PHENO2": str})
        hit = tab[((tab.PHENO1 == pheno1) & (tab.PHENO2 == pheno2)) | ((tab.PHENO1 == pheno2) & (tab.PHENO2 == pheno1))]
        if len(hit) != 1:
            raise ValueError(f"{totals} has no REL_M12 for {pheno1} and {pheno2}; run moments with --cross (with "
                             "--per-snp, both phenotypes need the same complete-case sample).")
        row, tcols = hit.iloc[0], tab.columns
    inb = row is not None and "REL_S12" in tcols and bool(np.isfinite(float(row["REL_S12"])))
    if per_snp:
        cols = pd.read_csv(rel_table, sep="\t", nrows=0).columns
        name = next((f"REL_M12_{a}__{b}" for a, b in ((pheno1, pheno2), (pheno2, pheno1))
                     if f"REL_M12_{a}__{b}" in cols), None)
        if name is None:
            raise ValueError(f"{rel_table} has no REL_M12 for {pheno1} and {pheno2}; run moments with --cross.")
        t = pd.read_csv(rel_table, sep="\t", usecols=["SNP", name], dtype={"SNP": str}).rename(columns={name: "REL_M12"})
        if inb:
            classes, A, inside = _annot_or_none(annot, t["SNP"], tcols, lambda c: (f"REL_S12_{c}",))
            t["REL_S12"] = _class_column(row, "REL_S12", classes, A, inside)
        t.to_csv(out, sep="\t", index=False, float_format="%.12g")
        return float(t["REL_M12"].mean())
    if annot is None:
        t = pd.DataFrame({"REL_M12": [float(row["REL_M12"])]})
        if inb:
            t["REL_S12"] = float(row["REL_S12"])
    else:
        snps = pd.read_csv(annot, sep=r"\s+", usecols=["SNP"], dtype={"SNP": str})["SNP"]
        need = lambda c: (f"REL_M12_{c}",) + ((f"REL_S12_{c}",) if inb else ())
        classes, A, inside = _annot_or_none(annot, snps, tcols, need)
        t = pd.DataFrame({"SNP": snps, "REL_M12": _class_column(row, "REL_M12", classes, A, inside)})
        if inb:
            t["REL_S12"] = _class_column(row, "REL_S12", classes, A, inside)
    t.to_csv(out, sep="\t", index=False, float_format="%.12g")
    return float(row["REL_M12"])
