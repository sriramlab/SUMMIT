"""Test and interval for the local-ancestry component, computed exactly from the genotypes.

The joint model has one kernel per annotation class (SNP bins and the LA class of `summit local-ancestry merge`),
K_c = P A_c P with A_c the GRM of the class's markers and P the projection orthogonal to the covariates. As in
`summit --h2`, the phenotype is projected on the covariates and scaled to y'y = nu, each individual's pairing with
themselves is kept, and related pairs are left out of the class equations. The fit equals SUMMIT's summary-statistic
fit with in-sample LD scores.

A jackknife over SNP blocks or chromosomes cannot test the component when it comes from a single locus, because deleting
the block that carries the locus removes the whole component. Instead:

  z test   the local-ancestry estimate divided by its standard deviation under no local-ancestry effect,
           2 tr(B S0 B S0), with S0 the phenotypic covariance implied by the SNP-only fit (SNP components clipped at 0)
           and B the quadratic form that gives the estimate;
  interval a sandwich over families (related individuals form one unit, everyone else is a unit of their own),
           sum over units of (psi_F - mean)^2, less the model expectation of the products of different units' residuals
           that this sum counts twice; psi_F is unit F's contribution to the linearized estimate.

Cost: one N x N matrix per class, so this is for cohorts of up to tens of thousands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from bed_reader import open_bed
from scipy.stats import norm

from ..relatives.moments import read_annotation_classes, read_relative_pairs
from ..ldscore.genotype_source import read_fam_sample_ids


def _project(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    AQ = A @ Q
    return A - Q @ AQ.T - AQ @ Q.T + Q @ (Q.T @ AQ) @ Q.T


def class_grams(geno: str, annot: str, rows: np.ndarray, block: int = 4096):
    """Unscaled class Gram matrices X_c X_c' over `rows` (genotypes standardized among `rows`) and markers per class."""
    bim = pd.read_csv(geno + ".bim", sep=r"\s+", header=None, usecols=[1], dtype=str).iloc[:, 0].to_numpy()
    Z, classes = read_annotation_classes(annot, bim)
    if (Z.sum(1) != 1).any():
        raise ValueError(f"Every marker of {geno}.bim must belong to one class of {annot}.")
    cls = Z.argmax(1)
    bed = open_bed(geno + ".bed", count_A1=True)
    n = rows.size
    A = np.zeros((len(classes), n, n))
    for s in range(0, bim.size, block):
        X = bed.read(index=np.s_[rows, s:s + block], dtype="float64")
        mu = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
        sd = X.std(axis=0, ddof=1)
        ok = sd > 0
        X[:, ok] /= sd[ok]
        X[:, ~ok] = 0.0
        cb = cls[s:s + block]
        for k in np.unique(cb):
            Xk = X[:, cb == k]
            A[k] += Xk @ Xk.T
    return A, np.bincount(cls, minlength=len(classes)).astype(float), classes


def unit_pairs(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairs (i < j) of individuals in the same unit."""
    order = np.argsort(unit, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(unit[order]) != 0, True])
    pi, pj = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s > 1:
            ii, jj = np.triu_indices(e - s, 1)
            pi.append(order[s:e][ii]); pj.append(order[s:e][jj])
    return np.concatenate(pi), np.concatenate(pj)


class JointFit:
    """Normal equations of the joint fit for projected kernels Ks, covariate basis Q, the related pairs (pi, pj) left out
    of the class equations, and the units of the sandwich (families)."""

    def __init__(self, Ks, Q, pi, pj, unit):
        self.Ks, self.Q, self.pi, self.pj = Ks, Q, pi, pj
        self.ui, self.uj = unit_pairs(unit)
        self.n, self.K = Q.shape[0], len(Ks)
        self.nu = self.n - Q.shape[1]
        K = self.K
        flat = np.array([Kc.reshape(-1) for Kc in Ks])
        self.T = flat @ flat.T                                   # tr(K_c K_d)
        del flat
        self.trK = np.array([np.trace(Kc) for Kc in Ks])
        self.fp = np.array([Kc[pi, pj] for Kc in Ks])           # kernel entries of the related pairs
        Hij = (Q[pi] * Q[pj]).sum(1)
        self.A = np.zeros((K + 1, K + 1))
        self.A[:K, :K] = self.T - 2 * self.fp @ self.fp.T
        self.A[:K, K] = self.trK + 2 * self.fp @ Hij
        self.A[K, :K] = self.trK
        self.A[K, K] = self.nu
        self.unit = unit
        self.U = sp.csr_matrix((np.ones(self.n), (np.arange(self.n), unit)), shape=(self.n, unit.max() + 1))

    def fit(self, y: np.ndarray) -> np.ndarray:
        q = [y @ Kc @ y - 2 * (self.fp[c] * y[self.pi] * y[self.pj]).sum() for c, Kc in enumerate(self.Ks)]
        return np.linalg.solve(self.A, np.r_[q, y @ y])

    def form(self, w: np.ndarray) -> np.ndarray:
        """The matrix B with w'theta = y'By (projected on the covariates)."""
        a = np.linalg.solve(self.A.T, np.r_[w, 0.0])
        B = np.zeros((self.n, self.n))
        for c, Kc in enumerate(self.Ks):
            B += a[c] * Kc
        v = a[:self.K] @ self.fp
        B[self.pi, self.pj] -= v
        B[self.pj, self.pi] -= v
        B[np.diag_indices(self.n)] += a[self.K]
        B -= a[self.K] * (self.Q @ self.Q.T)
        return _project(B, self.Q)


def null_variance(B: np.ndarray, S0: np.ndarray) -> float:
    """2 tr(B S0 B S0)."""
    BS = B @ S0
    return float(2 * (BS * BS.T).sum())


def sandwich_variance(fit: JointFit, B: np.ndarray, y: np.ndarray, estimate: float) -> float:
    """Variance of estimate = y'By / (y'y / nu) with families as units, less the doubly counted products of residuals of
    different units (their model expectation for a phenotype of unit variance)."""
    P = np.eye(fit.n) - fit.Q @ fit.Q.T
    r = estimate / fit.nu
    Bs = B - r * P                                                # linearization of the ratio
    Bsy = Bs @ y
    within_quad = fit.U.T @ (np.diag(Bs) * y * y)
    pair_bs = Bs[fit.ui, fit.uj]
    if fit.ui.size:
        within_quad += fit.U[fit.ui].T @ (2 * pair_bs * y[fit.ui] * y[fit.uj])
    psi = 2 * (fit.U.T @ (y * Bsy)) - within_quad
    naive = float(((psi - psi.mean()) ** 2).sum())
    within = float((np.diag(Bs) ** 2).sum() + 2 * (pair_bs ** 2).sum())
    return naive - 2 * (float((Bs * Bs).sum()) - within)


def local_ancestry_test(geno: str, annot: str, pheno: str, out: str, *, covar: str | None = None,
                        pheno_cols=None, pairs: str | None = None, kinship_col=None, kinship_min=None,
                        closure: bool = True, min_n: int = 500, log=print) -> pd.DataFrame:
    sample_ids = read_fam_sample_ids(geno + ".fam")
    n_all = len(sample_ids)
    keep = np.ones(n_all, dtype=bool)
    W = np.ones((n_all, 1))
    if covar is not None:
        cv = sample_ids.merge(pd.read_csv(covar, sep=r"\s+", dtype={"FID": str, "IID": str}), on=["FID", "IID"],
                              how="left").drop(columns=["FID", "IID"]).apply(pd.to_numeric, errors="coerce")
        keep = cv.notna().all(axis=1).to_numpy()
        W = np.column_stack([W, cv.to_numpy(float)])
    rows = np.flatnonzero(keep)
    if pairs is not None:
        rp = read_relative_pairs(pairs, sample_ids, kinship_col=kinship_col, kinship_min=kinship_min, closure=closure)
        pair_i, pair_j, family = rp.i, rp.j, rp.family
    else:
        pair_i = pair_j = np.zeros(0, dtype=np.int64)
        family = np.arange(n_all)
    log(f"[local-ancestry] class GRMs of {rows.size} individuals")
    A, M, classes = class_grams(geno, annot, rows)
    if "LA" not in classes:
        raise ValueError(f"{annot} has no LA class (see `summit local-ancestry merge`).")
    la = classes.index("LA")
    snp = [k for k in range(len(classes)) if k != la]
    ph = sample_ids.merge(pd.read_csv(pheno, sep=r"\s+", dtype={"FID": str, "IID": str}), on=["FID", "IID"], how="left")
    cols = pheno_cols or [c for c in ph.columns if c not in ("FID", "IID")]
    pos = -np.ones(n_all, dtype=np.int64)
    pos[rows] = np.arange(rows.size)
    results = []
    for t in cols:
        y_all = pd.to_numeric(ph[t], errors="coerce").to_numpy(float)
        m = np.flatnonzero(np.isfinite(y_all[rows]))                 # positions within `rows`
        if m.size < min_n:
            log(f"[local-ancestry] {t}: {m.size} individuals, skipped")
            continue
        Q, _ = np.linalg.qr(W[rows[m]])
        Ks = [_project(A[k][np.ix_(m, m)] / M[k], Q) for k in range(len(classes))]
        inside = -np.ones(n_all, dtype=np.int64)
        inside[rows[m]] = np.arange(m.size)
        a, b = inside[pair_i], inside[pair_j]
        ok = (a >= 0) & (b >= 0)
        unit = pd.factorize(family[rows[m]])[0]
        fit = JointFit(Ks, Q, a[ok], b[ok], unit)
        y = y_all[rows[m]]
        y = y - Q @ (Q.T @ y)
        y *= np.sqrt(fit.nu / (y @ y))
        theta = fit.fit(y)
        snp_only = JointFit([Ks[k] for k in snp], Q, a[ok], b[ok], unit)
        theta0 = snp_only.fit(y)
        weights = np.r_[np.maximum(theta0[:len(snp)], 0), theta0[-1]]
        S0 = sum(w * Ks[k] for w, k in zip(weights[:-1], snp)) + weights[-1] * (np.eye(m.size) - Q @ Q.T)
        row = dict(trait=t, n=int(m.size), families=int(unit.max() + 1), related_pairs=int(ok.sum()))
        for name, w in (("la", np.eye(len(classes))[la]), ("snp", np.isin(np.arange(len(classes)), snp).astype(float))):
            B = fit.form(w)
            est = float(w @ theta[:-1])
            row[name] = est
            row[f"{name}_se"] = float(np.sqrt(max(sandwich_variance(fit, B, y, est), 0.0)))
            if name == "la":
                row["la_null_sd"] = float(np.sqrt(null_variance(B, S0)))
        row["la_z"] = row["la"] / row["la_null_sd"]
        row["la_p"] = float(norm.sf(row["la_z"]))
        results.append(row)
        log(f"[local-ancestry] {t}: n={row['n']} LA {row['la']:.4f} (SE {row['la_se']:.4f}, z {row['la_z']:.2f}) "
            f"SNP {row['snp']:.4f} (SE {row['snp_se']:.4f})")
    res = pd.DataFrame(results)
    res.to_csv(out + ".la_test.tsv", sep="\t", index=False, float_format="%.6g")
    return res
