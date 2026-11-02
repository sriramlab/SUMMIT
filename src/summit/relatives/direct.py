"""Heritability due to direct genetic effects from sibships, computed exactly from the siblings' class GRMs.

Phenotypes and genotypes are centered within sibships (P), which removes everything that siblings share, including
effects of parental genotypes, assortative mating and population structure. With K_c = P A_c P for the SNPs of
annotation class c, and every pair of individuals of the same extended family left out (the diagonal included), the
moment equations

    sum_d sigma_d tr(K_c^u K_d^u) = y'K_c^u y,     c = 1..C

have no residual term and give unbiased direct-effect variances sigma_c for any terms shared within sibships and any
residual covariance within extended families. The population estimator on the same individuals (covariates projected,
off-diagonal pairs of each extended family left out, diagonal kept, residual equation) is reported alongside, so that
the two are compared in the same people. Both are scaled by the phenotype's residual variance after the covariates
among all individuals of the phenotype file, the scale of the population heritability.

Standard errors: the direct estimate uses a jackknife over contiguous blocks of SNPs, deleting each block from the
numerators and from the SNP side of the denominators as SUMMIT does. A jackknife over groups of families would count the
products of pairs of families twice and overstate it. The population estimate, and the difference and ratio of the two,
use a jackknife over 200 groups of whole extended families.

Memory: one N x N matrix per class (float32 on disk, cached in --grm).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from bed_reader import open_bed
from scipy.linalg import blas

from .moments import read_annotation_classes


class Layout:
    """Individuals sorted by extended family, sibship and IID, with contiguous groups of whole extended families."""

    def __init__(self, sibs: pd.DataFrame, ngroups: int = 200):
        key = {c: (pd.to_numeric(sibs[c]) if pd.to_numeric(sibs[c], errors="coerce").notna().all() else sibs[c])
               for c in ("FAMILY", "SIBSHIP")}
        d = sibs.assign(_f=key["FAMILY"], _s=key["SIBSHIP"]).sort_values(["_f", "_s", "IID"], kind="mergesort")
        d = d.reset_index(drop=True)
        self.iid = d["IID"].to_numpy()
        self.n = len(d)
        self.sib = pd.factorize(d["FAMILY"].astype(str) + "\t" + d["SIBSHIP"].astype(str))[0]
        self.fam = pd.factorize(d["FAMILY"])[0]
        self.sib_start = np.flatnonzero(np.r_[True, np.diff(self.sib) != 0])
        self.fam_start = np.flatnonzero(np.r_[True, np.diff(self.fam) != 0])
        self.fam_end = np.r_[self.fam_start[1:], self.n]
        cut = np.minimum(np.searchsorted(self.fam_start, np.linspace(0, self.n, ngroups + 1)[1:-1]),
                         len(self.fam_start) - 1)
        self.g_start = np.unique(np.r_[0, self.fam_start[cut]])
        self.g_end = np.r_[self.g_start[1:], self.n]

    def members(self, observed: np.ndarray):
        """0/1 member indicator (observed, in a sibship with at least two observed members) and members per sibship."""
        n_obs = np.add.reduceat(observed.astype(float), self.sib_start)
        m = observed & (n_obs[self.sib] >= 2)
        return m.astype(float), np.add.reduceat(m.astype(float), self.sib_start)

    def same_family_pairs(self):
        """Ordered pairs (a, b) of the same extended family, the diagonal included."""
        pa, pb = [], []
        for s, e in zip(self.fam_start, self.fam_end):
            ix = np.arange(s, e)
            pa.append(np.repeat(ix, e - s))
            pb.append(np.tile(ix, e - s))
        return np.concatenate(pa), np.concatenate(pb)


def _center_columns(B, lay, m, n_m):
    """B P: columns centered within sibships over members m; non-members zero."""
    S = np.add.reduceat(B * m[None, :], lay.sib_start, axis=1)
    mean = np.divide(S, n_m[None, :], out=np.zeros_like(S), where=n_m[None, :] > 0)
    return (B - mean[:, lay.sib]) * m[None, :]


def _center_rows(B, sib_rows, starts, m_rows, n_rows):
    """P B for a block of rows holding whole sibships (sibship labels relabelled from 0 within the block)."""
    S = np.add.reduceat(B * m_rows[:, None], starts, axis=0)
    mean = np.divide(S, n_rows[:, None], out=np.zeros_like(S), where=n_rows[:, None] > 0)
    return (B - mean[sib_rows]) * m_rows[:, None]


def _standardize(X):
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu[None, :], X) - mu[None, :]
    sd = np.sqrt((X ** 2).sum(0) / (X.shape[0] - 1))
    ok = sd > 0
    X[:, ok] /= sd[ok]
    X[:, ~ok] = 0.0
    return X, ok


def class_grms(geno: str, lay: Layout, cls: np.ndarray, classes: list[str], cache: str, chunk: int = 4096, log=print):
    """A_c = (sum over SNPs of class c of x x') / M_c for the siblings (standardized among them), as float32 arrays in
    `cache`, reused when present for the same individuals."""
    os.makedirs(cache, exist_ok=True)
    meta_path = os.path.join(cache, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if meta.get("iid") != list(lay.iid) or meta.get("classes") != classes:
        meta = {"iid": list(lay.iid), "classes": classes, "M": {}}
    fam = pd.read_csv(geno + ".fam", sep=r"\s+", header=None, usecols=[1], dtype=str).iloc[:, 0]
    perm = pd.Index(fam).get_indexer(lay.iid)
    if (perm < 0).any():
        raise ValueError(f"{(perm < 0).sum()} siblings are missing from {geno}.fam.")
    bed = open_bed(geno + ".bed", count_A1=True)
    for c, name in enumerate(classes):
        path = os.path.join(cache, f"A_{c}.npy")
        if os.path.exists(path) and name in meta["M"]:
            continue
        idx = np.flatnonzero(cls == c)
        acc = np.zeros((lay.n, lay.n), order="F")
        used = 0
        for s in range(0, idx.size, chunk):
            X, ok = _standardize(bed.read(index=np.s_[perm, idx[s:s + chunk]], dtype="float64", order="F"))
            blas.dsyrk(1.0, np.asfortranarray(X[:, ok]), beta=1.0, c=acc, lower=1, overwrite_c=1)
            used += int(ok.sum())
        acc = np.tril(acc) + np.tril(acc, -1).T
        np.save(path, (acc / used).astype(np.float32))
        meta["M"][name] = used
        json.dump(meta, open(meta_path, "w"))
        log(f"[direct] class {name}: {used} SNPs")
    return [np.load(os.path.join(cache, f"A_{c}.npy"), mmap_mode="r") for c in range(len(classes))], \
        np.array([meta["M"][n] for n in classes], float)


def accumulate(lay, As, m, n_m, yd, observed, Wq, yp):
    """Per group of families: numerators and Gram entries of the direct (centered, family pairs out) and population
    (covariates projected, off-diagonal family pairs out) estimators."""
    C, G = len(As), len(lay.g_start)
    o = observed.astype(float)
    Z = []                                                     # Wq' A_c P_W, for projecting rows
    for A in As:
        QA = np.zeros((Wq.shape[1], lay.n))
        for r0, r1 in zip(lay.g_start, lay.g_end):
            QA += Wq[r0:r1].T @ np.asarray(A[r0:r1], dtype=np.float64)
        Z.append((QA - (QA @ Wq) @ Wq.T) * o[None, :])
    acc = {k: np.zeros((G, C)) for k in ("T", "W", "Tp", "Wp", "tr")}
    acc.update({k: np.zeros((G, C, C)) for k in ("Gr", "Wg", "Gp", "Wgp")})
    acc["yy"] = np.zeros(G)
    for g, (r0, r1) in enumerate(zip(lay.g_start, lay.g_end)):
        rows = slice(r0, r1)
        sib_r = lay.sib[rows] - lay.sib[r0]
        starts = np.flatnonzero(np.r_[True, np.diff(sib_r) != 0])
        fams = [(s, e) for s, e in zip(lay.fam_start, lay.fam_end) if r0 <= s < r1]
        Kd, Kp = [], []
        for c, A in enumerate(As):
            B = np.asarray(A[r0:r1], dtype=np.float64)
            K = _center_rows(_center_columns(B, lay, m, n_m), sib_r, starts, m[rows], n_m[lay.sib[r0:r1]][starts])
            BP = (B - (B @ Wq) @ Wq.T) * o[None, :]
            Kq = (BP - Wq[rows] @ Z[c]) * o[rows][:, None]
            acc["tr"][g, c] = np.trace(Kq[:, rows])
            diag = np.diagonal(Kq[:, rows]).copy()
            for s, e in fams:
                K[s - r0:e - r0, s:e] = 0.0
                Kq[s - r0:e - r0, s:e] = 0.0
            Kq[np.arange(r1 - r0), np.arange(r0, r1)] = diag
            Kd.append(K)
            Kp.append(Kq)
        for c in range(C):
            acc["T"][g, c] = yd[rows] @ (Kd[c] @ yd)
            acc["W"][g, c] = yd[rows] @ (Kd[c][:, rows] @ yd[rows])
            acc["Tp"][g, c] = yp[rows] @ (Kp[c] @ yp)
            acc["Wp"][g, c] = yp[rows] @ (Kp[c][:, rows] @ yp[rows])
            for d in range(c + 1):
                for out, w, Ks in (("Gr", "Wg", Kd), ("Gp", "Wgp", Kp)):
                    acc[out][g, c, d] = acc[out][g, d, c] = (Ks[c] * Ks[d]).sum()
                    acc[w][g, c, d] = acc[w][g, d, c] = (Ks[c][:, rows] * Ks[d][:, rows]).sum()
        acc["yy"][g] = yp[rows] @ yp[rows]
    return acc


def _jackknife_se(full, delete):
    G = delete.shape[0]
    return float(np.sqrt((G - 1) / G * ((delete - delete.mean(0)) ** 2).sum(0)))


def population_fit(acc, nu, n_obs_g):
    def h2(rhs, gram, t, yy, n):
        C = gram.shape[0]
        A = np.zeros((C + 1, C + 1))
        A[:C, :C], A[:C, C], A[C, :C], A[C, C] = gram, t, t, n
        s = np.linalg.solve(A, np.r_[rhs, yy])
        return s[:C].sum() / (s[:C].sum() + s[C])
    tot = {k: v.sum(0) for k, v in acc.items()}
    full = h2(tot["Tp"], tot["Gp"], tot["tr"], tot["yy"], nu)
    dele = np.array([h2(tot["Tp"] - 2 * acc["Tp"][g] + acc["Wp"][g], tot["Gp"] - 2 * acc["Gp"][g] + acc["Wgp"][g],
                        tot["tr"] - acc["tr"][g], tot["yy"] - acc["yy"][g], nu - n_obs_g[g])
                     for g in range(len(n_obs_g))])
    return full, dele


def direct_fit(acc):
    T, W, Gr, Wg = acc["T"], acc["W"], acc["Gr"], acc["Wg"]
    rhs, gram = T.sum(0), Gr.sum(0)
    full = np.linalg.solve(gram, rhs)
    dele = np.array([np.linalg.solve(gram - 2 * Gr[g] + Wg[g], rhs - 2 * T[g] + W[g]) for g in range(T.shape[0])])
    return gram, rhs, full, dele


def snp_block_pieces(geno, lay, As, M, cls, blk, nblocks, Y, chunk=4096, log=print):
    """One genotype pass with every sibling a member. D[b, c, d]: contribution of the class-c SNPs of block b to
    tr(K_c^u K_d^u) (one-sided, as SUMMIT's LD-score jackknife); num[b, c, r]: the block's part of y_r'K_c^u y_r."""
    C = len(As)
    fam = pd.read_csv(geno + ".fam", sep=r"\s+", header=None, usecols=[1], dtype=str).iloc[:, 0]
    perm = pd.Index(fam).get_indexer(lay.iid)
    pa, pb = lay.same_family_pairs()
    m1, n1 = lay.members(np.ones(lay.n, dtype=bool))
    Af = [np.asarray(A, dtype=np.float32) for A in As]
    Ks = []                                                  # same-family entries of P A_d P
    for A in Af:
        v = []
        for s, e in zip(lay.fam_start, lay.fam_end):
            sib = lay.sib[s:e] - lay.sib[s]
            Pm = np.eye(e - s) - (sib[:, None] == sib[None, :]) / np.bincount(sib)[sib][None, :]
            v.append((Pm @ A[s:e, s:e].astype(np.float64) @ Pm).ravel())
        Ks.append(np.concatenate(v))
    D = np.zeros((nblocks, C, C))
    num = np.zeros((nblocks, C, Y.shape[1]))
    AS = np.zeros((nblocks, C, pa.size))
    bed = open_bed(geno + ".bed", count_A1=True)
    for s in range(0, cls.size, chunk):
        X, ok = _standardize(bed.read(index=np.s_[perm, s:s + chunk], dtype="float64", order="F"))
        Xt = _center_columns(X.T, lay, m1, n1).T
        Xf = np.asfortranarray(Xt, dtype=np.float32)
        prod = Xt[pa] * Xt[pb]
        q = np.column_stack([(Xf * (A @ Xf)).sum(0, dtype=np.float64) for A in Af])
        sst = np.column_stack([K @ prod for K in Ks])
        gg = X[pa] * X[pb]
        z2 = (X.T @ Y) ** 2
        cj, bj = cls[s:s + chunk], blk[s:s + chunk]
        for c in range(C):
            for b in np.unique(bj[(cj == c) & ok]):
                sel = (cj == c) & (bj == b) & ok
                D[b, c] += (q[sel] - sst[sel]).sum(0) / M[c]
                AS[b, c] += gg[:, sel].sum(1) / M[c]
                num[b, c] += z2[sel].sum(0) / M[c]
        log(f"[direct] SNP-block pass: {min(s + chunk, cls.size)} of {cls.size} SNPs")
    num -= np.einsum("bcp,pr->bcr", AS, Y[pa] * Y[pb])
    return D, num


def direct_effects(geno: str, sibships: str, annot: str, pheno: str, out: str, *, covar: str,
                   within_covar=None, pheno_cols=None, grm: str | None = None, nblocks: int = 200, log=print):
    sibs = pd.read_csv(sibships, sep=r"\s+", dtype=str)
    missing = {"IID", "SIBSHIP", "FAMILY"} - set(sibs.columns)
    if missing:
        raise ValueError(f"{sibships} needs columns IID, SIBSHIP and FAMILY (missing {sorted(missing)}).")
    lay = Layout(sibs)
    bim = pd.read_csv(geno + ".bim", sep=r"\s+", header=None, usecols=[1], dtype=str).iloc[:, 0].to_numpy()
    Zc, classes = read_annotation_classes(annot, bim)
    if (Zc.sum(1) != 1).any():
        raise ValueError(f"Every SNP of {geno}.bim must belong to one class of {annot}.")
    cls = Zc.argmax(1)
    blk = np.minimum((np.arange(cls.size) * nblocks) // cls.size, nblocks - 1)
    As, M = class_grms(geno, lay, cls, classes, grm or out + ".grm", log=log)

    ph = pd.read_csv(pheno, sep=r"\s+", dtype={"FID": str, "IID": str}).set_index("IID")
    cv = pd.read_csv(covar, sep=r"\s+", dtype={"FID": str, "IID": str}).set_index("IID").drop(columns=["FID"])
    cv = cv.apply(pd.to_numeric, errors="coerce")
    within_covar = list(within_covar or [])
    traits = pheno_cols or [c for c in ph.columns if c != "FID"]
    rows, Ys = [], []
    for t in traits:
        y = pd.to_numeric(ph[t], errors="coerce")
        ok = y.notna() & cv.reindex(y.index).notna().all(axis=1)
        W = np.column_stack([np.ones(ok.sum()), cv.loc[y.index[ok]].to_numpy(float)])
        res = y[ok].to_numpy() - W @ np.linalg.lstsq(W, y[ok].to_numpy(), rcond=None)[0]
        var_full = float(res @ res / (ok.sum() - W.shape[1]))       # scale of the population heritability
        W1 = np.column_stack([np.ones(ok.sum())] + ([cv.loc[y.index[ok], within_covar].to_numpy(float)]
                                                    if within_covar else []))
        y1 = pd.Series(y[ok].to_numpy() - W1 @ np.linalg.lstsq(W1, y[ok].to_numpy(), rcond=None)[0],
                       index=y.index[ok]).reindex(lay.iid).to_numpy()
        observed = np.isfinite(y1)
        m, n_m = lay.members(observed)
        yc = np.where(m > 0, y1, 0.0)
        mean = np.divide(np.add.reduceat(yc, lay.sib_start), n_m, out=np.zeros(n_m.size), where=n_m > 0)
        yd = (yc - mean[lay.sib]) * m
        Wsib = np.column_stack([np.ones(lay.n), cv.reindex(lay.iid).to_numpy(float)])
        Q = np.linalg.qr(Wsib[observed])[0]
        Wq = np.zeros((lay.n, Q.shape[1]))
        Wq[observed] = Q
        yo = np.where(observed, pd.to_numeric(ph[t], errors="coerce").reindex(lay.iid).to_numpy(float), 0.0)
        yp = (yo - Wq @ (Wq.T @ yo)) * observed
        acc = accumulate(lay, As, m, n_m, yd, observed, Wq, yp)
        gram, rhs, dfull, ddel = direct_fit(acc)
        n_obs_g = np.array([observed[s:e].sum() for s, e in zip(lay.g_start, lay.g_end)])
        pfull, pdel = population_fit(acc, int(observed.sum()) - Q.shape[1], n_obs_g)
        h_d, h_d_del = dfull.sum() / var_full, ddel.sum(1) / var_full
        rows.append(dict(trait=t, n_members=int(m.sum()), n_sibships=int((n_m >= 2).sum()), h2_direct=float(h_d),
                         h2_population=float(pfull), h2_population_se=_jackknife_se(pfull, pdel),
                         difference=float(pfull - h_d), difference_se=_jackknife_se(pfull - h_d, pdel - h_d_del),
                         ratio=float(h_d / pfull), ratio_se=_jackknife_se(h_d / pfull, h_d_del / pdel),
                         var_full=var_full, _gram=gram, _rhs=rhs, _classes=dfull / var_full))
        Ys.append(yd)
        log(f"[direct] {t}: direct {h_d:.3f}, population {pfull:.3f} in {int(m.sum())} siblings")

    D, num = snp_block_pieces(geno, lay, As, M, cls, blk, nblocks, np.column_stack(Ys), log=log)
    frac = D / D.sum(0, keepdims=True)
    for k, r in enumerate(rows):
        gram, rhs, nb = r.pop("_gram"), r.pop("_rhs"), num[:, :, k]
        if np.abs(nb.sum(0) - rhs).max() > 1e-5 * np.abs(rhs).max():
            raise RuntimeError(f"{r['trait']}: SNP-block numerators do not add up to the exact numerators.")
        dele = np.array([np.linalg.solve(gram * (1 - frac[b]), rhs - nb[b]).sum() for b in range(nblocks)])
        r["h2_direct_se"] = _jackknife_se(r["h2_direct"], dele / r["var_full"])
        for c, v in zip(classes, r.pop("_classes")):
            r[f"h2_direct_{c}"] = float(v)
    cols = ["trait", "n_members", "n_sibships", "h2_direct", "h2_direct_se", "h2_population", "h2_population_se",
            "difference", "difference_se", "ratio", "ratio_se", "var_full"]
    res = pd.DataFrame(rows)
    res = res[cols + [c for c in res.columns if c not in cols]]
    res.to_csv(out + ".direct.tsv", sep="\t", index=False, float_format="%.6g")
    return res
