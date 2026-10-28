"""Genotype-only block-pair traces for the rescaled jackknife (`summit relatives traces`).

The delete-one-block jackknife misses covariance between blocks (long-range admixture LD, the self-pair terms of
individuals whose self-relatedness tracks ancestry). Under the fitted model y ~ N(0, g K~ + e P) the variance of the
method-of-moments estimate and the expectation of its jackknife variance are both quadratic in the numerators
q_u = y'A~_u y of the jackknife units u, with Cov(q_u, q_v) = 2 [e^2 G1_uv + g e (G2_uv + G2_vu) + g^2 G3_uv],

    G1_uv = tr(A~_u A~_v),   G2_uv = tr(A~_u K~ A~_v),   G3_uv = tr(A~_u K~ A~_v K~),

so `summit --h2 --jk-traces` rescales the jackknife variance by AV / E_model[V_jk] (inference.jktraces).

Kernels are SUMMIT's SNP-variance-weighted ones: with x_j the standardized genotypes (in-sample mean and SD) projected
on the covariates (intercept included), K~_k = sum_{j in k} x_j x_j' / sum_{j in k} V_j, V_j = ||x_j||^2 / N_eff, and
the one-class K~ = sum_j x_j x_j' / sum_j V_j. A unit is the intersection of one of SUMMIT's equal-count contiguous
jackknife blocks (on this genotype file's SNP axis), a chromosome and a class, so that both the block and the
chromosome jackknife are unions of units. Its matrix is projected, A~_u = P A_u P, with
    A_u = K~_u^u - V_bar_u I,  V_bar_u = tr(K~_u) / nu                                   (self-pairs in, SUMMIT)
    A_u = K~_u^u - diag(K~_u) - (lambda_u / nu) I,  lambda_u = sum_i K~_u,ii H_ii + sum_{ordered pairs} K~_u,ii' H_ii'
                                                                                      (self-pairs out, --inbreeding)
where K~_u^u is the unit's share of K~_k without the related pairs and the second equation y'y has been eliminated.
Tb[u, d] = tr(A~_u K~_d) gives the moment matrices of every leave-out replicate. The signal part of the covariance uses
the one-class kernel for partitioned fits too (an approximation used only for the variance).

All traces are Rademacher-probe estimates (E z'Bz = tr B), from the projected genotypes held in memory (samples x
SNPs, float32).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ..inference.jackknife import _equal_count_block_boundaries
from .moments import (
    _center_impute,
    _covariate_basis,
    _auto_step,
    pairs_within_sample,
    read_annotation_classes,
    read_relative_pairs,
)
from ..ldscore.genotype_source import read_fam_sample_ids, resolve_genotype_input
from bed_reader import open_bed


def _units(bim: pd.DataFrame, nblocks: int, cls: np.ndarray, nclass: int):
    """Cells (block x chromosome) and units (cell x class) of every SNP, in SNP order."""
    M = len(bim)
    starts, ends = _equal_count_block_boundaries(M, int(nblocks))
    blk = np.empty(M, dtype=np.int64)
    for b, (s, e) in enumerate(zip(starts, ends)):
        blk[s:e] = b
    chrom = pd.factorize(bim["CHR"].astype(str))[0].astype(np.int64)
    key = blk * 1000 + chrom
    ucell, cell = np.unique(key, return_inverse=True)
    unit_key = cell.astype(np.int64) * nclass + cls
    ukeys, unit = np.unique(unit_key, return_inverse=True)
    return cell, ucell // 1000, unit, ukeys // nclass, ukeys % nclass


def compute_jackknife_traces(
    geno: str,
    out: str,
    *,
    covar: str | None = None,
    pairs: str | None = None,
    kinship_col: str | None = None,
    kinship_min: float | None = None,
    closure: bool = True,
    annot: str | None = None,
    inbreeding: bool = False,
    keep: str | None = None,
    nblocks: int = 200,
    probes: int = 100,
    seed: int = 20261029,
    step_size: int | None = None,
    log=None,
) -> dict:
    """Write `out`.jktraces.npz (G1, G2, G3, Tb and, with `inbreeding`, the self-pair-free G1n, G2n, G3n, Tbn) and
    `out`.jktraces.snps.tsv.gz (each SNP's cell), for everyone with complete covariates (and in `keep`, if given)."""
    def _log(msg):
        if log is not None:
            log._log(msg)

    t0 = time.time()
    gi = resolve_genotype_input(geno)
    if gi.format != "bed":
        raise ValueError("Jackknife traces currently read PLINK 1 BED input.")
    sample_ids = read_fam_sample_ids(gi.sample_path)
    bim = pd.read_csv(gi.variant_path, sep=r"\s+", header=None,
                      names=["CHR", "SNP", "CM", "BP", "A1", "A2"], dtype={"SNP": str, "CHR": str})
    bed = open_bed(gi.genotype_path)
    n_all, M = bed.shape
    ok = np.ones(n_all, dtype=bool)
    if covar is not None:
        cv = pd.read_csv(covar, sep=r"\s+", dtype={"FID": str, "IID": str}, keep_default_na=False)
        cm = sample_ids.merge(cv, on=["FID", "IID"], how="left")
        ok &= ~cm.drop(columns=["FID", "IID"]).apply(pd.to_numeric, errors="coerce").isna().any(axis=1).to_numpy()
    if keep is not None:
        kp = pd.read_csv(keep, sep=r"\s+", header=None, usecols=[0, 1], names=["FID", "IID"], dtype=str)
        ok &= sample_ids.merge(kp.assign(_k=True), on=["FID", "IID"], how="left")["_k"].fillna(False).to_numpy(bool)
    idx = np.flatnonzero(ok)
    n = int(idx.size)
    C = _covariate_basis(covar, sample_ids, idx)
    p = int(C.shape[1])
    n_eff = float(n - p - 1)
    nu = n_eff
    if annot is not None:
        Acls, classes = read_annotation_classes(annot, bim["SNP"].to_numpy())
        if np.any(Acls.sum(axis=1) != 1):
            raise ValueError("Every SNP of the genotype file must belong to exactly one class of --annot.")
        cls = Acls.argmax(axis=1)
    else:
        classes, cls = [], np.zeros(M, dtype=np.int64)
    K = max(1, len(classes))
    cell, cell_blk, unit, unit_cell, unit_cls = _units(bim, nblocks, cls, K)
    NU = unit_cell.size
    order = np.argsort(unit, kind="stable")                       # SNPs grouped by unit
    pos = np.empty(M, dtype=np.int64); pos[order] = np.arange(M)
    ustart = np.r_[0, np.cumsum(np.bincount(unit, minlength=NU))]
    cell_chr = bim["CHR"].astype(str).to_numpy()[np.array([np.flatnonzero(cell == c)[0] for c in range(cell_blk.size)])]

    X = np.empty((n, M), dtype=np.float32)
    ssq = np.zeros(M)
    step = _auto_step(step_size, n)
    for s in range(0, M, step):
        e = min(M, s + step)
        G = _center_impute(np.asarray(bed.read(index=np.s_[idx, s:e], dtype="float64")))
        raw = np.einsum("ij,ij->j", G, G)
        okc = raw > 1e-10
        G[:, okc] *= np.sqrt((n - 1) / raw[okc])[None, :]
        G[:, ~okc] = 0.0
        if p:
            G -= C @ (C.T @ G)
        ssq[s:e] = np.einsum("ij,ij->j", G, G)
        X[:, pos[s:e]] = G.astype(np.float32)
    ssq = ssq[order]
    ucls = unit_cls
    V = ssq / n_eff
    vsum = np.bincount(cls[order], weights=V, minlength=K)
    vsum_all = float(V.sum())
    norm = vsum[ucls]                                             # unit kernels are class-normalized
    trKu = np.array([ssq[ustart[u]:ustart[u + 1]].sum() for u in range(NU)]) / norm
    Vb = trKu / nu
    _log(f"[traces] {n} samples x {M} SNPs, {cell_blk.size} cells, {NU} units ({K} class(es)), "
         f"loaded in {time.time() - t0:.0f}s.")

    # related pairs: the unit kernels on the pairs (both orders appear through a, b)
    if pairs is not None:
        rp = read_relative_pairs(pairs, sample_ids, kinship_col=kinship_col, kinship_min=kinship_min, closure=closure)
        a, b = pairs_within_sample(rp, idx, n_all)
    else:
        a = b = np.zeros(0, dtype=np.int64)
    PK = np.zeros((a.size, NU))
    for u in range(NU) if a.size else ():
        sl = slice(ustart[u], ustart[u + 1])
        PK[:, u] = np.einsum("ij,ij->i", X[a, sl], X[b, sl], dtype=np.float64) / norm[u]
    Hd = np.full(n, 1.0 / n) + ((C * C).sum(axis=1) if p else 0.0)
    if inbreeding:
        dK = np.empty((n, NU))
        for u in range(NU):
            Xu = X[:, ustart[u]:ustart[u + 1]]
            dK[:, u] = np.einsum("ij,ij->i", Xu, Xu, dtype=np.float64) / norm[u]
        Hab = np.full(a.size, 1.0 / n) + ((C[a] * C[b]).sum(axis=1) if p else 0.0)
        lam = dK.T @ Hd + 2.0 * (PK.T @ Hab)

    def proj(v):
        v = v - v.mean(axis=0)
        return v - C @ (C.T @ v) if p else v

    def unit_mul(v):
        """[K~_u v]_u (n x NU) and the one-class K~ v, for a projected n-vector v."""
        xt = X.T @ v.astype(np.float32)
        out_ = np.empty((n, NU))
        for u in range(NU):
            out_[:, u] = (X[:, ustart[u]:ustart[u + 1]] @ xt[ustart[u]:ustart[u + 1]]).astype(np.float64) / norm[u]
        return out_, (X @ xt).astype(np.float64) / vsum_all

    def rel_mul(v):
        out_ = np.zeros((n, NU))
        if a.size:
            np.add.at(out_, a, PK * v[b][:, None])
            np.add.at(out_, b, PK * v[a][:, None])
        return out_

    def kmul(U):
        return (X @ (X.T @ U.astype(np.float32))).astype(np.float64) / vsum_all

    rng = np.random.default_rng(seed)
    acc = {k: np.zeros((NU, NU)) for k in ("G1", "G2", "G3")}
    acc["Tb"] = np.zeros((NU, K))
    if inbreeding:
        acc.update({k: np.zeros((NU, NU)) for k in ("G1n", "G2n", "G3n")})
        acc["Tbn"] = np.zeros((NU, K))
    for it in range(int(probes)):
        z = proj(rng.choice([-1.0, 1.0], size=n))
        Ku, Kz = unit_mul(z)
        Ru = proj(rel_mul(z))
        Kd = np.zeros((n, K))                                     # K~_d z by class (pairs and diagonal included)
        for u in range(NU):
            Kd[:, ucls[u]] += Ku[:, u]
        Ku2, _ = unit_mul(Kz)
        Ru2 = proj(rel_mul(Kz))
        variants = [("", Ku - Ru - z[:, None] * Vb[None, :], Ku2 - Ru2 - Kz[:, None] * Vb[None, :])]
        if inbreeding:
            ln = lam / nu
            variants.append(("n", Ku - Ru - proj(dK * z[:, None]) - z[:, None] * ln[None, :],
                             Ku2 - Ru2 - proj(dK * Kz[:, None]) - Kz[:, None] * ln[None, :]))
        for sfx, U, Vm in variants:                               # U = A~_u z, Vm = A~_u K~ z
            KU = kmul(U)
            acc["G1" + sfx] += U.T @ U
            acc["G2" + sfx] += KU.T @ U
            acc["G3" + sfx] += KU.T @ Vm
            acc["Tb" + sfx] += U.T @ Kd
        if (it + 1) % max(1, int(probes) // 10) == 0:
            _log(f"[traces] probe {it + 1}/{probes} ({time.time() - t0:.0f}s)")
    for k in acc:
        acc[k] /= float(probes)
        if k.startswith(("G1", "G3")):
            acc[k] = 0.5 * (acc[k] + acc[k].T)
    meta = dict(n=n, nu=nu, M=M, probes=int(probes), nblocks=int(nblocks), seed=int(seed), npairs=int(a.size),
                inbreeding=bool(inbreeding), classes=list(classes))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez(out + ".jktraces.npz", **acc, Vb=Vb, unit_cell=unit_cell, unit_cls=unit_cls, cell_blk=cell_blk,
             cell_chr=cell_chr.astype(str), vsum=vsum, vsum_all=vsum_all, classes=np.array(classes, dtype=str),
             **({"lam": lam} if inbreeding else {}), meta=json.dumps(meta))
    pd.DataFrame({"SNP": bim["SNP"], "CHR": bim["CHR"], "BP": bim["BP"], "CELL": cell}).to_csv(
        out + ".jktraces.snps.tsv.gz", sep="\t", index=False)
    _log(f"[traces] wrote {out}.jktraces.npz ({NU} units, {probes} probes) in {time.time() - t0:.0f}s.")
    return meta
