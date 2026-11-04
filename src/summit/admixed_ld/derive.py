"""A source missing from the reference panels, estimated from an admixed cohort (`summit admixed-ld derive`).

Some sources have no large reference panel (for example Native American ancestry). An admixed cohort that carries the
source supplies it. Given panels of the other sources, the cohort's genotypes and, optionally, a small panel of the
missing source, the command estimates:

1. Each person's ancestry and the missing source's allele frequencies, by alternating least squares on
   E(g_ij / 2) = sum_s theta_is p^s_j with the known sources' frequencies fixed (supervised admixture). Members of the
   small panel enter with their ancestry fixed to the missing source, which anchors its frequencies; without them the
   frequencies are identified only if some cohort members carry nearly all of the source.
2. Its dosage covariances within the window, by inverting the composition of the cohort's covariate-adjusted covariance,
       C^N_jk = [C_res(j,k) - sum_{s known} pi_s(d) C^s_jk - D_j' Kt(d) D_k] / pi_N(d),
   with pi_s(d) = m_s exp(-lam_s d), Kt(d) and the decays from the cohort's curve on a provisional reference, and m and
   Kt_res(0) from the regression of the cohort's adjusted genotype variances on the source variances and contrasts.
   The sampling variance of each derived covariance, [Var(C_res) + sum_s pi_s^2 Var(C^s)] / pi_N^2, is stored with it
   (covvar) and replaces the panel moments when `compose` removes sampling noise from squares.

The output is a reference with the known sources and the derived one, restricted to the cohort's matched SNPs, for
`curve` and `compose` of any cohort with these sources. It holds only aggregates: frequencies, covariances and their
sampling variances.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from ._io import DosageBlocks, center_impute, match_variants
from .compose import unbiased_squared_covariance
from .model import full_from_contrasts, regress_standard_errors_k, unbiased_products
from .reference import Reference, _products, write_manifest


def _simplex_ls(Y, P, fixed=None):
    """Per-row least squares of Y (n x S) on the columns of P (S x K) with theta >= 0 and sum(theta) = 1 (active set:
    negative coordinates are dropped and the rest refitted, at most K times). fixed: rows with given theta."""
    n, K = Y.shape[0], P.shape[1]
    active = np.ones((n, K), dtype=bool)
    theta = np.zeros((n, K))
    for _ in range(K):
        for pattern in np.unique(active, axis=0):
            rows = np.flatnonzero((active == pattern).all(axis=1))
            idx = np.flatnonzero(pattern)
            if idx.size == 1:
                theta[np.ix_(rows, idx)] = 1.0
                continue
            Pa = P[:, idx]
            base = Pa[:, -1]                                     # substitute the last active coordinate
            A = Pa[:, :-1] - base[:, None]
            sol = np.linalg.lstsq(A, (Y[rows] - base[None, :]).T, rcond=None)[0].T
            t = np.zeros((rows.size, K))
            t[:, idx[:-1]] = sol
            t[:, idx[-1]] = 1.0 - sol.sum(axis=1)
            theta[rows] = t
        neg = (theta < 0) & active
        if not neg.any():
            break
        active &= ~neg
        theta[~active] = 0.0
    theta = np.clip(theta, 0.0, None)
    theta /= theta.sum(axis=1, keepdims=True)
    if fixed is not None:
        rows, val = fixed
        theta[rows] = val
    return theta


def derive_source(geno, ref_path, out, *, name, known=None, panel=None, covar=None, keep=None, match="position",
                  als_snps=20000, als_iter=100, cell_cm=0.25, step=1024, threads=1, native=True, log=print):
    """Write a reference with the known sources of `ref_path` and `name` derived from the cohort `geno`."""
    from .curve import _cohort_samples, genotype_curve
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    known = list(known or ref.names)
    if name in known:
        raise ValueError(f"{name} is already a source of the reference.")
    K = len(known) + 1
    samples, rows, C = _cohort_samples(geno, covar, keep, log)
    blocks = DosageBlocks(geno, rows=rows, threads=threads, step=step)
    N = blocks.n
    rv = ref.variants(known)
    t_idx, r_idx, flip = match_variants(blocks.variants, rv, how=match)
    order = np.argsort(r_idx, kind="stable")                    # reference (genome) order
    t_idx, r_idx, flip = t_idx[order], r_idx[order], flip[order]
    M = t_idx.size
    # known frequencies of the cohort's counted allele
    Pk = np.column_stack([rv[f"p_{s}"].to_numpy()[r_idx] for s in known])
    Pk_c = np.where(flip[:, None], 1.0 - Pk, Pk)
    log(f"[admixed-ld] derive {name}: {M} of {len(blocks.variants)} cohort variants matched to the reference "
        f"({', '.join(known)}); {N} samples")

    # ------------------------------------------------------------------ 1. supervised admixture on a subset of SNPs
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(M, min(als_snps, M), replace=False))
    Ys = blocks.read(t_idx[sub]) / 2.0
    mu = np.nanmean(Ys, axis=0)
    Ys = np.where(np.isnan(Ys), mu[None, :], Ys)
    keep_sub = (mu > 0.02) & (mu < 0.98)
    sub, Ys = sub[keep_sub], Ys[:, keep_sub]
    fixed_rows = None
    if panel is not None:                                        # small panel of the missing source, theta = e_N
        pb = DosageBlocks(panel, threads=threads, step=step)
        pt, pr, pf = match_variants(pb.variants, rv, how=match)
        pos = pd.Series(np.arange(pr.size), index=pr)
        have = pos.reindex(r_idx[sub]).to_numpy()
        ok = ~np.isnan(have)
        Yp = np.full((pb.n, sub.size), np.nan)
        cols = pt[have[ok].astype(int)]
        Xp = pb.read(cols) / 2.0
        pflip = pf[have[ok].astype(int)] != flip[sub][ok]        # orient to the cohort's counted allele
        Xp[:, pflip] = 1.0 - Xp[:, pflip]
        Yp[:, ok] = Xp
        pmu = np.nanmean(Yp, axis=0)
        Yp = np.where(np.isnan(Yp), np.where(np.isnan(pmu), mu[keep_sub][None, :].repeat(pb.n, 0), pmu[None, :]), Yp)
        Ys = np.vstack([Ys, Yp])
        e = np.zeros(K)
        e[-1] = 1.0
        fixed_rows = (np.arange(N, N + pb.n), e)
        log(f"[admixed-ld] derive {name}: {pb.n} panel members of {name} anchor the frequencies")
    Psub = Pk_c[sub]
    pN = np.clip(2.0 * Ys[:N].mean(axis=0) - Psub.mean(axis=1), 0.02, 0.98)
    if fixed_rows is not None:
        pN = np.clip(Ys[N:].mean(axis=0), 0.02, 0.98)
    theta = None
    for it in range(als_iter):
        P = np.column_stack([Psub, pN])
        th_new = _simplex_ls(Ys, P, fixed_rows)
        R = Ys - th_new[:, :-1] @ Psub.T
        w = th_new[:, -1]
        pN = np.clip((w @ R) / max(float(w @ w), 1e-12), 1e-3, 1 - 1e-3)
        change = np.inf if theta is None else float(np.abs(th_new - theta).max())
        theta = th_new
        if change < 1e-5:
            break
    th_c = theta[:N]
    log(f"[admixed-ld] derive {name}: ancestry after {it + 1} iterations; mean "
        + ", ".join(f"{s} {x:.3f}" for s, x in zip(known + [name], th_c.mean(axis=0))))

    # ------------------------------------------------------------------ 2. frequencies at every SNP, adjusted variances
    p_rank = C.shape[1]
    df_res = N - 1 - p_rank
    wN = th_c[:, -1]
    a_num, var_res = np.zeros(M), np.zeros(M)
    b_den = float(wN @ wN)
    for s0 in range(0, M, step):
        X = blocks.read(t_idx[s0:s0 + step])
        mu = np.nanmean(X, axis=0)
        center_impute(X)
        G2 = X / 2.0 + mu[None, :] / 2.0                         # imputed g / 2
        a_num[s0:s0 + step] = wN @ (G2 - th_c[:, :-1] @ Pk_c[s0:s0 + step].T)
        Bx = C.T @ X if p_rank else np.zeros((0, X.shape[1]))
        var_res[s0:s0 + step] = (np.einsum("ij,ij->j", X, X) - np.einsum("ij,ij->j", Bx, Bx)) / df_res
    pN_c = np.clip(a_num / max(b_den, 1e-12), 1e-3, 1 - 1e-3)
    n_eff = b_den
    if fixed_rows is not None:                                   # the panel adds its members at weight one
        Xp_all = np.full((pb.n, M), np.nan)
        hv = pos.reindex(r_idx).to_numpy()
        okp = ~np.isnan(hv)
        Xp = pb.read(pt[hv[okp].astype(int)]) / 2.0
        pflip = pf[hv[okp].astype(int)] != flip[okp]
        Xp[:, pflip] = 1.0 - Xp[:, pflip]
        Xp_all[:, okp] = Xp
        cnt = np.sum(~np.isnan(Xp_all), axis=0)
        pN_c = np.clip((a_num + np.nansum(Xp_all, axis=0)) / (b_den + cnt), 1e-3, 1 - 1e-3)
        n_eff = b_den + float(np.median(cnt))
        pb.close()
    pN_ref = np.where(flip, 1.0 - pN_c, pN_c)                     # in the reference's orientation
    vN = 2.0 * pN_ref * (1.0 - pN_ref)                           # dosage variance of the source (Hardy-Weinberg)
    poly = var_res > 1e-8                                        # SNPs without variance in the cohort tell nothing
    if not poly.all():
        t_idx, r_idx, flip, Pk, Pk_c = t_idx[poly], r_idx[poly], flip[poly], Pk[poly], Pk_c[poly]
        pN_ref, vN, var_res = pN_ref[poly], vN[poly], var_res[poly]
        M = int(poly.sum())

    # ------------------------------------------------------------------ 3. provisional reference and the curve
    prov = out + "_provisional"
    os.makedirs(prov, exist_ok=True)
    chrom = rv["CHR"].to_numpy()[r_idx]
    local = rv["LOCAL"].to_numpy()[r_idx]
    ki = [ref.source_index(s) for s in known]
    stats = {}
    for c in ref.chromosomes:
        sel = np.flatnonzero(chrom == c)
        if sel.size < 2:
            continue
        z = ref.load(c)
        lo = local[sel]
        arrays = dict(snp=z["snp"][lo], bp=z["bp"][lo], cm=z["cm"][lo], counted=z["counted"][lo],
                      other=z["other"][lo], indptr=np.zeros(sel.size + 1, np.int64), indices=np.zeros(0, np.int32))
        for n_, i in enumerate(ki):
            arrays[f"p_{n_}"], arrays[f"v_{n_}"] = z[f"p_{i}"][lo], z[f"v_{i}"][lo]
            arrays[f"cov_{n_}"] = arrays[f"m22_{n_}"] = np.zeros(0, np.float32)
        arrays[f"p_{K - 1}"], arrays[f"v_{K - 1}"] = pN_ref[sel], vN[sel]
        arrays[f"cov_{K - 1}"] = arrays[f"covvar_{K - 1}"] = np.zeros(0, np.float32)
        np.savez(os.path.join(prov, f"chr{c}.npz"), **arrays)
        stats[c] = dict(n_snps=int(sel.size), n_pairs=0, cm_length=float(z["cm"][lo].max() - z["cm"][lo].min()))
    src_meta = [dict(name=s, n=int(ref.n[s]), panel=next(x.get("panel", "") for x in ref.manifest["sources"]
                                                     if x["name"] == s)) for s in known]
    src_meta.append(dict(name=name, n=int(round(n_eff)), derived=True, cohort=os.path.basename(geno),
                         n_people=int(N), panel=None if panel is None else os.path.basename(panel)))
    write_manifest(prov, src_meta, window_kb=ref.manifest["window_kb"], genetic_map=ref.manifest["genetic_map"],
                   match=match, dtype="float32", counted_allele=ref.manifest["counted_allele"], stats=stats,
                   products=dict(native=False, repaired_columns=0))
    sources = known + [name]
    _, cinfo = genotype_curve(geno, prov, out + ".anchor", sources=sources, covar=covar, keep=keep, match=match,
                              cell_cm=cell_cm, threads=threads, native=native, log=log)
    from .model import KernelMatrix
    d_c = pd.read_csv(out + ".anchor.curve.tsv", sep="\t")

    # ------------------------------------------------------------------ 4. m and Kt_res(0) from the adjusted variances
    Vm = np.column_stack([rv[f"v_{s}"].to_numpy()[r_idx] for s in known] + [vN])
    Pm = np.column_stack([Pk, pN_ref])
    Dm = Pm[:, :-1] - Pm[:, -1:]
    nsrc = np.array([ref.n[s] for s in known] + [n_eff])
    pvar = Vm / (4.0 * nsrc[None, :])
    O = unbiased_products(Dm, pvar[:, :-1], pvar[:, -1])
    reg = regress_standard_errors_k(np.ones(M), 1.0 / np.sqrt(np.maximum(var_res, 1e-12)), np.zeros(M), Vm, O)
    m_vec, k0 = np.asarray(reg["m"]), np.asarray(reg["kappa0"])
    k1 = K - 1
    kc = np.zeros((len(d_c), k1, k1))
    from .model import contrast_pairs
    for a, b in contrast_pairs(k1):
        col = "kernel" if k1 == 1 else f"kernel_{sources[a]}_{sources[b]}"
        kc[:, a, b] = kc[:, b, a] = d_c[col].to_numpy()
    inter = np.atleast_2d(cinfo["inter_matrix"])
    lam = np.array([cinfo["fit_per_source"][s]["lam"] for s in sources])
    kern = KernelMatrix.from_curve(d_c.d_cm.to_numpy(), kc, inter, k0, lam)
    log(f"[admixed-ld] derive {name}: m " + ", ".join(f"{s} {x:.4f}" for s, x in zip(sources, m_vec))
        + " (adjusted variances, R2 " + f"{reg['fit_r2']:.4f}); decay " + ", ".join(f"{x:.4f}" for x in lam))

    # ------------------------------------------------------------------ 5-6. window covariances and the inversion
    mp = _products(threads, native)
    sign = np.where(flip, -1.0, 1.0)
    os.makedirs(out, exist_ok=True)
    stats = {}
    for c in ref.chromosomes:
        sel = np.flatnonzero(chrom == c)
        if sel.size < 2:
            continue
        z = ref.load(c)
        lo = local[sel]
        nloc = z["bp"].size
        pos_in = -np.ones(nloc, dtype=np.int64)
        pos_in[lo] = np.arange(sel.size)
        indptr, indices = z["indptr"], z["indices"].astype(np.int64)
        rows_l = np.repeat(np.arange(nloc), np.diff(indptr))
        j, k = pos_in[rows_l], pos_in[indices]
        use = (j >= 0) & (k >= 0)
        src = np.flatnonzero(use)
        j, k = j[use], k[use]                                    # pairs among matched SNPs, positions within sel
        cnt = np.bincount(j, minlength=sel.size)
        new_indptr = np.zeros(sel.size + 1, dtype=np.int64)
        np.cumsum(cnt, out=new_indptr[1:])
        # cohort covariances of adjusted genotypes for these pairs (rows j in blocks, partners up to the band end)
        cR = np.zeros(j.size)
        m22R = np.zeros(j.size)
        hi = np.zeros(sel.size, dtype=np.int64)
        np.maximum.at(hi, j, k + 1)
        hi = np.maximum(hi, np.arange(sel.size) + 1)
        hi = np.maximum.accumulate(hi)
        for b0 in range(0, sel.size, step):
            b1 = min(sel.size, b0 + step)
            e = int(hi[b1 - 1])
            X = blocks.read(t_idx[sel[b0:e]])
            center_impute(X)
            if p_rank:
                X -= C @ (C.T @ X)
            X *= sign[sel[b0:e]][None, :]                         # to the reference's orientation
            Gm = mp.tn(X[:, :b1 - b0], X) / df_res
            X2 = X * X
            Q = mp.tn(X2[:, :b1 - b0], X2) / N
            pr = np.flatnonzero((j >= b0) & (j < b1))
            cR[pr] = Gm[j[pr] - b0, k[pr] - b0]
            m22R[pr] = Q[j[pr] - b0, k[pr] - b0]
        varR = np.maximum(m22R - cR * cR, 0.0) / N
        cm_ = z["cm"]
        d = np.abs(cm_[lo[k]] - cm_[lo[j]])
        gs, ge = sel[j], sel[k]
        Kd = kern(d)
        adm = np.einsum("pa,pab,pb->p", Dm[gs], Kd, Dm[ge])
        num = cR - adm
        noise = varR.copy()
        for n_, i in enumerate(ki):
            pi_s = m_vec[n_] * np.exp(-lam[n_] * d)
            cs_ = z[f"cov_{i}"][src].astype(np.float64)
            num -= pi_s * cs_
            if f"covvar_{i}" in z.files:
                vs_ = z[f"covvar_{i}"][src].astype(np.float64)
            else:
                vr = z[f"v_{i}"]
                vs_ = cs_ * cs_ - unbiased_squared_covariance(cs_, z[f"m22_{i}"][src].astype(np.float64),
                                                              vr[lo[j]] * vr[lo[k]], ref.n[known[n_]])
            noise += pi_s * pi_s * vs_
        piN = m_vec[-1] * np.exp(-lam[-1] * d)
        arrays = dict(snp=z["snp"][lo], bp=z["bp"][lo], cm=cm_[lo], counted=z["counted"][lo], other=z["other"][lo],
                      indptr=new_indptr, indices=k.astype(np.int32))
        for n_, i in enumerate(ki):
            arrays[f"p_{n_}"], arrays[f"v_{n_}"] = z[f"p_{i}"][lo], z[f"v_{i}"][lo]
            arrays[f"cov_{n_}"] = z[f"cov_{i}"][src]
            if f"covvar_{i}" in z.files:
                arrays[f"covvar_{n_}"] = z[f"covvar_{i}"][src]
            else:
                arrays[f"m22_{n_}"] = z[f"m22_{i}"][src]
        arrays[f"p_{K - 1}"], arrays[f"v_{K - 1}"] = pN_ref[sel], vN[sel]
        arrays[f"cov_{K - 1}"] = (num / piN).astype(np.float32)
        arrays[f"covvar_{K - 1}"] = (noise / piN ** 2).astype(np.float32)
        np.savez(os.path.join(out, f"chr{c}.npz"), **arrays)
        stats[c] = dict(n_snps=int(sel.size), n_pairs=int(j.size), cm_length=float(cm_[lo].max() - cm_[lo].min()))
        log(f"[admixed-ld] derive {name} chr{c}: {sel.size} SNPs, {j.size} pairs ({time.perf_counter() - t0:.0f} s)")
    info = mp.drain()
    blocks.close()
    src_meta[-1].update(m_anchor=m_vec.tolist(), kappa0_anchor=k0.tolist(), decay_anchor=lam.tolist())
    manifest = write_manifest(out, src_meta, window_kb=ref.manifest["window_kb"], genetic_map=ref.manifest["genetic_map"],
                              match=match, dtype="float32", counted_allele=ref.manifest["counted_allele"], stats=stats,
                              products=dict(native=info["native"], repaired_columns=int(info["repaired_columns"])))
    shutil.rmtree(prov, ignore_errors=True)
    log(f"[admixed-ld] derive {name}: wrote {out} (effective panel size {n_eff:.0f}; {time.perf_counter() - t0:.0f} s)")
    return manifest
