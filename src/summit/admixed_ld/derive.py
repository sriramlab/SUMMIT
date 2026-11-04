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
from .model import contrast_noise, full_from_contrasts, regress_standard_errors_k, unbiased_products_cov
from .reference import Reference, _products, write_manifest


def _simplex_ls(Y, P, fixed=None, YP=None):
    """Per-row least squares of Y (n x S) on the columns of P (S x K) with theta >= 0 and sum(theta) = 1 (active set:
    negative coordinates are dropped and the rest refitted, at most K times). fixed: rows with given theta. Works from
    the normal equations, so only Y P (n x K, computed once, or passed as YP) touches the data."""
    n, K = Y.shape[0], P.shape[1]
    YP = Y @ P if YP is None else YP
    PP = P.T @ P
    active = np.ones((n, K), dtype=bool)
    theta = np.zeros((n, K))
    for _ in range(K):
        for pattern in np.unique(active, axis=0):
            rows = np.flatnonzero((active == pattern).all(axis=1))
            idx = np.flatnonzero(pattern)
            if idx.size == 1:
                theta[np.ix_(rows, idx)] = 1.0
                continue
            a, last = idx[:-1], idx[-1]                          # substitute the last active coordinate
            # A = P_a - P_last; A'A and (Y - P_last) A from P'P and Y P
            AtA = PP[np.ix_(a, a)] - PP[a, last][:, None] - PP[last, a][None, :] + PP[last, last]
            rhs = (YP[np.ix_(rows, a)] - YP[rows, last][:, None]) - (PP[last, a] - PP[last, last])[None, :]
            sol = np.linalg.solve(AtA, rhs.T).T
            t = np.zeros((rows.size, K))
            t[:, a] = sol
            t[:, last] = 1.0 - sol.sum(axis=1)
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


def _als(Y, Pk, fixed, p0, *, iters, tol=1e-9):
    """Alternating least squares for E(Y) = theta [Pk, p] with theta on the simplex (rows `fixed` held at their given
    theta) and p the missing source's frequencies. Stops when the relative decrease of the residual sum of squares falls
    below tol. Each iteration reads Y twice (Y P and w' Y). Returns theta, p, iterations, the objective trace."""
    p = p0.copy()
    trace = []
    theta = None
    yy = float(np.einsum("ij,ij->", Y, Y))
    YPk = Y @ Pk                                                 # fixed part of Y P
    for it in range(iters):
        YP = np.column_stack([YPk, Y @ p])
        P = np.column_stack([Pk, p])
        theta = _simplex_ls(Y, P, fixed, YP=YP)
        w = theta[:, -1]
        wY = w @ Y
        p = np.clip((wY - (w @ theta[:, :-1]) @ Pk.T) / max(float(w @ w), 1e-12), 1e-3, 1 - 1e-3)
        # residual sum of squares at (theta, p): |Y|^2 - 2 sum_i theta_i' (Y P)_i + sum_i theta_i' P'P theta_i
        P = np.column_stack([Pk, p])
        YP = np.column_stack([YPk, Y @ p])
        rss = yy - 2.0 * float(np.einsum("ik,ik->", theta, YP)) + float(np.einsum("ik,kl,il->", theta, P.T @ P, theta))
        trace.append(rss)
        if it > 0 and trace[-2] - trace[-1] < tol * trace[-1]:
            break
    return theta, p, it + 1, trace


def ridge_calibration(f, n_anchor, p_der, var0, c, Pk, Vk, nk, *, blocks=50):
    """Position of a derived source on the family of sources that explain an admixed cohort equally well.

    The cohort identifies the missing source N only up to p' = (p^N - sum_s b_s p^s) / alpha with alpha + sum b_s = 1
    (ancestry moves between N and the known sources accordingly). Anchor members of N have expected frequencies
    p^N = alpha p' + sum_s b_s p^s, so regressing their frequencies f on p' and the known panels' frequencies over SNPs,
    with the coefficients summing to one, gives alpha and b. The sampling (co)variances of every input are removed from
    the normal equations (errors in variables): the panels' (Vk / 4 nk), the anchors' (f(1 - f) / 2 n_anchor), and
    those of p', which carries -c_s times each panel's error plus its own error of variance var0.
    f, p_der, var0: M; c: K-1; Pk, Vk: M x (K-1); nk: K-1. Returns dict(alpha, b, se_alpha, se_b) (jackknife over
    `blocks` contiguous groups of SNPs)."""
    f, p_der, var0 = (np.asarray(x, np.float64) for x in (f, p_der, var0))
    Pk, Vk = np.asarray(Pk, np.float64), np.asarray(Vk, np.float64)
    M, k = Pk.shape
    r = k - 1                                                    # the last known source is the reference column
    y = f - Pk[:, r]
    X = np.column_stack([p_der - Pk[:, r]] + [Pk[:, s] - Pk[:, r] for s in range(k - 1)])
    # base errors: panels 0..k-1, the derived part, the anchors
    var = np.column_stack([Vk / (4.0 * np.asarray(nk, np.float64)[None, :]), var0,
                           f * (1 - f) / (2.0 * n_anchor) * (2.0 * n_anchor / max(2.0 * n_anchor - 1, 1.0))])
    T = k + 2
    hy = np.zeros(T); hy[k + 1] = 1.0; hy[r] -= 1.0
    HX = np.zeros((X.shape[1], T))
    HX[0, :k] = -np.asarray(c, np.float64); HX[0, k] = 1.0; HX[0, r] -= 1.0
    for s in range(k - 1):
        HX[1 + s, s] += 1.0; HX[1 + s, r] -= 1.0
    edges = np.linspace(0, M, blocks + 1).astype(int)
    A = np.zeros((blocks, X.shape[1], X.shape[1])); bb = np.zeros((blocks, X.shape[1]))
    for g in range(blocks):
        sl = slice(edges[g], edges[g + 1])
        Xs, ys, vs = X[sl], y[sl], var[sl]
        A[g] = Xs.T @ Xs - np.einsum("at,jt,bt->ab", HX, vs, HX)
        bb[g] = Xs.T @ ys - np.einsum("at,jt,t->a", HX, vs, hy)
    sol = np.linalg.solve(A.sum(0), bb.sum(0))
    jk = np.array([np.linalg.solve(A.sum(0) - A[g], bb.sum(0) - bb[g]) for g in range(blocks)])
    se = np.sqrt((blocks - 1) / blocks * ((jk - jk.mean(0)) ** 2).sum(0))
    alpha = float(sol[0])
    b = np.zeros(k)
    b[: k - 1] = sol[1:]
    b[r] = 1.0 - alpha - sol[1:].sum()
    # the reference column's coefficient follows from the others; its jackknife SE likewise
    b_jk = np.column_stack([jk[:, 1:], 1.0 - jk[:, 0] - jk[:, 1:].sum(axis=1)])
    se_b = np.sqrt((blocks - 1) / blocks * ((b_jk - b_jk.mean(0)) ** 2).sum(0))
    return dict(alpha=alpha, b=b.tolist(), se_alpha=float(se[0]), se_b=se_b.tolist())


def _panel_band_moments(pb, pt, pf, hv, j, k, v_fallback, chunk=20_000_000):
    """Window moments of a small panel of the derived source on the band pairs (j, k) of one chromosome: covariance
    (n - 1 denominator), mean of x_j^2 x_k^2 and each SNP's variance, as `reference` writes them. hv: the panel row of
    each SNP (NaN if the panel lacks it); pt, pf: the panel's columns and flips against the reference. Pairs with a SNP
    missing from the panel get moments whose exact square is zero."""
    n = pb.n
    ok = ~np.isnan(hv)
    X = np.zeros((n, hv.size))
    if ok.any():
        rows = hv[ok].astype(int)
        Xr = pb.read(pt[rows])
        center_impute(Xr)
        Xr *= np.where(pf[rows], -1.0, 1.0)[None, :]               # to the reference's counted allele
        X[:, ok] = Xr
    vld = np.where(ok, np.einsum("ij,ij->j", X, X) / (n - 1), v_fallback)
    cov = np.zeros(j.size)
    m22 = np.zeros(j.size)
    X2 = X * X
    step = max(1, chunk // n)
    for s0 in range(0, j.size, step):
        jj, kk = j[s0:s0 + step], k[s0:s0 + step]
        cov[s0:s0 + step] = np.einsum("ip,ip->p", X[:, jj], X[:, kk]) / (n - 1)
        m22[s0:s0 + step] = np.einsum("ip,ip->p", X2[:, jj], X2[:, kk]) / n
    miss = ~(ok[j] & ok[k])
    m22[miss] = (n - 1) * vld[j[miss]] * vld[k[miss]] / n ** 2
    cov[miss] = 0.0
    return cov.astype(np.float32), m22.astype(np.float32), vld, int((~ok).sum())


def _inversion_parameters(geno, out, ref, known, name, src_meta, chrom, local, ki, pN_ref, vN, Pk, rv, r_idx, var_res,
                          M, K, c_mix, n_freq, *, covar, keep, match, cell_cm, threads, native, log):
    """Steps 3-4 of the inversion: the cohort's curve on a provisional reference (frequencies only), and m and
    Kt_res(0) from the regression of its adjusted variances. Returns m, Kt_res(0), decays, the kernel."""
    from .curve import genotype_curve
    prov = out + "_provisional"
    os.makedirs(prov, exist_ok=True)
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
    G = np.eye(K)
    G[K - 1, :K - 1] = -np.asarray(c_mix)                       # the derived source carries the panels' errors
    S_noise, _ = contrast_noise(Vm, np.array([ref.n_freq[s] for s in known] + [n_freq]), G)
    O = unbiased_products_cov(Dm, S_noise)
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

    return m_vec, k0, lam, kern


def derive_source(geno, ref_path, out, *, name, known=None, panel=None, covar=None, keep=None, match="position",
                  als_snps=20000, als_iter=500, als_tol=1e-9, calibrate=True, ld="inversion", cell_cm=0.25, step=1024,
                  threads=1, native=True, log=print):
    """Write a reference with the known sources of `ref_path` and `name` derived from the cohort `geno`.

    ld: "inversion" (window LD from the cohort's covariances, steps 3-6) or "panel" (window LD of the anchor panel, with
    its own moments for the exact squares; the frequencies still come from the cohort and the panel together)."""
    from .curve import _cohort_samples
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
    if fixed_rows is not None:
        p0 = np.clip(Ys[N:].mean(axis=0), 0.02, 0.98)
    else:
        p0 = np.clip(2.0 * Ys[:N].mean(axis=0) - Psub.mean(axis=1), 0.02, 0.98)
    theta, _, n_it, trace = _als(Ys, Psub, fixed_rows, p0, iters=als_iter, tol=als_tol)
    th_c = theta[:N].copy()
    log(f"[admixed-ld] derive {name}: ancestry after {n_it} iterations (residual sum of squares {trace[0]:.6g} -> "
        f"{trace[-1]:.6g}); mean " + ", ".join(f"{s} {x:.4f}" for s, x in zip(known + [name], th_c.mean(axis=0))))
    Vk_c = np.column_stack([rv[f"v_{s}"].to_numpy()[r_idx] for s in known])
    nk = np.array([ref.n_freq[s] for s in known], np.float64)
    calib = None
    if fixed_rows is not None and calibrate:
        # the cohort fixes the source only up to an affine family; the anchors fix its position (ridge_calibration),
        # from cohort-only frequencies on the ALS SNPs so that their errors are independent of the anchors'
        w = th_c[:, -1]
        ww = float(w @ w)
        R = Ys[:N] - th_c[:, :-1] @ Psub.T
        p_coh = (w @ R) / ww
        e_ = R - w[:, None] * p_coh[None, :]
        var0 = ((w * w) @ (e_ * e_)) / ww ** 2
        c_sub = (w @ th_c[:, :-1]) / ww
        f = Ys[N:].mean(axis=0)
        calib = ridge_calibration(f, pb.n, p_coh, var0, c_sub, Psub, Vk_c[sub], nk)
        alpha, b = calib["alpha"], np.asarray(calib["b"])
        tN = th_c[:, -1] / alpha                                 # theta_N = theta'_N / alpha, theta_s = theta'_s - b_s theta_N
        th_c = np.column_stack([th_c[:, :-1] - b[None, :] * tN[:, None], tN])
        calib["clipped_people"] = int((th_c < 0).any(axis=1).sum())
        th_c = np.clip(th_c, 0.0, None)
        th_c /= th_c.sum(axis=1, keepdims=True)
        log(f"[admixed-ld] derive {name}: anchors place the source at alpha {alpha:.4f} (SE {calib['se_alpha']:.4f}), "
            f"b " + ", ".join(f"{s} {x:.4f}" for s, x in zip(known, b)) + "; calibrated mean ancestry "
            + ", ".join(f"{s} {x:.4f}" for s, x in zip(known + [name], th_c.mean(axis=0))))
    elif fixed_rows is None:
        log(f"[admixed-ld] derive {name}: no anchor panel, so the source is identified only up to an affine family "
            "(its frequencies and mean ancestry are on the scale the fit settled on). LD scores composed with it are "
            "valid when every term uses this source consistently, not when it is combined with LD on the true scale.")

    # ------------------------------------------------------------------ 2. frequencies at every SNP, adjusted variances
    p_rank = C.shape[1]
    df_res = N - 1 - p_rank
    wN = th_c[:, -1]
    a_num, var_res, s_r2, s_w3r = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    b_den = float(wN @ wN)
    w2, w3, s4 = wN ** 2, wN ** 3, float(np.sum(wN ** 4))
    for s0 in range(0, M, step):
        X = blocks.read(t_idx[s0:s0 + step])
        mu = np.nanmean(X, axis=0)
        center_impute(X)
        G2 = X / 2.0 + mu[None, :] / 2.0                         # imputed g / 2
        r_ = G2 - th_c[:, :-1] @ Pk_c[s0:s0 + step].T
        a_num[s0:s0 + step] = wN @ r_
        s_r2[s0:s0 + step] = w2 @ (r_ * r_)
        s_w3r[s0:s0 + step] = w3 @ r_
        Bx = C.T @ X if p_rank else np.zeros((0, X.shape[1]))
        var_res[s0:s0 + step] = (np.einsum("ij,ij->j", X, X) - np.einsum("ij,ij->j", Bx, Bx)) / df_res
    ysum, ysq, cnt = np.zeros(M), np.zeros(M), np.zeros(M)
    if fixed_rows is not None:                                   # the panel adds its members at weight one
        Xp_all = np.full((pb.n, M), np.nan)
        hv = pos.reindex(r_idx).to_numpy()
        okp = ~np.isnan(hv)
        Xp = pb.read(pt[hv[okp].astype(int)]) / 2.0
        pflip = pf[hv[okp].astype(int)] != flip[okp]
        Xp[:, pflip] = 1.0 - Xp[:, pflip]
        Xp_all[:, okp] = Xp
        cnt = np.sum(~np.isnan(Xp_all), axis=0).astype(np.float64)
        ysum, ysq = np.nansum(Xp_all, axis=0), np.nansum(Xp_all ** 2, axis=0)
    den = b_den + cnt
    p_hat = (a_num + ysum) / den
    # sampling variance of p_hat from the people's residuals (sandwich): cohort sum w^2 (r - w p)^2 and anchors (y - p)^2
    var_p = (s_r2 - 2 * p_hat * s_w3r + p_hat ** 2 * s4 + ysq - 2 * p_hat * ysum + cnt * p_hat ** 2) / den ** 2
    pN_c = np.clip(p_hat, 1e-3, 1 - 1e-3)
    n_eff = float(np.median(den))
    with np.errstate(divide="ignore", invalid="ignore"):
        n0_j = 2.0 * pN_c * (1.0 - pN_c) / (4.0 * var_p)
    n_freq = float(np.nanmedian(n0_j[np.isfinite(n0_j) & (n0_j > 0)]))
    # its error carries -c_s times each known panel's error (the cohort's frequencies subtract their contributions)
    c_mix = (wN @ th_c[:, :-1]) / n_eff
    log(f"[admixed-ld] derive {name}: frequencies at {M} SNPs; effective sample size {n_freq:.0f} from the residuals "
        f"(sum of squared ancestry weights {n_eff:.0f}); errors carry " + ", ".join(
            f"{-x:.3f} x the {s} panel's" for s, x in zip(known, c_mix)))
    pN_ref = np.where(flip, 1.0 - pN_c, pN_c)                     # in the reference's orientation
    vN = 2.0 * pN_ref * (1.0 - pN_ref)                           # dosage variance of the source (Hardy-Weinberg)
    poly = var_res > 1e-8                                        # SNPs without variance in the cohort tell nothing
    if not poly.all():
        t_idx, r_idx, flip, Pk, Pk_c = t_idx[poly], r_idx[poly], flip[poly], Pk[poly], Pk_c[poly]
        pN_ref, vN, var_res = pN_ref[poly], vN[poly], var_res[poly]
        M = int(poly.sum())

    if ld not in ("inversion", "panel"):
        raise ValueError("ld must be 'inversion' or 'panel'.")
    if ld == "panel" and panel is None:
        raise ValueError("ld='panel' needs the anchor panel (--panel).")
    src_meta = [dict(name=s, n=int(ref.n[s]), panel=next(x.get("panel", "") for x in ref.manifest["sources"]
                                                     if x["name"] == s)) for s in known]
    src_meta.append(dict(name=name, n=int(pb.n) if ld == "panel" else int(round(n_eff)), n_freq=n_freq,
                         n_freq_weights=n_eff, noise_mix={s: float(-x) for s, x in zip(known, c_mix)}, derived=True,
                         ld=ld, cohort=os.path.basename(geno), n_people=int(N),
                         panel=None if panel is None else os.path.basename(panel), calibration=calib,
                         als=dict(iterations=int(n_it), snps=int(sub.size), rss_first=trace[0], rss_last=trace[-1])))
    chrom = rv["CHR"].to_numpy()[r_idx]
    local = rv["LOCAL"].to_numpy()[r_idx]
    ki = [ref.source_index(s) for s in known]
    if ld == "inversion":
        m_vec, k0, lam, kern = _inversion_parameters(
            geno, out, ref, known, name, src_meta, chrom, local, ki, pN_ref, vN, Pk, rv, r_idx, var_res, M, K, c_mix,
            n_freq, covar=covar, keep=keep, match=match, cell_cm=cell_cm, threads=threads, native=native, log=log)
        Dm = np.column_stack([Pk, pN_ref])
        Dm = Dm[:, :-1] - Dm[:, -1:]
    else:
        hv_all = pos.reindex(r_idx).to_numpy()

    # ------------------------------------------------------------------ 5-6. window covariances
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
        cm_ = z["cm"]
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
        if ld == "panel":
            cov_, m22_, vld_, n_miss = _panel_band_moments(pb, pt, pf, hv_all[sel], j, k, vN[sel])
            arrays[f"cov_{K - 1}"], arrays[f"m22_{K - 1}"], arrays[f"vld_{K - 1}"] = cov_, m22_, vld_
            stats[c] = dict(n_snps=int(sel.size), n_pairs=int(j.size), cm_length=float(cm_[lo].max() - cm_[lo].min()))
            log(f"[admixed-ld] derive {name} chr{c}: {sel.size} SNPs, {j.size} pairs, LD of the {pb.n} anchors"
                + (f" ({n_miss} SNPs missing from the panel: no LD of {name} there)" if n_miss else "")
                + f" ({time.perf_counter() - t0:.0f} s)")
            np.savez(os.path.join(out, f"chr{c}.npz"), **arrays)
            continue
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
        arrays[f"cov_{K - 1}"] = (num / piN).astype(np.float32)
        arrays[f"covvar_{K - 1}"] = (noise / piN ** 2).astype(np.float32)
        np.savez(os.path.join(out, f"chr{c}.npz"), **arrays)
        stats[c] = dict(n_snps=int(sel.size), n_pairs=int(j.size), cm_length=float(cm_[lo].max() - cm_[lo].min()))
        log(f"[admixed-ld] derive {name} chr{c}: {sel.size} SNPs, {j.size} pairs ({time.perf_counter() - t0:.0f} s)")
    info = mp.drain()
    blocks.close()
    if ld == "inversion":
        src_meta[-1].update(m_anchor=m_vec.tolist(), kappa0_anchor=np.asarray(k0).tolist(), decay_anchor=lam.tolist())
    if fixed_rows is not None:
        pb.close()
    manifest = write_manifest(out, src_meta, window_kb=ref.manifest["window_kb"], genetic_map=ref.manifest["genetic_map"],
                              match=match, dtype="float32", counted_allele=ref.manifest["counted_allele"], stats=stats,
                              products=dict(native=info["native"], repaired_columns=int(info["repaired_columns"])))
    shutil.rmtree(out + "_provisional", ignore_errors=True)
    log(f"[admixed-ld] derive {name}: wrote {out} (frequencies: effective sample size {n_freq:.0f}; window LD: "
        f"{'the ' + str(pb.n) + ' anchors' if ld == 'panel' else 'inversion of the cohort'}; "
        f"{time.perf_counter() - t0:.0f} s)")
    return manifest
