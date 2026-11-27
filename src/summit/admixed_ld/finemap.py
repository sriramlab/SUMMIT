"""Fine-mapping with composed regional LD matrices: the predicted error of every entry of a matrix, and the residual
variance of the second SuSiE fit (`summit admixed-ld matrix --error-variance`, `summit admixed-ld finemap-variance`).

Summary-statistic fine-mapping models the z-scores of a region as z ~ N(sqrt(N) R b, R). An error E in the matrix adds
N E b b' E' to the residual covariance. For SNPs j and k, V_jk is the predicted variance of the error of the composed
R_jk: the delta-method variance under sampling of the panels, with the genotype moments under Hardy-Weinberg
equilibrium that follow from the panels' frequencies and correlations, plus the squared bias of the taper and the
sampling variance of the correlation in the GWAS sample itself, of order 1/N. V depends on the panels' sizes,
frequencies and correlations, the ancestry weights and the taper, never on in-sample LD.

With the composed covariance C_jk = sum_s w_s(d) c^s_jk + A_jk (w_s(d) = m_s exp(-lam_s d) T_s(d), T_s the taper),
C_jj = sum_s m_s v^s_j + A_jj and R_jk = C_jk / sqrt(C_jj C_kk), the weight of panel s's correlation r_s in R is
a_s = w_s sqrt(v^s_j v^s_k) / sqrt(C_jj C_kk) and source s's share of SNP j's variance is b_sj = m_s v^s_j / C_jj, so
    Var(R_jk) = sum_s (1 / n_s) [a_s^2 V_c - a_s R (b_sj C_j + b_sk C_k) + (R^2 / 4)(b_sj^2 V_j + b_sk^2 V_k
                                  + 2 b_sj b_sk C_jk)],
with n x the sampling (co)variances V_c, V_j, C_j, C_jk of the relative errors of a panel's genotype covariance and
variances under Hardy-Weinberg equilibrium (rel_moments). The taper's bias is (sum_s (1 - T_s) w0_s sqrt(v v) / S r_s)^2
with each squared panel correlation replaced by its unbiased estimate, and the GWAS sample's own term is
var_corr(p, R) / N at the frequencies p = sum_s m_s p_s. Errors of the frequencies and of the kernel in the admixture
term are of smaller order and left out.

Two SuSiE fits: the first with residual variance 1 gives the posterior second moments E(b_c^2 | z) on the z scale (b
with z = R b + e); the second uses the residual covariance s^2 (R + lambda I) with
    s^2 = 1 + mean_j sum_{c != j} V_jc E(b_c^2 | z),
the predicted variance of an entry of the mismatch E b. This is the arithmetic of the project's fine-mapping code
(code/aou_fm/s2/fm_s2_lib.py and fit_susie_s2.R), ported line for line.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def rel_moments(p, q, r):
    """n x the sampling (co)variances of the relative errors e_c = dc / sqrt(v_j v_k), e_j = dv_j / v_j, e_k of a panel's
    genotype covariance and variances, for SNPs of allele frequencies p, q and genotype correlation r, under
    Hardy-Weinberg equilibrium (a genotype is the sum of two independent haplotypes, so its cumulants are twice the
    haplotype pair's, and every moment follows from p, q and D = r sqrt(p(1-p)q(1-q))). Returns Vc, Vj, Vk, Cj, Ck, Cjk."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    sd = np.sqrt(p * (1 - p) * q * (1 - q))
    lo = np.maximum(-p * q, -(1 - p) * (1 - q))
    hi = np.minimum(p * (1 - q), (1 - p) * q)
    Dl = np.clip(r * sd, lo, hi)
    P = {(1, 1): p * q + Dl, (1, 0): p * (1 - q) - Dl, (0, 1): (1 - p) * q - Dl, (0, 0): (1 - p) * (1 - q) + Dl}
    U = {1: 1 - p, 0: -p}
    W = {1: 1 - q, 0: -q}
    mu = {}
    for ab in ((2, 0), (0, 2), (1, 1), (4, 0), (0, 4), (3, 1), (1, 3), (2, 2)):
        mu[ab] = sum(P[(x, y)] * U[x] ** ab[0] * W[y] ** ab[1] for x in (0, 1) for y in (0, 1))
    k = {(2, 0): mu[2, 0], (0, 2): mu[0, 2], (1, 1): mu[1, 1],
         (4, 0): mu[4, 0] - 3 * mu[2, 0] ** 2, (0, 4): mu[0, 4] - 3 * mu[0, 2] ** 2,
         (3, 1): mu[3, 1] - 3 * mu[2, 0] * mu[1, 1], (1, 3): mu[1, 3] - 3 * mu[0, 2] * mu[1, 1],
         (2, 2): mu[2, 2] - mu[2, 0] * mu[0, 2] - 2 * mu[1, 1] ** 2}
    M20, M02, M11 = 2 * k[2, 0], 2 * k[0, 2], 2 * k[1, 1]
    M40 = 2 * k[4, 0] + 3 * M20 ** 2
    M04 = 2 * k[0, 4] + 3 * M02 ** 2
    M31 = 2 * k[3, 1] + 3 * M20 * M11
    M13 = 2 * k[1, 3] + 3 * M02 * M11
    M22 = 2 * k[2, 2] + M20 * M02 + 2 * M11 ** 2
    rr = M11 / np.sqrt(M20 * M02)
    return (M22 / (M20 * M02) - rr ** 2, M40 / M20 ** 2 - 1, M04 / M02 ** 2 - 1,
            M31 / (M20 ** 1.5 * M02 ** 0.5) - rr, M13 / (M02 ** 1.5 * M20 ** 0.5) - rr, M22 / (M20 * M02) - 1)


def var_corr(p, R):
    """n x the sampling variance of a sample correlation matrix's entries (Hardy-Weinberg delta method); p the
    frequencies."""
    Vc, Vj, Vk, Cj, Ck, Cjk = rel_moments(p[:, None], p[None, :], R)
    return Vc - R * (Cj + Ck) + R ** 2 / 4 * (Vj + Vk + 2 * Cjk)


def predict_error_variance(C_s, M_s, v, p, m, w0, T, cd, R, n_s, N):
    """Predicted variance of the error of every entry of a composed matrix R (p x p).

    C_s, M_s: per source the panel covariances (variances on the diagonal) and the m22 moments of the region's SNPs
    (dense p x p); v, p: the panels' variances and frequencies (p x K); m: mean ancestry (K); w0: per source
    m_s exp(-lam_s d); T: per source the taper exp(-d / tau_s); cd: the composed variances C_jj; n_s: panel sizes; N:
    the GWAS sample size. Returns dict(V (diagonal 0), parts) with the parts v_panel_hap (Hardy-Weinberg panel
    sampling), bias2 (the taper's squared bias) and v_gwas_hap (the GWAS sample's own term), and v_panel and v_gwas,
    the same with Gaussian moments, for comparison."""
    from .compose import unbiased_squared_covariance
    P = R.shape[0]
    S = np.sqrt(np.clip(cd[:, None] * cd[None, :], 1e-24, None))
    Vg_tot = np.zeros((P, P))
    Vh_tot = np.zeros((P, P))
    bsum = np.zeros((P, P))
    bsq_corr = np.zeros((P, P))
    for i in range(len(n_s)):
        vi = v[:, i]
        vv = np.sqrt(np.clip(vi[:, None] * vi[None, :], 1e-24, None))
        rs = C_s[i] / vv
        a = w0[i] * T[i] * vv / S
        bj = m[i] * vi / cd
        bJ, bK = bj[:, None], bj[None, :]
        cross = -2 * a * R * rs * (bJ + bK) + R ** 2 * (bJ ** 2 + bK ** 2 + 2 * bJ * bK * rs ** 2) / 2
        Vg_tot += (a ** 2 * (1 + rs ** 2) + cross) / n_s[i]
        ps = p[:, i]
        Vc, Vj, Vk, Cj, Ck, Cjk = rel_moments(ps[:, None], ps[None, :], rs)
        Vh_tot += (a ** 2 * Vc - a * R * (bJ * Cj + bK * Ck)
                   + R ** 2 / 4 * (bJ ** 2 * Vj + bK ** 2 * Vk + 2 * bJ * bK * Cjk)) / n_s[i]
        beta = (1 - T[i]) * w0[i] * vv / S
        U = unbiased_squared_covariance(C_s[i], np.nan_to_num(M_s[i]), vv ** 2, n_s[i]) / vv ** 2
        bsum += beta * rs
        bsq_corr += beta ** 2 * (rs ** 2 - U)                     # the noisy squares replaced by their unbiased estimates
    bias2 = bsum ** 2 - bsq_corr
    pmix = p @ m
    Vgw, Vgw_h = (1 - R ** 2) ** 2 / N, var_corr(pmix, R) / N
    V = Vh_tot + bias2 + Vgw_h
    V = 0.5 * (V + V.T)                                  # exactly symmetric, as the matrix R is
    np.fill_diagonal(V, 0.0)
    return dict(V=V, parts=dict(v_panel=Vg_tot, v_panel_hap=Vh_tot, bias2=bias2, v_gwas=Vgw, v_gwas_hap=Vgw_h))


def residual_variance(V, eb2):
    """s^2 = 1 + mean_j sum_{c != j} V_jc E(b_c^2) with E(b^2) the first fit's posterior second moments on the z scale."""
    V = np.asarray(V, np.float64)
    V = 0.5 * (V + V.T)
    np.fill_diagonal(V, 0.0)
    return float(1.0 + np.mean(V @ np.asarray(eb2, np.float64)))


def finemap_variance(matrix_prefix, posterior, out=None, *, snp_col="SNP", eb2_col="EB2", log=print):
    """The residual variance of the second SuSiE fit for a region written by `matrix --error-variance`: posterior is
    a table with the first fit's posterior second moments E(b^2 | z) on the z scale per SNP (susieR:
    colSums(alpha * mu2) / X_column_scale_factors^2 for a fit of z on R). Writes out (default MATRIX.s2.json)."""
    z = np.load(f"{matrix_prefix}.ld_var.npz", allow_pickle=False)
    snps = pd.read_csv(f"{matrix_prefix}.snps.tsv", sep="\t", dtype={"SNP": str}).SNP.to_numpy()
    post = pd.read_csv(posterior, sep=r"\s+", dtype={snp_col: str})
    if snp_col not in post or eb2_col not in post:
        raise ValueError(f"{posterior} needs the columns {snp_col} and {eb2_col}.")
    eb2 = post.set_index(snp_col)[eb2_col].reindex(snps)
    if eb2.isna().any():
        raise ValueError(f"{int(eb2.isna().sum())} of the matrix's {snps.size} SNPs are missing from {posterior}; give "
                         "the posterior of a fit on exactly the matrix's SNPs, in any order.")
    s2 = residual_variance(z["V"], eb2.to_numpy())
    rec = dict(format="summit-admixed-ld-finemap-variance", version=1, matrix=matrix_prefix, posterior=posterior,
               n_snps=int(snps.size), s2=s2, gwas_n=float(z["n"]),
               definition="s^2 = 1 + mean_j sum_{c != j} V_jc E(b_c^2 | z) (z scale); fit SuSiE again with residual "
                          "variance s^2 (covariance s^2 (R + lambda I), lambda as in the first fit)")
    out = out or f"{matrix_prefix}.s2.json"
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    log(f"[admixed-ld] finemap-variance: s^2 = {s2:.6g} for the {snps.size} SNPs of {matrix_prefix}; wrote {out}")
    return rec
