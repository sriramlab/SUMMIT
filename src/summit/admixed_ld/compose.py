"""Composed LD scores of an admixed GWAS sample (`summit admixed-ld compose`).

The population value of SUMMIT's V-weighted LD score of SNP j in class c is
    l_jc = sum_{k in c} C_res(j,k)^2 / (C_res(j,j) s_k^2)
         + (1/nu) sum_{k in c} Cov_theta(h_j, h_k) / (C_res(j,j) s_k^2),        h_k(theta) = 2 p_k(theta)(1 - p_k(theta)),
the second sum being the part of the in-sample score from each person's pairing with themselves, which SUMMIT's
noise correction leaves when heterozygosity follows ancestry. With s_k^2 = C(k,k) and C_res the covariance after the
covariates, it is computed as:
  pairs within the reference window   pi_A(d) C^A + pi_E(d) C^E + delta_j delta_k kappa(d), squared, with each squared
                                      panel covariance replaced by its unbiased estimate (order-4 U-statistic) and
                                      the sampling covariance of a source derived by inversion with the known panels
                                      it was computed from removed once (Reference.covariance_noise_mix); with the
                                      cohort's background at every distance, also less each panel's own background,
                                      pi_s(d)^2 v_sj v_sk S_s(d) / (n_s (n_s - 3)) (SmoothBackground);
  every pair on the same chromosome   delta_j^2 delta_k^2 kappa(d)^2, by FFT convolution on a grid of `grid_cm`
                                      (the window pairs' admixture-only part is taken out of the first sum);
  pairs on different chromosomes      delta_j^2 delta_k^2 far^2;
  the same-person term, each SNP with itself (V_j), and, optionally, relatives: with the curve's profile
                                      (f / nu^2) [B sum_{k in c, k != j} V_k + sum_{k in c on j's chromosome}
                                      (phi(d_jk) - B) V_k] (RelatednessProfile, BackgroundProfile, or the cohort's
                                      all-pairs background at every distance, SmoothBackground), or the uniform
                                      f (sum over their ordered pairs of g^2) / nu^2 * sum_{k in c} V_k, with g the
                                      relatedness of the covariate-adjusted, standardized genotypes. B, phi and the
                                      sum of g^2 are sums over the ordered pairs of the curve's n people; a GWAS of N
                                      of them taken at random holds the share f = N (N - 1) / (n (n - 1)) of those
                                      pairs (f = 1 when the curve is made on the GWAS's people).
V_j = C_res(j,j) / s_j^2 is the population share of SNP j's variance that the covariates leave, which SUMMIT's
in-sample V (one minus the adjusted R2) estimates, so neither V nor the partner weights need a degrees-of-freedom
factor. The output is written with SUMMIT's own LD-score writer, so `summit --h2/--rg --ldscores <out>.gw.ldscore.gz
--annot ...` read it unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd

from .. import utils
from ._io import check_match, match_variants
from .model import (KernelMatrix, ancestry_sd_from_kernel, check_kernel_join, contrast_pairs, dirichlet_moments,
                    moments_tensors, parameters_from_sumstats_k, contrast_noise, same_person_term,
                    unbiased_products_cov)
from .reference import Reference


def _sumstats_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.endswith((".tsv", ".tsv.gz", ".txt",
                                                                                     ".txt.gz", ".gz")))
        else:
            files.append(p)
    if not files:
        raise ValueError("No summary-statistic files found.")
    return files


def _target_variants(snps_path, first_sumstats):
    """CHR, BP, SNP, COUNTED, OTHER of the GWAS variants, from --snps (BIM, PVAR or a table with these columns) or
    from the summary statistics (SNP, CHR, BP, A1, A2)."""
    from ._io import read_bim
    if snps_path is not None:
        if snps_path.endswith(".bim"):
            v = read_bim(snps_path)
            return v.assign(COUNTED=v.A1.str.upper(), OTHER=v.A2.str.upper())
        if snps_path.endswith(".pvar"):
            from ..ldscore.genotype_source import read_pvar_variants
            v = read_pvar_variants(snps_path)
            return v.assign(COUNTED=v.A2.str.upper(), OTHER=v.A1.str.upper())
        df = pd.read_csv(snps_path, sep=r"\s+", dtype=str)
    else:
        df = pd.read_csv(first_sumstats, sep=r"\s+", dtype=str, compression="infer")
    cols = {c.upper().lstrip("#"): c for c in df.columns}
    pick = {k: next((cols[x] for x in alts if x in cols), None)
            for k, alts in dict(SNP=("SNP", "ID"), CHR=("CHR", "CHROM"), BP=("BP", "POS"), A1=("A1", "ALT"),
                                A2=("A2", "REF")).items()}
    if pick["SNP"] is None or pick["A1"] is None or pick["A2"] is None:
        raise ValueError("The variant table needs SNP, A1 and A2 columns.")
    out = pd.DataFrame({"SNP": df[pick["SNP"]].astype(str), "COUNTED": df[pick["A1"]].str.upper(),
                        "OTHER": df[pick["A2"]].str.upper()})
    if pick["CHR"] is not None and pick["BP"] is not None:
        out["CHR"] = df[pick["CHR"]].astype(str).str.replace(r"^(?i:chr)", "", regex=True)
        out["BP"] = pd.to_numeric(df[pick["BP"]], errors="coerce")
    return out


def _track_sumstats(path):
    """Whether a summary-statistic file is one local-ancestry track's (`tractor-sumstats` writes a TRACK column)."""
    head = pd.read_csv(path, sep=r"\s+", nrows=0, compression="infer")
    return "TRACK" in head.columns


def unbiased_squared_covariance(c, m22, vv, n):
    """Unbiased estimate of the squared covariance of two variants from a panel of n people (the order-4 U-statistic):
    c the sample covariance (n - 1 denominator), m22 the mean of x_j^2 x_k^2 over people (centred dosages), vv the
    product of the two sample variances (n - 1). Exact for any genotype distribution, so small panels are not biased
    (the first-order correction c^2 - (m22 - c^2)/n leaves a bias of about 4/n of the noise)."""
    n = float(n)
    return ((n - 1.0) ** 2 * (n - 2.0) * c * c - n * n * m22 + (n - 1.0) * vv) / (n * (n - 2.0) * (n - 3.0))


def panel_variance_scale(ref, i, n_panel, bins=20):
    """Scale factors that put the window covariances of a derived source whose LD comes from a panel (stored with the
    panel's own variances, vld) on the source's scale, for the LD scores.

    The source's dosage variance v_j comes from its derived frequencies; the panel's population has variance vP_j,
    which the panel's sample variance vld_j estimates with sampling variance about vld_j (1 - vld_j) / n (the dosage's
    fourth moment under Hardy-Weinberg is 2pq). The panel's covariance c_jk enters as c_jk f_j f_k with
    f_j = sqrt(v_j / vP~_j), the panel's correlation on the source's variances, vP~_j being the empirical-Bayes estimate
    of vP_j: in bins of the source's minor allele frequency, r_j = vld_j - v_j has mean mu_b and, beyond the sampling
    variances of vld_j and v_j, between-SNP variance tau_b^2 (method of moments), and
    vP~_j = v_j + mu_b + B_j (r_j - mu_b), B_j = tau_b^2 / (tau_b^2 + s_j^2).
    A panel drawn from the source has tau_b ~ 0, and its covariances are used as they are (f ~ 1): dividing by the
    noisy variances of a small panel would bias the squared correlations down (about 34% for 34 people). A panel of
    another population has tau_b^2 >> s_j^2, and its correlations are transferred (B ~ 1): its covariances, which
    carry that population's variances, would bias the squared correlations up (about 50% in the coalescent test of
    7 Nov 2026, source-panel F_ST 0.024). Returns ({chromosome: f over the chromosome file's SNPs}, summary)."""
    per, rs, s2s, mafs = {}, [], [], []
    for c in ref.chromosomes:
        z = ref.load(c)
        v = z[f"v_{i}"].astype(np.float64)
        vld = z[f"vld_{i}"].astype(np.float64)
        p = z[f"p_{i}"].astype(np.float64)
        fv = z[f"fvar_{i}"].astype(np.float64) if f"fvar_{i}" in z.files else np.zeros_like(v)
        # SNPs missing from the panel carry vld = v (no LD there); they inform nothing and keep f = 1
        present = (vld != v) & np.isfinite(vld) & np.isfinite(v) & (vld > 0) & (v > 0)
        s2 = np.clip(vld, 0, 1) * (1 - np.clip(vld, 0, 1)) / float(n_panel) + (2 * (1 - 2 * p)) ** 2 * fv
        maf = np.minimum(np.clip(p, 0, 1), 1 - np.clip(p, 0, 1))
        per[c] = (v, vld, s2, maf, present)
        rs.append((vld - v)[present])
        s2s.append(s2[present])
        mafs.append(maf[present])
    r, s2, maf = np.concatenate(rs), np.concatenate(s2s), np.concatenate(mafs)
    if r.size < 10 * bins:
        bins = max(1, r.size // 10)
    edges = np.unique(np.quantile(maf, np.linspace(0, 1, bins + 1))) if r.size else np.array([0.0, 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    b_of = lambda x: np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    nb = edges.size - 1
    bi = b_of(maf)
    cnt = np.maximum(np.bincount(bi, minlength=nb), 1)
    mu = np.bincount(bi, weights=r, minlength=nb) / cnt
    tau2 = np.maximum(np.bincount(bi, weights=(r - mu[bi]) ** 2, minlength=nb) / cnt
                      - np.bincount(bi, weights=s2, minlength=nb) / cnt, 0.0)
    out, Bs, f2s = {}, [], []
    for c, (v, vld, s2c, mafc, present) in per.items():
        b = b_of(mafc)
        B = tau2[b] / (tau2[b] + s2c)
        vP = np.maximum(v + mu[b] + B * (vld - v - mu[b]), 1e-6)
        f = np.where(present, np.sqrt(np.maximum(v, 0) / vP), 1.0)
        out[c] = f
        Bs.append(B[present])
        f2s.append((f * f)[present])
    Bs, f2s = np.concatenate(Bs), np.concatenate(f2s)
    info = dict(n_panel=int(n_panel), snps=int(r.size), bins=int(nb),
                shrinkage_weight_median=float(np.median(Bs)) if Bs.size else None,
                shrinkage_weight_mean=float(np.mean(Bs)) if Bs.size else None,
                tau2_mean=float(np.mean(tau2)), mean_shift=float(np.mean(mu)),
                mean_f2=float(np.mean(f2s)) if f2s.size else None)
    return out, info


def window_noise(pis, vns, cross, d):
    """Sampling variance of a composed window covariance sum_s pi_s(d) c_s: sum_s pi_s^2 Var(c_s), plus twice the
    covariance of the errors of a source n derived by inversion and of each known panel t it was computed from,
    pi_n pi_t Cov(e_n, e_t) = -pi_n pi_t (pi^a_t / pi^a_n) Var(e_t), with the anchor cohort's pi^a(d) = m^a exp(-lam^a d).
    pis, vns: per source, pi_s(d) and Var(c_s) over the pairs; cross: (n, t, m^a_n, lam^a_n, m^a_t, lam^a_t) tuples."""
    noise = np.zeros_like(np.asarray(d, np.float64))
    for p_, v_ in zip(pis, vns):
        noise += p_ * p_ * v_
    for n_, t, ma_n, la_n, ma_t, la_t in cross:
        noise -= 2.0 * pis[n_] * pis[t] * (ma_t / ma_n) * np.exp(-(la_t - la_n) * d) * vns[t]
    return noise


def track_weights(s2, freq, alpha=None):
    """Per-allele weights w_k of the track LD scores: 1 / s_k^2 (s_k^2 the cohort's genotype variance; per-allele
    effects with variance proportional to 1 / s^2, SUMMIT's scale) or, with alpha, [2 f_k (1 - f_k)]^alpha with f_k the
    frequency at the mean ancestry (Hou et al.'s frequency-dependent model; alpha -0.38 by default there)."""
    if alpha is None:
        return 1.0 / np.asarray(s2, np.float64)
    h = 2.0 * np.asarray(freq, np.float64) * (1.0 - np.asarray(freq, np.float64))
    out = np.zeros_like(h)
    np.power(h, float(alpha), out=out, where=h > 0)
    return out


class _TrackScores:
    """Accumulators of the ancestry-track LD scores inside compose's window loop (compose_ldscores, `tracks`).

    Partner arrays (over compose's partners): q^s_k = m_s v^s_k (variance of the track variable), w_k (per-allele
    weight), V^s_k = q^s_k w_k. Rows receive, for every window pair, the own scores pi_s^2 U[c_s^2] w_k / q^s_j, the
    cross scores pi_s pi_t c_s c_t w_k / sqrt(q^s_j q^t_j), and, with a second population X (its panel's covariance
    c_X and variance v^X), the scores of each track against X, pi_s c_s c_X w_k / sqrt(q^s_j v^X_j), and X's own
    U[c_X^2] w_k / v^X_j. The own terms (k = j) are added by finish()."""

    def __init__(self, sources, m_vec, VmP, PmP, s2P, clsP, M, K, alpha, cross_name, xs_n, vxP=None):
        self.sources, self.nK = list(sources), len(sources)
        self.m = np.asarray(m_vec, np.float64)
        self.alpha = alpha
        self.w = track_weights(s2P, PmP @ self.m, alpha)
        self.v = [np.asarray(VmP[:, s], np.float64) for s in range(self.nK)]
        self.q = [self.m[s] * self.v[s] for s in range(self.nK)]
        self.iq = [np.divide(1.0, q, out=np.zeros_like(q), where=q > 0) for q in self.q]
        self.V = [q * self.w for q in self.q]
        self.clsP, self.M, self.K = clsP, M, K
        self.pairs = [(s, t) for s in range(self.nK) for t in range(s + 1, self.nK)]
        self.own = [np.zeros((M, K)) for _ in range(self.nK)]
        self.cross = {p: np.zeros((M, K)) for p in self.pairs}
        self.cross_name, self.xs_n = cross_name, xs_n
        if cross_name is not None:
            self.vx = np.asarray(vxP, np.float64)
            self.Vx = self.vx * self.w
            self.vs_x = [np.zeros((M, K)) for _ in range(self.nK)]
            self.x_own = np.zeros((M, K))

    def add_pairs(self, scatter, j, k, rj, rk, d, pis, css, vnss, bgs, cross_c, xpanel=None):
        """One chunk of window pairs (partners j < k, their rows rj, rk or -1); pis, css, vnss, bgs: per source
        pi_s(d), the covariance on the source's scale, its sampling variance and the panel's own background;
        cross_c: compose's (n, t, m^a_n, lam^a_n, m^a_t, lam^a_t) tuples of sources derived by inversion from panel t;
        xpanel: (c_X, v^X_j, v^X_k, Var(c_X), background of X's panel) or None."""
        wj, wk, cj, ck = self.w[j], self.w[k], self.clsP[j], self.clsP[k]
        for s in range(self.nK):
            sq = pis[s] * pis[s] * (css[s] * css[s] - vnss[s]) - bgs[s]
            scatter(self.own[s], rj, ck, sq * wk * self.iq[s][j])
            scatter(self.own[s], rk, cj, sq * wj * self.iq[s][k])
        for s, t in self.pairs:
            prod = css[s] * css[t]
            for n_, t_, ma_n, la_n, ma_t, la_t in cross_c:   # E[c_N c_t] = C_N C_t - (pi^a_t / pi^a_N) Var(e_t)
                if {n_, t_} == {s, t}:
                    prod = prod + (ma_t / ma_n) * np.exp(-(la_t - la_n) * d) * vnss[t_]
            prod = pis[s] * pis[t] * prod
            scatter(self.cross[(s, t)], rj, ck, prod * wk * np.sqrt(self.iq[s][j] * self.iq[t][j]))
            scatter(self.cross[(s, t)], rk, cj, prod * wj * np.sqrt(self.iq[s][k] * self.iq[t][k]))
        if xpanel is None:
            return
        c2, vxj, vxk, vnx, bgx = xpanel
        ixj = np.divide(1.0, vxj, out=np.zeros_like(vxj), where=vxj > 0)
        ixk = np.divide(1.0, vxk, out=np.zeros_like(vxk), where=vxk > 0)
        for s in range(self.nK):
            if s == self.xs_n:                                 # the same panel: its square, unbiased
                prod = css[s] * c2 - vnx - bgx
            else:
                prod = css[s] * c2
                for n_, t_, ma_n, la_n, ma_t, la_t in cross_c:
                    if n_ == s and t_ == self.xs_n:
                        prod = prod + (ma_t / ma_n) * np.exp(-(la_t - la_n) * d) * vnx
            prod = pis[s] * prod
            scatter(self.vs_x[s], rj, ck, prod * wk * np.sqrt(self.iq[s][j] * ixj))
            scatter(self.vs_x[s], rk, cj, prod * wj * np.sqrt(self.iq[s][k] * ixk))
        sqx = c2 * c2 - vnx - bgx
        scatter(self.x_own, rj, ck, sqx * wk * ixj)
        scatter(self.x_own, rk, cj, sqx * wj * ixk)

    def finish(self, out, snpdf, classes, row_part, cls, writer, masses_writer, flags):
        """Add each row's own term, keep the rows polymorphic in every track (and in X), write the files and return
        the params block."""
        rows = np.arange(self.M)
        keep = np.ones(self.M, dtype=bool)
        for s in range(self.nK):
            keep &= self.q[s][row_part] > 0
        if self.cross_name is not None:
            keep &= self.vx[row_part] > 0
        files, masses = {}, {}
        counts = np.bincount(self.clsP, minlength=self.K).astype(float)

        def put(name, Lm, Vrow, kind, definition):
            Lm[rows, cls] += Vrow
            path = f"{out}.{name}"
            writer(path, snpdf[keep].reset_index(drop=True), pd.DataFrame(Lm[keep], columns=classes),
                   snp_var=Vrow[keep])
            masses_writer(path, counts)
            Vk = Vrow[keep][:, None]
            files[name] = dict(file=f"{path}.gw.ldscore.gz", kind=kind, definition=definition,
                               ld_mass=float((Vk * Lm[keep]).sum()),
                               own_term_share=float((Vrow[keep] ** 2).sum() / (Vk * Lm[keep]).sum()))

        nm = self.sources
        Vr = [self.V[s][row_part] for s in range(self.nK)]
        for s in range(self.nK):
            put(f"track_{nm[s]}", self.own[s], Vr[s], "own",
                f"sum_k pi_{nm[s]}(d)^2 U[C_jk^2] w_k / q_j over the window, k = j: V_j = m v_j w_j")
        for s, t in self.pairs:
            put(f"track_{nm[s]}_{nm[t]}", self.cross[(s, t)], np.sqrt(Vr[s] * Vr[t]), "cross",
                f"sum_k pi_{nm[s]}(d) pi_{nm[t]}(d) C^{nm[s]}_jk C^{nm[t]}_jk w_k / sqrt(q^{nm[s]}_j q^{nm[t]}_j), "
                "k = j: sqrt(V V)")
        for s in range(self.nK):
            masses[nm[s]] = np.bincount(self.clsP, weights=np.where(self.q[s] > 0, self.V[s], 0.0),
                                        minlength=self.K).tolist()
        if self.cross_name is not None:
            X = self.cross_name
            Vxr = self.Vx[row_part]
            for s in range(self.nK):
                put(f"track_{nm[s]}_vs_{X}", self.vs_x[s], np.sqrt(Vr[s] * Vxr), "track_vs_cohort",
                    f"sum_k pi_{nm[s]}(d) C^{nm[s]}_jk C^{X}_jk w_k / sqrt(q_j v^{X}_j), k = j: sqrt(V V^{X}); "
                    f"a GWAS of other people from {X}'s population")
            put(f"cohort_{X}", self.x_own, Vxr, "cohort",
                f"sum_k U[(C^{X}_jk)^2] w_k / v^{X}_j over the window, k = j: V^{X}_j = v^{X}_j w_j (per-allele "
                f"LD scores of a GWAS of {X}'s population)")
            masses[f"cohort_{X}"] = np.bincount(self.clsP, weights=np.where(self.vx > 0, self.Vx, 0.0),
                                                minlength=self.K).tolist()
        phi = {}
        for s, t in self.pairs:
            num = float((self.w * np.sqrt(self.v[s].clip(0) * self.v[t].clip(0))).sum())
            phi[f"{nm[s]}_{nm[t]}"] = num / float(np.sqrt((self.w * self.v[s].clip(0)).sum()
                                                         * (self.w * self.v[t].clip(0)).sum()))
        return dict(sources=nm, mean_ancestry=self.m.tolist(),
                    weights=("1 / s_k^2 (the cohort's genotype variance in the model)" if self.alpha is None
                             else f"[2 f_k (1 - f_k)]^{self.alpha} (f_k the frequency at the mean ancestry)"),
                    alpha=self.alpha, n_rows=int(keep.sum()), n_rows_dropped=int((~keep).sum()),
                    n_partners=int(self.clsP.size), partner_counts=counts.tolist(), partner_V_mass=masses,
                    classes=list(classes), files=files, cross_source=self.cross_name, phi=phi, flags=flags,
                    definition="per-allele effects with Var(b^s_k) = tau_s,c w_k and Cov(b^s_k, b^t_k) = tau_st,c "
                               "w_k for partners k in class c; E[(z^s_j)^2] - 1 = N sum_c tau_s,c l^ss_jc and "
                               "E[z^s_j z^t_j] = N sum_c tau_st,c l^st_jc for Tractor's z statistics with sandwich "
                               "SEs; the track heritability is sum_c tau_s,c partner_V_mass[s][c]")


def _fft_conv_sym(W: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """F[g] = sum_g' W[g'] kern[|g - g'|] for every column of W (grid x classes)."""
    n = W.shape[0]
    size = 1 << int(np.ceil(np.log2(max(2 * n - 1, 2))))
    sym = np.zeros(size)
    sym[:n] = kern[:n]
    sym[size - n + 1:] = kern[1:n][::-1]
    fk = np.fft.rfft(sym)
    fw = np.fft.rfft(W, n=size, axis=0)
    return np.fft.irfft(fw * fk[:, None], n=size, axis=0)[:n]


def cell_half(z, key):
    """One half's cell covariance matrix from a curve's cells.npz: stored whole (S1, S2) or as its upper triangle row
    by row (S1_upper, S2_upper, with its size)."""
    if key in z.files:
        return z[key]
    n = int(z["size"])
    S = np.zeros((n, n))
    iu = np.triu_indices(n)
    S[iu] = z[f"{key}_upper"]
    S[(iu[1], iu[0])] = z[f"{key}_upper"]
    return S


def _cell_correction(path, kern, chrom_out, cm_out, Oflat, Wn, cls, K, k1, switch_cm, log):
    """Long-range admixture LD from the cohort's cell matrix in place of the kernel: for pairs of cells on the same
    chromosome at least switch_cm apart and for every pair of cells on different chromosomes, the kernel's square
    (Kt(d) x Kt(d), or far x far) is replaced by the product of the two half-sample estimates of the cells' covariance
    matrix, which is unbiased for its square. Returns (M x K correction to multiply by 1 / cres, summary)."""
    from .curve import _solve_kernel
    from .model import contrast_pairs
    z = np.load(path)
    if int(z["k1"]) != k1:
        raise ValueError(f"The cell matrix has {int(z['k1'])} contrasts, the composition {k1}.")
    Q = z["Q"]                                                     # cells x k1 x k1
    nC = Q.shape[0]
    chrom_c = z["chrom"].astype(str)
    start, width = z["start_cm"], z["width_cm"]
    mid = start + width / 2.0
    Kh = []
    for key in ("S1", "S2"):
        Sr = cell_half(z, key).reshape(k1, nC, k1, nC)             # [a, cell, b, cell']
        num = np.transpose(Sr, (1, 3, 0, 2))                       # cell, cell', a, b
        den = np.einsum("iau,jbv->ijabuv", Q, Q)
        Kc = np.zeros((nC, nC, k1, k1))
        if k1 == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                Kc[..., 0, 0] = np.where(den[..., 0, 0, 0, 0] > 0, num[..., 0, 0] / den[..., 0, 0, 0, 0], 0.0)
        else:
            pairs = contrast_pairs(k1)
            A = np.stack([den[..., u, v] + (den[..., v, u] if u != v else 0.0) for u, v in pairs], axis=-1)
            A = A.reshape(nC, nC, k1 * k1, len(pairs))
            AtA = np.einsum("ijep,ijeq->ijpq", A, A) + 1e-12 * np.eye(len(pairs))
            Atb = np.einsum("ijep,ije->ijp", A, num.reshape(nC, nC, k1 * k1))
            sol = np.linalg.solve(AtA, Atb[..., None])[..., 0]
            for t, (u, v) in enumerate(pairs):
                Kc[..., u, v] = Kc[..., v, u] = sol[..., t]
        Kh.append(Kc)
    uu = [(a, c) for a in range(k1) for c in range(k1)]
    same = chrom_c[:, None] == chrom_c[None, :]
    dist = np.abs(mid[:, None] - mid[None, :])
    use = (~same) | (dist >= switch_cm)
    Kd = kern(np.where(same, dist, 0.0))                           # kernel at the cells' distance
    k2 = k1 * k1
    delta_F = np.zeros((nC, nC, k2, k2))
    for ui, (a, c) in enumerate(uu):
        for vi, (b, e) in enumerate(uu):
            cell_sq = 0.5 * (Kh[0][..., a, b] * Kh[1][..., c, e] + Kh[1][..., a, b] * Kh[0][..., c, e])
            model_sq = np.where(same, Kd[..., a, b] * Kd[..., c, e], kern.far[a, b] * kern.far[c, e])
            delta_F[..., ui, vi] = np.where(use, cell_sq - model_sq, 0.0)
    # the composed SNPs' cells
    cell_of = np.full(chrom_out.size, -1, dtype=np.int64)
    for c in np.unique(chrom_c):
        rows = np.flatnonzero(chrom_c == c)
        sel = np.flatnonzero(chrom_out == str(c))
        if sel.size == 0:
            continue
        k = np.floor((cm_out[sel] - start[rows[0]]) / width[rows[0]]).astype(np.int64)
        cell_of[sel] = rows[0] + np.clip(k, 0, rows.size - 1)
    have = cell_of >= 0
    Wc = np.zeros((nC, k2, K))
    for v in range(k2):
        np.add.at(Wc[:, v, :], (cell_of[have], cls[have]), Wn[have, v])
    per_cell = np.einsum("ijuv,jvk->iuk", delta_F, Wc)             # cells x k2 x classes
    corr = np.zeros((chrom_out.size, K))
    corr[have] = np.einsum("ju,juk->jk", Oflat[have], per_cell[cell_of[have]])
    log(f"[admixed-ld] compose: cell matrix of {nC} cells used beyond {switch_cm} cM and between chromosomes "
        f"({int((~have).sum())} SNPs outside the cells)")
    return corr, dict(file=path, n_cells=int(nC), switch_cm=float(switch_cm))


class RelatednessProfile:
    """The relatives' term of the LD scores from the curve's relatedness profile (`curve --relative-pairs`).

    For SNPs j and k the excess of nu^2 E(r_jk^2) over the population LD and the iid noise that relative pairs create
    is sum over ordered pairs of E(y_j y_k) less what unrelated pairs give, y_j = u_aj u_bj the pair's local relatedness
    in the covariate-adjusted, standardized genotypes. Between chromosomes it is B (about sum of g^2 over ordered pairs);
    on one chromosome it is phi(d), which exceeds B by the autocovariance of local relatedness at distance d (identical
    by descent segments). The term of SNP j in class c is
        (f / nu^2) [B sum_{k in c, k != j} V_k + sum_{k in c on j's chromosome, k != j} (phi(d_jk) - B) V_k],
    B and phi being sums over the ordered pairs of the curve's people and f the share of them that the GWAS holds
    (relatedness_scale).
    phi is pooled in bins of lags whose width grows with distance (`smooth` times the distance, at least one lag), so
    that the noise of long lags averages out, and interpolated linearly between bin centres; beyond the last bin it
    keeps the last value."""

    def __init__(self, table: pd.DataFrame, between: float, n_listed: int, n_controls: int, smooth: float = 0.1,
                 source: str = ""):
        t = table[table.w > 0].sort_values("lag")
        ratio = n_listed / n_controls if n_controls else 0.0
        d = (t.dsum.to_numpy() if "dsum" in t else t.d_cm.to_numpy() * t.w.to_numpy())
        num = t.listed.to_numpy() - ratio * t.controls.to_numpy()
        w = t.w.to_numpy()
        lag = t.lag.to_numpy()
        cell = float(np.nanmedian(np.diff(t.d_cm.to_numpy()[1:]))) if len(t) > 2 else 1.0
        centers, values, weights, i = [], [], [], 0
        while i < len(t):
            d0 = t.d_cm.to_numpy()[i]
            width = max(cell, smooth * d0) if lag[i] > 0 else 0.0     # lag 0 (within a cell) stays its own bin
            j = i + 1
            while j < len(t) and lag[j] > 0 and t.d_cm.to_numpy()[j] < d0 + width:
                j += 1
            centers.append(d[i:j].sum() / w[i:j].sum())
            values.append(num[i:j].sum() / w[i:j].sum())
            weights.append(w[i:j].sum())
            i = j
        self.d = np.asarray(centers)
        self.phi = np.asarray(values)
        self.w = np.asarray(weights)
        self.between = float(between)
        self.smooth = float(smooth)
        self.source = source

    @classmethod
    def from_curve(cls, curve_prefix: str, info: dict, smooth: float = 0.1, background_min_cm: float = 3.0,
                   background_window: str = "zero"):
        """The listed pairs' profile, or, when the curve also measured all pairs (--background): with
        background_window 'panels' the smooth all-pairs background at every distance (SmoothBackground, fitted from
        background_min_cm on; compose then removes the panels' own background inside the band), otherwise (the rules
        of 5 Nov 2026) that measurement from background_min_cm on and between chromosomes, with the listed profile
        plus the unlisted part (flat: its value at background_min_cm; zero: none) below."""
        if background_window == "panels" and "background" in info:
            return SmoothBackground.from_curve(curve_prefix, info, fit_min_cm=background_min_cm, smooth=smooth)
        r = info.get("relatedness")
        prof = None
        if r is not None and "profile" in r:
            t = pd.read_csv(f"{curve_prefix}.relatedness.tsv", sep="\t")
            if "dsum" not in t:
                t["dsum"] = t.d_cm * t.w
            prof = cls(t, r["between"]["value"], int(r["pairs"]), int(r["profile"]["n_controls"]), smooth,
                       source=f"{curve_prefix}.relatedness.tsv")
        if "background" not in info:
            return prof
        b = info["background"]
        t = pd.read_csv(b["file"] if os.path.exists(b["file"]) else f"{curve_prefix}.background.tsv", sep="\t")
        tb = pd.DataFrame({"lag": t.lag, "d_cm": t.d_cm, "w": t.w, "dsum": t.d_cm * t.w,
                           "listed": (t.excess * t.w).to_numpy(), "controls": 0.0})
        bgp = cls(tb, b["between"]["value"], 1, 0, smooth, source=b["file"])
        return BackgroundProfile(prof, bgp, background_min_cm, background_window)

    def __call__(self, d):
        d = np.asarray(d, np.float64)
        return np.interp(d, self.d, self.phi, left=self.phi[0], right=self.phi[-1])

    def describe(self) -> dict:
        at = [0.5, 1, 2, 5, 10, 20, 40, 80]
        return dict(mode="profile", between=self.between, smooth=self.smooth, source=self.source,
                    phi_at={str(x): float(self(x)) for x in at if x <= self.d.max()})


class BackgroundProfile:
    """The relatives' term of all pairs of people. The curve's measurement of all pairs (`background`, from the least
    differentiated SNPs) gives B between chromosomes and, beyond d0, the relatedness LD on one chromosome; the part of
    it that the listed pairs do not carry, r(d) = background(d) - listed(d), is smoothed as B_u + sum_m A_m exp(-l_m d)
    with A_m >= 0 (weighted non-negative least squares over d >= d0 on a grid of rates; B_u = B - B_listed), the form
    of the identity-by-descent autocovariance of any mix of relationships, which removes the noise of long lags. The
    term is listed(d) + r(d) from d0 on. Below d0 the measurement also holds within-source LD, and inside the
    reference window the panels' unbiased squared covariances already hold their own populations' background
    relatedness with weight pi_s(d)^2, so there the term is the listed profile plus r(d0) (flat) or plus nothing
    (zero, the default)."""

    RATES = np.array([0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2])

    def __init__(self, listed, background, d0=3.0, window="zero"):
        from scipy.optimize import nnls
        self.listed, self.bg, self.d0, self.window = listed, background, float(d0), window
        self.between = background.between
        self._listed = (lambda d: listed(d)) if listed is not None else (lambda d: np.zeros(np.shape(d)))
        self.b_unlisted = background.between - (listed.between if listed is not None else 0.0)
        use = (background.d >= self.d0) & (background.w > 0)
        dd, ww = background.d[use], background.w[use]
        r = background.phi[use] - self._listed(dd) - self.b_unlisted
        A = np.exp(-np.outer(dd, self.RATES)) * np.sqrt(ww)[:, None]
        self.amp, _ = nnls(A, r * np.sqrt(ww)) if dd.size else (np.zeros(self.RATES.size), 0.0)
        self.rem0 = float(self.remainder(self.d0))
        self.source = background.source
        self.d = background.d

    def remainder(self, d):
        d = np.asarray(d, np.float64)
        return self.b_unlisted + np.exp(-np.multiply.outer(d, self.RATES)) @ self.amp

    def __call__(self, d):
        d = np.asarray(d, np.float64)
        below = self._listed(d) + (self.rem0 if self.window == "flat" else 0.0)
        return np.where(d >= self.d0, self._listed(d) + self.remainder(d), below)

    def describe(self) -> dict:
        at = [0.5, 1, 2, 5, 10, 20, 40, 80]
        return dict(mode="background", between=self.between, between_listed=None if self.listed is None else
                    self.listed.between, d0=self.d0, window=self.window, unlisted_at_d0=self.rem0,
                    unlisted_rates=self.RATES.tolist(), unlisted_amplitudes=self.amp.tolist(), source=self.source,
                    phi_at={str(x): float(self(x)) for x in at if x <= self.d.max()})


def background_rates(fit_min_cm, n=8):
    """Decay rates per cM of SmoothBackground: geometric from 0.01 to 1 / fit_min_cm, the components whose
    autocovariance a fit from fit_min_cm on resolves (a component decaying faster is mostly gone by fit_min_cm, and
    its amplitude at short distance would be an extrapolation of noise or of within-source LD)."""
    return np.geomspace(0.01, 1.0 / float(fit_min_cm), n)


class SmoothBackground:
    """The background LD of a group of people, a cohort (`curve --background`) or a source panel (recorded by
    `clean-panels`), as a smooth function of genetic distance:
        S(d) = B + sum_m A_m exp(-l_m d),   A_m >= 0,
    with B the between-chromosome value and the rates of background_rates(fit_min_cm), fitted by least squares
    weighted by SNP pairs to the measured profile (lags pooled in bins growing at 10% of the distance) on
    d >= fit_min_cm. This is the form of the autocovariance of identity by descent for any mix of relationships and of
    structure from recent admixture, and it is evaluated at every distance. Below fit_min_cm the measured profile also
    holds within-source LD (in the cohort pi_s(d)^2 W_s, in a panel W_s), which the panels supply; the fit keeps only
    the components the fitting range resolves, sharing on segments longer than about fit_min_cm (the last 10 to 20
    generations), so its extrapolation into the band does not take up within-source LD. Values are sums over ordered
    pairs of people per SNP pair, the scale of `curve --background`; per_pair divides by n (n - 3), the scale on which
    a panel's background enters its unbiased squared covariances (E U_s = c_s^2 + v_j v_k S_s(d) / (n_s (n_s - 3)))."""

    def __init__(self, table, between, n, fit_min_cm=3.0, smooth=0.1, source="", name=""):
        from scipy.optimize import lsq_linear
        t = pd.DataFrame(table)
        tb = pd.DataFrame({"lag": t["lag"].to_numpy(), "d_cm": t["d_cm"].to_numpy(), "w": t["w"].to_numpy(),
                           "dsum": t["d_cm"].to_numpy() * t["w"].to_numpy(),
                           "listed": t["excess"].to_numpy() * t["w"].to_numpy(), "controls": 0.0})
        tb = tb[np.isfinite(tb.listed) & np.isfinite(tb.d_cm)]
        pool = RelatednessProfile(tb, between, 1, 0, smooth, source=source)
        self.between, self.n, self.fit_min_cm = float(between), int(n), float(fit_min_cm)
        self.rates = background_rates(fit_min_cm)
        use = (pool.d >= self.fit_min_cm) & (pool.w > 0)
        if use.sum() < 2:
            raise ValueError(f"The background profile {source or name} has fewer than 2 points beyond "
                             f"{fit_min_cm} cM.")
        dd, ww = pool.d[use], pool.w[use]
        A = np.exp(-np.outer(dd, self.rates)) * np.sqrt(ww)[:, None]
        self.amp = lsq_linear(A, (pool.phi[use] - self.between) * np.sqrt(ww), bounds=(0.0, np.inf),
                              method="bvls").x
        self.source, self.name, self.smooth = source, name, float(smooth)
        self.d = pool.d

    @classmethod
    def from_curve(cls, curve_prefix: str, info: dict, fit_min_cm=3.0, smooth=0.1):
        b = info["background"]
        path = b["file"] if os.path.exists(b["file"]) else f"{curve_prefix}.background.tsv"
        return cls(pd.read_csv(path, sep="\t"), b["between"]["value"], info["n_samples"], fit_min_cm, smooth,
                   source=path, name="cohort")

    @classmethod
    def from_record(cls, rec: dict, fit_min_cm=3.0, smooth=0.1, name=""):
        """From a panel's `panel_background` record in a reference manifest (written by clean-panels)."""
        return cls(rec["table"], rec["between"]["value"], rec["n"], fit_min_cm, smooth,
                   source=rec.get("file", ""), name=name)

    def __call__(self, d):
        d = np.asarray(d, np.float64)
        return self.between + np.exp(-np.multiply.outer(d, self.rates)) @ self.amp

    def per_pair(self, d):
        return self(d) / (self.n * (self.n - 3.0))

    def describe(self) -> dict:
        at = [0, 0.5, 1, 2, 3, 5, 10, 20, 40, 80]
        return dict(mode="background", window="panels", name=self.name, between=self.between, n=self.n,
                    fit_min_cm=self.fit_min_cm,
                    rates=self.rates.tolist(), amplitudes=self.amp.tolist(), source=self.source,
                    phi_at={str(x): float(self(x)) for x in at})


def panel_background_record(info: dict, table: pd.DataFrame, file: str = "") -> dict:
    """The aggregates of a panel's background measurement (genotype_curve(..., background=True) on the panel, without
    covariates) that a reference manifest keeps: the table by lag and the between-chromosome value."""
    b = info["background"]
    keep = table[np.isfinite(table.excess)]
    return dict(n=int(info["n_samples"]), quantile=float(b["quantile"]), cell_cm=float(b["cell_cm"]),
                null=float(b["null"]), n_snps=int(b["n_snps"]),
                between=dict(value=float(b["between"]["value"]), n_snps=int(b["between"]["n_snps"])),
                table={k: keep[k].astype(float).round(10).tolist() if k != "lag" else keep[k].astype(int).tolist()
                       for k in ("lag", "d_cm", "w", "excess")},
                file=file,
                definition="the panel's all-pairs cross-person LD among its least differentiated SNPs per SNP pair "
                           "(sum over ordered pairs of members), less its admixture part and the null, without "
                           "covariates (curve --background on the panel); its unbiased squared covariances hold "
                           "v_j v_k S(d) / (n (n - 3)) of it")


def relatedness_scale(n_curve, n_gwas):
    """The share of the curve's ordered pairs of people that a GWAS of n_gwas of them, taken at random, holds:
    N (N - 1) / (n (n - 1)). The relatives' sums of the curve (B, phi, the sum of g^2) are sums over the ordered pairs
    of its n people, so they enter the GWAS's LD scores times this share: exactly 1 when the curve is made on the
    GWAS's people, above 1 when the GWAS has more people than the curve (the same density of relatives assumed)."""
    n, N = float(n_curve), float(n_gwas)
    if not (n >= 2 and N >= 2):
        raise ValueError(f"The relatives' term needs at least 2 people in the curve and in the GWAS (curve {n_curve}, "
                         f"GWAS {n_gwas}).")
    return N * (N - 1.0) / (n * (n - 1.0))


def relatedness_term(L, rel, chrom, cm, cls, V, K, nu, grid_cm, scale=1.0, partners=None):
    """Add the relatives' term of `rel` (RelatednessProfile) to the LD scores L (M x K) in place: B between all pairs
    (each SNP without itself) and phi(d) - B along each chromosome, by FFT convolution on a grid of grid_cm, times
    `scale` (relatedness_scale: the share of the curve's pairs of people that the GWAS holds). partners: (chrom, cm,
    cls, V) of compose's partners when they are not the rows (`--partners reference`; every row is one of them)."""
    if partners is not None:
        chP, cmP, clsP, VP = partners
        L += (scale * rel.between / nu ** 2) * np.bincount(clsP, weights=VP, minlength=K)[None, :]
        L[np.arange(cls.size), cls] -= (scale * rel.between / nu ** 2) * V
        for c in pd.unique(chrom):
            sel, selP = np.flatnonzero(chrom == c), np.flatnonzero(chP == c)
            x0 = cmP[selP].min()
            gP = np.floor((cmP[selP] - x0) / grid_cm).astype(np.int64)
            g = np.floor((cm[sel] - x0) / grid_cm).astype(np.int64)
            ng = int(gP.max()) + 1
            kern = rel(np.arange(ng) * grid_cm) - rel.between
            Vg = np.zeros((ng, K))
            np.add.at(Vg, (gP, clsP[selP]), VP[selP])
            acc = _fft_conv_sym(Vg, kern)[g]
            acc[np.arange(sel.size), cls[sel]] -= kern[0] * V[sel]
            L[sel] += (scale * acc) / nu ** 2
        return
    Vtot = np.bincount(cls, weights=V, minlength=K)
    L += (scale * rel.between / nu ** 2) * Vtot[None, :]
    L[np.arange(cls.size), cls] -= (scale * rel.between / nu ** 2) * V
    for c in pd.unique(chrom):
        sel = np.flatnonzero(chrom == c)
        x = cm[sel]
        g = np.floor((x - x.min()) / grid_cm).astype(np.int64)
        ng = int(g.max()) + 1
        kern = rel(np.arange(ng) * grid_cm) - rel.between
        Vg = np.zeros((ng, K))
        np.add.at(Vg, (g, cls[sel]), V[sel])
        acc = _fft_conv_sym(Vg, kern)[g]
        acc[np.arange(sel.size), cls[sel]] -= kern[0] * V[sel]          # the convolution includes each SNP itself
        L[sel] += (scale * acc) / nu ** 2


def release_statistic(curve):
    """The sample statistic released with the curve CURVE (prefix): (path, None), or (None, why) when the curve records
    none (made with --no-sample-factors, before curves carried it, or its pass could not make it). The curve's json
    lists the statistic its own pass wrote (key sample_factors, a file next to it: CURVE.sample_factors.json, or the
    name listed when the curve's files were renamed without it). A listed statistic that is missing stops the
    composition, and so does one whose number of people (n_samples) or covariate rank differs from the curve's (a file
    left by another run). A statistic file next to a curve that does not list it is not used."""
    from .sample_factors import read_sample_factors
    cj = f"{curve}.curve.json"
    with open(cj) as fh:
        info = json.load(fh)
    remedy = ("remake the curve, pass --sample-factors PATH with the statistic of the GWAS's people, or "
              "--no-sample-factors to compose the same-person term under random mating")
    listed = info.get("sample_factors")
    if not listed:
        if "sample_factors" in info:
            notes = [str(f.get("detail")) for f in info.get("flags", []) if f.get("flag") == "sample_factors_not_made"]
            why = f"the curve {cj} has no sample statistic: its pass could not make one" + (
                f" ({'; '.join(notes)})" if notes else "")
        else:
            why = (f"the curve {cj} lists no sample statistic (made with --no-sample-factors, or before curves "
                   "carried it)")
        if os.path.exists(f"{curve}.sample_factors.json"):
            why += (f"; {curve}.sample_factors.json is not the curve's and is not used (pass --sample-factors "
                    f"{curve}.sample_factors.json if it is of the GWAS's people)")
        return None, why
    # the file next to the curve under its prefix (a curve's files renamed together keep it), else the name listed
    cands = list(dict.fromkeys([f"{curve}.sample_factors.json",
                                os.path.join(os.path.dirname(curve), os.path.basename(str(listed)))]))
    path = next((c for c in cands if os.path.exists(c)), None)
    if path is None:
        raise ValueError(f"The curve {cj} lists a sample statistic ({listed}), which does not exist next to it "
                         f"({' or '.join(cands)}); {remedy}.")
    sfi = read_sample_factors(path)
    bad = [f"{k} {sfi.get(k)} in the statistic, {info.get(k)} in the curve" for k in ("n_samples", "covariate_rank")
           if sfi.get(k) is None or info.get(k) is None or int(sfi[k]) != int(info[k])]
    if bad:
        raise ValueError(f"The sample statistic {path} is not that of the curve {cj} ({'; '.join(bad)}): a file of "
                         f"another run; {remedy}.")
    return path, None


def release_coverage(tables, n_med, n_release, min_share=0.85, log=print, against="release"):
    """Traits whose GWAS sample holds less than min_share of the people of the cohort's release (the people of the
    curve and of its sample statistic). Composition follows each GWAS's N, but not the ancestry and relatedness of the
    people with the trait, which can differ from the release's when the trait was measured in only part of the cohort.
    Such a trait is composed alone (its own summary statistics) with the cohort's curve and the sample statistic of
    its own people. The traits and their shares are recorded and logged. against: 'release' (n_release the curve's
    people), or 'statistic' when the same-person term comes from a statistic given to compose (n_release its people:
    a trait composed with the statistic of its own people is then not flagged)."""
    if n_release is None or not tables or min_share is None or min_share <= 0:
        return []
    share = np.asarray(n_med, np.float64) / float(n_release)
    low = np.flatnonzero(share < float(min_share))
    if low.size == 0:
        return []
    names = [t[0] for t in tables]
    whose = "the release's" if against == "release" else "the sample statistic's"
    detail = ", ".join(f"{names[i]} (N {n_med[i]:.0f}, {share[i]:.2f} of {int(n_release)})" for i in low)
    log(f"[admixed-ld] WARNING (recorded flag gwas_sample_smaller_than_release): {low.size} trait(s) were analysed "
        f"in less than {min_share:.0%} of {whose} {int(n_release)} people: {detail}. Their LD can differ from "
        "the release's in the ancestry and relatedness of the people measured. Compose each alone (its own summary "
        "statistics) with the cohort's curve and the sample statistic of its own people (summit admixed-ld "
        "sample-factors --keep <its people>, with the GWAS covariates).")
    return [dict(flag="gwas_sample_smaller_than_release", detail=detail, min_share=float(min_share),
                 against=str(against), people=int(n_release))]


def se_regression_outliers(per_trait, log=print, max_m=0.05, max_kappa=0.25, min_r2=0.9):
    """Traits whose standard-error regression stands apart from the others (first source's m more than max_m from the
    median, kappa_res(0) more than max_kappa of the median away, or a fit R2 below min_r2): logistic, mixed-model or
    otherwise weighted SEs, other covariates or another sample. The medians over traits are used either way; the
    traits are recorded and logged."""
    if per_trait is None or len(per_trait) < 2:
        return []
    m1 = np.array([np.atleast_1d(x)[0] for x in per_trait["m"]], np.float64)
    k1 = np.array([np.atleast_2d(x)[0, 0] for x in per_trait["kappa0"]], np.float64)
    r2 = per_trait["fit_r2"].to_numpy(np.float64)
    km = float(np.median(k1))
    odd = (np.abs(m1 - np.median(m1)) > max_m) | (np.abs(k1 - km) > max_kappa * abs(km)) | (r2 < min_r2)
    if not odd.any():
        return []
    names = per_trait["trait"].to_numpy()[odd]
    detail = ", ".join(f"{t} (m {a:.3f}, kappa_res(0) {b:.4f}, R2 {c:.3f})" for t, a, b, c in
                       zip(names, m1[odd], k1[odd], r2[odd]))
    log(f"[admixed-ld] WARNING (recorded flag se_regression_outlier_traits): the standard errors of {names.size} "
        f"trait(s) give other cohort quantities than the rest (median m {np.median(m1):.3f}, kappa_res(0) {km:.4f}): "
        f"{detail}. Logistic, mixed-model or otherwise weighted SEs do this, and so does a trait analysed in another "
        "subset of people or with other covariates; such a trait needs its own composition: for a GWAS sample of its "
        "own, compose it alone (its own summary statistics) with the cohort's curve and the sample statistic of its "
        "own people (summit admixed-ld sample-factors --keep <its people>).")
    return [dict(flag="se_regression_outlier_traits", detail=detail)]


def _parse_values(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in str(x).split(",") if v.strip()]


def _annotation_columns(annot, snps):
    """0/1 annotation columns (SNPs x columns) aligned to `snps` (absent SNPs: all 0). The file has a SNP column (CHR,
    BP and CM are ignored) and one 0/1 column per annotation; every listed SNP must be in at least one column. The
    columns may overlap (e.g. MAF/LD bins and functional annotations): compose then works on their disjoint
    combinations and sums those into the columns."""
    df = pd.read_csv(annot, sep=r"\s+", dtype={"SNP": str})
    _, cols = utils._split_snp_metadata(df.columns, source=f"annotation '{annot}'", required=("SNP",))
    if not cols:
        raise ValueError(f"{annot} has no annotation columns.")
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isin(vals, (0.0, 1.0)).all():
        raise ValueError(f"{annot}: annotation values must be 0 or 1.")
    if (vals.sum(axis=1) < 1).any():
        raise ValueError(f"{annot}: {int((vals.sum(axis=1) < 1).sum())} listed SNP(s) are in no annotation column; "
                         "leave them out of the file or give them a column.")
    if df["SNP"].duplicated().any():
        raise ValueError(f"{annot} has duplicate SNP IDs.")
    pos = pd.Index(df["SNP"]).get_indexer(np.asarray(snps).astype(str))
    out = np.zeros((pos.size, len(cols)))
    out[pos >= 0] = vals[pos[pos >= 0]]
    return out, list(cols)


def _n_groups(tables, n_med, N, nu, p, nu_given, tol):
    """Groups of traits whose median N agree within tol (relative to the group's smallest), each with its N (the
    group's median) and nu."""
    items = sorted(zip([t[0] for t in tables], [float(x) for x in n_med]), key=lambda x: x[1])
    groups, cur = [], []
    for name, n in items:
        if cur and n > cur[0][1] * (1 + tol):
            groups.append(cur)
            cur = []
        cur.append((name, n))
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        Ng = float(np.median([n for _, n in g]))
        out.append(dict(N=Ng, nu=float(Ng - p - 1.0), label=str(int(round(Ng))), traits=[[t, n] for t, n in g]))
    return out


def compose_ldscores(ref_path, sumstats, annot, out, *, sources=None, snps=None, match="position", curve=None,
                     generations=None, decay_per_cm=None, mean_ancestry=None, kappa0=None, theta_sd=None,
                     relatedness=None, relatedness_smooth=0.1, background_min_cm=3.0, background_window="panels",
                     nu=None, r2=0.99, grid_cm=0.02, cell_matrix=False,
                     cell_switch_cm=10.0, allow_flat_kernel_join=False, cross_source=None,
                     sample_factors=None, sample_factors_calibrate=True, calibrate_between_chromosomes=False,
                     per_trait_n=True, n_tol=0.01, tracks=False, track_alpha=None, partners="rows",
                     drop_ambiguous=False, pair_chunk=1 << 23, release_sample_factors=True, min_release_share=0.85,
                     dry_run=False, require=None, max_annotation_atoms=64, log=print):
    """Composed LD scores for an admixed GWAS sample with two or more sources (the last is the baseline of the
    contrasts). mean_ancestry: K-1 or K values (comma-separated); kappa0: the upper triangle of Kt_res(0) row by row.
    Departures from the standard recipe that are allowed but should be visible are recorded in the params json under
    `flags` (and logged as warnings); inconsistent inputs raise.

    cross_source: also write the cross-population LD scores of this GWAS with a second GWAS on other people from an
    unadmixed population whose LD is that of this reference source's panel (e.g. EUR), to <out>.cross_<name>:
        lx_jc = sum_{k in c} rho_1(j,k) rho_2(j,k) sqrt(V_1k V_2k),
    rho_1 the composed correlation of this sample's covariate-adjusted genotypes, rho_2 the panel's correlation (its
    population has no admixture LD, so only pairs within the window count) and V_2 = 1. The samples are disjoint, so
    there is no same-person or relatives' term. When the panel is also one of this sample's sources, its squared
    covariance takes the unbiased estimate and its own background is removed once.

    sample_factors: the cohort's person-factor statistic (`summit admixed-ld sample-factors`, made on the GWAS's people
    with its covariates). The same-person term is then (n / (nu_sf nu)) sum_{k in c} a_j' Sigma_u a_k / (cres_j s2_k)
    in place of the random-mating covariance of heterozygosity (sample_factors.py); both are recorded. With
    sample_factors_calibrate (default), the term is scaled so that its mass over pairs on different chromosomes equals
    the cohort's realized one, which the statistic records; the factors then allocate it across SNPs and classes.
    In curve mode without sample_factors (and release_sample_factors, the default) the statistic that the curve's json
    lists is used (release_statistic: it must exist and be of the curve's people and covariates, else compose stops); a
    curve that lists none gives the random-mating term with the flag curve_without_sample_factors.

    per_trait_n: the same-person term (1 / nu) and the relatives' term (f(N) / nu^2) belong to the GWAS's sample, and in a
    small cohort (with a strong background) traits with different missingness have measurably different in-sample LD.
    When the traits' median N differ by more than n_tol, one LD-score file is written per group of traits with the same N
    (out.n<N>.gw.ldscore.gz, listed by file name in out.n_groups.tsv; nu = N - rank - 1 with the traits' covariate
    rank), the population part composed once; out.gw.ldscore.gz stays the file for the median N (with the given nu, if
    any). A composition with one file for every trait removes an earlier run's out.n_groups.tsv.

    annot: 0/1 columns, which may overlap; overlapping columns are composed on their distinct combinations and summed
    into the columns. max_annotation_atoms (default 64) stops a composition with more combinations than that before
    any per-combination array is allocated (memory and time grow with their number).

    partners: 'rows' (the default) takes the partners k of every LD score to be the GWAS SNPs themselves; 'reference'
    takes every reference variant listed in the annotation (a dense reference holding the causal variants), with the
    GWAS SNPs as the rows j. Each SNP of the annotation is then a partner, the masses in <out>.gw.M are the partners'
    counts per class, and every term is a sum over the partners (the window pairs, the admixture LD, the same-person
    and relatives' terms); the cohort's cell matrix and sample factors, which are measured on the GWAS SNPs, are not
    available with it.

    tracks: also write the LD scores of the within-ancestry allele deviations that a local-ancestry-aware GWAS
    (Tractor: y ~ covariates + local-ancestry dosages + ancestry-specific allele counts) regresses on, for the
    correlation of per-allele causal effects across ancestry tracks in the same people (tracks.py):
        own track s     l^ss_j = sum_k pi_s(d)^2 U[(C^s_jk)^2] w_k / q^s_j                     (k = j: V^s_j)
        tracks s, t     l^st_j = sum_k pi_s(d) pi_t(d) C^s_jk C^t_jk w_k / sqrt(q^s_j q^t_j)   (k = j: sqrt(V^s_j V^t_j))
    over pairs within the reference window, pi_s(d) = m_s exp(-lam_s d), q^s_j = m_s v^s_j the variance of the track
    variable, V^s_k = q^s_k w_k and w_k the per-allele weight (1 / s_k^2 with s_k^2 the cohort's genotype variance, or
    [2 f_k (1 - f_k)]^track_alpha with f_k the frequency at the mean ancestry). The track variables carry no admixture
    LD, so the scores are short-range only, without same-person or relatives' terms (the sandwich standard errors of
    the GWAS take the heteroscedasticity out). Independent panels make the plain product of two sources' covariances
    unbiased; a source derived by inversion from a known panel has the covariance of their errors removed. With
    cross_source X the tracks against a GWAS of other people from population X are written too (per-allele scale,
    V^X_k = v^X_k w_k), and X's own per-allele LD scores from its panel. Files: <out>.track_<s>, <out>.track_<s>_<t>,
    <out>.track_<s>_vs_<X>, <out>.cohort_<X> (.gw.ldscore.gz and .gw.M), all on the same rows; the definitions, the
    partners' masses and the track parameters are in the params json under `tracks`."""
    from ..ldscore.gw_ldscore import write_gw_annotation_masses, write_gw_ldscore_table
    from ..relatives.moments import read_annotation_classes
    from ..sumstats.sumstats import Sumstats
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    sources = list(sources or ref.names[:2])
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("Composition needs two or more distinct sources.")
    nK, k1 = len(sources), len(sources) - 1
    si = [ref.source_index(s) for s in sources]
    ns = np.array([ref.n[s] for s in sources], np.float64)
    xs_i = xs_n = None                       # cross_source: its index in the reference, and in sources (if one)
    if cross_source is not None:
        xs_i = ref.source_index(cross_source)
        xm = ref.manifest["sources"][xs_i]
        if xm.get("derived"):
            raise ValueError(f"--cross-source {cross_source} is a derived source; it needs a panel's own LD.")
        xs_n = sources.index(cross_source) if cross_source in sources else None
    if sum(x is not None for x in (curve, generations, decay_per_cm)) != 1:
        raise ValueError("Give exactly one source for the decay: --curve, --admixture-generations or --decay-per-cm.")
    if partners not in ("rows", "reference"):
        raise ValueError("partners must be 'rows' (the GWAS SNPs) or 'reference' (every reference variant in the "
                         "annotation).")
    dense = partners == "reference"
    # curve mode's same-person term: the sample statistic the curve's pass wrote next to it, unless one is given
    sf_from_release = False
    pre_flags = []
    if sample_factors is None and release_sample_factors and curve is not None and not dense:
        sample_factors, why = release_statistic(curve)
        if sample_factors is not None:
            sf_from_release = True
            log(f"[admixed-ld] compose: same-person term from the curve's sample statistic ({sample_factors}); "
                "--no-sample-factors composes it under random mating instead")
        else:
            pre_flags.append(dict(flag="curve_without_sample_factors", detail=why))
            log(f"[admixed-ld] WARNING (recorded flag curve_without_sample_factors): {why}; the same-person term is "
                "composed under random mating. Pass --sample-factors PATH with the statistic of the GWAS's people, or "
                "--no-sample-factors to compose under random mating without this warning.")
    if dense and (cell_matrix is True or sample_factors is not None or calibrate_between_chromosomes):
        raise ValueError("--partners reference cannot be combined with --cell-matrix, --sample-factors or "
                         "--calibrate-between-chromosomes: those are measured on the GWAS SNPs.")
    sfi = None
    if sample_factors is not None:
        from .sample_factors import read_sample_factors
        sfi = read_sample_factors(sample_factors)
    if track_alpha is not None and not tracks:
        raise ValueError("--track-alpha sets the per-allele weights of the track LD scores; it needs --tracks.")

    # ---------------------------------------------------------------------------------------- target variants
    files = _sumstats_files(sumstats)
    tv = _target_variants(snps, files[0])
    if tv.empty:
        raise ValueError(f"No variants in {snps or files[0]}.")
    how = match
    if how == "position" and not {"CHR", "BP"} <= set(tv.columns):
        raise ValueError("Matching by position needs CHR and BP (in the summary statistics or --snps); "
                         "or use --match id.")
    if how == "position":
        tv = tv[tv.BP.notna()].copy()
        tv["BP"] = tv.BP.astype(np.int64)
    rv = ref.variants(sources)
    match_info = {}
    t_idx, r_idx, _ = match_variants(tv.reset_index(drop=True), rv, how=how, drop_ambiguous=drop_ambiguous,
                                     report=match_info)
    check_match(match_info, "compose: the GWAS", log)
    tv = tv.reset_index(drop=True)
    col_mat, col_names = _annotation_columns(annot, tv.SNP.to_numpy()[t_idx])
    col_pat = None                                  # overlapping columns: composed on their disjoint atoms, summed back
    if (col_mat.sum(axis=1) > 1).any():
        if dense or tracks:
            raise ValueError("Overlapping annotation columns are not supported with --partners reference or --tracks; "
                             "give a partition (each SNP in one class).")
        in_any = col_mat.sum(axis=1) >= 1
        col_pat, atom = np.unique(col_mat[in_any], axis=0, return_inverse=True)
        n_atoms, n_rows = int(col_pat.shape[0]), int(in_any.sum())
        if max_annotation_atoms is not None and n_atoms > int(max_annotation_atoms):
            # every term is composed per combination: about ten SNPs x combinations arrays, and the window pairs'
            # accumulation and convolutions per combination, so memory and time grow with their number
            gib = n_rows * n_atoms * 8 / 2 ** 30
            raise ValueError(
                f"{annot}: the {len(col_names)} overlapping annotation columns form {n_atoms} distinct combinations "
                f"on the {n_rows} GWAS SNPs, more than --max-annotation-atoms {int(max_annotation_atoms)}. Compose "
                f"works on the combinations, so its memory and time grow with their number: each SNPs x combinations "
                f"array would take {gib:.1f} GiB and the composition holds about ten of them. Use fewer or coarser "
                "columns (for example the MAF/LD bins and a few functional columns), or raise --max-annotation-atoms "
                "if the memory is available.")
        cls_mat = np.zeros((col_mat.shape[0], n_atoms))
        cls_mat[np.flatnonzero(in_any), np.asarray(atom).ravel()] = 1.0
        classes = [f"_atom{a}" for a in range(col_pat.shape[0])]
        log(f"[admixed-ld] compose: the {len(col_names)} annotation columns overlap; composed on their "
            f"{col_pat.shape[0]} disjoint combinations and summed into the columns")
    else:
        cls_mat, classes = col_mat, col_names
    to_cols = (lambda X: X) if col_pat is None else (lambda X: np.asarray(X) @ col_pat)   # atoms -> the user's columns
    out_classes = classes if col_pat is None else list(col_names)
    in_annot = cls_mat.sum(axis=1) == 1
    t_idx, r_idx, cls_mat = t_idx[in_annot], r_idx[in_annot], cls_mat[in_annot]
    order = np.argsort(r_idx, kind="stable")                       # reference (genome) order
    t_idx, r_idx, cls_mat = t_idx[order], r_idx[order], cls_mat[order]
    Pm = np.column_stack([rv[f"p_{s}"].to_numpy()[r_idx] for s in sources])
    Vm = np.column_stack([rv[f"v_{s}"].to_numpy()[r_idx] for s in sources])
    Dm = Pm[:, :-1] - Pm[:, -1:]                                    # contrasts against the baseline source
    good = np.isfinite(Pm).all(axis=1) & np.isfinite(Vm).all(axis=1) & (Vm.sum(axis=1) > 0)
    t_idx, r_idx, cls_mat = t_idx[good], r_idx[good], cls_mat[good]
    Pm, Vm, Dm = Pm[good], Vm[good], Dm[good]
    # products of contrasts without the sampling (co)variance of the frequencies (Var(p) = Var(dosage) / 4n), including
    # the panels' errors that a derived source's frequencies carry
    Vn, nn, Gn, _ = ref.frequency_noise(sources, Vm, r_idx, rv)
    S_noise, _ = contrast_noise(Vn, nn, Gn)
    O = unbiased_products_cov(Dm, S_noise)
    cls = np.argmax(cls_mat, axis=1)
    K = len(classes)
    M = r_idx.size
    if M == 0:
        raise ValueError("No GWAS variant is in the reference and the annotation.")
    snp_ids = tv.SNP.to_numpy()[t_idx]
    log(f"[admixed-ld] compose: {M} of {len(tv)} GWAS variants are in the reference ({', '.join(sources)}) and the "
        f"annotation ({K} classes)")
    # the partners: the GWAS SNPs themselves (the default), or every reference variant of the annotation, the GWAS SNPs
    # keeping the class of their own ID (prow maps each partner to its row, -1 for a partner off the GWAS)
    if dense:
        pc_mat, pclasses = read_annotation_classes(annot, rv.SNP.to_numpy())
        if list(pclasses) != list(classes):
            raise ValueError(f"{annot}: the classes differ between the GWAS SNPs and the reference variants.")
        pc_mat[r_idx] = cls_mat
        p_idx = np.flatnonzero(pc_mat.sum(axis=1) == 1)
        PmP = np.column_stack([rv[f"p_{s}"].to_numpy()[p_idx] for s in sources])
        VmP = np.column_stack([rv[f"v_{s}"].to_numpy()[p_idx] for s in sources])
        goodP = np.isfinite(PmP).all(axis=1) & np.isfinite(VmP).all(axis=1) & (VmP.sum(axis=1) > 0)
        p_idx, PmP, VmP = p_idx[goodP], PmP[goodP], VmP[goodP]
        DmP = PmP[:, :-1] - PmP[:, -1:]
        VnP, nnP, GnP, _ = ref.frequency_noise(sources, VmP, p_idx, rv)
        OP = unbiased_products_cov(DmP, contrast_noise(VnP, nnP, GnP)[0])
        clsP = np.argmax(pc_mat[p_idx], axis=1)
        n_listed = int((pc_mat.sum(axis=1) == 1).sum())
        log(f"[admixed-ld] compose: partners are the {p_idx.size} reference variants of the annotation ({n_listed} "
            f"listed, {len(rv) - n_listed} reference variants without a class are not partners); the {M} GWAS "
            "variants are the rows")
        if not np.isin(r_idx, p_idx).all():
            raise ValueError("Every GWAS variant must also be a partner.")

    # ---------------------------------------------------------------------------------------- parameters
    tables, n_med, p_rank = [], [], []
    for f in files:
        ss = Sumstats.from_file(f, name=os.path.basename(f), compute_diagnostics=False)
        ix = ss.index.get_indexer(snp_ids)
        t = pd.DataFrame({"N": np.where(ix >= 0, ss.n[ix], np.nan), "SE": np.where(ix >= 0, ss.se[ix], np.nan),
                          "BETA": np.where(ix >= 0, ss.beta[ix], np.nan)})
        tables.append((ss.name, t))
        n_med.append(float(np.median(ss.n)))
        p_rank.append(int(ss.cov_rank))
    N = float(np.median(n_med))
    p = int(np.median(p_rank))
    m_in, k_in = _parse_values(mean_ancestry), _parse_values(kappa0)
    flags = list(pre_flags)
    if _track_sumstats(files[0]):
        # one track's summary statistics (tractor-sumstats): their SEs follow the track variable, not the genotype, so
        # the SE regression for m and kappa_res(0) does not apply
        if m_in is None or (k_in is None and (theta_sd is None or nK != 2)):
            raise ValueError("The summary statistics are those of one local-ancestry track (tractor-sumstats); their "
                             "SEs do not give the cohort's mean ancestry and kappa_res(0). Pass --mean-ancestry (the "
                             "reader's tracks.json lists the mean local-ancestry share of each track) and --kappa0, or "
                             "--ancestry-sd for two sources.")
        if k_in is None:
            m0 = m_in[0] / (sum(m_in) if len(m_in) == nK else 1.0)
            k_in = [2.0 * m0 * (1.0 - m0) - 4.0 * float(theta_sd) ** 2]
            if not k_in[0] > 0:
                raise ValueError(f"--ancestry-sd {theta_sd} leaves no local-ancestry variance within people at mean "
                                 f"ancestry {m0:.3f}; pass --kappa0.")
            flags.append(dict(flag="kappa0_from_ancestry_sd", detail=f"kappa_res(0) = 2 m (1 - m) - 4 SD^2 = "
                                                                     f"{k_in[0]:.4g} (random mating, PCs taking up "
                                                                     "global ancestry)"))
            log(f"[admixed-ld] compose: kappa_res(0) {k_in[0]:.4g} from the mean ancestry and --ancestry-sd (random "
                "mating; the PCs take up global ancestry)")
    if m_in is None or k_in is None:
        m_se, k_se, per_trait = parameters_from_sumstats_k(tables, Vm, O, sources, log=log)
        flags += se_regression_outliers(per_trait, log)
    else:
        m_se, k_se, per_trait = np.full(nK, np.nan), np.full((k1, k1), np.nan), None
    if m_in is not None:
        if len(m_in) == k1:
            m_in = m_in + [1.0 - sum(m_in)]
        if len(m_in) != nK or min(m_in) <= 0:
            raise ValueError(f"--mean-ancestry needs {k1} or {nK} values in (0, 1).")
        m_vec = np.asarray(m_in) / np.sum(m_in)
    else:
        m_vec = np.asarray(m_se)
    if k_in is not None:
        pairs = contrast_pairs(k1)
        if len(k_in) != len(pairs):
            raise ValueError(f"--kappa0 needs {len(pairs)} values (the upper triangle of Kt_res(0), row by row).")
        k0 = np.zeros((k1, k1))
        for (a, b), v in zip(pairs, k_in):
            k0[a, b] = k0[b, a] = v
    else:
        k0 = np.asarray(k_se)
    nu_given = nu is not None
    nu = float(nu) if nu is not None else N - p - 1.0

    # ---------------------------------------------------------------------------------------- kernel and ancestry
    curve_info = None
    if curve is not None:
        cdf = pd.read_csv(f"{curve}.curve.tsv", sep="\t")
        with open(f"{curve}.curve.json") as fh:
            curve_info = json.load(fh)
        if list(curve_info.get("sources", sources)) != list(sources):
            raise ValueError(f"The curve was made with sources {curve_info.get('sources')}, not {sources}.")
        d_c = cdf.d_cm.to_numpy()
        kc = np.zeros((d_c.size, k1, k1))
        if k1 == 1:
            kc[:, 0, 0] = cdf.kernel.to_numpy()
            inter = np.array([[curve_info["inter_kernel"]]])
            lam = np.full(nK, curve_info["fit_2_to_20cm"]["lam"])
        else:
            for a, b in contrast_pairs(k1):
                kc[:, a, b] = kc[:, b, a] = cdf[f"kernel_{sources[a]}_{sources[b]}"].to_numpy()
            inter = np.asarray(curve_info["inter_matrix"])
            lam = np.array([curve_info["fit_per_source"][s]["lam"] for s in sources])
        kern = KernelMatrix.from_curve(d_c, kc, inter, k0, lam)
        c_rank = curve_info.get("covariate_rank")
        if c_rank is not None and p > 0 and int(c_rank) != p:
            flags.append(dict(flag="covariates_differ_from_curve",
                              detail=f"curve covariate rank {int(c_rank)}, GWAS COV_RANK {p}"))
            log(f"[admixed-ld] WARNING (recorded flag covariates_differ_from_curve): the curve was made with {int(c_rank)} "
                f"covariates and the GWAS reports {p}. The curve must use the GWAS's own covariates (and its PCs).")
        if sfi is not None and not sf_from_release:
            # a statistic given to compose (the remedy of the flag: the trait's own people): the share of its people
            flags += release_coverage(tables, n_med, sfi.get("n_samples"), min_release_share, log, against="statistic")
        else:
            flags += release_coverage(tables, n_med, curve_info.get("n_samples"), min_release_share, log)
    elif generations is not None:
        kern = KernelMatrix.from_decay(k0, float(generations) / 100.0, ref.genome_cm, nK,
                                       source=f"admixture {generations} generations ago")
    else:
        kern = KernelMatrix.from_decay(k0, float(decay_per_cm), ref.genome_cm, nK, source=f"decay {decay_per_cm} per cM")
    # curve mode by default: with a curve, its background term and cell matrix whenever it has them ('auto', the
    # command line's default); 'none' / False leave them out
    if str(relatedness) == "auto":
        relatedness = "curve" if (curve_info is not None and "background" in curve_info) else None
    elif str(relatedness).lower() == "none":
        relatedness = None
    if isinstance(cell_matrix, str) and cell_matrix == "auto":
        cell_matrix = bool(curve_info is not None and "cells" in curve_info and not dense)
    if curve_info is not None:
        log(f"[admixed-ld] compose: curve mode; background term {'from the curve' if relatedness else 'none'}, cell "
            f"matrix {'used' if cell_matrix else 'not used'}")
    flags = flags + check_kernel_join(kern, allow_flat_kernel_join, log)
    for n_, s in enumerate(sources):           # panels whose own admixture LD was never measured (clean-panels)
        src_m = ref.manifest["sources"][si[n_]]
        if "panel_admixture" not in src_m and not (src_m.get("derived") and src_m.get("ld") != "panel"):
            flags.append(dict(flag="panel_admixture_unmeasured", detail=s))
            log(f"[admixed-ld] WARNING (recorded flag panel_admixture_unmeasured): the {s} panel's own admixture LD "
                "was not measured; run `summit admixed-ld clean-panels` (it removes it when its gate passes)")
    if theta_sd is not None:
        th = dirichlet_moments(m_vec, theta_sd)
        th_source = f"Beta/Dirichlet with mean {np.round(m_vec, 3).tolist()} and SD {theta_sd} (first source)"
    elif curve_info is not None:
        th, th_source = curve_info["theta"], "the cohort's curve"
    else:                                   # date mode without a published SD: the SEs' m and kappa_res(0)
        sd_id = ancestry_sd_from_kernel(float(m_vec[0]), float(k0[0, 0]))
        th = dirichlet_moments(m_vec, sd_id)
        th_source = (f"Beta/Dirichlet with mean {np.round(m_vec, 3).tolist()} and SD {sd_id:.4f} (first source) from "
                     "m and kappa_res(0), Var = (2 m (1 - m) - kappa_res(0)) / 4")
        flags.append(dict(flag="ancestry_sd_from_identity", detail=f"global-ancestry SD {sd_id:.4f} from m and "
                          "kappa_res(0) under random mating (a lower bound when mating is assortative)"))
        log(f"[admixed-ld] WARNING (recorded flag ancestry_sd_from_identity): global-ancestry SD {sd_id:.4f} from m and "
            "kappa_res(0) (random mating; assortative mating makes the true SD larger). The moments set the "
            "same-person term, whose share of LD mass grows with SNPs per person (21% of the exact in-sample mass in "
            "UK Biobank t1); pass --ancestry-sd with a published value when the cohort reports one.")
    mom = moments_tensors(th, k1)
    rel_sum, rel_prof = None, None          # optional input: uniform sum of g^2, or the curve's profile
    if relatedness is None:
        pass
    elif str(relatedness) in ("curve", "uniform"):
        if curve_info is None or ("relatedness" not in curve_info
                                  and not (str(relatedness) == "curve" and "background" in curve_info)):
            raise ValueError(f"--relatedness {relatedness} needs a curve made with --relative-pairs.")
        if str(relatedness) == "curve" and ("profile" in curve_info.get("relatedness", {})
                                            or "background" in curve_info):
            rel_prof = RelatednessProfile.from_curve(curve, curve_info, smooth=relatedness_smooth,
                                                     background_min_cm=background_min_cm,
                                                     background_window=background_window)
            log("[admixed-ld] compose: relatives' term from the curve's "
                + ("all-pairs background at every distance" if isinstance(rel_prof, SmoothBackground) else "profile")
                + f", B = {rel_prof.between:.4g} between chromosomes, phi(d) within ("
                + ", ".join(f"{k} cM {v:.4g}" for k, v in rel_prof.describe()["phi_at"].items()) + ")")
        else:
            if str(relatedness) == "curve":
                if int(curve_info["relatedness"].get("pairs", -1)) == 0:   # listed pairs, none in the sample: no term
                    flags.append(dict(flag="relatedness_no_pairs", detail="no listed relative pair in the sample"))
                    log("[admixed-ld] WARNING (recorded flag relatedness_no_pairs): no listed relative pair is in the "
                        "sample, so the relatives' term is zero")
                else:
                    raise ValueError("--relatedness curve needs a curve with a relatedness profile or --background; "
                                     "this curve has neither. Pass --relatedness uniform for "
                                     "the uniform sum of g^2, or remake the curve.")
            rel_sum = float(curve_info["relatedness"]["sum_squared_ordered"])
    else:
        rel_sum = float(relatedness)
    # the curve's relatives' sums run over the ordered pairs of its own people (a number given by the user is the
    # GWAS's own sum and is not rescaled)
    rel_scale, rel_scale_info = 1.0, None
    if (rel_prof is not None or rel_sum is not None) and str(relatedness) in ("curve", "uniform"):
        if "n_samples" not in curve_info:
            raise ValueError("The curve does not record its number of people (n_samples), which the relatives' term "
                             "needs; remake the curve.")
        n_curve = int(curve_info["n_samples"])
        rel_scale = relatedness_scale(n_curve, N)
        rel_scale_info = dict(curve_people=n_curve, gwas_n=float(N), factor=float(rel_scale),
                              definition="N (N - 1) / (n (n - 1)): the share of the curve's ordered pairs of people "
                                         "that the GWAS holds, its people taken as a random subset of the curve's "
                                         "(or the curve's of the GWAS's)")
        log(f"[admixed-ld] compose: the curve's relatives' sums are over its {n_curve} people; times "
            f"{rel_scale:.6g} for the GWAS's N {N:.0f}")
        if abs(rel_scale - 1.0) > 0.01:
            flags.append(dict(flag="relatedness_curve_sample_differs",
                              detail=f"curve of {n_curve} people, GWAS N {N:.0f}: the relatives' term is scaled by "
                                     f"{rel_scale:.4g}, which assumes the GWAS's people are a random subset of the "
                                     "curve's (or the reverse)"))
            log(f"[admixed-ld] WARNING (recorded flag relatedness_curve_sample_differs): the curve was made on "
                f"{n_curve} people and the GWAS has N {N:.0f}; the relatives' term is scaled by {rel_scale:.4g}, "
                "the GWAS's people taken as a random subset of the curve's. Make the curve on the GWAS's people "
                "where possible.")
    # the panels' own background LD, which their unbiased squared covariances hold inside the band: removed when the
    # cohort's background is applied at every distance (background_window 'panels')
    panel_bg = {}
    if isinstance(rel_prof, SmoothBackground):
        missing = []
        for n_, s in enumerate(sources):
            src_m = ref.manifest["sources"][si[n_]]
            rec = src_m.get("panel_background")
            if rec is not None:
                panel_bg[n_] = SmoothBackground.from_record(rec, fit_min_cm=background_min_cm,
                                                            smooth=relatedness_smooth, name=s)
            elif src_m.get("derived") and src_m.get("ld") != "panel":
                log(f"[admixed-ld] note: {s} is a derived source with the cohort's own window covariances; no panel "
                    "background to remove")
            else:
                missing.append(s)
        if missing:
            raise ValueError("--relatedness curve with the cohort's background at every distance needs each panel's "
                             f"own background, which {ref_path} does not record for {', '.join(missing)}: run "
                             "`summit admixed-ld clean-panels` (it measures it; for a derived source with a panel's "
                             "LD give its anchor panel), or pass --relatedness none to compose without the "
                             "background term.")
        log("[admixed-ld] compose: the panels' own background LD is removed inside the band ("
            + ", ".join(f"{sources[n_]}: {pb.per_pair(2.0):.3g} per SNP pair at 2 cM, n {pb.n}"
                        for n_, pb in panel_bg.items()) + ")")

    # ---------------------------------------------------------------------------------------- the plan
    have = dict(curve=curve is not None, statistic=sample_factors is not None, cells=bool(cell_matrix),
                relatives=rel_prof is not None or rel_sum is not None,
                background=isinstance(rel_prof, SmoothBackground))
    plan_groups = (_n_groups(tables, n_med, N, nu, p, nu_given, n_tol) if per_trait_n
                   else [dict(label=f"{N:.0f}", N=float(N), nu=float(nu), traits=[(t[0], n) for t, n in zip(tables, n_med)])])
    plan = dict(
        mode="curve" if curve is not None else "date" if generations is not None else "decay rate",
        decay=(f"the curve {curve}" if curve is not None else f"{generations} generations" if generations is not None
               else f"{decay_per_cm} per cM"),
        same_person_term=("the sample statistic released with the curve" if sf_from_release
                          else f"the sample statistic {sample_factors}" if sample_factors is not None
                          else "random mating"),
        relatives=("the curve's all-pairs background" if have["background"]
                   else "the curve's relatives' profile" if rel_prof is not None
                   else f"a uniform sum of {rel_sum:.4g}" if rel_sum is not None else "none"),
        cell_matrix=bool(cell_matrix),
        ancestry=("given" if (m_in is not None or k_in is not None) else "from the traits' standard errors (median)"),
        mean_ancestry=[float(x) for x in np.atleast_1d(m_vec)],
        traits=len(tables), sample_sizes=[dict(N=float(g["N"]), nu=float(g["nu"]), traits=len(g["traits"]))
                                          for g in plan_groups],
        # each trait's LD-score file is known only after composing (sample sizes whose LD masses agree within 0.1%
        # share the main file); the params json's plan lists them
        files=None,
        components=have, flags=[f["flag"] for f in flags])
    sizes_txt = ", ".join("N {:.0f}".format(g["N"]) for g in plan_groups)
    log("[admixed-ld] compose plan: " + "; ".join([
        f"{plan['mode']} mode, decay from {plan['decay']}",
        f"same-person term from {plan['same_person_term']}",
        f"relatives and background from {plan['relatives']}",
        f"cell matrix {'used' if cell_matrix else 'not used'}",
        f"ancestry {plan['ancestry']}",
        f"{len(tables)} trait(s) in {len(plan_groups)} GWAS sample size(s) ({sizes_txt})",
        f"flags so far: {', '.join(plan['flags']) or 'none'}"]))
    if require:
        want = [x.strip() for x in (require.split(",") if isinstance(require, str) else require) if x.strip()]
        bad = [x for x in want if x not in have]
        if bad:
            raise ValueError(f"--require: unknown component(s) {bad}; choose among {', '.join(have)}.")
        missing = [x for x in want if not have[x]]
        if missing:
            raise ValueError(f"--require {','.join(want)}: this composition would run without {', '.join(missing)} "
                             f"(plan: {plan['mode']} mode, same-person term from {plan['same_person_term']}, relatives "
                             f"from {plan['relatives']}, cell matrix {'used' if cell_matrix else 'not used'}).")
    if dry_run:
        with open(f"{out}.plan.json", "w") as fh:
            json.dump(plan, fh, indent=1)
        log(f"[admixed-ld] compose: --dry-run, wrote {out}.plan.json and stopped before composing")
        return plan

    # ---------------------------------------------------------------------------------------- per-SNP model
    quad = lambda Mx: np.einsum("jab,ab->j", O, Mx)                 # D_j' Mx D_j without sampling noise
    cres = Vm @ m_vec + quad(k0)
    s2 = cres + 4.0 * r2 * quad(mom[1])
    ok = cres > 1e-8
    if not ok.all():
        # a source derived from a cohort carries unbiased dosage variances, which can be slightly negative where its
        # frequency is estimated below 0 or above 1; where every source is nearly fixed the modelled variance can then
        # be zero or negative, and such a SNP gets no LD score
        if not ok.any():
            raise ValueError("No SNP has a positive variance in the model.")
        log(f"[admixed-ld] compose: {int((~ok).sum())} SNPs without positive variance in the model (nearly fixed in "
            "every source) are left out")
        t_idx, r_idx, cls_mat, cls = t_idx[ok], r_idx[ok], cls_mat[ok], cls[ok]
        Pm, Vm, Dm, O, snp_ids, cres, s2 = Pm[ok], Vm[ok], Dm[ok], O[ok], snp_ids[ok], cres[ok], s2[ok]
        M = int(ok.sum())
    Vpop = cres / s2
    if dense:                                                      # the same model over the partners
        quadP = lambda Mx: np.einsum("jab,ab->j", OP, Mx)
        cresP = VmP @ m_vec + quadP(k0)
        s2P = cresP + 4.0 * r2 * quadP(mom[1])
        okP = cresP > 1e-8
        p_idx, PmP, VmP, DmP, OP, clsP, cresP, s2P = (x[okP] for x in (p_idx, PmP, VmP, DmP, OP, clsP, cresP, s2P))
        row_of = np.searchsorted(p_idx, r_idx)                     # each row's partner index
        if not (row_of < p_idx.size).all() or not (p_idx[np.minimum(row_of, p_idx.size - 1)] == r_idx).all():
            raise ValueError("Every GWAS variant must also be a partner.")
        prow = -np.ones(p_idx.size, dtype=np.int64)
        prow[row_of] = np.arange(M)
    else:
        p_idx, PmP, VmP, DmP, OP, clsP, cresP, s2P, prow = r_idx, Pm, Vm, Dm, O, cls, cres, s2, None
    MP = int(p_idx.size)
    VpopP = cresP / s2P
    chromP = rv["CHR"].to_numpy()[p_idx]
    cmP = rv["CM"].to_numpy()[p_idx]
    L = np.zeros((M, K))
    Lx = np.zeros((M, K)) if xs_i is not None else None
    v2_out = np.zeros(M) if xs_i is not None else None
    chrom_out = rv["CHR"].to_numpy()[r_idx]
    local_out = rv["LOCAL"].to_numpy()[r_idx]
    cm_out = rv["CM"].to_numpy()[r_idx]
    k2 = k1 * k1
    Oflat = O.reshape(M, k2)                                       # o_j[u], u = (a, c): D_ja D_jc
    Wn = Oflat / s2[:, None]                                       # partner weights of the admixture terms
    WnP = Wn if not dense else OP.reshape(MP, k2) / s2P[:, None]
    W_tot = np.stack([np.bincount(clsP, weights=WnP[:, u], minlength=K) for u in range(k2)])   # k2 x K
    uu = [(a, c) for a in range(k1) for c in range(k1)]
    far_F = np.array([[kern.far[a, b] * kern.far[c, e] for (b, e) in uu] for (a, c) in uu])  # F_uv(inf)
    # a source derived by inversion carries minus the known panels' covariance errors (Reference.covariance_noise_mix):
    # with the anchor cohort's pi^a, Cov(e_N, e_t) = -pi^a_t(d) Var(e_t) / pi^a_N(d), so the composed covariance's
    # noise variance has the cross term -2 pi_N(d) pi_t(d) pi^a_t(d) / pi^a_N(d) Var(e_t) for every known panel t that
    # is also composed. Without it the panels' noise is removed twice (once in covvar_N, once in their own terms).
    cross = []                                                     # (n, t, m^a_n, lam^a_n, m^a_t, lam^a_t)
    for n_, s in enumerate(sources):
        mix = ref.covariance_noise_mix(s)
        if mix is None:
            continue
        (ma_n, la_n), others = mix
        for t, (ma_t, la_t) in others.items():
            if t in sources:
                cross.append((n_, sources.index(t), ma_n, la_n, ma_t, la_t))
    if cross:
        log("[admixed-ld] compose: window covariances of " + ", ".join(sorted({sources[x[0]] for x in cross}))
            + " were derived by inversion; their covariance with the errors of "
            + ", ".join(sorted({sources[x[1]] for x in cross})) + " is removed once")
    # a derived source with a panel's window LD: the panel's covariances on the source's scale (panel_variance_scale)
    z0 = ref.load(ref.chromosomes[0])
    fsc, fsc_info = {}, {}
    for n_, i in enumerate(si):
        if f"vld_{i}" in z0.files:
            fsc[n_], fsc_info[sources[n_]] = panel_variance_scale(ref, i, ns[n_])
            q = fsc_info[sources[n_]]
            log(f"[admixed-ld] compose: {sources[n_]}'s window covariances from its {q['n_panel']}-person panel put on "
                f"its own variances (shrinkage weight median {q['shrinkage_weight_median']:.3f}: 0 keeps the panel's "
                f"covariances, 1 transfers its correlations; mean f^2 {q['mean_f2']:.4f})")
    T = None
    if tracks:
        vxP = (None if xs_i is None else
               ref.variants([cross_source])[f"v_{cross_source}"].to_numpy(np.float64)[p_idx])
        T = _TrackScores(sources, m_vec, VmP, PmP, s2P, clsP, M, K, track_alpha, cross_source, xs_n, vxP)
    localP = rv["LOCAL"].to_numpy()[p_idx]

    def scatter(acc_, r, cl, w):
        """acc_[r, cl] += w over the pairs (r: the receiving row; partners off the GWAS, r < 0, receive nothing)."""
        if prow is not None:
            m_ = r >= 0
            r, cl, w = r[m_], cl[m_], w[m_]
        acc_ += np.bincount(r * K + cl, weights=w, minlength=M * K).reshape(M, K)

    n_pairs = 0
    for c in ref.chromosomes:
        sel = np.flatnonzero(chrom_out == c)
        if sel.size == 0:
            continue
        selP = sel if not dense else np.flatnonzero(chromP == c)
        z = ref.load(c)
        nloc = z["bp"].size
        tmap = -np.ones(nloc, dtype=np.int64)                      # local position -> partner (the row by default)
        tmap[localP[selP]] = selP
        indptr, indices, cm_ref = z["indptr"], z["indices"], z["cm"]
        cov = [z[f"cov_{i}"] for i in si]
        # a derived source carries the sampling variance of each covariance (covvar) instead of the panel moments
        cvar = [z[f"covvar_{i}"] if f"covvar_{i}" in z.files else None for i in si]
        m22 = [z[f"m22_{i}"] if cvar[n_] is None else None for n_, i in enumerate(si)]
        # the variances of the people behind each covariance (a derived source with a small panel's LD keeps that
        # panel's variances apart from the dosage variances its frequencies imply)
        vr = [z[f"vld_{i}"] if f"vld_{i}" in z.files else z[f"v_{i}"] for i in si]
        cross_c = [x for x in cross if cvar[x[0]] is not None]          # only covariances stored with covvar
        if xs_i is not None:                                            # the second population's panel
            cov_x, v_x = z[f"cov_{xs_i}"], z[f"v_{xs_i}"].astype(np.float64)
            v2_out[sel] = v_x[local_out[sel]]
            m22_x = z[f"m22_{xs_i}"] if (T is not None and xs_n is None) else None
        # pairs within the window (upper band), in chunks of rows
        r0 = 0
        while r0 < nloc:
            r1 = int(np.searchsorted(indptr, indptr[r0] + pair_chunk, side="right")) - 1
            r1 = min(max(r1, r0 + 1), nloc)
            s0, s1 = int(indptr[r0]), int(indptr[r1])
            if s1 > s0:
                cnt = np.diff(indptr[r0:r1 + 1])
                jl = np.repeat(np.arange(r0, r1, dtype=np.int64), cnt)
                kl = indices[s0:s1].astype(np.int64)
                j, k = tmap[jl], tmap[kl]
                use = (j >= 0) & (k >= 0)
                if prow is not None:                                # a pair needs a row at one end at least
                    use &= (prow[np.maximum(j, 0)] >= 0) | (prow[np.maximum(k, 0)] >= 0)
                if use.any():
                    j, k, jl, kl = j[use], k[use], jl[use], kl[use]
                    rj, rk = (j, k) if prow is None else (prow[j], prow[k])
                    d = np.abs(cm_ref[kl] - cm_ref[jl])
                    Kd = kern(d)                                    # pairs x k1 x k1
                    adm = np.einsum("pa,pab,pb->p", DmP[j], Kd, DmP[k])
                    cr = adm.copy()
                    pis, vnss, css, bgs = [None] * nK, [None] * nK, [None] * nK, [0.0] * nK
                    for s in range(nK):
                        cs_ = cov[s][s0:s1][use].astype(np.float64)
                        pi_s = m_vec[s] * np.exp(-kern.lam[s] * d)
                        # sampling noise of the squared panel covariance: c^2 minus its unbiased estimate
                        if cvar[s] is not None:
                            vns = cvar[s][s0:s1][use].astype(np.float64)
                        else:
                            vns = cs_ * cs_ - unbiased_squared_covariance(cs_, m22[s][s0:s1][use],
                                                                          vr[s][jl] * vr[s][kl], ns[s])
                        if s in fsc:                              # panel covariances on the source's scale
                            g = fsc[s][c][jl] * fsc[s][c][kl]
                            cs_, vns = cs_ * g, vns * g * g
                        cr += pi_s * cs_
                        pis[s], vnss[s], css[s] = pi_s, vns, cs_
                    noise = window_noise(pis, vnss, cross_c, d)
                    for n_, pb in panel_bg.items():               # the panel's own background (SmoothBackground)
                        bg = (pis[n_] * pis[n_] * vr[n_][jl].astype(np.float64) * vr[n_][kl].astype(np.float64)
                              * pb.per_pair(d))
                        if n_ in fsc:                             # on the source's scale, as its covariances
                            bg *= (fsc[n_][c][jl] * fsc[n_][c][kl]) ** 2
                        noise += bg
                        bgs[n_] = bg
                    if xs_i is not None:
                        c2 = cov_x[s0:s1][use].astype(np.float64)
                        v2j, v2k = v_x[jl], v_x[kl]
                        px = cr * c2
                        if xs_n is not None:                      # pi_s c_s^2 of the shared panel: unbiased, less
                            px -= pis[xs_n] * vnss[xs_n]          # its own background
                            if xs_n in panel_bg:
                                px -= pis[xs_n] * v2j * v2k * panel_bg[xs_n].per_pair(d)
                        okx = (v2j > 0) & (v2k > 0)
                        px = np.where(okx, px, 0.0)
                        dj = np.sqrt(np.where(okx, cresP[j] * v2j * s2P[k] * v2k, 1.0))
                        dk = np.sqrt(np.where(okx, cresP[k] * v2k * s2P[j] * v2j, 1.0))
                        scatter(Lx, rj, clsP[k], px / dj)
                        scatter(Lx, rk, clsP[j], px / dk)
                    if T is not None:
                        xpanel = None
                        if xs_i is not None:                      # X's panel: Var(c_X) and its own background
                            vnx = (vnss[xs_n] if xs_n is not None else
                                   c2 * c2 - unbiased_squared_covariance(c2, m22_x[s0:s1][use], v2j * v2k,
                                                                         float(ref.n[cross_source])))
                            bgx = (v2j * v2k * panel_bg[xs_n].per_pair(d) if (xs_n is not None and xs_n in panel_bg)
                                   else 0.0)
                            xpanel = (c2, v2j, v2k, vnx, bgx)
                        T.add_pairs(scatter, j, k, rj, rk, d, pis, css, vnss, bgs, cross_c, xpanel)
                    sq = cr * cr - noise - adm * adm
                    scatter(L, rj, clsP[k], sq / (cresP[j] * s2P[k]))
                    scatter(L, rk, clsP[j], sq / (cresP[k] * s2P[j]))
                    n_pairs += int(use.sum())
            r0 = r1
        # admixture LD of every pair on the chromosome: (D_j' Kt D_k)^2 = sum_{u,v} o_j[u] F_uv(d) o_k[v] with
        # F_{(a,c),(b,e)} = Kt_ab Kt_ce, by FFT convolution on a grid (over the partners; read at the rows)
        x = cmP[selP]
        x0 = x.min()
        gP = np.floor((x - x0) / grid_cm).astype(np.int64)
        g = gP if not dense else np.floor((cm_out[sel] - x0) / grid_cm).astype(np.int64)
        ng = int(gP.max()) + 1
        grid_d = np.arange(ng) * grid_cm
        kg = kern(grid_d)                                           # ng x k1 x k1
        Wg = np.zeros((k2, ng, K))
        for v in range(k2):
            np.add.at(Wg[v], (gP, clsP[selP]), WnP[selP, v])
        acc = np.zeros((sel.size, K))
        self_ = np.zeros(sel.size)
        for ui, (a, c_) in enumerate(uu):
            Fu = np.zeros((sel.size, K))
            for vi, (b, e) in enumerate(uu):
                fk = kg[:, a, b] * kg[:, c_, e]
                Fu += _fft_conv_sym(Wg[vi], fk)[g]
                self_ += Oflat[sel, ui] * fk[0] * Wn[sel, vi]
                Fu += far_F[ui, vi] * (W_tot[vi] - Wg[vi].sum(axis=0))[None, :]
            acc += Oflat[sel, ui][:, None] * Fu
        acc[np.arange(sel.size), cls[sel]] -= self_                 # the convolution includes each SNP itself
        L[sel] += acc / cres[sel][:, None]
        log(f"[admixed-ld] compose chr{c}: {sel.size} SNPs" + (f", {selP.size} partners" if dense else "")
            + f" ({time.perf_counter() - t0:.0f} s)")
    cells_used = None
    if cell_matrix:
        if curve is None:
            raise ValueError("--cell-matrix needs --curve (the curve writes the cell matrix).")
        corr, cells_used = _cell_correction(f"{curve}.cells.npz", kern, chrom_out, cm_out, Oflat, Wn, cls, K, k1,
                                            cell_switch_cm, log)
        L += corr / cres[:, None]
    # same-person term, each SNP with itself, relatives
    pbar = Pm @ m_vec                                               # frequency at the mean ancestry
    beta_ = 2.0 * Dm * (1.0 - 2.0 * pbar)[:, None]
    if dense:                                                       # the class sums run over the partners
        pbarP = PmP @ m_vec
        sp_rm = same_person_term(beta_, O, s2, cres, cls, K, mom, nu,
                                 partners=(2.0 * DmP * (1.0 - 2.0 * pbarP)[:, None], OP, s2P, clsP))
    else:
        sp_rm = same_person_term(beta_, O, s2, cres, cls, K, mom, nu)
    sf_info = None
    Vpop_sp = cres / s2
    if sample_factors is not None:
        from .sample_factors import factor_between_chromosome_mass, factor_loadings, factor_same_person_term
        if list(sfi["sources"]) != list(sources):
            raise ValueError(f"The sample factors were made with sources {sfi['sources']}, compose uses {sources} (same "
                             "names, same order, baseline last).")
        a_sf = factor_loadings(Vm, O)
        sp = factor_same_person_term(a_sf, s2, cres, cls, K, sfi["sigma"], sfi["n_samples"], sfi["nu"], nu)
        mass_uncal = float((Vpop_sp[:, None] * sp).sum())
        calib = None
        realized = (sfi.get("diagnostics") or {}).get("between_chromosome_same_person_mass")
        if sample_factors_calibrate and realized is not None:
            # the cohort's realized between-chromosome same-person mass fixes the term's size; the factors allocate it
            # (a signed loading sum makes the term sensitive to small errors of the model variances s2 that follow the
            # sources' heterozygosity difference, and the factors themselves leave a few per cent out)
            # compared at the statistic's own nu, so that the factor is a pure model correction for any GWAS sample:
            # the term sp, at the GWAS's nu, keeps its 1 / nu scaling (realized and model mass both at nu_sf)
            btw = factor_between_chromosome_mass(a_sf, s2, chrom_out, sfi["sigma"], sfi["n_samples"], sfi["nu"],
                                                 float(sfi["nu"]))
            m_ratio = M / float(sfi["n_snps"])
            target = float(realized) * m_ratio ** 2
            if btw > 0 and target > 0:
                f = target / btw
                sp = sp * f
                calib = dict(factor=f, compose_between_chromosome_mass=btw, realized_between_chromosome_mass=realized,
                             snp_ratio=m_ratio, target=target)
                log(f"[admixed-ld] compose: sample-factor term scaled by {f:.4f} so that its between-chromosome mass "
                    f"equals the cohort's realized {realized:.6g} (x {m_ratio ** 2:.4f} for the SNP count)")
                if abs(m_ratio - 1) > 0.02:
                    flags.append(dict(flag="sample_factors_snps_differ",
                                      detail=f"{M} SNPs here, {sfi['n_snps']} in the sample factors"))
                if not 0.8 <= f <= 1.25:
                    flags.append(dict(flag="sample_factors_calibration_large", detail=f"factor {f:.4g}"))
            else:
                flags.append(dict(flag="sample_factors_not_calibrated",
                                  detail=f"between-chromosome masses {btw:.4g} (compose), {realized:.4g} (realized)"))
        sf_info = dict(file=sfi["path"], from_release=bool(sf_from_release), n_samples=sfi["n_samples"], nu=sfi["nu"], factors=sfi["factors"],
                       sigma=sfi["sigma"], diagnostics=sfi.get("diagnostics"), leverage=sfi.get("leverage"),
                       calibration=calib, same_person_mass_uncalibrated=mass_uncal,
                       same_person_mass=float((Vpop_sp[:, None] * sp).sum()),
                       same_person_mass_random_mating=float((Vpop_sp[:, None] * sp_rm).sum()))
        ev = np.linalg.eigvalsh(np.asarray(sfi["sigma"], np.float64))
        if ev.min() < 0:                                            # the split-half estimate need not be PSD
            flags.append(dict(flag="sample_factors_not_psd",
                              detail=f"smallest eigenvalue of Sigma_u {ev.min():.3g} (largest {ev.max():.3g}); used as "
                                     "estimated"))
            log(f"[admixed-ld] note (recorded flag sample_factors_not_psd): Sigma_u has the eigenvalue {ev.min():.3g} "
                f"(largest {ev.max():.3g}); the split-half estimate is used as it is")
        n_rel_people = curve_info.get("n_samples") if curve_info is not None else None
        if abs(sfi["n_samples"] - N) > 0.01 * N and n_rel_people is not None and int(sfi["n_samples"]) == int(n_rel_people):
            # the cohort's release (curve and statistic of the same people): the term is scaled to each GWAS's N, and
            # GWAS samples that hold much less of the release's people are flagged by gwas_sample_smaller_than_release
            flags.append(dict(flag="sample_factors_from_release",
                              detail=f"sample statistic of the curve's {sfi['n_samples']} people, GWAS N {N:.0f}"))
            log(f"[admixed-ld] compose: the sample statistic is the release's ({sfi['n_samples']} people, as the "
                f"curve); the same-person term is scaled to the GWAS's N {N:.0f}")
        elif abs(sfi["n_samples"] - N) > 0.01 * N:
            flags.append(dict(flag="sample_factors_sample_differs",
                              detail=f"sample factors of {sfi['n_samples']} people, GWAS N {N:.0f}"))
            log(f"[admixed-ld] WARNING (recorded flag sample_factors_sample_differs): the sample factors were made on "
                f"{sfi['n_samples']} people and the GWAS has N {N:.0f}; make them on the GWAS's people.")
        log(f"[admixed-ld] compose: same-person term from the sample factors ({sfi['path']}): LD mass "
            f"{sf_info['same_person_mass']:.6g}, against {sf_info['same_person_mass_random_mating']:.6g} under random "
            "mating")
    else:
        sp = sp_rm
    # the GWAS-sample terms: the same-person term (1 / nu) and the relatives' term (f(N) / nu^2); the rest is a population
    # quantity shared by every GWAS sample of the cohort
    L[np.arange(M), cls] += Vpop
    R1 = None
    if rel_sum is not None or rel_prof is not None:
        R1 = np.zeros((M, K))
        if rel_sum is not None:
            R1 += rel_sum * np.bincount(clsP, weights=VpopP, minlength=K)[None, :]
        if rel_prof is not None:
            relatedness_term(R1, rel_prof, chrom_out, cm_out, cls, Vpop, K, 1.0, grid_cm, scale=1.0,
                             partners=None if not dense else (chromP, cmP, clsP, VpopP))
    L_pop = L
    mass_check = None
    if sample_factors is not None and sfi.get("ld_mass"):
        # the cohort's realized in-sample LD mass (total) against this composition for the statistic's own sample
        n_sf, nu_sf = float(sfi["n_samples"]), float(sfi["nu"])
        n_curve_c = int(curve_info["n_samples"]) if (R1 is not None and curve_info is not None
                                                     and "n_samples" in curve_info) else None
        f_sf = relatedness_scale(n_curve_c, n_sf) if (R1 is not None and n_curve_c) else rel_scale
        L_sf = L_pop + sp * (nu / nu_sf) + (0.0 if R1 is None else (f_sf / nu_sf ** 2) * R1)
        comp_tot = float((Vpop[:, None] * L_sf).sum())
        m_ratio = M / float(sfi["n_snps"])
        lm = sfi["ld_mass"]
        real_tot = float(lm["total"]) * m_ratio ** 2
        real_btw = float(lm["between_chromosomes"]) * m_ratio ** 2
        method = str(lm.get("method", "exact"))                     # statistics before the probes were exact
        stoch = method == "stochastic"
        se_tot = float(lm["se_total"]) * m_ratio ** 2 if stoch and lm.get("se_total") is not None else None
        se_btw = (float(lm["se_between_chromosomes"]) * m_ratio ** 2
                  if stoch and lm.get("se_between_chromosomes") is not None else None)
        mass_check = dict(composed_total=comp_tot, realized_total=real_tot, ratio=comp_tot / real_tot,
                          realized_between_chromosomes=real_btw, method=method, probes=lm.get("probes"),
                          se_realized_total=se_tot, se_realized_between_chromosomes=se_btw)
        log(f"[admixed-ld] compose: LD mass {comp_tot:.6g} for the sample factors' {n_sf:.0f} people, against their "
            f"realized {real_tot:.6g} ({comp_tot / real_tot:.4f}"
            + (f"; stochastic, SE {se_tot:.3g}" if stoch and se_tot is not None else "") + ")")
        # a stochastic realized mass carries its probes' error: the difference must also exceed 2 SE
        beyond_noise = not stoch or (se_tot is not None and abs(comp_tot - real_tot) > 2.0 * se_tot)
        if abs(comp_tot / real_tot - 1) > 0.05 and beyond_noise:
            flags.append(dict(flag="ld_mass_differs_from_cohort", detail=f"composed / realized {comp_tot / real_tot:.4f}"
                              + (f" (realized mass stochastic, SE {se_tot:.4g})" if stoch else "")))
            log(f"[admixed-ld] WARNING (recorded flag ld_mass_differs_from_cohort): the composed LD mass is "
                f"{comp_tot / real_tot:.3f} of the cohort's realized in-sample LD mass")
        shift_skip = None
        if calibrate_between_chromosomes and stoch and not (se_tot is not None and se_tot <= 0.1 * abs(real_btw)):
            # the shift puts the whole difference on pairs on different chromosomes, and with it the probes' error
            shift_skip = (f"the realized LD mass is stochastic ({lm.get('probes')} probes) and the SE of its total "
                          + (f"({se_tot:.4g}) " if se_tot is not None else "(not recorded) ")
                          + f"is more than 10% of the realized between-chromosome mass (|{real_btw:.4g}|); more probes "
                            "(--ld-mass-probes, --sample-factors-probes) or the exact mass make it usable")
        if calibrate_between_chromosomes and shift_skip is not None:
            flags.append(dict(flag="between_chromosome_shift_skipped", detail=shift_skip))
            log(f"[admixed-ld] WARNING (recorded flag between_chromosome_shift_skipped): "
                f"--calibrate-between-chromosomes not applied: {shift_skip}")
        elif calibrate_between_chromosomes:
            # the remainder is put on pairs of SNPs on different chromosomes, uniformly per pair (as relatedness is):
            # where unmodelled structure (a source the panels do not represent) shows
            Vc_tot = np.bincount(cls, weights=Vpop, minlength=K)
            uch = pd.unique(chrom_out)
            Vc_chr = {c: np.bincount(cls[chrom_out == c], weights=Vpop[chrom_out == c], minlength=K) for c in uch}
            Vchr = {c: float(Vpop[chrom_out == c].sum()) for c in uch}
            pairs_btw = float(Vpop.sum() ** 2 - sum(v ** 2 for v in Vchr.values()))
            cshift = (real_tot - comp_tot) / pairs_btw
            shift = np.zeros((M, K))
            for c in uch:
                shift[chrom_out == c] = cshift * (Vc_tot - Vc_chr[c])[None, :]
            L_pop = L_pop + shift
            mass_check.update(between_chromosome_shift_per_pair=cshift, shift_mass=float(cshift * pairs_btw))
            log(f"[admixed-ld] compose: between-chromosome LD shifted by {cshift:.4g} per V-weighted pair "
                f"({cshift * pairs_btw:.6g} of mass) to the cohort's realized total")
    elif calibrate_between_chromosomes:
        # the shift needs the cohort's realized LD mass, which only a statistic made with it records
        why = (("the same-person term uses no sample statistic" if sample_factors is None
                else f"the sample statistic {sfi['path']} records no realized LD mass")
               + "; make the statistic with its LD mass (curve --sample-factors-ld-mass or --sample-factors-probes B, "
                 "or sample-factors without --no-ld-mass) and give it to compose")
        flags.append(dict(flag="between_chromosome_shift_skipped", detail=f"no realized LD mass: {why}"))
        log("[admixed-ld] WARNING (recorded flag between_chromosome_shift_skipped): --calibrate-between-chromosomes "
            f"not applied, no realized LD mass: {why}")
    L = L_pop + sp + (0.0 if R1 is None else (rel_scale / nu ** 2) * R1)
    V = Vpop
    groups_info, wrote_groups = None, False
    if nu_given and abs(nu - (N - p - 1.0)) > 0.02 * (N - p - 1.0):
        flags.append(dict(flag="nu_differs_from_gwas_n", detail=f"nu {nu:.0f} given; the traits' median N {N:.0f} and "
                                                                f"covariate rank {p} give {N - p - 1.0:.0f}"))
    if per_trait_n:
        groups_info = _n_groups(tables, n_med, N, nu, p, nu_given, n_tol)
        if len(groups_info) > 1:
            n_curve_rel = int(curve_info["n_samples"]) if (R1 is not None and curve_info is not None
                                                           and "n_samples" in curve_info) else None
            # the groups' files differ from the main one only in the sample terms; when no group's LD mass differs
            # from the main file's by 0.1% (large samples, few SNPs per person), the main file serves every trait
            mvec = lambda X: float((V[:, None] * X).sum())
            m_pop, m_sp = mvec(L_pop), mvec(sp)
            m_r1 = 0.0 if R1 is None else mvec(R1)
            m_main = m_pop + m_sp + (0.0 if R1 is None else rel_scale / nu ** 2 * m_r1)
            dev = max(abs(m_pop + m_sp * nu / g["nu"] + (0.0 if R1 is None else (
                relatedness_scale(n_curve_rel, g["N"]) if n_curve_rel else rel_scale) / g["nu"] ** 2 * m_r1)
                - m_main) / m_main for g in groups_info)
            if dev < 1e-3:
                log(f"[admixed-ld] compose: {len(groups_info)} GWAS sample sizes among the traits, but their LD masses "
                    f"differ from the median N's by at most {dev:.2e}; one file serves every trait")
                groups_info = [dict(g, file=f"{out}.gw.ldscore.gz", merged=True) for g in groups_info]
        if len(groups_info) > 1 and not groups_info[0].get("merged"):
            for g in groups_info:
                fg = relatedness_scale(n_curve_rel, g["N"]) if (R1 is not None and n_curve_rel) else rel_scale
                Lg = L_pop + sp * (nu / g["nu"]) + (0.0 if R1 is None else (fg / g["nu"] ** 2) * R1)
                g.update(relatedness_scale=float(fg), file=f"{out}.n{g['label']}.gw.ldscore.gz",
                         ld_mass=float((V[:, None] * Lg).sum()))
                snpdf_g = pd.DataFrame({"CHR": chrom_out, "SNP": snp_ids, "BP": rv["BP"].to_numpy()[r_idx]})
                write_gw_ldscore_table(f"{out}.n{g['label']}", snpdf_g, pd.DataFrame(to_cols(Lg), columns=out_classes),
                                       snp_var=V)
                write_gw_annotation_masses(f"{out}.n{g['label']}", to_cols(np.bincount(clsP, minlength=K).astype(float)))
            # the files by name, relative to the table's own directory (they are next to it), so that --h2 finds them
            # from any working directory and after the directory is moved
            tmp_tab = f"{out}.n_groups.tsv.{os.getpid()}.tmp"
            pd.DataFrame([dict(trait=t, N=nn, ldscores=os.path.basename(g["file"])) for g in groups_info
                          for t, nn in g["traits"]]).to_csv(tmp_tab, sep="\t", index=False)
            os.replace(tmp_tab, f"{out}.n_groups.tsv")
            wrote_groups = True
            log(f"[admixed-ld] compose: {len(groups_info)} GWAS sample sizes among the traits (within {n_tol:.0%}); "
                f"one LD-score file each (N {', '.join(g['label'] for g in groups_info)}), listed in {out}.n_groups.tsv")
    if not wrote_groups and os.path.lexists(f"{out}.n_groups.tsv"):
        # an earlier run's table to the same prefix: --h2 would fit its traits with its files
        os.remove(f"{out}.n_groups.tsv")
        log(f"[admixed-ld] compose: removed {out}.n_groups.tsv of an earlier run; this composition has one LD-score "
            f"file for every trait ({out}.gw.ldscore.gz)")
    # the plan in the params json: the file each trait gets (as n_groups.tsv lists it) and each sample size's file
    main_file = os.path.basename(f"{out}.gw.ldscore.gz")
    if wrote_groups:
        plan["sample_sizes"] = [dict(N=float(g["N"]), nu=float(g["nu"]), traits=len(g["traits"]),
                                     file=os.path.basename(g["file"])) for g in groups_info]
        plan["files"] = [dict(trait=t, N=float(nn), ldscores=os.path.basename(g["file"])) for g in groups_info
                         for t, nn in g["traits"]]
    else:
        plan["sample_sizes"] = [dict(s, file=main_file) for s in plan["sample_sizes"]]
        plan["files"] = [dict(trait=t[0], N=float(n_), ldscores=main_file) for t, n_ in zip(tables, n_med)]

    # ---------------------------------------------------------------------------------------- write
    snpdf = pd.DataFrame({"CHR": chrom_out, "SNP": snp_ids, "BP": rv["BP"].to_numpy()[r_idx]})
    write_gw_ldscore_table(out, snpdf, pd.DataFrame(to_cols(L), columns=out_classes), snp_var=V)
    masses_P = np.bincount(clsP, minlength=K).astype(float)        # the partners' counts (the rows' by default)
    write_gw_annotation_masses(out, to_cols(masses_P))
    params = dict(format="summit-admixed-ld-compose", version=1, sources=sources, baseline=sources[-1], n_snps=int(M),
                  n_window_pairs=int(n_pairs), mean_ancestry_vector=m_vec.tolist(), kappa0_matrix=k0.tolist(),
                  mean_ancestry_from_se=np.asarray(m_se).tolist(), kappa0_from_se=np.asarray(k_se).tolist(),
                  gwas_n=N, covariate_rank=p, nu=nu, kernel=kern.describe(), theta=th, theta_source=th_source,
                  relatedness=rel_sum, relatedness_profile=None if rel_prof is None else rel_prof.describe(),
                  relatedness_scale=rel_scale_info,
                  panel_background={sources[n_]: pb.describe() for n_, pb in panel_bg.items()},
                  r2=r2, grid_cm=grid_cm, cell_matrix=cells_used,
                  derived_covariance_cross_noise=[dict(source=sources[x[0]], panel=sources[x[1]]) for x in cross],
                  derived_panel_scale=fsc_info,
                  reference=dict(sources=ref.manifest["sources"], window_kb=ref.manifest["window_kb"],
                                 genetic_map=ref.manifest["genetic_map"]),
                  per_trait=None if per_trait is None else per_trait.to_dict(orient="records"),
                  ld_mass=float((V[:, None] * L).sum()), sample_factors=sf_info, n_groups=groups_info, plan=plan,
                  annotation=dict(columns=list(out_classes), overlapping=col_pat is not None,
                                  combinations=None if col_pat is None else int(col_pat.shape[0])),
                  match=dict(match_info, drop_ambiguous=bool(drop_ambiguous)),
                  ld_mass_check=mass_check, flags=flags)
    if xs_i is not None:                                          # each SNP with itself: sqrt(V_1 V_2), V_2 = 1
        keep = v2_out > 0
        Lx[np.arange(M), cls] += np.sqrt(Vpop)
        xout = f"{out}.cross_{cross_source}"
        write_gw_ldscore_table(xout, snpdf[keep].reset_index(drop=True),
                               pd.DataFrame(to_cols(Lx[keep]), columns=out_classes), snp_var=np.sqrt(Vpop[keep]))
        write_gw_annotation_masses(xout, to_cols(masses_P) if dense
                                   else to_cols(np.bincount(cls[keep], minlength=K).astype(float)))
        params["cross"] = dict(source=cross_source, panel_n=int(ref.n[cross_source]),
                               panel_is_a_source=xs_n is not None, file=f"{xout}.gw.ldscore.gz",
                               n_snps=int(keep.sum()), ld_mass=float((np.sqrt(Vpop[keep])[:, None] * Lx[keep]).sum()),
                               definition="sum_k rho_1(j,k) rho_2(j,k) sqrt(V_1k V_2k) over pairs within the window, "
                                          "rho_2 and V_2 = 1 from the panel of the second population")
        log(f"[admixed-ld] compose: wrote {xout}.gw.ldscore.gz, cross-population LD scores with {cross_source}'s "
            f"panel ({int(keep.sum())} SNPs polymorphic in it)")
    if dense:
        params["partners"] = dict(mode="reference", n_partners=MP, counts=masses_P.tolist(),
                                  definition="the reference variants listed in the annotation; the GWAS variants are "
                                             "the rows, and .gw.M holds the partners' counts per class")
    if T is not None:
        tflags = []
        if R1 is not None:
            tflags.append(dict(flag="track_scores_without_relatives",
                               detail="the relatives' term enters the own-track moments; the track scores leave it out"))
        tb = T.finish(out, snpdf, classes, np.arange(M) if prow is None else row_of, cls, write_gw_ldscore_table,
                      write_gw_annotation_masses, tflags)
        tb.update(partners=partners, decay_per_cm=dict(zip(sources, np.asarray(kern.lam, float).tolist())))
        params["tracks"] = tb
        log(f"[admixed-ld] compose: wrote the track LD scores ({', '.join(tb['files'])}) on {tb['n_rows']} SNPs "
            f"polymorphic in every track; per-allele weights {tb['weights']}; partners: {partners} ({tb['n_partners']})")
        if tb["n_rows_dropped"]:
            log(f"[admixed-ld] compose: {tb['n_rows_dropped']} SNPs monomorphic in a track (or in "
                f"{cross_source}) have no track scores")
    if k1 == 1:                                                    # the two-source keys
        params.update(mean_ancestry=float(m_vec[0]), kappa0=float(k0[0, 0]))
    with open(f"{out}.admixed_ld.json", "w") as fh:
        json.dump(params, fh, indent=1)
    log(f"[admixed-ld] compose: wrote {out}.gw.ldscore.gz and {out}.gw.M ({M} SNPs; m "
        + ", ".join(f"{s} {x:.4f}" for s, x in zip(sources, m_vec)) + "; kappa_res(0) "
        + np.array2string(k0, precision=4).replace("\n", "") + "; decay "
        + ", ".join(f"{x:.4f}" for x in kern.lam) + " per cM) in " + f"{time.perf_counter() - t0:.0f} s")
    return params
