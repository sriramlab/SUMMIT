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
                                      (1 / nu^2) [B sum_{k in c, k != j} V_k + sum_{k in c on j's chromosome}
                                      (phi(d_jk) - B) V_k] (RelatednessProfile, BackgroundProfile, or the cohort's
                                      all-pairs background at every distance, SmoothBackground), or the uniform
                                      (sum over their ordered pairs of g^2) / nu^2 * sum_{k in c} V_k, with g the
                                      relatedness of the covariate-adjusted, standardized genotypes.
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

from ._io import match_variants
from .model import (KernelMatrix, contrast_pairs, dirichlet_moments, moments_tensors, parameters_from_sumstats_k,
                    contrast_noise, same_person_term, unbiased_products_cov)
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


def unbiased_squared_covariance(c, m22, vv, n):
    """Unbiased estimate of the squared covariance of two variants from a panel of n people (the order-4 U-statistic):
    c the sample covariance (n - 1 denominator), m22 the mean of x_j^2 x_k^2 over people (centred dosages), vv the
    product of the two sample variances (n - 1). Exact for any genotype distribution, so small panels are not biased
    (the first-order correction c^2 - (m22 - c^2)/n leaves a bias of about 4/n of the noise)."""
    n = float(n)
    return ((n - 1.0) ** 2 * (n - 2.0) * c * c - n * n * m22 + (n - 1.0) * vv) / (n * (n - 2.0) * (n - 3.0))


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
        Sr = z[key].reshape(k1, nC, k1, nC)                        # [a, cell, b, cell']
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
        (1 / nu^2) [B sum_{k in c, k != j} V_k + sum_{k in c on j's chromosome, k != j} (phi(d_jk) - B) V_k].
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
        return dict(mode="background", name=self.name, between=self.between, n=self.n, fit_min_cm=self.fit_min_cm,
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


def relatedness_term(L, rel, chrom, cm, cls, V, K, nu, grid_cm):
    """Add the relatives' term of `rel` (RelatednessProfile) to the LD scores L (M x K) in place: B between all pairs
    (each SNP without itself) and phi(d) - B along each chromosome, by FFT convolution on a grid of grid_cm."""
    Vtot = np.bincount(cls, weights=V, minlength=K)
    L += (rel.between / nu ** 2) * Vtot[None, :]
    L[np.arange(cls.size), cls] -= (rel.between / nu ** 2) * V
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
        L[sel] += acc / nu ** 2


def _parse_values(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in str(x).split(",") if v.strip()]


def compose_ldscores(ref_path, sumstats, annot, out, *, sources=None, snps=None, match="position", curve=None,
                     generations=None, decay_per_cm=None, mean_ancestry=None, kappa0=None, theta_sd=None,
                     relatedness=None, relatedness_smooth=0.1, background_min_cm=3.0, background_window="panels",
                     nu=None, r2=0.99, grid_cm=0.02, cell_matrix=False,
                     cell_switch_cm=10.0,
                     pair_chunk=1 << 23, log=print):
    """Composed LD scores for an admixed GWAS sample with two or more sources (the last is the baseline of the
    contrasts). mean_ancestry: K-1 or K values (comma-separated); kappa0: the upper triangle of Kt_res(0) row by row."""
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
    if sum(x is not None for x in (curve, generations, decay_per_cm)) != 1:
        raise ValueError("Give exactly one source for the decay: --curve, --admixture-generations or --decay-per-cm.")

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
    t_idx, r_idx, _ = match_variants(tv.reset_index(drop=True), rv, how=how)
    tv = tv.reset_index(drop=True)
    cls_mat, classes = read_annotation_classes(annot, tv.SNP.to_numpy()[t_idx])
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
    if m_in is None or k_in is None:
        m_se, k_se, per_trait = parameters_from_sumstats_k(tables, Vm, O, sources, log=log)
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
    elif generations is not None:
        kern = KernelMatrix.from_decay(k0, float(generations) / 100.0, ref.genome_cm, nK,
                                       source=f"admixture {generations} generations ago")
    else:
        kern = KernelMatrix.from_decay(k0, float(decay_per_cm), ref.genome_cm, nK, source=f"decay {decay_per_cm} per cM")
    if kern.note:
        log(f"[admixed-ld] note: {kern.note}")
    if theta_sd is not None:
        th = dirichlet_moments(m_vec, theta_sd)
        th_source = f"Beta/Dirichlet with mean {np.round(m_vec, 3).tolist()} and SD {theta_sd} (first source)"
    elif curve_info is not None:
        th, th_source = curve_info["theta"], "the cohort's curve"
    else:
        th = dirichlet_moments(m_vec, 0.12)
        th_source = "Beta/Dirichlet with SD 0.12 (default; pass --ancestry-sd)"
        log("[admixed-ld] note: global-ancestry moments default to SD 0.12; they enter only the same-person term, "
            "which is small at biobank sample sizes.")
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
                log("[admixed-ld] note: the curve has no relatedness profile (made before 5 Nov 2026); the relatives' "
                    "term is the uniform sum of g^2")
            rel_sum = float(curve_info["relatedness"]["sum_squared_ordered"])
    else:
        rel_sum = float(relatedness)
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
            elif src_m.get("derived"):
                log(f"[admixed-ld] note: {s} is a derived source without genotypes; its covariances' own background "
                    "LD is not removed")
            else:
                missing.append(s)
        if missing:
            raise ValueError("--relatedness curve with the cohort's background at every distance needs each panel's "
                             f"own background, which {ref_path} does not record for {', '.join(missing)}: run "
                             "`summit admixed-ld clean-panels` (it measures it), or pass --background-window zero for "
                             "the rule of 5 Nov 2026.")
        log("[admixed-ld] compose: the panels' own background LD is removed inside the band ("
            + ", ".join(f"{sources[n_]}: {pb.per_pair(2.0):.3g} per SNP pair at 2 cM, n {pb.n}"
                        for n_, pb in panel_bg.items()) + ")")

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
    L = np.zeros((M, K))
    chrom_out = rv["CHR"].to_numpy()[r_idx]
    local_out = rv["LOCAL"].to_numpy()[r_idx]
    cm_out = rv["CM"].to_numpy()[r_idx]
    k2 = k1 * k1
    Oflat = O.reshape(M, k2)                                       # o_j[u], u = (a, c): D_ja D_jc
    Wn = Oflat / s2[:, None]                                       # partner weights of the admixture terms
    W_tot = np.stack([np.bincount(cls, weights=Wn[:, u], minlength=K) for u in range(k2)])   # k2 x K
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
    n_pairs = 0
    for c in ref.chromosomes:
        sel = np.flatnonzero(chrom_out == c)
        if sel.size == 0:
            continue
        z = ref.load(c)
        nloc = z["bp"].size
        tmap = -np.ones(nloc, dtype=np.int64)
        tmap[local_out[sel]] = sel
        indptr, indices, cm_ref = z["indptr"], z["indices"], z["cm"]
        cov = [z[f"cov_{i}"] for i in si]
        # a derived source carries the sampling variance of each covariance (covvar) instead of the panel moments
        cvar = [z[f"covvar_{i}"] if f"covvar_{i}" in z.files else None for i in si]
        m22 = [z[f"m22_{i}"] if cvar[n_] is None else None for n_, i in enumerate(si)]
        # the variances of the people behind each covariance (a derived source with a small panel's LD keeps that
        # panel's variances apart from the dosage variances its frequencies imply)
        vr = [z[f"vld_{i}"] if f"vld_{i}" in z.files else z[f"v_{i}"] for i in si]
        cross_c = [x for x in cross if cvar[x[0]] is not None]          # only covariances stored with covvar
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
                if use.any():
                    j, k, jl, kl = j[use], k[use], jl[use], kl[use]
                    d = np.abs(cm_ref[kl] - cm_ref[jl])
                    Kd = kern(d)                                    # pairs x k1 x k1
                    adm = np.einsum("pa,pab,pb->p", Dm[j], Kd, Dm[k])
                    cr = adm.copy()
                    pis, vnss = [None] * nK, [None] * nK
                    for s in range(nK):
                        cs_ = cov[s][s0:s1][use].astype(np.float64)
                        pi_s = m_vec[s] * np.exp(-kern.lam[s] * d)
                        # sampling noise of the squared panel covariance: c^2 minus its unbiased estimate
                        if cvar[s] is not None:
                            vns = cvar[s][s0:s1][use].astype(np.float64)
                        else:
                            vns = cs_ * cs_ - unbiased_squared_covariance(cs_, m22[s][s0:s1][use],
                                                                          vr[s][jl] * vr[s][kl], ns[s])
                        cr += pi_s * cs_
                        pis[s], vnss[s] = pi_s, vns
                    noise = window_noise(pis, vnss, cross_c, d)
                    for n_, pb in panel_bg.items():               # the panel's own background (SmoothBackground)
                        noise += (pis[n_] * pis[n_] * vr[n_][jl].astype(np.float64) * vr[n_][kl].astype(np.float64)
                                  * pb.per_pair(d))
                    sq = cr * cr - noise - adm * adm
                    L += np.bincount(j * K + cls[k], weights=sq / (cres[j] * s2[k]), minlength=M * K).reshape(M, K)
                    L += np.bincount(k * K + cls[j], weights=sq / (cres[k] * s2[j]), minlength=M * K).reshape(M, K)
                    n_pairs += int(use.sum())
            r0 = r1
        # admixture LD of every pair on the chromosome: (D_j' Kt D_k)^2 = sum_{u,v} o_j[u] F_uv(d) o_k[v] with
        # F_{(a,c),(b,e)} = Kt_ab Kt_ce, by FFT convolution on a grid
        x = cm_out[sel]
        g = np.floor((x - x.min()) / grid_cm).astype(np.int64)
        ng = int(g.max()) + 1
        grid_d = np.arange(ng) * grid_cm
        kg = kern(grid_d)                                           # ng x k1 x k1
        Wg = np.zeros((k2, ng, K))
        for v in range(k2):
            np.add.at(Wg[v], (g, cls[sel]), Wn[sel, v])
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
        log(f"[admixed-ld] compose chr{c}: {sel.size} SNPs ({time.perf_counter() - t0:.0f} s)")
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
    L += same_person_term(beta_, O, s2, cres, cls, K, mom, nu)
    L[np.arange(M), cls] += Vpop
    if rel_sum is not None:
        L += (rel_sum / nu ** 2) * np.bincount(cls, weights=Vpop, minlength=K)[None, :]
    if rel_prof is not None:
        relatedness_term(L, rel_prof, chrom_out, cm_out, cls, Vpop, K, nu, grid_cm)
    V = Vpop

    # ---------------------------------------------------------------------------------------- write
    snpdf = pd.DataFrame({"CHR": chrom_out, "SNP": snp_ids, "BP": rv["BP"].to_numpy()[r_idx]})
    write_gw_ldscore_table(out, snpdf, pd.DataFrame(L, columns=classes), snp_var=V)
    write_gw_annotation_masses(out, np.bincount(cls, minlength=K).astype(float))
    params = dict(format="summit-admixed-ld-compose", version=1, sources=sources, baseline=sources[-1], n_snps=int(M),
                  n_window_pairs=int(n_pairs), mean_ancestry_vector=m_vec.tolist(), kappa0_matrix=k0.tolist(),
                  mean_ancestry_from_se=np.asarray(m_se).tolist(), kappa0_from_se=np.asarray(k_se).tolist(),
                  gwas_n=N, covariate_rank=p, nu=nu, kernel=kern.describe(), theta=th, theta_source=th_source,
                  relatedness=rel_sum, relatedness_profile=None if rel_prof is None else rel_prof.describe(),
                  panel_background={sources[n_]: pb.describe() for n_, pb in panel_bg.items()},
                  r2=r2, grid_cm=grid_cm, cell_matrix=cells_used,
                  derived_covariance_cross_noise=[dict(source=sources[x[0]], panel=sources[x[1]]) for x in cross],
                  reference=dict(sources=ref.manifest["sources"], window_kb=ref.manifest["window_kb"],
                                 genetic_map=ref.manifest["genetic_map"]),
                  per_trait=None if per_trait is None else per_trait.to_dict(orient="records"),
                  ld_mass=float((V[:, None] * L).sum()))
    if k1 == 1:                                                    # the two-source keys
        params.update(mean_ancestry=float(m_vec[0]), kappa0=float(k0[0, 0]))
    with open(f"{out}.admixed_ld.json", "w") as fh:
        json.dump(params, fh, indent=1)
    log(f"[admixed-ld] compose: wrote {out}.gw.ldscore.gz and {out}.gw.M ({M} SNPs; m "
        + ", ".join(f"{s} {x:.4f}" for s, x in zip(sources, m_vec)) + "; kappa_res(0) "
        + np.array2string(k0, precision=4).replace("\n", "") + "; decay "
        + ", ".join(f"{x:.4f}" for x in kern.lam) + " per cM) in " + f"{time.perf_counter() - t0:.0f} s")
    return params
