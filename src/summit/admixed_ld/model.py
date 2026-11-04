"""Parameters of the composition: mean ancestry and kappa_res(0) from GWAS standard errors, the local-ancestry kernel,
and the moments of global ancestry.

Standard errors. For a least-squares GWAS with covariates, N_j SE_j^2 + beta_j^2 = sigma^2 / Var_res(g_j) up to terms of
order 1/N, and under the composition model Var_res(g_j) = m v^A_j + (1 - m) v^E_j + delta_j^2 kappa_res(0). Regressing
1 / (N_j SE_j^2 + beta_j^2) on (v^A_j, v^E_j, delta_j^2) without an intercept gives b, and m = b_1 / (b_1 + b_2),
kappa_res(0) = b_3 / (b_1 + b_2). The sampling scale sigma^2 cancels.

Kernel. kappa_res(d), with d in cM, is either the cohort's measured curve (smoothed) beyond `d0`, joined below `d0` by
an exponential from kappa_res(0), and the curve's between-chromosome value beyond the curve; or, from a decay rate
lambda (a published admixture date T gives lambda = T / 100 per cM), far + (kappa_res(0) - far) exp(-lambda d) with
far = -2 kappa_res(0) / (lambda G), the constant that adjustment for each person's realized ancestry leaves (G is the
genetic length of the genome in cM). lambda also sets the recombination factor exp(-lambda d) of within-source LD.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------------- SE regression
def regress_standard_errors(n, se, beta, vA, vE, delta, *, max_chisq=80.0):
    """Mean ancestry m and kappa_res(0) from one GWAS (arrays aligned to the reference SNPs). Returns a dict with m,
    kappa0, the number of SNPs used and the R2 of the fit; SNPs with large chi-square (strong effects), non-finite
    values or zero variance in both sources are left out, and residual outliers (beyond 8 robust SDs) are dropped
    once before the final fit."""
    n, se, beta = (np.asarray(x, np.float64) for x in (n, se, beta))
    vA, vE, d2 = np.asarray(vA, np.float64), np.asarray(vE, np.float64), np.asarray(delta, np.float64) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = (beta / se) ** 2
        y = 1.0 / (n * se * se + beta * beta)
    use = np.isfinite(y) & np.isfinite(z2) & (z2 < max_chisq) & (vA + vE > 0) & np.isfinite(d2)
    X = np.column_stack([vA, vE, d2])
    for it in range(2):
        b, *_ = np.linalg.lstsq(X[use], y[use], rcond=None)
        if it == 0:
            r = y - X @ b
            mad = np.median(np.abs(r[use] - np.median(r[use]))) * 1.4826
            use &= np.abs(r) < 8 * max(mad, np.finfo(float).tiny)
    fit = X[use] @ b
    r2 = 1.0 - np.sum((y[use] - fit) ** 2) / np.sum((y[use] - y[use].mean()) ** 2)
    s = b[0] + b[1]
    return dict(m=float(b[0] / s), kappa0=float(b[2] / s), n_snps=int(use.sum()), fit_r2=float(r2))


def parameters_from_sumstats(tables, vA, vE, delta, log=print):
    """`tables`: list of (name, DataFrame with N, SE, BETA aligned to the reference SNPs; rows may be NaN). Returns
    (m, kappa0, per-trait DataFrame); m and kappa0 are medians over traits."""
    rows = []
    for name, t in tables:
        r = regress_standard_errors(t["N"], t["SE"], t["BETA"], vA, vE, delta)
        r["trait"] = name
        rows.append(r)
        log(f"[admixed-ld] standard errors of {name}: m = {r['m']:.4f}, kappa_res(0) = {r['kappa0']:.4f} "
            f"({r['n_snps']} SNPs, R2 {r['fit_r2']:.4f})")
    df = pd.DataFrame(rows)[["trait", "m", "kappa0", "n_snps", "fit_r2"]]
    m, k0 = float(df.m.median()), float(df.kappa0.median())
    if not (0.0 < m < 1.0) or k0 < 0.0:
        raise ValueError(f"The standard errors give mean ancestry {m:.4f} and kappa_res(0) {k0:.4f}, outside the model. "
                         "Check that the reference sources match the cohort, or pass --mean-ancestry and --kappa0.")
    return m, k0, df


# ----------------------------------------------------------------------------------------------------- kernel
def fit_exponential(d, k, min_d=2.0, max_d=20.0):
    """Least-squares fit of A + H exp(-lam d) to a curve for min_d <= d <= max_d (the range where the decay since
    admixture dominates; farther out, adjustment for realized ancestry and sparse pairs bend the curve). Returns
    dict(A, H, lam)."""
    from scipy.optimize import curve_fit
    d, k = np.asarray(d, np.float64), np.asarray(k, np.float64)
    use = (d >= min_d) & (d <= max_d) & np.isfinite(k)
    if use.sum() < 5:
        raise ValueError("Too few curve points beyond the fitting start to fit the decay.")
    f = lambda x, A, H, lam: A + H * np.exp(-lam * x)
    p0 = (float(k[use][-1]), float(max(k[use][0], 1e-3)), 0.05)
    p, _ = curve_fit(f, d[use], k[use], p0=p0, bounds=([-1.0, 0.0, 1e-5], [1.0, 2.0, 5.0]), maxfev=20000)
    return dict(A=float(p[0]), H=float(p[1]), lam=float(p[2]))


def smooth_curve(d, k, *, half_width=0.75, rel_width=0.1):
    """Moving average of k over points with |d' - d| <= max(half_width, rel_width * d) (d sorted)."""
    d, k = np.asarray(d, np.float64), np.asarray(k, np.float64)
    if half_width <= 0 and rel_width <= 0:
        return k.copy()
    w = np.maximum(half_width, rel_width * d)
    lo = np.searchsorted(d, d - w, side="left")
    hi = np.searchsorted(d, d + w, side="right")
    cs = np.concatenate([[0.0], np.cumsum(k)])
    return (cs[hi] - cs[lo]) / (hi - lo)


@dataclass
class Kernel:
    """kappa_res(d) for d in cM. `lam` (per cM) also gives the recombination factor of within-source LD."""
    kappa0: float
    lam: float
    far: float
    source: str
    d: np.ndarray | None = None
    k: np.ndarray | None = None
    d0: float = 2.0
    lam_short: float = field(init=False, default=0.0)
    note: str = field(init=False, default="")

    def __post_init__(self):
        if self.d is not None:
            if np.any(np.diff(self.d) <= 0):
                raise ValueError("Curve distances must be strictly increasing.")
            k_d0 = float(np.interp(self.d0, self.d, self.k))
            if not k_d0 > 0.0:
                raise ValueError(f"The curve at {self.d0} cM is {k_d0:.4g}; a positive value is needed to join it to "
                                 "kappa_res(0). Check that the curve comes from this cohort and reference.")
            if k_d0 < self.kappa0:
                self.lam_short = float(-np.log(k_d0 / self.kappa0) / self.d0)
            else:                       # the curve at d0 exceeds kappa_res(0) from the SEs: join it flat
                self.lam_short = 0.0
                self.note = (f"curve at {self.d0} cM ({k_d0:.4g}) is not below kappa_res(0) ({self.kappa0:.4g}); "
                             "kappa is held at the curve value below it")

    @classmethod
    def from_curve(cls, d, k, inter, kappa0, *, half_width=0.75, rel_width=0.1, d0=2.0, fit_min=2.0, fit_max=20.0):
        """The curve smoothed by a moving average over distances within max(half_width, rel_width * d) of each point
        (pairs thin out with distance, so the window widens), its decay fitted from fit_min, and the
        between-chromosome value `inter`."""
        d = np.asarray(d, np.float64)
        k = np.asarray(k, np.float64)
        ks = smooth_curve(d, k, half_width=half_width, rel_width=rel_width)
        lam = fit_exponential(d, k, fit_min, fit_max)["lam"]
        return cls(kappa0=float(kappa0), lam=lam, far=float(inter), source="genotype curve", d=d, k=ks, d0=d0)

    @classmethod
    def from_decay(cls, kappa0, lam, genome_cm, source):
        if not lam > 0:
            raise ValueError("The decay rate must be positive.")
        far = -2.0 * kappa0 / (lam * genome_cm)
        return cls(kappa0=float(kappa0), lam=float(lam), far=float(far), source=source)

    def __call__(self, dist):
        dist = np.asarray(dist, np.float64)
        if self.d is None:
            return self.far + (self.kappa0 - self.far) * np.exp(-self.lam * dist)
        out = np.interp(dist, self.d, self.k)
        short = (self.kappa0 * np.exp(-self.lam_short * dist) if self.lam_short > 0
                 else np.full_like(dist, float(np.interp(self.d0, self.d, self.k))))
        out = np.where(dist < self.d0, short, out)
        return np.where(dist > self.d[-1], self.far, out)

    def describe(self):
        return dict(source=self.source, kappa0=self.kappa0, decay_per_cm=self.lam, far=self.far,
                    curve_points=None if self.d is None else int(self.d.size), d0_cm=self.d0, note=self.note)


# ----------------------------------------------------------------------------------------------------- global ancestry
def moments_from_values(theta) -> dict:
    t = np.asarray(theta, np.float64)
    c = t - t.mean()
    return dict(mean=float(t.mean()), var=float(np.mean(c ** 2)), m3=float(np.mean(c ** 3)), m4=float(np.mean(c ** 4)))


def beta_moments(mean, sd) -> dict:
    """Central moments of a Beta distribution with the given mean and standard deviation."""
    v = float(sd) ** 2
    if not (0 < mean < 1) or not (0 < v < mean * (1 - mean)):
        raise ValueError("Global ancestry needs 0 < mean < 1 and 0 < SD^2 < mean (1 - mean).")
    k = mean * (1 - mean) / v - 1
    a, b = mean * k, (1 - mean) * k
    m3 = 2 * (b - a) * np.sqrt(a + b + 1) / ((a + b + 2) * np.sqrt(a * b)) * v ** 1.5
    ex = 6 * ((a - b) ** 2 * (a + b + 1) - a * b * (a + b + 2)) / (a * b * (a + b + 2) * (a + b + 3))
    return dict(mean=float(mean), var=v, m3=float(m3), m4=float((ex + 3) * v * v))
