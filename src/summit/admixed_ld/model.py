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
def regress_standard_errors(n, se, beta, vA, vE, delta, *, delta_var=0.0, max_chisq=80.0):
    """Mean ancestry m and kappa_res(0) from one GWAS (arrays aligned to the reference SNPs). Returns a dict with m,
    kappa0, the number of SNPs used and the R2 of the fit; SNPs with large chi-square (strong effects), non-finite
    values or zero variance in both sources are left out, and residual outliers (beyond 8 robust SDs) are dropped
    once before the final fit."""
    n, se, beta = (np.asarray(x, np.float64) for x in (n, se, beta))
    vA, vE = np.asarray(vA, np.float64), np.asarray(vE, np.float64)
    d2 = np.asarray(delta, np.float64) ** 2 - delta_var            # unbiased for the squared frequency difference
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


def parameters_from_sumstats(tables, vA, vE, delta, log=print, delta_var=0.0):
    """`tables`: list of (name, DataFrame with N, SE, BETA aligned to the reference SNPs; rows may be NaN). Returns
    (m, kappa0, per-trait DataFrame); m and kappa0 are medians over traits."""
    rows = []
    for name, t in tables:
        r = regress_standard_errors(t["N"], t["SE"], t["BETA"], vA, vE, delta, delta_var=delta_var)
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


# ===================================================================================================== K sources
# With sources s = 0..K-1 and the last as baseline B, the contrasts D_ja = p^a_j - p^B_j (a < B) carry the admixture
# terms: the covariance of local-ancestry dosages enters as D_j' Kt(d) D_k with Kt the (K-1) x (K-1) block of the K x K
# matrix (whose rows sum to zero). Two sources give the scalar model above (D = delta, Kt = kappa).
def contrast_pairs(k1):
    """Unique contrast pairs (a, b), a <= b."""
    return [(a, b) for a in range(k1) for b in range(a, k1)]


def contrast_noise(V, n, mix=None):
    """Sampling covariance of the contrasts D_a = p^a - p^B (B the last source) at each SNP, and their covariance with
    the baseline's frequency error.

    Each source's frequency error is e_s = sum_t G_st eta_t, with independent base errors eta_t of variance
    V_jt / (4 n_t) (V: M x T dosage variances, n: T effective sizes; the first K columns are the sources in order, any
    further ones are panels that enter only through a derived source). Panels have G = identity; a source derived from a
    cohort carries minus the cohort's weights on the known sources' panel errors (its frequencies subtract their
    contributions). mix: K x T matrix G (default identity, which gives 1[a=b] var_a + var_B).
    Returns S (M x (K-1) x (K-1)) and R (M x (K-1)) with R_a = Cov(e_a - e_B, e_B)."""
    V = np.asarray(V, np.float64)
    T = V.shape[1]
    n = np.asarray(n, np.float64)
    G = np.eye(T) if mix is None else np.asarray(mix, np.float64)
    K = G.shape[0]
    L = np.zeros((K, K - 1))
    L[np.arange(K - 1), np.arange(K - 1)] = 1.0
    L[K - 1, :] = -1.0
    H = L.T @ G                                                    # contrast errors = H eta
    var_eta = V / (4.0 * n[None, :])
    S = np.einsum("at,jt,bt->jab", H, var_eta, H)
    R = np.einsum("at,jt,t->ja", H, var_eta, G[K - 1])
    return S, R


def unbiased_products_cov(D, S):
    """O[j, a, b] = D_ja D_jb - S_jab, with S the sampling covariance of the contrasts (contrast_noise)."""
    return D[:, :, None] * D[:, None, :] - S


def unbiased_products(D, var_c, var_base):
    """O[j, a, b] = D_ja D_jb without the panels' sampling covariance, Cov(e_a - e_B, e_b - e_B) = 1[a=b] var_a + var_B
    with var = Var(dosage) / 4n. D: M x (K-1); var_c: M x (K-1); var_base: M."""
    O = D[:, :, None] * D[:, None, :] - np.asarray(var_base, np.float64)[:, None, None]
    idx = np.arange(D.shape[1])
    O[:, idx, idx] -= var_c
    return O


def regress_standard_errors_k(n, se, beta, V, O, *, max_chisq=80.0):
    """Mean ancestry vector m (K) and Kt_res(0) ((K-1) x (K-1)) from one GWAS: 1 / (N SE^2 + beta^2) on the source
    dosage variances V (M x K) and the unique products O_ab (off-diagonal ones counted twice), without an intercept,
    with the same exclusions as regress_standard_errors. Two sources reproduce it exactly."""
    n, se, beta = (np.asarray(x, np.float64) for x in (n, se, beta))
    V, O = np.asarray(V, np.float64), np.asarray(O, np.float64)
    K, k1 = V.shape[1], O.shape[1]
    pairs = contrast_pairs(k1)
    X = np.column_stack([V] + [O[:, a, b] * (1.0 if a == b else 2.0) for a, b in pairs])
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = (beta / se) ** 2
        y = 1.0 / (n * se * se + beta * beta)
    use = np.isfinite(y) & np.isfinite(z2) & (z2 < max_chisq) & (V.sum(axis=1) > 0) & np.isfinite(X).all(axis=1)
    for it in range(2):
        b, *_ = np.linalg.lstsq(X[use], y[use], rcond=None)
        if it == 0:
            r = y - X @ b
            mad = np.median(np.abs(r[use] - np.median(r[use]))) * 1.4826
            use &= np.abs(r) < 8 * max(mad, np.finfo(float).tiny)
    fit = X[use] @ b
    r2 = 1.0 - np.sum((y[use] - fit) ** 2) / np.sum((y[use] - y[use].mean()) ** 2)
    s = b[:K].sum()
    kt = np.zeros((k1, k1))
    for (a, c), v in zip(pairs, b[K:]):
        kt[a, c] = kt[c, a] = v / s
    return dict(m=(b[:K] / s).tolist(), kappa0=kt.tolist(), n_snps=int(use.sum()), fit_r2=float(r2))


def parameters_from_sumstats_k(tables, V, O, names, log=print):
    """Medians over traits of m (renormalized to sum to one) and of each entry of Kt_res(0)."""
    rows = []
    for name, t in tables:
        r = regress_standard_errors_k(t["N"], t["SE"], t["BETA"], V, O)
        r["trait"] = name
        rows.append(r)
        log(f"[admixed-ld] standard errors of {name}: m = " + ", ".join(f"{s} {x:.4f}" for s, x in zip(names, r["m"]))
            + "; kappa_res(0) = " + np.array2string(np.asarray(r["kappa0"]), precision=4).replace("\n", "")
            + f" ({r['n_snps']} SNPs, R2 {r['fit_r2']:.4f})")
    m = np.median(np.array([r["m"] for r in rows]), axis=0)
    m = m / m.sum()
    kt = np.median(np.array([r["kappa0"] for r in rows]), axis=0)
    if np.any(m <= 0) or np.any(m >= 1) or np.any(np.diag(kt) < 0):
        raise ValueError(f"The standard errors give mean ancestry {np.round(m, 4)} and kappa_res(0) diagonal "
                         f"{np.round(np.diag(kt), 4)}, outside the model. Check that the reference sources match the "
                         "cohort, or pass --mean-ancestry and --kappa0.")
    return m, kt, pd.DataFrame(rows)


def full_from_contrasts(kt):
    """The K x K local-ancestry covariance (rows summing to zero) from its (K-1) x (K-1) contrast block (any leading
    shape)."""
    kt = np.asarray(kt, np.float64)
    k1 = kt.shape[-1]
    out = np.zeros(kt.shape[:-2] + (k1 + 1, k1 + 1))
    out[..., :k1, :k1] = kt
    out[..., :k1, k1] = -kt.sum(axis=-1)
    out[..., k1, :k1] = -kt.sum(axis=-2)
    out[..., k1, k1] = kt.sum(axis=(-1, -2))
    return out


class _Entry:
    """One entry of the kernel matrix as a function of distance (cM): the smoothed curve beyond d0, joined below d0 to
    kappa_res(0), and the between-chromosome value beyond the curve; or far + (k0 - far) exp(-lam d) from a date."""

    def __init__(self, k0, far, lam=None, d=None, k=None, d0=2.0, diagonal=True):
        self.k0, self.far, self.lam, self.d, self.k, self.d0 = float(k0), float(far), lam, d, k, d0
        self.mode, self.lam_short, self.note = "decay", 0.0, ""
        if d is not None:
            kd0 = float(np.interp(d0, d, k))
            if diagonal and not kd0 > 0.0:
                raise ValueError(f"The curve at {d0} cM is {kd0:.4g}; a positive value is needed to join it to "
                                 "kappa_res(0). Check that the curve comes from this cohort and reference.")
            self.kd0 = kd0
            if self.k0 > 0 and 0 < kd0 < self.k0:
                self.mode, self.lam_short = "exp", float(-np.log(kd0 / self.k0) / d0)
            elif self.k0 > 0 and kd0 >= self.k0:
                self.mode = "flat"
                self.note = (f"curve at {d0} cM ({kd0:.4g}) is not below kappa_res(0) ({self.k0:.4g}); "
                             "kappa is held at the curve value below it")
            else:
                self.mode = "linear"

    def __call__(self, dist):
        if self.d is None:
            return self.far + (self.k0 - self.far) * np.exp(-self.lam * dist)
        out = np.interp(dist, self.d, self.k)
        if self.mode == "exp":
            short = self.k0 * np.exp(-self.lam_short * dist)
        elif self.mode == "flat":
            short = np.full_like(dist, self.kd0)
        else:
            short = self.k0 + (self.kd0 - self.k0) * dist / self.d0
        out = np.where(dist < self.d0, short, out)
        return np.where(dist > self.d[-1], self.far, out)


def check_kernel_join(kern, allow=False, log=print):
    """Fail closed when the cohort's curve at d0 is not below kappa_res(0) from the standard errors. The kernel decays
    with distance, so a curve above kappa_res(0) means the two inputs disagree (for example kappa_res(0) from SEs of
    another covariate model or SNP set, a curve from another sample, or background LD in the curve at d0). Returns the
    recorded flags (empty when the join is regular); raises unless `allow`, in which case the kernel is held at the
    curve's value below d0 and the flag is recorded."""
    if not kern.note:
        return []
    if not allow:
        raise ValueError(f"{kern.note}. The curve and the standard errors disagree. Check that both come from this GWAS "
                         "sample and covariate model, give --kappa0, or pass --allow-flat-kernel-join to proceed with "
                         "the flag recorded.")
    log(f"[admixed-ld] WARNING (recorded flag kernel_flat_join): {kern.note}")
    return [dict(flag="kernel_flat_join", detail=kern.note)]


class KernelMatrix:
    """Kt_res(d) of the K-1 contrasts ((K-1) x (K-1), symmetric) and the per-source decay rates lam (K) of the
    recombination factor exp(-lam_s d) on within-source LD. For two sources this is the scalar Kernel."""

    def __init__(self, kappa0, far, lam, source, entries, note="", d=None):
        self.kappa0, self.far = np.atleast_2d(kappa0).astype(float), np.atleast_2d(far).astype(float)
        self.lam = np.atleast_1d(np.asarray(lam, float))
        self.source, self.entries, self.note, self.d = source, entries, note, d
        self.k1 = self.kappa0.shape[0]

    @classmethod
    def from_curve(cls, d, k, inter, kappa0, lam, *, half_width=0.75, rel_width=0.1, d0=2.0):
        """d (n_d), k (n_d x k1 x k1) curve, inter (k1 x k1) between-chromosome values, kappa0 from the SEs, lam (K)."""
        d = np.asarray(d, np.float64)
        k = np.asarray(k, np.float64).reshape(d.size, *np.atleast_2d(kappa0).shape)
        kappa0, inter = np.atleast_2d(kappa0), np.atleast_2d(inter)
        k1 = kappa0.shape[0]
        entries, notes = {}, []
        for a, b in contrast_pairs(k1):
            ks = smooth_curve(d, k[:, a, b], half_width=half_width, rel_width=rel_width)
            e = _Entry(kappa0[a, b], inter[a, b], d=d, k=ks, d0=d0, diagonal=(a == b))
            entries[(a, b)] = e
            if e.note:
                notes.append(e.note)
        return cls(kappa0, inter, lam, "genotype curve", entries, note="; ".join(notes), d=d)

    @classmethod
    def from_decay(cls, kappa0, lam, genome_cm, n_sources, source):
        if not lam > 0:
            raise ValueError("The decay rate must be positive.")
        kappa0 = np.atleast_2d(kappa0)
        far = -2.0 * kappa0 / (lam * genome_cm)
        entries = {(a, b): _Entry(kappa0[a, b], far[a, b], lam=float(lam)) for a, b in contrast_pairs(kappa0.shape[0])}
        return cls(kappa0, far, np.full(n_sources, float(lam)), source, entries)

    def __call__(self, dist):
        """Kt(d) with shape dist.shape + (k1, k1)."""
        dist = np.asarray(dist, np.float64)
        out = np.empty(dist.shape + (self.k1, self.k1))
        for (a, b), e in self.entries.items():
            v = e(dist)
            out[..., a, b] = v
            out[..., b, a] = v
        return out

    def entry(self, a, b, dist):
        return self.entries[(min(a, b), max(a, b))](np.asarray(dist, np.float64))

    def describe(self):
        return dict(source=self.source, kappa0=self.kappa0.tolist(), far=self.far.tolist(),
                    decay_per_cm=self.lam.tolist(), curve_points=None if self.d is None else int(self.d.size),
                    note=self.note)


def moments_from_vectors(U) -> dict:
    """Central moments of ancestry vectors U (people x k1): mean, cov, third and fourth moment tensors; for k1 = 1 also
    the scalar keys of moments_from_values."""
    U = np.atleast_2d(np.asarray(U, np.float64))
    if U.shape[0] == 1 and U.shape[1] > 1:
        U = U.T
    W = U - U.mean(axis=0)
    n = W.shape[0]
    out = dict(mean_vector=U.mean(axis=0).tolist(), cov=(W.T @ W / n).tolist(),
               m3_tensor=np.einsum("ia,ib,ic->abc", W, W, W, optimize=True).__truediv__(n).tolist(),
               m4_tensor=np.einsum("ia,ib,ic,id->abcd", W, W, W, W, optimize=True).__truediv__(n).tolist())
    if U.shape[1] == 1:
        out.update(moments_from_values(U[:, 0]))
    return out


def moments_tensors(th, k1):
    """(mean (k1), cov (k1 x k1), m3 (k1^3), m4 (k1^4)) from a moments dict (vector keys, or scalar keys when k1 = 1)."""
    if "cov" in th:
        return (np.asarray(th["mean_vector"]), np.asarray(th["cov"]), np.asarray(th["m3_tensor"]),
                np.asarray(th["m4_tensor"]))
    if k1 != 1:
        raise ValueError("Vector moments of global ancestry are needed for more than two sources.")
    return (np.array([th["mean"]]), np.array([[th["var"]]]), np.array([[[th["m3"]]]]), np.array([[[[th["m4"]]]]]))


def ancestry_sd_from_kernel(m_first, kappa0_first):
    """SD of the first source's global ancestry implied by the cohort quantities of the standard errors, for Tier S
    without a published value (decision of 7 Nov 2026). With independent haplotypes (random mating), the local-ancestry
    dosage of a source at one SNP has variance 2 m (1 - m) = 4 Var(theta) + H, and the GWAS PCs leave kappa_res(0) ~ H,
    so Var(theta) = (2 m (1 - m) - kappa_res(0)) / 4. Assortative mating raises Var(theta) above this, so the value is
    a lower bound in practice (UK Biobank t1 / t2: 0.261 / 0.242 against 0.293 / 0.301 from the genotypes)."""
    v = (2.0 * m_first * (1.0 - m_first) - kappa0_first) / 4.0
    if not v > 0:
        raise ValueError(f"kappa_res(0) {kappa0_first:.4g} is not below 2 m (1 - m) = {2 * m_first * (1 - m_first):.4g}, "
                         "so the standard errors imply no variance of global ancestry; give --ancestry-sd.")
    return float(np.sqrt(v))


def dirichlet_moments(m, sd_first, draws=400_000, seed=0) -> dict:
    """Moments of the first K-1 ancestries under Dirichlet(c m), with c set so that the first ancestry has SD sd_first
    (two sources: the Beta moments, in closed form)."""
    m = np.asarray(m, np.float64)
    if m.size == 2:
        return beta_moments(float(m[0]), sd_first)
    c = m[0] * (1 - m[0]) / sd_first ** 2 - 1.0
    if not c > 0:
        raise ValueError("The SD of global ancestry is too large for its mean.")
    U = np.random.default_rng(seed).dirichlet(c * m, size=draws)[:, :-1]
    return moments_from_vectors(U)


def same_person_term(beta, O, s2, cres, cls, n_classes, mom, nu):
    """(1 / nu) sum_{k in c} Cov(h_j, h_k) / (cres_j s2_k) for K sources, with h(theta) = 2 p (1 - p) quadratic in the
    ancestry contrasts w: h_j = h_j(mean) + beta_j' w - 2 (D_j' w)^2. beta: M x k1; O: M x k1 x k1 (unbiased products);
    mom = (mean, cov, m3, m4). Two sources reproduce the scalar formula."""
    _, S, M3, M4 = mom
    k1 = beta.shape[1]
    B = np.zeros((n_classes, k1))
    G = np.zeros((n_classes, k1, k1))
    for a in range(k1):
        B[:, a] = np.bincount(cls, weights=beta[:, a] / s2, minlength=n_classes)
        for b in range(k1):
            G[:, a, b] = np.bincount(cls, weights=O[:, a, b] / s2, minlength=n_classes)
    C4 = M4 - np.einsum("ab,cd->abcd", S, S)
    t1 = np.einsum("ja,ab,cb->jc", beta, S, B)
    t2 = -2.0 * np.einsum("ja,abe,cbe->jc", beta, M3, G)
    t3 = -2.0 * np.einsum("jab,abe,ce->jc", O, M3, B)
    t4 = 4.0 * np.einsum("jab,abef,cef->jc", O, C4, G)
    return (t1 + t2 + t3 + t4) / (nu * cres)[:, None]
