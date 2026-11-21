"""The cohort's person-factor statistic (`summit admixed-ld sample-factors`): an optional aggregate that gives the
same-person term of the in-sample LD scores for samples where the random-mating model of it fails (a few hundred
people, first-generation groups, in-sample PCs).

In-sample LD scores hold, for every pair of SNPs, the part from each person paired with themselves that SUMMIT's
noise correction leaves: (n / nu^2) Cov_i(y_ij, y_ik), the covariance across the GWAS's people of their squared
covariate-adjusted, standardized genotypes y_ij = x~_ij^2. Its share of the LD mass grows with SNPs per person. If a
person's genotypes at unlinked SNPs are independent given their pedigree,
    E[y_ij | i] s_j^2 = level_i v_Bj + sum_s theta_is (v_sj - v_Bj) + D_j' Lambda_i D_j = a_j' u_i,
with v the sources' dosage variances, D the contrasts with the baseline B, theta_i the person's ancestry, Lambda_i the
covariance matrix of the person's local-ancestry dosages (two sources: lambda_i = 2 theta_i (1 - theta_i) - psi_i / 2,
psi_i the squared difference between the parents' ancestries; an F1 has lambda = 0), and level_i a general scale. In a
few hundred people level_i varies by about 10%, almost all of it the leverage of the GWAS's covariates on the person
(in-sample PCs single out a few people and remove most of their genotype variance). Then
    Cov(y_j, y_k) = A_j' Sigma_u A_k,   A_j = a_j / s_j^2,
and the cohort releases only Sigma_u, (1 + (K-1) + K(K-1)/2)^2 numbers. u_i is estimated per person by least squares
of y_i on A across the SNPs of the odd and, separately, of the even chromosomes; Sigma_u is the covariance across people
between the two halves' estimates, whose estimation noise is independent (symmetrized). Under random mating
level = 1 and Lambda = 2 (diag(theta) - theta theta'), and A' Sigma_u A is the random-mating covariance of
heterozygosity that compose uses by default. The loadings use the reference's dosage variances and the contrasts'
products less their sampling covariance, as compose does.

Diagnostics written with the statistic (aggregates): the realized between-chromosome same-person mass of the sample,
from each person's per-chromosome sums of y, the factor model's prediction of it, and the leverage of the covariates.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ._io import DosageBlocks, center_impute, check_match, match_variants
from .reference import Reference


def factor_names(sources) -> list:
    """level, theta_<s> for each non-baseline source, lambda_<s>_<t> for s <= t (the last source is the baseline)."""
    s = list(sources)[:-1]
    return (["level"] + [f"theta_{a}" for a in s]
            + [f"lambda_{s[a]}_{s[b]}" for a in range(len(s)) for b in range(a, len(s))])


def factor_loadings(V, O) -> np.ndarray:
    """a_j (M x n_factors) from the sources' dosage variances V (M x K, baseline last) and the contrasts' unbiased
    products O (M x (K-1) x (K-1)): (v_B, v_s - v_B, O_st (twice for s < t))."""
    k1 = O.shape[1]
    cols = [V[:, -1]] + [V[:, a] - V[:, -1] for a in range(k1)]
    cols += [O[:, a, b] * (1.0 if a == b else 2.0) for a in range(k1) for b in range(a, k1)]
    return np.column_stack(cols)


def _sym_cross_cov(U0, U1):
    a, b = U0 - U0.mean(axis=0), U1 - U1.mean(axis=0)
    C = a.T @ b / a.shape[0]
    return (C + C.T) / 2.0


def sample_factors(geno, ref_path, out, *, sources=None, covar=None, keep=None, match="position", step=4096,
                   threads=1, ld_mass=True, memory_gb=32.0, drop_ambiguous=False, log=print) -> dict:
    """Estimate Sigma_u on the GWAS's people (with the GWAS's covariates) and write out.sample_factors.json.

    ld_mass: also the sample's realized in-sample LD mass, in total and over pairs of SNPs on different chromosomes,
    exactly as SUMMIT's in-sample LD scores define it (sum over pairs of V_j V_k (r_jk^2 - 1/nu), each SNP with itself
    included), from the per-chromosome Gram matrices G_c = X~_c X~_c' of the adjusted, standardized genotypes:
    total = <G, G> / nu^2 - (sum V)^2 / nu, between = (<G, G> - sum_c <G_c, G_c>) / nu^2 - ((sum V)^2 - sum_c (sum V_c)^2)
    / nu. A model-free check of a composition (O(n^2 M), the cost of a GRM; two n x n matrices in memory, skipped with a
    note above memory_gb)."""
    from .curve import _cohort_samples
    from .model import contrast_noise
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    sources = list(sources or ref.names[:2])
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("The sample factors need two or more distinct sources.")
    k1 = len(sources) - 1
    samples, rows, C = _cohort_samples(geno, covar, keep, log)
    blocks = DosageBlocks(geno, rows=rows, threads=threads, step=step)
    N = blocks.n
    p_rank = C.shape[1]
    nu = N - 1 - p_rank
    if nu < 10:
        raise ValueError("Too few samples for the number of covariates.")
    rv = ref.variants(sources)
    match_info = {}
    t_idx, r_idx, _flip = match_variants(blocks.variants, rv, how=match, drop_ambiguous=drop_ambiguous,
                                         report=match_info)
    check_match(match_info, "sample-factors: the cohort", log)
    P = np.column_stack([rv[f"p_{s}"].to_numpy()[r_idx] for s in sources])
    Vd = np.column_stack([rv[f"v_{s}"].to_numpy()[r_idx] for s in sources])
    D = P[:, :-1] - P[:, -1:]                                      # the orientation cancels in every loading
    Vn, nn, Gn, _ = ref.frequency_noise(sources, np.nan_to_num(Vd), r_idx, rv)
    S_noise, _ = contrast_noise(Vn, nn, Gn)
    O = D[:, :, None] * D[:, None, :] - S_noise
    a_all = factor_loadings(Vd, O)
    good = np.isfinite(a_all).all(axis=1)
    t_idx, r_idx, a_all = t_idx[good], r_idx[good], a_all[good]
    chrom = rv["CHR"].to_numpy()[r_idx]
    names = factor_names(sources)
    nf = len(names)
    chroms = list(pd.unique(chrom))
    YA = [np.zeros((N, nf)), np.zeros((N, nf))]
    AA = [np.zeros((nf, nf)), np.zeros((nf, nf))]
    R = np.zeros((len(chroms), N))                                 # per chromosome: each person's sum of y
    do_mass = bool(ld_mass) and 2.0 * N * N * 8 / 2 ** 30 <= float(memory_gb)
    G_tot = np.zeros((N, N)) if do_mass else None
    frob_c, Vsum_c = [], []
    Ac = np.zeros((len(chroms), nf))
    n_used = 0
    log(f"[admixed-ld] sample-factors: {t_idx.size} variants matched to the reference ({', '.join(sources)}; "
        f"baseline {sources[-1]}); {N} samples; covariate rank {p_rank}; factors {', '.join(names)}")
    for ci, c in enumerate(chroms):
        ix = np.flatnonzero(chrom == c)
        h = ci % 2
        G_c = np.zeros((N, N)) if do_mass else None
        v_c = 0.0
        for s in range(0, ix.size, step):
            sl = ix[s:s + step]
            X = blocks.read(t_idx[sl])
            center_impute(X)
            s2 = np.einsum("ij,ij->j", X, X) / (N - 1)
            ok = np.isfinite(s2) & (s2 > 1e-8)
            if not ok.any():
                continue
            X = X[:, ok]
            if p_rank:
                X -= C @ (C.T @ X)
            Y = X * X / s2[ok][None, :]
            if do_mass:
                Xs = X / np.sqrt(s2[ok])[None, :]
                G_c += Xs @ Xs.T
                v_c += float(Y.sum()) / nu
            A = a_all[sl][ok] / s2[ok][:, None]
            YA[h] += Y @ A
            AA[h] += A.T @ A
            R[ci] += Y.sum(axis=1)
            Ac[ci] += A.sum(axis=0)
            n_used += int(ok.sum())
        if do_mass:
            frob_c.append(float(np.vdot(G_c, G_c)))
            Vsum_c.append(v_c)
            G_tot += G_c
            del G_c
        log(f"[admixed-ld] sample-factors chr{c}: {ix.size} variants ({time.perf_counter() - t0:.0f} s)")
    U = [YA[h] @ np.linalg.inv(AA[h]) for h in (0, 1)]
    Sigma = _sym_cross_cov(U[0], U[1])
    mean_u = (U[0].mean(axis=0) + U[1].mean(axis=0)) / 2.0
    # diagnostics: the realized between-chromosome same-person mass (the referee's definition) and its prediction
    Rall = R.sum(axis=0)
    tot = float(Rall @ Rall - np.einsum("ci,ci->", R, R))
    vv = float(Rall.sum() ** 2 - (R.sum(axis=1) ** 2).sum())
    realized = tot / nu ** 2 - vv / (N * nu ** 2)
    At = Ac.sum(axis=0)
    predicted = float(N / nu ** 2 * (At @ Sigma @ At - np.einsum("ca,ab,cb->", Ac, Sigma, Ac)))
    ld_info = None
    if do_mass:
        frob_tot = float(np.vdot(G_tot, G_tot))
        Vs = np.array(Vsum_c)
        total = frob_tot / nu ** 2 - Vs.sum() ** 2 / nu
        between = (frob_tot - sum(frob_c)) / nu ** 2 - (Vs.sum() ** 2 - (Vs ** 2).sum()) / nu
        ld_info = dict(total=total, between_chromosomes=between,
                       between_chromosomes_population_part=between - realized,
                       definition="sum over SNP pairs of V_j V_k (r_jk^2 - 1/nu), each SNP with itself included (SUMMIT's "
                                  "in-sample LD scores), V-weighted; from per-chromosome Gram matrices")
        del G_tot
    elif ld_mass:
        log(f"[admixed-ld] sample-factors: the in-sample LD mass needs two {N} x {N} matrices "
            f"({2.0 * N * N * 8 / 2 ** 30:.1f} GiB > memory_gb {memory_gb}); skipped")
    lev = 1.0 / N + (C ** 2).sum(axis=1) if p_rank else np.full(N, 1.0 / N)
    eig = np.linalg.eigvalsh(Sigma)
    info = dict(format="summit-admixed-ld-sample-factors", version=1, sources=sources, baseline=sources[-1],
                factors=names, sigma=Sigma.tolist(), mean=mean_u.tolist(), n_samples=int(N), nu=int(nu),
                sigma_eigenvalues=eig.tolist(),
                covariate_rank=int(p_rank), n_snps=int(n_used),
                halves="least squares per person on the odd and on the even chromosomes (in genome order); sigma is "
                       "the symmetrized covariance between the halves' estimates",
                loadings="(v_B, v_s - v_B, D_s D_t less its sampling covariance, twice for s < t) / s_j^2",
                ld_mass=ld_info,
                diagnostics=dict(between_chromosome_same_person_mass=realized,
                                 predicted_by_factors=predicted,
                                 share_predicted=predicted / realized if realized != 0 else None),
                leverage=dict(mean=float(lev.mean()), sd=float(lev.std()), max=float(lev.max()),
                              quantiles_90_99=[float(x) for x in np.quantile(lev, [0.9, 0.99])]))
    with open(f"{out}.sample_factors.json", "w") as fh:
        json.dump(info, fh, indent=1)
    log(f"[admixed-ld] sample-factors: wrote {out}.sample_factors.json; between-chromosome same-person mass "
        f"{realized:.6g}, factors predict {predicted:.6g} ({predicted / realized:.3f}); leverage mean "
        f"{lev.mean():.4f}, SD {lev.std():.4f}, max {lev.max():.3f} ({time.perf_counter() - t0:.0f} s)")
    return info


def read_sample_factors(path) -> dict:
    p = path if str(path).endswith(".json") else f"{path}.sample_factors.json"
    with open(p) as fh:
        info = json.load(fh)
    if info.get("format") != "summit-admixed-ld-sample-factors":
        raise ValueError(f"{p} is not a sample-factors file.")
    info["path"] = p
    return info


def factor_between_chromosome_mass(a, s2, chrom, sigma, n_sf, nu_sf, nu):
    """The factor term's LD mass over pairs of SNPs on different chromosomes, in compose's units: (n_sf / (nu_sf nu))
    [A' Sigma A - sum_c A_c' Sigma A_c], A = sum_j a_j / s2_j (A_c over chromosome c)."""
    A = a / s2[:, None]
    S = np.asarray(sigma)
    At = A.sum(axis=0)
    within = sum(float(A[chrom == c].sum(axis=0) @ S @ A[chrom == c].sum(axis=0)) for c in pd.unique(chrom))
    return (n_sf / (nu_sf * nu)) * (float(At @ S @ At) - within)


def factor_same_person_term(a, s2, cres, cls, n_classes, sigma, n_sf, nu_sf, nu):
    """(n_sf / (nu_sf nu)) a_j' Sigma_u sum_{k in c} a_k / s2_k / cres_j (M x classes): the same-person term with the
    cohort's person factors; with nu_sf = nu and n_sf = n it is (n / nu^2) Cov(y_j, y_k) summed over the class, the
    referee's definition."""
    Bc = np.zeros((n_classes, a.shape[1]))
    for f in range(a.shape[1]):
        Bc[:, f] = np.bincount(cls, weights=a[:, f] / s2, minlength=n_classes)
    return (n_sf / (nu_sf * nu)) * (a @ np.asarray(sigma) @ Bc.T) / cres[:, None]
