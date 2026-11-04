"""The cohort's weighted-LD curve (`summit admixed-ld curve`), an aggregate that it can release with its GWAS.

With x_ij the covariate-adjusted genotype of SNP j scaled to unit variance and a_j = delta_j / sd_j (delta from the
reference sources, sd_j the adjusted genotype SD), each person's cell sums U_ib = sum_{j in cell b} a_j x_ij over cells
of `cell_cm` cM satisfy, for cells beyond the range of within-source LD,
    E(U_ib U_ib') = kappa_res(d) * sum_{j in b} a_j^2 * sum_{k in b'} a_k^2.
Summing (1/N) sum_i U_ib U_ib' over all pairs of cells L cells apart and dividing by sum q_b q_b' (q_b = sum a_j^2)
estimates kappa_res at L * cell_cm. The same ratio over pairs of chromosomes gives its between-chromosome value. This
is the weighted LD curve of ROLLOFF and ALDER, with ALDER's binning, normalized to the scale of the genotype covariance.

Computation: one pass over the genotypes. Projection on the covariates is linear, so it is applied to the cell sums
(samples x cells) instead of to every SNP: U = Y - C (C'Y) with Y = X diag(w) S, where S assigns SNPs to cells.
The pass also gives each person's global ancestry, theta_i = sum_j delta_j (g_ij / 2 - p^E_j) / sum_j delta_j^2 over
differentiated SNPs, whose moments enter the same-person term, and, for listed relative pairs, their relatedness in the
covariate-adjusted, standardized genotypes, g_ab = (1/M) sum_j x_aj x_bj. The relatives' part of the in-sample LD scores
depends on these values, not on relatedness before adjustment: in a small cohort a large family can take up a PC of its
own, which removes most of its members' relatedness. Only aggregates are written.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ._io import DosageBlocks, center_impute, match_variants
from .model import fit_exponential, moments_from_values
from .reference import Reference, _products


def _cohort_samples(blocks_path, covar, keep, log):
    """Rows (indices into the genotype samples) with every covariate and in --keep, and the orthonormal covariate basis
    for those rows (empty basis without covariates)."""
    from ..ldscore.genotype_source import read_fam_sample_ids, read_psam_sample_ids, resolve_genotype_input
    from ..ldscore.gw_ldscore import read_cov
    gi = resolve_genotype_input(blocks_path)
    samples = read_fam_sample_ids(gi.sample_path) if gi.format == "bed" else read_psam_sample_ids(gi.sample_path)
    idx = np.arange(len(samples))
    if keep is not None:
        k = pd.read_csv(keep, sep=r"\s+", header=None, dtype=str)
        key = samples["FID"] + "\t" + samples["IID"] if k.shape[1] >= 2 else samples["IID"]
        want = set((k[0] + "\t" + k[1]) if k.shape[1] >= 2 else k[0])
        idx = idx[key.isin(want).to_numpy()]
        if idx.size == 0:
            raise ValueError(f"No genotype samples are in {keep}.")
    if covar is None:
        return samples, idx, np.zeros((idx.size, 0))
    C, _, kept = read_cov(cov_filename=covar, fam_filename=None, std=True, cov_impute_method="ignore",
                          logger=None, ddof=1, sample_ids=samples.iloc[idx].reset_index(drop=True))
    rows = idx[np.asarray(kept, dtype=np.int64)]
    log(f"[admixed-ld] curve: {rows.size} samples with every covariate; covariate rank {C.shape[1]}")
    return samples, rows, np.asarray(C, dtype=np.float64)


class _PairRelatedness:
    """Relatedness of relative pairs in the covariate-adjusted, standardized genotypes, accumulated over blocks:
    g_ab = (1/M) sum_j x_aj x_bj with x_j = (g_j - C C' g_j) / sd_res_j. `a`, `b`: positions within the cohort rows."""

    _CHUNK = 1 << 24                         # elements of one pairs x SNPs product (128 MiB in float64)

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self.members, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
        self.ia, self.ib = inv[:a.size], inv[a.size:]
        self.dot = np.zeros(a.size)
        self.m = 0

    def add(self, X: np.ndarray, C: np.ndarray, Bx: np.ndarray, var_res: np.ndarray):
        """X: centred genotypes of the block (samples x SNPs); Bx = C'X; var_res: residual variances (SNPs used)."""
        if self.members.size == 0 or X.shape[1] == 0:
            return
        Xu = X[self.members]
        if C.shape[1]:
            Xu = Xu - C[self.members] @ Bx
        Xu = Xu / np.sqrt(var_res)[None, :]
        step = max(1, self._CHUNK // X.shape[1])
        for s in range(0, self.ia.size, step):
            self.dot[s:s + step] += np.einsum("ij,ij->i", Xu[self.ia[s:s + step]], Xu[self.ib[s:s + step]])
        self.m += X.shape[1]

    def summary(self) -> dict:
        g = self.dot / max(self.m, 1)
        return dict(sum_squared_ordered=float(2.0 * np.sum(g * g)), pairs=int(g.size), n_snps=int(self.m),
                    mean=float(g.mean()) if g.size else float("nan"),
                    max_abs=float(np.abs(g).max()) if g.size else float("nan"))


def genotype_curve(geno, ref_path, out, *, sources=None, covar=None, keep=None, match="position", cell_cm=0.25,
                   max_cm=None, theta_min_delta=0.2, relative_pairs=None, kinship_col=None, kinship_min=None,
                   closure=False, step=4096, threads=1, native=True, log=print):
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    sources = sources or ref.names[:2]
    if len(sources) != 2:
        raise ValueError("The curve needs exactly two sources.")
    A, E = sources
    samples, rows, C = _cohort_samples(geno, covar, keep, log)
    from ..relatives.moments import _auto_step
    step = min(int(step), _auto_step(None, rows.size))           # about 512 MiB of float64 genotypes per block
    blocks = DosageBlocks(geno, rows=rows, threads=threads, step=step)
    N = blocks.n
    rel = None
    if relative_pairs is not None:                             # SUMMIT's pair reader, with its options
        from ..relatives.moments import pairs_within_sample, read_relative_pairs
        rp = read_relative_pairs(relative_pairs, samples, kinship_col=kinship_col, kinship_min=kinship_min,
                                 closure=False)
        if closure:                                            # a dense list of distant pairs can join most people
            sizes = np.bincount(np.unique(rp.family, return_inverse=True)[1])
            n_closed = int(np.sum(sizes * (sizes - 1) // 2))
            if n_closed > max(10 * rp.npairs, 10 ** 7):
                raise ValueError(f"Closing the {rp.npairs} listed pairs into families gives {n_closed} pairs (largest "
                                 f"family {int(sizes.max())} people). Use the listed pairs (without --closure) or a "
                                 "higher --kinship-min.")
            rp = read_relative_pairs(relative_pairs, samples, kinship_col=kinship_col, kinship_min=kinship_min,
                                     closure=True)
        pa, pb = pairs_within_sample(rp, rows, len(samples))
        rel = _PairRelatedness(pa, pb)
        log(f"[admixed-ld] curve: {pa.size} relative pairs among the {N} samples")
    df_res = N - 1 - C.shape[1]                                 # degrees of freedom of the adjusted genotypes
    if df_res < 10:
        raise ValueError("Too few samples for the number of covariates.")
    rv = ref.variants(sources)
    t_idx, r_idx, flip = match_variants(blocks.variants, rv, how=match)
    sign = np.where(flip, -1.0, 1.0)
    pA, pE = rv[f"p_{A}"].to_numpy()[r_idx], rv[f"p_{E}"].to_numpy()[r_idx]
    delta = (pA - pE) * sign                                   # oriented to the cohort's counted allele
    pE_c = np.where(flip, 1.0 - pE, pE)
    good = np.isfinite(delta) & np.isfinite(pE_c)
    t_idx, r_idx, delta, pE_c = t_idx[good], r_idx[good], delta[good], pE_c[good]
    chrom = rv["CHR"].to_numpy()[r_idx]
    cm = rv["CM"].to_numpy()[r_idx]
    log(f"[admixed-ld] curve: {t_idx.size} of {len(blocks.variants)} cohort variants matched to the reference; "
        f"{N} samples")
    mp = _products(threads, native)
    span = max(float(np.ptp(cm[chrom == c])) for c in pd.unique(chrom))
    max_cm = span if max_cm is None else min(float(max_cm), span)
    L = int(np.ceil(max_cm / cell_cm))
    num, den = np.zeros(L + 1), np.zeros(L + 1)
    T = []                                                     # per person chromosome totals (projected)
    Q = []
    theta_num = np.zeros(N)
    theta_den = 0.0
    theta_c = 0.0
    p_rank = C.shape[1]
    for c in pd.unique(chrom):
        ix = np.flatnonzero(chrom == c)
        ix = ix[np.argsort(cm[ix], kind="stable")]
        cell = ((cm[ix] - cm[ix].min()) / cell_cm).astype(np.int64)
        nc = int(cell.max()) + 1
        Y = np.zeros((N, nc), order="F")
        B = np.zeros((p_rank, nc))
        q = np.zeros(nc)
        for s in range(0, ix.size, step):
            sl = ix[s:s + step]
            X = blocks.read(t_idx[sl])
            mu = np.nanmean(X, axis=0)
            center_impute(X)
            d = delta[sl]
            inf = np.isfinite(mu) & (np.abs(d) >= theta_min_delta)
            if inf.any():                                      # global ancestry: g/2 - pE = x/2 + (mu/2 - pE)
                theta_num += X[:, inf] @ (d[inf] / 2.0)
                theta_c += float(np.sum(d[inf] * (mu[inf] / 2.0 - pE_c[sl][inf])))
                theta_den += float(np.sum(d[inf] ** 2))
            ss = np.einsum("ij,ij->j", X, X)
            Bx = C.T @ X if p_rank else np.zeros((0, X.shape[1]))
            ss_res = ss - np.einsum("ij,ij->j", Bx, Bx)
            var_res = ss_res / df_res
            ok = np.isfinite(var_res) & (var_res > 1e-8) & np.isfinite(mu)
            if rel is not None and ok.any():
                rel.add(X[:, ok], C, Bx[:, ok], var_res[ok])
            w = np.zeros(sl.size)
            w[ok] = d[ok] / var_res[ok]                         # a_j / sd_j, applied to unscaled adjusted genotypes
            a2 = np.zeros(sl.size)
            a2[ok] = d[ok] ** 2 / var_res[ok]
            cs = cell[s:s + step]
            starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
            Y[:, cs[starts]] += np.add.reduceat(X * w[None, :], starts, axis=1)
            if p_rank:
                B[:, cs[starts]] += np.add.reduceat(Bx * w[None, :], starts, axis=1)
            q += np.bincount(cs, weights=a2, minlength=nc)
        U = Y - C @ B if p_rank else Y
        S = mp.tn(U, U) / df_res
        for lag in range(1, min(L, nc - 1) + 1):
            num[lag] += float(np.trace(S, offset=lag))
            den[lag] += float(q[:-lag] @ q[lag:])
        T.append(U.sum(axis=1))
        Q.append(float(q.sum()))
        log(f"[admixed-ld] curve chr{c}: {ix.size} SNPs, {nc} cells ({time.perf_counter() - t0:.0f} s)")
    T = np.column_stack(T)
    Qv = np.asarray(Q)
    inter_num = float(np.sum(T.sum(axis=1) ** 2 - np.sum(T * T, axis=1)) / df_res)
    inter_den = float(Qv.sum() ** 2 - np.sum(Qv * Qv))
    d = np.arange(1, L + 1) * cell_cm
    with np.errstate(divide="ignore", invalid="ignore"):
        k = num[1:] / den[1:]
    curve = pd.DataFrame({"d_cm": d, "numerator": num[1:], "denominator": den[1:], "kernel": k})
    curve = curve[curve.denominator > 0]
    fit = fit_exponential(curve.d_cm, curve.kernel, 2.0, 20.0)
    theta = (theta_num + theta_c) / theta_den
    info = dict(format="summit-admixed-ld-curve", version=1, sources=list(sources), n_samples=int(N),
                n_snps=int(t_idx.size), covariate_rank=int(p_rank), cell_cm=float(cell_cm), max_cm=float(max_cm),
                inter_numerator=inter_num, inter_denominator=inter_den,
                inter_kernel=(inter_num / inter_den if inter_den > 0 else float("nan")),
                fit_2_to_20cm=fit, theta=moments_from_values(theta),
                reference=dict(sources=ref.manifest["sources"], window_kb=ref.manifest["window_kb"],
                               genetic_map=ref.manifest["genetic_map"]))
    if rel is not None:
        info["relatedness"] = dict(**rel.summary(), kinship_col=kinship_col, kinship_min=kinship_min, closure=closure,
                                   definition="relatedness of the covariate-adjusted, standardized genotypes")
        r = info["relatedness"]
        log(f"[admixed-ld] curve: {r['pairs']} relative pairs, mean relatedness after the covariates {r['mean']:.4f} "
            f"(largest |g| {r['max_abs']:.3f}); sum over ordered pairs of g^2 {r['sum_squared_ordered']:.4g}")
    curve.to_csv(f"{out}.curve.tsv", sep="\t", index=False, float_format="%.8g")
    with open(f"{out}.curve.json", "w") as f:
        json.dump(info, f, indent=1)
    blocks.close()
    mp.drain()
    log(f"[admixed-ld] curve: kernel at 2 cM {np.interp(2.0, curve.d_cm, curve.kernel):.4f}, between chromosomes "
        f"{info['inter_kernel']:.5f}, decay {fit['lam']:.4f} per cM (2-20 cM); mean ancestry {info['theta']['mean']:.3f} "
        f"(SD {np.sqrt(info['theta']['var']):.3f}). Wrote {out}.curve.tsv and {out}.curve.json "
        f"({time.perf_counter() - t0:.0f} s)")
    return curve, info
