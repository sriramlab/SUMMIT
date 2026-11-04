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
from .model import fit_exponential
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
                   closure=False, cells_cm=2.0, step=4096, threads=1, native=True, noise_correction=True, log=print):
    """The weighted-LD curve of the cohort for two or more sources. With cells_cm, also the genome-wide matrix of
    covariances between cells of cells_cm cM (within and between chromosomes) and the sampling variance of each entry,
    written to out.cells.npz: it carries long-range structure that one value per distance cannot (for example what the
    genotype PCs leave). With K sources and the last as baseline, each of the
    K-1 contrasts D_a = p^a - p^B gives its own cell sums; per distance, the (K-1)^2 products of cell sums are linear in
    the unique entries of Kt_res(d) and are solved by least squares (two sources: numerator / denominator).
    noise_correction: remove the sampling (co)variance of the reference frequencies from the products of contrasts
    of each SNP in the denominators, and from both sides of the global-ancestry regression (the products D D' in its
    normal matrix, and the covariance of the contrasts' errors with the baseline frequency's error in its right-hand
    side). The cohort's covariances in the numerators are linear in each contrast, so their expectation is the
    noise-free product, while raw squared contrasts include the frequencies' sampling variance and shrink the kernel by
    the reliability of the contrasts (errors in variables). The noise model is Reference.frequency_noise: Var(dosage)
    / 4 n_freq per panel, a derived source's per-SNP fvar, and the panels' errors that a derived source's frequencies
    carry (noise_mix). It matters for small panels and derived sources; noise_correction=False (CLI
    --keep-panel-noise) restores the raw products used before 4 Nov 2026."""
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    sources = list(sources or ref.names[:2])
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("The curve needs two or more distinct sources.")
    k1 = len(sources) - 1
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
    P = np.column_stack([rv[f"p_{s}"].to_numpy()[r_idx] for s in sources])
    D = (P[:, :-1] - P[:, -1:]) * sign[:, None]                 # contrasts, oriented to the cohort's counted allele
    pB_c = np.where(flip, 1.0 - P[:, -1], P[:, -1])
    good = np.isfinite(D).all(axis=1) & np.isfinite(pB_c)
    t_idx, r_idx, D, pB_c = t_idx[good], r_idx[good], D[good], pB_c[good]
    if noise_correction:                                       # sampling covariance of the contrasts, per SNP
        from .model import contrast_noise
        Vs = np.column_stack([rv[f"v_{s}"].to_numpy()[r_idx] for s in sources])
        Vn, nn, Gn, _ = ref.frequency_noise(sources, np.nan_to_num(Vs), r_idx, rv)
        S_noise, R_noise = contrast_noise(Vn, nn, Gn)
    else:
        S_noise, R_noise = np.zeros((t_idx.size, k1, k1)), np.zeros((t_idx.size, k1))
    chrom = rv["CHR"].to_numpy()[r_idx]
    cm = rv["CM"].to_numpy()[r_idx]
    log(f"[admixed-ld] curve: {t_idx.size} of {len(blocks.variants)} cohort variants matched to the reference "
        f"({', '.join(sources)}; baseline {sources[-1]}); {N} samples")
    mp = _products(threads, native)
    span = max(float(np.ptp(cm[chrom == c])) for c in pd.unique(chrom))
    max_cm = span if max_cm is None else min(float(max_cm), span)
    L = int(np.ceil(max_cm / cell_cm))
    num = np.zeros((L + 1, k1, k1))                            # sum over cell pairs at lag L of U_a(b) U_b(b + L)
    den = np.zeros((L + 1, k1, k1, k1, k1))                    # sum of Q_au(b) Q_bv(b + L)
    T, Qtot = [], []                                           # per person chromosome totals (projected), and Q
    coarse_U, coarse_q, coarse_tab = [], [], []                # cells of cells_cm for the cell matrix
    th_num = np.zeros((N, k1))
    th_den = np.zeros((k1, k1))
    th_c = np.zeros(k1)
    p_rank = C.shape[1]
    for c in pd.unique(chrom):
        ix = np.flatnonzero(chrom == c)
        ix = ix[np.argsort(cm[ix], kind="stable")]
        cell = ((cm[ix] - cm[ix].min()) / cell_cm).astype(np.int64)
        nc = int(cell.max()) + 1
        Y = np.zeros((N, k1 * nc), order="F")                  # column a * nc + cell
        B = np.zeros((p_rank, k1 * nc))
        q = np.zeros((nc, k1, k1))
        for s in range(0, ix.size, step):
            sl = ix[s:s + step]
            X = blocks.read(t_idx[sl])
            mu = np.nanmean(X, axis=0)
            center_impute(X)
            d = D[sl]
            inf = np.isfinite(mu) & (np.abs(d).max(axis=1) >= theta_min_delta)
            Sn = S_noise[sl]
            if inf.any():                                      # ancestry: g/2 - pB = x/2 + (mu/2 - pB), on D
                th_num += X[:, inf] @ (d[inf] / 2.0)
                th_c += d[inf].T @ (mu[inf] / 2.0 - pB_c[sl][inf]) + R_noise[sl][inf].sum(axis=0)
                th_den += d[inf].T @ d[inf] - Sn[inf].sum(axis=0)
            ss = np.einsum("ij,ij->j", X, X)
            Bx = C.T @ X if p_rank else np.zeros((0, X.shape[1]))
            ss_res = ss - np.einsum("ij,ij->j", Bx, Bx)
            var_res = ss_res / df_res
            ok = np.isfinite(var_res) & (var_res > 1e-8) & np.isfinite(mu)
            if rel is not None and ok.any():
                rel.add(X[:, ok], C, Bx[:, ok], var_res[ok])
            inv = np.zeros(sl.size)
            inv[ok] = 1.0 / var_res[ok]
            cs = cell[s:s + step]
            starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
            for a in range(k1):
                w = d[:, a] * inv                               # a_j / sd_j, applied to unscaled adjusted genotypes
                Y[:, a * nc + cs[starts]] += np.add.reduceat(X * w[None, :], starts, axis=1)
                if p_rank:
                    B[:, a * nc + cs[starts]] += np.add.reduceat(Bx * w[None, :], starts, axis=1)
                for b in range(a, k1):
                    qq = np.bincount(cs, weights=(d[:, a] * d[:, b] - Sn[:, a, b]) * inv, minlength=nc)
                    q[:, a, b] += qq
                    if b != a:
                        q[:, b, a] += qq
        U = Y - C @ B if p_rank else Y
        S = mp.tn(U, U) / df_res
        for lag in range(1, min(L, nc - 1) + 1):
            for a in range(k1):
                for b in range(k1):
                    num[lag, a, b] += float(np.trace(S[a * nc:(a + 1) * nc, b * nc:(b + 1) * nc], offset=lag))
            den[lag] += np.einsum("iau,ibv->abuv", q[:-lag], q[lag:])
        T.append(U.reshape(N, k1, nc).sum(axis=2))
        Qtot.append(q.sum(axis=0))
        if cells_cm:
            per = max(1, int(round(cells_cm / cell_cm)))           # fine cells per coarse cell
            ncc = (nc + per - 1) // per
            Ur = U.reshape(N, k1, nc)
            Uc = np.zeros((N, k1, ncc))
            qc = np.zeros((ncc, k1, k1))
            for cc in range(ncc):
                Uc[:, :, cc] = Ur[:, :, cc * per:(cc + 1) * per].sum(axis=2)
                qc[cc] = q[cc * per:(cc + 1) * per].sum(axis=0)
            coarse_U.append(Uc)
            coarse_q.append(qc)
            coarse_tab.append(pd.DataFrame({"CHR": c, "start_cm": cm[ix].min() + np.arange(ncc) * per * cell_cm,
                                            "width_cm": per * cell_cm}))
        log(f"[admixed-ld] curve chr{c}: {ix.size} SNPs, {nc} cells ({time.perf_counter() - t0:.0f} s)")
    T = np.stack(T, axis=2)                                    # N x k1 x chromosomes
    Qtot = np.stack(Qtot)                                      # chromosomes x k1 x k1
    Ts = T.sum(axis=2)
    inter_num = (Ts.T @ Ts - np.einsum("iac,ibc->ab", T, T)) / df_res
    qs = Qtot.sum(axis=0)
    inter_den = np.einsum("au,bv->abuv", qs, qs) - np.einsum("cau,cbv->abuv", Qtot, Qtot)
    inter = _solve_kernel(inter_num, inter_den)
    dist = np.arange(1, L + 1) * cell_cm
    kern = np.full((L, k1, k1), np.nan)
    pos = np.array([den[lag].reshape(k1 * k1, -1).any() for lag in range(1, L + 1)])
    for i in np.flatnonzero(pos):
        kern[i] = _solve_kernel(num[i + 1], den[i + 1])
    keep_rows = pos & np.isfinite(kern).all(axis=(1, 2))
    from .model import contrast_pairs, full_from_contrasts, moments_from_vectors
    if k1 == 1:                                                # the two-source layout, unchanged
        curve = pd.DataFrame({"d_cm": dist, "numerator": num[1:, 0, 0], "denominator": den[1:, 0, 0, 0, 0],
                              "kernel": kern[:, 0, 0]})
        curve = curve[curve.denominator > 0]
    else:
        curve = pd.DataFrame({"d_cm": dist})
        for a, b in contrast_pairs(k1):
            curve[f"kernel_{sources[a]}_{sources[b]}"] = kern[:, a, b]
        curve = curve[keep_rows]
    kfull = full_from_contrasts(kern[keep_rows])                # K x K covariance of local-ancestry dosages
    fits = {s: fit_exponential(dist[keep_rows], kfull[:, i, i], 2.0, 20.0) for i, s in enumerate(sources)}
    U_th = (th_num + th_c[None, :]) @ np.linalg.inv(th_den)    # people x k1: ancestries of the non-baseline sources
    info = dict(format="summit-admixed-ld-curve", version=1, sources=list(sources), baseline=sources[-1],
                n_samples=int(N), n_snps=int(t_idx.size), covariate_rank=int(p_rank), cell_cm=float(cell_cm),
                noise_correction=bool(noise_correction),
                panel_n_freq={s: float(ref.n_freq[s]) for s in sources},
                max_cm=float(max_cm), inter_matrix=inter.tolist(), fit_per_source=fits,
                theta=moments_from_vectors(U_th),
                reference=dict(sources=ref.manifest["sources"], window_kb=ref.manifest["window_kb"],
                               genetic_map=ref.manifest["genetic_map"]))
    if k1 == 1:
        info.update(inter_numerator=float(inter_num[0, 0]), inter_denominator=float(inter_den[0, 0, 0, 0]),
                    inter_kernel=float(inter[0, 0]), fit_2_to_20cm=fit_exponential(curve.d_cm, curve.kernel, 2.0, 20.0))
    if rel is not None:
        info["relatedness"] = dict(**rel.summary(), kinship_col=kinship_col, kinship_min=kinship_min, closure=closure,
                                   definition="relatedness of the covariate-adjusted, standardized genotypes")
        r = info["relatedness"]
        log(f"[admixed-ld] curve: {r['pairs']} relative pairs, mean relatedness after the covariates {r['mean']:.4f} "
            f"(largest |g| {r['max_abs']:.3f}); sum over ordered pairs of g^2 {r['sum_squared_ordered']:.4g}")
    if cells_cm and coarse_U:
        # two random halves of the people give independent estimates of each cell-pair covariance, whose product is
        # unbiased for its square (and, with K sources, for the products of kernel entries)
        Uall = np.concatenate(coarse_U, axis=2)                 # N x k1 x cells
        nC = Uall.shape[2]
        rng_h = np.random.default_rng(20261104)
        if rel is not None:                                    # relatives in one half, so the halves stay independent
            fam = rp.family[rows]
            labels, inv = np.unique(fam, return_inverse=True)
            side = rng_h.permutation(labels.size) % 2 == 0
            half = side[inv]
        else:
            half = rng_h.permutation(N) < N // 2
        Sh = []
        for h in (half, ~half):
            Uh = Uall[h].reshape(int(h.sum()), k1 * nC)          # column a * n_cells + cell
            Sh.append(mp.tn(Uh, Uh) / (h.sum() - 1 - p_rank))
        tab = pd.concat(coarse_tab, ignore_index=True)
        np.savez(f"{out}.cells.npz", S1=Sh[0], S2=Sh[1], Q=np.concatenate(coarse_q), chrom=tab.CHR.to_numpy().astype(str),
                 start_cm=tab.start_cm.to_numpy(), width_cm=tab.width_cm.to_numpy(), k1=k1)
        info["cells"] = dict(file=f"{out}.cells.npz", n_cells=int(nC), cells_cm=float(cells_cm),
                             estimator="split halves")
        del Uall
    curve.to_csv(f"{out}.curve.tsv", sep="\t", index=False, float_format="%.8g")
    with open(f"{out}.curve.json", "w") as f:
        json.dump(info, f, indent=1)
    blocks.close()
    mp.drain()
    mean_th = np.asarray(info["theta"]["mean_vector"])
    log(f"[admixed-ld] curve: between chromosomes " + np.array2string(inter, precision=5).replace("\n", "")
        + "; decay per cM (2-20 cM) " + ", ".join(f"{s} {fits[s]['lam']:.4f}" for s in sources)
        + "; mean ancestry " + ", ".join(f"{s} {x:.3f}" for s, x in zip(sources[:-1], mean_th))
        + f". Wrote {out}.curve.tsv and {out}.curve.json ({time.perf_counter() - t0:.0f} s)")
    return curve, info


def _solve_kernel(num, den):
    """Unique entries of the symmetric Kt from num[a, b] = sum_{u,v} Kt_uv den[a, b, u, v] (least squares)."""
    k1 = num.shape[0]
    if k1 == 1:
        return np.array([[num[0, 0] / den[0, 0, 0, 0]]]) if den[0, 0, 0, 0] > 0 else np.full((1, 1), np.nan)
    from .model import contrast_pairs
    pairs = contrast_pairs(k1)
    A = np.column_stack([den[:, :, u, v].ravel() + (den[:, :, v, u].ravel() if u != v else 0.0) for u, v in pairs])
    sol, *_ = np.linalg.lstsq(A, num.ravel(), rcond=None)
    out = np.zeros((k1, k1))
    for (u, v), x in zip(pairs, sol):
        out[u, v] = out[v, u] = x
    return out
