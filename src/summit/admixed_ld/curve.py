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
own, which removes most of its members' relatedness. A pass of its own then gives the listed pairs' local relatedness
products by distance on one chromosome and between chromosomes, net of random control pairs (_RelativesProfile), and,
with background=True, the cross-person LD of all pairs of people among the least differentiated SNPs (_Background).
Only aggregates are written.
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


def _control_pairs(n, pa, pb, count, seed=20261105):
    """`count` distinct random pairs (a < b) of the n cohort rows that are not listed (control pairs)."""
    total = n * (n - 1) // 2
    listed = np.unique(np.minimum(pa, pb).astype(np.int64) * n + np.maximum(pa, pb))
    count = int(min(count, total - listed.size))
    if count <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rng = np.random.default_rng(seed)
    got = np.zeros(0, np.int64)
    while got.size < count:
        k = 2 * (count - got.size) + 16
        a, b = rng.integers(0, n, k), rng.integers(0, n, k)
        a, b = a[a != b], b[a != b]
        key = np.minimum(a, b) * n + np.maximum(a, b)
        key = key[~np.isin(key, listed)]
        _, first = np.unique(key, return_index=True)       # keep the order of the draws
        key = key[np.sort(first)]
        got = np.concatenate([got, key[~np.isin(key, got)]])
    got = got[:count]
    return got // n, got % n


def _lag_traces(G, nc):
    """Sums of the diagonals of the symmetric cells x cells matrix G: out[L] = sum_b G[b, b + L], L = 0 .. nc - 1."""
    out = np.zeros(nc)
    for L in range(nc):
        out[L] = np.trace(G, offset=L)
    return out


class _RelativesProfile:
    """Local relatedness of the listed relative pairs, and of random control pairs, in the covariate-adjusted,
    standardized genotypes u (as _PairRelatedness), accumulated chromosome by chromosome in a pass of its own.

    For a pair p = (a, b) the local relatedness at SNP j is y_j = u_aj u_bj, with cell sums R_pb = sum_{j in b} y_j over
    cells of `cell_cm` and chromosome sums G_pc = sum_{j in c} y_j. Written (sums over ordered pairs, i.e. twice the
    sums over listed pairs; aggregates only):
      lag sums      P(L) = 2 sum_p sum_b R_pb R_p,b+L (L >= 1), P(0) = 2 sum_p sum_b (R_pb^2 - sum_{j in b} y_j^2),
                    with the number of SNP pairs W(L) = sum_b n_b n_{b+L} (W(0) = sum_b n_b (n_b - 1) / 2) and their
                    mean genetic distance;
      between       Pb = 2 sum_p [(sum_c G_pc)^2 - sum_c G_pc^2] over Wb = M^2 - sum_c M_c^2 ordered SNP pairs,
    the same for the control pairs, and g = sum_c G_pc / M per listed pair (the uniform term's relatedness). The
    control pairs give what a pair of unrelated members contributes to the same sums (within-source and admixture LD
    of the two people, which the population part of the LD score already holds, and the covariates' projection), so
    phi(L) = [P(L) - (n_listed / n_controls) Pc(L)] / W(L) is the relatives' excess per SNP pair at that distance, and
    B = [Pb - ratio Pbc] / Wb between chromosomes; both are sums over ordered pairs, the scale of g^2."""

    _CHUNK = 1 << 24                         # elements of one pairs x SNPs product (128 MiB in float64)

    def __init__(self, a, b, ca, cb, cell_cm, memory_gb=4.0, threads=1):
        self.a = np.concatenate([a, ca]).astype(np.int64)
        self.b = np.concatenate([b, cb]).astype(np.int64)
        self.n_listed, self.n_controls = int(a.size), int(ca.size)
        self.cell_cm, self.memory = float(cell_cm), float(memory_gb) * 2 ** 30
        self.threads = max(1, int(threads))
        if self.threads > 1:                     # numpy releases the GIL in the products and reductions
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(self.threads)
        self.dot = np.zeros(self.a.size)                 # sum_c G_pc
        self.sq = np.zeros(self.a.size)                  # sum_c G_pc^2
        self.m = 0
        self.mc = []
        self.chr = []

    def chunks(self, nc):
        """Pair ranges whose cell sums (pairs x nc, float64) fit the memory budget, one genotype pass each."""
        size = max(64, int(self.memory // (8 * max(nc, 1))))
        return [(s, min(self.a.size, s + size)) for s in range(0, self.a.size, size)]

    def chromosome(self, name, read_block, ix_blocks, cell, nc, cm):
        """One chromosome: `read_block(sl, rows)` returns (u, ok) for SNP positions sl (the given cohort rows x SNPs,
        adjusted and standardized; ok marks SNPs with variance); `ix_blocks` the blocks of positions in genetic order,
        `cell` the cell of each position, `cm` its genetic position."""
        P = {"listed": np.zeros(nc), "controls": np.zeros(nc)}
        okc = None
        for c0, c1 in self.chunks(nc):
            a, b = self.a[c0:c1], self.b[c0:c1]
            mem, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
            ia, ib = inv[:a.size], inv[a.size:]
            R = np.zeros((c1 - c0, nc))
            G = np.zeros(c1 - c0)
            ysq = np.zeros(c1 - c0)
            ok_all = []
            for sl in ix_blocks:
                u, ok = read_block(sl, mem)
                ok_all.append(ok)
                cs = cell[sl][ok]
                if cs.size == 0:
                    continue
                u = np.ascontiguousarray(u[:, ok])
                starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
                step = max(1, min(self._CHUNK // u.shape[1], -(-a.size // self.threads)))
                cols = cs[starts]

                def work(s, u=u, cols=cols, starts=starts, step=step):   # rows s..s+step of R, G, ysq (disjoint)
                    Pr = u[ia[s:s + step]] * u[ib[s:s + step]]
                    R[s:s + step, cols] += np.add.reduceat(Pr, starts, axis=1)
                    G[s:s + step] += Pr.sum(axis=1)
                    ysq[s:s + step] += np.einsum("ij,ij->i", Pr, Pr)

                if self.threads > 1:
                    list(self.pool.map(work, range(0, a.size, step)))
                else:
                    for s in range(0, a.size, step):
                        work(s)
            if okc is None:
                okc = np.concatenate(ok_all)
            for kind, lo, hi in (("listed", c0, min(c1, self.n_listed)), ("controls", max(c0, self.n_listed), c1)):
                if hi <= lo:
                    continue
                Rk = R[lo - c0:hi - c0]
                t = _lag_traces(Rk.T @ Rk, nc)
                t[0] = (t[0] - ysq[lo - c0:hi - c0].sum()) / 2.0    # unordered SNP pairs within a cell, as W(0)
                P[kind] += 2.0 * t                                  # ordered pairs of people
            self.dot[c0:c1] += G
            self.sq[c0:c1] += G * G
        # SNP pairs per lag and their mean distance (SNPs with variance only)
        pos = np.concatenate(ix_blocks)
        cl, x = cell[pos][okc], cm[pos][okc]
        nb = np.bincount(cl, minlength=nc).astype(np.float64)
        sb = np.bincount(cl, weights=x, minlength=nc)
        w, dsum = np.zeros(nc), np.zeros(nc)
        for L in range(1, nc):
            w[L] = float(nb[:-L] @ nb[L:])
            dsum[L] = float(nb[:-L] @ sb[L:] - nb[L:] @ sb[:-L])
        w[0] = float((nb * (nb - 1)).sum() / 2.0)
        # within a cell, sum over j < k of x_k - x_j = sum_k x_(k) (2 r_k - n_b + 1) with x sorted and r its rank
        o = np.lexsort((x, cl))
        xs, cs_ = x[o], cl[o]
        first = np.r_[0, np.flatnonzero(np.diff(cs_)) + 1]
        size = np.diff(np.r_[first, xs.size])
        rank = np.arange(xs.size) - np.repeat(first, size)
        dsum[0] = float((xs * (2 * rank - np.repeat(size, size) + 1)).sum())
        self.chr.append(dict(name=str(name), listed=P["listed"], controls=P["controls"], w=w, dsum=dsum))
        self.mc.append(int(okc.sum()))
        self.m += int(okc.sum())

    def summary(self):
        nl, nct = self.n_listed, self.n_controls
        g = self.dot[:nl] / max(self.m, 1)
        ratio = nl / nct if nct else 0.0
        between = 2.0 * (self.dot * self.dot - self.sq)
        bl, bc = float(between[:nl].sum()), float(between[nl:].sum())
        wb = float(self.m) ** 2 - float(np.sum(np.square(np.asarray(self.mc, np.float64))))
        return dict(sum_squared_ordered=float(2.0 * np.sum(g * g)), pairs=int(nl), n_snps=int(self.m),
                    mean=float(g.mean()) if g.size else float("nan"),
                    max_abs=float(np.abs(g).max()) if g.size else float("nan"),
                    between=dict(listed=bl, controls=bc, n_controls=int(nct), w=wb,
                                 value=(bl - ratio * bc) / wb if wb > 0 else float("nan")))

    def table(self):
        """Lag sums pooled over chromosomes, with the relatives' excess per SNP pair phi."""
        L = max(x["w"].size for x in self.chr)
        pad = lambda v: np.pad(v, (0, L - v.size))
        tot = {k: sum(pad(x[k]) for x in self.chr) for k in ("listed", "controls", "w", "dsum")}
        ratio = self.n_listed / self.n_controls if self.n_controls else 0.0
        use = tot["w"] > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.where(use, (tot["listed"] - ratio * tot["controls"]) / tot["w"], np.nan)
            d = np.where(use, tot["dsum"] / tot["w"], np.nan)
        return pd.DataFrame({"lag": np.arange(L), "d_cm": d, "w": tot["w"], "listed": tot["listed"],
                             "controls": tot["controls"], "phi": phi})[use]


class _Background:
    """All ordered pairs of people, without a list: the cross-person LD among the least differentiated SNPs (aggregates).

    For SNPs j, k with adjusted, standardized genotypes u (unit residual variance), the cross-person sum
    x_jk = (u_j . u_k)^2 - sum_i u_ij^2 u_ik^2 = sum over ordered pairs i != i' of y_j(i, i') y_k(i, i') is what all pairs
    of people add to nu^2 r_jk^2 beyond each person with themselves. Its expectation is n(n - 1) rho_jk^2 (population LD)
    + p_tot nu / n (the covariates' projection, which SUMMIT's in-sample noise correction removes) + the relatedness of
    every pair, listed or not. Among SNPs whose source contrast is small (lowest `quantile` of max_a |D_ja|), rho_jk^2
    beyond within-source LD is only the admixture term (a_j' Kt(d) a_k)^2 with a = D / sd, which the curve's own kernel
    gives and which is subtracted; what remains is the relatedness LD of all pairs, a property of the people that does
    not depend on the SNPs. Within chromosomes: exact sums over SNP pairs by lag of `cell_cm` cells (per chromosome,
    n x m_c genotypes held); between chromosomes: exact over at most `between_snps` of these SNPs (n x m float32 held).
    Below a few cM the same sums also hold within-source LD, which is not subtracted (compose uses the measurement only
    beyond its --background-min-cm)."""

    def __init__(self, select, N, df_res, p_rank, cell_cm, k1, between_snps, seed=20261105):
        self.select = np.asarray(select, bool)                 # per matched SNP
        self.N, self.df_res, self.ptot = int(N), int(df_res), int(p_rank) + 1
        self.cell_cm, self.k1 = float(cell_cm), int(k1)
        idx = np.flatnonzero(self.select)
        rng = np.random.default_rng(seed)
        keep = idx if idx.size <= between_snps else np.sort(rng.choice(idx, int(between_snps), replace=False))
        self.between_mask = np.zeros(self.select.size, bool)
        self.between_mask[keep] = True
        self.cur, self.cur_o, self.cur_x = [], [], []
        self.bU, self.bO, self.bchr = [], [], []
        self.x, self.w, self.dsum, self.oo = None, None, None, None
        self.n_snps = 0

    def add(self, pos, X, C, Bx, var_res, ok, Ofl, cmv, chrom_name):
        """A block of the curve's pass: pos (matched-SNP indices), centred genotypes X, Bx = C'X, residual variances,
        ok, the products of contrasts (D D' - S) / var_res flattened (k1^2 per SNP) and the genetic positions."""
        use = ok & self.select[pos]
        if not use.any():
            return
        Xu = X[:, use] - C @ Bx[:, use] if C.shape[1] else X[:, use].copy()
        Xu /= np.sqrt(var_res[use])[None, :]
        self.cur.append(Xu)
        self.cur_o.append(Ofl[use])
        self.cur_x.append(cmv[use])
        bsel = self.between_mask[pos][use]
        if bsel.any():
            self.bU.append(Xu[:, bsel].astype(np.float32))
            self.bO.append(Ofl[use][bsel])
            self.bchr += [str(chrom_name)] * int(bsel.sum())

    def end_chromosome(self, cm_min):
        if not self.cur:
            return
        U = np.concatenate(self.cur, axis=1)
        O = np.concatenate(self.cur_o)
        x = np.concatenate(self.cur_x)
        self.cur, self.cur_o, self.cur_x = [], [], []
        o = np.argsort(x, kind="stable")
        U, O, x = U[:, o], O[o], x[o]
        cell = ((x - cm_min) / self.cell_cm).astype(np.int64)
        U2 = U * U
        m = U.shape[1]
        self.n_snps += m
        k2 = O.shape[1]
        for s0 in range(0, m, 1024):
            s1 = min(m, s0 + 1024)
            T = (U[:, s0:s1].T @ U[:, s0:], U2[:, s0:s1].T @ U2[:, s0:])
            T = T[0] * T[0] - T[1]                              # j in s0..s1, k >= s0
            jj, kk = np.nonzero(np.arange(s0, s1)[:, None] < np.arange(s0, m)[None, :])
            lag = np.abs(cell[s0 + jj] - cell[s0 + kk])
            dd = np.abs(x[s0 + jj] - x[s0 + kk])
            L = int(lag.max()) + 1 if lag.size else 1
            self._add("x", np.bincount(lag, weights=T[jj, kk], minlength=L))
            self._add("w", np.bincount(lag, minlength=L).astype(np.float64))
            self._add("dsum", np.bincount(lag, weights=dd, minlength=L))
            oo = np.zeros((L, k2, k2))
            for u_ in range(k2):
                for v_ in range(k2):
                    oo[:, u_, v_] = np.bincount(lag, weights=O[s0 + jj, u_] * O[s0 + kk, v_], minlength=L)
            self._add("oo", oo)

    def _add(self, key, v):
        cur = getattr(self, key)
        if cur is None:
            setattr(self, key, v.copy())
            return
        if cur.shape[0] < v.shape[0]:
            cur = np.concatenate([cur, np.zeros((v.shape[0] - cur.shape[0],) + cur.shape[1:])])
        elif v.shape[0] < cur.shape[0]:
            v = np.concatenate([v, np.zeros((cur.shape[0] - v.shape[0],) + v.shape[1:])])
        cur += v
        setattr(self, key, cur)

    def between(self):
        """Sums over ordered pairs of the subsample's SNPs on different chromosomes."""
        if not self.bU:
            return dict(x=0.0, w=0.0, oo=np.zeros((1, 1)), n_snps=0)
        U = np.concatenate(self.bU, axis=1)
        O = np.concatenate(self.bO)
        ch = np.asarray(self.bchr)
        self.bU = None
        xs, ws = 0.0, 0.0
        oo = np.zeros((O.shape[1], O.shape[1]))
        for s0 in range(0, U.shape[1], 1024):
            s1 = min(U.shape[1], s0 + 1024)
            A = U[:, s0:s1].astype(np.float64)
            G = A.T @ U
            F = (A * A).T @ (U * U)
            diff = ch[s0:s1][:, None] != ch[None, :]
            xs += float(((G * G - F) * diff).sum())
            ws += float(diff.sum())
            oo += O[s0:s1].T @ (diff @ O)
        return dict(x=xs, w=ws, oo=oo, n_snps=int(U.shape[1]))

    def finish(self, kernel_at, far):
        """Table by lag and the between-chromosome value; kernel_at(d) -> k1 x k1 kernels (n_d x k1 x k1)."""
        n = self.N
        null = self.ptot * self.df_res / n
        w = self.w
        d = np.where(w > 0, self.dsum / np.maximum(w, 1), np.nan)
        Kd = kernel_at(np.nan_to_num(d))
        k1 = self.k1
        F = np.einsum("lab,lce->lacbe", Kd, Kd).reshape(d.size, k1 * k1, k1 * k1)
        adm = n * (n - 1) * np.einsum("luv,luv->l", F, self.oo)
        tab = pd.DataFrame({"lag": np.arange(w.size), "d_cm": d, "w": w, "x": self.x, "adm": adm,
                            "excess": np.where(w > 0, (self.x - adm) / np.maximum(w, 1) - null, np.nan)})[w > 0]
        b = self.between()
        Ff = np.einsum("ab,ce->acbe", far, far).reshape(k1 * k1, k1 * k1)
        adm_b = n * (n - 1) * float(np.sum(Ff * b["oo"]))
        val = (b["x"] - adm_b) / b["w"] - null if b["w"] > 0 else float("nan")
        return tab, dict(null=null, n_snps=int(self.n_snps), cell_cm=self.cell_cm,
                         between=dict(x=b["x"], w=b["w"], adm=adm_b, n_snps=b["n_snps"], value=val))


def _relatives_pass(blocks, t_idx, chrom, cm, C, df_res, step, pa, pb, cell_cm, n_controls, memory_gb, log,
                    threads=1):
    """A genotype pass of its own for the relatives' profile (repeated per chunk of pairs if their cell sums exceed
    memory_gb): the adjusted, standardized genotypes of each block exactly as in the curve's pass (mean-imputed,
    centred, projected on the covariates, divided by the residual SD with df_res degrees of freedom)."""
    N = blocks.n
    if n_controls is None:                                     # as many controls as listed pairs, 20,000 to 200,000
        n_controls = int(min(max(pa.size, 20000), 200000))
    ca, cb = _control_pairs(N, pa, pb, n_controls)
    prof = _RelativesProfile(pa, pb, ca, cb, cell_cm, memory_gb, threads)
    p_rank = C.shape[1]
    t0 = time.perf_counter()

    def reader(cols):
        def read(sl, mem):
            X = blocks.read(cols[sl])
            mu = np.nanmean(X, axis=0)
            center_impute(X)
            ss = np.einsum("ij,ij->j", X, X)
            Bx = C.T @ X if p_rank else np.zeros((0, X.shape[1]))
            var_res = (ss - np.einsum("ij,ij->j", Bx, Bx)) / df_res
            ok = np.isfinite(var_res) & (var_res > 1e-8) & np.isfinite(mu)
            Xu = X[mem]
            if p_rank:
                Xu = Xu - C[mem] @ Bx
            return Xu / np.sqrt(np.where(ok, var_res, 1.0))[None, :], ok
        return read

    for c in pd.unique(chrom):
        ix = np.flatnonzero(chrom == c)
        ix = ix[np.argsort(cm[ix], kind="stable")]
        x = cm[ix]
        cell = ((x - x.min()) / cell_cm).astype(np.int64)
        nc = int(cell.max()) + 1
        bl = [np.arange(s, min(ix.size, s + step)) for s in range(0, ix.size, step)]
        prof.chromosome(c, reader(t_idx[ix]), bl, cell, nc, x)
        log(f"[admixed-ld] curve relatives chr{c}: {pa.size} listed and {ca.size} control pairs, {nc} cells of "
            f"{cell_cm} cM, {len(prof.chunks(nc))} pass(es) ({time.perf_counter() - t0:.0f} s)")
    if prof.threads > 1:
        prof.pool.shutdown()
    return prof


def _phi_at(tab, ds):
    t = tab[np.isfinite(tab.phi)]
    return [(d, float(np.interp(d, t.d_cm, t.phi))) for d in ds if d <= t.d_cm.max()]


def genotype_curve(geno, ref_path, out, *, sources=None, covar=None, keep=None, match="position", cell_cm=0.25,
                   max_cm=None, theta_min_delta=0.2, relative_pairs=None, kinship_col=None, kinship_min=None,
                   closure=False, cells_cm=2.0, per_chromosome=False, step=4096, threads=1, native=True,
                   noise_correction=True, relatedness_profile=True, relatedness_cell_cm=0.5, relatedness_controls=None,
                   relatedness_memory_gb=4.0, background=False, background_quantile=0.2, background_between_snps=None,
                   log=print):
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
        relatedness_profile = bool(relatedness_profile) and pa.size > 0
        rel = None if relatedness_profile else _PairRelatedness(pa, pb)
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
    chr_parts = []                                             # per-chromosome lag sums (per_chromosome)
    th_num = np.zeros((N, k1))
    th_den = np.zeros((k1, k1))
    th_c = np.zeros(k1)
    p_rank = C.shape[1]
    bg = None
    if background:                                             # all pairs, least differentiated SNPs (aggregates)
        amax = np.abs(D).max(axis=1)
        sel_bg = amax <= np.quantile(amax, float(background_quantile))
        nb_max = int(background_between_snps) if background_between_snps is not None else \
            int(min(40000, relatedness_memory_gb * 2 ** 30 // (4 * N)))
        bg = _Background(sel_bg, N, df_res, p_rank, relatedness_cell_cm, k1, nb_max)
        log(f"[admixed-ld] curve: background relatedness LD from all pairs of people, on the {int(sel_bg.sum())} SNPs "
            f"with the smallest source contrasts (|D| <= {np.quantile(amax, float(background_quantile)):.4f}); "
            f"between chromosomes on {int(bg.between_mask.sum())} of them")
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
            if bg is not None:
                Ofl = ((d[:, :, None] * d[:, None, :] - Sn) * inv[:, None, None]).reshape(sl.size, k1 * k1)
                bg.add(sl, X, C, Bx, np.where(ok, var_res, 1.0), ok, Ofl, cm[sl], c)
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
        if bg is not None:
            bg.end_chromosome(cm[ix].min())
        U = Y - C @ B if p_rank else Y
        S = mp.tn(U, U) / df_res
        num_c, den_c = np.zeros_like(num), np.zeros_like(den)
        for lag in range(1, min(L, nc - 1) + 1):
            for a in range(k1):
                for b in range(k1):
                    num_c[lag, a, b] = float(np.trace(S[a * nc:(a + 1) * nc, b * nc:(b + 1) * nc], offset=lag))
            den_c[lag] = np.einsum("iau,ibv->abuv", q[:-lag], q[lag:])
        num += num_c
        den += den_c
        if per_chromosome:
            chr_parts.append((str(c), num_c, den_c))
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
    if bg is not None:
        kc = kern.copy()
        fin = np.isfinite(kc).all(axis=(1, 2))

        def kernel_at(dd):
            out_ = np.zeros((np.size(dd), k1, k1))
            for a_ in range(k1):
                for b_ in range(k1):
                    out_[:, a_, b_] = np.interp(np.asarray(dd), dist[fin], kc[fin, a_, b_])
            return out_
        btab, binfo = bg.finish(kernel_at, inter)
        btab.to_csv(f"{out}.background.tsv", sep="\t", index=False, float_format="%.10g")
        info["background"] = dict(**binfo, file=f"{out}.background.tsv", quantile=float(background_quantile),
                                  definition="excess = (x - adm) / w - null: the cross-person LD of all ordered pairs of "
                                             "people per pair of the least differentiated SNPs at the lag's mean "
                                             "distance d_cm, less the admixture LD of the curve's kernel and SUMMIT's "
                                             "null p_tot nu / n; below a few cM it also holds within-source LD")
        bb = binfo["between"]
        log(f"[admixed-ld] curve: background (all pairs) between chromosomes {bb['value']:.4g} per SNP pair "
            f"({bb['n_snps']} SNPs; null {binfo['null']:.4f}); within chromosomes "
            + ", ".join(f"{dd:.0f} cM {vv:.4g}" for dd, vv in _phi_at(btab.rename(columns={"excess": "phi"}),
                                                                   (3.0, 5.0, 10.0, 20.0, 40.0)))
            + f"; wrote {out}.background.tsv")
    if relative_pairs is not None and relatedness_profile:
        prof = _relatives_pass(blocks, t_idx, chrom, cm, C, df_res, step, pa, pb, relatedness_cell_cm,
                               relatedness_controls, relatedness_memory_gb, log, threads=threads)
        summ = prof.summary()
        tab = prof.table()
        tab.to_csv(f"{out}.relatedness.tsv", sep="\t", index=False, float_format="%.10g")
        np.savez(f"{out}.relatedness_chr.npz", chrom=np.array([x["name"] for x in prof.chr]),
                 **{k: np.stack([np.pad(x[k], (0, max(y["w"].size for y in prof.chr) - x[k].size)) for x in prof.chr])
                    for k in ("listed", "controls", "w", "dsum")})
        summ["profile"] = dict(file=f"{out}.relatedness.tsv", per_chromosome=f"{out}.relatedness_chr.npz",
                               cell_cm=float(relatedness_cell_cm), n_controls=int(prof.n_controls),
                               definition="phi(L) = [P(L) - (n_listed / n_controls) Pc(L)] / W(L): the relatives' "
                                          "excess of sum_{ordered pairs} u_aj u_bj u_ak u_bk per SNP pair at the lag's "
                                          "mean distance d_cm, net of what random (control) pairs contribute")
        rel = prof
        info["relatedness"] = dict(**summ, kinship_col=kinship_col, kinship_min=kinship_min, closure=closure,
                                   definition="relatedness of the covariate-adjusted, standardized genotypes")
        b = summ["between"]
        log(f"[admixed-ld] curve: relatives between chromosomes B = {b['value']:.4g} per SNP pair (controls "
            f"{b['n_controls']}); within chromosomes phi at "
            + ", ".join(f"{d:.0f} cM {v:.4g}" for d, v in _phi_at(tab, (1.0, 5.0, 20.0, 50.0)))
            + f"; wrote {out}.relatedness.tsv")
    elif rel is not None:
        info["relatedness"] = dict(**rel.summary(), kinship_col=kinship_col, kinship_min=kinship_min, closure=closure,
                                   definition="relatedness of the covariate-adjusted, standardized genotypes")
    if "relatedness" in info:
        r = info["relatedness"]
        log(f"[admixed-ld] curve: {r['pairs']} relative pairs, mean relatedness after the covariates {r['mean']:.4f} "
            f"(largest |g| {r['max_abs']:.3f}); sum over ordered pairs of g^2 {r['sum_squared_ordered']:.4g}")
    if cells_cm and coarse_U:
        # two random halves of the people give independent estimates of each cell-pair covariance, whose product is
        # unbiased for its square (and, with K sources, for the products of kernel entries)
        Uall = np.concatenate(coarse_U, axis=2)                 # N x k1 x cells
        nC = Uall.shape[2]
        rng_h = np.random.default_rng(20261104)
        if relative_pairs is not None:                         # relatives in one half, so the halves stay independent
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
    if per_chromosome:
        # each chromosome's lag sums, for a leave-one-chromosome-out jackknife of the curve (aggregates only)
        np.savez(f"{out}.curve_chr.npz", chrom=np.array([x[0] for x in chr_parts]), cell_cm=float(cell_cm),
                 num=np.stack([x[1] for x in chr_parts]), den=np.stack([x[2] for x in chr_parts]))
        info["per_chromosome"] = f"{out}.curve_chr.npz"
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
