"""The reference resource of `summit admixed-ld`: banded dosage covariances of source panels.

For every source panel (unadmixed samples of one source population) and every chromosome the resource holds, on the
variants shared by all panels (oriented to the first panel's counted allele):
  p_<s>, v_<s>      allele frequency of the counted allele and dosage variance (n - 1 denominator)
  cov_<s>           dosage covariance of each pair (j, k), k > j, within the window (upper band, CSR by j)
  m22_<s>           mean of x_j^2 x_k^2 over samples (centred dosages); with cov and v it gives the unbiased estimate of
                    the squared covariance (compose.unbiased_squared_covariance)
  indptr, indices   the band structure (indices are positions within the chromosome)
  snp, bp, cm, counted, other
A manifest.json records the sources, sample sizes, window, genetic map, and format version. Files are uncompressed
`.npz`, so arrays load on demand.
"""
from __future__ import annotations

import datetime
import json
import os
import time

import numpy as np
import pandas as pd

from ._io import DosageBlocks, center_impute, genetic_positions, match_variants

FORMAT = "summit-admixed-ld-reference"
VERSION = 1


def _products(threads, native=True):
    from ..ldscore.matrix_products import MatrixProducts
    try:
        return MatrixProducts(native=native, threads=threads)
    except Exception:                         # no native module in this build: numpy products
        return MatrixProducts(native=False, threads=threads)


def _band_structure(bp: np.ndarray, window_bp: int):
    """CSR band of pairs (j, k), k > j, with bp_k - bp_j <= window (bp sorted)."""
    hi = np.searchsorted(bp, bp + int(window_bp), side="right")
    cnt = hi - np.arange(bp.size) - 1
    indptr = np.zeros(bp.size + 1, dtype=np.int64)
    np.cumsum(cnt, out=indptr[1:])
    nnz = int(indptr[-1])
    # indices: for each j, j+1 .. hi_j - 1
    rows = np.repeat(np.arange(bp.size, dtype=np.int64), cnt)
    indices = (np.arange(nnz, dtype=np.int64) - np.repeat(indptr[:-1], cnt) + rows + 1).astype(np.int32)
    return hi, indptr, indices, rows


class _SlidingPanel:
    """Centred, imputed dosages of a contiguous range of a chromosome's variants, reusing the overlap with the previous
    range (the band's partners) instead of decoding it again."""

    def __init__(self, blocks: DosageBlocks, cols: np.ndarray, flip: np.ndarray):
        self.blocks, self.cols, self.sign = blocks, cols, np.where(flip, -1.0, 1.0)
        self.lo = self.hi = 0
        self.X = np.empty((blocks.n, 0), order="F")
        self.mean = np.empty(0)

    def get(self, lo: int, hi: int):
        if not (self.lo <= lo and lo <= self.hi):
            self.lo = self.hi = lo
            self.X = np.empty((self.blocks.n, 0), order="F")
            self.mean = np.empty(0)
        keep = slice(lo - self.lo, self.hi - self.lo)
        parts, means = [self.X[:, keep]], [self.mean[keep]]
        if hi > self.hi:
            raw = self.blocks.read(self.cols[self.hi:hi])
            mu = np.nanmean(raw, axis=0)
            center_impute(raw)
            raw *= self.sign[None, self.hi:hi]
            parts.append(raw)
            means.append(mu)
        self.X = np.asfortranarray(np.concatenate(parts, axis=1)) if len(parts) > 1 else parts[0]
        self.mean = np.concatenate(means)
        self.lo, self.hi = lo, max(hi, self.hi)
        return self.X[:, : hi - lo], self.mean[: hi - lo]


def build_reference(panels, names, out, *, window_kb=3000, genetic_map=None, match="position", chromosomes=None,
                    step=2048, threads=1, dtype="float32", native=True, log=print):
    """Build the resource from source panels (PLINK 1 BED or PLINK 2 PGEN prefixes), one per source."""
    if len(panels) != len(names) or len(panels) < 2:
        raise ValueError("Give at least two source panels, each with a name.")
    if len(set(names)) != len(names):
        raise ValueError("Source names must be distinct.")
    blocks = [DosageBlocks(p, threads=threads, step=max(step, 1)) for p in panels]
    small = [n for n, b in zip(names, blocks) if b.n < 10]
    if small:
        raise ValueError(f"Source panels need at least 10 people: {', '.join(small)}.")
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    base = blocks[0].variants
    keep = np.ones(len(base), dtype=bool)
    cols = [np.arange(len(base))]
    flips = [np.zeros(len(base), dtype=bool)]
    for b in blocks[1:]:
        t_idx, r_idx, flip = match_variants(base, b.variants, how=match)
        col = np.full(len(base), -1, dtype=np.int64)
        fl = np.zeros(len(base), dtype=bool)
        col[t_idx], fl[t_idx] = r_idx, flip
        keep &= col >= 0
        cols.append(col)
        flips.append(fl)
    if chromosomes is None:
        chromosomes = [str(c) for c in range(1, 23)]
    chromosomes = [str(c) for c in chromosomes]
    keep &= base["CHR"].isin(chromosomes).to_numpy()
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("No variants are shared by all source panels on the requested chromosomes.")
    shared = base.iloc[idx].reset_index(drop=True)
    cm, map_source = genetic_positions(shared, genetic_map, log=log)
    log(f"[admixed-ld] reference: {idx.size} variants shared by {len(panels)} panels "
        f"({', '.join(f'{n} {b.n}' for n, b in zip(names, blocks))}); positions from {map_source}")
    mp = _products(threads, native)
    out_dtype = np.dtype(dtype)
    stats = {}
    for c in chromosomes:
        ci = np.flatnonzero(shared["CHR"].to_numpy() == c)
        if ci.size < 2:
            continue
        ci = ci[np.argsort(shared["BP"].to_numpy()[ci], kind="stable")]
        bp = shared["BP"].to_numpy(np.int64)[ci]
        hi, indptr, indices, rows = _band_structure(bp, int(window_kb) * 1000)
        nnz = int(indptr[-1])
        arrays = dict(snp=shared["SNP"].to_numpy(str)[ci], bp=bp, cm=cm[ci], counted=shared["COUNTED"].to_numpy(str)[ci],
                      other=shared["OTHER"].to_numpy(str)[ci], indptr=indptr, indices=indices)
        for s, (b, col, fl) in enumerate(zip(blocks, cols, flips)):
            panel = _SlidingPanel(b, col[idx[ci]], fl[idx[ci]])
            p = np.empty(ci.size)
            v = np.empty(ci.size)
            cov = np.empty(nnz, dtype=out_dtype)
            m22 = np.empty(nnz, dtype=out_dtype)
            n = b.n
            for b0 in range(0, ci.size, step):
                b1 = min(ci.size, b0 + step)
                e = int(hi[b1 - 1])
                X, mu = panel.get(b0, e)
                nb = b1 - b0
                p_raw = mu[:nb] / 2.0
                p[b0:b1] = np.where(fl[idx[ci[b0:b1]]], 1.0 - p_raw, p_raw)
                v[b0:b1] = np.einsum("ij,ij->j", X[:, :nb], X[:, :nb]) / (n - 1)
                s0, s1 = int(indptr[b0]), int(indptr[b1])
                if s1 == s0:
                    continue
                Xb = X[:, :nb]
                C = mp.tn(Xb, X) / (n - 1)
                X2 = X * X
                Q = mp.tn(X2[:, :nb], X2) / n
                r = rows[s0:s1] - b0
                k = indices[s0:s1].astype(np.int64) - b0
                cov[s0:s1] = C[r, k]
                m22[s0:s1] = Q[r, k]
            arrays[f"p_{s}"], arrays[f"v_{s}"], arrays[f"cov_{s}"], arrays[f"m22_{s}"] = p, v, cov, m22
        np.savez(os.path.join(out, f"chr{c}.npz"), **arrays)
        stats[c] = dict(n_snps=int(ci.size), n_pairs=nnz, cm_length=float(cm[ci].max() - cm[ci].min()))
        log(f"[admixed-ld] reference chr{c}: {ci.size} variants, {nnz} pairs within {window_kb} kb "
            f"({time.perf_counter() - t0:.0f} s)")
    info = mp.drain()
    manifest = write_manifest(out, [dict(name=nm, n=int(b.n), panel=os.path.basename(p))
                                    for nm, b, p in zip(names, blocks, panels)],
                              window_kb=window_kb, genetic_map=map_source, match=match, dtype=out_dtype.name,
                              counted_allele="first panel's counted allele (BED A1, PGEN REF)", stats=stats,
                              products=dict(native=info["native"], repaired_columns=int(info["repaired_columns"])))
    for b in blocks:
        b.close()
    log(f"[admixed-ld] reference written to {out} ({time.perf_counter() - t0:.0f} s)")
    return manifest


def write_manifest(out, sources, *, window_kb, genetic_map, match, dtype, counted_allele, stats, products):
    """The manifest of a reference directory (sources: name, n and, for a derived source, how it was made)."""
    manifest = dict(format=FORMAT, version=VERSION, created=datetime.datetime.now().isoformat(timespec="seconds"),
                    sources=sources, window_kb=int(window_kb), genetic_map=genetic_map, match=match, dtype=dtype,
                    counted_allele=counted_allele, chromosomes=stats,
                    genome_cm=float(sum(v["cm_length"] for v in stats.values())), products=products)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


class Reference:
    """Read access to a resource written by `build_reference`."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "manifest.json")) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != FORMAT or int(self.manifest.get("version", 0)) != VERSION:
            raise ValueError(f"{path} is not a {FORMAT} v{VERSION} resource.")
        self.names = [s["name"] for s in self.manifest["sources"]]
        self.n = {s["name"]: int(s["n"]) for s in self.manifest["sources"]}
        # sample size behind the frequencies, when it differs from the panel's (e.g. frequencies derived from a cohort)
        self.n_freq = {s["name"]: float(s.get("n_freq", s["n"])) for s in self.manifest["sources"]}
        # a derived source's frequency error contains the known panels' errors with these coefficients
        self.noise_mix = {s["name"]: dict(s.get("noise_mix", {})) for s in self.manifest["sources"]}
        self.chromosomes = list(self.manifest["chromosomes"])
        self.genome_cm = float(self.manifest["genome_cm"])

    def frequency_noise(self, sources, V, r_idx, rv=None):
        """Inputs of model.contrast_noise for these sources: V (M x T: the sources' dosage variances, then those of
        panels that enter only through a derived source), n (T) and the mixing matrix G (K x T). V and r_idx: the
        selected SNPs' dosage variances of `sources` and their rows in variants(); rv: that variants() table, whose
        fvar_<s> columns (sources derived from a cohort) replace V / 4n by each SNP's own frequency variance."""
        sources = list(sources)
        extra = [t for s in sources for t in self.noise_mix[s] if t not in sources]
        extra = list(dict.fromkeys(extra))
        cols = sources + extra
        G = np.zeros((len(sources), len(cols)))
        for i, s in enumerate(sources):
            G[i, i] = 1.0
            for t, c in self.noise_mix[s].items():
                G[i, cols.index(t)] += float(c)
        n = np.array([self.n_freq[t] for t in cols], np.float64)
        V = np.array(V, np.float64, copy=True)
        if extra:
            Vx = self.variants(extra)
            V = np.column_stack([V] + [Vx[f"v_{t}"].to_numpy()[r_idx] for t in extra])
        if rv is not None:
            for i, s in enumerate(sources):
                if f"fvar_{s}" in rv:
                    fv = rv[f"fvar_{s}"].to_numpy()[r_idx]
                    V[:, i] = np.where(np.isfinite(fv), 4.0 * n[i] * fv, V[:, i])
        return V, n, G, cols

    def covariance_noise_mix(self, name: str):
        """For a source whose window covariances were derived by inverting a cohort's composition (`derive`, LD by
        inversion): C^N = [C_res - sum_t pi_t(d) C^t - adm] / pi_N(d) with the anchor cohort's pi_t(d) =
        m_t exp(-lam_t d), so its sampling error carries -pi_t(d) / pi_N(d) times the error of each known panel t, and
        Cov(e_N, e_t) = -pi_t(d) Var(e_t) / pi_N(d). Returns ((m_N, lam_N), {t: (m_t, lam_t)}) of the anchor cohort, or
        None for a panel or a source whose covariances are independent of the known panels' (anchor-panel LD, masked
        admixed haplotypes). References written before the mix was recorded explicitly (no `covariance_noise` key)
        are read from m_anchor and decay_anchor, which derive aligns with the reference's sources."""
        src = self.manifest["sources"][self.source_index(name)]
        if "covvar without the doubly counted panel noise" in str(src.get("how", "")):
            raise ValueError(f"The covvar of {name} in {self.path} was already reduced by the research fix "
                             "(fix_derived_covvar.py); compose now removes the known panels' noise once by itself. "
                             "Use the reference as derive wrote it.")
        cn = src.get("covariance_noise")
        if cn is not None:
            own = tuple(float(x) for x in cn["pi"])
            return own, {t: tuple(float(x) for x in v) for t, v in cn["mix"].items()}
        if not src.get("derived") or src.get("ld", "inversion") != "inversion" or "m_anchor" not in src:
            return None
        # inversion stores each covariance with its sampling variance (covvar); a reference that kept the manifest
        # entry but took the source's covariances from a panel (m22, e.g. derived frequencies with a small panel's LD)
        # carries no panel errors
        i = self.source_index(name)
        if f"covvar_{i}" not in self.load(self.chromosomes[0]).files:
            return None
        names = [x["name"] for x in self.manifest["sources"]]
        m, lam = src["m_anchor"], src["decay_anchor"]
        if len(m) != len(names) or len(lam) != len(names) or names[-1] != name:
            raise ValueError(f"The anchor parameters of the derived source {name} do not match the reference's sources.")
        return (float(m[-1]), float(lam[-1])), {t: (float(m[i]), float(lam[i])) for i, t in enumerate(names[:-1])}

    def source_index(self, name: str) -> int:
        if name not in self.names:
            raise ValueError(f"Source {name!r} is not in the reference ({', '.join(self.names)}).")
        return self.names.index(name)

    def load(self, c: str):
        return np.load(os.path.join(self.path, f"chr{c}.npz"), allow_pickle=False)

    def variants(self, sources) -> pd.DataFrame:
        """CHR, SNP, BP, CM, COUNTED, OTHER, and p/v of the chosen sources, for every chromosome (genome order)."""
        frames = []
        for c in self.chromosomes:
            z = self.load(c)
            d = dict(CHR=c, SNP=z["snp"], BP=z["bp"], CM=z["cm"], COUNTED=z["counted"], OTHER=z["other"])
            for nm in sources:
                s = self.source_index(nm)
                d[f"p_{nm}"], d[f"v_{nm}"] = z[f"p_{s}"], z[f"v_{s}"]
                if f"fvar_{s}" in z.files:                     # a derived source's per-SNP frequency variance
                    d[f"fvar_{nm}"] = z[f"fvar_{s}"]
            df = pd.DataFrame(d)
            df["LOCAL"] = np.arange(len(df))
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
