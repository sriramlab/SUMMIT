"""Genotype blocks, variant tables, genetic maps and variant matching for `summit admixed-ld`.

Genotypes are decoded with SUMMIT's existing readers: bed_reader for PLINK 1 BED (counting A1, as SUMMIT's Python
readers do) and pgenlib through `PgenBlockReader` for PLINK 2 PGEN (counting REF, which `read_pvar_variants` exposes as
A2). Each table therefore carries a COUNTED and an OTHER allele, and every orientation decision compares them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..ldscore.genotype_source import (PgenBlockReader, read_fam_sample_ids, read_psam_sample_ids,
                                       read_pvar_variants, resolve_genotype_input, validate_variant_metadata)

CM_PER_MB_FALLBACK = 1.2          # sex-averaged autosomal recombination rate, used only without any genetic map
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


# ----------------------------------------------------------------------------------------------------- genotypes
def read_bim(path: str) -> pd.DataFrame:
    bim = pd.read_csv(path, sep=r"\s+", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
                      dtype={"CHR": str, "SNP": str, "A1": str, "A2": str})
    return validate_variant_metadata(bim, source=f"BIM '{path}'")


class DosageBlocks:
    """Allele dosages of chosen variants for a fixed set of samples, as float64 (samples x variants, NaN if missing).

    `variants` has CHR, SNP, CM, BP, A1, A2 and COUNTED, OTHER (the allele whose dosage is returned, and the other)."""

    def __init__(self, path: str, *, rows=None, threads: int = 1, step: int = 4096):
        gi = resolve_genotype_input(path)
        self.format, self.prefix, self.threads, self.step = gi.format, gi.prefix, max(1, int(threads)), int(step)
        if gi.format == "bed":
            self.samples = read_fam_sample_ids(gi.sample_path)
            v = read_bim(gi.variant_path)
            v["COUNTED"], v["OTHER"] = v["A1"].str.upper(), v["A2"].str.upper()
        else:
            self.samples = read_psam_sample_ids(gi.sample_path)
            v = validate_variant_metadata(read_pvar_variants(gi.variant_path), source=f"PVAR '{gi.variant_path}'")
            v["COUNTED"], v["OTHER"] = v["A2"].str.upper(), v["A1"].str.upper()
        self.variants = v.reset_index(drop=True)
        n_all = len(self.samples)
        if rows is None:
            self.rows = None
        else:
            r = np.asarray(rows, dtype=np.int64)
            if r.ndim != 1 or r.size == 0 or np.any(np.diff(r) <= 0) or r[0] < 0 or r[-1] >= n_all:
                raise ValueError("Sample rows must be strictly increasing indices into the genotype samples.")
            self.rows = r
        self.n = n_all if self.rows is None else int(self.rows.size)
        if gi.format == "bed":
            from bed_reader import open_bed
            self._bed = open_bed(gi.genotype_path, count_A1=True, iid_count=n_all, sid_count=len(v),
                                 num_threads=self.threads)
            self._pgen = None
        else:
            self._bed = None
            self._pgen = PgenBlockReader(gi.genotype_path, raw_sample_ct=n_all, variant_ct=len(v),
                                         sample_subset=self.rows, step_size=self.step, dtype=np.float64, ddof=1)

    def read(self, cols) -> np.ndarray:
        """Dosages of the variant columns `cols` (any order), samples x len(cols), Fortran order."""
        cols = np.asarray(cols, dtype=np.int64)
        if self._bed is not None:
            rows = slice(None) if self.rows is None else self.rows
            return self._bed.read(index=np.s_[rows, cols], dtype="float64", order="F", num_threads=self.threads)
        out = np.empty((self.n, cols.size), dtype=np.float64, order="F")
        order = np.argsort(cols, kind="stable")
        sc = cols[order]
        s = 0
        while s < sc.size:                     # contiguous ranges of at most `step` variants
            e = int(np.searchsorted(sc, sc[s] + self.step, side="left"))
            lo, hi = int(sc[s]), int(sc[e - 1]) + 1
            block = self._pgen.read_dosage_block(lo, hi)
            out[:, order[s:e]] = block[:, sc[s:e] - lo]
            s = e
        out[out == -9.0] = np.nan
        return out

    def close(self):
        self._bed = None                       # bed_reader closes its file when the object is released
        if self._pgen is not None:
            self._pgen.close()
            self._pgen = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def center_impute(X: np.ndarray) -> np.ndarray:
    """Mean-impute missing calls and center every column, in place (as SUMMIT's relatives code does)."""
    from ..relatives.moments import _center_impute
    return _center_impute(X)


# ----------------------------------------------------------------------------------------------------- genetic map
def _read_map_table(path: str) -> pd.DataFrame:
    """One genetic-map file as CHR (or None), BP, CM. Accepts the HapMap/1000 Genomes and SHAPEIT/Eagle layouts
    (with a header naming position and map columns) or headerless 3 (position, rate, cM) or 4 (chr, position, rate,
    cM) column files."""
    first = pd.read_csv(path, sep=r"\s+", header=None, nrows=1, dtype=str).iloc[0].tolist()
    has_header = any(pd.to_numeric(pd.Series(first), errors="coerce").isna())
    if has_header:
        df = pd.read_csv(path, sep=r"\s+")
        low = {c: str(c).lower() for c in df.columns}
        pos = [c for c, l in low.items() if ("pos" in l or l in ("bp", "physical_position")) and "rate" not in l]
        cm = [c for c, l in low.items() if ("cm" in l and "rate" not in l and "/" not in l) or "map" in l]
        cm = [c for c in cm if c not in pos]
        chrom = [c for c, l in low.items() if l.lstrip("#") in ("chr", "chrom", "chromosome")]
        if len(pos) != 1 or len(cm) != 1:
            raise ValueError(f"Cannot identify the position and cM columns of genetic map {path}: {list(df.columns)}")
        out = pd.DataFrame({"BP": pd.to_numeric(df[pos[0]], errors="coerce"),
                            "CM": pd.to_numeric(df[cm[0]], errors="coerce")})
        out["CHR"] = df[chrom[0]].astype(str).str.replace(r"^(?i:chr)", "", regex=True) if chrom else None
    else:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[1] == 3:
            out = pd.DataFrame({"CHR": None, "BP": df[0], "CM": df[2]})
        elif df.shape[1] == 4:
            out = pd.DataFrame({"CHR": df[0].astype(str).str.replace(r"^(?i:chr)", "", regex=True),
                                "BP": df[1], "CM": df[3]})
        else:
            raise ValueError(f"Genetic map {path} without a header must have 3 or 4 columns.")
    out = out.dropna(subset=["BP", "CM"])
    return out


def genetic_positions(variants: pd.DataFrame, genetic_map: str | None = None, log=print) -> tuple[np.ndarray, str]:
    """cM position of every variant (CHR, BP, optionally CM), non-decreasing within each chromosome.

    `genetic_map` is one file with a chromosome column, or a path containing {chr} with one file per chromosome.
    Without a map, a CM column that is positive for most variants is used; otherwise BP x 1.2 cM/Mb, with a warning."""
    chrom = variants["CHR"].astype(str).to_numpy()
    bp = variants["BP"].to_numpy(np.float64)
    cm = np.full(len(variants), np.nan)
    if genetic_map:
        if "{chr}" in genetic_map:
            tables = {}
            for c in pd.unique(chrom):
                f = genetic_map.replace("{chr}", str(c))
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Genetic map for chromosome {c} not found: {f}")
                t = _read_map_table(f)
                tables[c] = t if t["CHR"].isna().all() else t[t["CHR"] == str(c)]
        else:
            t = _read_map_table(genetic_map)
            if t["CHR"].isna().all():
                raise ValueError("A single genetic-map file needs a chromosome column; otherwise use a {chr} pattern.")
            tables = {c: t[t["CHR"] == str(c)] for c in pd.unique(chrom)}
        for c, t in tables.items():
            t = t.sort_values("BP")
            mp, mc = t["BP"].to_numpy(np.float64), np.maximum.accumulate(t["CM"].to_numpy(np.float64))
            if mp.size < 2:
                raise ValueError(f"Genetic map for chromosome {c} has fewer than two positions.")
            rate = (mc[-1] - mc[0]) / max(mp[-1] - mp[0], 1.0)            # cM per bp, for positions off the map
            ix = chrom == c
            x = bp[ix]
            y = np.interp(x, mp, mc)
            y = np.where(x < mp[0], mc[0] - (mp[0] - x) * rate, y)
            y = np.where(x > mp[-1], mc[-1] + (x - mp[-1]) * rate, y)
            cm[ix] = y
        source = f"genetic map {genetic_map}"
    elif "CM" in variants.columns and np.mean(variants["CM"].to_numpy(np.float64) > 0) > 0.9:
        cm = variants["CM"].to_numpy(np.float64).copy()
        source = "CM column of the variant table"
    else:
        cm = bp / 1e6 * CM_PER_MB_FALLBACK
        source = f"no genetic map: {CM_PER_MB_FALLBACK} cM per Mb"
        log(f"[admixed-ld] WARNING: {source}. Distances along the genome are approximate; pass --genetic-map.")
    for c in pd.unique(chrom):               # non-decreasing in genome order (map noise, unsorted input)
        ix = np.flatnonzero(chrom == c)
        o = ix[np.argsort(bp[ix], kind="stable")]
        cm[o] = np.maximum.accumulate(cm[o])
    return cm, source


# ----------------------------------------------------------------------------------------------------- matching
def _ambiguous(a, b):
    pair = np.char.add(a.astype(str), b.astype(str))
    return np.isin(pair, ["AT", "TA", "CG", "GC"])


def _complement(a):
    s = pd.Series(a.astype(str))
    return s.map(lambda x: _COMPLEMENT.get(x, "?") if len(x) == 1 else "?").to_numpy()


def match_variants(target: pd.DataFrame, ref: pd.DataFrame, how: str = "position"):
    """Indices of variants found in both tables (CHR, BP, SNP, COUNTED, OTHER) and whether the target's counted allele
    is the reference's other allele (flip). `how` is 'position' (CHR:BP plus alleles) or 'id' (SNP plus alleles).
    Keys that occur twice in either table are not used. Strand flips are resolved for non-ambiguous SNPs only; A/T and
    C/G SNPs are matched as listed. Returns (target_idx, ref_idx, flip) sorted by target_idx."""
    if how == "position":
        tk = target["CHR"].astype(str) + ":" + target["BP"].astype(np.int64).astype(str)
        rk = ref["CHR"].astype(str) + ":" + ref["BP"].astype(np.int64).astype(str)
    elif how == "id":
        tk, rk = target["SNP"].astype(str), ref["SNP"].astype(str)
    else:
        raise ValueError("how must be 'position' or 'id'.")
    tk, rk = pd.Series(tk.to_numpy()), pd.Series(rk.to_numpy())
    tk = tk[~tk.duplicated(keep=False)]
    rk = rk[~rk.duplicated(keep=False)]
    lookup = pd.Series(rk.index.to_numpy(), index=rk.to_numpy())
    r = lookup.reindex(tk.to_numpy()).to_numpy()
    have = ~pd.isna(r)
    t_idx = tk.index.to_numpy()[have].astype(np.int64)
    r_idx = r[have].astype(np.int64)
    tc, to = target["COUNTED"].to_numpy(str)[t_idx], target["OTHER"].to_numpy(str)[t_idx]
    rc, ro = ref["COUNTED"].to_numpy(str)[r_idx], ref["OTHER"].to_numpy(str)[r_idx]
    same = (tc == rc) & (to == ro)
    swap = (tc == ro) & (to == rc)
    amb = _ambiguous(tc, to)
    ctc, cto = _complement(tc), _complement(to)
    same_c = ~amb & (ctc == rc) & (cto == ro)
    swap_c = ~amb & (ctc == ro) & (cto == rc)
    ok = same | swap | same_c | swap_c
    flip = (swap | swap_c) & ~same
    order = np.argsort(t_idx[ok], kind="stable")
    return t_idx[ok][order], r_idx[ok][order], flip[ok][order]
