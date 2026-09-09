"""Shared raw-genotype ownership; no prediction-specific BED decoder."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np

from ._validation import array_digest, digest, indices, positive_int
from .spec import GenotypeScale, VariantAxis, sample_ids


def native_module():
    try:
        from summit import gxeldcore
    except ImportError as exc:
        raise ImportError("Build SUMMIT with prediction native support before using the native backend") from exc
    if not hasattr(gxeldcore, "PredictionBEDReader"):
        raise ImportError("Installed gxeldcore predates prediction support; rebuild this branch")
    return gxeldcore


@dataclass
class PassLedger:
    traversals: dict = field(default_factory=dict)
    source_blocks: int = 0
    source_variants: int = 0
    source_decoded_bytes: int = 0
    cache_blocks: int = 0
    cache_variants: int = 0
    operator_calls: int = 0
    active_rhs: list = field(default_factory=list)

    def begin(self, phase):
        self.traversals[phase] = self.traversals.get(phase, 0) + 1


class ArrayGenotypeSource:
    """Explicit dosage source for simulations and small in-memory applications.

    Missingness is NaN or -127, never an inferred value such as dosage zero.
    Float dosage values retain their source precision. Hard-call mode validates
    rather than rounds; it permits the same exact compact storage as BED.
    """
    def __init__(self, values, samples, variants: VariantAxis, *, hard_calls=False):
        raw = np.asarray(values)
        self.samples = sample_ids(samples)
        self.variants = variants
        self.hard_calls = bool(hard_calls)
        if raw.shape != (len(self.samples), len(variants.ids)) or raw.dtype.kind not in "fiu":
            raise ValueError("genotypes must be a numeric sample-by-variant matrix on the supplied axes")
        missing = np.isnan(raw) | (raw == -127)
        valid = raw[~missing]
        if np.any(~np.isfinite(valid)) or np.any((valid < 0) | (valid > 2)):
            raise ValueError("dosages must lie in [0,2] or use declared missingness")
        if self.hard_calls and np.any(valid != np.rint(valid)):
            raise ValueError("fractional dosages cannot be stored as int8 hard calls")
        self.values = np.array(np.where(missing, -127, raw), dtype=np.int8 if hard_calls else np.float64, order="F")
        self.values.setflags(write=False)
        self.identity = digest(["array", array_digest(self.values), digest(self.samples), variants.identity])
        self.rows = None
        self.generation = 0

    def prepare(self, rows, block_size, threads):
        self.rows = indices(rows, name="source rows", size=len(self.samples))
        self.generation += 1

    def read(self, variants):
        if self.rows is None:
            raise RuntimeError("source is not prepared")
        variants = indices(variants, name="source variants", size=len(self.variants.ids))
        return np.asfortranarray(self.values[np.ix_(self.rows, variants)])

    def check(self):
        pass

    def close(self):
        self.rows = None
        self.generation += 1


class FileGenotypeSource:
    """BED uses the descriptor-owned native reader; PGEN reuses PgenBlockReader.

    Reader lifetime belongs to this object. A new source is inexpensive and is
    recommended for each fit/scoring operation. The caller closes it explicitly.
    """
    def __init__(self, path, *, genome_build):
        from summit.ldscore.genotype_source import (
            resolve_genotype_input, read_fam_sample_ids, read_psam_sample_ids,
            read_pvar_variants, validate_variant_metadata,
        )
        import pandas as pd
        self.input = resolve_genotype_input(os.fspath(path))
        paths = [self.input.genotype_path, self.input.variant_path, self.input.sample_path]
        self._fds = []
        self._reader = None
        self.rows = None
        self.generation = 0
        try:
            for value in paths:
                self._fds.append(os.open(value, os.O_RDONLY | os.O_CLOEXEC))
            self._paths = paths
            self._states = [self._state(os.fstat(fd)) for fd in self._fds]
            self.hard_calls = self.input.format == "bed"
            samples = read_fam_sample_ids(paths[2]) if self.hard_calls else read_psam_sample_ids(paths[2])
            self.samples = sample_ids(samples[["FID", "IID"]].itertuples(index=False, name=None))
            if self.hard_calls:
                variants = pd.read_csv(paths[1], sep=r"\s+", header=None,
                    names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
                    dtype={"CHR": str, "SNP": str, "A1": str, "A2": str}, keep_default_na=False)
                variants = validate_variant_metadata(variants, source="prediction BIM")
                counted, other = "A1", "A2"
            else:
                variants = validate_variant_metadata(read_pvar_variants(paths[1]), source="prediction PVAR")
                counted, other = "A2", "A1"  # Existing PGEN reader counts REF.
            self.variants = VariantAxis(tuple(variants.SNP), tuple(variants.CHR), tuple(int(x) for x in variants.BP),
                tuple(variants[counted]), tuple(variants[other]), genome_build)
            self.check()
            self.identity = digest([self.input.format, paths, self._states, digest(self.samples), self.variants.identity])
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _state(s):
        import stat
        if not stat.S_ISREG(s.st_mode):
            raise ValueError("genotype inputs must be regular files")
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]

    def check(self):
        if not self._fds:
            raise RuntimeError("genotype source is closed")
        current = [self._state(os.fstat(fd)) for fd in self._fds]
        paths = [self._state(os.stat(p)) for p in self._paths]
        if current != self._states or paths != self._states:
            raise RuntimeError("genotype input was modified or replaced")

    def prepare(self, rows, block_size, threads):
        self.check()
        rows = indices(rows, name="source rows", size=len(self.samples))
        if np.any(np.diff(rows) <= 0):
            raise ValueError("raw source rows must be sorted")
        positive_int(block_size, "block_size")
        positive_int(threads, "threads")
        if self._reader is not None:
            self._reader.close()
        self.rows = rows
        self.generation += 1
        self.block_size = block_size
        if self.hard_calls:
            self._reader = native_module().PredictionBEDReader(*self._fds, rows,
                threads, int(len(rows) * block_size))
        else:
            from summit.ldscore.genotype_source import PgenBlockReader
            self._reader = PgenBlockReader(self.input.genotype_path,
                raw_sample_ct=len(self.samples), variant_ct=len(self.variants.ids),
                sample_subset=rows, step_size=block_size, ddof=1, dtype=np.float64,
                standardize_threads=threads)
        self.check()

    def read(self, variants):
        self.check()
        if self._reader is None:
            raise RuntimeError("source is not prepared")
        variants = indices(variants, name="source variants", size=len(self.variants.ids))
        if len(variants) > self.block_size or np.any(np.diff(variants) <= 0):
            raise ValueError("source variant request must be sorted and within block capacity")
        out = np.empty((len(self.rows), len(variants)), dtype=np.int8 if self.hard_calls else np.float64, order="F")
        if self.hard_calls:
            self._reader.read(np.ascontiguousarray(variants), out)
        else:
            # Each contiguous source run is read once. Views expire on next read.
            breaks = np.r_[0, np.flatnonzero(np.diff(variants) != 1) + 1, len(variants)]
            for begin, end in zip(breaks[:-1], breaks[1:]):
                out[:, begin:end] = self._reader.read_dosage_block(int(variants[begin]), int(variants[end-1]) + 1)
            out[out == -9] = -127
        self.check()
        return out

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        for fd in self._fds:
            os.close(fd)
        self._fds = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ShardedGenotypeSource:
    """Ordered chromosome trios with one authenticated common sample axis."""
    def __init__(self, paths, *, genome_build):
        self.sources = []
        self.generation = 0
        self.rows = None
        self.child_generations = None
        if not paths:
            raise ValueError("empty genotype shard manifest")
        try:
            for path in paths:
                source = FileGenotypeSource(path, genome_build=genome_build)
                self.sources.append(source)
                if len(self.sources) == 1:
                    self.samples = source.samples
                elif source.samples != self.samples:
                    raise ValueError("genotype shards must have exactly the same ordered FID/IID axis")
                # Do not retain a chromosome-specific copy of every sample ID.
                source.samples = self.samples
            self.variants = VariantAxis(**{name: tuple(v for s in self.sources for v in getattr(s.variants, name))
                for name in ("ids", "chromosome", "position", "counted", "other")}, genome_build=genome_build)
            self.offsets = np.r_[0, np.cumsum([len(s.variants.ids) for s in self.sources])]
            self.hard_calls = all(s.hard_calls for s in self.sources)
            self.identity = digest(["ordered_shards_v1", [s.identity for s in self.sources]])
        except BaseException:
            self.close()
            raise

    def prepare(self, rows, block_size, threads):
        for source in self.sources:
            source.prepare(rows, block_size, threads)
        self.rows = indices(rows, name="shard rows", size=len(self.samples))
        self.generation += 1
        self.child_generations = tuple(s.generation for s in self.sources)

    def read(self, variants):
        if self.rows is None:
            raise RuntimeError("sharded source is not prepared")
        self.check()
        variants = indices(variants, name="shard variants", size=len(self.variants.ids))
        if np.any(np.diff(variants) <= 0):
            raise ValueError("sharded variant request must be sorted")
        out = np.empty((len(self.rows), len(variants)), dtype=np.int8 if self.hard_calls else np.float64, order="F")
        for j, source in enumerate(self.sources):
            lo, hi = np.searchsorted(variants, self.offsets[j:j+2])
            if hi > lo:
                out[:, lo:hi] = source.read(variants[lo:hi]-self.offsets[j])
        return out

    def check(self):
        if not self.sources:
            raise RuntimeError("sharded genotype source is closed")
        for source in self.sources:
            source.check()
        if self.child_generations is not None and tuple(s.generation for s in self.sources) != self.child_generations:
            raise RuntimeError("a genotype shard was prepared by another stream")

    def close(self):
        for source in self.sources:
            source.close()
        self.sources = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def source_from_spec(spec, root):
    from ._validation import closed
    closed(spec, ("genome_build",), ("geno", "shards"), name="genotype specification")
    if ("geno" in spec) == ("shards" in spec):
        raise ValueError("supply exactly one genotype trio or ordered shard list")
    if "geno" in spec:
        return FileGenotypeSource(Path(root)/spec["geno"], genome_build=spec["genome_build"])
    if not isinstance(spec["shards"], list) or any(not isinstance(x, str) or not x for x in spec["shards"]):
        raise ValueError("genotype shards must be an ordered list of paths")
    return ShardedGenotypeSource([Path(root)/x for x in spec["shards"]], genome_build=spec["genome_build"])


def standardize(raw, mean, inverse_scale):
    result = np.array(raw, dtype=np.float64, order="F", copy=True)
    missing = result == -127
    result -= mean
    result *= inverse_scale
    result[missing] = 0.0
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite standardized genotypes")
    return result


class RawBlockStream:
    def __init__(self, source, rows, variants, *, block_size=512, storage="stream", threads=1, ledger=None):
        if storage not in ("stream", "compact", "standardized"):
            raise ValueError("unknown genotype storage mode")
        self.source = source
        self.rows = np.unique(rows)
        self.variants = np.unique(variants)
        self.block_size = positive_int(block_size, "block_size")
        self.storage = storage
        self.ledger = ledger if ledger is not None else PassLedger()
        self.cache = None
        source.prepare(self.rows, self.block_size, threads)
        self.generation = source.generation

    def check(self):
        self.source.check()
        if self.source.generation != self.generation:
            raise RuntimeError("genotype source was prepared by another stream; use a separate source owner")

    def blocks(self, phase, *, build_cache=False):
        self.check()
        self.ledger.begin(phase)
        existing = self.cache is not None
        if build_cache:
            if existing or self.storage != "compact":
                raise ValueError("compact cache can be built exactly once")
            self.cache = np.empty((len(self.variants), len(self.rows)),
                dtype=np.int8 if self.source.hard_calls else np.float64)
        for start in range(0, len(self.variants), self.block_size):
            self.check()
            stop = min(start + self.block_size, len(self.variants))
            variants = self.variants[start:stop]
            if existing:
                raw = self.cache[start:stop].T
                self.ledger.cache_blocks += 1
                self.ledger.cache_variants += len(variants)
            else:
                raw = self.source.read(variants)
                self.ledger.source_blocks += 1
                self.ledger.source_variants += len(variants)
                self.ledger.source_decoded_bytes += raw.nbytes
                if build_cache:
                    self.cache[start:stop] = raw.T
            yield start, variants, raw
        if build_cache:
            self.cache.setflags(write=False)
        self.check()


def estimate_scale(source, rows, variants, *, ddof=1, block_size=512, threads=1, memory_bytes=16*2**30):
    """Explicit one-pass discovery scale fitting, separate from architecture fitting.

    The denominator is N-ddof after training-mean imputation, including missing
    rows, matching DirectContext. Call before requesting an exact architecture
    reference; never substitute this for missing reference affine provenance.
    """
    rows = indices(rows, name="scale rows", size=len(source.samples))
    variants = indices(variants, name="scale variants", size=len(source.variants.ids))
    if ddof not in (0, 1) or len(rows) <= ddof or np.any(np.diff(variants) <= 0):
        raise ValueError("invalid scale sample count, ddof or variant order")
    positive_int(memory_bytes, "memory_bytes")
    positive_int(block_size, "block_size")
    required = 40*len(rows)*min(block_size, len(variants)) + 16*len(variants) + 256*2**20
    if required > memory_bytes:
        raise MemoryError("scale setup exceeds memory budget; reduce block size")
    stream = RawBlockStream(source, rows, variants, block_size=block_size, threads=threads)
    mean, inv = np.empty(len(variants)), np.empty(len(variants))
    for start, v, raw in stream.blocks("scale"):
        observed = raw != -127
        calls = np.where(observed, raw, 0).astype(np.float64)
        count = observed.sum(axis=0)
        mu = np.divide(calls.sum(axis=0), count, out=np.zeros(len(v)), where=count > 0)
        centered = np.where(observed, calls - mu, 0)
        ss = np.einsum("ij,ij->j", centered, centered)
        mean[start:start+len(v)] = mu
        inv[start:start+len(v)] = np.sqrt(np.divide(len(rows)-ddof, ss, out=np.ones_like(ss), where=ss > 0))
    return GenotypeScale(mean, inv, source.variants.subset(variants).identity,
        digest([source.samples[int(i)] for i in rows]),
        {"method": "training_mean_imputed_variance", "source": source.identity}, ddof)
