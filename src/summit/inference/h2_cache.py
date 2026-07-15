from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .. import utils

try:
    import fcntl
except ImportError:  # pragma: no cover - SUMMIT production targets are Unix.
    fcntl = None


_CACHE_VERSION = 3


def trace_axis_digest(snps: np.ndarray) -> str:
    """Return a deterministic digest of the ordered full Trace SNP axis."""
    values = np.asarray(snps)
    if values.ndim != 1:
        raise ValueError(f"Trace SNP axis must be one-dimensional; got {values.shape}")
    digest = hashlib.sha256()
    digest.update(str(values.shape[0]).encode("ascii"))
    digest.update(b"\0")
    # Trace SNPs normally have object dtype. Hashing that raw buffer would hash
    # process-specific PyObject addresses, so encode bounded Unicode blocks and
    # hash their actual code points instead. The block dtype width is included
    # to make boundaries and trailing nulls unambiguous.
    block_size = 100_000
    for start in range(0, values.size, block_size):
        block = np.asarray(values[start : start + block_size], dtype=np.str_)
        if not block.flags.c_contiguous:
            block = np.ascontiguousarray(block)
        digest.update(str(block.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(block.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _source_fingerprint(path_spec: str) -> dict:
    files = utils._resolve_chr_split_paths(path_spec, require=True)
    records = []
    for file_path in files:
        stat = os.stat(file_path)
        records.append(
            {
                "path": str(Path(file_path).resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    # Identity is based on resolved files rather than the requested '@' spec so
    # equivalent shard/symlink layouts can reuse the same cache entry.
    return {"files": records}


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (stem or "trait")[:96]


def _sha256_array(values: np.ndarray) -> str:
    values = np.asarray(values)
    if not values.flags.c_contiguous:
        values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class H2CacheEntry:
    key: str
    source: dict
    config: dict
    axis_digest: str
    prefix: Path

    @property
    def metadata_path(self) -> Path:
        return Path(str(self.prefix) + ".json")

    @property
    def y_path(self) -> Path:
        return Path(str(self.prefix) + ".y.npy")

    @property
    def active_path(self) -> Path:
        return Path(str(self.prefix) + ".active.packbits.npy")

    @property
    def lock_path(self) -> Path:
        return Path(str(self.prefix) + ".lock")


class H2TraitCache:
    def __init__(
        self,
        root: str,
        *,
        axis_digest: str,
        nsnps: int,
        mode: str = "readwrite",
        verify_checksum: bool = False,
    ):
        mode = str(mode).strip().lower()
        if mode not in {"read", "readwrite", "refresh"}:
            raise ValueError("h2 cache mode must be read, readwrite, or refresh")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.axis_digest = str(axis_digest)
        self.nsnps = int(nsnps)
        self.mode = mode
        self.verify_checksum = bool(verify_checksum)

    def entry(self, *, path: str, phen: str, cov_rank, chisq_threshold, chisq_action: str) -> H2CacheEntry:
        source = _source_fingerprint(path)
        config = {
            "explicit_cov_rank": None if cov_rank is None else int(cov_rank),
            "chisq_threshold": None if chisq_threshold is None else str(chisq_threshold),
            "chisq_action": str(chisq_action).strip().lower(),
        }
        identity = {
            "version": _CACHE_VERSION,
            "source": source,
            "config": config,
            "axis_digest": self.axis_digest,
            "nsnps": self.nsnps,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        prefix = self.root / key[:2] / f"{_safe_stem(phen)}.{key[:24]}"
        return H2CacheEntry(
            key=key,
            source=source,
            config=config,
            axis_digest=self.axis_digest,
            prefix=prefix,
        )

    @contextmanager
    def lock(self, entry: H2CacheEntry):
        if fcntl is None:
            raise RuntimeError("The fast-h2 shared cache requires POSIX file locking.")
        entry.prefix.parent.mkdir(parents=True, exist_ok=True)
        with open(entry.lock_path, "a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load_into(
        self,
        entry: H2CacheEntry,
        y_column: np.ndarray,
        active_column: np.ndarray,
    ) -> dict | None:
        if not entry.metadata_path.is_file():
            return None
        try:
            metadata = json.loads(entry.metadata_path.read_text())
            if int(metadata.get("version", -1)) != _CACHE_VERSION:
                return None
            if metadata.get("key") != entry.key:
                return None
            if metadata.get("source") != entry.source:
                return None
            if metadata.get("config") != entry.config:
                return None
            if metadata.get("axis_digest") != entry.axis_digest:
                return None
            if int(metadata.get("nsnps", -1)) != self.nsnps:
                return None

            y = np.load(entry.y_path, mmap_mode="r", allow_pickle=False)
            packed = np.load(entry.active_path, mmap_mode="r", allow_pickle=False)
            if y.shape != (self.nsnps,) or y.dtype != np.dtype(np.float64):
                return None
            expected_packed = (self.nsnps + 7) // 8
            if packed.shape != (expected_packed,) or packed.dtype != np.dtype(np.uint8):
                return None
            if self.verify_checksum:
                if _sha256_array(y) != metadata.get("y_sha256"):
                    raise RuntimeError(f"Cached y checksum failed: {entry.y_path}")
                if _sha256_array(packed) != metadata.get("active_sha256"):
                    raise RuntimeError(f"Cached active-mask checksum failed: {entry.active_path}")

            y_column[:] = y
            active_column[:] = np.unpackbits(packed, bitorder="little", count=self.nsnps).astype(bool, copy=False)
            if not np.isfinite(y_column).all():
                raise RuntimeError(f"Cached y contains non-finite values: {entry.y_path}")
            if np.any(y_column[~active_column] != 0.0):
                raise RuntimeError(f"Cached inactive y values are not zero: {entry.y_path}")
            return metadata["trait"]
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def write(
        self,
        entry: H2CacheEntry,
        y_column: np.ndarray,
        active_column: np.ndarray,
        trait_payload: dict,
    ):
        y = np.asarray(y_column, dtype=np.float64)
        active = np.asarray(active_column, dtype=bool)
        if y.shape != (self.nsnps,) or active.shape != (self.nsnps,):
            raise ValueError("Cannot cache h2 arrays with an unexpected shape")
        if not np.isfinite(y[active]).all() or np.any(y[~active] != 0.0):
            raise ValueError("Cannot cache invalid h2 moment arrays")

        packed = np.packbits(active, bitorder="little")
        token = f"{os.getpid()}.{id(y_column)}"
        tmp_y = Path(str(entry.y_path) + f".{token}.tmp")
        tmp_active = Path(str(entry.active_path) + f".{token}.tmp")
        tmp_meta = Path(str(entry.metadata_path) + f".{token}.tmp")
        entry.prefix.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "version": _CACHE_VERSION,
            "key": entry.key,
            "source": entry.source,
            "config": entry.config,
            "axis_digest": entry.axis_digest,
            "nsnps": self.nsnps,
            "y_sha256": _sha256_array(y),
            "active_sha256": _sha256_array(packed),
            "trait": trait_payload,
        }
        try:
            with open(tmp_y, "wb") as handle:
                np.save(handle, y, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            with open(tmp_active, "wb") as handle:
                np.save(handle, packed, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            with open(tmp_meta, "w") as handle:
                json.dump(metadata, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_y, entry.y_path)
            os.replace(tmp_active, entry.active_path)
            # Metadata is the completion marker and must be published last.
            os.replace(tmp_meta, entry.metadata_path)
        finally:
            for path in (tmp_y, tmp_active, tmp_meta):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
