"""Atomic, exclusively owned projected-PCG restart state (never genotype data).

Contains individual-level solver vectors. Store with the training data, never
in a portable/public model bundle. A checkpoint replaces only its own prior
generation after fsync; termination during writing leaves the last save valid.
"""
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ._validation import array_digest, canonical


class SolverCheckpoint(AbstractContextManager):
    def __init__(self, path, identity, *, resume=False):
        self.path, self.identity, self.resume = Path(path), identity, resume
        self.lock = None

    def __enter__(self):
        import fcntl
        self.lock = os.open(str(self.path)+".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume and not self.path.is_file():
                raise FileNotFoundError(self.path)
            if not self.resume and self.path.exists():
                raise FileExistsError(self.path)
            if self.resume:
                with np.load(self.path, allow_pickle=False) as data:
                    meta = json.loads(data["metadata"].tobytes())
                if meta["schema"] != 1 or meta["identity"] != self.identity:
                    raise ValueError("checkpoint fit/solver/implementation identity mismatch")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def save(self, state):
        if self.lock is None:
            raise RuntimeError("checkpoint requires an exclusive owner")
        arrays, mapped = {}, {}
        for field in ("x", "residual", "directions", "fixed_coefficients"):
            mapped[field] = []
            for key, value in sorted(state[field].items()):
                name = f"a{len(arrays)}"
                arrays[name] = value
                mapped[field].append([list(key), name, array_digest(value)])
        meta = dict(schema=1, identity=self.identity, arrays=mapped,
            iteration=state["iteration"], active=state["active"], pending=state["pending"],
            reports=[[list(k), v] for k, v in sorted(state["reports"].items())],
            rho=[[list(k), v] for k, v in sorted(state["rho"].items())],
            elapsed_seconds=state["elapsed_seconds"])
        arrays["metadata"] = np.frombuffer(canonical(meta).encode(), dtype=np.uint8)
        fd, name = tempfile.mkstemp(prefix=self.path.name+".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load(self, vectors, fixed_sizes):
        if self.lock is None:
            raise RuntimeError("checkpoint requires an exclusive owner")
        with np.load(self.path, allow_pickle=False) as data:
            metadata = data["metadata"]
            if metadata.dtype != np.uint8 or metadata.ndim != 1 or metadata.nbytes > 16*2**20:
                raise ValueError("invalid checkpoint metadata")
            meta = json.loads(metadata.tobytes())
            if meta["schema"] != 1 or meta["identity"] != self.identity:
                raise ValueError("checkpoint fit/solver/implementation identity mismatch")
            result = {k: meta[k] for k in ("iteration", "active", "pending", "elapsed_seconds")}
            if type(result["iteration"]) is not int or result["iteration"] < 0:
                raise ValueError("invalid checkpoint iteration")
            for field in ("reports", "rho"):
                result[field] = {tuple(k): v for k, v in meta[field]}
                if set(result[field]) != set(vectors):
                    raise ValueError("checkpoint model keys disagree")
            for field, entries in meta["arrays"].items():
                if field not in ("x", "residual", "directions", "fixed_coefficients"):
                    raise ValueError("unknown checkpoint state")
                result[field] = {}
                for key, name, checksum in entries:
                    key = tuple(key)
                    value = data[name]
                    if key not in vectors or key in result[field]:
                        raise ValueError("invalid checkpoint array key")
                    size = fixed_sizes[key] if field == "fixed_coefficients" else len(vectors[key])
                    if (value.shape != (size,) or value.dtype != np.float64 or
                        not np.all(np.isfinite(value)) or array_digest(value) != checksum):
                        raise ValueError("checkpoint array shape/checksum mismatch")
                    result[field][key] = value
            for field in ("x", "residual", "directions"):
                if set(result[field]) != set(vectors):
                    raise ValueError("checkpoint vector keys disagree")
            active, pending = set(map(tuple, result["active"])), set(map(tuple, result["pending"]))
            if active & pending or not (active | pending) <= set(vectors):
                raise ValueError("checkpoint active/pending keys disagree")
            converged = {k for k, rep in result["reports"].items() if rep["converged"]}
            if converged & (active | pending) or set(result["fixed_coefficients"]) != converged:
                raise ValueError("checkpoint convergence state disagrees")
            return result
