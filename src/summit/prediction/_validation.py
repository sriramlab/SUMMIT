"""Small, shared validation and content-identity primitives."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np


def array(value, *, name, ndim=None, dtype=np.float64):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=dtype, copy=True, order="F")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    result.setflags(write=False)
    return result


def indices(value, *, name, size=None):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not raw.size:
        raise ValueError(f"{name} must be a nonempty integer vector")
    if np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max) or (size is not None and np.any(raw >= size)):
        raise ValueError(f"{name} out of range")
    result = array(raw, name=name, ndim=1, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} contains duplicates")
    return result


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def identifier(value, name="identifier"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {name}")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def metadata(value):
    # Round trip detaches caller-owned mutable metadata; public serialization
    # always copies it again. Only JSON-compatible, finite values are allowed.
    return json.loads(canonical(value))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_digest(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError("object arrays have no portable numeric identity")
    h = hashlib.sha256(canonical([a.dtype.str, list(a.shape)]).encode())
    # C-order iteration is independent of layout and bounded even for mmap.
    it = np.nditer(a, flags=["external_loop", "buffered", "zerosize_ok"],
                   order="C", buffersize=131072)
    for chunk in it:
        h.update(chunk.tobytes())
    return h.hexdigest()


def closed(value, required, optional=(), *, name="specification"):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{name}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}")


def psd(value, *, name="covariance", rtol=1e-12):
    a = array(value, name=name, ndim=2)
    if not a.shape[0] or a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be nonempty and square")
    scale = max(float(np.max(np.abs(a))), np.finfo(float).tiny)
    if np.max(np.abs(a - a.T)) > rtol * scale:
        raise ValueError(f"{name} is not symmetric")
    a = a * 0.5 + a.T * 0.5
    w, v = np.linalg.eigh(a)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} has an unrepresentable eigensystem")
    if w[0] < -rtol * scale:
        raise ValueError(f"{name} is not positive semidefinite")
    # Remove only roundoff-negative eigenvalues, never ridge singular priors.
    if w[0] < 0:
        a = (v * np.maximum(w, 0)) @ v.T
    a.setflags(write=False)
    return a
