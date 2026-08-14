from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import stat
import tempfile
import time
import weakref
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from bed_reader import open_bed
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .. import utils
from .genotype_source import (
    PgenBlockReader,
    read_fam_sample_ids,
    read_psam_sample_ids,
    read_pvar_variants,
    resolve_genotype_input,
    validate_variant_metadata,
)



def _canonical_bfile_prefix(x: str) -> str:
    s = str(x)
    for ext in (".bed", ".bim", ".fam"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def _validate_plink_file_paths(paths: Mapping[str, Path]) -> tuple[int, int]:
    """Fail early on truncated/misaligned SNP-major PLINK 1 files."""
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PLINK input file(s): {missing}.")

    counts = {}
    for ext in (".fam", ".bim"):
        with open(paths[ext], "rb") as handle:
            count = 0
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{paths[ext]} contains a blank row at line {line_number}.")
                count += 1
        counts[ext] = count
    n, m = counts[".fam"], counts[".bim"]
    if n <= 0 or m <= 0:
        raise ValueError(f"PLINK FAM/BIM must be non-empty; observed N={n}, M={m}.")
    expected = 3 + ((n + 3) // 4) * m
    with open(paths[".bed"], "rb") as handle:
        magic = handle.read(3)
    observed = paths[".bed"].stat().st_size
    if magic != b"\x6c\x1b\x01" or observed != expected:
        raise ValueError(
            "Invalid or truncated SNP-major PLINK BED: "
            f"N={n}, M={m}, expected_bytes={expected}, observed_bytes={observed}, magic={magic.hex()}."
        )
    return n, m


def _validate_plink_bed_shape(prefix: str) -> tuple[int, int]:
    """Fail early on truncated/misaligned SNP-major PLINK 1 triples."""
    return _validate_plink_file_paths(
        {ext: Path(prefix + ext) for ext in (".bed", ".bim", ".fam")}
    )


def _close_file_descriptors(descriptors: Sequence[int]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _validate_jackknife_probe_count(nvecs: int, enabled: bool, allow_low: bool) -> None:
    if enabled and int(nvecs) < 100 and not allow_low:
        raise ValueError(
            "Exact GxE jackknife traces require at least 100 random probes by default; "
            f"got {int(nvecs)}. Low-probe within-block trace noise can materially distort SEs. "
            "Use --allow-low-probe-gxe-jackknife only for explicit diagnostics."
        )


def _ndarray_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x1f")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


_BACKEND_PROVENANCE_SCHEMA_VERSION = 3
_NATIVE_GEMM_INTEGRITY_CHECKS = 8
_NATIVE_GEMM_CHECK_MINIMUM_FLOPS = 1_000_000_000
_NATIVE_PREFERRED_CALL_WORKSPACE_BYTES = 3 * 1024**3


def _native_gemm_integrity_workspace_elements(
    build_info: Mapping | None, m: int, n: int, k: int
) -> int:
    """Mirror the native checksum and bounded operand-snapshot allocation."""
    if (
        build_info is None
        or min(int(m), int(n), int(k)) <= 0
        or 2 * int(m) * int(n) * int(k)
        < _NATIVE_GEMM_CHECK_MINIMUM_FLOPS
    ):
        return 0
    checks = _NATIVE_GEMM_INTEGRITY_CHECKS * (
        int(m) + 2 * int(k) + 2 * int(n)
    )
    return checks + int(m) * int(k) + int(k) * int(n)


def _native_execution_workspace_bytes(native_workspace_gib: float) -> int:
    """Use the configured workspace as a ceiling, not an allocation target."""
    configured = int(float(native_workspace_gib) * (1024**3))
    return min(configured, _NATIVE_PREFERRED_CALL_WORKSPACE_BYTES)


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)
    return digest.hexdigest()


def _loaded_native_binary_record(native_module) -> tuple[int, dict]:
    """Bind the extension's mapped inode to a stable descriptor and exact bytes."""
    module_path = Path(native_module.__file__).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(module_path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("The loaded GxE native extension is not a regular file.")
        mapped = False
        with open("/proc/self/maps", "r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.split(None, 5)
                if len(fields) < 5:
                    continue
                try:
                    major_text, minor_text = fields[3].split(":", 1)
                    device = os.makedev(int(major_text, 16), int(minor_text, 16))
                    inode = int(fields[4])
                except (ValueError, OSError):
                    continue
                if device == before.st_dev and inode == before.st_ino:
                    mapped = True
                    break
        if not mapped:
            raise RuntimeError(
                "The GxE native extension pathname does not identify the inode loaded "
                "into this process."
            )
        digest = _sha256_descriptor(descriptor)
        after = os.fstat(descriptor)
        identity = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        if identity != (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ):
            raise RuntimeError("The loaded GxE native extension changed while hashing.")
        return descriptor, {
            "path": str(module_path),
            "sha256": digest,
            "bytes": int(before.st_size),
            "identity": identity,
        }
    except Exception:
        os.close(descriptor)
        raise


def _native_strict_feature_moment_verification_policy(
    build_info: Mapping,
) -> tuple[bool, str]:
    """Select the optional duplicate feature-moment diagnostic."""
    override = os.environ.get(
        "SUMMIT_GXE_VERIFY_FEATURE_MOMENTS", "auto"
    ).strip().lower()
    if override not in {"auto", "always"}:
        raise ValueError(
            "SUMMIT_GXE_VERIFY_FEATURE_MOMENTS must be 'auto' or 'always'."
        )
    if override == "always":
        return True, "forced by SUMMIT_GXE_VERIFY_FEATURE_MOMENTS=always"

    if str(build_info.get("blas_vendor", "")).strip().lower() == "openblas":
        return False, (
            "deterministic tiled feature GEMMs; eight-check ABFT over "
            "partitioned single-thread OpenBLAS trace GEMMs"
        )
    return False, (
        "deterministic disjoint-output tiled GEMMs independent of "
        f"{build_info.get('blas_vendor')}"
    )


def _validate_backend_provenance(value: Mapping, *, expected_stage: str | None = None) -> None:
    """Validate implementation provenance without making it scientific identity."""
    if not isinstance(value, Mapping):
        raise ValueError("GxE backend provenance must be a JSON object.")
    required = {
        "schema_version", "artifact_stage", "backend_name", "backend_version",
        "source_commit", "source_tree_sha256", "native_binary_sha256", "compile_options",
        "native_workspace_cap_bytes", "configured_target_panel_columns",
        "actual_global_2b_source_columns", "actual_jackknife_2b_source_columns",
        "actual_target_source_columns",
    }
    if set(value) != required:
        raise ValueError("GxE backend provenance has unexpected or missing fields.")
    if value.get("schema_version") != _BACKEND_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported GxE backend-provenance schema.")
    stage = value.get("artifact_stage")
    if not isinstance(stage, str) or not stage or (
        expected_stage is not None and stage != expected_stage
    ):
        raise ValueError("GxE backend provenance has an invalid artifact stage.")
    for field in ("backend_name", "backend_version"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"GxE backend provenance field {field!r} is invalid.")
    source_commit = value.get("source_commit")
    if source_commit != "unknown" and not (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("GxE backend provenance source_commit is invalid.")
    source_tree_sha256 = value.get("source_tree_sha256")
    if source_tree_sha256 is not None and not _is_canonical_sha256(source_tree_sha256):
        raise ValueError("GxE backend provenance source_tree_sha256 is invalid.")
    native_hash = value.get("native_binary_sha256")
    if native_hash is not None and not _is_canonical_sha256(native_hash):
        raise ValueError("GxE backend provenance native binary SHA-256 is invalid.")
    compile_options = value.get("compile_options")
    if compile_options is not None and not isinstance(compile_options, Mapping):
        raise ValueError("GxE backend provenance compile_options must be null or an object.")
    integer_fields = (
        "native_workspace_cap_bytes", "configured_target_panel_columns",
        "actual_global_2b_source_columns", "actual_jackknife_2b_source_columns",
        "actual_target_source_columns",
    )
    for field in integer_fields:
        observed = value.get(field)
        if isinstance(observed, bool) or not isinstance(observed, (int, np.integer)) or observed < 0:
            raise ValueError(f"GxE backend provenance field {field!r} is invalid.")
    global_width = int(value["actual_global_2b_source_columns"])
    jackknife_width = int(value["actual_jackknife_2b_source_columns"])
    target_width = int(value["actual_target_source_columns"])
    if jackknife_width not in (0, global_width):
        raise ValueError("GxE backend provenance has inconsistent global/block widths.")
    if target_width != global_width:
        raise ValueError("GxE backend provenance has an inconsistent target width.")
    if value["backend_name"] == "gxeldcore_direct":
        if (
            source_commit == "unknown"
            or source_tree_sha256 is None
            or
            native_hash is None
            or compile_options is None
            or int(value["native_workspace_cap_bytes"]) <= 0
            or int(value["configured_target_panel_columns"]) <= 0
        ):
            raise ValueError("Direct GxE backend provenance is incomplete.")


_FEATURE_CACHE_SCHEMA_VERSION = 2
_FEATURE_CACHE_ARRAY_DTYPES = {
    "variant_chr": "U",
    "variant_snp": "U",
    "variant_bp": np.dtype(np.int64),
    "variant_a1": "U",
    "variant_a2": "U",
    "annotations": np.dtype(np.float64),
    "jackknife_ids": np.dtype(np.int32),
    "scale_x": np.dtype(np.float64),
    "scale_w": np.dtype(np.float64),
    "norm_x": np.dtype(np.float64),
    "norm_w": np.dtype(np.float64),
    "diag_nxe_x": np.dtype(np.float64),
    "diag_nxe_w": np.dtype(np.float64),
    "corr_xw": np.dtype(np.float64),
    "nxe_qdq": np.dtype(np.float64),
    "nxe_trace_terms": np.dtype(np.float64),
}

_GXE_RESERVED_ARTIFACT_COLUMNS = frozenset(
    {
        "CHR", "SNP", "BP", "A1", "A2", "NORM_X", "NORM_W",
        "SCALE_X", "SCALE_W", "DNXE_X", "DNXE_W", "CORR_XW",
        "BLOCK", "N", "DF", "SCORE", "SCORE_MODE",
    }
)


def _validate_gxe_annotation_names(names: Sequence[str]) -> list[str]:
    """Return canonical annotation names or fail on unsafe artifact columns."""
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("GxE annotation names must be a non-empty sequence.")
    canonical: list[str] = []
    invalid: list[str] = []
    for value in names:
        if not isinstance(value, str):
            invalid.append(repr(value))
            continue
        name = value
        canonical.append(name)
        if (
            not name
            or name.upper() in _GXE_RESERVED_ARTIFACT_COLUMNS
            or any(character.isspace() or ord(character) < 32 for character in name)
        ):
            invalid.append(repr(name))
    if invalid or len({name.casefold() for name in canonical}) != len(canonical):
        raise ValueError(
            "GxE annotation names must be unique (case-insensitive), non-empty, free of "
            "whitespace/control characters, and must not collide with reserved artifact "
            f"columns; invalid names: {invalid}."
        )
    return canonical


def _is_canonical_sha256(value) -> bool:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or value == "0" * 64
    ):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _metadata_int(metadata: Mapping, name: str, *, minimum: int | None = None) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"GxE feature-cache metadata field {name!r} must be an integer.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(
            f"GxE feature-cache metadata field {name!r} must be at least {minimum}; got {value}."
        )
    return value


def _metadata_float(metadata: Mapping, name: str, *, positive: bool = False) -> float:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"GxE feature-cache metadata field {name!r} must be numeric.")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"GxE feature-cache metadata field {name!r} must be {qualifier}.")
    return value


def _require_close(name: str, observed, expected, *, rtol: float = 2e-12, atol: float = 2e-10) -> None:
    try:
        close = bool(np.allclose(observed, expected, rtol=rtol, atol=atol, equal_nan=False))
    except (TypeError, ValueError):
        close = False
    if not close:
        raise ValueError(f"GxE feature cache has inconsistent {name}.")


def _environment_transforms_equal(observed: Any, expected: Any) -> bool:
    """Compare one fixed design exactly, allowing only floating reduction noise."""
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        return False
    if set(observed) != set(expected):
        return False
    exact_fields = (
        "standardized", "ddof", "units", "fixed_effect_design_sha256",
    )
    if any(observed.get(name) != expected.get(name) for name in exact_fields):
        return False
    numeric_fields = (
        "raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares",
    )
    try:
        observed_values = np.asarray(
            [observed[name] for name in numeric_fields], dtype=np.float64,
        )
        expected_values = np.asarray(
            [expected[name] for name in numeric_fields], dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.allclose(
        observed_values, expected_values, rtol=2e-12, atol=2e-10,
        equal_nan=False,
    ))


def _feature_cache_variant_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for values in zip(
        arrays["variant_chr"], arrays["variant_snp"], arrays["variant_bp"],
        arrays["variant_a1"], arrays["variant_a2"],
    ):
        digest.update("\x1f".join(str(value) for value in values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _feature_cache_annotation_digest(metadata: Mapping, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in metadata["annotation_names"]:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\n")
    digest.update(
        np.asarray(arrays["annotations"], dtype="<f8", order="C").tobytes(order="C")
    )
    return digest.hexdigest()


def _feature_cache_jackknife_digest(metadata: Mapping, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    labels = metadata["jackknife_labels"]
    if labels is None:
        digest.update(b"none\n")
    else:
        digest.update(
            np.asarray(arrays["jackknife_ids"], dtype="<i4", order="C").tobytes(order="C")
        )
        for label in labels:
            digest.update(str(label).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _validate_feature_cache_semantics(
    metadata: Mapping,
    arrays: Mapping[str, np.ndarray],
    *,
    expected_identity: Mapping | None = None,
    expected_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Validate the complete schema-v2 feature-cache contract.

    This is deliberately independent of ``GenomewideEnvLDScore`` so shard
    mergers can apply exactly the same checks.  Array hashes provide byte
    integrity; the remaining checks bind those bytes to the declared variants,
    annotations, jackknife, feature diagnostics, and exact NxE traces.  Cache
    authenticity is supplied separately by an expected current-data identity
    (during generation/loading) or by the cache digest sealed into each shard.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("GxE feature-cache metadata must be a JSON object.")
    if metadata.get("kind") != "summit.gxe.feature_cache":
        raise ValueError("Unsupported GxE feature-cache kind.")
    if metadata.get("schema_version") != _FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported GxE feature-cache schema; regenerate a schema-v2 cache "
            "so NxE traces can be verified."
        )
    if "backend_provenance" in metadata:
        _validate_backend_provenance(
            metadata["backend_provenance"], expected_stage="feature_construction"
        )
    if not isinstance(arrays, Mapping):
        raise ValueError("GxE feature-cache arrays must be a mapping.")
    required_names = set(_FEATURE_CACHE_ARRAY_DTYPES)
    if set(arrays) != required_names:
        missing = sorted(required_names - set(arrays))
        extra = sorted(set(arrays) - required_names)
        raise ValueError(
            f"GxE feature-cache array set is invalid; missing={missing}, extra={extra}."
        )

    declared_hashes = metadata.get("array_sha256")
    if not isinstance(declared_hashes, Mapping) or set(declared_hashes) != required_names:
        raise ValueError("GxE feature cache does not bind every schema-v2 array.")
    for name in sorted(required_names):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"GxE feature-cache array {name!r} is not an ndarray.")
        expected_dtype = _FEATURE_CACHE_ARRAY_DTYPES[name]
        if expected_dtype == "U":
            dtype_ok = value.dtype.kind == "U"
        else:
            dtype_ok = value.dtype == expected_dtype
        if not dtype_ok:
            raise ValueError(
                f"GxE feature-cache array {name!r} has dtype {value.dtype}; "
                f"expected {expected_dtype}."
            )
        declared = declared_hashes.get(name)
        if not _is_canonical_sha256(declared) or _ndarray_sha256(value) != declared:
            raise ValueError(f"GxE feature-cache array {name!r} failed its SHA-256 check.")

    n = _metadata_int(metadata, "n_samples", minimum=3)
    m = _metadata_int(metadata, "n_variants", minimum=1)
    p = _metadata_int(metadata, "fixed_effect_rank_excluding_intercept", minimum=1)
    residual_rank = _metadata_int(metadata, "residual_rank", minimum=1)
    if p >= n - 1 or residual_rank != n - p - 1:
        raise ValueError("GxE feature-cache fixed-effect and residual ranks are inconsistent.")
    ddof = _metadata_int(metadata, "ddof")
    if ddof not in (0, 1):
        raise ValueError("GxE feature-cache ddof must be 0 or 1.")
    _metadata_float(metadata, "eps_var", positive=True)
    kernel_mode = metadata.get("kernel_mode")
    genotype_scale = metadata.get("genotype_scale")
    if kernel_mode not in ("standardized", "genie"):
        raise ValueError("GxE feature-cache kernel_mode is invalid.")
    if genotype_scale not in ("sample", "hwe"):
        raise ValueError("GxE feature-cache genotype_scale is invalid.")

    annotation_names = metadata.get("annotation_names")
    if not isinstance(annotation_names, list):
        raise ValueError("GxE feature-cache annotation names must be a JSON list.")
    annotation_names = _validate_gxe_annotation_names(annotation_names)
    k = len(annotation_names)
    annotation_masses = metadata.get("annotation_masses")
    if not isinstance(annotation_masses, list) or len(annotation_masses) != k:
        raise ValueError("GxE feature-cache annotation masses have an invalid shape.")
    if any(isinstance(value, bool) for value in annotation_masses):
        raise ValueError("GxE feature-cache annotation masses must be numeric, not boolean.")
    try:
        annotation_masses = np.asarray(annotation_masses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("GxE feature-cache annotation masses must be numeric.") from exc
    if not np.all(np.isfinite(annotation_masses)) or np.any(annotation_masses <= 0.0):
        raise ValueError("GxE feature-cache annotation masses must be positive and finite.")

    one_dimensional = (
        "variant_chr", "variant_snp", "variant_bp", "variant_a1", "variant_a2",
        "scale_x", "scale_w", "norm_x", "norm_w", "diag_nxe_x", "diag_nxe_w",
        "corr_xw",
    )
    for name in one_dimensional:
        if arrays[name].shape != (m,):
            raise ValueError(f"GxE feature-cache array {name!r} must have shape ({m},).")
    if arrays["annotations"].shape != (m, k):
        raise ValueError(
            f"GxE feature-cache annotations must have shape ({m}, {k})."
        )
    q = p + 1
    if arrays["nxe_qdq"].shape != (q, q):
        raise ValueError(f"GxE feature-cache nxe_qdq must have shape ({q}, {q}).")
    if arrays["nxe_trace_terms"].shape != (3,):
        raise ValueError("GxE feature-cache nxe_trace_terms must have shape (3,).")

    for name in (
        "annotations", "scale_x", "scale_w", "norm_x", "norm_w", "diag_nxe_x",
        "diag_nxe_w", "corr_xw", "nxe_qdq", "nxe_trace_terms",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"GxE feature-cache array {name!r} contains NaN or infinity.")
    if np.any(arrays["annotations"] < 0.0):
        raise ValueError("GxE feature-cache annotations must be non-negative.")
    computed_masses = np.asarray(arrays["annotations"], dtype=np.float64).sum(axis=0)
    _require_close("annotation masses", annotation_masses, computed_masses, rtol=0.0, atol=0.0)
    if np.any(arrays["scale_x"] <= 0.0) or np.any(arrays["scale_w"] <= 0.0):
        raise ValueError("GxE feature cache contains nonpositive projected-feature scales.")
    if np.any(arrays["norm_x"] <= 0.0) or np.any(arrays["norm_w"] <= 0.0):
        raise ValueError("GxE feature cache contains nonpositive projected-feature norms.")
    if np.any(arrays["diag_nxe_x"] < 0.0) or np.any(arrays["diag_nxe_w"] < 0.0):
        raise ValueError("GxE feature cache contains a negative NxE feature diagonal.")
    cauchy_slack = (
        arrays["norm_x"] * arrays["norm_w"] - arrays["corr_xw"] ** 2
    )
    cauchy_tol = 5e-12 * np.maximum(1.0, arrays["norm_x"] * arrays["norm_w"])
    if np.any(cauchy_slack < -cauchy_tol):
        raise ValueError("GxE feature-cache X/W correlations violate Cauchy-Schwarz.")
    if kernel_mode == "standardized":
        if max(
            float(np.max(np.abs(arrays["norm_x"] - 1.0))),
            float(np.max(np.abs(arrays["norm_w"] - 1.0))),
        ) > 1.0e-9:
            raise ValueError("Standardized GxE feature-cache columns do not have unit residual norm.")
    elif not (
        np.array_equal(arrays["scale_x"], np.ones(m, dtype=np.float64))
        and np.array_equal(arrays["scale_w"], np.ones(m, dtype=np.float64))
    ):
        raise ValueError("GENIE-mode GxE feature-cache scales must be exactly one.")

    for name in ("variant_chr", "variant_snp", "variant_a1", "variant_a2"):
        if any(not str(value) for value in arrays[name]):
            raise ValueError(f"GxE feature-cache array {name!r} contains an empty value.")
    if len(set(arrays["variant_snp"].tolist())) != m:
        raise ValueError("GxE feature-cache SNP identifiers must be unique.")
    if np.any(arrays["variant_bp"] < 0):
        raise ValueError("GxE feature-cache base-pair positions must be non-negative.")

    labels = metadata.get("jackknife_labels")
    jackknife_ids = arrays["jackknife_ids"]
    if labels is None:
        if jackknife_ids.shape != (0,):
            raise ValueError("A feature cache without jackknife labels must have empty jackknife IDs.")
    else:
        if (
            not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("GxE feature-cache jackknife labels are invalid.")
        if jackknife_ids.shape != (m,):
            raise ValueError("GxE feature-cache jackknife IDs must align to all variants.")
        if np.any(jackknife_ids < 0) or np.any(jackknife_ids >= len(labels)):
            raise ValueError("GxE feature-cache jackknife IDs are out of range.")
        counts = np.bincount(jackknife_ids, minlength=len(labels))
        if np.any(counts == 0):
            raise ValueError("GxE feature-cache jackknife contains an empty block.")
        block_masses = np.zeros((len(labels), k), dtype=np.float64)
        np.add.at(block_masses, jackknife_ids, arrays["annotations"])
        if np.any(annotation_masses.reshape(1, -1) - block_masses <= 0.0):
            raise ValueError("GxE feature-cache jackknife deletion empties an annotation.")

    digest_fields = (
        "analysis_fingerprint", "variant_digest", "annotation_digest", "jackknife_digest",
    )
    for name in digest_fields:
        if not _is_canonical_sha256(metadata.get(name)):
            raise ValueError(f"GxE feature-cache digest {name!r} is invalid.")
    if metadata["variant_digest"] != _feature_cache_variant_digest(arrays):
        raise ValueError("GxE feature-cache variant digest is inconsistent with its arrays.")
    if metadata["annotation_digest"] != _feature_cache_annotation_digest(metadata, arrays):
        raise ValueError("GxE feature-cache annotation digest is inconsistent with its arrays.")
    if metadata["jackknife_digest"] != _feature_cache_jackknife_digest(metadata, arrays):
        raise ValueError("GxE feature-cache jackknife digest is inconsistent with its arrays.")

    environment = metadata.get("environment")
    covariates = metadata.get("covariates")
    if not isinstance(environment, str) or not environment:
        raise ValueError("GxE feature-cache environment name is invalid.")
    if (
        not isinstance(covariates, list)
        or any(not isinstance(name, str) or not name for name in covariates)
        or len(set(covariates)) != len(covariates)
        or environment in covariates
    ):
        raise ValueError("GxE feature-cache covariate names are invalid.")
    if p > len(covariates) + 1:
        raise ValueError("GxE feature-cache fixed-effect rank exceeds the declared design columns.")
    transform = metadata.get("environment_transform")
    required_transform = {
        "standardized", "raw_mean", "raw_sd", "ddof", "analysis_mean",
        "analysis_sum_squares", "units", "fixed_effect_design_sha256",
    }
    if not isinstance(transform, Mapping) or set(transform) != required_transform:
        raise ValueError("GxE feature-cache environment transform is incomplete or noncanonical.")
    if transform.get("standardized") is not True or transform.get("units") != "per_environment_sd":
        raise ValueError("GxE feature-cache environment must use the standardized analysis scale.")
    if _metadata_int(transform, "ddof") != ddof:
        raise ValueError("GxE feature-cache environment and genotype ddof declarations disagree.")
    _metadata_float(transform, "raw_mean")
    _metadata_float(transform, "raw_sd", positive=True)
    analysis_mean = _metadata_float(transform, "analysis_mean")
    analysis_ss = _metadata_float(transform, "analysis_sum_squares", positive=True)
    if abs(analysis_mean) > 1e-10:
        raise ValueError("GxE feature-cache standardized environment is not centered.")
    _require_close(
        "standardized environment sum of squares", analysis_ss, float(n - ddof),
    )
    if not _is_canonical_sha256(transform.get("fixed_effect_design_sha256")):
        raise ValueError("GxE feature-cache fixed-effect design digest is invalid.")

    genotype_files = metadata.get("genotype_files")
    if not isinstance(genotype_files, Mapping) or set(genotype_files) != {".bed", ".bim", ".fam"}:
        raise ValueError("GxE feature-cache genotype provenance is incomplete.")
    for extension, declaration in genotype_files.items():
        if not isinstance(declaration, Mapping) or set(declaration) != {"bytes", "sha256"}:
            raise ValueError(f"GxE feature-cache genotype provenance for {extension} is invalid.")
        size = declaration.get("bytes")
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) <= 0:
            raise ValueError(f"GxE feature-cache genotype byte count for {extension} is invalid.")
        if not _is_canonical_sha256(declaration.get("sha256")):
            raise ValueError(f"GxE feature-cache genotype digest for {extension} is invalid.")

    qdq = arrays["nxe_qdq"]
    terms = arrays["nxe_trace_terms"]
    qdq_scale = max(1.0, float(np.max(np.abs(qdq))))
    if float(np.max(np.abs(qdq - qdq.T))) > 2e-12 * qdq_scale:
        raise ValueError("GxE feature-cache nxe_qdq is not symmetric.")
    qdq_sym = 0.5 * (qdq + qdq.T)
    if float(np.linalg.eigvalsh(qdq_sym)[0]) < -2e-11 * qdq_scale:
        raise ValueError("GxE feature-cache nxe_qdq is not positive semidefinite.")
    sum_d, sum_d2, trace_qd2q = (float(value) for value in terms)
    if min(sum_d, sum_d2, trace_qd2q) < 0.0:
        raise ValueError("GxE feature-cache NxE trace sufficient statistics must be non-negative.")
    _require_close("NxE environment sum of squares", sum_d, analysis_ss)
    inequality_tol = 2e-10 * max(1.0, sum_d2, trace_qd2q)
    if sum_d2 + inequality_tol < (sum_d * sum_d) / float(n):
        raise ValueError("GxE feature-cache NxE moments violate scalar Cauchy-Schwarz.")
    qdq_sq_trace = float(np.sum(qdq * qdq.T))
    if trace_qd2q > sum_d2 + inequality_tol or qdq_sq_trace > trace_qd2q + inequality_tol:
        raise ValueError("GxE feature-cache NxE moments violate projection inequalities.")
    trace_nxe = sum_d - float(np.trace(qdq))
    trace_nxe_sq = sum_d2 - 2.0 * trace_qd2q + qdq_sq_trace
    nonnegative_tol = 2e-10 * max(1.0, sum_d, sum_d2)
    if trace_nxe < -nonnegative_tol or trace_nxe_sq < -nonnegative_tol:
        raise ValueError("GxE feature-cache sufficient statistics imply a negative NxE trace.")
    trace_nxe = max(0.0, trace_nxe)
    trace_nxe_sq = max(0.0, trace_nxe_sq)
    if trace_nxe_sq > trace_nxe * trace_nxe + nonnegative_tol:
        raise ValueError("GxE feature-cache NxE traces violate positive-semidefinite trace bounds.")
    _require_close("trace_nxe", _metadata_float(metadata, "trace_nxe"), trace_nxe)
    _require_close("trace_nxe_sq", _metadata_float(metadata, "trace_nxe_sq"), trace_nxe_sq)

    diagnostics = metadata.get("feature_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("GxE feature cache lacks feature diagnostics.")
    scalar_diagnostics = (
        "max_projection_leakage_additive", "max_projection_leakage_interaction",
        "min_norm_additive_over_rank", "max_norm_additive_over_rank",
        "min_norm_interaction_over_rank", "max_norm_interaction_over_rank",
        "max_norm_error_additive", "max_norm_error_interaction",
        "max_trace_error_additive", "max_trace_error_interaction",
    )
    for name in scalar_diagnostics:
        if _metadata_float(diagnostics, name) < 0.0:
            raise ValueError(
                f"GxE feature-cache diagnostic {name!r} must be non-negative."
            )
    if (
        _metadata_int(diagnostics, "valid_additive_columns", minimum=0) != m
        or _metadata_int(diagnostics, "valid_interaction_columns", minimum=0) != m
    ):
        raise ValueError("GxE feature-cache valid-column diagnostics are inconsistent.")
    if min(
        float(diagnostics["max_projection_leakage_additive"]),
        float(diagnostics["max_projection_leakage_interaction"]),
    ) < 0.0 or max(
        float(diagnostics["max_projection_leakage_additive"]),
        float(diagnostics["max_projection_leakage_interaction"]),
    ) > 1.0e-9:
        raise ValueError("GxE feature-cache diagnostics report fixed-effect projection leakage.")
    expected_diagnostics = {
        "min_norm_additive_over_rank": float(np.min(arrays["norm_x"])),
        "max_norm_additive_over_rank": float(np.max(arrays["norm_x"])),
        "min_norm_interaction_over_rank": float(np.min(arrays["norm_w"])),
        "max_norm_interaction_over_rank": float(np.max(arrays["norm_w"])),
        "max_norm_error_additive": float(np.max(np.abs(arrays["norm_x"] - 1.0))),
        "max_norm_error_interaction": float(np.max(np.abs(arrays["norm_w"] - 1.0))),
    }
    for name, expected in expected_diagnostics.items():
        _require_close(name, float(diagnostics[name]), expected)
    trace_x = residual_rank * (arrays["annotations"].T @ arrays["norm_x"]) / annotation_masses
    trace_w = residual_rank * (arrays["annotations"].T @ arrays["norm_w"]) / annotation_masses
    diagnostic_trace_x = diagnostics.get("kernel_traces_additive")
    diagnostic_trace_w = diagnostics.get("kernel_traces_interaction")
    if not isinstance(diagnostic_trace_x, list) or not isinstance(diagnostic_trace_w, list):
        raise ValueError("GxE feature-cache kernel-trace diagnostics are invalid.")
    try:
        diagnostic_trace_x = np.asarray(diagnostic_trace_x, dtype=np.float64)
        diagnostic_trace_w = np.asarray(diagnostic_trace_w, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("GxE feature-cache kernel-trace diagnostics must be numeric.") from exc
    if (
        diagnostic_trace_x.shape != (k,)
        or diagnostic_trace_w.shape != (k,)
        or not np.all(np.isfinite(diagnostic_trace_x))
        or not np.all(np.isfinite(diagnostic_trace_w))
    ):
        raise ValueError(
            f"GxE feature-cache kernel-trace diagnostics must each have shape ({k},)."
        )
    _require_close("additive kernel-trace diagnostics", diagnostic_trace_x, trace_x)
    _require_close("interaction kernel-trace diagnostics", diagnostic_trace_w, trace_w)
    _require_close(
        "max additive trace error", diagnostics["max_trace_error_additive"],
        float(np.max(np.abs(trace_x - residual_rank))),
    )
    _require_close(
        "max interaction trace error", diagnostics["max_trace_error_interaction"],
        float(np.max(np.abs(trace_w - residual_rank))),
    )

    if expected_identity is not None:
        mismatches = []
        for key, expected_value in expected_identity.items():
            observed_value = metadata.get(key)
            if key == "environment_transform":
                matches = _environment_transforms_equal(observed_value, expected_value)
            else:
                matches = observed_value == expected_value
            if not matches:
                mismatches.append(key)
        if mismatches:
            raise ValueError(
                "GxE feature cache does not match the current genotype/design/annotation/mode/jackknife "
                f"configuration; mismatched fields: {mismatches}."
            )
    if expected_arrays is not None:
        for name, expected in expected_arrays.items():
            if name not in arrays or not np.array_equal(arrays[name], expected):
                raise ValueError(f"GxE feature-cache canonical array {name!r} is inconsistent.")



def _round_up_to(x: int, gran: int) -> int:
    return int(((x + gran - 1) // gran) * gran)



def _build_balanced_vtiles(V: int, vmax: int, gran: int = 64, max_tiles: int = 8) -> list[tuple[int, int]]:
    V = int(V)
    vmax = int(vmax)
    gran = int(gran)
    max_tiles = int(max_tiles)
    if V <= 0:
        return []
    if vmax <= 0 or gran <= 0 or max_tiles <= 0:
        raise ValueError("vmax, gran, and max_tiles must be positive integers.")
    if vmax >= V:
        return [(0, V)]
    tile_count = math.ceil(V / vmax)
    if tile_count > max_tiles:
        raise RuntimeError(
            "The configured probe-memory/scratch ceilings require "
            f"{tile_count} tiles, exceeding max_tiles={max_tiles}; "
            "increase the relevant ceiling or reduce the probe count."
        )

    # Balance the required tile count to minimize peak memory.  Prefer a
    # granularity-aligned width only when it remains within the hard ceiling;
    # otherwise exact balancing still guarantees every tile is <= vmax.
    balanced_width = math.ceil(V / tile_count)
    aligned_width = _round_up_to(balanced_width, gran)
    width = aligned_width if aligned_width <= vmax else balanced_width
    tiles = [
        (start, min(width, V - start)) for start in range(0, V, width)
    ]
    if len(tiles) > max_tiles or max(size for _, size in tiles) > vmax:
        raise RuntimeError("Internal GxE probe tiling failed its hard resource caps.")
    return tiles



def _mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF



def _make_seed(root: int, block: int, probe_id: int) -> int:
    s = 0x1234ABCD
    s ^= _mix64(int(root))
    s ^= _mix64(int(block) + 0x9E37)
    s ^= _mix64(int(probe_id) + 0x85EB)
    return int(s & 0xFFFFFFFFFFFFFFFF)



def _orthonormalize_columns(X: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    if X.shape[1] == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64, order="F")
    # This is a tall-skinny fixed-effect problem (typically N x <30), not a
    # throughput bottleneck.  Running its LAPACK reduction on one thread avoids
    # exposing the analysis-defining basis to the rare wide threaded-BLAS
    # corruption observed on tabla; large genotype GEMMs remain multithreaded.
    with threadpool_limits(limits=1, user_api="blas"):
        U, singular, _ = np.linalg.svd(
            np.asarray(X, dtype=np.float64), full_matrices=False
        )
    if singular.size == 0 or singular[0] <= 0.0:
        return np.empty((X.shape[0], 0), dtype=np.float64, order="F")
    rank_tol = max(float(tol), max(X.shape) * np.finfo(np.float64).eps * float(singular[0]))
    rank = int(np.sum(singular > rank_tol))
    if rank == 0:
        return np.empty((X.shape[0], 0), dtype=np.float64, order="F")
    return np.asfortranarray(U[:, :rank])


def _stable_center_and_scale(
    values: np.ndarray, *, ddof: int
) -> tuple[np.ndarray, float, float]:
    """Center and scale one finite vector with deterministic reductions.

    NumPy/pandas reduction order can change across versions and CPU builds.
    The fixed-effect design is cryptographically sealed and must therefore be
    byte-identical when the same text inputs are read on another machine.
    ``math.fsum`` fixes the scalar reduction order; the elementwise subtract
    and divide then have deterministic IEEE-754 results.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size <= int(ddof) or not np.all(np.isfinite(array)):
        raise ValueError("Stable standardization requires a finite non-empty vector.")
    mean = math.fsum(float(value) for value in array) / array.size
    centered = np.asarray(array - mean, dtype=np.float64)
    sum_squares = math.fsum(float(value) * float(value) for value in centered)
    scale = math.sqrt(sum_squares / (array.size - int(ddof)))
    return centered, float(mean), float(scale)



def read_env_and_cov(
    env_filename: str,
    fam_filename: str | None,
    cov_filename: str | None = None,
    std: bool = True,
    cov_impute_method: str = "ignore",
    logger=None,
    verbose: bool = False,
    sample_idx=None,
    ddof: int = 1,
    pheno_filename: str | None = None,
    pheno_col: str | None = None,
    missing_values: Sequence[str] = ("-9", "NA", "NaN", "nan", ".", "None", "null"),
    sample_ids: pd.DataFrame | None = None,
    env_col: str | None = None,
):
    id_types = {"FID": str, "IID": str}
    if sample_ids is None:
        if fam_filename is None:
            raise ValueError("fam_filename or sample_ids must be provided.")
        fam = read_fam_sample_ids(fam_filename)
    else:
        fam = pd.DataFrame(sample_ids)[["FID", "IID"]].copy()
        fam[["FID", "IID"]] = fam[["FID", "IID"]].astype(str)
    if fam.duplicated(subset=["FID", "IID"]).any():
        raise ValueError("Genotype sample metadata contain duplicate FID/IID pairs.")
    if sample_idx is not None:
        sample_idx = np.asarray(sample_idx, dtype=int)
        fam = fam.iloc[sample_idx].reset_index(drop=True)

    env = pd.read_csv(
        env_filename,
        sep=r"\s+",
        dtype=id_types,
        na_values=list(missing_values),
        keep_default_na=True,
    )
    if "FID" not in env.columns or "IID" not in env.columns:
        raise ValueError("Environment file must contain FID and IID columns.")
    if env.duplicated(subset=["FID", "IID"]).any():
        raise ValueError("Environment file contains duplicate FID/IID rows.")
    env_cols = [c for c in env.columns if c not in ("FID", "IID")]
    if env_col is None and len(env_cols) != 1:
        raise ValueError(
            "Environment file must contain exactly one environment column in addition to FID and IID. "
            f"Found {len(env_cols)} column(s): {env_cols}."
        )
    if env_col is None:
        env_name = env_cols[0]
    else:
        env_name = str(env_col)
        if env_name not in env_cols:
            raise ValueError(
                f"Environment column {env_name!r} is not present in {env_filename}; "
                f"available columns are {env_cols}."
            )

    merged = fam.merge(env[["FID", "IID", env_name]], on=["FID", "IID"], how="left", indicator=True)
    n_missing_env = int((merged["_merge"] != "both").sum())
    if n_missing_env:
        raise ValueError(f"{n_missing_env} .fam samples not found in environment file (FID/IID mismatch).")
    merged.drop(columns=["_merge"], inplace=True)

    cov_cols: list[str] = []
    if cov_filename is not None:
        cov = pd.read_csv(
            cov_filename,
            sep=r"\s+",
            dtype=id_types,
            na_values=list(missing_values),
            keep_default_na=True,
        )
        if "FID" not in cov.columns or "IID" not in cov.columns:
            raise ValueError("Covariate file must contain FID and IID columns.")
        if cov.duplicated(subset=["FID", "IID"]).any():
            raise ValueError("Covariate file contains duplicate FID/IID rows.")
        overlap = set(cov.columns) & {env_name}
        if overlap:
            raise ValueError(
                f"Covariate file repeats the environment column {env_name!r}; "
                "include each fixed effect only once."
            )
        merged = merged.merge(cov, on=["FID", "IID"], how="left", indicator=True)
        n_missing_cov = int((merged["_merge"] != "both").sum())
        if n_missing_cov:
            raise ValueError(f"{n_missing_cov} .fam samples not found in covariate file (FID/IID mismatch).")
        merged.drop(columns=["_merge"], inplace=True)
        cov_cols = [c for c in merged.columns if c not in ("FID", "IID", env_name)]

    phenotype_name = None
    if pheno_filename is not None:
        pheno = pd.read_csv(
            pheno_filename,
            sep=r"\s+",
            dtype=id_types,
            na_values=list(missing_values),
            keep_default_na=True,
        )
        if "FID" not in pheno.columns or "IID" not in pheno.columns:
            raise ValueError("Phenotype file must contain FID and IID columns.")
        if pheno.duplicated(subset=["FID", "IID"]).any():
            raise ValueError("Phenotype file contains duplicate FID/IID rows.")
        available = [c for c in pheno.columns if c not in ("FID", "IID")]
        if pheno_col is None:
            if len(available) != 1:
                raise ValueError(
                    "Phenotype file must contain exactly one value column unless --gxe-pheno-col is supplied; "
                    f"found {available}."
                )
            phenotype_name = available[0]
        else:
            phenotype_name = str(pheno_col)
            if phenotype_name not in available:
                raise ValueError(
                    f"Phenotype column {phenotype_name!r} is not present in {pheno_filename}; "
                    f"available columns are {available}."
                )
        if phenotype_name in merged.columns:
            raise ValueError(
                f"Phenotype column {phenotype_name!r} is already present as an environment/covariate; "
                "the response cannot also be a fixed effect."
            )
        merged = merged.merge(
            pheno[["FID", "IID", phenotype_name]],
            on=["FID", "IID"],
            how="left",
            indicator=True,
        )
        n_missing_pheno_ids = int((merged["_merge"] != "both").sum())
        if n_missing_pheno_ids:
            raise ValueError(
                f"{n_missing_pheno_ids} .fam samples not found in phenotype file (FID/IID mismatch)."
            )
        merged.drop(columns=["_merge"], inplace=True)

    merged[env_name] = pd.to_numeric(merged[env_name], errors="coerce")
    for c in cov_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if phenotype_name is not None:
        merged[phenotype_name] = pd.to_numeric(merged[phenotype_name], errors="coerce")

    if cov_impute_method == "ignore":
        keep_mask = merged[env_name].notna()
        if cov_cols:
            keep_mask &= ~merged[cov_cols].isna().any(axis=1)
        if phenotype_name is not None:
            keep_mask &= merged[phenotype_name].notna()
    else:
        if cov_cols:
            for column in cov_cols:
                observed = merged[column].dropna().to_numpy(dtype=np.float64)
                if observed.size == 0:
                    continue
                fill_value = math.fsum(float(value) for value in observed) / observed.size
                merged[column] = merged[column].fillna(fill_value)
        keep_mask = merged[env_name].notna()

    dropped = int((~keep_mask).sum())
    if logger is not None:
        logger._log(f"[env] Dropping {dropped} samples due to missing environment/covariates.")

    merged = merged.loc[keep_mask].reset_index(drop=True)
    if merged.shape[0] == 0:
        raise ValueError("After filtering, no samples remain for the environment-specific LD-score calculation.")

    env_vec, env_mean, env_std = _stable_center_and_scale(
        merged[env_name].to_numpy(dtype=np.float64), ddof=ddof
    )
    if not np.isfinite(env_std) or env_std <= 0.0:
        raise ValueError(f"Environment '{env_name}' has zero or invalid variance after filtering.")
    if std:
        env_vec = env_vec / env_std
    env_transform = {
        "standardized": bool(std),
        "raw_mean": env_mean,
        "raw_sd": env_std,
        "ddof": int(ddof),
        # BLAS may reduce np.dot() in a thread-count-dependent order.  These
        # provenance diagnostics must be deterministic across thread counts.
        "analysis_mean": float(
            math.fsum(float(value) for value in env_vec) / env_vec.size
        ),
        "analysis_sum_squares": float(
            math.fsum(float(value) * float(value) for value in env_vec)
        ),
        "units": "per_environment_sd" if std else "input_units",
    }

    cov_base = np.empty((merged.shape[0], 0), dtype=np.float64)
    kept_cov_cols: list[str] = []
    if cov_cols:
        cov_df = merged[cov_cols].copy()
        centered_covariates: dict[str, np.ndarray] = {}
        drop_cols: list[str] = []
        for column in cov_df.columns:
            centered, _, scale = _stable_center_and_scale(
                cov_df[column].to_numpy(dtype=np.float64), ddof=ddof
            )
            if not np.isfinite(scale) or scale <= 0.0:
                drop_cols.append(str(column))
            else:
                centered_covariates[str(column)] = centered / scale
        if drop_cols:
            if logger is not None:
                logger._log(
                    f"[env] Dropping {len(drop_cols)} constant covariates: "
                    f"{drop_cols[:10]}{'...' if len(drop_cols) > 10 else ''}"
                )
            cov_df.drop(columns=drop_cols, inplace=True)
        if not cov_df.empty and std:
            cov_df = pd.DataFrame(
                {
                    str(column): centered_covariates[str(column)]
                    for column in cov_df.columns
                },
                index=cov_df.index,
            )
        if not cov_df.empty:
            kept_cov_cols = list(cov_df.columns)
            cov_base = cov_df.to_numpy(dtype=np.float64, copy=False)

    # Interaction / GWIS covariate space: user covariates + environment main effect.
    # We use the same space for the additive X side in the XW cross-score so that the
    # resulting summary objects match the score-scale derivation.
    design_base = np.column_stack([cov_base, env_vec.reshape(-1, 1)])
    design_digest = hashlib.sha256()
    for name in [*kept_cov_cols, str(env_name)]:
        design_digest.update(str(name).encode("utf-8"))
        design_digest.update(b"\n")
    design_digest.update(
        np.asarray(design_base, dtype="<f8", order="C").tobytes(order="C")
    )
    env_transform["fixed_effect_design_sha256"] = design_digest.hexdigest()
    C = _orthonormalize_columns(design_base)
    R = np.asfortranarray(C.T)

    pheno_vec = None
    phenotype_residual_fraction = None
    if phenotype_name is not None:
        # pandas 3 may expose a read-only zero-copy array.  Projection and
        # normalization below are intentionally in place, so request an
        # explicitly writable buffer at this mutation boundary.
        pheno_vec = merged[phenotype_name].to_numpy(dtype=np.float64, copy=True)
        raw_centered = pheno_vec - float(pheno_vec.mean())
        raw_ss = float(np.dot(raw_centered, raw_centered))
        if C.shape[1] > 0:
            pheno_vec -= C @ (C.T @ pheno_vec)
        pheno_vec -= float(pheno_vec.mean())
        residual_rank = merged.shape[0] - C.shape[1] - 1
        ss = float(np.dot(pheno_vec, pheno_vec))
        min_ss = max(np.finfo(np.float64).tiny, np.finfo(np.float64).eps * max(raw_ss, 1.0))
        if residual_rank <= 0 or not np.isfinite(ss) or ss <= min_ss:
            raise ValueError("Phenotype has zero or invalid residual variance after fixed-effect projection.")
        phenotype_residual_fraction = float(ss / raw_ss)
        pheno_vec *= math.sqrt(float(residual_rank) / ss)

    km = np.flatnonzero(keep_mask.to_numpy())
    if sample_idx is not None:
        keep_idx_global = np.asarray(sample_idx, dtype=int)[km]
    else:
        keep_idx_global = km

    if logger is not None:
        logger._log(
            f"[env] Read {env_filename}: kept {C.shape[0]} samples; environment='{env_name}'; "
            f"effective covariate rank={C.shape[1]} ({len(kept_cov_cols)} user covariate(s) + environment main effect)."
        )

    return (
        np.asarray(env_vec, dtype=np.float64),
        str(env_name),
        np.asfortranarray(C),
        np.asfortranarray(R),
        np.asarray(keep_idx_global, dtype=int),
        kept_cov_cols,
        None if pheno_vec is None else np.asarray(pheno_vec, dtype=np.float64),
        phenotype_name,
        phenotype_residual_fraction,
        env_transform,
    )


class GenomewideEnvLDScore:
    r"""
    Estimate the four directional additive/interaction trace panels required
    by one-environment projected-kernel normal equations using randomized
    sketches.

    For each annotation bin k, the module estimates
        ell^{WW}_{jk} = sum_{j' in S_k} (r^{WW}_{jj'})^2,
        ell^{XW}_{jk} = sum_{j' in S_k} (r^{XW}_{jj'})^2,
    X and W are projected onto the same fixed-effect residual space, including
    the environment main effect.  The default ``standardized`` SUMMIT mode
    rescales every valid projected column to squared norm ``rank(P)``.  The
    explicit ``genie`` compatibility mode retains each projected column's
    natural norm after HWE scaling.

    Output files:
        {out}.gxx.ldscore.gz   -> XX / additive-additive trace scores
        {out}.gxe.ldscore.gz   -> XW / additive-interaction trace scores
        {out}.exg.ldscore.gz   -> WX / interaction-additive trace scores
        {out}.gee.ldscore.gz   -> WW / interaction-interaction trace scores
    """

    def __init__(
        self,
        bed_path,
        env_path,
        annot_path,
        out_path,
        log,
        rand_dist,
        low_level,
        covar_path=None,
        num_vecs=10,
        step_size=1000,
        seed=None,
        verbose=False,
        dtype="float32",
        num_threads: int | None = None,
        eps_var: float = 1e-10,
        rand_samp=None,
        ddof=1,
        target_xz_mem="auto",
        target_mem=None,
        device="cpu",
        impute_method: str = "mean",
        kernel_mode: str = "standardized",
        genotype_scale: str | None = None,
        pheno_path: str | None = None,
        pheno_col: str | None = None,
        missing_values: Sequence[str] = ("-9", "NA", "NaN", "nan", ".", "None", "null"),
        write_jackknife: bool = False,
        jackknife_spec: str = "100",
        overwrite: bool = False,
        allow_low_probe_jackknife: bool = False,
        probe_offset: int = 0,
        feature_cache_path: str | None = None,
        shard_mode: bool = False,
        native_backend: str = "python",
        native_workspace_gib: float = 16.0,
        native_target_panel_columns: int = 64,
        env_col: str | None = None,
    ):
        self.eps_var = float(eps_var)
        if not np.isfinite(self.eps_var) or self.eps_var <= 0.0:
            raise ValueError(f"eps_var must be positive and finite; got {self.eps_var!r}.")
        self.genotype_input = resolve_genotype_input(bed_path)
        self.genotype_format = self.genotype_input.format
        self.genotype_prefix = self.genotype_input.prefix
        self.bed_prefix = self.genotype_prefix if self.genotype_format == "bed" else None
        self.env_path = str(env_path)
        self.covar_path = covar_path
        self._pgen_reader: PgenBlockReader | None = None
        proc_fds = Path("/proc/self/fd")
        if not proc_fds.is_dir():
            raise RuntimeError(
                "Stable zero-copy GxE genotype input requires Linux /proc/self/fd."
            )
        if self.genotype_format == "bed":
            source_paths = {
                ".bed": self.genotype_input.genotype_path,
                ".bim": self.genotype_input.variant_path,
                ".fam": self.genotype_input.sample_path,
            }
        else:
            source_paths = {
                ".pgen": self.genotype_input.genotype_path,
                ".pvar": self.genotype_input.variant_path,
                ".psam": self.genotype_input.sample_path,
            }
        self._genotype_extensions = tuple(source_paths)
        genotype_descriptors: dict[str, int] = {}
        try:
            for extension, source in source_paths.items():
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(source, flags)
                observed = os.fstat(descriptor)
                if not stat.S_ISREG(observed.st_mode):
                    os.close(descriptor)
                    raise ValueError(
                        f"PLINK input must be a regular non-symlink file: {source}."
                    )
                genotype_descriptors[extension] = descriptor
        except Exception:
            _close_file_descriptors(tuple(genotype_descriptors.values()))
            raise
        self._genotype_descriptors = genotype_descriptors
        self._stable_genotype_paths = {
            extension: Path(f"/proc/self/fd/{descriptor}")
            for extension, descriptor in genotype_descriptors.items()
        }
        self._descriptor_finalizer = weakref.finalize(
            self,
            _close_file_descriptors,
            tuple(genotype_descriptors.values()),
        )
        self.fam_path = (
            str(self._stable_genotype_paths[".fam"])
            if self.genotype_format == "bed" else None
        )
        self.bim_path = (
            str(self._stable_genotype_paths[".bim"])
            if self.genotype_format == "bed" else None
        )
        self.pgen_path = (
            str(self._stable_genotype_paths[".pgen"])
            if self.genotype_format == "pgen" else None
        )
        self.pvar_path = (
            str(self._stable_genotype_paths[".pvar"])
            if self.genotype_format == "pgen" else None
        )
        self.psam_path = (
            str(self._stable_genotype_paths[".psam"])
            if self.genotype_format == "pgen" else None
        )
        self._construction_genotype_state = self._capture_genotype_file_state()

        requested_reader_threads = int(num_threads) if num_threads is not None and int(num_threads) > 0 else None
        if self.genotype_format == "bed":
            expected_n, expected_m = _validate_plink_file_paths(
                self._stable_genotype_paths
            )
            self.G = open_bed(
                str(self._stable_genotype_paths[".bed"]),
                fam_filepath=str(self._stable_genotype_paths[".fam"]),
                bim_filepath=str(self._stable_genotype_paths[".bim"]),
                num_threads=requested_reader_threads,
            )
            self.nsamp_total, self.nsnps = self.G.shape
            if (self.nsamp_total, self.nsnps) != (expected_n, expected_m):
                raise RuntimeError(
                    "bed-reader dimensions disagree with validated FAM/BIM dimensions: "
                    f"reader={self.G.shape}, FAM/BIM={(expected_n, expected_m)}."
                )
            self.sample_ids = read_fam_sample_ids(self.fam_path)
            self.snplist = None
        else:
            self.G = None
            self.sample_ids = read_psam_sample_ids(self.psam_path)
            self.snplist = validate_variant_metadata(
                read_pvar_variants(self.pvar_path), source=f"PVAR '{self.genotype_prefix}.pvar'"
            )
            self.nsamp_total = int(len(self.sample_ids))
            self.nsnps = int(len(self.snplist))
        self.nvecs = int(num_vecs)
        self.step_size = int(step_size)
        if self.nvecs <= 0:
            raise ValueError("num_vecs must be positive.")
        if self.nvecs > 2**64:
            raise ValueError("num_vecs cannot exceed the uint64 probe-identity space.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")
        self.log = log
        self.verbose = bool(verbose)
        self.rand_dist = str(rand_dist).strip().lower()
        if self.rand_dist not in ("rademacher", "gaussian", "normal", "spherical"):
            raise ValueError(f"Unsupported rand_dist: {rand_dist!r}.")
        self.ddof = int(ddof)
        if self.ddof not in (0, 1):
            raise ValueError(f"GxE generation supports ddof 0 or 1; got {self.ddof}.")
        self.target_mem = target_mem
        requested_memory = target_xz_mem if target_mem is None else target_mem
        self.target_xz_mem, self.memory_budget = utils.resolve_memory_budget_gib(
            requested_memory
        )
        self.log._log(
            f"[memory] sketch budget={self.target_xz_mem:.3f} GiB "
            f"(mode={self.memory_budget['mode']})."
        )
        self.device = str(device).strip().lower()
        self.impute_method = str(impute_method).strip().lower()
        if self.impute_method != "mean":
            raise ValueError("The current Python GxE LD-score implementation supports only impute_method='mean'.")
        self.kernel_mode = str(kernel_mode).strip().lower()
        if self.kernel_mode not in ("genie", "standardized"):
            raise ValueError("kernel_mode must be 'genie' or 'standardized'.")
        if genotype_scale is None:
            genotype_scale = "sample" if self.kernel_mode == "standardized" else "hwe"
        self.genotype_scale = str(genotype_scale).strip().lower()
        if self.genotype_scale not in ("hwe", "sample"):
            raise ValueError("genotype_scale must be 'hwe' or 'sample'.")
        self.pheno_path = None if pheno_path is None else str(pheno_path)
        self.pheno_col = pheno_col
        self.missing_values = tuple(str(x) for x in missing_values)
        self.write_jackknife = bool(write_jackknife)
        self.jackknife_spec = str(jackknife_spec)
        self.overwrite = bool(overwrite)
        self.allow_low_probe_jackknife = bool(allow_low_probe_jackknife)
        self.probe_offset = int(probe_offset)
        if self.probe_offset < 0 or self.probe_offset >= 2**64:
            raise ValueError(
                f"probe_offset must be in [0, 2**64); got {self.probe_offset}."
            )
        if self.probe_offset + self.nvecs > 2**64:
            raise ValueError("The requested GxE probe interval exceeds the uint64 identity space.")
        self.feature_cache_path = (
            None if feature_cache_path is None else str(Path(feature_cache_path).expanduser().resolve())
        )
        self.feature_cache_sha256: str | None = None
        self.feature_cache_metadata: dict | None = None
        self.feature_backend_provenance: dict | None = None
        self.shard_mode = bool(shard_mode)
        if self.shard_mode and self.feature_cache_path is None:
            raise ValueError("GxE reference shards require a precomputed feature_cache_path.")
        if self.shard_mode and self.pheno_path is not None:
            raise ValueError("GxE reference shards must be phenotype-free.")
        if self.feature_cache_path is not None and self.pheno_path is not None:
            raise ValueError("Feature-cache trace generation is phenotype-free; score phenotypes separately.")
        requested_native_backend = str(native_backend).strip().lower()
        if requested_native_backend not in ("python", "direct"):
            raise ValueError("native_backend must be 'python' or 'direct'.")
        if self.genotype_format == "pgen":
            unsupported = []
            if requested_native_backend != "python":
                unsupported.append("native-direct execution")
            if self.feature_cache_path is not None:
                unsupported.append("feature-cache execution")
            if self.shard_mode:
                unsupported.append("reference shards")
            if self.genotype_scale != "sample":
                unsupported.append(f"genotype_scale={self.genotype_scale}")
            if unsupported:
                raise ValueError(
                    "PGEN GxE input currently supports the ordinary Python monolithic "
                    "sample-standardized path only; unsupported: " + ", ".join(unsupported) + "."
                )
        if self.device != "cpu":
            self.log._log(f"[gxe] device='{device}' requested, but this Python implementation is CPU-only. Falling back to CPU.")
            self.device = "cpu"

        if seed is None:
            self.root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            self.log._log(f"[seed] No seed provided; using generated root seed {self.root_seed}")
        else:
            self.root_seed = int(seed)
        if self.root_seed < 0 or self.root_seed >= 2**64:
            raise ValueError(f"seed must be in [0, 2**64); got {self.root_seed}.")
        self.rng = np.random.default_rng(self.root_seed)

        if low_level is not None:
            try:
                # Keep the pure-Python GxE module importable without the native
                # GWLD extension; the low-level helper is needed only here.
                from .gw_ldscore import apply_env as _apply_env

                actual_threads = _apply_env(low_level)
            except Exception as e:
                self.log._log(f"[threads] apply_env failed in GxE LD-score setup (non-fatal): {e}")
                actual_threads = None
        else:
            actual_threads = None
        if num_threads is not None and int(num_threads) > 0:
            try:
                affinity_threads = len(os.sched_getaffinity(0))
            except Exception:
                affinity_threads = max(1, os.cpu_count() or 1)
            self.num_threads = max(1, min(int(num_threads), affinity_threads))
        elif actual_threads is not None:
            self.num_threads = int(actual_threads)
        else:
            self.num_threads = max(1, os.cpu_count() or 1)
        decode_threads = os.environ.get("SUMMIT_DECODE_THREADS")
        if decode_threads is not None and decode_threads.isdigit():
            self.decode_threads = max(1, min(self.num_threads, int(decode_threads)))
        else:
            self.decode_threads = self.num_threads

        if dtype in (np.float32, "float32", "f4"):
            self.dtype = np.float32
        elif dtype in (np.float64, "float64", "f8"):
            self.dtype = np.float64
        else:
            raise ValueError(f"GxE dtype must be float32 or float64; got {dtype!r}.")
        self.start_time = utils._get_time()
        self.log._log("Genome-wide GxE LD-score calculation started at: " + utils._get_timestr(self.start_time))

        base_idx = np.arange(self.nsamp_total, dtype=int)
        sel_idx = None
        if rand_samp is not None:
            if isinstance(rand_samp, (float, np.floating)):
                if not (0.0 < rand_samp <= 1.0):
                    raise ValueError("--rand-samp float must be in (0,1].")
                k = int(np.floor(float(rand_samp) * self.nsamp_total))
                k = max(1, min(k, self.nsamp_total))
            else:
                k = int(rand_samp)
                if not (100 <= k <= self.nsamp_total):
                    raise ValueError("--rand-samp int must be in [100, N].")
            sel_idx = np.sort(self.rng.choice(base_idx, size=k, replace=False))
            self.log._log(f"Randomly subsampling individuals: {k}/{self.nsamp_total} ({k / self.nsamp_total:.1%})")

        if self.genotype_format == "bed":
            self._read_bim(self.bim_path)
        else:
            self.log._log(f"Reading {self.genotype_prefix}.pvar for variants")
        self._read_annot(annot_path)
        self.jackknife_ids, self.jackknife_labels = self._build_jackknife_blocks(
            self.jackknife_spec if self.write_jackknife else None
        )

        (
            env_vec,
            env_name,
            C_int,
            cov_R_int,
            keep_idx_global,
            cov_cols,
            pheno_vec,
            phenotype_name,
            phenotype_residual_fraction,
            env_transform,
        ) = read_env_and_cov(
            env_filename=self.env_path,
            fam_filename=self.fam_path,
            cov_filename=self.covar_path,
            std=True,
            cov_impute_method="ignore",
            logger=self.log,
            verbose=self.verbose,
            sample_idx=sel_idx if sel_idx is not None else None,
            ddof=self.ddof,
            pheno_filename=self.pheno_path,
            pheno_col=self.pheno_col,
            missing_values=self.missing_values,
            sample_ids=self.sample_ids,
            env_col=env_col,
        )
        self.row_sel = np.asarray(keep_idx_global, dtype=int)
        self.env = np.asarray(env_vec, dtype=np.float64)
        self.env_name = env_name
        self.C_int = np.asarray(C_int, dtype=np.float64, order="F")
        self.cov_R_int = np.asarray(cov_R_int, dtype=np.float64, order="F")
        self.cov_cols = list(cov_cols)
        self.pheno = pheno_vec
        self.phenotype_name = phenotype_name
        self.phenotype_residual_fraction = phenotype_residual_fraction
        self.environment_transform = dict(env_transform)

        self.nsamp = int(self.row_sel.shape[0])
        self.p_eff = int(self.C_int.shape[1])
        self.N_eff = self.nsamp - self.p_eff
        self.df_corr = self.N_eff - 1
        if self.df_corr <= 0:
            raise ValueError(
                f"Residual correlation degrees of freedom are non-positive: nsamp={self.nsamp}, "
                f"p_eff={self.p_eff}, df={self.df_corr}."
            )

        self.outpath = out_path
        self.inv_sqrt_resvar_x_all: np.ndarray | None = None
        self.inv_sqrt_resvar_w_all: np.ndarray | None = None
        self.norm_x_all: np.ndarray | None = None
        self.norm_w_all: np.ndarray | None = None
        self.diag_nxe_x_all: np.ndarray | None = None
        self.diag_nxe_w_all: np.ndarray | None = None
        self.corr_xw_all: np.ndarray | None = None
        self.score_x_all: np.ndarray | None = None
        self.score_w_all: np.ndarray | None = None
        self.feature_diagnostics: dict[str, float | int | list[float]] = {}
        self.resource_estimates: dict[str, float | int] = {}
        self._vtiles_used: list[tuple[int, int]] | None = None
        self.native_workspace_gib = float(native_workspace_gib)
        if not np.isfinite(self.native_workspace_gib) or self.native_workspace_gib <= 0.0:
            raise ValueError("native_workspace_gib must be positive and finite.")
        self.native_target_panel_columns = int(native_target_panel_columns)
        if self.native_target_panel_columns <= 0:
            raise ValueError("native_target_panel_columns must be positive.")
        self.native_backend = requested_native_backend
        self._native_context = None
        self._native_binary_descriptor: int | None = None
        self._native_binary_record: dict | None = None
        self._native_build_info: dict | None = None
        self.native_strict_feature_moment_verification = False
        self.native_feature_moment_integrity_reason = "Python backend"
        self.native_phase_timings: dict[str, float] = {}
        if self.native_backend != "python":
            unsupported = []
            if self.nbins != 1:
                unsupported.append(f"K={self.nbins} annotations")
            if self.kernel_mode != "standardized":
                unsupported.append(f"kernel_mode={self.kernel_mode}")
            if self.genotype_scale != "sample":
                unsupported.append(f"genotype_scale={self.genotype_scale}")
            if self.impute_method != "mean":
                unsupported.append(f"impute_method={self.impute_method}")
            if self.pheno is not None:
                unsupported.append("phenotype-coupled construction")
            if unsupported:
                message = (
                    "The bounded native GxE backend supports only phenotype-free K=1, "
                    "standardized/sample, mean-imputed construction; got "
                    + ", ".join(unsupported)
                    + "."
                )
                raise ValueError(message)
            else:
                try:
                    from .. import gxeldcore as _gxeldcore
                except Exception as exc:
                    raise RuntimeError(
                        "The requested bounded native GxE extension is unavailable."
                    ) from exc
                else:
                    native_descriptor, native_record = _loaded_native_binary_record(
                        _gxeldcore
                    )
                    build_info = dict(_gxeldcore.build_info())
                    source_commit = build_info.get("source_commit")
                    source_tree_sha256 = build_info.get("source_tree_sha256")
                    if not (
                        isinstance(source_commit, str)
                        and len(source_commit) == 40
                        and all(
                            character in "0123456789abcdef"
                            for character in source_commit
                        )
                        and _is_canonical_sha256(source_tree_sha256)
                    ):
                        os.close(native_descriptor)
                        raise RuntimeError(
                            "The direct GxE extension was not built with exact source "
                            "commit and source-snapshot provenance."
                        )
                    self._native_binary_descriptor = native_descriptor
                    self._native_binary_record = native_record
                    self._native_build_info = build_info
                    (
                        self.native_strict_feature_moment_verification,
                        self.native_feature_moment_integrity_reason,
                    ) = _native_strict_feature_moment_verification_policy(build_info)
                    self._native_binary_finalizer = weakref.finalize(
                        self, _close_file_descriptors, (native_descriptor,)
                    )
                    intercept = np.ones((self.nsamp, 1), dtype=np.float64)
                    q_basis = _orthonormalize_columns(
                        np.column_stack([intercept, self.C_int])
                    )
                    expected_rank = self.p_eff + 1
                    if q_basis.shape != (self.nsamp, expected_rank):
                        raise RuntimeError(
                            "Complete native GxE projection basis lost rank: "
                            f"expected {(self.nsamp, expected_rank)}, got {q_basis.shape}."
                        )
                    q_basis = np.asfortranarray(q_basis, dtype=np.float64)
                    native_row_sel = np.asarray(self.row_sel, dtype=np.int64)
                    workspace_bytes = int(self.native_workspace_gib * (1024 ** 3))
                    self._native_context = _gxeldcore.DirectContext(
                        bed_descriptor=int(self._genotype_descriptors[".bed"]),
                        bim_descriptor=int(self._genotype_descriptors[".bim"]),
                        fam_descriptor=int(self._genotype_descriptors[".fam"]),
                        row_sel=native_row_sel,
                        ddof=int(self.ddof),
                        env=np.asarray(self.env, dtype=np.float64),
                        q_basis=q_basis,
                        decode_threads=int(self.decode_threads),
                        max_workspace_bytes=workspace_bytes,
                        target_panel_columns=self.native_target_panel_columns,
                        strict_feature_moment_verification=(
                            self.native_strict_feature_moment_verification
                        ),
                    )
                    context_info = self._native_context.info()
                    if (
                        int(context_info["n_total"]) != self.nsamp_total
                        or int(context_info["m_total"]) != self.nsnps
                        or int(context_info["n_selected"]) != self.nsamp
                        or int(context_info["q_rank"]) != expected_rank
                        or int(context_info["decode_threads"]) != self.decode_threads
                        or bool(context_info["strict_feature_moment_verification"])
                        != self.native_strict_feature_moment_verification
                    ):
                        self._native_context.close()
                        self._native_context = None
                        raise RuntimeError(
                            "The bounded native GxE context disagrees with validated Python dimensions/state."
                        )
                    if self.jackknife_ids is not None:
                        group_counts = [
                            int(np.unique(self.jackknife_ids[s:e]).size)
                            for s, e in self._make_compute_blocks()
                        ]
                        if max(group_counts, default=1) > 4:
                            self._native_context.close()
                            self._native_context = None
                            raise ValueError(
                                "The native GxE source API supports at most four jackknife "
                                "groups per compute block; reduce --step-size or use the "
                                "Python backend."
                            )
                    self.native_backend = "direct"
                    self.log._log(
                        "[gxe:native] Enabled immutable descriptor-owned direct BED context "
                        "(K=1, float64, standardized/sample, complete [1,E,C] projection; "
                        f"decode_threads={self.decode_threads}, workspace={self.native_workspace_gib:.2f} GiB)."
                    )
                    self.log._log(
                        "[gxe:native] Strict duplicate feature-moment verification "
                        f"{'enabled' if self.native_strict_feature_moment_verification else 'disabled'}: "
                        f"{self.native_feature_moment_integrity_reason}."
                    )
        self.log._log(
            f"[env] Using environment '{self.env_name}' with {self.nsamp} samples; "
            f"effective covariate rank={self.p_eff}; correlation df={self.df_corr}."
        )
        if self.genotype_format == "pgen":
            self._pgen_reader = PgenBlockReader(
                pgen_path=self.pgen_path,
                raw_sample_ct=self.nsamp_total,
                variant_ct=self.nsnps,
                sample_subset=self.row_sel,
                step_size=self.step_size,
                dtype=np.float64,
                ddof=self.ddof,
                standardize_threads=self.num_threads,
            )
            self.log._log(
                "[gxe][pgen] Persistent streamed REF-dosage reader ready: "
                f"buffer={self._pgen_reader.block_capacity}x{self.nsamp}."
            )

    def close(self) -> None:
        try:
            pgen_reader = getattr(self, "_pgen_reader", None)
            if pgen_reader is not None:
                pgen_reader.close()
                self._pgen_reader = None
            native_context = getattr(self, "_native_context", None)
            if native_context is not None:
                native_context.close()
                self._native_context = None
            close_reader = getattr(getattr(self, "G", None), "close", None)
            if close_reader is not None:
                close_reader()
        finally:
            finalizer = getattr(self, "_descriptor_finalizer", None)
            if finalizer is not None and finalizer.alive:
                finalizer()
            native_finalizer = getattr(self, "_native_binary_finalizer", None)
            if native_finalizer is not None and native_finalizer.alive:
                native_finalizer()
                self._native_binary_descriptor = None
        if self.covar_path is None:
            self.log._log("[env] No additional user covariates supplied; projecting on the environment main effect only.")
        else:
            self.log._log(
                f"[env] Included {len(self.cov_cols)} user covariate column(s) together with the environment main effect in the projection."
            )

    def _read_bim(self, bim_path):
        if bim_path is None:
            self.log._log("No .bim file is provided; all (anonymous) SNPs will be used")
            self.snplist = None
            return
        self.log._log(f"Reading {bim_path} for SNPs")
        self.snplist = pd.read_csv(
            bim_path,
            header=None,
            sep=r"\s+",
            dtype={0: str, 1: str, 3: np.int64, 4: str, 5: str},
        )
        self.snplist.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        if len(self.snplist) != self.nsnps:
            raise ValueError(
                f"The number of SNPs in the genotype file ({self.nsnps}) does not match "
                f"the BIM file ({len(self.snplist)})."
            )
        if self.snplist["SNP"].duplicated().any():
            raise ValueError("BIM file contains duplicate SNP identifiers; unique IDs are required for summary alignment.")

    def _read_annot(self, annot_path):
        if annot_path is None:
            self.annot = np.ones((self.nsnps, 1), dtype=np.float64)
            self.nbins = 1
            self.l2cols = ["L2_0"]
            self.is_continuous = False
            self.annot = np.ascontiguousarray(self.annot.astype(self.dtype, copy=False))
            self.nsnps_bin = np.asarray(self.annot, dtype=np.float64).sum(axis=0)
            self.log._log("Calculating genome-wide (non-partitioned) GxE LD scores")
            self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
            return

        parsed_ldsc = False
        df = pd.read_csv(annot_path, sep=r"\s+", compression="infer", dtype={"CHR": str, "SNP": str})
        base_cols = {"CHR", "BP", "SNP", "CM"}
        if {"CHR", "BP", "SNP"}.issubset(set(df.columns)):
            if df["SNP"].duplicated().any():
                raise ValueError("Annotation file contains duplicate SNP identifiers.")
            annot_cols = [c for c in df.columns if c not in base_cols]
            if len(annot_cols) == 0:
                raise ValueError("No annotation columns found after CHR/SNP/BP metadata columns.")
            if len(set(annot_cols)) != len(annot_cols):
                raise ValueError("Annotation column names must be unique.")
            bim_snps = self.snplist["SNP"].astype(str).tolist()
            ann_snps = df["SNP"].astype(str).tolist()
            if ann_snps == bim_snps:
                aligned_rows = df
            else:
                ann_set = set(ann_snps)
                bim_set = set(bim_snps)
                missing_in_annot = len(bim_set - ann_set)
                extra_in_annot = len(ann_set - bim_set)
                if missing_in_annot > 0:
                    raise ValueError(
                        f"Annotation SNP set is missing {missing_in_annot} BIM SNP(s); "
                        "prepare an annotation aligned to this genotype set."
                    )
                if extra_in_annot > 0:
                    self.log._log(
                        f"[info] Annotation contains {extra_in_annot} extra SNP(s); "
                        "keeping and ordering only BIM SNPs."
                    )
                aligned_rows = df.set_index("SNP").loc[bim_snps].reset_index()
            ann_chr = aligned_rows["CHR"].astype(str).to_numpy()
            bim_chr = self.snplist["CHR"].astype(str).to_numpy()
            ann_bp = pd.to_numeric(aligned_rows["BP"], errors="coerce").to_numpy(dtype=np.float64)
            bim_bp = self.snplist["BP"].to_numpy(dtype=np.float64)
            if not np.array_equal(ann_chr, bim_chr):
                first = int(np.flatnonzero(ann_chr != bim_chr)[0])
                raise ValueError(
                    f"Annotation CHR metadata disagree with BIM at SNP {bim_snps[first]!r}: "
                    f"annotation={ann_chr[first]!r}, BIM={bim_chr[first]!r}."
                )
            if not np.array_equal(ann_bp, bim_bp):
                mismatch = (~np.isfinite(ann_bp)) | (ann_bp != bim_bp)
                first = int(np.flatnonzero(mismatch)[0])
                raise ValueError(
                    f"Annotation BP metadata disagree with BIM at SNP {bim_snps[first]!r}: "
                    f"annotation={aligned_rows.iloc[first]['BP']!r}, BIM={self.snplist.iloc[first]['BP']!r}."
                )
            ordered = aligned_rows.loc[:, annot_cols]
            try:
                ann_mat = ordered.to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Annotation values must be numeric: {exc}") from exc
            self.annot = ann_mat
            self.nbins = self.annot.shape[1]
            self.l2cols = annot_cols
            parsed_ldsc = True
            self.log._log(f"Read LDSC-style annotation with shape {self.annot.shape}")

        if not parsed_ldsc:
            self.l2cols, arr = utils._read_with_optional_header(annot_path)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arr = arr.astype(np.float64, copy=False)
            self.annot = arr
            if self.l2cols is None:
                self.l2cols = [f"L2_{i}" for i in range(self.annot.shape[1])]
            if len(set(self.l2cols)) != len(self.l2cols):
                raise ValueError("Annotation column names must be unique.")
            self.nbins = self.annot.shape[1]
            self.log._log(f"Read thin annotation matrix with shape {self.annot.shape}")

        if self.annot.shape[0] != self.nsnps:
            raise ValueError(
                f"Number of SNPs in annotation ({self.annot.shape[0]}) does not match "
                f"the input genotype file ({self.nsnps})."
            )
        self.l2cols = _validate_gxe_annotation_names(list(self.l2cols))
        if not np.all(np.isfinite(self.annot)):
            raise ValueError("Annotation values must all be finite; NaN/Inf values are not accepted.")
        if np.any(self.annot < 0.0):
            raise ValueError("Annotation values must be non-negative.")
        self.is_continuous = not np.all(np.isin(np.unique(self.annot), [0.0, 1.0]))

        self.annot = np.ascontiguousarray(self.annot.astype(self.dtype, copy=False))
        # The canonical stored dtype defines the kernels.  Compute masses after
        # casting so the manifest, diagonal table, full traces, and deleted
        # traces all use byte-identical annotation weights.
        self.nsnps_bin = np.asarray(self.annot, dtype=np.float64).sum(axis=0)
        if np.any(self.nsnps_bin <= 0.0):
            bad = np.flatnonzero(self.nsnps_bin <= 0.0).tolist()
            raise ValueError(f"Annotation columns must have positive mass; empty columns: {bad}.")
        self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
        self.log._log(f"Nbins: {self.nbins}")

    def _make_compute_blocks(self):
        return [(s, min(self.nsnps, s + self.step_size)) for s in range(0, self.nsnps, self.step_size)]

    @staticmethod
    def _coalesce_contiguous_blocks(
        blocks: Sequence[tuple[int, int]], maximum_variants: int
    ) -> list[tuple[tuple[int, int], ...]]:
        """Group adjacent blocks without changing their probe identities."""
        groups: list[tuple[tuple[int, int], ...]] = []
        current: list[tuple[int, int]] = []
        for start, end in blocks:
            block = (int(start), int(end))
            if current and (
                block[0] != current[-1][1]
                or block[1] - current[0][0] > int(maximum_variants)
            ):
                groups.append(tuple(current))
                current = []
            current.append(block)
        if current:
            groups.append(tuple(current))
        return groups

    def _generate_random_group(
        self,
        blocks: Sequence[tuple[int, int]],
        v_count: int,
        v_start: int,
    ) -> np.ndarray:
        """Concatenate existing block-seeded probes into one native call."""
        total = sum(int(end) - int(start) for start, end in blocks)
        probes = np.empty((total, v_count), dtype=self.dtype, order="F")
        offset = 0
        for start, end in blocks:
            length = int(end) - int(start)
            probes[offset:offset + length, :] = self._generate_random_block(
                L=length,
                v_count=v_count,
                blk_start=int(start),
                v_start=v_start,
            )
            offset += length
        return probes

    def _make_native_feature_compute_blocks(self):
        """Use sufficiently wide, workspace-bounded blocks for skinny Q products."""
        moment_copies = 8 if self.native_strict_feature_moment_verification else 4
        q_rank = self.p_eff + 1
        elements_per_variant = self.nsamp + moment_copies * q_rank + 11
        workspace_elements = _native_execution_workspace_bytes(
            self.native_workspace_gib
        ) // 8
        # ``step_size`` defines the reproducible probe blocks, not the native
        # GEMM width.  Use the full configured workspace for the latter: the
        # native call remains bounded by ``required`` below, while wider calls
        # avoid repeating a genotype decode, protected-input snapshot, and
        # checksum setup for every small probe block.
        desired = int(self.nsnps)

        def required(variants: int) -> int:
            return int(variants) * elements_per_variant

        low, high = 0, desired
        while low < high:
            middle = (low + high + 1) // 2
            if required(middle) <= workspace_elements:
                low = middle
            else:
                high = middle - 1
        maximum = low
        if maximum < 1:
            raise RuntimeError(
                "The native GxE workspace cannot hold one feature variant."
            )
        feature_step = int(maximum)
        self.native_feature_step_size = feature_step
        if feature_step != self.step_size:
            self.log._log(
                "[gxe:native] Coalesced feature blocks from "
                f"{self.step_size} to {feature_step} variants for efficient "
                "fixed-effect moment products."
            )
        return [
            (start, min(self.nsnps, start + feature_step))
            for start in range(0, self.nsnps, feature_step)
        ]

    def _make_jackknife_compute_blocks(self) -> list[list[tuple[int, int]]]:
        """Split every jackknife group into bounded contiguous genotype reads."""
        if self.jackknife_ids is None:
            return []
        block_ids = np.asarray(self.jackknife_ids, dtype=np.int32)
        nblocks = len(self.jackknife_labels)
        if block_ids.shape != (self.nsnps,):
            raise RuntimeError(
                "GxE jackknife IDs must have one entry per variant; "
                f"got {block_ids.shape} for {self.nsnps} variants."
            )
        if np.any(block_ids < 0) or np.any(block_ids >= nblocks):
            raise RuntimeError("GxE jackknife IDs contain an out-of-range block index.")
        grouped: list[list[tuple[int, int]]] = [[] for _ in range(nblocks)]
        boundaries = np.r_[
            0,
            np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1,
            self.nsnps,
        ]
        for run_index in range(boundaries.size - 1):
            run_start = int(boundaries[run_index])
            run_end = int(boundaries[run_index + 1])
            block_id = int(block_ids[run_start])
            for start in range(run_start, run_end, self.step_size):
                grouped[block_id].append((start, min(run_end, start + self.step_size)))
        if any(not chunks for chunks in grouped):
            raise RuntimeError("GxE jackknife construction produced an empty block.")
        return grouped

    def _build_jackknife_blocks(self, spec: str | None) -> tuple[np.ndarray | None, list[str]]:
        if spec is None:
            return None, []
        text = str(spec).strip().lower()
        if text == "chr":
            labels_raw = self.snplist["CHR"].astype(str).to_numpy()
            labels = list(dict.fromkeys(labels_raw.tolist()))
            lookup = {label: idx for idx, label in enumerate(labels)}
            block_ids = np.asarray([lookup[x] for x in labels_raw], dtype=np.int32)
            labels = [f"chr:{x}" for x in labels]
        else:
            try:
                count = int(text)
            except ValueError as exc:
                raise ValueError(
                    "GxE reference jackknife supports --njack chr or a positive integer; "
                    f"got {spec!r}."
                ) from exc
            if count < 2 or count > self.nsnps:
                raise ValueError(f"GxE jackknife block count must be in [2, {self.nsnps}]; got {count}.")
            block_ids = np.minimum(
                count - 1,
                (np.arange(self.nsnps, dtype=np.int64) * count) // self.nsnps,
            ).astype(np.int32)
            labels = [f"block:{idx + 1}" for idx in range(count)]
        counts = np.bincount(block_ids, minlength=len(labels))
        if len(labels) < 2:
            raise ValueError("GxE jackknife requires at least two non-empty SNP blocks.")
        if np.any(counts == 0):
            raise ValueError("GxE jackknife construction produced an empty block.")
        block_masses = np.zeros((len(labels), self.nbins), dtype=np.float64)
        np.add.at(block_masses, block_ids, np.asarray(self.annot, dtype=np.float64))
        remaining_masses = self.nsnps_bin.reshape(1, -1) - block_masses
        invalid = np.argwhere(remaining_masses <= 0.0)
        if invalid.size:
            block_id, annotation_id = (int(x) for x in invalid[0])
            raise ValueError(
                "GxE jackknife deletion would empty an annotation: "
                f"block={labels[block_id]!r}, annotation={self.l2cols[annotation_id]!r}. "
                "Choose blocks for which every annotation retains positive mass."
            )
        method = (
            "exact two-sided shard deletion"
            if self.shard_mode
            else "block-local LD-score deletion"
        )
        self.log._log(
            f"[gxe:jackknife] configured {len(labels)} {method} blocks "
            f"({int(counts.min())}-{int(counts.max())} variants per block)."
        )
        return block_ids, labels

    def _read_genotype_block(self, blk_start: int, blk_end: int) -> np.ndarray:
        if getattr(self, "genotype_format", "bed") == "pgen":
            if self._pgen_reader is None:
                raise RuntimeError("PGEN reader is closed.")
            G = self._pgen_reader.read_standardized_block(blk_start, blk_end)
        else:
            indexer = np.s_[self.row_sel, blk_start:blk_end]
            try:
                G = self.G.read(index=indexer, dtype=np.float64, num_threads=self.decode_threads)
            except TypeError:
                try:
                    G = self.G.read(index=indexer, dtype=np.float64)
                except TypeError:
                    try:
                        G = self.G.read(index=indexer)
                    except TypeError:
                        G = self.G.read(indexer)

        G = np.asarray(G, dtype=np.float64)
        if G.shape == (blk_end - blk_start, self.nsamp):
            G = G.T
        if G.shape != (self.nsamp, blk_end - blk_start):
            raise RuntimeError(
                f"Unexpected genotype block shape {G.shape} for block [{blk_start}:{blk_end}); "
                f"expected ({self.nsamp}, {blk_end - blk_start})."
            )

        mask = np.isnan(G)
        nobs = G.shape[0] - mask.sum(axis=0, dtype=np.int64)
        col_means = np.divide(
            np.nansum(G, axis=0, dtype=np.float64),
            nobs,
            out=np.zeros(G.shape[1], dtype=np.float64),
            where=nobs > 0,
        )
        if mask.any():
            rr, cc = np.where(mask)
            G[rr, cc] = col_means[cc]
        G -= col_means
        if self.genotype_scale == "hwe":
            # With mean dosage mu=2p, sqrt(mu * (1-mu/2)) is sqrt(2p(1-p)).
            col_std = np.sqrt(np.maximum(col_means * (1.0 - 0.5 * col_means), 0.0))
        else:
            col_std = G.std(axis=0, ddof=self.ddof)
        good = np.isfinite(col_std) & (col_std > self.eps_var)
        if np.any(good):
            G[:, good] /= col_std[good]
        if np.any(~good):
            G[:, ~good] = 0.0
        return np.asarray(G, dtype=np.float64, order="F")

    def _project_and_center_inplace(self, M: np.ndarray) -> np.ndarray:
        if self.p_eff > 0:
            M -= self.C_int @ (self.cov_R_int @ M)
        M -= M.mean(axis=0, keepdims=True)
        return M

    def _prepare_additive_block(self, blk_start: int, blk_end: int, G: np.ndarray | None = None, apply_scale: bool = True, out_dtype=None) -> np.ndarray:
        if G is None:
            G = self._read_genotype_block(blk_start, blk_end)
        X = np.array(G, copy=True, dtype=np.float64, order="F")
        self._project_and_center_inplace(X)
        if apply_scale and self.kernel_mode == "standardized":
            if self.inv_sqrt_resvar_x_all is None:
                raise RuntimeError("Additive residual variances have not been precomputed.")
            X *= self.inv_sqrt_resvar_x_all[blk_start:blk_end].reshape(1, -1)
        return np.asarray(X, dtype=(out_dtype or self.dtype), order="F")

    def _prepare_interaction_block(self, blk_start: int, blk_end: int, G: np.ndarray | None = None, apply_scale: bool = True, out_dtype=None) -> np.ndarray:
        if G is None:
            G = self._read_genotype_block(blk_start, blk_end)
        W = np.asarray(G * self.env[:, None], dtype=np.float64, order="F")
        self._project_and_center_inplace(W)
        if apply_scale and self.kernel_mode == "standardized":
            if self.inv_sqrt_resvar_w_all is None:
                raise RuntimeError("Interaction residual variances have not been precomputed.")
            W *= self.inv_sqrt_resvar_w_all[blk_start:blk_end].reshape(1, -1)
        return np.asarray(W, dtype=(out_dtype or self.dtype), order="F")

    def _precompute_residual_variances(self) -> tuple[np.ndarray, np.ndarray]:
        native_direct = getattr(self, "native_backend", "python") == "direct"
        if native_direct:
            return self._precompute_residual_variances_native()
        return self._precompute_residual_variances_python()

    def _precompute_residual_variances_native(self) -> tuple[np.ndarray, np.ndarray]:
        if self._native_context is None:
            raise RuntimeError("The bounded native GxE backend was not initialized.")
        self.log._log(
            "[gxe:native] Precomputing feature scales/diagonals algebraically without X/W blocks."
        )
        arrays = {
            name: np.empty(self.nsnps, dtype=np.float64)
            for name in (
                "scale_x", "scale_w", "norm_x", "norm_w",
                "diag_nxe_x", "diag_nxe_w", "corr_xw",
            )
        }
        max_leak_x = 0.0
        max_leak_w = 0.0
        repaired_feature_moment_columns = 0
        blocks = self._make_native_feature_compute_blocks()
        for s, e in tqdm(
            blocks, desc="GxE native var", unit="block", disable=(not self.verbose)
        ):
            result = self._native_context.feature_block(
                    blk_start=int(s),
                    blk_end=int(e),
                    eps_var=float(self.eps_var),
                    require_missing_free=True,
            )
            for name in arrays:
                arrays[name][s:e] = np.asarray(result[name], dtype=np.float64)
            max_leak_x = max(
                max_leak_x,
                float(result["max_projection_leakage_additive"]),
            )
            max_leak_w = max(
                max_leak_w,
                float(result["max_projection_leakage_interaction"]),
            )
            repaired_feature_moment_columns += int(
                result.get(
                    "repaired_feature_moment_columns",
                    result["repaired_additive_moment_columns"],
                )
            )

        self.inv_sqrt_resvar_x_all = arrays["scale_x"]
        self.inv_sqrt_resvar_w_all = arrays["scale_w"]
        self.norm_x_all = arrays["norm_x"]
        self.norm_w_all = arrays["norm_w"]
        self.diag_nxe_x_all = arrays["diag_nxe_x"]
        self.diag_nxe_w_all = arrays["diag_nxe_w"]
        self.corr_xw_all = arrays["corr_xw"]
        self.score_x_all = None
        self.score_w_all = None
        trace_x = (
            float(self.df_corr)
            * (np.asarray(self.annot, dtype=np.float64).T @ self.norm_x_all)
            / self.nsnps_bin
        )
        trace_w = (
            float(self.df_corr)
            * (np.asarray(self.annot, dtype=np.float64).T @ self.norm_w_all)
            / self.nsnps_bin
        )
        max_norm_error_x = float(np.max(np.abs(self.norm_x_all - 1.0)))
        max_norm_error_w = float(np.max(np.abs(self.norm_w_all - 1.0)))
        if max(max_leak_x, max_leak_w) > 1.0e-9:
            raise RuntimeError(
                "Native projected GxE features leak into the fixed-effect span: "
                f"X={max_leak_x:.6g}, W={max_leak_w:.6g}."
            )
        if max(max_norm_error_x, max_norm_error_w) > 1.0e-9:
            raise RuntimeError(
                "Native post-projection GxE normalization failed: "
                f"X={max_norm_error_x:.6g}, W={max_norm_error_w:.6g}."
            )
        self.feature_diagnostics = {
            "valid_additive_columns": int(self.nsnps),
            "valid_interaction_columns": int(self.nsnps),
            "max_projection_leakage_additive": max_leak_x,
            "max_projection_leakage_interaction": max_leak_w,
            "min_norm_additive_over_rank": float(np.min(self.norm_x_all)),
            "max_norm_additive_over_rank": float(np.max(self.norm_x_all)),
            "min_norm_interaction_over_rank": float(np.min(self.norm_w_all)),
            "max_norm_interaction_over_rank": float(np.max(self.norm_w_all)),
            "max_norm_error_additive": max_norm_error_x,
            "max_norm_error_interaction": max_norm_error_w,
            "kernel_traces_additive": trace_x.tolist(),
            "kernel_traces_interaction": trace_w.tolist(),
            "max_trace_error_additive": float(np.max(np.abs(trace_x - self.df_corr))),
            "max_trace_error_interaction": float(np.max(np.abs(trace_w - self.df_corr))),
            "repaired_additive_moment_columns": int(
                repaired_feature_moment_columns
            ),
            "repaired_feature_moment_columns": int(
                repaired_feature_moment_columns
            ),
            "strict_feature_moment_verification": bool(
                self.native_strict_feature_moment_verification
            ),
            "feature_moment_integrity_mode": (
                "strict_duplicate"
                if self.native_strict_feature_moment_verification
                else "deterministic_disjoint_output_tiled_gemm"
            ),
            "feature_moment_integrity_reason": (
                self.native_feature_moment_integrity_reason
            ),
        }
        self.log._log(
            "[gxe:native:invariants] max fixed-effect leakage "
            f"X={max_leak_x:.3e}, W={max_leak_w:.3e}; "
            f"max norm error X={max_norm_error_x:.3e}, W={max_norm_error_w:.3e}; "
            "integrity-repaired feature-moment columns="
            f"{repaired_feature_moment_columns}."
        )
        return self.inv_sqrt_resvar_x_all, self.inv_sqrt_resvar_w_all

    def _precompute_residual_variances_python(self) -> tuple[np.ndarray, np.ndarray]:
        self.log._log("[gxe] Precomputing projected feature norms and NxE diagonals in a single pass.")
        inv_x = np.zeros(self.nsnps, dtype=np.float64)
        inv_w = np.zeros(self.nsnps, dtype=np.float64)
        norm_x = np.zeros(self.nsnps, dtype=np.float64)
        norm_w = np.zeros(self.nsnps, dtype=np.float64)
        diag_x = np.zeros(self.nsnps, dtype=np.float64)
        diag_w = np.zeros(self.nsnps, dtype=np.float64)
        corr_xw = np.zeros(self.nsnps, dtype=np.float64)
        score_x = np.zeros(self.nsnps, dtype=np.float64) if self.pheno is not None else None
        score_w = np.zeros(self.nsnps, dtype=np.float64) if self.pheno is not None else None
        bad_x = 0
        bad_w = 0
        max_projection_leakage_x = 0.0
        max_projection_leakage_w = 0.0
        blocks = self._make_compute_blocks()
        for s, e in tqdm(blocks, desc="GxE var", unit="block", disable=(not self.verbose)):
            G = self._read_genotype_block(s, e)

            X = self._prepare_additive_block(s, e, G=G, apply_scale=False, out_dtype=np.float64)
            ssx = np.sum(X * X, axis=0, dtype=np.float64)
            varx = ssx / float(self.df_corr)
            good_x = np.isfinite(varx) & (varx > self.eps_var)
            if self.kernel_mode == "standardized":
                inv_x[s:e][good_x] = 1.0 / np.sqrt(varx[good_x])
            else:
                inv_x[s:e][good_x] = 1.0
            inv_x[s:e][~good_x] = 0.0
            bad_x += int((~good_x).sum())

            W = self._prepare_interaction_block(s, e, G=G, apply_scale=False, out_dtype=np.float64)
            ssw = np.sum(W * W, axis=0, dtype=np.float64)
            varw = ssw / float(self.df_corr)
            good_w = np.isfinite(varw) & (varw > self.eps_var)
            invalid = ~(good_x & good_w)
            if np.any(invalid):
                local = np.flatnonzero(invalid)
                examples = []
                for offset in local[:5]:
                    idx = s + int(offset)
                    snp = str(self.snplist.iloc[idx]["SNP"]) if self.snplist is not None else str(idx)
                    failed = []
                    if not good_x[offset]:
                        failed.append("additive")
                    if not good_w[offset]:
                        failed.append("interaction")
                    examples.append(f"{snp} ({'+'.join(failed)})")
                raise ValueError(
                    f"{int(invalid.sum())} variant feature(s) in block [{s}:{e}) have zero or invalid "
                    "projected variance. GxE kernel normalization cannot retain zero columns; "
                    f"QC/remove these variants and regenerate the complete bundle. Examples: {', '.join(examples)}."
                )
            if self.kernel_mode == "standardized":
                inv_w[s:e][good_w] = 1.0 / np.sqrt(varw[good_w])
            else:
                inv_w[s:e][good_w] = 1.0
            inv_w[s:e][~good_w] = 0.0
            bad_w += int((~good_w).sum())

            X *= inv_x[s:e].reshape(1, -1)
            W *= inv_w[s:e].reshape(1, -1)
            final_ssx = np.sum(X * X, axis=0, dtype=np.float64)
            final_ssw = np.sum(W * W, axis=0, dtype=np.float64)
            norm_x[s:e] = final_ssx / float(self.df_corr)
            norm_w[s:e] = final_ssw / float(self.df_corr)

            # C_int is orthonormal and centered; the normalized all-ones vector
            # completes the fixed-effect basis.  This diagnostic therefore
            # measures the fraction of each feature norm leaking back into the
            # complete projected-out span without ever materializing P.
            intercept_x = X.sum(axis=0, dtype=np.float64) / math.sqrt(float(self.nsamp))
            intercept_w = W.sum(axis=0, dtype=np.float64) / math.sqrt(float(self.nsamp))
            leaked_x_sq = intercept_x * intercept_x
            leaked_w_sq = intercept_w * intercept_w
            if self.p_eff > 0:
                projected_x = self.C_int.T @ X
                projected_w = self.C_int.T @ W
                leaked_x_sq += np.sum(projected_x * projected_x, axis=0, dtype=np.float64)
                leaked_w_sq += np.sum(projected_w * projected_w, axis=0, dtype=np.float64)
            max_projection_leakage_x = max(
                max_projection_leakage_x,
                float(np.max(np.sqrt(leaked_x_sq / final_ssx))),
            )
            max_projection_leakage_w = max(
                max_projection_leakage_w,
                float(np.max(np.sqrt(leaked_w_sq / final_ssw))),
            )
            ex = self.env[:, None] * X
            ew = self.env[:, None] * W
            diag_x[s:e] = np.sum(ex * ex, axis=0, dtype=np.float64) / float(self.df_corr)
            diag_w[s:e] = np.sum(ew * ew, axis=0, dtype=np.float64) / float(self.df_corr)
            corr_xw[s:e] = np.sum(X * W, axis=0, dtype=np.float64) / float(self.df_corr)
            if self.pheno is not None:
                root_r = math.sqrt(float(self.df_corr))
                score_x[s:e] = (X.T @ self.pheno) / root_r
                score_w[s:e] = (W.T @ self.pheno) / root_r

            del G, X, W, ex, ew, ssx, ssw, final_ssx, final_ssw, varx, varw, good_x, good_w

        if bad_x > 0 or bad_w > 0:
            raise RuntimeError("Internal error: invalid projected feature columns escaped validation.")
        self.norm_x_all = norm_x
        self.norm_w_all = norm_w
        self.diag_nxe_x_all = diag_x
        self.diag_nxe_w_all = diag_w
        self.corr_xw_all = corr_xw
        self.score_x_all = score_x
        self.score_w_all = score_w
        trace_x = (
            float(self.df_corr)
            * (np.asarray(self.annot, dtype=np.float64).T @ norm_x)
            / self.nsnps_bin
        )
        trace_w = (
            float(self.df_corr)
            * (np.asarray(self.annot, dtype=np.float64).T @ norm_w)
            / self.nsnps_bin
        )
        max_norm_error_x = float(np.max(np.abs(norm_x - 1.0)))
        max_norm_error_w = float(np.max(np.abs(norm_w - 1.0)))
        projection_tolerance = 1.0e-9
        if max(max_projection_leakage_x, max_projection_leakage_w) > projection_tolerance:
            raise RuntimeError(
                "Projected GxE features leak into the fixed-effect span: "
                f"max additive={max_projection_leakage_x:.6g}, "
                f"max interaction={max_projection_leakage_w:.6g}, "
                f"tolerance={projection_tolerance:.6g}."
            )
        if self.kernel_mode == "standardized":
            norm_tolerance = 1.0e-9
            if max(max_norm_error_x, max_norm_error_w) > norm_tolerance:
                raise RuntimeError(
                    "Post-projection SUMMIT feature normalization failed: "
                    f"max additive norm error={max_norm_error_x:.6g}, "
                    f"max interaction norm error={max_norm_error_w:.6g}, "
                    f"tolerance={norm_tolerance:.6g}."
                )
        self.feature_diagnostics = {
            "valid_additive_columns": int(self.nsnps - bad_x),
            "valid_interaction_columns": int(self.nsnps - bad_w),
            "max_projection_leakage_additive": max_projection_leakage_x,
            "max_projection_leakage_interaction": max_projection_leakage_w,
            "min_norm_additive_over_rank": float(np.min(norm_x)),
            "max_norm_additive_over_rank": float(np.max(norm_x)),
            "min_norm_interaction_over_rank": float(np.min(norm_w)),
            "max_norm_interaction_over_rank": float(np.max(norm_w)),
            "max_norm_error_additive": max_norm_error_x,
            "max_norm_error_interaction": max_norm_error_w,
            "kernel_traces_additive": trace_x.tolist(),
            "kernel_traces_interaction": trace_w.tolist(),
            "max_trace_error_additive": float(np.max(np.abs(trace_x - self.df_corr))),
            "max_trace_error_interaction": float(np.max(np.abs(trace_w - self.df_corr))),
        }
        self.log._log(
            "[gxe:invariants] max fixed-effect leakage "
            f"X={max_projection_leakage_x:.3e}, W={max_projection_leakage_w:.3e}; "
            f"max norm error X={max_norm_error_x:.3e}, W={max_norm_error_w:.3e}."
        )
        return np.asarray(inv_x, dtype=np.float64), np.asarray(inv_w, dtype=np.float64)

    def _generate_random_block(self, L: int, v_count: int, blk_start: int, v_start: int) -> np.ndarray:
        Z = np.empty((L, v_count), dtype=np.float64, order="F")
        for local_probe in range(v_count):
            probe_id = self.probe_offset + v_start + local_probe
            rng = np.random.Generator(
                np.random.Philox(_make_seed(self.root_seed, blk_start, probe_id))
            )
            if self.rand_dist == "rademacher":
                values = rng.integers(0, 2, size=L, dtype=np.int8).astype(np.float64)
                values = 2.0 * values - 1.0
            elif self.rand_dist in ("gaussian", "normal", "spherical"):
                values = rng.standard_normal(size=L).astype(np.float64, copy=False)
                if self.rand_dist == "spherical":
                    norm = float(np.linalg.norm(values))
                    if norm > 0.0:
                        values *= math.sqrt(float(L)) / norm
                    else:
                        values[:] = 0.0
            else:
                raise ValueError(f"Unsupported rand_dist: {self.rand_dist}")
            Z[:, local_probe] = values
        return np.asarray(Z, dtype=self.dtype, order="F")

    def _auto_vtiles(self) -> list[tuple[int, int]]:
        target_gib = float(self.target_xz_mem)
        itemsize = int(np.dtype(self.dtype).itemsize)
        native_direct = getattr(self, "native_backend", "python") == "direct"
        # Ordinary references use additive-analogous block-local deletion from
        # their completed LD-score rows.  Only legacy reference shards retain
        # the exact two-sided within-block machinery.
        exact_jackknife = self.jackknife_ids is not None and bool(
            getattr(self, "shard_mode", False)
        )
        resident_multiplier = 4 if exact_jackknife else 2
        if native_direct:
            # Let U=N*K*B*8.  Each opaque panel owns S and e*S.  Preparing
            # a panel peaks at its 2U input plus a 4U snapshot. Exact JK keeps
            # both 2B stored sketches during block preparation, but never keeps
            # the global and block opaque panels together.
            if itemsize == np.dtype(np.float64).itemsize:
                peak_multiplier = 8 if exact_jackknife else 6
            else:
                # The two float32 stored sketches use 2U total. Python's 2U
                # float64 call input and C++'s 4U opaque snapshot peak at 8U.
                peak_multiplier = 8 if exact_jackknife else 7
            denominator_itemsize = np.dtype(np.float64).itemsize
        else:
            peak_multiplier = resident_multiplier
            denominator_itemsize = itemsize
        denom = max(
            1,
            peak_multiplier
            * int(self.nsamp)
            * int(self.nbins)
            * int(denominator_itemsize),
        )
        memory_vmax = int((target_gib * (1024 ** 3)) // denom)
        if memory_vmax < 1:
            raise RuntimeError(
                "The configured target sketch-memory limit cannot hold one probe: "
                f"required={denom} bytes, limit={int(target_gib * (1024 ** 3))} bytes."
            )
        vmax = min(self.nvecs, memory_vmax)
        tiles = _build_balanced_vtiles(self.nvecs, vmax=vmax, gran=64, max_tiles=8)
        if not tiles:
            tiles = [(0, self.nvecs)]
        vdesc = ",".join(str(v) for _, v in tiles)
        approx = (
            peak_multiplier
            * self.nsamp
            * self.nbins
            * max(v for _, v in tiles)
            * denominator_itemsize
        ) / (1024 ** 3)
        label = (
            "native opaque-panel preparation peak"
            if native_direct
            else (
                "global+one-block in-memory sketches"
                if exact_jackknife
                else "paired global sketches"
            )
        )
        self.log._log(
            f"[gxe:auto_vchunk] dtype={self.dtype} target≈{target_gib:.1f} GiB, "
            f"v_tiles=[{vdesc}] -> {label}≈{approx:.2f} GiB"
        )
        return tiles

    def _accumulate_sketch_block(
        self,
        U_chunk: np.ndarray,
        W: np.ndarray,
        Z: np.ndarray,
        annot_blk: np.ndarray,
        mirror_chunk: np.ndarray | None = None,
    ) -> None:
        sqrt_annot = np.sqrt(np.maximum(annot_blk, 0))
        for k in range(self.nbins):
            wk = sqrt_annot[:, k]
            if not np.any(wk):
                continue
            seg = slice(k * Z.shape[1], (k + 1) * Z.shape[1])
            weighted_probes = wk.reshape(-1, 1) * Z
            contribution = W @ weighted_probes
            U_chunk[:, seg] += contribution
            if mirror_chunk is not None:
                mirror_chunk[:, seg] += contribution

    def _accumulate_left_scores(self, Work: np.ndarray, meansq_accum: np.ndarray, blk_start: int, blk_end: int, Vt: int) -> None:
        scale = 1.0 / float(self.df_corr ** 2)
        for k in range(self.nbins):
            seg = Work[:, k * Vt:(k + 1) * Vt]
            meansq_accum[blk_start:blk_end, k] += np.sum(seg * seg, axis=1, dtype=np.float64) * scale

    def _accumulate_left_scores_rows(
        self,
        Work: np.ndarray,
        meansq_accum: np.ndarray,
        rows: np.ndarray,
        Vt: int,
    ) -> None:
        scale = 1.0 / float(self.df_corr ** 2)
        for k in range(self.nbins):
            seg = Work[:, k * Vt:(k + 1) * Vt]
            meansq_accum[rows, k] += np.sum(seg * seg, axis=1, dtype=np.float64) * scale

    def _accumulate_annotation_pair_sums(
        self,
        Work: np.ndarray,
        annot_left: np.ndarray,
        out: np.ndarray,
        Vt: int,
    ) -> None:
        scale = 1.0 / float(self.df_corr ** 2)
        for source_bin in range(self.nbins):
            seg = Work[:, source_bin * Vt:(source_bin + 1) * Vt]
            row_sums = np.sum(seg * seg, axis=1, dtype=np.float64) * scale
            out[:, source_bin] += annot_left.T @ row_sums

    def _native_source_block(
        self,
        blk_start: int,
        blk_end: int,
        probes: np.ndarray,
        jackknife_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._native_context is None:
            raise RuntimeError("The bounded native GxE backend was not initialized.")
        length = blk_end - blk_start
        if jackknife_ids is None:
            unique_ids = np.asarray([0], dtype=np.int32)
            dense_ids = np.zeros(length, dtype=np.int32)
        else:
            unique_ids = np.unique(np.asarray(jackknife_ids, dtype=np.int32))
            if unique_ids.size > 4:
                raise RuntimeError(
                    "A GxE compute block intersects more than four jackknife groups; "
                    "reduce --step-size or use the Python backend."
                )
            dense_ids = np.searchsorted(unique_ids, jackknife_ids).astype(np.int32)
        started = time.perf_counter()
        try:
            source_x, source_w, missing = self._native_context.source_block(
                blk_start=int(blk_start),
                blk_end=int(blk_end),
                scale_x=np.asarray(
                    self.inv_sqrt_resvar_x_all[blk_start:blk_end], dtype=np.float64
                ),
                scale_w=np.asarray(
                    self.inv_sqrt_resvar_w_all[blk_start:blk_end], dtype=np.float64
                ),
                sqrt_annotation=np.sqrt(
                    np.maximum(
                        np.asarray(self.annot[blk_start:blk_end, 0], dtype=np.float64),
                        0.0,
                    )
                ),
                probes=np.asfortranarray(probes, dtype=np.float64),
                group_ids=dense_ids,
                num_groups=int(unique_ids.size),
                require_missing_free=True,
            )
        finally:
            if not hasattr(self, "native_phase_timings"):
                self.native_phase_timings = {}
            self.native_phase_timings["source_calls"] = (
                self.native_phase_timings.get("source_calls", 0.0)
                + time.perf_counter() - started
            )
        if int(missing) != 0:
            raise RuntimeError("Native GxE source unexpectedly consumed missing genotypes.")
        return unique_ids, np.asarray(source_x), np.asarray(source_w)

    def _native_target_block(
        self,
        blk_start: int,
        blk_end: int,
        sources,
        second_sources=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._native_context is None:
            raise RuntimeError("The bounded native GxE backend was not initialized.")
        arguments = {
            "blk_start": int(blk_start),
            "blk_end": int(blk_end),
            "scale_x": np.asarray(
                self.inv_sqrt_resvar_x_all[blk_start:blk_end], dtype=np.float64
            ),
            "scale_w": np.asarray(
                self.inv_sqrt_resvar_w_all[blk_start:blk_end], dtype=np.float64
            ),
            "require_missing_free": True,
        }
        started = time.perf_counter()
        try:
            if second_sources is None:
                work_x, work_w, missing, source_leakage = (
                    self._native_context.target_projected_block(
                        sources=sources, **arguments
                    )
                )
            else:
                work_x, work_w, missing, source_leakage = (
                    self._native_context.target_projected_pair_block(
                        first=sources, second=second_sources, **arguments
                    )
                )
        finally:
            if not hasattr(self, "native_phase_timings"):
                self.native_phase_timings = {}
            self.native_phase_timings["target_calls"] = (
                self.native_phase_timings.get("target_calls", 0.0)
                + time.perf_counter() - started
            )
        if int(missing) != 0:
            raise RuntimeError("Native GxE target unexpectedly consumed missing genotypes.")
        if not np.isfinite(float(source_leakage)):
            raise RuntimeError("Native GxE target returned a non-finite source leakage diagnostic.")
        return np.asarray(work_x), np.asarray(work_w)

    def _native_prepare_projected_sources(self, sources: np.ndarray):
        if self._native_context is None:
            raise RuntimeError("The bounded native GxE backend was not initialized.")
        source_dtype = np.asarray(sources).dtype
        if source_dtype == np.float32:
            # A global sketch receives one rounded float32 update per genotype
            # block.  The native panel is always reprojected in float64 before
            # use, so this gate distinguishes bounded accumulation roundoff
            # from a genuinely nonprojected source rather than requiring the
            # pre-correction sketch to meet float64 orthogonality.  The 1e-3
            # ceiling remains an absolute contract guard.
            accumulation_blocks = max(
                1, math.ceil(int(self.nsnps) / int(self.step_size))
            )
            tolerance = min(
                1.0e-3,
                max(
                    5.0e-6,
                    32.0
                    * float(np.finfo(np.float32).eps)
                    * math.sqrt(float(accumulation_blocks)),
                ),
            )
        else:
            tolerance = 1.0e-9
        started = time.perf_counter()
        try:
            panel = self._native_context.prepare_projected_sources(
                sources=np.asfortranarray(sources, dtype=np.float64),
                tolerance=tolerance,
            )
        finally:
            if not hasattr(self, "native_phase_timings"):
                self.native_phase_timings = {}
            self.native_phase_timings["panel_preparation"] = (
                self.native_phase_timings.get("panel_preparation", 0.0)
                + time.perf_counter() - started
            )
        leakage = float(panel.leakage)
        if not np.isfinite(leakage):
            raise RuntimeError("Native GxE projected-source validation was non-finite.")
        return panel, leakage

    def _native_validate_projected_sources(self, sources: np.ndarray) -> float:
        if self._native_context is None:
            raise RuntimeError("The bounded native GxE backend was not initialized.")
        leakage = float(
            self._native_context.validate_projected_sources(
                sources=np.asfortranarray(sources, dtype=np.float64),
                tolerance=1.0e-9,
            )
        )
        if not np.isfinite(leakage):
            raise RuntimeError("Native GxE projected-source validation was non-finite.")
        return leakage

    def _compute_within_jackknife_scores(
        self,
        blocks: list[tuple[int, int]],
        vtiles: list[tuple[int, int]],
    ) -> dict[str, np.ndarray] | None:
        """Legacy reread oracle retained for focused equivalence tests only."""
        if self.jackknife_ids is None:
            return None
        nblock = len(self.jackknife_labels)
        within = {
            key: np.zeros((nblock, self.nbins, self.nbins), dtype=np.float64)
            for key in ("xx", "xw", "wx", "ww")
        }
        self.log._log("[gxe:jackknife:oracle] using legacy reread implementation.")
        for v0, vt in vtiles:
            for block_id in range(nblock):
                relevant: list[tuple[int, int, np.ndarray]] = []
                for s, e in blocks:
                    local = np.flatnonzero(self.jackknife_ids[s:e] == block_id)
                    if local.size:
                        relevant.append((s, e, local))

                sketch_x = np.zeros((self.nsamp, self.nbins * vt), dtype=self.dtype, order="F")
                sketch_w = np.zeros((self.nsamp, self.nbins * vt), dtype=self.dtype, order="F")
                for s, e, local in relevant:
                    geno = self._read_genotype_block(s, e)
                    x = self._prepare_additive_block(s, e, G=geno, apply_scale=True, out_dtype=self.dtype)
                    w = self._prepare_interaction_block(s, e, G=geno, apply_scale=True, out_dtype=self.dtype)
                    probes = self._generate_random_block(L=e - s, v_count=vt, blk_start=s, v_start=v0)
                    annot_local = np.asarray(self.annot[s:e][local], dtype=self.dtype)
                    probes_local = np.asarray(probes[local], order="F")
                    self._accumulate_sketch_block(
                        sketch_x, np.asarray(x[:, local], order="F"), probes_local, annot_local
                    )
                    self._accumulate_sketch_block(
                        sketch_w, np.asarray(w[:, local], order="F"), probes_local, annot_local
                    )
                    del geno, x, w, probes, probes_local, annot_local

                for s, e, local in relevant:
                    geno = self._read_genotype_block(s, e)
                    annot_left = np.asarray(self.annot[s:e][local], dtype=np.float64)
                    x = self._prepare_additive_block(s, e, G=geno, apply_scale=True, out_dtype=self.dtype)
                    w = self._prepare_interaction_block(s, e, G=geno, apply_scale=True, out_dtype=self.dtype)
                    for key, left, sketch in (
                        ("xx", x, sketch_x), ("xw", x, sketch_w),
                        ("wx", w, sketch_x), ("ww", w, sketch_w),
                    ):
                        work = np.asarray(left[:, local].T @ sketch, dtype=np.float64)
                        self._accumulate_annotation_pair_sums(
                            work, annot_left, within[key][block_id], vt
                        )
                        del work
                    del geno, x, w, annot_left
                del sketch_x, sketch_w
                gc.collect()
        for key in within:
            within[key] /= float(self.nvecs)
        return within

    def _save_score_file(self, path: str, score: np.ndarray) -> None:
        snpcols = ["CHR", "SNP", "BP"]
        if self.snplist is None:
            snpdf = pd.DataFrame(np.nan * np.ones((self.nsnps, 3)), columns=snpcols)
        else:
            snpdf = self.snplist[["CHR", "SNP", "BP"]].copy()
            snpdf.columns = snpcols
        scores_df = pd.DataFrame(score, columns=self.l2cols)
        out_df = pd.concat([snpdf, scores_df], axis=1)
        self._atomic_dataframe(
            out_df,
            path,
            sep="\t",
            compression="gzip",
            float_format="%.17g" if self.shard_mode else "%.10g",
        )

    @staticmethod
    def _atomic_dataframe(frame: pd.DataFrame, path: str, **kwargs) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.fchmod(fd, 0o600)
        os.close(fd)
        try:
            frame.to_csv(temporary, index=False, **kwargs)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _atomic_json(payload: dict, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _atomic_npz(path: str, **arrays) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _planned_output_paths(self) -> list[Path]:
        suffixes = [
            ".gxx.ldscore.gz",
            ".gxe.ldscore.gz",
            ".exg.ldscore.gz",
            ".gee.ldscore.gz",
        ]
        if self.shard_mode:
            suffixes.extend([".gxe.shard.identity.json", ".gxe.shard.json"])
        else:
            suffixes.extend([".gxe.diag.tsv.gz", ".gxe.ref.json"])
        if self.pheno is not None:
            suffixes.extend([".gxe.gwas.tsv.gz", ".gxe.gwis.tsv.gz", ".gxe.moments.json"])
        if self.write_jackknife and self.shard_mode:
            suffixes.append(".gxe.jackknife.npz")
        return [Path(f"{self.outpath}{suffix}") for suffix in suffixes]

    def _assert_output_paths_available(self) -> None:
        existing = [str(path) for path in self._planned_output_paths() if path.exists()]
        if existing and not self.overwrite:
            shown = ", ".join(existing[:5]) + (" ..." if len(existing) > 5 else "")
            raise FileExistsError(
                "Refusing to overwrite an existing GxE bundle. Choose a new --out prefix or explicitly pass "
                f"--gxe-overwrite. Existing artifacts: {shown}"
            )

    def _variant_digest(self) -> str:
        digest = hashlib.sha256()
        if self.snplist is None:
            for idx in range(self.nsnps):
                digest.update(f"NA\x1f{idx}\x1fNA\x1fNA\x1fNA\n".encode("utf-8"))
        else:
            for row in self.snplist[["CHR", "SNP", "BP", "A1", "A2"]].itertuples(index=False, name=None):
                digest.update("\x1f".join(str(x) for x in row).encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()

    def _analysis_fingerprint(self) -> str:
        selected = self.sample_ids.iloc[self.row_sel]
        digest = hashlib.sha256()
        for fid, iid in selected.itertuples(index=False, name=None):
            digest.update(str(fid).encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(str(iid).encode("utf-8"))
            digest.update(b"\n")
        digest.update(np.asarray(self.env, dtype="<f8").tobytes(order="C"))
        design_hash = self.environment_transform.get("fixed_effect_design_sha256")
        if not isinstance(design_hash, str) or len(design_hash) != 64:
            raise RuntimeError("Fixed-effect design digest is missing from the GxE analysis metadata.")
        digest.update(bytes.fromhex(design_hash))
        digest.update(int(self.p_eff).to_bytes(8, byteorder="little", signed=False))
        return digest.hexdigest()

    def _nxe_reference_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return compact exact statistics for ``P diag(e**2) P`` traces."""
        n = self.nsamp
        intercept = np.ones((n, 1), dtype=np.float64) / math.sqrt(float(n))
        q_full = _orthonormalize_columns(np.column_stack([intercept, self.C_int]))
        expected_rank = int(self.C_int.shape[1]) + 1
        if q_full.shape != (n, expected_rank):
            raise RuntimeError(
                "Fixed-effect basis lost rank while constructing exact NxE trace statistics: "
                f"expected {(n, expected_rank)}, got {q_full.shape}."
            )
        d = np.asarray(self.env * self.env, dtype=np.float64)
        qdq = q_full.T @ (d[:, None] * q_full)
        trace_terms = np.asarray(
            [
                np.sum(d, dtype=np.float64),
                np.dot(d, d),
                np.sum((d[:, None] * q_full) ** 2, dtype=np.float64),
            ],
            dtype=np.float64,
        )
        return np.asarray(qdq, dtype=np.float64, order="C"), trace_terms

    def _nxe_reference_traces(self) -> tuple[float, float]:
        qdq, trace_terms = self._nxe_reference_statistics()
        trace_pd = float(trace_terms[0] - np.trace(qdq))
        trace_pd_sq = float(
            trace_terms[1] - 2.0 * trace_terms[2] + np.sum(qdq * qdq.T)
        )
        if trace_pd < -1e-8 or trace_pd_sq < -1e-8:
            raise RuntimeError("Computed a negative NxE kernel trace; projection metadata are inconsistent.")
        return max(0.0, trace_pd), max(0.0, trace_pd_sq)

    def _relative_output_path(self, target: str, manifest_path: str) -> str:
        return os.path.relpath(os.path.abspath(target), start=os.path.dirname(os.path.abspath(manifest_path)) or ".")

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _annotation_digest(self) -> str:
        digest = hashlib.sha256()
        for name in self.l2cols:
            digest.update(str(name).encode("utf-8"))
            digest.update(b"\n")
        digest.update(np.asarray(self.annot, dtype="<f8", order="C").tobytes(order="C"))
        return digest.hexdigest()

    def _jackknife_digest(self) -> str:
        digest = hashlib.sha256()
        if self.jackknife_ids is None:
            digest.update(b"none\n")
        else:
            digest.update(np.asarray(self.jackknife_ids, dtype="<i4").tobytes(order="C"))
            for label in self.jackknife_labels:
                digest.update(str(label).encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()

    def _genotype_provenance(self) -> dict[str, dict[str, int | str]]:
        return {
            ext: {
                "bytes": int(os.fstat(self._genotype_descriptors[ext]).st_size),
                "sha256": self._file_sha256(str(self._stable_genotype_paths[ext])),
            }
            for ext in self._genotype_extensions
        }

    def _capture_genotype_file_state(
        self,
    ) -> dict[str, tuple[int, int, int, int, int]]:
        state = {}
        for extension in self._genotype_extensions:
            observed = os.fstat(self._genotype_descriptors[extension])
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError(
                    f"PLINK input descriptor is no longer a regular file: {extension}."
                )
            state[extension] = (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
        return state

    def _assert_construction_genotype_state(self) -> None:
        observed = self._capture_genotype_file_state()
        if observed != self._construction_genotype_state:
            changed = [
                extension
                for extension in self._genotype_extensions
                if observed.get(extension)
                != self._construction_genotype_state.get(extension)
            ]
            raise RuntimeError(
                "PLINK inputs changed after the GxE estimator loaded its variant/sample "
                f"state; construct a fresh estimator. Changed files: {changed}."
            )

    def _assert_genotype_provenance_unchanged(
        self, expected: Mapping[str, Mapping[str, int | str]]
    ) -> None:
        self._assert_construction_genotype_state()
        observed = self._genotype_provenance()
        if observed != dict(expected):
            changed = [
                extension
                for extension in self._genotype_extensions
                if observed.get(extension) != expected.get(extension)
            ]
            raise RuntimeError(
                "PLINK genotype inputs changed while GxE artifacts were being computed; "
                f"aborting publication. Changed files: {changed}."
            )

    def _feature_cache_identity(
        self,
        genotype_files: Mapping[str, Mapping[str, int | str]] | None = None,
    ) -> dict:
        if genotype_files is None:
            genotype_files = getattr(self, "_active_genotype_provenance", None)
        if genotype_files is None:
            genotype_files = self._genotype_provenance()
        return {
            "analysis_fingerprint": self._analysis_fingerprint(),
            "variant_digest": self._variant_digest(),
            "annotation_digest": self._annotation_digest(),
            "jackknife_digest": self._jackknife_digest(),
            "n_samples": self.nsamp,
            "n_variants": self.nsnps,
            "fixed_effect_rank_excluding_intercept": self.p_eff,
            "residual_rank": self.df_corr,
            "kernel_mode": self.kernel_mode,
            "genotype_scale": self.genotype_scale,
            "ddof": self.ddof,
            "eps_var": self.eps_var,
            "annotation_names": list(self.l2cols),
            "annotation_masses": np.asarray(self.nsnps_bin, dtype=np.float64).tolist(),
            "jackknife_labels": None if self.jackknife_ids is None else list(self.jackknife_labels),
            "environment": self.env_name,
            "environment_transform": self.environment_transform,
            "covariates": list(self.cov_cols),
            "genotype_files": genotype_files,
        }

    def _backend_provenance(self, artifact_stage: str) -> dict:
        build_info = None
        native_binary_sha256 = None
        if self.native_backend == "direct":
            build_info = self._native_build_info
            descriptor = self._native_binary_descriptor
            binary_record = self._native_binary_record
            if build_info is None or descriptor is None or binary_record is None:
                raise RuntimeError("Direct GxE backend build provenance is unavailable.")
            observed = os.fstat(descriptor)
            identity = (
                observed.st_dev, observed.st_ino, observed.st_size,
                observed.st_mtime_ns, observed.st_ctime_ns,
            )
            if identity != tuple(binary_record["identity"]):
                raise RuntimeError(
                    "The loaded GxE native extension inode changed after initialization."
                )
            native_binary_sha256 = _sha256_descriptor(descriptor)
            if native_binary_sha256 != binary_record["sha256"]:
                raise RuntimeError(
                    "The loaded GxE native extension bytes changed after initialization."
                )
            backend_name = str(build_info.get("backend_name", "gxeldcore_direct"))
            backend_version = str(build_info.get("backend_version", "unknown"))
            compile_options = {
                key: build_info.get(key)
                for key in (
                    "api_version", "compiler_id", "compiler_version", "build_type",
                    "blas_vendor", "cxx_standard", "optimization",
                    "architecture_tuning", "openmp_enabled",
                    "native_optimization_enabled", "platform",
                    "blas_runtime_config",
                )
            }
            compile_options["strict_feature_moment_verification"] = bool(
                self.native_strict_feature_moment_verification
            )
            workspace_cap = int(self.native_workspace_gib * (1024 ** 3))
            panel_columns = int(self.native_target_panel_columns)
        else:
            backend_name = "python_numpy"
            backend_version = "gxe_python_v1"
            compile_options = None
            workspace_cap = 0
            panel_columns = 0
        source_commit = (
            str(build_info.get("source_commit", "unknown"))
            if build_info is not None
            else "unknown"
        )
        source_tree_sha256 = (
            str(build_info["source_tree_sha256"])
            if build_info is not None
            else None
        )
        global_width = int(
            self.resource_estimates.get("actual_global_2b_source_columns", 0)
        )
        jackknife_width = int(
            self.resource_estimates.get("actual_jackknife_2b_source_columns", 0)
        )
        target_width = int(self.resource_estimates.get("target_source_columns", 0))
        if artifact_stage == "feature_construction":
            global_width = jackknife_width = target_width = 0
        provenance = {
            "schema_version": _BACKEND_PROVENANCE_SCHEMA_VERSION,
            "artifact_stage": str(artifact_stage),
            "backend_name": backend_name,
            "backend_version": backend_version,
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
            "native_binary_sha256": native_binary_sha256,
            "compile_options": compile_options,
            "native_workspace_cap_bytes": workspace_cap,
            "configured_target_panel_columns": panel_columns,
            "actual_global_2b_source_columns": global_width,
            "actual_jackknife_2b_source_columns": jackknife_width,
            "actual_target_source_columns": target_width,
        }
        _validate_backend_provenance(provenance, expected_stage=str(artifact_stage))
        return provenance

    def write_feature_cache(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write exact phenotype-independent projected-feature metadata."""
        if self.genotype_format != "bed":
            raise ValueError("Reusable GxE feature caches currently require BED/BIM/FAM input.")
        if self.pheno is not None or self.pheno_path is not None:
            raise ValueError("A reusable GxE feature cache must be phenotype-free.")
        target = Path(path).expanduser().resolve()
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing GxE feature cache: {target}.")
        self._assert_construction_genotype_state()
        initial_provenance = self._genotype_provenance()
        # Recompute unconditionally so a cache can never publish stale feature
        # arrays retained on a long-lived estimator after its PLINK inputs were
        # replaced between calls.
        self.inv_sqrt_resvar_x_all, self.inv_sqrt_resvar_w_all = (
            self._precompute_residual_variances()
        )
        required = (
            self.inv_sqrt_resvar_x_all, self.inv_sqrt_resvar_w_all,
            self.norm_x_all, self.norm_w_all, self.diag_nxe_x_all,
            self.diag_nxe_w_all, self.corr_xw_all,
        )
        if any(value is None for value in required):
            raise RuntimeError("Projected-feature metadata are incomplete.")
        identity = self._feature_cache_identity(initial_provenance)
        nxe_qdq, nxe_trace_terms = self._nxe_reference_statistics()
        trace_nxe = max(0.0, float(nxe_trace_terms[0] - np.trace(nxe_qdq)))
        trace_nxe_sq = max(
            0.0,
            float(
                nxe_trace_terms[1]
                - 2.0 * nxe_trace_terms[2]
                + np.sum(nxe_qdq * nxe_qdq.T)
            ),
        )
        metadata = {
            "kind": "summit.gxe.feature_cache",
            "schema_version": _FEATURE_CACHE_SCHEMA_VERSION,
            **identity,
            "feature_diagnostics": self.feature_diagnostics,
            "trace_nxe": trace_nxe,
            "trace_nxe_sq": trace_nxe_sq,
            "backend_provenance": self._backend_provenance(
                "feature_construction"
            ),
        }
        variants = self.snplist[["CHR", "SNP", "BP", "A1", "A2"]]
        arrays = {
            "variant_chr": variants["CHR"].astype(str).to_numpy(dtype=np.str_),
            "variant_snp": variants["SNP"].astype(str).to_numpy(dtype=np.str_),
            "variant_bp": variants["BP"].to_numpy(dtype=np.int64),
            "variant_a1": variants["A1"].astype(str).to_numpy(dtype=np.str_),
            "variant_a2": variants["A2"].astype(str).to_numpy(dtype=np.str_),
            "annotations": np.asarray(self.annot, dtype=np.float64),
            "jackknife_ids": (
                np.asarray([], dtype=np.int32)
                if self.jackknife_ids is None
                else np.asarray(self.jackknife_ids, dtype=np.int32)
            ),
            "scale_x": np.asarray(self.inv_sqrt_resvar_x_all, dtype=np.float64),
            "scale_w": np.asarray(self.inv_sqrt_resvar_w_all, dtype=np.float64),
            "norm_x": np.asarray(self.norm_x_all, dtype=np.float64),
            "norm_w": np.asarray(self.norm_w_all, dtype=np.float64),
            "diag_nxe_x": np.asarray(self.diag_nxe_x_all, dtype=np.float64),
            "diag_nxe_w": np.asarray(self.diag_nxe_w_all, dtype=np.float64),
            "corr_xw": np.asarray(self.corr_xw_all, dtype=np.float64),
            "nxe_qdq": nxe_qdq,
            "nxe_trace_terms": nxe_trace_terms,
        }
        metadata["array_sha256"] = {
            name: _ndarray_sha256(value) for name, value in arrays.items()
        }
        _validate_feature_cache_semantics(
            metadata,
            arrays,
            expected_identity=identity,
        )
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, staged_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".stage", dir=target.parent
        )
        os.close(fd)
        staged = Path(staged_name)
        staged.unlink()
        published_inode: tuple[int, int] | None = None
        try:
            self._atomic_npz(str(staged), **arrays)
            staged_stat = staged.stat()
            staged_sha256 = self._file_sha256(str(staged))
            self._assert_genotype_provenance_unchanged(initial_provenance)
            if overwrite:
                os.replace(staged, target)
            else:
                try:
                    os.link(staged, target)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Refusing to overwrite existing GxE feature cache: {target}."
                    ) from exc
                published_inode = (staged_stat.st_dev, staged_stat.st_ino)
            observed = target.stat(follow_symlinks=False)
            if observed.st_dev != staged_stat.st_dev or observed.st_ino != staged_stat.st_ino:
                raise RuntimeError(
                    "Published GxE feature cache was concurrently replaced before verification."
                )
            if self._file_sha256(str(target)) != staged_sha256:
                raise RuntimeError(
                    "Published GxE feature cache was modified in place before verification."
                )
        except Exception:
            if not overwrite and published_inode is not None:
                try:
                    current = target.stat(follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == published_inode:
                        target.unlink(missing_ok=True)
            raise
        finally:
            staged.unlink(missing_ok=True)
        self.feature_cache_sha256 = staged_sha256
        self.feature_cache_metadata = metadata
        self.feature_backend_provenance = dict(metadata["backend_provenance"])
        return target

    def _load_feature_cache(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(target)
        # Copy once into an owner-only temporary snapshot while hashing, then
        # parse that exact immutable byte stream.  This avoids a pathname or
        # in-place-mutation TOCTOU window without retaining the compressed
        # cache in RAM.
        scratch_parent = Path(self.outpath).expanduser().resolve().parent
        scratch_parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with tempfile.TemporaryFile(
            prefix=".gxe-cache-validated-", suffix=".tmp", dir=scratch_parent
        ) as snapshot:
            os.fchmod(snapshot.fileno(), 0o600)
            with open(target, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    snapshot.write(block)
            cache_sha256 = digest.hexdigest()
            snapshot.seek(0)
            with np.load(snapshot, allow_pickle=False) as bundle:
                expected_members = {*_FEATURE_CACHE_ARRAY_DTYPES, "metadata_json"}
                if (
                    len(bundle.files) != len(expected_members)
                    or set(bundle.files) != expected_members
                ):
                    raise ValueError(
                        "GxE feature cache contains unexpected, duplicate, or missing "
                        "schema-v2 arrays."
                    )
                metadata = json.loads(str(bundle["metadata_json"].item()))
                arrays = {
                    name: np.asarray(bundle[name]).copy()
                    for name in _FEATURE_CACHE_ARRAY_DTYPES
                }
        expected_identity = self._feature_cache_identity()
        current_variants = self.snplist[["CHR", "SNP", "BP", "A1", "A2"]]
        expected_variant_arrays = {
            "variant_chr": current_variants["CHR"].astype(str).to_numpy(dtype=np.str_),
            "variant_snp": current_variants["SNP"].astype(str).to_numpy(dtype=np.str_),
            "variant_bp": current_variants["BP"].to_numpy(dtype=np.int64),
            "variant_a1": current_variants["A1"].astype(str).to_numpy(dtype=np.str_),
            "variant_a2": current_variants["A2"].astype(str).to_numpy(dtype=np.str_),
            "annotations": np.asarray(self.annot, dtype=np.float64),
            "jackknife_ids": (
                np.asarray([], dtype=np.int32)
                if self.jackknife_ids is None
                else np.asarray(self.jackknife_ids, dtype=np.int32)
            ),
        }
        _validate_feature_cache_semantics(
            metadata,
            arrays,
            expected_identity=expected_identity,
            expected_arrays=expected_variant_arrays,
        )
        self.inv_sqrt_resvar_x_all = arrays["scale_x"]
        self.inv_sqrt_resvar_w_all = arrays["scale_w"]
        self.norm_x_all = arrays["norm_x"]
        self.norm_w_all = arrays["norm_w"]
        self.diag_nxe_x_all = arrays["diag_nxe_x"]
        self.diag_nxe_w_all = arrays["diag_nxe_w"]
        self.corr_xw_all = arrays["corr_xw"]
        self.feature_diagnostics = dict(metadata["feature_diagnostics"])
        self.score_x_all = None
        self.score_w_all = None
        self.feature_cache_sha256 = cache_sha256
        self.feature_cache_metadata = metadata
        self.feature_backend_provenance = (
            None
            if metadata.get("backend_provenance") is None
            else dict(metadata["backend_provenance"])
        )
        self.log._log(f"[gxe:cache] loaded exact projected-feature cache: {target}")

    def _write_bundle_metadata(
        self,
        score_paths: dict[str, str],
        within_jackknife: dict[str, np.ndarray] | None = None,
    ) -> tuple[str, str | None]:
        if any(x is None for x in (self.norm_x_all, self.norm_w_all, self.diag_nxe_x_all, self.diag_nxe_w_all)):
            raise RuntimeError("Feature metadata were not computed before writing the GxE bundle.")
        if self.snplist is None:
            raise ValueError("A BIM file is required for a reusable GxE summary bundle.")

        variant_digest = self._variant_digest()
        analysis_fingerprint = self._analysis_fingerprint()
        trace_nxe, trace_nxe_sq = self._nxe_reference_traces()
        diag_path = f"{self.outpath}.gxe.diag.tsv.gz"
        diag = self.snplist[["CHR", "SNP", "BP", "A1", "A2"]].copy()
        diag["NORM_X"] = self.norm_x_all
        diag["NORM_W"] = self.norm_w_all
        diag["SCALE_X"] = self.inv_sqrt_resvar_x_all
        diag["SCALE_W"] = self.inv_sqrt_resvar_w_all
        diag["DNXE_X"] = self.diag_nxe_x_all
        diag["DNXE_W"] = self.diag_nxe_w_all
        diag["CORR_XW"] = self.corr_xw_all
        for idx in range(self.nbins):
            diag[f"ANNOT_{idx}"] = np.asarray(self.annot[:, idx], dtype=np.float64)
        if self.jackknife_ids is not None:
            diag["BLOCK"] = np.asarray(self.jackknife_ids, dtype=np.int32)
        self._atomic_dataframe(diag, diag_path, sep="\t", compression="gzip", float_format="%.12g")

        manifest_path = f"{self.outpath}.gxe.ref.json"
        files = {
            **{k: self._relative_output_path(v, manifest_path) for k, v in score_paths.items()},
            "diagonal": self._relative_output_path(diag_path, manifest_path),
        }
        jackknife_payload = None
        if within_jackknife is not None:
            jackknife_path = f"{self.outpath}.gxe.jackknife.npz"
            self._atomic_npz(
                jackknife_path,
                block_labels=np.asarray(self.jackknife_labels, dtype=np.str_),
                within_xx=within_jackknife["xx"],
                within_xw=within_jackknife["xw"],
                within_wx=within_jackknife["wx"],
                within_ww=within_jackknife["ww"],
            )
            files["jackknife"] = self._relative_output_path(jackknife_path, manifest_path)
            jackknife_payload = {
                "method": "two_sided_snp_kernel_deletion",
                "num_blocks": len(self.jackknife_labels),
                "block_labels": self.jackknife_labels,
                "within_scale": "cross_product_over_rank_squared",
            }
        elif self.jackknife_ids is not None:
            jackknife_payload = {
                "method": "block_local_ldscore_deletion",
                "num_blocks": len(self.jackknife_labels),
                "block_labels": self.jackknife_labels,
                "assumption": "cross_block_directional_ld_is_negligible",
            }
        payload = {
            "kind": "summit.gxe.reference",
            "schema_version": 3,
            "analysis_fingerprint": analysis_fingerprint,
            "variant_digest": variant_digest,
            "n_samples": self.nsamp,
            "fixed_effect_rank_excluding_intercept": self.p_eff,
            "residual_rank": self.df_corr,
            "environment": self.env_name,
            "environment_transform": self.environment_transform,
            "covariates": self.cov_cols,
            "kernel_mode": self.kernel_mode,
            "genotype_scale": self.genotype_scale,
            "ld_scale": "cross_product_over_rank_squared",
            "null_corrected": False,
            "annotation_names": list(self.l2cols),
            "annotation_masses": np.asarray(self.nsnps_bin, dtype=np.float64).tolist(),
            "feature_diagnostics": self.feature_diagnostics,
            "resource_estimates": self.resource_estimates,
            "backend_provenance": self._backend_provenance("reference"),
            "feature_backend_provenance": self.feature_backend_provenance,
            "trace_nxe": trace_nxe,
            "trace_nxe_sq": trace_nxe_sq,
            "randomization": {
                "distribution": self.rand_dist,
                "num_vectors": self.nvecs,
                "seed": self.root_seed,
                "algorithm": "philox_per_probe_block_v1",
                "probe_offset": self.probe_offset,
                "probe_stop": self.probe_offset + self.nvecs,
                "dtype": str(np.dtype(self.dtype)),
                "step_size": self.step_size,
                "target_paired_sketch_gib": float(self.target_xz_mem),
                "probe_tiles": [list(tile) for tile in (self._vtiles_used or [])],
                "low_probe_jackknife_override": self.allow_low_probe_jackknife,
            },
            "genotype_files": (
                self.feature_cache_metadata["genotype_files"]
                if self.feature_cache_metadata is not None
                else (
                    getattr(self, "_active_genotype_provenance", None)
                    or self._genotype_provenance()
                )
            ),
            "files": files,
        }
        if jackknife_payload is not None:
            payload["jackknife"] = jackknife_payload
        if self.feature_cache_path is not None and self.feature_cache_sha256 is not None:
            payload["feature_cache"] = {
                "path": self._relative_output_path(self.feature_cache_path, manifest_path),
                "sha256": self.feature_cache_sha256,
            }
        payload["artifact_sha256"] = {
            key: self._file_sha256(_resolve_output)
            for key, _resolve_output in {
                **score_paths,
                "diagonal": diag_path,
                **({"jackknife": jackknife_path} if within_jackknife is not None else {}),
            }.items()
        }
        self._atomic_json(payload, manifest_path)
        reference_manifest_sha256 = self._file_sha256(manifest_path)
        self.log._log(f"Saving GxE reference manifest into: {manifest_path}")

        moments_path = None
        if self.pheno is not None:
            if self.score_x_all is None or self.score_w_all is None:
                raise RuntimeError("Phenotype scores were not computed.")
            base = self.snplist[["CHR", "SNP", "BP", "A1", "A2"]].copy()
            base["N"] = self.nsamp
            base["DF"] = self.df_corr
            base["SCORE_MODE"] = "marginal_cross_product"
            gwas_path = f"{self.outpath}.gxe.gwas.tsv.gz"
            gwis_path = f"{self.outpath}.gxe.gwis.tsv.gz"
            gwas = base.copy()
            gwas["SCORE"] = self.score_x_all
            gwis = base.copy()
            gwis["SCORE"] = self.score_w_all
            self._atomic_dataframe(gwas, gwas_path, sep="\t", compression="gzip", float_format="%.12g")
            self._atomic_dataframe(gwis, gwis_path, sep="\t", compression="gzip", float_format="%.12g")
            moments_path = f"{self.outpath}.gxe.moments.json"
            moments = {
                "kind": "summit.gxe.phenotype_moments",
                "schema_version": 3,
                "analysis_fingerprint": analysis_fingerprint,
                "variant_digest": variant_digest,
                "phenotype": self.phenotype_name,
                "n_samples": self.nsamp,
                "residual_rank": self.df_corr,
                "score_definition": "feature_transpose_residualized_y_over_sqrt_residual_rank",
                "reference_manifest_sha256": reference_manifest_sha256,
                "score_sha256": {
                    "gwas": self._file_sha256(gwas_path),
                    "gwis": self._file_sha256(gwis_path),
                },
                "q_nxe": float(np.dot(self.env * self.pheno, self.env * self.pheno)),
                "q_residual": float(np.dot(self.pheno, self.pheno)),
                "phenotype_residual_variance_fraction": self.phenotype_residual_fraction,
                "files": {
                    "gwas": self._relative_output_path(gwas_path, moments_path),
                    "gwis": self._relative_output_path(gwis_path, moments_path),
                },
            }
            self._atomic_json(moments, moments_path)
            self.log._log(f"Saving marginal GWAS/GWIS scores and NxE phenotype moments with prefix: {self.outpath}")
        return manifest_path, moments_path

    def _write_shard_metadata(
        self,
        score_paths: dict[str, str],
        within_jackknife: dict[str, np.ndarray] | None,
    ) -> str:
        if not self.shard_mode or self.feature_cache_path is None or self.feature_cache_sha256 is None:
            raise RuntimeError("Shard metadata require a validated feature cache.")
        manifest_path = f"{self.outpath}.gxe.shard.json"
        files = {
            key: self._relative_output_path(value, manifest_path)
            for key, value in score_paths.items()
        }
        artifact_paths = dict(score_paths)
        jackknife = None
        if within_jackknife is not None:
            jackknife_path = f"{self.outpath}.gxe.jackknife.npz"
            self._atomic_npz(
                jackknife_path,
                block_labels=np.asarray(self.jackknife_labels, dtype=np.str_),
                within_xx=within_jackknife["xx"],
                within_xw=within_jackknife["xw"],
                within_wx=within_jackknife["wx"],
                within_ww=within_jackknife["ww"],
            )
            files["jackknife"] = self._relative_output_path(jackknife_path, manifest_path)
            artifact_paths["jackknife"] = jackknife_path
            jackknife = {
                "method": "two_sided_snp_kernel_deletion",
                "num_blocks": len(self.jackknife_labels),
                "block_labels": list(self.jackknife_labels),
                "within_scale": "cross_product_over_rank_squared",
            }
        randomization = {
            "distribution": self.rand_dist,
            "algorithm": "philox_per_probe_block_v1",
            "seed": self.root_seed,
            "dtype": str(np.dtype(self.dtype)),
            "step_size": self.step_size,
            "num_vectors": self.nvecs,
            "probe_offset": self.probe_offset,
            "probe_stop": self.probe_offset + self.nvecs,
            "probe_tiles": [list(tile) for tile in (self._vtiles_used or [])],
        }
        artifact_sha256 = {
            key: self._file_sha256(value) for key, value in artifact_paths.items()
        }
        # A shard manifest is easy to copy and edit.  Bind the declared probe
        # identities to the exact contribution bytes in a separate sidecar,
        # then bind that sidecar into the outer manifest.  The merger also
        # hashes parsed numerical contributions, so the same B-probe result
        # cannot be counted repeatedly under relabelled intervals.
        identity_path = f"{self.outpath}.gxe.shard.identity.json"
        identity = {
            "kind": "summit.gxe.reference_shard_identity",
            "schema_version": 1,
            "analysis_fingerprint": self._analysis_fingerprint(),
            "variant_digest": self._variant_digest(),
            "annotation_digest": self._annotation_digest(),
            "jackknife_digest": self._jackknife_digest(),
            "kernel_mode": self.kernel_mode,
            "genotype_scale": self.genotype_scale,
            "ld_scale": "cross_product_over_rank_squared",
            "annotation_names": list(self.l2cols),
            "feature_cache_sha256": self.feature_cache_sha256,
            "backend_provenance": self._backend_provenance("reference_shard"),
            "feature_backend_provenance": self.feature_backend_provenance,
            "randomization": {
                key: randomization[key]
                for key in (
                    "distribution", "algorithm", "seed", "dtype", "step_size",
                    "num_vectors", "probe_offset", "probe_stop",
                )
            },
            "artifact_sha256": artifact_sha256,
        }
        self._atomic_json(identity, identity_path)
        files["identity"] = self._relative_output_path(identity_path, manifest_path)
        artifact_sha256["identity"] = self._file_sha256(identity_path)

        payload = {
            "kind": "summit.gxe.reference_shard",
            "schema_version": 2,
            "analysis_fingerprint": identity["analysis_fingerprint"],
            "variant_digest": identity["variant_digest"],
            "annotation_digest": identity["annotation_digest"],
            "jackknife_digest": identity["jackknife_digest"],
            "kernel_mode": self.kernel_mode,
            "genotype_scale": self.genotype_scale,
            "ld_scale": "cross_product_over_rank_squared",
            "annotation_names": list(self.l2cols),
            "feature_cache": {
                "path": self._relative_output_path(self.feature_cache_path, manifest_path),
                "sha256": self.feature_cache_sha256,
            },
            "randomization": randomization,
            "files": files,
            "artifact_sha256": artifact_sha256,
            "resource_estimates": self.resource_estimates,
            "backend_provenance": identity["backend_provenance"],
            "feature_backend_provenance": identity["feature_backend_provenance"],
        }
        if jackknife is not None:
            payload["jackknife"] = jackknife
        self._atomic_json(payload, manifest_path)
        self.log._log(f"Saving non-fit-able GxE reference shard manifest into: {manifest_path}")
        return manifest_path

    def _log_score_summary(self, label: str, score: np.ndarray) -> None:
        try:
            scores_df = pd.DataFrame(score, columns=self.l2cols)
            desc = scores_df.describe(percentiles=[0.25, 0.5, 0.75]).loc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            self.log._log(f"Per-bin {label} summary (count/mean/std/min/25%/50%/75%/max):")
            with pd.option_context("display.width", 140, "display.max_columns", None, "display.float_format", "{:.6f}".format):
                self.log._log(desc.to_string() + "\n")

            corr = scores_df.corr(method="pearson")
            self.log._log(f"{label} correlation matrix across bins (Pearson):")
            with pd.option_context("display.width", 140, "display.max_columns", None, "display.float_format", "{:.4f}".format):
                self.log._log("\n" + corr.to_string())
        except Exception as e:
            self.log._log(f"[warn] Failed to compute summary stats / correlation for {label}: {e}")

    def _compute_ldscore(
        self,
        compute_callback=None,
        expected_provenance=None,
        *,
        provenance_preverified: bool = False,
    ):
        if provenance_preverified and (
            compute_callback is None or expected_provenance is None
        ):
            raise ValueError(
                "Preverified genotype provenance is valid only when publishing "
                "precomputed scores against an explicit provenance record."
            )
        original_outpath = self.outpath
        final_prefix = Path(original_outpath).expanduser().resolve()
        lock_path = Path(f"{final_prefix}.gxe.bundle.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Another GxE bundle writer holds output prefix {self.outpath}: {lock_path}."
            ) from exc
        lock_stat = os.fstat(lock_descriptor)
        os.close(lock_descriptor)
        published: list[tuple[Path, int, int]] = []
        initial_provenance = None
        try:
            # Fail cheaply on normal retries, but retain no-replace publication
            # below as the authority for concurrent, non-cooperating writers.
            self._assert_output_paths_available()
            # Seal the exact PLINK bytes before the first genotype pass.  A
            # final re-hash below prevents publication of arrays computed from
            # a moving or mixed BED/BIM/FAM target.
            self._assert_construction_genotype_state()
            initial_provenance = (
                {key: dict(value) for key, value in expected_provenance.items()}
                if provenance_preverified
                else self._genotype_provenance()
            )
            if (
                expected_provenance is not None
                and initial_provenance != expected_provenance
            ):
                raise RuntimeError(
                    "PLINK input provenance changed after shared multi-environment "
                    "computation and before bundle publication."
                )
            self._active_genotype_provenance = initial_provenance
            with tempfile.TemporaryDirectory(
                prefix=".gxe-bundle-stage-", dir=final_prefix.parent
            ) as stage_name:
                stage_dir = Path(stage_name)
                os.chmod(stage_dir, 0o700)
                stage_prefix = stage_dir / final_prefix.name
                self.outpath = str(stage_prefix)
                try:
                    try:
                        from threadpoolctl import threadpool_limits

                        blas_context = threadpool_limits(limits=self.num_threads, user_api="blas")
                    except Exception:
                        blas_context = nullcontext()
                    with blas_context:
                        result = (
                            self._compute_ldscore_impl()
                            if compute_callback is None
                            else compute_callback()
                        )

                    # Output artifacts remain beside their manifest after
                    # publication, so their basename-relative paths are
                    # unchanged.  An external feature cache is outside the
                    # staging directory and must be rebased to the final
                    # manifest directory before the manifest is sealed.
                    if self.feature_cache_path is not None:
                        for suffix in (".gxe.shard.json", ".gxe.ref.json"):
                            manifest = Path(f"{stage_prefix}{suffix}")
                            if not manifest.is_file():
                                continue
                            with open(manifest, "rt", encoding="utf-8") as handle:
                                payload = json.load(handle)
                            cache_decl = payload.get("feature_cache")
                            if not isinstance(cache_decl, dict):
                                raise RuntimeError(
                                    f"Staged GxE manifest lacks feature-cache metadata: {manifest}."
                                )
                            cache_decl["path"] = os.path.relpath(
                                Path(self.feature_cache_path).resolve(),
                                start=final_prefix.parent,
                            )
                            self._atomic_json(payload, str(manifest))

                    if provenance_preverified:
                        # The multi-environment owner brackets publication with
                        # shared full-file hashes. Avoid re-reading the same BED
                        # twice for every environment while still catching
                        # inode/size/timestamp changes before each bundle seal.
                        self._assert_construction_genotype_state()
                    else:
                        self._assert_genotype_provenance_unchanged(
                            initial_provenance
                        )

                    staged_outputs = self._planned_output_paths()
                    expected_published_hashes = {
                        Path(
                            f"{final_prefix}{str(staged)[len(str(stage_prefix)) :]}"
                        ): self._file_sha256(str(staged))
                        for staged in staged_outputs
                    }

                    def publication_priority(path: Path) -> tuple[int, str]:
                        name = path.name
                        if name.endswith(".gxe.shard.identity.json"):
                            return (10, name)
                        if name.endswith(".gxe.ref.json") or name.endswith(".gxe.shard.json"):
                            return (20, name)
                        if name.endswith(".gxe.moments.json"):
                            return (30, name)
                        return (0, name)

                    def verify_published_bundle() -> None:
                        for path, device, inode in published:
                            try:
                                observed = path.stat(follow_symlinks=False)
                            except FileNotFoundError as exc:
                                raise RuntimeError(
                                    "A published GxE artifact disappeared before manifest commit: "
                                    f"{path}."
                                ) from exc
                            if observed.st_dev != device or observed.st_ino != inode:
                                raise RuntimeError(
                                    "A published GxE artifact was concurrently replaced before "
                                    f"manifest commit: {path}."
                                )
                            if self._file_sha256(str(path)) != expected_published_hashes[path]:
                                raise RuntimeError(
                                    "A published GxE artifact was modified in place before "
                                    f"manifest commit: {path}."
                                )

                    for staged in sorted(staged_outputs, key=publication_priority):
                        priority, _ = publication_priority(staged)
                        if not self.overwrite and priority >= 20:
                            # All scientific dependencies are linked first;
                            # seal a manifest only while those exact staged
                            # inodes are still present at their final names.
                            verify_published_bundle()
                        suffix = str(staged)[len(str(stage_prefix)):]
                        final = Path(f"{final_prefix}{suffix}")
                        if self.overwrite:
                            os.replace(staged, final)
                            continue
                        staged_stat = staged.stat()
                        try:
                            os.link(staged, final)
                        except FileExistsError as exc:
                            raise FileExistsError(
                                f"Refusing to overwrite concurrently created GxE artifact: {final}."
                            ) from exc
                        published.append((final, staged_stat.st_dev, staged_stat.st_ino))
                    if not self.overwrite:
                        verify_published_bundle()

                    # The CLI owns its already-open tee log.  Direct API calls
                    # still receive an owner-only log without making it part of
                    # the multi-file scientific bundle transaction.
                    staged_log = Path(f"{stage_prefix}.gxe.log")
                    final_log = Path(f"{final_prefix}.gxe.log")
                    tee_path = getattr(self.log, "_tee_path", None)
                    tee_is_final = (
                        tee_path is not None
                        and Path(tee_path).expanduser().resolve() == final_log
                    )
                    if staged_log.is_file() and not tee_is_final:
                        if self.overwrite:
                            os.replace(staged_log, final_log)
                        elif not final_log.exists():
                            try:
                                os.link(staged_log, final_log)
                            except FileExistsError:
                                # A CLI tee or another diagnostic writer won
                                # the log race; scientific bundle publication
                                # must not overwrite or remove that log.
                                pass
                    return result
                finally:
                    self.outpath = original_outpath
        except Exception:
            if not self.overwrite:
                for path, device, inode in reversed(published):
                    try:
                        observed = path.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if observed.st_dev == device and observed.st_ino == inode:
                        path.unlink(missing_ok=True)
            raise
        finally:
            self._active_genotype_provenance = None
            self.outpath = original_outpath
            try:
                observed_lock = lock_path.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (
                    observed_lock.st_dev == lock_stat.st_dev
                    and observed_lock.st_ino == lock_stat.st_ino
                ):
                    lock_path.unlink(missing_ok=True)

    def _compute_ldscore_impl(self):
        self._assert_output_paths_available()
        _validate_jackknife_probe_count(
            self.nvecs,
            self.write_jackknife and not self.shard_mode,
            self.allow_low_probe_jackknife,
        )
        self.log._log(f"num_vecs: {self.nvecs}, step_size: {self.step_size}, seed: {self.root_seed}")
        self.log._log(f"Using {self.rand_dist} random vectors.")
        if self.native_backend == "direct":
            self.log._log(
                "[backend] Bounded native direct BED algebra; Python remains the oracle/fallback."
            )
        else:
            self.log._log("[backend] Python / NumPy only (non-Mailman path).")
        self.log._log(
            "[target] Estimating the full XX/XW/WX/WW directional trace basis "
            f"for environment '{self.env_name}' (kernel_mode={self.kernel_mode})."
        )

        self.native_phase_timings = {}
        feature_started = time.perf_counter()
        try:
            if self.feature_cache_path is None:
                self.inv_sqrt_resvar_x_all, self.inv_sqrt_resvar_w_all = self._precompute_residual_variances()
                self.feature_backend_provenance = self._backend_provenance(
                    "feature_construction"
                )
            else:
                self._load_feature_cache(self.feature_cache_path)
        finally:
            self.native_phase_timings["feature_precompute"] = (
                time.perf_counter() - feature_started
            )
        blocks = self._make_compute_blocks()
        exact_jackknife = self.jackknife_ids is not None and self.shard_mode
        jackknife_blocks = (
            self._make_jackknife_compute_blocks() if exact_jackknife else []
        )
        vtiles = self._auto_vtiles()
        self._vtiles_used = list(vtiles)

        max_vt = max(vt for _, vt in vtiles)
        itemsize = int(np.dtype(self.dtype).itemsize)
        if self.native_backend == "direct" and not exact_jackknife:
            native_cap_elements = _native_execution_workspace_bytes(
                self.native_workspace_gib
            ) // 8
            source_columns_for_cap = self.nbins * max_vt
            target_columns_for_cap = 2 * self.nbins * max_vt
            # Probe identities continue to be determined by ``blocks`` and
            # ``_generate_random_group``.  The native execution width is an
            # independent implementation detail, so coalesce as many adjacent
            # probe blocks as the explicit workspace cap permits.
            desired_native_width = int(self.nsnps)

            def source_required(variants: int) -> int:
                fused_columns = 2 * source_columns_for_cap
                base = (
                    2 * self.nsamp * source_columns_for_cap
                    + 2 * (self.p_eff + 1) * source_columns_for_cap
                    + int(variants)
                    * (
                        self.nsamp
                        + 2 * source_columns_for_cap
                        + max_vt
                        + 4
                    )
                )
                integrity = max(
                    _native_gemm_integrity_workspace_elements(
                        self._native_build_info,
                        self.nsamp,
                        fused_columns,
                        int(variants),
                    ),
                    _native_gemm_integrity_workspace_elements(
                        self._native_build_info,
                        self.p_eff + 1,
                        fused_columns,
                        self.nsamp,
                    ),
                )
                return base + integrity

            def target_required(variants: int) -> int:
                base = int(variants) * (
                    self.nsamp + 2 + 4 * target_columns_for_cap
                )
                integrity = _native_gemm_integrity_workspace_elements(
                    self._native_build_info,
                    int(variants),
                    2 * target_columns_for_cap,
                    self.nsamp,
                )
                return base + integrity

            low, high = 0, desired_native_width
            while low < high:
                middle = (low + high + 1) // 2
                if max(source_required(middle), target_required(middle)) <= native_cap_elements:
                    low = middle
                else:
                    high = middle - 1
            maximum_native_width = int(low)
            if maximum_native_width < min(int(self.step_size), int(self.nsnps)):
                raise RuntimeError(
                    "The native GxE workspace cannot hold one configured compute "
                    f"block: block={min(int(self.step_size), int(self.nsnps))} variants, "
                    f"capacity={maximum_native_width}."
                )
            execution_groups = self._coalesce_contiguous_blocks(
                blocks, maximum_native_width
            )
        else:
            execution_groups = [((int(start), int(end)),) for start, end in blocks]
        max_block = max(
            group[-1][1] - group[0][0] for group in execution_groups
        )
        if self.native_backend == "direct" and len(execution_groups) < len(blocks):
            self.log._log(
                "[gxe:native] Coalesced sketch compute blocks from "
                f"{self.step_size} to at most {max_block} variants while "
                "preserving the original per-block probe seeds."
            )
        max_feature_block = min(
            int(getattr(self, "native_feature_step_size", self.step_size)),
            self.nsnps,
        )
        resident_multiplier = 4 if exact_jackknife else 2
        # A native projected panel owns S and e*S for one 2B source sketch.
        # Block and global panels are never retained at the same time.
        native_opaque_retained_multiplier = 4 if self.native_backend == "direct" else 0
        native_opaque_prepare_peak_multiplier = (
            (
                (8 if exact_jackknife else 6)
                if itemsize == np.dtype(np.float64).itemsize
                else (8 if exact_jackknife else 7)
            )
            if self.native_backend == "direct"
            else 0
        )
        target_source_columns = 2 * self.nbins * max_vt
        source_columns = self.nbins * max_vt
        q_rank = self.p_eff + 1
        global_source_columns = 2 * self.nbins * max_vt
        jackknife_source_columns = (
            2 * self.nbins * max_vt if exact_jackknife else 0
        )
        jackknife_block_bytes = (
            0
            if not exact_jackknife
            else 2 * self.nsamp * self.nbins * max_vt * itemsize
        )
        native_feature_moment_copies = (
            8 if self.native_strict_feature_moment_verification else 4
        )
        native_feature_integrity_elements = 0
        native_feature_bytes = 8 * (
            self.nsamp * max_feature_block
            + native_feature_moment_copies * q_rank * max_feature_block
            + 11 * max_feature_block
            + native_feature_integrity_elements
        )
        native_source_integrity_elements = max(
            _native_gemm_integrity_workspace_elements(
                self._native_build_info,
                self.nsamp,
                2 * source_columns,
                max_block,
            ),
            _native_gemm_integrity_workspace_elements(
                self._native_build_info,
                q_rank,
                2 * source_columns,
                self.nsamp,
            ),
        )
        native_source_bytes = 8 * (
            self.nsamp * max_block
            + 2 * max_block * source_columns
            + 2 * self.nsamp * source_columns
            + 2 * q_rank * source_columns
            + max_block * max_vt
            + 4 * max_block
            + native_source_integrity_elements
        )
        native_panel_prepare_integrity_elements = (
            _native_gemm_integrity_workspace_elements(
                self._native_build_info,
                q_rank,
                global_source_columns,
                self.nsamp,
            )
        )
        native_panel_prepare_bytes = 8 * (
            2 * self.nsamp * global_source_columns
            + q_rank * global_source_columns
            + native_panel_prepare_integrity_elements
        )
        native_target_integrity_elements = (
            _native_gemm_integrity_workspace_elements(
                self._native_build_info,
                max_block,
                2 * target_source_columns,
                self.nsamp,
            )
        )
        native_target_bytes = 8 * (
            self.nsamp * max_block
            + (6 if exact_jackknife else 4) * max_block * target_source_columns
            + 2 * max_block
            + native_target_integrity_elements
        )
        native_workspace_cap_bytes = int(self.native_workspace_gib * (1024 ** 3))
        native_execution_target_bytes = _native_execution_workspace_bytes(
            self.native_workspace_gib
        )
        native_call_max_bytes = max(
            native_feature_bytes,
            native_source_bytes,
            native_panel_prepare_bytes,
            native_target_bytes,
        )
        if (
            self.native_backend == "direct"
            and native_call_max_bytes > native_workspace_cap_bytes
        ):
            raise RuntimeError(
                "The modeled native GxE call workspace exceeds the configured cap before "
                f"the sketch pass: required={native_call_max_bytes} bytes, "
                f"limit={native_workspace_cap_bytes} bytes."
            )
        self.resource_estimates = {
            "native_direct_backend": int(self.native_backend == "direct"),
            "decoded_genotype_block_gib": float(self.nsamp * max_block * 8 / (1024 ** 3)),
            "prepared_feature_pair_gib": float(2 * self.nsamp * max_block * itemsize / (1024 ** 3)),
            "projection_product_float64_gib": float(self.nsamp * max_block * 8 / (1024 ** 3)),
            "precompute_modeled_peak_workspace_gib": float(
                6 * self.nsamp * max_block * 8 / (1024 ** 3)
            ),
            "main_projection_modeled_peak_workspace_gib": float(
                self.nsamp * max_block * (24 + itemsize) / (1024 ** 3)
            ),
            "resident_sketch_workspace_gib": float(
                resident_multiplier * self.nsamp * self.nbins * max_vt * itemsize / (1024 ** 3)
            ),
            "native_opaque_projected_panel_resident_gib": float(
                native_opaque_retained_multiplier
                * self.nsamp * self.nbins * max_vt * 8
                / (1024 ** 3)
            ),
            "native_opaque_projected_panel_prepare_peak_gib": float(
                native_opaque_prepare_peak_multiplier
                * self.nsamp * self.nbins * max_vt * 8
                / (1024 ** 3)
            ),
            "source_contribution_gib": float(
                self.nsamp * max_vt * itemsize / (1024 ** 3)
            ),
            "weighted_probe_gib": float(
                max_block * source_columns * 8 / (1024 ** 3)
            ),
            "source_columns": int(source_columns),
            "actual_global_2b_source_columns": int(global_source_columns),
            "actual_jackknife_2b_source_columns": int(jackknife_source_columns),
            "target_source_columns": int(target_source_columns),
            "native_workspace_cap_bytes": native_workspace_cap_bytes,
            "native_execution_target_bytes": native_execution_target_bytes,
            "native_modeled_max_call_workspace_gib": float(
                native_call_max_bytes / (1024 ** 3)
            ),
            "native_feature_workspace_gib": float(native_feature_bytes / (1024 ** 3)),
            "native_feature_integrity_workspace_gib": float(
                8 * native_feature_integrity_elements / (1024 ** 3)
            ),
            "native_feature_step_size": int(max_feature_block),
            "native_compute_block_size": int(max_block),
            "native_feature_basis_resident_gib": float(
                (4 * self.nsamp * q_rank)
                * 8
                / (1024 ** 3)
            ),
            "native_strict_feature_moment_verification": int(
                self.native_strict_feature_moment_verification
            ),
            "native_source_workspace_gib": float(native_source_bytes / (1024 ** 3)),
            "native_source_integrity_workspace_gib": float(
                8 * native_source_integrity_elements / (1024 ** 3)
            ),
            "native_projected_panel_prepare_workspace_gib": float(
                native_panel_prepare_bytes / (1024 ** 3)
            ),
            "native_projected_panel_integrity_workspace_gib": float(
                8 * native_panel_prepare_integrity_elements / (1024 ** 3)
            ),
            "native_target_workspace_gib": float(native_target_bytes / (1024 ** 3)),
            "native_target_integrity_workspace_gib": float(
                8 * native_target_integrity_elements / (1024 ** 3)
            ),
            "target_work_native_plus_float64_gib": float(
                2 * max_block * target_source_columns * 8 / (1024 ** 3)
            ),
            "jackknife_in_memory_block_sketch_gib": float(
                jackknife_block_bytes / (1024 ** 3)
            ),
            "blas_threads": int(self.num_threads),
            "bed_reader_threads": int(self.decode_threads),
        }
        self.log._log(
            "[gxe:resources] modeled resident sketch workspace="
            f"{self.resource_estimates['resident_sketch_workspace_gib']:.3f} GiB; "
            "jackknife block sketch in memory="
            f"{self.resource_estimates['jackknife_in_memory_block_sketch_gib']:.3f} GiB; "
            f"BLAS threads={self.num_threads}, bed-reader threads={self.decode_threads}."
        )

        accum = {
            "xx": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
            "xw": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
            "wx": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
            "ww": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
        }

        jackknife_units = sum(len(group) for group in jackknife_blocks)
        units_per_tile = (
            2 * len(execution_groups)
            if not exact_jackknife
            else 2 * jackknife_units + len(blocks)
        )
        total_units = max(1, len(vtiles) * units_per_tile)
        bar = tqdm(total=total_units, desc="GxE-LD progress", unit="task", smoothing=0.2, disable=(not self.verbose))
        within_jackknife = None
        max_native_source_leakage = 0.0
        if exact_jackknife:
            within_jackknife = {
                key: np.zeros((len(self.jackknife_labels), self.nbins, self.nbins), dtype=np.float64)
                for key in ("xx", "xw", "wx", "ww")
            }
        try:
            for v0, Vt in vtiles:
                kt = self.nbins * Vt
                sources = np.zeros(
                    (self.nsamp, 2 * kt), dtype=self.dtype, order="F"
                )
                global_x = sources[:, :kt]
                global_w = sources[:, kt:2 * kt]
                block_sources = None
                block_x = None
                block_w = None
                global_panel = None
                block_panel = None
                try:
                    source_groups = (
                        [
                            [((int(start), int(end)),) for start, end in group]
                            for group in jackknife_blocks
                        ]
                        if within_jackknife is not None
                        else [execution_groups]
                    )
                    for block_id, source_blocks in enumerate(source_groups):
                        if within_jackknife is not None:
                            block_sources = np.zeros(
                                (self.nsamp, 2 * kt), dtype=self.dtype, order="F"
                            )
                            block_x = block_sources[:, :kt]
                            block_w = block_sources[:, kt:2 * kt]

                        # Each variant contributes once to the global source;
                        # exact jackknifing mirrors it into only the current
                        # in-memory block source.
                        for constituent_blocks in source_blocks:
                            s = int(constituent_blocks[0][0])
                            e = int(constituent_blocks[-1][1])
                            Z = self._generate_random_group(
                                constituent_blocks, v_count=Vt, v_start=v0
                            )
                            if self.native_backend == "direct":
                                unique_ids, source_x, source_w = self._native_source_block(
                                    s, e, Z, None
                                )
                                if unique_ids.tolist() != [0]:
                                    raise RuntimeError(
                                        "A single-group native GxE source call returned "
                                        f"unexpected group IDs {unique_ids.tolist()}."
                                    )
                                global_x += source_x[:, :kt]
                                global_w += source_w[:, :kt]
                                if block_sources is not None:
                                    block_x += source_x[:, :kt]
                                    block_w += source_w[:, :kt]
                                del unique_ids, source_x, source_w
                            else:
                                G = self._read_genotype_block(s, e)
                                X = self._prepare_additive_block(
                                    s, e, G=G, apply_scale=True, out_dtype=self.dtype
                                )
                                W = self._prepare_interaction_block(
                                    s, e, G=G, apply_scale=True, out_dtype=self.dtype
                                )
                                annot_blk = np.asarray(self.annot[s:e], dtype=self.dtype)
                                self._accumulate_sketch_block(
                                    global_x, X, Z, annot_blk, mirror_chunk=block_x
                                )
                                self._accumulate_sketch_block(
                                    global_w, W, Z, annot_blk, mirror_chunk=block_w
                                )
                                del G, X, W, annot_blk
                            del Z
                            bar.update(1)

                        if within_jackknife is None:
                            continue

                        # Seal and evaluate this block before allocating the
                        # next one. No block sketch is written to disk.
                        if self.native_backend == "direct":
                            block_panel, leakage = self._native_prepare_projected_sources(
                                block_sources
                            )
                            max_native_source_leakage = max(
                                max_native_source_leakage, leakage
                            )
                            del block_x, block_w, block_sources
                            block_x = block_w = block_sources = None
                        for constituent_blocks in source_blocks:
                            s = int(constituent_blocks[0][0])
                            e = int(constituent_blocks[-1][1])
                            annot_left = np.asarray(self.annot[s:e], dtype=np.float64)
                            if self.native_backend == "direct":
                                work_x, work_w = self._native_target_block(
                                    s, e, block_panel
                                )
                            else:
                                G = self._read_genotype_block(s, e)
                                X = self._prepare_additive_block(
                                    s, e, G=G, apply_scale=True, out_dtype=self.dtype
                                )
                                W = self._prepare_interaction_block(
                                    s, e, G=G, apply_scale=True, out_dtype=self.dtype
                                )
                                work_x = np.asarray(X.T @ block_sources, dtype=np.float64)
                                work_w = np.asarray(W.T @ block_sources, dtype=np.float64)
                                del G, X, W
                            self._accumulate_annotation_pair_sums(
                                work_x[:, :kt], annot_left,
                                within_jackknife["xx"][block_id], Vt,
                            )
                            self._accumulate_annotation_pair_sums(
                                work_x[:, kt:2 * kt], annot_left,
                                within_jackknife["xw"][block_id], Vt,
                            )
                            self._accumulate_annotation_pair_sums(
                                work_w[:, :kt], annot_left,
                                within_jackknife["wx"][block_id], Vt,
                            )
                            self._accumulate_annotation_pair_sums(
                                work_w[:, kt:2 * kt], annot_left,
                                within_jackknife["ww"][block_id], Vt,
                            )
                            del work_x, work_w, annot_left
                            bar.update(1)
                        block_panel = None
                        if block_x is not None:
                            del block_x
                        if block_w is not None:
                            del block_w
                        if block_sources is not None:
                            del block_sources
                        block_x = block_w = block_sources = None
                        gc.collect()

                    # The global target pass is independent of jackknife block
                    # count and uses the same bounded compute blocks as ordinary
                    # genome-wide LD-score estimation.
                    if self.native_backend == "direct":
                        global_panel, leakage = self._native_prepare_projected_sources(
                            sources
                        )
                        max_native_source_leakage = max(
                            max_native_source_leakage, leakage
                        )
                        del global_x, global_w, sources
                        global_x = global_w = sources = None
                    for constituent_blocks in execution_groups:
                        s = int(constituent_blocks[0][0])
                        e = int(constituent_blocks[-1][1])
                        if self.native_backend == "direct":
                            work_x, work_w = self._native_target_block(
                                s, e, global_panel
                            )
                        else:
                            G = self._read_genotype_block(s, e)
                            X = self._prepare_additive_block(
                                s, e, G=G, apply_scale=True, out_dtype=self.dtype
                            )
                            W = self._prepare_interaction_block(
                                s, e, G=G, apply_scale=True, out_dtype=self.dtype
                            )
                            work_x = np.asarray(X.T @ sources, dtype=np.float64)
                            work_w = np.asarray(W.T @ sources, dtype=np.float64)
                            del G, X, W
                        self._accumulate_left_scores(
                            work_x[:, :kt], accum["xx"], s, e, Vt
                        )
                        self._accumulate_left_scores(
                            work_x[:, kt:2 * kt], accum["xw"], s, e, Vt
                        )
                        self._accumulate_left_scores(
                            work_w[:, :kt], accum["wx"], s, e, Vt
                        )
                        self._accumulate_left_scores(
                            work_w[:, kt:2 * kt], accum["ww"], s, e, Vt
                        )
                        del work_x, work_w
                        bar.update(1)
                finally:
                    block_panel = None
                    global_panel = None
                    if block_x is not None:
                        del block_x
                    if block_w is not None:
                        del block_w
                    if block_sources is not None:
                        del block_sources
                    if global_x is not None:
                        del global_x
                    if global_w is not None:
                        del global_w
                    if sources is not None:
                        del sources
                    gc.collect()
        finally:
            try:
                bar.close()
            except Exception:
                pass

        if self.native_backend == "direct":
            self.resource_estimates["max_native_source_projection_leakage"] = float(
                max_native_source_leakage
            )
            native_info = dict(self._native_context.info())
            repaired_gemm_columns = int(
                native_info.get("repaired_gemm_output_columns", 0)
            )
            self.resource_estimates["native_repaired_gemm_output_columns"] = (
                repaired_gemm_columns
            )
            for phase, elapsed in self.native_phase_timings.items():
                self.resource_estimates[f"native_{phase}_seconds"] = float(elapsed)
            self.log._log(
                "[gxe:native:timing] feature="
                f"{self.native_phase_timings.get('feature_precompute', 0.0):.3f}s, "
                "source calls="
                f"{self.native_phase_timings.get('source_calls', 0.0):.3f}s, "
                "panel preparation="
                f"{self.native_phase_timings.get('panel_preparation', 0.0):.3f}s, "
                "target calls="
                f"{self.native_phase_timings.get('target_calls', 0.0):.3f}s; "
                "ABFT-repaired output columns="
                f"{repaired_gemm_columns}."
            )

        scores = {name: value / float(self.nvecs) for name, value in accum.items()}
        if within_jackknife is not None:
            within_jackknife = {
                name: value / float(self.nvecs) for name, value in within_jackknife.items()
            }

        return self._finalize_ldscore_outputs(scores, within_jackknife)

    def _finalize_ldscore_outputs(
        self,
        scores: Mapping[str, np.ndarray],
        within_jackknife: Mapping[str, np.ndarray] | None,
    ):
        """Validate and write already-computed in-memory directional scores."""
        self._assert_output_paths_available()
        if set(scores) != {"xx", "xw", "wx", "ww"}:
            raise ValueError("GxE directional score set must be exactly XX/XW/WX/WW.")
        for name, value in scores.items():
            array = np.asarray(value)
            if array.shape != (self.nsnps, self.nbins):
                raise ValueError(
                    f"GxE {name.upper()} score shape {array.shape} does not match "
                    f"({self.nsnps}, {self.nbins})."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"GxE {name.upper()} scores contain NaN or infinity.")
        if within_jackknife is not None:
            if self.jackknife_ids is None:
                raise ValueError(
                    "In-memory jackknife traces were supplied without configured "
                    "GxE jackknife blocks."
                )
            if set(within_jackknife) != {"xx", "xw", "wx", "ww"}:
                raise ValueError(
                    "GxE jackknife directional trace set must be exactly XX/XW/WX/WW."
                )
            expected_shape = (
                len(self.jackknife_labels),
                self.nbins,
                self.nbins,
            )
            for name, value in within_jackknife.items():
                array = np.asarray(value)
                if array.shape != expected_shape:
                    raise ValueError(
                        f"GxE {name.upper()} jackknife trace shape {array.shape} "
                        f"does not match {expected_shape}."
                    )
                if not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"GxE {name.upper()} jackknife traces contain NaN or infinity."
                    )

        # Persist the raw realized-sample directional scores.  A common XX-like
        # M/r subtraction is not a valid finite-sample correction for XW/WX,
        # whose same-SNP cross-products are neither zero nor one.  The normal
        # equations below need the raw kernel Gram traces in every direction.
        annot64 = np.asarray(self.annot, dtype=np.float64)
        max_cross_abs = 0.0
        max_cross_rel = 0.0
        r2 = float(self.df_corr * self.df_corr)
        for left_bin in range(self.nbins):
            for source_bin in range(self.nbins):
                forward = (
                    r2
                    * np.dot(annot64[:, left_bin], scores["xw"][:, source_bin])
                    / (self.nsnps_bin[left_bin] * self.nsnps_bin[source_bin])
                )
                reverse = (
                    r2
                    * np.dot(annot64[:, source_bin], scores["wx"][:, left_bin])
                    / (self.nsnps_bin[source_bin] * self.nsnps_bin[left_bin])
                )
                delta = abs(float(forward - reverse))
                max_cross_abs = max(max_cross_abs, delta)
                max_cross_rel = max(
                    max_cross_rel,
                    delta / max(abs(float(forward)), abs(float(reverse)), np.finfo(float).tiny),
                )
        self.feature_diagnostics["max_xw_wx_trace_asymmetry_absolute"] = max_cross_abs
        self.feature_diagnostics["max_xw_wx_trace_asymmetry_relative"] = max_cross_rel
        self.log._log(
            "[gxe:invariants] XW/WX aggregate trace asymmetry "
            f"max_abs={max_cross_abs:.3e}, max_rel={max_cross_rel:.3e}."
        )

        self.gxx_ldscore = np.asarray(scores["xx"], dtype=np.float64)
        self.gxe_ldscore = np.asarray(scores["xw"], dtype=np.float64)
        self.exg_ldscore = np.asarray(scores["wx"], dtype=np.float64)
        self.gee_ldscore = np.asarray(scores["ww"], dtype=np.float64)

        score_paths = {
            "xx": f"{self.outpath}.gxx.ldscore.gz",
            "xw": f"{self.outpath}.gxe.ldscore.gz",
            "wx": f"{self.outpath}.exg.ldscore.gz",
            "ww": f"{self.outpath}.gee.ldscore.gz",
        }
        labels = {
            "xx": "additive-additive (X<-X)",
            "xw": "additive-interaction (X<-W)",
            "wx": "interaction-additive (W<-X)",
            "ww": "interaction-interaction (W<-W)",
        }
        for name, path in score_paths.items():
            self.log._log(f"Saving {labels[name]} trace scores into: {path}")
            self._save_score_file(path, scores[name])
            self._log_score_summary(f"{labels[name]} LD scores", scores[name])
        if self.shard_mode:
            self._write_shard_metadata(score_paths, within_jackknife=within_jackknife)
        else:
            self._write_bundle_metadata(score_paths, within_jackknife=within_jackknife)

        try:
            col_sums = pd.Series(self.nsnps_bin, index=self.l2cols)
            lines = ["Annotation Column Sums"] + [f"{k:<35} {v:.6f}" for k, v in col_sums.items()]
            self.log._log("\n" + "\n".join(lines))
        except Exception as e:
            self.log._log(f"[warn] Failed to report annotation column sums: {e}")

        self.end_time = utils._get_time()
        self.log._log("Calculation of genome-wide GxE LD scores ended at " + utils._get_timestr(self.end_time))
        self.runtime = self.end_time - self.start_time
        self.log._log(
            "Runtime: " + format(self.runtime, ".3f") +
            f" s ({self.runtime // 3600} hr {(self.runtime % 3600) // 60} m {(self.runtime % 60):.3f} s)"
        )
        self.log._save_log(self.outpath + ".gxe.log")
        log_path = Path(self.outpath + ".gxe.log")
        if log_path.is_file():
            os.chmod(log_path, 0o600)
GenomewideGxELDScore = GenomewideEnvLDScore
