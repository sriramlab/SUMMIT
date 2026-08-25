from __future__ import annotations

import gc
import gzip
import io
import json
import math
import os
import re
import stat
import tempfile
import time
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from bed_reader import open_bed
from threadpoolctl import threadpool_info
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


def _non_blas_fp64_inner_product(left: np.ndarray, right: np.ndarray) -> float:
    """Use a compensated fixed-order diagnostic sum without entering BLAS."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1:
        raise ValueError("Diagnostic inner-product operands must be one-dimensional.")
    if left_array.shape != right_array.shape:
        raise ValueError("Diagnostic inner-product operands must have equal length.")
    products = np.multiply(left_array, right_array, dtype=np.float64)
    return float(math.fsum(products))


_REFERENCE_FLOAT_FORMAT = "%.17g"
_NATIVE_GEMM_INTEGRITY_CHECKS = 8
_NATIVE_GEMM_CHECK_MINIMUM_FLOPS = 1_000_000_000
_NATIVE_PREFERRED_CALL_WORKSPACE_BYTES = 3 * 1024**3
_NATIVE_PREFERRED_FEATURE_WORKSPACE_BYTES = 3 * 512 * 1024**2


def _native_gemm_integrity_workspace_elements(
    build_info: Mapping | None, m: int, n: int, k: int
) -> int:
    """Mirror native checksum and non-reconstructable operand protection."""
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
    # The right operand is non-reconstructable and is snapshotted. The decoded
    # left operand is fingerprinted and the native block is retried after a
    # fresh decode if the affected host mutates it.
    return checks + int(k) * int(n)


def _native_execution_workspace_bytes(native_workspace_gib: float) -> int:
    """Use the configured workspace as a ceiling, not an allocation target."""
    configured = int(float(native_workspace_gib) * (1024**3))
    return min(configured, _NATIVE_PREFERRED_CALL_WORKSPACE_BYTES)


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

    if (
        str(build_info.get("blas_runtime_isolation", "")).strip().lower()
        == "private_static"
        and build_info.get("gemm_integrity_enabled") is False
    ):
        return False, (
            "deterministic tiled feature moments; serialized fixed-thread "
            "private OpenBLAS trace GEMMs"
        )
    vendor = str(build_info.get("blas_vendor", "")).strip()
    if vendor.lower() in {"openblas", "blis"}:
        return False, (
            "deterministic tiled feature GEMMs; eight-check ABFT over "
            f"serialized-entry internally threaded {vendor} trace GEMMs with "
            "fresh-decode fallback"
        )
    return False, (
        "deterministic disjoint-output tiled GEMMs independent of "
        f"{build_info.get('blas_vendor')}"
    )


def _validate_native_blas_runtime(build_info: Mapping) -> dict[str, str | int]:
    """Validate the selected isolated or process-shared native BLAS runtime."""
    records = []
    for raw in threadpool_info():
        if str(raw.get("user_api", "")).strip().lower() != "blas":
            continue
        path = str(raw.get("filepath", "")).strip()
        records.append(
            {
                "internal_api": str(raw.get("internal_api", "")).strip().lower(),
                "version": str(raw.get("version", "")).strip(),
                "path": os.path.realpath(path) if path else "",
                "num_threads": int(raw.get("num_threads", 0)),
                "threading_layer": str(raw.get("threading_layer", "")).strip().lower(),
            }
        )
    unique = {
        (record["internal_api"], record["version"], record["path"]): record
        for record in records
    }
    isolation = str(build_info.get("blas_runtime_isolation", "")).strip().lower()
    if isolation == "private_static":
        execution_mode = str(build_info.get("gemm_execution_mode", ""))
        private_backend = str(
            build_info.get("private_blas_backend", "")
        ).strip().lower()
        expected_modes = {
            "openblas": "serialized_fixed_private_openblas",
            "upstream_blis": "serialized_fixed_private_blis",
        }
        if execution_mode != expected_modes.get(private_backend):
            raise RuntimeError(
                "The private GxE BLAS does not declare fixed serialized execution."
            )
        runtime_config = str(build_info.get("blas_runtime_config", "")).split()
        if private_backend == "openblas":
            version = (
                runtime_config[1]
                if len(runtime_config) >= 2
                and runtime_config[0].lower() == "openblas"
                else ""
            )
            version_match = re.match(
                r"^(\d+)\.(\d+)\.(\d+)(?:\D.*)?$", version
            )
            if version_match is None or tuple(
                int(value) for value in version_match.groups()
            ) < (0, 3, 31):
                raise RuntimeError(
                    "The private GxE backend requires OpenBLAS 0.3.31 or newer; "
                    f"embedded {version!r}."
                )
            internal_api = "openblas"
        else:
            if (
                str(build_info.get("blas_vendor", "")).strip().lower()
                != "blis"
                or build_info.get("gemm_integrity_enabled") is not True
                or build_info.get("gemm_vendor_entry_outer_openmp_guard") is not True
                or build_info.get("blas_runtime_owner_thread_configured") is not True
            ):
                raise RuntimeError(
                    "The private GxE BLIS runtime lacks its guarded integrity contract."
                )
            version = (
                runtime_config[1]
                if len(runtime_config) >= 2
                and runtime_config[0].lower() == "blis"
                else ""
            )
            if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version) is None:
                raise RuntimeError(
                    "The private GxE BLIS runtime lacks an exact version."
                )
            config_family = str(
                build_info.get("private_blas_config_family", "")
            ).strip()
            if (
                not config_family
                or runtime_config[2:] != [f"config={config_family}"]
                or str(build_info.get("blas_runtime_corename", ""))
                != config_family
            ):
                raise RuntimeError(
                    "The private GxE BLIS architecture/configuration is inconsistent."
                )
            if (
                build_info.get("blas_runtime_tls_enabled") is not True
                or build_info.get("blas_runtime_owner_thread_enforced") is not True
                or build_info.get("blas_runtime_environment_immutable") is not True
                or build_info.get("blas_runtime_environment_contract")
                != "blis_process_start_v1"
            ):
                raise RuntimeError(
                    "The private GxE BLIS runtime lacks its immutable TLS/owner contract."
                )
            internal_api = "blis"
        native_threads = build_info.get("blas_runtime_threads")
        if (
            isinstance(native_threads, bool)
            or not isinstance(native_threads, (int, np.integer))
            or int(native_threads) <= 0
        ):
            raise RuntimeError("The private GxE BLAS reports an invalid thread count.")
        threading_layer = str(
            build_info.get("blas_runtime_threading_layer", "")
        ).strip().lower()
        allowed_threading_layers = (
            {"pthreads", "openmp"}
            if private_backend == "openblas"
            else {"pthreads"}
        )
        if threading_layer not in allowed_threading_layers:
            raise RuntimeError(
                "The private GxE BLAS reports an unsupported threading layer: "
                f"{threading_layer!r}."
            )
        if private_backend == "upstream_blis":
            if (
                build_info.get("blas_runtime_worker_affinity_policy")
                != "inherit_authenticated_selected_cpu_set_per_call"
            ):
                raise RuntimeError(
                    "The private GxE BLIS runtime lacks its authenticated "
                    "pthread affinity contract."
                )
            strategy = str(
                build_info.get("blas_runtime_thread_strategy", "")
            ).strip().lower()
            ways = build_info.get("blas_runtime_thread_ways")
            if strategy not in {"automatic", "manual"} or not isinstance(
                ways, Mapping
            ) or set(ways) != {"jc", "pc", "ic", "jr", "ir"}:
                raise RuntimeError(
                    "The private GxE BLIS runtime reports an invalid thread strategy."
                )
            normalized_ways = []
            for name in ("jc", "pc", "ic", "jr", "ir"):
                value = ways[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) <= 0
                ):
                    raise RuntimeError(
                        "The private GxE BLIS runtime reports invalid loop ways."
                    )
                normalized_ways.append(int(value))
            if strategy == "automatic" and normalized_ways != [1, 1, 1, 1, 1]:
                raise RuntimeError(
                    "The automatic private GxE BLIS runtime has manual loop ways."
                )
            if strategy == "manual" and (
                normalized_ways[1] != 1
                or math.prod(normalized_ways) != int(native_threads)
            ):
                raise RuntimeError(
                    "The manual private GxE BLIS loop ways violate its thread contract."
                )
        return {
            "internal_api": internal_api,
            "version": version,
            "path": "private-static gxeldcore image",
            "num_threads": int(native_threads),
            "isolation": "private_static",
            "process_blas_runtimes": len(unique),
            "threading_layer": threading_layer,
        }
    if len(unique) != 1:
        descriptions = sorted(
            f"{api or 'unknown'} {version or 'unknown'} at {path or 'unknown'}"
            for api, version, path in unique
        )
        raise RuntimeError(
            "The direct GxE backend requires exactly one process-wide BLAS runtime; "
            f"observed {len(unique)}: {descriptions}. Install NumPy and SUMMIT "
            "against the same BLAS before running native GxE traces."
        )
    record = next(iter(unique.values()))
    record["isolation"] = "process_shared"
    record["process_blas_runtimes"] = len(unique)
    expected_vendor = str(build_info.get("blas_vendor", "")).strip().lower()
    observed_vendor = str(record["internal_api"])
    if expected_vendor and expected_vendor not in {"all", "generic"}:
        aliases = {
            "openblas": "openblas",
            "intel10_64lp": "mkl",
            "intel10_64ilp": "mkl",
            "intel10_64_dyn": "mkl",
            "mkl": "mkl",
            "blis": "blis",
        }
        expected_api = aliases.get(expected_vendor, expected_vendor)
        if expected_api != observed_vendor:
            raise RuntimeError(
                "The direct GxE extension and loaded BLAS runtime disagree: "
                f"built for {expected_vendor!r}, loaded {observed_vendor!r}."
            )
        if expected_api == "openblas":
            version_match = re.match(
                r"^(\d+)\.(\d+)\.(\d+)(?:\D.*)?$", str(record["version"])
            )
            if version_match is None or tuple(
                int(value) for value in version_match.groups()
            ) < (0, 3, 31):
                raise RuntimeError(
                    "The direct GxE backend requires OpenBLAS 0.3.31 or newer; "
                    f"loaded {record['version']!r}. OpenBLAS 0.3.30 contained a "
                    "parallel-GEMM race."
                )
        runtime_config = str(build_info.get("blas_runtime_config", "")).split()
        expected_version = (
            runtime_config[1]
            if expected_api == "openblas"
            and len(runtime_config) >= 2
            and runtime_config[0].lower() == "openblas"
            else ""
        )
        if expected_version and expected_version != record["version"]:
            raise RuntimeError(
                "The direct GxE extension and loaded OpenBLAS version disagree: "
                f"built against {expected_version!r}, loaded {record['version']!r}."
            )
    return record


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


_AUTO_STEP_SIZE_CANONICAL_WIDTH = 8192


def _auto_reference_step_size(nsnps: int) -> int:
    """Return the deterministic auto-selected canonical block width."""
    return max(1, min(int(nsnps), _AUTO_STEP_SIZE_CANONICAL_WIDTH))


def _canonicalize_annotation_matrix(raw) -> np.ndarray:
    """Return the canonical contiguous binary64 annotation matrix.

    The canonical matrix defines the scientific estimand; the estimator
    ``dtype`` option controls randomized probe/sketch retained storage only
    and never rounds annotation values.  The conversion must be exact and the
    canonical matrix is validated after its final conversion.
    """
    raw = np.asarray(raw)
    canonical = np.ascontiguousarray(raw, dtype=np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        if np.any(raw > 2 ** 53) or np.any(raw < -(2 ** 53)):
            raise ValueError(
                "Integer annotation values above 2**53 are not exactly "
                "representable in the canonical binary64 annotation matrix."
            )
    elif (
        np.issubdtype(raw.dtype, np.floating)
        and raw.dtype.itemsize > 8
        and not np.array_equal(canonical.astype(raw.dtype), raw, equal_nan=True)
    ):
        raise ValueError(
            "Extended-precision annotation values change under binary64 "
            "conversion; supply binary64-exact annotation values."
        )
    if not np.all(np.isfinite(canonical)):
        raise ValueError("Annotation values must all be finite; NaN/Inf values are not accepted.")
    if np.any(canonical < 0.0):
        raise ValueError("Annotation values must be non-negative.")
    return canonical


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
    # The process cap is installed once before numerical runtimes start. Do
    # not resize a process-global BLAS pool around this small factorization.
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
    return_common_basis: bool = False,
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
    C_common = _orthonormalize_columns(cov_base)
    design_base = np.column_stack([cov_base, env_vec.reshape(-1, 1)])
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

    result = (
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
    if return_common_basis:
        return result + (np.asfortranarray(C_common),)
    return result


class GenomewideEnvLDScore:
    r"""
    Estimate the four directional additive/interaction trace panels required
    by one-environment projected-kernel normal equations using randomized
    sketches.

    This mature variant-probe source/target path is the systems template for
    the generalized per-variant GxE LD-score estimator. The generalized path
    has a distinct artifact/contract and must not inherit this class's fixed
    X/W scientific layout or separate post-projection X/W scales.

    For each annotation bin k, the module estimates
        ell^{WW}_{jk} = sum_{j' in S_k} (r^{WW}_{jj'})^2,
        ell^{XW}_{jk} = sum_{j' in S_k} (r^{XW}_{jj'})^2,
    X and W are projected onto the same fixed-effect residual space, including
    the environment main effect.  The default ``standardized`` SUMMIT mode
    rescales every valid projected column to squared norm ``rank(P)``.  The
    explicit ``genie`` compatibility mode retains each projected column's
    natural norm after HWE scaling.

    Although ``R_WX = R_XW.T``, the per-variant XW and WX panels are not
    duplicates: they are respectively annotation-weighted squared row and
    column norms of ``R_XW``.  Their annotation-aggregated normal-equation
    entries agree after swapping the left/source bins.  Both directional
    panels are retained for per-variant summaries and exact deletion
    bookkeeping, while the native target path obtains both from one shared
    GEMM rather than repeating the expensive product.

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
        gxe_total_memory_gib="auto",
        device="cpu",
        impute_method: str = "mean",
        kernel_mode: str = "standardized_projected",
        genotype_scale: str | None = None,
        pheno_path: str | None = None,
        pheno_col: str | None = None,
        missing_values: Sequence[str] = ("-9", "NA", "NaN", "nan", ".", "None", "null"),
        overwrite: bool = False,
        probe_offset: int = 0,
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
        if isinstance(step_size, str) and step_size.strip().lower() == "auto":
            self.step_size = _auto_reference_step_size(self.nsnps)
            self.step_size_selection = "auto_v1"
        else:
            self.step_size = int(step_size)
            self.step_size_selection = "explicit"
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
        self.gxe_total_memory_request = utils.parse_memory_budget(
            gxe_total_memory_gib
        )
        self.log._log(
            f"[memory] sketch budget={self.target_xz_mem:.3f} GiB "
            f"(mode={self.memory_budget['mode']}); total-process budget request="
            f"{self.gxe_total_memory_request}."
        )
        self.device = str(device).strip().lower()
        self.impute_method = str(impute_method).strip().lower()
        if self.impute_method != "mean":
            raise ValueError("The current Python GxE LD-score implementation supports only impute_method='mean'.")
        self.feature_convention = str(kernel_mode).strip().lower()
        if self.feature_convention not in {
            "standardized_projected",
            "raw_projected",
        }:
            raise ValueError(
                "kernel_mode must be 'standardized_projected' or "
                "'raw_projected'."
            )
        self.kernel_mode = self.feature_convention
        if genotype_scale is None:
            genotype_scale = (
                "sample"
                if self.kernel_mode == "standardized_projected"
                else "hwe"
            )
        self.genotype_scale = str(genotype_scale).strip().lower()
        if self.genotype_scale not in ("hwe", "sample"):
            raise ValueError("genotype_scale must be 'hwe' or 'sample'.")
        self.pheno_path = None if pheno_path is None else str(pheno_path)
        self.pheno_col = pheno_col
        self.missing_values = tuple(str(x) for x in missing_values)
        self.overwrite = bool(overwrite)
        self.probe_offset = int(probe_offset)
        if self.probe_offset < 0 or self.probe_offset >= 2**64:
            raise ValueError(
                f"probe_offset must be in [0, 2**64); got {self.probe_offset}."
            )
        if self.probe_offset + self.nvecs > 2**64:
            raise ValueError("The requested GxE probe interval exceeds the uint64 identity space.")
        requested_native_backend = str(native_backend).strip().lower()
        if requested_native_backend not in ("python", "direct"):
            raise ValueError("native_backend must be 'python' or 'direct'.")
        if self.genotype_format == "pgen":
            unsupported = []
            if requested_native_backend != "python":
                unsupported.append("native-direct execution")
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

        self._gxe_group_worker_authenticated = bool(
            low_level is not None
            and low_level.get("_gxe_group_worker_authenticated") is True
        )
        self.cpu_placement: dict | None = None
        self.cpu_placement_complete = False
        try:
            caller_affinity_threads = len(os.sched_getaffinity(0))
        except Exception:
            caller_affinity_threads = max(1, os.cpu_count() or 1)
        if low_level is not None:
            # The authenticated socket worker has one singleton affinity on
            # its Python caller after libgomp binding. Its complete native
            # team attestation is the only permitted capacity override.
            from .gw_ldscore import (
                _validate_cpu_placement_attestation,
                _validated_thread_capacity,
                apply_env as _apply_env,
            )

            validated_capacity = _validated_thread_capacity(
                low_level, caller_affinity_threads
            )
            if self._gxe_group_worker_authenticated:
                placement = _validate_cpu_placement_attestation(
                    low_level.get("_gxe_cpu_placement"),
                    expected_cpu_ids=low_level.get("_gxe_worker_cpu_ids"),
                    expected_threads=low_level.get("num_threads"),
                )
                self.cpu_placement = placement
                self.cpu_placement_complete = True
            try:
                # Keep the pure-Python GxE module importable without the native
                # GWLD extension; the low-level helper is needed only here.
                actual_threads = _apply_env(low_level)
            except Exception as e:
                if self._gxe_group_worker_authenticated:
                    raise RuntimeError(
                        "Authenticated GxE group-worker runtime setup failed."
                    ) from e
                self.log._log(f"[threads] apply_env failed in GxE LD-score setup (non-fatal): {e}")
                actual_threads = None
        else:
            validated_capacity = caller_affinity_threads
            actual_threads = None
        if num_threads is not None and int(num_threads) > 0:
            requested_threads = int(num_threads)
            if (
                self._gxe_group_worker_authenticated
                and requested_threads != validated_capacity
            ):
                raise RuntimeError(
                    "Authenticated GxE group-worker threads disagree with its "
                    "verified OpenMP capacity."
                )
            self.num_threads = max(1, min(requested_threads, validated_capacity))
        elif actual_threads is not None:
            self.num_threads = int(actual_threads)
        else:
            self.num_threads = max(1, os.cpu_count() or 1)
        decode_threads = os.environ.get("SUMMIT_DECODE_THREADS")
        if decode_threads is not None and decode_threads.isdigit():
            self.decode_threads = max(1, min(self.num_threads, int(decode_threads)))
        else:
            self.decode_threads = self.num_threads

        # ``dtype`` selects the retained storage precision of randomized
        # probe/sketch panels only.  The canonical annotation matrix, its
        # masses, and every native arithmetic path stay binary64 regardless.
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
            C_common,
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
            return_common_basis=True,
        )
        self.row_sel = np.asarray(keep_idx_global, dtype=int)
        self.env = np.asarray(env_vec, dtype=np.float64)
        self.env_name = env_name
        self.C_int = np.asarray(C_int, dtype=np.float64, order="F")
        self.cov_R_int = np.asarray(cov_R_int, dtype=np.float64, order="F")
        self.C_common = np.asarray(C_common, dtype=np.float64, order="F")
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
        self.genotype_missing_call_count = np.zeros(self.nsnps, dtype=np.int64)
        self.genotype_missing_environment_correlation = np.zeros(
            self.nsnps, dtype=np.float64
        )
        self.genotype_missing_phenotype_correlation = np.zeros(
            self.nsnps, dtype=np.float64
        )
        self._missingness_warning_logged = False
        self.resource_estimates: dict[str, float | int] = {}
        self.population_same_individual_products: np.ndarray | None = None
        self._vtiles_used: list[tuple[int, int]] | None = None
        self.native_workspace_gib = float(native_workspace_gib)
        if not np.isfinite(self.native_workspace_gib) or self.native_workspace_gib <= 0.0:
            raise ValueError("native_workspace_gib must be positive and finite.")
        self.native_target_panel_columns = int(native_target_panel_columns)
        if self.native_target_panel_columns <= 0:
            raise ValueError("native_target_panel_columns must be positive.")
        self.native_backend = requested_native_backend
        self._native_context = None
        self._native_build_info: dict | None = None
        self.native_blas_runtime_record: dict[str, str | int] | None = None
        self.native_strict_feature_moment_verification = False
        self.native_feature_moment_integrity_reason = "Python backend"
        self.native_phase_timings: dict[str, float] = {}
        self.performance_phase_timings: dict[str, dict[str, float | int]] = {}
        if self.native_backend != "python":
            unsupported = []
            if self.nbins != 1:
                unsupported.append(f"K={self.nbins} annotations")
            if self.kernel_mode != "standardized_projected":
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
                    configure_blas_threads = getattr(
                        _gxeldcore, "configure_blas_threads", None
                    )
                    if callable(configure_blas_threads):
                        configured_threads = int(
                            configure_blas_threads(self.num_threads)
                        )
                        if configured_threads != self.num_threads:
                            raise RuntimeError(
                                "The direct GxE extension configured an unexpected "
                                "BLAS thread count."
                            )
                    build_info = dict(_gxeldcore.build_info())
                    runtime_record = _validate_native_blas_runtime(build_info)
                    self._native_build_info = build_info
                    self.native_blas_runtime_record = runtime_record
                    self.log._log(
                        "[gxe:native] Verified BLAS runtime: "
                        f"{runtime_record['internal_api']} "
                        f"{runtime_record['version']} at {runtime_record['path']} "
                        f"(isolation={runtime_record['isolation']})."
                    )
                    (
                        self.native_strict_feature_moment_verification,
                        self.native_feature_moment_integrity_reason,
                    ) = _native_strict_feature_moment_verification_policy(build_info)
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
                        blas_threads=int(self.num_threads),
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
                        or int(context_info["blas_threads"]) != self.num_threads
                        or bool(context_info["strict_feature_moment_verification"])
                        != self.native_strict_feature_moment_verification
                    ):
                        self._native_context.close()
                        self._native_context = None
                        raise RuntimeError(
                            "The bounded native GxE context disagrees with validated Python dimensions/state."
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
        # The canonical annotation matrix defines the scientific estimand and
        # is always contiguous binary64.  The estimator ``dtype`` option
        # controls randomized probe/sketch retained storage only; it never
        # rounds annotation values, masses, or sqrt-annotation source weights.
        if annot_path is None:
            self.annot = np.ascontiguousarray(
                np.ones((self.nsnps, 1), dtype=np.float64)
            )
            self.nbins = 1
            self.l2cols = ["L2_0"]
            self.is_continuous = False
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
            self.log._log("Calculating genome-wide (non-partitioned) GxE LD scores")
            self.log._log(f"Number of total SNPs: {self.nsnps}, annotation shape: {self.annot.shape}")
            return

        parsed_ldsc = False
        # The parsed decimal text defines the binary64 estimand, so parsing
        # must be correctly rounded; the default pandas float parser can be
        # one ulp off for extreme values.
        df = pd.read_csv(
            annot_path,
            sep=r"\s+",
            compression="infer",
            dtype={"CHR": str, "SNP": str},
            float_precision="round_trip",
        )
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
        self.annot = _canonicalize_annotation_matrix(self.annot)
        self.is_continuous = not np.all(np.isin(np.unique(self.annot), [0.0, 1.0]))
        # Masses come from the canonical binary64 matrix so the manifest,
        # diagonal table, native inputs, full traces, and deleted traces all
        # use byte-identical binary64 annotation weights.
        self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
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
        workspace_elements = min(
            _native_execution_workspace_bytes(self.native_workspace_gib),
            _NATIVE_PREFERRED_FEATURE_WORKSPACE_BYTES,
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

    def _record_performance_phase(
        self,
        phase: str,
        wall_seconds: float,
        process_cpu_seconds: float,
    ) -> None:
        """Accumulate bounded wall/CPU evidence for streamed phase accounting."""
        timings = getattr(self, "performance_phase_timings", None)
        if timings is None:
            # Some focused reader tests construct this class with ``__new__``.
            timings = {}
            self.performance_phase_timings = timings
        record = timings.setdefault(
            str(phase),
            {"wall_seconds": 0.0, "process_cpu_seconds": 0.0, "calls": 0},
        )
        record["wall_seconds"] = float(record["wall_seconds"]) + float(
            wall_seconds
        )
        record["process_cpu_seconds"] = float(
            record["process_cpu_seconds"]
        ) + float(process_cpu_seconds)
        record["calls"] = int(record["calls"]) + 1

    @contextmanager
    def _performance_phase(self, phase: str):
        """Time one output/read interval without changing its exception behavior."""
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        try:
            yield
        finally:
            self._record_performance_phase(
                phase,
                time.perf_counter() - wall_started,
                time.process_time() - cpu_started,
            )

    def _read_genotype_block(
        self, blk_start: int, blk_end: int, *, memory_order: str = "F"
    ) -> np.ndarray:
        memory_order = str(memory_order).strip().upper()
        if memory_order not in {"F", "C"}:
            raise ValueError("Genotype block memory_order must be 'F' or 'C'.")
        read_wall_started = time.perf_counter()
        read_cpu_started = time.process_time()
        bound_owner = None
        bound_array = None
        bound_byte_count = None
        bound_allocation_evidence = None
        bound_selected_nodes = None
        if getattr(self, "genotype_format", "bed") == "pgen":
            if memory_order != "F":
                raise RuntimeError(
                    "Direct row-major genotype decoding is currently limited to BED input."
                )
            if self._pgen_reader is None:
                raise RuntimeError("PGEN reader is closed.")
            G = self._pgen_reader.read_standardized_block(blk_start, blk_end)
        else:
            bound_nodes = getattr(
                self, "_native_numa_bound_decode_nodes", None
            )
            if bound_nodes is not None:
                if memory_order != "F":
                    raise RuntimeError(
                        "NUMA-bound direct BED decoding currently requires "
                        "Fortran-order genotype blocks."
                    )
                from bed_reader.bed_reader import read_f64

                from .._early_numa import (
                    allocate_numa_bound_anonymous_buffer,
                    verify_numa_bound_anonymous_buffer,
                )

                if not getattr(self, "_native_parallel_standardization", False):
                    raise RuntimeError(
                        "NUMA-bound direct BED decoding requires native in-place "
                        "genotype standardization."
                    )
                selected_nodes = tuple(bound_nodes)
                rows = int(self.nsamp)
                columns = int(blk_end - blk_start)
                if rows <= 0 or columns <= 0:
                    raise RuntimeError(
                        "NUMA-bound direct BED decoding requires a nonempty block."
                    )
                byte_count = rows * columns * np.dtype(np.float64).itemsize
                owner, allocation_evidence = (
                    allocate_numa_bound_anonymous_buffer(
                        byte_count, selected_nodes
                    )
                )
                G = np.ndarray(
                    (rows, columns),
                    dtype=np.float64,
                    buffer=owner,
                    order="F",
                )
                bound_owner = owner
                bound_array = G
                bound_byte_count = byte_count
                bound_allocation_evidence = allocation_evidence
                bound_selected_nodes = selected_nodes
                iid_index = getattr(
                    self, "_native_numa_bound_iid_index", None
                )
                if iid_index is None:
                    iid_index = np.ascontiguousarray(
                        self.row_sel, dtype=np.intp
                    )
                    if (
                        iid_index.ndim != 1
                        or iid_index.shape[0] != rows
                        or np.any(iid_index < 0)
                        or np.any(iid_index >= int(self.G.iid_count))
                    ):
                        raise RuntimeError(
                            "The NUMA-bound BED sample index is malformed."
                        )
                    self._native_numa_bound_iid_index = iid_index
                sid_index = np.arange(
                    blk_start, blk_end, dtype=np.intp
                )
                read_f64(
                    str(self.G.filepath),
                    iid_count=int(self.G.iid_count),
                    sid_count=int(self.G.sid_count),
                    is_a1_counted=bool(self.G.count_A1),
                    iid_index=iid_index,
                    sid_index=sid_index,
                    val=G,
                    num_threads=int(self.decode_threads),
                )
            else:
                indexer = np.s_[self.row_sel, blk_start:blk_end]
                if memory_order == "C":
                    try:
                        G = self.G.read(
                            index=indexer,
                            dtype=np.float64,
                            order="C",
                            num_threads=self.decode_threads,
                        )
                    except TypeError as exc:
                        raise RuntimeError(
                            "The optimized FP64 target layout requires a BED reader "
                            "that decodes directly into C order; refusing an implicit "
                            "N-by-block layout copy."
                        ) from exc
                else:
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

        self._record_performance_phase(
            "bed_read_decode",
            time.perf_counter() - read_wall_started,
            time.process_time() - read_cpu_started,
        )
        standardize_wall_started = time.perf_counter()
        standardize_cpu_started = time.process_time()

        G = np.asarray(G, dtype=np.float64)
        if G.shape == (blk_end - blk_start, self.nsamp):
            G = G.T
        if G.shape != (self.nsamp, blk_end - blk_start):
            raise RuntimeError(
                f"Unexpected genotype block shape {G.shape} for block [{blk_start}:{blk_end}); "
                f"expected ({self.nsamp}, {blk_end - blk_start})."
            )
        if memory_order == "C" and not G.flags.c_contiguous:
            raise RuntimeError(
                "The BED reader did not honor direct C-order genotype decoding; "
                "refusing an implicit N-by-block layout copy."
            )

        if getattr(self, "_native_parallel_standardization", False):
            from .. import gxeldcore

            targets = getattr(self, "_native_missingness_targets", None)
            if targets is None:
                centered_targets = []
                for target in (self.env, self.pheno):
                    if target is None:
                        centered_targets.append(np.zeros(self.nsamp, dtype=np.float64))
                    else:
                        centered = np.asarray(target, dtype=np.float64)
                        centered_targets.append(centered - centered.mean())
                targets = np.asfortranarray(
                    np.column_stack(centered_targets), dtype=np.float64
                )
                self._native_missingness_targets = targets
            if memory_order == "C":
                if not callable(
                    getattr(gxeldcore, "standardize_genotype_block_row_major", None)
                ):
                    raise RuntimeError(
                        "The loaded GxE extension lacks row-major standardization."
                    )
                standardize = gxeldcore.standardize_genotype_block_row_major
            else:
                G = np.asfortranarray(G, dtype=np.float64)
                standardize = gxeldcore.standardize_genotype_block
            missing_counts, missing_correlations = (
                standardize(
                    G,
                    targets,
                    int(self.ddof),
                    self.genotype_scale == "hwe",
                    float(self.eps_var),
                    int(self.num_threads),
                )
            )
            counts = np.asarray(missing_counts, dtype=np.int64)
            correlations = np.asarray(missing_correlations, dtype=np.float64)
            sinks = getattr(self, "_native_missingness_sinks", None)
            if sinks is not None:
                if len(sinks) != correlations.shape[0]:
                    raise RuntimeError(
                        "Shared GxE missingness sinks do not match the native "
                        "diagnostic target count."
                    )
                seen_estimators: set[int] = set()
                for target_index, sink in enumerate(sinks):
                    estimator, attribute_name = sink
                    identity = id(estimator)
                    if identity not in seen_estimators:
                        estimator.genotype_missing_call_count[
                            blk_start:blk_end
                        ] = counts
                        seen_estimators.add(identity)
                    getattr(estimator, attribute_name)[blk_start:blk_end] = (
                        correlations[target_index]
                    )
            elif hasattr(self, "genotype_missing_call_count"):
                self.genotype_missing_call_count[blk_start:blk_end] = counts
                self.genotype_missing_environment_correlation[blk_start:blk_end] = (
                    correlations[0]
                )
                self.genotype_missing_phenotype_correlation[blk_start:blk_end] = (
                    correlations[1]
                )
            self._record_performance_phase(
                "genotype_standardization",
                time.perf_counter() - standardize_wall_started,
                time.process_time() - standardize_cpu_started,
            )
            if bound_owner is not None:
                if (
                    bound_array is None
                    or bound_byte_count is None
                    or bound_allocation_evidence is None
                    or bound_selected_nodes is None
                    or not G.flags.f_contiguous
                    or not np.shares_memory(G, bound_array)
                ):
                    raise RuntimeError(
                        "Native standardization did not preserve the dedicated "
                        "NUMA-bound genotype mapping."
                    )
                verification = verify_numa_bound_anonymous_buffer(
                    bound_owner, bound_byte_count, bound_selected_nodes
                )
                if verification.get("complete") is not True:
                    raise RuntimeError(
                        "NUMA-bound BED decoding returned incomplete evidence."
                    )
                records = getattr(
                    self, "_native_numa_bound_decode_records", None
                )
                if records is None:
                    records = []
                    self._native_numa_bound_decode_records = records
                records.append(
                    {
                        "genotype_block": [int(blk_start), int(blk_end)],
                        "memory_order": "F",
                        "decoder": "bed_reader.read_f64_into_bound_mapping",
                        "allocation": bound_allocation_evidence,
                        "bound_mapping_preserved_after_standardization": True,
                        "verification_stage": "post_standardization_pre_return",
                        "verification": verification,
                    }
                )
            return G

        # A full N-by-block boolean mask followed by NumPy's bool-to-int64
        # reduction can transiently consume more memory than the genotype
        # block itself. Scan in bounded column chunks first. The common
        # missing-free case then needs no persistent mask or integer cast;
        # the complete diagnostic path is retained when a missing call exists.
        scan_columns = min(64, G.shape[1])
        has_missing = any(
            bool(np.isnan(G[:, start:start + scan_columns]).any())
            for start in range(0, G.shape[1], scan_columns)
        )
        if has_missing:
            mask = np.isnan(G)
            missing_counts = mask.sum(axis=0, dtype=np.int64)
            self._record_genotype_missingness(
                blk_start, blk_end, mask, counts=missing_counts
            )
            nobs = G.shape[0] - missing_counts
            col_means = np.divide(
                np.nansum(G, axis=0, dtype=np.float64),
                nobs,
                out=np.zeros(G.shape[1], dtype=np.float64),
                where=nobs > 0,
            )
            rr, cc = np.where(mask)
            G[rr, cc] = col_means[cc]
            del mask, rr, cc
        else:
            col_means = G.mean(axis=0, dtype=np.float64)
        G -= col_means
        if self.genotype_scale == "hwe":
            # With mean dosage mu=2p, sqrt(mu * (1-mu/2)) is sqrt(2p(1-p)).
            col_std = np.sqrt(np.maximum(col_means * (1.0 - 0.5 * col_means), 0.0))
        else:
            centered_ss = np.einsum("ij,ij->j", G, G, optimize=False)
            col_std = np.sqrt(centered_ss / float(G.shape[0] - self.ddof))
        good = np.isfinite(col_std) & (col_std > self.eps_var)
        if np.any(good):
            G[:, good] /= col_std[good]
        if np.any(~good):
            G[:, ~good] = 0.0
        result = np.asarray(G, dtype=np.float64, order="F")
        self._record_performance_phase(
            "genotype_standardization",
            time.perf_counter() - standardize_wall_started,
            time.process_time() - standardize_cpu_started,
        )
        return result

    def _record_genotype_missingness(
        self,
        blk_start: int,
        blk_end: int,
        mask: np.ndarray,
        *,
        counts: np.ndarray | None = None,
    ) -> None:
        """Retain call rate and differential-missingness evidence before imputation."""
        # A few low-level reader tests construct the object with ``__new__`` to
        # exercise decode forwarding in isolation.  Missingness diagnostics are
        # available only on fully initialized estimators.
        if not hasattr(self, "genotype_missing_call_count"):
            return
        if counts is None:
            counts = mask.sum(axis=0, dtype=np.int64)
        else:
            counts = np.asarray(counts, dtype=np.int64)
        estimators = getattr(self, "_shared_missingness_estimators", (self,))
        for estimator in estimators:
            estimator.genotype_missing_call_count[blk_start:blk_end] = counts
            correlations = []
            for target in (estimator.env, estimator.pheno):
                result = np.zeros(blk_end - blk_start, dtype=np.float64)
                if target is not None:
                    centered = np.asarray(target, dtype=np.float64)
                    centered = centered - centered.mean()
                    target_norm = float(np.linalg.norm(centered))
                    valid = (
                        (counts > 0)
                        & (counts < estimator.nsamp)
                        & (target_norm > 0.0)
                    )
                    if np.any(valid):
                        missing_norm = np.sqrt(
                            counts[valid]
                            * (1.0 - counts[valid] / float(estimator.nsamp))
                        )
                        numerator = (
                            np.asarray(mask[:, valid], dtype=np.float64).T
                            @ centered
                        )
                        result[valid] = numerator / (
                            missing_norm * target_norm
                        )
                correlations.append(result)
            estimator.genotype_missing_environment_correlation[
                blk_start:blk_end
            ] = correlations[0]
            estimator.genotype_missing_phenotype_correlation[
                blk_start:blk_end
            ] = correlations[1]

    def _missing_genotype_diagnostics(self) -> dict[str, float | int | str | bool]:
        """Summarize documented warning thresholds for cohort-mean imputation."""
        counts = np.asarray(self.genotype_missing_call_count, dtype=np.int64)
        missing_fraction = counts / float(self.nsamp)
        maximum_missing = float(np.max(missing_fraction, initial=0.0))
        maximum_environment = float(
            np.max(
                np.abs(self.genotype_missing_environment_correlation), initial=0.0
            )
        )
        maximum_phenotype = float(
            np.max(
                np.abs(self.genotype_missing_phenotype_correlation), initial=0.0
            )
        )
        call_rate_warning_threshold = 0.05
        differential_warning_threshold = 0.10
        warning = bool(
            maximum_missing > call_rate_warning_threshold
            or maximum_environment > differential_warning_threshold
            or maximum_phenotype > differential_warning_threshold
        )
        if warning and not self._missingness_warning_logged:
            self.log._log(
                "[gxe:missingness:warning] Cohort-mean genotype imputation "
                "requires sensitivity analysis: max missing fraction="
                f"{maximum_missing:.6g}, max |corr(missing,E)|="
                f"{maximum_environment:.6g}, max |corr(missing,y)|="
                f"{maximum_phenotype:.6g}."
            )
            self._missingness_warning_logged = True
        return {
            "genotype_imputation": "cohort_variant_mean_before_projection",
            "genotype_variants_with_missing_calls": int(np.count_nonzero(counts)),
            "genotype_missing_calls": int(np.sum(counts, dtype=np.int64)),
            "minimum_genotype_call_rate": float(1.0 - maximum_missing),
            "maximum_missing_environment_correlation": maximum_environment,
            "maximum_missing_phenotype_correlation": maximum_phenotype,
            "call_rate_warning_threshold": call_rate_warning_threshold,
            "differential_missingness_warning_threshold": differential_warning_threshold,
            "missingness_warning": warning,
            "mean_imputation_validity": (
                "requires_sensitivity_analysis"
                if warning
                else "no_threshold_exceedance_observed"
            ),
        }

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
        if apply_scale and self.kernel_mode == "standardized_projected":
            if self.inv_sqrt_resvar_x_all is None:
                raise RuntimeError("Additive residual variances have not been precomputed.")
            X *= self.inv_sqrt_resvar_x_all[blk_start:blk_end].reshape(1, -1)
        return np.asarray(X, dtype=(out_dtype or self.dtype), order="F")

    def _prepare_interaction_block(self, blk_start: int, blk_end: int, G: np.ndarray | None = None, apply_scale: bool = True, out_dtype=None) -> np.ndarray:
        if G is None:
            G = self._read_genotype_block(blk_start, blk_end)
        W = np.asarray(G * self.env[:, None], dtype=np.float64, order="F")
        self._project_and_center_inplace(W)
        if apply_scale and self.kernel_mode == "standardized_projected":
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
                    require_missing_free=False,
            )
            if int(result.get("missing_genotype_calls", 0)):
                self.log._log(
                    "[gxe:native] Missing genotype calls require per-variant "
                    "differential-missingness diagnostics; falling back to the "
                    "streaming Python oracle for this reference."
                )
                self.native_backend = "python"
                return self._precompute_residual_variances_python()
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
        self.feature_diagnostics.update(self._missing_genotype_diagnostics())
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
            if self.kernel_mode == "standardized_projected":
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
            if self.kernel_mode == "standardized_projected":
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
        if self.kernel_mode == "standardized_projected":
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
        self.feature_diagnostics.update(self._missing_genotype_diagnostics())
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
        resident_multiplier = 2
        if native_direct:
            # Let U=N*K*B*8. Each opaque panel owns S and e*S. Preparing
            # a panel peaks at its 2U input plus a 4U snapshot.
            if itemsize == np.dtype(np.float64).itemsize:
                peak_multiplier = 6
            else:
                # The two float32 stored sketches use 2U total. Python's 2U
                # float64 call input and C++'s 4U opaque snapshot peak at 8U.
                peak_multiplier = 7
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
            else "paired global sketches"
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

    def _accumulate_population_diagonal_moments(
        self,
        sources: np.ndarray,
        probes_per_bin: int,
        probe_square_sums: np.ndarray,
        same_probe_products: np.ndarray,
    ) -> None:
        """Accumulate a bounded-memory U-statistic for kernel diagonals.

        For source sketch ``S[a, v] = F_a sqrt(A_a) z_v``, independent probe
        indices satisfy

          E[S[a,v]^2 S[b,u]^2] = M_a M_b K_a(ii) K_b(ii),  v != u.

        Summing the order-two U-statistic over individuals gives the
        same-individual part of ``tr(K_a K_b)``. Only the N-by-2K running sums
        and a 2K-by-2K matrix persist; no probe sketch is written to disk.
        """
        source_array = np.asarray(sources)
        families = 2 * self.nbins
        expected = (self.nsamp, families * int(probes_per_bin))
        if source_array.shape != expected:
            raise RuntimeError(
                f"Population diagonal source shape {source_array.shape} does not match {expected}."
            )
        if probe_square_sums.shape != (self.nsamp, families):
            raise RuntimeError("Population diagonal probe-square accumulator is mis-sized.")
        if same_probe_products.shape != (families, families):
            raise RuntimeError("Population diagonal same-probe accumulator is mis-sized.")

        probe_chunk = min(8, int(probes_per_bin))
        for start in range(0, int(probes_per_bin), probe_chunk):
            stop = min(int(probes_per_bin), start + probe_chunk)
            for left in range(families):
                left_offset = left * int(probes_per_bin)
                left_values = np.asarray(
                    source_array[:, left_offset + start : left_offset + stop],
                    dtype=np.float64,
                )
                left_squared = np.square(left_values)
                probe_square_sums[:, left] += np.sum(
                    left_squared, axis=1, dtype=np.float64
                )
                for right in range(left, families):
                    if right == left:
                        right_squared = left_squared
                    else:
                        right_offset = right * int(probes_per_bin)
                        right_values = np.asarray(
                            source_array[
                                :, right_offset + start : right_offset + stop
                            ],
                            dtype=np.float64,
                        )
                        right_squared = np.square(right_values)
                    value = float(
                        np.sum(left_squared * right_squared, dtype=np.float64)
                    )
                    same_probe_products[left, right] += value
                    if right != left:
                        same_probe_products[right, left] += value

    def _finalize_population_diagonal_moments(
        self,
        probe_square_sums: np.ndarray,
        same_probe_products: np.ndarray,
    ) -> np.ndarray:
        """Return the 2K-by-2K same-individual kernel-product estimate."""
        if self.nvecs < 2:
            raise ValueError(
                "Reference-population trace transfer requires at least two random vectors."
            )
        families = 2 * self.nbins
        cross_probe = np.empty((families, families), dtype=np.float64)
        for left in range(families):
            for right in range(left, families):
                total = float(
                    np.sum(
                        probe_square_sums[:, left]
                        * probe_square_sums[:, right],
                        dtype=np.float64,
                    )
                )
                value = total - float(same_probe_products[left, right])
                cross_probe[left, right] = value
                cross_probe[right, left] = value
        masses = np.concatenate(
            [
                np.asarray(self.nsnps_bin, dtype=np.float64),
                np.asarray(self.nsnps_bin, dtype=np.float64),
            ]
        )
        denominator = (
            float(self.nvecs * (self.nvecs - 1))
            * masses[:, None]
            * masses[None, :]
        )
        result = cross_probe / denominator
        if not np.all(np.isfinite(result)):
            raise RuntimeError(
                "Reference-population same-individual trace estimate is non-finite."
            )
        return 0.5 * (result + result.T)

    def _native_source_block(
        self,
        blk_start: int,
        blk_end: int,
        probes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._native_context is None:
            raise RuntimeError("The bounded native GxE backend was not initialized.")
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
        return np.asarray(source_x), np.asarray(source_w)

    def _native_target_block(
        self,
        blk_start: int,
        blk_end: int,
        sources,
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
            work_x, work_w, missing, source_leakage = (
                self._native_context.target_projected_block(
                    sources=sources, **arguments
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

    def _save_score_file(self, path: str, score: np.ndarray) -> None:
        with self._performance_phase("output_conversion"):
            snpcols = ["CHR", "SNP", "BP"]
            if self.snplist is None:
                snpdf = pd.DataFrame(
                    np.nan * np.ones((self.nsnps, 3)), columns=snpcols
                )
            else:
                snpdf = self.snplist[["CHR", "SNP", "BP"]].copy()
                snpdf.columns = snpcols
            scores_df = pd.DataFrame(score, columns=self.l2cols)
            out_df = pd.concat([snpdf, scores_df], axis=1)
        with self._performance_phase("output_serialization_compression_staging"):
            self._atomic_dataframe(
                out_df,
                path,
                sep="\t",
                compression="gzip",
                float_format=_REFERENCE_FLOAT_FORMAT,
            )

    @staticmethod
    def _atomic_dataframe(frame: pd.DataFrame, path: str, **kwargs) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.fchmod(fd, 0o600)
        os.close(fd)
        try:
            compression = kwargs.pop("compression", None)
            kwargs.setdefault("chunksize", 65_536)
            if compression == "gzip":
                # Avoid embedding the random staging filename in the gzip
                # header and favor throughput over maximum compression. The
                # 17-digit numeric representation remains lossless.
                with open(temporary, "wb") as raw:
                    with gzip.GzipFile(
                        filename="",
                        mode="wb",
                        compresslevel=1,
                        fileobj=raw,
                        mtime=0,
                    ) as compressed:
                        with io.TextIOWrapper(
                            compressed, encoding="utf-8", newline=""
                        ) as text:
                            frame.to_csv(text, index=False, **kwargs)
            else:
                frame.to_csv(
                    temporary,
                    index=False,
                    compression=compression,
                    **kwargs,
                )
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
        suffixes.extend([".gxe.diag.tsv.gz", ".gxe.ref.json"])
        if self.pheno is not None:
            suffixes.extend([".gxe.gwas.tsv.gz", ".gxe.gwis.tsv.gz", ".gxe.moments.json"])
        return [Path(f"{self.outpath}{suffix}") for suffix in suffixes]

    def _assert_output_paths_available(self) -> None:
        existing = [str(path) for path in self._planned_output_paths() if path.exists()]
        if existing and not self.overwrite:
            shown = ", ".join(existing[:5]) + (" ..." if len(existing) > 5 else "")
            raise FileExistsError(
                "Refusing to overwrite an existing GxE bundle. Choose a new --out prefix or explicitly pass "
                f"--gxe-overwrite. Existing artifacts: {shown}"
            )

    def _nxe_reference_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return compact exact statistics for ``P diag(e**2) P`` traces."""
        n = self.nsamp
        intercept = np.ones((n, 1), dtype=np.float64) / math.sqrt(float(n))
        q_full = _orthonormalize_columns(
            np.column_stack([intercept, self.C_int])
        )
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

    def _write_bundle_metadata(
        self,
        score_paths: dict[str, str],
    ) -> tuple[str, str | None]:
        if any(x is None for x in (self.norm_x_all, self.norm_w_all, self.diag_nxe_x_all, self.diag_nxe_w_all)):
            raise RuntimeError("Feature metadata were not computed before writing the GxE bundle.")
        if self.snplist is None:
            raise ValueError("A BIM file is required for a reusable GxE summary bundle.")

        with self._performance_phase("fp64_output_diagnostics"):
            trace_nxe, trace_nxe_sq = self._nxe_reference_traces()
        diag_path = f"{self.outpath}.gxe.diag.tsv.gz"
        with self._performance_phase("output_conversion"):
            diag = self.snplist[["CHR", "SNP", "BP", "A1", "A2"]].copy()
            diag["NORM_X"] = self.norm_x_all
            diag["NORM_W"] = self.norm_w_all
            diag["SCALE_X"] = self.inv_sqrt_resvar_x_all
            diag["SCALE_W"] = self.inv_sqrt_resvar_w_all
            diag["DNXE_X"] = self.diag_nxe_x_all
            diag["DNXE_W"] = self.diag_nxe_w_all
            diag["CORR_XW"] = self.corr_xw_all
            for idx in range(self.nbins):
                diag[f"ANNOT_{idx}"] = np.asarray(
                    self.annot[:, idx], dtype=np.float64
                )
        with self._performance_phase("output_serialization_compression_staging"):
            self._atomic_dataframe(
                diag,
                diag_path,
                sep="\t",
                compression="gzip",
                float_format=_REFERENCE_FLOAT_FORMAT,
            )

        manifest_path = f"{self.outpath}.gxe.ref.json"
        files = {
            **{k: self._relative_output_path(v, manifest_path) for k, v in score_paths.items()},
            "diagonal": self._relative_output_path(diag_path, manifest_path),
        }
        payload = {
            "kind": "summit.gxe.reference",
            "schema_version": 4,
            "n_samples": self.nsamp,
            "fixed_effect_rank_excluding_intercept": self.p_eff,
            "residual_rank": self.df_corr,
            "environment": self.env_name,
            "environment_transform": dict(self.environment_transform),
            "covariates": self.cov_cols,
            "kernel_mode": self.feature_convention,
            "feature_convention": self.feature_convention,
            "feature_convention_version": 1,
            "genotype_scale": self.genotype_scale,
            "ld_scale": "cross_product_over_rank_squared",
            "null_corrected": False,
            "annotation_names": list(self.l2cols),
            "annotation_value_dtype": "float64",
            "annotation_masses": np.asarray(self.nsnps_bin, dtype=np.float64).tolist(),
            "feature_diagnostics": self.feature_diagnostics,
            "resource_estimates": self.resource_estimates,
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
                "step_size_selection": self.step_size_selection,
                "target_paired_sketch_gib": float(self.target_xz_mem),
                "probe_tiles": [list(tile) for tile in (self._vtiles_used or [])],
            },
            "files": files,
        }
        if self.cpu_placement is not None:
            if self.cpu_placement_complete is not True:
                raise RuntimeError(
                    "Refusing to publish incomplete CPU placement evidence."
                )
            payload["cpu_placement"] = dict(self.cpu_placement)
            payload["cpu_placement_complete"] = True
        if self.population_same_individual_products is not None:
            population_products = np.asarray(
                self.population_same_individual_products, dtype=np.float64
            )
            expected_shape = (2 * self.nbins, 2 * self.nbins)
            if (
                population_products.shape != expected_shape
                or not np.all(np.isfinite(population_products))
            ):
                raise RuntimeError(
                    "Population same-individual products are incomplete or mis-sized."
                )
            payload["population_trace"] = {
                "method": "independent_probe_u_statistic_v1",
                "sampling_axis": "individual",
                "num_vectors": int(self.nvecs),
                "feature_order": [
                    *[f"G:{name}" for name in self.l2cols],
                    *[f"GxE:{name}" for name in self.l2cols],
                ],
                "same_individual_kernel_products": population_products.tolist(),
            }
        with self._performance_phase("output_serialization_staging"):
            self._atomic_json(payload, manifest_path)
        self.log._log(f"Saving GxE reference manifest into: {manifest_path}")

        moments_path = None
        if self.pheno is not None:
            if self.score_x_all is None or self.score_w_all is None:
                raise RuntimeError("Phenotype scores were not computed.")
            with self._performance_phase("output_conversion"):
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
            with self._performance_phase(
                "output_serialization_compression_staging"
            ):
                self._atomic_dataframe(
                    gwas,
                    gwas_path,
                    sep="\t",
                    compression="gzip",
                    float_format="%.12g",
                )
                self._atomic_dataframe(
                    gwis,
                    gwis_path,
                    sep="\t",
                    compression="gzip",
                    float_format="%.12g",
                )
            moments_path = f"{self.outpath}.gxe.moments.json"
            moments = {
                "kind": "summit.gxe.phenotype_moments",
                # Paired with the schema-v4 reference manifest written above.
                "schema_version": 4,
                "phenotype": self.phenotype_name,
                "n_samples": self.nsamp,
                "residual_rank": self.df_corr,
                "feature_convention": self.feature_convention,
                "feature_convention_version": 1,
                "score_definition": "feature_transpose_residualized_y_over_sqrt_residual_rank",
                "q_nxe": float(np.dot(self.env * self.pheno, self.env * self.pheno)),
                "q_residual": float(np.dot(self.pheno, self.pheno)),
                "phenotype_residual_variance_fraction": self.phenotype_residual_fraction,
                "files": {
                    "gwas": self._relative_output_path(gwas_path, moments_path),
                    "gwis": self._relative_output_path(gwis_path, moments_path),
                },
            }
            with self._performance_phase("output_serialization_staging"):
                self._atomic_json(moments, moments_path)
            self.log._log(f"Saving marginal GWAS/GWIS scores and NxE phenotype moments with prefix: {self.outpath}")
        return manifest_path, moments_path

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
    ):
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
        try:
            # Fail cheaply on normal retries, but retain no-replace publication
            # below as the authority for concurrent, non-cooperating writers.
            self._assert_output_paths_available()
            self._assert_construction_genotype_state()
            with tempfile.TemporaryDirectory(
                prefix=".gxe-bundle-stage-", dir=final_prefix.parent
            ) as stage_name:
                stage_dir = Path(stage_name)
                os.chmod(stage_dir, 0o700)
                stage_prefix = stage_dir / final_prefix.name
                self.outpath = str(stage_prefix)
                try:
                    result = (
                        self._compute_ldscore_impl()
                        if compute_callback is None
                        else compute_callback()
                    )

                    self._assert_construction_genotype_state()

                    staged_outputs = self._planned_output_paths()

                    def publication_priority(path: Path) -> tuple[int, str]:
                        name = path.name
                        if name.endswith(".gxe.ref.json"):
                            return (20, name)
                        if name.endswith(".gxe.moments.json"):
                            return (30, name)
                        return (0, name)

                    def verify_published_bundle() -> None:
                        with self._performance_phase("output_validation"):
                            for path, device, inode in published:
                                try:
                                    observed = path.stat(follow_symlinks=False)
                                except FileNotFoundError as exc:
                                    raise RuntimeError(
                                        "A published GxE artifact disappeared before "
                                        f"manifest commit: {path}."
                                    ) from exc
                                if (
                                    observed.st_dev != device
                                    or observed.st_ino != inode
                                ):
                                    raise RuntimeError(
                                        "A published GxE artifact was concurrently replaced "
                                        f"before manifest commit: {path}."
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
                        with self._performance_phase("output_publication"):
                            if self.overwrite:
                                os.replace(staged, final)
                                continue
                            staged_stat = staged.stat()
                            try:
                                os.link(staged, final)
                            except FileExistsError as exc:
                                raise FileExistsError(
                                    "Refusing to overwrite concurrently created GxE "
                                    f"artifact: {final}."
                                ) from exc
                            published.append(
                                (final, staged_stat.st_dev, staged_stat.st_ino)
                            )
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
            self.inv_sqrt_resvar_x_all, self.inv_sqrt_resvar_w_all = self._precompute_residual_variances()
        finally:
            self.native_phase_timings["feature_precompute"] = (
                time.perf_counter() - feature_started
            )
        blocks = self._make_compute_blocks()
        vtiles = self._auto_vtiles()
        self._vtiles_used = list(vtiles)

        max_vt = max(vt for _, vt in vtiles)
        itemsize = int(np.dtype(self.dtype).itemsize)
        if self.native_backend == "direct":
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
        resident_multiplier = 2
        # A native projected panel owns S and e*S for one 2B source sketch.
        # Block and global panels are never retained at the same time.
        native_opaque_retained_multiplier = 4 if self.native_backend == "direct" else 0
        native_opaque_prepare_peak_multiplier = (
            (
                6
                if itemsize == np.dtype(np.float64).itemsize
                else 7
            )
            if self.native_backend == "direct"
            else 0
        )
        target_source_columns = 2 * self.nbins * max_vt
        source_columns = self.nbins * max_vt
        q_rank = self.p_eff + 1
        global_source_columns = 2 * self.nbins * max_vt
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
            + 4 * max_block * target_source_columns
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
            "blas_threads": int(self.num_threads),
            "bed_reader_threads": int(self.decode_threads),
        }
        self.log._log(
            "[gxe:resources] modeled resident sketch workspace="
            f"{self.resource_estimates['resident_sketch_workspace_gib']:.3f} GiB; "
            f"BLAS threads={self.num_threads}, bed-reader threads={self.decode_threads}."
        )

        accum = {
            "xx": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
            "xw": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
            "wx": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
            "ww": np.zeros((self.nsnps, self.nbins), dtype=np.float64),
        }

        units_per_tile = 2 * len(execution_groups)
        total_units = max(1, len(vtiles) * units_per_tile)
        bar = tqdm(total=total_units, desc="GxE-LD progress", unit="task", smoothing=0.2, disable=(not self.verbose))
        max_native_source_leakage = 0.0
        population_probe_square_sums = None
        population_same_probe_products = None
        if self.nvecs >= 2:
            population_probe_square_sums = np.zeros(
                (self.nsamp, 2 * self.nbins), dtype=np.float64
            )
            population_same_probe_products = np.zeros(
                (2 * self.nbins, 2 * self.nbins), dtype=np.float64
            )
            self.resource_estimates["population_trace_workspace_gib"] = float(
                population_probe_square_sums.nbytes
                + population_same_probe_products.nbytes
            ) / (1024 ** 3)
        try:
            for v0, Vt in vtiles:
                kt = self.nbins * Vt
                sources = np.zeros(
                    (self.nsamp, 2 * kt), dtype=self.dtype, order="F"
                )
                global_x = sources[:, :kt]
                global_w = sources[:, kt:2 * kt]
                global_panel = None
                try:
                    for constituent_blocks in execution_groups:
                        s = int(constituent_blocks[0][0])
                        e = int(constituent_blocks[-1][1])
                        Z = self._generate_random_group(
                            constituent_blocks, v_count=Vt, v_start=v0
                        )
                        if self.native_backend == "direct":
                            source_x, source_w = self._native_source_block(
                                s, e, Z
                            )
                            global_x += source_x[:, :kt]
                            global_w += source_w[:, :kt]
                            del source_x, source_w
                        else:
                            G = self._read_genotype_block(s, e)
                            X = self._prepare_additive_block(
                                s, e, G=G, apply_scale=True, out_dtype=self.dtype
                            )
                            W = self._prepare_interaction_block(
                                s, e, G=G, apply_scale=True, out_dtype=self.dtype
                            )
                            # Source weights always derive from the canonical
                            # binary64 annotations; only the retained sketch
                            # storage below is permitted to round.
                            annot_blk = np.asarray(
                                self.annot[s:e], dtype=np.float64
                            )
                            self._accumulate_sketch_block(
                                global_x, X, Z, annot_blk
                            )
                            self._accumulate_sketch_block(
                                global_w, W, Z, annot_blk
                            )
                            del G, X, W, annot_blk
                        del Z
                        bar.update(1)

                    if population_probe_square_sums is not None:
                        assert population_same_probe_products is not None
                        self._accumulate_population_diagonal_moments(
                            sources,
                            Vt,
                            population_probe_square_sums,
                            population_same_probe_products,
                        )
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
                    global_panel = None
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
            retried_input_mutations = int(
                native_info.get("retried_gemm_input_mutations", 0)
            )
            self.resource_estimates["native_retried_gemm_input_mutations"] = (
                retried_input_mutations
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
                f"{repaired_gemm_columns}; freshly decoded input retries="
                f"{retried_input_mutations}."
            )

        if population_probe_square_sums is not None:
            assert population_same_probe_products is not None
            self.population_same_individual_products = (
                self._finalize_population_diagonal_moments(
                    population_probe_square_sums,
                    population_same_probe_products,
                )
            )
            self.log._log(
                "[gxe:population] accumulated same-individual kernel products "
                "in memory for external-cohort trace transfer."
            )
        else:
            self.population_same_individual_products = None

        scores = {name: value / float(self.nvecs) for name, value in accum.items()}
        return self._finalize_ldscore_outputs(scores)

    def _finalize_ldscore_outputs(
        self,
        scores: Mapping[str, np.ndarray],
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
        # Persist the raw realized-sample directional scores.  A common XX-like
        # M/r subtraction is not a valid finite-sample correction for XW/WX,
        # whose same-SNP cross-products are neither zero nor one.  The normal
        # equations below need the raw kernel Gram traces in every direction.
        with self._performance_phase("fp64_output_diagnostics"):
            annot64 = np.asarray(self.annot, dtype=np.float64)
            max_cross_abs = 0.0
            max_cross_rel = 0.0
            r2 = float(self.df_corr * self.df_corr)
            for left_bin in range(self.nbins):
                for source_bin in range(self.nbins):
                    forward = (
                        r2
                        * _non_blas_fp64_inner_product(
                            annot64[:, left_bin], scores["xw"][:, source_bin]
                        )
                        / (
                            self.nsnps_bin[left_bin]
                            * self.nsnps_bin[source_bin]
                        )
                    )
                    reverse = (
                        r2
                        * _non_blas_fp64_inner_product(
                            annot64[:, source_bin], scores["wx"][:, left_bin]
                        )
                        / (
                            self.nsnps_bin[source_bin]
                            * self.nsnps_bin[left_bin]
                        )
                    )
                    delta = abs(float(forward - reverse))
                    max_cross_abs = max(max_cross_abs, delta)
                    max_cross_rel = max(
                        max_cross_rel,
                        delta
                        / max(
                            abs(float(forward)),
                            abs(float(reverse)),
                            np.finfo(float).tiny,
                        ),
                    )
            self.feature_diagnostics[
                "max_xw_wx_trace_asymmetry_absolute"
            ] = max_cross_abs
            self.feature_diagnostics[
                "max_xw_wx_trace_asymmetry_relative"
            ] = max_cross_rel
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
        self._write_bundle_metadata(score_paths)

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
