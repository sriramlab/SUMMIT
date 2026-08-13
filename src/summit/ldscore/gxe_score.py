"""Reusable phenotype scoring against a sealed schema-v3 GxE reference.

This module deliberately does not regenerate randomized trace panels.  It
validates that the supplied individual-level inputs reproduce the exact sample,
environment, fixed-effect, genotype, variant, and feature definition sealed by
an existing reference manifest, then makes one genotype decode pass to produce
the phenotype-dependent sufficient statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from bed_reader import open_bed
from threadpoolctl import threadpool_limits

from ..inference.gxe import (
    _BLOCK_LOCAL_JACKKNIFE_METHOD,
    _EXACT_JACKKNIFE_METHOD,
    _validate_reference_feature_cache_contract,
    ordered_variant_digest,
)
from .gwe_ldscore import (
    _FEATURE_CACHE_ARRAY_DTYPES,
    _canonical_bfile_prefix,
    _validate_feature_cache_semantics,
    _validate_backend_provenance,
    _validate_gxe_annotation_names,
    _validate_plink_bed_shape,
    read_env_and_cov,
)


_REFERENCE_KIND = "summit.gxe.reference"
_MOMENTS_KIND = "summit.gxe.phenotype_moments"
_SCHEMA_VERSION = 3
_SCORE_DEFINITION = "feature_transpose_residualized_y_over_sqrt_residual_rank"
_SCORE_MODE = "marginal_cross_product"
_GENOTYPE_EXTENSIONS = (".bed", ".bim", ".fam")
_REFERENCE_ARTIFACTS = frozenset({"xx", "xw", "wx", "ww", "diagonal"})
_MAX_IN_MEMORY_SCORE_BYTES = 8 * 1024**3
_MAX_WIDE_WORKING_BYTES = 12 * 1024**3


@dataclass(frozen=True)
class GxEPhenotypeScoreArtifacts:
    """Paths written by :func:`score_phenotype_from_reference`."""

    gwas: Path
    gwis: Path
    moments: Path


@dataclass(frozen=True)
class _ValidatedReference:
    path: Path
    payload: dict[str, Any]
    manifest_sha256: str
    diagonal: pd.DataFrame
    variants: pd.DataFrame
    scale_x: np.ndarray
    scale_w: np.ndarray
    norm_x: np.ndarray
    norm_w: np.ndarray
    feature_cache_path: Path | None
    feature_cache_sha256: str | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    # Parse and hash the same immutable byte snapshot.  A later moments file
    # binds this exact manifest digest.
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON reference manifest: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _load_json_and_sha256(path: Path) -> tuple[dict[str, Any], str]:
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON reference manifest: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_validated_feature_cache(
    path: Path, expected_sha256: str, *, scratch_dir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    digest = hashlib.sha256()
    with tempfile.TemporaryFile(
        prefix=".gxe-cache-snapshot-", suffix=".npz", dir=scratch_dir
    ) as snapshot:
        os.fchmod(snapshot.fileno(), 0o600)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
                snapshot.write(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Reference feature cache failed its SHA-256 check.")
        snapshot.seek(0)
        with np.load(snapshot, allow_pickle=False) as bundle:
            expected_members = {*_FEATURE_CACHE_ARRAY_DTYPES, "metadata_json"}
            if len(bundle.files) != len(expected_members) or set(bundle.files) != expected_members:
                raise ValueError(
                    "Reference feature cache has unexpected, duplicate, or missing arrays."
                )
            try:
                metadata = json.loads(str(bundle["metadata_json"].item()))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("Reference feature cache has invalid metadata_json.") from exc
            arrays = {
                name: np.asarray(bundle[name]).copy()
                for name in _FEATURE_CACHE_ARRAY_DTYPES
            }
    _validate_feature_cache_semantics(metadata, arrays)
    return metadata, arrays


def _resolve_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Reference manifest contains an invalid artifact path.")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _as_finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values; got shape {array.shape}.")
    return array


def _read_bim(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        header=None,
        sep=r"\s+",
        dtype={0: str, 1: str, 3: np.int64, 4: str, 5: str},
    )
    if frame.shape[1] < 6:
        raise ValueError(f"BIM file {path} must contain six columns.")
    frame = frame.iloc[:, [0, 1, 3, 4, 5]].copy()
    frame.columns = ["CHR", "SNP", "BP", "A1", "A2"]
    if frame["SNP"].duplicated().any():
        raise ValueError(f"BIM file {path} contains duplicate SNP identifiers.")
    return frame


def _read_diagonal(path_or_buffer, *, compression: str | None = "infer") -> pd.DataFrame:
    return pd.read_csv(
        path_or_buffer,
        sep=r"\s+",
        compression=compression,
        dtype={"CHR": str, "SNP": str, "A1": str, "A2": str},
    )


def _validate_reference_artifacts(
    manifest_path: Path, payload: dict[str, Any], *, scratch_dir: Path
) -> tuple[Path, pd.DataFrame]:
    files = payload.get("files")
    hashes = payload.get("artifact_sha256")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ValueError("Reference manifest must declare files and artifact_sha256 objects.")
    required = set(_REFERENCE_ARTIFACTS)
    if "jackknife" in files:
        required.add("jackknife")
    if not required.issubset(files) or not required.issubset(hashes):
        raise ValueError(
            "Reference manifest does not declare and hash every required reference artifact."
        )
    diagonal_path = _resolve_path(manifest_path, files["diagonal"])
    diagonal = None
    for label in sorted(required):
        expected = hashes[label]
        if not _is_sha256(expected):
            raise ValueError(f"Reference artifact {label!r} has an invalid SHA-256 declaration.")
        artifact = _resolve_path(manifest_path, files[label])
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        if label == "diagonal":
            digest = hashlib.sha256()
            with tempfile.TemporaryFile(
                prefix=".gxe-diagonal-snapshot-", suffix=".tmp", dir=scratch_dir
            ) as snapshot:
                os.fchmod(snapshot.fileno(), 0o600)
                with open(artifact, "rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        snapshot.write(block)
                if digest.hexdigest() != expected.lower():
                    raise ValueError(
                        f"Reference artifact {label!r} failed its SHA-256 check: {artifact}."
                    )
                snapshot.seek(0)
                diagonal = _read_diagonal(
                    snapshot,
                    compression="gzip" if artifact.suffix == ".gz" else None,
                )
        else:
            observed = _sha256_file(artifact)
            if observed != expected.lower():
                raise ValueError(
                    f"Reference artifact {label!r} failed its SHA-256 check: {artifact}."
                )
    if diagonal is None:
        raise RuntimeError("Reference diagonal was not parsed from its validated snapshot.")
    return diagonal_path, diagonal


def _validate_reference_manifest(
    reference_manifest: str | Path, *, scratch_dir: Path
) -> _ValidatedReference:
    path = Path(reference_manifest).resolve()
    payload, manifest_sha256 = _load_json_and_sha256(path)
    if payload.get("kind") != _REFERENCE_KIND or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Reusable phenotype scoring requires a schema-v3 SUMMIT GxE reference.")
    if payload.get("backend_provenance") is not None:
        _validate_backend_provenance(
            payload["backend_provenance"], expected_stage="reference"
        )
    shard_backends = payload.get("shard_backend_provenance")
    if shard_backends is not None:
        if not isinstance(shard_backends, list) or not shard_backends:
            raise ValueError("Merged reference has invalid shard backend provenance.")
        for backend in shard_backends:
            _validate_backend_provenance(
                backend, expected_stage="reference_shard"
            )

    kernel_mode = payload.get("kernel_mode")
    genotype_scale = payload.get("genotype_scale")
    if kernel_mode not in {"standardized", "genie"}:
        raise ValueError(f"Reference manifest has unsupported kernel_mode={kernel_mode!r}.")
    if genotype_scale not in {"sample", "hwe"}:
        raise ValueError(f"Reference manifest has unsupported genotype_scale={genotype_scale!r}.")
    if payload.get("ld_scale") != "cross_product_over_rank_squared":
        raise ValueError("Reference manifest has an unsupported LD-score scale.")
    if payload.get("null_corrected") is not False:
        raise ValueError("Schema-v3 reusable scoring requires raw, non-offset reference panels.")
    feature_cache = payload.get("feature_cache")
    feature_cache_path = None
    feature_cache_sha256 = None
    feature_cache_metadata = None
    feature_cache_arrays = None
    if feature_cache is not None:
        if (
            not isinstance(feature_cache, dict)
            or not isinstance(feature_cache.get("path"), str)
            or not feature_cache["path"]
            or not _is_sha256(feature_cache.get("sha256"))
        ):
            raise ValueError("Reference manifest has an invalid feature-cache binding.")
        feature_cache_path = _resolve_path(path, feature_cache["path"])
        if not feature_cache_path.is_file():
            raise FileNotFoundError(feature_cache_path)
        feature_cache_sha256 = str(feature_cache["sha256"])
        feature_cache_metadata, feature_cache_arrays = _load_validated_feature_cache(
            feature_cache_path,
            feature_cache_sha256,
            scratch_dir=scratch_dir,
        )

    scalar_fields = (
        "n_samples",
        "fixed_effect_rank_excluding_intercept",
        "residual_rank",
    )
    if any(isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) for key in scalar_fields):
        raise ValueError("Reference sample/rank metadata must be JSON integers.")
    n = int(payload["n_samples"])
    p_eff = int(payload["fixed_effect_rank_excluding_intercept"])
    residual_rank = int(payload["residual_rank"])
    if n <= 0 or p_eff < 0 or residual_rank != n - p_eff - 1 or residual_rank <= 0:
        raise ValueError("Reference sample count, fixed-effect rank, and residual rank are inconsistent.")
    if not _is_sha256(payload.get("analysis_fingerprint")) or not _is_sha256(
        payload.get("variant_digest")
    ):
        raise ValueError("Reference manifest contains an invalid analysis or variant fingerprint.")

    environment_transform = payload.get("environment_transform")
    if not isinstance(environment_transform, dict):
        raise ValueError("Schema-v3 reference is missing environment_transform metadata.")
    if (
        environment_transform.get("standardized") is not True
        or environment_transform.get("units") != "per_environment_sd"
        or environment_transform.get("ddof") not in (0, 1)
    ):
        raise ValueError("Reference environment transform is not a supported per-SD convention.")
    for key in ("raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares"):
        value = environment_transform.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Reference environment_transform[{key!r}] is invalid.")
    if float(environment_transform["raw_sd"]) <= 0.0:
        raise ValueError("Reference environment raw_sd must be positive.")
    if not _is_sha256(environment_transform.get("fixed_effect_design_sha256")):
        raise ValueError("Reference environment transform lacks a fixed-effect design digest.")

    names = payload.get("annotation_names")
    if not isinstance(names, list):
        raise ValueError("Reference annotation_names must be a JSON list.")
    names = _validate_gxe_annotation_names(names)
    if not isinstance(payload.get("environment"), str) or not payload["environment"]:
        raise ValueError("Reference environment name must be a non-empty string.")
    covariates = payload.get("covariates")
    if not isinstance(covariates, list) or any(
        not isinstance(name, str) or not name for name in covariates
    ):
        raise ValueError("Reference covariates must be a string list.")

    diagonal_path, diagonal = _validate_reference_artifacts(
        path, payload, scratch_dir=scratch_dir
    )
    jackknife = payload.get("jackknife")
    jackknife_file = payload.get("files", {}).get("jackknife")
    jackknife_labels: list[str] | None = None
    if jackknife is None:
        if jackknife_file is not None:
            raise ValueError("Reference declares a jackknife artifact without jackknife metadata.")
    else:
        if not isinstance(jackknife, dict):
            raise ValueError("Reference jackknife declaration must be an object.")
        method = jackknife.get("method")
        if method not in {
            _EXACT_JACKKNIFE_METHOD,
            _BLOCK_LOCAL_JACKKNIFE_METHOD,
        }:
            raise ValueError(f"Reference has unsupported jackknife method {method!r}.")
        if method == _EXACT_JACKKNIFE_METHOD and jackknife_file is None:
            raise ValueError("Exact two-sided GxE jackknife is missing its trace bundle.")
        if method == _BLOCK_LOCAL_JACKKNIFE_METHOD and jackknife_file is not None:
            raise ValueError("Block-local GxE jackknife must not declare an exact trace bundle.")
        labels = jackknife.get("block_labels")
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
            or jackknife.get("num_blocks") != len(labels)
        ):
            raise ValueError("Reference jackknife block labels/count are invalid.")
        jackknife_labels = labels
    weight_columns = [f"ANNOT_{idx}" for idx in range(len(names))]
    required_columns = [
        "CHR",
        "SNP",
        "BP",
        "A1",
        "A2",
        "NORM_X",
        "NORM_W",
        "SCALE_X",
        "SCALE_W",
        "DNXE_X",
        "DNXE_W",
        "CORR_XW",
        *weight_columns,
    ]
    if jackknife is not None:
        required_columns.append("BLOCK")
    observed_columns = diagonal.columns.astype(str).tolist()
    if observed_columns != required_columns or len(set(observed_columns)) != len(observed_columns):
        raise ValueError(
            "Reference diagonal must contain exactly the canonical ordered columns "
            f"{required_columns}; observed {observed_columns}."
        )
    if diagonal["SNP"].duplicated().any():
        raise ValueError("Reference diagonal contains duplicate SNP identifiers.")
    if jackknife_labels is not None:
        block_values = pd.to_numeric(diagonal["BLOCK"], errors="raise").to_numpy(
            dtype=np.int64
        )
        if set(np.unique(block_values).tolist()) != set(range(len(jackknife_labels))):
            raise ValueError("Reference jackknife block IDs do not match its labels.")
    if ordered_variant_digest(diagonal) != payload["variant_digest"]:
        raise ValueError("Reference diagonal variant digest disagrees with its manifest.")
    m = len(diagonal)
    if m <= 0:
        raise ValueError("Reference diagonal is empty.")

    annotations = diagonal.loc[:, weight_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(annotations)) or np.any(annotations < 0.0):
        raise ValueError("Reference diagonal annotations must be finite and non-negative.")
    observed_masses = annotations.sum(axis=0, dtype=np.float64)
    declared_masses = _as_finite_vector(
        "reference annotation_masses", payload.get("annotation_masses"), len(names)
    )
    if not np.allclose(declared_masses, observed_masses, rtol=5.0e-10, atol=1.0e-8):
        raise ValueError("Reference annotation masses disagree with its diagonal weights.")

    scale_x = _as_finite_vector("SCALE_X", diagonal["SCALE_X"], m)
    scale_w = _as_finite_vector("SCALE_W", diagonal["SCALE_W"], m)
    norm_x = _as_finite_vector("NORM_X", diagonal["NORM_X"], m)
    norm_w = _as_finite_vector("NORM_W", diagonal["NORM_W"], m)
    if np.any(scale_x <= 0.0) or np.any(scale_w <= 0.0):
        raise ValueError("Reference feature scales must be strictly positive.")
    if np.any(norm_x <= 0.0) or np.any(norm_w <= 0.0):
        raise ValueError("Reference feature norms must be strictly positive.")
    if kernel_mode == "standardized":
        if not np.allclose(norm_x, 1.0, rtol=1.0e-9, atol=1.0e-9) or not np.allclose(
            norm_w, 1.0, rtol=1.0e-9, atol=1.0e-9
        ):
            raise ValueError("Standardized reference violates its unit residual-norm contract.")
    elif not np.allclose(scale_x, 1.0, rtol=0.0, atol=1.0e-12) or not np.allclose(
        scale_w, 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("GENIE-compatible reference must store unit post-projection scales.")

    if feature_cache_metadata is not None:
        if feature_cache_arrays is None:
            raise RuntimeError("Validated feature-cache arrays are unavailable.")
        _validate_reference_feature_cache_contract(
            payload,
            diagonal,
            names,
            feature_cache_metadata,
            feature_cache_arrays,
        )
        # The reusable-score proof is complete; release the M-by-K cache arrays
        # before genotype blocks and wide phenotype scores are allocated.
        feature_cache_arrays.clear()
        feature_cache_arrays = None

    variants = diagonal.loc[:, ["CHR", "SNP", "BP", "A1", "A2"]].copy()
    return _ValidatedReference(
        path=path,
        payload=payload,
        manifest_sha256=manifest_sha256,
        diagonal=diagonal,
        variants=variants,
        scale_x=scale_x,
        scale_w=scale_w,
        norm_x=norm_x,
        norm_w=norm_w,
        feature_cache_path=feature_cache_path,
        feature_cache_sha256=feature_cache_sha256,
    )


def _validate_genotype_files(prefix: str, reference: _ValidatedReference) -> tuple[int, int]:
    n, m = _validate_plink_bed_shape(prefix)
    provenance = reference.payload.get("genotype_files")
    if not isinstance(provenance, dict) or not set(_GENOTYPE_EXTENSIONS).issubset(provenance):
        raise ValueError("Schema-v3 reference is missing complete genotype provenance.")
    for extension in _GENOTYPE_EXTENSIONS:
        entry = provenance[extension]
        if not isinstance(entry, dict):
            raise ValueError(f"Reference genotype provenance for {extension} is invalid.")
        expected_size = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or not _is_sha256(expected_hash)
        ):
            raise ValueError(f"Reference genotype provenance for {extension} is invalid.")
        path = Path(prefix + extension)
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            raise ValueError(
                f"Genotype {extension} byte size disagrees with the reference: "
                f"expected {expected_size}, observed {observed_size}."
            )
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash.lower():
            raise ValueError(f"Genotype {extension} failed its reference SHA-256 check.")
    if n != int(reference.payload["n_samples"]) and n < int(reference.payload["n_samples"]):
        raise ValueError("Genotype FAM has fewer samples than the reference analysis.")
    if m != len(reference.diagonal):
        raise ValueError(
            f"Genotype BIM has {m} variants but the reference diagonal has {len(reference.diagonal)}."
        )
    return n, m


@contextmanager
def _stable_genotype_prefix(prefix: str, staging_dir: Path):
    """Expose one-open-descriptor PLINK inputs through private /proc links.

    This neither copies the production BED nor changes source inode metadata.
    Path replacement cannot redirect an already-open descriptor, while final
    fstat plus SHA validation detects in-place mutation, including ABA edits.
    """
    stable_prefix = str(staging_dir / "validated-genotype")
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():
        raise RuntimeError(
            "Stable zero-copy PLINK scoring requires Linux /proc/self/fd; "
            "run on the supported Linux/Hoffman environment."
        )
    descriptors: dict[str, int] = {}
    state: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for extension in _GENOTYPE_EXTENSIONS:
            source = Path(prefix + extension)
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(source, flags)
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode):
                os.close(descriptor)
                raise ValueError(f"PLINK input must be a regular non-symlink file: {source}.")
            descriptors[extension] = descriptor
            destination = Path(stable_prefix + extension)
            destination.symlink_to(proc_fds / str(descriptor))
            state[extension] = (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
        yield stable_prefix, descriptors, state
    finally:
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass


def _assert_stable_genotype_snapshot(
    descriptors: Mapping[str, int],
    expected: Mapping[str, tuple[int, int, int, int, int]],
) -> None:
    changed = []
    for extension, sealed in expected.items():
        try:
            observed = os.fstat(descriptors[extension])
        except (KeyError, OSError):
            changed.append(extension)
            continue
        current = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        if current != sealed:
            changed.append(extension)
    if changed:
        raise RuntimeError(
            "Private PLINK scoring inputs changed during genotype decoding; "
            f"aborting publication. Changed files: {changed}."
        )


def _validate_variant_axis(prefix: str, reference: _ValidatedReference) -> None:
    bim = _read_bim(Path(prefix + ".bim"))
    if len(bim) != len(reference.variants):
        raise ValueError("BIM and reference diagonal have different variant counts.")
    for column in ("CHR", "SNP", "BP", "A1", "A2"):
        left = bim[column].astype(str).to_numpy()
        right = reference.variants[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            first = int(np.flatnonzero(left != right)[0])
            raise ValueError(
                f"BIM variant axis disagrees with the reference at row {first} ({column})."
            )
    if ordered_variant_digest(bim) != reference.payload["variant_digest"]:
        raise ValueError("BIM variant digest disagrees with the reference manifest.")


def _analysis_fingerprint(
    fam_path: Path,
    row_selection: np.ndarray,
    environment: np.ndarray,
    fixed_effect_basis: np.ndarray,
    fixed_effect_design_sha256: str,
) -> str:
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        dtype={0: str, 1: str},
    )
    selected = fam.iloc[np.asarray(row_selection, dtype=int)]
    digest = hashlib.sha256()
    for fid, iid in selected.itertuples(index=False, name=None):
        digest.update(str(fid).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(iid).encode("utf-8"))
        digest.update(b"\n")
    digest.update(np.asarray(environment, dtype="<f8").tobytes(order="C"))
    if not _is_sha256(fixed_effect_design_sha256):
        raise ValueError("Observed fixed-effect design digest is invalid.")
    digest.update(bytes.fromhex(fixed_effect_design_sha256))
    digest.update(
        int(fixed_effect_basis.shape[1]).to_bytes(8, byteorder="little", signed=False)
    )
    return digest.hexdigest()


def _validate_design_against_reference(
    *,
    prefix: str,
    reference: _ValidatedReference,
    environment: np.ndarray,
    environment_name: str,
    fixed_basis: np.ndarray,
    row_selection: np.ndarray,
    covariate_names: Sequence[str],
    observed_transform: dict[str, Any],
) -> None:
    if environment_name != reference.payload["environment"]:
        raise ValueError("Environment column name disagrees with the reference manifest.")
    if list(covariate_names) != list(reference.payload["covariates"]):
        raise ValueError("Covariate columns disagree with the reference manifest.")

    n = len(row_selection)
    rank = int(fixed_basis.shape[1])
    residual_rank = n - rank - 1
    if n != int(reference.payload["n_samples"]):
        raise ValueError(
            f"Retained phenotype sample count {n} disagrees with reference N={reference.payload['n_samples']}."
        )
    if rank != int(reference.payload["fixed_effect_rank_excluding_intercept"]):
        raise ValueError("Fixed-effect numerical rank disagrees with the reference manifest.")
    if residual_rank != int(reference.payload["residual_rank"]):
        raise ValueError("Residual rank disagrees with the reference manifest.")

    transform = reference.payload["environment_transform"]
    for key in ("raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares"):
        if not np.isclose(
            float(observed_transform[key]),
            float(transform[key]),
            rtol=2.0e-12,
            atol=2.0e-12,
        ):
            raise ValueError(f"Environment transform disagrees with the reference for {key}.")
    if observed_transform.get("fixed_effect_design_sha256") != transform.get(
        "fixed_effect_design_sha256"
    ):
        raise ValueError("Fixed-effect design values disagree with the reference.")
    fingerprint = _analysis_fingerprint(
        Path(prefix + ".fam"),
        row_selection,
        environment,
        fixed_basis,
        str(observed_transform["fixed_effect_design_sha256"]),
    )
    if fingerprint != reference.payload["analysis_fingerprint"]:
        raise ValueError(
            "Supplied samples, environment, or fixed-effect design do not match the reference analysis fingerprint."
        )


def _validate_analysis_inputs(
    *,
    prefix: str,
    reference: _ValidatedReference,
    env_path: str | Path,
    covar_path: str | Path | None,
    pheno_path: str | Path,
    pheno_col: str | None,
    missing_values: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float, dict[str, Any]]:
    transform = reference.payload["environment_transform"]
    (
        environment,
        environment_name,
        fixed_basis,
        _,
        row_selection,
        covariate_names,
        phenotype,
        phenotype_name,
        residual_fraction,
        observed_transform,
    ) = read_env_and_cov(
        env_filename=str(env_path),
        fam_filename=prefix + ".fam",
        cov_filename=None if covar_path is None else str(covar_path),
        std=True,
        cov_impute_method="ignore",
        logger=None,
        verbose=False,
        sample_idx=None,
        ddof=int(transform["ddof"]),
        pheno_filename=str(pheno_path),
        pheno_col=pheno_col,
        missing_values=missing_values,
    )
    if phenotype is None or phenotype_name is None or residual_fraction is None:
        raise ValueError("Reusable GxE scoring currently requires one quantitative phenotype.")
    _validate_design_against_reference(
        prefix=prefix,
        reference=reference,
        environment=np.asarray(environment, dtype=np.float64),
        environment_name=str(environment_name),
        fixed_basis=np.asarray(fixed_basis, dtype=np.float64),
        row_selection=np.asarray(row_selection, dtype=int),
        covariate_names=covariate_names,
        observed_transform=observed_transform,
    )
    return (
        np.asarray(row_selection, dtype=int),
        np.asarray(environment, dtype=np.float64),
        np.asfortranarray(fixed_basis, dtype=np.float64),
        np.asarray(phenotype, dtype=np.float64),
        str(phenotype_name),
        float(residual_fraction),
        observed_transform,
    )


def _validate_reference_design_inputs(
    *,
    prefix: str,
    reference: _ValidatedReference,
    env_path: str | Path,
    covar_path: str | Path | None,
    missing_values: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = reference.payload["environment_transform"]
    (
        environment,
        environment_name,
        fixed_basis,
        _,
        row_selection,
        covariate_names,
        phenotype,
        phenotype_name,
        residual_fraction,
        observed_transform,
    ) = read_env_and_cov(
        env_filename=str(env_path),
        fam_filename=prefix + ".fam",
        cov_filename=None if covar_path is None else str(covar_path),
        std=True,
        cov_impute_method="ignore",
        logger=None,
        verbose=False,
        sample_idx=None,
        ddof=int(transform["ddof"]),
        pheno_filename=None,
        pheno_col=None,
        missing_values=missing_values,
    )
    if phenotype is not None or phenotype_name is not None or residual_fraction is not None:
        raise RuntimeError("Phenotype-free reference design construction returned phenotype state.")
    _validate_design_against_reference(
        prefix=prefix,
        reference=reference,
        environment=np.asarray(environment, dtype=np.float64),
        environment_name=str(environment_name),
        fixed_basis=np.asarray(fixed_basis, dtype=np.float64),
        row_selection=np.asarray(row_selection, dtype=int),
        covariate_names=covariate_names,
        observed_transform=observed_transform,
    )
    return (
        np.asarray(row_selection, dtype=int),
        np.asarray(environment, dtype=np.float64),
        np.asfortranarray(fixed_basis, dtype=np.float64),
    )


def _validate_trait_names(names: Sequence[str]) -> tuple[str, ...]:
    traits = tuple(str(name) for name in names)
    if not traits or len(set(traits)) != len(traits):
        raise ValueError("pheno_cols must contain at least one unique trait name.")
    unsafe = [
        name
        for name in traits
        if not name
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None
    ]
    if unsafe:
        raise ValueError(
            "Wide phenotype output names must start with a letter or digit and then use "
            "only letters, digits, '.', '_', or '-'; "
            f"invalid traits: {unsafe}."
        )
    return traits


def _validate_num_threads(num_threads: int | None) -> int | None:
    if num_threads is None:
        return None
    if isinstance(num_threads, bool) or not isinstance(num_threads, (int, np.integer)):
        raise ValueError("num_threads must be a positive integer or None.")
    validated = int(num_threads)
    if validated <= 0:
        raise ValueError("num_threads must be a positive integer or None.")
    return validated


def _read_and_project_wide_phenotypes(
    *,
    pheno_path: str | Path,
    pheno_cols: Sequence[str] | None,
    fam_path: Path,
    row_selection: np.ndarray,
    fixed_basis: np.ndarray,
    residual_rank: int,
    missing_values: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    if pheno_cols is None:
        header = pd.read_csv(pheno_path, sep=r"\s+", nrows=0)
        available = [
            str(column) for column in header.columns if column not in {"FID", "IID"}
        ]
        traits = _validate_trait_names(available)
    elif isinstance(pheno_cols, str):
        traits = _validate_trait_names([pheno_cols])
    else:
        traits = _validate_trait_names(pheno_cols)
    phenotype_table = pd.read_csv(
        pheno_path,
        sep=r"\s+",
        usecols=["FID", "IID", *traits],
        dtype={"FID": str, "IID": str},
        na_values=list(missing_values),
        keep_default_na=True,
    )
    if "FID" not in phenotype_table or "IID" not in phenotype_table:
        raise ValueError("Wide phenotype file must contain FID and IID columns.")
    if phenotype_table.duplicated(subset=["FID", "IID"]).any():
        raise ValueError("Wide phenotype file contains duplicate FID/IID rows.")
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype={0: str, 1: str},
    )
    selected_ids = fam.iloc[np.asarray(row_selection, dtype=int)].reset_index(drop=True)
    selected = selected_ids.merge(
        phenotype_table.loc[:, ["FID", "IID", *traits]],
        on=["FID", "IID"],
        how="left",
        indicator=True,
        sort=False,
    )
    missing_ids = int((selected["_merge"] != "both").sum())
    if missing_ids:
        raise ValueError(
            f"Wide phenotype file is missing {missing_ids} reference-cohort FID/IID row(s)."
        )
    selected.drop(columns="_merge", inplace=True)
    for trait in traits:
        selected[trait] = pd.to_numeric(selected[trait], errors="coerce")
    phenotype = selected.loc[:, list(traits)].to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(phenotype)
    if np.any(invalid):
        rows, columns = np.where(invalid)
        examples = [
            f"{traits[int(column)]}@row{int(row)}"
            for row, column in zip(rows[:5], columns[:5])
        ]
        raise ValueError(
            "Every selected wide trait must be complete and finite on the exact reference cohort; "
            f"invalid values include {examples}."
        )

    raw_centered = phenotype - phenotype.mean(axis=0, keepdims=True)
    raw_ss = np.sum(raw_centered * raw_centered, axis=0, dtype=np.float64)
    projected = np.array(phenotype, copy=True, dtype=np.float64, order="F")
    if fixed_basis.shape[1] > 0:
        projected -= fixed_basis @ (fixed_basis.T @ projected)
    projected -= projected.mean(axis=0, keepdims=True)
    residual_ss = np.sum(projected * projected, axis=0, dtype=np.float64)
    minimum = np.maximum(
        np.finfo(np.float64).tiny,
        np.finfo(np.float64).eps * np.maximum(raw_ss, 1.0),
    )
    invalid_variance = (
        (~np.isfinite(raw_ss))
        | (raw_ss <= 0.0)
        | (~np.isfinite(residual_ss))
        | (residual_ss <= minimum)
    )
    if np.any(invalid_variance):
        bad = [traits[index] for index in np.flatnonzero(invalid_variance)]
        raise ValueError(
            f"Wide traits have zero or invalid residual variance after projection: {bad}."
        )
    residual_fraction = residual_ss / raw_ss
    projected *= np.sqrt(float(residual_rank) / residual_ss)[None, :]
    return traits, np.asfortranarray(projected), residual_fraction


def _read_genotype_block(
    bed,
    row_selection: np.ndarray,
    start: int,
    end: int,
    *,
    genotype_scale: str,
    ddof: int,
    eps_var: float,
    num_threads: int | None,
) -> np.ndarray:
    indexer = np.s_[row_selection, start:end]
    read_keywords: dict[str, Any] = {"index": indexer, "dtype": np.float64}
    if num_threads is not None:
        read_keywords["num_threads"] = num_threads
    try:
        genotype = bed.read(**read_keywords)
    except TypeError:
        # Older bed-reader releases do not accept num_threads on read().
        read_keywords.pop("num_threads", None)
        try:
            genotype = bed.read(**read_keywords)
        except TypeError:
            # Retain compatibility with readers that do not accept dtype or a
            # keyword index.  The conversion below restores float64 in either
            # case.
            try:
                genotype = bed.read(index=indexer)
            except TypeError:
                genotype = bed.read(indexer)
        genotype = np.asarray(genotype, dtype=np.float64)
    genotype = np.asarray(genotype, dtype=np.float64)
    expected_shape = (len(row_selection), end - start)
    if genotype.shape == expected_shape[::-1]:
        genotype = genotype.T
    if genotype.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected BED block shape {genotype.shape}; expected {expected_shape}."
        )

    means = np.nanmean(genotype, axis=0)
    if not np.all(np.isfinite(means)):
        raise ValueError(f"Genotype block [{start}:{end}) contains an all-missing variant.")
    missing = np.isnan(genotype)
    if missing.any():
        rows, columns = np.where(missing)
        genotype[rows, columns] = means[columns]
    genotype -= means
    if genotype_scale == "hwe":
        standard_deviation = np.sqrt(np.maximum(means * (1.0 - 0.5 * means), 0.0))
    else:
        standard_deviation = genotype.std(axis=0, ddof=ddof)
    valid = np.isfinite(standard_deviation) & (standard_deviation > eps_var)
    if not np.all(valid):
        first = start + int(np.flatnonzero(~valid)[0])
        raise ValueError(f"Variant row {first} has invalid pre-projection genotype variance.")
    genotype /= standard_deviation
    return np.asarray(genotype, dtype=np.float64, order="F")


def _project_and_center(matrix: np.ndarray, fixed_basis: np.ndarray) -> np.ndarray:
    if fixed_basis.shape[1] > 0:
        matrix -= fixed_basis @ (fixed_basis.T @ matrix)
    matrix -= matrix.mean(axis=0, keepdims=True)
    return matrix


def _score_one_genotype_pass(
    *,
    prefix: str,
    reference: _ValidatedReference,
    row_selection: np.ndarray,
    environment: np.ndarray,
    fixed_basis: np.ndarray,
    phenotype: np.ndarray,
    total_samples: int,
    step_size: int,
    eps_var: float,
    num_threads: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if num_threads is None:
        bed = open_bed(prefix + ".bed")
    else:
        try:
            bed = open_bed(prefix + ".bed", num_threads=num_threads)
        except TypeError:
            # Compatibility with older bed-reader versions; read() has its own
            # corresponding fallback below.
            bed = open_bed(prefix + ".bed")
    expected_shape = (int(total_samples), len(reference.diagonal))
    # The shape was already established from exact FAM/BIM row counts.  This
    # reader check guards an inconsistent backend before the first decode.
    if tuple(bed.shape) != expected_shape:
        raise RuntimeError(
            f"BED reader shape {bed.shape} disagrees with validated FAM/BIM shape {expected_shape}."
        )
    m = len(reference.diagonal)
    residual_rank = int(reference.payload["residual_rank"])
    root_rank = math.sqrt(float(residual_rank))
    phenotype_array = np.asarray(phenotype, dtype=np.float64)
    single_trait = phenotype_array.ndim == 1
    if single_trait:
        phenotype_matrix = phenotype_array.reshape(-1, 1)
    elif phenotype_array.ndim == 2:
        phenotype_matrix = phenotype_array
    else:
        raise ValueError("Phenotype input must be a vector or sample-by-trait matrix.")
    if phenotype_matrix.shape[0] != len(row_selection) or phenotype_matrix.shape[1] == 0:
        raise ValueError(
            f"Phenotype matrix shape {phenotype_matrix.shape} is incompatible with "
            f"{len(row_selection)} retained samples."
        )
    if not np.all(np.isfinite(phenotype_matrix)):
        raise ValueError("Phenotype matrix contains non-finite values.")
    score_bytes = 2 * m * phenotype_matrix.shape[1] * np.dtype(np.float64).itemsize
    if score_bytes > _MAX_IN_MEMORY_SCORE_BYTES:
        raise MemoryError(
            "Wide GxE scoring would allocate "
            f"{score_bytes / 1024**3:.2f} GiB for its two score matrices, exceeding "
            f"the {_MAX_IN_MEMORY_SCORE_BYTES / 1024**3:.1f} GiB safety limit. "
            "Select fewer phenotype columns per batch."
        )
    score_x = np.empty((m, phenotype_matrix.shape[1]), dtype=np.float64)
    score_w = np.empty((m, phenotype_matrix.shape[1]), dtype=np.float64)
    genotype_scale = str(reference.payload["genotype_scale"])
    ddof = int(reference.payload["environment_transform"]["ddof"])
    kernel_mode = str(reference.payload["kernel_mode"])

    blas_scope = (
        threadpool_limits(limits=num_threads) if num_threads is not None else nullcontext()
    )
    with blas_scope:
        for start in range(0, m, step_size):
            end = min(m, start + step_size)
            genotype = _read_genotype_block(
                bed,
                row_selection,
                start,
                end,
                genotype_scale=genotype_scale,
                ddof=ddof,
                eps_var=eps_var,
                num_threads=num_threads,
            )
            additive = _project_and_center(
                np.array(genotype, copy=True, order="F"), fixed_basis
            )
            interaction = _project_and_center(
                np.asarray(genotype * environment[:, None], dtype=np.float64, order="F"),
                fixed_basis,
            )
            if kernel_mode == "standardized":
                raw_norm_x = (
                    np.sum(additive * additive, axis=0, dtype=np.float64) / residual_rank
                )
                raw_norm_w = (
                    np.sum(interaction * interaction, axis=0, dtype=np.float64)
                    / residual_rank
                )
                if np.any(raw_norm_x <= eps_var) or np.any(raw_norm_w <= eps_var):
                    raise ValueError(
                        f"Projected feature variance is invalid in block [{start}:{end})."
                    )
                # SCALE_X/SCALE_W are the sealed feature-definition contract.  The
                # text diagonal stores them to 12 significant digits, so verify
                # those values against the exact block norms, then apply the exact
                # recomputed values.  This is still one genotype pass and reproduces
                # the generator's float64 phenotype scores rather than introducing
                # a second rounding at the last printed digit.
                exact_scale_x = 1.0 / np.sqrt(raw_norm_x)
                exact_scale_w = 1.0 / np.sqrt(raw_norm_w)
                if not np.allclose(
                    exact_scale_x,
                    reference.scale_x[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-10,
                ) or not np.allclose(
                    exact_scale_w,
                    reference.scale_w[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-10,
                ):
                    raise ValueError(
                        "Stored feature scales disagree with the supplied "
                        f"genotype/design in block [{start}:{end})."
                    )
                additive *= exact_scale_x[None, :]
                interaction *= exact_scale_w[None, :]

            observed_norm_x = (
                np.sum(additive * additive, axis=0, dtype=np.float64) / residual_rank
            )
            observed_norm_w = (
                np.sum(interaction * interaction, axis=0, dtype=np.float64) / residual_rank
            )
            if not np.allclose(
                observed_norm_x,
                reference.norm_x[start:end],
                rtol=1.0e-9,
                atol=1.0e-9,
            ) or not np.allclose(
                observed_norm_w,
                reference.norm_w[start:end],
                rtol=1.0e-9,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "Genotype scaling/projected feature norms disagree with the "
                    f"reference in block [{start}:{end})."
                )
            score_x[start:end, :] = (additive.T @ phenotype_matrix) / root_rank
            score_w[start:end, :] = (interaction.T @ phenotype_matrix) / root_rank
    if single_trait:
        return score_x[:, 0], score_w[:, 0]
    return score_x, score_w


def _relative_path(target: Path, manifest: Path) -> str:
    return os.path.relpath(target.resolve(), start=manifest.parent.resolve())


def _write_dataframe_temp(frame: pd.DataFrame, target: Path, staging_dir: Path) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=staging_dir
    )
    os.fchmod(fd, 0o600)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(
            temporary,
            sep="\t",
            index=False,
            compression="gzip",
            float_format="%.12g",
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_temp(payload: dict[str, Any], target: Path, staging_dir: Path) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=staging_dir
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_private_no_replace(temporary: Path, target: Path) -> tuple[int, int]:
    staged = temporary.stat()
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing GxE score output: {target}") from exc
    return staged.st_dev, staged.st_ino


def _verify_published_inodes(
    published: Sequence[tuple[Path, int, int]],
    expected_sha256: dict[Path, str],
    *,
    context: str,
) -> None:
    for path, device, inode in published:
        try:
            observed = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"A published {context} artifact disappeared before moments commit: {path}."
            ) from exc
        if observed.st_dev != device or observed.st_ino != inode:
            raise RuntimeError(
                f"A published {context} artifact was concurrently replaced before "
                f"moments commit: {path}."
            )
        if _sha256_file(path) != expected_sha256[path]:
            raise RuntimeError(
                f"A published {context} artifact was modified in place before "
                f"moments commit: {path}."
            )


def _with_output_lock(
    output_prefix: str | Path,
    action: Callable[
        [tuple[Path, Path, Path], Path, ExitStack], GxEPhenotypeScoreArtifacts
    ],
) -> GxEPhenotypeScoreArtifacts:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    targets = (
        Path(str(prefix) + ".gxe.gwas.tsv.gz"),
        Path(str(prefix) + ".gxe.gwis.tsv.gz"),
        Path(str(prefix) + ".gxe.moments.json"),
    )
    lock = Path(str(prefix) + ".gxe.score.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Another GxE score writer holds output prefix {prefix}.") from exc
    lock_stat = os.fstat(descriptor)
    os.close(descriptor)
    try:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing GxE phenotype-score output(s): "
                + ", ".join(str(path) for path in existing)
            )
        with tempfile.TemporaryDirectory(
            prefix=".gxe-score-stage-", dir=prefix.parent
        ) as stage_name:
            stage_dir = Path(stage_name)
            os.chmod(stage_dir, 0o700)
            with ExitStack() as resources:
                return action(targets, stage_dir, resources)
    finally:
        try:
            observed = lock.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if observed.st_dev == lock_stat.st_dev and observed.st_ino == lock_stat.st_ino:
                lock.unlink(missing_ok=True)


def score_phenotype_from_reference(
    *,
    reference_manifest: str | Path,
    bed_path: str | Path,
    env_path: str | Path,
    pheno_path: str | Path,
    output_prefix: str | Path,
    covar_path: str | Path | None = None,
    pheno_col: str | None = None,
    missing_values: Sequence[str] = ("-9", "NA", "NaN", "nan", ".", "None", "null"),
    step_size: int = 1000,
    eps_var: float = 1.0e-10,
    num_threads: int | None = None,
) -> GxEPhenotypeScoreArtifacts:
    """Score one quantitative phenotype against an existing GxE reference.

    The supplied environment/covariate/phenotype files must reproduce the
    reference's exact retained sample and common fixed-effect design.  The
    function verifies all schema-v3 hashes and invariants before decoding the
    BED, decodes each variant exactly once, and refuses to replace any output.
    """

    step_size = int(step_size)
    eps_var = float(eps_var)
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if not np.isfinite(eps_var) or eps_var <= 0.0:
        raise ValueError("eps_var must be positive and finite.")
    num_threads = _validate_num_threads(num_threads)
    prefix = os.path.abspath(_canonical_bfile_prefix(str(bed_path)))

    def run(
        targets: tuple[Path, Path, Path], stage_dir: Path, resources: ExitStack
    ) -> GxEPhenotypeScoreArtifacts:
        reference = _validate_reference_manifest(
            reference_manifest, scratch_dir=stage_dir
        )
        stable_prefix, stable_descriptors, stable_state = resources.enter_context(
            _stable_genotype_prefix(prefix, stage_dir)
        )
        total_samples, _ = _validate_genotype_files(stable_prefix, reference)
        _validate_variant_axis(stable_prefix, reference)
        (
            row_selection,
            environment,
            fixed_basis,
            phenotype,
            phenotype_name,
            residual_fraction,
            _,
        ) = _validate_analysis_inputs(
            prefix=stable_prefix,
            reference=reference,
            env_path=env_path,
            covar_path=covar_path,
            pheno_path=pheno_path,
            pheno_col=pheno_col,
            missing_values=tuple(str(value) for value in missing_values),
        )
        score_x, score_w = _score_one_genotype_pass(
            prefix=stable_prefix,
            reference=reference,
            row_selection=row_selection,
            environment=environment,
            fixed_basis=fixed_basis,
            phenotype=phenotype,
            total_samples=total_samples,
            step_size=step_size,
            eps_var=eps_var,
            num_threads=num_threads,
        )
        # Re-hash after the final decode.  These are staged controlled-data
        # copies on Hoffman, so a mismatch means the score could reflect a
        # moving/mixed PLINK input and must never be published.
        _validate_genotype_files(stable_prefix, reference)
        _assert_stable_genotype_snapshot(stable_descriptors, stable_state)

        gwas_target, gwis_target, moments_target = targets
        base = reference.variants.copy()
        base["N"] = int(reference.payload["n_samples"])
        base["DF"] = int(reference.payload["residual_rank"])
        base["SCORE_MODE"] = _SCORE_MODE
        gwas = base.copy()
        gwis = base.copy()
        gwas["SCORE"] = score_x
        gwis["SCORE"] = score_w

        temporary_paths: list[Path] = []
        published: list[tuple[Path, int, int]] = []
        try:
            temporary_gwas = _write_dataframe_temp(gwas, gwas_target, stage_dir)
            temporary_gwis = _write_dataframe_temp(gwis, gwis_target, stage_dir)
            temporary_paths.extend([temporary_gwas, temporary_gwis])
            moments = {
                "kind": _MOMENTS_KIND,
                "schema_version": _SCHEMA_VERSION,
                "analysis_fingerprint": reference.payload["analysis_fingerprint"],
                "variant_digest": reference.payload["variant_digest"],
                "phenotype": phenotype_name,
                "n_samples": int(reference.payload["n_samples"]),
                "residual_rank": int(reference.payload["residual_rank"]),
                "score_definition": _SCORE_DEFINITION,
                "reference_manifest_sha256": reference.manifest_sha256,
                "score_sha256": {
                    "gwas": _sha256_file(temporary_gwas),
                    "gwis": _sha256_file(temporary_gwis),
                },
                "q_nxe": float(np.dot(environment * phenotype, environment * phenotype)),
                "q_residual": float(np.dot(phenotype, phenotype)),
                "phenotype_residual_variance_fraction": residual_fraction,
                "files": {
                    "gwas": _relative_path(gwas_target, moments_target),
                    "gwis": _relative_path(gwis_target, moments_target),
                },
            }
            if reference.feature_cache_path is not None:
                moments["feature_cache_sha256"] = reference.feature_cache_sha256
                moments["feature_cache"] = {
                    "path": _relative_path(reference.feature_cache_path, moments_target),
                    "sha256": reference.feature_cache_sha256,
                }
            temporary_moments = _write_json_temp(moments, moments_target, stage_dir)
            temporary_paths.append(temporary_moments)
            expected_published_hashes = {
                target: _sha256_file(temporary)
                for temporary, target in zip(temporary_paths, targets)
            }
            for temporary, target in zip(temporary_paths, targets):
                if target == moments_target:
                    _verify_published_inodes(
                        published,
                        expected_published_hashes,
                        context="GxE phenotype-score",
                    )
                device, inode = _publish_private_no_replace(temporary, target)
                published.append((target, device, inode))
            _verify_published_inodes(
                published,
                expected_published_hashes,
                context="GxE phenotype-score",
            )
        except Exception:
            for target, device, inode in reversed(published):
                try:
                    observed = target.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if observed.st_dev == device and observed.st_ino == inode:
                    target.unlink(missing_ok=True)
            raise
        finally:
            for temporary in temporary_paths:
                temporary.unlink(missing_ok=True)
        return GxEPhenotypeScoreArtifacts(
            gwas=gwas_target, gwis=gwis_target, moments=moments_target
        )

    return _with_output_lock(output_prefix, run)


def _wide_targets(
    output_prefix: str | Path, traits: Sequence[str]
) -> dict[str, tuple[Path, Path, Path]]:
    prefix = Path(output_prefix)
    targets: dict[str, tuple[Path, Path, Path]] = {}
    for trait in traits:
        trait_prefix = Path(f"{prefix}.{trait}")
        targets[trait] = (
            Path(str(trait_prefix) + ".gxe.gwas.tsv.gz"),
            Path(str(trait_prefix) + ".gxe.gwis.tsv.gz"),
            Path(str(trait_prefix) + ".gxe.moments.json"),
        )
    return targets


def _with_wide_output_lock(
    output_prefix: str | Path,
    targets: dict[str, tuple[Path, Path, Path]],
    action: Callable[[Path, ExitStack], dict[str, GxEPhenotypeScoreArtifacts]],
) -> dict[str, GxEPhenotypeScoreArtifacts]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(prefix) + ".gxe.batch.score.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another wide GxE score writer holds output prefix {prefix}."
        ) from exc
    lock_stat = os.fstat(descriptor)
    os.close(descriptor)
    try:
        all_targets = [path for triplet in targets.values() for path in triplet]
        existing = [path for path in all_targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing wide GxE phenotype-score output(s): "
                + ", ".join(str(path) for path in existing)
            )
        with tempfile.TemporaryDirectory(
            prefix=".gxe-wide-score-stage-", dir=prefix.parent
        ) as stage_name:
            stage_dir = Path(stage_name)
            os.chmod(stage_dir, 0o700)
            with ExitStack() as resources:
                return action(stage_dir, resources)
    finally:
        try:
            observed = lock.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if observed.st_dev == lock_stat.st_dev and observed.st_ino == lock_stat.st_ino:
                lock.unlink(missing_ok=True)


def _write_wide_score_bundles(
    *,
    reference: _ValidatedReference,
    traits: Sequence[str],
    score_x: np.ndarray,
    score_w: np.ndarray,
    q_nxe: np.ndarray,
    q_residual: np.ndarray,
    residual_fraction: np.ndarray,
    targets: dict[str, tuple[Path, Path, Path]],
    staging_dir: Path,
) -> dict[str, GxEPhenotypeScoreArtifacts]:
    trait_count = len(traits)
    expected_score_shape = (len(reference.variants), trait_count)
    if score_x.shape != expected_score_shape or score_w.shape != expected_score_shape:
        raise RuntimeError(
            f"Wide score matrices must have shape {expected_score_shape}; "
            f"got {score_x.shape} and {score_w.shape}."
        )
    for name, values in {
        "q_nxe": q_nxe,
        "q_residual": q_residual,
        "residual_fraction": residual_fraction,
    }.items():
        if np.asarray(values).shape != (trait_count,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Wide {name} values are incomplete or non-finite.")

    base = reference.variants.copy()
    base["N"] = int(reference.payload["n_samples"])
    base["DF"] = int(reference.payload["residual_rank"])
    base["SCORE_MODE"] = _SCORE_MODE
    temporary_pairs: list[tuple[Path, Path]] = []
    published: list[tuple[Path, int, int]] = []
    results: dict[str, GxEPhenotypeScoreArtifacts] = {}
    try:
        for index, trait in enumerate(traits):
            gwas_target, gwis_target, moments_target = targets[trait]
            gwas = base.copy()
            gwis = base.copy()
            gwas["SCORE"] = score_x[:, index]
            gwis["SCORE"] = score_w[:, index]
            temporary_gwas = _write_dataframe_temp(gwas, gwas_target, staging_dir)
            temporary_pairs.append((temporary_gwas, gwas_target))
            temporary_gwis = _write_dataframe_temp(gwis, gwis_target, staging_dir)
            temporary_pairs.append((temporary_gwis, gwis_target))
            moments = {
                "kind": _MOMENTS_KIND,
                "schema_version": _SCHEMA_VERSION,
                "analysis_fingerprint": reference.payload["analysis_fingerprint"],
                "variant_digest": reference.payload["variant_digest"],
                "phenotype": trait,
                "n_samples": int(reference.payload["n_samples"]),
                "residual_rank": int(reference.payload["residual_rank"]),
                "score_definition": _SCORE_DEFINITION,
                "reference_manifest_sha256": reference.manifest_sha256,
                "score_sha256": {
                    "gwas": _sha256_file(temporary_gwas),
                    "gwis": _sha256_file(temporary_gwis),
                },
                "q_nxe": float(q_nxe[index]),
                "q_residual": float(q_residual[index]),
                "phenotype_residual_variance_fraction": float(
                    residual_fraction[index]
                ),
                "files": {
                    "gwas": _relative_path(gwas_target, moments_target),
                    "gwis": _relative_path(gwis_target, moments_target),
                },
            }
            if reference.feature_cache_path is not None:
                moments["feature_cache_sha256"] = reference.feature_cache_sha256
                moments["feature_cache"] = {
                    "path": _relative_path(reference.feature_cache_path, moments_target),
                    "sha256": reference.feature_cache_sha256,
                }
            temporary_moments = _write_json_temp(moments, moments_target, staging_dir)
            temporary_pairs.append((temporary_moments, moments_target))
            results[trait] = GxEPhenotypeScoreArtifacts(
                gwas=gwas_target, gwis=gwis_target, moments=moments_target
            )

        expected_published_hashes = {
            target: _sha256_file(temporary) for temporary, target in temporary_pairs
        }
        publication_order = [
            pair for pair in temporary_pairs if not pair[1].name.endswith(".gxe.moments.json")
        ] + [
            pair for pair in temporary_pairs if pair[1].name.endswith(".gxe.moments.json")
        ]
        for temporary, target in publication_order:
            if target.name.endswith(".gxe.moments.json"):
                _verify_published_inodes(
                    published,
                    expected_published_hashes,
                    context="wide GxE phenotype-score",
                )
            device, inode = _publish_private_no_replace(temporary, target)
            published.append((target, device, inode))
        _verify_published_inodes(
            published,
            expected_published_hashes,
            context="wide GxE phenotype-score",
        )
    except Exception:
        for target, device, inode in reversed(published):
            try:
                observed = target.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if observed.st_dev == device and observed.st_ino == inode:
                target.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in temporary_pairs:
            temporary.unlink(missing_ok=True)
    return results


def score_phenotypes_from_reference(
    *,
    reference_manifest: str | Path,
    bed_path: str | Path,
    env_path: str | Path,
    pheno_path: str | Path,
    output_prefix: str | Path,
    covar_path: str | Path | None = None,
    pheno_cols: Sequence[str] | None = None,
    missing_values: Sequence[str] = ("-9", "NA", "NaN", "nan", ".", "None", "null"),
    step_size: int = 1000,
    eps_var: float = 1.0e-10,
    num_threads: int | None = None,
) -> dict[str, GxEPhenotypeScoreArtifacts]:
    """Score an all-complete wide quantitative phenotype table in one BED pass.

    The environment and covariates first establish the exact reference cohort.
    Every selected trait must then be finite and complete on every one of those
    rows; the API never creates trait-specific sample sets.  All trait triplets
    are staged and published as one all-or-none batch.
    """

    step_size = int(step_size)
    eps_var = float(eps_var)
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if not np.isfinite(eps_var) or eps_var <= 0.0:
        raise ValueError("eps_var must be positive and finite.")
    num_threads = _validate_num_threads(num_threads)
    prefix = os.path.abspath(_canonical_bfile_prefix(str(bed_path)))

    # Trait names are needed to reserve every final path before expensive hash
    # and genotype work.  Read only the header here; the validated full table is
    # loaded exactly once later under the batch lock.
    header = pd.read_csv(pheno_path, sep=r"\s+", nrows=0)
    available = [str(column) for column in header.columns if column not in {"FID", "IID"}]
    if pheno_cols is None:
        traits = _validate_trait_names(available)
    elif isinstance(pheno_cols, str):
        traits = _validate_trait_names([pheno_cols])
    else:
        traits = _validate_trait_names(pheno_cols)
    missing_columns = [trait for trait in traits if trait not in available]
    if missing_columns:
        raise ValueError(f"Wide phenotype file is missing selected trait columns {missing_columns}.")
    targets = _wide_targets(output_prefix, traits)

    def run(
        stage_dir: Path, resources: ExitStack
    ) -> dict[str, GxEPhenotypeScoreArtifacts]:
        reference = _validate_reference_manifest(
            reference_manifest, scratch_dir=stage_dir
        )
        stable_prefix, stable_descriptors, stable_state = resources.enter_context(
            _stable_genotype_prefix(prefix, stage_dir)
        )
        total_samples, _ = _validate_genotype_files(stable_prefix, reference)
        _validate_variant_axis(stable_prefix, reference)
        row_selection, environment, fixed_basis = _validate_reference_design_inputs(
            prefix=stable_prefix,
            reference=reference,
            env_path=env_path,
            covar_path=covar_path,
            missing_values=tuple(str(value) for value in missing_values),
        )
        estimated_working_bytes = (
            6 * len(row_selection) * len(traits) * np.dtype(np.float64).itemsize
            + 2 * len(reference.variants) * len(traits) * np.dtype(np.float64).itemsize
        )
        if estimated_working_bytes > _MAX_WIDE_WORKING_BYTES:
            raise MemoryError(
                "Wide GxE scoring conservatively estimates "
                f"{estimated_working_bytes / 1024**3:.2f} GiB of phenotype/score workspace, "
                f"exceeding the {_MAX_WIDE_WORKING_BYTES / 1024**3:.1f} GiB safety limit. "
                "Split the selected traits into smaller batches."
            )
        validated_traits, phenotype, residual_fraction = (
            _read_and_project_wide_phenotypes(
                pheno_path=pheno_path,
                pheno_cols=traits,
                fam_path=Path(stable_prefix + ".fam"),
                row_selection=row_selection,
                fixed_basis=fixed_basis,
                residual_rank=int(reference.payload["residual_rank"]),
                missing_values=tuple(str(value) for value in missing_values),
            )
        )
        if validated_traits != traits:
            raise RuntimeError("Wide phenotype trait order changed during validation.")
        score_x, score_w = _score_one_genotype_pass(
            prefix=stable_prefix,
            reference=reference,
            row_selection=row_selection,
            environment=environment,
            fixed_basis=fixed_basis,
            phenotype=phenotype,
            total_samples=total_samples,
            step_size=step_size,
            eps_var=eps_var,
            num_threads=num_threads,
        )
        _validate_genotype_files(stable_prefix, reference)
        _assert_stable_genotype_snapshot(stable_descriptors, stable_state)
        # Match the scalar scorer's dot-product reduction exactly for each
        # trait; genotype scores themselves are produced by the single GEMM
        # above.  These O(NT) reductions are negligible beside BED decoding.
        q_nxe = np.asarray(
            [
                np.dot(environment * phenotype[:, index], environment * phenotype[:, index])
                for index in range(len(traits))
            ],
            dtype=np.float64,
        )
        q_residual = np.asarray(
            [np.dot(phenotype[:, index], phenotype[:, index]) for index in range(len(traits))],
            dtype=np.float64,
        )
        if not np.allclose(
            q_residual,
            float(reference.payload["residual_rank"]),
            rtol=1.0e-10,
            atol=1.0e-8,
        ):
            raise RuntimeError("Wide phenotype normalization failed the q_residual=rank contract.")
        return _write_wide_score_bundles(
            reference=reference,
            traits=traits,
            score_x=np.asarray(score_x, dtype=np.float64),
            score_w=np.asarray(score_w, dtype=np.float64),
            q_nxe=q_nxe,
            q_residual=q_residual,
            residual_fraction=residual_fraction,
            targets=targets,
            staging_dir=stage_dir,
        )

    return _with_wide_output_lock(output_prefix, targets, run)


__all__ = [
    "GxEPhenotypeScoreArtifacts",
    "score_phenotype_from_reference",
    "score_phenotypes_from_reference",
]
