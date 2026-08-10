"""Validated merging of disjoint randomized GxE reference shards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .gwe_ldscore import (
    _FEATURE_CACHE_ARRAY_DTYPES,
    _FEATURE_CACHE_SCHEMA_VERSION,
    _BACKEND_PROVENANCE_SCHEMA_VERSION,
    _ndarray_sha256,
    _validate_gxe_annotation_names,
    _validate_feature_cache_semantics,
    _validate_backend_provenance,
)


_SHARD_KIND = "summit.gxe.reference_shard"
_SHARD_IDENTITY_KIND = "summit.gxe.reference_shard_identity"
_CACHE_KIND = "summit.gxe.feature_cache"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


@contextmanager
def _snapshot_file(
    path: Path,
    *,
    scratch_dir: Path,
    expected_sha256: str | None = None,
):
    """Yield a private, owner-only snapshot and its digest.

    Hashing and then parsing one open source descriptor prevents pathname
    replacement, but another writer can still mutate that inode in place.  A
    private snapshot makes the exact bytes hashed identical to the bytes later
    parsed without retaining a potentially large compressed artifact in RAM.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.TemporaryFile(
        prefix=".gxe-validated-input-", suffix=".tmp", dir=scratch_dir
    ) as snapshot:
        os.fchmod(snapshot.fileno(), 0o600)
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                snapshot.write(block)
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(f"Input artifact failed its SHA-256 check: {path}.")
        snapshot.seek(0)
        yield snapshot, observed


def _load_json_bytes(value: bytes, path: Path) -> dict:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}.") from exc
    value = parsed
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return value


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _atomic_frame(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.fchmod(fd, 0o600)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, **kwargs)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(value: dict, path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
                f"A published {context} artifact disappeared before bundle commit: {path}."
            ) from exc
        if observed.st_dev != device or observed.st_ino != inode:
            raise RuntimeError(
                f"A published {context} artifact was concurrently replaced before "
                f"bundle commit: {path}."
            )
        if _sha256(path) != expected_sha256[path]:
            raise RuntimeError(
                f"A published {context} artifact was modified in place before "
                f"bundle commit: {path}."
            )


def _read_cache(path: Path, *, scratch_dir: Path) -> tuple[dict, dict[str, np.ndarray], str]:
    with _snapshot_file(path, scratch_dir=scratch_dir) as (handle, cache_hash):
        with np.load(handle, allow_pickle=False) as bundle:
            expected_members = {*_FEATURE_CACHE_ARRAY_DTYPES, "metadata_json"}
            if (
                len(bundle.files) != len(expected_members)
                or set(bundle.files) != expected_members
            ):
                raise ValueError(
                    "Feature cache contains unexpected, duplicate, or missing schema-v2 arrays."
                )
            if "metadata_json" not in bundle.files:
                raise ValueError(f"Feature cache lacks metadata_json: {path}.")
            metadata = json.loads(str(bundle["metadata_json"].item()))
            if (
                metadata.get("kind") != _CACHE_KIND
                or metadata.get("schema_version") != _FEATURE_CACHE_SCHEMA_VERSION
            ):
                raise ValueError(f"Unsupported GxE feature cache: {path}.")
            hashes = metadata.get("array_sha256")
            names = set(bundle.files) - {"metadata_json"}
            if not isinstance(hashes, dict) or set(hashes) != names:
                raise ValueError("Feature cache does not bind every stored array.")
            arrays = {name: np.asarray(bundle[name]).copy() for name in sorted(names)}
    _validate_feature_cache_semantics(metadata, arrays)
    return metadata, arrays, cache_hash


def _variant_frame(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": arrays["variant_chr"].astype(str),
            "SNP": arrays["variant_snp"].astype(str),
            "BP": arrays["variant_bp"].astype(np.int64),
            "A1": arrays["variant_a1"].astype(str),
            "A2": arrays["variant_a2"].astype(str),
        }
    )


def _read_panel_file(
    path: Path,
    expected_sha256: str,
    variants: pd.DataFrame,
    names: list[str],
    *,
    scratch_dir: Path,
) -> np.ndarray:
    with _snapshot_file(
        path, scratch_dir=scratch_dir, expected_sha256=expected_sha256
    ) as (handle, _):
        frame = pd.read_csv(
            handle,
            sep=r"\s+",
            dtype={"CHR": str, "SNP": str},
            compression="gzip",
        )
    required = ["CHR", "SNP", "BP", *names]
    observed_columns = frame.columns.astype(str).tolist()
    if observed_columns != required or len(set(observed_columns)) != len(observed_columns):
        raise ValueError(
            f"Invalid shard panel columns in {path}; expected exactly {required}, "
            f"observed {observed_columns}."
        )
    if len(frame) != len(variants):
        raise ValueError(
            f"Invalid shard panel row count in {path}; expected {len(variants)}, "
            f"observed {len(frame)}."
        )
    for column in ("CHR", "SNP", "BP"):
        if not np.array_equal(frame[column].astype(str), variants[column].astype(str)):
            raise ValueError(f"Shard panel {path} is misaligned at {column}.")
    values = frame[names].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"Shard panel contains a non-finite or negative value: {path}.")
    # Squares may serialize as signed zero.  Canonicalize it so numerical
    # duplicate detection cannot distinguish +0.0 from -0.0 byte patterns.
    values[values == 0.0] = 0.0
    return values


def _merge_reference_shards_impl(
    shard_manifests: Sequence[str | Path],
    *,
    feature_cache_path: str | Path,
    output_prefix: str | Path,
    allow_low_probe_jackknife: bool = False,
) -> Path:
    """Merge disjoint probe shards into one fit-able schema-v3 reference."""
    manifests = [Path(value).expanduser().resolve() for value in shard_manifests]
    if not manifests:
        raise ValueError("At least one GxE reference shard is required.")
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = prefix.parent
    paths = {
        "xx": Path(str(prefix) + ".gxx.ldscore.gz"),
        "xw": Path(str(prefix) + ".gxe.ldscore.gz"),
        "wx": Path(str(prefix) + ".exg.ldscore.gz"),
        "ww": Path(str(prefix) + ".gee.ldscore.gz"),
        "diagonal": Path(str(prefix) + ".gxe.diag.tsv.gz"),
        "reference": Path(str(prefix) + ".gxe.ref.json"),
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite merged GxE reference artifacts: {existing}.")
    cache_path = Path(feature_cache_path).expanduser().resolve()
    metadata, cache, cache_hash = _read_cache(cache_path, scratch_dir=scratch_dir)
    variants = _variant_frame(cache)
    annotations = np.asarray(cache["annotations"], dtype=np.float64)
    names = _validate_gxe_annotation_names(metadata["annotation_names"])
    if annotations.shape != (len(variants), len(names)):
        raise ValueError("Feature-cache annotation matrix has an invalid shape.")

    loaded: list[dict] = []
    intervals = []
    common = None
    common_keys = (
        "analysis_fingerprint", "variant_digest", "annotation_digest",
        "jackknife_digest", "kernel_mode", "genotype_scale", "ld_scale",
        "annotation_names",
    )
    random_keys = ("distribution", "algorithm", "seed", "dtype", "step_size")
    identity_random_keys = (
        "distribution", "algorithm", "seed", "dtype", "step_size",
        "num_vectors", "probe_offset", "probe_stop",
    )
    for manifest_path in manifests:
        manifest_raw = _read_bytes(manifest_path)
        manifest_hash = _sha256_bytes(manifest_raw)
        shard = _load_json_bytes(manifest_raw, manifest_path)
        if shard.get("kind") != _SHARD_KIND or shard.get("schema_version") != 2:
            raise ValueError(f"Unsupported or fit-able input passed as a shard: {manifest_path}.")
        cache_decl = shard.get("feature_cache")
        if not isinstance(cache_decl, dict) or cache_decl.get("sha256") != cache_hash:
            raise ValueError(f"Shard was generated from a different feature cache: {manifest_path}.")
        backend_provenance = shard.get("backend_provenance")
        _validate_backend_provenance(
            backend_provenance, expected_stage="reference_shard"
        )
        feature_backend_provenance = shard.get("feature_backend_provenance")
        cache_backend_provenance = metadata.get("backend_provenance")
        if feature_backend_provenance != cache_backend_provenance:
            raise ValueError(
                f"Shard feature-backend provenance differs from its cache: {manifest_path}."
            )
        config = {key: shard.get(key) for key in common_keys}
        randomization = shard.get("randomization")
        if not isinstance(randomization, dict):
            raise ValueError(f"Shard lacks randomization metadata: {manifest_path}.")
        config["randomization"] = {key: randomization.get(key) for key in random_keys}
        if common is None:
            common = config
        elif config != common:
            raise ValueError(f"Shard configuration differs from the first shard: {manifest_path}.")
        start = randomization.get("probe_offset")
        stop = randomization.get("probe_stop")
        count = randomization.get("num_vectors")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (start, stop, count)):
            raise ValueError(f"Shard has invalid probe identity metadata: {manifest_path}.")
        if (
            start < 0
            or start >= 2**64
            or count <= 0
            or count > 2**64
            or stop != start + count
            or stop > 2**64
        ):
            raise ValueError(f"Shard has an invalid probe interval [{start},{stop}): {manifest_path}.")

        files = shard.get("files")
        hashes = shard.get("artifact_sha256")
        expected_artifacts = {"xx", "xw", "wx", "ww", "identity"}
        if metadata.get("jackknife_labels") is not None:
            expected_artifacts.add("jackknife")
        if (
            not isinstance(files, dict)
            or not isinstance(hashes, dict)
            or set(files) != expected_artifacts
            or set(hashes) != expected_artifacts
        ):
            raise ValueError(
                f"Shard artifact declarations are incomplete or unexpected: {manifest_path}."
            )
        identity_path = _resolve(manifest_path, str(files["identity"]))
        identity_raw = _read_bytes(identity_path)
        if _sha256_bytes(identity_raw) != hashes["identity"]:
            raise ValueError(f"Shard identity sidecar failed its SHA-256 check: {identity_path}.")
        identity = _load_json_bytes(identity_raw, identity_path)
        if (
            identity.get("kind") != _SHARD_IDENTITY_KIND
            or identity.get("schema_version") != 1
        ):
            raise ValueError(f"Unsupported shard identity sidecar: {identity_path}.")
        identity_common = {key: identity.get(key) for key in common_keys}
        if identity_common != {key: shard.get(key) for key in common_keys}:
            raise ValueError(f"Shard identity configuration disagrees with its manifest: {manifest_path}.")
        if identity.get("feature_cache_sha256") != cache_hash:
            raise ValueError(f"Shard identity names a different feature cache: {manifest_path}.")
        if (
            identity.get("backend_provenance") != backend_provenance
            or identity.get("feature_backend_provenance")
            != feature_backend_provenance
        ):
            raise ValueError(
                f"Shard identity backend provenance disagrees with its manifest: {manifest_path}."
            )
        expected_identity_randomization = {
            key: randomization.get(key) for key in identity_random_keys
        }
        if identity.get("randomization") != expected_identity_randomization:
            raise ValueError(f"Shard identity probe interval disagrees with its manifest: {manifest_path}.")
        artifact_hashes = {key: hashes[key] for key in expected_artifacts - {"identity"}}
        if identity.get("artifact_sha256") != artifact_hashes:
            raise ValueError(f"Shard identity artifact hashes disagree with its manifest: {manifest_path}.")
        intervals.append((start, stop, count, manifest_path))
        loaded.append(
            {
                "path": manifest_path,
                "manifest": shard,
                "manifest_sha256": manifest_hash,
                "count": count,
                "backend_provenance": dict(backend_provenance),
            }
        )

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"Duplicate/overlapping probe identities: [{previous[0]},{previous[1]}) and "
                f"[{current[0]},{current[1]})."
            )
        if current[0] != previous[1]:
            raise ValueError(
                "GxE shard probe intervals must be contiguous so merged "
                "probe_offset/probe_stop metadata have an unambiguous count; "
                f"gap between [{previous[0]},{previous[1]}) and "
                f"[{current[0]},{current[1]})."
            )
    expected_common = {
        "analysis_fingerprint": metadata["analysis_fingerprint"],
        "variant_digest": metadata["variant_digest"],
        "annotation_digest": metadata["annotation_digest"],
        "jackknife_digest": metadata["jackknife_digest"],
        "kernel_mode": metadata["kernel_mode"],
        "genotype_scale": metadata["genotype_scale"],
        "ld_scale": "cross_product_over_rank_squared",
        "annotation_names": metadata["annotation_names"],
    }
    for key, value in expected_common.items():
        if common.get(key) != value:
            raise ValueError(f"Shard configuration field {key!r} differs from the feature cache.")
    random_common = common["randomization"]
    if random_common.get("distribution") not in {"rademacher", "gaussian", "normal", "spherical"}:
        raise ValueError("Shard randomization has an unsupported probe distribution.")
    if random_common.get("algorithm") != "philox_per_probe_block_v1":
        raise ValueError("Shard randomization has an unsupported probe generator contract.")
    if random_common.get("dtype") not in {"float32", "float64"}:
        raise ValueError("Shard randomization has an unsupported floating-point dtype.")
    seed = random_common.get("seed")
    step_size = random_common.get("step_size")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed >= 2**64
    ):
        raise ValueError("Shard randomization seed must be a uint64 JSON integer.")
    if isinstance(step_size, bool) or not isinstance(step_size, int) or step_size <= 0:
        raise ValueError("Shard randomization step_size must be a positive JSON integer.")
    order = {path: idx for idx, (_, _, _, path) in enumerate(intervals)}
    loaded.sort(key=lambda value: order[value["path"]])
    total_vectors = sum(value["count"] for value in loaded)

    weighted_panels = {
        key: np.zeros((len(variants), len(names)), dtype=np.float64)
        for key in ("xx", "xw", "wx", "ww")
    }
    weighted_within = None
    block_labels = None
    seen_contributions: dict[str, Path] = {}
    panel_keys = ("xx", "xw", "wx", "ww")
    for record in loaded:
        manifest_path = record["path"]
        shard = record["manifest"]
        count = record["count"]
        files = shard.get("files")
        hashes = shard.get("artifact_sha256")
        contribution_digest = hashlib.sha256()
        for key in panel_keys:
            panel_path = _resolve(manifest_path, str(files[key]))
            values = _read_panel_file(
                panel_path, hashes[key], variants, names, scratch_dir=scratch_dir
            )
            contribution_digest.update(key.encode("ascii"))
            contribution_digest.update(bytes.fromhex(_ndarray_sha256(values)))
            weighted_panels[key] += count * values
            del values
        has_jackknife = "jackknife" in files
        if has_jackknife != (metadata.get("jackknife_labels") is not None):
            raise ValueError("Shard jackknife artifacts disagree with the feature cache.")
        if has_jackknife:
            jack_path = _resolve(manifest_path, str(files["jackknife"]))
            expected_jack_hash = hashes.get("jackknife")
            if expected_jack_hash is None:
                raise ValueError(f"Shard jackknife lacks a SHA-256 declaration: {jack_path}.")
            with _snapshot_file(
                jack_path,
                scratch_dir=scratch_dir,
                expected_sha256=expected_jack_hash,
            ) as (handle, _):
                with np.load(handle, allow_pickle=False) as bundle:
                    expected_npz = {"block_labels", *{f"within_{key}" for key in panel_keys}}
                    if (
                        len(bundle.files) != len(expected_npz)
                        or set(bundle.files) != expected_npz
                    ):
                        raise ValueError(
                            f"Shard jackknife has unexpected, duplicate, or missing arrays: {jack_path}."
                        )
                    label_array = np.asarray(bundle["block_labels"])
                    if label_array.ndim != 1 or label_array.dtype.kind != "U":
                        raise ValueError(
                            f"Shard jackknife block_labels must be a one-dimensional Unicode array: {jack_path}."
                        )
                    labels = tuple(label_array.tolist())
                    expected_labels = tuple(str(value) for value in metadata["jackknife_labels"])
                    if labels != expected_labels:
                        raise ValueError("Shard jackknife block labels differ from the feature cache.")
                    if weighted_within is None:
                        block_labels = labels
                        expected_shape = (len(labels), len(names), len(names))
                        weighted_within = {
                            key: np.zeros(expected_shape, dtype=np.float64) for key in panel_keys
                        }
                    for key in panel_keys:
                        value = np.asarray(bundle[f"within_{key}"])
                        if (
                            value.dtype != np.dtype(np.float64)
                            or value.shape != weighted_within[key].shape
                            or not np.all(np.isfinite(value))
                            or np.any(value < 0.0)
                        ):
                            raise ValueError(f"Shard within_{key} has an invalid shape or value.")
                        value[value == 0.0] = 0.0
                        contribution_digest.update(f"within_{key}".encode("ascii"))
                        contribution_digest.update(bytes.fromhex(_ndarray_sha256(value)))
                        weighted_within[key] += count * value
                        del value
            contribution_digest.update("\x1f".join(labels).encode("utf-8"))

        canonical_digest = contribution_digest.hexdigest()
        duplicate = seen_contributions.get(canonical_digest)
        if duplicate is not None:
            raise ValueError(
                "Duplicate shard numerical contributions were relabelled as distinct probe intervals: "
                f"{duplicate} and {manifest_path}."
            )
        seen_contributions[canonical_digest] = manifest_path
        record["contribution_sha256"] = canonical_digest

    for value in weighted_panels.values():
        value /= total_vectors
    panels = weighted_panels
    within = weighted_within
    if within is not None:
        for value in within.values():
            value /= total_vectors
    low_probe_override = False
    if within is not None and total_vectors < 100:
        if not allow_low_probe_jackknife:
            raise ValueError(
                "A fit-able GxE jackknife reference requires at least 100 merged probes; "
                f"got {total_vectors}. Pass allow_low_probe_jackknife=True only for an "
                "explicit diagnostic merge."
            )
        low_probe_override = True

    if within is not None:
        paths["jackknife"] = Path(str(prefix) + ".gxe.jackknife.npz")
        if paths["jackknife"].exists():
            raise FileExistsError(
                "Refusing to overwrite merged GxE reference artifact: "
                f"{paths['jackknife']}."
            )

    output_files = {key: path for key, path in paths.items() if key != "reference"}
    manifest_path = paths["reference"]
    relative_files = {
        key: os.path.relpath(path, start=manifest_path.parent) for key, path in output_files.items()
    }

    # Stage the complete multi-file bundle in the destination filesystem and
    # publish it with no-replace hard links, placing the manifest last.  Any
    # failure rolls back only files created by this call, so a retry cannot see
    # a stale mixture of old and new shard products.
    published: list[tuple[Path, int, int]] = []
    with tempfile.TemporaryDirectory(prefix=".gxe-merge-stage-", dir=prefix.parent) as stage_name:
        stage_dir = Path(stage_name)
        os.chmod(stage_dir, 0o700)
        staged = {key: stage_dir / path.name for key, path in paths.items()}
        score_variants = variants[["CHR", "SNP", "BP"]]
        try:
            for key in ("xx", "xw", "wx", "ww"):
                frame = pd.concat(
                    [score_variants, pd.DataFrame(panels[key], columns=names)], axis=1
                )
                _atomic_frame(
                    frame,
                    staged[key],
                    sep="\t",
                    compression="gzip",
                    float_format="%.10g",
                )
                del frame
            diagonal = variants.copy()
            diagonal["NORM_X"] = cache["norm_x"]
            diagonal["NORM_W"] = cache["norm_w"]
            diagonal["SCALE_X"] = cache["scale_x"]
            diagonal["SCALE_W"] = cache["scale_w"]
            diagonal["DNXE_X"] = cache["diag_nxe_x"]
            diagonal["DNXE_W"] = cache["diag_nxe_w"]
            diagonal["CORR_XW"] = cache["corr_xw"]
            for index in range(len(names)):
                diagonal[f"ANNOT_{index}"] = annotations[:, index]
            if within is not None:
                diagonal["BLOCK"] = cache["jackknife_ids"].astype(np.int32)
            _atomic_frame(
                diagonal,
                staged["diagonal"],
                sep="\t",
                compression="gzip",
                float_format="%.12g",
            )
            if within is not None:
                _atomic_npz(
                    staged["jackknife"],
                    block_labels=np.asarray(block_labels, dtype=np.str_),
                    **{f"within_{key}": value for key, value in within.items()},
                )

            output_hashes = {
                key: _sha256(staged[key]) for key in output_files
            }
            payload = {
                "kind": "summit.gxe.reference",
                "schema_version": 3,
                "analysis_fingerprint": metadata["analysis_fingerprint"],
                "variant_digest": metadata["variant_digest"],
                "n_samples": int(metadata["n_samples"]),
                "fixed_effect_rank_excluding_intercept": int(metadata["fixed_effect_rank_excluding_intercept"]),
                "residual_rank": int(metadata["residual_rank"]),
                "environment": metadata["environment"],
                "environment_transform": metadata["environment_transform"],
                "covariates": metadata["covariates"],
                "kernel_mode": metadata["kernel_mode"],
                "genotype_scale": metadata["genotype_scale"],
                "ld_scale": "cross_product_over_rank_squared",
                "null_corrected": False,
                "annotation_names": names,
                "annotation_masses": metadata["annotation_masses"],
                "feature_diagnostics": metadata["feature_diagnostics"],
                "resource_estimates": {"merged_shards": len(loaded)},
                "trace_nxe": float(metadata["trace_nxe"]),
                "trace_nxe_sq": float(metadata["trace_nxe_sq"]),
                "randomization": {
                    **common["randomization"],
                    "num_vectors": total_vectors,
                    "probe_ranges": [[start, stop] for start, stop, _, _ in intervals],
                    "probe_offset": intervals[0][0],
                    "probe_stop": intervals[-1][1],
                    "low_probe_jackknife_override": low_probe_override,
                },
                "genotype_files": metadata["genotype_files"],
                "feature_cache": {
                    "path": os.path.relpath(cache_path, start=manifest_path.parent),
                    "sha256": cache_hash,
                },
                "files": relative_files,
                "artifact_sha256": output_hashes,
                "merge_provenance": {
                    "shards": [
                        {
                            "path": os.path.relpath(record["path"], start=manifest_path.parent),
                            # This is the digest of the exact bytes parsed
                            # above, not a second pathname read.
                            "sha256": record["manifest_sha256"],
                            "contribution_sha256": record["contribution_sha256"],
                            "probe_offset": int(
                                record["manifest"]["randomization"]["probe_offset"]
                            ),
                            "probe_stop": int(
                                record["manifest"]["randomization"]["probe_stop"]
                            ),
                        }
                        for record in loaded
                    ]
                },
            }
            source_backends = [record["backend_provenance"] for record in loaded]
            max_global_width = max(
                int(value["actual_global_2b_source_columns"])
                for value in source_backends
            )
            max_jackknife_width = max(
                int(value["actual_jackknife_4b_source_columns"])
                for value in source_backends
            )
            merged_backend = {
                "schema_version": _BACKEND_PROVENANCE_SCHEMA_VERSION,
                "artifact_stage": "reference",
                "backend_name": "summit_shard_merge",
                "backend_version": "gxe_merge_v1",
                "source_commit": "unknown",
                "source_tree_sha256": None,
                "native_binary_sha256": None,
                "compile_options": None,
                "native_workspace_cap_bytes": 0,
                "configured_target_panel_columns": 0,
                "actual_global_2b_source_columns": max_global_width,
                "actual_jackknife_4b_source_columns": max_jackknife_width,
                "actual_target_source_columns": (
                    max_jackknife_width or max_global_width
                ),
            }
            _validate_backend_provenance(
                merged_backend, expected_stage="reference"
            )
            payload["backend_provenance"] = merged_backend
            payload["feature_backend_provenance"] = metadata.get(
                "backend_provenance"
            )
            payload["shard_backend_provenance"] = source_backends
            if within is not None:
                payload["jackknife"] = {
                    "method": "two_sided_snp_kernel_deletion",
                    "num_blocks": len(block_labels),
                    "block_labels": list(block_labels),
                    "within_scale": "cross_product_over_rank_squared",
                }
            _atomic_json(payload, staged["reference"])
            expected_published_hashes = {
                **{paths[key]: value for key, value in output_hashes.items()},
                paths["reference"]: _sha256(staged["reference"]),
            }

            for key in [*output_files, "reference"]:
                if key == "reference":
                    _verify_published_inodes(
                        published,
                        expected_published_hashes,
                        context="merged GxE reference",
                    )
                staged_stat = staged[key].stat()
                try:
                    os.link(staged[key], paths[key])
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Refusing to overwrite merged GxE reference artifact: {paths[key]}."
                    ) from exc
                published.append(
                    (paths[key], staged_stat.st_dev, staged_stat.st_ino)
                )
            _verify_published_inodes(
                published,
                expected_published_hashes,
                context="merged GxE reference",
            )
        except Exception:
            for path, device, inode in reversed(published):
                try:
                    observed = path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if observed.st_dev == device and observed.st_ino == inode:
                    path.unlink(missing_ok=True)
            raise
    return manifest_path


def merge_reference_shards(
    shard_manifests: Sequence[str | Path],
    *,
    feature_cache_path: str | Path,
    output_prefix: str | Path,
    allow_low_probe_jackknife: bool = False,
) -> Path:
    """Merge reference shards under a cooperative, owner-only prefix lock."""
    prefix = Path(output_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(str(prefix) + ".gxe.merge.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Another GxE shard merger holds output prefix {prefix}: {lock}."
        ) from exc
    lock_stat = os.fstat(descriptor)
    os.close(descriptor)
    try:
        return _merge_reference_shards_impl(
            shard_manifests,
            feature_cache_path=feature_cache_path,
            output_prefix=prefix,
            allow_low_probe_jackknife=allow_low_probe_jackknife,
        )
    finally:
        try:
            observed = lock.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if observed.st_dev == lock_stat.st_dev and observed.st_ino == lock_stat.st_ino:
                lock.unlink(missing_ok=True)
