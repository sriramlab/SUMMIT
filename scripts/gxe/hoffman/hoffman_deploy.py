#!/usr/bin/env python3
"""Render and run private, hash-bound Hoffman GxE jobs without submitting them.

The ``render`` command creates one concrete UGE script below the configured
scratch root.  The generated script invokes one task-specific wrapper.  The
``run`` command revalidates every contract at job start, executes exactly one
operation, and writes a private receipt only after all expected outputs pass.
The separate ``record-qacct`` command is run after UGE accounting is available;
downstream jobs require its successful, hash-bound output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence


TASKS = ("stage_verify", "cache", "shard", "merge", "score", "fit")
WRAPPERS = {task: f"uge_{task}.sh" for task in TASKS}
SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_DISTRIBUTIONS = (
    "bed-reader",
    "charset-normalizer",
    "numpy",
    "packaging",
    "pandas",
    "platformdirs",
    "pooch",
    "psutil",
    "python-dateutil",
    "pytz",
    "scipy",
    "setuptools",
    "six",
    "threadpoolctl",
    "tqdm",
)
CRITICAL_CODE_FILES = frozenset(
    {
        "pyproject.toml",
        "scripts/gxe/hoffman/deployment_config.json",
        "scripts/gxe/hoffman/hoffman_deploy.py",
        "scripts/gxe/hoffman/panel_config.json",
        "scripts/gxe/hoffman/verify_staged_inputs.py",
        *(f"scripts/gxe/hoffman/{name}" for name in WRAPPERS.values()),
        "src/summit/__init__.py",
        "src/summit/cli.py",
        "src/summit/inference/gxe.py",
        "src/summit/ldscore/gwe_ldscore.py",
        "src/summit/ldscore/gxe_merge.py",
        "src/summit/ldscore/gxe_score.py",
    }
)


def _expected_code_files(root: Path) -> frozenset[str]:
    package_root = root / "src" / "summit"
    _require_directory(package_root, "frozen SUMMIT package root", private=False)
    package_files = {
        path.relative_to(root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    }
    return frozenset(CRITICAL_CODE_FILES | package_files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _distribution_fingerprint(name: str) -> dict:
    """Hash every installed file recorded by one required distribution."""
    distribution = importlib_metadata.distribution(name)
    recorded = distribution.files
    if recorded is None:
        raise RuntimeError(f"Installed distribution {name!r} has no file manifest.")
    digest = hashlib.sha256()
    count = 0
    total = 0
    for relative in sorted(recorded, key=lambda value: str(value)):
        relative_text = str(relative).replace(os.sep, "/")
        path = Path(distribution.locate_file(relative))
        try:
            status = os.lstat(path)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Installed distribution {name!r} is missing {relative_text!r}."
            ) from error
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError(
                f"Installed distribution {name!r} contains a non-regular recorded "
                f"file: {relative_text!r}."
            )
        file_sha = _sha256(path)
        digest.update(
            json.dumps(
                [relative_text, int(status.st_size), file_sha],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
        count += 1
        total += int(status.st_size)
    if count == 0:
        raise RuntimeError(
            f"Installed distribution {name!r} has an empty file manifest."
        )
    return {
        "version": distribution.version,
        "file_count": count,
        "bytes": total,
        "content_sha256": digest.hexdigest(),
    }


def _environment_fingerprint(python_path: Path) -> dict:
    if _absolute(sys.executable) != python_path:
        raise RuntimeError(
            "Deployment environment must be sealed and validated with the configured Python."
        )
    return {
        "python": _record(python_path),
        "distributions": {
            name: _distribution_fingerprint(name) for name in REQUIRED_DISTRIBUTIONS
        },
    }


def _absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(value).expanduser())))


def _is_below(path: Path, root: Path) -> bool:
    return path != root and root in path.parents


def _require_no_symlink_components(
    path: Path, label: str, *, final_may_be_missing: bool = False
) -> None:
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            if final_may_be_missing and index == len(parts) - 1:
                return
            raise FileNotFoundError(f"Missing {label}: {path}") from None
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")


def _require_directory(path: Path, label: str, *, private: bool) -> None:
    _require_no_symlink_components(path, label)
    status = os.lstat(path)
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} must be a real directory: {path}")
    if private:
        if status.st_uid != os.getuid():
            raise PermissionError(f"{label} is not owned by the current user: {path}")
        if stat.S_IMODE(status.st_mode) != 0o700:
            raise PermissionError(
                f"{label} must have mode 0700; observed {stat.S_IMODE(status.st_mode):04o}: {path}"
            )


def _require_file(path: Path, label: str, *, private: bool) -> None:
    _require_no_symlink_components(path, label)
    status = os.lstat(path)
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if private:
        if status.st_uid != os.getuid():
            raise PermissionError(f"{label} is not owned by the current user: {path}")
        if status.st_nlink != 1:
            raise PermissionError(f"{label} must have exactly one hard link: {path}")
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise PermissionError(
                f"{label} must have mode 0600; observed {stat.S_IMODE(status.st_mode):04o}: {path}"
            )


def _require_private_scratch_chain(path: Path, scratch_root: Path, label: str) -> None:
    """Require every scratch directory from the sealed root to ``path`` to be private."""
    _require_directory(scratch_root, "scratch root", private=True)
    if path != scratch_root and not _is_below(path, scratch_root):
        raise ValueError(f"{label} directory is outside scratch root: {path}")
    current = scratch_root
    relative_parts = (
        () if path == scratch_root else path.relative_to(scratch_root).parts
    )
    for part in relative_parts:
        current = current / part
        _require_directory(current, f"{label} directory component", private=True)


def _scratch_path(
    value: str | Path,
    scratch_root: Path,
    label: str,
    *,
    kind: str,
    must_exist: bool = True,
) -> Path:
    path = _absolute(value)
    if not _is_below(path, scratch_root):
        raise ValueError(f"{label} must be below scratch root {scratch_root}: {path}")
    chain_target = path if kind == "directory" else path.parent
    _require_private_scratch_chain(chain_target, scratch_root, label)
    if kind == "directory":
        if not must_exist:
            raise ValueError("Directory validation always requires an existing path.")
        _require_directory(path, label, private=True)
    elif kind == "file":
        if must_exist:
            _require_file(path, label, private=True)
        else:
            _require_no_symlink_components(path.parent, f"{label} parent")
            _require_directory(path.parent, f"{label} parent", private=True)
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing existing {label}: {path}")
    elif kind == "prefix":
        _require_directory(path.parent, f"{label} parent", private=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing {label}: {path}")
    else:
        raise RuntimeError(f"Unknown scratch path kind: {kind}")
    return path


def _read_json(path: Path, label: str, *, private: bool) -> tuple[dict, str, int]:
    _require_file(path, label, private=private)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload, _sha256_bytes(raw), len(raw)


def _verify_record(
    record: dict,
    label: str,
    *,
    scratch_root: Path | None,
    private: bool,
) -> tuple[Path, dict]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record must be an object.")
    path_value = record.get("path")
    expected_sha = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(path_value, str) or not SHA256.fullmatch(str(expected_sha)):
        raise ValueError(f"{label} record lacks a valid path/SHA256.")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ValueError(f"{label} record has an invalid byte count.")
    path = _absolute(path_value)
    if scratch_root is not None and not _is_below(path, scratch_root):
        raise ValueError(f"{label} is outside scratch root: {path}")
    if scratch_root is not None:
        _require_private_scratch_chain(path.parent, scratch_root, label)
    _require_file(path, label, private=private)
    observed_bytes = int(path.stat().st_size)
    observed_sha = _sha256(path)
    if observed_bytes != expected_bytes or observed_sha != expected_sha:
        raise ValueError(f"{label} size/SHA256 differs from its sealed record: {path}")
    return path, {"path": str(path), "bytes": observed_bytes, "sha256": observed_sha}


def _record(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _atomic_text_noreplace(text: str, target: Path, mode: int) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    _require_directory(target.parent, "output directory", private=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite: {target}") from None
        os.chmod(target, mode)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_noreplace(payload: dict, target: Path) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_text_noreplace(text, target, 0o600)


def _validate_resource(task: str, config: dict) -> dict:
    resource = config.get("resources", {}).get(task)
    if not isinstance(resource, dict):
        raise ValueError(f"Deployment config lacks resources for {task!r}.")
    slots = resource.get("slots")
    per_slot = resource.get("h_data_gib_per_slot")
    total = resource.get("total_memory_gib")
    h_rt = resource.get("h_rt")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (slots, per_slot, total)
    ):
        raise ValueError(f"Invalid slot/memory resource for {task!r}.")
    if total != slots * per_slot:
        raise ValueError(
            f"Resource memory error for {task!r}: total={total} but slots*per-slot={slots * per_slot}."
        )
    if not isinstance(h_rt, str) or re.fullmatch(r"\d{2,3}:\d{2}:\d{2}", h_rt) is None:
        raise ValueError(f"Invalid h_rt for {task!r}.")
    _, minutes, seconds = (int(value) for value in h_rt.split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid h_rt minute/second field for {task!r}.")
    if not isinstance(resource.get("highp"), bool):
        raise ValueError(f"Invalid highp flag for {task!r}.")
    return resource


def _load_config(path_value: str, expected_sha: str | None) -> tuple[Path, dict, str]:
    path = _absolute(path_value)
    payload, observed_sha, _ = _read_json(path, "deployment config", private=False)
    if expected_sha is not None and observed_sha != expected_sha:
        raise ValueError(
            "Deployment config SHA256 differs from the rendered job contract."
        )
    if (
        payload.get("kind") != "summit.gxe.hoffman_deployment"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Unsupported Hoffman deployment config.")
    scratch_root = _absolute(payload.get("scratch_root", ""))
    _require_directory(scratch_root, "scratch root", private=True)
    for task in TASKS:
        _validate_resource(task, payload)
    _validate_resource("shard_benchmark", payload)
    return path, payload, observed_sha


def _load_spec(
    path_value: str,
    expected_sha: str | None,
    scratch_root: Path,
) -> tuple[Path, dict, str]:
    path = _scratch_path(path_value, scratch_root, "job spec", kind="file")
    payload, observed_sha, _ = _read_json(path, "job spec", private=True)
    if expected_sha is not None and observed_sha != expected_sha:
        raise ValueError("Job spec SHA256 differs from the rendered job contract.")
    if (
        payload.get("kind") != "summit.gxe.hoffman_job"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Unsupported Hoffman job spec.")
    if payload.get("task") not in TASKS:
        raise ValueError(f"Unsupported Hoffman task: {payload.get('task')!r}")
    return path, payload, observed_sha


def _check_cache_merge_compatibility_sources(
    frozen_root: Path, estimator: dict
) -> None:
    writer = frozen_root / "src" / "summit" / "ldscore" / "gwe_ldscore.py"
    merger = frozen_root / "src" / "summit" / "ldscore" / "gxe_merge.py"
    _require_file(writer, "frozen cache writer source", private=False)
    _require_file(merger, "frozen shard merger source", private=False)
    writer_text = writer.read_text(encoding="utf-8")
    merger_text = merger.read_text(encoding="utf-8")
    expected_cache = int(estimator.get("feature_cache_schema_version", -1))
    expected_shard = int(estimator.get("reference_shard_schema_version", -1))
    expected_reference = int(estimator.get("reference_schema_version", -1))
    match = re.search(r"_FEATURE_CACHE_SCHEMA_VERSION\s*=\s*(\d+)", writer_text)
    if match is None or int(match.group(1)) != expected_cache:
        raise ValueError(
            "Frozen cache writer does not declare the required cache schema version."
        )
    if "_validate_feature_cache_semantics" not in merger_text:
        raise ValueError(
            "Frozen shard merger lacks the required feature-cache semantic validation; deployment is blocked."
        )
    if "_FEATURE_CACHE_SCHEMA_VERSION" not in merger_text:
        raise ValueError(
            "Frozen shard merger does not share the cache writer's schema-version contract; deployment is blocked."
        )
    if f'shard.get("schema_version") != {expected_shard}' not in merger_text:
        raise ValueError(
            "Frozen shard merger does not require the configured shard schema."
        )
    if f'"schema_version": {expected_reference}' not in merger_text:
        raise ValueError(
            "Frozen shard merger does not emit the configured reference schema."
        )


def _validate_code_manifest(
    manifest_path: Path,
    frozen_root: Path,
    python_path: Path,
    native_build_value: Any,
) -> dict:
    payload, _, _ = _read_json(manifest_path, "frozen code manifest", private=False)
    if (
        payload.get("kind") != "summit.gxe.frozen_code_manifest"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Unsupported frozen code manifest.")
    if _absolute(payload.get("root", "")) != frozen_root:
        raise ValueError("Frozen code manifest names a different snapshot root.")
    if _absolute(payload.get("python_executable", "")) != python_path:
        raise ValueError("Frozen code manifest names a different Python executable.")
    environment = payload.get("environment")
    observed_environment = _environment_fingerprint(python_path)
    if environment != observed_environment:
        raise ValueError(
            "Configured Python or required distribution files differ from the "
            "frozen environment manifest."
        )

    files = payload.get("files")
    expected_files = _expected_code_files(frozen_root)
    observed_files = set(files) if isinstance(files, dict) else set()
    if not isinstance(files, dict) or observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        extra = sorted(observed_files - expected_files)
        raise ValueError(
            f"Frozen code manifest file set is invalid; missing={missing}, extra={extra}."
        )
    verified: dict[str, dict] = {}
    for relative_text in sorted(expected_files):
        relative = Path(relative_text)
        path = _absolute(frozen_root / relative)
        if not _is_below(path, frozen_root):
            raise ValueError(f"Unsafe frozen code-manifest path: {relative_text}")
        record = files[relative_text]
        expected = {
            "path": str(path),
            "bytes": record.get("bytes") if isinstance(record, dict) else None,
            "sha256": record.get("sha256") if isinstance(record, dict) else None,
        }
        _, verified[relative_text] = _verify_record(
            expected,
            f"frozen code file {relative_text!r}",
            scratch_root=None,
            private=False,
        )

    native_build = _absolute(native_build_value or "")
    if not _is_below(native_build, frozen_root):
        raise ValueError(
            "Frozen native-build directory must be inside the snapshot root."
        )
    _require_directory(native_build, "frozen native-build directory", private=False)
    native_record = payload.get("native_module")
    if not isinstance(native_record, dict):
        raise ValueError("Frozen code manifest lacks a native-module record.")
    native_path_value = native_record.get("path")
    if not isinstance(native_path_value, str) or not native_path_value:
        raise ValueError("Frozen native-module record lacks a path.")
    native_relative = Path(native_path_value)
    if native_relative.is_absolute() or ".." in native_relative.parts:
        raise ValueError(
            "Frozen native-module path must be a safe snapshot-relative path."
        )
    native_path = _absolute(frozen_root / native_relative)
    if not _is_below(native_path, native_build):
        raise ValueError(
            "Frozen native module must be inside the configured native-build directory."
        )
    _, verified_native = _verify_record(
        {
            "path": str(native_path),
            "bytes": native_record.get("bytes"),
            "sha256": native_record.get("sha256"),
        },
        "frozen native module",
        scratch_root=None,
        private=False,
    )
    if not native_path.name.startswith("gwldcore") or native_path.suffix not in {
        ".so",
        ".pyd",
        ".dylib",
    }:
        raise ValueError("Frozen native-module record is not a gwldcore extension.")
    return {
        "payload": payload,
        "files": verified,
        "native": verified_native,
        "native_build": native_build,
        "environment": observed_environment,
    }


def _validate_frozen(
    spec: dict,
    config: dict,
    config_path: Path,
    *,
    expected_task: str,
) -> dict:
    frozen = spec.get("frozen")
    if not isinstance(frozen, dict):
        raise ValueError("Job spec lacks a frozen-code contract.")
    root = _absolute(frozen.get("root", ""))
    allowed = _absolute(config.get("allowed_frozen_root", ""))
    _require_directory(allowed, "allowed frozen-code root", private=False)
    if not _is_below(root, allowed):
        raise ValueError(f"Frozen code root must be below {allowed}: {root}")
    _require_directory(root, "frozen code root", private=False)

    python_path = _absolute(frozen.get("python", ""))
    if str(python_path) != str(_absolute(config.get("python_executable", ""))):
        raise ValueError("Frozen Python executable differs from deployment config.")
    if not python_path.exists() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(
            f"Frozen Python executable is unavailable: {python_path}"
        )

    deploy_path = root / "scripts" / "gxe" / "hoffman" / "hoffman_deploy.py"
    wrapper_path = root / "scripts" / "gxe" / "hoffman" / WRAPPERS[expected_task]
    cli_path = root / "src" / "summit" / "cli.py"
    panel_path = root / "scripts" / "gxe" / "hoffman" / "panel_config.json"
    for path, label, key in (
        (deploy_path, "frozen deployment runner", "deploy_sha256"),
        (wrapper_path, "frozen task wrapper", "wrapper_sha256"),
        (cli_path, "frozen SUMMIT CLI", "cli_sha256"),
    ):
        _require_file(path, label, private=False)
        expected = frozen.get(key)
        if (
            not isinstance(expected, str)
            or not SHA256.fullmatch(expected)
            or _sha256(path) != expected
        ):
            raise ValueError(f"{label} SHA256 differs from the frozen job contract.")
    _require_file(panel_path, "frozen panel config", private=False)
    panel_sha = _sha256(panel_path)
    if panel_sha != config.get("panel_config_sha256"):
        raise ValueError("Frozen panel config SHA256 differs from deployment config.")
    panel_payload = json.loads(panel_path.read_text(encoding="utf-8"))
    production = panel_payload.get("production_estimator", {})
    estimator = config.get("estimator", {})
    estimator_keys = {
        "annotation": "annotation",
        "annotation_contract": "annotation_contract",
        "kernel_mode": "kernel_mode",
        "genotype_scale": "genotype_scale",
        "jackknife_blocks": "jackknife_blocks",
        "probes_per_shard": "probes_per_shard",
        "production_probes": "num_probes",
        "random_distribution": "random_distribution",
        "seed": "seed_family",
        "dtype": "dtype",
        "step_size": "step_size",
    }
    for key, panel_key in estimator_keys.items():
        if production.get(panel_key) != estimator.get(key):
            raise ValueError(
                f"Frozen panel/deployment estimator contract differs for {key!r}."
            )
    if production.get("probe_shards") != estimator.get(
        "production_probes"
    ) // estimator.get("probes_per_shard"):
        raise ValueError("Frozen panel/deployment shard-count contract differs.")

    manifest_path, manifest_record = _verify_record(
        frozen.get("code_manifest"),
        "frozen code manifest",
        scratch_root=None,
        private=False,
    )
    if not _is_below(manifest_path, root):
        raise ValueError("Frozen code manifest must be inside the frozen code root.")
    code_manifest = _validate_code_manifest(
        manifest_path,
        root,
        python_path,
        frozen.get("native_build_dir"),
    )
    _check_cache_merge_compatibility_sources(root, estimator)
    if config_path != root / "scripts" / "gxe" / "hoffman" / "deployment_config.json":
        raise ValueError(
            "Jobs must use the deployment config inside the frozen snapshot."
        )
    return {
        "root": root,
        "python": python_path,
        "deploy": deploy_path,
        "wrapper": wrapper_path,
        "cli": cli_path,
        "panel": panel_path,
        "code_manifest": manifest_record,
        "code_manifest_payload": code_manifest,
        "native_build_dir": code_manifest["native_build"],
        "panel_payload": panel_payload,
    }


def _validate_qacct_dependencies(
    spec: dict,
    scratch_root: Path,
    task: str,
    *,
    expected_count: int | None = None,
) -> list[dict]:
    records = spec.get("qacct_dependencies", [])
    if not isinstance(records, list):
        raise ValueError("qacct_dependencies must be a list.")
    required_task = {
        "stage_verify": None,
        "cache": "stage_verify",
        "shard": "cache",
        "merge": "shard",
        "score": "merge",
        "fit": "score",
    }[task]
    if required_task is None and records:
        raise ValueError(
            "Stage verification must not declare upstream qacct dependencies."
        )
    if required_task is not None and not records:
        raise ValueError(f"Task {task!r} requires completed qacct provenance.")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Task {task!r} requires exactly {expected_count} qacct record(s); got {len(records)}."
        )

    payloads: list[dict] = []
    seen_jobs: set[int] = set()
    for index, record in enumerate(records):
        path, _ = _verify_record(
            record,
            f"qacct dependency {index}",
            scratch_root=scratch_root,
            private=True,
        )
        payload, observed_sha, _ = _read_json(
            path, f"qacct dependency {index}", private=True
        )
        if (
            payload.get("kind") != "summit.gxe.hoffman_qacct"
            or payload.get("schema_version") != 1
        ):
            raise ValueError(f"Unsupported qacct dependency: {path}")
        if payload.get("task") != required_task:
            raise ValueError(
                f"qacct dependency task {payload.get('task')!r} does not satisfy {task!r}."
            )
        if (
            int(payload.get("failed", -1)) != 0
            or int(payload.get("exit_status", -1)) != 0
        ):
            raise ValueError(f"Upstream qacct dependency was not successful: {path}")
        job_id = payload.get("job_id")
        if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
            raise ValueError(f"qacct dependency has an invalid job ID: {path}")
        if job_id in seen_jobs:
            raise ValueError("Duplicate upstream qacct job IDs are not permitted.")
        seen_jobs.add(job_id)
        receipt = payload.get("receipt")
        receipt_path, receipt_record = _verify_record(
            receipt,
            f"qacct dependency receipt {index}",
            scratch_root=scratch_root,
            private=True,
        )
        if payload.get("receipt_sha256") != receipt_record["sha256"]:
            raise ValueError(
                f"qacct dependency does not bind its process receipt: {path}"
            )
        receipt_payload, _, _ = _read_json(
            receipt_path, f"qacct dependency receipt {index}", private=True
        )
        if (
            receipt_payload.get("kind") != "summit.gxe.hoffman_process_receipt"
            or receipt_payload.get("schema_version") != 1
            or receipt_payload.get("job_id") != job_id
            or receipt_payload.get("task") != required_task
            or receipt_payload.get("qacct_pending") is not True
        ):
            raise ValueError(f"qacct dependency and process receipt disagree: {path}")
        for key in (
            "outputs",
            "task_details",
            "resource_profile",
            "resource",
            "deployment_config",
            "job_spec",
            "job_script",
            "attempt_lock",
            "attempt",
            "code_manifest",
        ):
            if payload.get(key) != receipt_payload.get(key):
                raise ValueError(
                    f"qacct dependency does not reproduce receipt field {key!r}: {path}"
                )
        if int(payload.get("slots", -1)) != int(
            receipt_payload.get("resource", {}).get("slots", -2)
        ):
            raise ValueError(
                f"qacct dependency slots differ from the sealed request: {path}"
            )
        raw_path, raw_record = _verify_record(
            payload.get("qacct_raw"),
            f"qacct dependency raw accounting {index}",
            scratch_root=scratch_root,
            private=True,
        )
        if raw_path.parent != path.parent or raw_record["sha256"] != payload.get(
            "qacct_stdout_sha256"
        ):
            raise ValueError(
                f"qacct dependency does not bind its raw accounting: {path}"
            )
        logs = payload.get("uge_logs")
        if not isinstance(logs, dict) or set(logs) != {"stdout", "stderr"}:
            raise ValueError(f"qacct dependency lacks exact UGE log records: {path}")
        for stream, log_record in logs.items():
            log_path, _ = _verify_record(
                log_record,
                f"qacct dependency {stream} log {index}",
                scratch_root=scratch_root,
                private=True,
            )
            if log_path.parent != path.parent:
                raise ValueError(
                    f"qacct dependency {stream} log is outside its job root."
                )
        _verify_record(
            payload.get("job_spec"),
            f"qacct dependency job spec {index}",
            scratch_root=scratch_root,
            private=True,
        )
        _verify_record(
            payload.get("job_script"),
            f"qacct dependency UGE script {index}",
            scratch_root=scratch_root,
            private=False,
        )
        _verify_record(
            payload.get("attempt_lock"),
            f"qacct dependency attempt lock {index}",
            scratch_root=scratch_root,
            private=True,
        )
        _verify_record(
            payload.get("attempt"),
            f"qacct dependency attempt marker {index}",
            scratch_root=scratch_root,
            private=True,
        )
        _verify_record(
            payload.get("deployment_config"),
            f"qacct dependency deployment config {index}",
            scratch_root=None,
            private=False,
        )
        _verify_record(
            payload.get("code_manifest"),
            f"qacct dependency code manifest {index}",
            scratch_root=None,
            private=False,
        )
        payloads.append({**payload, "record_sha256": observed_sha})
    return payloads


def _validate_common(
    spec_path: Path,
    spec: dict,
    config_path: Path,
    config: dict,
    *,
    expected_task: str,
    rendering: bool,
) -> dict:
    if spec.get("task") != expected_task:
        raise ValueError(
            f"Job spec task {spec.get('task')!r} does not match wrapper task {expected_task!r}."
        )
    job_name = spec.get("job_name")
    if not isinstance(job_name, str) or SAFE_NAME.fullmatch(job_name) is None:
        raise ValueError("job_name must be a safe UGE name of at most 128 characters.")
    scratch_root = _absolute(config["scratch_root"])
    job_root = _scratch_path(
        spec.get("job_root", ""), scratch_root, "job root", kind="directory"
    )
    if any(character.isspace() or ord(character) < 32 for character in str(job_root)):
        raise ValueError(
            "UGE job_root must not contain whitespace or control characters."
        )
    if _is_below(spec_path, job_root):
        raise ValueError("Job spec must be outside the fresh job root.")
    if rendering:
        entries = list(job_root.iterdir())
        if entries:
            raise FileExistsError(f"Fresh job root is not empty: {job_root}")
    else:
        expected_entries = {
            "job.sh",
            "stdout.log",
            "stderr.log",
            "tmp",
            "attempt.lock",
        }
        observed_entries = {path.name for path in job_root.iterdir()}
        if observed_entries != expected_entries:
            raise ValueError(
                f"Rendered job root has unexpected/missing entries: {sorted(observed_entries)}"
            )
        _require_file(job_root / "job.sh", "rendered UGE script", private=False)
        if stat.S_IMODE((job_root / "job.sh").stat().st_mode) != 0o700:
            raise PermissionError("Rendered UGE script must have mode 0700.")
        _require_file(job_root / "stdout.log", "UGE stdout log", private=True)
        _require_file(job_root / "stderr.log", "UGE stderr log", private=True)
        _require_file(job_root / "attempt.lock", "UGE attempt lock", private=True)
        _require_directory(job_root / "tmp", "UGE temporary directory", private=True)
        if any((job_root / "tmp").iterdir()):
            raise ValueError("UGE temporary directory is not empty at first job start.")
    frozen = _validate_frozen(spec, config, config_path, expected_task=expected_task)
    resource_profile = expected_task
    if (
        expected_task == "shard"
        and isinstance(spec.get("task_args"), dict)
        and spec["task_args"].get("role") == "benchmark"
    ):
        resource_profile = "shard_benchmark"
    resource = _validate_resource(resource_profile, config)
    return {
        "scratch_root": scratch_root,
        "job_root": job_root,
        "job_name": job_name,
        "frozen": frozen,
        "resource": resource,
        "resource_profile": resource_profile,
        "receipt": job_root / "process_receipt.json",
        "qacct": job_root / "completed_qacct.json",
        "attempt_lock": job_root / "attempt.lock",
        "attempt": job_root / "attempt.json",
        "invocation": job_root / "invocation.json",
        "artifacts": job_root / "artifacts",
        "tmp": job_root / "tmp",
    }


def _load_group(
    record: dict,
    scratch_root: Path,
    panel: dict,
) -> tuple[Path, dict, dict[str, Path]]:
    path, _ = _verify_record(
        record, "group manifest", scratch_root=scratch_root, private=True
    )
    payload, _, _ = _read_json(path, "group manifest", private=True)
    if (
        payload.get("kind") != "summit.gxe.common_cohort_group"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported common-cohort group manifest: {path}")
    label = payload.get("label")
    group_config = panel.get("groups", {}).get(label)
    if not isinstance(group_config, dict):
        raise ValueError(f"Group {label!r} is absent from frozen panel config.")
    if payload.get("environment", {}).get("source_column") != group_config.get(
        "environment"
    ):
        raise ValueError(f"Group {label!r} environment differs from panel config.")
    if list(payload.get("phenotype_labels", [])) != list(
        group_config.get("phenotypes", [])
    ):
        raise ValueError(f"Group {label!r} phenotype order differs from panel config.")
    outputs: dict[str, Path] = {}
    for key in ("environment", "covariates", "phenotypes"):
        output_record = payload.get("outputs", {}).get(key)
        if not isinstance(output_record, dict):
            raise ValueError(f"Group {label!r} lacks output {key!r}.")
        relative = Path(str(output_record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe group output path: {relative}")
        output_path = _absolute(path.parent / relative)
        expected_record = {
            "path": str(output_path),
            "bytes": output_record.get("bytes"),
            "sha256": output_record.get("sha256"),
        }
        verified, _ = _verify_record(
            expected_record,
            f"group {label!r} {key}",
            scratch_root=scratch_root,
            private=True,
        )
        outputs[key] = verified
    return path, payload, outputs


def _validate_genotype_prefix(prefix_value: str, scratch_root: Path) -> Path:
    prefix = _absolute(prefix_value)
    if not _is_below(prefix, scratch_root):
        raise ValueError(f"Genotype prefix is outside scratch root: {prefix}")
    _require_private_scratch_chain(prefix.parent, scratch_root, "staged genotype")
    for extension in (".bed", ".bim", ".fam"):
        _require_file(
            Path(str(prefix) + extension), f"staged genotype {extension}", private=True
        )
    return prefix


def _validate_stage_report(
    record: dict,
    scratch_root: Path,
    panel_sha: str,
    geno_prefix: Path,
    label: str,
    group_manifest_sha256: str,
) -> dict:
    path, _ = _verify_record(
        record, "stage verification report", scratch_root=scratch_root, private=True
    )
    payload, _, _ = _read_json(path, "stage verification report", private=True)
    if (
        payload.get("kind") != "summit.gxe.staged_input_verification"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported stage verification report: {path}")
    if payload.get("config", {}).get("sha256") != panel_sha:
        raise ValueError("Stage report panel-config SHA256 differs from frozen config.")
    if _absolute(payload.get("staged_genotype_prefix", "")) != geno_prefix:
        raise ValueError("Stage report genotype prefix differs from this job.")
    group_entries = payload.get("groups", [])
    labels = [entry.get("label") for entry in group_entries]
    if label not in labels:
        raise ValueError(f"Stage report does not include group {label!r}.")
    matching = [entry for entry in group_entries if entry.get("label") == label]
    if (
        len(matching) != 1
        or matching[0].get("manifest_sha256") != group_manifest_sha256
    ):
        raise ValueError(
            f"Stage report does not bind the exact group manifest for {label!r}."
        )
    return payload


def _validate_cache_stage_genotype(cache: dict, stage_report: dict) -> None:
    staged_hashes = stage_report.get("hashes", {}).get("staged", {})
    cache_files = cache.get("metadata", {}).get("genotype_files", {})
    for extension in ("bed", "bim", "fam"):
        if cache_files.get(f".{extension}", {}).get("sha256") != staged_hashes.get(
            extension
        ):
            raise ValueError(
                f"Feature cache .{extension} digest differs from completed stage verification."
            )


def _validate_reference_cache_identity(reference: dict, cache: dict) -> None:
    metadata = cache.get("metadata", {})
    exact_fields = (
        "analysis_fingerprint",
        "variant_digest",
        "n_samples",
        "fixed_effect_rank_excluding_intercept",
        "residual_rank",
        "kernel_mode",
        "genotype_scale",
        "environment",
        "environment_transform",
        "covariates",
        "genotype_files",
        "feature_diagnostics",
    )
    mismatched = [
        field for field in exact_fields if reference.get(field) != metadata.get(field)
    ]
    if reference.get("annotation_names") != metadata.get("annotation_names"):
        mismatched.append("annotation_names")
    if reference.get("annotation_masses") != metadata.get("annotation_masses"):
        mismatched.append("annotation_masses")
    if mismatched:
        raise ValueError(
            "Reference/shard identity differs from its exact feature cache for fields "
            f"{sorted(set(mismatched))}."
        )


def _verify_staged_sources(
    source_manifest_record: dict,
    staged_source_dir_value: str,
    scratch_root: Path,
    panel_sha: str,
) -> tuple[Path, dict, dict[tuple[str, str], dict]]:
    manifest_path, _ = _verify_record(
        source_manifest_record,
        "local source manifest",
        scratch_root=scratch_root,
        private=True,
    )
    payload, _, _ = _read_json(manifest_path, "local source manifest", private=True)
    if (
        payload.get("kind") != "summit.gxe.local_source_manifest"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported local source manifest: {manifest_path}")
    if payload.get("config", {}).get("sha256") != panel_sha:
        raise ValueError(
            "Local source manifest was generated from a different panel config."
        )
    source_dir = _scratch_path(
        staged_source_dir_value,
        scratch_root,
        "staged phenotype/covariate source directory",
        kind="directory",
    )
    records: dict[tuple[str, str], dict] = {}
    for source_kind in ("phenotypes", "covariates"):
        sources = payload.get("sources", {}).get(source_kind)
        if not isinstance(sources, dict):
            raise ValueError(f"Local source manifest lacks {source_kind} records.")
        for name, source in sources.items():
            relative = Path(str(source.get("relative_path", "")))
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError(
                    f"Unsafe staged source relative path for {name!r}: {relative}"
                )
            target = _absolute(source_dir / relative)
            expected = {
                "path": str(target),
                "bytes": source.get("bytes"),
                "sha256": source.get("sha256"),
            }
            _, verified = _verify_record(
                expected,
                f"staged {source_kind} source {name!r}",
                scratch_root=scratch_root,
                private=True,
            )
            records[(source_kind, str(name))] = {
                **verified,
                "relative_path": relative.as_posix(),
            }
    counts = payload.get("counts", {})
    if (
        counts.get("unique_phenotype_files") != 22
        or counts.get("configured_covariate_files") != 4
    ):
        raise ValueError(
            "Local source manifest does not contain the adjudicated 22/4 inventory."
        )
    return manifest_path, payload, records


def _ndarray_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x1f")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_cache_file(path: Path, config: dict, group: dict | None = None) -> dict:
    import numpy as np

    _require_file(path, "feature cache", private=True)
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
        if "metadata_json" not in bundle.files:
            raise ValueError("Feature cache lacks metadata_json.")
        metadata = json.loads(str(bundle["metadata_json"].item()))
        arrays = {
            name: np.asarray(bundle[name]).copy()
            for name in bundle.files
            if name != "metadata_json"
        }
    estimator = config["estimator"]
    if metadata.get("kind") != "summit.gxe.feature_cache":
        raise ValueError("Unsupported feature-cache kind.")
    if metadata.get("schema_version") != estimator["feature_cache_schema_version"]:
        raise ValueError("Feature-cache schema differs from deployment contract.")
    declared = metadata.get("array_sha256")
    if not isinstance(declared, dict) or set(declared) != set(arrays):
        raise ValueError("Feature cache does not bind exactly every array.")
    for name, value in arrays.items():
        if _ndarray_sha256(value) != declared[name]:
            raise ValueError(f"Feature-cache array {name!r} failed SHA256 validation.")
    required = {
        "variant_chr",
        "variant_snp",
        "variant_bp",
        "variant_a1",
        "variant_a2",
        "annotations",
        "jackknife_ids",
        "scale_x",
        "scale_w",
        "norm_x",
        "norm_w",
        "diag_nxe_x",
        "diag_nxe_w",
        "corr_xw",
        "nxe_qdq",
        "nxe_trace_terms",
    }
    if not required.issubset(arrays):
        raise ValueError(
            f"Feature cache lacks required arrays: {sorted(required - set(arrays))}"
        )
    m = int(estimator["annotation_mass"])
    if metadata.get("n_variants") != m:
        raise ValueError(
            "Feature-cache variant count differs from all-variant contract."
        )
    if metadata.get("kernel_mode") != estimator["kernel_mode"]:
        raise ValueError("Feature-cache kernel mode differs from production contract.")
    if metadata.get("genotype_scale") != estimator["genotype_scale"]:
        raise ValueError(
            "Feature-cache genotype scale differs from production contract."
        )
    if metadata.get("annotation_names") != [estimator["annotation_name"]]:
        raise ValueError("Feature cache is not the required one-bin annotation.")
    masses = np.asarray(metadata.get("annotation_masses"), dtype=np.float64)
    annotations = np.asarray(arrays["annotations"], dtype=np.float64)
    if masses.shape != (1,) or masses[0] != float(m):
        raise ValueError("Feature-cache annotation mass differs from M.")
    if annotations.shape != (m, 1) or not np.array_equal(annotations, np.ones((m, 1))):
        raise ValueError(
            "Feature-cache annotation is not the canonical all-ones vector."
        )
    jackknife = np.asarray(arrays["jackknife_ids"], dtype=np.int64)
    labels = metadata.get("jackknife_labels")
    j = int(estimator["jackknife_blocks"])
    if jackknife.shape != (m,) or sorted(np.unique(jackknife).tolist()) != list(
        range(j)
    ):
        raise ValueError(
            "Feature-cache jackknife IDs are not exactly J contiguous nonempty blocks."
        )
    if not isinstance(labels, list) or len(labels) != j:
        raise ValueError("Feature-cache jackknife labels differ from J.")
    for name in (
        "scale_x",
        "scale_w",
        "norm_x",
        "norm_w",
        "diag_nxe_x",
        "diag_nxe_w",
        "corr_xw",
    ):
        value = np.asarray(arrays[name], dtype=np.float64)
        if value.shape != (m,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Feature-cache array {name!r} has invalid shape/value.")
    if np.any(np.asarray(arrays["scale_x"]) <= 0) or np.any(
        np.asarray(arrays["scale_w"]) <= 0
    ):
        raise ValueError("Feature-cache projected-feature scales must be positive.")
    if group is not None:
        for metadata_key, group_key in (
            ("n_samples", "n_selected_samples"),
            (
                "fixed_effect_rank_excluding_intercept",
                "fixed_effect_rank_excluding_intercept",
            ),
            ("residual_rank", "residual_rank"),
        ):
            if int(metadata.get(metadata_key, -1)) != int(group.get(group_key, -2)):
                raise ValueError(f"Feature cache and group disagree on {metadata_key}.")
        if metadata.get("environment") != "ENV":
            raise ValueError(
                "Feature cache does not use the sealed group environment column ENV."
            )
    return {"metadata": metadata, "sha256": _sha256_bytes(raw), "arrays": arrays}


def _resolve_manifest_artifacts(
    manifest_path: Path,
    payload: dict,
    scratch_root: Path,
    *,
    required: set[str],
) -> dict[str, dict]:
    files = payload.get("files")
    hashes = payload.get("artifact_sha256")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ValueError(f"Manifest lacks bound artifact maps: {manifest_path}")
    if set(files) != required or set(hashes) != required:
        raise ValueError(
            "Manifest artifact set differs from its exact contract; "
            f"files={sorted(files)}, hashes={sorted(hashes)}, expected={sorted(required)}."
        )
    records: dict[str, dict] = {}
    for key in sorted(files):
        relative = Path(str(files[key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest artifact {key!r} is not a safe relative path.")
        path = _absolute(manifest_path.parent / relative)
        expected = {
            "path": str(path),
            "bytes": int(path.stat().st_size) if path.exists() else -1,
            "sha256": hashes.get(key),
        }
        _, record = _verify_record(
            expected,
            f"manifest artifact {key!r}",
            scratch_root=scratch_root,
            private=True,
        )
        records[key] = record
    return records


def _validate_shard_record(
    record: dict,
    scratch_root: Path,
    config: dict,
    cache_sha: str,
    cache: dict | None = None,
) -> tuple[Path, dict, dict[str, dict]]:
    path, _ = _verify_record(
        record, "reference shard", scratch_root=scratch_root, private=True
    )
    payload, _, _ = _read_json(path, "reference shard", private=True)
    estimator = config["estimator"]
    if payload.get("kind") != "summit.gxe.reference_shard":
        raise ValueError(f"Unsupported reference-shard kind: {path}")
    if payload.get("schema_version") != estimator["reference_shard_schema_version"]:
        raise ValueError("Reference-shard schema differs from deployment contract.")
    if payload.get("feature_cache", {}).get("sha256") != cache_sha:
        raise ValueError(
            "Reference shard was generated from a different feature cache."
        )
    if cache is not None:
        metadata = cache.get("metadata", {})
        cache_fields = (
            "analysis_fingerprint",
            "variant_digest",
            "annotation_digest",
            "jackknife_digest",
            "kernel_mode",
            "genotype_scale",
            "annotation_names",
        )
        mismatched = [
            key for key in cache_fields if payload.get(key) != metadata.get(key)
        ]
        if mismatched:
            raise ValueError(
                "Reference shard differs from its feature cache for fields "
                f"{mismatched}."
            )
    if (
        payload.get("kernel_mode") != estimator["kernel_mode"]
        or payload.get("genotype_scale") != estimator["genotype_scale"]
    ):
        raise ValueError(
            "Reference shard estimator mode differs from deployment contract."
        )
    if payload.get("ld_scale") != "cross_product_over_rank_squared" or payload.get(
        "annotation_names"
    ) != [estimator["annotation_name"]]:
        raise ValueError("Reference shard differs from the one-bin raw-panel contract.")
    randomization = payload.get("randomization", {})
    expected_random = {
        "distribution": estimator["random_distribution"],
        "algorithm": "philox_per_probe_block_v1",
        "seed": estimator["seed"],
        "dtype": estimator["dtype"],
        "step_size": estimator["step_size"],
        "num_vectors": estimator["probes_per_shard"],
    }
    for key, value in expected_random.items():
        if randomization.get(key) != value:
            raise ValueError(f"Reference shard randomization differs for {key!r}.")
    start = randomization.get("probe_offset")
    stop = randomization.get("probe_stop")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or start
        not in range(0, estimator["production_probes"], estimator["probes_per_shard"])
        or stop != start + estimator["probes_per_shard"]
    ):
        raise ValueError("Reference shard has an invalid probe interval.")
    if payload.get("jackknife", {}).get("num_blocks") != estimator["jackknife_blocks"]:
        raise ValueError("Reference shard jackknife block count differs from J.")
    required = {"xx", "xw", "wx", "ww", "jackknife", "identity"}
    artifacts = _resolve_manifest_artifacts(
        path, payload, scratch_root, required=required
    )
    identity_payload, _, _ = _read_json(
        Path(artifacts["identity"]["path"]), "reference-shard identity", private=True
    )
    if (
        identity_payload.get("kind") != "summit.gxe.reference_shard_identity"
        or identity_payload.get("schema_version") != 1
    ):
        raise ValueError("Reference-shard identity sidecar is invalid.")
    if identity_payload.get("feature_cache_sha256") != cache_sha:
        raise ValueError("Reference-shard identity binds a different feature cache.")
    identity_randomization = identity_payload.get("randomization", {})
    for key in (
        "distribution",
        "algorithm",
        "seed",
        "dtype",
        "step_size",
        "num_vectors",
        "probe_offset",
        "probe_stop",
    ):
        if identity_randomization.get(key) != randomization.get(key):
            raise ValueError(f"Reference-shard identity differs for {key!r}.")
    outer_hashes = payload.get("artifact_sha256", {})
    identity_hashes = identity_payload.get("artifact_sha256", {})
    if identity_hashes != {
        key: value for key, value in outer_hashes.items() if key != "identity"
    }:
        raise ValueError(
            "Reference-shard identity does not bind its exact contribution artifacts."
        )
    return path, payload, artifacts


def _validate_reference_record(
    record: dict,
    scratch_root: Path,
    config: dict,
    cache_sha: str,
    *,
    expected_probes: int,
) -> tuple[Path, dict, dict[str, dict]]:
    path, _ = _verify_record(
        record, "merged reference", scratch_root=scratch_root, private=True
    )
    payload, _, _ = _read_json(path, "merged reference", private=True)
    estimator = config["estimator"]
    if (
        payload.get("kind") != "summit.gxe.reference"
        or payload.get("schema_version") != estimator["reference_schema_version"]
    ):
        raise ValueError(
            "Merged reference kind/schema differs from deployment contract."
        )
    if payload.get("feature_cache", {}).get("sha256") != cache_sha:
        raise ValueError("Merged reference binds a different feature cache.")
    if (
        payload.get("kernel_mode") != estimator["kernel_mode"]
        or payload.get("genotype_scale") != estimator["genotype_scale"]
    ):
        raise ValueError(
            "Merged reference estimator mode differs from deployment contract."
        )
    if (
        payload.get("null_corrected") is not False
        or payload.get("ld_scale") != "cross_product_over_rank_squared"
    ):
        raise ValueError("Merged reference does not retain raw realized-sample panels.")
    if payload.get("annotation_names") != [estimator["annotation_name"]] or payload.get(
        "annotation_masses"
    ) != [estimator["annotation_mass"]]:
        raise ValueError("Merged reference differs from one-bin annotation contract.")
    randomization = payload.get("randomization", {})
    if randomization.get("num_vectors") != expected_probes:
        raise ValueError("Merged reference probe count differs from checkpoint.")
    expected_ranges = [
        [start, start + estimator["probes_per_shard"]]
        for start in range(0, expected_probes, estimator["probes_per_shard"])
    ]
    expected_randomization = {
        "distribution": estimator["random_distribution"],
        "algorithm": "philox_per_probe_block_v1",
        "seed": estimator["seed"],
        "dtype": estimator["dtype"],
        "step_size": estimator["step_size"],
        "probe_offset": 0,
        "probe_stop": expected_probes,
        "probe_ranges": expected_ranges,
    }
    for key, value in expected_randomization.items():
        if randomization.get(key) != value:
            raise ValueError(f"Merged reference randomization differs for {key!r}.")
    override = bool(randomization.get("low_probe_jackknife_override", False))
    if override != (expected_probes < estimator["production_probes"]):
        raise ValueError("Merged reference low-probe override policy is invalid.")
    required = {"xx", "xw", "wx", "ww", "diagonal", "jackknife"}
    artifacts = _resolve_manifest_artifacts(
        path, payload, scratch_root, required=required
    )
    if payload.get("jackknife", {}).get("num_blocks") != estimator["jackknife_blocks"]:
        raise ValueError("Merged reference jackknife block count differs from J.")
    return path, payload, artifacts


def _id_digest(frame: Any, mask: Any | None = None) -> str:
    import numpy as np

    selected = frame if mask is None else frame.loc[np.asarray(mask, dtype=bool)]
    digest = hashlib.sha256()
    for fid, iid in selected[["FID", "IID"]].itertuples(index=False, name=None):
        digest.update(str(fid).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(iid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _deep_validate_group(
    fam_path: Path,
    group: dict,
    outputs: dict[str, Path],
    panel: dict,
    dataset: str,
) -> dict:
    import numpy as np
    import pandas as pd

    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype=str,
    )
    if fam.empty or fam.duplicated(["FID", "IID"]).any():
        raise ValueError("Staged FAM is empty or has duplicate IDs.")
    tables = {
        key: pd.read_csv(path, sep=r"\s+", dtype={"FID": str, "IID": str})
        for key, path in outputs.items()
    }
    for key, table in tables.items():
        if list(table.columns[:2]) != ["FID", "IID"]:
            raise ValueError(f"Group {key} table must begin with FID IID.")
        if table.duplicated(["FID", "IID"]).any() or len(table) != len(fam):
            raise ValueError(f"Group {key} table has duplicate/wrong-count IDs.")
        if not table[["FID", "IID"]].equals(fam):
            raise ValueError(f"Group {key} table is not in exact FAM order.")
    env = tables["environment"]
    cov = tables["covariates"]
    phen = tables["phenotypes"]
    if list(env.columns) != ["FID", "IID", "ENV"]:
        raise ValueError("Group environment output header must be exactly FID IID ENV.")
    if list(phen.columns[2:]) != list(group.get("phenotype_labels", [])):
        raise ValueError("Group phenotype output header differs from manifest labels.")
    if list(cov.columns[2:]) != list(group.get("covariate_columns", [])):
        raise ValueError("Group covariate output header differs from manifest columns.")
    env_values = env[["ENV"]].apply(pd.to_numeric, errors="coerce")
    cov_values = cov.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    phen_values = phen.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    env_keep = env_values.notna().all(axis=1).to_numpy()
    cov_keep = cov_values.notna().all(axis=1).to_numpy()
    phen_keep = phen_values.notna().all(axis=1).to_numpy()
    if not np.array_equal(env_keep, cov_keep) or not np.array_equal(
        env_keep, phen_keep
    ):
        raise ValueError("Group outputs do not share one exact complete-case mask.")
    keep = env_keep
    if not keep.any():
        raise ValueError("Group complete-case cohort is empty.")
    if (
        not np.isfinite(env_values.loc[keep].to_numpy()).all()
        or not np.isfinite(cov_values.loc[keep].to_numpy()).all()
        or not np.isfinite(phen_values.loc[keep].to_numpy()).all()
    ):
        raise ValueError("Group selected rows contain nonfinite values.")
    if (
        env_values.loc[~keep].notna().any(axis=None)
        or cov_values.loc[~keep].notna().any(axis=None)
        or phen_values.loc[~keep].notna().any(axis=None)
    ):
        raise ValueError(
            "Group rows outside the common cohort are not uniformly missing."
        )
    if int(keep.sum()) != int(group.get("n_selected_samples", -1)):
        raise ValueError("Group selected N differs from its manifest claim.")
    if _id_digest(fam) != group.get("fam_order_digest") or _id_digest(
        fam, keep
    ) != group.get("selected_id_digest"):
        raise ValueError("Group FAM/selected ID digest differs from observed rows.")

    columns: list[np.ndarray] = []
    constant: list[str] = []
    kept: list[str] = []
    ddof = int(group.get("environment", {}).get("ddof", 1))
    for name in cov_values.columns:
        values = cov_values.loc[keep, name].to_numpy(dtype=np.float64)
        centered = values - values.mean()
        sd = float(centered.std(ddof=ddof))
        if not np.isfinite(sd) or sd <= 0:
            constant.append(str(name))
        else:
            kept.append(str(name))
            columns.append(centered / sd)
    env_array = env_values.loc[keep, "ENV"].to_numpy(dtype=np.float64)
    env_centered = env_array - env_array.mean()
    env_sd = float(env_centered.std(ddof=ddof))
    if not np.isfinite(env_sd) or env_sd <= 0:
        raise ValueError("Group environment is constant in selected rows.")
    columns.append(env_centered / env_sd)
    design = np.column_stack(columns)
    singular = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    tolerance = max(
        1e-10, max(design.shape) * np.finfo(np.float64).eps * float(singular[0])
    )
    rank = int(np.sum(singular > tolerance))
    residual = int(keep.sum()) - rank - 1
    if rank != int(
        group.get("fixed_effect_rank_excluding_intercept", -1)
    ) or residual != int(group.get("residual_rank", -1)):
        raise ValueError(
            "Observed group rank/residual rank differs from manifest claim."
        )
    if kept != list(group.get("kept_nonconstant_covariates", [])) or constant != list(
        group.get("constant_covariates", [])
    ):
        raise ValueError(
            "Observed constant/nonconstant covariates differ from manifest claims."
        )
    group_config = panel.get("groups", {}).get(group.get("label"), {})
    if dataset == "full":
        expected = (
            group_config.get("expected_common_n_full"),
            group_config.get("expected_rank_full"),
            group_config.get("expected_residual_rank_full"),
        )
        if (int(keep.sum()), rank, residual) != expected:
            raise ValueError(
                "Observed full-cohort N/rank/residual differs from panel config."
            )
    return {
        "n_samples": len(fam),
        "n_selected": int(keep.sum()),
        "rank": rank,
        "residual_rank": residual,
    }


def _common_generation_args(
    config: dict,
    geno: Path,
    outputs: dict[str, Path],
    *,
    slots: int,
) -> list[str]:
    estimator = config["estimator"]
    return [
        "--geno",
        str(geno),
        "--env",
        str(outputs["environment"]),
        "--covar",
        str(outputs["covariates"]),
        "--gxe-kernel-mode",
        str(estimator["kernel_mode"]),
        "--gxe-genotype-scale",
        str(estimator["genotype_scale"]),
        "--write-gxe-jackknife",
        "--njack",
        str(estimator["jackknife_blocks"]),
        "--rand-dist",
        str(estimator["random_distribution"]),
        "--seed",
        str(estimator["seed"]),
        "--dtype",
        str(estimator["dtype"]),
        "--ddof",
        str(estimator["ddof"]),
        f"--gxe-missing-values={estimator['missing_values']}",
        "--impute-method",
        str(estimator["impute_method"]),
        "--step_size",
        str(estimator["step_size"]),
        "--target-xz-mem",
        str(estimator["target_xz_mem_gib"]),
        "--num-threads",
        str(slots),
    ]


def _validate_score_triplet(
    moments_record: dict,
    gwas_record: dict,
    gwis_record: dict,
    scratch_root: Path,
    *,
    reference_sha: str,
    cache_sha: str,
    trait: str,
) -> dict:
    import math

    moments_path, _ = _verify_record(
        moments_record, "phenotype moments", scratch_root=scratch_root, private=True
    )
    gwas_path, gwas_verified = _verify_record(
        gwas_record, "marginal GWAS score", scratch_root=scratch_root, private=True
    )
    gwis_path, gwis_verified = _verify_record(
        gwis_record, "marginal GWIS score", scratch_root=scratch_root, private=True
    )
    payload, _, _ = _read_json(moments_path, "phenotype moments", private=True)
    if (
        payload.get("kind") != "summit.gxe.phenotype_moments"
        or payload.get("schema_version") != 3
    ):
        raise ValueError("Phenotype moments kind/schema is invalid.")
    if payload.get("phenotype") != trait:
        raise ValueError("Phenotype moments trait differs from requested fit.")
    if payload.get("feature_cache_sha256") != cache_sha:
        raise ValueError("Phenotype moments bind a different feature cache.")
    for key in ("n_samples", "residual_rank"):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Phenotype moments contain invalid {key!r}.")
    for key in ("q_nxe", "q_residual", "phenotype_residual_variance_fraction"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"Phenotype moments contain invalid {key!r}.")
    if float(payload["phenotype_residual_variance_fraction"]) > 1.0 + 1.0e-8:
        raise ValueError("Phenotype residual variance fraction exceeds one.")
    expected_reference = payload.get("reference_manifest_sha256")
    if expected_reference != reference_sha:
        raise ValueError("Phenotype moments bind a different merged reference.")
    declared = payload.get("files", {})
    for key, path in (("gwas", gwas_path), ("gwis", gwis_path)):
        relative = Path(str(declared.get(key, "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Phenotype moments contain an unsafe score path.")
        if _absolute(moments_path.parent / relative) != path:
            raise ValueError(
                "Phenotype moments score path differs from supplied artifact."
            )
    hashes = payload.get("score_sha256", {})
    if (
        hashes.get("gwas") != gwas_verified["sha256"]
        or hashes.get("gwis") != gwis_verified["sha256"]
    ):
        raise ValueError(
            "Phenotype moments score hashes differ from supplied artifacts."
        )
    return payload


def _expected_cli_outputs(
    task: str, prefix: Path, *, traits: Sequence[str] = ()
) -> list[Path]:
    if task == "cache":
        return [Path(str(prefix) + ".gxe.cache.npz"), Path(str(prefix) + ".gxe.log")]
    if task == "shard":
        return [
            Path(str(prefix) + suffix)
            for suffix in (
                ".gxx.ldscore.gz",
                ".gxe.ldscore.gz",
                ".exg.ldscore.gz",
                ".gee.ldscore.gz",
                ".gxe.jackknife.npz",
                ".gxe.shard.identity.json",
                ".gxe.shard.json",
                ".gxe.log",
            )
        ]
    if task == "merge":
        return [
            Path(str(prefix) + suffix)
            for suffix in (
                ".gxx.ldscore.gz",
                ".gxe.ldscore.gz",
                ".exg.ldscore.gz",
                ".gee.ldscore.gz",
                ".gxe.diag.tsv.gz",
                ".gxe.jackknife.npz",
                ".gxe.ref.json",
                ".gxe.log",
            )
        ]
    if task == "score":
        paths = [Path(str(prefix) + ".gxe.log")]
        for trait in traits:
            trait_prefix = Path(f"{prefix}.{trait}")
            paths.extend(
                Path(str(trait_prefix) + suffix)
                for suffix in (
                    ".gxe.gwas.tsv.gz",
                    ".gxe.gwis.tsv.gz",
                    ".gxe.moments.json",
                )
            )
        return paths
    if task == "fit":
        return [
            Path(str(prefix) + ".gxe.results.tsv"),
            Path(str(prefix) + ".gxe.fit.json"),
            Path(str(prefix) + ".gxe.log"),
        ]
    raise ValueError(f"No CLI outputs declared for task {task!r}.")


def _ensure_outputs_absent(paths: Sequence[Path], scratch_root: Path) -> None:
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            raise ValueError(f"Duplicate expected output path: {path}")
        seen.add(path)
        if not _is_below(path, scratch_root):
            raise ValueError(f"Expected output is outside scratch root: {path}")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing existing task output: {path}")


def _assert_frozen_imports(frozen: dict) -> dict:
    """Import the exact sealed modules before any SUMMIT operation."""
    root = Path(frozen["root"])
    source_root = root / "src"
    native_root = Path(frozen["native_build_dir"])
    if _absolute(sys.executable) != Path(frozen["python"]):
        raise RuntimeError(
            f"Runtime Python differs from frozen contract: {sys.executable} != {frozen['python']}"
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required before frozen imports.")

    for name in list(sys.modules):
        if name == "summit" or name.startswith("summit.") or name == "gwldcore":
            del sys.modules[name]
    interpreter_roots = {_absolute(sys.prefix), _absolute(sys.base_prefix)}
    interpreter_paths = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = _absolute(entry)
        if candidate in {source_root, native_root}:
            continue
        if any(
            candidate == prefix or _is_below(candidate, prefix)
            for prefix in interpreter_roots
        ):
            interpreter_paths.append(str(candidate))
    sys.path[:] = [str(source_root), str(native_root), *interpreter_paths]
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            getattr(finder, "__module__", "").startswith("_gwldcore_editable")
            or getattr(finder, "__module__", "").startswith("__editable__")
        )
    ]
    importlib.invalidate_caches()
    modules = {
        "summit": importlib.import_module("summit"),
        "summit.cli": importlib.import_module("summit.cli"),
        "summit.ldscore.gwe_ldscore": importlib.import_module(
            "summit.ldscore.gwe_ldscore"
        ),
        "summit.ldscore.gxe_merge": importlib.import_module("summit.ldscore.gxe_merge"),
        "summit.ldscore.gxe_score": importlib.import_module("summit.ldscore.gxe_score"),
        "summit.inference.gxe": importlib.import_module("summit.inference.gxe"),
        "gwldcore": importlib.import_module("gwldcore"),
    }
    expected_paths = {
        "summit": root / "src" / "summit" / "__init__.py",
        "summit.cli": root / "src" / "summit" / "cli.py",
        "summit.ldscore.gwe_ldscore": root
        / "src"
        / "summit"
        / "ldscore"
        / "gwe_ldscore.py",
        "summit.ldscore.gxe_merge": root
        / "src"
        / "summit"
        / "ldscore"
        / "gxe_merge.py",
        "summit.ldscore.gxe_score": root
        / "src"
        / "summit"
        / "ldscore"
        / "gxe_score.py",
        "summit.inference.gxe": root / "src" / "summit" / "inference" / "gxe.py",
    }
    manifest_files = frozen["code_manifest_payload"]["files"]
    module_records: dict[str, dict] = {}
    for name, expected in expected_paths.items():
        observed = _absolute(inspect.getfile(modules[name]))
        if observed != expected:
            raise RuntimeError(
                f"Frozen import {name!r} resolved unexpectedly: {observed}"
            )
        relative = expected.relative_to(root).as_posix()
        if _sha256(expected) != manifest_files[relative]["sha256"]:
            raise RuntimeError(
                f"Imported module {name!r} differs from the code manifest."
            )
        module_records[name] = _record(expected)
    native_path = _absolute(inspect.getfile(modules["gwldcore"]))
    expected_native = Path(frozen["code_manifest_payload"]["native"]["path"])
    if (
        native_path != expected_native
        or _sha256(native_path) != frozen["code_manifest_payload"]["native"]["sha256"]
    ):
        raise RuntimeError(f"Frozen native import resolved unexpectedly: {native_path}")
    module_records["gwldcore"] = _record(native_path)

    writer = modules["summit.ldscore.gwe_ldscore"]
    merger = modules["summit.ldscore.gxe_merge"]
    scorer = modules["summit.ldscore.gxe_score"]
    if int(writer._FEATURE_CACHE_SCHEMA_VERSION) != 2:
        raise RuntimeError("Imported feature-cache schema is not v2.")
    if not hasattr(merger, "_validate_feature_cache_semantics"):
        raise RuntimeError(
            "Imported shard merger lacks strict schema-v2 cache validation."
        )
    if int(scorer._SCHEMA_VERSION) != 3:
        raise RuntimeError(
            "Imported phenotype scorer does not require schema-v3 references."
        )

    try:
        package_versions = {
            name: importlib_metadata.version(name) for name in REQUIRED_DISTRIBUTIONS
        }
    except Exception as error:  # pragma: no cover - environment-specific failure detail
        raise RuntimeError("Could not seal required package versions.") from error
    return {
        "modules": module_records,
        "package_versions": package_versions,
        "sys_executable": sys.executable,
        "sys_path": list(sys.path),
        "cli_module": modules["summit.cli"],
        "cache_semantic_validator": writer._validate_feature_cache_semantics,
    }


def _prepare_task(
    spec: dict,
    common: dict,
    config: dict,
    *,
    artifacts_ready: bool,
) -> dict:
    task = spec["task"]
    args = spec.get("task_args")
    if not isinstance(args, dict):
        raise ValueError("Job spec lacks task_args.")
    scratch_root = common["scratch_root"]
    panel = common["frozen"]["panel_payload"]
    panel_sha = config["panel_config_sha256"]
    artifacts = common["artifacts"]
    if artifacts_ready:
        _require_directory(artifacts, "fresh artifact directory", private=True)

    if task == "stage_verify":
        _validate_qacct_dependencies(spec, scratch_root, task, expected_count=0)
        dataset = args.get("dataset")
        if dataset not in ("full", "subset_50k"):
            raise ValueError("Stage dataset must be 'full' or 'subset_50k'.")
        geno = _validate_genotype_prefix(args.get("geno_prefix", ""), scratch_root)
        _, source_manifest, staged_records = _verify_staged_sources(
            args.get("source_manifest"),
            args.get("staged_source_dir", ""),
            scratch_root,
            panel_sha,
        )
        group_records = args.get("group_manifests")
        if not isinstance(group_records, list):
            raise ValueError("Stage group_manifests must be a list.")
        groups = []
        group_paths = []
        for record in group_records:
            path, payload, outputs = _load_group(record, scratch_root, panel)
            group_paths.append(path)
            groups.append(payload)
            _deep_validate_group(
                Path(str(geno) + ".fam"), payload, outputs, panel, dataset
            )
            if dataset == "full" and payload.get(
                "fam_order_digest"
            ) != source_manifest.get("fam_order_digest"):
                raise ValueError(
                    f"Group {payload.get('label')!r} FAM order differs from the source manifest."
                )
            label = str(payload.get("label"))
            declared_sources = payload.get("sources", {})
            expected_source_records = [staged_records[("covariates", label)]]
            expected_source_records.extend(
                staged_records[("phenotypes", trait)]
                for trait in payload.get("phenotype_labels", [])
            )
            observed_source_records = [
                declared_sources.get("covariates_and_environment"),
                *declared_sources.get("phenotypes", []),
            ]
            if len(observed_source_records) != len(expected_source_records):
                raise ValueError(
                    f"Group {label!r} source count differs from source manifest."
                )
            for observed, expected in zip(
                observed_source_records, expected_source_records
            ):
                if not isinstance(observed, dict) or any(
                    observed.get(key) != expected[key]
                    for key in ("path", "bytes", "sha256")
                ):
                    raise ValueError(
                        f"Group {label!r} source path/size/hash differs from its allowlist."
                    )
            fam_source = declared_sources.get("fam")
            if not isinstance(fam_source, dict) or fam_source != _record(
                Path(str(geno) + ".fam")
            ):
                raise ValueError(
                    f"Group {label!r} FAM source differs from staged genotype."
                )
        labels = [group.get("label") for group in groups]
        expected_labels = list(panel.get("groups", {}).keys())
        if len(labels) != len(set(labels)) or set(labels) != set(expected_labels):
            raise ValueError(
                f"Stage verification requires exact group set {expected_labels}; got {labels}."
            )
        report = artifacts / "stage_verification.json"
        expected_outputs = [report]
        verify_script = (
            common["frozen"]["root"]
            / "scripts"
            / "gxe"
            / "hoffman"
            / "verify_staged_inputs.py"
        )
        command = [
            str(common["frozen"]["python"]),
            "-I",
            "-B",
            str(verify_script),
            "--config",
            str(common["frozen"]["panel"]),
            "--dataset",
            dataset,
            "--geno-prefix",
            str(geno),
        ]
        for path in group_paths:
            command.extend(["--group-manifest", str(path)])
        command.extend(["--report", str(report)])
        plan = {
            "mode": "subprocess",
            "command": command,
            "outputs": expected_outputs,
            "details": {
                "dataset": dataset,
                "groups": labels,
                "source_manifest_fam_order_digest": source_manifest.get(
                    "fam_order_digest"
                ),
            },
            "group_payloads": groups,
            "geno": geno,
        }
    else:
        group_path = None
        group = None
        group_outputs = None
        geno = None
        stage_report = None
        if task in {"cache", "shard", "score"}:
            group_path, group, group_outputs = _load_group(
                args.get("group_manifest"), scratch_root, panel
            )
            geno = _validate_genotype_prefix(args.get("geno_prefix", ""), scratch_root)
            stage_report = _validate_stage_report(
                args.get("stage_verification"),
                scratch_root,
                panel_sha,
                geno,
                str(group["label"]),
                _sha256(group_path),
            )

        if task == "cache":
            dependencies = _validate_qacct_dependencies(
                spec, scratch_root, task, expected_count=1
            )
            stage_record = args.get("stage_verification")
            if not isinstance(stage_record, dict) or stage_record.get("sha256") not in {
                item.get("sha256") for item in dependencies[0].get("outputs", [])
            }:
                raise ValueError(
                    "Cache stage-verification report is not an output of its completed stage job."
                )
            prefix = artifacts / "cache"
            command = [
                "--gxe-build-cache",
                *_common_generation_args(
                    config,
                    geno,
                    group_outputs,
                    slots=config["resources"]["cache"]["slots"],
                ),
                "--nvecs",
                str(config["estimator"]["production_probes"]),
                "--out",
                str(prefix),
            ]
            plan = {
                "mode": "summit",
                "command": command,
                "outputs": _expected_cli_outputs(task, prefix),
                "details": {
                    "group": group["label"],
                    "group_manifest_sha256": _sha256(group_path),
                    "stage_verification_sha256": stage_record["sha256"],
                },
                "group_payload": group,
                "stage_report": stage_report,
            }
        elif task == "shard":
            dependencies = _validate_qacct_dependencies(
                spec, scratch_root, task, expected_count=1
            )
            index = args.get("shard_index")
            role = args.get("role", "production")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index <= 9
            ):
                raise ValueError("shard_index must be an integer from 0 through 9.")
            if role not in ("production", "benchmark") or (
                role == "benchmark" and index != 0
            ):
                raise ValueError("Only shard 0 may use the benchmark role.")
            cache_path, cache_record = _verify_record(
                args.get("cache"),
                "feature cache",
                scratch_root=scratch_root,
                private=True,
            )
            cache = _validate_cache_file(cache_path, config, group)
            _validate_cache_stage_genotype(cache, stage_report)
            upstream_outputs = dependencies[0].get("outputs", [])
            if cache_record["sha256"] not in {
                item.get("sha256") for item in upstream_outputs
            }:
                raise ValueError(
                    "Shard cache is not an output of its completed cache job."
                )
            upstream_details = dependencies[0].get("task_details", {})
            if upstream_details.get("group") != group.get(
                "label"
            ) or upstream_details.get("group_manifest_sha256") != _sha256(group_path):
                raise ValueError(
                    "Shard group manifest differs from its completed cache job."
                )
            prefix = artifacts / "shard"
            command = [
                *_common_generation_args(
                    config,
                    geno,
                    group_outputs,
                    slots=common["resource"]["slots"],
                ),
                "--gxe-feature-cache",
                str(cache_path),
                "--gxe-reference-shard",
                "--gxe-probe-offset",
                str(index * config["estimator"]["probes_per_shard"]),
                "--nvecs",
                str(config["estimator"]["probes_per_shard"]),
                "--out",
                str(prefix),
            ]
            plan = {
                "mode": "summit",
                "command": command,
                "outputs": _expected_cli_outputs(task, prefix),
                "details": {
                    "group": group["label"],
                    "group_manifest_sha256": _sha256(group_path),
                    "cache_sha256": cache_record["sha256"],
                    "shard_index": index,
                    "role": role,
                },
                "cache": cache,
            }
        elif task == "merge":
            cache_path, cache_record = _verify_record(
                args.get("cache"),
                "feature cache",
                scratch_root=scratch_root,
                private=True,
            )
            cache = _validate_cache_file(cache_path, config)
            shard_records = args.get("shards")
            if not isinstance(shard_records, list) or not 1 <= len(shard_records) <= 10:
                raise ValueError("Merge requires one through ten shard records.")
            dependencies = _validate_qacct_dependencies(
                spec, scratch_root, task, expected_count=len(shard_records)
            )
            shards = [
                _validate_shard_record(
                    record,
                    scratch_root,
                    config,
                    cache_record["sha256"],
                    cache,
                )
                for record in shard_records
            ]
            intervals = sorted(
                (
                    item[1]["randomization"]["probe_offset"],
                    item[1]["randomization"]["probe_stop"],
                )
                for item in shards
            )
            expected_intervals = [
                (offset, offset + config["estimator"]["probes_per_shard"])
                for offset in range(
                    0,
                    len(shards) * config["estimator"]["probes_per_shard"],
                    config["estimator"]["probes_per_shard"],
                )
            ]
            if intervals != expected_intervals:
                raise ValueError(
                    "Merge shards must be the exact contiguous prefix of production intervals."
                )
            dependency_output_hashes = {
                output.get("sha256")
                for dependency in dependencies
                for output in dependency.get("outputs", [])
            }
            if any(
                _sha256(path) not in dependency_output_hashes for path, _, _ in shards
            ):
                raise ValueError(
                    "A merge shard is not bound by its completed shard qacct provenance."
                )
            if any(
                dependency.get("task_details", {}).get("role") != "production"
                for dependency in dependencies
            ):
                raise ValueError(
                    "Benchmark shard receipts cannot enter production/checkpoint merges."
                )
            probes = len(shards) * config["estimator"]["probes_per_shard"]
            prefix = artifacts / f"B{probes:03d}"
            command = [
                "--gxe-merge-shards",
                *[
                    str(item[0])
                    for item in sorted(
                        shards,
                        key=lambda item: item[1]["randomization"]["probe_offset"],
                    )
                ],
                "--gxe-feature-cache",
                str(cache_path),
            ]
            if probes < config["estimator"]["production_probes"]:
                command.append("--allow-low-probe-gxe-jackknife")
            command.extend(["--out", str(prefix)])
            plan = {
                "mode": "summit",
                "command": command,
                "outputs": _expected_cli_outputs(task, prefix),
                "details": {"probes": probes, "cache_sha256": cache_record["sha256"]},
                "cache": cache,
            }
        elif task == "score":
            dependencies = _validate_qacct_dependencies(
                spec, scratch_root, task, expected_count=1
            )
            cache_path, cache_record = _verify_record(
                args.get("cache"),
                "feature cache",
                scratch_root=scratch_root,
                private=True,
            )
            cache = _validate_cache_file(cache_path, config, group)
            _validate_cache_stage_genotype(cache, stage_report)
            reference_path, reference_record = _verify_record(
                args.get("reference"),
                "B100 reference",
                scratch_root=scratch_root,
                private=True,
            )
            _, reference, _ = _validate_reference_record(
                args.get("reference"),
                scratch_root,
                config,
                cache_record["sha256"],
                expected_probes=config["estimator"]["production_probes"],
            )
            _validate_reference_cache_identity(reference, cache)
            if reference_record["sha256"] not in {
                output.get("sha256") for output in dependencies[0].get("outputs", [])
            }:
                raise ValueError(
                    "Score reference is not an output of completed B100 merge."
                )
            traits = args.get("traits")
            if traits != group.get("phenotype_labels"):
                raise ValueError(
                    "Wide-score trait list must exactly equal configured group order."
                )
            prefix = artifacts / "scores"
            command = [
                "--gxe-score-reference",
                str(reference_path),
                "--geno",
                str(geno),
                "--env",
                str(group_outputs["environment"]),
                "--covar",
                str(group_outputs["covariates"]),
                "--gxe-pheno",
                str(group_outputs["phenotypes"]),
                "--gxe-pheno-cols",
                ",".join(traits),
                f"--gxe-missing-values={config['estimator']['missing_values']}",
                "--step_size",
                str(config["estimator"]["step_size"]),
                "--num-threads",
                str(config["resources"]["score"]["slots"]),
                "--out",
                str(prefix),
            ]
            plan = {
                "mode": "summit",
                "command": command,
                "outputs": _expected_cli_outputs(task, prefix, traits=traits),
                "details": {
                    "group": group["label"],
                    "group_manifest_sha256": _sha256(group_path),
                    "traits": traits,
                    "cache_sha256": cache_record["sha256"],
                    "reference_sha256": reference_record["sha256"],
                },
                "reference_sha": reference_record["sha256"],
                "cache_sha": cache_record["sha256"],
                "traits": traits,
            }
        elif task == "fit":
            dependencies = _validate_qacct_dependencies(
                spec, scratch_root, task, expected_count=1
            )
            cache_path, cache_record = _verify_record(
                args.get("cache"),
                "feature cache",
                scratch_root=scratch_root,
                private=True,
            )
            cache = _validate_cache_file(cache_path, config)
            reference_path, reference_record = _verify_record(
                args.get("reference"),
                "B100 reference",
                scratch_root=scratch_root,
                private=True,
            )
            reference_path, reference, _ = _validate_reference_record(
                args.get("reference"),
                scratch_root,
                config,
                cache_record["sha256"],
                expected_probes=config["estimator"]["production_probes"],
            )
            _validate_reference_cache_identity(reference, cache)
            group_label = args.get("group")
            trait = args.get("trait")
            if trait not in panel.get("groups", {}).get(group_label, {}).get(
                "phenotypes", []
            ):
                raise ValueError("Fit trait is absent from its configured group.")
            _validate_score_triplet(
                args.get("moments"),
                args.get("gwas"),
                args.get("gwis"),
                scratch_root,
                reference_sha=reference_record["sha256"],
                cache_sha=cache_record["sha256"],
                trait=trait,
            )
            supplied_hashes = {
                item.get("sha256")
                for item in (args.get("moments"), args.get("gwas"), args.get("gwis"))
                if isinstance(item, dict)
            }
            dependency_hashes = {
                output.get("sha256") for output in dependencies[0].get("outputs", [])
            }
            if not supplied_hashes.issubset(dependency_hashes):
                raise ValueError(
                    "Fit score triplet is not bound by completed score qacct provenance."
                )
            prefix = artifacts / "fit"
            command = [
                "--gxe-fit",
                str(reference_path),
                "--gxe-gwas",
                str(_absolute(args["gwas"]["path"])),
                "--gwis",
                str(_absolute(args["gwis"]["path"])),
                "--gxe-moments",
                str(_absolute(args["moments"]["path"])),
                "--gxe-max-condition",
                "1e12",
                "--out",
                str(prefix),
            ]
            plan = {
                "mode": "summit",
                "command": command,
                "outputs": _expected_cli_outputs(task, prefix),
                "details": {
                    "group": group_label,
                    "trait": trait,
                    "cache_sha256": cache_record["sha256"],
                    "reference_sha256": reference_record["sha256"],
                    "score_sha256": sorted(supplied_hashes),
                },
                "reference": reference,
            }
        else:
            raise RuntimeError(f"Unhandled task {task!r}.")

    _ensure_outputs_absent(plan["outputs"], scratch_root)
    return plan


def _validate_score_table(
    path: Path,
    *,
    reference: dict,
    moments: dict,
) -> None:
    import numpy as np
    import pandas as pd

    expected_columns = [
        "CHR",
        "SNP",
        "BP",
        "A1",
        "A2",
        "N",
        "DF",
        "SCORE_MODE",
        "SCORE",
    ]
    digest = hashlib.sha256()
    rows = 0
    seen: set[str] = set()
    reader = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        chunksize=100_000,
        dtype={"CHR": str, "SNP": str, "A1": str, "A2": str},
    )
    for chunk in reader:
        if list(chunk.columns) != expected_columns:
            raise ValueError(f"Score table has an unexpected header: {path}")
        if chunk.empty:
            continue
        if chunk["SNP"].duplicated().any() or any(
            value in seen for value in chunk["SNP"]
        ):
            raise ValueError(f"Score table contains duplicate SNP identifiers: {path}")
        seen.update(str(value) for value in chunk["SNP"])
        numeric = chunk[["BP", "N", "DF", "SCORE"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Score table contains nonfinite numeric values: {path}")
        if (
            not (numeric["N"] == int(moments["n_samples"])).all()
            or not (numeric["DF"] == int(moments["residual_rank"])).all()
        ):
            raise ValueError(f"Score table N/DF differs from phenotype moments: {path}")
        if not (chunk["SCORE_MODE"] == "marginal_cross_product").all():
            raise ValueError(f"Score table is not in marginal-score mode: {path}")
        for chrom, snp, bp, a1, a2 in chunk[
            ["CHR", "SNP", "BP", "A1", "A2"]
        ].itertuples(index=False, name=None):
            digest.update(
                "\x1f".join(
                    (str(chrom), str(snp), str(int(bp)), str(a1), str(a2))
                ).encode("utf-8")
            )
            digest.update(b"\n")
        rows += len(chunk)
    if rows != int(reference["annotation_masses"][0]):
        raise ValueError(f"Score table row count differs from M: {path}")
    if digest.hexdigest() != reference.get("variant_digest"):
        raise ValueError(
            f"Score table ordered variant digest differs from reference: {path}"
        )


def _validate_fit_outputs(prefix: Path, config: dict) -> None:
    import numpy as np
    import pandas as pd

    json_path = Path(str(prefix) + ".gxe.fit.json")
    table_path = Path(str(prefix) + ".gxe.results.tsv")
    payload, _, _ = _read_json(json_path, "GxE fit JSON", private=True)
    if payload.get("kind") != "summit.gxe.fit" or payload.get("schema_version") != 3:
        raise ValueError("Fit output kind/schema differs from production contract.")
    names = ["G:L2_0", "GxE:L2_0", "NxE", "residual"]
    if payload.get("component_names") != names:
        raise ValueError(
            "Fit output does not contain the exact four one-bin components."
        )
    if payload.get("rank") != len(names):
        raise ValueError("Primary GxE fit is not full rank.")
    condition = float(payload.get("condition_number", np.nan))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise ValueError("Primary GxE fit exceeds the fixed condition-number gate.")
    relative_residual = float(payload.get("relative_residual", np.nan))
    nxe_residual_correlation = float(
        payload.get("nxe_residual_kernel_correlation", np.nan)
    )
    residual_fraction = float(
        payload.get("phenotype_residual_variance_fraction", np.nan)
    )
    if (
        not np.isfinite(relative_residual)
        or relative_residual < 0.0
        or not np.isfinite(nxe_residual_correlation)
        or abs(nxe_residual_correlation) > 1.0 + 1.0e-8
        or not np.isfinite(residual_fraction)
        or not 0.0 < residual_fraction <= 1.0 + 1.0e-8
    ):
        raise ValueError("Fit JSON contains invalid scalar diagnostics.")
    vector_keys = (
        "kernel_traces",
        "coefficients",
        "variance_contributions",
        "proportions",
        "standard_errors",
    )
    vectors: dict[str, np.ndarray] = {}
    for key in vector_keys:
        values = np.asarray(payload.get(key), dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Fit JSON vector {key!r} is incomplete or nonfinite.")
        vectors[key] = values
    if np.any(vectors["standard_errors"] < 0.0):
        raise ValueError("Fit jackknife standard errors must be nonnegative.")
    if not np.isclose(vectors["proportions"].sum(), 1.0, rtol=1e-8, atol=1e-8):
        raise ValueError("Fit proportions do not sum to one.")
    labels = payload.get("jackknife_block_labels")
    replicates = np.asarray(payload.get("jackknife_estimates"), dtype=np.float64)
    j = int(config["estimator"]["jackknife_blocks"])
    if not isinstance(labels, list) or len(labels) != j or len(set(labels)) != j:
        raise ValueError("Fit JSON lacks the exact J unique jackknife labels.")
    if replicates.shape != (j, 4) or not np.all(np.isfinite(replicates)):
        raise ValueError("Fit JSON lacks finite J-by-four delete-block estimates.")
    for key, expected in (
        ("original_scale_proportions", vectors["proportions"] * residual_fraction),
        (
            "original_scale_standard_errors",
            vectors["standard_errors"] * residual_fraction,
        ),
    ):
        values = np.asarray(payload.get(key), dtype=np.float64)
        if (
            values.shape != (4,)
            or not np.all(np.isfinite(values))
            or not np.allclose(values, expected, rtol=1.0e-12, atol=1.0e-12)
        ):
            raise ValueError(
                f"Fit JSON {key!r} differs from its residual-scale conversion."
            )
    for key, shape in (
        ("normal_matrix", (4, 4)),
        ("kernel_correlation_matrix", (4, 4)),
        ("rhs", (4,)),
        ("singular_values", (4,)),
        ("normal_eigenvalues", (4,)),
    ):
        values = np.asarray(payload.get(key), dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Fit JSON diagnostic {key!r} is incomplete or nonfinite.")

    frame = pd.read_csv(table_path, sep="\t")
    required_columns = [
        "component",
        "coefficient",
        "kernel_trace",
        "variance_contribution",
        "proportion",
        "proportion_se",
        "original_scale_proportion",
        "z",
        "original_scale_se",
    ]
    if list(frame.columns) != required_columns or frame["component"].tolist() != names:
        raise ValueError("Fit TSV header/component order differs from fit JSON.")
    comparisons = {
        "coefficient": "coefficients",
        "kernel_trace": "kernel_traces",
        "variance_contribution": "variance_contributions",
        "proportion": "proportions",
        "proportion_se": "standard_errors",
    }
    for column, key in comparisons.items():
        observed = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.all(np.isfinite(observed)) or not np.allclose(
            observed, vectors[key], rtol=5e-10, atol=5e-12
        ):
            raise ValueError(f"Fit TSV column {column!r} differs from fit JSON.")


def _postvalidate_task(
    spec: dict,
    common: dict,
    config: dict,
    plan: dict,
    runtime: dict,
) -> list[dict]:
    task = spec["task"]
    expected = plan["outputs"]
    expected_set = {_absolute(path) for path in expected}
    observed_set = {_absolute(path) for path in common["artifacts"].iterdir()}
    if observed_set != expected_set:
        raise ValueError(
            "Task artifact set differs from its exact contract; "
            f"missing={sorted(map(str, expected_set - observed_set))}, "
            f"extra={sorted(map(str, observed_set - expected_set))}."
        )
    for path in expected:
        _require_file(path, f"{task} output", private=True)

    args = spec["task_args"]
    if task == "stage_verify":
        report_path = expected[0]
        report, _, _ = _read_json(
            report_path, "stage verification report", private=True
        )
        if (
            report.get("kind") != "summit.gxe.staged_input_verification"
            or report.get("schema_version") != 1
            or report.get("config", {}).get("sha256") != config["panel_config_sha256"]
            or report.get("dataset") != plan["details"]["dataset"]
            or _absolute(report.get("staged_genotype_prefix", "")) != plan["geno"]
        ):
            raise ValueError(
                "Stage verification report differs from the sealed task contract."
            )
        labels = [item.get("label") for item in report.get("groups", [])]
        expected_labels = list(common["frozen"]["panel_payload"].get("groups", {}))
        if labels != plan["details"]["groups"] or set(labels) != set(expected_labels):
            raise ValueError(
                "Stage verification report does not contain the exact group set/order."
            )
    elif task == "cache":
        cache_path = Path(str(common["artifacts"] / "cache") + ".gxe.cache.npz")
        cache = _validate_cache_file(cache_path, config, plan["group_payload"])
        runtime["cache_semantic_validator"](cache["metadata"], cache["arrays"])
        _validate_cache_stage_genotype(cache, plan["stage_report"])
    elif task == "shard":
        manifest = Path(str(common["artifacts"] / "shard") + ".gxe.shard.json")
        _, cache_record = _verify_record(
            args["cache"],
            "feature cache",
            scratch_root=common["scratch_root"],
            private=True,
        )
        _, payload, _ = _validate_shard_record(
            _record(manifest),
            common["scratch_root"],
            config,
            cache_record["sha256"],
            plan["cache"],
        )
        expected_start = (
            plan["details"]["shard_index"] * config["estimator"]["probes_per_shard"]
        )
        if payload["randomization"].get("probe_offset") != expected_start:
            raise ValueError(
                "Shard output probe interval differs from the task contract."
            )
    elif task == "merge":
        probes = int(plan["details"]["probes"])
        manifest = common["artifacts"] / f"B{probes:03d}.gxe.ref.json"
        _, cache_record = _verify_record(
            args["cache"],
            "feature cache",
            scratch_root=common["scratch_root"],
            private=True,
        )
        _, reference, _ = _validate_reference_record(
            _record(manifest),
            common["scratch_root"],
            config,
            cache_record["sha256"],
            expected_probes=probes,
        )
        _validate_reference_cache_identity(reference, plan["cache"])
    elif task == "score":
        reference_path, reference, _ = _validate_reference_record(
            args["reference"],
            common["scratch_root"],
            config,
            plan["cache_sha"],
            expected_probes=config["estimator"]["production_probes"],
        )
        reference_sha = _sha256(reference_path)
        for trait in plan["traits"]:
            prefix = common["artifacts"] / f"scores.{trait}"
            moments_path = Path(str(prefix) + ".gxe.moments.json")
            gwas_path = Path(str(prefix) + ".gxe.gwas.tsv.gz")
            gwis_path = Path(str(prefix) + ".gxe.gwis.tsv.gz")
            moments = _validate_score_triplet(
                _record(moments_path),
                _record(gwas_path),
                _record(gwis_path),
                common["scratch_root"],
                reference_sha=reference_sha,
                cache_sha=plan["cache_sha"],
                trait=trait,
            )
            if (
                moments.get("analysis_fingerprint")
                != reference.get("analysis_fingerprint")
                or moments.get("variant_digest") != reference.get("variant_digest")
                or moments.get("n_samples") != reference.get("n_samples")
                or moments.get("residual_rank") != reference.get("residual_rank")
            ):
                raise ValueError(
                    "Phenotype moments differ from the B100 reference identity."
                )
            _validate_score_table(gwas_path, reference=reference, moments=moments)
            _validate_score_table(gwis_path, reference=reference, moments=moments)
    elif task == "fit":
        _validate_fit_outputs(common["artifacts"] / "fit", config)
    else:  # pragma: no cover - parser and preparation already constrain this
        raise RuntimeError(f"Unhandled postvalidation task {task!r}.")
    return [_record(path) for path in expected]


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _bootstrap_environment(common: dict) -> dict[str, str]:
    environment_root = Path(common["frozen"]["python"]).parent.parent
    return {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": f"{environment_root / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(environment_root / "lib"),
        "TMPDIR": str(common["tmp"]),
        **{name: str(common["resource"]["slots"]) for name in THREAD_ENVIRONMENT},
    }


def _runtime_uge_environment(common: dict) -> dict:
    if (
        not sys.flags.isolated
        or not sys.flags.no_user_site
        or not sys.flags.dont_write_bytecode
    ):
        raise RuntimeError(
            "Hoffman jobs require Python -I -B so user-site imports are disabled and "
            "frozen imports cannot write bytecode."
        )
    resource = common["resource"]
    task_id = os.environ.get("SGE_TASK_ID")
    if task_id not in (None, "", "undefined"):
        raise RuntimeError("UGE array jobs are forbidden for this deployment workflow.")
    job_id_text = os.environ.get("JOB_ID", "")
    slots_text = os.environ.get("NSLOTS", "")
    if not job_id_text.isdigit() or int(job_id_text) <= 0:
        raise RuntimeError("A positive numeric UGE JOB_ID is required.")
    if not slots_text.isdigit() or int(slots_text) != int(resource["slots"]):
        raise RuntimeError(
            f"UGE NSLOTS must equal the sealed resource profile ({resource['slots']})."
        )
    expected_bootstrap = _bootstrap_environment(common)
    for name, value in expected_bootstrap.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"{name} differs from the sealed job bootstrap.")
    return {
        "job_id": int(job_id_text),
        "slots": int(slots_text),
        "task_id": task_id,
        "thread_environment": {name: os.environ[name] for name in THREAD_ENVIRONMENT},
        "python_no_user_site": os.environ["PYTHONNOUSERSITE"],
        "python_flags": {
            "isolated": int(sys.flags.isolated),
            "no_user_site": int(sys.flags.no_user_site),
            "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
            "hash_randomization": int(sys.flags.hash_randomization),
        },
        "bootstrap_environment": expected_bootstrap,
    }


def _rendered_job_script(
    common: dict,
    config_path: Path,
    config_sha: str,
    spec_path: Path,
    spec_sha: str,
) -> str:
    resource = common["resource"]
    stdout_path = common["job_root"] / "stdout.log"
    stderr_path = common["job_root"] / "stderr.log"
    resource_tokens = [
        f"h_data={resource['h_data_gib_per_slot']}G",
        f"h_rt={resource['h_rt']}",
    ]
    if resource["highp"]:
        resource_tokens.append("highp")
    exports = [
        *(
            f"export {name}={shlex.quote(value)}"
            for name, value in _bootstrap_environment(common).items()
        ),
        f"export GXE_FROZEN_PYTHON={shlex.quote(str(common['frozen']['python']))}",
    ]
    wrapper_command = [
        str(common["frozen"]["wrapper"]),
        "--config",
        str(config_path),
        "--expected-config-sha256",
        config_sha,
        "--job-spec",
        str(spec_path),
        "--expected-job-spec-sha256",
        spec_sha,
    ]
    script = "\n".join(
        [
            "#!/bin/bash",
            "#$ -S /bin/bash",
            f"#$ -N {common['job_name']}",
            f"#$ -wd {common['job_root']}",
            f"#$ -o {stdout_path}",
            f"#$ -e {stderr_path}",
            f"#$ -pe shared {resource['slots']}",
            f"#$ -l {','.join(resource_tokens)}",
            "set -euo pipefail",
            "umask 077",
            f"if ( set -o noclobber; : > {shlex.quote(str(common['job_root'] / 'attempt.lock'))} ) 2>/dev/null; then",
            f"  chmod 0600 {shlex.quote(str(common['job_root'] / 'attempt.lock'))}",
            "else",
            "  exit 73",
            "fi",
            *exports,
            f"exec {shlex.join(wrapper_command)}",
            "",
        ]
    )
    if "-tc" in script or "SGE_TASK_ID" in script:
        raise RuntimeError(
            "Rendered UGE script unexpectedly contains an array/concurrency directive."
        )
    return script


def _run_task(args: argparse.Namespace) -> None:
    os.umask(0o077)
    config_path, config, config_sha = _load_config(
        args.config, args.expected_config_sha256
    )
    scratch_root = _absolute(config["scratch_root"])
    spec_path, spec, spec_sha = _load_spec(
        args.job_spec, args.expected_job_spec_sha256, scratch_root
    )
    common = _validate_common(
        spec_path,
        spec,
        config_path,
        config,
        expected_task=args.task,
        rendering=False,
    )
    _atomic_json_noreplace(
        {
            "kind": "summit.gxe.hoffman_attempt",
            "schema_version": 1,
            "task": spec["task"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "deployment_config_sha256": config_sha,
            "job_spec_sha256": spec_sha,
            "observed_uge": {
                "JOB_ID": os.environ.get("JOB_ID"),
                "NSLOTS": os.environ.get("NSLOTS"),
                "SGE_TASK_ID": os.environ.get("SGE_TASK_ID"),
            },
        },
        common["attempt"],
    )
    expected_script = _rendered_job_script(
        common, config_path, config_sha, spec_path, spec_sha
    )
    if (common["job_root"] / "job.sh").read_text(encoding="utf-8") != expected_script:
        raise RuntimeError(
            "Rendered UGE script differs from its sealed deterministic form."
        )
    uge = _runtime_uge_environment(common)
    for target, label in (
        (common["artifacts"], "artifact directory"),
        (common["invocation"], "invocation provenance"),
        (common["receipt"], "process receipt"),
        (common["qacct"], "qacct completion receipt"),
    ):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing reused {label}: {target}")

    runtime = _assert_frozen_imports(common["frozen"])
    plan = _prepare_task(spec, common, config, artifacts_ready=False)
    os.mkdir(common["artifacts"], mode=0o700)
    _require_directory(common["artifacts"], "fresh artifact directory", private=True)
    command = list(plan["command"])
    recorded_command = command if plan["mode"] == "subprocess" else ["summit", *command]
    start = datetime.now(timezone.utc)
    config_record = _record(config_path)
    spec_record = _record(spec_path)
    invocation = {
        "kind": "summit.gxe.hoffman_invocation",
        "schema_version": 1,
        "task": spec["task"],
        "job_id": uge["job_id"],
        "hostname": os.uname().nodename,
        "started_utc": start.isoformat(),
        "deployment_config": config_record,
        "job_spec": spec_record,
        "job_script": _record(common["job_root"] / "job.sh"),
        "attempt_lock": _record(common["attempt_lock"]),
        "attempt": _record(common["attempt"]),
        "code_manifest": common["frozen"]["code_manifest"],
        "command": recorded_command,
        "execution_mode": plan["mode"],
        "resource_profile": common["resource_profile"],
        "resource": common["resource"],
        "uge": uge,
        "frozen_imports": {
            "sys_executable": runtime["sys_executable"],
            "sys_path": runtime["sys_path"],
            "modules": runtime["modules"],
            "package_versions": runtime["package_versions"],
        },
        "task_details": plan["details"],
    }
    _atomic_json_noreplace(invocation, common["invocation"])

    if plan["mode"] == "subprocess":
        subprocess.run(command, check=True, env=os.environ.copy())
    elif plan["mode"] == "summit":
        cli = runtime["cli_module"]
        previous_argv = sys.argv
        try:
            sys.argv = ["summit", *command]
            try:
                result = cli.main()
            except SystemExit as error:
                code = error.code
                if code not in (None, 0):
                    raise RuntimeError(
                        f"SUMMIT CLI exited unsuccessfully with status {code}."
                    ) from error
            else:
                if result not in (None, 0):
                    raise RuntimeError(
                        f"SUMMIT CLI returned unexpected status {result!r}."
                    )
        finally:
            sys.argv = previous_argv
    else:  # pragma: no cover - plan construction owns this enum
        raise RuntimeError(f"Unsupported execution mode: {plan['mode']!r}")

    outputs = _postvalidate_task(spec, common, config, plan, runtime)
    finished = datetime.now(timezone.utc)
    receipt = {
        "kind": "summit.gxe.hoffman_process_receipt",
        "schema_version": 1,
        "task": spec["task"],
        "job_id": uge["job_id"],
        "hostname": os.uname().nodename,
        "started_utc": start.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - start).total_seconds(),
        "deployment_config": config_record,
        "job_spec": spec_record,
        "job_script": _record(common["job_root"] / "job.sh"),
        "attempt_lock": _record(common["attempt_lock"]),
        "attempt": _record(common["attempt"]),
        "code_manifest": common["frozen"]["code_manifest"],
        "invocation": _record(common["invocation"]),
        "command": recorded_command,
        "execution_mode": plan["mode"],
        "resource_profile": common["resource_profile"],
        "resource": common["resource"],
        "uge": uge,
        "task_details": plan["details"],
        "outputs": outputs,
        "qacct_pending": True,
    }
    _atomic_json_noreplace(receipt, common["receipt"])


def _render_job(args: argparse.Namespace) -> None:
    os.umask(0o077)
    config_path, config, config_sha = _load_config(args.config, None)
    scratch_root = _absolute(config["scratch_root"])
    spec_path, spec, spec_sha = _load_spec(args.job_spec, None, scratch_root)
    task = spec["task"]
    common = _validate_common(
        spec_path,
        spec,
        config_path,
        config,
        expected_task=task,
        rendering=True,
    )
    _prepare_task(spec, common, config, artifacts_ready=False)
    job_script = common["job_root"] / "job.sh"
    stdout_path = common["job_root"] / "stdout.log"
    stderr_path = common["job_root"] / "stderr.log"
    for target in (job_script, stdout_path, stderr_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing existing rendered-job path: {target}")
    if common["tmp"].exists() or common["tmp"].is_symlink():
        raise FileExistsError(
            f"Refusing existing rendered-job temporary path: {common['tmp']}"
        )
    os.mkdir(common["tmp"], mode=0o700)
    _require_directory(common["tmp"], "rendered temporary directory", private=True)
    _atomic_text_noreplace("", stdout_path, 0o600)
    _atomic_text_noreplace("", stderr_path, 0o600)
    script = _rendered_job_script(common, config_path, config_sha, spec_path, spec_sha)
    _atomic_text_noreplace(script, job_script, 0o700)
    _require_file(stdout_path, "rendered stdout log", private=True)
    _require_file(stderr_path, "rendered stderr log", private=True)
    _require_file(job_script, "rendered UGE script", private=False)
    if stat.S_IMODE(job_script.stat().st_mode) != 0o700:
        raise PermissionError("Rendered UGE script is not mode 0700.")
    print(shlex.join(["qsub", str(job_script)]))


def _parse_qacct(raw: str) -> dict[str, str]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if set(line) == {"="}:
            if current:
                records.append(current)
                current = {}
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        if key in current:
            raise ValueError(f"qacct output repeats field {key!r} within one record.")
        current[key] = value.strip()
    if current:
        records.append(current)
    records = [record for record in records if "jobnumber" in record]
    if len(records) != 1:
        raise ValueError(
            f"Expected exactly one final qacct record; observed {len(records)}."
        )
    return records[0]


def _first_integer(value: Any, label: str) -> int:
    token = str(value).split(None, 1)[0]
    if not re.fullmatch(r"-?\d+", token):
        raise ValueError(f"qacct {label} is not an integer: {value!r}")
    return int(token)


def _record_qacct(args: argparse.Namespace) -> None:
    os.umask(0o077)
    config_path, config, config_sha = _load_config(
        args.config, args.expected_config_sha256
    )
    scratch_root = _absolute(config["scratch_root"])
    receipt_path, receipt_record = _verify_record(
        {
            "path": args.receipt,
            "bytes": args.receipt_bytes,
            "sha256": args.receipt_sha256,
        },
        "process receipt",
        scratch_root=scratch_root,
        private=True,
    )
    receipt, _, _ = _read_json(receipt_path, "process receipt", private=True)
    if (
        receipt.get("kind") != "summit.gxe.hoffman_process_receipt"
        or receipt.get("schema_version") != 1
        or receipt.get("qacct_pending") is not True
    ):
        raise ValueError("Unsupported or already-completed process receipt.")
    task = receipt.get("task")
    if task not in TASKS:
        raise ValueError("Process receipt contains an unsupported task.")
    if receipt.get("deployment_config") != _record(config_path):
        raise ValueError(
            "Process receipt was produced under a different deployment config."
        )
    _verify_record(
        receipt.get("job_spec"),
        "completed job spec",
        scratch_root=scratch_root,
        private=True,
    )
    job_script_path, _ = _verify_record(
        receipt.get("job_script"),
        "completed-job UGE script",
        scratch_root=scratch_root,
        private=False,
    )
    if job_script_path.parent != receipt_path.parent:
        raise ValueError("Completed-job UGE script is outside its job root.")
    attempt_lock_path, _ = _verify_record(
        receipt.get("attempt_lock"),
        "completed-job attempt lock",
        scratch_root=scratch_root,
        private=True,
    )
    if attempt_lock_path.parent != receipt_path.parent:
        raise ValueError("Completed-job attempt lock is outside its job root.")
    attempt_path, _ = _verify_record(
        receipt.get("attempt"),
        "completed-job attempt marker",
        scratch_root=scratch_root,
        private=True,
    )
    if attempt_path.parent != receipt_path.parent:
        raise ValueError("Completed-job attempt marker is outside its job root.")
    _verify_record(
        receipt.get("code_manifest"),
        "completed-job code manifest",
        scratch_root=None,
        private=False,
    )
    expected_profile = (
        "shard_benchmark"
        if task == "shard"
        and receipt.get("task_details", {}).get("role") == "benchmark"
        else task
    )
    if receipt.get("resource_profile") != expected_profile:
        raise ValueError("Process receipt resource-profile label is invalid.")
    resource = _validate_resource(expected_profile, config)
    if receipt.get("resource") != resource:
        raise ValueError("Process receipt resource differs from deployment config.")
    job_id = receipt.get("job_id")
    if not isinstance(job_id, int) or isinstance(job_id, bool) or job_id <= 0:
        raise ValueError("Process receipt has an invalid UGE job ID.")
    job_root = receipt_path.parent
    _scratch_path(job_root, scratch_root, "completed job root", kind="directory")
    target = job_root / "completed_qacct.json"
    raw_target = job_root / "qacct.txt"
    _scratch_path(
        target, scratch_root, "qacct completion receipt", kind="file", must_exist=False
    )
    _scratch_path(
        raw_target, scratch_root, "raw qacct output", kind="file", must_exist=False
    )
    invocation_path, _ = _verify_record(
        receipt.get("invocation"),
        "job invocation provenance",
        scratch_root=scratch_root,
        private=True,
    )
    if invocation_path.parent != job_root:
        raise ValueError(
            "Invocation provenance is outside the process-receipt job root."
        )
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Process receipt lacks task outputs.")
    for index, record in enumerate(outputs):
        _verify_record(
            record,
            f"process output {index}",
            scratch_root=scratch_root,
            private=True,
        )
    uge_logs = {}
    for stream in ("stdout", "stderr"):
        log_path = job_root / f"{stream}.log"
        _require_file(log_path, f"final UGE {stream} log", private=True)
        uge_logs[stream] = _record(log_path)

    qacct_command = args.qacct_command
    if not isinstance(qacct_command, str) or not qacct_command:
        raise ValueError("qacct command must be nonempty.")
    command_path = Path(qacct_command)
    if command_path.name != "qacct" or (
        command_path.parent != Path(".") and not command_path.is_absolute()
    ):
        raise ValueError(
            "qacct command must be 'qacct' or an absolute path named qacct."
        )
    completed = subprocess.run(
        [qacct_command, "-j", str(job_id)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    account = _parse_qacct(completed.stdout)
    observed_job = _first_integer(account.get("jobnumber"), "jobnumber")
    failed = _first_integer(account.get("failed"), "failed")
    exit_status = _first_integer(account.get("exit_status"), "exit_status")
    slots = _first_integer(account.get("slots"), "slots")
    if observed_job != job_id:
        raise ValueError("qacct jobnumber differs from the process receipt.")
    if failed != 0 or exit_status != 0:
        raise RuntimeError(
            f"UGE accounting did not complete successfully: failed={failed}, exit_status={exit_status}."
        )
    if slots != int(resource["slots"]):
        raise ValueError(
            "qacct slots differ from the sealed per-slot resource request."
        )
    for key in ("wallclock", "cpu", "maxvmem"):
        if not account.get(key):
            raise ValueError(f"qacct record lacks final {key} provenance.")
    _atomic_text_noreplace(completed.stdout, raw_target, 0o600)
    raw_record = _record(raw_target)
    payload = {
        "kind": "summit.gxe.hoffman_qacct",
        "schema_version": 1,
        "task": task,
        "job_id": job_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "receipt": receipt_record,
        "receipt_sha256": receipt_record["sha256"],
        "deployment_config": receipt["deployment_config"],
        "job_spec": receipt["job_spec"],
        "job_script": receipt["job_script"],
        "attempt_lock": receipt["attempt_lock"],
        "attempt": receipt["attempt"],
        "code_manifest": receipt["code_manifest"],
        "resource_profile": receipt["resource_profile"],
        "resource": receipt["resource"],
        "outputs": receipt["outputs"],
        "task_details": receipt["task_details"],
        "failed": failed,
        "exit_status": exit_status,
        "slots": slots,
        "wallclock": account["wallclock"],
        "cpu": account["cpu"],
        "maxvmem": account["maxvmem"],
        "uge_logs": uge_logs,
        "qacct_raw": raw_record,
        "qacct_stdout_sha256": raw_record["sha256"],
        "qacct_fields": account,
    }
    _atomic_json_noreplace(payload, target)
    print(f"{target}\t{_sha256(target)}\t{target.stat().st_size}")


def _seal_code_manifest(args: argparse.Namespace) -> None:
    os.umask(0o077)
    config_path, config, _ = _load_config(args.config, None)
    root = _absolute(args.root)
    allowed = _absolute(config["allowed_frozen_root"])
    _require_directory(allowed, "allowed frozen-code root", private=False)
    if not _is_below(root, allowed):
        raise ValueError(f"Frozen snapshot root must be below {allowed}: {root}")
    _require_directory(root, "frozen snapshot root", private=False)
    expected_config = root / "scripts" / "gxe" / "hoffman" / "deployment_config.json"
    if config_path != expected_config:
        raise ValueError(
            "seal-code-manifest requires the deployment config inside the snapshot."
        )
    python_path = _absolute(args.python)
    if python_path != _absolute(config["python_executable"]):
        raise ValueError("Code-manifest Python differs from deployment config.")
    if not python_path.exists() or not os.access(python_path, os.X_OK):
        raise FileNotFoundError(
            f"Frozen Python executable is unavailable: {python_path}"
        )
    native_build = _absolute(args.native_build_dir)
    if not _is_below(native_build, root):
        raise ValueError("Native-build directory must be inside the frozen snapshot.")
    _require_directory(native_build, "native-build directory", private=False)
    native_candidates = sorted(
        path
        for path in native_build.glob("gwldcore*")
        if path.suffix in {".so", ".pyd", ".dylib"}
    )
    if len(native_candidates) != 1:
        raise ValueError(
            f"Expected exactly one gwldcore extension in {native_build}; found {len(native_candidates)}."
        )
    native = native_candidates[0]
    _require_file(native, "frozen native module", private=False)
    files: dict[str, dict] = {}
    for relative_text in sorted(_expected_code_files(root)):
        path = root / relative_text
        _require_file(path, f"frozen code file {relative_text!r}", private=False)
        files[relative_text] = {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    output = _absolute(args.output)
    if not _is_below(output, root):
        raise ValueError("Frozen code manifest must be inside the snapshot root.")
    _require_directory(output.parent, "frozen code-manifest directory", private=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing frozen code manifest: {output}")
    payload = {
        "kind": "summit.gxe.frozen_code_manifest",
        "schema_version": 1,
        "root": str(root),
        "python_executable": str(python_path),
        "environment": _environment_fingerprint(python_path),
        "files": files,
        "native_module": {
            "path": native.relative_to(root).as_posix(),
            "bytes": int(native.stat().st_size),
            "sha256": _sha256(native),
        },
    }
    _atomic_json_noreplace(payload, output)
    print(f"{output}\t{_sha256(output)}\t{output.stat().st_size}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Validate and render one UGE job.")
    render.add_argument("--config", required=True)
    render.add_argument("--job-spec", required=True)
    render.set_defaults(handler=_render_job)

    run = subparsers.add_parser("run", help="Execute one rendered task inside UGE.")
    run.add_argument("--task", required=True, choices=TASKS)
    run.add_argument("--config", required=True)
    run.add_argument("--expected-config-sha256", required=True)
    run.add_argument("--job-spec", required=True)
    run.add_argument("--expected-job-spec-sha256", required=True)
    run.set_defaults(handler=_run_task)

    accounting = subparsers.add_parser(
        "record-qacct", help="Seal final accounting after a successful job exits."
    )
    accounting.add_argument("--config", required=True)
    accounting.add_argument("--expected-config-sha256", required=True)
    accounting.add_argument("--receipt", required=True)
    accounting.add_argument("--receipt-sha256", required=True)
    accounting.add_argument("--receipt-bytes", required=True, type=int)
    accounting.add_argument("--qacct-command", default="qacct")
    accounting.set_defaults(handler=_record_qacct)

    seal = subparsers.add_parser(
        "seal-code-manifest", help="Create the deterministic frozen-code allowlist."
    )
    seal.add_argument("--config", required=True)
    seal.add_argument("--root", required=True)
    seal.add_argument("--python", required=True)
    seal.add_argument("--native-build-dir", required=True)
    seal.add_argument("--output", required=True)
    seal.set_defaults(handler=_seal_code_manifest)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
