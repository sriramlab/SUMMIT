#!/usr/bin/env python3
"""Freeze the exact local phenotype/covariate sources for Hoffman staging.

The manifest contains paths, headers, byte sizes, SHA256 hashes, and only
cryptographic summaries of sample IDs.  It never prints or writes sample IDs,
does not copy source data, rejects symlinks, and creates one new mode-0600 JSON
file without overwriting an existing path.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import string
import tempfile
from pathlib import Path
from typing import BinaryIO


SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_no_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {label}: {path}") from None
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")


def _require_regular_file(path: Path, label: str) -> None:
    _require_no_symlink_components(path, label)
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {label}: {path}") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _require_real_directory(path: Path, label: str) -> None:
    _require_no_symlink_components(path, label)
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {label}: {path}") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError(f"{label} must be a real non-symlink directory: {path}")


def _open_regular_binary(path: Path, label: str) -> tuple[BinaryIO, os.stat_result]:
    _require_regular_file(path, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        initial = os.fstat(fd)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        return os.fdopen(fd, "rb"), initial
    except Exception:
        os.close(fd)
        raise


def _unchanged(
    initial: os.stat_result, final: os.stat_result, observed_bytes: int
) -> bool:
    return (
        initial.st_dev == final.st_dev
        and initial.st_ino == final.st_ino
        and initial.st_size == final.st_size == observed_bytes
        and initial.st_mtime_ns == final.st_mtime_ns
    )


def _decode_fields(raw: bytes, path: Path, line_number: int) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Non-UTF-8 input in {path} at line {line_number}.") from error
    fields = text.split()
    if not fields:
        raise ValueError(f"Blank row in {path} at line {line_number}.")
    return fields


def _update_id_digest(digest, fid: str, iid: str) -> None:
    digest.update(fid.encode("utf-8"))
    digest.update(b"\t")
    digest.update(iid.encode("utf-8"))
    digest.update(b"\n")


def _read_bytes_record(path: Path, label: str) -> tuple[bytes, dict]:
    handle, initial = _open_regular_binary(path, label)
    with handle:
        payload = handle.read()
        final = os.fstat(handle.fileno())
    if not _unchanged(initial, final, len(payload)):
        raise RuntimeError(f"{label} changed while it was being read: {path}")
    return payload, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_fam(path: Path) -> tuple[list[tuple[str, str]], dict]:
    handle, initial = _open_regular_binary(path, "reference FAM")
    raw_digest = hashlib.sha256()
    id_digest = hashlib.sha256()
    ids: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    observed_bytes = 0
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            raw_digest.update(raw)
            observed_bytes += len(raw)
            fields = _decode_fields(raw, path, line_number)
            if len(fields) != 6:
                raise ValueError(
                    f"Reference FAM must have exactly six fields; observed {len(fields)} "
                    f"at line {line_number}."
                )
            sample_id = (fields[0], fields[1])
            if sample_id in seen:
                raise ValueError(
                    f"Reference FAM has a duplicate FID/IID at line {line_number}."
                )
            seen.add(sample_id)
            ids.append(sample_id)
            _update_id_digest(id_digest, *sample_id)
        final = os.fstat(handle.fileno())
    if not ids:
        raise ValueError(f"Reference FAM is empty: {path}")
    if not _unchanged(initial, final, observed_bytes):
        raise RuntimeError(f"Reference FAM changed while it was being read: {path}")
    return ids, {
        "path": str(path),
        "bytes": observed_bytes,
        "sha256": raw_digest.hexdigest(),
        "n_samples": len(ids),
        "ordered_id_digest": id_digest.hexdigest(),
        "id_unique": True,
    }


def _read_header_table(
    path: Path,
    root: Path,
    fam_ids: list[tuple[str, str]],
    *,
    label: str,
    exact_header: list[str] | None = None,
    environment: str | None = None,
) -> dict:
    handle, initial = _open_regular_binary(path, label)
    raw_digest = hashlib.sha256()
    id_digest = hashlib.sha256()
    observed_bytes = 0
    header: list[str] | None = None
    n_rows = 0
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            raw_digest.update(raw)
            observed_bytes += len(raw)
            fields = _decode_fields(raw, path, line_number)
            if line_number == 1:
                header = fields
                if len(set(header)) != len(header):
                    raise ValueError(f"{label} has duplicate header fields: {path}")
                if exact_header is not None and header != exact_header:
                    raise ValueError(
                        f"{label} header must be {exact_header!r}; observed {header!r}: {path}"
                    )
                if exact_header is None and header[:2] != ["FID", "IID"]:
                    raise ValueError(f"{label} must begin with FID IID columns: {path}")
                if environment is not None and environment not in header[2:]:
                    raise ValueError(
                        f"{label} lacks configured environment column {environment!r}: {path}"
                    )
                continue

            assert header is not None
            if len(fields) != len(header):
                raise ValueError(
                    f"{label} has {len(fields)} fields but header has {len(header)} "
                    f"at line {line_number}: {path}"
                )
            if n_rows >= len(fam_ids):
                raise ValueError(
                    f"{label} has more rows than the reference FAM: {path}"
                )
            sample_id = (fields[0], fields[1])
            if sample_id != fam_ids[n_rows]:
                raise ValueError(
                    f"{label} FID/IID order differs from the reference FAM at data row "
                    f"{n_rows + 1}: {path}"
                )
            _update_id_digest(id_digest, *sample_id)
            n_rows += 1
        final = os.fstat(handle.fileno())

    if header is None:
        raise ValueError(f"{label} is empty: {path}")
    if n_rows != len(fam_ids):
        raise ValueError(
            f"{label} row count {n_rows} differs from reference FAM N={len(fam_ids)}: {path}"
        )
    if not _unchanged(initial, final, observed_bytes):
        raise RuntimeError(f"{label} changed while it was being read: {path}")
    relative = path.relative_to(root)
    return {
        "path": str(path),
        "relative_path": relative.as_posix(),
        "bytes": observed_bytes,
        "sha256": raw_digest.hexdigest(),
        "header": header,
        "n_rows": n_rows,
        "ordered_id_digest": id_digest.hexdigest(),
        "id_unique": True,
        "order_matches_fam": True,
    }


def _resolve_below_root(root: Path, raw_path: str | Path, label: str) -> Path:
    supplied = Path(raw_path).expanduser()
    candidate = supplied if supplied.is_absolute() else root / supplied
    candidate = _absolute(candidate)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise ValueError(
            f"{label} is outside the required source root {root}: {candidate}"
        ) from None
    if not relative.parts:
        raise ValueError(
            f"{label} must be a file below the required source root: {candidate}"
        )

    current = root
    for part in relative.parts:
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {label}: {candidate}") from None
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{label} path contains a symlink: {current}")
    _require_regular_file(candidate, label)
    return candidate


def _validate_template(template: str) -> None:
    fields = []
    for _, field_name, format_spec, conversion in string.Formatter().parse(template):
        if field_name is not None:
            fields.append(field_name)
            if format_spec or conversion:
                raise ValueError(
                    "Phenotype template must not use conversions or format specs."
                )
    if fields != ["trait"]:
        raise ValueError("Phenotype template must contain exactly one {trait} field.")
    rendered = Path(template.format(trait="example_trait"))
    if rendered.is_absolute() or ".." in rendered.parts:
        raise ValueError(
            "Phenotype template must render a safe relative path below the source root."
        )


def _atomic_json_noreplace(payload: dict, target: Path) -> None:
    parent = target.parent
    _require_real_directory(parent, "manifest output directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite source manifest: {target}")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(
                f"Refusing to overwrite source manifest: {target}"
            ) from None
        published = True
        os.chmod(target, 0o600)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
                if error.errno not in unsupported:
                    raise
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
        if (
            not published
            and target.exists()
            and stat.S_IMODE(target.stat().st_mode) != 0o600
        ):
            raise RuntimeError(
                f"Unexpected manifest target appeared during publication: {target}"
            )


def generate_manifest(args: argparse.Namespace) -> Path:
    config_path = _absolute(args.config)
    config_bytes, config_record = _read_bytes_record(config_path, "panel configuration")
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid UTF-8 JSON panel configuration: {config_path}"
        ) from error
    if (
        config.get("kind") != "summit.gxe.hoffman_panel"
        or config.get("schema_version") != 1
    ):
        raise ValueError("Unsupported Hoffman panel configuration.")

    layout = config.get("local_source_layout")
    if not isinstance(layout, dict):
        raise ValueError("Panel configuration lacks required local_source_layout.")
    source_root = _absolute(args.source_root)
    _require_real_directory(source_root, "source root")
    configured_root_value = layout.get("root")
    if not isinstance(configured_root_value, str) or not configured_root_value:
        raise ValueError("local_source_layout.root must be a nonempty path string.")
    configured_root = _absolute(configured_root_value)
    if source_root != configured_root:
        raise ValueError(
            f"--source-root {source_root} differs from configured local source root "
            f"{configured_root}."
        )
    output = _absolute(args.output)
    if output == source_root or source_root in output.parents:
        raise ValueError(
            "Source manifest output must be outside the read-only source root."
        )
    template = layout.get("phenotype_path_template")
    if not isinstance(template, str):
        raise ValueError(
            "local_source_layout.phenotype_path_template must be a string."
        )
    _validate_template(template)
    phenotype_header = layout.get("phenotype_header")
    if phenotype_header != ["FID", "IID", "pheno"]:
        raise ValueError(
            "local_source_layout.phenotype_header must be exactly ['FID', 'IID', 'pheno']."
        )

    fam_path = _absolute(args.fam)
    fam_ids, fam_record = _read_fam(fam_path)
    full_dataset = config.get("datasets", {}).get("full", {})
    expected_fam_hash = full_dataset.get("expected_fam_sha256")
    expected_n = full_dataset.get("expected_n_samples")
    if (
        not isinstance(expected_fam_hash, str)
        or fam_record["sha256"] != expected_fam_hash
    ):
        raise ValueError(
            "Reference FAM SHA256 differs from the configured full-cohort FAM hash."
        )
    if not isinstance(expected_n, int) or fam_record["n_samples"] != expected_n:
        raise ValueError(
            "Reference FAM N differs from the configured full-cohort sample count."
        )

    groups = config.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Panel configuration must contain at least one group.")
    group_records: dict[str, dict] = {}
    covariate_records: dict[str, dict] = {}
    trait_names: set[str] = set()
    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not SAFE_NAME.fullmatch(group_name):
            raise ValueError(
                "Panel group names must use safe alphanumeric/underscore labels."
            )
        if not isinstance(group, dict):
            raise ValueError(f"Invalid group configuration: {group_name!r}")
        environment = group.get("environment")
        phenotypes = group.get("phenotypes")
        covariate_source = group.get("covariate_source_local")
        if not isinstance(environment, str) or not SAFE_NAME.fullmatch(environment):
            raise ValueError(f"Group {group_name!r} lacks an environment name.")
        if not isinstance(phenotypes, list) or not phenotypes:
            raise ValueError(f"Group {group_name!r} lacks phenotypes.")
        if not all(
            isinstance(trait, str) and SAFE_NAME.fullmatch(trait)
            for trait in phenotypes
        ):
            raise ValueError(
                f"Group {group_name!r} has invalid or duplicate phenotype labels."
            )
        if len(set(phenotypes)) != len(phenotypes):
            raise ValueError(
                f"Group {group_name!r} has invalid or duplicate phenotype labels."
            )
        if not isinstance(covariate_source, str) or not covariate_source:
            raise ValueError(f"Group {group_name!r} lacks covariate_source_local.")
        covariate_path = _resolve_below_root(
            source_root, covariate_source, f"covariate source for group {group_name!r}"
        )
        covariate_records[group_name] = _read_header_table(
            covariate_path,
            source_root,
            fam_ids,
            label=f"covariate source for group {group_name!r}",
            environment=environment,
        )
        trait_names.update(phenotypes)
        group_records[group_name] = {
            "environment": environment,
            "covariate_record": group_name,
            "phenotypes": phenotypes,
        }

    phenotype_records: dict[str, dict] = {}
    for trait in sorted(trait_names):
        rendered = template.format(trait=trait)
        phenotype_path = _resolve_below_root(
            source_root, rendered, f"phenotype source for trait {trait!r}"
        )
        phenotype_records[trait] = _read_header_table(
            phenotype_path,
            source_root,
            fam_ids,
            label=f"phenotype source for trait {trait!r}",
            exact_header=phenotype_header,
        )

    payload = {
        "kind": "summit.gxe.local_source_manifest",
        "schema_version": 1,
        "config": config_record,
        "source_root": str(source_root),
        "phenotype_path_template": template,
        "reference_fam": fam_record,
        "fam_order_digest": fam_record["ordered_id_digest"],
        "counts": {
            "groups": len(group_records),
            "unique_phenotype_files": len(phenotype_records),
            "configured_covariate_files": len(covariate_records),
        },
        "groups": group_records,
        "sources": {
            "phenotypes": phenotype_records,
            "covariates": covariate_records,
        },
    }
    _atomic_json_noreplace(payload, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Hoffman panel_config.json.")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Existing local phenotype/covariate root; must equal the configured root.",
    )
    parser.add_argument(
        "--fam", required=True, help="Exact local full-cohort reference FAM."
    )
    parser.add_argument(
        "--output", required=True, help="New JSON manifest outside source root."
    )
    return parser


def main() -> None:
    previous_umask = os.umask(0o077)
    try:
        output = generate_manifest(build_parser().parse_args())
        with output.open("rt", encoding="utf-8") as handle:
            counts = json.load(handle)["counts"]
        print(
            f"Recorded {counts['unique_phenotype_files']} phenotype and "
            f"{counts['configured_covariate_files']} covariate source files; manifest={output}"
        )
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
