#!/usr/bin/env python3
"""Fail-closed verification of private staged GxE genotype/group inputs.

This verifier is read-only except for one new private JSON report. It hashes
both the configured project source and staged PLINK triples, validates exact BED
shape/magic, checks FAM order without printing IDs, and verifies common-group
manifests and their output hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path != root and root in path.parents


def _require_private_file(path: Path, scratch_root: Path) -> None:
    if not _is_within(path, scratch_root):
        raise ValueError(f"Staged file is outside configured scratch root: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Staged input must be a regular non-symlink file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"Staged file must have mode 0600; observed {mode:04o}: {path}")


def _require_private_directory(path: Path, scratch_root: Path) -> None:
    if not _is_within(path, scratch_root):
        raise ValueError(f"Staged directory is outside configured scratch root: {path}")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Staged directory must be a real directory: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o700:
        raise PermissionError(f"Staged directory must have mode 0700; observed {mode:04o}: {path}")


def _line_count_no_blanks(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank row in {path} at line {line_number}.")
            count += 1
    return count


def _plink_shape(prefix: Path) -> dict:
    fam = Path(str(prefix) + ".fam")
    bim = Path(str(prefix) + ".bim")
    bed = Path(str(prefix) + ".bed")
    for path in (fam, bim, bed):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing regular PLINK input: {path}")
    n = _line_count_no_blanks(fam)
    m = _line_count_no_blanks(bim)
    expected_bytes = 3 + ((n + 3) // 4) * m
    with bed.open("rb") as handle:
        magic = handle.read(3).hex()
    observed_bytes = int(bed.stat().st_size)
    if magic != "6c1b01" or observed_bytes != expected_bytes:
        raise ValueError(
            f"Invalid SNP-major BED for {prefix}: N={n}, M={m}, expected={expected_bytes}, "
            f"observed={observed_bytes}, magic={magic}."
        )
    return {
        "n_samples": n,
        "n_variants": m,
        "bed_bytes": observed_bytes,
        "expected_bed_bytes": expected_bytes,
        "bed_magic": magic,
    }


def _fam_id_summary(path: Path) -> tuple[str, int]:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype={0: str, 1: str},
    )
    duplicate_count = int(frame.duplicated(["FID", "IID"]).sum())
    if duplicate_count:
        raise ValueError(f"FAM contains {duplicate_count} duplicate FID/IID pair(s).")
    digest = hashlib.sha256()
    for fid, iid in frame.itertuples(index=False, name=None):
        digest.update(str(fid).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(iid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(frame)


def _canonical_header_id_digest(path: Path) -> tuple[str, int]:
    frame = pd.read_csv(path, sep=r"\s+", usecols=["FID", "IID"], dtype=str)
    duplicate_count = int(frame.duplicated(["FID", "IID"]).sum())
    if duplicate_count:
        raise ValueError(f"Staged table contains {duplicate_count} duplicate FID/IID pair(s): {path}")
    digest = hashlib.sha256()
    for fid, iid in frame.itertuples(index=False, name=None):
        digest.update(str(fid).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(iid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(frame)


def _verify_group(
    manifest_path: Path,
    config: dict,
    dataset_name: str,
    fam_digest: str,
    n_fam: int,
    scratch_root: Path,
) -> dict:
    _require_private_file(manifest_path, scratch_root)
    _require_private_directory(manifest_path.parent, scratch_root)
    with manifest_path.open("rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("kind") != "summit.gxe.common_cohort_group" or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported group manifest: {manifest_path}")
    label = str(manifest.get("label"))
    group_config = config.get("groups", {}).get(label)
    if group_config is None:
        raise ValueError(f"Group {label!r} is absent from the configured panel.")
    if int(manifest.get("n_fam_samples", -1)) != n_fam:
        raise ValueError(f"Group {label!r} FAM N does not match the staged genotype.")
    if manifest.get("fam_order_digest") != fam_digest:
        raise ValueError(f"Group {label!r} FAM order digest does not match the staged genotype.")
    if manifest.get("environment", {}).get("source_column") != group_config.get("environment"):
        raise ValueError(f"Group {label!r} environment does not match panel configuration.")
    if list(manifest.get("phenotype_labels", [])) != list(group_config.get("phenotypes", [])):
        raise ValueError(f"Group {label!r} phenotype order does not match panel configuration.")
    expected_id_contract = (
        "exact_fam_id_set" if dataset_name == "full" else "fam_subset_of_each_source"
    )
    if manifest.get("source_id_contract") != expected_id_contract:
        raise ValueError(
            f"Group {label!r} source-ID contract {manifest.get('source_id_contract')!r} "
            f"does not match dataset {dataset_name!r} ({expected_id_contract!r})."
        )
    if dataset_name == "full":
        for manifest_key, config_key in (
            ("n_selected_samples", "expected_common_n_full"),
            ("fixed_effect_rank_excluding_intercept", "expected_rank_full"),
            ("residual_rank", "expected_residual_rank_full"),
        ):
            if int(manifest.get(manifest_key, -1)) != int(group_config.get(config_key, -2)):
                raise ValueError(
                    f"Group {label!r} {manifest_key}={manifest.get(manifest_key)!r} does not match "
                    f"configured full-cohort value {group_config.get(config_key)!r}."
                )

    verified_outputs = {}
    for name in ("environment", "covariates", "phenotypes"):
        record = manifest.get("outputs", {}).get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Group {label!r} lacks output record {name!r}.")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative output path in group {label!r}: {relative}")
        path = (manifest_path.parent / relative).resolve()
        _require_private_file(path, scratch_root)
        observed_sha = _sha256(path)
        observed_bytes = int(path.stat().st_size)
        if observed_sha != record.get("sha256") or observed_bytes != int(record.get("bytes", -1)):
            raise ValueError(f"Hash/size mismatch for group {label!r} output {name!r}.")
        table_digest, table_n = _canonical_header_id_digest(path)
        if table_digest != fam_digest or table_n != n_fam:
            raise ValueError(f"ID order mismatch for group {label!r} output {name!r}.")
        verified_outputs[name] = {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": observed_sha,
        }
    return {
        "label": label,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "n_selected_samples": int(manifest["n_selected_samples"]),
        "fixed_effect_rank_excluding_intercept": int(
            manifest["fixed_effect_rank_excluding_intercept"]
        ),
        "residual_rank": int(manifest["residual_rank"]),
        "selected_id_digest": manifest["selected_id_digest"],
        "outputs": verified_outputs,
    }


def _atomic_json(payload: dict, target: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("rt", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("kind") != "summit.gxe.hoffman_panel" or config.get("schema_version") != 1:
        raise ValueError("Unsupported Hoffman panel configuration.")
    dataset = config.get("datasets", {}).get(args.dataset)
    if dataset is None:
        raise ValueError(f"Unknown dataset {args.dataset!r}.")

    scratch_root = Path(config["scratch_root"]).expanduser().resolve()
    if not scratch_root.is_dir():
        raise FileNotFoundError(f"Configured scratch root does not exist: {scratch_root}")
    root_mode = stat.S_IMODE(scratch_root.stat().st_mode)
    if root_mode != 0o700:
        raise PermissionError(
            f"Configured scratch root must have mode 0700; observed {root_mode:04o}: {scratch_root}"
        )

    staged_prefix = Path(args.geno_prefix).expanduser().resolve()
    if not _is_within(staged_prefix, scratch_root):
        raise ValueError("Staged genotype prefix must be below the configured scratch root.")
    _require_private_directory(staged_prefix.parent, scratch_root)
    source_prefix = Path(dataset["source_prefix"]).expanduser().resolve()

    staged_shape = _plink_shape(staged_prefix)
    source_shape = _plink_shape(source_prefix)
    for key in ("n_samples", "n_variants", "bed_bytes", "bed_magic"):
        expected = dataset.get(f"expected_{key}")
        if expected is not None and staged_shape[key] != expected:
            raise ValueError(
                f"Staged {key}={staged_shape[key]!r}; configured expected value is {expected!r}."
            )
        if staged_shape[key] != source_shape[key]:
            raise ValueError(f"Staged/source PLINK {key} mismatch.")

    hashes = {"source": {}, "staged": {}}
    for extension in ("fam", "bim", "bed"):
        staged_path = Path(str(staged_prefix) + f".{extension}")
        source_path = Path(str(source_prefix) + f".{extension}")
        _require_private_file(staged_path, scratch_root)
        print(f"Hashing source and staged .{extension} files...", flush=True)
        hashes["source"][extension] = _sha256(source_path)
        hashes["staged"][extension] = _sha256(staged_path)
        if hashes["source"][extension] != hashes["staged"][extension]:
            raise ValueError(f"Staged .{extension} hash differs from configured project source.")
        configured_hash = dataset.get(f"expected_{extension}_sha256")
        if configured_hash is not None and hashes["staged"][extension] != configured_hash:
            raise ValueError(f"Staged .{extension} hash differs from configured expected hash.")

    fam_digest, fam_n = _fam_id_summary(Path(str(staged_prefix) + ".fam"))
    if fam_n != staged_shape["n_samples"]:
        raise RuntimeError("Internal FAM count mismatch.")
    group_records = [
        _verify_group(
            Path(path).expanduser().resolve(),
            config,
            args.dataset,
            fam_digest,
            fam_n,
            scratch_root,
        )
        for path in args.group_manifest
    ]
    labels = [record["label"] for record in group_records]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate group manifests were supplied.")

    report_path = Path(args.report).expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite verification report: {report_path}")
    if not _is_within(report_path, scratch_root):
        raise ValueError("Verification report must be below the configured scratch root.")
    _require_private_directory(report_path.parent, scratch_root)
    payload = {
        "kind": "summit.gxe.staged_input_verification",
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "dataset": args.dataset,
        "scratch_root": str(scratch_root),
        "staged_genotype_prefix": str(staged_prefix),
        "source_genotype_prefix": str(source_prefix),
        "plink_shape": staged_shape,
        "fam_order_digest": fam_digest,
        "hashes": hashes,
        "groups": group_records,
    }
    _atomic_json(payload, report_path)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=("full", "subset_50k"))
    parser.add_argument("--geno-prefix", required=True, help="Private staged PLINK prefix.")
    parser.add_argument("--group-manifest", action="append", default=[])
    parser.add_argument("--report", required=True, help="New JSON report path below scratch root.")
    return parser


def main() -> None:
    previous_umask = os.umask(0o077)
    try:
        report = verify(build_parser().parse_args())
        with report.open("rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        print(
            f"Verified dataset {payload['dataset']}: N={payload['plink_shape']['n_samples']}, "
            f"M={payload['plink_shape']['n_variants']}, groups={len(payload['groups'])}; "
            f"report={report}"
        )
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
