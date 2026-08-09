#!/usr/bin/env python3
"""Create hash-bound manifests for an audited pre-schema SUMMIT GxE bundle.

This migration helper never changes the referenced artifacts and refuses to
overwrite its output manifests.  It does not certify scientific correctness;
use it only after independently auditing the legacy bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


REFERENCE_KIND = "summit.gxe.reference"
MOMENTS_KIND = "summit.gxe.phenotype_moments"


def _load(path: Path) -> dict:
    with open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_files(payload: dict, source_manifest: Path, output_manifest: Path) -> dict[str, Path]:
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{source_manifest} does not declare any artifacts.")
    resolved = {str(key): _resolve(source_manifest, str(value)) for key, value in files.items()}
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Legacy bundle references missing artifacts: {missing}.")
    payload["files"] = {
        key: os.path.relpath(path, start=output_manifest.parent)
        for key, path in resolved.items()
    }
    return resolved


def _atomic_json(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}.")
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--moments", required=True)
    parser.add_argument("--out-reference", required=True)
    parser.add_argument("--out-moments", required=True)
    parser.add_argument("--phenotype-residual-variance-fraction", type=float, default=None)
    parser.add_argument("--allow-low-probe-jackknife", action="store_true", default=False)
    args = parser.parse_args()

    reference_path = Path(args.reference).resolve()
    moments_path = Path(args.moments).resolve()
    output_reference = Path(args.out_reference).resolve()
    output_moments = Path(args.out_moments).resolve()
    if output_reference == reference_path or output_moments == moments_path:
        raise ValueError("Migration outputs must be new paths; legacy manifests are never changed in place.")
    existing = [str(path) for path in (output_reference, output_moments) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite migration output(s): {existing}.")

    reference = _load(reference_path)
    moments = _load(moments_path)
    if reference.get("kind") != REFERENCE_KIND or int(reference.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported legacy reference manifest: {reference_path}.")
    if moments.get("kind") != MOMENTS_KIND or int(moments.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported legacy moments manifest: {moments_path}.")
    for key in ("analysis_fingerprint", "variant_digest", "residual_rank"):
        if str(reference.get(key)) != str(moments.get(key)):
            raise ValueError(f"Reference and moments disagree on {key}; refusing to seal the bundle.")

    ref_files = _rewrite_files(reference, reference_path, output_reference)
    required_reference = {"xx", "xw", "wx", "ww", "diagonal"}
    if "jackknife" in ref_files:
        required_reference.add("jackknife")
    if not required_reference.issubset(ref_files):
        raise ValueError(f"Reference is missing required artifacts: {sorted(required_reference - set(ref_files))}.")
    if "jackknife" in required_reference:
        randomization = reference.get("randomization")
        if not isinstance(randomization, dict) or "num_vectors" not in randomization:
            raise ValueError("Legacy jackknife reference does not declare its random-probe count.")
        if int(randomization["num_vectors"]) < 100:
            if not args.allow_low_probe_jackknife:
                raise ValueError(
                    "Legacy jackknife uses fewer than 100 probes; pass --allow-low-probe-jackknife "
                    "only after auditing it as a diagnostic artifact."
                )
            randomization["low_probe_jackknife_override"] = True
    reference["artifact_sha256"] = {
        key: _sha256(ref_files[key]) for key in sorted(required_reference)
    }
    reference["schema_version"] = 2

    score_files = _rewrite_files(moments, moments_path, output_moments)
    if not {"gwas", "gwis"}.issubset(score_files):
        raise ValueError("Moments manifest must declare both gwas and gwis artifacts.")
    moments["score_sha256"] = {key: _sha256(score_files[key]) for key in ("gwas", "gwis")}
    moments["schema_version"] = 2
    if args.phenotype_residual_variance_fraction is not None:
        fraction = float(args.phenotype_residual_variance_fraction)
        if not 0.0 < fraction <= 1.0 + 1.0e-12:
            raise ValueError("Phenotype residual variance fraction must be in (0, 1].")
        moments["phenotype_residual_variance_fraction"] = fraction

    _atomic_json(reference, output_reference)
    moments["reference_manifest_sha256"] = _sha256(output_reference)
    _atomic_json(moments, output_moments)
    print(f"Wrote hash-bound manifests: {output_reference} and {output_moments}")


if __name__ == "__main__":
    main()
