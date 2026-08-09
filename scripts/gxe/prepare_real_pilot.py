#!/usr/bin/env python3
"""Create restrictive, FAM-aligned copies of controlled GxE pilot inputs.

The script never modifies source files and never prints sample identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ids(path: Path, *, header: bool) -> pd.DataFrame:
    if header:
        frame = pd.read_csv(path, sep=r"\s+", dtype={"FID": str, "IID": str})
        if "FID" not in frame or "IID" not in frame:
            raise ValueError(f"{path} must contain FID and IID columns.")
    else:
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            usecols=[0, 1],
            names=["FID", "IID"],
            dtype={0: str, 1: str},
        )
    if frame.duplicated(["FID", "IID"]).any():
        raise ValueError(f"{path} contains duplicate FID/IID pairs.")
    return frame


def _align(fam: pd.DataFrame, source: pd.DataFrame, label: str) -> pd.DataFrame:
    aligned = fam.merge(source, on=["FID", "IID"], how="left", indicator=True, validate="one_to_one")
    missing = int((aligned["_merge"] != "both").sum())
    if missing:
        raise ValueError(f"{label} is missing {missing} FAM samples.")
    return aligned.drop(columns="_merge")


def _atomic_table(frame: pd.DataFrame, target: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, sep="\t", index=False, na_rep="NA")
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(payload: dict, target: Path) -> None:
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
    parser.add_argument("--geno-prefix", required=True)
    parser.add_argument("--pheno", required=True)
    parser.add_argument("--covar", required=True)
    parser.add_argument("--env-source", required=True, help="Path containing FID, IID, and the environment column.")
    parser.add_argument("--env-column", required=True)
    parser.add_argument("--pheno-column", default="pheno")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    prefix = Path(args.geno_prefix).resolve()
    fam_path = Path(str(prefix) + ".fam")
    bim_path = Path(str(prefix) + ".bim")
    bed_path = Path(str(prefix) + ".bed")
    source_paths = {
        "fam": fam_path,
        "bim": bim_path,
        "bed": bed_path,
        "pheno": Path(args.pheno).resolve(),
        "covar": Path(args.covar).resolve(),
        "environment": Path(args.env_source).resolve(),
    }
    absent = [str(path) for path in source_paths.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing source files: {absent}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    outputs = {
        "pheno": out_dir / f"{args.label}.pheno.tsv",
        "covar": out_dir / f"{args.label}.covar.tsv",
        "environment": out_dir / f"{args.label}.env.tsv",
        "manifest": out_dir / f"{args.label}.provenance.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing pilot outputs: {existing}")

    fam = _read_ids(fam_path, header=False)
    with open(bim_path, "rb") as handle:
        bim_rows = sum(1 for _ in handle)
    with open(bed_path, "rb") as handle:
        bed_magic = handle.read(3)
    expected_bed_bytes = 3 + ((len(fam) + 3) // 4) * bim_rows
    if bed_magic != b"\x6c\x1b\x01" or bed_path.stat().st_size != expected_bed_bytes:
        raise ValueError(
            "PLINK BED/FAM/BIM shape validation failed: "
            f"N={len(fam)}, M={bim_rows}, expected BED bytes={expected_bed_bytes}, "
            f"observed={bed_path.stat().st_size}, magic={bed_magic.hex()}."
        )
    pheno = _read_ids(source_paths["pheno"], header=True)
    covar = _read_ids(source_paths["covar"], header=True)
    env = _read_ids(source_paths["environment"], header=True)
    if args.pheno_column not in pheno.columns:
        raise ValueError(f"Phenotype column {args.pheno_column!r} is absent from {source_paths['pheno']}.")
    if args.env_column not in env.columns:
        raise ValueError(f"Environment column {args.env_column!r} is absent from {source_paths['environment']}.")

    pheno_out = _align(fam, pheno[["FID", "IID", args.pheno_column]], "phenotype")
    pheno_out = pheno_out.rename(columns={args.pheno_column: "PHENO"})
    env_out = _align(fam, env[["FID", "IID", args.env_column]], "environment")
    env_out = env_out.rename(columns={args.env_column: "ENV"})
    covar_value_cols = [c for c in covar.columns if c not in ("FID", "IID", args.env_column)]
    if not covar_value_cols:
        raise ValueError("No fixed-effect covariates remain after removing the tested environment.")
    covar_out = _align(fam, covar[["FID", "IID", *covar_value_cols]], "covariates")

    for key, frame in (("pheno", pheno_out), ("covar", covar_out), ("environment", env_out)):
        _atomic_table(frame, outputs[key])

    provenance = {
        "kind": "summit.gxe.controlled_pilot_inputs",
        "label": args.label,
        "n_fam_samples": int(len(fam)),
        "n_bim_variants": int(bim_rows),
        "expected_bed_bytes": int(expected_bed_bytes),
        "geno_prefix": str(prefix),
        "environment_column": args.env_column,
        "phenotype_column": args.pheno_column,
        "sources": {},
        "outputs": {},
    }
    for name, path in source_paths.items():
        record = {"path": str(path), "bytes": path.stat().st_size}
        # The 1+ GiB BED is shape-validated elsewhere and is intentionally not
        # re-hashed for every pilot preparation.
        if name != "bed":
            record["sha256"] = _sha256(path)
        provenance["sources"][name] = record
    for name, path in outputs.items():
        if name != "manifest":
            provenance["outputs"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _atomic_json(provenance, outputs["manifest"])
    print(
        f"Prepared {args.label}: {len(fam)} aligned samples, "
        f"{len(covar_value_cols)} covariates, outputs in {out_dir}."
    )


if __name__ == "__main__":
    main()
