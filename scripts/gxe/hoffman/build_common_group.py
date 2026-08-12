#!/usr/bin/env python3
"""Build a private, fixed common-cohort input bundle for one GxE group.

The script is deliberately genotype-free. It aligns covariates and every
phenotype to a PLINK FAM, fixes one complete-case cohort across the whole group,
and writes FAM-order tables with rows outside that cohort marked missing. It
never prints sample identifiers and refuses to reuse an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MISSING_VALUES = ("-9", "NA", "NaN", "nan", ".", "None", "null")
SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path != root and root in path.parents


def _require_private_root(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Scratch root must be a real directory: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o700:
        raise PermissionError(f"Scratch root must have mode 0700; observed {mode:04o}: {path}")


def _require_private_input(path: Path, scratch_root: Path) -> None:
    if not _is_within(path, scratch_root):
        raise ValueError(f"Input is outside the configured scratch root: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Input must be a regular non-symlink file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"Input must have mode 0600; observed {mode:04o}: {path}")


def _id_digest(frame: pd.DataFrame, mask: np.ndarray | None = None) -> str:
    selected = frame if mask is None else frame.loc[np.asarray(mask, dtype=bool)]
    digest = hashlib.sha256()
    for fid, iid in selected[["FID", "IID"]].itertuples(index=False, name=None):
        digest.update(str(fid).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(iid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_fam(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype={0: str, 1: str},
    )
    if frame.empty:
        raise ValueError(f"FAM is empty: {path}")
    if frame.duplicated(["FID", "IID"]).any():
        raise ValueError("FAM contains duplicate FID/IID pairs.")
    return frame


def _read_header_table(path: Path, missing_values: Iterable[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        dtype={"FID": str, "IID": str},
        na_values=list(missing_values),
        keep_default_na=True,
    )
    if "FID" not in frame.columns or "IID" not in frame.columns:
        raise ValueError(f"Input must contain FID and IID columns: {path}")
    if frame.duplicated(["FID", "IID"]).any():
        raise ValueError(f"Input contains duplicate FID/IID pairs: {path}")
    return frame


def _align_exact(
    fam: pd.DataFrame,
    source: pd.DataFrame,
    label: str,
    *,
    allow_source_superset: bool = False,
) -> pd.DataFrame:
    ids = ["FID", "IID"]
    aligned = fam.merge(source, on=ids, how="left", indicator=True, validate="one_to_one")
    missing = int((aligned["_merge"] != "both").sum())
    extra = int(
        (source[ids].merge(fam, on=ids, how="left", indicator=True)["_merge"] != "both").sum()
    )
    if missing or (extra and not allow_source_superset):
        raise ValueError(
            f"{label} ID set differs from the FAM: missing FAM rows={missing}, extra rows={extra}."
        )
    return aligned.drop(columns="_merge")


def _numeric_column(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"{label} contains {int(invalid.sum())} non-numeric non-missing value(s).")
    return numeric.astype(np.float64)


def _parse_phenotype(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--phenotype must have LABEL=PATH form.")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not SAFE_LABEL.fullmatch(label) or label in {"FID", "IID", "ENV"}:
        raise argparse.ArgumentTypeError(f"Unsafe or reserved phenotype label: {label!r}.")
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("Phenotype path must not be empty.")
    return label, Path(raw_path).expanduser().resolve()


def _effective_rank(
    env: pd.Series,
    covariates: pd.DataFrame,
    keep: np.ndarray,
    ddof: int,
) -> tuple[int, list[str], list[str]]:
    kept_covariates: list[str] = []
    constant_covariates: list[str] = []
    columns: list[np.ndarray] = []
    for name in covariates.columns:
        values = covariates.loc[keep, name].to_numpy(dtype=np.float64)
        centered = values - float(values.mean())
        sd = float(centered.std(ddof=ddof))
        if not np.isfinite(sd) or sd <= 0.0:
            constant_covariates.append(str(name))
            continue
        kept_covariates.append(str(name))
        columns.append(centered / sd)

    env_values = env.loc[keep].to_numpy(dtype=np.float64)
    env_centered = env_values - float(env_values.mean())
    env_sd = float(env_centered.std(ddof=ddof))
    if not np.isfinite(env_sd) or env_sd <= 0.0:
        raise ValueError("Environment is constant or invalid in the common cohort.")
    columns.append(env_centered / env_sd)

    design = np.column_stack(columns)
    singular = np.linalg.svd(design, full_matrices=False, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        rank = 0
    else:
        tolerance = max(
            1e-10,
            max(design.shape) * np.finfo(np.float64).eps * float(singular[0]),
        )
        rank = int(np.sum(singular > tolerance))
    return rank, kept_covariates, constant_covariates


def _atomic_table(frame: pd.DataFrame, target: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        os.chmod(temporary, 0o600)
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
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def build_group(args: argparse.Namespace) -> Path:
    if not SAFE_LABEL.fullmatch(args.label):
        raise ValueError("--label must match ^[A-Za-z][A-Za-z0-9_]*$.")
    if args.ddof not in (0, 1):
        raise ValueError("--ddof must be 0 or 1.")
    if not args.phenotype:
        raise ValueError("At least one --phenotype LABEL=PATH is required.")
    phenotype_specs = [_parse_phenotype(x) for x in args.phenotype]
    labels = [label for label, _ in phenotype_specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Phenotype labels must be unique within a group.")

    fam_path = Path(args.fam).expanduser().resolve()
    covar_path = Path(args.covar).expanduser().resolve()
    source_paths = [fam_path, covar_path, *(path for _, path in phenotype_specs)]
    scratch_root = Path(args.scratch_root).expanduser().resolve()
    _require_private_root(scratch_root)
    missing_paths = [str(path) for path in source_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing input file(s): {missing_paths}")
    for path in source_paths:
        _require_private_input(path, scratch_root)

    missing_values = tuple(x.strip() for x in args.missing_values.split(",") if x.strip())
    fam = _read_fam(fam_path)
    covar_source = _read_header_table(covar_path, missing_values)
    covar_aligned = _align_exact(
        fam,
        covar_source,
        "Covariate source",
        allow_source_superset=args.allow_source_superset,
    )
    if args.environment_column not in covar_aligned.columns:
        raise ValueError(
            f"Environment column {args.environment_column!r} is absent from {covar_path}."
        )
    covariate_columns = [
        name
        for name in covar_aligned.columns
        if name not in {"FID", "IID", args.environment_column}
    ]
    if not covariate_columns:
        raise ValueError("No user covariates remain after removing the environment column.")

    environment = _numeric_column(
        covar_aligned[args.environment_column], f"Environment {args.environment_column!r}"
    )
    numeric_covariates = pd.DataFrame(index=fam.index)
    for name in covariate_columns:
        numeric_covariates[name] = _numeric_column(covar_aligned[name], f"Covariate {name!r}")

    phenotypes = pd.DataFrame(index=fam.index)
    phenotype_sources: list[dict] = []
    for label, path in phenotype_specs:
        source = _read_header_table(path, missing_values)
        aligned = _align_exact(
            fam,
            source,
            f"Phenotype {label!r}",
            allow_source_superset=args.allow_source_superset,
        )
        if args.phenotype_column not in aligned.columns:
            raise ValueError(
                f"Phenotype column {args.phenotype_column!r} is absent from {path}."
            )
        phenotypes[label] = _numeric_column(
            aligned[args.phenotype_column], f"Phenotype {label!r}"
        )
        phenotype_sources.append(
            {
                "label": label,
                "column": args.phenotype_column,
                **_file_record(path),
            }
        )

    # pandas 3 may return a read-only view; the complete-case mask is refined
    # in place below.
    keep = environment.notna().to_numpy(copy=True)
    keep &= ~numeric_covariates.isna().any(axis=1).to_numpy()
    keep &= ~phenotypes.isna().any(axis=1).to_numpy()
    n_selected = int(keep.sum())
    if n_selected < 3:
        raise ValueError(f"Common complete-case cohort is too small: N={n_selected}.")

    rank, kept_covariates, constant_covariates = _effective_rank(
        environment, numeric_covariates, keep, args.ddof
    )
    residual_rank = n_selected - rank - 1
    if residual_rank <= 0:
        raise ValueError(
            f"Non-positive residual rank: N={n_selected}, fixed-effect rank={rank}, intercept=1."
        )

    env_values = environment.loc[keep].to_numpy(dtype=np.float64)
    env_summary = {
        "mean": float(env_values.mean()),
        "sd": float(env_values.std(ddof=args.ddof)),
        "minimum": float(env_values.min()),
        "maximum": float(env_values.max()),
        "unique_count": int(np.unique(env_values).size),
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not _is_within(out_dir, scratch_root):
        raise ValueError(f"Output directory must be below the configured scratch root: {out_dir}")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {out_dir}")
    if not out_dir.parent.is_dir():
        raise FileNotFoundError(f"Output parent does not exist: {out_dir.parent}")
    parent_mode = stat.S_IMODE(out_dir.parent.stat().st_mode)
    if out_dir.parent.is_symlink() or parent_mode != 0o700:
        raise PermissionError(
            f"Output parent must be a real mode-0700 directory; observed {parent_mode:04o}: "
            f"{out_dir.parent}"
        )
    os.mkdir(out_dir, mode=0o700)
    os.chmod(out_dir, 0o700)

    env_out = fam.copy()
    env_out["ENV"] = environment
    covar_out = pd.concat([fam, numeric_covariates], axis=1)
    phenotype_out = pd.concat([fam, phenotypes], axis=1)
    outside = ~keep
    env_out.loc[outside, "ENV"] = np.nan
    covar_out.loc[outside, covariate_columns] = np.nan
    phenotype_out.loc[outside, labels] = np.nan

    outputs = {
        "environment": out_dir / f"{args.label}.env.tsv",
        "covariates": out_dir / f"{args.label}.covar.tsv",
        "phenotypes": out_dir / f"{args.label}.phenotypes.tsv",
        "manifest": out_dir / f"{args.label}.group.json",
    }
    _atomic_table(env_out, outputs["environment"])
    _atomic_table(covar_out, outputs["covariates"])
    _atomic_table(phenotype_out, outputs["phenotypes"])

    output_records = {
        name: {
            "path": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for name, path in outputs.items()
        if name != "manifest"
    }
    payload = {
        "kind": "summit.gxe.common_cohort_group",
        "schema_version": 1,
        "label": args.label,
        "source_id_contract": (
            "fam_subset_of_each_source" if args.allow_source_superset else "exact_fam_id_set"
        ),
        "complete_case_rule": "environment AND every user covariate AND every group phenotype",
        "n_fam_samples": int(len(fam)),
        "n_selected_samples": n_selected,
        "fam_order_digest": _id_digest(fam),
        "selected_id_digest": _id_digest(fam, keep),
        "environment": {
            "source_column": args.environment_column,
            "output_column": "ENV",
            "ddof": int(args.ddof),
            **env_summary,
        },
        "fixed_effect_rank_excluding_intercept": rank,
        "residual_rank": residual_rank,
        "covariate_columns": covariate_columns,
        "kept_nonconstant_covariates": kept_covariates,
        "constant_covariates": constant_covariates,
        "phenotype_labels": labels,
        "missing_counts_before_intersection": {
            "environment": int(environment.isna().sum()),
            "any_covariate": int(numeric_covariates.isna().any(axis=1).sum()),
            "phenotypes": {name: int(phenotypes[name].isna().sum()) for name in labels},
        },
        "sources": {
            "fam": _file_record(fam_path),
            "covariates_and_environment": _file_record(covar_path),
            "phenotypes": phenotype_sources,
        },
        "outputs": output_records,
    }
    _atomic_json(payload, outputs["manifest"])
    return outputs["manifest"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True, help="Existing private mode-0700 staging root.")
    parser.add_argument("--fam", required=True, help="FAM from the staged PLINK triple.")
    parser.add_argument("--covar", required=True, help="FID/IID covariate table containing the environment.")
    parser.add_argument("--environment-column", required=True)
    parser.add_argument(
        "--phenotype",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat once per phenotype in the common-cohort group.",
    )
    parser.add_argument("--phenotype-column", default="pheno")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True, help="Must be a new leaf under a private staging root.")
    parser.add_argument("--missing-values", default=",".join(DEFAULT_MISSING_VALUES))
    parser.add_argument("--ddof", type=int, default=1)
    parser.add_argument(
        "--allow-source-superset",
        action="store_true",
        default=False,
        help=(
            "Allow phenotype/covariate tables to contain IDs outside the FAM. "
            "Use only for the configured 50k calibration FAM; every FAM ID is still required."
        ),
    )
    return parser


def main() -> None:
    previous_umask = os.umask(0o077)
    try:
        args = build_parser().parse_args()
        manifest = build_group(args)
        with manifest.open("rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        print(
            f"Prepared group {payload['label']}: N={payload['n_selected_samples']}, "
            f"rank={payload['fixed_effect_rank_excluding_intercept']}, "
            f"phenotypes={len(payload['phenotype_labels'])}; manifest={manifest}"
        )
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
