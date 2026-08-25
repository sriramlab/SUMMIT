"""Reusable phenotype scoring against a SUMMIT GxE reference.

This module deliberately does not regenerate randomized trace panels.  It
validates that the supplied individual-level inputs reproduce the exact sample,
environment, fixed-effect, genotype, variant, and feature definition sealed by
an existing reference manifest, then makes one genotype decode pass to produce
the phenotype-dependent sufficient statistics.
"""

from __future__ import annotations

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
    _population_same_individual_products,
    _validate_reference_design_diagnostics,
)
from .gwe_ldscore import (
    _canonical_bfile_prefix,
    _native_strict_feature_moment_verification_policy,
    _validate_gxe_annotation_names,
    _validate_native_blas_runtime,
    _validate_plink_bed_shape,
    read_env_and_cov,
)


_REFERENCE_KIND = "summit.gxe.reference"
_MOMENTS_KIND = "summit.gxe.phenotype_moments"
_SUPPORTED_REFERENCE_SCHEMA_VERSIONS = frozenset({4})
_SCORE_DEFINITION = "feature_transpose_residualized_y_over_sqrt_residual_rank"
_SCORE_MODE = "marginal_cross_product"
_GENOTYPE_EXTENSIONS = (".bed", ".bim", ".fam")
_REFERENCE_ARTIFACTS = frozenset({"xx", "xw", "wx", "ww", "diagonal"})
_MAX_IN_MEMORY_SCORE_BYTES = 8 * 1024**3
_MAX_WIDE_WORKING_BYTES = 12 * 1024**3
_NATIVE_SCORE_WORKSPACE_BYTES = 16 * 1024**3


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
    diagonal: pd.DataFrame
    variants: pd.DataFrame
    scale_x: np.ndarray
    scale_w: np.ndarray
    norm_x: np.ndarray
    norm_w: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest file paths must be non-empty strings.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _as_finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values.")
    return array


def _read_bim(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        dtype={"CHR": str, "SNP": str, "A1": str, "A2": str},
    )


def _read_diagonal(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=r"\s+",
        compression="infer",
        dtype={"CHR": str, "SNP": str, "A1": str, "A2": str},
    )


def _validate_reference_manifest(
    reference_manifest: str | Path,
) -> _ValidatedReference:
    path = Path(reference_manifest).expanduser().resolve()
    payload = _load_json(path)
    if payload.get("kind") != _REFERENCE_KIND or payload.get("schema_version") != 4:
        raise ValueError("Only the current schema-v4 SUMMIT GxE reference is supported.")
    if payload.get("annotation_value_dtype") != "float64":
        raise ValueError("Reference annotations must use float64.")
    feature_convention = payload.get("feature_convention")
    if feature_convention not in {"standardized_projected", "raw_projected"}:
        raise ValueError("Reference has an unsupported feature convention.")
    if payload.get("kernel_mode") != feature_convention:
        raise ValueError("Reference kernel_mode must equal its feature convention.")
    if payload.get("genotype_scale") not in {"sample", "hwe"}:
        raise ValueError("Reference has an unsupported genotype scale.")
    if payload.get("ld_scale") != "cross_product_over_rank_squared":
        raise ValueError("Reference has an unsupported LD-score scale.")
    if payload.get("null_corrected") is not False:
        raise ValueError("Reference must contain raw, non-offset per-SNP scores.")
    for key in ("n_samples", "fixed_effect_rank_excluding_intercept", "residual_rank"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Reference {key} must be a JSON integer.")
    n = int(payload["n_samples"])
    fixed_rank = int(payload["fixed_effect_rank_excluding_intercept"])
    residual_rank = int(payload["residual_rank"])
    if n <= 0 or fixed_rank < 0 or residual_rank != n - fixed_rank - 1:
        raise ValueError("Reference sample size and fixed-effect rank are inconsistent.")
    transform = payload.get("environment_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Reference is missing environment_transform.")
    if (
        transform.get("standardized") is not True
        or transform.get("units") != "per_environment_sd"
        or transform.get("ddof") not in (0, 1)
    ):
        raise ValueError("Reference environment transform is unsupported.")
    for key in ("raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares"):
        value = transform.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Reference environment_transform[{key!r}] is invalid.")
    if float(transform["raw_sd"]) <= 0.0:
        raise ValueError("Reference environment standard deviation must be positive.")
    if not isinstance(payload.get("environment"), str) or not payload["environment"]:
        raise ValueError("Reference environment must be a non-empty string.")
    covariates = payload.get("covariates")
    if not isinstance(covariates, list) or any(
        not isinstance(name, str) or not name for name in covariates
    ):
        raise ValueError("Reference covariates must be a string list.")
    names_raw = payload.get("annotation_names")
    if not isinstance(names_raw, list):
        raise ValueError("Reference annotation_names must be a JSON list.")
    names = tuple(_validate_gxe_annotation_names(names_raw))
    files = payload.get("files")
    expected_files = {"xx", "xw", "wx", "ww", "diagonal"}
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise ValueError(f"Reference files must contain exactly {sorted(expected_files)}.")
    for key in ("xx", "xw", "wx", "ww"):
        artifact = _resolve_path(path, files[key])
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    diagonal = _read_diagonal(_resolve_path(path, files["diagonal"]))
    weight_columns = [f"ANNOT_{index}" for index in range(len(names))]
    expected_columns = [
        "CHR", "SNP", "BP", "A1", "A2",
        "NORM_X", "NORM_W", "SCALE_X", "SCALE_W",
        "DNXE_X", "DNXE_W", "CORR_XW", *weight_columns,
    ]
    observed_columns = diagonal.columns.astype(str).tolist()
    if observed_columns != expected_columns or len(set(observed_columns)) != len(observed_columns):
        raise ValueError(
            f"Reference diagonal must contain exactly the ordered columns {expected_columns}."
        )
    if len(diagonal) < 1 or diagonal["SNP"].duplicated().any():
        raise ValueError("Reference diagonal is empty or has duplicate SNP IDs.")
    annotations = diagonal.loc[:, weight_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(annotations)) or np.any(annotations < 0.0):
        raise ValueError("Reference annotations must be finite and non-negative.")
    masses = annotations.sum(axis=0, dtype=np.float64)
    declared_masses = _as_finite_vector(
        "reference annotation_masses", payload.get("annotation_masses"), len(names)
    )
    if np.any(masses <= 0.0) or not np.allclose(
        masses, declared_masses, rtol=5.0e-10, atol=1.0e-8
    ):
        raise ValueError("Reference annotation masses disagree with per-SNP rows.")
    m = len(diagonal)
    scale_x = _as_finite_vector("SCALE_X", diagonal["SCALE_X"], m)
    scale_w = _as_finite_vector("SCALE_W", diagonal["SCALE_W"], m)
    norm_x = _as_finite_vector("NORM_X", diagonal["NORM_X"], m)
    norm_w = _as_finite_vector("NORM_W", diagonal["NORM_W"], m)
    if np.any(scale_x <= 0.0) or np.any(scale_w <= 0.0):
        raise ValueError("Reference feature scales must be positive.")
    if np.any(norm_x <= 0.0) or np.any(norm_w <= 0.0):
        raise ValueError("Reference feature norms must be positive.")
    corr = _as_finite_vector("CORR_XW", diagonal["CORR_XW"], m)
    if np.any(np.abs(corr) > np.sqrt(norm_x * norm_w) + 1.0e-8):
        raise ValueError("Reference additive/interaction diagonals violate Cauchy-Schwarz.")
    if feature_convention == "standardized_projected":
        if not (
            np.allclose(norm_x, 1.0, rtol=1.0e-9, atol=1.0e-9)
            and np.allclose(norm_w, 1.0, rtol=1.0e-9, atol=1.0e-9)
        ):
            raise ValueError("Standardized reference violates its unit-norm contract.")
    elif not (
        np.allclose(scale_x, 1.0, rtol=0.0, atol=1.0e-12)
        and np.allclose(scale_w, 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Raw projected reference must store unit feature scales.")
    _validate_reference_design_diagnostics(
        payload, diagonal, annotations, masses, len(names), residual_rank
    )
    variants = diagonal.loc[:, ["CHR", "SNP", "BP", "A1", "A2"]].copy()
    return _ValidatedReference(
        path=path,
        payload=payload,
        diagonal=diagonal,
        variants=variants,
        scale_x=scale_x,
        scale_w=scale_w,
        norm_x=norm_x,
        norm_w=norm_w,
    )


def _validate_genotype_files(
    prefix: str,
    reference: _ValidatedReference,
    *,
    exact_reference: bool = True,
) -> tuple[int, int]:
    del exact_reference
    n, m = _validate_plink_bed_shape(prefix)
    if m != len(reference.diagonal):
        raise ValueError(
            f"Genotype BIM has {m} variants but the reference has {len(reference.diagonal)}."
        )
    if n < 1:
        raise ValueError("Genotype input contains no samples.")
    return n, m

@contextmanager
def _stable_genotype_prefix(prefix: str, staging_dir: Path):
    """Expose one-open-descriptor PLINK inputs through private /proc links.

    This neither copies the production BED nor changes source inode metadata.
    Path replacement cannot redirect an already-open descriptor, while final
    final fstat detects in-place mutation during the scoring pass.
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


def _validate_design_against_reference(
    *,
    reference: _ValidatedReference,
    environment_name: str,
    fixed_basis: np.ndarray,
    row_selection: np.ndarray,
    covariate_names: Sequence[str],
    observed_transform: dict[str, Any],
    exact_reference: bool = True,
) -> None:
    if environment_name != reference.payload["environment"]:
        raise ValueError("Environment column name disagrees with the reference manifest.")
    if list(covariate_names) != list(reference.payload["covariates"]):
        raise ValueError("Covariate columns disagree with the reference manifest.")

    n = len(row_selection)
    rank = int(fixed_basis.shape[1])
    residual_rank = n - rank - 1
    if residual_rank <= 0:
        raise ValueError(
            "The phenotype/environment-specific design has non-positive residual rank."
        )
    transform = reference.payload["environment_transform"]
    if (
        observed_transform.get("standardized") is not True
        or observed_transform.get("units") != "per_environment_sd"
        or int(observed_transform.get("ddof", -1)) != int(transform["ddof"])
    ):
        raise ValueError(
            "Study and reference environments must use the same per-SD/ddof convention."
        )
    if not exact_reference:
        return
    if n != int(reference.payload["n_samples"]):
        raise ValueError(
            f"Retained phenotype sample count {n} disagrees with reference N={reference.payload['n_samples']}."
        )
    if rank != int(reference.payload["fixed_effect_rank_excluding_intercept"]):
        raise ValueError("Fixed-effect numerical rank disagrees with the reference manifest.")
    if residual_rank != int(reference.payload["residual_rank"]):
        raise ValueError("Residual rank disagrees with the reference manifest.")

    for key in ("raw_mean", "raw_sd", "analysis_mean", "analysis_sum_squares"):
        if not np.isclose(
            float(observed_transform[key]),
            float(transform[key]),
            rtol=2.0e-12,
            atol=2.0e-12,
        ):
            raise ValueError(f"Environment transform disagrees with the reference for {key}.")


def _validate_analysis_inputs(
    *,
    prefix: str,
    reference: _ValidatedReference,
    env_path: str | Path,
    covar_path: str | Path | None,
    pheno_path: str | Path,
    pheno_col: str | None,
    missing_values: Sequence[str],
    exact_reference: bool = True,
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
        reference=reference,
        environment_name=str(environment_name),
        fixed_basis=np.asarray(fixed_basis, dtype=np.float64),
        row_selection=np.asarray(row_selection, dtype=int),
        covariate_names=covariate_names,
        observed_transform=observed_transform,
        exact_reference=exact_reference,
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
        reference=reference,
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


def _direct_native_score_module(reference: _ValidatedReference):
    """Return the guard-free API-v6 scorer when its runtime contract is met."""
    if (
        reference.payload.get("feature_convention") != "standardized_projected"
        or reference.payload.get("genotype_scale") != "sample"
    ):
        return None
    try:
        from .. import gxeldcore as native_module
    except (ImportError, OSError):
        return None
    build_info = dict(native_module.build_info())
    if (
        int(build_info.get("api_version", 0)) < 6
        or not callable(getattr(native_module.DirectContext, "phenotype_score_block", None))
    ):
        return None
    # This path deliberately promises no ABFT, repair, or retry overhead.  A
    # process-shared extension must keep using the compatibility scorer unless
    # its guarded runtime policy is selected explicitly elsewhere.
    if (
        build_info.get("blas_runtime_isolation") != "private_static"
        or build_info.get("gemm_integrity_enabled") is not False
        or build_info.get("gemm_execution_mode")
        != "serialized_fixed_private_openblas"
    ):
        return None
    return native_module


def _score_one_genotype_pass_native(
    *,
    native_module,
    reference: _ValidatedReference,
    genotype_descriptors: Mapping[str, int],
    row_selection: np.ndarray,
    environment: np.ndarray,
    fixed_basis: np.ndarray,
    phenotype_matrix: np.ndarray,
    step_size: int,
    eps_var: float,
    num_threads: int | None,
    residual_rank: int,
    exact_reference: bool,
    return_feature_nxe: bool,
    score_x: np.ndarray,
    score_w: np.ndarray,
    feature_nxe_x: np.ndarray | None,
    feature_nxe_w: np.ndarray | None,
) -> dict[str, Any]:
    initial_build_info = dict(native_module.build_info())
    native_threads = (
        int(num_threads)
        if num_threads is not None
        else int(initial_build_info["blas_runtime_threads"])
    )
    configured_threads = int(native_module.configure_blas_threads(native_threads))
    if configured_threads != native_threads:
        raise RuntimeError("The native GxE scorer configured an unexpected thread count.")
    build_info = dict(native_module.build_info())
    runtime_record = _validate_native_blas_runtime(build_info)
    strict_moments, integrity_reason = (
        _native_strict_feature_moment_verification_policy(build_info)
    )
    if strict_moments:
        raise RuntimeError(
            "The guard-free native phenotype scorer unexpectedly requested duplicate "
            "feature-moment verification."
        )

    n = len(row_selection)
    intercept = np.full((n, 1), 1.0 / math.sqrt(float(n)), dtype=np.float64)
    q_basis = np.asfortranarray(np.column_stack([intercept, fixed_basis]))
    expected_rank = fixed_basis.shape[1] + 1
    if q_basis.shape != (n, expected_rank) or not np.allclose(
        q_basis.T @ q_basis,
        np.eye(expected_rank),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise RuntimeError("The native phenotype scorer received a non-orthonormal design.")

    context = None
    max_leak_x = 0.0
    max_leak_w = 0.0
    max_phenotype_leakage = 0.0
    missing_calls = 0
    repaired_feature_columns = 0
    try:
        context = native_module.DirectContext(
            bed_descriptor=int(genotype_descriptors[".bed"]),
            bim_descriptor=int(genotype_descriptors[".bim"]),
            fam_descriptor=int(genotype_descriptors[".fam"]),
            row_sel=np.asarray(row_selection, dtype=np.int64),
            ddof=int(reference.payload["environment_transform"]["ddof"]),
            env=np.asarray(environment, dtype=np.float64),
            q_basis=q_basis,
            decode_threads=native_threads,
            blas_threads=native_threads,
            max_workspace_bytes=_NATIVE_SCORE_WORKSPACE_BYTES,
            target_panel_columns=max(1, int(phenotype_matrix.shape[1])),
            strict_feature_moment_verification=False,
        )
        context_info = dict(context.info())
        if (
            int(context_info["n_selected"]) != n
            or int(context_info["m_total"]) != len(reference.diagonal)
            or int(context_info["q_rank"]) != expected_rank
            or int(context_info["decode_threads"]) != native_threads
            or int(context_info["blas_threads"]) != native_threads
            or bool(context_info["strict_feature_moment_verification"])
        ):
            raise RuntimeError(
                "The native phenotype scorer disagrees with validated dimensions/state."
            )
        projected_phenotype = context.prepare_projected_sources(
            np.asfortranarray(phenotype_matrix, dtype=np.float64),
            tolerance=1.0e-10,
        )
        max_phenotype_leakage = float(projected_phenotype.leakage)
        root_rank = math.sqrt(float(residual_rank))
        m = len(reference.diagonal)
        for start in range(0, m, step_size):
            end = min(m, start + step_size)
            if exact_reference and not return_feature_nxe:
                block_score_x, block_score_w, block_missing, block_leakage = (
                    context.target_projected_block(
                        start,
                        end,
                        np.asarray(reference.scale_x[start:end], dtype=np.float64),
                        np.asarray(reference.scale_w[start:end], dtype=np.float64),
                        projected_phenotype,
                        False,
                    )
                )
                score_x[start:end, :] = (
                    np.asarray(block_score_x, dtype=np.float64) / root_rank
                )
                score_w[start:end, :] = (
                    np.asarray(block_score_w, dtype=np.float64) / root_rank
                )
                missing_calls += int(block_missing)
                max_phenotype_leakage = max(
                    max_phenotype_leakage, float(block_leakage)
                )
                continue

            block = dict(
                context.phenotype_score_block(
                    start,
                    end,
                    projected_phenotype,
                    eps_var,
                    False,
                )
            )
            exact_scale_x = np.asarray(block["scale_x"], dtype=np.float64)
            exact_scale_w = np.asarray(block["scale_w"], dtype=np.float64)
            observed_norm_x = np.asarray(block["norm_x"], dtype=np.float64)
            observed_norm_w = np.asarray(block["norm_w"], dtype=np.float64)
            if exact_reference and (
                not np.allclose(
                    exact_scale_x,
                    reference.scale_x[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-10,
                )
                or not np.allclose(
                    exact_scale_w,
                    reference.scale_w[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-10,
                )
            ):
                raise ValueError(
                    "Stored feature scales disagree with the supplied genotype/design "
                    f"in block [{start}:{end})."
                )
            if exact_reference and (
                not np.allclose(
                    observed_norm_x,
                    reference.norm_x[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
                or not np.allclose(
                    observed_norm_w,
                    reference.norm_w[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
            ):
                raise ValueError(
                    "Genotype scaling/projected feature norms disagree with the "
                    f"reference in block [{start}:{end})."
                )
            score_x[start:end, :] = (
                np.asarray(block["score_x"], dtype=np.float64) / root_rank
            )
            score_w[start:end, :] = (
                np.asarray(block["score_w"], dtype=np.float64) / root_rank
            )
            if return_feature_nxe:
                assert feature_nxe_x is not None and feature_nxe_w is not None
                feature_nxe_x[start:end] = np.asarray(
                    block["diag_nxe_x"], dtype=np.float64
                )
                feature_nxe_w[start:end] = np.asarray(
                    block["diag_nxe_w"], dtype=np.float64
                )
            max_leak_x = max(
                max_leak_x, float(block["max_projection_leakage_additive"])
            )
            max_leak_w = max(
                max_leak_w, float(block["max_projection_leakage_interaction"])
            )
            missing_calls += int(block["missing_genotype_calls"])
            repaired_feature_columns += int(block["repaired_feature_moment_columns"])

        final_context_info = dict(context.info())
        repaired_gemm_columns = int(final_context_info["repaired_gemm_output_columns"])
        retried_inputs = int(final_context_info["retried_gemm_input_mutations"])
        if repaired_feature_columns or repaired_gemm_columns or retried_inputs:
            raise RuntimeError(
                "The guard-free native phenotype scorer reported an impossible "
                "repair or retry event."
            )
    finally:
        if context is not None:
            context.close()

    return {
        "execution_mode": (
            "native_fused_sealed_scale_one_pass"
            if exact_reference and not return_feature_nxe
            else "native_fused_study_moment_one_pass"
        ),
        "genotype_passes": 1,
        "native_threads": native_threads,
        "native_workspace_cap_bytes": _NATIVE_SCORE_WORKSPACE_BYTES,
        "backend_version": str(build_info["backend_version"]),
        "api_version": int(build_info["api_version"]),
        "blas_runtime": runtime_record,
        "gemm_integrity_enabled": False,
        "strict_feature_moment_verification": False,
        "feature_moment_integrity_reason": integrity_reason,
        "repaired_feature_moment_columns": repaired_feature_columns,
        "repaired_gemm_output_columns": 0,
        "retried_gemm_input_mutations": 0,
        "missing_genotype_calls": missing_calls,
        "max_projection_leakage_additive": max_leak_x,
        "max_projection_leakage_interaction": max_leak_w,
        "phenotype_projection_leakage": max_phenotype_leakage,
    }


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
    residual_rank: int | None = None,
    exact_reference: bool = True,
    return_feature_nxe: bool = False,
    genotype_descriptors: Mapping[str, int] | None = None,
    backend_diagnostics: dict[str, Any] | None = None,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
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
    residual_rank = (
        int(reference.payload["residual_rank"])
        if residual_rank is None
        else int(residual_rank)
    )
    if residual_rank <= 0:
        raise ValueError("residual_rank must be positive for GxE scoring.")
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
    feature_nxe_x = (
        np.empty(m, dtype=np.float64) if return_feature_nxe else None
    )
    feature_nxe_w = (
        np.empty(m, dtype=np.float64) if return_feature_nxe else None
    )
    environment_squared = environment * environment if return_feature_nxe else None
    genotype_scale = str(reference.payload["genotype_scale"])
    ddof = int(reference.payload["environment_transform"]["ddof"])
    feature_convention = str(reference.payload["feature_convention"])

    native_module = _direct_native_score_module(reference)
    if native_module is not None and genotype_descriptors is not None:
        diagnostics = _score_one_genotype_pass_native(
            native_module=native_module,
            reference=reference,
            genotype_descriptors=genotype_descriptors,
            row_selection=row_selection,
            environment=environment,
            fixed_basis=fixed_basis,
            phenotype_matrix=phenotype_matrix,
            step_size=step_size,
            eps_var=eps_var,
            num_threads=num_threads,
            residual_rank=residual_rank,
            exact_reference=exact_reference,
            return_feature_nxe=return_feature_nxe,
            score_x=score_x,
            score_w=score_w,
            feature_nxe_x=feature_nxe_x,
            feature_nxe_w=feature_nxe_w,
        )
        if backend_diagnostics is not None:
            backend_diagnostics.update(diagnostics)
        if return_feature_nxe:
            assert feature_nxe_x is not None and feature_nxe_w is not None
            return (
                score_x[:, 0] if single_trait else score_x,
                score_w[:, 0] if single_trait else score_w,
                feature_nxe_x,
                feature_nxe_w,
            )
        return (
            score_x[:, 0] if single_trait else score_x,
            score_w[:, 0] if single_trait else score_w,
        )

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
            if feature_convention == "standardized_projected":
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
                if exact_reference and (
                    not np.allclose(
                        exact_scale_x,
                        reference.scale_x[start:end],
                        rtol=1.0e-9,
                        atol=1.0e-10,
                    )
                    or not np.allclose(
                        exact_scale_w,
                        reference.scale_w[start:end],
                        rtol=1.0e-9,
                        atol=1.0e-10,
                    )
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
            if exact_reference and (
                not np.allclose(
                    observed_norm_x,
                    reference.norm_x[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
                or not np.allclose(
                    observed_norm_w,
                    reference.norm_w[start:end],
                    rtol=1.0e-9,
                    atol=1.0e-9,
                )
            ):
                raise ValueError(
                    "Genotype scaling/projected feature norms disagree with the "
                    f"reference in block [{start}:{end})."
                )
            score_x[start:end, :] = (additive.T @ phenotype_matrix) / root_rank
            score_w[start:end, :] = (interaction.T @ phenotype_matrix) / root_rank
            if return_feature_nxe:
                assert feature_nxe_x is not None and feature_nxe_w is not None
                assert environment_squared is not None
                # Scores no longer need these feature blocks, so square them
                # in place and avoid allocating two additional N-by-block
                # temporaries for the exact genetic-by-NxE traces.
                np.square(additive, out=additive)
                np.square(interaction, out=interaction)
                feature_nxe_x[start:end] = (
                    environment_squared @ additive / float(residual_rank)
                )
                feature_nxe_w[start:end] = (
                    environment_squared @ interaction / float(residual_rank)
                )
    if backend_diagnostics is not None:
        backend_diagnostics.update(
            {
                "execution_mode": "python_numpy_one_pass",
                "genotype_passes": 1,
                "repaired_gemm_output_columns": 0,
                "retried_gemm_input_mutations": 0,
            }
        )
    if return_feature_nxe:
        assert feature_nxe_x is not None and feature_nxe_w is not None
        if not np.all(np.isfinite(feature_nxe_x)) or not np.all(
            np.isfinite(feature_nxe_w)
        ):
            raise RuntimeError("Study genetic-by-NxE feature moments are non-finite.")
    if single_trait:
        scores = (score_x[:, 0], score_w[:, 0])
    else:
        scores = (score_x, score_w)
    if return_feature_nxe:
        return scores[0], scores[1], feature_nxe_x, feature_nxe_w
    return scores


def _nxe_traces(
    environment: np.ndarray, fixed_basis: np.ndarray
) -> tuple[float, float]:
    """Return tr(P diag(E^2) P) and its squared-kernel trace exactly."""
    env = np.asarray(environment, dtype=np.float64)
    basis = np.asarray(fixed_basis, dtype=np.float64)
    if env.ndim != 1 or basis.ndim != 2 or basis.shape[0] != env.size:
        raise ValueError("Environment and fixed-effect basis are not sample aligned.")
    n = env.size
    intercept = np.full((n, 1), 1.0 / math.sqrt(float(n)), dtype=np.float64)
    q_basis = np.asfortranarray(np.column_stack([intercept, basis]))
    d = env * env
    qdq = q_basis.T @ (d[:, None] * q_basis)
    sum_d = float(np.sum(d, dtype=np.float64))
    sum_d2 = float(np.sum(d * d, dtype=np.float64))
    trace_qd2q = float(np.sum((d[:, None] * q_basis) ** 2, dtype=np.float64))
    trace_nxe = sum_d - float(np.trace(qdq))
    trace_nxe_sq = (
        sum_d2 - 2.0 * trace_qd2q + float(np.sum(qdq * qdq.T, dtype=np.float64))
    )
    tolerance = 2.0e-10 * max(1.0, sum_d, sum_d2)
    if trace_nxe < -tolerance or trace_nxe_sq < -tolerance:
        raise RuntimeError("Study NxE projection produced a negative trace.")
    return max(0.0, trace_nxe), max(0.0, trace_nxe_sq)


def _population_design_moments(
    reference: _ValidatedReference,
    feature_nxe_x: np.ndarray,
    feature_nxe_w: np.ndarray,
    residual_rank: int,
) -> dict[str, Any]:
    """Aggregate exact full-study genetic-by-NxE traces."""
    names = tuple(str(value) for value in reference.payload["annotation_names"])
    weight_columns = [f"ANNOT_{index}" for index in range(len(names))]
    annotations = reference.diagonal.loc[:, weight_columns].to_numpy(
        dtype=np.float64
    )
    masses = annotations.sum(axis=0, dtype=np.float64)
    x = np.asarray(feature_nxe_x, dtype=np.float64)
    w = np.asarray(feature_nxe_w, dtype=np.float64)
    if x.shape != (len(annotations),) or w.shape != (len(annotations),):
        raise RuntimeError("Study genetic-by-NxE vectors are not variant aligned.")
    full_sums = np.concatenate([annotations.T @ x, annotations.T @ w])
    full_masses = np.concatenate([masses, masses])
    full = float(residual_rank) * full_sums / full_masses
    if not np.all(np.isfinite(full)) or np.any(full < 0.0):
        raise RuntimeError("Study genetic-by-NxE traces are invalid.")
    return {
        "method": "exact_projected_feature_nxe_v1",
        "feature_order": [
            *[f"G:{name}" for name in names],
            *[f"GxE:{name}" for name in names],
        ],
        "genetic_nxe_traces": full.tolist(),
    }


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
    population_transfer: bool = False,
) -> GxEPhenotypeScoreArtifacts:
    """Score one quantitative phenotype against an existing GxE reference.

    By default the supplied inputs must reproduce the reference's exact cohort.
    With ``population_transfer=True``, the SNP axis and feature convention stay
    fixed but the retained study cohort may differ; the reference must contain
    the population trace statistic required to transfer its kernel moments.
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
        reference = _validate_reference_manifest(reference_manifest)
        if population_transfer:
            if reference.payload.get("feature_convention") != "standardized_projected":
                raise ValueError(
                    "Population-reference scoring requires post-projection standardized kernels."
                )
            _population_same_individual_products(
                reference.payload,
                reference.payload["annotation_names"],
            )
        stable_prefix, stable_descriptors, stable_state = resources.enter_context(
            _stable_genotype_prefix(prefix, stage_dir)
        )
        total_samples, _ = _validate_genotype_files(
            stable_prefix,
            reference,
            exact_reference=not population_transfer,
        )
        _validate_variant_axis(stable_prefix, reference)
        (
            row_selection,
            environment,
            fixed_basis,
            phenotype,
            phenotype_name,
            residual_fraction,
            observed_transform,
        ) = _validate_analysis_inputs(
            prefix=stable_prefix,
            reference=reference,
            env_path=env_path,
            covar_path=covar_path,
            pheno_path=pheno_path,
            pheno_col=pheno_col,
            missing_values=tuple(str(value) for value in missing_values),
            exact_reference=not population_transfer,
        )
        study_n = int(len(row_selection))
        study_rank = int(study_n - fixed_basis.shape[1] - 1)
        score_backend: dict[str, Any] = {}
        score_result = _score_one_genotype_pass(
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
            residual_rank=study_rank,
            exact_reference=not population_transfer,
            return_feature_nxe=population_transfer,
            genotype_descriptors=stable_descriptors,
            backend_diagnostics=score_backend,
        )
        if population_transfer:
            score_x, score_w, feature_nxe_x, feature_nxe_w = score_result
            population_design = _population_design_moments(
                reference,
                feature_nxe_x,
                feature_nxe_w,
                study_rank,
            )
        else:
            score_x, score_w = score_result
            population_design = None
        # Recheck shape and descriptor state after the final decode so a
        # moving PLINK input cannot be published as one coherent score set.
        _validate_genotype_files(
            stable_prefix,
            reference,
            exact_reference=not population_transfer,
        )
        _assert_stable_genotype_snapshot(stable_descriptors, stable_state)

        gwas_target, gwis_target, moments_target = targets
        base = reference.variants.copy()
        base["N"] = study_n
        base["DF"] = study_rank
        base["SCORE_MODE"] = _SCORE_MODE
        gwas = base.copy()
        gwis = base.copy()
        gwas["SCORE"] = score_x
        gwis["SCORE"] = score_w
        if population_transfer:
            gwas["DNXE"] = feature_nxe_x
            gwis["DNXE"] = feature_nxe_w

        temporary_paths: list[Path] = []
        published: list[tuple[Path, int, int]] = []
        try:
            temporary_gwas = _write_dataframe_temp(gwas, gwas_target, stage_dir)
            temporary_gwis = _write_dataframe_temp(gwis, gwis_target, stage_dir)
            temporary_paths.extend([temporary_gwas, temporary_gwis])
            trace_nxe, trace_nxe_sq = _nxe_traces(environment, fixed_basis)
            moments = {
                "kind": _MOMENTS_KIND,
                "schema_version": int(reference.payload["schema_version"]),
                "phenotype": phenotype_name,
                "n_samples": study_n,
                "fixed_effect_rank_excluding_intercept": int(fixed_basis.shape[1]),
                "residual_rank": study_rank,
                "feature_convention": reference.payload["feature_convention"],
                "feature_convention_version": 1,
                "reference_mode": (
                    "population" if population_transfer else "matched"
                ),
                "score_definition": _SCORE_DEFINITION,
                "score_backend": score_backend,
                "q_nxe": float(np.dot(environment * phenotype, environment * phenotype)),
                "q_residual": float(np.dot(phenotype, phenotype)),
                "trace_nxe": trace_nxe,
                "trace_nxe_sq": trace_nxe_sq,
                "phenotype_residual_variance_fraction": residual_fraction,
                "files": {
                    "gwas": _relative_path(gwas_target, moments_target),
                    "gwis": _relative_path(gwis_target, moments_target),
                },
            }
            if population_design is not None:
                moments["population_design"] = population_design
            temporary_moments = _write_json_temp(moments, moments_target, stage_dir)
            temporary_paths.append(temporary_moments)
            for temporary, target in zip(temporary_paths, targets):
                if target == moments_target:
                    _verify_published_inodes(
                        published,
                        context="GxE phenotype-score",
                    )
                device, inode = _publish_private_no_replace(temporary, target)
                published.append((target, device, inode))
            _verify_published_inodes(
                published,
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
    score_backend: Mapping[str, Any],
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
                "schema_version": int(reference.payload["schema_version"]),
                "phenotype": trait,
                "n_samples": int(reference.payload["n_samples"]),
                "residual_rank": int(reference.payload["residual_rank"]),
                "feature_convention": reference.payload["feature_convention"],
                "feature_convention_version": 1,
                "score_definition": _SCORE_DEFINITION,
                "score_backend": dict(score_backend),
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
            temporary_moments = _write_json_temp(moments, moments_target, staging_dir)
            temporary_pairs.append((temporary_moments, moments_target))
            results[trait] = GxEPhenotypeScoreArtifacts(
                gwas=gwas_target, gwis=gwis_target, moments=moments_target
            )

        publication_order = [
            pair for pair in temporary_pairs if not pair[1].name.endswith(".gxe.moments.json")
        ] + [
            pair for pair in temporary_pairs if pair[1].name.endswith(".gxe.moments.json")
        ]
        for temporary, target in publication_order:
            if target.name.endswith(".gxe.moments.json"):
                _verify_published_inodes(
                    published,
                    context="wide GxE phenotype-score",
                )
            device, inode = _publish_private_no_replace(temporary, target)
            published.append((target, device, inode))
        _verify_published_inodes(
            published,
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

    # Trait names are needed to reserve every final path before genotype work.
    # Read only the header here; the validated full table is
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
        reference = _validate_reference_manifest(reference_manifest)
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
        score_backend: dict[str, Any] = {}
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
            genotype_descriptors=stable_descriptors,
            backend_diagnostics=score_backend,
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
            score_backend=score_backend,
            targets=targets,
            staging_dir=stage_dir,
        )

    return _with_wide_output_lock(output_prefix, targets, run)


__all__ = [
    "GxEPhenotypeScoreArtifacts",
    "score_phenotype_from_reference",
    "score_phenotypes_from_reference",
]
