"""Input and output functions shared by the manuscript simulators."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import open_bed


def read_table(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        first = next((line.split() for line in handle if line.strip()), None)
    if first is None:
        raise ValueError(f"Empty table: {path}")
    try:
        [float(value) for value in first]
        header = None
    except ValueError:
        header = 0
    return pd.read_csv(path, sep=r"\s+", header=header)


def read_samples(bed_path):
    return pd.read_csv(
        Path(bed_path).with_suffix(".fam"),
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["FID", "IID"],
        dtype=str,
    )


def read_annotations(path, variants, *, columns=None, add_base=False, partition=True):
    table = read_table(path)
    metadata = {"chr", "bp", "snp", "cm"}
    if columns:
        names = {str(c).lower(): c for c in table.columns}
        requested = [c.strip().lower() for c in columns.split(",")]
        if len(set(requested)) != len(requested):
            raise ValueError("Annotation names must be unique.")
        if any(c in metadata or c not in names for c in requested):
            raise ValueError(f"Unknown or metadata annotation column in {columns}")
        selected = [names[c] for c in requested]
        if add_base and "base" in names:
            selected = [names["base"]] + [c for c in selected if c != names["base"]]
        table = table.loc[:, selected]
    else:
        table = table.drop(columns=[c for c in table if str(c).lower() in metadata])
    if add_base and not any(str(c).lower() == "base" for c in table):
        table.insert(0, "base", 1)
    a = table.to_numpy(dtype=np.float64)
    if a.shape[0] != variants or a.shape[1] == 0:
        raise ValueError("Annotation dimensions do not match the genotype variants.")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("Annotations must contain only 0 and 1.")
    if partition and not np.all(a.sum(axis=1) == 1):
        raise ValueError("Each variant must belong to exactly one annotation bin.")
    if add_base and not np.all(a[:, 0] == 1):
        raise ValueError("The base annotation must be the first column and all ones.")
    return a.astype(np.int8), list(map(str, table.columns))


def read_maf_ld(path, variants):
    table = read_table(path)
    table.columns = [str(c).lower() for c in table.columns]
    if len(table) != variants or table.shape[1] < 2:
        raise ValueError(
            "The MAF/LD table needs one row per variant and at least two columns."
        )
    if {"maf_feat", "ld_feat", "maf"} <= set(table.columns):
        features = table[["maf_feat", "ld_feat", "maf"]].to_numpy(float)
    elif list(table.columns) == ["0", "1", "2"]:
        features = table.to_numpy(float)
    else:
        maf_column = next(
            (c for c in ("maf", "af", "freq") if c in table), table.columns[0]
        )
        ld_column = next(
            (c for c in ("ldscore", "ld_score", "ld", "l2") if c in table),
            table.columns[1],
        )
        maf = table[maf_column].to_numpy(float)
        maf = np.minimum(maf, 1 - maf)
        features = np.column_stack([maf, table[ld_column].to_numpy(float), maf])
    if not np.all(np.isfinite(features)) or np.any(
        (features[:, 2] < 0) | (features[:, 2] > 0.5)
    ):
        raise ValueError("MAF/LD values must be finite; MAF must lie in [0, 0.5].")
    weight_column = next(
        (
            c
            for c in ("weight", "weights", "ldak_weight", "ldak_weights", "w")
            if c in table
        ),
        None,
    )
    weights = None if weight_column is None else table[weight_column].to_numpy(float)
    if weights is not None and (
        not np.all(np.isfinite(weights)) or np.any(weights < 0)
    ):
        raise ValueError("LDAK weights must be finite and nonnegative.")
    return features, weights


def ldak_weights(features, weights=None):
    maf = features[:, 2]
    q = np.power(np.maximum(maf * (1 - maf), 0), 0.75)
    if weights is None:
        if np.any(features[:, 1] <= 0):
            raise ValueError("LDAK simulation requires positive LD scores.")
        q /= np.maximum(features[:, 1], 1e-12)
    else:
        q *= weights
    if not np.any(q > 0):
        raise ValueError("All LDAK weights are zero.")
    return q


def genotype_blocks(path, chunk_size, dtype="float64", max_mem=False):
    with open_bed(str(path)) as bed:
        x = bed.read(dtype=dtype) if max_mem else None
        for start in range(0, bed.sid_count, chunk_size):
            end = min(start + chunk_size, bed.sid_count)
            block = (
                x[:, start:end]
                if max_mem
                else bed.read(index=np.s_[:, start:end], dtype=dtype)
            )
            yield start, end, block


def standardize(x, rng=None, *, missing_only=False):
    """Use HWE imputation when an RNG is supplied, otherwise mean imputation."""
    if rng is None:
        mean = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0, ddof=0)
        sd[sd == 0] = 1
        x -= mean
        x /= sd
        np.nan_to_num(x, copy=False, nan=0, posinf=0, neginf=0)
        return x
    p = np.nanmean(x, axis=0) * 0.5
    p0, p1 = (1 - p) ** 2, 2 * p * (1 - p)
    if missing_only:
        rows, cols = np.where(np.isnan(x))
        u = rng.random(len(rows))
        imputed = np.zeros(len(rows), dtype=x.dtype)
        imputed[(u >= p0[cols]) & (u < (p0 + p1)[cols])] = 1
        imputed[u >= (p0 + p1)[cols]] = 2
        x[rows, cols] = imputed
    else:
        # These draws are also consumed for observed genotypes in the manuscript runs.
        u = rng.random(x.shape)
        imputed = np.zeros_like(x)
        imputed[(u >= p0[None, :]) & (u < (p0 + p1)[None, :])] = 1
        imputed[u >= (p0 + p1)[None, :]] = 2
        missing = np.isnan(x)
        x[missing] = imputed[missing]
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    x[:] = (x - mean[None, :]) / sd[None, :]
    return x


def new_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_phenotype(path, samples, y, keep=None):
    table = samples.copy()
    table["pheno"] = y
    if keep is not None:
        table = table.loc[keep]
    table.to_csv(path, sep=" ", index=False, float_format="%.6f", na_rep="NA")


def write_settings(path, settings):
    Path(path).write_text(
        json.dumps(
            settings,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
        )
        + "\n"
    )
