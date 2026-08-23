from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable
import math

import numpy as np
import pandas as pd

from .. import utils
from ..inference.sumcore import Sumcore


_DEFAULT_PHENO_SUFFIXES = (
    ".pheno.gz", ".pheno", ".phen.gz", ".phen", ".txt.gz", ".txt", ".tsv.gz", ".tsv", ".gz"
)
_DEFAULT_COV_SUFFIXES = (
    ".cov.gz", ".cov", ".covar.gz", ".covar", ".txt.gz", ".txt", ".tsv.gz", ".tsv", ".gz"
)
_DEFAULT_SUMSTATS_SUFFIXES = (
    ".sumstats.gz", ".sumstats", ".sumstat", ".sumstat.gz", ".txt.gz", ".txt", ".tsv.gz", ".tsv", ".gz"
)

_TARGET_BLOCK_MEMORY_MB = 256.0
_FLOAT32_EXACT_INT_MAX = 16_777_216


_COMPACT_RG_MANIFEST_COLUMNS = (
    "phen1",
    "phen2",
    "sumstats1",
    "sumstats2",
    "overlap_covariance",
    "cov_rank1",
    "cov_rank2",
)

@dataclass(frozen=True)
class PairSpec:
    phen1: str
    phen2: str


@dataclass(frozen=True)
class TraitResidual:
    phen: str
    sumstats_path: str
    pheno_path: str | None
    cov_path: str | None
    cov_rank: int
    residuals: np.ndarray          # length n_obs_trait, float64
    rss: float                     # total residual sum of squares on the full trait sample
    sample_index: pd.MultiIndex | None = None
    positions: np.ndarray | None = None   # length n_obs_trait, positions on the common sample axis
    n_obs: int = 0


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------


def _read_delim_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", compression="infer", comment="#")
    except Exception:
        return pd.read_csv(path, sep=r"\s+", engine="python", compression="infer", comment="#")



def _read_name_list(path: str) -> list[str]:
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", compression="infer", comment="#", header=None)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", compression="infer", comment="#", header=None)
    if df.shape[1] < 1:
        raise ValueError(f"Phenotype list '{path}' has no columns.")
    vals = df.iloc[:, 0].astype(str).str.strip()
    if vals.empty:
        raise ValueError(f"Phenotype list '{path}' is empty.")
    if vals.iloc[0].lower() in {"phen", "phenotype", "trait", "name"}:
        vals = vals.iloc[1:]
    vals = [v for v in vals.tolist() if v != ""]
    if len(vals) == 0:
        raise ValueError(f"Phenotype list '{path}' contains no usable phenotype names.")
    if len(vals) != len(set(vals)):
        dup = pd.Series(vals).duplicated(keep=False)
        bad = sorted(set(pd.Series(vals)[dup].tolist()))
        raise ValueError(f"Phenotype list '{path}' contains duplicates: {bad}")
    return vals



def _read_pair_list(path: str) -> list[PairSpec]:
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", compression="infer", comment="#", header=None)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", compression="infer", comment="#", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Pair list '{path}' must have at least two columns.")
    a = df.iloc[:, 0].astype(str).str.strip()
    b = df.iloc[:, 1].astype(str).str.strip()
    if (
        not a.empty and not b.empty
        and a.iloc[0].lower() in {"phen1", "trait1", "pheno1", "name1"}
        and b.iloc[0].lower() in {"phen2", "trait2", "pheno2", "name2"}
    ):
        a = a.iloc[1:]
        b = b.iloc[1:]
    pairs = []
    for x, y in zip(a.tolist(), b.tolist()):
        if x == "" or y == "":
            continue
        pairs.append(PairSpec(x, y))
    if len(pairs) == 0:
        raise ValueError(f"Pair list '{path}' contains no usable pairs.")

    seen_ordered = set()
    seen_unordered = set()
    for p in pairs:
        if p.phen1 == p.phen2:
            raise ValueError(f"Pair list '{path}' contains a self-pair for '{p.phen1}'.")
        ordered = (p.phen1, p.phen2)
        if ordered in seen_ordered:
            raise ValueError(f"Pair list '{path}' contains a duplicate ordered pair {ordered}.")
        seen_ordered.add(ordered)
        unordered = tuple(sorted(ordered))
        if unordered in seen_unordered:
            raise ValueError(
                f"Pair list '{path}' contains the same unordered pair more than once: {unordered}."
            )
        seen_unordered.add(unordered)
    return pairs



def _manifest_column(df: pd.DataFrame, *names: str):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        hit = lower.get(str(name).strip().lower())
        if hit is not None:
            return hit
    return None



def _read_sumstats_mapping(path: str) -> dict[str, str]:
    df = _read_delim_table(path)
    phen_col = _manifest_column(df, "phen", "phenotype", "trait", "name")
    sum_col = _manifest_column(df, "sumstats", "sumstats_path", "path", "file")
    if phen_col is None or sum_col is None:
        raise ValueError(
            f"Sumstats mapping file '{path}' must contain columns like 'phen' and 'sumstats'."
        )
    out: dict[str, str] = {}
    for idx, row in df.iterrows():
        phen = str(row[phen_col]).strip() if not pd.isna(row[phen_col]) else ""
        spath_raw = str(row[sum_col]).strip() if not pd.isna(row[sum_col]) else ""
        if phen == "" or spath_raw == "":
            continue
        spath = utils._normalize_path_spec(spath_raw)
        if not utils._path_spec_exists(spath):
            raise ValueError(f"Sumstats mapping row {idx + 1}: file/spec not found '{spath_raw}'.")
        prev = out.get(phen)
        if prev is not None and prev != spath:
            raise ValueError(
                f"Sumstats mapping file '{path}' assigns multiple different paths to phenotype '{phen}'."
            )
        out[phen] = spath
    if len(out) == 0:
        raise ValueError(f"Sumstats mapping file '{path}' contains no usable rows.")
    return out



def _resolve_unique_named_file(directory: str | Path, phen: str, suffixes: Iterable[str], *, kind: str) -> str:
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f"{kind} source '{directory}' is not a directory.")
    candidates = []
    direct = d / phen
    if direct.is_file():
        candidates.append(str(direct.resolve()))
    for suf in suffixes:
        p = d / f"{phen}{suf}"
        if p.is_file():
            candidates.append(str(p.resolve()))
    stem_hits = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name == phen:
            continue
        for suf in suffixes:
            if name == f"{phen}{suf}":
                break
        else:
            if p.stem == phen or p.name.split(".")[0] == phen:
                stem_hits.append(str(p.resolve()))
    candidates.extend(stem_hits)
    candidates = sorted(set(candidates))
    if len(candidates) == 0:
        raise ValueError(f"Could not find a unique {kind} file for phenotype '{phen}' in '{directory}'.")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {kind} files matched phenotype '{phen}' in '{directory}': {candidates}"
        )
    return candidates[0]


def _resolve_unique_prefixed_file(directory: str | Path, phen: str, suffixes: Iterable[str], *, kind: str) -> str:
    """
    Resolve exactly one file matching the literal pattern

        <phen>*<candidate suffix>

    inside a directory.

    Examples for phen='albumin':
      - albumin_cov.insample10.sumstat      -> match
      - albumin.sumstats.gz                 -> match
      - albumin                              -> match (kept for backward compatibility)
      - xalbumin.sumstat                     -> no match
      - albumin2.sumstat                     -> match, because it satisfies the literal prefix rule
    """
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f"{kind} source '{directory}' is not a directory.")

    phen = str(phen).strip()
    if phen == "":
        raise ValueError(f"Cannot resolve {kind} file for an empty phenotype name.")

    suffixes = tuple(dict.fromkeys(str(s) for s in suffixes))
    candidates: list[str] = []

    # Preserve old exact bare-name behavior.
    direct = d / phen
    if direct.is_file():
        candidates.append(str(direct.resolve()))

    for p in d.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.startswith(phen):
            continue
        if any(name.endswith(suf) for suf in suffixes):
            candidates.append(str(p.resolve()))

    candidates = sorted(set(candidates))
    if len(candidates) == 0:
        raise ValueError(
            f"Could not find a unique {kind} file for phenotype '{phen}' in '{directory}'. "
            f"Expected exactly one match to the pattern '{phen}*<valid suffix>'."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple {kind} files matched phenotype '{phen}' in '{directory}': {candidates}"
        )
    return candidates[0]



def _pair_output_stem(phen1: str, phen2: str) -> str:
    fn = getattr(utils, "_pair_output_stem", None)
    if callable(fn):
        return str(fn(phen1, phen2))

    def _sanitize(x: str) -> str:
        s = str(x).strip()
        s = s.replace(os.sep, ".")
        s = "".join(ch if (ch.isalnum() or ch in "._-") else "." for ch in s)
        while ".." in s:
            s = s.replace("..", ".")
        s = s.strip(".")
        return s or "trait"

    import os
    return f"{_sanitize(phen1)}.{_sanitize(phen2)}"


# -----------------------------------------------------------------------------
# Input source wrappers
# -----------------------------------------------------------------------------


class _PhenotypeSource:
    def __init__(self, source: str, *, missing_tokens: list[str]):
        self.source = str(source)
        self.missing_tokens = list(missing_tokens)
        self.path = Path(self.source)
        if not self.path.exists():
            raise ValueError(f"Phenotype source '{source}' does not exist.")
        self._wide_header = None

    @property
    def is_dir(self) -> bool:
        return self.path.is_dir()

    def available_names(self) -> list[str] | None:
        if self.is_dir:
            return None
        hdr = pd.read_csv(self.source, sep=r"\s+", compression="infer", nrows=0)
        cols = list(hdr.columns)
        if len(cols) < 3:
            raise ValueError(
                f"Wide phenotype file '{self.source}' must have at least three columns: FID IID PHENO..."
            )
        self._wide_header = cols
        return [str(c) for c in cols[2:]]

    def read_trait(self, phen: str):
        if self.is_dir:
            pheno_path = _resolve_unique_named_file(
                self.source, phen, _DEFAULT_PHENO_SUFFIXES, kind="phenotype"
            )
            df, info = Sumcore._read_overlap_phenotype_file(pheno_path, self.missing_tokens)
            return df, pheno_path, info

        hdr = pd.read_csv(self.source, sep=r"\s+", compression="infer", nrows=0)
        cols = list(hdr.columns)
        if phen not in cols[2:]:
            raise ValueError(f"Wide phenotype file '{self.source}' does not contain phenotype column '{phen}'.")

        str_missing, num_missing = Sumcore._split_missing_tokens(self.missing_tokens)
        df = pd.read_csv(
            self.source,
            sep=r"\s+",
            engine="python",
            header=0,
            usecols=[cols[0], cols[1], phen],
            na_values=list(str_missing),
            keep_default_na=True,
            comment="#",
        )
        n_raw = int(df.shape[0])
        df.columns = ["FID", "IID", "_y_raw"]
        df["FID"] = df["FID"].astype(str).str.strip()
        df["IID"] = df["IID"].astype(str).str.strip()
        df["_y_raw"] = Sumcore._mask_numeric_missing_in_series(df["_y_raw"], num_missing)
        n_missing_pheno = int(df["_y_raw"].isna().sum())
        n_missing_id = int(df["FID"].isna().sum() + df["IID"].isna().sum())
        df = df.dropna(subset=["FID", "IID", "_y_raw"])
        if df.shape[0] == 0:
            raise ValueError(f"Wide phenotype file '{self.source}' leaves no usable rows for trait '{phen}'.")
        if df.duplicated(subset=["FID", "IID"]).any():
            raise ValueError(
                f"Wide phenotype file '{self.source}' contains duplicate FID/IID rows for trait '{phen}'."
            )
        info = {
            "path": self.source,
            "n_raw": n_raw,
            "n_missing_pheno": n_missing_pheno,
            "n_missing_id": n_missing_id,
            "n_used": int(df.shape[0]),
        }
        return df, self.source, info


class _CovariateSource:
    def __init__(self, source: str | None, *, missing_tokens: list[str]):
        self.source = None if source is None else str(source)
        self.missing_tokens = list(missing_tokens)
        self._shared_cov = None
        if self.source is not None and (not Path(self.source).exists()):
            raise ValueError(f"Covariate source '{self.source}' does not exist.")

    @property
    def is_none(self) -> bool:
        return self.source is None

    @property
    def is_dir(self) -> bool:
        return (self.source is not None) and Path(self.source).is_dir()

    def read_trait(self, phen: str):
        if self.source is None:
            return None, None, None
        if self.is_dir:
            cov_path = _resolve_unique_named_file(
                self.source, phen, _DEFAULT_COV_SUFFIXES, kind="covariate"
            )
            cov_df, info = Sumcore._read_overlap_covariate_file(cov_path, self.missing_tokens)
            return cov_df, cov_path, info

        if self._shared_cov is None:
            self._shared_cov = Sumcore._read_overlap_covariate_file(self.source, self.missing_tokens)
        cov_df, info = self._shared_cov
        return cov_df.copy(), self.source, dict(info)


class _SumstatsSource:
    def __init__(self, source: str):
        self.source = str(source)
        self.path = Path(self.source)
        if not self.path.exists():
            raise ValueError(f"Sumstats source '{source}' does not exist.")
        self._mapping = None
        if self.path.is_file() and ("," not in self.source):
            lower = self.path.name.lower()
            if lower.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz")):
                try:
                    self._mapping = _read_sumstats_mapping(self.source)
                except Exception:
                    self._mapping = None

    def resolve(self, phen: str) -> str:
        if self.path.is_dir():
            return _resolve_unique_prefixed_file(
                self.source,
                phen,
                _DEFAULT_SUMSTATS_SUFFIXES,
                kind="sumstats",
            )
        if self._mapping is not None:
            hit = self._mapping.get(phen)
            if hit is None:
                raise ValueError(f"Sumstats mapping '{self.source}' does not contain phenotype '{phen}'.")
            return hit
        raise ValueError(
            "--sum-dir must be either a directory of per-trait sumstats files or a mapping file with columns phen,sumstats."
        )


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------



def _merge_trait_and_cov(pheno_df: pd.DataFrame, cov_df: pd.DataFrame | None, *, pheno_path: str, cov_path: str | None) -> pd.DataFrame:
    if cov_df is None:
        out = pheno_df.copy()
    else:
        out = pheno_df.merge(cov_df, on=["FID", "IID"], how="inner")
        if out.shape[0] == 0:
            raise RuntimeError(
                f"No overlapping FID/IID rows remained after merging phenotype '{pheno_path}' "
                f"with covariates '{cov_path}'."
            )
        if out.duplicated(subset=["FID", "IID"]).any():
            raise RuntimeError(
                f"Duplicate FID/IID rows after phenotype/covariate merge for phenotype '{pheno_path}'."
            )
    out = out.sort_values(["FID", "IID"], kind="mergesort").reset_index(drop=True)
    out.index = pd.MultiIndex.from_frame(out[["FID", "IID"]], names=["FID", "IID"])
    return out



def _solve_linear_multi(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs = 0.5 * (lhs + lhs.T)

    if rhs.ndim == 1:
        rhs = rhs.reshape(-1, 1)

    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        try:
            sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(lhs) @ rhs

    sol = np.asarray(sol, dtype=np.float64)
    if not np.isfinite(sol).all():
        raise RuntimeError("Non-finite solution encountered while residualizing phenotype on covariates.")
    return sol



def _fit_residuals(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    A = X.T @ X
    B = X.T @ Y
    alpha = _solve_linear_multi(A, B)
    R = Y - X @ alpha
    rss = np.sum(R * R, axis=0, dtype=np.float64)
    return R, rss



def _build_design_matrix_from_cov(cov_df: pd.DataFrame | None, nrows: int) -> tuple[np.ndarray, list[str]]:
    if cov_df is None:
        return np.ones((int(nrows), 1), dtype=np.float64), []
    cov_cols = [c for c in cov_df.columns if str(c).startswith("_cov")]
    if len(cov_cols) == 0:
        return np.ones((int(nrows), 1), dtype=np.float64), []
    X = np.empty((int(nrows), 1 + len(cov_cols)), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = cov_df.loc[:, cov_cols].to_numpy(dtype=np.float64, copy=False)
    return X, cov_cols


def _design_matrix_rank(X: np.ndarray) -> int:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Design matrix must be 2D; got shape {X.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        return 0
    gram = X.T @ X
    gram = 0.5 * (gram + gram.T)
    return int(np.linalg.matrix_rank(gram, hermitian=True))


# -----------------------------------------------------------------------------
# Trait residual preparation
# -----------------------------------------------------------------------------



def _prepare_trait_residual_general(
    phen: str,
    phen_source: _PhenotypeSource,
    cov_source: _CovariateSource,
    sumstats_source: _SumstatsSource,
    *,
    log=None,
) -> TraitResidual:
    pheno_df, pheno_path, pheno_info = phen_source.read_trait(phen)
    cov_df, cov_path, cov_info = cov_source.read_trait(phen)
    table = _merge_trait_and_cov(pheno_df, cov_df, pheno_path=pheno_path, cov_path=cov_path)

    y, X, _ = Sumcore._design_from_trait_table(table.reset_index(drop=True))
    R, rss = _fit_residuals(X, y)
    resid = np.asarray(R[:, 0], dtype=np.float64, order="C")
    rss_scalar = float(rss[0])
    cov_rank = int(_design_matrix_rank(X) - 1)
    if cov_rank < 0:
        raise RuntimeError(f"Derived negative cov_rank for phenotype '{phen}'.")

    sumstats_path = sumstats_source.resolve(phen)
    if log is not None:
        msg = (
            f"[make-rg-manifest] trait '{phen}': n_obs={table.shape[0]}, cov_rank={cov_rank}, "
            f"rss={rss_scalar:.6g}, sumstats='{sumstats_path}'"
        )
        log._log(msg)
        if cov_info is not None:
            log._log(
                f"[make-rg-manifest] trait '{phen}': pheno_rows={pheno_info['n_used']}, "
                f"cov_rows={cov_info['n_used']}, n_cov={cov_info['n_cov']}"
            )

    return TraitResidual(
        phen=phen,
        sumstats_path=sumstats_path,
        pheno_path=pheno_path,
        cov_path=cov_path,
        cov_rank=cov_rank,
        residuals=resid,
        rss=rss_scalar,
        sample_index=table.index,
        positions=None,
        n_obs=int(table.shape[0]),
    )



def _prepare_trait_residuals_wide_shared(
    phen_order: list[str],
    phen_source: _PhenotypeSource,
    cov_source: _CovariateSource,
    sumstats_source: _SumstatsSource,
    *,
    log=None,
) -> tuple[list[TraitResidual], pd.MultiIndex]:
    """
    Exact batched path for:
      - a single wide phenotype file
      - no covariates OR one shared covariate file

    This is the big simulation speed-up path: traits sharing the same observed-mask
    are residualized together with a single multi-RHS solve, and the no-missingness
    case becomes one shared projection for all requested phenotypes.
    """
    if phen_source.is_dir:
        raise ValueError("wide_shared residual path requires a wide phenotype file, not a directory.")
    if cov_source.is_dir:
        raise ValueError("wide_shared residual path requires no covariates or a single shared covariate file.")

    hdr = pd.read_csv(phen_source.source, sep=r"\s+", compression="infer", nrows=0)
    cols = list(hdr.columns)
    if len(cols) < 3:
        raise ValueError(
            f"Wide phenotype file '{phen_source.source}' must have at least three columns: FID IID PHENO..."
        )

    missing = [p for p in phen_order if p not in cols[2:]]
    if missing:
        raise ValueError(
            f"Wide phenotype file '{phen_source.source}' is missing requested phenotype columns: {missing}"
        )

    str_missing, num_missing = Sumcore._split_missing_tokens(phen_source.missing_tokens)
    usecols = [cols[0], cols[1]] + list(phen_order)
    pheno_df = pd.read_csv(
        phen_source.source,
        sep=r"\s+",
        engine="python",
        header=0,
        usecols=usecols,
        na_values=list(str_missing),
        keep_default_na=True,
        comment="#",
    )
    pheno_df = pheno_df.rename(columns={cols[0]: "FID", cols[1]: "IID"})
    pheno_df["FID"] = pheno_df["FID"].astype(str).str.strip()
    pheno_df["IID"] = pheno_df["IID"].astype(str).str.strip()
    if pheno_df.duplicated(subset=["FID", "IID"]).any():
        raise ValueError(f"Wide phenotype file '{phen_source.source}' contains duplicate FID/IID rows.")

    for phen in phen_order:
        pheno_df[phen] = Sumcore._mask_numeric_missing_in_series(pheno_df[phen], num_missing)

    cov_df = None
    cov_path = None
    cov_info = None
    if not cov_source.is_none:
        cov_df, cov_path, cov_info = cov_source.read_trait(phen_order[0])

    base = _merge_trait_and_cov(pheno_df, cov_df, pheno_path=phen_source.source, cov_path=cov_path)
    base_index = base.index
    X_base, cov_cols = _build_design_matrix_from_cov(base[[c for c in base.columns if str(c).startswith("_cov")]] if cov_df is not None else None, base.shape[0])

    Y = base.loc[:, phen_order].to_numpy(dtype=np.float64, copy=False)
    if Y.ndim != 2 or Y.shape[1] != len(phen_order):
        raise RuntimeError("Unexpected wide phenotype matrix shape after merge.")

    valid = np.isfinite(Y)
    if np.any(np.sum(valid, axis=0) == 0):
        bad = [phen_order[j] for j in np.where(np.sum(valid, axis=0) == 0)[0].tolist()]
        raise ValueError(f"No usable rows remain for phenotype(s): {bad}")

    def _mask_key(mask: np.ndarray) -> bytes:
        return np.packbits(mask.astype(np.uint8, copy=False), bitorder="little").tobytes()

    groups: dict[bytes, list[int]] = {}
    first_mask: dict[bytes, np.ndarray] = {}
    for j, phen in enumerate(phen_order):
        key = _mask_key(valid[:, j])
        groups.setdefault(key, []).append(j)
        if key not in first_mask:
            first_mask[key] = valid[:, j].copy()

    if log is not None:
        n_groups = len(groups)
        no_missing = bool(np.all(valid))
        log._log(
            f"[make-rg-manifest] wide/shared batched residualization: traits={len(phen_order)}, "
            f"rows={base.shape[0]}, cov_cols={len(cov_cols)}, unique_missing_patterns={n_groups}, "
            f"no_missing={no_missing}."
        )
        if cov_info is not None:
            log._log(
                f"[make-rg-manifest] shared covariates: rows={cov_info['n_used']}, n_cov={cov_info['n_cov']}"
            )

    out: list[TraitResidual | None] = [None] * len(phen_order)
    sumstats_paths = {phen: sumstats_source.resolve(phen) for phen in phen_order}

    for key, cols_group in groups.items():
        mask = first_mask[key]
        rows = np.flatnonzero(mask)
        Xg = X_base[rows, :]
        Yg = Y[rows[:, None], np.asarray(cols_group, dtype=np.int64)]
        Rg, rss_g = _fit_residuals(Xg, Yg)
        cov_rank_g = int(_design_matrix_rank(Xg) - 1)
        if cov_rank_g < 0:
            raise RuntimeError("Derived negative cov_rank in wide/shared residualization.")

        for local, j in enumerate(cols_group):
            phen = phen_order[j]
            out[j] = TraitResidual(
                phen=phen,
                sumstats_path=sumstats_paths[phen],
                pheno_path=phen_source.source,
                cov_path=cov_path,
                cov_rank=cov_rank_g,
                residuals=np.asarray(Rg[:, local], dtype=np.float64, order="C"),
                rss=float(rss_g[local]),
                sample_index=None,
                positions=rows.astype(np.int64, copy=False),
                n_obs=int(rows.size),
            )
            if log is not None:
                log._log(
                    f"[make-rg-manifest] trait '{phen}': n_obs={rows.size}, cov_rank={cov_rank_g}, "
                    f"rss={float(rss_g[local]):.6g}, sumstats='{sumstats_paths[phen]}'"
                )

    traits = [tr for tr in out if tr is not None]
    if len(traits) != len(phen_order):
        raise RuntimeError("Internal error: some wide/shared traits were not residualized.")
    return traits, base_index



def _assign_common_positions(traits: list[TraitResidual], *, log=None) -> tuple[list[TraitResidual], int]:
    if len(traits) == 0:
        return [], 0
    if all(tr.positions is not None for tr in traits):
        axis_len = max((int(np.max(tr.positions)) + 1) if tr.positions.size > 0 else 0 for tr in traits)
        return traits, int(axis_len)

    first = traits[0].sample_index
    if first is None:
        raise RuntimeError("Trait residual missing both sample_index and positions.")

    same_axis = True
    for tr in traits[1:]:
        if tr.sample_index is None:
            raise RuntimeError("Trait residual missing sample_index during position assignment.")
        if not tr.sample_index.equals(first):
            same_axis = False
            break

    if same_axis:
        axis_len = len(first)
        out = []
        pos = np.arange(axis_len, dtype=np.int64)
        for tr in traits:
            out.append(TraitResidual(
                phen=tr.phen,
                sumstats_path=tr.sumstats_path,
                pheno_path=tr.pheno_path,
                cov_path=tr.cov_path,
                cov_rank=tr.cov_rank,
                residuals=tr.residuals,
                rss=tr.rss,
                sample_index=None,
                positions=pos.copy(),
                n_obs=tr.n_obs,
            ))
        if log is not None:
            log._log(f"[make-rg-manifest] all traits share a common sample axis of length {axis_len}.")
        return out, int(axis_len)

    union_index = first
    for tr in traits[1:]:
        union_index = union_index.union(tr.sample_index, sort=False)

    axis_len = int(len(union_index))
    out = []
    for tr in traits:
        pos = union_index.get_indexer(tr.sample_index)
        if np.any(pos < 0):
            raise RuntimeError(f"Failed to map sample IDs onto the common axis for trait '{tr.phen}'.")
        out.append(TraitResidual(
            phen=tr.phen,
            sumstats_path=tr.sumstats_path,
            pheno_path=tr.pheno_path,
            cov_path=tr.cov_path,
            cov_rank=tr.cov_rank,
            residuals=tr.residuals,
            rss=tr.rss,
            sample_index=None,
            positions=pos.astype(np.int64, copy=False),
            n_obs=tr.n_obs,
        ))
    if log is not None:
        log._log(
            f"[make-rg-manifest] built a union sample axis of length {axis_len} for {len(traits)} traits."
        )
    return out, axis_len


# -----------------------------------------------------------------------------
# Pairwise cross-product engine
# -----------------------------------------------------------------------------



def _choose_trait_block_size(axis_len: int, ntraits: int) -> int:
    if axis_len <= 0 or ntraits <= 0:
        return 1
    target_bytes = float(_TARGET_BLOCK_MEMORY_MB) * 1024.0 * 1024.0
    # two residual blocks (float64) + two availability blocks (float32) live at once
    per_trait_bytes = float(axis_len) * (8.0 + 4.0)
    if per_trait_bytes <= 0.0:
        return max(1, ntraits)
    b = int(max(1.0, math.floor(target_bytes / (2.0 * per_trait_bytes))))
    return max(1, min(int(ntraits), b))



def _assemble_block(traits: list[TraitResidual], trait_indices: list[int], axis_len: int, *, overlap_dtype) -> tuple[np.ndarray, np.ndarray]:
    R = np.zeros((int(axis_len), len(trait_indices)), dtype=np.float64, order="F")
    M = np.zeros((int(axis_len), len(trait_indices)), dtype=overlap_dtype, order="F")
    for col, tidx in enumerate(trait_indices):
        tr = traits[tidx]
        pos = np.asarray(tr.positions, dtype=np.int64)
        if pos.ndim != 1 or pos.size != tr.residuals.size:
            raise RuntimeError(f"Invalid positions/residual lengths for trait '{tr.phen}'.")
        R[pos, col] = np.asarray(tr.residuals, dtype=np.float64)
        M[pos, col] = 1
    return R, M



def _compute_pairs_blockwise(
    traits: list[TraitResidual],
    pairs: list[PairSpec],
    *,
    axis_len: int,
    allow_zero_overlap: bool = False,
    log=None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(traits) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    phen_to_idx = {tr.phen: i for i, tr in enumerate(traits)}
    rss = np.asarray([tr.rss for tr in traits], dtype=np.float64)
    if np.any(~np.isfinite(rss)) or np.any(rss <= 0.0):
        bad = [traits[i].phen for i in np.where((~np.isfinite(rss)) | (rss <= 0.0))[0].tolist()]
        raise RuntimeError(f"Non-finite or non-positive residual RSS for phenotype(s): {bad}")

    block_size = _choose_trait_block_size(axis_len, len(traits))
    block_of = {i: i // block_size for i in range(len(traits))}
    block_ranges: dict[int, list[int]] = {}
    for i in range(len(traits)):
        block_ranges.setdefault(block_of[i], []).append(i)

    pairs_by_block: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for ridx, pair in enumerate(pairs):
        i = phen_to_idx[pair.phen1]
        j = phen_to_idx[pair.phen2]
        bi = block_of[i]
        bj = block_of[j]
        key = (bi, bj) if bi <= bj else (bj, bi)
        if bi <= bj:
            li = i - bi * block_size
            lj = j - bj * block_size
        else:
            li = j - bj * block_size
            lj = i - bi * block_size
        pairs_by_block.setdefault(key, []).append((li, lj, ridx))

    overlap_dtype = np.float32 if axis_len <= _FLOAT32_EXACT_INT_MAX else np.float64
    overlap_covariances = np.full(len(pairs), np.nan, dtype=np.float64)
    overlaps = np.zeros(len(pairs), dtype=np.int64)

    if log is not None:
        log._log(
            f"[make-rg-manifest] pairwise engine: traits={len(traits)}, pairs={len(pairs)}, "
            f"axis_len={axis_len}, trait_block_size={block_size}, block_pairs={len(pairs_by_block)}."
        )

    needed_block_ids = sorted(set(k[0] for k in pairs_by_block).union(k[1] for k in pairs_by_block))
    block_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _get_block(block_id: int) -> tuple[np.ndarray, np.ndarray]:
        got = block_cache.get(block_id)
        if got is not None:
            return got
        trait_indices = block_ranges[block_id]
        built = _assemble_block(traits, trait_indices, axis_len, overlap_dtype=overlap_dtype)
        block_cache[block_id] = built
        return built

    for bi in needed_block_ids:
        Ri, Mi = _get_block(bi)
        for bj in sorted(k[1] for k in pairs_by_block if k[0] == bi):
            Rj, Mj = _get_block(bj)
            G = Ri.T @ Rj
            O = Mi.T @ Mj
            for li, lj, ridx in pairs_by_block[(bi, bj)]:
                pair = pairs[ridx]
                i = phen_to_idx[pair.phen1]
                j = phen_to_idx[pair.phen2]
                num = float(G[li, lj])
                denom = math.sqrt(float(rss[i]) * float(rss[j]))
                if not (math.isfinite(num) and math.isfinite(denom) and denom > 0.0):
                    raise RuntimeError(
                        f"Failed to compute overlap covariance for pair "
                        f"'{pair.phen1}' vs '{pair.phen2}'."
                    )
                overlap_covariances[ridx] = num / denom
                ov_raw = float(O[li, lj])
                ov = int(round(ov_raw))
                if abs(ov_raw - float(ov)) > 1e-3:
                    raise RuntimeError(
                        f"Non-integer overlap count encountered for pair '{pair.phen1}' vs '{pair.phen2}': {ov_raw}"
                    )
                overlaps[ridx] = ov

        # keep only the current left block cached to cap memory
        kill = [k for k in block_cache.keys() if k != bi]
        for k in kill:
            del block_cache[k]

    if np.any(~np.isfinite(overlap_covariances)):
        bad = np.where(~np.isfinite(overlap_covariances))[0].tolist()
        raise RuntimeError(
            "Non-finite overlap-covariance estimates encountered for pair "
            f"indices: {bad[:10]}"
        )
    zero_overlap = overlaps == 0
    if np.any(overlaps < 0) or (np.any(zero_overlap) and not allow_zero_overlap):
        invalid = overlaps < 0 if allow_zero_overlap else overlaps <= 0
        bad = np.where(invalid)[0].tolist()
        raise RuntimeError(f"Non-positive overlap counts encountered for pair indices: {bad[:10]}")
    if np.any(zero_overlap):
        # Disjoint study samples imply an exact overlap covariance of zero.
        overlap_covariances[zero_overlap] = 0.0
        if log is not None:
            log._log(
                f"[make-rg-manifest] assigned overlap_covariance=0 to "
                f"{int(np.sum(zero_overlap))} zero-overlap pair(s)."
            )

    return overlap_covariances, overlaps


# -----------------------------------------------------------------------------
# Public builder
# -----------------------------------------------------------------------------



def build_rg_manifest(
    *,
    output_path: str,
    phen_source: str,
    sumstats_source: str,
    cov_source: str | None = None,
    phen_list_path: str | None = None,
    pair_list_path: str | None = None,
    all_pairwise: bool = False,
    pheno_missing_values: list[str] | None = None,
    cov_missing_values: list[str] | None = None,
    compact: bool = False,
    allow_zero_overlap: bool = False,
    log=None,
) -> pd.DataFrame:
    """
    Build an rg manifest with the phenotype-side supplied-overlap covariance,
    without recomputing covariate regression and overlap cross-products one pair
    at a time.

    Mathematical identity used
    --------------------------
    For trait t, let r_t be the full-sample residual after projecting its phenotype
    onto [1, covariates_t] using its own study sample. Then for pair (s,t), the
    SUM-CORE sample-overlap covariance is

        c_st = <r_s[overlap], r_t[overlap]> / sqrt( ||r_s||^2 ||r_t||^2 ).

    So once each trait residual vector is computed once, every requested pair only
    needs the overlap dot product of two residual vectors. By embedding each trait's
    residuals onto a common sample axis with zeros outside its observed samples,
    those pairwise numerators become a blockwise Gram matrix, which we compute with
    BLAS-backed matrix multiplications.
    """
    if phen_list_path is None and pair_list_path is None:
        raise ValueError("Provide at least one of --phen-list or --pair-list in --make-rg-manifest mode.")
    if pair_list_path is None and (phen_list_path is not None) and (not all_pairwise):
        raise ValueError("With --phen-list alone, pass --all-pairwise to request all phenotype pairs.")

    pheno_missing_values = ["-9"] if pheno_missing_values is None else list(pheno_missing_values)
    cov_missing_values = ["-9", "NA", "NaN", "nan", ".", "None", "NONE", "null", "NULL"] if cov_missing_values is None else list(cov_missing_values)

    phen_source_obj = _PhenotypeSource(phen_source, missing_tokens=pheno_missing_values)
    cov_source_obj = _CovariateSource(cov_source, missing_tokens=cov_missing_values)
    sumstats_source_obj = _SumstatsSource(sumstats_source)

    phen_whitelist = None
    if phen_list_path is not None:
        phen_whitelist = _read_name_list(phen_list_path)

    if pair_list_path is not None:
        pairs = _read_pair_list(pair_list_path)
        if phen_whitelist is not None:
            allowed = set(phen_whitelist)
            for p in pairs:
                if p.phen1 not in allowed or p.phen2 not in allowed:
                    raise ValueError(
                        f"Pair list contains phenotype(s) outside --phen-list: ({p.phen1}, {p.phen2})."
                    )
        phen_order = []
        seen = set()
        for p in pairs:
            for phen in (p.phen1, p.phen2):
                if phen not in seen:
                    seen.add(phen)
                    phen_order.append(phen)
    else:
        phen_order = list(phen_whitelist)
        pairs = [PairSpec(a, b) for a, b in combinations(phen_order, 2)]

    if len(pairs) == 0:
        raise ValueError("No phenotype pairs were requested.")

    if not phen_source_obj.is_dir:
        available = set(phen_source_obj.available_names() or [])
        missing = [p for p in phen_order if p not in available]
        if missing:
            raise ValueError(
                f"Wide phenotype file '{phen_source}' is missing requested phenotype columns: {missing}"
            )

    # Fast exact path: wide phenotype file + no covariates or one shared covariate file.
    if (not phen_source_obj.is_dir) and (cov_source_obj.is_none or (not cov_source_obj.is_dir)):
        traits, base_index = _prepare_trait_residuals_wide_shared(
            phen_order,
            phen_source_obj,
            cov_source_obj,
            sumstats_source_obj,
            log=log,
        )
        axis_len = int(len(base_index))
    else:
        traits = [
            _prepare_trait_residual_general(
                phen,
                phen_source_obj,
                cov_source_obj,
                sumstats_source_obj,
                log=log,
            )
            for phen in phen_order
        ]
        traits, axis_len = _assign_common_positions(traits, log=log)

    # Prechecks that should fail early before any heavy pairwise work.
    path_to_phen: dict[str, str] = {}
    for tr in traits:
        prev = path_to_phen.get(tr.sumstats_path)
        if prev is not None and prev != tr.phen:
            raise ValueError(
                f"Resolved the same sumstats path '{tr.sumstats_path}' for two different phenotypes: '{prev}' and '{tr.phen}'."
            )
        path_to_phen[tr.sumstats_path] = tr.phen

    seen_stems: dict[str, tuple[str, str]] = {}
    for pair in pairs:
        stem = _pair_output_stem(pair.phen1, pair.phen2)
        prev = seen_stems.get(stem)
        if prev is not None:
            raise ValueError(
                f"Output-name collision after sanitization for pairs {prev} and {(pair.phen1, pair.phen2)}."
            )
        seen_stems[stem] = (pair.phen1, pair.phen2)

    phen_set = {tr.phen for tr in traits}
    for pair in pairs:
        if pair.phen1 not in phen_set or pair.phen2 not in phen_set:
            raise ValueError(f"Requested pair uses an unknown phenotype: ({pair.phen1}, {pair.phen2})")

    overlap_covariances, overlaps = _compute_pairs_blockwise(
        traits,
        pairs,
        axis_len=axis_len,
        allow_zero_overlap=allow_zero_overlap,
        log=log,
    )

    trait_map = {tr.phen: tr for tr in traits}
    rows = []
    for ridx, pair in enumerate(pairs):
        tr1 = trait_map[pair.phen1]
        tr2 = trait_map[pair.phen2]
        overlap_covariance = float(overlap_covariances[ridx])
        n_overlap = int(overlaps[ridx])
        if not np.isfinite(overlap_covariance):
            raise RuntimeError(
                f"Computed non-finite overlap_covariance for pair "
                f"'{pair.phen1}' vs '{pair.phen2}'."
            )
        if n_overlap < 0 or (n_overlap == 0 and not allow_zero_overlap):
            raise RuntimeError(f"No overlapping individuals for pair '{pair.phen1}' vs '{pair.phen2}'.")

        rows.append({
            "phen1": pair.phen1,
            "phen2": pair.phen2,
            "sumstats1": tr1.sumstats_path,
            "sumstats2": tr2.sumstats_path,
            "overlap_covariance": overlap_covariance,
            "cov_rank1": int(tr1.cov_rank),
            "cov_rank2": int(tr2.cov_rank),
            "n_overlap": n_overlap,
            "pheno_path1": tr1.pheno_path,
            "pheno_path2": tr2.pheno_path,
            "cov_path1": tr1.cov_path,
            "cov_path2": tr2.cov_path,
        })
        if log is not None:
            log._log(
                f"[make-rg-manifest] pair {pair.phen1} vs {pair.phen2}: "
                f"n_overlap={n_overlap}, overlap_covariance={overlap_covariance:.15g}"
            )

    out_df = pd.DataFrame(rows)

    if compact:
        missing_cols = [c for c in _COMPACT_RG_MANIFEST_COLUMNS if c not in out_df.columns]
        if missing_cols:
            raise RuntimeError(
                f"Internal error: compact rg manifest columns are missing from output: {missing_cols}"
            )
        out_df = out_df.loc[:, list(_COMPACT_RG_MANIFEST_COLUMNS)].copy()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False, float_format="%.15g")
    if log is not None:
        log._log(
            f"[make-rg-manifest] wrote {out_df.shape[0]} pair(s) across {len(traits)} trait(s) to '{out_path}'."
        )
    return out_df
