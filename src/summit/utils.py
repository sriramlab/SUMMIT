import gzip
import numpy as np
import pandas as pd
import os
import re
import time
import datetime
import glob
from pathlib import Path

# ----------------------- Time helpers ----------------------- #
def _get_time():
    current_time = time.time()
    return current_time


def _get_timestr(current_time):
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    timestr = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)))+" "+str(timezone)
    return timestr

# ----------------------- I/O & parsing helpers ----------------------- #

def _open_text_maybe_gzip(file_path, mode="rt"):
    path = str(file_path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def _read_with_optional_header(file_path):
    paths = _resolve_chr_split_paths(file_path, require=True)
    if len(paths) == 1 and not _is_chr_split_spec(file_path):
        return _read_with_optional_header_one(paths[0])

    header = None
    saw_first = False
    arrays = []
    for p in paths:
        h, arr = _read_with_optional_header_one(p)
        if not saw_first:
            header = h
            saw_first = True
        elif h != header:
            raise ValueError(
                f"Chromosome-split files for '{file_path}' have inconsistent headers; "
                f"first header={header}, file '{p}' header={h}."
            )
        arrays.append(np.asarray(arr))

    if not arrays:
        raise ValueError(f"No files resolved for chromosome-split path spec: {file_path}")
    if arrays[0].ndim == 1:
        data = np.concatenate([np.asarray(a).reshape(-1) for a in arrays], axis=0)
    else:
        data = np.vstack(arrays)
    return header, data


def _parse_column_name(df_hdr, names, default_pos):
    cols = list(df_hdr.columns)
    cols_lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    if default_pos >= len(cols):
        raise ValueError(f"Could not infer column {names}; header too short.")
    return cols[default_pos]


def _normalize_path_spec(path) -> str:
    """Expand user/env vars while preserving chromosome placeholders."""
    s = os.path.expanduser(os.path.expandvars(str(path)))
    return str(Path(s).resolve(strict=False))


def _is_chr_split_spec(path) -> bool:
    return "@" in str(path)


def _chr_path_candidates(path_spec: str, chrom: int) -> list[str]:
    chrom = int(chrom)
    repls = [str(chrom), f"{chrom:02d}"]
    out = []
    for repl in repls:
        p = path_spec.replace("@", repl)
        if p not in out:
            out.append(p)
    return out


def _resolve_chr_split_paths(path_spec, *, n_chr: int = 22, require: bool = True) -> list[str]:
    """
    Resolve a chromosome-split path spec.

    The placeholder '@' is replaced by chromosome numbers 1..n_chr. For convenience
    we try both unpadded and zero-padded chromosome strings, so one spec can match
    paths like chr_1 and chr01. Only files that exist are returned, in chromosome
    order. Non-split paths are returned as a one-element list.
    """
    spec = _normalize_path_spec(path_spec)
    if not _is_chr_split_spec(spec):
        if require and not Path(spec).is_file():
            raise ValueError(f"Could not find file: {path_spec}")
        return [spec]

    out = []
    for chrom in range(1, int(n_chr) + 1):
        hit = None
        for cand in _chr_path_candidates(spec, chrom):
            if Path(cand).is_file():
                hit = cand
                break
        if hit is not None:
            out.append(hit)

    if require and not out:
        raise ValueError(
            f"Could not resolve chromosome-split path spec '{path_spec}'. "
            "Use '@' where the chromosome number should be inserted."
        )
    return out


def _resolve_chr_split_dirs(path_spec, *, n_chr: int = 22, require: bool = True) -> list[str]:
    """
    Resolve a chromosome-split directory spec.

    This is the directory analogue of _resolve_chr_split_paths. It supports h2
    batch layouts such as /path/to/sums/chr@, where each resolved chr directory
    contains the same set of per-trait sumstats files.
    """
    spec = _normalize_path_spec(path_spec)
    if not _is_chr_split_spec(spec):
        if require and not Path(spec).is_dir():
            raise ValueError(f"Could not find directory: {path_spec}")
        return [spec]

    out = []
    for chrom in range(1, int(n_chr) + 1):
        hit = None
        for cand in _chr_path_candidates(spec, chrom):
            if Path(cand).is_dir():
                hit = cand
                break
        if hit is not None:
            out.append(hit)

    if require and not out:
        raise ValueError(
            f"Could not resolve chromosome-split directory spec '{path_spec}'. "
            "Use '@' where the chromosome number should be inserted."
        )
    return out


def _path_spec_exists(path_spec) -> bool:
    try:
        return len(_resolve_chr_split_paths(path_spec, require=True)) > 0
    except Exception:
        return False


def _read_csv_maybe_chr_split(path_spec, **kwargs):
    paths = _resolve_chr_split_paths(path_spec, require=True)
    if len(paths) == 1 and not _is_chr_split_spec(path_spec):
        return pd.read_csv(paths[0], **kwargs)

    frames = []
    columns = None
    for p in paths:
        df = pd.read_csv(p, **kwargs)
        if columns is None:
            columns = list(df.columns)
        elif list(df.columns) != columns:
            raise ValueError(
                f"Chromosome-split files for '{path_spec}' have inconsistent columns; "
                f"first columns={columns}, file '{p}' columns={list(df.columns)}."
            )
        frames.append(df)

    if not frames:
        raise ValueError(f"No files resolved for chromosome-split path spec: {path_spec}")
    return pd.concat(frames, axis=0, ignore_index=True)


def _check_file_or_chr_split_spec(path_spec, *, label: str = "file"):
    try:
        _resolve_chr_split_paths(path_spec, require=True)
    except Exception as e:
        raise ValueError(f"Could not find {label}: {path_spec}") from e


def _read_with_optional_header_one(file_path):
    with _open_text_maybe_gzip(file_path, "rt") as fd:
        line = fd.readline().strip()
        try:
            [float(x) for x in line.split()]
            is_header = False
        except ValueError:
            is_header = True

    if is_header:
        header = line.split()
        with _open_text_maybe_gzip(file_path, "rt") as fd:
            data = np.loadtxt(fd, skiprows=1, ndmin=2)
        return header, data

    with _open_text_maybe_gzip(file_path, "rt") as fd:
        data = np.loadtxt(fd, ndmin=2)
    return None, data


def _parse_sumdir(h2_path):
    if h2_path is None:
        raise ValueError("h2_path must be provided.")
    if _is_chr_split_spec(h2_path):
        split_files = _resolve_chr_split_paths(h2_path, require=False)
        if split_files:
            return [_normalize_path_spec(h2_path)]

        split_dirs = _resolve_chr_split_dirs(h2_path, require=False)
        if split_dirs:
            file_sets = []
            for d in split_dirs:
                names = {
                    x.name
                    for x in Path(d).iterdir()
                    if x.is_file() and not x.name.startswith(".")
                }
                file_sets.append(names)
            common = set.intersection(*file_sets) if file_sets else set()
            union = set.union(*file_sets) if file_sets else set()
            if not common:
                raise ValueError(f"No common files found in h2_path chromosome-split directories: {h2_path}")
            if common != union:
                missing_by_dir = []
                for d, names in zip(split_dirs, file_sets):
                    missing = sorted(union - names)
                    if missing:
                        missing_by_dir.append(f"{d}: missing {missing[:5]}")
                detail = "; ".join(missing_by_dir[:5])
                raise ValueError(
                    f"h2_path chromosome-split directories do not contain the same files: {h2_path}. "
                    f"{detail}"
                )
            base = _normalize_path_spec(h2_path).rstrip("/")
            return [f"{base}/{name}" for name in sorted(common)]

        _check_file_or_chr_split_spec(h2_path, label="h2 chromosome-split sumstats")
        return [_normalize_path_spec(h2_path)]
    p = Path(h2_path)
    if p.is_dir():
        out = sorted(
            [str(x) for x in p.iterdir() if x.is_file() and not x.name.startswith(".")]
        )
        if not out:
            raise ValueError(f"No files found in h2_path directory: {h2_path}")
        return out
    if p.is_file():
        return [str(p)]
    raise ValueError(f"Could not resolve h2_path: {h2_path}")


def _parse_rg_pair(rg):
    if rg is None:
        raise ValueError("--rg must be provided.")
    parts = [x.strip() for x in str(rg).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--rg must be exactly two comma-separated sumstats paths.")
    for p in parts:
        _check_file_or_chr_split_spec(p, label="sumstats file")
    return parts


def _phen_name_from_path(path: str) -> str:
    name = os.path.basename(path)
    for suf in (".sumstats.gz", ".sumstats", ".txt.gz", ".txt", ".tsv.gz", ".tsv", ".gz"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


def _is_rg_manifest_arg(raw) -> bool:
    if raw is None:
        return False
    s = str(raw).strip()
    return ("," not in s) and Path(s).is_file()


def _sanitize_output_component(text: str, *, default: str = "trait") -> str:
    s = str(text).strip()
    if s == "":
        return default
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or default


def _pair_output_stem(phen1: str, phen2: str) -> str:
    return f"{_sanitize_output_component(phen1)}.{_sanitize_output_component(phen2)}"


def _parse_verbose(verbose) -> int:
    if isinstance(verbose, str):
        s = verbose.strip().lower()
        if s in ("0", "false", "none", "off", "jack", "normeq"):
            return 0
        if s in ("1", "true", "yes", "on"):
            return 1
        if s in ("2", "all", "both", "max"):
            return 2
        return 1
    try:
        return 2 if int(verbose) >= 2 else (1 if int(verbose) == 1 else 0)
    except Exception:
        return 1 if bool(verbose) else 0


def _parse_verbose_outputs(verbose):
    if isinstance(verbose, str):
        s = verbose.strip().lower()
        if s == "jack":
            return True, False
        if s == "normeq":
            return False, True
        if s in ("2", "all", "both", "max"):
            return True, True
        return False, False

    try:
        return (True, True) if int(verbose) >= 2 else (False, False)
    except Exception:
        return False, False


def _resolve_chisq_threshold(nmax: float, raw=None):
    if raw is None:
        return None, "none"
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "auto":
            return float(max(80.0, 0.001 * float(nmax))), "auto"
        if s in ("none", "null"):
            return None, "none"
        try:
            return float(s), "manual"
        except Exception as e:
            raise ValueError(f"Invalid chisq_threshold string value: {raw!r}") from e
    try:
        return float(raw), "manual"
    except Exception as e:
        raise ValueError(f"Invalid chisq_threshold value: {raw!r}") from e


# -----------------------
# Batched vectorized trace estimators
# -----------------------

def _calc_trace_from_ld_batch(ldsum, n, m1, m2, delta=None):
    ldsum = np.asarray(ldsum, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)

    denom = m1 * m2

    # (..., K, K) with broadcasted compute
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ldsum * (n**2) / denom + n

    # where denom <= 0 (or NaN), fall back to n
    valid = (denom > 0) & np.isfinite(denom)
    out = np.where(valid, out, n)

    # ---- optional δ-based correction: out_{kℓ} ← out_{kℓ} - N * δ_{kℓ} ----
    if delta is not None:
        delta = np.asarray(delta, dtype=np.float64)
        if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
            raise ValueError(f"delta must be KxK; got shape {delta.shape}")

        # Get scalar N (first element of n broadcast)
        n_scalar = float(n.ravel()[0])

        # Correction term: N * δ_{kℓ}, broadcast across jackknife rows, but
        # only for valid (denom > 0) entries.
        corr = n_scalar * delta  # (K, K)

        # Broadcast corr and apply only where denom is valid
        out = np.where(valid, out - corr, out)

    return out


def _calc_rg_trace_from_ld_batch(ldsum, n1, n2, m1, m2):
    ldsum = np.asarray(ldsum, dtype=np.float64)
    n1 = float(n1)
    n2 = float(n2)
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)

    denom = m1 * m2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ldsum * (n1 * n2) / denom

    valid = (denom > 0) & np.isfinite(denom)
    out = np.where(valid, out, 0.0)
    return out


def _sym_clip_weight(v1, v2, c12):
    denom_w = float(v1 + v2 - 2.0 * c12)
    if (not np.isfinite(denom_w)) or (denom_w <= 0.0):
        return 0.5

    w = float((v2 - c12) / denom_w)
    if not np.isfinite(w):
        return 0.5
    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return w


def _sym_center_1d(vals, center="mean", full_value=None, weights=None):
    vals = np.asarray(vals, dtype=np.float64).ravel()

    if center == "full":
        fv = float(full_value)
        return fv if np.isfinite(fv) else np.nan

    if vals.size == 0:
        return np.nan

    if center == "mean":
        if weights is None:
            return float(np.mean(vals))
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != vals.size:
            raise ValueError("weights/vals length mismatch in _sym_center_1d.")
        sw = float(np.sum(w))
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan
        return float(np.sum(w * vals) / sw)

    if center == "median":
        return float(np.median(vals))

    raise ValueError("center must be one of {'full','mean','median'}")


def _equal_weight_pair_moments(x_rep, y_rep, x_full, y_full, center="mean", nan_policy="omit"):
    x_rep = np.asarray(x_rep, dtype=np.float64).ravel()
    y_rep = np.asarray(y_rep, dtype=np.float64).ravel()

    if x_rep.size != y_rep.size:
        raise ValueError("x_rep/y_rep length mismatch.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    if nan_policy == "propagate":
        if (not np.isfinite(x_full)) or (not np.isfinite(y_full)):
            return np.nan, np.nan, np.nan
        if (not np.isfinite(x_rep).all()) or (not np.isfinite(y_rep).all()):
            return np.nan, np.nan, np.nan
        xv = x_rep
        yv = y_rep
    else:
        valid = np.isfinite(x_rep) & np.isfinite(y_rep)
        if valid.sum() <= 1:
            return np.nan, np.nan, np.nan
        xv = x_rep[valid]
        yv = y_rep[valid]

    cx = _sym_center_1d(xv, center=center, full_value=x_full, weights=None)
    cy = _sym_center_1d(yv, center=center, full_value=y_full, weights=None)
    if (not np.isfinite(cx)) or (not np.isfinite(cy)):
        return np.nan, np.nan, np.nan

    n = int(xv.size)
    if n <= 1:
        return np.nan, np.nan, np.nan

    dx = xv - cx
    dy = yv - cy

    # exact covariance analogue of the equal-weight jackknife SE branch:
    # Var = ((n-1)/n) * sum(diff^2)
    fac = float(n - 1) / float(n)
    v1 = fac * float(np.sum(dx * dx))
    v2 = fac * float(np.sum(dy * dy))
    c12 = fac * float(np.sum(dx * dy))
    return v1, v2, c12


def _weighted_delete1_pair_moments(
    x_rep,
    y_rep,
    x_full,
    y_full,
    block_sizes,
    center="mean",
    nan_policy="omit",
):
    """
    Exact covariance analogue of the weighted delete-1 pseudovalue branch used in
    _calc_jackknife_se(..., use_pseudovalues=True).

    IMPORTANT:
    This assumes delete-1 over a true partition with sum(m_b)=M.
    """
    x_rep = np.asarray(x_rep, dtype=np.float64).ravel()
    y_rep = np.asarray(y_rep, dtype=np.float64).ravel()
    m = np.asarray(block_sizes, dtype=np.float64).ravel()

    if x_rep.size != y_rep.size:
        raise ValueError("x_rep/y_rep length mismatch.")
    if m.size != x_rep.size:
        raise ValueError("block_sizes length mismatch.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    good = np.isfinite(m) & (m > 0.0)
    if good.sum() <= 1:
        return np.nan, np.nan, np.nan

    x = x_rep[good]
    y = y_rep[good]
    m = m[good]

    M = float(np.sum(m))
    if not (np.isfinite(M) and M > 0.0):
        return np.nan, np.nan, np.nan

    w = m / M  # sums to 1 on the retained partition blocks

    PVx = (M * float(x_full) - (M - m) * x) / m
    PVy = (M * float(y_full) - (M - m) * y) / m

    if nan_policy == "propagate":
        if (not np.isfinite(x_full)) or (not np.isfinite(y_full)):
            return np.nan, np.nan, np.nan
        if (not np.isfinite(PVx).all()) or (not np.isfinite(PVy).all()):
            return np.nan, np.nan, np.nan

        pvx = PVx
        pvy = PVy
        ww = w  # already sums to 1
    else:
        valid = np.isfinite(PVx) & np.isfinite(PVy)
        if valid.sum() <= 1:
            return np.nan, np.nan, np.nan

        pvx = PVx[valid]
        pvy = PVy[valid]
        ww = w[valid]

        sw = float(np.sum(ww))
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan, np.nan, np.nan
        ww = ww / sw  # exact mirror of _calc_jackknife_se(..., nan_policy='omit')

    cx = _sym_center_1d(pvx, center=center, full_value=x_full, weights=(ww if center == "mean" else None))
    cy = _sym_center_1d(pvy, center=center, full_value=y_full, weights=(ww if center == "mean" else None))
    if (not np.isfinite(cx)) or (not np.isfinite(cy)):
        return np.nan, np.nan, np.nan

    dx = pvx - cx
    dy = pvy - cy

    denom = 1.0 - ww
    if np.any((~np.isfinite(denom)) | (denom <= 0.0)):
        return np.nan, np.nan, np.nan

    # exact covariance analogue of the SE formula:
    # Var = sum( w^2/(1-w) * diff^2 )
    alpha = (ww * ww) / denom
    if not np.isfinite(alpha).all():
        return np.nan, np.nan, np.nan

    # same effective-dof guard as the SE logic
    w2 = float(np.sum(ww * ww))
    n_eff = (1.0 / w2) if (w2 > 0.0 and np.isfinite(w2)) else 0.0
    if not (n_eff > 1.0):
        return np.nan, np.nan, np.nan

    v1 = float(np.sum(alpha * (dx * dx)))
    v2 = float(np.sum(alpha * (dy * dy)))
    c12 = float(np.sum(alpha * (dx * dy)))

    if (not np.isfinite(v1)) or (not np.isfinite(v2)) or (not np.isfinite(c12)):
        return np.nan, np.nan, np.nan
    return v1, v2, c12


def _unit_pseudovalue_pair_moments(
    pvx_all,
    pvy_all,
    x_full,
    y_full,
    unit_sizes,
    M,
    good_u_base,
    center="mean",
    nan_policy="omit",
):
    """
    Exact covariance analogue of the unit-level unequal-size delete-1 formula used
    after reconstructing unit pseudovalues in _calc_jackknife_se_from_delete_sets.
    """
    pvx_all = np.asarray(pvx_all, dtype=np.float64).ravel()
    pvy_all = np.asarray(pvy_all, dtype=np.float64).ravel()
    unit_sizes = np.asarray(unit_sizes, dtype=np.float64).ravel()
    good_u_base = np.asarray(good_u_base, dtype=bool).ravel()

    if not (pvx_all.size == pvy_all.size == unit_sizes.size == good_u_base.size):
        raise ValueError("length mismatch in _unit_pseudovalue_pair_moments.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    if nan_policy == "propagate":
        if (not np.isfinite(x_full)) or (not np.isfinite(y_full)):
            return np.nan, np.nan, np.nan
        if not (np.isfinite(pvx_all[good_u_base]).all() and np.isfinite(pvy_all[good_u_base]).all()):
            return np.nan, np.nan, np.nan

        valid = good_u_base
        pvx = pvx_all[valid]
        pvy = pvy_all[valid]
        ww = unit_sizes[valid] / float(M)  # sums to 1 on the valid partition
    else:
        valid = good_u_base & np.isfinite(pvx_all) & np.isfinite(pvy_all)
        if valid.sum() <= 1:
            return np.nan, np.nan, np.nan

        pvx = pvx_all[valid]
        pvy = pvy_all[valid]
        ww = unit_sizes[valid] / float(M)

        sw = float(np.sum(ww))
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan, np.nan, np.nan
        ww = ww / sw  # exact mirror of omit behavior in the SE code

    cx = _sym_center_1d(pvx, center=center, full_value=x_full, weights=(ww if center == "mean" else None))
    cy = _sym_center_1d(pvy, center=center, full_value=y_full, weights=(ww if center == "mean" else None))
    if (not np.isfinite(cx)) or (not np.isfinite(cy)):
        return np.nan, np.nan, np.nan

    dx = pvx - cx
    dy = pvy - cy

    denom = 1.0 - ww
    if np.any((~np.isfinite(denom)) | (denom <= 0.0)):
        return np.nan, np.nan, np.nan

    alpha = (ww * ww) / denom
    if not np.isfinite(alpha).all():
        return np.nan, np.nan, np.nan

    w2 = float(np.sum(ww * ww))
    n_eff = (1.0 / w2) if (w2 > 0.0 and np.isfinite(w2)) else 0.0
    if not (n_eff > 1.0):
        return np.nan, np.nan, np.nan

    v1 = float(np.sum(alpha * (dx * dx)))
    v2 = float(np.sum(alpha * (dy * dy)))
    c12 = float(np.sum(alpha * (dx * dy)))

    if (not np.isfinite(v1)) or (not np.isfinite(v2)) or (not np.isfinite(c12)):
        return np.nan, np.nan, np.nan
    return v1, v2, c12


def _direct_delete_d_pair_moments(
    x_rep,
    y_rep,
    x_full,
    y_full,
    m_del,
    M,
    good_rep_base,
    center="mean",
    nan_policy="omit",
):
    """
    Exact covariance analogue of the direct delete-d fallback used in
    _calc_jackknife_se_from_delete_sets when the delete-set design is underidentified.
    """
    x_rep = np.asarray(x_rep, dtype=np.float64).ravel()
    y_rep = np.asarray(y_rep, dtype=np.float64).ravel()
    m_del = np.asarray(m_del, dtype=np.float64).ravel()
    good_rep_base = np.asarray(good_rep_base, dtype=bool).ravel()

    if not (x_rep.size == y_rep.size == m_del.size == good_rep_base.size):
        raise ValueError("length mismatch in _direct_delete_d_pair_moments.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    finite_xy = np.isfinite(x_rep) & np.isfinite(y_rep)

    if nan_policy == "propagate":
        if (not np.isfinite(x_full)) or (not np.isfinite(y_full)):
            return np.nan, np.nan, np.nan
        if not np.all((~good_rep_base) | finite_xy):
            return np.nan, np.nan, np.nan
        valid = good_rep_base
    else:
        valid = good_rep_base & finite_xy
        if valid.sum() <= 1:
            return np.nan, np.nan, np.nan

    xv = x_rep[valid]
    yv = y_rep[valid]
    sc = ((float(M) - m_del[valid]) / m_del[valid]).astype(np.float64)

    if (not np.isfinite(sc).all()) or np.any(sc <= 0.0):
        return np.nan, np.nan, np.nan

    cx = _sym_center_1d(xv, center=center, full_value=x_full, weights=None)
    cy = _sym_center_1d(yv, center=center, full_value=y_full, weights=None)
    if (not np.isfinite(cx)) or (not np.isfinite(cy)):
        return np.nan, np.nan, np.nan

    dx = xv - cx
    dy = yv - cy

    # exact covariance analogue of the fallback SE formula:
    # Var = mean( sc * diff^2 )
    v1 = float(np.mean(sc * (dx * dx)))
    v2 = float(np.mean(sc * (dy * dy)))
    c12 = float(np.mean(sc * (dx * dy)))

    if (not np.isfinite(v1)) or (not np.isfinite(v2)) or (not np.isfinite(c12)):
        return np.nan, np.nan, np.nan
    return v1, v2, c12


def _delete_sets_pair_moments(
    x_rep,
    y_rep,
    x_full,
    y_full,
    D,
    unit_sizes,
    center="mean",
    nan_policy="omit",
):
    """
    Exact covariance analogue of _calc_jackknife_se_from_delete_sets, specialized to
    one off-diagonal pair (x = theta_{k,l}, y = theta_{l,k}).

    This is used for:
      - exact LOCO delete-1
      - sampled delete-1
      - general delete-d
    """
    x_rep = np.asarray(x_rep, dtype=np.float64).ravel()
    y_rep = np.asarray(y_rep, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64, order="C")
    unit_sizes = np.asarray(unit_sizes, dtype=np.float64).ravel()

    if D.ndim != 2:
        raise ValueError("D must be 2D (R, U).")
    R, U = D.shape

    if x_rep.size != R or y_rep.size != R:
        raise ValueError("replicate length mismatch with D.")
    if unit_sizes.size != U:
        raise ValueError("unit_sizes length mismatch with D.")

    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    M = float(np.sum(unit_sizes))
    if not (np.isfinite(M) and M > 0.0):
        return np.nan, np.nan, np.nan

    good_u_base = np.isfinite(unit_sizes) & (unit_sizes > 0.0) & (unit_sizes < M)
    use_u = np.flatnonzero(good_u_base)
    if use_u.size <= 1:
        return np.nan, np.nan, np.nan

    m_del = D @ unit_sizes
    good_rep_base = np.isfinite(m_del) & (m_del > 0.0) & (m_del < M)
    if not np.any(good_rep_base):
        return np.nan, np.nan, np.nan

    mask = D > 0.5
    exact_loco = (
        R == U
        and np.all(mask.sum(axis=1) == 1)
        and np.all(mask.sum(axis=0) == 1)
    )

    # ------------------------------------------------------------
    # Case A: exact delete-1 LOCO full set
    # ------------------------------------------------------------
    if exact_loco:
        if nan_policy == "propagate":
            if (not np.isfinite(x_full)) or (not np.isfinite(y_full)):
                return np.nan, np.nan, np.nan
            if (not np.isfinite(x_rep).all()) or (not np.isfinite(y_rep).all()):
                return np.nan, np.nan, np.nan

        unit_of_rep = mask.argmax(axis=1)
        rep_for_unit = np.empty(U, dtype=np.int64)
        rep_for_unit[unit_of_rep] = np.arange(R, dtype=np.int64)

        pvx_all = np.full(U, np.nan, dtype=np.float64)
        pvy_all = np.full(U, np.nan, dtype=np.float64)

        for u in use_u:
            r = int(rep_for_unit[u])
            thx = float(x_rep[r])
            thy = float(y_rep[r])
            if np.isfinite(thx):
                mu = float(unit_sizes[u])
                pvx_all[u] = (M * float(x_full) - (M - mu) * thx) / mu
            if np.isfinite(thy):
                mu = float(unit_sizes[u])
                pvy_all[u] = (M * float(y_full) - (M - mu) * thy) / mu

        return _unit_pseudovalue_pair_moments(
            pvx_all=pvx_all,
            pvy_all=pvy_all,
            x_full=x_full,
            y_full=y_full,
            unit_sizes=unit_sizes,
            M=M,
            good_u_base=good_u_base,
            center=center,
            nan_policy=nan_policy,
        )

    # ------------------------------------------------------------
    # Case B: general delete-d / sampled delete-1
    # ------------------------------------------------------------
    finite_xy = np.isfinite(x_rep) & np.isfinite(y_rep)
    if nan_policy == "propagate":
        if (not np.isfinite(x_full)) or (not np.isfinite(y_full)):
            return np.nan, np.nan, np.nan
        if not np.all((~good_rep_base) | finite_xy):
            return np.nan, np.nan, np.nan
        valid_rep = good_rep_base
    else:
        valid_rep = good_rep_base & finite_xy
        if valid_rep.sum() <= 1:
            return np.nan, np.nan, np.nan

    D_use = D[:, use_u]
    Dv = D_use[valid_rep, :]
    U_use = int(Dv.shape[1])

    # If underidentified, use the exact covariance analogue of the fallback
    if (Dv.shape[0] < U_use) or (np.linalg.matrix_rank(Dv) < U_use):
        return _direct_delete_d_pair_moments(
            x_rep=x_rep,
            y_rep=y_rep,
            x_full=x_full,
            y_full=y_full,
            m_del=m_del,
            M=M,
            good_rep_base=good_rep_base,
            center=center,
            nan_policy=nan_policy,
        )

    m_u = unit_sizes[use_u]
    yx = M * float(x_full) - (M - m_del[valid_rep]) * x_rep[valid_rep]
    yy = M * float(y_full) - (M - m_del[valid_rep]) * y_rep[valid_rep]

    RHS = np.column_stack([yx, yy])  # (Rv, 2)
    AtA = Dv.T @ Dv
    AtY = Dv.T @ RHS

    try:
        G = np.linalg.solve(AtA, AtY)
    except np.linalg.LinAlgError:
        tr = float(np.trace(AtA))
        lam = 1e-10 * (tr / U_use if (np.isfinite(tr) and tr > 0.0) else 1.0)
        try:
            G = np.linalg.solve(AtA + lam * np.eye(U_use, dtype=np.float64), AtY)
        except np.linalg.LinAlgError:
            G = np.linalg.lstsq(Dv, RHS, rcond=None)[0]

    pvx_use = G[:, 0] / m_u
    pvy_use = G[:, 1] / m_u

    pvx_all = np.full(U, np.nan, dtype=np.float64)
    pvy_all = np.full(U, np.nan, dtype=np.float64)
    pvx_all[use_u] = pvx_use
    pvy_all[use_u] = pvy_use

    return _unit_pseudovalue_pair_moments(
        pvx_all=pvx_all,
        pvy_all=pvy_all,
        x_full=x_full,
        y_full=y_full,
        unit_sizes=unit_sizes,
        M=M,
        good_u_base=good_u_base,
        center=center,
        nan_policy=nan_policy,
    )


def _exact_loco_sym_weights_finite(trace_KK, D, unit_sizes, center="mean"):
    """Optimized exact-LOCO weights for the finite, fully active case."""
    trace_KK = np.asarray(trace_KK, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64, order="C")
    unit_sizes = np.asarray(unit_sizes, dtype=np.float64).ravel()

    B = int(trace_KK.shape[0] - 1)
    K = int(trace_KK.shape[1])
    U = int(unit_sizes.size)
    mask = D > 0.5
    if (
        B != U
        or D.shape != (B, U)
        or not np.all(mask.sum(axis=1) == 1)
        or not np.all(mask.sum(axis=0) == 1)
        or not np.isfinite(trace_KK).all()
    ):
        return None

    M = float(np.sum(unit_sizes))
    if (
        not np.isfinite(M)
        or M <= 0.0
        or not np.isfinite(unit_sizes).all()
        or np.any(unit_sizes <= 0.0)
        or np.any(unit_sizes >= M)
    ):
        return None

    unit_of_rep = mask.argmax(axis=1)
    rep_for_unit = np.empty(U, dtype=np.int64)
    rep_for_unit[unit_of_rep] = np.arange(B, dtype=np.int64)

    full = trace_KK[B]
    jack_by_unit = trace_KK[:B][rep_for_unit]
    ww = unit_sizes / M
    ww = ww / float(np.sum(ww))
    alpha = (ww * ww) / (1.0 - ww)
    w2 = float(np.sum(ww * ww))
    n_eff = (1.0 / w2) if (w2 > 0.0 and np.isfinite(w2)) else 0.0
    if not (n_eff > 1.0):
        return np.full((K, K), 0.5, dtype=np.float64)

    weights = np.full((K, K), 0.5, dtype=np.float64)
    remaining = M - unit_sizes
    center_weight_sum = float(np.sum(ww)) if center == "mean" else None
    for k in range(K):
        for l in range(k + 1, K):
            x_full = float(full[k, l])
            y_full = float(full[l, k])
            pvx = (M * x_full - remaining * jack_by_unit[:, k, l]) / unit_sizes
            pvy = (M * y_full - remaining * jack_by_unit[:, l, k]) / unit_sizes
            if center == "mean":
                cx = float(np.sum(ww * pvx) / center_weight_sum)
                cy = float(np.sum(ww * pvy) / center_weight_sum)
            elif center == "full":
                cx = x_full
                cy = y_full
            elif center == "median":
                cx = float(np.median(pvx))
                cy = float(np.median(pvy))
            else:
                raise ValueError("center must be one of {'full','mean','median'}")
            dx = pvx - cx
            dy = pvy - cy
            v1 = float(np.sum(alpha * (dx * dx)))
            v2 = float(np.sum(alpha * (dy * dy)))
            c12 = float(np.sum(alpha * (dx * dy)))
            weight = _sym_clip_weight(v1, v2, c12)
            weights[k, l] = weight
            weights[l, k] = 1.0 - weight
    return weights


def estimate_offdiag_variances_from_jackknife(trace_KK, center="mean", nan_policy="omit"):
    """
    Legacy equal-weight off-diagonal symmetrization weights.

    This now mirrors the exact covariance analogue of the equal-weight branch of
    _calc_jackknife_se(..., use_pseudovalues=False), with consistent nan handling.
    """
    trace_KK = np.asarray(trace_KK, dtype=np.float64)
    if trace_KK.ndim != 3:
        raise ValueError("trace_KK must have shape (B+1, K, K)")

    B_plus, K, K2 = trace_KK.shape
    if K != K2:
        raise ValueError("trace_KK last two dimensions must be equal (KxK)")

    if center not in {"full", "mean", "median"}:
        raise ValueError("center must be one of {'full','mean','median'}")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    B = B_plus - 1

    var1 = np.zeros((K, K), dtype=np.float64)
    var2 = np.zeros((K, K), dtype=np.float64)
    cov12 = np.zeros((K, K), dtype=np.float64)
    w_opt = np.full((K, K), 0.5, dtype=np.float64)

    if B <= 1:
        return var1, var2, cov12, w_opt

    jack = trace_KK[:B]
    full = trace_KK[B]

    for k in range(K):
        for l in range(k + 1, K):
            x = jack[:, k, l]
            y = jack[:, l, k]
            x_full = float(full[k, l])
            y_full = float(full[l, k])

            v1, v2, c12 = _equal_weight_pair_moments(
                x_rep=x,
                y_rep=y,
                x_full=x_full,
                y_full=y_full,
                center=center,
                nan_policy=nan_policy,
            )

            if np.isfinite(v1) and np.isfinite(v2) and np.isfinite(c12):
                var1[k, l] = var1[l, k] = v1
                var2[k, l] = var2[l, k] = v2
                cov12[k, l] = cov12[l, k] = c12
                w = _sym_clip_weight(v1, v2, c12)
            else:
                w = 0.5

            w_opt[k, l] = w
            w_opt[l, k] = 1.0 - w

    for k in range(K):
        var1[k, k] = 0.0
        var2[k, k] = 0.0
        cov12[k, k] = 0.0
        w_opt[k, k] = 0.5

    return var1, var2, cov12, w_opt


def symmetrize_trace_with_jackknife(
    trace_KK,
    logger=None,
    verbose=False,
    jk_block_sizes=None,      # delete-1 partition blocks
    jk_delete_matrix=None,    # (R, U) delete incidence matrix for delete-set designs
    jk_unit_sizes=None,       # (U,) unit sizes for delete-set designs
    jk_n_units=None,          # optional sanity/logging only
    jk_delete_d: int = 1,
    center: str = "mean",
    nan_policy: str = "omit",
    exact_loco_fast: bool = False,
):
    """
    Symmetrize off-diagonal trace entries using pair-specific jackknife-optimal weights.

    Branches
    --------
    1) delete-set branch (preferred when jk_delete_matrix + jk_unit_sizes are provided):
       Mirrors the covariance analogue of _calc_jackknife_se_from_delete_sets.
       This handles:
         - exact LOCO delete-1
         - sampled delete-1
         - general delete-d

    2) weighted delete-1 partition branch (jk_block_sizes provided):
       Mirrors the covariance analogue of the weighted pseudovalue branch in
       _calc_jackknife_se(..., use_pseudovalues=True).

    3) legacy equal-weight branch:
       Mirrors the covariance analogue of the equal-weight jackknife SE branch.

    IMPORTANT
    ---------
    For sampled delete-1 / delete-d designs, do NOT rely on jk_n_units alone.
    Pass jk_delete_matrix and jk_unit_sizes explicitly, otherwise this function
    raises instead of silently falling back to equal-weight logic.
    """
    trace_KK = np.asarray(trace_KK, dtype=np.float64)
    if trace_KK.ndim != 3:
        raise ValueError("trace_KK must have shape (B+1, K, K)")

    B_plus, K, K2 = trace_KK.shape
    if K != K2:
        raise ValueError("trace_KK last two dimensions must be equal (KxK)")

    if center not in {"full", "mean", "median"}:
        raise ValueError("center must be one of {'full','mean','median'}")
    if nan_policy not in {"omit", "propagate"}:
        raise ValueError("nan_policy must be one of {'omit','propagate'}")

    B = B_plus - 1

    if B <= 0:
        sym = trace_KK.copy()
        full = sym[-1]
        for k in range(K):
            for l in range(k + 1, K):
                v = 0.5 * (full[k, l] + full[l, k])
                full[k, l] = full[l, k] = v
        sym[-1] = full
        return sym

    sym = trace_KK.copy()

    if logger is not None:
        full_before = trace_KK[B]
        tri = np.triu_indices(K, k=1)
        max_asym_before = float(np.abs(full_before - full_before.T)[tri].max(initial=0.0))

    # ----------------------------
    # choose weight-estimation mode
    # ----------------------------
    has_delete_set = (jk_delete_matrix is not None) or (jk_unit_sizes is not None)
    if has_delete_set:
        if (jk_delete_matrix is None) or (jk_unit_sizes is None):
            raise ValueError("Provide both jk_delete_matrix and jk_unit_sizes together.")

        D = np.asarray(jk_delete_matrix, dtype=np.float64, order="C")
        unit_sizes = np.asarray(jk_unit_sizes, dtype=np.float64).ravel()

        if D.ndim != 2:
            raise ValueError("jk_delete_matrix must be 2D (R, U).")
        if D.shape[0] != B:
            raise ValueError(f"jk_delete_matrix must have R={B} rows; got {D.shape[0]}.")
        if unit_sizes.size != D.shape[1]:
            raise ValueError(
                f"jk_unit_sizes must have length U={D.shape[1]}; got {unit_sizes.size}."
            )
        if jk_n_units is not None and int(jk_n_units) != int(D.shape[1]):
            raise ValueError(
                f"jk_n_units={jk_n_units} inconsistent with delete matrix U={D.shape[1]}."
            )

        w_opt = None
        if exact_loco_fast:
            w_opt = _exact_loco_sym_weights_finite(
                trace_KK,
                D,
                unit_sizes,
                center=center,
            )
        fast_exact_loco = w_opt is not None
        if w_opt is None:
            w_opt = np.full((K, K), 0.5, dtype=np.float64)
        full = trace_KK[B]
        jack = trace_KK[:B]

        if not fast_exact_loco:
            for k in range(K):
                w_opt[k, k] = 0.5

            for k in range(K):
                for l in range(k + 1, K):
                    v1, v2, c12 = _delete_sets_pair_moments(
                        x_rep=jack[:, k, l],
                        y_rep=jack[:, l, k],
                        x_full=float(full[k, l]),
                        y_full=float(full[l, k]),
                        D=D,
                        unit_sizes=unit_sizes,
                        center=center,
                        nan_policy=nan_policy,
                    )
                    w = _sym_clip_weight(v1, v2, c12) if (
                        np.isfinite(v1) and np.isfinite(v2) and np.isfinite(c12)
                    ) else 0.5

                    w_opt[k, l] = w
                    w_opt[l, k] = 1.0 - w

        if logger is not None and verbose:
            logger._log(
                f"[Trace] symmetrize: delete-set mode used "
                f"(U={D.shape[1]}, d={int(jk_delete_d)}, R={B}, center={center}, nan_policy={nan_policy})."
            )

    else:
        # If the caller claims a delete-set design but did not pass D / unit sizes,
        # do not silently fall back.
        if (jk_n_units is not None) or (int(jk_delete_d) > 1):
            raise ValueError(
                "Delete-set symmetrization now requires jk_delete_matrix and jk_unit_sizes. "
                "Do not rely on jk_n_units / jk_delete_d alone."
            )

        if jk_block_sizes is not None:
            m = np.asarray(jk_block_sizes, dtype=np.float64).ravel()
            if m.size != B:
                raise ValueError(f"jk_block_sizes must have length B={B}, got {m.size}")

            w_opt = np.full((K, K), 0.5, dtype=np.float64)
            full = trace_KK[B]
            jack = trace_KK[:B]

            for k in range(K):
                w_opt[k, k] = 0.5

            for k in range(K):
                for l in range(k + 1, K):
                    v1, v2, c12 = _weighted_delete1_pair_moments(
                        x_rep=jack[:, k, l],
                        y_rep=jack[:, l, k],
                        x_full=float(full[k, l]),
                        y_full=float(full[l, k]),
                        block_sizes=m,
                        center=center,
                        nan_policy=nan_policy,
                    )
                    w = _sym_clip_weight(v1, v2, c12) if (
                        np.isfinite(v1) and np.isfinite(v2) and np.isfinite(c12)
                    ) else 0.5

                    w_opt[k, l] = w
                    w_opt[l, k] = 1.0 - w

            if logger is not None and verbose:
                logger._log(
                    f"[Trace] symmetrize: delete-1 weighted partition used "
                    f"(B={B}, center={center}, nan_policy={nan_policy})."
                )
        else:
            # legacy equal-weight branch
            _, _, _, w_opt = estimate_offdiag_variances_from_jackknife(
                trace_KK,
                center=center,
                nan_policy=nan_policy,
            )
            if logger is not None and verbose:
                logger._log(
                    f"[Trace] symmetrize: legacy equal-weight branch used "
                    f"(B={B}, center={center}, nan_policy={nan_policy})."
                )

    # ----------------------------
    # apply pairwise weights
    # ----------------------------
    for k in range(K):
        for l in range(k + 1, K):
            w = float(w_opt[k, l])

            x = trace_KK[:B, k, l]
            y = trace_KK[:B, l, k]
            v_jk = w * x + (1.0 - w) * y
            sym[:B, k, l] = v_jk
            sym[:B, l, k] = v_jk

            x_full = trace_KK[B, k, l]
            y_full = trace_KK[B, l, k]
            v_full = w * x_full + (1.0 - w) * y_full
            sym[B, k, l] = v_full
            sym[B, l, k] = v_full

    if logger is not None and verbose:
        full_after = sym[B]
        tri = np.triu_indices(K, k=1)
        max_asym_after = float(np.abs(full_after - full_after.T)[tri].max(initial=0.0))
        logger._log(
            f"[Trace] Jackknife-based symmetrization: max off-diagonal asym "
            f"(before, after) = ({max_asym_before:.4e}, {max_asym_after:.4e})"
        )

    return sym


# -----------------------
# Jackknife helpers
# -----------------------

def _calc_jackknife_se_from_delete_sets(
    alist,
    D,
    unit_sizes,
    axis=0,
    center="mean",
    nan_policy="omit",
):
    """
    Overlap-aware jackknife SE for chr-mode delete-* replicates.

    Design
    ------
    1) Exact delete-1 LOCO full set (R == U and D is a permutation of I):
       use the exact unequal-unit delete-1 pseudovalue jackknife.

    2) General delete-d / sampled delete-1:
       reconstruct unit-level pseudovalues from the overlapping delete-set equations
           M * theta_full - (M - m_S) * theta_{-S} ~= sum_{u in S} g_u
       where g_u = m_u * PV_u,
       then apply the unequal-unit delete-1 variance formula to the recovered PV_u.

       This is the overlap-aware generalization you were aiming for with your
       second implementation, but fixed so that it also works for LOCO delete-1,
       handles NaNs per-parameter, and falls back safely if the delete-set design
       is not identifiable.

    Notes
    -----
    - For equal-size units and full combinatorial delete-d, this is asymptotically
      equivalent to the usual delete-d jackknife scaling.
    - For unequal chromosome sizes, this is the safer formulation.
    """
    a = np.asarray(alist)
    est_full = np.take(a, indices=-1, axis=axis)

    # First R slices = replicates, last slice = full
    slicer = [slice(None)] * a.ndim
    slicer[axis] = slice(0, -1)
    reps = a[tuple(slicer)]
    reps = np.moveaxis(reps, axis, 0)  # (R, ...)
    R = reps.shape[0]

    if R <= 0:
        return est_full, np.full_like(np.asarray(est_full, dtype=np.float64), np.nan, dtype=np.float64)

    D = np.asarray(D, dtype=np.float64, order="C")
    if D.ndim != 2:
        raise ValueError("D must be 2D with shape (R, U).")
    if D.shape[0] != R:
        raise ValueError(f"D has R={D.shape[0]} rows but alist has R={R} replicates.")

    U = int(D.shape[1])

    unit_sizes = np.asarray(unit_sizes, dtype=np.float64).ravel()
    if unit_sizes.size != U:
        raise ValueError(f"unit_sizes must have length U={U}, got {unit_sizes.size}.")

    M = float(np.sum(unit_sizes))
    if not (np.isfinite(M) and M > 0.0):
        raise ValueError(f"Total unit size M must be positive finite; got {M}.")

    reps_flat = np.asarray(reps, dtype=np.float64).reshape(R, -1)  # (R, P)
    full_flat = np.asarray(est_full, dtype=np.float64).reshape(-1) # (P,)
    P = full_flat.size

    m_del = D @ unit_sizes  # (R,)

    good_rep_base = np.isfinite(m_del) & (m_del > 0.0) & (m_del < M)
    good_u_base = np.isfinite(unit_sizes) & (unit_sizes > 0.0) & (unit_sizes < M)

    use_u = np.flatnonzero(good_u_base)
    if use_u.size == 0:
        return est_full, np.full_like(np.asarray(est_full, dtype=np.float64), np.nan, dtype=np.float64)

    m_u = unit_sizes[use_u]
    w_u = m_u / M

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _center_on_values(vals, weights, full_scalar):
        vals = np.asarray(vals, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        if center == "full":
            return float(full_scalar)

        finite = np.isfinite(vals) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(finite):
            return np.nan

        vf = vals[finite]
        wf = weights[finite]
        sw = float(wf.sum())
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan
        wf = wf / sw

        if center == "mean":
            return float(np.sum(wf * vf))
        elif center == "median":
            return float(np.median(vf))
        else:
            raise ValueError("center must be one of {'full','mean','median'}")

    def _se_from_unit_pseudovalues(pv_all, full_scalar):
        """
        pv_all : shape (U,)
        """
        pv_all = np.asarray(pv_all, dtype=np.float64)

        if nan_policy == "propagate":
            valid = good_u_base
            if not (np.isfinite(full_scalar) and np.all(np.isfinite(pv_all[valid]))):
                return np.nan

            vals = pv_all[valid]
            ww = (unit_sizes[valid] / M).astype(np.float64)
            ctr = _center_on_values(vals, ww, full_scalar)
            if not np.isfinite(ctr):
                return np.nan

            diffs = vals - ctr
            denom = 1.0 - ww
            with np.errstate(divide="ignore", invalid="ignore"):
                term = (ww * ww / denom) * (diffs * diffs)
            var = float(np.sum(term))
            return float(np.sqrt(var)) if np.isfinite(var) else np.nan

        # omit
        valid = good_u_base & np.isfinite(pv_all)
        if not np.any(valid):
            return np.nan

        vals = pv_all[valid]
        ww = (unit_sizes[valid] / M).astype(np.float64)
        sw = float(ww.sum())
        if not (np.isfinite(sw) and sw > 0.0):
            return np.nan
        ww = ww / sw

        ctr = _center_on_values(vals, ww, full_scalar)
        if not np.isfinite(ctr):
            return np.nan

        diffs = vals - ctr
        denom = 1.0 - ww
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (ww * ww / denom) * (diffs * diffs)
        term = np.where(np.isfinite(term), term, 0.0)

        w2 = float(np.sum(ww * ww))
        n_eff = (1.0 / w2) if (w2 > 0.0 and np.isfinite(w2)) else 0.0
        var = float(np.sum(term))

        if not (np.isfinite(var) and n_eff > 1.0):
            return np.nan
        return float(np.sqrt(var))

    def _se_direct_delete_d(rep_vals, full_scalar):
        """
        Fallback only if the delete-set design is underidentified for pseudovalue
        reconstruction. This is not the preferred path for your chromosome setting.
        """
        rep_vals = np.asarray(rep_vals, dtype=np.float64)
        valid = good_rep_base & np.isfinite(rep_vals)

        if nan_policy == "propagate":
            if not (np.isfinite(full_scalar) and np.all(valid)):
                return np.nan
            x = rep_vals
            sc = (M - m_del) / m_del
        else:
            if not np.any(valid):
                return np.nan
            x = rep_vals[valid]
            sc = ((M - m_del[valid]) / m_del[valid]).astype(np.float64)

        if center == "full":
            ctr = float(full_scalar)
        elif center == "mean":
            ctr = float(np.mean(x))
        elif center == "median":
            ctr = float(np.median(x))
        else:
            raise ValueError("center must be one of {'full','mean','median'}")

        diffs = x - ctr
        var = float(np.mean(sc * diffs * diffs))
        return float(np.sqrt(var)) if np.isfinite(var) else np.nan

    # ------------------------------------------------------------------
    # Detect exact delete-1 LOCO full set: D is a permutation of I
    # ------------------------------------------------------------------
    mask = D > 0.5
    exact_loco = (
        R == U
        and np.all(mask.sum(axis=1) == 1)
        and np.all(mask.sum(axis=0) == 1)
    )

    se_flat = np.full(P, np.nan, dtype=np.float64)

    # ------------------------------------------------------------------
    # Case A: exact LOCO delete-1 (exact unequal-unit pseudovalues)
    # ------------------------------------------------------------------
    if exact_loco:
        unit_of_rep = mask.argmax(axis=1)  # replicate r deletes unit unit_of_rep[r]
        rep_for_unit = np.empty(U, dtype=np.int64)  # inverse map: unit u -> replicate index
        rep_for_unit[unit_of_rep] = np.arange(R, dtype=np.int64)

        for p in range(P):
            full_p = float(full_flat[p])
            rep_p = reps_flat[:, p]

            if nan_policy == "propagate" and (not np.isfinite(full_p) or not np.all(np.isfinite(rep_p))):
                se_flat[p] = np.nan
                continue
            if not np.isfinite(full_p):
                se_flat[p] = np.nan
                continue

            pv_all = np.full(U, np.nan, dtype=np.float64)
            for u in use_u:
                r = int(rep_for_unit[u])
                th_r = rep_p[r]
                if np.isfinite(th_r):
                    mu = float(unit_sizes[u])
                    pv_all[u] = (M * full_p - (M - mu) * th_r) / mu

            se_flat[p] = _se_from_unit_pseudovalues(pv_all, full_p)

        return est_full, se_flat.reshape(est_full.shape)

    # ------------------------------------------------------------------
    # Case B: general delete-d / sampled delete-1
    # overlap-aware reconstruction of unit pseudovalues
    # ------------------------------------------------------------------
    D_use = D[:, use_u]  # (R, U_use)
    U_use = D_use.shape[1]

    for p in range(P):
        full_p = float(full_flat[p])
        rep_p = reps_flat[:, p]

        if not np.isfinite(full_p):
            se_flat[p] = np.nan
            continue

        valid_rep = good_rep_base & np.isfinite(rep_p)
        if nan_policy == "propagate" and not np.all(valid_rep):
            se_flat[p] = np.nan
            continue
        if nan_policy == "omit" and not np.any(valid_rep):
            se_flat[p] = np.nan
            continue

        Dv = D_use[valid_rep, :]  # (Rv, U_use)
        yv = M * full_p - (M - m_del[valid_rep]) * rep_p[valid_rep]  # (Rv,)

        # If the delete-set design is not identifiable, fall back safely.
        # This can happen if too few random subsets were sampled.
        rank = np.linalg.matrix_rank(Dv) if Dv.size else 0
        if (Dv.shape[0] < U_use) or (rank < U_use):
            se_flat[p] = _se_direct_delete_d(rep_p, full_p)
            continue

        # Solve Dv g ~= yv for g_u = m_u * PV_u
        # Prefer normal equations solve; add tiny ridge only if needed.
        AtA = Dv.T @ Dv
        Aty = Dv.T @ yv

        try:
            g = np.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            tr = float(np.trace(AtA))
            lam = 1e-10 * (tr / U_use if (np.isfinite(tr) and tr > 0.0) else 1.0)
            try:
                g = np.linalg.solve(AtA + lam * np.eye(U_use, dtype=np.float64), Aty)
            except np.linalg.LinAlgError:
                g = np.linalg.lstsq(Dv, yv, rcond=None)[0]

        pv_use = g / m_u
        pv_all = np.full(U, np.nan, dtype=np.float64)
        pv_all[use_u] = pv_use

        se_flat[p] = _se_from_unit_pseudovalues(pv_all, full_p)

    return est_full, se_flat.reshape(est_full.shape)
