from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from .. import gwldcore
except Exception:
    import gwldcore

from .. import utils


@dataclass(frozen=True)
class GenotypeInput:
    format: str
    prefix: str
    genotype_path: str
    variant_path: str
    sample_path: str


def _complete_trio(prefix: str, suffixes: tuple[str, str, str]) -> bool:
    return all(Path(prefix + suffix).is_file() for suffix in suffixes)


def _require_trio(prefix: str, suffixes: tuple[str, str, str], label: str) -> None:
    missing = [prefix + suffix for suffix in suffixes if not Path(prefix + suffix).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {label} input; missing: {', '.join(missing)}")


def resolve_genotype_input(path: str) -> GenotypeInput:
    """Resolve an explicit BED/PGEN path or an unambiguous trio prefix."""
    raw = os.fspath(path)
    lower = raw.lower()

    explicit_format = None
    prefix = raw
    for suffix in (".bed", ".bim", ".fam"):
        if lower.endswith(suffix):
            explicit_format = "bed"
            prefix = raw[: -len(suffix)]
            break
    if explicit_format is None:
        for suffix in (".pgen", ".pvar", ".psam"):
            if lower.endswith(suffix):
                explicit_format = "pgen"
                prefix = raw[: -len(suffix)]
                break

    bed_suffixes = (".bed", ".bim", ".fam")
    pgen_suffixes = (".pgen", ".pvar", ".psam")
    if explicit_format == "bed":
        _require_trio(prefix, bed_suffixes, "PLINK 1 BED/BIM/FAM")
        return GenotypeInput(
            "bed", os.path.abspath(prefix), os.path.abspath(prefix + ".bed"),
            os.path.abspath(prefix + ".bim"), os.path.abspath(prefix + ".fam")
        )
    if explicit_format == "pgen":
        _require_trio(prefix, pgen_suffixes, "PLINK 2 PGEN/PVAR/PSAM")
        return GenotypeInput(
            "pgen", os.path.abspath(prefix), os.path.abspath(prefix + ".pgen"),
            os.path.abspath(prefix + ".pvar"), os.path.abspath(prefix + ".psam")
        )

    bed_complete = _complete_trio(prefix, bed_suffixes)
    pgen_complete = _complete_trio(prefix, pgen_suffixes)
    if bed_complete and pgen_complete:
        raise ValueError(
            f"Both BED and PGEN trios exist for prefix '{prefix}'. "
            "Pass an explicit .bed or .pgen path."
        )
    if bed_complete:
        return resolve_genotype_input(prefix + ".bed")
    if pgen_complete:
        return resolve_genotype_input(prefix + ".pgen")

    present_bed = [prefix + suffix for suffix in bed_suffixes if Path(prefix + suffix).exists()]
    present_pgen = [prefix + suffix for suffix in pgen_suffixes if Path(prefix + suffix).exists()]
    if present_bed:
        _require_trio(prefix, bed_suffixes, "PLINK 1 BED/BIM/FAM")
    if present_pgen:
        _require_trio(prefix, pgen_suffixes, "PLINK 2 PGEN/PVAR/PSAM")
    raise FileNotFoundError(
        f"No complete BED/BIM/FAM or PGEN/PVAR/PSAM trio found for '{prefix}'."
    )


def read_fam_sample_ids(path: str) -> pd.DataFrame:
    samples = pd.read_csv(
        path, sep=r"\s+", header=None, usecols=[0, 1], names=["FID", "IID"],
        dtype=str, keep_default_na=False,
    )
    if samples.empty:
        raise ValueError(f"FAM contains no samples: {path}")
    if samples.duplicated(["FID", "IID"]).any():
        raise ValueError(f"FAM contains duplicate FID/IID pairs: {path}")
    return samples


def _find_psam_header(path: str) -> int:
    with open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("#FID") or line.startswith("#IID"):
                return line_number
            if line.strip() and not line.startswith("##"):
                raise ValueError(f"PSAM header must start with #FID or #IID: {path}")
    raise ValueError(f"PSAM header not found: {path}")


def read_psam_sample_ids(path: str) -> pd.DataFrame:
    header_line = _find_psam_header(path)
    samples = pd.read_csv(
        path, sep=r"\s+", skiprows=header_line, header=0,
        usecols=lambda column: column.lstrip("#") in {"FID", "IID"},
        dtype=str, keep_default_na=False,
    )
    samples.rename(columns={samples.columns[0]: samples.columns[0].lstrip("#")}, inplace=True)
    if "IID" not in samples.columns:
        raise ValueError(f"PSAM is missing an IID column: {path}")
    if "FID" not in samples.columns:
        samples.insert(0, "FID", "0")
    samples = samples[["FID", "IID"]].copy()
    if samples.empty:
        raise ValueError(f"PSAM contains no samples: {path}")
    if samples.duplicated(["FID", "IID"]).any():
        raise ValueError(f"PSAM contains duplicate FID/IID pairs: {path}")
    return samples


def _find_pvar_header(path: str) -> int:
    with open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("#CHROM"):
                return line_number
            if line.strip() and not line.startswith("##"):
                raise ValueError(f"PVAR header must start with #CHROM: {path}")
    raise ValueError(f"PVAR header not found: {path}")


def read_pvar_variants(path: str) -> pd.DataFrame:
    header_line = _find_pvar_header(path)
    required = ["#CHROM", "POS", "ID", "REF", "ALT"]
    pvar = pd.read_csv(
        path, sep=r"\s+", skiprows=header_line, header=0,
        usecols=lambda column: column in set(required),
        dtype={"#CHROM": str, "POS": np.int64, "ID": str, "REF": str, "ALT": str},
        keep_default_na=False,
    )
    missing = [column for column in required if column not in pvar.columns]
    if missing:
        raise ValueError(f"PVAR is missing required column(s) {missing}: {path}")
    if pvar.empty:
        raise ValueError(f"PVAR contains no variants: {path}")
    if pvar["ALT"].astype(str).str.contains(",", regex=False).any():
        raise ValueError("Multiallelic PGEN variants are not supported by genome-wide LD estimation.")
    # Expose PVAR ALT/REF as A1/A2.  SUMMIT's native BED decoder counts BIM A2;
    # allele_idx=0 (REF count) therefore preserves numeric orientation when a
    # comparator BED was exported with REF as A2.  This is not a claim that an
    # arbitrary BIM's A2 label is necessarily the biological reference allele.
    return pd.DataFrame({
        "CHR": pvar["#CHROM"].astype(str),
        "SNP": pvar["ID"].astype(str),
        "CM": np.zeros(len(pvar), dtype=np.float64),
        "BP": pvar["POS"].to_numpy(dtype=np.int64, copy=False),
        "A1": pvar["ALT"].astype(str),
        "A2": pvar["REF"].astype(str),
    })


def _canonical_chromosome(values, *, source: str) -> np.ndarray:
    raw = pd.Series(values, copy=False)
    labels = raw.astype("string").str.strip()
    bad = raw.isna() | labels.isna() | (labels == "")
    if bool(bad.any()):
        rows = np.flatnonzero(bad.to_numpy())[:10].tolist()
        raise ValueError(f"{source} contains missing/empty chromosome labels; first bad rows: {rows}.")
    labels = labels.str.replace(r"^(?i:chr)", "", regex=True).str.upper()
    numeric = pd.to_numeric(labels, errors="coerce")
    numeric_arr = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    integer_numeric = np.isfinite(numeric_arr) & (numeric_arr == np.rint(numeric_arr))
    out = labels.astype(str).to_numpy()
    if bool(integer_numeric.any()):
        out[integer_numeric] = (
            np.rint(numeric_arr[integer_numeric])
            .astype(np.int64)
            .astype(str)
        )
    return out


def validate_variant_metadata(variants: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Validate a BIM/PVAR-like variant table without changing row order."""
    required = {"CHR", "SNP", "BP"}
    missing = sorted(required - set(variants.columns))
    if missing:
        raise ValueError(f"{source} is missing required variant column(s): {missing}.")
    out = variants.copy()
    raw_snp = out["SNP"]
    snp = raw_snp.astype("string").str.strip()
    bad = raw_snp.isna() | snp.isna() | (snp == "")
    if bool(bad.any()):
        rows = np.flatnonzero(bad.to_numpy())[:10].tolist()
        raise ValueError(f"{source} contains missing/empty SNP IDs; first bad rows: {rows}.")
    out["SNP"] = snp.astype(str)
    dup = out["SNP"].duplicated(keep=False)
    if bool(dup.any()):
        examples = out.loc[dup, "SNP"].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{source} contains duplicate SNP IDs; variant alignment would be ambiguous. "
            f"First duplicates: {examples}."
        )
    out["CHR"] = _canonical_chromosome(out["CHR"], source=source)
    bp = pd.to_numeric(out["BP"], errors="coerce").to_numpy(dtype=np.float64)
    good_bp = np.isfinite(bp) & (bp >= 0.0) & (bp == np.rint(bp))
    if not bool(np.all(good_bp)):
        rows = np.flatnonzero(~good_bp)[:10].tolist()
        raise ValueError(f"{source} contains invalid BP coordinates; first bad rows: {rows}.")
    out["BP"] = np.rint(bp).astype(np.int64)
    return out


def read_aligned_annotations(
    annot_path: str,
    variants: pd.DataFrame,
    *,
    log=None,
    source_label: str = "genotype metadata",
) -> tuple[list[str], np.ndarray, bool]:
    """Read a full or thin annotation matrix on an exact genotype variant axis."""
    variants = validate_variant_metadata(variants, source=source_label)
    paths = utils._resolve_chr_split_paths(annot_path, require=True)
    # Probe the first resolved file only to distinguish metadata-bearing full
    # annotations from numeric/headered thin matrices.  The appropriate shared
    # loader below then reads and validates every chromosome file.
    probe = pd.read_csv(
        paths[0], sep=r"\s+", compression="infer", nrows=0
    )
    cols = probe.columns.tolist()
    first4 = cols[:4]
    required = {"CHR", "BP", "SNP"}
    is_full = required.issubset(set(first4))
    if (not is_full) and required.intersection(set(cols)):
        raise ValueError(
            "Malformed annotation metadata: a full annotation must contain CHR, BP, SNP "
            "in its first columns (and optional CM as the 4th column)."
        )
    if "CM" in cols and (len(first4) < 4 or first4[3] != "CM"):
        raise ValueError(
            "Malformed annotation metadata: CM is reserved for the optional "
            "4th metadata column."
        )

    if is_full:
        df = utils._read_csv_maybe_chr_split(
            annot_path,
            sep=r"\s+",
            compression="infer",
            index_col=False,
        )
        start = 4 if "CM" in first4 else 3
        annot_cols = [str(c) for c in cols[start:]]
        if not annot_cols:
            raise ValueError("Annotation file contains no annotation columns.")
        ann = validate_variant_metadata(
            df[["CHR", "BP", "SNP", *annot_cols]].copy(),
            source=f"annotation input '{annot_path}'",
        )
        pos = pd.Index(ann["SNP"]).get_indexer(variants["SNP"].to_numpy())
        missing = pos < 0
        if np.any(missing):
            examples = variants.loc[missing, "SNP"].head(10).tolist()
            raise ValueError(
                f"Annotation is missing {int(np.sum(missing))} genotype variant(s); "
                f"first missing IDs: {examples}."
            )
        aligned = ann.iloc[pos].reset_index(drop=True)
        chr_bad = aligned["CHR"].to_numpy() != variants["CHR"].to_numpy()
        bp_bad = aligned["BP"].to_numpy(dtype=np.int64) != variants["BP"].to_numpy(dtype=np.int64)
        bad_coord = chr_bad | bp_bad
        if np.any(bad_coord):
            idx = np.flatnonzero(bad_coord)[:10]
            details = [
                f"{variants['SNP'].iloc[i]}:{variants['CHR'].iloc[i]}:{int(variants['BP'].iloc[i])}"
                f"!={aligned['CHR'].iloc[i]}:{int(aligned['BP'].iloc[i])}"
                for i in idx
            ]
            raise ValueError(
                f"CHR/BP mismatch between {source_label} and annotation for "
                f"{int(np.sum(bad_coord))} variant(s). First mismatches: {details}."
            )
        extra = int(len(ann) - len(variants))
        if extra > 0 and log is not None:
            log._log(f"[info] Annotation contains {extra} extra variant(s); keeping genotype variants only.")
        arr = aligned[annot_cols].to_numpy(dtype=np.float64, copy=False)
    else:
        annot_cols, arr = utils._read_with_optional_header(annot_path)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != len(variants):
            raise ValueError(
                f"Thin annotation has {arr.shape[0]} rows; expected {len(variants)} genotype variants."
            )
        if annot_cols is None:
            annot_cols = [f"L2_{i}" for i in range(arr.shape[1])]
        else:
            annot_cols = [str(c) for c in annot_cols]

    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("Annotation matrix must have at least one column.")
    if not np.isfinite(arr).all():
        rows = np.flatnonzero(~np.isfinite(arr).all(axis=1))[:10].tolist()
        raise ValueError(f"Annotation contains non-finite values; first bad rows: {rows}.")
    arr = np.asarray(arr, dtype=np.float64, order="C")
    if np.any(arr < 0.0):
        if log is not None:
            log._log("[warn] Negative annotation values found; clipping to zero.")
        arr = np.maximum(arr, 0.0)
    is_continuous = not bool(np.all(np.isin(np.unique(arr), [0.0, 1.0])))
    return annot_cols, arr, is_continuous


class PgenBlockReader:
    """Persistent, preallocated variant-major PGEN dosage reader."""

    def __init__(
        self,
        pgen_path: str,
        raw_sample_ct: int,
        variant_ct: int,
        sample_subset,
        step_size: int,
        dtype,
        ddof: int,
        standardize_threads: int = 1,
    ) -> None:
        try:
            import pgenlib
        except ImportError as exc:
            raise ImportError(
                "PGEN input requires pgenlib. Install SUMMIT's runtime dependencies "
                "or run `python -m pip install 'pgenlib>=0.94,<1'`."
            ) from exc

        self.pgen_path = os.path.abspath(pgen_path)
        self.raw_sample_ct = int(raw_sample_ct)
        self.variant_ct = int(variant_ct)
        self.ddof = int(ddof)
        self.standardize_threads = max(1, int(standardize_threads))
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("PGEN dosage decoding supports only float32 and float64.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        subset = None
        if sample_subset is not None:
            subset = np.asarray(sample_subset, dtype=np.uint32)
            if subset.ndim != 1 or subset.size == 0:
                raise ValueError("PGEN sample subset must be a non-empty 1D array.")
            if np.any(subset[1:] <= subset[:-1]):
                raise ValueError("PGEN sample subset indexes must be strictly increasing.")
            if int(subset[-1]) >= self.raw_sample_ct:
                raise ValueError("PGEN sample subset index is out of range.")
            subset = np.ascontiguousarray(subset)
        self.sample_subset = subset
        self.sample_ct = self.raw_sample_ct if subset is None else int(subset.size)

        try:
            self._reader = pgenlib.PgenReader(
                os.fsencode(self.pgen_path),
                raw_sample_ct=self.raw_sample_ct,
                variant_ct=self.variant_ct,
                sample_subset=self.sample_subset,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to open PGEN with the sample/variant counts implied by PSAM/PVAR."
            ) from exc
        if int(self._reader.get_raw_sample_ct()) != self.raw_sample_ct:
            self.close()
            raise ValueError("PGEN header sample count does not match PSAM.")
        if int(self._reader.get_variant_ct()) != self.variant_ct:
            self.close()
            raise ValueError("PGEN header variant count does not match PVAR.")

        self.block_capacity = min(int(step_size), self.variant_ct)
        self._buffer = np.empty(
            (self.block_capacity, self.sample_ct), dtype=self.dtype, order="C"
        )
        self.blocks_read = 0
        self.variants_read = 0
        self.missing_values = 0
        self.decode_seconds = 0.0

    def _read_dosage_range(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if not (0 <= start < end <= self.variant_ct):
            raise IndexError(f"Invalid PGEN variant range [{start}, {end}).")
        block_size = end - start
        if block_size > self.block_capacity:
            raise ValueError(
                f"PGEN block size {block_size} exceeds preallocated capacity {self.block_capacity}."
            )

        variant_major = self._buffer[:block_size, :]
        self._reader.read_dosages_range(
            start, end, variant_major, allele_idx=0, sample_maj=0
        )
        geno = variant_major.T
        if not geno.flags.f_contiguous:
            raise RuntimeError("Internal PGEN transpose is not Fortran-contiguous.")
        return geno

    def read_dosage_block(self, start: int, end: int) -> np.ndarray:
        """Decode a REF-dosage block without changing its numeric scale.

        The returned sample-by-variant array is a view of the reader's reusable
        buffer and remains valid only until the next range read. Missing dosages
        use pgenlib's ``-9`` sentinel.
        """
        t0 = time.perf_counter()
        geno = self._read_dosage_range(start, end)
        self.decode_seconds += time.perf_counter() - t0
        self.blocks_read += 1
        self.variants_read += int(end) - int(start)
        self.missing_values += int(np.count_nonzero(geno == -9.0))
        return geno

    def read_standardized_block(self, start: int, end: int) -> np.ndarray:
        """Decode, mean-impute, and standardize one dosage block in place."""
        t0 = time.perf_counter()
        geno = self._read_dosage_range(start, end)
        missing = int(
            gwldcore.standardize_dosage_inplace(
                geno,
                ddof=self.ddof,
                missing_value=-9.0,
                num_threads=self.standardize_threads,
            )
        )
        # Keep the historical statistic as decode + standardization wall time.
        self.decode_seconds += time.perf_counter() - t0
        self.blocks_read += 1
        self.variants_read += int(end) - int(start)
        self.missing_values += missing
        return geno

    def close(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()
            self._reader = None

    def __enter__(self) -> "PgenBlockReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
