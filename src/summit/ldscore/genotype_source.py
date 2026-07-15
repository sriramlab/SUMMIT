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
    pvar = pd.read_csv(
        path, sep=r"\s+", skiprows=header_line, header=0,
        dtype={"#CHROM": str, "POS": np.int64, "ID": str, "REF": str, "ALT": str},
        keep_default_na=False,
    )
    required = ["#CHROM", "POS", "ID", "REF", "ALT"]
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

    def read_standardized_block(self, start: int, end: int) -> np.ndarray:
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
        t0 = time.perf_counter()
        self._reader.read_dosages_range(
            start, end, variant_major, allele_idx=0, sample_maj=0
        )
        geno = variant_major.T
        if not geno.flags.f_contiguous:
            raise RuntimeError("Internal PGEN transpose is not Fortran-contiguous.")
        missing = int(
            gwldcore.standardize_dosage_inplace(
                geno,
                ddof=self.ddof,
                missing_value=-9.0,
                num_threads=self.standardize_threads,
            )
        )
        self.decode_seconds += time.perf_counter() - t0
        self.blocks_read += 1
        self.variants_read += block_size
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
