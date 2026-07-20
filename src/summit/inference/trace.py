from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import utils

_META_COLS = {"CHR", "BP", "SNP", "CM"}


def _normalize_variant_keys(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Validate and normalize the variant key used by all Trace joins."""
    # Normalize the three metadata columns in place; copying every LD/annotation
    # column here can transiently double memory for whole-genome partitioned files.
    out = df
    raw_snp = out["SNP"]
    snp = raw_snp.astype("string").str.strip()
    bad_snp = raw_snp.isna() | snp.isna() | (snp == "")
    if bool(bad_snp.any()):
        rows = np.flatnonzero(bad_snp.to_numpy())[:10].tolist()
        raise ValueError(f"{source} contains missing/empty SNP IDs; first bad rows: {rows}.")
    out["SNP"] = snp.astype(str)

    dup = out["SNP"].duplicated(keep=False)
    if bool(dup.any()):
        examples = out.loc[dup, "SNP"].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{source} contains duplicate SNP IDs; joins would be ambiguous. "
            f"First duplicates: {examples}."
        )

    for col in ("CHR", "BP"):
        numeric = pd.to_numeric(out[col], errors="coerce").to_numpy(copy=False)
        if np.issubdtype(numeric.dtype, np.integer):
            good = np.ones(numeric.shape, dtype=bool)
            if col == "BP":
                good &= numeric >= 0
            normalized = numeric.astype(np.int64, copy=False)
        else:
            numeric = np.asarray(numeric, dtype=np.float64)
            good = np.isfinite(numeric) & (numeric == np.rint(numeric))
            if col == "BP":
                good &= numeric >= 0.0
            normalized = None
        if not bool(np.all(good)):
            rows = np.flatnonzero(~good)[:10].tolist()
            raise ValueError(
                f"{source} contains invalid {col} values; expected finite integer coordinates. "
                f"First bad rows: {rows}."
            )
        if normalized is None:
            normalized = np.rint(numeric).astype(np.int64)
        out[col] = normalized
    return out


def _assert_matching_coordinates(
    primary: pd.DataFrame,
    other: pd.DataFrame,
    *,
    other_source: str,
) -> None:
    """Require CHR/BP agreement for two already SNP-aligned data frames."""
    if primary.shape[0] != other.shape[0]:
        raise RuntimeError("Internal error: coordinate comparison axes differ in length.")
    chr1 = primary["CHR"].to_numpy(dtype=np.int64, copy=False)
    bp1 = primary["BP"].to_numpy(dtype=np.int64, copy=False)
    chr2 = other["CHR"].to_numpy(dtype=np.int64, copy=False)
    bp2 = other["BP"].to_numpy(dtype=np.int64, copy=False)
    bad = (chr1 != chr2) | (bp1 != bp2)
    if bool(np.any(bad)):
        rows = np.flatnonzero(bad)[:10]
        details = [
            f"{primary['SNP'].iloc[i]}:primary={chr1[i]}:{bp1[i]},other={chr2[i]}:{bp2[i]}"
            for i in rows
        ]
        raise ValueError(
            f"CHR/BP mismatch between primary LD scores and {other_source} for "
            f"{int(np.sum(bad))} shared SNP(s). First mismatches: {details}."
        )


@dataclass(frozen=True)
class TraceView:
    snps: np.ndarray
    chr: np.ndarray | None
    bp: np.ndarray | None
    annot: np.ndarray
    annot_header: np.ndarray
    ldscores: np.ndarray
    ldscores_reg: np.ndarray | None = None
    ldscores_reg_w: np.ndarray | None = None
    delta: np.ndarray | None = None
    kmoments: dict | None = None
    kmoments_path: str | None = None
    kmoments_valid: bool = False

    @property
    def nsnps(self) -> int:
        return int(self.annot.shape[0])

    @property
    def nbins(self) -> int:
        return int(self.annot.shape[1])

    def subset(self, keep_mask) -> "TraceView":
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or keep_mask.size != self.nsnps:
            raise ValueError(
                f"keep_mask must be length {self.nsnps}; got {keep_mask.shape}."
            )
        full_keep = bool(np.all(keep_mask))
        return TraceView(
            snps=self.snps[keep_mask],
            chr=None if self.chr is None else self.chr[keep_mask],
            bp=None if self.bp is None else self.bp[keep_mask],
            annot=self.annot[keep_mask, :],
            annot_header=self.annot_header,
            ldscores=self.ldscores[keep_mask, :],
            ldscores_reg=None if self.ldscores_reg is None else self.ldscores_reg[keep_mask, :],
            ldscores_reg_w=None if self.ldscores_reg_w is None else self.ldscores_reg_w[keep_mask, :],
            delta=self.delta,
            kmoments=self.kmoments,
            kmoments_path=self.kmoments_path,
            kmoments_valid=bool(self.kmoments is not None and self.kmoments_valid and full_keep),
        )


class Trace:
    """
    Immutable base reference panel / annotation object.

    Responsibilities:
      - read main LD-scores (and optional regression LD-scores)
      - read/align annotation to the LD-score SNP universe
      - freeze a base SNP order and expose cheap TraceView materialization
    """

    def __init__(
        self,
        *,
        bimpath=None,
        sumpath=None,
        savepath=None,
        log=None,
        ldscores=None,
        ldscores_reg=None,
        annot=None,
        ldscores_reg_w=None,
        verbose=False,
        delta=None,
    ):
        if sumpath is not None:
            raise NotImplementedError(
                "Trace summaries (.tr/.MN) are not supported in this refactor yet. "
                "Use per-SNP LD-scores for now."
            )
        if savepath is not None:
            # harmlessly ignored for now
            pass
        if ldscores is None:
            raise ValueError("Trace requires per-SNP LD-scores via `ldscores=`.")

        self.log = log
        self.verbose = verbose
        self.delta = None if delta is None else np.asarray(delta, dtype=np.float64)

        main_df, main_L, main_start = self._read_ldscores_file(ldscores, which="main")
        reg_df = None
        reg_L = None
        reg_start = None
        if ldscores_reg is not None:
            reg_df, reg_L, reg_start = self._read_ldscores_file(ldscores_reg, which="reg")

        regw_df = None
        regw_L = None
        regw_start = None
        if ldscores_reg_w is not None:
            regw_df, regw_L, regw_start = self._read_ldscores_file(ldscores_reg_w, which="reg_w")

        self._ldscore_start_idx = int(main_start)
        self._ldscore_reg_start_idx = None if reg_start is None else int(reg_start)
        self._ldscore_reg_w_start_idx = None if regw_start is None else int(regw_start)

        main_ld_nsnps_raw = int(main_df.shape[0])
        # Preserve the effect-reference universe before optional regression-LD
        # intersections. Constrained LDSC keeps this mass fixed across all
        # downstream SNP filtering and delete refits.
        self.source_nsnps = main_ld_nsnps_raw

        self.kmoments = None
        self.kmoments_path = None
        km_path = self._infer_kmoments_path(ldscores)
        if km_path is not None:
            try:
                self.kmoments = self._read_kmoments_file(km_path)
                self.kmoments_path = km_path
            except Exception as e:
                if self.log is not None:
                    self.log._log(f"[kmom] failed to read {km_path}: {e}; ignoring.")
                self.kmoments = None
                self.kmoments_path = None

        # Align optional regression LD inputs to the main SNP order first.
        if reg_df is not None:
            reg_index = pd.Index(reg_df["SNP"].to_numpy())
            idx = reg_index.get_indexer(main_df["SNP"].to_numpy())
            keep = idx >= 0
            n_drop = int((~keep).sum())
            if n_drop > 0 and self.log is not None:
                self.log._log(
                    f"Dropping {n_drop} SNPs from primary LD-scores because they are missing in ldscores_reg."
                )
            main_df = main_df.loc[keep].reset_index(drop=True)
            main_L = main_L[keep, :]
            reg_df = reg_df.iloc[idx[keep]].reset_index(drop=True)
            _assert_matching_coordinates(main_df, reg_df, other_source="regression LD scores")
            reg_L = reg_df.iloc[:, reg_start:].to_numpy(dtype=np.float64, copy=False)

        if regw_df is not None:
            regw_index = pd.Index(regw_df["SNP"].to_numpy())
            idx = regw_index.get_indexer(main_df["SNP"].to_numpy())
            keep = idx >= 0
            n_drop = int((~keep).sum())
            if n_drop > 0 and self.log is not None:
                self.log._log(
                    f"Dropping {n_drop} SNPs from primary LD-scores because they are missing in ldscores_reg_w."
                )
            main_df = main_df.loc[keep].reset_index(drop=True)
            main_L = main_L[keep, :]
            if reg_df is not None:
                reg_df = reg_df.loc[keep].reset_index(drop=True)
                reg_L = reg_df.iloc[:, reg_start:].to_numpy(dtype=np.float64, copy=False)
            regw_df = regw_df.iloc[idx[keep]].reset_index(drop=True)
            _assert_matching_coordinates(main_df, regw_df, other_source="weight LD scores")
            regw_L = regw_df.iloc[:, regw_start:].to_numpy(dtype=np.float64, copy=False)
        annot_df, annot_header, annot_matrix = self._read_annotation(
            annot_path=annot,
            bimpath=bimpath,
            snps_main=main_df["SNP"].astype(str).to_numpy(),
            chr_main=main_df["CHR"].to_numpy(dtype=np.int32, copy=False),
            bp_main=main_df["BP"].to_numpy(dtype=np.int64, copy=False),
        )

        # Align main (and reg) LD-scores to the annotation-pruned SNP order.
        main_index = main_df.set_index("SNP")
        main_df = main_index.loc[annot_df["SNP"].astype(str).to_numpy()].reset_index()
        main_L = main_df.iloc[:, main_start:].to_numpy(dtype=np.float64, copy=False)

        if reg_df is not None:
            reg_df = reg_df.set_index("SNP").loc[annot_df["SNP"].astype(str).to_numpy()].reset_index()
            reg_L = reg_df.iloc[:, reg_start:].to_numpy(dtype=np.float64, copy=False)

        if regw_df is not None:
            regw_df = regw_df.set_index("SNP").loc[annot_df["SNP"].astype(str).to_numpy()].reset_index()
            regw_L = regw_df.iloc[:, regw_start:].to_numpy(dtype=np.float64, copy=False)
        # Sort once to stable genomic order. This keeps block jackknife contiguous on the genome
        # and makes chr jackknife valid without any further mutation.
        chr_arr = main_df["CHR"].to_numpy(dtype=np.int32, copy=False)
        bp_arr = main_df["BP"].to_numpy(dtype=np.int64, copy=False)
        snp_arr = main_df["SNP"].astype(str).to_numpy()
        order = np.lexsort((snp_arr, bp_arr, chr_arr))
        if not np.array_equal(order, np.arange(order.size)) and self.log is not None:
            self.log._log("[Trace] sorting SNPs by (CHR,BP,SNP).")

        self.snps = snp_arr[order]
        self.chr = chr_arr[order]
        self.bp = bp_arr[order]
        self.annot = np.asarray(annot_matrix, dtype=np.float64, order="C")[order, :]
        self.annot_header = np.asarray(annot_header)
        self.ldscores = np.asarray(main_L, dtype=np.float64, order="C")[order, :]
        self.ldscores_reg_w = None if regw_L is None else np.asarray(regw_L, dtype=np.float64, order="C")[order, :]
        self.ldscores_reg = None if reg_L is None else np.asarray(reg_L, dtype=np.float64, order="C")[order, :]

        self.nsnps = int(self.annot.shape[0])
        self.nbins = int(self.annot.shape[1])
        self.index = pd.Index(self.snps)

        if self.kmoments is not None:
            km_nsnps = int(round(float(self.kmoments.get("nsnps", np.nan))))

            if km_nsnps != main_ld_nsnps_raw:
                if self.log is not None:
                    self.log._log(
                        f"[kmom] ignoring {self.kmoments_path}: "
                        f"kmoments nsnps={km_nsnps} but main LD-score file has {main_ld_nsnps_raw} SNPs."
                    )
                self.kmoments = None
                self.kmoments_path = None

            elif self.nsnps != main_ld_nsnps_raw:
                if self.log is not None:
                    self.log._log(
                        f"[kmom] ignoring {self.kmoments_path}: "
                        f"the final Trace SNP axis ({self.nsnps}) differs from the raw main LD-score axis "
                        f"({main_ld_nsnps_raw}) after annotation / regression-LD alignment."
                    )
                self.kmoments = None
                self.kmoments_path = None

            elif self.log is not None:
                alpha_probe = float(self.kmoments.get("alpha_probe", np.nan))
                self.log._log(
                    f"[kmom] loaded {self.kmoments_path} for {km_nsnps} SNPs "
                    f"(alpha_probe={alpha_probe:.6g}); valid only when rg keeps the full Trace SNP axis."
                )

        if self.log is not None:
            reg_bins = 'no' if self.ldscores_reg is None else self.ldscores_reg.shape[1]
            regw_bins = 'no' if self.ldscores_reg_w is None else self.ldscores_reg_w.shape[1]
            self.log._log(
                f"Loaded Trace with {self.nsnps} SNPs, {self.nbins} annotation bins, "
                f"{reg_bins} regression LD bins, and {regw_bins} regression-weight LD bins."
            )

    def materialize_view(self, keep_mask=None) -> TraceView:
        if keep_mask is None:
            keep_mask = np.ones(self.nsnps, dtype=bool)
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or keep_mask.size != self.nsnps:
            raise ValueError(
                f"keep_mask must be length {self.nsnps}; got {keep_mask.shape}."
            )

        full_keep = bool(np.all(keep_mask))

        if full_keep:
            snps = self.snps
            chr_arr = self.chr
            bp = self.bp
            annot = self.annot
            ldscores = self.ldscores
            ldscores_reg = self.ldscores_reg
            ldscores_reg_w = self.ldscores_reg_w
        else:
            snps = self.snps[keep_mask]
            chr_arr = self.chr[keep_mask]
            bp = self.bp[keep_mask]
            annot = self.annot[keep_mask, :]
            ldscores = self.ldscores[keep_mask, :]
            ldscores_reg = None if self.ldscores_reg is None else self.ldscores_reg[keep_mask, :]
            ldscores_reg_w = None if self.ldscores_reg_w is None else self.ldscores_reg_w[keep_mask, :]

        return TraceView(
            snps=snps,
            chr=chr_arr,
            bp=bp,
            annot=annot,
            annot_header=self.annot_header,
            ldscores=ldscores,
            ldscores_reg=ldscores_reg,
            ldscores_reg_w=ldscores_reg_w,
            delta=self.delta,
            kmoments=self.kmoments,
            kmoments_path=self.kmoments_path,
            kmoments_valid=bool(self.kmoments is not None and full_keep),
        )

    def _read_ldscores_file(self, path, *, which: str):
        if self.log is not None and utils._is_chr_split_spec(path):
            paths = utils._resolve_chr_split_paths(path, require=True)
            self.log._log(
                f"[Trace] reading chromosome-split LD-scores ({which}) from "
                f"{len(paths)} file(s): {utils._normalize_path_spec(path)}"
            )
        df = utils._read_csv_maybe_chr_split(
            path, compression="infer", sep=r"\s+", index_col=False
        )
        cols = df.columns.tolist()
        first4 = cols[:4]
        required = {"CHR", "BP", "SNP"}
        if not required.issubset(set(first4)):
            raise ValueError(
                "Input LD score file must have CHR, BP, SNP in the first columns "
                "(and optional CM as the 4th column)."
            )
        if "CM" in cols and (len(first4) < 4 or first4[3] != "CM"):
            raise ValueError(
                "Malformed LD-score metadata: CM, when present, must be the 4th column."
            )
        start_idx = 4 if ("CM" in first4) else 3
        df = _normalize_variant_keys(df, source=f"LD-score input ({which}) '{path}'")
        L = df.iloc[:, start_idx:].to_numpy(dtype=np.float64, copy=False)

        if which == "reg_w" and L.ndim == 2 and L.shape[1] != 1:
            raise ValueError(
                f"Regression-weight LD file '{path}' must contain exactly one LD-score column; got {L.shape[1]}."
            )

        finite = np.isfinite(L).all(axis=1)
        if which in {"reg", "reg_w"}:
            ltot = L.sum(axis=1)
            finite &= np.isfinite(ltot) & (ltot > 0.0)

        n_drop = int((~finite).sum())
        if n_drop > 0 and self.log is not None:
            self.log._log(
                f"Dropping {n_drop} SNPs from LD-scores ({which}) due to non-finite LD values "
                f"or non-positive total LD score."
            )

        df = df.loc[finite].reset_index(drop=True)
        L = L[finite, :]
        return df, L, start_idx

    def _read_annotation(self, *, annot_path, bimpath, snps_main, chr_main, bp_main):
        snps_main = np.asarray(snps_main, dtype=str)

        # single-bin fallback
        if annot_path is None:
            annot_header = np.array(["L2"])
            annot = np.ones((snps_main.size, 1), dtype=np.float64)
            annot_df = pd.DataFrame(
                {"CHR": chr_main, "BP": bp_main, "SNP": snps_main, "L2": np.ones(snps_main.size)}
            )
            if self.log is not None:
                self.log._log("Running with single-component annotation.")
            return annot_df, annot_header, annot

        if self.log is not None and utils._is_chr_split_spec(annot_path):
            paths = utils._resolve_chr_split_paths(annot_path, require=True)
            self.log._log(
                f"[Trace] reading chromosome-split annotation from "
                f"{len(paths)} file(s): {utils._normalize_path_spec(annot_path)}"
            )
        df = utils._read_csv_maybe_chr_split(
            annot_path, sep=r"\s+", compression="infer"
        )
        cols = df.columns.tolist()
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
                "Malformed annotation metadata: CM, when present, must be the 4th column."
            )

        if is_full:
            start_idx = 4 if ("CM" in first4) else 3
            annot_cols = cols[start_idx:]
            if len(annot_cols) == 0:
                raise ValueError("Annotation file contains no annotation columns.")

            ann = df[["CHR", "BP", "SNP"] + annot_cols].copy()
            ann = _normalize_variant_keys(ann, source=f"annotation input '{annot_path}'")
            pos = pd.Index(ann["SNP"]).get_indexer(snps_main)
            keep = pos >= 0
            n_missing = int(np.sum(~keep))
            if n_missing > 0 and self.log is not None:
                self.log._log(
                    f"Dropping {n_missing} SNPs because they are missing in the annotation file."
                )
            primary = pd.DataFrame(
                {
                    "CHR": np.asarray(chr_main)[keep],
                    "BP": np.asarray(bp_main)[keep],
                    "SNP": snps_main[keep],
                }
            )
            aligned = ann.iloc[pos[keep]].reset_index(drop=True)
            _assert_matching_coordinates(primary, aligned, other_source="annotation")

            annot_header = np.asarray(annot_cols)
            annot = aligned[annot_cols].to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(annot).all():
                bad = np.flatnonzero(~np.isfinite(annot).all(axis=1))[:10].tolist()
                raise ValueError(
                    f"Annotation contains non-finite values. First bad rows: {bad}"
                )
            if self.log is not None:
                self.log._log(f"Read full annotation of shape {annot.shape}.")
            return aligned[["CHR", "BP", "SNP"] + annot_cols], annot_header, annot

        else:
            # Thin annotations have no SNP/coordinate metadata; their row order is
            # explicitly defined by BIM (when supplied) or by the primary LD file.
            if bimpath is not None and Path(str(bimpath)).suffix == ".bim":
                bim_snps = []
                with open(bimpath, "r") as fd:
                    for line in fd:
                        bim_snps.append(line.split()[1])
                bim_snps = np.asarray(bim_snps, dtype=str)
            else:
                bim_snps = snps_main
            bim_index = pd.Index(bim_snps)
            if bim_index.has_duplicates:
                examples = bim_index[bim_index.duplicated(keep=False)].unique()[:10].tolist()
                raise ValueError(
                    f"Thin annotation SNP axis contains duplicate SNP IDs; first duplicates: {examples}."
                )

            header, annot = self._read_with_optional_header(annot_path)
            annot = np.asarray(annot, dtype=np.float64)
            if annot.ndim == 1:
                annot = annot.reshape(-1, 1)
            if annot.shape[0] != bim_snps.size:
                raise ValueError(
                    f"Thin annotation row count ({annot.shape[0]}) does not match the SNP list ({bim_snps.size})."
                )
            if header is None:
                header = np.array([f"bin_{i}" for i in range(annot.shape[1])], dtype=object)
            else:
                header = np.asarray(header)

            ann_df = pd.DataFrame(annot, columns=header)
            ann_df.insert(0, "SNP", bim_snps)

            aligned = ann_df.set_index("SNP").reindex(snps_main)
            missing = aligned[header.tolist()].isna().all(axis=1)
            n_missing = int(missing.sum())
            if n_missing > 0 and self.log is not None:
                self.log._log(
                    f"Dropping {n_missing} SNPs because they are missing in the thin annotation."
                )
            aligned = aligned.loc[~missing].reset_index().rename(columns={"index": "SNP"})

            keep_mask = pd.Index(snps_main).isin(aligned["SNP"].astype(str).to_numpy())
            chr_keep = np.asarray(chr_main)[keep_mask]
            bp_keep = np.asarray(bp_main)[keep_mask]
            aligned.insert(0, "BP", bp_keep)
            aligned.insert(0, "CHR", chr_keep)

            annot = aligned[header.tolist()].to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(annot).all():
                bad = np.flatnonzero(~np.isfinite(annot).all(axis=1))[:10].tolist()
                raise ValueError(
                    f"Thin annotation contains non-finite values. First bad rows: {bad}"
                )
            if self.log is not None:
                self.log._log(f"Read thin annotation of shape {annot.shape}.")
            return aligned[["CHR", "BP", "SNP"] + header.tolist()], np.asarray(header), annot

    @staticmethod
    def _read_with_optional_header(path):
        return utils._read_with_optional_header(path)

    @staticmethod
    def _infer_kmoments_path(ldscores_path):
        if ldscores_path is None:
            return None

        s = str(ldscores_path)
        if utils._is_chr_split_spec(s):
            return None
        candidates = []

        if s.endswith(".gw.ldscore.gz"):
            candidates.append(s[: -len(".gw.ldscore.gz")] + ".gw.kmoments")
        if s.endswith(".ldscore.gz"):
            candidates.append(s[: -len(".ldscore.gz")] + ".kmoments")

        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            if Path(c).exists():
                return c
        return None

    @staticmethod
    def _read_kmoments_file(path):
        df = pd.read_csv(path, sep=r"\s+", compression="infer")
        if df.shape[0] != 1:
            raise ValueError(f"Expected exactly one row in kmoments file '{path}', got {df.shape[0]}.")

        row = df.iloc[0].to_dict()
        out = {}
        for k, v in row.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v

        required = {"nsnps", "proj_rank", "t0_rank", "t1_rank", "t2_rank"}
        missing = [k for k in required if k not in out]
        if missing:
            raise ValueError(f"kmoments file '{path}' is missing required columns: {missing}")

        for k in required:
            val = float(out[k])
            if not np.isfinite(val):
                raise ValueError(f"kmoments file '{path}' has non-finite value for '{k}': {val}")

        return out
