from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_META_COLS = {"CHR", "BP", "SNP", "CM"}


@dataclass(frozen=True)
class TraceView:
    snps: np.ndarray
    chr: np.ndarray | None
    bp: np.ndarray | None
    annot: np.ndarray
    annot_header: np.ndarray
    ldscores: np.ndarray
    ldscores_reg: np.ndarray | None = None
    delta: np.ndarray | None = None

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
        return TraceView(
            snps=self.snps[keep_mask],
            chr=None if self.chr is None else self.chr[keep_mask],
            bp=None if self.bp is None else self.bp[keep_mask],
            annot=self.annot[keep_mask, :],
            annot_header=self.annot_header,
            ldscores=self.ldscores[keep_mask, :],
            ldscores_reg=None if self.ldscores_reg is None else self.ldscores_reg[keep_mask, :],
            delta=self.delta,
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

        self._ldscore_start_idx = int(main_start)
        self._ldscore_reg_start_idx = None if reg_start is None else int(reg_start)

        # Align optional regression LD to main SNP order first.
        if reg_df is not None:
            reg_index = pd.Index(reg_df["SNP"].astype(str).to_numpy())
            idx = reg_index.get_indexer(main_df["SNP"].astype(str).to_numpy())
            keep = idx >= 0
            n_drop = int((~keep).sum())
            if n_drop > 0 and self.log is not None:
                self.log._log(
                    f"Dropping {n_drop} SNPs from primary LD-scores because they are missing in ldscores_reg."
                )
            main_df = main_df.loc[keep].reset_index(drop=True)
            main_L = main_L[keep, :]
            reg_df = reg_df.set_index("SNP").loc[main_df["SNP"].astype(str).to_numpy()].reset_index()
            reg_L = reg_df.iloc[:, reg_start:].to_numpy(dtype=np.float64, copy=False)

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
        self.ldscores_reg = None if reg_L is None else np.asarray(reg_L, dtype=np.float64, order="C")[order, :]

        self.nsnps = int(self.annot.shape[0])
        self.nbins = int(self.annot.shape[1])
        self.index = pd.Index(self.snps)

        if self.log is not None:
            self.log._log(
                f"Loaded Trace with {self.nsnps} SNPs, {self.nbins} annotation bins, "
                f"and {'no' if self.ldscores_reg is None else self.ldscores_reg.shape[1]} regression LD bins."
            )

    def materialize_view(self, keep_mask=None) -> TraceView:
        if keep_mask is None:
            keep_mask = np.ones(self.nsnps, dtype=bool)
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or keep_mask.size != self.nsnps:
            raise ValueError(
                f"keep_mask must be length {self.nsnps}; got {keep_mask.shape}."
            )
        return TraceView(
            snps=self.snps[keep_mask],
            chr=self.chr[keep_mask],
            bp=self.bp[keep_mask],
            annot=self.annot[keep_mask, :],
            annot_header=self.annot_header,
            ldscores=self.ldscores[keep_mask, :],
            ldscores_reg=None if self.ldscores_reg is None else self.ldscores_reg[keep_mask, :],
            delta=self.delta,
        )

    def _read_ldscores_file(self, path, *, which: str):
        df = pd.read_csv(path, compression="infer", sep=r"\s+", index_col=False)
        cols = df.columns.tolist()
        first4 = cols[:4]
        required = {"CHR", "BP", "SNP"}
        if not required.issubset(set(first4)):
            raise ValueError(
                "Input LD score file must have CHR, BP, SNP in the first columns "
                "(and optional CM as the 4th column)."
            )
        start_idx = 4 if ("CM" in first4) else 3
        L = df.iloc[:, start_idx:].to_numpy(dtype=np.float64, copy=False)
        snps = df["SNP"].astype(str).to_numpy()

        finite = np.isfinite(L).all(axis=1)
        if which == "reg":
            ltot = L.sum(axis=1)
            finite &= np.isfinite(ltot) & (ltot > 0.1)

        n_drop = int((~finite).sum())
        if n_drop > 0 and self.log is not None:
            self.log._log(
                f"Dropping {n_drop} SNPs from LD-scores ({which}) due to non-finite LD values "
                f"or non-positive total LD score."
            )

        df = df.loc[finite].reset_index(drop=True)
        L = L[finite, :]
        snps = snps[finite]
        df["SNP"] = snps
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

        # Try full annotation first.
        try:
            df = pd.read_csv(annot_path, sep=r"\s+", compression="infer")
            if "SNP" not in df.columns:
                raise ValueError("No SNP column found in annotation file.")

            cols = df.columns.tolist()
            first4 = cols[:4]
            required = {"CHR", "BP", "SNP"}
            if not required.issubset(set(first4)):
                raise ValueError("Not a full annotation file.")

            start_idx = 4 if ("CM" in first4) else 3
            annot_cols = cols[start_idx:]
            if len(annot_cols) == 0:
                raise ValueError("Annotation file contains no annotation columns.")

            ann = df[["CHR", "BP", "SNP"] + annot_cols].copy()
            ann["SNP"] = ann["SNP"].astype(str)
            ann = ann.drop_duplicates(subset="SNP", keep="first").reset_index(drop=True)

            aligned = ann.set_index("SNP").reindex(snps_main)
            missing = aligned[annot_cols].isna().all(axis=1)
            n_missing = int(missing.sum())
            if n_missing > 0 and self.log is not None:
                self.log._log(
                    f"Dropping {n_missing} SNPs because they are missing in the annotation file."
                )
            aligned = aligned.loc[~missing].reset_index().rename(columns={"index": "SNP"})

            # Replace CHR/BP with the main LD-based values to ensure a single source of truth.
            keep_mask = pd.Index(snps_main).isin(aligned["SNP"].astype(str).to_numpy())
            chr_keep = np.asarray(chr_main)[keep_mask]
            bp_keep = np.asarray(bp_main)[keep_mask]
            aligned["CHR"] = chr_keep
            aligned["BP"] = bp_keep

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

        except Exception:
            # Thin annotation path.
            if bimpath is not None and Path(str(bimpath)).suffix == ".bim":
                bim_snps = []
                with open(bimpath, "r") as fd:
                    for line in fd:
                        bim_snps.append(line.split()[1])
                bim_snps = np.asarray(bim_snps, dtype=str)
            else:
                bim_snps = snps_main

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
            ann_df = ann_df.drop_duplicates(subset="SNP", keep="first").reset_index(drop=True)

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
        arr = np.genfromtxt(path, dtype=None, encoding=None, comments=None)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Try header parse via first row string-ness.
        try:
            header = np.genfromtxt(path, max_rows=1, dtype=str)
            body = np.genfromtxt(path, skip_header=1, dtype=float)
            if body.ndim == 1:
                body = body.reshape(-1, 1)
            if header.ndim == 0:
                header = np.array([header.item()])
            return header, body
        except Exception:
            body = np.genfromtxt(path, dtype=float)
            if body.ndim == 1:
                body = body.reshape(-1, 1)
            return None, body
