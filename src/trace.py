'''
Read in the trace (or LD) summary statistics to estimate the trace for the summary stats.
For .tr file, there should be (njackknife+1) rows excluding the header, where
the last row is the trace summary (i.e., sum of the LD scores) from the entire genotype.
The SNP sets should be the same for all the outputs -- it is recommended to input .bim for the list of SNPs (read in once)

For .ldscore.gz, standard LDSC ld score format works.
'''
import utils

import numpy as np
import pandas as pd
from os import listdir, path
import sys, math
from itertools import combinations

class Trace:
    def __init__(
        self,
        bimpath=None,
        sumpath=None,
        savepath=None,
        log=None,
        ldscores=None,
        ldscores_reg=None,
        nblks=100,
        annot=None,
        verbose=False,
        adjust_delta: bool = False,
    ):
        self.log = log
        self.sumpath = sumpath
        self.savepath = savepath

        # ----------------------------
        # Jackknife scheme (NEW)
        # ----------------------------
        self.jackknife_mode = "block"   # 'block' or 'chr'
        self.jackknife_delete = 1       # d (only for chr mode)
        self.jackknife_nrep = None      # optional R (only for chr mode)
        self.jackknife_seed = None      # optional RNG seed
        self.jackknife_chrs = None      # (U,) chromosome labels
        self.jackknife_units = 0        # U
        self.jackknife_delete_sets = None  # (R,d) unit indices
        self._rep_del_mat = None        # (R,U) float32

        # Parse nblks / njack spec
        if isinstance(nblks, str):
            s = nblks.strip().lower()
            if s.startswith("chr"):
                self.jackknife_mode = "chr"

                # Support: "chr", "chr2", "chr:2", "chr:2:100", "chr:2:100:0"
                rest = s[3:]
                d = 1
                nrep = None
                seed = None

                if rest == "":
                    pass
                elif rest.startswith(":"):
                    parts = s.split(":")  # parts[0] == 'chr'
                    if len(parts) >= 2 and parts[1] != "":
                        d = int(parts[1])
                    if len(parts) >= 3 and parts[2] != "":
                        nrep = int(parts[2])
                    if len(parts) >= 4 and parts[3] != "":
                        seed = int(parts[3])
                    if len(parts) > 4:
                        raise ValueError(
                            f"Invalid njack spec {nblks!r}. Use chr[:d[:nrep[:seed]]]."
                        )
                else:
                    # "chr2" style
                    if rest.isdigit():
                        d = int(rest)
                    else:
                        raise ValueError(
                            f"Invalid njack spec {nblks!r}. Use chr[:d[:nrep[:seed]]]."
                        )

                if d < 1:
                    raise ValueError(f"delete-d must be >=1; got d={d}")
                if nrep is not None and nrep <= 0:
                    raise ValueError(f"nrep must be positive; got {nrep}")

                self.jackknife_delete = d
                self.jackknife_nrep = nrep
                self.jackknife_seed = seed

                # placeholder: replicate count set after we know U
                self.nblks = 0
            else:
                self.nblks = int(float(s))
        else:
            self.nblks = int(nblks)

        if self.jackknife_mode == "block" and self.nblks <= 0:
            raise ValueError(f"nblks must be positive; got {self.nblks}")

        # Primary (stochastic, used for SUMCORE / trace)
        self.ldscorespath = ldscores
        self.ldscores = None
        self.ldscores_df = None
        self._ldscore_start_idx = None

        # Secondary (windowed, used for regression / intercept)
        self.ldscores_reg_path = ldscores_reg
        self.ldscores_reg = None
        self.ldscores_reg_df = None
        self._ldscore_reg_start_idx = None
        self.nbins_reg = None
        self.chr_reg = None
        self.bp_reg = None

        self.sums = []
        self.ntrace = 0
        self.K = []
        self.snplist = []
        self.nsamp = []
        self.nsnps = 0
        self.nsnps_blk = None
        self.nsnps_bin = None
        self.verbose = verbose
        self.nbins = None
        self.effective_K = None
        self.adjust_delta = adjust_delta

        # Optional: read BIM for SNP names
        if (bimpath is None) or (bimpath == ""):
            if (self.ldscorespath is None) and (self.sumpath is not None):
                self.log._log(
                    "!!! SNP list (.bim) is recommended when using trace summaries (.tr) !!!"
                )
        elif bimpath.endswith(".bim"):
            with open(bimpath, "r") as fd:
                for line in fd:
                    self.snplist.append(line.split()[1])
        else:
            self.log._log(f"!!! {bimpath} is not a .bim file! !!!")

        # Read trace summaries or LD-scores
        if sumpath is not None:
            if self.jackknife_mode == "chr":
                raise ValueError(
                    "delete-d LOCO requires per-SNP LD-scores, not trace summaries (.tr)."
                )
            self._read_all_trace()
            if savepath is not None:
                self._save_trace()

        elif ldscores is not None:
            # --- read primary LD-scores (SUMCORE) ---
            self._read_ldscores(path=self.ldscorespath, which="main")

            # --- optionally read regression LD-scores (windowed) and align to main order ---
            if self.ldscores_reg_path is not None:
                self._read_ldscores(path=self.ldscores_reg_path, which="reg")

                snps_main = np.asarray(self.snplist, dtype=str)
                reg_index = pd.Index(self.ldscores_reg_df["SNP"].astype(str).to_numpy())
                idx = reg_index.get_indexer(snps_main)

                keep = idx >= 0
                n_drop = int((~keep).sum())
                if n_drop > 0:
                    self.log._log(
                        f"Dropping {n_drop} SNPs from primary LD-scores because they are missing in ldscores_reg."
                    )

                snps_keep = snps_main[keep]
                main_aligned = (
                    self.ldscores_df.set_index("SNP").loc[snps_keep].reset_index()
                )
                self.ldscores_df = main_aligned

                sidx = self._ldscore_start_idx
                self.ldscores = main_aligned.iloc[:, sidx:].to_numpy(
                    dtype=np.float64, copy=False
                )
                self.chr = main_aligned["CHR"].to_numpy(dtype=np.int32, copy=False)
                self.bp = main_aligned["BP"].to_numpy(dtype=np.int64, copy=False)
                self.snplist = snps_keep.tolist()
                self.nsnps = int(self.ldscores.shape[0])

                snps_main2 = np.asarray(self.snplist, dtype=str)
                reg_aligned = (
                    self.ldscores_reg_df.set_index("SNP")
                    .loc[snps_main2]
                    .reset_index()
                )
                self.ldscores_reg_df = reg_aligned

                sidxr = self._ldscore_reg_start_idx
                self.ldscores_reg = reg_aligned.iloc[:, sidxr:].to_numpy(
                    dtype=np.float64, copy=False
                )
                self.chr_reg = reg_aligned["CHR"].to_numpy(dtype=np.int32, copy=False)
                self.bp_reg = reg_aligned["BP"].to_numpy(dtype=np.int64, copy=False)
                self.nbins_reg = int(self.ldscores_reg.shape[1])

                self.log._log(
                    f"Loaded ldscores_reg with {self.ldscores_reg.shape[0]} SNPs and "
                    f"{self.ldscores_reg.shape[1]} bins (aligned to primary SNP order)."
                )

            # Read annotations
            self._read_annot(annot)

            # If regression LD is present, prune it to match annotation-filtered SNP list
            if (
                self.ldscores_reg_df is not None
                and self.snplist is not None
                and len(self.snplist) > 0
            ):
                snps_now = np.asarray(self.snplist, dtype=str)
                reg_index_now = self.ldscores_reg_df.set_index("SNP")

                missing = ~pd.Index(snps_now).isin(reg_index_now.index)
                if missing.any():
                    miss_snps = snps_now[missing].tolist()
                    self.log._log(
                        f"Dropping {len(miss_snps)} SNPs because they are missing in ldscores_reg after annotation pruning."
                    )
                    keep_mask_tmp = ~missing
                    self.snplist = snps_now[keep_mask_tmp].tolist()
                    self.annot = np.asarray(self.annot)[keep_mask_tmp, :]
                    self.nsnps = int(len(self.snplist))

                    if self.ldscores is not None:
                        self.ldscores = np.asarray(self.ldscores)[keep_mask_tmp, :]
                        self.chr = np.asarray(self.chr)[keep_mask_tmp]
                        self.bp = np.asarray(self.bp)[keep_mask_tmp]
                        self.ldscores_df = (
                            self.ldscores_df.loc[keep_mask_tmp].reset_index(drop=True)
                        )

                reg_aligned2 = reg_index_now.loc[self.snplist].reset_index()
                self.ldscores_reg_df = reg_aligned2

                sidxr = self._ldscore_reg_start_idx
                self.ldscores_reg = reg_aligned2.iloc[:, sidxr:].to_numpy(
                    dtype=np.float64, copy=False
                )
                self.chr_reg = reg_aligned2["CHR"].to_numpy(dtype=np.int32, copy=False)
                self.bp_reg = reg_aligned2["BP"].to_numpy(dtype=np.int64, copy=False)
                self.nbins_reg = int(self.ldscores_reg.shape[1])

        # ----------------------------
        # If LOCO mode, enforce sorting by (CHR,BP,SNP) and build replicate sets
        # ----------------------------
        if self.jackknife_mode == "chr":
            if self.ldscores is None or not hasattr(self, "chr"):
                raise ValueError(
                    "delete-d LOCO requires per-SNP LD-scores with CHR available."
                )

            chr_arr = np.asarray(self.chr, dtype=np.int32)
            bp_arr = np.asarray(self.bp, dtype=np.int64)
            snp_arr = np.asarray(self.snplist, dtype=str)

            order = np.lexsort((snp_arr, bp_arr, chr_arr))
            if not np.array_equal(order, np.arange(order.size)):
                self.log._log(
                    "[Trace] chr-mode: sorting SNPs by (CHR,BP,SNP) for chromosome-contiguous units."
                )

                self.snplist = snp_arr[order].tolist()
                self.annot = np.asarray(self.annot)[order, :]
                self.nsnps = int(self.annot.shape[0])

                self.ldscores = np.asarray(self.ldscores)[order, :]
                self.chr = chr_arr[order]
                self.bp = bp_arr[order]

                if self.ldscores_df is not None:
                    self.ldscores_df = self.ldscores_df.iloc[order].reset_index(
                        drop=True
                    )

                if self.ldscores_reg is not None:
                    self.ldscores_reg = np.asarray(self.ldscores_reg)[order, :]
                if self.ldscores_reg_df is not None:
                    self.ldscores_reg_df = self.ldscores_reg_df.iloc[order].reset_index(
                        drop=True
                    )
                if self.chr_reg is not None:
                    self.chr_reg = np.asarray(self.chr_reg)[order]
                if self.bp_reg is not None:
                    self.bp_reg = np.asarray(self.bp_reg)[order]

                # rebuild annot_df to match the sorted order
                adf = pd.DataFrame(self.annot, columns=self.annot_header)
                adf.insert(0, "SNP", self.snplist)
                self.annot_df = adf

            # Ensure annot_df has CHR/BP for Sumstats LOCO
            if self.annot_df is not None:
                if "CHR" not in self.annot_df.columns:
                    self.annot_df.insert(0, "CHR", np.asarray(self.chr, dtype=np.int32))
                if "BP" not in self.annot_df.columns:
                    self.annot_df.insert(1, "BP", np.asarray(self.bp, dtype=np.int64))

            # Units = chromosomes
            chrs = np.unique(np.asarray(self.chr, dtype=np.int32))
            chrs = chrs[np.isfinite(chrs)]
            chrs = np.asarray(chrs, dtype=np.int32)
            chrs.sort()

            self.jackknife_chrs = chrs
            U = int(chrs.size)
            self.jackknife_units = U

            d = int(self.jackknife_delete)
            if not (1 <= d < U):
                raise ValueError(
                    f"delete-d must satisfy 1 <= d < #chromosomes; got d={d}, U={U}."
                )

            total = math.comb(U, d)
            nrep = self.jackknife_nrep

            if (nrep is None) or (nrep >= total):
                # use all combinations
                del_sets = np.fromiter(
                    (x for combi in combinations(range(U), d) for x in combi),
                    dtype=np.int16,
                    count=total * d,
                ).reshape(total, d)
                R = int(total)
                self.log._log(
                    f"[Trace] chr delete-{d}: using ALL combinations: C({U},{d}) = {R} replicates."
                )
            else:
                # sample without replacement uniformly from all combinations
                all_combos = list(combinations(range(U), d))
                rng = np.random.default_rng(self.jackknife_seed)
                pick = rng.choice(len(all_combos), size=int(nrep), replace=False)
                del_sets = np.asarray([all_combos[i] for i in pick], dtype=np.int16)
                R = int(del_sets.shape[0])
                self.log._log(
                    f"[Trace] chr delete-{d}: using RANDOM {R} replicates out of "
                    f"C({U},{d})={total} (seed={self.jackknife_seed})."
                )

            self.jackknife_delete_sets = del_sets
            self.nblks = R  # IMPORTANT: nblks == number of jackknife replicates now

            # deletion incidence matrix D: (R,U)
            D = np.zeros((R, U), dtype=np.float32)
            rr = np.arange(R, dtype=np.int64)[:, None]
            D[rr, del_sets.astype(np.int64)] = 1.0
            self._rep_del_mat = D

            # Base SNP universe (for fast filtering)
            self._base_snps = self.annot_df["SNP"].astype(str).to_numpy()
            self.snp_index = pd.Index(self._base_snps)

            # Base arrays (no copy; unfiltered aligned arrays)
            self._annot_base = np.asarray(self.annot)
            self._ldscores_base = (
                np.asarray(self.ldscores) if (self.ldscores is not None) else None
            )

            # Also cache reg LD as base if present
            self._ldscores_reg_base = (
                np.asarray(self.ldscores_reg)
                if (self.ldscores_reg is not None)
                else None
            )

            # chr/bp bases
            if self._ldscores_base is not None:
                self._chr_base = np.asarray(self.chr)
                self._bp_base = np.asarray(self.bp)
            else:
                self._chr_base = None
                self._bp_base = None

            if self._ldscores_reg_base is not None:
                self._chr_reg_base = (
                    np.asarray(self.chr_reg) if (self.chr_reg is not None) else None
                )
                self._bp_reg_base = (
                    np.asarray(self.bp_reg) if (self.bp_reg is not None) else None
                )
            else:
                self._chr_reg_base = None
                self._bp_reg_base = None

        # Refresh bounds
        if self.jackknife_mode == "chr":
            self._refresh_unit_bounds()
        else:
            if (self.nsnps > 0) and (self.nblks > 0) and not hasattr(self, "blk_size"):
                self.blk_size = max(self.nsnps // self.nblks, 1)
            self._refresh_block_bounds()

    # ----------------------------
    # I/O & setup
    # ----------------------------
    def _refresh_unit_bounds(self):
        """
        (chr-mode) Compute per-chromosome unit bounds on CURRENT SNP order.
        IMPORTANT: does NOT touch _blk_starts/_blk_ends (those are block-mode only).
        """
        if self.jackknife_mode != "chr":
            return
        if self.jackknife_chrs is None:
            raise RuntimeError("jackknife_chrs is None in chr-mode.")

        chr_arr = np.asarray(self.chr, dtype=np.int32).ravel()
        chrs = np.asarray(self.jackknife_chrs, dtype=np.int32).ravel()
        U = int(chrs.size)

        if chr_arr.size > 1 and np.any(chr_arr[1:] < chr_arr[:-1]):
            raise RuntimeError("chr-mode requires SNPs sorted by CHR (and BP).")

        starts = np.searchsorted(chr_arr, chrs, side="left").astype(np.int64)
        ends = np.searchsorted(chr_arr, chrs, side="right").astype(np.int64)

        self._unit_starts = starts
        self._unit_ends = ends
        self._unit_sizes = (ends - starts).astype(np.int64)

        # unit index per SNP
        uidx = np.searchsorted(chrs, chr_arr).astype(np.int64)
        valid = (uidx >= 0) & (uidx < U) & (chrs[uidx] == chr_arr)
        if not np.all(valid):
            bad = np.flatnonzero(~valid)[:10].tolist()
            raise RuntimeError(
                f"chr-mode: found CHR not in jackknife_chrs. First bad indices: {bad}"
            )
        self.unit_idx = uidx

    def get_jackknife_unit_sizes(self, dtype=None):
        """Return unit (chromosome) sizes after filtering, shape (U,)."""
        if getattr(self, "jackknife_mode", "block") != "chr":
            return None
        if not hasattr(self, "_unit_sizes"):
            self._refresh_unit_bounds()

        out = np.asarray(self._unit_sizes, dtype=np.float64)
        if dtype is None:
            return out
        return out.astype(dtype, copy=False)

    def _refresh_block_bounds(self):
        """Compute and cache per-block slice bounds (after any filtering)."""
        if (self.nsnps <= 0) or (self.nblks <= 0):
            return

        B = int(self.nblks)

        if getattr(self, "jackknife_mode", "block") == "chr":
            if self.jackknife_chrs is None:
                if not hasattr(self, "chr") or self.chr is None:
                    raise RuntimeError("LOCO jackknife requires self.chr.")
                chrs = np.unique(np.asarray(self.chr, dtype=np.int32))
                chrs = chrs[np.isfinite(chrs)]
                chrs = np.asarray(chrs, dtype=np.int32)
                chrs.sort()
                self.jackknife_chrs = chrs
                self.nblks = int(chrs.size)
                B = int(self.nblks)

            chrs = np.asarray(self.jackknife_chrs, dtype=np.int32).ravel()
            if chrs.size != B:
                raise RuntimeError(f"jackknife_chrs length {chrs.size} != nblks {B}")

            chr_arr = np.asarray(self.chr, dtype=np.int32).ravel()
            if chr_arr.size != self.nsnps:
                raise RuntimeError("chr array length mismatch with nsnps")

            # requires non-decreasing chr order for searchsorted bounds
            if np.any(chr_arr[1:] < chr_arr[:-1]):
                raise RuntimeError(
                    "LOCO mode requires SNPs sorted by CHR (and BP). "
                    "Trace.__init__ should sort to genomic order."
                )

            starts = np.searchsorted(chr_arr, chrs, side="left").astype(np.int64)
            ends = np.searchsorted(chr_arr, chrs, side="right").astype(np.int64)
            self._blk_starts = starts
            self._blk_ends = ends

            # blk_idx: map each SNP chr -> block index (0..B-1)
            blk_idx = np.searchsorted(chrs, chr_arr).astype(np.int64)
            valid = (blk_idx >= 0) & (blk_idx < B) & (chrs[blk_idx] == chr_arr)
            if not np.all(valid):
                bad = np.flatnonzero(~valid)[:10].tolist()
                raise RuntimeError(
                    f"Found SNPs with chromosomes not in jackknife_chrs. First bad indices: {bad}"
                )
            self.blk_idx = blk_idx

            # keep blk_size for compatibility (not used for LOCO bounds)
            self.blk_size = max(self.nsnps // B, 1)
            return

        # -------- default: contiguous equal-size blocks --------
        self.blk_size = max(getattr(self, "blk_size", self.nsnps // B), 1)
        starts = self.blk_size * np.arange(B, dtype=np.int64)
        ends = starts + self.blk_size
        ends[-1] = self.nsnps

        self._blk_starts = starts
        self._blk_ends = ends

        blk_idx = np.arange(self.nsnps, dtype=np.int64) // self.blk_size
        blk_idx[blk_idx >= B] = B - 1
        self.blk_idx = blk_idx

    def _read_annot(self, annot_path):
        # single-bin fallback
        if annot_path is None:
            self.annot_header = np.array(["L2"])
            annot = np.ones((self.nsnps, 1), dtype=np.float32)
            annot_df = pd.DataFrame(
                annot, index=self.snplist, columns=self.annot_header.tolist()
            )
            annot_df.reset_index(inplace=True)
            annot_df.rename(columns={"index": "SNP"}, inplace=True)

            self.annot_df = annot_df
            self.annot = self.annot_df[self.annot_header].values
            self.nbins = 1

            # full-bin counts from annotation
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)
            self.log._log("Running with single component annotation...")
            return

        try:
            # full .annot or .annot.gz
            df = pd.read_csv(annot_path, sep=r"\s+", compression="infer")
            if "SNP" not in df.columns:
                raise ValueError("!!! Input annotation file is not in correct format !!!")

            cols = df.columns.tolist()
            first4 = cols[:4]
            must = {"CHR", "BP", "SNP"}
            if not must.issubset(set(first4)):
                raise ValueError(
                    "!!! Input annotation file is not in correct format: "
                    "first columns must include CHR,BP,SNP (and optional CM) !!!"
                )

            start_idx = 4 if ("CM" in first4) else 3
            annot_cols = cols[start_idx:]  # bins start here
            self.annot_header = np.array(annot_cols)

            # Only keep SNP + annotation columns
            annot_df = df[["SNP"] + annot_cols].copy()

            if self.snplist:
                # Align to BIM order; drop SNPs missing from annotation
                ann_indexed = annot_df.set_index("SNP")
                aligned = ann_indexed.reindex(self.snplist)  # BIM order, NaN for missing

                missing_mask = aligned[annot_cols].isna().all(axis=1)
                n_missing = int(missing_mask.sum())
                if n_missing:
                    self.log._log(
                        f"Dropping {n_missing} SNPs from annotation as they are missing LD information."
                    )
                    aligned = aligned[~missing_mask]

                self.annot_df = aligned.reset_index().rename(columns={"index": "SNP"})
                self.annot = self.annot_df[annot_cols].values
                self.nsnps = self.annot.shape[0]
                self.snplist = self.annot_df["SNP"].tolist()
            else:
                # no BIM provided; keep all rows
                self.annot_df = annot_df.copy()
                self.annot = self.annot_df[annot_cols].values
                self.nsnps = self.annot.shape[0]
                self.snplist = self.annot_df["SNP"].tolist()

            self.nbins = len(annot_cols)
            self.log._log("Read full annotation of shape " + str(self.annot.shape))

            # full-bin counts from annotation
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)

            # prune LD-scores if present
            if getattr(self, "ldscores", None) is not None:
                ld_df = self.ldscores_df.set_index("SNP").loc[self.snplist].reset_index()
                self.ldscores_df = ld_df

                start_idx = getattr(self, "_ldscore_start_idx", None)
                if start_idx is None:
                    ldcols = self.ldscores_df.columns.tolist()
                    first4 = ldcols[:4]
                    start_idx = 4 if ("CM" in first4) else 3
                    self._ldscore_start_idx = start_idx

                self.ldscores = ld_df.iloc[:, start_idx:].to_numpy()
                self.chr = self.ldscores_df["CHR"].to_numpy(dtype=np.int32, copy=False)
                self.bp = self.ldscores_df["BP"].to_numpy(dtype=np.int64, copy=False)
                self.log._log(
                    f"Pruned LD-score to {self.nsnps} SNPs that match the annotation file."
                )

        except ValueError:
            # thin annotation
            if not self.snplist:
                raise ValueError(
                    "!!! Thin annotation requires a BIM/snplist when using trace-summaries !!!"
                )

            self.annot_header, self.annot = utils._read_with_optional_header(annot_path)

            if self.annot.ndim == 1:
                self.annot = self.annot.reshape(-1, 1)
            else:
                self.annot = self.annot.reshape(-1, self.annot.shape[-1])

            if self.annot_header is None:
                self.annot_header = np.array(
                    [f"bin_{i}" for i in range(self.annot.shape[1])]
                )

            self.annot_header = list(self.annot_header)
            self.annot_df = pd.DataFrame(
                self.annot, index=self.snplist, columns=self.annot_header
            )
            self.annot_df.reset_index(inplace=True)
            self.annot_df.rename(columns={"index": "SNP"}, inplace=True)

            self.nsnps = self.annot.shape[0]
            self.nbins = self.annot.shape[1]
            self.log._log("Read thin annotation matrix of shape " + str(self.annot.shape))

            # full-bin counts from annotation
            self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)

        if (self.nbins is None) or (self.nsnps is None):
            self.log._log("!!! number of components or SNP count unresolved !!!")
            sys.exit(1)

    def _save_trace(self):
        """Save trace summaries as files (format preserved from your original code)."""
        with open(self.savepath + ".MN", "w") as fd:
            fd.write("NSAMPLE,NSNPS,NBLKS,NBINS,K\n")
            fd.write(
                f"{self.nsamp:.0f},{self.nsnps:.0f},{self.nblks:.0f},{self.nbins:.0f},{self.effective_K:.0f}"
            )

        with open(self.savepath + ".tr", "w") as fd:
            header_str = ",".join(f"LD_SUM_{i:d}" for i in range(self.nbins))
            fd.write(header_str + ",NSNPS_JACKKNIFE\n")
            for j in range(self.nblks + 1):
                for k in range(self.nbins):
                    row_str = ",".join(
                        f"{self.sums[j, k, l]:.3f}" for l in range(self.nbins)
                    )
                    row_str += f",{self.nsnps_blk[j, k]:.0f}\n"
                    fd.write(row_str)

        self.log._log(f"Saved trace summary into {self.savepath}(.tr/.MN)")

    def _read_trace(self, filename, idx):
        """Read trace summaries (block-wise LD scores)."""
        # read metadata
        with open(filename + ".MN", "r") as fd:
            next(fd)
            nsamp, nsnps, nblks, nbins, K = map(int, fd.readline().split(","))

        self.nsamp.append(nsamp)
        self.K.append(K)

        if idx == 0:
            self.nsnps = nsnps
            self.nblks = nblks
            self.nbins = nbins
        elif nblks != self.nblks:
            self.log._log(
                "!!! Trace summary "
                + filename
                + " has incorrect number of jackknife blocks !!!"
            )
            return
        elif nbins != self.nbins:
            self.log._log(
                "!!! Trace summary "
                + filename
                + " has incorrect number of annotation bins !!!"
            )
            return

        sums = np.zeros((self.nblks + 1, self.nbins, self.nbins))
        nsnps_blk = np.zeros((self.nblks + 1, self.nbins))

        # read values
        for cnt, vals in enumerate(utils._read_multiple_lines(filename + ".tr", self.nbins)):
            sums[cnt] = vals[:, :-1]
            nsnps_blk[cnt] = vals[:, -1].transpose()

        if idx == 0:
            self.nsnps_blk = nsnps_blk
        elif not np.array_equal(self.nsnps_blk, nsnps_blk):
            self.log._log("!!! Trace summary " + filename + " has different annotations !!!")
            sys.exit(1)

        self.sums.append(sums)
        self.ntrace += 1

        self.log._log(
            f"Read in trace summaries from {filename} generated with {K} random vectors"
        )
        if self.verbose:
            self.log._log(
                "-- avg. jackknife LDscore sum:\n"
                + np.array2string(sums[:-1].mean(axis=0), precision=3, separator=", ")
                + "\n-- number of jackknife blocks:\t"
                + str(self.nblks)
                + "\n-- genome-wide LDscore sum:\n"
                + np.array2string(sums[-1], precision=3, separator=", ")
            )

        return self.sums

    def _read_all_trace(self):
        trace_files = [self.sumpath]
        if path.isdir(self.sumpath):
            prefix = self.sumpath.rstrip("/") + "/"
            trace_files = sorted(
                [prefix + f.rstrip(".tr") for f in listdir(self.sumpath) if f.endswith(".tr")]
            )

        for i, f in enumerate(trace_files):
            self._read_trace(f, i)

        self.log._log("Finished reading " + str(len(trace_files)) + " trace summaries.")

        if (np.std(self.sums, axis=0).sum() == 0.0) and (self.ntrace > 1):
            self.log._log(
                "!!! Duplicate trace summaries are used -- effective number of random vectors remains unchanged. !!!"
            )
            self.effective_K = np.min(self.K)
        else:
            self.effective_K = np.sum(self.K)

        # weighted average by sample size
        self.sums = np.average(self.sums, axis=0, weights=self.nsamp)
        self.nsamp = float(np.mean(self.nsamp))

    def _read_ldscores(self, path: str | None = None, which: str = "main"):
        """
        Read an LD-score matrix.
        Supports either CHR,BP,SNP,(optional CM), then LD-score columns.

        which="main": populates
            self.ldscores / self.ldscores_df / self.snplist / self.chr,bp / self.nsnps,self.nbins
        which="reg" : populates
            self.ldscores_reg / self.ldscores_reg_df / self.chr_reg,bp_reg / self.nbins_reg
            (does NOT override snplist order)
        """
        if path is None:
            path = self.ldscorespath if which == "main" else self.ldscores_reg_path
        if path is None:
            raise ValueError("Trace._read_ldscores: path is None")

        df = pd.read_csv(path, compression="infer", sep=r"\s+", index_col=False)

        ldcols = df.columns.tolist()
        first4 = ldcols[:4]
        must = {"CHR", "BP", "SNP"}
        if not must.issubset(set(first4)):
            raise ValueError(
                "!!! Input LD score file is not in correct format: "
                "first columns must include CHR,BP,SNP (and optional CM) !!!"
            )

        start_idx = 4 if ("CM" in first4) else 3

        snps = df["SNP"].astype(str).to_numpy()
        L = df.iloc[:, start_idx:].to_numpy(dtype=np.float64, copy=False)

        # Drop non-finite LD-score rows
        finite_row = np.isfinite(L).all(axis=1)
        if which == "reg":
            # Drop rows with non-positive total LD score
            ltot = L.sum(axis=1)
            good_ltot = np.isfinite(ltot) & (ltot > 0.1)
            keep = finite_row & good_ltot
        else:
            keep = finite_row

        n_drop = int((~keep).sum())
        if n_drop > 0:
            self.log._log(
                f"Dropping {n_drop} SNPs from LD-scores ({which}) due to non-finite LD values "
                f"or non-positive total LD score."
            )
            df = df.loc[keep].reset_index(drop=True)
            snps = snps[keep]
            L = L[keep, :]

        chr_ = df["CHR"].to_numpy(dtype=np.int32, copy=False)
        bp_ = df["BP"].to_numpy(dtype=np.int64, copy=False)

        if which == "main":
            self.ldscores_df = df
            self.ldscores = L
            self.snplist = snps.tolist()
            self.nsnps = int(L.shape[0])
            self.nbins = int(L.shape[1])
            self.chr = chr_
            self.bp = bp_
            self._ldscore_start_idx = start_idx

            self.log._log(
                f"Loaded the LD score matrix with {self.nsnps} SNPs and {self.nbins} bins "
                f"(dropped {n_drop})."
            )

        elif which == "reg":
            self.ldscores_reg_df = df
            self.ldscores_reg = L
            self.nbins_reg = int(L.shape[1])
            self.chr_reg = chr_
            self.bp_reg = bp_
            self._ldscore_reg_start_idx = start_idx

            self.log._log(
                f"Loaded the regression LD score matrix (ldscores_reg) with {L.shape[0]} SNPs "
                f"and {L.shape[1]} bins (dropped {n_drop})."
            )

        else:
            raise ValueError("which must be 'main' or 'reg'")

    # ----------------------------
    # public API
    # ----------------------------
    def _calc_trace(self, nsample: float):
        if self.verbose:
            self.log._log("Calculating trace...")
        if self.ldscores is not None:
            return self._calc_trace_from_ldscores(nsample)
        return self._calc_trace_from_sums(nsample)

    # ----------------------------
    # vectorized core
    # ----------------------------
    def get_jackknife_delete_matrix(self, dtype=None):
        if self.jackknife_mode != "chr":
            return None
        D = self._rep_del_mat
        if dtype is None:
            return D
        return np.asarray(D, dtype=dtype, order="C")

    def _calc_trace_from_sums(self, N: float):
        """Vectorized trace when pre-aggregated trace summaries are provided."""
        K = self.nbins
        B = self.nblks
        sums = np.asarray(self.sums, dtype=np.float64)  # (B+1, K, K)

        if self.nsnps_blk is None:
            raise RuntimeError("nsnps_blk must be available when using trace summaries.")

        M_k = self.nsnps_blk.astype(np.float64)[:, :, None]  # (B+1, K, 1)
        M_l = self.nsnps_blk.astype(np.float64)[:, None, :]  # (B+1, 1, K)

        trace_KK = utils._calc_trace_from_ld_batch(sums, N, M_k, M_l)  # (B+1, K, K)

        trace_KK = utils.symmetrize_trace_with_jackknife(
            trace_KK,
            logger=self.log,
            verbose=self.verbose,
            jk_block_sizes=(self._blk_ends - self._blk_starts),
        )

        out = np.full((B + 1, K + 1, K + 1), float(N), dtype=np.float64)
        out[:, :K, :K] = trace_KK
        # IMPORTANT: match RHS convention (nsamp - 1) for the noise term
        out[:, K, K] = float(N - 1)
        return out

    def _calc_trace_from_ldscores(self, N: float):
        """
        Vectorized trace from LD-scores. Uses get_ldsum_all() so that chr-mode
        delete-d and exact LOCO both go through the same replicate LD-sum
        construction.

        In chr-mode, replicate rows therefore include the deleted-source
        correction implemented in get_ldsum_all().
        """
        import numpy as np
        import utils

        N = float(N)
        if not (np.isfinite(N) and N > 0):
            raise ValueError(f"N must be positive finite; got {N!r}")

        delta = getattr(self, "delta", None) if getattr(self, "adjust_delta", False) else None

        K = int(self.nbins)
        R = int(self.nblks)

        ld_sum_all = self.get_ldsum_all(use_cache=True)

        if (
            getattr(self, "nsnps_blk", None) is None
            or np.asarray(self.nsnps_blk).shape != (R + 1, K)
        ):
            raise RuntimeError(
                "Trace._calc_trace_from_ldscores: nsnps_blk missing or wrong shape."
            )

        nsnps_blk = np.asarray(self.nsnps_blk, dtype=np.float64, order="C")
        M_k = nsnps_blk[:, :, None]  # (R+1, K, 1)
        M_l = nsnps_blk[:, None, :]  # (R+1, 1, K)

        trace_KK = utils._calc_trace_from_ld_batch(ld_sum_all, N, M_k, M_l, delta=delta)

        # ------------------------------------------------------------
        # symmetrization
        # ------------------------------------------------------------
        if getattr(self, "jackknife_mode", "block") != "chr":
            if not hasattr(self, "_blk_starts") or not hasattr(self, "_blk_ends"):
                self._refresh_block_bounds()

            m_b = (
                np.asarray(self._blk_ends, dtype=np.float64)
                - np.asarray(self._blk_starts, dtype=np.float64)
            )

            trace_KK = utils.symmetrize_trace_with_jackknife(
                trace_KK,
                logger=self.log,
                verbose=getattr(self, "verbose", False),
                jk_block_sizes=m_b,
                jk_delete_d=1,
            )
        else:
            U = int(getattr(self, "jackknife_units", 0))
            d = int(getattr(self, "jackknife_delete", 1))

            if not hasattr(self, "_unit_starts") or not hasattr(self, "_unit_ends"):
                self._refresh_unit_bounds()

            if d == 1 and R == U:
                m_chr = (
                    np.asarray(self._unit_ends, dtype=np.float64)
                    - np.asarray(self._unit_starts, dtype=np.float64)
                )
                trace_KK = utils.symmetrize_trace_with_jackknife(
                    trace_KK,
                    logger=self.log,
                    verbose=getattr(self, "verbose", False),
                    jk_block_sizes=m_chr,
                    jk_delete_d=1,
                )
            else:
                trace_KK = utils.symmetrize_trace_with_jackknife(
                    trace_KK,
                    logger=self.log,
                    verbose=getattr(self, "verbose", False),
                    jk_n_units=U,
                    jk_delete_d=d,
                )

        out = np.full((R + 1, K + 1, K + 1), float(N), dtype=np.float64)
        out[:, :K, :K] = trace_KK
        out[:, K, K] = float(N - 1.0)
        return out

    # ----------------------------
    # filtering
    # ----------------------------
    def _apply_keep_mask(self, keep_mask: "np.ndarray"):
        if self._ldscores_base is None:
            return

        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.ndim != 1 or keep_mask.size != self._base_snps.size:
            raise ValueError("keep_mask must be a 1D boolean mask over base SNPs.")

        if hasattr(self, "_ldsum_all_cache"):
            delattr(self, "_ldsum_all_cache")

        self.nsnps = int(keep_mask.sum())
        self.snplist = self._base_snps[keep_mask].tolist()
        self.annot = self._annot_base[keep_mask, :]
        self.ldscores = self._ldscores_base[keep_mask, :]

        if self._chr_base is not None:
            self.chr = self._chr_base[keep_mask]
            self.bp = self._bp_base[keep_mask]

        if getattr(self, "_ldscores_reg_base", None) is not None:
            self.ldscores_reg = self._ldscores_reg_base[keep_mask, :]
            if getattr(self, "_chr_reg_base", None) is not None:
                self.chr_reg = self._chr_reg_base[keep_mask]
                self.bp_reg = self._bp_reg_base[keep_mask]

        self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)

        if self.jackknife_mode == "chr":
            self._refresh_unit_bounds()
            # nsnps_blk is replicate-specific now; computed inside _calc_trace_from_ldscores()
            self.nsnps_blk = None
            return

        # ---- block mode legacy behavior ----
        B = int(self.nblks)
        self.blk_size = max(self.nsnps // max(B, 1), 1)
        self._refresh_block_bounds()

        K = int(self.nbins)
        self.nsnps_blk = np.empty((B + 1, K), dtype=np.float64)
        self.nsnps_blk[B] = self.nsnps_bin

        starts = self.blk_size * np.arange(B, dtype=np.int64)
        ends = starts + self.blk_size
        if B > 0:
            ends[-1] = self.nsnps

        csum = np.cumsum(self.annot, axis=0, dtype=np.float64)
        for j in range(B):
            s = int(starts[j])
            e = int(ends[j])

            if e <= s:
                blk_sum = np.zeros(K, dtype=np.float64)
            elif s == 0:
                blk_sum = csum[e - 1]
            else:
                blk_sum = csum[e - 1] - csum[s - 1]

            self.nsnps_blk[j] = self.nsnps_bin - blk_sum

    def _filter_snps(self, removesnps):
        """
        Remove SNPs in removesnps from trace calculation (LD-scores input only).
        Faster implementation: uses precomputed SNP indexer + base arrays.
        """
        if self.ldscores is None:
            return

        if removesnps is None or len(removesnps) == 0:
            keep_mask = np.ones(self._base_snps.size, dtype=bool)
            self._apply_keep_mask(keep_mask)
            return

        rm = np.asarray(removesnps, dtype=str)
        idx = self.snp_index.get_indexer(rm)
        idx = idx[idx >= 0]

        keep_mask = np.ones(self._base_snps.size, dtype=bool)
        keep_mask[idx] = False
        n_removed = int((~keep_mask).sum())

        self._apply_keep_mask(keep_mask)

        self.log._log(
            f"Filtered {n_removed} SNPs from the Trace module. "
            f"Shape of final annotation used for analysis: {self.annot.shape}"
        )

        if n_removed / len(self.annot_df) > 0.01:
            self.log._log(
                "[WARNING: Removing too many Trace SNPs will result in under-estimated heritability!]\n"
                "[We recommend using a better curated reference LD score panel with a more similar SNP set to the summary statistics SNPs.]"
            )

    def _filter_keep_snps(self, keep_snps):
        """Keep only SNPs listed in keep_snps (LD-scores input only)."""
        if self.ldscores is None:
            return

        ks = np.asarray(keep_snps, dtype=str)
        idx = self.snp_index.get_indexer(ks)
        idx = idx[idx >= 0]

        keep_mask = np.zeros(self._base_snps.size, dtype=bool)
        keep_mask[idx] = True
        self._apply_keep_mask(keep_mask)

    def get_ldsum_all(self, use_cache: bool = True):
        """
        Return ld_sum_all with shape (R+1, K, K), where rows 0..R-1 are jackknife replicates
        and row R is the full sample.

        block-mode:
            ld_sum_all[b] = A_{-b}^T L_{-b}
            ld_sum_all[B] = A^T L

        chr-mode:
            raw row-delete gives A_{-S}^T L_full which still contains deleted-source
            contributions in the kept rows. When self.adjust_delta and self.delta are
            available, we subtract the missing deleted-source null term:
                correction = Delta(A_keep, A_del)
            so that replicate rows are internally consistent with leave-set LD under
            the "cross-chrom true LD ~ 0" approximation.
        """
        if self.ldscores is None:
            raise RuntimeError("Trace.get_ldsum_all requires LD-scores input.")

        R = int(self.nblks)
        K = int(self.nbins)

        if use_cache and hasattr(self, "_ldsum_all_cache"):
            cached = self._ldsum_all_cache
            if (
                cached.shape == (R + 1, K, K)
                and getattr(self, "nsnps_blk", None) is not None
                and np.asarray(self.nsnps_blk).shape == (R + 1, K)
            ):
                return cached

        A = np.asarray(self.annot, dtype=np.float64, order="C")     # (M, K)
        L = np.asarray(self.ldscores, dtype=np.float64, order="C")  # (M, K)

        # ------------------------------------------------------------------
        # helper: deleted-source correction for chr delete-d replicates
        # ------------------------------------------------------------------
        def _pair_correction_from_deleted_mass(A_keep, A_del, delta):
            """
            A_keep: (R, K)
            A_del : (R, K)
            Returns correction with shape (R, K, K).

            Supported delta shapes:
              - scalar
              - (K,)   : source-bin specific null term
              - (K, K) : full pairwise null matrix
            """
            A_keep = np.asarray(A_keep, dtype=np.float64)
            A_del = np.asarray(A_del, dtype=np.float64)
            d = np.asarray(delta, dtype=np.float64)

            if d.ndim == 0:
                return (A_keep[:, :, None] * A_del[:, None, :]) * float(d)

            if d.ndim == 1:
                if d.size != K:
                    raise ValueError(f"delta vector has length {d.size}, expected K={K}.")
                return A_keep[:, :, None] * (A_del * d[None, :])[:, None, :]

            if d.ndim == 2:
                if d.shape != (K, K):
                    raise ValueError(
                        f"delta matrix has shape {d.shape}, expected ({K},{K})."
                    )
                return (A_keep[:, :, None] * A_del[:, None, :]) * d[None, :, :]

            raise ValueError("delta must be scalar, (K,), or (K,K).")

        # ============================================================
        # block-mode: unchanged
        # ============================================================
        if getattr(self, "jackknife_mode", "block") != "chr":
            if not hasattr(self, "_blk_starts") or not hasattr(self, "_blk_ends"):
                self._refresh_block_bounds()

            full_ld = A.T @ L  # (K, K)
            blk_ld = np.empty((R, K, K), dtype=np.float64)
            counts_blk = np.empty((R, K), dtype=np.float64)

            for b in range(R):
                s = int(self._blk_starts[b])
                e = int(self._blk_ends[b])

                if e <= s:
                    blk_ld[b].fill(0.0)
                    counts_blk[b].fill(0.0)
                    continue

                Ab = A[s:e, :]
                Lb = L[s:e, :]
                blk_ld[b] = Ab.T @ Lb
                counts_blk[b] = Ab.sum(axis=0, dtype=np.float64)

            ld_sum_all = np.empty((R + 1, K, K), dtype=np.float64)
            ld_sum_all[:R] = full_ld[None, :, :] - blk_ld
            ld_sum_all[R] = full_ld

            counts_full = A.sum(axis=0, dtype=np.float64)
            nsnps_blk = np.empty((R + 1, K), dtype=np.float64)
            nsnps_blk[:R] = counts_full[None, :] - counts_blk
            nsnps_blk[R] = counts_full

            self.nsnps_blk = nsnps_blk

            if use_cache:
                self._ldsum_all_cache = ld_sum_all
            return ld_sum_all

        # ============================================================
        # chr-mode delete-d / LOCO
        # ============================================================
        U = int(getattr(self, "jackknife_units", 0))
        if U <= 0:
            raise RuntimeError("chr-mode requires jackknife_units > 0.")

        if not hasattr(self, "_unit_starts") or not hasattr(self, "_unit_ends"):
            self._refresh_unit_bounds()

        # exact LOCO delete-1 still uses blk_idx elsewhere
        if int(getattr(self, "jackknife_delete", 1)) == 1 and R == U:
            if (
                not hasattr(self, "_blk_starts")
                or not hasattr(self, "_blk_ends")
                or not hasattr(self, "blk_idx")
            ):
                self._refresh_block_bounds()

        D = np.asarray(self._rep_del_mat, dtype=np.float64, order="C")  # (R, U)
        if D.shape != (R, U):
            raise RuntimeError(
                f"Delete matrix shape mismatch: expected ({R},{U}), got {D.shape}."
            )

        starts_u = np.asarray(self._unit_starts, dtype=np.int64)
        ends_u = np.asarray(self._unit_ends, dtype=np.int64)
        if starts_u.size != U or ends_u.size != U:
            raise RuntimeError("Unit bounds length mismatch with jackknife_units.")

        unit_ld_flat = np.zeros((U, K * K), dtype=np.float64)
        unit_Ak = np.zeros((U, K), dtype=np.float64)

        full_ld = np.zeros((K, K), dtype=np.float64)
        Ak_full = np.zeros((K,), dtype=np.float64)

        for u in range(U):
            s = int(starts_u[u])
            e = int(ends_u[u])
            if e <= s:
                continue

            Au = A[s:e, :]
            Lu = L[s:e, :]
            ld_u = Au.T @ Lu
            ak_u = Au.sum(axis=0, dtype=np.float64)

            unit_ld_flat[u, :] = ld_u.reshape(-1)
            unit_Ak[u, :] = ak_u
            full_ld += ld_u
            Ak_full += ak_u

        del_ld_flat = D @ unit_ld_flat  # (R, K*K)
        del_Ak = D @ unit_Ak            # (R, K)

        rep_Ak = Ak_full[None, :] - del_Ak  # kept bin masses
        ld_sum_rep = full_ld.reshape(1, -1) - del_ld_flat
        ld_sum_rep = ld_sum_rep.reshape(R, K, K)

        # Missing deleted-source correction:
        # raw row-delete uses A_keep^T L_full; for chr delete-d we want A_keep^T L_keep.
        delta = getattr(self, "delta", None) if getattr(self, "adjust_delta", False) else None
        if delta is not None:
            corr = _pair_correction_from_deleted_mass(rep_Ak, del_Ak, delta)
            ld_sum_rep = ld_sum_rep - corr

        ld_sum_all = np.empty((R + 1, K, K), dtype=np.float64)
        ld_sum_all[:R] = ld_sum_rep
        ld_sum_all[R] = full_ld

        nsnps_rep = np.empty((R + 1, K), dtype=np.float64)
        nsnps_rep[:R] = rep_Ak
        nsnps_rep[R] = Ak_full
        self.nsnps_blk = nsnps_rep

        # keep these current after filtering
        self._unit_sizes = (ends_u - starts_u).astype(np.int64, copy=False)

        if use_cache:
            self._ldsum_all_cache = ld_sum_all

        return ld_sum_all