'''
Read the phenotype-specific summary statistics (generally PLINK)
The standard LDSC summary statistics format (.sumstat) works.
Format should be:
    SNPID, NMISS (or OBS_CT), Z
'''
import utils

import numpy as np
import pandas as pd
from os import listdir
import sys

class Sumstats:
    def __init__(self, nblks=100, chisq_threshold=0, log=None, both_side=False, annot_df=None, nbins=1):
        self.log = log
        self.nblks = nblks
        self.nbins = nbins
        self.annot_df = annot_df
        self.snpids = len(annot_df)
        self.annot = None
        self.zscores = None
        self.rhs = None
        self.nsamp = 0
        self.nsnps = 0
        self.nsnps_blk = None
        self.nsnps_bin = None
        self.chisq_threshold = chisq_threshold
        self.matched_snps = None
        self.name = None
        self.removesnps = []

        # ---- caches for fast matching ----
        self._ann_cols = None
        self._annot_snps = None
        self._annot_ann = None
        self._annot_has_dups = False

        self._sum_snps = None
        self._sum_z = None
        self._sum_a1 = None
        self._sum_a2 = None
        self._sum_index = None
        self._sum_has_dups = False

        self._chisq_applied = False

        if annot_df is not None:
            self._set_annot_df(annot_df)

    def _set_annot_df(self, annot_df: pd.DataFrame):
        """Cache annotation SNP order + annotation matrix for fast rematching."""
        self.annot_df = annot_df
        self._ann_cols = [c for c in annot_df.columns if c != 'SNP']

        # ensure string SNP ids once
        self._annot_snps = annot_df['SNP'].astype(str).to_numpy()

        # keep numeric matrix; works for binary/overlapping/continuous
        self._annot_ann = annot_df[self._ann_cols].to_numpy(dtype=np.float64, copy=False)

        # basic sanity (fail fast if annotation has NaNs / inf)
        if not np.isfinite(self._annot_ann).all():
            bad = np.flatnonzero(~np.isfinite(self._annot_ann).any(axis=1))[:10]
            raise ValueError(
                "Annotation contains non-finite values (NaN/inf). "
                f"First bad row indices (in annot_df order): {bad.tolist()}"
            )

        self._annot_has_dups = pd.Index(self._annot_snps).has_duplicates


    def _read_sumstats(self, path, name):
        self.name = name
        sumdf = pd.read_csv(path, sep=r'\s+', compression='infer')

        ncol = utils._parse_column_name(sumdf, ['N', 'n'], 3)
        zcol = utils._parse_column_name(sumdf, ['Z', 'z'], 3)
        idcol = utils._parse_column_name(sumdf, ['ID', 'id', 'snp', 'SNP'], 0)
        a1col = utils._parse_column_name(sumdf, ['A1', 'ALT'], 1)
        a2col = utils._parse_column_name(sumdf, ['A2', 'REF'], 1)

        drop_mask = sumdf[ncol].isna() | sumdf[zcol].isna()
        self.removesnps = sumdf.loc[drop_mask, idcol].dropna().astype(str).tolist()
        self.log._log(f"Dropping {len(self.removesnps)} SNPs with NA values [{self.name}].")

        sumdf = sumdf.loc[~drop_mask].copy()
        sumdf = sumdf.rename(columns={idcol: 'SNP', zcol: 'Z', ncol: 'N', a1col: 'A1', a2col: 'A2'})

        # ensure types
        sumdf['SNP'] = sumdf['SNP'].astype(str)
        sumdf['A1'] = sumdf['A1'].astype(str).str.upper()
        sumdf['A2'] = sumdf['A2'].astype(str).str.upper()
        sumdf['N'] = pd.to_numeric(sumdf['N'], errors='coerce')
        sumdf['Z'] = pd.to_numeric(sumdf['Z'], errors='coerce')

        # drop non-finite N/Z rows
        bad = (~np.isfinite(sumdf['N'].to_numpy())) | (~np.isfinite(sumdf['Z'].to_numpy()))
        if bad.any():
            bad_snps = sumdf.loc[bad, 'SNP'].tolist()
            self.removesnps += bad_snps
            sumdf = sumdf.loc[~bad].copy()
            self.log._log(f"Dropping {len(bad_snps)} SNPs with non-finite N/Z values.")

        # scale Z by sqrt(N / Nmax)
        nmax = float(sumdf['N'].max())
        sumdf['Z'] = sumdf['Z'] * np.sqrt(sumdf['N'] / nmax)

        # drop non-finite Z after scaling
        badz = ~np.isfinite(sumdf['Z'].to_numpy())
        if badz.any():
            bad_snps = sumdf.loc[badz, 'SNP'].tolist()
            self.removesnps += bad_snps
            sumdf = sumdf.loc[~badz].copy()
            self.log._log(f"Dropping {len(bad_snps)} SNPs with non-finite Z after scaling.")

        # set nsamp
        self.nsamp = float(nmax)

        # deduplicate SNPs (recommended for correctness + stable indexing)
        if sumdf['SNP'].duplicated().any():
            # keep max-N row per SNP; ties resolved by earliest in file
            sumdf['_row'] = np.arange(sumdf.shape[0], dtype=np.int64)
            sumdf = (
                sumdf.sort_values(['SNP', 'N', '_row'], ascending=[True, False, True])
                    .drop_duplicates(subset='SNP', keep='first')
                    .sort_values('_row')
                    .drop(columns=['_row'])
            )
            self.log._log(f"Detected duplicate SNP IDs; kept 1 row per SNP (remaining={sumdf.shape[0]}).")

        self.sumdf = sumdf[['SNP', 'Z', 'A1', 'A2']].copy()

        # cached arrays + index
        self._sum_snps = self.sumdf['SNP'].to_numpy(dtype=str, copy=False)
        self._sum_z = self.sumdf['Z'].to_numpy(dtype=np.float64, copy=False)
        self._sum_a1 = self.sumdf['A1'].to_numpy(dtype=str, copy=False)
        self._sum_a2 = self.sumdf['A2'].to_numpy(dtype=str, copy=False)

        self._sum_index = pd.Index(self._sum_snps)
        self._sum_has_dups = self._sum_index.has_duplicates  # should now be False
        self._chisq_applied = False


    def _apply_chisq_filter_once(self):
        """Apply chi^2 filter at most once per Sumstats object, updating BOTH caches and self.sumdf."""
        if self._chisq_applied:
            return
        self._chisq_applied = True

        if self.chisq_threshold is None:
            return

        thr = float(self.chisq_threshold)
        self.log._log(f"Filtering SNPs with chi-sq greater than {thr}")

        chisq = self._sum_z ** 2
        keep = (chisq <= thr) & np.isfinite(chisq)

        if np.all(keep):
            return

        chisq_snps = self._sum_snps[~keep].tolist()
        self.removesnps += chisq_snps

        # filter cached arrays
        self._sum_snps = self._sum_snps[keep]
        self._sum_z = self._sum_z[keep]
        self._sum_a1 = self._sum_a1[keep]
        self._sum_a2 = self._sum_a2[keep]

        # filter dataframe (same row order as caches)
        self.sumdf = self.sumdf.loc[keep].reset_index(drop=True)

        # rebuild index
        self._sum_index = pd.Index(self._sum_snps)
        self._sum_has_dups = self._sum_index.has_duplicates

        self.log._log(
            f"Removed {len(chisq_snps)} SNPs with chi-sq above the threshold "
            f"{thr} ({self._sum_snps.size} SNPs remaining)"
        )


    def _match_snps(self, printlog=True):
        """
        Match SNPs between sumstats and annot_df while preserving annot_df order (blocks match Trace ordering).

        Supports:
        - non-overlapping binary annotations
        - overlapping binary annotations
        - continuous (nonnegative) annotations

        For continuous/overlapping, we compute weighted sufficient statistics:
            M_k   = sum_j a_{j,k}
            S_k   = sum_j a_{j,k} * z_j^2
        and their in-block counterparts for jackknife.
        """
        if self.annot_df is None:
            raise RuntimeError("Sumstats.annot_df is None; cannot match SNPs.")
        if self._annot_snps is None or (self._annot_ann is None):
            self._set_annot_df(self.annot_df)

        # chi^2 filter
        self._apply_chisq_filter_once()

        # ----------------------------
        # SNP matching
        # ----------------------------
        if self._sum_has_dups or self._annot_has_dups:
            df = self.annot_df.merge(self.sumdf, how='inner', on='SNP', sort=False)

            matched_set = pd.Index(df['SNP'].astype(str))
            missing_mask = ~self.annot_df['SNP'].astype(str).isin(matched_set)
            if missing_mask.any():
                self.removesnps.extend(self.annot_df.loc[missing_mask, 'SNP'].astype(str).tolist())

            self.matched_snps = df['SNP'].astype(str).to_numpy()
            if printlog:
                self.log._log(
                    f"Matched {len(df)} SNPs in phenotype {self.name} out of {len(self.annot_df)} "
                    f"annotated SNPs ({int(missing_mask.sum())} missing or filtered)"
                )

            all_z = df['Z'].to_numpy(dtype=np.float64, copy=False)
            self.zscores = all_z
            self.a1 = df['A1'].astype(str).str.upper().to_numpy()
            self.a2 = df['A2'].astype(str).str.upper().to_numpy()

            ann_cols = [c for c in self.annot_df.columns if c != 'SNP']
            all_ann = df[ann_cols].to_numpy(dtype=np.float64, copy=False)

        else:
            # fast indexer path (preserve annot_df order)
            annot_snps = self._annot_snps
            indexer = self._sum_index.get_indexer(annot_snps)  # position in sumstats arrays, -1 if missing
            keep_mask = indexer >= 0

            missing_n = int((~keep_mask).sum())
            if missing_n:
                self.removesnps.extend(annot_snps[~keep_mask].tolist())

            pos = indexer[keep_mask]  # positions into sumstats arrays
            self.matched_snps = annot_snps[keep_mask]
            if printlog:
                self.log._log(
                    f"Matched {pos.size} SNPs in phenotype {self.name} out of {annot_snps.size} "
                    f"annotated SNPs ({missing_n} missing or filtered)"
                )

            all_z = self._sum_z[pos]
            self.zscores = all_z

            self.a1 = self._sum_a1[pos]
            self.a2 = self._sum_a2[pos]

            all_ann = self._annot_ann[keep_mask, :]

        # ----------------------------
        # Weighted sufficient statistics for RHS
        # ----------------------------
        all_z = np.asarray(all_z, dtype=np.float64)
        if not np.isfinite(all_z).all():
            raise RuntimeError("Non-finite z-scores encountered after matching; drop upstream.")

        A = np.asarray(all_ann, dtype=np.float64, order="C")
        if A.ndim != 2 or A.shape[1] != self.nbins:
            raise RuntimeError(f"Annotation matrix shape mismatch: got {A.shape}, expected (*,{self.nbins}).")
        if not np.isfinite(A).all():
            bad = np.flatnonzero(~np.isfinite(A).any(axis=1))[:10]
            raise RuntimeError(f"Non-finite annotation values encountered after matching. First bad rows: {bad.tolist()}")

        M = int(all_z.size)
        self.nsnps = M  # matched SNP count

        # block partition (must match Trace convention: blk_size = M//B, last block takes remainder)
        blk_size = max(M // self.nblks, 1)
        blk_idx = (np.arange(M, dtype=np.int64) // blk_size)
        blk_idx[blk_idx >= self.nblks] = self.nblks - 1
        self.blk_idx = blk_idx  # optional, but handy for debugging

        z2 = all_z * all_z  # (M,)

        # Full sums: M_k = sum a_{jk}; S_k = sum a_{jk} z_j^2
        Ak_full = A.sum(axis=0, dtype=np.float64)          # (K,)
        Az2_full = (A.T @ z2).astype(np.float64, copy=False)  # (K,)

        # In-block contributions for jackknife
        B = self.nblks
        K = self.nbins
        Ak_blk = np.zeros((B, K), dtype=np.float64)
        Az2_blk = np.zeros((B, K), dtype=np.float64)

        tmp = np.empty(M, dtype=np.float64)
        for k in range(K):
            col = A[:, k]
            # M_k^{(b)} = sum_{j in block b} a_{jk}
            tmp[:] = col
            Ak_blk[:, k] = np.bincount(blk_idx, weights=tmp, minlength=B).astype(np.float64, copy=False)
            # S_k^{(b)} = sum_{j in block b} a_{jk} z_j^2
            tmp[:] = col * z2
            Az2_blk[:, k] = np.bincount(blk_idx, weights=tmp, minlength=B).astype(np.float64, copy=False)

        # expose the "bin sizes" using the same names as before (now = sum of weights)
        self.nsnps_bin = Ak_full
        self.nsnps_blk = Ak_blk

        # store sufficient stats for RHS computation
        self._Ak_full = Ak_full
        self._Az2_full = Az2_full
        self._Ak_blk = Ak_blk
        self._Az2_blk = Az2_blk

        # sanity: bins must have positive total weight (for binary, this is count>0)
        bad_bins = np.flatnonzero(~np.isfinite(Ak_full) | (Ak_full <= 0.0))
        if bad_bins.size:
            self.log._log(
                "!!! One or more annotation bins have non-positive total weight after matching. "
                f"Bad bins: {bad_bins.tolist()} (Ak_full={Ak_full[bad_bins].tolist()}) !!!"
            )
            sys.exit(1)


    def _calc_rhs_h2(self):
        """
        Build RHS for univariate normal equations.

        For each bin k:
        M_k     = sum_j a_{j,k}
        S_k     = sum_j a_{j,k} z_j^2
        M_k^b   = sum_{j in block b} a_{j,k}
        S_k^b   = sum_{j in block b} a_{j,k} z_j^2

        Full:
        rhs[B,k] = (S_k * N) / M_k
        LOO:
        rhs[b,k] = ((S_k - S_k^b) * N) / (M_k - M_k^b)

        Noise term:
        rhs[:,K] = N - 1
        """
        if not hasattr(self, "_Ak_full") or self._Ak_full is None:
            raise RuntimeError("Missing cached annotation-weight sums. Did you call _match_snps()?")

        B = self.nblks
        K = self.nbins
        N = float(self.nsamp)

        Ak_full = np.asarray(self._Ak_full, dtype=np.float64)       # (K,)
        Az2_full = np.asarray(self._Az2_full, dtype=np.float64)     # (K,)
        Ak_blk = np.asarray(self._Ak_blk, dtype=np.float64)         # (B,K) in-block
        Az2_blk = np.asarray(self._Az2_blk, dtype=np.float64)       # (B,K) in-block

        self.rhs = np.full((B + 1, K + 1), N - 1.0, dtype=np.float64)

        warned = np.zeros(K, dtype=bool)

        for k in range(K):
            denom_full = float(Ak_full[k])
            if not (np.isfinite(denom_full) and denom_full > 0.0):
                raise RuntimeError(f"Bin {k} has non-positive total weight (Ak_full={denom_full}).")

            # full row
            self.rhs[B, k] = float(Az2_full[k]) * N / denom_full

            # LOO rows
            for b in range(B):
                denom = float(Ak_full[k] - Ak_blk[b, k])
                if not (np.isfinite(denom) and denom > 0.0):
                    self.rhs[b, k] = np.nan
                    if not warned[k]:
                        warned[k] = True
                        self.log._log(
                            f"[WARNING] Bin {k} has non-positive total weight in some LOO replicates "
                            f"(denom<=0). Consider fewer blocks or coarser/less sparse annotations."
                        )
                    continue

                num = float(Az2_full[k] - Az2_blk[b, k])
                self.rhs[b, k] = num * N / denom

        self.log._log(f"Calculated the RHS for phenotype [{self.name}]")



    def _process(self, path, name):
        self._read_sumstats(path, name)
        self._match_snps()
        self._calc_rhs_h2()
        return self.removesnps
    
    def read_only(self, path: str, name: str):
        """
        Read + cache sumstats once (including chi^2 filtering).
        Does NOT match to annotation or compute RHS.
        """
        self._read_sumstats(path, name)
        self._apply_chisq_filter_once()
        return

    def rematch_and_recompute(self, annot_df: pd.DataFrame):
        """
        Rematch to a (possibly new) annot_df using cached sumstats arrays,
        then recompute RHS. No file I/O.
        """
        self._set_annot_df(annot_df)

        self.matched_snps = None
        self.zscores = None
        self.zscores_bin = []
        self.zscores_blk = []
        self.nsnps_blk = None
        self.nsnps_bin = None

        # reset weighted caches
        self._Ak_full = None
        self._Az2_full = None
        self._Ak_blk = None
        self._Az2_blk = None

        self._match_snps(printlog=False)
        self._calc_rhs_h2()
        return
