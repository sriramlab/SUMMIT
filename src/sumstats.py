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
        self.zscores_bin = []
        self.zscores_blk = []
        self.rhs = None
        self.nsamp = 0
        self.nsnps = 0
        self.nsnps_blk = None
        self.nsnps_bin = None
        self.chisq_threshold = chisq_threshold
        self.matched_snps = None
        self.name = None
        self.removesnps = []

        # ---- caches for fast matching (NEW) ----
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
        """(NEW) Cache annotation SNP order + annotation matrix for fast rematching."""
        self.annot_df = annot_df
        self._ann_cols = [c for c in annot_df.columns if c != 'SNP']
        # ensure string SNP ids once
        self._annot_snps = annot_df['SNP'].astype(str).to_numpy()
        self._annot_ann = annot_df[self._ann_cols].to_numpy()
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

        # drop non-finite N/Z rows (defensive)
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
        Match SNPs between sumstats and annot_df.
        IMPORTANT: preserve annot_df order (blocks match Trace ordering).
        """
        if self.annot_df is None:
            raise RuntimeError("Sumstats.annot_df is None; cannot match SNPs.")
        if self._annot_snps is None or (self._annot_ann is None):
            self._set_annot_df(self.annot_df)

        # chi^2 filter (unchanged logic, just applied once)
        self._apply_chisq_filter_once()

        matched_zscores = []
        nsnps_blk = np.zeros((self.nblks, self.nbins), dtype=np.float64)

        # If duplicates exist, fall back to merge to preserve legacy semantics
        if self._sum_has_dups or self._annot_has_dups:
            # ---- legacy merge path (exactly your old behavior) ----
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
            all_ann = df[ann_cols].to_numpy()

        else:
            # ---- fast indexer path (NEW) ----
            annot_snps = self._annot_snps
            indexer = self._sum_index.get_indexer(annot_snps)  # position in sumstats arrays, -1 if missing
            keep_mask = indexer >= 0

            # record missing for Trace filtering
            missing_n = int((~keep_mask).sum())
            if missing_n:
                self.removesnps.extend(annot_snps[~keep_mask].tolist())

            # aligned in annot_df order
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

        # ---- blocks & bins (unchanged) ----
        total = len(all_z)
        blk_size = max(total // self.nblks, 1)

        for i in range(self.nblks):
            start = blk_size * i
            end = blk_size * (i + 1) if (i < self.nblks - 1) else total

            blk_zscores = all_z[start:end]
            blk_annot = all_ann[start:end]

            partition, nsnps_partition = utils._partition_bin_overlapping(
                blk_zscores, blk_annot, self.nbins
            )
            matched_zscores.append(partition)
            nsnps_blk[i] = nsnps_partition

        self.zscores_bin, self.nsnps_bin = utils._partition_bin_overlapping(all_z, all_ann, self.nbins)

        for b in range(self.nbins):
            if self.nsnps_bin[b] == 0:
                self.log._log(
                    f"!!! Bin {b} contains zero SNPs after matching. "
                    f"Please check your annotation and SNP lists. !!!"
                )
                sys.exit(1)

        self.nsnps_blk = nsnps_blk
        self.zscores_blk = matched_zscores

    def _calc_rhs_h2(self):
        self.rhs = np.full((self.nblks + 1, self.nbins + 1), self.nsamp - 1)

        for i in range(self.nbins):
            total_zTz = float(np.dot(self.zscores_bin[i], self.zscores_bin[i]))

            # full row
            if self.nsnps_bin[i] <= 0:
                raise RuntimeError(f"Bin {i} has zero SNPs (nsnps_bin=0) after matching.")
            self.rhs[self.nblks, i] = total_zTz * self.nsamp / float(self.nsnps_bin[i])

            # LOO rows
            for j in range(self.nblks):
                blk_zTz = float(np.dot(self.zscores_blk[j][i], self.zscores_blk[j][i]))
                denom = float(self.nsnps_bin[i] - self.nsnps_blk[j][i])

                if denom <= 0:
                    # LOO set is empty for this bin; mark as NaN (jackknife will omit if configured)
                    self.rhs[j, i] = np.nan
                    if j == 0:
                        self.log._log(
                            f"[WARNING] Bin {i} becomes empty in some LOO replicates "
                            f"(denom<=0). Consider fewer blocks or coarser bins."
                        )
                else:
                    self.rhs[j, i] = (total_zTz - blk_zTz) * self.nsamp / denom

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

        self._match_snps(printlog=False)
        self._calc_rhs_h2()
        return