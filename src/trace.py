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
import sys

class Trace:
    def __init__(self, bimpath=None, sumpath=None, savepath=None, log=None,
                 ldscores=None, nblks=100, annot=None, verbose=False):
        self.log = log
        self.sumpath = sumpath
        self.savepath = savepath
        self.ldscorespath = ldscores
        self.ldscores = None
        self.sums = []
        self.nblks = nblks  # nblks specified only if using ld proj; otherwise overwritten by trace summaries.
        self.ntrace = 0
        self.K = []
        self.snplist = []              # <-- initialize
        self.nsamp = []                # number of samples used for trace summaries (can vary)
        self.nsnps = 0                 # total SNPs
        self.nsnps_blk = None          # (B+1, K): LOO bin counts
        self.nsnps_bin = None          # (K,)
        self.verbose = verbose
        self.nbins = None
        self.effective_K = None

        # Optional: read BIM for SNP names
        if (bimpath is None) or (bimpath == ""):
            if (self.ldscorespath is None) and (self.sumpath is not None):
                self.log._log("!!! SNP list (.bim) is recommended when using trace summaries (.tr) !!!")
        elif bimpath.endswith(".bim"):
            with open(bimpath, 'r') as fd:
                for line in fd:
                    self.snplist.append(line.split()[1])
        else:
            self.log._log(f'!!! {bimpath} is not a .bim file! !!!')

        # Read trace summaries or LD-scores
        if (sumpath is not None):
            self._read_all_trace()
            if (savepath is not None):
                self._save_trace()
        elif (ldscores is not None):
            self._read_ldscores()

        # Read annotations (thin or full .annot)
        self._read_annot(annot)

        # Precompute block starts/ends if we already know nsnps/nblks
        if (self.nsnps > 0) and (self.nblks > 0) and not hasattr(self, "blk_size"):
            self.blk_size = max(self.nsnps // self.nblks, 1)
        self._refresh_block_bounds()

    # ---------------------------- I/O & setup ---------------------------- #

    def _refresh_block_bounds(self):
        """Compute and cache per-block slice bounds (after any filtering)."""
        if (self.nsnps <= 0) or (self.nblks <= 0):
            return
        B = self.nblks
        self.blk_size = max(getattr(self, "blk_size", self.nsnps // B), 1)
        starts = self.blk_size * np.arange(B, dtype=np.int64)
        ends = starts + self.blk_size
        if B > 0:
            ends[-1] = self.nsnps  # last block reaches the end
        self._blk_starts = starts
        self._blk_ends = ends

    def _read_annot(self, annot_path):
        # single-bin fallback
        if (annot_path is None):
            self.annot_header = np.array(['L2'])
            annot = np.ones((self.nsnps, 1), dtype=np.float32)
            annot_df = pd.DataFrame(annot, index=self.snplist, columns=self.annot_header.tolist())
            annot_df.reset_index(inplace=True)
            annot_df.rename(columns={'index': 'SNP'}, inplace=True)
            self.annot_df = annot_df
            self.annot = annot_df[self.annot_header].values
            self.nbins = 1
            self.log._log("Running with single component annotation...")
            return

        try:
            # full .annot or .annot.gz
            df = pd.read_csv(annot_path, sep=r'\s+', compression='infer')
            if 'SNP' not in df.columns:
                raise ValueError("!!! Input annotation file is not in correct format !!!")
            annot_cols = df.columns.tolist()[3:]  # after CHR,SNP,BP
            self.annot_header = np.array(annot_cols)
            annot_df = df[['SNP'] + annot_cols].copy()

            if self.snplist:
                overlap = [snp for snp in self.snplist if snp in annot_df['SNP'].values]
                missing = set(self.snplist) - set(overlap)
                if missing:
                    self.log._log(f"Dropping {len(missing)} SNPs from annotation as they are missing LD information.")
                self.annot_df = (annot_df.set_index('SNP').loc[overlap].reset_index())
                self.annot = self.annot_df[annot_cols].values
                self.nsnps = self.annot.shape[0]
                self.snplist = overlap
            else:
                # no BIM provided; keep all rows
                self.annot_df = annot_df.copy()
                self.annot = annot_df[annot_cols].values
                self.nsnps = self.annot.shape[0]
                self.snplist = annot_df['SNP'].tolist()

            self.nbins = len(annot_cols)
            self.log._log("Read full annotation of shape " + str(self.annot.shape))

            # prune LD-scores if present
            if getattr(self, 'ldscores', None) is not None:
                ld_df = (self.ldscores_df.set_index('SNP').loc[self.snplist].reset_index())
                self.ldscores_df = ld_df
                self.ldscores = ld_df.iloc[:, 3:].to_numpy()
                self.log._log(f"Pruned LD-score to {self.nsnps} SNPs that match the annotation file.")

        except ValueError:
            # thin annotation (no header)
            if (not self.snplist):
                raise ValueError("!!! Thin annotation requires a BIM/snplist when using trace-summaries !!!")
            self.annot_header, self.annot = utils._read_with_optional_header(annot_path)
            if self.annot.ndim == 1:
                self.annot = self.annot.reshape(-1, 1)
            else:
                self.annot = self.annot.reshape(-1, self.annot.shape[-1])
            if self.annot_header is None:
                self.annot_header = np.array([f'bin_{i}' for i in range(self.annot.shape[1])])
            self.annot_header = list(self.annot_header)
            self.annot_df = pd.DataFrame(self.annot, index=self.snplist, columns=self.annot_header)
            self.annot_df.reset_index(inplace=True)
            self.annot_df.rename(columns={'index': 'SNP'}, inplace=True)
            self.nsnps = self.annot.shape[0]
            self.nbins = self.annot.shape[1]
            self.log._log("Read thin annotation matrix of shape " + str(self.annot.shape))

        if (self.nbins is None) or (self.nsnps is None):
            self.log._log("!!! number of components or SNP count unresolved !!!")
            sys.exit(1)

    def _save_trace(self):
        """Save trace summaries as files (format preserved from your original code)."""
        with open(self.savepath + ".MN", 'w') as fd:
            fd.write("NSAMPLE,NSNPS,NBLKS,NBINS,K\n")
            fd.write(f"{self.nsamp:.0f},{self.nsnps:.0f},{self.nblks:.0f},{self.nbins:.0f},{self.effective_K:.0f}")

        with open(self.savepath + ".tr", 'w') as fd:
            header_str = ','.join(f'LD_SUM_{i:d}' for i in range(self.nbins))
            fd.write(header_str + ",NSNPS_JACKKNIFE\n")
            for j in range(self.nblks + 1):
                for k in range(self.nbins):
                    row_str = ','.join(f'{self.sums[j, k, l]:.3f}' for l in range(self.nbins))
                    row_str += f',{self.nsnps_blk[j, k]:.0f}\n'
                    fd.write(row_str)
        self.log._log(f"Saved trace summary into {self.savepath}(.tr/.MN)")

    def _read_trace(self, filename, idx):
        """
        Read trace summaries (block-wise LD scores).
        """
        # read metadata
        with open(filename + ".MN", 'r') as fd:
            next(fd)
            nsamp, nsnps, nblks, nbins, K = map(int, fd.readline().split(','))
        self.nsamp.append(nsamp)
        self.K.append(K)

        if idx == 0:
            self.nsnps = nsnps
            self.nblks = nblks
            self.nbins = nbins
        elif (nblks != self.nblks):
            self.log._log("!!! Trace summary " + filename + " has incorrect number of jackknife blocks !!!")
            return
        elif (nbins != self.nbins):
            self.log._log("!!! Trace summary " + filename + " has incorrect number of annotation bins !!!")
            return

        sums = np.zeros((self.nblks + 1, self.nbins, self.nbins))
        nsnps_blk = np.zeros((self.nblks + 1, self.nbins))

        # read values
        for cnt, vals in enumerate(utils._read_multiple_lines(filename + ".tr", self.nbins)):
            sums[cnt] = vals[:, :-1]
            nsnps_blk[cnt] = vals[:, -1].transpose()

        if idx == 0:
            self.nsnps_blk = nsnps_blk
        elif not (np.array_equal(self.nsnps_blk, nsnps_blk)):
            self.log._log("!!! Trace summary " + filename + " has different annotations !!!")
            sys.exit(1)

        self.sums.append(sums)
        self.ntrace += 1

        self.log._log(f"Read in trace summaries from {filename} generated with {K} random vectors")
        if self.verbose:
            self.log._log("-- avg. jackknife LDscore sum:\n" + np.array2string(sums[:-1].mean(axis=0), precision=3, separator=', ')
                          + "\n-- number of jackknife blocks:\t" + str(self.nblks)
                          + "\n-- genome-wide LDscore sum:\n" + np.array2string(sums[-1], precision=3, separator=', '))
        return self.sums

    def _read_all_trace(self):
        trace_files = [self.sumpath]
        if path.isdir(self.sumpath):
            prefix = self.sumpath.rstrip('/') + "/"
            trace_files = sorted([prefix + f.rstrip('.tr') for f in listdir(self.sumpath) if f.endswith('.tr')])
        for i, f in enumerate(trace_files):
            self._read_trace(f, i)
        self.log._log("Finished reading " + str(len(trace_files)) + " trace summaries.")
        if (np.std(self.sums, axis=0).sum() == 0.0) and (self.ntrace > 1):
            self.log._log("!!! Duplicate trace summaries are used -- effective number of random vectors remains unchanged. !!!")
            self.effective_K = np.min(self.K)
        else:
            self.effective_K = np.sum(self.K)

        # weighted average by sample size
        self.sums = np.average(self.sums, axis=0, weights=self.nsamp)
        self.nsamp = float(np.mean(self.nsamp))

    def _read_ldscores(self):
        """
        Read the LD-score matrix instead of trace summaries. Works with either the (truncated) LDSC LD scores (.l2.ldscore.gz) or
        the genome-wide LD scores (.gw.ldscore.gz)
        """
        self.ldscores_df = pd.read_csv(self.ldscorespath, compression='gzip', sep=r'\s+', index_col=False)
        self.ldscores = self.ldscores_df.iloc[:, 3:].to_numpy()
        self.snplist = self.ldscores_df['SNP'].to_numpy().tolist()
        self.nsnps = self.ldscores.shape[0]
        self.nbins = self.ldscores.shape[1]
        self.log._log("Loaded the LD score matrix with " + str(self.nsnps) + " SNPs and " + str(self.nbins) + " bins")

    # ---------------------------- public API ---------------------------- #

    def _calc_trace(self, nsample: float):
        self.log._log("Calculating trace...")
        if (self.ldscores is not None):
            return self._calc_trace_from_ldscores(nsample)
        else:
            return self._calc_trace_from_sums(nsample)

    # ---------------------------- vectorized core ---------------------------- #

    def _calc_trace_from_sums(self, N: float):
        """
        Vectorized trace when pre-aggregated trace summaries are provided.
        Replaces triple loops with a batched conversion.
        """
        K = self.nbins
        B = self.nblks
        sums = np.asarray(self.sums, dtype=np.float64)                     # (B+1, K, K)
        if self.nsnps_blk is None:
            raise RuntimeError("nsnps_blk must be available when using trace summaries.")
        M_k = self.nsnps_blk.astype(np.float64)[:, :, None]                # (B+1, K, 1)
        M_l = self.nsnps_blk.astype(np.float64)[:, None, :]                # (B+1, 1, K)

        # Convert (LD sums) -> trace blocks (KxK), batched over jackknife rows
        trace_KK = utils._calc_trace_from_ld_batch(sums, N, M_k, M_l)      # (B+1, K, K)

        # Assemble (K+1)x(K+1): fill noise row/col with N
        out = np.full((B + 1, K + 1, K + 1), float(N), dtype=np.float64)
        out[:, :K, :K] = trace_KK
        return out

    def _calc_trace_from_ldscores(self, N: float):
        """
        Vectorized trace from LD-scores.
        Builds:
          full_ld   = A^T L          -> (K, K)
          blk_ld[j] = A_j^T L_j      -> (B, K, K)   for each SNP-block j
        Then LOO sums are full_ld - blk_ld[j], with the full row appended.
        Finally, turns (ld_sum, M_k, M_l, N) into the (K+1)x(K+1) normal-equation blocks
        via a batched utils function.
        """
        if not hasattr(self, "_blk_starts"):
            self._refresh_block_bounds()

        A = np.asarray(self.annot, dtype=np.float32, order='C')     # (M, K)
        L = np.asarray(self.ldscores, dtype=np.float32, order='C')  # (M, K)
        K = self.nbins
        B = self.nblks

        # Full KxK LD sums once
        full_ld = A.T @ L                                           # (K, K)

        # Per-block KxK LD sums: A_j^T L_j (B gemms)
        blk_ld = np.empty((B, K, K), dtype=np.float32)
        for j in range(B):
            sl = slice(self._blk_starts[j], self._blk_ends[j])
            Aj = A[sl, :]       # (m_j, K)
            Lj = L[sl, :]       # (m_j, K)
            blk_ld[j] = Aj.T @ Lj

        # LOO ld sums for j=0..B-1; full row at j=B
        ld_sum_all = np.empty((B + 1, K, K), dtype=np.float64)
        ld_sum_all[:B] = (full_ld[None, :, :] - blk_ld)
        ld_sum_all[B] = full_ld

        # LOO bin counts for each (j, k) and (j, l)
        if self.nsnps_blk is None:
            # build default counts from annotation if not present (rare when using LD-scores path)
            counts_full = A.sum(axis=0, dtype=np.float64)                        # (K,)
            counts_blk = np.empty((B, K), dtype=np.float64)
            for j in range(B):
                sl = slice(self._blk_starts[j], self._blk_ends[j])
                counts_blk[j] = A[sl, :].sum(axis=0, dtype=np.float64)
            nsnps_blk = np.empty((B + 1, K), dtype=np.float64)
            nsnps_blk[:B] = counts_full[None, :] - counts_blk
            nsnps_blk[B] = counts_full
        else:
            nsnps_blk = self.nsnps_blk.astype(np.float64)

        M_k = nsnps_blk[:, :, None]    # (B+1, K, 1)
        M_l = nsnps_blk[:, None, :]    # (B+1, 1, K)

        # Convert (LD sums) -> trace blocks (KxK), batched over jackknife rows
        trace_KK = utils._calc_trace_from_ld_batch(ld_sum_all, N, M_k, M_l)

        # Assemble (K+1)x(K+1): fill noise row/col with N
        out = np.full((B + 1, K + 1, K + 1), float(N), dtype=np.float64)
        out[:, :K, :K] = trace_KK
        out[:, K, K] = float(N-1)
        return out

    # ------------------------- (optional) rg path ------------------------- #

    def _calc_trace_rg(self, nsample1: float, nsample2: float, noverlap: float | None):
        """
        Vectorized trace for genetic correlation. Uses LD-scores path.
        If noverlap is not None, adds it to each (k,l) entry (constrained path),
        and expands the matrix to (K+1)x(K+1) with the last row/col set to noverlap.
        """
        if self.ldscores is None:
            raise NotImplementedError("rg path currently expects LD-scores input.")

        if not hasattr(self, "_blk_starts"):
            self._refresh_block_bounds()

        A = np.asarray(self.annot, dtype=np.float32, order='C')     # (M, K)
        L = np.asarray(self.ldscores, dtype=np.float32, order='C')  # (M, K)
        K = self.nbins
        B = self.nblks

        # Full and per-block LD sums
        full_ld = A.T @ L
        blk_ld = np.empty((B, K, K), dtype=np.float32)
        for j in range(B):
            sl = slice(self._blk_starts[j], self._blk_ends[j])
            Aj = A[sl, :]
            Lj = L[sl, :]
            blk_ld[j] = Aj.T @ Lj

        ld_sum_all = np.empty((B + 1, K, K), dtype=np.float64)
        ld_sum_all[:B] = (full_ld[None, :, :] - blk_ld)
        ld_sum_all[B] = full_ld

        # LOO bin-counts
        counts_full = A.sum(axis=0, dtype=np.float64)
        counts_blk = np.empty((B, K), dtype=np.float64)
        for j in range(B):
            sl = slice(self._blk_starts[j], self._blk_ends[j])
            counts_blk[j] = A[sl, :].sum(axis=0, dtype=np.float64)

        nsnps_blk = np.empty((B + 1, K), dtype=np.float64)
        nsnps_blk[:B] = counts_full[None, :] - counts_blk
        nsnps_blk[B] = counts_full

        M_k = nsnps_blk[:, :, None]
        M_l = nsnps_blk[:, None, :]

        # Convert (LD sums) -> rg trace blocks (KxK)
        rg_KK = utils._calc_rg_trace_from_ld_batch(ld_sum_all, nsample1, nsample2, M_k, M_l)

        if noverlap is None:
            # Unconstrained: return KxK padded to (K)x(K) (no extra row/col)
            return rg_KK
        else:
            # Constrained: add noverlap to each (k,l)
            rg_KK = rg_KK + float(noverlap)

            # Assemble (K+1)x(K+1) with the last row/col set to noverlap
            out = np.full((B + 1, K + 1, K + 1), float(noverlap), dtype=np.float64)
            out[:, :K, :K] = rg_KK
            return out

    # ---------------------------- filtering ---------------------------- #

    def _filter_snps(self, removesnps):
        """
        Remove the SNPs in the removesnps from trace calculation. Only possible when LD scores are used as input.
        """
        if self.ldscores is None:
            return

        mask = ~self.annot_df['SNP'].isin(removesnps)

        new_snps = self.annot_df.loc[mask, 'SNP'].tolist()
        self.nsnps = len(new_snps)

        self.annot = self.annot_df.loc[mask, list(self.annot_header)].to_numpy()
        self.ldscores = self.ldscores_df.loc[mask, self.ldscores_df.columns[3:]].to_numpy()

        self.blk_size = max(self.nsnps // self.nblks, 1)
        self.nsnps_bin = self.annot.sum(axis=0, dtype=np.float64)

        # Build LOO bin counts: row j is counts after dropping block j; last row is full counts
        self.nsnps_blk = np.full((self.nblks + 1, self.nbins), self.nsnps_bin, dtype=np.float64)
        for j in range(self.nblks):
            start = self.blk_size * j
            end = self.blk_size * (j + 1) if (j < self.nblks - 1) else self.nsnps
            self.nsnps_blk[j] = self.nsnps_bin - self.annot[start:end].sum(axis=0, dtype=np.float64)

        n_removed = int(len(self.annot_df) - mask.sum())
        self.log._log(f"Filtered {n_removed} SNPs from the Trace module. Shape of final annotation used for analysis: {self.annot.shape}")
        if (n_removed / len(self.annot_df) > 0.01):
            self.log._log("[WARNING: Removing too many Trace SNPs will result in under-estimated heritability!]\n"
                          "[We recommend using a better curated reference LD score panel with a more similar SNP set to the summary statistics SNPs.]")

        # refresh cached block bounds for the new nsnps
        self._refresh_block_bounds()