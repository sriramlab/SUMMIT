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

_CHI2_MEDIAN_1DF = 0.454936423119572

class Sumstats:
    def __init__(self, nblks=100, chisq_threshold=0, chisq_action='drop', log=None, both_side=False, annot_df=None, nbins=1):
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
        # NOTE: chisq_threshold can be:
        #   - 'auto' (string): use default cap max(80, 0.001 * Nmax)
        #   - None: no filtering
        #   - number: legacy behavior
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
        self._chisq_filter_removed = 0
        self._chisq_action = chisq_action

        # --- chi^2 diagnostics caches (per phenotype) ---
        # "read": after parsing + scaling Z, before matching to annotations
        # "used": after matching to annotation SNPs (and after any active chi^2 filter)
        self._chisq_diag_read = None
        self._chisq_diag_used = None
        self._chisq_top_read = None
        self._chisq_top_used = None

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

    def _suggested_chisq_max(self, nmax: float) -> float:
        """
        A commonly used cap for extreme chi^2 outliers:
            chi2_max = max(80, 0.001 * Nmax)
        This function is report-only unless you explicitly set chisq_threshold.
        """
        if not np.isfinite(nmax) or nmax <= 0:
            return 80.0
        return float(max(80.0, 0.001 * nmax))

    def _resolve_chisq_threshold(self):
        """
        Resolve chisq_threshold into an effective numeric threshold.

        Returns:
            (thr: float|None, mode: str)
              mode in {'none','auto','manual'}.

        Semantics:
          - chisq_threshold is None -> (None, 'none')  [no filtering]
          - chisq_threshold == 'auto' (case-insensitive) -> (max(80, 0.001*Nmax), 'auto')
          - otherwise -> float(value) if possible -> ('manual')
        """
        raw = getattr(self, "chisq_threshold", None)
        if raw is None:
            return None, "none"

        if isinstance(raw, str):
            s = raw.strip().lower()
            if s == "auto":
                nmax = float(getattr(self, "nsamp", np.nan))
                return float(self._suggested_chisq_max(nmax)), "auto"
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


    def _chisq_summary(self, chisq: np.ndarray) -> dict:
        chisq = np.asarray(chisq, dtype=np.float64)
        finite = np.isfinite(chisq)
        x = chisq[finite]
        out = {
            "M": int(chisq.size),
            "M_finite": int(x.size),
            "M_nonfinite": int(chisq.size - x.size),
        }
        if x.size == 0:
            return out

        out["mean"] = float(np.mean(x))
        out["median"] = float(np.median(x))
        out["max"] = float(np.max(x))
        out["lambda_gc"] = float(out["median"] / _CHI2_MEDIAN_1DF)

        for q in [90, 95, 99, 99.9, 99.99, 99.999]:
            out[f"p{q}"] = float(np.percentile(x, q))

        return out


    def _top_chisq_rows(self, snps, a1, a2, chisq, topk: int = 10):
        """Return [(SNP, A1, A2, chi2), ...] for the largest chi2 values."""
        if topk <= 0:
            return []
        chisq = np.asarray(chisq, dtype=np.float64)
        finite = np.isfinite(chisq)
        if not finite.any():
            return []

        idx = np.where(finite)[0]
        x = chisq[finite]
        k = min(topk, x.size)

        part = np.argpartition(-x, kth=k - 1)[:k]
        best = part[np.argsort(-x[part])]

        rows = []
        for t in best:
            j = int(idx[t])
            rows.append((str(snps[j]), str(a1[j]), str(a2[j]), float(chisq[j])))
        return rows


    def _compute_chisq_diag(self, z, snps, a1, a2, nmax: float, topk: int = 10) -> tuple[dict, list]:
        z = np.asarray(z, dtype=np.float64)
        chisq = z * z
        summ = self._chisq_summary(chisq)

        thr = self._suggested_chisq_max(float(nmax))
        finite = np.isfinite(chisq)
        n_hi = int(np.sum(chisq[finite] > thr))
        frac_hi = float(n_hi / max(int(np.sum(finite)), 1))

        summ["suggested_chisq_max"] = float(thr)
        summ["n_gt_suggested"] = int(n_hi)
        summ["frac_gt_suggested"] = float(frac_hi)

        top = self._top_chisq_rows(snps, a1, a2, chisq, topk=topk)
        return summ, top


    def log_chisq_diagnostics(
        self,
        topk: int = 10,
        warn_min_count: int = 10,
        warn_min_frac: float = 1e-5,
        verbose: bool = False,
        include_read_when_verbose: bool = True,
    ):
        """
        Report chi^2 diagnostics for the most recently processed phenotype.

        Default behavior (non-verbose):
        - report ONLY the 'used' stage once (matched/filtered set)
        - print a concise summary line
        - print detailed tail + top outliers only if warning triggers

        Verbose behavior:
        - print active chi^2 filter status once
        - print 'read' first (optional)
        - print 'used' second
        - expanded details for both stages
        """
        name = getattr(self, "name", "UNKNOWN")
        nmax = float(getattr(self, "nsamp", np.nan))

        thr_eff, thr_mode = self._resolve_chisq_threshold()
        removed_user = int(getattr(self, "_chisq_filter_removed", 0))

        thr_user = float(thr_eff) if thr_eff is not None else 0.0

        def _should_expand(summ: dict) -> bool:
            if not summ:
                return False
            n_hi = int(summ.get("n_gt_suggested", 0))
            frac_hi = float(summ.get("frac_gt_suggested", 0.0))
            return (n_hi >= warn_min_count) or (frac_hi >= warn_min_frac)

        def _log_stage(stage: str, summ: dict, top_rows: list, expand: bool):
            if not summ:
                return

            # always: one-line summary
            if "mean" in summ:
                self.log._log(
                    f"[chisq] [{name}] {stage}  "
                    f"M={summ.get('M', 0)} (finite={summ.get('M_finite', 0)})  "
                    f"lambda_gc={summ['lambda_gc']:.4f}  mean={summ['mean']:.4f}  "
                    f"p99.9={summ.get('p99.9', float('nan')):.2f}  max={summ['max']:.2f}"
                )
            else:
                self.log._log(
                    f"[chisq] [{name}] {stage}  "
                    f"M={summ.get('M', 0)} (finite={summ.get('M_finite', 0)})"
                )

            if not expand or ("mean" not in summ):
                return

            thr = float(summ.get("suggested_chisq_max", np.nan))
            if np.isfinite(thr):
                self.log._log(
                    f"[chisq] [{name}] suggested chi^2 cap = max(80, 0.001*Nmax) with Nmax={nmax:.1f}: {thr:.3f}"
                )
                self.log._log(
                    f"[chisq] [{name}] SNPs with chi^2 > {thr:.3f}: "
                    f"{int(summ.get('n_gt_suggested', 0))} ({float(summ.get('frac_gt_suggested', 0.0)):.3e})"
                )

            self.log._log(
                f"[chisq] [{name}] tail: "
                f"p99={summ.get('p99', float('nan')):.2f} "
                f"p99.9={summ.get('p99.9', float('nan')):.2f} "
                f"p99.99={summ.get('p99.99', float('nan')):.2f} "
                f"p99.999={summ.get('p99.999', float('nan')):.2f}"
            )

            if _should_expand(summ):
                self.log._log(
                    f"[WARNING] [{name}] many extremely large chi^2 SNPs detected. "
                    f"This can violate MoM / variance-component assumptions and destabilize estimates."
                )
                if top_rows:
                    self.log._log(f"[chisq] [{name}] top outliers (SNP A1 A2 chi2):")
                    for r, (snp, aa1, aa2, chi2) in enumerate(top_rows, start=1):
                        self.log._log(f"[chisq] [{name}]  {r:2d}. {snp}\t{aa1}\t{aa2}\t{chi2:.3f}")

        # ---- Print active user filter status ONCE at the top (verbose only) ----
        if verbose:
            if (thr_user > 0) and np.isfinite(thr_user):
                mode_tag = " (auto)" if thr_mode == "auto" else ""
                self.log._log(
                    f"[chisq] [{name}] active chi^2 filter: threshold={thr_user:.3f}{mode_tag}; removed={removed_user} SNPs."
                )
            else:
                self.log._log(f"[chisq] [{name}] no active chi^2 filter.")

        # ---- Ordering: verbose prints read first, then used ----
        if verbose and include_read_when_verbose:
            _log_stage("read", self._chisq_diag_read, self._chisq_top_read, expand=True)

        # Always report "used" once (concise unless verbose or warning-triggered)
        used_summ = self._chisq_diag_used
        used_top = self._chisq_top_used
        expand_used = bool(verbose) or _should_expand(used_summ)
        _log_stage("used", used_summ, used_top, expand=expand_used)




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

        # deduplicate SNPs
        if sumdf['SNP'].duplicated().any():
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
        self._sum_has_dups = self._sum_index.has_duplicates
        self._chisq_applied = False
        self._chisq_filter_removed = 0

        # diagnostics on the read/scaled set (before matching)
        self._chisq_diag_read, self._chisq_top_read = self._compute_chisq_diag(
            z=self._sum_z, snps=self._sum_snps, a1=self._sum_a1, a2=self._sum_a2, nmax=self.nsamp, topk=10
        )
        self._chisq_diag_used = None
        self._chisq_top_used = None


    def _apply_chisq_filter_once(self):
        """
        Apply chi^2 handling at most once per Sumstats object.

        Modes (set via self.chisq_action; default='drop'):
        - 'drop': remove SNPs with chi^2 > threshold from sumstats caches (current behavior)
        - 'clip': do NOT drop here (clipping happens later on the matched set)
        - 'warn'/'none': do nothing here

        Convention: chisq_threshold <= 0 or non-finite => disabled.
        Special:
          - chisq_threshold == 'auto' => threshold = max(80, 0.001*Nmax)
          - chisq_threshold is None => no filtering
        """
        if self._chisq_applied:
            return
        self._chisq_applied = True
        self._chisq_filter_removed = 0

        action = str(getattr(self, "_chisq_action", "drop")).strip().lower()
        if action not in ("drop", "clip", "warn", "none"):
            # fail-safe: preserve legacy behavior
            self.log._log(f"[chisq] Unrecognized chisq_action='{action}', defaulting to 'drop'.")
            action = "drop"
        self._chisq_action_used = action

        thr, _thr_mode = self._resolve_chisq_threshold()
        if thr is None:
            return

        thr = float(thr)

        # Convention: <=0 means "disabled"
        if (not np.isfinite(thr)) or (thr <= 0.0):
            return

        # Only 'drop' modifies caches here. 'clip' is handled post-matching.
        if action != "drop":
            return

        self.log._log(f"[chisq] Dropping SNPs with chi^2 greater than {thr}")

        chisq = self._sum_z ** 2
        keep = (chisq <= thr) & np.isfinite(chisq)

        if np.all(keep):
            return

        chisq_snps = self._sum_snps[~keep].tolist()
        self.removesnps += chisq_snps
        self._chisq_filter_removed = int(len(chisq_snps))

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
            f"[chisq] Removed {len(chisq_snps)} SNPs with chi^2 above {thr} "
            f"({self._sum_snps.size} SNPs remaining)"
        )




    def _match_snps(self, printlog=True):
        """
        Match SNPs between sumstats and annot_df while preserving annot_df order.

        Supports binary/overlapping/continuous annotations.

        For overlapping/continuous, we compute weighted sufficient stats:
            M_k   = sum_j a_{j,k}
            S_k   = sum_j a_{j,k} * z_j^2
        and their in-block counterparts.

        Robust chi^2 handling (self.chisq_action; default='drop'):
        - 'drop': SNPs were already removed pre-matching by _apply_chisq_filter_once()
        - 'clip': after matching, cap z_j^2 at threshold (winsorization) when building RHS
        - 'warn'/'none': no modification; only diagnostics
        """
        if self.annot_df is None:
            raise RuntimeError("Sumstats.annot_df is None; cannot match SNPs.")
        if self._annot_snps is None or (self._annot_ann is None):
            self._set_annot_df(self.annot_df)

        # optional pre-match handling (only active for action='drop')
        self._apply_chisq_filter_once()

        action = str(getattr(self, "chisq_action", "drop")).strip().lower()
        if action not in ("drop", "clip", "warn", "none"):
            self.log._log(f"[chisq] Unrecognized chisq_action='{action}', defaulting to 'drop'.")
            action = "drop"

        thr, _thr_mode = self._resolve_chisq_threshold()
        thr_enabled = (thr is not None) and np.isfinite(float(thr)) and (float(thr) > 0.0)
        thr = float(thr) if thr is not None else None

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
            annot_snps = self._annot_snps
            indexer = self._sum_index.get_indexer(annot_snps)
            keep_mask = indexer >= 0

            missing_n = int((~keep_mask).sum())
            if missing_n:
                self.removesnps.extend(annot_snps[~keep_mask].tolist())

            pos = indexer[keep_mask]
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

        # diagnostics on the matched set (raw; before any clipping)
        self._chisq_diag_used, self._chisq_top_used = self._compute_chisq_diag(
            z=all_z, snps=self.matched_snps, a1=self.a1, a2=self.a2, nmax=self.nsamp, topk=10
        )

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
        self.nsnps = M

        blk_size = max(M // self.nblks, 1)
        blk_idx = (np.arange(M, dtype=np.int64) // blk_size)
        blk_idx[blk_idx >= self.nblks] = self.nblks - 1
        self.blk_idx = blk_idx

        # chi^2 = z^2
        z2 = all_z * all_z
        if not np.isfinite(z2).all():
            # extremely rare (overflow). Treat as outliers only if clipping is enabled;
            # otherwise refuse to proceed because moments are undefined.
            if action == "clip" and thr_enabled:
                bad = ~np.isfinite(z2)
                z2[bad] = thr
            else:
                raise RuntimeError("Non-finite chi^2 encountered after squaring z-scores.")

        # Robustification: winsorize chi^2 on matched set (RHS only)
        self._chisq_clip_applied = False
        self._chisq_clip_count = 0
        self._chisq_clip_threshold = float(thr) if thr_enabled else 0.0

        if action == "clip" and thr_enabled:
            # count how many would have been clipped
            clip_mask = (z2 > thr) & np.isfinite(z2)
            self._chisq_clip_count = int(np.sum(clip_mask))
            if self._chisq_clip_count > 0:
                np.minimum(z2, thr, out=z2)  # in-place winsorization
                self._chisq_clip_applied = True

        # Full sums: M_k = sum a_{jk}; S_k = sum a_{jk} z_j^2 (robust if clip applied)
        Ak_full = A.sum(axis=0, dtype=np.float64)
        Az2_full = (A.T @ z2).astype(np.float64, copy=False)

        B = self.nblks
        K = self.nbins
        Ak_blk = np.zeros((B, K), dtype=np.float64)
        Az2_blk = np.zeros((B, K), dtype=np.float64)

        tmp = np.empty(M, dtype=np.float64)
        for k in range(K):
            col = A[:, k]
            tmp[:] = col
            Ak_blk[:, k] = np.bincount(blk_idx, weights=tmp, minlength=B).astype(np.float64, copy=False)
            tmp[:] = col * z2
            Az2_blk[:, k] = np.bincount(blk_idx, weights=tmp, minlength=B).astype(np.float64, copy=False)

        self.nsnps_bin = Ak_full
        self.nsnps_blk = Ak_blk

        self._Ak_full = Ak_full
        self._Az2_full = Az2_full
        self._Ak_blk = Ak_blk
        self._Az2_blk = Az2_blk

        bad_bins = np.flatnonzero(~np.isfinite(Ak_full) | (Ak_full <= 0.0))
        if bad_bins.size:
            self.log._log(
                "!!! One or more annotation bins have non-positive total weight after matching. "
                f"Bad bins: {bad_bins.tolist()} (Ak_full={Ak_full[bad_bins].tolist()}) !!!"
            )
            sys.exit(1)

        # optional: in 'warn' mode, just log that threshold exists but nothing was applied
        if action == "warn" and thr_enabled and printlog:
            # keep it short (full tail/top already available via your diagnostics)
            self.log._log(f"[chisq] [{self.name}] warning-only threshold set at {thr:.3f}; no dropping/clipping applied.")


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

