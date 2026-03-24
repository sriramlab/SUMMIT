from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pandas as pd

import utils
from jackknife import JackknifeSpec, JackknifeDesign
from trace import Trace
from sumstats import Sumstats, MatchedSumstats
from h2core import prepare_h2, fit_h2
from rgcore import prepare_rg, fit_intercept, fit_rg, RGResultWriter
from moments import build_rg_summary_moment


class Sumcore:
    """
    Thin wrapper around the refactored bivariate rg core.

    Pipeline:
        Trace (immutable base) -> Sumstats x 2 (immutable read-QC)
        -> final main keep mask on the trace base axis
        -> TraceView + MatchedSumstats x 2 on the SAME SNP axis
        -> one JackknifeDesign built on that final main SNP axis
        -> h2 fits for trait 1 / trait 2 on that same axis
        -> intercept fit on a stricter subset mask of that same axis
        -> gamma_g / rg fit on the same axis / same replicate ordering
    """

    def __init__(
        self,
        bim_path=None,
        rg=None,
        ldscores=None,
        ldscores_reg=None,
        log=None,
        verbose=False,
        chisq_threshold=0,
        annot=None,
        njack=None,
        out=None,
        align_alleles=False,
        drop_ambiguous=True,
        collapse_reg_ld=False,
        enrich_mode: str = "auto",
        jack_mode: str = "mean",
        clip_nonfinite_vals=False,
        rg_se_method: str = "jackknife",
        intercept_chisq_thr=None,
        intercept_weight_mode: str = "ldsc",
        intercept_rg=None,
        pheno_rg=None,
        pheno_rg_cov=None,
        pheno_rg_missing_values=None,
        pheno_rg_cov_missing_values=None,
        chisq_action: str = "drop",
        report_tau: bool = True,
        allow_neg_enr: bool = False,
        adjust_delta: bool = False,
        cov_rank=None,
    ):

        self.log = log
        self.verbose = verbose
        self.verbose_level = utils._parse_verbose(verbose)
        self.start_time = utils._get_time()
        if self.log is not None:
            self.log._log("Analysis started at: " + utils._get_timestr(self.start_time))

        self.phen_paths = utils._parse_rg_pair(rg)
        self.phen_names = [utils._phen_name_from_path(p) for p in self.phen_paths]

        self.trace = Trace(
            bimpath=bim_path,
            sumpath=None,
            savepath=None,
            log=self.log,
            ldscores=ldscores,
            ldscores_reg=ldscores_reg,
            annot=annot,
            verbose=bool(self.verbose_level),
            delta=None,
        )

        self.jackknife_spec = JackknifeSpec.parse(njack)
        self.align_alleles = bool(align_alleles)
        self.drop_ambiguous = bool(drop_ambiguous)
        self.collapse_reg_ld = bool(collapse_reg_ld)
        self.chisq_threshold = chisq_threshold
        self.intercept_chisq_thr = intercept_chisq_thr
        self.intercept_weight_mode = str(intercept_weight_mode).strip().lower()
        self.chisq_action = str(chisq_action).strip().lower()
        self.enrich_mode = enrich_mode
        self.jack_mode = jack_mode
        self.clip_nonfinite_vals = bool(clip_nonfinite_vals)
        self.report_tau = bool(report_tau)
        self.allow_neg_enr = bool(allow_neg_enr)
        self.adjust_delta = bool(adjust_delta)
        self.nan_policy = "propagate" if self.clip_nonfinite_vals else "omit"
        self.rg_se_method = str(rg_se_method).strip().lower()
        if self.rg_se_method not in {"jackknife", "delta", "robust", "kmoments"}:
            raise ValueError("rg_se_method must be one of {'jackknife','delta','robust','kmoments'}")
        if self.chisq_action not in ("drop", "clip", "warn", "none"):
            raise ValueError("chisq_action must be one of {'drop','clip','warn','none'}")
        if self.intercept_weight_mode not in {"ldsc", "score"}:
            raise ValueError("intercept_weight_mode must be one of {'ldsc','score'}")

        self.intercept_rg = None if intercept_rg is None else float(intercept_rg)
        self.pheno_rg_paths = None if pheno_rg is None else utils._parse_rg_pair(pheno_rg)
        self.pheno_rg_cov_paths = None if pheno_rg_cov is None else utils._parse_rg_pair(pheno_rg_cov)

        if self.intercept_rg is not None and (
            self.pheno_rg_paths is not None or self.pheno_rg_cov_paths is not None
        ):
            raise ValueError("Provide at most one of --intercept-rg or (--pheno-rg [and --pheno-rg-cov]).")
        if self.pheno_rg_cov_paths is not None and self.pheno_rg_paths is None:
            raise ValueError("--pheno-rg-cov requires --pheno-rg.")
        if self.pheno_rg_paths is not None and len(self.pheno_rg_paths) != 2:
            raise ValueError("--pheno-rg must contain exactly two files.")
        if self.pheno_rg_cov_paths is not None and len(self.pheno_rg_cov_paths) != 2:
            raise ValueError("--pheno-rg-cov must contain exactly two files.")

        self.pheno_rg_missing_values = self._parse_missing_tokens(pheno_rg_missing_values)
        self.pheno_rg_cov_missing_values = self._parse_missing_tokens(pheno_rg_cov_missing_values)
        self.cov_rank_values = self._parse_cov_rank_values(cov_rank, expected=2)

        self.out = out
        self.result = None
        self.trace_view = None
        self.jackknife = None
        self.matched1 = None
        self.matched2 = None

    def _run(self):
        pheno_cov_rank_override = self._derive_cov_rank_overrides_from_pheno()

        if pheno_cov_rank_override is not None:
            cov_rank1, cov_rank2 = pheno_cov_rank_override
            cov_rank_source1 = "pheno-rg-cov"
            cov_rank_source2 = "pheno-rg-cov"
        else:
            cov_rank1, cov_rank2 = self.cov_rank_values
            cov_rank_source1 = "cli" if cov_rank1 is not None else None
            cov_rank_source2 = "cli" if cov_rank2 is not None else None

        ss1 = Sumstats.from_file(
            self.phen_paths[0],
            name=self.phen_names[0],
            log=self.log,
            cov_rank=cov_rank1,
            cov_rank_source=cov_rank_source1,
        )
        ss2 = Sumstats.from_file(
            self.phen_paths[1],
            name=self.phen_names[1],
            log=self.log,
            cov_rank=cov_rank2,
            cov_rank_source=cov_rank_source2,
        )

        aligned1 = ss1.align_to_trace(self.trace)
        aligned2 = ss2.align_to_trace(self.trace)

        keep1 = aligned1.keep_mask(
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
        )
        keep2 = aligned2.keep_mask(
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
        )
        main_mask = keep1 & keep2

        flip_keep = None
        if self.align_alleles:
            main_mask, flip_keep = self._apply_allele_alignment_filter(aligned1, aligned2, main_mask)

        n_keep = int(np.sum(main_mask))
        n_base = int(self.trace.nsnps)
        if n_keep == 0:
            raise RuntimeError("No SNPs remain after matching / filtering / allele alignment.")
        if self.log is not None:
            self.log._log(
                f"Final SNP set for SUMCORE: keeping {n_keep} SNPs out of {n_base} Trace SNPs."
            )

        tv = self.trace.materialize_view(main_mask)
        matched1 = aligned1.materialize(
            main_mask,
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
        )
        matched2 = aligned2.materialize(
            main_mask,
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
        )
        if flip_keep is not None:
            matched2 = self._flip_matched_sumstats(matched2, flip_keep)

        jk = JackknifeDesign.from_trace_view(tv, self.jackknife_spec, log=self.log)

        h2_fit1 = fit_h2(
            prepare_h2(tv, matched1, jk, adjust_delta=self.adjust_delta),
            enrich_mode=self.enrich_mode,
            report_tau=self.report_tau,
            allow_neg_enr=self.allow_neg_enr,
            clip_nonfinite_vals=self.clip_nonfinite_vals,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )
        h2_fit2 = fit_h2(
            prepare_h2(tv, matched2, jk, adjust_delta=self.adjust_delta),
            enrich_mode=self.enrich_mode,
            report_tau=self.report_tau,
            allow_neg_enr=self.allow_neg_enr,
            clip_nonfinite_vals=self.clip_nonfinite_vals,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )

        fixed_c, fixed_info = self._resolve_external_intercept(matched1, matched2)

        summary_y, summary_y_info = build_rg_summary_moment(matched1, matched2)
        if self.log is not None:
            self.log._log(
                f"[rg:rhs] summary moment mode={summary_y_info.get('mode', 'unknown')}; "
                f"trait1_cov_rank={summary_y_info.get('trait1_cov_rank', 'NA')} "
                f"({summary_y_info.get('trait1_cov_rank_source', 'NA')}), "
                f"trait2_cov_rank={summary_y_info.get('trait2_cov_rank', 'NA')} "
                f"({summary_y_info.get('trait2_cov_rank_source', 'NA')}), "
                f"trait1_n_scale={summary_y_info.get('trait1_n_scale', np.nan):.6g}, "
                f"trait2_n_scale={summary_y_info.get('trait2_n_scale', np.nan):.6g}, "
                f"nonfinite_reconstructed_snps={summary_y_info.get('n_nonfinite', 0)}"
            )

        intercept = fit_intercept(
            tv,
            matched1,
            matched2,
            jk,
            h2_fit1,
            h2_fit2,
            summary_y=summary_y,
            summary_y_info=summary_y_info,
            fixed_c=fixed_c,
            fixed_info=fixed_info,
            intercept_chisq_threshold=self.intercept_chisq_thr,
            intercept_weight_mode=self.intercept_weight_mode,
            collapse_reg_ld=self.collapse_reg_ld,
            log=self.log,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )

        rg_fit = fit_rg(
            prepare_rg(
                tv,
                matched1,
                matched2,
                jk,
                summary_y=summary_y,
                summary_y_info=summary_y_info,
                adjust_delta=self.adjust_delta,
            ),
            h2_fit1,
            h2_fit2,
            intercept,
            rg_se_method=self.rg_se_method,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )

        self.trace_view = tv
        self.jackknife = jk
        self.matched1 = matched1
        self.matched2 = matched2
        self.result = {
            "h2_fit1": h2_fit1,
            "h2_fit2": h2_fit2,
            "intercept": intercept,
            "rg_fit": rg_fit,
            "sumstats1": ss1,
            "sumstats2": ss2,
        }

        if hasattr(ss1, "log_chisq_diagnostics"):
            ss1.log_chisq_diagnostics(
                matched1,
                chisq_threshold=self.chisq_threshold,
                verbose=(self.verbose_level >= 1),
            )
            ss2.log_chisq_diagnostics(
                matched2,
                chisq_threshold=self.chisq_threshold,
                verbose=(self.verbose_level >= 1),
            )

        if self.verbose_level >= 2 and self.out is not None:
            jack_path = f"{self.out}.rg.jack"
            RGResultWriter.save_jackknife_text(rg_fit, jack_path)
            if self.log is not None:
                self.log._log(f"Saved rg jackknife replicate dump to {jack_path}")

            info = intercept.info if isinstance(intercept.info, dict) else {}
            n_overlap = info.get("n_overlap", None)
            if n_overlap is not None:
                eq_path = f"{self.out}.rg.scoreeq.json"
                RGResultWriter.save_score_normal_equations_json(rg_fit, eq_path)
                if self.log is not None:
                    self.log._log(f"Saved explicit full SCORE normal-equation dump to {eq_path}")

        return self.result

    def _logoff(self):
        if self.result is None:
            return

        h2_fit1 = self.result["h2_fit1"]
        h2_fit2 = self.result["h2_fit2"]
        intercept = self.result["intercept"]
        rg_fit = self.result["rg_fit"]
        km_info = getattr(rg_fit, "kmoment_info", None)
        if km_info is not None:
            self.log._log(
                "[rg:kmom] "
                f"single-component model-based SE used; "
                f"moment_source={km_info.get('moment_source', 'NA')}, "
                f"alpha_probe={km_info.get('alpha_probe', np.nan):.6g}, "
                f"alpha_probe_err={km_info.get('alpha_probe_err', np.nan):.3e}, "
                f"delta_reff={km_info.get('delta_reff', np.nan):.6g}, "
                f"var_gamma={km_info.get('var_gamma', np.nan):.6g}"
            )
        fits = [h2_fit1, h2_fit2]

        for t, fit in enumerate(fits):
            name = self.phen_names[t]
            if fit.enrich_mode_used:
                self.log._log(f"^^^ Phenotype [{name}] enrichment_mode_used: {fit.enrich_mode_used}")

            if self.trace.nbins > 1:
                for j, header in enumerate(self.trace.annot_header):
                    sig = fit.sigmas[j, 0]
                    sigse = fit.sigmas[j, 1]
                    h2 = fit.h2[j, 0]
                    h2se = fit.h2[j, 1]
                    enr = fit.enrich[j, 0]
                    ense = fit.enrich[j, 1]

                    line = (
                        f"^^^ Phenotype [{name}] Bin [{header}] "
                        f"sigma_g^2: {sig:.6g} (SE: {sigse:.6g}) "
                        f"h^2_cat: {h2:.6g} (SE: {h2se:.6g}) "
                        f"Enrichment: {enr:.6g} (SE: {ense:.6g})"
                    )
                    if fit.enrich_nonoverlap is not None and fit.enrich_overlap is not None:
                        line += (
                            f" Enrichment_nonoverlap: {fit.enrich_nonoverlap[j, 0]:.6g} "
                            f"(SE: {fit.enrich_nonoverlap[j, 1]:.6g})"
                            f" Enrichment_overlap: {fit.enrich_overlap[j, 0]:.6g} "
                            f"(SE: {fit.enrich_overlap[j, 1]:.6g})"
                        )
                    if self.report_tau and fit.tau is not None and fit.tau_star is not None:
                        line += (
                            f" tau: {fit.tau[j, 0]:.6g} (SE: {fit.tau[j, 1]:.6g})"
                            f" tau_*: {fit.tau_star[j, 0]:.6g} (SE: {fit.tau_star[j, 1]:.6g})"
                        )
                    self.log._log(line)

            self.log._log(
                f"^^^ Phenotype [{name}] Total SNP heritability (h^2): "
                f"{fit.h2[-1, 0]:.6g} SE: {fit.h2[-1, 1]:.6g}"
            )

        self.log._log(
            f"^^^ Phenotype [{self.phen_names[0]}] & [{self.phen_names[1]}] "
            f"Intercept (c): {intercept.c[0]:.9g} (SE: {intercept.c[1]:.6g})"
        )

        for j, header in enumerate(self.trace.annot_header):
            self.log._log(
                f"^^^ Phenotype [{self.phen_names[0]}] & [{self.phen_names[1]}] Bin [{header}] "
                f"gamma_g: {rg_fit.gamma[j, 0]:.6g} (SE: {rg_fit.gamma[j, 1]:.6g}) "
                f"rg: {rg_fit.rg[j, 0]:.6g} (SE: {rg_fit.rg[j, 1]:.6g})"
            )

        self.log._log(
            f"^^^ Phenotype [{self.phen_names[0]}] & [{self.phen_names[1]}] "
            f"Total genetic covariance (gamma_g): {rg_fit.gamma_total[0]:.6g} "
            f"(SE: {rg_fit.gamma_total[1]:.6g})"
        )
        self.log._log(
            f"^^^ Phenotype [{self.phen_names[0]}] & [{self.phen_names[1]}] "
            f"Total genetic correlation (rg): {rg_fit.rg_total[0]:.6g} "
            f"(SE: {rg_fit.rg_total[1]:.6g})"
        )

        self.end_time = utils._get_time()
        self.log._log("Analysis ended at: " + utils._get_timestr(self.end_time))
        self.log._log("run time: " + format(self.end_time - self.start_time, ".3f") + " s")
        if self.out is not None:
            self.log._log("Saved log in " + self.out + ".log")
            self.log._save_log(self.out + ".log")

    @staticmethod
    def _parse_missing_tokens(raw):
        default = "-9,NA,NaN,nan,.,None,NONE,null,NULL"
        s = default if raw is None else str(raw)
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                out.append(tok)
        return out

    @staticmethod
    def _split_missing_tokens(missing_tokens):
        """
        Split user-provided missing tokens into:
          - string tokens used directly in read_csv(na_values=...)
          - numeric sentinels that must ALSO be masked after numeric parsing
            (so '-9' catches -9, -9.0, -9.000000, etc.)
        """
        str_tokens = []
        num_tokens = []
        for tok in missing_tokens:
            st = str(tok).strip()
            if st == "":
                continue
            str_tokens.append(st)
            try:
                num_tokens.append(float(st))
            except Exception:
                pass

        if len(num_tokens) > 0:
            num_tokens = np.unique(np.asarray(num_tokens, dtype=np.float64))
        else:
            num_tokens = np.empty(0, dtype=np.float64)

        return str_tokens, num_tokens

    @staticmethod
    def _mask_numeric_missing_in_series(s: pd.Series, numeric_missing: np.ndarray) -> pd.Series:
        """
        Convert a phenotype-like column to numeric and additionally mask numeric sentinels.
        """
        x = pd.to_numeric(s, errors="coerce")
        if numeric_missing.size == 0:
            return x

        vals = x.to_numpy(dtype=np.float64, copy=False)
        bad = np.zeros(vals.shape, dtype=bool)
        for mv in numeric_missing:
            bad |= np.isfinite(vals) & (vals == mv)
        if np.any(bad):
            x = x.mask(bad)
        return x

    @staticmethod
    def _mask_numeric_missing_in_frame(df: pd.DataFrame, numeric_missing: np.ndarray) -> pd.DataFrame:
        """
        For each covariate column, detect numeric sentinel values even if the column was
        read as float or object. This does NOT force categorical columns to numeric;
        it only marks rows as missing when the numeric parse equals a sentinel.
        """
        if numeric_missing.size == 0 or df.shape[1] == 0:
            return df

        out = df.copy()
        for col in out.columns:
            num = pd.to_numeric(out[col], errors="coerce")
            vals = num.to_numpy(dtype=np.float64, copy=False)
            bad = np.zeros(vals.shape, dtype=bool)
            for mv in numeric_missing:
                bad |= np.isfinite(vals) & (vals == mv)
            if np.any(bad):
                out.loc[bad, col] = np.nan
        return out

    @staticmethod
    def _read_overlap_phenotype_file(path: str, missing_tokens):
        """
        Read a whitespace-delimited phenotype file with header.
        Assumptions:
            - first two columns are FID, IID
            - last column is the phenotype
        Missingness is handled in two stages:
            1) string tokens via read_csv(na_values=...)
            2) numeric sentinels (e.g. -9.0) after numeric parsing
        """
        str_missing, num_missing = Sumcore._split_missing_tokens(missing_tokens)

        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=0,
            na_values=list(str_missing),
            keep_default_na=True,
            comment="#",
        )
        if df.shape[1] < 3:
            raise ValueError(
                f"Phenotype file '{path}' must have at least 3 columns: FID IID PHENO."
            )

        n_raw = int(df.shape[0])

        df = df.iloc[:, [0, 1, -1]].copy()
        df.columns = ["FID", "IID", "_y_raw"]

        df["FID"] = df["FID"].astype(str).str.strip()
        df["IID"] = df["IID"].astype(str).str.strip()

        df["_y_raw"] = Sumcore._mask_numeric_missing_in_series(df["_y_raw"], num_missing)

        n_missing_pheno = int(df["_y_raw"].isna().sum())
        n_missing_id = int(df["FID"].isna().sum() + df["IID"].isna().sum())

        df = df.dropna(subset=["FID", "IID", "_y_raw"])

        if df.shape[0] == 0:
            raise ValueError(f"Phenotype file '{path}' has no usable rows after filtering missing values.")
        if df.duplicated(subset=["FID", "IID"]).any():
            raise ValueError(f"Phenotype file '{path}' contains duplicate FID/IID rows.")

        return df, {
            "path": path,
            "n_raw": n_raw,
            "n_missing_pheno": n_missing_pheno,
            "n_missing_id": n_missing_id,
            "n_used": int(df.shape[0]),
        }


    def _prepare_trait_table_for_external_intercept(self, pheno_path: str, cov_path: str | None):
        pheno_df, pheno_info = self._read_overlap_phenotype_file(
            pheno_path,
            self.pheno_rg_missing_values,
        )

        if self.log is not None:
            self.log._log(
                f"[rg:c] pheno file '{pheno_path}': "
                f"raw={pheno_info['n_raw']}, "
                f"dropped_missing_pheno={pheno_info['n_missing_pheno']}, "
                f"kept={pheno_info['n_used']}"
            )

        if cov_path is None:
            return pheno_df, {
                "pheno_path": pheno_path,
                "cov_path": None,
                "n_pheno_used": pheno_info["n_used"],
                "n_final": int(pheno_df.shape[0]),
                "n_cov": 0,
            }

        cov_df, cov_info = self._read_overlap_covariate_file(
            cov_path,
            self.pheno_rg_cov_missing_values,
        )

        if self.log is not None:
            self.log._log(
                f"[rg:c] cov file '{cov_path}': "
                f"raw={cov_info['n_raw']}, "
                f"dropped_rows_with_missing_cov={cov_info['n_missing_cov_rows']}, "
                f"kept={cov_info['n_used']}, "
                f"n_cov={cov_info['n_cov']}"
            )

        merged = pheno_df.merge(cov_df, on=["FID", "IID"], how="inner")
        if merged.shape[0] == 0:
            raise RuntimeError(
                f"No overlapping FID/IID rows remained after merging phenotype and covariate files:\n"
                f"  phenotype: {pheno_path}\n"
                f"  covariates: {cov_path}"
            )
        if merged.duplicated(subset=["FID", "IID"]).any():
            raise RuntimeError("Duplicate FID/IID rows after phenotype/covariate merge.")

        if self.log is not None:
            self.log._log(
                f"[rg:c] merged external table for '{pheno_path}': kept={merged.shape[0]}"
            )

        return merged, {
            "pheno_path": pheno_path,
            "cov_path": cov_path,
            "n_pheno_used": pheno_info["n_used"],
            "n_cov_used": cov_info["n_used"],
            "n_final": int(merged.shape[0]),
            "n_cov": cov_info["n_cov"],
        }
    
    @staticmethod
    def _read_overlap_covariate_file(path: str, missing_tokens):
        """
        Read a whitespace-delimited covariate file with header.
        Assumptions:
            - first two columns are FID, IID
            - all remaining columns are covariates
        Missingness is handled in two stages:
            1) string tokens via read_csv(na_values=...)
            2) numeric sentinels (e.g. -9.0) after parsing
        Non-numeric covariates are one-hot encoded.
        Rows with any missing covariate are dropped.
        """
        str_missing, num_missing = Sumcore._split_missing_tokens(missing_tokens)

        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            header=0,
            na_values=list(str_missing),
            keep_default_na=True,
            comment="#",
        )
        if df.shape[1] < 3:
            raise ValueError(
                f"Covariate file '{path}' must have at least 3 columns: FID IID COV1 ..."
            )

        n_raw = int(df.shape[0])

        ids = df.iloc[:, [0, 1]].copy()
        ids.columns = ["FID", "IID"]
        ids["FID"] = ids["FID"].astype(str).str.strip()
        ids["IID"] = ids["IID"].astype(str).str.strip()

        cov = df.iloc[:, 2:].copy()
        cov = Sumcore._mask_numeric_missing_in_frame(cov, num_missing)

        tmp = pd.concat([ids, cov], axis=1)
        cov_cols = list(tmp.columns[2:])

        n_missing_cov_rows = int(tmp.iloc[:, 2:].isna().any(axis=1).sum())
        tmp = tmp.dropna(subset=cov_cols)

        if tmp.shape[0] == 0:
            raise ValueError(f"Covariate file '{path}' has no usable rows after filtering missing values.")
        if tmp.duplicated(subset=["FID", "IID"]).any():
            raise ValueError(f"Covariate file '{path}' contains duplicate FID/IID rows.")

        cov = pd.get_dummies(tmp.iloc[:, 2:], drop_first=False, dummy_na=False)
        if cov.shape[1] == 0:
            raise ValueError(f"Covariate file '{path}' has no usable covariate columns after parsing.")

        for col in cov.columns:
            cov[col] = pd.to_numeric(cov[col], errors="raise").astype(np.float64)

        cov.columns = [f"_cov{k}" for k in range(cov.shape[1])]
        out = pd.concat(
            [tmp.loc[:, ["FID", "IID"]].reset_index(drop=True), cov.reset_index(drop=True)],
            axis=1,
        )

        return out, {
            "path": path,
            "n_raw": n_raw,
            "n_missing_cov_rows": n_missing_cov_rows,
            "n_used": int(out.shape[0]),
            "n_cov": int(cov.shape[1]),
        }

    @staticmethod
    def _design_from_trait_table(df: pd.DataFrame):
        y = df["_y_raw"].to_numpy(dtype=np.float64, copy=False)
        cov_cols = [c for c in df.columns if c.startswith("_cov")]
        n = int(df.shape[0])

        if cov_cols:
            X = np.empty((n, 1 + len(cov_cols)), dtype=np.float64)
            X[:, 0] = 1.0
            X[:, 1:] = df.loc[:, cov_cols].to_numpy(dtype=np.float64, copy=False)
        else:
            X = np.ones((n, 1), dtype=np.float64)

        return y, X, cov_cols

    @staticmethod
    def _solve_small_linear(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lhs = np.asarray(lhs, dtype=np.float64)
        rhs = np.asarray(rhs, dtype=np.float64)
        lhs = 0.5 * (lhs + lhs.T)

        try:
            sol = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            try:
                sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            except np.linalg.LinAlgError:
                sol = np.linalg.pinv(lhs) @ rhs

        sol = np.asarray(sol, dtype=np.float64)
        if not np.isfinite(sol).all():
            raise RuntimeError("Non-finite solution encountered while residualizing phenotype on covariates.")
        return sol

    @classmethod
    def _compute_external_c_from_summaries(
        cls,
        A1,
        b1,
        yy1,
        A2,
        b2,
        yy2,
        A12,
        b1y2,
        b2y1,
        y12,
    ) -> float:
        """
        Compute the exact score-scale intercept

            c = <r1, r2> / sqrt(RSS1 * RSS2)

        where r_a is the phenotype residual after projecting y_a onto [1, covariates_a].

        This is the correct intercept scale for the exact score moments used in
        build_rg_summary_moment(), and is independent of n_scale.
        """
        try:
            alpha1 = cls._solve_small_linear(A1, b1)
            alpha2 = cls._solve_small_linear(A2, b2)
        except RuntimeError:
            return np.nan

        rss1 = float(yy1 - np.dot(b1, alpha1))
        rss2 = float(yy2 - np.dot(b2, alpha2))
        rss1 = max(rss1, 0.0)
        rss2 = max(rss2, 0.0)
        if not (np.isfinite(rss1) and np.isfinite(rss2) and rss1 > 0.0 and rss2 > 0.0):
            return np.nan

        cross = float(
            y12
            - np.dot(alpha1, b1y2)
            - np.dot(alpha2, b2y1)
            + np.dot(alpha1, A12 @ alpha2)
        )

        den = float(np.sqrt(rss1 * rss2))
        if not (np.isfinite(den) and den > 0.0):
            return np.nan

        return float(cross / den)

    def _compute_fixed_intercept_from_pheno(self, matched1, matched2):
        if self.pheno_rg_paths is None or len(self.pheno_rg_paths) != 2:
            raise ValueError("--pheno-rg must contain exactly two comma-separated phenotype files.")

        cov_paths = (None, None) if self.pheno_rg_cov_paths is None else tuple(self.pheno_rg_cov_paths)

        trait1_df, trait1_info = self._prepare_trait_table_for_external_intercept(
            self.pheno_rg_paths[0],
            cov_paths[0],
        )
        trait2_df, trait2_info = self._prepare_trait_table_for_external_intercept(
            self.pheno_rg_paths[1],
            cov_paths[1],
        )

        y1_full, C1_full, cov1_cols = self._design_from_trait_table(trait1_df)
        y2_full, C2_full, cov2_cols = self._design_from_trait_table(trait2_df)

        cov_rank1_ext = int(np.linalg.matrix_rank(C1_full) - 1)
        cov_rank2_ext = int(np.linalg.matrix_rank(C2_full) - 1)
        if cov_rank1_ext < 0 or cov_rank2_ext < 0:
            raise RuntimeError("Derived negative cov_rank from external phenotype/covariate design.")

        if self.log is not None:
            if int(matched1.cov_rank) != cov_rank1_ext:
                self.log._log(
                    f"[rg:c] WARNING: trait1 matched cov_rank={matched1.cov_rank} "
                    f"but external design implies cov_rank={cov_rank1_ext}."
                )
            if int(matched2.cov_rank) != cov_rank2_ext:
                self.log._log(
                    f"[rg:c] WARNING: trait2 matched cov_rank={matched2.cov_rank} "
                    f"but external design implies cov_rank={cov_rank2_ext}."
                )

        overlap_df = trait1_df.merge(
            trait2_df,
            on=["FID", "IID"],
            how="inner",
            suffixes=("_1", "_2"),
        )
        n_overlap = int(overlap_df.shape[0])
        if n_overlap <= 0:
            raise RuntimeError("No overlapping individuals were found across the two --pheno-rg files.")

        overlap_df = overlap_df.sort_values(["FID", "IID"], kind="mergesort").reset_index(drop=True)

        y1_ov = overlap_df["_y_raw_1"].to_numpy(dtype=np.float64, copy=False)
        y2_ov = overlap_df["_y_raw_2"].to_numpy(dtype=np.float64, copy=False)

        C1_ov = np.empty((n_overlap, C1_full.shape[1]), dtype=np.float64)
        C1_ov[:, 0] = 1.0
        if cov1_cols:
            C1_ov[:, 1:] = overlap_df[[f"{c}_1" for c in cov1_cols]].to_numpy(dtype=np.float64, copy=False)

        C2_ov = np.empty((n_overlap, C2_full.shape[1]), dtype=np.float64)
        C2_ov[:, 0] = 1.0
        if cov2_cols:
            C2_ov[:, 1:] = overlap_df[[f"{c}_2" for c in cov2_cols]].to_numpy(dtype=np.float64, copy=False)

        A1_full = C1_full.T @ C1_full
        b1_full = C1_full.T @ y1_full
        yy1_full = float(np.dot(y1_full, y1_full))

        A2_full = C2_full.T @ C2_full
        b2_full = C2_full.T @ y2_full
        yy2_full = float(np.dot(y2_full, y2_full))

        A12_full = C1_ov.T @ C2_ov
        b1y2_full = C1_ov.T @ y2_ov
        b2y1_full = C2_ov.T @ y1_ov
        y12_full = float(np.dot(y1_ov, y2_ov))

        c_full = self._compute_external_c_from_summaries(
            A1_full,
            b1_full,
            yy1_full,
            A2_full,
            b2_full,
            yy2_full,
            A12_full,
            b1y2_full,
            b2y1_full,
            y12_full,
        )
        if not np.isfinite(c_full):
            raise RuntimeError("Computed non-finite fixed intercept from --pheno-rg / --pheno-rg-cov.")

        c_se = 0.0
        nblocks = self._choose_pheno_nblocks(n_overlap)
        if nblocks >= 2:
            jk_ph = self._make_block_jackknife_from_length(n_overlap, nblocks)
            B = jk_ph.nunit
            p1 = C1_full.shape[1]
            p2 = C2_full.shape[1]

            A1_del = np.zeros((B, p1, p1), dtype=np.float64)
            b1_del = np.zeros((B, p1), dtype=np.float64)
            yy1_del = np.zeros(B, dtype=np.float64)

            A2_del = np.zeros((B, p2, p2), dtype=np.float64)
            b2_del = np.zeros((B, p2), dtype=np.float64)
            yy2_del = np.zeros(B, dtype=np.float64)

            A12_del = np.zeros((B, p1, p2), dtype=np.float64)
            b1y2_del = np.zeros((B, p1), dtype=np.float64)
            b2y1_del = np.zeros((B, p2), dtype=np.float64)
            y12_del = np.zeros(B, dtype=np.float64)

            for u in range(B):
                s = int(jk_ph.starts[u])
                e = int(jk_ph.ends[u])
                if e <= s:
                    continue

                C1u = C1_ov[s:e]
                C2u = C2_ov[s:e]
                y1u = y1_ov[s:e]
                y2u = y2_ov[s:e]

                A1_del[u] = C1u.T @ C1u
                b1_del[u] = C1u.T @ y1u
                yy1_del[u] = float(np.dot(y1u, y1u))

                A2_del[u] = C2u.T @ C2u
                b2_del[u] = C2u.T @ y2u
                yy2_del[u] = float(np.dot(y2u, y2u))

                A12_del[u] = C1u.T @ C2u
                b1y2_del[u] = C1u.T @ y2u
                b2y1_del[u] = C2u.T @ y1u
                y12_del[u] = float(np.dot(y1u, y2u))

            c_rep = np.full(B, np.nan, dtype=np.float64)
            for u in range(B):
                c_rep[u] = self._compute_external_c_from_summaries(
                    A1_full - A1_del[u],
                    b1_full - b1_del[u],
                    yy1_full - yy1_del[u],
                    A2_full - A2_del[u],
                    b2_full - b2_del[u],
                    yy2_full - yy2_del[u],
                    A12_full - A12_del[u],
                    b1y2_full - b1y2_del[u],
                    b2y1_full - b2y1_del[u],
                    y12_full - y12_del[u],
                )

            c_reps = np.concatenate([c_rep, np.array([c_full], dtype=np.float64)], axis=0)
            try:
                _, c_se_raw = jk_ph.summarize(
                    c_reps,
                    unit_sizes=jk_ph.unit_sizes(dtype=np.float64),
                    axis=0,
                    center=self.jack_mode,
                    nan_policy=self.nan_policy,
                )
                c_se = float(c_se_raw)
            except Exception:
                c_se = np.nan

            if not np.isfinite(c_se) or c_se < 0.0:
                if self.log is not None:
                    self.log._log(
                        "[rg:c] WARNING: phenotype-side jackknife SE for --pheno-rg "
                        "was non-finite; falling back to SE=0 for the extra c-uncertainty component."
                    )
                c_se = 0.0

        if self.log is not None:
            if abs(float(trait1_df.shape[0]) - float(matched1.nsamp)) > 0.5 or abs(float(trait2_df.shape[0]) - float(matched2.nsamp)) > 0.5:
                self.log._log(
                    "[rg:c] WARNING: external phenotype/covariate rows differ from summary-stat sample sizes."
                )

        cov_adjusted = self.pheno_rg_cov_paths is not None
        info = {
            "source": "pheno",
            "cov_adjusted": bool(cov_adjusted),
            "pheno_paths": tuple(self.pheno_rg_paths),
            "cov_paths": None if self.pheno_rg_cov_paths is None else tuple(self.pheno_rg_cov_paths),
            "n1_summary": float(matched1.nsamp),
            "n2_summary": float(matched2.nsamp),
            "n1_scale": float(matched1.n_scale),
            "n2_scale": float(matched2.n_scale),
            "n1_trait_table": int(trait1_df.shape[0]),
            "n2_trait_table": int(trait2_df.shape[0]),
            "n_overlap": n_overlap,
            "external_c_se": float(c_se),
            "external_c_se_method": "sample_block_jackknife",
            "pheno_nblocks": int(nblocks),
            "trait1_cov_rank": int(cov_rank1_ext),
            "trait2_cov_rank": int(cov_rank2_ext),
            "n_cov_trait1": int(cov_rank1_ext),
            "n_cov_trait2": int(cov_rank2_ext),
        }

        if self.log is not None:
            if cov_adjusted:
                self.log._log(
                    f"[rg:c] --pheno-rg-cov => fixed covariate-adjusted c={c_full:.6g} "
                    f"(SE: {c_se:.6g}) from {n_overlap} overlapping samples "
                    f"with {nblocks} sample jackknife blocks; "
                    f"trait1_cov_rank={cov_rank1_ext}, trait2_cov_rank={cov_rank2_ext}."
                )
            else:
                self.log._log(
                    f"[rg:c] --pheno-rg => fixed c={c_full:.6g} "
                    f"(SE: {c_se:.6g}) from {n_overlap} overlapping samples "
                    f"with {nblocks} sample jackknife blocks."
                )

        return c_full, info

    def _resolve_external_intercept(self, matched1, matched2):
        if self.intercept_rg is not None:
            if not np.isfinite(self.intercept_rg):
                raise ValueError("--intercept-rg must be finite.")
            return float(self.intercept_rg), {"source": "cli"}

        if self.pheno_rg_paths is not None:
            return self._compute_fixed_intercept_from_pheno(matched1, matched2)

        return None, None

    def _choose_pheno_nblocks(self, n_overlap: int) -> int:
        if n_overlap <= 1:
            return int(n_overlap)

        if self.jackknife_spec.mode == "block" and self.jackknife_spec.nblocks is not None:
            b = int(self.jackknife_spec.nblocks)
        else:
            b = 100

        b = max(2, min(b, int(n_overlap)))
        return b

    @staticmethod
    def _make_block_jackknife_from_length(nobs: int, nblocks: int) -> JackknifeDesign:
        nobs = int(nobs)
        nblocks = int(nblocks)
        if nobs <= 0:
            raise ValueError("Cannot construct a block jackknife on an empty sample axis.")
        nblocks = max(1, min(nblocks, nobs))

        starts = (np.arange(nblocks, dtype=np.int64) * nobs) // nblocks
        ends = (np.arange(1, nblocks + 1, dtype=np.int64) * nobs) // nblocks

        unit_id = np.empty(nobs, dtype=np.int64)
        for b in range(nblocks):
            unit_id[starts[b]:ends[b]] = b

        unit_labels = np.arange(nblocks, dtype=np.int64)
        delete_sets = np.arange(nblocks, dtype=np.int32)[:, None]
        D = np.eye(nblocks, dtype=np.float64)

        return JackknifeDesign(
            spec=JackknifeSpec(mode="block", nblocks=nblocks),
            nrep=nblocks,
            nunit=nblocks,
            unit_id=unit_id,
            unit_labels=unit_labels,
            delete_sets=delete_sets,
            D=D,
            starts=starts,
            ends=ends,
            nsnps=nobs,
        )
    
    @staticmethod
    def _parse_cov_rank_values(raw, expected: int):
        expected = int(expected)
        if raw is None:
            return [None] * expected

        vals = []
        for tok in str(raw).split(","):
            tok = tok.strip()
            if tok == "":
                continue
            v = int(tok)
            if v < 0:
                raise ValueError(f"--cov-rank values must be non-negative; got {v}")
            vals.append(v)

        if len(vals) == 1:
            vals = vals * expected

        elif len(vals) != expected:
            raise ValueError(
                f"--cov-rank must contain exactly {expected} value(s) for --rg; got {len(vals)}"
            )
        return vals


    @staticmethod
    def _alleles_to_int(a):
        a = np.asarray(a, dtype=str)
        a = np.char.upper(a)
        out = np.full(a.shape, -1, dtype=np.int8)
        out[a == "A"] = 0
        out[a == "C"] = 1
        out[a == "G"] = 2
        out[a == "T"] = 3
        return out

    def _allele_masks(self, a1_ref, a2_ref, a1, a2):
        a1r = self._alleles_to_int(a1_ref)
        a2r = self._alleles_to_int(a2_ref)
        a1 = self._alleles_to_int(a1)
        a2 = self._alleles_to_int(a2)

        valid = (a1r >= 0) & (a2r >= 0) & (a1 >= 0) & (a2 >= 0)
        amb = (
            ((a1r == 0) & (a2r == 3)) |
            ((a1r == 3) & (a2r == 0)) |
            ((a1r == 1) & (a2r == 2)) |
            ((a1r == 2) & (a2r == 1))
        )

        comp_map = np.array([3, 2, 1, 0], dtype=np.int8)
        comp_a1 = np.where(a1 >= 0, comp_map[a1], -1)
        comp_a2 = np.where(a2 >= 0, comp_map[a2], -1)

        direct = (a1r == a1) & (a2r == a2)
        strand = (a1r == comp_a1) & (a2r == comp_a2)
        swapped = (a1r == a2) & (a2r == a1)
        swapped_strand = (a1r == comp_a2) & (a2r == comp_a1)

        flip = swapped | swapped_strand
        match = direct | strand | swapped | swapped_strand
        keep = valid & match
        if self.drop_ambiguous:
            keep = keep & (~amb)
        return keep, flip

    def _apply_allele_alignment_filter(self, aligned1, aligned2, base_keep_mask):
        base_keep_mask = np.asarray(base_keep_mask, dtype=bool)
        idx = np.flatnonzero(base_keep_mask)
        if idx.size == 0:
            return base_keep_mask, None

        pos1 = aligned1.pos_on_trace[idx]
        pos2 = aligned2.pos_on_trace[idx]
        keep2, flip2 = self._allele_masks(
            aligned1.sumstats.a1[pos1],
            aligned1.sumstats.a2[pos1],
            aligned2.sumstats.a1[pos2],
            aligned2.sumstats.a2[pos2],
        )
        out = base_keep_mask.copy()
        out[idx] = keep2
        flip_keep = flip2[keep2]

        if self.log is not None:
            n_drop = int(np.sum(~keep2))
            n_flip = int(np.sum(flip_keep))
            if n_drop > 0 or n_flip > 0:
                self.log._log(
                    f"Allele alignment: dropping {n_drop} SNPs (drop_ambiguous={self.drop_ambiguous}); "
                    f"flipping trait 2 z for {n_flip} kept SNPs."
                )
        return out, flip_keep

    @staticmethod
    def _flip_matched_sumstats(matched: MatchedSumstats, flip_mask) -> MatchedSumstats:
        flip_mask = np.asarray(flip_mask, dtype=bool)
        if flip_mask.ndim != 1 or flip_mask.size != matched.nsnps:
            raise ValueError("flip_mask length mismatch with MatchedSumstats.")
        z = matched.z.copy()
        z[flip_mask] *= -1.0
        beta = matched.beta.copy()
        beta[flip_mask] *= -1.0
        a1 = matched.a1.copy()
        a2 = matched.a2.copy()
        tmp = a1[flip_mask].copy()
        a1[flip_mask] = a2[flip_mask]
        a2[flip_mask] = tmp
        return replace(matched, z=z, beta=beta, a1=a1, a2=a2)

    def _derive_cov_rank_overrides_from_pheno(self):
        """
        If --pheno-rg and --pheno-rg-cov are both provided, derive cov_rank
        (non-intercept df) from the external covariate designs and use those
        to override any CLI/file defaults for BOTH h2 and rg estimation.
        """
        if self.pheno_rg_paths is None or self.pheno_rg_cov_paths is None:
            return None

        out = []
        for pheno_path, cov_path in zip(self.pheno_rg_paths, self.pheno_rg_cov_paths):
            trait_df, _ = self._prepare_trait_table_for_external_intercept(pheno_path, cov_path)
            _, C, _ = self._design_from_trait_table(trait_df)
            cov_rank = int(np.linalg.matrix_rank(C) - 1)
            if cov_rank < 0:
                raise RuntimeError(
                    f"Derived a negative cov_rank from external design for phenotype file '{pheno_path}'."
                )
            out.append(cov_rank)

        out = tuple(int(v) for v in out)

        if self.log is not None:
            if any(
                (cli is not None) and (int(cli) != int(ext))
                for cli, ext in zip(self.cov_rank_values, out)
            ):
                self.log._log(
                    f"[cov-rank] overriding --cov-rank with values derived from "
                    f"--pheno-rg-cov: {out[0]}, {out[1]}."
                )
            else:
                self.log._log(
                    f"[cov-rank] using cov_rank derived from --pheno-rg-cov: "
                    f"{out[0]}, {out[1]}."
                )

        return out

