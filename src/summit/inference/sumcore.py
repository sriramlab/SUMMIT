from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pandas as pd

from .. import utils
from .jackknife import JackknifeSpec, JackknifeDesign
from .trace import Trace
from ..sumstats.sumstats import Sumstats, MatchedSumstats, harmonize_allele_codes
from .h2core import prepare_h2, fit_h2
from .rgcore import prepare_rg, fit_intercept, fit_rg, RGResultWriter
from .ldsc_h2 import (
    prepare_h2_ldsc,
    read_ldsc_m,
    read_ldsc_weight_ld_aligned,
    resolve_ldsc_reference_moments,
)
from ..sumstats.moments import build_rg_summary_moment


class Sumcore:
    """
    Thin wrapper around the refactored bivariate rg core.

    Pipeline:
        Trace (immutable base) -> Sumstats x 2 (immutable read-QC)
        -> final main keep mask on the trace base axis
        -> TraceView + MatchedSumstats x 2 on the SAME compact active SNP axis
        -> JackknifeDesign on that axis for contiguous block jackknife, or a
           subsetted chromosome JackknifeDesign from the full Trace axis
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
        ldscores_reg_w=None,
        ldscores_w=None,
        log=None,
        verbose=False,
        chisq_threshold=0,
        annot=None,
        njack=None,
        out=None,
        align_alleles=False,
        drop_ambiguous=True,
        collapse_reg_ld=True,
        enrich_mode: str = "auto",
        jack_mode: str = "mean",
        clip_nonfinite_vals=False,
        rg_se_method: str = "jackknife",
        intercept_chisq_thr=None,
        intercept_weight_mode: str = "score",
        intercept_rg=None,
        intercept_rg_source: str = "cli",
        pheno_rg=None,
        pheno_rg_cov=None,
        pheno_rg_missing_values=None,
        pheno_rg_cov_missing_values=None,
        chisq_action: str = "drop",
        report_tau: bool = True,
        allow_neg_enr: bool = False,
        adjust_delta: bool = False,
        cov_rank=None,
        trace_obj=None,
        sumstats_pair=None,
        aligned_pair=None,
        keep_masks=None,
        phen_names=None,
        write_jack: bool = False,
        write_normeq: bool = False,
        weight_mode: str = "he",
        ldsc_m=None,
        ldsc_irwls_iters: int = 3,
        ldsc_irwls_tol: float = 0.0,
        **_unused_kwargs,
    ):

        self.log = log
        self.verbose = verbose
        self.verbose_level = utils._parse_verbose(verbose)
        parsed_write_jack, parsed_write_normeq = utils._parse_verbose_outputs(verbose)
        self.verbose_write_jack = bool(write_jack) or bool(parsed_write_jack)
        self.verbose_write_normeq = bool(write_normeq) or bool(parsed_write_normeq)
        self.collect_diagnostics = bool(self.verbose_level >= 1)
        self.start_time = utils._get_time()
        if self.log is not None:
            self.log._log("Analysis started at: " + utils._get_timestr(self.start_time))

        self.weight_mode = str(weight_mode).strip().lower().replace("-", "_")
        if self.weight_mode in {"summit", "score", "he_regression"}:
            self.weight_mode = "he"
        if self.weight_mode not in {"he", "ldsc"}:
            raise ValueError("weight_mode must be one of {'he','ldsc'}")
        if self.weight_mode == "he" and (ldscores_w is not None or ldsc_m is not None):
            raise ValueError("--ldscores-w and --ldsc-m require --weight-mode ldsc.")
        if self.weight_mode == "ldsc" and self.verbose_write_normeq:
            raise ValueError(
                "--write-normeq is unavailable for --weight-mode ldsc because "
                "IRWLS uses replicate-specific normal equations."
            )
        if ldscores_w is not None and ldscores_reg_w is not None:
            raise ValueError(
                "Provide at most one main --ldscores-w input and legacy "
                "ldscores_reg_w input."
            )

        self._preloaded_sumstats = None if sumstats_pair is None else tuple(sumstats_pair)
        if self._preloaded_sumstats is not None and len(self._preloaded_sumstats) != 2:
            raise ValueError("sumstats_pair must contain exactly two Sumstats objects.")

        self._prealigned_pair = None if aligned_pair is None else tuple(aligned_pair)
        if self._prealigned_pair is not None and len(self._prealigned_pair) != 2:
            raise ValueError("aligned_pair must contain exactly two aligned Sumstats objects.")

        if keep_masks is None:
            self._precomputed_keep_masks = None
        else:
            masks = tuple(np.asarray(mask, dtype=bool) for mask in keep_masks)
            if len(masks) != 2:
                raise ValueError("keep_masks must contain exactly two boolean masks.")
            self._precomputed_keep_masks = masks

        if self._preloaded_sumstats is not None:
            self.phen_paths = [None, None]
            if phen_names is not None:
                if len(phen_names) != 2:
                    raise ValueError("phen_names must contain exactly two names when provided.")
                self.phen_names = [str(phen_names[0]), str(phen_names[1])]
            else:
                self.phen_names = [
                    str(getattr(self._preloaded_sumstats[0], "name", "trait1")),
                    str(getattr(self._preloaded_sumstats[1], "name", "trait2")),
                ]
        elif self._prealigned_pair is not None:
            self.phen_paths = [None, None]
            if phen_names is not None:
                if len(phen_names) != 2:
                    raise ValueError("phen_names must contain exactly two names when provided.")
                self.phen_names = [str(phen_names[0]), str(phen_names[1])]
            else:
                self.phen_names = [
                    str(getattr(self._prealigned_pair[0].sumstats, "name", "trait1")),
                    str(getattr(self._prealigned_pair[1].sumstats, "name", "trait2")),
                ]
        else:
            self.phen_paths = utils._parse_rg_pair(rg)
            if phen_names is not None:
                if len(phen_names) != 2:
                    raise ValueError("phen_names must contain exactly two names when provided.")
                self.phen_names = [str(phen_names[0]), str(phen_names[1])]
            else:
                self.phen_names = [utils._phen_name_from_path(p) for p in self.phen_paths]

        if trace_obj is None:
            self.trace = Trace(
                bimpath=bim_path,
                sumpath=None,
                savepath=None,
                log=self.log,
                ldscores=ldscores,
                ldscores_reg=ldscores_reg,
                ldscores_reg_w=ldscores_reg_w,
                annot=annot,
                verbose=bool(self.verbose_level),
                delta=None,
            )
        else:
            self.trace = trace_obj
            if hasattr(self.trace, "log"):
                self.trace.log = self.log

        self.ldsc_irwls_iters = int(ldsc_irwls_iters)
        self.ldsc_irwls_tol = float(ldsc_irwls_tol)
        if self.ldsc_irwls_iters < 1:
            raise ValueError("ldsc_irwls_iters must be at least 1")
        if not (np.isfinite(self.ldsc_irwls_tol) and self.ldsc_irwls_tol >= 0.0):
            raise ValueError("ldsc_irwls_tol must be non-negative and finite")

        self.ldsc_weight_ld = None
        self.ldsc_weight_present = None
        self.ldsc_m_annot = None
        self.ldsc_overlap_matrix = None
        self.ldsc_source_nsnps = None
        self.ldsc_m_source = None
        if self.weight_mode == "ldsc":
            m_override = (
                None
                if ldsc_m is None
                else read_ldsc_m(ldsc_m, nbins=self.trace.nbins)
            )
            reference = resolve_ldsc_reference_moments(
                annot_path=annot,
                trace_annot=self.trace.annot,
                trace_header=self.trace.annot_header,
                m_override=m_override,
                unpartitioned_source_nsnps=getattr(self.trace, "source_nsnps", None),
            )
            self.ldsc_m_annot = reference.m_annot
            self.ldsc_overlap_matrix = reference.overlap_matrix
            self.ldsc_source_nsnps = reference.source_nsnps
            self.ldsc_m_source = (
                reference.source
                if ldsc_m is None
                else f"{ldsc_m} ({reference.source})"
            )
            if ldscores_w is not None:
                self.ldsc_weight_ld, self.ldsc_weight_present = (
                    read_ldsc_weight_ld_aligned(
                        ldscores_w,
                        self.trace.snps,
                        target_chr=self.trace.chr,
                        target_bp=self.trace.bp,
                    )
                )
                n_missing = int(np.sum(~self.ldsc_weight_present))
                if self.log is not None:
                    self.log._log(
                        "[rg:ldsc] aligned scalar weight LD to the primary Trace; "
                        f"{n_missing} SNP(s) will be excluded from the common regression axis."
                    )

        if self._prealigned_pair is not None:
            for aligned in self._prealigned_pair:
                if getattr(aligned, "trace", None) is not self.trace:
                    raise ValueError(
                        "aligned_pair must be built on the same Trace object passed via trace_obj."
                    )

        if self._precomputed_keep_masks is not None:
            for keep_mask in self._precomputed_keep_masks:
                if keep_mask.ndim != 1 or keep_mask.size != self.trace.nsnps:
                    raise ValueError(
                        f"Each keep mask must be length {self.trace.nsnps}; got {keep_mask.shape}."
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
        if self.weight_mode == "ldsc" and self.rg_se_method != "jackknife":
            raise ValueError(
                "--weight-mode ldsc currently supports --rg-se-method jackknife only."
            )
        if self.weight_mode == "ldsc" and self.adjust_delta:
            raise ValueError(
                "--adjust-delta is an HE trace correction and is incompatible with "
                "--weight-mode ldsc."
            )
        if self.chisq_action not in ("drop", "clip", "warn", "none"):
            raise ValueError("chisq_action must be one of {'drop','clip','warn','none'}")
        if self.intercept_weight_mode not in {"ldsc", "score"}:
            raise ValueError("intercept_weight_mode must be one of {'ldsc','score'}")

        self.intercept_rg = None if intercept_rg is None else float(intercept_rg)
        self.intercept_rg_source = str(intercept_rg_source).strip() or "cli"
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
        stage_times = {}

        def _stage_start():
            return utils._get_time()

        def _stage_stop(name: str, started_at: float):
            stage_times[name] = stage_times.get(name, 0.0) + (utils._get_time() - started_at)

        t_stage = _stage_start()

        if self._prealigned_pair is not None:
            aligned1, aligned2 = self._prealigned_pair
            ss1 = aligned1.sumstats
            ss2 = aligned2.sumstats
            if hasattr(ss1, "log"):
                ss1.log = self.log
            if hasattr(ss2, "log"):
                ss2.log = self.log
        elif self._preloaded_sumstats is not None:
            ss1, ss2 = self._preloaded_sumstats
            if hasattr(ss1, "log"):
                ss1.log = self.log
            if hasattr(ss2, "log"):
                ss2.log = self.log
            aligned1 = ss1.align_to_trace(self.trace)
            aligned2 = ss2.align_to_trace(self.trace)
        else:
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
                compute_diagnostics=self.collect_diagnostics,
                require_alleles=self.align_alleles,
            )
            ss2 = Sumstats.from_file(
                self.phen_paths[1],
                name=self.phen_names[1],
                log=self.log,
                cov_rank=cov_rank2,
                cov_rank_source=cov_rank_source2,
                compute_diagnostics=self.collect_diagnostics,
                require_alleles=self.align_alleles,
            )

            aligned1 = ss1.align_to_trace(self.trace)
            aligned2 = ss2.align_to_trace(self.trace)

        if self._precomputed_keep_masks is not None:
            keep1, keep2 = self._precomputed_keep_masks
        else:
            keep1 = aligned1.keep_mask(
                chisq_threshold=self.chisq_threshold,
                chisq_action=self.chisq_action,
            )
            keep2 = aligned2.keep_mask(
                chisq_threshold=self.chisq_threshold,
                chisq_action=self.chisq_action,
            )

        keep1 = np.asarray(keep1, dtype=bool)
        keep2 = np.asarray(keep2, dtype=bool)
        main_mask = keep1 & keep2
        if self.ldsc_weight_present is not None:
            main_mask &= self.ldsc_weight_present
        _stage_stop("load_align_filter", t_stage)

        t_stage = _stage_start()
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
        if self.ldsc_weight_ld is not None:
            tv = replace(
                tv,
                ldscores_reg_w=np.asarray(
                    self.ldsc_weight_ld[main_mask, :],
                    dtype=np.float64,
                    order="C",
                ),
            )
        matched1 = aligned1.materialize(
            main_mask,
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
            allowed_mask=keep1,
            compute_diagnostics=self.collect_diagnostics,
        )
        matched2 = aligned2.materialize(
            main_mask,
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
            allowed_mask=keep2,
            compute_diagnostics=self.collect_diagnostics,
            allele_flip_mask=flip_keep,
        )

        if self.jackknife_spec.mode == "chr":
            full_tv = self.trace.materialize_view()
            full_jk = JackknifeDesign.from_trace_view(full_tv, self.jackknife_spec, log=self.log)
            jk = full_jk.subset(main_mask, log=self.log)
        else:
            jk = JackknifeDesign.from_trace_view(tv, self.jackknife_spec, log=self.log)
        _stage_stop("materialize_jackknife", t_stage)

        t_stage = _stage_start()
        h2_prepared1 = (
            prepare_h2_ldsc(tv, matched1, jk)
            if self.weight_mode == "ldsc"
            else prepare_h2(tv, matched1, jk, adjust_delta=self.adjust_delta)
        )
        h2_fit1 = fit_h2(
            h2_prepared1,
            enrich_mode=self.enrich_mode,
            report_tau=self.report_tau,
            allow_neg_enr=self.allow_neg_enr,
            clip_nonfinite_vals=self.clip_nonfinite_vals,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
            weight_mode=self.weight_mode,
            ldsc_m_annot=self.ldsc_m_annot,
            ldsc_overlap_matrix=self.ldsc_overlap_matrix,
            ldsc_source_nsnps=self.ldsc_source_nsnps,
            ldsc_irwls_iters=self.ldsc_irwls_iters,
            ldsc_irwls_tol=self.ldsc_irwls_tol,
        )
        _stage_stop("h2_trait1", t_stage)

        t_stage = _stage_start()
        h2_prepared2 = (
            prepare_h2_ldsc(tv, matched2, jk)
            if self.weight_mode == "ldsc"
            else prepare_h2(tv, matched2, jk, adjust_delta=self.adjust_delta)
        )
        h2_fit2 = fit_h2(
            h2_prepared2,
            enrich_mode=self.enrich_mode,
            report_tau=self.report_tau,
            allow_neg_enr=self.allow_neg_enr,
            clip_nonfinite_vals=self.clip_nonfinite_vals,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
            weight_mode=self.weight_mode,
            ldsc_m_annot=self.ldsc_m_annot,
            ldsc_overlap_matrix=self.ldsc_overlap_matrix,
            ldsc_source_nsnps=self.ldsc_source_nsnps,
            ldsc_irwls_iters=self.ldsc_irwls_iters,
            ldsc_irwls_tol=self.ldsc_irwls_tol,
        )
        _stage_stop("h2_trait2", t_stage)

        t_stage = _stage_start()
        fixed_c, fixed_info = self._resolve_external_intercept(matched1, matched2)
        _stage_stop("external_intercept", t_stage)

        t_stage = _stage_start()
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

        rg_prepared = prepare_rg(
            tv,
            matched1,
            matched2,
            jk,
            summary_y=summary_y,
            summary_y_info=summary_y_info,
            adjust_delta=self.adjust_delta,
        )
        _stage_stop("prepare_rg", t_stage)

        t_stage = _stage_start()
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
            score_prepared=rg_prepared,
            log=self.log,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )
        _stage_stop("fit_intercept", t_stage)

        t_stage = _stage_start()
        rg_fit = fit_rg(
            rg_prepared,
            h2_fit1,
            h2_fit2,
            intercept,
            rg_se_method=self.rg_se_method,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
            weight_mode=self.weight_mode,
            ldsc_m_annot=self.ldsc_m_annot,
            ldsc_irwls_iters=self.ldsc_irwls_iters,
            ldsc_irwls_tol=self.ldsc_irwls_tol,
        )
        if self.log is not None and self.weight_mode == "ldsc":
            info = rg_fit.weight_info or {}
            self.log._log(
                f"[rg:ldsc] constrained score-scale cov-LDSC IRWLS using M source "
                f"'{self.ldsc_m_source}', weight LD source "
                f"'{info.get('weight_ld_source', 'unknown')}', intercept source "
                f"'{info.get('intercept_source', 'unknown')}', "
                f"iterations={info.get('irwls_iters', 'NA')}."
            )
        _stage_stop("fit_rg", t_stage)

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

        if self.collect_diagnostics and hasattr(ss1, "log_chisq_diagnostics"):
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

        if self.out is not None and self.verbose_write_jack:
            jack_path = f"{self.out}.rg.jack"
            RGResultWriter.save_jackknife_text(rg_fit, jack_path)
            if self.log is not None:
                self.log._log(f"Saved rg jackknife replicate dump to {jack_path}")

        if self.out is not None and self.verbose_write_normeq:
            info = intercept.info if isinstance(intercept.info, dict) else {}
            n_overlap = info.get("n_overlap", None)
            if n_overlap is not None:
                eq_path = f"{self.out}.rg.scoreeq.json"
                RGResultWriter.save_score_normal_equations_json(rg_fit, eq_path)
                if self.log is not None:
                    self.log._log(f"Saved explicit full SCORE normal-equation dump to {eq_path}")

        if self.log is not None and self.verbose_level >= 1:
            ordered = [
                "load_align_filter",
                "materialize_jackknife",
                "h2_trait1",
                "h2_trait2",
                "external_intercept",
                "prepare_rg",
                "fit_intercept",
                "fit_rg",
            ]
            parts = [
                f"{name}={stage_times[name]:.3f}s"
                for name in ordered
                if name in stage_times
            ]
            if parts:
                self.log._log("[perf] " + ", ".join(parts))

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
            return float(self.intercept_rg), {"source": self.intercept_rg_source}

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


    def _apply_allele_alignment_filter(self, aligned1, aligned2, base_keep_mask):
        base_keep_mask = np.asarray(base_keep_mask, dtype=bool)
        idx = np.flatnonzero(base_keep_mask)
        if idx.size == 0:
            return base_keep_mask, None

        pos1 = aligned1.pos_on_trace[idx]
        pos2 = aligned2.pos_on_trace[idx]
        keep2, flip2 = harmonize_allele_codes(
            aligned1.sumstats.a1_code[pos1],
            aligned1.sumstats.a2_code[pos1],
            aligned2.sumstats.a1_code[pos2],
            aligned2.sumstats.a2_code[pos2],
            drop_ambiguous=self.drop_ambiguous,
        )
        out = base_keep_mask.copy()
        out[idx] = keep2
        flip_keep = flip2[keep2]

        if self.log is not None:
            n_drop = int(np.sum(~keep2))
            n_flip = int(np.sum(flip_keep))
            self.log._log(
                f"Allele alignment: checked {idx.size} shared SNPs; dropping {n_drop} "
                f"(drop_ambiguous={self.drop_ambiguous}); flipping trait 2 z/beta for "
                f"{n_flip} kept SNPs."
            )
        return out, flip_keep

    def _derive_cov_rank_overrides_from_pheno(self):
        """
        If --pheno-rg and --pheno-rg-cov are both provided, derive cov_rank
        (non-intercept df) from the external covariate designs and attach those
        values to the rg summary-moment path. h2 summary mode still forces
        cov_rank=0 inside moments.build_h2_summary_moment().
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
