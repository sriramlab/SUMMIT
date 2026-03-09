from __future__ import annotations

from dataclasses import replace
import os

import numpy as np

import utils
from jackknife import JackknifeSpec, JackknifeDesign
from trace import Trace
from sumstats import Sumstats, MatchedSumstats
from h2core import prepare_h2, fit_h2
from rgcore import prepare_rg, fit_intercept, fit_rg, RGResultWriter


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
        chisq_action: str = "drop",
        report_tau: bool = True,
        allow_neg_enr: bool = False,
        adjust_delta: bool = False,
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
        self.chisq_action = str(chisq_action).strip().lower()
        self.enrich_mode = enrich_mode
        self.jack_mode = jack_mode
        self.clip_nonfinite_vals = bool(clip_nonfinite_vals)
        self.report_tau = bool(report_tau)
        self.allow_neg_enr = bool(allow_neg_enr)
        self.adjust_delta = bool(adjust_delta)
        self.nan_policy = "propagate" if self.clip_nonfinite_vals else "omit"
        self.rg_se_method = str(rg_se_method).strip().lower()
        if self.rg_se_method not in {"jackknife", "delta"}:
            raise ValueError("rg_se_method must be one of {'jackknife','delta'}")
        if self.chisq_action not in ("drop", "clip", "warn", "none"):
            raise ValueError("chisq_action must be one of {'drop','clip','warn','none'}")

        self.out = out
        self.result = None
        self.trace_view = None
        self.jackknife = None
        self.matched1 = None
        self.matched2 = None

    def _run(self):
        ss1 = Sumstats.from_file(self.phen_paths[0], name=self.phen_names[0], log=self.log)
        ss2 = Sumstats.from_file(self.phen_paths[1], name=self.phen_names[1], log=self.log)
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

        intercept = fit_intercept(
            tv,
            matched1,
            matched2,
            jk,
            h2_fit1,
            h2_fit2,
            intercept_chisq_threshold=self.intercept_chisq_thr,
            collapse_reg_ld=self.collapse_reg_ld,
            log=self.log,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )

        rg_fit = fit_rg(
            prepare_rg(tv, matched1, matched2, jk, adjust_delta=self.adjust_delta),
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
            jack_path = f"{self.out}.{self.phen_names[0]}__{self.phen_names[1]}.rg.jack"
            RGResultWriter.save_jackknife_text(rg_fit, jack_path)
            if self.log is not None:
                self.log._log(f"Saved rg jackknife replicate dump to {jack_path}")

        return self.result

    def _logoff(self):
        if self.result is None:
            return

        h2_fit1 = self.result["h2_fit1"]
        h2_fit2 = self.result["h2_fit2"]
        intercept = self.result["intercept"]
        rg_fit = self.result["rg_fit"]
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
            f"Intercept (c): {intercept.c[0]:.6g} (SE: {intercept.c[1]:.6g})"
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
        a1 = matched.a1.copy()
        a2 = matched.a2.copy()
        tmp = a1[flip_mask].copy()
        a1[flip_mask] = a2[flip_mask]
        a2[flip_mask] = tmp
        return replace(matched, z=z, a1=a1, a2=a2)
