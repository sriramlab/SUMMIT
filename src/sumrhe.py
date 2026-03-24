from __future__ import annotations

import os

import numpy as np

import utils
from jackknife import JackknifeSpec, JackknifeDesign
from trace import Trace
from sumstats import Sumstats
from h2core import prepare_h2, fit_h2
from moments import build_h2_summary_moment


class Sumrhe:
    """
    Thin wrapper around the refactored univariate h2 core.

    Data flow per phenotype:
        Trace (immutable base) -> Sumstats (immutable read-QC) -> AlignedSumstats
        -> final keep mask -> TraceView + MatchedSumstats -> JackknifeDesign
        -> H2Prepared -> H2Fit
    """

    def __init__(
        self,
        bim_path=None,
        sum_path=None,
        h2_path=None,
        out=None,
        chisq_threshold=0,
        log=None,
        verbose=False,
        ldscores=None,
        njack=None,
        annot=None,
        chisq_action="drop",
        report_tau: bool = True,
        allow_neg_enr: bool = False,
        clip_nonfinite_vals: bool = False,
        adjust_delta: bool = False,
        enrich_mode: str = "auto",
        jack_mode: str = "mean",
        delta=None,
        cov_rank=None,
    ):

        self.log = log
        self.verbose = verbose
        self.verbose_level = utils._parse_verbose(verbose)
        self.start_time = utils._get_time()
        if self.log is not None:
            self.log._log("Analysis started at: " + utils._get_timestr(self.start_time))

        if sum_path is not None:
            raise NotImplementedError(
                "Trace summaries are intentionally unsupported in this refactor for now. "
                "Use per-SNP LD-scores."
            )

        self.trace = Trace(
            bimpath=bim_path,
            sumpath=None,
            savepath=None,
            log=self.log,
            ldscores=ldscores,
            annot=annot,
            verbose=bool(self.verbose_level),
            delta=delta,
        )

        self.jackknife_spec = JackknifeSpec.parse(njack)
        self.chisq_threshold = chisq_threshold
        self.chisq_action = str(chisq_action).strip().lower()
        self.report_tau = bool(report_tau)
        self.allow_neg_enr = bool(allow_neg_enr)
        self.clip_nonfinite_vals = bool(clip_nonfinite_vals)
        self.adjust_delta = bool(adjust_delta)
        self.enrich_mode = enrich_mode
        self.jack_mode = jack_mode
        self.nan_policy = "propagate" if self.clip_nonfinite_vals else "omit"

        self.h2_paths = utils._parse_sumdir(h2_path)
        self.phen_names = [utils._phen_name_from_path(p) for p in self.h2_paths]
        self.npheno = len(self.h2_paths)
        self.cov_rank_values = self._parse_cov_rank_values(cov_rank, self.npheno)

        self.results = []
        self.nsamp = []

        self.out = out
        self.annot_header = self.trace.annot_header
        self.nbins = self.trace.nbins

    def _run(self):
        self.results = []
        self.nsamp = []

        for path, phen_name, cov_rank_value in zip(
            self.h2_paths,
            self.phen_names,
            self.cov_rank_values,
        ):
            fit = self._fit_one(path, phen_name, cov_rank_value)
            self.results.append(fit)
            self.nsamp.append(float(fit.prepared.matched.nsamp))

        return self.results

    def _fit_one(self, path: str, phen_name: str, cov_rank_value):
        ss = Sumstats.from_file(
            path,
            name=phen_name,
            log=self.log,
            cov_rank=cov_rank_value,
            cov_rank_source=("cli" if cov_rank_value is not None else None),
        )
        aligned = ss.align_to_trace(self.trace)
        keep_mask = aligned.keep_mask(
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
        )

        n_keep = int(np.sum(keep_mask))
        n_base = int(self.trace.nsnps)
        if self.log is not None:
            self.log._log(
                f"Matched {n_keep} SNPs in phenotype {phen_name} out of {n_base} Trace SNPs."
            )

        tv = self.trace.materialize_view(keep_mask)
        matched = aligned.materialize(
            keep_mask,
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
        )
        jk = JackknifeDesign.from_trace_view(tv, self.jackknife_spec, log=self.log)

        prepared = prepare_h2(
            tv,
            matched,
            jk,
            active_mask=None,
            ld_kind="main",
            adjust_delta=self.adjust_delta,
        )
        fit = fit_h2(
            prepared,
            enrich_mode=self.enrich_mode,
            report_tau=self.report_tau,
            allow_neg_enr=self.allow_neg_enr,
            clip_nonfinite_vals=self.clip_nonfinite_vals,
            jack_mode=self.jack_mode,
            nan_policy=self.nan_policy,
        )

        if hasattr(ss, "log_chisq_diagnostics"):
            ss.log_chisq_diagnostics(
                matched,
                chisq_threshold=self.chisq_threshold,
                verbose=(self.verbose_level >= 1),
            )

        if self.verbose_level >= 2 and self.out is not None:
            jack_path = f"{self.out}.{phen_name}.jack"
            from h2core import H2ResultWriter
            H2ResultWriter.save_jackknife_text(fit, jack_path)
            if self.log is not None:
                self.log._log(f"Saved jackknife replicate dump to {jack_path}")

        return fit

    def _logoff(self):
        for i, fit in enumerate(self.results):
            if fit.enrich_mode_used:
                self.log._log(
                    f"^^^ Phenotype {i} enrichment_mode_used: {fit.enrich_mode_used}"
                )

            if self.nbins > 1:
                for j in range(self.nbins):
                    sig = fit.sigmas[j, 0]
                    sigse = fit.sigmas[j, 1]
                    h2 = fit.h2[j, 0]
                    h2se = fit.h2[j, 1]
                    enr = fit.enrich[j, 0]
                    ense = fit.enrich[j, 1]

                    enr_ov = ense_ov = None
                    enr_no = ense_no = None
                    if fit.enrich_overlap is not None:
                        enr_ov = fit.enrich_overlap[j, 0]
                        ense_ov = fit.enrich_overlap[j, 1]
                    if fit.enrich_nonoverlap is not None:
                        enr_no = fit.enrich_nonoverlap[j, 0]
                        ense_no = fit.enrich_nonoverlap[j, 1]

                    line = (
                        f"^^^ Phenotype {i} Bin [{self.annot_header[j]}] "
                        f"sigma_g^2: {sig:.5f} (SE: {sigse:.5f}) "
                        f"h^2_cat: {h2:.5f} (SE: {h2se:.5f}) "
                        f"Enrichment: {enr:.5f} (SE: {ense:.5f})"
                    )

                    if (enr_no is not None) and (enr_ov is not None):
                        line += (
                            f" Enrichment_nonoverlap: {enr_no:.5f} (SE: {ense_no:.5f})"
                            f" Enrichment_overlap: {enr_ov:.5f} (SE: {ense_ov:.5f})"
                        )

                    if self.report_tau and fit.tau is not None and fit.tau_star is not None:
                        t = fit.tau[j, 0]
                        tse = fit.tau[j, 1]
                        ts = fit.tau_star[j, 0]
                        tsse = fit.tau_star[j, 1]
                        line += f" tau: {t:.6g} (SE: {tse:.6g}) tau_*: {ts:.6g} (SE: {tsse:.6g})"
                    self.log._log(line)

            h2tot = fit.h2[-1, 0]
            h2totse = fit.h2[-1, 1]
            self.log._log(
                "^^^ Phenotype "
                + str(i)
                + " Total SNP heritability (h^2): "
                + format(h2tot, ".5f")
                + " SE: "
                + format(h2totse, ".5f")
            )

        self.end_time = utils._get_time()
        self.log._log("Analysis ended at: " + utils._get_timestr(self.end_time))
        self.log._log("run time: " + format(self.end_time - self.start_time, ".3f") + " s")
        if self.out is not None:
            self.log._log("Saved log in " + self.out + ".log")
            self.log._save_log(self.out + ".log")
        return


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

        if len(vals) != expected:
            raise ValueError(
                f"--cov-rank must contain exactly {expected} value(s) for --h2; got {len(vals)}"
            )
        return vals

