from __future__ import annotations

import gc
import os
from dataclasses import replace

import numpy as np

from .. import utils
from .jackknife import JackknifeSpec, JackknifeDesign
from .trace import Trace
from ..sumstats.sumstats import Sumstats
from .h2core import (
    H2MatchedMetadata,
    compute_h2_mass_unit_stats,
    compute_h2_structural_unit_stats,
    prepare_h2,
    prepare_h2_reference_axis,
    fit_h2,
)
from .ldsc_h2 import (
    prepare_h2_ldsc,
    read_ldsc_m,
    read_ldsc_weight_ld_aligned,
    resolve_ldsc_reference_moments,
)
from ..sumstats.moments import h2_moment_from_arrays, effective_n_scale


class Sumrhe:
    """
    Thin wrapper around the refactored univariate h2 core.

    Data flow per phenotype:
        Trace (immutable base) -> Sumstats (immutable read-QC) -> AlignedSumstats
        -> final keep mask
        -> compact TraceView + MatchedSumstats + JackknifeDesign for contiguous
           block jackknife, or reference-axis preparation for chromosome
           jackknife
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
        ldscores_w=None,
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
        write_jack: bool = False,
        weight_mode: str = "he",
        ldsc_m=None,
        ldsc_irwls_iters: int = 3,
        ldsc_irwls_tol: float = 0.0,
        snp_var_weights: bool = True,
    ):

        self.log = log
        self.verbose = verbose
        self.verbose_level = utils._parse_verbose(verbose)
        parsed_write_jack, _ = utils._parse_verbose_outputs(verbose)
        self.verbose_write_jack = bool(write_jack) or bool(parsed_write_jack)
        self.start_time = utils._get_time()
        if self.log is not None:
            self.log._log("Analysis started at: " + utils._get_timestr(self.start_time))

        if sum_path is not None:
            raise NotImplementedError(
                "Trace summaries are intentionally unsupported in this refactor for now. "
                "Use per-SNP LD-scores."
            )

        self.weight_mode = str(weight_mode).strip().lower().replace("-", "_")
        if self.weight_mode in {"summit", "score", "he_regression"}:
            self.weight_mode = "he"
        if self.weight_mode not in {"he", "ldsc"}:
            raise ValueError("weight_mode must be one of {'he','ldsc'}")
        if self.weight_mode == "he" and (ldscores_w is not None or ldsc_m is not None):
            raise ValueError("--ldscores-w and --ldsc-m require --weight-mode ldsc.")

        self.trace = Trace(
            bimpath=bim_path,
            sumpath=None,
            savepath=None,
            log=self.log,
            ldscores=ldscores,
            ldscores_reg_w=None,
            annot=annot,
            verbose=bool(self.verbose_level),
            delta=delta,
            snp_var_weights=snp_var_weights,
        )
        if self.trace.annot_mass is not None:
            unsupported = [name for name, on in (("--weight-mode ldsc", self.weight_mode == "ldsc"),
                                                 ("--adjust-delta", bool(adjust_delta))) if on]
            if unsupported:
                raise ValueError(
                    f"{' and '.join(unsupported)} cannot be used with SNP-variance-weighted LD scores (V column); "
                    "recompute the LD scores with --snp-var-weights-off."
                )
        self._rel_notes = set()

        self.jackknife_spec = JackknifeSpec.parse(njack)
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
            self.ldsc_m_source = reference.source if ldsc_m is None else f"{ldsc_m} ({reference.source})"
            if ldscores_w is not None:
                self.ldsc_weight_ld, self.ldsc_weight_present = read_ldsc_weight_ld_aligned(
                    ldscores_w,
                    self.trace.snps,
                    target_chr=self.trace.chr,
                    target_bp=self.trace.bp,
                )
                n_missing = int(np.sum(~self.ldsc_weight_present))
                if self.log is not None:
                    self.log._log(
                        f"[h2:ldsc] aligned scalar weight LD to the primary Trace; "
                        f"{n_missing} SNP(s) will be excluded from the regression axis."
                    )

        if self.jackknife_spec.mode == "chr" and self.weight_mode == "he":
            self.full_trace_view = self.trace.materialize_view()
            self.full_jackknife = JackknifeDesign.from_trace_view(
                self.full_trace_view,
                self.jackknife_spec,
                log=self.log,
            )
        else:
            self.full_trace_view = None
            self.full_jackknife = None
        self.full_struct = None
        self.full_mass = None
        self.full_trace_view_rel = None
        self.full_struct_rel = None
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

        self._ensure_reference_precompute()

        for i, (path, phen_name, cov_rank_value) in enumerate(zip(
            self.h2_paths,
            self.phen_names,
            self.cov_rank_values,
        )):
            fit = self._fit_one(path, phen_name, cov_rank_value)
            self.nsamp.append(float(fit.prepared.matched.nsamp))
            if self.npheno > 1:
                # Directory/batch h2 mode should stream phenotypes. H2Fit.prepared
                # holds the heavy per-SNP TraceView/MatchedSumstats/H2Prepared
                # arrays for the completed phenotype; keeping it would make RSS
                # grow roughly linearly with the number of traits.
                object.__setattr__(fit, "prepared", None)
            self.results.append(fit)
            if self.log is not None:
                self.log._log(
                    f"[h2] completed phenotype {i + 1}/{self.npheno} {phen_name}: "
                    f"h2={fit.h2[-1, 0]:.6g} SE={fit.h2[-1, 1]:.6g}"
                )
            self._write_results_table()
            if self.npheno > 1:
                gc.collect()

        return self.results

    def _ensure_reference_precompute(self):
        if self.jackknife_spec.mode != "chr" or self.weight_mode != "he":
            return
        if self.full_struct is not None:
            return
        t0 = utils._get_time()
        self.full_struct = compute_h2_structural_unit_stats(
            self.full_trace_view,
            self.full_jackknife,
            ld_kind="main",
        )
        if self.full_trace_view.annot_mass is not None:
            self.full_mass = compute_h2_mass_unit_stats(self.full_trace_view.annot_mass, self.full_jackknife)
        if self.log is not None:
            self.log._log(
                f"[h2] precomputed full reference-axis structural stats in "
                f"{utils._get_time() - t0:.3f}s."
            )

    def _reference_axis_for(self, exclude_pairs: bool):
        """Full reference-axis view and structural stats, with LD_REL subtracted when related pairs are excluded."""
        self._ensure_reference_precompute()
        if not exclude_pairs:
            return self.full_trace_view, self.full_struct
        if self.full_struct_rel is None:
            self.full_trace_view_rel = self.full_trace_view.pair_excluded()
            self.full_struct_rel = compute_h2_structural_unit_stats(
                self.full_trace_view_rel,
                self.full_jackknife,
                ld_kind="main",
            )
        return self.full_trace_view_rel, self.full_struct_rel

    def _exclude_related_pairs(self, ss) -> bool:
        """Subtract LD_REL for sumstats with REL_M; LD scores without LD_REL are taken to be from unrelated samples."""
        has_rel_m = getattr(ss, "rel_m", None) is not None
        has_ld_rel = self.trace.ldscores_rel is not None
        note = None
        if has_rel_m and has_ld_rel:
            note = "[relatives] subtracting the LD_REL columns from the LD scores for sumstats with REL_M."
        elif has_rel_m and not has_ld_rel:
            note = ("[relatives] sumstats have REL_M but the LD-score file has no LD_REL columns: the LD scores are "
                    "taken to come from unrelated individuals.")
        elif has_ld_rel and not has_rel_m:
            note = ("[relatives] sumstats without REL_M keep the related pairs, so the LD_REL columns are not "
                    "subtracted for them.")
        if note is not None and note not in self._rel_notes and self.log is not None:
            self._rel_notes.add(note)
            self.log._log(note)
        return has_rel_m and has_ld_rel

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
        if self.ldsc_weight_present is not None:
            keep_mask &= self.ldsc_weight_present

        n_keep = int(np.sum(keep_mask))
        n_base = int(self.trace.nsnps)
        if self.log is not None:
            self.log._log(
                f"Matched {n_keep} SNPs in phenotype {phen_name} out of {n_base} Trace SNPs."
            )

        exclude_pairs = self._exclude_related_pairs(ss)
        if self.jackknife_spec.mode == "chr" and self.weight_mode == "he":
            ref_view, ref_struct = self._reference_axis_for(exclude_pairs)
            summary_y, summary_y_info = self._build_full_axis_h2_summary_y(
                ss,
                aligned,
                keep_mask,
            )
            matched = self._build_h2_matched_metadata(ss, aligned, keep_mask)

            prepared = prepare_h2_reference_axis(
                ref_view,
                matched,
                self.full_jackknife,
                keep_mask,
                summary_y=summary_y,
                summary_y_info=summary_y_info,
                full_struct=ref_struct,
                adjust_delta=self.adjust_delta,
                full_mass=self.full_mass,
            )
        else:
            tv = self.trace.materialize_view(keep_mask)
            if exclude_pairs:
                tv = tv.pair_excluded()
            if self.ldsc_weight_ld is not None:
                tv = replace(
                    tv,
                    ldscores_reg_w=np.asarray(
                        self.ldsc_weight_ld[keep_mask, :],
                        dtype=np.float64,
                        order="C",
                    ),
                )
            matched = aligned.materialize(
                keep_mask,
                chisq_threshold=self.chisq_threshold,
                chisq_action=self.chisq_action,
                allowed_mask=keep_mask,
                compute_diagnostics=True,
            )
            jk = JackknifeDesign.from_trace_view(tv, self.jackknife_spec, log=self.log)
            if self.weight_mode == "ldsc":
                prepared = prepare_h2_ldsc(tv, matched, jk)
            else:
                prepared = prepare_h2(
                    tv,
                    matched,
                    jk,
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
            weight_mode=self.weight_mode,
            ldsc_m_annot=self.ldsc_m_annot,
            ldsc_overlap_matrix=self.ldsc_overlap_matrix,
            ldsc_source_nsnps=self.ldsc_source_nsnps,
            ldsc_irwls_iters=self.ldsc_irwls_iters,
            ldsc_irwls_tol=self.ldsc_irwls_tol,
        )

        if self.log is not None and self.weight_mode == "ldsc":
            info = fit.weight_info or {}
            self.log._log(
                f"[h2:ldsc] constrained score-scale LDSC IRWLS using M source "
                f"'{self.ldsc_m_source}', "
                f"weight LD source '{info.get('weight_ld_source', 'unknown')}', "
                f"iterations={info.get('irwls_iters', 'NA')}."
            )

        if hasattr(ss, "log_chisq_diagnostics"):
            ss.log_chisq_diagnostics(
                matched,
                chisq_threshold=self.chisq_threshold,
                verbose=(self.verbose_level >= 1),
            )

        if self.verbose_write_jack and self.out is not None:
            jack_path = f"{self.out}.{phen_name}.jack"
            from .h2core import H2ResultWriter
            H2ResultWriter.save_jackknife_text(fit, jack_path)
            if self.log is not None:
                self.log._log(f"Saved jackknife replicate dump to {jack_path}")

        return fit

    def _build_full_axis_h2_summary_y(self, ss, aligned, keep_mask):
        keep_mask = np.asarray(keep_mask, dtype=bool)
        M = int(self.trace.nsnps)
        if keep_mask.ndim != 1 or keep_mask.size != M:
            raise ValueError(f"keep_mask must be length {M}; got {keep_mask.shape}")

        pos = np.asarray(aligned.pos_on_trace[keep_mask], dtype=np.int64)
        if np.any(pos < 0):
            raise ValueError("keep_mask includes SNPs that are absent from the sumstats.")

        y_used = h2_moment_from_arrays(
            beta=ss.beta[pos],
            se=ss.se[pos],
            n_obs=ss.n[pos],
            nsamp=float(ss.nsamp),
            cov_rank=0,
        )
        rel_info = None
        if ss.rel_m is not None:
            # Exclude related pairs from the HE numerator (summit.relatives).
            rel_used = ss.rel_m[pos]
            y_used = y_used - rel_used
            rel_info = {"c_rel": float(np.nanmean(rel_used)), "sum_rel_m": float(np.nansum(rel_used))}
            if self.log is not None:
                self.log._log(
                    f"[relatives] [{ss.name}] subtracting related-pair HE moments: "
                    f"relative-pair covariance in HE scale c_rel={rel_info['c_rel']:.6g}."
                )
        y_used[~np.isfinite(y_used)] = np.nan

        y = np.full(M, np.nan, dtype=np.float64)
        y[keep_mask] = y_used

        info = {
            "mode": "beta_se_exact",
            "cov_rank": 0,
            "cov_rank_source": "forced0_no_covrank_h2",
            "n_scale": float(effective_n_scale(ss.nsamp, 0)),
            "n_nonfinite": int(np.sum(~np.isfinite(y_used))),
        }
        if rel_info is not None:
            info["related_pairs"] = rel_info
        return y, info

    def _build_h2_matched_metadata(self, ss, aligned, keep_mask):
        used_summary, used_top, clip_count, clip_threshold = aligned.diagnostics_for_keep(
            keep_mask,
            chisq_threshold=self.chisq_threshold,
            chisq_action=self.chisq_action,
            compute_diagnostics=True,
        )
        return H2MatchedMetadata(
            nsnps=int(np.sum(keep_mask)),
            nsamp=float(ss.nsamp),
            n_scale=float(ss.n_scale),
            cov_rank=int(ss.cov_rank),
            cov_rank_source=str(ss.cov_rank_source),
            name=str(ss.name),
            used_summary=used_summary,
            used_top=used_top,
            clip_count=clip_count,
            clip_threshold=clip_threshold,
        )

    def _write_results_table(self):
        if self.out is None or not self.results:
            return

        path = f"{self.out}.results.tsv"
        tmp = f"{path}.{os.getpid()}.tmp"
        k = int(self.nbins)

        header = ["phen_index", "phen", "num_bins", "h2", "h2_se"]
        header.extend([f"h2bin_{j}" for j in range(k)])
        header.extend([f"h2bin_se_{j}" for j in range(k)])
        header.extend([f"enrichment_{j}" for j in range(k)])
        header.extend([f"enrichment_se_{j}" for j in range(k)])
        header.append("enrichment_mode")
        header.append("estimator")

        with open(tmp, "w") as fout:
            fout.write("\t".join(header) + "\n")
            for i, fit in enumerate(self.results):
                h2tot = fit.h2[-1, 0]
                h2totse = fit.h2[-1, 1]
                row = [
                    str(i),
                    str(self.phen_names[i] if i < len(self.phen_names) else i),
                    str(k),
                    format(float(h2tot), ".12g"),
                    format(float(h2totse), ".12g"),
                ]
                row.extend(format(float(fit.h2[j, 0]), ".12g") for j in range(k))
                row.extend(format(float(fit.h2[j, 1]), ".12g") for j in range(k))
                row.extend(format(float(fit.enrich[j, 0]), ".12g") for j in range(k))
                row.extend(format(float(fit.enrich[j, 1]), ".12g") for j in range(k))
                row.append(str(fit.enrich_mode_used))
                row.append(str((fit.weight_info or {}).get("estimator", "he")))
                fout.write("\t".join(row) + "\n")

        os.replace(tmp, path)
        if self.log is not None:
            self.log._log(f"Saved h2 results table in {path}")

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

        self._write_results_table()

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
