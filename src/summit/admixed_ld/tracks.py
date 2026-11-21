"""Correlation of per-allele causal effects across local-ancestry tracks from summary statistics
(`summit admixed-ld tractor-sumstats` and `summit admixed-ld track-rg`).

A Tractor-style GWAS of an admixed cohort regresses the phenotype, SNP by SNP, on the covariates, the local-ancestry
dosages and the allele counts on each ancestry's haplotypes. With the local-ancestry term, the regression variable of
track s at SNP j is the within-track allele deviation u^s_j = g^s_j - p^s_j l^s_j, which is uncorrelated with local
ancestry and with every other track's alleles, so the moments of the track z statistics are
    E[(z^s_j)^2] - 1 = N sum_c tau_s,c l^ss_jc,      E[z^s_j z^t_j] = N sum_c tau_st,c l^st_jc
with per-allele effects Var(b^s_k) = tau_s,c w_k and Cov(b^s_k, b^t_k) = tau_st,c w_k for partner variants k in class
c, the short-range track LD scores l of `compose --tracks`, and a cross intercept of exactly zero (at one locus the two
tracks' alleles lie on different haplotypes). These are SUMMIT's cross-population equations with the overlapping
covariance fixed at 0, so each pair of tracks is fitted by Sumcore in cross mode. The correlation across tracks (Hou
et al. 2023's r_admix, per-allele scale) is
    r_st = sum_c tau_st,c n_c / sqrt(sum_c tau_s,c n_c  sum_c tau_t,c n_c),
n_c the number of partner variants in class c (compose's .gw.M): the per-bin variances absorb the dependence of the
effect variance on frequency and LD across bins, and n_c weights each per-allele variance by 1 / w_k. With the GWAS
SNPs as partners this is SUMMIT's own rg_total; with a dense reference as partners the masses are the reference's.

Ordinary (OLS) standard errors assume residuals that do not vary between people, but the track variable's variance
follows each person's local ancestry, so the own moments of OLS z statistics carry first-order same-person terms
(including the other track's heritability). Sandwich standard errors remove them, so the reader requires them and the
fit refuses OLS input unless asked. The sandwich must take each person's variance from the residuals of the
covariates-only model, e0_i^2 / (1 - h_ii) with h_ii the hat value under the covariates (columns se_hcn_anc<s>): then
z^2 - 1 is the moment over distinct people that the track LD scores model, at any MAF and with heteroscedastic noise.
The usual HC0-HC3 sandwich takes the residuals of the full model, which the fit shrinks at alleles carried by few
people on one track, so its SEs are biased low there; it is accepted with a recorded flag.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

TRACK_COLUMNS = ["SNP", "CHR", "BP", "A1", "A2", "N", "BETA", "SE", "Z", "COV_RANK", "AF", "LA_PROP", "SE_TYPE",
                 "TRACK"]


class _Log:
    """Adapter of a print-like function to the `_log` interface that Sumcore uses."""

    def __init__(self, fn):
        self.fn = fn

    def _log(self, msg):
        if self.fn is not None:
            self.fn(msg)


def _col(cols, names, anc=None):
    """The first column of `cols` whose name matches one of `names` (case-insensitive; '{s}' is the ancestry index)."""
    low = {c.lower(): c for c in cols}
    for n in names:
        key = (n.format(s=anc) if anc is not None else n).lower()
        if key in low:
            return low[key]
    return None


def _hc_column(cols, anc, kind="se", null=False):
    """The sandwich SE (kind 'se') or t statistic ('t') column of ancestry `anc`. null=True: the sandwich with the
    covariates-only residuals, se_hcn_anc0 or sehcn_anc0 (tval_hcn_anc0, thcn_anc0); otherwise a full-model sandwich,
    se_hc_anc0, sehc_anc0, se_hc1_anc0, se_robust_anc0 (any case), and tval_hc_anc0, thc_anc0 for the t statistic."""
    if null:
        pat = (r"^se_?hcn_anc%d$" if kind == "se" else r"^(tval_?hcn|t_?hcn)_anc%d$") % int(anc)
    else:
        pat = (r"^(se_?hc[0-3]?|se_?robust|se_?sandwich)_anc%d$" if kind == "se"
               else r"^(tval_?hc[0-3]?|t_?hc[0-3]?|tval_?robust)_anc%d$") % int(anc)
    hits = [c for c in cols if re.match(pat, str(c), flags=re.IGNORECASE)]
    if len(hits) > 1:
        raise ValueError(f"Several sandwich {kind.upper()} columns for ancestry {anc}: {', '.join(hits)}.")
    return hits[0] if hits else None


def tractor_sumstats(path, out, *, sources=None, se="hc", allow_ols=False, n_covariates=None, log=print):
    """Split a Tractor output table into one SUMMIT summary-statistic file per local-ancestry track.

    Columns read (Tractor 1.4 names, any case): CHR, POS, ID, REF, ALT, N and, per ancestry index s (anc0, anc1, ...),
    beta_anc{s}, se_anc{s} and/or tval_anc{s}, AF_anc{s} (the track's allele frequency), LAprop_anc{s} (the
    local-ancestry share at the SNP), and the sandwich SEs: se_hcn_anc{s} (or sehcn_; optionally tval_hcn_anc{s}),
    whose meat takes each person's variance from the covariates-only residuals, or else a full-model sandwich
    se_hc_anc{s} (sehc_, se_hc1_, se_robust_; tval_hc_anc{s}), with a recorded flag. Tractor counts the ALT allele,
    which becomes A1. Optional: COV_RANK (covariates plus the
    local-ancestry and other-track terms); otherwise n_covariates (the user's covariates) gives
    COV_RANK = n_covariates + 2 (K - 1), the other regressors of each track's partial regression.

    se='hc' (the default) needs the sandwich SEs of every track; se='ols' uses Tractor's OLS SEs (beta / tval when
    tval is present, since se is rounded to four decimals) and needs allow_ols, because the own moments of OLS z
    statistics hold same-person terms that the track LD scores do not model. Writes <out>.<source>.sumstats per
    track (TRACK and SE_TYPE columns mark them) and <out>.tracks.json (mean local-ancestry share per track for
    compose's --mean-ancestry, N, the SE type, flags and the residual-variance check). Returns that record."""
    if se not in ("hc", "ols"):
        raise ValueError("se must be 'hc' (sandwich SEs, required for the track moments) or 'ols'.")
    if se == "ols" and not allow_ols:
        raise ValueError("OLS standard errors of a Tractor GWAS leave first-order same-person terms in the own-track "
                         "moments (the track variable's variance follows each person's local ancestry), which the "
                         "track LD scores do not model; ask the cohort for sandwich SEs from the covariates-only "
                         "residuals (se_hcn_anc<s>). Pass --allow-ols to proceed with the flag recorded.")
    df = pd.read_csv(path, sep=r"\s+", compression="infer", dtype={"ID": str, "REF": str, "ALT": str})
    cols = list(df.columns)
    anc = sorted(int(m.group(1)) for c in cols for m in [re.match(r"^beta_anc(\d+)$", str(c), flags=re.IGNORECASE)]
                 if m)
    if len(anc) < 2 or anc != list(range(len(anc))):
        raise ValueError(f"{path}: expected beta_anc0, beta_anc1, ... (Tractor's columns); found {anc}.")
    K = len(anc)
    sources = list(sources) if sources is not None else [f"anc{s}" for s in anc]
    if len(sources) != K or len(set(sources)) != K:
        raise ValueError(f"Give {K} distinct source names (one per ancestry index of {path}).")
    need = {k: _col(cols, alts) for k, alts in dict(CHR=("CHR", "CHROM", "#CHROM"), BP=("POS", "BP"),
                                                     SNP=("ID", "SNP", "RSID"), A2=("REF", "A2"), A1=("ALT", "A1"),
                                                     N=("N", "OBS_CT")).items()}
    miss = [k for k, v in need.items() if v is None]
    if miss:
        raise ValueError(f"{path} lacks {', '.join(miss)} (CHR, POS, ID, REF, ALT, N).")
    rank_col = _col(cols, ("COV_RANK",))
    flags = []
    if rank_col is not None:
        rank = pd.to_numeric(df[rank_col], errors="coerce").to_numpy()
    else:
        nc = 0 if n_covariates is None else int(n_covariates)
        rank = np.full(len(df), nc + 2 * (K - 1), dtype=np.float64)
        if n_covariates is None:
            flags.append(dict(flag="cov_rank_without_covariates",
                              detail=f"COV_RANK = 2 (K - 1) = {2 * (K - 1)}: the number of the GWAS's covariates was "
                                     "not given (it changes z* by about rank / N)"))
    base = pd.DataFrame({"SNP": df[need["SNP"]].astype(str),
                         "CHR": df[need["CHR"]].astype(str).str.replace(r"^(?i:chr)", "", regex=True),
                         "BP": pd.to_numeric(df[need["BP"]], errors="coerce"),
                         "A1": df[need["A1"]].astype(str).str.upper(), "A2": df[need["A2"]].astype(str).str.upper(),
                         "N": pd.to_numeric(df[need["N"]], errors="coerce")})
    if (base.A1 == base.A2).any():
        raise ValueError(f"{path}: REF and ALT are the same allele at {int((base.A1 == base.A2).sum())} SNPs.")
    tables, rec = {}, dict(source_file=os.path.abspath(path), sources=sources, se_type=None, tracks={}, flags=flags,
                           files={}, definition="per track s: SNP, CHR, BP, A1 = ALT (the counted allele), A2 = REF, "
                                                "N, BETA, SE (sandwich unless SE_TYPE says OLS), Z = BETA / SE, "
                                                "COV_RANK, AF (the track's allele frequency), LA_PROP (the "
                                                "local-ancestry share), TRACK")
    se_type = "HC" if se == "hc" else "OLS"
    full_model = []                                            # full-model sandwich columns used
    la_sum = np.zeros(len(df))
    for s, name in zip(anc, sources):
        b = _col(cols, ("beta_anc{s}",), s)
        af = _col(cols, ("AF_anc{s}", "freq_anc{s}"), s)
        la = _col(cols, ("LAprop_anc{s}", "LA_prop_anc{s}", "laprop_anc{s}"), s)
        if af is None or la is None:
            raise ValueError(f"{path} lacks AF_anc{s} or LAprop_anc{s} (the track's frequency and local-ancestry share).")
        beta = pd.to_numeric(df[b], errors="coerce").to_numpy(np.float64)
        if se == "hc":
            sc, tc = _hc_column(cols, s, "se", null=True), _hc_column(cols, s, "t", null=True)
            if sc is None and tc is None:
                sc, tc = _hc_column(cols, s, "se"), _hc_column(cols, s, "t")
                if sc is not None or tc is not None:
                    full_model.append(sc or tc)
            if sc is None and tc is None:
                raise ValueError(f"{path} has no sandwich SE column for ancestry {s} (se_hcn_anc{s}; Tractor itself "
                                 "writes only OLS SEs). Add sandwich SEs whose meat takes each person's variance from "
                                 "the residuals of the covariates-only model, e0^2 / (1 - h) (see the documentation), or "
                                 "pass --se ols --allow-ols to proceed with the flag recorded.")
            sev = (pd.to_numeric(df[sc], errors="coerce").to_numpy(np.float64) if sc is not None
                   else np.full(len(df), np.nan))
            if tc is not None:                                 # beta / t keeps the precision rounding takes away
                t = pd.to_numeric(df[tc], errors="coerce").to_numpy(np.float64)
                sev = np.where(np.isfinite(t) & (t != 0), beta / np.where(t != 0, t, 1.0), sev)
        else:
            sc, tc = _col(cols, ("se_anc{s}",), s), _col(cols, ("tval_anc{s}",), s)
            if sc is None and tc is None:
                raise ValueError(f"{path} lacks se_anc{s} and tval_anc{s}.")
            sev = (pd.to_numeric(df[sc], errors="coerce").to_numpy(np.float64) if sc is not None
                   else np.full(len(df), np.nan))
            if tc is not None:
                t = pd.to_numeric(df[tc], errors="coerce").to_numpy(np.float64)
                sev = np.where(np.isfinite(t) & (t != 0), beta / np.where(t != 0, t, 1.0), sev)
            elif sc is not None:
                ok_ = np.isfinite(sev) & (sev > 0)
                if ok_.any() and np.allclose(np.round(sev[ok_], 4), sev[ok_], rtol=0, atol=1e-12):
                    flags.append(dict(flag="se_rounded", detail=f"se_anc{s} has four decimals and no tval_anc{s}"))
        sev = np.abs(sev)
        laprop = pd.to_numeric(df[la], errors="coerce").to_numpy(np.float64)
        freq = pd.to_numeric(df[af], errors="coerce").to_numpy(np.float64)
        la_sum += np.nan_to_num(laprop)
        t = base.assign(BETA=beta, SE=sev, Z=beta / np.where(sev > 0, sev, np.nan), COV_RANK=rank, AF=freq,
                        LA_PROP=laprop, SE_TYPE=se_type, TRACK=name)
        good = (np.isfinite(t.BETA) & np.isfinite(t.SE) & (t.SE > 0) & np.isfinite(t.N) & (t.N > 0)
                & np.isfinite(t.BP) & np.isfinite(t.COV_RANK))
        t = t[good].copy()
        t["BP"] = t.BP.astype(np.int64)
        t["COV_RANK"] = t.COV_RANK.astype(np.int64)
        tables[name] = t
        # the track variable's variance 2 LAprop AF (1 - AF) against 1 / (N SE^2): their product estimates the noise
        # variance (the residual variance for OLS SEs, each track's effective noise for sandwich SEs)
        q = 2.0 * t.LA_PROP.to_numpy() * t.AF.to_numpy() * (1.0 - t.AF.to_numpy())
        sig = t.N.to_numpy() * t.SE.to_numpy() ** 2 * q
        sig = sig[np.isfinite(sig) & (q > 0)]
        rec["tracks"][name] = dict(index=int(s), n_snps=int(len(t)), mean_local_ancestry=float(np.nanmean(laprop)),
                                   median_n=float(np.median(t.N)) if len(t) else None,
                                   noise_variance_median=float(np.median(sig)) if sig.size else None,
                                   noise_variance_iqr=([float(np.quantile(sig, 0.25)), float(np.quantile(sig, 0.75))]
                                                       if sig.size else None))
    rec["se_type"] = se_type
    if full_model:
        flags.append(dict(flag="sandwich_from_full_model_residuals",
                          detail=f"{', '.join(full_model)}: the meat uses the full model's residuals, which are shrunk "
                                 "at alleles rare on a track, so these SEs are biased low there"))
        log(f"[admixed-ld] WARNING (recorded flag sandwich_from_full_model_residuals): {', '.join(full_model)} take "
            "each person's variance from the full model's residuals; at alleles rare on a track they are biased low. "
            "Prefer se_hcn_anc<s>, the sandwich with the covariates-only residuals e0^2 / (1 - h).")
    if se_type == "OLS":
        flags.append(dict(flag="ols_standard_errors",
                          detail="OLS SEs: the own-track moments hold same-person terms that the track LD scores do "
                                 "not model (first order; larger in small cohorts and when ancestry varies widely)"))
        log("[admixed-ld] WARNING (recorded flag ols_standard_errors): OLS SEs of a Tractor GWAS bias the own-track "
            "moments; use sandwich SEs where possible")
    dev = np.nanmax(np.abs(la_sum - 1.0)) if len(df) else 0.0
    if dev > 0.02:
        flags.append(dict(flag="local_ancestry_shares_do_not_sum_to_one", detail=f"largest deviation {dev:.3g}"))
    m = np.array([rec["tracks"][nm]["mean_local_ancestry"] for nm in sources])
    rec["mean_ancestry"] = (m / m.sum()).tolist()
    rec["mean_ancestry_arg"] = ",".join(f"{x:.6g}" for x in m / m.sum())
    for name, t in tables.items():
        f = f"{out}.{name}.sumstats"
        t[TRACK_COLUMNS].to_csv(f, sep="\t", index=False, float_format="%.10g")
        rec["files"][name] = f
    with open(f"{out}.tracks.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    noise = {nm: rec["tracks"][nm]["noise_variance_median"] for nm in sources}
    log(f"[admixed-ld] tractor-sumstats: {len(df)} SNPs, {K} tracks ({', '.join(sources)}), {se_type} SEs; mean "
        "local-ancestry shares " + ", ".join(f"{nm} {x:.4f}" for nm, x in zip(sources, m / m.sum()))
        + " (compose --mean-ancestry " + rec["mean_ancestry_arg"] + "); noise variance N SE^2 2 LAprop AF (1 - AF), "
        "median " + ", ".join(f"{nm} {v:.4g}" if v is not None else f"{nm} NA" for nm, v in noise.items()))
    return rec


def _se_type(path):
    head = pd.read_csv(path, sep=r"\s+", nrows=5, compression="infer")
    if "SE_TYPE" not in head.columns:
        return None
    vals = set(head["SE_TYPE"].astype(str).str.upper())
    return "OLS" if "OLS" in vals else ("HC" if vals <= {"HC", "HC1", "HC0", "HC2", "HC3"} else ",".join(sorted(vals)))


def _track_files(sumstats, sources):
    """Per-track summary-statistic files in source order: a tractor-sumstats prefix (with PREFIX.tracks.json) or a
    list of files (in the order of compose's sources)."""
    if isinstance(sumstats, str) and os.path.exists(f"{sumstats}.tracks.json"):
        with open(f"{sumstats}.tracks.json") as fh:
            rec = json.load(fh)
        missing = [s for s in sources if s not in rec["files"]]
        if missing:
            raise ValueError(f"{sumstats}.tracks.json has no track {', '.join(missing)} (it has "
                             f"{', '.join(rec['files'])}); name the tracks as compose's sources.")
        return [rec["files"][s] for s in sources], rec
    files = list(sumstats) if not isinstance(sumstats, str) else [x for x in sumstats.split(",") if x]
    if len(files) != len(sources):
        raise ValueError(f"Give one summary-statistic file per track ({len(sources)}: {', '.join(sources)}), in "
                         "that order, or a tractor-sumstats prefix.")
    return files, None


def _fit_pair(ss1, ss2, cross_file, own1, own2, annot, *, njack, max_chisq, drop_ambiguous, log):
    from ..inference.sumcore import Sumcore
    sc = Sumcore(rg=f"{ss1},{ss2}", ldscores=cross_file, ldscores_h2=(own1, own2), annot=annot, njack=njack,
                 chisq_threshold=max_chisq, intercept_chisq_thr=None, align_alleles=True, intercept_rg=0.0,
                 intercept_rg_source="track-rg: zero by design", drop_ambiguous=drop_ambiguous, log=_Log(log))
    sc._run()
    return sc.result


def _taus(res):
    """Per-replicate, per-class moment estimates (R+1 x K) of the three equations of a cross-mode fit: tau_1, tau_2
    (each trait's own-score fit) and tau_12 (the cross score), in units of each trait's phenotypic variance after the
    covariates (SUMMIT's HE equations: sigma_c = tau_c times the class's full V-weighted row mass)."""
    out = []
    for fit in (res["h2_fit1"], res["h2_fit2"]):
        p = fit.prepared
        K = p.trace_view.nbins
        G = np.asarray(fit.sigma_reps[:, :K]) * np.asarray(fit.pheno_var_reps)[:, None]   # tau A_r / Av_r sigma_core
        out.append(G * p.Ak_rep / (p.mass_Ak_rep * p.Ak_rep[-1][None, :]))
    rf = res["rg_fit"]
    p = rf.prepared
    vp = np.sqrt(np.asarray(res["h2_fit1"].pheno_var_reps) * np.asarray(res["h2_fit2"].pheno_var_reps))
    out.append(np.asarray(rf.gamma_reps) * vp[:, None] * p.Ak_rep / (p.mass_Ak_rep * p.Ak_rep[-1][None, :]))
    return out


def _summ(jk, unit_sizes, reps):
    est, se = jk.summarize(np.asarray(reps, np.float64), unit_sizes=unit_sizes, axis=0, center="mean",
                           nan_policy="omit")
    return np.asarray(est, np.float64), np.asarray(se, np.float64)


def _ratio(num, a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((a > 0) & (b > 0), num / np.sqrt(np.where(a > 0, a, 1.0) * np.where(b > 0, b, 1.0)), np.nan)


def _pair_summary(res, masses, pv1, pv2, classes, groups):
    """r = sum_c tau_12,c n_c / sqrt(sum_c tau_1,c n_c sum_c tau_2,c n_c) with the partners' counts n_c, per class and
    per group of classes, the track heritabilities sum_c tau_s,c PV_s,c, and SUMMIT's own rg_total, with jackknife SEs."""
    t1, t2, t12 = _taus(res)
    rf = res["rg_fit"]
    jk, us = rf.prepared.jackknife, rf.prepared.unit_sizes
    n = np.asarray(masses, np.float64)
    tot = lambda t, idx: (t[:, idx] * n[idx][None, :]).sum(axis=1)
    allc = np.arange(len(classes))
    r_reps = _ratio(tot(t12, allc), tot(t1, allc), tot(t2, allc))
    out = dict(r=_summ(jk, us, r_reps), classes={}, groups={})
    out["summit_rg_total"] = np.asarray(rf.rg_total, np.float64)
    h1, h2 = (t1 * np.asarray(pv1)[None, :]).sum(axis=1), (t2 * np.asarray(pv2)[None, :]).sum(axis=1)
    out["track_h2"] = (_summ(jk, us, h1), _summ(jk, us, h2))
    for c, name in enumerate(classes):
        out["classes"][name] = dict(r=_summ(jk, us, _ratio(t12[:, c], t1[:, c], t2[:, c])),
                                    tau_1=_summ(jk, us, t1[:, c]), tau_2=_summ(jk, us, t2[:, c]),
                                    tau_12=_summ(jk, us, t12[:, c]), partners=float(n[c]))
    for gname, members in (groups or {}).items():
        idx = np.array([classes.index(x) for x in members])
        out["groups"][gname] = dict(r=_summ(jk, us, _ratio(tot(t12, idx), tot(t1, idx), tot(t2, idx))),
                                    classes=list(members))
    out["n_snps"] = int(np.sum(rf.prepared.active_mask))
    out["_reps"] = r_reps
    out["_snps"] = np.asarray(rf.prepared.trace_view.snps)
    out["_jk"] = (jk, us)
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def parse_groups(specs, classes):
    """Groups of classes: 'NAME=COL1,COL2' strings (or a dict); every column must be a class of the annotation."""
    if not specs:
        return {}
    groups = dict(specs) if isinstance(specs, dict) else {}
    if not isinstance(specs, dict):
        for sp in specs:
            if "=" not in sp:
                raise ValueError(f"--group {sp!r}: expected NAME=COL1,COL2,...")
            k, v = sp.split("=", 1)
            groups[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]
    for k, v in groups.items():
        bad = [x for x in v if x not in classes]
        if bad or not v:
            raise ValueError(f"Group {k}: {', '.join(bad) or 'no classes'} not among the annotation's classes "
                             f"({', '.join(classes)}).")
    return groups


# ----------------------------------------------------------------------------------------------- pooling over traits
def _maf_of(c):
    return str(c).split("_")[0]


def pool_reps(T, n, sel):
    """Per-replicate pooled r over the traits in sel: sum_traits sum_c tau_12,c n_c / sqrt(sum sum tau_1,c n_c
    sum sum tau_2,c n_c), from each trait's per-replicate moments T[trait] = (tau_1, tau_2, tau_12) (_taus)."""
    tot = lambda t: (t * n[None, :]).sum(axis=1)
    return _ratio(sum(tot(T[k][2]) for k in sel), sum(tot(T[k][0]) for k in sel), sum(tot(T[k][1]) for k in sel))


def merge_rule(T, jk, us, n, classes, sel, thr=2.0):
    """The classes whose two LD halves are merged: per class the own-track moments pooled over the traits in sel,
    sum_traits tau_s,c n_c (s = each track), and their jackknife z; a MAF class (the class name up to its first '_')
    with two halves is merged when any of its four moments has z below thr or no finite z. Returns (merged MAF
    classes, the z table)."""
    z = {}
    for i in (0, 1):
        est, se = _summ(jk, us, sum(T[k][i] * n[None, :] for k in sel))
        with np.errstate(invalid="ignore", divide="ignore"):
            z[i] = np.where(se > 0, est / se, np.nan)
    merge = []
    for m in dict.fromkeys(_maf_of(c) for c in classes):
        idx = [i for i, c in enumerate(classes) if _maf_of(c) == m]
        if len(idx) == 2 and not all(np.isfinite(z[s][i]) and z[s][i] >= thr for s in (0, 1) for i in idx):
            merge.append(m)
    return merge, {str(s): dict(zip(classes, map(float, z[s]))) for s in z}


def merged_classes(classes, merge):
    """Classes after merging the LD halves of the MAF classes in merge, in order, with their member columns."""
    out = {}
    for c in classes:
        out.setdefault(_maf_of(c) if _maf_of(c) in merge else c, []).append(c)
    return out


def write_merged(tracks, annot, members, out):
    """Merged copies of a `compose --tracks` prefix (every file of its tracks block) and of the annotation: sums of the
    member classes' columns, partner counts (.gw.M) and partner V masses. Returns (prefix, annotation path)."""
    import copy
    from ..ldscore.gw_ldscore import write_gw_annotation_masses, write_gw_ldscore_table
    with open(f"{tracks}.admixed_ld.json") as fh:
        params = json.load(fh)
    tb = params["tracks"]
    cls = list(tb["classes"])
    idx = {c: [cls.index(x) for x in mem] for c, mem in members.items()}
    p2 = copy.deepcopy(params)
    t2 = p2["tracks"]
    t2["classes"] = list(members)
    t2["partner_counts"] = [float(sum(tb["partner_counts"][i] for i in idx[c])) for c in members]
    t2["partner_V_mass"] = {k: [float(sum(v[i] for i in idx[c])) for c in members] for k, v in tb["partner_V_mass"].items()}
    for nm, rec in tb["files"].items():
        df = pd.read_csv(rec["file"], sep="\t", dtype={"SNP": str, "CHR": str})
        if [c for c in df.columns if c not in ("CHR", "SNP", "BP", "V")] != cls:
            raise ValueError(f"{rec['file']}: its columns are not the classes {cls}.")
        scores = pd.DataFrame({c: df[mem].to_numpy(np.float64).sum(axis=1) for c, mem in members.items()})
        dst = f"{out}.{nm}"
        write_gw_ldscore_table(dst, df[["CHR", "SNP", "BP"]], scores, snp_var=df["V"].to_numpy() if "V" in df else None)
        M = np.array(open(rec["file"][:-len(".gw.ldscore.gz")] + ".gw.M").read().split(), dtype=np.float64)
        write_gw_annotation_masses(dst, np.array([M[idx[c]].sum() for c in members]))
        t2["files"][nm]["file"] = f"{dst}.gw.ldscore.gz"
    p2["class_merge"] = dict(prefix=os.path.abspath(tracks), members=members)
    with open(f"{out}.admixed_ld.json", "w") as fh:
        json.dump(p2, fh, indent=1)
    A = pd.read_csv(annot, sep=r"\s+", dtype={"SNP": str})
    meta = [c for c in A.columns if c in ("CHR", "BP", "SNP", "CM")]
    if [c for c in A.columns if c not in meta] != cls:
        raise ValueError(f"{annot}: its class columns are not the track LD scores' classes {cls}.")
    B = A[meta].copy()
    for c, mem in members.items():
        B[c] = A[mem].sum(axis=1)
    af = f"{out}.annot.tsv.gz"
    B.to_csv(af, sep="\t", index=False, compression="gzip")
    return out, af


def _pooled_track_rg(tracks, sumstats, annot, out, *, njack, groups, allow_ols, max_chisq, merge_ld_halves,
                     merge_z, log):
    """track_rg over several traits (one tractor-sumstats prefix each): every trait's pair fit, and r pooled over the
    traits on shared jackknife blocks (traits whose SNPs differ from the first trait's are left out of the pool), with
    the optional merge of LD halves decided on the pooled own-track moments."""
    with open(f"{tracks}.admixed_ld.json") as fh:
        tb = json.load(fh)["tracks"]
    sources = tb["sources"]
    if len(sources) != 2:
        raise ValueError("Pooling over traits covers two tracks.")
    s, t = sources
    names = [os.path.basename(str(x).rstrip("/")) for x in sumstats]
    if len(set(names)) != len(names):
        raise ValueError("The trait prefixes must have distinct names.")
    files = {}
    for nm, pre in zip(names, sumstats):
        f, _ = _track_files(pre, sources)
        for x in f:
            st = _se_type(x)
            if st not in (None, "HC") and not allow_ols:
                raise ValueError(f"{x} has {st} standard errors; use sandwich SEs or pass --allow-ols.")
        files[nm] = f

    def fit_all(pre, annot_file, classes):
        with open(f"{pre}.admixed_ld.json") as fh:
            tb_ = json.load(fh)["tracks"]
        fl, n, pv = tb_["files"], np.asarray(tb_["partner_counts"], np.float64), tb_["partner_V_mass"]
        T, per, snps0, jk0, excluded = {}, {}, None, None, []
        for nm in names:
            res = _fit_pair(files[nm][0], files[nm][1], fl[f"track_{s}_{t}"]["file"], fl[f"track_{s}"]["file"],
                            fl[f"track_{t}"]["file"], annot_file, njack=njack, max_chisq=max_chisq,
                            drop_ambiguous=False, log=lambda m: None)
            ps = _pair_summary(res, n, pv[s], pv[t], classes, groups)
            if snps0 is None:
                snps0, jk0 = ps["_snps"], ps["_jk"]
            if np.array_equal(ps["_snps"], snps0):
                T[nm] = _taus(res)
            else:
                excluded.append(nm)
            per[nm] = _jsonable(dict(r=ps["r"], track_h2={s: ps["track_h2"][0], t: ps["track_h2"][1]},
                                     n_snps=ps["n_snps"]))
            log(f"[admixed-ld] track-rg: {nm}: r {per[nm]['r'][0]:.4f} (SE {per[nm]['r'][1]:.4f}), {ps['n_snps']} SNPs")
        return T, per, jk0, excluded, n

    classes = list(tb["classes"])
    T, per, (jk, us), excluded, n = fit_all(tracks, annot, classes)
    sel = [k for k in names if k in T]
    merge_info = None
    if merge_ld_halves:
        merge, ztab = merge_rule(T, jk, us, n, classes, sel, merge_z)
        members = merged_classes(classes, merge)
        merge_info = dict(threshold=float(merge_z), merged=merge, z={sources[int(k)]: v for k, v in ztab.items()},
                          classes=list(members))
        log(f"[admixed-ld] track-rg: merge rule (pooled own-track z < {merge_z:g}): the LD halves of "
            f"{', '.join(merge) if merge else 'no MAF class'} are merged")
        if merge:
            pre2, annot2 = write_merged(tracks, annot, members, f"{out}.merged")
            classes = list(members)
            T, per, (jk, us), excluded, n = fit_all(pre2, annot2, classes)
            sel = [k for k in names if k in T]
            merge_info.update(prefix=pre2, annot=annot2)
    reps = pool_reps(T, n, sel)
    r = _summ(jk, us, reps)
    by_class = {}
    for c, name in enumerate(classes):
        tot = lambda i: sum(T[k][i][:, c] * n[c] for k in sel)
        by_class[name] = _summ(jk, us, _ratio(tot(2), tot(0), tot(1)))
    pooled = _jsonable(dict(r=r, classes=by_class, n_traits=len(sel), traits=sel, excluded=excluded,
                            definition="sum over traits and classes of tau_12,c n_c over the root of the products of the "
                                       "two tracks' sums; one jackknife over the traits' shared blocks"))
    if excluded:
        log(f"[admixed-ld] WARNING: {', '.join(excluded)} use other SNPs than {sel[0]} and are left out of the pool")
    log(f"[admixed-ld] track-rg: pooled r({s}, {t}) over {len(sel)} traits = {float(r[0]):.4f} (SE {float(r[1]):.4f})")
    summary = dict(format="summit-admixed-ld-track-rg", version=1, tracks=tracks, sources=sources, annot=annot,
                   classes=classes, njack=njack, traits=per, pooled=pooled, class_merge=merge_info)
    with open(f"{out}.track_rg.json", "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    rows = [dict(pair=f"{s}_{t}", what="r pooled", est=float(r[0]), se=float(r[1]))]
    rows += [dict(pair=f"{s}_{t}", what=f"r pooled[{c}]", est=float(v[0]), se=float(v[1])) for c, v in by_class.items()]
    rows += [dict(pair=f"{s}_{t}", what=f"r[{k}]", est=v["r"][0], se=v["r"][1]) for k, v in per.items()]
    pd.DataFrame(rows).to_csv(f"{out}.track_rg.tsv", sep="\t", index=False, float_format="%.6g")
    log(f"[admixed-ld] track-rg: wrote {out}.track_rg.json and {out}.track_rg.tsv")
    return _jsonable(summary)


def track_rg(tracks, sumstats, annot, out, *, external=None, cohort_track=None, njack=200, groups=None,
             allow_ols=False, max_chisq=None, keep_ambiguous=False, merge_ld_halves=False, merge_z=2.0, log=print):
    """Correlation of per-allele causal effects across local-ancestry tracks (and, with `external`, of each track with a
    GWAS of other people from the cross-source population of compose).

    tracks: compose's output prefix (made with --tracks); sumstats: a tractor-sumstats prefix or one file per track in
    compose's source order; annot: the fit's annotation (as for compose); external: the other population's summary
    statistics (compose --tracks --cross-source X); cohort_track: the track of the external cohort's ancestry for the
    triangle check r(s, X) = r(s, t) r(t, X) (default: X itself when it is one of the sources). Each pair is SUMMIT's cross-mode fit with the overlapping covariance
    fixed at 0 (Sumcore: --rg S,T --ldscores track_S_T --ldscores-h2-1 track_S --ldscores-h2-2 track_T
    --overlap-covariance-rg 0); the summary weights the classes by the partners' counts (see the module docstring).
    Writes <out>.track_rg.json and <out>.track_rg.tsv and returns the summary.

    Several traits (sumstats a list of tractor-sumstats prefixes, one per trait): each trait is fitted, and r is pooled
    over the traits as the ratio of their summed moments, with one jackknife over the shared blocks (_pooled_track_rg).
    merge_ld_halves: annotation classes named <maf class>_<half>, two halves per MAF class (e.g. ten classes, five MAF
    classes split at their median LD score); a MAF class's halves are merged when any of its four own-track moments,
    pooled over the traits, has a jackknife z below merge_z, and the traits are fitted again on the merged classes."""
    if not isinstance(sumstats, str) and len(sumstats) > 1 and all(os.path.exists(f"{x}.tracks.json") for x in sumstats):
        if external is not None:
            raise ValueError("--external is fitted one trait at a time; give one trait's --sumstats.")
        return _pooled_track_rg(tracks, list(sumstats), annot, out, njack=njack, groups=parse_groups(groups, list(
            json.load(open(f"{tracks}.admixed_ld.json"))["tracks"]["classes"])), allow_ols=allow_ols,
            max_chisq=max_chisq, merge_ld_halves=merge_ld_halves, merge_z=merge_z, log=log)
    if not isinstance(sumstats, str) and len(sumstats) == 1:
        sumstats = sumstats[0]
    if merge_ld_halves:
        raise ValueError("--merge-ld-halves decides on moments pooled over traits; give several traits' --sumstats.")
    with open(f"{tracks}.admixed_ld.json") as fh:
        params = json.load(fh)
    tb = params.get("tracks")
    if tb is None:
        raise ValueError(f"{tracks}.admixed_ld.json has no track LD scores; run compose with --tracks.")
    sources = tb["sources"]
    files, rec = _track_files(sumstats, sources)
    flags = list(tb.get("flags", []))
    for f, s in zip(files, sources):
        st = _se_type(f)
        if st is None:
            flags.append(dict(flag="se_type_unknown", detail=f"{f}: no SE_TYPE column; sandwich SEs are assumed"))
            log(f"[admixed-ld] WARNING (recorded flag se_type_unknown): {f} does not say whether its SEs are sandwich "
                "SEs (tractor-sumstats writes SE_TYPE); the own-track moments need them")
        elif st != "HC":
            if not allow_ols:
                raise ValueError(f"{f} has {st} standard errors. The own-track moments of OLS z statistics hold "
                                 "same-person terms that the track LD scores do not model; use sandwich SEs, or pass "
                                 "--allow-ols to proceed with the flag recorded.")
            flags.append(dict(flag="ols_standard_errors", detail=f"{f}: {st} SEs (own-track moments biased)"))
            log(f"[admixed-ld] WARNING (recorded flag ols_standard_errors): {f} has {st} SEs")
    classes = list(tb["classes"])
    groups = parse_groups(groups, classes)
    masses = tb["partner_counts"]
    pv = tb["partner_V_mass"]
    fl = tb["files"]
    njack = njack if njack is not None else 200
    summary = dict(format="summit-admixed-ld-track-rg", version=1, tracks=tracks, sources=sources, annot=annot,
                   classes=classes, partners=tb.get("partners"), partner_counts=masses, weights=tb["weights"],
                   njack=njack, sumstats=files, flags=flags, pairs={}, external=None,
                   units="track heritabilities in units of each track's noise variance (sandwich SEs: the phenotypic "
                         "variance after covariates, weighted by the track's local ancestry); r is unit-free",
                   definition="r = sum_c tau_st,c n_c / sqrt(sum_c tau_s,c n_c sum_c tau_t,c n_c), n_c the partners' "
                              "count in class c; per class r_c = tau_st,c / sqrt(tau_s,c tau_t,c)")
    rows = []
    fits = {}
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            s, t = sources[i], sources[j]
            log(f"[admixed-ld] track-rg: tracks {s} and {t}")
            res = _fit_pair(files[i], files[j], fl[f"track_{s}_{t}"]["file"], fl[f"track_{s}"]["file"],
                            fl[f"track_{t}"]["file"], annot, njack=njack, max_chisq=max_chisq, drop_ambiguous=False,
                            log=log)
            ps = _pair_summary(res, masses, pv[s], pv[t], classes, groups)
            fits[("tracks", s, t)] = ps
            summary["pairs"][f"{s}_{t}"] = _jsonable(dict(
                r=ps["r"], summit_rg_total=ps["summit_rg_total"], track_h2={s: ps["track_h2"][0], t: ps["track_h2"][1]},
                classes=ps["classes"], groups=ps["groups"],
                n_snps=ps["n_snps"]))
            rows.append(dict(pair=f"{s}_{t}", what="r", est=ps["r"][0], se=ps["r"][1]))
            for nm, v in ps["classes"].items():
                rows.append(dict(pair=f"{s}_{t}", what=f"r[{nm}]", est=v["r"][0], se=v["r"][1]))
            for nm, v in ps["groups"].items():
                rows.append(dict(pair=f"{s}_{t}", what=f"r[group {nm}]", est=v["r"][0], se=v["r"][1]))
            for nm, h in zip((s, t), ps["track_h2"]):
                rows.append(dict(pair=f"{s}_{t}", what=f"track_h2[{nm}]", est=h[0], se=h[1]))
            log(f"[admixed-ld] track-rg: r({s}, {t}) = {ps['r'][0]:.4f} (SE {ps['r'][1]:.4f}); track h2 {s} "
                f"{ps['track_h2'][0][0]:.4g} (SE {ps['track_h2'][0][1]:.3g}), {t} {ps['track_h2'][1][0]:.4g} "
                f"(SE {ps['track_h2'][1][1]:.3g}); {ps['n_snps']} SNPs, {len(classes)} classes")
    if external is not None:
        X = tb.get("cross_source")
        if X is None:
            raise ValueError("--external needs track LD scores against that population: compose --tracks "
                             "--cross-source NAME.")
        ext = dict(cohort=X, sumstats=external, tracks={})
        for i, s in enumerate(sources):
            log(f"[admixed-ld] track-rg: track {s} and the {X} GWAS")
            res = _fit_pair(files[i], external, fl[f"track_{s}_vs_{X}"]["file"], fl[f"track_{s}"]["file"],
                            fl[f"cohort_{X}"]["file"], annot, njack=njack, max_chisq=max_chisq,
                            drop_ambiguous=not keep_ambiguous, log=log)
            ps = _pair_summary(res, masses, pv[s], pv[f"cohort_{X}"], classes, groups)
            fits[("cohort", s)] = ps
            ext["tracks"][s] = _jsonable(dict(r=ps["r"], summit_rg_total=ps["summit_rg_total"],
                                              track_h2=ps["track_h2"][0], cohort_h2=ps["track_h2"][1],
                                              classes=ps["classes"], groups=ps["groups"], n_snps=ps["n_snps"]))
            rows.append(dict(pair=f"{s}_vs_{X}", what="r", est=ps["r"][0], se=ps["r"][1]))
            log(f"[admixed-ld] track-rg: r({s} track, {X} GWAS) = {ps['r'][0]:.4f} (SE {ps['r'][1]:.4f})")
        # the triangle: with the same people held fixed r_st isolates ancestry, with the ancestry held fixed r(t, X)
        # isolates the cohorts; if the two factors separate, r(s, X) = r_st r(t, X)
        t = cohort_track if cohort_track is not None else (X if X in sources else None)
        if t is not None and t not in sources:
            raise ValueError(f"--cohort-track {t} is not one of the tracks ({', '.join(sources)}).")
        if t is not None and len(sources) == 2:
            s = [x for x in sources if x != t][0]
            pair = fits.get(("tracks", s, t)) or fits.get(("tracks", t, s))
            a, b = fits[("cohort", s)], fits[("cohort", t)]
            prod = pair["r"][0] * b["r"][0]
            chk = dict(track=s, same_ancestry_track=t, r_track_cohort=a["r"][0], product=prod,
                       difference=a["r"][0] - prod, definition=f"r({s}, {X} GWAS) - r({s}, {t}) r({t}, {X} GWAS)")
            if np.array_equal(pair["_snps"], a["_snps"]) and np.array_equal(pair["_snps"], b["_snps"]):
                jk, us = pair["_jk"]
                d = _summ(jk, us, a["_reps"] - pair["_reps"] * b["_reps"])
                chk["difference_se"] = float(d[1])
            ext["triangle_check"] = chk
            log(f"[admixed-ld] track-rg: triangle r({s}, {X} GWAS) {a['r'][0]:.4f} against r({s}, {t}) r({t}, {X} GWAS) "
                f"{prod:.4f}" + (f" (difference SE {chk['difference_se']:.4f})" if "difference_se" in chk else ""))
        summary["external"] = ext
    with open(f"{out}.track_rg.json", "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    pd.DataFrame(rows).to_csv(f"{out}.track_rg.tsv", sep="\t", index=False, float_format="%.6g")
    log(f"[admixed-ld] track-rg: wrote {out}.track_rg.json and {out}.track_rg.tsv")
    return _jsonable(summary)
