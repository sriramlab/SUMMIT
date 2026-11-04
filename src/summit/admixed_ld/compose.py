"""Composed LD scores of an admixed GWAS sample (`summit admixed-ld compose`).

The population value of SUMMIT's V-weighted LD score of SNP j in class c is
    l_jc = sum_{k in c} C_res(j,k)^2 / (C_res(j,j) s_k^2)
         + (1/nu) sum_{k in c} Cov_theta(h_j, h_k) / (C_res(j,j) s_k^2),        h_k(theta) = 2 p_k(theta)(1 - p_k(theta)),
the second sum being the part of the in-sample score from each person's pairing with themselves, which SUMMIT's
noise correction leaves when heterozygosity follows ancestry. With s_k^2 = C(k,k) and C_res the covariance after the
covariates, it is computed as:
  pairs within the reference window   pi_A(d) C^A + pi_E(d) C^E + delta_j delta_k kappa(d), squared, minus the sampling
                                      variance of the squared reference covariances;
  every pair on the same chromosome   delta_j^2 delta_k^2 kappa(d)^2, by FFT convolution on a grid of `grid_cm`
                                      (the window pairs' admixture-only part is taken out of the first sum);
  pairs on different chromosomes      delta_j^2 delta_k^2 far^2;
  the same-person term, each SNP with itself (V_j), and, optionally, relatives
                                      (sum over their ordered pairs of g^2) / nu^2 * sum_{k in c} V_k, with g the
                                      relatedness of the covariate-adjusted, standardized genotypes.
V_j = C_res(j,j) / s_j^2 is the population share of SNP j's variance that the covariates leave, which SUMMIT's
in-sample V (one minus the adjusted R2) estimates, so neither V nor the partner weights need a degrees-of-freedom
factor. The output is written with SUMMIT's own LD-score writer, so `summit --h2/--rg --ldscores <out>.gw.ldscore.gz
--annot ...` read it unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd

from ._io import match_variants
from .model import Kernel, beta_moments, parameters_from_sumstats
from .reference import Reference


def _sumstats_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.endswith((".tsv", ".tsv.gz", ".txt",
                                                                                     ".txt.gz", ".gz")))
        else:
            files.append(p)
    if not files:
        raise ValueError("No summary-statistic files found.")
    return files


def _target_variants(snps_path, first_sumstats):
    """CHR, BP, SNP, COUNTED, OTHER of the GWAS variants, from --snps (BIM, PVAR or a table with these columns) or
    from the summary statistics (SNP, CHR, BP, A1, A2)."""
    from ._io import read_bim
    if snps_path is not None:
        if snps_path.endswith(".bim"):
            v = read_bim(snps_path)
            return v.assign(COUNTED=v.A1.str.upper(), OTHER=v.A2.str.upper())
        if snps_path.endswith(".pvar"):
            from ..ldscore.genotype_source import read_pvar_variants
            v = read_pvar_variants(snps_path)
            return v.assign(COUNTED=v.A2.str.upper(), OTHER=v.A1.str.upper())
        df = pd.read_csv(snps_path, sep=r"\s+", dtype=str)
    else:
        df = pd.read_csv(first_sumstats, sep=r"\s+", dtype=str, compression="infer")
    cols = {c.upper().lstrip("#"): c for c in df.columns}
    pick = {k: next((cols[x] for x in alts if x in cols), None)
            for k, alts in dict(SNP=("SNP", "ID"), CHR=("CHR", "CHROM"), BP=("BP", "POS"), A1=("A1", "ALT"),
                                A2=("A2", "REF")).items()}
    if pick["SNP"] is None or pick["A1"] is None or pick["A2"] is None:
        raise ValueError("The variant table needs SNP, A1 and A2 columns.")
    out = pd.DataFrame({"SNP": df[pick["SNP"]].astype(str), "COUNTED": df[pick["A1"]].str.upper(),
                        "OTHER": df[pick["A2"]].str.upper()})
    if pick["CHR"] is not None and pick["BP"] is not None:
        out["CHR"] = df[pick["CHR"]].astype(str).str.replace(r"^(?i:chr)", "", regex=True)
        out["BP"] = pd.to_numeric(df[pick["BP"]], errors="coerce")
    return out


def _fft_conv_sym(W: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """F[g] = sum_g' W[g'] kern[|g - g'|] for every column of W (grid x classes)."""
    n = W.shape[0]
    size = 1 << int(np.ceil(np.log2(max(2 * n - 1, 2))))
    sym = np.zeros(size)
    sym[:n] = kern[:n]
    sym[size - n + 1:] = kern[1:n][::-1]
    fk = np.fft.rfft(sym)
    fw = np.fft.rfft(W, n=size, axis=0)
    return np.fft.irfft(fw * fk[:, None], n=size, axis=0)[:n]


def compose_ldscores(ref_path, sumstats, annot, out, *, sources=None, snps=None, match="position", curve=None,
                     generations=None, decay_per_cm=None, mean_ancestry=None, kappa0=None, theta_sd=None,
                     relatedness=None, nu=None, r2=0.99, grid_cm=0.02, pair_chunk=1 << 23, log=print):
    from ..ldscore.gw_ldscore import write_gw_annotation_masses, write_gw_ldscore_table
    from ..relatives.moments import read_annotation_classes
    from ..sumstats.sumstats import Sumstats
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    sources = sources or ref.names[:2]
    if len(sources) != 2:
        raise ValueError("Composition uses exactly two sources (the first is the ancestry whose mean is m).")
    A, E = sources
    iA, iE = ref.source_index(A), ref.source_index(E)
    nA, nE = ref.n[A], ref.n[E]
    if sum(x is not None for x in (curve, generations, decay_per_cm)) != 1:
        raise ValueError("Give exactly one source for the decay: --curve, --admixture-generations or --decay-per-cm.")

    # ---------------------------------------------------------------------------------------- target variants
    files = _sumstats_files(sumstats)
    tv = _target_variants(snps, files[0])
    if tv.empty:
        raise ValueError(f"No variants in {snps or files[0]}.")
    how = match
    if how == "position" and not {"CHR", "BP"} <= set(tv.columns):
        raise ValueError("Matching by position needs CHR and BP (in the summary statistics or --snps); "
                         "or use --match id.")
    if how == "position":
        tv = tv[tv.BP.notna()].copy()
        tv["BP"] = tv.BP.astype(np.int64)
    rv = ref.variants(sources)
    t_idx, r_idx, _ = match_variants(tv.reset_index(drop=True), rv, how=how)
    tv = tv.reset_index(drop=True)
    cls_mat, classes = read_annotation_classes(annot, tv.SNP.to_numpy()[t_idx])
    in_annot = cls_mat.sum(axis=1) == 1
    t_idx, r_idx, cls_mat = t_idx[in_annot], r_idx[in_annot], cls_mat[in_annot]
    order = np.argsort(r_idx, kind="stable")                       # reference (genome) order
    t_idx, r_idx, cls_mat = t_idx[order], r_idx[order], cls_mat[order]
    pA, pE = rv[f"p_{A}"].to_numpy()[r_idx], rv[f"p_{E}"].to_numpy()[r_idx]
    vA, vE = rv[f"v_{A}"].to_numpy()[r_idx], rv[f"v_{E}"].to_numpy()[r_idx]
    delta = pA - pE
    good = np.isfinite(delta) & np.isfinite(vA) & np.isfinite(vE) & (vA + vE > 0)
    t_idx, r_idx, cls_mat = t_idx[good], r_idx[good], cls_mat[good]
    pA, pE, vA, vE, delta = pA[good], pE[good], vA[good], vE[good], delta[good]
    cls = np.argmax(cls_mat, axis=1)
    K = len(classes)
    M = r_idx.size
    if M == 0:
        raise ValueError("No GWAS variant is in the reference and the annotation.")
    snp_ids = tv.SNP.to_numpy()[t_idx]
    log(f"[admixed-ld] compose: {M} of {len(tv)} GWAS variants are in the reference ({A}, {E}) and the annotation "
        f"({K} classes)")

    # ---------------------------------------------------------------------------------------- parameters
    tables, n_med, p_rank = [], [], []
    for f in files:
        ss = Sumstats.from_file(f, name=os.path.basename(f), compute_diagnostics=False)
        ix = ss.index.get_indexer(snp_ids)
        t = pd.DataFrame({"N": np.where(ix >= 0, ss.n[ix], np.nan), "SE": np.where(ix >= 0, ss.se[ix], np.nan),
                          "BETA": np.where(ix >= 0, ss.beta[ix], np.nan)})
        tables.append((ss.name, t))
        n_med.append(float(np.median(ss.n)))
        p_rank.append(int(ss.cov_rank))
    N = float(np.median(n_med))
    p = int(np.median(p_rank))
    if mean_ancestry is None or kappa0 is None:
        m_se, k_se, per_trait = parameters_from_sumstats(tables, vA, vE, delta, log=log)
    else:
        m_se, k_se, per_trait = float("nan"), float("nan"), None
    m = float(mean_ancestry) if mean_ancestry is not None else m_se
    k0 = float(kappa0) if kappa0 is not None else k_se
    nu = float(nu) if nu is not None else N - p - 1.0

    # ---------------------------------------------------------------------------------------- kernel and ancestry
    curve_info = None
    if curve is not None:
        cdf = pd.read_csv(f"{curve}.curve.tsv", sep="\t")
        with open(f"{curve}.curve.json") as fh:
            curve_info = json.load(fh)
        if list(curve_info.get("sources", sources)) != list(sources):
            raise ValueError(f"The curve was made with sources {curve_info.get('sources')}, not {sources}.")
        kern = Kernel.from_curve(cdf.d_cm, cdf.kernel, curve_info["inter_kernel"], k0)
    elif generations is not None:
        kern = Kernel.from_decay(k0, float(generations) / 100.0, ref.genome_cm,
                                 source=f"admixture {generations} generations ago")
    else:
        kern = Kernel.from_decay(k0, float(decay_per_cm), ref.genome_cm, source=f"decay {decay_per_cm} per cM")
    if kern.note:
        log(f"[admixed-ld] note: {kern.note}")
    if theta_sd is not None:
        th = beta_moments(m, theta_sd)
        th_source = f"Beta with mean {m:.3f} and SD {theta_sd}"
    elif curve_info is not None:
        th, th_source = curve_info["theta"], "the cohort's curve"
    else:
        th = beta_moments(m, 0.12)
        th_source = "Beta with SD 0.12 (default; pass --ancestry-sd)"
        log("[admixed-ld] note: global-ancestry moments default to SD 0.12; they enter only the same-person term, "
            "which is small at biobank sample sizes.")
    if relatedness is None:                 # optional input
        rel_sum = None
    elif str(relatedness) == "curve":
        if curve_info is None or "relatedness" not in curve_info:
            raise ValueError("--relatedness curve needs a curve made with --relative-pairs.")
        rel_sum = float(curve_info["relatedness"]["sum_squared_ordered"])
    else:
        rel_sum = float(relatedness)

    # ---------------------------------------------------------------------------------------- per-SNP model
    cres = m * vA + (1.0 - m) * vE + delta ** 2 * k0
    s2 = cres + delta ** 2 * 4.0 * r2 * th["var"]
    ok = cres > 1e-8
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} SNPs have no variance in the model; they should have been removed.")
    Vpop = cres / s2
    L = np.zeros((M, K))
    chrom_out = rv["CHR"].to_numpy()[r_idx]
    local_out = rv["LOCAL"].to_numpy()[r_idx]
    cm_out = rv["CM"].to_numpy()[r_idx]
    wd = delta ** 2 / s2                                           # partner weight of the admixture terms
    W_tot = np.bincount(cls, weights=wd, minlength=K)
    far2 = kern.far ** 2
    n_pairs = 0
    for c in ref.chromosomes:
        sel = np.flatnonzero(chrom_out == c)
        if sel.size == 0:
            continue
        z = ref.load(c)
        nloc = z["bp"].size
        tmap = -np.ones(nloc, dtype=np.int64)
        tmap[local_out[sel]] = sel
        indptr, indices, cm_ref = z["indptr"], z["indices"], z["cm"]
        covA, covE, m22A, m22E = z[f"cov_{iA}"], z[f"cov_{iE}"], z[f"m22_{iA}"], z[f"m22_{iE}"]
        # pairs within the window (upper band), in chunks of rows
        r0 = 0
        while r0 < nloc:
            r1 = int(np.searchsorted(indptr, indptr[r0] + pair_chunk, side="right")) - 1
            r1 = min(max(r1, r0 + 1), nloc)
            s0, s1 = int(indptr[r0]), int(indptr[r1])
            if s1 > s0:
                cnt = np.diff(indptr[r0:r1 + 1])
                jl = np.repeat(np.arange(r0, r1, dtype=np.int64), cnt)
                kl = indices[s0:s1].astype(np.int64)
                j, k = tmap[jl], tmap[kl]
                use = (j >= 0) & (k >= 0)
                if use.any():
                    j, k, jl, kl = j[use], k[use], jl[use], kl[use]
                    cA = covA[s0:s1][use].astype(np.float64)
                    cE = covE[s0:s1][use].astype(np.float64)
                    vAp = np.maximum(m22A[s0:s1][use] - cA * cA, 0.0) / nA
                    vEp = np.maximum(m22E[s0:s1][use] - cE * cE, 0.0) / nE
                    d = np.abs(cm_ref[kl] - cm_ref[jl])
                    rec = np.exp(-kern.lam * d)
                    piA, piE = m * rec, (1.0 - m) * rec
                    adm = delta[j] * delta[k] * kern(d)
                    cr = piA * cA + piE * cE + adm
                    sq = cr * cr - piA * piA * vAp - piE * piE * vEp - adm * adm
                    L += np.bincount(j * K + cls[k], weights=sq / (cres[j] * s2[k]), minlength=M * K).reshape(M, K)
                    L += np.bincount(k * K + cls[j], weights=sq / (cres[k] * s2[j]), minlength=M * K).reshape(M, K)
                    n_pairs += int(use.sum())
            r0 = r1
        # admixture LD of every pair on the chromosome
        x = cm_out[sel]
        g = np.floor((x - x.min()) / grid_cm).astype(np.int64)
        ng = int(g.max()) + 1
        Wg = np.zeros((ng, K))
        np.add.at(Wg, (g, cls[sel]), wd[sel])
        k2 = kern(np.arange(ng) * grid_cm) ** 2
        F = _fft_conv_sym(Wg, k2)[g]                                # (SNPs on chr) x classes, includes each SNP itself
        F[np.arange(sel.size), cls[sel]] -= wd[sel] * k2[0]
        W_chr = Wg.sum(axis=0)
        L[sel] += (delta[sel] ** 2 / cres[sel])[:, None] * (F + far2 * (W_tot - W_chr)[None, :])
        log(f"[admixed-ld] compose chr{c}: {sel.size} SNPs ({time.perf_counter() - t0:.0f} s)")
    # same-person term, each SNP with itself, relatives
    pbar = m * pA + (1.0 - m) * pE
    beta_ = 2.0 * delta * (1.0 - 2.0 * pbar)
    gam = -2.0 * delta ** 2
    Bc = np.bincount(cls, weights=beta_ / s2, minlength=K)
    Gc = np.bincount(cls, weights=gam / s2, minlength=K)
    mu2, mu3, mu4 = th["var"], th["m3"], th["m4"]
    same = (beta_[:, None] * mu2 * Bc[None, :] + (beta_[:, None] * Gc[None, :] + gam[:, None] * Bc[None, :]) * mu3
            + gam[:, None] * Gc[None, :] * (mu4 - mu2 * mu2)) / (nu * cres)[:, None]
    L += same
    L[np.arange(M), cls] += Vpop
    if rel_sum is not None:
        L += (rel_sum / nu ** 2) * np.bincount(cls, weights=Vpop, minlength=K)[None, :]
    V = Vpop

    # ---------------------------------------------------------------------------------------- write
    snpdf = pd.DataFrame({"CHR": chrom_out, "SNP": snp_ids, "BP": rv["BP"].to_numpy()[r_idx]})
    write_gw_ldscore_table(out, snpdf, pd.DataFrame(L, columns=classes), snp_var=V)
    write_gw_annotation_masses(out, np.bincount(cls, minlength=K).astype(float))
    params = dict(format="summit-admixed-ld-compose", version=1, sources=[A, E], n_snps=int(M),
                  n_window_pairs=int(n_pairs), mean_ancestry=m, kappa0=k0,
                  mean_ancestry_from_se=m_se, kappa0_from_se=k_se, gwas_n=N, covariate_rank=p, nu=nu,
                  kernel=kern.describe(), theta=th, theta_source=th_source, relatedness=rel_sum, r2=r2,
                  grid_cm=grid_cm, reference=dict(sources=ref.manifest["sources"], window_kb=ref.manifest["window_kb"],
                                                  genetic_map=ref.manifest["genetic_map"]),
                  per_trait=None if per_trait is None else per_trait.to_dict(orient="records"),
                  ld_mass=float((V[:, None] * L).sum()))
    with open(f"{out}.admixed_ld.json", "w") as fh:
        json.dump(params, fh, indent=1)
    log(f"[admixed-ld] compose: wrote {out}.gw.ldscore.gz and {out}.gw.M ({M} SNPs; m {m:.4f}, kappa_res(0) "
        f"{k0:.4f}, decay {kern.lam:.4f} per cM, far {kern.far:.5f}) in {time.perf_counter() - t0:.0f} s")
    return params
