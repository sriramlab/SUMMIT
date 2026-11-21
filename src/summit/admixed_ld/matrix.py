"""Regional LD matrices of an admixed GWAS sample (`summit admixed-ld matrix`), for fine-mapping, conditional analysis
and other methods that take a correlation matrix with summary statistics.

The covariance of the covariate-adjusted dosages of SNPs j and k at genetic distance d (cM) is composed as
    C(j, k) = sum_s m_s exp(-lam_s d) T_s(j, k) C^s(j, k) + D_j' K(d) D_k,
    C(j, j) = sum_s m_s v_s(j) + D_j' K(0) D_j,
and the matrix written is the correlation R = diag(C)^-1/2 C diag(C)^-1/2, oriented to the reference's counted allele.
C^s is the panel covariance of source s, D_j the frequency contrasts against the baseline (last) source, K(d) the
residual local-ancestry kernel after the GWAS covariates (the cohort's curve or a published date, with K(0) and the
mean ancestry m from `compose`'s parameters), and T_s an optional taper of the panel term.

The panel term carries sampling error of order 1 / sqrt(n_s) at every pair, while true within-source LD vanishes beyond
a few hundred kb; the admixture term is long-range but built from frequencies and the genome-wide kernel. The taper
T_s(j, k) = exp(-d / tau_s) removes the panel noise at long range and keeps the admixture term. With --taper auto,
tau_s minimizes an unbiased estimate of the squared error of the tapered panel correlations over the chromosome's pairs,
sum w [(1 - T)^2 U + T^2 (c^2 - U)], U the unbiased estimate of the squared covariance (from the panel's m22),
w = 1 / (v_j v_k). Exponential kernels are positive definite, so by the Schur product theorem the panel term stays
positive semi-definite, and the admixture term is too whenever the kernel matrix over the region's positions is; if it
is not (a noisy curve), that block matrix is projected onto the positive semi-definite cone.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .compose import unbiased_squared_covariance
from .model import KernelMatrix, check_kernel_join, contrast_pairs
from .reference import Reference

_TAUS = np.exp(np.linspace(np.log(0.002), np.log(5.0), 120))


def _parameters(params_path, sources, mean_ancestry=None, kappa0=None):
    """Mean ancestry (K) and K(0) ((K-1) x (K-1)) from compose's parameters, unless given."""
    k1 = len(sources) - 1
    p = json.load(open(params_path)) if params_path else {}
    if p and list(p.get("sources", sources)) != list(sources):
        raise ValueError(f"The parameters were composed with sources {p.get('sources')}, not {sources}.")
    if mean_ancestry is not None:
        m = np.asarray([float(x) for x in str(mean_ancestry).split(",")])
    elif "mean_ancestry_vector" in p:
        m = np.asarray(p["mean_ancestry_vector"], float)
    elif "mean_ancestry" in p:
        m = np.atleast_1d(np.asarray(p["mean_ancestry"], float))
    else:
        raise ValueError("Give --params (compose's .admixed_ld.json) or --mean-ancestry.")
    if m.size == k1:
        m = np.r_[m, 1.0 - m.sum()]
    if m.size != k1 + 1 or m.min() <= 0:
        raise ValueError(f"Mean ancestry needs {k1} or {k1 + 1} values in (0, 1).")
    m = m / m.sum()
    if kappa0 is not None:
        vals = [float(x) for x in str(kappa0).split(",")]
        pairs = contrast_pairs(k1)
        if len(vals) != len(pairs):
            raise ValueError(f"--kappa0 needs {len(pairs)} values (upper triangle, row by row).")
        k0 = np.zeros((k1, k1))
        for (a, b), v in zip(pairs, vals):
            k0[a, b] = k0[b, a] = v
    elif "kappa0_matrix" in p:
        k0 = np.asarray(p["kappa0_matrix"], float).reshape(k1, k1)
    elif "kappa0" in p:
        k0 = np.atleast_2d(np.asarray(p["kappa0"], float)).reshape(k1, k1)
    else:
        raise ValueError("Give --params (compose's .admixed_ld.json) or --kappa0.")
    return m, k0


def _kernel(ref, sources, k0, curve=None, generations=None, decay_per_cm=None):
    """The kernel as `compose` builds it from the same inputs."""
    k1 = len(sources) - 1
    if sum(x is not None for x in (curve, generations, decay_per_cm)) != 1:
        raise ValueError("Give exactly one source for the decay: --curve, --admixture-generations or --decay-per-cm.")
    if curve is not None:
        cdf = pd.read_csv(f"{curve}.curve.tsv", sep="\t")
        info = json.load(open(f"{curve}.curve.json"))
        if list(info.get("sources", sources)) != list(sources):
            raise ValueError(f"The curve was made with sources {info.get('sources')}, not {sources}.")
        d = cdf.d_cm.to_numpy()
        kc = np.zeros((d.size, k1, k1))
        if k1 == 1:
            kc[:, 0, 0] = cdf.kernel.to_numpy()
            inter = np.array([[info["inter_kernel"]]])
            lam = np.full(k1 + 1, info["fit_2_to_20cm"]["lam"])
        else:
            for a, b in contrast_pairs(k1):
                kc[:, a, b] = kc[:, b, a] = cdf[f"kernel_{sources[a]}_{sources[b]}"].to_numpy()
            inter = np.asarray(info["inter_matrix"])
            lam = np.array([info["fit_per_source"][s]["lam"] for s in sources])
        return KernelMatrix.from_curve(d, kc, inter, k0, lam)
    lam = float(generations) / 100.0 if generations is not None else float(decay_per_cm)
    return KernelMatrix.from_decay(k0, lam, ref.genome_cm, k1 + 1, source="date" if generations else "decay")


def _band(z, loc):
    """Pairs (a, b) of positions within loc (sorted local indices) that the reference stores, and their slice."""
    p = loc.size
    pos = -np.ones(z["snp"].size, np.int64)
    pos[loc] = np.arange(p)
    ip, ix = z["indptr"], z["indices"]
    lo, hi = int(ip[loc[0]]), int(ip[loc[-1] + 1])
    rows = np.repeat(np.arange(loc[0], loc[-1] + 1), np.diff(ip[loc[0]:loc[-1] + 2]))
    cols = ix[lo:hi].astype(np.int64)
    a, b = pos[rows], pos[cols]
    use = (a >= 0) & (b >= 0)
    return a[use], b[use], lo, hi, use


def taper_scales(ref, si, z, taus=_TAUS, max_pairs=2_000_000):
    """tau_s (cM) per source minimizing the unbiased squared-error risk of the tapered panel correlations over all pairs
    of the chromosome file z; returns (tau, risk at tau relative to no taper)."""
    ip, ix, cm = z["indptr"], z["indices"].astype(np.int64), z["cm"]
    rows = np.repeat(np.arange(z["snp"].size), np.diff(ip))
    d = np.abs(cm[ix] - cm[rows])
    out_t, out_r = [], []
    for s in si:
        n = ref.n[ref.names[s]]
        # the variances of the people behind the covariances (a derived source with an anchor panel's LD keeps them as
        # vld), and for a source that stores each covariance's sampling variance (covvar: derived by inversion, masked
        # admixed haplotypes) U = c^2 - covvar, as in compose
        v = (z[f"vld_{s}"] if f"vld_{s}" in z.files else z[f"v_{s}"]).astype(np.float64)
        c = z[f"cov_{s}"].astype(np.float64)
        if f"covvar_{s}" in z.files:
            U = c * c - z[f"covvar_{s}"].astype(np.float64)
        else:
            U = unbiased_squared_covariance(c, z[f"m22_{s}"].astype(np.float64), v[rows] * v[ix], n)
        ok = (v[rows] > 0) & (v[ix] > 0) & np.isfinite(U)
        if ok.sum() > max_pairs:                                     # a fixed random subset keeps this fast
            idx = np.flatnonzero(ok)
            ok = np.zeros_like(ok)
            ok[np.random.default_rng(0).choice(idx, max_pairs, replace=False)] = True
        w = 1.0 / (v[rows][ok] * v[ix][ok])
        Uo, var, do = U[ok], c[ok] ** 2 - U[ok], d[ok]
        risk = np.array([np.sum(w * ((1 - np.exp(-do / t)) ** 2 * Uo + np.exp(-do / t) ** 2 * var)) for t in taus])
        none = float(np.sum(w * var))
        k = int(np.argmin(risk))
        out_t.append(float(taus[k]))
        out_r.append(float(risk[k] / none) if none > 0 else np.nan)
    return np.array(out_t), np.array(out_r)


def _align_sumstats(path, z, loc, keep_ambiguous, log):
    """The GWAS z-scores of the region's SNPs, on the matrix's counted allele: matched by SNP ID and alleles, the sign
    flipped where the GWAS counts the other allele; strand-ambiguous SNPs are left out unless keep_ambiguous. Returns
    (the region rows kept, z, N)."""
    from ..sumstats.sumstats import Sumstats
    from ._io import match_variants
    ss = Sumstats.from_file(path, name=str(path), compute_diagnostics=False)
    tv = pd.DataFrame({"SNP": ss.snps, "CHR": "0", "BP": 0, "COUNTED": pd.Series(ss.a1).str.upper().to_numpy(),
                       "OTHER": pd.Series(ss.a2).str.upper().to_numpy()})
    rv = pd.DataFrame({"SNP": z["snp"][loc].astype(str), "CHR": "0", "BP": 0, "COUNTED": z["counted"][loc].astype(str),
                       "OTHER": z["other"][loc].astype(str)})
    rep = {}
    ti, ri, flip = match_variants(tv, rv, how="id", drop_ambiguous=not keep_ambiguous, report=rep)
    if ri.size < 2:
        raise ValueError(f"Fewer than two of the region's SNPs are in {path} with matching alleles.")
    o = np.argsort(ri)
    ti, ri, flip = ti[o], ri[o], flip[o]
    zz = np.where(flip, -1.0, 1.0) * ss.z[ti]
    log(f"[admixed-ld] matrix: {ri.size} of the region's {loc.size} SNPs in {path} ({int(flip.sum())} on the other "
        f"allele, flipped; {rep['n_ambiguous_dropped']} strand-ambiguous left out, {rep['n_allele_mismatch']} with "
        "other alleles)")
    return ri, zz, ss.n[ti]


def compose_matrix(ref_path, out, *, region=None, snps=None, sources=None, params=None, curve=None, generations=None,
                   decay_per_cm=None, mean_ancestry=None, kappa0=None, taper="auto", admixture=True,
                   allow_flat_kernel_join=False, sumstats=None, keep_ambiguous=False, error_variance=False, n=None,
                   log=print):
    """Write out.ld (text, one row per SNP), out.ld.npz (R, float64) and out.snps.tsv for a region 'CHR:START-END'
    (base pairs, inclusive) or a file of SNP IDs on one chromosome.

    sumstats: the GWAS summary statistics; the matrix is then restricted to the region's SNPs in them, and out.z.tsv
    holds their z-scores on the matrix's counted allele. error_variance: also out.ld_var.npz and out.ld_var (text), V,
    the predicted variance of the error of every entry of R for a GWAS of n people (finemap.py; n defaults to the
    summary statistics' median N), for the residual variance of the second SuSiE fit (finemap-variance)."""
    t0 = time.perf_counter()
    ref = Reference(ref_path)
    sources = list(sources or ref.names[:2])
    if len(sources) < 2 or len(set(sources)) != len(sources):
        raise ValueError("The matrix needs two or more distinct sources.")
    si = [ref.source_index(s) for s in sources]
    k1 = len(sources) - 1
    m, k0 = _parameters(params, sources, mean_ancestry, kappa0)
    kern = _kernel(ref, sources, k0, curve, generations, decay_per_cm)
    flags = check_kernel_join(kern, allow_flat_kernel_join, log)
    if (region is None) == (snps is None):
        raise ValueError("Give exactly one of --region and --snps.")
    if region is not None:
        c, rng = str(region).split(":")
        lo_bp, hi_bp = (int(x.replace(",", "")) for x in rng.split("-"))
        c = c.replace("chr", "")
        if c not in ref.chromosomes:
            raise ValueError(f"Chromosome {c} is not in the reference.")
        z = ref.load(c)
        loc = np.flatnonzero((z["bp"] >= lo_bp) & (z["bp"] <= hi_bp))
    else:
        ids = pd.read_csv(snps, sep=r"\s+", header=None, usecols=[0], dtype=str)[0].to_numpy()
        hit = None
        for c in ref.chromosomes:
            zc = ref.load(c)
            idx = pd.Index(zc["snp"].astype(str)).get_indexer(ids)
            if (idx >= 0).any():
                if hit is not None:
                    raise ValueError("--snps lists SNPs on more than one chromosome.")
                hit, z, loc = c, zc, np.sort(idx[idx >= 0])
        if hit is None:
            raise ValueError("None of the SNPs in --snps is in the reference.")
        c = hit
        if loc.size < ids.size:
            log(f"[admixed-ld] matrix: {ids.size - loc.size} of {ids.size} listed SNPs are not in the reference")
    if loc.size < 2:
        raise ValueError("Fewer than two reference SNPs in the region.")
    zs = None
    if sumstats is not None:
        rows_kept, zz, nn = _align_sumstats(sumstats, z, loc, keep_ambiguous, log)
        loc = loc[rows_kept]
        zs = (zz, nn)
    V = np.column_stack([z[f"v_{s}"][loc].astype(np.float64) for s in si])
    keep = V.sum(axis=1) > 0
    if not keep.all():
        log(f"[admixed-ld] matrix: {int((~keep).sum())} SNPs without variance in every panel are left out")
        loc, V = loc[keep], V[keep]
        if zs is not None:
            zs = (zs[0][keep], zs[1][keep])
    p = loc.size
    a, b, lo, hi, use = _band(z, loc)
    complete = a.size == p * (p - 1) // 2
    if not complete:
        log(f"[admixed-ld] matrix: the region is wider than the reference window ({ref.manifest['window_kb']} kb); "
            "panel covariances of pairs beyond it are zero")
    P = np.column_stack([z[f"p_{s}"][loc].astype(np.float64) for s in si])
    D = P[:, :-1] - P[:, -1:]
    cm = z["cm"][loc].astype(np.float64)
    d = np.abs(cm[:, None] - cm[None, :])
    if taper == "none" or taper is None:
        tau = np.full(len(si), np.inf)
        tau_note = "none"
    elif taper == "auto":
        tau, rr = taper_scales(ref, si, z)
        tau_note = {s: dict(tau_cm=float(t), risk_ratio=float(r)) for s, t, r in zip(sources, tau, rr)}
    else:
        tau = np.full(len(si), float(taper))
        tau_note = float(taper)
    C = np.zeros((p, p))
    pieces = dict(C_s=[], M_s=[], w0=[], T=[])                       # for the predicted error (error_variance)
    for n_, s in enumerate(si):
        Cs = np.zeros((p, p))
        cv = z[f"cov_{s}"][lo:hi][use].astype(np.float64)
        if f"vld_{s}" in z.files:
            # a derived source whose window LD comes from a panel: the panel's correlations on the source's variances
            # (its covariances carry the panel's variances; with the source's on the diagonal the matrix need not be
            # positive semidefinite)
            vl = z[f"vld_{s}"][loc].astype(np.float64)
            g = np.where(vl > 0, np.sqrt(np.maximum(V[:, n_], 0) / np.where(vl > 0, vl, 1.0)), 0.0)
            cv = cv * g[a] * g[b]
        Cs[a, b] = cv
        Cs[b, a] = cv
        Cs[np.arange(p), np.arange(p)] = V[:, n_]
        T = np.exp(-d / tau[n_]) if np.isfinite(tau[n_]) else 1.0
        C += m[n_] * np.exp(-kern.lam[n_] * d) * T * Cs
        if error_variance:
            if f"vld_{s}" in z.files or f"covvar_{s}" in z.files:
                raise ValueError(f"--error-variance covers panels only; {sources[n_]} is a derived source.")
            Ms = np.full((p, p), np.nan)
            mm = z[f"m22_{s}"][lo:hi][use].astype(np.float64)
            Ms[a, b], Ms[b, a] = mm, mm
            pieces["C_s"].append(Cs)
            pieces["M_s"].append(Ms)
            pieces["w0"].append(m[n_] * np.exp(-kern.lam[n_] * d))
            pieces["T"].append(T * np.ones_like(d))
    psd_note = None
    if admixture:
        Kd = kern(d)                                                  # p x p x k1 x k1
        B = Kd.transpose(0, 2, 1, 3).reshape(p * k1, p * k1)
        B = 0.5 * (B + B.T)
        w, Uv = np.linalg.eigh(B)
        if w.min() < -1e-10 * max(w.max(), 1e-300):
            Kd = ((Uv * np.clip(w, 0, None)) @ Uv.T).reshape(p, k1, p, k1).transpose(0, 2, 1, 3)
            psd_note = f"kernel matrix projected onto the PSD cone (smallest eigenvalue {w.min():.3g})"
            log(f"[admixed-ld] matrix: {psd_note}")
        C += np.einsum("ja,jkab,kb->jk", D, Kd, D)
    s_ = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = C / s_[:, None] / s_[None, :]
    np.fill_diagonal(R, 1.0)
    R = 0.5 * (R + R.T)
    ev = float(np.linalg.eigvalsh(R).min())
    snp = pd.DataFrame({"SNP": z["snp"][loc].astype(str), "CHR": c, "BP": z["bp"][loc], "CM": cm,
                        "COUNTED": z["counted"][loc].astype(str), "OTHER": z["other"][loc].astype(str)})
    for i, s in enumerate(sources):
        snp[f"p_{s}"] = P[:, i]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    snp.to_csv(out + ".snps.tsv", sep="\t", index=False, float_format="%.6g")
    np.savez(out + ".ld.npz", R=R, snps=snp.SNP.to_numpy())
    np.savetxt(out + ".ld", R, fmt="%.6f")
    if zs is not None:
        snp.assign(Z=zs[0], N=zs[1])[["SNP", "CHR", "BP", "COUNTED", "OTHER", "Z", "N"]].to_csv(
            out + ".z.tsv", sep="\t", index=False, float_format="%.8g")
    ev_info = None
    if error_variance:
        from .finemap import predict_error_variance
        N = float(n) if n is not None else (float(np.median(zs[1])) if zs is not None else None)
        if N is None or not N > 0:
            raise ValueError("--error-variance needs the GWAS sample size: --n, or --sumstats with an N column.")
        ns = np.array([ref.n[sources[i]] for i in range(len(si))], np.float64)
        pr = predict_error_variance(pieces["C_s"], pieces["M_s"], V, P, m, pieces["w0"], pieces["T"], np.diag(C).copy(),
                                    R, ns, N)
        off = ~np.eye(p, dtype=bool)
        np.savez(out + ".ld_var.npz", V=pr["V"], n=N, snps=snp.SNP.to_numpy(),
                 **{k: v for k, v in pr["parts"].items()})
        np.savetxt(out + ".ld_var", pr["V"], fmt="%.6e")
        ev_info = dict(n=N, panel_n=ns.tolist(), mean_offdiagonal=float(pr["V"][off].mean()),
                       parts_mean={k: float(v[off].mean()) for k, v in pr["parts"].items()},
                       files=[out + ".ld_var.npz", out + ".ld_var"],
                       definition="V_jk: predicted variance of the error of R_jk from the panels' sampling (Hardy-"
                                  "Weinberg moments), the taper's squared bias and the GWAS sample's own 1/N term")
    info = dict(format="summit-admixed-ld-matrix", version=1, sources=sources, n_snps=int(p), chrom=str(c),
                mean_ancestry=m.tolist(), kappa0=k0.tolist(), kernel=kern.describe(), taper=tau_note,
                admixture_term=bool(admixture), complete_band=bool(complete), min_eigenvalue=ev, psd_note=psd_note,
                reference=dict(sources=ref.manifest["sources"], window_kb=ref.manifest["window_kb"]), flags=flags,
                sumstats=None if zs is None else dict(file=str(sumstats), n_snps=int(p), keep_ambiguous=bool(keep_ambiguous),
                                                     file_z=out + ".z.tsv"),
                error_variance=ev_info)
    with open(out + ".matrix.json", "w") as f:
        json.dump(info, f, indent=1)
    log(f"[admixed-ld] matrix: {p} SNPs on chr{c}, taper {tau_note}, smallest eigenvalue {ev:.3g}; wrote {out}.ld, "
        f"{out}.ld.npz, {out}.snps.tsv ({time.perf_counter() - t0:.1f} s)")
    return R, snp, info
