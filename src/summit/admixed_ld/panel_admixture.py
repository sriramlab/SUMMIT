"""Admixture LD inside the source panels (`summit admixed-ld reference --remove-panel-admixture`, `clean-panels`).

A source panel drawn from a biobank is rarely free of admixture: a few percent of ancestry from another source, varying
between people and along the genome, gives the panel its own admixture LD. In the panel's covariances it appears as
D_j' Kt_p(d) D_k, the same form as in an admixed cohort but small (D_j the contrasts of the reference sources against
the baseline, Kt_p(d) the covariance of the panel's local-ancestry dosages of the non-baseline sources at distance d),
and in its dosage variances as D_j' Kt_p(0) D_j. Composition adds the cohort's admixture LD to the within-source LD
of the panels, so a panel's own admixture LD would be counted twice at the distances of the reference band; in
UK Biobank target 1 (African panel, ancestry SD 0.03) it inflated the composed LD at 1-3 cM by 13%.

The second count does not depend on who is in the target. Beyond 2 cM the kernel is the target's own curve, its
covariance along D at each distance, and kappa_res(0) comes from its own standard errors, so the composition's
admixture term already holds all of the target's admixture LD, that of members drawn from the panel's own population
included, at their share of the target (the GWAS PCs remove their global-ancestry part from the target and from its
curve alike). The panel's copy is added on top inside the band with weight pi_s(d) = m_s exp(-lam d). Cleaning is
optional and off by default. It matters when the panel's kernel passes the gate below and its duplicated share of the
target's kernel in the band, sum_s pi_s(d) Kt_s(d) / Kt_T(d) at 2 cM, is not negligible (UK Biobank African panel of
4,905: 9% for admixed members only, 15% when 40% of the target are unadmixed members of the panel's population, 97% for
a target of those members alone; simulated pure panels stay below the gate). It would remove real LD only if the
composition's kernel were not measured on the target along D, for example a kernel from local-ancestry calls that do
not see the panel-like tracts.

Kt_p is the panel's weighted-LD curve (`genotype_curve` on the panel, without covariates), fitted entry by entry over
lo-hi cM (default 3-40) as A + H exp(-lam d) with lam in [0.005, 0.5] per cM (admixture 0.5 to 50 generations ago)
and A, H >= 0 on the diagonal (a panel is not adjusted for covariates, so its ancestry variance is not negative).
Below lo the measured curve also carries within-source LD of the weighted sums, so the fit is extrapolated there.
A panel is cleaned only when its Kt_p(0) is clearly above zero (at least 3 jackknife standard errors, leaving one
chromosome out), so that a panel without admixture keeps its covariances. The cleaned covariance is
c - D_j' Kt_p(d) D_k, with m22 changed so that the sampling variance of the covariance is kept (m22 - c^2 + c'^2);
the variance becomes v - D_j' Kt_p(0) D_j. Frequencies are kept: a panel's mean admixture
shortens every contrast by the same factor, which the cohort's curve and the standard errors absorb.

The same pass records each panel's own background LD (`panel_background` in the manifest): the cross-person LD of its
members among the least differentiated SNPs, less its admixture part and the null, by distance (the estimator of
`curve --background`, without covariates). It is the panel's cryptic relatedness and the structure among its members,
which its unbiased squared covariances hold on top of the population's LD, E U = c^2 + v_j v_k S(d) / (n (n - 3)).
The cohort's own background measurement holds the cohort's version, so when compose applies it at every distance it
removes the panels' copy inside the band (compose.SmoothBackground), as cleaning removes the panels' admixture LD.

A derived source whose window LD comes from a panel (`derive --ld panel`, its anchors) is cleaned the same way when its
anchor panel is given: the anchors' admixture LD is removed from its covariances and from the panel variances behind
them (vld; the source's own variances v come from the cohort's frequencies and are kept), and the anchors' background
is recorded. Without the record compose would count the anchors' background twice at the distances of the band (7 Nov
2026: 5-6% too much LD mass in the coalescent testbed).
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

from .model import contrast_pairs
from .reference import Reference

LAM_BOUNDS = (0.005, 0.5)


def fit_panel_kernel(d, k, w=None, *, nonnegative=True, lo=3.0, hi=40.0):
    """A + H exp(-lam d) fitted to a weighted-LD curve over lo-hi cM (weights w, e.g. the curve's denominators).
    Returns dict(A, H, lam)."""
    from scipy.optimize import least_squares
    d, k = np.asarray(d, np.float64), np.asarray(k, np.float64)
    w = np.ones_like(d) if w is None else np.asarray(w, np.float64)
    s = (d >= lo) & (d <= hi) & np.isfinite(k) & np.isfinite(w) & (w > 0)
    if s.sum() < 4:
        raise ValueError(f"The panel curve has fewer than 4 points between {lo} and {hi} cM.")
    d, k, sw = d[s], k[s], np.sqrt(w[s] / w[s].mean())
    # diagonal entries: a panel has no covariates, so the asymptote (its global-ancestry variance) and H are >= 0
    lower = [0.0 if nonnegative else -np.inf, 0.0 if nonnegative else -np.inf, LAM_BOUNDS[0]]
    upper = [np.inf, np.inf, LAM_BOUNDS[1]]
    best = None
    for lam0 in (0.01, 0.05, 0.2):
        h0 = max(float(k[0]), 0.0) if nonnegative else float(k[0])
        r = least_squares(lambda t: sw * (t[0] + t[1] * np.exp(-t[2] * d) - k), [0.0, h0, lam0], bounds=(lower, upper))
        if best is None or r.cost < best.cost:
            best = r
    A, H, lam = (float(x) for x in best.x)
    return dict(A=A, H=H, lam=lam)


class PanelKernel:
    """Kt_p(d) of one panel, (K-1) x (K-1), from the entry-wise fits."""

    def __init__(self, fits: dict, k1: int):
        self.k1 = k1
        self.fits = fits

    def __call__(self, d):
        d = np.asarray(d, np.float64)
        out = np.zeros(d.shape + (self.k1, self.k1))
        for (a, b), f in self.fits.items():
            val = f["A"] + f["H"] * np.exp(-f["lam"] * d)
            out[..., a, b] = out[..., b, a] = val
        return out


def _kernel_from_sums(num, den, cell_cm):
    """Distances and Kt per lag (L x k1 x k1) from the lag sums of `genotype_curve`, with the k1 = 1 weights."""
    from .curve import _solve_kernel
    lags = [g for g in range(1, num.shape[0]) if np.any(den[g])]
    k = np.stack([_solve_kernel(num[g], den[g]) for g in lags])
    return np.array(lags) * float(cell_cm), k, np.array([den[g, 0, 0, 0, 0] for g in lags])


def _fit_entries(d, k, w, k1, lo, hi):
    fits = {}
    for a, b in contrast_pairs(k1):
        fits[(a, b)] = fit_panel_kernel(d, k[:, a, b], w if k1 == 1 else None, nonnegative=(a == b), lo=lo, hi=hi)
    return fits


def panel_curve_kernel(chr_sums, *, lo=3.0, hi=40.0):
    """The PanelKernel of a panel from its per-chromosome curve sums (`genotype_curve(..., per_chromosome=True)`), with
    leave-one-chromosome-out jackknife standard errors of Kt_p(0). Returns (kernel, se of Kt_p(0))."""
    z = np.load(chr_sums)
    num, den, cell_cm = z["num"], z["den"], float(z["cell_cm"])
    k1 = num.shape[2]
    fits = _fit_entries(*_kernel_from_sums(num.sum(0), den.sum(0), cell_cm), k1, lo, hi)
    kern = PanelKernel(fits, k1)
    C = num.shape[0]
    jk = []
    for c in range(C):
        f = _fit_entries(*_kernel_from_sums(num.sum(0) - num[c], den.sum(0) - den[c], cell_cm), k1, lo, hi)
        jk.append(PanelKernel(f, k1)(np.zeros(1))[0])
    jk = np.stack(jk)
    se = np.sqrt((C - 1) / C * ((jk - jk.mean(0)) ** 2).sum(0)) if C > 1 else np.full((k1, k1), np.inf)
    return kern, se


def remove_panel_admixture(ref_path, panels: dict, out=None, *, match="position", lo=3.0, hi=40.0, z_min=3.0,
                           threads=1, step=4096, native=True, background=True, background_snps=20000,
                           background_quantile=0.2, log=print):
    """Remove each listed panel's admixture LD from the reference at ref_path (in place when out is None, otherwise
    into a copy at out). panels: {source name: PLINK prefix of the panel the reference was built from}. A panel is
    cleaned only if some diagonal entry of its Kt_p(0) is at least z_min jackknife standard errors (leave one
    chromosome out) above zero; otherwise the fit is noise and the panel is left as it is (recorded in the manifest).
    With background (the default) the same pass also measures the panel's own background LD (the cross-person LD of
    its members among the least differentiated SNPs: their cryptic relatedness and the structure among them, which the
    panel's unbiased squared covariances hold; `curve --background` on the panel, between chromosomes on
    background_snps SNPs) and records it in the manifest as `panel_background`. Compose removes it inside the band
    when it applies the cohort's own background at every distance (compose.SmoothBackground). It is recorded whether
    or not the panel's admixture is removed."""
    from .compose import panel_background_record
    from .curve import genotype_curve
    if out is not None and os.path.abspath(out) != os.path.abspath(ref_path):
        if os.path.exists(os.path.join(out, "manifest.json")):
            raise ValueError(f"{out} already holds a reference.")
        shutil.copytree(ref_path, out, dirs_exist_ok=True)
        ref_path = out
    ref = Reference(ref_path)
    names = list(ref.names)
    k1 = len(names) - 1
    man = ref.manifest
    kernels = {}
    for s, prefix in panels.items():
        i = ref.source_index(s)
        src = man["sources"][i]
        if "panel_admixture" in src:
            raise ValueError(f"The panel admixture of {s} was already removed from {ref_path}.")
        if src.get("derived") and src.get("ld") != "panel":
            raise ValueError(f"{s} is a derived source whose window LD comes from the cohort (ld '{src.get('ld')}'); "
                             "only panels with genotypes, and derived sources with a panel's LD (their anchors), can "
                             "be cleaned.")
        cprefix = os.path.join(ref_path, f"panel_admixture_{s}")
        log(f"[admixed-ld] panel admixture: weighted-LD curve of the {s} panel ({prefix})")
        _, info = genotype_curve(prefix, ref_path, cprefix, sources=names, covar=None, match=match, max_cm=hi,
                                 cells_cm=0, per_chromosome=True, step=step, threads=threads, native=native,
                                 background=background, background_quantile=background_quantile,
                                 background_between_snps=background_snps, log=log)
        if background:
            src["panel_background"] = panel_background_record(info, pd.read_csv(f"{cprefix}.background.tsv", sep="\t"),
                                                              file=os.path.basename(cprefix) + ".background.tsv")
        kern, se = panel_curve_kernel(f"{cprefix}.curve_chr.npz", lo=lo, hi=hi)
        k0 = kern(np.zeros(1))[0]
        zdiag = np.diag(k0) / np.maximum(np.diag(se), 1e-300)
        removed_ok = bool(np.max(zdiag) >= z_min)
        if removed_ok:
            kernels[i] = kern
        src["panel_admixture"] = dict(removed=removed_ok, fits={f"{names[a]},{names[b]}": f for (a, b), f in
                                                               kern.fits.items()},
                                      kappa0=k0.tolist(), kappa0_se=se.tolist(), z=zdiag.tolist(), z_min=z_min,
                                      fit_cm=[lo, hi], curve=os.path.basename(cprefix),
                                      ancestry_cov=info["theta"].get("cov", info["theta"].get("var")))
        log(f"[admixed-ld] panel admixture of {s}: Kt_p(0) = {np.array2string(k0, precision=5)} (jackknife SE "
            f"{np.array2string(se, precision=5)}; z {np.array2string(zdiag, precision=1)}), fits "
            + "; ".join(f"{names[a]}/{names[b]} A {f['A']:.2e} H {f['H']:.2e} lam {f['lam']:.3f}"
                        for (a, b), f in kern.fits.items())
            + ("" if removed_ok else f"; not removed (z below {z_min})"))
    _apply_kernels(ref_path, ref, kernels, names, log)
    with open(os.path.join(ref_path, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def _apply_kernels(ref_path, ref, kernels, names, log):
    """Subtract D_j' Kt_p(d) D_k from the band covariances (and D_j' Kt_p(0) D_j from the variances of the people
    behind them: the panel variances vld of a derived source with a panel's LD, otherwise v) of each source whose
    kernel is given, keeping the sampling variance (m22 - cov^2) of every covariance. Rewrites the chromosome files of
    ref_path in place."""
    removed = {i: 0.0 for i in kernels}
    for c in ref.chromosomes:
        path = os.path.join(ref_path, f"chr{c}.npz")
        with np.load(path, allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}
        P = np.column_stack([z[f"p_{t}"] for t in range(len(names))])
        D = P[:, :-1] - P[:, -1:]
        ip, ix = z["indptr"], z["indices"].astype(np.int64)
        jl = np.repeat(np.arange(z["snp"].size), np.diff(ip))
        d = np.abs(z["cm"][ix] - z["cm"][jl])
        for i, kern in kernels.items():
            a = np.einsum("pa,pab,pb->p", D[jl], kern(d), D[ix])
            cov = z[f"cov_{i}"].astype(np.float64)
            new = cov - a
            z[f"m22_{i}"] = (z[f"m22_{i}"].astype(np.float64) - cov * cov + new * new).astype(z[f"m22_{i}"].dtype)
            z[f"cov_{i}"] = new.astype(z[f"cov_{i}"].dtype)
            vk = f"vld_{i}" if f"vld_{i}" in z else f"v_{i}"
            z[vk] = z[vk] - np.einsum("ja,ab,jb->j", D, kern(np.zeros(1))[0], D)
            removed[i] += float(np.abs(a).sum())
        np.savez(path, **z)
    log("[admixed-ld] panel admixture removed: " + ", ".join(f"{names[i]} (sum |change| {v:.4g})"
                                                             for i, v in removed.items()) + f"; reference {ref_path}")
    return removed


def kernels_from_manifest(manifest, names):
    """{source index: PanelKernel} from the panel_admixture records of a cleaned reference's manifest, for a reference
    with the same sources in the same order (names). Sources whose fit was not removed (z below z_min) are left out."""
    src_names = [s["name"] for s in manifest["sources"]]
    if src_names != list(names):
        raise ValueError(f"The fits are for sources {src_names}, not {list(names)} (same names and order needed).")
    k1 = len(names) - 1
    kernels = {}
    for i, s in enumerate(manifest["sources"]):
        pa = s.get("panel_admixture")
        if not pa or not pa.get("removed"):
            continue
        fits = {}
        for key, f in pa["fits"].items():
            a_name, b_name = key.split(",")
            fits[(names.index(a_name), names.index(b_name))] = dict(A=float(f["A"]), H=float(f["H"]),
                                                                     lam=float(f["lam"]))
        kernels[i] = PanelKernel(fits, k1)
    return kernels


def apply_panel_admixture(ref_path, fits_from, out=None, *, log=print):
    """Remove the panels' admixture LD with kernels fitted elsewhere: fits_from is a reference cleaned by
    remove_panel_admixture (its directory or manifest.json), made from the same panels. For a reference converted or
    subset from those panels' reference (another SNP set or build) where the panel genotypes are not available, e.g.
    inside a data enclave. In place when out is None, otherwise into a copy at out."""
    src_man = fits_from if not os.path.isdir(fits_from) else os.path.join(fits_from, "manifest.json")
    with open(src_man) as f:
        fm = json.load(f)
    if out is not None and os.path.abspath(out) != os.path.abspath(ref_path):
        if os.path.exists(os.path.join(out, "manifest.json")):
            raise ValueError(f"{out} already holds a reference.")
        shutil.copytree(ref_path, out, dirs_exist_ok=True)
        ref_path = out
    ref = Reference(ref_path)
    names = list(ref.names)
    man = ref.manifest
    if any("panel_admixture" in s for s in man["sources"]):
        raise ValueError(f"The panel admixture was already removed from {ref_path}.")
    kernels = kernels_from_manifest(fm, names)
    for i, s in enumerate(fm["sources"]):
        if "panel_admixture" in s:
            man["sources"][i]["panel_admixture"] = dict(s["panel_admixture"], applied_from=os.path.abspath(src_man))
        if "panel_background" in s:                # a property of the panel's people, carried with the fits
            man["sources"][i]["panel_background"] = dict(s["panel_background"], applied_from=os.path.abspath(src_man))
            log(f"[admixed-ld] panel background of {names[i]} from {src_man}: between chromosomes "
                f"{s['panel_background']['between']['value']:.4g} per SNP pair, n {s['panel_background']['n']}")
    for i in kernels:
        log(f"[admixed-ld] panel admixture of {names[i]} from {src_man}: "
            + "; ".join(f"{k} A {f['A']:.2e} H {f['H']:.2e} lam {f['lam']:.3f}" for k, f in
                        fm["sources"][i]["panel_admixture"]["fits"].items()))
    _apply_kernels(ref_path, ref, kernels, names, log)
    with open(os.path.join(ref_path, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man
