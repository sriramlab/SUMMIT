"""Simulated source panels and an admixed cohort for the `summit admixed-ld` tests.

The simulator is the one of the public example (example/prepare_admixed_ld_inputs.py), so the tests and the example
share one model. This module adds the exact in-sample LD scores that the composed ones are compared with.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

_spec = spec_from_file_location("prepare_admixed_ld_inputs",
                                Path(__file__).resolve().parents[1] / "example" / "prepare_admixed_ld_inputs.py")
_example = module_from_spec(_spec)
_spec.loader.exec_module(_example)
simulate, write_bed, principal_components, gwas = (_example.simulate, _example.write_bed,
                                                   _example.principal_components, _example.gwas)


def insample_ldscores(G, C, cls, K):
    """SUMMIT's V-weighted in-sample LD scores computed exactly: genotypes standardized, adjusted for the covariates,
    correlations of the adjusted genotypes, partner weights V_k = (1 - R2_k)(N - 1)/(N - p - 1), and the noise
    correction sum_k V_k / (N - p - 1). Returns (L, V)."""
    n = G.shape[0]
    X = (G - G.mean(0)) / G.std(0, ddof=1)
    Cs = (C - C.mean(0)) / C.std(0, ddof=1)
    Q, _ = np.linalg.qr(Cs)
    p = Q.shape[1]
    Xr = X - Q @ (Q.T @ X)
    raw = np.einsum("ij,ij->j", X, X)
    ss = np.einsum("ij,ij->j", Xr, Xr)
    V = ss / raw * (n - 1) / (n - p - 1)
    Xs = Xr / np.sqrt(ss / (n - p - 1))
    R = Xs.T @ Xs / (n - p - 1)
    A = np.zeros((G.shape[1], K))
    A[np.arange(G.shape[1]), cls] = V
    L = (R * R) @ A - A.sum(0)[None, :] / (n - p - 1)
    return L, V


def _haplotypes_k(rng, founders, cm, n_hap, copy_rate, theta_h=None, source=None, break_rate=0.0):
    """Haplotypes from K sources' founders (list of F x M arrays). Unadmixed (source=s) or admixed: breakpoints since
    admixture at break_rate per cM, each segment drawing its ancestry from theta_h (n_hap x K) and a fresh founder;
    within a segment the copied founder switches at copy_rate per cM. Returns (haplotypes, local ancestry)."""
    K, M = len(founders), cm.size
    F = founders[0].shape[0]
    cw = None if theta_h is None else np.cumsum(theta_h, axis=1)
    draw = lambda idx: (np.full(idx.size, source) if theta_h is None
                        else (rng.random(idx.size)[:, None] > cw[idx]).sum(1))
    out = np.empty((n_hap, M), dtype=np.int8)
    la = np.empty((n_hap, M), dtype=np.int8)
    allh = np.arange(n_hap)
    a, f = draw(allh), rng.integers(0, F, n_hap)
    for j in range(M):
        if j > 0:
            dcm = cm[j] - cm[j - 1]
            if dcm < 0:
                brk = np.ones(n_hap, bool)
                sw = np.ones(n_hap, bool)
            else:
                brk = (rng.random(n_hap) < 1 - np.exp(-break_rate * dcm)) if theta_h is not None else np.zeros(n_hap, bool)
                sw = rng.random(n_hap) < 1 - np.exp(-copy_rate * dcm)
            nb = np.flatnonzero(brk)
            a[nb] = draw(nb)
            f = np.where(sw | brk, rng.integers(0, F, n_hap), f)
        for k in range(K):
            hk = a == k
            out[hk, j] = founders[k][f[hk], j]
        la[:, j] = a
    return out, la


def simulate_k(seed=1, n_chr=3, snps_per_chr=400, chr_cm=60.0, n_panel=1000, n_cohort=3000, m=(0.5, 0.3, 0.2),
               conc=10.0, generations=8.0, n_founders=40, copy_rate=3.0):
    """Panels of K sources and an admixed cohort with Dirichlet(conc m) ancestry (founder-copying model)."""
    rng = np.random.default_rng(seed)
    K = len(m)
    chrom = np.repeat(np.arange(1, n_chr + 1), snps_per_chr)
    cm = np.concatenate([np.sort(rng.uniform(0, chr_cm, snps_per_chr)) for _ in range(n_chr)])
    bp = np.rint(cm * 1e6).astype(np.int64) + 1
    M = cm.size
    base = rng.uniform(0.05, 0.95, M)
    freqs = [base] + [np.clip(base + rng.normal(0, 0.25, M), 0.03, 0.97) for _ in range(K - 1)]
    founders = [(rng.random((n_founders, M)) < p[None, :]).astype(np.int8) for p in freqs]
    geno = lambda h: (h[0::2] + h[1::2]).astype(np.float64)
    panels = [geno(_haplotypes_k(rng, founders, cm, 2 * n_panel, copy_rate, source=s)[0]) for s in range(K)]
    theta = rng.dirichlet(conc * np.asarray(m), n_cohort)
    hC, la = _haplotypes_k(rng, founders, cm, 2 * n_cohort, copy_rate, theta_h=np.repeat(theta, 2, axis=0),
                           break_rate=generations / 100.0)
    variants = pd.DataFrame({"CHR": chrom.astype(str), "SNP": [f"sim{j:05d}" for j in range(M)], "CM": cm, "BP": bp,
                             "A1": np.where(rng.random(M) < 0.5, "A", "C"), "A2": "G"})
    return dict(variants=variants, panels=panels, cohort=geno(hC), theta=theta, m=np.asarray(m), generations=generations)
