"""Simulated source panels and an admixed cohort for the `summit admixed-ld` tests.

The simulator is the one of the public example (example/prepare_admixed_ld_inputs.py), so the tests and the example
share one model. This module adds the exact in-sample LD scores that the composed ones are compared with.
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

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
