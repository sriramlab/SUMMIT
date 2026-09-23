"""Exact study cross-products for traits with different missing-person masks.

The master environment values and covariate column span are shared. Each trait
retains its own covariate projection and phenotype normalization. A genotype
block is decoded once; shared products are corrected by subtracting missing
people (or computed directly when fewer people are observed). This module does
not approximate missingness or perform reference-population transfer.
"""
from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from summit.context.spec import ContextPairIndex
from summit.ldscore.generalized_gxe_trait_summary import (
    _compressed_column_span,
    _finite_matrix,
)


class MaskedTraitBatch:
    """Prepare shared weights, then yield one trait's statistics per SNP block.

    ``traits`` contains dictionaries with ``name``, sorted master-row
    ``indices``, orthonormal ``fixed_basis``, and one phenotype vector.
    The caller reduces yielded blocks immediately; no genome-wide per-SNP
    residual-information array is retained.
    """

    def __init__(self, *, basis, fixed_basis, residual_basis, traits):
        phi = _finite_matrix('basis', basis)
        u = _finite_matrix('fixed_basis', fixed_basis)
        d = _finite_matrix('residual_basis', residual_basis)
        n, q = phi.shape
        if u.shape[0] != n or d.shape[0] != n or u.shape[1] >= n:
            raise ValueError('incompatible master sample axes')
        if not np.allclose(u.T @ u, np.eye(u.shape[1]), rtol=0, atol=1e-10):
            raise ValueError('master fixed basis must be orthonormal')
        self.n, self.q, self.h, self.fixed_rank = n, q, d.shape[1], u.shape[1]
        self.pairs = tuple((p.q, p.r) for p in ContextPairIndex(q).entries)
        multipliers = np.column_stack([phi] + [d[:, h, None]*phi for h in range(self.h)])
        span, self.fixed_coefficients = _compressed_column_span(multipliers)
        self.multiplier_rank = span.shape[1]
        self.fixed_weights = np.ascontiguousarray(
            (span[:, :, None]*u[:, None, :]).reshape(n, -1))
        products = np.column_stack([phi[:, left]*phi[:, right] for left, right in self.pairs])
        square = np.column_stack([products] + [d[:, h, None]*products for h in range(self.h)])
        self.square_weights, self.square_coefficients = _compressed_column_span(square)
        master_residual = np.stack([u.T @ (d[:, h, None]*u) for h in range(self.h)])
        self.traits = []
        names = set()
        score_weights = []
        for spec in traits:
            name = spec['name']
            idx = np.asarray(spec['indices'])
            if (not isinstance(name, str) or not name or name in names or idx.ndim != 1
                    or idx.dtype.kind not in 'iu' or not len(idx) or idx[0] < 0
                    or idx[-1] >= n or np.any(idx[1:] <= idx[:-1])):
                raise ValueError('trait names and sample indices must be unique and valid')
            names.add(name)
            fixed = _finite_matrix('trait fixed_basis', spec['fixed_basis'])
            if fixed.shape != (len(idx), self.fixed_rank):
                raise ValueError('trait covariate rank or sample axis differs')
            master = u[idx]
            transform = np.linalg.solve(master.T @ master, master.T @ fixed)
            if not np.allclose(master @ transform, fixed, rtol=1e-10, atol=1e-11):
                raise ValueError('trait fixed basis is outside the shared covariate span')
            if not np.allclose(fixed.T @ fixed, np.eye(self.fixed_rank), rtol=0, atol=1e-10):
                raise ValueError('trait fixed basis must be orthonormal')
            y = np.asarray(spec['phenotype'], dtype=float).reshape(-1, 1)
            if y.shape != (len(idx), 1) or not np.isfinite(y).all():
                raise ValueError('each trait must contain one phenotype')
            y = y-fixed @ (fixed.T @ y)
            ss = float(np.sum(y*y))
            if ss <= np.finfo(float).eps*len(idx) or len(idx) <= self.fixed_rank:
                raise ValueError('trait has zero residual variance or no residual degrees of freedom')
            y *= np.sqrt((len(idx)-self.fixed_rank)/ss)
            absent = np.ones(n, dtype=bool)
            absent[idx] = False
            missing = np.flatnonzero(absent)
            subtract = len(missing) <= len(idx)
            correction = missing if subtract else idx
            if len(correction):
                uc = u[correction]
                corrected = np.stack([uc.T @ (d[correction, h, None]*uc) for h in range(self.h)])
                if subtract:
                    corrected = master_residual-corrected
            else:
                corrected = master_residual
            compressed_residual = transform.T @ corrected @ transform
            ds = d[idx]
            leverage = np.sum(fixed*fixed, axis=1)
            common = SimpleNamespace(normalized_phenotypes=y,
                residual_rank=len(idx)-self.fixed_rank, residual_rhs=ds.T @ (y*y),
                residual_traces=ds.sum(axis=0)-np.trace(compressed_residual, axis1=1, axis2=2),
                residual_gram=ds.T @ (ds*(1-2*leverage)[:, None])
                    + np.einsum('hij,kji->hk', compressed_residual, compressed_residual))
            weights = np.zeros((n, q))
            weights[idx] = phi[idx]*y
            score_weights.append(weights)
            self.traits.append(dict(name=name, indices=idx.copy(), common=common,
                transform=transform, compressed_residual=compressed_residual,
                correction=correction, subtract=subtract))
        if not self.traits:
            raise ValueError('at least one trait is required')
        self.score_weights = np.ascontiguousarray(np.column_stack(score_weights))

    @property
    def report(self):
        return dict(traits=len(self.traits), master_samples=self.n,
            fixed_multiplier_rank=self.multiplier_rank,
            square_multiplier_rank=self.square_weights.shape[1],
            equivalent_cohort_products=1+sum(len(t['correction']) for t in self.traits)/self.n,
            persistent_weight_bytes=sum(a.nbytes for a in (
                self.fixed_weights, self.square_weights, self.score_weights)))

    def block(self, genotype):
        """Yield (name, scores[M,Q], information[M,P], residual[M,P,H]).

        Genotypes have shape variants x master people, use the reference
        affine scaling, and have mean-imputed missing calls (standardized 0).
        """
        x = _finite_matrix('genotype block', genotype)
        if x.shape[1] != self.n:
            raise ValueError('genotype block does not match master sample axis')
        width = len(x)
        shared_linear = x @ self.fixed_weights
        scores = (x @ self.score_weights).reshape(width, len(self.traits), self.q)
        squared = x*x
        shared_square = squared @ self.square_weights
        p = len(self.pairs)
        for index, trait in enumerate(self.traits):
            rows = trait['correction']
            if len(rows):
                linear = x[:, rows] @ self.fixed_weights[rows]
                square = squared[:, rows] @ self.square_weights[rows]
                if trait['subtract']:
                    linear = shared_linear-linear
                    square = shared_square-square
            else:
                linear, square = shared_linear, shared_square
            cross = np.einsum('brp,rt->btp',
                linear.reshape(width, self.multiplier_rank, self.fixed_rank),
                self.fixed_coefficients, optimize=True) @ trait['transform']
            raw_square = square @ self.square_coefficients
            information = np.empty((width, p))
            residual = np.empty((width, p, self.h))
            for j, (left, right) in enumerate(self.pairs):
                a, b = cross[:, left], cross[:, right]
                information[:, j] = raw_square[:, j]-np.einsum('bp,bp->b', a, b)
                for h in range(self.h):
                    wa = cross[:, self.q*(h+1)+left]
                    wb = cross[:, self.q*(h+1)+right]
                    residual[:, j, h] = (raw_square[:, p*(h+1)+j]
                        - np.einsum('bp,bp->b', a, wb)
                        - np.einsum('bp,bp->b', wa, b)
                        + np.einsum('bp,pv,bv->b', a,
                            trait['compressed_residual'][h], b, optimize=True))
            yield trait['name'], scores[:, index], information, residual
