"""Additive target-block estimating equations on fixed full-genome units.

The source kernels and residual projection stay fixed. Only target moments
are resampled. Stored person diagonals have no target-block resolution, so
their contribution is apportioned by target annotation mass. This is exact
for proportional blocks, not an exact reconstruction of missing diagonals.
"""
from __future__ import annotations

import numpy as np


class TargetMomentJackknife:
    """Cache the block decomposition once, independently of phenotype RHS."""

    def __init__(self, records, *, masses, same_person, residual_inverse, width):
        self.ids = np.unique(np.concatenate([c['block_ids'] for c in records]))
        self.masses = np.asarray(masses, dtype=float)
        p = len(self.masses) * width
        inv = np.repeat(1 / self.masses, width)
        h = residual_inverse.shape[0]
        self.block_masses = np.zeros((len(self.ids), len(self.masses)))
        self.b = np.zeros((len(self.ids), p, h))
        self.off = np.zeros((len(self.ids), p, p))
        self.rhs = None
        for c in records:
            idx = np.searchsorted(self.ids, c['block_ids'])
            np.add.at(self.block_masses, idx, c['block_masses'])
            np.add.at(self.b, idx, c['block_genetic_residual'].reshape(len(idx), p, h)*inv[None, :, None])
            np.add.at(self.off, idx, c['gram'].different_person_blocks)
            # Preserve any trailing phenotype axis for batched simulations.
            rhs = np.asarray(c['block_rhs']).reshape(len(idx), p, -1)*inv[None, :, None]
            if self.rhs is None:
                self.rhs = np.zeros((len(self.ids), p, rhs.shape[-1]))
            np.add.at(self.rhs, idx, rhs)
        if not np.allclose(self.block_masses.sum(0), self.masses, rtol=1e-12, atol=0):
            raise ValueError('target block masses do not sum to full masses')
        self.full_b = self.b.sum(0)
        self.projection = self.b @ residual_inverse
        fractions = np.repeat(self.block_masses/self.masses, width, axis=1)
        block = self.off + fractions[:, :, None]*same_person - self.projection @ self.full_b.T
        self.profile_blocks = (block + block.swapaxes(-1, -2))/2
        self.full_profile = self.profile_blocks.sum(0)
        self.residual_inverse = residual_inverse

    def retained(self, deleted, residual_rhs):
        if not np.isin(deleted, self.ids).all():
            raise ValueError('unknown deleted block')
        keep = ~np.isin(self.ids, deleted)
        masses = self.block_masses[keep].sum(0)
        if np.any(masses <= 0):
            raise ValueError('deletion exhausts an annotation')
        r = np.asarray(residual_rhs)
        if r.ndim == 1:
            r = r[:, None]
        rhs = self.rhs - self.projection @ r
        return self.profile_blocks[keep].sum(0), rhs[keep].sum(0), masses
