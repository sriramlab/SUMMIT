"""Ordered genetic Grams from a sealed, orientation-summed GxE reference.

The default uses the study cohorts' exposure moments for different people and
the reference kernel diagonals on the actual overlapping people. Neither is
an exact change of projector. Reference Monte Carlo noise is also retained.
``reconstruct_ordered`` is exact only on the commuting exposure-moment span;
it must not be described as recovery of an arbitrary ordered Gram.

Ordered pairs are row-major (a*Q+b). Saved pairs use ContextPairIndex, and
annotation components are annotation-major. No genotype input is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np

from .spec import ContextPairIndex

GRAM_MODES = ('factorized', 'factorized_plus_residual', 'legacy_transport',
              'legacy_transport_exact')


def _matrix(name, value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite matrix')
    return a


def cohort_rows(rows, n):
    """Validate sorted indices into one authenticated master sample axis."""
    a = np.asarray(rows)
    if (a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a) or a[0] < 0
            or a[-1] >= n or np.any(a[1:] <= a[:-1])):
        raise ValueError('cohort rows must be sorted, unique, nonempty master indices')
    return a


@lru_cache(maxsize=16)
def orientation_matrix(q):
    """Map ordered kernel coordinates to the saved symmetric kernel basis."""
    pairs = ContextPairIndex(q)
    b = np.zeros((len(pairs), q*q))
    for p in pairs.entries:
        b[p.index, p.q*q+p.r] = 1
        b[p.index, p.r*q+p.q] = 1
    b.setflags(write=False)
    return b


def saved(ordered, q):
    """Orientation sum on the last two axes, preserving any batch axes."""
    a = np.asarray(ordered, dtype=float)
    if a.shape[-2:] != (q*q, q*q) or not np.isfinite(a).all():
        raise ValueError('ordered Gram dimensions/values disagree')
    b = orientation_matrix(q)
    return b @ a @ b.T


@lru_cache(maxsize=16)
def reconstruction_matrix(q):
    """One Q^4 by P^2 read-out of the least-squares commuting tensor.

    For Q=5 the intermediate design is 225 by 120. Its columns represent
    symmetric pair-pairs of symmetric exposure moments, not kernel pairs.
    """
    pairs = [(p.q, p.r) for p in ContextPairIndex(q).entries]
    unknown = {}
    def key(a, b, c, d):
        return tuple(sorted((tuple(sorted((a, c))), tuple(sorted((b, d))))))
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    unknown.setdefault(key(a, b, c, d), len(unknown))
    design = np.zeros((len(pairs)**2, len(unknown)))
    for i, p in enumerate(pairs):
        for j, s in enumerate(pairs):
            for a, b in ((p,) if p[0] == p[1] else (p, p[::-1])):
                for c, d in ((s,) if s[0] == s[1] else (s, s[::-1])):
                    design[i*len(pairs)+j, unknown[key(a, b, c, d)]] += 1
    if np.linalg.matrix_rank(design) != len(unknown):
        raise RuntimeError('commuting reconstruction design is rank deficient')
    readout = np.zeros((q**4, len(unknown)))
    for a, b, c, d in np.ndindex(q, q, q, q):
        readout[((a*q+b)*q+c)*q+d, unknown[key(a, b, c, d)]] = 1
    result = readout @ np.linalg.pinv(design, rcond=1e-14)
    result.setflags(write=False)
    return result


def reconstruct_ordered(symmetric, q):
    a = np.asarray(symmetric, dtype=float)
    p = q*(q+1)//2
    if a.shape[-2:] != (p, p) or not np.isfinite(a).all():
        raise ValueError('saved Gram dimensions/values disagree')
    result = np.einsum('ij,...j->...i', reconstruction_matrix(q),
                       a.reshape(*a.shape[:-2], p*p), optimize=True)
    return result.reshape(*a.shape[:-2], q*q, q*q)


def exposure_factor(phi, rows_x=None, rows_y=None):
    """Exact ordered exposure factor with the overlap fourth moment removed."""
    phi = _matrix('basis', phi)
    n, q = phi.shape
    x = np.arange(n) if rows_x is None else cohort_rows(rows_x, n)
    y = x if rows_y is None else cohort_rows(rows_y, n)
    overlap = np.intersect1d(x, y, assume_unique=True)
    mx, my = phi[x].T @ phi[x], phi[y].T @ phi[y]
    z = phi[overlap]
    products = (z[:, :, None]*z[:, None, :]).reshape(len(z), q*q)
    fourth = (products.T @ products).reshape(q, q, q, q)
    return (np.einsum('ac,bd->abcd', mx, my)
            - fourth.transpose(0, 2, 1, 3)).reshape(q*q, q*q)


def ordered_diagonals(diagonal, q):
    """Expand saved annotation-major diagonals; halve off-diagonal values."""
    d = _matrix('kernel diagonal', diagonal)
    p = q*(q+1)//2
    if d.shape[0] % p:
        raise ValueError('kernel diagonal component count disagrees')
    lookup = np.empty((q, q), dtype=int)
    factor = np.empty((q, q))
    for pair in ContextPairIndex(q).entries:
        lookup[pair.q, pair.r] = lookup[pair.r, pair.q] = pair.index
        factor[pair.q, pair.r] = factor[pair.r, pair.q] = 1 if pair.q == pair.r else .5
    k = d.shape[0]//p
    return (d.reshape(k, p, d.shape[1])[:, lookup.ravel()]
            * factor.ravel()[None, :, None]).reshape(k*q*q, d.shape[1])


def same_person(diagonal, rows_x, rows_y=None, *, q, ordered=True):
    d = _matrix('kernel diagonal', diagonal)
    x = cohort_rows(rows_x, d.shape[1])
    y = x if rows_y is None else cohort_rows(rows_y, d.shape[1])
    overlap = np.intersect1d(x, y, assume_unique=True)
    d = d[:, overlap]
    if ordered:
        d = ordered_diagonals(d, q)
    return d @ d.T


@dataclass(frozen=True)
class ChromosomeGram:
    """A full-mass genetic Gram and frozen-target block contributions."""
    different_person_blocks: np.ndarray
    same_person: np.ndarray
    diagnostics: dict

    @property
    def matrix(self):
        value = self.same_person + self.different_person_blocks.sum(axis=0)
        return (value+value.T)/2


def chromosome_gram(reference, diagonal, phi, rows_x, rows_y=None, *,
                    global_masses, mode='factorized', same_person_mode='own_rows',
                    ordered=True, repaired_blocks=None):
    """Assemble one chromosome on global annotation masses.

    The target block share of D_R uses target-annotation mass. It sums back
    to the full chromosome D_R. Its only effect in factorized mode is on ell.
    Full D_XY is frozen during jackknife; retained different-person blocks
    receive the usual mass restoration in the downstream assembler.
    """
    if mode not in GRAM_MODES or same_person_mode not in ('own_rows', 'scaled'):
        raise ValueError('unknown Gram or same-person mode')
    phi = _matrix('basis', phi)
    n, q = phi.shape
    if n != reference.n_samples or q != reference.num_basis:
        raise ValueError('reference and master basis dimensions differ')
    if not np.array_equal(phi[:, 0], np.ones(n)) or n < 2:
        raise ValueError('basis must begin with the constant-one column')
    x = cohort_rows(rows_x, n)
    y = x if rows_y is None else cohort_rows(rows_y, n)
    overlap = np.intersect1d(x, y, assume_unique=True)
    k = len(reference.annotation_names); p = q*(q+1)//2
    masses = np.asarray(global_masses, dtype=float)
    if masses.shape != (k,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('global masses must be finite and positive')
    diagonal = _matrix('kernel diagonal', diagonal)
    if diagonal.shape != (k*p, n):
        raise ValueError('reference diagonal axes disagree')
    inverse = np.repeat(1/masses, p)
    tr = reference.block_directed * reference.residual_rank**2
    tr = tr * inverse[None, :, None] * inverse[None, None, :]
    dr = diagonal @ diagonal.T
    chr_mass = reference.block_masses.sum(axis=0)
    fractions = np.divide(reference.block_masses, chr_mass,
                          out=np.zeros_like(reference.block_masses), where=chr_mass > 0)
    dr_blocks = np.repeat(fractions, p, axis=1)[:, :, None]*dr[None]
    off = tr-dr_blocks
    fr = exposure_factor(phi)
    fxy = exposure_factor(phi, x, y)
    fr_saved = saved(fr, q)
    ell = off[:, ::p, ::p]/(n*(n-1))
    off_pairs = off.reshape(-1, k, p, k, p).transpose(0, 1, 3, 2, 4)
    residual_pairs = off_pairs-ell[..., None, None]*fr_saved
    norm = np.linalg.norm(off_pairs, axis=(-2, -1))
    relative = np.divide(np.linalg.norm(residual_pairs, axis=(-2, -1)), norm,
                         out=np.zeros_like(norm), where=norm > 0)
    n_distinct = len(x)*len(y)-len(overlap)
    scale = n_distinct/(n*(n-1))
    width = q*q if ordered else p
    factor = fxy if ordered else saved(fxy, q)
    if mode == 'factorized':
        blocks = ell[..., None, None]*factor
    elif mode == 'factorized_plus_residual':
        residual = reconstruct_ordered(residual_pairs, q) if ordered else residual_pairs
        blocks = ell[..., None, None]*factor + scale*residual
    elif mode == 'legacy_transport_exact':
        if repaired_blocks is None:
            raise ValueError('exact legacy mode requires authenticated repaired block Grams')
        repair = np.asarray(repaired_blocks, dtype=float)
        if repair.shape != (len(off), k*q*q, k*q*q) or not np.isfinite(repair).all():
            raise ValueError('repaired ordered block axes disagree')
        full_dr_ordered = ordered_diagonals(diagonal, q)
        d = full_dr_ordered @ full_dr_ordered.T
        corrected = repair-np.repeat(fractions, q*q, axis=1)[:, :, None]*d
        blocks = corrected.reshape(-1, k, q*q, k, q*q).transpose(0, 1, 3, 2, 4)*scale
        if not ordered:
            blocks = saved(blocks, q)
    else:
        blocks = scale*(reconstruct_ordered(off_pairs, q) if ordered else off_pairs)
    blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(-1, k*width, k*width)
    study_diagonal = ordered_diagonals(diagonal, q) if ordered else diagonal
    dxy = (study_diagonal[:, overlap] @ study_diagonal[:, overlap].T
           if same_person_mode == 'own_rows'
           else (len(overlap)/n)*(study_diagonal @ study_diagonal.T))
    same_pairs = dr_blocks.reshape(-1, k, p, k, p).transpose(0, 1, 3, 2, 4)
    total_norm = np.linalg.norm(tr.reshape(-1, k, p, k, p).transpose(0, 1, 3, 2, 4), axis=(-2, -1))
    share = np.divide(np.linalg.norm(same_pairs, axis=(-2, -1)), total_norm,
                      out=np.zeros_like(total_norm), where=total_norm > 0)
    return ChromosomeGram(blocks, dxy, dict(mode=mode, same_person_mode=same_person_mode,
        factorization_residual=relative, same_person_share=share, ld_scalar=ell,
        n_x=len(x), n_y=len(y), n_overlap=len(overlap), reference_n=n,
        block_same_person_policy='target_annotation_mass_apportionment_v1',
        projector_policy='reference_diagonals_on_study_overlap_v1'))


def within_trait_equations(reference_chromosomes, study_chromosomes, *,
        reference_diagonals, phi, rows, mode='factorized', same_person_mode='own_rows',
        reference_residual_gram, reference_residual_traces, reference_same_person,
        residual_gram, residual_rhs, residual_traces, residual_names,
        deleted_blocks=(), expected_chromosomes=None, prepared_grams=None,full_same_person=None):
    """Corrected within-trait chromosome assembly, with a bit-exact old arm.

    The legacy/scaled arm delegates to the unchanged in-house implementation.
    Corrected arms form the full same-person Gram from the sum of chromosome
    diagonals and retain exact study genetic/residual moments. Deletions use
    the existing frozen-source residual-profile change relative to full data;
    the full-data profile cannot replace the explicit full same-person Gram.
    """
    from summit.ldscore.generalized_gxe_chromosome import (
        joint_chromosome_equations, transferred_chromosome_equations)
    ref, study = tuple(reference_chromosomes), tuple(study_chromosomes)
    common = dict(residual_gram=residual_gram, residual_rhs=residual_rhs,
        residual_traces=residual_traces, residual_names=residual_names,
        deleted_blocks=deleted_blocks, expected_chromosomes=expected_chromosomes)
    if mode == 'legacy_transport' and same_person_mode == 'scaled':
        return transferred_chromosome_equations(ref, study,
            reference_residual_gram=reference_residual_gram,
            reference_residual_traces=reference_residual_traces,
            reference_same_person=reference_same_person, **common)
    target = joint_chromosome_equations(study, **common)
    if len(rows) != target.study_n:
        raise ValueError('trait rows disagree with study sample count')
    ref_by_chr = {c.chromosome:c for c in ref}
    if set(ref_by_chr) != {c.chromosome for c in study}:
        raise ValueError('reference/study chromosome sets differ')
    full_masses = sum(c.block_masses.sum(axis=0) for c in study)
    p = study[0].num_basis*(study[0].num_basis+1)//2
    mass_restore = np.repeat(full_masses/target.annotation_masses, p)
    if full_same_person is None:
        if same_person_mode=='scaled':
            full_same_person=(len(rows)/ref[0].n_samples)*reference_same_person
        else:
            if reference_diagonals is None:
                raise ValueError('own-row assembly requires full same-person moments or all chromosome diagonals')
            diagonal=sum(reference_diagonals[c.chromosome] for c in ref)
            full_same_person=same_person(diagonal,rows,q=ref[0].num_basis,ordered=False)
    replacement=np.array(full_same_person,dtype=float,copy=True)
    if (replacement.shape!=(target.genetic_count,target.genetic_count) or not np.isfinite(replacement).all()
            or not np.allclose(replacement,replacement.T,rtol=1e-12,atol=1e-10)):
        raise ValueError('invalid full same-person Gram')
    full_inverse=np.repeat(1/full_masses,p)
    full_b=[c.block_genetic_residual.sum(0)*full_inverse[:,None] for c in study]
    full_total=sum(full_b)
    full_profile=(full_total@np.linalg.solve(residual_gram,full_total.T)
        -sum(b@np.linalg.solve(residual_gram,b.T) for b in full_b)) if len(study)>1 else np.zeros_like(replacement)
    for c in study:
        r = ref_by_chr[c.chromosome]
        if (not np.array_equal(r.block_ids, c.block_ids)
                or not np.array_equal(r.block_masses, c.block_masses)
                or r.annotation_names != c.annotation_names):
            raise ValueError('reference/study block designs differ')
        if prepared_grams is None:
            g = chromosome_gram(r, reference_diagonals[c.chromosome], phi, rows,
                global_masses=full_masses, mode=mode, same_person_mode=same_person_mode,
                ordered=False)
        else:
            g = prepared_grams[c.chromosome]
            if (g.diagnostics['mode'] != mode or g.diagnostics['same_person_mode'] != same_person_mode
                    or g.diagnostics['n_x'] != len(rows) or g.diagnostics['n_y'] != len(rows)
                    or g.same_person.shape != (target.genetic_count, target.genetic_count)):
                raise ValueError('prepared chromosome Gram design differs')
        take = ~np.isin(c.block_ids, deleted_blocks)
        off = g.different_person_blocks[take].sum(axis=0)
        replacement += off*mass_restore[:, None]*mass_restore[None]
        # Match the existing frozen-source, mass-restored chromosome profile.
        if len(deleted_blocks):
            inv = np.repeat(1/target.annotation_masses, p)
            b = c.block_genetic_residual[take].sum(axis=0)*inv[:, None]
            source = c.block_genetic_residual.sum(axis=0)*inv[:, None]
            profile = b @ np.linalg.solve(residual_gram, source.T)
            replacement -= (profile+profile.T)/2
    btotal = target.matrix[:target.genetic_count, target.genetic_count:]
    if len(deleted_blocks):
        replacement += btotal @ np.linalg.solve(residual_gram, btotal.T)-full_profile
    replacement = (replacement+replacement.T)/2
    matrix = target.matrix.copy()
    matrix[:target.genetic_count, :target.genetic_count] = replacement
    return replace(target, matrix=matrix, transferred_genetic_gram=replacement,
                   reference_n=ref[0].n_samples)
