"""Within-chromosome LD moments with a single genome-wide covariance fit.

This is a distinct approximation, not a full-genome LD reference. It assumes
cross-chromosome genetic kernels are uncorrelated after projection away from
the residual-kernel span. The primary assembler uses exact-cohort moments;
the separate transfer adapter requires explicit same-person reference moments.
Chromosome kernels use global annotation masses; coefficients are therefore
genome-wide covariance contributions, not chromosome-specific estimates.

The native LD construction has no deletion groups. Fixed target rows are
reduced afterwards, and source sketches remain frozen for block deletion.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
import json
from pathlib import Path
import numpy as np

from summit.context.fit import ContextNormalEquations
from summit.context.oracle import intercept_ld_scalars
from summit.context.spec import ContextComponentIndex, ContextPairIndex, owned_readonly_array
from summit.context.spec import array_sha256, canonical_json, canonical_sha256
from summit.ldscore.generalized_gxe_native import GeneralizedGxENativeResult

CHROMOSOME_LD_POLICY = "within_chromosome_residual_profile_global_mass_v1"
# Optional raw intercept-pair same-person entries of one chromosome (K x K,
# mass-free).  With block_directed and block_masses they give the per-block
# LD scalars of the different-person transport.  Older artifacts lack them.
_OPTIONAL_CHROMOSOME_ARRAYS = ("same_person_intercept",)


def write_chromosome_moments(path, moments, *, provenance):
    """Publish an authenticated, compact artifact without replacing outputs."""
    if not isinstance(moments, ChromosomeMoments) or not provenance:
        raise ValueError("validated chromosome moments and provenance are required")
    arrays, scalar = {}, {}
    for field in fields(moments):
        value = getattr(moments, field.name)
        if value is None and field.name in _OPTIONAL_CHROMOSOME_ARRAYS:
            continue  # absent optional arrays keep the earlier manifest layout
        (arrays if isinstance(value, np.ndarray) else scalar)[field.name] = value
    manifest = dict(schema=CHROMOSOME_LD_POLICY, scalar=scalar, provenance=provenance,
                    arrays={name: array_sha256(value) for name, value in arrays.items()})
    with Path(path).open('xb') as stream:
        np.savez(stream, manifest=np.array(canonical_json(manifest)), **arrays)


def load_chromosome_moments(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['manifest']))
        if manifest['schema'] != CHROMOSOME_LD_POLICY or not manifest['provenance']:
            raise ValueError("unsupported chromosome moment artifact")
        if set(archive.files) != {'manifest', *manifest['arrays']}:
            raise ValueError("chromosome artifact array names disagree")
        arrays = {name: archive[name] for name in manifest['arrays']}
        if any(array_sha256(value) != manifest['arrays'][name] for name, value in arrays.items()):
            raise ValueError("chromosome artifact checksum mismatch")
    return ChromosomeMoments(**manifest['scalar'], **arrays), manifest['provenance']


@dataclass(frozen=True)
class ChromosomeMoments:
    chromosome: str
    cohort_identity: str
    annotation_names: tuple[str, ...]
    trait_names: tuple[str, ...]
    num_basis: int
    n_samples: int
    residual_rank: int
    block_ids: np.ndarray
    block_masses: np.ndarray
    block_directed: np.ndarray
    block_genetic_rhs: np.ndarray
    block_genetic_residual: np.ndarray
    # This chromosome's same-person matrix at the intercept pairs, in raw
    # (mass-free) units: D_c[(a,0),(b,0)] * M_a * M_b.  K x K, person-summed.
    same_person_intercept: np.ndarray | None = None

    def __post_init__(self):
        if self.chromosome not in tuple(map(str, range(1, 23))) or not self.cohort_identity:
            raise ValueError("autosomal chromosome and cohort identity are required")
        for name in ("annotation_names", "trait_names"):
            value = tuple(getattr(self, name))
            if not value or len(value) != len(set(value)) or any(not x for x in value):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if not 0 < self.residual_rank <= self.n_samples or self.num_basis < 1:
            raise ValueError("invalid sample or basis dimensions")
        ids = np.asarray(self.block_ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids < 0) or np.any(np.diff(ids) <= 0):
            raise ValueError("block IDs must be sorted unique nonnegative integers")
        object.__setattr__(self, 'block_ids', owned_readonly_array(ids, dtype=np.int64))
        for name in ('block_masses', 'block_directed', 'block_genetic_rhs', 'block_genetic_residual'):
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite {name}")
            object.__setattr__(self, name, owned_readonly_array(value))
        b, k, t = len(ids), len(self.annotation_names), len(self.trait_names)
        c = k*self.num_basis*(self.num_basis+1)//2
        if (self.block_masses.shape != (b, k) or np.any(self.block_masses < 0)
                or self.block_directed.shape != (b, c, c)
                or self.block_genetic_rhs.shape != (b, c, t)
                or self.block_genetic_residual.ndim != 3
                or self.block_genetic_residual.shape[:2] != (b, c)
                or self.block_genetic_residual.shape[2] < 1):
            raise ValueError("incompatible chromosome moment axes")
        if self.same_person_intercept is not None:
            value = np.asarray(self.same_person_intercept, dtype=float)
            if (value.shape != (k, k) or not np.isfinite(value).all()
                    or not np.allclose(value, value.T, rtol=1e-12, atol=0)):
                raise ValueError("same_person_intercept must be a finite symmetric K x K array")
            object.__setattr__(self, 'same_person_intercept', owned_readonly_array(value))

    def ld_scalars(self, same_person_intercept=None):
        """Raw per-block intercept LD scalars (blocks x K x K), or None.

        Uses ``same_person_intercept`` (raw K x K) when given, else the stored
        one. See summit.context.oracle.intercept_ld_scalars.
        """
        intercept = self.same_person_intercept if same_person_intercept is None else same_person_intercept
        if intercept is None:
            return None
        return intercept_ld_scalars(self.block_directed*self.residual_rank**2, self.block_masses,
            intercept, reference_n=self.n_samples, pair_count=self.num_basis*(self.num_basis+1)//2)


def reduce_chromosome_result(
    result: GeneralizedGxENativeResult, *, annotations, annotation_names,
    active_annotations, block_ids, chromosome, cohort_identity, trait_names,
) -> ChromosomeMoments:
    """Reduce fixed native target rows to compact post-hoc block numerators.

Zero-mass chromosome annotations are omitted from the native execution and
embedded here on the shared genome-wide component axis. No source is rebuilt.
"""
    if (result.trait_scores is None or result.trait_residual_information is None
            or result.component_kernel_diagonal is None):
        raise ValueError("fused trait statistics and component diagonals are required")
    a = np.asarray(annotations, dtype=float)
    groups = np.asarray(block_ids)
    active = np.asarray(active_annotations)
    m, q, t = result.trait_scores.shape
    k, p = len(annotation_names), len(result.pair_table)
    if (a.shape != (m, k) or not np.isfinite(a).all() or np.any(a < 0)
            or groups.shape != (m,) or groups.dtype.kind not in 'iu' or np.any(groups < 0)
            or np.any(np.diff(groups) < 0) or active.ndim != 1
            or active.dtype.kind not in 'iu' or np.any(np.diff(active) <= 0)
            or not np.array_equal(active, np.flatnonzero(a.sum(axis=0) > 0))):
        raise ValueError("invalid annotations, active columns or contiguous block order")
    local_masses = np.asarray([np.sum(a[:, j], dtype=np.longdouble) for j in active], dtype=float)
    np.testing.assert_allclose(local_masses, result.annotation_masses, rtol=1e-13)
    if result.directional_ldscores.shape != (m, p, len(active)*p) or len(trait_names) != t:
        raise ValueError("native component or trait axes disagree")
    labels, starts = np.unique(groups, return_index=True)
    stops = np.r_[starts[1:], m]
    b, c, h = len(labels), k*p, result.trait_residual_information.shape[2]
    masses = np.zeros((b, k)); directed = np.zeros((b, c, c))
    rhs = np.zeros((b, c, t)); cross = np.zeros((b, c, h))
    columns = (active[:, None]*p+np.arange(p)).ravel()
    for block, (lo, hi) in enumerate(zip(starts, stops)):
        weights = a[lo:hi]
        masses[block] = weights.sum(axis=0)
        # Directional rows already include both orientations for off-diagonal
        # covariance components, and are normalized by residual_rank**2 only.
        reduced = (weights.T @ result.directional_ldscores[lo:hi].reshape(hi-lo, -1)).reshape(c, -1)
        directed[block][:, columns] = reduced
        scores = result.trait_scores[lo:hi]
        for pair, (left, right) in enumerate(result.pair_table):
            factor = 1. if left == right else 2.
            rhs[block, pair::p] = factor*(weights.T @ (scores[:, left]*scores[:, right]))
            cross[block, pair::p] = factor*(weights.T @ result.trait_residual_information[lo:hi, pair])
    # Raw (mass-free) intercept-pair same-person entries of this chromosome.
    # The native diagonals use local masses; multiply each annotation back.
    intercepts = np.zeros((k, result.component_kernel_diagonal.shape[1]))
    intercepts[active] = (result.component_kernel_diagonal[::p]
        * np.asarray(result.annotation_masses, dtype=float)[:, None])
    same_intercept = intercepts@intercepts.T
    return ChromosomeMoments(
        str(chromosome), cohort_identity, tuple(annotation_names), tuple(trait_names), q,
        result.component_kernel_diagonal.shape[1],
        result.residual_rank, labels, masses, directed, rhs, cross,
        (same_intercept+same_intercept.T)/2)


def combine_chromosome_annotations(chunk, weights, annotation_names):
    """Reuse fixed LD moments for the annotation design ``A_new = A @ weights``.

    This is an exact linear reduction of the existing stochastic moments. It
    needs no genotypes, new sketches, or separately fitted chromosome effects.
    For disjoint bins covering every SNP, an all-one column merges them into
    the unpartitioned model. Global masses are recomputed during joint fitting,
    including each target-block deletion. Nonnegative overlapping combinations
    are supported; they cannot recover a finer partition than the input design.
    """
    if not isinstance(chunk, ChromosomeMoments):
        raise ValueError('validated chromosome moments are required')
    if np.iscomplexobj(weights):
        raise ValueError('annotation combination must be real')
    weights = np.asarray(weights, dtype=float)
    names = tuple(annotation_names)
    if (weights.shape != (len(chunk.annotation_names), len(names)) or not names
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.any(weights.sum(axis=0) <= 0)):
        raise ValueError('invalid annotation combination weights or names')
    p = chunk.num_basis*(chunk.num_basis+1)//2
    transform = np.kron(weights, np.eye(p))
    identity = canonical_sha256(dict(cohort=chunk.cohort_identity,
        annotation_combination=array_sha256(weights), annotation_names=names))
    intercept = (None if chunk.same_person_intercept is None
                 else weights.T@chunk.same_person_intercept@weights)
    return ChromosomeMoments(
        chunk.chromosome, identity, names, chunk.trait_names, chunk.num_basis,
        chunk.n_samples, chunk.residual_rank, chunk.block_ids,
        chunk.block_masses@weights,
        transform.T@chunk.block_directed@transform,
        transform.T@chunk.block_genetic_rhs,
        transform.T@chunk.block_genetic_residual,
        None if intercept is None else (intercept+intercept.T)/2)


def joint_chromosome_equations(
    chromosomes, *, residual_gram, residual_rhs, residual_traces, residual_names,
    trait=0, deleted_blocks=(), expected_chromosomes=None,
) -> ContextNormalEquations:
    """One genome-wide solve; residual moments are counted exactly once.

For each chromosome c, B_c denotes genetic/residual cross moments and C the
common residual Gram. The genetic information is sum_c(T_c-B_c C^-1 B_c'),
and the RHS is sum_c(g_c-B_c C^-1 r). A full system is reconstructed with
one common residual coefficient vector. No cross-chromosome LD is evaluated.

With deletions, the target B_c uses retained SNP rows and the source B_c stays
frozen, matching target-only LD-row deletion. Both annotation denominators
use the global retained masses, as in the existing variant-row jackknife.
The residual basis must start with the constant-one column for trace recovery.
"""
    chunks = tuple(chromosomes)
    if not chunks or any(not isinstance(x, ChromosomeMoments) for x in chunks):
        raise ValueError("chromosome moments are required")
    labels = [x.chromosome for x in chunks]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate chromosome contributions")
    if expected_chromosomes is not None and set(labels) != set(map(str, expected_chromosomes)):
        raise ValueError("incomplete or unexpected chromosome set")
    first = chunks[0]
    fields = ('cohort_identity', 'annotation_names', 'trait_names', 'num_basis', 'n_samples', 'residual_rank')
    if any(any(getattr(x, key) != getattr(first, key) for key in fields) for x in chunks):
        raise ValueError("chromosome cohort/design/annotation identities disagree")
    if isinstance(trait, bool) or not isinstance(trait, (int, np.integer)) or not 0 <= trait < len(first.trait_names):
        raise ValueError("invalid trait index")
    gram = np.asarray(residual_gram, dtype=float)
    rrhs = np.asarray(residual_rhs, dtype=float)
    traces = np.asarray(residual_traces, dtype=float)
    h = first.block_genetic_residual.shape[2]
    if (gram.shape != (h, h) or rrhs.shape != (h, len(first.trait_names))
            or traces.shape != (h,) or len(residual_names) != h
            or not all(np.isfinite(x).all() for x in (gram, rrhs, traces))
            or not np.allclose(gram, gram.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("invalid common residual moments")
    np.linalg.cholesky(gram)
    # Centered residual kernels can have nearly zero trace although their
    # individual summands are O(N). Compare cancellation error on the natural
    # Gram scale, not with a fixed absolute tolerance in sample-count units.
    trace_scale = np.sqrt(gram[0, 0]*np.diag(gram))
    if not np.all(np.abs(gram[0]-traces) <= 1e-10*np.abs(traces)+1e-12*trace_scale) or not np.isclose(
            traces[0], first.residual_rank, rtol=1e-10, atol=1e-10):
        raise ValueError("residual moments must start with the projected constant-one kernel")
    deleted = tuple(deleted_blocks)
    available = {int(i) for x in chunks for i in x.block_ids}
    if (len(set(deleted)) != len(deleted) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
            or i not in available for i in deleted)):
        raise ValueError("unknown or duplicated deleted block")
    keep = [~np.isin(x.block_ids, deleted) for x in chunks]
    masses = sum((x.block_masses[take].sum(axis=0) for x, take in zip(chunks, keep)))
    if np.any(masses <= 0):
        raise ValueError("deletion leaves an annotation with zero genome-wide mass")
    components = ContextComponentIndex(first.annotation_names, ContextPairIndex(first.num_basis))
    c = len(components); p = len(components.pair_index)
    inverse = np.repeat(1/masses, p)
    information = np.zeros((c, c)); genetic_rhs = np.zeros((c, len(first.trait_names)))
    total_cross = np.zeros((c, h))
    for chunk, take in zip(chunks, keep):
        directed = chunk.block_directed[take].sum(axis=0)
        within = directed*(first.residual_rank**2)*inverse[:, None]*inverse[None, :]
        target = chunk.block_genetic_residual[take].sum(axis=0)*inverse[:, None]
        source = chunk.block_genetic_residual.sum(axis=0)*inverse[:, None]
        profiled = within-target@np.linalg.solve(gram, source.T)
        information += (profiled+profiled.T)/2
        total_cross += target
        genetic_rhs += chunk.block_genetic_rhs[take].sum(axis=0)*inverse[:, None]
    genetic_gram = information+total_cross@np.linalg.solve(gram, total_cross.T)
    matrix = np.block([[genetic_gram, total_cross], [total_cross.T, gram]])
    return ContextNormalEquations(
        matrix=(matrix+matrix.T)/2, rhs=np.r_[genetic_rhs[:, trait], rrhs[:, trait]],
        traces=np.r_[total_cross[:, 0], traces],
        component_names=components.names+tuple(residual_names), genetic_count=c,
        annotation_masses=masses, deleted_groups=tuple(map(str, deleted)),
        reference_genetic_gram=genetic_gram, transferred_genetic_gram=genetic_gram,
        reference_n=first.n_samples, study_n=first.n_samples)


def transferred_chromosome_equations(
    reference_chromosomes, study_chromosomes, *, reference_residual_gram,
    reference_residual_traces, reference_same_person, residual_gram,
    residual_rhs, residual_traces, residual_names, deleted_blocks=(),
    expected_chromosomes=None, study_same_person=None,
    same_person_transport=None, reference_kernel_diagonal=None,
    study_rows=None, reference_fourth_moments=None,
    study_fourth_moments=None, reference_second_moments=None,
    study_second_moments=None, reference_same_person_intercepts=None,
    different_person_transport=None,
):
    """Transfer one shared chromosome reference to one trait's exact moments.

    ``T_S = D_S + s (T_R - D_R) + kron(L, F_S - s F_R)``, s = n_S(n_S-1)/(n_R(n_R-1)).

    Different-person part (``different_person_transport``):

    - "composition_corrected" when every reference chromosome has its raw
      intercept same-person entries (``reference_same_person_intercepts``, a
      mapping chromosome -> K x K, or the chunks' ``same_person_intercept``)
      and both cohorts' context second and fourth moments are given. ``L`` sums
      the retained blocks' raw LD scalars (ChromosomeMoments.ld_scalars) over
      the retained annotation masses of this deletion, the masses that
      normalize the deleted T_R; ``F`` is distinct_pair_exposure(M, G).
    - else "scaled_legacy" (s O_R only), with a DifferentPersonTransportWarning
      when the sizes differ unless requested explicitly.

    Same-person part (``same_person_transport``):

    - ``study_same_person`` (C x C), when the caller formed it, with
      ``same_person_transport`` naming how ("own_rows_exact" or
      "fourth_moment_difference");
    - else the exact own-row term from ``reference_kernel_diagonal`` (the
      genome-wide C x N_R diagonal, i.e. the sum of every chromosome's
      sample-aligned component diagonal) on ``study_rows`` (the trait's people
      as reference sample indices);
    - else the fourth-moment difference form from ``reference_fourth_moments``
      and ``study_fourth_moments`` (summit.context.oracle.context_fourth_moments);
    - else the legacy ``(n_S/n_R) D_R``, with a SamePersonTransportWarning when
      the sizes differ unless ``same_person_transport="scaled_legacy"``.

    Both choices are recorded on the returned equations. For approximate
    target-row deletions the full study same-person term is reused, as the
    full reference D_R is; callers looping over deletions should form it once
    and pass ``study_same_person`` (ChromosomeTransport.for_study does this).
    Shared environment/covariate definitions, compatible populations and
    identical genotype affine scaling remain caller requirements.
    """
    from dataclasses import replace
    from summit.context.oracle import (
        different_person_correction, resolve_different_person_exposure,
        resolve_study_same_person, transfer_reference_gram)

    ref, study = tuple(reference_chromosomes), tuple(study_chromosomes)
    if not ref or not study:
        raise ValueError('reference and study chromosomes are required')
    ref_by_chr = {x.chromosome: x for x in ref}
    if set(ref_by_chr) != {x.chromosome for x in study}:
        raise ValueError('reference and study chromosome sets differ')
    for chunk in study:
        other = ref_by_chr[chunk.chromosome]
        if (chunk.annotation_names != other.annotation_names or chunk.num_basis != other.num_basis
                or not np.array_equal(chunk.block_ids, other.block_ids)
                or not np.array_equal(chunk.block_masses, other.block_masses)):
            raise ValueError('reference and study variant block designs differ')
    options = dict(residual_names=residual_names, deleted_blocks=deleted_blocks,
                   expected_chromosomes=expected_chromosomes)
    reference = joint_chromosome_equations(ref,
        residual_gram=reference_residual_gram, residual_traces=reference_residual_traces,
        residual_rhs=np.zeros((len(residual_names), len(ref[0].trait_names))), **options)
    target = joint_chromosome_equations(study, residual_gram=residual_gram,
        residual_traces=residual_traces, residual_rhs=residual_rhs, **options)
    c = target.genetic_count
    if study_same_person is not None:
        if same_person_transport not in ('own_rows_exact', 'fourth_moment_difference'):
            raise ValueError('an explicit study_same_person needs its own_rows_exact '
                             'or fourth_moment_difference label')
        transport = same_person_transport
        study_same = np.asarray(study_same_person, dtype=float)
        if study_same.shape != (c, c) or not np.isfinite(study_same).all():
            raise ValueError('study_same_person must be a finite C by C matrix')
    else:
        study_same, transport = resolve_study_same_person(
            reference_same_person, reference_n=reference.reference_n,
            study_n=target.study_n, annotation_count=len(ref[0].annotation_names),
            component_kernel_diagonal=reference_kernel_diagonal, study_rows=study_rows,
            reference_fourth_moments=reference_fourth_moments,
            study_fourth_moments=study_fourth_moments,
            requested=same_person_transport, stacklevel=2)
    intercepts = reference_same_person_intercepts
    if intercepts is None and all(x.same_person_intercept is not None for x in ref):
        intercepts = {x.chromosome: x.same_person_intercept for x in ref}
    if intercepts is not None and set(map(str, intercepts)) != set(ref_by_chr):
        raise ValueError('same-person intercepts must cover every reference chromosome')
    exposure, different_transport = resolve_different_person_exposure(
        reference_n=reference.reference_n, study_n=target.study_n,
        reference_second_moments=reference_second_moments,
        reference_fourth_moments=reference_fourth_moments,
        study_second_moments=study_second_moments,
        study_fourth_moments=study_fourth_moments,
        ld_scalars_available=intercepts is not None,
        requested=different_person_transport, stacklevel=2)
    correction = None
    if exposure is not None:
        lookup = {str(key): value for key, value in intercepts.items()}
        deleted = tuple(deleted_blocks)
        scalars = np.concatenate([x.ld_scalars(lookup[x.chromosome]) for x in ref])
        keep = np.concatenate([~np.isin(x.block_ids, deleted) for x in ref])
        correction = different_person_correction(
            scalars, exposure, masses=reference.annotation_masses, retained=keep)
    transferred = transfer_reference_gram(reference.reference_genetic_gram,
        reference_same_person, reference_n=reference.reference_n, study_n=target.study_n,
        study_same_person=study_same, different_person_correction=correction)
    matrix = target.matrix.copy()
    matrix[:c, :c] = transferred
    return replace(target, matrix=matrix,
        reference_genetic_gram=reference.reference_genetic_gram,
        transferred_genetic_gram=transferred, reference_n=reference.reference_n,
        same_person_transport=transport,
        different_person_transport=different_transport)


def chromosome_same_person_intercept(diagonal, masses):
    """Raw intercept same-person entries (K x K) of one chromosome.

    ``diagonal`` is the chromosome's sample-aligned C x N component diagonal
    normalized by ``masses`` (the K annotation masses, e.g. the genome-wide
    masses of a reference_aux ``global_kernel_diagonal``). Only its intercept
    rows are used: ``x = diagonal[::P] * masses``, returning ``x x'``.
    """
    d = np.asarray(diagonal, dtype=float)
    m = np.asarray(masses, dtype=float)
    if d.ndim != 2 or m.ndim != 1 or d.shape[0] % len(m) or not np.isfinite(d).all():
        raise ValueError('diagonal must be a finite C by N array with C a multiple of K')
    pairs = d.shape[0]//len(m)
    x = d[::pairs]*m[:, None]
    value = x@x.T
    return (value+value.T)/2


@dataclass(frozen=True)
class ChromosomeTransport:
    """Reference-side inputs of the own-row, composition-corrected transport.

    Built once per reference root by :func:`prepare_chromosome_transport`.
    :meth:`for_study` returns the keyword arguments of
    :func:`transferred_chromosome_equations` for one trait's rows, so loops
    over jackknife deletions reuse every study-level term.
    """
    reference_n: int
    diagonal: np.ndarray
    reference_same_person: np.ndarray
    same_person_intercepts: dict
    basis: np.ndarray
    reference_second_moments: np.ndarray
    reference_fourth_moments: np.ndarray

    def for_study(self, rows):
        from summit.context.oracle import exact_study_same_person
        from summit.context.spec import context_fourth_moments, context_second_moments
        return dict(
            reference_same_person=self.reference_same_person,
            study_same_person=exact_study_same_person(self.diagonal, rows),
            same_person_transport='own_rows_exact',
            reference_same_person_intercepts=self.same_person_intercepts,
            reference_second_moments=self.reference_second_moments,
            reference_fourth_moments=self.reference_fourth_moments,
            study_second_moments=context_second_moments(self.basis, rows),
            study_fourth_moments=context_fourth_moments(self.basis, rows))


def prepare_chromosome_transport(reference_chromosomes, reference_chromosome_diagonals,
                                 basis, *, diagonal_masses=None):
    """Prepare a reference root for own-row, composition-corrected transport.

    ``reference_chromosome_diagonals`` maps each chromosome to its C x N
    per-person component diagonal normalized by ``diagonal_masses`` (default:
    the genome-wide annotation masses, the sum of every chunk's block masses,
    as in reference_aux ``global_kernel_diagonal``). ``basis`` is the
    reference's N x Q context basis, first column constant one. The genome-wide
    D_R uses the summed diagonals, so it includes cross-chromosome products.
    """
    from summit.context.oracle import exact_study_same_person
    from summit.context.spec import context_fourth_moments, context_second_moments
    ref = tuple(reference_chromosomes)
    if not ref:
        raise ValueError('reference chromosomes are required')
    lookup = {str(key): value for key, value in reference_chromosome_diagonals.items()}
    if set(lookup) != {x.chromosome for x in ref}:
        raise ValueError('diagonals must cover exactly the reference chromosomes')
    n = ref[0].n_samples
    k, p = len(ref[0].annotation_names), ref[0].num_basis*(ref[0].num_basis+1)//2
    masses = (sum(x.block_masses.sum(axis=0) for x in ref) if diagonal_masses is None
              else np.asarray(diagonal_masses, dtype=float))
    phi = np.asarray(basis, dtype=float)
    if phi.shape != (n, ref[0].num_basis) or not np.array_equal(phi[:, 0], np.ones(n)):
        raise ValueError('basis must be N x Q and begin with the constant-one column')
    total = None
    intercepts = {}
    for chunk in ref:
        d = np.asarray(lookup[chunk.chromosome], dtype=float)
        if d.shape != (k*p, n):
            raise ValueError('chromosome diagonal axes disagree with the reference')
        intercepts[chunk.chromosome] = chromosome_same_person_intercept(d, masses)
        total = d.copy() if total is None else total+d
    total.setflags(write=False)
    return ChromosomeTransport(
        reference_n=n, diagonal=total,
        reference_same_person=exact_study_same_person(total, np.arange(n)),
        same_person_intercepts=intercepts, basis=phi,
        reference_second_moments=context_second_moments(phi),
        reference_fourth_moments=context_fourth_moments(phi))


def own_row_transported_equations(reference_chromosomes, study_chromosomes, *, transport,
                                  study_rows, **moments):
    """One call: own-row exact same-person and composition-corrected
    different-person transport of a prepared reference to one trait.

    ``moments`` are the remaining transferred_chromosome_equations arguments
    (reference/study residual moments, residual_names, deleted_blocks,
    expected_chromosomes).
    """
    return transferred_chromosome_equations(reference_chromosomes, study_chromosomes,
        **transport.for_study(study_rows), **moments)
