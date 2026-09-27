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
from summit.context.spec import ContextComponentIndex, ContextPairIndex, owned_readonly_array
from summit.context.spec import array_sha256, canonical_json, canonical_sha256
from summit.ldscore.generalized_gxe_native import GeneralizedGxENativeResult

CHROMOSOME_LD_POLICY = "within_chromosome_residual_profile_global_mass_v1"


def write_chromosome_moments(path, moments, *, provenance):
    """Publish an authenticated, compact artifact without replacing outputs."""
    if not isinstance(moments, ChromosomeMoments) or not provenance:
        raise ValueError("validated chromosome moments and provenance are required")
    arrays, scalar = {}, {}
    for field in fields(moments):
        value = getattr(moments, field.name)
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
    return ChromosomeMoments(
        str(chromosome), cohort_identity, tuple(annotation_names), tuple(trait_names), q,
        result.component_kernel_diagonal.shape[1],
        result.residual_rank, labels, masses, directed, rhs, cross)


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
    return ChromosomeMoments(
        chunk.chromosome, identity, names, chunk.trait_names, chunk.num_basis,
        chunk.n_samples, chunk.residual_rank, chunk.block_ids,
        chunk.block_masses@weights,
        transform.T@chunk.block_directed@transform,
        transform.T@chunk.block_genetic_rhs,
        transform.T@chunk.block_genetic_residual)


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
    expected_chromosomes=None,
):
    """Transfer one shared chromosome reference to one trait's exact moments.

    This uses the existing same-person/distinct-person population transfer.
    Shared environment/covariate definitions and compatible populations remain
    caller requirements: sample-size scaling cannot correct selective missingness.
    Genotype affine scaling must be identical for the reference and study.
    Approximate target-row jackknife reuses full-reference same-person moments,
    matching the existing generalized-GxE variant-reference convention.
    """
    from dataclasses import replace
    from summit.context.oracle import transfer_reference_gram

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
    transferred = transfer_reference_gram(reference.reference_genetic_gram,
        reference_same_person, reference_n=reference.reference_n, study_n=target.study_n)
    matrix = target.matrix.copy()
    c = target.genetic_count
    matrix[:c, :c] = transferred
    return replace(target, matrix=matrix,
        reference_genetic_gram=reference.reference_genetic_gram,
        transferred_genetic_gram=transferred, reference_n=reference.reference_n)
