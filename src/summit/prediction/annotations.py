"""Nonnegative, possibly overlapping annotation-dependent covariance priors.

At variant j the prior is Lambda_j = sum_k A_jk Omega_k / sum_j A_jk.
Each Omega_k is a PSD covariance contribution in model phenotype units.
Overlapping annotation coefficients are conditional contributions; a set's
total covariance includes contributions from every annotation on its variants.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np

from ._validation import array, array_digest, canonical, closed, digest, psd


def _sealed(value):
    # Cached content identities require a non-writable backing store, not
    # just a flag which callers can switch back on for an owning ndarray.
    a = np.asarray(value, dtype=np.float64)
    return np.frombuffer(a.tobytes(order='F'), dtype=np.float64).reshape(a.shape, order='F')


def write_annotation_design(path, design):
    """Write a portable SNP-axis-authenticated design without overwriting."""
    if not isinstance(design, AnnotationDesign):
        raise ValueError('an AnnotationDesign is required')
    record = dict(kind='summit.prediction.annotation_design', schema_version=1,
                  names=design.names, variant_identity=design.variant_identity, identity=design.identity)
    with Path(path).open('xb') as stream:
        np.savez(stream, weights=design.weights, manifest=np.array(canonical(record)))


def load_annotation_design(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'weights', 'manifest'}:
            raise ValueError('invalid annotation design arrays')
        record = json.loads(str(archive['manifest']))
        closed(record, ('kind', 'schema_version', 'names', 'variant_identity', 'identity'), name='annotation design')
        if record['kind'] != 'summit.prediction.annotation_design' or record['schema_version'] != 1:
            raise ValueError('unsupported annotation design schema')
        design = AnnotationDesign(archive['weights'], tuple(record['names']), record['variant_identity'])
        if design.identity != record['identity']:
            raise ValueError('annotation design checksum mismatch')
        return design


@dataclass(frozen=True)
class AnnotationDesign:
    weights: np.ndarray
    names: tuple[str, ...]
    variant_identity: str
    masses: np.ndarray = field(init=False, repr=False)
    identity: str = field(init=False)

    def __post_init__(self):
        weights = _sealed(array(self.weights, name='annotation weights', ndim=2))
        names = tuple(self.names)
        if (not weights.shape[0] or not names or weights.shape[1] != len(names)
                or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names)
                or np.any(weights < 0) or not isinstance(self.variant_identity, str) or not self.variant_identity):
            raise ValueError('invalid annotation weights, names or variant identity')
        masses = np.asarray(np.sum(weights, axis=0, dtype=np.longdouble), dtype=float)
        if np.any(masses <= 0) or not np.isfinite(masses).all():
            raise ValueError('each annotation must have finite positive genome-wide mass')
        masses = _sealed(masses)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'names', names)
        object.__setattr__(self, 'masses', masses)
        object.__setattr__(self, 'identity', digest(dict(weights=array_digest(weights),
            names=names, variant_identity=self.variant_identity, masses=masses.tolist())))


@dataclass(frozen=True)
class AnnotationPrior:
    design: AnnotationDesign
    covariances: np.ndarray
    aggregate: np.ndarray = field(init=False, repr=False)
    scaled_covariances: np.ndarray = field(init=False, repr=False)
    identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.design, AnnotationDesign):
            raise ValueError('an authenticated AnnotationDesign is required')
        values = array(self.covariances, name='annotation covariances', ndim=3)
        if len(values) != len(self.design.names):
            raise ValueError('annotation covariance axis mismatch')
        values = _sealed(np.stack([psd(value) for value in values]))
        aggregate = _sealed(psd(values.sum(axis=0)))
        # Native products divide by M once, just as the homogeneous path does.
        # Fold M/M_k into these small K-by-Q-by-Q arrays, never into genotypes.
        scaled = values*(len(self.design.weights)/self.design.masses)[:, None, None]
        if not np.isfinite(scaled).all():
            raise ValueError('annotation prior normalization overflows')
        scaled = _sealed(scaled)
        object.__setattr__(self, 'covariances', values)
        object.__setattr__(self, 'aggregate', aggregate)
        object.__setattr__(self, 'scaled_covariances', scaled)
        object.__setattr__(self, 'identity', digest([self.design.identity, array_digest(values)]))

    @property
    def specification(self):
        return dict(kind='summit.prediction.annotation_prior.v1', identity=self.identity,
            design_identity=self.design.identity, names=list(self.design.names),
            variant_identity=self.design.variant_identity, masses=self.design.masses.tolist(),
            covariances=self.covariances.tolist(), normalization='sum_k A_jk Omega_k / M_k',
            preconditioner='positive aggregate-covariance approximation')

    def candidate(self, name, residual, specification):
        from .spec import CandidatePrior
        return CandidatePrior(name, self.aggregate, residual, specification, annotation_prior=self)

    def block(self, start, stop):
        """Bounded variant-by-Q² covariance tile, including M/M_k scaling."""
        if not 0 <= start < stop <= len(self.design.weights):
            raise ValueError('annotation block outside the sealed variant axis')
        return self.design.weights[start:stop]@self.scaled_covariances.reshape(len(self.covariances), -1)
