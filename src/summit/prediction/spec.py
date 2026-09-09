"""Explicit axes, sealed scales and trait-local training specifications."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from ._validation import array, array_digest, digest, identifier, indices, metadata, positive_int, psd


@dataclass(frozen=True)
class VariantAxis:
    ids: tuple[str, ...]
    chromosome: tuple[str, ...]
    position: tuple[int, ...]
    counted: tuple[str, ...]
    other: tuple[str, ...]
    genome_build: str

    def __post_init__(self):
        for key in ("ids", "chromosome", "position", "counted", "other"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        m = len(self.ids)
        if not m or any(len(getattr(self, x)) != m for x in ("chromosome", "position", "counted", "other")):
            raise ValueError("variant axis lengths disagree or are empty")
        if len(set(self.ids)) != m or any(not isinstance(x, str) or not x.strip() for x in self.ids):
            raise ValueError("variant IDs must be unique nonempty strings")
        if not isinstance(self.genome_build, str) or not self.genome_build:
            raise ValueError("genome build must be declared")
        if any(x not in tuple(str(i) for i in range(1, 23)) for x in self.chromosome):
            raise ValueError("V1 supports autosomal diploid variants only")
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1 for x in self.position):
            raise ValueError("invalid variant position")
        if any(a not in ("A", "C", "G", "T") or b not in ("A", "C", "G", "T") or a == b
               for a, b in zip(self.counted, self.other)):
            raise ValueError("V1 requires distinct biallelic SNP alleles")

    def to_dict(self):
        return dict(ids=list(self.ids), chromosome=list(self.chromosome), position=[int(x) for x in self.position],
                    counted=list(self.counted), other=list(self.other), genome_build=self.genome_build)

    @property
    def identity(self):
        return digest(self.to_dict())

    def subset(self, rows):
        return VariantAxis(**{k: tuple(getattr(self, k)[int(i)] for i in rows)
                              for k in ("ids", "chromosome", "position", "counted", "other")},
                           genome_build=self.genome_build)


def sample_ids(values):
    values = tuple(values)
    if any(isinstance(x, (str, bytes)) for x in values):
        raise ValueError("sample IDs must be pairs, not strings")
    values = tuple(tuple(x) for x in values)
    if not values or any(len(x) != 2 or any(not isinstance(v, str) or not v for v in x) for x in values):
        raise ValueError("sample axis requires nonempty string FID/IID pairs")
    if len(set(values)) != len(values):
        raise ValueError("duplicate FID/IID pairs")
    return values


@dataclass(frozen=True)
class GenotypeScale:
    mean: np.ndarray
    inverse_scale: np.ndarray
    variant_identity: str
    sample_identity: str
    provenance: dict
    ddof: int = 1
    arithmetic: str = "affine64"

    def __post_init__(self):
        mean = array(self.mean, name="mean", ndim=1)
        inv = array(self.inverse_scale, name="inverse_scale", ndim=1)
        if not mean.size or mean.shape != inv.shape or np.any((mean < 0) | (mean > 2)) or np.any(inv <= 0):
            raise ValueError("invalid genotype affine vectors")
        if self.ddof not in (0, 1) or isinstance(self.ddof, bool) or self.arithmetic != "affine64":
            raise ValueError("V1 requires ddof 0/1 and affine64 arithmetic")
        if not self.sample_identity or not self.variant_identity or not self.provenance:
            raise ValueError("genotype scale requires sample, variant and provenance identities")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "inverse_scale", inv)
        object.__setattr__(self, "provenance", metadata(self.provenance))

    @property
    def identity(self):
        return digest([array_digest(self.mean), array_digest(self.inverse_scale), self.variant_identity,
                       self.sample_identity, self.ddof, self.arithmetic, self.provenance])


@dataclass(frozen=True)
class CandidatePrior:
    id: str
    covariance: np.ndarray
    residual: np.ndarray
    specification: dict

    def __post_init__(self):
        identifier(self.id, "model ID")
        covariance = psd(self.covariance)
        residual = array(self.residual, name="residual variances", ndim=1)
        if not residual.size or np.any(residual <= 0):
            raise ValueError("residual variances must be strictly positive")
        if not self.specification:
            raise ValueError("prior/residual provenance must be declared")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "specification", metadata(self.specification))


@dataclass(frozen=True)
class TraitTraining:
    id: str
    rows: np.ndarray
    variants: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    fixed: np.ndarray
    scale: GenotypeScale
    candidates: tuple[CandidatePrior, ...]
    context_spec: dict
    fixed_spec: dict
    phenotype_spec: dict
    geometry: object = None

    def __post_init__(self):
        identifier(self.id, "trait ID")
        for key in ("rows", "variants"):
            object.__setattr__(self, key, indices(getattr(self, key), name=key))
        if np.any(np.diff(self.variants) <= 0):
            raise ValueError("training variants must follow increasing source order")
        for key, ndim in (("y", 1), ("phi", 2), ("fixed", 2)):
            object.__setattr__(self, key, array(getattr(self, key), name=key, ndim=ndim))
        n, q = self.phi.shape
        if n < 2 or q < 1 or not np.array_equal(self.phi[:, 0], np.ones(n)):
            raise ValueError("phi must start with an exact baseline-one column")
        if self.y.size != n or len(self.rows) != n or self.fixed.shape[0] != n or len(self.scale.mean) != len(self.variants):
            raise ValueError("trait axes do not agree")
        candidates = tuple(self.candidates)
        if not candidates or len({x.id for x in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be nonempty and unique within a trait")
        if any(x.covariance.shape != (q, q) or x.residual.shape != (n,) for x in candidates):
            raise ValueError("candidate prior/residual dimensions disagree with trait")
        # Equal immutable residual surfaces are shared, including when callers
        # supplied a separate vector for each genetic candidate. Copy only the
        # small dataclass shell; do not mutate caller-owned prior objects.
        surfaces, compact_candidates = {}, []
        for candidate in candidates:
            key = array_digest(candidate.residual)
            if key in surfaces:
                candidate = copy.copy(candidate)
                object.__setattr__(candidate, "residual", surfaces[key])
            else:
                surfaces[key] = candidate.residual
            compact_candidates.append(candidate)
        object.__setattr__(self, "candidates", tuple(compact_candidates))
        for key in ("context_spec", "fixed_spec", "phenotype_spec"):
            value = metadata(getattr(self, key))
            if not value:
                raise ValueError(f"{key} is required for portable scoring")
            object.__setattr__(self, key, value)
        names = self.context_spec.get("names", ())
        fixed_names = self.fixed_spec.get("names", [term["name"] for term in self.fixed_spec.get("terms", [])])
        if len(names) != q or len(set(names)) != q or any(not isinstance(v, str) or not v for v in names):
            raise ValueError("context specification must name each basis column uniquely")
        if len(fixed_names) != self.fixed.shape[1] or len(set(fixed_names)) != len(fixed_names):
            raise ValueError("fixed specification must identify the ordered design columns")
        if not isinstance(self.phenotype_spec.get("units"), str) or not self.phenotype_spec["units"]:
            raise ValueError("phenotype units are required")
        if self.geometry is not None and self.geometry.omega.shape != (q, q):
            raise ValueError("response geometry dimension mismatch")


@dataclass(frozen=True)
class SolverSpec:
    rtol: float = 5e-4
    atol: float = 0.0
    max_iterations: int = 150
    qr_rtol: float = 1e-12
    max_restarts: int = 3

    def __post_init__(self):
        for name in ("rtol", "qr_rtol"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must lie in (0,1)")
        if not np.isfinite(self.atol) or self.atol < 0:
            raise ValueError("atol must be finite and nonnegative")
        positive_int(self.max_iterations, "max_iterations")
        if isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int) or self.max_restarts < 0:
            raise ValueError("max_restarts must be a nonnegative integer")
