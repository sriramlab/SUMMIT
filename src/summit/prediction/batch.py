"""Admission planning for ragged traits and shared genotype storage."""
from dataclasses import dataclass, asdict
import numpy as np

from ._validation import array_digest, digest, positive_int


@dataclass(frozen=True)
class ResourcePlan:
    storage: str
    block_size: int
    rhs_columns: int
    threads: int
    memory_bytes: int
    estimated_peak_bytes: int
    allocations: dict
    union_samples: int
    union_variants: int
    models: int
    fit_identity: str

    def to_dict(self):
        return asdict(self)


def plan_prediction(traits, source, *, storage="stream", block_size=512, rhs_columns=64,
                    threads=1, memory_bytes=16 * 2**30):
    """Metadata-only conservative allocation estimate; never scans genotypes."""
    traits = tuple(traits)
    if not traits or len({t.id for t in traits}) != len(traits):
        raise ValueError("traits must have unique IDs")
    for key, value in dict(block_size=block_size, rhs_columns=rhs_columns, threads=threads,
                           memory_bytes=memory_bytes).items():
        positive_int(value, key)
    if storage not in ("stream", "compact", "standardized"):
        raise ValueError("unknown genotype storage mode")
    rows = np.unique(np.concatenate([t.rows for t in traits]))
    variants = np.unique(np.concatenate([t.variants for t in traits]))
    if rows[-1] >= len(source.samples) or variants[-1] >= len(source.variants.ids):
        raise ValueError("training indices exceed source axes")
    for t in traits:
        if t.scale.provenance.get("source") != source.identity:
            raise ValueError(f"{t.id}: scale genotype source identity mismatch; attach authenticated source provenance")
        if t.scale.variant_identity != source.variants.subset(t.variants).identity:
            raise ValueError(f"{t.id}: scale variant/allele identity mismatch")
        if t.scale.sample_identity != digest([source.samples[int(i)] for i in t.rows]):
            raise ValueError(f"{t.id}: scale discovery sample/order identity mismatch")
    nmax = max(len(t.rows) for t in traits)
    qmax = max(t.phi.shape[1] for t in traits)
    if rhs_columns < qmax:
        raise ValueError("RHS tile must fit a complete response vector")
    nk = sum(len(t.rows)*len(t.candidates) for t in traits)
    nkq = sum(len(t.rows)*len(t.candidates)*t.phi.shape[1] for t in traits)
    raw_itemsize = 1 if source.hard_calls else 8
    cache = len(rows)*len(variants)*raw_itemsize if storage == "compact" else 0
    if storage == "standardized":
        seen = set()
        for t in traits:
            key = (array_digest(t.rows), array_digest(t.variants), t.scale.identity)
            if key not in seen:
                cache += 8*len(t.rows)*len(t.variants)
                seen.add(key)
    b = min(block_size, len(variants))
    allocations = {
        "genotype_cache": cache,
        "solver_and_true_check_vectors": 12*8*nk,
        "packed_active_rhs": 8*nkq,
        "trait_design_projection_residuals": sum(8*len(t.rows)*(3*t.fixed.shape[1]+t.phi.shape[1]+4+len(t.candidates)) for t in traits),
        "raw_gather_and_affine_blocks": 2*raw_itemsize*len(rows)*b + 3*8*nmax*b,
        "native_gemm_tiles_and_integrity_reserve": 8*(8*nmax*rhs_columns+8*b*rhs_columns),
        "axes_and_scales": 128*(len(source.samples)+len(source.variants.ids)) + sum(24*len(t.variants) for t in traits),
        "runtime_reserve": 256*2**20,
    }
    peak = sum(allocations.values())
    if peak > memory_bytes:
        raise MemoryError(f"prediction plan needs approximately {peak/2**30:.3f} GiB; budget {memory_bytes/2**30:.3f} GiB; reduce tiles or use stream storage")
    identity = digest([source.identity, [dict(id=t.id, scale=t.scale.identity,
        y=array_digest(t.y), phi=array_digest(t.phi), fixed=array_digest(t.fixed),
        contexts=t.context_spec, fixed_spec=t.fixed_spec, phenotype=t.phenotype_spec,
        geometry=None if t.geometry is None else [array_digest(t.geometry.omega), array_digest(t.geometry.metric),
                                                  t.geometry.reference, t.geometry.anchor],
        candidates=[dict(id=c.id, covariance=array_digest(c.covariance), residual=array_digest(c.residual),
                         specification=c.specification) for c in t.candidates]) for t in traits]])
    return ResourcePlan(storage, block_size, rhs_columns, threads, memory_bytes, peak, allocations,
                        len(rows), len(variants), sum(len(t.candidates) for t in traits), identity)
