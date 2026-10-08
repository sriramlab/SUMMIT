"""One raw block traversal per batched covariance application.

Trait masks, scales, contexts and priors remain statistically independent.
Identical genotype descriptors share standardized blocks; all others share raw
calls only. RHS tiles stay inside each block and always contain complete models.
"""
from __future__ import annotations

import numpy as np

from ._validation import array_digest
from .genotype import RawBlockStream, native_module, StandardizedBlock
from .runtime import configure_prediction_threads


class GenotypeOperator:
    def __init__(self, source, traits, plan, *, backend="native"):
        if backend not in ("native", "numpy"):
            raise ValueError("backend must be native or numpy (reference)")
        self.source, self.traits, self.plan = source, tuple(traits), plan
        self.backend = backend
        self.native = native_module() if backend == "native" else None
        if self.native is not None and any(
                c.annotation_prior is not None for t in self.traits for c in t.candidates):
            build = self.native.build_info()
            # Full-array qualification found intermittent discrepancies in
            # the unchecked private-BLIS path. The protected path passed the
            # candidate-switch and independent weighted-product checks. Keep
            # the new prior family on that qualified path until this is resolved.
            if build.get("blas_vendor") == "BLIS" and not (
                    build.get("gemm_integrity_enabled") and build.get("gemm_checksum_enabled")):
                raise ValueError(
                    "Annotation-prior BLIS fits require GXELDCORE_GEMM_INTEGRITY=ON "
                    "and GXELDCORE_GEMM_CHECKSUM=ON; the unchecked BLIS path "
                    "has not passed full-array numerical qualification")
        if self.native is not None:
            configure_prediction_threads(self.native, plan.threads)
        self.affine_block = StandardizedBlock(self.native, plan.threads)
        self.workspace = self.native.PredictionWorkspace() if self.native is not None else None
        self.stream = RawBlockStream(source, np.concatenate([t.rows for t in traits]),
            np.concatenate([t.variants for t in traits]), block_size=plan.block_size,
            storage=plan.storage, threads=plan.threads, native=self.native)
        self.groups = {}
        self.trait_group = {}
        for t in traits:
            key = (array_digest(t.rows), array_digest(t.variants), t.scale.identity)
            self.groups.setdefault(key, []).append(t)
            self.trait_group[t.id] = key
        self.row_maps = {k: np.searchsorted(self.stream.rows, ts[0].rows) for k, ts in self.groups.items()}
        self.context_hash = {t.id: array_digest(t.phi) for t in self.traits}
        self.row_diagonal = {}
        self.standardized_cache = {}
        self.ready = False

    @property
    def ledger(self):
        return self.stream.ledger

    def _group_block(self, key, variants, raw):
        t = self.groups[key][0]
        lo = int(np.searchsorted(t.variants, variants[0]))
        hi = int(np.searchsorted(t.variants, variants[-1], side="right"))
        if lo == hi:
            return lo, hi, None
        if key in self.standardized_cache and self.ready:
            return lo, hi, self.standardized_cache[key][:, lo:hi]
        positions = np.searchsorted(variants, t.variants[lo:hi])
        return lo, hi, self.affine_block.prepare(raw, self.row_maps[key], positions,
            t.scale.mean[lo:hi], t.scale.inverse_scale[lo:hi])

    def setup(self):
        if self.ready:
            raise RuntimeError("operator setup may run only once")
        for key, ts in self.groups.items():
            self.row_diagonal[key] = np.zeros(len(ts[0].rows))
            if self.plan.storage == "standardized":
                self.standardized_cache[key] = np.empty((len(ts[0].rows), len(ts[0].variants)), order="F")
        for _, variants, raw in self.stream.blocks("setup", build_cache=self.plan.storage in ("compact", "packed")):
            for key, ts in self.groups.items():
                lo, hi, g = self._group_block(key, variants, raw)
                if g is None:
                    continue
                self.row_diagonal[key] += np.einsum("ij,ij->i", g, g) / len(ts[0].variants)
                if self.plan.storage == "standardized":
                    self.standardized_cache[key][:, lo:hi] = g
        for value in self.standardized_cache.values():
            value.setflags(write=False)
        self.ready = True

    def blocks(self, phase):
        if not self.ready:
            raise RuntimeError("operator setup is required")
        if self.plan.storage == "standardized":
            self.stream.check()
            self.ledger.begin(phase)
            for start in range(0, len(self.stream.variants), self.plan.block_size):
                variants = self.stream.variants[start:start+self.plan.block_size]
                self.ledger.cache_blocks += 1
                self.ledger.cache_variants += len(variants)
                yield variants, None
            self.stream.check()
        else:
            for _, variants, raw in self.stream.blocks(phase):
                yield variants, raw

    def product(self, left, right, *, transpose=False):
        left = np.asfortranarray(left, dtype=np.float64)
        right = np.asfortranarray(right, dtype=np.float64)
        if self.native is None:
            return (left.T if transpose else left) @ right
        out = np.empty((left.shape[1] if transpose else left.shape[0], right.shape[1]), order="F")
        self.native.prediction_product(left, right, out, transpose, self.plan.threads)
        return out

    def _packed_groups(self, vectors):
        """Pack independent model inputs with an identical genotype/context map."""
        groups = {}
        count = 0
        for genotype_key, traits in self.groups.items():
            contexts = {}
            for t in traits:
                active = [c for c in t.candidates if (t.id, c.id) in vectors]
                if active:
                    contexts.setdefault(self.context_hash[t.id], []).extend((t, c) for c in active)
            for context_key, models in contexts.items():
                t = models[0][0]
                q = t.phi.shape[1]
                packed = np.empty((len(t.rows), len(models)*q), order="F")
                for j, (trait, candidate) in enumerate(models):
                    vector = np.asarray(vectors[(trait.id, candidate.id)])
                    if vector.shape != (len(trait.rows),) or not np.all(np.isfinite(vector)):
                        raise ValueError("invalid covariance input vector")
                    np.multiply(trait.phi, vector[:, None], out=packed[:, j*q:(j+1)*q])
                groups.setdefault(genotype_key, []).append((models, packed))
                count += len(models)
        if not count or count != len(vectors):
            raise ValueError("empty or unknown model IDs in covariance application")
        return groups

    def apply(self, vectors, *, phase="cg"):
        """Apply independent covariances, sharing compatible cross-trait GEMMs."""
        self.ledger.operator_calls += 1
        groups = self._packed_groups(vectors)
        output_groups = {}
        for key, batches in groups.items():
            output_groups[key] = [np.zeros((len(packed), len(models)), order="F")
                                  for models, packed in batches]
        self.ledger.active_rhs.append(sum(p.shape[1] for batches in groups.values() for _, p in batches))
        for variants, raw in self.blocks(phase):
            for key, batches in groups.items():
                lo, hi, g = self._group_block(key, variants, raw)
                if g is None:
                    continue
                for (models, pack), output in zip(batches, output_groups[key]):
                    t = models[0][0]
                    q = t.phi.shape[1]
                    width = max(1, self.plan.rhs_columns // q)
                    for begin in range(0, len(models), width):
                        batch = models[begin:begin+width]
                        end = begin+len(batch)
                        packed = pack[:, begin*q:end*q]
                        annotated = any(c.annotation_prior is not None for _, c in batch)
                        if annotated:
                            covariance = np.empty((hi-lo, len(batch), q*q))
                            for j, (_, candidate) in enumerate(batch):
                                covariance[:, j] = (candidate.covariance.ravel() if candidate.annotation_prior is None
                                    else candidate.annotation_prior.block(lo, hi))
                            covariance = covariance.reshape((hi-lo)*len(batch), q*q)
                        else:
                            covariance = np.ascontiguousarray([c.covariance.ravel() for _, c in batch])
                        out = output[:, begin:end]
                        if self.native is not None:
                            self.native.prediction_covariance_block(g, packed, covariance, t.phi,
                                out, float(len(t.variants)), self.plan.threads, self.workspace)
                        else:
                            transposed = g.T @ packed
                            if annotated:
                                mixed = np.einsum("bkq,bkqr->bkr", transposed.reshape(len(g.T), len(batch), q),
                                    covariance.reshape(len(g.T), len(batch), q, q)).reshape(len(g.T), -1) / len(t.variants)
                            else:
                                mixed = np.einsum("bkq,kqr->bkr", transposed.reshape(len(g.T), len(batch), q),
                                    covariance.reshape(len(batch), q, q)).reshape(len(g.T), -1) / len(t.variants)
                            product = g @ mixed
                            out += np.einsum("nkq,nq->nk", product.reshape(len(t.rows), len(batch), q), t.phi)
        result = {}
        for key, batches in groups.items():
            for (models, _), output in zip(batches, output_groups[key]):
                for j, (t, c) in enumerate(models):
                    model = (t.id, c.id)
                    result[model] = output[:, j]
                    result[model] += c.residual * vectors[model]
                    if not np.all(np.isfinite(result[model])):
                        raise FloatingPointError(f"nonfinite covariance product for {model}")
        return result

    def extract(self, solutions, sink):
        """Write bounded posterior blocks, sharing compatible trait products."""
        groups = self._packed_groups(solutions)
        for variants, raw in self.blocks("extraction"):
            for key, batches in groups.items():
                lo, hi, g = self._group_block(key, variants, raw)
                if g is None:
                    continue
                for models, packed in batches:
                    t = models[0][0]
                    q, width = t.phi.shape[1], max(1, self.plan.rhs_columns // t.phi.shape[1])
                    for begin in range(0, len(models), width):
                        batch = models[begin:begin+width]
                        product = self.product(g, packed[:, begin*q:(begin+len(batch))*q], transpose=True)
                        for j, (trait, c) in enumerate(batch):
                            if c.annotation_prior is None:
                                weights = product[:, j*q:(j+1)*q] @ c.covariance / len(trait.variants)
                            else:
                                weights = np.einsum('bq,bqr->br', product[:, j*q:(j+1)*q],
                                    c.annotation_prior.block(lo, hi).reshape(hi-lo, q, q))/len(trait.variants)
                            sink((trait.id, c.id), lo, hi, weights)

    def release(self):
        self.stream.cache = None
        self.stream.cache_ready = False
        self.stream.unpack_buffer = None
        self.standardized_cache.clear()
        self.affine_block.buffer = np.empty(0, dtype=np.float64)
        self.workspace = None
