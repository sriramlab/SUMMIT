"""One raw block traversal per batched covariance application.

Trait masks, scales, contexts and priors remain statistically independent.
Identical genotype descriptors share standardized blocks; all others share raw
calls only. RHS tiles stay inside each block and always contain complete models.
"""
from __future__ import annotations

import numpy as np

from ._validation import array_digest
from .genotype import RawBlockStream, native_module, standardize


class GenotypeOperator:
    def __init__(self, source, traits, plan, *, backend="native"):
        if backend not in ("native", "numpy"):
            raise ValueError("backend must be native or numpy (reference)")
        self.source, self.traits, self.plan = source, tuple(traits), plan
        self.backend = backend
        self.native = native_module() if backend == "native" else None
        if self.native is not None:
            self.native.configure_blas_threads(plan.threads)
        self.stream = RawBlockStream(source, np.concatenate([t.rows for t in traits]),
            np.concatenate([t.variants for t in traits]), block_size=plan.block_size,
            storage=plan.storage, threads=plan.threads)
        self.groups = {}
        self.trait_group = {}
        for t in traits:
            key = (array_digest(t.rows), array_digest(t.variants), t.scale.identity)
            self.groups.setdefault(key, []).append(t)
            self.trait_group[t.id] = key
        self.row_maps = {k: np.searchsorted(self.stream.rows, ts[0].rows) for k, ts in self.groups.items()}
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
        selected = raw[np.ix_(self.row_maps[key], positions)]
        return lo, hi, standardize(selected, t.scale.mean[lo:hi], t.scale.inverse_scale[lo:hi])

    def setup(self):
        if self.ready:
            raise RuntimeError("operator setup may run only once")
        for key, ts in self.groups.items():
            self.row_diagonal[key] = np.zeros(len(ts[0].rows))
            if self.plan.storage == "standardized":
                self.standardized_cache[key] = np.empty((len(ts[0].rows), len(ts[0].variants)), order="F")
        for _, variants, raw in self.stream.blocks("setup", build_cache=self.plan.storage == "compact"):
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

    def apply(self, vectors, *, phase="cg"):
        """vectors maps (trait ID, candidate ID) to unprojected V inputs."""
        self.ledger.operator_calls += 1
        outputs, packs, selected, priors = {}, {}, {}, {}
        for t in self.traits:
            active = [c for c in t.candidates if (t.id, c.id) in vectors]
            if not active:
                continue
            selected[t.id] = active
            q = t.phi.shape[1]
            packed = np.empty((len(t.rows), len(active)*q), order="F")
            for j, c in enumerate(active):
                vector = np.asarray(vectors[(t.id, c.id)])
                if vector.shape != (len(t.rows),) or not np.all(np.isfinite(vector)):
                    raise ValueError("invalid covariance input vector")
                np.multiply(t.phi, vector[:, None], out=packed[:, j*q:(j+1)*q])
            packs[t.id] = packed
            priors[t.id] = np.ascontiguousarray([c.covariance.ravel() for c in active])
            outputs[t.id] = np.zeros((len(t.rows), len(active)), order="F")
        if not selected:
            raise ValueError("empty covariance application")
        if sum(len(v) for v in selected.values()) != len(vectors):
            raise ValueError("unknown model ID in covariance application")
        self.ledger.active_rhs.append(sum(v.shape[1] for v in packs.values()))
        for variants, raw in self.blocks(phase):
            for key, traits in self.groups.items():
                if not any(t.id in selected for t in traits):
                    continue
                _, _, g = self._group_block(key, variants, raw)
                if g is None:
                    continue
                for t in traits:
                    if t.id not in selected:
                        continue
                    q = t.phi.shape[1]
                    width = max(1, self.plan.rhs_columns // q)
                    for begin in range(0, len(selected[t.id]), width):
                        end = min(begin+width, len(selected[t.id]))
                        packed = packs[t.id][:, begin*q:end*q]
                        covariance = priors[t.id][begin:end]
                        out = outputs[t.id][:, begin:end]
                        if self.native is not None:
                            self.native.prediction_covariance_block(g, packed, covariance, t.phi,
                                out, float(len(t.variants)), self.plan.threads)
                        else:
                            transposed = g.T @ packed
                            mixed = np.einsum("bkq,kqr->bkr", transposed.reshape(len(g.T), end-begin, q),
                                covariance.reshape(end-begin, q, q)).reshape(len(g.T), -1) / len(t.variants)
                            product = g @ mixed
                            out += np.einsum("nkq,nq->nk", product.reshape(len(t.rows), end-begin, q), t.phi)
        result = {}
        for t in self.traits:
            for j, c in enumerate(selected.get(t.id, [])):
                key = (t.id, c.id)
                result[key] = outputs[t.id][:, j]
                result[key] += c.residual * vectors[key]
                if not np.all(np.isfinite(result[key])):
                    raise FloatingPointError(f"nonfinite covariance product for {key}")
        return result

    def extract(self, solutions, sink):
        """Write bounded (variant, Q) posterior blocks directly to a model sink."""
        packs = {}
        for t in self.traits:
            active = [c for c in t.candidates if (t.id, c.id) in solutions]
            q = t.phi.shape[1]
            packed = np.empty((len(t.rows), len(active)*q), order="F")
            for j, c in enumerate(active):
                packed[:, j*q:(j+1)*q] = t.phi * solutions[(t.id, c.id)][:, None]
            packs[t.id] = (active, packed)
        for variants, raw in self.blocks("extraction"):
            for group, traits in self.groups.items():
                lo, hi, g = self._group_block(group, variants, raw)
                if g is None:
                    continue
                for t in traits:
                    active, packed = packs[t.id]
                    q, width = t.phi.shape[1], max(1, self.plan.rhs_columns // t.phi.shape[1])
                    for begin in range(0, len(active), width):
                        batch = active[begin:begin+width]
                        product = self.product(g, packed[:, begin*q:(begin+len(batch))*q], transpose=True)
                        for j, c in enumerate(batch):
                            weights = product[:, j*q:(j+1)*q] @ c.covariance / len(t.variants)
                            sink((t.id, c.id), lo, hi, weights)

    def release(self):
        self.stream.cache = None
        self.standardized_cache.clear()
