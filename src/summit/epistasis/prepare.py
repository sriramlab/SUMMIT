"""Selected independent contextual components on a common study genotype scale.

Two complete variant-probe reference passes precede two trait passes. The last
trait pass exports exact phenotype-dependent cubic moments. No N-by-N kernel
or full Q(Q+1)/2 context covariance expansion is allocated here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from summit.context.fixed import thin_rank_revealing_fixed_effect_basis
from summit.context.oracle import validate_annotations, residual_moments_low_rank
from summit.context.spec import array_sha256, canonical_sha256, owned_readonly_array, freeze_context_mapping
from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator, ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator, ProtectedTNOperator
from summit.ldscore.generalized_gxe_variant import (
    generate_global_variant_probes, native_global_variant_probes, GlobalVariantProbeSpec,
)
from summit.prediction._validation import digest, indices, positive_int
from summit.prediction.genotype import RawBlockStream, StandardizedBlock, estimate_scale
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.spec import GenotypeScale

from .summary import EpistasisSummary


def fit_scale(source, rows, *, method="hwe", block_size=256, threads=1,
              memory_bytes=2**30):
    """Use the shared missingness/mean machinery; declare HWE or empirical scale.

    Monomorphic/all-missing variants must be excluded explicitly before fitting.
    The first implementation requires one ordered full variant axis.
    """
    if method not in ("hwe", "empirical"):
        raise ValueError("scale must be hwe or empirical")
    scale = estimate_scale(source, rows, np.arange(len(source.variants.ids)),
                           block_size=block_size, threads=threads, memory_bytes=memory_bytes)
    variance = scale.mean * (1 - scale.mean / 2)
    if np.any(variance <= 0):
        raise ValueError("remove monomorphic/all-missing variants from the input axis")
    if method == "empirical":
        return scale
    return GenotypeScale(scale.mean, 1 / np.sqrt(variance), scale.variant_identity,
                         scale.sample_identity, {"method": "study_mean_imputed_hwe",
                         "source": source.identity}, ddof=0)


@dataclass(frozen=True)
class SelectedReference:
    """Completed per-variant directional numerators, no inference blocks."""
    directional: np.ndarray  # M, target component, source component; unnormalized
    information: np.ndarray  # M, component; w_j ||f_j||²
    weights: np.ndarray
    matrix: np.ndarray
    traces: np.ndarray
    residual_information: np.ndarray  # M, genetic component, residual component
    metadata: dict
    probe_matrices: np.ndarray | None = None

    def __post_init__(self):
        for key in ("directional", "information", "weights", "matrix", "traces", "residual_information"):
            object.__setattr__(self, key, owned_readonly_array(getattr(self, key)))
        object.__setattr__(self, "metadata", freeze_context_mapping(self.metadata))
        if self.probe_matrices is not None:
            object.__setattr__(self,"probe_matrices",owned_readonly_array(self.probe_matrices))


class SelectedStudy:
    """Cohort-side owner, reusable only for traits with the same sample mask.

    ``modifiers[:, a]`` and ``weights[:, a]`` define component a. Use a column
    of ones for additive effects. Supply the scientific definitions explicitly;
    all arrays and ordered axes are fingerprinted in addition to those labels.
    """
    def __init__(self, source, rows, scale, *, fixed_effects, modifiers, weights,
                 component_names, definitions, block_size=256, threads=1,
                 memory_bytes=2**30, backend="native", residual_basis=None,
                 residual_names=None, metadata_variants=None):
        self.rows = indices(rows, name="sample rows", size=len(source.samples))
        if np.any(np.diff(self.rows) <= 0):
            raise ValueError("sample rows must be strictly increasing")
        self.n, self.m = len(self.rows), len(source.variants.ids)
        self.names = tuple(component_names)
        self.c = len(self.names)
        if (not self.names or len(set(self.names)) != self.c
                or any(not isinstance(v, str) or not v or v == "residual" for v in self.names)):
            raise ValueError("invalid component names")
        self.modifiers = np.array(modifiers, dtype=float, order="F", copy=True)
        if self.modifiers.shape != (self.n, self.c) or not np.all(np.isfinite(self.modifiers)):
            raise ValueError("modifiers must be finite and sample/component aligned")
        self.weights, self.masses = validate_annotations(weights, self.m, self.c)
        self.weights = self.weights.copy()
        self.fixed = owned_readonly_array(fixed_effects, dtype=float)
        if self.fixed.ndim != 2 or self.fixed.shape[0] != self.n:
            raise ValueError("fixed effects are not sample aligned")
        self.u = thin_rank_revealing_fixed_effect_basis(self.fixed)
        self.rank = self.n - self.u.shape[1]
        if self.rank < 1:
            raise ValueError("fixed effects leave no residual degrees of freedom")
        self.residual_basis = (np.ones((self.n, 1)) if residual_basis is None
                               else np.array(residual_basis, dtype=float, copy=True))
        self.residual_names = ("residual",) if residual_names is None else tuple(residual_names)
        if (self.residual_basis.ndim != 2 or self.residual_basis.shape[0] != self.n
                or self.residual_basis.shape[1] != len(self.residual_names)
                or not self.residual_names or self.residual_names[-1] != "residual"
                or len(set((*self.names, *self.residual_names))) != self.c + len(self.residual_names)
                or not np.all(np.isfinite(self.residual_basis)) or np.any(self.residual_basis < 0)
                or not np.array_equal(self.residual_basis[:, -1], np.ones(self.n))):
            raise ValueError("residual surfaces must be nonnegative, named, and end with iid residual")
        self.h = len(self.residual_names)
        self.k = self.c + self.h
        if not definitions or not isinstance(definitions, dict):
            raise ValueError("scientific component definitions are required")
        expected_samples = digest([source.samples[int(i)] for i in self.rows])
        if (scale.variant_identity != source.variants.identity
                or scale.sample_identity != expected_samples
                or scale.provenance.get("source") != source.identity):
            raise ValueError("scale source/sample/variant identity mismatch")
        if len(scale.mean) != self.m:
            raise ValueError("scale has the wrong variant count")
        self.scale, self.source = scale, source
        lookup = {v: i for i, v in enumerate(source.variants.ids)}
        imputation = definitions.get("main_imputation", {})
        if any(v not in lookup or not np.isfinite(mu) or not 0 <= mu <= 2
               for v, mu in imputation.items()):
            raise ValueError("invalid common frozen genotype imputation")
        self.imputation = {lookup[v]: float(mu) for v, mu in imputation.items()}
        self.block_size = positive_int(block_size, "block_size")
        self.memory_bytes = positive_int(memory_bytes, "memory_bytes")
        self.threads = positive_int(threads, "threads")
        self.native = None
        if backend == "native":
            from summit.prediction.genotype import native_module
            self.native = native_module()
            configure_prediction_threads(self.native, threads)
            self.nn = ProtectedNNOperator(threads=threads, native_module=self.native)
            self.tn = ProtectedTNOperator(threads=threads, native_module=self.native)
        elif backend == "numpy":
            self.nn, self.tn = NumpyNNOperator(threads=threads), NumpyTNOperator(threads=threads)
        else:
            raise ValueError("backend must be native or numpy")
        self.stream = RawBlockStream(source, self.rows, np.arange(self.m),
                                     block_size=block_size, threads=threads)
        self.standardizer = StandardizedBlock(self.native, threads)
        # Finite-feature summaries do not need a copy of every background
        # allele. Retain explicitly requested allele records (including all
        # supplied pairs) and authenticate the complete source axis separately.
        # Other estimators retain their existing full-axis representation.
        variant_metadata = source.variants.to_dict() if metadata_variants is None else {
            key: [getattr(source.variants, key)[lookup[v]] for v in metadata_variants]
            for key in ("ids", "chromosome", "position", "counted", "other")
        }
        if metadata_variants is not None:
            variant_metadata["genome_build"] = source.variants.genome_build
        self.metadata = dict(
            method="selected_common_scale_variant_moments_v1", definitions=definitions,
            variants=variant_metadata, sample_hash=expected_samples,
            sample_mask_hash=array_sha256(self.rows), source_identity=source.identity,
            scale_identity=scale.identity, scale_method=scale.provenance,
            fixed_effect_hash=array_sha256(self.fixed), fixed_rank=self.n-self.rank,
            modifier_hash=array_sha256(self.modifiers), weights_hash=array_sha256(self.weights),
            component_names=self.names, masses=self.masses.tolist(),
            n_samples=self.n, residual_rank=self.rank, backend=backend,
            genetic_count=self.c, residual_names=self.residual_names,
            residual_basis_hash=array_sha256(self.residual_basis),
        )
        if metadata_variants is not None:
            self.metadata.update(complete_variant_axis_identity=source.variants.identity,
                                 complete_variant_count=self.m,
                                 variant_records="explicit effect and local alleles; full axis authenticated by identity")
        self.definition_hash = canonical_sha256(self.metadata)
        self.metadata = freeze_context_mapping(self.metadata)
        self.modifiers.setflags(write=False)
        self.weights.setflags(write=False)
        self.residual_basis.setflags(write=False)

    def _nn(self, a, b):
        return self.nn.matmul(np.asfortranarray(a), np.asfortranarray(b))

    def _tn(self, a, b):
        return self.tn.matmul_tn(np.asfortranarray(a), np.asfortranarray(b))

    def project(self, value):
        return value - self._nn(self.u, self._tn(self.u, value)) if self.u.shape[1] else value.copy()

    def standardize(self, raw, variants):
        """One declared missing-value basis for all main and product features."""
        variants = np.asarray(variants, dtype=np.int64)
        x = self.standardizer.prepare(raw, np.arange(self.n), np.arange(len(variants)),
                                     self.scale.mean[variants], self.scale.inverse_scale[variants])
        for j, index in enumerate(variants):
            if int(index) in self.imputation:
                x[raw[:, j] == -127, j] = (self.imputation[int(index)] - self.scale.mean[index]) * self.scale.inverse_scale[index]
        return x

    def _blocks(self, phase):
        for start, variants, raw in self.stream.blocks(phase):
            stop = start + len(variants)
            x = self.standardize(raw, variants)
            yield start, stop, x
            # Telemetry is bounded native state, not an unbounded execution log.
            # Drain and validate each completed block before more GEMMs arrive.
            self.nn.finish_execution()

    def _features(self, x, component):
        return self.project(x * self.modifiers[:, component, None])

    def _budget(self, probes=0, traits=0, retain_variant_rows=False):
        # Conservative live arrays, including readonly publication copies and
        # block temporaries. Does not include the caller-owned raw input array.
        estimated = 8 * (self.n*self.c*probes + 3*self.m*self.c**2
                        + 4*self.n*self.k*traits + 2*traits*self.k**3
                        + 12*self.n*min(self.block_size, self.m)
                        + 4*min(self.block_size, self.m)*max(probes, self.k*traits)
                        + 4*self.n*self.u.shape[1] + 4*self.m*self.c
                        + 3*self.m*self.c*self.h
                        + (self.m*self.c*traits if retain_variant_rows else 0)) + 64*2**20
        if estimated > self.memory_bytes:
            raise MemoryError(f"selected preparation needs approximately {estimated} bytes; budget {self.memory_bytes}")
        return estimated

    def reference(self, *, nvecs=128, seed=1, exact=False):
        """Two traversals, all source sketches completed before target scoring.

        Identity probes in exact mode are limited to 2048 variants and samples.
        Random mode uses the existing global variant/probe counter stream.
        """
        if exact and max(self.n, self.m) > 2048:
            raise ValueError("exact identity reference is limited to tiny inputs")
        b = self.m if exact else positive_int(nvecs, "nvecs")
        probe_spec = GlobalVariantProbeSpec(seed, 0, b)
        planned = self._budget(probes=b)
        if not exact:
            planned+=24*b*self.k**2
            if planned>self.memory_bytes:raise MemoryError("probe covariance summaries exceed memory budget")
        self.nn.begin_execution()
        self.tn.begin_execution()
        sketches = [np.zeros((self.n, b), order="F") for _ in self.names]
        for start, stop, x in self._blocks("reference_source"):
            if exact:
                z = (np.arange(start, stop)[:, None] == np.arange(b)[None, :]).astype(float)
            else:
                if self.native is None:
                    z = generate_global_variant_probes(np.arange(start, stop), np.arange(b), root_seed=seed)
                else:
                    z = native_global_variant_probes(np.arange(start, stop), np.arange(b),
                        root_seed=seed, threads=self.threads, native_module=self.native)
                z /= np.sqrt(b)
            for c in range(self.c):
                sketches[c] += self.modifiers[:, c, None] * self._nn(x, z * np.sqrt(self.weights[start:stop, c, None]))
        sketches = [self.project(s) for s in sketches]
        # No target rows are computed until all sketches are complete.
        directional = np.empty((self.m, self.c, self.c))
        information = np.empty((self.m, self.c))
        residual_information = np.empty((self.m, self.c, self.h))
        probe_directed=None if exact else np.zeros((b,self.c,self.c))
        for start, stop, x in self._blocks("reference_target"):
            for a in range(self.c):
                f = self._features(x, a)
                weight = self.weights[start:stop, a]
                information[start:stop, a] = weight * np.einsum("ij,ij->j", f, f)
                residual_information[start:stop, a] = weight[:, None]*self._tn(f*f, self.residual_basis)
                for b_index in range(self.c):
                    product = self._tn(f, sketches[b_index])
                    directional[start:stop, a, b_index] = weight * np.einsum("ij,ij->i", product, product)
                    if probe_directed is not None:
                        probe_directed[:,a,b_index]+=b*np.einsum("i,ij,ij->j",weight,product,product)
        directed = directional.sum(axis=0)
        gram = (directed + directed.T) / (2 * np.outer(self.masses, self.masses))
        residual = residual_moments_low_rank(self.u, self.residual_basis, np.zeros(self.n))
        traces = np.r_[information.sum(axis=0) / self.masses, residual.traces]
        matrix = np.empty((self.k, self.k))
        matrix[:self.c, :self.c] = gram
        cross = residual_information.sum(axis=0)/self.masses[:, None]
        matrix[:self.c, self.c:] = cross
        matrix[self.c:, :self.c] = cross.T
        matrix[self.c:, self.c:] = residual.gram
        if np.any(traces <= 1e-12):
            raise ValueError("a component is annihilated by the fixed effects")
        telemetry = self.nn.finish_execution()
        probe_matrices=None
        if probe_directed is not None:
            probe_matrices=np.broadcast_to(matrix,(b,*matrix.shape)).copy()
            probe_matrices[:,:self.c,:self.c]=(probe_directed+probe_directed.transpose(0,2,1))/(2*np.outer(self.masses,self.masses))
        metadata = dict(self.metadata, definition_hash=self.definition_hash,
                        reference=dict(exact=exact, nvecs=b, seed=seed,
                                       probe_axis="variant", passes=2,
                                       probe_spec=probe_spec.to_metadata(),
                                       planned_workspace_bytes=planned),
                        protected_repaired_columns=self.nn.repaired_columns + self.tn.repaired_columns,
                        protected_telemetry_available=telemetry["available"])
        return SelectedReference(directional, information, self.weights, matrix, traces, residual_information, metadata,probe_matrices)

    def summarize(self, reference, phenotypes, *, trait_names, phenotype_scale="raw",
                  retain_variant_rows=False):
        """Two trait traversals for q, K_a y, and all C_acb, batched over traits."""
        if reference.metadata["definition_hash"] != self.definition_hash:
            raise ValueError("reference does not match the study definitions")
        y = np.asarray(phenotypes, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != self.n or not np.all(np.isfinite(y)):
            raise ValueError("traits must be finite on the declared sample mask")
        names = tuple(trait_names)
        if len(names) != y.shape[1] or len(set(names)) != len(names):
            raise ValueError("trait names do not match phenotype columns")
        planned = self._budget(traits=y.shape[1], retain_variant_rows=retain_variant_rows)
        self.nn.begin_execution()
        self.tn.begin_execution()
        raw_ss = np.einsum("nt,nt->t", y, y)
        y = self.project(y)
        ss = np.einsum("nt,nt->t", y, y)
        if np.any(ss <= np.finfo(float).eps * raw_ss):
            raise ValueError("zero residual phenotype variance")
        multiplier = np.ones(y.shape[1])
        if phenotype_scale == "residual_variance":
            multiplier = np.sqrt(self.rank / ss)
            y = y * multiplier
        elif phenotype_scale != "raw":
            raise ValueError("phenotype_scale must be raw or residual_variance")
        actions = np.zeros((self.k, self.n, y.shape[1]))
        q = np.zeros((self.k, y.shape[1]))
        for h in range(self.h):
            actions[self.c+h] = self.project(self.residual_basis[:, h, None]*y)
            q[self.c+h] = np.einsum("nt,nt->t", y, actions[self.c+h])
        # Rows are retained only as squared aggregates for post-hoc comparison;
        # signed pair effects are exported by the separate pair-score API.
        q_rows = np.empty((self.m, self.c, y.shape[1])) if retain_variant_rows else None
        for start, stop, x in self._blocks("trait_actions"):
            for a in range(self.c):
                f = self._features(x, a)
                score = self._tn(f, y)
                weighted = self.weights[start:stop, a, None] * score
                if q_rows is not None:
                    q_rows[start:stop, a] = score * weighted
                q[a] += (score * weighted).sum(axis=0) / self.masses[a]
                actions[a] += self._nn(f, weighted) / self.masses[a]
        cubic = np.zeros((y.shape[1], self.k, self.k, self.k))
        packed = actions.transpose(1, 0, 2).reshape(self.n, -1)
        for start, stop, x in self._blocks("trait_analytic_moments"):
            for c in range(self.c):
                z = self._tn(self._features(x, c), packed).reshape(stop-start, self.k, y.shape[1])
                cubic[:, :, c, :] += np.einsum("jat,jbt,j->tab", z, z,
                                               self.weights[start:stop, c] / self.masses[c])
        for h in range(self.h):
            cubic[:, :, self.c+h, :] = np.einsum("ant,bnt,n->tab", actions, actions, self.residual_basis[:, h])
        telemetry = self.nn.finish_execution()
        summary = EpistasisSummary(
            reference.matrix, q, reference.traces, cubic, (*self.names, *self.residual_names), names,
            self.n, self.rank, dict(reference.metadata,
                trait=dict(phenotype_scale=phenotype_scale, multiplier=multiplier.tolist(),
                           passes=2, planned_workspace_bytes=planned),
                protected_analytic_telemetry_available=telemetry["available"]),
            probe_matrices=reference.probe_matrices,
        )
        return summary, q_rows


def block_jackknife(reference, summary, q_rows, block_ids, *, trait=0):
    """Existing frozen-full-genome variant-row deletion, applied post reference.

    No reference sketch is recomputed; both retained masses rescale each
    symmetrized directional numerator. Invalid/empty deletions are reported.
    This comparison is not a calibrated small-region or single-pair test.
    """
    from .summary import normal_equations
    from summit.context.fit import ContextNormalEquations, solve_context_normal_equations
    if reference.metadata["definition_hash"] != summary.metadata.get("definition_hash"):
        raise ValueError("reference and trait summary definitions disagree")
    blocks = np.asarray(block_ids)
    if blocks.shape != (len(reference.weights),) or blocks.dtype.kind not in "iu":
        raise ValueError("block IDs must be an integer variant vector")
    labels = np.unique(blocks)
    if len(labels) < 2 or np.any(labels < 0):
        raise ValueError("at least two nonnegative nonempty blocks required")
    rows = np.asarray(q_rows)
    c = reference.weights.shape[1]
    if rows.shape != (*reference.weights.shape, len(summary.trait_names)):
        raise ValueError("trait rows have incompatible axes")
    if not np.allclose(rows.sum(axis=0)/reference.weights.sum(axis=0)[:, None], summary.rhs[:c]):
        raise ValueError("trait rows disagree with summary")
    full = normal_equations(summary, trait)
    values, failures = [], []
    for label in labels:
        keep = blocks != label
        masses = reference.weights[keep].sum(axis=0)
        try:
            if np.any(masses <= 0):
                raise ValueError("deletion empties a component")
            directed = reference.directional[keep].sum(axis=0)
            matrix = summary.matrix.copy()
            matrix[:c, :c] = (directed + directed.T) / (2*np.outer(masses, masses))
            trace = np.r_[reference.information[keep].sum(axis=0)/masses, summary.traces[c:]]
            cross = reference.residual_information[keep].sum(axis=0)/masses[:, None]
            matrix[:c, c:] = cross
            matrix[c:, :c] = cross.T
            rhs = np.r_[rows[keep, :, trait].sum(axis=0)/masses, summary.rhs[c:, trait]]
            system = ContextNormalEquations(matrix, rhs, trace, full.component_names,
                full.genetic_count, masses, (str(label),), matrix[:c, :c], matrix[:c, :c],
                summary.n_samples, summary.n_samples)
            values.append(solve_context_normal_equations(system).coefficients)
            failures.append(None)
        except ValueError as exc:
            values.append(np.full(len(summary.component_names), np.nan))
            failures.append(str(exc))
    values = np.asarray(values)
    centered = values - values.mean(axis=0)
    covariance = (len(values)-1)/len(values) * centered.T @ centered
    return dict(method="frozen_full_genome_variant_ldscore_delete_block_v1",
                block_labels=labels, coefficients=values, failures=failures, covariance=covariance,
                standard_errors=np.sqrt(np.diag(covariance)),
                interpretation="genomic_block_sensitivity_not_small_set_inference")
