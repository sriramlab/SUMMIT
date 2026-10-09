"""Adapt SUMMIT's two-pass generalized reference without changing its kernels."""
from __future__ import annotations

import numpy as np

from summit.context.spec import array_sha256
from summit.ldscore.generalized_gxe_pass1 import GeneralizedGxEPass1Executor, NumpyNNOperator, ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import GeneralizedGxEPass2Executor, NumpyTNOperator, ProtectedTNOperator
from summit.ldscore.generalized_gxe_variant import GeneralizedGxEPlanInputs, GlobalVariantProbeSpec, plan_generalized_gxe_variant_work
from summit.sumstats.binary import finite_array
from .moments import BinaryMoments, annotations_array, method_vectors


class ProbeGramCollector:
    """Small per-probe normal matrices; same-person terms are deterministic."""
    def __init__(self,annotations,num_contexts,probes,n_samples,*,native=False,threads=1):
        if type(probes) is not int or probes < 2:
            raise ValueError("reference-probe uncertainty needs at least two probes")
        self.annotations = annotations
        self.pairs = num_contexts*(num_contexts+1)//2
        self.n_samples = n_samples
        from summit.ldscore.matrix_products import MatrixProducts
        self.products = MatrixProducts(native=native,threads=threads)
        c = annotations.shape[1]*self.pairs
        self.directed = np.zeros((probes,c,c))

    def __call__(self,start,stop,annotation,probe_start,panels,plan):
        p = self.pairs
        width = panels.shape[-1]
        for target in range(p):
            for source in range(p):
                product = np.zeros((stop-start,width))
                for term in plan.terms[target][source]:
                    product += panels[term.first_target,term.first_source]*panels[term.second_target,term.second_source]
                values = self.products.tn(self.annotations[start:stop],product)
                self.directed[probe_start:probe_start+width,target::p,annotation*p+source] += values.T

    def deviations(self):
        mass = np.repeat(self.annotations.sum(0),self.pairs)
        value = self.n_samples**2*self.directed/np.outer(mass,mass)[None]
        return value-value.mean(0)

    def add_squared(self,start,stop,annotation,probe_start,squared):
        if self.pairs != 1:
            raise ValueError("rank-one probe products require one context")
        width = squared.shape[1]
        values = self.products.tn(self.annotations[start:stop],squared)
        self.directed[probe_start:probe_start+width,:,annotation] += values.T/self.n_samples**2


def reference_provenance(options):
    """Small replay metadata; large feature and genotype arrays stay private."""
    probe = GlobalVariantProbeSpec(root_seed=options.get("seed", 0), probe_offset=0,
                                   probe_count=options.get("probes", 256))
    result = dict(probes=probe.to_metadata(), threads=options.get("threads", 1),
                  requested_block_size=options.get("block_size", 256),
                  requested_memory_bytes=options.get("memory_bytes", 2**30),
                  backend="native" if options.get("native", True) else "numpy_differential")
    if options.get("native", True):
        from summit.prediction.genotype import native_module
        info = native_module().build_info()
        result["native_build"] = {k: info.get(k) for k in
                                  ("backend_version", "source_commit", "source_tree_sha256", "blas_vendor", "gemm_execution_mode")}
    return result


class ScoredOperator:
    """Add batched raw trait products to the existing second genotype pass."""
    def __init__(self, operator, responses, tn, nn, *, diagonal_annotations=None, diagonal_weights=None,
                 diagonal_responses=None, kernel_summary_weights=None):
        self.operator, self.tn, self.nn = operator, tn, nn
        self.responses = np.asfortranarray(finite_array("score responses", responses, 2))
        with np.errstate(over='ignore', invalid='ignore'):
            squares = self.responses**2 if diagonal_responses is None else diagonal_responses
        self.response_squares = np.asfortranarray(finite_array("diagonal responses", squares, 2))
        if len(self.responses) != operator.num_samples:
            raise ValueError("score responses do not match the sample axis")
        if len(self.response_squares) != operator.num_samples:
            raise ValueError("diagonal responses do not match the sample axis")
        self.scores = np.empty((operator.num_variants, self.responses.shape[1]))
        self.diagonals = np.empty((operator.num_variants, self.response_squares.shape[1]))
        self.diagonal_annotations = diagonal_annotations
        self.diagonal_rhs = None
        self.kernel_summary_weights = kernel_summary_weights
        self.kernel_summary = None
        if diagonal_annotations is not None:
            self.diagonal_weights = np.asfortranarray(finite_array("diagonal weights", diagonal_weights, 2))
            if len(self.diagonal_weights) != operator.num_samples or not self.diagonal_weights.shape[1]:
                raise ValueError("diagonal weights must match the sample axis and have nonempty columns")
            self.kernel_diagonal_numerators = np.zeros((operator.num_samples, diagonal_annotations.shape[1]))
            self.reference_diagonal_rows = np.empty((operator.num_variants, self.diagonal_weights.shape[1], diagonal_annotations.shape[1]))

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def read_block(self, start, stop):
        block = self.operator.read_block(start, stop)
        x = block.values
        squares = None
        if self.diagonal_annotations is not None:
            squares = np.asfortranarray(x*x)
            if self.operator.observed_passes == 1:
                self.kernel_diagonal_numerators += self.nn.matmul(
                    squares, np.asfortranarray(self.diagonal_annotations[start:stop]))
        if self.operator.observed_passes == 2:
            if self.responses.shape[1]:
                self.scores[start:stop] = self.tn.matmul_tn(x, self.responses)
            if self.diagonal_rhs is None:
                if self.diagonal_annotations is None:
                    self.diagonal_rhs = self.response_squares
                else:
                    if self.kernel_summary_weights is not None:
                        self.kernel_summary = self.tn.matmul_tn(
                            self.kernel_summary_weights, np.asfortranarray(self.kernel_diagonal_numerators))
                    k = self.diagonal_annotations.shape[1]
                    r = self.response_squares.shape[1]
                    self.diagonal_rhs = np.empty((len(x), r+k*self.diagonal_weights.shape[1]), order="F")
                    self.diagonal_rhs[:, :r] = self.response_squares
                    for q, weight in enumerate(self.diagonal_weights.T):
                        np.multiply(self.kernel_diagonal_numerators, weight[:, None],
                                    out=self.diagonal_rhs[:, r+q*k:r+(q+1)*k])
                    self.kernel_diagonal_numerators = None
            if squares is None:
                squares = np.asfortranarray(x*x)
            values = self.tn.matmul_tn(squares, self.diagonal_rhs)
            r = self.response_squares.shape[1]
            self.diagonals[start:stop] = values[:, :r]
            if self.diagonal_annotations is not None:
                self.reference_diagonal_rows[start:stop] = values[:, r:].reshape(stop-start, *self.reference_diagonal_rows.shape[1:])
        return block


def generalized_reference(operator, annotations, basis, *, responses=None, probes=256,
                          seed=0, threads=1, memory_bytes=2**30, block_size=256, native=True,
                          collect_diagonal_rows=False, diagonal_responses=None, diagonal_weights=None,
                          admit_resident_output=False, kernel_summary_weights=None, probe_product_sink=None):
    """Exactly two genotype traversals; risk fitting/scale preparation precede it.

    There is intentionally no projection argument: applying P D X would change
    the binary moment and create non-diagonal residual noise.
    """
    a = annotations_array(annotations, operator.num_variants)
    phi = finite_array("basis", basis, 2)
    if phi.shape[0] != operator.num_samples or phi.shape[1] < 1:
        raise ValueError("basis must have shape N by Q")
    module = None
    if native:
        from summit.prediction.genotype import native_module
        from summit.prediction.runtime import configure_prediction_threads
        module = native_module()
        configure_prediction_threads(module, threads)
    nn = ProtectedNNOperator(threads=threads, native_module=module) if native else NumpyNNOperator(threads=threads)
    tn = ProtectedTNOperator(threads=threads, native_module=module) if native else NumpyTNOperator(threads=threads)
    score_bytes = 0
    if kernel_summary_weights is not None:
        kernel_summary_weights = np.asfortranarray(finite_array("kernel summary weights", kernel_summary_weights, 2))
        if not collect_diagonal_rows or len(kernel_summary_weights) != operator.num_samples:
            raise ValueError("kernel summary weights require source diagonals and the sample axis")
        score_bytes += 8*kernel_summary_weights.shape[1]*(operator.num_samples+a.shape[1])
    if collect_diagonal_rows:
        if diagonal_weights is None:
            if phi.shape[1] != 1:
                raise ValueError("multiple features require explicit per-SNP diagonal weights")
            diagonal_weights = phi**4
        if responses is None:
            responses = np.empty((operator.num_samples, 0))
    if responses is not None:
        responses = finite_array("score responses", responses, 2)
        n, m, r = operator.num_samples, operator.num_variants, responses.shape[1]
        s = r if diagonal_responses is None else finite_array("diagonal responses", diagonal_responses, 2).shape[1]
        b = min(block_size, m)
        score_bytes += 8*((n+m)*(r+s) + n*b)
        if collect_diagonal_rows:
            diagonal_weights = finite_array("diagonal weights", diagonal_weights, 2)
            q, k = diagonal_weights.shape[1], a.shape[1]
            score_bytes += 8*(n*k + m*q*k + n*q + (n+b)*(s+q*k))
    if memory_bytes <= score_bytes:
        raise MemoryError("memory budget cannot accommodate PCGC score buffers")
    output_bytes = (8*operator.num_variants*(phi.shape[1]*(phi.shape[1]+1)//2)**2*a.shape[1]
                    if admit_resident_output else 0)
    planning_budget = memory_bytes-score_bytes-output_bytes
    if planning_budget <= 0:
        raise MemoryError("memory budget cannot accommodate resident contextual output")
    plan = plan_generalized_gxe_variant_work(GeneralizedGxEPlanInputs(
        num_samples=operator.num_samples, num_variants=operator.num_variants,
        num_basis=phi.shape[1], num_annotations=a.shape[1], num_probes=probes,
        memory_limit_bytes=planning_budget, genotype_format=operator.genotype_format, threads=threads,
        preferred_variant_block_width=block_size,
        preferred_rhs_tile_columns=phi.shape[1]**2 * min(probes, 64), rhs_policy="tiled"))
    if admit_resident_output:
        from dataclasses import replace
        plan = replace(plan, memory_limit_bytes=memory_bytes-score_bytes)
    # Admit the complete plan BEFORE allocating M-by-trait/annotation buffers.
    configure = getattr(operator, 'configure_block_width', None)
    if configure is not None:
        configure(plan.tiling['variant_block_width'])
    if responses is not None:
        operator = ScoredOperator(operator, responses, tn, nn,
                                  diagonal_annotations=a if collect_diagonal_rows else None,
                                  diagonal_weights=diagonal_weights,
                                  diagonal_responses=diagonal_responses,
                                  kernel_summary_weights=kernel_summary_weights)
        operator.pcgc_score_buffer_bytes = score_bytes
    fixed = np.empty((operator.num_samples, 0))
    names = tuple(f"annotation_{i}" for i in range(a.shape[1]))
    p1 = GeneralizedGxEPass1Executor(
        genotype_operator=operator, basis=phi, fixed_effect_basis=fixed,
        annotations=a, annotation_names=names, annotation_masses=a.sum(axis=0),
        probe_spec=GlobalVariantProbeSpec(root_seed=seed, probe_offset=0, probe_count=probes),
        work_plan=plan, nn_operator=nn,
        annotation_tile_width=(plan.tiling["source_annotation_batch_width"] if admit_resident_output else a.shape[1]),
        probe_tile_width=(min(probes, plan.tiling["source_probe_tile_width"]) if admit_resident_output else min(probes, 64)),
        native_probe_module=module if native else False).execute()
    p2 = GeneralizedGxEPass2Executor(
        pass1_result=p1, genotype_operator=operator, basis=phi, fixed_effect_basis=fixed,
        annotations=a, annotation_names=names, work_plan=plan, tn_operator=tn,
        probe_tile_width=(min(probes, plan.tiling["rhs_tile_columns"]//phi.shape[1]**2)
                          if admit_resident_output else min(probes, 64)),probe_product_sink=probe_product_sink).execute()
    return p2, operator, plan


def population_ld_reference(operator, annotations, **options):
    """Finite-reference corrected unweighted LD rows from the two-pass engine.

    This is an external-reference *approximation* for PCGC and requires the
    risk/genotype independence assumption; transfer is an explicit later step.
    """
    n = operator.num_samples
    if n < 2:
        raise ValueError("external reference needs at least two samples")
    from .rank_one import rank_one_reference
    ref = rank_one_reference(operator, annotations, np.ones(n), **options)
    return ref.ldscores*n/(n-1), ref.plan


def contract_reference(reference, coefficients):
    """Contract symmetric basis kernels: off-diagonal coefficient is c_q c_r.

    The stored off-diagonal kernel already includes both orientations.
    """
    c = finite_array("basis coefficients", coefficients, 1)
    if len(c) != 1 + max(max(pair) for pair in reference.pair_table):
        raise ValueError("coefficient and basis axes do not match")
    pair_weights = np.array([c[q]*c[r] for q, r in reference.pair_table])
    k = len(reference.annotation_masses)
    contraction = np.zeros((len(reference.component_table), k))
    for index, (annotation, pair) in enumerate(reference.component_table):
        contraction[index, annotation] = pair_weights[pair]
    ld = np.einsum("mpc,p,ca->ma", reference.directional_ldscores, pair_weights, contraction, optimize=True)
    same_person = contraction.T @ reference.same_person @ contraction
    return ld, same_person


def prepare_moments(operator, annotations, risk, method="pcgc", *, basis=None,
                    coefficients=None, allow_basis_approximation=False, **reference_options):
    if operator.num_samples != risk.n_samples:
        raise ValueError("risk and genotype sample axes do not match")
    diagnostics = {}
    if method == "pcgc-basis":
        phi = finite_array("basis", basis, 2)
        c = finite_array("basis coefficients", coefficients, 1)
        if phi.shape != (risk.n_samples, len(c)):
            raise ValueError("basis/coefficients must match risk sample axis")
        d_hat = phi @ c
        if np.any(d_hat <= 0):
            raise ValueError("contracted risk sensitivity must be positive")
        error = d_hat - risk.sensitivity
        relative_error = float(np.linalg.norm(error) / np.linalg.norm(risk.sensitivity))
        maximum_relative_error = float(np.max(np.abs(error / risk.sensitivity)))
        diagnostics = {"basis_relative_error": relative_error,
                       "basis_max_relative_error": maximum_relative_error,
                       "basis_exact": bool(relative_error < 1e-12 and maximum_relative_error < 1e-10)}
        if not diagnostics["basis_exact"] and not allow_basis_approximation:
            raise ValueError("basis does not span the sensitivity; approximation must be explicitly requested")
        response = d_hat * risk.z
    else:
        if basis is not None or coefficients is not None:
            raise ValueError("basis arguments are only valid for pcgc-basis")
        weight, response = method_vectors(risk, method)
        phi, c = weight[:, None], np.ones(1)
        if method == "pcgc-ld":
            raise ValueError("pcgc-ld requires an explicitly transferred external reference")
    # A single requested contraction is exactly the rank-one feature X*(phi c).
    # Avoid constructing Q(Q+1)/2 kernels only to discard them after pass 2.
    from .rank_one import rank_one_reference
    ref = rank_one_reference(operator, annotations, phi @ c, responses=response[:, None], **reference_options)
    moments = BinaryMoments(annotations, ref.ldscores, ref.same_person, ref.scores[:, 0]**2 - ref.diagonals[:, 0],
                            risk.n_samples, method, risk.covariate_variance)
    diagnostics.update(ref.diagnostics)
    diagnostics.update({"genotype_passes": operator.observed_passes,
                        "peak_planned_reference_bytes": ref.plan.peak_resident_bytes,
                        "peak_planned_total_bytes": ref.plan.peak_resident_bytes,
                        "reference_plan": ref.plan.to_dict(),
                        "reference_feature_columns": 1,
                        "reference_kind": "study_specific_variant_probes",
                        "genotype_scale_id": operator.genotype_scale_id,
                        "reference_execution": reference_provenance(reference_options),
                        "feature_basis_sha256": array_sha256(phi),
                        "feature_coefficients": c.tolist()})
    return moments, diagnostics
