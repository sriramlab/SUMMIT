"""Unprojected, single-feature specialization of SUMMIT's variant probes.

Same counters, genotype operators and protected GEMMs as the generalized
engine. One source arena is transformed in place at the pass barrier. Exact
diagonal corrections are consumed as target tiles, never retained as M by K
arrays. This path does not implement projected or cross-feature kernels.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from summit.ldscore.generalized_gxe_pass1 import NumpyNNOperator, ProtectedNNOperator
from summit.ldscore.generalized_gxe_pass2 import NumpyTNOperator, ProtectedTNOperator
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEPlanInputs, GeneralizedGxEWorkPlan, GlobalVariantProbeSpec,
    TwoPassLedger, native_global_variant_probes,
)
from summit.sumstats.binary import finite_array
from .moments import annotations_array


def plan_pcgc_reference(*, num_samples, num_variants, num_annotations, probes=256,
                        responses=1, memory_bytes=2**30, block_size=256,
                        threads=1, genotype_format='bed'):
    """Dimension-only admission, without allocating a genotype or source array.

    Includes input/ownership copies, decode/affine scratch, final moment copies,
    15% allocator headroom and native thread headroom. Metadata belonging to the
    caller and unrelated runtime allocations are outside the workspace budget.
    """
    inputs = GeneralizedGxEPlanInputs(
        num_samples=num_samples, num_variants=num_variants, num_basis=1,
        num_annotations=num_annotations, num_probes=probes,
        memory_limit_bytes=memory_bytes, preferred_variant_block_width=block_size,
        threads=threads, genotype_format=genotype_format)
    if isinstance(responses, bool) or not isinstance(responses, int) or responses < 0:
        raise ValueError('responses must be a nonnegative integer')
    if probes < 2:
        raise ValueError('PCGC reference needs at least two probes')
    n, m, k, b, r = num_samples, num_variants, num_annotations, probes, responses
    v = min(block_size, m)
    p = min(b, 128)
    batch = min(k, 4, max(1, (256*1024**2)//(8*n*p)))
    target = min(k*b, 512)

    def widths(preferred):
        values = [preferred]
        while values[-1] > 1:
            values.append(max(1, values[-1]//2))
        return values

    # Shrinking the genotype width to one before considering a smaller probe
    # tile can admit a needlessly slow full-genome run. Balance the two GEMM
    # axes; for equally sized products prefer wider genotype blocks, which
    # also reduce decode and target-call overhead. Probe COUNT never changes.
    candidates = sorted(((nv, np_, nb) for nv in widths(v) for np_ in widths(p)
                         for nb in range(batch, 0, -1)),
                        key=lambda item: (math.prod(item), item[0], item[1]), reverse=True)
    for v, p, batch in candidates:
        memory = dict(
            source_arena=8*n*k*b,
            diagonal_rhs=8*n*(k+r),
            score_inputs=8*n*(2*r+3),
            annotation_and_result_ownership=8*m*(4*k+3*r),
            # Raw decode, standardization, X^2 and an annotation subset copy.
            genotype_and_affine_scratch=8*n*v*4,
            probe_and_source_rhs=8*v*p*(batch+1),
            source_contribution=8*n*p*batch,
            annotation_diagonal_product=8*n*k,
            target_products=8*v*(target+k+r),
            same_person=8*k*k*3,
        )
        subtotal = sum(memory.values())
        memory.update(allocator_headroom=math.ceil(subtotal*.15),
                      thread_headroom=threads*8*1024**2,
                      runtime_headroom=64*1024**2)
        peak = sum(memory.values())
        if peak > (1 << 63)-1:
            raise OverflowError('PCGC reference memory exceeds signed 64-bit range')
        if peak <= memory_bytes:
            break
    else:
        raise MemoryError(f'memory budget cannot accommodate PCGC score buffers and source arena: '
                          f'at least {peak} bytes required, {memory_bytes} supplied')
    flops = 4*n*m*k*b
    if flops > (1 << 63)-1:
        raise OverflowError('PCGC reference work exceeds signed 64-bit range')
    return GeneralizedGxEWorkPlan(
        dimensions=dict(N=n, M=m, K=k, B=b, Q=1, P=1, C=k, T=r),
        work=dict(pass1_dense_flops=2*n*m*k*b, pass1_partition_flops=2*n*m*b,
                  pass2_flops=2*n*m*k*b, total_leading_flops=flops),
        memory=memory,
        tiling=dict(variant_block_width=v, source_probe_tile_width=p,
                    source_annotation_batch_width=batch, target_tile_columns=target),
        descriptor=dict(format=inputs.genotype_format, planned_complete_passes=2),
        ledger=dict(planned_reference_genotype_passes=2, planned_retained_variant_visits=2*m),
        output_size_bytes=8*(m*(k+2*r)+k*k), peak_resident_bytes=peak,
        memory_limit_bytes=memory_bytes)


@dataclass(frozen=True)
class RankOneReference:
    ldscores: np.ndarray
    same_person: np.ndarray
    scores: np.ndarray
    diagonals: np.ndarray
    plan: GeneralizedGxEWorkPlan
    diagnostics: dict


def rank_one_reference(operator, annotations, weight, *, responses=None, probes=256,
                       seed=0, threads=1, memory_bytes=2**30, block_size=256,
                       native=True, probe_square_sink=None):
    if probe_square_sink is not None and not callable(probe_square_sink):
        raise TypeError('probe_square_sink must be callable')
    a = annotations_array(annotations, operator.num_variants)
    w = finite_array('feature weight', weight, 1)
    n, m = operator.num_samples, operator.num_variants
    if w.shape != (n,):
        raise ValueError('feature weight must match the sample axis')
    if responses is None:
        responses = np.empty((n, 0))
    response = finite_array('score responses', responses, 2)
    if response.shape[0] != n:
        raise ValueError('score responses must match the sample axis')
    k, r = a.shape[1], response.shape[1]
    spec = GlobalVariantProbeSpec(root_seed=seed, probe_offset=0, probe_count=probes)
    plan = plan_pcgc_reference(num_samples=n, num_variants=m, num_annotations=k,
                               responses=r, probes=probes, memory_bytes=memory_bytes,
                               block_size=block_size, threads=threads,
                               genotype_format=operator.genotype_format)
    with np.errstate(over='ignore', invalid='ignore'):
        response2 = finite_array('diagonal responses', response**2, 2)
        w2 = finite_array('squared feature weight', w*w, 1)
        finite_array('fourth feature weight', w2*w2, 1)
    configure = getattr(operator, 'configure_block_width', None)
    if configure is not None:
        configure(plan.tiling['variant_block_width'])
    module = None
    if native:
        from summit.prediction.genotype import native_module
        from summit.prediction.runtime import configure_prediction_threads
        module = native_module()
        configure_prediction_threads(module, threads)
    nn = ProtectedNNOperator(threads=threads, native_module=module) if native else NumpyNNOperator(threads=threads)
    tn = ProtectedTNOperator(threads=threads, native_module=module) if native else NumpyTNOperator(threads=threads)
    response = np.asfortranarray(response)
    masses = a.sum(axis=0)
    partition = bool(np.all(np.count_nonzero(a, axis=1) <= 1))
    # Annotation-major probe columns give contiguous F-order slices to GEMM.
    sources = np.zeros((n, k*probes), order='F')
    diagonal_rhs = np.zeros((n, r+k), order='F')
    diagonal_rhs[:, :r] = response2
    del response2
    kernel_diagonal = diagonal_rhs[:, r:]
    ld = np.zeros((m, k))
    scores, diagonals = np.empty((m, r)), np.empty((m, r))
    v = plan.tiling['variant_block_width']
    p = plan.tiling['source_probe_tile_width']
    batch = plan.tiling['source_annotation_batch_width']
    target = plan.tiling['target_tile_columns']
    ledger = TwoPassLedger(m)
    phase_seconds = {}
    source_flops = 0
    nn.begin_execution()
    tn.begin_execution()
    for pass_number in (1, 2):
        started = time.perf_counter()
        ledger.begin_pass(pass_number)
        operator.begin_pass(pass_number)
        for start in range(0, m, v):
            stop = min(m, start+v)
            block = operator.read_block(start, stop)
            if (block.row_start != start or block.row_stop != stop or
                    block.genotype_scale_id != operator.genotype_scale_id):
                ledger.record_integrity_failure()
                raise RuntimeError('PCGC decoded block identity/scale mismatch')
            x = block.values
            squares = np.square(x, order='F')
            if pass_number == 1:
                weights = a[start:stop]
                kernel_diagonal += nn.matmul(squares, np.asfortranarray(weights))
                variants = np.arange(start, stop, dtype=np.int64)
                active = np.flatnonzero(np.any(weights != 0, axis=0))
                for ps in range(0, probes, p):
                    pe = min(probes, ps+p)
                    if native:
                        signs = native_global_variant_probes(variants, np.arange(ps, pe, dtype=np.int64),
                                    root_seed=seed, namespace=spec.namespace, threads=threads, native_module=module)
                    else:
                        from summit.ldscore.generalized_gxe_variant import generate_global_variant_probes
                        signs = generate_global_variant_probes(variants, np.arange(ps, pe, dtype=np.int64),
                                                               root_seed=seed, namespace=spec.namespace)
                    if partition:
                        for j in active:
                            selected = np.flatnonzero(weights[:, j])
                            if len(selected) == stop-start:
                                subset, local_signs = x, signs
                            else:
                                subset, local_signs = np.asfortranarray(x[:, selected]), signs[selected]
                            rhs = np.asfortranarray(np.sqrt(weights[selected, j])[:, None]*local_signs)
                            contribution = nn.matmul(subset, rhs)
                            sources[:, j*probes+ps:j*probes+pe] += contribution
                            source_flops += 2*n*len(selected)*(pe-ps)
                            del subset, local_signs, rhs, contribution
                    else:
                        for first in range(0, len(active), batch):
                            chosen = active[first:first+batch]
                            rhs = np.empty((stop-start, len(chosen)*(pe-ps)), order='F')
                            for offset, j in enumerate(chosen):
                                rhs[:, offset*(pe-ps):(offset+1)*(pe-ps)] = np.sqrt(weights[:, j, None])*signs
                            contribution = nn.matmul(x, rhs)
                            for offset, j in enumerate(chosen):
                                sources[:, j*probes+ps:j*probes+pe] += contribution[:, offset*(pe-ps):(offset+1)*(pe-ps)]
                            source_flops += 2*n*(stop-start)*len(chosen)*(pe-ps)
                            del rhs, contribution
                    del signs
            else:
                if r:
                    scores[start:stop] = tn.matmul_tn(x, response)
                correction = tn.matmul_tn(squares, diagonal_rhs)
                diagonals[start:stop] = correction[:, :r]
                ld[start:stop] = -correction[:, r:]/n**2
                del correction
                for cs in range(0, k*probes, target):
                    ce = min(k*probes, cs+target)
                    cross = tn.matmul_tn(x, sources[:, cs:ce])
                    np.square(cross, out=cross)
                    for j in range(cs//probes, (ce-1)//probes+1):
                        lo, hi = max(cs, j*probes)-cs, min(ce, (j+1)*probes)-cs
                        if probe_square_sink is not None:
                            view = cross[:,lo:hi].view()
                            view.setflags(write=False)
                            probe_square_sink(start,stop,j,cs+lo-j*probes,view)
                        ld[start:stop, j] += cross[:, lo:hi].sum(axis=1)/(probes*n**2)
                    del cross
            ledger.record_block(start, stop)
            del block, x, squares
        operator.finish_pass()
        ledger.finish_pass()
        phase_seconds[f'pass{pass_number}'] = time.perf_counter()-started
        if pass_number == 1:
            ledger.validate_pass1_barrier()
            started = time.perf_counter()
            # Transform each F-order column without an N x K x B temporary.
            for cs in range(0, k*probes, target):
                sources[:, cs:cs+target] *= w2[:, None]
            kernel_diagonal *= w2[:, None]
            same = tn.matmul_tn(kernel_diagonal, kernel_diagonal)/np.outer(masses, masses)
            kernel_diagonal *= w2[:, None]
            phase_seconds['barrier'] = time.perf_counter()-started
    # Preserve native audits/repair telemetry; the reference never retries a pass.
    ledger.repair_count = nn.repaired_columns+tn.repaired_columns
    ledger.validate_clean_completion()
    diagnostics = dict(reference_engine='pcgc_unprojected_rank_one_v1',
                       annotation_partition=partition, source_gemm_flops=source_flops,
                       phase_seconds=phase_seconds, two_pass_ledger=ledger.to_dict(),
                       protected_nn_calls=nn.calls, protected_tn_calls=tn.calls,
                       native_execution=tn.finish_execution())
    return RankOneReference(ld, same, scores, diagonals, plan, diagnostics)
