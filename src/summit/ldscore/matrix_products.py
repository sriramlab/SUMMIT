"""Shared protected matrix products and bounded native execution summaries."""
from time import perf_counter
import numpy as np


class NativeExecutionEvidence:
    """Drain the shared native buffers while retaining bounded audit evidence.

    NN and TN operators use the same collector because their native buffers are
    process-wide. Every record contributes to the cost summary. Detailed records
    are retained for small runs and sampled by arrival order for longer runs.
    Native failure and overflow counters are checked before every drain.
    """

    record_limit = 1024

    def __init__(self, module):
        self.module = module
        self._clear()

    def _clear(self):
        self.records = []
        self.output_records = []
        self.summary = {}
        self.record_count = 0
        self.output_count = 0

    def begin(self):
        self._clear()
        for name in ("reset_gemm_telemetry", "reset_native_gemm_output_numa_evidence"):
            reset = getattr(self.module, name, None)
            if callable(reset):
                reset()

    def _status(self):
        statuses = []
        for name in ("gemm_telemetry_status", "native_gemm_output_numa_evidence_status"):
            getter = getattr(self.module, name, None)
            status = dict(getter()) if callable(getter) else {}
            if int(status.get("dropped_records", 0)):
                raise RuntimeError("native GEMM telemetry overflowed")
            if int(status.get("failed_calls", 0)):
                raise RuntimeError("native protected-output NUMA verification failed")
            statuses.append(status)
        return statuses

    def check(self):
        statuses = self._status()
        if any(int(s.get("buffered_records", 0)) >= max(1, int(s.get("capacity", 16384)) // 2)
               for s in statuses):
            self._consume()

    def _consume(self):
        consume = getattr(self.module, "consume_gemm_telemetry", None)
        if callable(consume):
            for item in consume():
                record = dict(item)
                self.record_count += 1
                if len(self.records) < self.record_limit:
                    self.records.append(record)
                summary = self.summary.setdefault(record.get("operation", "unknown"),
                    dict(calls=0, wall_seconds=0., process_cpu_seconds=0., flop_count=0.))
                summary["calls"] += 1
                for key in ("wall_seconds", "process_cpu_seconds", "flop_count"):
                    summary[key] += record.get(key, 0.)
        consume = getattr(self.module, "consume_native_gemm_output_numa_evidence", None)
        if callable(consume):
            for item in consume():
                self.output_count += 1
                if len(self.output_records) < self.record_limit:
                    self.output_records.append(dict(item))

    def finish(self):
        status, output_status = self._status()
        self._consume()
        result = dict(
            available=callable(getattr(self.module, "consume_gemm_telemetry", None)),
            gemm_records=self.records, gemm_status=status,
            gemm_record_count=self.record_count, gemm_summary=self.summary,
            gemm_records_summarized=self.record_count-len(self.records),
            output_numa_evidence=self.output_records, output_numa_status=output_status,
            output_numa_record_count=self.output_count,
            output_numa_records_summarized=self.output_count-len(self.output_records),
        )
        self._clear()
        return result


def native_execution_evidence(module):
    collector = getattr(module, "_summit_execution_evidence", None)
    if collector is None:
        collector = NativeExecutionEvidence(module)
        module._summit_execution_evidence = collector
    return collector


class MatrixProducts:
    def __init__(self, *, native=True, threads=None):
        from .generalized_gxe_pass1 import ProtectedNNOperator, NumpyNNOperator
        from .generalized_gxe_pass2 import ProtectedTNOperator, NumpyTNOperator
        self.native = bool(native)
        self.module = None
        if native:
            from summit.prediction.genotype import native_module
            from summit.prediction.runtime import configure_prediction_threads
            self.module = native_module()
            if threads is None:
                info = self.module.build_info()
                threads = (int(info['blas_runtime_threads']) if info.get('blas_runtime_environment_immutable',False)
                           else int(self.module.configured_blas_threads()) or 1)
            configure_prediction_threads(self.module,threads)
        elif threads is None:
            threads = 1
        self.threads = threads
        self.left = (ProtectedNNOperator if native else NumpyNNOperator)(threads=threads)
        self.right = (ProtectedTNOperator if native else NumpyTNOperator)(threads=threads)
        self.seconds = 0.
        self.operations = {}
        self.native_records = 0

    def nn(self, a, b):
        start = perf_counter()
        result = self.left.matmul(np.asfortranarray(a),np.asfortranarray(b))
        self.seconds += perf_counter()-start
        return result

    def tn(self, a, b):
        start = perf_counter()
        result = self.right.matmul_tn(np.asfortranarray(a),np.asfortranarray(b))
        self.seconds += perf_counter()-start
        return result

    def subtract_rank(self, target, a, b):
        if self.native:
            self.module.protected_rank_update_nn(np.asfortranarray(a),np.asfortranarray(b),target,self.threads)
            native_execution_evidence(self.module).check()
        else:
            target -= a@b

    def center_strata(self, values, cases):
        if self.native:
            self.module.pcgc_center_strata(values,np.asarray(cases,dtype=np.uint8),self.threads)
        else:
            for case in (False,True):
                mask = cases == case
                if mask.sum() < 4:
                    raise ValueError('individual-sampling inference needs at least four cases and four controls')
                values[mask] -= values[mask].mean(axis=0)

    def drain(self):
        # Both orientations share process-wide native evidence. Validate before
        # consuming it, and retain aggregate costs instead of unbounded records.
        value = self.left.finish_execution()
        for operation, summary in value.get('gemm_summary',{}).items():
            target = self.operations.setdefault(operation,dict(calls=0,wall_seconds=0.,process_cpu_seconds=0.,flop_count=0.))
            self.native_records += summary['calls']
            for key in target:
                target[key] += summary[key]
        return dict(native=self.native,threads=self.threads,
            protected_nn_calls=self.left.calls,protected_tn_calls=self.right.calls,
            repaired_columns=self.left.repaired_columns+self.right.repaired_columns,
            matrix_product_seconds=self.seconds,native_records=self.native_records,
            vendor_operations=self.operations.copy())
