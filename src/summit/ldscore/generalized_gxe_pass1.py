"""First physical genotype pass for generalized per-variant GxE LD scores.

This module is a scientific orchestration boundary, not a genotype decoder.
Production readers provide already-imputed genotype blocks on one declared
scale.  Every source tile is consumed while its block is resident, after which
contextual sources are constructed as ``P diag(phi_q) V_k`` without genotype
access.  Target scoring belongs to pass 2 and is intentionally absent here.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import resource
import stat
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
)
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEWorkPlan,
    GlobalVariantProbeSpec,
    TwoPassLedger,
    native_global_variant_probes,
)


Array = np.ndarray
ReadBlock = Callable[[int, int], Array]


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. SUMMIT's production platform is
    # Linux, but keep the diagnostic correct on either platform.
    return value if os.uname().sysname == "Darwin" else value * 1024


def _readonly(value: Array) -> Array:
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class DescriptorIdentity:
    device: int
    inode: int
    byte_count: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, descriptor: int, label: str) -> "DescriptorIdentity":
        try:
            observed = os.fstat(descriptor)
        except OSError as exc:
            raise RuntimeError(
                f"stable genotype descriptor {label!r} is unavailable"
            ) from exc
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(
                f"stable genotype descriptor {label!r} must identify a regular file"
            )
        return cls(
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "byte_count": self.byte_count,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


class StableDescriptorGuard:
    """Authenticate caller-owned descriptors before and after every decode."""

    def __init__(self, descriptors: Mapping[str, int]) -> None:
        if not isinstance(descriptors, Mapping) or not descriptors:
            raise ValueError("stable genotype descriptors must be a nonempty mapping")
        owned: dict[str, int] = {}
        identities: dict[str, DescriptorIdentity] = {}
        for raw_label, raw_descriptor in descriptors.items():
            label = str(raw_label)
            if not label or label in owned:
                raise ValueError("stable genotype descriptor labels must be unique")
            if (
                isinstance(raw_descriptor, bool)
                or not isinstance(raw_descriptor, int)
                or raw_descriptor < 0
            ):
                raise ValueError(f"descriptor {label!r} must be a nonnegative integer")
            owned[label] = raw_descriptor
            identities[label] = DescriptorIdentity.capture(raw_descriptor, label)
        self._descriptors = MappingProxyType(owned)
        self._identities = MappingProxyType(identities)

    def verify(self, phase: str) -> None:
        changed: list[str] = []
        for label, descriptor in self._descriptors.items():
            try:
                current = DescriptorIdentity.capture(descriptor, label)
            except (RuntimeError, ValueError):
                changed.append(label)
                continue
            if current != self._identities[label]:
                changed.append(label)
        if changed:
            raise RuntimeError(
                "stable genotype descriptors changed during "
                f"{phase}; changed inputs: {changed}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "sealed_fstat_identity_before_after_each_block_v1",
            "files": {
                label: identity.to_dict()
                for label, identity in self._identities.items()
            },
        }


@dataclass(frozen=True)
class DecodedGenotypeBlock:
    row_start: int
    row_stop: int
    values: Array
    genotype_scale_id: str


class MatureSequentialGenotypeOperator:
    """Compose a mature block reader without copying its decoding machinery.

    ``read_block`` is normally a bound mature ``_read_genotype_block`` method.
    It must return mean-imputed FP64 values on the declared common genotype
    scale. Descriptor ownership and closure remain with the caller.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        num_variants: int,
        genotype_format: str,
        genotype_scale_id: str,
        stable_descriptors: Mapping[str, int],
        read_block: ReadBlock,
        backend_name: str,
    ) -> None:
        self.num_samples = _positive_int("num_samples", num_samples)
        self.num_variants = _positive_int("num_variants", num_variants)
        if genotype_format not in {"bed", "pgen"}:
            raise ValueError("genotype_format must be 'bed' or 'pgen'")
        self.genotype_format = genotype_format
        if not isinstance(genotype_scale_id, str) or not genotype_scale_id:
            raise ValueError("genotype_scale_id must be a nonempty string")
        self.genotype_scale_id = genotype_scale_id
        if not callable(read_block):
            raise TypeError("read_block must be callable")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError("backend_name must be a nonempty string")
        self.backend_name = backend_name
        self._read_callback = read_block
        self._guard = StableDescriptorGuard(stable_descriptors)
        self._active_pass: int | None = None
        self._next_variant = {1: 0, 2: 0}
        self.observed_passes = 0
        self.observed_variant_visits = 0
        self.blocks_read = 0
        self.decode_seconds = 0.0

    @classmethod
    def from_genomewide_estimator(
        cls, estimator: Any
    ) -> "MatureSequentialGenotypeOperator":
        """Bind the established ``GenomewideEnvLDScore`` reader operation."""
        descriptors = getattr(estimator, "_genotype_descriptors", None)
        if not isinstance(descriptors, Mapping) or not descriptors:
            raise ValueError("mature estimator has no stable genotype descriptors")
        genotype_scale = str(getattr(estimator, "genotype_scale", ""))
        ddof = int(getattr(estimator, "ddof"))
        eps = float(getattr(estimator, "eps_var"))
        scale_id = (
            f"mean_imputed_{genotype_scale}_ddof={ddof}_eps={eps:.17g}_fp64_v1"
        )
        callback = getattr(estimator, "_read_genotype_block", None)
        if not callable(callback):
            raise ValueError("mature estimator lacks its genotype block reader")
        return cls(
            num_samples=int(getattr(estimator, "nsamp")),
            num_variants=int(getattr(estimator, "nsnps")),
            genotype_format=str(getattr(estimator, "genotype_format")),
            genotype_scale_id=scale_id,
            stable_descriptors=descriptors,
            read_block=lambda start, stop: callback(
                start, stop, memory_order="F"
            ),
            backend_name="GenomewideEnvLDScore._read_genotype_block",
        )

    @property
    def descriptor_record(self) -> dict[str, Any]:
        return self._guard.to_dict()

    def begin_pass(self, pass_number: int) -> None:
        if pass_number not in {1, 2}:
            raise ValueError("pass_number must be 1 or 2")
        if self._active_pass is not None:
            raise RuntimeError("genotype operator already has an active pass")
        if pass_number != self.observed_passes + 1:
            raise RuntimeError("genotype operator passes must occur once in order")
        self._guard.verify(f"pass-{pass_number} entry")
        self._active_pass = pass_number
        self.observed_passes += 1

    def read_block(self, row_start: int, row_stop: int) -> DecodedGenotypeBlock:
        if self._active_pass is None:
            raise RuntimeError("genotype operator has no active pass")
        expected = self._next_variant[self._active_pass]
        if row_start != expected or row_stop <= row_start or row_stop > self.num_variants:
            raise RuntimeError(
                "genotype operator requires contiguous retained-order blocks"
            )
        self._guard.verify("pre-decode authentication")
        started = time.perf_counter()
        values = self._read_callback(row_start, row_stop)
        self.decode_seconds += time.perf_counter() - started
        self._guard.verify("post-decode authentication")
        array = np.asarray(values)
        expected_shape = (self.num_samples, row_stop - row_start)
        if array.dtype != np.dtype(np.float64) or array.shape != expected_shape:
            raise RuntimeError(
                "mature genotype reader returned an unexpected dtype or shape"
            )
        if not array.flags.f_contiguous:
            raise RuntimeError(
                "mature genotype reader must return a Fortran-contiguous FP64 block"
            )
        if not np.all(np.isfinite(array)):
            raise RuntimeError(
                "mature genotype reader did not finish imputation/scaling"
            )
        self._next_variant[self._active_pass] = row_stop
        self.observed_variant_visits += row_stop - row_start
        self.blocks_read += 1
        return DecodedGenotypeBlock(
            row_start, row_stop, array, self.genotype_scale_id
        )

    def finish_pass(self) -> None:
        if self._active_pass is None:
            raise RuntimeError("genotype operator has no active pass")
        if self._next_variant[self._active_pass] != self.num_variants:
            raise RuntimeError("genotype operator pass ended before all variants")
        self._guard.verify(f"pass-{self._active_pass} exit")
        self._active_pass = None


class ArraySequentialGenotypeOperator:
    """In-memory differential/benchmark operator; never a production decoder."""

    def __init__(
        self,
        genotype: Array,
        *,
        genotype_scale_id: str = "test_common_scale_v1",
        genotype_format: str = "bed",
    ) -> None:
        array = np.array(genotype, dtype=np.float64, order="F", copy=True)
        if array.ndim != 2 or not np.all(np.isfinite(array)):
            raise ValueError("test genotype must be a finite two-dimensional matrix")
        if genotype_format not in {"bed", "pgen"}:
            raise ValueError("genotype_format must be 'bed' or 'pgen'")
        if not isinstance(genotype_scale_id, str) or not genotype_scale_id:
            raise ValueError("genotype_scale_id must be a nonempty string")
        self._genotype = _readonly(array)
        self.num_samples, self.num_variants = array.shape
        self.genotype_format = genotype_format
        self.genotype_scale_id = genotype_scale_id
        self.backend_name = "in_memory_differential_only"
        self.descriptor_record = {
            "policy": "in_memory_differential_only",
            "files": {},
        }
        self._active_pass: int | None = None
        self._next_variant = {1: 0, 2: 0}
        self.observed_passes = 0
        self.observed_variant_visits = 0
        self.blocks_read = 0
        self.decode_seconds = 0.0

    def begin_pass(self, pass_number: int) -> None:
        if pass_number not in {1, 2} or pass_number != self.observed_passes + 1:
            raise RuntimeError("array genotype passes must occur once in order")
        if self._active_pass is not None:
            raise RuntimeError("array genotype operator already has an active pass")
        self._active_pass = pass_number
        self.observed_passes += 1

    def read_block(self, row_start: int, row_stop: int) -> DecodedGenotypeBlock:
        if self._active_pass is None:
            raise RuntimeError("array genotype operator has no active pass")
        expected = self._next_variant[self._active_pass]
        if row_start != expected or row_stop <= row_start or row_stop > self.num_variants:
            raise RuntimeError(
                "array genotype operator requires contiguous retained-order blocks"
            )
        values = self._genotype[:, row_start:row_stop]
        if not values.flags.f_contiguous:
            raise RuntimeError("internal genotype block lost Fortran layout")
        self._next_variant[self._active_pass] = row_stop
        self.observed_variant_visits += row_stop - row_start
        self.blocks_read += 1
        return DecodedGenotypeBlock(
            row_start, row_stop, values, self.genotype_scale_id
        )

    def finish_pass(self) -> None:
        if self._active_pass is None:
            raise RuntimeError("array genotype operator has no active pass")
        if self._next_variant[self._active_pass] != self.num_variants:
            raise RuntimeError("array genotype pass ended before all variants")
        self._active_pass = None


class NumpyNNOperator:
    """Independent dense matrix-product backend used only for differential tests."""

    backend_name = "numpy_differential_only"

    def __init__(self, *, threads: int = 1) -> None:
        self.threads = _positive_int("threads", threads)
        self.calls = 0
        self.repaired_columns = 0

    def begin_execution(self) -> None:
        self.calls = 0
        self.repaired_columns = 0

    def matmul(self, left: Array, right: Array) -> Array:
        self.calls += 1
        return np.asfortranarray(left @ right, dtype=np.float64)

    def finish_execution(self) -> dict[str, Any]:
        return {
            "available": False,
            "reason": "NumPy differential backend has no native telemetry",
            "gemm_records": [],
            "gemm_status": {},
            "output_numa_evidence": [],
            "output_numa_status": {},
        }


class ProtectedNNOperator:
    """Thin owner of the mature protected-NN entry point and its telemetry."""

    backend_name = "gxeldcore.protected_matmul_nn"

    def __init__(self, *, threads: int, native_module: Any | None = None) -> None:
        self.threads = _positive_int("threads", threads)
        if native_module is None:
            from summit import gxeldcore as native_module
        self._module = native_module
        protected = getattr(native_module, "protected_matmul_nn", None)
        if not callable(protected):
            raise RuntimeError("native extension lacks protected_matmul_nn")
        self._protected = protected
        configure = getattr(native_module, "configure_blas_threads", None)
        if callable(configure) and int(configure(self.threads)) != self.threads:
            raise RuntimeError("native extension configured an unexpected thread count")
        self.calls = 0
        self.repaired_columns = 0

    def begin_execution(self) -> None:
        self.calls = 0
        self.repaired_columns = 0
        reset = getattr(self._module, "reset_gemm_telemetry", None)
        if callable(reset):
            reset()
        reset_output = getattr(
            self._module, "reset_native_gemm_output_numa_evidence", None
        )
        if callable(reset_output):
            reset_output()

    def matmul(self, left: Array, right: Array) -> Array:
        output, repaired = self._protected(left, right, self.threads)
        result = np.asarray(output)
        expected = (left.shape[0], right.shape[1])
        if result.dtype != np.dtype(np.float64) or result.shape != expected:
            raise RuntimeError("protected NN returned an unexpected dtype or shape")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("protected NN returned non-finite output")
        repaired_count = int(repaired)
        if repaired_count < 0:
            raise RuntimeError("protected NN returned a negative repair count")
        self.calls += 1
        self.repaired_columns += repaired_count
        return result

    def finish_execution(self) -> dict[str, Any]:
        status_getter = getattr(self._module, "gemm_telemetry_status", None)
        consumer = getattr(self._module, "consume_gemm_telemetry", None)
        output_status_getter = getattr(
            self._module, "native_gemm_output_numa_evidence_status", None
        )
        output_consumer = getattr(
            self._module, "consume_native_gemm_output_numa_evidence", None
        )
        status = dict(status_getter()) if callable(status_getter) else {}
        if int(status.get("dropped_records", 0)) != 0:
            raise RuntimeError("native GEMM telemetry overflowed")
        output_status = (
            dict(output_status_getter()) if callable(output_status_getter) else {}
        )
        if int(output_status.get("failed_calls", 0)) != 0:
            raise RuntimeError("native protected-output NUMA verification failed")
        return {
            "available": callable(consumer),
            "gemm_records": (
                [dict(item) for item in consumer()] if callable(consumer) else []
            ),
            "gemm_status": status,
            "output_numa_evidence": (
                [dict(item) for item in output_consumer()]
                if callable(output_consumer)
                else []
            ),
            "output_numa_status": output_status,
        }


@dataclass(frozen=True)
class GeneralizedGxEPass1Result:
    contextual_sources: Array
    same_person: Array
    annotation_masses: Array
    pair_table: tuple[tuple[int, int], ...]
    component_table: tuple[tuple[int, int], ...]
    ledger: TwoPassLedger
    genotype_scale_id: str
    genotype_operator_identity: int
    maximum_projection_leakage: float
    maximum_relative_projection_leakage: float
    same_person_presymmetry_error: float
    base_sources: Array | None
    telemetry: Mapping[str, Any]
    pass1_barrier_sealed: bool = True


class GeneralizedGxEPass1Executor:
    """Execute and seal the global-source half of the exact two-pass design."""

    def __init__(
        self,
        *,
        genotype_operator: Any,
        basis: Array,
        fixed_effect_basis: Array,
        annotations: Array,
        annotation_names: Sequence[str],
        annotation_masses: Sequence[float] | Array,
        probe_spec: GlobalVariantProbeSpec,
        work_plan: GeneralizedGxEWorkPlan,
        nn_operator: Any,
        annotation_tile_width: int = 1,
        probe_tile_width: int | None = None,
        same_person_sample_tile_width: int = 4096,
        projection_tolerance: float = 2.0e-11,
        retain_base_sources: bool = False,
        native_probe_module: Any | None = None,
    ) -> None:
        self.genotype_operator = genotype_operator
        self.nn_operator = nn_operator
        self.work_plan = work_plan
        self.probe_spec = probe_spec
        self.annotation_tile_width = _positive_int(
            "annotation_tile_width", annotation_tile_width
        )
        self.same_person_sample_tile_width = _positive_int(
            "same_person_sample_tile_width", same_person_sample_tile_width
        )
        if probe_tile_width is None:
            probe_tile_width = min(
                probe_spec.probe_count,
                int(work_plan.tiling["rhs_tile_columns"]),
            )
        self.probe_tile_width = _positive_int(
            "probe_tile_width", probe_tile_width
        )
        if not np.isfinite(projection_tolerance) or projection_tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive and finite")
        self.projection_tolerance = float(projection_tolerance)
        if not isinstance(retain_base_sources, bool):
            raise ValueError("retain_base_sources must be boolean")
        self.retain_base_sources = retain_base_sources
        self.native_probe_module = native_probe_module

        n = int(genotype_operator.num_samples)
        m = int(genotype_operator.num_variants)
        basis_array = np.array(basis, dtype=np.float64, order="F", copy=True)
        fixed = np.array(
            fixed_effect_basis, dtype=np.float64, order="F", copy=True
        )
        weights = np.array(annotations, dtype=np.float64, order="C", copy=True)
        if basis_array.ndim != 2 or basis_array.shape[0] != n:
            raise ValueError("basis must have shape (N,Q)")
        if fixed.ndim != 2 or fixed.shape[0] != n:
            raise ValueError("fixed_effect_basis must have shape (N,R)")
        if weights.ndim != 2 or weights.shape[0] != m:
            raise ValueError("annotations must have shape (M,K)")
        if not (
            np.all(np.isfinite(basis_array))
            and np.all(np.isfinite(fixed))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError("basis, fixed effects, and annotations must be finite")
        if np.any(weights < 0.0):
            raise ValueError("annotations must be nonnegative")
        names = tuple(str(name) for name in annotation_names)
        if len(names) != weights.shape[1]:
            raise ValueError("annotation_names must match the annotation columns")
        pair_index = ContextPairIndex(basis_array.shape[1])
        component_index = ContextComponentIndex(names, pair_index)
        masses = np.asarray(annotation_masses, dtype=np.float64)
        observed_masses = np.sum(weights, axis=0, dtype=np.float64)
        if (
            masses.shape != (weights.shape[1],)
            or not np.all(np.isfinite(masses))
            or np.any(masses <= 0.0)
            or not np.allclose(masses, observed_masses, rtol=2.0e-15, atol=0.0)
        ):
            raise ValueError(
                "annotation masses must be positive and match full annotations"
            )
        if fixed.shape[1]:
            gram = fixed.T @ fixed
            if not np.allclose(
                gram, np.eye(fixed.shape[1]), rtol=2.0e-13, atol=2.0e-13
            ):
                raise ValueError("fixed_effect_basis must be orthonormal")
        dimensions = dict(work_plan.dimensions)
        expected_dimensions = {
            "N": n,
            "M": m,
            "Q": basis_array.shape[1],
            "P": len(pair_index),
            "K": weights.shape[1],
            "C": len(component_index),
            "B": probe_spec.probe_count,
        }
        for name, expected in expected_dimensions.items():
            if int(dimensions.get(name, -1)) != expected:
                raise ValueError(f"work plan dimension {name} does not match inputs")
        if work_plan.descriptor.get("planned_complete_passes") != 2:
            raise ValueError("work plan must admit exactly two physical passes")
        if work_plan.descriptor.get("format") != genotype_operator.genotype_format:
            raise ValueError("work plan genotype format does not match its operator")
        if work_plan.peak_resident_bytes > work_plan.memory_limit_bytes:
            raise MemoryError("work plan was not admitted under its memory limit")
        if probe_spec.probe_count < 2:
            raise ValueError("same-person estimation requires at least two probes")
        if self.probe_tile_width > probe_spec.probe_count:
            raise ValueError("probe_tile_width cannot exceed the probe count")
        self._basis = _readonly(basis_array)
        self._fixed = _readonly(fixed)
        self._annotations = _readonly(weights)
        self._masses = _readonly(np.array(masses, copy=True))
        self._pairs = pair_index
        self._components = component_index

    def _global_probes(self, variants: Array, probes: Array) -> Array:
        if self.native_probe_module is False:
            # Explicit test path; production passes a module or lets it import.
            from summit.ldscore.generalized_gxe_variant import (
                generate_global_variant_probes,
            )

            return generate_global_variant_probes(
                variants,
                probes,
                root_seed=self.probe_spec.root_seed,
                namespace=self.probe_spec.namespace,
            )
        return native_global_variant_probes(
            variants,
            probes,
            root_seed=self.probe_spec.root_seed,
            namespace=self.probe_spec.namespace,
            threads=int(self.nn_operator.threads),
            native_module=self.native_probe_module,
        )

    def execute(self) -> GeneralizedGxEPass1Result:
        n = self._basis.shape[0]
        q_count = self._basis.shape[1]
        m = self._annotations.shape[0]
        k_count = self._annotations.shape[1]
        b_count = self.probe_spec.probe_count
        c_count = len(self._components)
        variant_width = int(self.work_plan.tiling["variant_block_width"])
        phase_seconds = {
            "pass1_total": 0.0,
            "decode": 0.0,
            "probe_rhs": 0.0,
            "protected_nn": 0.0,
            "projection": 0.0,
            "same_person": 0.0,
        }
        started_total = time.perf_counter()
        rss_entry = _peak_rss_bytes()
        ledger = TwoPassLedger(m)
        base = np.zeros((k_count, n, b_count), dtype=np.float64, order="C")
        maximum_decoded_bytes = 0
        maximum_probe_bytes = 0
        maximum_rhs_bytes = 0
        maximum_nn_output_bytes = 0
        nn_flops = 0
        nn_dimension_counts: dict[str, int] = {}

        self.nn_operator.begin_execution()
        ledger.begin_pass(1)
        self.genotype_operator.begin_pass(1)
        for row_start in range(0, m, variant_width):
            row_stop = min(m, row_start + variant_width)
            decode_started = time.perf_counter()
            block = self.genotype_operator.read_block(row_start, row_stop)
            phase_seconds["decode"] += time.perf_counter() - decode_started
            if (
                block.row_start != row_start
                or block.row_stop != row_stop
                or block.genotype_scale_id
                != self.genotype_operator.genotype_scale_id
            ):
                ledger.record_integrity_failure()
                raise RuntimeError("decoded genotype block identity/scale mismatch")
            genotype = block.values
            maximum_decoded_bytes = max(maximum_decoded_bytes, genotype.nbytes)
            variants = np.arange(row_start, row_stop, dtype=np.int64)
            for probe_start in range(0, b_count, self.probe_tile_width):
                probe_stop = min(b_count, probe_start + self.probe_tile_width)
                global_probes = np.arange(
                    self.probe_spec.probe_offset + probe_start,
                    self.probe_spec.probe_offset + probe_stop,
                    dtype=np.int64,
                )
                rhs_started = time.perf_counter()
                probes = self._global_probes(variants, global_probes)
                maximum_probe_bytes = max(maximum_probe_bytes, probes.nbytes)
                for annotation_start in range(
                    0, k_count, self.annotation_tile_width
                ):
                    annotation_stop = min(
                        k_count, annotation_start + self.annotation_tile_width
                    )
                    for annotation in range(annotation_start, annotation_stop):
                        rhs = np.asfortranarray(
                            np.sqrt(
                                self._annotations[row_start:row_stop, annotation]
                            )[:, None]
                            * probes,
                            dtype=np.float64,
                        )
                        maximum_rhs_bytes = max(maximum_rhs_bytes, rhs.nbytes)
                        phase_seconds["probe_rhs"] += (
                            time.perf_counter() - rhs_started
                        )
                        nn_started = time.perf_counter()
                        contribution = self.nn_operator.matmul(genotype, rhs)
                        phase_seconds["protected_nn"] += (
                            time.perf_counter() - nn_started
                        )
                        maximum_nn_output_bytes = max(
                            maximum_nn_output_bytes, contribution.nbytes
                        )
                        base[
                            annotation, :, probe_start:probe_stop
                        ] += contribution
                        length = row_stop - row_start
                        columns = probe_stop - probe_start
                        nn_flops += 2 * n * length * columns
                        shape_key = f"{n}x{length}x{columns}"
                        nn_dimension_counts[shape_key] = (
                            nn_dimension_counts.get(shape_key, 0) + 1
                        )
                        rhs_started = time.perf_counter()
                        del rhs, contribution
                del probes
            ledger.record_block(row_start, row_stop)
            del genotype, block
        self.genotype_operator.finish_pass()
        ledger.finish_pass()
        ledger.validate_pass1_barrier()

        native_telemetry = self.nn_operator.finish_execution()
        for _ in range(int(self.nn_operator.repaired_columns)):
            ledger.record_repair()

        projection_started = time.perf_counter()
        contextual = np.empty(
            (k_count, q_count, n, b_count), dtype=np.float64, order="C"
        )
        maximum_projection_scratch_bytes = 0
        maximum_absolute_leakage = 0.0
        maximum_relative_leakage = 0.0
        for annotation in range(k_count):
            for coordinate in range(q_count):
                weighted = np.asfortranarray(
                    self._basis[:, coordinate, None] * base[annotation]
                )
                coefficients = self._fixed.T @ weighted
                projected = weighted - self._fixed @ coefficients
                contextual[annotation, coordinate] = projected
                leakage = self._fixed.T @ projected
                absolute = float(np.max(np.abs(leakage), initial=0.0))
                relative = float(
                    np.linalg.norm(leakage)
                    / max(1.0, float(np.linalg.norm(projected)))
                )
                maximum_absolute_leakage = max(maximum_absolute_leakage, absolute)
                maximum_relative_leakage = max(maximum_relative_leakage, relative)
                maximum_projection_scratch_bytes = max(
                    maximum_projection_scratch_bytes,
                    weighted.nbytes
                    + coefficients.nbytes
                    + projected.nbytes
                    + leakage.nbytes,
                )
                del weighted, coefficients, projected, leakage
        phase_seconds["projection"] = time.perf_counter() - projection_started
        if maximum_relative_leakage > self.projection_tolerance:
            ledger.record_integrity_failure()
            raise RuntimeError(
                "contextual source projection leakage exceeds its tolerance"
            )

        base_bytes = base.nbytes
        if self.retain_base_sources:
            base_result = _readonly(base)
        else:
            base_result = None
            del base

        same_started = time.perf_counter()
        sample_accumulator = np.zeros((c_count, n), dtype=np.float64)
        same_probe = np.zeros((c_count, c_count), dtype=np.float64)
        maximum_same_person_tile_bytes = 0
        component_entries = self._components.entries
        for probe_start in range(0, b_count, self.probe_tile_width):
            probe_stop = min(b_count, probe_start + self.probe_tile_width)
            for sample_start in range(0, n, self.same_person_sample_tile_width):
                sample_stop = min(n, sample_start + self.same_person_sample_tile_width)
                for left_position, component in enumerate(component_entries):
                    left_values = (
                        float(component.kernel_factor)
                        * contextual[
                            component.annotation_index,
                            component.q,
                            sample_start:sample_stop,
                            probe_start:probe_stop,
                        ]
                        * contextual[
                            component.annotation_index,
                            component.r,
                            sample_start:sample_stop,
                            probe_start:probe_stop,
                        ]
                        / self._masses[component.annotation_index]
                    )
                    sample_accumulator[
                        component.index, sample_start:sample_stop
                    ] += np.sum(left_values, axis=1, dtype=np.float64)
                    maximum_same_person_tile_bytes = max(
                        maximum_same_person_tile_bytes, left_values.nbytes
                    )
                    for right_component in component_entries[left_position:]:
                        if right_component.index == component.index:
                            right_values = left_values
                            scratch_bytes = left_values.nbytes
                        else:
                            right_values = (
                                float(right_component.kernel_factor)
                                * contextual[
                                    right_component.annotation_index,
                                    right_component.q,
                                    sample_start:sample_stop,
                                    probe_start:probe_stop,
                                ]
                                * contextual[
                                    right_component.annotation_index,
                                    right_component.r,
                                    sample_start:sample_stop,
                                    probe_start:probe_stop,
                                ]
                                / self._masses[
                                    right_component.annotation_index
                                ]
                            )
                            scratch_bytes = left_values.nbytes + right_values.nbytes
                        value = float(
                            np.einsum(
                                "iv,iv->",
                                left_values,
                                right_values,
                                dtype=np.float64,
                                optimize=False,
                            )
                        )
                        same_probe[component.index, right_component.index] += value
                        if right_component.index != component.index:
                            same_probe[right_component.index, component.index] += value
                        maximum_same_person_tile_bytes = max(
                            maximum_same_person_tile_bytes, scratch_bytes
                        )
                        if right_values is not left_values:
                            del right_values
                    del left_values
        same_person_raw = (
            sample_accumulator @ sample_accumulator.T - same_probe
        ) / float(b_count * (b_count - 1))
        same_person_presymmetry_error = float(
            np.max(np.abs(same_person_raw - same_person_raw.T), initial=0.0)
        )
        same_person = _readonly(
            np.ascontiguousarray(0.5 * (same_person_raw + same_person_raw.T))
        )
        phase_seconds["same_person"] = time.perf_counter() - same_started

        contextual = _readonly(contextual)
        masses = _readonly(np.array(self._masses, copy=True))
        phase_seconds["pass1_total"] = time.perf_counter() - started_total
        rss_exit = _peak_rss_bytes()
        allocation_ledger = {
            "base_sources_bytes": base_bytes,
            "contextual_sources_bytes": contextual.nbytes,
            "same_person_sample_accumulator_bytes": sample_accumulator.nbytes,
            "same_person_same_probe_bytes": same_probe.nbytes,
            "same_person_result_bytes": same_person.nbytes,
            "maximum_decoded_genotype_block_bytes": maximum_decoded_bytes,
            "maximum_probe_tile_bytes": maximum_probe_bytes,
            "maximum_rhs_tile_bytes": maximum_rhs_bytes,
            "maximum_nn_output_bytes": maximum_nn_output_bytes,
            "maximum_projection_scratch_bytes": maximum_projection_scratch_bytes,
            "maximum_same_person_tile_bytes": maximum_same_person_tile_bytes,
            "planned_peak_resident_bytes": self.work_plan.peak_resident_bytes,
            "memory_limit_bytes": self.work_plan.memory_limit_bytes,
            "process_peak_rss_bytes_at_entry": rss_entry,
            "process_peak_rss_bytes_at_exit": rss_exit,
            "base_source_scratch_released": not self.retain_base_sources,
        }
        telemetry = MappingProxyType(
            {
                "schema": "summit.generalized_gxe.pass1_telemetry.v1",
                "backend": {
                    "genotype_operator": self.genotype_operator.backend_name,
                    "nn_operator": self.nn_operator.backend_name,
                    "threads": int(self.nn_operator.threads),
                    "genotype_scale_id": self.genotype_operator.genotype_scale_id,
                    "descriptor": self.genotype_operator.descriptor_record,
                    "packed_backend_used": False,
                    "packed_backend_stage": "stage06_descriptor_native_extraction",
                },
                "phase_seconds": phase_seconds,
                "pass_ledger": ledger.to_dict(),
                "operator_counters": {
                    "observed_passes": self.genotype_operator.observed_passes,
                    "observed_variant_visits": (
                        self.genotype_operator.observed_variant_visits
                    ),
                    "blocks_read": self.genotype_operator.blocks_read,
                    "decode_seconds": self.genotype_operator.decode_seconds,
                },
                "source_nn": {
                    "calls": int(self.nn_operator.calls),
                    "repaired_columns": int(self.nn_operator.repaired_columns),
                    "leading_flops": nn_flops,
                    "dimension_counts": dict(sorted(nn_dimension_counts.items())),
                    "gflops_per_second": (
                        nn_flops
                        / max(phase_seconds["protected_nn"], np.finfo(float).tiny)
                        / 1.0e9
                    ),
                },
                "allocation_ledger": allocation_ledger,
                "native": native_telemetry,
                "barrier": {
                    "pass1_sealed": True,
                    "target_scoring_started": False,
                    "contextual_sources_readonly": not contextual.flags.writeable,
                    "same_person_cross_tile_finalized": True,
                },
            }
        )
        pair_table = tuple((entry.q, entry.r) for entry in self._pairs.entries)
        component_table = tuple(
            (entry.annotation_index, entry.pair_index)
            for entry in component_entries
        )
        return GeneralizedGxEPass1Result(
            contextual_sources=contextual,
            same_person=same_person,
            annotation_masses=masses,
            pair_table=pair_table,
            component_table=component_table,
            ledger=ledger,
            genotype_scale_id=self.genotype_operator.genotype_scale_id,
            genotype_operator_identity=id(self.genotype_operator),
            maximum_projection_leakage=maximum_absolute_leakage,
            maximum_relative_projection_leakage=maximum_relative_leakage,
            same_person_presymmetry_error=same_person_presymmetry_error,
            base_sources=base_result,
            telemetry=telemetry,
        )
