"""Second and final genotype pass for generalized per-variant GxE LD scores."""

from __future__ import annotations

from dataclasses import dataclass
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    canonical_sha256,
)
from summit.ldscore.generalized_gxe_pass1 import GeneralizedGxEPass1Result
from summit.ldscore.generalized_gxe_variant import GeneralizedGxEWorkPlan


Array = np.ndarray
RowCompleteSink = Callable[[int, int, Array], None]


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _readonly(value: Array) -> Array:
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class PairProductTerm:
    first_target: int
    first_source: int
    second_target: int
    second_source: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.first_target,
            self.first_source,
            self.second_target,
            self.second_source,
        )


@dataclass(frozen=True)
class PairProductPlan:
    pairs: tuple[tuple[int, int], ...]
    terms: tuple[tuple[tuple[PairProductTerm, ...], ...], ...]
    digest: str

    @property
    def multiplicities(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(len(entry) for entry in row) for row in self.terms
        )


def build_pair_product_plan(num_basis: int) -> PairProductPlan:
    """Build the immutable orientation-product plan without special cases."""
    pair_index = ContextPairIndex(_positive_int("num_basis", num_basis))
    pairs = tuple((entry.q, entry.r) for entry in pair_index.entries)

    def orientations(pair: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        q, r = pair
        return ((q, q),) if q == r else ((q, r), (r, q))

    rows: list[tuple[tuple[PairProductTerm, ...], ...]] = []
    serial_rows: list[list[list[list[int]]]] = []
    for target_pair in pairs:
        row: list[tuple[PairProductTerm, ...]] = []
        serial_row: list[list[list[int]]] = []
        for source_pair in pairs:
            entries = tuple(
                PairProductTerm(
                    first_target=target_right,
                    first_source=source_left,
                    second_target=target_left,
                    second_source=source_right,
                )
                for target_left, target_right in orientations(target_pair)
                for source_left, source_right in orientations(source_pair)
            )
            row.append(entries)
            serial_row.append([list(entry.to_tuple()) for entry in entries])
        rows.append(tuple(row))
        serial_rows.append(serial_row)
    record = {
        "schema": "summit.generalized_gxe.pair_product_plan.v1",
        "pairs": [list(pair) for pair in pairs],
        "terms": serial_rows,
    }
    return PairProductPlan(
        pairs=pairs,
        terms=tuple(rows),
        digest=canonical_sha256(record),
    )


class NumpyTNOperator:
    """Independent dense target backend used only for differential tests."""

    backend_name = "numpy_tn_differential_only"

    def __init__(self, *, threads: int = 1) -> None:
        self.threads = _positive_int("threads", threads)
        self.calls = 0
        self.repaired_columns = 0

    def begin_execution(self) -> None:
        self.calls = 0
        self.repaired_columns = 0

    def matmul_tn(self, left: Array, right: Array) -> Array:
        self.calls += 1
        return np.asfortranarray(left.T @ right, dtype=np.float64)

    def finish_execution(self) -> dict[str, Any]:
        return {
            "available": False,
            "reason": "NumPy differential backend has no native telemetry",
            "gemm_records": [],
            "gemm_status": {},
            "output_numa_evidence": [],
            "output_numa_status": {},
        }


class ProtectedTNOperator:
    """Thin owner of the mature protected-TN entry point and telemetry."""

    backend_name = "gxeldcore.protected_matmul_tn"

    def __init__(self, *, threads: int, native_module: Any | None = None) -> None:
        self.threads = _positive_int("threads", threads)
        if native_module is None:
            from summit import gxeldcore as native_module
        function = getattr(native_module, "protected_matmul_tn", None)
        if not callable(function):
            raise RuntimeError("native extension lacks protected_matmul_tn")
        configure = getattr(native_module, "configure_blas_threads", None)
        if callable(configure) and int(configure(self.threads)) != self.threads:
            raise RuntimeError("native extension configured an unexpected thread count")
        self._module = native_module
        self._protected = function
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

    def matmul_tn(self, left: Array, right: Array) -> Array:
        output, repaired = self._protected(left, right, self.threads)
        result = np.asarray(output)
        expected = (left.shape[1], right.shape[1])
        if result.dtype != np.dtype(np.float64) or result.shape != expected:
            raise RuntimeError("protected TN returned an unexpected dtype or shape")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("protected TN returned non-finite output")
        repaired_count = int(repaired)
        if repaired_count < 0:
            raise RuntimeError("protected TN returned a negative repair count")
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
class GeneralizedGxEPass2Result:
    directional_ldscores: Array
    directed_numerator: Array
    symmetric_numerator: Array
    genetic_gram: Array
    block_directed_numerator: Array
    block_annotation_mass: Array
    same_person: Array
    annotation_masses: Array
    pair_table: tuple[tuple[int, int], ...]
    component_table: tuple[tuple[int, int], ...]
    product_plan_digest: str
    residual_rank: int
    ledger: Any
    block_reconstruction_error: float
    presymmetry_absolute_error: float
    presymmetry_relative_error: float
    source_panel_sha256: str
    telemetry: Mapping[str, Any]
    same_person_reused_for_all_deletions: bool = True


class GeneralizedGxEPass2Executor:
    """Score every target SNP in one final physical genotype traversal."""

    def __init__(
        self,
        *,
        pass1_result: GeneralizedGxEPass1Result,
        genotype_operator: Any,
        basis: Array,
        fixed_effect_basis: Array,
        annotations: Array,
        annotation_names: Sequence[str],
        block_ids: Sequence[int] | Array,
        work_plan: GeneralizedGxEWorkPlan,
        tn_operator: Any,
        probe_tile_width: int | None = None,
        row_complete_sink: RowCompleteSink | None = None,
    ) -> None:
        if not isinstance(pass1_result, GeneralizedGxEPass1Result):
            raise TypeError("pass1_result must be a generalized pass-1 result")
        pass1_result.ledger.validate_pass1_barrier()
        if not pass1_result.pass1_barrier_sealed:
            raise RuntimeError("pass-1 result does not attest a sealed barrier")
        if pass1_result.contextual_sources.flags.writeable:
            raise RuntimeError("pass-1 contextual sources are not sealed read-only")
        if pass1_result.same_person.flags.writeable:
            raise RuntimeError("pass-1 same-person matrix is not sealed read-only")
        if id(genotype_operator) != pass1_result.genotype_operator_identity:
            raise RuntimeError("pass 2 must use the same genotype operator as pass 1")
        if genotype_operator.genotype_scale_id != pass1_result.genotype_scale_id:
            raise RuntimeError("pass-1/pass-2 genotype scale identities differ")
        if (
            genotype_operator.observed_passes != 1
            or genotype_operator.observed_variant_visits
            != genotype_operator.num_variants
        ):
            raise RuntimeError("genotype operator is not at the pass-1 barrier")

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
        if array_sha256(basis_array) != pass1_result.basis_sha256:
            raise RuntimeError("basis identity changed after pass 1")
        if array_sha256(fixed) != pass1_result.fixed_effect_basis_sha256:
            raise RuntimeError("fixed-effect identity changed after pass 1")
        if array_sha256(weights) != pass1_result.annotation_sha256:
            raise RuntimeError("annotation identity changed after pass 1")
        if (
            array_sha256(pass1_result.contextual_sources)
            != pass1_result.contextual_source_sha256
        ):
            raise RuntimeError("sealed contextual source hash does not verify")

        names = tuple(str(name) for name in annotation_names)
        if len(names) != weights.shape[1]:
            raise ValueError("annotation_names must match annotation columns")
        pair_index = ContextPairIndex(basis_array.shape[1])
        component_index = ContextComponentIndex(names, pair_index)
        pair_table = tuple((entry.q, entry.r) for entry in pair_index.entries)
        component_table = tuple(
            (entry.annotation_index, entry.pair_index)
            for entry in component_index.entries
        )
        if pair_table != pass1_result.pair_table:
            raise RuntimeError("pass-1 pair table identity changed")
        if component_table != pass1_result.component_table:
            raise RuntimeError("pass-1 component table identity changed")
        masses = np.sum(weights, axis=0, dtype=np.float64)
        if not np.array_equal(masses, pass1_result.annotation_masses):
            raise RuntimeError("pass-1 annotation masses changed")
        residual_rank = n - fixed.shape[1]
        if residual_rank < 1:
            raise ValueError("fixed effects leave no residual rank")

        blocks = np.asarray(block_ids)
        if blocks.ndim != 1 or blocks.shape[0] != m or blocks.dtype.kind not in "iu":
            raise ValueError("block_ids must be a variant-aligned integer vector")
        blocks = np.asarray(blocks, dtype=np.int64)
        if np.any(blocks < 0) or blocks[0] != 0 or np.any(np.diff(blocks) < 0):
            raise ValueError("block IDs must be contiguous, ordered, and start at zero")
        block_count = int(blocks[-1]) + 1
        if block_count < 2 or not np.array_equal(
            np.unique(blocks), np.arange(block_count, dtype=np.int64)
        ):
            raise ValueError("block IDs must contain at least two nonempty groups")

        dimensions = dict(work_plan.dimensions)
        expected_dimensions = {
            "N": n,
            "M": m,
            "Q": basis_array.shape[1],
            "P": len(pair_index),
            "K": weights.shape[1],
            "C": len(component_index),
            "B": pass1_result.contextual_sources.shape[3],
            "J": block_count,
        }
        for name, expected in expected_dimensions.items():
            if int(dimensions.get(name, -1)) != expected:
                raise ValueError(f"work plan dimension {name} does not match inputs")
        if work_plan.descriptor.get("planned_complete_passes") != 2:
            raise ValueError("work plan must admit exactly two physical passes")
        if work_plan.descriptor.get("format") != genotype_operator.genotype_format:
            raise ValueError("work plan genotype format does not match its operator")
        expected_output_bytes = (
            m * len(pair_index) * len(component_index) * np.dtype(np.float64).itemsize
        )
        if work_plan.output_size_bytes != expected_output_bytes:
            raise ValueError("Stage 05 requires an admitted FP64 directional panel")
        if (
            work_plan.peak_resident_bytes + expected_output_bytes
            > work_plan.memory_limit_bytes
        ):
            raise MemoryError("memory limit does not admit the Stage 05 resident panel")
        family_count = basis_array.shape[1] ** 2
        rhs_columns = int(work_plan.tiling["rhs_tile_columns"])
        if rhs_columns < family_count:
            raise ValueError("RHS plan cannot hold all Q-squared families for one probe")
        maximum_probe_width = min(
            pass1_result.contextual_sources.shape[3],
            rhs_columns // family_count,
        )
        if probe_tile_width is None:
            probe_tile_width = maximum_probe_width
        self.probe_tile_width = _positive_int(
            "probe_tile_width", probe_tile_width
        )
        if self.probe_tile_width > maximum_probe_width:
            raise ValueError("probe tile exceeds the admitted RHS width")
        if row_complete_sink is not None and not callable(row_complete_sink):
            raise TypeError("row_complete_sink must be callable")

        self.pass1_result = pass1_result
        self.genotype_operator = genotype_operator
        self.work_plan = work_plan
        self.tn_operator = tn_operator
        self.row_complete_sink = row_complete_sink
        self._basis = _readonly(basis_array)
        self._fixed = _readonly(fixed)
        self._annotations = _readonly(weights)
        self._masses = _readonly(np.array(masses, copy=True))
        self._blocks = _readonly(np.array(blocks, copy=True))
        self._pairs = pair_index
        self._components = component_index
        self._product_plan = build_pair_product_plan(basis_array.shape[1])
        self._residual_rank = residual_rank
        self._rhs_precomputed = bool(work_plan.tiling["rhs_precomputed"])

    def _block_masses(self) -> Array:
        result = np.zeros(
            (int(self._blocks[-1]) + 1, self._annotations.shape[1]),
            dtype=np.float64,
        )
        starts = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(np.diff(self._blocks)) + 1,
            ]
        )
        stops = np.concatenate(
            [starts[1:], np.asarray([self._blocks.size], dtype=np.int64)]
        )
        for start, stop in zip(starts, stops, strict=True):
            block = int(self._blocks[start])
            result[block] = np.sum(
                self._annotations[start:stop], axis=0, dtype=np.float64
            )
        return result

    def _precompute_rhs(self) -> Array:
        sources = self.pass1_result.contextual_sources
        k_count, q_count, n, b_count = sources.shape
        family_count = q_count * q_count
        rhs = np.empty(
            (n, k_count * b_count * family_count),
            dtype=np.float64,
            order="F",
        )
        probe_offsets = np.arange(b_count, dtype=np.int64) * family_count
        for annotation in range(k_count):
            base = annotation * b_count * family_count
            for target in range(q_count):
                for source in range(q_count):
                    columns = base + probe_offsets + target * q_count + source
                    rhs[:, columns] = (
                        self._basis[:, target, None]
                        * sources[annotation, source]
                    )
        return _readonly(rhs)

    def _fill_rhs_tile(
        self,
        arena: Array,
        annotation: int,
        probe_start: int,
        probe_stop: int,
    ) -> Array:
        q_count = self._basis.shape[1]
        probe_count = probe_stop - probe_start
        family_count = q_count * q_count
        required_columns = probe_count * family_count
        target = arena[:, :required_columns]
        probe_offsets = np.arange(probe_count, dtype=np.int64) * family_count
        sources = self.pass1_result.contextual_sources
        for target_coordinate in range(q_count):
            for source_coordinate in range(q_count):
                columns = (
                    probe_offsets
                    + target_coordinate * q_count
                    + source_coordinate
                )
                target[:, columns] = (
                    self._basis[:, target_coordinate, None]
                    * sources[
                        annotation,
                        source_coordinate,
                        :,
                        probe_start:probe_stop,
                    ]
                )
        return target

    def execute(self) -> GeneralizedGxEPass2Result:
        sources = self.pass1_result.contextual_sources
        k_count, q_count, n, b_count = sources.shape
        m = self._annotations.shape[0]
        p_count = len(self._pairs)
        c_count = len(self._components)
        block_count = int(self._blocks[-1]) + 1
        variant_width = int(self.work_plan.tiling["variant_block_width"])
        family_count = q_count * q_count
        phase_seconds = {
            "pass2_total": 0.0,
            "rhs_precompute": 0.0,
            "decode": 0.0,
            "rhs_prepare": 0.0,
            "protected_tn": 0.0,
            "row_products": 0.0,
            "numerator_reduction": 0.0,
            "output_sink": 0.0,
            "postprocess": 0.0,
        }
        total_started = time.perf_counter()
        source_hash_before = array_sha256(sources)
        if source_hash_before != self.pass1_result.contextual_source_sha256:
            raise RuntimeError("contextual source identity failed before pass 2")

        precomputed_rhs: Array | None = None
        rhs_arena: Array | None = None
        rhs_started = time.perf_counter()
        if self._rhs_precomputed:
            precomputed_rhs = self._precompute_rhs()
        else:
            rhs_arena = np.empty(
                (
                    n,
                    self.probe_tile_width * family_count,
                ),
                dtype=np.float64,
                order="F",
            )
        phase_seconds["rhs_precompute"] = time.perf_counter() - rhs_started

        directional = np.empty(
            (m, p_count, c_count), dtype=np.float64, order="C"
        )
        directed = np.zeros((c_count, c_count), dtype=np.float64)
        block_directed = np.zeros(
            (block_count, c_count, c_count), dtype=np.float64
        )
        block_masses = self._block_masses()
        block_mass_reconstruction_error = float(
            np.max(
                np.abs(
                    np.sum(block_masses, axis=0, dtype=np.float64)
                    - self._masses
                ),
                initial=0.0,
            )
        )
        block_mass_tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * float(max(1, m))
            * max(1.0, float(np.max(np.abs(self._masses), initial=0.0)))
        )
        if block_mass_reconstruction_error > block_mass_tolerance:
            raise RuntimeError("block annotation masses do not reconstruct full masses")
        if np.any(self._masses[None, :] - block_masses <= 0.0):
            raise ValueError("deleting a block empties at least one annotation")

        operator_passes_before = self.genotype_operator.observed_passes
        operator_visits_before = self.genotype_operator.observed_variant_visits
        operator_blocks_before = self.genotype_operator.blocks_read
        operator_decode_before = self.genotype_operator.decode_seconds
        maximum_decoded_bytes = 0
        maximum_rhs_bytes = 0
        maximum_cross_bytes = 0
        maximum_lsum_bytes = 0
        maximum_lrow_bytes = 0
        maximum_reduction_bytes = 0
        tn_flops = 0
        tn_dimension_counts: dict[str, int] = {}

        self.tn_operator.begin_execution()
        ledger = self.pass1_result.ledger
        ledger.begin_pass(2)
        self.genotype_operator.begin_pass(2)
        for row_start in range(0, m, variant_width):
            row_stop = min(m, row_start + variant_width)
            decode_started = time.perf_counter()
            block = self.genotype_operator.read_block(row_start, row_stop)
            phase_seconds["decode"] += time.perf_counter() - decode_started
            if (
                block.row_start != row_start
                or block.row_stop != row_stop
                or block.genotype_scale_id != self.pass1_result.genotype_scale_id
            ):
                ledger.record_integrity_failure()
                raise RuntimeError("decoded pass-2 genotype identity/scale mismatch")
            genotype = block.values
            block_width = row_stop - row_start
            maximum_decoded_bytes = max(maximum_decoded_bytes, genotype.nbytes)
            lrow = np.empty(
                (block_width, p_count, c_count), dtype=np.float64, order="C"
            )
            maximum_lrow_bytes = max(maximum_lrow_bytes, lrow.nbytes)
            for annotation in range(k_count):
                lsum = np.zeros(
                    (block_width, p_count, p_count), dtype=np.float64
                )
                maximum_lsum_bytes = max(maximum_lsum_bytes, lsum.nbytes)
                for probe_start in range(0, b_count, self.probe_tile_width):
                    probe_stop = min(
                        b_count, probe_start + self.probe_tile_width
                    )
                    rhs_prepare_started = time.perf_counter()
                    if precomputed_rhs is not None:
                        annotation_base = annotation * b_count * family_count
                        column_start = (
                            annotation_base + probe_start * family_count
                        )
                        column_stop = annotation_base + probe_stop * family_count
                        rhs = precomputed_rhs[:, column_start:column_stop]
                    else:
                        assert rhs_arena is not None
                        rhs = self._fill_rhs_tile(
                            rhs_arena, annotation, probe_start, probe_stop
                        )
                    phase_seconds["rhs_prepare"] += (
                        time.perf_counter() - rhs_prepare_started
                    )
                    maximum_rhs_bytes = max(maximum_rhs_bytes, rhs.nbytes)
                    tn_started = time.perf_counter()
                    cross_raw = self.tn_operator.matmul_tn(genotype, rhs)
                    cross = np.asfortranarray(
                        cross_raw / float(self._residual_rank),
                        dtype=np.float64,
                    )
                    phase_seconds["protected_tn"] += (
                        time.perf_counter() - tn_started
                    )
                    maximum_cross_bytes = max(
                        maximum_cross_bytes, cross_raw.nbytes + cross.nbytes
                    )
                    probe_count = probe_stop - probe_start
                    panels = cross.T.reshape(
                        probe_count,
                        q_count,
                        q_count,
                        block_width,
                    ).transpose(1, 2, 3, 0)
                    product_started = time.perf_counter()
                    for target_pair in range(p_count):
                        for source_pair in range(p_count):
                            for term in self._product_plan.terms[
                                target_pair
                            ][source_pair]:
                                lsum[:, target_pair, source_pair] += np.einsum(
                                    "vb,vb->v",
                                    panels[
                                        term.first_target,
                                        term.first_source,
                                    ],
                                    panels[
                                        term.second_target,
                                        term.second_source,
                                    ],
                                    dtype=np.float64,
                                    optimize=False,
                                )
                    phase_seconds["row_products"] += (
                        time.perf_counter() - product_started
                    )
                    columns = probe_count * family_count
                    tn_flops += 2 * block_width * n * columns
                    dimension_key = f"{block_width}x{n}x{columns}"
                    tn_dimension_counts[dimension_key] = (
                        tn_dimension_counts.get(dimension_key, 0) + 1
                    )
                    del panels, cross, cross_raw, rhs
                lrow[
                    :, :, annotation * p_count : (annotation + 1) * p_count
                ] = lsum / float(b_count)
                del lsum
            if not np.all(np.isfinite(lrow)):
                ledger.record_integrity_failure()
                raise RuntimeError("per-variant directional scores are non-finite")
            directional[row_start:row_stop] = lrow

            reduction_started = time.perf_counter()
            block_reduction = np.einsum(
                "vk,vpc->kpc",
                self._annotations[row_start:row_stop],
                lrow,
                dtype=np.float64,
                optimize=True,
            ).reshape(c_count, c_count)
            directed += block_reduction
            maximum_reduction_bytes = max(
                maximum_reduction_bytes, block_reduction.nbytes
            )
            local_blocks = self._blocks[row_start:row_stop]
            local_starts = np.concatenate(
                [
                    np.asarray([0], dtype=np.int64),
                    np.flatnonzero(np.diff(local_blocks)) + 1,
                ]
            )
            local_stops = np.concatenate(
                [
                    local_starts[1:],
                    np.asarray([block_width], dtype=np.int64),
                ]
            )
            for local_start, local_stop in zip(
                local_starts, local_stops, strict=True
            ):
                jackknife_block = int(local_blocks[local_start])
                segmented = np.einsum(
                    "vk,vpc->kpc",
                    self._annotations[
                        row_start + local_start : row_start + local_stop
                    ],
                    lrow[local_start:local_stop],
                    dtype=np.float64,
                    optimize=True,
                ).reshape(c_count, c_count)
                block_directed[jackknife_block] += segmented
                maximum_reduction_bytes = max(
                    maximum_reduction_bytes, segmented.nbytes
                )
                del segmented
            phase_seconds["numerator_reduction"] += (
                time.perf_counter() - reduction_started
            )

            if self.row_complete_sink is not None:
                sink_started = time.perf_counter()
                sink_value = _readonly(np.array(lrow, copy=True, order="C"))
                self.row_complete_sink(row_start, row_stop, sink_value)
                phase_seconds["output_sink"] += (
                    time.perf_counter() - sink_started
                )
            ledger.record_block(row_start, row_stop)
            del block_reduction, lrow, genotype, block
        self.genotype_operator.finish_pass()
        ledger.finish_pass()
        native_telemetry = self.tn_operator.finish_execution()
        for _ in range(int(self.tn_operator.repaired_columns)):
            ledger.record_repair()

        source_hash_after = array_sha256(sources)
        if source_hash_after != source_hash_before:
            ledger.record_integrity_failure()
            raise RuntimeError("sealed contextual sources changed during pass 2")
        if (
            self.genotype_operator.observed_passes - operator_passes_before != 1
            or self.genotype_operator.observed_variant_visits
            - operator_visits_before
            != m
            or self.genotype_operator.blocks_read - operator_blocks_before
            != int(self.work_plan.tiling["pass2_decoded_blocks"])
        ):
            ledger.record_integrity_failure()
            raise RuntimeError("genotype operator pass-2 counters are inconsistent")
        ledger.validate_clean_completion()

        post_started = time.perf_counter()
        reconstructed = np.sum(block_directed, axis=0, dtype=np.float64)
        block_reconstruction_error = float(
            np.max(np.abs(reconstructed - directed), initial=0.0)
        )
        reconstruction_tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * float(max(1, m))
            * max(1.0, float(np.max(np.abs(directed), initial=0.0)))
        )
        if block_reconstruction_error > reconstruction_tolerance:
            raise RuntimeError("block directed numerators do not reconstruct full sum")
        presymmetry_absolute_error = float(
            np.max(np.abs(directed - directed.T), initial=0.0)
        )
        presymmetry_relative_error = float(
            presymmetry_absolute_error
            / max(1.0, float(np.max(np.abs(directed), initial=0.0)))
        )
        symmetric = 0.5 * (directed + directed.T)
        component_annotations = np.asarray(
            [entry.annotation_index for entry in self._components.entries],
            dtype=np.int64,
        )
        component_masses = self._masses[component_annotations]
        genetic_gram = (
            float(self._residual_rank**2)
            * symmetric
            / (component_masses[:, None] * component_masses[None, :])
        )
        for name, value in (
            ("directional_ldscores", directional),
            ("directed_numerator", directed),
            ("symmetric_numerator", symmetric),
            ("genetic_gram", genetic_gram),
            ("block_directed_numerator", block_directed),
            ("block_annotation_mass", block_masses),
        ):
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"{name} is non-finite after pass 2")
        phase_seconds["postprocess"] = time.perf_counter() - post_started
        phase_seconds["pass2_total"] = time.perf_counter() - total_started

        directional = _readonly(directional)
        directed = _readonly(directed)
        symmetric = _readonly(symmetric)
        genetic_gram = _readonly(genetic_gram)
        block_directed = _readonly(block_directed)
        block_masses = _readonly(block_masses)
        allocation_ledger = {
            "directional_panel_bytes": directional.nbytes,
            "directed_numerator_bytes": directed.nbytes,
            "block_directed_numerator_bytes": block_directed.nbytes,
            "block_annotation_mass_bytes": block_masses.nbytes,
            "precomputed_rhs_bytes": (
                0 if precomputed_rhs is None else precomputed_rhs.nbytes
            ),
            "rhs_arena_bytes": 0 if rhs_arena is None else rhs_arena.nbytes,
            "maximum_decoded_genotype_block_bytes": maximum_decoded_bytes,
            "maximum_rhs_view_bytes": maximum_rhs_bytes,
            "maximum_cross_sketch_bytes": maximum_cross_bytes,
            "maximum_lsum_bytes": maximum_lsum_bytes,
            "maximum_lrow_bytes": maximum_lrow_bytes,
            "maximum_reduction_bytes": maximum_reduction_bytes,
            "planned_peak_resident_bytes": self.work_plan.peak_resident_bytes,
            "memory_limit_bytes": self.work_plan.memory_limit_bytes,
        }
        telemetry = MappingProxyType(
            {
                "schema": "summit.generalized_gxe.pass2_telemetry.v1",
                "backend": {
                    "genotype_operator": self.genotype_operator.backend_name,
                    "tn_operator": self.tn_operator.backend_name,
                    "threads": int(self.tn_operator.threads),
                    "rhs_precomputed": self._rhs_precomputed,
                    "rhs_tile_columns": int(
                        self.work_plan.tiling["rhs_tile_columns"]
                    ),
                    "probe_tile_width": self.probe_tile_width,
                },
                "phase_seconds": phase_seconds,
                "pass_ledger": ledger.to_dict(),
                "operator_pass2_counters": {
                    "observed_passes": (
                        self.genotype_operator.observed_passes
                        - operator_passes_before
                    ),
                    "observed_variant_visits": (
                        self.genotype_operator.observed_variant_visits
                        - operator_visits_before
                    ),
                    "blocks_read": (
                        self.genotype_operator.blocks_read - operator_blocks_before
                    ),
                    "decode_seconds": (
                        self.genotype_operator.decode_seconds
                        - operator_decode_before
                    ),
                },
                "target_tn": {
                    "calls": int(self.tn_operator.calls),
                    "repaired_columns": int(self.tn_operator.repaired_columns),
                    "leading_flops": tn_flops,
                    "dimension_counts": dict(sorted(tn_dimension_counts.items())),
                    "gflops_per_second": (
                        tn_flops
                        / max(
                            phase_seconds["protected_tn"],
                            np.finfo(float).tiny,
                        )
                        / 1.0e9
                    ),
                },
                "allocation_ledger": allocation_ledger,
                "native": native_telemetry,
                "checks": {
                    "block_reconstruction_error": block_reconstruction_error,
                    "block_reconstruction_tolerance": reconstruction_tolerance,
                    "block_mass_reconstruction_error": (
                        block_mass_reconstruction_error
                    ),
                    "block_mass_reconstruction_tolerance": block_mass_tolerance,
                    "source_hash_before": source_hash_before,
                    "source_hash_after": source_hash_after,
                    "same_person_reused_for_all_deletions": True,
                    "target_rows_recomputed_for_deletions": False,
                    "retained_ldscore_rows_changed": False,
                },
            }
        )
        return GeneralizedGxEPass2Result(
            directional_ldscores=directional,
            directed_numerator=directed,
            symmetric_numerator=symmetric,
            genetic_gram=genetic_gram,
            block_directed_numerator=block_directed,
            block_annotation_mass=block_masses,
            same_person=self.pass1_result.same_person,
            annotation_masses=self.pass1_result.annotation_masses,
            pair_table=self.pass1_result.pair_table,
            component_table=self.pass1_result.component_table,
            product_plan_digest=self._product_plan.digest,
            residual_rank=self._residual_rank,
            ledger=ledger,
            block_reconstruction_error=block_reconstruction_error,
            presymmetry_absolute_error=presymmetry_absolute_error,
            presymmetry_relative_error=presymmetry_relative_error,
            source_panel_sha256=source_hash_after,
            telemetry=telemetry,
        )
