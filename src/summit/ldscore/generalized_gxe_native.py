"""Descriptor-owned native generalized per-variant GxE LD-score execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from summit.context.spec import (
    ContextComponentIndex,
    ContextPairIndex,
)
from summit.ldscore.generalized_gxe_pass2 import build_pair_product_plan
from summit.ldscore.generalized_gxe_variant import (
    GeneralizedGxEWorkPlan,
    GlobalVariantProbeSpec,
)


Array = np.ndarray


def _readonly(value: Array) -> Array:
    value.setflags(write=False)
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class GeneralizedGxENativeResult:
    directional_ldscores: Array
    directed_numerator: Array
    symmetric_numerator: Array
    genetic_gram: Array
    same_person: Array
    annotation_masses: Array
    affine_mean: Array
    affine_inverse_scale: Array
    contextual_sources: Array
    base_sources: Array | None
    pair_table: tuple[tuple[int, int], ...]
    component_table: tuple[tuple[int, int], ...]
    residual_rank: int
    genotype_scale: Mapping[str, str]
    ledger: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    presymmetry_absolute_error: float
    presymmetry_relative_error: float


def generalized_gxe_performance_ledger_from_native(
    result: GeneralizedGxENativeResult,
) -> dict[str, Any]:
    """Return the closed artifact performance ledger from measured native data."""
    if not isinstance(result, GeneralizedGxENativeResult):
        raise TypeError("result must be a GeneralizedGxENativeResult")
    telemetry = dict(result.telemetry)
    phases = {"pass1", "barrier", "pass2", "finalize"}
    wall = dict(telemetry.get("phase_wall_seconds", {}))
    cpu = dict(telemetry.get("phase_cpu_seconds", {}))
    if set(wall) != phases or set(cpu) != phases:
        raise ValueError("native telemetry lacks the four measured phases")
    for name, values in (("wall", wall), ("CPU", cpu)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
            for value in values.values()
        ):
            raise ValueError(f"native {name} phase telemetry is invalid")
    dimensions = telemetry.get("gemm_dimensions")
    if not isinstance(dimensions, list):
        raise ValueError("native telemetry lacks GEMM dimensions")
    return {
        "backend": str(telemetry["backend"]),
        "threads": int(telemetry["blas_threads"]),
        "affinity": dict(telemetry.get("affinity", {})),
        "numa_evidence": dict(telemetry.get("numa_evidence", {})),
        "phase_wall_seconds": {name: float(wall[name]) for name in sorted(phases)},
        "phase_cpu_seconds": {name: float(cpu[name]) for name in sorted(phases)},
        "bytes_read": int(telemetry["logical_genotype_bytes_read"]),
        "gemm_dimensions": [dict(value) for value in dimensions],
        "peak_rss_bytes": int(telemetry["peak_rss_bytes"]),
        "output_bytes": int(telemetry["output_bytes"]),
    }


class GeneralizedGxENativeBEDExecutor:
    """Validate and run the single-owner native two-pass BED context.

    This wrapper never reads a genotype block. The native context duplicates
    the supplied descriptors and owns its decoder through both passes.
    """

    def __init__(
        self,
        *,
        stable_descriptors: Mapping[str, int],
        row_selection: Sequence[int] | Array | None,
        ddof: int,
        basis: Array,
        fixed_effect_basis: Array,
        annotations: Array,
        annotation_names: Sequence[str],
        annotation_masses: Sequence[float] | Array,
        probe_spec: GlobalVariantProbeSpec,
        work_plan: GeneralizedGxEWorkPlan,
        probe_tile_width: int | None = None,
        source_probe_tile_width: int | None = None,
        same_person_sample_tile_width: int = 4096,
        threads: int | None = None,
        decode_threads: int | None = None,
        retain_base_sources: bool = False,
        backend: str = "dense",
        qualification_fault_injection: Mapping[str, Any] | None = None,
        native_module: Any | None = None,
    ) -> None:
        if not isinstance(stable_descriptors, Mapping):
            raise TypeError("stable_descriptors must be a mapping")
        required = (".bed", ".bim", ".fam")
        descriptors: dict[str, int] = {}
        for suffix in required:
            value = stable_descriptors.get(suffix)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"stable descriptor {suffix!r} is unavailable")
            descriptors[suffix] = value
        if ddof not in {0, 1}:
            raise ValueError("ddof must be zero or one")
        if not isinstance(probe_spec, GlobalVariantProbeSpec):
            raise TypeError("probe_spec must be a GlobalVariantProbeSpec")
        if not isinstance(work_plan, GeneralizedGxEWorkPlan):
            raise TypeError("work_plan must be a GeneralizedGxEWorkPlan")
        if work_plan.descriptor.get("format") != "bed":
            raise ValueError("descriptor-native Stage 06 execution requires BED")
        if work_plan.descriptor.get("planned_complete_passes") != 2:
            raise ValueError("native work plan must require exactly two passes")

        basis_value = np.array(basis, dtype=np.float64, order="F", copy=True)
        fixed_value = np.array(
            fixed_effect_basis, dtype=np.float64, order="F", copy=True
        )
        annotation_value = np.array(
            annotations, dtype=np.float64, order="C", copy=True
        )
        masses = np.array(annotation_masses, dtype=np.float64, copy=True)
        if basis_value.ndim != 2 or fixed_value.ndim != 2:
            raise ValueError("basis and fixed_effect_basis must be matrices")
        n, q = basis_value.shape
        if fixed_value.shape[0] != n:
            raise ValueError("fixed_effect_basis has the wrong sample axis")
        if annotation_value.ndim != 2:
            raise ValueError("annotations must be a matrix")
        m, k = annotation_value.shape
        names = tuple(str(name) for name in annotation_names)
        if len(names) != k or masses.shape != (k,):
            raise ValueError("annotation names or masses are mis-sized")
        if not (
            np.all(np.isfinite(basis_value))
            and np.all(np.isfinite(fixed_value))
            and np.all(np.isfinite(annotation_value))
            and np.all(np.isfinite(masses))
        ):
            raise ValueError("native scientific inputs must be finite")
        if np.any(annotation_value < 0.0) or np.any(masses <= 0.0):
            raise ValueError("annotations must be nonnegative with positive masses")
        if not np.allclose(
            masses,
            np.sum(annotation_value, axis=0, dtype=np.float64),
            rtol=2.0e-15,
            atol=0.0,
        ):
            raise ValueError("annotation masses do not match annotations")
        pair_index = ContextPairIndex(q)
        component_index = ContextComponentIndex(names, pair_index)
        pairs = tuple((entry.q, entry.r) for entry in pair_index.entries)
        components = tuple(
            (entry.annotation_index, entry.pair_index)
            for entry in component_index.entries
        )
        pair_table = np.asarray(pairs, dtype=np.int64, order="C")
        component_table = np.asarray(components, dtype=np.int64, order="C")
        product_plan = build_pair_product_plan(q)
        offsets = [0]
        terms: list[tuple[int, int, int, int]] = []
        for row in product_plan.terms:
            for cell in row:
                terms.extend(term.to_tuple() for term in cell)
                offsets.append(len(terms))
        product_offsets = np.asarray(offsets, dtype=np.int64)
        product_terms = np.asarray(terms, dtype=np.int64, order="C")

        dimensions = dict(work_plan.dimensions)
        expected_dimensions = {
            "N": n,
            "M": m,
            "Q": q,
            "P": len(pairs),
            "K": k,
            "C": len(components),
            "B": probe_spec.probe_count,
        }
        for name, expected in expected_dimensions.items():
            if int(dimensions.get(name, -1)) != expected:
                raise ValueError(f"work plan dimension {name} does not match inputs")
        if work_plan.peak_resident_bytes > work_plan.memory_limit_bytes:
            raise MemoryError("native work plan exceeds its memory limit")

        if threads is None:
            threads = 1
        threads = _positive_int("threads", threads)
        if decode_threads is None:
            decode_threads = threads
        decode_threads = _positive_int("decode_threads", decode_threads)
        if probe_tile_width is None:
            rhs_columns = int(work_plan.tiling["rhs_tile_columns"])
            probe_tile_width = min(
                probe_spec.probe_count, rhs_columns // (q * q)
            )
        probe_tile_width = _positive_int(
            "probe_tile_width", probe_tile_width
        )
        if probe_tile_width > probe_spec.probe_count:
            raise ValueError("probe tile exceeds the global probe count")
        if source_probe_tile_width is None:
            source_probe_tile_width = probe_tile_width
        source_probe_tile_width = _positive_int(
            "source_probe_tile_width", source_probe_tile_width
        )
        if source_probe_tile_width > probe_spec.probe_count:
            raise ValueError("source probe tile exceeds the global probe count")
        sample_tile = _positive_int(
            "same_person_sample_tile_width", same_person_sample_tile_width
        )
        if not isinstance(retain_base_sources, bool):
            raise ValueError("retain_base_sources must be boolean")
        if backend not in {"dense", "packed"}:
            raise ValueError("backend must be 'dense' or 'packed'")
        fault_phase = "none"
        fault_row = -1
        fault_column = -1
        fault_delta = 0.0
        if qualification_fault_injection is not None:
            if not isinstance(qualification_fault_injection, Mapping):
                raise TypeError("qualification_fault_injection must be a mapping")
            fault = dict(qualification_fault_injection)
            if set(fault) != {"phase", "row", "column", "delta"}:
                raise ValueError("qualification fault injection has wrong fields")
            fault_phase = fault["phase"]
            fault_row = fault["row"]
            fault_column = fault["column"]
            fault_delta = fault["delta"]
            if (
                fault_phase not in {"pass1_nn", "pass2_tn"}
                or isinstance(fault_row, bool)
                or not isinstance(fault_row, int)
                or fault_row < 0
                or isinstance(fault_column, bool)
                or not isinstance(fault_column, int)
                or fault_column < 0
                or isinstance(fault_delta, bool)
                or not isinstance(fault_delta, (int, float))
                or not np.isfinite(fault_delta)
                or float(fault_delta) == 0.0
                or backend != "dense"
            ):
                raise ValueError("qualification fault injection is invalid")
        if row_selection is None:
            rows = np.arange(n, dtype=np.int64)
        else:
            rows = np.asarray(row_selection, dtype=np.int64)
            if rows.ndim != 1 or rows.size != n:
                raise ValueError("row_selection must match the basis sample axis")
            rows = np.ascontiguousarray(rows)

        if native_module is None:
            from summit import gxeldcore as native_module
        context_type = getattr(
            native_module, "GeneralizedGxELDScoreDirectContext", None
        )
        if context_type is None:
            raise RuntimeError(
                "loaded native extension lacks generalized descriptor execution"
            )
        self._context = context_type(
            bed_descriptor=descriptors[".bed"],
            bim_descriptor=descriptors[".bim"],
            fam_descriptor=descriptors[".fam"],
            row_sel=rows,
            ddof=ddof,
            basis=basis_value,
            fixed_effect_basis=fixed_value,
            annotations=annotation_value,
            annotation_masses=np.ascontiguousarray(masses),
            pair_table=pair_table,
            component_table=component_table,
            product_offsets=product_offsets,
            product_terms=product_terms,
            root_seed=probe_spec.root_seed,
            namespace_key=probe_spec.namespace_key,
            probe_offset=probe_spec.probe_offset,
            probe_count=probe_spec.probe_count,
            variant_block_width=int(work_plan.tiling["variant_block_width"]),
            source_probe_tile_width=source_probe_tile_width,
            probe_tile_width=probe_tile_width,
            same_person_sample_tile_width=sample_tile,
            max_workspace_bytes=work_plan.memory_limit_bytes,
            decode_threads=decode_threads,
            threads=threads,
            retain_base_sources=retain_base_sources,
            dense_blas_hybrid=backend == "dense",
            qualification_fault_phase=fault_phase,
            qualification_fault_row=fault_row,
            qualification_fault_column=fault_column,
            qualification_fault_delta=float(fault_delta),
        )
        self._pairs = pairs
        self._components = components
        self._dimensions = expected_dimensions
        self._retain_base_sources = retain_base_sources
        self._backend = backend

    def info(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._context.info()))

    def execute(self) -> GeneralizedGxENativeResult:
        raw = dict(self._context.run())
        shapes = {name: tuple(value) for name, value in dict(raw["shapes"]).items()}

        def reshape(name: str) -> Array:
            value = np.asarray(raw[name], dtype=np.float64)
            expected = shapes[name]
            if value.size != int(np.prod(expected, dtype=np.int64)):
                raise RuntimeError(f"native {name} has the wrong element count")
            return _readonly(np.ascontiguousarray(value.reshape(expected)))

        contextual = reshape("contextual_sources")
        directional = reshape("directional_ldscores")
        base: Array | None
        if self._retain_base_sources:
            base = reshape("base_sources")
        else:
            if raw["base_sources"] is not None:
                raise RuntimeError("native context published unrequested base sources")
            base = None
        same = _readonly(np.ascontiguousarray(raw["same_person"], dtype=np.float64))
        directed = _readonly(
            np.ascontiguousarray(raw["directed_numerator"], dtype=np.float64)
        )
        symmetric = _readonly(
            np.ascontiguousarray(raw["symmetric_numerator"], dtype=np.float64)
        )
        gram = _readonly(np.ascontiguousarray(raw["genetic_gram"], dtype=np.float64))
        masses = _readonly(
            np.ascontiguousarray(raw["annotation_masses"], dtype=np.float64)
        )
        affine_mean = _readonly(
            np.ascontiguousarray(raw["affine_mean"], dtype=np.float64)
        )
        affine_inverse_scale = _readonly(
            np.ascontiguousarray(raw["affine_inverse_scale"], dtype=np.float64)
        )
        expected_scale_shape = (self._dimensions["M"],)
        if (
            affine_mean.shape != expected_scale_shape
            or affine_inverse_scale.shape != expected_scale_shape
            or not np.all(np.isfinite(affine_mean))
            or not np.all(np.isfinite(affine_inverse_scale))
            or np.any(affine_inverse_scale <= 0.0)
        ):
            raise RuntimeError("native genotype-scale vectors are invalid")
        scale_fields = (
            "genotype_scale_policy",
            "allele_orientation",
            "allele_coding",
            "centering_source",
            "centering_formula",
            "scaling_formula",
            "missing_imputation",
            "ploidy_policy",
        )
        try:
            scale = {name: str(raw[name]) for name in scale_fields}
        except KeyError as exc:
            raise RuntimeError("native genotype-scale metadata is incomplete") from exc
        if any(not value for value in scale.values()):
            raise RuntimeError("native genotype-scale metadata is invalid")
        ledger = dict(raw["ledger"])
        expected_visits = 2 * self._dimensions["M"]
        if (
            int(ledger["observed_reference_genotype_passes"]) != 2
            or int(ledger["observed_retained_variant_visits"]) != expected_visits
            or int(ledger["duplicate_variant_visits"]) != 0
            or int(ledger["integrity_failures"]) != 0
        ):
            raise RuntimeError("native two-pass ledger is not a clean completion")
        telemetry = dict(raw["telemetry"])
        if telemetry.get("schema") != (
            "summit.generalized_gxe.native_execution_telemetry.v2"
        ):
            raise RuntimeError("native execution telemetry schema is unsupported")
        if int(telemetry.get("full_serial_output_witness_calls", -1)) != 0:
            raise RuntimeError("native execution used a forbidden serial witness")
        if int(telemetry.get("tile_induced_descriptor_rereads", -1)) != 0:
            raise RuntimeError("native execution reread descriptors for a tile")
        if self._backend == "dense" and int(
            telemetry.get("integrity_audit_count", 0)
        ) < 2:
            raise RuntimeError("native dense execution lacks phase checksum audits")
        return GeneralizedGxENativeResult(
            directional_ldscores=directional,
            directed_numerator=directed,
            symmetric_numerator=symmetric,
            genetic_gram=gram,
            same_person=same,
            annotation_masses=masses,
            affine_mean=affine_mean,
            affine_inverse_scale=affine_inverse_scale,
            contextual_sources=contextual,
            base_sources=base,
            pair_table=self._pairs,
            component_table=self._components,
            residual_rank=int(raw["residual_rank"]),
            genotype_scale=MappingProxyType(scale),
            ledger=MappingProxyType(ledger),
            telemetry=MappingProxyType(telemetry),
            presymmetry_absolute_error=float(
                telemetry["presymmetry_absolute_error"]
            ),
            presymmetry_relative_error=float(
                telemetry["presymmetry_relative_error"]
            ),
        )
