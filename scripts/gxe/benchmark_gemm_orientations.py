#!/usr/bin/env python3
"""Benchmark exact-shape protected GxE GEMMs and algebraic orientations.

Each case runs in a fresh process because the protected ``gxeldcore`` BLAS
thread count is immutable for the lifetime of a process.  Calls are strictly
serial.  The production profile covers the SUMMIT block-width candidates;
the default smoke profile uses small allocations but the same call paths.

The current protected API exposes column-major NN and TN entry points.  A
row-major source product has an exact zero-copy column-major lowering through
transposed views.  A row-major target product needs an NT entry point, so the
harness reports that case as unsupported unless a future protected
``protected_matmul_nt`` API is present.  It never disguises an O(NM) layout
copy as a row-major benchmark.

Examples
--------
Run the allocation-safe, protocol-conformant smoke sweep against an exact
development build::

    python scripts/gxe/benchmark_gemm_orientations.py \\
      --profile smoke --threads 2 \\
      --native-module /path/to/gxeldcore.cpython-312-x86_64-linux-gnu.so \\
      --json /new/path/gemm_smoke.json

Run one pruned production-shape comparison on explicitly pinned cores::

    taskset -c 0-31 python scripts/gxe/benchmark_gemm_orientations.py \\
      --profile production --block-widths 2000 --probe-tiles 32 \\
      --environment-tiles 3 --threads 32 --layouts column_major \\
      --native-module /path/to/gxeldcore.cpython-312-x86_64-linux-gnu.so \\
      --json /new/path/gemm_N289111_K2000_B32_L3_T32.json

Use ``--dry-run`` to inspect shapes and conservative memory bounds without
loading the native module or allocating operands.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


SCHEMA_NAME = "summit.gxe.protected_gemm_orientation_benchmark"
SCHEMA_VERSION = 1
PRODUCTION_N = 289_111
PRODUCTION_BLOCK_WIDTHS = (1024, 2000, 3072, 4096, 6144)
PROBE_TILE_CANDIDATES = (32, 128, 256, 512, 1024)
ENVIRONMENT_TILE_CANDIDATES = (1, 2, 3, 5)
DEFAULT_WARMUPS = 3
DEFAULT_REPEATS = 5
MAX_CONFIGURATION_SECONDS = 90.0
MAX_SWEEP_SECONDS = 20.0 * 60.0
_RESULT_PREFIX = "SUMMIT_GXE_GEMM_BENCHMARK_RESULT="


@dataclass(frozen=True)
class CaseSpec:
    operation: str
    orientation: str
    layout: str
    n_samples: int
    block_width: int
    probe_tile: int
    environment_tile: int
    threads: int

    @property
    def panel_columns(self) -> int:
        if self.operation == "source":
            return 2 * self.probe_tile * self.environment_tile
        return 4 * self.probe_tile * self.environment_tile

    @property
    def case_id(self) -> str:
        return (
            f"{self.operation}.{self.orientation}.{self.layout}"
            f".N{self.n_samples}.K{self.block_width}"
            f".B{self.probe_tile}.L{self.environment_tile}.T{self.threads}"
        )


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _csv_choices(value: str, choices: set[str]) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values).difference(choices))
    if not values or unknown:
        rendered = ", ".join(sorted(choices))
        raise argparse.ArgumentTypeError(f"expected a subset of {{{rendered}}}")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile", choices=("smoke", "production", "custom"), default="smoke",
        help="Smoke is allocation-safe; production selects exact SUMMIT N and block widths.",
    )
    parser.add_argument("--n", type=int, default=None, help="Sample count override.")
    parser.add_argument(
        "--block-widths", type=_csv_ints, default=None,
        help="Comma-separated genotype block widths.",
    )
    parser.add_argument(
        "--probe-tiles", type=_csv_ints, default=None,
        help="Comma-separated B_tile values.",
    )
    parser.add_argument(
        "--environment-tiles", type=_csv_ints, default=None,
        help="Comma-separated L_tile values.",
    )
    parser.add_argument(
        "--threads", type=_csv_ints, default=None,
        help="Comma-separated immutable BLAS thread counts; each runs in a fresh process.",
    )
    parser.add_argument(
        "--operations",
        type=lambda value: _csv_choices(value, {"source", "target"}),
        default=("source", "target"),
    )
    parser.add_argument(
        "--orientations",
        type=lambda value: _csv_choices(value, {"current", "transposed"}),
        default=("current", "transposed"),
    )
    parser.add_argument(
        "--layouts",
        type=lambda value: _csv_choices(value, {"column_major", "row_major"}),
        default=("column_major",),
    )
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--allow-short-protocol", action="store_true",
        help="Permit fewer than 3 warmups and 5 repeats for harness tests only.",
    )
    parser.add_argument(
        "--max-configuration-seconds", type=float,
        default=MAX_CONFIGURATION_SECONDS,
        help="Hard subprocess limit per case; cannot exceed 90 seconds.",
    )
    parser.add_argument(
        "--max-sweep-seconds", type=float, default=MAX_SWEEP_SECONDS,
        help="Whole-sweep wall budget; cannot exceed 1200 seconds.",
    )
    parser.add_argument(
        "--max-memory-gib", type=float, default=None,
        help="Skip cases whose conservative allocation estimate exceeds this value.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=64,
        help="Refuse an accidental Cartesian sweep larger than this count.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--native-module", type=Path, default=None,
        help=(
            "Exact gxeldcore extension file to load. Prefer this for development "
            "builds so an older installed module cannot be benchmarked accidentally."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-nonprivate", action="store_true",
        help="Development escape hatch; production evidence must use private_static BLAS.",
    )

    # Private worker arguments.  The controller always supplies all of them.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_operation", choices=("source", "target"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--_orientation", choices=("current", "transposed"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_layout", choices=("column_major", "row_major"), help=argparse.SUPPRESS)
    parser.add_argument("--_block-width", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_probe-tile", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_environment-tile", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_threads", type=int, help=argparse.SUPPRESS)
    return parser


def _resolved_axes(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile == "production":
        defaults = {
            "n_samples": PRODUCTION_N,
            "block_widths": PRODUCTION_BLOCK_WIDTHS,
            # Staged anchor: broaden these explicitly only after pruning block widths.
            "probe_tiles": (32,),
            "environment_tiles": (1,),
            "threads": (32,),
            "max_memory_gib": 96.0,
        }
    else:
        defaults = {
            "n_samples": 2048,
            "block_widths": (1024,),
            "probe_tiles": (4,),
            "environment_tiles": (1,),
            "threads": (1,),
            "max_memory_gib": 2.0,
        }
    return {
        "n_samples": args.n if args.n is not None else defaults["n_samples"],
        "block_widths": args.block_widths or defaults["block_widths"],
        "probe_tiles": args.probe_tiles or defaults["probe_tiles"],
        "environment_tiles": args.environment_tiles or defaults["environment_tiles"],
        "threads": args.threads or defaults["threads"],
        "max_memory_gib": (
            args.max_memory_gib
            if args.max_memory_gib is not None
            else defaults["max_memory_gib"]
        ),
    }


def _build_case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    axes = _resolved_axes(args)
    return [
        CaseSpec(operation, orientation, layout, axes["n_samples"], block, probe, env, threads)
        for threads in axes["threads"]
        for probe in axes["probe_tiles"]
        for env in axes["environment_tiles"]
        for block in axes["block_widths"]
        for operation in args.operations
        for orientation in args.orientations
        for layout in args.layouts
    ]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    axes = _resolved_axes(args)
    if axes["n_samples"] <= 0:
        parser.error("--n must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if (
        (args.warmups < DEFAULT_WARMUPS or args.repeats < DEFAULT_REPEATS)
        and not args.allow_short_protocol
    ):
        parser.error(
            "evidence runs require at least 3 warmups and 5 repeats; "
            "use --allow-short-protocol only for smoke tests"
        )
    if not 0.0 < args.max_configuration_seconds <= MAX_CONFIGURATION_SECONDS:
        parser.error("--max-configuration-seconds must be in (0, 90]")
    if not 0.0 < args.max_sweep_seconds <= MAX_SWEEP_SECONDS:
        parser.error("--max-sweep-seconds must be in (0, 1200]")
    if axes["max_memory_gib"] <= 0.0:
        parser.error("--max-memory-gib must be positive")
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.native_module is not None and not args.native_module.is_file():
        parser.error(f"--native-module is not a file: {args.native_module}")
    cases = _build_case_specs(args)
    if len(cases) > args.max_cases:
        parser.error(
            f"selection expands to {len(cases)} cases, above --max-cases={args.max_cases}; "
            "use a staged/pruned selection or raise the explicit guard"
        )


def _estimated_bytes(case: CaseSpec) -> dict[str, int | float]:
    n = case.n_samples
    k = case.block_width
    p = case.panel_columns
    itemsize = np.dtype(np.float64).itemsize
    if case.operation == "source":
        operands = itemsize * (n * k + k * p)
        output = itemsize * n * p
    else:
        operands = itemsize * (n * k + n * p)
        output = itemsize * k * p
    # One output, allocator/vendor allowance, and 20% explicit headroom.  No full
    # transpose allocation is included because the harness refuses such paths.
    vendor_allowance = max(256 * 1024**2, (operands + output) // 20)
    modeled = operands + output + vendor_allowance
    conservative = math.ceil(1.20 * modeled)
    return {
        "operand_bytes": int(operands),
        "output_bytes": int(output),
        "vendor_workspace_allowance_bytes": int(vendor_allowance),
        "headroom_fraction": 0.20,
        "conservative_peak_bytes": int(conservative),
        "conservative_peak_gib": conservative / 1024**3,
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        return _jsonable(dict(value))
    except (TypeError, ValueError):
        return str(value)


def _read_status_field(name: str) -> str | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _cpu_to_numa_node(cpu: int | None) -> int | None:
    if cpu is None:
        return None
    cpu_path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    try:
        nodes = sorted(cpu_path.glob("node[0-9]*"))
    except OSError:
        return None
    if not nodes:
        return None
    try:
        return int(nodes[0].name.removeprefix("node"))
    except ValueError:
        return None


def _current_cpu() -> int | None:
    getter = getattr(os, "sched_getcpu", None)
    if getter is not None:
        try:
            return int(getter())
        except OSError:
            return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.sched_getcpu.restype = ctypes.c_int
        observed = int(libc.sched_getcpu())
        return None if observed < 0 else observed
    except (AttributeError, OSError):
        return None


def _affinity_record() -> dict[str, Any]:
    try:
        cpus = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = []
    current = _current_cpu()
    nodes = sorted({
        node for node in (_cpu_to_numa_node(cpu) for cpu in cpus) if node is not None
    })
    return {
        "allowed_cpus": cpus,
        "allowed_cpu_count": len(cpus),
        "cpus_allowed_list": _read_status_field("Cpus_allowed_list"),
        "allowed_numa_nodes_from_cpus": nodes,
        "mems_allowed_list": _read_status_field("Mems_allowed_list"),
        "current_cpu": current,
        "current_numa_node": _cpu_to_numa_node(current),
    }


def _numa_mapping_for_address(address: int) -> dict[str, Any] | None:
    """Return the Linux VMA/NUMA page record containing an array pointer."""
    try:
        mapping_start = None
        mapping_end = None
        mapping_name = None
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=5)
            lower_text, upper_text = fields[0].split("-", 1)
            lower, upper = int(lower_text, 16), int(upper_text, 16)
            if lower <= address < upper:
                mapping_start, mapping_end = lower, upper
                mapping_name = fields[5] if len(fields) == 6 else None
                break
        if mapping_start is None or mapping_end is None:
            return None
        prefix = f"{mapping_start:x} "
        numa_line = next(
            (
                line
                for line in Path("/proc/self/numa_maps").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.startswith(prefix)
            ),
            None,
        )
        record: dict[str, Any] = {
            "vma_start": mapping_start,
            "vma_end": mapping_end,
            "vma_bytes": mapping_end - mapping_start,
            "mapping_name": mapping_name,
        }
        if numa_line is not None:
            fields = numa_line.split()
            record["policy"] = fields[1] if len(fields) > 1 else None
            node_pages: dict[str, int] = {}
            attributes: dict[str, str | bool] = {}
            for field in fields[2:]:
                if "=" in field:
                    key, raw = field.split("=", 1)
                    if key.startswith("N") and key[1:].isdigit():
                        try:
                            node_pages[key] = int(raw)
                        except ValueError:
                            attributes[key] = raw
                    else:
                        attributes[key] = raw
                else:
                    attributes[field] = True
            record["node_pages"] = node_pages
            record["attributes"] = attributes
        return record
    except (OSError, StopIteration, ValueError):
        return None


def _array_record(name: str, array: np.ndarray) -> dict[str, Any]:
    strides = [int(value // array.dtype.itemsize) for value in array.strides]
    return {
        "name": name,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "order": (
            "F" if array.flags.f_contiguous and not array.flags.c_contiguous
            else "C" if array.flags.c_contiguous and not array.flags.f_contiguous
            else "both"
        ),
        "strides_elements": strides,
        "leading_dimension": strides[1] if array.flags.c_contiguous else strides[1],
        "bytes": int(array.nbytes),
        "numa_mapping": _numa_mapping_for_address(int(array.ctypes.data)),
    }


def _fill_random(array: np.ndarray, generator: np.random.Generator, scale: float) -> None:
    flat = array.ravel(order="K")
    generator.standard_normal(size=flat.shape, out=flat)
    flat *= scale


@dataclass
class _PreparedCall:
    invoke: Callable[[], tuple[np.ndarray, int]]
    expected_entry: Callable[[int, int], float]
    logical_output_shape: tuple[int, int]
    returned_is_transposed: bool
    operands: list[tuple[str, np.ndarray]]
    requested_cblas: dict[str, Any]
    actual_protected_call: dict[str, Any]
    explanation: str


def _protected_metadata(
    operation: str, left: np.ndarray, right: np.ndarray
) -> dict[str, Any]:
    if operation == "NN":
        m, k, n = left.shape[0], left.shape[1], right.shape[1]
        trans_a = "N"
    elif operation == "TN":
        k, m, n = left.shape[0], left.shape[1], right.shape[1]
        trans_a = "T"
    elif operation == "NT":
        m, k, n = left.shape[0], left.shape[1], right.shape[0]
        trans_a = "N"
    else:
        raise ValueError(f"unknown protected operation: {operation}")
    return {
        "api": f"protected_matmul_{operation.lower()}",
        "cblas_interface": "column_major",
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "transpose_a": trans_a,
        "transpose_b": "T" if operation == "NT" else "N",
        "lda": int(left.shape[0]),
        "ldb": int(right.shape[0]),
        "ldc": int(m),
        "output_layout": "column_major",
    }


def _requested_cblas(case: CaseSpec) -> dict[str, Any]:
    n, k, p = case.n_samples, case.block_width, case.panel_columns
    reduction = k if case.operation == "source" else n
    if case.operation == "source":
        if case.orientation == "current":
            m_out, n_out = n, p
            ta, tb = "N", "N"
            lda_col, ldb_col, ldc_col = n, k, n
            lda_row, ldb_row, ldc_row = k, p, p
        else:
            m_out, n_out = p, n
            ta, tb = "T", "N"
            lda_col, ldb_col, ldc_col = k, k, p
            lda_row, ldb_row, ldc_row = k, n, n
    else:
        if case.orientation == "current":
            m_out, n_out = k, p
            ta, tb = "T", "N"
            lda_col, ldb_col, ldc_col = n, n, k
            lda_row, ldb_row, ldc_row = k, p, p
        else:
            m_out, n_out = p, k
            ta, tb = "T", "N"
            lda_col, ldb_col, ldc_col = n, n, p
            lda_row, ldb_row, ldc_row = p, k, k
    row = case.layout == "row_major"
    return {
        "interface": case.layout,
        "m": int(m_out),
        "n": int(n_out),
        "k": int(reduction),
        "transpose_a": ta,
        "transpose_b": tb,
        "lda": int(lda_row if row else lda_col),
        "ldb": int(ldb_row if row else ldb_col),
        "ldc": int(ldc_row if row else ldc_col),
        "output_layout": case.layout,
    }


def _prepare_call(case: CaseSpec, module: Any, seed: int) -> _PreparedCall | str:
    generator = np.random.Generator(np.random.PCG64(seed))
    n, k, p = case.n_samples, case.block_width, case.panel_columns
    requested = _requested_cblas(case)

    if case.operation == "source" and case.layout == "column_major":
        if case.orientation == "current":
            genotype = np.empty((n, k), dtype=np.float64, order="F")
            weights = np.empty((k, p), dtype=np.float64, order="F")
            _fill_random(genotype, generator, 1.0 / math.sqrt(k))
            _fill_random(weights, generator, 1.0)
            left, right, native_operation = genotype, weights, "NN"
            expected = lambda i, j: float(np.sum(genotype[i, :] * weights[:, j]))
            logical_shape = (n, p)
            returned_is_transposed = False
            explanation = "C = G @ W via column-major NN."
        else:
            weights = np.empty((k, p), dtype=np.float64, order="F")
            genotype_t = np.empty((k, n), dtype=np.float64, order="F")
            _fill_random(weights, generator, 1.0)
            _fill_random(genotype_t, generator, 1.0 / math.sqrt(k))
            left, right, native_operation = weights, genotype_t, "TN"
            expected = lambda i, j: float(np.sum(weights[:, i] * genotype_t[:, j]))
            logical_shape = (p, n)
            returned_is_transposed = False
            explanation = (
                "C' = W' @ G' via column-major TN; G' is generated directly "
                "in the phase-appropriate layout."
            )
    elif case.operation == "source" and case.layout == "row_major":
        genotype = np.empty((n, k), dtype=np.float64, order="C")
        weights = np.empty((k, p), dtype=np.float64, order="C")
        _fill_random(genotype, generator, 1.0 / math.sqrt(k))
        _fill_random(weights, generator, 1.0)
        # A row-major NN product is exactly its reversed column-major NN
        # transpose.  Both views are F-contiguous and allocate no copy.
        left, right, native_operation = weights.T, genotype.T, "NN"
        if case.orientation == "current":
            expected = lambda i, j: float(np.sum(genotype[i, :] * weights[:, j]))
            logical_shape = (n, p)
            returned_is_transposed = True
        else:
            expected = lambda i, j: float(np.sum(weights[:, i] * genotype[j, :]))
            logical_shape = (p, n)
            returned_is_transposed = False
        explanation = (
            "Row-major G @ W lowered exactly to column-major W' @ G' through "
            "zero-copy F-contiguous views; the protected API has no row-major enum."
        )
    elif case.operation == "target" and case.layout == "column_major":
        genotype = np.empty((n, k), dtype=np.float64, order="F")
        sources = np.empty((n, p), dtype=np.float64, order="F")
        _fill_random(genotype, generator, 1.0 / math.sqrt(n))
        _fill_random(sources, generator, 1.0)
        if case.orientation == "current":
            left, right = genotype, sources
            expected = lambda i, j: float(np.sum(genotype[:, i] * sources[:, j]))
            logical_shape = (k, p)
            explanation = "C = G' @ S via column-major TN."
        else:
            left, right = sources, genotype
            expected = lambda i, j: float(np.sum(sources[:, i] * genotype[:, j]))
            logical_shape = (p, k)
            explanation = "C' = S' @ G via column-major TN."
        native_operation = "TN"
        returned_is_transposed = False
    else:
        nt = getattr(module, "protected_matmul_nt", None)
        if nt is None:
            return (
                "row-major target requires a protected NT/row-major entry point; "
                "the current API cannot lower it without an explicit O(NM) copy"
            )
        genotype = np.empty((n, k), dtype=np.float64, order="C")
        sources = np.empty((n, p), dtype=np.float64, order="C")
        _fill_random(genotype, generator, 1.0 / math.sqrt(n))
        _fill_random(sources, generator, 1.0)
        left, right, native_operation = sources.T, genotype.T, "NT"
        if case.orientation == "current":
            expected = lambda i, j: float(np.sum(genotype[:, i] * sources[:, j]))
            logical_shape = (k, p)
            returned_is_transposed = True
        else:
            expected = lambda i, j: float(np.sum(sources[:, i] * genotype[:, j]))
            logical_shape = (p, k)
            returned_is_transposed = False
        explanation = (
            "Row-major G' @ S lowered exactly to protected column-major "
            "S' @ G through zero-copy views."
        )

    function = getattr(module, f"protected_matmul_{native_operation.lower()}")

    def invoke() -> tuple[np.ndarray, int]:
        result, repaired = function(left, right, case.threads)
        return np.asarray(result), int(repaired)

    return _PreparedCall(
        invoke=invoke,
        expected_entry=expected,
        logical_output_shape=logical_shape,
        returned_is_transposed=returned_is_transposed,
        operands=[("left", left), ("right", right)],
        requested_cblas=requested,
        actual_protected_call=_protected_metadata(native_operation, left, right),
        explanation=explanation,
    )


def _telemetry_reset(module: Any) -> bool:
    reset = getattr(module, "reset_gemm_telemetry", None)
    if reset is None:
        return False
    reset()
    return True


def _telemetry_consume(module: Any) -> tuple[bool, list[dict[str, Any]]]:
    for name in ("consume_gemm_telemetry", "drain_gemm_telemetry"):
        function = getattr(module, name, None)
        if function is not None:
            value = _jsonable(function())
            if isinstance(value, dict) and "records" in value:
                return True, list(value["records"])
            return True, list(value or [])
    getter = getattr(module, "get_gemm_telemetry", None)
    if getter is not None:
        value = _jsonable(getter())
        if isinstance(value, dict) and "records" in value:
            return True, list(value["records"])
        return True, list(value or [])
    return False, []


def _sample_indices(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, columns = shape
    candidates = [(0, 0), (rows // 2, columns // 2), (rows - 1, columns - 1)]
    return list(dict.fromkeys(candidates))


def _validate_samples(prepared: _PreparedCall, output: np.ndarray) -> dict[str, Any]:
    expected_returned_shape = (
        tuple(reversed(prepared.logical_output_shape))
        if prepared.returned_is_transposed
        else prepared.logical_output_shape
    )
    if tuple(output.shape) != expected_returned_shape:
        raise RuntimeError(
            f"protected result shape {output.shape} does not match {expected_returned_shape}"
        )
    errors: list[float] = []
    values: list[float] = []
    for logical_i, logical_j in _sample_indices(prepared.logical_output_shape):
        observed = (
            float(output[logical_j, logical_i])
            if prepared.returned_is_transposed
            else float(output[logical_i, logical_j])
        )
        expected = prepared.expected_entry(logical_i, logical_j)
        error = abs(observed - expected)
        tolerance = 1.0e-10 + 1.0e-10 * abs(expected)
        if not math.isfinite(observed) or error > tolerance:
            raise RuntimeError(
                f"sampled dense-oracle mismatch at ({logical_i}, {logical_j}): "
                f"observed={observed:.17g}, expected={expected:.17g}, "
                f"absolute_error={error:.3g}, tolerance={tolerance:.3g}"
            )
        errors.append(error)
        values.append(observed)
    return {
        "sample_count": len(errors),
        "maximum_absolute_error": max(errors, default=0.0),
        "sample_checksum": float(math.fsum(values)),
        "rtol": 1.0e-10,
        "atol": 1.0e-10,
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": float(min(values)),
        "median": float(statistics.median(values)),
        "maximum": float(max(values)),
        "mean": float(statistics.fmean(values)),
    }


def _native_provenance(module: Any) -> dict[str, Any]:
    binary_path = Path(module.__file__).resolve()
    build_info = _jsonable(module.build_info())
    return {
        "module_path": str(binary_path),
        "native_binary_sha256": _sha256(binary_path),
        "build_info": build_info,
        "exact_build_hash": _sha256(binary_path),
    }


def _load_gxeldcore(native_module: Path | None) -> Any:
    if native_module is None:
        from summit import gxeldcore

        return gxeldcore
    resolved = native_module.resolve()
    spec = importlib.util.spec_from_file_location("gxeldcore", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load gxeldcore extension: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_worker_case(
    case: CaseSpec,
    *,
    warmups: int,
    repeats: int,
    seed: int,
    allow_nonprivate: bool,
    native_module: Path | None,
) -> dict[str, Any]:
    # Import only in the worker.  The controller must never freeze a thread count.
    gxeldcore = _load_gxeldcore(native_module)

    provenance = _native_provenance(gxeldcore)
    build = provenance["build_info"]
    if not allow_nonprivate and build.get("blas_runtime_isolation") != "private_static":
        raise RuntimeError(
            "benchmark evidence requires blas_runtime_isolation=private_static; "
            "use --allow-nonprivate only for development smoke tests"
        )
    configured = int(gxeldcore.configure_blas_threads(case.threads))
    if configured != case.threads:
        raise RuntimeError(
            f"protected runtime configured {configured} threads, requested {case.threads}"
        )
    # Refresh after configuration so the immutable observed count is recorded.
    provenance = _native_provenance(gxeldcore)
    prepared = _prepare_call(case, gxeldcore, seed)
    common = {
        "case_id": case.case_id,
        "case": asdict(case),
        "panel_columns": case.panel_columns,
        "arithmetic_dtype": "float64",
        "storage_dtype": "float64",
        "flops_per_logical_call": int(
            2 * case.n_samples * case.block_width * case.panel_columns
        ),
        "warmups": warmups,
        "measured_repeats": repeats,
        "memory_model": _estimated_bytes(case),
        "native": provenance,
        "affinity_and_numa": _affinity_record(),
    }
    if isinstance(prepared, str):
        return {
            **common,
            "status": "unsupported",
            "reason": prepared,
            "requested_cblas": _requested_cblas(case),
            "explicit_full_matrix_transpose": False,
        }

    warmup_checks: list[dict[str, Any]] = []
    for _ in range(warmups):
        output, repaired = prepared.invoke()
        if repaired:
            raise RuntimeError(f"warmup repaired {repaired} output columns")
        warmup_checks.append(_validate_samples(prepared, output))
        del output

    observations: list[dict[str, Any]] = []
    for repeat in range(repeats):
        telemetry_reset = _telemetry_reset(gxeldcore)
        affinity_before = _affinity_record()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        output, repaired = prepared.invoke()
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        affinity_after = _affinity_record()
        telemetry_available, native_records = _telemetry_consume(gxeldcore)
        correctness = _validate_samples(prepared, output)
        output_mapping = _numa_mapping_for_address(int(output.ctypes.data))
        if repaired:
            raise RuntimeError(f"measured call repaired {repaired} output columns")
        flops = common["flops_per_logical_call"]
        observations.append(
            {
                "repeat": repeat,
                "wall_seconds": wall_seconds,
                "process_cpu_seconds": cpu_seconds,
                "average_active_cores": cpu_seconds / wall_seconds,
                "matrix_seconds": wall_seconds,
                "matrix_minutes": wall_seconds / 60.0,
                "flops": flops,
                "gflops_per_second": flops / wall_seconds / 1.0e9,
                "requested_threads": case.threads,
                "observed_blas_threads": provenance["build_info"].get(
                    "blas_runtime_threads"
                ),
                "affinity_before": affinity_before,
                "affinity_after": affinity_after,
                "output_numa_mapping": output_mapping,
                "correctness": correctness,
                "repaired_output_columns": repaired,
                "native_telemetry": {
                    "available": telemetry_available,
                    "reset_available": telemetry_reset,
                    "records": native_records,
                    "fallback_when_unavailable": "process CPU divided by wrapper wall time",
                },
            }
        )
        del output

    wall_values = [item["wall_seconds"] for item in observations]
    cpu_values = [item["process_cpu_seconds"] for item in observations]
    core_values = [item["average_active_cores"] for item in observations]
    rate_values = [item["gflops_per_second"] for item in observations]
    return {
        **common,
        "status": "ok",
        "requested_cblas": prepared.requested_cblas,
        "actual_protected_call": prepared.actual_protected_call,
        "orientation_explanation": prepared.explanation,
        "explicit_full_matrix_transpose": False,
        "operands": [_array_record(name, value) for name, value in prepared.operands],
        "warmup_correctness": warmup_checks,
        "observations": observations,
        "summary": {
            "wall_seconds": _summary(wall_values),
            "process_cpu_seconds": _summary(cpu_values),
            "average_active_cores": _summary(core_values),
            "gflops_per_second": _summary(rate_values),
            "cumulative_measured_wall_seconds": float(math.fsum(wall_values)),
            "cumulative_process_cpu_seconds": float(math.fsum(cpu_values)),
            "matrix_minutes": float(math.fsum(wall_values) / 60.0),
        },
    }


def _worker_command(args: argparse.Namespace, case: CaseSpec) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_operation", case.operation,
        "--_orientation", case.orientation,
        "--_layout", case.layout,
        "--n", str(case.n_samples),
        "--_block-width", str(case.block_width),
        "--_probe-tile", str(case.probe_tile),
        "--_environment-tile", str(case.environment_tile),
        "--_threads", str(case.threads),
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
        "--allow-short-protocol",
    ]
    if args.allow_nonprivate:
        command.append("--allow-nonprivate")
    if args.native_module is not None:
        command.extend(["--native-module", str(args.native_module.resolve())])
    return command


def _worker_environment(case: CaseSpec) -> dict[str, str]:
    environment = dict(os.environ)
    threads = str(case.threads)
    environment["OMP_NUM_THREADS"] = threads
    environment["OPENBLAS_NUM_THREADS"] = threads
    environment["OMP_DYNAMIC"] = "FALSE"
    return environment


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(
            f"worker emitted {len(lines)} benchmark result records; expected exactly one"
        )
    return json.loads(lines[0][len(_RESULT_PREFIX):])


def _host_provenance() -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "argv": sys.argv,
        "controller_pid": os.getpid(),
        "affinity_and_numa": _affinity_record(),
    }


def _run_controller(args: argparse.Namespace) -> dict[str, Any]:
    cases = _build_case_specs(args)
    axes = _resolved_axes(args)
    report: dict[str, Any] = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "provenance": _host_provenance(),
        "protocol": {
            "warmups": args.warmups,
            "measured_repeats": args.repeats,
            "required_warmups": DEFAULT_WARMUPS,
            "required_measured_repeats": DEFAULT_REPEATS,
            "conformant": (
                args.warmups >= DEFAULT_WARMUPS and args.repeats >= DEFAULT_REPEATS
            ),
            "max_configuration_seconds": args.max_configuration_seconds,
            "max_sweep_seconds": args.max_sweep_seconds,
            "fresh_process_per_case": True,
            "concurrent_vendor_calls": False,
            "explicit_full_matrix_transposes": False,
        },
        "selection": {
            "profile": args.profile,
            **_jsonable(axes),
            "operations": list(args.operations),
            "orientations": list(args.orientations),
            "layouts": list(args.layouts),
            "case_count": len(cases),
            "requested_native_module": (
                None if args.native_module is None else str(args.native_module.resolve())
            ),
            "production_block_width_candidates": list(PRODUCTION_BLOCK_WIDTHS),
            "production_probe_tile_candidates": list(PROBE_TILE_CANDIDATES),
            "production_environment_tile_candidates": list(ENVIRONMENT_TILE_CANDIDATES),
        },
        "cases": [],
    }
    if args.dry_run:
        report["cases"] = [
            {
                "case_id": case.case_id,
                "case": asdict(case),
                "panel_columns": case.panel_columns,
                "memory_model": _estimated_bytes(case),
                "status": "planned",
            }
            for case in cases
        ]
        report["sweep_wall_seconds"] = 0.0
        return report

    sweep_started = time.monotonic()
    memory_limit = int(axes["max_memory_gib"] * 1024**3)
    native_identities: set[tuple[str, str]] = set()
    for index, case in enumerate(cases):
        elapsed = time.monotonic() - sweep_started
        remaining = args.max_sweep_seconds - elapsed
        if remaining <= 0.0:
            for unrun in cases[index:]:
                report["cases"].append(
                    {
                        "case_id": unrun.case_id,
                        "case": asdict(unrun),
                        "status": "skipped_sweep_budget",
                    }
                )
            break
        memory = _estimated_bytes(case)
        if memory["conservative_peak_bytes"] > memory_limit:
            report["cases"].append(
                {
                    "case_id": case.case_id,
                    "case": asdict(case),
                    "panel_columns": case.panel_columns,
                    "memory_model": memory,
                    "memory_limit_bytes": memory_limit,
                    "status": "skipped_memory_budget",
                }
            )
            continue
        timeout_seconds = min(args.max_configuration_seconds, remaining)
        command = _worker_command(args, case)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=_worker_environment(case),
                timeout=timeout_seconds,
            )
            duration = time.monotonic() - started
            if completed.returncode != 0:
                result = {
                    "case_id": case.case_id,
                    "case": asdict(case),
                    "status": "error",
                    "returncode": completed.returncode,
                    "controller_wall_seconds": duration,
                    "stderr": completed.stderr[-8000:],
                    "stdout_without_result": completed.stdout[-2000:],
                }
            else:
                result = _parse_worker_stdout(completed.stdout)
                result["controller_wall_seconds"] = duration
                result["worker_stderr"] = completed.stderr[-8000:]
                native = result.get("native")
                if native:
                    identity = (
                        str(native.get("module_path")),
                        str(native.get("native_binary_sha256")),
                    )
                    native_identities.add(identity)
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            result = {
                "case_id": case.case_id,
                "case": asdict(case),
                "status": "timeout",
                "timeout_seconds": timeout_seconds,
                "controller_wall_seconds": duration,
                "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            }
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            result = {
                "case_id": case.case_id,
                "case": asdict(case),
                "status": "controller_error",
                "controller_wall_seconds": time.monotonic() - started,
                "reason": str(exc),
            }
        report["cases"].append(result)

    report["sweep_wall_seconds"] = time.monotonic() - sweep_started
    report["native_binary_identity_count"] = len(native_identities)
    report["native_binary_identities"] = [
        {"module_path": path, "native_binary_sha256": digest}
        for path, digest in sorted(native_identities)
    ]
    report["status_counts"] = {
        status: sum(1 for case in report["cases"] if case.get("status") == status)
        for status in sorted({str(case.get("status")) for case in report["cases"]})
    }
    return report


def _write_report(report: dict[str, Any], target: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if target is None:
        sys.stdout.write(rendered)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args._worker:
        missing = [
            name
            for name in (
                "_operation", "_orientation", "_layout", "_block_width",
                "_probe_tile", "_environment_tile", "_threads", "n",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"internal worker missing arguments: {', '.join(missing)}")
        case = CaseSpec(
            args._operation,
            args._orientation,
            args._layout,
            args.n,
            args._block_width,
            args._probe_tile,
            args._environment_tile,
            args._threads,
        )
        try:
            result = _run_worker_case(
                case,
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed,
                allow_nonprivate=args.allow_nonprivate,
                native_module=args.native_module,
            )
        except Exception as exc:  # Worker reports a bounded error to its controller.
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(_RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return

    _validate_args(parser, args)
    _write_report(_run_controller(args), args.json)


if __name__ == "__main__":
    main()
