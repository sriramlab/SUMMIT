#!/usr/bin/env python3
"""Diagnose the rejected API-8 source-TT/current-target FP64 wrappers.

The controller never imports the native module.  It launches one fresh
``python -S`` process per source/target case, confines that process to a
selected physical-core cpuset, fixes the private OpenBLAS thread configuration
for the life of the worker, and selects a caller-supplied installed
``gxeldcore`` extension by exact path and SHA256.

The diagnostic uses the source TT wrapper and the established column-major
protected target-pair wrapper.  Source TT has shown intermittent repairs and an
undetected dense-oracle mismatch, so this tool can never publish accepted
production evidence.  The exact profile uses production
dimensions N=289111, K=2000, B_tile=32, and L_tile=3.  Use ``--dry-run`` before
an evidence run.  The smoke profile uses the same wrappers with bounded
allocations.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any, Callable, Sequence


SCHEMA = "summit.gxe.fp64_optimized_production_wrapper_benchmark"
SCHEMA_VERSION = 1
RESULT_PREFIX = "SUMMIT_GXE_FP64_OPTIMIZED_RESULT="
DEFAULT_ARCHIVE_SHA256 = (
    "49609db51d9c91bb4beaa40af20ef6a57698698b8043b3d9ac15ffc25554f521"
)
MIN_WARMUPS = 3
MIN_REPEATS = 5
MAX_CONFIGURATION_SECONDS = 90.0
MAX_SWEEP_SECONDS = 20.0 * 60.0
GIB = 1024**3
MIB = 1024**2
VENDOR_WORKSPACE_FALLBACK_BYTES = 16 * GIB
INTEGRITY_CHECK_COUNT = 8
THREAD_STACK_BYTES = 8 * MIB
ALLOCATOR_SLACK_FLOOR_BYTES = GIB // 2
ALLOCATOR_SLACK_FRACTION = 0.10
TOTAL_HEADROOM_FRACTION = 0.20
MIN_VENDOR_ACTIVE_CORE_FRACTION = 0.75
FULL_PRECISION_LAYOUT = "source_tt_target_current"
PRODUCTION_ACCEPTANCE_ELIGIBLE = False
PRODUCTION_REJECTION_REASON = (
    "source TT failed the zero-repair/dense-oracle reproducibility gate; "
    "native API-8 source/row-target entry points are diagnostic only"
)


@dataclasses.dataclass(frozen=True)
class Case:
    operation: str
    n_samples: int
    block_width: int
    probe_tile: int
    environment_tile: int
    threads: int

    @property
    def source_columns(self) -> int:
        return 2 * self.probe_tile * self.environment_tile

    @property
    def target_columns(self) -> int:
        return 2 * self.source_columns

    @property
    def case_id(self) -> str:
        return (
            f"{self.operation}.{FULL_PRECISION_LAYOUT}"
            f".N{self.n_samples}.K{self.block_width}"
            f".B{self.probe_tile}.L{self.environment_tile}.T{self.threads}"
        )

    @property
    def flops(self) -> int:
        columns = (
            self.source_columns if self.operation == "source" else self.target_columns
        )
        return 2 * self.n_samples * self.block_width * columns

    def expected_telemetry(self) -> dict[str, Any]:
        if self.operation == "source":
            return {
                "operation": "dgemm_tt",
                "layout": "column_major",
                "transpose_a": "T",
                "transpose_b": "T",
                "m": self.source_columns,
                "n": self.n_samples,
                "k": self.block_width,
                "lda": self.block_width,
                "ldb": self.n_samples,
                "ldc": self.source_columns,
            }
        return {
            "operation": "dgemm_tn",
            "layout": "column_major",
            "transpose_a": "T",
            "transpose_b": "N",
            "m": self.block_width,
            "n": self.target_columns,
            "k": self.n_samples,
            "lda": self.n_samples,
            "ldb": self.n_samples,
            "ldc": self.block_width,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("smoke", "exact", "custom"), default="smoke"
    )
    parser.add_argument(
        "--operations",
        type=lambda value: _csv_choices(value, {"source", "target"}),
        default=("source", "target"),
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--block-width", type=int, default=None)
    parser.add_argument("--probe-tile", type=int, default=None)
    parser.add_argument("--environment-tile", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--cpus", type=_expand_cpu_list, default=None)
    parser.add_argument("--warmups", type=int, default=MIN_WARMUPS)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument(
        "--allow-short-protocol",
        action="store_true",
        help="Permit fewer repeats only for bounded harness smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--max-configuration-seconds",
        type=float,
        default=MAX_CONFIGURATION_SECONDS,
    )
    parser.add_argument("--max-sweep-seconds", type=float, default=MAX_SWEEP_SECONDS)
    parser.add_argument("--max-memory-gib", type=float, default=None)
    parser.add_argument("--install-prefix", type=Path, default=None)
    parser.add_argument("--native-module", type=Path, default=None)
    parser.add_argument("--expected-native-sha256", default=None)
    parser.add_argument("--expected-archive-sha256", default=DEFAULT_ARCHIVE_SHA256)
    parser.add_argument(
        "--allow-integrity-disabled",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch; accepted evidence requires an "
            "integrity-enabled API-8 build."
        ),
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_operation", choices=("source", "target"), help=argparse.SUPPRESS
    )
    return parser


def _csv_choices(value: str, allowed: set[str]) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(values).difference(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated subset of " + ", ".join(sorted(allowed))
        )
    return values


def _expand_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    try:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first_text, last_text = part.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first < 0 or last < first:
                    raise ValueError
                result.extend(range(first, last + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError
                result.append(cpu)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid CPU list") from error
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("CPU list must be nonempty and unique")
    return result


def _compress_cpu_list(cpus: Sequence[int]) -> str:
    ordered = sorted(cpus)
    if not ordered:
        return ""
    ranges: list[str] = []
    first = last = ordered[0]
    for cpu in ordered[1:]:
        if cpu == last + 1:
            last = cpu
            continue
        ranges.append(str(first) if first == last else f"{first}-{last}")
        first = last = cpu
    ranges.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(ranges)


def _profile(args: argparse.Namespace) -> dict[str, int | float]:
    if args.profile == "exact":
        defaults: dict[str, int | float] = {
            "n_samples": 289_111,
            "block_width": 2000,
            "probe_tile": 32,
            "environment_tile": 3,
            "threads": 32,
            "max_memory_gib": 96.0,
        }
    else:
        defaults = {
            "n_samples": 2048,
            "block_width": 256,
            "probe_tile": 4,
            "environment_tile": 1,
            "threads": 1,
            # The tiny integrity path does not enter vendor BLAS, but retain
            # enough admission budget for the strict uncertified-workspace
            # fallback so the default smoke profile does not silently skip.
            "max_memory_gib": 32.0,
        }
    return {
        "n_samples": args.n if args.n is not None else defaults["n_samples"],
        "block_width": (
            args.block_width
            if args.block_width is not None
            else defaults["block_width"]
        ),
        "probe_tile": (
            args.probe_tile if args.probe_tile is not None else defaults["probe_tile"]
        ),
        "environment_tile": (
            args.environment_tile
            if args.environment_tile is not None
            else defaults["environment_tile"]
        ),
        "threads": args.threads if args.threads is not None else defaults["threads"],
        "max_memory_gib": (
            args.max_memory_gib
            if args.max_memory_gib is not None
            else defaults["max_memory_gib"]
        ),
    }


def _cases(args: argparse.Namespace) -> list[Case]:
    values = _profile(args)
    return [
        Case(
            operation=operation,
            n_samples=int(values["n_samples"]),
            block_width=int(values["block_width"]),
            probe_tile=int(values["probe_tile"]),
            environment_tile=int(values["environment_tile"]),
            threads=int(values["threads"]),
        )
        for operation in args.operations
    ]


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    values = _profile(args)
    if any(
        int(values[name]) <= 0
        for name in (
            "n_samples",
            "block_width",
            "probe_tile",
            "environment_tile",
            "threads",
        )
    ):
        parser.error("shape values and --threads must be positive")
    if float(values["max_memory_gib"]) <= 0.0:
        parser.error("--max-memory-gib must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be nonnegative and --repeats positive")
    if (
        args.warmups < MIN_WARMUPS or args.repeats < MIN_REPEATS
    ) and not args.allow_short_protocol:
        parser.error(
            "evidence runs require at least 3 warmups and 5 timed repeats; "
            "--allow-short-protocol is only for bounded smoke tests"
        )
    if not 0.0 < args.max_configuration_seconds <= MAX_CONFIGURATION_SECONDS:
        parser.error("--max-configuration-seconds must be in (0, 90]")
    if not 0.0 < args.max_sweep_seconds <= MAX_SWEEP_SECONDS:
        parser.error("--max-sweep-seconds must be in (0, 1200]")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.output is not None and not args.output.parent.is_dir():
        parser.error(
            "output parent must already exist for atomic publication: "
            f"{args.output.parent}"
        )
    if args.dry_run:
        return
    required = {
        "--install-prefix": args.install_prefix,
        "--native-module": args.native_module,
        "--expected-native-sha256": args.expected_native_sha256,
    }
    if not args._worker:
        required["--output"] = args.output
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("actual runs require " + ", ".join(missing))
    assert args.expected_native_sha256 is not None
    if not _canonical_sha256(args.expected_native_sha256):
        parser.error("--expected-native-sha256 must be lowercase canonical SHA256")
    if not _canonical_sha256(args.expected_archive_sha256):
        parser.error("--expected-archive-sha256 must be lowercase canonical SHA256")
    assert args.install_prefix is not None
    assert args.native_module is not None
    prefix = args.install_prefix.resolve()
    module = args.native_module.resolve()
    if not prefix.is_dir():
        parser.error(f"--install-prefix is not a directory: {prefix}")
    if not module.is_file():
        parser.error(f"--native-module is not a file: {module}")
    try:
        relative_module = module.relative_to(prefix)
    except ValueError:
        parser.error("--native-module must resolve inside --install-prefix")
    if (
        relative_module.parent.name != "summit"
        or not module.name.startswith("gxeldcore.")
        or module.suffix != ".so"
    ):
        parser.error("--native-module must be an installed summit/gxeldcore.*.so")
    observed = _sha256(module)
    if observed != args.expected_native_sha256:
        parser.error(
            "native module SHA256 mismatch: "
            f"expected {args.expected_native_sha256}, observed {observed}"
        )
    if not args.python_executable.resolve().is_file():
        parser.error(f"--python-executable is not a file: {args.python_executable}")


def _estimated_memory(case: Case) -> dict[str, Any]:
    itemsize = 8
    n, k, p = case.n_samples, case.block_width, case.source_columns
    if case.operation == "source":
        named = {
            "genotype_f_bytes": itemsize * n * k,
            "weights_f_bytes": itemsize * k * p,
            "output_c_bytes": itemsize * n * p,
        }
    else:
        named = {
            "genotype_f_bytes": itemsize * n * k,
            "source_panel_f_bytes": itemsize * n * p,
            "environment_f_bytes": itemsize * n * case.environment_tile,
            "sealed_pair_bytes": itemsize * n * 2 * p,
            "output_f_bytes": itemsize * k * 2 * p,
        }
    if case.operation == "source":
        integrity_weight_snapshot = itemsize * k * p
        integrity_coefficients = itemsize * INTEGRITY_CHECK_COUNT * p
        integrity_projection = itemsize * INTEGRITY_CHECK_COUNT * k
        integrity_expected_observed = itemsize * 2 * INTEGRITY_CHECK_COUNT * n
    else:
        integrity_weight_snapshot = 0
        integrity_coefficients = itemsize * INTEGRITY_CHECK_COUNT * k
        integrity_projection = itemsize * INTEGRITY_CHECK_COUNT * n
        integrity_expected_observed = (
            itemsize * 2 * INTEGRITY_CHECK_COUNT * case.target_columns
        )
    integrity_check_scratch = (
        integrity_coefficients + integrity_projection + integrity_expected_observed
    )
    candidate_owned = (
        sum(named.values()) + integrity_weight_snapshot + integrity_check_scratch
    )
    additional_thread_stacks = max(0, case.threads - 1) * THREAD_STACK_BYTES
    allocator_slack = max(
        ALLOCATOR_SLACK_FLOOR_BYTES,
        (candidate_owned + 9) // 10,
    )
    modeled = (
        candidate_owned
        + additional_thread_stacks
        + allocator_slack
        + VENDOR_WORKSPACE_FALLBACK_BYTES
    )
    conservative = (modeled * 120 + 99) // 100
    return {
        **named,
        "integrity_weight_snapshot_bytes": integrity_weight_snapshot,
        "integrity_coefficient_bytes": integrity_coefficients,
        "integrity_projection_bytes": integrity_projection,
        "integrity_expected_observed_bytes": integrity_expected_observed,
        "integrity_check_scratch_bytes": integrity_check_scratch,
        "integrity_mode_assumption": (
            "included_unconditionally_for_integrity_enabled_or_unknown_build"
        ),
        "candidate_owned_peak_bytes": candidate_owned,
        "additional_thread_stack_count": max(0, case.threads - 1),
        "thread_stack_bytes_each": THREAD_STACK_BYTES,
        "additional_thread_stacks_bytes": additional_thread_stacks,
        "allocator_slack_floor_bytes": ALLOCATOR_SLACK_FLOOR_BYTES,
        "allocator_slack_fraction_of_candidate_owned": ALLOCATOR_SLACK_FRACTION,
        "allocator_slack_bytes": allocator_slack,
        "vendor_workspace_allowance_bytes": VENDOR_WORKSPACE_FALLBACK_BYTES,
        "vendor_workspace_allowance_gib": (VENDOR_WORKSPACE_FALLBACK_BYTES / GIB),
        "vendor_workspace_bound_source": (
            "strict_16gib_uncertified_private_openblas_fallback"
        ),
        "vendor_workspace_bound_uncertainty": (
            "the exact archive does not export or certify its workspace; "
            "the source-derived T32 pool estimate of 4.125 GiB is informative "
            "only and is not used as the bound"
        ),
        "modeled_subtotal_before_headroom_bytes": modeled,
        "headroom_fraction": TOTAL_HEADROOM_FRACTION,
        "headroom_bytes": conservative - modeled,
        "conservative_peak_bytes": conservative,
        "conservative_peak_gib": conservative / GIB,
        "bound_formula": (
            "ceil(1.20 * (candidate_owned_peak + (threads-1)*8MiB + "
            "max(0.5GiB, 0.10*candidate_owned_peak) + "
            "16GiB_uncertified_vendor_workspace))"
        ),
    }


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _physical_core_key(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    package = _read_int(topology / "physical_package_id")
    core = _read_int(topology / "core_id")
    if package is None or core is None:
        return (0, cpu)
    return package, core


def _cpu_numa_node(cpu: int) -> int | None:
    try:
        nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    except OSError:
        return None
    if not nodes:
        return None
    try:
        return int(nodes[0].name.removeprefix("node"))
    except ValueError:
        return None


def _default_physical_cpus() -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    result: list[int] = []
    observed: set[tuple[int, int]] = set()
    for cpu in available:
        key = _physical_core_key(cpu)
        if key in observed:
            continue
        observed.add(key)
        result.append(cpu)
    return result


def _validate_cpu_pool(cpus: Sequence[int], threads: int) -> None:
    outside = sorted(set(cpus).difference(os.sched_getaffinity(0)))
    if outside:
        raise RuntimeError(f"requested CPUs are outside controller affinity: {outside}")
    if len(cpus) < threads:
        raise RuntimeError(f"need {threads} physical CPUs, received {len(cpus)}")
    selected = list(cpus[:threads])
    keys = [_physical_core_key(cpu) for cpu in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("selected CPU list contains SMT siblings")


def _prefix_identity(prefix: Path) -> dict[str, Any]:
    package = prefix / "summit"
    if not package.is_dir():
        raise RuntimeError(f"installed prefix lacks summit package: {prefix}")
    records: list[tuple[str, int, str]] = []
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        records.append(
            (str(path.relative_to(prefix)), path.stat().st_size, _sha256(path))
        )
    if not records:
        raise RuntimeError(f"installed summit package is empty: {package}")
    digest = hashlib.sha256()
    for relative, size, file_hash in records:
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
    return {
        "install_prefix": str(prefix),
        "installed_package_file_count": len(records),
        "installed_package_bytes": sum(size for _, size, _ in records),
        "installed_package_manifest_sha256": digest.hexdigest(),
    }


def _run_checked(command: Sequence[str], timeout: float) -> str:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    ).stdout


def _linkage_evidence(module: Path) -> dict[str, Any]:
    readelf = shutil.which("readelf")
    nm = shutil.which("nm")
    if readelf is None or nm is None:
        raise RuntimeError("readelf and nm are required for linkage verification")
    dynamic = _run_checked([readelf, "-d", str(module)], 20.0)
    symbols = _run_checked([nm, "-D", "--defined-only", str(module)], 20.0)
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    dynamic_blas = [name for name in needed if "blas" in name.lower()]
    exported_blas = [
        line
        for line in symbols.splitlines()
        if re.search(r"\b(?:cblas_|openblas_)", line)
    ]
    if dynamic_blas:
        raise RuntimeError(f"dynamic BLAS dependency defeats isolation: {dynamic_blas}")
    if exported_blas:
        raise RuntimeError("private OpenBLAS symbols escaped from the extension")
    return {
        "needed_shared_libraries": needed,
        "dynamic_blas_dependencies": dynamic_blas,
        "exported_blas_symbol_count": len(exported_blas),
        "readelf_dynamic_sha256": hashlib.sha256(dynamic.encode()).hexdigest(),
        "defined_dynamic_symbols_sha256": hashlib.sha256(symbols.encode()).hexdigest(),
        "private_static_linkage_verified": True,
    }


def _worker_environment(case: Case, cpus: Sequence[int]) -> dict[str, str]:
    if len(cpus) != case.threads:
        raise ValueError("worker CPU list must match the requested thread count")
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "OMP_PLACES",
        "OMP_NESTED",
        "GOMP_CPU_AFFINITY",
        "KMP_AFFINITY",
        "KMP_HW_SUBSET",
        "KMP_PLACE_THREADS",
    ):
        environment.pop(name, None)
    threads = str(case.threads)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "OMP_THREAD_LIMIT": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "GOTO_NUM_THREADS": threads,
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_WAIT_POLICY": "PASSIVE",
            "GOMP_SPINCOUNT": "0",
        }
    )
    return environment


def _worker_command(args: argparse.Namespace, case: Case) -> list[str]:
    assert args.install_prefix is not None
    assert args.native_module is not None
    assert args.expected_native_sha256 is not None
    command = [
        str(args.python_executable.resolve()),
        "-S",
        str(Path(__file__).resolve()),
        "--_worker",
        "--_operation",
        case.operation,
        "--profile",
        "custom",
        "--n",
        str(case.n_samples),
        "--block-width",
        str(case.block_width),
        "--probe-tile",
        str(case.probe_tile),
        "--environment-tile",
        str(case.environment_tile),
        "--threads",
        str(case.threads),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--allow-short-protocol",
        "--install-prefix",
        str(args.install_prefix.resolve()),
        "--native-module",
        str(args.native_module.resolve()),
        "--expected-native-sha256",
        args.expected_native_sha256,
        "--expected-archive-sha256",
        args.expected_archive_sha256,
    ]
    if args.allow_integrity_disabled:
        command.append("--allow-integrity-disabled")
    return command


def _bootstrap_worker_paths(prefix: Path) -> list[str]:
    inserted: list[str] = []
    # ``-S`` deliberately suppresses site initialization and .pth processing.
    # Add only the selected install prefix and the exact interpreter's NumPy
    # dependency directory, derived from sysconfig without importing site.
    candidates = [
        str(prefix.resolve()),
        sysconfig.get_path("platlib"),
        sysconfig.get_path("purelib"),
    ]
    for candidate in reversed([value for value in candidates if value]):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
            inserted.append(candidate)
    return list(reversed(inserted))


def _load_exact_native(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("gxeldcore", module_path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != module_path.resolve():
        raise RuntimeError("loaded native module path differs from requested path")
    return module


def _status_fields() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key in {
                "VmRSS",
                "VmHWM",
                "RssAnon",
                "RssFile",
                "Cpus_allowed_list",
                "Mems_allowed_list",
            }:
                result[key] = value.strip()
    except OSError:
        pass
    return result


def _resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
        "maximum_rss_kib": int(usage.ru_maxrss),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "proc_status": _status_fields(),
    }


def _resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "maximum_rss_kib": after["maximum_rss_kib"],
        "minor_fault_delta": after["minor_faults"] - before["minor_faults"],
        "major_fault_delta": after["major_faults"] - before["major_faults"],
        "voluntary_context_switch_delta": (
            after["voluntary_context_switches"] - before["voluntary_context_switches"]
        ),
        "involuntary_context_switch_delta": (
            after["involuntary_context_switches"]
            - before["involuntary_context_switches"]
        ),
        "before_proc_status": before["proc_status"],
        "after_proc_status": after["proc_status"],
    }


def _memory_psi() -> dict[str, str] | None:
    try:
        lines = Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return {
        fields[0]: " ".join(fields[1:]) for line in lines if (fields := line.split())
    }


def _affinity() -> dict[str, Any]:
    cpus = sorted(os.sched_getaffinity(0))
    nodes = sorted(
        {node for node in (_cpu_numa_node(cpu) for cpu in cpus) if node is not None}
    )
    return {
        "scope": "calling_thread_sched_getaffinity",
        "allowed_cpus": cpus,
        "allowed_cpu_count": len(cpus),
        "allowed_cpu_list": _compress_cpu_list(cpus),
        "physical_core_keys": [list(_physical_core_key(cpu)) for cpu in cpus],
        "assigned_numa_nodes_from_cpus": nodes,
        "status": _status_fields(),
    }


def _affinity_contract(record: dict[str, Any]) -> dict[str, Any]:
    status = dict(record.get("status", {}))
    return {
        name: record.get(name)
        for name in (
            "scope",
            "allowed_cpus",
            "allowed_cpu_count",
            "allowed_cpu_list",
            "physical_core_keys",
            "assigned_numa_nodes_from_cpus",
        )
    } | {
        "status": {
            name: status.get(name)
            for name in ("Cpus_allowed_list", "Mems_allowed_list")
        }
    }


def _array_record(name: str, array: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "order": (
            "F"
            if array.flags.f_contiguous and not array.flags.c_contiguous
            else "C"
            if array.flags.c_contiguous and not array.flags.f_contiguous
            else "both"
        ),
        "strides_elements": [
            int(value // array.dtype.itemsize) for value in array.strides
        ],
        "bytes": int(array.nbytes),
    }


def _contiguous_array_sha256(
    array: Any, chunk_bytes: int = 8 * MIB
) -> tuple[str, float]:
    if chunk_bytes <= 0:
        raise ValueError("hash chunk size must be positive")
    if array.flags.c_contiguous:
        storage_view = array
    elif array.flags.f_contiguous:
        storage_view = array.T
        if not storage_view.flags.c_contiguous:
            raise RuntimeError("F-contiguous output did not expose a C-order view")
    else:
        raise RuntimeError("full-output hashing requires a contiguous array")
    view = memoryview(storage_view).cast("B")
    digest = hashlib.sha256()
    started = time.perf_counter()
    for offset in range(0, len(view), chunk_bytes):
        digest.update(view[offset : offset + chunk_bytes])
    wall_seconds = time.perf_counter() - started
    view.release()
    return digest.hexdigest(), wall_seconds


@dataclasses.dataclass
class Prepared:
    invoke: Callable[[], tuple[Any, int]]
    expected_entry: Callable[[int, int], float]
    output_shape: tuple[int, int]
    output_order: str
    operands: list[tuple[str, Any]]
    pair_construction: dict[str, Any] | None
    pair_metadata: dict[str, Any] | None


def _fill_random(array: Any, generator: Any, scale: float) -> None:
    flat = array.ravel(order="K")
    generator.standard_normal(size=flat.shape, out=flat)
    flat *= scale


def _timed_pair_construction(function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    before = _resource_snapshot()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    pair = function()
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    after = _resource_snapshot()
    return pair, {
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "average_active_cores": cpu_seconds / wall_seconds,
        "resource_usage": _resource_delta(before, after),
    }


def _prepare(case: Case, module: Any, seed: int) -> Prepared:
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(seed))
    n, k, p = case.n_samples, case.block_width, case.source_columns
    if case.operation == "source":
        genotype = np.empty((n, k), dtype=np.float64, order="F")
        weights = np.empty((k, p), dtype=np.float64, order="F")
        _fill_random(genotype, generator, 1.0 / math.sqrt(k))
        _fill_random(weights, generator, 1.0)

        def invoke() -> tuple[Any, int]:
            result, repaired = module.protected_matmul_tt_row_major_output(
                weights, genotype, case.threads
            )
            return np.asarray(result), int(repaired)

        def expected(i: int, j: int) -> float:
            return float(np.dot(genotype[i, :], weights[:, j]))

        return Prepared(
            invoke=invoke,
            expected_entry=expected,
            output_shape=(n, p),
            output_order="C",
            operands=[("genotype_f", genotype), ("weights_f", weights)],
            pair_construction=None,
            pair_metadata=None,
        )

    genotype = np.empty((n, k), dtype=np.float64, order="F")
    source_panel = np.empty((n, p), dtype=np.float64, order="F")
    environments = np.empty((n, case.environment_tile), dtype=np.float64, order="F")
    _fill_random(genotype, generator, 1.0 / math.sqrt(n))
    _fill_random(source_panel, generator, 1.0)
    _fill_random(environments, generator, 1.0)
    module.reset_gemm_telemetry()
    pair, pair_timing = _timed_pair_construction(
        lambda: module.prepare_protected_row_weighted_pair(
            source_panel, environments, case.threads
        )
    )
    pair_records = list(module.consume_gemm_telemetry())
    if pair_records:
        raise RuntimeError("target pair construction unexpectedly entered vendor GEMM")

    def invoke() -> tuple[Any, int]:
        result, repaired = module.protected_matmul_tn_pair(
            genotype, pair, case.threads
        )
        return np.asarray(result), int(repaired)

    def expected(i: int, j: int) -> float:
        source_column = j % p
        values = genotype[:, i] * source_panel[:, source_column]
        if j >= p:
            group = source_column // (2 * case.probe_tile)
            values = values * environments[:, group]
        return float(np.sum(values, dtype=np.float64))

    return Prepared(
        invoke=invoke,
        expected_entry=expected,
        output_shape=(k, 2 * p),
        output_order="F",
        operands=[
            ("genotype_f", genotype),
            ("source_panel_f", source_panel),
            ("environments_f", environments),
        ],
        pair_construction=pair_timing,
        pair_metadata={
            "rows": int(pair.rows),
            "unweighted_columns": int(pair.columns),
            "sealed_total_columns": int(2 * pair.columns),
            "input_mode": "mprotect_read_only",
            "vendor_gemm_records_during_construction": 0,
        },
    )


def _sample_indices(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, columns = shape
    candidates = [
        (0, 0),
        (rows // 3, columns // 3),
        (rows // 2, columns // 2),
        ((2 * rows) // 3, (2 * columns) // 3),
        (rows - 1, columns - 1),
    ]
    return list(dict.fromkeys(candidates))


def _validate_samples(prepared: Prepared, output: Any) -> dict[str, Any]:
    import numpy as np

    if tuple(output.shape) != prepared.output_shape:
        raise RuntimeError(
            f"output shape {output.shape} differs from {prepared.output_shape}"
        )
    expected_contiguous = (
        output.flags.c_contiguous
        if prepared.output_order == "C"
        else output.flags.f_contiguous
    )
    if output.dtype != np.dtype(np.float64) or not expected_contiguous:
        raise RuntimeError(
            "hybrid FP64 wrapper output has the wrong dtype or memory order"
        )
    observed_values: list[float] = []
    expected_values: list[float] = []
    absolute_errors: list[float] = []
    normalized_errors: list[float] = []
    for row, column in _sample_indices(prepared.output_shape):
        observed = float(output[row, column])
        expected = prepared.expected_entry(row, column)
        error = abs(observed - expected)
        tolerance = 1.0e-10 + 1.0e-10 * abs(expected)
        if not math.isfinite(observed) or error > tolerance:
            raise RuntimeError(
                f"sample oracle mismatch at ({row}, {column}): "
                f"observed={observed:.17g}, expected={expected:.17g}, "
                f"error={error:.3g}, tolerance={tolerance:.3g}"
            )
        observed_values.append(observed)
        expected_values.append(expected)
        absolute_errors.append(error)
        normalized_errors.append(error / tolerance)
    return {
        "indices": [list(item) for item in _sample_indices(prepared.output_shape)],
        "observed_values": observed_values,
        "expected_values": expected_values,
        "sample_count": len(observed_values),
        "maximum_absolute_error": max(absolute_errors, default=0.0),
        "maximum_normalized_error": max(normalized_errors, default=0.0),
        "sample_checksum": float(math.fsum(observed_values)),
        "rtol": 1.0e-10,
        "atol": 1.0e-10,
    }


def _validate_vendor_record(
    case: Case, record: dict[str, Any], affinity: dict[str, Any]
) -> dict[str, Any]:
    expected = case.expected_telemetry()
    mismatches = {
        name: (expected_value, record.get(name))
        for name, expected_value in expected.items()
        if record.get(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"native telemetry shape/layout mismatch: {mismatches}")
    required = {
        "schema_version": 1,
        "arithmetic_dtype": "float64",
        "requested_threads": case.threads,
        "configured_threads": case.threads,
        "backend_threads": case.threads,
        "omp_in_parallel": False,
        "omp_level": 0,
        "omp_active_level": 0,
        "completed": True,
        "cpu_affinity_count": affinity["allowed_cpu_count"],
        "cpu_affinity_list": affinity["allowed_cpu_list"],
    }
    bad = {
        name: (expected_value, record.get(name))
        for name, expected_value in required.items()
        if record.get(name) != expected_value
    }
    if bad:
        raise RuntimeError(f"native telemetry execution contract mismatch: {bad}")
    if float(record.get("flop_count", -1.0)) != float(case.flops):
        raise RuntimeError("native telemetry FLOP count mismatch")
    for name in (
        "wall_seconds",
        "process_cpu_seconds",
        "gflops_per_second",
        "active_core_equivalents",
    ):
        if (
            not math.isfinite(float(record.get(name, 0.0)))
            or float(record.get(name, 0.0)) <= 0.0
        ):
            raise RuntimeError(f"native telemetry {name} must be positive")
    numa = record.get("operand_numa_page_samples")
    if not isinstance(numa, dict) or set(numa.get("operands", {})) != {"a", "b", "c"}:
        raise RuntimeError("native telemetry lacks A/B/C NUMA page samples")
    if numa.get("sampling_method") != "move_pages_query_no_migration":
        raise RuntimeError("native telemetry has an unexpected NUMA sampling method")
    assigned_nodes = {int(node) for node in affinity["assigned_numa_nodes_from_cpus"]}
    statuses: dict[str, str] = {}
    locality_nodes: set[int] = set()
    explicit_unavailable = {"permission_denied", "unsupported"}
    for operand_name, raw_operand in dict(numa["operands"]).items():
        operand = dict(raw_operand)
        status = str(operand.get("query_status"))
        statuses[str(operand_name)] = status
        if status in explicit_unavailable:
            continue
        if status != "queried":
            raise RuntimeError(
                "NUMA page query returned an incomplete/error status for "
                f"operand {operand_name}: {status}"
            )
        selected = int(operand.get("selected_sample_pages", -1))
        resolved = int(operand.get("resolved_sample_pages", -1))
        page_errors = int(operand.get("page_query_error_pages", -1))
        if selected <= 0 or resolved != selected or page_errors != 0:
            raise RuntimeError(
                "queried NUMA samples must resolve completely with zero page errors"
            )
        histogram = {
            int(node): int(count)
            for node, count in dict(operand.get("node_histogram", {})).items()
        }
        if any(node < 0 or count <= 0 for node, count in histogram.items()):
            raise RuntimeError("queried NUMA node histogram is invalid")
        if sum(histogram.values()) != resolved:
            raise RuntimeError(
                "queried NUMA node histogram does not match resolved pages"
            )
        if not assigned_nodes:
            raise RuntimeError(
                "cannot validate queried pages without CPU-to-NUMA topology"
            )
        remote_nodes = sorted(set(histogram).difference(assigned_nodes))
        if remote_nodes:
            raise RuntimeError(
                "queried GEMM operand pages are outside the selected CPU NUMA nodes: "
                f"{remote_nodes}"
            )
        locality_nodes.update(histogram)
    queried = [name for name, status in statuses.items() if status == "queried"]
    unavailable = [
        name for name, status in statuses.items() if status in explicit_unavailable
    ]
    return {
        "operand_statuses": statuses,
        "assigned_numa_nodes_from_selected_cpus": sorted(assigned_nodes),
        "observed_numa_nodes": sorted(locality_nodes),
        "queried_operand_count": len(queried),
        "explicitly_unavailable_operand_count": len(unavailable),
        "all_operands_queried_and_local": len(queried) == 3,
        "acceptance_eligible_on_exact_host": len(queried) == 3,
        "unavailable_status_is_explicit": len(queried) + len(unavailable) == 3,
    }


def _validate_vendor_core_utilization(
    case: Case, record: dict[str, Any], *, phase: str, repeat: int
) -> dict[str, Any]:
    if phase not in {"warmup", "measured"}:
        raise ValueError(f"unexpected benchmark phase: {phase}")
    observed = float(record["active_core_equivalents"])
    minimum = case.threads * MIN_VENDOR_ACTIVE_CORE_FRACTION
    applies = phase == "measured"
    if applies and observed < minimum:
        raise RuntimeError(
            "measured native telemetry shows insufficient vendor core use: "
            f"observed {observed:.3f}, required at least {minimum:.3f} "
            f"at measured repeat {repeat}"
        )
    return {
        "applies": applies,
        "scope": "each_measured_call_excludes_warmups",
        "minimum_active_core_fraction": MIN_VENDOR_ACTIVE_CORE_FRACTION,
        "minimum_active_cores": minimum,
        "observed_active_cores": observed,
        "passed": observed >= minimum if applies else None,
    }


def _vendor_telemetry_required(case: Case, build_info: dict[str, Any]) -> bool:
    if not bool(build_info.get("gemm_integrity_enabled")):
        return True
    threshold = int(build_info.get("gemm_integrity_minimum_vendor_flops", 0))
    if threshold <= 0:
        raise RuntimeError("integrity build lacks a positive vendor-GEMM threshold")
    return case.flops >= threshold


def _invoke_observed(
    case: Case,
    prepared: Prepared,
    module: Any,
    repeat: int,
    phase: str,
    baseline_samples: list[float] | None,
    baseline_output_sha256: str | None,
    require_vendor_telemetry: bool,
) -> tuple[dict[str, Any], list[float], str]:
    module.reset_gemm_telemetry()
    before = _resource_snapshot()
    affinity_before = _affinity()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    output, repaired = prepared.invoke()
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    affinity_after = _affinity()
    after = _resource_snapshot()
    records = [dict(value) for value in module.consume_gemm_telemetry()]
    status = dict(module.gemm_telemetry_status())
    if repaired != 0:
        raise RuntimeError(f"protected call repaired {repaired} output columns")
    expected_record_count = 1 if require_vendor_telemetry else 0
    if len(records) != expected_record_count or status.get("dropped_records") != 0:
        raise RuntimeError(
            "wrapper call captured an unexpected number of vendor records: "
            f"expected {expected_record_count}, observed {len(records)}"
        )
    if _affinity_contract(affinity_before) != _affinity_contract(affinity_after):
        raise RuntimeError("worker affinity changed during protected call")
    numa_validation = None
    core_utilization_gate = None
    if require_vendor_telemetry:
        numa_validation = _validate_vendor_record(case, records[0], affinity_before)
        core_utilization_gate = _validate_vendor_core_utilization(
            case, records[0], phase=phase, repeat=repeat
        )
    correctness = _validate_samples(prepared, output)
    samples = list(correctness["observed_values"])
    if baseline_samples is not None and samples != baseline_samples:
        raise RuntimeError("sampled protected output is not bitwise deterministic")
    output_sha256, output_hash_wall_seconds = _contiguous_array_sha256(output)
    if baseline_output_sha256 is not None and output_sha256 != baseline_output_sha256:
        raise RuntimeError("full protected output is not bitwise deterministic")
    del output
    return (
        {
            "phase": phase,
            "repeat": repeat,
            "wrapper_wall_seconds": wall_seconds,
            "wrapper_process_cpu_seconds": cpu_seconds,
            "wrapper_average_active_cores": cpu_seconds / wall_seconds,
            "flops": case.flops,
            "wrapper_gflops_per_second": case.flops / wall_seconds / 1.0e9,
            "requested_threads": case.threads,
            "repaired_output_columns": repaired,
            "sampled_correctness": correctness,
            "sampled_bitwise_deterministic_against_first_call": (
                True if baseline_samples is not None else None
            ),
            "full_output_sha256": output_sha256,
            "full_output_hash_method": "sha256_native_contiguous_memoryview_8mib_chunks",
            "full_output_hash_chunk_bytes": 8 * MIB,
            "full_output_hash_wall_seconds": output_hash_wall_seconds,
            "full_output_bitwise_deterministic_against_first_call": (
                True if baseline_output_sha256 is not None else None
            ),
            "affinity": affinity_before,
            "resource_usage": _resource_delta(before, after),
            "native_vendor_telemetry_required": require_vendor_telemetry,
            "native_vendor_record": records[0] if records else None,
            "numa_locality_validation": numa_validation,
            "vendor_core_utilization_gate": core_utilization_gate,
            "telemetry_status_after_consume": status,
        },
        samples,
        output_sha256,
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": float(min(values)),
        "median": float(statistics.median(values)),
        "maximum": float(max(values)),
        "mean": float(statistics.fmean(values)),
    }


def _native_provenance(
    module: Any,
    module_path: Path,
    expected_archive_sha256: str,
    allow_integrity_disabled: bool,
) -> dict[str, Any]:
    build = {str(key): value for key, value in dict(module.build_info()).items()}
    required = {
        "api_version": 8,
        "backend_version": "1.5",
        "blas_runtime_isolation": "private_static",
        "gemm_vendor_entry_outer_openmp_guard": True,
        "optimized_fp64_layout": "source_column_major_tt_target_row_major_tn_v1",
        "private_openblas_archive_sha256": expected_archive_sha256,
    }
    mismatches = {
        name: (expected, build.get(name))
        for name, expected in required.items()
        if build.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"native API/build contract mismatch: {mismatches}")
    integrity_enabled = bool(build.get("gemm_integrity_enabled"))
    if not integrity_enabled and not allow_integrity_disabled:
        raise RuntimeError(
            "benchmark evidence requires an integrity-enabled API-8 build; "
            "use --allow-integrity-disabled only for explicit diagnostics"
        )
    source_tree = str(build.get("source_tree_sha256", ""))
    if not _canonical_sha256(source_tree):
        raise RuntimeError("native build lacks canonical source-tree SHA256")
    return {
        "module_path": str(module_path.resolve()),
        "native_binary_sha256": _sha256(module_path.resolve()),
        "exact_build_hash": _sha256(module_path.resolve()),
        "embedded_source_tree_sha256": source_tree,
        "embedded_source_commit": build.get("source_commit"),
        "embedded_private_openblas_archive_sha256": build[
            "private_openblas_archive_sha256"
        ],
        "integrity_mode": {
            "enabled": integrity_enabled,
            "required_for_evidence": not allow_integrity_disabled,
            "diagnostic_override_used": (
                allow_integrity_disabled and not integrity_enabled
            ),
            "acceptance_eligible": integrity_enabled,
        },
        "build_info": build,
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    assert args.install_prefix is not None
    assert args.native_module is not None
    assert args.expected_native_sha256 is not None
    if not sys.flags.no_site:
        raise RuntimeError("benchmark worker must run under python -S")
    inserted_paths = _bootstrap_worker_paths(args.install_prefix)
    import numpy as np

    case = _cases(args)[0]
    if case.operation != args._operation:
        raise RuntimeError("worker operation selection is inconsistent")
    observed_module_sha = _sha256(args.native_module.resolve())
    if observed_module_sha != args.expected_native_sha256:
        raise RuntimeError("worker observed a different native module SHA256")
    module = _load_exact_native(args.native_module)
    configured = int(module.configure_blas_threads(case.threads))
    if configured != case.threads:
        raise RuntimeError(
            f"native runtime configured {configured}, requested {case.threads}"
        )
    # Repeating the same request proves idempotence without mutating the fixed
    # process-lifetime configuration.
    if int(module.configure_blas_threads(case.threads)) != case.threads:
        raise RuntimeError("native thread configuration is not immutable/idempotent")
    provenance = _native_provenance(
        module,
        args.native_module,
        args.expected_archive_sha256,
        args.allow_integrity_disabled,
    )
    build_info = provenance["build_info"]
    vendor_telemetry_required = _vendor_telemetry_required(case, build_info)
    affinity = _affinity()
    if affinity["allowed_cpu_count"] != case.threads:
        raise RuntimeError(
            "worker process affinity must contain one CPU per requested BLAS thread"
        )
    if len({tuple(value) for value in affinity["physical_core_keys"]}) != case.threads:
        raise RuntimeError("worker process affinity contains SMT siblings")
    prepared = _prepare(case, module, args.seed)
    operands = [_array_record(name, value) for name, value in prepared.operands]
    baseline_samples: list[float] | None = None
    baseline_output_sha256: str | None = None
    warmups: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    for repeat in range(args.warmups):
        record, samples, output_sha256 = _invoke_observed(
            case,
            prepared,
            module,
            repeat,
            "warmup",
            baseline_samples,
            baseline_output_sha256,
            vendor_telemetry_required,
        )
        if baseline_samples is None:
            baseline_samples = samples
            baseline_output_sha256 = output_sha256
        warmups.append(record)
    for repeat in range(args.repeats):
        record, samples, output_sha256 = _invoke_observed(
            case,
            prepared,
            module,
            repeat,
            "measured",
            baseline_samples,
            baseline_output_sha256,
            vendor_telemetry_required,
        )
        if baseline_samples is None:
            baseline_samples = samples
            baseline_output_sha256 = output_sha256
        measured.append(record)
    native_records = [
        record["native_vendor_record"]
        for record in measured
        if record["native_vendor_record"] is not None
    ]
    wrapper_wall = [record["wrapper_wall_seconds"] for record in measured]
    wrapper_cpu = [record["wrapper_process_cpu_seconds"] for record in measured]
    wrapper_cores = [record["wrapper_average_active_cores"] for record in measured]
    wrapper_rates = [record["wrapper_gflops_per_second"] for record in measured]
    vendor_wall = [float(record["wall_seconds"]) for record in native_records]
    vendor_cpu = [float(record["process_cpu_seconds"]) for record in native_records]
    vendor_cores = [
        float(record["active_core_equivalents"]) for record in native_records
    ]
    vendor_rates = [float(record["gflops_per_second"]) for record in native_records]
    summary: dict[str, Any] = {
        "wrapper_wall_seconds": _summary(wrapper_wall),
        "wrapper_process_cpu_seconds": _summary(wrapper_cpu),
        "wrapper_average_active_cores": _summary(wrapper_cores),
        "wrapper_gflops_per_second": _summary(wrapper_rates),
        "cumulative_wrapper_wall_seconds": float(math.fsum(wrapper_wall)),
        "maximum_rss_kib": max(
            record["resource_usage"]["maximum_rss_kib"] for record in measured
        ),
        "major_faults": sum(
            record["resource_usage"]["major_fault_delta"] for record in measured
        ),
    }
    if native_records:
        summary.update(
            {
                "vendor_wall_seconds": _summary(vendor_wall),
                "vendor_process_cpu_seconds": _summary(vendor_cpu),
                "vendor_average_active_cores": _summary(vendor_cores),
                "vendor_gflops_per_second": _summary(vendor_rates),
                "cumulative_vendor_wall_seconds": float(math.fsum(vendor_wall)),
                "matrix_minutes": float(math.fsum(vendor_wall) / 60.0),
            }
        )
    else:
        summary["vendor_telemetry"] = {
            "available": False,
            "reason": "integrity_tiled_path_below_vendor_flop_threshold",
        }
    all_observations = warmups + measured
    full_output_hashes = {
        str(record["full_output_sha256"]) for record in all_observations
    }
    full_output_determinism_verified = (
        len(all_observations) >= 2 and len(full_output_hashes) == 1
    )
    numa_validations = [
        record["numa_locality_validation"]
        for record in all_observations
        if record["numa_locality_validation"] is not None
    ]
    numa_acceptance_eligible = (
        all(
            bool(record["acceptance_eligible_on_exact_host"])
            for record in numa_validations
        )
        if numa_validations
        else None
    )
    protocol_conformant = args.warmups >= MIN_WARMUPS and args.repeats >= MIN_REPEATS
    measured_core_utilization_verified = all(
        record["vendor_core_utilization_gate"] is not None
        and record["vendor_core_utilization_gate"]["passed"] is True
        for record in measured
    )
    evidence_acceptance_eligible = (
        PRODUCTION_ACCEPTANCE_ELIGIBLE
        and protocol_conformant
        and bool(provenance["integrity_mode"]["acceptance_eligible"])
        and full_output_determinism_verified
        and vendor_telemetry_required
        and numa_acceptance_eligible is True
        and measured_core_utilization_verified
    )
    return {
        "case_id": case.case_id,
        "status": "ok",
        "case": dataclasses.asdict(case),
        "layout_mode": FULL_PRECISION_LAYOUT,
        "arithmetic_dtype": "float64",
        "storage_dtype": "float64",
        "source_panel_columns": case.source_columns,
        "target_sealed_pair_columns": case.target_columns,
        "flops_per_call": case.flops,
        "expected_vendor_call": case.expected_telemetry(),
        "native_vendor_telemetry": {
            "required_for_this_shape": vendor_telemetry_required,
            "integrity_minimum_vendor_flops": build_info.get(
                "gemm_integrity_minimum_vendor_flops"
            ),
            "exact_profile_requires_vendor_telemetry": True,
        },
        "evidence_acceptance": {
            "eligible": evidence_acceptance_eligible,
            "production_layout_eligible": PRODUCTION_ACCEPTANCE_ELIGIBLE,
            "production_rejection_reason": PRODUCTION_REJECTION_REASON,
            "protocol_conformant": protocol_conformant,
            "integrity_enabled": bool(provenance["integrity_mode"]["enabled"]),
            "full_output_determinism_verified": (full_output_determinism_verified),
            "vendor_telemetry_captured": bool(native_records),
            "numa_queried_and_local": numa_acceptance_eligible,
            "measured_vendor_core_utilization_verified": (
                measured_core_utilization_verified
            ),
        },
        "protocol": {
            "fresh_python_no_site": True,
            "sys_flags_no_site": int(sys.flags.no_site),
            "warmups": args.warmups,
            "measured_repeats": args.repeats,
            "sampled_dense_oracle": True,
            "full_output_bitwise_determinism_required_for_evidence": True,
            "full_output_bitwise_determinism_verified": (
                full_output_determinism_verified
            ),
            "full_output_hash_method": (
                "sha256_native_contiguous_memoryview_8mib_chunks"
            ),
            "repair_count_required": 0,
            "minimum_vendor_active_core_fraction": (
                MIN_VENDOR_ACTIVE_CORE_FRACTION
            ),
            "vendor_active_core_gate_scope": (
                "each_measured_call_excludes_warmups"
            ),
            "concurrent_vendor_calls": False,
            "immutable_thread_configuration": True,
            "integrity_enabled_required": not args.allow_integrity_disabled,
            "integrity_disabled_diagnostic_override": (args.allow_integrity_disabled),
        },
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": _sha256(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "explicit_worker_sys_paths": inserted_paths,
            "pythonpath_environment_removed": "PYTHONPATH" not in os.environ,
            "pythonhome_environment_removed": "PYTHONHOME" not in os.environ,
        },
        "native": provenance,
        "thread_configuration": {
            "requested_threads": case.threads,
            "configured_threads": configured,
            "second_same_request": case.threads,
            "placement_mode": "taskset_process_cpuset_confinement",
            "process_cpuset_confinement_verified": True,
            "individual_worker_cpu_placement_measured": False,
            "acceptance_caveat": (
                "OMP_PROC_BIND=false; this harness measures the established "
                "process-cpuset confinement convention; individual OpenMP "
                "worker CPU placement is not measured"
            ),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "OPENBLAS_NUM_THREADS",
                    "GOTO_NUM_THREADS",
                    "OMP_DYNAMIC",
                    "OMP_PROC_BIND",
                    "OMP_PLACES",
                    "OMP_MAX_ACTIVE_LEVELS",
                    "OMP_WAIT_POLICY",
                    "GOMP_SPINCOUNT",
                )
            },
        },
        "affinity": affinity,
        "memory_model": _estimated_memory(case),
        "memory_psi_before_calls": _memory_psi(),
        "operands": operands,
        "pair_construction": prepared.pair_construction,
        "pair_metadata": prepared.pair_metadata,
        "full_output_determinism": {
            "baseline_sha256": baseline_output_sha256,
            "unique_sha256_count": len(full_output_hashes),
            "call_count": len(all_observations),
            "verified": full_output_determinism_verified,
        },
        "numa_locality": {
            "vendor_record_count": len(numa_validations),
            "acceptance_eligible_on_exact_host": numa_acceptance_eligible,
            "requirement": (
                "all queried pages resolve without errors on NUMA nodes "
                "assigned to selected CPUs; unsupported/permission-denied "
                "is recorded but is not exact-host acceptance evidence"
            ),
        },
        "warmup_observations": warmups,
        "measured_observations": measured,
        "summary": summary,
        "memory_psi_after_calls": _memory_psi(),
        "final_resource_usage": _resource_snapshot(),
    }


def _kill_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.communicate()


def _parse_worker_result(stdout: str) -> dict[str, Any]:
    records = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(records) != 1:
        raise RuntimeError(
            f"worker emitted {len(records)} result records; expected exactly one"
        )
    return json.loads(records[0][len(RESULT_PREFIX) :])


def _run_case(
    args: argparse.Namespace,
    case: Case,
    cpu_pool: Sequence[int],
    timeout_seconds: float,
) -> dict[str, Any]:
    selected = list(cpu_pool[: case.threads])
    command = [
        "taskset",
        "-c",
        _compress_cpu_list(selected),
        *_worker_command(args, case),
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_worker_environment(case, selected),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_process_group(process)
        return {
            "case_id": case.case_id,
            "case": dataclasses.asdict(case),
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "controller_wall_seconds": time.monotonic() - started,
            "selected_physical_cpus": selected,
            "command": command,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
        }
    base = {
        "case_id": case.case_id,
        "case": dataclasses.asdict(case),
        "controller_wall_seconds": time.monotonic() - started,
        "selected_physical_cpus": selected,
        "selected_physical_core_keys": [
            list(_physical_core_key(cpu)) for cpu in selected
        ],
        "command": command,
        "returncode": process.returncode,
        "worker_stderr": stderr[-8000:],
    }
    if process.returncode != 0:
        return {**base, "status": "failed", "stdout_tail": stdout[-4000:]}
    try:
        result = _parse_worker_result(stdout)
    except (RuntimeError, json.JSONDecodeError) as error:
        return {
            **base,
            "status": "invalid_result",
            "reason": str(error),
            "stdout_tail": stdout[-4000:],
        }
    return {**base, **result}


def _dry_run_report(args: argparse.Namespace, cases: Sequence[Case]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "diagnostic_only": True,
        "production_acceptance_eligible": PRODUCTION_ACCEPTANCE_ELIGIBLE,
        "production_rejection_reason": PRODUCTION_REJECTION_REASON,
        "profile": args.profile,
        "resolved_profile": _profile(args),
        "limits": {
            "per_configuration_seconds": args.max_configuration_seconds,
            "sweep_seconds": args.max_sweep_seconds,
            "minimum_warmups": MIN_WARMUPS,
            "minimum_measured_repeats": MIN_REPEATS,
            "minimum_vendor_active_core_fraction": (
                MIN_VENDOR_ACTIVE_CORE_FRACTION
            ),
            "vendor_active_core_gate_scope": (
                "each_measured_call_excludes_warmups"
            ),
            "integrity_enabled_required": not args.allow_integrity_disabled,
            "atomic_no_replace_publication": True,
            "output_parent_must_preexist": True,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "case": dataclasses.asdict(case),
                "source_panel_columns": case.source_columns,
                "target_sealed_pair_columns": case.target_columns,
                "expected_vendor_call": case.expected_telemetry(),
                "flops_per_call": case.flops,
                "memory_model": _estimated_memory(case),
            }
            for case in cases
        ],
    }


def _controller_outcome(
    results: Sequence[dict[str, Any]], *, evidence_requested: bool
) -> dict[str, Any]:
    execution_complete = bool(results) and all(
        result.get("status") == "ok" for result in results
    )
    eligible_case_count = sum(
        result.get("status") == "ok"
        and result.get("evidence_acceptance", {}).get("eligible") is True
        for result in results
    )
    # This controller exercises a permanently rejected diagnostic orientation.
    # Per-case protocol success remains useful diagnostic evidence, but it must
    # never be promoted to an accepted production-layout result.
    evidence_eligible = False
    if not execution_complete:
        status = "incomplete"
    elif evidence_requested:
        status = "rejected"
    else:
        status = "diagnostic_complete"
    return {
        "status": status,
        "execution_complete": execution_complete,
        "evidence_requested": evidence_requested,
        "eligible": evidence_eligible,
        "eligible_case_count": eligible_case_count,
        "case_count": len(results),
    }


def _controller(args: argparse.Namespace) -> dict[str, Any]:
    assert args.install_prefix is not None
    assert args.native_module is not None
    cases = _cases(args)
    cpu_pool = args.cpus if args.cpus is not None else _default_physical_cpus()
    _validate_cpu_pool(cpu_pool, max(case.threads for case in cases))
    prefix = args.install_prefix.resolve()
    module = args.native_module.resolve()
    provenance = {
        **_prefix_identity(prefix),
        "native_module": str(module),
        "native_binary_sha256": _sha256(module),
        "expected_native_binary_sha256": args.expected_native_sha256,
        "expected_private_openblas_archive_sha256": args.expected_archive_sha256,
        "benchmark_script": str(Path(__file__).resolve()),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "python_executable": str(args.python_executable.resolve()),
        "python_executable_sha256": _sha256(args.python_executable.resolve()),
        "linkage": _linkage_evidence(module),
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "controller_python": sys.version,
            "controller_affinity": _affinity(),
        },
        "profile": args.profile,
        "resolved_profile": _profile(args),
        "diagnostic_only": True,
        "production_acceptance_eligible": PRODUCTION_ACCEPTANCE_ELIGIBLE,
        "production_rejection_reason": PRODUCTION_REJECTION_REASON,
        "limits": {
            "per_configuration_seconds": args.max_configuration_seconds,
            "sweep_seconds": args.max_sweep_seconds,
            "minimum_warmups": MIN_WARMUPS,
            "minimum_measured_repeats": MIN_REPEATS,
            "actual_warmups": args.warmups,
            "actual_measured_repeats": args.repeats,
            "minimum_vendor_active_core_fraction": (
                MIN_VENDOR_ACTIVE_CORE_FRACTION
            ),
            "vendor_active_core_gate_scope": (
                "each_measured_call_excludes_warmups"
            ),
            "protocol_conformant": (
                args.warmups >= MIN_WARMUPS and args.repeats >= MIN_REPEATS
            ),
            "fresh_python_no_site_per_case": True,
            "one_vendor_call_at_a_time": True,
            "full_production_run": False,
            "integrity_enabled_required": not args.allow_integrity_disabled,
            "atomic_no_replace_publication": True,
            "output_parent_must_preexist": True,
        },
        "provenance": provenance,
        "placement": {
            "scope": "process_cpuset_confinement",
            "physical_cpu_pool": cpu_pool,
            "physical_core_keys": [list(_physical_core_key(cpu)) for cpu in cpu_pool],
        },
        "results": [],
    }
    memory_limit = int(float(_profile(args)["max_memory_gib"]) * 1024**3)
    sweep_started = time.monotonic()
    for case in cases:
        elapsed = time.monotonic() - sweep_started
        remaining = args.max_sweep_seconds - elapsed
        if remaining <= 0.5:
            report["results"].append(
                {
                    "case_id": case.case_id,
                    "case": dataclasses.asdict(case),
                    "status": "not_started_sweep_budget_exhausted",
                }
            )
            continue
        memory = _estimated_memory(case)
        if int(memory["conservative_peak_bytes"]) > memory_limit:
            report["results"].append(
                {
                    "case_id": case.case_id,
                    "case": dataclasses.asdict(case),
                    "status": "skipped_memory_bound",
                    "memory_model": memory,
                    "memory_limit_bytes": memory_limit,
                }
            )
            continue
        report["results"].append(
            _run_case(
                args,
                case,
                cpu_pool,
                min(args.max_configuration_seconds, remaining - 0.5),
            )
        )
    report["sweep_wall_seconds"] = time.monotonic() - sweep_started
    report["status_counts"] = {
        status: sum(result.get("status") == status for result in report["results"])
        for status in sorted(
            {str(result.get("status")) for result in report["results"]}
        )
    }
    report["evidence_acceptance"] = _controller_outcome(
        report["results"], evidence_requested=args.profile == "exact"
    )
    report["status"] = report["evidence_acceptance"]["status"]
    return report


def _write_json(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    parent = output.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"output parent must already exist for atomic publication: {parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", dir=parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        linked = True
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not linked:
        raise RuntimeError("atomic report publication did not create the target")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args._worker:
        if args._operation is None:
            parser.error("internal worker requires --_operation")
        args.operations = (args._operation,)
    _validate_args(parser, args)
    if args._worker:
        try:
            result = _run_worker(args)
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return 0
    cases = _cases(args)
    if args.dry_run:
        _write_json(_dry_run_report(args, cases), args.output)
        return 0
    report = _controller(args)
    _write_json(report, args.output)
    outcome = report["evidence_acceptance"]
    if not outcome["execution_complete"]:
        return 2
    if outcome["evidence_requested"] and not outcome["eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
