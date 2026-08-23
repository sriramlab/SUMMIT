#!/usr/bin/env python3
"""Run bounded fresh-process diagnostics for the rare protected NN event.

Each child is a new ``python -S`` process confined to exactly two physical
CPUs.  It constructs the fixed seed-70131, page-offset-eight Fortran operands,
then makes exactly one call to the private integrity diagnostic API.  Optional
history modes run a declared, bounded prelude before that diagnostic call.

The controller is deliberately fail-closed.  Its diagnostic gate passes only
when every child completes and every result is either clean or is classified
by native long-double evidence as a checksum-tolerance false positive.  Vendor
damage, changed inputs, and unclassified repairs always fail that gate.  The
top-level ``accepted`` field is always false: this is diagnostic evidence,
never production-layout acceptance evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from typing import Any, Mapping, Sequence


SCHEMA = "summit.gxe.current_nn_integrity_fresh_process_diagnostic"
SCHEMA_VERSION = 1
RESULT_PREFIX = "SUMMIT_GXE_NN_INTEGRITY_RESULT="
DIAGNOSTIC_API = "_test_protected_matmul_nn_integrity_diagnostic"
OPENMP_PLACEMENT_SCHEMA = "summit.openmp_placement_attestation.v1"
DEFAULT_ARCHIVE_SHA256 = (
    "49609db51d9c91bb4beaa40af20ef6a57698698b8043b3d9ac15ffc25554f521"
)
EXPECTED_BACKENDS = ("openblas", "blis")
BLIS_SOURCE_COMMIT = "e8566eb3e773fb54d11b33e371d13f22d2941e50"
BLIS_SOURCE_TREE_SHA256 = (
    "eefbd29a5cbb1d6982bdce76e8034037ea2a3f3ead33d90d9f3cef6da5728154"
)
BLIS_HEADER_SHA256 = (
    "85d5ac5f094a0a2824fd4245baaeb58e4880a9cc90049eec733dc676ea7a4132"
)
BLIS_CBLAS_HEADER_SHA256 = (
    "e62171d7e7f66faf6a801981b743e10bb07d17261219c6b1a4aba071c27af6d6"
)
BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR = (
    "BLIS_NT",
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
    "BLIS_THREAD_IMPL",
    "BLIS_TI",
    "BLIS_ARCH_TYPE",
    "BLIS_ARCH_DEBUG",
    "BLIS_PACK_A",
    "BLIS_PACK_B",
)
SEED = 70131
M = 512
N = 512
K = 2048
THREADS = 2
LEFT_PAGE_OFFSET = 8
FLOPS = 2 * M * N * K
INTEGRITY_MINIMUM_VENDOR_FLOPS = 1_000_000_000
DEFAULT_WORKERS = 20
MAX_WORKERS = 20
DEFAULT_WORKER_TIMEOUT_SECONDS = 40.0
MAX_WORKER_TIMEOUT_SECONDS = 40.0
DEFAULT_CONTROLLER_TIMEOUT_SECONDS = 20.0 * 60.0
MAX_CONTROLLER_TIMEOUT_SECONDS = 20.0 * 60.0
HISTORY_MODES = (
    "none",
    "allocation-churn",
    "subthreshold-nn",
    "subthreshold-tn",
)
SAFE_CLASSIFICATIONS = {
    "no_current_gate_flags",
    "checksum_tolerance_false_positive_under_cancellation_aware_bound",
}
REJECTED_CLASSIFICATIONS = {
    "vendor_result_outside_forward_error_bound",
    "input_fingerprint_changed",
    "unclassified",
    "unclassified_detail_cap_exceeded",
}


class _ControllerDeadlineExpired(TimeoutError):
    pass


def _raise_controller_deadline(_signum: int, _frame: Any) -> None:
    raise _ControllerDeadlineExpired(
        "NN integrity diagnostic exceeded its controller wall-clock cap"
    )
FINGERPRINT_EQUALITY_KEYS = {
    "original_unchanged_after_expected",
    "original_unchanged_after_copy",
    "original_unchanged_after_vendor",
    "original_unchanged_after_deterministic",
    "original_unchanged_after_reference",
    "protected_unchanged_after_vendor",
    "original_equals_protected_before_vendor",
    "original_equals_protected_after_vendor",
}
ORIGINAL_FINGERPRINT_KEYS = {
    "initial",
    "after_expected",
    "after_copy",
    "after_vendor",
    "after_deterministic",
    "after_reference",
}
PROTECTED_FINGERPRINT_KEYS = {"before_vendor", "after_vendor"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _csv_history_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes or len(modes) != len(set(modes)):
        raise argparse.ArgumentTypeError("history modes must be nonempty and unique")
    unknown = sorted(set(modes).difference(HISTORY_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown history mode(s): " + ", ".join(unknown)
        )
    return modes


def _expand_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
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
                cpus.extend(range(first, last + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError
                cpus.append(cpu)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid CPU list") from error
    if not cpus or len(cpus) != len(set(cpus)):
        raise argparse.ArgumentTypeError("CPU list must be nonempty and unique")
    return cpus


def _compress_cpu_list(cpus: Sequence[int]) -> str:
    ordered = sorted(cpus)
    ranges: list[str] = []
    if not ordered:
        return ""
    first = last = ordered[0]
    for cpu in ordered[1:]:
        if cpu == last + 1:
            last = cpu
            continue
        ranges.append(str(first) if first == last else f"{first}-{last}")
        first = last = cpu
    ranges.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(ranges)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument(
        "--expected-archive-sha256", default=DEFAULT_ARCHIVE_SHA256
    )
    parser.add_argument(
        "--expected-backend", choices=EXPECTED_BACKENDS, default="openblas"
    )
    parser.add_argument(
        "--require-openmp-placement",
        action="store_true",
        help=(
            "Opt into the API-9/backend-1.6 exact singleton OpenMP placement "
            "contract. The default retains the historical API-8 diagnostic."
        ),
    )
    parser.add_argument(
        "--python-executable", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--dependency-path", type=Path, action="append", default=None,
        help="Explicit dependency directory for python -S; repeat as needed.",
    )
    parser.add_argument("--cpus", type=_expand_cpu_list, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--history-modes", type=_csv_history_modes, default=("none",),
        help="Comma-separated deterministic history modes, scheduled round-robin.",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--controller-timeout-seconds",
        type=float,
        default=DEFAULT_CONTROLLER_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_history-mode", choices=HISTORY_MODES, help=argparse.SUPPRESS
    )
    return parser


def _expected_backend(args: argparse.Namespace) -> str:
    value = getattr(args, "expected_backend", "openblas")
    if value not in EXPECTED_BACKENDS:
        raise RuntimeError(f"unsupported expected private BLAS backend: {value}")
    return value


def _placement_requested(args: argparse.Namespace) -> bool:
    value = getattr(args, "require_openmp_placement", False)
    if type(value) is not bool:
        raise RuntimeError("OpenMP placement mode must be an exact boolean")
    return value


def _regular_executable(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    mode = resolved.stat().st_mode
    if not resolved.is_file() or not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise ValueError(f"not a regular executable file: {resolved}")
    return resolved


def _dependency_paths(args: argparse.Namespace) -> list[Path]:
    values = args.dependency_path
    if values is None:
        values = [
            Path(value)
            for value in (sysconfig.get_path("platlib"), sysconfig.get_path("purelib"))
            if value
        ]
    result: list[Path] = []
    for value in values:
        resolved = value.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"dependency path is not a directory: {resolved}")
        if resolved not in result:
            result.append(resolved)
    if not result:
        raise ValueError("at least one dependency path is required for python -S")
    return result


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be in [1, {MAX_WORKERS}]")
    if not 0.0 < args.worker_timeout_seconds <= MAX_WORKER_TIMEOUT_SECONDS:
        parser.error(
            f"--worker-timeout-seconds must be in (0, {MAX_WORKER_TIMEOUT_SECONDS:g}]"
        )
    if not 0.0 < args.controller_timeout_seconds <= MAX_CONTROLLER_TIMEOUT_SECONDS:
        parser.error(
            "--controller-timeout-seconds must be in "
            f"(0, {MAX_CONTROLLER_TIMEOUT_SECONDS:g}]"
        )
    if args.workers * args.worker_timeout_seconds > args.controller_timeout_seconds:
        parser.error("workers multiplied by worker timeout exceeds controller cap")
    if not _canonical_sha256(args.expected_native_sha256):
        parser.error("--expected-native-sha256 must be lowercase canonical SHA256")
    if not _canonical_sha256(args.expected_archive_sha256):
        parser.error("--expected-archive-sha256 must be lowercase canonical SHA256")
    lexical_output = args.output.expanduser().absolute()
    if lexical_output.is_symlink() or lexical_output.exists():
        parser.error(f"refusing to overwrite existing output: {lexical_output}")
    try:
        output_parent = lexical_output.parent.resolve(strict=True)
    except FileNotFoundError:
        parser.error(f"output parent must already exist: {lexical_output.parent}")
    if not output_parent.is_dir():
        parser.error(f"output parent must be a directory: {output_parent}")
    resolved_output = output_parent / lexical_output.name
    if resolved_output.is_symlink() or resolved_output.exists():
        parser.error(f"refusing to overwrite existing output: {resolved_output}")
    try:
        prefix = args.install_prefix.expanduser().resolve(strict=True)
        module = args.native_module.expanduser().resolve(strict=True)
        _regular_executable(args.python_executable)
        _dependency_paths(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if not prefix.is_dir():
        parser.error(f"--install-prefix is not a directory: {prefix}")
    if not module.is_file() or not stat.S_ISREG(module.stat().st_mode):
        parser.error(f"--native-module is not a regular file: {module}")
    try:
        relative = module.relative_to(prefix)
    except ValueError:
        parser.error("--native-module must resolve inside --install-prefix")
    if (
        relative.parent != Path("summit")
        or not module.name.startswith("gxeldcore.")
        or module.suffix != ".so"
    ):
        parser.error("--native-module must be an installed summit/gxeldcore.*.so")
    observed = _sha256(module)
    if observed != args.expected_native_sha256:
        parser.error(
            f"native SHA256 mismatch: expected {args.expected_native_sha256}, observed {observed}"
        )
    if args.cpus is not None and len(args.cpus) != THREADS:
        parser.error(f"--cpus must name exactly {THREADS} CPUs")
    if args._worker:
        if args._worker_index is None or args._worker_index < 0:
            parser.error("internal worker requires nonnegative --_worker-index")
        if args._history_mode is None:
            parser.error("internal worker requires --_history-mode")


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
        raise RuntimeError(f"physical-core topology is unavailable for CPU {cpu}")
    return package, core


def _cpu_numa_node(cpu: int) -> int | None:
    nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    if not nodes:
        return None
    try:
        return int(nodes[0].name.removeprefix("node"))
    except ValueError:
        return None


def _default_physical_cpus() -> list[int]:
    result: list[int] = []
    seen: set[tuple[int, int]] = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        key = _physical_core_key(cpu)
        if key in seen:
            continue
        seen.add(key)
        result.append(cpu)
        if len(result) == THREADS:
            break
    return result


def _resolve_cpus(requested: Sequence[int] | None) -> list[int]:
    selected = sorted(requested) if requested is not None else _default_physical_cpus()
    if len(selected) != THREADS:
        raise RuntimeError(f"exact diagnostic requires {THREADS} physical CPUs")
    outside = sorted(set(selected).difference(os.sched_getaffinity(0)))
    if outside:
        raise RuntimeError(f"selected CPUs are outside controller affinity: {outside}")
    keys = [_physical_core_key(cpu) for cpu in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError("selected CPUs contain SMT siblings")
    return selected


def _package_identity(prefix: Path) -> dict[str, Any]:
    package = prefix / "summit"
    if not package.is_dir():
        raise RuntimeError(f"installed prefix lacks summit package: {prefix}")
    records: list[tuple[str, int, str]] = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            raise RuntimeError(f"installed package contains bytecode/cache: {relative}")
        if path.is_symlink():
            raise RuntimeError(f"installed package contains a symbolic link: {relative}")
        if path.is_file():
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
        "file_count": len(records),
        "bytes": sum(size for _, size, _ in records),
        "manifest_sha256": digest.hexdigest(),
    }


def _run_checked(command: Sequence[str], timeout: float = 20.0) -> str:
    return subprocess.run(
        list(command), check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=timeout,
    ).stdout


def _linkage_evidence(module: Path) -> dict[str, Any]:
    readelf, nm = shutil.which("readelf"), shutil.which("nm")
    if readelf is None or nm is None:
        raise RuntimeError("readelf and nm are required for exact linkage verification")
    dynamic = _run_checked([readelf, "-d", str(module)])
    symbols = _run_checked([nm, "-D", str(module)])
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    dynamic_blas = [
        name for name in needed
        if "blas" in Path(name).name.lower()
        or Path(name).name.lower().startswith("libblis")
    ]
    dynamic_blas_symbols = [
        line for line in symbols.splitlines()
        if re.search(
            r"\b(?:cblas_|openblas_|bli_|dgemm_|sgemm_|xerbla_|"
            r"CBLAS_CallFromC\b|RowMajorStrg\b)",
            line,
        )
    ]
    if dynamic_blas:
        raise RuntimeError(f"extension has a dynamic BLAS dependency: {dynamic_blas}")
    if dynamic_blas_symbols:
        raise RuntimeError(
            "private BLAS symbols are exported or unresolved in the extension"
        )
    return {
        "needed_shared_libraries": needed,
        "dynamic_blas_dependencies": dynamic_blas,
        "exported_or_unresolved_blas_symbol_count": len(dynamic_blas_symbols),
        "readelf_dynamic_sha256": hashlib.sha256(dynamic.encode()).hexdigest(),
        "all_dynamic_symbols_sha256": hashlib.sha256(symbols.encode()).hexdigest(),
        "private_static_linkage_verified": True,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": info.st_size,
        "sha256": _sha256(resolved),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }


def _affinity() -> dict[str, Any]:
    cpus = sorted(os.sched_getaffinity(0))
    return {
        "allowed_cpus": cpus,
        "allowed_cpu_list": _compress_cpu_list(cpus),
        "allowed_cpu_count": len(cpus),
        "physical_core_keys": [list(_physical_core_key(cpu)) for cpu in cpus],
        "numa_nodes": sorted(
            {node for cpu in cpus if (node := _cpu_numa_node(cpu)) is not None}
        ),
    }


def _worker_environment(
    expected_backend: str = "openblas",
    placement_cpus: Sequence[int] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    if expected_backend not in EXPECTED_BACKENDS:
        raise RuntimeError(f"unsupported expected private BLAS backend: {expected_backend}")
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX", "OMP_PLACES",
        "OMP_NESTED", "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "KMP_HW_SUBSET",
        "KMP_PLACE_THREADS", "LD_PRELOAD",
    ):
        environment.pop(name, None)
    if expected_backend == "blis":
        for name in BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR:
            environment.pop(name, None)
    settings = {
        "OMP_NUM_THREADS": str(THREADS),
        "OMP_THREAD_LIMIT": str(THREADS),
        "OPENBLAS_NUM_THREADS": str(THREADS),
        "GOTO_NUM_THREADS": str(THREADS),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "GOMP_SPINCOUNT": "0",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    if expected_backend == "blis":
        settings["BLIS_NUM_THREADS"] = str(THREADS)
    if placement_cpus is not None:
        if (
            len(placement_cpus) != THREADS
            or len(set(placement_cpus)) != THREADS
            or any(type(cpu) is not int or cpu < 0 for cpu in placement_cpus)
        ):
            raise RuntimeError(
                f"exact OpenMP placement requires {THREADS} unique CPU IDs"
            )
        settings.update(
            {
                "OMP_PROC_BIND": "SPREAD",
                "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in placement_cpus),
            }
        )
    environment.update(settings)
    return environment, settings


def _validate_blis_process_environment(environment: Mapping[str, str]) -> None:
    if environment.get("BLIS_NUM_THREADS") != str(THREADS):
        raise RuntimeError("BLIS_NUM_THREADS was not fixed before process import")
    unexpected = [
        name for name in BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR
        if name in environment
    ]
    if unexpected:
        raise RuntimeError(f"unexpected BLIS process-start overrides: {unexpected}")


def _validate_openmp_placement_process_environment(
    environment: Mapping[str, str], expected_cpus: Sequence[int]
) -> None:
    expected = {
        "OMP_NUM_THREADS": str(THREADS),
        "OMP_THREAD_LIMIT": str(THREADS),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in expected_cpus),
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }
    mismatches = {
        name: {"expected": value, "observed": environment.get(name)}
        for name, value in expected.items()
        if environment.get(name) != value
    }
    forbidden = (
        "OMP_NESTED", "GOMP_CPU_AFFINITY", "KMP_AFFINITY",
        "KMP_HW_SUBSET", "KMP_PLACE_THREADS",
    )
    present = [name for name in forbidden if name in environment]
    if mismatches or present:
        raise RuntimeError(
            "OpenMP placement process-start environment mismatch: "
            f"values={mismatches}, forbidden={present}"
        )


def _worker_command(
    args: argparse.Namespace, worker_index: int, history_mode: str,
    dependencies: Sequence[Path],
) -> list[str]:
    command = [
        str(args.python_executable.expanduser().resolve()), "-S",
        str(Path(__file__).resolve()), "--_worker",
        "--_worker-index", str(worker_index),
        "--_history-mode", history_mode,
        "--install-prefix", str(args.install_prefix.expanduser().resolve()),
        "--native-module", str(args.native_module.expanduser().resolve()),
        "--expected-native-sha256", args.expected_native_sha256,
        "--expected-archive-sha256", args.expected_archive_sha256,
        "--workers", "1", "--worker-timeout-seconds", "1",
        "--controller-timeout-seconds", "1",
        "--output", str(args.output.expanduser().resolve()),
    ]
    for dependency in dependencies:
        command.extend(("--dependency-path", str(dependency)))
    if _placement_requested(args):
        command.append("--require-openmp-placement")
    expected_backend = _expected_backend(args)
    if expected_backend != "openblas":
        command.extend(("--expected-backend", expected_backend))
    return command


def _taskset_worker_command(
    args: argparse.Namespace, worker_index: int, history_mode: str,
    dependencies: Sequence[Path], cpus: Sequence[int],
) -> list[str]:
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required")
    return [
        taskset, "-c", _compress_cpu_list(cpus),
        *_worker_command(args, worker_index, history_mode, dependencies),
    ]


def _bootstrap_worker_paths(prefix: Path, dependencies: Sequence[Path]) -> list[str]:
    inserted: list[str] = []
    for path in reversed([prefix, *dependencies]):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
            inserted.append(value)
    return list(reversed(inserted))


def _load_exact_native(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("gxeldcore", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("loaded native path differs from the requested path")
    return module


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0.0 else "-inf"
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    return value


def _array_sha256(array: Any) -> str:
    flat = array.ravel(order="K")
    view = memoryview(flat).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start : start + 8 * 1024 * 1024])
    return digest.hexdigest()


def _array_record(array: Any) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "strides": list(array.strides),
        "bytes": int(array.nbytes),
        "fortran_contiguous": bool(array.flags.f_contiguous),
        "c_contiguous": bool(array.flags.c_contiguous),
        "aligned": bool(array.flags.aligned),
        "sha256_storage_order": _array_sha256(array),
    }


def _misaligned_fortran_normal(rng: Any, shape: tuple[int, int], np: Any) -> Any:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    byte_count = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float64).itemsize
    backing = np.empty(byte_count + page_size, dtype=np.uint8)
    offset = (LEFT_PAGE_OFFSET - int(backing.ctypes.data) % page_size) % page_size
    result = np.ndarray(shape, dtype=np.float64, buffer=backing, offset=offset, order="F")
    result[...] = rng.normal(size=shape)
    if (
        not result.flags.f_contiguous or not result.flags.aligned
        or int(result.ctypes.data) % page_size != LEFT_PAGE_OFFSET
    ):
        raise RuntimeError("failed to construct the exact page-offset-eight operand")
    return result


def _run_history(mode: str, module: Any, np: Any) -> dict[str, Any]:
    started = time.perf_counter()
    module.reset_gemm_telemetry()
    integrity_threshold = int(
        dict(module.build_info()).get("gemm_integrity_minimum_vendor_flops", 0)
    )
    if integrity_threshold <= 0:
        raise RuntimeError("history prelude lacks a positive integrity threshold")
    record: dict[str, Any] = {
        "mode": mode,
        "deterministic": True,
        "completed": False,
        "vendor_call_count": 0,
        "diagnostic_call_count": 0,
    }
    if mode == "none":
        pass
    elif mode == "allocation-churn":
        sizes = (1 * 1024**2, 4 * 1024**2, 16 * 1024**2, 4 * 1024**2, 1 * 1024**2)
        fingerprints: list[str] = []
        for ordinal, size in enumerate(sizes):
            allocation = np.empty(size, dtype=np.uint8)
            allocation[::4096] = (ordinal * 37 + 11) % 256
            fingerprints.append(hashlib.sha256(allocation[::4096].tobytes()).hexdigest())
            del allocation
        gc.collect()
        record.update(
            {
                "allocation_sizes_bytes": list(sizes),
                "maximum_live_requested_bytes": max(sizes),
                "touched_page_stride_bytes": 4096,
                "touched_page_fingerprints": fingerprints,
            }
        )
    else:
        rng = np.random.default_rng(170131)
        hm, hn, hk = 128, 128, 256
        if mode == "subthreshold-nn":
            left = np.asfortranarray(rng.normal(size=(hm, hk)))
            right = np.asfortranarray(rng.normal(size=(hk, hn)))
            output, repaired = module.protected_matmul_nn(left, right, THREADS)
            operation = "dgemm_nn"
        elif mode == "subthreshold-tn":
            left = np.asfortranarray(rng.normal(size=(hk, hm)))
            right = np.asfortranarray(rng.normal(size=(hk, hn)))
            output, repaired = module.protected_matmul_tn(left, right, THREADS)
            operation = "dgemm_tn"
        else:
            raise RuntimeError(f"unsupported history mode: {mode}")
        if int(repaired) != 0:
            raise RuntimeError("sub-threshold history call unexpectedly repaired output")
        record.update(
            {
                "protected_call_count": 1,
                "operation": operation,
                "shape": {"m": hm, "n": hn, "k": hk},
                "flops": 2 * hm * hn * hk,
                "integrity_threshold_crossed": (
                    2 * hm * hn * hk >= integrity_threshold
                ),
                "output_sha256": _array_sha256(output),
                "repaired_columns": int(repaired),
            }
        )
        del left, right, output
        gc.collect()
    telemetry = [_json_safe(dict(item)) for item in module.consume_gemm_telemetry()]
    telemetry_status = _json_safe(dict(module.gemm_telemetry_status()))
    if record.get("integrity_threshold_crossed") is True:
        raise RuntimeError("declared subthreshold history crossed the native threshold")
    if telemetry:
        raise RuntimeError("history prelude unexpectedly entered vendor GEMM")
    if (
        telemetry_status.get("buffered_records") != 0
        or telemetry_status.get("dropped_records") != 0
    ):
        raise RuntimeError("history telemetry was not empty and lossless")
    record.update(
        {
            "completed": True,
            "wall_seconds": time.perf_counter() - started,
            "vendor_call_count": len(telemetry),
            "telemetry": telemetry,
            "telemetry_status": telemetry_status,
            "native_integrity_minimum_vendor_flops": integrity_threshold,
        }
    )
    return record


def _validate_extended_precision_capability(raw: Mapping[str, Any]) -> dict[str, Any]:
    capabilities = dict(raw.get("floating_point_capabilities", {}))
    if set(capabilities) != {"long_double", "double"}:
        raise RuntimeError("diagnostic floating-point capability schema is incomplete")
    parsed: dict[str, dict[str, Any]] = {}
    required = {"sizeof_bytes", "digits", "digits10", "max_digits10", "epsilon"}
    for name in ("long_double", "double"):
        values = dict(capabilities.get(name, {}))
        if set(values) != required:
            raise RuntimeError(f"diagnostic {name} capability schema is incomplete")
        for integer_name in ("sizeof_bytes", "digits", "digits10", "max_digits10"):
            if type(values[integer_name]) is not int or values[integer_name] <= 0:
                raise RuntimeError(
                    f"diagnostic {name}.{integer_name} is not a positive built-in integer"
                )
        if type(values["epsilon"]) is not float:
            raise RuntimeError(
                f"diagnostic {name}.epsilon is not a built-in float"
            )
        if not math.isfinite(values["epsilon"]) or values["epsilon"] <= 0.0:
            raise RuntimeError(f"diagnostic {name}.epsilon is not finite positive")
        parsed[name] = values
    extended, binary64 = parsed["long_double"], parsed["double"]
    if extended["sizeof_bytes"] < binary64["sizeof_bytes"]:
        raise RuntimeError("long double storage is narrower than double storage")
    if extended["digits"] <= binary64["digits"]:
        raise RuntimeError("long double does not provide more precision bits than double")
    if extended["epsilon"] >= binary64["epsilon"]:
        raise RuntimeError("long double epsilon is not smaller than double epsilon")
    return {
        "validated": True,
        "long_double_digits_exceed_double": True,
        "long_double_epsilon_smaller_than_double": True,
        "long_double_size_at_least_double": True,
    }


def _valid_fingerprint(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "xor_hash_uint64", "sum_hash_uint64"
    }:
        return False
    for item in value.values():
        if type(item) is not str or not re.fullmatch(r"[0-9]{1,20}", item):
            return False
        if int(item) > 2**64 - 1:
            return False
    return True


def _compact_diagnostic(diagnostic: Mapping[str, Any], output: Any, np: Any) -> dict[str, Any]:
    raw = dict(diagnostic)
    precision_validation = _validate_extended_precision_capability(raw)
    required_contract = {
        "schema_version": 1,
        "integrity_check_eligible": True,
        "diagnostic_executed": True,
        "m": M,
        "n": N,
        "k": K,
        "long_double_reference_accumulator": "C++ long double",
        "long_double_reference_array_storage": (
            "binary64_cast_with_full_precision_decimal_companion"
        ),
        "classification_changes_production_decision": False,
    }
    contract_mismatches = {
        name: {"expected": expected, "observed": raw.get(name)}
        for name, expected in required_contract.items()
        if raw.get(name) != expected
    }
    if contract_mismatches:
        raise RuntimeError(
            f"native diagnostic contract mismatch: {contract_mismatches}"
        )
    minimum_vendor_flops = int(raw.get("minimum_vendor_flops", 0))
    if minimum_vendor_flops != INTEGRITY_MINIMUM_VENDOR_FLOPS:
        raise RuntimeError("native diagnostic reports a changed vendor threshold")
    fault_injection = dict(raw.get("fault_injection", {}))
    if fault_injection.get("enabled") is not False:
        raise RuntimeError("fresh-process evidence forbids diagnostic fault injection")
    arrays: dict[str, Any] = {}
    for name in (
        "raw_vendor_columns",
        "deterministic_tiled_columns",
        "long_double_reference_columns",
    ):
        if name not in raw:
            raise RuntimeError(f"diagnostic lacks {name}")
        arrays[name] = raw.pop(name)
    decimals = raw.pop("long_double_reference_decimal_columns", None)
    if decimals is None:
        raise RuntimeError("diagnostic lacks long-double decimal columns")
    repaired = int(raw.get("flagged_column_count", -1))
    flagged_columns = [int(value) for value in raw.get("flagged_columns", [])]
    if (
        repaired != len(flagged_columns)
        or flagged_columns != sorted(set(flagged_columns))
        or any(column < 0 or column >= N for column in flagged_columns)
    ):
        raise RuntimeError("diagnostic flagged-column count is inconsistent")
    captured_columns = [
        int(value) for value in raw.get("captured_flagged_columns", [])
    ]
    captured = int(raw.get("captured_flagged_column_count", -1))
    dropped = int(raw.get("dropped_flagged_column_count", -1))
    detail_limit = int(raw.get("detailed_column_limit", -1))
    if (
        detail_limit != 16
        or captured != len(captured_columns)
        or captured_columns != flagged_columns[:captured]
        or not 0 <= captured <= repaired
        or captured > detail_limit
        or dropped != repaired - captured
    ):
        raise RuntimeError("diagnostic captured-column count is inconsistent")
    for name, array in arrays.items():
        if tuple(array.shape) != (M, captured) or str(array.dtype) != "float64":
            raise RuntimeError(f"diagnostic {name} has an unexpected shape/dtype")
        if not array.flags.f_contiguous:
            raise RuntimeError(f"diagnostic {name} is not Fortran contiguous")
    if len(decimals) != captured or any(len(column) != M for column in decimals):
        raise RuntimeError("long-double decimal column shape is inconsistent")
    decimal_witnesses_valid = True
    for ordinal, column in enumerate(decimals):
        if any(type(value) is not str for value in column):
            decimal_witnesses_valid = False
            continue
        try:
            parsed = np.asarray(column, dtype=np.longdouble)
        except (TypeError, ValueError, OverflowError):
            decimal_witnesses_valid = False
            continue
        companion = arrays["long_double_reference_columns"][:, ordinal]
        if (
            not bool(np.isfinite(parsed).all())
            or not np.array_equal(parsed.astype(np.float64), companion)
        ):
            decimal_witnesses_valid = False
    deterministic = arrays["deterministic_tiled_columns"]
    output_repair_match = all(
        np.array_equal(output[:, column], deterministic[:, ordinal], equal_nan=True)
        for ordinal, column in enumerate(captured_columns)
    )
    comparisons = [dict(item) for item in raw.get("column_comparisons", [])]
    if (
        len(comparisons) != captured
        or [int(item.get("column", -1)) for item in comparisons]
        != captured_columns
    ):
        raise RuntimeError("diagnostic column-comparison count is inconsistent")
    classification = str(raw.get("classification", "unclassified"))
    known = SAFE_CLASSIFICATIONS | REJECTED_CLASSIFICATIONS
    if classification not in known:
        raise RuntimeError(f"unknown native diagnostic classification: {classification}")
    failed_checks = [dict(item) for item in raw.get("flagged_checks", [])]
    flagged_check_ids = [dict(item) for item in raw.get("flagged_check_ids", [])]
    total_checks = int(raw.get("flagged_check_count", -1))
    captured_checks = int(raw.get("captured_flagged_check_count", -1))
    dropped_checks = int(raw.get("dropped_flagged_check_count", -1))
    if (
        total_checks != len(flagged_check_ids)
        or captured_checks != len(failed_checks)
        or dropped_checks != total_checks - captured_checks
        or not 0 <= captured_checks <= total_checks
    ):
        raise RuntimeError("diagnostic flagged-check detail counts are inconsistent")
    check_identifiers: list[tuple[int, int]] = []
    for item in flagged_check_ids:
        if type(item.get("column")) is not int or type(item.get("check")) is not int:
            raise RuntimeError("diagnostic check identifiers are not built-in integers")
        identifier = (item["column"], item["check"])
        if identifier[0] not in flagged_columns or not 0 <= identifier[1] < 8:
            raise RuntimeError("diagnostic check identifier is out of range")
        check_identifiers.append(identifier)
    if check_identifiers != sorted(set(check_identifiers)):
        raise RuntimeError("diagnostic check identifiers are duplicated or unordered")
    if repaired > 0 and {
        column for column, _check in check_identifiers
    } != set(flagged_columns):
        raise RuntimeError("not every flagged column has a failed check identifier")
    detailed_identifiers: list[tuple[int, int]] = []
    for item in failed_checks:
        if type(item.get("column")) is not int or type(item.get("check")) is not int:
            raise RuntimeError("diagnostic detailed check IDs are not built-in integers")
        detailed_identifiers.append((item["column"], item["check"]))
    expected_detailed = [
        identifier for identifier in check_identifiers
        if identifier[0] in captured_columns
    ]
    if detailed_identifiers != expected_detailed:
        raise RuntimeError("diagnostic detailed checks do not match captured IDs")
    expected_first = check_identifiers[0] if check_identifiers else (None, None)
    if (
        raw.get("first_flagged_column") != expected_first[0]
        or raw.get("first_flagged_check") != expected_first[1]
    ):
        raise RuntimeError("diagnostic first flagged check is inconsistent")

    checks_numerically_consistent = True
    binary64_epsilon = float(
        dict(dict(raw["floating_point_capabilities"])["double"])["epsilon"]
    )
    gamma_m = (M * binary64_epsilon) / (1.0 - M * binary64_epsilon)
    gamma_k = (K * binary64_epsilon) / (1.0 - K * binary64_epsilon)
    production_relative_bound = 32.0 * (
        2.0 * gamma_m + gamma_k + binary64_epsilon
    )
    for item in failed_checks:
        numeric_names = (
            "expected", "observed", "difference", "absolute_difference",
            "current_relative_bound", "current_tolerance",
            "direct_absolute_product_sum", "factored_expected_absolute_sum",
            "observed_checksum_absolute_sum", "projection_roundoff_component",
            "expected_reduction_roundoff_component",
            "vendor_product_roundoff_component",
            "observed_reduction_roundoff_component", "cancellation_aware_bound",
        )
        if any(type(item.get(name)) is not float for name in numeric_names):
            checks_numerically_consistent = False
            continue
        values = {name: float(item[name]) for name in numeric_names}
        if not all(math.isfinite(value) for value in values.values()):
            checks_numerically_consistent = False
            continue
        expected_difference = values["expected"] - values["observed"]
        expected_tolerance = values["current_relative_bound"] * max(
            1.0, abs(values["expected"]), abs(values["observed"])
        )
        expected_cancellation_disagrees = (
            values["absolute_difference"] > values["cancellation_aware_bound"]
        )
        absolute_sum_names = (
            "direct_absolute_product_sum",
            "factored_expected_absolute_sum",
            "observed_checksum_absolute_sum",
            "projection_roundoff_component",
            "expected_reduction_roundoff_component",
            "vendor_product_roundoff_component",
            "observed_reduction_roundoff_component",
            "cancellation_aware_bound",
        )
        component_sum = np.longdouble(0.0)
        for name in (
            "projection_roundoff_component",
            "expected_reduction_roundoff_component",
            "vendor_product_roundoff_component",
            "observed_reduction_roundoff_component",
        ):
            component_sum += np.longdouble(values[name])
        epsilon_scale = max(
            1.0,
            values["direct_absolute_product_sum"],
            values["factored_expected_absolute_sum"],
            values["observed_checksum_absolute_sum"],
        )
        reconstructed_cancellation_bound = float(
            np.longdouble(32.0)
            * (
                component_sum
                + np.longdouble(binary64_epsilon)
                * np.longdouble(epsilon_scale)
            )
        )
        if (
            not math.isclose(
                values["difference"], expected_difference,
                rel_tol=2.0e-15, abs_tol=0.0,
            )
            or values["absolute_difference"] != abs(values["difference"])
            or not math.isclose(
                values["current_tolerance"], expected_tolerance,
                rel_tol=2.0e-15, abs_tol=0.0,
            )
            or values["current_relative_bound"] <= 0.0
            or not math.isclose(
                values["current_relative_bound"], production_relative_bound,
                rel_tol=2.0e-15, abs_tol=0.0,
            )
            or values["current_tolerance"] <= 0.0
            or values["absolute_difference"] <= values["current_tolerance"]
            or any(values[name] < 0.0 for name in absolute_sum_names)
            or not math.isclose(
                values["cancellation_aware_bound"],
                reconstructed_cancellation_bound,
                rel_tol=2.0e-15, abs_tol=0.0,
            )
            or item.get("cancellation_aware_disagrees")
            is not expected_cancellation_disagrees
        ):
            checks_numerically_consistent = False

    comparison_witnesses_consistent = True
    for ordinal, item in enumerate(comparisons):
        raw_column = arrays["raw_vendor_columns"][:, ordinal]
        tiled_column = arrays["deterministic_tiled_columns"][:, ordinal]
        reference_column = arrays["long_double_reference_columns"][:, ordinal]
        extended_reference_column = np.asarray(
            [str(value) for value in decimals[ordinal]], dtype=np.longdouble
        )
        expected_counts = {
            "raw_vendor_nonfinite_count": int(np.count_nonzero(~np.isfinite(raw_column))),
            "deterministic_tiled_nonfinite_count": int(np.count_nonzero(~np.isfinite(tiled_column))),
            "long_double_reference_nonfinite_count": int(np.count_nonzero(~np.isfinite(reference_column))),
            "vendor_tiled_unequal_count": int(np.count_nonzero(raw_column != tiled_column)),
            "vendor_reference_unequal_count": int(np.count_nonzero(raw_column != reference_column)),
            "tiled_reference_unequal_count": int(np.count_nonzero(tiled_column != reference_column)),
        }
        if any(
            type(item.get(name)) is not int or item.get(name) != value
            for name, value in expected_counts.items()
        ):
            comparison_witnesses_consistent = False
        expected_maxima = {
            "max_abs_vendor_minus_tiled": (
                raw_column.astype(np.longdouble)
                - tiled_column.astype(np.longdouble)
            ),
            "max_abs_vendor_minus_long_double": (
                raw_column.astype(np.longdouble) - extended_reference_column
            ),
            "max_abs_tiled_minus_long_double": (
                tiled_column.astype(np.longdouble) - extended_reference_column
            ),
        }
        for name, differences in expected_maxima.items():
            absolute_differences = np.abs(differences)
            expected_value = (
                float(np.max(absolute_differences, initial=0.0))
                if bool(np.isfinite(absolute_differences).all())
                else math.inf
            )
            observed_value = item.get(name)
            if type(observed_value) is not float or not (
                observed_value == expected_value
                or (
                    math.isfinite(observed_value)
                    and math.isfinite(expected_value)
                    and math.isclose(
                        observed_value, expected_value,
                        rel_tol=2.0e-15, abs_tol=0.0,
                    )
                )
            ):
                comparison_witnesses_consistent = False

    equalities = dict(raw.get("fingerprint_equalities", {}))
    fingerprints = dict(raw.get("fingerprints", {}))
    original_fingerprints = dict(fingerprints.get("original_b", {}))
    protected_fingerprints = dict(fingerprints.get("protected_b_snapshot", {}))
    fingerprint_schema_valid = (
        set(fingerprints) == {"original_b", "protected_b_snapshot"}
        and set(original_fingerprints) == ORIGINAL_FINGERPRINT_KEYS
        and set(protected_fingerprints) == PROTECTED_FINGERPRINT_KEYS
        and all(_valid_fingerprint(value) for value in original_fingerprints.values())
        and all(_valid_fingerprint(value) for value in protected_fingerprints.values())
        and set(equalities) == FINGERPRINT_EQUALITY_KEYS
        and all(type(value) is bool for value in equalities.values())
    )
    expected_equalities = {}
    if fingerprint_schema_valid:
        initial = original_fingerprints["initial"]
        expected_equalities = {
            "original_unchanged_after_expected": (
                initial == original_fingerprints["after_expected"]
            ),
            "original_unchanged_after_copy": (
                initial == original_fingerprints["after_copy"]
            ),
            "original_unchanged_after_vendor": (
                initial == original_fingerprints["after_vendor"]
            ),
            "original_unchanged_after_deterministic": (
                initial == original_fingerprints["after_deterministic"]
            ),
            "original_unchanged_after_reference": (
                initial == original_fingerprints["after_reference"]
            ),
            "protected_unchanged_after_vendor": (
                protected_fingerprints["before_vendor"]
                == protected_fingerprints["after_vendor"]
            ),
            "original_equals_protected_before_vendor": (
                original_fingerprints["after_copy"]
                == protected_fingerprints["before_vendor"]
            ),
            "original_equals_protected_after_vendor": (
                original_fingerprints["after_vendor"]
                == protected_fingerprints["after_vendor"]
            ),
        }
    fingerprint_reports_consistent = (
        fingerprint_schema_valid and equalities == expected_equalities
    )
    fingerprints_stable = fingerprint_reports_consistent and all(
        value is True for value in expected_equalities.values()
    )
    common_evidence_valid = (
        precision_validation["validated"] is True
        and fingerprint_schema_valid
        and fingerprint_reports_consistent
        and comparison_witnesses_consistent
        and checks_numerically_consistent
        and decimal_witnesses_valid
    )
    if repaired == 0:
        contract_valid = (
            classification == "no_current_gate_flags"
            and not failed_checks
            and total_checks == 0
            and captured == 0
            and dropped == 0
            and fingerprints_stable
            and output_repair_match
            and common_evidence_valid
        )
    elif classification == "checksum_tolerance_false_positive_under_cancellation_aware_bound":
        contract_valid = (
            bool(failed_checks)
            and all(item.get("cancellation_aware_disagrees") is False for item in failed_checks)
            and all(
                item.get("classification")
                == "checksum_tolerance_false_positive_under_cancellation_aware_bound"
                for item in comparisons
            )
            and all(
                int(item.get("raw_vendor_nonfinite_count", -1)) == 0
                and int(item.get("deterministic_tiled_nonfinite_count", -1)) == 0
                and int(item.get("long_double_reference_nonfinite_count", -1)) == 0
                and int(item.get("vendor_rows_outside_forward_error_bound", -1)) == 0
                and int(item.get("tiled_rows_outside_forward_error_bound", -1)) == 0
                for item in comparisons
            )
            and dropped == 0
            and dropped_checks == 0
            and fingerprints_stable and output_repair_match
            and common_evidence_valid
        )
    elif classification == "vendor_result_outside_forward_error_bound":
        contract_valid = (
            any(int(item.get("vendor_rows_outside_forward_error_bound", 0)) > 0 for item in comparisons)
            and all(int(item.get("tiled_rows_outside_forward_error_bound", -1)) == 0 for item in comparisons)
            and dropped == 0
            and dropped_checks == 0
            and fingerprints_stable and output_repair_match
            and common_evidence_valid
        )
    elif classification == "input_fingerprint_changed":
        contract_valid = (
            not fingerprints_stable
            and output_repair_match
            and common_evidence_valid
        )
    elif classification == "unclassified_detail_cap_exceeded":
        contract_valid = (
            dropped > 0 and output_repair_match and common_evidence_valid
        )
    else:
        contract_valid = output_repair_match and common_evidence_valid
    if not contract_valid:
        classification = "unclassified"
    compact_arrays = {name: _array_record(array) for name, array in arrays.items()}
    decimal_summaries = [
        {
            "value_count": len(column),
            "canonical_json_sha256": _canonical_json_sha256(column),
        }
        for column in decimals
    ]
    numeric_witnesses = {
        name: {
            "captured_flagged_columns": list(captured_columns),
            "columns": [
                _json_safe(array[:, ordinal].tolist())
                for ordinal in range(captured)
            ],
        }
        for name, array in arrays.items()
    }
    decimal_witnesses = [
        [str(value) for value in column] for column in decimals
    ]
    return {
        **_json_safe(raw),
        "classification": classification,
        "native_classification": str(raw.get("classification", "unclassified")),
        "classification_contract_valid": contract_valid,
        "safe_diagnostic_classification": classification in SAFE_CLASSIFICATIONS,
        "all_repairs_classified": classification not in {
            "unclassified", "unclassified_detail_cap_exceeded"
        },
        "fingerprints_stable": fingerprints_stable,
        "fingerprint_schema_valid": fingerprint_schema_valid,
        "fingerprint_reports_consistent": fingerprint_reports_consistent,
        "check_numeric_evidence_consistent": checks_numerically_consistent,
        "comparison_witnesses_consistent": comparison_witnesses_consistent,
        "decimal_witnesses_valid": decimal_witnesses_valid,
        "extended_precision_capability_validation": precision_validation,
        "returned_output_matches_deterministic_repair_at_flagged_columns": output_repair_match,
        "column_arrays": compact_arrays,
        "captured_numeric_column_witnesses": numeric_witnesses,
        "long_double_reference_decimal_column_summaries": decimal_summaries,
        "long_double_reference_decimal_columns": decimal_witnesses,
        "raw_numeric_columns_persisted": captured > 0,
    }


def _validate_openmp_placement_attestation(
    value: Mapping[str, Any], expected_cpus: Sequence[int]
) -> dict[str, Any]:
    required_keys = {
        "schema", "schema_version", "verified", "immutable",
        "requested_threads", "expected_cpu_ids", "omp_dynamic",
        "omp_thread_limit", "omp_max_active_levels", "omp_proc_bind",
        "omp_binding_active", "omp_num_places", "effective_openmp_capacity",
        "place_cpu_ids", "team_size", "exact_singleton_places",
        "exact_team_coverage", "workers", "vendor_calls",
    }
    if not isinstance(value, Mapping) or set(value) != required_keys:
        raise RuntimeError("OpenMP placement attestation has an unexpected schema")
    cpus = list(expected_cpus)
    exact = {
        "schema": OPENMP_PLACEMENT_SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": THREADS,
        "expected_cpu_ids": cpus,
        "omp_dynamic": False,
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": THREADS,
        "place_cpu_ids": [[cpu] for cpu in cpus],
        "team_size": THREADS,
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "vendor_calls": 0,
    }
    mismatches = {
        name: {"expected": expected, "observed": value.get(name)}
        for name, expected in exact.items() if value.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"OpenMP placement contract mismatch: {mismatches}")
    integer_fields = (
        "schema_version", "requested_threads", "omp_thread_limit",
        "omp_max_active_levels", "omp_num_places", "effective_openmp_capacity",
        "team_size", "vendor_calls",
    )
    boolean_fields = (
        "verified", "immutable", "omp_dynamic", "omp_binding_active",
        "exact_singleton_places", "exact_team_coverage",
    )
    if any(type(value.get(name)) is not int for name in integer_fields):
        raise RuntimeError("OpenMP placement integer fields are not exact integers")
    if any(type(value.get(name)) is not bool for name in boolean_fields):
        raise RuntimeError("OpenMP placement boolean fields are not exact booleans")
    if (
        type(value.get("expected_cpu_ids")) is not list
        or any(type(cpu) is not int for cpu in value["expected_cpu_ids"])
        or type(value.get("place_cpu_ids")) is not list
        or any(
            type(place) is not list or any(type(cpu) is not int for cpu in place)
            for place in value["place_cpu_ids"]
        )
    ):
        raise RuntimeError("OpenMP placement CPU fields are not exact integer lists")
    if (
        value["omp_thread_limit"] < THREADS
        or value["effective_openmp_capacity"] < THREADS
    ):
        raise RuntimeError("OpenMP placement capacity is below two threads")
    workers = value.get("workers")
    if type(workers) is not list or len(workers) != THREADS:
        raise RuntimeError("OpenMP placement worker count mismatch")
    worker_keys = {
        "thread_num", "place_num", "place_cpu_ids", "sched_affinity_cpu_ids",
        "current_cpu", "verified",
    }
    for index, worker in enumerate(workers):
        if not isinstance(worker, Mapping) or set(worker) != worker_keys:
            raise RuntimeError("OpenMP placement worker schema mismatch")
        expected_worker = {
            "thread_num": index,
            "place_num": index,
            "place_cpu_ids": [cpus[index]],
            "sched_affinity_cpu_ids": [cpus[index]],
            "current_cpu": cpus[index],
            "verified": True,
        }
        if dict(worker) != expected_worker:
            raise RuntimeError(
                f"OpenMP placement worker {index} mismatch: {dict(worker)}"
            )
        if (
            type(worker["thread_num"]) is not int
            or type(worker["place_num"]) is not int
            or type(worker["current_cpu"]) is not int
            or type(worker["verified"]) is not bool
            or type(worker["place_cpu_ids"]) is not list
            or type(worker["sched_affinity_cpu_ids"]) is not list
            or any(type(cpu) is not int for cpu in worker["place_cpu_ids"])
            or any(
                type(cpu) is not int
                for cpu in worker["sched_affinity_cpu_ids"]
            )
        ):
            raise RuntimeError("OpenMP placement worker types are not exact")
    return dict(value)


def _validate_build_info(
    module: Any, module_path: Path, expected_archive_sha256: str,
    expected_backend: str = "openblas",
    placement_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_build = dict(module.build_info())
    build = {str(key): _json_safe(value) for key, value in raw_build.items()}
    placement_required = placement_attestation is not None
    required = {
        "api_version": 9 if placement_required else 8,
        "backend_version": "1.9" if placement_required else "1.5",
        "blas_runtime_isolation": "private_static",
        "gemm_integrity_enabled": True,
        "gemm_vendor_entry_outer_openmp_guard": True,
    }
    if placement_required:
        required.update(
            {
                "openmp_effective_capacity_policy": (
                    "bound_places_else_sched_affinity_v1"
                ),
                "openmp_placement_contract_supported": True,
                "openmp_placement_contract_schema": OPENMP_PLACEMENT_SCHEMA,
                "openmp_placement_contract_configured": True,
                "openmp_placement_contract_immutable": True,
                "openmp_placement_probe_vendor_calls": 0,
            }
        )
    if expected_backend == "openblas":
        required["private_openblas_archive_sha256"] = expected_archive_sha256
    elif expected_backend == "blis":
        required.update(
            {
                "blas_vendor": "BLIS",
                "gemm_execution_mode": "serialized_fixed_private_blis",
                "blas_runtime_config": "BLIS 2.0 config=zen",
                "blas_runtime_corename": "zen",
                "blas_runtime_threads": THREADS,
                "blas_runtime_threading_layer": "pthreads",
                "blas_runtime_worker_affinity_policy": (
                    "inherit_authenticated_selected_cpu_set_per_call"
                ),
                "private_openblas_archive_sha256": "none",
                "private_blas_backend": "upstream_blis",
                "private_blas_archive_sha256": expected_archive_sha256,
                "private_blas_source_commit": BLIS_SOURCE_COMMIT,
                "private_blas_source_tree_sha256": BLIS_SOURCE_TREE_SHA256,
                "private_blas_config_family": "zen",
                "private_blas_header_sha256": BLIS_HEADER_SHA256,
                "private_blas_cblas_header_sha256": BLIS_CBLAS_HEADER_SHA256,
                "blas_runtime_thread_strategy": "automatic",
                "blas_runtime_owner_thread_enforced": True,
                "blas_runtime_owner_thread_configured": True,
                "blas_runtime_environment_immutable": True,
                "blas_runtime_environment_contract": "blis_process_start_v1",
                "blas_runtime_tls_enabled": True,
            }
        )
    else:
        raise RuntimeError(f"unsupported expected private BLAS backend: {expected_backend}")
    mismatches = {
        name: {"expected": expected, "observed": build.get(name)}
        for name, expected in required.items() if build.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"native build contract mismatch: {mismatches}")
    if placement_required:
        strict_integer_fields = (
            "api_version", "openmp_placement_probe_vendor_calls",
        )
        strict_true_fields = (
            "gemm_integrity_enabled", "gemm_vendor_entry_outer_openmp_guard",
            "openmp_placement_contract_supported",
            "openmp_placement_contract_configured",
            "openmp_placement_contract_immutable",
        )
        if any(type(raw_build.get(name)) is not int for name in strict_integer_fields):
            raise RuntimeError(
                "OpenMP placement build integer fields must be built-in ints"
            )
        if any(type(raw_build.get(name)) is not bool for name in strict_true_fields):
            raise RuntimeError(
                "OpenMP placement build boolean fields must be built-in bools"
            )
        raw_evidence = raw_build.get("openmp_placement_contract_evidence")
        if not isinstance(raw_evidence, Mapping):
            raise RuntimeError("native build lacks OpenMP placement evidence")
        validated_evidence = _validate_openmp_placement_attestation(
            raw_evidence, placement_attestation["expected_cpu_ids"]
        )
        if _canonical_json_sha256(validated_evidence) != _canonical_json_sha256(
            placement_attestation
        ):
            raise RuntimeError(
                "native build OpenMP placement evidence differs from configured evidence"
            )
    if expected_backend == "blis":
        strict_integer_fields = (
            "api_version",
            "blas_runtime_threads",
            "gemm_integrity_minimum_vendor_flops",
        )
        if any(type(build.get(name)) is not int for name in strict_integer_fields):
            raise RuntimeError("BLIS integer build contract fields must be built-in ints")
        strict_true_fields = (
            "gemm_integrity_enabled",
            "gemm_vendor_entry_outer_openmp_guard",
            "blas_runtime_owner_thread_enforced",
            "blas_runtime_owner_thread_configured",
            "blas_runtime_environment_immutable",
            "blas_runtime_tls_enabled",
        )
        if any(type(build.get(name)) is not bool for name in strict_true_fields):
            raise RuntimeError("BLIS boolean build contract fields must be built-in bools")
        ways = build.get("blas_runtime_thread_ways")
        expected_way_names = {"jc", "pc", "ic", "jr", "ir"}
        if not isinstance(ways, dict) or set(ways) != expected_way_names:
            raise RuntimeError("BLIS runtime thread ways have an unexpected schema")
        if any(type(value) is not int or value <= 0 for value in ways.values()):
            raise RuntimeError("BLIS runtime thread ways must be positive built-in integers")
        if any(value != 1 for value in ways.values()):
            raise RuntimeError("automatic BLIS runtime thread ways are not all one")
        canonical_build_hashes = (
            "source_tree_sha256",
            "private_blas_source_tree_sha256",
            "private_blas_header_sha256",
            "private_blas_cblas_header_sha256",
        )
        if any(
            type(build.get(name)) is not str
            or not _canonical_sha256(build[name])
            for name in canonical_build_hashes
        ):
            raise RuntimeError("BLIS build provenance lacks canonical SHA256 evidence")
        if type(build.get("private_blas_source_commit")) is not str or not re.fullmatch(
            r"[0-9a-f]{40}", build["private_blas_source_commit"]
        ):
            raise RuntimeError("BLIS build provenance lacks a canonical source commit")
    source_tree = str(build.get("source_tree_sha256", ""))
    if not _canonical_sha256(source_tree):
        raise RuntimeError("native build lacks canonical source-tree SHA256")
    threshold = int(build.get("gemm_integrity_minimum_vendor_flops", 0))
    if threshold != INTEGRITY_MINIMUM_VENDOR_FLOPS or FLOPS < threshold:
        raise RuntimeError("native build reports a changed integrity threshold")
    if not callable(getattr(module, DIAGNOSTIC_API, None)):
        raise RuntimeError(f"native module lacks {DIAGNOSTIC_API}")
    return {
        "module_path": str(module_path.resolve()),
        "module_sha256": _sha256(module_path.resolve()),
        "build_info": build,
        "build_info_canonical_json_sha256": _canonical_json_sha256(build),
    }


def _validate_telemetry_gate(
    record: Mapping[str, Any], affinity: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    positive_fields = (
        "wall_seconds",
        "process_cpu_seconds",
        "gflops_per_second",
        "process_cpu_to_wall_ratio",
        "active_core_equivalents",
    )
    numeric_values: dict[str, float | None] = {}
    for name in positive_fields:
        try:
            value = float(record.get(name))
        except (TypeError, ValueError):
            value = math.nan
        numeric_values[name] = value if math.isfinite(value) else None
        if not math.isfinite(value) or value <= 0.0:
            reasons.append(f"{name}_not_finite_positive")
    minimum_active_cores = 0.75 * THREADS
    active = numeric_values["active_core_equivalents"]
    if active is None or active < minimum_active_cores:
        reasons.append("active_core_equivalents_below_75_percent_of_threads")
    allowed_cpus = set(int(value) for value in affinity.get("allowed_cpus", []))
    for name in ("entry_cpu", "exit_cpu"):
        try:
            cpu = int(record.get(name))
        except (TypeError, ValueError):
            cpu = -1
        if cpu not in allowed_cpus:
            reasons.append(f"{name}_outside_taskset")

    numa = dict(record.get("operand_numa_page_samples", {}))
    if numa.get("sampling_method") != "move_pages_query_no_migration":
        reasons.append("unexpected_numa_sampling_method")
    operands = {
        str(name): dict(value)
        for name, value in dict(numa.get("operands", {})).items()
    }
    if set(operands) != {"a", "b", "c"}:
        reasons.append("numa_operand_set_mismatch")
    selected_nodes = set(int(value) for value in affinity.get("numa_nodes", []))
    if not selected_nodes:
        reasons.append("selected_cpu_numa_nodes_unavailable")
    operand_gates: dict[str, Any] = {}
    for name in ("a", "b", "c"):
        operand = operands.get(name, {})
        try:
            selected_pages = int(operand.get("selected_sample_pages", -1))
            resolved_pages = int(operand.get("resolved_sample_pages", -1))
            error_pages = int(operand.get("page_query_error_pages", -1))
            histogram = {
                int(node): int(count)
                for node, count in dict(operand.get("node_histogram", {})).items()
            }
        except (TypeError, ValueError):
            selected_pages = resolved_pages = error_pages = -1
            histogram = {}
        nodes = set(histogram)
        passed = (
            operand.get("query_status") == "queried"
            and selected_pages > 0
            and resolved_pages == selected_pages
            and error_pages == 0
            and sum(histogram.values()) == resolved_pages
            and bool(nodes)
            and nodes.issubset(selected_nodes)
        )
        if not passed:
            reasons.append(f"numa_operand_{name}_not_fully_local_and_resolved")
        operand_gates[name] = {
            "passed": passed,
            "query_status": operand.get("query_status"),
            "selected_sample_pages": selected_pages,
            "resolved_sample_pages": resolved_pages,
            "page_query_error_pages": error_pages,
            "node_histogram": {str(node): count for node, count in histogram.items()},
            "observed_nodes": sorted(nodes),
            "within_selected_cpu_numa_nodes": bool(nodes) and nodes.issubset(selected_nodes),
        }
    return {
        "passed": not reasons,
        "reasons": reasons,
        "minimum_active_core_equivalents": minimum_active_cores,
        "numeric_values": numeric_values,
        "selected_cpu_numa_nodes": sorted(selected_nodes),
        "operands": operand_gates,
    }


def _dense_oracle_comparison(left: Any, right: Any, output: Any, np: Any) -> dict[str, Any]:
    rtol = 3.0e-14
    atol = 3.0e-12
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    oracle = np.matmul(left, right)
    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    finite = bool(np.isfinite(output).all() and np.isfinite(oracle).all())
    if finite:
        absolute = np.abs(output - oracle)
        tolerance = atol + rtol * np.abs(oracle)
        normalized = absolute / tolerance
        relative = absolute / np.maximum(
            np.abs(oracle), np.finfo(np.float64).tiny
        )
        flat = int(np.argmax(normalized))
        row, column = (int(value) for value in np.unravel_index(flat, output.shape))
        maximum_absolute = float(np.max(absolute))
        maximum_relative = float(np.max(relative))
        maximum_normalized = float(normalized[row, column])
        passed = bool(maximum_normalized <= 1.0)
        output_at_max = float(output[row, column])
        oracle_at_max = float(oracle[row, column])
        tolerance_at_max = float(tolerance[row, column])
    else:
        row = column = -1
        maximum_absolute = maximum_relative = maximum_normalized = math.inf
        output_at_max = oracle_at_max = tolerance_at_max = math.nan
        passed = False
    return {
        "passed": passed,
        "finite": finite,
        "rtol": rtol,
        "atol": atol,
        "comparison": "abs(output-oracle) <= atol + rtol*abs(oracle)",
        "maximum_absolute_error": maximum_absolute,
        "maximum_relative_error": maximum_relative,
        "maximum_tolerance_normalized_error": maximum_normalized,
        "maximum_normalized_error_location": {"row": row, "column": column},
        "output_at_maximum_normalized_error": output_at_max,
        "oracle_at_maximum_normalized_error": oracle_at_max,
        "tolerance_at_maximum_normalized_error": tolerance_at_max,
        "output_sha256": _array_sha256(output),
        "oracle_sha256": _array_sha256(oracle),
        "oracle_shape": list(oracle.shape),
        "oracle_dtype": str(oracle.dtype),
        "oracle_wall_seconds": wall_seconds,
        "oracle_process_cpu_seconds": cpu_seconds,
        "oracle_executed_after_native_diagnostic": True,
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not sys.flags.no_site:
        raise RuntimeError("worker must run under python -S")
    if Path(sys.executable).resolve() != args.python_executable.resolve():
        raise RuntimeError("worker interpreter differs from --python-executable")
    expected_backend = _expected_backend(args)
    placement_requested = _placement_requested(args)
    launcher_affinity = _affinity()
    if launcher_affinity["allowed_cpu_count"] != THREADS:
        raise RuntimeError("worker launcher affinity does not contain exactly two CPUs")
    if (
        len({tuple(value) for value in launcher_affinity["physical_core_keys"]})
        != THREADS
    ):
        raise RuntimeError("worker launcher affinity contains SMT siblings")
    placement_cpus = launcher_affinity["allowed_cpus"]
    if expected_backend == "blis":
        _validate_blis_process_environment(os.environ)
    if placement_requested:
        _validate_openmp_placement_process_environment(os.environ, placement_cpus)
    assert args._worker_index is not None and args._history_mode is not None
    prefix = args.install_prefix.resolve()
    module_path = args.native_module.resolve()
    dependencies = _dependency_paths(args)
    inserted = _bootstrap_worker_paths(prefix, dependencies)
    import numpy as np

    numpy_module_path = Path(np.__file__).resolve()
    if not any(
        numpy_module_path == dependency
        or numpy_module_path.is_relative_to(dependency)
        for dependency in dependencies
    ):
        raise RuntimeError("NumPy was not imported from a declared dependency path")

    if _sha256(module_path) != args.expected_native_sha256:
        raise RuntimeError("native module changed before import")
    module = _load_exact_native(module_path)
    placement_attestation = None
    if placement_requested:
        configure_placement = getattr(module, "configure_openmp_placement", None)
        if not callable(configure_placement):
            raise RuntimeError(
                "API-9 native module lacks configure_openmp_placement"
            )
        placement_attestation = _validate_openmp_placement_attestation(
            configure_placement(list(placement_cpus), THREADS), placement_cpus
        )
    configured = int(module.configure_blas_threads(THREADS))
    if configured != THREADS or int(module.configure_blas_threads(THREADS)) != THREADS:
        raise RuntimeError("private BLAS thread configuration is not fixed at two")
    native = _validate_build_info(
        module, module_path, args.expected_archive_sha256, expected_backend,
        placement_attestation,
    )
    calling_thread_affinity = _affinity()
    if (
        placement_requested
        and calling_thread_affinity["allowed_cpus"] != [placement_cpus[0]]
    ):
        raise RuntimeError(
            "configured calling-thread affinity differs from singleton place zero"
        )

    history = _run_history(args._history_mode, module, np)
    preparation_started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    left = _misaligned_fortran_normal(rng, (M, K), np)
    right = np.asfortranarray(rng.normal(size=(K, N)))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    before = {"left": _array_record(left), "right": _array_record(right)}
    preparation_seconds = time.perf_counter() - preparation_started

    module.reset_gemm_telemetry()
    call_wall_started = time.perf_counter()
    call_cpu_started = time.process_time()
    output, repaired_count, diagnostic = getattr(module, DIAGNOSTIC_API)(
        left, right, THREADS
    )
    call_cpu_seconds = time.process_time() - call_cpu_started
    call_wall_seconds = time.perf_counter() - call_wall_started
    postprocess_started = time.perf_counter()
    telemetry = [_json_safe(dict(item)) for item in module.consume_gemm_telemetry()]
    telemetry_status = _json_safe(dict(module.gemm_telemetry_status()))
    after = {"left": _array_record(left), "right": _array_record(right)}
    if before != after:
        raise RuntimeError("Python operand bytes or metadata changed across the call")
    if tuple(output.shape) != (M, N) or str(output.dtype) != "float64" or not output.flags.f_contiguous:
        raise RuntimeError("diagnostic returned an unexpected output array")
    if int(repaired_count) != int(dict(diagnostic).get("flagged_column_count", -1)):
        raise RuntimeError("repaired count differs from diagnostic flagged count")
    if len(telemetry) != 1:
        raise RuntimeError(f"diagnostic call produced {len(telemetry)} telemetry records")
    if (
        telemetry_status.get("buffered_records") != 0
        or telemetry_status.get("dropped_records") != 0
    ):
        raise RuntimeError("diagnostic telemetry was not consumed without loss")
    expected_telemetry = {
        "operation": "dgemm_nn", "arithmetic_dtype": "float64",
        "layout": "column_major", "transpose_a": "N", "transpose_b": "N",
        "m": M, "n": N, "k": K, "lda": M, "ldb": K, "ldc": M,
        "requested_threads": THREADS, "configured_threads": THREADS,
        "backend_threads": THREADS, "omp_in_parallel": False,
        "omp_level": 0, "omp_active_level": 0, "completed": True,
    }
    if expected_backend == "blis":
        expected_telemetry.update(
            {
                "backend": "BLIS",
                "backend_config": "BLIS 2.0 config=zen",
                "backend_corename": "zen",
            }
        )
    mismatches = {
        key: {"expected": expected, "observed": telemetry[0].get(key)}
        for key, expected in expected_telemetry.items()
        if telemetry[0].get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"vendor telemetry contract mismatch: {mismatches}")
    if float(telemetry[0].get("flop_count", -1.0)) != float(FLOPS):
        raise RuntimeError("vendor telemetry reports the wrong flop count")
    if (
        telemetry[0].get("cpu_affinity_count")
        != (1 if placement_requested else THREADS)
        or telemetry[0].get("cpu_affinity_list")
        != (
            str(placement_cpus[0])
            if placement_requested else launcher_affinity["allowed_cpu_list"]
        )
    ):
        raise RuntimeError("vendor telemetry affinity differs from taskset placement")
    if placement_requested and any(
        telemetry[0].get(name) != placement_cpus[0]
        for name in ("entry_cpu", "exit_cpu")
    ):
        raise RuntimeError(
            "vendor entry/exit CPU differs from singleton place zero"
        )
    telemetry_gate = _validate_telemetry_gate(telemetry[0], launcher_affinity)
    compact = _compact_diagnostic(dict(diagnostic), output, np)
    output_record = _array_record(output)
    postprocess_seconds = time.perf_counter() - postprocess_started
    dense_oracle = _dense_oracle_comparison(left, right, output, np)
    if _sha256(module_path) != args.expected_native_sha256:
        raise RuntimeError("native module changed after diagnostic execution")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "status": "ok",
        "worker_index": args._worker_index,
        "pid": os.getpid(),
        "history_mode": args._history_mode,
        "runtime_attestation": {
            "python_no_site": bool(sys.flags.no_site),
            "python_version": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "inserted_import_paths": inserted,
            "numpy_version": np.__version__,
            "numpy_module": _file_identity(numpy_module_path),
            "native": native,
            "affinity": launcher_affinity,
            "thread_environment": {
                name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
                    "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_MAX_ACTIVE_LEVELS",
                    "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
                    *(("OMP_PLACES",) if placement_requested else ()),
                    *(
                        ("BLIS_NUM_THREADS", *BLIS_ENVIRONMENT_OVERRIDES_TO_CLEAR)
                        if expected_backend == "blis" else ()
                    ),
                )
            },
            **(
                {
                    "openmp_placement": placement_attestation,
                    "post_configuration_calling_thread_affinity": (
                        calling_thread_affinity
                    ),
                }
                if placement_requested else {}
            ),
        },
        "phase_seconds": {
            "history": history["wall_seconds"],
            "operand_preparation_and_hash": preparation_seconds,
            "diagnostic_call_wall": call_wall_seconds,
            "diagnostic_call_process_cpu": call_cpu_seconds,
            "diagnostic_postprocess_and_output_hash": postprocess_seconds,
            "independent_dense_numpy_oracle_wall": dense_oracle[
                "oracle_wall_seconds"
            ],
            "independent_dense_numpy_oracle_process_cpu": dense_oracle[
                "oracle_process_cpu_seconds"
            ],
        },
        "history": history,
        "call": {
            "api": DIAGNOSTIC_API,
            "diagnostic_call_count": 1,
            "seed": SEED,
            "threads": THREADS,
            "m": M, "n": N, "k": K, "flops": FLOPS,
            "left_page_offset": int(left.ctypes.data) % page_size,
            "left_order": "F", "right_order": "F", "output_order": "F",
            "operands_before_and_after_equal": True,
            "operands": before,
            "output": output_record,
        },
        "repair": {
            "repaired_columns": int(repaired_count),
            "classification": compact["classification"],
            "all_repairs_classified": compact["all_repairs_classified"],
            "safe_diagnostic_classification": compact[
                "safe_diagnostic_classification"
            ],
        },
        "telemetry": {
            "records": telemetry,
            "status_after_consume": telemetry_status,
            "gate": telemetry_gate,
        },
        "correctness": {"dense_numpy_oracle": dense_oracle},
        "diagnostic": compact,
        "resource_usage": {
            "ru_maxrss_kib": int(usage.ru_maxrss),
            "minor_faults": int(usage.ru_minflt),
            "major_faults": int(usage.ru_majflt),
        },
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
        raise RuntimeError(f"worker emitted {len(records)} result records, expected one")
    return json.loads(records[0][len(RESULT_PREFIX) :])


def _worker_result_evidence_consistent(result: Mapping[str, Any]) -> bool:
    repair = dict(result.get("repair", {}))
    diagnostic = dict(result.get("diagnostic", {}))
    classification = repair.get("classification")
    if classification not in SAFE_CLASSIFICATIONS | REJECTED_CLASSIFICATIONS:
        return False
    if diagnostic.get("classification") != classification:
        return False
    if type(repair.get("repaired_columns")) is not int:
        return False
    if repair["repaired_columns"] != diagnostic.get("flagged_column_count"):
        return False
    contract_valid = diagnostic.get("classification_contract_valid")
    if type(contract_valid) is not bool:
        return False
    expected_all_classified = classification not in {
        "unclassified", "unclassified_detail_cap_exceeded"
    }
    expected_safe = classification in SAFE_CLASSIFICATIONS and contract_valid
    if repair.get("all_repairs_classified") is not expected_all_classified:
        return False
    if diagnostic.get("all_repairs_classified") is not expected_all_classified:
        return False
    if repair.get("safe_diagnostic_classification") is not expected_safe:
        return False
    if diagnostic.get("safe_diagnostic_classification") is not expected_safe:
        return False
    captured = diagnostic.get("captured_flagged_column_count")
    if type(captured) is not int or not 0 <= captured <= repair["repaired_columns"]:
        return False
    if diagnostic.get("raw_numeric_columns_persisted") is not (captured > 0):
        return False
    return True


def _run_one_worker(
    args: argparse.Namespace, worker_index: int, history_mode: str,
    dependencies: Sequence[Path], cpus: Sequence[int], timeout_seconds: float,
) -> dict[str, Any]:
    command = _taskset_worker_command(
        args, worker_index, history_mode, dependencies, cpus
    )
    environment, _settings = _worker_environment(
        _expected_backend(args),
        cpus if _placement_requested(args) else None,
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_process_group(process)
        return {
            "status": "timeout", "worker_index": worker_index,
            "history_mode": history_mode, "pid": process.pid,
            "timeout_seconds": timeout_seconds,
            "controller_wall_seconds": time.monotonic() - started,
            "command": command, "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
        }
    base = {
        "worker_index": worker_index, "history_mode": history_mode,
        "pid": process.pid, "returncode": process.returncode,
        "controller_wall_seconds": time.monotonic() - started,
        "command": command, "stderr_tail": stderr[-4000:],
    }
    if process.returncode != 0:
        return {**base, "status": "failed", "stdout_tail": stdout[-4000:]}
    try:
        result = _parse_worker_result(stdout)
    except (RuntimeError, json.JSONDecodeError) as error:
        return {
            **base, "status": "invalid_result", "reason": str(error),
            "stdout_tail": stdout[-4000:],
        }
    if result.get("worker_index") != worker_index or result.get("pid") != process.pid:
        return {**base, "status": "invalid_result", "reason": "worker identity mismatch"}
    if result.get("history_mode") != history_mode:
        return {**base, "status": "invalid_result", "reason": "history mode mismatch"}
    runtime = dict(result.get("runtime_attestation", {}))
    native = dict(runtime.get("native", {}))
    worker_affinity = dict(runtime.get("affinity", {}))
    build_hash = str(native.get("build_info_canonical_json_sha256", ""))
    runtime_mismatches = []
    if runtime.get("python_no_site") is not True:
        runtime_mismatches.append("python_no_site")
    if Path(str(runtime.get("python_executable", ""))).resolve() != args.python_executable.resolve():
        runtime_mismatches.append("python_executable")
    if native.get("module_path") != str(args.native_module.resolve()):
        runtime_mismatches.append("native_module_path")
    if native.get("module_sha256") != args.expected_native_sha256:
        runtime_mismatches.append("native_module_sha256")
    if not _canonical_sha256(build_hash):
        runtime_mismatches.append("native_build_info_sha256")
    if worker_affinity.get("allowed_cpus") != list(cpus):
        runtime_mismatches.append("worker_affinity")
    if _placement_requested(args):
        try:
            placement = _validate_openmp_placement_attestation(
                runtime.get("openmp_placement", {}), cpus
            )
            calling_affinity = dict(
                runtime.get("post_configuration_calling_thread_affinity", {})
            )
            if calling_affinity.get("allowed_cpus") != [cpus[0]]:
                raise RuntimeError(
                    "calling-thread affinity differs from singleton place zero"
                )
            build_info = dict(native.get("build_info", {}))
            build_evidence = build_info.get(
                "openmp_placement_contract_evidence"
            )
            if not isinstance(build_evidence, Mapping):
                raise RuntimeError("worker build-info placement evidence is absent")
            _validate_openmp_placement_attestation(build_evidence, cpus)
            if _canonical_json_sha256(build_evidence) != _canonical_json_sha256(
                placement
            ):
                raise RuntimeError(
                    "worker build-info placement evidence differs from attestation"
                )
        except (RuntimeError, TypeError, ValueError) as error:
            runtime_mismatches.append(f"openmp_placement: {error}")
    if runtime_mismatches:
        return {
            **base,
            "status": "invalid_result",
            "reason": "runtime attestation mismatch: " + ", ".join(runtime_mismatches),
        }
    if not _worker_result_evidence_consistent(result):
        return {
            **base,
            "status": "invalid_result",
            "reason": "worker repair/diagnostic evidence mappings disagree",
        }
    return {**base, **result}


def _schedule(workers: int, history_modes: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"worker_index": index, "history_mode": history_modes[index % len(history_modes)]}
        for index in range(workers)
    ]


def _static_provenance(
    args: argparse.Namespace, cpus: Sequence[int], dependencies: Sequence[Path],
) -> dict[str, Any]:
    prefix = args.install_prefix.resolve()
    module = args.native_module.resolve()
    python = _regular_executable(args.python_executable)
    taskset_path = shutil.which("taskset")
    if taskset_path is None:
        raise RuntimeError("taskset is required")
    taskset = _regular_executable(Path(taskset_path))
    script = Path(__file__).resolve()
    provenance = {
        "installed_package": _package_identity(prefix),
        "native_module": _file_identity(module),
        "expected_native_sha256": args.expected_native_sha256,
        "linkage": _linkage_evidence(module),
        "runner": _file_identity(script),
        "python_executable": _file_identity(python),
        "taskset_executable": _file_identity(taskset),
        "dependency_paths": [str(path) for path in dependencies],
        "selected_cpus": list(cpus),
        "selected_physical_core_keys": [list(_physical_core_key(cpu)) for cpu in cpus],
        "selected_numa_nodes": sorted(
            {node for cpu in cpus if (node := _cpu_numa_node(cpu)) is not None}
        ),
    }
    if _expected_backend(args) == "openblas":
        provenance["expected_private_openblas_archive_sha256"] = (
            args.expected_archive_sha256
        )
    else:
        provenance.update(
            {
                "expected_private_blas_backend": "upstream_blis",
                "expected_private_blas_archive_sha256": (
                    args.expected_archive_sha256
                ),
            }
        )
    if _placement_requested(args):
        provenance.update(
            {
                "expected_native_api_version": 9,
                "expected_native_backend_version": "1.9",
                "expected_openmp_placement_schema": OPENMP_PLACEMENT_SCHEMA,
            }
        )
    return provenance


def _base_report(
    args: argparse.Namespace, cpus: Sequence[int], dependencies: Sequence[Path],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    environment, thread_settings = _worker_environment(
        _expected_backend(args),
        cpus if _placement_requested(args) else None,
    )
    del environment
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "diagnostic_only": True,
        "production_acceptance_eligible": False,
        "exact_case": {
            "seed": SEED, "m": M, "n": N, "k": K, "threads": THREADS,
            "flops": FLOPS, "left_layout": "Fortran",
            "right_layout": "Fortran", "left_page_offset": LEFT_PAGE_OFFSET,
        },
        "limits": {
            "requested_workers": args.workers,
            "maximum_workers": MAX_WORKERS,
            "worker_timeout_seconds": args.worker_timeout_seconds,
            "maximum_worker_timeout_seconds": MAX_WORKER_TIMEOUT_SECONDS,
            "controller_timeout_seconds": args.controller_timeout_seconds,
            "maximum_controller_timeout_seconds": MAX_CONTROLLER_TIMEOUT_SECONDS,
            "controller_deadline_covers_preflight_execution_and_publication": True,
            "children_sequential": True,
            "fresh_python_no_site_per_worker": True,
            "exact_diagnostic_calls_per_worker": 1,
            "atomic_no_replace_publication": True,
        },
        "history_schedule": _schedule(args.workers, args.history_modes),
        "worker_thread_environment": thread_settings,
        "placement": {
            "taskset_required": True,
            "selected_cpus": list(cpus),
            "physical_core_keys": [list(_physical_core_key(cpu)) for cpu in cpus],
            "smt_siblings_excluded": True,
            **(
                {
                    "native_openmp_placement_required": True,
                    "native_openmp_placement_schema": OPENMP_PLACEMENT_SCHEMA,
                    "ordered_singleton_places": [[cpu] for cpu in cpus],
                    "omp_proc_bind": "spread",
                }
                if _placement_requested(args) else {}
            ),
        },
        "provenance": dict(provenance),
        "workers": [],
    }


def _dry_run_report(
    args: argparse.Namespace, cpus: Sequence[int], dependencies: Sequence[Path],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    report = _base_report(args, cpus, dependencies, provenance)
    report.update(
        {
            "dry_run": True,
            "status": "validated_dry_run",
            "accepted": False,
            "acceptance_eligible": False,
            "acceptance_reason": "dry_run_performs_no_scientific_execution",
            "scientific_case_executed": False,
            "planned_worker_command_template": _taskset_worker_command(
                args, 0, args.history_modes[0], dependencies, cpus
            ),
        }
    )
    return report


def _controller(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    cpus = _resolve_cpus(args.cpus)
    dependencies = _dependency_paths(args)
    provenance = _static_provenance(args, cpus, dependencies)
    report = _base_report(args, cpus, dependencies, provenance)
    report["dry_run"] = False
    initial_package = provenance["installed_package"]
    schedule = _schedule(args.workers, args.history_modes)
    for item in schedule:
        remaining = args.controller_timeout_seconds - (time.monotonic() - started)
        if remaining <= 0.25:
            report["workers"].append(
                {**item, "status": "not_started_controller_budget_exhausted"}
            )
            continue
        report["workers"].append(
            _run_one_worker(
                args, item["worker_index"], item["history_mode"], dependencies,
                cpus, min(args.worker_timeout_seconds, max(0.1, remaining - 0.1)),
            )
        )
        if _sha256(args.native_module.resolve()) != args.expected_native_sha256:
            raise RuntimeError("native module changed between workers")
    package_stable = _package_identity(args.install_prefix.resolve()) == initial_package
    runner_stable = _file_identity(Path(__file__).resolve()) == provenance["runner"]
    controller_seconds = time.monotonic() - started
    execution_complete = (
        len(report["workers"]) == args.workers
        and all(item.get("status") == "ok" for item in report["workers"])
    )
    classifications = [
        str(item.get("repair", {}).get("classification", "unclassified"))
        for item in report["workers"] if item.get("status") == "ok"
    ]
    all_build_hashes = {
        item.get("runtime_attestation", {}).get("native", {}).get(
            "build_info_canonical_json_sha256"
        )
        for item in report["workers"] if item.get("status") == "ok"
    }
    build_provenance_consistent = (
        execution_complete
        and len(all_build_hashes) == 1
        and all(_canonical_sha256(str(value)) for value in all_build_hashes)
    )
    worker_result_schema_consistent = execution_complete and all(
        _worker_result_evidence_consistent(item) for item in report["workers"]
    )
    numpy_attestations = []
    for item in report["workers"]:
        if item.get("status") != "ok":
            continue
        runtime = dict(item.get("runtime_attestation", {}))
        numpy_module = dict(runtime.get("numpy_module", {}))
        numpy_attestations.append(
            {
                "numpy_version": runtime.get("numpy_version"),
                "numpy_module_path": numpy_module.get("path"),
                "numpy_module_sha256": numpy_module.get("sha256"),
                "python_executable": runtime.get("python_executable"),
            }
        )
    numpy_provenance_consistent = (
        execution_complete
        and len(numpy_attestations) == args.workers
        and all(type(item["numpy_version"]) is str for item in numpy_attestations)
        and all(
            type(item["numpy_module_path"]) is str
            and _canonical_sha256(str(item["numpy_module_sha256"]))
            and type(item["python_executable"]) is str
            for item in numpy_attestations
        )
        and len({_canonical_json_sha256(item) for item in numpy_attestations}) == 1
    )
    protocol_complete = (
        execution_complete
        and package_stable
        and runner_stable
        and build_provenance_consistent
        and worker_result_schema_consistent
        and numpy_provenance_consistent
        and controller_seconds <= args.controller_timeout_seconds
    )
    telemetry_gate_passed = execution_complete and all(
        item.get("telemetry", {}).get("gate", {}).get("passed") is True
        for item in report["workers"]
    )
    dense_oracle_gate_passed = execution_complete and all(
        item.get("correctness", {}).get("dense_numpy_oracle", {}).get("passed")
        is True
        for item in report["workers"]
    )
    diagnostic_contracts_valid = execution_complete and all(
        item.get("diagnostic", {}).get("classification_contract_valid") is True
        for item in report["workers"]
    )
    unclassified_repairs = sum(
        int(item.get("repair", {}).get("repaired_columns", 0))
        for item in report["workers"]
        if item.get("status") == "ok"
        and item.get("repair", {}).get("classification") in {
            "unclassified", "unclassified_detail_cap_exceeded"
        }
    )
    rejected_events = [value for value in classifications if value in REJECTED_CLASSIFICATIONS]
    diagnostic_gate_passed = (
        protocol_complete
        and telemetry_gate_passed
        and dense_oracle_gate_passed
        and diagnostic_contracts_valid
        and unclassified_repairs == 0 and not rejected_events
        and all(value in SAFE_CLASSIFICATIONS for value in classifications)
    )
    counts = {
        value: classifications.count(value) for value in sorted(set(classifications))
    }
    repairs = sum(
        int(item.get("repair", {}).get("repaired_columns", 0))
        for item in report["workers"] if item.get("status") == "ok"
    )
    event_reproduced = repairs > 0
    event_classified = (
        execution_complete and event_reproduced
        and all(
            item.get("diagnostic", {}).get("classification_contract_valid") is True
            and item.get("repair", {}).get("classification")
            not in {"unclassified", "unclassified_detail_cap_exceeded"}
            for item in report["workers"]
            if int(item.get("repair", {}).get("repaired_columns", 0)) > 0
        )
    )
    if diagnostic_gate_passed and repairs == 0:
        status = "completed_no_repair_observed"
        conclusion = "rare repair event was not reproduced; root cause remains unresolved"
    elif diagnostic_gate_passed:
        status = "completed_checksum_false_positive_classification"
        conclusion = "every repair was classified as a cancellation-sensitive checksum false positive"
    elif rejected_events:
        status = "completed_rejected_integrity_event"
        conclusion = "at least one worker found vendor damage, changed inputs, or an unclassified event"
    elif not dense_oracle_gate_passed and execution_complete:
        status = "completed_rejected_dense_oracle_gate"
        conclusion = "at least one returned output failed the independent dense NumPy oracle"
    elif not telemetry_gate_passed and execution_complete:
        status = "completed_rejected_telemetry_gate"
        conclusion = "at least one vendor record failed utilization or NUMA-locality gates"
    elif not diagnostic_contracts_valid and execution_complete:
        status = "completed_rejected_diagnostic_contract"
        conclusion = "at least one native classification failed its evidence contract"
    else:
        status = "incomplete"
        conclusion = "the bounded fresh-process protocol did not complete all gates"
    report.update(
        {
            "status": status,
            "accepted": False,
            "protocol_complete": protocol_complete,
            "diagnostic_gate_passed": diagnostic_gate_passed,
            "scientific_case_executed": True,
            "execution_complete": execution_complete,
            "controller_wall_seconds": controller_seconds,
            "installed_package_stable": package_stable,
            "runner_identity_stable": runner_stable,
            "worker_build_provenance_consistent": build_provenance_consistent,
            "worker_result_schema_consistent": worker_result_schema_consistent,
            "numpy_provenance_consistent": numpy_provenance_consistent,
            "numpy_runtime_attestation": (
                numpy_attestations[0] if numpy_provenance_consistent else None
            ),
            "telemetry_gate_passed": telemetry_gate_passed,
            "dense_oracle_gate_passed": dense_oracle_gate_passed,
            "diagnostic_classification_contracts_valid": (
                diagnostic_contracts_valid
            ),
            "repairs_observed": repairs,
            "unclassified_repairs": unclassified_repairs,
            "classification_counts": counts,
            "rejected_event_classifications": rejected_events,
            "diagnostic_conclusion": conclusion,
            "event_reproduced": event_reproduced,
            "event_classified": event_classified,
            "safe_diagnostic_classification": diagnostic_gate_passed,
            "root_cause_evidence_complete": bool(
                event_classified
                and protocol_complete
                and telemetry_gate_passed
                and dense_oracle_gate_passed
                and diagnostic_contracts_valid
            ),
            "root_cause_resolved": bool(
                event_classified
                and protocol_complete
                and telemetry_gate_passed
                and dense_oracle_gate_passed
                and diagnostic_contracts_valid
            ),
        }
    )
    return report


def _atomic_json_no_replace(payload: Mapping[str, Any], output: Path) -> None:
    lexical_target = output.expanduser().absolute()
    if lexical_target.is_symlink() or lexical_target.exists():
        raise FileExistsError(f"refusing existing report: {lexical_target}")
    parent = lexical_target.parent.resolve(strict=True)
    target = parent / lexical_target.name
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"refusing existing report: {target}")
    rendered = json.dumps(
        _json_safe(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, target)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args._worker:
        try:
            payload = _run_worker(args)
            print(RESULT_PREFIX + json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False))
            return 0
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        parser.error("a POSIX real-time controller deadline is required")
    previous_handler = signal.signal(
        signal.SIGALRM, _raise_controller_deadline
    )
    previous_timer = signal.setitimer(
        signal.ITIMER_REAL, args.controller_timeout_seconds
    )
    try:
        if args.dry_run:
            cpus = _resolve_cpus(args.cpus)
            dependencies = _dependency_paths(args)
            provenance = _static_provenance(args, cpus, dependencies)
            payload = _dry_run_report(args, cpus, dependencies, provenance)
        else:
            payload = _controller(args)
        _atomic_json_no_replace(payload, args.output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(
                signal.ITIMER_REAL, previous_timer[0], previous_timer[1]
            )
    print(args.output.expanduser().resolve())
    return 0 if args.dry_run or payload.get("diagnostic_gate_passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
