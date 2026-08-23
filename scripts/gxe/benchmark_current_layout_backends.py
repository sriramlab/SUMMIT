#!/usr/bin/env python3
"""Benchmark the exact current SUMMIT source/target GEMMs fail-closed.

This evidence harness runs exactly two current-layout FP64 cases: source NN and
target TN.  Each case uses a fresh ``python -S`` process, a private static BLAS,
strict early NUMA membind, and the API-9 OpenMP singleton-place contract.  It is
not a production reference runner and never marks a dry run as accepted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


SCHEMA = "summit.gxe.current_layout_backend_benchmark"
SCHEMA_VERSION = 1
RESULT_PREFIX = "SUMMIT_GXE_CURRENT_LAYOUT_RESULT="
PLACEMENT_SCHEMA = "summit.openmp_placement_attestation.v1"
NUMA_SCHEMA = "summit.numa_policy_attestation.v1"
NUMA_ADDRESS_SELECTION_POLICY = "evenly_spaced_fully_contained_page_bases"
NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA = (
    "summit.native_integrity_snapshot_numa.v1"
)
NATIVE_GEMM_OUTPUT_NUMA_SCHEMA = "summit.native_gemm_output_numa.v1"
NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT = 65_536
NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY = 16_384
NATIVE_STATIC_MEMBIND_POLICY_VALUE = 32_770
EXPECTED_API_VERSION = 9
EXPECTED_BACKEND_VERSION = "1.9"
EXPECTED_BACKENDS = ("openblas", "blis")
DEFAULT_N = 289_111
DEFAULT_BLOCK_WIDTH = 2_000
DEFAULT_PROBE_TILE = 32
DEFAULT_ENVIRONMENT_TILE = 3
DEFAULT_THREADS = 32
WARMUPS = 3
MEASURED_REPEATS = 5
MAX_CASE_SECONDS = 90.0
MAX_SWEEP_SECONDS = 20.0 * 60.0
MINIMUM_ACTIVE_CORE_FRACTION = 0.75
VENDOR_WORKSPACE_ALLOWANCE_BYTES = 16 * 1024**3
GEMM_INTEGRITY_CHECKS = 8
GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS = 1_000_000_000
BLIS_WAY_NAMES = ("jc", "pc", "ic", "jr", "ir")
BLIS_WAY_ENV = {
    "jc": "BLIS_JC_NT",
    "pc": "BLIS_PC_NT",
    "ic": "BLIS_IC_NT",
    "jr": "BLIS_JR_NT",
    "ir": "BLIS_IR_NT",
}
BLIS_ALIASES = ("BLIS_NT", "BLIS_TI")
BLIS_UNCONTROLLED_OVERRIDES = (
    "BLIS_ARCH_TYPE",
    "BLIS_ARCH_DEBUG",
    "BLIS_PACK_A",
    "BLIS_PACK_B",
)
SAFE_ENV_PASSTHROUGH = (
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PATH",
    "TMPDIR",
    "TZ",
)


@dataclass(frozen=True)
class Case:
    operation: str
    n_samples: int
    block_width: int
    probe_tile: int
    environment_tile: int
    threads: int

    @property
    def panel_columns(self) -> int:
        multiplier = 2 if self.operation == "source" else 4
        return multiplier * self.probe_tile * self.environment_tile

    @property
    def cblas(self) -> dict[str, Any]:
        if self.operation == "source":
            return {
                "operation": "dgemm_nn",
                "layout": "column_major",
                "transpose_a": "N",
                "transpose_b": "N",
                "m": self.n_samples,
                "n": self.panel_columns,
                "k": self.block_width,
                "lda": self.n_samples,
                "ldb": self.block_width,
                "ldc": self.n_samples,
            }
        return {
            "operation": "dgemm_tn",
            "layout": "column_major",
            "transpose_a": "T",
            "transpose_b": "N",
            "m": self.block_width,
            "n": self.panel_columns,
            "k": self.n_samples,
            "lda": self.n_samples,
            "ldb": self.n_samples,
            "ldc": self.block_width,
        }

    @property
    def output_shape(self) -> tuple[int, int]:
        metadata = self.cblas
        return int(metadata["m"]), int(metadata["n"])

    @property
    def flops(self) -> int:
        metadata = self.cblas
        return 2 * int(metadata["m"]) * int(metadata["n"]) * int(metadata["k"])

    @property
    def case_id(self) -> str:
        return (
            f"current_{self.operation}.N{self.n_samples}.K{self.block_width}"
            f".B{self.probe_tile}.L{self.environment_tile}.T{self.threads}"
        )


class _SweepDeadlineExpired(TimeoutError):
    pass


def _raise_sweep_deadline(_signum: int, _frame: Any) -> None:
    raise _SweepDeadlineExpired("backend benchmark exceeded the 20-minute sweep cap")


def _canonical_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _canonical_commit(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return "+inf" if value > 0 else "-inf"
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    return value


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    selected: list[int] = []
    try:
        for component in value.split(","):
            part = component.strip()
            if not part:
                continue
            if "-" in part:
                first_text, last_text = part.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first < 0 or last < first:
                    raise ValueError
                selected.extend(range(first, last + 1))
            else:
                cpu = int(part)
                if cpu < 0:
                    raise ValueError
                selected.append(cpu)
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid CPU list") from error
    if not selected or len(selected) != len(set(selected)):
        raise argparse.ArgumentTypeError("CPU list must be nonempty and unique")
    return tuple(selected)


def _compress_ints(values: Sequence[int]) -> str:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return ""
    rendered: list[str] = []
    first = last = ordered[0]
    for value in ordered[1:]:
        if value == last + 1:
            last = value
            continue
        rendered.append(str(first) if first == last else f"{first}-{last}")
        first = last = value
    rendered.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(rendered)


def _parse_blis_ways(value: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    try:
        for component in value.split(","):
            name, raw = component.strip().split("=", 1)
            if name in parsed:
                raise ValueError
            parsed[name] = int(raw)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "BLIS ways must be jc=N,pc=N,ic=N,jr=N,ir=N"
        ) from error
    if set(parsed) != set(BLIS_WAY_NAMES) or any(
        value <= 0 for value in parsed.values()
    ):
        raise argparse.ArgumentTypeError(
            "BLIS ways must contain exactly positive jc,pc,ic,jr,ir values"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument("--native-module", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--private-archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-backend", choices=EXPECTED_BACKENDS, required=True)
    parser.add_argument("--expected-private-source-commit")
    parser.add_argument("--expected-private-source-tree-sha256")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--dependency-path", type=Path, action="append", default=None)
    parser.add_argument("--cpus", type=_parse_cpu_list, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=argparse.SUPPRESS)
    parser.add_argument(
        "--block-width",
        type=int,
        default=DEFAULT_BLOCK_WIDTH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-tile",
        type=int,
        default=DEFAULT_PROBE_TILE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--environment-tile",
        type=int,
        default=DEFAULT_ENVIRONMENT_TILE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--blis-thread-strategy",
        choices=("automatic", "manual"),
        default="automatic",
    )
    parser.add_argument("--blis-thread-ways", type=_parse_blis_ways)
    parser.add_argument("--case-timeout-seconds", type=float, default=MAX_CASE_SECONDS)
    parser.add_argument(
        "--sweep-timeout-seconds", type=float, default=MAX_SWEEP_SECONDS
    )
    parser.add_argument("--max-memory-gib", type=float, default=64.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_operation", choices=("source", "target"), help=argparse.SUPPRESS
    )
    return parser


def _regular_file(path: Path, *, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    mode = resolved.stat().st_mode
    if not resolved.is_file() or not stat.S_ISREG(mode):
        raise ValueError(f"not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"not executable: {resolved}")
    return resolved


def _dependency_paths(args: argparse.Namespace) -> list[Path]:
    raw = args.dependency_path
    if raw is None:
        raw = [
            Path(value)
            for value in (sysconfig.get_path("platlib"), sysconfig.get_path("purelib"))
            if value
        ]
    result: list[Path] = []
    for path in raw:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"dependency path is not a directory: {resolved}")
        if resolved not in result:
            result.append(resolved)
    if not result:
        raise ValueError("at least one dependency path is required for python -S")
    return result


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _cpu_numa_node(cpu: int) -> int | None:
    try:
        nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    except OSError:
        return None
    if len(nodes) != 1:
        return None
    try:
        return int(nodes[0].name.removeprefix("node"))
    except ValueError:
        return None


def _cpu_contract(cpus: Sequence[int]) -> dict[str, Any]:
    try:
        allowed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        raise RuntimeError("sched_getaffinity is required") from error
    records: list[dict[str, int]] = []
    physical: set[tuple[int, int]] = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = _read_int(topology / "physical_package_id")
        core = _read_int(topology / "core_id")
        node = _cpu_numa_node(cpu)
        if cpu not in allowed or package is None or core is None or node is None:
            raise RuntimeError(f"CPU {cpu} lacks an allowed exact topology/NUMA record")
        key = (package, core)
        if key in physical:
            raise RuntimeError("selected CPUs include SMT siblings")
        physical.add(key)
        records.append(
            {"cpu": cpu, "package": package, "core": core, "numa_node": node}
        )
    return {
        "ordered_cpu_ids": list(cpus),
        "cpu_list": _compress_ints(cpus),
        "records": records,
        "numa_nodes": sorted({record["numa_node"] for record in records}),
        "one_hardware_thread_per_physical_core": True,
        "within_controller_affinity": True,
    }


def _package_identity(prefix: Path) -> dict[str, Any]:
    package = prefix.resolve() / "summit"
    if not package.is_dir():
        raise RuntimeError(f"installed prefix lacks summit package: {prefix}")
    records: list[tuple[str, int, str]] = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            raise RuntimeError(f"installed package contains bytecode/cache: {relative}")
        if path.is_symlink():
            raise RuntimeError(
                f"installed package contains a symbolic link: {relative}"
            )
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise RuntimeError(
                f"cannot stat installed package entry: {relative}"
            ) from error
        if stat.S_ISREG(mode):
            records.append(
                (str(path.relative_to(prefix)), path.stat().st_size, _sha256(path))
            )
        elif not stat.S_ISDIR(mode):
            raise RuntimeError(
                f"installed package contains a nonregular entry: {relative}"
            )
    if not records:
        raise RuntimeError("installed summit package is empty")
    digest = hashlib.sha256()
    for relative, size, file_hash in records:
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode("utf-8"))
    return {
        "install_prefix": str(prefix.resolve()),
        "file_count": len(records),
        "bytes": sum(size for _, size, _ in records),
        "manifest_sha256": digest.hexdigest(),
        "bytecode_and_cache_absent": True,
        "symbolic_links_absent": True,
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


def _run_checked(command: Sequence[str], timeout: float = 20.0) -> str:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    ).stdout


_BLAS_ROUTINE = re.compile(
    r"^(?:cblas_.*|openblas_.*|bli_.*|goto_.*|gotoblas.*|blas_.*|"
    r"xerbla(?:_.*|$)|"
    r"lsame(?:_|$)|"
    r"i[sdcz]amax(?:_.*|64_)?|"
    r"(?:dsdot|scasum|dzasum|scnrm2|dznrm2|csscal|zdscal)(?:_.*|64_)?|"
    r"[sdcz](?:asum|axpy|copy|dot|dotc|dotu|dsdot|sdsdot|gbmv|gemm|gemv|"
    r"ger|gerc|geru|hbmv|hemm|hemv|her|her2|her2k|herk|hpmv|hpr|hpr2|"
    r"nrm2|rot|rotg|rotm|rotmg|sbmv|scal|spmv|spr|spr2|swap|symm|symv|"
    r"syr|syr2|syr2k|syrk|tbmv|tbsv|tpmv|tpsv|trmm|trmv|trsm|trsv)"
    r"(?:_.*|64_)?$)",
    re.IGNORECASE,
)


def _nm_symbol(line: str) -> str | None:
    fields = line.split()
    if not fields:
        return None
    candidate = fields[-1].split("@", 1)[0]
    return candidate if _BLAS_ROUTINE.fullmatch(candidate) else None


def _linkage_evidence(
    module: Path, private_archive: Path, expected_backend: str
) -> dict[str, Any]:
    from shutil import which

    readelf, nm = which("readelf"), which("nm")
    if readelf is None or nm is None:
        raise RuntimeError("readelf and nm are required for exact linkage verification")
    dynamic = _run_checked([readelf, "-d", str(module)])
    dynamic_symbols = _run_checked([nm, "-D", str(module)])
    undefined = _run_checked([nm, "-u", str(module)])
    # Release modules are intentionally stripped and hide archive symbols with
    # --exclude-libs,ALL. Inventory required capabilities from the exact
    # hash-pinned static archive, while the module checks below independently
    # reject dynamic dependencies, exports, and unresolved BLAS entry points.
    defined_globals = _run_checked(
        [nm, "-g", "--defined-only", str(private_archive)]
    )
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    dynamic_blas = [
        name
        for name in needed
        if "blas" in Path(name).name.lower()
        or Path(name).name.lower().startswith("libblis")
    ]
    exported_or_dynamic = sorted(
        symbol
        for line in dynamic_symbols.splitlines()
        if (symbol := _nm_symbol(line)) is not None
    )
    unresolved = sorted(
        symbol
        for line in undefined.splitlines()
        if (symbol := _nm_symbol(line)) is not None
    )
    globals_found = sorted(
        symbol
        for line in defined_globals.splitlines()
        if (symbol := _nm_symbol(line)) is not None
    )
    required = {"cblas_dgemm"}
    if expected_backend == "openblas":
        required.add("openblas_get_config")
    else:
        required.update(
            {
                "cblas_sgemm",
                "bli_info_get_version_str",
                "bli_info_get_enable_tls",
            }
        )
    missing = sorted(required.difference(globals_found))
    if dynamic_blas:
        raise RuntimeError(f"extension has a dynamic BLAS dependency: {dynamic_blas}")
    if exported_or_dynamic:
        raise RuntimeError(
            "private BLAS globals are exported or unresolved dynamically: "
            f"{exported_or_dynamic[:8]}"
        )
    if unresolved:
        raise RuntimeError(f"extension has unresolved BLAS globals: {unresolved[:8]}")
    if missing:
        raise RuntimeError(f"extension lacks required private BLAS globals: {missing}")
    return {
        "readelf_executable": _file_identity(Path(readelf)),
        "nm_executable": _file_identity(Path(nm)),
        "needed_shared_libraries": needed,
        "dynamic_blas_dependencies": [],
        "dynamic_blas_global_count": 0,
        "unresolved_blas_global_count": 0,
        "defined_private_blas_global_count": len(globals_found),
        "defined_private_blas_globals": globals_found,
        "defined_private_blas_globals_sha256": _canonical_json_sha256(globals_found),
        "required_private_blas_globals": sorted(required),
        "defined_private_blas_inventory_source": "hash_pinned_static_archive",
        "private_archive": _file_identity(private_archive),
        "readelf_dynamic_sha256": hashlib.sha256(dynamic.encode()).hexdigest(),
        "dynamic_symbols_sha256": hashlib.sha256(dynamic_symbols.encode()).hexdigest(),
        "undefined_symbols_sha256": hashlib.sha256(undefined.encode()).hexdigest(),
        "defined_global_symbols_sha256": hashlib.sha256(
            defined_globals.encode()
        ).hexdigest(),
        "private_static_linkage_verified": True,
    }


def _cases(args: argparse.Namespace) -> list[Case]:
    return [
        Case(
            operation=operation,
            n_samples=args.n,
            block_width=args.block_width,
            probe_tile=args.probe_tile,
            environment_tile=args.environment_tile,
            threads=args.threads,
        )
        for operation in ("source", "target")
    ]


def _estimated_memory(case: Case) -> dict[str, Any]:
    n, k, p = case.n_samples, case.block_width, case.panel_columns
    if case.operation == "source":
        left_bytes = 8 * n * k
        right_bytes = 8 * k * p
    else:
        left_bytes = 8 * n * k
        right_bytes = 8 * n * p
    input_bytes = left_bytes + right_bytes
    output_bytes = 8 * case.output_shape[0] * case.output_shape[1]
    metadata = case.cblas
    integrity_right_snapshot = right_bytes
    integrity_checksum_and_repair_scratch = (
        8
        * GEMM_INTEGRITY_CHECKS
        * (int(metadata["m"]) + 2 * int(metadata["k"]) + 2 * int(metadata["n"]))
    )
    # Baseline and current/oracle outputs can coexist. Norm vectors and bounded
    # column comparison scratch are included explicitly; no full abs(A) copy.
    oracle_and_determinism = 2 * output_bytes
    norm_and_chunk_scratch = 8 * (case.output_shape[0] + p) + 64 * 1024**2
    diagnostic_before_vendor = (
        input_bytes
        + integrity_right_snapshot
        + integrity_checksum_and_repair_scratch
        + oracle_and_determinism
        + norm_and_chunk_scratch
    )
    diagnostic_with_vendor = diagnostic_before_vendor + VENDOR_WORKSPACE_ALLOWANCE_BYTES
    conservative = math.ceil(1.20 * diagnostic_with_vendor)
    production_target_before_vendor = (
        left_bytes + right_bytes + output_bytes + integrity_checksum_and_repair_scratch
        if case.operation == "target"
        else None
    )
    production_target_conservative = (
        math.ceil(
            1.20 * (production_target_before_vendor + VENDOR_WORKSPACE_ALLOWANCE_BYTES)
        )
        if production_target_before_vendor is not None
        else None
    )
    return {
        "peak_model_scope": "direct_diagnostic_integrity_wrapper",
        "left_input_bytes": left_bytes,
        "right_input_bytes": right_bytes,
        "input_bytes": input_bytes,
        "one_output_bytes": output_bytes,
        "coexisting_output_bytes": oracle_and_determinism,
        "norm_and_chunk_scratch_bytes": norm_and_chunk_scratch,
        "integrity_check_count": GEMM_INTEGRITY_CHECKS,
        "integrity_right_snapshot_bytes": integrity_right_snapshot,
        "integrity_checksum_and_repair_scratch_bytes": (
            integrity_checksum_and_repair_scratch
        ),
        "diagnostic_wrapper_bytes_before_vendor_fallback": diagnostic_before_vendor,
        "uncertified_vendor_workspace_fallback_bytes": (
            VENDOR_WORKSPACE_ALLOWANCE_BYTES
        ),
        "diagnostic_wrapper_bytes_before_headroom": diagnostic_with_vendor,
        "headroom_fraction": 0.20,
        "conservative_peak_bytes": conservative,
        "conservative_peak_gib": conservative / 1024**3,
        "production_sealed_pair_target_peak": (
            {
                "scope": "production_immutable_sealed_pair_target_gemm",
                "full_production_phase_peak": False,
                "benchmark_oracle_and_determinism_bytes": 0,
                "per_call_integrity_right_snapshot_bytes": 0,
                "bytes_before_vendor_fallback": production_target_before_vendor,
                "uncertified_vendor_workspace_fallback_bytes": (
                    VENDOR_WORKSPACE_ALLOWANCE_BYTES
                ),
                "bytes_before_headroom": (
                    production_target_before_vendor + VENDOR_WORKSPACE_ALLOWANCE_BYTES
                ),
                "headroom_fraction": 0.20,
                "conservative_peak_bytes": production_target_conservative,
                "conservative_peak_gib": (production_target_conservative / 1024**3),
            }
            if case.operation == "target"
            else None
        ),
    }


def _validate_blis_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.expected_backend == "blis":
        if not _canonical_commit(args.expected_private_source_commit):
            parser.error(
                "BLIS requires --expected-private-source-commit (40 lowercase hex)"
            )
        if not _canonical_sha256(args.expected_private_source_tree_sha256):
            parser.error("BLIS requires --expected-private-source-tree-sha256")
        if args.blis_thread_strategy == "automatic":
            if args.blis_thread_ways is not None:
                parser.error("automatic BLIS must not receive explicit loop ways")
        else:
            ways = args.blis_thread_ways
            if ways is None:
                parser.error("manual BLIS requires --blis-thread-ways")
            if ways["pc"] != 1 or math.prod(ways.values()) != args.threads:
                parser.error("manual BLIS ways require pc=1 and product=--threads")
    else:
        if (
            args.blis_thread_strategy != "automatic"
            or args.blis_thread_ways is not None
        ):
            parser.error("BLIS thread controls are invalid for OpenBLAS")
        if (
            args.expected_private_source_commit is not None
            or args.expected_private_source_tree_sha256 is not None
        ):
            parser.error("private BLIS source options are invalid for OpenBLAS")


def _validate_output_target(parser: argparse.ArgumentParser, output: Path) -> Path:
    lexical = output.expanduser().absolute()
    if lexical.is_symlink() or lexical.exists():
        parser.error(f"refusing to overwrite existing output: {lexical}")
    try:
        parent = lexical.parent.resolve(strict=True)
    except FileNotFoundError:
        parser.error(f"output parent must already exist: {lexical.parent}")
    target = parent / lexical.name
    if not parent.is_dir() or target.is_symlink() or target.exists():
        parser.error(f"refusing invalid or existing output: {target}")
    return target


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "expected_native_sha256",
        "expected_package_manifest_sha256",
        "expected_archive_sha256",
        "expected_source_tree_sha256",
    ):
        if not _canonical_sha256(getattr(args, name)):
            parser.error(
                f"--{name.replace('_', '-')} must be lowercase canonical SHA256"
            )
    if not _canonical_commit(args.expected_source_commit):
        parser.error("--expected-source-commit must be 40 lowercase hex")
    if any(
        value <= 0
        for value in (
            args.n,
            args.block_width,
            args.probe_tile,
            args.environment_tile,
            args.threads,
        )
    ):
        parser.error("all shape and thread values must be positive")
    observed_shape = (args.n, args.block_width, args.probe_tile, args.environment_tile)
    required_shape = (
        DEFAULT_N,
        DEFAULT_BLOCK_WIDTH,
        DEFAULT_PROBE_TILE,
        DEFAULT_ENVIRONMENT_TILE,
    )
    if observed_shape != required_shape:
        parser.error(
            "this evidence harness requires exact current production "
            f"(N,K,B_tile,L_tile)={required_shape}"
        )
    if len(args.cpus) != args.threads:
        parser.error("--cpus must contain exactly --threads CPUs")
    if not 0 < args.case_timeout_seconds <= MAX_CASE_SECONDS:
        parser.error("--case-timeout-seconds must be in (0, 90]")
    if not 0 < args.sweep_timeout_seconds <= MAX_SWEEP_SECONDS:
        parser.error("--sweep-timeout-seconds must be in (0, 1200]")
    if args.case_timeout_seconds * 2 > args.sweep_timeout_seconds:
        parser.error("two case timeouts exceed the sweep timeout")
    if args.max_memory_gib <= 0:
        parser.error("--max-memory-gib must be positive")
    _validate_blis_args(parser, args)
    _validate_output_target(parser, args.output)
    try:
        prefix = args.install_prefix.expanduser().resolve(strict=True)
        module = _regular_file(args.native_module)
        archive = _regular_file(args.private_archive)
        _regular_file(args.python_executable, executable=True)
        _dependency_paths(args)
        cpu_contract = _cpu_contract(args.cpus)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if not prefix.is_dir():
        parser.error(f"install prefix is not a directory: {prefix}")
    try:
        relative = module.relative_to(prefix)
    except ValueError:
        parser.error("--native-module must resolve inside --install-prefix")
    if (
        relative.parent != Path("summit")
        or not module.name.startswith("gxeldcore.")
        or module.suffix != ".so"
    ):
        parser.error("--native-module must be installed summit/gxeldcore.*.so")
    if _sha256(module) != args.expected_native_sha256:
        parser.error("native module SHA256 differs from --expected-native-sha256")
    if _sha256(archive) != args.expected_archive_sha256:
        parser.error("private archive SHA256 differs from --expected-archive-sha256")
    package = _package_identity(prefix)
    if package["manifest_sha256"] != args.expected_package_manifest_sha256:
        parser.error(
            "installed package differs from --expected-package-manifest-sha256"
        )
    if not cpu_contract["numa_nodes"]:
        parser.error("selected CPUs have no NUMA nodes")


def _thread_environment(args: argparse.Namespace) -> dict[str, str]:
    threads = str(args.threads)
    places = ",".join(f"{{{cpu}}}" for cpu in args.cpus)
    settings = {
        "OMP_NUM_THREADS": threads,
        "OMP_THREAD_LIMIT": threads,
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "SPREAD",
        "OMP_PLACES": places,
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "GOMP_SPINCOUNT": "0",
        "OPENBLAS_NUM_THREADS": threads,
        "OPENBLAS_THREAD_TIMEOUT": "1",
        "GOTO_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "MKL_DYNAMIC": "FALSE",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    if args.expected_backend == "blis":
        settings["BLIS_NUM_THREADS"] = threads
        if args.blis_thread_strategy == "manual":
            assert args.blis_thread_ways is not None
            settings["BLIS_THREAD_IMPL"] = "openmp"
            for name, env_name in BLIS_WAY_ENV.items():
                settings[env_name] = str(args.blis_thread_ways[name])
    return settings


def _worker_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = {
        name: os.environ[name] for name in SAFE_ENV_PASSTHROUGH if name in os.environ
    }
    settings = _thread_environment(args)
    environment.update(settings)
    # No inherited credentials, Python path, preload, NUMA marker, or BLIS
    # aliases are copied. Automatic BLIS therefore starts without loop ways.
    return environment, settings


def _validate_process_environment(args: argparse.Namespace) -> dict[str, Any]:
    expected = _thread_environment(args)
    mismatches = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    forbidden = [
        "GOMP_CPU_AFFINITY", *BLIS_ALIASES, *BLIS_UNCONTROLLED_OVERRIDES,
    ]
    if args.expected_backend == "openblas":
        forbidden.extend(
            ["BLIS_NUM_THREADS", "BLIS_THREAD_IMPL", *BLIS_WAY_ENV.values()]
        )
    elif args.blis_thread_strategy == "automatic":
        forbidden.extend(["BLIS_THREAD_IMPL", *BLIS_WAY_ENV.values()])
    unexpected = sorted(name for name in forbidden if name in os.environ)
    if mismatches or unexpected:
        raise RuntimeError(
            f"fresh-process thread environment mismatch: {mismatches}; "
            f"unexpected={unexpected}"
        )
    return {
        "validated": True,
        "settings": expected,
        "forbidden_variables_absent": unexpected == [],
    }


def _bootstrap_paths(prefix: Path, dependencies: Sequence[Path]) -> list[str]:
    inserted: list[str] = []
    for path in reversed([prefix, *dependencies]):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
            inserted.append(value)
    return list(reversed(inserted))


def _validate_early_numa_attestation(
    value: Mapping[str, Any], nodes: Sequence[int]
) -> dict[str, Any]:
    required_keys = {
        "schema",
        "mode",
        "requested_nodes",
        "effective_nodes",
        "task_count_at_application",
        "applied_before_numeric_import",
        "verified",
        "source",
        "applied_policy",
        "static_nodes",
        "pid",
    }
    if set(value) != required_keys:
        raise RuntimeError("early NUMA attestation has an unexpected schema")
    expected_nodes = sorted(int(node) for node in nodes)
    expected = {
        "schema": NUMA_SCHEMA,
        "mode": "membind",
        "requested_nodes": _compress_ints(expected_nodes),
        "effective_nodes": expected_nodes,
        "task_count_at_application": 1,
        "applied_before_numeric_import": True,
        "verified": True,
        "source": "libnuma",
        "applied_policy": "libnuma:membind:" + ",".join(map(str, expected_nodes)),
        "static_nodes": True,
        "pid": os.getpid(),
    }
    mismatches = {
        name: {"expected": expected_value, "observed": value.get(name)}
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    strict_ints = ("task_count_at_application", "pid")
    strict_bools = ("applied_before_numeric_import", "verified", "static_nodes")
    if (
        mismatches
        or any(type(value.get(name)) is not int for name in strict_ints)
        or any(type(value.get(name)) is not bool for name in strict_bools)
        or not isinstance(value.get("effective_nodes"), list)
        or any(type(node) is not int for node in value.get("effective_nodes", []))
    ):
        raise RuntimeError(f"early NUMA attestation mismatch: {mismatches}")
    return dict(value)


def _validate_placement_attestation(
    value: Mapping[str, Any], cpus: Sequence[int], threads: int
) -> dict[str, Any]:
    required_keys = {
        "schema",
        "schema_version",
        "verified",
        "immutable",
        "requested_threads",
        "expected_cpu_ids",
        "omp_dynamic",
        "omp_thread_limit",
        "omp_max_active_levels",
        "omp_proc_bind",
        "omp_binding_active",
        "omp_num_places",
        "effective_openmp_capacity",
        "place_cpu_ids",
        "team_size",
        "exact_singleton_places",
        "exact_team_coverage",
        "workers",
        "vendor_calls",
    }
    if set(value) != required_keys:
        raise RuntimeError("OpenMP placement attestation has an unexpected schema")
    expected_cpus = list(cpus)
    exact = {
        "schema": PLACEMENT_SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": threads,
        "expected_cpu_ids": expected_cpus,
        "omp_dynamic": False,
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": threads,
        "place_cpu_ids": [[cpu] for cpu in expected_cpus],
        "team_size": threads,
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "vendor_calls": 0,
    }
    mismatches = {
        name: {"expected": expected, "observed": value.get(name)}
        for name, expected in exact.items()
        if value.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"OpenMP placement contract mismatch: {mismatches}")
    integer_fields = (
        "schema_version",
        "requested_threads",
        "omp_thread_limit",
        "omp_max_active_levels",
        "omp_num_places",
        "effective_openmp_capacity",
        "team_size",
        "vendor_calls",
    )
    boolean_fields = (
        "verified",
        "immutable",
        "omp_dynamic",
        "omp_binding_active",
        "exact_singleton_places",
        "exact_team_coverage",
    )
    if any(type(value.get(name)) is not int for name in integer_fields):
        raise RuntimeError("OpenMP placement integer fields are not exact integers")
    if any(type(value.get(name)) is not bool for name in boolean_fields):
        raise RuntimeError("OpenMP placement boolean fields are not exact booleans")
    if (
        not isinstance(value.get("expected_cpu_ids"), list)
        or any(type(cpu) is not int for cpu in value["expected_cpu_ids"])
        or not isinstance(value.get("place_cpu_ids"), list)
        or any(
            not isinstance(place, list) or any(type(cpu) is not int for cpu in place)
            for place in value["place_cpu_ids"]
        )
    ):
        raise RuntimeError("OpenMP placement CPU fields are not exact integer lists")
    if (
        value["omp_thread_limit"] < threads
        or value["effective_openmp_capacity"] < threads
    ):
        raise RuntimeError("OpenMP placement capacity is below requested threads")
    workers = value.get("workers")
    if not isinstance(workers, list) or len(workers) != threads:
        raise RuntimeError("OpenMP placement worker count mismatch")
    worker_keys = {
        "thread_num",
        "place_num",
        "place_cpu_ids",
        "sched_affinity_cpu_ids",
        "current_cpu",
        "verified",
    }
    for index, worker in enumerate(workers):
        if not isinstance(worker, Mapping) or set(worker) != worker_keys:
            raise RuntimeError("OpenMP placement worker schema mismatch")
        expected_worker = {
            "thread_num": index,
            "place_num": index,
            "place_cpu_ids": [expected_cpus[index]],
            "sched_affinity_cpu_ids": [expected_cpus[index]],
            "current_cpu": expected_cpus[index],
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
            or any(type(cpu) is not int for cpu in worker["place_cpu_ids"])
            or any(type(cpu) is not int for cpu in worker["sched_affinity_cpu_ids"])
        ):
            raise RuntimeError("OpenMP placement worker types are not exact")
    return dict(value)


def _exact_build_required(args: argparse.Namespace) -> dict[str, Any]:
    required: dict[str, Any] = {
        "api_version": EXPECTED_API_VERSION,
        "backend_version": EXPECTED_BACKEND_VERSION,
        "source_commit": args.expected_source_commit,
        "source_tree_sha256": args.expected_source_tree_sha256,
        "blas_runtime_isolation": "private_static",
        "gemm_integrity_enabled": True,
        "gemm_vendor_entry_outer_openmp_guard": True,
        "blas_runtime_threads": args.threads,
        "openmp_effective_capacity_policy": ("bound_places_else_sched_affinity_v1"),
        "openmp_placement_contract_supported": True,
        "openmp_placement_contract_schema": PLACEMENT_SCHEMA,
        "openmp_placement_contract_configured": True,
        "openmp_placement_contract_immutable": True,
        "openmp_placement_probe_vendor_calls": 0,
    }
    if args.expected_backend == "openblas":
        required.update(
            {
                "blas_runtime_threading_layer": "openmp",
                "blas_vendor": "OpenBLAS",
                "gemm_execution_mode": "serialized_fixed_private_openblas",
                "private_openblas_archive_sha256": args.expected_archive_sha256,
                "private_blas_backend": "openblas",
                "private_blas_archive_sha256": args.expected_archive_sha256,
            }
        )
    else:
        ways = (
            {name: 1 for name in BLIS_WAY_NAMES}
            if args.blis_thread_strategy == "automatic"
            else dict(args.blis_thread_ways)
        )
        required.update(
            {
                "blas_runtime_threading_layer": "pthreads",
                "blas_runtime_worker_affinity_policy": (
                    "inherit_authenticated_selected_cpu_set_per_call"
                ),
                "blas_vendor": "BLIS",
                "gemm_execution_mode": "serialized_fixed_private_blis",
                "blas_runtime_config": "BLIS 2.0 config=zen",
                "blas_runtime_corename": "zen",
                "private_openblas_archive_sha256": "none",
                "private_blas_backend": "upstream_blis",
                "private_blas_archive_sha256": args.expected_archive_sha256,
                "private_blas_source_commit": args.expected_private_source_commit,
                "private_blas_source_tree_sha256": (
                    args.expected_private_source_tree_sha256
                ),
                "private_blas_config_family": "zen",
                "blas_runtime_thread_strategy": args.blis_thread_strategy,
                "blas_runtime_thread_ways": ways,
                "blas_runtime_owner_thread_enforced": True,
                "blas_runtime_owner_thread_configured": True,
                "blas_runtime_environment_immutable": True,
                "blas_runtime_environment_contract": "blis_process_start_v1",
                "blas_runtime_tls_enabled": True,
                "gemm_integrity_minimum_vendor_flops": (
                    GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
                ),
                "native_integrity_snapshot_numa_contract_supported": True,
                "native_integrity_snapshot_numa_contract_schema": (
                    NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA
                ),
                "native_integrity_snapshot_numa_query_chunk_page_limit": (
                    NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
                ),
                "native_gemm_output_numa_contract_supported": True,
                "native_gemm_output_numa_contract_schema": (
                    NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
                ),
                "native_gemm_output_numa_query_chunk_page_limit": (
                    NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
                ),
                "native_gemm_output_numa_evidence_capacity": (
                    NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
                ),
            }
        )
    return required


def _validate_native_numa_build_contract(
    raw_build: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "gemm_integrity_minimum_vendor_flops": (
            GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
        ),
        "native_integrity_snapshot_numa_contract_supported": True,
        "native_integrity_snapshot_numa_contract_schema": (
            NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA
        ),
        "native_integrity_snapshot_numa_query_chunk_page_limit": (
            NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
        ),
        "native_gemm_output_numa_contract_supported": True,
        "native_gemm_output_numa_contract_schema": NATIVE_GEMM_OUTPUT_NUMA_SCHEMA,
        "native_gemm_output_numa_query_chunk_page_limit": (
            NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
        ),
        "native_gemm_output_numa_evidence_capacity": (
            NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        ),
    }
    mismatches = {
        name: {"expected": expected, "observed": raw_build.get(name)}
        for name, expected in required.items()
        if raw_build.get(name) != expected
    }
    boolean_fields = (
        "native_integrity_snapshot_numa_contract_supported",
        "native_gemm_output_numa_contract_supported",
    )
    integer_fields = (
        "gemm_integrity_minimum_vendor_flops",
        "native_integrity_snapshot_numa_query_chunk_page_limit",
        "native_gemm_output_numa_query_chunk_page_limit",
        "native_gemm_output_numa_evidence_capacity",
    )
    if (
        mismatches
        or any(type(raw_build.get(name)) is not bool for name in boolean_fields)
        or any(type(raw_build.get(name)) is not int for name in integer_fields)
        or any(
            type(raw_build.get(name)) is not str
            for name in (
                "native_integrity_snapshot_numa_contract_schema",
                "native_gemm_output_numa_contract_schema",
            )
        )
    ):
        raise RuntimeError(
            f"native NUMA build contract mismatch: {mismatches}"
        )
    return dict(required)


def _validate_build_info(
    module: Any,
    module_path: Path,
    args: argparse.Namespace,
    placement: Mapping[str, Any],
) -> dict[str, Any]:
    raw_build = dict(module.build_info())
    build = _json_safe(raw_build)
    required = _exact_build_required(args)
    mismatches = {
        name: {"expected": expected, "observed": build.get(name)}
        for name, expected in required.items()
        if build.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"native build contract mismatch: {mismatches}")
    strict_ints = (
        "api_version",
        "blas_runtime_threads",
        "openmp_placement_probe_vendor_calls",
    )
    strict_bools = (
        "gemm_integrity_enabled",
        "gemm_vendor_entry_outer_openmp_guard",
        "openmp_placement_contract_supported",
        "openmp_placement_contract_configured",
        "openmp_placement_contract_immutable",
    )
    if args.expected_backend == "blis":
        _validate_native_numa_build_contract(raw_build)
        strict_ints += ("gemm_integrity_minimum_vendor_flops",)
        strict_bools += (
            "blas_runtime_owner_thread_enforced",
            "blas_runtime_owner_thread_configured",
            "blas_runtime_environment_immutable",
            "blas_runtime_tls_enabled",
        )
        for name in ("private_blas_header_sha256", "private_blas_cblas_header_sha256"):
            if not _canonical_sha256(build.get(name)):
                raise RuntimeError(f"native build lacks canonical {name}")
        observed_ways = raw_build.get("blas_runtime_thread_ways")
        if (
            not isinstance(observed_ways, Mapping)
            or set(observed_ways) != set(BLIS_WAY_NAMES)
            or any(type(observed_ways.get(name)) is not int for name in BLIS_WAY_NAMES)
            or any(observed_ways[name] <= 0 for name in BLIS_WAY_NAMES)
        ):
            raise RuntimeError(
                "native BLIS thread ways are not exact positive integers"
            )
    if any(type(raw_build.get(name)) is not int for name in strict_ints):
        raise RuntimeError("native build integer contract fields are not exact")
    if any(type(raw_build.get(name)) is not bool for name in strict_bools):
        raise RuntimeError("native build boolean contract fields are not exact")
    evidence = raw_build.get("openmp_placement_contract_evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("build-info placement evidence is not a mapping")
    try:
        _validate_placement_attestation(evidence, args.cpus, args.threads)
    except RuntimeError as error:
        raise RuntimeError(
            f"build-info placement evidence is invalid: {error}"
        ) from error
    if _canonical_json_sha256(evidence) != _canonical_json_sha256(placement):
        raise RuntimeError(
            "build-info placement evidence differs from configured evidence"
        )
    if args.expected_backend == "openblas":
        config = build.get("blas_runtime_config")
        corename = build.get("blas_runtime_corename")
        if (
            type(config) is not str
            or not config.startswith("OpenBLAS ")
            or "USE_OPENMP" not in config.split()
            or type(corename) is not str
            or corename.lower() != "zen"
        ):
            raise RuntimeError(
                "OpenBLAS version/threading/architecture contract mismatch"
            )
    return {
        "module_path": str(module_path.resolve()),
        "module_sha256": _sha256(module_path.resolve()),
        "build_info": build,
        "build_info_canonical_json_sha256": _canonical_json_sha256(build),
    }


def _load_exact_native(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("gxeldcore", path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("loaded native path differs from requested path")
    return module


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
        "aligned": bool(array.flags.aligned),
        "sha256_storage_order": _array_sha256(array),
    }


def _fill_random(array: Any, rng: Any, scale: float) -> None:
    flat = array.ravel(order="K")
    rng.standard_normal(size=flat.shape, out=flat)
    flat *= scale


def _prepare_operands(
    case: Case,
    seed: int,
    np: Any,
    *,
    nodes: Sequence[int] | None = None,
    allocate_bound_buffer: Any = None,
    verify_bound_buffer: Any = None,
) -> tuple[Any, Any, dict[str, Any] | None]:
    rng = np.random.default_rng(seed + (0 if case.operation == "source" else 1))
    n, k, p = case.n_samples, case.block_width, case.panel_columns
    contracted = nodes is not None
    if contracted != (
        callable(allocate_bound_buffer) and callable(verify_bound_buffer)
    ):
        raise RuntimeError(
            "Bound operand preparation requires nodes and both buffer APIs."
        )

    def allocate(shape: tuple[int, int]) -> tuple[Any, Any, dict | None]:
        byte_count = math.prod(shape) * 8
        if not contracted:
            return np.empty(shape, dtype=np.float64, order="F"), None, None
        owner, allocation = allocate_bound_buffer(byte_count, list(nodes))
        array = np.ndarray(
            shape, dtype=np.float64, order="F", buffer=owner
        )
        return array, owner, allocation

    left_shape = (n, k)
    right_shape = (k, p) if case.operation == "source" else (n, p)
    left, left_owner, left_allocation = allocate(left_shape)
    right, right_owner, right_allocation = allocate(right_shape)
    if case.operation == "source":
        _fill_random(left, rng, 1.0)
        _fill_random(right, rng, 1.0 / math.sqrt(k))
    else:
        _fill_random(left, rng, 1.0)
        _fill_random(right, rng, 1.0 / math.sqrt(n))
    if not contracted:
        return left, right, None
    assert left_owner is not None and right_owner is not None
    input_numa = {
        "schema": "summit.gxe.exact_harness_bound_operands.v1",
        "schema_version": 1,
        "selected_nodes": list(nodes),
        "verification_boundary": "post_fill_pre_ready",
        "left": {
            "allocation": left_allocation,
            "verification": verify_bound_buffer(
                left_owner, int(left.nbytes), list(nodes)
            ),
        },
        "right": {
            "allocation": right_allocation,
            "verification": verify_bound_buffer(
                right_owner, int(right.nbytes), list(nodes)
            ),
        },
        "complete": True,
    }
    return left, right, input_numa


def _require_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{field} must be an exact integer")
    return value


def _operand_byte_count(record: Mapping[str, Any], operand: str) -> int:
    dimensions = {
        name: _require_int(record.get(name), f"telemetry.{name}")
        for name in ("m", "n", "k", "lda", "ldb", "ldc")
    }
    m, n, k = dimensions["m"], dimensions["n"], dimensions["k"]
    if operand == "a":
        rows, columns = (m, k) if record.get("transpose_a") == "N" else (k, m)
        leading = dimensions["lda"]
    elif operand == "b":
        rows, columns = (k, n) if record.get("transpose_b") == "N" else (n, k)
        leading = dimensions["ldb"]
    elif operand == "c":
        rows, columns, leading = m, n, dimensions["ldc"]
    else:
        raise RuntimeError(f"unknown operand {operand}")
    if record.get("layout") != "column_major" or leading < rows:
        raise RuntimeError("telemetry operand layout/leading dimension is invalid")
    return ((columns - 1) * leading + rows) * 8


def _positive_histogram(value: object, field: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a mapping")
    result: dict[int, int] = {}
    for raw_node, raw_count in value.items():
        if not isinstance(raw_node, str) or not raw_node.isdigit():
            raise RuntimeError(f"{field} has a noncanonical node")
        node = int(raw_node)
        count = _require_int(raw_count, f"{field}.{raw_node}")
        if str(node) != raw_node or count <= 0:
            raise RuntimeError(f"{field} has a noncanonical entry")
        result[node] = count
    return result


def _validate_bound_operand_evidence(
    value: object, case: Case, selected_nodes: Sequence[int]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("exact-layout bound operand evidence is absent")
    nodes = list(selected_nodes)
    if (
        value.get("schema") != "summit.gxe.exact_harness_bound_operands.v1"
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("selected_nodes") != nodes
        or value.get("verification_boundary") != "post_fill_pre_ready"
        or value.get("complete") is not True
    ):
        raise RuntimeError("exact-layout bound operand contract is malformed")
    shapes = {
        "left": (case.n_samples, case.block_width),
        "right": (
            (case.block_width, case.panel_columns)
            if case.operation == "source"
            else (case.n_samples, case.panel_columns)
        ),
    }
    summaries: dict[str, Any] = {}
    for name, shape in shapes.items():
        record = value.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"bound operand {name} evidence is absent")
        allocation = record.get("allocation")
        verification = record.get("verification")
        if not isinstance(allocation, Mapping) or not isinstance(
            verification, Mapping
        ):
            raise RuntimeError(f"bound operand {name} evidence is malformed")
        byte_count = math.prod(shape) * 8
        page_size = _require_int(
            verification.get("page_size"), f"bound.{name}.page_size"
        )
        mapping_bytes = ((byte_count + page_size - 1) // page_size) * page_size
        page_count = mapping_bytes // page_size
        common = {
            "schema": "summit.numa_bound_anonymous_buffer.v1",
            "schema_version": 1,
            "byte_count": byte_count,
            "mapping_bytes": mapping_bytes,
            "page_size": page_size,
            "page_count": page_count,
            "selected_nodes": nodes,
            "policy_mode": "bind_static_nodes",
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "range_policy_verified": True,
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }
        for evidence_name, evidence in (
            ("allocation", allocation),
            ("verification", verification),
        ):
            mismatches = {
                key: {"expected": expected, "observed": evidence.get(key)}
                for key, expected in common.items()
                if evidence.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"bound operand {name} {evidence_name} disagrees: {mismatches}"
                )
        chunks = math.ceil(page_count / 65_536)
        histogram = _positive_histogram(
            verification.get("node_histogram"),
            f"bound.{name}.node_histogram",
        )
        required_verification = {
            "post_decode_complete_page_query": True,
            "post_decode_strict_policy_verified": True,
            "queried_pages": page_count,
            "resolved_pages": page_count,
            "query_chunks": chunks,
            "query_chunk_page_limit": 65_536,
            "ordered_status_encoding": f"native_32bit_signed_{sys.byteorder}",
            "complete": True,
        }
        if (
            any(
                verification.get(key) != expected
                for key, expected in required_verification.items()
            )
            or sum(histogram.values()) != page_count
            or not set(histogram).issubset(nodes)
            or not _canonical_sha256(
                verification.get("ordered_status_sha256")
            )
            or allocation.get("post_decode_complete_page_query") is not False
        ):
            raise RuntimeError(
                f"bound operand {name} exhaustive page evidence is incomplete"
            )
        summaries[name] = {
            "byte_count": byte_count,
            "mapping_bytes": mapping_bytes,
            "page_count": page_count,
            "node_histogram": {str(key): histogram[key] for key in sorted(histogram)},
            "complete": True,
        }
    return {
        "passed": True,
        "verification_boundary": "post_fill_pre_ready",
        "selected_nodes": nodes,
        "operands": summaries,
    }


def _ordered_single_node_status_sha256(node: int, page_count: int) -> str:
    encoded = node.to_bytes(4, byteorder="little", signed=True)
    digest = hashlib.sha256()
    remaining = page_count
    while remaining:
        count = min(remaining, NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT)
        digest.update(encoded * count)
        remaining -= count
    return digest.hexdigest()


def _validate_native_mapping_geometry(
    value: Mapping[str, Any],
    *,
    logical_byte_count: int,
    nodes: Sequence[int],
    field: str,
) -> dict[str, Any]:
    page_size = _require_int(value.get("page_size"), f"{field}.page_size")
    mapping_bytes = _require_int(
        value.get("mapping_bytes"), f"{field}.mapping_bytes"
    )
    mapping_page_count = _require_int(
        value.get("mapping_page_count"), f"{field}.mapping_page_count"
    )
    queried_pages = _require_int(
        value.get("queried_pages"), f"{field}.queried_pages"
    )
    resolved_pages = _require_int(
        value.get("resolved_pages"), f"{field}.resolved_pages"
    )
    query_chunks = _require_int(
        value.get("query_chunks"), f"{field}.query_chunks"
    )
    query_limit = _require_int(
        value.get("query_chunk_page_limit"),
        f"{field}.query_chunk_page_limit",
    )
    system_page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size != system_page_size or logical_byte_count <= 0:
        raise RuntimeError(f"{field} page/logical-byte geometry is invalid")
    expected_pages = (logical_byte_count - 1) // page_size + 1
    expected_mapping_bytes = expected_pages * page_size
    expected_chunks = (
        expected_pages + NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT - 1
    ) // NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
    expected_nodes = list(nodes)
    selected_nodes = value.get("selected_nodes")
    if (
        not isinstance(selected_nodes, list)
        or any(type(node) is not int for node in selected_nodes)
        or selected_nodes != expected_nodes
        or expected_nodes != sorted(set(expected_nodes))
    ):
        raise RuntimeError(f"{field} selected nodes are not exact")
    histogram = _positive_histogram(
        value.get("node_histogram"), f"{field}.node_histogram"
    )
    geometry = {
        "mapping_bytes": mapping_bytes,
        "mapping_page_count": mapping_page_count,
        "queried_pages": queried_pages,
        "resolved_pages": resolved_pages,
        "query_chunks": query_chunks,
        "query_chunk_page_limit": query_limit,
    }
    expected = {
        "mapping_bytes": expected_mapping_bytes,
        "mapping_page_count": expected_pages,
        "queried_pages": expected_pages,
        "resolved_pages": expected_pages,
        "query_chunks": expected_chunks,
        "query_chunk_page_limit": NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT,
    }
    mismatches = {
        name: {"expected": expected_value, "observed": geometry[name]}
        for name, expected_value in expected.items()
        if geometry[name] != expected_value
    }
    if (
        mismatches
        or sum(histogram.values()) != expected_pages
        or not set(histogram).issubset(expected_nodes)
    ):
        raise RuntimeError(f"{field} mapping/query geometry mismatch: {mismatches}")
    ordered_digest = value.get("ordered_status_sha256")
    if not _canonical_sha256(ordered_digest):
        raise RuntimeError(f"{field} ordered status digest is not canonical SHA256")
    digest_recomputed = len(histogram) == 1
    if digest_recomputed:
        only_node = next(iter(histogram))
        if ordered_digest != _ordered_single_node_status_sha256(
            only_node, expected_pages
        ):
            raise RuntimeError(f"{field} ordered status digest mismatch")
    return {
        "page_size": page_size,
        "mapping_page_count": expected_pages,
        "mapping_bytes": expected_mapping_bytes,
        "query_chunks": expected_chunks,
        "selected_nodes": expected_nodes,
        "node_histogram": {str(node): histogram[node] for node in sorted(histogram)},
        "ordered_status_sha256": ordered_digest,
        "ordered_status_digest_independently_recomputed": digest_recomputed,
    }


def _validate_native_gemm_output_numa_evidence(
    value: Mapping[str, Any],
    case: Case,
    nodes: Sequence[int],
    *,
    expected_call_id: int,
) -> dict[str, Any]:
    required_keys = {
        "schema",
        "schema_version",
        "applicable",
        "operand_role",
        "contract_required",
        "complete",
        "call_id",
        "logical_rows",
        "logical_columns",
        "storage_layout",
        "logical_byte_count",
        "allocation_mode",
        "mapping_bytes",
        "page_size",
        "mapping_page_count",
        "selected_nodes",
        "policy_mode",
        "policy_mode_value",
        "anonymous_private_mapping",
        "page_aligned_mapping",
        "writable_output",
        "bound_before_first_touch",
        "pre_touch_live_owner_policy_verified",
        "pre_touch_range_policy_verified",
        "post_repair_live_owner_policy_verified",
        "post_repair_range_policy_verified",
        "post_repair_complete_page_query",
        "queried_pages",
        "resolved_pages",
        "query_chunks",
        "query_chunk_page_limit",
        "node_histogram",
        "ordered_status_sha256",
        "ordered_status_encoding",
        "post_repair_strict_policy_verified",
        "verification_boundary",
        "strict_policy_check",
        "page_query_method",
        "page_migration_requested",
        "placement_repair_performed",
        "sealed_read_only",
    }
    if not isinstance(value, Mapping) or set(value) != required_keys:
        raise RuntimeError("native GEMM output evidence has an unexpected schema")
    rows, columns = case.output_shape
    logical_bytes = rows * columns * 8
    exact = {
        "schema": NATIVE_GEMM_OUTPUT_NUMA_SCHEMA,
        "schema_version": 1,
        "applicable": True,
        "operand_role": "protected_gemm_output",
        "contract_required": True,
        "complete": True,
        "call_id": expected_call_id,
        "logical_rows": rows,
        "logical_columns": columns,
        "storage_layout": "column_major",
        "logical_byte_count": logical_bytes,
        "allocation_mode": "mmap_private_anonymous",
        "policy_mode": "bind_static_nodes",
        "policy_mode_value": NATIVE_STATIC_MEMBIND_POLICY_VALUE,
        "anonymous_private_mapping": True,
        "page_aligned_mapping": True,
        "writable_output": True,
        "bound_before_first_touch": True,
        "pre_touch_live_owner_policy_verified": True,
        "pre_touch_range_policy_verified": True,
        "post_repair_live_owner_policy_verified": True,
        "post_repair_range_policy_verified": True,
        "post_repair_complete_page_query": True,
        "ordered_status_encoding": "signed_int32_little_endian",
        "post_repair_strict_policy_verified": True,
        "verification_boundary": (
            "after_partitioned_or_integrity_repair_before_python_return"
        ),
        "strict_policy_check": "MPOL_MF_STRICT_without_MPOL_MF_MOVE",
        "page_query_method": "move_pages_query_no_migration",
        "page_migration_requested": False,
        "placement_repair_performed": False,
        "sealed_read_only": False,
    }
    mismatches = {
        name: {"expected": expected, "observed": value.get(name)}
        for name, expected in exact.items()
        if value.get(name) != expected
    }
    integer_fields = (
        "schema_version",
        "call_id",
        "logical_rows",
        "logical_columns",
        "logical_byte_count",
        "policy_mode_value",
    )
    boolean_fields = (
        "applicable",
        "contract_required",
        "complete",
        "anonymous_private_mapping",
        "page_aligned_mapping",
        "writable_output",
        "bound_before_first_touch",
        "pre_touch_live_owner_policy_verified",
        "pre_touch_range_policy_verified",
        "post_repair_live_owner_policy_verified",
        "post_repair_range_policy_verified",
        "post_repair_complete_page_query",
        "post_repair_strict_policy_verified",
        "page_migration_requested",
        "placement_repair_performed",
        "sealed_read_only",
    )
    if (
        mismatches
        or any(type(value.get(name)) is not int for name in integer_fields)
        or any(type(value.get(name)) is not bool for name in boolean_fields)
    ):
        raise RuntimeError(f"native GEMM output evidence mismatch: {mismatches}")
    mapping = _validate_native_mapping_geometry(
        value,
        logical_byte_count=logical_bytes,
        nodes=nodes,
        field="native_gemm_output_numa",
    )
    return {
        "passed": True,
        "call_id": expected_call_id,
        "logical_shape": [rows, columns],
        "logical_byte_count": logical_bytes,
        "mapping": mapping,
        "strict_post_repair_no_move_verified": True,
        "writable_output_verified": True,
    }


def _validate_native_integrity_snapshot_numa_evidence(
    value: Mapping[str, Any],
    case: Case,
    nodes: Sequence[int],
    *,
    integrity_minimum_vendor_flops: int,
) -> dict[str, Any]:
    if (
        type(integrity_minimum_vendor_flops) is not int
        or integrity_minimum_vendor_flops <= 0
    ):
        raise RuntimeError("native integrity threshold is not an exact positive integer")
    base_keys = {
        "schema",
        "schema_version",
        "operand_role",
        "integrity_check_shape_eligible",
        "integrity_check_executed",
        "snapshot_available",
        "contract_required",
        "complete",
    }
    if not isinstance(value, Mapping) or not base_keys.issubset(value):
        raise RuntimeError("native integrity snapshot evidence is malformed")
    eligible = (
        case.operation == "source"
        and case.flops >= integrity_minimum_vendor_flops
    )
    base = {
        "schema": NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA,
        "schema_version": 1,
        "operand_role": "native_integrity_snapshot_of_logical_b",
        "integrity_check_shape_eligible": eligible,
        "integrity_check_executed": eligible,
        "snapshot_available": eligible,
        "contract_required": True,
        "complete": eligible,
    }
    mismatches = {
        name: {"expected": expected, "observed": value.get(name)}
        for name, expected in base.items()
        if value.get(name) != expected
    }
    boolean_fields = (
        "integrity_check_shape_eligible",
        "integrity_check_executed",
        "snapshot_available",
        "contract_required",
        "complete",
    )
    if (
        mismatches
        or type(value.get("schema_version")) is not int
        or any(type(value.get(name)) is not bool for name in boolean_fields)
    ):
        raise RuntimeError(f"native integrity snapshot mismatch: {mismatches}")
    if not eligible:
        if set(value) != base_keys:
            raise RuntimeError(
                "noneligible native integrity snapshot discriminator is not exact"
            )
        return {
            "passed": True,
            "integrity_check_shape_eligible": False,
            "snapshot_available": False,
            "discriminator_only": True,
        }
    required_keys = base_keys | {
        "logical_byte_count",
        "mapping_bytes",
        "page_size",
        "mapping_page_count",
        "selected_nodes",
        "policy_mode",
        "policy_mode_value",
        "anonymous_private_mapping",
        "page_aligned_mapping",
        "bound_before_first_touch",
        "live_owner_policy_verified",
        "pre_touch_range_policy_verified",
        "pre_vendor_complete_page_query",
        "queried_pages",
        "resolved_pages",
        "query_chunks",
        "query_chunk_page_limit",
        "node_histogram",
        "ordered_status_sha256",
        "ordered_status_encoding",
        "pre_vendor_strict_policy_verified",
        "sealed_read_only_before_vendor",
        "strict_policy_check",
        "page_query_method",
        "page_migration_requested",
        "placement_repair_performed",
    }
    if set(value) != required_keys:
        raise RuntimeError(
            "eligible native integrity snapshot evidence has an unexpected schema"
        )
    logical_bytes = case.cblas["k"] * case.cblas["n"] * 8
    exact = {
        "logical_byte_count": logical_bytes,
        "policy_mode": "bind_static_nodes",
        "policy_mode_value": NATIVE_STATIC_MEMBIND_POLICY_VALUE,
        "anonymous_private_mapping": True,
        "page_aligned_mapping": True,
        "bound_before_first_touch": True,
        "live_owner_policy_verified": True,
        "pre_touch_range_policy_verified": True,
        "pre_vendor_complete_page_query": True,
        "ordered_status_encoding": "signed_int32_little_endian",
        "pre_vendor_strict_policy_verified": True,
        "sealed_read_only_before_vendor": True,
        "strict_policy_check": "MPOL_MF_STRICT_without_MPOL_MF_MOVE",
        "page_query_method": "move_pages_query_no_migration",
        "page_migration_requested": False,
        "placement_repair_performed": False,
    }
    mismatches = {
        name: {"expected": expected, "observed": value.get(name)}
        for name, expected in exact.items()
        if value.get(name) != expected
    }
    if (
        mismatches
        or any(
            type(value.get(name)) is not int
            for name in ("logical_byte_count", "policy_mode_value")
        )
        or any(
            type(value.get(name)) is not bool
            for name in (
                "anonymous_private_mapping",
                "page_aligned_mapping",
                "bound_before_first_touch",
                "live_owner_policy_verified",
                "pre_touch_range_policy_verified",
                "pre_vendor_complete_page_query",
                "pre_vendor_strict_policy_verified",
                "sealed_read_only_before_vendor",
                "page_migration_requested",
                "placement_repair_performed",
            )
        )
    ):
        raise RuntimeError(f"native integrity snapshot mismatch: {mismatches}")
    mapping = _validate_native_mapping_geometry(
        value,
        logical_byte_count=logical_bytes,
        nodes=nodes,
        field="native_integrity_snapshot_numa",
    )
    return {
        "passed": True,
        "integrity_check_shape_eligible": True,
        "snapshot_available": True,
        "logical_byte_count": logical_bytes,
        "mapping": mapping,
        "sealed_read_only_before_vendor": True,
        "strict_pre_vendor_no_move_verified": True,
    }


def _validate_native_gemm_output_reset_status(
    status: Mapping[str, Any], *, expected_next_call_id: int
) -> dict[str, Any]:
    exact = {
        "schema_version": 1,
        "capacity": NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY,
        "buffered_records": 0,
        "captured_records": 0,
        "dropped_records": 0,
        "next_call_id": expected_next_call_id,
        "attempted_calls": 0,
        "verified_calls": 0,
        "legacy_calls": 0,
        "failed_calls": 0,
        "query_chunk_page_limit": NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT,
    }
    if set(status) != set(exact):
        raise RuntimeError("native GEMM output reset status has an unexpected schema")
    mismatches = {
        name: {"expected": expected, "observed": status.get(name)}
        for name, expected in exact.items()
        if status.get(name) != expected
    }
    if mismatches or any(type(status.get(name)) is not int for name in exact):
        raise RuntimeError(f"native GEMM output reset status mismatch: {mismatches}")
    return dict(status)


def _validate_native_gemm_output_post_status(
    status: Mapping[str, Any], *, call_id: int
) -> dict[str, Any]:
    exact = {
        "schema_version": 1,
        "capacity": NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY,
        "buffered_records": 0,
        "captured_records": 1,
        "dropped_records": 0,
        "next_call_id": call_id + 1,
        "attempted_calls": 1,
        "verified_calls": 1,
        "legacy_calls": 0,
        "failed_calls": 0,
        "query_chunk_page_limit": NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT,
    }
    if set(status) != set(exact):
        raise RuntimeError("native GEMM output status has an unexpected schema")
    mismatches = {
        name: {"expected": expected, "observed": status.get(name)}
        for name, expected in exact.items()
        if status.get(name) != expected
    }
    if mismatches or any(type(status.get(name)) is not int for name in exact):
        raise RuntimeError(f"native GEMM output status mismatch: {mismatches}")
    return dict(status)


def _validate_protected_call_native_numa_contract(
    *,
    telemetry: Mapping[str, Any],
    output_evidence: Mapping[str, Any],
    reset_status: Mapping[str, Any],
    post_status: Mapping[str, Any],
    case: Case,
    nodes: Sequence[int],
    expected_call_id: int,
    integrity_minimum_vendor_flops: int,
) -> dict[str, Any]:
    _validate_native_gemm_output_reset_status(
        reset_status, expected_next_call_id=expected_call_id
    )
    output_gate = _validate_native_gemm_output_numa_evidence(
        output_evidence, case, nodes, expected_call_id=expected_call_id
    )
    nested_output = telemetry.get("native_gemm_output_numa")
    if not isinstance(nested_output, Mapping) or dict(nested_output) != dict(
        output_evidence
    ):
        raise RuntimeError(
            "native GEMM output queue evidence differs from nested vendor evidence"
        )
    _validate_native_gemm_output_post_status(post_status, call_id=expected_call_id)
    snapshot = telemetry.get("native_integrity_snapshot_numa")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("vendor telemetry lacks native integrity snapshot evidence")
    snapshot_gate = _validate_native_integrity_snapshot_numa_evidence(
        snapshot,
        case,
        nodes,
        integrity_minimum_vendor_flops=integrity_minimum_vendor_flops,
    )
    return {
        "passed": True,
        "call_id": expected_call_id,
        "queue_record_count": 1,
        "queue_and_nested_vendor_evidence_identical": True,
        "output": output_gate,
        "integrity_snapshot": snapshot_gate,
        "status_counters_exact": True,
    }


def _validate_operand_numa(
    record: Mapping[str, Any],
    name: str,
    value: Mapping[str, Any],
    *,
    page_size: int,
    sample_limit: int,
    selected_nodes: set[int],
) -> dict[str, Any]:
    byte_count = _operand_byte_count(record, name)
    fields = {
        field: _require_int(value.get(field), f"numa.{name}.{field}")
        for field in (
            "storage_span_pages",
            "fully_contained_pages",
            "operand_byte_count",
            "operand_start_address_page_offset",
            "operand_end_exclusive_address_page_offset",
            "selected_sample_pages",
            "resolved_sample_pages",
            "page_query_error_pages",
        )
    }
    start = fields["operand_start_address_page_offset"]
    if not 0 <= start < page_size:
        raise RuntimeError(f"NUMA operand {name} has an invalid start offset")
    first_full = 0 if start == 0 else page_size - start
    full_pages = (
        (byte_count - first_full) // page_size if first_full <= byte_count else 0
    )
    selected = min(full_pages, sample_limit)
    expected = {
        "operand_byte_count": byte_count,
        "storage_span_pages": (start + byte_count - 1) // page_size + 1,
        "fully_contained_pages": full_pages,
        "operand_end_exclusive_address_page_offset": (start + byte_count) % page_size,
        "selected_sample_pages": selected,
        "resolved_sample_pages": selected,
        "page_query_error_pages": 0,
    }
    mismatches = {
        field: {"expected": expected_value, "observed": fields[field]}
        for field, expected_value in expected.items()
        if fields[field] != expected_value
    }
    if selected <= 0 or mismatches or value.get("query_status") != "queried":
        raise RuntimeError(f"NUMA operand {name} span/query mismatch: {mismatches}")
    samples = value.get("ordered_samples")
    if not isinstance(samples, list) or len(samples) != selected:
        raise RuntimeError(f"NUMA operand {name} ordered sample count mismatch")
    observed_nodes: list[int] = []
    for ordinal, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise RuntimeError(f"NUMA operand {name} sample is malformed")
        page_index = (
            0 if selected == 1 else ordinal * (full_pages - 1) // (selected - 1)
        )
        byte_offset = first_full + page_index * page_size
        sample_expected = {
            "sample_ordinal": ordinal,
            "full_page_index": page_index,
            "byte_offset_from_operand_start": byte_offset,
        }
        if any(
            sample.get(field) != expected_value
            for field, expected_value in sample_expected.items()
        ):
            raise RuntimeError(f"NUMA operand {name} sample schedule mismatch")
        if (start + byte_offset) % page_size or byte_offset + page_size > byte_count:
            raise RuntimeError(f"NUMA operand {name} sampled a partial page")
        node = _require_int(sample.get("numa_node"), f"numa.{name}.node")
        raw_status = _require_int(
            sample.get("raw_move_pages_status"), f"numa.{name}.raw_status"
        )
        if (
            sample.get("status_kind") != "numa_node"
            or sample.get("page_query_errno") is not None
            or raw_status != node
            or node not in selected_nodes
        ):
            rejection = {
                "operand": name,
                "sample_ordinal": ordinal,
                "full_page_index": page_index,
                "byte_offset_from_operand_start": byte_offset,
                "raw_move_pages_status": raw_status,
                "numa_node": node,
                "status_kind": sample.get("status_kind"),
                "page_query_errno": sample.get("page_query_errno"),
                "selected_nodes": sorted(selected_nodes),
                "ordered_samples": samples,
            }
            raise RuntimeError(
                "NUMA operand page is unresolved or nonlocal: "
                + json.dumps(rejection, sort_keys=True, separators=(",", ":"))
            )
        observed_nodes.append(node)
    histogram = _positive_histogram(
        value.get("node_histogram"), f"numa.{name}.histogram"
    )
    ordered_histogram = {
        node: observed_nodes.count(node) for node in sorted(set(observed_nodes))
    }
    error_histogram = value.get("page_error_errno_histogram")
    if histogram != ordered_histogram or error_histogram not in ({}, None):
        raise RuntimeError(f"NUMA operand {name} histogram mismatch")
    return {
        "passed": True,
        "fully_contained_pages": full_pages,
        "selected_sample_pages": selected,
        "observed_nodes": sorted(histogram),
        "within_selected_nodes": set(histogram).issubset(selected_nodes),
    }


def _validate_numa(
    record: Mapping[str, Any], selected_nodes: set[int]
) -> dict[str, Any]:
    numa = record.get("operand_numa_page_samples")
    if not isinstance(numa, Mapping):
        raise RuntimeError("telemetry lacks operand NUMA page samples")
    page_size = _require_int(numa.get("system_page_size"), "numa.system_page_size")
    sample_limit = _require_int(
        numa.get("sample_limit_per_operand"), "numa.sample_limit_per_operand"
    )
    required = {
        "schema_version": 1,
        "sampling_method": "move_pages_query_no_migration",
        "sampling_timing": "after_vendor_call_outside_timed_interval",
        "address_selection_schema_version": 1,
        "address_selection_policy": NUMA_ADDRESS_SELECTION_POLICY,
        "sample_limit_per_operand": 8,
        "system_page_size": int(os.sysconf("SC_PAGE_SIZE")),
        "selected_addresses_are_page_bases": True,
        "partial_boundary_pages_included": False,
        "first_and_last_fully_contained_pages_selected": True,
        "virtual_addresses_exposed": False,
        "operand_byte_range_semantics": "[start_address,end_exclusive_address)",
        "address_evidence": (
            "ordered_samples_with_operand_relative_byte_offsets_and_full_page_indices"
        ),
        "syscall_result": 0,
        "syscall_errno": 0,
    }
    mismatches = {
        name: {"expected": expected, "observed": numa.get(name)}
        for name, expected in required.items()
        if numa.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"NUMA sampling contract mismatch: {mismatches}")
    integer_fields = (
        "schema_version",
        "address_selection_schema_version",
        "sample_limit_per_operand",
        "system_page_size",
        "syscall_result",
        "syscall_errno",
    )
    boolean_fields = (
        "selected_addresses_are_page_bases",
        "partial_boundary_pages_included",
        "first_and_last_fully_contained_pages_selected",
        "virtual_addresses_exposed",
    )
    if any(type(numa.get(name)) is not int for name in integer_fields):
        raise RuntimeError("NUMA sampling integer fields are not exact integers")
    if any(type(numa.get(name)) is not bool for name in boolean_fields):
        raise RuntimeError("NUMA sampling boolean fields are not exact booleans")
    operands = numa.get("operands")
    if not isinstance(operands, Mapping) or set(operands) != {"a", "b", "c"}:
        raise RuntimeError("NUMA sampling lacks exact A/B/C operands")
    summaries = {
        name: _validate_operand_numa(
            record,
            name,
            operands[name],
            page_size=page_size,
            sample_limit=sample_limit,
            selected_nodes=selected_nodes,
        )
        for name in ("a", "b", "c")
    }
    return {
        "passed": True,
        "selected_nodes": sorted(selected_nodes),
        "operands": summaries,
    }


def _validate_telemetry(
    record: Mapping[str, Any],
    case: Case,
    cpus: Sequence[int],
    nodes: Sequence[int],
    build: Mapping[str, Any],
    *,
    measured: bool,
) -> dict[str, Any]:
    expected = {
        **case.cblas,
        "schema_version": 1,
        "arithmetic_dtype": "float64",
        "alpha": 1.0,
        "beta": 0.0,
        "requested_threads": case.threads,
        "configured_threads": case.threads,
        "backend_threads": case.threads,
        "omp_in_parallel": False,
        "omp_level": 0,
        "omp_active_level": 0,
        "omp_max_active_levels": 1,
        "omp_max_threads": case.threads,
        "omp_num_threads": 1,
        "omp_thread_num": 0,
        # The immutable singleton-place contract binds the initial/master
        # thread to place zero.  The placement attestation, rather than this
        # calling-thread snapshot, proves coverage of the complete worker set.
        "cpu_affinity_count": 1,
        "cpu_affinity_list": str(cpus[0]),
        "completed": True,
        "backend": build["blas_vendor"],
        "backend_config": build["blas_runtime_config"],
        "backend_corename": build["blas_runtime_corename"],
    }
    mismatches = {
        name: {"expected": expected_value, "observed": record.get(name)}
        for name, expected_value in expected.items()
        if record.get(name) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"vendor telemetry mismatch: {mismatches}")
    for name in (
        "schema_version",
        "sequence",
        "m",
        "n",
        "k",
        "lda",
        "ldb",
        "ldc",
        "requested_threads",
        "configured_threads",
        "backend_threads",
        "omp_level",
        "omp_active_level",
        "omp_max_active_levels",
        "omp_max_threads",
        "omp_num_threads",
        "omp_thread_num",
        "cpu_affinity_count",
    ):
        if type(record.get(name)) is not int:
            raise RuntimeError(f"telemetry {name} is not an exact integer")
    if record["sequence"] <= 0:
        raise RuntimeError("telemetry sequence is not positive")
    for name in ("omp_in_parallel", "completed"):
        if type(record.get(name)) is not bool:
            raise RuntimeError(f"telemetry {name} is not an exact boolean")
    if float(record.get("flop_count", -1)) != float(case.flops):
        raise RuntimeError("telemetry flop count mismatch")
    numeric: dict[str, float] = {}
    numeric_fields = (
        "alpha",
        "beta",
        "flop_count",
        "wall_seconds",
        "process_cpu_seconds",
        "gflops_per_second",
        "process_cpu_to_wall_ratio",
        "active_core_equivalents",
    )
    if any(type(record.get(name)) is not float for name in numeric_fields):
        raise RuntimeError("telemetry floating-point fields are not exact floats")
    for name in numeric_fields:
        value = record[name]
        if not math.isfinite(value):
            raise RuntimeError(f"telemetry {name} is not finite")
        numeric[name] = value
    for name in (
        "flop_count",
        "wall_seconds",
        "process_cpu_seconds",
        "gflops_per_second",
        "process_cpu_to_wall_ratio",
        "active_core_equivalents",
    ):
        if numeric[name] <= 0:
            raise RuntimeError(f"telemetry {name} is not positive")
    recomputed_ratio = numeric["process_cpu_seconds"] / numeric["wall_seconds"]
    if not math.isclose(
        numeric["active_core_equivalents"],
        recomputed_ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        numeric["process_cpu_to_wall_ratio"],
        recomputed_ratio,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("telemetry active-core fields are internally inconsistent")
    recomputed_gflops = numeric["flop_count"] / (numeric["wall_seconds"] * 1.0e9)
    if not math.isclose(
        numeric["gflops_per_second"],
        recomputed_gflops,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError("telemetry GFLOP/s is internally inconsistent")
    minimum = MINIMUM_ACTIVE_CORE_FRACTION * case.threads
    if measured and numeric["active_core_equivalents"] < minimum:
        raise RuntimeError(
            "measured vendor call used fewer than 75% of requested core equivalents"
        )
    for name in ("entry_cpu", "exit_cpu"):
        if _require_int(record.get(name), f"telemetry.{name}") != cpus[0]:
            raise RuntimeError(f"telemetry {name} differs from singleton place zero")
    numa = _validate_numa(record, set(nodes))
    return {
        "passed": True,
        "measured_active_core_gate_required": measured,
        "minimum_active_core_equivalents": minimum,
        "active_core_equivalents": numeric["active_core_equivalents"],
        "numa": numa,
    }


def _validate_telemetry_status(
    status: Mapping[str, Any], *, observed_sequence: int
) -> dict[str, Any]:
    required_keys = {
        "schema_version",
        "capacity",
        "buffered_records",
        "captured_records",
        "dropped_records",
        "next_sequence",
        "operand_numa_sampling_method",
        "operand_numa_sample_limit_per_operand",
        "operand_numa_address_selection_schema_version",
        "operand_numa_address_selection_policy",
        "operand_numa_partial_boundary_pages_included",
    }
    if set(status) != required_keys:
        raise RuntimeError("telemetry status has an unexpected schema")
    exact = {
        "schema_version": 1,
        "buffered_records": 0,
        "captured_records": 1,
        "dropped_records": 0,
        "next_sequence": observed_sequence + 1,
        "operand_numa_sampling_method": "move_pages_query_no_migration",
        "operand_numa_sample_limit_per_operand": 8,
        "operand_numa_address_selection_schema_version": 1,
        "operand_numa_address_selection_policy": (NUMA_ADDRESS_SELECTION_POLICY),
        "operand_numa_partial_boundary_pages_included": False,
    }
    mismatches = {
        name: {"expected": expected, "observed": status.get(name)}
        for name, expected in exact.items()
        if status.get(name) != expected
    }
    integer_fields = (
        "schema_version",
        "capacity",
        "buffered_records",
        "captured_records",
        "dropped_records",
        "next_sequence",
        "operand_numa_sample_limit_per_operand",
        "operand_numa_address_selection_schema_version",
    )
    if (
        mismatches
        or any(type(status.get(name)) is not int for name in integer_fields)
        or type(status.get("operand_numa_partial_boundary_pages_included")) is not bool
        or status["capacity"] <= 0
    ):
        raise RuntimeError(f"telemetry status contract mismatch: {mismatches}")
    return dict(status)


def _sample_indices(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, columns = shape
    candidates = (
        (0, 0),
        (0, columns - 1),
        (rows // 2, columns // 2),
        (rows - 1, 0),
        (rows - 1, columns - 1),
    )
    return list(dict.fromkeys(candidates))


def _full_oracle(
    case: Case, left: Any, right: Any, output: Any, np: Any
) -> dict[str, Any]:
    if tuple(output.shape) != case.output_shape:
        raise RuntimeError("candidate output shape differs from exact case")
    oracle_started = time.perf_counter()
    oracle_cpu_started = time.process_time()
    oracle = left @ right if case.operation == "source" else left.T @ right
    oracle_cpu = time.process_time() - oracle_cpu_started
    oracle_wall = time.perf_counter() - oracle_started
    if tuple(oracle.shape) != case.output_shape or not np.isfinite(oracle).all():
        raise RuntimeError("independent full NumPy oracle is malformed or nonfinite")
    reduction = case.cblas["k"]
    epsilon = float(np.finfo(np.float64).eps)
    gamma = reduction * epsilon / (1.0 - reduction * epsilon)
    relative_factor = 32.0 * (2.0 * gamma + epsilon)
    absolute_floor = 32.0 * epsilon
    if case.operation == "source":
        left_norms = np.sqrt(np.einsum("ij,ij->i", left, left, optimize=False))
    else:
        left_norms = np.sqrt(np.einsum("ij,ij->j", left, left, optimize=False))
    right_norms = np.sqrt(np.einsum("ij,ij->j", right, right, optimize=False))
    maximum_absolute = 0.0
    maximum_normalized = 0.0
    maximum_location = (0, 0)
    column_chunk = 8
    for start in range(0, case.output_shape[1], column_chunk):
        stop = min(case.output_shape[1], start + column_chunk)
        candidate_chunk = output[:, start:stop]
        oracle_chunk = oracle[:, start:stop]
        if not np.isfinite(candidate_chunk).all():
            raise RuntimeError("candidate output contains nonfinite values")
        difference = np.abs(candidate_chunk - oracle_chunk)
        tolerance = (
            absolute_floor
            + relative_factor * left_norms[:, None] * right_norms[None, start:stop]
        )
        normalized = difference / tolerance
        local_flat = int(np.argmax(normalized))
        local = np.unravel_index(local_flat, normalized.shape)
        local_normalized = float(normalized[local])
        if local_normalized > maximum_normalized:
            maximum_normalized = local_normalized
            maximum_location = (int(local[0]), start + int(local[1]))
        maximum_absolute = max(maximum_absolute, float(np.max(difference)))
        if np.any(difference > tolerance):
            raise RuntimeError(
                "candidate differs from the full NumPy oracle outside the "
                "predeclared forward-error bound"
            )
    long_double = np.finfo(np.longdouble)
    binary64 = np.finfo(np.float64)
    if long_double.nmant <= binary64.nmant:
        raise RuntimeError("long double does not exceed binary64 precision")
    witnesses: list[dict[str, Any]] = []
    for row, column in _sample_indices(case.output_shape):
        a = left[row, :] if case.operation == "source" else left[:, row]
        b = right[:, column]
        extended_a = a.astype(np.longdouble)
        extended_b = b.astype(np.longdouble)
        reference = np.sum(extended_a * extended_b, dtype=np.longdouble)
        absolute_product_sum = np.sum(
            np.abs(extended_a * extended_b), dtype=np.longdouble
        )
        candidate = np.longdouble(output[row, column])
        independent = np.longdouble(oracle[row, column])
        witness_tolerance = np.longdouble(32.0) * (
            np.longdouble(2.0 * gamma + epsilon) * absolute_product_sum
            + np.longdouble(epsilon)
        )
        candidate_error = abs(candidate - reference)
        oracle_error = abs(independent - reference)
        if candidate_error > witness_tolerance or oracle_error > witness_tolerance:
            raise RuntimeError("long-double witness exceeds its forward-error bound")
        witnesses.append(
            {
                "row": row,
                "column": column,
                "reference_decimal": str(reference),
                "candidate_absolute_error_decimal": str(candidate_error),
                "oracle_absolute_error_decimal": str(oracle_error),
                "tolerance_decimal": str(witness_tolerance),
            }
        )
    return {
        "passed": True,
        "full_output_compared": True,
        "independent_oracle": "numpy_matmul_plus_long_double_dot_witnesses",
        "oracle_shape": list(oracle.shape),
        "oracle_dtype": str(oracle.dtype),
        "oracle_sha256_storage_order": _array_sha256(oracle),
        "oracle_wall_seconds": oracle_wall,
        "oracle_process_cpu_seconds": oracle_cpu,
        "reduction_length": reduction,
        "binary64_epsilon": epsilon,
        "gamma_reduction": gamma,
        "relative_forward_error_factor": relative_factor,
        "absolute_tolerance_floor": absolute_floor,
        "maximum_absolute_error": maximum_absolute,
        "maximum_tolerance_normalized_error": maximum_normalized,
        "maximum_normalized_error_location": {
            "row": maximum_location[0],
            "column": maximum_location[1],
        },
        "long_double_mantissa_bits": int(long_double.nmant),
        "long_double_witnesses": witnesses,
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": float(min(values)),
        "median": float(statistics.median(values)),
        "maximum": float(max(values)),
        "mean": float(statistics.fmean(values)),
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not sys.flags.no_site:
        raise RuntimeError("worker must run under python -S")
    if args._operation is None:
        raise RuntimeError("internal worker lacks --_operation")
    if Path(sys.executable).resolve() != args.python_executable.resolve():
        raise RuntimeError("worker interpreter differs from --python-executable")
    environment = _validate_process_environment(args)
    prefix = args.install_prefix.resolve()
    module_path = args.native_module.resolve()
    archive_path = args.private_archive.resolve()
    dependencies = _dependency_paths(args)
    package_before = _package_identity(prefix)
    if package_before["manifest_sha256"] != args.expected_package_manifest_sha256:
        raise RuntimeError("installed package changed before worker import")
    if _sha256(module_path) != args.expected_native_sha256:
        raise RuntimeError("native module changed before worker import")
    if _sha256(archive_path) != args.expected_archive_sha256:
        raise RuntimeError("private archive changed before worker import")
    inserted = _bootstrap_paths(prefix, dependencies)
    import summit

    if Path(summit.__file__).resolve().parent != prefix / "summit":
        raise RuntimeError("summit imported outside the exact installed prefix")
    from summit._early_numa import (
        allocate_numa_bound_anonymous_buffer,
        apply_early_numa_membind,
        verify_numa_bound_anonymous_buffer,
    )

    cpu = _cpu_contract(args.cpus)
    if set(os.sched_getaffinity(0)) != set(args.cpus):
        raise RuntimeError("worker taskset affinity differs from selected CPUs")
    nodes = cpu["numa_nodes"]
    numa_attestation = _validate_early_numa_attestation(
        apply_early_numa_membind(_compress_ints(nodes)), nodes
    )
    import numpy as np

    numpy_path = Path(np.__file__).resolve()
    if not any(
        numpy_path == dependency or numpy_path.is_relative_to(dependency)
        for dependency in dependencies
    ):
        raise RuntimeError("NumPy imported outside declared dependency paths")
    module = _load_exact_native(module_path)
    configure_placement = getattr(module, "configure_openmp_placement", None)
    if not callable(configure_placement):
        raise RuntimeError("API-9 native module lacks configure_openmp_placement")
    placement = _validate_placement_attestation(
        configure_placement(list(args.cpus), args.threads),
        args.cpus,
        args.threads,
    )
    configured = module.configure_blas_threads(args.threads)
    if type(configured) is not int or configured != args.threads:
        raise RuntimeError("private BLAS thread configuration is not exact and fixed")
    native = _validate_build_info(module, module_path, args, placement)
    _validate_native_numa_build_contract(native["build_info"])
    case = Case(
        args._operation,
        args.n,
        args.block_width,
        args.probe_tile,
        args.environment_tile,
        args.threads,
    )
    memory = _estimated_memory(case)
    if memory["conservative_peak_gib"] > args.max_memory_gib:
        raise RuntimeError("case exceeds --max-memory-gib before allocation")
    if set(os.sched_getaffinity(0)) != {args.cpus[0]}:
        raise RuntimeError(
            "configured master affinity differs from singleton place zero"
        )
    left, right, input_numa = _prepare_operands(
        case,
        args.seed,
        np,
        nodes=nodes,
        allocate_bound_buffer=allocate_numa_bound_anonymous_buffer,
        verify_bound_buffer=verify_numa_bound_anonymous_buffer,
    )
    input_numa_gate = _validate_bound_operand_evidence(input_numa, case, nodes)
    inputs_before = {"left": _array_record(left), "right": _array_record(right)}
    function = getattr(
        module,
        "protected_matmul_nn" if case.operation == "source" else "protected_matmul_tn",
        None,
    )
    if not callable(function):
        raise RuntimeError("native module lacks exact current-layout protected GEMM")
    for name in (
        "reset_gemm_telemetry",
        "consume_gemm_telemetry",
        "gemm_telemetry_status",
        "reset_native_gemm_output_numa_evidence",
        "consume_native_gemm_output_numa_evidence",
        "get_native_gemm_output_numa_evidence",
        "native_gemm_output_numa_evidence_status",
    ):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"native module lacks required telemetry API {name}")

    calls: list[dict[str, Any]] = []
    baseline = None
    baseline_hash = None
    for ordinal in range(WARMUPS + MEASURED_REPEATS):
        measured = ordinal >= WARMUPS
        module.reset_gemm_telemetry()
        output_reset_status = _json_safe(
            dict(module.native_gemm_output_numa_evidence_status())
        )
        _validate_native_gemm_output_reset_status(
            output_reset_status, expected_next_call_id=ordinal + 1
        )
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        output, repaired = function(left, right, args.threads)
        wrapper_cpu = time.process_time() - cpu_started
        wrapper_wall = time.perf_counter() - wall_started
        if type(repaired) is not int or repaired != 0:
            raise RuntimeError("protected call repaired output columns")
        records = [_json_safe(dict(item)) for item in module.consume_gemm_telemetry()]
        status = _json_safe(dict(module.gemm_telemetry_status()))
        output_records = [
            _json_safe(dict(item))
            for item in module.consume_native_gemm_output_numa_evidence()
        ]
        output_status = _json_safe(
            dict(module.native_gemm_output_numa_evidence_status())
        )
        if len(records) != 1:
            raise RuntimeError(
                "each protected call must emit exactly one telemetry record"
            )
        if len(output_records) != 1:
            raise RuntimeError(
                "each protected call must emit exactly one output NUMA record"
            )
        sequence = records[0].get("sequence")
        if type(sequence) is not int or sequence != ordinal + 1:
            raise RuntimeError("fresh-process telemetry sequence is not exact")
        _validate_telemetry_status(status, observed_sequence=sequence)
        if (
            tuple(output.shape) != case.output_shape
            or str(output.dtype) != "float64"
            or not output.flags.f_contiguous
            or not output.flags.aligned
        ):
            raise RuntimeError("protected call returned an unexpected output array")
        output_hash = _array_sha256(output)
        if baseline is None:
            baseline = output
            baseline_hash = output_hash
        elif output_hash != baseline_hash:
            raise RuntimeError("protected output is not bitwise deterministic")
        telemetry_gate = _validate_telemetry(
            records[0], case, args.cpus, nodes, native["build_info"], measured=measured
        )
        native_numa_gate = _validate_protected_call_native_numa_contract(
            telemetry=records[0],
            output_evidence=output_records[0],
            reset_status=output_reset_status,
            post_status=output_status,
            case=case,
            nodes=nodes,
            expected_call_id=ordinal + 1,
            integrity_minimum_vendor_flops=native["build_info"][
                "gemm_integrity_minimum_vendor_flops"
            ],
        )
        calls.append(
            {
                "ordinal": ordinal,
                "phase": "measured" if measured else "warmup",
                "phase_ordinal": ordinal - WARMUPS if measured else ordinal,
                "wrapper_wall_seconds": wrapper_wall,
                "wrapper_process_cpu_seconds": wrapper_cpu,
                "wrapper_active_core_equivalents": wrapper_cpu / wrapper_wall,
                "repaired_columns": repaired,
                "output_sha256_storage_order": output_hash,
                "telemetry": records[0],
                "telemetry_status": status,
                "telemetry_gate": telemetry_gate,
                "native_gemm_output_numa_evidence": output_records[0],
                "native_gemm_output_numa_reset_status": output_reset_status,
                "native_gemm_output_numa_status": output_status,
                "native_gemm_output_numa_gate": native_numa_gate["output"],
                "native_integrity_snapshot_numa_evidence": records[0][
                    "native_integrity_snapshot_numa"
                ],
                "native_integrity_snapshot_numa_gate": native_numa_gate[
                    "integrity_snapshot"
                ],
                "protected_call_native_numa_gate": native_numa_gate,
            }
        )
        if output is not baseline:
            del output
    assert baseline is not None and baseline_hash is not None
    oracle = _full_oracle(case, left, right, baseline, np)
    inputs_after = {"left": _array_record(left), "right": _array_record(right)}
    if inputs_after != inputs_before:
        raise RuntimeError("GEMM inputs changed across benchmark/oracle calls")
    if _sha256(module_path) != args.expected_native_sha256:
        raise RuntimeError("native module changed during worker execution")
    if _sha256(archive_path) != args.expected_archive_sha256:
        raise RuntimeError("private archive changed during worker execution")
    package_after = _package_identity(prefix)
    if package_after != package_before:
        raise RuntimeError("installed package changed during worker execution")
    measured_calls = [item for item in calls if item["phase"] == "measured"]
    native_wall = [float(item["telemetry"]["wall_seconds"]) for item in measured_calls]
    native_cpu = [
        float(item["telemetry"]["process_cpu_seconds"]) for item in measured_calls
    ]
    active = [
        float(item["telemetry"]["active_core_equivalents"]) for item in measured_calls
    ]
    rates = [float(item["telemetry"]["gflops_per_second"]) for item in measured_calls]
    return {
        "status": "accepted",
        "accepted": True,
        "case_id": case.case_id,
        "case": asdict(case),
        "cblas": case.cblas,
        "flops_per_call": case.flops,
        "panel_columns": case.panel_columns,
        "protocol": {
            "warmups": WARMUPS,
            "measured_repeats": MEASURED_REPEATS,
            "fresh_python_no_site": True,
            "zero_repairs_all_calls": True,
            "bitwise_deterministic_all_calls": True,
            "full_oracle_after_measured_calls": True,
        },
        "pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "inserted_import_paths": inserted,
        "numpy_module": _file_identity(numpy_path),
        "thread_environment": environment,
        "cpu_contract": cpu,
        "early_numa_attestation": numa_attestation,
        "openmp_placement_attestation": placement,
        "native": native,
        "package_identity": package_before,
        "private_archive": _file_identity(archive_path),
        "memory_model": memory,
        "inputs": inputs_before,
        "input_numa_bound_buffers": input_numa,
        "input_numa_bound_buffers_gate": input_numa_gate,
        "output": _array_record(baseline),
        "calls": calls,
        "oracle": oracle,
        "measured_summary": {
            "vendor_wall_seconds": _summary(native_wall),
            "vendor_process_cpu_seconds": _summary(native_cpu),
            "active_core_equivalents": _summary(active),
            "gflops_per_second": _summary(rates),
            "cumulative_vendor_wall_seconds": math.fsum(native_wall),
            "cumulative_vendor_process_cpu_seconds": math.fsum(native_cpu),
            "matrix_minutes": math.fsum(native_wall) / 60.0,
            "process_cpu_minutes": math.fsum(native_cpu) / 60.0,
        },
    }


def _worker_command(
    args: argparse.Namespace,
    case: Case,
    dependencies: Sequence[Path],
    taskset: Path,
) -> list[str]:
    command = [
        str(taskset),
        "-c",
        _compress_ints(args.cpus),
        str(args.python_executable.resolve()),
        "-S",
        str(Path(__file__).resolve()),
        "--_worker",
        "--_operation",
        case.operation,
        "--install-prefix",
        str(args.install_prefix.resolve()),
        "--native-module",
        str(args.native_module.resolve()),
        "--expected-native-sha256",
        args.expected_native_sha256,
        "--expected-package-manifest-sha256",
        args.expected_package_manifest_sha256,
        "--private-archive",
        str(args.private_archive.resolve()),
        "--expected-archive-sha256",
        args.expected_archive_sha256,
        "--expected-source-commit",
        args.expected_source_commit,
        "--expected-source-tree-sha256",
        args.expected_source_tree_sha256,
        "--expected-backend",
        args.expected_backend,
        "--python-executable",
        str(args.python_executable.resolve()),
        "--cpus",
        _compress_ints(args.cpus),
        "--n",
        str(args.n),
        "--block-width",
        str(args.block_width),
        "--probe-tile",
        str(args.probe_tile),
        "--environment-tile",
        str(args.environment_tile),
        "--threads",
        str(args.threads),
        "--blis-thread-strategy",
        args.blis_thread_strategy,
        "--case-timeout-seconds",
        str(args.case_timeout_seconds),
        "--sweep-timeout-seconds",
        str(args.sweep_timeout_seconds),
        "--max-memory-gib",
        str(args.max_memory_gib),
        "--seed",
        str(args.seed),
        "--output",
        str(args.output.resolve()),
    ]
    if args.expected_private_source_commit is not None:
        command.extend(
            ["--expected-private-source-commit", args.expected_private_source_commit]
        )
    if args.expected_private_source_tree_sha256 is not None:
        command.extend(
            [
                "--expected-private-source-tree-sha256",
                args.expected_private_source_tree_sha256,
            ]
        )
    if args.blis_thread_ways is not None:
        rendered = ",".join(
            f"{name}={args.blis_thread_ways[name]}" for name in BLIS_WAY_NAMES
        )
        command.extend(["--blis-thread-ways", rendered])
    for dependency in dependencies:
        command.extend(["--dependency-path", str(dependency)])
    return command


def _kill_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        return process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return process.communicate()


def _parse_worker_result(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"worker emitted {len(lines)} result records")
    value = json.loads(lines[0][len(RESULT_PREFIX) :])
    if not isinstance(value, dict):
        raise RuntimeError("worker result is not a JSON object")
    return value


def _revalidate_worker_native_numa_evidence(
    result: Mapping[str, Any],
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    native = result.get("native")
    build = native.get("build_info") if isinstance(native, Mapping) else None
    if (
        not isinstance(native, Mapping)
        or native.get("module_sha256") != args.expected_native_sha256
        or not isinstance(build, Mapping)
    ):
        raise RuntimeError("worker lacks exact native module/build evidence")
    _validate_native_numa_build_contract(build)
    cpu = _cpu_contract(args.cpus)
    if result.get("cpu_contract") != cpu:
        raise RuntimeError("worker CPU contract differs from controller reconstruction")
    nodes = cpu["numa_nodes"]
    calls = result.get("calls")
    if not isinstance(calls, list) or len(calls) != WARMUPS + MEASURED_REPEATS:
        raise RuntimeError("worker protected-call count is not exact")
    for ordinal, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise RuntimeError("worker protected-call evidence is malformed")
        measured = ordinal >= WARMUPS
        expected_phase = {
            "ordinal": ordinal,
            "phase": "measured" if measured else "warmup",
            "phase_ordinal": ordinal - WARMUPS if measured else ordinal,
        }
        if any(call.get(name) != value for name, value in expected_phase.items()):
            raise RuntimeError("worker protected-call ordinal/phase mismatch")
        telemetry = call.get("telemetry")
        telemetry_status = call.get("telemetry_status")
        output_evidence = call.get("native_gemm_output_numa_evidence")
        reset_status = call.get("native_gemm_output_numa_reset_status")
        post_status = call.get("native_gemm_output_numa_status")
        if not all(
            isinstance(value, Mapping)
            for value in (
                telemetry,
                telemetry_status,
                output_evidence,
                reset_status,
                post_status,
            )
        ):
            raise RuntimeError("worker call lacks retained native NUMA evidence")
        sequence = telemetry.get("sequence")
        if type(sequence) is not int or sequence != ordinal + 1:
            raise RuntimeError("worker telemetry sequence is not exact")
        _validate_telemetry_status(telemetry_status, observed_sequence=sequence)
        telemetry_gate = _validate_telemetry(
            telemetry,
            case,
            args.cpus,
            nodes,
            build,
            measured=measured,
        )
        if call.get("telemetry_gate") != telemetry_gate:
            raise RuntimeError(
                "worker telemetry gate differs from controller reconstruction"
            )
        native_gate = _validate_protected_call_native_numa_contract(
            telemetry=telemetry,
            output_evidence=output_evidence,
            reset_status=reset_status,
            post_status=post_status,
            case=case,
            nodes=nodes,
            expected_call_id=ordinal + 1,
            integrity_minimum_vendor_flops=build[
                "gemm_integrity_minimum_vendor_flops"
            ],
        )
        if (
            call.get("native_gemm_output_numa_gate") != native_gate["output"]
            or call.get("native_integrity_snapshot_numa_evidence")
            != telemetry.get("native_integrity_snapshot_numa")
            or call.get("native_integrity_snapshot_numa_gate")
            != native_gate["integrity_snapshot"]
            or call.get("protected_call_native_numa_gate") != native_gate
        ):
            raise RuntimeError(
                "worker native NUMA gate differs from controller reconstruction"
            )
    return {
        "passed": True,
        "protected_call_count": len(calls),
        "controller_revalidated": True,
    }


def _run_case(
    args: argparse.Namespace,
    case: Case,
    dependencies: Sequence[Path],
    taskset: Path,
    timeout: float,
) -> dict[str, Any]:
    command = _worker_command(args, case, dependencies, taskset)
    environment, _settings = _worker_environment(args)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_process_group(process)
        return {
            "case_id": case.case_id,
            "case": asdict(case),
            "status": "timeout",
            "accepted": False,
            "pid": process.pid,
            "timeout_seconds": timeout,
            "controller_wall_seconds": time.monotonic() - started,
            "command": command,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
        }
    except BaseException:
        if process.poll() is None:
            _kill_process_group(process)
        raise
    base = {
        "case_id": case.case_id,
        "case": asdict(case),
        "pid": process.pid,
        "returncode": process.returncode,
        "controller_wall_seconds": time.monotonic() - started,
        "command": command,
        "stderr_tail": stderr[-4000:],
    }
    if process.returncode != 0:
        return {
            **base,
            "status": "failed",
            "accepted": False,
            "stdout_tail": stdout[-4000:],
        }
    try:
        result = _parse_worker_result(stdout)
    except (RuntimeError, json.JSONDecodeError) as error:
        return {
            **base,
            "status": "invalid_result",
            "accepted": False,
            "reason": str(error),
        }
    if (
        result.get("pid") != process.pid
        or result.get("case_id") != case.case_id
        or result.get("status") != "accepted"
        or result.get("accepted") is not True
    ):
        return {
            **base,
            "status": "invalid_result",
            "accepted": False,
            "reason": "worker identity/status mismatch",
        }
    try:
        controller_native_numa_gate = _revalidate_worker_native_numa_evidence(
            result, case, args
        )
    except RuntimeError as error:
        return {
            **base,
            "status": "invalid_result",
            "accepted": False,
            "reason": str(error),
        }
    result["controller_native_numa_gate"] = controller_native_numa_gate
    return {**base, **result}


def _static_provenance(args: argparse.Namespace) -> dict[str, Any]:
    from shutil import which

    prefix = args.install_prefix.resolve()
    module = args.native_module.resolve()
    archive = args.private_archive.resolve()
    python = _regular_file(args.python_executable, executable=True)
    taskset_raw = which("taskset")
    if taskset_raw is None:
        raise RuntimeError("taskset is required")
    taskset = _regular_file(Path(taskset_raw), executable=True)
    package = _package_identity(prefix)
    if package["manifest_sha256"] != args.expected_package_manifest_sha256:
        raise RuntimeError("package manifest changed during static preflight")
    module_identity = _file_identity(module)
    archive_identity = _file_identity(archive)
    if module_identity["sha256"] != args.expected_native_sha256:
        raise RuntimeError("native module changed during static preflight")
    if archive_identity["sha256"] != args.expected_archive_sha256:
        raise RuntimeError("private archive changed during static preflight")
    return {
        "installed_package": package,
        "native_module": module_identity,
        "private_archive": archive_identity,
        "expected_native_sha256": args.expected_native_sha256,
        "expected_package_manifest_sha256": args.expected_package_manifest_sha256,
        "expected_archive_sha256": args.expected_archive_sha256,
        "expected_source_commit": args.expected_source_commit,
        "expected_source_tree_sha256": args.expected_source_tree_sha256,
        "expected_private_source_commit": args.expected_private_source_commit,
        "expected_private_source_tree_sha256": (
            args.expected_private_source_tree_sha256
        ),
        "expected_backend": args.expected_backend,
        "linkage": _linkage_evidence(module, archive, args.expected_backend),
        "runner": _file_identity(Path(__file__).resolve()),
        "python_executable": _file_identity(python),
        "taskset_executable": _file_identity(taskset),
        "cpu_contract": _cpu_contract(args.cpus),
        "thread_environment": _thread_environment(args),
        "environment_persistence_policy": {
            "whole_environment_persisted": False,
            "allowlisted_thread_settings_only": True,
            "authentication_tokens_persisted": False,
        },
    }


def _base_report(
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    dependencies: Sequence[Path],
) -> dict[str, Any]:
    cases = _cases(args)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_only": True,
        "production_run": False,
        "host": {
            "hostname": socket.gethostname(),
            "uname": list(os.uname()),
            "controller_python": sys.version,
            "controller_affinity": _compress_ints(sorted(os.sched_getaffinity(0))),
        },
        "protocol": {
            "warmups_per_case": WARMUPS,
            "measured_repeats_per_case": MEASURED_REPEATS,
            "case_timeout_seconds": args.case_timeout_seconds,
            "maximum_case_timeout_seconds": MAX_CASE_SECONDS,
            "sweep_timeout_seconds": args.sweep_timeout_seconds,
            "maximum_sweep_timeout_seconds": MAX_SWEEP_SECONDS,
            "fresh_python_no_site_per_case": True,
            "cases_sequential": True,
            "private_static_blas_required": True,
            "integrity_enabled_required": True,
            "zero_repairs_required": True,
            "full_oracle_required": True,
            "bitwise_determinism_required": True,
            "minimum_active_core_fraction_per_measured_call": (
                MINIMUM_ACTIVE_CORE_FRACTION
            ),
            "early_verified_membind_required": True,
            "fully_contained_move_pages_locality_required": True,
            "native_integrity_snapshot_numa_contract_required": True,
            "native_protected_output_numa_contract_required": True,
            "one_output_evidence_record_per_protected_call": True,
            "controller_revalidates_worker_native_numa_evidence": True,
            "post_vendor_abc_sampler_remains_independent": True,
            "api9_exact_openmp_singleton_placement_required": True,
            "atomic_no_replace_publication": True,
        },
        "selection": {
            "current_layout_only": True,
            "source_orientation": "column_major_nn",
            "target_orientation": "column_major_tn",
            "n_samples": args.n,
            "block_width": args.block_width,
            "probe_tile": args.probe_tile,
            "environment_tile": args.environment_tile,
            "threads": args.threads,
            "backend": args.expected_backend,
            "blis_thread_strategy": (
                args.blis_thread_strategy if args.expected_backend == "blis" else None
            ),
            "blis_thread_ways": args.blis_thread_ways,
            "case_ids": [case.case_id for case in cases],
        },
        "dependency_paths": [str(path) for path in dependencies],
        "provenance": dict(provenance),
        "cases": [],
    }


def _dry_run_report(
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    dependencies: Sequence[Path],
    taskset: Path,
) -> dict[str, Any]:
    report = _base_report(args, provenance, dependencies)
    report.update(
        {
            "dry_run": True,
            "status": "validated_dry_run",
            "accepted": False,
            "acceptance_reason": "dry_run_performs_no_scientific_execution",
            "scientific_execution": False,
        }
    )
    report["cases"] = [
        {
            "case_id": case.case_id,
            "case": asdict(case),
            "cblas": case.cblas,
            "flops_per_call": case.flops,
            "panel_columns": case.panel_columns,
            "memory_model": _estimated_memory(case),
            "status": "planned",
            "command": _worker_command(args, case, dependencies, taskset),
        }
        for case in _cases(args)
    ]
    return report


def _controller(
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    dependencies: Sequence[Path],
    taskset: Path,
) -> dict[str, Any]:
    report = _base_report(args, provenance, dependencies)
    report["dry_run"] = False
    report["scientific_execution"] = True
    started = time.monotonic()
    memory_limit = int(args.max_memory_gib * 1024**3)
    for case in _cases(args):
        memory = _estimated_memory(case)
        if memory["conservative_peak_bytes"] > memory_limit:
            report["cases"].append(
                {
                    "case_id": case.case_id,
                    "case": asdict(case),
                    "status": "skipped_memory_budget",
                    "accepted": False,
                    "memory_model": memory,
                    "memory_limit_bytes": memory_limit,
                }
            )
            continue
        remaining = args.sweep_timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            report["cases"].append(
                {
                    "case_id": case.case_id,
                    "case": asdict(case),
                    "status": "skipped_sweep_budget",
                    "accepted": False,
                }
            )
            continue
        result = _run_case(
            args,
            case,
            dependencies,
            taskset,
            min(args.case_timeout_seconds, remaining),
        )
        report["cases"].append(result)
    report["sweep_wall_seconds"] = time.monotonic() - started
    package_after = _package_identity(args.install_prefix.resolve())
    artifact_stable = (
        package_after == provenance["installed_package"]
        and _file_identity(args.native_module.resolve()) == provenance["native_module"]
        and _file_identity(args.private_archive.resolve())
        == provenance["private_archive"]
        and _file_identity(Path(__file__).resolve()) == provenance["runner"]
        and _file_identity(args.python_executable.resolve())
        == provenance["python_executable"]
        and _file_identity(Path(provenance["taskset_executable"]["path"]))
        == provenance["taskset_executable"]
    )
    build_hashes = {
        str(case.get("native", {}).get("build_info_canonical_json_sha256", ""))
        for case in report["cases"]
        if case.get("status") == "accepted"
    }
    all_accepted = (
        len(report["cases"]) == 2
        and all(case.get("accepted") is True for case in report["cases"])
        and artifact_stable
        and len(build_hashes) == 1
        and report["sweep_wall_seconds"] <= args.sweep_timeout_seconds
    )
    report.update(
        {
            "artifact_identity_stable": artifact_stable,
            "native_build_identity_count": len(build_hashes),
            "status": "accepted" if all_accepted else "completed_rejected",
            "accepted": all_accepted,
            "acceptance_reason": (
                "both exact current-layout cases passed every gate"
                if all_accepted
                else "one or more required benchmark gates failed"
            ),
        }
    )
    return report


def _atomic_json_no_replace(payload: Mapping[str, Any], output: Path) -> None:
    lexical = output.expanduser().absolute()
    if lexical.is_symlink() or lexical.exists():
        raise FileExistsError(f"refusing existing report: {lexical}")
    parent = lexical.parent.resolve(strict=True)
    target = parent / lexical.name
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"refusing existing report: {target}")
    rendered = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
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
            result = _run_worker(args)
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(
            RESULT_PREFIX
            + json.dumps(_json_safe(result), sort_keys=True, allow_nan=False)
        )
        return 0
    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        parser.error("POSIX real-time deadlines are required")
    previous_handler = signal.signal(signal.SIGALRM, _raise_sweep_deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, args.sweep_timeout_seconds)
    try:
        dependencies = _dependency_paths(args)
        provenance = _static_provenance(args)
        taskset = Path(provenance["taskset_executable"]["path"])
        payload = (
            _dry_run_report(args, provenance, dependencies, taskset)
            if args.dry_run
            else _controller(args, provenance, dependencies, taskset)
        )
        _atomic_json_no_replace(payload, args.output)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
    print(args.output.expanduser().resolve())
    return 0 if args.dry_run or payload.get("accepted") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
