#!/usr/bin/env python3
"""Build and run the standalone SUMMIT GxE ceiling/orientation harness.

Every benchmark configuration is a fresh ``exec`` of a binary containing one
statically linked OpenBLAS image.  The controller pins that process to a set of
physical cores, caps each configuration at 90 seconds, and caps a sweep at 20
minutes.  It never imports NumPy or a process-shared BLAS.

The default ``smoke`` profile exercises the complete protocol (three warmups
and five timed repeats) with small operands.  The ``exact`` profile uses
N=289111, genotype block width 2000, probe tile 32, and environment tile 1;
use the individual selectors to keep production-shape sweeps staged and
pruned.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks" / "gxe" / "gxe_ceiling_orientation.cpp"
DEFAULT_ARCHIVE = Path(
    "/tmp/summit-openblas-audit.lyA9tE/OpenBLAS/"
    "libopenblas_zenp-r0.3.34.a"
)
DEFAULT_ARCHIVE_SHA256 = (
    "49609db51d9c91bb4beaa40af20ef6a57698698b8043b3d9ac15ffc25554f521"
)
MAX_CONFIGURATION_SECONDS = 90.0
MAX_SWEEP_SECONDS = 20.0 * 60.0
MIN_WARMUPS = 3
MIN_REPEATS = 5
SCHEMA = "summit.gxe.local_ceiling_orientation_sweep"
SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class Case:
    label: str
    mode: str
    dtype: str
    layout: str
    orientation: str
    threads: int
    size: int
    n_samples: int
    block_width: int
    probe_tile: int
    environment_tile: int
    stream_elements: int

    @property
    def panel_width(self) -> int:
        if self.mode == "source":
            return 2 * self.probe_tile * self.environment_tile
        if self.mode == "target":
            return 4 * self.probe_tile * self.environment_tile
        return 0

    def estimated_operand_bytes(self) -> int:
        itemsize = 8 if self.dtype == "f64" else 4
        if self.mode == "square":
            return 3 * self.size * self.size * itemsize
        if self.mode == "stream":
            return 3 * self.stream_elements * itemsize
        if self.mode == "source":
            return (
                self.n_samples * self.block_width
                + self.block_width * self.panel_width
                + self.n_samples * self.panel_width
            ) * itemsize
        if self.mode == "target":
            return (
                self.n_samples * self.block_width
                + self.n_samples * self.panel_width
                + self.block_width * self.panel_width
            ) * itemsize
        raise AssertionError(self.mode)

    def worker_arguments(self, warmups: int, repeats: int, seed: int) -> list[str]:
        return [
            "--mode", self.mode,
            "--dtype", self.dtype,
            "--layout", self.layout,
            "--orientation", self.orientation,
            "--size", str(self.size),
            "--n", str(self.n_samples),
            "--k", str(self.block_width),
            "--probe-tile", str(self.probe_tile),
            "--environment-tile", str(self.environment_tile),
            "--stream-elements", str(self.stream_elements),
            "--threads", str(self.threads),
            "--warmups", str(warmups),
            "--repeats", str(repeats),
            "--correctness-samples", "12",
            "--seed", str(seed),
        ]


def _csv_choices(value: str, allowed: set[str]) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(values).difference(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated subset of " + ", ".join(sorted(allowed))
        )
    return values


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all integers must be positive")
    return values


def _expand_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise argparse.ArgumentTypeError("invalid CPU range")
            cpus.extend(range(first, last + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise argparse.ArgumentTypeError("CPU IDs must be nonnegative")
            cpus.append(cpu)
    if not cpus or len(cpus) != len(set(cpus)):
        raise argparse.ArgumentTypeError("CPU list must be nonempty and unique")
    return cpus


def _compress_cpu_list(cpus: Sequence[int]) -> str:
    ordered = sorted(cpus)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "exact"), default="smoke")
    parser.add_argument(
        "--modes",
        type=lambda value: _csv_choices(
            value, {"dgemm", "sgemm", "stream", "source", "target"}
        ),
        default=("dgemm", "sgemm", "stream", "source", "target"),
    )
    parser.add_argument(
        "--layouts",
        type=lambda value: _csv_choices(value, {"col", "row"}),
        default=("col", "row"),
    )
    parser.add_argument(
        "--orientations",
        type=lambda value: _csv_choices(value, {"current", "transposed"}),
        default=("current", "transposed"),
    )
    parser.add_argument(
        "--exact-dtypes",
        type=lambda value: _csv_choices(value, {"f64", "f32"}),
        default=("f64",),
        help="Arithmetic/storage dtypes for source and target cases.",
    )
    parser.add_argument("--threads", type=_csv_positive_ints, default=(1,))
    parser.add_argument(
        "--cpus", type=_expand_cpu_list, default=None,
        help=(
            "Physical CPU IDs available to each worker.  A T-thread case uses "
            "the first T IDs.  By default the controller selects physical cores "
            "from its current affinity, excluding SMT siblings."
        ),
    )
    parser.add_argument("--size", type=int, default=None, help="Square GEMM order.")
    parser.add_argument("--n", type=int, default=None, help="Exact-shape sample count.")
    parser.add_argument("--k", type=int, default=None, help="Genotype block width.")
    parser.add_argument("--probe-tile", type=int, default=None)
    parser.add_argument("--environment-tile", type=int, default=None)
    parser.add_argument("--stream-elements", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=MIN_WARMUPS)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--max-configuration-seconds", type=float,
        default=MAX_CONFIGURATION_SECONDS,
    )
    parser.add_argument("--max-sweep-seconds", type=float, default=MAX_SWEEP_SECONDS)
    parser.add_argument(
        "--max-memory-gib", type=float, default=None,
        help="Skip a configuration above this conservative operand limit.",
    )
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--expected-archive-sha256", default=DEFAULT_ARCHIVE_SHA256,
        help="Required archive identity; set explicitly for another candidate backend.",
    )
    parser.add_argument("--include-dir", type=Path, default=None)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument(
        "--build-dir", type=Path, default=None,
        help="Required for an actual run; no build output is placed in the repository.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _profile_values(args: argparse.Namespace) -> dict[str, int | float]:
    if args.profile == "exact":
        defaults: dict[str, int | float] = {
            "size": 8192,
            "n_samples": 289_111,
            "block_width": 2000,
            "probe_tile": 32,
            "environment_tile": 1,
            "stream_elements": 32 * 1024 * 1024,
            "max_memory_gib": 96.0,
        }
    else:
        defaults = {
            "size": 512,
            "n_samples": 2048,
            "block_width": 256,
            "probe_tile": 4,
            "environment_tile": 1,
            "stream_elements": 4 * 1024 * 1024,
            "max_memory_gib": 2.0,
        }
    return {
        "size": args.size if args.size is not None else defaults["size"],
        "n_samples": args.n if args.n is not None else defaults["n_samples"],
        "block_width": args.k if args.k is not None else defaults["block_width"],
        "probe_tile": (
            args.probe_tile if args.probe_tile is not None else defaults["probe_tile"]
        ),
        "environment_tile": (
            args.environment_tile
            if args.environment_tile is not None
            else defaults["environment_tile"]
        ),
        "stream_elements": (
            args.stream_elements
            if args.stream_elements is not None
            else defaults["stream_elements"]
        ),
        "max_memory_gib": (
            args.max_memory_gib
            if args.max_memory_gib is not None
            else defaults["max_memory_gib"]
        ),
    }


def _cases(args: argparse.Namespace) -> list[Case]:
    values = _profile_values(args)
    common = {
        "size": int(values["size"]),
        "n_samples": int(values["n_samples"]),
        "block_width": int(values["block_width"]),
        "probe_tile": int(values["probe_tile"]),
        "environment_tile": int(values["environment_tile"]),
        "stream_elements": int(values["stream_elements"]),
    }
    result: list[Case] = []
    for threads in args.threads:
        if "dgemm" in args.modes:
            result.append(Case(f"dgemm.col.T{threads}", "square", "f64", "col", "current", threads, **common))
        if "sgemm" in args.modes:
            result.append(Case(f"sgemm.col.T{threads}", "square", "f32", "col", "current", threads, **common))
        if "stream" in args.modes:
            result.append(Case(f"stream.f64.T{threads}", "stream", "f64", "col", "current", threads, **common))
        for mode in ("source", "target"):
            if mode not in args.modes:
                continue
            for dtype in args.exact_dtypes:
                for layout in args.layouts:
                    for orientation in args.orientations:
                        label = f"{mode}.{dtype}.{layout}.{orientation}.T{threads}"
                        result.append(Case(label, mode, dtype, layout, orientation, threads, **common))
    return result


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.max_configuration_seconds <= 0 or args.max_configuration_seconds > MAX_CONFIGURATION_SECONDS:
        parser.error("--max-configuration-seconds must be in (0, 90]")
    if args.max_sweep_seconds <= 0 or args.max_sweep_seconds > MAX_SWEEP_SECONDS:
        parser.error("--max-sweep-seconds must be in (0, 1200]")
    if args.warmups < MIN_WARMUPS or args.repeats < MIN_REPEATS:
        parser.error("the protocol requires at least 3 warmups and 5 timed repeats")
    if any(value <= 0 for value in (
        args.size or 1, args.n or 1, args.k or 1, args.probe_tile or 1,
        args.environment_tile or 1, args.stream_elements or 1,
        args.max_cases,
    )):
        parser.error("shape values and --max-cases must be positive")
    cases = _cases(args)
    if len(cases) > args.max_cases:
        parser.error(f"refusing {len(cases)} cases; raise --max-cases explicitly")
    if not args.dry_run and args.build_dir is None:
        parser.error("--build-dir is required unless --dry-run is used")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _resolved_include_dir(args: argparse.Namespace) -> Path:
    if args.include_dir is not None:
        return args.include_dir
    candidates = [
        Path(sys.executable).resolve().parent.parent / "include",
        Path.home() / "anaconda3" / "envs" / "summit" / "include",
        Path("/usr/include/x86_64-linux-gnu"),
        Path("/usr/include"),
    ]
    for candidate in candidates:
        if (candidate / "cblas.h").is_file():
            return candidate
    raise RuntimeError("could not locate a public CBLAS include directory")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _physical_core_key(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    package = _read_int(topology / "physical_package_id")
    core = _read_int(topology / "core_id")
    if package is None or core is None:
        return (0, cpu)
    return package, core


def _default_physical_cpus() -> list[int]:
    available = sorted(os.sched_getaffinity(0))
    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    for cpu in available:
        key = _physical_core_key(cpu)
        if key in seen:
            continue
        seen.add(key)
        selected.append(cpu)
    return selected


def _validate_cpu_pool(cpus: Sequence[int], maximum_threads: int) -> None:
    available = os.sched_getaffinity(0)
    unavailable = sorted(set(cpus).difference(available))
    if unavailable:
        raise RuntimeError(f"requested CPUs are outside current affinity: {unavailable}")
    if len(cpus) < maximum_threads:
        raise RuntimeError(
            f"need at least {maximum_threads} physical CPU IDs, received {len(cpus)}"
        )
    keys = [_physical_core_key(cpu) for cpu in cpus]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise RuntimeError(f"CPU pool contains SMT siblings for cores {duplicates}")


def _compiler_command(
    args: argparse.Namespace, executable: Path, archive_sha256: str,
    source_sha256: str, archive_capabilities: dict[str, bool],
) -> list[str]:
    include_dir = _resolved_include_dir(args)
    command = [
        args.compiler,
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-fno-omit-frame-pointer",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        f"-I{include_dir}",
        f'-DSUMMIT_OPENBLAS_ARCHIVE="{args.archive.resolve()}"',
        f'-DSUMMIT_OPENBLAS_ARCHIVE_SHA256="{archive_sha256}"',
        f'-DSUMMIT_BENCHMARK_SOURCE_SHA256="{source_sha256}"',
    ]
    if archive_capabilities["cblas_sgemm"]:
        command.append("-DSUMMIT_OPENBLAS_HAS_SGEMM=1")
    command.extend([
        str(SOURCE),
        str(args.archive.resolve()),
        "-Wl,--exclude-libs,ALL",
        "-Wl,-z,defs",
        "-pthread",
        "-ldl",
        "-lm",
        "-o",
        str(executable),
    ])
    return command


def _archive_capabilities(archive: Path) -> dict[str, bool]:
    nm = shutil.which("nm")
    if nm is None:
        raise RuntimeError("nm is required to inspect the static BLAS archive")
    symbols = _run_checked([nm, "-g", "--defined-only", str(archive)], 30.0).stdout
    return {
        name: re.search(rf"\b{name}$", symbols, flags=re.MULTILINE) is not None
        for name in (
            "cblas_dgemm",
            "cblas_sgemm",
            "openblas_get_config",
            "openblas_get_corename",
            "openblas_get_parallel",
            "openblas_get_num_threads",
        )
    }


def _run_checked(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _linkage_evidence(executable: Path) -> dict[str, object]:
    readelf = shutil.which("readelf")
    nm = shutil.which("nm")
    if readelf is None or nm is None:
        raise RuntimeError("readelf and nm are required to verify private static linkage")
    dynamic = _run_checked([readelf, "-d", str(executable)], 20.0).stdout
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    dynamic_symbols = _run_checked([nm, "-D", str(executable)], 20.0).stdout
    blas_dependencies = [name for name in needed if "blas" in name.lower()]
    exported_blas_symbols = [
        line for line in dynamic_symbols.splitlines()
        if re.search(r"\b(?:cblas_|openblas_)", line)
    ]
    if blas_dependencies:
        raise RuntimeError(f"dynamic BLAS dependency defeats isolation: {blas_dependencies}")
    if exported_blas_symbols:
        raise RuntimeError("OpenBLAS symbols escaped into the dynamic symbol table")
    return {
        "needed_shared_libraries": needed,
        "dynamic_blas_dependencies": blas_dependencies,
        "exported_blas_symbol_count": len(exported_blas_symbols),
        "private_static_link_verified": True,
    }


def _worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": str(threads),
        "GOTO_NUM_THREADS": str(threads),
    })
    return environment


def _kill_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        # The worker contains only benchmark-owned threads and has no output to
        # publish transactionally.  Kill immediately at the hard deadline so a
        # grace interval cannot extend a nominal 90-second configuration.
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.communicate()


def _run_case(
    executable: Path, case: Case, cpus: Sequence[int], args: argparse.Namespace,
    timeout_seconds: float,
) -> dict[str, object]:
    selected_cpus = list(cpus[: case.threads])
    command = [
        "taskset", "-c", _compress_cpu_list(selected_cpus), str(executable),
        *case.worker_arguments(args.warmups, args.repeats, args.seed),
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_environment(case.threads),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        stdout, stderr = _kill_process_group(process)
        return {
            "case": dataclasses.asdict(case),
            "status": "timeout",
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "command": command,
            "selected_physical_cpus": selected_cpus,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
    base: dict[str, object] = {
        "case": dataclasses.asdict(case),
        "elapsed_seconds": time.monotonic() - started,
        "command": command,
        "selected_physical_cpus": selected_cpus,
        "returncode": process.returncode,
        "stderr": stderr,
    }
    if process.returncode != 0:
        base.update({"status": "failed", "stdout_tail": stdout[-4000:]})
        return base
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        base.update({
            "status": "invalid_json",
            "json_error": str(error),
            "stdout_tail": stdout[-4000:],
        })
        return base
    base.update({"status": "ok", "result": payload})
    return base


def _dry_run_report(args: argparse.Namespace, cases: Sequence[Case]) -> dict[str, object]:
    values = _profile_values(args)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "profile": args.profile,
        "limits": {
            "per_configuration_seconds": args.max_configuration_seconds,
            "sweep_seconds": args.max_sweep_seconds,
            "minimum_warmups": MIN_WARMUPS,
            "minimum_timed_repeats": MIN_REPEATS,
        },
        "resolved_profile": values,
        "case_count": len(cases),
        "cases": [
            {
                **dataclasses.asdict(case),
                "panel_width": case.panel_width,
                "estimated_operand_bytes": case.estimated_operand_bytes(),
                "worker_arguments": case.worker_arguments(
                    args.warmups, args.repeats, args.seed
                ),
            }
            for case in cases
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(args, parser)
    cases = _cases(args)
    if args.dry_run:
        print(json.dumps(_dry_run_report(args, cases), indent=2, sort_keys=True))
        return 0

    assert args.build_dir is not None
    assert args.output is not None
    if not SOURCE.is_file():
        parser.error(f"benchmark source does not exist: {SOURCE}")
    if not args.archive.is_file():
        parser.error(f"OpenBLAS archive does not exist: {args.archive}")
    include_dir = _resolved_include_dir(args)
    if not (include_dir / "cblas.h").is_file():
        parser.error(f"cblas.h does not exist in {include_dir}")
    archive_sha256 = _sha256(args.archive)
    if archive_sha256 != args.expected_archive_sha256:
        parser.error(
            "OpenBLAS archive SHA256 mismatch: "
            f"expected {args.expected_archive_sha256}, observed {archive_sha256}"
        )
    source_sha256 = _sha256(SOURCE)
    cblas_header_sha256 = _sha256(include_dir / "cblas.h")
    archive_capabilities = _archive_capabilities(args.archive)
    required_symbols = (
        "cblas_dgemm",
        "openblas_get_config",
        "openblas_get_corename",
        "openblas_get_parallel",
        "openblas_get_num_threads",
    )
    missing_required = [
        name for name in required_symbols if not archive_capabilities[name]
    ]
    if missing_required:
        parser.error(f"archive lacks required symbols: {missing_required}")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    executable = args.build_dir / "gxe_ceiling_orientation"
    compile_command = _compiler_command(
        args, executable, archive_sha256, source_sha256, archive_capabilities
    )
    compile_started = time.monotonic()
    compile_result = _run_checked(compile_command, 120.0)
    compile_seconds = time.monotonic() - compile_started
    compiler_version = _run_checked([args.compiler, "--version"], 20.0).stdout
    linkage = _linkage_evidence(executable)

    cpu_pool = args.cpus if args.cpus is not None else _default_physical_cpus()
    _validate_cpu_pool(cpu_pool, max(args.threads))
    maximum_bytes = int(float(_profile_values(args)["max_memory_gib"]) * 1024**3)

    sweep_started = time.monotonic()
    results: list[dict[str, object]] = []
    for case in cases:
        elapsed = time.monotonic() - sweep_started
        remaining = args.max_sweep_seconds - elapsed
        if remaining <= 0.5:
            results.append({
                "case": dataclasses.asdict(case),
                "status": "not_started_sweep_budget_exhausted",
            })
            continue
        estimated_bytes = case.estimated_operand_bytes()
        required_bytes = (estimated_bytes * 120 + 99) // 100
        if required_bytes > maximum_bytes:
            results.append({
                "case": dataclasses.asdict(case),
                "status": "skipped_memory_bound",
                "estimated_operand_bytes": estimated_bytes,
                "required_bytes_with_20_percent_headroom": required_bytes,
                "maximum_operand_bytes": maximum_bytes,
            })
            continue
        if case.dtype == "f32" and not archive_capabilities["cblas_sgemm"]:
            results.append({
                "case": dataclasses.asdict(case),
                "status": "unsupported_backend_capability",
                "missing_symbol": "cblas_sgemm",
                "archive_sha256": archive_sha256,
            })
            continue
        results.append(_run_case(
            executable, case, cpu_pool, args,
            min(args.max_configuration_seconds, remaining - 0.5),
        ))
    sweep_seconds = time.monotonic() - sweep_started
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": args.profile,
        "limits": {
            "per_configuration_seconds": args.max_configuration_seconds,
            "sweep_seconds": args.max_sweep_seconds,
            "minimum_warmups": MIN_WARMUPS,
            "minimum_timed_repeats": MIN_REPEATS,
        },
        "provenance": {
            "source": str(SOURCE),
            "source_sha256": source_sha256,
            "cblas_include_dir": str(include_dir.resolve()),
            "cblas_header_sha256": cblas_header_sha256,
            "archive": str(args.archive.resolve()),
            "archive_sha256": archive_sha256,
            "archive_capabilities": archive_capabilities,
            "compiler": args.compiler,
            "compiler_version": compiler_version,
            "compiler_command": compile_command,
            "compiler_stdout": compile_result.stdout,
            "compiler_stderr": compile_result.stderr,
            "compile_seconds": compile_seconds,
            "executable": str(executable),
            "executable_sha256": _sha256(executable),
            "linkage": linkage,
        },
        "placement": {
            "physical_cpu_pool": cpu_pool,
            "physical_core_keys": [list(_physical_core_key(cpu)) for cpu in cpu_pool],
        },
        "resolved_profile": _profile_values(args),
        "sweep_seconds": sweep_seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(str(args.output))
    failure_statuses = {"failed", "timeout", "invalid_json"}
    return 2 if any(item.get("status") in failure_statuses for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
