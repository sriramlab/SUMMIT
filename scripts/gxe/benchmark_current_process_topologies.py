#!/usr/bin/env python3
"""Screen exact current-layout GEMMs across safe process topologies.

The controller is deliberately standard-library-only.  Each numerical worker is
a fresh ``python -S`` process with a private backend image, an exact API-9
singleton-place contract, and early verified NUMA binding.  Workers rendezvous
before every one of three warmups and five measured calls.  This is a bounded
benchmark harness, not a production reference runner; dry runs are never
accepted as scientific evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


SCHEMA = "summit.gxe.current_process_topology_benchmark"
SCHEMA_VERSION = 1
WORKER_RESULT_PREFIX = "SUMMIT_GXE_PROCESS_TOPOLOGY_RESULT="
MAX_CONFIGURATION_SECONDS = 90.0
MAX_SWEEP_SECONDS = 20.0 * 60.0
MAX_START_SKEW_SECONDS = 0.25
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.10
CLEANUP_TERM_WAIT_SECONDS = 2.0
CLEANUP_KILL_WAIT_SECONDS = 2.0
FAILURE_STDOUT_TAIL_CHARACTERS = 4_000
FAILURE_STDERR_TAIL_CHARACTERS = 4_000
FAILURE_EVENT_STREAM_TAIL_BYTES = 8_192
FAILURE_EVENT_DRAIN_MAX_BYTES = 65_536
FAILURE_EVENT_DRAIN_MAX_SECONDS = 0.05
MAX_UNPARSED_EVENT_BUFFER_BYTES = 65_536
TERMINATE_PROCESS_GROUP_SIGNAL = 15
KILL_PROCESS_GROUP_SIGNAL = 9
WATCHDOG_JOIN_TIMEOUT_SECONDS = 1.0
WATCHDOG_WAKE_TOKEN = b"D"


def _load_exact_harness() -> Any:
    path = Path(__file__).with_name("benchmark_current_layout_backends.py").resolve()
    name = "_summit_gxe_exact_current_layout_harness"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("exact-layout harness module identity collision")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact-layout harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("loaded exact-layout harness differs from sibling source")
    return module


EXACT = _load_exact_harness()


@dataclass(frozen=True)
class Topology:
    topology_id: str
    threads_per_process: int
    cpu_groups: tuple[tuple[int, ...], ...]
    cpu_contracts: tuple[dict[str, Any], ...]
    environment_tiles: tuple[int, ...]
    production_promotable: bool
    non_promotable_reason: str | None

    @property
    def process_count(self) -> int:
        return len(self.cpu_groups)

    @property
    def total_physical_cores(self) -> int:
        return self.process_count * self.threads_per_process


@dataclass(frozen=True)
class Configuration:
    topology: Topology
    operation: str

    @property
    def config_id(self) -> str:
        return f"{self.topology.topology_id}.{self.operation}"


class _SweepDeadlineExpired(TimeoutError):
    pass


class _SweepDeadlineWatchdog:
    """Latch one absolute CLOCK_MONOTONIC deadline and wake selectors."""

    _ARMED = "armed"
    _EXPIRED = "expired"
    _DISARMED = "disarmed"

    def __init__(self, started_ns: int, timeout_seconds: float) -> None:
        if type(started_ns) is not int or started_ns < 0:
            raise ValueError("watchdog start must be a nonnegative integer")
        if not math.isfinite(timeout_seconds):
            raise ValueError("watchdog timeout must be finite")
        if timeout_seconds <= 0:
            raise ValueError("watchdog timeout must be positive")
        if not hasattr(os, "pipe2"):
            raise RuntimeError("a nonblocking CLOEXEC watchdog pipe is required")
        timeout_ns = math.ceil(timeout_seconds * 1_000_000_000)
        self.started_ns = started_ns
        self.deadline_ns = started_ns + timeout_ns
        self._cancel = threading.Event()
        self._state_lock = threading.Lock()
        self._state = self._ARMED
        self._expired_at_ns: int | None = None
        self._finished_at_ns: int | None = None
        self._closed = False
        self._read_fd, self._write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self._thread = threading.Thread(
            target=self._watch,
            name="summit-topology-sweep-watchdog",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException:
            _close_fd(self._read_fd)
            _close_fd(self._write_fd)
            self._read_fd = -1
            self._write_fd = -1
            self._closed = True
            raise

    @classmethod
    def start(cls, timeout_seconds: float) -> _SweepDeadlineWatchdog:
        return cls(
            time.clock_gettime_ns(time.CLOCK_MONOTONIC), timeout_seconds
        )

    @property
    def read_fd(self) -> int:
        if self._closed or self._read_fd < 0:
            raise RuntimeError("watchdog wake pipe is closed")
        return self._read_fd

    @property
    def expired(self) -> bool:
        with self._state_lock:
            return self._state == self._EXPIRED

    @property
    def expired_at_ns(self) -> int | None:
        with self._state_lock:
            return self._expired_at_ns

    @property
    def finished_at_ns(self) -> int | None:
        with self._state_lock:
            return self._finished_at_ns

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def thread_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def closed(self) -> bool:
        return self._closed

    def _watch(self) -> None:
        while True:
            now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            remaining_ns = self.deadline_ns - now_ns
            if remaining_ns <= 0:
                self._latch(now_ns)
                return
            if self._cancel.wait(remaining_ns / 1_000_000_000):
                return

    def _notify_expiry(self, write_fd: int) -> None:
        if write_fd < 0:
            return
        try:
            os.write(write_fd, WATCHDOG_WAKE_TOKEN)
        except (BlockingIOError, BrokenPipeError, OSError):
            # The state transition is authoritative. A full pipe cannot erase
            # the latch, and selectors also use bounded local timeouts.
            pass

    def _latch(self, observed_at_ns: int | None = None) -> bool:
        when_ns = (
            time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            if observed_at_ns is None
            else observed_at_ns
        )
        with self._state_lock:
            if self._state != self._ARMED:
                return False
            if when_ns < self.deadline_ns:
                return False
            self._state = self._EXPIRED
            self._expired_at_ns = when_ns
            write_fd = self._write_fd
        self._notify_expiry(write_fd)
        return True

    def expiration_error(
        self, context: str, *, observed_at_ns: int | None = None
    ) -> _SweepDeadlineExpired | None:
        now_ns = (
            time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            if observed_at_ns is None
            else observed_at_ns
        )
        self._latch(now_ns)
        with self._state_lock:
            expired = self._state == self._EXPIRED
        if not expired:
            return None
        return _SweepDeadlineExpired(
            f"topology sweep deadline expired {context}"
        )

    def raise_if_expired(
        self, context: str, *, observed_at_ns: int | None = None
    ) -> None:
        error = self.expiration_error(context, observed_at_ns=observed_at_ns)
        if error is not None:
            raise error

    def drain_wake(self) -> None:
        if self._closed or self._read_fd < 0:
            return
        while True:
            try:
                if not os.read(self._read_fd, 4096):
                    return
            except BlockingIOError:
                return
            except OSError:
                return

    def finish_execution(
        self, context: str
    ) -> tuple[int, _SweepDeadlineExpired | None]:
        notify_fd = -1
        with self._state_lock:
            finished_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            if self._finished_at_ns is None:
                self._finished_at_ns = finished_ns
            else:
                finished_ns = self._finished_at_ns
            if self._state == self._ARMED:
                if finished_ns >= self.deadline_ns:
                    self._state = self._EXPIRED
                    self._expired_at_ns = finished_ns
                    notify_fd = self._write_fd
                else:
                    self._state = self._DISARMED
            expired = self._state == self._EXPIRED
            self._cancel.set()
        if notify_fd >= 0:
            self._notify_expiry(notify_fd)
        error = (
            _SweepDeadlineExpired(
                f"topology sweep deadline expired {context}"
            )
            if expired
            else None
        )
        return finished_ns, error

    def cancel_join_close(self) -> None:
        if self.state == self._ARMED:
            self.finish_execution("before watchdog cancellation")
        else:
            self._cancel.set()
        self._thread.join(timeout=WATCHDOG_JOIN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            # Keep both descriptors owned by this still-live writer. Closing
            # them would permit descriptor reuse before a late notification.
            raise RuntimeError(
                "deadline watchdog did not stop after bounded cancellation"
            )
        with self._state_lock:
            if not self._closed:
                read_fd, write_fd = self._read_fd, self._write_fd
                self._read_fd = -1
                self._write_fd = -1
                self._closed = True
            else:
                read_fd = write_fd = -1
        _close_fd(read_fd)
        _close_fd(write_fd)


def _remaining_before_deadline(
    watchdog: _SweepDeadlineWatchdog,
    local_deadline_ns: int,
    context: str,
) -> float:
    now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    watchdog.raise_if_expired(context, observed_at_ns=now_ns)
    if now_ns >= local_deadline_ns:
        raise TimeoutError(f"configuration deadline expired {context}")
    remaining_ns = min(watchdog.deadline_ns, local_deadline_ns) - now_ns
    return remaining_ns / 1_000_000_000


def _watchdog_evidence(
    watchdog: _SweepDeadlineWatchdog, finished_ns: int
) -> dict[str, Any]:
    return {
        "mechanism": "controller_owned_absolute_deadline_watchdog",
        "clock": "CLOCK_MONOTONIC",
        "clock_implementation": time.get_clock_info("monotonic").implementation,
        "started_ns": watchdog.started_ns,
        "deadline_ns": watchdog.deadline_ns,
        "finished_ns": finished_ns,
        "measured_wall_seconds": (
            finished_ns - watchdog.started_ns
        )
        / 1_000_000_000,
        "terminal_state": watchdog.state,
        "latched": watchdog.expired,
        "expired": watchdog.expired,
        "expired_at_ns": watchdog.expired_at_ns,
        "deadline_equality_rejects": True,
        "selector_notification": "private_nonblocking_cloexec_self_pipe",
        "notification_is_wake_hint_only": True,
        "watchdog_thread_stopped": not watchdog.thread_alive,
        "wake_pipe_closed": watchdog.closed,
        "publication_outside_measured_cap": True,
    }


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
    parser.add_argument(
        "--expected-backend", choices=EXACT.EXPECTED_BACKENDS, required=True
    )
    parser.add_argument("--expected-private-source-commit")
    parser.add_argument("--expected-private-source-tree-sha256")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--dependency-path", type=Path, action="append", default=None)
    parser.add_argument("--one-by-32-cpus", type=EXACT._parse_cpu_list)
    parser.add_argument("--two-by-16-cpus", type=EXACT._parse_cpu_list, action="append")
    parser.add_argument("--two-by-32-cpus", type=EXACT._parse_cpu_list, action="append")
    parser.add_argument("--four-by-8-cpus", type=EXACT._parse_cpu_list, action="append")
    parser.add_argument(
        "--configuration-timeout-seconds",
        type=float,
        default=MAX_CONFIGURATION_SECONDS,
    )
    parser.add_argument(
        "--sweep-timeout-seconds", type=float, default=MAX_SWEEP_SECONDS
    )
    parser.add_argument("--max-memory-gib-per-process", type=float, default=64.0)
    parser.add_argument("--max-process-tree-memory-gib", type=float, default=128.0)
    parser.add_argument(
        "--max-start-skew-seconds", type=float, default=MAX_START_SKEW_SECONDS
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--n", type=int, default=EXACT.DEFAULT_N, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--block-width",
        type=int,
        default=EXACT.DEFAULT_BLOCK_WIDTH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-tile",
        type=int,
        default=EXACT.DEFAULT_PROBE_TILE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--environment-tile",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_topology-id", help=argparse.SUPPRESS)
    parser.add_argument("--_group-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_operation", choices=("source", "target"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_worker-cpus", type=EXACT._parse_cpu_list, help=argparse.SUPPRESS
    )
    parser.add_argument("--_threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_control-read-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_event-write-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_expected-runner-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--_expected-harness-sha256", help=argparse.SUPPRESS)
    return parser


def _strict_group(cpus: Sequence[int], expected_count: int) -> tuple[int, ...]:
    group = tuple(int(cpu) for cpu in cpus)
    if len(group) != expected_count:
        raise RuntimeError(
            f"CPU group requires exactly {expected_count} logical CPU IDs"
        )
    if group != tuple(sorted(group)) or len(group) != len(set(group)):
        raise RuntimeError("CPU groups must be unique and strictly increasing")
    return group


def _one_package(contract: Mapping[str, Any], label: str) -> int:
    records = contract.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{label} lacks exact CPU topology records")
    packages = {record.get("package") for record in records}
    if len(packages) != 1 or any(type(value) is not int for value in packages):
        raise RuntimeError(f"{label} must occupy exactly one physical socket")
    return next(iter(packages))


def _topology_contracts(args: argparse.Namespace) -> list[Topology]:
    if args.one_by_32_cpus is None:
        raise RuntimeError("--one-by-32-cpus is required")
    if args.two_by_16_cpus is None or len(args.two_by_16_cpus) != 2:
        raise RuntimeError("--two-by-16-cpus must be supplied exactly twice")
    if args.two_by_32_cpus is None or len(args.two_by_32_cpus) != 2:
        raise RuntimeError("--two-by-32-cpus must be supplied exactly twice")
    if args.four_by_8_cpus is None or len(args.four_by_8_cpus) != 4:
        raise RuntimeError("--four-by-8-cpus must be supplied exactly four times")

    one = _strict_group(args.one_by_32_cpus, 32)
    halves = tuple(_strict_group(value, 16) for value in args.two_by_16_cpus)
    sockets = tuple(_strict_group(value, 32) for value in args.two_by_32_cpus)
    quarters = tuple(_strict_group(value, 8) for value in args.four_by_8_cpus)
    if set(halves[0]).intersection(halves[1]):
        raise RuntimeError("2x16 CPU groups overlap")
    if set(halves[0]).union(halves[1]) != set(one):
        raise RuntimeError("2x16 groups must exactly partition the 1x32 CPU set")
    if sockets[0] != one:
        raise RuntimeError("2x32 group zero must exactly match the 1x32 CPU set")
    if set(sockets[0]).intersection(sockets[1]):
        raise RuntimeError("2x32 socket CPU groups overlap")
    quarter_union: set[int] = set()
    for group in quarters:
        if quarter_union.intersection(group):
            raise RuntimeError("4x8 CPU groups overlap")
        quarter_union.update(group)
    if quarter_union != set(one):
        raise RuntimeError("4x8 groups must exactly partition the 1x32 CPU set")

    one_contract = EXACT._cpu_contract(one)
    half_contracts = tuple(EXACT._cpu_contract(value) for value in halves)
    socket_contracts = tuple(EXACT._cpu_contract(value) for value in sockets)
    quarter_contracts = tuple(EXACT._cpu_contract(value) for value in quarters)
    package = _one_package(one_contract, "1x32")
    if any(_one_package(value, "2x16") != package for value in half_contracts):
        raise RuntimeError("2x16 groups are not on the 1x32 physical socket")
    socket_packages = tuple(_one_package(value, "2x32") for value in socket_contracts)
    if socket_packages[0] != package or socket_packages[0] == socket_packages[1]:
        raise RuntimeError("2x32 requires two distinct sockets and the shared baseline")
    if any(_one_package(value, "4x8") != package for value in quarter_contracts):
        raise RuntimeError("4x8 groups are not on the 1x32 physical socket")

    all_socket_physical: set[tuple[int, int]] = set()
    for contract in socket_contracts:
        for record in contract["records"]:
            physical = (record["package"], record["core"])
            if physical in all_socket_physical:
                raise RuntimeError("2x32 groups share a physical core or SMT sibling")
            all_socket_physical.add(physical)
    return [
        Topology("1x32", 32, (one,), (one_contract,), (5,), True, None),
        Topology(
            "2x16_same_socket",
            16,
            halves,
            half_contracts,
            (3, 2),
            True,
            None,
        ),
        Topology(
            "2x32_two_socket",
            32,
            sockets,
            socket_contracts,
            (3, 2),
            True,
            None,
        ),
        Topology(
            "4x8_same_socket",
            8,
            quarters,
            quarter_contracts,
            (2, 1, 1, 1),
            False,
            "kernel_only_until_one_environment_production_worker_merge_oracle_exists",
        ),
    ]


def _group_args(
    args: argparse.Namespace, cpus: Sequence[int], threads: int
) -> argparse.Namespace:
    values = dict(vars(args))
    values.update(
        {
            "cpus": tuple(cpus),
            "threads": threads,
            "blis_thread_strategy": "automatic",
            "blis_thread_ways": None,
        }
    )
    return SimpleNamespace(**values)


def _validate_common_identity(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for name in (
        "expected_native_sha256",
        "expected_package_manifest_sha256",
        "expected_archive_sha256",
        "expected_source_tree_sha256",
    ):
        if not EXACT._canonical_sha256(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be canonical SHA256")
    if not EXACT._canonical_commit(args.expected_source_commit):
        parser.error("--expected-source-commit must be canonical lowercase hex")
    validation = _group_args(args, (0,), 1)
    try:
        EXACT._validate_blis_args(parser, validation)
    except SystemExit:
        raise


def _validate_worker_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    _validate_common_identity(parser, args)
    required = {
        "_topology_id": args._topology_id,
        "_group_index": args._group_index,
        "_operation": args._operation,
        "_worker_cpus": args._worker_cpus,
        "_threads": args._threads,
        "_control_read_fd": args._control_read_fd,
        "_event_write_fd": args._event_write_fd,
    }
    if any(value is None for value in required.values()):
        parser.error(f"internal worker contract is incomplete: {required}")
    if args._group_index < 0 or args._threads <= 0:
        parser.error("internal worker group/thread values are invalid")
    if len(args._worker_cpus) != args._threads:
        parser.error("internal worker CPU count differs from thread count")
    if args._control_read_fd < 3 or args._event_write_fd < 3:
        parser.error("internal synchronization descriptors are invalid")
    for name in ("_expected_runner_sha256", "_expected_harness_sha256"):
        if not EXACT._canonical_sha256(getattr(args, name)):
            parser.error(f"internal {name} is not canonical SHA256")
    expected_worker_shape = (
        EXACT.DEFAULT_N,
        EXACT.DEFAULT_BLOCK_WIDTH,
        EXACT.DEFAULT_PROBE_TILE,
    )
    if (
        args.n,
        args.block_width,
        args.probe_tile,
    ) != expected_worker_shape:
        parser.error("internal worker shape differs from exact current production")
    topology_workers = {
        "1x32": (32, {0: 5}),
        "2x16_same_socket": (16, {0: 3, 1: 2}),
        "2x32_two_socket": (32, {0: 3, 1: 2}),
        "4x8_same_socket": (8, {0: 2, 1: 1, 2: 1, 3: 1}),
    }
    expected = topology_workers.get(args._topology_id)
    if (
        expected is None
        or args._threads != expected[0]
        or args._group_index not in expected[1]
        or args.environment_tile != expected[1].get(args._group_index)
    ):
        parser.error("internal worker topology/thread/group contract is invalid")


def _validate_controller_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[Topology]:
    _validate_common_identity(parser, args)
    if args.output is None:
        parser.error("--output is required")
    shape = (args.n, args.block_width, args.probe_tile)
    expected_shape = (
        EXACT.DEFAULT_N,
        EXACT.DEFAULT_BLOCK_WIDTH,
        EXACT.DEFAULT_PROBE_TILE,
    )
    if shape != expected_shape:
        parser.error(
            "topology evidence requires exact current production "
            f"(N,K,B_tile)={expected_shape} and five decomposed environments"
        )
    if not 0 < args.configuration_timeout_seconds <= MAX_CONFIGURATION_SECONDS:
        parser.error("--configuration-timeout-seconds must be in (0, 90]")
    if not 0 < args.sweep_timeout_seconds <= MAX_SWEEP_SECONDS:
        parser.error("--sweep-timeout-seconds must be in (0, 1200]")
    if not 0 < args.max_start_skew_seconds <= MAX_START_SKEW_SECONDS:
        parser.error("--max-start-skew-seconds must be in (0, 0.25]")
    if args.max_memory_gib_per_process <= 0 or args.max_process_tree_memory_gib <= 0:
        parser.error("memory limits must be positive")
    try:
        prefix = args.install_prefix.expanduser().resolve(strict=True)
        module = EXACT._regular_file(args.native_module)
        archive = EXACT._regular_file(args.private_archive)
        EXACT._regular_file(args.python_executable, executable=True)
        EXACT._dependency_paths(args)
        topologies = _topology_contracts(args)
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
    if EXACT._sha256(module) != args.expected_native_sha256:
        parser.error("native module SHA256 mismatch")
    if EXACT._sha256(archive) != args.expected_archive_sha256:
        parser.error("private archive SHA256 mismatch")
    package = EXACT._package_identity(prefix)
    if package["manifest_sha256"] != args.expected_package_manifest_sha256:
        parser.error("installed package manifest SHA256 mismatch")
    EXACT._validate_output_target(parser, args.output)
    return topologies


def _write_all(
    fd: int,
    payload: bytes,
    *,
    watchdog: _SweepDeadlineWatchdog | None = None,
    local_deadline_ns: int | None = None,
    context: str = "writing synchronization pipe",
) -> None:
    view = memoryview(payload)
    if watchdog is None:
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError("synchronization pipe closed during write")
            view = view[written:]
        return
    if local_deadline_ns is None:
        raise RuntimeError("bounded controller writes require a local deadline")
    os.set_blocking(fd, False)
    selector = selectors.DefaultSelector()
    selector.register(fd, selectors.EVENT_WRITE, "control")
    selector.register(watchdog.read_fd, selectors.EVENT_READ, "deadline")
    try:
        while view:
            remaining = _remaining_before_deadline(
                watchdog, local_deadline_ns, context
            )
            events = selector.select(timeout=remaining)
            watchdog.raise_if_expired(context)
            if not events:
                _remaining_before_deadline(watchdog, local_deadline_ns, context)
                continue
            for key, _mask in events:
                if key.data == "deadline":
                    watchdog.drain_wake()
                    watchdog.raise_if_expired(context)
                    continue
                try:
                    written = os.write(fd, view)
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise RuntimeError("synchronization pipe closed during write")
                view = view[written:]
        _remaining_before_deadline(watchdog, local_deadline_ns, context)
    finally:
        selector.close()


def _write_event(fd: int, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(EXACT._json_safe(payload), sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > 4096:
        raise RuntimeError("worker synchronization event exceeds PIPE_BUF contract")
    _write_all(fd, encoded)


def _read_control(fd: int, operation: str, ordinal: int) -> None:
    expected = f"GO {operation} {ordinal}\n".encode("ascii")
    received = bytearray()
    while len(received) < len(expected):
        chunk = os.read(fd, len(expected) - len(received))
        if not chunk:
            raise RuntimeError("controller synchronization pipe closed")
        received.extend(chunk)
    if bytes(received) != expected:
        raise RuntimeError("controller synchronization token mismatch")


def _read_finalize(fd: int) -> None:
    expected = b"FINALIZE\n"
    received = bytearray()
    while len(received) < len(expected):
        chunk = os.read(fd, len(expected) - len(received))
        if not chunk:
            raise RuntimeError("controller synchronization pipe closed before finalize")
        received.extend(chunk)
    if bytes(received) != expected:
        raise RuntimeError("controller finalize token mismatch")


def _worker_case(args: argparse.Namespace) -> Any:
    return EXACT.Case(
        args._operation,
        args.n,
        args.block_width,
        args.probe_tile,
        args.environment_tile,
        args._threads,
    )


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not sys.flags.no_site:
        raise RuntimeError("topology worker must run under python -S")
    if Path(sys.executable).resolve() != args.python_executable.resolve():
        raise RuntimeError("worker interpreter differs from requested executable")
    runner = Path(__file__).resolve()
    harness = Path(EXACT.__file__).resolve()
    if EXACT._sha256(runner) != args._expected_runner_sha256:
        raise RuntimeError("topology runner changed before worker execution")
    if EXACT._sha256(harness) != args._expected_harness_sha256:
        raise RuntimeError("exact-layout harness changed before worker execution")
    group_args = _group_args(args, args._worker_cpus, args._threads)
    environment = EXACT._validate_process_environment(group_args)
    prefix = args.install_prefix.resolve()
    module_path = args.native_module.resolve()
    archive_path = args.private_archive.resolve()
    dependencies = EXACT._dependency_paths(args)
    package_before = EXACT._package_identity(prefix)
    if package_before["manifest_sha256"] != args.expected_package_manifest_sha256:
        raise RuntimeError("installed package changed before worker import")
    if EXACT._sha256(module_path) != args.expected_native_sha256:
        raise RuntimeError("native module changed before worker import")
    if EXACT._sha256(archive_path) != args.expected_archive_sha256:
        raise RuntimeError("private archive changed before worker import")
    inserted = EXACT._bootstrap_paths(prefix, dependencies)
    import summit

    if Path(summit.__file__).resolve().parent != prefix / "summit":
        raise RuntimeError("summit imported outside exact installed prefix")
    from summit._early_numa import (
        allocate_numa_bound_anonymous_buffer,
        apply_early_numa_membind,
        verify_numa_bound_anonymous_buffer,
    )

    cpu = EXACT._cpu_contract(args._worker_cpus)
    if set(os.sched_getaffinity(0)) != set(args._worker_cpus):
        raise RuntimeError("worker taskset differs from declared CPU group")
    nodes = cpu["numa_nodes"]
    numa = EXACT._validate_early_numa_attestation(
        apply_early_numa_membind(EXACT._compress_ints(nodes)), nodes
    )
    import numpy as np

    numpy_path = Path(np.__file__).resolve()
    if not any(
        numpy_path == dependency or numpy_path.is_relative_to(dependency)
        for dependency in dependencies
    ):
        raise RuntimeError("NumPy imported outside declared dependency paths")
    module = EXACT._load_exact_native(module_path)
    configure = getattr(module, "configure_openmp_placement", None)
    if not callable(configure):
        raise RuntimeError("API-9 native module lacks placement configuration")
    placement = EXACT._validate_placement_attestation(
        configure(list(args._worker_cpus), args._threads),
        args._worker_cpus,
        args._threads,
    )
    configured = module.configure_blas_threads(args._threads)
    if type(configured) is not int or configured != args._threads:
        raise RuntimeError("private BLAS thread configuration is not immutable/exact")
    native = EXACT._validate_build_info(module, module_path, group_args, placement)
    EXACT._validate_native_numa_build_contract(native["build_info"])
    case = _worker_case(args)
    memory = EXACT._estimated_memory(case)
    if memory["conservative_peak_gib"] > args.max_memory_gib_per_process:
        raise RuntimeError("worker case exceeds per-process memory limit")
    if set(os.sched_getaffinity(0)) != {args._worker_cpus[0]}:
        raise RuntimeError("configured master differs from singleton place zero")
    left, right, input_numa = EXACT._prepare_operands(
        case,
        args.seed,
        np,
        nodes=nodes,
        allocate_bound_buffer=allocate_numa_bound_anonymous_buffer,
        verify_bound_buffer=verify_numa_bound_anonymous_buffer,
    )
    input_numa_gate = EXACT._validate_bound_operand_evidence(
        input_numa, case, nodes
    )
    inputs_before = {
        "left": EXACT._array_record(left),
        "right": EXACT._array_record(right),
    }
    function = getattr(
        module,
        "protected_matmul_nn" if case.operation == "source" else "protected_matmul_tn",
        None,
    )
    if not callable(function):
        raise RuntimeError("native module lacks exact protected current-layout GEMM")
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
            raise RuntimeError(f"native module lacks telemetry API {name}")

    _write_event(
        args._event_write_fd,
        {
            "event": "ready",
            "pid": os.getpid(),
            "topology_id": args._topology_id,
            "group_index": args._group_index,
            "operation": args._operation,
            "case_id": case.case_id,
        },
    )
    calls: list[dict[str, Any]] = []
    baseline = None
    baseline_hash = None
    for ordinal in range(EXACT.WARMUPS + EXACT.MEASURED_REPEATS):
        _read_control(args._control_read_fd, case.operation, ordinal)
        measured = ordinal >= EXACT.WARMUPS
        module.reset_gemm_telemetry()
        output_reset_status = EXACT._json_safe(
            dict(module.native_gemm_output_numa_evidence_status())
        )
        EXACT._validate_native_gemm_output_reset_status(
            output_reset_status, expected_next_call_id=ordinal + 1
        )
        entry_ns = time.monotonic_ns()
        wrapper_started = time.perf_counter()
        cpu_started = time.process_time()
        output, repaired = function(left, right, args._threads)
        wrapper_cpu = time.process_time() - cpu_started
        wrapper_wall = time.perf_counter() - wrapper_started
        exit_ns = time.monotonic_ns()
        if type(repaired) is not int or repaired != 0:
            raise RuntimeError("protected topology call repaired output columns")
        records = [
            EXACT._json_safe(dict(item)) for item in module.consume_gemm_telemetry()
        ]
        status = EXACT._json_safe(dict(module.gemm_telemetry_status()))
        output_records = [
            EXACT._json_safe(dict(item))
            for item in module.consume_native_gemm_output_numa_evidence()
        ]
        output_status = EXACT._json_safe(
            dict(module.native_gemm_output_numa_evidence_status())
        )
        if len(records) != 1:
            raise RuntimeError("protected topology call emitted non-unit telemetry")
        if len(output_records) != 1:
            raise RuntimeError(
                "protected topology call emitted non-unit output NUMA evidence"
            )
        sequence = records[0].get("sequence")
        if type(sequence) is not int or sequence != ordinal + 1:
            raise RuntimeError("topology telemetry sequence is not exact")
        EXACT._validate_telemetry_status(status, observed_sequence=sequence)
        if (
            tuple(output.shape) != case.output_shape
            or str(output.dtype) != "float64"
            or not output.flags.f_contiguous
            or not output.flags.aligned
        ):
            raise RuntimeError("protected topology output array is malformed")
        output_hash = EXACT._array_sha256(output)
        if baseline is None:
            baseline, baseline_hash = output, output_hash
        elif output_hash != baseline_hash:
            raise RuntimeError("topology worker output is not bitwise deterministic")
        telemetry_gate = EXACT._validate_telemetry(
            records[0],
            case,
            args._worker_cpus,
            nodes,
            native["build_info"],
            measured=measured,
        )
        native_numa_gate = EXACT._validate_protected_call_native_numa_contract(
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
        call = {
            "ordinal": ordinal,
            "phase": "measured" if measured else "warmup",
            "phase_ordinal": ordinal - EXACT.WARMUPS if measured else ordinal,
            "wrapper_entry_monotonic_ns": entry_ns,
            "wrapper_exit_monotonic_ns": exit_ns,
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
        calls.append(call)
        _write_event(
            args._event_write_fd,
            {
                "event": "call_done",
                "pid": os.getpid(),
                "topology_id": args._topology_id,
                "group_index": args._group_index,
                "operation": case.operation,
                "ordinal": ordinal,
                "phase": call["phase"],
                "wrapper_entry_monotonic_ns": entry_ns,
                "wrapper_exit_monotonic_ns": exit_ns,
                "vendor_wall_seconds": records[0]["wall_seconds"],
                "vendor_process_cpu_seconds": records[0]["process_cpu_seconds"],
                "active_core_equivalents": records[0]["active_core_equivalents"],
                "gflops_per_second": records[0]["gflops_per_second"],
                "output_sha256_storage_order": output_hash,
            },
        )
        if output is not baseline:
            del output
    assert baseline is not None and baseline_hash is not None
    # Keep the last call-done event as a strict barrier.  Without this token a
    # very small oracle could publish its final event in the same pipe read.
    _read_finalize(args._control_read_fd)
    oracle = EXACT._full_oracle(case, left, right, baseline, np)
    inputs_after = {
        "left": EXACT._array_record(left),
        "right": EXACT._array_record(right),
    }
    if inputs_after != inputs_before:
        raise RuntimeError("topology GEMM inputs changed")
    if EXACT._sha256(module_path) != args.expected_native_sha256:
        raise RuntimeError("native module changed during worker execution")
    if EXACT._sha256(archive_path) != args.expected_archive_sha256:
        raise RuntimeError("private archive changed during worker execution")
    if EXACT._package_identity(prefix) != package_before:
        raise RuntimeError("installed package changed during worker execution")
    if EXACT._sha256(runner) != args._expected_runner_sha256:
        raise RuntimeError("topology runner changed during worker execution")
    if EXACT._sha256(harness) != args._expected_harness_sha256:
        raise RuntimeError("exact-layout harness changed during worker execution")
    _write_event(
        args._event_write_fd,
        {
            "event": "final_ready",
            "pid": os.getpid(),
            "topology_id": args._topology_id,
            "group_index": args._group_index,
            "operation": case.operation,
            "case_id": case.case_id,
        },
    )
    return {
        "status": "accepted",
        "accepted": True,
        "pid": os.getpid(),
        "topology_id": args._topology_id,
        "group_index": args._group_index,
        "operation": case.operation,
        "case_id": case.case_id,
        "case": asdict(case),
        "cblas": case.cblas,
        "flops_per_call": case.flops,
        "protocol": {
            "warmups": EXACT.WARMUPS,
            "measured_repeats": EXACT.MEASURED_REPEATS,
            "fresh_python_no_site": True,
            "barrier_before_every_call": True,
            "zero_repairs_all_calls": True,
            "bitwise_deterministic_all_calls": True,
            "full_oracle_after_measured_calls": True,
        },
        "python_executable": str(Path(sys.executable).resolve()),
        "inserted_import_paths": inserted,
        "numpy_module": EXACT._file_identity(numpy_path),
        "thread_environment": environment,
        "cpu_contract": cpu,
        "early_numa_attestation": numa,
        "openmp_placement_attestation": placement,
        "native": native,
        "package_identity": package_before,
        "private_archive": EXACT._file_identity(archive_path),
        "memory_model": memory,
        "inputs": inputs_before,
        "input_numa_bound_buffers": input_numa,
        "input_numa_bound_buffers_gate": input_numa_gate,
        "output": EXACT._array_record(baseline),
        "calls": calls,
        "oracle": oracle,
    }


def _common_worker_command(args: argparse.Namespace) -> list[str]:
    command = [
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
        "--configuration-timeout-seconds",
        str(args.configuration_timeout_seconds),
        "--sweep-timeout-seconds",
        str(args.sweep_timeout_seconds),
        "--max-memory-gib-per-process",
        str(args.max_memory_gib_per_process),
        "--max-process-tree-memory-gib",
        str(args.max_process_tree_memory_gib),
        "--max-start-skew-seconds",
        str(args.max_start_skew_seconds),
        "--seed",
        str(args.seed),
        "--n",
        str(args.n),
        "--block-width",
        str(args.block_width),
        "--probe-tile",
        str(args.probe_tile),
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
    for dependency in EXACT._dependency_paths(args):
        command.extend(["--dependency-path", str(dependency)])
    return command


def _worker_command(
    args: argparse.Namespace,
    configuration: Configuration,
    group_index: int,
    taskset: Path,
    control_read_fd: int | str,
    event_write_fd: int | str,
    runner_sha256: str,
    harness_sha256: str,
) -> list[str]:
    cpus = configuration.topology.cpu_groups[group_index]
    return [
        str(taskset),
        "-c",
        EXACT._compress_ints(cpus),
        str(args.python_executable.resolve()),
        "-S",
        str(Path(__file__).resolve()),
        "--_worker",
        "--_topology-id",
        configuration.topology.topology_id,
        "--_group-index",
        str(group_index),
        "--_operation",
        configuration.operation,
        "--_worker-cpus",
        EXACT._compress_ints(cpus),
        "--_threads",
        str(configuration.topology.threads_per_process),
        "--_control-read-fd",
        str(control_read_fd),
        "--_event-write-fd",
        str(event_write_fd),
        "--_expected-runner-sha256",
        runner_sha256,
        "--_expected-harness-sha256",
        harness_sha256,
        "--environment-tile",
        str(configuration.topology.environment_tiles[group_index]),
        *_common_worker_command(args),
    ]


def _parse_smaps_rollup(path: Path) -> tuple[int, int]:
    fields: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in {"Rss:", "Pss:"} and parts[2] == "kB":
            fields[parts[0][:-1]] = int(parts[1]) * 1024
    if set(fields) != {"Rss", "Pss"} or any(value <= 0 for value in fields.values()):
        raise RuntimeError(f"incomplete live RSS/PSS evidence: {path}")
    return fields["Rss"], fields["Pss"]


def _read_live_process_memory(
    pid: int, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    process = proc_root / str(pid)
    rss, pss = _parse_smaps_rollup(process / "smaps_rollup")
    children = (
        (process / "task" / str(pid) / "children").read_text(encoding="utf-8").strip()
    )
    if children:
        raise RuntimeError(f"worker {pid} unexpectedly spawned child processes")
    return {"pid": pid, "rss_bytes": rss, "pss_bytes": pss, "children": []}


class _MemoryTracker:
    def __init__(self, pids: Sequence[int]) -> None:
        self.pids = tuple(pids)
        self.sample_count = 0
        self.first: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self.peak_rss: dict[str, Any] | None = None
        self.peak_pss: dict[str, Any] | None = None
        self.per_process_peak: dict[int, dict[str, int]] = {
            pid: {"rss_bytes": 0, "pss_bytes": 0} for pid in self.pids
        }

    def observe(self, label: str) -> None:
        workers = [_read_live_process_memory(pid) for pid in self.pids]
        snapshot = {
            "sample_ordinal": self.sample_count,
            "label": label,
            "controller_monotonic_ns": time.monotonic_ns(),
            "workers": workers,
            "combined_rss_bytes": sum(value["rss_bytes"] for value in workers),
            "combined_pss_bytes": sum(value["pss_bytes"] for value in workers),
        }
        self.sample_count += 1
        if self.first is None:
            self.first = snapshot
        self.last = snapshot
        if (
            self.peak_rss is None
            or snapshot["combined_rss_bytes"] > self.peak_rss["combined_rss_bytes"]
        ):
            self.peak_rss = snapshot
        if (
            self.peak_pss is None
            or snapshot["combined_pss_bytes"] > self.peak_pss["combined_pss_bytes"]
        ):
            self.peak_pss = snapshot
        for worker in workers:
            peak = self.per_process_peak[worker["pid"]]
            peak["rss_bytes"] = max(peak["rss_bytes"], worker["rss_bytes"])
            peak["pss_bytes"] = max(peak["pss_bytes"], worker["pss_bytes"])

    def report(self) -> dict[str, Any]:
        if self.sample_count <= 0 or any(
            value is None
            for value in (self.first, self.last, self.peak_rss, self.peak_pss)
        ):
            raise RuntimeError("configuration has no live process memory samples")
        return {
            "source": "linux_proc_smaps_rollup",
            "scope": "sum_of_all_live_worker_processes",
            "sample_interval_seconds": MEMORY_SAMPLE_INTERVAL_SECONDS,
            "sample_count": self.sample_count,
            "worker_children_required_absent": True,
            "first_snapshot": self.first,
            "last_snapshot": self.last,
            "peak_combined_rss_snapshot": self.peak_rss,
            "peak_combined_pss_snapshot": self.peak_pss,
            "per_process_sampled_peaks": {
                str(pid): value for pid, value in sorted(self.per_process_peak.items())
            },
        }


@dataclass
class _WorkerHandle:
    group_index: int
    process: subprocess.Popen[str]
    control_write_fd: int
    event_read_fd: int
    command: list[str]
    event_buffer: bytearray
    event_stream_tail: bytearray = field(default_factory=bytearray)
    stdout_tail: str = ""
    stderr_tail: str = ""
    output_collected: bool = False


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _append_event_stream_tail(handle: _WorkerHandle, chunk: bytes) -> None:
    handle.event_stream_tail.extend(chunk)
    excess = len(handle.event_stream_tail) - FAILURE_EVENT_STREAM_TAIL_BYTES
    if excess > 0:
        del handle.event_stream_tail[:excess]


def _drain_event_stream_tail(handle: _WorkerHandle) -> None:
    if handle.event_read_fd < 0:
        return
    drained = 0
    deadline = time.monotonic() + FAILURE_EVENT_DRAIN_MAX_SECONDS
    try:
        os.set_blocking(handle.event_read_fd, False)
        while (
            drained < FAILURE_EVENT_DRAIN_MAX_BYTES
            and time.monotonic() < deadline
        ):
            chunk = os.read(
                handle.event_read_fd,
                min(65_536, FAILURE_EVENT_DRAIN_MAX_BYTES - drained),
            )
            if not chunk:
                break
            drained += len(chunk)
            _append_event_stream_tail(handle, chunk)
    except BlockingIOError:
        pass
    except OSError:
        pass


def _bounded_text_tail(value: object, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-limit:]


def _capture_worker_failure_evidence(
    handles: Sequence[_WorkerHandle],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for handle in handles:
        capture_error: str | None = None
        reaped = handle.process.poll() is not None
        if reaped and not handle.output_collected:
            try:
                stdout, stderr = handle.process.communicate(timeout=0.0)
                handle.output_collected = True
                if stdout:
                    handle.stdout_tail = _bounded_text_tail(
                        stdout, FAILURE_STDOUT_TAIL_CHARACTERS
                    )
                if stderr:
                    handle.stderr_tail = _bounded_text_tail(
                        stderr, FAILURE_STDERR_TAIL_CHARACTERS
                    )
            except Exception as error:
                capture_error = f"{type(error).__name__}: {error}"
        elif not reaped:
            capture_error = "process_not_reaped_after_bounded_cleanup"
        evidence.append(
            {
                "group_index": handle.group_index,
                "pid": handle.process.pid,
                "returncode": handle.process.poll(),
                "command": handle.command,
                "stdout_tail": handle.stdout_tail[-FAILURE_STDOUT_TAIL_CHARACTERS:],
                "stderr_tail": handle.stderr_tail[-FAILURE_STDERR_TAIL_CHARACTERS:],
                "event_stream_tail_utf8": bytes(handle.event_stream_tail).decode(
                    "utf-8", errors="replace"
                ),
                "unparsed_event_buffer_tail_utf8": bytes(
                    handle.event_buffer[-MAX_UNPARSED_EVENT_BUFFER_BYTES:]
                ).decode("utf-8", errors="replace"),
                "output_capture_error": capture_error,
            }
        )
    return evidence


def _kill_handles(handles: Sequence[_WorkerHandle]) -> None:
    for handle in handles:
        _close_fd(handle.control_write_fd)
        if handle.process.poll() is None:
            try:
                os.killpg(handle.process.pid, TERMINATE_PROCESS_GROUP_SIGNAL)
            except (ProcessLookupError, PermissionError):
                pass
    term_deadline = time.monotonic() + CLEANUP_TERM_WAIT_SECONDS
    for handle in handles:
        if handle.process.poll() is None:
            try:
                handle.process.wait(timeout=max(0.0, term_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
    survivors = [handle for handle in handles if handle.process.poll() is None]
    for handle in survivors:
        try:
            os.killpg(handle.process.pid, KILL_PROCESS_GROUP_SIGNAL)
        except (ProcessLookupError, PermissionError):
            pass
    kill_deadline = time.monotonic() + CLEANUP_KILL_WAIT_SECONDS
    for handle in survivors:
        if handle.process.poll() is None:
            try:
                handle.process.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
    for handle in handles:
        _drain_event_stream_tail(handle)
        _close_fd(handle.event_read_fd)
        handle.event_read_fd = -1
    unreaped = [
        handle.process.pid for handle in survivors if handle.process.poll() is None
    ]
    if unreaped:
        raise RuntimeError(
            "worker processes remained live/unreaped after bounded SIGKILL cleanup: "
            f"{unreaped}"
        )


def _terminate_unregistered_process(process: subprocess.Popen[str]) -> None:
    """Bound and reap a child returned by Popen before handle registration."""

    if process.poll() is None:
        try:
            os.killpg(process.pid, TERMINATE_PROCESS_GROUP_SIGNAL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=CLEANUP_TERM_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, KILL_PROCESS_GROUP_SIGNAL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=CLEANUP_KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        raise RuntimeError(
            "unregistered worker remained live/unreaped after bounded cleanup: "
            f"{process.pid}"
        )
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _decode_event(line: bytes, handle: _WorkerHandle) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"worker {handle.group_index} emitted malformed event"
        ) from error
    if not isinstance(value, dict) or value.get("pid") != handle.process.pid:
        raise RuntimeError(f"worker {handle.group_index} event identity mismatch")
    return value


def _wait_for_events(
    handles: Sequence[_WorkerHandle],
    expected_event: str,
    local_deadline_ns: int,
    tracker: _MemoryTracker,
    watchdog: _SweepDeadlineWatchdog,
    *,
    ordinal: int | None = None,
) -> dict[int, dict[str, Any]]:
    pending = {handle.group_index: handle for handle in handles}
    received: dict[int, dict[str, Any]] = {}
    selector = selectors.DefaultSelector()
    try:
        for handle in handles:
            os.set_blocking(handle.event_read_fd, False)
            selector.register(handle.event_read_fd, selectors.EVENT_READ, handle)
        selector.register(watchdog.read_fd, selectors.EVENT_READ, "deadline")
        next_sample = 0.0
        while pending:
            remaining = _remaining_before_deadline(
                watchdog,
                local_deadline_ns,
                f"waiting for worker {expected_event} events",
            )
            now = time.monotonic()
            if now >= next_sample:
                tracker.observe(f"waiting_{expected_event}")
                next_sample = now + MEMORY_SAMPLE_INTERVAL_SECONDS
            events = selector.select(
                timeout=min(
                    0.05,
                    remaining,
                    max(0.0, next_sample - time.monotonic()),
                )
            )
            # The deadline wins when a worker event and the wake pipe become
            # ready in the same selector turn.
            watchdog.raise_if_expired(
                f"waiting for worker {expected_event} events"
            )
            for key, _mask in events:
                if key.data == "deadline":
                    watchdog.drain_wake()
                    watchdog.raise_if_expired(
                        f"waiting for worker {expected_event} events"
                    )
                    continue
                handle = key.data
                chunk = os.read(handle.event_read_fd, 65536)
                if not chunk:
                    raise RuntimeError(
                        f"worker {handle.group_index} closed event pipe before {expected_event}"
                    )
                _append_event_stream_tail(handle, chunk)
                handle.event_buffer.extend(chunk)
                if len(handle.event_buffer) > MAX_UNPARSED_EVENT_BUFFER_BYTES:
                    del handle.event_buffer[:-MAX_UNPARSED_EVENT_BUFFER_BYTES]
                    raise RuntimeError(
                        f"worker {handle.group_index} exceeded the bounded event buffer"
                    )
                while b"\n" in handle.event_buffer:
                    raw, _, remainder = handle.event_buffer.partition(b"\n")
                    handle.event_buffer[:] = remainder
                    event = _decode_event(raw, handle)
                    if event.get("event") == "error":
                        raise RuntimeError(
                            f"worker {handle.group_index} failed: {event.get('reason')}"
                        )
                    if handle.group_index not in pending:
                        raise RuntimeError(
                            "worker emitted more than one event per barrier"
                        )
                    if event.get("event") != expected_event:
                        raise RuntimeError(
                            f"expected {expected_event}, observed {event.get('event')}"
                        )
                    if ordinal is not None and event.get("ordinal") != ordinal:
                        raise RuntimeError("worker call ordinal differs from barrier")
                    received[handle.group_index] = event
                    pending.pop(handle.group_index)
            for group_index, handle in list(pending.items()):
                returncode = handle.process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"worker {group_index} exited {returncode} before {expected_event}"
                    )
        # A final-ready worker is allowed to exit immediately after publishing
        # its stdout record; all preceding waits already sampled its live peak.
        if expected_event != "final_ready":
            tracker.observe(f"received_{expected_event}")
        _remaining_before_deadline(
            watchdog,
            local_deadline_ns,
            f"after receiving worker {expected_event} events",
        )
        return received
    finally:
        selector.close()


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    lines = [
        line for line in stdout.splitlines() if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError(f"worker emitted {len(lines)} result records")
    value = json.loads(lines[0][len(WORKER_RESULT_PREFIX) :])
    if not isinstance(value, dict):
        raise RuntimeError("worker result is not a JSON object")
    return value


def _static_build_digest(worker: Mapping[str, Any]) -> str:
    build = dict(worker["native"]["build_info"])
    build.pop("openmp_placement_contract_evidence", None)
    return EXACT._canonical_json_sha256(build)


def _aggregate_configuration(
    configuration: Configuration,
    workers: Sequence[Mapping[str, Any]],
    events_by_ordinal: Sequence[Mapping[int, Mapping[str, Any]]],
    release_ns_by_ordinal: Sequence[Sequence[int]],
    max_start_skew_seconds: float,
) -> dict[str, Any]:
    count = configuration.topology.process_count
    if (
        len(workers) != count
        or len(configuration.topology.environment_tiles) != count
        or len(events_by_ordinal) != EXACT.WARMUPS + EXACT.MEASURED_REPEATS
    ):
        raise RuntimeError("configuration worker/call cardinality mismatch")
    if [worker.get("group_index") for worker in workers] != list(range(count)):
        raise RuntimeError("configuration worker ordering mismatch")
    expected_cases = [
        EXACT.Case(
            configuration.operation,
            EXACT.DEFAULT_N,
            EXACT.DEFAULT_BLOCK_WIDTH,
            EXACT.DEFAULT_PROBE_TILE,
            environment_tile,
            configuration.topology.threads_per_process,
        )
        for environment_tile in configuration.topology.environment_tiles
    ]
    static_builds = {_static_build_digest(worker) for worker in workers}
    if len(static_builds) != 1:
        raise RuntimeError("synchronized workers differ in static build identity")
    for group_index, worker in enumerate(workers):
        expected_case = expected_cases[group_index]
        expected_worker = {
            "status": "accepted",
            "accepted": True,
            "topology_id": configuration.topology.topology_id,
            "group_index": group_index,
            "operation": configuration.operation,
            "case_id": expected_case.case_id,
            "case": asdict(expected_case),
            "cblas": expected_case.cblas,
            "flops_per_call": expected_case.flops,
        }
        mismatches = {
            name: {"expected": value, "observed": worker.get(name)}
            for name, value in expected_worker.items()
            if worker.get(name) != value
        }
        if mismatches:
            raise RuntimeError(f"worker result contract mismatch: {mismatches}")
        if (
            not EXACT._canonical_sha256(
                worker.get("output", {}).get("sha256_storage_order")
            )
            or not EXACT._canonical_sha256(
                worker.get("oracle", {}).get("oracle_sha256_storage_order")
            )
            or worker.get("oracle", {}).get("passed") is not True
            or worker.get("oracle", {}).get("full_output_compared") is not True
            or not isinstance(worker.get("inputs"), Mapping)
        ):
            raise RuntimeError("worker lacks its independent identity/oracle evidence")
        if (
            worker.get("cpu_contract")
            != configuration.topology.cpu_contracts[group_index]
        ):
            raise RuntimeError("worker CPU contract differs from controller topology")
        bound_operand_gate = EXACT._validate_bound_operand_evidence(
            worker.get("input_numa_bound_buffers"),
            expected_case,
            configuration.topology.cpu_contracts[group_index]["numa_nodes"],
        )
        if worker.get("input_numa_bound_buffers_gate") != bound_operand_gate:
            raise RuntimeError(
                "worker bound-operand gate differs from controller reconstruction"
            )
        calls = worker.get("calls")
        if (
            not isinstance(calls, list)
            or len(calls) != EXACT.WARMUPS + EXACT.MEASURED_REPEATS
        ):
            raise RuntimeError("worker result has an invalid call count")
        native = worker.get("native")
        build = native.get("build_info") if isinstance(native, Mapping) else None
        if not isinstance(build, Mapping):
            raise RuntimeError("worker lacks native build evidence")
        EXACT._validate_native_numa_build_contract(build)
        cpu_contract = configuration.topology.cpu_contracts[group_index]
        nodes = cpu_contract.get("numa_nodes")
        if (
            not isinstance(nodes, list)
            or any(type(node) is not int for node in nodes)
            or nodes != sorted(set(nodes))
        ):
            raise RuntimeError("controller topology NUMA-node contract is malformed")
        for ordinal, call in enumerate(calls):
            if not isinstance(call, Mapping):
                raise RuntimeError("worker call evidence is malformed")
            measured = ordinal >= EXACT.WARMUPS
            expected_call = {
                "ordinal": ordinal,
                "phase": "measured" if measured else "warmup",
                "phase_ordinal": ordinal - EXACT.WARMUPS if measured else ordinal,
            }
            if any(call.get(name) != value for name, value in expected_call.items()):
                raise RuntimeError("worker call ordinal/phase evidence mismatch")
            telemetry = call.get("telemetry")
            telemetry_status = call.get("telemetry_status")
            if not isinstance(telemetry, Mapping) or not isinstance(
                telemetry_status, Mapping
            ):
                raise RuntimeError("worker call lacks telemetry/status evidence")
            sequence = telemetry.get("sequence")
            if type(sequence) is not int or sequence != ordinal + 1:
                raise RuntimeError("worker telemetry sequence is not exact")
            EXACT._validate_telemetry_status(
                telemetry_status, observed_sequence=sequence
            )
            telemetry_gate = EXACT._validate_telemetry(
                telemetry,
                expected_case,
                configuration.topology.cpu_groups[group_index],
                nodes,
                build,
                measured=measured,
            )
            if call.get("telemetry_gate") != telemetry_gate:
                raise RuntimeError(
                    "worker telemetry gate differs from controller reconstruction"
                )
            output_evidence = call.get("native_gemm_output_numa_evidence")
            reset_status = call.get("native_gemm_output_numa_reset_status")
            post_status = call.get("native_gemm_output_numa_status")
            if not all(
                isinstance(value, Mapping)
                for value in (output_evidence, reset_status, post_status)
            ):
                raise RuntimeError("worker call lacks native output NUMA evidence")
            native_numa_gate = (
                EXACT._validate_protected_call_native_numa_contract(
                    telemetry=telemetry,
                    output_evidence=output_evidence,
                    reset_status=reset_status,
                    post_status=post_status,
                    case=expected_case,
                    nodes=nodes,
                    expected_call_id=ordinal + 1,
                    integrity_minimum_vendor_flops=build[
                        "gemm_integrity_minimum_vendor_flops"
                    ],
                )
            )
            if (
                call.get("native_gemm_output_numa_gate")
                != native_numa_gate["output"]
                or call.get("native_integrity_snapshot_numa_evidence")
                != telemetry.get("native_integrity_snapshot_numa")
                or call.get("native_integrity_snapshot_numa_gate")
                != native_numa_gate["integrity_snapshot"]
                or call.get("protected_call_native_numa_gate") != native_numa_gate
            ):
                raise RuntimeError(
                    "worker native NUMA gate differs from controller reconstruction"
                )

    waves: list[dict[str, Any]] = []
    for ordinal, events in enumerate(events_by_ordinal):
        if set(events) != set(range(count)):
            raise RuntimeError("synchronized wave lacks one or more process events")
        ordered = [events[index] for index in range(count)]
        for group_index, event in enumerate(ordered):
            expected_event = {
                "event": "call_done",
                "pid": workers[group_index]["pid"],
                "topology_id": configuration.topology.topology_id,
                "group_index": group_index,
                "operation": configuration.operation,
                "ordinal": ordinal,
                "phase": "measured" if ordinal >= EXACT.WARMUPS else "warmup",
            }
            if any(event.get(name) != value for name, value in expected_event.items()):
                raise RuntimeError("worker barrier event identity/phase mismatch")
            call = workers[group_index]["calls"][ordinal]
            comparable = {
                "wrapper_entry_monotonic_ns": call["wrapper_entry_monotonic_ns"],
                "wrapper_exit_monotonic_ns": call["wrapper_exit_monotonic_ns"],
                "vendor_wall_seconds": call["telemetry"]["wall_seconds"],
                "vendor_process_cpu_seconds": call["telemetry"]["process_cpu_seconds"],
                "active_core_equivalents": call["telemetry"]["active_core_equivalents"],
                "gflops_per_second": call["telemetry"]["gflops_per_second"],
                "output_sha256_storage_order": call["output_sha256_storage_order"],
            }
            if any(event.get(name) != value for name, value in comparable.items()):
                raise RuntimeError(
                    "worker barrier event differs from final call record"
                )
        entries = [int(event["wrapper_entry_monotonic_ns"]) for event in ordered]
        exits = [int(event["wrapper_exit_monotonic_ns"]) for event in ordered]
        if any(exit_value <= entry for entry, exit_value in zip(entries, exits)):
            raise RuntimeError("worker wrapper timing is nonpositive")
        start_skew = (max(entries) - min(entries)) / 1.0e9
        if start_skew > max_start_skew_seconds:
            raise RuntimeError("synchronized worker start skew exceeds declared bound")
        makespan = (max(exits) - min(entries)) / 1.0e9
        vendor_walls = [float(event["vendor_wall_seconds"]) for event in ordered]
        vendor_cpus = [float(event["vendor_process_cpu_seconds"]) for event in ordered]
        active = [float(event["active_core_equivalents"]) for event in ordered]
        rates = [float(event["gflops_per_second"]) for event in ordered]
        if any(
            not math.isfinite(value) or value <= 0
            for value in [makespan, *vendor_walls, *vendor_cpus, *active, *rates]
        ):
            raise RuntimeError("synchronized wave contains invalid performance values")
        release_values = list(release_ns_by_ordinal[ordinal])
        if len(release_values) != count:
            raise RuntimeError("synchronized wave release cardinality mismatch")
        total_flops = sum(case.flops for case in expected_cases)
        waves.append(
            {
                "ordinal": ordinal,
                "phase": "measured" if ordinal >= EXACT.WARMUPS else "warmup",
                "phase_ordinal": ordinal - EXACT.WARMUPS
                if ordinal >= EXACT.WARMUPS
                else ordinal,
                "controller_release_skew_seconds": (
                    max(release_values) - min(release_values)
                )
                / 1.0e9,
                "worker_start_skew_seconds": start_skew,
                "worker_completion_skew_seconds": (max(exits) - min(exits)) / 1.0e9,
                "wrapper_makespan_seconds": makespan,
                "total_flops": total_flops,
                "aggregate_gflops_per_second": total_flops / (makespan * 1.0e9),
                "aggregate_process_cpu_seconds": math.fsum(vendor_cpus),
                "aggregate_active_core_equivalents": math.fsum(vendor_cpus) / makespan,
                "per_worker_vendor_wall_seconds": vendor_walls,
                "per_worker_active_core_equivalents": active,
                "per_worker_gflops_per_second": rates,
                "per_worker_flops": [case.flops for case in expected_cases],
                "per_worker_environment_tiles": list(
                    configuration.topology.environment_tiles
                ),
                "per_worker_panel_columns": [
                    case.panel_columns for case in expected_cases
                ],
                "straggler": {
                    "fastest_vendor_wall_seconds": min(vendor_walls),
                    "median_vendor_wall_seconds": statistics.median(vendor_walls),
                    "slowest_vendor_wall_seconds": max(vendor_walls),
                    "slowest_minus_fastest_seconds": max(vendor_walls)
                    - min(vendor_walls),
                    "slowest_over_fastest_ratio": max(vendor_walls) / min(vendor_walls),
                },
            }
        )
    measured = [wave for wave in waves if wave["phase"] == "measured"]
    return {
        "passed": True,
        "static_build_identity_count": len(static_builds),
        "per_worker_independent_identity_and_oracle_required": True,
        "cross_worker_input_output_oracle_equality_required": False,
        "worker_cases": [
            {
                "group_index": index,
                "environment_tile": configuration.topology.environment_tiles[index],
                "case": asdict(case),
                "cblas": case.cblas,
                "panel_columns": case.panel_columns,
                "flops_per_call": case.flops,
            }
            for index, case in enumerate(expected_cases)
        ],
        "barrier_wave_count": len(waves),
        "waves": waves,
        "measured_summary": {
            "wrapper_makespan_seconds": EXACT._summary(
                [wave["wrapper_makespan_seconds"] for wave in measured]
            ),
            "aggregate_gflops_per_second": EXACT._summary(
                [wave["aggregate_gflops_per_second"] for wave in measured]
            ),
            "aggregate_active_core_equivalents": EXACT._summary(
                [wave["aggregate_active_core_equivalents"] for wave in measured]
            ),
            "worker_start_skew_seconds": EXACT._summary(
                [wave["worker_start_skew_seconds"] for wave in measured]
            ),
            "straggler_slowest_over_fastest_ratio": EXACT._summary(
                [wave["straggler"]["slowest_over_fastest_ratio"] for wave in measured]
            ),
            "cumulative_synchronized_makespan_seconds": math.fsum(
                wave["wrapper_makespan_seconds"] for wave in measured
            ),
            "matrix_minutes": math.fsum(
                wave["wrapper_makespan_seconds"] for wave in measured
            )
            / 60.0,
        },
    }


def _run_configuration(
    args: argparse.Namespace,
    configuration: Configuration,
    taskset: Path,
    runner_sha256: str,
    harness_sha256: str,
    watchdog: _SweepDeadlineWatchdog,
) -> dict[str, Any]:
    started_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    local_deadline_ns = started_ns + math.ceil(
        args.configuration_timeout_seconds * 1_000_000_000
    )
    handles: list[_WorkerHandle] = []
    events_by_ordinal: list[dict[int, dict[str, Any]]] = []
    releases_by_ordinal: list[list[int]] = []
    tracker: _MemoryTracker | None = None
    outcome: dict[str, Any] | None = None
    failure: BaseException | None = None
    failure_status = "failed"
    try:
        for group_index, cpus in enumerate(configuration.topology.cpu_groups):
            _remaining_before_deadline(
                watchdog,
                local_deadline_ns,
                f"before spawning worker {group_index}",
            )
            control_read = control_write = event_read = event_write = -1
            process: subprocess.Popen[str] | None = None
            handle: _WorkerHandle | None = None
            registered = False
            try:
                control_read, control_write = os.pipe()
                event_read, event_write = os.pipe()
                command = _worker_command(
                    args,
                    configuration,
                    group_index,
                    taskset,
                    control_read,
                    event_write,
                    runner_sha256,
                    harness_sha256,
                )
                group_args = _group_args(
                    args, cpus, configuration.topology.threads_per_process
                )
                environment, _settings = EXACT._worker_environment(group_args)
                _remaining_before_deadline(
                    watchdog,
                    local_deadline_ns,
                    f"immediately before spawning worker {group_index}",
                )
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=environment,
                    pass_fds=(control_read, event_write),
                    start_new_session=True,
                )
                # A returned child is registered before any deadline check or
                # other intentional raising operation.  This transaction
                # closes the Popen-return race without process-wide signals.
                handle = _WorkerHandle(
                    group_index,
                    process,
                    control_write,
                    event_read,
                    command,
                    bytearray(),
                )
                handles.append(handle)
                registered = True
                _close_fd(control_read)
                control_read = -1
                _close_fd(event_write)
                event_write = -1
            except BaseException as spawn_error:
                registered = registered or (
                    handle is not None
                    and any(existing is handle for existing in handles)
                )
                for fd in (control_read, event_write):
                    _close_fd(fd)
                if not registered:
                    for fd in (control_write, event_read):
                        _close_fd(fd)
                    if process is not None:
                        try:
                            _terminate_unregistered_process(process)
                        except Exception as cleanup_error:
                            raise RuntimeError(
                                "failed to reap worker after handle registration failure: "
                                f"{type(cleanup_error).__name__}: {cleanup_error}"
                            ) from spawn_error
                raise
            _remaining_before_deadline(
                watchdog,
                local_deadline_ns,
                f"after registering worker {group_index}",
            )
        tracker = _MemoryTracker([handle.process.pid for handle in handles])
        ready = _wait_for_events(
            handles, "ready", local_deadline_ns, tracker, watchdog
        )
        for group_index, event in ready.items():
            expected_case = EXACT.Case(
                configuration.operation,
                args.n,
                args.block_width,
                args.probe_tile,
                configuration.topology.environment_tiles[group_index],
                configuration.topology.threads_per_process,
            )
            if (
                event.get("topology_id") != configuration.topology.topology_id
                or event.get("group_index") != group_index
                or event.get("operation") != configuration.operation
                or event.get("case_id") != expected_case.case_id
            ):
                raise RuntimeError("worker ready event configuration mismatch")
        for ordinal in range(EXACT.WARMUPS + EXACT.MEASURED_REPEATS):
            releases: list[int] = []
            token = f"GO {configuration.operation} {ordinal}\n".encode("ascii")
            for handle in handles:
                releases.append(time.monotonic_ns())
                _write_all(
                    handle.control_write_fd,
                    token,
                    watchdog=watchdog,
                    local_deadline_ns=local_deadline_ns,
                    context=f"releasing worker call {ordinal}",
                )
            releases_by_ordinal.append(releases)
            events_by_ordinal.append(
                _wait_for_events(
                    handles,
                    "call_done",
                    local_deadline_ns,
                    tracker,
                    watchdog,
                    ordinal=ordinal,
                )
            )
        for handle in handles:
            _write_all(
                handle.control_write_fd,
                b"FINALIZE\n",
                watchdog=watchdog,
                local_deadline_ns=local_deadline_ns,
                context="releasing worker finalization",
            )
        final_events = _wait_for_events(
            handles, "final_ready", local_deadline_ns, tracker, watchdog
        )
        for group_index, event in final_events.items():
            expected_case = EXACT.Case(
                configuration.operation,
                args.n,
                args.block_width,
                args.probe_tile,
                configuration.topology.environment_tiles[group_index],
                configuration.topology.threads_per_process,
            )
            if (
                event.get("topology_id") != configuration.topology.topology_id
                or event.get("group_index") != group_index
                or event.get("operation") != configuration.operation
                or event.get("case_id") != expected_case.case_id
            ):
                raise RuntimeError("worker final-ready event configuration mismatch")
        workers: list[dict[str, Any]] = []
        for handle in handles:
            _close_fd(handle.control_write_fd)
            handle.control_write_fd = -1
            remaining = _remaining_before_deadline(
                watchdog,
                local_deadline_ns,
                f"before collecting worker {handle.group_index}",
            )
            try:
                stdout, stderr = handle.process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                deadline_error = watchdog.expiration_error(
                    f"while collecting worker {handle.group_index}"
                )
                if deadline_error is not None:
                    raise deadline_error from error
                raise TimeoutError(
                    "configuration timed out collecting workers"
                ) from error
            handle.stdout_tail = _bounded_text_tail(
                stdout, FAILURE_STDOUT_TAIL_CHARACTERS
            )
            handle.stderr_tail = _bounded_text_tail(
                stderr, FAILURE_STDERR_TAIL_CHARACTERS
            )
            handle.output_collected = True
            _remaining_before_deadline(
                watchdog,
                local_deadline_ns,
                f"after collecting worker {handle.group_index}",
            )
            _close_fd(handle.event_read_fd)
            handle.event_read_fd = -1
            if handle.process.returncode != 0:
                raise RuntimeError(
                    f"worker {handle.group_index} exited {handle.process.returncode}: "
                    f"{stderr[-2000:]}"
                )
            result = _parse_worker_stdout(stdout)
            if (
                result.get("pid") != handle.process.pid
                or result.get("group_index") != handle.group_index
                or result.get("topology_id") != configuration.topology.topology_id
                or result.get("operation") != configuration.operation
                or result.get("accepted") is not True
            ):
                raise RuntimeError("worker final result identity/status mismatch")
            result["command"] = handle.command
            result["stderr_tail"] = stderr[-4000:]
            workers.append(result)
        aggregate = _aggregate_configuration(
            configuration,
            workers,
            events_by_ordinal,
            releases_by_ordinal,
            args.max_start_skew_seconds,
        )
        _remaining_before_deadline(
            watchdog,
            local_deadline_ns,
            "after configuration evidence aggregation",
        )
        wall = (
            time.clock_gettime_ns(time.CLOCK_MONOTONIC) - started_ns
        ) / 1_000_000_000
        outcome = {
            "config_id": configuration.config_id,
            "status": "accepted",
            "accepted": True,
            "operation": configuration.operation,
            "topology_id": configuration.topology.topology_id,
            "controller_wall_seconds": wall,
            "configuration_timeout_seconds": args.configuration_timeout_seconds,
            "topology": asdict(configuration.topology),
            "workers": workers,
            "synchronization_and_throughput": aggregate,
            "live_process_memory": tracker.report(),
        }
    except BaseException as error:
        failure = error
        if isinstance(error, _SweepDeadlineExpired):
            failure_status = "sweep_timeout"
        elif isinstance(error, TimeoutError):
            failure_status = "timeout"
    cleanup_error: Exception | None = None
    cleanup_deadline: _SweepDeadlineExpired | None = None
    local_cleanup_timeout: TimeoutError | None = None
    try:
        _kill_handles(handles)
    except Exception as error:
        cleanup_error = error
    try:
        worker_failure_evidence = _capture_worker_failure_evidence(handles)
    except Exception as error:
        cleanup_error = cleanup_error or error
        worker_failure_evidence = []
    retained_events = EXACT._json_safe(events_by_ordinal)
    retained_releases = EXACT._json_safe(releases_by_ordinal)
    try:
        retained_memory = (
            None
            if tracker is None or tracker.sample_count <= 0
            else tracker.report()
        )
    except Exception as error:
        cleanup_error = cleanup_error or error
        retained_memory = {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    cleanup_observed_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    cleanup_deadline = watchdog.expiration_error(
        "during bounded configuration cleanup",
        observed_at_ns=cleanup_observed_ns,
    )
    if cleanup_deadline is None and cleanup_observed_ns >= local_deadline_ns:
        local_cleanup_timeout = TimeoutError(
            "configuration deadline expired during bounded cleanup"
        )
    if cleanup_deadline is not None:
        failure_status = "sweep_timeout"
    elif local_cleanup_timeout is not None:
        failure_status = "timeout"
    if failure is not None and not isinstance(failure, Exception):
        raise failure
    if (
        failure is None
        and cleanup_error is None
        and cleanup_deadline is None
        and local_cleanup_timeout is None
    ):
        if outcome is None:
            raise RuntimeError("configuration completed without an outcome")
        return outcome
    reasons: list[str] = []
    if failure is not None:
        reasons.append(f"{type(failure).__name__}: {failure}")
    if cleanup_deadline is not None and not isinstance(
        failure, _SweepDeadlineExpired
    ):
        reasons.append(
            f"{type(cleanup_deadline).__name__}: {cleanup_deadline}"
        )
    if local_cleanup_timeout is not None and not isinstance(failure, TimeoutError):
        reasons.append(
            f"{type(local_cleanup_timeout).__name__}: {local_cleanup_timeout}"
        )
    if cleanup_error is not None:
        reasons.append(
            f"bounded cleanup {type(cleanup_error).__name__}: {cleanup_error}"
        )
    if (
        failure is None
        and cleanup_deadline is None
        and local_cleanup_timeout is None
    ):
        failure_status = "cleanup_failed"
    original_failure = (
        None
        if failure is None
        else {"type": type(failure).__name__, "message": str(failure)}
    )
    deadline_failure = (
        None
        if cleanup_deadline is None
        else {
            "type": type(cleanup_deadline).__name__,
            "message": str(cleanup_deadline),
            "observed_at_ns": cleanup_observed_ns,
            "absolute_deadline_ns": watchdog.deadline_ns,
        }
    )
    return {
        "config_id": configuration.config_id,
        "status": failure_status,
        "accepted": False,
        "operation": configuration.operation,
        "topology_id": configuration.topology.topology_id,
        "controller_wall_seconds": (
            time.clock_gettime_ns(time.CLOCK_MONOTONIC) - started_ns
        )
        / 1_000_000_000,
        "configuration_timeout_seconds": args.configuration_timeout_seconds,
        "reason": "; ".join(reasons),
        "original_failure": original_failure,
        "sweep_deadline_failure": deadline_failure,
        "cleanup": {
            "bounded": True,
            "sigterm_wait_seconds": CLEANUP_TERM_WAIT_SECONDS,
            "sigkill_wait_seconds": CLEANUP_KILL_WAIT_SECONDS,
            "completed": cleanup_error is None,
            "error": (
                None
                if cleanup_error is None
                else f"{type(cleanup_error).__name__}: {cleanup_error}"
            ),
        },
        "worker_processes": worker_failure_evidence,
        "completed_call_event_waves": retained_events,
        "controller_release_ns_by_completed_wave": retained_releases,
        "live_process_memory": retained_memory,
    }


def _configuration_memory_plan(
    args: argparse.Namespace, configuration: Configuration
) -> dict[str, Any]:
    per_process = []
    for group_index, environment_tile in enumerate(
        configuration.topology.environment_tiles
    ):
        case = EXACT.Case(
            configuration.operation,
            args.n,
            args.block_width,
            args.probe_tile,
            environment_tile,
            configuration.topology.threads_per_process,
        )
        per_process.append(
            {
                "group_index": group_index,
                "environment_tile": environment_tile,
                "case": asdict(case),
                "cblas": case.cblas,
                "panel_columns": case.panel_columns,
                "flops_per_call": case.flops,
                "memory": EXACT._estimated_memory(case),
            }
        )
    tree = sum(value["memory"]["conservative_peak_bytes"] for value in per_process)
    per_limit = int(args.max_memory_gib_per_process * 1024**3)
    tree_limit = int(args.max_process_tree_memory_gib * 1024**3)
    return {
        "per_process": per_process,
        "process_count": configuration.topology.process_count,
        "aggregate_conservative_peak_bytes": tree,
        "per_process_limit_bytes": per_limit,
        "process_tree_limit_bytes": tree_limit,
        "fits_per_process_limit": all(
            value["memory"]["conservative_peak_bytes"] <= per_limit
            for value in per_process
        ),
        "fits_process_tree_limit": tree <= tree_limit,
    }


def _configurations(topologies: Sequence[Topology]) -> list[Configuration]:
    return [
        Configuration(topology, operation)
        for topology in topologies
        for operation in ("source", "target")
    ]


def _topology_summaries(
    topologies: Sequence[Topology], configurations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for topology in topologies:
        selected = {
            value.get("operation"): value
            for value in configurations
            if value.get("topology_id") == topology.topology_id
        }
        if set(selected) != {"source", "target"} or any(
            value.get("accepted") is not True for value in selected.values()
        ):
            summaries.append(
                {
                    "topology_id": topology.topology_id,
                    "status": "incomplete_rejected",
                    "accepted": False,
                    "production_promotable": topology.production_promotable,
                    "non_promotable_reason": topology.non_promotable_reason,
                    "available_operations": sorted(selected),
                }
            )
            continue
        measured_waves = [
            wave
            for operation in ("source", "target")
            for wave in selected[operation]["synchronization_and_throughput"]["waves"]
            if wave["phase"] == "measured"
        ]
        makespan = math.fsum(
            wave["wrapper_makespan_seconds"] for wave in measured_waves
        )
        total_flops = sum(wave["total_flops"] for wave in measured_waves)
        process_cpu = math.fsum(
            wave["aggregate_process_cpu_seconds"] for wave in measured_waves
        )
        if makespan <= 0 or total_flops <= 0 or process_cpu <= 0:
            raise RuntimeError("accepted topology summary has nonpositive measurements")
        peak_rss = max(
            selected[operation]["live_process_memory"]["peak_combined_rss_snapshot"][
                "combined_rss_bytes"
            ]
            for operation in ("source", "target")
        )
        peak_pss = max(
            selected[operation]["live_process_memory"]["peak_combined_pss_snapshot"][
                "combined_pss_bytes"
            ]
            for operation in ("source", "target")
        )
        summaries.append(
            {
                "topology_id": topology.topology_id,
                "status": "accepted",
                "accepted": True,
                "benchmark_acceptance_scope": (
                    "production_candidate_kernel_screen"
                    if topology.production_promotable
                    else "kernel_only_non_promotable_pruning_screen"
                ),
                "production_promotable": topology.production_promotable,
                "non_promotable_reason": topology.non_promotable_reason,
                "process_count": topology.process_count,
                "threads_per_process": topology.threads_per_process,
                "total_physical_cores": topology.total_physical_cores,
                "source_target_measured_wave_count": len(measured_waves),
                "source_target_cumulative_kernel_makespan_seconds": makespan,
                "source_target_total_flops": total_flops,
                "source_target_aggregate_gflops_per_second": total_flops
                / (makespan * 1.0e9),
                "source_target_aggregate_process_cpu_seconds": process_cpu,
                "source_target_aggregate_active_core_equivalents": process_cpu
                / makespan,
                "maximum_measured_worker_start_skew_seconds": max(
                    wave["worker_start_skew_seconds"] for wave in measured_waves
                ),
                "maximum_measured_straggler_slowest_over_fastest_ratio": max(
                    wave["straggler"]["slowest_over_fastest_ratio"]
                    for wave in measured_waves
                ),
                "source_target_controller_wall_seconds": math.fsum(
                    selected[operation]["controller_wall_seconds"]
                    for operation in ("source", "target")
                ),
                "maximum_sampled_combined_rss_bytes_across_operations": peak_rss,
                "maximum_sampled_combined_pss_bytes_across_operations": peak_pss,
                "operation_config_ids": {
                    operation: selected[operation]["config_id"]
                    for operation in ("source", "target")
                },
            }
        )
    return summaries


def _static_provenance(
    args: argparse.Namespace, topologies: Sequence[Topology]
) -> dict[str, Any]:
    from shutil import which

    prefix = args.install_prefix.resolve()
    module = args.native_module.resolve()
    archive = args.private_archive.resolve()
    python = EXACT._regular_file(args.python_executable, executable=True)
    taskset_raw = which("taskset")
    if taskset_raw is None:
        raise RuntimeError("taskset is required")
    taskset = EXACT._regular_file(Path(taskset_raw), executable=True)
    package = EXACT._package_identity(prefix)
    if package["manifest_sha256"] != args.expected_package_manifest_sha256:
        raise RuntimeError("installed package changed during static preflight")
    identities = {
        "native_module": EXACT._file_identity(module),
        "private_archive": EXACT._file_identity(archive),
        "runner": EXACT._file_identity(Path(__file__).resolve()),
        "exact_layout_harness": EXACT._file_identity(Path(EXACT.__file__).resolve()),
        "python_executable": EXACT._file_identity(python),
        "taskset_executable": EXACT._file_identity(taskset),
    }
    if identities["native_module"]["sha256"] != args.expected_native_sha256:
        raise RuntimeError("native module changed during static preflight")
    if identities["private_archive"]["sha256"] != args.expected_archive_sha256:
        raise RuntimeError("private archive changed during static preflight")
    return {
        "installed_package": package,
        **identities,
        "expected_native_sha256": args.expected_native_sha256,
        "expected_package_manifest_sha256": args.expected_package_manifest_sha256,
        "expected_archive_sha256": args.expected_archive_sha256,
        "expected_source_commit": args.expected_source_commit,
        "expected_source_tree_sha256": args.expected_source_tree_sha256,
        "expected_private_source_commit": args.expected_private_source_commit,
        "expected_private_source_tree_sha256": args.expected_private_source_tree_sha256,
        "expected_backend": args.expected_backend,
        "linkage": EXACT._linkage_evidence(module, archive, args.expected_backend),
        "topologies": [asdict(topology) for topology in topologies],
        "environment_persistence_policy": {
            "whole_environment_persisted": False,
            "allowlisted_thread_settings_only": True,
            "credentials_persisted": False,
        },
    }


def _base_report(
    args: argparse.Namespace,
    topologies: Sequence[Topology],
    provenance: Mapping[str, Any],
    dependencies: Sequence[Path],
) -> dict[str, Any]:
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
            "controller_affinity": EXACT._compress_ints(
                sorted(os.sched_getaffinity(0))
            ),
        },
        "protocol": {
            "current_layout_only": True,
            "warmups_per_worker": EXACT.WARMUPS,
            "measured_repeats_per_worker": EXACT.MEASURED_REPEATS,
            "barrier_before_every_worker_call": True,
            "source_and_target_each_screened": True,
            "configuration_count": 8,
            "configuration_timeout_seconds": args.configuration_timeout_seconds,
            "maximum_configuration_timeout_seconds": MAX_CONFIGURATION_SECONDS,
            "sweep_timeout_seconds": args.sweep_timeout_seconds,
            "maximum_sweep_timeout_seconds": MAX_SWEEP_SECONDS,
            "fresh_exec_no_site_per_worker": True,
            "private_static_blas_per_process_required": True,
            "disjoint_physical_cores_required": True,
            "api9_singleton_placement_required": True,
            "early_verified_membind_required": True,
            "integrity_and_zero_repairs_required": True,
            "native_integrity_snapshot_numa_contract_required": True,
            "native_protected_output_numa_contract_required": True,
            "one_output_evidence_record_per_protected_call": True,
            "controller_revalidates_worker_native_numa_evidence": True,
            "post_vendor_abc_sampler_remains_independent": True,
            "full_oracle_and_bitwise_repeatability_required": True,
            "cross_worker_input_output_oracle_equality_required": False,
            "five_environment_work_partition_required": True,
            "four_by_eight_kernel_only_non_promotable": True,
            "minimum_active_core_fraction_per_measured_worker_call": EXACT.MINIMUM_ACTIVE_CORE_FRACTION,
            "maximum_worker_start_skew_seconds": args.max_start_skew_seconds,
            "live_combined_rss_pss_sampling_required": True,
            "atomic_no_replace_publication": True,
            "stop_after_first_rejected_configuration": True,
            "sweep_timeout_rejection_publication_required": True,
            "sweep_deadline_clock": "CLOCK_MONOTONIC",
            "controller_owned_absolute_deadline_watchdog_required": True,
            "watchdog_selector_wake_self_pipe_required": True,
            "watchdog_signal_delivery_required": False,
            "publication_outside_measured_sweep_cap": True,
            "cross_process_timing_clock": time.get_clock_info(
                "monotonic"
            ).implementation,
        },
        "shape": {
            "n_samples": args.n,
            "block_width": args.block_width,
            "probe_tile": args.probe_tile,
            "total_environment_count": 5,
            "environment_tiles_by_topology": {
                topology.topology_id: list(topology.environment_tiles)
                for topology in topologies
            },
        },
        "topologies": [asdict(topology) for topology in topologies],
        "dependency_paths": [str(path) for path in dependencies],
        "provenance": dict(provenance),
        "configurations": [],
    }


def _dry_run_report(
    args: argparse.Namespace,
    topologies: Sequence[Topology],
    provenance: Mapping[str, Any],
    dependencies: Sequence[Path],
    taskset: Path,
    watchdog: _SweepDeadlineWatchdog,
) -> dict[str, Any]:
    watchdog.raise_if_expired("before dry-run report construction")
    report = _base_report(args, topologies, provenance, dependencies)
    report.update(
        {
            "dry_run": True,
            "scientific_execution": False,
            "status": "validated_dry_run",
            "accepted": False,
            "acceptance_reason": "dry_run_performs_no_numerical_execution",
        }
    )
    report["topology_summaries"] = [
        {
            "topology_id": topology.topology_id,
            "status": "planned",
            "accepted": False,
            "production_promotable": topology.production_promotable,
            "non_promotable_reason": topology.non_promotable_reason,
        }
        for topology in topologies
    ]
    runner_sha = provenance["runner"]["sha256"]
    harness_sha = provenance["exact_layout_harness"]["sha256"]
    for configuration in _configurations(topologies):
        watchdog.raise_if_expired(
            f"before planning dry-run configuration {configuration.config_id}"
        )
        plan = _configuration_memory_plan(args, configuration)
        report["configurations"].append(
            {
                "config_id": configuration.config_id,
                "status": "planned",
                "accepted": False,
                "topology": asdict(configuration.topology),
                "operation": configuration.operation,
                "memory_plan": plan,
                "worker_commands": [
                    _worker_command(
                        args,
                        configuration,
                        group_index,
                        taskset,
                        f"<control-read-fd-{group_index}>",
                        f"<event-write-fd-{group_index}>",
                        runner_sha,
                        harness_sha,
                    )
                    for group_index in range(configuration.topology.process_count)
                ],
            }
        )
    watchdog.raise_if_expired("after dry-run report construction")
    return report


def _artifact_stability_evidence(
    args: argparse.Namespace, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    checks = (
        (
            "installed_package",
            lambda: EXACT._package_identity(args.install_prefix.resolve()),
        ),
        (
            "native_module",
            lambda: EXACT._file_identity(args.native_module.resolve()),
        ),
        (
            "private_archive",
            lambda: EXACT._file_identity(args.private_archive.resolve()),
        ),
        ("runner", lambda: EXACT._file_identity(Path(__file__).resolve())),
        (
            "exact_layout_harness",
            lambda: EXACT._file_identity(Path(EXACT.__file__).resolve()),
        ),
        (
            "python_executable",
            lambda: EXACT._file_identity(args.python_executable.resolve()),
        ),
        (
            "taskset_executable",
            lambda: EXACT._file_identity(
                Path(provenance["taskset_executable"]["path"])
            ),
        ),
    )
    records: list[dict[str, Any]] = []
    for name, observe in checks:
        expected = provenance.get(name)
        try:
            observed = observe()
            error = None
            matches = observed == expected
        except Exception as caught:
            observed = None
            error = f"{type(caught).__name__}: {caught}"
            matches = False
        records.append(
            {
                "artifact": name,
                "expected": expected,
                "observed": observed,
                "matches": matches,
                "error": error,
            }
        )
    return {
        "stable": all(record["matches"] is True for record in records),
        "checks": records,
        "mismatched_artifacts": [
            record["artifact"]
            for record in records
            if record["matches"] is not True
        ],
    }


def _artifacts_stable(args: argparse.Namespace, provenance: Mapping[str, Any]) -> bool:
    return bool(_artifact_stability_evidence(args, provenance)["stable"])


def _controller(
    args: argparse.Namespace,
    topologies: Sequence[Topology],
    provenance: Mapping[str, Any],
    dependencies: Sequence[Path],
    taskset: Path,
    watchdog: _SweepDeadlineWatchdog,
) -> dict[str, Any]:
    report = _base_report(args, topologies, provenance, dependencies)
    report["dry_run"] = False
    report["scientific_execution"] = True
    runner_sha = provenance["runner"]["sha256"]
    harness_sha = provenance["exact_layout_harness"]["sha256"]
    current_configuration: Configuration | None = None
    planned_configurations: list[Configuration] | None = None
    deadline_error: _SweepDeadlineExpired | None = None
    deadline_event: dict[str, Any] | None = None
    controller_error_event: dict[str, Any] | None = None
    try:
        watchdog.raise_if_expired("before configuration planning")
        planned_configurations = _configurations(topologies)
        watchdog.raise_if_expired("after configuration planning")
        for configuration in planned_configurations:
            current_configuration = configuration
            watchdog.raise_if_expired(
                f"before configuration {configuration.config_id}"
            )
            memory = _configuration_memory_plan(args, configuration)
            watchdog.raise_if_expired(
                f"after memory planning for {configuration.config_id}"
            )
            if (
                not memory["fits_per_process_limit"]
                or not memory["fits_process_tree_limit"]
            ):
                report["configurations"].append(
                    {
                        "config_id": configuration.config_id,
                        "operation": configuration.operation,
                        "topology_id": configuration.topology.topology_id,
                        "status": "rejected_memory_budget",
                        "accepted": False,
                        "memory_plan": memory,
                    }
                )
                break
            try:
                result = _run_configuration(
                    args,
                    configuration,
                    taskset,
                    runner_sha,
                    harness_sha,
                    watchdog,
                )
            except _SweepDeadlineExpired:
                raise
            except Exception as error:
                result = {
                    "config_id": configuration.config_id,
                    "operation": configuration.operation,
                    "topology_id": configuration.topology.topology_id,
                    "status": "controller_failure",
                    "accepted": False,
                    "reason": f"{type(error).__name__}: {error}",
                }
            result["memory_plan"] = memory
            report["configurations"].append(result)
            if result.get("status") == "sweep_timeout":
                deadline_error = _SweepDeadlineExpired(
                    str(result.get("reason", "configuration hit sweep deadline"))
                )
                deadline_event = {
                    "timing": "during_configuration_or_cleanup",
                    "trigger_config_id": configuration.config_id,
                    "reason": str(deadline_error),
                }
            if result.get("accepted") is not True:
                break
    except _SweepDeadlineExpired as error:
        deadline_error = error
        deadline_event = {
            "timing": (
                "before_first_configuration"
                if current_configuration is None
                else "controller_watchdog"
            ),
            "trigger_config_id": (
                None
                if current_configuration is None
                else current_configuration.config_id
            ),
            "reason": f"{type(error).__name__}: {error}",
        }
        # A pre-first-configuration expiry remains sweep-level evidence.  Do
        # not fabricate a ninth configuration with a null identity.
        if current_configuration is not None:
            rejection: dict[str, Any] = {
                "config_id": current_configuration.config_id,
                "operation": current_configuration.operation,
                "topology_id": current_configuration.topology.topology_id,
                "status": "sweep_timeout",
                "accepted": False,
                "reason": f"{type(error).__name__}: {error}",
            }
            if (
                not report["configurations"]
                or report["configurations"][-1].get("config_id")
                != rejection["config_id"]
            ):
                report["configurations"].append(rejection)
    except Exception as error:
        controller_error_event = {
            "timing": (
                "before_first_configuration"
                if current_configuration is None
                else "during_controller_execution"
            ),
            "trigger_config_id": (
                None if current_configuration is None else current_configuration.config_id
            ),
            "reason": f"{type(error).__name__}: {error}",
        }
        if current_configuration is not None:
            rejection = {
                "config_id": current_configuration.config_id,
                "operation": current_configuration.operation,
                "topology_id": current_configuration.topology.topology_id,
                "status": "controller_failure",
                "accepted": False,
                "reason": controller_error_event["reason"],
            }
            if (
                not report["configurations"]
                or report["configurations"][-1].get("config_id")
                != rejection["config_id"]
            ):
                report["configurations"].append(rejection)

    # This locked sample is the linearization point between an on-time finish
    # and expiry.  Everything below is evidence aggregation/publication work
    # outside the measured sweep cap.
    finished_ns, finish_deadline = watchdog.finish_execution(
        "at controller execution finish"
    )
    if finish_deadline is not None and deadline_error is None:
        deadline_error = finish_deadline
    if finish_deadline is not None and deadline_event is None:
        deadline_event = {
            "timing": "at_controller_execution_finish",
            "trigger_config_id": (
                None
                if current_configuration is None
                else current_configuration.config_id
            ),
            "reason": f"{type(finish_deadline).__name__}: {finish_deadline}",
        }
    sweep_wall = (finished_ns - watchdog.started_ns) / 1_000_000_000
    deadline_reached = finished_ns >= watchdog.deadline_ns
    if planned_configurations is None:
        # Planning may itself be the expiry point.  Materialize the fixed 4x2
        # protocol only after the watchdog is disarmed so all eight truthful
        # not-run stubs can still be published.
        planned_configurations = [
            Configuration(topology, operation)
            for topology in topologies
            for operation in ("source", "target")
        ]
    observed_ids = {
        value.get("config_id")
        for value in report["configurations"]
        if isinstance(value.get("config_id"), str)
    }
    rejected_actual = next(
        (
            value
            for value in report["configurations"]
            if value.get("accepted") is not True
        ),
        None,
    )
    if len(observed_ids) < len(planned_configurations) and (
        rejected_actual is not None
        or deadline_error is not None
        or controller_error_event is not None
    ):
        trigger_config_id = (
            rejected_actual.get("config_id")
            if rejected_actual is not None
            else (
                None
                if current_configuration is None
                else current_configuration.config_id
            )
        )
        trigger_status = (
            rejected_actual.get("status")
            if rejected_actual is not None
            else "sweep_timeout"
            if deadline_error is not None
            else "controller_failure"
        )
        for configuration in planned_configurations:
            if configuration.config_id in observed_ids:
                continue
            report["configurations"].append(
                {
                    "config_id": configuration.config_id,
                    "operation": configuration.operation,
                    "topology_id": configuration.topology.topology_id,
                    "topology": asdict(configuration.topology),
                    "status": "not_run_after_first_failure",
                    "accepted": False,
                    "trigger_config_id": trigger_config_id,
                    "trigger_status": trigger_status,
                    "reason": "sweep stopped without executing this configuration",
                }
            )
            observed_ids.add(configuration.config_id)
    post_run_validation_errors: list[str] = []
    try:
        artifact_evidence = _artifact_stability_evidence(args, provenance)
    except Exception as error:
        artifact_evidence = {
            "stable": False,
            "checks": [],
            "mismatched_artifacts": [],
            "validation_error": f"{type(error).__name__}: {error}",
        }
        post_run_validation_errors.append(
            f"artifact stability validation failed: {type(error).__name__}: {error}"
        )
    stable = artifact_evidence.get("stable") is True
    try:
        topology_summaries = _topology_summaries(
            topologies, report["configurations"]
        )
    except Exception as error:
        post_run_validation_errors.append(
            f"topology summary validation failed: {type(error).__name__}: {error}"
        )
        topology_summaries = [
            {
                "topology_id": topology.topology_id,
                "status": "summary_validation_rejected",
                "accepted": False,
                "production_promotable": topology.production_promotable,
                "non_promotable_reason": topology.non_promotable_reason,
                "reason": f"{type(error).__name__}: {error}",
            }
            for topology in topologies
        ]
    accepted = (
        len(report["configurations"]) == 8
        and all(value.get("accepted") is True for value in report["configurations"])
        and stable
        and not post_run_validation_errors
        and all(value.get("accepted") is True for value in topology_summaries)
        and deadline_error is None
        and not deadline_reached
    )
    rejected_configuration = any(
        value.get("accepted") is not True for value in report["configurations"]
    )
    report.update(
        {
            "sweep_wall_seconds": sweep_wall,
            "sweep_wall_exceeded_declared_cap": (
                deadline_reached
            ),
            "artifact_identity_stable": stable,
            "artifact_identity_evidence": artifact_evidence,
            "topology_summaries": topology_summaries,
            "post_run_validation_errors": post_run_validation_errors,
            "sweep_deadline_event": deadline_event,
            "sweep_deadline_watchdog": _watchdog_evidence(
                watchdog, finished_ns
            ),
            "controller_failure_event": controller_error_event,
            "configuration_execution_count": sum(
                value.get("status") != "not_run_after_first_failure"
                for value in report["configurations"]
            ),
            "status": (
                "accepted"
                if accepted
                else "sweep_timeout_rejected"
                if deadline_error is not None
                else "completed_rejected"
            ),
            "accepted": accepted,
            "first_failure_stopped_sweep": (
                rejected_configuration
                or deadline_error is not None
                or controller_error_event is not None
            ),
            "acceptance_reason": (
                "all eight synchronized topology-operation configurations passed"
                if accepted
                else "sweep deadline expired; rejection evidence was retained"
                if deadline_error is not None
                else "post-run artifact or summary validation failed"
                if post_run_validation_errors or not stable
                else "one or more required topology gates failed"
            ),
        }
    )
    return report


def _minimal_sweep_timeout_rejection(
    args: argparse.Namespace,
    topologies: Sequence[Topology],
    reason: str,
    *,
    provenance: Mapping[str, Any] | None,
    dependencies: Sequence[Path],
) -> dict[str, Any]:
    if provenance is not None:
        report = _base_report(args, topologies, provenance, dependencies)
    else:
        report = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "benchmark_only": True,
            "production_run": False,
            "configurations": [],
        }
    report.update(
        {
            "dry_run": bool(args.dry_run),
            "scientific_execution": not args.dry_run,
            "status": "sweep_timeout_rejected",
            "accepted": False,
            "acceptance_reason": reason,
            "first_failure_stopped_sweep": True,
            "topology_summaries": [],
        }
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args._worker:
        _validate_worker_args(parser, args)
        try:
            result = _run_worker(args)
        except Exception as error:
            try:
                _write_event(
                    args._event_write_fd,
                    {
                        "event": "error",
                        "pid": os.getpid(),
                        "reason": f"{type(error).__name__}: {error}",
                    },
                )
            except Exception:
                pass
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(EXACT._json_safe(result), sort_keys=True, allow_nan=False)
        )
        return 0

    # Validate the clock budget before starting its owner; all dependency,
    # identity, and provenance preflight that follows is inside this deadline.
    if not 0 < args.sweep_timeout_seconds <= MAX_SWEEP_SECONDS:
        parser.error("--sweep-timeout-seconds must be in (0, 1200]")
    watchdog = _SweepDeadlineWatchdog.start(args.sweep_timeout_seconds)
    topologies: Sequence[Topology] | None = None
    dependencies: list[Path] = []
    provenance: Mapping[str, Any] | None = None
    payload: dict[str, Any] | None = None
    deadline_error: _SweepDeadlineExpired | None = None
    execution_error: BaseException | None = None
    watchdog_shutdown_error: BaseException | None = None
    try:
        watchdog.raise_if_expired("before controller argument preflight")
        topologies = _validate_controller_args(parser, args)
        watchdog.raise_if_expired("after controller argument preflight")
        dependencies = EXACT._dependency_paths(args)
        watchdog.raise_if_expired("after dependency preflight")
        provenance = _static_provenance(args, topologies)
        watchdog.raise_if_expired("after provenance preflight")
        taskset = Path(provenance["taskset_executable"]["path"])
        payload = (
            _dry_run_report(
                args,
                topologies,
                provenance,
                dependencies,
                taskset,
                watchdog,
            )
            if args.dry_run
            else _controller(
                args,
                topologies,
                provenance,
                dependencies,
                taskset,
                watchdog,
            )
        )
    except _SweepDeadlineExpired as error:
        deadline_error = error
    except BaseException as error:
        execution_error = error
    finally:
        try:
            _finished_ns, finish_deadline = watchdog.finish_execution(
                "at measured execution finish"
            )
            if finish_deadline is not None and deadline_error is None:
                deadline_error = finish_deadline
        except BaseException as error:
            watchdog_shutdown_error = error
        try:
            watchdog.cancel_join_close()
        except BaseException as error:
            # A live watchdog retains ownership of its descriptors; do not
            # publish while it could still notify a subsequently reused FD.
            watchdog_shutdown_error = watchdog_shutdown_error or error
    if watchdog_shutdown_error is not None:
        raise watchdog_shutdown_error
    if deadline_error is None and execution_error is not None:
        raise execution_error
    if deadline_error is not None and (
        payload is None or payload.get("status") != "sweep_timeout_rejected"
    ):
        payload = _minimal_sweep_timeout_rejection(
            args,
            [] if topologies is None else topologies,
            f"{type(deadline_error).__name__}: {deadline_error}",
            provenance=provenance,
            dependencies=dependencies,
        )
    if deadline_error is not None and execution_error is not None:
        if payload is None:
            raise RuntimeError("deadline rejection payload was not constructed")
        payload["concurrent_execution_failure"] = {
            "type": type(execution_error).__name__,
            "message": str(execution_error),
        }
    if payload is None:
        raise RuntimeError("controller completed without a report payload")
    payload["sweep_deadline_watchdog"] = _watchdog_evidence(
        watchdog, _finished_ns
    )
    # Atomic no-replace publication is deliberately outside the measured cap
    # and occurs only after the watchdog thread has stopped and both private
    # descriptors have closed.
    EXACT._atomic_json_no_replace(payload, args.output)
    print(args.output.expanduser().resolve())
    return (
        0
        if payload.get("accepted") is True
        or payload.get("status") == "validated_dry_run"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
