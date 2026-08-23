"""Fail-closed NUMA binding before numerical runtimes are imported.

This module deliberately uses only the Python standard library so the CLI and
benchmark bootstraps can import it before NumPy, pandas, or native extensions.
"""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_ATTESTATION_SCHEMA = "summit.numa_policy_attestation.v1"
_APPLIED_ENV = "SUMMIT_NUMA_POLICY_APPLIED"
_PROVENANCE_ENV = "SUMMIT_NUMA_POLICY_PROVENANCE"
_PRE_NUMERIC_IMPORT = "pre_numeric_import"
_MPOL_BIND = 2
_MPOL_F_ADDR = 1 << 1
_MPOL_F_STATIC_NODES = 1 << 15
_MPOL_MF_STRICT = 1 << 0
_BOUND_BUFFER_SCHEMA = "summit.numa_bound_anonymous_buffer.v1"
_NUMERIC_MODULE_ROOTS = frozenset(
    {
        "bed_reader",
        "gwldcore",
        "gxeldcore",
        "numpy",
        "pandas",
        "scipy",
        "winldcore",
    }
)


@dataclass(frozen=True, slots=True)
class _NumaPolicyAttestation:
    mode: str
    requested_nodes: str
    effective_nodes: tuple[int, ...]
    task_count_at_application: int
    applied_policy: str
    pid: int
    static_nodes: bool


_ATTESTATION: _NumaPolicyAttestation | None = None


def _normalized_nodes_request(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        raise ValueError("NUMA node list must not be empty.")
    return text


def _parse_node_list(value: object) -> tuple[int, ...]:
    text = _normalized_nodes_request(value)
    if text == "all":
        raise ValueError("'all' must be resolved against Mems_allowed_list.")
    parsed: set[int] = set()
    for component in text.split(","):
        part = component.strip()
        if not part:
            raise ValueError("NUMA node list contains an empty component.")
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start, stop = int(left), int(right)
            except ValueError as exc:
                raise ValueError(f"Invalid NUMA node range {part!r}.") from exc
            if start < 0 or stop < start:
                raise ValueError(f"Invalid NUMA node range {part!r}.")
            parsed.update(range(start, stop + 1))
        else:
            try:
                node = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid NUMA node {part!r}.") from exc
            if node < 0:
                raise ValueError("NUMA nodes must be non-negative.")
            parsed.add(node)
    if not parsed:
        raise ValueError("NUMA node list must select at least one node.")
    return tuple(sorted(parsed))


def _read_mems_allowed() -> tuple[int, ...]:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(
            "Cannot verify NUMA Mems_allowed_list before numerical imports."
        ) from exc
    for line in lines:
        if line.startswith("Mems_allowed_list:"):
            value = line.split(":", 1)[1].strip()
            try:
                return _parse_node_list(value)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid /proc/self/status Mems_allowed_list {value!r}."
                ) from exc
    raise RuntimeError("/proc/self/status has no Mems_allowed_list entry.")


def _read_task_count() -> int:
    try:
        count = sum(
            entry.name.isdigit() for entry in Path("/proc/self/task").iterdir()
        )
    except OSError as exc:
        raise RuntimeError(
            "Cannot verify the process task count before applying NUMA policy."
        ) from exc
    if count < 1:
        raise RuntimeError("The process task list is unexpectedly empty.")
    return count


def _loaded_numeric_modules() -> tuple[str, ...]:
    loaded = []
    for name in sys.modules:
        root = name.split(".", 1)[0]
        leaf = name.rsplit(".", 1)[-1]
        if root in _NUMERIC_MODULE_ROOTS or leaf in _NUMERIC_MODULE_ROOTS:
            loaded.append(name)
    return tuple(sorted(loaded))


class _LibnumaBindings:
    def __init__(self) -> None:
        try:
            library = ctypes.CDLL("libnuma.so.1", use_errno=True)
        except OSError as exc:
            raise RuntimeError(
                "Explicit --numa-mode=membind requires libnuma.so.1."
            ) from exc
        library.numa_available.argtypes = []
        library.numa_available.restype = ctypes.c_int
        if int(library.numa_available()) < 0:
            raise RuntimeError("libnuma reports that NUMA policy is unavailable.")
        library.numa_max_node.argtypes = []
        library.numa_max_node.restype = ctypes.c_int
        library.set_mempolicy.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
        ]
        library.set_mempolicy.restype = ctypes.c_long
        library.get_mempolicy.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        library.get_mempolicy.restype = ctypes.c_long
        library.mbind.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_uint,
        ]
        library.mbind.restype = ctypes.c_long
        library.numa_move_pages.argtypes = [
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        library.numa_move_pages.restype = ctypes.c_int
        self._library = library

    def maximum_node(self) -> int:
        maximum = int(self._library.numa_max_node())
        if maximum < 0:
            raise RuntimeError("libnuma returned an invalid maximum NUMA node.")
        return maximum

    @staticmethod
    def _raw_mask(
        nodes: tuple[int, ...], maximum_node: int
    ) -> tuple[object, int]:
        maxnode = maximum_node + 1
        bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
        words = max(1, (maxnode + bits_per_word - 1) // bits_per_word)
        raw = (ctypes.c_ulong * words)()
        for node in nodes:
            raw[node // bits_per_word] |= 1 << (node % bits_per_word)
        return raw, maxnode

    def set_static_membind(self, nodes: tuple[int, ...]) -> None:
        raw, maxnode = self._raw_mask(nodes, self.maximum_node())
        ctypes.set_errno(0)
        result = self._library.set_mempolicy(
            _MPOL_BIND | _MPOL_F_STATIC_NODES, raw, maxnode
        )
        error = ctypes.get_errno()
        if result != 0:
            raise OSError(error or 1, os.strerror(error or 1))

    def query_policy(
        self, *, address: int | None = None
    ) -> tuple[int, tuple[int, ...]]:
        maximum_node = self.maximum_node()
        raw, maxnode = self._raw_mask((), maximum_node)
        mode = ctypes.c_int(-1)
        flags = _MPOL_F_ADDR if address is not None else 0
        ctypes.set_errno(0)
        result = self._library.get_mempolicy(
            ctypes.byref(mode),
            raw,
            maxnode,
            None if address is None else ctypes.c_void_p(address),
            flags,
        )
        error = ctypes.get_errno()
        if result != 0:
            raise OSError(error or 1, os.strerror(error or 1))
        return mode.value, tuple(
            node
            for node in range(maximum_node + 1)
            if raw[node // (ctypes.sizeof(ctypes.c_ulong) * 8)]
            & (1 << (node % (ctypes.sizeof(ctypes.c_ulong) * 8)))
        )

    def bind_memory_range(
        self,
        address: int,
        length: int,
        nodes: tuple[int, ...],
        *,
        strict: bool,
    ) -> None:
        raw, maxnode = self._raw_mask(nodes, self.maximum_node())
        flags = _MPOL_MF_STRICT if strict else 0
        ctypes.set_errno(0)
        result = self._library.mbind(
            ctypes.c_void_p(address),
            length,
            _MPOL_BIND | _MPOL_F_STATIC_NODES,
            raw,
            maxnode,
            flags,
        )
        error = ctypes.get_errno()
        if result != 0:
            raise OSError(error or 1, os.strerror(error or 1))

    def query_page_nodes(
        self, address: int, page_count: int, page_size: int, *, chunk_pages: int
    ) -> tuple[dict[int, int], str, int]:
        histogram: dict[int, int] = {}
        digest = hashlib.sha256()
        chunks = 0
        for start in range(0, page_count, chunk_pages):
            count = min(chunk_pages, page_count - start)
            pages = (ctypes.c_void_p * count)(
                *(
                    address + (start + offset) * page_size
                    for offset in range(count)
                )
            )
            statuses = (ctypes.c_int * count)()
            ctypes.set_errno(0)
            result = self._library.numa_move_pages(
                0, count, pages, None, statuses, 0
            )
            error = ctypes.get_errno()
            if result != 0:
                raise OSError(error or 1, os.strerror(error or 1))
            raw_statuses = ctypes.string_at(
                ctypes.addressof(statuses), count * ctypes.sizeof(ctypes.c_int)
            )
            digest.update(raw_statuses)
            for raw_status in statuses:
                status = int(raw_status)
                if status < 0:
                    raise OSError(-status, os.strerror(-status))
                histogram[status] = histogram.get(status, 0) + 1
            chunks += 1
        return histogram, digest.hexdigest(), chunks

def _load_libnuma() -> _LibnumaBindings:
    return _LibnumaBindings()


def _attestation_for_current_process() -> _NumaPolicyAttestation | None:
    attestation = _ATTESTATION
    if attestation is not None and attestation.pid != os.getpid():
        raise RuntimeError(
            "A NUMA policy attestation inherited across fork is not valid in this process."
        )
    return attestation


def _publish_attestation(attestation: _NumaPolicyAttestation) -> None:
    global _ATTESTATION
    os.environ[_APPLIED_ENV] = attestation.applied_policy
    os.environ[_PROVENANCE_ENV] = _PRE_NUMERIC_IMPORT
    _ATTESTATION = attestation


def apply_early_numa_membind(nodes: object = "all") -> dict[str, object]:
    """Apply and attest a strict membind before numerical runtimes are imported."""
    requested_nodes = _normalized_nodes_request(nodes)
    existing = _attestation_for_current_process()
    if existing is not None:
        if existing.mode != "membind" or existing.requested_nodes != requested_nodes:
            raise RuntimeError(
                "Conflicting NUMA policy request after early policy attestation: "
                f"attested {existing.mode}:{existing.requested_nodes}, requested "
                f"membind:{requested_nodes}."
            )
        _publish_attestation(existing)
        result = current_numa_policy_attestation()
        assert result is not None
        return result

    loaded_numeric_modules = _loaded_numeric_modules()
    if loaded_numeric_modules:
        raise RuntimeError(
            "Explicit NUMA membind must precede numerical imports; already loaded: "
            + ", ".join(loaded_numeric_modules[:8])
        )

    allowed_nodes = _read_mems_allowed()
    selected_nodes = (
        allowed_nodes if requested_nodes == "all" else _parse_node_list(requested_nodes)
    )
    if not set(selected_nodes).issubset(allowed_nodes):
        raise ValueError(
            "Requested NUMA membind nodes must be a subset of Mems_allowed_list; "
            f"requested={list(selected_nodes)}, allowed={list(allowed_nodes)}."
        )

    library = _load_libnuma()
    maximum_node = library.maximum_node()
    if selected_nodes[-1] > maximum_node:
        raise ValueError(
            f"Requested NUMA node {selected_nodes[-1]} exceeds libnuma maximum "
            f"node {maximum_node}."
        )

    task_count = _read_task_count()
    if task_count != 1:
        raise RuntimeError(
            "Explicit NUMA membind must be applied before worker creation; "
            f"observed {task_count} process tasks."
        )
    library.set_static_membind(selected_nodes)
    observed_mode, observed_static_nodes = library.query_policy()
    if (
        observed_mode != (_MPOL_BIND | _MPOL_F_STATIC_NODES)
        or observed_static_nodes != selected_nodes
    ):
        raise RuntimeError(
            "libnuma static membind verification failed: "
            f"mode={observed_mode}, nodes={list(observed_static_nodes)}."
        )

    applied_policy = "libnuma:membind:" + ",".join(map(str, selected_nodes))
    _publish_attestation(
        _NumaPolicyAttestation(
            mode="membind",
            requested_nodes=requested_nodes,
            effective_nodes=selected_nodes,
            task_count_at_application=task_count,
            applied_policy=applied_policy,
            pid=os.getpid(),
            static_nodes=True,
        )
    )
    result = current_numa_policy_attestation()
    assert result is not None
    return result


def current_numa_policy_attestation() -> dict[str, object] | None:
    """Return a JSON-safe copy of the immutable in-process attestation."""
    attestation = _attestation_for_current_process()
    if attestation is None:
        return None
    return {
        "schema": _ATTESTATION_SCHEMA,
        "mode": attestation.mode,
        "requested_nodes": attestation.requested_nodes,
        "effective_nodes": list(attestation.effective_nodes),
        "task_count_at_application": attestation.task_count_at_application,
        "applied_before_numeric_import": True,
        "verified": True,
        "source": "libnuma",
        "applied_policy": attestation.applied_policy,
        "static_nodes": attestation.static_nodes,
        "pid": attestation.pid,
    }


def attest_numa_policy_request(mode: object, nodes: object = "all") -> bool:
    """Validate a later request against the in-process early attestation."""
    attestation = _attestation_for_current_process()
    if attestation is None:
        return False
    requested_mode = str(mode).strip().lower() if mode is not None else ""
    requested_nodes = _normalized_nodes_request(nodes)
    if (
        requested_mode != attestation.mode
        or requested_nodes != attestation.requested_nodes
    ):
        raise RuntimeError(
            "Runtime NUMA policy conflicts with the pre-numeric-import "
            f"attestation: attested {attestation.mode}:{attestation.requested_nodes}, "
            f"requested {requested_mode or '<none>'}:{requested_nodes}."
        )
    _publish_attestation(attestation)
    return True


def _validated_bound_nodes(nodes: object) -> tuple[int, ...]:
    if not isinstance(nodes, (tuple, list)) or not nodes:
        raise TypeError("NUMA-bound buffers require a nonempty integer node list.")
    if any(isinstance(node, bool) or not isinstance(node, int) for node in nodes):
        raise TypeError("NUMA-bound buffer node IDs must be built-in integers.")
    normalized = tuple(nodes)
    if normalized != tuple(sorted(set(normalized))) or normalized[0] < 0:
        raise ValueError("NUMA-bound buffer node IDs must be sorted and unique.")
    attestation = _attestation_for_current_process()
    if (
        attestation is None
        or attestation.mode != "membind"
        or attestation.static_nodes is not True
        or attestation.effective_nodes != normalized
    ):
        raise RuntimeError(
            "NUMA-bound buffers require the exact live pre-numeric-import "
            "static membind attestation."
        )
    if not set(normalized).issubset(_read_mems_allowed()):
        raise RuntimeError(
            "NUMA-bound buffer nodes escaped the current Mems_allowed_list."
        )
    return normalized


def allocate_numa_bound_anonymous_buffer(
    byte_count: int, nodes: object
) -> tuple[mmap.mmap, dict[str, object]]:
    """Allocate a dedicated page-aligned mapping and bind it before first touch."""
    if isinstance(byte_count, bool) or not isinstance(byte_count, int):
        raise TypeError("NUMA-bound buffer byte_count must be a built-in integer.")
    if byte_count <= 0:
        raise ValueError("NUMA-bound buffer byte_count must be positive.")
    selected_nodes = _validated_bound_nodes(nodes)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    if page_size <= 0:
        raise RuntimeError("The operating system returned an invalid page size.")
    mapping_bytes = ((byte_count + page_size - 1) // page_size) * page_size
    owner = mmap.mmap(
        -1,
        mapping_bytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        address = ctypes.addressof(ctypes.c_char.from_buffer(owner))
        if address % page_size:
            raise RuntimeError("The anonymous NUMA mapping is not page aligned.")
        maximum_address = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
        if address > maximum_address - mapping_bytes:
            raise RuntimeError("The anonymous NUMA mapping address range overflows.")
        library = _load_libnuma()
        live_mode, live_nodes = library.query_policy()
        expected_mode = _MPOL_BIND | _MPOL_F_STATIC_NODES
        if live_mode != expected_mode or live_nodes != selected_nodes:
            raise RuntimeError(
                "The live owner-thread NUMA policy differs from the static "
                "membind attestation."
            )
        library.bind_memory_range(
            address, mapping_bytes, selected_nodes, strict=False
        )
        range_mode, range_nodes = library.query_policy(address=address)
        if range_mode != expected_mode or range_nodes != selected_nodes:
            raise RuntimeError(
                "The anonymous mapping did not retain its exact static bind policy."
            )
    except Exception:
        owner.close()
        raise
    return owner, {
        "schema": _BOUND_BUFFER_SCHEMA,
        "schema_version": 1,
        "byte_count": byte_count,
        "mapping_bytes": mapping_bytes,
        "page_size": page_size,
        "page_count": mapping_bytes // page_size,
        "selected_nodes": list(selected_nodes),
        "policy_mode": "bind_static_nodes",
        "page_aligned_mapping": True,
        "bound_before_first_touch": True,
        "live_owner_policy_verified": True,
        "range_policy_verified": True,
        "post_decode_complete_page_query": False,
        "page_migration_requested": False,
        "placement_repair_performed": False,
    }


def verify_numa_bound_anonymous_buffer(
    owner: mmap.mmap,
    byte_count: int,
    nodes: object,
    *,
    chunk_pages: int = 65536,
) -> dict[str, object]:
    """Verify every page after decode without migrating or changing its contents."""
    if not isinstance(owner, mmap.mmap):
        raise TypeError("NUMA-bound buffer verification requires an mmap owner.")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int):
        raise TypeError("NUMA-bound buffer byte_count must be a built-in integer.")
    if isinstance(chunk_pages, bool) or not isinstance(chunk_pages, int):
        raise TypeError("NUMA page-query chunk size must be a built-in integer.")
    if byte_count <= 0 or chunk_pages <= 0:
        raise ValueError("NUMA buffer size and page-query chunk must be positive.")
    selected_nodes = _validated_bound_nodes(nodes)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    mapping_bytes = len(owner)
    expected_mapping_bytes = (
        (byte_count + page_size - 1) // page_size
    ) * page_size
    if mapping_bytes != expected_mapping_bytes:
        raise RuntimeError(
            "NUMA-bound buffer mapping length differs from its declared byte count."
        )
    address = ctypes.addressof(ctypes.c_char.from_buffer(owner))
    if address % page_size:
        raise RuntimeError("The anonymous NUMA mapping lost page alignment.")
    maximum_address = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
    if address > maximum_address - mapping_bytes:
        raise RuntimeError("The anonymous NUMA mapping address range overflows.")
    library = _load_libnuma()
    expected_mode = _MPOL_BIND | _MPOL_F_STATIC_NODES
    live_mode, live_nodes = library.query_policy()
    range_mode, range_nodes = library.query_policy(address=address)
    if live_mode != expected_mode or live_nodes != selected_nodes:
        raise RuntimeError(
            "The live owner-thread NUMA policy changed before buffer verification."
        )
    if range_mode != expected_mode or range_nodes != selected_nodes:
        raise RuntimeError(
            "The NUMA-bound mapping policy changed after decoder first touch."
        )
    page_count = mapping_bytes // page_size
    histogram, status_sha256, chunks = library.query_page_nodes(
        address,
        page_count,
        page_size,
        chunk_pages=chunk_pages,
    )
    if sum(histogram.values()) != page_count:
        raise RuntimeError(
            "The complete NUMA page query returned an inconsistent page count."
        )
    unexpected_nodes = sorted(set(histogram).difference(selected_nodes))
    if unexpected_nodes:
        raise RuntimeError(
            "The NUMA-bound decoded buffer contains pages outside its selected "
            f"nodes: selected={list(selected_nodes)}, histogram={histogram}."
        )
    # MPOL_MF_STRICT is intentionally used without MPOL_MF_MOVE. Any mismatch
    # rejects; this acceptance boundary never repairs decoder placement.
    library.bind_memory_range(
        address, mapping_bytes, selected_nodes, strict=True
    )
    return {
        "schema": _BOUND_BUFFER_SCHEMA,
        "schema_version": 1,
        "byte_count": byte_count,
        "mapping_bytes": mapping_bytes,
        "page_size": page_size,
        "page_count": page_count,
        "selected_nodes": list(selected_nodes),
        "policy_mode": "bind_static_nodes",
        "page_aligned_mapping": True,
        "bound_before_first_touch": True,
        "live_owner_policy_verified": True,
        "range_policy_verified": True,
        "post_decode_complete_page_query": True,
        "post_decode_strict_policy_verified": True,
        "queried_pages": page_count,
        "resolved_pages": page_count,
        "query_chunks": chunks,
        "query_chunk_page_limit": chunk_pages,
        "page_migration_requested": False,
        "placement_repair_performed": False,
        "node_histogram": {
            str(node): count for node, count in sorted(histogram.items())
        },
        "ordered_status_sha256": status_sha256,
        "ordered_status_encoding": (
            f"native_{ctypes.sizeof(ctypes.c_int) * 8}bit_signed_{sys.byteorder}"
        ),
        "complete": True,
    }


def _last_long_option(
    argv: Sequence[str], option: str, *, minimum_prefix: str
) -> tuple[bool, str | None]:
    found = False
    value = None
    tokens = list(argv)
    for index, token in enumerate(tokens):
        text = str(token)
        name = text.split("=", 1)[0]
        matches = name == option or (
            name.startswith(minimum_prefix) and option.startswith(name)
        )
        if not matches:
            continue
        if "=" not in text:
            found = True
            if index + 1 < len(tokens) and not str(tokens[index + 1]).startswith("-"):
                value = str(tokens[index + 1])
            else:
                value = None
        else:
            found = True
            value = text.split("=", 1)[1]
    return found, value


def preconfigure_numa_from_argv(argv: Sequence[str]) -> dict[str, object] | None:
    """Apply an explicitly requested CLI membind; otherwise make no change."""
    for token in argv:
        option = str(token).split("=", 1)[0]
        if option in {"--numa", "--numa-"}:
            raise ValueError(
                "Ambiguous NUMA option; use --numa-mode or --numa-nodes."
            )
    mode_present, mode = _last_long_option(
        argv, "--numa-mode", minimum_prefix="--numa-m"
    )
    if not mode_present or mode is None or mode.strip().lower() != "membind":
        return None
    nodes_present, nodes = _last_long_option(
        argv, "--numa-nodes", minimum_prefix="--numa-n"
    )
    if nodes_present and nodes is None:
        return None
    return apply_early_numa_membind(nodes if nodes_present else "all")
