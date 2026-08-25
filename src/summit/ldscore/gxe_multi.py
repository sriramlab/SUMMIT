"""Common-cohort multi-environment GxE reference construction.

The statistical model remains one independent G+GxE+NxE+residual system per
environment.  This module shares each standardized genotype block across those
systems; it never writes feature matrices or randomized sketches to disk.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import resource
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil

from .. import utils
from .._early_numa import current_numa_policy_attestation
from .gw_ldscore import (
    _validate_cpu_placement_attestation,
    _validate_openmp_placement_build_contract,
)
from .gwe_ldscore import (
    GenomewideEnvLDScore,
    _build_balanced_vtiles,
    _make_seed,
    _orthonormalize_columns,
    _validate_native_blas_runtime,
)


_SCORE_NAMES = ("xx", "xw", "wx", "ww")
_PERFORMANCE_TELEMETRY_SCHEMA_VERSION = 1
_MAX_GEMM_TELEMETRY_RECORDS = 65536
_NUMA_BOUND_DECODE_SCHEMA = "summit.numa_bound_bed_decode.v1"
_NUMA_BOUND_BUFFER_SCHEMA = "summit.numa_bound_anonymous_buffer.v1"
_NUMA_PAGE_QUERY_CHUNK_LIMIT = 65536
_NATIVE_GEMM_OUTPUT_NUMA_SCHEMA = "summit.native_gemm_output_numa.v1"
_PACKED_SOURCE_PANEL_NUMA_SCHEMA = "summit.packed_source_panel_numa.v1"
_NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT = 65536
_NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY = 16384
_MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES = 10
_COMPLETE_MEMORY_PLAN_SCHEMA = "summit.gxe.complete_process_memory_plan.v1"


def _process_rss_bytes() -> int:
    return int(psutil.Process().memory_info().rss)


def _frozen_mailman_environment() -> dict:
    """Return the frozen SUMMIT_MAILMAN_* configuration, failing closed.

    The native kernels read these variables with atoi/atoll.  The protected
    production path only admits values whose Python and native parses are
    provably identical, so an unmodeled override can never invalidate the
    memory plan: any set-but-non-canonical value is rejected here before a
    native context is constructed.
    """
    frozen = {}
    for name in (
        "SUMMIT_MAILMAN_SEGMENT_SIZE",
        "SUMMIT_MAILMAN_QPANEL",
        "SUMMIT_MAILMAN_WORK_MB",
    ):
        value = os.environ.get(name)
        if value is None:
            frozen[name] = None
            continue
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(
                f"{name}={value!r} is not a canonical positive integer; the "
                "protected GxE path rejects unmodeled Mailman overrides."
            )
        frozen[name] = int(value)
    return frozen


def _mailman_segment_size(rows: int, frozen_environment: Mapping) -> int:
    """Mirror compute_mailman_segment_size_from_n exactly."""
    override = frozen_environment.get("SUMMIT_MAILMAN_SEGMENT_SIZE")
    if override is not None:
        return min(int(override), 19)
    if rows <= 27:
        return 1
    return min(8, max(1, math.floor(math.log(rows, 3)) - 1))


def _mailman_qpanel_width(
    table_size: int,
    q_total: int,
    segment_size: int,
    frozen_environment: Mapping,
    segment_buffers: int = 1,
) -> int:
    """Mirror summit::mailman::qpanel_width<double> exactly."""
    if q_total <= 0:
        return 1
    override = frozen_environment.get("SUMMIT_MAILMAN_QPANEL")
    if override is not None:
        return max(1, min(int(override), q_total))
    work_mb = frozen_environment.get("SUMMIT_MAILMAN_WORK_MB")
    table_megabytes = 8 if work_mb is None else int(work_mb)
    table_budget = table_megabytes * 1024 * 1024
    table_bytes_per_column = (
        table_size * 8
        + max(1, segment_size) * 8 * max(1, segment_buffers)
        + 8
    )
    width = table_budget // table_bytes_per_column
    width = max(1, min(width, q_total))
    if width >= 64:
        width = (width // 64) * 64
    return max(1, width)


def _mailman_worker_scratch_bytes(
    *,
    rows: int,
    feature_rhs_columns: int,
    wide_columns: int,
    threads: int,
    frozen_environment: Mapping,
) -> dict:
    """Exact per-worker packed Mailman scratch from the frozen plan.

    Mirrors the native context's worker-arena capacity computation: one
    lookup table sized by the widest q-panel across the feature, source, and
    target kernels, one shared segment buffer (feature linear / target raw
    segment), and one squared segment buffer (feature only).
    """
    segment = _mailman_segment_size(rows, frozen_environment)
    table = 3**segment
    target_columns = 2 * wide_columns
    qpanel_feature = _mailman_qpanel_width(
        table, feature_rhs_columns, segment, frozen_environment,
        segment_buffers=2,
    )
    qpanel_source = _mailman_qpanel_width(
        table, wide_columns, segment, frozen_environment
    )
    qpanel_target = _mailman_qpanel_width(
        table, target_columns, segment, frozen_environment
    )
    feature_panel = min(qpanel_feature, feature_rhs_columns)
    source_panel = min(qpanel_source, wide_columns)
    target_panel = min(qpanel_target, target_columns)
    table_capacity = table * max(feature_panel, source_panel, target_panel)
    segment_a_capacity = segment * max(feature_panel, target_panel)
    segment_b_capacity = segment * feature_panel
    per_worker = (table_capacity + segment_a_capacity + segment_b_capacity) * 8
    return {
        "segment_size": segment,
        "table_size": table,
        "qpanel_feature": qpanel_feature,
        "qpanel_source": qpanel_source,
        "qpanel_target": qpanel_target,
        "per_worker_bytes": per_worker,
        "total_bytes": threads * per_worker,
    }
_GIB = 1024**3
_THREAD_STACK_ALLOWANCE_BYTES = 8 * 1024**2
_TELEMETRY_ALLOWANCE_BYTES = 256 * 1024**2
_MINIMUM_ALLOCATOR_SLACK_BYTES = 512 * 1024**2
_TOTAL_MEMORY_HEADROOM_FRACTION = 0.20
_DETERMINISTIC_TN_MAXIMUM_COLUMNS = 64
_DETERMINISTIC_TN_MAXIMUM_FLOPS = 2_000_000_000
_DETERMINISTIC_FEATURE_TN_MAXIMUM_ROWS = 128
_DETERMINISTIC_FEATURE_TN_MAXIMUM_COLUMNS = 2048
_DETERMINISTIC_FEATURE_TN_MAXIMUM_REDUCTION = 16384
_DETERMINISTIC_FEATURE_TN_MAXIMUM_FLOPS = 5_000_000_000


def _json_safe(value):
    """Return bounded telemetry values using only JSON-native scalar types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parsed_numa_node_request(value, effective_nodes) -> list[int] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if value.strip().lower() == "all":
        return effective_nodes if isinstance(effective_nodes, list) else None
    parsed = set()
    try:
        for component in value.split(","):
            bounds = component.strip().split("-", 1)
            if not bounds[0] or len(bounds) > 2:
                raise ValueError
            start = int(bounds[0])
            stop = int(bounds[-1])
            if start < 0 or stop < start:
                raise ValueError
            parsed.update(range(start, stop + 1))
    except (TypeError, ValueError):
        return None
    return sorted(parsed)


def _validated_early_numa_nodes(
    attestation: Mapping,
    *,
    require_static: bool,
    expected_pid: int | None = None,
) -> tuple[int, ...]:
    """Validate the exact early membind evidence used by protected workers."""
    if not isinstance(attestation, Mapping):
        raise RuntimeError("Early NUMA attestation is absent or malformed.")
    effective_nodes = attestation.get("effective_nodes")
    requested_nodes = _parsed_numa_node_request(
        attestation.get("requested_nodes"), effective_nodes
    )
    task_count = attestation.get("task_count_at_application")
    pid = attestation.get("pid")
    valid = (
        attestation.get("schema") == "summit.numa_policy_attestation.v1"
        and attestation.get("mode") == "membind"
        and attestation.get("source") == "libnuma"
        and isinstance(effective_nodes, list)
        and bool(effective_nodes)
        and all(
            not isinstance(node, bool) and isinstance(node, int) and node >= 0
            for node in effective_nodes
        )
        and effective_nodes == sorted(set(effective_nodes))
        and requested_nodes == effective_nodes
        and not isinstance(task_count, bool)
        and isinstance(task_count, int)
        and task_count == 1
        and not isinstance(pid, bool)
        and isinstance(pid, int)
        and pid > 0
        and (expected_pid is None or pid == expected_pid)
        and attestation.get("applied_policy")
        == "libnuma:membind:" + ",".join(map(str, effective_nodes))
        and attestation.get("applied_before_numeric_import") is True
        and attestation.get("verified") is True
        and (
            attestation.get("static_nodes") is True
            if require_static
            else (
                "static_nodes" not in attestation
                or attestation.get("static_nodes") is True
            )
        )
    )
    if not valid:
        raise RuntimeError("Early NUMA attestation is incomplete or malformed.")
    return tuple(effective_nodes)


def _validated_numa_bound_decode_report(
    records,
    *,
    blocks: Sequence[tuple[int, int]],
    passes: int,
    sample_count: int,
    num_variants: int,
    selected_nodes: tuple[int, ...],
) -> dict:
    """Validate and normalize every dedicated BED-decode allocation record."""
    try:
        raw_blocks = tuple(tuple(block) for block in blocks)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NUMA-bound BED decode plan is malformed.") from exc
    if any(
        len(block) != 2 or any(type(endpoint) is not int for endpoint in block)
        for block in raw_blocks
    ):
        raise RuntimeError("NUMA-bound BED decode plan is malformed.")
    canonical_blocks = raw_blocks
    if (
        isinstance(passes, bool)
        or not isinstance(passes, int)
        or passes <= 0
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or isinstance(num_variants, bool)
        or not isinstance(num_variants, int)
        or num_variants <= 0
        or not canonical_blocks
        or canonical_blocks[0][0] != 0
        or canonical_blocks[-1][1] != num_variants
        or any(
            start < 0
            or stop <= start
            or (index and start != canonical_blocks[index - 1][1])
            for index, (start, stop) in enumerate(canonical_blocks)
        )
    ):
        raise RuntimeError("NUMA-bound BED decode plan is malformed.")
    if (
        not selected_nodes
        or any(type(node) is not int or node < 0 for node in selected_nodes)
        or tuple(sorted(set(selected_nodes))) != selected_nodes
    ):
        raise RuntimeError("NUMA-bound BED decode nodes are malformed.")
    if not isinstance(records, list):
        raise RuntimeError("NUMA-bound BED decode records are absent.")
    expected_blocks = list(canonical_blocks) * passes
    if len(records) != len(expected_blocks):
        raise RuntimeError(
            "NUMA-bound BED decode evidence count disagrees with the exact "
            "shared genotype pass plan."
        )

    normalized_records = []
    total_payload_bytes = 0
    total_mapping_bytes = 0
    maximum_mapping_bytes = 0
    decoder_name = None
    for index, (record, expected_block) in enumerate(
        zip(records, expected_blocks, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise RuntimeError("NUMA-bound BED decode record is malformed.")
        start, stop = expected_block
        expected_byte_count = sample_count * (stop - start) * 8
        record_block = record.get("genotype_block")
        if (
            set(record)
            != {
                "genotype_block",
                "memory_order",
                "decoder",
                "allocation",
                "bound_mapping_preserved_after_standardization",
                "verification_stage",
                "verification",
            }
            or not isinstance(record_block, list)
            or len(record_block) != 2
            or any(type(endpoint) is not int for endpoint in record_block)
            or record_block != [start, stop]
            or record.get("memory_order") != "F"
            or record.get("decoder")
            not in {
                "bed_reader.read_f64_into_bound_mapping",
                "gxeldcore.DirectContext.decode_block",
            }
            or record.get("verification_stage")
            != "post_standardization_pre_return"
            or record.get("bound_mapping_preserved_after_standardization")
            is not True
        ):
            raise RuntimeError(
                "NUMA-bound BED decode record disagrees with the protected "
                f"block plan at record {index}."
            )
        if decoder_name is None:
            decoder_name = record["decoder"]
        elif record["decoder"] != decoder_name:
            raise RuntimeError(
                "NUMA-bound BED decode records mix incompatible decoder owners."
            )
        allocation = record.get("allocation")
        verification = record.get("verification")
        if not isinstance(allocation, Mapping) or not isinstance(
            verification, Mapping
        ):
            raise RuntimeError("NUMA-bound BED decode evidence is malformed.")
        page_size = allocation.get("page_size")
        mapping_bytes = allocation.get("mapping_bytes")
        page_count = allocation.get("page_count")
        # Schema v2 buffers reuse one maximum-capacity mapping: the mapping is
        # sized by capacity_byte_count and per-record byte_count is the
        # logical decode inside it.  Schema v1 kept mapping == logical.
        buffer_schema_version = allocation.get("schema_version")
        if buffer_schema_version not in (1, 2):
            raise RuntimeError(
                "NUMA-bound BED decode buffer schema version is unsupported."
            )
        capacity_byte_count = (
            allocation.get("capacity_byte_count")
            if buffer_schema_version >= 2
            else expected_byte_count
        )
        if (
            isinstance(capacity_byte_count, bool)
            or not isinstance(capacity_byte_count, int)
            or capacity_byte_count < expected_byte_count
            or (
                buffer_schema_version >= 2
                and verification.get("capacity_byte_count")
                != capacity_byte_count
            )
        ):
            raise RuntimeError(
                "NUMA-bound BED decode capacity accounting is invalid."
            )
        expected_common = {
            "schema": _NUMA_BOUND_BUFFER_SCHEMA,
            "schema_version": buffer_schema_version,
            "byte_count": expected_byte_count,
            "selected_nodes": list(selected_nodes),
            "policy_mode": "bind_static_nodes",
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "range_policy_verified": True,
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }
        allocation_keys = {
            *expected_common,
            "mapping_bytes",
            "page_size",
            "page_count",
            "post_decode_complete_page_query",
            *(
                {"capacity_byte_count"}
                if buffer_schema_version >= 2
                else set()
            ),
        }
        verification_keys = {
            *allocation_keys,
            "post_decode_strict_policy_verified",
            "queried_pages",
            "resolved_pages",
            "query_chunks",
            "query_chunk_page_limit",
            "node_histogram",
            "ordered_status_encoding",
            "complete",
        }
        allocation_nodes = allocation.get("selected_nodes")
        verification_nodes = verification.get("selected_nodes")
        if (
            set(allocation) != allocation_keys
            or set(verification) - {"ordered_status_sha256"} != verification_keys
            or type(allocation.get("schema_version")) is not int
            or type(verification.get("schema_version")) is not int
            or type(allocation.get("byte_count")) is not int
            or type(verification.get("byte_count")) is not int
            or not isinstance(allocation_nodes, list)
            or not isinstance(verification_nodes, list)
            or any(type(node) is not int for node in allocation_nodes)
            or any(type(node) is not int for node in verification_nodes)
            or isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or page_size <= 0
            or isinstance(mapping_bytes, bool)
            or not isinstance(mapping_bytes, int)
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count <= 0
            or mapping_bytes != page_count * page_size
            or mapping_bytes < capacity_byte_count
            or mapping_bytes - capacity_byte_count >= page_size
            or any(allocation.get(key) != value for key, value in expected_common.items())
            or allocation.get("post_decode_complete_page_query") is not False
            or any(
                verification.get(key) != value
                for key, value in (
                    *expected_common.items(),
                    ("mapping_bytes", mapping_bytes),
                    ("page_size", page_size),
                    ("page_count", page_count),
                )
            )
            or verification.get("post_decode_complete_page_query") is not True
            or verification.get("post_decode_strict_policy_verified") is not True
            or verification.get("page_migration_requested") is not False
            or verification.get("placement_repair_performed") is not False
            or verification.get("queried_pages") != page_count
            or verification.get("resolved_pages") != page_count
            or verification.get("complete") is not True
        ):
            raise RuntimeError(
                "NUMA-bound BED decode allocation or complete-page evidence "
                f"is inconsistent at record {index}."
            )
        chunk_limit = verification.get("query_chunk_page_limit")
        query_chunks = verification.get("query_chunks")
        if (
            isinstance(chunk_limit, bool)
            or not isinstance(chunk_limit, int)
            or not 1 <= chunk_limit <= _NUMA_PAGE_QUERY_CHUNK_LIMIT
            or isinstance(query_chunks, bool)
            or not isinstance(query_chunks, int)
            or query_chunks != (page_count + chunk_limit - 1) // chunk_limit
            or verification.get("ordered_status_encoding")
            != f"native_32bit_signed_{sys.byteorder}"
        ):
            raise RuntimeError(
                f"NUMA complete-page query metadata is malformed at record {index}."
            )
        histogram = verification.get("node_histogram")
        if not isinstance(histogram, Mapping) or not histogram:
            raise RuntimeError("NUMA page-node histogram is absent.")
        normalized_histogram = {}
        for node_text, count in histogram.items():
            try:
                node = int(node_text)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("NUMA page-node histogram is malformed.") from exc
            if (
                str(node) != node_text
                or node not in selected_nodes
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise RuntimeError("NUMA page-node histogram is malformed.")
            normalized_histogram[node] = count
        if sum(normalized_histogram.values()) != page_count:
            raise RuntimeError("NUMA page-node histogram is incomplete.")
        normalized_record = _json_safe(dict(record))
        normalized_record["verification"].pop("ordered_status_sha256", None)
        normalized_records.append(normalized_record)
        total_payload_bytes += expected_byte_count
        total_mapping_bytes += mapping_bytes
        maximum_mapping_bytes = max(maximum_mapping_bytes, mapping_bytes)

    max_payload_bytes = max(
        sample_count * (stop - start) * 8 for start, stop in canonical_blocks
    )
    return {
        "schema": _NUMA_BOUND_DECODE_SCHEMA,
        "schema_version": 1,
        "required": True,
        "bounded": True,
        "decoder": decoder_name,
        "memory_order": "F",
        "verification_stage": "post_standardization_pre_return",
        "selected_nodes": list(selected_nodes),
        "sample_count": sample_count,
        "num_variants": num_variants,
        "float64_itemsize": 8,
        "genotype_blocks": [list(block) for block in canonical_blocks],
        "shared_genotype_passes": passes,
        "expected_block_read_count": len(expected_blocks),
        "observed_block_read_count": len(normalized_records),
        "max_payload_bytes_per_block": max_payload_bytes,
        "max_mapping_bytes_per_block": maximum_mapping_bytes,
        "total_payload_bytes_across_reads": total_payload_bytes,
        "total_mapping_bytes_across_reads": total_mapping_bytes,
        "complete_page_query_records": len(normalized_records),
        "records_included": True,
        "records": normalized_records,
        "complete": True,
    }


def _decode_report_for_output(report: Mapping, *, include_records: bool) -> dict:
    result = _json_safe(dict(report))
    if not include_records:
        result.pop("records", None)
        result["records_included"] = False
    return result


def _validate_persisted_numa_bound_decode_report(
    report: Mapping,
    *,
    expected_nodes: tuple[int, ...],
    expected_sample_count: int,
    expected_num_variants: int,
    expected_passes: int,
) -> dict:
    """Reconstruct contracted decode telemetry and reject edited summaries."""
    if not isinstance(report, Mapping) or report.get("records_included") is not True:
        raise RuntimeError("Complete NUMA-bound BED decode records are absent.")
    integer_fields = (
        "schema_version",
        "sample_count",
        "num_variants",
        "float64_itemsize",
        "shared_genotype_passes",
        "expected_block_read_count",
        "observed_block_read_count",
        "max_payload_bytes_per_block",
        "max_mapping_bytes_per_block",
        "total_payload_bytes_across_reads",
        "total_mapping_bytes_across_reads",
        "complete_page_query_records",
    )
    boolean_fields = ("required", "bounded", "records_included", "complete")
    report_nodes = report.get("selected_nodes")
    report_blocks = report.get("genotype_blocks")
    if (
        any(type(report.get(field)) is not int for field in integer_fields)
        or any(type(report.get(field)) is not bool for field in boolean_fields)
        or not isinstance(report_nodes, list)
        or any(type(node) is not int for node in report_nodes)
        or tuple(report_nodes) != expected_nodes
        or not isinstance(report_blocks, list)
        or any(
            not isinstance(block, list)
            or len(block) != 2
            or any(type(endpoint) is not int for endpoint in block)
            for block in report_blocks
        )
        or not isinstance(report.get("records"), list)
    ):
        raise RuntimeError("Persisted NUMA-bound BED decode types are malformed.")
    blocks = report.get("genotype_blocks")
    try:
        canonical_blocks = tuple(tuple(block) for block in blocks)
    except TypeError as exc:
        raise RuntimeError("Persisted NUMA-bound BED block plan is malformed.") from exc
    rebuilt = _validated_numa_bound_decode_report(
        report.get("records"),
        blocks=canonical_blocks,
        passes=expected_passes,
        sample_count=expected_sample_count,
        num_variants=expected_num_variants,
        selected_nodes=expected_nodes,
    )
    if dict(report) != rebuilt:
        raise RuntimeError("Persisted NUMA-bound BED decode summary was modified.")
    return rebuilt


def _validate_packed_source_panel_numa_records(
    records: object,
    *,
    expected_nodes: tuple[int, ...] | None,
    expected_count: int,
    expected_combined_pair: bool = False,
) -> list[dict]:
    """Validate persistent source/weighted mappings used by direct execution."""
    if not isinstance(records, list) or len(records) != expected_count:
        raise RuntimeError(
            "Packed source-panel NUMA evidence count disagrees with the tile plan."
        )
    validated: list[dict] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("Packed source-panel NUMA evidence is not a mapping.")
        integer_fields = (
            "schema_version",
            "logical_rows",
            "logical_columns",
            "logical_byte_count",
            "environment_start",
            "environment_stop",
            "probe_start",
            "probe_count",
        )
        operand_role = record.get("operand_role")
        verification_boundary = record.get("verification_boundary")
        source_role = operand_role == "persistent_packed_source_panel"
        combined_role = (
            expected_combined_pair
            and operand_role
            == "persistent_source_and_environment_weighted_target_pair"
        )
        if (
            record.get("schema") != _PACKED_SOURCE_PANEL_NUMA_SCHEMA
            or any(type(record.get(field)) is not int for field in integer_fields)
            or record.get("schema_version") != 1
            or record.get("applicable") is not True
            or not (source_role or combined_role)
            or record.get("storage_layout") != "column_major"
            or verification_boundary
            != (
                "after_projection_before_target_scoring"
                if source_role
                else "after_environment_weighting_before_target_scoring"
            )
            or record.get("sealed_read_only") is not True
            or record["logical_rows"] <= 0
            or record["logical_columns"] <= 0
            or record["logical_byte_count"]
            != record["logical_rows"] * record["logical_columns"] * 8
            or record["environment_start"] < 0
            or record["environment_stop"] <= record["environment_start"]
            or record["probe_start"] < 0
            or record["probe_count"] <= 0
        ):
            raise RuntimeError("Packed source-panel NUMA evidence is malformed.")
        contracted = expected_nodes is not None
        if record.get("contract_required") is not contracted:
            raise RuntimeError(
                "Packed source-panel NUMA evidence contract state is incorrect."
            )
        if not contracted:
            if (
                record.get("complete") is not False
                or record.get("allocation_mode") != "legacy_anonymous_mapping"
            ):
                raise RuntimeError(
                    "Uncontracted packed source-panel allocation evidence is malformed."
                )
            validated.append(_json_safe(dict(record)))
            continue
        positive_integer_fields = (
            "mapping_bytes",
            "page_size",
            "mapping_page_count",
            "queried_pages",
            "resolved_pages",
            "query_chunks",
            "query_chunk_page_limit",
        )
        histogram = record.get("node_histogram")
        if (
            record.get("complete") is not True
            or any(
                type(record.get(field)) is not int or record[field] <= 0
                for field in positive_integer_fields
            )
            or record.get("selected_nodes") != list(expected_nodes)
            or record.get("policy_mode") != "bind_static_nodes"
            or record.get("allocation_mode") != "mmap_private_anonymous"
            or record.get("anonymous_private_mapping") is not True
            or record.get("page_aligned_mapping") is not True
            or record.get("writable_output") is not False
            or record.get("bound_before_first_touch") is not True
            or record.get("pre_touch_live_owner_policy_verified") is not True
            or record.get("pre_touch_range_policy_verified") is not True
            or record.get("post_repair_live_owner_policy_verified") is not True
            or record.get("post_repair_range_policy_verified") is not True
            or record.get("post_repair_complete_page_query") is not True
            or record.get("post_repair_strict_policy_verified") is not True
            or record.get("strict_policy_check")
            != "MPOL_MF_STRICT_without_MPOL_MF_MOVE"
            or record.get("page_query_method")
            != "move_pages_query_no_migration"
            or record.get("page_migration_requested") is not False
            or record.get("placement_repair_performed") is not False
            or record["mapping_bytes"]
            != record["mapping_page_count"] * record["page_size"]
            or record["mapping_bytes"] < record["logical_byte_count"]
            or record["mapping_bytes"] - record["logical_byte_count"]
            >= record["page_size"]
            or record["queried_pages"] != record["mapping_page_count"]
            or record["resolved_pages"] != record["mapping_page_count"]
            or record["query_chunks"]
            != math.ceil(
                record["mapping_page_count"] / record["query_chunk_page_limit"]
            )
            or not isinstance(histogram, Mapping)
            or any(
                type(key) is not str
                or not key.isdigit()
                or int(key) not in expected_nodes
                or type(value) is not int
                or value <= 0
                for key, value in histogram.items()
            )
            or sum(histogram.values()) != record["mapping_page_count"]
        ):
            raise RuntimeError(
                "Contracted packed source-panel NUMA evidence is incomplete."
            )
        normalized = _json_safe(dict(record))
        normalized.pop("ordered_status_sha256", None)
        validated.append(normalized)
    tile_roles: dict[tuple[int, int, int, int], list[str]] = {}
    for record in validated:
        tile_key = (
            record["environment_start"],
            record["environment_stop"],
            record["probe_start"],
            record["probe_count"],
        )
        tile_roles.setdefault(tile_key, []).append(record["operand_role"])
    expected_roles = {
        "persistent_source_and_environment_weighted_target_pair"
        if expected_combined_pair
        else "persistent_packed_source_panel"
    }
    if any(
        len(roles) != len(expected_roles) or set(roles) != expected_roles
        for roles in tile_roles.values()
    ):
        raise RuntimeError(
            "Persistent target-panel NUMA evidence does not contain the exact "
            "role once per tile."
        )
    return validated


def _recheck_openmp_cpu_placement(
    native_module, early_placement: Mapping, threads: int
) -> dict:
    """Perform and compare the executor-boundary no-BLAS placement probe."""
    early = _validate_cpu_placement_attestation(
        early_placement, expected_threads=threads
    )
    configure = getattr(native_module, "configure_openmp_placement", None)
    if not callable(configure):
        raise RuntimeError(
            "The protected GxE extension lacks the OpenMP placement contract API."
        )
    observed = _validate_cpu_placement_attestation(
        dict(configure(list(early["expected_cpu_ids"]), threads)),
        expected_cpu_ids=early["expected_cpu_ids"],
        expected_threads=threads,
    )
    if observed != early:
        raise RuntimeError(
            "Executor OpenMP placement recheck disagrees with the early worker probe."
        )
    return observed


def _aggregate_estimator_phase_timings(
    estimators: Sequence[GenomewideEnvLDScore],
) -> dict[str, dict[str, float | int]]:
    """Sum non-overlapping estimator-local phase counters across environments."""
    combined: dict[str, dict[str, float | int]] = {}
    for estimator in estimators:
        for phase, observed in getattr(
            estimator, "performance_phase_timings", {}
        ).items():
            target = combined.setdefault(
                str(phase),
                {"wall_seconds": 0.0, "process_cpu_seconds": 0.0, "calls": 0},
            )
            for key in ("wall_seconds", "process_cpu_seconds"):
                target[key] = float(target[key]) + float(observed.get(key, 0.0))
            target["calls"] = int(target["calls"]) + int(
                observed.get("calls", 0)
            )
    return combined


def _log_phase_progress(log, phase: str, completed: int, total: int, started: float) -> None:
    interval = max(1, total // 10)
    if completed != total and completed % interval:
        return
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = (total - completed) / rate if rate > 0.0 else math.inf
    log._log(
        f"[gxe:multi:{phase}] {completed}/{total} blocks; "
        f"elapsed={elapsed:.1f}s; estimated_remaining={remaining:.1f}s."
    )


def safe_environment_suffix(name: str) -> str:
    """Return a stable filename component for an environment column name."""
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    if not suffix:
        raise ValueError(f"Environment name {name!r} has no safe filename characters.")
    return suffix


class _MultiEnvironmentGemm:
    """Matrix executor for the shared stream, with exact native provenance."""

    def __init__(
        self,
        requested_backend: str,
        estimator: GenomewideEnvLDScore,
        full_precision_layout: str = "current",
    ) -> None:
        backend = str(requested_backend).strip().lower()
        if backend not in {"python", "direct"}:
            raise ValueError("requested_backend must be 'python' or 'direct'.")
        self.requested_backend = backend
        self.protected = backend == "direct"
        self.threads = int(estimator.num_threads)
        self.storage_dtype = np.dtype(estimator.dtype)
        layout = str(full_precision_layout).strip().lower().replace("-", "_")
        if layout != "current":
            raise ValueError(
                "The only supported production full_precision_layout is "
                "'current'; source-TT and row-major target layouts are "
                "diagnostic-only native primitives."
            )
        self.full_precision_layout = "current"
        self.compute_dtype = np.dtype(
            np.float64 if self.protected else self.storage_dtype
        )
        # Shared genotype operands remain float64 even when --dtype requests
        # float32 storage, so NumPy promotes the bulk products to float64 too.
        self.arithmetic_dtype = np.dtype(np.float64)
        self.nn_calls = 0
        self.tn_calls = 0
        self._gemm_shapes: dict[tuple[str, int, int, int], int] = {}
        self._semantic_context_stack: list[dict] = []
        self._phase_totals: dict[str, dict[str, float | int]] = {}
        self._gemm_records: list[dict] = []
        self._dropped_gemm_records = 0
        self._native_telemetry_errors: list[str] = []
        self._native_telemetry_consumer = None
        self._native_telemetry_status_getter = None
        self._native_telemetry_available = False
        self._native_gemm_output_numa_consumer = None
        self._native_gemm_output_numa_status_getter = None
        self._native_gemm_output_numa_available = False
        self._native_gemm_output_numa_contract_supported = False
        self._native_gemm_output_numa_contract_required = False
        self._native_gemm_output_numa_records: list[dict] = []
        self._native_gemm_output_numa_seen_call_ids: set[int] = set()
        self._dropped_native_gemm_output_numa_records = 0
        self._hot_gemm_logical_calls = 0
        self._hot_gemm_vendor_observed_calls = 0
        self._hot_gemm_deterministic_tiled_calls = 0
        self._hot_gemm_unobserved_calls = 0
        self._hot_gemm_phase_counts = {
            phase: {
                "logical_calls": 0,
                "vendor_observed_calls": 0,
                "deterministic_tiled_calls": 0,
                "unobserved_calls": 0,
            }
            for phase in ("source_gemm", "target_gemm")
        }
        self._affinity_cores = (
            sorted(int(core) for core in os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else []
        )
        self._cpu_numa_nodes = {}
        for core in self._affinity_cores:
            nodes = list(
                Path(f"/sys/devices/system/cpu/cpu{core}").glob("node[0-9]*")
            )
            if nodes:
                self._cpu_numa_nodes[core] = int(nodes[0].name[4:])
        early_numa_attestation = current_numa_policy_attestation()
        self._numa_process_evidence = {
            "applied_policy": (
                early_numa_attestation.get("applied_policy")
                if isinstance(early_numa_attestation, Mapping)
                else os.environ.get("SUMMIT_NUMA_POLICY_APPLIED")
            ),
            "policy_provenance": (
                "pre_numeric_import"
                if isinstance(early_numa_attestation, Mapping)
                else os.environ.get("SUMMIT_NUMA_POLICY_PROVENANCE")
            ),
            "early_numa_attestation": early_numa_attestation,
            "mems_allowed_list": None,
        }
        try:
            for line in Path("/proc/self/status").read_text(
                encoding="utf-8"
            ).splitlines():
                if line.startswith("Mems_allowed_list:"):
                    self._numa_process_evidence["mems_allowed_list"] = (
                        line.split(":", 1)[1].strip()
                    )
                    break
        except OSError:
            pass
        self.backend_name = "numpy_matmul"
        self.blas_vendor = "none"
        self.vendor_probe_chunk_width: int | None = None
        self.backend_version = str(np.__version__)
        self.repaired_output_columns = 0
        self.integrity_enabled = False
        self.checksum_enabled = False
        self.integrity_minimum_vendor_flops = 0
        self.runtime_isolation = "python"
        self.cpu_placement: dict | None = None
        self.cpu_placement_complete = False
        self._numa_bound_bed_decode_required = False
        self._numa_bound_bed_decode_report: dict | None = None
        self._packed_source_panel_numa_records: list[dict] = []
        self._packed_source_panel_numa_complete = False
        early_placement = getattr(estimator, "cpu_placement", None)
        placement_complete = getattr(estimator, "cpu_placement_complete", False)
        placement_authenticated = getattr(
            estimator, "_gxe_group_worker_authenticated", False
        )
        if early_placement is None:
            if placement_complete is not False or placement_authenticated is not False:
                raise RuntimeError(
                    "Authenticated shared GxE execution lacks CPU placement evidence."
                )
        else:
            if placement_complete is not True or placement_authenticated is not True:
                raise RuntimeError(
                    "Shared GxE CPU placement evidence is not authenticated and complete."
                )
            self.cpu_placement = _validate_cpu_placement_attestation(
                early_placement, expected_threads=self.threads
            )
            self.cpu_placement_complete = True
            self._native_gemm_output_numa_contract_required = True
            self._numa_bound_bed_decode_required = True
            if not self.protected:
                raise RuntimeError(
                    "Authenticated CPU placement requires the direct protected backend."
                )
        self._module = None
        self._multi_environment_kernel = None
        self._multi_environment_kernel_info: dict | None = None
        self._multi_environment_direct_context = None
        self._multi_environment_direct_context_info: dict | None = None
        self._multi_environment_direct_result: dict | None = None
        self._multi_environment_direct_context_required = False
        self._multi_environment_kernel_call_counts = {
            "feature_calls": 0,
            "packed_feature_calls": 0,
            "source_calls": 0,
            "packed_source_calls": 0,
            "projection_calls": 0,
            "target_calls": 0,
            "packed_target_calls": 0,
            "normalization_calls": 0,
        }
        self._multi_environment_kernel_repaired_columns = 0
        self._multi_environment_kernel_checksum_recomputed_columns = 0
        self._multi_environment_kernel_roundoff_only_columns = 0
        if not self.protected:
            return

        try:
            from .. import gxeldcore
        except Exception as exc:
            raise RuntimeError(
                "The requested protected multi-environment GxE extension is unavailable."
            ) from exc
        for function in (
            "MultiEnvironmentKernel",
            "MultiEnvironmentDirectContext",
            "numpy_philox_rademacher_block",
            "protected_matmul_nn",
            "protected_matmul_tn",
            "protected_rank_update_nn",
            "standardize_genotype_block",
            "fused_feature_scalar_moments",
            "prepare_protected_row_weighted_pair",
            "protected_matmul_tn_pair",
        ):
            if not callable(getattr(gxeldcore, function, None)):
                raise RuntimeError(
                    "The loaded GxE extension predates protected shared GEMMs; "
                    "rebuild/install SUMMIT from the current source before running "
                    "--gxe-env-cols with --gxe-native-backend direct."
                )
        try:
            if self.cpu_placement is not None:
                self.cpu_placement = _recheck_openmp_cpu_placement(
                    gxeldcore, self.cpu_placement, self.threads
                )
            configure_blas_threads = getattr(
                gxeldcore, "configure_blas_threads", None
            )
            if callable(configure_blas_threads):
                configured_threads = int(configure_blas_threads(self.threads))
                if configured_threads != self.threads:
                    raise RuntimeError(
                        "The shared GxE executor configured an unexpected BLAS "
                        "thread count."
                    )
            build_info = dict(gxeldcore.build_info())
            if (
                build_info.get("multi_environment_native_kernel_supported")
                is not True
                or build_info.get("multi_environment_native_kernel_schema")
                != "summit.multi_environment_native_kernel.v1"
                or build_info.get("multi_environment_native_kernel_execution")
                != "feature_source_projection_target_reduction_normalization"
                or build_info.get("multi_environment_direct_context_supported")
                is not True
                or build_info.get("multi_environment_direct_context_schema")
                != "summit.multi_environment_direct_context.v3"
                or build_info.get("multi_environment_direct_context_execution")
                != "descriptor_adaptive_dense_blas_or_packed_mailman_feature_source_target_projection_reduction_normalization"
            ):
                raise RuntimeError(
                    "The protected GxE extension lacks the exact end-to-end "
                    "multi-environment native-kernel contract."
                )
            raw_vendor_probe_chunk = build_info.get(
                "multi_environment_direct_context_max_vendor_probe_chunk"
            )
            expected_vendor_probe_chunk = 0
            if (
                type(raw_vendor_probe_chunk) is not int
                or raw_vendor_probe_chunk != expected_vendor_probe_chunk
            ):
                raise RuntimeError(
                    "The protected GxE extension exposes an unexpected native "
                    "vendor probe-width boundary."
                )
            self.vendor_probe_chunk_width = (
                raw_vendor_probe_chunk if raw_vendor_probe_chunk > 0 else None
            )
            if (
                type(
                    build_info.get(
                        "multi_environment_direct_context_mailman_maximum_probes"
                    )
                )
                is not int
                or build_info[
                    "multi_environment_direct_context_mailman_maximum_probes"
                ]
                != _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
            ):
                raise RuntimeError(
                    "The protected GxE extension exposes an unexpected Mailman "
                    "probe-count boundary."
                )
            if self.cpu_placement is not None:
                _validate_openmp_placement_build_contract(
                    build_info, self.cpu_placement
                )
            self.integrity_enabled = bool(
                build_info.get("gemm_integrity_enabled", False)
            )
            if type(build_info.get("gemm_checksum_enabled")) is not bool:
                raise RuntimeError(
                    "The protected GxE extension lacks the exact checksum mode."
                )
            self.checksum_enabled = build_info["gemm_checksum_enabled"]
            self.integrity_minimum_vendor_flops = int(
                build_info.get(
                    "gemm_integrity_minimum_vendor_flops",
                    1_000_000_000 if self.integrity_enabled else 0,
                )
            )
            self.runtime_isolation = str(
                build_info.get("blas_runtime_isolation", "unknown")
            )
            runtime_record = _validate_native_blas_runtime(build_info)
            output_contract_keys = (
                "native_gemm_output_numa_contract_supported",
                "native_gemm_output_numa_contract_schema",
                "native_gemm_output_numa_query_chunk_page_limit",
                "native_gemm_output_numa_evidence_capacity",
            )
            output_contract_present = [
                key in build_info for key in output_contract_keys
            ]
            if any(output_contract_present) and not all(output_contract_present):
                raise RuntimeError(
                    "The protected shared GxE extension exposes an incomplete "
                    "native GEMM output NUMA contract."
                )
            if all(output_contract_present):
                supported = build_info[
                    "native_gemm_output_numa_contract_supported"
                ]
                if type(supported) is not bool:
                    raise RuntimeError(
                        "The native GEMM output NUMA support flag is malformed."
                    )
                if supported and (
                    build_info["native_gemm_output_numa_contract_schema"]
                    != _NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
                    or type(build_info[
                        "native_gemm_output_numa_query_chunk_page_limit"
                    ]) is not int
                    or build_info[
                        "native_gemm_output_numa_query_chunk_page_limit"
                    ] != _NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
                    or type(build_info[
                        "native_gemm_output_numa_evidence_capacity"
                    ]) is not int
                    or build_info[
                        "native_gemm_output_numa_evidence_capacity"
                    ] != _NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
                ):
                    raise RuntimeError(
                        "The protected shared GxE extension exposes a malformed "
                        "native GEMM output NUMA contract."
                    )
                if self.cpu_placement is not None and supported is not True:
                    raise RuntimeError(
                        "Authenticated shared GxE placement requires the native "
                        "GEMM output NUMA contract."
                    )
                self._native_gemm_output_numa_contract_supported = supported
            elif self.cpu_placement is not None:
                raise RuntimeError(
                    "Authenticated shared GxE placement lacks the native GEMM "
                    "output NUMA contract."
                )
        except Exception:
            raise
        self._module = gxeldcore
        self.backend_name = str(build_info.get("backend_name", "gxeldcore_direct"))
        self.blas_vendor = str(build_info.get("blas_vendor", "unknown"))
        self.backend_version = str(build_info.get("backend_version", "unknown"))
        self._initialize_native_telemetry()
        self._initialize_native_gemm_output_numa_evidence()

    def __enter__(self) -> "_MultiEnvironmentGemm":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        # Exception cleanup must not leak native execution mappings: release
        # any completed context's scratch before dropping the reference.  The
        # native release rejects an incomplete context and a cleanup failure
        # never masks the original error.
        context = getattr(self, "_multi_environment_direct_context", None)
        if context is not None:
            try:
                context.release_execution_scratch()
            except Exception:
                pass
        self._multi_environment_direct_result = None
        self._multi_environment_direct_context = None
        self._multi_environment_direct_context_info = None
        self._multi_environment_kernel = None
        self._multi_environment_kernel_info = None

    def _initialize_native_telemetry(self) -> None:
        """Enable native vendor-call telemetry (introduced in API 7)."""
        assert self._module is not None
        consume = getattr(self._module, "consume_gemm_telemetry", None)
        status = getattr(self._module, "gemm_telemetry_status", None)
        if not callable(consume):
            return
        try:
            reset = getattr(self._module, "reset_gemm_telemetry", None)
            if callable(reset):
                reset()
            self._native_telemetry_consumer = consume
            self._native_telemetry_status_getter = (
                status if callable(status) else None
            )
            self._native_telemetry_available = True
        except Exception as exc:
            self._native_telemetry_errors.append(
                f"native_telemetry_initialization_failed: {type(exc).__name__}: {exc}"
            )

    def _initialize_native_gemm_output_numa_evidence(self) -> None:
        """Enable the one-record-per-protected-output NUMA evidence drain."""
        if not self._native_gemm_output_numa_contract_supported:
            return
        assert self._module is not None
        consume = getattr(
            self._module, "consume_native_gemm_output_numa_evidence", None
        )
        status = getattr(
            self._module, "native_gemm_output_numa_evidence_status", None
        )
        if not callable(consume) or not callable(status):
            self._native_telemetry_errors.append(
                "native_gemm_output_numa_initialization_failed: "
                "contracted evidence APIs are unavailable"
            )
            return
        self._native_gemm_output_numa_consumer = consume
        self._native_gemm_output_numa_status_getter = status
        self._native_gemm_output_numa_available = True

    def _semantic_context(self, semantic: Mapping | None = None) -> dict:
        context: dict = {}
        for record in self._semantic_context_stack:
            context.update(record)
        if semantic is not None:
            context.update(_json_safe(dict(semantic)))
        return context

    @contextmanager
    def semantic_context(self, semantic: Mapping | None = None):
        context = {} if semantic is None else _json_safe(dict(semantic))
        self._semantic_context_stack.append(context)
        try:
            yield
        finally:
            self._semantic_context_stack.pop()

    @contextmanager
    def phase(self, name: str, semantic: Mapping | None = None):
        """Aggregate wall/process-CPU time while exposing context to nested GEMMs."""
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        completed = False
        with self.semantic_context(
            {"phase": str(name)} | ({} if semantic is None else dict(semantic))
        ):
            try:
                yield
                completed = True
            finally:
                self.record_phase_elapsed(
                    name, wall_started, cpu_started, completed=completed
                )

    def record_phase_elapsed(
        self, name: str, wall_started: float, cpu_started: float, *, completed=True
    ) -> None:
        total = self._phase_totals.setdefault(
            str(name),
            {
                "calls": 0,
                "completed_calls": 0,
                "wall_seconds": 0.0,
                "process_cpu_seconds": 0.0,
            },
        )
        total["calls"] = int(total["calls"]) + 1
        total["completed_calls"] = int(total["completed_calls"]) + int(completed)
        total["wall_seconds"] = float(total["wall_seconds"]) + (
            time.perf_counter() - wall_started
        )
        total["process_cpu_seconds"] = float(total["process_cpu_seconds"]) + (
            time.process_time() - cpu_started
        )

    def _consume_native_gemm_output_numa(
        self, *, output_expected: bool
    ) -> dict | None:
        """Drain and retain the exact evidence for one protected output call."""
        if not (
            self.protected
            and self._native_gemm_output_numa_contract_supported
        ):
            return None
        if self._native_gemm_output_numa_consumer is None:
            raise RuntimeError(
                "The contracted native GEMM output NUMA evidence drain is unavailable."
            )
        try:
            drained = list(self._native_gemm_output_numa_consumer())
        except Exception as exc:
            self._native_telemetry_errors.append(
                "native_gemm_output_numa_read_failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._native_gemm_output_numa_consumer = None
            self._native_gemm_output_numa_available = False
            raise RuntimeError(
                "The contracted native GEMM output NUMA evidence drain failed."
            ) from exc
        expected_count = 1 if output_expected else 0
        if len(drained) != expected_count:
            raise RuntimeError(
                "The contracted native GEMM output NUMA evidence drain returned "
                f"{len(drained)} records for a call requiring {expected_count}."
            )
        if not output_expected:
            return None
        raw = drained[0]
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "The contracted native GEMM output NUMA evidence is not a mapping."
            )
        call_id = raw.get("call_id")
        # schema_version 2 adds capacity_byte_count: reusable outputs hold one
        # maximum-capacity mapping, so the mapping may exceed the per-call
        # logical shape by up to that capacity.
        if (
            raw.get("schema") != _NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
            or type(raw.get("schema_version")) is not int
            or raw.get("schema_version") not in (1, 2)
            or raw.get("applicable") is not True
            or type(raw.get("applicable")) is not bool
            or type(raw.get("contract_required")) is not bool
            or type(raw.get("complete")) is not bool
            or raw.get("operand_role") != "protected_gemm_output"
            or type(call_id) is not int
            or call_id <= 0
            or call_id in self._native_gemm_output_numa_seen_call_ids
        ):
            raise RuntimeError(
                "The contracted native GEMM output NUMA evidence discriminator "
                "or call ID is not exact."
            )
        acceptance_contract_required = (
            self._native_gemm_output_numa_contract_required
        )
        if raw.get("contract_required") is True:
            if (
                raw.get("complete") is not True
            ):
                raise RuntimeError(
                    "The native GEMM output NUMA evidence is contracted but incomplete."
                )
        else:
            if acceptance_contract_required:
                raise RuntimeError(
                    "Authenticated placement requires contracted and complete "
                    "native GEMM output NUMA evidence."
                )
            uncontracted_keys = {
                "schema", "schema_version", "applicable", "operand_role",
                "contract_required", "complete", "call_id", "logical_rows",
                "logical_columns", "storage_layout", "logical_byte_count",
                "allocation_mode",
            }
            if raw.get("schema_version") == 2:
                uncontracted_keys.add("capacity_byte_count")
                capacity = raw.get("capacity_byte_count")
                if (
                    isinstance(capacity, bool)
                    or not isinstance(capacity, int)
                    or capacity < raw.get("logical_byte_count", 0)
                ):
                    raise RuntimeError(
                        "The native GEMM output capacity accounting is not exact."
                    )
            if (
                set(raw) != uncontracted_keys
                or raw.get("contract_required") is not False
                or raw.get("complete") is not False
                or raw.get("allocation_mode") != "legacy_posix_memalign"
                or type(raw.get("logical_rows")) is not int
                or raw.get("logical_rows") <= 0
                or type(raw.get("logical_columns")) is not int
                or raw.get("logical_columns") <= 0
                or type(raw.get("logical_byte_count")) is not int
                or raw.get("logical_byte_count")
                != raw.get("logical_rows") * raw.get("logical_columns") * 8
                or raw.get("storage_layout")
                not in {"column_major", "row_major"}
            ):
                raise RuntimeError(
                    "The uncontracted native GEMM output allocation evidence is malformed."
                )
        evidence = _json_safe(dict(raw))
        if (
            len(self._native_gemm_output_numa_records)
            >= _NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        ):
            self._dropped_native_gemm_output_numa_records += 1
            raise RuntimeError(
                "The Python native GEMM output NUMA evidence buffer overflowed."
            )
        self._native_gemm_output_numa_seen_call_ids.add(call_id)
        self._native_gemm_output_numa_records.append(evidence)
        return evidence

    def _record_gemm(
        self,
        fallback: Mapping,
        semantic: Mapping | None = None,
        *,
        protected_output_expected: bool = False,
    ) -> None:
        """Drain exact vendor records, or retain a call-boundary fallback."""
        records = []
        if self._native_telemetry_consumer is not None:
            try:
                records = [
                    dict(record) for record in self._native_telemetry_consumer()
                ]
            except Exception as exc:
                self._native_telemetry_errors.append(
                    f"native_telemetry_read_failed: {type(exc).__name__}: {exc}"
                )
                self._native_telemetry_consumer = None
                self._native_telemetry_available = False
        vendor_records = bool(records)
        output_evidence = self._consume_native_gemm_output_numa(
            output_expected=protected_output_expected
        )
        if vendor_records and output_evidence is not None:
            for observed in records:
                nested = observed.get("native_gemm_output_numa")
                if not isinstance(nested, Mapping) or _json_safe(
                    dict(nested)
                ) != output_evidence:
                    raise RuntimeError(
                        "Native vendor telemetry does not match the drained GEMM "
                        "output NUMA evidence for its protected call ID."
                    )
        if not records:
            fallback_record = dict(fallback)
            fallback_flops = int(
                fallback_record.get(
                    "flop_count",
                    2
                    * int(fallback_record["m"])
                    * int(fallback_record["n"])
                    * int(fallback_record["k"]),
                )
            )
            deterministic_tiled = bool(
                self.protected
                and (
                    fallback_record.get("operation") == "nn_update"
                    or (
                        self.integrity_enabled
                        and fallback_flops < self.integrity_minimum_vendor_flops
                    )
                    or (
                        self.integrity_enabled
                        and fallback_record.get("transpose_a") == "T"
                        and fallback_record.get("transpose_b") == "N"
                        and int(fallback_record["n"])
                        <= _DETERMINISTIC_TN_MAXIMUM_COLUMNS
                        and fallback_flops <= _DETERMINISTIC_TN_MAXIMUM_FLOPS
                    )
                    or (
                        self.integrity_enabled
                        and fallback_record.get("operation") == "multi_feature"
                        and fallback_record.get("transpose_a") == "T"
                        and fallback_record.get("transpose_b") == "N"
                        and int(fallback_record["m"])
                        <= _DETERMINISTIC_FEATURE_TN_MAXIMUM_ROWS
                        and int(fallback_record["n"])
                        <= _DETERMINISTIC_FEATURE_TN_MAXIMUM_COLUMNS
                        and int(fallback_record["k"])
                        <= _DETERMINISTIC_FEATURE_TN_MAXIMUM_REDUCTION
                        and fallback_flops
                        <= _DETERMINISTIC_FEATURE_TN_MAXIMUM_FLOPS
                    )
                )
            )
            fallback_record["telemetry_scope"] = (
                "deterministic_tiled_call_boundary"
                if deterministic_tiled
                else (
                    "protected_call_boundary" if self.protected
                    else "python_call_boundary"
                )
            )
            if output_evidence is not None:
                fallback_record["native_gemm_output_numa"] = output_evidence
            records = [fallback_record]
        context = self._semantic_context(semantic)
        hot_phase_name = context.get("phase")
        hot_phase = hot_phase_name in self._hot_gemm_phase_counts
        if hot_phase:
            phase_counts = self._hot_gemm_phase_counts[hot_phase_name]
            self._hot_gemm_logical_calls += 1
            phase_counts["logical_calls"] += 1
            if vendor_records:
                self._hot_gemm_vendor_observed_calls += 1
                phase_counts["vendor_observed_calls"] += 1
            elif records[0]["telemetry_scope"] == "deterministic_tiled_call_boundary":
                self._hot_gemm_deterministic_tiled_calls += 1
                phase_counts["deterministic_tiled_calls"] += 1
            else:
                self._hot_gemm_unobserved_calls += 1
                phase_counts["unobserved_calls"] += 1
        for observed in records:
            record = _json_safe(observed)
            for dtype_key in (
                "actual_left_storage_dtype",
                "actual_right_storage_dtype",
                "actual_output_storage_dtype",
            ):
                record.setdefault(dtype_key, fallback.get(dtype_key))
            if vendor_records:
                record.setdefault("blas_backend", record.get("backend"))
                record.setdefault(
                    "blas_backend_corename", record.get("backend_corename")
                )
                record.setdefault(
                    "blas_backend_config", record.get("backend_config")
                )
                record["backend"] = self.backend_name
            record.setdefault(
                "telemetry_scope",
                "vendor_call" if vendor_records else (
                    "protected_call_boundary" if self.protected
                    else "python_call_boundary"
                ),
            )
            record.setdefault("backend", self.backend_name)
            record.setdefault("backend_version", self.backend_version)
            record.setdefault("arithmetic_dtype", self.arithmetic_dtype.name)
            record.setdefault("requested_storage_dtype", self.storage_dtype.name)
            record.setdefault("requested_blas_threads", self.threads)
            record.setdefault(
                "affinity_core_list",
                record.get("cpu_affinity_list", list(self._affinity_cores)),
            )
            entry_cpu = int(record.get("entry_cpu", -1))
            operand_numa = record.get("operand_numa_page_samples")
            record.setdefault(
                "numa_node_placement",
                {
                    "entry_cpu_node": self._cpu_numa_nodes.get(entry_cpu),
                    "process_policy": dict(self._numa_process_evidence),
                    "operand_memory_page_samples": operand_numa,
                },
            )
            record.setdefault(
                "numa_evidence",
                (
                    "entry CPU mapped through sysfs; A/B/C pages sampled with "
                    "move_pages after the vendor call"
                    if operand_numa is not None
                    else "entry CPU mapped through sysfs; operand page placement "
                    "not available for this call-boundary record"
                ),
            )
            for key, value in context.items():
                record.setdefault(key, value)
            m, n, k = (int(record[key]) for key in ("m", "n", "k"))
            record.setdefault(
                "flops", int(record.get("flop_count", 2 * m * n * k))
            )
            wall = float(record.get("wall_seconds", 0.0))
            cpu = float(record.get("process_cpu_seconds", 0.0))
            if wall > 0.0:
                record.setdefault(
                    "achieved_gflops",
                    record.get("gflops_per_second", record["flops"] / wall / 1.0e9),
                )
                record.setdefault(
                    "average_active_cores",
                    record.get("active_core_equivalents", cpu / wall),
                )
            if record.get("omp_in_parallel") is not None:
                record.setdefault(
                    "call_site_inside_openmp", bool(record["omp_in_parallel"])
                )
            if "sequence" in record:
                record["native_sequence"] = record["sequence"]
            record["sequence"] = len(self._gemm_records) + 1
            if len(self._gemm_records) < _MAX_GEMM_TELEMETRY_RECORDS:
                self._gemm_records.append(record)
            else:
                self._dropped_gemm_records += 1

    def performance_report(
        self,
        *,
        estimator_phase_timings: Mapping | None = None,
        include_records: bool,
        capture_boundary: str = "current_executor_state",
        phase_telemetry_complete: bool = False,
    ) -> dict:
        phases = {}
        for name, observed in sorted(self._phase_totals.items()):
            record = dict(observed)
            wall = float(record["wall_seconds"])
            cpu = float(record["process_cpu_seconds"])
            record["average_active_cores"] = cpu / wall if wall > 0.0 else 0.0
            phases[name] = record
        status = {}
        if self._native_telemetry_status_getter is not None:
            try:
                status = _json_safe(dict(self._native_telemetry_status_getter()))
            except Exception as exc:
                self._native_telemetry_errors.append(
                    f"native_telemetry_status_failed: {type(exc).__name__}: {exc}"
                )
        output_numa_status = {}
        output_numa_status_getter = getattr(
            self, "_native_gemm_output_numa_status_getter", None
        )
        if output_numa_status_getter is not None:
            try:
                output_numa_status = _json_safe(
                    dict(output_numa_status_getter())
                )
            except Exception as exc:
                self._native_telemetry_errors.append(
                    "native_gemm_output_numa_status_failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        native_dropped = int(status.get("dropped_records", 0))
        vendor_record_count = sum(
            record.get("telemetry_scope") == "vendor_call"
            for record in self._gemm_records
        )
        protected_fallback_count = sum(
            record.get("telemetry_scope") == "protected_call_boundary"
            for record in self._gemm_records
        )
        native_captured = int(status.get("captured_records", -1))
        vendor_call_telemetry_complete = bool(
            not self.protected
            or (
                self._native_telemetry_available
                and self._native_telemetry_status_getter is not None
                and not self._native_telemetry_errors
                and native_dropped == 0
                and self._dropped_gemm_records == 0
                and native_captured == vendor_record_count
            )
        )
        output_numa_contract_supported = bool(getattr(
            self, "_native_gemm_output_numa_contract_supported", False
        ))
        output_numa_contract_required = bool(getattr(
            self, "_native_gemm_output_numa_contract_required", False
        ))
        output_numa_records = list(getattr(
            self, "_native_gemm_output_numa_records", []
        ))
        output_numa_python_dropped = int(getattr(
            self, "_dropped_native_gemm_output_numa_records", 0
        ))
        output_numa_native_dropped = int(
            output_numa_status.get("dropped_records", 0)
        )
        output_numa_captured = int(
            output_numa_status.get("captured_records", -1)
        )
        output_numa_complete = bool(
            not self.protected
            or not output_numa_contract_required
            or not output_numa_contract_supported
            or (
                getattr(self, "_native_gemm_output_numa_available", False)
                and output_numa_status_getter is not None
                and output_numa_native_dropped == 0
                and output_numa_python_dropped == 0
                and output_numa_status.get("buffered_records") == 0
                and output_numa_captured == len(output_numa_records)
                and output_numa_status.get("attempted_calls")
                == len(output_numa_records)
                and output_numa_status.get("verified_calls")
                == len(output_numa_records)
                and output_numa_status.get("legacy_calls") == 0
                and output_numa_status.get("failed_calls") == 0
            )
        )
        hot_gemm_phase_coverage = {}
        for phase in ("source_gemm", "target_gemm"):
            phase_counts = dict(self._hot_gemm_phase_counts[phase])
            expected_calls = int(
                self._phase_totals.get(phase, {}).get("calls", 0)
            )
            phase_counts["expected_calls"] = expected_calls
            phase_counts["complete"] = bool(
                expected_calls > 0
                and phase_counts["unobserved_calls"] == 0
                and phase_counts["logical_calls"] == expected_calls
                and phase_counts["logical_calls"]
                == phase_counts["vendor_observed_calls"]
                + phase_counts["deterministic_tiled_calls"]
            )
            hot_gemm_phase_coverage[phase] = phase_counts
        expected_hot_gemm_calls = sum(
            record["expected_calls"]
            for record in hot_gemm_phase_coverage.values()
        )
        hot_gemm_telemetry_complete = bool(
            not self.protected
            or all(
                record["complete"]
                for record in hot_gemm_phase_coverage.values()
            )
        )
        optimized_layout_telemetry = (
            self._optimized_layout_telemetry_status()
        )
        optimized_layout_telemetry_complete = bool(
            optimized_layout_telemetry["complete"]
        )
        numa_bound_decode_complete = bool(
            not self._numa_bound_bed_decode_required
            or (
                isinstance(self._numa_bound_bed_decode_report, Mapping)
                and self._numa_bound_bed_decode_report.get("complete") is True
            )
        )
        packed_source_records = list(getattr(
            self, "_packed_source_panel_numa_records", []
        ))
        packed_source_required = bool(
            self.protected
            and getattr(self, "_multi_environment_direct_context_required", False)
            and self.cpu_placement is not None
        )
        packed_source_complete = bool(
            not packed_source_required
            or (
                getattr(self, "_packed_source_panel_numa_complete", False)
                and packed_source_records
            )
        )
        gemm_wall = sum(
            float(record.get("wall_seconds", 0.0)) for record in self._gemm_records
        )
        gemm_cpu = sum(
            float(record.get("process_cpu_seconds", 0.0))
            for record in self._gemm_records
        )
        native_kernel_info = None
        native_kernel_complete = not self.protected
        native_kernel = getattr(self, "_multi_environment_kernel", None)
        if native_kernel is not None:
            native_kernel_info = _json_safe(
                dict(native_kernel.info())
            )
            expected_kernel_info = dict(self._multi_environment_kernel_info)
            expected_kernel_info.update(
                self._multi_environment_kernel_call_counts
            )
            expected_kernel_info["repaired_gemm_output_columns"] = int(
                self._multi_environment_kernel_repaired_columns
            )
            expected_kernel_info["checksum_recomputed_gemm_output_columns"] = int(
                self._multi_environment_kernel_checksum_recomputed_columns
            )
            expected_kernel_info["roundoff_only_gemm_output_columns"] = int(
                self._multi_environment_kernel_roundoff_only_columns
            )
            native_kernel_complete = native_kernel_info == expected_kernel_info
            if not native_kernel_complete:
                raise RuntimeError(
                    "The multi-environment native kernel counters disagree with "
                    f"the Python call boundary: {native_kernel_info}."
                )
        elif getattr(self, "_multi_environment_direct_result", None) is not None:
            native_kernel_info = _json_safe(
                dict(self._multi_environment_direct_result["kernel_info"])
            )
            expected_kernel_info = dict(self._multi_environment_kernel_info)
            expected_kernel_info.update(self._multi_environment_kernel_call_counts)
            expected_kernel_info["repaired_gemm_output_columns"] = int(
                self._multi_environment_kernel_repaired_columns
            )
            expected_kernel_info["checksum_recomputed_gemm_output_columns"] = int(
                self._multi_environment_kernel_checksum_recomputed_columns
            )
            expected_kernel_info["roundoff_only_gemm_output_columns"] = int(
                self._multi_environment_kernel_roundoff_only_columns
            )
            native_kernel_complete = native_kernel_info == expected_kernel_info
            if not native_kernel_complete:
                raise RuntimeError(
                    "The descriptor-owned multi-environment kernel counters "
                    "disagree with replayed call boundaries."
                )
        native_direct_info = None
        native_direct_required = bool(getattr(
            self, "_multi_environment_direct_context_required", False
        ))
        native_direct_complete = not native_direct_required
        if getattr(self, "_multi_environment_direct_context", None) is not None:
            native_direct_info = _json_safe(
                dict(self._multi_environment_direct_context.info())
            )
            expected_direct_info = dict(
                self._multi_environment_direct_context_info
            )
            expected_direct_info["completed"] = (
                getattr(self, "_multi_environment_direct_result", None) is not None
            )
            native_direct_complete = (
                getattr(self, "_multi_environment_direct_result", None) is not None
                and native_direct_info == expected_direct_info
            )
            direct_result = getattr(
                self, "_multi_environment_direct_result", None
            )
            if native_direct_complete and isinstance(direct_result, Mapping):
                native_direct_info = dict(native_direct_info)
                native_direct_info["decoded_scratch_allocations"] = int(
                    direct_result.get("decoded_scratch_allocations", 0)
                )
                native_direct_info["decoded_scratch_reuses"] = int(
                    direct_result.get("decoded_scratch_reuses", 0)
                )
        report = {
            "schema_version": _PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
            "bounded": True,
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "arithmetic_dtype": self.arithmetic_dtype.name,
            "requested_storage_dtype": self.storage_dtype.name,
            "requested_sketch_storage_dtype": self.storage_dtype.name,
            "full_precision_layout": self.full_precision_layout,
            "requested_blas_threads": self.threads,
            "gemm_integrity_enabled": bool(getattr(
                self, "integrity_enabled", False
            )),
            "gemm_checksum_enabled": bool(getattr(
                self, "checksum_enabled", False
            )),
            "multi_environment_native_kernel_required": bool(self.protected),
            "multi_environment_native_kernel_complete": bool(
                native_kernel_complete
            ),
            "multi_environment_native_kernel": native_kernel_info,
            "multi_environment_direct_context_required": bool(
                native_direct_required
            ),
            "multi_environment_direct_context_complete": bool(
                native_direct_complete
            ),
            "multi_environment_direct_context": native_direct_info,
            "execution_scratch_release": _json_safe(
                getattr(self, "_execution_scratch_release_evidence", None)
            ),
            "early_numa_attestation": _json_safe(
                self._numa_process_evidence.get("early_numa_attestation")
            ),
            "native_telemetry_available": self._native_telemetry_available,
            "native_telemetry_status": status,
            "vendor_call_telemetry_complete": vendor_call_telemetry_complete,
            "vendor_call_record_count": vendor_record_count,
            "protected_call_boundary_fallback_count": protected_fallback_count,
            "expected_protected_gemm_call_count": self.nn_calls + self.tn_calls,
            "hot_gemm_logical_call_count": self._hot_gemm_logical_calls,
            "expected_hot_gemm_call_count": expected_hot_gemm_calls,
            "hot_gemm_vendor_observed_call_count": (
                self._hot_gemm_vendor_observed_calls
            ),
            "hot_gemm_deterministic_tiled_call_count": (
                self._hot_gemm_deterministic_tiled_calls
            ),
            "hot_gemm_unobserved_call_count": self._hot_gemm_unobserved_calls,
            "hot_gemm_phase_coverage": hot_gemm_phase_coverage,
            "hot_gemm_telemetry_complete": hot_gemm_telemetry_complete,
            "optimized_fp64_layout_telemetry": optimized_layout_telemetry,
            "optimized_fp64_layout_telemetry_complete": (
                optimized_layout_telemetry_complete
            ),
            "native_gemm_output_numa_contract_supported": (
                output_numa_contract_supported
            ),
            "native_gemm_output_numa_contract_required": (
                output_numa_contract_required
            ),
            "native_gemm_output_numa_evidence_available": bool(getattr(
                self, "_native_gemm_output_numa_available", False
            )),
            "native_gemm_output_numa_evidence_status": output_numa_status,
            "native_gemm_output_numa_evidence_complete": output_numa_complete,
            "native_gemm_output_numa_record_count": len(output_numa_records),
            "dropped_native_gemm_output_numa_records": output_numa_python_dropped,
            "packed_source_panel_numa_required": packed_source_required,
            "packed_source_panel_numa_complete": packed_source_complete,
            "packed_source_panel_numa_record_count": len(
                packed_source_records
            ),
            "repaired_gemm_output_columns": int(self.repaired_output_columns),
            "native_telemetry_errors": list(self._native_telemetry_errors),
            "telemetry_complete": (
                native_dropped == 0
                and self._dropped_gemm_records == 0
                and phase_telemetry_complete
                and optimized_layout_telemetry_complete
                and numa_bound_decode_complete
                and native_kernel_complete
                and native_direct_complete
                and packed_source_complete
                and (
                    not self.protected
                    or (
                        vendor_call_telemetry_complete
                        and hot_gemm_telemetry_complete
                        and output_numa_complete
                    )
                )
            ),
            "capture_boundary": str(capture_boundary),
            "phase_telemetry_complete": bool(phase_telemetry_complete),
            "phase_accounting": (
                "Phase totals are named intervals and may overlap their enclosing "
                "whole-phase totals; estimator phase totals are summed across "
                "environment-local output work."
            ),
            "phase_totals": phases,
            "estimator_phase_totals": _json_safe(
                {} if estimator_phase_timings is None else estimator_phase_timings
            ),
            "gemm_record_count": len(self._gemm_records),
            "dropped_gemm_records": self._dropped_gemm_records,
            "gemm_wall_seconds": gemm_wall,
            "gemm_process_cpu_seconds": gemm_cpu,
            "gemm_average_active_cores": (
                gemm_cpu / gemm_wall if gemm_wall > 0.0 else 0.0
            ),
            "gemm_matrix_minutes": gemm_wall / 60.0,
        }
        if include_records:
            report["gemm_records"] = list(self._gemm_records)
            if output_numa_contract_supported:
                report["native_gemm_output_numa_records"] = output_numa_records
            if packed_source_records:
                report["packed_source_panel_numa_records"] = (
                    packed_source_records
                )
        if self._numa_bound_bed_decode_required:
            report["numa_bound_bed_decode_required"] = True
            report["numa_bound_bed_decode_complete"] = (
                numa_bound_decode_complete
            )
            report["numa_bound_bed_decode"] = (
                None
                if self._numa_bound_bed_decode_report is None
                else _decode_report_for_output(
                    self._numa_bound_bed_decode_report,
                    include_records=include_records,
                )
            )
        if self.cpu_placement is not None:
            if self.cpu_placement_complete is not True:
                raise RuntimeError("CPU placement evidence became incomplete.")
            report["cpu_placement"] = dict(self.cpu_placement)
            report["cpu_placement_complete"] = True
        return report

    def _optimized_layout_telemetry_status(self) -> dict:
        """Retain the telemetry schema while reporting current-layout records."""
        required = False
        phases = {}
        violations = []
        specifications = {
            "source_gemm": {
                "vendor_operation": "dgemm_nn",
                "deterministic_operations": ("nn",),
                "layout": "column_major",
                "transpose_a": "N",
                "transpose_b": "N",
                "leading_dimensions": ("m", "k", "m"),
            },
            "target_gemm": {
                "vendor_operation": "dgemm_tn",
                "deterministic_operations": ("tn_pair",),
                "layout": "column_major",
                "transpose_a": "T",
                "transpose_b": "N",
                "leading_dimensions": ("k", "k", "m"),
            },
        }
        for phase, specification in specifications.items():
            records = [
                record for record in self._gemm_records
                if record.get("phase") == phase
            ]
            expected_calls = int(
                self._hot_gemm_phase_counts[phase]["logical_calls"]
            )
            invalid = 0
            for record in records:
                try:
                    dimensions = {
                        key: int(record[key]) for key in ("m", "n", "k")
                    }
                    observed_leading_dimensions = tuple(
                        int(record[key]) for key in ("lda", "ldb", "ldc")
                    )
                except (KeyError, TypeError, ValueError):
                    dimensions = {}
                    observed_leading_dimensions = ()
                expected_leading_dimensions = tuple(
                    dimensions.get(key)
                    for key in specification["leading_dimensions"]
                )
                telemetry_scope = record.get("telemetry_scope")
                operation = record.get("operation")
                operation_valid = bool(
                    (
                        telemetry_scope == "vendor_call"
                        and operation == specification["vendor_operation"]
                    )
                    or (
                        telemetry_scope == "deterministic_tiled_call_boundary"
                        and operation in specification["deterministic_operations"]
                    )
                )
                valid = bool(
                    dimensions
                    and all(value > 0 for value in dimensions.values())
                    and operation_valid
                    and record.get("layout") == specification["layout"]
                    and record.get("transpose_a")
                    == specification["transpose_a"]
                    and record.get("transpose_b")
                    == specification["transpose_b"]
                    and observed_leading_dimensions
                    == expected_leading_dimensions
                )
                if valid:
                    continue
                invalid += 1
                if len(violations) < 16:
                    violations.append(
                        {
                            "phase": phase,
                            "sequence": record.get("sequence"),
                            "telemetry_scope": telemetry_scope,
                            "operation": operation,
                            "layout": record.get("layout"),
                            "transpose_a": record.get("transpose_a"),
                            "transpose_b": record.get("transpose_b"),
                            "m": record.get("m"),
                            "n": record.get("n"),
                            "k": record.get("k"),
                            "lda": record.get("lda"),
                            "ldb": record.get("ldb"),
                            "ldc": record.get("ldc"),
                        }
                    )
            phases[phase] = {
                "expected_logical_calls": expected_calls,
                "record_count": len(records),
                "invalid_record_count": invalid,
                "complete": bool(
                    not required
                    or (
                        expected_calls > 0
                        and len(records) == expected_calls
                        and invalid == 0
                    )
                ),
            }
        complete = bool(
            not required
            or all(record["complete"] for record in phases.values())
        )
        return {
            "schema_version": 1,
            "required": required,
            "full_precision_layout": self.full_precision_layout,
            "complete": complete,
            "phases": phases,
            "violation_count": sum(
                int(record["invalid_record_count"])
                for record in phases.values()
            ),
            "violations": violations,
        }

    @staticmethod
    def _fallback_gemm_record(
        operation: str,
        left: np.ndarray,
        n: int,
        left_dtype,
        right_dtype,
        output_dtype,
        *,
        layout: str,
        transpose_a: str,
        ldb: int,
        ldc: int,
        alpha: float,
        beta: float,
        wall_seconds: float,
        process_cpu_seconds: float,
        transpose_b: str = "N",
        lda: int | None = None,
    ) -> dict:
        m = int(left.shape[1] if transpose_a == "T" else left.shape[0])
        k = int(left.shape[0] if transpose_a == "T" else left.shape[1])
        default_lda = (
            left.shape[1] if str(layout) == "row_major" else left.shape[0]
        )
        return {
            "operation": operation,
            "arithmetic_dtype": np.result_type(left_dtype, right_dtype).name,
            "actual_left_storage_dtype": np.dtype(left_dtype).name,
            "actual_right_storage_dtype": np.dtype(right_dtype).name,
            "actual_output_storage_dtype": np.dtype(output_dtype).name,
            "layout": str(layout),
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
            "m": m,
            "n": int(n),
            "k": k,
            "lda": int(default_lda if lda is None else lda),
            "ldb": int(ldb),
            "ldc": int(ldc),
            "alpha": float(alpha),
            "beta": float(beta),
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "omp_in_parallel": None,
            "completed": True,
        }

    def nn(
        self, left: np.ndarray, right: np.ndarray, *, semantic: Mapping | None = None
    ) -> np.ndarray:
        self.nn_calls += 1
        shape = ("nn", int(left.shape[0]), int(left.shape[1]), int(right.shape[1]))
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        if self.protected:
            assert self._module is not None
            result, repaired = self._module.protected_matmul_nn(
                np.asfortranarray(left, dtype=np.float64),
                np.asfortranarray(right, dtype=np.float64),
                self.threads,
            )
            self.repaired_output_columns += int(repaired)
            result = np.asarray(result)
        else:
            result = left @ right
        self._record_gemm(
            self._fallback_gemm_record(
                "nn",
                left,
                right.shape[1],
                np.float64 if self.protected else left.dtype,
                np.float64 if self.protected else right.dtype,
                result.dtype,
                layout="column_major" if self.protected else "numpy",
                transpose_a="N",
                ldb=right.shape[0],
                ldc=left.shape[0],
                alpha=1.0,
                beta=0.0,
                wall_seconds=time.perf_counter() - wall_started,
                process_cpu_seconds=time.process_time() - cpu_started,
            ),
            semantic,
            protected_output_expected=self.protected,
        )
        return result

    def source_product(
        self,
        genotype: np.ndarray,
        weights: np.ndarray,
        *,
        semantic: Mapping | None = None,
    ) -> np.ndarray:
        """Return logical ``genotype @ weights`` in the current layout."""
        return self.nn(genotype, weights, semantic=semantic)

    def tn(
        self, left: np.ndarray, right: np.ndarray, *, semantic: Mapping | None = None
    ) -> np.ndarray:
        self.tn_calls += 1
        shape = ("tn", int(left.shape[1]), int(left.shape[0]), int(right.shape[1]))
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        if self.protected:
            assert self._module is not None
            result, repaired = self._module.protected_matmul_tn(
                np.asfortranarray(left, dtype=np.float64),
                np.asfortranarray(right, dtype=np.float64),
                self.threads,
            )
            self.repaired_output_columns += int(repaired)
            result = np.asarray(result)
        else:
            result = left.T @ right
        self._record_gemm(
            self._fallback_gemm_record(
                "tn",
                left,
                right.shape[1],
                np.float64 if self.protected else left.dtype,
                np.float64 if self.protected else right.dtype,
                result.dtype,
                layout="column_major" if self.protected else "numpy",
                transpose_a="T",
                ldb=right.shape[0],
                ldc=left.shape[1],
                alpha=1.0,
                beta=0.0,
                wall_seconds=time.perf_counter() - wall_started,
                process_cpu_seconds=time.process_time() - cpu_started,
            ),
            semantic,
            protected_output_expected=self.protected,
        )
        return result

    def nn_update(
        self,
        left: np.ndarray,
        right: np.ndarray,
        target: np.ndarray,
        *,
        semantic: Mapping | None = None,
    ) -> None:
        """Apply target -= left @ right without a projection allocation."""
        self.nn_calls += 1
        shape = ("nn", int(left.shape[0]), int(left.shape[1]), int(right.shape[1]))
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        if self.protected:
            assert self._module is not None
            self._module.protected_rank_update_nn(
                np.asfortranarray(left, dtype=np.float64),
                np.asfortranarray(right, dtype=np.float64),
                target,
                self.threads,
            )
        else:
            target -= left @ right
        self._record_gemm(
            self._fallback_gemm_record(
                "nn_update",
                left,
                right.shape[1],
                np.float64 if self.protected else left.dtype,
                np.float64 if self.protected else right.dtype,
                target.dtype,
                layout="column_major" if self.protected else "numpy",
                transpose_a="N",
                ldb=right.shape[0],
                ldc=target.shape[0],
                alpha=-1.0,
                beta=1.0,
                wall_seconds=time.perf_counter() - wall_started,
                process_cpu_seconds=time.process_time() - cpu_started,
            ),
            semantic,
        )

    def fused_feature_scalar_moments(
        self, genotype: np.ndarray, environments: np.ndarray
    ) -> np.ndarray:
        if not self.protected or self._module is None:
            raise RuntimeError(
                "Fused feature reductions require the protected native executor."
            )
        return np.asarray(
            self._module.fused_feature_scalar_moments(
                np.asfortranarray(genotype, dtype=np.float64),
                np.asfortranarray(environments, dtype=np.float64),
                self.threads,
            ),
            dtype=np.float64,
        )

    def initialize_multi_environment_kernel(
        self,
        plan: Mapping,
        common_basis: np.ndarray,
        directions: np.ndarray,
        estimators: Sequence[GenomewideEnvLDScore],
    ) -> dict:
        """Freeze the complete shared numerical plan in one native owner."""
        if not self.protected or self._module is None:
            raise RuntimeError(
                "The end-to-end multi-environment kernel requires the protected backend."
            )
        if self._multi_environment_kernel is not None:
            raise RuntimeError("The multi-environment native kernel is already initialized.")
        packed = _pack_native_feature_plan(plan, estimators)
        kernel = self._module.MultiEnvironmentKernel(
            environments=np.asfortranarray(plan["environments"], dtype=np.float64),
            feature_basis=np.asfortranarray(plan["basis"], dtype=np.float64),
            **packed,
            common_basis=np.asfortranarray(common_basis, dtype=np.float64),
            directions=np.asfortranarray(directions, dtype=np.float64),
            ddof=int(estimators[0].ddof),
            annotation_bins=int(estimators[0].nbins),
            threads=self.threads,
        )
        info = _json_safe(dict(kernel.info()))
        expected = {
            "schema": "summit.multi_environment_native_kernel.v1",
            "schema_version": 1,
            "rows": int(estimators[0].nsamp),
            "environment_count": len(estimators),
            "annotation_bins": int(estimators[0].nbins),
            "feature_basis_columns": int(plan["basis"].shape[1]),
            "common_basis_rank": int(common_basis.shape[1]),
            "ddof": int(estimators[0].ddof),
            "threads": self.threads,
            "feature_calls": 0,
            "source_calls": 0,
            "projection_calls": 0,
            "target_calls": 0,
            "normalization_calls": 0,
            "repaired_gemm_output_columns": 0,
            "checksum_recomputed_gemm_output_columns": 0,
            "roundoff_only_gemm_output_columns": 0,
        }
        if info != expected:
            raise RuntimeError(
                f"The multi-environment native kernel disagrees with its frozen plan: {info}."
            )
        self._multi_environment_kernel = kernel
        self._multi_environment_kernel_info = info
        self._multi_environment_kernel_repaired_columns = 0
        self._multi_environment_kernel_checksum_recomputed_columns = 0
        self._multi_environment_kernel_roundoff_only_columns = 0
        return dict(info)

    def initialize_multi_environment_direct_context(
        self,
        plan: Mapping,
        common_basis: np.ndarray,
        directions: np.ndarray,
        estimators: Sequence[GenomewideEnvLDScore],
        blocks: Sequence[tuple[int, int]],
        environment_tiles: Sequence[tuple[int, int]],
        probe_tiles: Sequence[tuple[int, int]],
    ) -> dict | None:
        """Freeze descriptor ownership and the complete numerical plan in C++."""
        if not self.protected or self._module is None:
            return None
        first = estimators[0]
        if first.rand_dist != "rademacher":
            raise ValueError(
                "The descriptor-owned direct GxE pipeline currently requires "
                "--rand-dist rademacher; choose the Python backend for Gaussian "
                "or spherical probes."
            )
        if self._multi_environment_direct_context is not None:
            raise RuntimeError(
                "The multi-environment direct context is already initialized."
            )
        packed = _pack_native_feature_plan(plan, estimators)
        keys = np.empty(
            (len(blocks) * int(first.nvecs), 2), dtype=np.uint64
        )
        key_index = 0
        for block_start, _block_stop in blocks:
            for local_probe in range(int(first.nvecs)):
                probe_id = int(first.probe_offset) + local_probe
                bit_generator = np.random.Philox(
                    _make_seed(int(first.root_seed), int(block_start), probe_id)
                )
                keys[key_index] = np.asarray(
                    bit_generator.state["state"]["key"], dtype=np.uint64
                )
                key_index += 1
        reader_basis = _orthonormalize_columns(
            np.column_stack(
                [np.ones(first.nsamp, dtype=np.float64), first.C_int]
            )
        )
        expected_reader_rank = int(first.p_eff) + 1
        if reader_basis.shape != (first.nsamp, expected_reader_rank):
            raise RuntimeError(
                "The descriptor-owned reader projection basis lost rank."
            )
        complete_memory_plan = first.resource_estimates.get(
            "complete_process_memory_plan"
        )
        if not isinstance(complete_memory_plan, Mapping):
            raise RuntimeError(
                "The descriptor context lacks its complete memory plan."
            )
        direct_kernel_mode = complete_memory_plan.get("direct_kernel_mode")
        if direct_kernel_mode not in {
            "dense_blas_hybrid", "packed_mailman"
        }:
            raise RuntimeError(
                "The descriptor memory plan has no executable kernel mode."
            )
        dense_blas_hybrid = direct_kernel_mode == "dense_blas_hybrid"
        context = self._module.MultiEnvironmentDirectContext(
            bed_descriptor=int(first._genotype_descriptors[".bed"]),
            bim_descriptor=int(first._genotype_descriptors[".bim"]),
            fam_descriptor=int(first._genotype_descriptors[".fam"]),
            row_sel=np.asarray(first.row_sel, dtype=np.int64),
            reader_environment=np.asarray(first.env, dtype=np.float64),
            reader_q_basis=np.asfortranarray(reader_basis, dtype=np.float64),
            decode_threads=int(first.decode_threads),
            max_workspace_bytes=int(first.native_workspace_gib * 1024**3),
            environments=np.asfortranarray(
                plan["environments"], dtype=np.float64
            ),
            feature_basis=np.asfortranarray(plan["basis"], dtype=np.float64),
            **packed,
            common_basis=np.asfortranarray(common_basis, dtype=np.float64),
            directions=np.asfortranarray(directions, dtype=np.float64),
            annotations=np.ascontiguousarray(first.annot, dtype=np.float64),
            annotation_masses=np.asarray(first.nsnps_bin, dtype=np.float64),
            blocks=np.ascontiguousarray(blocks, dtype=np.int64),
            environment_tiles=np.ascontiguousarray(
                environment_tiles, dtype=np.int64
            ),
            probe_tiles=np.ascontiguousarray(probe_tiles, dtype=np.int64),
            philox_keys=keys,
            ddof=int(first.ddof),
            total_probes=int(first.nvecs),
            eps_var=float(first.eps_var),
            standardized=first.kernel_mode == "standardized_projected",
            dense_blas_hybrid=dense_blas_hybrid,
            threads=self.threads,
        )
        # The native constructor copied the Philox key table; drop the Python
        # copy immediately so the only constructor-time overlap is the bounded
        # window inside the constructor call itself, which the plan charges
        # as constructor_transient_overlap.
        del keys
        info = _json_safe(dict(context.info()))
        if self.vendor_probe_chunk_width is not None:
            execution_probe_chunks = sum(
                math.ceil(int(count) / self.vendor_probe_chunk_width)
                for _start, count in probe_tiles
            )
            maximum_execution_probe_chunk_width = min(
                max(int(count) for _start, count in probe_tiles),
                self.vendor_probe_chunk_width,
            )
        else:
            execution_probe_chunks = len(probe_tiles)
            maximum_execution_probe_chunk_width = max(
                int(count) for _start, count in probe_tiles
            )
        fused_two_pass_execution = (
            len(environment_tiles) == 1 and len(probe_tiles) == 1
        )
        execution_tile_products = (
            len(environment_tiles) * execution_probe_chunks
        )
        # One maximum-capacity allocation per native execution-scratch role.
        # These formulas must match the native constructor exactly; the
        # frozen-plan equality check below fails closed on any disagreement.
        max_genotype_block_width = max(
            int(stop) - int(start) for start, stop in blocks
        )
        maximum_environment_tile_width = max(
            int(stop) - int(start) for start, stop in environment_tiles
        )
        wide_capacity_columns = (
            2
            * maximum_environment_tile_width
            * int(first.nbins)
            * maximum_execution_probe_chunk_width
        )
        decoded_scratch_capacity_bytes = (
            int(first.nsamp) * max_genotype_block_width * 8
            if dense_blas_hybrid
            else 0
        )
        source_output_scratch_capacity_bytes = (
            int(first.nsamp) * wide_capacity_columns * 8
            if dense_blas_hybrid
            else 0
        )
        target_output_scratch_capacity_bytes = (
            max_genotype_block_width * 2 * wide_capacity_columns * 8
            if dense_blas_hybrid
            else 0
        )
        # Mirror the native context's frozen Mailman configuration exactly;
        # the info equality below fails closed on any divergence, including
        # an unmodeled SUMMIT_MAILMAN_* override.
        frozen_mailman = _frozen_mailman_environment()
        feature_rhs_columns = (
            int(np.asarray(plan["basis"]).shape[1]) + 1 + 3 * len(estimators)
        )
        expected_mailman = _mailman_worker_scratch_bytes(
            rows=int(first.nsamp),
            feature_rhs_columns=feature_rhs_columns,
            wide_columns=wide_capacity_columns,
            threads=int(self.threads),
            frozen_environment=frozen_mailman,
        )
        mailman_worker_capacity_per_worker = (
            0
            if dense_blas_hybrid
            else expected_mailman["per_worker_bytes"]
        )
        expected = {
            "schema": "summit.multi_environment_direct_context.v3",
            "schema_version": 1,
            "rows": int(first.nsamp),
            "variants": int(first.nsnps),
            "environment_count": len(estimators),
            "annotation_bins": int(first.nbins),
            "probe_count": int(first.nvecs),
            "mailman_maximum_probe_count": (
                _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
            ),
            "mailman_probe_count_eligible": bool(
                first.nvecs <= _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
            ),
            "block_count": len(blocks),
            "environment_tile_count": len(environment_tiles),
            "probe_tile_count": len(probe_tiles),
            "execution_probe_chunk_count": execution_probe_chunks,
            "maximum_execution_probe_chunk_width": (
                maximum_execution_probe_chunk_width
            ),
            "fused_two_pass_execution": fused_two_pass_execution,
            "planned_output_calls": (
                len(blocks) * (1 + 2 * execution_tile_products)
                + (
                    execution_tile_products
                    if common_basis.shape[1] > 0 else 0
                )
                if dense_blas_hybrid
                else (
                    execution_tile_products
                    if common_basis.shape[1] > 0 else 0
                )
            ),
            "planned_semantic_call_maximum": (
                len(blocks) * (1 + 2 * execution_tile_products)
                + 2 * execution_tile_products
            ),
            "planned_genotype_passes": (
                2
                if fused_two_pass_execution
                else 1 + 2 * execution_tile_products
            ),
            "decode_threads": int(first.decode_threads),
            "threads": self.threads,
            "descriptor_owned_bed": True,
            "native_probe_generation": True,
            "execution_kernel": (
                "dense_private_blas_streamed_pair"
                if dense_blas_hybrid
                else "packed_mailman_low_memory_fallback"
            ),
            "packed_genotype_mailman": not dense_blas_hybrid,
            "packed_feature_moments": not dense_blas_hybrid,
            "packed_source_direct_accumulation": not dense_blas_hybrid,
            "virtual_environment_weighted_target_rhs": not dense_blas_hybrid,
            "materialized_target_rhs": dense_blas_hybrid,
            "dense_genotype_feature_decode": dense_blas_hybrid,
            "dense_genotype_target_decode": dense_blas_hybrid,
            "persistent_dense_decode_scratch": dense_blas_hybrid,
            "persistent_environment_weighted_target_panel": (
                dense_blas_hybrid
            ),
            "source_panel_released_before_target": False,
            "source_panel_reused_as_protected_pair_first_half": False,
            "contracted_numa_decode": bool(
                dense_blas_hybrid and self.cpu_placement is not None
            ),
            "contracted_packed_source_panel_numa": (
                self.cpu_placement is not None
            ),
            "max_genotype_block_width": max_genotype_block_width,
            "mailman_frozen_segment_size": expected_mailman["segment_size"],
            "mailman_frozen_table_size": expected_mailman["table_size"],
            "mailman_qpanel_feature": expected_mailman["qpanel_feature"],
            "mailman_qpanel_source": expected_mailman["qpanel_source"],
            "mailman_qpanel_target": expected_mailman["qpanel_target"],
            "mailman_worker_scratch_capacity_bytes_per_worker": (
                mailman_worker_capacity_per_worker
            ),
            "decoded_scratch_capacity_bytes": decoded_scratch_capacity_bytes,
            "source_output_scratch_capacity_bytes": (
                source_output_scratch_capacity_bytes
            ),
            "target_output_scratch_capacity_bytes": (
                target_output_scratch_capacity_bytes
            ),
            "single_use": True,
            "completed": False,
            "execution_scratch_released": False,
            "decoded_scratch_released_bytes": 0,
        }
        if info != expected:
            raise RuntimeError(
                "The multi-environment direct context disagrees with its "
                f"frozen plan: {info}."
            )
        self._numa_bound_bed_decode_required = bool(
            dense_blas_hybrid and self.cpu_placement is not None
        )
        self._multi_environment_direct_context = context
        self._multi_environment_direct_context_info = info
        self._multi_environment_direct_context_required = True
        # Admitted per-role scratch capacities from the frozen plan; the run
        # wrapper fails closed if native-reported capacity ever exceeds them.
        component_bytes = complete_memory_plan["component_bytes"]
        self._multi_environment_admitted_scratch_bytes = {
            "decoded_scratch_capacity_bytes": int(
                component_bytes["decoded_genotype_block"]
            ),
            "source_output_scratch_capacity_bytes": int(
                component_bytes["source_contribution"]
            ),
            "target_output_scratch_capacity_bytes": int(
                component_bytes["target_output"]
            ),
            "source_weights": int(component_bytes["source_weights"]),
            "source_annotation": int(
                component_bytes["source_block_annotation"]
            ),
            # The plan charges the packed feature projected and scalar
            # panels as one combined term; their capacities are compared
            # against it jointly.
            "feature_products_combined": int(
                component_bytes["feature_mailman_products"]
            ),
            "target_mailman_output": int(component_bytes["target_output"]),
            "mailman_worker_scratch": int(
                component_bytes["mailman_worker_scratch"]
            ),
        }
        return dict(info)

    def run_multi_environment_direct_context(self) -> dict:
        """Execute the frozen descriptor context and replay bounded telemetry."""
        context = self._multi_environment_direct_context
        if context is None:
            raise RuntimeError("The multi-environment direct context is unavailable.")
        if self._multi_environment_direct_result is not None:
            raise RuntimeError("The multi-environment direct context already ran.")
        with self.phase("native_descriptor_end_to_end"):
            result = dict(context.run())
        completed_info = _json_safe(dict(result["context_info"]))
        expected_info = dict(self._multi_environment_direct_context_info)
        expected_info["completed"] = True
        if completed_info != expected_info:
            raise RuntimeError(
                "The completed descriptor context disagrees with its frozen plan."
            )
        expected_reads = (
            int(expected_info["planned_genotype_passes"])
            * int(expected_info["block_count"])
        )
        observed_reads = result.get("observed_genotype_block_reads")
        if (
            isinstance(observed_reads, bool)
            or not isinstance(observed_reads, int)
            or observed_reads != expected_reads
        ):
            raise RuntimeError(
                "The descriptor context genotype-read count disagrees with its "
                f"pass plan: expected={expected_reads}, observed={observed_reads}."
            )
        expected_nodes = None
        if self.cpu_placement is not None:
            expected_nodes = _validated_early_numa_nodes(
                self._numa_process_evidence.get("early_numa_attestation"),
                require_static=True,
                expected_pid=os.getpid(),
            )
        self._packed_source_panel_numa_records = (
            _validate_packed_source_panel_numa_records(
                result.get("packed_source_panel_numa_records"),
                expected_nodes=expected_nodes,
                expected_count=(
                    int(expected_info["environment_tile_count"])
                    * int(expected_info["execution_probe_chunk_count"])
                ),
                expected_combined_pair=bool(
                    expected_info["materialized_target_rhs"]
                ),
            )
        )
        self._packed_source_panel_numa_complete = bool(
            expected_nodes is None
            or all(
                record.get("contract_required") is True
                and record.get("complete") is True
                for record in self._packed_source_panel_numa_records
            )
        )
        observed_semantic_records = len(result.get("calls", ()))
        if observed_semantic_records > int(
            expected_info["planned_semantic_call_maximum"]
        ):
            raise RuntimeError(
                "The descriptor context produced more semantic call records "
                f"than its frozen plan admits: {observed_semantic_records} > "
                f"{expected_info['planned_semantic_call_maximum']}."
            )
        self._replay_multi_environment_direct_calls(result)
        kernel_info = _json_safe(dict(result["kernel_info"]))
        self._require_native_scratch_within_admitted_plan(
            completed_info, kernel_info
        )
        self._multi_environment_kernel_info = {
            key: value
            for key, value in kernel_info.items()
            if key
            not in {
                "feature_calls", "packed_feature_calls", "source_calls",
                "packed_source_calls", "projection_calls", "target_calls",
                "packed_target_calls", "normalization_calls",
                "repaired_gemm_output_columns",
                "checksum_recomputed_gemm_output_columns",
                "roundoff_only_gemm_output_columns",
            }
        }
        for key in self._multi_environment_kernel_call_counts:
            self._multi_environment_kernel_call_counts[key] = int(
                kernel_info[key]
            )
        self._multi_environment_kernel_repaired_columns = int(
            kernel_info["repaired_gemm_output_columns"]
        )
        self._multi_environment_kernel_checksum_recomputed_columns = int(
            kernel_info["checksum_recomputed_gemm_output_columns"]
        )
        self._multi_environment_kernel_roundoff_only_columns = int(
            kernel_info["roundoff_only_gemm_output_columns"]
        )
        self.repaired_output_columns += self._multi_environment_kernel_repaired_columns
        self._multi_environment_direct_result = result
        # Every final score, diagnostic, same-person, and evidence value has
        # been detached into separately owned arrays and validated above;
        # release the execution-only native scratch before any pandas
        # construction or compressed serialization begins.
        self._release_native_execution_scratch()
        return result

    def _release_native_execution_scratch(self) -> None:
        """Release native execution scratch after a completed, validated run."""
        context = getattr(self, "_multi_environment_direct_context", None)
        if context is None or getattr(
            self, "_multi_environment_direct_result", None
        ) is None:
            return
        rss_before = _process_rss_bytes()
        evidence = _json_safe(dict(context.release_execution_scratch()))
        if evidence.get("released") is not True or (
            evidence.get("live_scratch_capacity_bytes_after") != 0
        ):
            raise RuntimeError(
                "Native execution scratch was not fully released before "
                f"publication: {evidence}."
            )
        evidence["rss_bytes_before_release"] = rss_before
        evidence["rss_bytes_after_release"] = _process_rss_bytes()
        self._execution_scratch_release_evidence = evidence
        # The stored frozen-plan expectation must reflect the legitimate
        # released state so the report-time info equality keeps failing
        # closed on any other drift.
        stored = dict(self._multi_environment_direct_context_info)
        stored["execution_scratch_released"] = True
        stored["decoded_scratch_released_bytes"] = int(
            evidence["decoded_scratch_released_bytes"]
        )
        self._multi_environment_direct_context_info = stored

    def _require_native_scratch_within_admitted_plan(
        self, completed_info: Mapping, kernel_info: Mapping
    ) -> None:
        """Fail closed if any native scratch capacity exceeds the admitted plan."""
        admitted = getattr(
            self, "_multi_environment_admitted_scratch_bytes", None
        )
        if not isinstance(admitted, Mapping):
            raise RuntimeError(
                "The admitted native scratch capacities are unavailable."
            )
        for key in (
            "decoded_scratch_capacity_bytes",
            "source_output_scratch_capacity_bytes",
            "target_output_scratch_capacity_bytes",
        ):
            observed = completed_info.get(key)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed < 0
                or observed > int(admitted[key])
            ):
                raise RuntimeError(
                    "Native execution scratch exceeds its admitted capacity: "
                    f"{key} observed={observed}, admitted={admitted[key]}."
                )
        roles = kernel_info.get("scratch_roles")
        if not isinstance(roles, Mapping):
            raise RuntimeError(
                "The native kernel did not report its scratch-role telemetry."
            )

        def role_capacity(name: str) -> int:
            role = roles.get(name)
            capacity = None if not isinstance(role, Mapping) else role.get(
                "capacity_bytes"
            )
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 0
            ):
                raise RuntimeError(
                    f"Native scratch role {name!r} reported invalid capacity."
                )
            return capacity

        role_bounds = (
            ("source_output", int(admitted["source_output_scratch_capacity_bytes"])),
            ("target_output", int(admitted["target_output_scratch_capacity_bytes"])),
            ("source_weights", int(admitted["source_weights"])),
            ("source_annotation", int(admitted["source_annotation"])),
            ("target_mailman_output", int(admitted["target_mailman_output"])),
            ("mailman_worker", int(admitted["mailman_worker_scratch"])),
        )
        for name, bound in role_bounds:
            observed = role_capacity(name)
            if observed > bound:
                raise RuntimeError(
                    "Native scratch role exceeds its admitted capacity: "
                    f"{name} observed={observed}, admitted={bound}."
                )
        combined_feature = (
            role_capacity("feature_projected") + role_capacity("feature_scalar")
        )
        if combined_feature > int(admitted["feature_products_combined"]):
            raise RuntimeError(
                "Native packed feature scratch exceeds its admitted capacity: "
                f"observed={combined_feature}, "
                f"admitted={admitted['feature_products_combined']}."
            )

    def _replay_multi_environment_direct_calls(self, result: Mapping) -> None:
        """Reconstruct the former Python/C++ call boundaries without matrix transfer."""
        if (
            self._native_telemetry_consumer is None
            or self._native_gemm_output_numa_consumer is None
        ):
            raise RuntimeError(
                "The descriptor context requires both native evidence drains."
            )
        vendor_consumer = self._native_telemetry_consumer
        output_consumer = self._native_gemm_output_numa_consumer
        vendor_records = [dict(item) for item in vendor_consumer()]
        output_records = [dict(item) for item in output_consumer()]
        calls = [dict(item) for item in result.get("calls", ())]
        output_index = 0
        vendor_index = 0
        try:
            for call in calls:
                fallback = _json_safe(dict(call["fallback"]))
                semantic = _json_safe(dict(call["semantic"]))
                output_expected = call.get("output_expected") is True
                evidence = None
                if output_expected:
                    if output_index >= len(output_records):
                        raise RuntimeError(
                            "The descriptor context lost protected-output evidence."
                        )
                    evidence = output_records[output_index]
                    output_index += 1
                grouped_vendor: list[dict] = []
                while vendor_index < len(vendor_records):
                    candidate = vendor_records[vendor_index]
                    nested = candidate.get("native_gemm_output_numa")
                    if not isinstance(nested, Mapping):
                        raise RuntimeError(
                            "Native vendor telemetry lacks output discrimination."
                        )
                    applicable = nested.get("applicable") is True
                    if output_expected:
                        if not applicable:
                            break
                        if nested.get("call_id") != evidence.get("call_id"):
                            break
                    elif applicable:
                        break
                    grouped_vendor.append(candidate)
                    vendor_index += 1
                self._native_telemetry_consumer = (
                    lambda records=tuple(grouped_vendor): list(records)
                )
                self._native_gemm_output_numa_consumer = (
                    lambda record=evidence: [] if record is None else [record]
                )
                operation = str(fallback["operation"])
                m, n, k = (int(fallback[key]) for key in ("m", "n", "k"))
                if operation in {
                    "multi_source", "multi_source_mailman", "nn_update"
                }:
                    if operation != "multi_source_mailman":
                        self.nn_calls += 1
                        shape = ("nn", m, k, n)
                        self._gemm_shapes[shape] = (
                            self._gemm_shapes.get(shape, 0) + 1
                        )
                elif operation != "multi_feature_mailman" and operation != (
                    "multi_target_mailman"
                ):
                    self.tn_calls += 1
                    shape = ("tn", m, k, n)
                    self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
                self._record_gemm(
                    fallback,
                    semantic,
                    protected_output_expected=output_expected,
                )
                self.record_phase_elapsed(
                    str(semantic["phase"]),
                    time.perf_counter() - float(fallback["wall_seconds"]),
                    time.process_time() - float(
                        fallback["process_cpu_seconds"]
                    ),
                )
        finally:
            self._native_telemetry_consumer = vendor_consumer
            self._native_gemm_output_numa_consumer = output_consumer
        if output_index != len(output_records) or vendor_index != len(vendor_records):
            raise RuntimeError(
                "The descriptor context left unmatched native telemetry or output evidence."
            )

    @staticmethod
    def _native_kernel_fallback(
        operation: str,
        *,
        m: int,
        n: int,
        k: int,
        transpose_a: str,
        lda: int,
        ldb: int,
        ldc: int,
        wall_seconds: float,
        process_cpu_seconds: float,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> dict:
        return {
            "operation": operation,
            "arithmetic_dtype": "float64",
            "actual_left_storage_dtype": "float64",
            "actual_right_storage_dtype": "float64",
            "actual_output_storage_dtype": "float64",
            "layout": "column_major",
            "transpose_a": transpose_a,
            "transpose_b": "N",
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "lda": int(lda),
            "ldb": int(ldb),
            "ldc": int(ldc),
            "alpha": float(alpha),
            "beta": float(beta),
            "wall_seconds": float(wall_seconds),
            "process_cpu_seconds": float(process_cpu_seconds),
            "omp_in_parallel": None,
            "completed": True,
        }

    def native_feature_block(
        self,
        genotype: np.ndarray,
        *,
        eps_var: float,
        standardized: bool,
        semantic: Mapping | None = None,
    ) -> dict:
        if self._multi_environment_kernel is None:
            raise RuntimeError("The multi-environment native kernel is not initialized.")
        width = int(genotype.shape[1])
        feature_columns = int(
            self._multi_environment_kernel_info["feature_basis_columns"]
        )
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        result = dict(
            self._multi_environment_kernel.feature_block(
                np.asfortranarray(genotype, dtype=np.float64),
                float(eps_var),
                bool(standardized),
            )
        )
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        repaired = int(result.pop("repaired_gemm_output_columns"))
        self.repaired_output_columns += repaired
        self._multi_environment_kernel_repaired_columns += repaired
        self.tn_calls += 1
        shape = ("tn", feature_columns, int(genotype.shape[0]), width)
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        self._record_gemm(
            self._native_kernel_fallback(
                "multi_feature",
                m=feature_columns,
                n=width,
                k=int(genotype.shape[0]),
                transpose_a="T",
                lda=int(genotype.shape[0]),
                ldb=int(genotype.shape[0]),
                ldc=feature_columns,
                wall_seconds=wall,
                process_cpu_seconds=cpu,
            ),
            semantic,
            protected_output_expected=True,
        )
        self._multi_environment_kernel_call_counts["feature_calls"] += 1
        return {key: np.asarray(value) for key, value in result.items()}

    def native_source_block(
        self,
        target: np.ndarray,
        genotype: np.ndarray,
        probes: np.ndarray,
        annotation: np.ndarray,
        scale_x: np.ndarray,
        scale_w: np.ndarray,
        environment_start: int,
        *,
        semantic: Mapping | None = None,
    ) -> None:
        if self._multi_environment_kernel is None:
            raise RuntimeError("The multi-environment native kernel is not initialized.")
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        repaired = int(
            self._multi_environment_kernel.source_block(
                target,
                np.asfortranarray(genotype, dtype=np.float64),
                np.asfortranarray(probes, dtype=np.float64),
                np.asfortranarray(annotation, dtype=np.float64),
                np.asfortranarray(scale_x, dtype=np.float64),
                np.asfortranarray(scale_w, dtype=np.float64),
                int(environment_start),
            )
        )
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        self.repaired_output_columns += repaired
        self._multi_environment_kernel_repaired_columns += repaired
        m, k, n = int(genotype.shape[0]), int(genotype.shape[1]), int(target.shape[1])
        self.nn_calls += 1
        shape = ("nn", m, k, n)
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        self._record_gemm(
            self._native_kernel_fallback(
                "multi_source",
                m=m,
                n=n,
                k=k,
                transpose_a="N",
                lda=m,
                ldb=k,
                ldc=m,
                wall_seconds=wall,
                process_cpu_seconds=cpu,
            ),
            semantic,
            protected_output_expected=True,
        )
        self._multi_environment_kernel_call_counts["source_calls"] += 1

    def native_project_sources(
        self,
        panel: np.ndarray,
        columns_per_environment: int,
        environment_start: int,
        *,
        semantic: Mapping | None = None,
    ) -> list[float]:
        if self._multi_environment_kernel is None:
            raise RuntimeError("The multi-environment native kernel is not initialized.")
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        leakages = np.asarray(
            self._multi_environment_kernel.project_sources(
                panel, int(columns_per_environment), int(environment_start)
            ),
            dtype=np.float64,
        )
        observed_repairs = int(
            dict(self._multi_environment_kernel.info())[
                "repaired_gemm_output_columns"
            ]
        )
        repair_delta = (
            observed_repairs - self._multi_environment_kernel_repaired_columns
        )
        if repair_delta < 0:
            raise RuntimeError(
                "The multi-environment native repair counter moved backwards."
            )
        self.repaired_output_columns += repair_delta
        self._multi_environment_kernel_repaired_columns = observed_repairs
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        common_rank = int(self._multi_environment_kernel_info["common_basis_rank"])
        if common_rank:
            columns = int(panel.shape[1])
            self.tn_calls += 1
            self.nn_calls += 1
            tn_shape = ("tn", common_rank, int(panel.shape[0]), columns)
            nn_shape = ("nn", int(panel.shape[0]), common_rank, columns)
            self._gemm_shapes[tn_shape] = self._gemm_shapes.get(tn_shape, 0) + 1
            self._gemm_shapes[nn_shape] = self._gemm_shapes.get(nn_shape, 0) + 1
            self._record_gemm(
                self._native_kernel_fallback(
                    "multi_projection",
                    m=common_rank,
                    n=columns,
                    k=int(panel.shape[0]),
                    transpose_a="T",
                    lda=int(panel.shape[0]),
                    ldb=int(panel.shape[0]),
                    ldc=common_rank,
                    wall_seconds=wall,
                    process_cpu_seconds=cpu,
                ),
                semantic,
                protected_output_expected=True,
            )
            self._record_gemm(
                self._native_kernel_fallback(
                    "nn_update",
                    m=int(panel.shape[0]),
                    n=columns,
                    k=common_rank,
                    transpose_a="N",
                    lda=int(panel.shape[0]),
                    ldb=common_rank,
                    ldc=int(panel.shape[0]),
                    wall_seconds=0.0,
                    process_cpu_seconds=0.0,
                    alpha=-1.0,
                    beta=1.0,
                ),
                semantic,
            )
        self._multi_environment_kernel_call_counts["projection_calls"] += 1
        return [float(value) for value in leakages]

    def native_target_score_block(
        self,
        genotype: np.ndarray,
        protected_sources,
        scale_x: np.ndarray,
        scale_w: np.ndarray,
        accumulator_matrices: Mapping[str, np.ndarray],
        *,
        block_start: int,
        total_variants: int,
        probe_count: int,
        environment_start: int,
        semantic: Mapping | None = None,
    ) -> None:
        if self._multi_environment_kernel is None:
            raise RuntimeError("The multi-environment native kernel is not initialized.")
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        repaired_raw = self._multi_environment_kernel.target_score_block(
            np.asfortranarray(genotype, dtype=np.float64),
            protected_sources,
            np.asfortranarray(scale_x, dtype=np.float64),
            np.asfortranarray(scale_w, dtype=np.float64),
            accumulator_matrices["xx"],
            accumulator_matrices["xw"],
            accumulator_matrices["wx"],
            accumulator_matrices["ww"],
            int(block_start),
            int(total_variants),
            int(probe_count),
            int(environment_start),
        )
        if type(repaired_raw) is not int or repaired_raw < 0:
            raise RuntimeError(
                "The multi-environment target returned a malformed repair count."
            )
        repaired = int(repaired_raw)
        kernel_info = dict(self._multi_environment_kernel.info())
        integrity_keys = {
            "repaired_gemm_output_columns",
            "checksum_recomputed_gemm_output_columns",
            "roundoff_only_gemm_output_columns",
        }
        if any(
            type(kernel_info.get(key)) is not int or kernel_info[key] < 0
            for key in integrity_keys
        ):
            raise RuntimeError(
                "The multi-environment kernel returned malformed integrity counters."
            )
        repaired_total = int(kernel_info["repaired_gemm_output_columns"])
        checksum_recomputed_total = int(
            kernel_info["checksum_recomputed_gemm_output_columns"]
        )
        roundoff_only_total = int(
            kernel_info["roundoff_only_gemm_output_columns"]
        )
        if (
            repaired_total != self._multi_environment_kernel_repaired_columns + repaired
            or checksum_recomputed_total < (
                self._multi_environment_kernel_checksum_recomputed_columns
            )
            or roundoff_only_total < self._multi_environment_kernel_roundoff_only_columns
            or checksum_recomputed_total != repaired_total + roundoff_only_total
        ):
            raise RuntimeError(
                "The multi-environment target integrity counters do not reconcile."
            )
        wall = time.perf_counter() - wall_started
        cpu = time.process_time() - cpu_started
        self.repaired_output_columns += repaired
        self._multi_environment_kernel_repaired_columns = repaired_total
        self._multi_environment_kernel_checksum_recomputed_columns = (
            checksum_recomputed_total
        )
        self._multi_environment_kernel_roundoff_only_columns = roundoff_only_total
        m = int(genotype.shape[1])
        k = int(genotype.shape[0])
        n = 2 * int(protected_sources.columns)
        self.tn_calls += 1
        shape = ("tn", m, k, n)
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        self._record_gemm(
            self._native_kernel_fallback(
                "multi_target_score",
                m=m,
                n=n,
                k=k,
                transpose_a="T",
                lda=k,
                ldb=k,
                ldc=m,
                wall_seconds=wall,
                process_cpu_seconds=cpu,
            ),
            semantic,
            protected_output_expected=True,
        )
        self._multi_environment_kernel_call_counts["target_calls"] += 1

    def native_normalize_scores(
        self,
        accumulator_matrices: Mapping[str, np.ndarray],
        probe_count: int,
    ) -> None:
        if self._multi_environment_kernel is None:
            raise RuntimeError("The multi-environment native kernel is not initialized.")
        self._multi_environment_kernel.normalize_scores(
            accumulator_matrices["xx"],
            accumulator_matrices["xw"],
            accumulator_matrices["wx"],
            accumulator_matrices["ww"],
            int(probe_count),
        )
        self._multi_environment_kernel_call_counts["normalization_calls"] += 1

    def prepare_row_weighted_pair(
        self, right: np.ndarray, row_weights: np.ndarray
    ):
        """Seal one immutable [right, row_weight*right] target operand."""
        if not self.protected or self._module is None:
            raise RuntimeError(
                "Prepared row-weighted target pairs require the protected executor."
            )
        return self._module.prepare_protected_row_weighted_pair(
            np.asfortranarray(right, dtype=np.float64),
            np.asfortranarray(row_weights, dtype=np.float64),
            self.threads,
        )

    def tn_pair(
        self, left: np.ndarray, right_pair, *, semantic: Mapping | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multiply one left block by both halves of a sealed target pair."""
        if not self.protected or self._module is None:
            raise RuntimeError("Prepared target pairs require the protected executor.")
        columns = int(right_pair.columns)
        self.tn_calls += 1
        shape = ("tn", int(left.shape[1]), int(left.shape[0]), 2 * columns)
        self._gemm_shapes[shape] = self._gemm_shapes.get(shape, 0) + 1
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        result, repaired = self._module.protected_matmul_tn_pair(
            np.asfortranarray(left, dtype=np.float64), right_pair, self.threads
        )
        self.repaired_output_columns += int(repaired)
        combined = np.asarray(result)
        self._record_gemm(
            self._fallback_gemm_record(
                "tn_pair",
                left,
                2 * columns,
                np.float64,
                np.float64,
                combined.dtype,
                layout="column_major",
                transpose_a="T",
                ldb=left.shape[0],
                ldc=left.shape[1],
                alpha=1.0,
                beta=0.0,
                wall_seconds=time.perf_counter() - wall_started,
                process_cpu_seconds=time.process_time() - cpu_started,
            ),
            semantic,
            protected_output_expected=True,
        )
        return combined[:, :columns], combined[:, columns:]

    def gemm_shape_records(self) -> list[dict[str, int | str]]:
        """Return a bounded histogram using conventional C[m,n]=A[m,k]B[k,n]."""
        return [
            {
                "operation": operation,
                "m": m,
                "k": k,
                "n": n,
                "calls": calls,
                "flops": int(2 * m * k * n * calls),
            }
            for (operation, m, k, n), calls in sorted(self._gemm_shapes.items())
        ]


def _peak_process_rss_gib() -> float:
    """Return process-lifetime ru_maxrss in GiB on Linux and macOS."""
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_per_unit = 1 if sys.platform == "darwin" else 1024
    return maximum * bytes_per_unit / float(1024**3)


def _prepare_feature_block(
    estimator: GenomewideEnvLDScore,
    genotype: np.ndarray,
    start: int,
    stop: int,
    *,
    interaction: bool,
    apply_scale: bool,
    executor: _MultiEnvironmentGemm,
) -> np.ndarray:
    if not executor.protected:
        method = (
            estimator._prepare_interaction_block
            if interaction else estimator._prepare_additive_block
        )
        return method(
            start, stop, G=genotype, apply_scale=apply_scale,
            out_dtype=executor.compute_dtype,
        )

    if interaction:
        feature = np.asarray(
            genotype * estimator.env[:, None], dtype=np.float64, order="F"
        )
    else:
        feature = np.array(genotype, copy=True, dtype=np.float64, order="F")
    if estimator.p_eff > 0:
        coefficients = executor.tn(estimator.C_int, feature)
        feature -= executor.nn(estimator.C_int, coefficients)
    feature -= feature.mean(axis=0, keepdims=True)
    if apply_scale and estimator.kernel_mode == "standardized_projected":
        scales = (
            estimator.inv_sqrt_resvar_w_all
            if interaction else estimator.inv_sqrt_resvar_x_all
        )
        if scales is None:
            label = "Interaction" if interaction else "Additive"
            raise RuntimeError(f"{label} residual variances have not been precomputed.")
        feature *= scales[start:stop].reshape(1, -1)
    return np.asarray(feature, dtype=executor.compute_dtype, order="F")


def _new_feature_state(estimator: GenomewideEnvLDScore) -> dict:
    return {
        name: np.zeros(estimator.nsnps, dtype=np.float64)
        for name in (
            "inv_x", "inv_w", "norm_x", "norm_w", "diag_x", "diag_w",
            "corr_xw",
        )
    } | {"max_leak_x": 0.0, "max_leak_w": 0.0}


def _build_algebraic_feature_plan(
    estimators: Sequence[GenomewideEnvLDScore],
) -> dict:
    """Pack the exact one-environment feature-moment bases across environments."""
    first = estimators[0]
    environments = np.asfortranarray(
        np.column_stack([estimator.env for estimator in estimators]),
        dtype=np.float64,
    )
    ranks = [int(estimator.p_eff) + 1 for estimator in estimators]
    intercept = np.ones((first.nsamp, 1), dtype=np.float64)
    shared_basis = _orthonormalize_columns(
        np.column_stack([intercept, first.C_common])
    )
    shared_rank = int(shared_basis.shape[1])
    residual_bases: list[np.ndarray] = []
    for estimator, rank in zip(estimators, ranks, strict=True):
        residual = np.asarray(estimator.C_int, dtype=np.float64)
        if shared_rank:
            residual = residual - shared_basis @ (shared_basis.T @ residual)
        residual_basis = _orthonormalize_columns(residual)
        if shared_rank + residual_basis.shape[1] != rank:
            raise RuntimeError(
                f"Could not factor the complete projection basis for environment "
                f"{estimator.env_name!r}: shared rank {shared_rank}, residual rank "
                f"{residual_basis.shape[1]}, expected total {rank}."
            )
        residual_bases.append(np.asfortranarray(residual_basis, dtype=np.float64))
    # The power-zero intercept/common-covariate block is identical across all
    # environments. Store it once; environment-specific residual directions
    # and every row-weighted power remain independent.
    total_columns = (
        shared_rank
        + sum(basis.shape[1] for basis in residual_bases)
        + 3 * sum(ranks)
    )
    packed_basis = np.empty(
        (first.nsamp, total_columns), dtype=np.float64, order="F"
    )
    records: list[dict] = []
    packed_basis[:, :shared_rank] = shared_basis
    shared_indices = np.arange(shared_rank, dtype=np.int64)
    offset = shared_rank
    power_zero_indices: list[np.ndarray] = []
    full_bases: list[np.ndarray] = []
    for estimator, rank, residual_basis in zip(
        estimators, ranks, residual_bases, strict=True
    ):
        residual_rank = int(residual_basis.shape[1])
        residual_indices = np.arange(offset, offset + residual_rank, dtype=np.int64)
        if residual_rank:
            packed_basis[:, offset:offset + residual_rank] = residual_basis
        offset += residual_rank
        power_zero_indices.append(
            np.concatenate([shared_indices, residual_indices])
        )
        basis = np.asfortranarray(
            np.column_stack([shared_basis, residual_basis]), dtype=np.float64
        )
        if basis.shape != (first.nsamp, rank):
            raise RuntimeError(
                "Complete fused GxE projection basis lost rank for environment "
                f"{estimator.env_name!r}: expected {(first.nsamp, rank)}, "
                f"got {basis.shape}."
            )
        full_bases.append(basis)

    for index, (estimator, basis, rank) in enumerate(
        zip(estimators, full_bases, ranks, strict=True)
    ):
        environment = environments[:, index]
        environment_squared = environment * environment
        gram = np.asarray(basis.T @ basis, dtype=np.float64)
        gram_error = float(np.max(np.abs(gram - np.eye(rank))))
        if gram_error > 1.0e-10:
            raise RuntimeError(
                "Fused GxE projection basis is not orthonormal for environment "
                f"{estimator.env_name!r}: max Gram error={gram_error:.6g}."
            )
        environment_squared_gram = np.asarray(
            basis.T @ (environment_squared[:, None] * basis),
            dtype=np.float64,
        )
        multiplier = np.ones(first.nsamp, dtype=np.float64)
        power_indices = [power_zero_indices[index]]
        multiplier *= environment
        for _power in range(1, 4):
            packed_basis[:, offset:offset + rank] = basis * multiplier[:, None]
            power_indices.append(np.arange(offset, offset + rank, dtype=np.int64))
            multiplier *= environment
            offset += rank
        records.append(
            {
                "rank": rank,
                "power_indices": power_indices,
                "gram": gram,
                "environment_squared_gram": environment_squared_gram,
            }
        )
    if offset != total_columns:
        raise RuntimeError("Internal fused GxE feature-basis packing failure.")
    return {
        "environments": environments,
        "basis": packed_basis,
        "records": records,
        "shared_power_zero_rank": shared_rank,
    }


def _pack_native_feature_plan(
    plan: Mapping,
    estimators: Sequence[GenomewideEnvLDScore],
) -> dict[str, np.ndarray]:
    """Pack one validated feature plan for either native execution owner."""
    records = list(plan["records"])
    if len(records) != len(estimators):
        raise RuntimeError(
            "The multi-environment feature plan does not match the estimator count."
        )
    power_indices: list[int] = []
    power_offsets = [0]
    ranks: list[int] = []
    gram_values: list[float] = []
    gram_offsets = [0]
    e2_gram_values: list[float] = []
    for record in records:
        rank = int(record["rank"])
        ranks.append(rank)
        indices = list(record["power_indices"])
        if len(indices) != 4:
            raise RuntimeError(
                "The multi-environment feature plan lacks four power bases."
            )
        for values in indices:
            observed = np.asarray(values, dtype=np.int64)
            if observed.shape != (rank,):
                raise RuntimeError(
                    "The multi-environment power-basis index vector is malformed."
                )
            power_indices.extend(int(value) for value in observed)
            power_offsets.append(len(power_indices))
        gram = np.asfortranarray(record["gram"], dtype=np.float64)
        e2_gram = np.asfortranarray(
            record["environment_squared_gram"], dtype=np.float64
        )
        if gram.shape != (rank, rank) or e2_gram.shape != (rank, rank):
            raise RuntimeError(
                "The multi-environment feature Gram matrix is malformed."
            )
        gram_values.extend(float(value) for value in gram.ravel(order="F"))
        e2_gram_values.extend(
            float(value) for value in e2_gram.ravel(order="F")
        )
        gram_offsets.append(len(gram_values))
    return {
        "feature_power_indices": np.asarray(power_indices, dtype=np.int64),
        "feature_power_offsets": np.asarray(power_offsets, dtype=np.int64),
        "feature_ranks": np.asarray(ranks, dtype=np.int32),
        "feature_gram_values": np.asarray(gram_values, dtype=np.float64),
        "feature_gram_offsets": np.asarray(gram_offsets, dtype=np.int64),
        "feature_e2_gram_values": np.asarray(e2_gram_values, dtype=np.float64),
    }


def _accumulate_algebraic_feature_block(
    estimators: Sequence[GenomewideEnvLDScore],
    states: Sequence[dict],
    genotype: np.ndarray,
    start: int,
    stop: int,
    plan: dict,
    executor: _MultiEnvironmentGemm,
) -> None:
    """Compute exact projected feature moments without materializing X or W."""
    scalar = executor.fused_feature_scalar_moments(
        genotype, plan["environments"]
    )
    projected = executor.tn(plan["basis"], genotype)
    width = stop - start
    expected_scalar_shape = (1 + 3 * len(estimators), width)
    if scalar.shape != expected_scalar_shape:
        raise RuntimeError(
            f"Fused feature scalar shape {scalar.shape} does not match "
            f"{expected_scalar_shape}."
        )
    for index, (estimator, state, record) in enumerate(
        zip(estimators, states, plan["records"], strict=True)
    ):
        rank = int(record["rank"])
        indices = record["power_indices"]
        u0 = projected[indices[0]]
        u1 = projected[indices[1]]
        u2 = projected[indices[2]]
        u3 = projected[indices[3]]
        s0 = scalar[0]
        s1 = scalar[1 + 3 * index]
        s2 = scalar[2 + 3 * index]
        s4 = scalar[3 + 3 * index]

        u00 = np.einsum("ij,ij->j", u0, u0, optimize=False)
        u11 = np.einsum("ij,ij->j", u1, u1, optimize=False)
        u01 = np.einsum("ij,ij->j", u0, u1, optimize=False)
        u0u2 = np.einsum("ij,ij->j", u0, u2, optimize=False)
        u1u3 = np.einsum("ij,ij->j", u1, u3, optimize=False)
        e2_u0 = record["environment_squared_gram"] @ u0
        e2_u1 = record["environment_squared_gram"] @ u1
        u0e2u0 = np.einsum("ij,ij->j", u0, e2_u0, optimize=False)
        u1e2u1 = np.einsum("ij,ij->j", u1, e2_u1, optimize=False)

        ssx = s0 - u00
        ssw = s2 - u11
        varx = ssx / float(estimator.df_corr)
        varw = ssw / float(estimator.df_corr)
        good_x = np.isfinite(varx) & (varx > estimator.eps_var)
        good_w = np.isfinite(varw) & (varw > estimator.eps_var)
        invalid = ~(good_x & good_w)
        if np.any(invalid):
            examples = []
            for block_offset in np.flatnonzero(invalid)[:5]:
                variant_index = start + int(block_offset)
                snp = (
                    str(estimator.snplist.iloc[variant_index]["SNP"])
                    if estimator.snplist is not None else str(variant_index)
                )
                failed = []
                if not good_x[block_offset]:
                    failed.append("additive")
                if not good_w[block_offset]:
                    failed.append("interaction")
                examples.append(f"{snp} ({'+'.join(failed)})")
            raise ValueError(
                f"{int(invalid.sum())} variant feature(s) in block "
                f"[{start}:{stop}) have zero or invalid projected variance for "
                f"environment {estimator.env_name!r}. QC/remove these variants "
                f"and regenerate the complete batch. Examples: {', '.join(examples)}."
            )

        if estimator.kernel_mode == "standardized_projected":
            scale_x = 1.0 / np.sqrt(varx)
            scale_w = 1.0 / np.sqrt(varw)
        else:
            scale_x = np.ones_like(varx)
            scale_w = np.ones_like(varw)
        state["inv_x"][start:stop] = scale_x
        state["inv_w"][start:stop] = scale_w
        state["norm_x"][start:stop] = (
            scale_x * scale_x * ssx / float(estimator.df_corr)
        )
        state["norm_w"][start:stop] = (
            scale_w * scale_w * ssw / float(estimator.df_corr)
        )
        state["diag_x"][start:stop] = (
            scale_x * scale_x * (s2 - 2.0 * u0u2 + u0e2u0)
            / float(estimator.df_corr)
        )
        state["diag_w"][start:stop] = (
            scale_w * scale_w * (s4 - 2.0 * u1u3 + u1e2u1)
            / float(estimator.df_corr)
        )
        state["corr_xw"][start:stop] = (
            scale_x * scale_w * (s1 - u01) / float(estimator.df_corr)
        )

        gram_u0 = record["gram"] @ u0
        gram_u1 = record["gram"] @ u1
        leak_x_sq = np.sum((u0 - gram_u0) ** 2, axis=0, dtype=np.float64)
        leak_w_sq = np.sum((u1 - gram_u1) ** 2, axis=0, dtype=np.float64)
        state["max_leak_x"] = max(
            float(state["max_leak_x"]),
            float(np.max(np.sqrt(np.maximum(0.0, leak_x_sq) / ssx))),
        )
        state["max_leak_w"] = max(
            float(state["max_leak_w"]),
            float(np.max(np.sqrt(np.maximum(0.0, leak_w_sq) / ssw))),
        )
        for name in (
            "inv_x", "inv_w", "norm_x", "norm_w", "diag_x", "diag_w",
            "corr_xw",
        ):
            if not np.all(np.isfinite(state[name][start:stop])):
                raise RuntimeError(
                    f"Fused GxE feature moment {name!r} contains NaN or infinity "
                    f"for environment {estimator.env_name!r}."
                )


def _accumulate_native_feature_block(
    estimators: Sequence[GenomewideEnvLDScore],
    states: Sequence[dict],
    genotype: np.ndarray,
    start: int,
    stop: int,
    executor: _MultiEnvironmentGemm,
    semantic: Mapping | None = None,
) -> None:
    """Store one block of feature invariants computed by the native kernel."""
    if not estimators:
        raise RuntimeError("The native feature block has no environments.")
    standardized = estimators[0].kernel_mode == "standardized_projected"
    if any(
        (estimator.kernel_mode == "standardized_projected") != standardized
        for estimator in estimators
    ):
        raise RuntimeError(
            "Multi-environment native feature kernels disagree on standardization."
        )
    eps_var = float(estimators[0].eps_var)
    if any(float(estimator.eps_var) != eps_var for estimator in estimators):
        raise RuntimeError(
            "Multi-environment native feature kernels disagree on eps_var."
        )
    result = executor.native_feature_block(
        genotype,
        eps_var=eps_var,
        standardized=standardized,
        semantic=semantic,
    )
    width = stop - start
    matrix_fields = {
        "scale_x": "inv_x",
        "scale_w": "inv_w",
        "norm_x": "norm_x",
        "norm_w": "norm_w",
        "diag_nxe_x": "diag_x",
        "diag_nxe_w": "diag_w",
        "corr_xw": "corr_xw",
    }
    expected_shape = (width, len(estimators))
    for native_name, state_name in matrix_fields.items():
        values = np.asarray(result.get(native_name), dtype=np.float64)
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Native multi-environment feature field {native_name!r} has "
                f"shape {values.shape}, expected {expected_shape}, or is non-finite."
            )
        for index, state in enumerate(states):
            state[state_name][start:stop] = values[:, index]
    leakage_fields = {
        "max_projection_leakage_additive": "max_leak_x",
        "max_projection_leakage_interaction": "max_leak_w",
    }
    for native_name, state_name in leakage_fields.items():
        values = np.asarray(result.get(native_name), dtype=np.float64)
        if values.shape != (len(estimators),) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Native multi-environment feature field {native_name!r} is malformed."
            )
        for index, state in enumerate(states):
            state[state_name] = max(float(state[state_name]), float(values[index]))


def _accumulate_feature_block(
    estimator: GenomewideEnvLDScore,
    state: dict,
    genotype: np.ndarray,
    start: int,
    stop: int,
    executor: _MultiEnvironmentGemm,
) -> None:
    """Consume one shared standardized genotype block for one environment."""
    x = _prepare_feature_block(
        estimator, genotype, start, stop,
        interaction=False, apply_scale=False, executor=executor,
    )
    w = _prepare_feature_block(
        estimator, genotype, start, stop,
        interaction=True, apply_scale=False, executor=executor,
    )
    _accumulate_prepared_feature_block(estimator, state, x, w, start, stop)


def _accumulate_prepared_feature_block(
    estimator: GenomewideEnvLDScore,
    state: dict,
    x: np.ndarray,
    w: np.ndarray,
    start: int,
    stop: int,
) -> None:
    """Accumulate invariants from already projected X/W feature blocks."""
    ssx = np.sum(x * x, axis=0, dtype=np.float64)
    ssw = np.sum(w * w, axis=0, dtype=np.float64)
    varx = ssx / float(estimator.df_corr)
    varw = ssw / float(estimator.df_corr)
    good_x = np.isfinite(varx) & (varx > estimator.eps_var)
    good_w = np.isfinite(varw) & (varw > estimator.eps_var)
    invalid = ~(good_x & good_w)
    if np.any(invalid):
        examples = []
        for offset in np.flatnonzero(invalid)[:5]:
            index = start + int(offset)
            snp = (
                str(estimator.snplist.iloc[index]["SNP"])
                if estimator.snplist is not None else str(index)
            )
            failed = []
            if not good_x[offset]:
                failed.append("additive")
            if not good_w[offset]:
                failed.append("interaction")
            examples.append(f"{snp} ({'+'.join(failed)})")
        raise ValueError(
            f"{int(invalid.sum())} variant feature(s) in block [{start}:{stop}) "
            f"have zero or invalid projected variance for environment "
            f"{estimator.env_name!r}. QC/remove these variants and regenerate the "
            f"complete batch. Examples: {', '.join(examples)}."
        )

    if estimator.kernel_mode == "standardized_projected":
        state["inv_x"][start:stop] = 1.0 / np.sqrt(varx)
        state["inv_w"][start:stop] = 1.0 / np.sqrt(varw)
    else:
        state["inv_x"][start:stop] = 1.0
        state["inv_w"][start:stop] = 1.0
    x *= state["inv_x"][start:stop].reshape(1, -1)
    w *= state["inv_w"][start:stop].reshape(1, -1)

    final_ssx = np.sum(x * x, axis=0, dtype=np.float64)
    final_ssw = np.sum(w * w, axis=0, dtype=np.float64)
    state["norm_x"][start:stop] = final_ssx / float(estimator.df_corr)
    state["norm_w"][start:stop] = final_ssw / float(estimator.df_corr)

    root_n = math.sqrt(float(estimator.nsamp))
    leaked_x_sq = (x.sum(axis=0, dtype=np.float64) / root_n) ** 2
    leaked_w_sq = (w.sum(axis=0, dtype=np.float64) / root_n) ** 2
    if estimator.p_eff > 0:
        # These small fixed-effect diagnostics deliberately avoid dispatching
        # one protected GEMM per environment.  The packed construction below
        # already applies the projector; this independent contraction checks
        # its numerical leakage without defeating GEMM fusion.
        projected_x = np.einsum(
            "ni,nj->ij", estimator.C_int, x, optimize=False
        )
        projected_w = np.einsum(
            "ni,nj->ij", estimator.C_int, w, optimize=False
        )
        leaked_x_sq += np.sum(projected_x * projected_x, axis=0, dtype=np.float64)
        leaked_w_sq += np.sum(projected_w * projected_w, axis=0, dtype=np.float64)
    state["max_leak_x"] = max(
        float(state["max_leak_x"]),
        float(np.max(np.sqrt(leaked_x_sq / final_ssx))),
    )
    state["max_leak_w"] = max(
        float(state["max_leak_w"]),
        float(np.max(np.sqrt(leaked_w_sq / final_ssw))),
    )

    ex = estimator.env[:, None] * x
    ew = estimator.env[:, None] * w
    state["diag_x"][start:stop] = (
        np.sum(ex * ex, axis=0, dtype=np.float64) / float(estimator.df_corr)
    )
    state["diag_w"][start:stop] = (
        np.sum(ew * ew, axis=0, dtype=np.float64) / float(estimator.df_corr)
    )
    state["corr_xw"][start:stop] = (
        np.sum(x * w, axis=0, dtype=np.float64) / float(estimator.df_corr)
    )


def _environment_directions(
    estimators: Sequence[GenomewideEnvLDScore],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shared covariate basis and one residual direction per env."""
    common = np.asfortranarray(estimators[0].C_common, dtype=np.float64)
    directions = np.zeros((estimators[0].nsamp, len(estimators)), dtype=np.float64)
    # Use the process-lifetime configuration installed before runtime
    # initialization. Repeated BLAS pool resizing was the unsafe handoff.
    for index, estimator in enumerate(estimators):
        observed = np.asarray(estimator.C_common, dtype=np.float64)
        if observed.shape != common.shape:
            raise ValueError(
                "Multi-environment estimators disagree on common covariate rank."
            )
        if common.shape[1] and not np.allclose(
            np.abs(common.T @ observed),
            np.eye(common.shape[1]),
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise ValueError(
                "Multi-environment estimators disagree on the common covariate span."
            )
        direction = np.asarray(estimator.env, dtype=np.float64).copy()
        direction -= direction.mean()
        if common.shape[1]:
            direction -= common @ (common.T @ direction)
        norm = float(np.linalg.norm(direction))
        tolerance = 2.0e-10 * max(1.0, float(np.linalg.norm(estimator.env)))
        columns = [common]
        if norm > tolerance:
            direction /= norm
            directions[:, index] = direction
            columns.append(direction.reshape(-1, 1))
        reconstructed = np.column_stack(columns)
        # Compare the two spans without forming an N-by-N projector.
        residual_full = estimator.C_int - reconstructed @ (
            reconstructed.T @ estimator.C_int
        )
        residual_reconstructed = reconstructed - estimator.C_int @ (
            estimator.C_int.T @ reconstructed
        )
        residual_norm = max(
            float(np.linalg.norm(residual_full)),
            float(np.linalg.norm(residual_reconstructed)),
        )
        span_tolerance = 2.0e-9 * max(
            1.0, math.sqrt(float(estimator.p_eff))
        )
        if residual_norm > span_tolerance:
            raise RuntimeError(
                f"Could not factor the fixed-effect projector for environment "
                f"{estimator.env_name!r} into common covariates plus one "
                "environment direction: residual norm "
                f"{residual_norm:.6g} exceeds {span_tolerance:.6g}."
            )
    return common, np.asfortranarray(directions)


def _accumulate_fused_feature_moments(
    estimator: GenomewideEnvLDScore,
    state: dict,
    common_x: np.ndarray,
    common_sum: np.ndarray,
    common_ss: np.ndarray,
    direction: np.ndarray,
    direction_coefficient: np.ndarray,
    direction_sum: float,
    direction_fixed: np.ndarray,
    environment_squared: np.ndarray,
    direction_environment_squared_norm: float,
    w: np.ndarray,
    start: int,
    stop: int,
) -> None:
    """Accumulate X moments algebraically and W moments from one streamed block."""
    ux = np.einsum("n,nj->j", direction, common_x, optimize=False)
    ssx = common_ss - 2.0 * direction_coefficient * ux
    ssx += direction_coefficient * direction_coefficient
    # ``einsum`` contracts each Fortran-contiguous column without allocating
    # an N-by-block temporary.  ``np.sum(w * w, axis=0)`` was a material peak-
    # memory cost for production-sized feature blocks.
    ssw = np.einsum("ij,ij->j", w, w, optimize=False)
    varx = ssx / float(estimator.df_corr)
    varw = ssw / float(estimator.df_corr)
    good_x = np.isfinite(varx) & (varx > estimator.eps_var)
    good_w = np.isfinite(varw) & (varw > estimator.eps_var)
    invalid = ~(good_x & good_w)
    if np.any(invalid):
        examples = []
        for offset in np.flatnonzero(invalid)[:5]:
            index = start + int(offset)
            snp = (
                str(estimator.snplist.iloc[index]["SNP"])
                if estimator.snplist is not None else str(index)
            )
            failed = []
            if not good_x[offset]:
                failed.append("additive")
            if not good_w[offset]:
                failed.append("interaction")
            examples.append(f"{snp} ({'+'.join(failed)})")
        raise ValueError(
            f"{int(invalid.sum())} variant feature(s) in block [{start}:{stop}) "
            f"have zero or invalid projected variance for environment "
            f"{estimator.env_name!r}. QC/remove these variants and regenerate the "
            f"complete batch. Examples: {', '.join(examples)}."
        )

    if estimator.kernel_mode == "standardized_projected":
        inv_x = 1.0 / np.sqrt(varx)
        inv_w = 1.0 / np.sqrt(varw)
    else:
        inv_x = np.ones_like(varx)
        inv_w = np.ones_like(varw)
    state["inv_x"][start:stop] = inv_x
    state["inv_w"][start:stop] = inv_w
    w *= inv_w.reshape(1, -1)
    final_ssx = ssx * inv_x * inv_x
    final_ssw = np.einsum("ij,ij->j", w, w, optimize=False)
    state["norm_x"][start:stop] = final_ssx / float(estimator.df_corr)
    state["norm_w"][start:stop] = final_ssw / float(estimator.df_corr)

    x_sum = (common_sum - direction_sum * direction_coefficient) * inv_x
    root_n = math.sqrt(float(estimator.nsamp))
    leaked_x_sq = (x_sum / root_n) ** 2
    leaked_w_sq = (w.sum(axis=0, dtype=np.float64) / root_n) ** 2
    if estimator.p_eff:
        common_fixed = np.einsum(
            "ni,nj->ij", estimator.C_int, common_x, optimize=False
        )
        fixed_x = (
            common_fixed
            - direction_fixed[:, None] * direction_coefficient.reshape(1, -1)
        ) * inv_x.reshape(1, -1)
        fixed_w = np.einsum(
            "ni,nj->ij", estimator.C_int, w, optimize=False
        )
        leaked_x_sq += np.sum(fixed_x * fixed_x, axis=0, dtype=np.float64)
        leaked_w_sq += np.sum(fixed_w * fixed_w, axis=0, dtype=np.float64)
    state["max_leak_x"] = max(
        float(state["max_leak_x"]),
        float(np.max(np.sqrt(leaked_x_sq / final_ssx))),
    )
    state["max_leak_w"] = max(
        float(state["max_leak_w"]),
        float(np.max(np.sqrt(leaked_w_sq / final_ssw))),
    )

    diagonal_x = np.einsum(
        "n,nj,nj->j",
        environment_squared,
        common_x,
        common_x,
        optimize=False,
    )
    diagonal_x -= 2.0 * direction_coefficient * np.einsum(
        "n,n,nj->j",
        direction,
        environment_squared,
        common_x,
        optimize=False,
    )
    diagonal_x += (
        direction_coefficient
        * direction_coefficient
        * direction_environment_squared_norm
    )
    state["diag_x"][start:stop] = (
        diagonal_x * inv_x * inv_x / float(estimator.df_corr)
    )
    state["diag_w"][start:stop] = (
        np.einsum(
            "n,nj,nj->j", environment_squared, w, w, optimize=False
        )
        / float(estimator.df_corr)
    )
    cross = np.einsum("ij,ij->j", common_x, w, optimize=False)
    cross -= direction_coefficient * np.einsum(
        "n,nj->j", direction, w, optimize=False
    )
    state["corr_xw"][start:stop] = (
        inv_x * cross / float(estimator.df_corr)
    )


def _accumulate_fused_feature_block(
    estimators: Sequence[GenomewideEnvLDScore],
    states: Sequence[dict],
    genotype: np.ndarray,
    start: int,
    stop: int,
    environment_start: int,
    environment_stop: int,
    common: np.ndarray,
    directions: np.ndarray,
    environment_constants: Sequence[tuple[float, np.ndarray, np.ndarray, float]],
    executor: _MultiEnvironmentGemm,
) -> None:
    """Project one block for an environment tile with a bounded GEMM count."""
    genotype = np.asarray(genotype, dtype=executor.compute_dtype, order="F")
    width = stop - start
    common_x = np.array(genotype, copy=True, dtype=executor.compute_dtype, order="F")
    common_rank = common.shape[1]
    if common_rank:
        coefficients = executor.tn(common, common_x)
        common_x -= executor.nn(common, coefficients)
    common_x -= common_x.mean(axis=0, keepdims=True)
    common_sum = common_x.sum(axis=0, dtype=np.float64)
    common_ss = np.einsum("ij,ij->j", common_x, common_x, optimize=False)

    tile_directions = directions[:, environment_start:environment_stop]
    direction_coefficients = executor.tn(tile_directions, genotype)

    interaction_rows: list[np.ndarray] = []
    for local_index, estimator in enumerate(
        estimators[environment_start:environment_stop]
    ):
        if common_rank:
            interaction_rows.append(common * estimator.env[:, None])
        interaction_rows.append(
            (tile_directions[:, local_index] * estimator.env).reshape(-1, 1)
        )
    packed_rows = np.asfortranarray(np.column_stack(interaction_rows))
    packed_coefficients = executor.tn(packed_rows, genotype)

    row_offset = 0
    for local_index, estimator in enumerate(
        estimators[environment_start:environment_stop]
    ):
        direction = tile_directions[:, local_index]
        w = np.asarray(
            genotype * estimator.env[:, None],
            dtype=executor.compute_dtype,
            order="F",
        )
        w -= w.mean(axis=0, keepdims=True)
        if common_rank:
            executor.nn_update(
                common,
                packed_coefficients[row_offset:row_offset + common_rank],
                w,
            )
            row_offset += common_rank
        w -= direction[:, None] * packed_coefficients[row_offset]
        row_offset += 1
        global_index = environment_start + local_index
        (
            direction_sum,
            direction_fixed,
            environment_squared,
            direction_environment_squared_norm,
        ) = environment_constants[global_index]
        _accumulate_fused_feature_moments(
            estimator,
            states[global_index],
            common_x,
            common_sum,
            common_ss,
            direction,
            direction_coefficients[local_index],
            direction_sum,
            direction_fixed,
            environment_squared,
            direction_environment_squared_norm,
            w,
            start,
            stop,
        )


def _finish_feature_state(estimator: GenomewideEnvLDScore, state: dict) -> None:
    estimator.inv_sqrt_resvar_x_all = state["inv_x"]
    estimator.inv_sqrt_resvar_w_all = state["inv_w"]
    estimator.norm_x_all = state["norm_x"]
    estimator.norm_w_all = state["norm_w"]
    estimator.diag_nxe_x_all = state["diag_x"]
    estimator.diag_nxe_w_all = state["diag_w"]
    estimator.corr_xw_all = state["corr_xw"]
    estimator.score_x_all = None
    estimator.score_w_all = None

    annotations = np.asarray(estimator.annot, dtype=np.float64)
    trace_x = (
        float(estimator.df_corr)
        * (annotations.T @ estimator.norm_x_all)
        / estimator.nsnps_bin
    )
    trace_w = (
        float(estimator.df_corr)
        * (annotations.T @ estimator.norm_w_all)
        / estimator.nsnps_bin
    )
    max_norm_x = float(np.max(np.abs(estimator.norm_x_all - 1.0)))
    max_norm_w = float(np.max(np.abs(estimator.norm_w_all - 1.0)))
    max_leak_x = float(state["max_leak_x"])
    max_leak_w = float(state["max_leak_w"])
    if max(max_leak_x, max_leak_w) > 1.0e-9:
        raise RuntimeError(
            f"Projected features for environment {estimator.env_name!r} leak into "
            f"the fixed-effect span: X={max_leak_x:.6g}, W={max_leak_w:.6g}."
        )
    if (
        estimator.kernel_mode == "standardized_projected"
        and max(max_norm_x, max_norm_w) > 1.0e-9
    ):
        raise RuntimeError(
            f"Post-projection normalization failed for environment "
            f"{estimator.env_name!r}: X={max_norm_x:.6g}, W={max_norm_w:.6g}."
        )
    estimator.feature_diagnostics = {
        "valid_additive_columns": int(estimator.nsnps),
        "valid_interaction_columns": int(estimator.nsnps),
        "max_projection_leakage_additive": max_leak_x,
        "max_projection_leakage_interaction": max_leak_w,
        "min_norm_additive_over_rank": float(np.min(estimator.norm_x_all)),
        "max_norm_additive_over_rank": float(np.max(estimator.norm_x_all)),
        "min_norm_interaction_over_rank": float(np.min(estimator.norm_w_all)),
        "max_norm_interaction_over_rank": float(np.max(estimator.norm_w_all)),
        "max_norm_error_additive": max_norm_x,
        "max_norm_error_interaction": max_norm_w,
        "kernel_traces_additive": trace_x.tolist(),
        "kernel_traces_interaction": trace_w.tolist(),
        "max_trace_error_additive": float(
            np.max(np.abs(trace_x - estimator.df_corr))
        ),
        "max_trace_error_interaction": float(
            np.max(np.abs(trace_w - estimator.df_corr))
        ),
    }
    estimator.feature_diagnostics.update(
        estimator._missing_genotype_diagnostics()
    )
    estimator.log._log(
        f"[gxe:multi:invariants:{estimator.env_name}] max fixed-effect leakage "
        f"X={max_leak_x:.3e}, W={max_leak_w:.3e}; max norm error "
        f"X={max_norm_x:.3e}, W={max_norm_w:.3e}."
    )


def _require_common_contract(estimators: Sequence[GenomewideEnvLDScore]) -> None:
    if not estimators:
        raise ValueError("GxE reference construction requires at least one environment.")
    # ``overwrite`` is part of the common fused contract.  Fused batch
    # publication has no backup/journal/restore transaction: replacing an
    # existing bundle with os.replace loses the original inode, and a later
    # failure makes the outer rollback delete the replacement without being
    # able to restore the original.  Until a real transaction exists, fused
    # publication refuses overwrite entirely — before any byte changes.
    overwrite_settings = {bool(estimator.overwrite) for estimator in estimators}
    if len(overwrite_settings) > 1:
        raise ValueError(
            "Multi-environment estimators disagree on overwrite; fused "
            "publication requires one common overwrite setting."
        )
    if overwrite_settings == {True}:
        raise ValueError(
            "Fused multi-environment publication cannot overwrite existing "
            "references: a failed batch would roll back a replaced bundle "
            "without restoring the original bytes. Publish to a fresh "
            "prefix, or overwrite one environment at a time with the "
            "single-environment generator."
        )
    first = estimators[0]
    if first.genotype_format != "bed":
        raise ValueError("Shared multi-environment construction currently requires BED input.")
    first_placement = getattr(first, "cpu_placement", None)
    first_placement_complete = getattr(first, "cpu_placement_complete", False)
    first_placement_authenticated = getattr(
        first, "_gxe_group_worker_authenticated", False
    )
    if first_placement is None:
        if (
            first_placement_complete is not False
            or first_placement_authenticated is not False
        ):
            raise ValueError("Authenticated estimator lacks CPU placement evidence.")
    else:
        if (
            first_placement_complete is not True
            or first_placement_authenticated is not True
        ):
            raise ValueError("Estimator CPU placement is not authenticated and complete.")
        first_placement = _validate_cpu_placement_attestation(
            first_placement, expected_threads=int(first.num_threads)
        )
    scalar_fields = (
        "nsamp_total", "nsamp", "nsnps", "nbins", "nvecs", "step_size",
        "step_size_selection",
        "root_seed", "probe_offset", "rand_dist", "ddof", "kernel_mode",
        "genotype_scale", "eps_var", "dtype", "target_xz_mem",
        "gxe_total_memory_request", "native_workspace_gib",
    )
    for estimator in estimators:
        if any(
            hasattr(estimator, name)
            for name in (
                "_native_numa_bound_decode_nodes",
                "_native_numa_bound_iid_index",
                "_native_numa_bound_decode_records",
            )
        ):
            raise ValueError(
                "NUMA-bound BED decode attributes are internal to an "
                "authenticated protected multi-environment execution."
            )
        placement = getattr(estimator, "cpu_placement", None)
        placement_complete = getattr(estimator, "cpu_placement_complete", False)
        placement_authenticated = getattr(
            estimator, "_gxe_group_worker_authenticated", False
        )
        if (
            placement_complete is not first_placement_complete
            or placement_authenticated is not first_placement_authenticated
            or placement != first_placement
        ):
            raise ValueError(
                "Multi-environment estimators disagree on CPU placement evidence."
            )
        if estimator.native_backend != "python":
            raise ValueError(
                "Multi-environment construction owns one shared decoded genotype "
                "stream and therefore requires Python-orchestrated BLAS estimators."
            )
        if estimator.pheno is not None:
            raise ValueError("Multi-environment reference construction is phenotype-free.")
        if not np.array_equal(first.row_sel, estimator.row_sel):
            raise ValueError(
                "Selected environments do not retain exactly the same complete-case "
                "cohort. Intersect the input rows explicitly or create separate batches."
            )
        if estimator._construction_genotype_state != first._construction_genotype_state:
            raise ValueError(
                "Multi-environment estimators must refer to the same opened genotype files."
            )
        if not np.array_equal(
            first.sample_ids[["FID", "IID"]].to_numpy(),
            estimator.sample_ids[["FID", "IID"]].to_numpy(),
        ):
            raise ValueError("Multi-environment estimators disagree on genotype sample IDs.")
        for field in scalar_fields:
            if getattr(estimator, field) != getattr(first, field):
                raise ValueError(f"Multi-environment estimators disagree on {field}.")
        for name in ("annot", "nsnps_bin"):
            left = getattr(first, name)
            right = getattr(estimator, name)
            if left is None or right is None:
                if left is not right:
                    raise ValueError(f"Multi-environment estimators disagree on {name}.")
            elif not np.array_equal(left, right):
                raise ValueError(f"Multi-environment estimators disagree on {name}.")
        if estimator.l2cols != first.l2cols:
            raise ValueError("Multi-environment estimators disagree on annotation names.")
    names = [estimator.env_name for estimator in estimators]
    if len(set(names)) != len(names):
        raise ValueError(f"Multi-environment column names must be unique: {names}.")
    suffixes = [safe_environment_suffix(name) for name in names]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(
            f"Environment names collide after filename normalization: {names}."
        )


def _enable_contracted_numa_bound_decode(
    estimator: GenomewideEnvLDScore,
    executor: _MultiEnvironmentGemm,
) -> tuple[int, ...] | None:
    """Enable dedicated decode only at the authenticated FP64 boundary."""
    if not executor._numa_bound_bed_decode_required:
        return None
    if (
        executor.cpu_placement is None
        or executor.cpu_placement_complete is not True
        or getattr(estimator, "_gxe_group_worker_authenticated", False) is not True
        or not executor.protected
        or executor.requested_backend != "direct"
        or executor.full_precision_layout != "current"
        or executor.compute_dtype != np.dtype(np.float64)
        or executor.arithmetic_dtype != np.dtype(np.float64)
        or estimator.genotype_format != "bed"
    ):
        raise RuntimeError(
            "Authenticated CPU placement requires protected direct FP64 BED "
            "execution before NUMA-bound decoding can be enabled."
        )
    try:
        selected_nodes = _validated_early_numa_nodes(
            executor._numa_process_evidence.get("early_numa_attestation"),
            require_static=True,
            expected_pid=os.getpid(),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Authenticated protected FP64 decoding requires the exact live "
            "early static NUMA attestation."
        ) from exc
    estimator._native_numa_bound_decode_nodes = selected_nodes
    estimator._native_numa_bound_decode_records = []
    return selected_nodes


def _balanced_index_tiles(size: int, maximum: int) -> list[tuple[int, int]]:
    count = max(1, math.ceil(size / maximum))
    base, remainder = divmod(size, count)
    tiles: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        width = base + int(index < remainder)
        tiles.append((start, start + width))
        start += width
    return tiles


def _balanced_probe_tiles_for_count(
    size: int, count: int
) -> list[tuple[int, int]]:
    if count <= 0 or count > size:
        raise ValueError("Probe tile count must lie within the probe axis.")
    base, remainder = divmod(size, count)
    tiles: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        width = base + int(index < remainder)
        tiles.append((start, width))
        start += width
    return tiles


def _integrity_workspace_bytes(
    *, enabled: bool, m: int, n: int, k: int, copy_right: bool
) -> int:
    if (
        not enabled
        or min(m, n, k) <= 0
        or 2 * int(m) * int(n) * int(k) < 1_000_000_000
    ):
        return 0
    elements = 8 * (int(m) + 2 * int(k) + 2 * int(n))
    if copy_right:
        elements += int(k) * int(n)
    return elements * np.dtype(np.float64).itemsize


def _resolve_total_process_budget(
    request,
    *,
    baseline_rss_bytes: int,
    available_memory: tuple[int, Mapping] | None = None,
) -> dict:
    parsed = utils.parse_memory_budget(request)
    if baseline_rss_bytes <= 0:
        raise RuntimeError("The process RSS baseline must be positive.")
    if available_memory is None:
        available_bytes, available_evidence = utils.available_memory_bytes()
    else:
        available_bytes, available_evidence = available_memory
    available_bytes = int(available_bytes)
    if available_bytes <= 0:
        raise RuntimeError("No available memory remains for GxE execution.")
    if parsed == "auto":
        reserve = 4 * _GIB
        increment = min(
            int(math.floor(0.65 * available_bytes)), available_bytes - reserve
        )
        if increment < 256 * 1024**2:
            raise RuntimeError(
                "Automatic GxE total-memory planning found less than 0.25 GiB "
                "after its reserve."
            )
        resolved = baseline_rss_bytes + increment
        mode = "auto"
    else:
        resolved = int(float(parsed) * _GIB)
        increment = resolved - baseline_rss_bytes
        mode = "explicit"
        if increment <= 0:
            raise RuntimeError(
                "The explicit GxE total-process budget does not exceed the "
                "current process RSS baseline."
            )
        if increment > available_bytes:
            raise RuntimeError(
                "The explicit GxE total-process budget exceeds currently "
                "available host/cgroup/RLIMIT memory."
            )
    return {
        "requested": parsed,
        "mode": mode,
        "baseline_rss_bytes": int(baseline_rss_bytes),
        "available_increment_bytes": available_bytes,
        "resolved_increment_bytes": int(increment),
        "resolved_bytes": int(resolved),
        "resolved_gib": float(resolved / _GIB),
        "availability_evidence": dict(available_evidence),
    }


def _descriptor_memory_candidate(
    *,
    rows: int,
    variants: int,
    annotation_bins: int,
    environments: int,
    probes: int,
    block_width: int,
    block_count: int,
    feature_columns: int,
    common_rank: int,
    reader_rank: int,
    threads: int,
    environment_tile_count: int,
    probe_tile_count: int,
    protected: bool,
    integrity_enabled: bool,
    native_workspace_gib: float,
    baseline_rss_bytes: int,
    vendor_probe_chunk_width: int | None = None,
    direct_kernel_mode: str = "packed_mailman",
) -> dict:
    if direct_kernel_mode not in {"packed_mailman", "dense_blas_hybrid"}:
        raise ValueError("Unsupported direct descriptor kernel mode.")
    dense_protected = protected and direct_kernel_mode == "dense_blas_hybrid"
    if not protected and direct_kernel_mode != "packed_mailman":
        raise ValueError("The Python fallback has no native dense-hybrid mode.")
    if (
        protected
        and not dense_protected
        and probes > _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
    ):
        raise ValueError(
            "Packed Mailman GxE execution is restricted to at most 10 probes."
        )
    environment_width = math.ceil(environments / environment_tile_count)
    probe_width = math.ceil(probes / probe_tile_count)
    wide_columns = 2 * annotation_bins * environment_width * probe_width
    source_panel = rows * wide_columns * 8
    frozen_mailman = _frozen_mailman_environment()
    mailman_segment = _mailman_segment_size(rows, frozen_mailman)
    mailman_table = 3**mailman_segment
    mailman_code_bytes = 2 if mailman_table <= 65_535 else 4
    # The packed direct path retains one integer index per missing genotype so
    # feature diagnostics can exactly reconstruct mean-imputation corrections.
    # Charge the all-missing upper bound rather than assuming UKBB sparsity.
    packed_missing_indices = rows * block_width * 4
    packed_missing_vectors = block_width * 24
    packed_genotype = (
        rows * math.ceil(block_width / mailman_segment) * mailman_code_bytes
        + block_width * (2 * 8 + 4)
        + packed_missing_indices
        + packed_missing_vectors
    )
    decoded_block = rows * block_width * 8
    source_weights = block_width * wide_columns * 8
    probes_bytes = block_width * probe_width * 8
    block_annotation = block_width * annotation_bins * 8
    block_scales = 2 * block_width * environment_width * 8
    target_columns = 2 * wide_columns
    target_output = block_width * target_columns * 8
    scalar_rows = 1 + 3 * environments
    feature_rhs = rows * (feature_columns + scalar_rows) * 8
    feature_products = block_width * (feature_columns + scalar_rows) * 8
    # Exact context-owned per-worker Mailman scratch from the frozen
    # segment size and q-panel widths (both env-overridable only through
    # the canonical values admitted above).  The generated target kernel
    # does not materialize an RHS panel.
    mailman_worker_plan = _mailman_worker_scratch_bytes(
        rows=rows,
        feature_rhs_columns=feature_columns + scalar_rows,
        wide_columns=wide_columns,
        threads=threads,
        frozen_environment=frozen_mailman,
    )
    mailman_worker_scratch = mailman_worker_plan["total_bytes"]
    # The target Mailman kernel consumes S and the virtual e*S view directly;
    # it does not materialize an N-by-q right-hand-side panel.
    target_rhs = 0
    configured_vendor_workspace = min(
        int(float(native_workspace_gib) * _GIB), 3 * _GIB
    )
    maximum_dense_gemm_flops = 2 * max(
        feature_columns * block_width * rows,
        rows * wide_columns * block_width,
        common_rank * wide_columns * rows,
        block_width * target_columns * rows,
    )
    # The configured native-call workspace is a ceiling, not an allocation.
    # Below the native integrity threshold, no checksum snapshot/workspace is
    # constructed, so charge the exact arrays modeled below rather than a
    # fictitious multi-GiB reserve. In the packed direct path BLAS is otherwise
    # used only for Q^T S and the in-place projection update; native dispatch is
    # deterministically tiled whenever Q has at most 64 columns, and the update
    # is always tiled because beta is nonzero.
    vendor_workspace = (
        0
        if (protected and not integrity_enabled)
        or (protected and not dense_protected and common_rank <= 64)
        or maximum_dense_gemm_flops < 1_000_000_000
        else configured_vendor_workspace
    )

    # variants*annotation_bins appears twice: once for the canonical
    # annotation copy and once for the constructor-time sqrt(A) cache.
    context_copies = 8 * (
        2 * variants * annotation_bins
        + rows * environments
        + rows * feature_columns
        + rows * common_rank
        + rows * environments
        + rows
        + rows * reader_rank
        + annotation_bins
    ) + rows * 4 + block_count * probes * 2 * 8
    persistent_outputs = 8 * (
        8 * variants * environments
        + variants
        + 4 * variants * environments * annotation_bins
    )
    if probes >= 2:
        families = 2 * annotation_bins
        persistent_outputs += 8 * (
            environments * rows * families
            + environments * families * families
        )
    # While the native constructor copies its inputs, the Python-side
    # temporaries coexist with the native copies.  The canonical float64
    # annotation matrix is passed without a fresh copy, so the transient
    # overlap excludes the annotation term; the Philox key table is dropped
    # immediately after construction, bounding its overlap to this window.
    constructor_transient_overlap = (
        context_copies - 8 * 2 * variants * annotation_bins
    )
    fixed_owned = context_copies + persistent_outputs

    if dense_protected:
        feature_integrity = 0
        source_integrity = 0
        projection_integrity = 0
        target_integrity = 0
        dense_target_output = target_output
        phase_peak_bytes = {
            "construction": fixed_owned + constructor_transient_overlap,
            "feature": fixed_owned
            + decoded_block
            + feature_columns * block_width * 8
            + scalar_rows * block_width * 8
            + vendor_workspace,
            # The native source kernel accumulates through one reusable
            # contribution buffer into the persistent source panel.
            "source": fixed_owned
            + decoded_block
            + 2 * source_panel
            + source_weights
            + probes_bytes
            + block_annotation
            + block_scales
            + vendor_workspace,
            "projection": fixed_owned
            + source_panel
            + common_rank * wide_columns * 8
            + vendor_workspace,
            # Source and environment-weighted source are the only persistent
            # target operands; no third source copy or checksum snapshot exists.
            "target_pair_preparation": fixed_owned + 2 * source_panel,
            "target": fixed_owned
            + 2 * source_panel
            + decoded_block
            + dense_target_output
            + block_scales
            + vendor_workspace,
            "publication": fixed_owned
            + max(_GIB, 8 * variants * (4 * annotation_bins + 16)),
        }
        panel_live_peak = 2 * source_panel
    elif protected:
        feature_integrity = 0
        source_integrity = 0
        projection_integrity = _integrity_workspace_bytes(
            enabled=integrity_enabled,
            m=common_rank,
            n=wide_columns,
            k=rows,
            copy_right=True,
        )
        target_integrity = 0
        phase_peak_bytes = {
            "construction": fixed_owned + constructor_transient_overlap,
            "feature": fixed_owned
            + packed_genotype
            + feature_rhs
            + feature_products
            + mailman_worker_scratch,
            "source": fixed_owned
            + packed_genotype
            + source_panel
            + source_weights
            + probes_bytes
            + block_annotation
            + block_scales
            + mailman_worker_scratch,
            "projection": fixed_owned
            + source_panel
            + common_rank * wide_columns * 8
            + projection_integrity
            + vendor_workspace,
            "source_panel_sealing": fixed_owned + source_panel,
            # Target scoring reads S/e*S through a generated view and never
            # materializes the old immutable pair or an N-by-q RHS scratch.
            "target": fixed_owned
            + source_panel
            + packed_genotype
            + target_rhs
            + target_output
            + block_scales
            + mailman_worker_scratch,
            "publication": fixed_owned
            + max(_GIB, 8 * variants * (4 * annotation_bins + 16)),
        }
        panel_live_peak = source_panel
    else:
        # The ordinary Python backend still decodes dense FP64 blocks and
        # materializes both S and e*S. Keep its independent accounting exact;
        # the packed-direct savings must never be used to admit a Python plan.
        feature_integrity = _integrity_workspace_bytes(
            enabled=integrity_enabled,
            m=feature_columns,
            n=block_width,
            k=rows,
            copy_right=True,
        )
        source_integrity = _integrity_workspace_bytes(
            enabled=integrity_enabled,
            m=rows,
            n=wide_columns,
            k=block_width,
            copy_right=True,
        )
        projection_integrity = _integrity_workspace_bytes(
            enabled=integrity_enabled,
            m=common_rank,
            n=wide_columns,
            k=rows,
            copy_right=True,
        )
        target_integrity = _integrity_workspace_bytes(
            enabled=integrity_enabled,
            m=block_width,
            n=target_columns,
            k=rows,
            copy_right=False,
        )
        phase_peak_bytes = {
            "construction": fixed_owned + constructor_transient_overlap,
            "feature": fixed_owned
            + decoded_block
            + feature_columns * block_width * 8
            + 8 * block_width * environments * 8
            + feature_integrity
            + vendor_workspace,
            "source": fixed_owned
            + decoded_block
            + 2 * source_panel
            + source_weights
            + probes_bytes
            + block_annotation
            + block_scales
            + source_integrity
            + vendor_workspace,
            "projection": fixed_owned
            + source_panel
            + common_rank * wide_columns * 8
            + projection_integrity
            + vendor_workspace,
            "source_pair_preparation": fixed_owned + 2 * source_panel,
            "target": fixed_owned
            + 2 * source_panel
            + decoded_block
            + target_output
            + block_scales
            + target_integrity
            + vendor_workspace,
            "publication": fixed_owned
            + max(_GIB, 8 * variants * (4 * annotation_bins + 16)),
        }
        panel_live_peak = 2 * source_panel
    peak_phase = max(phase_peak_bytes, key=phase_peak_bytes.get)
    owned_peak = int(phase_peak_bytes[peak_phase])
    stacks = max(0, threads - 1) * _THREAD_STACK_ALLOWANCE_BYTES
    subtotal_before_slack = (
        baseline_rss_bytes + owned_peak + stacks + _TELEMETRY_ALLOWANCE_BYTES
    )
    allocator_slack = max(
        _MINIMUM_ALLOCATOR_SLACK_BYTES,
        int(math.ceil(0.10 * subtotal_before_slack)),
    )
    subtotal = subtotal_before_slack + allocator_slack
    modeled_peak = int(
        math.ceil((1.0 + _TOTAL_MEMORY_HEADROOM_FRACTION) * subtotal)
    )
    if vendor_probe_chunk_width is None:
        execution_probe_chunks = probe_tile_count
        maximum_execution_probe_chunk_width = probe_width
    else:
        execution_probe_chunks = sum(
            math.ceil((stop - start) / vendor_probe_chunk_width)
            for start, stop in _balanced_index_tiles(probes, probe_width)
        )
        maximum_execution_probe_chunk_width = min(
            probe_width, vendor_probe_chunk_width
        )
    execution_tile_products = environment_tile_count * execution_probe_chunks
    fused_two_pass_execution = (
        protected and environment_tile_count == 1 and probe_tile_count == 1
    )
    passes = (
        2 if fused_two_pass_execution else 1 + 2 * execution_tile_products
    )
    if dense_protected:
        output_calls = (
            block_count * (1 + 2 * execution_tile_products)
            + (execution_tile_products if common_rank > 0 else 0)
        )
    elif protected:
        output_calls = execution_tile_products if common_rank > 0 else 0
    else:
        output_calls = (
            block_count
            + 2 * block_count * execution_tile_products
            + (execution_tile_products if common_rank > 0 else 0)
        )
    # Bounded semantic evidence (audit Finding 6): the native context
    # appends at most one record per block for the feature pass, one per
    # block per tile product for source and target, and one per tile
    # product for projection.  Charge a conservative 4 KiB per record
    # inside the telemetry allowance and refuse plans whose record volume
    # would not fit it, so per-record Python overhead can never dominate
    # the admitted process increment (for example under step_size=1).
    semantic_call_records_maximum = (
        block_count * (1 + 2 * execution_tile_products)
        + 2 * execution_tile_products
    )
    semantic_call_record_allowance_bytes = (
        semantic_call_records_maximum * 4096
    )
    return {
        "block_count": block_count,
        "common_basis_rank": common_rank,
        "protected_execution": protected,
        "direct_kernel_mode": (
            direct_kernel_mode if protected else "python_dense_fallback"
        ),
        "environment_tile_count": environment_tile_count,
        "probe_tile_count": probe_tile_count,
        "maximum_environment_tile_width": environment_width,
        "maximum_probe_tile_width": probe_width,
        "execution_probe_chunk_count": execution_probe_chunks,
        "maximum_execution_probe_chunk_width": (
            maximum_execution_probe_chunk_width
        ),
        "fused_two_pass_execution": fused_two_pass_execution,
        "planned_genotype_passes": passes,
        "planned_native_output_calls": output_calls,
        "semantic_call_records_maximum": semantic_call_records_maximum,
        "semantic_call_record_allowance_bytes": (
            semantic_call_record_allowance_bytes
        ),
        "panel_live_peak_bytes": panel_live_peak,
        "component_bytes": {
            "context_copies": context_copies,
            "constructor_transient_overlap": constructor_transient_overlap,
            "persistent_outputs": persistent_outputs,
            "decoded_genotype_block": (
                decoded_block if dense_protected or not protected else 0
            ),
            "packed_genotype_block": (
                packed_genotype if protected and not dense_protected else 0
            ),
            "packed_missing_index_upper_bound": (
                packed_missing_indices + packed_missing_vectors
                if protected and not dense_protected else 0
            ),
            "mailman_segment_size": mailman_segment,
            "mailman_table_size": mailman_table,
            "mailman_worker_scratch": (
                mailman_worker_scratch
                if protected and not dense_protected else 0
            ),
            "feature_mailman_rhs": (
                feature_rhs if protected and not dense_protected else 0
            ),
            "feature_mailman_products": (
                feature_products if protected and not dense_protected else 0
            ),
            "source_panel": source_panel,
            "source_contribution": (
                source_panel if dense_protected or not protected else 0
            ),
            "source_weights": source_weights,
            "source_probes": probes_bytes,
            "source_block_annotation": block_annotation,
            "source_block_scales": block_scales,
            "target_output": target_output,
            "target_virtual_rhs": (
                target_rhs if protected and not dense_protected else 0
            ),
            "environment_weighted_target_panel": (
                source_panel if dense_protected else 0
            ),
            "feature_integrity": feature_integrity,
            "source_integrity": source_integrity,
            "projection_integrity": projection_integrity,
            "target_integrity": target_integrity,
            "vendor_workspace_allowance": vendor_workspace,
        },
        "phase_peak_bytes": phase_peak_bytes,
        "peak_phase": peak_phase,
        "owned_peak_bytes": owned_peak,
        "thread_stack_allowance_bytes": stacks,
        "telemetry_allowance_bytes": _TELEMETRY_ALLOWANCE_BYTES,
        "allocator_slack_bytes": allocator_slack,
        "headroom_fraction": _TOTAL_MEMORY_HEADROOM_FRACTION,
        "modeled_complete_process_peak_bytes": modeled_peak,
    }


def _shared_execution_tiles(
    estimators: Sequence[GenomewideEnvLDScore],
    compute_dtype: np.dtype,
    *,
    protected: bool,
    blocks: Sequence[tuple[int, int]],
    feature_plan: Mapping,
    common_rank: int,
    integrity_enabled: bool,
    vendor_probe_chunk_width: int | None = None,
    baseline_rss_bytes: int | None = None,
    available_memory: tuple[int, Mapping] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict]:
    first = estimators[0]
    panel_budget_bytes = int(
        min(float(estimator.target_xz_mem) for estimator in estimators) * 1024**3
    )
    if np.dtype(compute_dtype) != np.dtype(np.float64):
        raise RuntimeError(
            "The descriptor-owned memory planner requires FP64 arithmetic."
        )
    requests = [estimator.gxe_total_memory_request for estimator in estimators]
    if any(request != requests[0] for request in requests[1:]):
        raise ValueError(
            "All environments in one process must use the same total-memory budget."
        )
    baseline = (
        int(psutil.Process().memory_info().rss)
        if baseline_rss_bytes is None
        else int(baseline_rss_bytes)
    )
    total_budget = _resolve_total_process_budget(
        requests[0],
        baseline_rss_bytes=baseline,
        available_memory=available_memory,
    )
    maximum_block = max(stop - start for start, stop in blocks)
    candidates: list[tuple[tuple, dict]] = []
    if (
        vendor_probe_chunk_width is not None
        and (
            type(vendor_probe_chunk_width) is not int
            or vendor_probe_chunk_width <= 0
        )
    ):
        raise ValueError(
            "vendor_probe_chunk_width must be a positive integer or None."
        )
    for environment_count in range(1, len(estimators) + 1):
        for probe_count in range(1, first.nvecs + 1):
            modes = (
                (
                    ("packed_mailman", "dense_blas_hybrid")
                    if first.nvecs <= _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
                    else ("dense_blas_hybrid",)
                )
                if protected
                else ("packed_mailman",)
            )
            for mode in modes:
                candidate = _descriptor_memory_candidate(
                    rows=int(first.nsamp),
                    variants=int(first.nsnps),
                    annotation_bins=int(first.nbins),
                    environments=len(estimators),
                    probes=int(first.nvecs),
                    block_width=int(maximum_block),
                    block_count=len(blocks),
                    feature_columns=int(feature_plan["basis"].shape[1]),
                    common_rank=int(common_rank),
                    reader_rank=int(first.p_eff) + 1,
                    threads=int(first.num_threads),
                    environment_tile_count=environment_count,
                    probe_tile_count=probe_count,
                    protected=protected,
                    integrity_enabled=integrity_enabled,
                    native_workspace_gib=float(first.native_workspace_gib),
                    baseline_rss_bytes=baseline,
                    vendor_probe_chunk_width=vendor_probe_chunk_width,
                    direct_kernel_mode=mode,
                )
                if (
                    candidate["panel_live_peak_bytes"] > panel_budget_bytes
                    or candidate["modeled_complete_process_peak_bytes"]
                    > total_budget["resolved_bytes"]
                    or candidate["planned_native_output_calls"]
                    > _NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
                    or candidate["planned_native_output_calls"]
                    > _MAX_GEMM_TELEMETRY_RECORDS
                    # Bounded semantic evidence (audit Finding 6): a
                    # candidate whose per-call record volume cannot fit the
                    # telemetry allowance is infeasible.
                    or (
                        protected
                        and candidate["semantic_call_record_allowance_bytes"]
                        > _TELEMETRY_ALLOWANCE_BYTES
                    )
                ):
                    continue
                key = (
                    candidate["planned_genotype_passes"],
                    (
                        0
                        if (
                            first.nvecs
                            <= _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
                            and mode == "packed_mailman"
                        )
                        or (
                            first.nvecs
                            > _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
                            and mode == "dense_blas_hybrid"
                        )
                        else 1
                    ),
                    -candidate["maximum_probe_tile_width"],
                    -candidate["maximum_environment_tile_width"],
                    candidate["modeled_complete_process_peak_bytes"],
                    environment_count,
                    probe_count,
                )
                candidates.append((key, candidate))
    if not candidates:
        minimum = _descriptor_memory_candidate(
            rows=int(first.nsamp),
            variants=int(first.nsnps),
            annotation_bins=int(first.nbins),
            environments=len(estimators),
            probes=int(first.nvecs),
            block_width=int(maximum_block),
            block_count=len(blocks),
            feature_columns=int(feature_plan["basis"].shape[1]),
            common_rank=int(common_rank),
            reader_rank=int(first.p_eff) + 1,
            threads=int(first.num_threads),
            environment_tile_count=len(estimators),
            probe_tile_count=int(first.nvecs),
            protected=protected,
            integrity_enabled=integrity_enabled,
            native_workspace_gib=float(first.native_workspace_gib),
            baseline_rss_bytes=baseline,
            vendor_probe_chunk_width=vendor_probe_chunk_width,
            direct_kernel_mode=(
                "dense_blas_hybrid"
                if protected
                and first.nvecs > _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
                else "packed_mailman"
            ),
        )
        raise RuntimeError(
            "No GxE descriptor tile plan satisfies both memory contracts; "
            f"minimum_panel={minimum['panel_live_peak_bytes']} bytes, "
            f"panel_limit={panel_budget_bytes} bytes, minimum_process_peak="
            f"{minimum['modeled_complete_process_peak_bytes']} bytes, "
            f"total_limit={total_budget['resolved_bytes']} bytes."
        )
    _, selected = min(candidates, key=lambda item: item[0])
    environment_tiles = _balanced_index_tiles(
        len(estimators), selected["maximum_environment_tile_width"]
    )
    vtiles = _balanced_probe_tiles_for_count(
        int(first.nvecs), selected["probe_tile_count"]
    )
    memory_plan = {
        "schema": _COMPLETE_MEMORY_PLAN_SCHEMA,
        "schema_version": 1,
        "panel_budget_bytes": panel_budget_bytes,
        "panel_budget_gib": float(panel_budget_bytes / _GIB),
        "panel_budget_semantics": (
            "source_plus_persistent_environment_weighted_target_panel"
            if selected.get("direct_kernel_mode") == "dense_blas_hybrid"
            else "single_persistent_source_panel_with_virtual_weighted_rhs"
            if protected
            else "source_plus_weighted_source_peak"
        ),
        "total_process_budget": total_budget,
        "candidate_count": (
            len(estimators)
            * int(first.nvecs)
            * (
                2
                if protected
                and first.nvecs <= _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
                else 1
            )
        ),
        "mailman_maximum_probe_count": (
            _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
        ),
        "mailman_probe_count_eligible": bool(
            first.nvecs <= _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
        ),
        "feasible_candidate_count": len(candidates),
        "objective": (
            "minimum_passes_then_widest_probe_then_widest_environment_"
            "then_lowest_modeled_peak"
        ),
        **selected,
        "memory_contract_satisfied": True,
        "budget_margin_bytes": int(
            total_budget["resolved_bytes"]
            - selected["modeled_complete_process_peak_bytes"]
        ),
    }
    return environment_tiles, vtiles, memory_plan


def _accumulate_sketch_block(
    estimator: GenomewideEnvLDScore,
    target: np.ndarray,
    feature: np.ndarray,
    probes: np.ndarray,
    annotation: np.ndarray,
    executor: _MultiEnvironmentGemm,
) -> None:
    if not executor.protected:
        estimator._accumulate_sketch_block(target, feature, probes, annotation)
        return
    sqrt_annotation = np.sqrt(np.maximum(annotation, 0))
    for index in range(estimator.nbins):
        weights = sqrt_annotation[:, index]
        if not np.any(weights):
            continue
        segment = slice(
            index * probes.shape[1], (index + 1) * probes.shape[1]
        )
        weighted_probes = np.asfortranarray(
            weights.reshape(-1, 1) * probes, dtype=np.float64
        )
        target[:, segment] += executor.nn(feature, weighted_probes)


def _accumulate_unprojected_source_block(
    estimator: GenomewideEnvLDScore,
    target: np.ndarray,
    genotype: np.ndarray,
    probes: np.ndarray,
    annotation: np.ndarray,
    start: int,
    stop: int,
    *,
    interaction: bool,
    executor: _MultiEnvironmentGemm,
) -> None:
    """Accumulate a raw source; one panel projection follows the full pass."""
    scales = (
        estimator.inv_sqrt_resvar_w_all
        if interaction else estimator.inv_sqrt_resvar_x_all
    )
    if scales is None:
        raise RuntimeError("GxE source scales have not been precomputed.")
    sqrt_annotation = np.sqrt(np.maximum(annotation, 0))
    block_scales = scales[start:stop]
    for index in range(estimator.nbins):
        weights = sqrt_annotation[:, index]
        if not np.any(weights):
            continue
        segment = slice(
            index * probes.shape[1], (index + 1) * probes.shape[1]
        )
        weighted_probes = np.asfortranarray(
            (weights * block_scales).reshape(-1, 1) * probes,
            dtype=np.float64,
        )
        contribution = executor.nn(genotype, weighted_probes)
        if interaction:
            contribution *= estimator.env[:, None]
        target[:, segment] += contribution


def _project_source_panel_inplace(
    estimator: GenomewideEnvLDScore,
    panel: np.ndarray,
    executor: _MultiEnvironmentGemm,
) -> float:
    """Project one completed source panel and return relative leakage."""
    if estimator.p_eff > 0:
        coefficients = executor.tn(estimator.C_int, panel)
        panel -= executor.nn(estimator.C_int, coefficients)
    panel -= panel.mean(axis=0, keepdims=True)

    denominator = np.sum(panel * panel, axis=0, dtype=np.float64)
    root_n = math.sqrt(float(estimator.nsamp))
    leaked = (panel.sum(axis=0, dtype=np.float64) / root_n) ** 2
    if estimator.p_eff > 0:
        coefficients = executor.tn(estimator.C_int, panel)
        leaked += np.sum(coefficients * coefficients, axis=0, dtype=np.float64)
    relative = np.sqrt(
        np.divide(
            leaked,
            denominator,
            out=np.full_like(leaked, np.inf),
            where=denominator > 0.0,
        )
    )
    maximum = float(np.max(relative))
    if not np.isfinite(maximum) or maximum > 1.0e-9:
        raise RuntimeError(
            f"Projected source panel for environment {estimator.env_name!r} "
            f"has excessive fixed-effect leakage: {maximum:.6g}."
        )
    return maximum


def _accumulate_packed_source_block(
    estimators: Sequence[GenomewideEnvLDScore],
    environment_start: int,
    environment_stop: int,
    target: np.ndarray,
    genotype: np.ndarray,
    probes: np.ndarray,
    annotation: np.ndarray,
    start: int,
    stop: int,
    columns: int,
    executor: _MultiEnvironmentGemm,
) -> None:
    """Accumulate all raw X/W sources in one environment-tile GEMM."""
    tile_size = environment_stop - environment_start
    if executor._multi_environment_kernel is not None:
        scale_x = np.asfortranarray(
            np.column_stack(
                [
                    estimator.inv_sqrt_resvar_x_all[start:stop]
                    for estimator in estimators[
                        environment_start:environment_stop
                    ]
                ]
            ),
            dtype=np.float64,
        )
        scale_w = np.asfortranarray(
            np.column_stack(
                [
                    estimator.inv_sqrt_resvar_w_all[start:stop]
                    for estimator in estimators[
                        environment_start:environment_stop
                    ]
                ]
            ),
            dtype=np.float64,
        )
        phase_wall, phase_cpu = time.perf_counter(), time.process_time()
        executor.native_source_block(
            target,
            genotype,
            probes,
            annotation,
            scale_x,
            scale_w,
            environment_start,
            semantic={"phase": "source_gemm"},
        )
        executor.record_phase_elapsed("source_gemm", phase_wall, phase_cpu)
        return
    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
    weighted = np.zeros(
        (stop - start, tile_size * 2 * columns),
        dtype=executor.compute_dtype,
        order="F",
    )
    sqrt_annotation = np.sqrt(np.maximum(annotation, 0.0))
    for local_index, estimator in enumerate(
        estimators[environment_start:environment_stop]
    ):
        scale_x = estimator.inv_sqrt_resvar_x_all
        scale_w = estimator.inv_sqrt_resvar_w_all
        if scale_x is None or scale_w is None:
            raise RuntimeError("GxE source scales have not been precomputed.")
        offset = local_index * 2 * columns
        for annotation_index in range(estimator.nbins):
            annotation_weights = sqrt_annotation[:, annotation_index]
            if not np.any(annotation_weights):
                continue
            segment = slice(
                annotation_index * probes.shape[1],
                (annotation_index + 1) * probes.shape[1],
            )
            weighted[:, offset + segment.start:offset + segment.stop] = (
                (annotation_weights * scale_x[start:stop]).reshape(-1, 1)
                * probes
            )
            interaction_offset = offset + columns
            weighted[
                :,
                interaction_offset + segment.start:interaction_offset + segment.stop,
            ] = (
                (annotation_weights * scale_w[start:stop]).reshape(-1, 1)
                * probes
            )
    executor.record_phase_elapsed("source_weight_packing", phase_wall, phase_cpu)
    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
    with executor.semantic_context({"phase": "source_gemm"}):
        contribution = executor.source_product(genotype, weighted)
    executor.record_phase_elapsed("source_gemm", phase_wall, phase_cpu)
    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
    for local_index, estimator in enumerate(
        estimators[environment_start:environment_stop]
    ):
        offset = local_index * 2 * columns
        target[:, offset:offset + columns] += contribution[
            :, offset:offset + columns
        ]
        target[:, offset + columns:offset + 2 * columns] += (
            estimator.env[:, None]
            * contribution[:, offset + columns:offset + 2 * columns]
        )
    executor.record_phase_elapsed("source_context_correction", phase_wall, phase_cpu)


def _project_packed_sources_inplace(
    estimators: Sequence[GenomewideEnvLDScore],
    environment_start: int,
    environment_stop: int,
    panel: np.ndarray,
    columns: int,
    common: np.ndarray,
    directions: np.ndarray,
    executor: _MultiEnvironmentGemm,
) -> list[float]:
    """Apply every environment projector to a packed source panel."""
    if executor._multi_environment_kernel is not None:
        return executor.native_project_sources(
            panel,
            columns,
            environment_start,
            semantic={"phase": "projection_context_correction"},
        )
    if common.shape[1]:
        coefficients = executor.tn(common, panel)
        panel -= executor.nn(common, coefficients)
    for local_index in range(environment_stop - environment_start):
        segment = slice(
            local_index * 2 * columns, (local_index + 1) * 2 * columns
        )
        direction = directions[:, environment_start + local_index]
        coefficients = np.einsum(
            "n,nj->j", direction, panel[:, segment], optimize=False
        )
        panel[:, segment] -= direction[:, None] * coefficients
    panel -= panel.mean(axis=0, keepdims=True)

    leakages: list[float] = []
    root_n = math.sqrt(float(estimators[0].nsamp))
    for local_index, estimator in enumerate(
        estimators[environment_start:environment_stop]
    ):
        segment = slice(local_index * 2 * columns, (local_index + 1) * 2 * columns)
        source = panel[:, segment]
        denominator = np.sum(source * source, axis=0, dtype=np.float64)
        leaked = (source.sum(axis=0, dtype=np.float64) / root_n) ** 2
        if estimator.p_eff:
            fixed_coefficients = np.einsum(
                "ni,nj->ij", estimator.C_int, source, optimize=False
            )
            leaked += np.sum(
                fixed_coefficients * fixed_coefficients,
                axis=0,
                dtype=np.float64,
            )
        relative = np.sqrt(
            np.divide(
                leaked,
                denominator,
                out=np.full_like(leaked, np.inf),
                where=denominator > 0.0,
            )
        )
        maximum = float(np.max(relative))
        if not np.isfinite(maximum) or maximum > 1.0e-9:
            raise RuntimeError(
                f"Projected source panel for environment {estimator.env_name!r} "
                f"has excessive fixed-effect leakage: {maximum:.6g}."
            )
        leakages.append(maximum)
    return leakages


def _publish_json_no_replace(payload: dict, target: Path) -> tuple[int, int]:
    """Atomically seal a new batch manifest without replacing another writer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        staged = temporary_path.stat(follow_symlinks=False)
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing concurrently created multi-environment manifest: {target}."
            ) from exc
        return staged.st_dev, staged.st_ino
    finally:
        temporary_path.unlink(missing_ok=True)


def _published_file_identity(path: Path) -> tuple[Path, int, int]:
    observed = path.stat(follow_symlinks=False)
    return path, observed.st_dev, observed.st_ino


def _rollback_published_files(
    published: Sequence[tuple[Path, int, int]],
) -> None:
    """Remove only files whose published inode is still ours."""
    for path, device, inode in reversed(published):
        try:
            observed = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if observed.st_dev != device or observed.st_ino != inode:
            continue
        path.unlink(missing_ok=True)


def _validated_complete_memory_plan(value: Mapping) -> dict:
    """Reconstruct the bounded per-process plan from JSON-native evidence."""
    if not isinstance(value, Mapping):
        raise ValueError("Complete-process memory plan is absent or malformed.")
    if (
        value.get("schema") != _COMPLETE_MEMORY_PLAN_SCHEMA
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("memory_contract_satisfied") is not True
    ):
        raise ValueError("Complete-process memory plan contract is unsupported.")

    def exact_nonnegative(mapping: Mapping, key: str) -> int:
        observed = mapping.get(key)
        if type(observed) is not int or observed < 0:
            raise ValueError(
                f"Complete-process memory plan has malformed {key}."
            )
        return observed

    panel_budget = exact_nonnegative(value, "panel_budget_bytes")
    panel_peak = exact_nonnegative(value, "panel_live_peak_bytes")
    modeled_peak = exact_nonnegative(
        value, "modeled_complete_process_peak_bytes"
    )
    margin = exact_nonnegative(value, "budget_margin_bytes")
    passes = exact_nonnegative(value, "planned_genotype_passes")
    calls = exact_nonnegative(value, "planned_native_output_calls")
    if passes <= 0:
        raise ValueError("Complete-process memory plan has an empty execution plan.")
    total = value.get("total_process_budget")
    if not isinstance(total, Mapping):
        raise ValueError("Complete-process total budget evidence is malformed.")
    resolved = exact_nonnegative(total, "resolved_bytes")
    baseline = exact_nonnegative(total, "baseline_rss_bytes")
    increment = exact_nonnegative(total, "resolved_increment_bytes")
    if (
        total.get("mode") not in ("auto", "explicit")
        or resolved != baseline + increment
        or panel_peak > panel_budget
        or modeled_peak > resolved
        or margin != resolved - modeled_peak
        or calls > _NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        or calls > _MAX_GEMM_TELEMETRY_RECORDS
    ):
        raise ValueError("Complete-process memory plan violates its hard bounds.")
    phases = value.get("phase_peak_bytes")
    peak_phase = value.get("peak_phase")
    if (
        not isinstance(phases, Mapping)
        or not phases
        or type(peak_phase) is not str
        or peak_phase not in phases
    ):
        raise ValueError("Complete-process phase peaks are malformed.")
    phase_values = {
        str(key): exact_nonnegative(phases, key) for key in phases
    }
    if exact_nonnegative(value, "owned_peak_bytes") != phase_values[peak_phase]:
        raise ValueError("Complete-process peak phase disagrees with its value.")
    if phase_values[peak_phase] != max(phase_values.values()):
        raise ValueError("Complete-process peak phase is not maximal.")
    for key in (
        "block_count",
        "environment_tile_count",
        "probe_tile_count",
        "execution_probe_chunk_count",
        "maximum_execution_probe_chunk_width",
        "maximum_environment_tile_width",
        "maximum_probe_tile_width",
        "candidate_count",
        "feasible_candidate_count",
    ):
        if exact_nonnegative(value, key) <= 0:
            raise ValueError(f"Complete-process memory plan has invalid {key}.")
    fused_two_pass = value.get("fused_two_pass_execution")
    protected_execution = value.get("protected_execution")
    if type(fused_two_pass) is not bool or type(protected_execution) is not bool:
        raise ValueError(
            "Complete-process memory plan has malformed fused execution evidence."
        )
    common_basis_rank = exact_nonnegative(value, "common_basis_rank")
    direct_kernel_mode = value.get("direct_kernel_mode")
    if direct_kernel_mode not in {
        "dense_blas_hybrid",
        "packed_mailman",
        "python_dense_fallback",
    }:
        raise ValueError("Complete-process memory plan has an invalid kernel mode.")
    if protected_execution:
        if direct_kernel_mode == "python_dense_fallback":
            raise ValueError("Protected memory plan selected the Python kernel.")
    elif direct_kernel_mode != "python_dense_fallback":
        raise ValueError("Unprotected memory plan selected a native kernel.")
    mailman_limit = exact_nonnegative(
        value, "mailman_maximum_probe_count"
    )
    mailman_eligible = value.get("mailman_probe_count_eligible")
    if (
        mailman_limit != _MULTI_ENVIRONMENT_MAILMAN_MAXIMUM_PROBES
        or type(mailman_eligible) is not bool
        or (
            direct_kernel_mode == "packed_mailman"
            and mailman_eligible is not True
        )
    ):
        raise ValueError("Complete-process Mailman eligibility is malformed.")
    environment_tiles = value["environment_tile_count"]
    probe_tiles = value["probe_tile_count"]
    execution_probe_chunks = value["execution_probe_chunk_count"]
    execution_products = environment_tiles * execution_probe_chunks
    expected_fused = (
        protected_execution and environment_tiles == 1 and probe_tiles == 1
    )
    expected_passes = 2 if expected_fused else 1 + 2 * execution_products
    if direct_kernel_mode == "dense_blas_hybrid":
        expected_calls = (
            value["block_count"] * (1 + 2 * execution_products)
            + (execution_products if common_basis_rank > 0 else 0)
        )
    elif direct_kernel_mode == "packed_mailman":
        expected_calls = execution_products if common_basis_rank > 0 else 0
    else:
        expected_calls = (
            value["block_count"] * (1 + 2 * execution_products)
            + (execution_products if common_basis_rank > 0 else 0)
        )
    if (
        fused_two_pass != expected_fused
        or passes != expected_passes
        or calls != expected_calls
    ):
        raise ValueError(
            "Complete-process memory plan execution counts are inconsistent."
        )
    components = value.get("component_bytes")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("Complete-process component evidence is malformed.")
    for key in components:
        exact_nonnegative(components, key)
    return _json_safe(dict(value))


def combine_multi_environment_reference_batches(
    batch_manifests: Sequence[str | Path],
    *,
    batch_manifest: str | Path,
    environment_order: Sequence[str],
    require_cpu_placement: bool = False,
    expected_cpu_groups: Sequence[Sequence[int]] | None = None,
    expected_numa_groups: Sequence[Sequence[int]] | None = None,
) -> Path:
    """Publish one validated index over isolated environment-group batches."""
    sources = tuple(Path(path).expanduser().resolve() for path in batch_manifests)
    target = Path(batch_manifest).expanduser().resolve()
    if type(require_cpu_placement) is not bool:
        raise ValueError("require_cpu_placement must be a built-in bool.")
    if not sources or (len(sources) < 2 and not require_cpu_placement):
        raise ValueError("Combining multi-environment batches requires at least two inputs.")
    if target.exists():
        raise FileExistsError(f"Refusing existing multi-environment manifest: {target}.")
    expected_groups: tuple[tuple[int, ...], ...] | None = None
    if expected_cpu_groups is not None:
        try:
            expected_groups = tuple(tuple(group) for group in expected_cpu_groups)
        except TypeError as exc:
            raise ValueError(
                "Expected CPU groups must be ordered CPU-ID sequences."
            ) from exc
        if (
            len(expected_groups) != len(sources)
            or any(
                not group
                or any(type(cpu) is not int or cpu < 0 for cpu in group)
                or tuple(sorted(set(group))) != group
                for group in expected_groups
            )
        ):
            raise ValueError(
                "Expected CPU groups must be ordered canonical CPU-ID tuples matching the inputs."
            )
    expected_numa: tuple[tuple[int, ...], ...] | None = None
    if expected_numa_groups is not None:
        try:
            expected_numa = tuple(
                tuple(group) for group in expected_numa_groups
            )
        except TypeError as exc:
            raise ValueError(
                "Expected NUMA groups must be ordered node-ID sequences."
            ) from exc
        if (
            len(expected_numa) != len(sources)
            or any(
                not group
                or any(type(node) is not int or node < 0 for node in group)
                or tuple(sorted(set(group))) != group
                for group in expected_numa
            )
        ):
            raise ValueError(
                "Expected NUMA groups must be ordered canonical node-ID tuples matching the inputs."
            )
    if (expected_groups is None) != (expected_numa is None):
        raise ValueError(
            "Controller CPU and NUMA group assignments must be supplied together."
        )
    if require_cpu_placement and (
        expected_groups is None or expected_numa is None
    ):
        raise ValueError(
            "Authenticated parallel combination requires its ordered expected CPU and NUMA groups."
        )

    payloads: list[dict] = []
    payload_layouts: list[str] = []
    payload_placements: list[dict | None] = []
    payload_numa_groups: list[tuple[int, ...] | None] = []
    payload_decode_reports: list[dict | None] = []
    payload_packed_source_summaries: list[dict | None] = []
    payload_memory_plans: list[dict | None] = []
    resolved_references: dict[str, Path] = {}
    current_layout_metadata = {
        "source_panel_memory_order": "F",
        "target_panel_memory_order": "F",
        "target_genotype_memory_order": "F",
        "source_to_target_layout_transition": "none",
    }

    def optional_nonnegative_gib(payload: Mapping, key: str) -> float | None:
        if key not in payload or payload[key] is None:
            return None
        raw = payload[key]
        if isinstance(raw, bool):
            raise ValueError(f"Malformed {key}: expected a nonnegative finite number.")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed {key}: expected a nonnegative finite number."
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Malformed {key}: expected a nonnegative finite number.")
        return value

    def sum_optional_gib(key: str) -> float | None:
        values = [optional_nonnegative_gib(payload, key) for payload in payloads]
        if any(value is None for value in values):
            return None
        return float(sum(value for value in values if value is not None))

    def validated_optional_cpu_placement(
        payload: Mapping, *, source_description: str
    ) -> dict | None:
        placement_present = "cpu_placement" in payload
        complete_present = "cpu_placement_complete" in payload
        if placement_present != complete_present:
            raise ValueError(
                f"Mixed CPU placement field presence in {source_description}."
            )
        if not placement_present:
            return None
        if payload.get("cpu_placement_complete") is not True:
            raise ValueError(
                f"Incomplete CPU placement evidence in {source_description}."
            )
        try:
            return _validate_cpu_placement_attestation(payload["cpu_placement"])
        except RuntimeError as exc:
            raise ValueError(
                f"Malformed CPU placement evidence in {source_description}."
            ) from exc

    def validate_early_numa_attestation(
        attestation: Mapping, source: Path
    ) -> tuple[int, ...]:
        try:
            return _validated_early_numa_nodes(
                attestation,
                require_static=require_cpu_placement,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Early NUMA attestation is incomplete or malformed in {source}."
            ) from exc

    invariant_keys = (
        "kind",
        "schema_version",
        "execution",
        "requested_backend",
        "protected_native_gemm",
        "multi_environment_native_end_to_end",
        "multi_environment_native_kernel_schema",
        "multi_environment_descriptor_owned_end_to_end",
        "multi_environment_direct_context_schema",
        "arithmetic_dtype",
        "requested_storage_dtype",
        "gemm_backend",
        "native_gemm_integrity_enabled",
        "native_gemm_checksum_enabled",
        "native_blas_runtime_isolation",
        "source_panel_memory_order",
        "target_panel_memory_order",
        "target_genotype_memory_order",
        "source_to_target_layout_transition",
        "common_complete_case_samples",
        "num_variants",
        "randomization",
    )
    for source in sources:
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(
                f"Multi-environment group manifest is missing or not a regular file: {source}."
            )
        observed = json.loads(source.read_text(encoding="utf-8"))
        if observed.get("kind") != "summit.gxe.multi_environment_reference_batch":
            raise ValueError(f"Unexpected multi-environment batch kind in {source}.")
        if int(observed.get("schema_version", -1)) != 1:
            raise ValueError(f"Unsupported multi-environment batch schema in {source}.")
        declared_layout = observed.get("full_precision_layout")
        if declared_layout != "current":
            raise ValueError(
                f"Unsupported full-precision layout in {source}: "
                f"{declared_layout!r}."
            )
        for key, value in current_layout_metadata.items():
            if observed.get(key) != value:
                raise ValueError(
                    f"Current-layout manifest {source} reports noncanonical "
                    f"{key}={observed.get(key)!r}; expected {value!r}."
                )

        packed_key = "modeled_packed_source_panel_gib"
        seal_key = "modeled_target_pair_sealing_live_peak_gib"
        observed[packed_key] = optional_nonnegative_gib(observed, packed_key)
        observed[seal_key] = optional_nonnegative_gib(observed, seal_key)
        if observed[packed_key] is None or observed[seal_key] is None:
            raise ValueError(f"Current memory fields are required in {source}.")
        performance = observed.get("performance_telemetry")
        if not isinstance(performance, Mapping):
            raise ValueError(f"Malformed performance telemetry in {source}.")
        if performance.get("full_precision_layout") != "current":
            raise ValueError(
                "Performance telemetry full_precision_layout disagrees "
                f"with group manifest {source}."
            )
        if not (
            performance.get("phase_telemetry_complete") is True
            and performance.get("telemetry_complete") is True
            and performance.get("capture_boundary")
            == "post_output_artifact_publication_pre_batch_manifest"
        ):
            raise ValueError(f"Incomplete final performance telemetry in {source}.")
        group_placement = validated_optional_cpu_placement(
            observed, source_description=f"group manifest {source}"
        )
        performance_placement = validated_optional_cpu_placement(
            performance if isinstance(performance, Mapping) else {},
            source_description=f"performance telemetry in {source}",
        )
        if performance_placement != group_placement:
            raise ValueError(
                f"CPU placement disagrees between group manifest and "
                f"performance telemetry in {source}."
            )
        top_level_numa = observed.get("early_numa_attestation")
        performance_numa = (
            performance.get("early_numa_attestation")
            if isinstance(performance, Mapping) else None
        )
        group_numa: tuple[int, ...] | None = None
        if top_level_numa is not None or performance_numa is not None:
            if (
                not isinstance(top_level_numa, Mapping)
                or not isinstance(performance_numa, Mapping)
                or dict(top_level_numa) != dict(performance_numa)
            ):
                raise ValueError(
                    f"Early NUMA attestation disagrees between manifest and "
                    f"performance telemetry in {source}."
                )
            group_numa = validate_early_numa_attestation(top_level_numa, source)
        group_decode_report: dict | None = None
        group_packed_source_summary: dict | None = None
        if require_cpu_placement:
            descriptor_direct = bool(
                observed.get("execution")
                == "descriptor_owned_native_multi_environment_end_to_end"
                and observed.get(
                    "multi_environment_descriptor_owned_end_to_end"
                ) is True
                and observed.get("multi_environment_direct_context_schema")
                == "summit.multi_environment_direct_context.v3"
            )
            if (
                group_placement is None
                or group_numa is None
                or observed.get("requested_backend") != "direct"
                or observed.get("protected_native_gemm") is not True
                or observed.get("arithmetic_dtype") != "float64"
                or not isinstance(performance, Mapping)
            ):
                raise ValueError(
                    f"Contracted environment group lacks the protected direct "
                    f"FP64 execution contract in {source}."
                )
            requires_decode = not descriptor_direct
            if descriptor_direct:
                direct_info = performance.get(
                    "multi_environment_direct_context"
                )
                dense_direct = bool(
                    isinstance(direct_info, Mapping)
                    and direct_info.get("materialized_target_rhs") is True
                )
                if (
                    not isinstance(direct_info, Mapping)
                    or direct_info.get("schema")
                    != "summit.multi_environment_direct_context.v3"
                    or direct_info.get("contracted_numa_decode")
                    is not dense_direct
                    or direct_info.get("execution_kernel")
                    != (
                        "dense_private_blas_streamed_pair"
                        if dense_direct
                        else "packed_mailman_low_memory_fallback"
                    )
                    or direct_info.get(
                        "contracted_packed_source_panel_numa"
                    ) is not True
                    or type(direct_info.get("environment_tile_count")) is not int
                    or direct_info["environment_tile_count"] <= 0
                    or type(direct_info.get("execution_probe_chunk_count"))
                    is not int
                    or direct_info["execution_probe_chunk_count"] <= 0
                ):
                    raise ValueError(
                        f"Descriptor direct-context declaration is malformed in "
                        f"{source}."
                    )
                expected_source_records = (
                    direct_info["environment_tile_count"]
                    * direct_info["execution_probe_chunk_count"]
                )
                top_level_source = observed.get(
                    "packed_source_panel_numa"
                )
                try:
                    source_records = _validate_packed_source_panel_numa_records(
                        performance.get("packed_source_panel_numa_records"),
                        expected_nodes=group_numa,
                        expected_count=expected_source_records,
                        expected_combined_pair=dense_direct,
                    )
                except RuntimeError as exc:
                    raise ValueError(
                        f"Contracted packed source-panel NUMA evidence is "
                        f"malformed in {source}."
                    ) from exc
                group_packed_source_summary = {
                    "schema": _PACKED_SOURCE_PANEL_NUMA_SCHEMA,
                    "record_count": len(source_records),
                    "records_included": False,
                    "complete": True,
                }
                top_level_source_matches = bool(
                    isinstance(top_level_source, Mapping)
                    and dict(top_level_source)
                    == group_packed_source_summary
                )
                if (
                    observed.get("packed_source_panel_numa_required") is not True
                    or observed.get("packed_source_panel_numa_complete") is not True
                    or not top_level_source_matches
                    or performance.get(
                        "packed_source_panel_numa_required"
                    ) is not True
                    or performance.get(
                        "packed_source_panel_numa_complete"
                    ) is not True
                    or performance.get(
                        "packed_source_panel_numa_record_count"
                    ) != expected_source_records
                ):
                    raise ValueError(
                        f"Contracted packed source-panel NUMA summary is "
                        f"incomplete or inconsistent in {source}."
                    )
                requires_decode = dense_direct
            if requires_decode:
                top_level_decode = observed.get("numa_bound_bed_decode")
                performance_decode = performance.get("numa_bound_bed_decode")
                if (
                    observed.get("numa_bound_bed_decode_required") is not True
                    or observed.get("numa_bound_bed_decode_complete") is not True
                    or performance.get("numa_bound_bed_decode_required") is not True
                    or performance.get("numa_bound_bed_decode_complete") is not True
                    or not isinstance(top_level_decode, Mapping)
                    or not isinstance(performance_decode, Mapping)
                ):
                    raise ValueError(
                        f"Contracted environment group lacks complete protected "
                        f"FP64 NUMA-bound BED decode evidence in {source}."
                    )
                try:
                    # Full per-read evidence is retained once, inside the final
                    # performance record. The top-level declaration is its
                    # bounded summary.
                    group_decode_report = (
                        _validate_persisted_numa_bound_decode_report(
                            performance_decode,
                            expected_nodes=group_numa,
                            expected_sample_count=int(
                                observed["common_complete_case_samples"]
                            ),
                            expected_num_variants=int(observed["num_variants"]),
                            expected_passes=int(
                                observed["shared_genotype_passes"]
                            ),
                        )
                    )
                    if dict(top_level_decode) != _decode_report_for_output(
                        group_decode_report, include_records=False
                    ):
                        raise RuntimeError(
                            "Top-level NUMA-bound BED decode summary disagrees "
                            "with complete performance evidence."
                        )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError(
                        f"Contracted NUMA-bound BED decode evidence is malformed "
                        f"in {source}."
                    ) from exc
        group_memory_plan = _validated_complete_memory_plan(
            observed.get("complete_process_memory_plan")
        )
        if payloads:
            baseline = payloads[0]
            disagreements = [
                key for key in invariant_keys
                if observed.get(key) != baseline.get(key)
            ]
            if declared_layout != payload_layouts[0]:
                disagreements.append("full_precision_layout")
            if disagreements:
                raise ValueError(
                    f"Environment-group manifest {source} disagrees on: "
                    + ", ".join(disagreements)
                    + "."
                )
        payloads.append(observed)
        payload_layouts.append(str(declared_layout))
        payload_placements.append(group_placement)
        payload_numa_groups.append(group_numa)
        payload_decode_reports.append(group_decode_report)
        payload_memory_plans.append(group_memory_plan)
        for record in observed.get("references", ()):
            if not isinstance(record, Mapping) or set(record) != {
                "environment",
                "reference",
            }:
                raise ValueError(f"Malformed reference declaration in {source}.")
            environment = str(record.get("environment", ""))
            if not environment:
                raise ValueError(f"Malformed reference declaration in {source}.")
            if environment in resolved_references:
                raise ValueError(
                    f"Environment {environment!r} occurs in more than one group manifest."
                )
            reference = (source.parent / str(record.get("reference", ""))).resolve()
            if not reference.is_file() or reference.is_symlink():
                raise FileNotFoundError(
                    f"Declared environment reference is missing or not regular: {reference}."
                )
            reference_payload = json.loads(reference.read_text(encoding="utf-8"))
            if not isinstance(reference_payload, Mapping):
                raise ValueError(
                    f"Malformed environment reference manifest: {reference}."
                )
            reference_placement = validated_optional_cpu_placement(
                reference_payload,
                source_description=f"environment reference {reference}",
            )
            if reference_placement != group_placement:
                raise ValueError(
                    f"CPU placement in environment reference {reference} "
                    f"disagrees with its group manifest."
                )
            if require_cpu_placement:
                reference_resources = reference_payload.get(
                    "resource_estimates"
                )
                if group_packed_source_summary is not None:
                    reference_performance = (
                        reference_resources.get("performance_telemetry")
                        if isinstance(reference_resources, Mapping)
                        else None
                    )
                    if (
                        not isinstance(reference_performance, Mapping)
                        or reference_performance.get(
                            "packed_source_panel_numa_required"
                        ) is not True
                        or reference_performance.get(
                            "packed_source_panel_numa_complete"
                        ) is not True
                        or reference_performance.get(
                            "packed_source_panel_numa_record_count"
                        ) != group_packed_source_summary["record_count"]
                    ):
                        raise ValueError(
                            f"Contracted packed source-panel NUMA summary in "
                            f"environment reference {reference} is incomplete "
                            "or disagrees with its group manifest."
                        )
                if group_decode_report is not None:
                    expected_decode_summary = (
                        _decode_report_for_output(
                            group_decode_report, include_records=False
                        )
                    )
                    if (
                        not isinstance(reference_resources, Mapping)
                        or reference_resources.get(
                            "numa_bound_bed_decode_required"
                        ) != 1
                        or reference_resources.get(
                            "numa_bound_bed_decode_complete"
                        ) != 1
                        or reference_resources.get("numa_bound_bed_decode")
                        != expected_decode_summary
                    ):
                        raise ValueError(
                            f"Contracted NUMA-bound BED decode summary in "
                            f"environment reference {reference} is incomplete or "
                            "disagrees with its group manifest."
                        )
            resolved_references[environment] = reference

        payload_packed_source_summaries.append(
            group_packed_source_summary
        )

    order = tuple(str(name) for name in environment_order)
    if len(order) != len(set(order)) or set(order) != set(resolved_references):
        raise ValueError(
            "Combined environment order must name every grouped reference exactly once."
        )
    placement_presence = [record is not None for record in payload_placements]
    if any(placement_presence) and not all(placement_presence):
        raise ValueError(
            "Environment-group manifests mix present and absent CPU placement evidence."
        )
    if require_cpu_placement and not all(placement_presence):
        raise ValueError(
            "Authenticated parallel environment groups require complete CPU placement evidence."
        )
    memory_plan_presence = [True for _plan in payload_memory_plans]
    numa_presence = [record is not None for record in payload_numa_groups]
    if require_cpu_placement and not all(numa_presence):
        raise ValueError(
            "Authenticated parallel environment groups require complete early NUMA evidence."
        )
    if expected_groups is not None:
        if not all(placement_presence):
            raise ValueError(
                "Expected CPU groups require complete child CPU placement evidence."
            )
        for index, (expected, placement) in enumerate(
            zip(expected_groups, payload_placements, strict=True)
        ):
            assert placement is not None
            if placement["expected_cpu_ids"] != list(expected):
                raise ValueError(
                    "Child CPU placement disagrees with the controller assignment: "
                    f"group={index}."
                )
    if expected_numa is not None:
        if not all(numa_presence):
            raise ValueError(
                "Expected NUMA groups require complete child early NUMA evidence."
            )
        claimed_numa_nodes: set[int] = set()
        for index, (expected, observed_numa) in enumerate(
            zip(expected_numa, payload_numa_groups, strict=True)
        ):
            assert observed_numa is not None
            if observed_numa != expected:
                raise ValueError(
                    "Child early NUMA attestation disagrees with the controller "
                    f"assignment: group={index}."
                )
            overlap = claimed_numa_nodes & set(observed_numa)
            if overlap:
                raise ValueError(
                    "Environment-group NUMA assignments overlap: "
                    f"group={index}, nodes={sorted(overlap)}."
                )
            claimed_numa_nodes.update(observed_numa)
    packed_source_presence = [
        summary is not None for summary in payload_packed_source_summaries
    ]
    decode_presence = [report is not None for report in payload_decode_reports]
    if require_cpu_placement and (
        not (all(packed_source_presence) or all(decode_presence))
        or (any(packed_source_presence) and not all(packed_source_presence))
        or (any(decode_presence) and not all(decode_presence))
    ):
        raise ValueError(
            "Authenticated environment groups must uniformly publish complete "
            "persistent-panel and/or dense-decode NUMA evidence."
        )
    if all(placement_presence):
        claimed_cpu_ids: set[int] = set()
        for index, placement in enumerate(payload_placements):
            assert placement is not None
            group_cpu_ids = set(placement["expected_cpu_ids"])
            overlap = claimed_cpu_ids & group_cpu_ids
            if overlap:
                raise ValueError(
                    "Environment-group CPU placements overlap: "
                    f"group={index}, CPUs={sorted(overlap)}."
                )
            claimed_cpu_ids.update(group_cpu_ids)
    baseline = payloads[0]
    integrity_resolution_keys = (
        "checksum_recomputed_gemm_output_columns",
        "roundoff_only_gemm_output_columns",
    )
    for payload in payloads:
        if any(key not in payload for key in integrity_resolution_keys):
            raise ValueError(
                "Environment-group integrity resolution counters are incomplete."
            )
    has_integrity_resolution = True
    observed_read_presence = [
        payload.get("observed_genotype_block_reads") is not None
        for payload in payloads
    ]
    if any(observed_read_presence) and not all(observed_read_presence):
        raise ValueError(
            "Environment groups mix present and absent observed genotype-read counts."
        )
    if all(observed_read_presence) and any(
        type(payload["observed_genotype_block_reads"]) is not int
        or payload["observed_genotype_block_reads"] <= 0
        for payload in payloads
    ):
        raise ValueError("Observed genotype-read counts are malformed.")
    call_keys = ("nn", "tn", "total")
    combined_calls = {
        key: int(sum(int(payload["fused_gemm_calls"][key]) for payload in payloads))
        for key in call_keys
    }
    group_records = [
        {
            "manifest": os.path.relpath(source, start=target.parent),
            "environments": [
                str(record["environment"]) for record in payload["references"]
            ],
            "shared_genotype_passes": int(payload["shared_genotype_passes"]),
            "observed_genotype_block_reads": payload.get(
                "observed_genotype_block_reads"
            ),
            "fused_gemm_total_flops": int(payload["fused_gemm_total_flops"]),
            "repaired_gemm_output_columns": int(
                payload["repaired_gemm_output_columns"]
            ),
            **(
                {}
                if not has_integrity_resolution
                else {
                    key: int(payload[key])
                    for key in integrity_resolution_keys
                }
            ),
            "multi_environment_native_end_to_end": payload.get(
                "multi_environment_native_end_to_end"
            ),
            "multi_environment_native_kernel_schema": payload.get(
                "multi_environment_native_kernel_schema"
            ),
            "source_panel_memory_order": payload.get(
                "source_panel_memory_order"
            ),
            "target_panel_memory_order": payload.get(
                "target_panel_memory_order"
            ),
            "target_genotype_memory_order": payload.get(
                "target_genotype_memory_order"
            ),
            "source_to_target_layout_transition": payload.get(
                "source_to_target_layout_transition"
            ),
            "early_numa_attestation": payload.get(
                "early_numa_attestation"
            ),
            **(
                {}
                if decode_report is None
                else {
                    "numa_bound_bed_decode_summary": (
                        _decode_report_for_output(
                            decode_report, include_records=False
                        )
                    )
                }
            ),
            **(
                {}
                if packed_source_summary is None
                else {
                    "packed_source_panel_numa_summary": (
                        packed_source_summary
                    )
                }
            ),
            "modeled_packed_source_panel_gib": payload[
                "modeled_packed_source_panel_gib"
            ],
            "modeled_target_pair_sealing_live_peak_gib": payload[
                "modeled_target_pair_sealing_live_peak_gib"
            ],
            **(
                {}
                if memory_plan is None
                else {"complete_process_memory_plan": memory_plan}
            ),
            "performance_telemetry_summary": (
                None
                if not isinstance(payload.get("performance_telemetry"), Mapping)
                else {
                    key: payload["performance_telemetry"].get(key)
                    for key in (
                        "schema_version",
                        "backend",
                        "arithmetic_dtype",
                        "requested_storage_dtype",
                        "full_precision_layout",
                        "requested_blas_threads",
                        "multi_environment_native_kernel_required",
                        "multi_environment_native_kernel_complete",
                        "multi_environment_native_kernel",
                        "early_numa_attestation",
                        "vendor_call_telemetry_complete",
                        "hot_gemm_telemetry_complete",
                        "optimized_fp64_layout_telemetry_complete",
                        "optimized_fp64_layout_telemetry",
                        "repaired_gemm_output_columns",
                        "checksum_recomputed_gemm_output_columns",
                        "roundoff_only_gemm_output_columns",
                        "telemetry_complete",
                        "phase_telemetry_complete",
                        "capture_boundary",
                        "dropped_gemm_records",
                        "native_telemetry_errors",
                        "gemm_record_count",
                        "gemm_wall_seconds",
                        "gemm_process_cpu_seconds",
                        "gemm_average_active_cores",
                        "gemm_matrix_minutes",
                        "phase_totals",
                        "estimator_phase_totals",
                    )
                }
            ),
        }
        for (
            source,
            payload,
            decode_report,
            packed_source_summary,
            memory_plan,
        ) in zip(
            sources,
            payloads,
            payload_decode_reports,
            payload_packed_source_summaries,
            payload_memory_plans,
            strict=True,
        )
    ]
    combined = {
        "kind": "summit.gxe.multi_environment_reference_batch",
        "schema_version": 1,
        "execution": "parallel_isolated_environment_groups",
        "requested_backend": baseline["requested_backend"],
        "full_precision_layout": payload_layouts[0],
        "protected_native_gemm": baseline["protected_native_gemm"],
        "multi_environment_native_end_to_end": baseline.get(
            "multi_environment_native_end_to_end"
        ),
        "multi_environment_native_kernel_schema": baseline.get(
            "multi_environment_native_kernel_schema"
        ),
        "arithmetic_dtype": baseline.get("arithmetic_dtype"),
        "requested_storage_dtype": baseline.get("requested_storage_dtype"),
        "gemm_backend": baseline.get("gemm_backend"),
        "native_gemm_integrity_enabled": baseline[
            "native_gemm_integrity_enabled"
        ],
        "native_gemm_checksum_enabled": bool(
            baseline.get("native_gemm_checksum_enabled", False)
        ),
        "native_blas_runtime_isolation": baseline[
            "native_blas_runtime_isolation"
        ],
        "repaired_gemm_output_columns": int(
            sum(int(payload["repaired_gemm_output_columns"]) for payload in payloads)
        ),
        **(
            {}
            if not has_integrity_resolution
            else {
                key: int(sum(int(payload[key]) for payload in payloads))
                for key in integrity_resolution_keys
            }
        ),
        "source_panel_memory_order": baseline.get("source_panel_memory_order"),
        "target_panel_memory_order": baseline.get("target_panel_memory_order"),
        "target_genotype_memory_order": baseline.get(
            "target_genotype_memory_order"
        ),
        "source_to_target_layout_transition": baseline.get(
            "source_to_target_layout_transition"
        ),
        "early_numa_attestations": [
            (
                None
                if not isinstance(payload.get("performance_telemetry"), Mapping)
                else payload["performance_telemetry"].get(
                    "early_numa_attestation"
                )
            )
            for payload in payloads
        ],
        "modeled_packed_source_panel_gib": sum_optional_gib(
            "modeled_packed_source_panel_gib"
        ),
        "modeled_target_pair_sealing_live_peak_gib": sum_optional_gib(
            "modeled_target_pair_sealing_live_peak_gib"
        ),
        **(
            {}
            if not all(memory_plan_presence)
            else {
                "complete_process_memory_plans": payload_memory_plans,
                "maximum_modeled_complete_process_peak_bytes": max(
                    int(plan["modeled_complete_process_peak_bytes"])
                    for plan in payload_memory_plans
                    if plan is not None
                ),
                "aggregate_modeled_complete_process_peak_bytes": sum(
                    int(plan["modeled_complete_process_peak_bytes"])
                    for plan in payload_memory_plans
                    if plan is not None
                ),
                "aggregate_resolved_total_process_budget_bytes": sum(
                    int(plan["total_process_budget"]["resolved_bytes"])
                    for plan in payload_memory_plans
                    if plan is not None
                ),
            }
        ),
        "num_environments": len(order),
        "environment_groups": group_records,
        "common_complete_case_samples": int(
            baseline["common_complete_case_samples"]
        ),
        "num_variants": int(baseline["num_variants"]),
        "randomization": baseline["randomization"],
        # Every group makes the same number of passes concurrently.  The
        # aggregate count is recorded separately so the wall-time and I/O
        # interpretations cannot be confused.
        "shared_genotype_passes": int(
            max(int(payload["shared_genotype_passes"]) for payload in payloads)
        ),
        "aggregate_genotype_passes": int(
            sum(int(payload["shared_genotype_passes"]) for payload in payloads)
        ),
        "aggregate_observed_genotype_block_reads": (
            None
            if not all(observed_read_presence)
            else int(
                sum(
                    payload["observed_genotype_block_reads"]
                    for payload in payloads
                )
            )
        ),
        "fused_gemm_calls": combined_calls,
        "fused_gemm_shapes": [
            shape for payload in payloads for shape in payload["fused_gemm_shapes"]
        ],
        "fused_gemm_total_flops": int(
            sum(int(payload["fused_gemm_total_flops"]) for payload in payloads)
        ),
        "modeled_total_resident_sketch_workspace_gib": sum_optional_gib(
            "modeled_total_resident_sketch_workspace_gib"
        ),
        "modeled_transient_sketch_peak_gib": sum_optional_gib(
            "modeled_transient_sketch_peak_gib"
        ),
        "peak_process_rss_gib_at_manifest": float(
            max(float(payload["peak_process_rss_gib_at_manifest"]) for payload in payloads)
        ),
        "peak_rss_scope": "maximum_group_process_lifetime_ru_maxrss_at_group_manifest",
        "references": [
            {
                "environment": environment,
                "reference": os.path.relpath(
                    resolved_references[environment], start=target.parent
                ),
            }
            for environment in order
        ],
    }
    if all(placement_presence):
        combined["cpu_placements"] = [
            dict(record) for record in payload_placements if record is not None
        ]
        combined["cpu_placement_complete"] = True
    if require_cpu_placement:
        if all(packed_source_presence):
            combined["packed_source_panel_numa_required"] = True
            combined["packed_source_panel_numa_complete"] = True
            combined["packed_source_panel_numa_groups"] = [
                summary
                for summary in payload_packed_source_summaries
                if summary is not None
            ]
        if all(decode_presence):
            if not all(
                isinstance(report, Mapping) and report.get("complete") is True
                for report in payload_decode_reports
            ):
                raise ValueError(
                    "Contracted environment groups require complete NUMA-bound "
                    "BED decode evidence."
                )
            combined["numa_bound_bed_decode_required"] = True
            combined["numa_bound_bed_decode_complete"] = True
            combined["numa_bound_bed_decode_groups"] = [
                _decode_report_for_output(report, include_records=False)
                for report in payload_decode_reports
                if report is not None
            ]
    _publish_json_no_replace(combined, target)
    return target


def generate_multi_environment_references(
    estimators: Sequence[GenomewideEnvLDScore],
    *,
    batch_manifest: str | Path,
    requested_backend: str = "python",
    full_precision_layout: str = "current",
) -> Path:
    """Generate independent references while sharing every genotype block read."""
    estimators = tuple(estimators)
    _require_common_contract(estimators)
    first = estimators[0]
    temporary_names = (
        "_native_missingness_targets",
        "_native_missingness_sinks",
        "_shared_missingness_estimators",
        "_native_parallel_standardization",
        "_native_numa_bound_decode_nodes",
        "_native_numa_bound_iid_index",
        "_native_numa_bound_decode_records",
    )
    missing = object()
    previous_temporary_state = {
        name: getattr(first, name, missing) for name in temporary_names
    }
    for estimator in estimators:
        estimator.performance_phase_timings = {}
    with _MultiEnvironmentGemm(
        requested_backend,
        estimators[0],
        full_precision_layout=full_precision_layout,
    ) as executor:
        try:
            return _generate_multi_environment_references(
                estimators,
                batch_manifest=batch_manifest,
                requested_backend=requested_backend,
                executor=executor,
            )
        finally:
            # These attributes bind one shared standardization result to all
            # independent environment diagnostics.  Restore the caller's
            # state exactly so estimator reuse cannot inherit this run mode.
            for name, previous in previous_temporary_state.items():
                if previous is missing:
                    if hasattr(first, name):
                        delattr(first, name)
                else:
                    setattr(first, name, previous)


def _generate_multi_environment_references(
    estimators: Sequence[GenomewideEnvLDScore],
    *,
    batch_manifest: str | Path,
    requested_backend: str,
    executor: _MultiEnvironmentGemm,
) -> Path:
    first = estimators[0]
    first._shared_missingness_estimators = tuple(estimators)
    for estimator in estimators:
        if executor.cpu_placement is not None:
            estimator.cpu_placement = dict(executor.cpu_placement)
            estimator.cpu_placement_complete = True
    target = Path(batch_manifest).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing existing multi-environment manifest: {target}.")
    for estimator in estimators:
        estimator._assert_output_paths_available()

    first._assert_construction_genotype_state()
    blocks = first._make_compute_blocks()
    common_basis, environment_directions = _environment_directions(estimators)
    feature_environment_constants = []
    for index, estimator in enumerate(estimators):
        direction = environment_directions[:, index]
        environment_squared = np.asarray(
            estimator.env * estimator.env, dtype=np.float64
        )
        feature_environment_constants.append(
            (
                float(direction.sum(dtype=np.float64)),
                np.einsum(
                    "ni,n->i", estimator.C_int, direction, optimize=False
                ),
                environment_squared,
                float(
                    np.einsum(
                        "n,n,n->",
                        environment_squared,
                        direction,
                        direction,
                        optimize=False,
                    )
                ),
            )
        )
    first.log._log(
        "[gxe:multi] Sharing each standardized genotype block across "
        f"{len(estimators)} independent environments: "
        f"{[estimator.env_name for estimator in estimators]}."
    )
    if executor.protected:
        first._native_parallel_standardization = True
        centered_targets = []
        missingness_sinks = []
        for estimator in estimators:
            for target_values, attribute_name in (
                (
                    estimator.env,
                    "genotype_missing_environment_correlation",
                ),
                (
                    estimator.pheno,
                    "genotype_missing_phenotype_correlation",
                ),
            ):
                if target_values is None:
                    continue
                centered = np.asarray(target_values, dtype=np.float64)
                centered_targets.append(centered - centered.mean(dtype=np.float64))
                missingness_sinks.append((estimator, attribute_name))
        if not centered_targets:
            raise RuntimeError(
                "Shared native GxE standardization requires at least one "
                "missingness diagnostic target."
            )
        first._native_missingness_targets = np.asfortranarray(
            np.column_stack(centered_targets), dtype=np.float64
        )
        first._native_missingness_sinks = tuple(missingness_sinks)
        if executor.checksum_enabled:
            first.log._log(
                "[gxe:multi] Shared decoded blocks use double-precision native "
                "serialized-entry GEMMs with checksum detection and deterministic repair."
            )
        else:
            first.log._log(
                "[gxe:multi] The adaptive native pipeline uses the isolated "
                "private-BLIS runtime without redundant checksum recomputation; "
                "immutable-input, NUMA, placement, and output guards remain active."
            )

    # Freeze the authenticated decode contract before constructing the native
    # descriptor owner.  Contracted runs allocate, bind, populate, and verify
    # every decoded block wholly inside C++; unplaced runs retain an
    # uncontracted native allocation without making locality claims.
    bound_decode_nodes = None
    feature_states: list[dict] = []
    with executor.phase(
        "feature_plan_construction",
        {
            "environment_tile": [0, len(estimators)],
            "environment_names": [estimator.env_name for estimator in estimators],
        },
    ):
        algebraic_feature_plan = (
            _build_algebraic_feature_plan(estimators) if executor.protected else None
        )
        planning_feature_plan = (
            algebraic_feature_plan
            if algebraic_feature_plan is not None
            else _build_algebraic_feature_plan(estimators)
        )
        environment_tiles, vtiles, complete_memory_plan = (
            _shared_execution_tiles(
                estimators,
                executor.compute_dtype,
                protected=executor.protected,
                blocks=blocks,
                feature_plan=planning_feature_plan,
                common_rank=int(common_basis.shape[1]),
                # Only checksum recomputation owns the modeled integrity
                # scratch. Private BLIS keeps the immutable-input/output/NUMA
                # safety envelope while deliberately disabling that redundant
                # arithmetic.
                integrity_enabled=bool(executor.checksum_enabled),
                vendor_probe_chunk_width=executor.vendor_probe_chunk_width,
            )
        )
        for estimator in estimators:
            estimator._vtiles_used = list(vtiles)
            estimator.resource_estimates["complete_process_memory_plan"] = dict(
                complete_memory_plan
            )
        if algebraic_feature_plan is not None:
            direct_context_info = executor.initialize_multi_environment_direct_context(
                algebraic_feature_plan,
                common_basis,
                environment_directions,
                estimators,
                blocks,
                environment_tiles,
                vtiles,
            )
            if executor._numa_bound_bed_decode_required:
                bound_decode_nodes = _enable_contracted_numa_bound_decode(
                    first, executor
                )
            if direct_context_info is None:
                if bound_decode_nodes is None:
                    bound_decode_nodes = _enable_contracted_numa_bound_decode(
                        first, executor
                    )
                executor.initialize_multi_environment_kernel(
                    algebraic_feature_plan,
                    common_basis,
                    environment_directions,
                    estimators,
                )
        if executor._multi_environment_direct_context is None:
            feature_states = [
                _new_feature_state(estimator) for estimator in estimators
            ]
    native_direct_result = None
    if executor._multi_environment_direct_context is not None:
        native_direct_result = executor.run_multi_environment_direct_context()
        if bound_decode_nodes is not None:
            native_decode_records = native_direct_result.get(
                "numa_bound_decode_records"
            )
            if not isinstance(native_decode_records, list):
                raise RuntimeError(
                    "The descriptor context omitted contracted BED decode evidence."
                )
            first._native_numa_bound_decode_records = list(
                native_decode_records
            )
    fuse_feature_source = bool(
        executor.protected
        and algebraic_feature_plan is not None
        and len(environment_tiles) == 1
        and len(vtiles) == 1
    )
    passes_per_tile = 2
    passes = (
        2
        if fuse_feature_source
        else 1 + len(environment_tiles) * len(vtiles) * passes_per_tile
    )
    if passes != int(complete_memory_plan["planned_genotype_passes"]):
        raise RuntimeError(
            "The execution pass count disagrees with the complete memory plan."
        )
    fused_packed_sources = None
    if fuse_feature_source and native_direct_result is None:
        environment_start, environment_stop = environment_tiles[0]
        probe_start, probe_count = vtiles[0]
        columns = first.nbins * probe_count
        fused_packed_sources = np.zeros(
            (
                first.nsamp,
                (environment_stop - environment_start) * 2 * columns,
            ),
            dtype=executor.compute_dtype,
            order="F",
        )
        # Each feature block determines the exact source scales for the same
        # variants. Expose those in-progress arrays so the source contribution
        # can be accumulated before releasing the decoded genotype block.
        for estimator, state in zip(estimators, feature_states, strict=True):
            estimator.inv_sqrt_resvar_x_all = state["inv_x"]
            estimator.inv_sqrt_resvar_w_all = state["inv_w"]
        first.log._log(
            "[gxe:multi] Fusing feature and random-source construction into "
            "one genotype pass."
        )
    if algebraic_feature_plan is not None:
        first.log._log(
            "[gxe:multi:feature] Computing packed algebraic projected moments "
            "without materializing environment-specific X/W feature blocks."
        )
    feature_blocks = blocks
    if native_direct_result is not None:
        native_features = {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in dict(native_direct_result["features"]).items()
        }
        native_missing_counts = np.asarray(
            native_direct_result["missing_counts"], dtype=np.int64
        )
        native_missing_correlations = np.asarray(
            native_direct_result["missing_environment_correlations"],
            dtype=np.float64,
        )
        if (
            native_missing_counts.shape != (first.nsnps,)
            or native_missing_correlations.shape
            != (first.nsnps, len(estimators))
        ):
            raise RuntimeError(
                "The descriptor context returned malformed missingness diagnostics."
            )
        for environment_index, estimator in enumerate(estimators):
            estimator.genotype_missing_call_count[:] = native_missing_counts
            estimator.genotype_missing_environment_correlation[:] = (
                native_missing_correlations[:, environment_index]
            )
        feature_map = {
            "inv_x": "scale_x",
            "inv_w": "scale_w",
            "norm_x": "norm_x",
            "norm_w": "norm_w",
            "diag_x": "diag_nxe_x",
            "diag_w": "diag_nxe_w",
            "corr_xw": "corr_xw",
        }
        for environment_index, estimator in enumerate(estimators):
            state = {
                state_key: native_features[native_key][:, environment_index]
                for state_key, native_key in feature_map.items()
            }
            for state_key, native_key in feature_map.items():
                observed = native_features[native_key]
                if observed.shape != (first.nsnps, len(estimators)):
                    raise RuntimeError(
                        "The descriptor context returned malformed feature arrays."
                    )
            state["max_leak_x"] = float(
                native_features["max_projection_leakage_additive"][
                    environment_index
                ]
            )
            feature_states.append(state)
            state["max_leak_w"] = float(
                native_features["max_projection_leakage_interaction"][
                    environment_index
                ]
            )
        feature_blocks = ()
    feature_started = time.perf_counter()
    for block_index, (start, stop) in enumerate(feature_blocks, start=1):
        block_context = {
            "genotype_block": [start, stop],
            "genotype_block_index": block_index - 1,
            "genotype_block_width": stop - start,
            "probe_tile": None,
        }
        with executor.phase(
            "genotype_read_decode_standardization", block_context
        ):
            genotype = first._read_genotype_block(start, stop)
        if algebraic_feature_plan is not None:
            feature_context = dict(block_context)
            feature_context.update(
                {
                    "environment_tile": [0, len(estimators)],
                    "environment_names": [
                        estimator.env_name for estimator in estimators
                    ],
                }
            )
            with executor.phase("feature_moments", feature_context):
                if executor._multi_environment_kernel is not None:
                    _accumulate_native_feature_block(
                        estimators,
                        feature_states,
                        genotype,
                        start,
                        stop,
                        executor,
                        semantic=feature_context,
                    )
                else:
                    _accumulate_algebraic_feature_block(
                        estimators,
                        feature_states,
                        genotype,
                        start,
                        stop,
                        algebraic_feature_plan,
                        executor,
                    )
        else:
            for environment_start, environment_stop in environment_tiles:
                feature_context = dict(block_context)
                feature_context.update(
                    {
                        "environment_tile": [
                            environment_start, environment_stop
                        ],
                        "environment_names": [
                            estimator.env_name
                            for estimator in estimators[
                                environment_start:environment_stop
                            ]
                        ],
                    }
                )
                with executor.phase("feature_moments", feature_context):
                    _accumulate_fused_feature_block(
                        estimators,
                        feature_states,
                        genotype,
                        start,
                        stop,
                        environment_start,
                        environment_stop,
                        common_basis,
                        environment_directions,
                        feature_environment_constants,
                        executor,
                    )
        if fuse_feature_source:
            assert fused_packed_sources is not None
            environment_start, environment_stop = environment_tiles[0]
            probe_start, probe_count = vtiles[0]
            columns = first.nbins * probe_count
            source_context = dict(block_context)
            source_context.update(
                {
                    "environment_tile": [environment_start, environment_stop],
                    "environment_names": [
                        estimator.env_name
                        for estimator in estimators[
                            environment_start:environment_stop
                        ]
                    ],
                    "probe_tile": [probe_start, probe_count],
                }
            )
            with executor.phase("random_generation", source_context):
                probes = first._generate_random_block(
                    L=stop - start,
                    v_count=probe_count,
                    blk_start=start,
                    v_start=probe_start,
                )
            # Annotation weights stay canonical binary64; only the retained
            # sketch storage may round under the estimator dtype option.
            annotation = np.asarray(
                first.annot[start:stop], dtype=np.float64
            )
            with executor.semantic_context(source_context):
                _accumulate_packed_source_block(
                    estimators,
                    environment_start,
                    environment_stop,
                    fused_packed_sources,
                    genotype,
                    probes,
                    annotation,
                    start,
                    stop,
                    columns,
                    executor,
                )
            del probes, annotation
        del genotype
        _log_phase_progress(
            first.log, "feature", block_index, len(blocks), feature_started
        )
    with executor.phase("feature_finalization_fp64"):
        for estimator, state in zip(estimators, feature_states, strict=True):
            _finish_feature_state(estimator, state)
    del feature_states
    del feature_environment_constants
    algebraic_feature_basis_bytes = (
        0
        if algebraic_feature_plan is None
        else int(algebraic_feature_plan["basis"].nbytes)
    )
    del algebraic_feature_plan
    gc.collect()

    max_vt = max(size for _, size in vtiles)
    itemsize = executor.compute_dtype.itemsize
    # One source panel contains both X/W families. Dense BLIS retains exactly
    # one additional environment-weighted panel; Mailman generates that view.
    # This remains a panel-only bound, not the complete process bound.
    resident_multiplier = 2
    max_environment_tile = max(
        stop - start for start, stop in environment_tiles
    )
    packed_panel_bytes = (
        resident_multiplier
        * max_environment_tile
        * first.nsamp
        * first.nbins
        * max_vt
        * itemsize
    )
    dense_blas_hybrid = bool(
        complete_memory_plan.get("direct_kernel_mode") == "dense_blas_hybrid"
    )
    total_resident = packed_panel_bytes * (2 if dense_blas_hybrid else 1)
    target_pair_sealing_live_peak_bytes = total_resident
    total_transient_peak = target_pair_sealing_live_peak_bytes
    max_block = min(first.step_size, first.nsnps)
    for estimator in estimators:
        estimator.resource_estimates = {
            "native_direct_backend": int(executor.protected),
            "multi_environment_shared_decode": 1,
            "multi_environment_protected_gemm": int(executor.protected),
            "multi_environment_native_end_to_end": int(
                executor._multi_environment_kernel is not None
                or native_direct_result is not None
            ),
            "multi_environment_descriptor_owned_end_to_end": int(
                native_direct_result is not None
            ),
            "multi_environment_native_kernel_schema": (
                None
                if executor._multi_environment_kernel is None
                and native_direct_result is None
                else "summit.multi_environment_native_kernel.v1"
            ),
            "multi_environment_direct_context_schema": (
                "summit.multi_environment_direct_context.v3"
                if native_direct_result is not None
                else None
            ),
            "native_gemm_integrity_enabled": int(executor.integrity_enabled),
            "native_gemm_checksum_enabled": int(executor.checksum_enabled),
            "performance_telemetry_schema_version": (
                _PERFORMANCE_TELEMETRY_SCHEMA_VERSION
            ),
            "arithmetic_dtype": executor.arithmetic_dtype.name,
            "requested_storage_dtype": executor.storage_dtype.name,
            "full_precision_layout": executor.full_precision_layout,
            "source_panel_memory_order": "F",
            "target_panel_memory_order": "F",
            "target_genotype_memory_order": "F",
            "source_to_target_layout_transition": "none",
            "modeled_packed_source_panel_gib": float(
                packed_panel_bytes / 1024**3
            ),
            "modeled_target_pair_sealing_live_peak_gib": float(
                target_pair_sealing_live_peak_bytes / 1024**3
            ),
            "modeled_panel_memory_bound_scope": (
                (
                    "source_plus_persistent_environment_weighted_target_panel; "
                    if dense_blas_hybrid
                    else "single_persistent_source_panel_with_generated_weighted_view; "
                )
                + "excludes packed genotype, feature/output arrays, vendor "
                "workspace, allocator retention, and publication buffers"
            ),
            "complete_process_memory_plan": complete_memory_plan,
            "gemm_backend": executor.backend_name,
            "multi_environment_count": len(estimators),
            "multi_environment_tiles": len(environment_tiles),
            "multi_environment_max_tile_size": max_environment_tile,
            "shared_genotype_passes": passes,
            "observed_genotype_block_reads": (
                None
                if native_direct_result is None
                else int(
                    native_direct_result["observed_genotype_block_reads"]
                )
            ),
            "decoded_genotype_block_gib": float(
                first.nsamp * max_block * 8 / 1024**3
            ),
            "prepared_feature_pair_gib": float(
                2 * first.nsamp * max_block * itemsize / 1024**3
            ),
            "packed_algebraic_feature_basis_gib": float(
                algebraic_feature_basis_bytes / 1024**3
            ),
            "resident_sketch_workspace_gib": float(
                resident_multiplier * (2 if dense_blas_hybrid else 1)
                * first.nsamp * first.nbins * max_vt * itemsize / 1024**3
            ),
            "multi_environment_total_resident_sketch_workspace_gib": float(
                total_resident / 1024**3
            ),
            "multi_environment_transient_sketch_peak_gib": float(
                total_transient_peak / 1024**3
            ),
            "source_columns": int(first.nbins * max_vt),
            "actual_global_2b_source_columns": int(2 * first.nbins * max_vt),
            "target_source_columns": int(2 * first.nbins * max_vt),
            "blas_threads": int(first.num_threads),
            "bed_reader_threads": int(first.decode_threads),
        }
    first.log._log(
        "[gxe:multi:resources] modeled packed source panel="
        f"{packed_panel_bytes / 1024**3:.3f} GiB; panel-only transient "
        f"peak={total_transient_peak / 1024**3:.3f} GiB; "
        f"environment_tiles={environment_tiles}; v_tiles={vtiles}."
    )

    native_accumulator_matrices = None
    if native_direct_result is not None:
        native_scores = {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in dict(native_direct_result["scores"]).items()
        }
        expected_score_shape = (
            len(estimators) * first.nsnps,
            first.nbins,
        )
        if set(native_scores) != set(_SCORE_NAMES) or any(
            value.shape != expected_score_shape
            for value in native_scores.values()
        ):
            raise RuntimeError(
                "The descriptor context returned malformed score accumulators."
            )
        accumulators = [
            {
                name: native_scores[name][
                    environment_index * first.nsnps:
                    (environment_index + 1) * first.nsnps
                ]
                for name in _SCORE_NAMES
            }
            for environment_index in range(len(estimators))
        ]
    elif executor._multi_environment_kernel is not None:
        native_accumulator_storage = np.zeros(
            (len(_SCORE_NAMES), len(estimators), first.nsnps, first.nbins),
            dtype=np.float64,
            order="C",
        )
        accumulators = [
            {
                name: native_accumulator_storage[family_index, environment_index]
                for family_index, name in enumerate(_SCORE_NAMES)
            }
            for environment_index in range(len(estimators))
        ]
        native_accumulator_matrices = {
            name: native_accumulator_storage[family_index].reshape(
                len(estimators) * first.nsnps, first.nbins
            )
            for family_index, name in enumerate(_SCORE_NAMES)
        }
        if not all(
            value.flags.c_contiguous
            for value in native_accumulator_matrices.values()
        ):
            raise RuntimeError(
                "Native multi-environment score accumulators lost C contiguity."
            )
    else:
        accumulators = [
            {
                name: np.zeros((first.nsnps, first.nbins), dtype=np.float64)
                for name in _SCORE_NAMES
            }
            for _ in estimators
        ]
    population_enabled = first.nvecs >= 2
    population_probe_square_sums = (
        [
            np.zeros((estimator.nsamp, 2 * estimator.nbins), dtype=np.float64)
            for estimator in estimators
        ]
        if population_enabled and native_direct_result is None
        else []
    )
    population_same_probe_products = (
        [
            np.zeros((2 * estimator.nbins, 2 * estimator.nbins), dtype=np.float64)
            for estimator in estimators
        ]
        if population_enabled and native_direct_result is None
        else []
    )
    if population_enabled and native_direct_result is None:
        for index, estimator in enumerate(estimators):
            estimator.resource_estimates["population_trace_workspace_gib"] = float(
                population_probe_square_sums[index].nbytes
                + population_same_probe_products[index].nbytes
            ) / 1024**3
    execution_vtiles = () if native_direct_result is not None else vtiles
    for probe_start, probe_count in execution_vtiles:
        columns = first.nbins * probe_count
        for environment_start, environment_stop in environment_tiles:
            tile_size = environment_stop - environment_start
            tile_context = {
                "environment_tile": [environment_start, environment_stop],
                "environment_names": [
                    estimator.env_name
                    for estimator in estimators[
                        environment_start:environment_stop
                    ]
                ],
                "probe_tile": [probe_start, probe_count],
            }
            if fuse_feature_source:
                assert (environment_start, environment_stop) == environment_tiles[0]
                assert (probe_start, probe_count) == vtiles[0]
                assert fused_packed_sources is not None
                packed_sources = fused_packed_sources
                fused_packed_sources = None
            else:
                packed_sources = np.zeros(
                    (first.nsamp, tile_size * 2 * columns),
                    dtype=executor.compute_dtype,
                    order="F",
                )
                source_started = time.perf_counter()
                for block_index, (start, stop) in enumerate(blocks, start=1):
                    block_context = dict(tile_context)
                    block_context.update(
                        {
                            "genotype_block": [start, stop],
                            "genotype_block_index": block_index - 1,
                            "genotype_block_width": stop - start,
                        }
                    )
                    with executor.phase(
                        "genotype_read_decode_standardization", block_context
                    ):
                        genotype = first._read_genotype_block(start, stop)
                    with executor.phase("random_generation", block_context):
                        probes = first._generate_random_block(
                            L=stop - start,
                            v_count=probe_count,
                            blk_start=start,
                            v_start=probe_start,
                        )
                    # Annotation weights stay canonical binary64; only the
                    # retained sketch storage may round under the estimator
                    # dtype option.
                    annotation = np.asarray(
                        first.annot[start:stop], dtype=np.float64
                    )
                    with executor.semantic_context(block_context):
                        _accumulate_packed_source_block(
                            estimators,
                            environment_start,
                            environment_stop,
                            packed_sources,
                            genotype,
                            probes,
                            annotation,
                            start,
                            stop,
                            columns,
                            executor,
                        )
                    del genotype, probes, annotation
                    _log_phase_progress(
                        first.log, "source", block_index, len(blocks), source_started
                    )
            with executor.phase(
                "projection_context_correction", tile_context
            ):
                leakages = _project_packed_sources_inplace(
                    estimators,
                    environment_start,
                    environment_stop,
                    packed_sources,
                    columns,
                    common_basis,
                    environment_directions,
                    executor,
                )
            for local_index, estimator in enumerate(
                estimators[environment_start:environment_stop]
            ):
                estimator.resource_estimates[
                    "max_source_projection_leakage"
                ] = max(
                    float(
                        estimator.resource_estimates.get(
                            "max_source_projection_leakage", 0.0
                        )
                    ),
                    leakages[local_index],
                )
                if executor.protected:
                    estimator.resource_estimates[
                        "max_native_source_projection_leakage"
                    ] = estimator.resource_estimates[
                        "max_source_projection_leakage"
                    ]
                if population_enabled:
                    index = environment_start + local_index
                    segment = slice(
                        local_index * 2 * columns,
                        (local_index + 1) * 2 * columns,
                    )
                    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
                    estimator._accumulate_population_diagonal_moments(
                        packed_sources[:, segment],
                        probe_count,
                        population_probe_square_sums[index],
                        population_same_probe_products[index],
                    )
                    executor.record_phase_elapsed(
                        "fp64_same_person_accumulation", phase_wall, phase_cpu
                    )

            if executor.protected:
                phase_wall, phase_cpu = time.perf_counter(), time.process_time()
                row_weights = np.asfortranarray(
                    np.column_stack(
                        [
                            estimator.env
                            for estimator in estimators[
                                environment_start:environment_stop
                            ]
                        ]
                    ),
                    dtype=np.float64,
                )
                protected_sources = executor.prepare_row_weighted_pair(
                    packed_sources, row_weights
                )
                executor.record_phase_elapsed(
                    "target_operand_construction", phase_wall, phase_cpu
                )
                del packed_sources, row_weights
                environment_weighted_sources = None
            else:
                protected_sources = None
                phase_wall, phase_cpu = time.perf_counter(), time.process_time()
                environment_weighted_sources = np.empty_like(packed_sources)
                for local_index, estimator in enumerate(
                    estimators[environment_start:environment_stop]
                ):
                    segment = slice(
                        local_index * 2 * columns,
                        (local_index + 1) * 2 * columns,
                    )
                    environment_weighted_sources[:, segment] = (
                        estimator.env[:, None] * packed_sources[:, segment]
                    )
                executor.record_phase_elapsed(
                    "target_operand_construction", phase_wall, phase_cpu
                )

            target_started = time.perf_counter()
            for block_index, (start, stop) in enumerate(blocks, start=1):
                block_context = dict(tile_context)
                block_context.update(
                    {
                        "genotype_block": [start, stop],
                        "genotype_block_index": block_index - 1,
                        "genotype_block_width": stop - start,
                    }
                )
                phase_wall, phase_cpu = time.perf_counter(), time.process_time()
                genotype = first._read_genotype_block(start, stop)
                executor.record_phase_elapsed(
                    "genotype_read_decode_standardization", phase_wall, phase_cpu
                )
                gemm_semantic = dict(block_context, phase="target_gemm")
                phase_wall, phase_cpu = time.perf_counter(), time.process_time()
                with executor.semantic_context(gemm_semantic):
                    if executor._multi_environment_kernel is not None:
                        assert protected_sources is not None
                        assert native_accumulator_matrices is not None
                        scale_x = np.asfortranarray(
                            np.column_stack(
                                [
                                    estimator.inv_sqrt_resvar_x_all[start:stop]
                                    for estimator in estimators[
                                        environment_start:environment_stop
                                    ]
                                ]
                            ),
                            dtype=np.float64,
                        )
                        scale_w = np.asfortranarray(
                            np.column_stack(
                                [
                                    estimator.inv_sqrt_resvar_w_all[start:stop]
                                    for estimator in estimators[
                                        environment_start:environment_stop
                                    ]
                                ]
                            ),
                            dtype=np.float64,
                        )
                        executor.native_target_score_block(
                            genotype,
                            protected_sources,
                            scale_x,
                            scale_w,
                            native_accumulator_matrices,
                            block_start=start,
                            total_variants=first.nsnps,
                            probe_count=probe_count,
                            environment_start=environment_start,
                            semantic=gemm_semantic,
                        )
                        packed_x = packed_w = None
                    elif executor.protected:
                        assert protected_sources is not None
                        packed_x, packed_w = executor.tn_pair(
                            genotype, protected_sources
                        )
                    else:
                        assert environment_weighted_sources is not None
                        packed_x = np.asarray(
                            executor.tn(genotype, packed_sources), dtype=np.float64
                        )
                        packed_w = np.asarray(
                            executor.tn(genotype, environment_weighted_sources),
                            dtype=np.float64,
                        )
                executor.record_phase_elapsed(
                    "target_gemm", phase_wall, phase_cpu
                )
                if executor._multi_environment_kernel is None:
                    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
                    for local_index, estimator in enumerate(
                        estimators[environment_start:environment_stop]
                    ):
                        index = environment_start + local_index
                        segment = slice(
                            local_index * 2 * columns,
                            (local_index + 1) * 2 * columns,
                        )
                        work_x = packed_x[:, segment]
                        work_w = packed_w[:, segment]
                        work_x *= estimator.inv_sqrt_resvar_x_all[
                            start:stop
                        ].reshape(-1, 1)
                        work_w *= estimator.inv_sqrt_resvar_w_all[
                            start:stop
                        ].reshape(-1, 1)
                        estimator._accumulate_left_scores(
                            work_x[:, :columns], accumulators[index]["xx"],
                            start, stop, probe_count,
                        )
                        estimator._accumulate_left_scores(
                            work_x[:, columns:2 * columns], accumulators[index]["xw"],
                            start, stop, probe_count,
                        )
                        estimator._accumulate_left_scores(
                            work_w[:, :columns], accumulators[index]["wx"],
                            start, stop, probe_count,
                        )
                        estimator._accumulate_left_scores(
                            work_w[:, columns:2 * columns], accumulators[index]["ww"],
                            start, stop, probe_count,
                        )
                    executor.record_phase_elapsed(
                        "fp64_score_reductions", phase_wall, phase_cpu
                    )
                    del packed_x, packed_w
                else:
                    del scale_x, scale_w
                del genotype
                _log_phase_progress(
                    first.log, "target", block_index, len(blocks), target_started
                )
            if executor.protected:
                del protected_sources
            else:
                del environment_weighted_sources, packed_sources
            gc.collect()

    if bound_decode_nodes is not None:
        decode_report = _validated_numa_bound_decode_report(
            getattr(first, "_native_numa_bound_decode_records", None),
            blocks=blocks,
            passes=passes,
            sample_count=int(first.nsamp),
            num_variants=int(first.nsnps),
            selected_nodes=bound_decode_nodes,
        )
        executor._numa_bound_bed_decode_report = decode_report
        decode_summary = _decode_report_for_output(
            decode_report, include_records=False
        )
        for estimator in estimators:
            estimator.resource_estimates[
                "numa_bound_bed_decode_required"
            ] = 1
            estimator.resource_estimates[
                "numa_bound_bed_decode_complete"
            ] = 1
            estimator.resource_estimates[
                "numa_bound_bed_decode"
            ] = decode_summary
        first.log._log(
            "[gxe:multi:numa-decode] Verified every page for "
            f"{decode_report['observed_block_read_count']} bounded BED block "
            f"read(s) on NUMA nodes {list(bound_decode_nodes)}."
        )

    for estimator in estimators:
        estimator.resource_estimates["fused_gemm_nn_calls"] = int(
            executor.nn_calls
        )
        estimator.resource_estimates["fused_gemm_tn_calls"] = int(
            executor.tn_calls
        )
        estimator.resource_estimates["fused_gemm_total_calls"] = int(
            executor.nn_calls + executor.tn_calls
        )
    first._assert_construction_genotype_state()
    if executor.protected:
        for estimator in estimators:
            estimator.resource_estimates[
                "native_repaired_gemm_output_columns"
            ] = int(executor.repaired_output_columns)
            estimator.resource_estimates[
                "native_checksum_recomputed_gemm_output_columns"
            ] = int(
                executor._multi_environment_kernel_checksum_recomputed_columns
            )
            estimator.resource_estimates[
                "native_roundoff_only_gemm_output_columns"
            ] = int(
                executor._multi_environment_kernel_roundoff_only_columns
            )
            estimator.resource_estimates[
                "native_retried_gemm_input_mutations"
            ] = 0
        first.log._log(
            "[gxe:multi:integrity] checksum-recomputed output columns="
            f"{executor._multi_environment_kernel_checksum_recomputed_columns}; "
            "roundoff-only columns="
            f"{executor._multi_environment_kernel_roundoff_only_columns}; "
            "materially repaired columns="
            f"{executor.repaired_output_columns}; input mutation policy=abort."
        )
    if population_enabled:
        if native_direct_result is not None:
            population = np.asarray(
                native_direct_result["population_same_individual_products"],
                dtype=np.float64,
            )
            families = 2 * first.nbins
            if population.shape != (len(estimators) * families, families):
                raise RuntimeError(
                    "The descriptor context returned malformed population moments."
                )
            for index, estimator in enumerate(estimators):
                estimator.population_same_individual_products = np.array(
                    population[index * families:(index + 1) * families],
                    copy=True,
                )
        else:
            for index, estimator in enumerate(estimators):
                with executor.phase(
                    "fp64_same_person_finalization",
                    {"environment": estimator.env_name},
                ):
                    estimator.population_same_individual_products = (
                        estimator._finalize_population_diagonal_moments(
                            population_probe_square_sums[index],
                            population_same_probe_products[index],
                        )
                    )
    peak_rss_before_output = _peak_process_rss_gib()
    gemm_shapes = executor.gemm_shape_records()
    fused_gemm_flops = int(sum(record["flops"] for record in gemm_shapes))
    for estimator in estimators:
        estimator.resource_estimates["peak_process_rss_gib_before_output"] = float(
            peak_rss_before_output
        )
        estimator.resource_estimates["peak_rss_scope"] = (
            "process_lifetime_ru_maxrss_before_reference_output"
        )
        estimator.resource_estimates["fused_gemm_shapes"] = gemm_shapes
        estimator.resource_estimates["fused_gemm_total_flops"] = fused_gemm_flops
    # Accumulators are the final score arrays.  Normalize them in place rather
    # than retaining an equally large divided copy for every environment
    # throughout bundle publication.
    with executor.phase("fp64_score_normalization"):
        if native_direct_result is not None:
            pass
        elif executor._multi_environment_kernel is not None:
            assert native_accumulator_matrices is not None
            executor.native_normalize_scores(
                native_accumulator_matrices, first.nvecs
            )
        else:
            for estimator, accumulator in zip(
                estimators, accumulators, strict=True
            ):
                inverse_probe_count = 1.0 / float(estimator.nvecs)
                for value in accumulator.values():
                    value *= inverse_probe_count

    compact_performance = executor.performance_report(
        estimator_phase_timings=_aggregate_estimator_phase_timings(estimators),
        include_records=False,
        capture_boundary="pre_output_bundle_staging",
        phase_telemetry_complete=False,
    )
    protected_prepublication_checks = {
        "vendor_call_telemetry_complete": compact_performance[
            "vendor_call_telemetry_complete"
        ],
        "hot_gemm_telemetry_complete": compact_performance[
            "hot_gemm_telemetry_complete"
        ],
        "native_gemm_output_numa_evidence_complete": compact_performance[
            "native_gemm_output_numa_evidence_complete"
        ],
        "optimized_fp64_layout_telemetry_complete": compact_performance[
            "optimized_fp64_layout_telemetry_complete"
        ],
        "multi_environment_native_kernel_complete": compact_performance[
            "multi_environment_native_kernel_complete"
        ],
        "multi_environment_direct_context_complete": compact_performance[
            "multi_environment_direct_context_complete"
        ],
        "packed_source_panel_numa_complete": compact_performance[
            "packed_source_panel_numa_complete"
        ],
        "numa_bound_bed_decode_complete": (
            not executor._numa_bound_bed_decode_required
            or compact_performance.get("numa_bound_bed_decode_complete") is True
        ),
    }
    failed_prepublication_checks = sorted(
        key
        for key, value in protected_prepublication_checks.items()
        if value is not True
    )
    if executor.protected and failed_prepublication_checks:
        raise RuntimeError(
            "Protected GxE performance telemetry or layout semantics are incomplete; "
            "refusing to publish an accepted report: failed_checks="
            f"{failed_prepublication_checks}."
        )
    for estimator in estimators:
        estimator.resource_estimates["performance_telemetry"] = compact_performance

    references = []
    published: list[tuple[Path, int, int]] = []
    try:
        for index, estimator in enumerate(estimators):
            scores = accumulators[index]
            output_context = {
                "environment": estimator.env_name,
                "environment_index": index,
            }
            with executor.phase(
                "output_bundle_end_to_end", output_context
            ):
                estimator._compute_ldscore(
                    compute_callback=(
                        lambda estimator=estimator, scores=scores:
                        estimator._finalize_ldscore_outputs(scores)
                    ),
                )
            with executor.phase("output_validation", output_context):
                published.extend(
                    _published_file_identity(path.resolve())
                    for path in estimator._planned_output_paths()
                )
                reference = Path(f"{estimator.outpath}.gxe.ref.json").resolve()
            references.append(
                {
                    "environment": estimator.env_name,
                    "reference": os.path.relpath(reference, start=target.parent),
                }
            )

        final_performance = executor.performance_report(
            estimator_phase_timings=_aggregate_estimator_phase_timings(
                estimators
            ),
            include_records=True,
            capture_boundary=(
                "post_output_artifact_publication_pre_batch_manifest"
            ),
            phase_telemetry_complete=True,
        )
        if executor.protected and not final_performance["telemetry_complete"]:
            raise RuntimeError(
                "Protected GxE final performance telemetry is incomplete; "
                "refusing to publish the batch manifest."
            )
        payload = {
            "kind": "summit.gxe.multi_environment_reference_batch",
            "schema_version": 1,
            "execution": (
                "descriptor_owned_native_multi_environment_end_to_end"
                if native_direct_result is not None
                else "shared_native_multi_environment_end_to_end"
                if executor._multi_environment_kernel is not None
                else "shared_in_memory_decoded_blocks"
            ),
            "multi_environment_native_end_to_end": bool(
                executor._multi_environment_kernel is not None
                or native_direct_result is not None
            ),
            "multi_environment_descriptor_owned_end_to_end": bool(
                native_direct_result is not None
            ),
            "multi_environment_native_kernel_schema": (
                None
                if executor._multi_environment_kernel is None
                and native_direct_result is None
                else "summit.multi_environment_native_kernel.v1"
            ),
            "multi_environment_direct_context_schema": (
                "summit.multi_environment_direct_context.v3"
                if native_direct_result is not None
                else None
            ),
            "requested_backend": str(requested_backend),
            "full_precision_layout": executor.full_precision_layout,
            "protected_native_gemm": bool(executor.protected),
            "arithmetic_dtype": executor.arithmetic_dtype.name,
            "requested_storage_dtype": executor.storage_dtype.name,
            "gemm_backend": executor.backend_name,
            "native_gemm_integrity_enabled": bool(executor.integrity_enabled),
            "native_gemm_checksum_enabled": bool(executor.checksum_enabled),
            "native_blas_runtime_isolation": executor.runtime_isolation,
            "repaired_gemm_output_columns": int(
                executor.repaired_output_columns
            ),
            "checksum_recomputed_gemm_output_columns": int(
                executor._multi_environment_kernel_checksum_recomputed_columns
            ),
            "roundoff_only_gemm_output_columns": int(
                executor._multi_environment_kernel_roundoff_only_columns
            ),
            "source_panel_memory_order": "F",
            "target_panel_memory_order": "F",
            "target_genotype_memory_order": "F",
            "source_to_target_layout_transition": "none",
            "early_numa_attestation": final_performance.get(
                "early_numa_attestation"
            ),
            "complete_process_memory_plan": complete_memory_plan,
            "modeled_packed_source_panel_gib": float(
                packed_panel_bytes / 1024**3
            ),
            "modeled_target_pair_sealing_live_peak_gib": float(
                target_pair_sealing_live_peak_bytes / 1024**3
            ),
            "num_environments": len(estimators),
            "environment_tiles": [list(tile) for tile in environment_tiles],
            "common_complete_case_samples": first.nsamp,
            "num_variants": first.nsnps,
            "randomization": {
                "distribution": first.rand_dist,
                "num_vectors": first.nvecs,
                "seed": first.root_seed,
                "probe_offset": first.probe_offset,
                "probe_tiles": [list(tile) for tile in vtiles],
            },
            "shared_genotype_passes": passes,
            "observed_genotype_block_reads": (
                None
                if native_direct_result is None
                else int(
                    native_direct_result["observed_genotype_block_reads"]
                )
            ),
            "fused_gemm_calls": {
                "nn": int(executor.nn_calls),
                "tn": int(executor.tn_calls),
                "total": int(executor.nn_calls + executor.tn_calls),
            },
            "fused_gemm_shapes": gemm_shapes,
            "fused_gemm_total_flops": fused_gemm_flops,
            "modeled_total_resident_sketch_workspace_gib": float(
                total_resident / 1024**3
            ),
            "modeled_transient_sketch_peak_gib": float(
                total_transient_peak / 1024**3
            ),
            "peak_process_rss_gib_at_manifest": _peak_process_rss_gib(),
            "peak_rss_scope": "process_lifetime_ru_maxrss_at_batch_manifest",
            "performance_telemetry": final_performance,
            "references": references,
        }
        if executor.cpu_placement is not None:
            if executor.cpu_placement_complete is not True:
                raise RuntimeError(
                    "Refusing to publish incomplete group CPU placement evidence."
                )
            payload["cpu_placement"] = dict(executor.cpu_placement)
            payload["cpu_placement_complete"] = True
            if native_direct_result is not None:
                source_evidence = final_performance.get(
                    "packed_source_panel_numa_records"
                )
                if (
                    final_performance.get(
                        "packed_source_panel_numa_required"
                    ) is not True
                    or final_performance.get(
                        "packed_source_panel_numa_complete"
                    ) is not True
                    or not isinstance(source_evidence, list)
                    or not source_evidence
                ):
                    raise RuntimeError(
                        "Refusing to publish incomplete packed source-panel "
                        "NUMA evidence."
                    )
                payload["packed_source_panel_numa_required"] = True
                payload["packed_source_panel_numa_complete"] = True
                payload["packed_source_panel_numa"] = {
                    "schema": _PACKED_SOURCE_PANEL_NUMA_SCHEMA,
                    "record_count": len(source_evidence),
                    "records_included": False,
                    "complete": True,
                }
            if (
                native_direct_result is None
                or executor._numa_bound_bed_decode_required
            ):
                decode_evidence = final_performance.get(
                    "numa_bound_bed_decode"
                )
                if (
                    final_performance.get(
                        "numa_bound_bed_decode_required"
                    ) is not True
                    or final_performance.get(
                        "numa_bound_bed_decode_complete"
                    ) is not True
                    or not isinstance(decode_evidence, Mapping)
                    or decode_evidence.get("records_included") is not True
                    or decode_evidence.get("complete") is not True
                ):
                    raise RuntimeError(
                        "Refusing to publish incomplete contracted NUMA-bound "
                        "BED decode evidence."
                    )
                payload["numa_bound_bed_decode_required"] = True
                payload["numa_bound_bed_decode_complete"] = True
                payload["numa_bound_bed_decode"] = _decode_report_for_output(
                    decode_evidence, include_records=False
                )
        first._assert_construction_genotype_state()
        _publish_json_no_replace(payload, target)
    except Exception:
        _rollback_published_files(published)
        raise
    return target
