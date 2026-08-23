from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from scripts.gxe import benchmark_current_process_topologies as benchmark


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT_A = "a" * 40


def _controller_report(*args):
    watchdog = benchmark._SweepDeadlineWatchdog.start(
        args[0].sweep_timeout_seconds
    )
    try:
        return benchmark._controller(*args, watchdog)
    finally:
        watchdog.finish_execution("after focused controller test")
        watchdog.cancel_join_close()


def _cpu_contract(cpus: tuple[int, ...]) -> dict:
    records = []
    for cpu in cpus:
        package = 0 if cpu < 100 else 1
        core = cpu if package == 0 else cpu - 100
        records.append(
            {
                "cpu": cpu,
                "package": package,
                "core": core,
                "numa_node": package,
            }
        )
    return {
        "ordered_cpu_ids": list(cpus),
        "cpu_list": benchmark.EXACT._compress_ints(cpus),
        "records": records,
        "numa_nodes": sorted({record["numa_node"] for record in records}),
        "one_hardware_thread_per_physical_core": True,
        "within_controller_affinity": True,
    }


def _topologies() -> list[benchmark.Topology]:
    one = tuple(range(32))
    halves = (tuple(range(16)), tuple(range(16, 32)))
    sockets = (one, tuple(range(100, 132)))
    quarters = tuple(tuple(range(start, start + 8)) for start in range(0, 32, 8))
    return [
        benchmark.Topology("1x32", 32, (one,), (_cpu_contract(one),), (5,), True, None),
        benchmark.Topology(
            "2x16_same_socket",
            16,
            halves,
            tuple(_cpu_contract(group) for group in halves),
            (3, 2),
            True,
            None,
        ),
        benchmark.Topology(
            "2x32_two_socket",
            32,
            sockets,
            tuple(_cpu_contract(group) for group in sockets),
            (3, 2),
            True,
            None,
        ),
        benchmark.Topology(
            "4x8_same_socket",
            8,
            quarters,
            tuple(_cpu_contract(group) for group in quarters),
            (2, 1, 1, 1),
            False,
            "kernel_only_until_one_environment_production_worker_merge_oracle_exists",
        ),
    ]


def _topology_args() -> argparse.Namespace:
    return argparse.Namespace(
        one_by_32_cpus=tuple(range(32)),
        two_by_16_cpus=[tuple(range(16)), tuple(range(16, 32))],
        two_by_32_cpus=[tuple(range(32)), tuple(range(100, 132))],
        four_by_8_cpus=[tuple(range(start, start + 8)) for start in range(0, 32, 8)],
    )


def _native_contract_build(cpus: tuple[int, ...]) -> dict:
    return {
        "api_version": 9,
        "backend_version": "1.9",
        "blas_vendor": "BLIS",
        "blas_runtime_config": "BLIS 2.0 config=zen",
        "blas_runtime_corename": "zen",
        "gemm_integrity_minimum_vendor_flops": (
            benchmark.EXACT.GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
        ),
        "native_integrity_snapshot_numa_contract_supported": True,
        "native_integrity_snapshot_numa_contract_schema": (
            benchmark.EXACT.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA
        ),
        "native_integrity_snapshot_numa_query_chunk_page_limit": (
            benchmark.EXACT.NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
        ),
        "native_gemm_output_numa_contract_supported": True,
        "native_gemm_output_numa_contract_schema": (
            benchmark.EXACT.NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
        ),
        "native_gemm_output_numa_query_chunk_page_limit": (
            benchmark.EXACT.NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
        ),
        "native_gemm_output_numa_evidence_capacity": (
            benchmark.EXACT.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        ),
        "openmp_placement_contract_evidence": {
            "expected_cpu_ids": list(cpus)
        },
    }


def _mapping_fields(logical_bytes: int, node: int) -> dict:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = (logical_bytes - 1) // page_size + 1
    chunks = (
        pages + benchmark.EXACT.NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT - 1
    ) // benchmark.EXACT.NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
    return {
        "mapping_bytes": pages * page_size,
        "page_size": page_size,
        "mapping_page_count": pages,
        "selected_nodes": [node],
        "queried_pages": pages,
        "resolved_pages": pages,
        "query_chunks": chunks,
        "query_chunk_page_limit": (
            benchmark.EXACT.NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT
        ),
        "node_histogram": {str(node): pages},
        "ordered_status_sha256": (
            benchmark.EXACT._ordered_single_node_status_sha256(node, pages)
        ),
    }


def _output_evidence(
    case: benchmark.EXACT.Case, node: int, call_id: int
) -> dict:
    rows, columns = case.output_shape
    logical_bytes = rows * columns * 8
    return {
        "schema": benchmark.EXACT.NATIVE_GEMM_OUTPUT_NUMA_SCHEMA,
        "schema_version": 1,
        "applicable": True,
        "operand_role": "protected_gemm_output",
        "contract_required": True,
        "complete": True,
        "call_id": call_id,
        "logical_rows": rows,
        "logical_columns": columns,
        "storage_layout": "column_major",
        "logical_byte_count": logical_bytes,
        "allocation_mode": "mmap_private_anonymous",
        **_mapping_fields(logical_bytes, node),
        "policy_mode": "bind_static_nodes",
        "policy_mode_value": benchmark.EXACT.NATIVE_STATIC_MEMBIND_POLICY_VALUE,
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


def _snapshot_evidence(case: benchmark.EXACT.Case, node: int) -> dict:
    value = {
        "schema": benchmark.EXACT.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA,
        "schema_version": 1,
        "operand_role": "native_integrity_snapshot_of_logical_b",
        "integrity_check_shape_eligible": case.operation == "source",
        "integrity_check_executed": case.operation == "source",
        "snapshot_available": case.operation == "source",
        "contract_required": True,
        "complete": case.operation == "source",
    }
    if case.operation != "source":
        return value
    logical_bytes = case.cblas["k"] * case.cblas["n"] * 8
    value.update(
        {
            "logical_byte_count": logical_bytes,
            **_mapping_fields(logical_bytes, node),
            "policy_mode": "bind_static_nodes",
            "policy_mode_value": benchmark.EXACT.NATIVE_STATIC_MEMBIND_POLICY_VALUE,
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
    )
    return value


def _operand_numa(record: dict, name: str, node: int) -> dict:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    byte_count = benchmark.EXACT._operand_byte_count(record, name)
    pages = byte_count // page_size
    selected = min(pages, 8)
    samples = []
    for ordinal in range(selected):
        page_index = 0 if selected == 1 else ordinal * (pages - 1) // (selected - 1)
        samples.append(
            {
                "sample_ordinal": ordinal,
                "full_page_index": page_index,
                "byte_offset_from_operand_start": page_index * page_size,
                "status_kind": "numa_node",
                "raw_move_pages_status": node,
                "numa_node": node,
                "page_query_errno": None,
            }
        )
    return {
        "query_status": "queried",
        "storage_span_pages": (byte_count - 1) // page_size + 1,
        "fully_contained_pages": pages,
        "operand_byte_count": byte_count,
        "operand_start_address_page_offset": 0,
        "operand_end_exclusive_address_page_offset": byte_count % page_size,
        "selected_sample_pages": selected,
        "resolved_sample_pages": selected,
        "page_query_error_pages": 0,
        "node_histogram": {str(node): selected},
        "page_error_errno_histogram": {},
        "ordered_samples": samples,
    }


def _telemetry(
    case: benchmark.EXACT.Case,
    *,
    cpu: int,
    node: int,
    sequence: int,
    wall: float,
) -> dict:
    active = float(case.threads)
    record = {
        **case.cblas,
        "schema_version": 1,
        "sequence": sequence,
        "arithmetic_dtype": "float64",
        "alpha": 1.0,
        "beta": 0.0,
        "flop_count": float(case.flops),
        "wall_seconds": wall,
        "process_cpu_seconds": wall * active,
        "gflops_per_second": float(case.flops) / (wall * 1.0e9),
        "process_cpu_to_wall_ratio": active,
        "active_core_equivalents": active,
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
        "entry_cpu": cpu,
        "exit_cpu": cpu,
        "cpu_affinity_count": 1,
        "cpu_affinity_list": str(cpu),
        "backend": "BLIS",
        "backend_config": "BLIS 2.0 config=zen",
        "backend_corename": "zen",
        "completed": True,
        "native_integrity_snapshot_numa": _snapshot_evidence(case, node),
        "native_gemm_output_numa": _output_evidence(case, node, sequence),
    }
    record["operand_numa_page_samples"] = {
        "schema_version": 1,
        "sampling_method": "move_pages_query_no_migration",
        "sampling_timing": "after_vendor_call_outside_timed_interval",
        "sample_limit_per_operand": 8,
        "address_selection_schema_version": 1,
        "address_selection_policy": benchmark.EXACT.NUMA_ADDRESS_SELECTION_POLICY,
        "selected_addresses_are_page_bases": True,
        "partial_boundary_pages_included": False,
        "first_and_last_fully_contained_pages_selected": True,
        "virtual_addresses_exposed": False,
        "operand_byte_range_semantics": "[start_address,end_exclusive_address)",
        "address_evidence": (
            "ordered_samples_with_operand_relative_byte_offsets_and_full_page_indices"
        ),
        "system_page_size": int(os.sysconf("SC_PAGE_SIZE")),
        "syscall_result": 0,
        "syscall_errno": 0,
        "operands": {
            name: _operand_numa(record, name, node) for name in ("a", "b", "c")
        },
    }
    return record


def _telemetry_status(sequence: int) -> dict:
    return {
        "schema_version": 1,
        "capacity": 16_384,
        "buffered_records": 0,
        "captured_records": 1,
        "dropped_records": 0,
        "next_sequence": sequence + 1,
        "operand_numa_sampling_method": "move_pages_query_no_migration",
        "operand_numa_sample_limit_per_operand": 8,
        "operand_numa_address_selection_schema_version": 1,
        "operand_numa_address_selection_policy": (
            benchmark.EXACT.NUMA_ADDRESS_SELECTION_POLICY
        ),
        "operand_numa_partial_boundary_pages_included": False,
    }


def _output_status(call_id: int, *, reset: bool) -> dict:
    return {
        "schema_version": 1,
        "capacity": benchmark.EXACT.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY,
        "buffered_records": 0,
        "captured_records": 0 if reset else 1,
        "dropped_records": 0,
        "next_call_id": call_id if reset else call_id + 1,
        "attempted_calls": 0 if reset else 1,
        "verified_calls": 0 if reset else 1,
        "legacy_calls": 0,
        "failed_calls": 0,
        "query_chunk_page_limit": benchmark.EXACT.NATIVE_NUMA_QUERY_CHUNK_PAGE_LIMIT,
    }


def test_fixed_topology_contract_is_disjoint_socket_exact_and_fair(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(benchmark.EXACT, "_cpu_contract", _cpu_contract)
    topologies = benchmark._topology_contracts(_topology_args())
    assert [value.topology_id for value in topologies] == [
        "1x32",
        "2x16_same_socket",
        "2x32_two_socket",
        "4x8_same_socket",
    ]
    assert [
        (value.process_count, value.threads_per_process) for value in topologies
    ] == [
        (1, 32),
        (2, 16),
        (2, 32),
        (4, 8),
    ]
    assert [value.environment_tiles for value in topologies] == [
        (5,),
        (3, 2),
        (3, 2),
        (2, 1, 1, 1),
    ]
    assert set(topologies[1].cpu_groups[0]) | set(topologies[1].cpu_groups[1]) == set(
        topologies[0].cpu_groups[0]
    )
    assert topologies[2].cpu_groups[0] == topologies[0].cpu_groups[0]
    assert topologies[3].production_promotable is False
    assert set().union(*map(set, topologies[3].cpu_groups)) == set(
        topologies[0].cpu_groups[0]
    )

    damaged = _topology_args()
    damaged.two_by_16_cpus = [tuple(range(16)), tuple(range(15, 31))]
    with pytest.raises(RuntimeError, match="overlap"):
        benchmark._topology_contracts(damaged)

    damaged = _topology_args()
    damaged.two_by_32_cpus = [tuple(range(32)), tuple(range(32, 64))]
    with pytest.raises(RuntimeError, match="two distinct sockets"):
        benchmark._topology_contracts(damaged)

    damaged = _topology_args()
    damaged.one_by_32_cpus = tuple(reversed(range(32)))
    with pytest.raises(RuntimeError, match="strictly increasing"):
        benchmark._topology_contracts(damaged)


def test_five_environment_case_widths_and_flops_are_apples_to_apples():
    topologies = _topologies()
    expected_widths = {
        "source": [[320], [192, 128], [192, 128], [128, 64, 64, 64]],
        "target": [[640], [384, 256], [384, 256], [256, 128, 128, 128]],
    }
    for operation in ("source", "target"):
        observed_widths = []
        total_flops = []
        for topology in topologies:
            cases = [
                benchmark.EXACT.Case(
                    operation,
                    benchmark.EXACT.DEFAULT_N,
                    benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
                    benchmark.EXACT.DEFAULT_PROBE_TILE,
                    environment_tile,
                    topology.threads_per_process,
                )
                for environment_tile in topology.environment_tiles
            ]
            observed_widths.append([case.panel_columns for case in cases])
            total_flops.append(sum(case.flops for case in cases))
        assert observed_widths == expected_widths[operation]
        assert len(set(total_flops)) == 1


def _bound_operand_evidence(case: benchmark.EXACT.Case, node: int) -> dict:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))

    def one(shape: tuple[int, int]) -> dict:
        byte_count = math.prod(shape) * 8
        mapping_bytes = math.ceil(byte_count / page_size) * page_size
        page_count = mapping_bytes // page_size
        common = {
            "schema": "summit.numa_bound_anonymous_buffer.v1",
            "schema_version": 1,
            "byte_count": byte_count,
            "mapping_bytes": mapping_bytes,
            "page_size": page_size,
            "page_count": page_count,
            "selected_nodes": [node],
            "policy_mode": "bind_static_nodes",
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "range_policy_verified": True,
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }
        return {
            "allocation": {
                **common,
                "post_decode_complete_page_query": False,
            },
            "verification": {
                **common,
                "post_decode_complete_page_query": True,
                "post_decode_strict_policy_verified": True,
                "queried_pages": page_count,
                "resolved_pages": page_count,
                "query_chunks": math.ceil(page_count / 65_536),
                "query_chunk_page_limit": 65_536,
                "node_histogram": {str(node): page_count},
                "ordered_status_sha256": "a" * 64,
                "ordered_status_encoding": (
                    f"native_32bit_signed_{sys.byteorder}"
                ),
                "complete": True,
            },
        }

    right_shape = (
        (case.block_width, case.panel_columns)
        if case.operation == "source"
        else (case.n_samples, case.panel_columns)
    )
    return {
        "schema": "summit.gxe.exact_harness_bound_operands.v1",
        "schema_version": 1,
        "selected_nodes": [node],
        "verification_boundary": "post_fill_pre_ready",
        "left": one((case.n_samples, case.block_width)),
        "right": one(right_shape),
        "complete": True,
    }


def _worker_and_events(
    topology: benchmark.Topology,
    operation: str,
) -> tuple[list[dict], list[dict[int, dict]], list[list[int]]]:
    cases = [
        benchmark.EXACT.Case(
            operation,
            benchmark.EXACT.DEFAULT_N,
            benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
            benchmark.EXACT.DEFAULT_PROBE_TILE,
            environment_tile,
            topology.threads_per_process,
        )
        for environment_tile in topology.environment_tiles
    ]
    workers = []
    events = [
        dict()
        for _ in range(benchmark.EXACT.WARMUPS + benchmark.EXACT.MEASURED_REPEATS)
    ]
    releases = []
    for ordinal in range(len(events)):
        releases.append(
            [
                10_000_000_000 + ordinal * 1_000_000_000 + group
                for group in range(topology.process_count)
            ]
        )
    for group_index in range(topology.process_count):
        case = cases[group_index]
        calls = []
        output_hash = f"{group_index + 1:x}" * 64
        cpu_id = topology.cpu_groups[group_index][0]
        node = topology.cpu_contracts[group_index]["numa_nodes"][0]
        build = _native_contract_build(topology.cpu_groups[group_index])
        input_numa = _bound_operand_evidence(case, node)
        input_numa_gate = benchmark.EXACT._validate_bound_operand_evidence(
            input_numa, case, [node]
        )
        for ordinal in range(len(events)):
            wall = 0.100 + 0.010 * group_index
            cpu = wall * topology.threads_per_process
            entry = 20_000_000_000 + ordinal * 1_000_000_000 + group_index * 1_000_000
            exit_value = entry + int(wall * 1.0e9)
            rate = case.flops / (wall * 1.0e9)
            sequence = ordinal + 1
            telemetry = _telemetry(
                case,
                cpu=cpu_id,
                node=node,
                sequence=sequence,
                wall=wall,
            )
            telemetry_status = _telemetry_status(sequence)
            telemetry_gate = benchmark.EXACT._validate_telemetry(
                telemetry,
                case,
                topology.cpu_groups[group_index],
                [node],
                build,
                measured=ordinal >= benchmark.EXACT.WARMUPS,
            )
            output_evidence = telemetry["native_gemm_output_numa"]
            reset_status = _output_status(sequence, reset=True)
            post_status = _output_status(sequence, reset=False)
            native_gate = (
                benchmark.EXACT._validate_protected_call_native_numa_contract(
                    telemetry=telemetry,
                    output_evidence=output_evidence,
                    reset_status=reset_status,
                    post_status=post_status,
                    case=case,
                    nodes=[node],
                    expected_call_id=sequence,
                    integrity_minimum_vendor_flops=(
                        benchmark.EXACT.GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
                    ),
                )
            )
            call = {
                "ordinal": ordinal,
                "phase": "measured" if ordinal >= benchmark.EXACT.WARMUPS else "warmup",
                "phase_ordinal": (
                    ordinal - benchmark.EXACT.WARMUPS
                    if ordinal >= benchmark.EXACT.WARMUPS
                    else ordinal
                ),
                "wrapper_entry_monotonic_ns": entry,
                "wrapper_exit_monotonic_ns": exit_value,
                "output_sha256_storage_order": output_hash,
                "telemetry": telemetry,
                "telemetry_status": telemetry_status,
                "telemetry_gate": telemetry_gate,
                "native_gemm_output_numa_evidence": output_evidence,
                "native_gemm_output_numa_reset_status": reset_status,
                "native_gemm_output_numa_status": post_status,
                "native_gemm_output_numa_gate": native_gate["output"],
                "native_integrity_snapshot_numa_evidence": telemetry[
                    "native_integrity_snapshot_numa"
                ],
                "native_integrity_snapshot_numa_gate": native_gate[
                    "integrity_snapshot"
                ],
                "protected_call_native_numa_gate": native_gate,
            }
            calls.append(call)
            events[ordinal][group_index] = {
                "event": "call_done",
                "pid": 5000 + group_index,
                "topology_id": topology.topology_id,
                "group_index": group_index,
                "operation": operation,
                "ordinal": ordinal,
                "phase": call["phase"],
                "wrapper_entry_monotonic_ns": entry,
                "wrapper_exit_monotonic_ns": exit_value,
                "vendor_wall_seconds": wall,
                "vendor_process_cpu_seconds": cpu,
                "active_core_equivalents": topology.threads_per_process,
                "gflops_per_second": rate,
                "output_sha256_storage_order": output_hash,
            }
        workers.append(
            {
                "status": "accepted",
                "accepted": True,
                "pid": 5000 + group_index,
                "topology_id": topology.topology_id,
                "group_index": group_index,
                "operation": operation,
                "case_id": case.case_id,
                "case": benchmark.asdict(case),
                "cblas": case.cblas,
                "flops_per_call": case.flops,
                "cpu_contract": topology.cpu_contracts[group_index],
                "native": {"module_sha256": SHA_A, "build_info": build},
                "inputs": {"group_specific": group_index},
                "input_numa_bound_buffers": input_numa,
                "input_numa_bound_buffers_gate": input_numa_gate,
                "output": {"sha256_storage_order": output_hash},
                "oracle": {
                    "passed": True,
                    "full_output_compared": True,
                    "oracle_sha256_storage_order": f"{group_index + 5:x}" * 64,
                },
                "calls": calls,
            }
        )
    return workers, events, releases


def test_synchronized_aggregate_reports_makespan_rate_core_use_and_stragglers():
    topology = _topologies()[1]
    configuration = benchmark.Configuration(topology, "source")
    workers, events, releases = _worker_and_events(topology, "source")
    result = benchmark._aggregate_configuration(
        configuration,
        workers,
        events,
        releases,
        benchmark.MAX_START_SKEW_SECONDS,
    )
    assert result["passed"] is True
    assert result["barrier_wave_count"] == 8
    assert result["cross_worker_input_output_oracle_equality_required"] is False
    assert [value["panel_columns"] for value in result["worker_cases"]] == [192, 128]
    measured = result["measured_summary"]
    assert measured["wrapper_makespan_seconds"]["median"] == pytest.approx(0.111)
    assert measured["aggregate_active_core_equivalents"]["median"] == pytest.approx(
        (1.6 + 1.76) / 0.111
    )
    wave = result["waves"][-1]
    assert wave["worker_start_skew_seconds"] == pytest.approx(0.001)
    assert wave["straggler"]["slowest_over_fastest_ratio"] == pytest.approx(1.1)
    assert wave["aggregate_gflops_per_second"] > max(
        wave["per_worker_gflops_per_second"]
    )

    skewed = copy.deepcopy(events)
    skewed[0][1]["wrapper_entry_monotonic_ns"] += 300_000_000
    skewed[0][1]["wrapper_exit_monotonic_ns"] += 300_000_000
    skewed_workers = copy.deepcopy(workers)
    skewed_workers[1]["calls"][0]["wrapper_entry_monotonic_ns"] += 300_000_000
    skewed_workers[1]["calls"][0]["wrapper_exit_monotonic_ns"] += 300_000_000
    with pytest.raises(RuntimeError, match="start skew"):
        benchmark._aggregate_configuration(
            configuration,
            skewed_workers,
            skewed,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    damaged = copy.deepcopy(events)
    damaged[3][1]["output_sha256_storage_order"] = "0" * 64
    with pytest.raises(RuntimeError, match="differs from final call"):
        benchmark._aggregate_configuration(
            configuration,
            workers,
            damaged,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )


def test_controller_reconstructs_all_eight_topology_operation_configurations():
    observed = []
    for topology in _topologies():
        for operation in ("source", "target"):
            configuration = benchmark.Configuration(topology, operation)
            workers, events, releases = _worker_and_events(topology, operation)
            gate = benchmark._aggregate_configuration(
                configuration,
                workers,
                events,
                releases,
                benchmark.MAX_START_SKEW_SECONDS,
            )
            assert gate["passed"] is True
            assert gate["barrier_wave_count"] == 8
            observed.append(configuration.config_id)
    assert observed == [
        configuration.config_id
        for configuration in benchmark._configurations(_topologies())
    ]


def test_controller_rejects_missing_mutated_or_cross_call_native_numa_evidence():
    topology = _topologies()[0]
    configuration = benchmark.Configuration(topology, "source")
    workers, events, releases = _worker_and_events(topology, "source")

    missing = copy.deepcopy(workers)
    missing[0]["calls"][0].pop("native_gemm_output_numa_evidence")
    with pytest.raises(RuntimeError, match="lacks native output NUMA evidence"):
        benchmark._aggregate_configuration(
            configuration,
            missing,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    mutated_output = copy.deepcopy(workers)
    mutated_output[0]["calls"][0]["native_gemm_output_numa_evidence"][
        "logical_rows"
    ] += 1
    with pytest.raises(RuntimeError, match="output evidence mismatch"):
        benchmark._aggregate_configuration(
            configuration,
            mutated_output,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    queue_vendor_mismatch = copy.deepcopy(workers)
    nested = copy.deepcopy(
        queue_vendor_mismatch[0]["calls"][0]["telemetry"][
            "native_gemm_output_numa"
        ]
    )
    nested["ordered_status_sha256"] = "0" * 64
    queue_vendor_mismatch[0]["calls"][0]["telemetry"][
        "native_gemm_output_numa"
    ] = nested
    with pytest.raises(RuntimeError, match="differs from nested vendor evidence"):
        benchmark._aggregate_configuration(
            configuration,
            queue_vendor_mismatch,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    bad_status = copy.deepcopy(workers)
    bad_status[0]["calls"][0]["native_gemm_output_numa_status"][
        "verified_calls"
    ] = 0
    with pytest.raises(RuntimeError, match="output status mismatch"):
        benchmark._aggregate_configuration(
            configuration,
            bad_status,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    bad_reset_id = copy.deepcopy(workers)
    bad_reset_id[0]["calls"][0]["native_gemm_output_numa_reset_status"][
        "next_call_id"
    ] = 2
    with pytest.raises(RuntimeError, match="reset status mismatch"):
        benchmark._aggregate_configuration(
            configuration,
            bad_reset_id,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    bad_call_id = copy.deepcopy(workers)
    bad_call_id[0]["calls"][0]["native_gemm_output_numa_evidence"][
        "call_id"
    ] = 2
    with pytest.raises(RuntimeError, match="output evidence mismatch"):
        benchmark._aggregate_configuration(
            configuration,
            bad_call_id,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    for counter in ("dropped_records", "legacy_calls", "failed_calls"):
        damaged_counter = copy.deepcopy(workers)
        damaged_counter[0]["calls"][0]["native_gemm_output_numa_status"][
            counter
        ] = 1
        with pytest.raises(RuntimeError, match="output status mismatch"):
            benchmark._aggregate_configuration(
                configuration,
                damaged_counter,
                events,
                releases,
                benchmark.MAX_START_SKEW_SECONDS,
            )

    bad_snapshot = copy.deepcopy(workers)
    bad_snapshot[0]["calls"][0]["telemetry"][
        "native_integrity_snapshot_numa"
    ]["logical_byte_count"] += 8
    with pytest.raises(RuntimeError, match="native integrity snapshot mismatch"):
        benchmark._aggregate_configuration(
            configuration,
            bad_snapshot,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )

    bad_build_type = copy.deepcopy(workers)
    bad_build_type[0]["native"]["build_info"][
        "native_gemm_output_numa_contract_supported"
    ] = 1
    with pytest.raises(RuntimeError, match="native NUMA build contract"):
        benchmark._aggregate_configuration(
            configuration,
            bad_build_type,
            events,
            releases,
            benchmark.MAX_START_SKEW_SECONDS,
        )


def test_target_snapshot_discriminator_is_exact_and_rejects_extra_claims():
    case = benchmark.EXACT.Case(
        "target",
        benchmark.EXACT.DEFAULT_N,
        benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
        benchmark.EXACT.DEFAULT_PROBE_TILE,
        5,
        32,
    )
    evidence = _snapshot_evidence(case, 0)
    gate = benchmark.EXACT._validate_native_integrity_snapshot_numa_evidence(
        evidence,
        case,
        [0],
        integrity_minimum_vendor_flops=(
            benchmark.EXACT.GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
        ),
    )
    assert gate["discriminator_only"] is True
    damaged = dict(evidence, logical_byte_count=8)
    with pytest.raises(RuntimeError, match="discriminator is not exact"):
        benchmark.EXACT._validate_native_integrity_snapshot_numa_evidence(
            damaged,
            case,
            [0],
            integrity_minimum_vendor_flops=(
                benchmark.EXACT.GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
            ),
        )


def test_below_threshold_source_snapshot_uses_exact_discriminator():
    case = benchmark.EXACT.Case("source", 10, 10, 1, 1, 1)
    evidence = {
        "schema": benchmark.EXACT.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA,
        "schema_version": 1,
        "operand_role": "native_integrity_snapshot_of_logical_b",
        "integrity_check_shape_eligible": False,
        "integrity_check_executed": False,
        "snapshot_available": False,
        "contract_required": True,
        "complete": False,
    }
    gate = benchmark.EXACT._validate_native_integrity_snapshot_numa_evidence(
        evidence,
        case,
        [0],
        integrity_minimum_vendor_flops=(
            benchmark.EXACT.GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
        ),
    )
    assert gate["discriminator_only"] is True
    damaged = dict(evidence, integrity_check_shape_eligible=True)
    with pytest.raises(RuntimeError, match="native integrity snapshot mismatch"):
        benchmark.EXACT._validate_native_integrity_snapshot_numa_evidence(
            damaged,
            case,
            [0],
            integrity_minimum_vendor_flops=(
                benchmark.EXACT.GEMM_INTEGRITY_MINIMUM_VENDOR_FLOPS
            ),
        )


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("mapping_page_count", "mapping/query geometry mismatch"),
        ("query_chunks", "mapping/query geometry mismatch"),
        ("node_histogram", "mapping/query geometry mismatch"),
        ("ordered_status_sha256", "ordered status digest mismatch"),
    ],
)
def test_output_mapping_geometry_chunk_histogram_and_digest_tamper_rejected(
    field: str, expected_error: str
):
    case = benchmark.EXACT.Case(
        "target",
        benchmark.EXACT.DEFAULT_N,
        benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
        benchmark.EXACT.DEFAULT_PROBE_TILE,
        5,
        32,
    )
    evidence = _output_evidence(case, 0, 1)
    if field == "node_histogram":
        evidence[field] = {"0": evidence["mapping_page_count"] - 1}
    elif field == "ordered_status_sha256":
        evidence[field] = "0" * 64
    else:
        evidence[field] += 1
    with pytest.raises(RuntimeError, match=expected_error):
        benchmark.EXACT._validate_native_gemm_output_numa_evidence(
            evidence, case, [0], expected_call_id=1
        )


def test_valid_multi_node_output_retains_canonical_unrecomputed_digest():
    case = benchmark.EXACT.Case(
        "target",
        benchmark.EXACT.DEFAULT_N,
        benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
        benchmark.EXACT.DEFAULT_PROBE_TILE,
        5,
        32,
    )
    evidence = _output_evidence(case, 0, 1)
    pages = evidence["mapping_page_count"]
    evidence["selected_nodes"] = [0, 1]
    evidence["node_histogram"] = {"0": pages // 2, "1": pages - pages // 2}
    evidence["ordered_status_sha256"] = "a" * 64
    gate = benchmark.EXACT._validate_native_gemm_output_numa_evidence(
        evidence, case, [0, 1], expected_call_id=1
    )
    assert gate["mapping"][
        "ordered_status_digest_independently_recomputed"
    ] is False
    assert gate["mapping"]["ordered_status_sha256"] == "a" * 64


def test_standalone_layout_controller_revalidates_and_rejects_shape_tamper(
    monkeypatch: pytest.MonkeyPatch,
):
    topology = _topologies()[0]
    workers, _events, _releases = _worker_and_events(topology, "source")
    worker = workers[0]
    case = benchmark.EXACT.Case(
        "source",
        benchmark.EXACT.DEFAULT_N,
        benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
        benchmark.EXACT.DEFAULT_PROBE_TILE,
        5,
        32,
    )
    monkeypatch.setattr(
        benchmark.EXACT,
        "_cpu_contract",
        lambda _cpus: topology.cpu_contracts[0],
    )
    args = argparse.Namespace(
        cpus=topology.cpu_groups[0], expected_native_sha256=SHA_A
    )
    gate = benchmark.EXACT._revalidate_worker_native_numa_evidence(
        worker, case, args
    )
    assert gate["protected_call_count"] == 8
    damaged = copy.deepcopy(worker)
    damaged["calls"][0]["native_gemm_output_numa_evidence"][
        "logical_columns"
    ] += 1
    with pytest.raises(RuntimeError, match="output evidence mismatch"):
        benchmark.EXACT._revalidate_worker_native_numa_evidence(
            damaged, case, args
        )


def test_controller_stops_immediately_after_first_rejected_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    attempted: list[str] = []
    monkeypatch.setattr(
        benchmark,
        "_base_report",
        lambda *_args: {"configurations": []},
    )
    monkeypatch.setattr(
        benchmark,
        "_configuration_memory_plan",
        lambda *_args: {
            "fits_per_process_limit": True,
            "fits_process_tree_limit": True,
        },
    )

    def reject(_args, configuration, *_rest):
        attempted.append(configuration.config_id)
        return {
            "config_id": configuration.config_id,
            "operation": configuration.operation,
            "topology_id": configuration.topology.topology_id,
            "status": "failed",
            "accepted": False,
            "reason": "synthetic rejection",
        }

    monkeypatch.setattr(benchmark, "_run_configuration", reject)
    monkeypatch.setattr(
        benchmark,
        "_artifact_stability_evidence",
        lambda *_args: {"stable": True, "checks": [], "mismatched_artifacts": []},
    )
    monkeypatch.setattr(benchmark, "_topology_summaries", lambda *_args: [])
    args = argparse.Namespace(
        sweep_timeout_seconds=60.0,
        configuration_timeout_seconds=10.0,
    )
    report = _controller_report(
        args,
        _topologies(),
        {
            "runner": {"sha256": SHA_A},
            "exact_layout_harness": {"sha256": SHA_B},
        },
        [],
        Path("/bin/true"),
    )
    assert attempted == ["1x32.source"]
    assert len(report["configurations"]) == 8
    assert report["configuration_execution_count"] == 1
    assert all(
        value["status"] == "not_run_after_first_failure"
        and value["trigger_config_id"] == "1x32.source"
        for value in report["configurations"][1:]
    )
    assert report["accepted"] is False
    assert report["first_failure_stopped_sweep"] is True


def test_deadline_before_first_configuration_is_sweep_level_with_eight_stubs(
    monkeypatch: pytest.MonkeyPatch,
):
    topologies = _topologies()
    planned = benchmark._configurations(topologies)

    monkeypatch.setattr(
        benchmark, "_base_report", lambda *_args: {"configurations": []}
    )
    monkeypatch.setattr(
        benchmark,
        "_artifact_stability_evidence",
        lambda *_args: {"stable": True, "checks": [], "mismatched_artifacts": []},
    )
    monkeypatch.setattr(
        benchmark,
        "_run_configuration",
        lambda *_args: pytest.fail("expired controller launched a configuration"),
    )
    monkeypatch.setattr(benchmark, "_topology_summaries", lambda *_args: [])
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    assert watchdog._latch(watchdog.deadline_ns) is True
    try:
        report = benchmark._controller(
            argparse.Namespace(
                sweep_timeout_seconds=60.0,
                configuration_timeout_seconds=10.0,
            ),
            topologies,
            {
                "runner": {"sha256": SHA_A},
                "exact_layout_harness": {"sha256": SHA_B},
            },
            [],
            Path("/bin/true"),
            watchdog,
        )
    finally:
        watchdog.cancel_join_close()
    assert report["status"] == "sweep_timeout_rejected"
    assert report["configuration_execution_count"] == 0
    assert report["sweep_deadline_event"]["trigger_config_id"] is None
    assert "before configuration planning" in report["sweep_deadline_event"][
        "reason"
    ]
    assert len(report["configurations"]) == 8
    assert [value["config_id"] for value in report["configurations"]] == [
        configuration.config_id for configuration in planned
    ]
    assert all(
        value["status"] == "not_run_after_first_failure"
        and value["trigger_config_id"] is None
        for value in report["configurations"]
    )


def test_post_run_artifact_and_summary_exceptions_become_rejection_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        benchmark, "_base_report", lambda *_args: {"configurations": []}
    )
    monkeypatch.setattr(
        benchmark,
        "_configuration_memory_plan",
        lambda *_args: {
            "fits_per_process_limit": True,
            "fits_process_tree_limit": True,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_run_configuration",
        lambda _args, configuration, *_rest: {
            "config_id": configuration.config_id,
            "operation": configuration.operation,
            "topology_id": configuration.topology.topology_id,
            "status": "failed",
            "accepted": False,
            "reason": "synthetic failure",
        },
    )

    def artifact_error(*_args):
        raise OSError("artifact disappeared")

    def summary_error(*_args):
        raise RuntimeError("summary malformed")

    monkeypatch.setattr(benchmark, "_artifact_stability_evidence", artifact_error)
    monkeypatch.setattr(benchmark, "_topology_summaries", summary_error)
    report = _controller_report(
        argparse.Namespace(
            sweep_timeout_seconds=60.0,
            configuration_timeout_seconds=10.0,
        ),
        _topologies(),
        {
            "runner": {"sha256": SHA_A},
            "exact_layout_harness": {"sha256": SHA_B},
        },
        [],
        Path("/bin/true"),
    )
    assert report["accepted"] is False
    assert report["artifact_identity_stable"] is False
    assert len(report["post_run_validation_errors"]) == 2
    assert "artifact disappeared" in report["post_run_validation_errors"][0]
    assert all(
        value["status"] == "summary_validation_rejected"
        for value in report["topology_summaries"]
    )


def test_artifact_stability_retains_each_observed_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = argparse.Namespace(
        install_prefix=tmp_path / "install",
        native_module=tmp_path / "native.so",
        private_archive=tmp_path / "libblis.a",
        python_executable=tmp_path / "python",
    )
    taskset = tmp_path / "taskset"
    monkeypatch.setattr(
        benchmark.EXACT,
        "_package_identity",
        lambda _path: {"manifest": "observed"},
    )
    monkeypatch.setattr(
        benchmark.EXACT,
        "_file_identity",
        lambda path: {"path": str(path)},
    )
    provenance = {
        "installed_package": {"manifest": "expected"},
        "native_module": {"path": str(args.native_module.resolve())},
        "private_archive": {"path": str(args.private_archive.resolve())},
        "runner": {"path": str(Path(benchmark.__file__).resolve())},
        "exact_layout_harness": {
            "path": str(Path(benchmark.EXACT.__file__).resolve())
        },
        "python_executable": {"path": str(args.python_executable.resolve())},
        "taskset_executable": {"path": str(taskset)},
    }
    evidence = benchmark._artifact_stability_evidence(args, provenance)
    assert evidence["stable"] is False
    assert evidence["mismatched_artifacts"] == ["installed_package"]
    installed = evidence["checks"][0]
    assert installed["expected"] == {"manifest": "expected"}
    assert installed["observed"] == {"manifest": "observed"}
    assert installed["error"] is None


def test_watchdog_expiry_after_last_result_rejects_without_duplicate_configs(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        benchmark, "_base_report", lambda *_args: {"configurations": []}
    )
    monkeypatch.setattr(
        benchmark,
        "_configuration_memory_plan",
        lambda *_args: {
            "fits_per_process_limit": True,
            "fits_process_tree_limit": True,
        },
    )
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    completed: list[str] = []

    def accepted_then_expire(_args, configuration, *_rest):
        completed.append(configuration.config_id)
        if len(completed) == 8:
            assert watchdog._latch(watchdog.deadline_ns) is True
        return {
            "config_id": configuration.config_id,
            "operation": configuration.operation,
            "topology_id": configuration.topology.topology_id,
            "status": "accepted",
            "accepted": True,
        }

    monkeypatch.setattr(benchmark, "_run_configuration", accepted_then_expire)
    monkeypatch.setattr(
        benchmark,
        "_artifact_stability_evidence",
        lambda *_args: {"stable": True, "checks": [], "mismatched_artifacts": []},
    )
    monkeypatch.setattr(
        benchmark,
        "_topology_summaries",
        lambda topologies, _configs: [
            {"topology_id": topology.topology_id, "accepted": True}
            for topology in topologies
        ],
    )
    try:
        report = benchmark._controller(
            argparse.Namespace(
                sweep_timeout_seconds=60.0,
                configuration_timeout_seconds=10.0,
            ),
            _topologies(),
            {
                "runner": {"sha256": SHA_A},
                "exact_layout_harness": {"sha256": SHA_B},
            },
            [],
            Path("/bin/true"),
            watchdog,
        )
    finally:
        watchdog.cancel_join_close()
    assert report["status"] == "sweep_timeout_rejected"
    assert report["configuration_execution_count"] == 8
    assert len(report["configurations"]) == 8
    assert len({value["config_id"] for value in report["configurations"]}) == 8
    assert report["sweep_deadline_event"]["timing"] == (
        "at_controller_execution_finish"
    )


def test_controller_preserves_configuration_cleanup_deadline_phase(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        benchmark, "_base_report", lambda *_args: {"configurations": []}
    )
    monkeypatch.setattr(
        benchmark,
        "_configuration_memory_plan",
        lambda *_args: {
            "fits_per_process_limit": True,
            "fits_process_tree_limit": True,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_run_configuration",
        lambda _args, configuration, *_rest: {
            "config_id": configuration.config_id,
            "operation": configuration.operation,
            "topology_id": configuration.topology.topology_id,
            "status": "sweep_timeout",
            "accepted": False,
            "reason": "deadline delivered after bounded cleanup",
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_artifact_stability_evidence",
        lambda *_args: {"stable": True, "checks": [], "mismatched_artifacts": []},
    )
    monkeypatch.setattr(benchmark, "_topology_summaries", lambda *_args: [])
    during_cleanup = _controller_report(
        argparse.Namespace(
            sweep_timeout_seconds=60.0,
            configuration_timeout_seconds=10.0,
        ),
        _topologies(),
        {
            "runner": {"sha256": SHA_A},
            "exact_layout_harness": {"sha256": SHA_B},
        },
        [],
        Path("/bin/true"),
    )
    assert during_cleanup["status"] == "sweep_timeout_rejected"
    assert during_cleanup["sweep_deadline_event"]["timing"] == (
        "during_configuration_or_cleanup"
    )


def test_watchdog_expiry_publishes_rejected_no_replace_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    output = tmp_path / "watchdog-rejection.json"
    topologies = _topologies()
    sibling_stop = threading.Event()
    sibling_started = threading.Event()

    def sibling_waiter():
        sibling_started.set()
        sibling_stop.wait()

    sibling = threading.Thread(target=sibling_waiter, daemon=True)
    sibling.start()
    assert sibling_started.wait(1.0)
    assert "signal" not in benchmark.__dict__
    monkeypatch.setattr(
        benchmark, "_validate_controller_args", lambda *_args: topologies
    )
    monkeypatch.setattr(benchmark.EXACT, "_dependency_paths", lambda *_args: [])
    monkeypatch.setattr(
        benchmark,
        "_static_provenance",
        lambda *_args: {"taskset_executable": {"path": "/bin/true"}},
    )

    def expire_with_worker_error(*call_args):
        watchdog = call_args[-1]
        assert watchdog._latch(watchdog.deadline_ns) is True
        raise RuntimeError("synthetic controller failure after deadline")

    def expire(*call_args):
        watchdog = call_args[-1]
        assert watchdog._latch(watchdog.deadline_ns) is True
        watchdog.raise_if_expired("inside synthetic controller")

    monkeypatch.setattr(benchmark, "_controller", expire_with_worker_error)
    argv = [
        "--install-prefix",
        str(tmp_path),
        "--native-module",
        str(tmp_path / "gxeldcore.test.so"),
        "--expected-native-sha256",
        SHA_A,
        "--expected-package-manifest-sha256",
        SHA_B,
        "--private-archive",
        str(tmp_path / "libblis.a"),
        "--expected-archive-sha256",
        SHA_A,
        "--expected-source-commit",
        COMMIT_A,
        "--expected-source-tree-sha256",
        SHA_B,
        "--expected-backend",
        "blis",
        "--output",
        str(output),
    ]
    assert benchmark.main(argv) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "sweep_timeout_rejected"
    assert report["accepted"] is False
    assert report["first_failure_stopped_sweep"] is True
    assert "at measured execution finish" in report["acceptance_reason"]
    assert report["concurrent_execution_failure"] == {
        "type": "RuntimeError",
        "message": "synthetic controller failure after deadline",
    }
    assert sibling.is_alive() is True
    deadline = report["sweep_deadline_watchdog"]
    assert deadline["clock"] == "CLOCK_MONOTONIC"
    assert type(deadline["started_ns"]) is int
    assert type(deadline["deadline_ns"]) is int
    assert type(deadline["finished_ns"]) is int
    assert deadline["started_ns"] < deadline["deadline_ns"]
    assert deadline["terminal_state"] == "expired"
    assert deadline["latched"] is True
    assert deadline["expired"] is True
    assert deadline["expired_at_ns"] == deadline["deadline_ns"]
    assert deadline["selector_notification"] == (
        "private_nonblocking_cloexec_self_pipe"
    )
    assert deadline["watchdog_thread_stopped"] is True
    assert deadline["wake_pipe_closed"] is True
    assert deadline["publication_outside_measured_cap"] is True
    assert output.stat().st_mode & 0o777 == 0o600

    dry_output = tmp_path / "dry-watchdog-rejection.json"
    monkeypatch.setattr(benchmark, "_dry_run_report", expire)
    dry_argv = [
        value if value != str(output) else str(dry_output) for value in argv
    ] + ["--dry-run"]
    assert benchmark.main(dry_argv) == 2
    dry_report = json.loads(dry_output.read_text())
    assert dry_report["dry_run"] is True
    assert dry_report["status"] == "sweep_timeout_rejected"
    assert dry_report["sweep_deadline_watchdog"]["clock"] == "CLOCK_MONOTONIC"
    assert dry_report["sweep_deadline_watchdog"]["wake_pipe_closed"] is True
    assert sibling.is_alive() is True
    sibling_stop.set()
    sibling.join(timeout=1.0)
    assert sibling.is_alive() is False


def test_linux_live_process_memory_is_combined_and_rejects_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    process = tmp_path / "123"
    (process / "task" / "123").mkdir(parents=True)
    (process / "smaps_rollup").write_text(
        "Rss:                2048 kB\nPss:                1024 kB\n"
    )
    children = process / "task" / "123" / "children"
    children.write_text("\n")
    observed = benchmark._read_live_process_memory(123, tmp_path)
    assert observed == {
        "pid": 123,
        "rss_bytes": 2048 * 1024,
        "pss_bytes": 1024 * 1024,
        "children": [],
    }
    children.write_text("456\n")
    with pytest.raises(RuntimeError, match="spawned child"):
        benchmark._read_live_process_memory(123, tmp_path)

    samples = {
        1: {"pid": 1, "rss_bytes": 100, "pss_bytes": 80, "children": []},
        2: {"pid": 2, "rss_bytes": 200, "pss_bytes": 120, "children": []},
    }
    monkeypatch.setattr(
        benchmark, "_read_live_process_memory", lambda pid: dict(samples[pid])
    )
    tracker = benchmark._MemoryTracker((1, 2))
    tracker.observe("barrier")
    report = tracker.report()
    assert report["sample_count"] == 1
    assert report["peak_combined_rss_snapshot"]["combined_rss_bytes"] == 300
    assert report["peak_combined_pss_snapshot"]["combined_pss_bytes"] == 200


@pytest.mark.parametrize("fill_wake_pipe", [False, True])
def test_watchdog_latch_is_authoritative_with_empty_or_full_wake_pipe(
    fill_wake_pipe: bool,
):
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    read_fd = watchdog.read_fd
    write_fd = watchdog._write_fd
    assert os.get_inheritable(read_fd) is False
    assert os.get_inheritable(write_fd) is False
    assert os.get_blocking(read_fd) is False
    assert os.get_blocking(write_fd) is False
    try:
        if fill_wake_pipe:
            while True:
                try:
                    os.write(write_fd, b"x" * 4096)
                except BlockingIOError:
                    break
        assert watchdog._latch(watchdog.deadline_ns) is True
        assert watchdog._latch(watchdog.deadline_ns) is False
        watchdog.drain_wake()
        with pytest.raises(
            benchmark._SweepDeadlineExpired, match="after draining wake pipe"
        ):
            watchdog.raise_if_expired("after draining wake pipe")
    finally:
        watchdog.finish_execution("after wake-pipe test")
        watchdog.cancel_join_close()
    assert watchdog.closed is True
    assert watchdog.thread_alive is False
    with pytest.raises(OSError):
        os.fstat(read_fd)
    with pytest.raises(OSError):
        os.fstat(write_fd)
    watchdog.cancel_join_close()


def test_finish_execution_rejects_exact_deadline_equality(
    monkeypatch: pytest.MonkeyPatch,
):
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    monkeypatch.setattr(
        benchmark.time,
        "clock_gettime_ns",
        lambda _clock: watchdog.deadline_ns,
    )
    finished_ns, error = watchdog.finish_execution("at exact equality")
    try:
        assert finished_ns == watchdog.deadline_ns
        assert isinstance(error, benchmark._SweepDeadlineExpired)
        assert watchdog.state == watchdog._EXPIRED
        assert watchdog.expired_at_ns == watchdog.deadline_ns
    finally:
        watchdog.cancel_join_close()


def test_blocked_controller_write_wakes_at_sweep_deadline():
    control_read, control_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    while True:
        try:
            os.write(control_write, b"x" * 4096)
        except BlockingIOError:
            break
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    started = threading.Event()
    errors: list[BaseException] = []

    def blocked_write():
        started.set()
        try:
            benchmark._write_all(
                control_write,
                b"z",
                watchdog=watchdog,
                local_deadline_ns=time_ns() + 60_000_000_000,
                context="writing to a full controller pipe",
            )
        except BaseException as error:
            errors.append(error)

    writer = threading.Thread(target=blocked_write, daemon=True)
    writer.start()
    try:
        assert started.wait(1.0)
        assert watchdog._latch(watchdog.deadline_ns) is True
        writer.join(timeout=1.0)
        assert writer.is_alive() is False
        assert len(errors) == 1
        assert isinstance(errors[0], benchmark._SweepDeadlineExpired)
    finally:
        watchdog._latch(watchdog.deadline_ns)
        writer.join(timeout=1.0)
        _close_test_fd(control_read)
        _close_test_fd(control_write)
        watchdog.finish_execution("after blocked-write test")
        watchdog.cancel_join_close()


def test_watchdog_selector_wakes_with_active_sibling_thread(
    monkeypatch: pytest.MonkeyPatch,
):
    real_selector = benchmark.selectors.DefaultSelector
    selector_entered = threading.Event()
    selected_fds: list[int] = []

    class ObservedSelector:
        def __init__(self):
            self.inner = real_selector()

        def register(self, *args, **kwargs):
            return self.inner.register(*args, **kwargs)

        def select(self, timeout=None):
            selector_entered.set()
            events = self.inner.select(timeout)
            selected_fds.extend(int(key.fileobj) for key, _mask in events)
            return events

        def close(self):
            self.inner.close()

    class SilentProcess:
        pid = 9910

        @staticmethod
        def poll():
            return None

    class Tracker:
        @staticmethod
        def observe(_label):
            pass

    event_read, event_write = os.pipe()
    handle = benchmark._WorkerHandle(
        group_index=0,
        process=SilentProcess(),
        control_write_fd=-1,
        event_read_fd=event_read,
        command=["silent-worker"],
        event_buffer=bytearray(),
    )
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    watchdog_read_fd = watchdog.read_fd
    errors: list[BaseException] = []
    monkeypatch.setattr(benchmark.selectors, "DefaultSelector", ObservedSelector)

    def wait_for_worker():
        try:
            benchmark._wait_for_events(
                [handle],
                "ready",
                time_ns() + 60_000_000_000,
                Tracker(),
                watchdog,
            )
        except BaseException as error:
            errors.append(error)

    waiter = threading.Thread(target=wait_for_worker, daemon=True)
    waiter.start()
    try:
        assert selector_entered.wait(1.0)
        assert watchdog._latch(watchdog.deadline_ns) is True
        waiter.join(timeout=1.0)
        assert waiter.is_alive() is False
        assert len(errors) == 1
        assert isinstance(errors[0], benchmark._SweepDeadlineExpired)
        assert watchdog_read_fd in selected_fds
        assert "signal" not in benchmark.__dict__
    finally:
        watchdog._latch(watchdog.deadline_ns)
        waiter.join(timeout=1.0)
        _close_test_fd(event_read)
        _close_test_fd(event_write)
        watchdog.finish_execution("after selector wake test")
        watchdog.cancel_join_close()


def test_watchdog_join_failure_retains_owned_fds_until_writer_stops():
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    read_fd, write_fd = watchdog.read_fd, watchdog._write_fd
    watchdog.finish_execution("before synthetic join failure")
    real_thread = watchdog._thread
    real_thread.join(timeout=1.0)
    assert real_thread.is_alive() is False

    class SyntheticThread:
        alive = True

        @staticmethod
        def join(timeout=None):
            assert timeout == benchmark.WATCHDOG_JOIN_TIMEOUT_SECONDS

        def is_alive(self):
            return self.alive

    synthetic = SyntheticThread()
    watchdog._thread = synthetic
    with pytest.raises(RuntimeError, match="did not stop"):
        watchdog.cancel_join_close()
    assert watchdog.closed is False
    os.fstat(read_fd)
    os.fstat(write_fd)

    synthetic.alive = False
    watchdog.cancel_join_close()
    assert watchdog.closed is True
    with pytest.raises(OSError):
        os.fstat(read_fd)
    with pytest.raises(OSError):
        os.fstat(write_fd)


def test_controller_source_has_no_realtime_signal_deadline_api():
    source = Path(benchmark.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    referenced_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "signal" not in imported_modules
    assert "signal" not in imported_from
    assert "signal" not in referenced_names
    assert {
        "SIGALRM",
        "setitimer",
        "pthread_sigmask",
        "sigpending",
        "sigwait",
    }.isdisjoint(referenced_attributes)


def time_ns() -> int:
    return benchmark.time.clock_gettime_ns(benchmark.time.CLOCK_MONOTONIC)


def _close_test_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def test_cleanup_never_uses_unbounded_wait_and_fails_closed_after_sigkill(
    monkeypatch: pytest.MonkeyPatch,
):
    waits: list[float | None] = []
    signals: list[tuple[int, int]] = []

    class NonterminatingProcess:
        pid = 9876

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout=None):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("mock-worker", timeout)

    monkeypatch.setattr(
        benchmark.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )
    handle = benchmark._WorkerHandle(
        group_index=0,
        process=NonterminatingProcess(),
        control_write_fd=-1,
        event_read_fd=-1,
        command=["mock-worker"],
        event_buffer=bytearray(),
    )
    with pytest.raises(RuntimeError, match="bounded SIGKILL cleanup"):
        benchmark._kill_handles([handle])
    assert len(waits) == 2
    assert all(timeout is not None and 0.0 <= timeout <= 2.0 for timeout in waits)
    assert signals == [
        (9876, benchmark.TERMINATE_PROCESS_GROUP_SIGNAL),
        (9876, benchmark.KILL_PROCESS_GROUP_SIGNAL),
    ]


def test_failure_stream_drain_and_already_collected_output_are_truthfully_bounded(
    monkeypatch: pytest.MonkeyPatch,
):
    class ReapedProcess:
        pid = 9878

        @staticmethod
        def poll():
            return -benchmark.TERMINATE_PROCESS_GROUP_SIGNAL

        @staticmethod
        def communicate(timeout=None):
            raise AssertionError(f"already-collected output reread with {timeout=}")

    handle = benchmark._WorkerHandle(
        group_index=0,
        process=ReapedProcess(),
        control_write_fd=-1,
        event_read_fd=123,
        command=["mock-worker"],
        event_buffer=bytearray(),
        stdout_tail="retained stdout",
        stderr_tail="retained stderr",
        output_collected=True,
    )
    read_sizes: list[int] = []
    monkeypatch.setattr(benchmark.os, "set_blocking", lambda *_args: None)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: 0.0)

    def continuously_readable(_fd, requested):
        size = min(requested, 1024)
        read_sizes.append(size)
        return b"x" * size

    monkeypatch.setattr(benchmark.os, "read", continuously_readable)
    benchmark._drain_event_stream_tail(handle)
    assert sum(read_sizes) == benchmark.FAILURE_EVENT_DRAIN_MAX_BYTES
    assert len(handle.event_stream_tail) == benchmark.FAILURE_EVENT_STREAM_TAIL_BYTES

    evidence = benchmark._capture_worker_failure_evidence([handle])
    assert evidence[0]["returncode"] == -benchmark.TERMINATE_PROCESS_GROUP_SIGNAL
    assert evidence[0]["stdout_tail"] == "retained stdout"
    assert evidence[0]["stderr_tail"] == "retained stderr"
    assert evidence[0]["output_capture_error"] is None


@pytest.mark.parametrize(
    "expiry_timing",
    ["before_spawn", "before_popen", "after_spawn", "during_cleanup"],
)
def test_watchdog_spawn_and_cleanup_expiry_reaps_and_retains_evidence(
    monkeypatch: pytest.MonkeyPatch,
    expiry_timing: str,
):
    class FakeProcess:
        pid = 9877

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def communicate(self, timeout=None):
            del timeout
            return ("bounded stdout evidence", "bounded stderr evidence")

    process = FakeProcess()
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    popen_calls = 0
    killpg_calls: list[int] = []
    if expiry_timing == "before_spawn":
        assert watchdog._latch(watchdog.deadline_ns) is True

    def popen(*_args, **_kwargs):
        nonlocal popen_calls
        popen_calls += 1
        if expiry_timing == "after_spawn":
            assert watchdog._latch(watchdog.deadline_ns) is True
        return process

    def killpg(_pid, _sent_signal):
        killpg_calls.append(_sent_signal)
        if expiry_timing == "during_cleanup":
            assert watchdog._latch(watchdog.deadline_ns) is True
        process.returncode = -benchmark.TERMINATE_PROCESS_GROUP_SIGNAL

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark.os, "killpg", killpg)
    monkeypatch.setattr(benchmark, "_worker_command", lambda *_args: ["worker"])

    def worker_environment(*_args):
        if expiry_timing == "before_popen":
            assert watchdog._latch(watchdog.deadline_ns) is True
        return {}, {}

    monkeypatch.setattr(
        benchmark.EXACT,
        "_worker_environment",
        worker_environment,
    )
    if expiry_timing == "during_cleanup":
        monkeypatch.setattr(
            benchmark,
            "_wait_for_events",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic worker failure")
            ),
        )
    topology = benchmark.Topology(
        "test",
        1,
        ((0,),),
        (_cpu_contract((0,)),),
        (5,),
        True,
        None,
    )
    configuration = benchmark.Configuration(topology, "source")
    args = argparse.Namespace(
        configuration_timeout_seconds=10.0,
        n=benchmark.EXACT.DEFAULT_N,
        block_width=benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
        probe_tile=benchmark.EXACT.DEFAULT_PROBE_TILE,
    )
    try:
        result = benchmark._run_configuration(
            args,
            configuration,
            Path("/bin/true"),
            SHA_A,
            SHA_B,
            watchdog,
        )
    finally:
        watchdog.finish_execution("after focused configuration test")
        watchdog.cancel_join_close()
    assert result["status"] == "sweep_timeout"
    assert result["accepted"] is False
    assert result["cleanup"]["completed"] is True
    if expiry_timing in {"before_spawn", "before_popen"}:
        assert popen_calls == 0
        assert killpg_calls == []
        assert result["worker_processes"] == []
        return
    assert popen_calls == 1
    assert process.poll() is not None
    assert len(result["worker_processes"]) == 1
    worker = result["worker_processes"][0]
    assert worker["stdout_tail"] == "bounded stdout evidence"
    assert worker["stderr_tail"] == "bounded stderr evidence"
    assert worker["output_capture_error"] is None
    if expiry_timing == "after_spawn":
        assert result["original_failure"]["type"] == "_SweepDeadlineExpired"
    if expiry_timing == "during_cleanup":
        assert "synthetic worker failure" in result["reason"]
        assert result["original_failure"] == {
            "type": "RuntimeError",
            "message": "synthetic worker failure",
        }
        assert "during bounded configuration cleanup" in result["reason"]
        assert result["sweep_deadline_failure"] is not None


def test_cleanup_deadline_after_success_remains_sweep_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeProcess:
        pid = 9879

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            del timeout
            self.returncode = 0
            return ("accepted worker", "")

    class FakeTracker:
        sample_count = 1

        def __init__(self, _pids):
            pass

        @staticmethod
        def report():
            return {"sample_count": 1}

    process = FakeProcess()
    topology = benchmark.Topology(
        "test",
        1,
        ((0,),),
        (_cpu_contract((0,)),),
        (5,),
        True,
        None,
    )
    configuration = benchmark.Configuration(topology, "source")

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(benchmark, "_worker_command", lambda *_args: ["worker"])
    monkeypatch.setattr(
        benchmark.EXACT, "_worker_environment", lambda *_args: ({}, {})
    )
    monkeypatch.setattr(benchmark, "_MemoryTracker", FakeTracker)
    monkeypatch.setattr(
        benchmark, "_write_all", lambda *_args, **_kwargs: None
    )

    def events(
        _handles,
        expected_event,
        _deadline,
        _tracker,
        _watchdog,
        *,
        ordinal=None,
    ):
        del ordinal
        if expected_event in {"ready", "final_ready"}:
            return {
                0: {
                    "topology_id": topology.topology_id,
                    "group_index": 0,
                    "operation": configuration.operation,
                    "case_id": benchmark.EXACT.Case(
                        configuration.operation,
                        benchmark.EXACT.DEFAULT_N,
                        benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
                        benchmark.EXACT.DEFAULT_PROBE_TILE,
                        5,
                        1,
                    ).case_id,
                }
            }
        return {0: {"event": "call_done"}}

    monkeypatch.setattr(benchmark, "_wait_for_events", events)
    monkeypatch.setattr(
        benchmark,
        "_parse_worker_stdout",
        lambda _stdout: {
            "pid": process.pid,
            "group_index": 0,
            "topology_id": topology.topology_id,
            "operation": configuration.operation,
            "accepted": True,
        },
    )
    monkeypatch.setattr(benchmark, "_aggregate_configuration", lambda *_a: {})
    original_kill_handles = benchmark._kill_handles
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)

    def cleanup_then_expiry(handles):
        original_kill_handles(handles)
        assert watchdog._latch(watchdog.deadline_ns) is True

    monkeypatch.setattr(benchmark, "_kill_handles", cleanup_then_expiry)
    args = argparse.Namespace(
        configuration_timeout_seconds=10.0,
        max_start_skew_seconds=benchmark.MAX_START_SKEW_SECONDS,
        n=benchmark.EXACT.DEFAULT_N,
        block_width=benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
        probe_tile=benchmark.EXACT.DEFAULT_PROBE_TILE,
    )
    try:
        result = benchmark._run_configuration(
            args,
            configuration,
            Path("/bin/true"),
            SHA_A,
            SHA_B,
            watchdog,
        )
    finally:
        watchdog.finish_execution("after focused configuration test")
        watchdog.cancel_join_close()
    assert result["status"] == "sweep_timeout"
    assert result["accepted"] is False
    assert result["cleanup"]["completed"] is True
    assert "during bounded configuration cleanup" in result["reason"]
    assert result["original_failure"] is None
    assert result["sweep_deadline_failure"] is not None


def test_handle_registration_failure_reaps_locally_spawned_child(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeProcess:
        pid = 9920
        stdout = None
        stderr = None

        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

    process = FakeProcess()
    sent_signals: list[int] = []

    def killpg(_pid, sent_signal):
        sent_signals.append(sent_signal)
        process.returncode = -sent_signal

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(benchmark.os, "killpg", killpg)
    monkeypatch.setattr(benchmark, "_worker_command", lambda *_args: ["worker"])
    monkeypatch.setattr(
        benchmark.EXACT, "_worker_environment", lambda *_args: ({}, {})
    )

    def fail_handle_registration(*_args, **_kwargs):
        raise RuntimeError("synthetic handle registration failure")

    monkeypatch.setattr(benchmark, "_WorkerHandle", fail_handle_registration)
    topology = benchmark.Topology(
        "test",
        1,
        ((0,),),
        (_cpu_contract((0,)),),
        (5,),
        True,
        None,
    )
    watchdog = benchmark._SweepDeadlineWatchdog.start(60.0)
    try:
        result = benchmark._run_configuration(
            argparse.Namespace(
                configuration_timeout_seconds=10.0,
                n=benchmark.EXACT.DEFAULT_N,
                block_width=benchmark.EXACT.DEFAULT_BLOCK_WIDTH,
                probe_tile=benchmark.EXACT.DEFAULT_PROBE_TILE,
            ),
            benchmark.Configuration(topology, "source"),
            Path("/bin/true"),
            SHA_A,
            SHA_B,
            watchdog,
        )
    finally:
        watchdog.finish_execution("after registration-failure test")
        watchdog.cancel_join_close()
    assert result["status"] == "failed"
    assert result["original_failure"] == {
        "type": "RuntimeError",
        "message": "synthetic handle registration failure",
    }
    assert sent_signals == [benchmark.TERMINATE_PROCESS_GROUP_SIGNAL]
    assert process.poll() is not None
    assert result["worker_processes"] == []


def test_dry_run_plans_all_fresh_workers_without_numerical_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    prefix = tmp_path / "install"
    package = prefix / "summit"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# exact test package\n")
    module = package / "gxeldcore.test.so"
    module.write_bytes(b"native-test")
    archive = tmp_path / "libopenblas.a"
    archive.write_bytes(b"archive-test")
    output = tmp_path / "topology-dry.json"
    package_hash = benchmark.EXACT._package_identity(prefix)["manifest_sha256"]
    topologies = _topologies()
    monkeypatch.setattr(benchmark, "_topology_contracts", lambda _args: topologies)
    monkeypatch.setattr(
        benchmark.EXACT,
        "_linkage_evidence",
        lambda *_args: {"private_static_linkage_verified": True},
    )
    argv = [
        "--install-prefix",
        str(prefix),
        "--native-module",
        str(module),
        "--expected-native-sha256",
        hashlib.sha256(b"native-test").hexdigest(),
        "--expected-package-manifest-sha256",
        package_hash,
        "--private-archive",
        str(archive),
        "--expected-archive-sha256",
        hashlib.sha256(b"archive-test").hexdigest(),
        "--expected-source-commit",
        COMMIT_A,
        "--expected-source-tree-sha256",
        SHA_B,
        "--expected-backend",
        "openblas",
        "--python-executable",
        sys.executable,
        "--dependency-path",
        str(tmp_path),
        "--configuration-timeout-seconds",
        "10",
        "--sweep-timeout-seconds",
        "60",
        "--output",
        str(output),
        "--dry-run",
    ]
    assert benchmark.main(argv) == 0
    report = json.loads(output.read_text())
    assert report["accepted"] is False
    assert report["status"] == "validated_dry_run"
    assert report["scientific_execution"] is False
    deadline = report["sweep_deadline_watchdog"]
    assert deadline["clock"] == "CLOCK_MONOTONIC"
    assert deadline["terminal_state"] == "disarmed"
    assert deadline["latched"] is False
    assert deadline["expired"] is False
    assert deadline["expired_at_ns"] is None
    assert deadline["started_ns"] <= deadline["finished_ns"] < deadline["deadline_ns"]
    assert deadline["watchdog_thread_stopped"] is True
    assert deadline["wake_pipe_closed"] is True
    assert deadline["publication_outside_measured_cap"] is True
    assert len(report["configurations"]) == 8
    assert all(value["status"] == "planned" for value in report["configurations"])
    assert (
        sum(len(value["worker_commands"]) for value in report["configurations"]) == 18
    )
    planned_widths = {
        (value["topology"]["topology_id"], value["operation"]): [
            worker["panel_columns"] for worker in value["memory_plan"]["per_process"]
        ]
        for value in report["configurations"]
    }
    assert planned_widths[("1x32", "source")] == [320]
    assert planned_widths[("2x16_same_socket", "source")] == [192, 128]
    assert planned_widths[("1x32", "target")] == [640]
    assert planned_widths[("2x32_two_socket", "target")] == [384, 256]
    assert planned_widths[("4x8_same_socket", "source")] == [128, 64, 64, 64]
    assert report["topology_summaries"][-1]["production_promotable"] is False
    for configuration in report["configurations"]:
        for command in configuration["worker_commands"]:
            assert "-S" in command
            assert "--_worker" in command
            assert command[0].endswith("taskset")
            assert any("<control-read-fd-" in value for value in command)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(SystemExit):
        benchmark.main(argv)


def test_protocol_caps_and_exact_configuration_count_are_fixed():
    assert benchmark.MAX_CONFIGURATION_SECONDS == 90.0
    assert benchmark.MAX_SWEEP_SECONDS == 1_200.0
    assert benchmark.MAX_START_SKEW_SECONDS == 0.25
    configurations = benchmark._configurations(_topologies())
    assert len(configurations) == 8
    assert {value.operation for value in configurations} == {"source", "target"}
    assert all(value.topology.process_count in {1, 2, 4} for value in configurations)
