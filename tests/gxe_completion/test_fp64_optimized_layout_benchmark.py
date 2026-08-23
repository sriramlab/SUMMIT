from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.gxe import benchmark_fp64_optimized_layout as benchmark


class _Pair:
    def __init__(self, source: np.ndarray, environments: np.ndarray):
        self.rows = source.shape[0]
        self.columns = source.shape[1]
        columns_per_group = source.shape[1] // environments.shape[1]
        weighted = np.empty_like(source)
        for group in range(environments.shape[1]):
            segment = slice(group * columns_per_group, (group + 1) * columns_per_group)
            weighted[:, segment] = source[:, segment] * environments[:, group, None]
        self.sealed = np.concatenate([source, weighted], axis=1)


class _FakeOptimizedModule:
    def __init__(self):
        self.records = []

    def reset_gemm_telemetry(self):
        self.records.clear()

    def consume_gemm_telemetry(self):
        records = list(self.records)
        self.records.clear()
        return records

    @staticmethod
    def protected_matmul_tt_row_major_output(weights, genotype, threads):
        assert threads == 2
        return np.ascontiguousarray(genotype @ weights), 0

    @staticmethod
    def prepare_protected_row_weighted_pair(source, environments, threads):
        assert threads == 2
        return _Pair(source.copy(), environments.copy())

    @staticmethod
    def protected_matmul_tn_pair(genotype, pair, threads):
        assert threads == 2
        return np.asfortranarray(genotype.T @ pair.sealed), 0


def test_exact_profile_has_literal_production_wrapper_shapes():
    parser = benchmark._parser()
    args = parser.parse_args(["--profile", "exact", "--dry-run"])
    benchmark._validate_args(parser, args)
    source, target = benchmark._cases(args)

    assert (source.n_samples, source.block_width) == (289_111, 2000)
    assert (source.probe_tile, source.environment_tile, source.threads) == (
        32,
        3,
        32,
    )
    assert source.source_columns == 192
    assert target.target_columns == 384
    assert source.expected_telemetry() == {
        "operation": "dgemm_tt",
        "layout": "column_major",
        "transpose_a": "T",
        "transpose_b": "T",
        "m": 192,
        "n": 289_111,
        "k": 2000,
        "lda": 2000,
        "ldb": 289_111,
        "ldc": 192,
    }
    assert target.expected_telemetry() == {
        "operation": "dgemm_tn",
        "layout": "column_major",
        "transpose_a": "T",
        "transpose_b": "N",
        "m": 2000,
        "n": 384,
        "k": 289_111,
        "lda": 289_111,
        "ldb": 289_111,
        "ldc": 2000,
    }
    dry_run = benchmark._dry_run_report(args, (source, target))
    assert dry_run["diagnostic_only"] is True
    assert dry_run["production_acceptance_eligible"] is False
    assert "failed" in dry_run["production_rejection_reason"]
    assert dry_run["limits"]["minimum_vendor_active_core_fraction"] == 0.75
    assert dry_run["limits"]["vendor_active_core_gate_scope"] == (
        "each_measured_call_excludes_warmups"
    )
    for planned in dry_run["cases"]:
        memory = planned["memory_model"]
        assert memory["vendor_workspace_allowance_gib"] == 16.0
        assert "uncertified" in memory["vendor_workspace_bound_source"]
        assert "4.125 GiB" in memory["vendor_workspace_bound_uncertainty"]


def test_fake_optimized_wrappers_match_sampled_source_and_target_oracles():
    module = _FakeOptimizedModule()
    for operation in ("source", "target"):
        case = benchmark.Case(operation, 37, 11, 3, 2, 2)
        prepared = benchmark._prepare(case, module, seed=9081)
        output, repaired = prepared.invoke()
        assert repaired == 0
        if operation == "source":
            assert output.flags.c_contiguous
        else:
            assert output.flags.f_contiguous
        check = benchmark._validate_samples(prepared, output)
        assert check["maximum_normalized_error"] <= 1.0
        if operation == "source":
            assert [array.shape for _, array in prepared.operands] == [
                (37, 11),
                (11, 12),
            ]
            assert all(array.flags.f_contiguous for _, array in prepared.operands)
            assert prepared.pair_construction is None
        else:
            assert [array.shape for _, array in prepared.operands] == [
                (37, 11),
                (37, 12),
                (37, 2),
            ]
            assert prepared.operands[0][1].flags.f_contiguous
            assert prepared.operands[1][1].flags.f_contiguous
            assert prepared.operands[2][1].flags.f_contiguous
            assert prepared.pair_metadata == {
                "rows": 37,
                "unweighted_columns": 12,
                "sealed_total_columns": 24,
                "input_mode": "mprotect_read_only",
                "vendor_gemm_records_during_construction": 0,
            }
            assert prepared.pair_construction["wall_seconds"] > 0.0


def test_worker_command_is_fresh_no_site_and_thread_environment_is_immutable(tmp_path):
    prefix = tmp_path / "installed"
    module = prefix / "summit" / "gxeldcore.cpython-312-x86_64-linux-gnu.so"
    args = argparse.Namespace(
        install_prefix=prefix,
        native_module=module,
        expected_native_sha256="a" * 64,
        expected_archive_sha256="b" * 64,
        python_executable=Path("/opt/python/bin/python"),
        warmups=3,
        repeats=5,
        seed=7,
        allow_integrity_disabled=False,
    )
    case = benchmark.Case("target", 37, 11, 3, 2, 2)
    command = benchmark._worker_command(args, case)
    assert command[:3] == [
        "/opt/python/bin/python",
        "-S",
        str(Path(benchmark.__file__).resolve()),
    ]
    assert command[command.index("--native-module") + 1] == str(module.resolve())
    assert command[command.index("--install-prefix") + 1] == str(prefix.resolve())
    assert "--allow-integrity-disabled" not in command
    args.allow_integrity_disabled = True
    assert "--allow-integrity-disabled" in benchmark._worker_command(args, case)
    environment = benchmark._worker_environment(case, [4, 7])
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["OMP_THREAD_LIMIT"] == "2"
    assert environment["OPENBLAS_NUM_THREADS"] == "2"
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert environment["OMP_PROC_BIND"] == "FALSE"
    assert "OMP_PLACES" not in environment
    assert "GOMP_CPU_AFFINITY" not in environment
    assert environment["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert environment["OMP_WAIT_POLICY"] == "PASSIVE"
    assert environment["GOMP_SPINCOUNT"] == "0"


def test_affinity_contract_excludes_changing_rss_but_retains_cpu_and_memory_masks():
    first = {
        "scope": "calling_thread_sched_getaffinity",
        "allowed_cpus": [0, 1],
        "allowed_cpu_count": 2,
        "allowed_cpu_list": "0-1",
        "physical_core_keys": [[0, 0], [0, 1]],
        "assigned_numa_nodes_from_cpus": [0],
        "status": {
            "Cpus_allowed_list": "0-1",
            "Mems_allowed_list": "0-3",
            "VmRSS": "1024 kB",
            "VmHWM": "2048 kB",
        },
    }
    changed_rss = {
        **first,
        "status": {
            **first["status"],
            "VmRSS": "4096 kB",
            "VmHWM": "8192 kB",
        },
    }
    assert benchmark._affinity_contract(first) == benchmark._affinity_contract(
        changed_rss
    )
    changed_cpu_mask = {
        **changed_rss,
        "status": {**changed_rss["status"], "Cpus_allowed_list": "0"},
    }
    assert benchmark._affinity_contract(first) != benchmark._affinity_contract(
        changed_cpu_mask
    )


def test_literal_vendor_telemetry_contract_rejects_layout_or_outer_openmp():
    case = benchmark.Case("target", 37, 11, 3, 2, 2)
    affinity = {
        "allowed_cpu_count": 2,
        "allowed_cpu_list": "4,7",
        "assigned_numa_nodes_from_cpus": [0],
    }
    numa_operands = {
        name: {
            "query_status": "queried",
            "selected_sample_pages": 2,
            "resolved_sample_pages": 2,
            "page_query_error_pages": 0,
            "node_histogram": {"0": 2},
        }
        for name in ("a", "b", "c")
    }
    record = {
        **case.expected_telemetry(),
        "schema_version": 1,
        "arithmetic_dtype": "float64",
        "requested_threads": 2,
        "configured_threads": 2,
        "backend_threads": 2,
        "omp_in_parallel": False,
        "omp_level": 0,
        "omp_active_level": 0,
        "completed": True,
        "cpu_affinity_count": 2,
        "cpu_affinity_list": "4,7",
        "flop_count": float(case.flops),
        "wall_seconds": 0.1,
        "process_cpu_seconds": 0.2,
        "gflops_per_second": case.flops / 0.1 / 1.0e9,
        "active_core_equivalents": 2.0,
        "operand_numa_page_samples": {
            "sampling_method": "move_pages_query_no_migration",
            "operands": numa_operands,
        },
    }
    locality = benchmark._validate_vendor_record(case, record, affinity)
    assert locality["all_operands_queried_and_local"] is True
    assert locality["acceptance_eligible_on_exact_host"] is True
    with pytest.raises(RuntimeError, match="shape/layout"):
        benchmark._validate_vendor_record(
            case, {**record, "layout": "row_major"}, affinity
        )
    with pytest.raises(RuntimeError, match="execution contract"):
        benchmark._validate_vendor_record(
            case, {**record, "omp_in_parallel": True}, affinity
        )
    low_core_record = {**record, "active_core_equivalents": 1.0}
    benchmark._validate_vendor_record(case, low_core_record, affinity)
    warmup_gate = benchmark._validate_vendor_core_utilization(
        case, low_core_record, phase="warmup", repeat=0
    )
    assert warmup_gate["applies"] is False
    assert warmup_gate["passed"] is None
    with pytest.raises(RuntimeError, match="measured native telemetry"):
        benchmark._validate_vendor_core_utilization(
            case, low_core_record, phase="measured", repeat=0
        )
    remote_operands = {name: dict(value) for name, value in numa_operands.items()}
    remote_operands["b"] = {
        **remote_operands["b"],
        "node_histogram": {"1": 2},
    }
    with pytest.raises(RuntimeError, match="outside the selected CPU NUMA nodes"):
        benchmark._validate_vendor_record(
            case,
            {
                **record,
                "operand_numa_page_samples": {
                    "sampling_method": "move_pages_query_no_migration",
                    "operands": remote_operands,
                },
            },
            affinity,
        )
    denied_operands = {
        name: {"query_status": "permission_denied"} for name in ("a", "b", "c")
    }
    denied = benchmark._validate_vendor_record(
        case,
        {
            **record,
            "operand_numa_page_samples": {
                "sampling_method": "move_pages_query_no_migration",
                "operands": denied_operands,
            },
        },
        affinity,
    )
    assert denied["unavailable_status_is_explicit"] is True
    assert denied["acceptance_eligible_on_exact_host"] is False
    incomplete_operands = {name: dict(value) for name, value in numa_operands.items()}
    incomplete_operands["c"] = {
        **incomplete_operands["c"],
        "resolved_sample_pages": 1,
        "page_query_error_pages": 1,
        "node_histogram": {"0": 1},
    }
    with pytest.raises(RuntimeError, match="zero page errors"):
        benchmark._validate_vendor_record(
            case,
            {
                **record,
                "operand_numa_page_samples": {
                    "sampling_method": "move_pages_query_no_migration",
                    "operands": incomplete_operands,
                },
            },
            affinity,
        )


def test_integrity_smoke_requires_no_vendor_record_but_exact_shapes_do():
    build = {
        "gemm_integrity_enabled": True,
        "gemm_integrity_minimum_vendor_flops": 1_000_000_000,
    }
    smoke = benchmark.Case("source", 64, 16, 2, 1, 2)
    exact_source = benchmark.Case("source", 289_111, 2000, 32, 3, 32)
    exact_target = benchmark.Case("target", 289_111, 2000, 32, 3, 32)
    assert benchmark._vendor_telemetry_required(smoke, build) is False
    assert benchmark._vendor_telemetry_required(exact_source, build) is True
    assert benchmark._vendor_telemetry_required(exact_target, build) is True
    assert (
        benchmark._vendor_telemetry_required(smoke, {"gemm_integrity_enabled": False})
        is True
    )


def test_full_output_hash_covers_every_native_contiguous_byte_without_layout_copy():
    output = np.arange(3 * 17, dtype=np.float64).reshape(3, 17)
    expected = hashlib.sha256(memoryview(output).cast("B")).hexdigest()
    first, first_seconds = benchmark._contiguous_array_sha256(output, chunk_bytes=19)
    second, second_seconds = benchmark._contiguous_array_sha256(
        output.copy(order="C"), chunk_bytes=7
    )
    assert first == second == expected
    assert first_seconds >= 0.0
    assert second_seconds >= 0.0
    changed = output.copy(order="C")
    changed[-1, -1] += 1.0
    assert benchmark._contiguous_array_sha256(changed)[0] != expected
    output_f = np.asfortranarray(output)
    expected_f = hashlib.sha256(memoryview(output_f.T).cast("B")).hexdigest()
    assert benchmark._contiguous_array_sha256(output_f, chunk_bytes=11)[0] == expected_f
    with pytest.raises(RuntimeError, match="contiguous array"):
        benchmark._contiguous_array_sha256(output[:, ::2])


def test_observed_call_rejects_full_output_change_outside_oracle_samples(
    monkeypatch,
):
    calls = 0

    def invoke():
        nonlocal calls
        calls += 1
        output = np.zeros((7, 7), dtype=np.float64, order="C")
        if calls == 3:
            output[1, 2] = 1.0  # Deliberately outside the five oracle samples.
        return output, 0

    prepared = benchmark.Prepared(
        invoke=invoke,
        expected_entry=lambda _row, _column: 0.0,
        output_shape=(7, 7),
        output_order="C",
        operands=[],
        pair_construction=None,
        pair_metadata=None,
    )

    class NoVendorModule:
        @staticmethod
        def reset_gemm_telemetry():
            return None

        @staticmethod
        def consume_gemm_telemetry():
            return []

        @staticmethod
        def gemm_telemetry_status():
            return {"dropped_records": 0}

    affinity = {
        "scope": "calling_thread_sched_getaffinity",
        "allowed_cpus": [0],
        "allowed_cpu_count": 1,
        "allowed_cpu_list": "0",
        "physical_core_keys": [[0, 0]],
        "assigned_numa_nodes_from_cpus": [0],
        "status": {},
    }
    monkeypatch.setattr(benchmark, "_affinity", lambda: affinity)
    case = benchmark.Case("source", 7, 7, 1, 1, 1)
    first, samples, output_sha256 = benchmark._invoke_observed(
        case, prepared, NoVendorModule(), 0, "warmup", None, None, False
    )
    assert first["full_output_bitwise_deterministic_against_first_call"] is None
    second, _, second_sha256 = benchmark._invoke_observed(
        case,
        prepared,
        NoVendorModule(),
        0,
        "measured",
        samples,
        output_sha256,
        False,
    )
    assert second_sha256 == output_sha256
    assert second["full_output_bitwise_deterministic_against_first_call"] is True
    with pytest.raises(RuntimeError, match="full protected output"):
        benchmark._invoke_observed(
            case,
            prepared,
            NoVendorModule(),
            1,
            "measured",
            samples,
            output_sha256,
            False,
        )


def test_integrity_enabled_is_required_unless_diagnostic_override_is_explicit(
    tmp_path,
):
    module_path = tmp_path / "gxeldcore.cpython-312-x86_64-linux-gnu.so"
    module_path.write_bytes(b"native-test-placeholder")
    archive_hash = "b" * 64

    class FakeModule:
        @staticmethod
        def build_info():
            return {
                "api_version": 8,
                "backend_version": "1.5",
                "blas_runtime_isolation": "private_static",
                "gemm_vendor_entry_outer_openmp_guard": True,
                "optimized_fp64_layout": (
                    "source_column_major_tt_target_row_major_tn_v1"
                ),
                "optimized_fp64_row_pair_input_mode": "mprotect_read_only",
                "private_openblas_archive_sha256": archive_hash,
                "source_tree_sha256": "c" * 64,
                "source_commit": "d" * 40,
                "gemm_integrity_enabled": False,
            }

    with pytest.raises(RuntimeError, match="integrity-enabled"):
        benchmark._native_provenance(FakeModule(), module_path, archive_hash, False)
    diagnostic = benchmark._native_provenance(
        FakeModule(), module_path, archive_hash, True
    )
    assert diagnostic["integrity_mode"] == {
        "enabled": False,
        "required_for_evidence": False,
        "diagnostic_override_used": True,
        "acceptance_eligible": False,
    }


def test_runtime_caps_memory_bound_and_no_overwrite_are_explicit(tmp_path):
    assert benchmark.MIN_WARMUPS == 3
    assert benchmark.MIN_REPEATS == 5
    assert benchmark.MAX_CONFIGURATION_SECONDS == 90.0
    assert benchmark.MAX_SWEEP_SECONDS == 1200.0
    n, k, p, threads = 289_111, 2000, 192, 32
    source = benchmark.Case("source", n, k, 32, 3, threads)
    target = benchmark.Case("target", n, k, 32, 3, threads)
    for case in (source, target):
        memory = benchmark._estimated_memory(case)
        named = (
            memory["genotype_f_bytes"]
            + memory["weights_f_bytes"]
            + memory["output_c_bytes"]
            if case.operation == "source"
            else memory["genotype_f_bytes"]
            + memory["source_panel_f_bytes"]
            + memory["environment_f_bytes"]
            + memory["sealed_pair_bytes"]
            + memory["output_f_bytes"]
        )
        snapshot = 8 * k * p if case.operation == "source" else 0
        coefficient = (
            8
            * benchmark.INTEGRITY_CHECK_COUNT
            * (p if case.operation == "source" else k)
        )
        projection = (
            8
            * benchmark.INTEGRITY_CHECK_COUNT
            * (k if case.operation == "source" else n)
        )
        expected_observed = (
            8
            * 2
            * benchmark.INTEGRITY_CHECK_COUNT
            * (n if case.operation == "source" else 2 * p)
        )
        check_scratch = coefficient + projection + expected_observed
        candidate_owned = named + snapshot + check_scratch
        stacks = (threads - 1) * 8 * benchmark.MIB
        allocator_slack = max(benchmark.GIB // 2, (candidate_owned + 9) // 10)
        subtotal = candidate_owned + stacks + allocator_slack + 16 * benchmark.GIB
        assert memory["integrity_weight_snapshot_bytes"] == snapshot
        assert memory["integrity_coefficient_bytes"] == coefficient
        assert memory["integrity_projection_bytes"] == projection
        assert memory["integrity_expected_observed_bytes"] == expected_observed
        assert memory["integrity_check_scratch_bytes"] == check_scratch
        assert memory["candidate_owned_peak_bytes"] == candidate_owned
        assert memory["additional_thread_stacks_bytes"] == stacks
        assert memory["allocator_slack_bytes"] == allocator_slack
        assert memory["vendor_workspace_allowance_bytes"] == 16 * benchmark.GIB
        assert memory["modeled_subtotal_before_headroom_bytes"] == subtotal
        assert memory["headroom_fraction"] == 0.20
        assert memory["conservative_peak_bytes"] == (subtotal * 120 + 99) // 100
        assert memory["conservative_peak_gib"] < 96.0
        assert memory["vendor_workspace_bound_source"] == (
            "strict_16gib_uncertified_private_openblas_fallback"
        )
        assert "not used as the bound" in memory["vendor_workspace_bound_uncertainty"]

    output = tmp_path / "report.json"
    benchmark._write_json({"status": "first"}, output)
    with pytest.raises(FileExistsError):
        benchmark._write_json({"status": "second"}, output)
    assert not list(tmp_path.glob(".report.json.tmp.*"))
    missing_parent = tmp_path / "missing" / "report.json"
    with pytest.raises(FileNotFoundError, match="parent must already exist"):
        benchmark._write_json({"status": "missing"}, missing_parent)
    assert not missing_parent.parent.exists()


def test_protocol_and_hard_cap_validation_require_explicit_smoke_escape():
    parser = benchmark._parser()
    assert parser.parse_args(["--dry-run"]).allow_integrity_disabled is False
    assert (
        parser.parse_args(
            ["--dry-run", "--allow-integrity-disabled"]
        ).allow_integrity_disabled
        is True
    )
    short = parser.parse_args(["--dry-run", "--warmups", "0", "--repeats", "1"])
    with pytest.raises(SystemExit):
        benchmark._validate_args(parser, short)
    allowed = parser.parse_args(
        [
            "--dry-run",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--allow-short-protocol",
        ]
    )
    benchmark._validate_args(parser, allowed)
    too_long = parser.parse_args(["--dry-run", "--max-configuration-seconds", "90.01"])
    with pytest.raises(SystemExit):
        benchmark._validate_args(parser, too_long)


def test_controller_outcome_rejects_exact_ineligibility_and_incomplete_cases():
    eligible = {
        "status": "ok",
        "evidence_acceptance": {"eligible": True},
    }
    ineligible = {
        "status": "ok",
        "evidence_acceptance": {"eligible": False},
    }
    skipped = {"status": "skipped_memory_bound"}

    production_rejected = benchmark._controller_outcome(
        [eligible, eligible], evidence_requested=True
    )
    assert production_rejected == {
        "status": "rejected",
        "execution_complete": True,
        "evidence_requested": True,
        "eligible": False,
        "eligible_case_count": 2,
        "case_count": 2,
    }
    rejected = benchmark._controller_outcome(
        [eligible, ineligible], evidence_requested=True
    )
    assert rejected["status"] == "rejected"
    assert rejected["execution_complete"] is True
    assert rejected["eligible"] is False
    incomplete = benchmark._controller_outcome(
        [eligible, skipped], evidence_requested=True
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["execution_complete"] is False
    diagnostic = benchmark._controller_outcome(
        [ineligible], evidence_requested=False
    )
    assert diagnostic["status"] == "diagnostic_complete"
    assert diagnostic["execution_complete"] is True
    assert diagnostic["eligible"] is False
