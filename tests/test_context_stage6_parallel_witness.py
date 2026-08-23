from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "native"
    / "contextual_streamed_reference_v1.inc"
)
RESULT_PREFIX = "SUMMIT_STAGE6_PARALLEL_WITNESS="
EXECUTION_BACKEND = "deterministic_tiled_fp64_with_scalar_witness_v1"
RECOVERABLE_MODES = (
    "one_shot",
    "repeated",
    "repair_corruption",
    "force_fallback",
    "nan",
    "inf",
    "canary",
)
TERMINAL_MODES = (
    "fallback_corruption",
    "fallback_failure",
    "operand_mutation",
)


CHILD_SOURCE = r"""
import json
from pathlib import Path
import sys

import numpy as np

from summit import gxeldcore

sys.path.insert(0, str(Path.cwd() / "tests"))
from test_context_stage2_streamed_reference import _make_case
from test_context_stage3_complete_reference_native import (
    _stage3_executor,
    _variant_probes,
)
from test_context_stage4_trait_native import (
    SCIENCE_ARRAYS as TRAIT_SCIENCE_ARRAYS,
    _trait_executor,
    _trait_inputs,
)


RESULT_PREFIX = "SUMMIT_STAGE6_PARALLEL_WITNESS="
REFERENCE_SCIENCE_ARRAYS = (
    "gram",
    "raw_gram_numerator",
    "annotation_masses",
    "same_person",
    "group_gram_unnormalized_num",
    "group_annotation_masses",
    "group_variant_counts",
    "pair_q",
    "pair_r",
    "pair_eta",
    "component_annotation",
    "component_pair",
)
RECOVERABLE_MODES = (
    "one_shot",
    "repeated",
    "repair_corruption",
    "force_fallback",
    "nan",
    "inf",
    "canary",
)
TERMINAL_MODES = (
    "fallback_corruption",
    "fallback_failure",
    "operand_mutation",
)


def science_evidence(result, names):
    arrays = {}
    for name in names:
        value = np.asarray(result[name])
        arrays[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "c_order_bytes_hex": np.ascontiguousarray(value).tobytes().hex(),
        }
    return {
        "arrays": arrays,
        "published_sha256": dict(result["scientific_array_sha256"]),
    }


def fingerprint_trace(result):
    keys = (
        "sequence",
        "operation",
        "semantic_anchor",
        "attempt",
        "rows",
        "columns",
        "reduction",
        "transpose_left",
        "operand_fingerprint_fnv64",
        "witness_fingerprint_fnv64",
        "accepted_output_fingerprint_fnv64",
    )
    return [
        {key: event[key] for key in keys}
        for event in result["telemetry"]["events"]
        if event["event_class"] == "semantic_verification"
    ]


def performance_evidence(executor, result):
    report = dict(executor.performance_report())
    operations = {
        name: dict(value) for name, value in report["operations"].items()
    }
    called = {
        name: {
            "calls": value["calls"],
            "logical_flop_count": value["logical_flop_count"],
            "output_elements": value["output_elements"],
            "witness_wall_ns": value["witness_wall_ns"],
            "shape_histogram": [dict(shape) for shape in value["shape_histogram"]],
        }
        for name, value in operations.items()
        if value["calls"]
    }
    return {
        "report_keys": sorted(report),
        "operation_keys": sorted(next(iter(operations.values()))),
        "result_keys": sorted(result),
        "blas_threads": report["runtime_policy"]["blas_threads"],
        "witness_wall_ns": report["totals"]["witness_wall_ns"],
        "called_operations": called,
    }


def identity(result):
    return {
        "contextual_native_api_version": result["contextual_native_api_version"],
        "contextual_backend_version": result["contextual_backend_version"],
        "contextual_backend": result["contextual_backend"],
        "contextual_execution_backend": result["contextual_execution_backend"],
    }


def anchor(executor, operation):
    matches = [
        value["semantic_anchor"]
        for value in executor.semantic_anchors()
        if value["operation"] == operation
    ]
    if len(matches) != 1:
        raise AssertionError((operation, matches))
    return matches[0]


def assert_science_close(observed, expected, names):
    for name in names:
        np.testing.assert_allclose(
            observed[name], expected[name], rtol=2.0e-12, atol=2.0e-12
        )


def fault_evidence(
    make_executor,
    operation,
    operation_anchor,
    baseline,
    science_names,
):
    recovered = {}
    for mode in RECOVERABLE_MODES:
        executor = make_executor(
            fault_operation=operation,
            fault_mode=mode,
            fault_semantic_anchor=operation_anchor,
        )
        result = dict(executor.run())
        assert_science_close(result, baseline, science_names)
        events = [dict(value) for value in result["telemetry"]["events"]]
        recovered[mode] = {
            "injection_count": result["telemetry"]["injection_count"],
            "retry_count": result["telemetry"]["retry_count"],
            "repair_count": result["telemetry"]["repair_count"],
            "fallback_count": result["telemetry"]["fallback_count"],
            "fallback_resolutions": [
                value["resolution"]
                for value in events
                if value["event_class"] == "trusted_fallback"
            ],
            "detection_count": sum(
                value["event_class"] == "fault_detection" for value in events
            ),
        }

    terminal = {}
    for mode in TERMINAL_MODES:
        executor = make_executor(
            fault_operation=operation,
            fault_mode=mode,
            fault_semantic_anchor=operation_anchor,
        )
        try:
            executor.run()
        except RuntimeError as error:
            message = str(error)
            report = dict(executor.failure_report())
        else:
            raise AssertionError((operation, mode, "unexpected success"))
        events = [dict(value) for value in report["events"]]
        terminal[mode] = {
            "message": message,
            "event_ledger_complete_without_drop": report[
                "event_ledger_complete_without_drop"
            ],
            "terminal_event_recorded": report["terminal_event_recorded"],
            "terminal_event_unique": report["terminal_event_unique"],
            "terminal_event_last": report["terminal_event_last"],
            "fallback_resolutions": [
                value["resolution"]
                for value in events
                if value["event_class"] == "trusted_fallback"
            ],
            "last_event_class": events[-1]["event_class"],
        }
    return {"recovered": recovered, "terminal": terminal}


threads = int(sys.argv[1])
cpu_ids = json.loads(sys.argv[2])
work = Path(sys.argv[3])
run_faults = sys.argv[4] == "1"
gxeldcore.configure_openmp_placement(cpu_ids, threads)

case = _make_case(work, q_count=2, name="parallel-witness")
probes = _variant_probes(case)
reference_options = {
    "variant_block": 3,
    "sample_probe_resident": 2,
    "sample_probe_tile": 1,
    "action_tile": 2,
    "annotation_tile": 1,
    "context_tile": 1,
    "variant_probe_tile": 2,
    "group_tile": 2,
    "decode_threads": 1,
    "blas_threads": threads,
}


def make_reference(**changes):
    options = dict(reference_options)
    options.update(changes)
    return _stage3_executor(case, probes, **options)


reference_executor = make_reference()
reference_anchor = anchor(reference_executor, "source_tn")
reference_result = dict(reference_executor.run())

phenotypes, residual_basis = _trait_inputs(case)
trait_options = {
    "variant_block": 3,
    "trait_feature_tile": 2,
    "decode_threads": 1,
    "blas_threads": threads,
}


def make_trait(**changes):
    options = dict(trait_options)
    options.update(changes)
    return _trait_executor(case, phenotypes, residual_basis, **options)


trait_executor = make_trait()
trait_anchor = anchor(trait_executor, "trait_score_tn")
trait_result = dict(trait_executor.run())

payload = {
    "threads": threads,
    "reference": {
        "identity": identity(reference_result),
        "science": science_evidence(reference_result, REFERENCE_SCIENCE_ARRAYS),
        "trace": fingerprint_trace(reference_result),
        "performance": performance_evidence(reference_executor, reference_result),
    },
    "trait": {
        "identity": identity(trait_result),
        "science": science_evidence(trait_result, TRAIT_SCIENCE_ARRAYS),
        "trace": fingerprint_trace(trait_result),
        "performance": performance_evidence(trait_executor, trait_result),
    },
}
if run_faults:
    payload["faults"] = {
        "reference": fault_evidence(
            make_reference,
            "source_tn",
            reference_anchor,
            reference_result,
            REFERENCE_SCIENCE_ARRAYS,
        ),
        "trait": fault_evidence(
            make_trait,
            "trait_score_tn",
            trait_anchor,
            trait_result,
            TRAIT_SCIENCE_ARRAYS,
        ),
    }

print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
"""


def _cpu_ids() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("exact OpenMP placement requires Linux affinity APIs")
    values = sorted(os.sched_getaffinity(0))
    if len(values) < 2:
        pytest.skip("parallel witness test requires two visible CPUs")
    return values[:2]


def _environment(cpu_ids: list[int]) -> dict[str, str]:
    environment = dict(os.environ)
    paths = [
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parents[1]),
        environment.get("PYTHONPATH", ""),
        *(value for value in sys.path if value),
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys(value for value in paths if value)
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("GOMP_CPU_AFFINITY", None)
    for name in (
        "BLIS_NT",
        "BLIS_TI",
        "BLIS_THREAD_IMPL",
        "BLIS_ARCH_TYPE",
        "BLIS_ARCH_DEBUG",
        "BLIS_PACK_A",
        "BLIS_PACK_B",
        "BLIS_JC_NT",
        "BLIS_PC_NT",
        "BLIS_IC_NT",
        "BLIS_JR_NT",
        "BLIS_IR_NT",
    ):
        environment.pop(name, None)
    threads = len(cpu_ids)
    environment.update(
        {
            "BLIS_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "OMP_THREAD_LIMIT": str(threads),
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_PROC_BIND": "SPREAD",
            "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in cpu_ids),
        }
    )
    return environment


def _run_child(
    tmp_path: Path,
    cpu_ids: list[int],
    *,
    run_faults: bool,
) -> dict[str, Any]:
    work = tmp_path / f"threads-{len(cpu_ids)}"
    work.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(CHILD_SOURCE),
            str(len(cpu_ids)),
            json.dumps(cpu_ids),
            str(work),
            "1" if run_faults else "0",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=_environment(cpu_ids),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 and (
        "explicit OpenMP placement contract is unavailable" in completed.stderr
        or "No module named 'summit'" in completed.stderr
        or "cannot import name 'gxeldcore'" in completed.stderr
    ):
        pytest.skip("the OpenMP contextual native extension is unavailable")
    assert completed.returncode == 0, completed.stderr
    records = [
        line[len(RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(records) == 1, completed.stdout
    return json.loads(records[0])


def _assert_performance_fields(evidence: dict[str, Any], *, threads: int) -> None:
    performance = evidence["performance"]
    assert performance["blas_threads"] == threads
    assert performance["witness_wall_ns"] > 0
    assert performance["witness_wall_ns"] == sum(
        operation["witness_wall_ns"]
        for operation in performance["called_operations"].values()
    )
    for operation in performance["called_operations"].values():
        assert operation["calls"] > 0
        assert operation["logical_flop_count"] > 0.0
        assert operation["output_elements"] > 0
        assert operation["witness_wall_ns"] > 0


def test_serial_scalar_witness_is_exact_when_primary_threads_change(
    tmp_path: Path,
) -> None:
    cpu_ids = _cpu_ids()
    single = _run_child(tmp_path, cpu_ids[:1], run_faults=False)
    threaded_primary = _run_child(tmp_path, cpu_ids, run_faults=True)

    expected_backends = {
        "reference": "plink_bed_descriptor_stream_stage2_v1",
        "trait": "plink_bed_descriptor_stream_trait_v1",
    }
    for mode in ("reference", "trait"):
        one = single[mode]
        many = threaded_primary[mode]
        assert one["science"] == many["science"]
        assert one["trace"] == many["trace"]
        assert one["trace"]
        assert (
            one["identity"]
            == many["identity"]
            == {
                "contextual_native_api_version": 1,
                "contextual_backend_version": 2,
                "contextual_backend": expected_backends[mode],
                "contextual_execution_backend": EXECUTION_BACKEND,
            }
        )
        _assert_performance_fields(one, threads=1)
        _assert_performance_fields(many, threads=2)
        assert one["performance"]["report_keys"] == many["performance"]["report_keys"]
        assert (
            one["performance"]["operation_keys"]
            == many["performance"]["operation_keys"]
        )
        assert one["performance"]["result_keys"] == many["performance"]["result_keys"]
        for name, operation in one["performance"]["called_operations"].items():
            parallel_operation = many["performance"]["called_operations"][name]
            for key in (
                "calls",
                "logical_flop_count",
                "output_elements",
                "shape_histogram",
            ):
                assert operation[key] == parallel_operation[key], (mode, name, key)

    for mode in ("reference", "trait"):
        recovered = threaded_primary["faults"][mode]["recovered"]
        assert set(recovered) == set(RECOVERABLE_MODES)
        for fault_mode, evidence in recovered.items():
            assert evidence["injection_count"] == 1, (mode, fault_mode)
            assert evidence["detection_count"] >= 1, (mode, fault_mode)
            if evidence["fallback_count"]:
                assert evidence["fallback_resolutions"] == ["one_thread_scalar_fp64"]

        terminal = threaded_primary["faults"][mode]["terminal"]
        assert set(terminal) == set(TERMINAL_MODES)
        for fault_mode, evidence in terminal.items():
            assert evidence["event_ledger_complete_without_drop"] is True
            assert evidence["terminal_event_recorded"] is True
            assert evidence["terminal_event_unique"] is True
            assert evidence["terminal_event_last"] is True
            assert evidence["last_event_class"] == "terminal_failure"
            if fault_mode.startswith("fallback_"):
                assert evidence["fallback_resolutions"] == ["one_thread_scalar_fp64"]


def test_rejected_parallel_witness_is_serial_and_guards_remain_lazy() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    witness_begin = source.index("void ContextualReferenceExecutorV1::scalar_gemm(")
    fallback_begin = source.index(
        "void ContextualReferenceExecutorV1::scalar_fallback_gemm("
    )
    witness = source[witness_begin:fallback_begin]
    fallback_end = source.index(
        "void ContextualReferenceExecutorV1::protected_tn(", fallback_begin
    )
    fallback = source[fallback_begin:fallback_end]
    assert "#pragma omp" not in witness
    assert "requested_threads" not in witness
    assert "witness_threads" not in witness
    column_loop = witness.index("for (int column = 0; column < columns; ++column)")
    row_loop = witness.index("for (int row = 0; row < rows; ++row)")
    reduction_loop = witness.index("for (int inner = 0; inner < reduction; ++inner)")
    assert column_loop < row_loop < reduction_loop
    assert "#pragma omp" not in fallback
    assert "reduction-major accumulation" in fallback

    protected_begin = source.index(
        "void ContextualReferenceExecutorV1::protected_gemm("
    )
    protected = source[protected_begin:]
    initial_fill = protected.index("&primary_buffer_, &witness_buffer_")
    retry_branch = protected.index("} else if (!force_fallback) {")
    retry_fill = protected.index("retry_buffer_.begin()", retry_branch)
    fallback_branch = protected.index("if (accepted == nullptr) {")
    fallback_fill = protected.index("fallback_buffer_.begin()", fallback_branch)
    assert initial_fill < retry_branch < retry_fill < fallback_branch < fallback_fill
    assert "primary_buffer_.assign(protected_count, kCanaryV1);" in source
    assert "witness_buffer_.assign(protected_count, kCanaryV1);" in source
    assert "retry_buffer_.assign(protected_count, kCanaryV1);" in source
    assert "fallback_buffer_.assign(protected_count, kCanaryV1);" in source
    assert '"one_thread_scalar_fp64"' in protected
