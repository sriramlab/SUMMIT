from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gxe import run_ceiling_orientation_benchmark as harness


def _arguments(*extra: str):
    parser = harness._parser()
    args = parser.parse_args(["--dry-run", *extra])
    harness._validate(args, parser)
    return args


def test_smoke_matrix_covers_both_ceilings_stream_and_literal_layouts():
    cases = harness._cases(_arguments())
    assert len(cases) == 11
    assert [(case.mode, case.dtype) for case in cases[:3]] == [
        ("square", "f64"),
        ("square", "f32"),
        ("stream", "f64"),
    ]
    exact = cases[3:]
    assert {case.mode for case in exact} == {"source", "target"}
    assert {case.layout for case in exact} == {"col", "row"}
    assert {case.orientation for case in exact} == {"current", "transposed"}
    assert {case.dtype for case in exact} == {"f64"}


def test_exact_shape_metadata_and_memory_estimates_match_phase_panels():
    args = _arguments(
        "--profile", "exact",
        "--modes", "source,target",
        "--layouts", "col",
        "--orientations", "current",
    )
    source, target = harness._cases(args)
    assert (source.n_samples, source.block_width) == (289_111, 2000)
    assert (source.probe_tile, source.environment_tile) == (32, 1)
    assert source.panel_width == 64
    assert target.panel_width == 128
    assert source.estimated_operand_bytes() == 8 * (
        289_111 * 2000 + 2000 * 64 + 289_111 * 64
    )
    assert target.estimated_operand_bytes() == 8 * (
        289_111 * 2000 + 289_111 * 128 + 2000 * 128
    )


def test_runtime_caps_and_protocol_minima_are_hard_limits():
    assert harness.MAX_CONFIGURATION_SECONDS == 90.0
    assert harness.MAX_SWEEP_SECONDS == 1200.0
    parser = harness._parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args(["--dry-run", "--max-configuration-seconds", "91"])
        harness._validate(args, parser)
    with pytest.raises(SystemExit):
        args = parser.parse_args(["--dry-run", "--max-sweep-seconds", "1201"])
        harness._validate(args, parser)
    with pytest.raises(SystemExit):
        args = parser.parse_args(["--dry-run", "--warmups", "2"])
        harness._validate(args, parser)
    with pytest.raises(SystemExit):
        args = parser.parse_args(["--dry-run", "--repeats", "4"])
        harness._validate(args, parser)


def test_compile_and_complete_protocol_smoke_against_preserved_private_archive(tmp_path):
    archive = harness.DEFAULT_ARCHIVE
    if not archive.is_file():
        pytest.skip("preserved private OpenBLAS audit archive is not available")
    output = tmp_path / "ceiling_orientation_smoke.json"
    exit_code = harness.main([
        "--profile", "smoke",
        "--modes", "dgemm,sgemm,stream,source,target",
        "--threads", "1",
        "--size", "128",
        "--n", "512",
        "--k", "64",
        "--probe-tile", "2",
        "--environment-tile", "1",
        "--stream-elements", "262144",
        "--max-configuration-seconds", "30",
        "--max-sweep-seconds", "120",
        "--build-dir", str(tmp_path / "build"),
        "--output", str(output),
        "--archive", str(archive),
        "--expected-archive-sha256", harness.DEFAULT_ARCHIVE_SHA256,
    ])
    assert exit_code == 0
    report = json.loads(output.read_text())
    assert report["provenance"]["linkage"]["private_static_link_verified"] is True
    assert report["provenance"]["linkage"]["dynamic_blas_dependencies"] == []
    assert len(report["results"]) == 11
    assert report["provenance"]["archive_capabilities"]["cblas_dgemm"] is True
    assert report["provenance"]["archive_capabilities"]["cblas_sgemm"] is False
    assert [
        item["case"]["label"] for item in report["results"]
        if item["status"] == "unsupported_backend_capability"
    ] == ["sgemm.col.T1"]
    successful = [item for item in report["results"] if item["status"] == "ok"]
    assert len(successful) == 10
    for item in successful:
        result = item["result"]
        assert result["backend"]["archive_sha256"] == harness.DEFAULT_ARCHIVE_SHA256
        assert result["backend"]["openblas_threads"] == 1
        assert result["protocol"] == {
            "warmups": 3,
            "timed_repeats": 5,
            "one_serial_vendor_entry_per_process": True,
            "full_matrix_transpose_performed": False,
        }
        assert result["correctness"]["finite"] is True
        assert result["correctness"]["within_roundoff_bound"] is True
        assert result["correctness"][
            "repeated_outputs_bitwise_equal_at_samples"
        ] is True
        assert result["summary"]["median_wall_seconds"] > 0.0
        assert result["summary"]["median_process_cpu_seconds"] > 0.0
        if item["case"]["mode"] != "stream":
            for repeat in result["measurements"]:
                assert repeat["vendor_entry"]["omp_in_parallel"] == 0
                assert repeat["vendor_entry"]["omp_get_level"] == 0
                assert repeat["vendor_entry"]["omp_get_active_level"] == 0

    by_label = {item["case"]["label"]: item["result"] for item in successful}
    assert by_label["source.f64.row.transposed.T1"]["cblas"] == {
        "layout": "CblasRowMajor",
        "transpose_a": "CblasTrans",
        "transpose_b": "CblasTrans",
        "m": 4,
        "n": 512,
        "k": 64,
        "lda": 4,
        "ldb": 64,
        "ldc": 512,
    }
    assert by_label["target.f64.col.current.T1"]["cblas"] == {
        "layout": "CblasColMajor",
        "transpose_a": "CblasTrans",
        "transpose_b": "CblasNoTrans",
        "m": 64,
        "n": 8,
        "k": 512,
        "lda": 512,
        "ldb": 512,
        "ldc": 64,
    }
