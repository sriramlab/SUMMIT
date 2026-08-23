from __future__ import annotations

import gzip
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.gxe import benchmark_real_genotype_partial as benchmark


def _synthetic_plink(root: Path, *, samples: int = 5, variants: int = 10) -> Path:
    prefix = root / "source"
    bytes_per_variant = (samples + 3) // 4
    body = bytes((index * 19 + 7) % 256 for index in range(variants * bytes_per_variant))
    Path(f"{prefix}.bed").write_bytes(benchmark.BED_MAGIC + body)
    Path(f"{prefix}.bim").write_text(
        "".join(f"1 rs{index} 0 {index + 1} A C\n" for index in range(variants)),
        encoding="utf-8",
    )
    Path(f"{prefix}.fam").write_text(
        "".join(f"F{index} I{index} 0 0 0 -9\n" for index in range(samples)),
        encoding="utf-8",
    )
    return prefix


def _tabular_inputs(root: Path, samples: int = 5) -> tuple[Path, Path]:
    environment = root / "environment.tsv"
    covariates = root / "covariates.tsv"
    header = "FID\tIID\tage\tsex\tbmi\n"
    environment.write_text(
        header + "".join(
            f"F{i}\tI{i}\t{i + 0.1}\t{i % 2}\t{20 + i}\n" for i in range(samples)
        ),
        encoding="utf-8",
    )
    covariates.write_text(
        "FID\tIID\tC\n" + "".join(f"F{i}\tI{i}\t{i / 10}\n" for i in range(samples)),
        encoding="utf-8",
    )
    return environment, covariates


def _blis_cli_identity_args(
    root: Path, install: Path, native: Path
) -> list[str]:
    archive = root / "libblis-private.a"
    if not archive.exists():
        archive.write_bytes(b"test-only private BLIS archive")
    return [
        "--expected-backend", "blis",
        "--expected-native-sha256", benchmark._sha256(native),
        "--expected-package-manifest-sha256",
        benchmark._installed_package_identity(install)["manifest_sha256"],
        "--private-archive", str(archive),
        "--expected-archive-sha256", benchmark._sha256(archive),
        "--expected-source-commit", "a" * 40,
        "--expected-source-tree-sha256", "b" * 64,
        "--expected-private-source-commit", "c" * 40,
        "--expected-private-source-tree-sha256", "d" * 64,
    ]


def _placeholder_blis_cli_args(root: Path) -> list[str]:
    return [
        "--expected-backend", "blis",
        "--expected-native-sha256", "a" * 64,
        "--expected-package-manifest-sha256", "b" * 64,
        "--private-archive", str(root / "unused-libblis.a"),
        "--expected-archive-sha256", "c" * 64,
        "--expected-source-commit", "d" * 40,
        "--expected-source-tree-sha256", "e" * 64,
        "--expected-private-source-commit", "f" * 40,
        "--expected-private-source-tree-sha256", "0" * 64,
    ]


def test_literal_prefix_subset_has_exact_bed_bim_and_fam_bytes(tmp_path, monkeypatch):
    source_prefix = _synthetic_plink(tmp_path)
    source = benchmark._validate_plink_prefix(source_prefix)
    destination = tmp_path / "subset" / "genotype"
    subset, identities = benchmark._create_prefix_subset(source, destination, 8)

    assert subset.samples == 5
    assert subset.variants == 8
    assert subset.bytes_per_variant == 2
    assert Path(f"{destination}.bed").read_bytes() == Path(
        f"{source_prefix}.bed"
    ).read_bytes()[: 3 + 8 * 2]
    assert Path(f"{destination}.bim").read_bytes() == b"".join(
        Path(f"{source_prefix}.bim").read_bytes().splitlines(keepends=True)[:8]
    )
    assert Path(f"{destination}.fam").read_bytes() == Path(
        f"{source_prefix}.fam"
    ).read_bytes()
    assert identities["bed"]["bytes"] == 19
    assert all(len(record["sha256"]) == 64 for record in identities.values())

    subset_paths = [
        Path(f"{destination}{extension}")
        for extension in (".bed", ".bim", ".fam")
    ]
    stat_identities = {
        str(path): benchmark._file_identity(path) for path in subset_paths
    }
    benchmark._assert_staged_subset_integrity(
        subset_paths, stat_identities, identities
    )
    bed = Path(f"{destination}.bed")
    changed = bytearray(bed.read_bytes())
    changed[-1] ^= 1
    bed.write_bytes(changed)
    monkeypatch.setattr(benchmark, "_same_file_identity", lambda *_: True)
    with pytest.raises(RuntimeError, match="subset hash changed"):
        benchmark._assert_staged_subset_integrity(
            subset_paths, stat_identities, identities
        )


def test_affinity_normalization_accepts_native_ranges_and_python_lists() -> None:
    assert benchmark._affinity_cpu_set("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
    assert benchmark._affinity_cpu_set([0, 2, 4]) == {0, 2, 4}
    assert benchmark._affinity_cpu_set(None) is None
    with pytest.raises(ValueError, match="unsupported GEMM affinity"):
        benchmark._affinity_cpu_set({0, 1})


def test_child_environment_scrubs_inherited_affinity_and_caps_openmp(tmp_path, monkeypatch):
    package = tmp_path / "summit"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    native = package / "gxeldcore.test.so"
    native.write_bytes(b"native")
    monkeypatch.setenv("OMP_PLACES", "{0},{1}")
    monkeypatch.setenv("GOMP_CPU_AFFINITY", "0 1")
    monkeypatch.setenv("KMP_AFFINITY", "compact")
    monkeypatch.setenv("OMP_NESTED", "TRUE")
    monkeypatch.setenv("PYTHONHOME", "/stale/python")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/stale/pycache")
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_APPLIED", "forged")
    monkeypatch.setenv("SUMMIT_NUMA_POLICY_PROVENANCE", "forged")
    for name in benchmark.BLIS_AUTOMATIC_CONFLICTS:
        monkeypatch.setenv(name, "forged")
    args = SimpleNamespace(
        threads=3,
        allow_integrity_disabled=False,
        expected_source_commit="a" * 40,
        expected_source_tree_sha256="b" * 64,
        expected_archive_sha256="c" * 64,
        expected_private_source_commit="d" * 40,
        expected_private_source_tree_sha256="e" * 64,
    )
    cpu_records = [
        benchmark.CpuRecord(cpu=cpu, core=cpu, socket=0, node=0)
        for cpu in (2, 4, 6)
    ]
    environment, settings = benchmark._child_environment(
        args, tmp_path, native, [tmp_path], cpu_records
    )
    assert environment["OMP_PLACES"] == "{2},{4},{6}"
    assert "GOMP_CPU_AFFINITY" not in environment
    assert "KMP_AFFINITY" not in environment
    assert "OMP_NESTED" not in environment
    assert environment["OMP_PROC_BIND"] == "SPREAD"
    assert environment["BLIS_NUM_THREADS"] == "3"
    assert all(name not in environment for name in benchmark.BLIS_AUTOMATIC_CONFLICTS)
    assert "PYTHONHOME" not in environment
    assert "PYTHONPYCACHEPREFIX" not in environment
    assert "SUMMIT_NUMA_POLICY_APPLIED" not in environment
    assert "SUMMIT_NUMA_POLICY_PROVENANCE" not in environment
    assert settings["OMP_THREAD_LIMIT"] == "3"
    assert settings["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert environment["OMP_THREAD_LIMIT"] == "3"
    assert environment["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert settings["SUMMIT_GXE_EXPECTED_INSTALL_PREFIX"] == str(tmp_path)
    assert settings["SUMMIT_GXE_REQUIRE_EARLY_NUMA"] == "1"
    assert len(settings["SUMMIT_GXE_EXPECTED_PACKAGE_MANIFEST_SHA256"]) == 64
    ordered_bootstrap_gates = [
        "if manifest.hexdigest() != expected_package_manifest_sha256",
        "early_numa_attestation = preconfigure_numa_from_argv",
        "from summit import cli as summit_cli",
        "from summit import gxeldcore",
        "if observed != expected:",
        "if digest != expected_sha256:",
        "build = dict(gxeldcore.build_info())",
        'if require_integrity and build.get("gemm_integrity_enabled") is not True:',
        'if not callable(getattr(gxeldcore, "consume_gemm_telemetry", None)):',
        "summit_cli.main()",
    ]
    positions = [
        benchmark._CLI_BOOTSTRAP.index(marker)
        for marker in ordered_bootstrap_gates
    ]
    assert positions == sorted(positions)

    py_compile.compile(str(package / "__init__.py"), doraise=True)
    with pytest.raises(RuntimeError, match="executable bytecode/cache"):
        benchmark._installed_package_identity(tmp_path)


def test_bootstrap_subprocess_captures_affinity_before_mock_native_narrows_it(
    tmp_path,
):
    allowed = sorted(os.sched_getaffinity(0))
    taskset = shutil.which("taskset")
    if len(allowed) < 2 or taskset is None:
        pytest.skip("two allowed CPUs and taskset are required")
    selected = allowed[:2]
    install = tmp_path / "install"
    package = install / "summit"
    package.mkdir(parents=True)
    log = tmp_path / "bootstrap-order.jsonl"
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_early_numa.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            _LOG = Path(@LOG@)
            def preconfigure_numa_from_argv(argv):
                with _LOG.open("a", encoding="utf-8") as handle:
                    handle.write('{"event": "early_numa"}\\n')
                return {"verified": True}
            """
        ).replace("@LOG@", repr(str(log))).lstrip(),
        encoding="utf-8",
    )
    (package / "cli.py").write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            _LOG = Path(@LOG@)
            CAPTURED = sorted(os.sched_getaffinity(0))
            with _LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "cli_import", "affinity": CAPTURED}) + "\\n")
            def main():
                with _LOG.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "cli_main", "affinity": sorted(os.sched_getaffinity(0))}) + "\\n")
            """
        ).replace("@LOG@", repr(str(log))).lstrip(),
        encoding="utf-8",
    )
    build = {
        "api_version": 9, "backend_version": "1.9",
        "source_commit": "a" * 40, "source_tree_sha256": "b" * 64,
        "blas_vendor": "BLIS", "blas_runtime_isolation": "private_static",
        "gemm_execution_mode": "serialized_fixed_private_blis",
        "private_openblas_archive_sha256": "none",
        "private_blas_backend": "upstream_blis",
        "private_blas_archive_sha256": "c" * 64,
        "private_blas_source_commit": "d" * 40,
        "private_blas_source_tree_sha256": "e" * 64,
        "private_blas_config_family": "zen",
        "blas_runtime_config": "BLIS 2.0 config=zen",
        "blas_runtime_corename": "zen", "blas_runtime_threads": 2,
        "blas_runtime_threading_layer": "pthreads",
        "blas_runtime_worker_affinity_policy": (
            "inherit_authenticated_selected_cpu_set_per_call"
        ),
        "blas_runtime_thread_strategy": "automatic",
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_owner_thread_configured": True,
        "blas_runtime_environment_immutable": True,
        "blas_runtime_environment_contract": "blis_process_start_v1",
        "blas_runtime_tls_enabled": True,
        "gemm_vendor_entry_outer_openmp_guard": True,
        "openmp_effective_capacity_policy": "bound_places_else_sched_affinity_v1",
        "openmp_placement_contract_supported": True,
        "openmp_placement_contract_schema": "summit.openmp_placement_attestation.v1",
        "openmp_placement_contract_configured": False,
        "openmp_placement_contract_immutable": True,
        "openmp_placement_probe_vendor_calls": 0,
        "openmp_placement_contract_evidence": None,
        "blas_runtime_thread_ways": {
            "jc": 1, "pc": 1, "ic": 1, "jr": 1, "ir": 1,
        },
        "private_blas_header_sha256": "f" * 64,
        "private_blas_cblas_header_sha256": "0" * 64,
        "gemm_integrity_enabled": True,
        "gemm_integrity_minimum_vendor_flops": 1_000_000_000,
        "native_integrity_snapshot_numa_contract_supported": True,
        "native_integrity_snapshot_numa_contract_schema": (
            benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA
        ),
        "native_integrity_snapshot_numa_query_chunk_page_limit": (
            benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "native_gemm_output_numa_contract_supported": True,
        "native_gemm_output_numa_contract_schema": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
        ),
        "native_gemm_output_numa_query_chunk_page_limit": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "native_gemm_output_numa_evidence_capacity": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        ),
    }
    native = package / "gxeldcore.py"
    native.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            _LOG = Path(@LOG@)
            _BUILD = @BUILD@
            BEFORE = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {BEFORE[0]})
            with _LOG.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "native_import", "affinity_before": BEFORE, "affinity_after": sorted(os.sched_getaffinity(0))}) + "\\n")
            def build_info():
                return dict(_BUILD)
            def consume_gemm_telemetry():
                return []
            def consume_native_gemm_output_numa_evidence():
                return []
            def native_gemm_output_numa_evidence_status():
                return {}
            def reset_native_gemm_output_numa_evidence():
                return None
            """
        ).replace("@LOG@", repr(str(log))).replace("@BUILD@", repr(build)).lstrip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(install), "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SUMMIT_GXE_EXPECTED_INSTALL_PREFIX": str(install),
        "SUMMIT_GXE_EXPECTED_PACKAGE_MANIFEST_SHA256": (
            benchmark._installed_package_identity(install)["manifest_sha256"]
        ),
        "SUMMIT_GXE_EXPECTED_NATIVE_MODULE": str(native),
        "SUMMIT_GXE_EXPECTED_NATIVE_SHA256": benchmark._sha256(native),
        "SUMMIT_GXE_EXPECTED_SOURCE_COMMIT": "a" * 40,
        "SUMMIT_GXE_EXPECTED_SOURCE_TREE_SHA256": "b" * 64,
        "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_ARCHIVE_SHA256": "c" * 64,
        "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_SOURCE_COMMIT": "d" * 40,
        "SUMMIT_GXE_EXPECTED_PRIVATE_BLAS_SOURCE_TREE_SHA256": "e" * 64,
        "SUMMIT_GXE_EXPECTED_BLAS_THREADS": "2",
        "SUMMIT_GXE_REQUIRE_INTEGRITY": "1",
        "SUMMIT_GXE_REQUIRE_EARLY_NUMA": "1",
    })
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    command = [
        taskset, "-c", ",".join(map(str, selected)), sys.executable,
        "-S", "-c", benchmark._CLI_BOOTSTRAP, "--mock-cli",
    ]
    completed = subprocess.run(
        command, env=environment, text=True, capture_output=True,
        timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "early_numa", "cli_import", "native_import", "cli_main",
    ]
    assert records[1]["affinity"] == selected
    assert records[2]["affinity_before"] == selected
    assert records[2]["affinity_after"] == selected[:1]

    log.unlink()
    rejected_environment = dict(environment)
    rejected_environment["SUMMIT_GXE_EXPECTED_NATIVE_SHA256"] = "1" * 64
    rejected = subprocess.run(
        command, env=rejected_environment, text=True, capture_output=True,
        timeout=10, check=False,
    )
    assert rejected.returncode != 0
    rejected_records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [record["event"] for record in rejected_records] == [
        "early_numa", "cli_import", "native_import",
    ]


def test_dry_run_validates_caps_and_removes_temporary_subset(tmp_path):
    source = _synthetic_plink(tmp_path)
    environment, covariates = _tabular_inputs(tmp_path)
    install = tmp_path / "install"
    package = install / "summit"
    package.mkdir(parents=True)
    native = package / "gxeldcore.test.so"
    native.write_bytes(b"dry-run native placeholder")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "report.json"
    cpu = min(os.sched_getaffinity(0))

    assert benchmark.main(
        [
            "--geno-prefix", str(source),
            "--environment-file", str(environment),
            "--covariate-file", str(covariates),
            "--install-prefix", str(install),
            "--native-module", str(native),
            *_blis_cli_identity_args(tmp_path, install, native),
            "--python-executable", sys.executable,
            "--dependency-path", str(tmp_path),
            "--cpu-list", str(cpu), "--threads", "1",
            "--blocks", "4", "--block-width", "2", "--probes", "32",
            "--timeout-seconds", "30", "--temporary-parent", str(scratch),
            "--output", str(output), "--dry-run",
        ]
    ) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["status"] == "validated_dry_run"
    assert report["accepted"] is False
    assert report["acceptance_eligible"] is False
    assert report["acceptance_reason"] == (
        "dry_run_performs_no_scientific_execution"
    )
    assert report["mode"] == "single_layout"
    assert report["arguments"]["requested_storage_dtype"] == "float32"
    assert report["arguments"]["full_precision_layout"] == "current"
    assert report["command"][report["command"].index("--dtype") + 1] == "float32"
    assert report["command"][
        report["command"].index("--gxe-fp64-layout") + 1
    ] == "current"
    assert "--gxe-explicit-openmp-placement" in report["command"]
    assert report["command"][
        report["command"].index("--gxe-parallel-environment-groups") + 1
    ] == "1"
    assert report["child_environment"]["OMP_PROC_BIND"] == "SPREAD"
    assert report["child_environment"]["OMP_PLACES"] == f"{{{cpu}}}"
    assert report["child_environment"]["BLIS_NUM_THREADS"] == "1"
    assert report["expected_private_blis_provenance"][
        "private_blas_backend"
    ] == "upstream_blis"
    assert report["expected_private_blis_provenance"]["tls_required"] is True
    assert report["safety"]["private_openblas_acceptance_eligible"] is False
    assert report["inputs"]["subset_plink"]["variants"] == 8
    assert report["inputs"]["private_blas_archive"]["sha256"] == (
        report["expected_private_blis_provenance"][
            "private_blas_archive_sha256"
        ]
    )
    assert report["safety"]["b1024_is_projection_only"] is True
    assert report["cpu_placement"]["physical_cores_only"] is True
    assert any("<temporary_directory>" in token for token in report["command"])
    assert list(scratch.iterdir()) == []
    assert output.stat().st_mode & 0o777 == 0o600

    comparison_output = tmp_path / "comparison-report.json"
    with pytest.raises(SystemExit):
        benchmark.main(
            [
                "--geno-prefix", str(source),
                "--environment-file", str(environment),
                "--covariate-file", str(covariates),
                "--install-prefix", str(install),
                "--native-module", str(native),
                *_blis_cli_identity_args(tmp_path, install, native),
                "--python-executable", sys.executable,
                "--dependency-path", str(tmp_path),
                "--cpu-list", str(cpu), "--threads", "1",
                "--blocks", "4", "--block-width", "2", "--probes", "32",
                "--storage-dtype", "float64", "--compare-fp64-layouts",
                "--timeout-seconds", "30", "--temporary-parent", str(scratch),
                "--output", str(comparison_output), "--dry-run",
            ]
        )
    assert not comparison_output.exists()
    assert list(scratch.iterdir()) == []

    parser = benchmark._parser()
    for arguments in (
        ["--blocks", "3"],
        ["--blocks", "17"],
        ["--block-width", "6145"],
        ["--probes", "33"],
        ["--timeout-seconds", "1801"],
        ["--environment-columns", "a,b,c,d"],
        ["--compare-fp64-layouts", "--storage-dtype", "float64",
         "--timeout-seconds", "601"],
        ["--controller-timeout-seconds", "1201"],
        ["--compare-fp64-layouts"],
        ["--full-precision-layout", "source-tt-target-current"],
        ["--compare-fp64-layouts", "--storage-dtype", "float64",
         "--full-precision-layout", "source-tt-target-current"],
    ):
        with pytest.raises(SystemExit):
            args = parser.parse_args(
                [
                    "--install-prefix", str(install),
                    "--native-module", str(native),
                    "--output", str(tmp_path / "new-report.json"),
                    *arguments,
                ]
            )
            benchmark._validate_args(parser, args)

    with pytest.raises(SystemExit):
        args = parser.parse_args(
            [
                "--install-prefix", str(install),
                "--native-module", str(native),
                "--output", str(output),
            ]
        )
        benchmark._validate_args(parser, args)


def test_current_block_width_comparison_dry_run_is_fair_and_reports_terminal_block(
    tmp_path,
):
    source = _synthetic_plink(tmp_path, variants=12_000)
    environment, covariates = _tabular_inputs(tmp_path)
    install = tmp_path / "install"
    package = install / "summit"
    package.mkdir(parents=True)
    native = package / "gxeldcore.test.so"
    native.write_bytes(b"dry-run native placeholder")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "block-width-dry-run.json"
    cpu = min(os.sched_getaffinity(0))

    assert benchmark.main(
        [
            "--geno-prefix", str(source),
            "--environment-file", str(environment),
            "--covariate-file", str(covariates),
            "--install-prefix", str(install),
            "--native-module", str(native),
            *_blis_cli_identity_args(tmp_path, install, native),
            "--python-executable", sys.executable,
            "--dependency-path", str(tmp_path),
            "--cpu-list", str(cpu), "--threads", "1",
            "--subset-variants", "12000",
            "--compare-block-widths", "2000,3072",
            "--probes", "32", "--temporary-parent", str(scratch),
            "--output", str(output), "--dry-run",
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == (
        "validated_current_block_width_comparison_dry_run"
    )
    assert report["accepted"] is False
    assert report["acceptance_eligible"] is False
    assert report["mode"] == "current_block_width_comparison"
    assert report["arguments"]["subset_variants"] == 12_000
    assert report["arguments"]["compare_block_widths"] == [2000, 3072]
    assert report["arguments"]["timeout_seconds"] == 240
    assert report["arguments"]["controller_timeout_seconds"] == 540
    assert report["inputs"]["subset_plink"]["variants"] == 12_000
    assert report["block_width_schedules"]["2000"] == {
        "subset_variants": 12_000,
        "block_width": 2000,
        "block_count": 6,
        "full_width_block_count": 6,
        "terminal_block_width": 2000,
        "has_terminal_partial_block": False,
    }
    assert report["block_width_schedules"]["3072"]["block_count"] == 4
    assert report["block_width_schedules"]["3072"][
        "terminal_block_width"
    ] == 2784
    assert report["block_width_schedules"]["3072"][
        "has_terminal_partial_block"
    ] is True
    assert set(report["commands"]) == {"2000", "3072"}
    for width, command in report["commands"].items():
        assert command[command.index("--gxe-fp64-layout") + 1] == "current"
        assert command[command.index("--step_size") + 1] == width
        assert command[command.index("-S") + 1] == "-c"
    assert report["comparison_control_identity"]["controlled_differences"] == [
        "step_size", "output_prefix"
    ]
    assert report["safety"]["maximum_command_seconds"] == 240
    assert report["safety"][
        "block_width_comparison_maximum_controller_seconds"
    ] == 600
    assert list(scratch.iterdir()) == []


def test_current_block_width_comparison_argument_caps_are_fail_closed(tmp_path):
    parser = benchmark._parser()
    required = [
        "--install-prefix", str(tmp_path / "install"),
        "--native-module", str(tmp_path / "native.so"),
        "--output", str(tmp_path / "report.json"),
    ]
    invalid = (
        ["--subset-variants", "12000"],
        ["--compare-block-widths", "2000,3072"],
        ["--compare-block-widths", "2000,4096", "--subset-variants", "12000"],
        ["--compare-block-widths", "2000,3072", "--subset-variants", "6000"],
        [
            "--compare-block-widths", "2000,3072",
            "--subset-variants", "12000", "--timeout-seconds", "241",
        ],
        [
            "--compare-block-widths", "2000,3072",
            "--subset-variants", "12000",
            "--controller-timeout-seconds", "601",
        ],
    )
    for extra in invalid:
        with pytest.raises(SystemExit):
            args = parser.parse_args([*required, *extra])
            benchmark._validate_args(parser, args)


def test_numeric_comparisons_enforce_schema_order_and_report_all_error_metrics(tmp_path):
    reference = tmp_path / "reference.tsv.gz"
    candidate = tmp_path / "candidate.tsv.gz"
    rows = "CHR\tSNP\tBP\tL2_0\n1\trs1\t1\t2\n1\trs2\t2\t4\n"
    with gzip.open(reference, "wt", encoding="utf-8") as handle:
        handle.write(rows)
    with gzip.open(candidate, "wt", encoding="utf-8") as handle:
        handle.write(rows.replace("\t4\n", "\t4.00000000001\n"))

    observed = benchmark._compare_tsv_files(
        reference, candidate, key_columns=3, rtol=5e-12, atol=5e-12
    )
    assert observed["schema_and_row_order_equal"] is True
    assert observed["rows"] == 2
    assert observed["value_count"] == 2
    assert observed["relative_frobenius_error"] > 0.0
    assert observed["maximum_absolute_error"] > 0.0
    assert observed["maximum_normalized_error"] > 0.0
    assert observed["maximum_absolute_error_label"] == (
        "reference.tsv.gz:3:L2_0"
    )
    assert observed["maximum_normalized_error_label"] == (
        "reference.tsv.gz:3:L2_0"
    )
    assert observed["maximum_normalized_error_reference_value"] == 4.0
    assert observed["maximum_normalized_error_candidate_value"] == (
        4.00000000001
    )
    assert observed["maximum_normalized_error_signed_difference"] == (
        4.00000000001 - 4.0
    )
    assert observed["maximum_normalized_error_tolerance"] == (
        5e-12 + 5e-12 * 4.0
    )
    assert observed["within_declared_tolerance"] is True

    tree = benchmark._compare_json_tree(
        {"labels": ["a", "b"], "matrix": [[1.0, 2.0], [3.0, 4.0]]},
        {"labels": ["a", "b"], "matrix": [[1.0, 2.0], [3.0, 4.0]]},
        label="diagnostic", rtol=5e-12, atol=5e-12,
    )
    assert tree["schema_and_order_equal"] is True
    assert tree["value_count"] == 4
    assert tree["maximum_normalized_error"] == 0.0

    with pytest.raises(RuntimeError, match="row identity/order"):
        with gzip.open(candidate, "wt", encoding="utf-8") as handle:
            handle.write(
                "CHR\tSNP\tBP\tL2_0\n1\trs2\t2\t4\n1\trs1\t1\t2\n"
            )
        benchmark._compare_tsv_files(
            reference, candidate, key_columns=3, rtol=5e-12, atol=5e-12
        )


def _synthetic_reference_bundle(root: Path, *, delta: float) -> tuple[Path, dict]:
    root.mkdir()
    manifest = root / "reference.age.gxe.ref.json"
    files = {}
    hashes = {}
    for family in benchmark.SCORE_FAMILIES:
        path = root / f"reference.age.{family}.tsv.gz"
        value = 2.0 + (delta if family == "xw" else 0.0)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(f"CHR\tSNP\tBP\tL2_0\n1\trs1\t1\t{value:.17g}\n")
        files[family] = path.name
        hashes[family] = benchmark._sha256(path)
    diagonal = root / "reference.age.diag.tsv.gz"
    with gzip.open(diagonal, "wt", encoding="utf-8") as handle:
        handle.write(
            "CHR\tSNP\tBP\tA1\tA2\tNORM_X\tNORM_W\tBLOCK\n"
            "1\trs1\t1\tA\tC\t1\t1\t0\n"
        )
    files["diagonal"] = diagonal.name
    hashes["diagonal"] = benchmark._sha256(diagonal)
    payload = {
        "kind": "summit.gxe.reference", "schema_version": 3,
        "analysis_fingerprint": "b" * 64, "variant_digest": "c" * 64,
        "n_samples": 5, "fixed_effect_rank_excluding_intercept": 1,
        "residual_rank": 3, "environment": "age",
        "environment_transform": {"name": "population_centered_unit_norm"},
        "covariates": ["C"], "kernel_mode": "standardized",
        "feature_convention": "annotation_weighted_projected_genotype",
        "feature_convention_version": 1, "genotype_scale": "sample",
        "ld_scale": "cross_product_over_rank_squared", "null_corrected": False,
        "annotation_names": ["L2_0"], "annotation_masses": [8.0],
        "genotype_files": {
            ".bed": {"bytes": 19, "sha256": "d" * 64},
            ".bim": {"bytes": 24, "sha256": "e" * 64},
            ".fam": {"bytes": 20, "sha256": "f" * 64},
        },
        "resource_estimates": {
            "max_native_source_projection_leakage": 1.0e-16,
            "max_source_projection_leakage": 1.0e-16,
            "native_gemm_integrity_enabled": True,
            "native_repaired_gemm_output_columns": 0,
            "native_retried_gemm_input_mutations": 0,
        },
        "feature_diagnostics": {"valid_additive_columns": 8, "warning": False},
        "trace_nxe": 3.0, "trace_nxe_sq": 5.0,
        "population_trace": {
            "method": "independent_probe_u_statistic_v1",
            "sampling_axis": "individual", "num_vectors": 32,
            "feature_order": ["G:L2_0", "GxE:L2_0"],
            "same_individual_kernel_products": [[2.0, 1.0], [1.0, 4.0]],
            "jackknife_diagonal_method": "full_reference_reuse",
        },
        "jackknife": {
            "method": "block_local_ldscore_deletion", "num_blocks": 2,
            "block_labels": ["block:1", "block:2"],
            "assumption": "cross_block_directional_ld_is_negligible",
        },
        "files": files, "artifact_sha256": hashes,
    }
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return manifest, payload


def test_artifact_identities_cover_every_declared_disposable_output(tmp_path):
    reference_path, reference = _synthetic_reference_bundle(
        tmp_path / "bundle", delta=0.0
    )
    jackknife = reference_path.parent / "reference.age.gxe.jackknife.npz"
    jackknife.write_bytes(b"synthetic jackknife")
    reference["files"]["jackknife"] = jackknife.name
    reference["artifact_sha256"]["jackknife"] = benchmark._sha256(jackknife)
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    batch = tmp_path / "reference.gxe.multi.json"
    batch_payload = {
        "references": [
            {
                "environment": "age",
                "reference": str(reference_path.relative_to(batch.parent)),
                "sha256": benchmark._sha256(reference_path),
            }
        ]
    }
    batch.write_text(json.dumps(batch_payload) + "\n", encoding="utf-8")

    identities = benchmark._artifact_identities(batch, batch_payload)
    assert [(item["role"], item.get("family")) for item in identities] == [
        ("reference_manifest", None),
        ("score", "xx"), ("score", "xw"),
        ("score", "wx"), ("score", "ww"),
        ("diagonal", None), ("jackknife", None),
        ("batch_manifest", None),
    ]
    assert all(len(item["sha256"]) == 64 for item in identities)
    assert all(
        item.get("environment") == "age"
        for item in identities if item["role"] != "batch_manifest"
    )

    extra = reference_path.parent / "reference.age.extra.bin"
    extra.write_bytes(b"extra declared artifact")
    reference["files"]["extra"] = extra.name
    reference["artifact_sha256"]["extra"] = benchmark._sha256(extra)
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    batch_payload["references"][0]["sha256"] = benchmark._sha256(reference_path)
    batch.write_text(json.dumps(batch_payload) + "\n", encoding="utf-8")
    identities = benchmark._artifact_identities(batch, batch_payload)
    retained_extra = [item for item in identities if item.get("declared_role") == "extra"]
    assert len(retained_extra) == 1
    assert retained_extra[0]["role"] == "declared_artifact"

    del reference["artifact_sha256"]["extra"]
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    batch_payload["references"][0]["sha256"] = benchmark._sha256(reference_path)
    batch.write_text(json.dumps(batch_payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file/hash declarations differ"):
        benchmark._artifact_identities(batch, batch_payload)


def test_reference_bundle_comparison_covers_scores_population_and_jackknife(tmp_path):
    current_path, current = _synthetic_reference_bundle(
        tmp_path / "current", delta=0.0
    )
    optimized_path, optimized = _synthetic_reference_bundle(
        tmp_path / "optimized", delta=1.0e-11
    )
    observed = benchmark._compare_reference_bundle(
        current_path, current, optimized_path, optimized,
        rtol=5e-12, atol=5e-12,
    )
    assert list(observed["score_families"]) == ["xx", "xw", "wx", "ww"]
    assert observed["score_families"]["xw"]["maximum_absolute_error"] > 0.0
    assert observed["final_manifest_diagnostics"]["population_trace"][
        "schema_metadata_and_feature_order_equal"
    ] is True
    assert observed["final_manifest_diagnostics"]["population_trace"][
        "same_individual_kernel_products"
    ]["value_count"] == 4
    assert observed["final_manifest_diagnostics"]["jackknife"][
        "schema_and_order_equal"
    ] is True
    assert observed["diagonal_diagnostics"]["schema_and_row_order_equal"] is True
    assert observed["accuracy_gate_passed"] is True


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("genotype", "genotype_files SHA/size identity"),
        ("repair_counter", "native_repaired_gemm_output_columns"),
    ],
)
def test_reference_bundle_comparison_rejects_identity_and_integrity_differences(
    tmp_path, fault, message
):
    reference_path, reference = _synthetic_reference_bundle(
        tmp_path / "reference", delta=0.0
    )
    candidate_path, candidate = _synthetic_reference_bundle(
        tmp_path / "candidate", delta=0.0
    )
    if fault == "genotype":
        candidate["genotype_files"][".bed"]["sha256"] = "0" * 64
    else:
        candidate["resource_estimates"][
            "native_repaired_gemm_output_columns"
        ] = 1
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        benchmark._compare_reference_bundle(
            reference_path, reference, candidate_path, candidate,
            rtol=5e-12, atol=5e-12,
        )


def test_current_block_width_comparison_checks_all_outputs_and_diagnostics(tmp_path):
    baseline_path, baseline_reference = _synthetic_reference_bundle(
        tmp_path / "baseline", delta=0.0
    )
    candidate_path, candidate_reference = _synthetic_reference_bundle(
        tmp_path / "candidate", delta=1.0e-11
    )
    shared_randomization = {
        "distribution": "rademacher", "num_vectors": 32,
        "probe_offset": 0, "seed": 7,
    }
    baseline_reference["randomization"] = {
        **shared_randomization, "step_size": 2000,
    }
    candidate_reference["randomization"] = {
        **shared_randomization, "step_size": 3072,
    }
    baseline_path.write_text(json.dumps(baseline_reference) + "\n", encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate_reference) + "\n", encoding="utf-8")

    def batch(reference_path: Path) -> tuple[Path, dict]:
        path = reference_path.parent / "reference.gxe.multi.json"
        payload = {
            "kind": "summit.gxe.multi_environment_reference_batch",
            "schema_version": 1,
            "execution": "shared_in_memory_decoded_blocks",
            "requested_backend": "direct",
            "protected_native_gemm": True,
            "arithmetic_dtype": "float64",
            "requested_storage_dtype": "float64",
            "gemm_backend": "gxeldcore_direct",
            "gemm_backend_build_sha256": "a" * 64,
            "native_source_commit": "b" * 40,
            "native_source_tree_sha256": "c" * 64,
            "native_gemm_integrity_enabled": True,
            "native_blas_runtime_isolation": "private_static",
            "num_environments": 1,
            "common_complete_case_samples": 5,
            "num_variants": 8,
            "randomization": dict(shared_randomization),
            "shared_genotype_passes": 2,
            "environment_tiles": [[0, 1]],
            "full_precision_layout": "current",
            "references": [
                {
                    "environment": "age",
                    "reference": reference_path.name,
                    "sha256": benchmark._sha256(reference_path),
                }
            ],
            "performance_telemetry": {
                "phase_totals": {
                    "source_gemm": {"wall_seconds": 1.0},
                    "target_gemm": {"wall_seconds": 2.0},
                }
            },
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path, payload

    baseline_batch_path, baseline_batch = batch(baseline_path)
    candidate_batch_path, candidate_batch = batch(candidate_path)
    observed = benchmark._compare_completed_block_width_runs(
        baseline_batch_path, baseline_batch,
        candidate_batch_path, candidate_batch,
        baseline_width=2000, candidate_width=3072,
        environment_order=["age"], rtol=5e-12, atol=5e-12,
    )

    assert observed["layout"] == "current"
    assert observed["controlled_batch_identity_equal"] is True
    assert observed[
        "declared_outputs_and_allowlisted_final_diagnostics_compared"
    ] is True
    assert observed["environments"][0][
        "declared_artifact_roles_equal_in_order"
    ] is True
    assert observed["environments"][0][
        "reference_manifest_key_schema_equal_in_order"
    ] is True
    assert observed["environments"][0][
        "randomization_equal_except_step_size"
    ] is True
    assert observed["environments"][0][
        "genotype_files_sha_size_identity_equal"
    ] is True
    resource_diagnostics = observed["environments"][0][
        "resource_estimate_diagnostics"
    ]
    assert resource_diagnostics["integrity_counters_exactly_equal"] is True
    assert resource_diagnostics["leakage"]["within_declared_tolerance"] is True
    assert observed["accuracy_gate_passed"] is True


def test_available_jackknife_npz_arrays_preserve_key_schema_shape_and_label_order(tmp_path):
    np = pytest.importorskip("numpy")
    current = tmp_path / "current.npz"
    optimized = tmp_path / "optimized.npz"
    arrays = {
        "block_labels": np.asarray(["block:1", "block:2"]),
        "within_xx": np.arange(8, dtype=np.float64).reshape(2, 2, 2),
        "within_xw": np.ones((2, 2, 2), dtype=np.float64),
        "within_wx": np.full((2, 2, 2), 2.0, dtype=np.float64),
        "within_ww": np.full((2, 2, 2), 3.0, dtype=np.float64),
    }
    np.savez(current, **arrays)
    np.savez(optimized, **arrays)
    observed = benchmark._compare_npz_files(
        current, optimized, rtol=5e-12, atol=5e-12
    )
    assert observed["key_schema_and_order_equal"] is True
    assert observed["arrays"]["block_labels"]["labels_equal_in_order"] is True
    assert observed["arrays"]["within_xx"]["maximum_normalized_error"] == 0.0


def test_run_rejects_dual_layout_even_when_called_without_parser_validation(
    tmp_path, monkeypatch
):
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--install-prefix", str(tmp_path / "unused-install"),
            "--native-module", str(tmp_path / "unused-native.so"),
            *_placeholder_blis_cli_args(tmp_path),
            "--output", str(tmp_path / "report.json"),
            "--compare-fp64-layouts",
        ]
    )
    execute_called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal execute_called
        execute_called = True
        raise AssertionError("disabled layout child must not execute")

    monkeypatch.setattr(benchmark, "_execute_layout", fail_if_called)
    with pytest.raises(RuntimeError, match="disabled at the execution boundary"):
        benchmark.run(args)
    assert execute_called is False
    assert not args.output.exists()


def test_current_block_width_controller_runs_fresh_current_children_and_projects(
    tmp_path, monkeypatch
):
    source = _synthetic_plink(tmp_path, variants=12_000)
    environment, covariates = _tabular_inputs(tmp_path)
    install = tmp_path / "install"
    package = install / "summit"
    package.mkdir(parents=True)
    native = package / "gxeldcore.test.so"
    native.write_bytes(b"mock native")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cpu = min(os.sched_getaffinity(0))
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--geno-prefix", str(source),
            "--environment-file", str(environment),
            "--covariate-file", str(covariates),
            "--install-prefix", str(install), "--native-module", str(native),
            *_blis_cli_identity_args(tmp_path, install, native),
            "--python-executable", sys.executable,
            "--dependency-path", str(tmp_path),
            "--cpu-list", str(cpu), "--threads", "1",
            "--subset-variants", "12000",
            "--compare-block-widths", "2000,3072",
            "--probes", "32", "--temporary-parent", str(scratch),
            "--output", str(tmp_path / "report.json"),
        ]
    )
    benchmark._validate_args(parser, args)
    observed_children = []

    def fake_execute(
        _args, *, layout, block_width, output_prefix, timeout_seconds, **_kwargs
    ):
        observed_children.append((layout, block_width, timeout_seconds))
        projected = 10.0 if block_width == 2000 else 8.0
        return {
            "layout": "current",
            "block_schedule": benchmark._block_schedule(12_000, block_width),
            "command": [],
            "process": {
                "returncode": 0, "wall_seconds": projected,
                "controller_children_ru_maxrss_gib_high_water": 1.0,
                "controller_children_ru_maxrss_scope": "cumulative",
                "authoritative_per_layout_peak_rss_field": (
                    "batch_summary.peak_process_rss_gib_at_manifest"
                ),
                "stdout": {}, "stderr": {},
            },
            "manifest_path": Path(f"{output_prefix}.gxe.multi.json"),
            "payload": {
                "full_precision_layout": "current",
                "randomization": {"num_vectors": 32, "seed": args.seed},
                "peak_process_rss_gib_at_manifest": 1.0,
            },
            "performance": {
                "full_precision_layout": "current", "phase_totals": {},
            },
            "projections": {
                "full_m_b32": {"projected_seconds": projected},
                "full_m_b1024_same_tile_schedule": {
                    "projected_seconds": projected * 10.0
                },
            },
            "gemm_summaries": [],
            "temporary_artifact_identities": [
                {"role": "batch_manifest", "sha256": str(block_width) * 16}
            ],
        }

    def fake_compare(*_args, **kwargs):
        assert observed_children == [
            ("current", 2000, 240),
            ("current", 3072, 240),
        ]
        assert kwargs["baseline_width"] == 2000
        assert kwargs["candidate_width"] == 3072
        return {
            "layout": "current",
            "baseline_block_width": 2000,
            "candidate_block_width": 3072,
            "accuracy_gate_passed": True,
        }

    monkeypatch.setattr(benchmark, "_execute_layout", fake_execute)
    monkeypatch.setattr(
        benchmark, "_compare_completed_block_width_runs", fake_compare
    )
    report = benchmark.run(args)

    assert report["status"] == (
        "completed_current_block_width_comparison_candidate_ineligible"
    )
    assert report["comparison_completed"] is True
    assert report["accepted"] is False
    assert report["accuracy_gate_passed"] is True
    assert report["candidate_eligible"] is False
    assert report["candidate_selected"] is False
    selection = report["candidate_selection"]
    assert selection["material_speedup_threshold"] == 1.05
    assert selection["material_speedup_gate_passed"] is True
    assert selection["memory_rss_bound_gate_passed"] is True
    assert selection["memory_pressure_evidence_available"] is False
    assert selection["major_fault_evidence_available"] is False
    assert selection["ineligibility_reasons"] == [
        "memory_pressure_evidence_unavailable",
        "major_fault_evidence_unavailable",
    ]
    assert report["process"]["fresh_exec_children"] == 2
    assert report["process"]["children_sequential"] is True
    assert set(report["block_width_runs"]) == {"2000", "3072"}
    assert all(
        run["layout"] == "current"
        for run in report["block_width_runs"].values()
    )
    assert report["block_width_runs"]["3072"]["block_schedule"][
        "terminal_block_width"
    ] == 2784
    assert report["projections"]["speedups"]["full_m_b32"][
        "baseline_over_candidate_speedup"
    ] == 1.25
    assert report["block_width_runs"]["2000"][
        "temporary_artifact_identities"
    ][0]["role"] == "batch_manifest"
    assert list(scratch.iterdir()) == []


def test_rejected_accuracy_report_is_published_no_replace_and_main_returns_nonzero(
    tmp_path, monkeypatch
):
    output = tmp_path / "rejected.json"
    payload = {
        "schema": benchmark.SCHEMA_NAME,
        "schema_version": benchmark.SCHEMA_VERSION,
        "status": "rejected_dual_layout_accuracy_gate",
        "accepted": False,
        "accuracy_gate_passed": False,
    }
    monkeypatch.setattr(benchmark, "_validate_args", lambda *_: None)
    monkeypatch.setattr(benchmark, "run", lambda _args: dict(payload))
    exit_code = benchmark.main(
        [
            "--install-prefix", str(tmp_path / "unused-install"),
            "--native-module", str(tmp_path / "unused.so"),
            *_placeholder_blis_cli_args(tmp_path),
            "--output", str(output),
        ]
    )
    assert exit_code != 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError, match="refusing existing report"):
        benchmark._atomic_json_no_replace(payload, output)


def test_executed_main_requires_explicit_true_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_validate_args", lambda *_: None)
    accepted_output = tmp_path / "accepted.json"
    monkeypatch.setattr(
        benchmark, "run", lambda _args: {"status": "completed", "accepted": True}
    )
    assert benchmark.main(
        [
            "--install-prefix", str(tmp_path / "unused-install"),
            "--native-module", str(tmp_path / "unused.so"),
            *_placeholder_blis_cli_args(tmp_path),
            "--output", str(accepted_output),
        ]
    ) == 0

    ambiguous_output = tmp_path / "ambiguous.json"
    monkeypatch.setattr(
        benchmark, "run", lambda _args: {"status": "completed"}
    )
    assert benchmark.main(
        [
            "--install-prefix", str(tmp_path / "unused-install"),
            "--native-module", str(tmp_path / "unused.so"),
            *_placeholder_blis_cli_args(tmp_path),
            "--output", str(ambiguous_output),
        ]
    ) == 2


def _queried_numa_samples(record: dict, node: int = 0) -> dict:
    page_size = 4096
    sample_limit = 8
    operands = {}
    for operand_name in ("a", "b", "c"):
        byte_count = benchmark._gemm_operand_byte_count(record, operand_name)
        start_offset = 64
        first_full_offset = page_size - start_offset
        full_pages = (byte_count - first_full_offset) // page_size
        selected = min(full_pages, sample_limit)
        ordered_samples = []
        for ordinal in range(selected):
            full_page_index = (
                0 if selected == 1
                else ordinal * (full_pages - 1) // (selected - 1)
            )
            ordered_samples.append(
                {
                    "sample_ordinal": ordinal,
                    "full_page_index": full_page_index,
                    "byte_offset_from_operand_start": (
                        first_full_offset + full_page_index * page_size
                    ),
                    "status_kind": "numa_node",
                    "raw_move_pages_status": node,
                    "numa_node": node,
                    "page_query_errno": None,
                }
            )
        operands[operand_name] = {
            "query_status": "queried",
            "storage_span_pages": (
                (start_offset + byte_count - 1) // page_size + 1
            ),
            "fully_contained_pages": full_pages,
            "operand_byte_count": byte_count,
            "operand_start_address_page_offset": start_offset,
            "operand_end_exclusive_address_page_offset": (
                start_offset + byte_count
            ) % page_size,
            "selected_sample_pages": selected,
            "resolved_sample_pages": selected,
            "page_query_error_pages": 0,
            "node_histogram": {str(node): selected},
            "page_error_errno_histogram": {},
            "ordered_samples": ordered_samples,
        }
    return {
        "schema_version": 1,
        "sampling_method": "move_pages_query_no_migration",
        "sampling_timing": "after_vendor_call_outside_timed_interval",
        "address_selection_schema_version": 1,
        "address_selection_policy": benchmark.NUMA_ADDRESS_SELECTION_POLICY,
        "selected_addresses_are_page_bases": True,
        "partial_boundary_pages_included": False,
        "first_and_last_fully_contained_pages_selected": True,
        "virtual_addresses_exposed": False,
        "operand_byte_range_semantics": "[start_address,end_exclusive_address)",
        "address_evidence": (
            "ordered_samples_with_operand_relative_byte_offsets_and_"
            "full_page_indices"
        ),
        "sample_limit_per_operand": sample_limit, "system_page_size": page_size,
        "syscall_result": 0, "syscall_errno": 0,
        "operands": operands,
    }


def _native_integrity_snapshot_numa(
    record: dict, *, node: int = 0, minimum_vendor_flops: int = 1
) -> dict:
    eligible = bool(
        record.get("phase") == "source_gemm"
        and record.get("operation") == "dgemm_nn"
        and 2 * record["m"] * record["n"] * record["k"]
        >= minimum_vendor_flops
    )
    evidence = {
        "schema": benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA,
        "schema_version": 1,
        "contract_required": True,
        "operand_role": benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_ROLE,
        "integrity_check_shape_eligible": eligible,
        "integrity_check_executed": eligible,
        "snapshot_available": eligible,
        "complete": eligible,
    }
    if not eligible:
        return evidence
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    logical_bytes = record["k"] * record["n"] * 8
    mapping_bytes = (
        (logical_bytes + page_size - 1) // page_size
    ) * page_size
    page_count = mapping_bytes // page_size
    chunk_limit = benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
    encoded_status = int(node).to_bytes(4, "little", signed=True)
    evidence.update(
        {
            "logical_byte_count": logical_bytes,
            "mapping_bytes": mapping_bytes,
            "page_size": page_size,
            "mapping_page_count": page_count,
            "selected_nodes": [node],
            "policy_mode": "bind_static_nodes",
            "policy_mode_value": (
                benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_POLICY_VALUE
            ),
            "anonymous_private_mapping": True,
            "page_aligned_mapping": True,
            "bound_before_first_touch": True,
            "live_owner_policy_verified": True,
            "pre_touch_range_policy_verified": True,
            "sealed_read_only_before_vendor": True,
            "pre_vendor_complete_page_query": True,
            "queried_pages": page_count,
            "resolved_pages": page_count,
            "query_chunks": (page_count + chunk_limit - 1) // chunk_limit,
            "query_chunk_page_limit": chunk_limit,
            "node_histogram": {str(node): page_count},
            "ordered_status_sha256": hashlib.sha256(
                encoded_status * page_count
            ).hexdigest(),
            "ordered_status_encoding": "signed_int32_little_endian",
            "pre_vendor_strict_policy_verified": True,
            "strict_policy_check": "MPOL_MF_STRICT_without_MPOL_MF_MOVE",
            "page_query_method": "move_pages_query_no_migration",
            "page_migration_requested": False,
            "placement_repair_performed": False,
        }
    )
    return evidence


def _native_gemm_output_numa(
    record: dict, *, call_id: int, node: int = 0
) -> dict:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    logical_bytes = record["m"] * record["n"] * 8
    mapping_bytes = (
        (logical_bytes + page_size - 1) // page_size
    ) * page_size
    page_count = mapping_bytes // page_size
    chunk_limit = benchmark.NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
    encoded_status = int(node).to_bytes(4, "little", signed=True)
    return {
        "schema": benchmark.NATIVE_GEMM_OUTPUT_NUMA_SCHEMA,
        "schema_version": 1,
        "applicable": True,
        "call_id": call_id,
        "contract_required": True,
        "complete": True,
        "operand_role": benchmark.NATIVE_GEMM_OUTPUT_NUMA_ROLE,
        "logical_rows": record["m"],
        "logical_columns": record["n"],
        "logical_byte_count": logical_bytes,
        "storage_layout": record["layout"],
        "mapping_bytes": mapping_bytes,
        "page_size": page_size,
        "mapping_page_count": page_count,
        "selected_nodes": [node],
        "policy_mode": "bind_static_nodes",
        "policy_mode_value": benchmark.NATIVE_GEMM_OUTPUT_NUMA_POLICY_VALUE,
        "allocation_mode": "mmap_private_anonymous",
        "anonymous_private_mapping": True,
        "page_aligned_mapping": True,
        "writable_output": True,
        "bound_before_first_touch": True,
        "pre_touch_live_owner_policy_verified": True,
        "pre_touch_range_policy_verified": True,
        "post_repair_live_owner_policy_verified": True,
        "post_repair_range_policy_verified": True,
        "post_repair_complete_page_query": True,
        "queried_pages": page_count,
        "resolved_pages": page_count,
        "query_chunks": (page_count + chunk_limit - 1) // chunk_limit,
        "query_chunk_page_limit": chunk_limit,
        "node_histogram": {str(node): page_count},
        "ordered_status_sha256": hashlib.sha256(
            encoded_status * page_count
        ).hexdigest(),
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


def test_numa_summary_preserves_full_call_operand_and_future_ordered_context():
    payload, _args = _completed_payload("current")
    record = payload["performance_telemetry"]["gemm_records"][0]
    record["sequence"] = 17
    summary = benchmark._hot_vendor_numa_summary(
        payload["performance_telemetry"]["gemm_records"], {0}
    )
    observed = summary["records"][0]
    assert observed["sequence"] == 17
    assert observed["phase"] == "source_gemm"
    assert observed["operation"] == "dgemm_nn"
    assert [observed[name] for name in ("m", "n", "k")] == [1024, 128, 512]
    assert [observed[name] for name in ("lda", "ldb", "ldc")] == [
        1024, 512, 1024
    ]
    assert observed["selected_numa_nodes"] == [0]
    assert observed["operands"]["a"]["operand"] == "a"
    assert observed["operands"]["a"]["histogram"] == {"0": 8}
    assert len(observed["operands"]["a"]["ordered_samples"]) == 8

    record["operand_numa_page_samples"]["operands"]["b"][
        "node_histogram"
    ] = {"1": 8}
    for sample in record["operand_numa_page_samples"]["operands"]["b"][
        "ordered_samples"
    ]:
        sample["raw_move_pages_status"] = 1
        sample["numa_node"] = 1
    with pytest.raises(RuntimeError, match="outside selected NUMA nodes") as caught:
        benchmark._hot_vendor_numa_summary(
            payload["performance_telemetry"]["gemm_records"], {0}
        )
    message = str(caught.value)
    for expected in (
        '"sequence":17', '"phase":"source_gemm"',
        '"operation":"dgemm_nn"', '"operand":"b"',
        '"m":1024', '"n":128', '"k":512',
        '"lda":1024', '"ldb":512', '"ldc":1024',
        '"selected_numa_nodes":[0]', '"histogram":{"1":8}',
    ):
        assert expected in message


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("byte_count", "literal GEMM storage span"),
        ("ordered_offset", "exact page-selection schedule"),
        ("ordered_node", "do not match ordered evidence"),
        ("boolean_counter", "is malformed"),
        ("noncanonical_histogram", "histogram is malformed"),
    ],
)
def test_numa_summary_rejects_forged_span_order_and_json_types(mutation, message):
    payload, _args = _completed_payload("current")
    record = payload["performance_telemetry"]["gemm_records"][0]
    operand = record["operand_numa_page_samples"]["operands"]["a"]
    if mutation == "byte_count":
        operand["operand_byte_count"] += 8
    elif mutation == "ordered_offset":
        operand["ordered_samples"][0]["byte_offset_from_operand_start"] += 4096
    elif mutation == "ordered_node":
        operand["ordered_samples"][0]["raw_move_pages_status"] = 1
        operand["ordered_samples"][0]["numa_node"] = 1
    elif mutation == "boolean_counter":
        operand["selected_sample_pages"] = True
    elif mutation == "noncanonical_histogram":
        operand["node_histogram"] = {"00": 8}
    else:  # pragma: no cover - the parameter list is closed above.
        raise AssertionError(mutation)
    with pytest.raises(RuntimeError, match=message) as caught:
        benchmark._hot_vendor_numa_summary(
            payload["performance_telemetry"]["gemm_records"], {0}
        )
    assert '"operand":"a"' in str(caught.value)
    assert '"phase":"source_gemm"' in str(caught.value)


def _early_numa_attestation(node: int = 0) -> dict:
    return {
        "schema": "summit.numa_policy_attestation.v1",
        "mode": "membind",
        "requested_nodes": str(node),
        "effective_nodes": [node],
        "task_count_at_application": 1,
        "applied_before_numeric_import": True,
        "verified": True,
        "source": "libnuma",
        "applied_policy": f"libnuma:membind:{node}",
        "static_nodes": True,
        "pid": 12345,
    }


def _openmp_placement(cpus: list[int]) -> dict:
    return {
        "schema": benchmark.OPENMP_PLACEMENT_SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": len(cpus),
        "expected_cpu_ids": list(cpus),
        "omp_dynamic": False,
        "omp_thread_limit": len(cpus),
        "omp_max_active_levels": 1,
        "omp_proc_bind": "spread",
        "omp_binding_active": True,
        "omp_num_places": len(cpus),
        "effective_openmp_capacity": len(cpus),
        "place_cpu_ids": [[cpu] for cpu in cpus],
        "team_size": len(cpus),
        "exact_singleton_places": True,
        "exact_team_coverage": True,
        "workers": [
            {
                "thread_num": index,
                "place_num": index,
                "place_cpu_ids": [cpu],
                "sched_affinity_cpu_ids": [cpu],
                "current_cpu": cpu,
                "verified": True,
            }
            for index, cpu in enumerate(cpus)
        ],
        "vendor_calls": 0,
    }


def _blis_compile_options(args: SimpleNamespace, placement: dict) -> dict:
    return {
        "api_version": 9,
        "blas_vendor": "BLIS",
        "blas_runtime_config": "BLIS 2.0 config=zen",
        "blas_runtime_corename": "zen",
        "blas_runtime_isolation": "private_static",
        "blas_runtime_threads": args.threads,
        "blas_runtime_threading_layer": "pthreads",
        "blas_runtime_worker_affinity_policy": (
            "inherit_authenticated_selected_cpu_set_per_call"
        ),
        "blas_runtime_thread_strategy": "automatic",
        "blas_runtime_thread_ways": {
            name: 1 for name in benchmark.BLIS_WAY_NAMES
        },
        "blas_runtime_owner_thread_enforced": True,
        "blas_runtime_owner_thread_configured": True,
        "blas_runtime_environment_immutable": True,
        "blas_runtime_environment_contract": "blis_process_start_v1",
        "blas_runtime_tls_enabled": True,
        "gemm_execution_mode": "serialized_fixed_private_blis",
        "private_openblas_archive_sha256": "none",
        "private_blas_backend": "upstream_blis",
        "private_blas_archive_sha256": args.expected_archive_sha256,
        "private_blas_source_commit": args.expected_private_source_commit,
        "private_blas_source_tree_sha256": (
            args.expected_private_source_tree_sha256
        ),
        "private_blas_config_family": "zen",
        "private_blas_header_sha256": "e" * 64,
        "private_blas_cblas_header_sha256": "f" * 64,
        "gemm_integrity_enabled": True,
        "gemm_integrity_minimum_vendor_flops": 1_000_000_000,
        "native_integrity_snapshot_numa_contract_supported": True,
        "native_integrity_snapshot_numa_contract_schema": (
            benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_SCHEMA
        ),
        "native_integrity_snapshot_numa_query_chunk_page_limit": (
            benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "native_gemm_output_numa_contract_supported": True,
        "native_gemm_output_numa_contract_schema": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_SCHEMA
        ),
        "native_gemm_output_numa_query_chunk_page_limit": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "native_gemm_output_numa_evidence_capacity": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
        ),
        "gemm_vendor_entry_outer_openmp_guard": True,
        "openmp_effective_capacity_policy": (
            "bound_places_else_sched_affinity_v1"
        ),
        "openmp_placement_contract_supported": True,
        "openmp_placement_contract_schema": benchmark.OPENMP_PLACEMENT_SCHEMA,
        "openmp_placement_contract_configured": True,
        "openmp_placement_contract_immutable": True,
        "openmp_placement_probe_vendor_calls": 0,
        "openmp_placement_contract_evidence": placement,
        "loaded_blas_runtime": {
            "internal_api": "blis",
            "version": "2.0",
            "path": "private-static gxeldcore image",
            "num_threads": args.threads,
            "isolation": "private_static",
            "process_blas_runtimes": 1,
            "threading_layer": "openmp",
        },
    }


def test_api9_openmp_placement_and_blis_tls_contracts_are_exact() -> None:
    cpus = [2, 4]
    records = [
        benchmark.CpuRecord(cpu=cpu, core=index, socket=0, node=0)
        for index, cpu in enumerate(cpus)
    ]
    placement = _openmp_placement(cpus)
    assert benchmark._validate_openmp_placement_attestation(
        placement, cpu_records=records, threads=2
    ) == placement
    args = SimpleNamespace(
        threads=2,
        expected_archive_sha256="a" * 64,
        expected_private_source_commit="b" * 40,
        expected_private_source_tree_sha256="c" * 64,
        allow_integrity_disabled=False,
    )
    options = _blis_compile_options(args, placement)
    assert benchmark._validate_blis_compile_options(
        options,
        args=args,
        placement=placement,
    )["blas_runtime_tls_enabled"] is True

    wrong_worker = json.loads(json.dumps(placement))
    wrong_worker["workers"].reverse()
    with pytest.raises(RuntimeError, match="worker 0"):
        benchmark._validate_openmp_placement_attestation(
            wrong_worker, cpu_records=records, threads=2
        )
    for field, value, message in (
        ("blas_runtime_tls_enabled", False, "compile contract"),
        ("blas_runtime_threading_layer", "openmp", "compile contract"),
        ("private_blas_backend", "openblas", "compile contract"),
        ("openmp_placement_contract_configured", False, "compile contract"),
        (
            "native_integrity_snapshot_numa_contract_supported",
            False,
            "compile contract",
        ),
        (
            "native_integrity_snapshot_numa_contract_schema",
            "summit.native_integrity_snapshot_numa.v0",
            "compile contract",
        ),
        (
            "native_integrity_snapshot_numa_query_chunk_page_limit",
            True,
            "compile contract",
        ),
        (
            "native_gemm_output_numa_contract_supported",
            False,
            "compile contract",
        ),
        (
            "native_gemm_output_numa_contract_schema",
            "summit.native_gemm_output_numa.v0",
            "compile contract",
        ),
        (
            "native_gemm_output_numa_query_chunk_page_limit",
            True,
            "compile contract",
        ),
        (
            "native_gemm_output_numa_evidence_capacity",
            True,
            "compile contract",
        ),
        (
            "gemm_integrity_minimum_vendor_flops",
            0,
            "noncanonical JSON types",
        ),
    ):
        rejected = dict(options)
        rejected[field] = value
        with pytest.raises(RuntimeError, match=message):
            benchmark._validate_blis_compile_options(
                rejected,
                args=args,
                placement=placement,
            )


def test_early_numa_attestation_gate_is_mandatory_exact_and_single_task():
    attestation = _early_numa_attestation()
    observed = benchmark._early_numa_attestation_status(
        {"early_numa_attestation": dict(attestation)},
        {"early_numa_attestation": dict(attestation)},
        {0}, expected_child_pid=12345,
    )
    assert observed["complete"] is True
    assert observed["effective_nodes"] == [0]
    assert observed["task_count_at_application"] == 1

    with pytest.raises(RuntimeError, match="lacks the required structured"):
        benchmark._early_numa_attestation_status(
            {}, {"early_numa_attestation": dict(attestation)}, {0},
            expected_child_pid=12345,
        )
    forged = dict(attestation, task_count_at_application=33)
    with pytest.raises(RuntimeError, match="single-task child"):
        benchmark._early_numa_attestation_status(
            {"early_numa_attestation": forged},
            {"early_numa_attestation": forged},
            {0}, expected_child_pid=12345,
        )
    forged = dict(attestation, verified=1)
    with pytest.raises(RuntimeError, match="not verified before numeric import"):
        benchmark._early_numa_attestation_status(
            {"early_numa_attestation": forged},
            {"early_numa_attestation": forged},
            {0}, expected_child_pid=12345,
        )
    for static_nodes in (False, 1, None):
        forged = dict(attestation, static_nodes=static_nodes)
        with pytest.raises(
            RuntimeError, match="not verified before numeric import"
        ):
            benchmark._early_numa_attestation_status(
                {"early_numa_attestation": forged},
                {"early_numa_attestation": forged},
                {0}, expected_child_pid=12345,
            )
    forged = dict(attestation, pid=54321)
    with pytest.raises(RuntimeError, match="expected single-task child"):
        benchmark._early_numa_attestation_status(
            {"early_numa_attestation": forged},
            {"early_numa_attestation": forged},
            {0}, expected_child_pid=12345,
        )
    for field, value in (
        ("effective_nodes", "0"),
        ("task_count_at_application", True),
        ("pid", "12345"),
    ):
        forged = dict(attestation, **{field: value})
        with pytest.raises(RuntimeError, match="malformed"):
            benchmark._early_numa_attestation_status(
                {"early_numa_attestation": forged},
                {"early_numa_attestation": forged},
                {0}, expected_child_pid=12345,
            )


def _current_terminal_vendor_records() -> list[dict]:
    records = []
    environment_names = ["age", "bmi"]
    for environment_start, environment_stop in ((0, 1), (1, 2)):
        for probe_start, probe_count in ((0, 8), (8, 8)):
            semantic_p = 2 * (environment_stop - environment_start) * 2 * probe_count
            for genotype_index, genotype_start, genotype_stop in (
                (0, 0, 8), (1, 8, 13)
            ):
                genotype_width = genotype_stop - genotype_start
                common = {
                    "telemetry_scope": "vendor_call",
                    "completed": True,
                    "wall_seconds": 0.25,
                    "process_cpu_seconds": 0.75,
                    "achieved_gflops": 12.0,
                    "gflops_per_second": 12.0,
                    "active_core_equivalents": 3.0,
                    "configured_threads": 4,
                    "requested_threads": 4,
                    "requested_blas_threads": 4,
                    "backend_threads": 4,
                    "environment_tile": [environment_start, environment_stop],
                    "environment_names": environment_names[
                        environment_start:environment_stop
                    ],
                    "probe_tile": [probe_start, probe_count],
                    "genotype_block": [genotype_start, genotype_stop],
                    "genotype_block_index": genotype_index,
                    "genotype_block_width": genotype_width,
                }
                records.extend(
                    [
                        {
                            **common,
                            "phase": "source_gemm",
                            "operation": "dgemm_nn", "layout": "column_major",
                            "transpose_a": "N", "transpose_b": "N",
                            "m": 7, "n": semantic_p, "k": genotype_width,
                            "lda": 7, "ldb": genotype_width, "ldc": 7,
                        },
                        {
                            **common,
                            "phase": "target_gemm",
                            "operation": "dgemm_tn", "layout": "column_major",
                            "transpose_a": "T", "transpose_b": "N",
                            "m": genotype_width, "n": 2 * semantic_p, "k": 7,
                            "lda": 7, "ldb": 7, "ldc": genotype_width,
                        },
                    ]
                )
    return records


def test_current_vendor_acceptance_passes_exact_terminal_semantics_and_summarizes():
    records = _current_terminal_vendor_records()
    benchmark._validate_layout_records(records, "current")
    observed = benchmark._validate_current_vendor_acceptance(
        records, samples=7, variants=13, block_width=8,
        probes=16, annotation_bins=2, threads=4,
        environment_count=2, environment_tiles=[[0, 1], [1, 2]],
        probe_tiles=[[0, 8], [8, 8]],
    )
    assert observed["gate_passed"] is True
    assert observed["source_record_count"] == 8
    assert observed["target_record_count"] == 8
    assert observed["minimum_active_core_equivalents_required"] == 3.0
    assert observed["minimum_active_core_equivalents"] == 3.0
    assert observed["terminal_block_context_count"] == 4
    assert observed["expected_cartesian_context_count"] == 8
    assert observed["complete_cartesian_context_gate_passed"] is True
    assert observed["semantic_p_values"] == [32]
    assert observed["genotype_block_widths"] == [5, 8]
    assert observed["source_semantic_dimensions"] == [[7, 32, 5], [7, 32, 8]]
    assert observed["target_semantic_dimensions"] == [[5, 64, 7], [8, 64, 7]]


@pytest.mark.parametrize(
    ("phase", "field", "value", "message"),
    [
        ("source_gemm", "completed", False, "not completed"),
        ("target_gemm", "wall_seconds", 0.0, "finite and positive"),
        ("source_gemm", "process_cpu_seconds", float("nan"), "finite and positive"),
        ("target_gemm", "achieved_gflops", 0.0, "finite and positive"),
        ("source_gemm", "gflops_per_second", float("inf"), "finite and positive"),
        ("target_gemm", "configured_threads", 3, "expected 4"),
        ("source_gemm", "requested_threads", 3, "expected 4"),
        ("target_gemm", "requested_blas_threads", 3, "expected 4"),
        ("source_gemm", "backend_threads", 3, "expected 4"),
        ("target_gemm", "active_core_equivalents", 2.99, "75% threshold"),
        ("source_gemm", "m", 8, "semantic dimensions"),
        ("target_gemm", "n", 255, "semantic dimensions"),
    ],
)
def test_current_vendor_acceptance_rejects_incomplete_metrics_threads_and_shapes(
    phase, field, value, message
):
    records = _current_terminal_vendor_records()
    next(record for record in records if record["phase"] == phase)[field] = value
    with pytest.raises(RuntimeError, match=message):
        benchmark._validate_current_vendor_acceptance(
            records, samples=7, variants=13, block_width=8,
            probes=16, annotation_bins=2, threads=4,
            environment_count=2, environment_tiles=[[0, 1], [1, 2]],
            probe_tiles=[[0, 8], [8, 8]],
        )


@pytest.mark.parametrize("missing_phase", ["source_gemm", "target_gemm"])
def test_current_vendor_acceptance_requires_both_hot_phases(missing_phase):
    records = [
        record for record in _current_terminal_vendor_records()
        if record["phase"] != missing_phase
    ]
    with pytest.raises(RuntimeError, match=f"no vendor telemetry for {missing_phase}"):
        benchmark._validate_current_vendor_acceptance(
            records, samples=7, variants=13, block_width=8,
            probes=16, annotation_bins=2, threads=4,
            environment_count=2, environment_tiles=[[0, 1], [1, 2]],
            probe_tiles=[[0, 8], [8, 8]],
        )


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "terminal_only"])
def test_current_vendor_acceptance_requires_exact_cartesian_context_once(fault):
    records = _current_terminal_vendor_records()
    if fault == "missing":
        records.pop(0)
    elif fault == "duplicate":
        records.append(dict(records[0]))
    elif fault == "extra":
        extra = dict(next(record for record in records if record["phase"] == "source_gemm"))
        extra.update(
            {
                "environment_tile": [0, 2],
                "environment_names": ["age", "bmi"],
                "m": 7, "n": 64, "k": 8,
                "lda": 7, "ldb": 8, "ldc": 7,
            }
        )
        records.append(extra)
    else:
        records = [
            record for record in records
            if record["genotype_block_index"] == 1
        ]
    with pytest.raises(RuntimeError, match="complete .* schedule once each"):
        benchmark._validate_current_vendor_acceptance(
            records, samples=7, variants=13, block_width=8,
            probes=16, annotation_bins=2, threads=4,
            environment_count=2, environment_tiles=[[0, 1], [1, 2]],
            probe_tiles=[[0, 8], [8, 8]],
        )


def _numa_bound_decode_report(
    *, samples: int, variants: int, block_width: int, passes: int,
    nodes: list[int] | None = None,
) -> dict:
    nodes = [0] if nodes is None else list(nodes)
    page_size = os.sysconf("SC_PAGE_SIZE")
    blocks = [
        [start, min(start + block_width, variants)]
        for start in range(0, variants, block_width)
    ]
    records = []
    total_payload = 0
    total_mapping = 0
    maximum_payload = 0
    maximum_mapping = 0
    for start, stop in blocks * passes:
        byte_count = samples * (stop - start) * 8
        mapping_bytes = (
            (byte_count + page_size - 1) // page_size
        ) * page_size
        page_count = mapping_bytes // page_size
        common = {
            "schema": benchmark.NUMA_BOUND_BUFFER_SCHEMA,
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
        records.append(
            {
                "genotype_block": [start, stop],
                "memory_order": "F",
                "decoder": "bed_reader.read_f64_into_bound_mapping",
                "bound_mapping_preserved_after_standardization": True,
                "verification_stage": "post_standardization_pre_return",
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
                    "query_chunks": (
                        page_count + benchmark.NUMA_PAGE_QUERY_CHUNK_LIMIT - 1
                    ) // benchmark.NUMA_PAGE_QUERY_CHUNK_LIMIT,
                    "query_chunk_page_limit": (
                        benchmark.NUMA_PAGE_QUERY_CHUNK_LIMIT
                    ),
                    "node_histogram": {str(nodes[0]): page_count},
                    "ordered_status_sha256": "f" * 64,
                    "ordered_status_encoding": (
                        f"native_32bit_signed_{sys.byteorder}"
                    ),
                    "complete": True,
                },
            }
        )
        total_payload += byte_count
        total_mapping += mapping_bytes
        maximum_payload = max(maximum_payload, byte_count)
        maximum_mapping = max(maximum_mapping, mapping_bytes)
    return {
        "schema": benchmark.NUMA_BOUND_DECODE_SCHEMA,
        "schema_version": 1,
        "required": True,
        "bounded": True,
        "decoder": "bed_reader.read_f64_into_bound_mapping",
        "memory_order": "F",
        "verification_stage": "post_standardization_pre_return",
        "selected_nodes": nodes,
        "sample_count": samples,
        "num_variants": variants,
        "float64_itemsize": 8,
        "genotype_blocks": blocks,
        "shared_genotype_passes": passes,
        "expected_block_read_count": len(records),
        "observed_block_read_count": len(records),
        "max_payload_bytes_per_block": maximum_payload,
        "max_mapping_bytes_per_block": maximum_mapping,
        "total_payload_bytes_across_reads": total_payload,
        "total_mapping_bytes_across_reads": total_mapping,
        "complete_page_query_records": len(records),
        "records_included": True,
        "records": records,
        "complete": True,
    }


def test_numa_bound_decode_controller_reconstructs_every_page_and_block():
    report = _numa_bound_decode_report(
        samples=7, variants=13, block_width=8, passes=3, nodes=[0, 1]
    )
    # Use both selected nodes while retaining exact complete coverage.
    first = report["records"][0]["verification"]
    page_count = first["page_count"]
    if page_count > 1:
        first["node_histogram"] = {"0": page_count - 1, "1": 1}
    observed = benchmark._validate_numa_bound_decode_report(
        report,
        selected_nodes=[0, 1],
        samples=7,
        variants=13,
        block_width=8,
        passes=3,
    )
    assert observed["complete"] is True
    assert observed["expected_block_read_count"] == 6
    assert observed["queried_pages"] == sum(
        record["verification"]["page_count"] for record in report["records"]
    )
    assert observed["page_migration_requested"] is False
    assert observed["placement_repair_performed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remote", "malformed histogram"),
        ("migration", "exact page contract"),
        ("missing_record", "absent or incomplete"),
        ("wrong_block", "changed its block contract"),
        ("boolean_count", "exact page contract"),
        ("boolean_node", "invalid JSON types"),
        ("float_block", "invalid JSON types"),
        ("boolean_record_node", "exact page contract"),
    ],
)
def test_numa_bound_decode_controller_rejects_forged_evidence(
    mutation, message
):
    report = _numa_bound_decode_report(
        samples=7, variants=13, block_width=8, passes=2
    )
    if mutation == "remote":
        verification = report["records"][0]["verification"]
        verification["node_histogram"] = {
            "0": verification["page_count"] - 1,
            "5": 1,
        }
    elif mutation == "migration":
        report["records"][0]["verification"][
            "page_migration_requested"
        ] = True
    elif mutation == "missing_record":
        report["records"].pop()
    elif mutation == "wrong_block":
        report["records"][0]["genotype_block"] = [1, 8]
    elif mutation == "boolean_count":
        report["records"][0]["verification"]["queried_pages"] = True
    elif mutation == "boolean_node":
        report["selected_nodes"] = [False]
    elif mutation == "float_block":
        report["genotype_blocks"] = [
            [float(start), float(stop)]
            for start, stop in report["genotype_blocks"]
        ]
    else:
        report["records"][0]["allocation"]["selected_nodes"] = [False]
    with pytest.raises(RuntimeError, match=message):
        benchmark._validate_numa_bound_decode_report(
            report,
            selected_nodes=[0],
            samples=7,
            variants=13,
            block_width=8,
            passes=2,
        )


def _completed_payload(layout: str) -> tuple[dict, SimpleNamespace]:
    canonical = layout.replace("-", "_")
    optimized = canonical == "source_tt_target_current"
    source = (
        "dgemm_tt", "column_major", "T", "T", 128, 1024, 512,
        512, 1024, 128
    ) if optimized else (
        "dgemm_nn", "column_major", "N", "N", 1024, 128, 512,
        1024, 512, 1024
    )
    target = (
        "dgemm_tn", "column_major", "T", "N", 512, 256, 1024,
        1024, 1024, 512
    )
    early_attestation = _early_numa_attestation()
    decode_report = _numa_bound_decode_report(
        samples=1024, variants=512, block_width=512, passes=2
    )
    records = []
    for phase, specification in (("source_gemm", source), ("target_gemm", target)):
        (
            operation, storage_layout, transpose_a, transpose_b,
            m, n, k, lda, ldb, ldc,
        ) = specification
        record = {
                "sequence": len(records) + 1,
                "native_sequence": len(records) + 101,
                "phase": phase, "telemetry_scope": "vendor_call",
                "operation": operation, "layout": storage_layout,
                "transpose_a": transpose_a, "transpose_b": transpose_b,
                "m": m, "n": n, "k": k,
                "lda": lda, "ldb": ldb, "ldc": ldc,
                "completed": True,
                "wall_seconds": 0.1, "process_cpu_seconds": 0.1,
                "achieved_gflops": 1.0, "gflops_per_second": 1.0,
                "active_core_equivalents": 1.0,
                "configured_threads": 1, "requested_threads": 1,
                "requested_blas_threads": 1, "backend_threads": 1,
                "environment_tile": [0, 2],
                "environment_names": ["age", "bmi"],
                "probe_tile": [0, 32],
                "genotype_block": [0, 512],
                "genotype_block_index": 0,
                "genotype_block_width": 512,
                "omp_in_parallel": False,
                "affinity_core_list": "0",
                "cpu_affinity_list": "0",
                "cpu_affinity_count": 1,
                "entry_cpu": 0,
                "exit_cpu": 0,
                "numa_node_placement": {
                    "process_policy": {
                        "applied_policy": "libnuma:membind:0",
                        "policy_provenance": "pre_numeric_import",
                        "early_numa_attestation": dict(early_attestation),
                    }
                },
            }
        record["operand_numa_page_samples"] = _queried_numa_samples(record)
        record["native_integrity_snapshot_numa"] = (
            _native_integrity_snapshot_numa(record, minimum_vendor_flops=1)
        )
        record["native_gemm_output_numa"] = _native_gemm_output_numa(
            record, call_id=len(records) + 1
        )
        records.append(record)
    output_numa_records = [
        dict(record["native_gemm_output_numa"]) for record in records
    ]
    output_numa_status = {
        "schema_version": 1,
        "capacity": benchmark.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY,
        "buffered_records": 0,
        "captured_records": len(output_numa_records),
        "dropped_records": 0,
        "next_call_id": len(output_numa_records) + 1,
        "query_chunk_page_limit": (
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
        ),
        "attempted_calls": len(output_numa_records),
        "verified_calls": len(output_numa_records),
        "legacy_calls": 0,
        "failed_calls": 0,
    }
    performance = {
        "requested_blas_threads": 1, "arithmetic_dtype": "float64",
        "requested_storage_dtype": "float64", "full_precision_layout": canonical,
        "backend_build_sha256": "a" * 64,
        "capture_boundary": "post_output_artifact_publication_pre_batch_manifest",
        "telemetry_complete": True, "vendor_call_telemetry_complete": True,
        "hot_gemm_telemetry_complete": True, "phase_telemetry_complete": True,
        "optimized_fp64_layout_telemetry_complete": True,
        "optimized_fp64_layout_telemetry": {
            "required": optimized,
            "complete": True,
            "full_precision_layout": canonical,
            "violation_count": 0,
            "phases": {
                "source_gemm": {"complete": True},
                "target_gemm": {"complete": True},
            },
        },
        "repaired_gemm_output_columns": 0,
        "optimized_fp64_layout_zero_repair": True,
        "early_numa_attestation": dict(early_attestation),
        "numa_bound_bed_decode_required": True,
        "numa_bound_bed_decode_complete": True,
        "numa_bound_bed_decode": decode_report,
        "native_gemm_output_numa_contract_supported": True,
        "native_gemm_output_numa_contract_required": True,
        "native_gemm_output_numa_evidence_available": True,
        "native_gemm_output_numa_evidence_complete": True,
        "native_gemm_output_numa_record_count": len(output_numa_records),
        "dropped_native_gemm_output_numa_records": 0,
        "native_gemm_output_numa_evidence_status": output_numa_status,
        "native_gemm_output_numa_records": output_numa_records,
        "gemm_records": records,
    }
    payload = {
        "kind": "summit.gxe.multi_environment_reference_batch",
        "execution": "shared_in_memory_decoded_blocks",
        "requested_backend": "direct", "full_precision_layout": canonical,
        "protected_native_gemm": True, "requested_storage_dtype": "float64",
        "arithmetic_dtype": "float64", "num_environments": 2,
        "num_variants": 512, "common_complete_case_samples": 1024,
        "environment_tiles": [[0, 2]],
        "shared_genotype_passes": 2,
        "randomization": {
            "distribution": "rademacher", "num_vectors": 32,
            "seed": 7, "probe_offset": 0, "probe_tiles": [[0, 32]],
        },
        "gemm_backend_build_sha256": "a" * 64,
        "native_blas_runtime_isolation": "private_static",
        "native_gemm_integrity_enabled": True,
        "repaired_gemm_output_columns": 0,
        "early_numa_attestation": dict(early_attestation),
        "numa_bound_bed_decode_required": True,
        "numa_bound_bed_decode_complete": True,
        "numa_bound_bed_decode": benchmark._compact_numa_bound_decode_report(
            decode_report
        ),
        "source_panel_memory_order": "C" if optimized else "F",
        "target_panel_memory_order": "F",
        "target_genotype_memory_order": "F",
        "source_to_target_layout_transition": (
            "explicit_c_to_f_before_protected_pair_sealing"
            if optimized else "none"
        ),
        "source_to_target_layout_copy_count": 1 if optimized else 0,
        "source_to_target_layout_copy_total_gib": 0.01 if optimized else 0.0,
        "source_to_target_layout_copy_max_tile_gib": 0.01 if optimized else 0.0,
        "modeled_packed_source_panel_gib": 0.01,
        "modeled_layout_conversion_live_peak_gib": 0.02 if optimized else 0.01,
        "modeled_target_pair_sealing_live_peak_gib": 0.03,
        "performance_telemetry": performance,
    }
    args = SimpleNamespace(
        environment_columns="age,bmi", full_precision_layout=layout,
        storage_dtype="float64", probes=32, seed=7,
        allow_integrity_disabled=False, threads=1, block_width=512,
    )
    return payload, args


@pytest.mark.parametrize("layout", ["current", "source-tt-target-current"])
def test_completed_manifest_validates_storage_layout_hash_and_final_telemetry(layout):
    payload, args = _completed_payload(layout)
    subset = benchmark.PlinkMetadata(
        "/unused", 1024, 512, 128, 65539, 65539
    )
    performance = benchmark._validate_completed_manifest(
        payload, subset=subset, args=args, native_sha256="a" * 64,
        cpu_records=[benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)],
        full_precision_layout=layout, annotation_bin_count=1,
        reference_sample_count=1024, expected_child_pid=12345,
    )
    assert performance["telemetry_complete"] is True
    assert performance["controller_hot_vendor_numa_validation"][
        "all_hot_operands_queried_resolved_local"
    ] is True
    assert performance["controller_early_numa_attestation_requirement"][
        "complete"
    ] is True
    if layout == "current":
        acceptance = performance["controller_current_vendor_acceptance"]
        assert acceptance["gate_passed"] is True
        assert acceptance["source_record_count"] == 1
        assert acceptance["target_record_count"] == 1
        assert acceptance["semantic_p_values"] == [128]

    payload["performance_telemetry"]["capture_boundary"] = "pre_output"
    with pytest.raises(RuntimeError, match="final boundary"):
        benchmark._validate_completed_manifest(
            payload, subset=subset, args=args, native_sha256="a" * 64,
            cpu_records=[benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)],
            full_precision_layout=layout, annotation_bin_count=1,
            reference_sample_count=1024, expected_child_pid=12345,
        )


    payload, args = _completed_payload(layout)
    payload["performance_telemetry"]["gemm_records"][0]["lda"] += 1
    with pytest.raises(RuntimeError, match="leading dimensions"):
        benchmark._validate_completed_manifest(
            payload, subset=subset, args=args, native_sha256="a" * 64,
            cpu_records=[benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)],
            full_precision_layout=layout, annotation_bin_count=1,
            reference_sample_count=1024, expected_child_pid=12345,
        )

    payload, args = _completed_payload(layout)
    payload["performance_telemetry"]["gemm_records"][0][
        "operand_numa_page_samples"
    ]["operands"]["b"]["node_histogram"] = {"1": 8}
    for sample in payload["performance_telemetry"]["gemm_records"][0][
        "operand_numa_page_samples"
    ]["operands"]["b"]["ordered_samples"]:
        sample["raw_move_pages_status"] = 1
        sample["numa_node"] = 1
    with pytest.raises(RuntimeError, match="outside selected NUMA nodes"):
        benchmark._validate_completed_manifest(
            payload, subset=subset, args=args, native_sha256="a" * 64,
            cpu_records=[benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)],
            full_precision_layout=layout, annotation_bin_count=1,
            reference_sample_count=1024, expected_child_pid=12345,
        )

    payload, args = _completed_payload(layout)
    for operand in payload["performance_telemetry"]["gemm_records"][0][
        "operand_numa_page_samples"
    ]["operands"].values():
        operand["query_status"] = "unsupported"
        operand["resolved_sample_pages"] = 0
        operand["node_histogram"] = {}
        for sample in operand["ordered_samples"]:
            sample["status_kind"] = "unavailable"
            sample["raw_move_pages_status"] = None
            sample["numa_node"] = None
    unsupported_summary = benchmark._hot_vendor_numa_summary(
        payload["performance_telemetry"]["gemm_records"], {0}
    )
    assert unsupported_summary["all_hot_operands_queried_resolved_local"] is False
    assert set(unsupported_summary["records"][0]["operand_statuses"].values()) == {
        "unsupported"
    }
    with pytest.raises(RuntimeError, match="requires all hot vendor"):
        benchmark._validate_completed_manifest(
            payload, subset=subset, args=args, native_sha256="a" * 64,
            cpu_records=[benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)],
            full_precision_layout=layout, annotation_bin_count=1,
            reference_sample_count=1024, expected_child_pid=12345,
        )


def test_native_integrity_snapshot_numa_gate_derives_eligibility_and_ignores_fallback():
    payload, _args = _completed_payload("current")
    records = payload["performance_telemetry"]["gemm_records"]
    fallback = {
        "telemetry_scope": "deterministic_tiled_call_boundary",
        "phase": "source_gemm",
        "operation": "dgemm_nn",
    }
    records.append(fallback)
    summary = benchmark._native_integrity_snapshot_numa_summary(
        records, {0}, integrity_enabled=True,
        integrity_minimum_vendor_flops=1,
        query_chunk_page_limit=(
            benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
        ),
    )
    source = next(record for record in records if record.get("phase") == "source_gemm")
    evidence = source["native_integrity_snapshot_numa"]
    assert summary["checked_source_nn_vendor_record_count"] == 1
    assert summary["nonchecked_hot_vendor_record_count"] == 1
    assert summary["non_vendor_record_count_ignored"] == 1
    assert summary["mapping_page_count_total"] == evidence["mapping_page_count"]
    assert summary["record_evidence_included"] is True
    assert len(summary["records"]) == 1
    retained = summary["records"][0]
    assert retained == {
        "record_index": 0,
        "hot_vendor_index": 0,
        "sequence": 1,
        "native_sequence": 101,
        "phase": "source_gemm",
        "operation": "dgemm_nn",
        "m": source["m"], "n": source["n"], "k": source["k"],
        "environment_tile": [0, 2],
        "environment_names": ["age", "bmi"],
        "probe_tile": [0, 32],
        "genotype_block": [0, 512],
        "genotype_block_index": 0,
        "genotype_block_width": 512,
        "native_integrity_snapshot_numa": evidence,
    }
    evidence["node_histogram"]["0"] = -1
    assert retained["native_integrity_snapshot_numa"]["node_histogram"]["0"] > 0

    source_flops = 2 * source["m"] * source["n"] * source["k"]
    source["native_integrity_snapshot_numa"] = _native_integrity_snapshot_numa(
        source, minimum_vendor_flops=source_flops + 1
    )
    nonchecked = benchmark._native_integrity_snapshot_numa_summary(
        records, {0}, integrity_enabled=True,
        integrity_minimum_vendor_flops=source_flops + 1,
        query_chunk_page_limit=(
            benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
        ),
    )
    assert nonchecked["checked_source_nn_vendor_record_count"] == 0
    assert nonchecked["nonchecked_hot_vendor_record_count"] == 2
    assert nonchecked["records"] == []


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing", "noncanonical native integrity snapshot schema"),
        ("extra", "noncanonical native integrity snapshot schema"),
        ("role", "discriminator is not exact"),
        ("contract_type", "discriminator is not exact"),
        ("logical_type", "must be an exact JSON integer"),
        ("sequence_type", "must be an exact JSON integer"),
        ("native_sequence_type", "must be an exact JSON integer"),
        ("sequence_zero", "sequences must be positive"),
        ("native_sequence_zero", "sequences must be positive"),
        ("mapping_pages", "range/NUMA contract"),
        ("selected_nodes", "range/NUMA contract"),
        ("seal", "dedicated, prebound, sealed"),
        ("pre_touch", "dedicated, prebound, sealed"),
        ("queried_pages", "range/NUMA contract"),
        ("query_chunks", "range/NUMA contract"),
        ("histogram", "incomplete or outside selected"),
        ("digest", "dedicated, prebound, sealed"),
        ("strict", "dedicated, prebound, sealed"),
        ("migration", "dedicated, prebound, sealed"),
        ("repair", "dedicated, prebound, sealed"),
    ],
)
def test_native_integrity_snapshot_numa_gate_rejects_forged_evidence(
    fault, message
):
    payload, _args = _completed_payload("current")
    records = json.loads(json.dumps(
        payload["performance_telemetry"]["gemm_records"]
    ))
    source = next(record for record in records if record["phase"] == "source_gemm")
    evidence = source["native_integrity_snapshot_numa"]
    if fault == "missing":
        evidence.pop("sealed_read_only_before_vendor")
    elif fault == "extra":
        evidence["unversioned_extra"] = True
    elif fault == "role":
        evidence["operand_role"] = "logical_b"
    elif fault == "contract_type":
        evidence["contract_required"] = 1
    elif fault == "logical_type":
        evidence["logical_byte_count"] = True
    elif fault == "sequence_type":
        source["sequence"] = True
    elif fault == "native_sequence_type":
        source["native_sequence"] = True
    elif fault == "sequence_zero":
        source["sequence"] = 0
    elif fault == "native_sequence_zero":
        source["native_sequence"] = 0
    elif fault == "mapping_pages":
        evidence["mapping_page_count"] += 1
    elif fault == "selected_nodes":
        evidence["selected_nodes"] = [False]
    elif fault == "seal":
        evidence["sealed_read_only_before_vendor"] = False
    elif fault == "pre_touch":
        evidence["bound_before_first_touch"] = False
    elif fault == "queried_pages":
        evidence["queried_pages"] -= 1
    elif fault == "query_chunks":
        evidence["query_chunks"] = 0
    elif fault == "histogram":
        evidence["node_histogram"] = {"1": evidence["mapping_page_count"]}
    elif fault == "digest":
        evidence["ordered_status_sha256"] = "A" * 64
    elif fault == "strict":
        evidence["pre_vendor_strict_policy_verified"] = False
    elif fault == "migration":
        evidence["page_migration_requested"] = 0
    else:
        evidence["placement_repair_performed"] = True
    with pytest.raises(RuntimeError, match=message):
        benchmark._native_integrity_snapshot_numa_summary(
            records, {0}, integrity_enabled=True,
            integrity_minimum_vendor_flops=1,
            query_chunk_page_limit=(
                benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
            ),
        )


def test_native_integrity_snapshot_numa_nonchecked_vendor_schema_is_minimal():
    payload, _args = _completed_payload("current")
    records = payload["performance_telemetry"]["gemm_records"]
    target = next(record for record in records if record["phase"] == "target_gemm")
    target["native_integrity_snapshot_numa"]["mapping_bytes"] = 4096
    with pytest.raises(RuntimeError, match="nonchecked hot vendor record"):
        benchmark._native_integrity_snapshot_numa_summary(
            records, {0}, integrity_enabled=True,
            integrity_minimum_vendor_flops=1,
            query_chunk_page_limit=(
                benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
            ),
        )


def _validated_native_output_summary(performance: dict) -> dict:
    return benchmark._native_gemm_output_numa_summary(
        performance["gemm_records"],
        performance["native_gemm_output_numa_records"],
        performance["native_gemm_output_numa_evidence_status"],
        {0},
        query_chunk_page_limit=(
            benchmark.NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
        ),
        evidence_capacity=benchmark.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY,
    )


def test_native_gemm_output_numa_gate_joins_vendor_and_deterministic_calls():
    payload, _args = _completed_payload("current")
    performance = payload["performance_telemetry"]
    performance["gemm_records"][1]["telemetry_scope"] = (
        "deterministic_tiled_call_boundary"
    )
    summary = _validated_native_output_summary(performance)
    assert summary["complete"] is True
    assert summary["protected_output_call_count"] == 2
    assert summary["hot_source_target_record_count"] == 2
    assert summary["hot_source_target_call_count"] == 2
    assert summary["mapping_page_count_total"] == sum(
        evidence["mapping_page_count"]
        for evidence in performance["native_gemm_output_numa_records"]
    )
    assert summary["one_evidence_record_per_call_id"] is True
    assert summary["exact_contiguous_call_id_queue_order"] is True
    assert summary["every_evidence_record_joined_to_gemm_telemetry"] is True
    assert summary["record_evidence_included"] is True
    assert summary["single_node_ordered_status_digest_recomputed_count"] == 2
    assert "canonical native SHA-256 retained" in summary[
        "ordered_status_digest_evidence"
    ]
    assert [item["call_id"] for item in summary["records"]] == [1, 2]
    performance["native_gemm_output_numa_records"][0]["node_histogram"]["0"] = -1
    assert summary["records"][0]["native_gemm_output_numa"][
        "node_histogram"
    ]["0"] > 0


def _canonical_tt_output_performance() -> dict:
    payload, _args = _completed_payload("source-tt-target-current")
    performance = payload["performance_telemetry"]
    source = performance["gemm_records"][0]
    assert (
        source["operation"], source["layout"],
        source["transpose_a"], source["transpose_b"],
        source["lda"], source["ldb"], source["ldc"],
    ) == (
        "dgemm_tt", "column_major", "T", "T",
        source["k"], source["n"], source["m"],
    )
    call_id = source["native_gemm_output_numa"]["call_id"]
    drained = next(
        evidence
        for evidence in performance["native_gemm_output_numa_records"]
        if evidence["call_id"] == call_id
    )
    for evidence in (source["native_gemm_output_numa"], drained):
        evidence["logical_rows"] = source["n"]
        evidence["logical_columns"] = source["m"]
        evidence["storage_layout"] = "row_major"
    return performance


def test_native_gemm_output_numa_gate_accepts_exact_tt_row_major_alias():
    summary = _validated_native_output_summary(
        _canonical_tt_output_performance()
    )
    source = summary["records"][0]
    assert source["operation"] == "dgemm_tt"
    assert source["native_gemm_output_numa"]["storage_layout"] == "row_major"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", "dgemm_nn"),
        ("layout", "row_major"),
        ("transpose_a", "N"),
        ("transpose_b", "N"),
        ("lda", 511),
        ("ldb", 1023),
        ("ldc", 127),
        ("lda", True),
        ("evidence_storage_layout", "column_major"),
        ("evidence_logical_rows", 128),
        ("evidence_logical_columns", 1024),
    ],
)
def test_native_gemm_output_numa_gate_rejects_near_tt_aliases(field, value):
    performance = _canonical_tt_output_performance()
    if field.startswith("evidence_"):
        evidence_field = field.removeprefix("evidence_")
        call_id = performance["gemm_records"][0][
            "native_gemm_output_numa"
        ]["call_id"]
        performance["gemm_records"][0]["native_gemm_output_numa"][
            evidence_field
        ] = value
        next(
            evidence
            for evidence in performance["native_gemm_output_numa_records"]
            if evidence["call_id"] == call_id
        )[evidence_field] = value
    else:
        performance["gemm_records"][0][field] = value
    with pytest.raises(
        RuntimeError,
        match="applicable executor output mapping|leading dimensions|not exact",
    ):
        _validated_native_output_summary(performance)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing", "noncanonical schema"),
        ("extra", "noncanonical schema"),
        ("logical_columns", "not exact"),
        ("bool_type", "not exact"),
        ("queried_pages", "not exact"),
        ("histogram", "incomplete or outside"),
        ("digest", "not exact"),
        ("canonical_wrong_digest", "ordered-status digest"),
        ("strict", "not exact"),
        ("boundary", "not exact"),
        ("migration", "not exact"),
        ("repair", "not exact"),
        ("duplicate_call_id", "not one-to-one"),
        ("reversed_order", "exact contiguous queue order"),
        ("call_id_gap", "exact contiguous queue order"),
        ("next_call_id", "exact contiguous queue order"),
        ("nested_join", "does not join exactly"),
        ("missing_hot", "lacks applicable"),
        ("nonhot_shape", "applicable executor output mapping"),
        ("status_drop", "incomplete or lossy"),
        ("status_type", "exact JSON integer"),
    ],
)
def test_native_gemm_output_numa_gate_rejects_forged_evidence(fault, message):
    payload, _args = _completed_payload("current")
    performance = json.loads(json.dumps(payload["performance_telemetry"]))
    evidence = performance["native_gemm_output_numa_records"][0]
    if fault == "missing":
        evidence.pop("writable_output")
    elif fault == "extra":
        evidence["unversioned_extra"] = True
    elif fault == "logical_columns":
        evidence["logical_columns"] += 1
    elif fault == "bool_type":
        evidence["writable_output"] = 1
    elif fault == "queried_pages":
        evidence["queried_pages"] -= 1
    elif fault == "histogram":
        evidence["node_histogram"] = {
            "1": evidence["mapping_page_count"]
        }
    elif fault == "digest":
        evidence["ordered_status_sha256"] = "A" * 64
    elif fault == "canonical_wrong_digest":
        evidence["ordered_status_sha256"] = "0" * 64
    elif fault == "strict":
        evidence["post_repair_strict_policy_verified"] = False
    elif fault == "boundary":
        evidence["verification_boundary"] = "before_repair"
    elif fault == "migration":
        evidence["page_migration_requested"] = 0
    elif fault == "repair":
        evidence["placement_repair_performed"] = True
    elif fault == "duplicate_call_id":
        performance["native_gemm_output_numa_records"][1]["call_id"] = 1
    elif fault == "reversed_order":
        performance["native_gemm_output_numa_records"].reverse()
    elif fault == "call_id_gap":
        performance["native_gemm_output_numa_records"][1]["call_id"] = 3
        performance["gemm_records"][1]["native_gemm_output_numa"][
            "call_id"
        ] = 3
    elif fault == "next_call_id":
        performance["native_gemm_output_numa_evidence_status"][
            "next_call_id"
        ] = 4
    elif fault == "nested_join":
        performance["gemm_records"][0]["native_gemm_output_numa"][
            "call_id"
        ] = 999
    elif fault == "missing_hot":
        performance["gemm_records"][0].pop("native_gemm_output_numa")
    elif fault == "nonhot_shape":
        performance["gemm_records"][0]["phase"] = "feature_moments"
        performance["gemm_records"][0]["m"] += 1
    elif fault == "status_drop":
        performance["native_gemm_output_numa_evidence_status"][
            "dropped_records"
        ] = 1
    else:
        performance["native_gemm_output_numa_evidence_status"][
            "captured_records"
        ] = True
    with pytest.raises(RuntimeError, match=message):
        _validated_native_output_summary(performance)


def test_completed_manifest_requires_exact_blis_vendor_and_placement_contract():
    payload, args = _completed_payload("current")
    args.threads = 2
    args.expected_source_commit = "b" * 40
    args.expected_source_tree_sha256 = "c" * 64
    placement = _openmp_placement([0, 1])
    payload.update(
        {
            "gemm_backend": "gxeldcore_direct",
            "native_source_commit": args.expected_source_commit,
            "native_source_tree_sha256": args.expected_source_tree_sha256,
            "cpu_placement": placement,
            "cpu_placement_complete": True,
        }
    )
    performance = payload["performance_telemetry"]
    performance["requested_blas_threads"] = 2
    performance["cpu_placement"] = placement
    performance["cpu_placement_complete"] = True
    for record in performance["gemm_records"]:
        record.update(
            {
                "active_core_equivalents": 2.0,
                "configured_threads": 2,
                "requested_threads": 2,
                "requested_blas_threads": 2,
                "backend_threads": 2,
                "backend": "gxeldcore_direct",
                "backend_version": "1.9",
                "backend_build_sha256": "a" * 64,
                "native_source_commit": args.expected_source_commit,
                "native_source_tree_sha256": args.expected_source_tree_sha256,
                "blas_backend": "BLIS",
                "blas_backend_config": "BLIS 2.0 config=zen",
                "blas_backend_corename": "zen",
            }
        )
    fallback = dict(performance["gemm_records"][0])
    fallback.pop("phase")
    fallback["telemetry_scope"] = "deterministic_tiled_call_boundary"
    fallback["operation"] = "nn_update"
    fallback.pop("native_gemm_output_numa")
    fallback["affinity_core_list"] = [0]
    for field in (
        "cpu_affinity_list", "cpu_affinity_count", "entry_cpu", "exit_cpu"
    ):
        fallback.pop(field)
    performance["gemm_records"].append(fallback)
    subset = benchmark.PlinkMetadata(
        "/unused", 1024, 512, 128, 65539, 65539
    )
    cpu_records = [
        benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0),
        benchmark.CpuRecord(cpu=1, core=1, socket=0, node=0),
    ]
    def validate():
        return benchmark._validate_completed_manifest(
            payload,
            subset=subset,
            args=args,
            native_sha256="a" * 64,
            cpu_records=cpu_records,
            full_precision_layout="current",
            annotation_bin_count=1,
            reference_sample_count=1024,
            expected_child_pid=12345,
            require_explicit_blis_placement=True,
            expected_placement=placement,
            integrity_minimum_vendor_flops=1,
            native_integrity_snapshot_numa_query_chunk_page_limit=(
                benchmark.NATIVE_INTEGRITY_SNAPSHOT_NUMA_QUERY_CHUNK_LIMIT
            ),
            native_gemm_output_numa_query_chunk_page_limit=(
                benchmark.NATIVE_GEMM_OUTPUT_NUMA_QUERY_CHUNK_LIMIT
            ),
            native_gemm_output_numa_evidence_capacity=(
                benchmark.NATIVE_GEMM_OUTPUT_NUMA_EVIDENCE_CAPACITY
            ),
        )

    observed = validate()
    assert observed["controller_current_vendor_acceptance"][
        "private_blis_identity_required"
    ] is True
    snapshot_summary = observed[
        "controller_native_integrity_snapshot_numa_validation"
    ]
    assert snapshot_summary["complete"] is True
    assert snapshot_summary["checked_source_nn_vendor_record_count"] == 1
    assert snapshot_summary["nonchecked_hot_vendor_record_count"] == 1
    assert snapshot_summary["non_vendor_record_count_ignored"] == 1
    assert snapshot_summary[
        "every_checked_source_nn_snapshot_fully_resolved_local"
    ] is True
    assert snapshot_summary["record_evidence_included"] is True
    assert snapshot_summary["records"][0]["sequence"] == 1
    assert snapshot_summary["records"][0]["native_sequence"] == 101
    assert snapshot_summary["records"][0][
        "native_integrity_snapshot_numa"
    ]["complete"] is True
    assert benchmark._compact_performance(observed)[
        "controller_native_integrity_snapshot_numa_validation"
    ] == snapshot_summary
    output_summary = observed["controller_native_gemm_output_numa_validation"]
    assert output_summary["complete"] is True
    assert output_summary["protected_output_call_count"] == 2
    assert output_summary["hot_source_target_call_count"] == 2
    assert output_summary["one_evidence_record_per_call_id"] is True
    assert output_summary["every_hot_source_target_output_fully_resolved_local"] is True
    assert output_summary["record_evidence_included"] is True
    assert benchmark._compact_performance(observed)[
        "controller_native_gemm_output_numa_validation"
    ] == output_summary
    assert observed["controller_gemm_calling_thread_affinity_validation"] == {
        "complete": True,
        "scope": (
            "affinity_core_list field validated on every GEMM record; "
            "raw native vendor-entry affinity/count and entry/exit CPU "
            "validated on vendor_call records; full OpenMP team placement "
            "validated separately"
        ),
        "record_count": 3,
        "vendor_record_count": 2,
        "non_vendor_record_count": 1,
        "expected_calling_cpu": 0,
        "expected_calling_affinity_cpu_ids": [0],
        "full_team_cpu_ids": [0, 1],
        "full_team_evidence_field": "cpu_placement",
    }

    payload["performance_telemetry"]["gemm_records"][0].update(
        {
            "affinity_core_list": "0-1",
            "cpu_affinity_list": "0-1",
            "cpu_affinity_count": 2,
        }
    )
    with pytest.raises(RuntimeError, match="bound calling-thread placement"):
        validate()
    payload["performance_telemetry"]["gemm_records"][0].update(
        {
            "affinity_core_list": "0",
            "cpu_affinity_list": "0",
            "cpu_affinity_count": 1,
        }
    )

    record = payload["performance_telemetry"]["gemm_records"][0]
    for field, invalid in (
        ("affinity_core_list", "1"),
        ("cpu_affinity_list", "1"),
        ("affinity_core_list", [0]),
        ("cpu_affinity_list", [0]),
        ("cpu_affinity_count", True),
        ("cpu_affinity_count", 2),
        ("entry_cpu", 1),
        ("exit_cpu", 1),
    ):
        original = record[field]
        record[field] = invalid
        with pytest.raises(RuntimeError, match="bound calling-thread placement"):
            validate()
        record[field] = original

    native_affinity = record.pop("cpu_affinity_list")
    with pytest.raises(RuntimeError, match="bound calling-thread placement"):
        validate()
    record["cpu_affinity_list"] = native_affinity

    affinity_alias = record.pop("affinity_core_list")
    with pytest.raises(RuntimeError, match="bound calling-thread placement"):
        validate()
    record["affinity_core_list"] = affinity_alias

    fallback["affinity_core_list"] = [1]
    with pytest.raises(RuntimeError, match="bound calling-thread placement"):
        validate()
    fallback["affinity_core_list"] = [0]

    fallback["telemetry_scope"] = "python_call_boundary"
    with pytest.raises(RuntimeError, match="unexpected scope"):
        validate()
    fallback["telemetry_scope"] = "deterministic_tiled_call_boundary"

    fallback["cpu_affinity_count"] = 1
    with pytest.raises(RuntimeError, match="unexpectedly contains native"):
        validate()
    fallback.pop("cpu_affinity_count")

    payload["performance_telemetry"]["gemm_records"][0][
        "blas_backend"
    ] = "OpenBLAS"
    with pytest.raises(RuntimeError, match="private-BLIS identity mismatch"):
        validate()


def test_single_explicit_group_resolves_hashed_worker_and_reference_contract(
    tmp_path,
):
    placement = _openmp_placement([0])
    early = _early_numa_attestation()
    decode_report = _numa_bound_decode_report(
        samples=100, variants=8000, block_width=2000, passes=2
    )
    args = SimpleNamespace(
        environment_columns="age,bmi",
        storage_dtype="float64",
        threads=1,
        allow_integrity_disabled=False,
        expected_archive_sha256="a" * 64,
        expected_source_commit="b" * 40,
        expected_source_tree_sha256="c" * 64,
        expected_private_source_commit="d" * 40,
        expected_private_source_tree_sha256="e" * 64,
    )
    native_sha256 = "f" * 64
    compile_options = _blis_compile_options(args, placement)
    references = []
    for environment in ("age", "bmi"):
        path = tmp_path / f"reference.{environment}.gxe.ref.json"
        provenance = {
            "artifact_stage": "reference",
            "backend_name": "gxeldcore_direct",
            "backend_version": "1.9",
            "source_commit": args.expected_source_commit,
            "source_tree_sha256": args.expected_source_tree_sha256,
            "native_binary_sha256": native_sha256,
            "compile_options": compile_options,
        }
        feature = dict(provenance)
        feature["artifact_stage"] = "feature_construction"
        reference = {
            "kind": "summit.gxe.reference",
            "schema_version": 3,
            "environment": environment,
            "cpu_placement": placement,
            "cpu_placement_complete": True,
            "backend_provenance": provenance,
            "feature_backend_provenance": feature,
        }
        path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
        references.append(
            {
                "environment": environment,
                "reference": path.name,
                "sha256": benchmark._sha256(path),
            }
        )
    group = {
        "kind": "summit.gxe.multi_environment_reference_batch",
        "schema_version": 1,
        "execution": "shared_in_memory_decoded_blocks",
        "requested_backend": "direct",
        "full_precision_layout": "current",
        "protected_native_gemm": True,
        "arithmetic_dtype": "float64",
        "requested_storage_dtype": "float64",
        "gemm_backend": "gxeldcore_direct",
        "gemm_backend_build_sha256": native_sha256,
        "native_source_commit": args.expected_source_commit,
        "native_source_tree_sha256": args.expected_source_tree_sha256,
        "native_gemm_integrity_enabled": True,
        "native_blas_runtime_isolation": "private_static",
        "repaired_gemm_output_columns": 0,
        "source_panel_memory_order": "F",
        "target_panel_memory_order": "F",
        "target_genotype_memory_order": "F",
        "source_to_target_layout_transition": "none",
        "common_complete_case_samples": 100,
        "num_environments": 2,
        "num_variants": 8000,
        "shared_genotype_passes": 2,
        "randomization": {"distribution": "rademacher", "num_vectors": 32},
        "references": references,
        "cpu_placement": placement,
        "cpu_placement_complete": True,
        "early_numa_attestation": early,
        "numa_bound_bed_decode_required": True,
        "numa_bound_bed_decode_complete": True,
        "numa_bound_bed_decode": benchmark._compact_numa_bound_decode_report(
            decode_report
        ),
        "performance_telemetry": {
            "backend": "gxeldcore_direct",
            "cpu_placement": placement,
            "cpu_placement_complete": True,
            "early_numa_attestation": early,
            "numa_bound_bed_decode_required": True,
            "numa_bound_bed_decode_complete": True,
            "numa_bound_bed_decode": decode_report,
        },
    }
    group_path = tmp_path / "reference.group0.gxe.multi.json"
    group_path.write_text(json.dumps(group) + "\n", encoding="utf-8")
    canonical = dict(group)
    canonical.pop("cpu_placement")
    canonical.pop("performance_telemetry")
    canonical.pop("numa_bound_bed_decode")
    canonical["execution"] = "parallel_isolated_environment_groups"
    canonical["cpu_placements"] = [placement]
    canonical["cpu_placement_complete"] = True
    canonical["early_numa_attestations"] = [early]
    canonical["numa_bound_bed_decode_required"] = True
    canonical["numa_bound_bed_decode_complete"] = True
    canonical["numa_bound_bed_decode_groups"] = [
        benchmark._compact_numa_bound_decode_report(decode_report)
    ]
    canonical["environment_groups"] = [
        {
            "manifest": group_path.name,
            "sha256": benchmark._sha256(group_path),
            "environments": ["age", "bmi"],
            "repaired_gemm_output_columns": 0,
            "early_numa_attestation": early,
            "numa_bound_bed_decode_summary": (
                benchmark._compact_numa_bound_decode_report(decode_report)
            ),
            "performance_telemetry_summary": {"backend": "gxeldcore_direct"},
        }
    ]
    subset = benchmark.PlinkMetadata(
        "/unused", 100, 8000, 25, 200003, 200003
    )
    cpu_records = [benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)]
    observed = benchmark._resolve_single_explicit_blis_group(
        tmp_path / "reference.gxe.multi.json",
        canonical,
        subset=subset,
        args=args,
        native_sha256=native_sha256,
        cpu_records=cpu_records,
        outer_child_pid=999,
    )
    assert observed["group_path"] == group_path
    assert observed["worker_pid"] == 12345
    assert observed["placement"] == placement
    assert observed["integrity_minimum_vendor_flops"] == 1_000_000_000
    assert observed["snapshot_query_chunk_page_limit"] == 65536

    canonical["cpu_placements"] = []
    with pytest.raises(RuntimeError, match="canonical batch placement differs"):
        benchmark._resolve_single_explicit_blis_group(
            tmp_path / "reference.gxe.multi.json",
            canonical,
            subset=subset,
            args=args,
            native_sha256=native_sha256,
            cpu_records=cpu_records,
            outer_child_pid=999,
        )


def test_current_manifest_uses_complete_case_cohort_not_total_fam_and_fails_closed(
    tmp_path, monkeypatch,
):
    payload, args = _completed_payload("current")
    subset = benchmark.PlinkMetadata(
        "/unused", 1026, 512, 257, 131587, 131587
    )
    cpu_records = [benchmark.CpuRecord(cpu=0, core=0, socket=0, node=0)]
    performance = benchmark._validate_completed_manifest(
        payload, subset=subset, args=args, native_sha256="a" * 64,
        cpu_records=cpu_records, full_precision_layout="current",
        annotation_bin_count=1, reference_sample_count=1024,
        expected_child_pid=12345,
    )
    assert performance["controller_current_vendor_acceptance"][
        "analysis_complete_case_samples"
    ] == 1024

    unpublished = tmp_path / "must-not-exist.json"
    for manifest_samples, reference_samples, message in (
        (1023, 1024, "does not match hashed reference"),
        (1027, 1027, "no larger than the validated FAM"),
        (1024, 1023, "does not match hashed reference"),
    ):
        rejected, rejected_args = _completed_payload("current")
        rejected["common_complete_case_samples"] = manifest_samples
        with pytest.raises(RuntimeError, match=message):
            benchmark._validate_completed_manifest(
                rejected, subset=subset, args=rejected_args,
                native_sha256="a" * 64, cpu_records=cpu_records,
                full_precision_layout="current", annotation_bin_count=1,
                reference_sample_count=reference_samples,
                expected_child_pid=12345,
            )
        assert not unpublished.exists()
    monkeypatch.setattr(
        benchmark, "run",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("common_complete_case_samples mismatch")
        ),
    )
    monkeypatch.setattr(benchmark, "_validate_args", lambda *_: None)
    with pytest.raises(RuntimeError, match="common_complete_case_samples mismatch"):
        benchmark.main(
            [
                "--install-prefix", str(tmp_path / "unused-install"),
                "--native-module", str(tmp_path / "unused-native.so"),
                *_placeholder_blis_cli_args(tmp_path),
                "--output", str(unpublished),
            ]
        )
    assert not unpublished.exists()


def test_hashed_environment_references_must_agree_on_complete_case_samples(tmp_path):
    references = []
    for environment in ("age", "bmi"):
        path = tmp_path / f"reference.{environment}.json"
        path.write_text(
            json.dumps(
                {
                    "kind": "summit.gxe.reference", "schema_version": 3,
                    "environment": environment,
                    "annotation_names": ["L2_0"], "n_samples": 5,
                }
            ) + "\n",
            encoding="utf-8",
        )
        references.append(
            {
                "environment": environment, "reference": path.name,
                "sha256": benchmark._sha256(path),
            }
        )
    batch_path = tmp_path / "reference.gxe.multi.json"
    payload = {"references": references}
    observed = benchmark._batch_reference_dimensions(
        batch_path, payload, ["age", "bmi"]
    )
    assert observed == {"annotation_bin_count": 1, "complete_case_samples": 5}

    bmi_path = tmp_path / "reference.bmi.json"
    bmi = json.loads(bmi_path.read_text(encoding="utf-8"))
    bmi["n_samples"] = 4
    bmi_path.write_text(json.dumps(bmi) + "\n", encoding="utf-8")
    references[1]["sha256"] = benchmark._sha256(bmi_path)
    with pytest.raises(RuntimeError, match="disagree on complete-case n_samples"):
        benchmark._batch_reference_dimensions(
            batch_path, payload, ["age", "bmi"]
        )


def test_projection_uses_explicit_phase_and_pass_scaling() -> None:
    performance = {
        "phase_totals": {
            "genotype_read_decode_standardization": {"wall_seconds": 20.0},
            "source_gemm": {"wall_seconds": 10.0},
            "projection_context_correction": {"wall_seconds": 5.0},
            "source_to_target_layout_conversion": {"wall_seconds": 2.0},
            "feature_moments": {"wall_seconds": 5.0},
            "native_binary_integrity_hashing": {"wall_seconds": 5.0},
            "output_bundle_end_to_end": {"wall_seconds": 5.0},
        }
    }
    manifest = {
        "shared_genotype_passes": 2,
        "environment_tiles": [[0, 3]],
        "randomization": {"probe_tiles": [[0, 32]]},
    }

    observed = benchmark._projections(
        elapsed=100.0,
        subset_variants=100,
        full_variants=1000,
        block_width=25,
        measured_probes=32,
        manifest=manifest,
        performance=performance,
    )

    assert observed["measured_subset"]["attributed_phase_seconds"] == 52.0
    assert observed["measured_subset"]["unattributed_seconds"] == 48.0
    assert observed["measured_subset"]["block_count"] == 4
    assert observed["measured_subset"]["terminal_block_width"] == 25
    # Full-M B32: 200 decode + 100 source + 5 projection + 50 feature
    # + 2 layout conversion + 5 fixed hash + 50 output + conservative
    # 480-second residual.
    assert observed["full_m_b32"]["projected_seconds"] == 892.0
    schedule = observed["full_m_b1024_same_tile_schedule"]
    assert schedule["projected_probe_tiles"] == 32
    assert schedule["projected_genotype_passes"] == 65
    # B1024: 6500 decode + 3200 source + 160 projection + 50 feature
    # + 64 layout conversion + 5 fixed hash + 50 output + conservative
    # 480-second residual.
    assert schedule["projected_seconds"] == 10509.0
    assert schedule["projected_seconds_with_fixed_unattributed_residual"] == 10077.0
    by_phase = {item["phase"]: item for item in observed["phase_schedule_model"]}
    assert by_phase["projection_context_correction"]["scaling_class"] == "tile"
    assert by_phase["source_to_target_layout_conversion"]["scaling_class"] == "tile"
    assert by_phase["source_gemm"]["scaling_class"] == "tile_variant"
    assert by_phase["genotype_read_decode_standardization"]["scaling_class"] == (
        "genotype_pass_variant"
    )

    terminal = benchmark._block_schedule(12_000, 3072)
    assert terminal["block_count"] == 4
    assert terminal["full_width_block_count"] == 3
    assert terminal["terminal_block_width"] == 2784
    assert terminal["has_terminal_partial_block"] is True

    speedups = benchmark._block_width_projection_speedups(
        {
            "full_m_b32": {"projected_seconds": 10.0},
            "full_m_b1024_same_tile_schedule": {"projected_seconds": 100.0},
        },
        {
            "full_m_b32": {"projected_seconds": 8.0},
            "full_m_b1024_same_tile_schedule": {"projected_seconds": 80.0},
        },
        baseline_width=2000, candidate_width=3072,
    )
    assert speedups["full_m_b32"]["baseline_over_candidate_speedup"] == 1.25
    assert speedups["full_m_b1024_same_tile_schedule"][
        "candidate_block_width"
    ] == 3072


def test_private_blis_real_gate_has_no_integrity_disable_cli_escape(tmp_path):
    parser = benchmark._parser()
    assert "--allow-integrity-disabled" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--install-prefix", str(tmp_path / "unused-install"),
                "--native-module", str(tmp_path / "unused.so"),
                *_placeholder_blis_cli_args(tmp_path),
                "--allow-integrity-disabled",
                "--output", str(tmp_path / "must-not-exist.json"),
            ]
        )
    with pytest.raises(RuntimeError, match="requires integrity checks"):
        benchmark.run(SimpleNamespace(allow_integrity_disabled=True))


def test_private_blis_acceptance_comparator_is_fail_closed() -> None:
    valid = {
        "native_gemm_integrity_enabled": True,
        "repaired_gemm_output_columns": 0,
    }
    common = {
        "reference_path": Path("/unused-reference.json"),
        "reference": valid,
        "candidate_path": Path("/unused-candidate.json"),
        "candidate": dict(valid),
        "environment_order": ["age", "sex", "bmi"],
        "rtol": benchmark.DEFAULT_COMPARISON_RTOL,
        "atol": benchmark.DEFAULT_COMPARISON_ATOL,
        "reference_run": "reference_t1",
        "candidate_run": "candidate_t32",
        "reference_threads": 1,
        "candidate_threads": 32,
        "acceptance_reference": True,
    }
    disabled = dict(valid, native_gemm_integrity_enabled=False)
    with pytest.raises(RuntimeError, match="lacks enabled integrity"):
        benchmark._compare_completed_blis_outputs(
            **dict(common, reference=disabled)
        )
    repaired = dict(valid, repaired_gemm_output_columns=1)
    with pytest.raises(RuntimeError, match="did not report zero repairs"):
        benchmark._compare_completed_blis_outputs(
            **dict(common, candidate=repaired)
        )
    with pytest.raises(RuntimeError, match="rtol=atol=5e-12"):
        benchmark._compare_completed_blis_outputs(
            **dict(common, rtol=1.0e-9)
        )
    with pytest.raises(RuntimeError, match="T1 numerical reference"):
        benchmark._compare_completed_blis_outputs(
            **dict(common, reference_threads=2)
        )


def test_t1_reference_acceptance_requires_production_block_width(tmp_path):
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--install-prefix", str(tmp_path / "unused-install"),
            "--native-module", str(tmp_path / "unused.so"),
            *_placeholder_blis_cli_args(tmp_path),
            "--compare-blis-t1-reference",
            "--block-width", "1999",
            "--output", str(tmp_path / "must-not-exist.json"),
        ]
    )
    with pytest.raises(SystemExit):
        benchmark._validate_args(parser, args)
    args.allow_integrity_disabled = False
    with pytest.raises(RuntimeError, match="four full K=2000"):
        benchmark.run(args)


def test_t1_reference_acceptance_requires_exact_candidate_cpus(tmp_path):
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--install-prefix", str(tmp_path / "unused-install"),
            "--native-module", str(tmp_path / "unused.so"),
            *_placeholder_blis_cli_args(tmp_path),
            "--compare-blis-t1-reference",
            "--cpu-list", "1-32",
            "--output", str(tmp_path / "must-not-exist.json"),
        ]
    )
    with pytest.raises(SystemExit):
        benchmark._validate_args(parser, args)
    with pytest.raises(RuntimeError, match="candidate CPUs 0-31"):
        benchmark.run(args)


def test_full_socket_numa_nodes_are_derived_from_verified_topology() -> None:
    selected = [
        benchmark.CpuRecord(cpu=cpu, core=cpu, socket=0, node=cpu // 8)
        for cpu in range(32)
    ]
    topology = [
        *selected,
        *[
            benchmark.CpuRecord(
                cpu=cpu, core=cpu - 32, socket=1, node=4 + (cpu - 32) // 8
            )
            for cpu in range(32, 64)
        ],
    ]
    assert benchmark._full_socket_numa_nodes(
        selected, topology_records=topology
    ) == [0, 1, 2, 3]

    cross_socket_node = list(topology)
    cross_socket_node[-1] = benchmark.CpuRecord(
        cpu=63, core=31, socket=1, node=3
    )
    with pytest.raises(RuntimeError, match="spans multiple physical sockets"):
        benchmark._full_socket_numa_nodes(
            selected, topology_records=cross_socket_node
        )


def test_full_socket_early_numa_attestation_is_exact() -> None:
    attestation = _early_numa_attestation()
    attestation.update(
        {
            "requested_nodes": "0-3",
            "effective_nodes": [0, 1, 2, 3],
            "applied_policy": "libnuma:membind:0,1,2,3",
        }
    )
    observed = benchmark._early_numa_attestation_status(
        {"early_numa_attestation": dict(attestation)},
        {"early_numa_attestation": dict(attestation)},
        {0, 1, 2, 3},
        expected_child_pid=12345,
    )
    assert observed["effective_nodes"] == [0, 1, 2, 3]

    wrong = dict(attestation, effective_nodes=[0])
    with pytest.raises(RuntimeError, match="do not match the selected nodes"):
        benchmark._early_numa_attestation_status(
            {"early_numa_attestation": wrong},
            {"early_numa_attestation": wrong},
            {0, 1, 2, 3},
            expected_child_pid=12345,
        )


def test_full_socket_hot_operand_pages_must_remain_within_memory_nodes() -> None:
    payload, _args = _completed_payload("current")
    records = payload["performance_telemetry"]["gemm_records"]
    observed = benchmark._hot_vendor_numa_summary(records, {0, 1, 2, 3})
    assert observed["all_hot_operands_queried_resolved_local"] is True
    operand = records[0]["operand_numa_page_samples"]["operands"]["a"]
    operand["node_histogram"] = {"4": operand["selected_sample_pages"]}
    for sample in operand["ordered_samples"]:
        sample["raw_move_pages_status"] = 4
        sample["numa_node"] = 4
    with pytest.raises(RuntimeError, match="outside selected NUMA nodes"):
        benchmark._hot_vendor_numa_summary(records, {0, 1, 2, 3})


def test_t1_reference_t32_candidate_controller_contract(tmp_path, monkeypatch):
    source = _synthetic_plink(tmp_path, variants=8000)
    environment, covariates = _tabular_inputs(tmp_path)
    install = tmp_path / "install"
    package = install / "summit"
    package.mkdir(parents=True)
    native = package / "gxeldcore.test.so"
    native.write_bytes(b"mock private BLIS native")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    parser = benchmark._parser()
    args = parser.parse_args(
        [
            "--geno-prefix", str(source),
            "--environment-file", str(environment),
            "--covariate-file", str(covariates),
            "--install-prefix", str(install),
            "--native-module", str(native),
            *_blis_cli_identity_args(tmp_path, install, native),
            "--python-executable", sys.executable,
            "--dependency-path", str(tmp_path),
            "--cpu-list", "0-31", "--threads", "32",
            "--blocks", "4", "--block-width", "2000", "--probes", "32",
            "--compare-blis-t1-reference",
            "--reference-timeout-seconds", "420",
            "--timeout-seconds", "240",
            "--controller-timeout-seconds", "600",
            "--temporary-parent", str(scratch),
            "--output", str(tmp_path / "report.json"),
        ]
    )
    benchmark._validate_args(parser, args)
    cpu_records = [
        benchmark.CpuRecord(
            cpu=cpu, core=cpu, socket=0, node=cpu // 8
        )
        for cpu in range(32)
    ]
    monkeypatch.setattr(
        benchmark, "_resolve_cpu_records", lambda _cpu_list, _threads: cpu_records
    )
    monkeypatch.setattr(
        benchmark, "_full_socket_numa_nodes", lambda _records: [0, 1, 2, 3]
    )
    observed_calls = []

    def fake_execute(
        run_args, *, cpu_records, child_environment, timeout_seconds,
        output_prefix, explicit_memory_nodes, **_kwargs,
    ):
        cpus = [record.cpu for record in cpu_records]
        observed_calls.append(
            {
                "threads": run_args.threads,
                "cpus": cpus,
                "nodes": sorted({record.node for record in cpu_records}),
                "memory_nodes": list(explicit_memory_nodes),
                "timeout": timeout_seconds,
                "omp_places": child_environment["OMP_PLACES"],
                "blis_threads": child_environment["BLIS_NUM_THREADS"],
            }
        )
        payload = {
            "native_gemm_integrity_enabled": True,
            "repaired_gemm_output_columns": 0,
            "randomization": {
                "distribution": "rademacher", "num_vectors": 32,
                "seed": args.seed,
            },
            "performance_telemetry": {"phase_totals": {}},
        }
        return {
            "manifest_path": Path(f"{output_prefix}.gxe.multi.json"),
            "payload": payload,
            "canonical_payload": dict(payload),
            "process": {
                "wall_seconds": 1.0,
                "controller_children_ru_maxrss_gib_high_water": 1.0,
            },
            "cpu_placement": _openmp_placement(cpus),
            "performance": payload["performance_telemetry"],
            "gemm_summaries": [],
            "temporary_artifact_identities": [],
            "projections": {
                "full_m_b32": {"projected_seconds": 10.0},
                "full_m_b1024_same_tile_schedule": {
                    "projected_seconds": 320.0
                },
            },
        }

    def fake_compare(*_positional, **kwargs):
        assert kwargs["reference_run"] == "reference_t1"
        assert kwargs["candidate_run"] == "candidate_t32"
        assert kwargs["reference_threads"] == 1
        assert kwargs["candidate_threads"] == 32
        assert kwargs["acceptance_reference"] is True
        assert kwargs["rtol"] == kwargs["atol"] == 5.0e-12
        return {
            "accuracy_gate_passed": True,
            "artifact_level_reference": (
                "fresh_private_blis_t1_numerical_reference"
            ),
        }

    monkeypatch.setattr(benchmark, "_execute_layout", fake_execute)
    monkeypatch.setattr(
        benchmark, "_compare_completed_blis_outputs", fake_compare
    )
    report = benchmark.run(args)

    assert observed_calls == [
        {
            "threads": 1, "cpus": [0], "nodes": [0],
            "memory_nodes": [0, 1, 2, 3], "timeout": 420.0,
            "omp_places": "{0}", "blis_threads": "1",
        },
        {
            "threads": 32, "cpus": list(range(32)),
            "nodes": [0, 1, 2, 3], "memory_nodes": [0, 1, 2, 3],
            "timeout": 240.0,
            "omp_places": ",".join(f"{{{cpu}}}" for cpu in range(32)),
            "blis_threads": "32",
        },
    ]
    assert report["accepted"] is True
    assert report["acceptance_eligible"] is True
    assert report["candidate_selected"] is True
    assert report["status"] == "completed_private_blis_t1_reference_acceptance"
    assert report["process"]["fresh_exec_children"] == 2
    assert report["process"]["reference_child_cap_seconds"] == 420.0
    assert report["process"]["candidate_child_cap_seconds"] == 240.0
    assert report["blis_t1_reference_runs"]["reference_t1"]["role"] == (
        "numerical_reference"
    )
    assert report["blis_t1_reference_runs"]["candidate_t32"]["role"] == (
        "selection_candidate"
    )
    assert report["comparison_control_identity"]["controlled_differences"] == [
        "taskset_cpu_list", "num_threads", "openmp_places", "output_prefix",
    ]
    for command in report["commands"].values():
        assert command[
            command.index("--gxe-explicit-openmp-memory-scope") + 1
        ] == "selected-socket"
        assert command[command.index("--numa-nodes") + 1] == "0-3"
    assert report["cpu_placement"]["memory_numa_nodes"] == [0, 1, 2, 3]
    assert report["blis_t1_reference_runs"]["reference_t1"][
        "numa_nodes"
    ] == [0, 1, 2, 3]
    assert list(scratch.iterdir()) == []
