from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_RESULT_PREFIX = "SUMMIT_OPENMP_PLACEMENT_TEST_RESULT="
_SCHEMA = "summit.openmp_placement_attestation.v1"


def _test_cpu_ids() -> tuple[list[int], int]:
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("the native placement contract requires Linux affinity APIs")
    cpu_ids = sorted(os.sched_getaffinity(0))
    if len(cpu_ids) < 3:
        pytest.skip(
            "the native placement mutation test requires three visible CPUs"
        )
    return cpu_ids[:2], cpu_ids[2]


def _placement_environment(
    cpu_ids: list[int], *, proc_bind: str, max_active_levels: str = "1"
) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = [environment.get("PYTHONPATH", "")]
    python_paths.extend(path for path in sys.path if path)
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys(path for path in python_paths if path)
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
            "OMP_MAX_ACTIVE_LEVELS": max_active_levels,
            "OMP_PROC_BIND": proc_bind,
            "OMP_PLACES": ",".join(f"{{{cpu}}}" for cpu in cpu_ids),
        }
    )
    return environment


def _run_placement_child(
    source: str,
    cpu_ids: list[int],
    *,
    proc_bind: str,
    max_active_levels: str = "1",
) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[1],
        env=_placement_environment(
            cpu_ids,
            proc_bind=proc_bind,
            max_active_levels=max_active_levels,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 and (
        "No module named 'summit'" in completed.stderr
        or "cannot import name 'gxeldcore'" in completed.stderr
    ):
        pytest.skip("the isolated gxeldcore extension is not importable")
    assert completed.returncode == 0, completed.stderr
    records = [
        line[len(_RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(records) == 1, completed.stdout
    return json.loads(records[0])


def _assert_exact_attestation(
    evidence: dict[str, object], cpu_ids: list[int], *, proc_bind: str
) -> None:
    threads = len(cpu_ids)
    assert evidence == {
        "schema": _SCHEMA,
        "schema_version": 1,
        "verified": True,
        "immutable": True,
        "requested_threads": threads,
        "expected_cpu_ids": cpu_ids,
        "omp_dynamic": False,
        "omp_thread_limit": threads,
        "omp_max_active_levels": 1,
        "omp_proc_bind": proc_bind,
        "omp_binding_active": True,
        "omp_num_places": threads,
        "effective_openmp_capacity": threads,
        "place_cpu_ids": [[cpu] for cpu in cpu_ids],
        "team_size": threads,
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
            for index, cpu in enumerate(cpu_ids)
        ],
        "vendor_calls": 0,
    }


@pytest.mark.parametrize(
    ("proc_bind_environment", "proc_bind_evidence"),
    (("SPREAD", "spread"), ("TRUE", "true")),
)
def test_api9_exact_openmp_placement_is_immutable_and_json_safe(
    proc_bind_environment: str, proc_bind_evidence: str
) -> None:
    cpu_ids, outside_cpu = _test_cpu_ids()
    payload = _run_placement_child(
        f"""
        import json
        import os
        from summit import gxeldcore

        cpu_ids = {cpu_ids!r}
        outside_cpu = {outside_cpu!r}
        initial = dict(gxeldcore.build_info())
        gxeldcore.reset_gemm_telemetry()
        rejected_before_configuration = ""
        try:
            gxeldcore.configure_openmp_placement(list(reversed(cpu_ids)), 2)
        except RuntimeError as error:
            rejected_before_configuration = str(error)
        first = dict(gxeldcore.configure_openmp_placement(cpu_ids, 2))
        second = dict(gxeldcore.configure_openmp_placement(cpu_ids, 2))
        configured = dict(gxeldcore.build_info())
        telemetry = dict(gxeldcore.gemm_telemetry_status())

        original_places = os.environ["OMP_PLACES"]
        os.environ["OMP_PLACES"] = "{{{cpu_ids[0]}}}"
        try:
            gxeldcore.configure_openmp_placement(cpu_ids, 2)
        except RuntimeError as error:
            environment_error = str(error)
        else:
            environment_error = ""
        finally:
            os.environ["OMP_PLACES"] = original_places

        import numpy as np
        genotype = np.asfortranarray([[0.0], [1.0]])
        missingness_targets = np.empty((2, 0), dtype=np.float64, order="F")
        os.environ["OMP_PLACES"] = "{{{cpu_ids[0]}}}"
        try:
            gxeldcore.standardize_genotype_block(
                genotype.copy(order="F"), missingness_targets,
                1, False, 1.0e-10, 2,
            )
        except RuntimeError as error:
            entry_environment_error = str(error)
        else:
            entry_environment_error = ""
        finally:
            os.environ["OMP_PLACES"] = original_places

        original_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {{outside_cpu}})
        try:
            gxeldcore.configure_openmp_placement(cpu_ids, 2)
        except RuntimeError as error:
            recheck_affinity_error = str(error)
        else:
            recheck_affinity_error = ""
        try:
            gxeldcore.standardize_genotype_block(
                genotype.copy(order="F"), missingness_targets,
                1, False, 1.0e-10, 2,
            )
        except RuntimeError as error:
            entry_affinity_error = str(error)
        else:
            entry_affinity_error = ""
        finally:
            os.sched_setaffinity(0, original_affinity)

        try:
            gxeldcore.configure_openmp_placement(list(reversed(cpu_ids)), 2)
        except RuntimeError as error:
            identity_error = str(error)
        else:
            identity_error = ""

        print({_RESULT_PREFIX!r} + json.dumps({{
            "initial": initial,
            "first": first,
            "second": second,
            "configured": configured,
            "telemetry": telemetry,
            "main_affinity": sorted(os.sched_getaffinity(0)),
            "rejected_before_configuration": rejected_before_configuration,
            "environment_error": environment_error,
            "entry_environment_error": entry_environment_error,
            "recheck_affinity_error": recheck_affinity_error,
            "entry_affinity_error": entry_affinity_error,
            "identity_error": identity_error,
        }}, sort_keys=True))
        """,
        cpu_ids,
        proc_bind=proc_bind_environment,
    )

    initial = dict(payload["initial"])
    assert initial["api_version"] == 9
    assert initial["backend_version"] == "1.9"
    assert initial["openmp_effective_capacity_policy"] == (
        "bound_places_else_sched_affinity_v1"
    )
    assert initial["openmp_placement_contract_supported"] is True
    assert initial["openmp_placement_contract_schema"] == _SCHEMA
    assert initial["openmp_placement_contract_configured"] is False
    assert initial["openmp_placement_contract_immutable"] is True
    assert initial["openmp_placement_probe_vendor_calls"] == 0
    assert initial["openmp_placement_contract_evidence"] is None

    first = dict(payload["first"])
    second = dict(payload["second"])
    _assert_exact_attestation(first, cpu_ids, proc_bind=proc_bind_evidence)
    _assert_exact_attestation(second, cpu_ids, proc_bind=proc_bind_evidence)
    assert second == first

    configured = dict(payload["configured"])
    assert configured["openmp_placement_contract_configured"] is True
    assert configured["openmp_placement_contract_evidence"] == first
    assert payload["main_affinity"] == [cpu_ids[0]]
    assert payload["telemetry"]["captured_records"] == 0
    assert payload["telemetry"]["buffered_records"] == 0
    assert "expected ordered singleton CPU set" in str(
        payload["rejected_before_configuration"]
    )
    assert "environment changed" in str(payload["environment_error"])
    assert "environment changed" in str(payload["entry_environment_error"])
    assert "affinity escaped" in str(payload["recheck_affinity_error"])
    assert "affinity escaped" in str(payload["entry_affinity_error"])
    assert "different CPU IDs or thread count" in str(payload["identity_error"])


def test_api9_openmp_placement_rejects_nested_active_level_capacity() -> None:
    cpu_ids, _ = _test_cpu_ids()
    payload = _run_placement_child(
        f"""
        import json
        from summit import gxeldcore

        gxeldcore.reset_gemm_telemetry()
        try:
            gxeldcore.configure_openmp_placement({cpu_ids!r}, 2)
        except RuntimeError as error:
            placement_error = str(error)
        else:
            placement_error = ""
        info = dict(gxeldcore.build_info())
        telemetry = dict(gxeldcore.gemm_telemetry_status())
        print({_RESULT_PREFIX!r} + json.dumps({{
            "placement_error": placement_error,
            "configured": info["openmp_placement_contract_configured"],
            "evidence": info["openmp_placement_contract_evidence"],
            "captured_records": telemetry["captured_records"],
        }}, sort_keys=True))
        """,
        cpu_ids,
        proc_bind="SPREAD",
        max_active_levels="2",
    )

    assert "exactly one maximum active level" in str(payload["placement_error"])
    assert payload["configured"] is False
    assert payload["evidence"] is None
    assert payload["captured_records"] == 0
