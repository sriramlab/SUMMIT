from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

import pytest


_RESULT_PREFIX = "SUMMIT_BLIS_TEST_RESULT="
_BLIS_WAY_NAMES = (
    "BLIS_JC_NT",
    "BLIS_PC_NT",
    "BLIS_IC_NT",
    "BLIS_JR_NT",
    "BLIS_IR_NT",
)


def _blis_child_environment(*, manual: bool) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "BLIS_NT",
        "BLIS_TI",
        "BLIS_THREAD_IMPL",
        "BLIS_ARCH_TYPE",
        "BLIS_ARCH_DEBUG",
        "BLIS_PACK_A",
        "BLIS_PACK_B",
        *_BLIS_WAY_NAMES,
    ):
        environment.pop(name, None)
    environment.update(
        {
            "BLIS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "OMP_THREAD_LIMIT": "2",
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
        }
    )
    if manual:
        environment.update(
            {
                "BLIS_JC_NT": "2",
                "BLIS_PC_NT": "1",
                "BLIS_IC_NT": "1",
                "BLIS_JR_NT": "1",
                "BLIS_IR_NT": "1",
            }
        )
    return environment


def _run_blis_child(source: str, *, manual: bool = False) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[1],
        env=_blis_child_environment(manual=manual),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 and (
        "No module named 'summit'" in completed.stderr
        or "cannot import name 'gxeldcore'" in completed.stderr
    ):
        return {"backend": None, "import_error": completed.stderr.strip()}
    assert completed.returncode == 0, completed.stderr
    records = [
        line[len(_RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(records) == 1, completed.stdout
    return json.loads(records[0])


def _require_blis(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("backend") != "upstream_blis":
        pytest.skip("the loaded gxeldcore extension is not the upstream BLIS candidate")
    return payload


def test_private_blis_automatic_contract_is_immutable_and_owner_threaded() -> None:
    payload = _require_blis(
        _run_blis_child(
            f"""
            import json
            import os
            import threading
            from summit import gxeldcore

            initial = dict(gxeldcore.build_info())
            backend = initial.get("private_blas_backend")
            if backend != "upstream_blis":
                print({_RESULT_PREFIX!r} + json.dumps({{"backend": backend}}))
            else:
                os.environ["BLIS_NUM_THREADS"] = "3"
                try:
                    gxeldcore.configure_blas_threads(3)
                except RuntimeError as error:
                    rebaseline_error = str(error)
                else:
                    rebaseline_error = ""
                finally:
                    os.environ["BLIS_NUM_THREADS"] = "2"
                configured = int(gxeldcore.configure_blas_threads(2))
                info = dict(gxeldcore.build_info())
                thread_errors = []

                def configure_from_nonowner():
                    try:
                        gxeldcore.configure_blas_threads(2)
                    except RuntimeError as error:
                        thread_errors.append(str(error))

                worker = threading.Thread(target=configure_from_nonowner)
                worker.start()
                worker.join()
                os.environ["BLIS_NUM_THREADS"] = "3"
                try:
                    gxeldcore.configure_blas_threads(2)
                except RuntimeError as error:
                    environment_error = str(error)
                else:
                    environment_error = ""
                finally:
                    os.environ["BLIS_NUM_THREADS"] = "2"
                restored = int(gxeldcore.configure_blas_threads(2))
                print({_RESULT_PREFIX!r} + json.dumps({{
                    "backend": backend,
                    "initial_owner_configured": initial[
                        "blas_runtime_owner_thread_configured"
                    ],
                    "configured": configured,
                    "restored": restored,
                    "info": info,
                    "thread_errors": thread_errors,
                    "environment_error": environment_error,
                    "rebaseline_error": rebaseline_error,
                    "module_path": gxeldcore.__file__,
                }}, sort_keys=True))
            """
        )
    )

    assert payload["initial_owner_configured"] is True
    assert payload["configured"] == payload["restored"] == 2
    info = dict(payload["info"])
    assert info["api_version"] == 9
    assert info["backend_version"] == "1.9"
    assert info["blas_vendor"] == "BLIS"
    assert info["blas_runtime_isolation"] == "private_static"
    assert info["gemm_integrity_enabled"] is True
    assert info["gemm_checksum_enabled"] is False
    assert info["gemm_execution_mode"] == "serialized_fixed_private_blis"
    assert info["blas_runtime_threads"] == 2
    assert info["blas_runtime_threading_layer"] == "pthreads"
    assert (
        info["blas_runtime_worker_affinity_policy"]
        == "inherit_authenticated_selected_cpu_set_per_call"
    )
    assert info["blas_runtime_thread_strategy"] == "automatic"
    assert info["blas_runtime_thread_ways"] == {
        "jc": 1,
        "pc": 1,
        "ic": 1,
        "jr": 1,
        "ir": 1,
    }
    assert info["blas_runtime_owner_thread_enforced"] is True
    assert info["blas_runtime_owner_thread_configured"] is True
    assert info["blas_runtime_environment_immutable"] is True
    assert info["blas_runtime_environment_contract"] == "blis_process_start_v1"
    assert info["blas_runtime_tls_enabled"] is True
    assert info["private_blas_backend"] == "upstream_blis"
    assert info["private_openblas_archive_sha256"] == "none"
    for name in (
        "private_blas_archive_sha256",
        "private_blas_source_tree_sha256",
        "private_blas_header_sha256",
        "private_blas_cblas_header_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", str(info[name]))
    assert re.fullmatch(
        r"[0-9a-f]{40}", str(info["private_blas_source_commit"])
    )
    assert len(payload["thread_errors"]) == 1
    assert "owner thread" in str(payload["thread_errors"][0])
    assert "BLIS_NUM_THREADS" in str(payload["environment_error"])
    assert "different thread count" in str(payload["rebaseline_error"])

    nm = shutil.which("nm")
    readelf = shutil.which("readelf")
    if nm is None or readelf is None:
        pytest.skip("nm and readelf are required for private-symbol inspection")
    module_path = Path(str(payload["module_path"])).resolve()
    dynamic_symbols = subprocess.run(
        [nm, "-D", str(module_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    ).stdout
    assert not re.search(
        r"\b(?:cblas_|openblas_|bli_|dgemm_|sgemm_|xerbla_|"
        r"CBLAS_CallFromC\b|RowMajorStrg\b)",
        dynamic_symbols,
    )
    dynamic_section = subprocess.run(
        [readelf, "-d", str(module_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    ).stdout.lower()
    assert not re.search(r"needed.*lib(?:(?:open)?blas|blis)", dynamic_section)


def test_private_blis_manual_ways_are_explicit_and_fixed() -> None:
    payload = _require_blis(
        _run_blis_child(
            f"""
            import json
            from summit import gxeldcore

            initial = dict(gxeldcore.build_info())
            backend = initial.get("private_blas_backend")
            if backend != "upstream_blis":
                print({_RESULT_PREFIX!r} + json.dumps({{"backend": backend}}))
            else:
                configured = int(gxeldcore.configure_blas_threads(2))
                info = dict(gxeldcore.build_info())
                print({_RESULT_PREFIX!r} + json.dumps({{
                    "backend": backend,
                    "configured": configured,
                    "initial_strategy": initial["blas_runtime_thread_strategy"],
                    "initial_ways": initial["blas_runtime_thread_ways"],
                    "info": info,
                }}, sort_keys=True))
            """,
            manual=True,
        )
    )

    expected_ways = {"jc": 2, "pc": 1, "ic": 1, "jr": 1, "ir": 1}
    assert payload["configured"] == 2
    assert payload["initial_strategy"] == "manual"
    assert payload["initial_ways"] == expected_ways
    info = dict(payload["info"])
    assert info["blas_runtime_thread_strategy"] == "manual"
    assert info["blas_runtime_thread_ways"] == expected_ways
    assert info["blas_runtime_owner_thread_configured"] is True
