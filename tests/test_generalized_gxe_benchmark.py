from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.benchmark_generalized_gxe_variant import _validate_clean_ledger


def _clean_ledger() -> dict[str, int]:
    return {
        "observed_reference_genotype_passes": 2,
        "observed_retained_variant_visits": 20,
        "duplicate_retained_variant_visits": 0,
        "retry_count": 0,
        "repair_count": 0,
        "fallback_count": 0,
        "integrity_failures": 0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duplicate_retained_variant_visits", 1),
        ("retry_count", 1),
        ("repair_count", 1),
        ("fallback_count", 1),
        ("integrity_failures", 1),
    ),
)
def test_benchmark_rejects_every_nonzero_clean_run_counter(
    field: str, value: int
) -> None:
    ledger = _clean_ledger()
    ledger[field] = value
    with pytest.raises(RuntimeError, match="clean two-pass ledger"):
        _validate_clean_ledger(ledger, variants=10)


def test_benchmark_accepts_clean_two_pass_ledger() -> None:
    _validate_clean_ledger(_clean_ledger(), variants=10)


@pytest.mark.parametrize(
    ("basis", "annotations", "probes"),
    ((3, 1, 128), (3, 1, 1024), (4, 8, 128)),
)
def test_primary_shape_benchmark_plan_is_an_exact_two_pass_dry_run(
    basis: int,
    annotations: int,
    probes: int,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_generalized_gxe_variant.py"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(pytest.__file__).resolve().parent.parent),
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "plan",
            "--samples",
            "300000",
            "--variants",
            "1000000",
            "--basis",
            str(basis),
            "--annotations",
            str(annotations),
            "--probes",
            str(probes),
            "--threads",
            "64",
            "--variant-block-width",
            "4096",
            "--probe-tile-width",
            "4",
            "--memory-gib",
            "900",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema"] == (
        "summit.generalized_gxe.variant_benchmark_plan.v1"
    )
    plan = payload["plan"]
    assert plan["dimensions"] == {
        "N": 300000,
        "M": 1000000,
        "Q": basis,
        "K": annotations,
        "B": probes,
        "P": basis * (basis + 1) // 2,
        "C": annotations * basis * (basis + 1) // 2,
    }
    assert plan["descriptor"]["planned_complete_passes"] == 2
    assert plan["descriptor"]["planned_variant_record_visits"] == 2000000
    assert plan["ledger"]["planned_reference_genotype_passes"] == 2
    assert plan["ledger"]["planned_retained_variant_visits"] == 2000000
    assert plan["tiling"]["rhs_tile_columns"] == basis**2 * 4
    assert plan["peak_resident_bytes"] < plan["memory_limit_bytes"]
