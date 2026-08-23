from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_independent_reference_fit_converges_toward_exact_study_fit(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/context/validate_fit_reference_convergence.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""
    payload = json.loads(
        (tmp_path / "04_fit_reference_convergence.json").read_text(encoding="utf-8")
    )
    assert payload["contains_row_data"] is False
    assert payload["convergence"]["largest_reference_has_smaller_genetic_vector_rms"]
    assert (
        payload["convergence"][
            "largest_over_smallest_reference_genetic_vector_rms_ratio"
        ]
        < 0.9
    )
    small, large = payload["reference_results"]
    small_rms = small["transferred_minus_exact_same_phenotype"]["coefficient_rms"]
    large_rms = large["transferred_minus_exact_same_phenotype"]["coefficient_rms"]
    assert all(later < earlier for earlier, later in zip(small_rms, large_rms))
    for diagnostics in (
        payload["fit_diagnostics"]["exact_in_study_t"],
        *(result["fit_diagnostics"] for result in payload["reference_results"]),
    ):
        assert diagnostics["ranks_observed"] == [4]
        assert diagnostics["maximum_relative_solve_residual"] < 1.0e-10
