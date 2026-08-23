from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_matched_population_transport_beats_blind_scaling_and_converges() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/context/validate_reference_transport.py"),
            "--replicates",
            "60",
            "--study-n",
            "128",
            "--reference-sizes",
            "64",
            "256",
            "--m",
            "24",
            "--seed",
            "20260819",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["contains_row_data"] is False
    assert {setting["name"] for setting in payload["settings"]} == {
        "balanced_independent",
        "balanced_genotype_context_correlation",
        "skewed_context",
    }
    for setting in payload["settings"]:
        assert setting["large_over_small_correct_target_rmse_ratio"] < 0.75
        for result in setting["results"]:
            assert result["correct_has_smaller_bias"] is True
            assert result["blind_over_correct_bias_ratio"] > 2.0
