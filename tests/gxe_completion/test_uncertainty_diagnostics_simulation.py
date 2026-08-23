from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("uncertainty_diagnostics_simulation.py")
SPEC = importlib.util.spec_from_file_location(
    "uncertainty_diagnostics_simulation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
SIMULATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIMULATION)


def test_uncertainty_diagnostics_are_finite_and_expose_known_tradeoffs() -> None:
    result = SIMULATION.run(repetitions=80)
    probe = result["probe_estimators"]
    assert probe["truth_minimum_eigenvalue"] >= -1.0e-10
    for case in probe["cases"].values():
        for metrics in case.values():
            assert all(np.isfinite(value) for value in metrics.values())
        assert case["plugin"]["non_psd_fraction"] == 0.0
    deletion = result["delete_block_same_person_diagonal"]
    assert deletion["maximum_relative_frobenius_error"] > 0.0
    assert deletion["maximum_relative_frobenius_error"] >= deletion[
        "median_relative_frobenius_error"
    ]
    calibration = result["block_local_jackknife_calibration"]
    assert calibration["blocks"] == 100
    assert calibration["limited_calibration_gate"]["pass"] is True
    assert 0.03 <= calibration["gxe_null"][
        "gxe_null_two_sided_type_i_error"
    ] <= 0.07
    for scenario in (calibration["moderate_gxe"], calibration["gxe_null"]):
        for field in (
            "empirical_sd",
            "exact_mean_reported_se_over_empirical_sd",
            "approximate_mean_reported_se_over_empirical_sd",
            "exact_95pct_coverage",
            "approximate_95pct_coverage",
            "approximate_over_exact_se_ratio_median",
        ):
            assert np.all(np.isfinite(scenario[field]))
