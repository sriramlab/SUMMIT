from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("reference_transfer_simulation.py")
SPEC = importlib.util.spec_from_file_location(
    "reference_transfer_simulation", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
SIMULATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIMULATION)


def test_independent_reference_simulation_reproduces_reported_errors():
    result = SIMULATION.run_simulation()
    observed = result["median_absolute_relative_error_percent"]
    np.testing.assert_allclose(
        observed["same_different_person_transfer"],
        [2.17, 3.14, 2.23],
        rtol=0.0,
        atol=0.005,
    )
    np.testing.assert_allclose(
        observed["naive_squared_residual_rank_scaling"],
        [9.27, 86.46, 23.51],
        rtol=0.0,
        atol=0.005,
    )
