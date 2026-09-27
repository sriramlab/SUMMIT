import importlib.util
from pathlib import Path
import sys

import numpy as np


def test_shared_control_generator_preserves_marginal_sampling(monkeypatch):
    root = Path(__file__).parents[1]/"scripts/pcgc"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("pcgc_cross_simulation", root/"cross_simulation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.generate(71351, "BB_shared", n=20000, m=80)
    shared, l, r = np.intersect1d(data["left_rows"], data["right_rows"], return_indices=True)
    assert len(shared) == 5000
    assert np.all(data["left"].z[l] < 0) and np.all(data["right"].z[r] < 0)
    assert len(np.union1d(data["left_rows"], data["right_rows"])) == len(data["x"])
    for risk in (data["left"], data["right"]):
        plus = risk.population_risk == risk.population_risk.max()
        high_risk = risk.population_risk.max()
        for case in (True, False):
            selected = (risk.z > 0) == case
            target = .5*high_risk/.1 if case else .5*(1-high_risk)/.9
            assert abs(plus[selected].mean()-target) < 4*np.sqrt(target*(1-target)/selected.sum())
    cells = np.array(data["diagnostics"]["observed_population_cells"])
    target = np.array(data["diagnostics"]["population_cell_probabilities"])
    assert np.all(np.abs(cells/cells.sum()-target) < 5*np.sqrt(target*(1-target)/cells.sum()))
