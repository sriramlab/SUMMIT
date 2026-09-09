import numpy as np
import pytest

from summit.prediction.selection import CalibrationCandidate, select_and_calibrate, load_mean_calibration
from summit.prediction.artifacts import write_json
from summit.prediction.calibration import CalPredIntervals, paired_r2_gain


def test_nested_selection_fold_local_scaling_and_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 3))
    x[:, 0] *= 10000
    y = .0002*x[:, 0] + .3*x[:, 1] + rng.normal(size=120)*.1
    # Keep an explicitly unpenalized backbone and penalized response addition.
    candidates = [CalibrationCandidate(f"ridge{j}", x, ("pgs", "response", "noise"),
        (False, True, True), penalty, {"models": ["frozen"]}) for j, penalty in enumerate([0, .01, 10])]
    samples = [("f", str(i)) for i in range(len(y))]
    fit, report, nested = select_and_calibrate(y, candidates, samples, role="pilot", folds=4, outer_folds=5, seed=32)
    assert report["nested_mse"] < .04
    assert report["selected"] != "ridge2"
    prediction = fit.predict(x, names=candidates[0].names)
    write_json(tmp_path/"calibration.json", fit.to_dict())
    loaded = load_mean_calibration(tmp_path/"calibration.json")
    np.testing.assert_array_equal(prediction, loaded.predict(x, names=candidates[0].names))
    with pytest.raises(ValueError, match="overlap"):
        select_and_calibrate(y, candidates, samples, role="pilot", discovery_samples=samples[:1])
    with pytest.raises(ValueError, match="role"):
        select_and_calibrate(y, candidates, samples, role="replication")
    with pytest.raises(ValueError, match="order"):
        fit.predict(x, names=("response", "pgs", "noise"))


def test_intervals_and_paired_uncertainty_targets():
    rng = np.random.default_rng(3)
    x = np.column_stack([np.ones(1000), rng.normal(size=1000)])
    z = np.ones((1000, 1))
    fit = CalPredIntervals(np.array([0., .8]), np.array([np.log(.5)]), ("intercept", "pgs"), ("intercept",), {})
    mu, lower, upper = fit.predict(x, z, mean_names=fit.mean_names, variance_names=fit.variance_names)
    y = mu + rng.normal(size=len(mu))*np.sqrt(.5)
    assert .86 < np.mean((y >= lower) & (y <= upper)) < .94
    comparison = paired_r2_gain(y, np.zeros(len(y)), mu, bootstrap=100, seed=2)
    assert comparison["delta_r2"] > 0
    assert comparison["interval"][0] > 0
