"""Numerical and restart regression tests for bounded prediction execution."""
from dataclasses import replace
import json
import os

import numpy as np
import pytest

from prediction_helpers import prediction_threads

from test_prediction_core import fixture, dense
from summit.prediction.api import fit_prediction
from summit.prediction.batch import plan_prediction
from summit.prediction.checkpoint import SolverCheckpoint
from summit.prediction.genotype import StandardizedBlock, native_module, standardize
from summit.prediction.operator import GenotypeOperator
from summit.prediction.runtime import configure_prediction_threads
from summit.prediction.solver import solve
from summit.prediction.spec import SolverSpec


@pytest.mark.parametrize("dtype", [np.int8, np.float64])
@pytest.mark.parametrize("subset", [False, True])
def test_fused_affine_missing_alleles_axes_and_buffer_reuse(dtype, subset):
    rng = np.random.default_rng(43)
    raw = rng.integers(0, 3, (71, 13)).astype(dtype, order="F")
    if dtype == np.float64:
        raw[1::3] *= .317
    raw[::5, ::3] = -127
    raw.setflags(write=False)
    original = raw.copy()
    rows = np.arange(len(raw), dtype=np.int64)
    columns = np.arange(raw.shape[1], dtype=np.int64)
    if subset:
        rows, columns = rows[::-2].copy(), columns[::2].copy()
    mean, inv = rng.uniform(0, 2, len(columns)), rng.uniform(0, 7, len(columns))
    inv[0] = 0
    flips = np.arange(len(columns)) % 2 == 0
    reference = StandardizedBlock(None, 1).prepare(raw, rows, columns, mean, inv, flips)
    block = StandardizedBlock(native_module(), 1)
    actual = block.prepare(raw, rows, columns, mean, inv, flips)
    np.testing.assert_array_equal(actual, reference)
    pointer = actual.ctypes.data
    again = block.prepare(raw, rows[:9], columns[:3], mean[:3], inv[:3], flips[:3])
    assert again.flags.f_contiguous and again.ctypes.data == pointer
    np.testing.assert_array_equal(again, reference[:9, :3])
    np.testing.assert_array_equal(raw, original)


def test_fused_affine_rejects_alias_bad_indices_and_nonfinite():
    raw = np.ones((3, 2), order="F")
    rows, cols = np.arange(3), np.arange(2)
    native = native_module()
    args = [raw, rows, cols, np.zeros(2), np.ones(2), np.zeros(2, dtype=bool), raw, 1]
    with pytest.raises(RuntimeError, match="alias"):
        native.prediction_standardize(*args)
    args[6] = np.empty_like(raw)
    args[1] = np.array([0, 1, 3])
    with pytest.raises(RuntimeError, match="row out of range"):
        native.prediction_standardize(*args)
    args[1] = rows
    raw[1, 1] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        native.prediction_standardize(*args)


def test_combined_verification_preserves_independent_dense_solves():
    source, traits = fixture()
    operator = GenotypeOperator(source, traits, plan_prediction(traits, source, block_size=8), backend="numpy")
    operator.setup()
    result = solve(operator, SolverSpec(rtol=1e-10))
    assert operator.ledger.traversals["cg_and_verification"] > 0
    assert operator.ledger.traversals.get("verification", 0) == 1
    for t in traits:
        for c in t.candidates:
            key = (t.id, c.id)
            expected, *_ = dense(source, t, c.covariance, c.residual)
            np.testing.assert_allclose(result.solutions[key], expected, atol=2e-8, rtol=2e-8)
            assert result.reports[key]["relative_true_residual"] <= 1e-10


def test_failed_true_check_restarts_without_accepting_recursive_convergence(monkeypatch):
    source, traits = fixture()
    t = replace(traits[0], candidates=traits[0].candidates[:1])
    operator = GenotypeOperator(source, [t], plan_prediction([t], source, block_size=8), backend="numpy")
    operator.setup()
    apply = operator.apply
    disturbed = []
    def perturb(vectors, **kw):
        result = apply(vectors, **kw)
        if kw.get("phase") == "verification" and not disturbed:
            key = next(iter(result))
            result[key] += np.linspace(-1, 1, len(result[key]))
            disturbed.append(key)
        return result
    monkeypatch.setattr(operator, "apply", perturb)
    result = solve(operator, SolverSpec(rtol=1e-10))
    key = (t.id, t.candidates[0].id)
    assert disturbed and result.reports[key]["restarts"] >= 1
    expected, *_ = dense(source, t, t.candidates[0].covariance, t.candidates[0].residual)
    np.testing.assert_allclose(result.solutions[key], expected, atol=2e-8, rtol=2e-8)
    assert result.reports[key]["relative_true_residual"] <= 1e-10


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_interrupted_fit_resumes_identical_models_and_rejects_changed_fit(tmp_path, monkeypatch, backend):
    source, traits = fixture()
    spec = SolverSpec(rtol=1e-10)
    plan = plan_prediction(traits, source, storage="compact", block_size=7, rhs_columns=6, threads=prediction_threads())
    expected = fit_prediction(traits, source, output=tmp_path/"expected", plan=plan, solver=spec, backend=backend)
    checkpoint = tmp_path/"solver.npz"
    original = GenotypeOperator.apply
    calls = []
    def interrupt(self, vectors, **kw):
        calls.append(kw.get("phase"))
        if len(calls) == 6:
            raise InterruptedError("simulated scheduler termination mid-pass")
        return original(self, vectors, **kw)
    with monkeypatch.context() as patch:
        patch.setattr(GenotypeOperator, "apply", interrupt)
        with pytest.raises(InterruptedError):
            fit_prediction(traits, source, output=tmp_path/"actual", plan=plan, solver=spec,
                backend=backend, checkpoint=checkpoint)
    assert checkpoint.is_file() and not (tmp_path/"actual").exists()
    assert checkpoint.stat().st_mode & 0o077 == 0
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert json.loads(saved["metadata"].tobytes())["iteration"] == 5
    with monkeypatch.context() as patch:
        patch.setattr(GenotypeOperator, "setup", lambda self: pytest.fail("mismatch must fail before genotype scan"))
        with pytest.raises(ValueError, match="identity mismatch"):
            fit_prediction(traits, source, output=tmp_path/"actual", plan=plan,
                solver=replace(spec, rtol=1e-9), backend=backend, checkpoint=checkpoint, resume=True)
    actual = fit_prediction(traits, source, output=tmp_path/"actual", plan=plan, solver=spec,
        backend=backend, checkpoint=checkpoint, resume=True)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(a.fixed_coefficients, b.fixed_coefficients)
    # A fully solved checkpoint can re-export without another covariance pass.
    with monkeypatch.context() as patch:
        patch.setattr(GenotypeOperator, "apply", lambda *a, **kw: pytest.fail("completed solve repeated"))
        fit_prediction(traits, source, output=tmp_path/"reexport", plan=plan, solver=spec,
            backend=backend, checkpoint=checkpoint, resume=True)


def test_checkpoint_lock_atomicity_and_checksum(tmp_path, monkeypatch):
    path = tmp_path/"state.npz"
    key = ("trait", "candidate")
    vectors = {key: np.ones(3)}
    state = dict(iteration=1, active=[key], pending=[], x=vectors, residual=vectors,
        directions=vectors, fixed_coefficients={}, reports={key: {"converged": False}},
        rho={key: 1.0}, elapsed_seconds=1.0)
    with SolverCheckpoint(path, "test") as checkpoint:
        with pytest.raises(BlockingIOError):
            with SolverCheckpoint(path, "test"):
                pass
        checkpoint.save(state)
        original = path.read_bytes()
        def interrupt(*args):
            raise OSError("simulated interrupted atomic publication")
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", interrupt)
            with pytest.raises(OSError):
                checkpoint.save(state)
        assert path.read_bytes() == original
        assert not list(tmp_path.glob("*.tmp"))
        checkpoint.load(vectors, {key: 2})
    with np.load(path, allow_pickle=False) as data:
        corrupted = {k: data[k] for k in data.files}
    corrupted["a0"][0] += .1
    np.savez(path, **corrupted)
    with SolverCheckpoint(path, "test", resume=True) as checkpoint:
        with pytest.raises(ValueError, match="checksum"):
            checkpoint.load(vectors, {key: 2})


def test_prediction_registers_bound_worker_set_and_rejects_ambiguous_placement(monkeypatch):
    class Native:
        def configure_openmp_placement(self, cpus, threads):
            assert cpus == [4, 6] and threads == 2
            self.configured = True
        def configure_blas_threads(self, threads):
            assert self.configured and threads == 2
    monkeypatch.setenv("OMP_PROC_BIND", "SPREAD")
    monkeypatch.setenv("OMP_PLACES", "{4},{6}")
    configure_prediction_threads(Native(), 2)
    monkeypatch.setenv("OMP_PLACES", "cores")
    with pytest.raises(RuntimeError, match="singleton"):
        configure_prediction_threads(Native(), 2)


@pytest.mark.parametrize("storage", ["stream", "compact", "standardized"])
def test_wide_rhs_tiles_match_narrow_independent_gls(storage):
    source, traits = fixture(m=137)
    native = GenotypeOperator(source, traits, plan_prediction(traits, source,
        storage=storage, block_size=67, rhs_columns=160, threads=prediction_threads()))
    native.setup()
    result = solve(native, SolverSpec(rtol=1e-10))
    for t in traits:
        for c in t.candidates:
            expected, *_ = dense(source, t, c.covariance, c.residual)
            np.testing.assert_allclose(result.solutions[(t.id, c.id)], expected, atol=2e-8, rtol=2e-8)
    native.release()
