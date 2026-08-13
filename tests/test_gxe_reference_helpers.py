from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _build_balanced_vtiles,
    _orthonormalize_columns,
    _stable_center_and_scale,
    _validate_jackknife_probe_count,
    read_env_and_cov,
)
from summit.logger import Logger


def test_svd_projector_preserves_rank_deficient_design_span():
    rng = np.random.default_rng(4)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    design = np.column_stack([a, a, b, 2.0 * a - b])
    q = _orthonormalize_columns(design)
    assert q.shape == (50, 2)
    residual = design - q @ (q.T @ design)
    assert np.linalg.norm(residual) / np.linalg.norm(design) < 1e-13
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=2e-14)


def test_fixed_effect_standardization_uses_deterministic_reductions():
    values = np.asarray(
        [1.0e8, -3.0, 2.0, 7.5, -1.0e8, 11.25, 4.0], dtype=np.float64
    )
    centered, mean, scale = _stable_center_and_scale(values, ddof=1)
    # Use the implementation's explicit mathematical contract rather than a
    # NumPy reduction whose order may vary across releases.
    expected_mean = math.fsum(float(value) for value in values) / values.size
    expected_centered = values - expected_mean
    expected_scale = math.sqrt(
        math.fsum(float(value) * float(value) for value in expected_centered)
        / (values.size - 1)
    )
    assert mean == expected_mean
    assert scale == expected_scale
    np.testing.assert_array_equal(centered, expected_centered)


def test_probe_tiles_never_exceed_memory_derived_cap():
    for total, cap in ((1000, 64), (2000, 64), (101, 17), (3, 8)):
        required_tiles = (total + cap - 1) // cap
        tiles = _build_balanced_vtiles(
            total, cap, max_tiles=max(8, required_tiles)
        )
        assert sum(size for _, size in tiles) == total
        assert max(size for _, size in tiles) <= cap
        assert [start for start, _ in tiles] == list(
            np.cumsum([0] + [size for _, size in tiles[:-1]])
        )


def test_low_probe_jackknife_requires_explicit_diagnostic_override():
    with pytest.raises(ValueError, match="at least 100"):
        _validate_jackknife_probe_count(99, True, False)
    _validate_jackknife_probe_count(99, True, True)
    _validate_jackknife_probe_count(100, True, False)


def test_single_fractional_annotation_is_applied():
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.nbins = 1
    source = np.arange(12, dtype=float).reshape(4, 3)
    probes = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]])
    annot = np.array([[0.25], [1.0], [0.0]])
    observed = np.zeros((4, 2), dtype=float)
    obj._accumulate_sketch_block(observed, source, probes, annot)
    expected = (source * np.sqrt(annot[:, 0])[None, :]) @ probes
    np.testing.assert_allclose(observed, expected)


def test_complete_case_reader_treats_minus9_as_missing_and_deduplicates_rank(tmp_path):
    n = 10
    ids = [f"id{i}" for i in range(n)]
    fam = tmp_path / "x.fam"
    env = tmp_path / "x.env"
    cov = tmp_path / "x.cov"
    phen = tmp_path / "x.pheno"
    pd.DataFrame({0: ids, 1: ids, 2: 0, 3: 0, 4: 0, 5: -9}).to_csv(
        fam, sep="\t", header=False, index=False
    )
    e = np.linspace(-2.0, 2.0, n).astype(object)
    e[3] = "-9"
    pd.DataFrame({"FID": ids, "IID": ids, "E": e}).to_csv(env, sep="\t", index=False)
    c = np.linspace(0.0, 1.0, n) ** 2
    pd.DataFrame({"FID": ids, "IID": ids, "C1": c, "Cdup": c}).to_csv(
        cov, sep="\t", index=False
    )
    y = np.sin(np.arange(n, dtype=float)).astype(object)
    y[7] = "-9"
    pd.DataFrame({"FID": ids, "IID": ids, "Y": y}).to_csv(phen, sep="\t", index=False)

    evec, _, q, _, keep, cols, yvec, name, residual_fraction, env_transform = read_env_and_cov(
        str(env), str(fam), str(cov), pheno_filename=str(phen)
    )
    assert len(keep) == n - 2
    assert 3 not in keep and 7 not in keep
    assert q.shape[1] == 2  # one covariate span plus E, despite duplicated C
    assert cols == ["C1", "Cdup"]
    assert name == "Y"
    assert abs(evec.mean()) < 1e-14
    assert abs(yvec.mean()) < 1e-14
    assert np.linalg.norm(q.T @ yvec) < 1e-13
    assert np.isclose(yvec @ yvec, len(keep) - q.shape[1] - 1)
    assert 0.0 < residual_fraction <= 1.0 + 1e-12
    assert env_transform["standardized"] is True
    assert env_transform["units"] == "per_environment_sd"


def test_nxe_trace_formula_matches_explicit_projected_kernel():
    rng = np.random.default_rng(9)
    n = 31
    c = rng.normal(size=(n, 3))
    c -= c.mean(axis=0)
    q = _orthonormalize_columns(c)
    e = rng.normal(size=n)
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.nsamp = n
    obj.C_int = q
    obj.env = e
    trace_d, trace_d2 = obj._nxe_reference_traces()
    qfull = _orthonormalize_columns(np.column_stack([np.ones(n), q]))
    pmat = np.eye(n) - qfull @ qfull.T
    hn = pmat @ np.diag(e * e) @ pmat
    np.testing.assert_allclose(trace_d, np.trace(hn), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(trace_d2, np.trace(hn @ hn), rtol=2e-14, atol=2e-14)


def test_jackknife_preflight_rejects_one_block_and_empty_annotation_deletion():
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.nsnps = 4
    obj.nbins = 1
    obj.l2cols = ["a"]
    obj.log = Logger(suppress=True)
    obj.snplist = pd.DataFrame({"CHR": ["1"] * 4})
    obj.annot = np.ones((4, 1))
    obj.nsnps_bin = np.array([4.0])
    with pytest.raises(ValueError, match="at least two"):
        obj._build_jackknife_blocks("chr")

    obj.snplist = pd.DataFrame({"CHR": ["1", "1", "2", "2"]})
    obj.annot = np.array([[1.0], [1.0], [0.0], [0.0]])
    obj.nsnps_bin = np.array([2.0])
    with pytest.raises(ValueError, match="empty an annotation"):
        obj._build_jackknife_blocks("2")
