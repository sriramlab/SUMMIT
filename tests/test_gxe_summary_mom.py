from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from summit.inference import gxe as gxe_module
from summit.inference.gxe import (
    GxENormalEquations,
    _assemble_deleted_normal_equations,
    assemble_normal_equations,
    solve_normal_equations,
    write_fit,
)


def _projector(design: np.ndarray) -> tuple[np.ndarray, int]:
    u, s, _ = np.linalg.svd(np.asarray(design, dtype=np.float64), full_matrices=False)
    tol = max(design.shape) * np.finfo(np.float64).eps * s[0]
    q = u[:, s > tol]
    return np.eye(design.shape[0]) - q @ q.T, design.shape[0] - q.shape[1]


def _exact_fixture(seed: int = 90210):
    rng = np.random.default_rng(seed)
    n, m = 41, 19
    e = rng.normal(size=n)
    e = (e - e.mean()) / e.std(ddof=1)
    c = rng.normal(size=n)
    # Include a deliberately duplicated fixed effect to exercise rank handling
    # in the independent oracle construction.
    design = np.column_stack([np.ones(n), e, c, c])
    pmat, rank_p = _projector(design)

    x0 = rng.binomial(2, rng.uniform(0.1, 0.45, size=m), size=(n, m)).astype(float)
    x0 -= x0.mean(axis=0)
    x0 /= x0.std(axis=0, ddof=1)
    x = pmat @ x0
    w = pmat @ (e[:, None] * x0)
    y = pmat @ rng.normal(size=n)
    y *= np.sqrt(rank_p / np.dot(y, y))

    annot = np.column_stack(
        [
            np.linspace(0.2, 1.0, m),
            0.15 + (np.arange(m) % 3) / 3.0,
        ]
    )
    masses = annot.sum(axis=0)
    d = np.diag(e * e)
    hn = pmat @ d @ pmat
    kernels = []
    for a in range(annot.shape[1]):
        kernels.append((x * annot[:, a]) @ x.T / masses[a])
    for a in range(annot.shape[1]):
        kernels.append((w * annot[:, a]) @ w.T / masses[a])
    kernels.extend([hn, pmat])

    direct_lhs = np.asarray([[np.trace(a @ b) for b in kernels] for a in kernels])
    direct_rhs = np.asarray([y @ a @ y for a in kernels])
    direct_traces = np.asarray([np.trace(a) for a in kernels])

    def panel(left: np.ndarray, source: np.ndarray) -> np.ndarray:
        return ((left.T @ source) / rank_p) ** 2 @ annot

    equations = assemble_normal_equations(
        annotations=annot,
        score_x=x.T @ y / np.sqrt(rank_p),
        score_w=w.T @ y / np.sqrt(rank_p),
        ld_xx=panel(x, x),
        ld_xw=panel(x, w),
        ld_wx=panel(w, x),
        ld_ww=panel(w, w),
        norm_x=np.sum(x * x, axis=0) / rank_p,
        norm_w=np.sum(w * w, axis=0) / rank_p,
        diag_nxe_x=np.sum((e[:, None] * x) ** 2, axis=0) / rank_p,
        diag_nxe_w=np.sum((e[:, None] * w) ** 2, axis=0) / rank_p,
        residual_rank=rank_p,
        q_nxe=float(np.sum(e * e * y * y)),
        q_residual=float(y @ y),
        trace_nxe=float(np.trace(hn)),
        trace_nxe_sq=float(np.trace(hn @ hn)),
        annotation_names=("fractional", "overlap"),
    )
    return equations, direct_lhs, direct_rhs, direct_traces


def test_summary_equations_equal_explicit_genie_kernels():
    equations, lhs, rhs, traces = _exact_fixture()
    np.testing.assert_allclose(equations.matrix, lhs, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(equations.rhs, rhs, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(equations.traces, traces, rtol=2e-13, atol=2e-13)

    direct = np.linalg.solve(lhs, rhs)
    fitted = solve_normal_equations(equations, max_condition=1e16)
    np.testing.assert_allclose(fitted.coefficients, direct, rtol=2e-12, atol=2e-12)


def test_directional_cross_panels_are_not_interchangeable_per_snp():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(23, 11))
    w = rng.normal(size=(23, 11))
    a = np.ones((11, 1))
    xw = (x.T @ w) ** 2 @ a
    wx = (w.T @ x) ** 2 @ a
    assert np.max(np.abs(xw - wx)) > 1.0
    np.testing.assert_allclose(xw.sum(), wx.sum(), rtol=0.0, atol=1e-10)


def test_balanced_binary_environment_detects_nxe_residual_aliasing():
    rng = np.random.default_rng(1)
    n, m = 20, 9
    e = np.tile(np.array([-1.0, 1.0]), n // 2)
    pmat, r = _projector(np.column_stack([np.ones(n), e]))
    x0 = rng.normal(size=(n, m))
    x = pmat @ x0
    w = pmat @ (e[:, None] * x0)
    y = pmat @ rng.normal(size=n)
    y *= np.sqrt(r / np.dot(y, y))
    annot = np.ones((m, 1))

    def panel(left, source):
        return ((left.T @ source) / r) ** 2 @ annot

    equations = assemble_normal_equations(
        annotations=annot,
        score_x=x.T @ y / np.sqrt(r),
        score_w=w.T @ y / np.sqrt(r),
        ld_xx=panel(x, x),
        ld_xw=panel(x, w),
        ld_wx=panel(w, x),
        ld_ww=panel(w, w),
        norm_x=np.sum(x * x, axis=0) / r,
        norm_w=np.sum(w * w, axis=0) / r,
        diag_nxe_x=np.sum((e[:, None] * x) ** 2, axis=0) / r,
        diag_nxe_w=np.sum((e[:, None] * w) ** 2, axis=0) / r,
        residual_rank=r,
        q_nxe=float(np.sum(e * e * y * y)),
        q_residual=float(y @ y),
        trace_nxe=float(r),
        trace_nxe_sq=float(r),
    )
    with pytest.raises(ValueError, match="not identifiable"):
        solve_normal_equations(equations)


def test_solver_rejects_materially_non_psd_trace_matrix():
    equations = GxENormalEquations(
        matrix=np.array([[1.0, 2.0], [2.0, 1.0]]),
        rhs=np.array([1.0, 1.0]),
        traces=np.array([1.0, 1.0]),
        component_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="not positive semidefinite"):
        solve_normal_equations(equations, allow_ill_conditioned=True)
    with pytest.raises(ValueError, match="max_condition"):
        solve_normal_equations(equations, max_condition=np.nan)
    with pytest.raises(ValueError, match="rcond"):
        solve_normal_equations(equations, rcond=0.0)


def test_fit_json_serializes_nonfinite_diagnostics_as_null(tmp_path):
    equations, _, _, _ = _exact_fixture()
    fitted = solve_normal_equations(equations, max_condition=1e16)
    altered = replace(
        fitted,
        condition_number=np.inf,
        proportions=np.where(np.arange(len(fitted.proportions)) == 0, np.nan, fitted.proportions),
    )
    _, json_path = write_fit(tmp_path / "nonfinite", altered, equations)
    payload = json.loads(json_path.read_text())
    assert payload["condition_number"] is None
    assert payload["proportions"][0] is None


def test_fit_publication_rollback_preserves_competing_json(tmp_path, monkeypatch):
    equations, _, _, _ = _exact_fixture()
    fitted = solve_normal_equations(equations, max_condition=1e16)
    prefix = tmp_path / "race-fit"
    table = tmp_path / "race-fit.gxe.results.tsv"
    manifest = tmp_path / "race-fit.gxe.fit.json"
    competitor = b'{"writer":"competitor"}\n'
    actual_link = gxe_module.os.link
    calls = 0

    def publish_table_then_compete(source, target):
        nonlocal calls
        result = actual_link(source, target)
        calls += 1
        if calls == 1:
            manifest.write_bytes(competitor)
        return result

    monkeypatch.setattr(gxe_module.os, "link", publish_table_then_compete)
    with pytest.raises(FileExistsError, match="concurrently created"):
        write_fit(prefix, fitted, equations)
    assert not table.exists()
    assert manifest.read_bytes() == competitor
    assert not list(tmp_path.glob(".gxe-fit-stage-*"))


def test_fit_does_not_path_chmod_published_output(tmp_path, monkeypatch):
    equations, _, _, _ = _exact_fixture()
    fitted = solve_normal_equations(equations, max_condition=1e16)
    prefix = tmp_path / "chmod-fit"
    actual_chmod = gxe_module.os.chmod

    def reject_final_table_chmod(path, mode):
        if str(path).endswith("chmod-fit.gxe.results.tsv"):
            raise AssertionError("published path must inherit private staged mode")
        return actual_chmod(path, mode)

    monkeypatch.setattr(gxe_module.os, "chmod", reject_final_table_chmod)
    table, manifest = write_fit(prefix, fitted, equations)
    assert (table.stat().st_mode & 0o777) == 0o600
    assert (manifest.stat().st_mode & 0o777) == 0o600
    assert not list(tmp_path.glob(".gxe-fit-stage-*"))


def test_null_corrected_storage_offset_is_exactly_reversed():
    equations, lhs, _, _ = _exact_fixture()
    # Reconstruct the fixture and subtract the documented source mass/r offset
    # from every row of every panel.
    rng = np.random.default_rng(90210)
    n, m = 41, 19
    e = rng.normal(size=n)
    e = (e - e.mean()) / e.std(ddof=1)
    c = rng.normal(size=n)
    pmat, r = _projector(np.column_stack([np.ones(n), e, c, c]))
    x0 = rng.binomial(2, rng.uniform(0.1, 0.45, size=m), size=(n, m)).astype(float)
    x0 -= x0.mean(axis=0)
    x0 /= x0.std(axis=0, ddof=1)
    x = pmat @ x0
    w = pmat @ (e[:, None] * x0)
    y = pmat @ rng.normal(size=n)
    y *= np.sqrt(r / np.dot(y, y))
    annot = np.column_stack([np.linspace(0.2, 1.0, m), 0.15 + (np.arange(m) % 3) / 3.0])
    masses = annot.sum(axis=0)

    def panel(left, source):
        return ((left.T @ source) / r) ** 2 @ annot - masses[None, :] / r

    hn = pmat @ np.diag(e * e) @ pmat
    corrected = assemble_normal_equations(
        annotations=annot,
        score_x=x.T @ y / np.sqrt(r),
        score_w=w.T @ y / np.sqrt(r),
        ld_xx=panel(x, x),
        ld_xw=panel(x, w),
        ld_wx=panel(w, x),
        ld_ww=panel(w, w),
        norm_x=np.sum(x * x, axis=0) / r,
        norm_w=np.sum(w * w, axis=0) / r,
        diag_nxe_x=np.sum((e[:, None] * x) ** 2, axis=0) / r,
        diag_nxe_w=np.sum((e[:, None] * w) ** 2, axis=0) / r,
        residual_rank=r,
        q_nxe=float(np.sum(e * e * y * y)),
        q_residual=float(y @ y),
        trace_nxe=float(np.trace(hn)),
        trace_nxe_sq=float(np.trace(hn @ hn)),
        annotation_names=("fractional", "overlap"),
        null_corrected=True,
    )
    np.testing.assert_allclose(corrected.matrix, lhs, rtol=3e-13, atol=3e-13)


def test_two_sided_block_deletion_matches_explicit_deleted_kernels():
    rng = np.random.default_rng(77)
    n, m = 34, 18
    e = rng.normal(size=n)
    e = (e - e.mean()) / e.std(ddof=1)
    pmat, r = _projector(np.column_stack([np.ones(n), e, rng.normal(size=n)]))
    x0 = rng.normal(size=(n, m))
    x = pmat @ x0
    w = pmat @ (e[:, None] * x0)
    y = pmat @ rng.normal(size=n)
    annot = np.column_stack([0.2 + rng.random(m), 0.1 + (np.arange(m) % 4) / 4.0])
    blocks = np.repeat(np.arange(3), m // 3)

    def panel(left, source):
        return ((left.T @ source) / r) ** 2 @ annot

    panels = {
        "xx": panel(x, x),
        "xw": panel(x, w),
        "wx": panel(w, x),
        "ww": panel(w, w),
    }
    within = {key: np.zeros((3, 2, 2)) for key in panels}
    for block_id in range(3):
        take = blocks == block_id
        a = annot[take]
        for key, (left, source) in {
            "xx": (x, x),
            "xw": (x, w),
            "wx": (w, x),
            "ww": (w, w),
        }.items():
            within[key][block_id] = a.T @ (((left[:, take].T @ source[:, take]) / r) ** 2) @ a

    hn = pmat @ np.diag(e * e) @ pmat
    for block_id in range(3):
        deleted = _assemble_deleted_normal_equations(
            block_id=block_id,
            annotations=annot,
            blocks=blocks,
            score_x=x.T @ y / np.sqrt(r),
            score_w=w.T @ y / np.sqrt(r),
            panels=panels,
            within=within,
            norm_x=np.sum(x * x, axis=0) / r,
            norm_w=np.sum(w * w, axis=0) / r,
            diag_x=np.sum((e[:, None] * x) ** 2, axis=0) / r,
            diag_w=np.sum((e[:, None] * w) ** 2, axis=0) / r,
            residual_rank=r,
            q_nxe=float(y @ hn @ y),
            q_residual=float(y @ y),
            trace_nxe=float(np.trace(hn)),
            trace_nxe_sq=float(np.trace(hn @ hn)),
            annotation_names=("a", "b"),
            null_corrected=False,
        )
        keep = blocks != block_id
        masses = annot[keep].sum(axis=0)
        kernels = []
        for a in range(2):
            kernels.append((x[:, keep] * annot[keep, a]) @ x[:, keep].T / masses[a])
        for a in range(2):
            kernels.append((w[:, keep] * annot[keep, a]) @ w[:, keep].T / masses[a])
        kernels.extend([hn, pmat])
        expected_lhs = np.asarray([[np.trace(a @ b) for b in kernels] for a in kernels])
        expected_rhs = np.asarray([y @ a @ y for a in kernels])
        np.testing.assert_allclose(deleted.matrix, expected_lhs, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(deleted.rhs, expected_rhs, rtol=3e-13, atol=3e-13)
