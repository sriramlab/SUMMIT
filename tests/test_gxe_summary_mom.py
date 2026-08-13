from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from summit.inference import gxe as gxe_module
from summit.inference.gxe import (
    GxEConsumedInputProvenance,
    GxEInputProvenance,
    GxENormalEquations,
    _assemble_deleted_normal_equations,
    _equations_from_prepared_scores,
    _prepare_reference_sufficient_statistics,
    assemble_normal_equations,
    load_fit_batch_manifest,
    solve_normal_equations,
    write_fit,
    write_fits,
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
    _, direct_json = write_fit(tmp_path / "direct-fit", fitted, equations)
    assert json.loads(direct_json.read_text())["consumed_input_provenance"] is None

    record = GxEInputProvenance(
        path=str((tmp_path / "input").resolve()),
        bytes=1,
        sha256="0" * 64,
    )
    malformed = replace(
        fitted,
        consumed_input_provenance=GxEConsumedInputProvenance(
            reference_manifest=record,
            feature_cache=record,
            phenotype_moments=record,
            gwas=replace(record, path="relative/path"),
            gwis=record,
        ),
    )
    with pytest.raises(ValueError, match="canonical absolute path"):
        write_fit(tmp_path / "malformed-provenance", malformed, equations)

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


@pytest.mark.parametrize("null_corrected", [False, True])
def test_preaggregated_deletion_matches_legacy_for_overlapping_annotations(
    null_corrected,
):
    rng = np.random.default_rng(947)
    n, m, k, nblock = 31, 29, 3, 5
    r = 27
    x = rng.normal(size=(n, m))
    w = rng.normal(size=(n, m))
    y = rng.normal(size=n)
    e = rng.normal(size=n)
    annotations = 0.05 + rng.random((m, k))
    # Deliberately interleave blocks; the optimization cannot assume slices.
    blocks = (np.arange(m) * 7 + 3) % nblock
    cross = {
        "xx": (x.T @ x / r) ** 2,
        "xw": (x.T @ w / r) ** 2,
        "wx": (w.T @ x / r) ** 2,
        "ww": (w.T @ w / r) ** 2,
    }
    raw_panels = {key: value @ annotations for key, value in cross.items()}
    masses = annotations.sum(axis=0)
    panels = {
        key: value - masses[None, :] / r if null_corrected else value
        for key, value in raw_panels.items()
    }
    within = {key: np.empty((nblock, k, k)) for key in panels}
    for block_id in range(nblock):
        take = blocks == block_id
        for key, value in cross.items():
            within[key][block_id] = (
                annotations[take].T
                @ value[np.ix_(take, take)]
                @ annotations[take]
            )
    score_x = x.T @ y / np.sqrt(r)
    score_w = w.T @ y / np.sqrt(r)
    norm_x = np.sum(x * x, axis=0) / r
    norm_w = np.sum(w * w, axis=0) / r
    diag_x = np.sum((e[:, None] * x) ** 2, axis=0) / r
    diag_w = np.sum((e[:, None] * w) ** 2, axis=0) / r
    names = ("a", "b", "c")
    full = assemble_normal_equations(
        annotations=annotations,
        score_x=score_x,
        score_w=score_w,
        ld_xx=panels["xx"],
        ld_xw=panels["xw"],
        ld_wx=panels["wx"],
        ld_ww=panels["ww"],
        norm_x=norm_x,
        norm_w=norm_w,
        diag_nxe_x=diag_x,
        diag_nxe_w=diag_w,
        residual_rank=r,
        q_nxe=3.1,
        q_residual=r,
        trace_nxe=25.4,
        trace_nxe_sq=30.2,
        annotation_names=names,
        null_corrected=null_corrected,
    )
    variants = pd.DataFrame(
        {
            "CHR": "1",
            "SNP": [f"rs{index}" for index in range(m)],
            "BP": np.arange(m),
            "A1": "A",
            "A2": "C",
        }
    )
    prepared = _prepare_reference_sufficient_statistics(
        path=Path("reference.json"),
        payload={
            "residual_rank": r,
            "trace_nxe": 25.4,
            "trace_nxe_sq": 30.2,
        },
        manifest_sha256="0" * 64,
        schema_version=2,
        feature_cache_sha256=None,
        reference_provenance=GxEInputProvenance(
            path="/reference.json", bytes=1, sha256="0" * 64
        ),
        feature_cache_provenance=None,
        variants=variants,
        annotations=annotations,
        annotation_names=names,
        panels=panels,
        norm_x=norm_x,
        norm_w=norm_w,
        diag_x=diag_x,
        diag_w=diag_w,
        equations=full,
        block_values=blocks,
        block_labels=tuple(f"block:{index}" for index in range(nblock)),
        within=within,
        null_corrected=null_corrected,
    )
    prepared_full, prepared_deleted = _equations_from_prepared_scores(
        prepared,
        score_x,
        score_w,
        q_nxe=3.1,
        q_residual=r,
    )
    np.testing.assert_allclose(prepared_full.matrix, full.matrix, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(prepared_full.rhs, full.rhs, rtol=3e-15, atol=3e-14)
    for block_id, observed in enumerate(prepared_deleted):
        expected = _assemble_deleted_normal_equations(
            block_id=block_id,
            annotations=annotations,
            blocks=blocks,
            score_x=score_x,
            score_w=score_w,
            panels=panels,
            within=within,
            norm_x=norm_x,
            norm_w=norm_w,
            diag_x=diag_x,
            diag_w=diag_w,
            residual_rank=r,
            q_nxe=3.1,
            q_residual=r,
            trace_nxe=25.4,
            trace_nxe_sq=30.2,
            annotation_names=names,
            null_corrected=null_corrected,
        )
        np.testing.assert_allclose(observed.matrix, expected.matrix, rtol=3e-15, atol=1e-12)
        np.testing.assert_allclose(observed.rhs, expected.rhs, rtol=3e-15, atol=1e-12)
        np.testing.assert_allclose(observed.traces, expected.traces, rtol=3e-15, atol=1e-12)

    block_local = _prepare_reference_sufficient_statistics(
        path=Path("reference.json"),
        payload={
            "residual_rank": r,
            "trace_nxe": 25.4,
            "trace_nxe_sq": 30.2,
        },
        manifest_sha256="0" * 64,
        schema_version=3,
        feature_cache_sha256=None,
        reference_provenance=GxEInputProvenance(
            path="/reference.json", bytes=1, sha256="0" * 64
        ),
        feature_cache_provenance=None,
        variants=variants,
        annotations=annotations,
        annotation_names=names,
        panels=panels,
        norm_x=norm_x,
        norm_w=norm_w,
        diag_x=diag_x,
        diag_w=diag_w,
        equations=full,
        block_values=blocks,
        block_labels=tuple(f"block:{index}" for index in range(nblock)),
        within=None,
        jackknife_method=gxe_module._BLOCK_LOCAL_JACKKNIFE_METHOD,
        null_corrected=null_corrected,
    )
    _, local_deleted = _equations_from_prepared_scores(
        block_local,
        score_x,
        score_w,
        q_nxe=3.1,
        q_residual=r,
    )
    for block_id, observed in enumerate(local_deleted):
        keep = blocks != block_id
        # This is the additive-style approximation: remove completed
        # LD-score rows and all per-variant moments, then renormalize by the
        # annotation mass that remains. Source-side LD from the deleted block
        # is intentionally left in retained rows.
        expected = assemble_normal_equations(
            annotations=annotations[keep],
            score_x=score_x[keep],
            score_w=score_w[keep],
            ld_xx=raw_panels["xx"][keep],
            ld_xw=raw_panels["xw"][keep],
            ld_wx=raw_panels["wx"][keep],
            ld_ww=raw_panels["ww"][keep],
            norm_x=norm_x[keep],
            norm_w=norm_w[keep],
            diag_nxe_x=diag_x[keep],
            diag_nxe_w=diag_w[keep],
            residual_rank=r,
            q_nxe=3.1,
            q_residual=r,
            trace_nxe=25.4,
            trace_nxe_sq=30.2,
            annotation_names=names,
            null_corrected=False,
        )
        np.testing.assert_allclose(observed.matrix, expected.matrix, rtol=3e-15, atol=1e-12)
        np.testing.assert_allclose(observed.rhs, expected.rhs, rtol=3e-15, atol=1e-12)
        np.testing.assert_allclose(observed.traces, expected.traces, rtol=3e-15, atol=1e-12)


def test_fit_batch_manifest_is_strict_and_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "summit.gxe.fit_batch",
                "schema_version": 1,
                "reference": "reference.json",
                "traits": [
                    {
                        "name": "Y1",
                        "moments": "scores/Y1.moments.json",
                        "gwas": "scores/Y1.gwas.gz",
                        "gwis": "scores/Y1.gwis.gz",
                        "out": "fits/Y1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reference, entries = load_fit_batch_manifest(manifest)
    assert reference == (tmp_path / "reference.json").resolve()
    assert entries[0].phenotype_input.gwas_scores == (
        tmp_path / "scores/Y1.gwas.gz"
    ).resolve()
    assert entries[0].output_prefix == (tmp_path / "fits/Y1").resolve()

    payload = json.loads(manifest.read_text())
    payload["schema_version"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_fit_batch_manifest(manifest)

    payload["schema_version"] = 1
    payload["unexpected"] = "field"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_fit_batch_manifest(manifest)

    payload.pop("unexpected")
    duplicate = dict(payload["traits"][0])
    duplicate["name"] = "Y2"
    duplicate["out"] = "fits/./Y1"
    payload["traits"].append(duplicate)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate.*output prefix"):
        load_fit_batch_manifest(manifest)

    payload["traits"][1]["out"] = "other/Y2"
    payload["traits"][1]["moments"] = "scores/Y2.moments.json"
    payload["traits"][1]["gwas"] = "scores/Y2.gwas.gz"
    payload["traits"][1]["gwis"] = "scores/Y2.gwis.gz"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="share one parent"):
        load_fit_batch_manifest(manifest)

    payload["traits"] = payload["traits"][:1]
    occupied = tmp_path / "fits/Y1.gxe.results.tsv"
    occupied.parent.mkdir(parents=True)
    occupied.write_text("existing\n", encoding="utf-8")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        load_fit_batch_manifest(manifest)


@pytest.mark.parametrize("nblock", [3, 101])
def test_block_aggregation_scan_count_is_independent_of_jackknife_blocks(
    monkeypatch, nblock
):
    actual = np.bincount
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(gxe_module.np, "bincount", counted)
    blocks = np.arange(303, dtype=np.int64) % nblock
    values = np.arange(303 * 4, dtype=np.float64).reshape(303, 4)
    observed = gxe_module._block_weighted_sums(blocks, values, nblock)
    assert observed.shape == (nblock, 4)
    assert calls == values.shape[1]


def test_batch_writer_preflights_and_publishes_all_traits_atomically(tmp_path):
    equations, _, _, _ = _exact_fixture()
    fitted = solve_normal_equations(equations, max_condition=1e16)
    outputs = {
        "Y1": (tmp_path / "Y1", fitted, equations),
        "Y2": (tmp_path / "Y2", fitted, equations),
    }
    published = write_fits(outputs)
    assert set(published) == {"Y1", "Y2"}
    for pair in published.values():
        assert all(path.is_file() for path in pair)
        assert all((path.stat().st_mode & 0o777) == 0o600 for path in pair)

    fresh = tmp_path / "fresh"
    competing = tmp_path / "occupied.gxe.fit.json"
    competing.write_text('{"writer":"competitor"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_fits(
            {
                "fresh": (fresh, fitted, equations),
                "occupied": (tmp_path / "occupied", fitted, equations),
            }
        )
    assert not Path(str(fresh) + ".gxe.results.tsv").exists()
    assert not Path(str(fresh) + ".gxe.fit.json").exists()
    assert competing.read_text(encoding="utf-8") == '{"writer":"competitor"}\n'


def test_batch_writer_rolls_back_only_its_inodes_on_publication_race(
    tmp_path, monkeypatch
):
    equations, _, _, _ = _exact_fixture()
    fitted = solve_normal_equations(equations, max_condition=1e16)
    first = tmp_path / "first"
    second = tmp_path / "second"
    competing_path = Path(str(second) + ".gxe.results.tsv")
    competing_bytes = b"competitor\n"
    actual_link = gxe_module.os.link
    calls = 0

    def publish_then_compete(source, target):
        nonlocal calls
        result = actual_link(source, target)
        calls += 1
        if calls == 1:
            competing_path.write_bytes(competing_bytes)
        return result

    monkeypatch.setattr(gxe_module.os, "link", publish_then_compete)
    with pytest.raises(FileExistsError, match="concurrently created"):
        write_fits(
            {
                "first": (first, fitted, equations),
                "second": (second, fitted, equations),
            }
        )
    assert not Path(str(first) + ".gxe.results.tsv").exists()
    assert not Path(str(first) + ".gxe.fit.json").exists()
    assert competing_path.read_bytes() == competing_bytes
    assert not Path(str(second) + ".gxe.fit.json").exists()
    assert not list(tmp_path.glob(".gxe-fit-batch-stage-*"))
