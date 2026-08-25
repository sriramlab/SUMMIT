from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from summit.inference import gxe as gxe_module
from summit.inference.gxe import (
    GxENormalEquations,
    _equations_from_prepared_scores,
    _prepare_reference_sufficient_statistics,
    assemble_normal_equations,
    load_fit_batch_manifest,
    solve_normal_equations,
    transfer_reference_normal_equations,
    write_fit,
    write_fits,
)
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore


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


def test_population_trace_transfer_is_identity_at_the_reference_rank():
    equations, lhs, _, _ = _exact_fixture()
    genetic_count = lhs.shape[0] - 2
    reference_traces = np.array(equations.traces, copy=True)
    reference_traces[:genetic_count] = 37.0
    equations = replace(equations, traces=reference_traces)
    same = np.diag(np.diag(lhs[:genetic_count, :genetic_count]))
    transferred = transfer_reference_normal_equations(
        equations,
        reference_n_samples=41,
        study_n_samples=41,
        reference_residual_rank=37,
        study_residual_rank=37,
        same_individual_products=same,
        genetic_nxe_traces=lhs[:genetic_count, genetic_count],
        q_nxe=4.0,
        q_residual=37.0,
        trace_nxe=8.0,
        trace_nxe_sq=11.0,
    )
    np.testing.assert_allclose(
        transferred.matrix[:genetic_count, :genetic_count],
        equations.matrix[:genetic_count, :genetic_count],
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        transferred.traces[:genetic_count],
        equations.traces[:genetic_count],
        rtol=0.0,
        atol=0.0,
    )
    assert transferred.matrix[-2, -2] == 11.0
    assert transferred.matrix[-2, -1] == 8.0
    assert transferred.matrix[-1, -1] == 37.0


def test_population_trace_transfer_scales_same_and_different_person_terms():
    equations, lhs, _, _ = _exact_fixture()
    genetic_count = lhs.shape[0] - 2
    reference_traces = np.array(equations.traces, copy=True)
    reference_traces[:genetic_count] = 20.0
    equations = replace(equations, traces=reference_traces)
    same = 0.35 * lhs[:genetic_count, :genetic_count]
    genetic_nxe = np.linspace(4.0, 7.0, genetic_count)
    transferred = transfer_reference_normal_equations(
        equations,
        reference_n_samples=30,
        study_n_samples=70,
        reference_residual_rank=20,
        study_residual_rank=50,
        same_individual_products=same,
        genetic_nxe_traces=genetic_nxe,
        q_nxe=7.0,
        q_residual=50.0,
        trace_nxe=12.0,
        trace_nxe_sq=18.0,
    )
    expected = (
        (70.0 / 30.0) * same
        + (70.0 * 69.0 / (30.0 * 29.0))
        * (lhs[:genetic_count, :genetic_count] - same)
    )
    np.testing.assert_allclose(
        transferred.matrix[:genetic_count, :genetic_count], expected
    )
    np.testing.assert_allclose(
        transferred.matrix[:genetic_count, genetic_count], genetic_nxe
    )


def test_population_same_person_probe_u_statistic_matches_dense_target():
    rng = np.random.default_rng(481)
    n, m, probes = 25, 12, 4096
    x = rng.normal(size=(n, m))
    w = rng.normal(size=(n, m))
    z = rng.choice(np.array([-1.0, 1.0]), size=(m, probes))
    sources = np.asfortranarray(np.column_stack([x @ z, w @ z]))
    square_sums = np.zeros((n, 2), dtype=np.float64)
    same_probe = np.zeros((2, 2), dtype=np.float64)
    estimator = SimpleNamespace(
        nsamp=n,
        nbins=1,
        nvecs=probes,
        nsnps_bin=np.array([m], dtype=np.float64),
    )
    GenomewideEnvLDScore._accumulate_population_diagonal_moments(
        estimator,
        sources,
        probes,
        square_sums,
        same_probe,
    )
    observed = GenomewideEnvLDScore._finalize_population_diagonal_moments(
        estimator,
        square_sums,
        same_probe,
    )
    diagonal_x = np.sum(x * x, axis=1) / m
    diagonal_w = np.sum(w * w, axis=1) / m
    expected = np.array(
        [
            [diagonal_x @ diagonal_x, diagonal_x @ diagonal_w],
            [diagonal_w @ diagonal_x, diagonal_w @ diagonal_w],
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=0.025, atol=0.0)


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
        proportions=np.where(
            np.arange(len(fitted.proportions)) == 0,
            np.nan,
            fitted.proportions,
        ),
    )
    table_path, json_path = write_fit(tmp_path / "nonfinite", altered, equations)
    payload = json.loads(json_path.read_text())
    assert table_path.is_file()
    assert payload["condition_number"] is None
    assert payload["proportions"][0] is None
    assert payload["jackknife_method"] == (
        "frozen_full_genome_variant_ldscore_delete_block_v1"
    )

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


def test_posthoc_deletion_drops_only_fixed_target_rows_and_rescales():
    rng = np.random.default_rng(947)
    n, m, k, nblock = 31, 29, 3, 5
    residual_rank = 27
    x = rng.normal(size=(n, m))
    w = rng.normal(size=(n, m))
    y = rng.normal(size=n)
    environment = rng.normal(size=n)
    annotations = 0.05 + rng.random((m, k))
    blocks = (np.arange(m) * 7 + 3) % nblock

    def panel(left, source):
        return ((left.T @ source / residual_rank) ** 2) @ annotations

    panels = {
        "xx": panel(x, x),
        "xw": panel(x, w),
        "wx": panel(w, x),
        "ww": panel(w, w),
    }
    score_x = x.T @ y / np.sqrt(residual_rank)
    score_w = w.T @ y / np.sqrt(residual_rank)
    norm_x = np.sum(x * x, axis=0) / residual_rank
    norm_w = np.sum(w * w, axis=0) / residual_rank
    diag_x = np.sum((environment[:, None] * x) ** 2, axis=0) / residual_rank
    diag_w = np.sum((environment[:, None] * w) ** 2, axis=0) / residual_rank
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
        residual_rank=residual_rank,
        q_nxe=3.1,
        q_residual=residual_rank,
        trace_nxe=25.4,
        trace_nxe_sq=30.2,
        annotation_names=names,
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
            "residual_rank": residual_rank,
            "trace_nxe": 25.4,
            "trace_nxe_sq": 30.2,
        },
        schema_version=4,
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
        block_labels=tuple(f"block_{index}" for index in range(nblock)),
    )
    observed_full, deleted = _equations_from_prepared_scores(
        prepared,
        score_x,
        score_w,
        q_nxe=3.1,
        q_residual=residual_rank,
    )
    np.testing.assert_allclose(observed_full.matrix, full.matrix, rtol=0.0, atol=0.0)
    for block_id, observed in enumerate(deleted):
        keep = blocks != block_id
        expected = assemble_normal_equations(
            annotations=annotations[keep],
            score_x=score_x[keep],
            score_w=score_w[keep],
            ld_xx=panels["xx"][keep],
            ld_xw=panels["xw"][keep],
            ld_wx=panels["wx"][keep],
            ld_ww=panels["ww"][keep],
            norm_x=norm_x[keep],
            norm_w=norm_w[keep],
            diag_nxe_x=diag_x[keep],
            diag_nxe_w=diag_w[keep],
            residual_rank=residual_rank,
            q_nxe=3.1,
            q_residual=residual_rank,
            trace_nxe=25.4,
            trace_nxe_sq=30.2,
            annotation_names=names,
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
