from __future__ import annotations

import math
import os
import types

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _JackknifeSketchStore,
    _build_balanced_vtiles,
    _orthonormalize_columns,
    _secure_memmap,
)
from summit.logger import Logger


def _make_toy(tmp_path, out_name: str):
    rng = np.random.default_rng(194)
    n, m = 29, 11
    raw = rng.binomial(2, rng.uniform(0.15, 0.42, m), size=(n, m)).astype(float)
    prefix = tmp_path / f"geno-{out_name}"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env_raw = rng.normal(size=n)
    env_path = tmp_path / f"{out_name}.env"
    ids.assign(E=env_raw).to_csv(env_path, sep="\t", index=False)
    annot = np.column_stack(
        [0.2 + 0.8 * np.arange(m) / (m - 1), 0.35 + 0.65 * (np.arange(m) % 2)]
    )
    annot_path = tmp_path / f"{out_name}.annot"
    pd.DataFrame(
        {
            "CHR": bim[0].astype(str),
            "SNP": bim[1].astype(str),
            "BP": bim[3].astype(int),
            "a": annot[:, 0],
            "b": annot[:, 1],
        }
    ).to_csv(annot_path, sep="\t", index=False)
    obj = GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(env_path),
        annot_path=str(annot_path),
        out_path=str(tmp_path / out_name),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=m,
        step_size=4,
        seed=17,
        dtype="float64",
        kernel_mode="genie",
        genotype_scale="sample",
        target_xz_mem=0.01,
        write_jackknife=True,
        jackknife_spec="3",
        allow_low_probe_jackknife=True,
        num_threads=1,
    )

    def exact_probes(self, L, v_count, blk_start, v_start):
        eye_rows = np.eye(m)[blk_start:blk_start + L, v_start:v_start + v_count]
        return np.asfortranarray(math.sqrt(m) * eye_rows)

    obj._generate_random_block = types.MethodType(exact_probes, obj)
    return obj, raw, env_raw, annot


def _dense_panels(raw, env_raw, annot, block_ids):
    n, _ = raw.shape
    e = (env_raw - env_raw.mean()) / env_raw.std(ddof=1)
    qfull = _orthonormalize_columns(np.column_stack([np.ones(n), e]))
    pmat = np.eye(n) - qfull @ qfull.T
    g = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    x = pmat @ g
    w = pmat @ (e[:, None] * g)
    rank = n - qfull.shape[1]
    cross = {
        "xx": x.T @ x / rank,
        "xw": x.T @ w / rank,
        "wx": w.T @ x / rank,
        "ww": w.T @ w / rank,
    }
    scores = {key: (value * value) @ annot for key, value in cross.items()}
    within = {
        key: np.zeros((int(block_ids.max()) + 1, annot.shape[1], annot.shape[1]))
        for key in cross
    }
    for block_id in range(int(block_ids.max()) + 1):
        rows = np.flatnonzero(block_ids == block_id)
        for key, value in cross.items():
            square = value[np.ix_(rows, rows)] ** 2
            within[key][block_id] = annot[rows].T @ square @ annot[rows]
    return scores, within


def test_fused_three_pass_matches_dense_and_old_reread_oracle(tmp_path):
    obj, raw, env_raw, annot = _make_toy(tmp_path, "fused")
    original_read = obj._read_genotype_block
    read_count = 0

    def counted_read(self, start, end):
        nonlocal read_count
        read_count += 1
        return original_read(start, end)

    obj._read_genotype_block = types.MethodType(counted_read, obj)
    obj._compute_ldscore()
    assert read_count == 3 * math.ceil(raw.shape[1] / obj.step_size)
    assert not list(tmp_path.glob(".gxe-jackknife-*"))

    dense_scores, dense_within = _dense_panels(raw, env_raw, annot, obj.jackknife_ids)
    suffix = {"xx": "gxx", "xw": "gxe", "wx": "exg", "ww": "gee"}
    for key, stem in suffix.items():
        observed = pd.read_csv(tmp_path / f"fused.{stem}.ldscore.gz", sep="\t")[["a", "b"]].to_numpy()
        # Text score bundles are intentionally written with %.10g.
        np.testing.assert_allclose(observed, dense_scores[key], rtol=1e-9, atol=1e-9)
    saved = np.load(tmp_path / "fused.gxe.jackknife.npz")
    for key in dense_within:
        np.testing.assert_allclose(saved[f"within_{key}"], dense_within[key], rtol=3e-12, atol=3e-12)

    old, _, _, _ = _make_toy(tmp_path, "old")
    old._precompute_residual_variances()
    old_within = old._compute_within_jackknife_scores(
        old._make_compute_blocks(), [(0, raw.shape[1])]
    )
    for key in dense_within:
        np.testing.assert_allclose(saved[f"within_{key}"], old_within[key], rtol=3e-12, atol=3e-12)


def test_jackknife_probe_tiling_accounts_for_global_and_block_workspace():
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.target_xz_mem = 4 * 101 * 3 * 7.5 * 4 / (1024 ** 3)
    obj.dtype = np.float32
    obj.nsamp = 101
    obj.nbins = 3
    obj.nvecs = 23
    obj.jackknife_ids = np.zeros(5, dtype=np.int32)
    obj.log = Logger(suppress=True)
    tiles = obj._auto_vtiles()
    assert max(size for _, size in tiles) == 6
    assert sum(size for _, size in tiles) == 23
    resident = 4 * obj.nsamp * obj.nbins * max(size for _, size in tiles) * 4
    assert resident <= obj.target_xz_mem * (1024 ** 3)


def test_production_shaped_jackknife_tiling_bounds_1024_probe_scratch():
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.target_xz_mem = 32.0
    obj.jackknife_scratch_gib = 64.0
    obj.dtype = np.float64
    obj.nsamp = 300_000
    obj.nbins = 1
    obj.nvecs = 1_024
    obj.jackknife_ids = np.arange(100, dtype=np.int32)
    obj.jackknife_labels = [f"block:{index}" for index in range(100)]
    obj.log = Logger(suppress=True)

    monolithic = 2 * 100 * obj.nsamp * obj.nvecs * 8
    assert monolithic / (1024 ** 3) > 450.0
    tiles = obj._auto_vtiles()
    assert sum(size for _, size in tiles) == obj.nvecs
    assert len(tiles) == 8
    assert {size for _, size in tiles} == {128}
    tiled_scratch = 2 * 100 * obj.nsamp * max(size for _, size in tiles) * 8
    assert tiled_scratch <= obj.jackknife_scratch_gib * (1024 ** 3)
    assert tiled_scratch == 61_440_000_000
    assert tiled_scratch / (1024 ** 3) == pytest.approx(57.220458984375)


def test_native_tiling_accounts_for_opaque_panel_preparation_peak():
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.target_xz_mem = 10 * 101 * 3 * 7.5 * 8 / (1024 ** 3)
    obj.jackknife_scratch_gib = 1.0
    obj.dtype = np.float64
    obj.nsamp = 101
    obj.nbins = 3
    obj.nvecs = 23
    obj.jackknife_ids = np.zeros(5, dtype=np.int32)
    obj.jackknife_labels = ["one"]
    obj.native_backend = "direct"
    obj.log = Logger(suppress=True)
    tiles = obj._auto_vtiles()
    assert max(size for _, size in tiles) == 6
    assert sum(size for _, size in tiles) == 23
    peak = 10 * obj.nsamp * obj.nbins * max(size for _, size in tiles) * 8
    assert peak <= obj.target_xz_mem * 1024**3


def test_balanced_probe_tiling_rejects_infeasible_max_tile_cap():
    with pytest.raises(RuntimeError, match="exceeding max_tiles=8"):
        _build_balanced_vtiles(1_024, vmax=127, gran=64, max_tiles=8)
    tiles = _build_balanced_vtiles(1_024, vmax=128, gran=64, max_tiles=8)
    assert tiles == [(start, 128) for start in range(0, 1_024, 128)]


def test_jackknife_store_maps_one_reserved_block_at_a_time(tmp_path):
    store = _JackknifeSketchStore(
        tmp_path / "store",
        nblocks=3,
        columns=4,
        samples=5,
        dtype=np.float64,
        max_total_bytes=3 * 4 * 5 * 8,
    )
    try:
        assert store.total_bytes == 3 * 4 * 5 * 8
        assert store.peak_mapped_bytes == 4 * 5 * 8
        first = store.view(0)
        first[:] = 2.0
        del first
        second = store.view(1)
        second[:] = 3.0
        del second
        np.testing.assert_array_equal(store.view(0), np.full((5, 4), 2.0))
        assert all((path.stat().st_mode & 0o777) == 0o600 for path in store._paths)
    finally:
        store.close()
    assert not list((tmp_path / "store").glob("*.bin"))

    with pytest.raises(RuntimeError, match="exceeds the configured tile limit"):
        _JackknifeSketchStore(
            tmp_path / "too-small",
            nblocks=3,
            columns=4,
            samples=5,
            dtype=np.float64,
            max_total_bytes=3 * 4 * 5 * 8 - 1,
        )


def test_secure_memmap_and_bed_reader_thread_forwarding(tmp_path):
    path = tmp_path / "private.bin"
    mapping = _secure_memmap(path, (2, 3), np.float32)
    assert (path.stat().st_mode & 0o777) == 0o600
    mapping[:] = 2.0
    mapping.flush()
    mapping._mmap.close()

    calls = []

    class FakeBed:
        def read(self, **kwargs):
            calls.append(kwargs)
            return np.arange(12, dtype=float).reshape(4, 3, order="F")

    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.G = FakeBed()
    obj.row_sel = np.arange(4)
    obj.nsamp = 4
    obj.decode_threads = 3
    obj.genotype_scale = "sample"
    obj.ddof = 1
    obj.eps_var = 1e-10
    observed = obj._read_genotype_block(0, 3)
    assert observed.shape == (4, 3)
    assert calls[0]["num_threads"] == 3


def test_jackknife_scratch_is_removed_after_injected_source_failure(tmp_path):
    obj, _, _, _ = _make_toy(tmp_path, "failure")

    def fail_after_scratch_creation(self, *args, **kwargs):
        raise RuntimeError("injected sketch failure")

    obj._accumulate_sketch_block = types.MethodType(fail_after_scratch_creation, obj)
    with pytest.raises(RuntimeError, match="injected sketch failure"):
        obj._compute_ldscore()
    assert not list(tmp_path.glob(".gxe-jackknife-*"))
    assert not list(tmp_path.glob("failure.g*"))
    assert not list(tmp_path.glob("*.gxe.bundle.lock"))
