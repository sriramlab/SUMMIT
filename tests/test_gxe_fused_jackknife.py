from __future__ import annotations

import math
import types

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _build_balanced_vtiles,
    _orthonormalize_columns,
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
        num_threads=1,
    )

    def exact_probes(self, L, v_count, blk_start, v_start):
        eye_rows = np.eye(m)[blk_start:blk_start + L, v_start:v_start + v_count]
        return np.asfortranarray(math.sqrt(m) * eye_rows)

    obj._generate_random_block = types.MethodType(exact_probes, obj)
    return obj, raw, env_raw, annot


def _dense_panels(raw, env_raw, annot):
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
    return scores


def test_xw_wx_are_transposed_aggregates_not_duplicate_variant_scores():
    rng = np.random.default_rng(903)
    x = rng.normal(size=(19, 7))
    w = rng.normal(size=(19, 7))
    annotations = rng.uniform(0.1, 1.0, size=(7, 3))
    cross = x.T @ w
    xw = (cross * cross) @ annotations
    wx = (cross.T * cross.T) @ annotations

    # A row norm and the corresponding column norm need not agree for a SNP.
    assert not np.allclose(xw, wx, rtol=1e-12, atol=1e-12)
    # The normal-equation aggregate is the same only after transposing the
    # directional annotation indices.  This identity does not reconstruct one
    # per-variant panel from the other.
    for left in range(annotations.shape[1]):
        for source in range(annotations.shape[1]):
            forward = annotations[:, left] @ xw[:, source]
            reverse = annotations[:, source] @ wx[:, left]
            np.testing.assert_allclose(forward, reverse, rtol=2e-15, atol=2e-12)


def test_reference_uses_global_passes_and_matches_dense_scores(tmp_path):
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

    dense_scores = _dense_panels(raw, env_raw, annot)
    suffix = {"xx": "gxx", "xw": "gxe", "wx": "exg", "ww": "gee"}
    for key, stem in suffix.items():
        observed = pd.read_csv(tmp_path / f"fused.{stem}.ldscore.gz", sep="\t")[["a", "b"]].to_numpy()
        # Text score bundles round-trip the binary64 reference values.
        np.testing.assert_allclose(observed, dense_scores[key], rtol=2e-14, atol=2e-14)
    assert not (tmp_path / "fused.gxe.jackknife.npz").exists()

def test_native_tiling_accounts_for_opaque_panel_preparation_peak():
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.target_xz_mem = 6 * 101 * 3 * 7.5 * 8 / (1024 ** 3)
    obj.dtype = np.float64
    obj.nsamp = 101
    obj.nbins = 3
    obj.nvecs = 23
    obj.jackknife_ids = None
    obj.jackknife_labels = []
    obj.shard_mode = False
    obj.native_backend = "direct"
    obj.log = Logger(suppress=True)
    tiles = obj._auto_vtiles()
    assert max(size for _, size in tiles) == 6
    assert sum(size for _, size in tiles) == 23
    peak = 6 * obj.nsamp * obj.nbins * max(size for _, size in tiles) * 8
    assert peak <= obj.target_xz_mem * 1024**3


def test_balanced_probe_tiling_rejects_infeasible_max_tile_cap():
    with pytest.raises(RuntimeError, match="exceeding max_tiles=8"):
        _build_balanced_vtiles(1_024, vmax=127, gran=64, max_tiles=8)
    tiles = _build_balanced_vtiles(1_024, vmax=128, gran=64, max_tiles=8)
    assert tiles == [(start, 128) for start in range(0, 1_024, 128)]


def test_bed_reader_thread_forwarding():
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


def test_reference_failure_leaves_no_outputs(tmp_path):
    obj, _, _, _ = _make_toy(tmp_path, "failure")

    def fail_during_source_construction(self, *args, **kwargs):
        raise RuntimeError("injected sketch failure")

    obj._accumulate_sketch_block = types.MethodType(fail_during_source_construction, obj)
    with pytest.raises(RuntimeError, match="injected sketch failure"):
        obj._compute_ldscore()
    assert not list(tmp_path.glob(".gxe-jackknife-*"))
    assert not list(tmp_path.glob("failure.g*"))
    assert not list(tmp_path.glob("*.gxe.bundle.lock"))
