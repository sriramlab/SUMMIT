"""Maximum-capacity native scratch allocation (audit Findings 2 and 3).

Every native execution-scratch role must hold exactly one physical
allocation whose capacity comes from the frozen execution plan, reused by
full blocks, the terminal genotype block, terminal environment tiles,
terminal probe tiles, and every legal logical column width — never one
mapping per exact logical shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.logger import Logger

N, M = 31, 13
STEP = 7  # genotype blocks [0,7) and [7,13): full width K=7, terminal K-1=6.


def _inputs(tmp_path: Path):
    rng = np.random.default_rng(8675309)
    raw = rng.binomial(2, rng.uniform(0.2, 0.4, size=M), size=(N, M)).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    env1 = rng.normal(size=N)
    env2 = 0.3 * env1 + rng.normal(size=N)
    env3 = rng.normal(size=N)
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "age": env1,
            "bmi": env2,
            "smk": env3,
        }
    ).to_csv(environment, sep="\t", index=False, na_rep="NA")
    return prefix, environment


def _estimator(prefix, environment, out_path, column, *, num_vectors):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(out_path),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=num_vectors,
        step_size=STEP,
        seed=2718,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend="python",
    )


def _score(prefix: Path, suffix: str) -> np.ndarray:
    frame = pd.read_csv(f"{prefix}.{suffix}.ldscore.gz", sep="\t")
    return frame[["L2_0"]].to_numpy()


def test_terminal_geometry_uses_one_allocation_per_scratch_role(tmp_path):
    prefix, environment = _inputs(tmp_path)
    columns = ("age", "bmi", "smk")

    independent = {}
    for column in columns:
        estimator = _estimator(
            prefix, environment, tmp_path / f"solo.{column}", column,
            num_vectors=63,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"solo.{column}"

    estimators = [
        _estimator(
            prefix, environment, tmp_path / f"multi.{column}", column,
            num_vectors=63,
        )
        for column in columns
    ]
    # A tiny paired-panel budget admits fewer than three environments per
    # tile, producing a terminal environment tile; a width-63 dense probe
    # plan under the same budget also exercises a terminal probe tile.
    for estimator in estimators:
        estimator.target_xz_mem = 11_000 / 1024**3
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "multi.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()

    payload = json.loads(Path(batch).read_text(encoding="utf-8"))
    plan = payload["complete_process_memory_plan"]
    performance = payload["performance_telemetry"]
    context = performance["multi_environment_direct_context"]
    kernel = performance["multi_environment_native_kernel"]
    components = plan["component_bytes"]

    assert plan["direct_kernel_mode"] == "dense_blas_hybrid"
    assert context["environment_tile_count"] >= 2, (
        "geometry must include a terminal environment tile"
    )
    assert context["max_genotype_block_width"] == STEP
    block_count = int(context["block_count"])
    passes = int(context["planned_genotype_passes"])
    assert block_count == 2

    # Exactly one live physical allocation per scratch role.  Source scratch
    # is deliberately released after each source pass (the planner's phase
    # model excludes it from the target phase), so its allocation count
    # equals the number of source passes — never one mapping per exact
    # logical shape, and never two mappings alive at once.
    roles = kernel["scratch_roles"]
    source_passes = (
        int(context["environment_tile_count"])
        * int(context["execution_probe_chunk_count"])
    )
    assert roles["source_output"]["allocations"] == source_passes
    assert roles["source_output"]["reuses"] >= block_count - 1
    assert roles["source_output"]["released_bytes"] == (
        roles["source_output"]["capacity_bytes"]
    )
    assert roles["target_output"]["allocations"] == 1
    assert roles["target_output"]["reuses"] >= 1
    for name in ("source_output", "target_output", "source_weights"):
        assert roles[name]["capacity_bytes"] > 0, name
    # The direct context supplies its constructor-time sqrt(A) cache, so the
    # kernel's per-call annotation scratch role stays untouched.
    assert roles["source_annotation"]["allocations"] == 0
    assert roles["source_annotation"]["capacity_bytes"] == 0
    assert context["decoded_scratch_allocations"] == 1
    assert context["decoded_scratch_reuses"] == passes * block_count - 1

    # Native capacity never exceeds the admitted plan, and covers all
    # logical shapes (the run completed without a capacity rejection).
    assert (
        context["decoded_scratch_capacity_bytes"]
        <= components["decoded_genotype_block"]
    )
    assert (
        context["source_output_scratch_capacity_bytes"]
        <= components["source_contribution"]
    )
    assert (
        context["target_output_scratch_capacity_bytes"]
        <= components["target_output"]
    )
    assert roles["source_output"]["capacity_bytes"] == (
        context["source_output_scratch_capacity_bytes"]
    )
    assert roles["target_output"]["capacity_bytes"] == (
        context["target_output_scratch_capacity_bytes"]
    )
    assert roles["source_weights"]["capacity_bytes"] <= (
        components["source_weights"]
    )
    assert roles["source_annotation"]["capacity_bytes"] <= (
        components["source_block_annotation"]
    )

    # No second multi-shape mapping: capacity equals the single maximum
    # admitted logical shape, not a sum over distinct shapes.
    n_samples = int(context["rows"])
    assert context["decoded_scratch_capacity_bytes"] == n_samples * STEP * 8

    # Smaller logical shapes follow larger ones inside this run (terminal
    # genotype block after the full block, terminal environment tile after
    # the full tile, in both genotype passes).  Exact agreement with the
    # independently constructed one-environment references proves the
    # reused capacity carries no stale contamination.
    for column in columns:
        observed = tmp_path / f"multi.{column}"
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=5e-12,
                atol=5e-12,
            )


def test_fused_single_tile_plan_reports_exact_capacities(tmp_path):
    prefix, environment = _inputs(tmp_path)
    estimators = [
        _estimator(
            prefix, environment, tmp_path / f"fused.{column}", column,
            num_vectors=64,
        )
        for column in ("age", "bmi")
    ]
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "fused.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()
    payload = json.loads(Path(batch).read_text(encoding="utf-8"))
    context = payload["performance_telemetry"][
        "multi_environment_direct_context"
    ]
    kernel = payload["performance_telemetry"]["multi_environment_native_kernel"]
    assert context["fused_two_pass_execution"] is True
    assert context["planned_genotype_passes"] == 2
    n_samples = int(context["rows"])
    chunk = int(context["maximum_execution_probe_chunk_width"])
    wide = 2 * 2 * 1 * chunk  # 2 environments, one annotation bin
    assert context["source_output_scratch_capacity_bytes"] == (
        n_samples * wide * 8
    )
    assert context["target_output_scratch_capacity_bytes"] == (
        STEP * 2 * wide * 8
    )
    assert kernel["scratch_roles"]["source_output"]["allocations"] == 1
    assert kernel["scratch_roles"]["target_output"]["allocations"] == 1
    assert context["decoded_scratch_allocations"] == 1
    assert kernel["execution_scratch_released"] is False
