"""Bounded semantic evidence and native mass validation (audit Findings 6/7).

The native context freezes a maximum semantic-call record count that the
plan charges inside the telemetry allowance, and rejects annotation
masses inconsistent with its copied canonical annotation matrix before
any allocation-heavy execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore import gxe_multi
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.logger import Logger

N, M, STEP = 31, 13, 7


def _inputs(tmp_path: Path):
    rng = np.random.default_rng(60221)
    raw = rng.binomial(2, rng.uniform(0.2, 0.4, size=M), size=(N, M)).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    env1 = rng.normal(size=N)
    env2 = 0.3 * env1 + rng.normal(size=N)
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "age": env1,
            "bmi": env2,
        }
    ).to_csv(environment, sep="\t", index=False)
    return prefix, environment


def _estimators(tmp_path, prefix, environment, tag):
    return [
        GenomewideEnvLDScore(
            bed_path=str(prefix),
            env_path=str(environment),
            env_col=column,
            annot_path=None,
            out_path=str(tmp_path / f"{tag}.{column}"),
            log=Logger(suppress=True),
            rand_dist="rademacher",
            low_level=None,
            num_vecs=64,
            step_size=STEP,
            seed=2718,
            dtype="float64",
            num_threads=2,
            target_xz_mem=0.01,
            kernel_mode="standardized_projected",
            genotype_scale="sample",
            native_backend="python",
        )
        for column in ("age", "bmi")
    ]


def test_semantic_call_records_are_bounded_and_admitted(tmp_path):
    prefix, environment = _inputs(tmp_path)
    estimators = _estimators(tmp_path, prefix, environment, "bounded")
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "bounded.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()
    payload = json.loads(Path(batch).read_text(encoding="utf-8"))
    plan = payload["complete_process_memory_plan"]
    context = payload["performance_telemetry"][
        "multi_environment_direct_context"
    ]
    tile_products = (
        int(context["environment_tile_count"])
        * int(context["execution_probe_chunk_count"])
    )
    expected_maximum = (
        int(context["block_count"]) * (1 + 2 * tile_products)
        + 2 * tile_products
    )
    assert context["planned_semantic_call_maximum"] == expected_maximum
    assert plan["semantic_call_records_maximum"] == expected_maximum
    assert plan["semantic_call_record_allowance_bytes"] == (
        expected_maximum * 4096
    )
    assert plan["semantic_call_record_allowance_bytes"] <= 256 * 1024**2


def test_oversized_semantic_record_plans_are_rejected(tmp_path, monkeypatch):
    prefix, environment = _inputs(tmp_path)
    monkeypatch.setattr(gxe_multi, "_TELEMETRY_ALLOWANCE_BYTES", 4096)
    estimators = _estimators(tmp_path, prefix, environment, "oversized")
    try:
        # Every candidate's record volume exceeds the shrunken telemetry
        # allowance, so the tile enumeration finds no feasible plan.
        with pytest.raises(RuntimeError, match="No GxE descriptor tile plan"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "oversized.gxe.multi.json",
                requested_backend="direct",
            )
    finally:
        for estimator in estimators:
            estimator.close()


def test_inconsistent_annotation_masses_fail_before_execution(tmp_path):
    prefix, environment = _inputs(tmp_path)
    estimators = _estimators(tmp_path, prefix, environment, "masses")
    try:
        for estimator in estimators:
            # Half the true mass with unchanged annotations: the same-person
            # denominator would silently change if the native boundary
            # accepted it (audit Finding 7's four-times counterexample).
            estimator.nsnps_bin = np.asarray(
                estimator.nsnps_bin, dtype=np.float64
            ) / 2.0
        with pytest.raises(
            RuntimeError,
            match="inconsistent with the copied annotation matrix",
        ):
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "masses.gxe.multi.json",
                requested_backend="direct",
            )
    finally:
        for estimator in estimators:
            estimator.close()
    assert not (tmp_path / "masses.gxe.multi.json").exists()
