"""Execution scratch is released before publication (audit Priority 0.3).

After the completed run's results are detached into separately owned
arrays and every native validation has passed, the direct context frees
its execution-only mappings; publication-phase memory evidence excludes
them, repeated release is harmless, and use-after-release fails clearly.
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
    rng = np.random.default_rng(1729)
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


def _estimator(prefix, environment, out_path, column):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(out_path),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=64,
        step_size=STEP,
        seed=2718,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized",
        genotype_scale="sample",
        native_backend="python",
    )


def _run(tmp_path, prefix, environment, *, tag, capture=None, monkeypatch=None):
    if capture is not None:
        original = gxe_multi._MultiEnvironmentGemm.run_multi_environment_direct_context

        def spy(self):
            result = original(self)
            capture["context"] = self._multi_environment_direct_context
            capture["release_evidence"] = dict(
                self._execution_scratch_release_evidence
            )
            return result

        monkeypatch.setattr(
            gxe_multi._MultiEnvironmentGemm,
            "run_multi_environment_direct_context",
            spy,
        )
    estimators = [
        _estimator(prefix, environment, tmp_path / f"{tag}.{column}", column)
        for column in ("age", "bmi")
    ]
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / f"{tag}.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()
    return json.loads(Path(batch).read_text(encoding="utf-8"))


def test_release_before_publication_and_results_survive(
    tmp_path, monkeypatch
):
    prefix, environment = _inputs(tmp_path)
    capture = {}
    payload = _run(
        tmp_path, prefix, environment, tag="released",
        capture=capture, monkeypatch=monkeypatch,
    )

    evidence = capture["release_evidence"]
    assert evidence["released"] is True
    assert evidence["live_scratch_capacity_bytes_before"] > 0
    assert evidence["live_scratch_capacity_bytes_after"] == 0
    assert evidence["decoded_scratch_released_bytes"] == N * STEP * 8
    assert evidence["kernel_scratch_released_bytes"] > 0
    assert evidence["rss_bytes_before_release"] > 0
    assert evidence["rss_bytes_after_release"] > 0

    # The published report carries the release evidence and the completed
    # context acknowledges the released state.
    published = payload["performance_telemetry"]["execution_scratch_release"]
    assert published["released"] is True
    assert published["live_scratch_capacity_bytes_after"] == 0
    context_info = payload["performance_telemetry"][
        "multi_environment_direct_context"
    ]
    assert context_info["execution_scratch_released"] is True
    assert context_info["decoded_scratch_released_bytes"] == N * STEP * 8

    # Returned values stayed valid through publication: the artifacts were
    # written after the release and remain finite and complete.
    for column in ("age", "bmi"):
        for suffix in ("gxx", "gxe", "exg", "gee"):
            frame = pd.read_csv(
                tmp_path / f"released.{column}.{suffix}.ldscore.gz", sep="\t"
            )
            values = frame["L2_0"].to_numpy()
            assert values.shape == (M,)
            assert np.all(np.isfinite(values))

    context = capture["context"]
    info = dict(context.info())
    assert info["execution_scratch_released"] is True

    # Repeated release is harmless and reports zero live capacity.
    second = dict(context.release_execution_scratch())
    assert second["released"] is True
    assert second["live_scratch_capacity_bytes_before"] == 0
    assert second["live_scratch_capacity_bytes_after"] == 0

    # Execution after terminal release is rejected.
    with pytest.raises(RuntimeError, match="single-use"):
        context.run()


def test_exception_cleanup_releases_scratch_without_masking(
    tmp_path, monkeypatch
):
    prefix, environment = _inputs(tmp_path)
    capture = {}
    original = (
        gxe_multi._MultiEnvironmentGemm._require_native_scratch_within_admitted_plan
    )

    def failing_validation(self, completed_info, kernel_info):
        # Fail after the run completed but before the wrapper's release, so
        # only the executor's exception cleanup can free the scratch.
        capture["context"] = self._multi_environment_direct_context
        original(self, completed_info, kernel_info)
        raise RuntimeError("injected post-run failure before publication")

    monkeypatch.setattr(
        gxe_multi._MultiEnvironmentGemm,
        "_require_native_scratch_within_admitted_plan",
        failing_validation,
    )
    estimators = [
        _estimator(prefix, environment, tmp_path / f"fail.{column}", column)
        for column in ("age", "bmi")
    ]
    try:
        with pytest.raises(RuntimeError, match="injected post-run failure"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "fail.gxe.multi.json",
                requested_backend="direct",
            )
    finally:
        for estimator in estimators:
            estimator.close()
    # The executor's exception cleanup released the completed context's
    # execution scratch; no mapping leaked past the failure.
    info = dict(capture["context"].info())
    assert info["execution_scratch_released"] is True
    follow_up = dict(capture["context"].release_execution_scratch())
    assert follow_up["live_scratch_capacity_bytes_after"] == 0
