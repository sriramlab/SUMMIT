"""Q-panelled packed-feature Mailman scratch (audit Finding 2).

The packed feature contraction must not allocate a lookup table over the
entire fused feature RHS width; per-worker scratch is context-owned,
bounded by the shared q-panel policy, exactly mirrored by the planner,
and every SUMMIT_MAILMAN_* override is either frozen and represented
exactly or rejected before native construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.gxe_multi import (
    _frozen_mailman_environment,
    _mailman_qpanel_width,
    _mailman_segment_size,
    _mailman_worker_scratch_bytes,
    generate_multi_environment_references,
)
from summit.logger import Logger

ENVIRONMENTS = ("e1", "e2", "e3", "e4", "e5")


def _inputs(tmp_path: Path, *, n: int, m: int, covariate_columns: int):
    rng = np.random.default_rng(24601)
    raw = rng.binomial(2, rng.uniform(0.2, 0.4, size=m), size=(n, m)).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = {"FID": fam[0].astype(str), "IID": fam[1].astype(str)}
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {**ids, **{name: rng.normal(size=n) for name in ENVIRONMENTS}}
    ).to_csv(environment, sep="\t", index=False)
    covariates = None
    if covariate_columns:
        covariates = tmp_path / "covariates.tsv"
        pd.DataFrame(
            {
                **ids,
                **{
                    f"c{index}": rng.normal(size=n)
                    for index in range(covariate_columns)
                },
            }
        ).to_csv(covariates, sep="\t", index=False)
    return prefix, environment, covariates


def _estimator(prefix, environment, covariates, out_path, column, *, num_vectors):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(out_path),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        covar_path=None if covariates is None else str(covariates),
        num_vecs=num_vectors,
        step_size=7,
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


def _run_packed(tmp_path, prefix, environment, covariates, *, tag):
    estimators = [
        _estimator(
            prefix, environment, covariates,
            tmp_path / f"{tag}.{column}", column, num_vectors=6,
        )
        for column in ENVIRONMENTS
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


def test_high_rank_many_environment_worker_scratch_is_exact_and_bounded(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SUMMIT_MAILMAN_SEGMENT_SIZE", "8")
    n, m, covariate_columns = 400, 12, 175
    prefix, environment, covariates = _inputs(
        tmp_path, n=n, m=m, covariate_columns=covariate_columns
    )

    independent = {}
    for column in ENVIRONMENTS[:2]:
        estimator = _estimator(
            prefix, environment, covariates,
            tmp_path / f"solo.{column}", column, num_vectors=6,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"solo.{column}"

    payload = _run_packed(
        tmp_path, prefix, environment, covariates, tag="packed"
    )
    plan = payload["complete_process_memory_plan"]
    performance = payload["performance_telemetry"]
    context = performance["multi_environment_direct_context"]
    kernel = performance["multi_environment_native_kernel"]
    assert plan["direct_kernel_mode"] == "packed_mailman"
    assert kernel["mailman_plan_frozen"] is True

    frozen = _frozen_mailman_environment()
    segment = _mailman_segment_size(n, frozen)
    assert segment == 8
    table = 3**segment
    assert context["mailman_frozen_segment_size"] == segment
    assert context["mailman_frozen_table_size"] == table

    feature_rhs_columns = (
        int(kernel["feature_basis_columns"]) + 1 + 3 * len(ENVIRONMENTS)
    )
    assert context["mailman_qpanel_feature"] == _mailman_qpanel_width(
        table, feature_rhs_columns, segment, frozen, 2
    )
    full_width_table_bytes = table * feature_rhs_columns * 8
    assert full_width_table_bytes > 8 * 1024**2, (
        "geometry must reproduce the audited >8 MiB full-width table regime"
    )

    per_worker = context["mailman_worker_scratch_capacity_bytes_per_worker"]
    threads = int(context["threads"])
    assert per_worker > 0
    assert per_worker < full_width_table_bytes
    role = kernel["scratch_roles"]["mailman_worker"]
    assert role["capacity_bytes"] == threads * per_worker
    assert role["allocations"] == threads
    assert plan["component_bytes"]["mailman_worker_scratch"] >= (
        threads * per_worker
    )

    for column in ENVIRONMENTS[:2]:
        observed = tmp_path / f"packed.{column}"
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=5e-12,
                atol=5e-12,
            )


def test_mailman_overrides_are_frozen_exactly_or_rejected(tmp_path, monkeypatch):
    n, m = 60, 10
    prefix, environment, covariates = _inputs(
        tmp_path, n=n, m=m, covariate_columns=0
    )

    monkeypatch.setenv("SUMMIT_MAILMAN_WORK_MB", "1")
    monkeypatch.setenv("SUMMIT_MAILMAN_QPANEL", "3")
    payload = _run_packed(
        tmp_path, prefix, environment, covariates, tag="override"
    )
    context = payload["performance_telemetry"][
        "multi_environment_direct_context"
    ]
    assert context["mailman_qpanel_feature"] == 3
    assert context["mailman_qpanel_source"] == 3
    assert context["mailman_qpanel_target"] == 3
    frozen = _frozen_mailman_environment()
    assert frozen["SUMMIT_MAILMAN_WORK_MB"] == 1
    assert frozen["SUMMIT_MAILMAN_QPANEL"] == 3

    for name in (
        "SUMMIT_MAILMAN_SEGMENT_SIZE",
        "SUMMIT_MAILMAN_QPANEL",
        "SUMMIT_MAILMAN_WORK_MB",
    ):
        for bad in ("8x", "-2", "0", " 4"):
            monkeypatch.setenv(name, bad)
            with pytest.raises(ValueError, match="unmodeled Mailman override"):
                _frozen_mailman_environment()
        monkeypatch.delenv(name)


def test_qpanel_mirror_matches_native_policy():
    frozen = {"SUMMIT_MAILMAN_SEGMENT_SIZE": None,
              "SUMMIT_MAILMAN_QPANEL": None,
              "SUMMIT_MAILMAN_WORK_MB": None}
    # The default 8 MiB budget with one segment buffer reproduces the
    # historical source/target widths, including the >=64 rounding.
    assert _mailman_qpanel_width(6561, 100_000, 8, frozen) == (
        (8 * 1024 * 1024 // (6561 * 8 + 8 * 8 + 8)) // 64 * 64
    )
    assert _mailman_qpanel_width(6561, 10, 8, frozen) == 10
    # Two segment buffers (the feature kernel) shrink the panel.
    assert _mailman_qpanel_width(6561, 100_000, 8, frozen, 2) <= (
        _mailman_qpanel_width(6561, 100_000, 8, frozen, 1)
    )
    scratch = _mailman_worker_scratch_bytes(
        rows=3**9,
        feature_rhs_columns=212,
        wide_columns=120,
        threads=32,
        frozen_environment=frozen,
    )
    assert scratch["segment_size"] == 8
    assert scratch["table_size"] == 6561
    assert scratch["total_bytes"] == 32 * scratch["per_worker_bytes"]
    # The audited defect: a full-width feature table at these dimensions
    # costs 11,154,592 bytes per worker; the q-panelled bound stays inside
    # the shared 8 MiB budget policy.
    full_width = 6561 * 212 * 8 + 2 * 8 * 212 * 8
    assert full_width > 11_000_000
    assert scratch["per_worker_bytes"] <= 9 * 1024**2
