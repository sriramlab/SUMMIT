"""Deterministic automatic canonical block width (step_size="auto").

The auto width is a pure function of the variant count — never machine
state — so an auto-selected finite-probe realization is exactly the one
an explicit --step_size of the same value would produce, and the manifest
records the resolved numeric width plus its selection provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit import cli
from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _AUTO_STEP_SIZE_CANONICAL_WIDTH,
    _auto_reference_step_size,
)
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.logger import Logger

N, M = 31, 29


def _inputs(tmp_path: Path):
    rng = np.random.default_rng(9219)
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


def _estimator(prefix, environment, out_path, column, *, step_size):
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
        step_size=step_size,
        seed=2718,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend="python",
    )


def _run(tmp_path, prefix, environment, tag, *, step_size):
    estimators = [
        _estimator(
            prefix, environment, tmp_path / f"{tag}.{column}", column,
            step_size=step_size,
        )
        for column in ("age", "bmi")
    ]
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / f"{tag}.gxe.multi.json",
            requested_backend="python",
        )
    finally:
        for estimator in estimators:
            estimator.close()
    return json.loads(Path(batch).read_text(encoding="utf-8"))


def test_auto_width_is_a_pure_function_of_the_variant_count():
    assert _auto_reference_step_size(29) == 29
    assert _auto_reference_step_size(8192) == 8192
    assert _auto_reference_step_size(454_207) == _AUTO_STEP_SIZE_CANONICAL_WIDTH
    assert _AUTO_STEP_SIZE_CANONICAL_WIDTH == 8192


def test_auto_resolution_and_manifest_provenance(tmp_path):
    prefix, environment = _inputs(tmp_path)
    estimator = _estimator(
        prefix, environment, tmp_path / "probe", "age", step_size="auto"
    )
    try:
        assert estimator.step_size == min(M, 8192)
        assert estimator.step_size_selection == "auto_v1"
    finally:
        estimator.close()
    explicit = _estimator(
        prefix, environment, tmp_path / "probe2", "age", step_size=7
    )
    try:
        assert explicit.step_size == 7
        assert explicit.step_size_selection == "explicit"
    finally:
        explicit.close()

    payload = _run(tmp_path, prefix, environment, "auto", step_size="auto")
    manifest = json.loads(
        (tmp_path / "auto.age.gxe.ref.json").read_text(encoding="utf-8")
    )
    randomization = manifest["randomization"]
    assert randomization["step_size"] == min(M, 8192)
    assert randomization["step_size_selection"] == "auto_v1"
    assert payload["requested_backend"] == "python"


def test_auto_realization_equals_the_explicit_width(tmp_path):
    prefix, environment = _inputs(tmp_path)
    _run(tmp_path, prefix, environment, "auto", step_size="auto")
    _run(tmp_path, prefix, environment, "explicit", step_size=min(M, 8192))
    for column in ("age", "bmi"):
        for suffix in ("gxx", "gxe", "exg", "gee"):
            left = pd.read_csv(
                tmp_path / f"auto.{column}.{suffix}.ldscore.gz", sep="\t"
            )
            right = pd.read_csv(
                tmp_path / f"explicit.{column}.{suffix}.ldscore.gz", sep="\t"
            )
            cols = [
                name for name in left.columns
                if name not in ("CHR", "SNP", "BP", "A1", "A2")
            ]
            np.testing.assert_array_equal(
                left[cols].to_numpy(), right[cols].to_numpy()
            )
        auto_manifest = json.loads(
            (tmp_path / f"auto.{column}.gxe.ref.json").read_text()
        )
        explicit_manifest = json.loads(
            (tmp_path / f"explicit.{column}.gxe.ref.json").read_text()
        )
        assert (
            auto_manifest["randomization"]["step_size"]
            == explicit_manifest["randomization"]["step_size"]
        )
        assert auto_manifest["randomization"]["step_size_selection"] == "auto_v1"
        assert (
            explicit_manifest["randomization"]["step_size_selection"]
            == "explicit"
        )


def test_mixed_selection_provenance_is_rejected(tmp_path):
    prefix, environment = _inputs(tmp_path)
    estimators = [
        _estimator(
            prefix, environment, tmp_path / "mix.age", "age",
            step_size="auto",
        ),
        _estimator(
            prefix, environment, tmp_path / "mix.bmi", "bmi",
            step_size=min(M, 8192),
        ),
    ]
    try:
        with pytest.raises(ValueError, match="step_size_selection"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "mix.gxe.multi.json",
                requested_backend="direct",
            )
    finally:
        for estimator in estimators:
            estimator.close()


def test_cli_step_size_parser_and_non_reference_guard():
    assert cli._step_size_argument("auto") == "auto"
    assert cli._step_size_argument(" AUTO ") == "auto"
    assert cli._step_size_argument("2000") == 2000
    with pytest.raises(ValueError):
        cli._step_size_argument("fast")

    class _Args:
        step_size = "auto"

    with pytest.raises(ValueError, match="only supported for GxE reference"):
        cli._require_integer_step_size(_Args(), "windowed LD scores")
    _Args.step_size = 1000
    cli._require_integer_step_size(_Args(), "windowed LD scores")
