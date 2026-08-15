from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore import gxe_multi
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.logger import Logger


def _inputs(tmp_path: Path, *, missing_second: bool = False):
    rng = np.random.default_rng(314159)
    n, m = 31, 13
    raw = rng.binomial(2, rng.uniform(0.16, 0.43, size=m), size=(n, m)).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    env1 = rng.normal(size=n)
    env2 = 0.35 * env1 + rng.normal(size=n)
    if missing_second:
        env2[0] = np.nan
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "age": env1,
            "bmi": env2,
        }
    ).to_csv(environment, sep="\t", index=False, na_rep="NA")
    return prefix, environment, m


def _estimator(
    prefix: Path,
    environment: Path,
    output: Path,
    column: str,
    *,
    num_vectors: int = 17,
    target_xz_mem: float = 0.01,
    write_jackknife: bool = True,
):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(output),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=num_vectors,
        step_size=5,
        seed=2718,
        dtype="float64",
        num_threads=1,
        target_xz_mem=target_xz_mem,
        kernel_mode="standardized",
        genotype_scale="sample",
        write_jackknife=write_jackknife,
        jackknife_spec="3",
        allow_low_probe_jackknife=True,
        native_backend="python",
    )


def _score(prefix: Path, suffix: str) -> np.ndarray:
    frame = pd.read_csv(f"{prefix}.{suffix}.ldscore.gz", sep="\t")
    return frame[["L2_0"]].to_numpy()


def test_shared_multi_environment_matches_independent_references(tmp_path):
    prefix, environment, variants = _inputs(tmp_path)
    independent = {}
    for column in ("age", "bmi"):
        estimator = _estimator(
            prefix, environment, tmp_path / f"independent.{column}", column
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"independent.{column}"

    estimators = [
        _estimator(prefix, environment, tmp_path / f"multi.{column}", column)
        for column in ("age", "bmi")
    ]
    read_counts = [0, 0]
    for index, estimator in enumerate(estimators):
        original = estimator._read_genotype_block

        def counted(self, start, stop, *, _index=index, _original=original):
            read_counts[_index] += 1
            return _original(start, stop)

        estimator._read_genotype_block = types.MethodType(counted, estimator)
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "multi.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()

    # Norm, global source, and global target: three shared reads per bounded
    # variant block, independent of environment count. Jackknife replicates
    # are formed later from completed block-local LD-score rows.
    assert read_counts == [3 * math.ceil(variants / 5), 0]
    payload = json.loads(batch.read_text(encoding="utf-8"))
    assert payload["kind"] == "summit.gxe.multi_environment_reference_batch"
    assert payload["num_environments"] == 2
    assert payload["shared_genotype_passes"] == 3
    assert payload["protected_native_gemm"] is True
    assert payload["repaired_gemm_output_columns"] == 0
    assert [item["environment"] for item in payload["references"]] == ["age", "bmi"]

    for column in ("age", "bmi"):
        observed = tmp_path / f"multi.{column}"
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=2e-12,
                atol=2e-12,
            )
        assert not Path(f"{observed}.gxe.jackknife.npz").exists()
        assert not Path(f"{independent[column]}.gxe.jackknife.npz").exists()
        manifest = json.loads(
            Path(f"{observed}.gxe.ref.json").read_text(encoding="utf-8")
        )
        assert manifest["jackknife"]["method"] == "block_local_ldscore_deletion"
        resources = manifest["resource_estimates"]
        assert resources["multi_environment_shared_decode"] == 1
        assert resources["multi_environment_protected_gemm"] == 1
        assert resources["multi_environment_count"] == 2
        assert resources["shared_genotype_passes"] == 3
        assert resources["native_repaired_gemm_output_columns"] == 0
        provenance = manifest["backend_provenance"]
        assert provenance["backend_name"] == "gxeldcore_direct"
        assert (
            provenance["compile_options"]["execution_mode"]
            == "shared_multi_environment_protected_gemm"
        )


def test_multi_environment_rejects_different_complete_case_cohorts(tmp_path):
    prefix, environment, _ = _inputs(tmp_path, missing_second=True)
    estimators = [
        _estimator(prefix, environment, tmp_path / f"different.{column}", column)
        for column in ("age", "bmi")
    ]
    try:
        try:
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "different.gxe.multi.json",
            )
        except ValueError as error:
            assert "same complete-case cohort" in str(error)
        else:
            raise AssertionError("different complete-case cohorts were accepted")
    finally:
        for estimator in estimators:
            estimator.close()


def test_shared_multi_environment_no_jackknife_accounts_for_probe_tiles(tmp_path):
    prefix, environment, variants = _inputs(tmp_path)
    # For two float64, one-annotation environments at N=31, this budget holds
    # two probes, forcing four shared tiles for seven probes.
    target_xz_mem = 2200 / 1024**3
    independent = {}
    for column in ("age", "bmi"):
        estimator = _estimator(
            prefix,
            environment,
            tmp_path / f"independent.nojk.{column}",
            column,
            num_vectors=7,
            target_xz_mem=target_xz_mem,
            write_jackknife=False,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()
        independent[column] = tmp_path / f"independent.nojk.{column}"

    estimators = [
        _estimator(
            prefix,
            environment,
            tmp_path / f"multi.nojk.{column}",
            column,
            num_vectors=7,
            target_xz_mem=target_xz_mem,
            write_jackknife=False,
        )
        for column in ("age", "bmi")
    ]
    reads = 0
    original = estimators[0]._read_genotype_block

    def counted(self, start, stop):
        nonlocal reads
        reads += 1
        return original(start, stop)

    estimators[0]._read_genotype_block = types.MethodType(counted, estimators[0])
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=tmp_path / "multi.nojk.gxe.multi.json",
        )
    finally:
        for estimator in estimators:
            estimator.close()

    payload = json.loads(batch.read_text(encoding="utf-8"))
    assert payload["randomization"]["probe_tiles"] == [[0, 2], [2, 2], [4, 2], [6, 1]]
    assert payload["shared_genotype_passes"] == 9
    assert reads == 9 * math.ceil(variants / 5)
    for column in ("age", "bmi"):
        observed = tmp_path / f"multi.nojk.{column}"
        assert not Path(f"{observed}.gxe.jackknife.npz").exists()
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(observed, suffix),
                _score(independent[column], suffix),
                rtol=2e-12,
                atol=2e-12,
            )


def test_multi_environment_batch_manifest_failure_rolls_back_bundles(
    tmp_path, monkeypatch
):
    prefix, environment, _ = _inputs(tmp_path)
    estimators = [
        _estimator(prefix, environment, tmp_path / f"rollback.{column}", column)
        for column in ("age", "bmi")
    ]
    monkeypatch.setattr(
        gxe_multi,
        "_publish_json_no_replace",
        lambda *_: (_ for _ in ()).throw(OSError("injected batch seal failure")),
    )
    try:
        with pytest.raises(OSError, match="injected batch seal failure"):
            generate_multi_environment_references(
                estimators,
                batch_manifest=tmp_path / "rollback.gxe.multi.json",
            )
    finally:
        for estimator in estimators:
            estimator.close()

    for column in ("age", "bmi"):
        output = tmp_path / f"rollback.{column}"
        assert not Path(f"{output}.gxe.ref.json").exists()
        assert not Path(f"{output}.gxx.ldscore.gz").exists()
        assert not Path(f"{output}.gxe.jackknife.npz").exists()
