"""Fused overwrite fails safe (audit Finding 4).

Fused batch publication has no backup/journal/restore transaction, so
``overwrite=True`` is rejected deterministically before any byte
changes; mixed overwrite settings are rejected as a broken common
contract.  The existing no-overwrite rollback tests remain in
test_gxe_multi_environment.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.logger import Logger

N, M, STEP = 31, 13, 7


def _inputs(tmp_path: Path):
    rng = np.random.default_rng(4104)
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


def _estimator(prefix, environment, out_path, column, *, overwrite):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None,
        out_path=str(out_path),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=17,
        step_size=STEP,
        seed=2718,
        dtype="float64",
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend="python",
        overwrite=overwrite,
    )


def _snapshot(root: Path) -> dict:
    state = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        state[str(path.relative_to(root))] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            stat.st_ino,
            stat.st_size,
        )
    return state


def _generate(tmp_path, prefix, environment, out_root, *, overwrite):
    estimators = [
        _estimator(
            prefix, environment, out_root / f"ref.{column}", column,
            overwrite=(
                overwrite[index] if isinstance(overwrite, (list, tuple))
                else overwrite
            ),
        )
        for index, column in enumerate(("age", "bmi"))
    ]
    try:
        return generate_multi_environment_references(
            estimators,
            batch_manifest=out_root / "batch.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()


def test_fused_overwrite_is_rejected_before_any_byte_changes(tmp_path):
    prefix, environment = _inputs(tmp_path)
    out_root = tmp_path / "bundle"
    out_root.mkdir()
    batch = _generate(tmp_path, prefix, environment, out_root, overwrite=False)
    payload = json.loads(Path(batch).read_text(encoding="utf-8"))
    assert payload["num_environments"] == 2
    before = _snapshot(out_root)
    assert before, "expected prepopulated environment references"

    with pytest.raises(ValueError, match="cannot overwrite existing"):
        _generate(tmp_path, prefix, environment, out_root, overwrite=True)

    after = _snapshot(out_root)
    # Original names, contents, sizes, and inodes are untouched, and no
    # partial batch manifest, staging file, or orphan lock appeared.
    assert after == before
    assert not [name for name in after if name.endswith(".lock")]
    assert not [name for name in after if ".tmp" in name or ".staging" in name]


def test_mixed_overwrite_settings_are_rejected(tmp_path):
    prefix, environment = _inputs(tmp_path)
    out_root = tmp_path / "mixed"
    out_root.mkdir()
    with pytest.raises(ValueError, match="disagree on overwrite"):
        _generate(
            tmp_path, prefix, environment, out_root, overwrite=(True, False)
        )
    assert _snapshot(out_root) == {}
