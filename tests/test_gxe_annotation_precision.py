"""Canonical binary64 annotation semantics (audit Finding 1).

The user-supplied annotation matrix defines the scientific estimand.  It must
stay contiguous binary64 independently of the randomized retained-storage
``dtype`` option, and the artifact schema must distinguish binary64-annotation
references (schema v4) from legacy references that may have rounded continuous
annotations to the storage dtype (schema v3).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _canonicalize_annotation_matrix,
)
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.ldscore import gxe_score
from summit.logger import Logger

N, M = 41, 29
STEP = 7

# Continuous binary64 values that a float32 rounding would silently change.
COLLAPSING_HIGH = 1.0 + 2.0 ** -30      # collapses to exactly 1.0 in float32
TINY_POSITIVE = 2.0 ** -150             # underflows to 0.0 in float32
LARGE_FINITE = 2.0 * float(np.finfo(np.float32).max)  # overflows to inf in float32


def _write_inputs(tmp_path: Path):
    rng = np.random.default_rng(97531)
    raw = rng.binomial(2, rng.uniform(0.18, 0.42, size=M), size=(N, M)).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    env1 = rng.normal(size=N)
    env2 = 0.4 * env1 + rng.normal(size=N)
    environment = tmp_path / "environment.tsv"
    pd.DataFrame(
        {
            "FID": fam[0].astype(str),
            "IID": fam[1].astype(str),
            "age": env1,
            "bmi": env2,
        }
    ).to_csv(environment, sep="\t", index=False, na_rep="NA")
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None)
    meta = {
        "CHR": bim[0].astype(str),
        "BP": bim[3].astype(np.int64),
        "SNP": bim[1].astype(str),
    }
    return prefix, environment, meta


def _write_annotation(tmp_path: Path, meta, columns: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "annotation.tsv"
    pd.DataFrame({**meta, **columns}).to_csv(
        path, sep="\t", index=False, float_format="%.17g"
    )
    return path


def _continuous_columns():
    c0 = np.ones(M)
    c0[::2] = COLLAPSING_HIGH
    c1 = np.full(M, TINY_POSITIVE)
    c1[0] = 1.0
    c2 = np.full(M, 1.0)
    c2[3] = LARGE_FINITE
    return {"CONT_A": c0, "CONT_B": c1, "CONT_C": c2}


def _binary_columns():
    return {
        "BIN_A": (np.arange(M) % 2 == 0).astype(float),
        "BIN_B": (np.arange(M) % 3 != 0).astype(float),
    }


def _estimator(prefix, environment, out_path, column, *, annot, dtype, num_vecs=64):
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=None if annot is None else str(annot),
        out_path=str(out_path),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=num_vecs,
        step_size=STEP,
        seed=2718,
        dtype=dtype,
        num_threads=2,
        target_xz_mem=0.01,
        kernel_mode="standardized",
        genotype_scale="sample",
        native_backend="python",
    )


def _run_reference(tmp_path, prefix, environment, annot, dtype, case):
    out_dir = tmp_path / case
    out_dir.mkdir()
    estimators = [
        _estimator(
            prefix, environment, out_dir / f"ref.{column}", column,
            annot=annot, dtype=dtype,
        )
        for column in ("age", "bmi")
    ]
    try:
        batch = generate_multi_environment_references(
            estimators,
            batch_manifest=out_dir / "batch.gxe.multi.json",
            requested_backend="direct",
        )
    finally:
        for estimator in estimators:
            estimator.close()
    return out_dir, Path(batch)


def _scores(out_dir: Path, column: str) -> dict[str, np.ndarray]:
    result = {}
    for suffix in ("gxx", "gxe", "exg", "gee"):
        frame = pd.read_csv(out_dir / f"ref.{column}.{suffix}.ldscore.gz", sep="\t")
        value_columns = [
            name for name in frame.columns
            if name not in ("CHR", "SNP", "BP", "A1", "A2")
        ]
        result[suffix] = frame[value_columns].to_numpy()
    return result


def _expected_digest(names, matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\n")
    digest.update(np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def test_canonical_conversion_is_exact_and_validated():
    exact = _canonicalize_annotation_matrix(
        np.array([[1.0, COLLAPSING_HIGH], [TINY_POSITIVE, LARGE_FINITE]])
    )
    assert exact.dtype == np.float64 and exact.flags["C_CONTIGUOUS"]
    assert exact[0, 1] == COLLAPSING_HIGH != np.float64(np.float32(COLLAPSING_HIGH))
    assert exact[1, 0] == TINY_POSITIVE > 0.0
    assert np.isfinite(exact[1, 1]) and exact[1, 1] == LARGE_FINITE

    small_int = _canonicalize_annotation_matrix(np.array([[1, 2 ** 53]], dtype=np.int64))
    assert np.array_equal(small_int, np.array([[1.0, float(2 ** 53)]]))
    with pytest.raises(ValueError, match="2\\*\\*53"):
        _canonicalize_annotation_matrix(np.array([[2 ** 53 + 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        _canonicalize_annotation_matrix(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        _canonicalize_annotation_matrix(np.array([[np.inf]]))
    with pytest.raises(ValueError, match="non-negative"):
        _canonicalize_annotation_matrix(np.array([[-1.0]]))
    if np.dtype(np.longdouble).itemsize > 8:
        inexact = np.array([[1.0]], dtype=np.longdouble)
        inexact += np.finfo(np.longdouble).eps
        with pytest.raises(ValueError, match="Extended-precision"):
            _canonicalize_annotation_matrix(inexact)
        assert _canonicalize_annotation_matrix(
            np.array([[1.0, 0.5]], dtype=np.longdouble)
        ).dtype == np.float64


def test_annotation_matrix_and_masses_ignore_storage_dtype(tmp_path):
    prefix, environment, meta = _write_inputs(tmp_path)
    annot = _write_annotation(tmp_path, meta, _continuous_columns())
    columns = _continuous_columns()
    expected = np.column_stack([columns[k] for k in ("CONT_A", "CONT_B", "CONT_C")])

    observed = {}
    for dtype in ("float32", "float64"):
        estimator = _estimator(
            prefix, environment, tmp_path / f"probe.{dtype}", "age",
            annot=annot, dtype=dtype,
        )
        try:
            assert estimator.annot.dtype == np.float64
            assert estimator.annot.flags["C_CONTIGUOUS"]
            assert np.array_equal(estimator.annot, expected)
            assert estimator.is_continuous is True
            assert np.array_equal(
                estimator.nsnps_bin, expected.sum(axis=0, dtype=np.float64)
            )
            assert np.all(estimator.nsnps_bin > 0.0)
            observed[dtype] = (
                estimator.annot.tobytes(),
                estimator.nsnps_bin.tobytes(),
                estimator._annotation_digest(),
            )
        finally:
            estimator.close()
    assert observed["float32"] == observed["float64"]
    assert observed["float64"][2] == _expected_digest(
        ["CONT_A", "CONT_B", "CONT_C"], expected
    )


def test_reference_outputs_identical_across_storage_dtypes(tmp_path):
    prefix, environment, meta = _write_inputs(tmp_path)
    annot = _write_annotation(tmp_path, meta, _continuous_columns())
    dirs = {}
    for dtype in ("float32", "float64"):
        dirs[dtype], _ = _run_reference(
            tmp_path, prefix, environment, annot, dtype, f"run_{dtype}"
        )
    for column in ("age", "bmi"):
        left = _scores(dirs["float32"], column)
        right = _scores(dirs["float64"], column)
        for suffix in left:
            assert np.array_equal(left[suffix], right[suffix]), (
                f"{column}.{suffix} differs between storage dtypes"
            )
        left_manifest = json.loads(
            (dirs["float32"] / f"ref.{column}.gxe.ref.json").read_text()
        )
        right_manifest = json.loads(
            (dirs["float64"] / f"ref.{column}.gxe.ref.json").read_text()
        )
        for manifest, dtype in ((left_manifest, "float32"), (right_manifest, "float64")):
            assert manifest["schema_version"] == 4
            assert manifest["annotation_value_dtype"] == "float64"
            assert manifest["randomization"]["dtype"] == dtype
        assert left_manifest["annotation_digest"] == right_manifest["annotation_digest"]
        assert left_manifest["annotation_masses"] == right_manifest["annotation_masses"]
        assert left_manifest["annotation_masses"] == [
            float(value)
            for value in np.column_stack(
                [_continuous_columns()[k] for k in ("CONT_A", "CONT_B", "CONT_C")]
            ).sum(axis=0, dtype=np.float64)
        ]


def test_binary_annotation_reference_is_storage_dtype_invariant(tmp_path):
    prefix, environment, meta = _write_inputs(tmp_path)
    annot = _write_annotation(tmp_path, meta, _binary_columns())
    dirs = {}
    for dtype in ("float32", "float64"):
        dirs[dtype], _ = _run_reference(
            tmp_path, prefix, environment, annot, dtype, f"binary_{dtype}"
        )
    for column in ("age", "bmi"):
        left = _scores(dirs["float32"], column)
        right = _scores(dirs["float64"], column)
        for suffix in left:
            assert np.array_equal(left[suffix], right[suffix])
    manifest = json.loads(
        (dirs["float64"] / "ref.age.gxe.ref.json").read_text()
    )
    assert manifest["annotation_masses"] == [15.0, 19.0]


def test_reference_schema_pledge_handling(tmp_path):
    prefix, environment, meta = _write_inputs(tmp_path)
    annot = _write_annotation(tmp_path, meta, _binary_columns())
    out_dir, _ = _run_reference(
        tmp_path, prefix, environment, annot, "float64", "pledge"
    )
    manifest_path = out_dir / "ref.age.gxe.ref.json"
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    validated = gxe_score._validate_reference_manifest(
        manifest_path, scratch_dir=scratch
    )
    assert validated.payload["schema_version"] == 4
    assert validated.payload["annotation_value_dtype"] == "float64"

    def doctored(case, mutate):
        bundle = tmp_path / case
        shutil.copytree(out_dir, bundle)
        target = bundle / "ref.age.gxe.ref.json"
        payload = json.loads(target.read_text())
        mutate(payload)
        target.write_text(json.dumps(payload))
        return target

    def drop_pledge(payload):
        del payload["annotation_value_dtype"]

    with pytest.raises(ValueError, match="binary64 annotation pledge"):
        gxe_score._validate_reference_manifest(
            doctored("v4_missing_pledge", drop_pledge), scratch_dir=scratch
        )

    def downgrade_keep_pledge(payload):
        payload["schema_version"] = 3

    with pytest.raises(ValueError, match="unexpected annotation dtype pledge"):
        gxe_score._validate_reference_manifest(
            doctored("v3_with_pledge", downgrade_keep_pledge), scratch_dir=scratch
        )

    def legacy(payload):
        payload["schema_version"] = 3
        del payload["annotation_value_dtype"]
        del payload["annotation_digest"]

    legacy_validated = gxe_score._validate_reference_manifest(
        doctored("v3_legacy", legacy), scratch_dir=scratch
    )
    assert legacy_validated.payload["schema_version"] == 3

    def unsupported(payload):
        payload["schema_version"] = 5

    with pytest.raises(ValueError, match="schema-v3 or schema-v4"):
        gxe_score._validate_reference_manifest(
            doctored("v5_unknown", unsupported), scratch_dir=scratch
        )
