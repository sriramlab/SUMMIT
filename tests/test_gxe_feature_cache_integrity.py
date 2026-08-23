from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import (
    GenomewideEnvLDScore,
    _ndarray_sha256,
    _orthonormalize_columns,
    _validate_feature_cache_semantics,
)
from summit.logger import Logger


def _make_builder(tmp_path: Path, name: str = "cache") -> GenomewideEnvLDScore:
    rng = np.random.default_rng(9917)
    n, m = 27, 9
    raw = rng.binomial(2, rng.uniform(0.18, 0.44, m), size=(n, m)).astype(float)
    prefix = tmp_path / f"{name}-geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env = tmp_path / f"{name}.env.tsv"
    cov = tmp_path / f"{name}.cov.tsv"
    ids.assign(E=rng.normal(size=n)).to_csv(env, sep="\t", index=False)
    ids.assign(C=rng.normal(size=n)).to_csv(cov, sep="\t", index=False)
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(env),
        covar_path=str(cov),
        annot_path=None,
        out_path=str(tmp_path / name),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=100,
        step_size=4,
        seed=73,
        dtype="float64",
        num_threads=1,
        kernel_mode="standardized",
        genotype_scale="sample",
        target_xz_mem=0.01,
    )


def _read_cache(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        arrays = {
            name: np.asarray(bundle[name]).copy()
            for name in bundle.files
            if name != "metadata_json"
        }
    return metadata, arrays


def _reseal_cache(path: Path, metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    metadata["array_sha256"] = {
        name: _ndarray_sha256(value) for name, value in arrays.items()
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )


def test_schema_v2_cache_seals_exact_nxe_sufficient_statistics(tmp_path):
    builder = _make_builder(tmp_path)
    cache = tmp_path / "features.gxe.cache.npz"
    builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)

    assert metadata["schema_version"] == 2
    backend = metadata["backend_provenance"]
    assert backend["artifact_stage"] == "feature_construction"
    assert backend["backend_name"] == "python_numpy"
    assert backend["actual_global_2b_source_columns"] == 0
    assert backend["actual_jackknife_2b_source_columns"] == 0
    assert metadata["environment_transform"]["analysis_mean"] == (
        math.fsum(float(value) for value in builder.env) / builder.env.size
    )
    assert metadata["environment_transform"]["analysis_sum_squares"] == math.fsum(
        float(value) * float(value) for value in builder.env
    )
    assert arrays["nxe_qdq"].shape == (builder.p_eff + 1, builder.p_eff + 1)
    assert arrays["nxe_trace_terms"].shape == (3,)
    _validate_feature_cache_semantics(metadata, arrays)

    intercept = np.ones((builder.nsamp, 1), dtype=np.float64) / math.sqrt(builder.nsamp)
    q_full = _orthonormalize_columns(np.column_stack([intercept, builder.C_int]))
    pmat = np.eye(builder.nsamp) - q_full @ q_full.T
    d = builder.env * builder.env
    nxe = pmat @ np.diag(d) @ pmat
    np.testing.assert_allclose(metadata["trace_nxe"], np.trace(nxe), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        metadata["trace_nxe_sq"], np.trace(nxe @ nxe), rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(arrays["nxe_qdq"], q_full.T @ (d[:, None] * q_full))
    np.testing.assert_allclose(
        arrays["nxe_trace_terms"],
        [d.sum(), np.dot(d, d), np.sum((d[:, None] * q_full) ** 2)],
    )


def test_feature_cache_backend_provenance_is_semantically_validated(tmp_path):
    builder = _make_builder(tmp_path, "backend-provenance")
    cache = tmp_path / "backend-provenance.gxe.cache.npz"
    builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)
    metadata["backend_provenance"]["actual_global_2b_source_columns"] = 1
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="inconsistent target width"):
        _validate_feature_cache_semantics(metadata, arrays)


def test_cache_identity_tolerates_only_environment_reduction_roundoff(tmp_path):
    builder = _make_builder(tmp_path, "environment-roundoff")
    cache = tmp_path / "environment-roundoff.gxe.cache.npz"
    builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)

    transform = metadata["environment_transform"]
    transform["analysis_sum_squares"] += (
        1.0e-12 * abs(transform["analysis_sum_squares"])
    )
    _reseal_cache(cache, metadata, arrays)
    builder._load_feature_cache(cache)

    metadata, arrays = _read_cache(cache)
    metadata["environment_transform"]["fixed_effect_design_sha256"] = "f" * 64
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="environment_transform"):
        builder._load_feature_cache(cache)

    metadata, arrays = _read_cache(cache)
    metadata["environment_transform"]["fixed_effect_design_sha256"] = (
        builder.environment_transform["fixed_effect_design_sha256"]
    )
    metadata["environment_transform"]["analysis_sum_squares"] += 1.0
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="environment sum of squares"):
        builder._load_feature_cache(cache)


@pytest.mark.parametrize("field", ["trace_nxe", "trace_nxe_sq"])
def test_cache_rejects_metadata_only_nxe_trace_tampering(tmp_path, field):
    builder = _make_builder(tmp_path, field)
    cache = tmp_path / f"{field}.gxe.cache.npz"
    builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)
    metadata[field] += 1.0
    _reseal_cache(cache, metadata, arrays)

    with pytest.raises(ValueError, match=field):
        builder._load_feature_cache(cache)


def test_cache_rejects_coordinated_nxe_array_and_hash_tampering(tmp_path):
    builder = _make_builder(tmp_path, "nxe-array")
    cache = tmp_path / "nxe-array.gxe.cache.npz"
    builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)
    arrays["nxe_trace_terms"][0] += 1.0
    metadata["trace_nxe"] += 1.0
    _reseal_cache(cache, metadata, arrays)

    with pytest.raises(ValueError, match="environment sum of squares"):
        builder._load_feature_cache(cache)


def test_cache_rejects_rehashed_norm_and_jackknife_tampering(tmp_path):
    builder = _make_builder(tmp_path, "features")
    cache = tmp_path / "features.gxe.cache.npz"
    builder.write_feature_cache(cache)

    metadata, arrays = _read_cache(cache)
    arrays["norm_x"][0] = 1.01
    metadata["feature_diagnostics"]["max_norm_additive_over_rank"] = 1.01
    metadata["feature_diagnostics"]["max_norm_error_additive"] = 0.01
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="unit residual norm"):
        builder._load_feature_cache(cache)


def test_cache_aborts_if_open_genotype_inode_changes_during_generation(
    tmp_path, monkeypatch
):
    builder = _make_builder(tmp_path, "moving-genotype")
    cache = tmp_path / "moving-genotype.gxe.cache.npz"
    bed = Path(builder.bed_prefix + ".bed")
    original_writer = builder._atomic_npz

    def touch_after_compute(path, **arrays):
        original_writer(path, **arrays)
        os.utime(bed, None)

    monkeypatch.setattr(builder, "_atomic_npz", touch_after_compute)
    with pytest.raises(RuntimeError, match="changed after the GxE estimator loaded"):
        builder.write_feature_cache(cache)
    assert not cache.exists()

    # The contract intentionally makes an estimator unusable once its bound
    # input inode changes. A fresh estimator binds the now-current input and
    # remains suitable for the independent legacy-jackknife rejection below.
    fresh_builder = _make_builder(tmp_path, "moving-genotype")
    fresh_builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)
    arrays["jackknife_ids"] = (
        np.arange(arrays["annotations"].shape[0], dtype=np.int32) % 2
    )
    metadata["jackknife_labels"] = ["block:1", "block:2"]
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="jackknife digest"):
        fresh_builder._load_feature_cache(cache)


def test_cache_rejects_legacy_schema_and_wrong_dtype(tmp_path):
    builder = _make_builder(tmp_path, "schema")
    cache = tmp_path / "schema.gxe.cache.npz"
    builder.write_feature_cache(cache)

    metadata, arrays = _read_cache(cache)
    metadata["schema_version"] = 1
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="schema-v2"):
        builder._load_feature_cache(cache)

    builder.write_feature_cache(cache, overwrite=True)
    metadata, arrays = _read_cache(cache)
    arrays["scale_x"] = arrays["scale_x"].astype(np.float32)
    _reseal_cache(cache, metadata, arrays)
    with pytest.raises(ValueError, match="dtype"):
        builder._load_feature_cache(cache)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unbound", "SHA-256"),
        ("shape", "shape"),
        ("nonfinite", "NaN or infinity"),
        ("rank", "ranks are inconsistent"),
        ("mode", "kernel_mode"),
        ("mass", "annotation masses"),
        ("scale", "nonpositive"),
        ("correlation", "Cauchy-Schwarz"),
        ("variant_digest", "variant digest"),
        ("annotation_digest", "annotation digest"),
        ("annotation_name", "reserved artifact"),
        ("trace_shape", "shape"),
        ("genotype_digest", "genotype digest"),
        ("design_digest", "design digest"),
    ],
)
def test_semantic_validator_rejects_resealed_malformed_contracts(tmp_path, case, message):
    builder = _make_builder(tmp_path, case)
    cache = tmp_path / f"{case}.gxe.cache.npz"
    builder.write_feature_cache(cache)
    metadata, arrays = _read_cache(cache)

    if case == "unbound":
        arrays["corr_xw"][0] += 0.01
    elif case == "shape":
        arrays["corr_xw"] = arrays["corr_xw"][:-1]
    elif case == "nonfinite":
        arrays["diag_nxe_x"][0] = np.inf
    elif case == "rank":
        metadata["residual_rank"] += 1
    elif case == "mode":
        metadata["kernel_mode"] = "unknown"
    elif case == "mass":
        metadata["annotation_masses"][0] += 1.0
    elif case == "scale":
        arrays["scale_w"][0] = 0.0
    elif case == "correlation":
        arrays["corr_xw"][0] = 2.0
    elif case == "variant_digest":
        arrays["variant_snp"][0] = "tampered"
    elif case == "annotation_digest":
        arrays["annotations"][0, 0] += 0.1
        arrays["annotations"][1, 0] -= 0.1
    elif case == "annotation_name":
        metadata["annotation_names"][0] = "CHR"
    elif case == "trace_shape":
        metadata["feature_diagnostics"]["kernel_traces_additive"] = [
            metadata["feature_diagnostics"]["kernel_traces_additive"][0],
            metadata["feature_diagnostics"]["kernel_traces_additive"][0],
        ]
    elif case == "genotype_digest":
        metadata["genotype_files"][".bed"]["sha256"] = "0" * 64
    elif case == "design_digest":
        metadata["environment_transform"]["fixed_effect_design_sha256"] = "invalid"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    if case != "unbound":
        metadata["array_sha256"] = {
            name: _ndarray_sha256(value) for name, value in arrays.items()
        }
    with pytest.raises(ValueError, match=message):
        _validate_feature_cache_semantics(metadata, arrays)
