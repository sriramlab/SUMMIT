from __future__ import annotations

import json
import math
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bed_reader import to_bed

from summit.inference.gxe import fit_from_files
from summit.ldscore import gwe_ldscore, gxe_merge
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore.gxe_merge import merge_reference_shards
from summit.ldscore.gxe_score import score_phenotype_from_reference
from summit.logger import Logger


def _toy_inputs(tmp_path: Path):
    rng = np.random.default_rng(4471)
    n, m = 31, 13
    raw = rng.binomial(2, rng.uniform(0.15, 0.43, m), size=(n, m)).astype(float)
    prefix = tmp_path / "geno"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None)
    ids = pd.DataFrame({"FID": fam[0].astype(str), "IID": fam[1].astype(str)})
    env = tmp_path / "env.tsv"
    cov = tmp_path / "cov.tsv"
    pheno = tmp_path / "pheno.tsv"
    ids.assign(E=rng.normal(size=n)).to_csv(env, sep="\t", index=False)
    ids.assign(C=rng.normal(size=n)).to_csv(cov, sep="\t", index=False)
    ids.assign(Y=rng.normal(size=n)).to_csv(pheno, sep="\t", index=False)
    annot = tmp_path / "annot.tsv"
    pd.DataFrame(
        {
            "CHR": bim[0].astype(str),
            "SNP": bim[1].astype(str),
            "BP": bim[3].astype(int),
            "a": np.linspace(0.3, 1.0, m),
            "b": 0.25 + 0.75 * (np.arange(m) % 2),
        }
    ).to_csv(annot, sep="\t", index=False)
    return prefix, env, cov, pheno, annot, m


def _estimator(
    inputs,
    out: Path,
    *,
    probes: int,
    offset: int = 0,
    cache: Path | None = None,
    shard: bool = False,
    seed: int = 81,
):
    prefix, env, cov, _, annot, m = inputs
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(env),
        covar_path=str(cov),
        annot_path=str(annot),
        out_path=str(out),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=probes,
        step_size=m,
        seed=seed,
        dtype="float64",
        num_threads=1,
        kernel_mode="standardized",
        genotype_scale="sample",
        target_xz_mem=0.01,
        write_jackknife=True,
        jackknife_spec="3",
        allow_low_probe_jackknife=False,
        probe_offset=offset,
        feature_cache_path=None if cache is None else str(cache),
        shard_mode=shard,
    )


@pytest.fixture(scope="module")
def merged_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("gxe-shard-merge")
    inputs = _toy_inputs(root)
    cache = root / "features.gxe.cache.npz"
    builder = _estimator(inputs, root / "builder", probes=100)
    builder.write_feature_cache(cache)

    monolithic = _estimator(inputs, root / "mono", probes=100, cache=cache)
    reads = 0
    original = monolithic._read_genotype_block

    def counted(self, start, end):
        nonlocal reads
        reads += 1
        return original(start, end)

    monolithic._read_genotype_block = types.MethodType(counted, monolithic)
    monolithic._compute_ldscore()
    assert reads == 2

    shards = []
    for index in range(10):
        estimator = _estimator(
            inputs,
            root / f"shard-{index}",
            probes=10,
            offset=index * 10,
            cache=cache,
            shard=True,
        )
        estimator._compute_ldscore()
        shards.append(root / f"shard-{index}.gxe.shard.json")
    merged = merge_reference_shards(
        shards,
        feature_cache_path=cache,
        output_prefix=root / "merged",
    )
    return root, inputs, cache, shards, merged


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian", "spherical"])
def test_probe_rng_is_tiling_and_offset_invariant(distribution):
    obj = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    obj.root_seed = 991
    obj.probe_offset = 7
    obj.rand_dist = distribution
    obj.dtype = np.float64
    full = obj._generate_random_block(19, 17, 23, 0)
    tiled = np.column_stack(
        [obj._generate_random_block(19, 5, 23, 0), obj._generate_random_block(19, 12, 23, 5)]
    )
    np.testing.assert_array_equal(full, tiled)
    obj.probe_offset = 12
    shifted = obj._generate_random_block(19, 12, 23, 0)
    np.testing.assert_array_equal(shifted, full[:, 5:])


@pytest.mark.parametrize("malformation", ["duplicate_reserved", "negative"])
def test_panel_reader_rejects_ambiguous_or_negative_contributions(tmp_path, malformation):
    variants = pd.DataFrame(
        {"CHR": ["1", "1"], "SNP": ["rs1", "rs2"], "BP": [10, 20]}
    )
    panel = tmp_path / f"{malformation}.ldscore.gz"
    if malformation == "duplicate_reserved":
        frame = pd.DataFrame(
            np.asarray([[1, "rs1", 10, 0.2], [1, "rs2", 20, 0.3]], dtype=object),
            columns=["CHR", "SNP", "BP", "CHR"],
        )
        names = ["CHR"]
        message = "expected exactly"
    else:
        frame = variants.assign(a=[0.2, -0.3])
        names = ["a"]
        message = "negative"
    frame.to_csv(panel, sep="\t", index=False, compression="gzip")
    with pytest.raises(ValueError, match=message):
        gxe_merge._read_panel_file(
            panel,
            gxe_merge._sha256(panel),
            variants,
            names,
            scratch_dir=tmp_path,
        )


def test_cache_skip_shards_merge_and_fit_equal_monolithic(merged_bundle):
    root, inputs, cache, shards, merged = merged_bundle
    mono = root / "mono.gxe.ref.json"
    for suffix in ("gxx", "gxe", "exg", "gee"):
        left = pd.read_csv(root / f"mono.{suffix}.ldscore.gz", sep=r"\s+")
        right = pd.read_csv(root / f"merged.{suffix}.ldscore.gz", sep=r"\s+")
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=2e-9, atol=2e-9)
    mono_jack = np.load(root / "mono.gxe.jackknife.npz")
    merged_jack = np.load(root / "merged.gxe.jackknife.npz")
    for key in ("xx", "xw", "wx", "ww"):
        np.testing.assert_allclose(
            mono_jack[f"within_{key}"], merged_jack[f"within_{key}"], rtol=2e-13, atol=2e-13
        )

    prefix, env, cov, pheno, _, _ = inputs
    mono_scores = score_phenotype_from_reference(
        reference_manifest=mono,
        bed_path=prefix,
        env_path=env,
        covar_path=cov,
        pheno_path=pheno,
        pheno_col="Y",
        output_prefix=root / "mono-score",
        step_size=13,
    )
    merged_scores = score_phenotype_from_reference(
        reference_manifest=merged,
        bed_path=prefix,
        env_path=env,
        covar_path=cov,
        pheno_path=pheno,
        pheno_col="Y",
        output_prefix=root / "merged-score",
        step_size=13,
    )
    fit_mono, eq_mono = fit_from_files(
        mono, mono_scores.moments, mono_scores.gwas, mono_scores.gwis, max_condition=1e16
    )
    fit_merged, eq_merged = fit_from_files(
        merged, merged_scores.moments, merged_scores.gwas, merged_scores.gwis, max_condition=1e16
    )
    np.testing.assert_allclose(eq_mono.matrix, eq_merged.matrix, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(eq_mono.rhs, eq_merged.rhs, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(fit_mono.proportions, fit_merged.proportions, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(
        fit_mono.jackknife_estimates, fit_merged.jackknife_estimates, rtol=3e-8, atol=3e-8
    )
    np.testing.assert_allclose(fit_mono.standard_errors, fit_merged.standard_errors, rtol=3e-8, atol=3e-8)

    # Marginal phenotype scores are feature-cache quantities, not randomized
    # trace quantities.  A second merge of the identical cache/probes has a
    # different manifest hash/path but must reuse the sealed score triplet.
    alternate = merge_reference_shards(
        shards,
        feature_cache_path=cache,
        output_prefix=root / "merged-alternate",
    )
    fit_alternate, eq_alternate = fit_from_files(
        alternate,
        merged_scores.moments,
        merged_scores.gwas,
        merged_scores.gwis,
        max_condition=1e16,
    )
    np.testing.assert_allclose(eq_alternate.matrix, eq_merged.matrix, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(fit_alternate.proportions, fit_merged.proportions, rtol=0.0, atol=0.0)
    moments_payload = json.loads(merged_scores.moments.read_text(encoding="utf-8"))
    assert moments_payload["feature_cache_sha256"] == json.loads(
        Path(merged).read_text(encoding="utf-8")
    )["feature_cache"]["sha256"]
    nonexistent_cache_reference = root / "nonexistent-cache-reference.json"
    invalid_reference = json.loads(Path(alternate).read_text(encoding="utf-8"))
    invalid_reference["feature_cache"]["path"] = "does-not-exist.gxe.cache.npz"
    nonexistent_cache_reference.write_text(
        json.dumps(invalid_reference), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        fit_from_files(
            nonexistent_cache_reference,
            merged_scores.moments,
            merged_scores.gwas,
            merged_scores.gwis,
            max_condition=1e16,
        )

    for label, num_vectors, override, message in (
        ("string-count", "10", False, "num_vectors"),
        ("string-override", 10, "false", "JSON boolean"),
    ):
        unsafe_reference = root / f"unsafe-{label}.gxe.ref.json"
        unsafe = json.loads(Path(merged).read_text(encoding="utf-8"))
        unsafe["randomization"]["num_vectors"] = num_vectors
        unsafe["randomization"]["low_probe_jackknife_override"] = override
        unsafe_reference.write_text(json.dumps(unsafe), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            fit_from_files(
                unsafe_reference,
                merged_scores.moments,
                merged_scores.gwas,
                merged_scores.gwis,
                max_condition=1e16,
            )

    for path in [cache, merged, *shards, *root.glob("merged.g*")]:
        assert (Path(path).stat().st_mode & 0o777) == 0o600
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        identity = shard.parent / payload["files"]["identity"]
        assert identity.is_file()
        assert (identity.stat().st_mode & 0o777) == 0o600
    assert not list(root.glob(".gxe-jackknife-*"))


def test_duplicate_mixed_and_overwrite_rejected(merged_bundle, monkeypatch):
    root, inputs, cache, shards, _ = merged_bundle
    with pytest.raises(ValueError, match="overlapping probe"):
        merge_reference_shards(
            [shards[0], shards[0]],
            feature_cache_path=cache,
            output_prefix=root / "duplicate",
        )
    with pytest.raises(ValueError, match="must be contiguous"):
        merge_reference_shards(
            [shards[0], shards[2]],
            feature_cache_path=cache,
            output_prefix=root / "gapped",
            allow_low_probe_jackknife=True,
        )
    mixed_estimator = _estimator(
        inputs, root / "mixed", probes=10, offset=100, cache=cache, shard=True, seed=999
    )
    mixed_estimator._compute_ldscore()
    with pytest.raises(ValueError, match="configuration differs"):
        merge_reference_shards(
            [shards[0], root / "mixed.gxe.shard.json"],
            feature_cache_path=cache,
            output_prefix=root / "mixed-merge",
        )
    mixed_step = _estimator(
        inputs, root / "mixed-step", probes=10, offset=100, cache=cache, shard=True
    )
    mixed_step.step_size = 5
    mixed_step._compute_ldscore()
    with pytest.raises(ValueError, match="configuration differs"):
        merge_reference_shards(
            [shards[0], root / "mixed-step.gxe.shard.json"],
            feature_cache_path=cache,
            output_prefix=root / "mixed-step-merge",
        )
    with pytest.raises(ValueError, match="at least 100 merged probes"):
        merge_reference_shards(
            [shards[0]],
            feature_cache_path=cache,
            output_prefix=root / "low-probe",
        )
    diagnostic = merge_reference_shards(
        [shards[0]],
        feature_cache_path=cache,
        output_prefix=root / "low-probe-diagnostic",
        allow_low_probe_jackknife=True,
    )
    diagnostic_manifest = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert diagnostic_manifest["randomization"]["num_vectors"] == 10
    assert diagnostic_manifest["randomization"]["low_probe_jackknife_override"] is True
    monkeypatch.setattr(
        gxe_merge,
        "_read_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing-output preflight must precede cache loading")
        ),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        merge_reference_shards(
            shards,
            feature_cache_path=cache,
            output_prefix=root / "merged",
        )


def test_relabelled_identical_shard_contributions_are_rejected(merged_bundle):
    root, _, cache, shards, _ = merged_bundle
    source_manifest = json.loads(shards[0].read_text(encoding="utf-8"))
    crafted = []
    for index, start in enumerate((100, 110)):
        payload = json.loads(json.dumps(source_manifest))
        payload["randomization"]["probe_offset"] = start
        payload["randomization"]["probe_stop"] = start + 10
        source_identity = shards[0].parent / source_manifest["files"]["identity"]
        identity = json.loads(source_identity.read_text(encoding="utf-8"))
        identity["randomization"]["probe_offset"] = start
        identity["randomization"]["probe_stop"] = start + 10
        identity_path = root / f"relabelled-{index}.identity.json"
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        os.chmod(identity_path, 0o600)
        payload["files"]["identity"] = identity_path.name
        payload["artifact_sha256"]["identity"] = gxe_merge._sha256(identity_path)
        manifest_path = root / f"relabelled-{index}.gxe.shard.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        crafted.append(manifest_path)

    with pytest.raises(ValueError, match="Duplicate shard numerical contributions"):
        merge_reference_shards(
            crafted,
            feature_cache_path=cache,
            output_prefix=root / "relabelled-merge",
            allow_low_probe_jackknife=True,
        )


def test_merge_hashes_and_parses_each_panel_from_same_bytes(merged_bundle, monkeypatch):
    root, _, cache, shards, _ = merged_bundle
    payload = json.loads(shards[0].read_text(encoding="utf-8"))
    panel = (shards[0].parent / payload["files"]["xx"]).resolve()
    original_panel = panel.read_bytes()
    actual_read_csv = gxe_merge.pd.read_csv
    replaced = False

    def replace_after_snapshot(handle, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            # Mutate the source inode after the merger has made its private
            # checked snapshot.  Parsing must consume the snapshot, not reopen
            # or continue reading the mutable source.
            panel.write_bytes(b"concurrent in-place mutation")
        return actual_read_csv(handle, *args, **kwargs)

    monkeypatch.setattr(gxe_merge.pd, "read_csv", replace_after_snapshot)
    try:
        result = merge_reference_shards(
            [shards[0]],
            feature_cache_path=cache,
            output_prefix=root / "read-once",
            allow_low_probe_jackknife=True,
        )
        assert result.is_file()
        assert replaced
    finally:
        panel.write_bytes(original_panel)
        os.chmod(panel, 0o600)


def test_cache_skip_is_two_passes_versus_three(tmp_path):
    inputs = _toy_inputs(tmp_path)
    cache = tmp_path / "features.npz"
    builder = _estimator(inputs, tmp_path / "builder", probes=100)
    builder.write_feature_cache(cache)
    with np.load(cache, allow_pickle=False) as bundle:
        for name in ("scale_x", "scale_w", "norm_x", "norm_w", "diag_nxe_x", "diag_nxe_w"):
            assert bundle[name].dtype == np.float64
    incompatible = _estimator(inputs, tmp_path / "incompatible", probes=100)
    incompatible.kernel_mode = "genie"
    with pytest.raises(ValueError, match="mismatched fields"):
        incompatible._load_feature_cache(cache)
    counts = []
    for name, cache_path in (("uncached", None), ("cached", cache)):
        estimator = _estimator(inputs, tmp_path / name, probes=100, cache=cache_path)
        count = 0
        original = estimator._read_genotype_block

        def counted(self, start, end):
            nonlocal count
            count += 1
            return original(start, end)

        estimator._read_genotype_block = types.MethodType(counted, estimator)
        estimator._compute_ldscore()
        counts.append(count)
    assert counts == [3, 2]


def test_feature_cache_link_failure_does_not_leave_published_cache(tmp_path, monkeypatch):
    inputs = _toy_inputs(tmp_path)
    builder = _estimator(inputs, tmp_path / "builder", probes=10)
    target = tmp_path / "chmod-cache.npz"
    monkeypatch.setattr(
        gwe_ldscore.os,
        "link",
        lambda *_: (_ for _ in ()).throw(OSError("injected cache link failure")),
    )
    with pytest.raises(OSError, match="injected cache link failure"):
        builder.write_feature_cache(target)
    assert not target.exists()
    assert not list(tmp_path.glob(".chmod-cache.npz.*"))


def test_direct_generation_does_not_replace_concurrent_artifact(tmp_path):
    inputs = _toy_inputs(tmp_path)
    cache = tmp_path / "race-cache.npz"
    builder = _estimator(inputs, tmp_path / "race-builder", probes=10)
    builder.write_feature_cache(cache)

    prefix = tmp_path / "race"
    estimator = _estimator(
        inputs, prefix, probes=10, cache=cache, shard=True
    )
    competitor = Path(str(prefix) + ".gxx.ldscore.gz")
    competitor_bytes = b"competing writer\n"
    original_impl = estimator._compute_ldscore_impl

    def create_competitor_after_preflight():
        competitor.write_bytes(competitor_bytes)
        return original_impl()

    estimator._compute_ldscore_impl = create_competitor_after_preflight
    with pytest.raises(FileExistsError, match="concurrently created"):
        estimator._compute_ldscore()
    assert competitor.read_bytes() == competitor_bytes
    assert sorted(tmp_path.glob("race.g*")) == [competitor]
    assert not list(tmp_path.glob(".gxe-bundle-stage-*"))


def test_direct_generation_rollback_preserves_competing_future_manifest(tmp_path, monkeypatch):
    inputs = _toy_inputs(tmp_path)
    cache = tmp_path / "future-cache.npz"
    builder = _estimator(inputs, tmp_path / "future-builder", probes=10)
    builder.write_feature_cache(cache)

    prefix = tmp_path / "future"
    estimator = _estimator(
        inputs, prefix, probes=10, cache=cache, shard=True
    )
    competitor = Path(str(prefix) + ".gxe.shard.json")
    competitor_bytes = b'{"writer":"competitor"}\n'
    actual_link = gwe_ldscore.os.link
    calls = 0

    def publish_and_create_future(source, target):
        nonlocal calls
        result = actual_link(source, target)
        calls += 1
        if calls == 1:
            competitor.write_bytes(competitor_bytes)
        return result

    monkeypatch.setattr(gwe_ldscore.os, "link", publish_and_create_future)
    with pytest.raises(FileExistsError, match="concurrently created"):
        estimator._compute_ldscore()
    assert competitor.read_bytes() == competitor_bytes
    assert sorted(tmp_path.glob("future.g*")) == [competitor]
    assert not list(tmp_path.glob(".gxe-bundle-stage-*"))


def test_merge_publication_failure_rolls_back_complete_bundle(merged_bundle, monkeypatch):
    root, _, cache, shards, _ = merged_bundle
    prefix = root / "publish-failure"
    actual_link = gxe_merge.os.link
    calls = 0

    def fail_third_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected publication failure")
        return actual_link(source, target)

    monkeypatch.setattr(gxe_merge.os, "link", fail_third_link)
    with pytest.raises(OSError, match="injected publication failure"):
        merge_reference_shards(
            shards,
            feature_cache_path=cache,
            output_prefix=prefix,
        )
    assert not list(root.glob("publish-failure.g*"))
    assert not list(root.glob(".gxe-merge-stage-*"))


def test_merge_rollback_does_not_delete_replacement_writer(merged_bundle, monkeypatch):
    root, _, cache, shards, _ = merged_bundle
    prefix = root / "replacement-race"
    competitor = root / "replacement-race.gxx.ldscore.gz"
    competitor_bytes = b"replacement writer\n"
    actual_link = gxe_merge.os.link
    calls = 0

    def replace_first_then_fail(source, target):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected later publication failure")
        result = actual_link(source, target)
        if calls == 1:
            replacement = root / ".replacement-writer.tmp"
            replacement.write_bytes(competitor_bytes)
            os.replace(replacement, target)
        return result

    monkeypatch.setattr(gxe_merge.os, "link", replace_first_then_fail)
    with pytest.raises(OSError, match="injected later publication failure"):
        merge_reference_shards(
            shards,
            feature_cache_path=cache,
            output_prefix=prefix,
        )
    assert competitor.read_bytes() == competitor_bytes
    assert sorted(root.glob("replacement-race.g*")) == [competitor]
    assert not list(root.glob(".gxe-merge-stage-*"))


def test_merge_does_not_path_chmod_published_artifact(merged_bundle, monkeypatch):
    root, _, cache, shards, _ = merged_bundle
    prefix = root / "chmod-failure"
    actual_chmod = gxe_merge.os.chmod

    def reject_final_chmod(path, mode):
        if Path(path).name == "chmod-failure.gxx.ldscore.gz":
            raise AssertionError("published path must inherit private staged mode")
        return actual_chmod(path, mode)

    monkeypatch.setattr(gxe_merge.os, "chmod", reject_final_chmod)
    result = merge_reference_shards(
        shards,
        feature_cache_path=cache,
        output_prefix=prefix,
    )
    assert result.is_file()
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in root.glob("chmod-failure.g*"))
    assert not list(root.glob(".gxe-merge-stage-*"))
