from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import to_bed

from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore
from summit.ldscore import gxe_multi
from summit.ldscore.gxe_multi import generate_multi_environment_references
from summit.logger import Logger


ENVIRONMENTS = ("continuous", "binary", "correlated", "skewed", "collinear")


def test_environment_factorization_does_not_resize_blas_pool() -> None:
    rng = np.random.default_rng(1976)
    common_input = rng.normal(size=(61, 3))
    common_input -= common_input.mean(axis=0, keepdims=True)
    common, _ = np.linalg.qr(common_input)
    environment = rng.normal(size=61)
    environment -= environment.mean()
    environment -= common @ (common.T @ environment)
    environment /= np.linalg.norm(environment)
    full = np.column_stack([common, environment])
    estimator = types.SimpleNamespace(
        C_common=np.asfortranarray(common),
        C_int=np.asfortranarray(full),
        env=environment,
        env_name="observed",
        nsamp=61,
        p_eff=4,
    )
    assert not hasattr(gxe_multi, "threadpool_limits")
    observed_common, observed_directions = gxe_multi._environment_directions(
        [estimator]
    )
    np.testing.assert_allclose(observed_common, common, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.abs(observed_directions[:, 0]), np.abs(environment), rtol=0.0, atol=2e-15
    )


def _inputs(root: Path) -> tuple[Path, Path, Path, Path, int]:
    rng = np.random.default_rng(44551)
    n, m = 37, 11
    raw = rng.binomial(2, rng.uniform(0.12, 0.46, m), size=(n, m)).astype(float)
    prefix = root / "genotype"
    to_bed(str(prefix) + ".bed", raw)
    fam = pd.read_csv(str(prefix) + ".fam", sep=r"\s+", header=None)
    ids = {"FID": fam[0].astype(str), "IID": fam[1].astype(str)}
    common = rng.normal(size=n)
    continuous = rng.normal(size=n)
    environment = root / "environment.tsv"
    pd.DataFrame(
        {
            **ids,
            "continuous": continuous,
            "binary": (rng.uniform(size=n) < 0.38).astype(float),
            "correlated": 0.7 * continuous + rng.normal(scale=0.5, size=n),
            "skewed": rng.lognormal(mean=0.0, sigma=0.8, size=n),
            "collinear": common,
        }
    ).to_csv(environment, sep="\t", index=False)
    covariates = root / "covariates.tsv"
    pd.DataFrame({**ids, "C": common, "C_DUP": common}).to_csv(
        covariates, sep="\t", index=False
    )
    bim = pd.read_csv(str(prefix) + ".bim", sep=r"\s+", header=None)
    annotations = root / "annotations.tsv"
    pd.DataFrame(
        {
            "CHR": bim[0].astype(str),
            "SNP": bim[1].astype(str),
            "BP": bim[3].astype(int),
            "broad": np.linspace(0.2, 1.0, m),
            "overlap": 0.35 + 0.65 * (np.arange(m) % 3 == 0),
        }
    ).to_csv(annotations, sep="\t", index=False)
    return prefix, environment, covariates, annotations, m


def _estimator(
    prefix: Path,
    environment: Path,
    covariates: Path,
    annotations: Path,
    output: Path,
    column: str,
    *,
    memory_gib: float = 0.01,
) -> GenomewideEnvLDScore:
    return GenomewideEnvLDScore(
        bed_path=str(prefix),
        env_path=str(environment),
        env_col=column,
        annot_path=str(annotations),
        out_path=str(output),
        covar_path=str(covariates),
        log=Logger(suppress=True),
        rand_dist="rademacher",
        low_level=None,
        num_vecs=3,
        step_size=4,
        seed=781,
        dtype="float64",
        num_threads=2,
        target_xz_mem=memory_gib,
        kernel_mode="standardized_projected",
        genotype_scale="sample",
        native_backend="python",
    )


def _score(prefix: Path, suffix: str) -> np.ndarray:
    return pd.read_csv(f"{prefix}.{suffix}.ldscore.gz", sep="\t")[
        ["broad", "overlap"]
    ].to_numpy()


def _population(prefix: Path) -> np.ndarray:
    payload = json.loads(Path(f"{prefix}.gxe.ref.json").read_text())
    return np.asarray(
        payload["population_trace"]["same_individual_kernel_products"],
        dtype=np.float64,
    )


def _run_fused(
    prefix: Path,
    environment: Path,
    covariates: Path,
    annotations: Path,
    root: Path,
    *,
    memory_gib: float,
    full_precision_layout: str | None = None,
) -> tuple[Path, list[int]]:
    estimators = [
        _estimator(
            prefix,
            environment,
            covariates,
            annotations,
            root / name,
            name,
            memory_gib=memory_gib,
        )
        for name in ENVIRONMENTS
    ]
    reads = [0, 0, 0, 0, 0]
    for index, estimator in enumerate(estimators):
        original = estimator._read_genotype_block

        def counted(
            self,
            start,
            stop,
            *,
            _index=index,
            _original=original,
            **read_options,
        ):
            reads[_index] += 1
            return _original(start, stop, **read_options)

        estimator._read_genotype_block = types.MethodType(counted, estimator)
    try:
        options = {}
        if full_precision_layout is not None:
            options["full_precision_layout"] = full_precision_layout
        manifest = generate_multi_environment_references(
            estimators,
            batch_manifest=root / "batch.gxe.multi.json",
            requested_backend="direct",
            **options,
        )
    finally:
        for estimator in estimators:
            estimator.close()
    return manifest, reads


def test_five_environment_fusion_matches_independent_oracles(tmp_path: Path) -> None:
    prefix, environment, covariates, annotations, variants = _inputs(tmp_path)
    independent_root = tmp_path / "independent"
    fused_root = tmp_path / "fused"
    independent_root.mkdir()
    fused_root.mkdir()
    for name in ENVIRONMENTS:
        estimator = _estimator(
            prefix,
            environment,
            covariates,
            annotations,
            independent_root / name,
            name,
        )
        try:
            estimator._compute_ldscore()
        finally:
            estimator.close()

    manifest, reads = _run_fused(
        prefix,
        environment,
        covariates,
        annotations,
        fused_root,
        memory_gib=0.01,
    )
    payload = json.loads(manifest.read_text())
    assert payload["environment_tiles"] == [[0, 5]]
    assert payload["shared_genotype_passes"] == 2
    # The descriptor-owned context decodes every planned pass internally; no
    # Python estimator materializes a genotype block.
    assert reads == [0, 0, 0, 0, 0]
    # Packed Mailman kernels own feature/source/target products. BLAS is used
    # only for the two small common-basis projection calls for this one tile.
    assert payload["fused_gemm_calls"] == {"nn": 1, "tn": 1, "total": 2}
    assert sum(item["calls"] for item in payload["fused_gemm_shapes"]) == 2
    assert payload["fused_gemm_total_flops"] == sum(
        item["flops"] for item in payload["fused_gemm_shapes"]
    )
    assert payload["fused_gemm_total_flops"] > 0
    assert payload["peak_process_rss_gib_at_manifest"] > 0.0
    assert payload["peak_rss_scope"] == (
        "process_lifetime_ru_maxrss_at_batch_manifest"
    )
    assert payload["repaired_gemm_output_columns"] == 0
    performance = payload["performance_telemetry"]
    kernel = performance["multi_environment_native_kernel"]
    assert performance["multi_environment_native_kernel_complete"] is True
    assert kernel["environment_count"] == len(ENVIRONMENTS)
    assert kernel["common_basis_rank"] == 1
    assert kernel["normalization_calls"] == 1
    output_status = performance["native_gemm_output_numa_evidence_status"]
    # Packed feature/source/target kernels reuse caller-owned buffers. Only the
    # projection coefficient result crosses the protected native-output owner.
    expected_native_outputs = kernel["projection_calls"]
    assert output_status["attempted_calls"] == expected_native_outputs
    assert output_status["captured_records"] == expected_native_outputs
    assert output_status["failed_calls"] == 0

    for name in ENVIRONMENTS:
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(fused_root / name, suffix),
                _score(independent_root / name, suffix),
                rtol=3e-12,
                atol=3e-12,
            )
        np.testing.assert_allclose(
            _population(fused_root / name),
            _population(independent_root / name),
            rtol=3e-12,
            atol=3e-12,
        )
    assert not list(fused_root.rglob("*.npz"))
    assert not list(fused_root.rglob("*cache*"))
    assert not list(fused_root.rglob("*sketch*"))


def test_environment_tiling_is_deterministic_and_preserves_outputs(
    tmp_path: Path,
) -> None:
    prefix, environment, covariates, annotations, variants = _inputs(tmp_path)
    one_tile = tmp_path / "one-tile"
    tiled = tmp_path / "tiled"
    one_tile.mkdir()
    tiled.mkdir()
    _, _ = _run_fused(
        prefix,
        environment,
        covariates,
        annotations,
        one_tile,
        memory_gib=0.01,
    )
    # The one-panel source and generated target view keep all probes together.
    # The small budget still forces five deterministic environment tiles, for
    # 1 + 2*5 genotype passes.
    budget_gib = 3600 / 1024**3
    manifest, reads = _run_fused(
        prefix,
        environment,
        covariates,
        annotations,
        tiled,
        memory_gib=budget_gib,
    )
    payload = json.loads(manifest.read_text())
    assert payload["environment_tiles"] == [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]
    ]
    assert payload["randomization"]["probe_tiles"] == [[0, 3]]
    assert payload["shared_genotype_passes"] == 11
    assert reads == [0, 0, 0, 0, 0]
    for name in ENVIRONMENTS:
        for suffix in ("gxx", "gxe", "exg", "gee"):
            np.testing.assert_allclose(
                _score(tiled / name, suffix),
                _score(one_tile / name, suffix),
                rtol=3e-12,
                atol=3e-12,
            )
        np.testing.assert_allclose(
            _population(tiled / name),
            _population(one_tile / name),
            rtol=3e-12,
            atol=3e-12,
        )


def test_explicit_current_layout_matches_difficult_five_environment_oracle_across_tiles(
    tmp_path: Path,
) -> None:
    prefix, environment, covariates, annotations, variants = _inputs(tmp_path)
    default = tmp_path / "default"
    explicit_one_tile = tmp_path / "explicit-one-tile"
    explicit_tiled = tmp_path / "explicit-tiled"
    default.mkdir()
    explicit_one_tile.mkdir()
    explicit_tiled.mkdir()

    default_manifest, default_reads = _run_fused(
        prefix,
        environment,
        covariates,
        annotations,
        default,
        memory_gib=0.01,
    )
    explicit_manifest, explicit_reads = _run_fused(
        prefix,
        environment,
        covariates,
        annotations,
        explicit_one_tile,
        memory_gib=0.01,
        full_precision_layout="current",
    )
    budget_gib = 3600 / 1024**3
    tiled_manifest, tiled_reads = _run_fused(
        prefix,
        environment,
        covariates,
        annotations,
        explicit_tiled,
        memory_gib=budget_gib,
        full_precision_layout="current",
    )

    default_payload = json.loads(default_manifest.read_text())
    explicit_payload = json.loads(explicit_manifest.read_text())
    tiled_payload = json.loads(tiled_manifest.read_text())
    assert default_payload["full_precision_layout"] == "current"
    assert explicit_payload["full_precision_layout"] == "current"
    assert tiled_payload["full_precision_layout"] == "current"
    assert default_payload["environment_tiles"] == [[0, 5]]
    assert explicit_payload["environment_tiles"] == [[0, 5]]
    assert default_payload["shared_genotype_passes"] == 2
    assert explicit_payload["shared_genotype_passes"] == 2
    assert default_reads == [0, 0, 0, 0, 0]
    assert explicit_reads == [0, 0, 0, 0, 0]
    assert tiled_payload["environment_tiles"] == [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ]
    assert tiled_payload["randomization"]["probe_tiles"] == [[0, 3]]
    assert tiled_payload["shared_genotype_passes"] == 11
    assert tiled_reads == [0, 0, 0, 0, 0]
    for payload in (default_payload, explicit_payload, tiled_payload):
        assert payload["source_panel_memory_order"] == "F"
        assert payload["target_panel_memory_order"] == "F"
        assert payload["target_genotype_memory_order"] == "F"
        assert payload["source_to_target_layout_transition"] == "none"
        assert payload["modeled_target_pair_sealing_live_peak_gib"] == (
            payload["modeled_packed_source_panel_gib"]
        )

    for name in ENVIRONMENTS:
        for observed_root in (explicit_one_tile, explicit_tiled):
            for suffix in ("gxx", "gxe", "exg", "gee"):
                np.testing.assert_allclose(
                    _score(observed_root / name, suffix),
                    _score(default / name, suffix),
                    rtol=5e-12,
                    atol=5e-12,
                )
            np.testing.assert_allclose(
                _population(observed_root / name),
                _population(default / name),
                rtol=5e-12,
                atol=5e-12,
            )
