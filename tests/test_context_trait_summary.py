from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from summit.context import (
    ContextComponentIndex,
    ContextPairIndex,
    array_sha256,
    batched_contextual_scores,
    build_context_trait_summary,
    canonical_sha256,
    common_scale_features,
    contextual_scores,
    dense_genetic_kernels,
    dense_residual_kernels,
    genetic_residual_cross_traces,
    genetic_rhs_from_scores,
    genetic_trace_from_features,
    kernel_rhs,
    kernel_traces,
    load_context_trait_summary,
    project_normalize_phenotype,
    rank_revealing_projector,
    residual_moments_low_rank,
    scale_aware_max_discrepancy,
    trait_moments_after_deleting_groups,
    write_context_trait_summary,
)


def _fixture(
    q: int,
    *,
    seed: int = 1,
    n: int = 36,
    m: int = 23,
    residual_mode: str = "current",
) -> dict[str, object]:
    rng = np.random.default_rng(seed + q)
    genotype = rng.normal(size=(n, m))
    genotype -= genotype.mean(axis=0)
    genotype /= genotype.std(axis=0, ddof=1)
    environment = rng.normal(size=n)
    basis = np.ones((n, q))
    if q > 1:
        basis[:, 1:] = rng.normal(size=(n, q - 1))
        basis[:, 1] = environment
    fixed = np.column_stack([np.ones(n), basis[:, 1:], rng.normal(size=n)])
    projector = rank_revealing_projector(fixed)
    phenotype = rng.normal(size=n)
    annotations = np.column_stack(
        [np.arange(m) < m // 2, np.arange(m) >= m // 2]
    ).astype(np.float64)
    components = ContextComponentIndex(("left", "right"), ContextPairIndex(q))
    if residual_mode == "homoskedastic":
        residual_basis = np.ones((n, 1))
        residual_names = ("constant",)
    elif residual_mode == "binary":
        binary = (environment > np.median(environment)).astype(np.float64)
        residual_basis = np.column_stack([1.0 - binary, binary])
        residual_names = ("stratum:0", "stratum:1")
    else:
        residual_basis = np.column_stack([np.ones(n), environment**2])
        residual_names = ("constant", "environment_squared")
    groups = tuple(f"group:{index // 3}" for index in range(m))
    return {
        "genotype": genotype,
        "basis": basis,
        "fixed": fixed,
        "projector": projector,
        "phenotype": phenotype,
        "annotations": annotations,
        "components": components,
        "residual_basis": residual_basis,
        "residual_names": residual_names,
        "groups": groups,
    }


def _build(fixture: dict[str, object], *, block_size: int = 7):
    return build_context_trait_summary(
        genotype=fixture["genotype"],
        basis=fixture["basis"],
        phenotype=fixture["phenotype"],
        projector=fixture["projector"],
        annotations=fixture["annotations"],
        component_index=fixture["components"],
        residual_basis=fixture["residual_basis"],
        residual_names=fixture["residual_names"],
        basis_hash=array_sha256(fixture["basis"]),
        fixed_effect_hash=array_sha256(fixture["fixed"]),
        variant_hash=canonical_sha256(
            {"variants": list(range(np.asarray(fixture["genotype"]).shape[1]))}
        ),
        loo_groups=fixture["groups"],
        block_size=block_size,
    )


@pytest.mark.parametrize("q", [1, 2, 3, 4])
@pytest.mark.parametrize("residual_mode", ["homoskedastic", "current", "binary"])
def test_trait_summary_reconstructs_dense_nonreference_system(
    q: int, residual_mode: str
) -> None:
    fixture = _fixture(q, seed=100, residual_mode=residual_mode)
    summary = _build(fixture)
    projector = fixture["projector"]
    y = project_normalize_phenotype(fixture["phenotype"], projector)
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], projector.projector
    )
    genetic_kernels = dense_genetic_kernels(
        features, fixture["annotations"], fixture["components"]
    )
    residual_kernels = dense_residual_kernels(
        projector.projector, fixture["residual_basis"]
    )
    assert (
        scale_aware_max_discrepancy(summary.genetic_rhs, kernel_rhs(genetic_kernels, y))
        < 1e-12
    )
    assert (
        scale_aware_max_discrepancy(
            summary.genetic_traces, kernel_traces(genetic_kernels)
        )
        < 1e-12
    )
    dense_cross = np.einsum(
        "aij,hij->ah", genetic_kernels, residual_kernels, optimize=True
    )
    assert scale_aware_max_discrepancy(summary.genetic_residual, dense_cross) < 1e-12
    residual = residual_moments_low_rank(
        projector.fixed_basis, fixture["residual_basis"], y
    )
    np.testing.assert_allclose(summary.residual_rhs, residual.rhs, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        summary.residual_traces, residual.traces, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(summary.residual_gram, residual.gram, rtol=0.0, atol=0.0)
    assert summary.decode_passes == 1
    assert summary.decoded_blocks == 4
    assert summary.manifest["feature_mode"] == "raw_projected"
    assert summary.manifest["genotype_scaling"] == "pre_scaled_input"
    assert summary.manifest["approximate_loo"]["exact_deleted_kernels"] is False


def test_snp_contributions_sum_and_group_deletion_renormalizes_every_term() -> None:
    fixture = _fixture(3, seed=222, m=24, residual_mode="binary")
    summary = _build(fixture, block_size=5)
    component_masses = np.asarray(
        [
            summary.annotation_masses[entry.annotation_index]
            for entry in summary.component_index.entries
        ]
    )
    np.testing.assert_allclose(
        summary.rhs_numerator_contributions.sum(axis=0) / component_masses,
        summary.genetic_rhs,
        rtol=2e-15,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        summary.trace_numerator_contributions.sum(axis=0) / component_masses,
        summary.genetic_traces,
        rtol=2e-15,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        summary.genetic_residual_numerator_contributions.sum(axis=0)
        / component_masses[:, None],
        summary.genetic_residual,
        rtol=2e-15,
        atol=2e-13,
    )

    deleted_group = "group:2"
    deleted = np.asarray(summary.loo_group_ids) == deleted_group
    keep = ~deleted
    observed = trait_moments_after_deleting_groups(summary, [deleted_group])
    projector = fixture["projector"]
    y = project_normalize_phenotype(fixture["phenotype"], projector)
    features = common_scale_features(
        np.asarray(fixture["genotype"])[:, keep],
        fixture["basis"],
        projector.projector,
    )
    annotations = np.asarray(fixture["annotations"])[keep]
    scores = contextual_scores(features, y, projector.residual_rank)
    expected_rhs = genetic_rhs_from_scores(
        scores, annotations, fixture["components"], projector.residual_rank
    )
    expected_traces = genetic_trace_from_features(
        features, annotations, fixture["components"]
    )
    expected_cross = genetic_residual_cross_traces(
        features, fixture["residual_basis"], annotations, fixture["components"]
    )
    np.testing.assert_allclose(
        observed.genetic_rhs, expected_rhs, rtol=2e-14, atol=2e-13
    )
    np.testing.assert_allclose(
        observed.genetic_traces, expected_traces, rtol=2e-14, atol=2e-13
    )
    np.testing.assert_allclose(
        observed.genetic_residual, expected_cross, rtol=2e-14, atol=2e-13
    )
    np.testing.assert_array_equal(observed.residual_rhs, summary.residual_rhs)
    np.testing.assert_array_equal(observed.residual_gram, summary.residual_gram)


def test_batched_q_times_l_scores_equal_separate_score_generation() -> None:
    fixture = _fixture(4, seed=301, n=40, m=31)
    projector = fixture["projector"]
    rng = np.random.default_rng(302)
    phenotypes = np.column_stack(
        [project_normalize_phenotype(rng.normal(size=40), projector) for _ in range(3)]
    )
    ranks = np.full(3, projector.residual_rank)
    observed = batched_contextual_scores(
        fixture["genotype"], fixture["basis"], phenotypes, ranks, block_size=6
    )
    features = common_scale_features(
        fixture["genotype"], fixture["basis"], projector.projector
    )
    expected = np.stack(
        [
            contextual_scores(features, phenotypes[:, index], projector.residual_rank)
            for index in range(3)
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=3e-15, atol=3e-14)


def test_trait_summary_round_trip_is_hash_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(2, seed=401)
    summary = _build(fixture)
    manifest, arrays = write_context_trait_summary(summary, tmp_path / "trait")
    loaded = load_context_trait_summary(
        manifest, expected={"basis_hash": summary.manifest["basis_hash"]}
    )
    np.testing.assert_array_equal(loaded.genetic_rhs, summary.genetic_rhs)
    np.testing.assert_array_equal(
        loaded.rhs_numerator_contributions, summary.rhs_numerator_contributions
    )
    assert loaded.component_index.names == summary.component_index.names
    with pytest.raises(ValueError, match="mismatch"):
        load_context_trait_summary(
            manifest, expected={"variant_hash": canonical_sha256({"wrong": True})}
        )
    payload = json.loads(manifest.read_text())
    payload["artifact"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_context_trait_summary(manifest)
    assert arrays.stat().st_size > 0


def test_trait_summary_rejects_nonfinite_or_zero_scale_genotype() -> None:
    fixture = _fixture(2, seed=501)
    bad = np.array(fixture["genotype"], copy=True)
    bad[:, 0] = 0.0
    fixture["genotype"] = bad
    with pytest.raises(ValueError, match="near-zero scale"):
        _build(fixture)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _build(fixture)


def test_small_trait_benchmark_records_time_memory_and_output_size(
    tmp_path: Path,
) -> None:
    fixture = _fixture(4, seed=601, n=100, m=200)
    summary = _build(fixture, block_size=32)
    manifest, arrays = write_context_trait_summary(summary, tmp_path / "benchmark")
    assert summary.phase_times_seconds["genotype_pass"] > 0.0
    assert (
        summary.phase_times_seconds["total"]
        >= summary.phase_times_seconds["genotype_pass"]
    )
    assert summary.peak_rss_bytes > 0
    assert summary.decoded_blocks == 7
    assert manifest.stat().st_size > 0
    assert arrays.stat().st_size > 0
