from __future__ import annotations

import numpy as np

from summit.ldscore.generalized_gxe_trait_summary import (
    aggregate_generalized_gxe_trait_statistics,
    generalized_gxe_per_variant_trait_statistics,
    load_generalized_gxe_trait_summary,
    write_generalized_gxe_trait_summary,
)


def test_low_rank_per_variant_statistics_equal_explicit_projected_features() -> None:
    rng = np.random.default_rng(20260824)
    n, m, q, rank, traits, residuals = 41, 29, 3, 5, 2, 4
    genotype = rng.normal(size=(n, m))
    fixed, _ = np.linalg.qr(rng.normal(size=(n, rank)), mode="reduced")
    basis = rng.normal(size=(n, q))
    phenotype = rng.normal(size=(n, traits))
    residual_basis = rng.normal(size=(n, residuals))

    observed = generalized_gxe_per_variant_trait_statistics(
        genotype=genotype,
        basis=basis,
        fixed_basis=fixed,
        phenotypes=phenotype,
        residual_basis=residual_basis,
    )

    projected_y = phenotype - fixed @ (fixed.T @ phenotype)
    projected_y *= np.sqrt(
        (n - rank) / np.sum(projected_y * projected_y, axis=0)
    )[None, :]
    features = np.stack(
        [
            basis[:, index, None] * genotype
            - fixed @ (fixed.T @ (basis[:, index, None] * genotype))
            for index in range(q)
        ]
    )
    expected_scores = np.stack(
        [genotype.T @ (basis[:, index, None] * projected_y) for index in range(q)],
        axis=1,
    )
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    expected_information = np.column_stack(
        [
            np.einsum("nm,nm->m", features[left], features[right])
            for left, right in pairs
        ]
    )
    expected_residual = np.stack(
        [
            np.column_stack(
                [
                    np.einsum(
                        "nm,n,nm->m",
                        features[left],
                        residual_basis[:, h],
                        features[right],
                    )
                    for h in range(residuals)
                ]
            )
            for left, right in pairs
        ],
        axis=1,
    )
    np.testing.assert_allclose(observed.normalized_phenotypes, projected_y)
    np.testing.assert_allclose(observed.scores, expected_scores, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        observed.information, expected_information, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        observed.residual_information,
        expected_residual,
        rtol=3e-12,
        atol=3e-12,
    )


def test_block_aggregation_retains_exact_per_snp_numerators() -> None:
    rng = np.random.default_rng(11)
    n, m = 31, 23
    genotype = rng.normal(size=(n, m))
    fixed, _ = np.linalg.qr(rng.normal(size=(n, 4)), mode="reduced")
    basis = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    residual_basis = np.column_stack(
        [np.ones(n), basis[:, 1], basis[:, 2], basis[:, 1] ** 2]
    )
    statistics = generalized_gxe_per_variant_trait_statistics(
        genotype=genotype,
        basis=basis,
        fixed_basis=fixed,
        phenotypes=rng.normal(size=(n, 2)),
        residual_basis=residual_basis,
    )
    annotations = rng.uniform(0.2, 1.0, size=(m, 2))
    group_ids = np.arange(m) % 5
    summary = aggregate_generalized_gxe_trait_statistics(
        statistics,
        annotations=annotations,
        annotation_names=("a", "b"),
        variant_group_ids=group_ids,
        group_labels=tuple(f"g{index}" for index in range(5)),
        trait_ids=("t0", "t1"),
        residual_names=tuple(f"r{index}" for index in range(4)),
        n_samples=n,
    )
    np.testing.assert_allclose(
        np.sum(summary.group_rhs_unnormalized_num, axis=0),
        summary.genetic_rhs
        * np.repeat(summary.annotation_masses, 6)[:, None],
    )
    np.testing.assert_allclose(
        np.sum(summary.group_trace_unnormalized_num, axis=0),
        summary.genetic_traces * np.repeat(summary.annotation_masses, 6),
    )
    np.testing.assert_allclose(
        np.sum(summary.group_genetic_residual_num, axis=0),
        summary.genetic_residual
        * np.repeat(summary.annotation_masses, 6)[:, None],
    )
    assert summary.per_variant is statistics


def test_hash_free_trait_summary_roundtrip_loads_compact_moments(tmp_path) -> None:
    rng = np.random.default_rng(918)
    n, m = 17, 13
    fixed, _ = np.linalg.qr(rng.normal(size=(n, 3)), mode="reduced")
    basis = np.column_stack([np.ones(n), rng.normal(size=n)])
    statistics = generalized_gxe_per_variant_trait_statistics(
        genotype=rng.normal(size=(n, m)),
        basis=basis,
        fixed_basis=fixed,
        phenotypes=rng.normal(size=(n, 2)),
        residual_basis=np.column_stack([np.ones(n), basis[:, 1]]),
    )
    summary = aggregate_generalized_gxe_trait_statistics(
        statistics,
        annotations=np.ones((m, 1)),
        annotation_names=("all",),
        variant_group_ids=np.arange(m) % 3,
        group_labels=("g0", "g1", "g2"),
        trait_ids=("t0", "t1"),
        residual_names=("r0", "r1"),
        n_samples=n,
    )
    path = write_generalized_gxe_trait_summary(summary, tmp_path / "trait")
    loaded = load_generalized_gxe_trait_summary(path)
    assert loaded.n_samples == summary.n_samples
    assert loaded.n_variants == summary.n_variants
    assert loaded.component_index.entries == summary.component_index.entries
    assert loaded.group_ids == summary.group_ids
    assert loaded.trait_ids == summary.trait_ids
    assert loaded.per_variant is None
    for name in (
        "genetic_rhs",
        "genetic_traces",
        "genetic_residual",
        "residual_rhs",
        "residual_traces",
        "residual_gram",
        "group_rhs_unnormalized_num",
        "group_trace_unnormalized_num",
        "group_genetic_residual_num",
        "annotation_masses",
        "group_annotation_masses",
        "group_variant_counts",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(summary, name))
