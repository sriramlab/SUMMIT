from __future__ import annotations

import numpy as np
import pytest

from generalized_gxe_variant_ldscore_oracle import (
    contextual_features,
    dense_kernel_gram,
    exact_directional_ldscores,
    exact_recomputed_delete_block_grams,
    exact_same_person_matrix,
    frozen_ldscore_delete_block,
    normalized_kernels,
    orthonormalize,
    pair_order,
    pass1_sources,
    pass2_cross_sketches,
    randomized_directional_products,
    randomized_two_pass_ldscores,
    randomized_two_pass_ldscores_streaming,
    same_person_ustatistic,
    snp_atoms,
    trait_rhs_and_traces,
)
from summit.context.spec import ContextPairIndex
from summit.ldscore.gwe_ldscore import GenomewideEnvLDScore


def _fixture(
    num_basis: int,
    num_annotations: int,
    *,
    seed: int,
    probe_count: int = 31,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    n_samples, n_variants = 11, 10
    environment = rng.normal(size=n_samples)
    covariate = rng.normal(size=n_samples)
    fixed_basis = orthonormalize(
        np.column_stack([np.ones(n_samples), covariate])
    )
    raw = rng.normal(size=(n_samples, n_variants))
    # Deliberately correlate G with the context and nuisance directions.
    genotype = (
        raw
        + 0.45 * environment[:, None] * rng.normal(size=(1, n_variants))
        + 0.25 * covariate[:, None]
    )
    basis_columns = [np.ones(n_samples), environment]
    if num_basis >= 3:
        basis_columns.append(0.4 * environment**2 + rng.normal(size=n_samples))
    basis = np.column_stack(basis_columns[:num_basis])
    annotations = rng.uniform(
        0.15,
        1.4,
        size=(n_variants, num_annotations),
    )
    probes = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(n_variants, probe_count),
    )
    phenotype = rng.normal(size=n_samples)
    return genotype, basis, fixed_basis, annotations, probes, phenotype


def test_pair_order_matches_schema_and_has_independent_q3_anchor() -> None:
    expected = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    assert pair_order(3) == expected
    schema_pairs = tuple(
        (entry.q, entry.r) for entry in ContextPairIndex(num_basis=3).entries
    )
    assert schema_pairs == expected
    assert tuple(
        entry.kernel_factor for entry in ContextPairIndex(num_basis=3).entries
    ) == (1, 1, 1, 2, 2, 2)


@pytest.mark.parametrize(
    ("num_basis", "num_annotations", "seed"),
    ((1, 1, 101), (2, 1, 202), (3, 1, 303), (3, 2, 304)),
)
def test_exact_per_snp_aggregation_reconstructs_dense_kernel_gram(
    num_basis: int,
    num_annotations: int,
    seed: int,
) -> None:
    genotype, basis, fixed_basis, annotations, _probes, _phenotype = _fixture(
        num_basis,
        num_annotations,
        seed=seed,
    )
    residual_rank = genotype.shape[0] - fixed_basis.shape[1]
    exact = exact_directional_ldscores(
        contextual_features(genotype, basis, fixed_basis),
        annotations,
        residual_rank,
    )
    dense = dense_kernel_gram(genotype, basis, fixed_basis, annotations)
    np.testing.assert_allclose(exact.gram, dense, rtol=1.0e-11, atol=1.0e-11)


def test_q1_additive_score_is_squared_feature_correlation_sum() -> None:
    genotype, basis, fixed_basis, annotations, _probes, _phenotype = _fixture(
        1,
        1,
        seed=411,
    )
    features = contextual_features(genotype, basis, fixed_basis)
    residual_rank = genotype.shape[0] - fixed_basis.shape[1]
    correlation = features[0].T @ features[0] / float(residual_rank)
    expected = (correlation * correlation) @ annotations[:, 0]
    exact = exact_directional_ldscores(features, annotations, residual_rank)
    np.testing.assert_allclose(
        exact.directional_ldscores[:, 0, 0],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_q2_diagonal_offdiagonal_factors_are_one_two_four() -> None:
    rng = np.random.default_rng(765)
    n_samples, n_variants = 9, 6
    genotype = rng.normal(size=(n_samples, n_variants))
    environment = rng.normal(size=n_samples)
    basis = np.column_stack([environment, environment])
    fixed_basis = orthonormalize(np.ones((n_samples, 1)))
    annotations = np.ones((n_variants, 1))
    gram = dense_kernel_gram(genotype, basis, fixed_basis, annotations)
    expected_ratio = np.asarray(
        [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 4.0]]
    )
    np.testing.assert_allclose(
        gram / gram[0, 0],
        expected_ratio,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_snp_atoms_kernels_trait_rhs_and_traces_are_literal_dense_objects() -> None:
    genotype, basis, fixed_basis, annotations, _probes, phenotype = _fixture(
        3,
        2,
        seed=501,
    )
    features = contextual_features(genotype, basis, fixed_basis)
    pairs = pair_order(3)
    atoms = snp_atoms(features, pairs)
    kernels, masses, component_annotation, component_pair = normalized_kernels(
        features,
        annotations,
        pairs,
    )
    for component, (annotation, pair_index) in enumerate(
        zip(component_annotation, component_pair, strict=True)
    ):
        expected = np.zeros_like(kernels[component])
        for variant in range(genotype.shape[1]):
            expected += annotations[variant, annotation] * atoms[variant, pair_index]
        expected /= masses[annotation]
        np.testing.assert_allclose(kernels[component], expected, atol=1.0e-12)
    rhs, traces = trait_rhs_and_traces(kernels, phenotype)
    literal_rhs = np.asarray([phenotype @ kernel @ phenotype for kernel in kernels])
    literal_traces = np.asarray(
        [sum(kernel[i, i] for i in range(kernel.shape[0])) for kernel in kernels]
    )
    np.testing.assert_allclose(rhs, literal_rhs, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(traces, literal_traces, rtol=1.0e-13, atol=1.0e-13)


def test_signed_offdiagonal_directional_scores_are_preserved() -> None:
    genotype, basis, fixed_basis, annotations, _probes, _phenotype = _fixture(
        3,
        2,
        seed=20260821,
    )
    residual_rank = genotype.shape[0] - fixed_basis.shape[1]
    exact = exact_directional_ldscores(
        contextual_features(genotype, basis, fixed_basis),
        annotations,
        residual_rank,
    )
    offdiagonal = slice(basis.shape[1], None)
    assert np.min(exact.directional_ldscores[:, offdiagonal, :]) < 0.0
    assert np.min(exact.directional_ldscores[:, :, basis.shape[1] :]) < 0.0


def test_fixed_probe_layers_and_streaming_are_tiling_invariant_across_jk_boundaries() -> None:
    genotype, basis, fixed_basis, annotations, probes, _phenotype = _fixture(
        3,
        2,
        seed=611,
        probe_count=37,
    )
    block_ids = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    _base, sources = pass1_sources(
        genotype,
        basis,
        fixed_basis,
        annotations,
        probes,
    )
    residual_rank = genotype.shape[0] - fixed_basis.shape[1]
    cross = pass2_cross_sketches(genotype, basis, sources, residual_rank)
    scores = randomized_directional_products(cross, pair_order(3))
    dense, dense_sources, _dense_cross = randomized_two_pass_ldscores(
        genotype,
        basis,
        fixed_basis,
        annotations,
        probes,
    )
    np.testing.assert_allclose(scores, dense.directional_ldscores, atol=0.0)
    for variant_width, probe_width in ((4, 5), (6, 11), (10, 37)):
        streamed, observed_sources, block_numerators, block_masses, ledger = (
            randomized_two_pass_ldscores_streaming(
                genotype,
                basis,
                fixed_basis,
                annotations,
                probes,
                block_ids,
                variant_block_width=variant_width,
                probe_chunk_width=probe_width,
            )
        )
        np.testing.assert_allclose(
            observed_sources,
            dense_sources,
            rtol=3.0e-14,
            atol=3.0e-14,
        )
        np.testing.assert_allclose(
            streamed.directional_ldscores,
            dense.directional_ldscores,
            rtol=8.0e-14,
            atol=8.0e-14,
        )
        np.testing.assert_allclose(
            np.sum(block_numerators, axis=0),
            dense.directed_numerator,
            rtol=8.0e-14,
            atol=8.0e-14,
        )
        np.testing.assert_allclose(
            np.sum(block_masses, axis=0),
            np.sum(annotations, axis=0),
            rtol=1.0e-15,
            atol=1.0e-15,
        )
        assert ledger == {
            "planned_reference_genotype_passes": 2,
            "observed_reference_genotype_passes": 2,
            "pass1_variant_visits": genotype.shape[1],
            "pass2_variant_visits": genotype.shape[1],
            "observed_retained_variant_visits": 2 * genotype.shape[1],
            "duplicate_retained_variant_visits": 0,
        }


def test_frozen_row_deletion_differs_from_literal_two_sided_deletion() -> None:
    genotype, basis, fixed_basis, annotations, _probes, _phenotype = _fixture(
        3,
        2,
        seed=701,
    )
    blocks = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    residual_rank = genotype.shape[0] - fixed_basis.shape[1]
    exact = exact_directional_ldscores(
        contextual_features(genotype, basis, fixed_basis),
        annotations,
        residual_rank,
    )
    frozen, block_numerators, _block_masses = frozen_ldscore_delete_block(
        exact,
        annotations,
        blocks,
    )
    recomputed = exact_recomputed_delete_block_grams(
        genotype,
        basis,
        fixed_basis,
        annotations,
        blocks,
    )
    assert np.max(np.abs(frozen - recomputed)) > 1.0e-6
    np.testing.assert_allclose(
        np.sum(block_numerators, axis=0),
        exact.directed_numerator,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_global_same_person_ustatistic_converges() -> None:
    genotype, basis, fixed_basis, annotations, probes, _phenotype = _fixture(
        3,
        2,
        seed=801,
        probe_count=40000,
    )
    randomized, sources, _cross = randomized_two_pass_ldscores(
        genotype,
        basis,
        fixed_basis,
        annotations,
        probes,
    )
    observed = same_person_ustatistic(sources, annotations, randomized.pairs)
    expected = exact_same_person_matrix(genotype, basis, fixed_basis, annotations)
    relative_error = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert relative_error < 0.035


def test_randomized_normal_gram_converges() -> None:
    genotype, basis, fixed_basis, annotations, probes, _phenotype = _fixture(
        3,
        2,
        seed=901,
        probe_count=40000,
    )
    randomized, _sources, _cross = randomized_two_pass_ldscores(
        genotype,
        basis,
        fixed_basis,
        annotations,
        probes,
    )
    expected = dense_kernel_gram(genotype, basis, fixed_basis, annotations)
    relative_error = np.linalg.norm(randomized.gram - expected) / np.linalg.norm(expected)
    assert relative_error < 0.025


def test_matched_feature_bridge_reproduces_mature_xw_directional_diagonals() -> None:
    genotype, basis, fixed_basis, annotations, probes, _phenotype = _fixture(
        2,
        1,
        seed=1001,
        probe_count=23,
    )
    features = contextual_features(genotype, basis, fixed_basis)
    generalized, sources, _cross = randomized_two_pass_ldscores(
        genotype,
        basis,
        fixed_basis,
        annotations,
        probes,
    )

    mature = GenomewideEnvLDScore.__new__(GenomewideEnvLDScore)
    mature.nbins = 1
    mature.df_corr = genotype.shape[0] - fixed_basis.shape[1]
    source_x = np.zeros((genotype.shape[0], probes.shape[1]))
    source_w = np.zeros_like(source_x)
    mature._accumulate_sketch_block(
        source_x,
        features[0],
        probes,
        annotations,
    )
    mature._accumulate_sketch_block(
        source_w,
        features[1],
        probes,
        annotations,
    )
    np.testing.assert_allclose(source_x, sources[0, 0], atol=2.0e-14)
    np.testing.assert_allclose(source_w, sources[0, 1], atol=2.0e-14)

    work_x = features[0].T @ np.column_stack([source_x, source_w])
    work_w = features[1].T @ np.column_stack([source_x, source_w])
    mature_panels = []
    work_panels = (
        work_x[:, : probes.shape[1]],
        work_x[:, probes.shape[1] :],
        work_w[:, : probes.shape[1]],
        work_w[:, probes.shape[1] :],
    )
    for work in work_panels:
        panel = np.zeros((genotype.shape[1], 1))
        mature._accumulate_left_scores(
            work,
            panel,
            0,
            genotype.shape[1],
            probes.shape[1],
        )
        mature_panels.append(panel[:, 0] / probes.shape[1])

    # Compare only XX, XW, WX, WW diagonal-pair directions.  The mature
    # constrained model does not contain the generalized off-diagonal kernel
    # coefficient, so no bridge assertion is made for that component.
    np.testing.assert_allclose(
        np.column_stack(mature_panels),
        np.column_stack(
            [
                generalized.directional_ldscores[:, 0, 0],
                generalized.directional_ldscores[:, 0, 1],
                generalized.directional_ldscores[:, 1, 0],
                generalized.directional_ldscores[:, 1, 1],
            ]
        ),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
