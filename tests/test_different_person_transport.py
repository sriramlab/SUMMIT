"""Composition-corrected different-person term of transported SUMMIT-GxE Grams.

``T_S = D_S + s (T_R - D_R) + kron(L, F_S - s F_R)``: the scaled reference
different-person part plus the study's context-composition correction, as in
cross_trait_gram's ``factorized_plus_residual`` mode.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from summit.context.cross_trait_gram import (
    chromosome_gram,
    exposure_factor,
    saved,
    within_trait_equations,
)
from summit.context.oracle import (
    DifferentPersonTransportWarning,
    context_fourth_moments,
    context_second_moments,
    different_person_correction,
    distinct_pair_exposure,
    exact_study_same_person,
    intercept_ld_scalars,
    raw_same_person_intercept,
    resolve_different_person_exposure,
    transfer_reference_gram,
)
from summit.context.spec import ContextPairIndex
from summit.ldscore.generalized_gxe_chromosome import (
    ChromosomeMoments,
    chromosome_same_person_intercept,
    joint_chromosome_equations,
    load_chromosome_moments,
    own_row_transported_equations,
    prepare_chromosome_transport,
    transferred_chromosome_equations,
    write_chromosome_moments,
)


def _basis(rng, n):
    sex = rng.integers(0, 2, size=n).astype(float)
    age = rng.normal(size=n)
    return np.column_stack([np.ones(n), sex - 0.5, age])


def _selective_rows(rng, phi):
    keep = 0.25 + 0.5 * (phi[:, 1] > 0) + 0.15 * np.tanh(phi[:, 2])
    return np.flatnonzero(rng.random(phi.shape[0]) < keep)


def _design(seed=31, n=48, k=2):
    rng = np.random.default_rng(seed)
    q = 3
    p = q * (q + 1) // 2
    c = k * p
    phi = _basis(rng, n)
    rows = _selective_rows(rng, phi)
    ns = len(rows)
    pairs = ContextPairIndex(q).entries
    unit = np.stack([e.kernel_factor * phi[:, e.q] * phi[:, e.r] for e in pairs])
    reference, study, diagonals = [], [], {}
    block = 0
    for chromosome, blocks in ((1, 3), (2, 2)):
        directed = rng.normal(size=(blocks, c, c))
        directed = np.einsum("bij,bkj->bik", directed, directed) + 0.3 * rng.normal(
            size=(blocks, c, c)
        )
        masses = rng.uniform(2.0, 6.0, size=(blocks, k))
        ids = np.arange(block, block + blocks)
        block += blocks
        chunk = ChromosomeMoments(
            str(chromosome), "master", tuple(f"a{j}" for j in range(k)), ("trait",),
            q, n, n - 2, ids, masses, directed,
            rng.normal(size=(blocks, c, 1)), rng.normal(size=(blocks, c, 2)),
        )
        reference.append(chunk)
        study.append(
            replace(
                chunk,
                cohort_identity="study",
                n_samples=ns,
                residual_rank=ns - 2,
                block_genetic_rhs=chunk.block_genetic_rhs * 0.8,
                block_genetic_residual=chunk.block_genetic_residual * 0.8,
            )
        )
        mass = rng.uniform(0.02, 0.05, size=(k, n))
        diagonals[str(chromosome)] = np.vstack([unit * mass[a] for a in range(k)])
    moments = dict(
        reference_residual_gram=np.diag([n - 2.0, float(n)]),
        reference_residual_traces=np.array([n - 2.0, 0.0]),
        residual_gram=np.diag([ns - 2.0, float(ns)]),
        residual_rhs=np.ones((2, 1)),
        residual_traces=np.array([ns - 2.0, 0.0]),
        residual_names=("one", "context"),
    )
    return reference, study, diagonals, phi, rows, moments


def test_distinct_pair_exposure_matches_saved_exposure_factor() -> None:
    rng = np.random.default_rng(1)
    phi = _basis(rng, 37)
    rows = np.sort(rng.choice(37, size=21, replace=False))
    for selection in (None, rows):
        expected = saved(exposure_factor(phi, selection, selection), 3)
        observed = distinct_pair_exposure(
            context_second_moments(phi, selection), context_fourth_moments(phi, selection)
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-10)


def test_ld_scalars_match_chromosome_gram() -> None:
    reference, _, diagonals, phi, rows, _ = _design()
    masses = sum(x.block_masses.sum(axis=0) for x in reference)
    for chunk in reference:
        diagonal = diagonals[chunk.chromosome]
        gram = chromosome_gram(
            chunk, diagonal, phi, rows, global_masses=masses,
            mode="factorized_plus_residual", ordered=False,
        )
        intercept = chromosome_same_person_intercept(diagonal, masses)
        np.testing.assert_allclose(
            intercept,
            raw_same_person_intercept(diagonal @ diagonal.T, masses, pair_count=6),
            rtol=1e-12,
        )
        scalars = chunk.ld_scalars(intercept) / np.outer(masses, masses)
        np.testing.assert_allclose(
            scalars, gram.diagnostics["ld_scalar"], rtol=1e-11, atol=1e-14
        )


def _transport_gap(reference, transport, moments, study_n):
    """s [D_R - sum_c D_c + sum_c B_c C^-1 B_c' - B C^-1 B']: the known
    difference between the published joint assembly and the research
    per-chromosome assembly of the different-person part."""
    masses = sum(x.block_masses.sum(axis=0) for x in reference)
    p = reference[0].num_basis * (reference[0].num_basis + 1) // 2
    inverse = np.repeat(1 / masses, p)
    gram = moments["reference_residual_gram"]
    per_chromosome = [x.block_genetic_residual.sum(axis=0) * inverse[:, None] for x in reference]
    total = sum(per_chromosome)
    profile = sum(b @ np.linalg.solve(gram, b.T) for b in per_chromosome)
    profile = 0.5 * (profile + profile.T) - total @ np.linalg.solve(gram, total.T)
    n = reference[0].n_samples
    s = study_n * (study_n - 1) / (n * (n - 1))
    return transport.reference_same_person, s, profile


def test_own_row_composition_transport_matches_research_assembly() -> None:
    reference, study, diagonals, phi, rows, moments = _design()
    transport = prepare_chromosome_transport(reference, diagonals, phi)
    equations = own_row_transported_equations(
        reference, study, transport=transport, study_rows=rows, **moments
    )
    assert equations.same_person_transport == "own_rows_exact"
    assert equations.different_person_transport == "composition_corrected"
    research = within_trait_equations(
        reference, study, reference_diagonals=diagonals, phi=phi, rows=rows,
        mode="factorized_plus_residual", same_person_mode="own_rows",
        reference_same_person=transport.reference_same_person,
        deletion_method="legacy", **moments,
    )
    same, s, profile = _transport_gap(reference, transport, moments, len(rows))
    chromosome_same = sum(d @ d.T for d in diagonals.values())
    gap = s * (same - chromosome_same + profile)
    c = equations.genetic_count
    np.testing.assert_allclose(
        equations.matrix[:c, :c] + gap, research.matrix[:c, :c], rtol=1e-11, atol=1e-11
    )
    np.testing.assert_array_equal(equations.matrix[:c, c:], research.matrix[:c, c:])
    np.testing.assert_array_equal(equations.rhs, research.rhs)
    # The correction is what separates the two assemblies' legacy arms too.
    legacy_research = within_trait_equations(
        reference, study, reference_diagonals=diagonals, phi=phi, rows=rows,
        mode="legacy_transport", same_person_mode="own_rows",
        reference_same_person=transport.reference_same_person,
        deletion_method="legacy", **moments,
    )
    legacy = transferred_chromosome_equations(
        reference, study, **transport.for_study(rows),
        different_person_transport="scaled_legacy", **moments,
    )
    assert legacy.different_person_transport == "scaled_legacy"
    np.testing.assert_allclose(
        equations.matrix[:c, :c] - legacy.matrix[:c, :c],
        research.matrix[:c, :c] - legacy_research.matrix[:c, :c],
        rtol=1e-10, atol=1e-11,
    )


@pytest.mark.parametrize("deleted", [(1,), (0, 4), (2, 3)])
def test_deletions_drop_blocks_and_use_retained_masses(deleted) -> None:
    reference, study, diagonals, phi, rows, moments = _design(seed=33)
    transport = prepare_chromosome_transport(reference, diagonals, phi)
    options = transport.for_study(rows)
    corrected = transferred_chromosome_equations(
        reference, study, deleted_blocks=deleted, **options, **moments
    )
    legacy = transferred_chromosome_equations(
        reference, study, deleted_blocks=deleted, **options,
        different_person_transport="scaled_legacy", **moments,
    )
    # Convention: drop the deleted blocks' raw LD scalars and divide their sum
    # by the retained annotation masses, which normalize the deleted T_R.
    retained = sum(x.block_masses[~np.isin(x.block_ids, deleted)].sum(axis=0) for x in reference)
    scalars = sum(
        x.ld_scalars(transport.same_person_intercepts[x.chromosome])[
            ~np.isin(x.block_ids, deleted)
        ].sum(axis=0)
        for x in reference
    ) / np.outer(retained, retained)
    n, ns = reference[0].n_samples, len(rows)
    s = ns * (ns - 1) / (n * (n - 1))
    exposure = distinct_pair_exposure(
        options["study_second_moments"], options["study_fourth_moments"]
    ) - s * distinct_pair_exposure(
        options["reference_second_moments"], options["reference_fourth_moments"]
    )
    expected = np.kron(scalars, exposure)
    c = corrected.genetic_count
    np.testing.assert_allclose(
        corrected.matrix[:c, :c] - legacy.matrix[:c, :c],
        0.5 * (expected + expected.T), rtol=1e-11, atol=1e-12,
    )
    # The research path's legacy deletions restore retained masses the same way.
    research = {
        mode: within_trait_equations(
            reference, study, reference_diagonals=diagonals, phi=phi, rows=rows,
            mode=mode, same_person_mode="own_rows", deleted_blocks=deleted,
            reference_same_person=transport.reference_same_person,
            deletion_method="legacy", **moments,
        )
        for mode in ("factorized_plus_residual", "legacy_transport")
    }
    np.testing.assert_allclose(
        corrected.matrix[:c, :c] - legacy.matrix[:c, :c],
        research["factorized_plus_residual"].matrix[:c, :c]
        - research["legacy_transport"].matrix[:c, :c],
        rtol=1e-10, atol=1e-11,
    )


def test_study_equal_to_reference_reproduces_published_transport_exactly() -> None:
    reference, _, diagonals, phi, _, moments = _design(seed=35)
    n = reference[0].n_samples
    transport = prepare_chromosome_transport(reference, diagonals, phi)
    own = dict(moments, residual_gram=moments["reference_residual_gram"],
               residual_traces=moments["reference_residual_traces"])
    for deleted in ((), (2,)):
        published = transferred_chromosome_equations(
            reference, reference, reference_same_person=transport.reference_same_person,
            deleted_blocks=deleted, **own,
        )
        observed = own_row_transported_equations(
            reference, reference, transport=transport, study_rows=np.arange(n),
            deleted_blocks=deleted, **own,
        )
        assert observed.different_person_transport == "composition_corrected"
        np.testing.assert_array_equal(observed.matrix, published.matrix)


def test_chromosome_moments_store_raw_intercepts_and_reload(tmp_path) -> None:
    reference, study, diagonals, phi, rows, moments = _design(seed=37)
    masses = sum(x.block_masses.sum(axis=0) for x in reference)
    stored = [
        replace(x, same_person_intercept=chromosome_same_person_intercept(diagonals[x.chromosome], masses))
        for x in reference
    ]
    for chunk in stored:
        path = tmp_path / f"chr{chunk.chromosome}.npz"
        write_chromosome_moments(path, chunk, provenance={"test": True})
        loaded, _ = load_chromosome_moments(path)
        np.testing.assert_array_equal(loaded.same_person_intercept, chunk.same_person_intercept)
    plain = tmp_path / "plain.npz"
    write_chromosome_moments(plain, reference[0], provenance={"test": True})
    with np.load(plain, allow_pickle=False) as archive:
        assert "same_person_intercept" not in archive.files
    assert load_chromosome_moments(plain)[0].same_person_intercept is None
    transport = prepare_chromosome_transport(reference, diagonals, phi)
    options = transport.for_study(rows)
    via_diagonals = transferred_chromosome_equations(reference, study, **options, **moments)
    del options["reference_same_person_intercepts"]
    via_stored = transferred_chromosome_equations(stored, study, **options, **moments)
    np.testing.assert_allclose(via_stored.matrix, via_diagonals.matrix, rtol=1e-13, atol=1e-13)
    with pytest.warns(DifferentPersonTransportWarning):
        missing = transferred_chromosome_equations(reference, study, **options, **moments)
    assert missing.different_person_transport == "scaled_legacy"


def test_resolver_policy_and_zero_correction() -> None:
    rng = np.random.default_rng(39)
    phi = _basis(rng, 30)
    rows = np.arange(0, 30, 2)
    m_r, g_r = context_second_moments(phi), context_fourth_moments(phi)
    m_s, g_s = context_second_moments(phi, rows), context_fourth_moments(phi, rows)
    exposure, label = resolve_different_person_exposure(
        reference_n=30, study_n=15, reference_second_moments=m_r,
        reference_fourth_moments=g_r, study_second_moments=m_s, study_fourth_moments=g_s,
    )
    assert label == "composition_corrected"
    s = 15 * 14 / (30 * 29)
    np.testing.assert_allclose(
        exposure, distinct_pair_exposure(m_s, g_s) - s * distinct_pair_exposure(m_r, g_r)
    )
    same, label = resolve_different_person_exposure(
        reference_n=30, study_n=30, reference_second_moments=m_r,
        reference_fourth_moments=g_r, study_second_moments=m_r, study_fourth_moments=g_r,
    )
    assert label == "composition_corrected" and not np.any(same)
    with pytest.warns(DifferentPersonTransportWarning):
        assert resolve_different_person_exposure(reference_n=30, study_n=15) == (
            None, "scaled_legacy")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_different_person_exposure(
            reference_n=30, study_n=15, requested="scaled_legacy"
        ) == (None, "scaled_legacy")
        assert resolve_different_person_exposure(reference_n=30, study_n=30) == (
            None, "scaled_legacy")
    with pytest.raises(ValueError, match="requires reference LD scalars"):
        resolve_different_person_exposure(
            reference_n=30, study_n=15, reference_second_moments=m_r,
            reference_fourth_moments=g_r, study_second_moments=m_s,
            study_fourth_moments=g_s, ld_scalars_available=False,
            requested="composition_corrected",
        )
    gram = rng.normal(size=(6, 6))
    gram = gram @ gram.T
    d = 0.1 * np.eye(6)
    zero = different_person_correction(
        np.ones((3, 1, 1)), np.zeros((6, 6)), masses=np.ones(1)
    )
    np.testing.assert_array_equal(
        transfer_reference_gram(gram, d, reference_n=30, study_n=30,
                                different_person_correction=zero),
        gram,
    )
    np.testing.assert_array_equal(
        transfer_reference_gram(gram, d, reference_n=30, study_n=15,
                                different_person_correction=zero),
        transfer_reference_gram(gram, d, reference_n=30, study_n=15),
    )
    shift = 0.01 * np.ones((6, 6))
    np.testing.assert_allclose(
        transfer_reference_gram(gram, d, reference_n=30, study_n=15,
                                different_person_correction=shift)
        - transfer_reference_gram(gram, d, reference_n=30, study_n=15),
        shift, rtol=1e-12, atol=1e-14,
    )


def test_intercept_ld_scalars_validate_axes() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        intercept_ld_scalars(np.zeros((2, 6, 6)), np.ones((2, 2)), np.eye(2),
                             reference_n=10, pair_count=6)
    out = intercept_ld_scalars(np.ones((2, 6, 6)), np.ones((2, 1)), np.eye(1),
                               reference_n=10, pair_count=6)
    np.testing.assert_allclose(out, np.full((2, 1, 1), 0.5 / 90))


def test_selective_study_gram_and_fit_approach_its_own_with_the_correction() -> None:
    import test_same_person_transport_artifacts as artifacts
    from generalized_gxe_variant_ldscore_oracle import orthonormalize
    from summit.context.oracle import dense_genetic_kernels, kernel_gram, rank_revealing_projector
    from summit.context.spec import ContextComponentIndex
    from summit.ldscore.generalized_gxe_fit_v1 import fit_generalized_gxe_variant_model_v1
    from summit.ldscore.generalized_gxe_variant import GlobalVariantProbeSpec

    gram_errors, deviations = [], []
    for seed in (3, 4, 5, 6):
        _, basis, genotype, fixed_design, rows, phenotype = artifacts._selective_cohort(seed)
        m = genotype.shape[1]
        annotations, names = np.ones((m, 1)), ("all",)
        blocks = np.arange(m, dtype=np.int64) % 10
        labels = tuple(f"b{index}" for index in range(10))
        reference = artifacts._reference(
            genotype, basis, orthonormalize(fixed_design), annotations, names, blocks,
            labels, GlobalVariantProbeSpec(seed, 0, 64), "composable",
        )
        trait = artifacts._trait(
            genotype[rows], basis[rows], fixed_design[rows], phenotype, annotations,
            names, blocks, labels,
        )
        # Known answer: the study's own exact Gram (own people and projector).
        projector = rank_revealing_projector(fixed_design[rows]).projector
        features = np.stack([projector @ (basis[rows, q, None] * genotype[rows]) for q in range(3)])
        own = kernel_gram(dense_genetic_kernels(
            features, annotations, ContextComponentIndex(names, ContextPairIndex(3))))
        fits = [
            fit_generalized_gxe_variant_model_v1(
                reference, trait, study_rows=rows, different_person_transport=label
            )
            for label in ("scaled_legacy", None)
        ]
        assert [fit.manifest["different_person_transport"] for fit in fits] == [
            "scaled_legacy", "composition_corrected"]
        c = len(reference.component_index)
        matrix = np.array(fits[1].normal_matrix)
        matrix[:c, :c] = own
        target = np.linalg.solve(matrix, fits[1].normal_rhs)[:c]
        se = fits[1].raw_standard_errors[:c]
        gram_errors.append([
            np.linalg.norm(fit.transferred_genetic_gram - own) / np.linalg.norm(own) for fit in fits])
        deviations.append([
            np.max(np.abs(fit.raw_coefficients[:c] - target) / se) for fit in fits])
    gram_errors, deviations = np.asarray(gram_errors), np.asarray(deviations)
    # With the exact own-row same-person term in both arms, the remaining
    # transport error falls on every draw once the different-person term is
    # corrected for composition; the rest is probe noise and the projector.
    assert np.all(gram_errors[:, 1] < 0.6 * gram_errors[:, 0])
    assert deviations[:, 1].mean() < 0.5 * deviations[:, 0].mean()


def test_main_era_artifacts_keep_their_fits_and_lack_only_second_moments() -> None:
    """Artifacts written by SUMMIT 4757960 carry fourth but not second moments."""
    import json
    from pathlib import Path

    from summit.context import fit_context_model, load_context_reference, load_context_trait_summary
    from summit.ldscore.generalized_gxe_fit_v1 import fit_generalized_gxe_variant_model_v1
    from summit.ldscore.generalized_gxe_reference_v1 import load_generalized_gxe_variant_reference_v1
    from summit.ldscore.generalized_gxe_trait_summary import load_generalized_gxe_trait_summary

    root = Path(__file__).parent / "fixtures" / "different_person_transport"
    expected = json.loads((root / "main_expected.json").read_text())
    reference = load_generalized_gxe_variant_reference_v1(
        root / "main_reference.generalized-gxe-variant-ldscore-v1.npz")
    trait = load_generalized_gxe_trait_summary(root / "main_trait.generalized-gxe-trait-summary-v1.npz")
    assert reference.context_fourth_moments is not None and reference.context_second_moments is None
    assert trait.context_fourth_moments is not None and trait.context_second_moments is None
    with pytest.warns(DifferentPersonTransportWarning):
        fit = fit_generalized_gxe_variant_model_v1(reference, trait)
    assert fit.manifest["same_person_transport"] == expected["generalized_same_person_transport"]
    assert fit.manifest["different_person_transport"] == "scaled_legacy"
    np.testing.assert_allclose(fit.raw_coefficients, expected["generalized_raw_coefficients"], rtol=1e-12)
    np.testing.assert_allclose(
        fit.raw_standard_errors, expected["generalized_raw_standard_errors"], rtol=1e-12)
    context_reference = load_context_reference(root / "main_context.context-reference.json")
    context_summary = load_context_trait_summary(root / "main_context.context-trait.json")
    assert context_reference.context_second_moments is None
    with pytest.warns(DifferentPersonTransportWarning):
        context_fit = fit_context_model(context_reference, context_summary)
    assert context_fit.manifest["assembly"]["same_person_transport"] == (
        expected["context_same_person_transport"])
    assert context_fit.manifest["assembly"]["different_person_transport"] == "scaled_legacy"
    np.testing.assert_allclose(context_fit.raw_coefficients, expected["context_raw_coefficients"], rtol=1e-12)
    np.testing.assert_allclose(context_fit.standard_errors, expected["context_standard_errors"], rtol=1e-12)
