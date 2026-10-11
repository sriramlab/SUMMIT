"""Study same-person term of transported SUMMIT-GxE Grams.

The transported Gram keeps the different-person scaling and replaces the
legacy ``(n_S/n_R) D_R`` by the study's own same-person term: exactly from
per-person kernel diagonals on the study rows, or from aggregate context
fourth moments.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from summit.context.cross_trait_gram import ordered_diagonals
from summit.context.oracle import (
    SAME_PERSON_TRANSPORTS,
    SamePersonTransportWarning,
    approximate_study_same_person,
    context_fourth_moments,
    dense_genetic_kernels,
    exact_same_person_matrix,
    exact_study_same_person,
    rank_revealing_projector,
    resolve_study_same_person,
    transfer_reference_gram,
)
from summit.context.spec import ContextComponentIndex, ContextPairIndex


def _basis(rng: np.random.Generator, n: int) -> np.ndarray:
    sex = rng.integers(0, 2, size=n).astype(float)
    age = rng.normal(size=n)
    return np.column_stack([np.ones(n), sex - 0.5, age])


def _kernels(genotype, basis, annotations, names, projector=None):
    n = genotype.shape[0]
    p = np.eye(n) if projector is None else projector
    features = np.stack([p @ (basis[:, q, None] * genotype) for q in range(basis.shape[1])])
    components = ContextComponentIndex(names, ContextPairIndex(basis.shape[1]))
    return dense_genetic_kernels(features, annotations, components), components


def _selective_rows(rng: np.random.Generator, basis: np.ndarray) -> np.ndarray:
    # Missingness depends on both contexts: one sex and older people are kept
    # more often, so the study is not an exchangeable subset.
    keep = 0.25 + 0.5 * (basis[:, 1] > 0) + 0.15 * np.tanh(basis[:, 2])
    return np.flatnonzero(rng.random(basis.shape[0]) < keep)


def test_fourth_moments_use_pair_order_and_saved_kernel_factor() -> None:
    rng = np.random.default_rng(11)
    phi = _basis(rng, 40)
    pairs = ContextPairIndex(phi.shape[1]).entries
    explicit = np.zeros((len(pairs), len(pairs)))
    for i in range(phi.shape[0]):
        u = np.array([p.kernel_factor * phi[i, p.q] * phi[i, p.r] for p in pairs])
        explicit += np.outer(u, u)
    np.testing.assert_allclose(context_fourth_moments(phi), explicit, rtol=1e-13)
    rows = np.array([3, 1, 17, 30])
    np.testing.assert_array_equal(
        context_fourth_moments(phi, rows), context_fourth_moments(phi[rows])
    )
    assert context_fourth_moments(phi)[0, 0] == phi.shape[0]
    with pytest.raises(ValueError, match="repeat"):
        context_fourth_moments(phi, [1, 1])
    with pytest.raises(ValueError, match="sample indices"):
        context_fourth_moments(phi, [40])


def test_kappa_convention_matches_saved_dense_kernel_diagonals() -> None:
    rng = np.random.default_rng(12)
    n, m = 30, 50
    phi = _basis(rng, n)
    genotype = rng.normal(size=(n, m))
    annotations = np.column_stack([np.ones(m), rng.random(m)])
    kernels, components = _kernels(genotype, phi, annotations, ("all", "w"))
    diagonal = np.diagonal(kernels, axis1=1, axis2=2)
    masses = annotations.sum(axis=0)
    h = (genotype**2 @ annotations) / masses
    for component in components.entries:
        expected = (
            component.kernel_factor
            * phi[:, component.q]
            * phi[:, component.r]
            * h[:, component.annotation_index]
        )
        np.testing.assert_allclose(diagonal[component.index], expected, rtol=1e-12)
    # The research module halves saved off-diagonal values to ordered entries.
    ordered = ordered_diagonals(diagonal, phi.shape[1])
    q = phi.shape[1]
    np.testing.assert_allclose(
        ordered[1 * q + 2], phi[:, 1] * phi[:, 2] * h[:, 0], rtol=1e-12
    )


def test_constant_mass_genotypes_make_the_summary_form_exact() -> None:
    rng = np.random.default_rng(13)
    n, m = 60, 80
    phi = _basis(rng, n)
    genotype = rng.choice([-1.0, 1.0], size=(n, m))
    annotations = np.column_stack([np.ones(m), (rng.random(m) < 0.4).astype(float)])
    kernels, components = _kernels(genotype, phi, annotations, ("all", "part"))
    diagonal = np.diagonal(kernels, axis1=1, axis2=2)
    reference = exact_same_person_matrix(kernels)
    rows = _selective_rows(rng, phi)
    exact = exact_study_same_person(diagonal, rows)
    np.testing.assert_allclose(
        exact, exact_same_person_matrix(kernels[:, rows][:, :, rows]), rtol=1e-12
    )
    summary = approximate_study_same_person(
        reference,
        context_fourth_moments(phi),
        context_fourth_moments(phi, rows),
        reference_n=n,
        study_n=len(rows),
        annotation_count=2,
    )
    np.testing.assert_allclose(summary, exact, rtol=1e-11, atol=1e-11)
    scaled = len(rows) / n * reference
    assert np.linalg.norm(scaled - exact) > 0.03 * np.linalg.norm(exact)


def test_summary_form_beats_scaling_for_projected_selective_study() -> None:
    rng = np.random.default_rng(14)
    n, m = 240, 120
    phi = _basis(rng, n)
    frequencies = rng.uniform(0.1, 0.5, size=m)
    raw = rng.binomial(2, frequencies, size=(n, m)).astype(float)
    genotype = (raw - 2 * frequencies) / np.sqrt(2 * frequencies * (1 - frequencies))
    fixed = np.column_stack([np.ones(n), phi[:, 1:], rng.normal(size=n)])
    projector = rank_revealing_projector(fixed).projector
    kernels, _ = _kernels(genotype, phi, np.ones((m, 1)), ("all",), projector)
    diagonal = np.diagonal(kernels, axis1=1, axis2=2)
    reference = exact_same_person_matrix(kernels)
    rows = _selective_rows(rng, phi)
    exact = exact_study_same_person(diagonal, rows)
    summary = approximate_study_same_person(
        reference,
        context_fourth_moments(phi),
        context_fourth_moments(phi, rows),
        reference_n=n,
        study_n=len(rows),
        annotation_count=1,
    )
    scaled = len(rows) / n * reference
    summary_error = np.linalg.norm(summary - exact) / np.linalg.norm(exact)
    scaled_error = np.linalg.norm(scaled - exact) / np.linalg.norm(exact)
    assert scaled_error > 0.05
    assert summary_error < 0.25 * scaled_error


def test_study_equal_to_reference_reproduces_reference_bit_for_bit() -> None:
    rng = np.random.default_rng(15)
    n, m = 25, 30
    phi = _basis(rng, n)
    kernels, _ = _kernels(rng.normal(size=(n, m)), phi, np.ones((m, 1)), ("all",))
    diagonal = np.ascontiguousarray(np.diagonal(kernels, axis1=1, axis2=2))
    reference = diagonal @ diagonal.T
    reference = 0.5 * (reference + reference.T)
    everyone = np.arange(n)
    g = context_fourth_moments(phi)
    summary = approximate_study_same_person(
        reference,
        g,
        context_fourth_moments(phi, everyone),
        reference_n=n,
        study_n=n,
        annotation_count=1,
    )
    np.testing.assert_array_equal(summary, reference)
    np.testing.assert_array_equal(exact_study_same_person(diagonal, everyone), reference)
    gram = reference + rng.normal(size=reference.shape)
    gram = 0.5 * (gram + gram.T)
    np.testing.assert_array_equal(
        transfer_reference_gram(
            gram, reference, reference_n=n, study_n=n, study_same_person=summary
        ),
        gram,
    )


def test_transfer_with_scaled_study_term_reduces_to_legacy_result() -> None:
    rng = np.random.default_rng(16)
    gram = rng.normal(size=(6, 6))
    gram = gram @ gram.T
    same = rng.normal(size=(6, 6))
    same = 0.1 * (same @ same.T)
    for study_n in (7, 20, 31):
        legacy = transfer_reference_gram(gram, same, reference_n=20, study_n=study_n)
        replaced = transfer_reference_gram(
            gram,
            same,
            reference_n=20,
            study_n=study_n,
            study_same_person=(study_n / 20) * same,
        )
        np.testing.assert_array_equal(replaced, legacy)
    other = transfer_reference_gram(
        gram, same, reference_n=20, study_n=7, study_same_person=2.0 * same
    )
    expected = 2.0 * same + (7 * 6 / (20 * 19)) * (gram - same)
    np.testing.assert_allclose(other, 0.5 * (expected + expected.T), rtol=1e-14)
    with pytest.raises(ValueError, match="study_same_person"):
        transfer_reference_gram(
            gram, same, reference_n=20, study_n=7, study_same_person=np.eye(5)
        )


def test_resolver_priority_warning_and_forced_choices() -> None:
    rng = np.random.default_rng(17)
    n, m = 30, 40
    phi = _basis(rng, n)
    kernels, _ = _kernels(rng.normal(size=(n, m)), phi, np.ones((m, 1)), ("all",))
    diagonal = np.diagonal(kernels, axis1=1, axis2=2)
    same = exact_same_person_matrix(kernels)
    rows = np.arange(0, n, 2)
    g_r, g_s = context_fourth_moments(phi), context_fourth_moments(phi, rows)
    common = dict(reference_n=n, study_n=len(rows), annotation_count=1)
    value, label = resolve_study_same_person(
        same,
        component_kernel_diagonal=diagonal,
        study_rows=rows,
        reference_fourth_moments=g_r,
        study_fourth_moments=g_s,
        **common,
    )
    assert label == "own_rows_exact"
    np.testing.assert_array_equal(value, exact_study_same_person(diagonal, rows))
    value, label = resolve_study_same_person(
        same, reference_fourth_moments=g_r, study_fourth_moments=g_s, **common
    )
    assert label == "fourth_moment_difference"
    with pytest.warns(SamePersonTransportWarning):
        value, label = resolve_study_same_person(same, **common)
    assert (value, label) == (None, "scaled_legacy")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_study_same_person(same, requested="scaled_legacy", **common)[1] == (
            "scaled_legacy"
        )
        assert resolve_study_same_person(
            same, reference_n=n, study_n=n, annotation_count=1
        ) == (None, "scaled_legacy")
    _, label = resolve_study_same_person(
        same,
        component_kernel_diagonal=diagonal,
        study_rows=rows,
        reference_fourth_moments=g_r,
        study_fourth_moments=g_s,
        requested="fourth_moment_difference",
        **common,
    )
    assert label == "fourth_moment_difference"
    with pytest.raises(ValueError, match="own_rows_exact requires"):
        resolve_study_same_person(same, requested="own_rows_exact", **common)
    with pytest.raises(ValueError, match="study's people"):
        resolve_study_same_person(
            same, component_kernel_diagonal=diagonal, study_rows=rows[:-1], **common
        )
    with pytest.raises(ValueError, match="same_person_transport"):
        resolve_study_same_person(same, requested="exact", **common)
    assert SAME_PERSON_TRANSPORTS == (
        "own_rows_exact",
        "fourth_moment_difference",
        "scaled_legacy",
    )


def test_summary_form_validates_dimensions() -> None:
    same = np.eye(12)
    g = np.eye(6)
    approximate_study_same_person(
        same, g, g, reference_n=10, study_n=5, annotation_count=2
    )
    with pytest.raises(ValueError, match="annotation count"):
        approximate_study_same_person(
            same, g, g, reference_n=10, study_n=5, annotation_count=3
        )
    with pytest.raises(ValueError, match="equally square"):
        approximate_study_same_person(
            same, g, np.eye(5), reference_n=10, study_n=5, annotation_count=2
        )
    with pytest.raises(ValueError, match="intercept"):
        approximate_study_same_person(
            same, 0 * g, g, reference_n=10, study_n=5, annotation_count=2
        )


def _chromosome_design():
    from dataclasses import replace

    from summit.ldscore.generalized_gxe_chromosome import ChromosomeMoments

    rng = np.random.default_rng(18)
    n, q, k = 40, 3, 2
    p = q * (q + 1) // 2
    c = k * p
    phi = _basis(rng, n)
    rows = _selective_rows(rng, phi)
    ns = len(rows)
    reference, study = [], []
    for chromosome in (1, 2):
        directed = rng.normal(size=(2, c, c))
        directed = np.einsum("bij,bkj->bik", directed, directed)
        chunk = ChromosomeMoments(
            str(chromosome), "master", ("a", "b"), ("trait",), q, n, n - 2,
            np.array([2 * chromosome - 2, 2 * chromosome - 1]),
            np.array([[9.0, 4.0], [11.0, 6.0]]), directed,
            rng.normal(size=(2, c, 1)), rng.normal(size=(2, c, 2)),
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
    pairs = ContextPairIndex(q).entries
    unit = np.stack(
        [p_.kernel_factor * phi[:, p_.q] * phi[:, p_.r] for p_ in pairs]
    )
    mass = rng.uniform(0.95, 1.05, size=(k, n))
    diagonal = np.vstack([unit * mass[a] for a in range(k)])
    options = dict(
        reference_residual_gram=np.diag([n - 2.0, float(n)]),
        reference_residual_traces=np.array([n - 2.0, 0.0]),
        reference_same_person=diagonal @ diagonal.T,
        residual_gram=np.diag([ns - 2.0, float(ns)]),
        residual_rhs=np.ones((2, 1)),
        residual_traces=np.array([ns - 2.0, 0.0]),
        residual_names=("one", "context"),
    )
    return reference, study, phi, rows, diagonal, options


def test_chromosome_transfer_uses_the_study_same_person_term() -> None:
    from summit.ldscore.generalized_gxe_chromosome import (
        joint_chromosome_equations,
        transferred_chromosome_equations,
    )

    reference, study, phi, rows, diagonal, options = _chromosome_design()
    same = options["reference_same_person"]
    n, ns = reference[0].n_samples, len(rows)
    exact = exact_study_same_person(diagonal, rows)
    summary = approximate_study_same_person(
        same,
        context_fourth_moments(phi),
        context_fourth_moments(phi, rows),
        reference_n=n,
        study_n=ns,
        annotation_count=2,
    )
    for deleted in ((), (1,), (0, 3)):
        joint = joint_chromosome_equations(
            reference,
            residual_gram=options["reference_residual_gram"],
            residual_traces=options["reference_residual_traces"],
            residual_rhs=np.zeros((2, 1)),
            residual_names=options["residual_names"],
            deleted_blocks=deleted,
        )
        own = transferred_chromosome_equations(
            reference, study, deleted_blocks=deleted,
            reference_kernel_diagonal=diagonal, study_rows=rows, **options,
        )
        assert own.same_person_transport == "own_rows_exact"
        # Deletions reuse the full study term, as they reuse the full D_R.
        np.testing.assert_array_equal(
            own.transferred_genetic_gram,
            transfer_reference_gram(
                joint.reference_genetic_gram, same, reference_n=n, study_n=ns,
                study_same_person=exact,
            ),
        )
        explicit = transferred_chromosome_equations(
            reference, study, deleted_blocks=deleted, study_same_person=exact,
            same_person_transport="own_rows_exact", **options,
        )
        np.testing.assert_array_equal(explicit.matrix, own.matrix)
        moments = transferred_chromosome_equations(
            reference, study, deleted_blocks=deleted,
            reference_fourth_moments=context_fourth_moments(phi),
            study_fourth_moments=context_fourth_moments(phi, rows), **options,
        )
        assert moments.same_person_transport == "fourth_moment_difference"
        np.testing.assert_array_equal(
            moments.transferred_genetic_gram,
            transfer_reference_gram(
                joint.reference_genetic_gram, same, reference_n=n, study_n=ns,
                study_same_person=summary,
            ),
        )
        with pytest.warns(SamePersonTransportWarning):
            legacy = transferred_chromosome_equations(
                reference, study, deleted_blocks=deleted, **options
            )
        assert legacy.same_person_transport == "scaled_legacy"
        np.testing.assert_array_equal(
            legacy.transferred_genetic_gram,
            transfer_reference_gram(
                joint.reference_genetic_gram, same, reference_n=n, study_n=ns
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            acknowledged = transferred_chromosome_equations(
                reference, study, deleted_blocks=deleted,
                same_person_transport="scaled_legacy", **options,
            )
        np.testing.assert_array_equal(acknowledged.matrix, legacy.matrix)
    # Per-person masses vary, so the summary form is close but not exact.
    assert np.linalg.norm(summary - exact) < 0.2 * np.linalg.norm(ns / n * same - exact)
    with pytest.raises(ValueError, match="label"):
        transferred_chromosome_equations(
            reference, study, study_same_person=exact, **options
        )
    with pytest.raises(ValueError, match="C by C"):
        transferred_chromosome_equations(
            reference, study, study_same_person=exact[:-1, :-1],
            same_person_transport="own_rows_exact", **options,
        )
